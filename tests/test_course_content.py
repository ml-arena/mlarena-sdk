"""SDK tests for the course-content methods.

These mock the REST surface by replacing the client's single HTTP chokepoint
(``MLArenaClient._request``) with a recorder that captures every call and
returns canned responses — no network, no backend, no extra dependencies
(stdlib + ``requests`` only). Each test asserts the method → (verb, path, body,
params) mapping that the parity rule requires.

Runs under pytest *or* directly: ``python tests/test_course_content.py``.
"""
import io
import json
import os
import sys
import tempfile
import warnings
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena.client import MLArenaClient
from mlarena.exceptions import AuthenticationError, MLArenaError


# --------------------------------------------------------------------------- #
# Test harness: a fake transport recording calls + returning routed responses
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content=b"",
                 headers=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        yield self.content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status {self.status_code}")


class Recorder:
    """Drop-in replacement for ``client._request``.

    ``router(method, path, kwargs)`` may return a ``FakeResponse``, a
    ``(status, json)`` tuple, or ``None`` to fall through to a generic 200.
    """

    def __init__(self, router=None):
        self.calls = []
        self.router = router

    def __call__(self, method, url, **kwargs):
        path = url.split("/api", 1)[1]
        self.calls.append({"method": method, "path": path, "url": url, **kwargs})
        if self.router is not None:
            result = self.router(method, path, kwargs)
            if result is not None:
                if isinstance(result, FakeResponse):
                    return result
                status, data = result
                return FakeResponse(status, data)
        return FakeResponse(200, {"ok": True})

    @property
    def last(self):
        return self.calls[-1]


def make_client(router=None, scope="author"):
    client = mlarena.connect(
        api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
        base_url="https://example.com",
    )
    rec = Recorder(router)
    client._request = rec  # shadow the instance method with the recorder
    return client, rec


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# Authoring — modules
# --------------------------------------------------------------------------- #


def test_the_module_library_methods_are_gone():
    """4.23.0: a module is a section of one course. The library routes
    answer 410, and their methods are removed in the same minor."""
    c, rec = make_client()
    for name in ("create_module", "list_modules", "fork_module", "link_module",
                 "unlink_module", "attach_challenge", "detach_challenge",
                 "update_challenge_link", "remove_course_challenge",
                 "add_course_exercise_set", "challenges_for_course",
                 "copyable_challenges", "challenge_courses",
                 "use_challenge_with_class"):
        _expect(not hasattr(c, name), name)
    _expect(rec.calls == [], rec.calls)


def test_create_course_module_is_the_module_create():
    c, rec = make_client(lambda *_: (201, {"module_id": 4}))
    c.create_course_module(7, "NLP", summary="intro")
    _expect((rec.last["method"], rec.last["path"]) ==
            ("POST", "/teacher/course/7/modules/create"), rec.last)
    _expect(rec.last["json"] == {"title": "NLP", "summary": "intro"}, rec.last["json"])
    auth = rec.last["headers"]["Authorization"]
    _expect(auth.startswith("Bearer mlk_author_"), auth)


def test_update_module_partial_and_unknown_field():
    c, rec = make_client(lambda *_: (200, {"id": 2}))
    c.update_module(2, is_published=False)
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/2", rec.last["path"])
    _expect(rec.last["json"] == {"is_published": False}, rec.last["json"])

    # A module has no visibility of its own any more: refused before any call.
    for bad in ({"bogus": "x"}, {"visibility": "public"}):
        try:
            c.update_module(2, **bad)
            raise AssertionError(f"expected MLArenaError on {bad}")
        except MLArenaError:
            pass

    try:
        c.update_module(2)
        raise AssertionError("expected MLArenaError on empty update")
    except MLArenaError:
        pass


def test_delete_module_takes_no_force():
    c, rec = make_client(lambda *_: (200, {"message": "ok"}))
    c.delete_module(3)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/3", rec.last["path"])
    _expect(rec.last["params"] is None, rec.last["params"])
    try:
        c.delete_module(3, force=True)
    except TypeError:
        pass
    else:
        raise AssertionError("force= is gone")


def test_a_module_that_holds_a_challenge_is_not_deleted():
    c, _ = make_client(lambda *_: (409, {
        "error": "This module holds 1 challenge(s): delete them or move them to "
                 "another module first", "challenge_count": 1}))
    try:
        c.delete_module(3)
    except MLArenaError as err:
        _expect(err.status_code == 409 and err.body["challenge_count"] == 1, err.body)
        _expect("move them to another module" in str(err), str(err))
    else:
        raise AssertionError("a 409 must raise")


def test_update_course_challenge_sets_the_bar_and_moves_it():
    c, rec = make_client(lambda *_: (200, {"challenge_id": 42}))
    c.update_course_challenge(2, 42, pass_threshold=None)
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/42", rec.last["path"])
    _expect(rec.last["json"] == {"pass_threshold": None}, rec.last["json"])

    # `module_id` is the body key (the module it moves to); the path's module
    # is the one it is in now.
    c.update_course_challenge(2, 42, module_id=5, pass_threshold=70.0)
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/42", rec.last["path"])
    _expect(rec.last["json"] == {"module_id": 5, "pass_threshold": 70.0}, rec.last["json"])

    for bad in ({}, {"label": "x"}):
        try:
            c.update_course_challenge(2, 42, **bad)
            raise AssertionError(f"expected MLArenaError on {bad}")
        except MLArenaError:
            pass


def test_the_frozen_competition_aliases():
    """`update_competition_link` points at the surviving update route;
    `attach_competition` / `detach_competition` raise the sentence."""
    c, rec = make_client(lambda *_: (200, {"challenge_id": 42}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.update_competition_link(2, 42, pass_threshold=50.0)
    _expect(any(issubclass(w.category, DeprecationWarning) for w in caught), caught)
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/42", rec.last["path"])
    _expect(rec.last["json"] == {"pass_threshold": 50.0}, rec.last["json"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        c.reorder_module_competitions(2, [3, 1])
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/reorder", rec.last["path"])

    calls = len(rec.calls)
    for name in ("attach_competition", "detach_competition"):
        try:
            getattr(c, name)(2, 42)
        except MLArenaError as err:
            _expect("a challenge now belongs to its course: create it with "
                    "`add_course_challenge`" in str(err), str(err))
        else:
            raise AssertionError(f"{name} must raise")
    _expect(len(rec.calls) == calls, "no request")


def test_reorder_module_challenges():
    c, rec = make_client(lambda *_: (200, []))
    c.reorder_module_challenges(2, [3, 1])
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/reorder",
            rec.last["path"])
    _expect(rec.last["json"] == {"ordered_ids": [3, 1]}, rec.last["json"])


# --------------------------------------------------------------------------- #
# Authoring — lessons
# --------------------------------------------------------------------------- #


def test_create_lesson_body():
    c, rec = make_client(lambda *_: (201, {"id": 10}))
    c.create_lesson(2, "What is RL?", body_md="# Hi", gated=True)
    _expect(rec.last["path"] == "/teacher/modules/2/lessons", rec.last["path"])
    _expect(rec.last["json"] == {"title": "What is RL?", "kind": "lesson",
                                 "body_md": "# Hi", "gated": True}, rec.last["json"])


def test_update_lesson_partial():
    c, rec = make_client(lambda *_: (200, {"id": 5}))
    c.update_lesson(5, is_published=True, estimated_minutes=12)
    _expect(rec.last["path"] == "/teacher/lessons/5", rec.last["path"])
    _expect(rec.last["json"] == {"is_published": True, "estimated_minutes": 12},
            rec.last["json"])


def test_reorder_lessons():
    c, rec = make_client(lambda *_: (200, []))
    c.reorder_lessons(2, [3, 1, 2])
    _expect(rec.last["path"] == "/teacher/modules/2/lessons/reorder", rec.last["path"])
    _expect(rec.last["json"] == {"ordered_ids": [3, 1, 2]}, rec.last["json"])


def test_preview_lesson_strict():
    c, rec = make_client(lambda *_: (200, {"directives": [], "warnings": []}))
    c.preview_lesson(7, body_md="```mlarena:leaderboard id=1```")
    _expect(rec.last["path"] == "/teacher/lessons/7/preview", rec.last["path"])
    _expect("body_md" in rec.last["json"], rec.last["json"])

    c.preview_lesson(7)
    _expect(rec.last["json"] == {}, rec.last["json"])


def test_preview_lesson_error_says_which_directive_and_why():
    # The 400's `error` is only "Invalid directive"; the reason is beside it.
    reason = "mlarena:leaderboard: challenge 999 does not exist"
    c, _rec = make_client(lambda *_: (
        400, {"error": "Invalid directive", "directive_error": reason}))
    try:
        c.preview_lesson(7, body_md="```mlarena:leaderboard id=999```")
    except MLArenaError as err:
        _expect(str(err) == f"preview_lesson failed: Invalid directive: {reason}", str(err))
        _expect(err.status_code == 400, err.status_code)
        _expect(err.body["directive_error"] == reason, err.body)
    else:
        raise AssertionError("preview_lesson did not raise")


def test_update_module_and_lesson_accept_a_slug():
    c, rec = make_client(lambda *_: (200, {"id": 2}))
    c.update_module(2, slug="week-1")
    _expect(rec.last["json"] == {"slug": "week-1"}, rec.last["json"])
    c.update_lesson(5, slug="intro")
    _expect(rec.last["path"] == "/teacher/lessons/5", rec.last["path"])
    _expect(rec.last["json"] == {"slug": "intro"}, rec.last["json"])


def test_detach_module_is_gone():
    """4.26.0: one writer per course (`edited_in`), not per module: the
    detach route is deleted, and its method with it."""
    c, rec = make_client()
    _expect(not hasattr(c, "detach_module"), "detach_module")
    _expect(rec.calls == [], rec.calls)


def test_upload_module_asset_multipart():
    c, rec = make_client(lambda *_: (201, {"path": "media/a.png", "url": "/x"}))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG")
        out = c.upload_module_asset(5, p)
        _expect(rec.last["method"] == "POST", rec.last["method"])
        _expect(rec.last["path"] == "/teacher/modules/5/assets", rec.last["path"])
        _expect(rec.last["files"]["file"][0] == "a.png", rec.last["files"])
        # No `path`: the server stores it as media/<name>.
        _expect(rec.last["data"] is None, rec.last["data"])
        _expect(out["path"] == "media/a.png", out)

        c.upload_module_asset(5, p, path="data/week1/a.png")
        _expect(rec.last["data"] == {"path": "data/week1/a.png"}, rec.last["data"])


def test_list_and_delete_module_assets():
    c, rec = make_client(lambda method, *_: (
        (200, [{"path": "media/a.png", "url": "/x", "size_bytes": 4}])
        if method == "GET" else (200, {"message": "Asset deleted"})))
    out = c.list_module_assets(5)
    _expect(rec.last["path"] == "/teacher/modules/5/assets", rec.last["path"])
    _expect(out[0]["size_bytes"] == 4, out)

    c.delete_module_asset(5, "data/week 1/a.png")
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/5/assets/data/week%201/a.png",
            rec.last["path"])


def test_download_module_asset_keeps_the_relative_path():
    c, rec = make_client(lambda *_: FakeResponse(200, content=b"x,y\n"))
    with tempfile.TemporaryDirectory() as d:
        out = c.download_module_asset(5, "data/week1/a.csv", d)
        _expect(out == os.path.join(d, "data", "week1", "a.csv"), out)
        with open(out, "rb") as fh:
            _expect(fh.read() == b"x,y\n", "wrong content")
        _expect(rec.last["path"]
                == "/academic_courses/assets/modules/5/data/week1/a.csv",
                rec.last["path"])
        # A path that leaves the destination is refused before any request.
        n_calls = len(rec.calls)
        for bad in ("../a.csv", "/etc/passwd", "data//a.csv"):
            try:
                c.download_module_asset(5, bad, d)
                raise AssertionError(f"expected MLArenaError for {bad}")
            except MLArenaError:
                pass
        _expect(len(rec.calls) == n_calls, "a refused path made a request")


def test_lesson_media_write_methods_are_gone():
    # The lesson-scoped media routes were replaced by the module asset store.
    c, _ = make_client()
    for name in ("upload_lesson_media", "list_lesson_media", "delete_lesson_media"):
        _expect(not hasattr(c, name), name)


# --------------------------------------------------------------------------- #
# Authoring — course composition
# --------------------------------------------------------------------------- #


def test_update_course_fields():
    c, rec = make_client(lambda *_: (200, {"id": 1}))
    c.update_course(1, name="New", visibility="public")
    _expect(rec.last["path"] == "/teacher/course/1", rec.last["path"])
    _expect(rec.last["json"] == {"name": "New", "visibility": "public"},
            rec.last["json"])


def test_instructor_name_is_gone_from_create_and_update():
    # 4.25.0: a course names its owner and its organization, no free text.
    c, rec = make_client(lambda *_: (201, {"id": 1}))
    c.create_course("RL", "RL101", "2026-01-05", "2026-06-30")
    _expect("instructor_name" not in rec.last["json"], rec.last["json"])
    for call in (lambda: c.create_course("RL", instructor_name="Prof"),
                 lambda: c.create_course("RL", None, None, None, "Prof")):
        try:
            call()
        except TypeError:
            pass
        else:
            raise AssertionError("create_course took instructor_name")
    try:
        c.update_course(1, instructor_name="Prof")
    except MLArenaError as e:
        _expect("unknown field" in str(e), str(e))
    else:
        raise AssertionError("update_course sent instructor_name")


def test_update_course_end_date_none_clears_it():
    # end_date=None is an explicit null (clear); leaving it out sends no key.
    c, rec = make_client(lambda *_: (200, {"id": 1}))
    c.update_course(1, end_date=None)
    _expect(rec.last["json"] == {"end_date": None}, rec.last["json"])

    c.update_course(1, name="New")
    _expect(rec.last["json"] == {"name": "New"}, rec.last["json"])

    # start_date is a date the route clears the same way.
    c.update_course(1, end_date="2026-12-31", start_date=None)
    _expect(rec.last["json"] == {"start_date": None, "end_date": "2026-12-31"},
            rec.last["json"])
    c.update_course(1, start_date="2026-01-05")
    _expect(rec.last["json"] == {"start_date": "2026-01-05"}, rec.last["json"])


def test_update_course_with_nothing_to_send_raises():
    c, rec = make_client(lambda *_: (200, {"id": 1}))
    try:
        c.update_course(1)
    except MLArenaError as err:
        _expect("end_date" in str(err), str(err))
    else:
        raise AssertionError("update_course(1) did not raise")
    # A None for a field that is not a date is not sent: nothing is left.
    try:
        c.update_course(1, name=None)
    except MLArenaError:
        pass
    else:
        raise AssertionError("update_course(1, name=None) did not raise")
    _expect(rec.calls == [], rec.calls)


def test_reorder_modules():
    c, rec = make_client(lambda *_: (200, []))
    c.reorder_modules(7, [3, 1])
    _expect(rec.last["path"] == "/teacher/course/7/modules/reorder", rec.last["path"])
    _expect(rec.last["json"] == {"ordered_ids": [3, 1]}, rec.last["json"])


def test_course_progress_path():
    c, rec = make_client(lambda *_: (200, {"students": []}))
    c.course_progress(4)
    _expect(rec.last["method"] == "GET", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/4/progress", rec.last["path"])


# --------------------------------------------------------------------------- #
# Consumption
# --------------------------------------------------------------------------- #


def test_course_catalog_params():
    c, rec = make_client(lambda *_: (200, {"courses": [], "total": 0}))
    c.course_catalog(search="rl", limit=10, offset=5)
    _expect(rec.last["path"] == "/academic_courses/catalog", rec.last["path"])
    _expect(rec.last["params"] == {"search": "rl", "limit": 10, "offset": 5},
            rec.last["params"])
    c.course_catalog(example=True)
    _expect(rec.last["params"] == {"example": "true"}, rec.last["params"])
    c.course_catalog()
    _expect(rec.last["params"] is None, rec.last["params"])


def test_lesson_path():
    c, rec = make_client(lambda *_: (200, {"body_md": "x", "directives": []}))
    c.lesson("intro", "foundations", "what-is-rl")
    _expect(rec.last["path"]
            == "/academic_courses/intro/modules/foundations/lessons/what-is-rl",
            rec.last["path"])


def test_lesson_progress_takes_no_course():
    """A lesson has one course: the progress writes send no body."""
    c, rec = make_client(lambda *_: (200, {"status": "completed"}))
    for method, verb in (("mark_lesson_viewed", "view"),
                         ("mark_lesson_complete", "complete"),
                         ("mark_lesson_incomplete", "uncomplete")):
        getattr(c, method)(9)
        _expect(rec.last["path"] == f"/academic_courses/lessons/9/{verb}", rec.last["path"])
        _expect(rec.last["json"] is None, rec.last["json"])
    try:
        c.mark_lesson_complete(9, course_id=4)
    except TypeError:
        pass
    else:
        raise AssertionError("course_id= is gone")


def test_not_enrolled_in_the_lessons_course_is_a_permission_error():
    c, _ = make_client(lambda *_: (403, {"error": "Not enrolled in this lesson's course"}))
    try:
        c.mark_lesson_viewed(9)
    except mlarena.PermissionDeniedError as err:
        _expect("Not enrolled in this lesson's course" in str(err), str(err))
    else:
        raise AssertionError("a 403 must raise")


def test_gated_lesson_403_maps_to_auth_error():
    c, rec = make_client(lambda *_: (403, {"error": "This lesson requires enrollment"}))
    try:
        c.lesson("intro", "m", "secret")
        raise AssertionError("expected AuthenticationError on gated 403")
    except AuthenticationError:
        pass


# --------------------------------------------------------------------------- #
# Course create / enroll
# --------------------------------------------------------------------------- #


def test_create_course_sends_only_what_was_passed():
    """`start_date` / `end_date` are optional and left out of the body when
    not given (it used to raise locally without them): a bare
    `create_course("X")` is one `{"name": "X"}` POST."""
    c, rec = make_client(lambda *_: (201, {"id": 1, "slug": "x"}))
    c.create_course("X")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["json"] == {"name": "X"}, rec.last["json"])

    c.create_course("X", code="RL101", start_date="2026-01-01", end_date="2026-02-01",
                    slug="x", description="d", visibility="public",
                    ask_student_identity=True)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/academic_courses/", rec.last["path"])
    body = rec.last["json"]
    for k, v in {"name": "X", "code": "RL101", "start_date": "2026-01-01",
                 "end_date": "2026-02-01", "slug": "x", "description": "d",
                 "visibility": "public", "ask_student_identity": True}.items():
        _expect(body.get(k) == v, f"{k}={body.get(k)}")


def test_a_third_course_visibility_carries_the_servers_sentence():
    c, _ = make_client(lambda *_: (400, {"error": "a course is public or private"}))
    try:
        c.create_course("X", visibility="unlisted")
    except MLArenaError as err:
        _expect("a course is public or private" in str(err), str(err))
    else:
        raise AssertionError("a 400 must raise")


def test_list_courses_takes_no_challenge_filter():
    c, rec = make_client(lambda *_: (200, []))
    c.list_courses(show_all=True)
    _expect(rec.last["path"] == "/academic_courses/", rec.last["path"])
    _expect(rec.last["params"] == {"show_all": "true"}, rec.last["params"])
    try:
        c.list_courses(challenge_id=7)
    except TypeError:
        pass
    else:
        raise AssertionError("challenge_id= is gone")


def test_enroll_uses_join_code():
    c, rec = make_client(lambda *_: (201, {"message": "ok"}))
    c.enroll_in_course("CODE123")
    _expect(rec.last["path"] == "/academic_courses/enroll/CODE123", rec.last["path"])

    c.enroll_in_course(join_code="JOINME")
    _expect(rec.last["path"] == "/academic_courses/enroll/JOINME", rec.last["path"])

    c.enroll_in_course("JOINME", student_email="a@b.c", student_number="42")
    _expect(rec.last["path"] == "/academic_courses/enroll/JOINME", rec.last["path"])
    _expect(rec.last["json"] == {"student_email": "a@b.c", "student_number": "42"},
            rec.last["json"])

    try:
        c.enroll_in_course()
        raise AssertionError("expected MLArenaError with no join code")
    except MLArenaError:
        pass


def test_enroll_takes_the_join_link():
    # The console hands students the link, never the bare code.
    c, rec = make_client(lambda *_: (201, {"message": "ok"}))
    c.enroll_in_course("https://ml-arena.com/enroll/QXM0W9DE")
    _expect(rec.last["path"] == "/academic_courses/enroll/QXM0W9DE", rec.last["path"])
    c.enroll_in_course("https://ml-arena.com/enroll/QXM0W9DE/?redirect=%2Fcourses%2Frl")
    _expect(rec.last["path"] == "/academic_courses/enroll/QXM0W9DE", rec.last["path"])

    c, rec = make_client(lambda *_: (200, {"course": {}}))
    c.enrollment_info("http://localhost:3000/enroll/AB12CD34")
    _expect(rec.last["path"] == "/academic_courses/enroll/AB12CD34", rec.last["path"])

    for bad in ("https://ml-arena.com/courses/rl", "https://ml-arena.com/enroll/"):
        try:
            c.enrollment_info(bad)
            raise AssertionError(f"expected MLArenaError for {bad}")
        except MLArenaError:
            pass


# --------------------------------------------------------------------------- #
# High-level helpers
# --------------------------------------------------------------------------- #


_PLAN = {
    "dry_run": False,
    "course": {"id": 100, "slug": "intro-to-rl", "action": "created"},
    "modules": [{
        "slug": "foundations", "id": 200, "action": "created",
        "renamed_from": None,
        "lessons": [{"slug": "what-is-rl", "id": 300, "action": "created",
                     "renamed_from": None}],
        "orphaned_lessons": [], "pruned_lessons": [],
        "assets": {"written": 1, "removed": 0},
    }],
    "unmanaged_modules": [],
    "source_revision": None,
    "url": "/courses/intro-to-rl",
}


def _write_manifest(d, manifest):
    with open(os.path.join(d, "course.json"), "w") as fh:
        json.dump(manifest, fh)


def _write_course_dir(d, slug="intro-to-rl"):
    """A course directory in the bundle layout: manifest, a lesson body and the
    image it links (a link is relative to the lesson file's folder)."""
    os.makedirs(os.path.join(d, "foundations", "images"))
    with open(os.path.join(d, "foundations", "what-is-rl.md"), "w") as fh:
        fh.write("# What is RL\n![loop](images/loop.png)\n")
    with open(os.path.join(d, "foundations", "images", "loop.png"), "wb") as fh:
        fh.write(b"\x89PNG")
    course = {"name": "Intro to RL"}
    if slug is not None:
        course["slug"] = slug
    _write_manifest(d, {
        "course": course,
        "modules": [{
            "title": "Foundations", "slug": "foundations",
            "challenges": [{"slug": "cartpole-v1", "label": "CartPole"}],
            "lessons": [{"title": "What is RL?", "slug": "what-is-rl",
                         "file": "foundations/what-is-rl.md",
                         "is_published": True}],
        }],
    })


def _archive_names(call):
    filename, payload = call["files"]["archive"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return filename, archive.namelist()


def test_push_course_uploads_the_directory_as_one_bundle():
    c, rec = make_client(lambda *_: (200, _PLAN))
    with tempfile.TemporaryDirectory() as d:
        _write_course_dir(d)
        out = c.push_course(d)

    _expect(out == _PLAN, out)
    # One request: the server applies the bundle, the client composes nothing.
    _expect(len(rec.calls) == 1, [call["path"] for call in rec.calls])
    call = rec.last
    _expect(call["method"] == "PUT", call["method"])
    # The slug of the route is the manifest's course.slug.
    _expect(call["path"] == "/teacher/courses/intro-to-rl/bundle", call["path"])
    _expect(call["params"] is None, call["params"])
    filename, names = _archive_names(call)
    _expect(filename == "bundle.zip", filename)
    # Rooted at the directory, manifest + body + linked file.
    _expect(names == ["course.json", "foundations/images/loop.png", "foundations/what-is-rl.md"],
            names)


def test_push_course_flags_and_explicit_slug():
    c, rec = make_client(lambda *_: (200, {**_PLAN, "dry_run": True}))
    with tempfile.TemporaryDirectory() as d:
        _write_course_dir(d, slug=None)
        # Without course.slug in the manifest there is no course to push to.
        try:
            c.push_course(d)
            raise AssertionError("expected MLArenaError without a slug")
        except MLArenaError as err:
            _expect("course.slug" in str(err), str(err))
        _expect(rec.calls == [], "a request was made without a slug")

        c.push_course(d, slug="other", dry_run=True, prune=True)
    _expect(rec.last["path"] == "/teacher/courses/other/bundle", rec.last["path"])
    _expect(rec.last["params"] == {"dry_run": "true", "prune": "true"},
            rec.last["params"])


def test_push_course_sends_an_existing_archive_as_it_is():
    c, rec = make_client(lambda *_: (200, _PLAN))
    with tempfile.TemporaryDirectory() as d:
        archive = os.path.join(d, "course.tar.gz")
        with open(archive, "wb") as fh:
            fh.write(b"not-inspected")
        # The manifest is inside the archive: the slug must be given.
        try:
            c.push_course(archive)
            raise AssertionError("expected MLArenaError: archive without slug")
        except MLArenaError:
            pass
        c.push_course(archive, slug="intro-to-rl")
    _expect(rec.last["files"]["archive"] == ("course.tar.gz", b"not-inspected"),
            rec.last["files"])


def test_push_course_refusal_carries_the_server_message():
    c, rec = make_client(lambda *_: (400, {
        "error": "module 'foundations': lesson 'x': 'foundations/x.md' is not in the bundle."}))
    with tempfile.TemporaryDirectory() as d:
        _write_course_dir(d)
        try:
            c.push_course(d)
            raise AssertionError("expected MLArenaError on a refused bundle")
        except MLArenaError as err:
            _expect("is not in the bundle" in str(err), str(err))
            _expect(err.status_code == 400, err.status_code)


def test_import_course_from_github_body():
    c, rec = make_client(lambda *_: (200, _PLAN))
    c.import_course_from_github("intro-to-rl", "prof/rl-course")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/courses/intro-to-rl/import-github",
            rec.last["path"])
    _expect(rec.last["json"] == {"repository": "prof/rl-course", "ref": None,
                                 "path": None, "dry_run": False, "prune": False},
            rec.last["json"])

    c.import_course_from_github("intro-to-rl", "prof/rl-course", ref="v2",
                                path="courses/rl", dry_run=True, prune=True)
    _expect(rec.last["json"] == {"repository": "prof/rl-course", "ref": "v2",
                                 "path": "courses/rl", "dry_run": True,
                                 "prune": True}, rec.last["json"])


def test_author_course_from_dir_is_a_deprecated_alias_of_push_course():
    """It used to compose create calls and duplicated the course on a second
    run. It now uploads the same bundle `push_course` does: one idempotent
    request, the server's plan as the result."""
    c, rec = make_client(lambda *_: (200, _PLAN))
    with tempfile.TemporaryDirectory() as d:
        _write_course_dir(d)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = c.author_course_from_dir(d)
            second = c.author_course_from_dir(d)
    _expect(any(issubclass(w.category, DeprecationWarning)
                and "push_course" in str(w.message) for w in caught), caught)
    _expect(first == _PLAN and second == _PLAN, (first, second))
    # Two runs, two identical bundle uploads, and no create route at all.
    _expect([(call["method"], call["path"]) for call in rec.calls]
            == [("PUT", "/teacher/courses/intro-to-rl/bundle")] * 2,
            [(call["method"], call["path"]) for call in rec.calls])
    _expect(rec.calls[0]["files"]["archive"][1] == rec.calls[1]["files"]["archive"][1],
            "the same directory gave two different bundles")


def test_author_course_from_dir_ships_legacy_manifest_keys_unchanged():
    # course.yaml files written before SDK 2.0 say competitions: /
    # competition_id:. The server reads those aliases; the client sends the
    # manifest as it is on disk.
    c, rec = make_client(lambda *_: (200, _PLAN))
    manifest = {
        "course": {"name": "Intro", "slug": "intro"},
        "modules": [{"title": "Foundations", "slug": "foundations",
                     "competitions": [{"competition_id": 42}]}],
    }
    with tempfile.TemporaryDirectory() as d:
        _write_manifest(d, manifest)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            c.author_course_from_dir(d)
    _, payload = rec.last["files"]["archive"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        _expect(json.loads(archive.read("course.json")) == manifest,
                "the manifest was rewritten")


_PNG = b"\x89PNG\r\n"

_EXPORTED_YAML = """\
course:
  name: Intro to RL
  slug: intro
  cover: cover.png
modules:
- title: Foundations
  slug: foundations
  lessons:
  - title: What is RL?
    slug: what-is-rl
    file: foundations/what-is-rl.md
    is_published: true
"""


def _zip(entries):
    """A .zip of `{name: bytes}`; a name ending in `@link` is a symbolic link."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            if name.endswith("@link"):
                info = zipfile.ZipInfo(name[:-len("@link")])
                info.external_attr = (0o120777 << 16)
                archive.writestr(info, content)
            else:
                archive.writestr(name, content)
    return buffer.getvalue()


_EXPORT = {
    "course.yaml": _EXPORTED_YAML.encode(),
    "cover.png": _PNG,
    "foundations/what-is-rl.md": b"# What is RL\n![loop](images/loop.png)\n",
    "foundations/images/loop.png": _PNG,
    ".github/workflows/push.yml": b"name: Push course to ML-Arena\n",
    "README.md": b"# Intro to RL\n",
}


def _export_client(entries=None, status=200):
    payload = _zip(_EXPORT if entries is None else entries)

    def router(method, path, kwargs):
        if (method, path) == ("GET", "/academic_courses/intro/bundle"):
            if status != 200:
                return (status, {"error": "You may not export this course"})
            return FakeResponse(200, content=payload)
        raise AssertionError(f"unexpected request {method} {path}")
    return make_client(router)


def _tree(d):
    found = []
    for current, _dirs, files in os.walk(d):
        for name in files:
            found.append(os.path.relpath(os.path.join(current, name), d).replace(os.sep, "/"))
    return sorted(found)


def test_export_course_to_dir_unpacks_the_server_zip():
    c, rec = _export_client()
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "intro")          # created when missing
        manifest = c.export_course_to_dir("intro", d)
        _expect(_tree(d) == sorted(_EXPORT), _tree(d))
        with open(os.path.join(d, "foundations", "images", "loop.png"), "rb") as fh:
            _expect(fh.read() == _PNG, "image not written")
    _expect([(call["method"], call["path"]) for call in rec.calls]
            == [("GET", "/academic_courses/intro/bundle")], rec.calls)
    _expect(manifest["course"]["slug"] == "intro", manifest)
    _expect(manifest["modules"][0]["lessons"][0]["file"] == "foundations/what-is-rl.md",
            manifest)


def test_export_then_push_round_trips():
    """The exported folder is a bundle `push_course` sends as it is."""
    c, _rec = _export_client()
    with tempfile.TemporaryDirectory() as d:
        c.export_course_to_dir("intro", d)
        c._request = Recorder(lambda *_: (200, _PLAN))
        c.push_course(d)
        call = c._request.last
    _expect(call["path"] == "/teacher/courses/intro/bundle", call["path"])
    _, names = _archive_names(call)
    _expect(names == sorted(_EXPORT), names)


def test_export_refuses_a_folder_that_holds_files_before_any_request():
    c, rec = _export_client()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "notes.txt"), "w") as fh:
            fh.write("mine")
        try:
            c.export_course_to_dir("intro", d)
            raise AssertionError("expected MLArenaError on a non-empty folder")
        except MLArenaError as err:
            _expect("not empty" in str(err) and "overwrite=True" in str(err), str(err))
        _expect(rec.calls == [], rec.calls)
        _expect(_tree(d) == ["notes.txt"], _tree(d))


def test_export_overwrite_replaces_the_course_files_and_keeps_the_others():
    c, _rec = _export_client()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, ".git"))
        with open(os.path.join(d, ".git", "HEAD"), "w") as fh:
            fh.write("ref: refs/heads/main\n")
        os.makedirs(os.path.join(d, "foundations"))
        with open(os.path.join(d, "foundations", "what-is-rl.md"), "w") as fh:
            fh.write("old body")
        c.export_course_to_dir("intro", d, overwrite=True)
        _expect(_tree(d) == sorted([*_EXPORT, ".git/HEAD"]), _tree(d))
        with open(os.path.join(d, "foundations", "what-is-rl.md"), "rb") as fh:
            _expect(fh.read() == _EXPORT["foundations/what-is-rl.md"], "body not replaced")


def test_export_refuses_an_unsafe_archive_and_writes_nothing():
    for bad in ({"../escape.md": b"x"}, {"/etc/passwd": b"x"},
                {"a\\b.md": b"x"}, {"C:/x.md": b"x"},
                {"foundations/link@link": b"/etc/passwd"},
                {"foundations": b"x", "foundations/a.md": b"y"}):
        c, _rec = _export_client({**_EXPORT, **bad})
        with tempfile.TemporaryDirectory() as parent:
            d = os.path.join(parent, "out")
            try:
                c.export_course_to_dir("intro", d)
                raise AssertionError(f"expected MLArenaError for {sorted(bad)}")
            except MLArenaError as err:
                _expect("nothing was written" in str(err), str(err))
            _expect(not os.path.exists(d) or _tree(d) == [], (bad, _tree(d)))
            _expect(not os.path.exists(os.path.join(parent, "escape.md")), bad)


def _refused(c, d, overwrite=False, match="nothing was written"):
    try:
        c.export_course_to_dir("intro", d, overwrite=overwrite)
    except MLArenaError as err:
        _expect(match in str(err), str(err))
        return str(err)
    raise AssertionError("expected MLArenaError")


def test_export_refuses_a_colon_in_any_part_of_a_path():
    c, _rec = _export_client({**_EXPORT, "foundations/a:b.md": b"x"})
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "out")
        _refused(c, d)
        _expect(not os.path.exists(d), "the folder was created")


def test_export_refuses_entries_that_differ_only_by_case():
    for extra in ({"readme.md": b"x"}, {"Foundations/other.md": b"x"}):
        c, _rec = _export_client({**_EXPORT, **extra})
        with tempfile.TemporaryDirectory() as parent:
            d = os.path.join(parent, "out")
            message = _refused(c, d)
            _expect("differ only by case" in message, message)
            _expect(not os.path.exists(d), "the folder was created")


def test_export_refuses_more_files_or_bytes_than_a_push_takes(monkeypatch):
    import mlarena.bundle as bundle

    monkeypatch.setattr(bundle, "MAX_EXTRACT_FILES", len(_EXPORT) - 1)
    c, _rec = _export_client()
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "out")
        _expect("files, more than" in _refused(c, d), "file cap")
        _expect(not os.path.exists(d), "the folder was created")
    monkeypatch.setattr(bundle, "MAX_EXTRACT_FILES", 5000)

    monkeypatch.setattr(bundle, "MAX_EXTRACT_BYTES", sum(map(len, _EXPORT.values())) - 1)
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "out")
        _expect("bytes, more than" in _refused(c, d), "byte cap")
        _expect(not os.path.exists(d), "the folder was created")


def _lying_zip(content=b"x" * 100, declared=10):
    """A stored entry, `big.bin`, whose headers declare `declared` bytes for
    `content` (the local header's size field at +22, the central
    directory's at +24)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("course.yaml", _EXPORTED_YAML.encode())
        archive.writestr("big.bin", content)
    payload = bytearray(buffer.getvalue())
    size, small = len(content).to_bytes(4, "little"), declared.to_bytes(4, "little")
    local = payload.rfind(b"PK\x03\x04")
    central = payload.rfind(b"PK\x01\x02")
    for offset in (local + 22, central + 24):
        _expect(payload[offset:offset + 4] == size, offset)
        payload[offset:offset + 4] = small
    return bytes(payload)


def test_export_refuses_an_entry_that_holds_more_than_it_declares():
    payload = _lying_zip()

    def router(method, path, kwargs):
        return FakeResponse(200, content=payload)
    c, _rec = make_client(router)
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "out")
        message = _refused(c, d)
        _expect("big.bin" in message, message)
        _expect(not os.path.exists(d), "a partial folder was left")


def test_export_entries_are_streamed_and_staged(monkeypatch):
    """No entry is read whole into memory, and the staging folder is gone."""
    def no_read(self, *args, **kwargs):
        raise AssertionError("ZipFile.read loads an entry into memory")
    monkeypatch.setattr(zipfile.ZipFile, "read", no_read)
    c, _rec = _export_client()
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "out")
        c.export_course_to_dir("intro", d)
        _expect(_tree(d) == sorted(_EXPORT), _tree(d))


def test_export_overwrite_refuses_a_folder_where_a_file_is():
    # A directory entry over an existing file, and a file under one.
    for entries, existing in (({**_EXPORT, "notes/": b""}, "notes"),
                              (_EXPORT, "foundations")):
        c, _rec = _export_client(entries)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, existing), "w") as fh:
                fh.write("mine")
            message = _refused(c, d, overwrite=True)
            _expect("is a file, and the course has a folder there" in message, message)
            _expect(_tree(d) == [existing], _tree(d))


def test_export_overwrite_does_not_write_through_a_symbolic_link():
    c, _rec = _export_client()
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "out")
        outside = os.path.join(parent, "outside")
        os.makedirs(d)
        os.makedirs(outside)
        os.symlink(outside, os.path.join(d, "foundations"))
        try:
            c.export_course_to_dir("intro", d, overwrite=True)
            raise AssertionError("expected MLArenaError on a symbolic link")
        except MLArenaError as err:
            _expect("symbolic link" in str(err), str(err))
        _expect(os.listdir(outside) == [], os.listdir(outside))
        _expect(not os.path.exists(os.path.join(d, "course.yaml")), "partial write")


def test_export_refusal_keeps_the_server_message():
    c, _rec = _export_client(status=403)
    with tempfile.TemporaryDirectory() as parent:
        try:
            c.export_course_to_dir("intro", os.path.join(parent, "out"))
            raise AssertionError("expected an error on a 403")
        except MLArenaError as err:
            _expect("You may not export this course" in str(err), str(err))


# --------------------------------------------------------------------------- #
# Plain-script runner (so the suite runs without pytest installed)
# --------------------------------------------------------------------------- #


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 — test runner reports all failures
            failures += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)


# --------------------------------------------------------------------------- #
# Phase 2: copy a module from another course, view as student
# --------------------------------------------------------------------------- #


def test_copy_module_into_course_body_and_status():
    row = {"module_id": 12, "title": "Week 1", "challenges": []}
    c, rec = make_client(lambda *_: (201, row))
    out = c.copy_module_into_course(7, 3)
    _expect(out == row, out)
    _expect((rec.last["method"], rec.last["path"]) ==
            ("POST", "/teacher/course/7/modules/copy"), rec.last)
    _expect(rec.last["json"] == {"module_id": 3}, rec.last["json"])


def test_create_course_module_sends_only_what_is_given():
    c, rec = make_client(lambda *_: (201, {"module_id": 12}))
    out = c.create_course_module(7, "Week 3")
    _expect(out["module_id"] == 12, out)
    _expect((rec.last["method"], rec.last["path"]) ==
            ("POST", "/teacher/course/7/modules/create"), rec.last)
    _expect(rec.last["json"] == {"title": "Week 3"}, rec.last["json"])
    c.create_course_module(7, "Week 3", slug="w3", summary="s", icon="book",
                           is_published=False)
    _expect(rec.last["json"] == {"title": "Week 3", "slug": "w3", "summary": "s",
                                 "icon": "book", "is_published": False}, rec.last["json"])
    try:
        c.create_course_module(7, "Week 3", visibility="private")
    except TypeError:
        pass
    else:
        raise AssertionError("visibility= is gone")


def test_copy_module_into_course_a_module_the_caller_may_not_read():
    c, _rec = make_client(lambda *_: (404, {"error": "Module not found"}))
    try:
        c.copy_module_into_course(7, 3)
    except mlarena.MLArenaError as exc:
        _expect("Module not found" in str(exc), str(exc))
    else:
        raise AssertionError("a 404 must raise")


def test_list_copyable_modules_path():
    rows = [{"module_id": 3, "title": "Week 1", "summary": None, "lesson_count": 4,
             "challenge_count": 1, "course": {"id": 5, "name": "RL 2025"}}]
    c, rec = make_client(lambda *_: (200, rows))
    _expect(c.list_copyable_modules(7) == rows, "rows")
    _expect((rec.last["method"], rec.last["path"]) ==
            ("GET", "/teacher/course/7/copyable-modules"), rec.last)


def test_the_learner_reads_take_as_student():
    c, rec = make_client()
    c.course("intro")
    _expect(rec.last["params"] is None, rec.last["params"])
    c.course("intro", as_student=True)
    _expect(rec.last["path"] == "/academic_courses/intro", rec.last["path"])
    _expect(rec.last["params"] == {"as_student": "true"}, rec.last["params"])
    c.module_overview("intro", "week-1", as_student=True)
    _expect(rec.last["path"] == "/academic_courses/intro/modules/week-1", rec.last["path"])
    _expect(rec.last["params"] == {"as_student": "true"}, rec.last["params"])
    c.module_overview("intro", "week-1")
    _expect(rec.last["params"] is None, rec.last["params"])
    c.lesson("intro", "week-1", "open", as_student=True)
    _expect(rec.last["path"] == "/academic_courses/intro/modules/week-1/lessons/open",
            rec.last["path"])
    _expect(rec.last["params"] == {"as_student": "true"}, rec.last["params"])
    c.lesson("intro", "week-1", "open")
    _expect(rec.last["params"] is None, rec.last["params"])
