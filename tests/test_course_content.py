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


def make_client(router=None, scope="teacher"):
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


def test_create_module_body_and_auth():
    c, rec = make_client(lambda *_: (201, {"id": 1}))
    c.create_module("NLP", summary="intro", visibility="public")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules", rec.last["path"])
    # `is_published` (the student gate) rides in every create body, default
    # True, so an existing course dir publishes unchanged.
    _expect(rec.last["json"] == {"title": "NLP", "visibility": "public",
                                 "is_published": True, "summary": "intro"},
            rec.last["json"])
    auth = rec.last["headers"]["Authorization"]
    _expect(auth.startswith("Bearer mlk_teacher_"), auth)


def test_list_modules_library_param():
    c, rec = make_client(lambda *_: (200, []))
    c.list_modules(library="public")
    _expect(rec.last["path"] == "/teacher/modules", rec.last["path"])
    _expect(rec.last["params"] == {"library": "public"}, rec.last["params"])

    c.list_modules()
    _expect(rec.last["params"] is None, rec.last["params"])


def test_update_module_partial_and_unknown_field():
    c, rec = make_client(lambda *_: (200, {"id": 2}))
    c.update_module(2, visibility="unlisted")
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/2", rec.last["path"])
    _expect(rec.last["json"] == {"visibility": "unlisted"}, rec.last["json"])

    try:
        c.update_module(2, bogus="x")
        raise AssertionError("expected MLArenaError on unknown field")
    except MLArenaError:
        pass

    try:
        c.update_module(2)
        raise AssertionError("expected MLArenaError on empty update")
    except MLArenaError:
        pass


def test_delete_module_force_flag():
    c, rec = make_client(lambda *_: (200, {"message": "ok"}))
    c.delete_module(3, force=True)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/3", rec.last["path"])
    _expect(rec.last["params"] == {"force": "1"}, rec.last["params"])

    c.delete_module(3)
    _expect(rec.last["params"] is None, rec.last["params"])


def test_attach_challenge_body():
    c, rec = make_client(lambda *_: (201, {"id": 9}))
    c.attach_challenge(2, 42, label="CartPole")
    _expect(rec.last["path"] == "/teacher/modules/2/challenges", rec.last["path"])
    _expect(rec.last["json"] == {"challenge_id": 42, "label": "CartPole"},
            rec.last["json"])


def test_update_and_detach_challenge_link_paths():
    c, rec = make_client(lambda *_: (200, {"ok": True}))
    c.update_challenge_link(2, 42, pass_threshold=None)
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/42", rec.last["path"])
    _expect(rec.last["json"] == {"pass_threshold": None}, rec.last["json"])

    c.detach_challenge(2, 42)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/2/challenges/42", rec.last["path"])


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


def test_update_module_and_lesson_accept_a_slug():
    c, rec = make_client(lambda *_: (200, {"id": 2}))
    c.update_module(2, slug="week-1")
    _expect(rec.last["json"] == {"slug": "week-1"}, rec.last["json"])
    c.update_lesson(5, slug="intro")
    _expect(rec.last["path"] == "/teacher/lessons/5", rec.last["path"])
    _expect(rec.last["json"] == {"slug": "intro"}, rec.last["json"])


def test_detach_module():
    c, rec = make_client(lambda *_: (200, {"id": 2, "content_origin": "editor"}))
    out = c.detach_module(2)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/modules/2/detach", rec.last["path"])
    _expect(out["content_origin"] == "editor", out)


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


def test_link_and_reorder_modules():
    # link returns 201 (created), reorder returns 200 — route by verb.
    def router(method, path, kwargs):
        return (201, {"id": 1}) if method == "POST" else (200, [])

    c, rec = make_client(router)
    c.link_module(7, 3, position=1)
    _expect(rec.last["path"] == "/teacher/course/7/modules", rec.last["path"])
    _expect(rec.last["json"] == {"module_id": 3, "position": 1}, rec.last["json"])

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


def test_lesson_path():
    c, rec = make_client(lambda *_: (200, {"body_md": "x", "directives": []}))
    c.lesson("intro", "foundations", "what-is-rl")
    _expect(rec.last["path"]
            == "/academic_courses/intro/modules/foundations/lessons/what-is-rl",
            rec.last["path"])


def test_mark_lesson_complete_course_id():
    c, rec = make_client(lambda *_: (200, {"status": "completed"}))
    c.mark_lesson_complete(9, course_id=4)
    _expect(rec.last["path"] == "/academic_courses/lessons/9/complete", rec.last["path"])
    _expect(rec.last["json"] == {"course_id": 4}, rec.last["json"])

    c.mark_lesson_complete(9)
    _expect(rec.last["json"] == {}, rec.last["json"])


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
                    slug="x", description="d", visibility="public")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/academic_courses/", rec.last["path"])
    body = rec.last["json"]
    for k, v in {"name": "X", "code": "RL101", "start_date": "2026-01-01",
                 "end_date": "2026-02-01", "slug": "x", "description": "d",
                 "visibility": "public"}.items():
        _expect(body.get(k) == v, f"{k}={body.get(k)}")


def test_list_courses_challenge_filter():
    c, rec = make_client(lambda *_: (200, []))
    c.list_courses(show_all=True, challenge_id=7)
    _expect(rec.last["path"] == "/academic_courses/", rec.last["path"])
    _expect(rec.last["params"] == {"show_all": "true", "challenge_id": 7},
            rec.last["params"])


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
    image it links (paths relative to the root)."""
    os.makedirs(os.path.join(d, "foundations"))
    os.makedirs(os.path.join(d, "media"))
    with open(os.path.join(d, "foundations", "what-is-rl.md"), "w") as fh:
        fh.write("# What is RL\n![loop](media/loop.png)\n")
    with open(os.path.join(d, "media", "loop.png"), "wb") as fh:
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
    _expect(names == ["course.json", "foundations/what-is-rl.md", "media/loop.png"],
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


def _export_router(method, path, kwargs):
    if path == "/academic_courses/intro":
        return (200, {
            "id": 9, "name": "Intro to RL", "code": "RL101", "slug": "intro",
            "description": "d", "visibility": "public",
            "instructor_name": "Prof", "start_date": "2026-01-01",
            "end_date": "2026-06-01", "has_cover": True, "can_manage": True,
            "modules": [{
                "id": 1, "title": "Foundations", "slug": "foundations",
                "summary": "s", "icon": "book", "layout": "advanced",
                "position": 0, "is_published": True,
                "challenges": [
                    {"challenge": {"id": 42, "name": "CartPole-v1",
                                   "slug": "cartpole-v1"},
                     "label": "CartPole", "pass_threshold": 195.0, "metric": None},
                    {"challenge": {"id": 43, "name": "Old", "slug": "old"},
                     "label": None, "pass_threshold": None, "metric": None},
                ],
                "lessons": [
                    {"id": 10, "parent_lesson_id": None, "title": "What is RL?",
                     "slug": "what-is-rl", "kind": "lesson", "gated": False,
                     "is_published": True, "estimated_minutes": 10},
                    {"id": 11, "parent_lesson_id": 10, "title": "Lab",
                     "slug": "lab", "kind": "exercise", "gated": True,
                     "is_published": False, "estimated_minutes": None},
                ],
            }],
        })
    if path.endswith("/lessons/what-is-rl"):
        return (200, {
            "body_md": ("# What is RL\nexported body\n"
                        "![loop](media/loop.png) [site](https://example.org/x.png)\n"
                        "```\n![not a link](media/in-code.png)\n```\n"
                        "[gone](media/gone.csv) [up](../secret.txt)\n"),
            "asset_base_url": "/api/academic_courses/assets/modules/1/",
        })
    if path.endswith("/lessons/lab"):
        return (200, {
            "body_md": "[data](./data/week1.csv?raw=1#top)",
            "asset_base_url": "/api/academic_courses/assets/modules/1/",
        })
    if path == "/teacher/modules/1":
        return (200, {"id": 1, "visibility": "unlisted"})
    if path == "/academic_courses/assets/modules/1/media/loop.png":
        return FakeResponse(200, content=_PNG)
    if path == "/academic_courses/assets/modules/1/data/week1.csv":
        return FakeResponse(200, content=b"x,y\n")
    if path == "/academic_courses/assets/modules/1/media/gone.csv":
        return (404, {"error": "File not found"})
    if path == "/academic_courses/assets/courses/9/cover":
        return FakeResponse(200, content=_PNG,
                            headers={"Content-Type": "image/png"})
    raise AssertionError(f"unexpected request {method} {path}")


def test_export_course_to_dir():
    c, rec = make_client(_export_router)
    with tempfile.TemporaryDirectory() as d:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manifest = c.export_course_to_dir("intro", d)

        def read(*parts):
            with open(os.path.join(d, *parts), "rb") as fh:
                return fh.read()

        _expect(b"exported body" in read("foundations", "what-is-rl.md"),
                "wrong body content")
        # Linked files land where the body's relative link points, from the root.
        _expect(read("media", "loop.png") == _PNG, "image not exported")
        _expect(read("data", "week1.csv") == b"x,y\n", "data file not exported")
        _expect(read("cover.png") == _PNG, "cover not exported")
        _expect(not os.path.exists(os.path.join(d, "media", "in-code.png")),
                "a link inside a code fence was followed")
        manifest_files = [n for n in ("course.yaml", "course.json")
                          if os.path.isfile(os.path.join(d, n))]
        _expect(len(manifest_files) == 1, manifest_files)

    # A linked file the module does not hold is reported, not fatal.
    _expect(any("media/gone.csv" in str(w.message) for w in caught), caught)
    paths = [call["path"] for call in rec.calls]
    _expect(not any("secret" in p or "example.org" in p or "in-code" in p
                    for p in paths), paths)

    _expect(manifest["course"] == {
        "name": "Intro to RL", "slug": "intro", "code": "RL101",
        "description": "d", "visibility": "public", "instructor_name": "Prof",
        "start_date": "2026-01-01", "end_date": "2026-06-01",
        "cover": "cover.png",
    }, manifest["course"])
    module = manifest["modules"][0]
    _expect(module["visibility"] == "unlisted", module)
    # By slug (every challenge has one): the same on every deployment.
    _expect(module["challenges"] == [
        {"slug": "cartpole-v1", "label": "CartPole", "pass_threshold": 195.0},
        {"slug": "old", "label": None, "pass_threshold": None},
    ], module["challenges"])
    _expect(module["lessons"] == [{
        "title": "What is RL?", "slug": "what-is-rl", "kind": "lesson",
        "file": "foundations/what-is-rl.md", "gated": False,
        "is_published": True, "estimated_minutes": 10,
        "children": [{
            "title": "Lab", "slug": "lab", "kind": "exercise",
            "file": "foundations/lab.md", "gated": True,
            "is_published": False, "estimated_minutes": None,
        }],
    }], module["lessons"])


def test_export_then_push_round_trips():
    """What export writes is a bundle `push_course` accepts as it is: the
    manifest's slug names the route and every file the bodies link is in the
    archive under the path they link."""
    c, rec = make_client(_export_router)
    with tempfile.TemporaryDirectory() as d:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c.export_course_to_dir("intro", d)
        # course.yaml with PyYAML installed, course.json without.
        manifest_name = next(n for n in ("course.yaml", "course.json")
                             if os.path.isfile(os.path.join(d, n)))
        c._request = Recorder(lambda *_: (200, _PLAN))
        c.push_course(d)
        call = c._request.last
    _expect(call["path"] == "/teacher/courses/intro/bundle", call["path"])
    _, names = _archive_names(call)
    _expect(names == sorted([manifest_name, "cover.png", "data/week1.csv",
                             "foundations/lab.md", "foundations/what-is-rl.md",
                             "media/loop.png"]), names)


def test_export_refuses_two_modules_with_one_asset_path():
    def router(method, path, kwargs):
        if path == "/academic_courses/intro":
            lesson = {"id": 1, "parent_lesson_id": None, "title": "L", "slug": "l",
                      "kind": "lesson", "gated": False, "is_published": True,
                      "estimated_minutes": None}
            module = {"title": "M", "summary": None, "icon": None,
                      "is_published": True, "challenges": [], "lessons": [lesson]}
            return (200, {
                "id": 9, "name": "Intro", "code": None, "slug": "intro",
                "description": None, "visibility": "public",
                "instructor_name": None, "start_date": None, "end_date": None,
                "has_cover": False, "can_manage": False,
                "modules": [{**module, "id": 1, "slug": "a"},
                            {**module, "id": 2, "slug": "b"}],
            })
        for module_id, slug in ((1, "a"), (2, "b")):
            if path == f"/academic_courses/intro/modules/{slug}/lessons/l":
                return (200, {
                    "body_md": "![f](media/fig.png)",
                    "asset_base_url": f"/api/academic_courses/assets/modules/{module_id}/",
                })
            if path == f"/academic_courses/assets/modules/{module_id}/media/fig.png":
                return FakeResponse(200, content=slug.encode())
        raise AssertionError(f"unexpected request {method} {path}")

    c, rec = make_client(router)
    with tempfile.TemporaryDirectory() as d:
        try:
            c.export_course_to_dir("intro", d)
            raise AssertionError("expected MLArenaError on the asset conflict")
        except MLArenaError as err:
            _expect("media/fig.png" in str(err), str(err))


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


def test_a_link_without_its_challenge_is_a_clear_export_error():
    """An authoring link serves `challenge: None` for a challenge the caller
    may not see: the manifest entry names the id, it does not crash."""
    from mlarena.client import _manifest_challenge_entry

    try:
        _manifest_challenge_entry(
            {"id": 3, "challenge_id": 43, "challenge": None, "label": None,
             "pass_threshold": None})
    except MLArenaError as exc:
        _expect("challenge 43" in str(exc), str(exc))
    else:
        raise AssertionError("a link without its challenge must raise MLArenaError")
