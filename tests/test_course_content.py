"""SDK tests for the course-content methods.

These mock the REST surface by replacing the client's single HTTP chokepoint
(``MLArenaClient._request``) with a recorder that captures every call and
returns canned responses — no network, no backend, no extra dependencies
(stdlib + ``requests`` only). Each test asserts the method → (verb, path, body,
params) mapping that the parity rule requires.

Runs under pytest *or* directly: ``python tests/test_course_content.py``.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena.client import MLArenaClient
from mlarena.exceptions import AuthenticationError, MLArenaError


# --------------------------------------------------------------------------- #
# Test harness: a fake transport recording calls + returning routed responses
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text

    def json(self):
        return self._json

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


def test_upload_lesson_media_multipart():
    c, rec = make_client(lambda *_: (201, {"filename": "a.png", "url": "/x"}))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.png")
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG")
        out = c.upload_lesson_media(5, p)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/lessons/5/media", rec.last["path"])
    _expect("files" in rec.last, "multipart files missing")
    _expect(out["url"] == "/x", out)


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


def _author_router(method, path, kwargs):
    if method == "POST" and path == "/academic_courses/":
        return (201, {"id": 100, "slug": "intro-to-rl", "join_code": "JOINME"})
    if method == "POST" and path == "/teacher/modules":
        return (201, {"id": 200})
    if method == "POST" and path.endswith("/lessons"):
        return (201, {"id": 300})
    if method == "PUT" and path.startswith("/teacher/lessons/"):
        return (200, {"id": 300, "is_published": True})
    if method == "POST" and path.endswith("/challenges"):
        return (201, {"id": 1})
    if method == "POST" and path.endswith("/modules"):  # link_module
        return (201, {"id": 1})
    if method == "PUT" and path.endswith("/modules/reorder"):
        return (200, [])
    return (200, {"ok": True})


def _write_manifest(d, manifest):
    import json
    with open(os.path.join(d, "course.json"), "w") as fh:
        json.dump(manifest, fh)


def _one_module_manifest(module_links):
    return {
        "course": {"name": "Intro", "start_date": "2026-01-01",
                   "end_date": "2026-06-01"},
        "modules": [{"title": "Foundations", **module_links}],
    }


def test_author_course_from_dir():
    c, rec = make_client(_author_router)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "foundations"))
        with open(os.path.join(d, "foundations", "what-is-rl.md"), "w") as fh:
            fh.write("# What is RL\nbody")
        manifest = {
            "course": {
                "name": "Intro to RL", "code": "RL101",
                "start_date": "2026-01-01", "end_date": "2026-06-01",
                "visibility": "public",
            },
            "modules": [
                {
                    "title": "Foundations", "visibility": "public",
                    "challenges": [{"challenge_id": 42, "label": "CartPole"}],
                    "lessons": [
                        {"title": "What is RL?", "file": "foundations/what-is-rl.md",
                         "is_published": True, "estimated_minutes": 10},
                    ],
                },
                {"title": "Reuse", "module_id": 555},  # link an existing module
            ],
        }
        import json
        with open(os.path.join(d, "course.json"), "w") as fh:
            json.dump(manifest, fh)

        out = c.author_course_from_dir(d)

    _expect(out["course_id"] == 100, out)
    _expect(out["join_code"] == "JOINME", out)
    _expect(len(out["modules"]) == 2, out["modules"])
    _expect(out["modules"][0]["created"] is True, out["modules"][0])
    _expect(out["modules"][1]["created"] is False, out["modules"][1])

    paths = [(call["method"], call["path"]) for call in rec.calls]
    _expect(("POST", "/academic_courses/") in paths, paths)
    _expect(("POST", "/teacher/modules") in paths, paths)
    _expect(("POST", "/teacher/modules/200/lessons") in paths, paths)
    _expect(("PUT", "/teacher/lessons/300") in paths, paths)
    _expect(("POST", "/teacher/modules/200/challenges") in paths, paths)
    attach = next(c for c in rec.calls if c["path"] == "/teacher/modules/200/challenges")
    _expect(attach["json"] == {"challenge_id": 42, "label": "CartPole"}, attach["json"])
    _expect(("POST", "/teacher/course/100/modules") in paths, paths)
    _expect(("PUT", "/teacher/course/100/modules/reorder") in paths, paths)

    # The lesson body was read from disk and posted.
    lesson_call = next(c for c in rec.calls if c["path"] == "/teacher/modules/200/lessons")
    _expect("body" in lesson_call["json"]["body_md"], lesson_call["json"])

    # The existing module (555) was linked, never re-created.
    link_calls = [c["json"]["module_id"] for c in rec.calls
                  if c["path"] == "/teacher/course/100/modules"]
    _expect(link_calls == [200, 555], link_calls)


def test_author_course_from_dir_reads_legacy_manifest_keys():
    # course.yaml files written before SDK 2.0 say competitions: / competition_id:.
    c, rec = make_client(_author_router)
    with tempfile.TemporaryDirectory() as d:
        _write_manifest(d, _one_module_manifest({
            "competitions": [{"competition_id": 42, "label": "CartPole",
                              "pass_threshold": 195.0}],
        }))
        c.author_course_from_dir(d)
    attach = [call for call in rec.calls
              if call["path"] == "/teacher/modules/200/challenges"]
    _expect(len(attach) == 1, [call["path"] for call in rec.calls])
    _expect(attach[0]["json"] == {"challenge_id": 42, "label": "CartPole",
                                  "pass_threshold": 195.0}, attach[0]["json"])


def test_author_course_from_dir_rejects_both_spellings():
    for links in (
        {"challenges": [{"challenge_id": 1}], "competitions": [{"competition_id": 1}]},
        {"challenges": [{"challenge_id": 1, "competition_id": 1}]},
    ):
        c, rec = make_client(_author_router)
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(d, _one_module_manifest(links))
            try:
                c.author_course_from_dir(d)
                raise AssertionError(f"expected MLArenaError for {links}")
            except MLArenaError:
                pass
        # Validated before the first request: nothing was created.
        _expect(rec.calls == [], [call["path"] for call in rec.calls])


def _export_router(method, path, kwargs):
    if path == "/academic_courses/intro":
        return (200, {
            "name": "Intro to RL", "code": "RL101", "slug": "intro",
            "description": "d", "visibility": "public",
            "instructor_name": "Prof", "start_date": "2026-01-01",
            "end_date": "2026-06-01",
            "modules": [{
                "id": 1, "title": "Foundations", "slug": "foundations",
                "summary": "s", "icon": "book", "layout": "advanced",
                "challenges": [{
                    "challenge": {"id": 42, "name": "CartPole-v1"},
                    "label": "CartPole", "pass_threshold": None,
                    "ranked_order": "desc",
                }],
                "lessons": [{"title": "What is RL?", "slug": "what-is-rl",
                             "kind": "lesson", "gated": False,
                             "is_published": True, "estimated_minutes": 10}],
            }],
        })
    if path.endswith("/lessons/what-is-rl"):
        return (200, {"body_md": "# What is RL\nexported body"})
    return (200, {"ok": True})


def test_export_course_to_dir():
    c, rec = make_client(_export_router)
    with tempfile.TemporaryDirectory() as d:
        manifest = c.export_course_to_dir("intro", d)
        body_path = os.path.join(d, "foundations", "what-is-rl.md")
        _expect(os.path.isfile(body_path), "lesson body not written")
        with open(body_path) as fh:
            _expect("exported body" in fh.read(), "wrong body content")
        manifest_files = [os.path.join(d, n) for n in ("course.yaml", "course.json")
                          if os.path.isfile(os.path.join(d, n))]
        _expect(manifest_files, "manifest not written")
        with open(manifest_files[0]) as fh:
            written = fh.read()
        _expect("challenge_id" in written and "competition" not in written, written)

    _expect(manifest["course"]["name"] == "Intro to RL", manifest)
    _expect(manifest["modules"][0]["lessons"][0]["file"]
            == os.path.join("foundations", "what-is-rl.md"), manifest)
    _expect(manifest["modules"][0]["challenges"]
            == [{"challenge_id": 42, "label": "CartPole", "pass_threshold": None}],
            manifest["modules"][0])
    _expect("competitions" not in manifest["modules"][0], manifest["modules"][0])


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
