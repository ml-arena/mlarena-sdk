"""SDK tests for the teacher administration methods (roster, assistants,
exports, course cover) — the routes the console had and the SDK did not.

Same offline harness as ``test_course_content.py``: the client's single HTTP
chokepoint (``MLArenaClient._request``) is replaced by a recorder, so each test
asserts the method → (verb, path, body, params) mapping the parity rule
requires without a network or a backend.

Runs under pytest *or* directly: ``python tests/test_teacher_admin.py``.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena.exceptions import MLArenaError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content=b""):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text
        self._content = content

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        yield self._content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status {self.status_code}")


class Recorder:
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
    client._request = rec
    return client, rec


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# Course list + challenge picker
# --------------------------------------------------------------------------- #


def test_teacher_courses_reads_role_per_row():
    rows = [{"id": 1, "name": "RL", "role": "teacher"},
            {"id": 2, "name": "NLP", "role": "assistant"}]
    c, rec = make_client(lambda *_: (200, rows))
    out = c.teacher_courses()
    _expect(rec.last["method"] == "GET", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/courses", rec.last["path"])
    roles = list(out["role"]) if hasattr(out, "columns") else [r["role"] for r in out]
    _expect(roles == ["teacher", "assistant"], roles)


def test_challenges_for_course_lists_the_picker():
    rows = [{"id": 42, "name": "CartPole-v1", "ranked_by": "reward",
             "ranked_order": "desc", "precision": 2}]
    c, rec = make_client(lambda *_: (200, rows))
    out = c.challenges_for_course()
    _expect(rec.last["path"] == "/teacher/challenges-for-course", rec.last["path"])
    ids = list(out["id"]) if hasattr(out, "columns") else [r["id"] for r in out]
    _expect(ids == [42], ids)


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #


def test_course_students_scopes_to_a_challenge():
    payload = {
        "challenges": [{"id": 42, "name": "CartPole-v1"}],
        "selected_challenge_id": 42,
        "students": [{
            "user_id": 7, "username": "ana", "student_number": "S1",
            "student_email": "ana@uni.edu", "project_url": None,
            "team_id": None, "team_name": None,
            "enrolled_at_ts": "2026-02-01T09:00:00Z",
        }],
    }
    c, rec = make_client(lambda *_: (200, payload))
    out = c.course_students(5, challenge_id=42)
    _expect(rec.last["method"] == "GET", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/students/5", rec.last["path"])
    _expect(rec.last["params"] == {"challenge_id": 42}, rec.last["params"])
    student = out["students"][0]
    # The enrollment row is keyed by its FK, and dated by when they enrolled.
    _expect(student["user_id"] == 7, student)
    _expect(student["enrolled_at_ts"].endswith("Z"), student)
    _expect("id" not in student and "created_at" not in student, student)


def test_course_students_without_a_challenge_sends_no_params():
    c, rec = make_client(lambda *_: (200, {"students": []}))
    c.course_students(5)
    _expect(rec.last["params"] is None, rec.last["params"])


def test_remove_student_deletes_the_enrollment():
    c, rec = make_client(lambda *_: (200, {"message": "Student removed successfully"}))
    out = c.remove_student(5, 7)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/student/5/7", rec.last["path"])
    _expect(out["message"].startswith("Student removed"), out)


# --------------------------------------------------------------------------- #
# Teaching assistants
# --------------------------------------------------------------------------- #


def test_course_assistants_round_trip():
    rows = [{"user_id": 9, "username": "bob", "created_at_ts": "2026-02-01T09:00:00Z"}]
    c, rec = make_client(lambda *_: (200, rows))
    out = c.course_assistants(5)
    _expect(rec.last["path"] == "/teacher/course/5/assistants", rec.last["path"])
    ids = list(out["user_id"]) if hasattr(out, "columns") else [r["user_id"] for r in out]
    _expect(ids == [9], ids)


def test_add_course_assistant_sends_the_username():
    c, rec = make_client(lambda *_: (201, {"message": "ok", "user_id": 9,
                                           "username": "bob"}))
    out = c.add_course_assistant(5, "bob")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/assistants", rec.last["path"])
    _expect(rec.last["json"] == {"username": "bob"}, rec.last["json"])
    _expect(out["user_id"] == 9, out)


def test_remove_course_assistant_takes_the_user_id():
    c, rec = make_client(lambda *_: (200, {"message": "ok"}))
    c.remove_course_assistant(5, 9)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/assistants/9", rec.last["path"])


def test_assistant_errors_surface():
    c, _rec = make_client(lambda *_: (409, {"error": "already a TA"}))
    try:
        c.add_course_assistant(5, "bob")
        raise AssertionError("expected MLArenaError on a 409")
    except MLArenaError:
        pass


# --------------------------------------------------------------------------- #
# Downloads (CSV export + cover)
# --------------------------------------------------------------------------- #


def _csv_router(method, path, kwargs):
    if path.startswith("/teacher/export-csv/"):
        return FakeResponse(200, content=b"Rank,Username\n1,ana\n")
    return None


def test_export_course_csv_writes_the_file():
    c, rec = make_client(_csv_router)
    with tempfile.TemporaryDirectory() as d:
        out = c.export_course_csv(5, challenge_id=42, by_participant=True, dest_dir=d)
        _expect(os.path.isfile(out), out)
        _expect(os.path.basename(out) == "leaderboard_course_5_challenge_42.csv", out)
        with open(out) as fh:
            _expect("ana" in fh.read(), "csv body not written")
    _expect(rec.last["path"] == "/teacher/export-csv/5", rec.last["path"])
    _expect(rec.last["params"] == {"by_participant": "true", "challenge_id": 42},
            rec.last["params"])


def test_export_course_csv_defaults_to_by_submission():
    c, rec = make_client(_csv_router)
    with tempfile.TemporaryDirectory() as d:
        c.export_course_csv(5, 42, dest_dir=d)
    _expect(rec.last["params"] == {"by_participant": "false", "challenge_id": 42},
            rec.last["params"])


def test_export_course_csv_requires_the_challenge():
    """The route exports one attached challenge; there is no "first" default."""
    c, rec = make_client(_csv_router)
    try:
        c.export_course_csv(5)
    except TypeError:
        pass
    else:
        raise AssertionError("export_course_csv(course_id) without a challenge_id")
    _expect(rec.calls == [], rec.calls)


# --------------------------------------------------------------------------- #
# Add a challenge to a course (one transactional route)
# --------------------------------------------------------------------------- #


def test_add_course_challenge_is_one_call():
    row = {"id": 9, "module_id": 4, "title": "Week 2", "challenges": []}
    c, rec = make_client(lambda *_: (201, row))
    out = c.add_course_challenge(7, 42, title="Week 2", summary="Read me.",
                                 pass_threshold=0.5)
    _expect(out == row, out)
    _expect(len(rec.calls) == 1, rec.calls)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/7/challenges", rec.last["path"])
    _expect(rec.last["json"] == {
        "challenge_id": 42, "is_published": True, "title": "Week 2",
        "summary": "Read me.", "pass_threshold": 0.5,
    }, rec.last["json"])


def test_remove_course_challenge_is_one_call():
    body = {"message": "Challenge removed", "module_deleted": True}
    c, rec = make_client(lambda *_: (200, body))
    out = c.remove_course_challenge(7, 4)
    _expect(out == body, out)
    _expect(len(rec.calls) == 1, rec.calls)
    _expect(rec.last["method"] == "DELETE", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/7/challenges/4", rec.last["path"])


def test_add_course_challenge_sends_only_what_was_given():
    c, rec = make_client(lambda *_: (201, {"id": 1}))
    c.add_course_challenge(7, 42, is_published=False)
    _expect(rec.last["json"] == {"challenge_id": 42, "is_published": False},
            rec.last["json"])


def test_add_course_challenge_refusal_raises_with_the_reason():
    c, _rec = make_client(lambda *_: (400, {"error": "challenge_id 42 has no evaluation"}))
    try:
        c.add_course_challenge(7, 42)
    except MLArenaError as e:
        _expect("has no evaluation" in str(e), str(e))
        _expect(e.status_code == 400, e.status_code)
    else:
        raise AssertionError("expected MLArenaError")


# --------------------------------------------------------------------------- #
# Maintenance mode (503)
# --------------------------------------------------------------------------- #


def test_maintenance_503_is_a_maintenance_error_with_the_admin_message():
    body = {"error": "Back at noon.", "maintenance_mode": True}
    c, _rec = make_client(lambda *_: (503, body))
    try:
        c.teacher_courses()
    except mlarena.MaintenanceError as e:
        _expect(str(e) == "Back at noon.", str(e))
        _expect(e.status_code == 503 and e.body == body, (e.status_code, e.body))
        _expect(isinstance(e, MLArenaError), type(e).__mro__)
    else:
        raise AssertionError("expected MaintenanceError")


def test_a_503_without_the_flag_is_not_maintenance():
    c, _rec = make_client(lambda *_: (503, {"error": "upstream down",
                                             "upstream": "data_collection"}))
    try:
        c.teacher_courses()
    except mlarena.MaintenanceError:
        raise AssertionError("a plain 503 is not maintenance")
    except MLArenaError as e:
        _expect("upstream down" in str(e), str(e))
    else:
        raise AssertionError("expected MLArenaError")


def test_teacher_courses_says_whether_there_is_a_cover_not_where_it_lives():
    """A course payload never carries the NFS path — the catalog and the
    landing are public reads (see the backend's `has_cover`)."""
    rows = [{"id": 1, "name": "RL", "role": "teacher", "has_cover": True}]
    c, _rec = make_client(lambda *_: (200, rows))
    out = c.teacher_courses()
    row = out.to_dict("records")[0] if hasattr(out, "columns") else out[0]
    _expect(row["has_cover"] is True, row)
    _expect("cover_image_path" not in row, row)


def test_set_course_cover_answers_the_flag():
    def router(method, path, kwargs):
        if path.endswith("/cover"):
            return (201, {"has_cover": True})
        return None

    c, rec = make_client(router)
    with tempfile.TemporaryDirectory() as d:
        image = os.path.join(d, "cover.png")
        with open(image, "wb") as fh:
            fh.write(b"\x89PNG")
        out = c.set_course_cover(5, image)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == "/teacher/course/5/cover", rec.last["path"])
    _expect(out == {"has_cover": True}, out)


def test_course_cover_downloads_from_the_asset_route():
    def router(method, path, kwargs):
        if path.endswith("/cover"):
            return FakeResponse(200, content=b"\x89PNG")
        return None

    c, rec = make_client(router)
    with tempfile.TemporaryDirectory() as d:
        out = c.course_cover(5, dest_dir=d)
        _expect(os.path.isfile(out), out)
    _expect(rec.last["path"] == "/academic_courses/assets/courses/5/cover",
            rec.last["path"])


def test_course_cover_404_raises():
    c, _rec = make_client(lambda *_: FakeResponse(404, {"error": "No cover image"}))
    with tempfile.TemporaryDirectory() as d:
        try:
            c.course_cover(5, dest_dir=d)
            raise AssertionError("expected an error for a course with no cover")
        except Exception as exc:  # NotFoundError or MLArenaError, both fine
            _expect(not isinstance(exc, AssertionError), repr(exc))


# --------------------------------------------------------------------------- #
# Every new method carries the bearer token
# --------------------------------------------------------------------------- #


def test_all_teacher_admin_methods_send_the_token():
    c, rec = make_client(lambda *_: (200, []))
    c.teacher_courses()
    c.challenges_for_course()
    c.course_students(1)
    c.course_assistants(1)
    for call in rec.calls:
        auth = call["headers"]["Authorization"]
        _expect(auth.startswith("Bearer mlk_teacher_"), auth)


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
