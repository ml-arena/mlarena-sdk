"""SDK tests for the helpers of the New dialog and of the examples
(docs/course_challenges.md §7.2, §7.3): `create_challenge_for_class`,
`use_example_challenge`, `start_from_example_course`. Each is a client-side
composition of public routes — `POST /academic_courses/`,
`POST /teacher/course/{id}/challenges`, `POST /teacher/course/{id}/duplicate`
and the reads that name the course or the challenge. No route of their own.

Offline harness: the client's single HTTP chokepoint (`MLArenaClient._request`)
is replaced by a recorder. Runs under pytest *or* directly:
``python tests/test_course_challenge_helpers.py``.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena.exceptions import MLArenaError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = ""

    def json(self):
        return self._json


class Recorder:
    def __init__(self, routes):
        self.calls = []
        self.routes = routes

    def __call__(self, method, url, **kwargs):
        path = url.split("/api", 1)[1]
        self.calls.append({"method": method, "path": path, **kwargs})
        status, data = self.routes[(method, path)]
        return FakeResponse(status, data)

    def signature(self):
        return [(c["method"], c["path"]) for c in self.calls]


def make_client(routes, scope="author"):
    client = mlarena.connect(
        api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
        base_url="https://example.com",
    )
    rec = Recorder(routes)
    client._request = rec
    return client, rec


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


COURSE = {
    "id": 42, "name": "Titanic", "code": None, "slug": "titanic", "description": None,
    "visibility": "private", "has_cover": False, "instructor_name": None,
    "join_code": "JOIN1234", "start_date": None, "end_date": None,
    "archived_at_ts": None, "single_challenge_id": None, "example_listed_at_ts": None,
}
CREATED = {
    "message": "Challenge created", "challenge_id": 77, "name": "Titanic",
    "slug": "titanic-2", "course_id": 42, "module_id": 90, "module_created": True,
    "position": 0, "pass_threshold": None, "copied_from_challenge_id": None,
    "machine_name": "small",
}
BUILDING_REFUSAL = ("Building a file, code or chat challenge needs creator access for now. You"
                    " can build an exercise set, or start from an example: copy a challenge of an example course into yours.")
SOURCE = {"id": 5, "name": "Titanic", "slug": "titanic", "status": "started"}


def test_a_challenge_for_my_class_is_a_course_then_its_challenge():
    client, rec = make_client({
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (201, CREATED),
    })
    out = client.create_challenge_for_class("Titanic", "file_v1", end_date="2027-06-30")
    _expect(out == {"course": COURSE, "challenge": CREATED}, out)
    _expect(rec.signature() == [
        ("POST", "/academic_courses/"), ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    # The course is private by default (no visibility sent) and named after
    # the challenge.
    _expect(rec.calls[0]["json"] == {"name": "Titanic", "end_date": "2027-06-30"},
            rec.calls[0]["json"])
    _expect(rec.calls[1]["json"] == {"name": "Titanic", "kernel_version": "file_v1"},
            rec.calls[1]["json"])


def test_a_refused_challenge_leaves_the_empty_course_and_says_so():
    client, rec = make_client({
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (403, {"error": BUILDING_REFUSAL}),
    })
    try:
        client.create_challenge_for_class("Titanic", "flex_v1")
    except mlarena.PermissionDeniedError as err:
        _expect(BUILDING_REFUSAL in str(err), str(err))
        _expect("course 42 'Titanic' was created and holds no challenge" in str(err), str(err))
        _expect(err.status_code == 403 and err.body["course_id"] == 42, err.body)
    else:
        raise AssertionError("expected PermissionDeniedError")
    _expect(len(rec.calls) == 2, rec.calls)


def test_use_an_example_challenge_in_a_course_of_mine():
    detail = {**COURSE, "id": 10, "name": "M2 RL", "role": "owner"}
    client, rec = make_client({
        ("POST", "/teacher/course/10/challenges"): (201, {**CREATED, "course_id": 10}),
        ("GET", "/teacher/course/10"): (200, detail),
    })
    out = client.use_example_challenge(5, 10, module_id=3)
    _expect(out["course"] == detail and out["challenge"]["course_id"] == 10, out)
    _expect(rec.signature() == [
        ("POST", "/teacher/course/10/challenges"), ("GET", "/teacher/course/10"),
    ], rec.signature())
    _expect(rec.calls[0]["json"] == {"copy_from_challenge_id": 5, "module_id": 3},
            rec.calls[0]["json"])


def test_use_an_example_challenge_in_a_new_course_named_after_it():
    client, rec = make_client({
        ("GET", "/challenges/5"): (200, SOURCE),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (201, {**CREATED, "copied_from_challenge_id": 5}),
    })
    out = client.use_example_challenge(5)
    _expect(out["challenge"]["copied_from_challenge_id"] == 5, out)
    _expect(rec.signature() == [
        ("GET", "/challenges/5"), ("POST", "/academic_courses/"),
        ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    _expect(rec.calls[1]["json"] == {"name": "Titanic"}, rec.calls[1]["json"])
    _expect(rec.calls[2]["json"] == {"name": "Titanic", "copy_from_challenge_id": 5},
            rec.calls[2]["json"])


def test_a_module_needs_an_existing_course():
    client, rec = make_client({})
    try:
        client.use_example_challenge(5, module_id=3)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
    _expect(rec.calls == [], rec.calls)


def test_start_from_an_example_course_by_slug():
    copy = {**COURSE, "id": 43, "name": "Weather (copy)"}
    client, rec = make_client({
        ("GET", "/academic_courses/weather"): (200, {"id": 8, "slug": "weather"}),
        ("POST", "/teacher/course/8/duplicate"): (201, copy),
    })
    _expect(client.start_from_example_course("weather") == copy, "copy")
    _expect(rec.signature() == [
        ("GET", "/academic_courses/weather"), ("POST", "/teacher/course/8/duplicate"),
    ], rec.signature())
    _expect(rec.calls[1]["json"] == {}, rec.calls[1]["json"])


def test_start_from_an_example_course_by_id_with_a_name():
    client, rec = make_client({
        ("POST", "/teacher/course/8/duplicate"): (201, COURSE),
    })
    client.start_from_example_course(8, "Weather 2027")
    _expect(rec.signature() == [("POST", "/teacher/course/8/duplicate")], rec.signature())
    _expect(rec.calls[0]["json"] == {"name": "Weather 2027"}, rec.calls[0]["json"])


def test_a_course_that_is_not_an_example_is_refused():
    reason = "You may not copy this course"
    client, _ = make_client({("POST", "/teacher/course/8/duplicate"): (403, {"error": reason})})
    try:
        client.start_from_example_course(8)
    except MLArenaError as err:
        _expect(reason in str(err), str(err))
    else:
        raise AssertionError("a 403 must raise")


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
