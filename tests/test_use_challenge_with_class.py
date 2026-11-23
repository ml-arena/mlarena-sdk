"""SDK tests for `use_challenge_with_class`: the console's "Use with my class"
(Studio › New › "Run a challenge with my class") as a client-side composition
of the public routes — `POST /academic_courses/` and
`POST /teacher/course/{id}/challenges`, plus the reads that name the course
and the challenge. No route of its own.

Offline harness: the client's single HTTP chokepoint (`MLArenaClient._request`)
is replaced by a recorder. Runs under pytest *or* directly:
``python tests/test_use_challenge_with_class.py``.
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


def make_client(routes, scope="user"):
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
    "archived_at_ts": None, "enrollment_state": "open", "is_enrolled": False,
}
ENTRY = {"module_id": 90, "title": "Titanic", "challenges": []}
CHALLENGE = {"id": 5, "name": "Titanic", "slug": "titanic", "status": "started"}


def test_a_new_course_named_after_the_challenge():
    client, rec = make_client({
        ("GET", "/challenges/5"): (200, CHALLENGE),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (201, ENTRY),
    })
    out = client.use_challenge_with_class(5)
    _expect(out == COURSE, out)
    _expect(out["join_code"] == "JOIN1234", out)
    _expect(rec.signature() == [
        ("GET", "/challenges/5"),
        ("POST", "/academic_courses/"),
        ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    _expect(rec.calls[1]["json"] == {"name": "Titanic"}, rec.calls[1]["json"])
    _expect(rec.calls[2]["json"] == {"challenge_id": 5, "is_published": True}, rec.calls[2]["json"])


def test_a_named_new_course_with_an_end_date_reads_no_challenge():
    client, rec = make_client({
        ("POST", "/academic_courses/"): (201, {**COURSE, "name": "Spring", "end_date": "2027-06-30"}),
        ("POST", "/teacher/course/42/challenges"): (201, ENTRY),
    })
    out = client.use_challenge_with_class(5, name="Spring", end_date="2027-06-30")
    _expect(out["name"] == "Spring", out)
    _expect(rec.signature() == [
        ("POST", "/academic_courses/"),
        ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    _expect(rec.calls[0]["json"] == {"name": "Spring", "end_date": "2027-06-30"}, rec.calls[0]["json"])


def test_a_slug_is_resolved_once():
    client, rec = make_client({
        ("GET", "/challenges/slug/titanic"): (200, CHALLENGE),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (201, ENTRY),
    })
    client.use_challenge_with_class("titanic")
    _expect(rec.signature() == [
        ("GET", "/challenges/slug/titanic"),
        ("POST", "/academic_courses/"),
        ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    _expect(rec.calls[2]["json"]["challenge_id"] == 5, rec.calls[2]["json"])


def test_an_existing_course_is_read_back_with_its_join_code():
    detail = {**COURSE, "id": 10, "name": "M2 RL", "join_code": "M2RL0001", "role": "owner"}
    client, rec = make_client({
        ("POST", "/teacher/course/10/challenges"): (201, ENTRY),
        ("GET", "/teacher/course/10"): (200, detail),
    })
    out = client.use_challenge_with_class(5, course=10)
    _expect(out == detail, out)
    _expect(rec.signature() == [
        ("POST", "/teacher/course/10/challenges"),
        ("GET", "/teacher/course/10"),
    ], rec.signature())


def test_a_new_course_s_fields_are_refused_with_an_existing_course():
    client, rec = make_client({})
    for kwargs in ({"name": "Spring"}, {"end_date": "2027-06-30"}):
        try:
            client.use_challenge_with_class(5, course=10, **kwargs)
        except TypeError as e:
            _expect("not both" in str(e), str(e))
        else:
            raise AssertionError("expected TypeError")
    _expect(rec.calls == [], rec.calls)


def test_a_refused_challenge_raises_after_the_course_was_created():
    client, rec = make_client({
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (400, {"error": "challenge_id 5 does not match any challenge"}),
    })
    try:
        client.use_challenge_with_class(5, name="Titanic")
    except MLArenaError as e:
        _expect("does not match any challenge" in str(e), str(e))
        _expect(e.status_code == 400, e.status_code)
    else:
        raise AssertionError("expected MLArenaError")
    _expect(len(rec.calls) == 2, rec.calls)


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


def test_a_course_that_already_holds_the_challenge_is_returned_unchanged():
    """`use_challenge_with_class(challenge, course)` twice: the second add is
    the server's 409 naming the existing entry, and the call answers the
    course as the first one did."""
    client, rec = make_client({
        ("POST", "/teacher/course/42/challenges"): (
            409, {"error": "This course already has the challenge 'Titanic'", "module_id": 90}),
        ("GET", "/teacher/course/42"): (200, COURSE),
    })
    out = client.use_challenge_with_class(5, 42)
    _expect(out == COURSE, out)
    _expect(rec.signature() == [
        ("POST", "/teacher/course/42/challenges"), ("GET", "/teacher/course/42"),
    ], rec.signature())


def test_another_refusal_of_the_add_is_raised():
    client, _rec = make_client({
        ("POST", "/teacher/course/42/challenges"): (409, {"error": "Something else"}),
    })
    try:
        client.use_challenge_with_class(5, 42)
    except MLArenaError as err:
        _expect(err.status_code == 409, err)
    else:
        raise AssertionError("a 409 that names no entry must be raised")
