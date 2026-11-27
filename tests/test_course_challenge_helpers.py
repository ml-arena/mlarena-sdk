"""SDK tests for the helpers of the New dialog and of the examples
(docs/course_challenges.md §7.2, §7.3): `create_challenge_for_class`,
`use_example_challenge`, `start_from_example_course`. Each is a client-side
composition of public routes — `POST /academic_courses/`,
`POST /teacher/course/{id}/challenges`, `POST /teacher/course/{id}/duplicate`
and the reads that name the course or the challenge. No route of their own.

Creating a challenge needs the terms of use accepted
(`backend/app/services/terms.py`): the two helpers that create a course
before the challenge read `GET /auth/current_user` first and refuse, with the
route's own error, before anything is created.

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
    "visibility": "private", "has_cover": False, "organization": None,
    "join_code": "JOIN1234", "start_date": None, "end_date": None,
    "archived_at_ts": None, "single_challenge_id": None, "example_listed_at_ts": None,
}
CREATED = {
    "message": "Challenge created", "challenge_id": 77, "name": "Titanic",
    "slug": "titanic-2", "course_id": 42, "module_id": 90, "module_created": True,
    "position": 0, "pass_threshold": None, "copied_from_challenge_id": None,
    "machine_name": "small",
}
KIND_NOT_SET_UP = "This kind is not set up on the platform yet."
SOURCE = {"id": 5, "name": "Titanic", "slug": "titanic", "status": "started"}

# `CurrentUserOut` (`backend/app/auth/routes/_schemas.py`), every key.
ME = {
    "is_authenticated": True, "id": 2, "username": "alice",
    "email": "alice@example.com", "avatar_key": None, "is_admin": False,
    "is_human_verified": True, "has_teacher_access": True, "can_create_course": True,
    "max_courses_limit": 10, "active_course_count": 1, "github_login": None,
    "terms_accepted_at_ts": "2026-11-26T10:15:00Z",
}
NOT_ACCEPTED = {**ME, "terms_accepted_at_ts": None}
CURRENT_USER = ("GET", "/auth/current_user")
# The 403 of every route that creates a challenge (`services/terms.py`).
TERMS_REQUIRED = (
    "Accept the terms of use before you create a challenge: once per account, in the "
    "console when you create your first challenge, or with POST /api/auth/terms/accept."
)
TERMS_REFUSAL = {"error": TERMS_REQUIRED, "reason": "terms_required"}


def _expect_terms_refusal(err):
    """The error of a `terms_required` 403, from the route or raised before
    it: same class, status, body; a message that says where the terms are
    and who accepts them, and how."""
    _expect(type(err) is mlarena.PermissionDeniedError, type(err))
    _expect(err.status_code == 403 and err.body == TERMS_REFUSAL, err.body)
    _expect(str(err).startswith(TERMS_REQUIRED), str(err))
    _expect("https://example.com/terms" in str(err), str(err))
    _expect("client.accept_terms()" in str(err), str(err))


def test_a_challenge_for_my_class_is_a_course_then_its_challenge():
    client, rec = make_client({
        CURRENT_USER: (200, ME),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (201, CREATED),
    })
    out = client.create_challenge_for_class("Titanic", "file_v1", end_date="2027-06-30", machine=3)
    _expect(out == {"course": COURSE, "challenge": CREATED}, out)
    # The terms are read first, before anything is created.
    _expect(rec.signature() == [
        CURRENT_USER,
        ("POST", "/academic_courses/"), ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    # The course is private by default (no visibility sent) and named after
    # the challenge.
    _expect(rec.calls[1]["json"] == {"name": "Titanic", "end_date": "2027-06-30"},
            rec.calls[1]["json"])
    _expect(rec.calls[2]["json"] == {"machine_id": 3, "name": "Titanic", "kernel_version": "file_v1"},
            rec.calls[2]["json"])


def test_a_challenge_for_my_class_without_the_terms_creates_no_course():
    """The challenge route would refuse (403 `terms_required`): the helper
    raises that error before the course, so no empty course is left."""
    client, rec = make_client({CURRENT_USER: (200, NOT_ACCEPTED)})
    for kind in ("file_v1", "flex_v1", "chat_v1", "exercise_v1"):
        try:
            client.create_challenge_for_class("Titanic", kind, machine=3)
        except mlarena.PermissionDeniedError as err:
            _expect_terms_refusal(err)
            _expect("course" not in err.body, err.body)
        else:
            raise AssertionError("expected PermissionDeniedError")
    # One read each, and no request that creates anything (the recorder
    # knows no other route: a POST would have raised KeyError).
    _expect(rec.signature() == [CURRENT_USER] * 4, rec.signature())


def test_the_helper_never_accepts_the_terms_itself():
    client, rec = make_client({
        CURRENT_USER: (200, NOT_ACCEPTED),
        ("POST", "/auth/terms/accept"): (200, {"terms_accepted_at_ts": "2026-11-26T10:15:00Z"}),
    })
    try:
        client.create_challenge_for_class("Titanic", "file_v1", machine=3)
    except mlarena.PermissionDeniedError:
        pass
    else:
        raise AssertionError("expected PermissionDeniedError")
    _expect(("POST", "/auth/terms/accept") not in rec.signature(), rec.signature())


def test_an_unknown_credential_is_left_to_the_first_route():
    """`current_user` answers 200 with every key null to a bearer it does not
    know: that is not "terms not accepted", and the create route says 401."""
    nobody = {key: None for key in ME} | {"is_authenticated": False}
    client, rec = make_client({
        CURRENT_USER: (200, nobody),
        ("POST", "/academic_courses/"): (401, {"error": "Authentication required"}),
    })
    try:
        client.create_challenge_for_class("Titanic", "file_v1", machine=3)
    except mlarena.PermissionDeniedError:
        raise AssertionError("a 401 is not a refusal for the terms")
    except mlarena.AuthenticationError as err:
        _expect(err.status_code == 401, err.status_code)
    else:
        raise AssertionError("expected AuthenticationError")
    _expect(rec.signature() == [CURRENT_USER, ("POST", "/academic_courses/")], rec.signature())


def test_a_refused_challenge_leaves_the_empty_course_and_says_so():
    """A refusal the terms do not explain (here: no machine runs the kind)
    comes after the course was created: it stays, and the error names it."""
    client, rec = make_client({
        CURRENT_USER: (200, ME),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (503, {"error": KIND_NOT_SET_UP}),
    })
    try:
        client.create_challenge_for_class("Titanic", "flex_v1", machine=3)
    except MLArenaError as err:
        _expect(KIND_NOT_SET_UP in str(err), str(err))
        _expect("course 42 'Titanic' was created and holds no challenge" in str(err), str(err))
        _expect(err.status_code == 503 and err.body["course_id"] == 42, err.body)
    else:
        raise AssertionError("expected MLArenaError")
    _expect(len(rec.calls) == 3, rec.calls)


def test_the_routes_own_terms_refusal_still_names_the_course_it_left():
    """The route is the rule: if it refuses for the terms after the read
    said they were accepted, the error is the route's and names the course
    that was created."""
    client, _ = make_client({
        CURRENT_USER: (200, ME),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (403, TERMS_REFUSAL),
    })
    try:
        client.create_challenge_for_class("Titanic", "file_v1", machine=3)
    except mlarena.PermissionDeniedError as err:
        _expect(err.body == {**TERMS_REFUSAL, "course_id": 42}, err.body)
        _expect("client.accept_terms()" in str(err), str(err))
        _expect("course 42 'Titanic' was created" in str(err), str(err))
    else:
        raise AssertionError("expected PermissionDeniedError")


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


def test_an_example_challenge_into_a_course_of_mine_without_the_terms():
    """One route, which creates the challenge or nothing: its refusal is the
    error, the course already existed and nothing is read after it."""
    client, rec = make_client({("POST", "/teacher/course/10/challenges"): (403, TERMS_REFUSAL)})
    try:
        client.use_example_challenge(5, 10)
    except mlarena.PermissionDeniedError as err:
        _expect_terms_refusal(err)
    else:
        raise AssertionError("expected PermissionDeniedError")
    _expect(rec.signature() == [("POST", "/teacher/course/10/challenges")], rec.signature())


def test_use_an_example_challenge_in_a_new_course_named_after_it():
    client, rec = make_client({
        ("GET", "/challenges/5"): (200, SOURCE),
        CURRENT_USER: (200, ME),
        ("POST", "/academic_courses/"): (201, COURSE),
        ("POST", "/teacher/course/42/challenges"): (201, {**CREATED, "copied_from_challenge_id": 5}),
    })
    out = client.use_example_challenge(5)
    _expect(out["challenge"]["copied_from_challenge_id"] == 5, out)
    _expect(rec.signature() == [
        ("GET", "/challenges/5"), CURRENT_USER, ("POST", "/academic_courses/"),
        ("POST", "/teacher/course/42/challenges"),
    ], rec.signature())
    _expect(rec.calls[2]["json"] == {"name": "Titanic"}, rec.calls[2]["json"])
    _expect(rec.calls[3]["json"] == {"name": "Titanic", "copy_from_challenge_id": 5},
            rec.calls[3]["json"])


def test_an_example_challenge_in_a_new_course_without_the_terms_creates_no_course():
    client, rec = make_client({
        ("GET", "/challenges/5"): (200, SOURCE),
        CURRENT_USER: (200, NOT_ACCEPTED),
    })
    for call in (lambda: client.use_example_challenge(5),
                 lambda: client.use_example_challenge(5, name="My Titanic")):
        try:
            call()
        except mlarena.PermissionDeniedError as err:
            _expect_terms_refusal(err)
        else:
            raise AssertionError("expected PermissionDeniedError")
    # Reads only: the challenge's name (first call), the caller's terms.
    _expect(rec.signature() == [("GET", "/challenges/5"), CURRENT_USER, CURRENT_USER],
            rec.signature())


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


def test_start_from_an_example_course_without_the_terms_is_the_routes_refusal():
    """`duplicate_course` is one route that copies everything or nothing
    (`backend/tests/test_terms_at_first_creation.py`: no course is left), and
    only it knows whether the course holds a challenge to copy (a course of
    lessons needs no terms): the helper sends it and raises its refusal."""
    client, rec = make_client({
        ("GET", "/academic_courses/weather"): (200, {"id": 8, "slug": "weather"}),
        ("POST", "/teacher/course/8/duplicate"): (403, TERMS_REFUSAL),
    })
    try:
        client.start_from_example_course("weather")
    except mlarena.PermissionDeniedError as err:
        _expect_terms_refusal(err)
    else:
        raise AssertionError("expected PermissionDeniedError")
    # No course was created by another request before the refused one.
    _expect(rec.signature() == [
        ("GET", "/academic_courses/weather"), ("POST", "/teacher/course/8/duplicate"),
    ], rec.signature())


def test_a_course_of_lessons_is_started_from_without_reading_the_terms():
    client, rec = make_client({("POST", "/teacher/course/8/duplicate"): (201, COURSE)})
    client.start_from_example_course(8)
    _expect(CURRENT_USER not in rec.signature(), rec.signature())


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
