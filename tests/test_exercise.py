"""Exercise challenge (`exercise_v1`) surface of the SDK (design of record:
the platform's `workers/exercise_v1/PROCESS.md`).

Covers, over a fake transport (the `requests.request` monkeypatch of
`test_chat.py`, so the chokepoint's defaults are part of what is asserted):

* the URL / method / body / params of `exercise_challenge`, `submit_exercise`,
  `exercise_attempt`, `exercise_attempts` (payload keys the backend's
  `extra="forbid"` `CreateExerciseAttemptRequest` accepts);
* `submit_exercise(wait=True)`: pending → running → passed / failed returned,
  `error` and the timeout raised with the attempt as `.body`; exactly one of
  code / answer, checked before any request;
* the 404 split — `ChallengeNotFoundError` on a challenge-scoped route,
  `ExerciseAttemptNotFoundError` on an attempt; 409 / 429 carry status and body;
* `check()`: the source it sends for a function, a nested function, a class and
  a str; the printed verdict; the refusals.
"""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena  # noqa: E402
from mlarena import client as client_module  # noqa: E402
from mlarena.exceptions import (  # noqa: E402
    ChallengeNotFoundError,
    ExerciseAttemptNotFoundError,
    MLArenaError,
    NotFoundError,
)
from mlarena.exercise import format_attempt, source_of  # noqa: E402


# --------------------------------------------------------------------------- #
# Fake transport (as in test_chat.py)
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = text

    def json(self):
        return self._json


class Recorder:
    def __init__(self, router=None):
        self.calls = []
        self.router = router

    def __call__(self, method, url, **kwargs):
        path = url.split("/api", 1)[1]
        self.calls.append({"method": method, "path": path, **kwargs})
        if self.router is not None:
            result = self.router(method, path, kwargs)
            if result is not None:
                return FakeResponse(*result)
        return FakeResponse(200, {"ok": True})

    @property
    def last(self):
        return self.calls[-1]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(client_module, "time", fake)
    return fake


def make_client(monkeypatch, router=None):
    rec = Recorder(router)
    monkeypatch.setattr(requests, "request", rec)
    c = mlarena.connect(api_key="mlk_user_abcd1234_deadbeef0000",
                        base_url="https://example.com")
    return c, rec


# --------------------------------------------------------------------------- #
# Response-shape builders (the backend's `backend/app/views/exercise/_schemas.py`)
# --------------------------------------------------------------------------- #


def _feedback(assertion=None, line=None, stdout=None, traceback=None, expected_hint=None):
    return {"assertion": assertion, "traceback": traceback, "stdout": stdout,
            "line": line, "expected_hint": expected_hint}


def _attempt(attempt_id=31, status="pending", error_type=None, message=None,
             feedback=None, duration_ms=None, code="def f():\n    return 1\n",
             answer=None, key="imc"):
    return {
        "id": attempt_id, "challenge_id": 42, "exercise_key": key, "user_id": 7,
        "status": status, "error_type": error_type, "message": message,
        "feedback": feedback, "duration_ms": duration_ms, "code": code,
        "answer": answer, "created_at_ts": "2026-11-04T10:00:00",
        "completed_at_ts": None if status in ("pending", "running") else "2026-11-04T10:00:01",
    }


def _poll_router(attempts):
    """POST → 202 pending; each GET of the attempt → the next of `attempts`."""
    it = iter(attempts)

    def router(method, path, kwargs):
        if method == "POST":
            return (202, _attempt(31, "pending"))
        return (200, next(it))
    return router


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


def test_exercise_challenge_route(monkeypatch):
    view = {"challenge_id": 42, "env_status": "ready", "exercises": []}
    c, rec = make_client(monkeypatch, lambda *_: (200, view))
    assert c.exercise_challenge(42) == view
    assert rec.last["method"] == "GET" and rec.last["path"] == "/exercise/challenge/42"
    assert rec.last["allow_redirects"] is False and rec.last["timeout"] == 30
    assert rec.last["headers"]["Authorization"].startswith("Bearer mlk_user_")


def test_submit_exercise_without_wait_returns_the_202_body(monkeypatch):
    c, rec = make_client(monkeypatch, lambda *_: (202, _attempt(31, "pending")))
    out = c.submit_exercise(42, "imc", code="def calculer_imc(p, t):\n    return 0\n", wait=False)
    assert out["status"] == "pending" and out["id"] == 31
    assert rec.last["method"] == "POST"
    assert rec.last["path"] == "/exercise/challenge/42/exercises/imc/attempts"
    # Both keys of CreateExerciseAttemptRequest, the unused one null.
    assert rec.last["json"] == {"code": "def calculer_imc(p, t):\n    return 0\n", "answer": None}
    assert rec.last["headers"]["Content-Type"] == "application/json"


def test_submit_exercise_sends_an_answer_for_a_text_exercise(monkeypatch):
    c, rec = make_client(monkeypatch, lambda *_: (202, _attempt(31, "pending", code=None, answer="2")))
    c.submit_exercise(42, "division-entiere", answer="2", wait=False)
    assert rec.last["json"] == {"code": None, "answer": "2"}
    assert rec.last["path"] == "/exercise/challenge/42/exercises/division-entiere/attempts"


@pytest.mark.parametrize("kwargs", [{}, {"code": "x = 1", "answer": "1"}])
def test_submit_exercise_takes_exactly_one_of_code_and_answer(monkeypatch, kwargs):
    c, rec = make_client(monkeypatch)
    with pytest.raises(MLArenaError, match="exactly one of code= .* or answer="):
        c.submit_exercise(42, "imc", wait=False, **kwargs)
    assert rec.calls == []


def test_exercise_attempt_route(monkeypatch):
    c, rec = make_client(monkeypatch, lambda *_: (200, _attempt(31, "running")))
    assert c.exercise_attempt(31)["status"] == "running"
    assert rec.last["method"] == "GET" and rec.last["path"] == "/exercise/attempts/31"


def test_exercise_attempts_sends_only_the_filters_passed(monkeypatch):
    c, rec = make_client(monkeypatch, lambda *_: (200, {"attempts": []}))
    assert c.exercise_attempts(42) == {"attempts": []}
    assert rec.last["path"] == "/exercise/challenge/42/attempts"
    assert rec.last["params"] is None
    c.exercise_attempts(42, user_id=7)
    assert rec.last["params"] == {"user_id": 7}
    c.exercise_attempts(42, user_id=7, key="imc")
    assert rec.last["params"] == {"user_id": 7, "key": "imc"}
    c.exercise_attempts(42, key="imc")
    assert rec.last["params"] == {"key": "imc"}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_a_404_on_a_challenge_route_is_a_challenge_not_found(monkeypatch):
    c, _ = make_client(monkeypatch, lambda *_: (404, {"error": "Unknown exercise"}))
    with pytest.raises(ChallengeNotFoundError, match="Unknown exercise"):
        c.submit_exercise(42, "nope", code="x = 1")
    with pytest.raises(ChallengeNotFoundError):
        c.exercise_challenge(42)


def test_a_404_on_an_attempt_is_an_exercise_attempt_not_found(monkeypatch):
    c, _ = make_client(monkeypatch, lambda *_: (404, {"error": "Attempt not found"}))
    with pytest.raises(ExerciseAttemptNotFoundError) as e:
        c.exercise_attempt(999)
    assert isinstance(e.value, NotFoundError) and e.value.status_code == 404


@pytest.mark.parametrize("status,reason", [
    (409, "An attempt of this exercise is still being checked."),
    (409, "This challenge is not started."),
    (429, "Too many attempts: 20 per minute."),
])
def test_a_refusal_carries_the_servers_reason_status_and_body(monkeypatch, status, reason):
    c, rec = make_client(monkeypatch, lambda *_: (status, {"error": reason}))
    with pytest.raises(MLArenaError) as e:
        c.submit_exercise(42, "imc", code="x = 1")
    assert not isinstance(e.value, NotFoundError)
    assert e.value.status_code == status and e.value.body == {"error": reason}
    assert str(e.value) == f"submit_exercise failed: {reason}"
    assert len(rec.calls) == 1   # a refused POST is never polled


# --------------------------------------------------------------------------- #
# submit_exercise(wait=True)
# --------------------------------------------------------------------------- #


def test_submit_exercise_waits_until_the_attempt_passes(monkeypatch, clock):
    passed = _attempt(31, "passed", feedback=_feedback(), duration_ms=8)
    c, rec = make_client(monkeypatch, _poll_router([
        _attempt(31, "pending"), _attempt(31, "running"), passed,
    ]))
    assert c.submit_exercise(42, "imc", code="x = 1") == passed
    assert [call["path"] for call in rec.calls] == [
        "/exercise/challenge/42/exercises/imc/attempts",
    ] + ["/exercise/attempts/31"] * 3
    assert clock.sleeps == [0.4, 0.4]   # the console's poll cadence


def test_submit_exercise_returns_a_failed_attempt(monkeypatch, clock):
    failed = _attempt(31, "failed", error_type="failed", message="assertion failed",
                      feedback=_feedback(assertion="70 kg, 1,75 m : l'IMC attendu est 22.9"))
    c, _ = make_client(monkeypatch, _poll_router([failed]))
    out = c.submit_exercise(42, "imc", code="x = 1", poll_interval=0.1)
    assert out["status"] == "failed" and out["error_type"] == "failed"
    assert clock.sleeps == []


def test_submit_exercise_raises_on_an_errored_attempt_with_the_attempt_as_body(monkeypatch, clock):
    errored = _attempt(31, "error", error_type="grader_error",
                       message="The checker failed — not your fault; the creator was told.")
    c, _ = make_client(monkeypatch, _poll_router([_attempt(31, "running"), errored]))
    with pytest.raises(MLArenaError, match=r"attempt 31 errored \(grader_error\): The checker failed") as e:
        c.submit_exercise(42, "imc", code="x = 1")
    assert e.value.body == errored and e.value.status_code is None


def test_submit_exercise_times_out_naming_the_state(monkeypatch, clock):
    c, _ = make_client(monkeypatch, lambda m, p, k: (202, _attempt(31, "pending"))
                       if m == "POST" else (200, _attempt(31, "pending")))
    with pytest.raises(MLArenaError, match=r"timed out after 1s: attempt 31 is still 'pending'") as e:
        c.submit_exercise(42, "imc", code="x = 1", timeout=1)
    assert e.value.body["status"] == "pending"


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #


def calculer_imc(poids_kg, taille_m):
    return round(poids_kg / taille_m ** 2, 1)


class Patient:
    def __init__(self, poids_kg, taille_m):
        self.poids_kg = poids_kg
        self.taille_m = taille_m

    def imc(self):
        return round(self.poids_kg / self.taille_m ** 2, 1)


def test_source_of_a_function_a_class_and_a_str():
    assert source_of(calculer_imc) == (
        "def calculer_imc(poids_kg, taille_m):\n"
        "    return round(poids_kg / taille_m ** 2, 1)\n"
    )
    assert source_of(Patient).startswith("class Patient:\n    def __init__")
    assert source_of("x = 1") == "x = 1"


def test_source_of_a_nested_function_is_dedented():
    def outer():
        def inner(x):
            return x + 1
        return inner

    source = source_of(outer())
    assert source == "def inner(x):\n    return x + 1\n"
    compile(source, "<check>", "exec")


def test_source_of_refuses_other_objects():
    with pytest.raises(TypeError, match="function, a class or a str of code, got int"):
        source_of(3)
    with pytest.raises(TypeError, match="got builtin_function_or_method"):
        source_of(len)


def test_source_of_a_function_without_source_fails_loud():
    namespace: dict = {}
    exec("def f():\n    return 1\n", namespace)
    with pytest.raises(MLArenaError, match="no source code for f .*pass the code as a str"):
        source_of(namespace["f"])


def test_check_sends_the_source_waits_and_prints_the_verdict(monkeypatch, clock, capsys):
    passed = _attempt(31, "passed", feedback=_feedback(), duration_ms=8)
    c, rec = make_client(monkeypatch, _poll_router([_attempt(31, "running"), passed]))
    assert c.check(42, "imc", calculer_imc) == passed
    post = rec.calls[0]
    assert post["method"] == "POST" and post["path"] == "/exercise/challenge/42/exercises/imc/attempts"
    assert post["json"] == {"code": source_of(calculer_imc), "answer": None}
    assert capsys.readouterr().out == "imc: passed (8 ms)\n"


def test_check_with_echo_false_prints_nothing(monkeypatch, clock, capsys):
    c, _ = make_client(monkeypatch, _poll_router([_attempt(31, "passed", duration_ms=3)]))
    c.check(42, "imc", "def calculer_imc(p, t):\n    return 1\n", echo=False)
    assert capsys.readouterr().out == ""


def test_check_propagates_a_refusal(monkeypatch, clock):
    c, _ = make_client(monkeypatch, lambda *_: (409, {"error": "This challenge is not started."}))
    with pytest.raises(MLArenaError) as e:
        c.check(42, "imc", calculer_imc)
    assert e.value.status_code == 409


def test_format_attempt_lines():
    assert format_attempt(_attempt(status="passed", duration_ms=12)) == "imc: passed (12 ms)"
    assert format_attempt(_attempt(status="passed", duration_ms=None)) == "imc: passed"
    failed = _attempt(status="failed", error_type="failed", message="assertion failed",
                      feedback=_feedback(assertion="expected 22.9", stdout="22.857\n"))
    assert format_attempt(failed) == (
        "imc: failed (failed)\n  expected 22.9\n  output:\n    22.857"
    )
    syntax = _attempt(status="failed", error_type="syntax_error",
                      message="invalid syntax", feedback=_feedback(line=2))
    assert format_attempt(syntax) == (
        "imc: failed (syntax_error)\n  invalid syntax\n  at line 2 of your code"
    )
    timeout = _attempt(status="failed", error_type="timeout",
                       message="Your code ran longer than 5 s.", feedback=None)
    assert format_attempt(timeout) == "imc: failed (timeout)\n  Your code ran longer than 5 s."
    hinted = _attempt(status="failed", error_type="failed", message=None,
                      feedback=_feedback(assertion="wrong", expected_hint="round to 1 decimal"))
    assert format_attempt(hinted) == "imc: failed (failed)\n  wrong\n  hint: round to 1 decimal"


def test_exercise_names_are_exported_at_package_level():
    assert mlarena.ExerciseAttemptNotFoundError is ExerciseAttemptNotFoundError
    assert issubclass(mlarena.ExerciseAttemptNotFoundError, mlarena.NotFoundError)
    assert mlarena.__version__ == "4.19.1"


def test_update_settings_sends_the_exercise_limits_passed(monkeypatch):
    c, rec = make_client(monkeypatch, lambda *_: (200, {"configuration": {}}))
    c.update_settings(42, time_limit_sec=3, max_attempts_per_minute=60)
    assert rec.last["method"] == "PUT"
    assert rec.last["path"] == "/creator_challenge/challenge/42/settings"
    assert rec.last["json"] == {"time_limit_sec": 3, "max_attempts_per_minute": 60}
    c.update_settings(42, memory_limit_mb=256, max_concurrent_attempts=4)
    assert rec.last["json"] == {"memory_limit_mb": 256, "max_concurrent_attempts": 4}


def test_exercise_admin_route(monkeypatch):
    view = {"challenge_id": 42, "env_status": "error", "env_status_message": "imc: solution failed"}
    c, rec = make_client(monkeypatch, lambda *_: (200, view))
    assert c.exercise_admin(42) == view
    assert rec.last["method"] == "GET" and rec.last["path"] == "/exercise/challenge/42/admin"


def test_exercise_admin_of_a_challenge_you_do_not_manage_is_a_challenge_not_found(monkeypatch):
    c, _ = make_client(monkeypatch, lambda *_: (404, {"error": "Not an exercise challenge"}))
    with pytest.raises(ChallengeNotFoundError):
        c.exercise_admin(42)


# --------------------------------------------------------------------------- #
# Exercises as data, the self-test before start, "Add exercises" (4.18.0)
# --------------------------------------------------------------------------- #

EXERCISES = [
    {"key": "bmi", "kind": "code", "title": "Body-mass index", "points": 1,
     "check": "assert student.function('bmi')(70, 1.75) == 22.9\n",
     "solution": "def bmi(w, h):\n    return round(w / h ** 2, 1)\n"},
    {"key": "floor-div", "kind": "text", "title": "What does 7 // 2 print?", "points": 1,
     "accept": ["3"]},
]


def _self_test(status, passed=None, results=None, message=None):
    return {"status": status, "self_test_requested_at_ts": "2026-11-21T10:00:00Z",
            "self_test_at_ts": "2026-11-21T10:00:40Z" if status == "done" else None,
            "passed": passed, "self_test_error": message, "self_test": results}


def test_exercise_definition_routes(monkeypatch):
    c, rec = make_client(monkeypatch)
    c.exercise_definition(7)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/exercise/challenge/7/definition")
    c.update_exercise_definition(7, EXERCISES)
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/exercise/challenge/7/definition")
    # The body is the exercises.json document itself.
    assert rec.last["json"] == {"version": 1, "exercises": EXERCISES}


def test_update_exercise_definition_refusals_carry_the_reason(monkeypatch):
    blocked = {"error": "Cannot modify the exercises of a started set: stop it first",
               "write_blocked_by": "started"}
    c, _ = make_client(monkeypatch, lambda m, p, k: (409, blocked))
    with pytest.raises(MLArenaError) as raised:
        c.update_exercise_definition(7, EXERCISES)
    assert raised.value.status_code == 409 and raised.value.body["write_blocked_by"] == "started"
    c, _ = make_client(monkeypatch, lambda m, p, k: (404, {"error": "Not an exercise challenge"}))
    with pytest.raises(ChallengeNotFoundError):
        c.exercise_definition(7)


def test_self_test_exercises_polls_until_done(monkeypatch, clock):
    states = iter([_self_test("requested"), _self_test("running"),
                   _self_test("done", True, [{"key": "bmi", "passed": True, "message": None}])])

    def router(method, path, kwargs):
        assert path == "/exercise/challenge/7/self-test"
        return (202 if method == "POST" else 200, next(states))

    c, rec = make_client(monkeypatch, router)
    done = c.self_test_exercises(7)
    assert done["passed"] is True and done["self_test"][0]["key"] == "bmi"
    assert [call["method"] for call in rec.calls] == ["POST", "GET", "GET"]
    assert clock.sleeps == [2, 2]


def test_self_test_exercises_without_wait_returns_the_202(monkeypatch, clock):
    c, rec = make_client(monkeypatch, lambda m, p, k: (202, _self_test("requested")))
    assert c.self_test_exercises(7, wait=False)["status"] == "requested"
    assert len(rec.calls) == 1 and clock.sleeps == []


def test_self_test_exercises_raises_on_timed_out_and_on_its_own_timeout(monkeypatch, clock):
    answers = {"POST": (202, _self_test("requested")), "GET": (200, _self_test("timed_out"))}
    c, _ = make_client(monkeypatch, lambda m, p, k: answers[m])
    with pytest.raises(MLArenaError, match="timed_out") as raised:
        c.self_test_exercises(7)
    assert raised.value.body["status"] == "timed_out"

    answers["GET"] = (200, _self_test("running"))
    with pytest.raises(MLArenaError, match="timed out after 10s") as raised:
        c.self_test_exercises(7, timeout=10)
    assert raised.value.body["status"] == "running"


def test_self_test_exercises_refused_on_a_started_set(monkeypatch, clock):
    c, _ = make_client(monkeypatch, lambda m, p, k: (409, {"error": "The set is started"}))
    with pytest.raises(MLArenaError) as raised:
        c.self_test_exercises(7)
    assert raised.value.status_code == 409 and clock.sleeps == []


def test_exercise_self_test_reads_the_state(monkeypatch):
    c, rec = make_client(monkeypatch, lambda m, p, k: (200, _self_test("done", False, None, "boom")))
    assert c.exercise_self_test(7)["self_test_error"] == "boom"
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/exercise/challenge/7/self-test")


def test_add_course_exercise_set_sends_only_what_was_named(monkeypatch):
    link = {"id": 4, "challenge_id": 91, "pass_threshold": 70.0}
    c, rec = make_client(monkeypatch, lambda m, p, k: (201, link))
    assert c.add_course_exercise_set(3, 12, "Week 1 exercises")["challenge_id"] == 91
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/teacher/course/3/modules/12/exercise-sets")
    # No pass_threshold key: the server applies the kind's default (70).
    assert rec.last["json"] == {"name": "Week 1 exercises"}
    c.add_course_exercise_set(3, 12, "Ungraded", pass_threshold=None, label="Practice")
    assert rec.last["json"] == {"name": "Ungraded", "label": "Practice", "pass_threshold": None}
    c.add_course_exercise_set(3, 12, "Half", pass_threshold=50)
    assert rec.last["json"] == {"name": "Half", "pass_threshold": 50}


def test_self_test_exercises_refusals_carry_the_cap_and_the_blockers(monkeypatch, clock):
    capped = {"error": "You already have 2 self-tests running (the limit per author): wait for one to finish.",
              "limit": 2}
    c, _ = make_client(monkeypatch, lambda m, p, k: (429, capped))
    with pytest.raises(MLArenaError) as raised:
        c.self_test_exercises(7)
    assert raised.value.status_code == 429 and raised.value.body["limit"] == 2
    blocked = {"error": "The set has no exercise yet: add one before starting", "details": ["…"],
               "start_blockers": [{"code": "env_invalid"}]}
    c, _ = make_client(monkeypatch, lambda m, p, k: (400, blocked))
    with pytest.raises(MLArenaError) as raised:
        c.self_test_exercises(7)
    assert raised.value.body["start_blockers"][0]["code"] == "env_invalid" and clock.sleeps == []


def test_exercise_list_reads_the_authors_listing(monkeypatch):
    listing = {"source": "file", "reason": None,
               "exercises": [{"key": "bmi", "title": "Body-mass index", "kind": "code", "points": 2}]}
    c, rec = make_client(monkeypatch, lambda m, p, k: (200, listing))
    assert c.exercise_list(7)["exercises"][0]["key"] == "bmi"
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/exercise/challenge/7/exercises")
    c, _ = make_client(monkeypatch, lambda m, p, k: (404, {"error": "Challenge not found or unauthorized"}))
    with pytest.raises(ChallengeNotFoundError):
        c.exercise_list(7)


def test_python_checks_without_creator_access_are_a_permission_error(monkeypatch, clock):
    from mlarena.exceptions import PermissionDeniedError

    body = {"error": "Exercises checked by Python code need creator access for now; "
                     "short-answer and multiple-choice exercises do not",
            "reason": "code_needs_creator_access"}
    c, rec = make_client(monkeypatch, lambda m, p, k: (403, body))
    for call in (lambda: c.update_exercise_definition(7, EXERCISES), lambda: c.self_test_exercises(7)):
        with pytest.raises(PermissionDeniedError, match="need creator access for now"):
            call()
    assert clock.sleeps == []


def test_the_self_test_cooldown_carries_the_seconds_left(monkeypatch, clock):
    body = {"error": "A self-test starts a checker: wait 42 s before asking for another "
                     "(one request per 60 s per author).", "retry_after_sec": 42}
    c, _ = make_client(monkeypatch, lambda m, p, k: (429, body))
    with pytest.raises(MLArenaError) as raised:
        c.self_test_exercises(7)
    assert raised.value.status_code == 429 and raised.value.body["retry_after_sec"] == 42
