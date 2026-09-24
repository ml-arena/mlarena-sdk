"""SDK 2.0 wire + Python-compat tests.

Covers what the challenge/submission rename changed:

* every participant method hits `/api/submissions/...` with `submission_*`
  payload keys, and the creator/public methods hit `/api/challenges`,
  `/api/creator_challenge`, `/api/challenge_tags`, `/api/leaderboard/challenge`;
* `submit()` returns exactly `{"submission_id", "deploy"}`;
* the old Python names (methods and keyword arguments) still work, warn, and
  send only the new wire.

Same fake transport idea as `test_course_content.py`: the client's single HTTP
chokepoint (`MLArenaClient._request`) is replaced by a recorder.
"""
import os
import re
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlarena
from mlarena import client as client_module
from mlarena.client import MLArenaClient
from mlarena.exceptions import (
    AuthenticationError,
    ChallengeNotFoundError,
    CompetitionNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    SubmissionError,
    SubmissionNotFoundError,
)

SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = ""
        self.content = content

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        """Streamed downloads (`download_submission_file`) read the body this
        way; one chunk is enough to prove the bytes land unmodified."""
        yield self.content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status {self.status_code}")


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


def make_client(router=None, scope="user"):
    c = mlarena.connect(
        api_key=f"mlk_{scope}_abcd1234_deadbeef0000",
        base_url="https://example.com",
    )
    rec = Recorder(router)
    c._request = rec
    return c, rec


def _tmp_file(d, name, content=b"x = 1\n"):
    path = os.path.join(d, name)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


# --------------------------------------------------------------------------- #
# Submissions — routes and payload keys
# --------------------------------------------------------------------------- #


def test_create_submission_route_and_body():
    c, rec = make_client(lambda *_: (201, {"submission_id": 11}))
    out = c.create_submission(4, "my-sub")
    assert out == {"submission_id": 11}
    assert rec.last["method"] == "POST"
    assert rec.last["path"] == "/submissions/challenge/4"
    assert rec.last["json"] == {"submission_name": "my-sub"}

    c.create_submission(4, "copy", copy_from_submission_id=9)
    assert rec.last["json"] == {"submission_name": "copy",
                                "copy_from_submission_id": 9}


def test_create_submission_error_is_submission_error():
    c, _ = make_client(lambda *_: (400, {"error": "bad"}))
    try:
        c.create_submission(4, "x")
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "create_submission failed: bad" in str(exc)


def test_submission_file_routes():
    c, rec = make_client(lambda m, p, k: (200, {"content": "src"})
                         if p.endswith("/content") else (200, {"status": "ok"}))
    with tempfile.TemporaryDirectory() as d:
        c.upload_submission_file(4, 11, _tmp_file(d, "agent.py"))
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/submissions/challenge/4/11/file")
    assert rec.last["files"]["file"][0] == "agent.py"

    c.list_submission_files(4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/submissions/challenge/4/11/file")

    assert c.get_submission_file_content(4, 11, "agent.py") == "src"
    assert rec.last["path"] == "/submissions/challenge/4/11/file/agent.py/content"

    c.update_submission_file_content(4, 11, "agent.py", "print(1)")
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/submissions/challenge/4/11/file")
    assert rec.last["files"]["file"][0] == "agent.py"

    c.delete_submission_file(4, 11, "old.py")
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/submissions/challenge/4/11/file")
    assert rec.last["data"] == {"delete_file": "old.py"}


def test_submission_deploy_status_delete_games_routes():
    c, rec = make_client()
    c.deploy_submission(4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/submissions/challenge/4/11/deploy")
    c.submission_deploy_status(4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/submissions/challenge/4/11/deploy")
    c.submission_status(4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/submissions/challenge/4/11/status")
    c.submission_games(11)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/submissions/submission/11/games")
    c.delete_submission(4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("DELETE", "/submissions/challenge/4/11")


def test_runtime_routes_keep_agent_runtime_segment():
    c, rec = make_client(lambda m, p, k: (200, [{"id": 3, "language": "python"}])
                         if "runtime_options" in p else (200, {"id": 3}))
    c.runtime_options(4)
    assert rec.last["path"] == "/submissions/runtime_options/4"
    c.agent_runtime(11)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/submissions/agent_runtime/11")
    c.set_agent_runtime(11, 3)
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/submissions/agent_runtime/11")
    assert rec.last["json"] == {"docker_image_agent_runtime_id": 3}


def test_agent_runtime_is_none_for_a_submission_without_agent_container():
    # A file_v1 / chat_v1 submission pins no runtime: the route answers null.
    c, _rec = make_client()
    null_body = FakeResponse(200)
    null_body._json = None  # FakeResponse turns a None body into {}
    c._request = lambda method, url, **kwargs: null_body
    assert c.agent_runtime(11) is None


def test_submission_file_download_and_docs_routes():
    """Parity with the console: the download button, the Documentation panel
    and the "copy an existing submission" picker."""
    c, rec = make_client(lambda m, p, k: (200, None, b"\x80weights")
                         if p.endswith("/file/model.pt") else None)
    with tempfile.TemporaryDirectory() as d:
        out = c.download_submission_file(4, 11, "model.pt", dest_dir=d)
        assert (rec.last["method"], rec.last["path"]) == (
            "GET", "/submissions/challenge/4/11/file/model.pt")
        # Bytes, not decoded text: this is how weights come back out.
        with open(out, "rb") as fh:
            assert fh.read() == b"\x80weights"

        c.upload_submission_docs(4, 11, _tmp_file(d, "README.md", b"# hi\n"))
    assert (rec.last["method"], rec.last["path"]) == (
        "PUT", "/submissions/challenge/4/11/docs")
    assert rec.last["files"]["file"][0] == "README.md"

    c.delete_submission_docs(4, 11, "README.md")
    assert (rec.last["method"], rec.last["path"]) == (
        "DELETE", "/submissions/challenge/4/11/docs")
    assert rec.last["params"] == {"filename": "README.md"}

    c.copyable_submissions()
    assert (rec.last["method"], rec.last["path"]) == (
        "GET", "/submissions/copyable_submissions")


def test_submission_docs_upload_needs_a_real_file():
    c, _ = make_client()
    try:
        c.upload_submission_docs(4, 11, "/nope/README.md")
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "File not found" in str(exc)


def test_new_submission_methods_map_errors_like_their_neighbours():
    c, _ = make_client(lambda *_: (500, {"error": "boom"}))
    for label, call in (
        ("download_submission_file",
         lambda: c.download_submission_file(4, 11, "model.pt", dest_dir=tempfile.mkdtemp())),
        ("delete_submission_docs", lambda: c.delete_submission_docs(4, 11, "a.md")),
        ("copyable_submissions", lambda: c.copyable_submissions()),
    ):
        try:
            call()
            raise AssertionError(f"expected SubmissionError from {label}")
        except SubmissionError as exc:
            assert f"{label} failed: boom" in str(exc)


# --------------------------------------------------------------------------- #
# tail_logs()
# --------------------------------------------------------------------------- #


class FakeClock:
    """Stands in for the module's `time`: no real waiting, and a monotonic
    clock that only advances when the generator sleeps."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def with_fake_clock(fn):
    clock = FakeClock()
    real = client_module.time
    client_module.time = clock
    try:
        return fn()
    finally:
        client_module.time = real


def test_tail_logs_stops_on_is_settled():
    c, rec = make_client(lambda *_: (200, {"status": "active",
                                           "last_status_message": "done",
                                           "is_settled": True}))
    lines = list(c.tail_logs(4, 11))
    assert lines == ["[active] done"]
    assert rec.last["path"] == "/submissions/challenge/4/11/status"


def test_tail_logs_returns_on_a_never_deployed_submission():
    """`created` is settled — nothing is running on the row. Before the server
    answered that, the client's own terminal list did not contain `created`
    and the loop never ended."""
    c, rec = make_client(lambda *_: (200, {"status": "created",
                                           "last_status_message": None,
                                           "is_settled": True}))
    assert with_fake_clock(lambda: list(c.tail_logs(4, 11))) == ["[created]"]
    assert len(rec.calls) == 1


def test_tail_logs_does_not_repeat_unchanged_run_lines():
    state = {"polls": 0}

    def router(method, path, kwargs):
        state["polls"] += 1
        if state["polls"] <= 3:
            return (200, {
                "status": "deploy_run",
                "last_status_message": "running",
                "is_settled": False,
                "run_info": {"results": [
                    {"job_status": "running", "env_error_type": None,
                     "submission_results": [
                         {"submission_id": 11, "agent_nb_steps": 5,
                          "submission_reward": 1.0, "game_outcome": None,
                          "agent_error_type": None},
                     ]},
                ]},
            })
        return (200, {"status": "active", "last_status_message": "done",
                      "is_settled": True, "run_info": {"results": []}})

    c, _ = make_client(router)
    lines = with_fake_clock(lambda: list(c.tail_logs(4, 11)))

    assert lines == [
        "[deploy_run] running",
        "  run: job_status=running steps=5 reward=1.0 outcome=None",
        "[active] done",
    ], lines


def test_tail_logs_emits_a_run_line_again_when_it_changes():
    """A run is the one `RunResult` model: `job_status` and `env_error_type`
    are the run's, the caller's steps / reward / errors are on *their own*
    `submission_results` row — the one whose `submission_id` is theirs, not
    the first one (here an opponent's, whose failure must not be printed).
    The env's message is the creator's (null on a participant read), so the
    env line has no message."""
    state = {"polls": 0}

    def router(method, path, kwargs):
        state["polls"] += 1
        if state["polls"] == 1:
            steps, job_status, agent_error, env_error = 5, "running", None, None
        elif state["polls"] == 2:
            steps, job_status, agent_error, env_error = 9, "failed", "code_error", None
        elif state["polls"] == 3:
            steps, job_status, agent_error, env_error = 9, "failed", "code_error", "simulation_error"
        else:
            return (200, {"status": "deploy_failed", "last_status_message": "crash",
                          "is_settled": True, "run_info": {"results": []}})
        return (200, {
            "status": "deploy_run",
            "last_status_message": "running",
            "is_settled": False,
            "run_info": {"results": [
                {"job_status": job_status, "env_error_type": env_error,
                 "env_error_message": None,
                 "submission_results": [
                     {"submission_id": 12, "agent_nb_steps": 3,
                      "submission_reward": 0.0, "game_outcome": "loser",
                      "agent_error_type": "pod_crash",
                      "agent_error_message": None},
                     {"submission_id": 11, "agent_nb_steps": steps,
                      "submission_reward": 1.0, "game_outcome": None,
                      "agent_error_type": agent_error,
                      "agent_error_message": "boom" if agent_error else None},
                 ]},
            ]},
        })

    c, _ = make_client(router)
    lines = with_fake_clock(lambda: list(c.tail_logs(4, 11)))

    assert lines == [
        "[deploy_run] running",
        "  run: job_status=running steps=5 reward=1.0 outcome=None",
        "  run: job_status=failed steps=9 reward=1.0 outcome=None",
        "    agent error[code_error]: boom",
        "  run: job_status=failed steps=9 reward=1.0 outcome=None",
        "    agent error[code_error]: boom",
        "    env error[simulation_error]",
        "[deploy_failed] crash",
    ], lines


def test_tail_logs_fails_loudly_on_another_run_shape():
    """The 2.x flat run (`error_type` next to `agent_nb_steps`) is gone from
    the server. A backend still serving it — or any run without
    `submission_results` — must raise here, not print `steps=None` forever."""
    c, _ = make_client(lambda *_: (200, {
        "status": "deploy_run", "last_status_message": "running",
        "is_settled": False,
        "run_info": {"results": [{"job_status": "running", "agent_nb_steps": 5,
                                  "error_type": None}]},
    }))
    try:
        with_fake_clock(lambda: list(c.tail_logs(4, 11)))
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert exc.args == ("submission_results",)


def test_tail_logs_raises_on_timeout_instead_of_returning_silently():
    c, _ = make_client(lambda *_: (200, {"status": "deploy_run",
                                         "last_status_message": "running",
                                         "is_settled": False,
                                         "run_info": {"results": []}}))
    try:
        with_fake_clock(lambda: list(
            c.tail_logs(4, 11, poll_sec=5.0, timeout_sec=10.0)))
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "timed out after 10.0s" in str(exc)
        assert "still 'deploy_run'" in str(exc)


# --------------------------------------------------------------------------- #
# submit() / status()
# --------------------------------------------------------------------------- #


def _submit_router(is_deployable=True):
    """Fake backend for submit(): the status route answers with the block."""
    def router(method, path, kwargs):
        if method == "POST" and path == "/submissions/challenge/4":
            return (201, {"submission_id": 11, "status": "created"})
        if path.startswith("/submissions/runtime_options/"):
            return (200, [{"id": 7, "language": "python", "framework": "torch"}])
        if path.endswith("/file"):
            return (200, {"message": "File operation successful",
                          "validation_message": "bad name"})
        if path.endswith("/status"):
            return (200, {
                "status": "upload_validated" if is_deployable else "upload_failed",
                "phase": "upload",
                "last_status_message": "bad name",
                "status_update_ts": None,
                "is_uploadable": True,
                "is_deployable": is_deployable,
                "is_settled": True,
            })
        if path.endswith("/deploy"):
            # 202, as the backend answers an accepted deploy.
            return (202, {"message": "queued"})
        return (200, {"status": "deploy_queue"})
    return router


_DEPLOY_409 = {
    "error": "Daily deployment limit (5) reached",
    "deployment_limits": {
        "daily_deploy_limit": 5, "daily_deploys_used": 5,
        "daily_deploys_remaining": 0,
        "next_deploy_available_at": "2026-09-23T00:00:00+00:00",
        "can_deploy": False,
    },
    "active_submission_limits": {
        "max_active_submissions": 3, "active_submissions_count": 1,
        "active_submissions_remaining": 2,
    },
}


def _submit_router_with(deploy=None, after_deploy=None):
    """`_submit_router` whose deploy route answers `deploy` and whose status
    route, once the deploy was called, answers `after_deploy`."""
    base = _submit_router()
    state = {"deployed": False}

    def router(method, path, kwargs):
        if path.endswith("/deploy"):
            state["deployed"] = True
            return deploy if deploy is not None else base(method, path, kwargs)
        if path.endswith("/status") and state["deployed"] and after_deploy is not None:
            return (200, after_deploy)
        return base(method, path, kwargs)
    return router


class MyAgent:
    def act(self, observation):
        return 0


def test_submit_with_agent_class_returns_submission_id_and_deploy():
    c, rec = make_client(_submit_router())
    out = c.submit(4, agent=MyAgent, runtime={"framework": "torch"})
    assert out == {"submission_id": 11, "deploy": {"message": "queued"}}

    seq = [(call["method"], call["path"]) for call in rec.calls]
    assert seq == [
        ("POST", "/submissions/challenge/4"),
        ("GET", "/submissions/runtime_options/4"),
        ("PUT", "/submissions/agent_runtime/11"),
        ("PUT", "/submissions/challenge/4/11/file"),
        ("GET", "/submissions/challenge/4/11/status"),
        ("PUT", "/submissions/challenge/4/11/deploy"),
    ], seq
    assert rec.calls[0]["json"] == {"submission_name": "MyAgent"}
    assert rec.calls[3]["files"]["file"][0] == "agent.py"

    # status() defaults to the last submission.
    c.status()
    assert rec.last["path"] == "/submissions/challenge/4/11/status"


def test_submit_with_files_uses_submission_name():
    c, rec = make_client(_submit_router())
    with tempfile.TemporaryDirectory() as d:
        f1 = _tmp_file(d, "submission.csv", b"id,y\n1,0\n")
        out = c.submit(4, files=[f1])
        assert rec.calls[0]["json"] == {"submission_name": "submission"}
        c.submit(4, files=[f1], submission_name="named")
        # create, upload, status, deploy — the create body is four calls back.
        assert rec.calls[-4]["json"] == {"submission_name": "named"}
    assert set(out) == {"submission_id", "deploy"}


def test_submit_not_deployable_raises_before_deploy():
    c, rec = make_client(_submit_router(is_deployable=False))
    try:
        c.submit(4, agent=MyAgent)
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "Submission 11 did not pass upload validation: bad name" in str(exc)
        assert "delete_submission(4, 11)" in str(exc)
        # Not an HTTP refusal: the status payload rides on `.body` instead.
        assert exc.status_code is None
        assert exc.body["is_deployable"] is False
    assert not any(call["path"].endswith("/deploy") for call in rec.calls)

    # The submission exists on the server: `status()` must find it. It used
    # to be remembered only after the deploy, so this raised "no previous
    # submission found" right after telling the caller which id to delete.
    assert (c._last_submission_id, c._last_challenge) == (11, 4)
    c.status()
    assert rec.last["path"] == "/submissions/challenge/4/11/status"


def test_submit_propagates_a_refused_deploy_with_the_servers_limits():
    """A 409 on the deploy is the `SubmissionError` `deploy_submission()`
    raises; the quota blocks the server sent are on `.body`."""
    c, rec = make_client(_submit_router_with(deploy=(409, _DEPLOY_409)))
    try:
        c.submit(4, agent=MyAgent)
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert str(exc) == "deploy_submission failed: Daily deployment limit (5) reached"
        assert exc.status_code == 409
        assert exc.body == _DEPLOY_409
        assert exc.body["deployment_limits"]["daily_deploys_remaining"] == 0
        assert exc.body["active_submission_limits"]["active_submissions_remaining"] == 2
    assert (c._last_submission_id, c._last_challenge) == (11, 4)


def _deploy_failed_status(failure_message, last_status_message="deploy failed"):
    return {
        "status": "deploy_failed", "phase": "deployment",
        "last_status_message": last_status_message, "status_update_ts": None,
        "is_uploadable": True, "is_deployable": True, "is_settled": True,
        "latest_deploy": {"id": 9, "created_at_ts": None, "status": "failed",
                          "finished_at_ts": None,
                          "failure_message": failure_message},
        "run_info": {"results": []}, "queue_info": {},
    }


def test_submit_wait_raises_when_the_deploy_fails():
    """`wait=True` used to return `{"status": <deploy_failed block>}` as if
    the deploy had succeeded, while the upload half of the same call raised.
    The attempt's own reason is the message; the final payload is `.body`."""
    final = _deploy_failed_status("Traceback: ModuleNotFoundError: torch")
    c, rec = make_client(_submit_router_with(after_deploy=final))
    try:
        with_fake_clock(lambda: c.submit(4, agent=MyAgent, wait=True))
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert str(exc) == "Traceback: ModuleNotFoundError: torch"
        assert exc.status_code is None
        assert exc.body == final
    assert rec.last["path"] == "/submissions/challenge/4/11/status"


def test_submit_wait_falls_back_to_the_status_message_without_a_failure_message():
    final = _deploy_failed_status(None, last_status_message="engine gone")
    c, _ = make_client(_submit_router_with(after_deploy=final))
    try:
        with_fake_clock(lambda: c.submit(4, agent=MyAgent, wait=True))
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert str(exc) == "engine gone"
        assert exc.body["status"] == "deploy_failed"


def test_submit_without_wait_returns_as_soon_as_the_deploy_is_accepted():
    """Default behaviour is unchanged: no extra status poll after the deploy."""
    c, rec = make_client(_submit_router())
    out = c.submit(4, agent=MyAgent)
    assert set(out) == {"submission_id", "deploy"}
    assert rec.calls[-1]["path"] == "/submissions/challenge/4/11/deploy"


def test_submit_wait_polls_until_settled_and_returns_the_status():
    c, rec = make_client(_submit_router())
    out = with_fake_clock(lambda: c.submit(4, agent=MyAgent, wait=True))

    assert set(out) == {"submission_id", "deploy", "status"}
    assert out["status"]["is_settled"] is True
    # The wait is a client-side composition of the public routes — no new
    # endpoint: after the deploy it only polls /status again.
    paths = [call["path"] for call in rec.calls]
    after_deploy = paths[paths.index("/submissions/challenge/4/11/deploy") + 1:]
    assert after_deploy == ["/submissions/challenge/4/11/status"] * 2, after_deploy


def test_status_requires_ids():
    c, _ = make_client()
    try:
        c.status()
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "submission_id and challenge_id are required" in str(exc)
    c.status(submission_id=5, challenge_id=2)


# --------------------------------------------------------------------------- #
# Challenges — routes and payload keys
# --------------------------------------------------------------------------- #


def test_challenge_read_routes():
    def router(m, p, k):
        if p == "/challenges/":
            return (200, {"items": [], "metadata": {}})
        if p.startswith("/leaderboard/"):
            return (200, {**_ENVELOPE, "leaders": []})
        return (200, {"datasets": []})
    c, rec = make_client(router)
    c.challenges()
    assert rec.last["path"] == "/challenges/"
    c.challenges(page=1)
    assert rec.last["path"] == "/challenges/"
    # The list route surfaces the caller's enrolled-course challenges on
    # page 1 and, for admins, unstarted ones: the token must travel.
    assert rec.last["headers"]["Authorization"].startswith("Bearer ")
    c.challenge(4)
    assert rec.last["path"] == "/challenges/4"
    # Public route, but a non-public course challenge is only visible to the
    # caller the backend can identify: the token must travel.
    assert rec.last["headers"]["Authorization"].startswith("Bearer ")
    c.datasets(4)
    assert rec.last["path"] == "/challenges/4/datasets"
    c.recent_replays(4)
    assert rec.last["path"] == "/challenges/4/recent-replays"
    c.leaderboard(4)
    assert rec.last["path"] == "/leaderboard/challenge/4"
    # Same: `is_my_submission` is computed from the caller the token names. It
    # was always False over the SDK because no token was sent.
    assert rec.last["headers"]["Authorization"].startswith("Bearer ")


_CHALLENGE_BLOCK = {
    "challenge_id": 4, "is_elo_score": False, "metric_order": "desc",
    "ranked_order": "desc", "metric": "mean_reward", "metric2": None,
    "frontend_precision": 2, "metrics_schema": None, "has_gpu": False,
    "is_continuous": False,
}

_ENVELOPE = {
    "challenge": _CHALLENGE_BLOCK,
    "total": 12,
    "leaders": [{"rank": 1, "username": "jo", "submission_id": 11,
                 "mean_reward": 0.7, "is_my_submission": False,
                 "passed": True}],
    "me": {"rank": 4, "percentile": 66.7, "row": {}, "neighbors": []},
    "matches": [{"rank": 1, "username": "jo"}],
    "course_context": {"course_id": 3, "pass_threshold": 0.5},
}


def _has_pandas():
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    return True


def test_leaderboard_sends_the_consoles_query_keys():
    """The console sends `aggregate`, `course_id`, `limit`, `window`, `me`, `q`;
    the SDK sent only `limit`. Each is sent only when passed."""
    c, rec = make_client(lambda m, p, k: (200, _ENVELOPE))
    c.leaderboard(4, top=10, aggregate="user", course_id=3, me=True, q="jo", window=5)
    assert rec.last["path"] == "/leaderboard/challenge/4"
    assert rec.last["params"] == {"limit": 10, "aggregate": "user", "course_id": 3,
                                  "window": 5, "me": "true", "q": "jo"}
    c.leaderboard(4)
    assert rec.last["params"] is None
    c.leaderboard(4, aggregate="user")
    assert rec.last["params"] == {"aggregate": "user"}
    c.leaderboard(4, top=2)
    assert rec.last["params"] == {"limit": 2}


def test_leaderboard_is_one_shape_whatever_the_parameters():
    """The backend serves one envelope, with or without `limit`. With
    pandas the rows are `leaders` and every other block rides on `df.attrs`;
    without pandas the envelope comes back as served. Nothing depends on
    which parameters were passed."""
    c, _ = make_client(lambda *_: (200, _ENVELOPE))
    for kwargs in ({}, {"top": 10}, {"top": 10, "aggregate": "user"},
                   {"top": 10, "course_id": 3}, {"top": 10, "me": True},
                   {"q": "jo"}, {"window": 5}):
        out = c.leaderboard(4, **kwargs)
        if _has_pandas():
            assert out.to_dict("records") == _ENVELOPE["leaders"], kwargs
            assert out.attrs == {k: v for k, v in _ENVELOPE.items() if k != "leaders"}
            assert out.attrs["challenge"]["ranked_order"] == "desc"
            assert out.attrs["me"]["rank"] == 4
        else:
            assert out == _ENVELOPE, kwargs


def test_leaderboard_without_course_context_attaches_none():
    plain = {k: v for k, v in _ENVELOPE.items() if k != "course_context"}
    c, _ = make_client(lambda *_: (200, plain))
    out = c.leaderboard(4)
    if _has_pandas():
        assert "course_context" not in out.attrs
    else:
        assert "course_context" not in out


def test_leaderboard_refused_query_carries_the_reply():
    c, _ = make_client(lambda *_: (400, {"error": "Invalid aggregate value. Must be 'user' or omitted."}))
    try:
        c.leaderboard(4, aggregate="team")
        raise AssertionError("expected MLArenaError")
    except mlarena.MLArenaError as exc:
        assert exc.status_code == 400
        assert exc.body["error"].startswith("Invalid aggregate value")


def test_challenge_admin_and_creator_routes():
    tags = [{"id": 1, "name": "rl"}]

    def router(method, path, kwargs):
        if path == "/challenge_tags/tags":
            return (200, tags)
        return (201, {"id": 4}) if method == "POST" else (200, {"id": 4})

    c, rec = make_client(router, scope="creator")
    c.update_challenge_configuration(4, engine_id=2)
    assert (rec.last["method"], rec.last["path"]) == ("PUT", "/challenges/4/configuration")
    c.creator_challenges()
    assert rec.last["path"] == "/creator_challenge/challenges"
    c.list_tags()
    assert rec.last["path"] == "/challenge_tags/tags"

    c.create_challenge("New", "flex_v1", copy_from_challenge_id=3, tag_names=["RL"])
    assert (rec.last["method"], rec.last["path"]) == ("POST", "/creator_challenge/challenge")
    assert rec.last["json"] == {"name": "New", "kernel_version": "flex_v1",
                                "copy_from_challenge_id": 3, "tag_ids": [1]}

    c.set_challenge_tags(4, tag_ids=[1])
    assert rec.last["path"] == "/creator_challenge/challenge/4/tags"
    c.update_agent_template(4, "class Agent: ...")
    assert rec.last["path"] == "/creator_challenge/challenge/4/agent-template"
    assert rec.last["json"] == {"agent_template": "class Agent: ..."}
    c.start_challenge(4)
    assert rec.last["path"] == "/creator_challenge/challenge/4/start"


def test_update_settings_sends_the_metric_direction():
    """Every settings kwarg is the Evaluation column's own name — the
    `evaluation_` prefix the payload used to carry is gone."""
    c, rec = make_client(scope="creator")
    c.update_settings(4, metric_order="asc",
                      episode_budget_brackets=[[0.5, 3], [0.1, 10]])
    assert rec.last["path"] == "/creator_challenge/challenge/4/settings"
    assert rec.last["json"] == {
        "metric_order": "asc",
        "episode_budget_brackets": [[0.5, 3], [0.1, 10]],
    }


def test_update_settings_max_active_submissions_key():
    c, rec = make_client(scope="creator")
    c.update_settings(4, max_active_submissions_per_participant=3)
    assert rec.last["path"] == "/creator_challenge/challenge/4/settings"
    assert rec.last["json"] == {"max_active_submissions_per_participant": 3}


# --------------------------------------------------------------------------- #
# Python compat: old method names and keyword arguments
# --------------------------------------------------------------------------- #


def _call_warns(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    messages = [str(w.message) for w in caught
                if issubclass(w.category, DeprecationWarning)]
    return result, messages


def test_every_renamed_method_is_an_alias_of_a_real_method():
    for old, new in client_module._RENAMED_METHODS.items():
        alias = getattr(MLArenaClient, old)
        assert alias.__doc__ == f"Deprecated alias for :meth:`{new}`.", old
        target = getattr(MLArenaClient, new)
        assert not (target.__doc__ or "").startswith("Deprecated alias"), new
    # Kept names are real methods, not aliases.
    for kept in ("agent_runtime", "set_agent_runtime", "update_agent_template",
                 "submit", "status"):
        assert not (getattr(MLArenaClient, kept).__doc__ or "").startswith(
            "Deprecated alias"), kept


def test_old_submission_method_names_forward_to_new_wire():
    c, rec = make_client(lambda m, p, k: (201, {"submission_id": 11}) if m == "POST"
                         else (200, {"status": "ok"}))

    out, msgs = _call_warns(c.create_attached_agent, 4, agent_name="old",
                            copy_from_agent_id=9)
    assert out == {"submission_id": 11}
    assert rec.last["path"] == "/submissions/challenge/4"
    assert rec.last["json"] == {"submission_name": "old", "copy_from_submission_id": 9}
    assert any("create_attached_agent() is deprecated" in m for m in msgs), msgs
    assert any("agent_name= is deprecated, use submission_name=" in m for m in msgs), msgs
    assert any("copy_from_agent_id= is deprecated" in m for m in msgs), msgs

    _, msgs = _call_warns(c.agent_status, 4, attache_agent_id=11)
    assert rec.last["path"] == "/submissions/challenge/4/11/status"
    assert any("attache_agent_id= is deprecated, use submission_id=" in m for m in msgs)

    _, msgs = _call_warns(c.agent_games, 11)
    assert rec.last["path"] == "/submissions/submission/11/games"

    _, msgs = _call_warns(c.deploy_agent, 4, 11)
    assert rec.last["path"] == "/submissions/challenge/4/11/deploy"
    _, _ = _call_warns(c.agent_deploy_status, 4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("GET", "/submissions/challenge/4/11/deploy")
    _, _ = _call_warns(c.delete_agent, 4, 11)
    assert (rec.last["method"], rec.last["path"]) == ("DELETE", "/submissions/challenge/4/11")
    _, _ = _call_warns(c.list_agent_files, 4, 11)
    assert rec.last["path"] == "/submissions/challenge/4/11/file"
    _, _ = _call_warns(c.delete_agent_file, 4, 11, "a.py")
    assert rec.last["data"] == {"delete_file": "a.py"}

    # set_agent_runtime keeps its name; its old id keyword still works.
    _, msgs = _call_warns(c.set_agent_runtime, attache_agent_id=11, runtime_id=3)
    assert rec.last["path"] == "/submissions/agent_runtime/11"


def test_old_upload_names_and_status_agent_id():
    c, rec = make_client(lambda m, p, k: (200, {"content": "src"})
                         if p.endswith("/content") else (200, {"status": "ok"}))
    with tempfile.TemporaryDirectory() as d:
        _call_warns(c.upload_agent_file, 4, 11, _tmp_file(d, "agent.py"))
    assert rec.last["path"] == "/submissions/challenge/4/11/file"
    out, _ = _call_warns(c.get_agent_file_content, 4, 11, "agent.py")
    assert out == "src"
    _call_warns(c.update_agent_file_content, 4, 11, "agent.py", "x")
    assert rec.last["files"]["file"][0] == "agent.py"

    _, msgs = _call_warns(c.status, agent_id=5, challenge_id=2)
    assert rec.last["path"] == "/submissions/challenge/2/5/status"
    assert any("agent_id= is deprecated, use submission_id=" in m for m in msgs), msgs


def test_old_submit_kwarg_and_both_spellings_rejected():
    c, rec = make_client(_submit_router())
    out, msgs = _call_warns(c.submit, 4, agent=MyAgent, agent_name="legacy")
    assert rec.calls[0]["json"] == {"submission_name": "legacy"}
    assert out == {"submission_id": 11, "deploy": {"message": "queued"}}
    assert any("agent_name= is deprecated" in m for m in msgs), msgs

    try:
        c.submit(4, agent=MyAgent, agent_name="a", submission_name="b")
        raise AssertionError("expected TypeError")
    except TypeError as exc:
        assert "pass only submission_name=" in str(exc)


def test_old_settings_kwarg_sends_new_key():
    c, rec = make_client(scope="creator")
    _, msgs = _call_warns(c.update_settings, 4, max_active_agents_per_participant=2)
    assert rec.last["json"] == {"max_active_submissions_per_participant": 2}
    assert any("max_active_agents_per_participant= is deprecated" in m for m in msgs)


def test_challenge_rename_aliases_still_work():
    c, rec = make_client(lambda m, p, k: (200, {**_ENVELOPE, "leaders": []}
                                          if p.startswith("/leaderboard/")
                                          else {"items": [], "metadata": {}}),
                         scope="creator")
    _, msgs = _call_warns(c.competitions)
    assert rec.last["path"] == "/challenges/"
    assert any("competitions() is deprecated" in m for m in msgs), msgs
    _call_warns(c.leaderboard, competition_id=43)
    assert rec.last["path"] == "/leaderboard/challenge/43"
    _call_warns(c.create_competition, "N", "file_v1", copy_from_competition_id=3)
    assert rec.last["json"] == {"name": "N", "kernel_version": "file_v1",
                                "copy_from_challenge_id": 3}
    _call_warns(c.attach_competition, 2, competition_id=42)
    assert rec.last["path"] == "/teacher/modules/2/challenges"
    assert rec.last["json"] == {"challenge_id": 42}
    _call_warns(c.reorder_module_competitions, 2, ordered_competition_ids=[3, 1])
    assert rec.last["path"] == "/teacher/modules/2/challenges/reorder"
    assert CompetitionNotFoundError is ChallengeNotFoundError


# --------------------------------------------------------------------------- #
# Hard cut: the client source only speaks the new wire
# --------------------------------------------------------------------------- #


def test_client_source_has_no_old_route_or_payload_key():
    with open(os.path.join(SDK_ROOT, "mlarena", "client.py"), encoding="utf-8") as fh:
        src = fh.read()
    for old in ("direct_attache_agents", "creator_competition", "competition_tags",
                "/competitions/", "/leaderboard/competition", "/attaches/",
                "agent_attached_result", '"attache_agent_id"', '"copy_from_agent_id"',
                '"max_active_agents_per_participant"'):
        # Old names may only appear as keys of the compat maps.
        hits = [line for line in src.splitlines()
                if old in line and not re.match(r'\s*"[a-z_]+": "[a-z_]+",$', line)]
        assert hits == [], (old, hits)


def test_the_two_version_strings_agree():
    """`pyproject.toml` is what the wheel is built and published under;
    `__version__` is what the installed package reports. A bump that touches
    only one of them ships a wheel whose metadata and runtime disagree, which
    is invisible until someone reads `mlarena.__version__` to check what they
    have. Asserting they match, rather than pinning a literal, keeps the guard
    without making every release edit this file."""
    with open(os.path.join(SDK_ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        assert f'version = "{mlarena.__version__}"' in fh.read()


# --------------------------------------------------------------------------- #
# The read / monitor surface: one error contract, one exception per resource,
# and the routes the console had to itself.
# --------------------------------------------------------------------------- #


def test_a_missing_submission_is_a_submission_not_found_error():
    """A 404 on a submission used to raise `ChallengeNotFoundError`."""
    c, _ = make_client(lambda *_: (404, {"error": "Submission not found"}))

    for call in (
        lambda: c.submission_status(4, 11),
        lambda: c.submission_games(11),
        lambda: c.submission_overview(4, 11),
        lambda: c.submission_deploy_status(4, 11),
        lambda: c.delete_submission(4, 11),
        lambda: c.set_submission_visibility(11, True),
    ):
        try:
            call()
            raise AssertionError("expected SubmissionNotFoundError")
        except SubmissionNotFoundError as exc:
            # The server's reason, not "Not found".
            assert str(exc) == "Submission not found", str(exc)
        # `NotFoundError` is the catch-all for "it is not there".
        assert issubclass(SubmissionNotFoundError, NotFoundError)
        assert issubclass(ChallengeNotFoundError, NotFoundError)


def test_a_403_is_a_permission_denied_that_carries_the_reason():
    """It raised `AuthenticationError("Access denied")` — the wrong name, and
    the server's reason dropped. `PermissionDeniedError` still *is* an
    `AuthenticationError`, so existing `except` clauses keep working."""
    c, _ = make_client(lambda *_: (403, {"error": "You may not deploy this"}))
    try:
        c.deploy_submission(4, 11)
        raise AssertionError("expected PermissionDeniedError")
    except PermissionDeniedError as exc:
        assert str(exc) == "You may not deploy this"
        assert isinstance(exc, AuthenticationError)


def test_a_failed_deploy_status_keeps_the_servers_reason():
    """It ended on `raise_for_status()`, which threw the body away."""
    c, _ = make_client(lambda *_: (500, {"error": "the engine is down"}))
    try:
        c.submission_deploy_status(4, 11)
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "the engine is down" in str(exc)
        assert exc.status_code == 500
        assert exc.body == {"error": "the engine is down"}


def test_a_refused_deploy_carries_the_servers_limits():
    """A deploy 409 body is `{"error", "deployment_limits",
    "active_submission_limits"}` — the blocks `submission_deploy_status()`
    returns. The message is unchanged; the body is new, on the exception."""
    c, _ = make_client(lambda *_: (409, _DEPLOY_409))
    try:
        c.deploy_submission(4, 11)
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert str(exc) == "deploy_submission failed: Daily deployment limit (5) reached"
        assert exc.status_code == 409
        assert exc.body["deployment_limits"]["next_deploy_available_at"] == "2026-09-23T00:00:00+00:00"
        assert exc.body["active_submission_limits"]["max_active_submissions"] == 3
        assert isinstance(exc, mlarena.MLArenaError)


def test_every_http_refusal_carries_status_code_and_body():
    """401 / 403 / 404 / other: the same two attributes, the same messages."""
    for code, cls, message in (
        (401, AuthenticationError, "Invalid or missing API credentials."),
        (403, PermissionDeniedError, "Access denied"),
        (404, SubmissionNotFoundError, "Not found"),
        (500, SubmissionError, "submission_status failed: request failed"),
    ):
        c, _ = make_client(lambda *_, code=code: (code, {}))
        try:
            c.submission_status(4, 11)
            raise AssertionError(f"expected {cls.__name__}")
        except cls as exc:
            assert str(exc) == message, (code, str(exc))
            assert exc.status_code == code
            assert exc.body == {}

    # A reply with no JSON object: the text is the message, `body` is None.
    class NoJson(FakeResponse):
        def json(self):
            raise ValueError("no json")
    c, _ = make_client()
    c._request = lambda *a, **k: NoJson(502)
    try:
        c.submission_status(4, 11)
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert exc.status_code == 502
        assert exc.body is None

    # Not raised for a reply at all: both attributes default to None, and the
    # positional message still works.
    local = SubmissionError("Provide exactly one of agent= or files=")
    assert (local.status_code, local.body) == (None, None)
    assert str(local) == "Provide exactly one of agent= or files="


def test_exception_classes_follow_their_section():
    """The runtime methods raised `MLArenaError` while every other submission
    method raises `SubmissionError`; `recent_replays` (a challenge read) raised
    `SubmissionError`. `SubmissionError` is an `MLArenaError`, so `except
    MLArenaError` around the runtime calls keeps working."""
    c, _ = make_client(lambda *_: (500, {"error": "down"}))
    for call in (lambda: c.runtime_options(4),
                 lambda: c.agent_runtime(11),
                 lambda: c.set_agent_runtime(11, 7)):
        try:
            call()
            raise AssertionError("expected SubmissionError")
        except SubmissionError as exc:
            assert exc.status_code == 500
    try:
        c.recent_replays(4)
        raise AssertionError("expected MLArenaError")
    except SubmissionError:
        raise AssertionError("recent_replays is a challenge read, not a submission error")
    except mlarena.MLArenaError as exc:
        assert str(exc) == "recent_replays failed: down"


def test_create_submission_404_names_the_id_that_is_wrong():
    """`copy_from_submission_id` pointing at a missing (or someone else's)
    submission answers 404 "Source submission not found"; it read as a
    missing *challenge*."""
    c, _ = make_client(lambda *_: (404, {"error": "Source submission not found"}))
    try:
        c.create_submission(4, "copy", copy_from_submission_id=99)
        raise AssertionError("expected SubmissionNotFoundError")
    except SubmissionNotFoundError as exc:
        assert str(exc) == "Source submission not found"
        assert exc.status_code == 404
    c, _ = make_client(lambda *_: (404, {"error": "Challenge not found"}))
    try:
        c.create_submission(4, "plain")
        raise AssertionError("expected ChallengeNotFoundError")
    except ChallengeNotFoundError as exc:
        assert str(exc) == "Challenge not found"


def test_to_dataframe_passes_a_dict_through():
    """The `{columns, data}` branch is gone: no route ever produced it."""
    env = {"total": 1, "leaders": [{"rank": 1}]}
    assert client_module._to_dataframe(env) is env
    assert "columns" not in client_module._to_dataframe.__code__.co_consts


def test_my_submissions_route_and_shape():
    c, rec = make_client(lambda *_: (200, {
        "submissions": [{"id": 11, "challenge_id": 4}],
        "started_submissions_count": 1,
        "deployment_limits": {"daily_deploys_remaining": 3},
    }))

    out = c.my_submissions()

    assert rec.last["method"] == "GET"
    assert rec.last["path"] == "/submissions/mine"
    assert out["deployment_limits"]["daily_deploys_remaining"] == 3


def test_set_submission_visibility_route_and_body():
    c, rec = make_client(lambda *_: (200, {"is_public": True}))

    out = c.set_submission_visibility(11, True)

    assert rec.last["method"] == "PUT"
    assert rec.last["path"] == "/submissions/submission/11/visibility"
    assert rec.last["json"] == {"is_public": True}
    assert out == {"is_public": True}


def test_submission_overview_route():
    c, rec = make_client(lambda *_: (200, {"rank": 2, "agent_error_type": None}))

    out = c.submission_overview(4, 11)

    assert rec.last["method"] == "GET"
    assert rec.last["path"] == "/submission_result/4/11/overview"
    assert out["rank"] == 2


def test_the_client_reads_no_retired_payload_key():
    """The keys this surface renamed. They are gone from the server, so a
    `.get()` left behind here would silently read None forever.

    Only *quoted* occurrences count: the docstrings name the old spellings on
    purpose, in backticks, so a reader upgrading knows what moved.
    """
    with open(os.path.join(SDK_ROOT, "mlarena", "client.py"), encoding="utf-8") as fh:
        src = fh.read()
    for old in ("position_in_queue", "steps_completed", "current_reward",
                "deploymentLimits", "latestDeploy", "creation_date",
                "daily_deploys_count", "isEloRanked", "latest_error",
                "submission_performance",
                # 3.0: the flat 2.x run and the PascalCase leaderboard row
                "error_type", "opponents", "number_agent", "summary_status",
                "Rank", "Username", "SubmissionName", "MeanReward",
                "IsMySubmission", "submissionId", "RankedOrder",
                "IsEloRanked", "PassThreshold", "Passed", "ranked_by",
                # the overview's renamed error pair (the run's column names)
                "last_error_type", "last_error_message"):
        for literal in (f'"{old}"', f"'{old}'"):
            assert literal not in src, literal


# --------------------------------------------------------------------------- #
# 3.0: every read sends the bearer token, every refusal is an SDK exception
# --------------------------------------------------------------------------- #


def _every_read(c):
    """One call per method that used to skip the token or end on
    `requests`' `raise_for_status()` (the tag resolution behind
    `set_challenge_tags(tag_names=…)` included)."""
    return [
        lambda: c.challenges(),
        lambda: c.challenges(page=1),
        lambda: c.challenge(4),
        lambda: c.list_tags(),
        lambda: c.set_challenge_tags(4, tag_names=["rl"]),
        lambda: c.benchmark_status(4),
        lambda: c.global_ranking(),
        lambda: c.user_global_rank(7),
        lambda: c.data_source_weather_cities(),
    ]


def test_every_read_sends_the_bearer_token():
    """`user_global_rank` is login-required and was sent without the token,
    so it answered 401 for everyone. `list_tags`, `global_ranking` and the
    data-source reads are public, but a public route answers differently to
    a caller it can identify, and PROCESS.md promises the token on every
    read."""
    def router(method, path, kwargs):
        if path == "/challenge_tags/tags":
            return (200, [{"id": 1, "name": "rl"}])
        if path == "/ranking/":
            return (200, {"rankings": [], "metadata": {"total_users": 0}})
        return (200, {"ok": True})

    c, rec = make_client(router, scope="creator")
    for call in _every_read(c):
        rec.calls.clear()
        call()
        assert rec.calls
        for made in rec.calls:
            auth = (made.get("headers") or {}).get("Authorization", "")
            assert auth.startswith("Bearer "), made["path"]


def test_no_api_call_ends_on_raise_for_status():
    """A 5xx used to surface as `requests.HTTPError` with the server's
    reason thrown away, and a 401 on the routes that never called
    `_handle_response` was an `HTTPError` too. Every API refusal is an SDK
    exception carrying the reply. (`FakeResponse.raise_for_status` raises
    `AssertionError`, so a leftover call fails this test on its own.)"""
    c, _ = make_client(lambda *_: (500, {"error": "boom"}), scope="creator")
    for call in _every_read(c):
        try:
            call()
            raise AssertionError("expected MLArenaError")
        except mlarena.MLArenaError as exc:
            assert (exc.status_code, exc.body) == (500, {"error": "boom"})

    c, _ = make_client(lambda *_: (401, {"error": "nope"}), scope="creator")
    for call in _every_read(c):
        try:
            call()
            raise AssertionError("expected AuthenticationError")
        except mlarena.AuthenticationError as exc:
            assert (exc.status_code, exc.body) == (401, {"error": "nope"})

    # The one `.raise_for_status()` left is the signed-GCS dataset download
    # in `download_dataset`: a GCS reply, not an API one (no `error` body).
    with open(os.path.join(SDK_ROOT, "mlarena", "client.py"), encoding="utf-8") as fh:
        assert fh.read().count(".raise_for_status()") == 1
