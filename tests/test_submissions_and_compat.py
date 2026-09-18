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
                    {"job_status": "running", "agent_nb_steps": 5,
                     "submission_reward": 1.0, "game_outcome": None,
                     "error_type": None},
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
    state = {"polls": 0}

    def router(method, path, kwargs):
        state["polls"] += 1
        # A run is failed when it carries an `error_type`, not when a derived
        # status string says "error": `job_status` is the job's own column.
        if state["polls"] == 1:
            steps, job_status, error_type = 5, "running", None
        elif state["polls"] == 2:
            steps, job_status, error_type = 9, "failed", "code_error"
        else:
            return (200, {"status": "deploy_failed", "last_status_message": "crash",
                          "is_settled": True, "run_info": {"results": []}})
        return (200, {
            "status": "deploy_run",
            "last_status_message": "running",
            "is_settled": False,
            "run_info": {"results": [
                {"job_status": job_status, "agent_nb_steps": steps,
                 "submission_reward": 1.0, "game_outcome": None,
                 "error_type": error_type, "error_message": "boom"},
            ]},
        })

    c, _ = make_client(router)
    lines = with_fake_clock(lambda: list(c.tail_logs(4, 11)))

    assert lines == [
        "[deploy_run] running",
        "  run: job_status=running steps=5 reward=1.0 outcome=None",
        "  run: job_status=failed steps=9 reward=1.0 outcome=None",
        "    error[code_error]: boom",
        "[deploy_failed] crash",
    ], lines


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
            return (200, {"message": "queued"})
        return (200, {"status": "deploy_queue"})
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
    assert not any(call["path"].endswith("/deploy") for call in rec.calls)


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
    c, rec = make_client(lambda m, p, k: (200, {"items": [], "metadata": {}})
                         if p == "/challenges/" else (200, {"datasets": []}))
    c.challenges()
    assert rec.last["path"] == "/challenges/"
    c.challenges(page=1)
    assert rec.last["path"] == "/challenges/"
    c.challenge(4)
    assert rec.last["path"] == "/challenges/4"
    c.datasets(4)
    assert rec.last["path"] == "/challenges/4/datasets"
    c.recent_replays(4)
    assert rec.last["path"] == "/challenges/4/recent-replays"
    c.leaderboard(4)
    assert rec.last["path"] == "/leaderboard/challenge/4"


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
    c, rec = make_client(scope="creator")
    c.update_settings(4, evaluation_metric_order="asc",
                      evaluation_episode_budget_brackets=[[0.5, 3], [0.1, 10]])
    assert rec.last["path"] == "/creator_challenge/challenge/4/settings"
    assert rec.last["json"] == {
        "evaluation_metric_order": "asc",
        "evaluation_episode_budget_brackets": [[0.5, 3], [0.1, 10]],
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
    c, rec = make_client(lambda m, p, k: (200, {"items": [], "metadata": {}}),
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


def test_version_is_2_0_0():
    assert mlarena.__version__ == "2.0.0"
    with open(os.path.join(SDK_ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        assert 'version = "2.0.0"' in fh.read()


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
    c, rec = make_client(lambda *_: (200, {"rank": 2, "last_error_type": None}))

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
                "submission_performance"):
        for literal in (f'"{old}"', f"'{old}'"):
            assert literal not in src, literal
