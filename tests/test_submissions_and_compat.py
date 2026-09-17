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
    ChallengeNotFoundError,
    CompetitionNotFoundError,
    SubmissionError,
)

SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.text = ""

    def json(self):
        return self._json

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


def test_tail_logs_polls_submission_status():
    c, rec = make_client(lambda *_: (200, {"status": "active",
                                           "last_status_message": "done"}))
    lines = list(c.tail_logs(4, 11))
    assert lines == ["[active] done"]
    assert rec.last["path"] == "/submissions/challenge/4/11/status"


# --------------------------------------------------------------------------- #
# submit() / status()
# --------------------------------------------------------------------------- #


def _submit_router(upload_status="upload_validated"):
    def router(method, path, kwargs):
        if method == "POST" and path == "/submissions/challenge/4":
            return (201, {"submission_id": 11, "status": "created"})
        if path.startswith("/submissions/runtime_options/"):
            return (200, [{"id": 7, "language": "python", "framework": "torch"}])
        if path.endswith("/file"):
            return (200, {"status": upload_status, "validation_message": "bad name"})
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
        assert rec.calls[-3]["json"] == {"submission_name": "named"}
    assert set(out) == {"submission_id", "deploy"}


def test_submit_upload_failed_raises_before_deploy():
    c, rec = make_client(_submit_router(upload_status="upload_failed"))
    try:
        c.submit(4, agent=MyAgent)
        raise AssertionError("expected SubmissionError")
    except SubmissionError as exc:
        assert "Submission 11 did not pass upload validation: bad name" in str(exc)
        assert "delete_submission(4, 11)" in str(exc)
    assert not any(call["path"].endswith("/deploy") for call in rec.calls)


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
