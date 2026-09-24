"""SDK tests for the creator-challenge methods the console had and the SDK did
not: the authoring reads (challenge detail, env / benchmark files, markdown,
image, tags, runs, submissions, assistants, kinds), the deletes, and the
clean-redeploy actions — plus the payload keys the settings and configuration
writes now send.

Same offline harness as ``test_teacher_admin.py``: the client's single HTTP
chokepoint (``MLArenaClient._request``) is replaced by a recorder, so each test
asserts the method → (verb, path, body, params) mapping the parity rule
requires without a network or a backend.

Runs under pytest *or* directly: ``python tests/test_creator.py``.
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


def make_client(router=None, scope="creator"):
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


CREATOR = "/creator_challenge/challenge/7"


# --------------------------------------------------------------------------- #
# Reads: every creator GET the console had
# --------------------------------------------------------------------------- #

READS = [
    ("creator_challenge", (7,), CREATOR),
    ("list_env_files", (7,), CREATOR + "/env/files"),
    ("list_benchmark_files", (7,), CREATOR + "/benchmark/files"),
    ("challenge_markdown", (7,), CREATOR + "/markdown"),
    ("challenge_tags", (7,), CREATOR + "/tags"),
    ("creator_runs", (7,), CREATOR + "/runs"),
    ("creator_submissions", (7,), CREATOR + "/submissions"),
    ("challenge_assistants", (7,), CREATOR + "/assistants"),
    ("csv_ground_truth", (7,), CREATOR + "/csv-ground-truth"),
    ("available_kinds", (), "/creator_challenge/available_kinds"),
    ("copyable_challenges", (), "/creator_challenge/copyable_challenges"),
]


def test_creator_reads_hit_their_routes():
    c, rec = make_client(lambda *_: (200, {"ok": True}))
    for name, args, path in READS:
        getattr(c, name)(*args)
        _expect(rec.last["method"] == "GET", f"{name}: {rec.last['method']}")
        _expect(rec.last["path"] == path, f"{name}: {rec.last['path']} != {path}")


def test_creator_reads_send_the_token():
    c, rec = make_client(lambda *_: (200, {"ok": True}))
    for name, args, _path in READS:
        getattr(c, name)(*args)
    for call in rec.calls:
        auth = call["headers"]["Authorization"]
        _expect(auth.startswith("Bearer mlk_creator_"), auth)


def test_a_read_of_a_challenge_you_cannot_edit_raises_not_found():
    c, _rec = make_client(lambda *_: (404, {"error": "Challenge not found or unauthorized"}))
    for name in ("creator_challenge", "list_env_files", "creator_runs"):
        try:
            getattr(c, name)(7)
        except mlarena.exceptions.ChallengeNotFoundError as exc:
            _expect("not found" in str(exc).lower(), str(exc))
        else:
            raise AssertionError(f"{name} did not raise on a 404")


def test_creator_challenge_returns_the_three_sibling_rows():
    body = {
        "id": 7, "name": "Pong", "role": "owner",
        "configuration": {"submission_filename": "submission.csv"},
        "evaluation": {"metric": "reward", "metric_order": "desc"},
        "environment": {"benchmark_simulation_result_id": 41},
    }
    c, _rec = make_client(lambda *_: (200, body))
    got = c.creator_challenge(7)
    _expect(got["evaluation"]["metric"] == "reward", got)
    _expect(got["environment"]["benchmark_simulation_result_id"] == 41, got)
    # The evaluation is its own object, never flattened into the configuration.
    _expect("evaluation_metric" not in got["configuration"], got["configuration"])


def test_benchmark_status_is_the_run_or_none():
    """The latest benchmark run, in the run shape `creator_runs()` lists, or
    None (a JSON `null`) before the first run."""
    run = {"simulation_result_id": 41, "job_status": "completed",
           "env_error_type": None,
           "submission_results": [{"submission_id": 3, "submission_reward": 1.5}]}
    c, rec = make_client(lambda *_: (200, run))
    got = c.benchmark_status(7)
    _expect(rec.last["method"] == "GET" and rec.last["path"] == CREATOR + "/benchmark/status",
            rec.last)
    _expect(got["submission_results"][0]["submission_reward"] == 1.5, got)

    c, _rec = make_client()
    c._request = lambda *a, **k: _NullJson()
    _expect(c.benchmark_status(7) is None, "no run yet is None")


class _NullJson(FakeResponse):
    """A 200 whose body is JSON `null` (FakeResponse maps None to {})."""

    def __init__(self):
        super().__init__(200)

    def json(self):
        return None


# --------------------------------------------------------------------------- #
# Deletes
# --------------------------------------------------------------------------- #

DELETES = [
    ("delete_env_file", (7, "helper.py"), CREATOR + "/env/files/helper.py"),
    ("delete_benchmark_file", (7, "agent.py"), CREATOR + "/benchmark/files/agent.py"),
    ("delete_dataset", (7, 3), CREATOR + "/datasets/3"),
    ("delete_challenge_image", (7,), CREATOR + "/image"),
    ("soft_delete_submission", (7, 42), CREATOR + "/submissions/42"),
    ("remove_challenge_assistant", (7, 9), CREATOR + "/assistants/9"),
]


def test_creator_deletes_hit_their_routes():
    c, rec = make_client(lambda *_: (200, {"message": "ok"}))
    for name, args, path in DELETES:
        getattr(c, name)(*args)
        _expect(rec.last["method"] == "DELETE", f"{name}: {rec.last['method']}")
        _expect(rec.last["path"] == path, f"{name}: {rec.last['path']} != {path}")


def test_a_refused_delete_raises_with_the_server_reason():
    c, _rec = make_client(lambda *_: (400, {"error": "Cannot delete env.py"}))
    try:
        c.delete_env_file(7, "env.py")
    except MLArenaError as exc:
        _expect("Cannot delete env.py" in str(exc), str(exc))
    else:
        raise AssertionError("delete_env_file did not raise on a 400")


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #

def test_check_env_posts_the_buffer():
    c, rec = make_client(lambda *_: (200, {"errors": []}))
    c.check_env(7, "class Env:\n    pass\n")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == CREATOR + "/env/check", rec.last["path"])
    _expect(rec.last["json"] == {"content": "class Env:\n    pass\n"}, rec.last["json"])
    # The default reports the missing-source case rather than sending nothing.
    c.check_env(7)
    _expect(rec.last["json"] == {"content": ""}, rec.last["json"])


def test_clean_redeploy_actions_post_to_their_routes():
    c, rec = make_client(lambda *_: (200, {"submission_id": 42}))
    c.clean_redeploy_submission(7, 42)
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == CREATOR + "/submissions/42/clean_redeploy",
            rec.last["path"])

    c.clean_redeploy_all(7)
    _expect(rec.last["path"] == CREATOR + "/submissions/clean_redeploy_all",
            rec.last["path"])


def test_add_challenge_assistant_sends_the_username():
    c, rec = make_client(lambda *_: (201, {"user_id": 9, "username": "ta"}))
    c.add_challenge_assistant(7, "ta")
    _expect(rec.last["method"] == "POST", rec.last["method"])
    _expect(rec.last["path"] == CREATOR + "/assistants", rec.last["path"])
    _expect(rec.last["json"] == {"username": "ta"}, rec.last["json"])


def test_challenge_image_writes_the_bytes_it_got():
    png = b"\x89PNG\r\n\x1a\nbody"
    c, rec = make_client(lambda *_: FakeResponse(200, {}, content=png))
    with tempfile.TemporaryDirectory() as tmp:
        path = c.challenge_image(7, tmp)
        _expect(path.endswith("miniature.png"), path)
        with open(path, "rb") as fh:
            _expect(fh.read() == png, "image bytes round-trip")
    _expect(rec.last["path"] == CREATOR + "/image", rec.last["path"])


# --------------------------------------------------------------------------- #
# Settings + configuration: the payload keys are the column names
# --------------------------------------------------------------------------- #

def test_update_settings_sends_the_column_names():
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(
        7,
        metric="rmse",
        metric_order="asc",
        is_elo_score=False,
        frontend_precision=3,
        episode_budget_brackets=[[1.0, 3], [0.3, 10]],
        max_upload_files=4,
    )
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == CREATOR + "/settings", rec.last["path"])
    _expect(rec.last["json"] == {
        "metric": "rmse",
        "metric_order": "asc",
        "is_elo_score": False,
        "frontend_precision": 3,
        "episode_budget_brackets": [[1.0, 3], [0.3, 10]],
        "max_upload_files": 4,
    }, rec.last["json"])


def test_update_settings_without_a_field_raises_before_any_request():
    c, rec = make_client()
    try:
        c.update_settings(7)
    except MLArenaError:
        _expect(rec.calls == [], "no request was sent")
    else:
        raise AssertionError("update_settings accepted an empty update")


def test_update_settings_refuses_the_old_evaluation_prefix():
    """The rename is a hard cut: `evaluation_metric` is not a kwarg any more."""
    c, _rec = make_client()
    try:
        c.update_settings(7, evaluation_metric="rmse")
    except TypeError:
        pass
    else:
        raise AssertionError("update_settings still accepts evaluation_metric")


def test_update_settings_sends_the_data_feed_fields():
    """Admin-only feed keys go through the same PUT /settings, under their
    column names; None is an explicit null (clears the filter)."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(
        7,
        data_source_enabled=True,
        data_source_url="http://dc.test/api-dc",
        data_source_asset="weather",
        data_source_filter={"cities": "Paris:FR"},
        data_source_history_hours=72,
        batch_cron="0 */6 * * *",
    )
    _expect(rec.last["json"] == {
        "data_source_enabled": True,
        "data_source_url": "http://dc.test/api-dc",
        "data_source_asset": "weather",
        "data_source_filter": {"cities": "Paris:FR"},
        "data_source_history_hours": 72,
        "batch_cron": "0 */6 * * *",
    }, rec.last["json"])

    c.update_settings(7, data_source_filter=None)
    _expect(rec.last["json"] == {"data_source_filter": None}, rec.last["json"])


def test_update_settings_sends_the_run_limit_fields():
    """The admin-only step deadlines and step budget go through PUT /settings
    under their column names; the configuration PUT no longer takes them."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(
        7,
        agent_max_time_per_step_second=0.25,
        env_max_time_per_step_second=1.5,
        simulation_max_steps=500,
    )
    _expect(rec.last["path"] == CREATOR + "/settings", rec.last["path"])
    _expect(rec.last["json"] == {
        "agent_max_time_per_step_second": 0.25,
        "env_max_time_per_step_second": 1.5,
        "simulation_max_steps": 500,
    }, rec.last["json"])


def test_data_source_weather_series_sends_the_response_keys():
    c, rec = make_client(lambda *_: (200, {"city_name": "Paris", "points": []}))
    c.data_source_weather_series(city_name="Paris", country_code="FR", hours=24)
    _expect(rec.last["path"] == "/data_sources/weather/series", rec.last["path"])
    _expect(rec.last["params"] == {"city_name": "Paris", "country_code": "FR", "hours": 24},
            rec.last["params"])


def test_update_challenge_configuration_is_the_infrastructure_fields_only():
    c, rec = make_client(lambda *_: (200, {"engine_id": 3}))
    c.update_challenge_configuration(
        7, engine_id=3, docker_image_env_runtime_id=2, render_delay_second=0.05
    )
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == "/challenges/7/configuration", rec.last["path"])
    _expect(rec.last["json"] == {
        "engine_id": 3,
        "docker_image_env_runtime_id": 2,
        "render_delay_second": 0.05,
    }, rec.last["json"])

    # The step deadlines and the simulation budget are update_settings'.
    for field in ("agent_max_time_per_step_second", "simulation_max_steps",
                  "env_max_time_per_step_second", "agent_template"):
        try:
            c.update_challenge_configuration(7, **{field: 1})
        except MLArenaError:
            pass
        else:
            raise AssertionError(f"configuration still accepts {field}")


def test_recent_replays_passes_its_limit():
    c, rec = make_client(lambda *_: (200, {"replays": []}))
    c.recent_replays(7, limit=5)
    _expect(rec.last["path"] == "/challenges/7/recent-replays", rec.last["path"])
    _expect(rec.last["params"] == {"limit": 5}, rec.last["params"])


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
