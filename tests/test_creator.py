"""SDK tests for the creator-challenge methods the console had and the SDK did
not: the authoring reads (challenge detail, env / benchmark files, markdown,
image, tags, runs, submissions, assistants, kinds), the deletes, and the
clean-redeploy actions — plus the payload keys the settings write sends and
the machine list.

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


def _spec(key, label, source, agg, order, fmt, unit, precision, is_ranking,
          visible=True):
    return {"key": key, "label": label, "source": source, "agg": agg,
            "order": order, "format": fmt, "unit": unit,
            "precision": precision, "is_ranking": is_ranking,
            "visible": visible}


# docs/score_model.md §4.1 declarations (U1, U2 and an asc board).
REWARD_SPEC = _spec("reward", "Reward", "score", "mean", "desc", "number",
                    None, 2, True)
RMSE_SPEC = _spec("rmse", "RMSE", "score", "mean", "asc", "number", None, 3,
                  True)
RATED_METRICS = [
    _spec("elo", "Rating", "platform", "rating", "desc", "integer", None, 0,
          True),
    _spec("reward", "Reward", "score", "mean", "desc", "number", None, 2,
          False),
]


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
    ("agent_template", (7,), CREATOR + "/agent-template"),
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
        "evaluation": {"metrics": [REWARD_SPEC], "window_days": None},
        "environment": {"benchmark_simulation_result_id": 41},
    }
    c, _rec = make_client(lambda *_: (200, body))
    got = c.creator_challenge(7)
    _expect(got["evaluation"]["metrics"] == [REWARD_SPEC], got)
    _expect(got["environment"]["benchmark_simulation_result_id"] == 41, got)
    # The evaluation is its own object, never flattened into the configuration.
    _expect("evaluation_metric" not in got["configuration"], got["configuration"])


def test_benchmark_status_is_the_run_and_its_checks():
    """`{run, checks}` as served (4.3.0): the run in the shape `creator_runs()`
    lists (None before the first run), the config's budgets checked against it."""
    run = {"simulation_result_id": 41, "job_status": "completed",
           "env_error_type": None,
           "submission_results": [{"submission_id": 3, "score": 1.5}]}
    check = {"key": "simulation_timeout_sec", "label": "Run wallclock", "configured": 360,
             "measured": 12.0, "ratio": 0.0333, "status": "ok", "message": "…"}
    c, rec = make_client(lambda *_: (200, {"run": run, "checks": [check]}))
    got = c.benchmark_status(7)
    _expect(rec.last["method"] == "GET" and rec.last["path"] == CREATOR + "/benchmark/status",
            rec.last)
    _expect(got["run"]["submission_results"][0]["score"] == 1.5, got)
    _expect(got["checks"][0]["status"] == "ok", got)

    c, _rec = make_client(lambda *_: (200, {"run": None, "checks": []}))
    _expect(c.benchmark_status(7) == {"run": None, "checks": []}, "no run yet")


# --------------------------------------------------------------------------- #
# The challenge config (4.3.0)
# --------------------------------------------------------------------------- #

def test_config_fields_and_schema_are_public_reads():
    field = {"key": "simulation_timeout_sec", "group": "evaluation", "level": "basic",
             "writer": "creator", "capability": "runs", "running_editable": False,
             "in_file": True, "help": "…"}
    c, rec = make_client(lambda *_: (200, {"fields": [field]}))
    _expect(c.config_fields() == [field], "the registry rows")
    _expect(rec.last["method"] == "GET" and rec.last["path"] == "/creator_challenge/config_fields",
            rec.last)

    c, rec = make_client(lambda *_: (200, {"title": "challenge.toml", "properties": {}}))
    _expect(c.config_schema()["title"] == "challenge.toml", "the JSON Schema")
    _expect(rec.last["path"] == "/creator_challenge/config_schema.json", rec.last)


def test_export_config_returns_and_writes_the_toml():
    text = "#:schema x\nsimulation_timeout_sec = 600\n"
    c, rec = make_client(lambda *_: FakeResponse(200, text=text))
    _expect(c.export_config(7) == text, "the text")
    _expect(rec.last["method"] == "GET" and rec.last["path"] == CREATOR + "/config.toml", rec.last)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "challenge.toml")
        c.export_config(7, path=path)
        with open(path) as fh:
            _expect(fh.read() == text, "written to path")


def test_a_refused_challenge_toml_raises_with_the_line():
    error = {"error": "challenge.toml:2: unknown key 'bogus'"}
    c, _rec = make_client(lambda *_: (400, error))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "challenge.toml")
        with open(path, "w") as fh:
            fh.write("simulation_timeout_sec = 600\nbogus = 1\n")
        try:
            c.upload_env_file(7, path)
        except MLArenaError as exc:
            _expect(exc.status_code == 400 and "challenge.toml:2" in str(exc), str(exc))
        else:
            raise AssertionError("a refused file raises")


def test_update_settings_sends_required_files_and_refuses_the_dropped_key():
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(7, required_files=["y_test.csv"])
    _expect(rec.last["json"] == {"required_files": ["y_test.csv"]}, rec.last["json"])
    try:
        c.update_settings(7, env_max_time_per_step_second=1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("env_max_time_per_step_second is gone (4.3.0)")


def test_a_file_owned_key_is_the_backend_400():
    error = {"error": "simulation_timeout_sec is set in challenge.toml (line 3); edit the file"}
    c, _rec = make_client(lambda *_: (400, error))
    try:
        c.update_settings(7, simulation_timeout_sec=300)
    except MLArenaError as exc:
        _expect("challenge.toml (line 3)" in str(exc), str(exc))
    else:
        raise AssertionError("a file-owned key raises")


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
        metrics=[RMSE_SPEC],
        window_days=30,
        episode_budget_brackets=[[1.0, 3], [0.3, 10]],
        max_upload_files=4,
    )
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == CREATOR + "/settings", rec.last["path"])
    _expect(rec.last["json"] == {
        "metrics": [RMSE_SPEC],
        "window_days": 30,
        "episode_budget_brackets": [[1.0, 3], [0.3, 10]],
        "max_upload_files": 4,
    }, rec.last["json"])


def test_update_settings_removed_score_kwargs_raise_naming_metrics():
    """SDK 4.0.0 (D5): the six legacy score keywords are gone, no alias. Each
    raises a TypeError that points at `metrics`, before any request."""
    removed = {
        "metric": "rmse", "metric2": "mae", "is_elo_score": True,
        "metric_order": "asc", "frontend_precision": 3,
        "metrics_schema": [{"key": "rmse"}],
    }
    for name, value in removed.items():
        c, rec = make_client()
        try:
            c.update_settings(7, **{name: value})
        except TypeError as exc:
            _expect("metrics=" in str(exc), str(exc))
            _expect(name in str(exc), str(exc))
            _expect(rec.calls == [], "no request was sent")
        else:
            raise AssertionError(f"update_settings still accepts {name}=")
    # Mixed with the new keyword, still refused: no silent drop.
    c, rec = make_client()
    try:
        c.update_settings(7, metrics=[RMSE_SPEC], metric_order="asc")
    except TypeError:
        _expect(rec.calls == [], "no request was sent")
    else:
        raise AssertionError("update_settings dropped metric_order= silently")


def test_update_settings_sends_the_elo_tunables():
    """The ELO tunables are Evaluation columns, sent under their own names."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    tunables = {
        "elo_k_factor": 16, "elo_d0": 400.0, "elo_alpha": 0.0, "elo_beta": 5.0,
        "elo_momentum": 0.5, "elo_initial_variance": 10000.0,
        "elo_initial_score": 1000.0,
    }
    c.update_settings(7, metrics=RATED_METRICS, **tunables)
    _expect(rec.last["json"] == {"metrics": RATED_METRICS, **tunables},
            rec.last["json"])


def test_create_challenge_docstring_names_only_real_kinds():
    """The kinds are the backend's KINDS keys; env-image families are not."""
    doc = mlarena.client.MLArenaClient.create_challenge.__doc__
    _expect('"gymnasium"' not in doc and '"pettingzoo"' not in doc, doc)
    _expect("available_kinds()" in doc, doc)


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
    """The agent-call deadline and step budget (creator keys since 4.3.0) go
    through PUT /settings under their column names."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(
        7,
        agent_max_time_per_step_second=0.25,
        simulation_max_steps=500,
    )
    _expect(rec.last["path"] == CREATOR + "/settings", rec.last["path"])
    _expect(rec.last["json"] == {
        "agent_max_time_per_step_second": 0.25,
        "simulation_max_steps": 500,
    }, rec.last["json"])


def test_data_source_weather_series_sends_the_response_keys():
    c, rec = make_client(lambda *_: (200, {"city_name": "Paris", "points": []}))
    c.data_source_weather_series(city_name="Paris", country_code="FR", hours=24)
    _expect(rec.last["path"] == "/data_sources/weather/series", rec.last["path"])
    _expect(rec.last["params"] == {"city_name": "Paris", "country_code": "FR", "hours": 24},
            rec.last["params"])


RUNTIME_KWARGS = {
    "machine_id": 3,
    "env_cpu_request": "500m", "env_cpu_limit": "1",
    "env_memory_request": "512Mi", "env_memory_limit": "1Gi",
    "agent_cpu_request": "250m", "agent_cpu_limit": "500m",
    "agent_memory_request": "256Mi", "agent_memory_limit": "512Mi",
    "agent_ephemeral_storage_limit": "1Gi",
    "env_gpu_count": 0, "agent_gpu_count": 1, "gpu_memory_limit": "8Gi",
    "number_of_agents": 2,
    "docker_image_env_runtime_id": 2, "render_delay_second": 0.05,
}


def test_update_settings_sends_the_runtime_fields():
    """The machine, the sizing columns, the seat count and the infrastructure
    keys of the removed configuration route go through PUT /settings under
    their column names (machine_model D7)."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(7, **RUNTIME_KWARGS)
    _expect(rec.last["method"] == "PUT", rec.last["method"])
    _expect(rec.last["path"] == CREATOR + "/settings", rec.last["path"])
    _expect(rec.last["json"] == RUNTIME_KWARGS, rec.last["json"])


def test_update_settings_gpu_memory_limit_none_clears_it():
    """`gpu_memory_limit=None` is an explicit null (the backend clears the
    cap); the other runtime keywords' None means "not sent"."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(7, gpu_memory_limit=None, machine_id=None)
    _expect(rec.last["json"] == {"gpu_memory_limit": None}, rec.last["json"])


def test_update_settings_window_days_none_removes_the_window():
    """`window_days=None` is an explicit null (the board goes back to no
    window, as the console can do); omitting it sends nothing."""
    c, rec = make_client(lambda *_: (200, {"configuration": {}, "evaluation": {}}))
    c.update_settings(7, window_days=None)
    _expect(rec.last["json"] == {"window_days": None}, rec.last["json"])
    c.update_settings(7, window_days=30)
    _expect(rec.last["json"] == {"window_days": 30}, rec.last["json"])
    c.update_settings(7, simulation_timeout_sec=60)
    _expect(rec.last["json"] == {"simulation_timeout_sec": 60}, rec.last["json"])


def test_update_settings_refuses_engine_id():
    """Engines became machines: no silent id mapping."""
    c, _rec = make_client()
    try:
        c.update_settings(7, engine_id=3)
    except TypeError:
        pass
    else:
        raise AssertionError("update_settings accepts engine_id")


def test_update_challenge_configuration_is_gone():
    """The 4.1.0 deprecated alias was removed at 4.2.0."""
    _expect(not hasattr(mlarena.client.MLArenaClient, "update_challenge_configuration"),
            "update_challenge_configuration still defined")


def test_machines_lists_the_visible_queues():
    rows = [{"id": 3, "name": "gke-autoscaling", "kernels": []}]
    c, rec = make_client(lambda *_: (200, rows))
    _expect(c.machines() == rows, "rows as served")
    _expect((rec.last["method"], rec.last["path"]) == ("GET", "/machines/"),
            (rec.last["method"], rec.last["path"]))
    _expect(rec.last["params"] == {}, rec.last["params"])
    c.machines(kernel_version="flex_v1")
    _expect(rec.last["params"] == {"kernel_version": "flex_v1"}, rec.last["params"])


def test_machines_refusal_carries_the_reply():
    c, _ = make_client(lambda *_: (400, {"error": "Unknown kernel_version 'x'"}))
    try:
        c.machines(kernel_version="x")
    except MLArenaError as exc:
        _expect(exc.status_code == 400, exc.status_code)
    else:
        raise AssertionError("expected MLArenaError")


def test_no_engine_left_in_the_public_docstrings():
    """engine → machine (machine_model): the kinds, challenge and creator
    docstrings name the served keys."""
    cls = mlarena.client.MLArenaClient
    for name in ("challenge", "creator_challenge", "available_kinds",
                 "create_challenge", "copyable_challenges", "challenges"):
        doc = getattr(cls, name).__doc__
        _expect("engine" not in doc.lower(), f"{name}: {doc}")
    _expect("has_machine" in cls.available_kinds.__doc__, "has_machine")
    _expect("runtime_presets" in cls.available_kinds.__doc__, "runtime_presets")
    _expect("machine_name" in cls.creator_challenge.__doc__, "machine_name")


def test_recent_replays_passes_its_limit():
    c, rec = make_client(lambda *_: (200, {"replays": []}))
    c.recent_replays(7, limit=5)
    _expect(rec.last["path"] == "/challenges/7/recent-replays", rec.last["path"])
    _expect(rec.last["params"] == {"limit": 5}, rec.last["params"])



def test_agent_template_returns_the_served_template():
    c, rec = make_client(lambda *_: (200, {"agent_template": "class Agent: ..."}))
    _expect(c.agent_template(7) == {"agent_template": "class Agent: ..."},
            "reply as served")
    _expect((rec.last["method"], rec.last["path"])
            == ("GET", CREATOR + "/agent-template"), rec.last["path"])


def test_public_stats_reads_the_landing_counters():
    stats = {"users": 12, "challenges": 3, "submissions": 40}
    c, rec = make_client(lambda *_: (200, stats))
    _expect(c.public_stats() == stats, "reply as served")
    _expect((rec.last["method"], rec.last["path"]) == ("GET", "/public/stats/"),
            (rec.last["method"], rec.last["path"]))


def test_public_stats_refusal_carries_the_reply():
    c, _ = make_client(lambda *_: (500, {"error": "boom"}))
    try:
        c.public_stats()
    except MLArenaError as exc:
        _expect(exc.status_code == 500, exc.status_code)
    else:
        raise AssertionError("expected MLArenaError")

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
