"""`mlarena.local.eval_context`: the env's `eval_context`, built locally from
challenge.toml (and the challenge's resolved config) the way the platform's
executor builds it. Offline; runs under pytest or directly."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlarena.local import eval_context  # noqa: E402

TOML = """#:schema x
agent_max_time_per_step_second = 60.0
episode_budget_brackets = [[1000000000, 3]]
metrics = [
  { key = "crps", label = "Mean CRPS", source = "score", order = "asc", is_ranking = true },
  { key = "n", label = "Samples", source = "env", agg = "sum" },
]
"""


def _env_dir(text):
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "challenge.toml"), "w") as fh:
        fh.write(text)
    return tmp


def test_file_only():
    ctx = eval_context(_env_dir(TOML))
    cfg = ctx["config"]
    assert cfg["agent_max_time_per_step_second"] == 60.0
    assert ctx["agent_call_timeout_sec"] == 60.0
    assert ctx["n_episodes"] == 3
    assert ctx["metric_order"] == "asc"
    assert ctx["expected_metric_keys"] == ["n"]
    # Metric entries are completed with the platform's defaults.
    assert cfg["metrics"][1]["order"] is None and cfg["metrics"][1]["visible"] is True
    assert ctx["metrics_schema"][0]["key"] == "n"
    assert ctx["simulation_timeout_sec"] is None and ctx["deadline_monotonic"] is None


def test_a_missing_key_names_where_to_set_it():
    cfg = eval_context(_env_dir(TOML))["config"]
    try:
        cfg["simulation_max_steps"]
    except KeyError as exc:
        assert "simulation_max_steps is not in challenge.toml" in str(exc)
        assert "challenge_id" in str(exc)
    else:
        raise AssertionError("expected KeyError")


class _Client:
    def __init__(self):
        self.asked = []

    def creator_challenge(self, cid):
        self.asked.append(cid)
        return {"config": {"values": {
            "simulation_timeout_sec": 600, "simulation_max_steps": 1000,
            "agent_max_time_per_step_second": 0.5,
        }}}


def test_the_file_overlays_the_resolved_config():
    client = _Client()
    ctx = eval_context(_env_dir(TOML), challenge_id=177, client=client)
    assert client.asked == [177]
    assert ctx["config"]["simulation_max_steps"] == 1000          # resolved
    assert ctx["config"]["agent_max_time_per_step_second"] == 60.0  # the file wins
    assert ctx["simulation_timeout_sec"] == 600 and ctx["deadline_monotonic"] is not None


def test_challenge_id_and_client_go_together():
    try:
        eval_context(_env_dir(TOML), challenge_id=1)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_no_file_is_an_empty_config():
    ctx = eval_context(tempfile.mkdtemp())
    assert dict(ctx["config"]) == {} and ctx["n_episodes"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
