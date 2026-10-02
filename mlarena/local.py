"""Run an env.py locally with the platform's numbers.

`eval_context(env_dir, …)` builds the dict the platform's executor attaches to
the env as `self.eval_context` before `evaluate`: the frozen keys env code has
always read, plus `config`, the challenge config as one flat dict keyed by
config key (`challenge.toml` keys, `update_settings` keywords,
`config_fields()`). A local test sets it the same way::

    from mlarena.local import eval_context
    env = Env(is_evaluation=True)
    env.eval_context = eval_context(".")                    # challenge.toml only
    env.eval_context = eval_context(".", challenge_id=177, client=c)  # + the platform's values
    env.evaluate(agents, agent_infos)

The values come from `challenge.toml` in `env_dir`; with `challenge_id` and
`client`, from the challenge's resolved config (`creator_challenge(id)
["config"]["values"]`) with the file laid over it, as the platform applies
it. A key the env reads that neither holds raises `KeyError` naming it,
rather than a value the platform would not use.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: the same parser, from PyPI (a dependency there).
    import tomli as tomllib

CONFIG_FILENAME = "challenge.toml"

# A metric entry's fields and their defaults when a file omits them (TOML has
# no null): the platform's `metrics_schema.spec`.
_SPEC_DEFAULTS = {
    "agg": "mean", "order": None, "format": "number", "unit": None,
    "precision": 2, "is_ranking": False, "visible": True,
}
_SPEC_FIELDS = ("key", "label", "source") + tuple(_SPEC_DEFAULTS)


class ChallengeConfig(dict):
    """The flat config; a missing key is a `KeyError` that says where to set it."""

    def __missing__(self, key):
        raise KeyError(
            f"{key} is not in {CONFIG_FILENAME}: add it, or pass challenge_id to "
            "read the challenge's resolved config"
        )


def _complete_metric(entry: Any) -> Any:
    if not isinstance(entry, dict) or not {"key", "label", "source"} <= entry.keys():
        return entry
    if set(entry) - set(_SPEC_FIELDS):
        return entry
    out = {**_SPEC_DEFAULTS, **entry}
    return {k: out[k] for k in _SPEC_FIELDS}


def read_config_file(env_dir: str) -> dict:
    """The values of `env_dir/challenge.toml` (empty without one), metric
    entries completed as the platform completes them."""
    path = os.path.join(env_dir, CONFIG_FILENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as fh:
        values = tomllib.load(fh)
    if isinstance(values.get("metrics"), list):
        values["metrics"] = [_complete_metric(m) for m in values["metrics"]]
    return values


def _match_order(metrics: list) -> Any:
    score = next((d for d in metrics if d.get("source") == "score"), None)
    if score is not None and score.get("order") is not None:
        return score["order"]
    ranking = [d for d in metrics if d.get("is_ranking")]
    return ranking[0].get("order") if len(ranking) == 1 else None


def _legacy_metrics_schema(metrics: list) -> Any:
    out = [
        {
            "key": d["key"], "label": d["label"], "source": d["source"],
            "agg": d["agg"], "format": d["format"], "precision": d["precision"],
            "higher_is_better": None if d["order"] is None else d["order"] == "desc",
            "is_ranking": d["is_ranking"], "visible": d["visible"],
        }
        for d in metrics
        if d["source"] in ("env", "platform")
    ]
    return out or None


def eval_context(env_dir: str, challenge_id: int | None = None, client: Any = None,
                 *, is_evaluation: bool = True) -> dict:
    """The env's `eval_context`, as the platform's flex_v1 executor builds it.

    `config` is a `ChallengeConfig`: `challenge.toml` of `env_dir` over the
    challenge's resolved config when `challenge_id` and `client` are given.
    The frozen keys (`n_episodes`, `max_steps`, `agent_call_timeout_sec`,
    `episode_budget_brackets`, `metric_order`, `simulation_timeout_sec`,
    `deadline_monotonic`, `metrics`, …) are derived from it the way the
    executor derives them; one whose config key is missing is None. A
    file_v1 env reads `config` only.
    """
    if (challenge_id is None) != (client is None):
        raise TypeError("pass challenge_id and client together")
    values: dict = {}
    if client is not None:
        values.update(client.creator_challenge(challenge_id)["config"]["values"])
    values.update(read_config_file(env_dir))
    config = ChallengeConfig(values)

    metrics = config.get("metrics") or []
    brackets = config.get("episode_budget_brackets")
    timeout = config.get("simulation_timeout_sec")
    deadline = None
    if timeout is not None:
        # The executor's library deadline: a little before the wallclock.
        deadline = time.monotonic() + timeout - max(5.0, 0.05 * timeout)
    return {
        "n_episodes": int(brackets[-1][-1]) if brackets else 1,
        "max_steps": config.get("simulation_max_steps"),
        "agent_call_timeout_sec": config.get("agent_max_time_per_step_second"),
        # Dropped from the config (the platform never enforced it): off.
        "env_step_soft_deadline_sec": 0.0,
        "episode_budget_brackets": brackets,
        "metric_order": _match_order(metrics) if metrics else None,
        "role_assignment_seed": None,
        "is_render": is_evaluation,
        "deadline_monotonic": deadline,
        "simulation_timeout_sec": timeout,
        "is_evaluation": is_evaluation,
        "metrics_schema": _legacy_metrics_schema(metrics) if metrics else None,
        "expected_metric_keys": [d["key"] for d in metrics if d.get("source") == "env"],
        "metrics": copy.deepcopy(metrics),
        "config": config,
    }
