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

`record_render(env, out_dir)` injects the platform's replay API on the env
(`save_render`, `save_render_frames`, `save_render_roles`,
`start_render_video`, `clear_render`) with the platform's rules, and writes
what the run recorded as files you can open, under `out_dir/replay/`::

    from mlarena.local import record_render
    with record_render(env, "out") as rec:
        env.evaluate(agents, agent_infos)
    print(rec.render_format, rec.path)   # e.g. frames_v2 out/replay/frames.json
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
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
        # Frozen key the executor still sets to `is_evaluation`
        # (workers/flex_v1/executor/main.py): flexkit records a replay only
        # when it is true, so a local run with `record_render` renders too.
        "is_render": is_evaluation,
        "deadline_monotonic": deadline,
        "simulation_timeout_sec": timeout,
        "is_evaluation": is_evaluation,
        "metrics_schema": _legacy_metrics_schema(metrics) if metrics else None,
        "expected_metric_keys": [d["key"] for d in metrics if d.get("source") == "env"],
        "metrics": copy.deepcopy(metrics),
        "config": config,
    }


# --------------------------------------------------------------------------- #
# record_render: the platform's replay recorder, writing reviewable files
# --------------------------------------------------------------------------- #

# The machine caps' defaults (`machine.replay_frames_max`,
# `machine.replay_video_second_max`) and `render_delay_second`'s column
# default: what a challenge on a default machine gets.
REPLAY_FRAMES_MAX = 900
REPLAY_VIDEO_SECOND_MAX = 600
RENDER_DELAY_SECOND = 0.1

# The platform's H.264 profile (every browser plays it, seekable from the
# first byte; odd sizes padded to even).
_VIDEO_OUTPUT_PARAMS = [
    "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
]

_REPLAY_DIR = "replay"
_OUTPUTS = ("frames", "frames.json", "replay.mp4", "replay.png")


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "record_render needs Pillow to read and write frames: "
            "pip install pillow (or mlarena-sdk[video])"
        ) from exc
    return Image


def _to_rgb_image(image):
    """The frame as an RGB PIL image: a numpy array (H,W[,3|4]), a PIL
    image, or a base64 PNG string with or without its data-URI prefix — the
    inputs the platform's `save_render` accepts."""
    Image = _pil()
    if isinstance(image, Image.Image):
        pil = image
    elif hasattr(image, "__array_interface__"):
        pil = Image.fromarray(image)
    elif isinstance(image, str):
        data = image.split(",", 1)[1] if image.startswith("data:image/") else image
        pil = Image.open(io.BytesIO(base64.b64decode(data)))
        pil.load()
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Expected numpy array, PIL Image, or base64 string."
        )
    return pil if pil.mode == "RGB" else pil.convert("RGB")


def _plain(value):
    """A numpy scalar as its Python value (`.item()`), anything else as is."""
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return item()
    return value


def _labels(cum_rewards) -> dict:
    """`cum_rewards` as the platform stores it: `{str: int | float | str}`;
    another value type is a `TypeError` at the call."""
    if not cum_rewards:
        return {}
    if not isinstance(cum_rewards, dict):
        raise TypeError(f"cum_rewards must be a dict, got {type(cum_rewards).__name__}")
    out = {}
    for key, value in cum_rewards.items():
        value = _plain(value)
        if not isinstance(value, (int, float, str)):
            raise TypeError(
                f"cum_rewards[{key!r}] must be int, float or str, "
                f"got {type(value).__name__}"
            )
        out[str(key)] = value
    return out


class LocalRender:
    """What `record_render` attached to an env: the platform's
    `RenderCollector` rules, with files instead of an upload.

    The first call fixes the mode: `frames` (`save_render` /
    `save_render_frames`) or `video` (`start_render_video`); a call of the
    other mode raises `RuntimeError`, as `save_render` and
    `save_render_frames` do between themselves.

    Frames: identical images are stored once; past `replay_frames_max`
    frames, every other kept frame is dropped and its time given to its
    neighbour, so the whole episode and its played time survive (the last
    frame is always kept). Video: `fps` frames per second, cut after
    `replay_video_second_max` seconds with one warning.

    `finish()` (or leaving the `with` block without an error) writes
    `out_dir/replay/`: `replay.png` when exactly one frame has an image,
    else `frames/0001.png…` + `frames.json`; `replay.mp4` in video mode.
    """

    def __init__(self, out_dir, *, render_delay_second=RENDER_DELAY_SECOND,
                 replay_frames_max=REPLAY_FRAMES_MAX,
                 replay_video_second_max=REPLAY_VIDEO_SECOND_MAX):
        if not render_delay_second > 0:
            # The platform's bound (ck_challenge_configuration_render_delay_second).
            raise ValueError("render_delay_second must be > 0")
        if replay_frames_max < 1 or replay_video_second_max < 1:
            raise ValueError("replay caps must be > 0")
        self.dir = Path(out_dir) / _REPLAY_DIR
        self._delay = float(render_delay_second)
        self._frames_max = int(replay_frames_max)
        self._video_second_max = int(replay_video_second_max)
        self.render_format = None
        self.path = None
        self._reset()

    def _reset(self):
        self._mode = None            # None | "incremental" | "bulk" | "video"
        self._images: dict[bytes, bytes] = {}   # sha1 of RGB bytes -> PNG
        self._frames: list[dict] = []
        self._stride = 1             # input frames per kept frame
        self._pending = None         # the open group of `_stride` inputs
        self._step = 0
        self._roles: dict = {}
        self._fps = None
        self._video_writer = None
        self._video_size = None
        self._video_frames = 0
        self._video_cut = False
        self._finished = False
        partial = self.dir / "replay.partial.mp4"
        if partial.exists():
            partial.unlink()

    # -- the injected API ---------------------------------------------------

    def save_render(self, image, cum_rewards=None, *, duration_s=None):
        """One frame. `image=None` keeps step alignment; `cum_rewards` labels
        it (`{key: int | float | str}`); `duration_s` (> 0) how long it is
        shown, else the challenge's default frame duration."""
        if self._mode == "video":
            if duration_s is not None:
                raise ValueError("duration_s has no meaning in a video replay (fixed fps)")
            if image is not None:
                self._video_frame(image)
            return
        self._enter("incremental")
        self._add(image, cum_rewards, duration_s, self._step)
        self._step += 1

    def save_render_frames(self, images):
        """Every frame at once, replacing what was recorded."""
        self._enter("bulk")
        self._images, self._frames, self._stride, self._pending = {}, [], 1, None
        for step, image in enumerate(images):
            if image is None:
                raise TypeError("save_render_frames takes images, not None")
            self._add(image, None, None, step)

    def save_render_roles(self, roles):
        """`cum_rewards` key → agent index (the seat whose live reward it
        is). The last call wins; unmapped keys stay unmapped."""
        if not isinstance(roles, dict):
            raise TypeError(f"roles must be a dict, got {type(roles).__name__}")
        out = {}
        for key, index in roles.items():
            index = _plain(index)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise TypeError(f"roles[{key!r}] must be an agent index (int >= 0), got {index!r}")
            out[str(key)] = index
        self._roles = out

    def start_render_video(self, fps=30):
        """Later `save_render` calls feed an H.264 video at `fps` (1..60)."""
        self._enter("video")
        if self._fps is not None:
            raise RuntimeError("start_render_video() was already called")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 1 <= fps <= 60:
            raise ValueError(f"fps must be between 1 and 60, got {fps!r}")
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "video replays need imageio-ffmpeg: pip install \"mlarena-sdk[video]\""
            ) from exc
        self._fps = fps

    def clear_render(self):
        """Forget everything recorded (e.g. a run restarted on a fallback
        path), so the replay matches the run that is scored."""
        if self._video_writer is not None:
            self._video_writer.close()
        self._reset()

    # -- frames -------------------------------------------------------------

    def _enter(self, mode):
        if self._mode is None:
            self._mode = mode
        elif self._mode != mode:
            raise RuntimeError(
                "Cannot mix save_render(), save_render_frames() and "
                "start_render_video(). Use one API per run."
            )

    def _add(self, image, cum_rewards, duration_s, step):
        if duration_s is not None:
            duration_s = _plain(duration_s)
            if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) \
                    or not duration_s > 0:
                raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
            duration_s = float(duration_s)
        # Time is kept as explicit seconds + a count of default-duration
        # inputs, so merging never loses played time (`_duration`).
        frame = {
            "image": None if image is None else self._store(image),
            "step": int(step),
            "cum_rewards": _labels(cum_rewards),
            "_sec": duration_s or 0.0,
            "_nulls": 0 if duration_s is not None else 1,
            "_inputs": 1,
        }
        # A kept frame stands for `_stride` inputs: the group's last input,
        # shown for the group's whole time.
        if self._pending is not None:
            frame = self._merge(self._pending, frame)
        self._pending = frame
        if frame["_inputs"] >= self._stride:
            self._keep(frame)
            self._pending = None

    @staticmethod
    def _merge(a, b):
        """`b` shown for the time of both."""
        return dict(b, _sec=a["_sec"] + b["_sec"], _nulls=a["_nulls"] + b["_nulls"],
                    _inputs=a["_inputs"] + b["_inputs"])

    def _duration(self, frame):
        """`duration_s` as stored: null for one default-duration input,
        else seconds (default-duration inputs at `render_delay_second`)."""
        if frame["_inputs"] == 1 and frame["_nulls"] == 1:
            return None
        return frame["_sec"] + frame["_nulls"] * self._delay

    def _keep(self, frame):
        self._frames.append(frame)
        if len(self._frames) > self._frames_max:
            self._decimate()

    def _decimate(self):
        """Drop every other kept frame, its time going to the next one (so
        the last frame always stays), and double the stride."""
        frames = self._frames
        start = len(frames) % 2   # an odd count keeps the first frame alone
        kept = frames[:start]
        for i in range(start, len(frames), 2):
            kept.append(self._merge(frames[i], frames[i + 1]))
        self._frames = kept
        self._stride *= 2

    def _store(self, image) -> str:
        pil = _to_rgb_image(image)
        raw = pil.tobytes()
        key = hashlib.sha1(f"{pil.width}x{pil.height}:".encode() + raw).digest()
        if key not in self._images:
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG")
            self._images[key] = buffer.getvalue()
        return key

    # -- video --------------------------------------------------------------

    def _video_frame(self, image):
        pil = _to_rgb_image(image)
        if self._video_size is None:
            self._video_size = pil.size
        elif pil.size != self._video_size:
            raise ValueError(
                f"video frame of {pil.size[0]}x{pil.size[1]} differs from the "
                f"first frame's {self._video_size[0]}x{self._video_size[1]}"
            )
        if self._video_frames >= int(self._fps * self._video_second_max):
            if not self._video_cut:
                self._video_cut = True
                warnings.warn(
                    f"video replay cut at {self._video_second_max} s "
                    f"({self._video_frames} frames at {self._fps} fps); later frames dropped",
                    RuntimeWarning, stacklevel=3,
                )
            return
        if self._video_writer is None:
            import imageio_ffmpeg
            self.dir.mkdir(parents=True, exist_ok=True)
            self._video_writer = imageio_ffmpeg.write_frames(
                str(self.dir / "replay.partial.mp4"), self._video_size, fps=self._fps,
                codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
                output_params=list(_VIDEO_OUTPUT_PARAMS),
            )
            self._video_writer.send(None)
        self._video_writer.send(pil.tobytes())
        self._video_frames += 1

    # -- output -------------------------------------------------------------

    def finish(self):
        """Write what was recorded; returns the written file's `Path`
        (`frames.json`, `replay.mp4` or `replay.png`), None when nothing was
        recorded. `render_format` names the format the platform would store.
        A second call returns the first one's path."""
        if self._finished:
            return self.path
        self._finished = True
        if self._pending is not None:   # the last input always stays
            self._keep(self._pending)
            self._pending = None
            while len(self._frames) > self._frames_max:
                self._decimate()
        self._clear_outputs()
        if self._mode == "video":
            if self._video_writer is None:
                return self._done(None, None)
            self._video_writer.close()
            self._video_writer = None
            final = self.dir / "replay.mp4"
            os.replace(self.dir / "replay.partial.mp4", final)
            return self._done("video_mp4", final)
        drawn = [f for f in self._frames if f["image"] is not None]
        if not drawn:
            return self._done(None, None)
        self.dir.mkdir(parents=True, exist_ok=True)
        if len(drawn) == 1:   # one still: its labels give way to the score
            final = self.dir / "replay.png"
            final.write_bytes(self._images[drawn[0]["image"]])
            return self._done("image_png", final)
        frames_dir = self.dir / "frames"
        frames_dir.mkdir()
        names: dict[bytes, str] = {}
        rows = []
        for frame in self._frames:
            name = None
            if frame["image"] is not None:
                name = names.get(frame["image"])
                if name is None:
                    name = f"frames/{len(names) + 1:04d}.png"
                    names[frame["image"]] = name
                    (self.dir / name).write_bytes(self._images[frame["image"]])
            rows.append({"image": name, "step": frame["step"],
                         "duration_s": self._duration(frame),
                         "cum_rewards": frame["cum_rewards"]})
        final = self.dir / "frames.json"
        final.write_text(json.dumps({
            "render_format": "frames_v2",
            "render_delay_second": self._delay,
            "roles": self._roles,
            "frames": rows,
        }, indent=1))
        return self._done("frames_v2", final)

    def _done(self, render_format, path):
        self.render_format, self.path = render_format, path
        return path

    def _clear_outputs(self):
        for name in _OUTPUTS:
            target = self.dir / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()
        elif self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        return False


def record_render(env, out_dir, *, render_delay_second=RENDER_DELAY_SECOND,
                  replay_frames_max=REPLAY_FRAMES_MAX,
                  replay_video_second_max=REPLAY_VIDEO_SECOND_MAX) -> LocalRender:
    """Attach the platform's replay API to `env`, as the executor does
    before `evaluate`, and return the `LocalRender` that collects it.

    Injected: `save_render(image, cum_rewards=None, *, duration_s=None)`,
    `save_render_frames(images)`, `save_render_roles(roles)`,
    `start_render_video(fps=30)` and `clear_render()`, with the platform's
    mode, label-type and budget rules. The keywords default to a default
    machine's caps and the challenge default frame duration; pass the
    challenge's own values to preview exactly what it would store. Call
    `.finish()` after `evaluate` (or use it as a context manager) to write
    `out_dir/replay/`.
    """
    recorder = LocalRender(out_dir, render_delay_second=render_delay_second,
                           replay_frames_max=replay_frames_max,
                           replay_video_second_max=replay_video_second_max)
    env.save_render = recorder.save_render
    env.save_render_frames = recorder.save_render_frames
    env.save_render_roles = recorder.save_render_roles
    env.start_render_video = recorder.start_render_video
    env.clear_render = recorder.clear_render
    return recorder
