"""`mlarena.local.record_render`: the platform's replay API on a local env,
writing reviewable files. Needs Pillow (and imageio-ffmpeg for video, the
`video` extra); skipped without them. Offline."""
import json
import os
import sys
import tempfile
import warnings

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

Image = pytest.importorskip("PIL.Image")

from mlarena.local import eval_context, record_render  # noqa: E402


class Env:
    pass


def _img(value, size=(4, 3)):
    return Image.new("RGB", size, (value, 0, 0))


def _record(**caps):
    env = Env()
    out = tempfile.mkdtemp()
    return env, record_render(env, out, **caps), out


def test_injects_the_platform_api():
    env, _, _ = _record()
    for name in ("save_render", "save_render_frames", "save_render_roles",
                 "start_render_video", "clear_render"):
        assert callable(getattr(env, name)), name


def test_frames_dedupe_labels_roles_and_alignment_frames():
    env, rec, out = _record()
    env.save_render_roles({"player_0": 1, "player_1": 0})
    env.save_render(_img(1), {"player_0": 0, "player_1": 0})
    env.save_render(None)                                   # alignment frame
    env.save_render(_img(1), {"player_0": 1.5, "player_1": "not scored yet"})
    env.save_render(_img(2), duration_s=2.0)
    path = rec.finish()
    assert rec.render_format == "frames_v2"
    assert str(path) == os.path.join(out, "replay", "frames.json")
    doc = json.loads(path.read_text())
    assert doc["roles"] == {"player_0": 1, "player_1": 0}
    assert [f["image"] for f in doc["frames"]] == [
        "frames/0001.png", None, "frames/0001.png", "frames/0002.png"]
    assert [f["step"] for f in doc["frames"]] == [0, 1, 2, 3]
    assert [f["duration_s"] for f in doc["frames"]] == [None, None, None, 2.0]
    assert doc["frames"][2]["cum_rewards"] == {"player_0": 1.5, "player_1": "not scored yet"}
    assert sorted(os.listdir(os.path.join(out, "replay", "frames"))) == ["0001.png", "0002.png"]


def test_one_still_is_an_image_png():
    env, rec, out = _record()
    env.save_render(None)
    env.save_render(_img(9), {"score": 1.0})
    path = rec.finish()
    assert rec.render_format == "image_png" and path.name == "replay.png"
    assert Image.open(path).size == (4, 3)
    assert not os.path.exists(os.path.join(out, "replay", "frames.json"))


def test_nothing_recorded_writes_nothing():
    env, rec, out = _record()
    assert rec.finish() is None and rec.render_format is None
    assert not os.path.exists(os.path.join(out, "replay", "frames.json"))


def test_decimation_keeps_the_last_frame_and_the_played_time():
    env, rec, _ = _record(replay_frames_max=10, render_delay_second=0.5)
    for i in range(37):
        env.save_render(_img(i), {"t": i})
    doc = json.loads(rec.finish().read_text())
    frames = doc["frames"]
    assert len(frames) <= 10
    assert frames[-1]["step"] == 36 and frames[-1]["cum_rewards"] == {"t": 36}
    played = sum(0.5 if f["duration_s"] is None else f["duration_s"] for f in frames)
    assert played == pytest.approx(37 * 0.5)


def test_decimation_adds_explicit_and_default_durations():
    env, rec, _ = _record(replay_frames_max=2, render_delay_second=0.1)
    env.save_render(_img(1), duration_s=1.0)
    env.save_render(_img(2))
    env.save_render(_img(3), duration_s=3.0)
    frames = json.loads(rec.finish().read_text())["frames"]
    assert sum(f["duration_s"] or 0.1 for f in frames) == pytest.approx(4.1)
    assert frames[-1]["step"] == 2


def test_label_and_duration_types_raise_at_the_call():
    env, _, _ = _record()
    with pytest.raises(TypeError):
        env.save_render(_img(1), {"p": [1, 2]})
    with pytest.raises(ValueError):
        env.save_render(_img(1), duration_s=0)
    with pytest.raises(TypeError):
        env.save_render_roles({"player_0": "one"})


def test_modes_do_not_mix():
    env, _, _ = _record()
    env.save_render(_img(1))
    with pytest.raises(RuntimeError):
        env.save_render_frames([_img(1)])
    with pytest.raises(RuntimeError):
        env.start_render_video(30)


def test_bulk_frames_and_clear():
    env, rec, _ = _record()
    env.save_render_frames([_img(1), _img(2), _img(1)])
    env.clear_render()
    env.save_render(_img(5))
    env.save_render(_img(6))
    frames = json.loads(rec.finish().read_text())["frames"]
    assert [f["step"] for f in frames] == [0, 1]


def test_numpy_frames_and_labels():
    np = pytest.importorskip("numpy")
    env, rec, _ = _record()
    env.save_render(np.zeros((3, 4, 3), dtype=np.uint8), {"r": np.float32(0.5)})
    env.save_render(np.full((3, 4, 3), 200, dtype=np.uint8), {"r": np.int64(2)})
    frames = json.loads(rec.finish().read_text())["frames"]
    assert frames[1]["cum_rewards"] == {"r": 2}


def test_eval_context_keeps_the_executors_frozen_is_render():
    ctx = eval_context(tempfile.mkdtemp(), is_evaluation=False)
    assert ctx["is_render"] is False and ctx["is_evaluation"] is False


# -- video ---------------------------------------------------------------- #

def test_video_round_trip_with_the_platform_profile():
    pytest.importorskip("imageio_ffmpeg")
    env, rec, _ = _record()
    env.start_render_video(fps=30)
    for i in range(30):
        env.save_render(_img(i * 8, size=(33, 21)), {"ignored": 1})
    with pytest.raises(ValueError):
        env.save_render(_img(0, size=(33, 21)), duration_s=1.0)
    with pytest.raises(ValueError):
        env.save_render(_img(0, size=(32, 20)))
    path = rec.finish()
    assert rec.render_format == "video_mp4" and path.name == "replay.mp4"
    data = path.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")      # +faststart


def test_video_is_cut_at_the_cap():
    pytest.importorskip("imageio_ffmpeg")
    env, rec, _ = _record(replay_video_second_max=1)
    env.start_render_video(fps=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(5):
            env.save_render(_img(i * 40, size=(16, 16)))
    assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1
    assert rec._video_frames == 2
    assert rec.finish().name == "replay.mp4"


def test_fps_range():
    pytest.importorskip("imageio_ffmpeg")
    env, _, _ = _record()
    with pytest.raises(ValueError):
        env.start_render_video(fps=0)
