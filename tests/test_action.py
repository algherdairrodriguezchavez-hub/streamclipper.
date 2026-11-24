"""Gameplay crop that follows the action: motion columns -> smoothed, speed-limited crop path -> keyframes."""

import subprocess

import numpy as np
import pytest

from streamclipper import action, camfit, reframe


def _row(centre, w=320, spread=6):
    r = np.zeros(w, dtype=np.float32)
    c = int(centre * w)
    r[max(0, c - spread) : c + spread] = 20.0
    return r


def test_best_x_centres_the_crop_on_the_motion_and_respects_the_range():
    x, m = action.best_x(_row(0.65), 0.49)
    assert abs(x + 0.245 - 0.65) < 0.02 and m > action.MIN_MOTION
    assert action.best_x(_row(0.95), 0.49)[0] == pytest.approx(0.51)  # never past the frame edge
    x, _ = action.best_x(_row(0.2), 0.49, lo=0.33)
    assert x == pytest.approx(0.33)


def test_path_is_speed_limited_and_still_scenes_stay():
    times = [0.0, 1.5, 3.0, 4.5]
    cols = np.stack([_row(0.3), _row(0.8), _row(0.8), np.zeros(320, dtype=np.float32)])
    path = action.action_path(times, cols, 0.49, 0.25)
    xs = [x for _, x in path]
    assert xs[0] == pytest.approx(0.055, abs=0.02)  # the first position goes straight to the action
    assert xs[1] - xs[0] <= action.MAX_SPEED * 1.5 + 1e-6  # then at most MAX_SPEED per second
    assert xs[3] == xs[2]  # no motion: the crop stays


def test_allowed_range_keeps_clear_of_a_tall_cam_overlay():
    lo, hi = action.allowed_range(0.49, [[0.0, 0.44, 0.31, 1.0]])
    assert lo == pytest.approx(0.31) and hi == pytest.approx(0.51)
    assert action.allowed_range(0.49, [[0.0, 0.0, 0.8, 1.0]]) == (0.0, pytest.approx(0.51))  # no room: ignored


def test_follow_segments_adds_keyframes_that_track_a_moving_object(tmp_path):
    v = tmp_path / "g.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=15:d=12",
         "-f", "lavfi", "-i", "color=c=orange:s=50x30:r=15:d=12", "-filter_complex",
         "[0:v][1:v]overlay=x='40+45*t':y=200[v]", "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", str(v)],
        check=True,
    )  # fmt: skip
    game = [0.25, 0.0, 0.75, 1.0]
    segs = camfit.runs_to_segments([{"start": 0, "cam": None, "game": None}], 12.0, "9:16", 640, 360)
    segs[0]["windows"]["game"] = reframe.lock_aspect(camfit.norm_to_px(game, 640, 360), 1080 / 1920, 640, 360)
    out, note = action.follow_segments(segs, v, 640, 360)
    assert "follows the action" in note
    xs = [k["game"][0] for k in out[0]["keys"]]
    assert xs == sorted(xs) and xs[-1] > xs[0]  # the object drives right; so does the crop
    reframe.validate_segments(out, {"width": 640, "height": 360, "ratio": "9:16"}, 0.0, 12.0)


def test_box_fit_shows_the_whole_cam_overlay():
    face, overlay = [0.1, 0.6, 0.22, 0.84], [0.02, 0.44, 0.31, 0.96]
    box = camfit.fit_cam_box(face, overlay, "box", "9:16", 1920, 1080)
    w, h = (box[2] - box[0]) * 1920, (box[3] - box[1]) * 1080
    assert abs(w / h - 1080 / 672) < 0.02 and box[0] >= overlay[0] and box[2] <= overlay[2] + 1e-6
    assert w == pytest.approx((overlay[2] - overlay[0]) * 1920, rel=0.01)  # full overlay width
