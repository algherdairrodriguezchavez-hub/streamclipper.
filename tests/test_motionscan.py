"""Motion scan: quiet visual moments become candidates."""

import json
import subprocess

import pytest

from streamclipper import motionscan, visual_rank


def _scan(n=900, busy=(), step=2.0, base=3.0, hot=20.0):
    t = [i * step for i in range(1, n)]
    d = [hot if any(a <= x < b for a, b in busy) else base for x in t]
    return {"t": t, "d": d, "y": [90.0] * len(t)}


def test_moments_find_busy_stretches_above_the_usual_motion():
    scan = _scan(busy=((600, 640), (1500, 1530)))
    got = motionscan.moments(scan, 1800.0, 90.0)
    ts = sorted(m["t"] for m in got)
    assert len(ts) == 2
    assert 600 <= ts[0] <= 640 and 1500 <= ts[1] <= 1530


def test_moments_ignore_a_stream_that_is_always_busy_and_dark_bins():
    assert motionscan.moments(_scan(base=20.0), 1800.0, 90.0) == []
    scan = _scan(busy=((600, 640),))
    scan["y"] = [5.0 if 600 <= x < 640 else 90.0 for x in scan["t"]]  # fade to black
    assert motionscan.moments(scan, 1800.0, 90.0) == []


def test_moments_keep_separation_and_cap():
    busy = tuple((a, a + 20) for a in range(100, 3500, 200))
    got = motionscan.moments(_scan(n=1800, busy=busy), 3600.0, 300.0)
    assert len(got) <= motionscan.PER_HOUR
    ts = sorted(m["t"] for m in got)
    assert all(b - a >= 300 for a, b in zip(ts, ts[1:], strict=False))


def test_motion_stage_scans_once_and_reuses_the_cache(tmp_path, monkeypatch):
    src = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=12", "-g", "10", "-y", str(src)],
        check=True,
    )
    got = motionscan.motion_stage(src, "", tmp_path, 12.0, 30.0)
    assert isinstance(got, list)
    saved = json.loads((tmp_path / "motion_scan.json").read_text())
    assert len(saved["scan"]["t"]) >= 3
    monkeypatch.setattr(motionscan, "scan_video", lambda *a, **k: pytest.fail("scanned twice"))
    motionscan.motion_stage(src, "", tmp_path, 12.0, 30.0)


def test_rerank_keeps_slots_for_quiet_motion_clips():
    clips = [{"start": i * 100.0, "end": i * 100.0 + 60, "score": 90 - i, "sources": ["voice"]} for i in range(30)]
    clips[27]["sources"] = ["motion"]
    rated = visual_rank._to_rate(clips)
    assert len(rated) == visual_rank.TOP_N
    assert clips[27] in rated and clips[19] not in rated


def test_a_quiet_motion_moment_becomes_a_candidate_and_passes_at_half_the_bar():
    import numpy as np

    from streamclipper.audio import find_peaks
    from streamclipper.discovery import build_candidates, select

    n = 20000
    env = {"vox": np.full(n, -40.0), "full": np.full(n, -35.0), "dur": 200.0}
    env["vox"][::7] = -38.0
    assert find_peaks(env, 0, 20.0) == []
    islands = [{"speech_start": a, "speech_end": a + 3.0} for a in range(5, 195, 6)]
    st = {"min_dur": 15.0, "max_dur": 60.0, "target_dur": 30.0, "min_pause": 0.55}
    mv = [{"t": 120.0, "motion": 20.0, "ratio": 5.0}]
    args = (env, [], islands, [], [], None, [], [], None, st, 200.0)
    assert build_candidates(*args, transcribed=False) == []
    cands = build_candidates(*args, transcribed=False, motion=mv)
    assert len(cands) == 1 and cands[0]["sources"] == ["motion"]
    assert cands[0]["start"] <= 120.0 <= cands[0]["end"]
    c = cands[0]
    bar = max(1.0, float(c["score"]) * 2 - 1)  # the full bar would drop it; half the bar keeps it
    assert select(cands, dict(st, min_score=bar))
    assert select(cands, dict(st, min_score=c["score"] * 2 + 2)) == []
