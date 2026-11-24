"""
Fast motion scan of the whole stream: finds visual moments (chases, crashes, shootouts) even when the
streamer stays quiet, so they can become candidates next to the voice peaks.

One pass over the keyframes (about one frame every 2 s, 160 px wide, grey) of the local video or of a
144p video-only download of the link; the scan is saved in <work_dir>/motion_scan.json and reused.
A moment is a stretch whose motion is well above the stream's usual motion around it (a rolling median),
so a stream that is always busy does not flag everything and a calm one still finds its busy parts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from streamclipper.audio import _np
from streamclipper.state import write_json_atomic
from streamclipper.util import fwd

VERSION = 1
SCAN_W = 160  # analysis width (px)
SCAN_STEP = 2.0  # seconds between sampled keyframes
SCAN_H = 144  # height of the video-only download for links
BIN_S = 10.0  # motion is pooled over bins of this length (median: a single scene cut does not count)
BASE_S = 300.0  # the usual motion is the median over this many seconds around a bin
MIN_RATIO = 1.8  # a moment moves this many times more than usual
MIN_ABS = 6.0  # and at least this much (mean abs grey difference, 0-255)
DARK = 18.0  # bins this dark (loading or fade to black) are skipped
PER_HOUR = 3  # moments kept per stream hour (at most MAX_MOMENTS)
MAX_MOMENTS = 12


def scan_video(video: Path, step: float = SCAN_STEP, width: int = SCAN_W) -> dict:
    """{'t': [...], 'd': [...], 'y': [...]}: per sampled keyframe its time, the motion since the previous one
    and its mean brightness."""
    from streamclipper.hud.frames import keyframes

    np = _np()
    ts, ds, ys, prev = [], [], [], None
    for t, rgb in keyframes(Path(video), width=width, step=step):
        g = rgb.astype(np.float32).mean(axis=2)
        if prev is not None and g.shape == prev.shape:
            ts.append(round(float(t), 2))
            ds.append(round(float(np.abs(g - prev).mean()), 3))
            ys.append(round(float(g.mean()), 1))
        prev = g
    return {"t": ts, "d": ds, "y": ys}


def moments(scan: dict, dur: float, sep: float) -> list:
    """[{'t', 'motion', 'ratio'}] best first: bins well above the rolling median motion, at least sep apart."""
    np = _np()
    t = np.asarray(scan.get("t") or [], dtype=float)
    if t.size < 8:
        return []
    d = np.asarray(scan["d"], dtype=float)
    y = np.asarray(scan.get("y") or [128.0] * t.size, dtype=float)
    nb = int(max(t.max(), dur) // BIN_S) + 1
    idx = np.minimum((t // BIN_S).astype(int), nb - 1)
    val = np.full(nb, np.nan)
    for b in np.unique(idx):
        sel = idx == b
        if y[sel].mean() >= DARK:
            val[b] = float(np.median(d[sel]))
    half = max(1, int(BASE_S / BIN_S / 2))
    found = []
    for b in range(nb):
        if np.isnan(val[b]):
            continue
        around = val[max(0, b - half) : b + half + 1]
        base = float(np.nanmedian(around)) if np.any(~np.isnan(around)) else 0.0
        ratio = val[b] / max(base, 0.5)
        if val[b] >= MIN_ABS and ratio >= MIN_RATIO:
            found.append(
                {"t": round((b + 0.5) * BIN_S, 1), "motion": round(float(val[b]), 2), "ratio": round(ratio, 2)}
            )
    found.sort(key=lambda m: -m["ratio"] * m["motion"])
    keep = []
    cap = min(MAX_MOMENTS, max(2, int(round(PER_HOUR * dur / 3600.0))))
    for m in found:
        if all(abs(m["t"] - k["t"]) >= sep for k in keep):
            keep.append(m)
        if len(keep) >= cap:
            break
    return keep


def _video_for(source: Path, url: str, work_dir: Path):
    from streamclipper.hud.frames import has_video

    if source is not None and has_video(source):
        return Path(source)
    hud360 = [p for p in (Path(work_dir) / "hud").glob("analysis_*p.*") if not p.name.endswith((".part", ".ytdl"))]
    if hud360:
        return hud360[0]  # the HUD pass already downloaded a low-res video
    if not url:
        return None
    from streamclipper.download import download_lowres_video

    return download_lowres_video(url, Path(work_dir) / "motion", SCAN_H)


def motion_stage(source: Path, url: str, work_dir: Path, dur: float, sep: float) -> list:
    """The stream's motion moments (cached in work_dir/motion_scan.json); [] when no video can be read."""
    work_dir = Path(work_dir)
    cache_p = work_dir / "motion_scan.json"
    key = f"v{VERSION}:{url or Path(source).name}:{SCAN_STEP:g}:{SCAN_W}"
    try:
        saved = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.is_file() else {}
    except ValueError:
        saved = {}
    if saved.get("key") == key and "scan" in saved:
        scan = saved["scan"]
        print(f"Resuming: motion scan already done ({len(scan['t'])} frames)")
    else:
        try:
            video = _video_for(source, url, work_dir)
        except Exception as e:  # noqa: BLE001 - the scan only adds candidates; the audio ones remain
            print(f"Motion scan skipped ({type(e).__name__}: {str(e)[:160]})")
            return []
        if video is None:
            print("Motion scan skipped: no video to read.")
            return []
        t0 = time.time()
        print(f"Motion scan of {video.name} (one frame every {SCAN_STEP:g} s) ...")
        scan = scan_video(video)
        write_json_atomic(
            cache_p, {"key": key, "video": fwd(video), "seconds": round(time.time() - t0, 1), "scan": scan}
        )
        print(f"Motion scan: {len(scan['t'])} frames in {time.time() - t0:.0f} s")
    found = moments(scan, dur, sep)
    if found:
        print(f"Motion scan: {len(found)} visual moment(s) " + ", ".join(f"{m['t'] / 60:.1f} min" for m in found[:6]))
    return found
