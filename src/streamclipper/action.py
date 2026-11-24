"""
Gameplay crop that follows the action: the 9:16 gameplay panel shows only part of the 16:9 stream, so its
crop moves sideways to where things happen (characters, cars, shooting), measured as frame-to-frame
motion, and never jumps: the path is smoothed, speed-limited and becomes a few reframe keyframes.

The streamer's cam overlay (and the whole streamer, for a keyed cam) is masked out, so the crop never
chases the streamer's own movement. A still scene (menus, a parked car) keeps the crop where it was.
"""

from __future__ import annotations

from pathlib import Path

from streamclipper.audio import _np
from streamclipper.util import run_cmd

SAMPLE_FPS = 4.0  # motion samples per second
SAMPLE_W = 320  # analysis width (px)
BIN_S = 1.5  # motion is pooled over bins of this length before the crop position is chosen
MAX_SPEED = 0.12  # the crop moves at most this share of the frame width per second
KEY_EVERY = 3.0  # seconds between keyframes (reframe allows 12 per segment)
MIN_MOTION = 0.3  # mean abs difference (0-255) per crop column below which a bin counts as still
CENTRE_BIAS = 0.15  # pull towards the frame centre (share of the motion score), so ties stay centred
DEAD_ZONE = 0.03  # moves smaller than this share of the width are ignored (no wobble)


def _gray_frames(video: Path, s: float, e: float, w: int, h: int):
    np = _np()
    res = run_cmd(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, s):.3f}", "-t", f"{max(0.1, e - s):.3f}",
         "-i", str(video), "-an", "-vf", f"fps={SAMPLE_FPS},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        "motion analysis frames", binary=True, check=False,
    )  # fmt: skip
    buf = res.stdout or b""
    n = len(buf) // (w * h)
    if n < 2:
        return None
    return np.frombuffer(buf[: n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)


def motion_columns(video: Path, s: float, e: float, vw: int, vh: int, masks: list = ()):
    """(times, cols): per bin of BIN_S seconds, the motion of each analysis column (masked boxes zeroed)."""
    np = _np()
    w = SAMPLE_W
    h = max(2, int(round(w * vh / vw / 2)) * 2)
    fr = _gray_frames(video, s, e, w, h)
    if fr is None:
        return None, None
    diff = np.abs(np.diff(fr, axis=0))  # (n-1, h, w)
    for b in masks or ():
        if b:
            x0, y0, x1, y1 = int(b[0] * w), int(b[1] * h), int(np.ceil(b[2] * w)), int(np.ceil(b[3] * h))
            diff[:, max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 0.0
    per = diff.mean(axis=1)  # (n-1, w) motion per column
    k = max(1, int(round(BIN_S * SAMPLE_FPS)))
    nb = max(1, per.shape[0] // k)
    cols = per[: nb * k].reshape(nb, k, w).mean(axis=1)
    times = [s + (i + 0.5) * k / SAMPLE_FPS for i in range(nb)]
    return times, cols


def best_x(cols_row, crop_w: float, lo: float = 0.0, hi: float = None) -> tuple:
    """(left edge 0..1, motion) of the crop of width crop_w (share of the width) centred on the motion: the
    crop window holding the most motion, then centred on the motion-weighted centre inside it; left edge
    kept in [lo, hi]."""
    np = _np()
    w = cols_row.shape[0]
    hi = (1.0 - crop_w) if hi is None else hi
    cw = max(1, min(w, int(round(crop_w * w))))
    c = np.concatenate([[0.0], np.cumsum(cols_row)])
    sums = c[cw:] - c[: w - cw + 1]  # motion inside each crop position
    lefts = np.arange(sums.shape[0]) / w
    centre = 1.0 - np.abs(lefts + crop_w / 2 - 0.5) * 2  # 1 in the middle, 0 at the edges
    i = int((sums * (1.0 + CENTRE_BIAS * centre)).argmax())
    part = cols_row[i : i + cw]
    tot = float(part.sum())
    cx = (i + float((np.arange(cw) * part).sum()) / tot) / w if tot > 0 else lefts[i] + crop_w / 2
    x = min(max(cx - crop_w / 2, lo), max(lo, hi))
    return float(x), float(sums[i] / cw)


def allowed_range(crop_w: float, masks: list) -> tuple:
    """[lo, hi] for the crop's left edge so it stays clear of tall keep-out boxes (the cam overlay, the
    streamer) when the frame leaves room; boxes that leave no room are ignored."""
    lo, hi = 0.0, 1.0 - crop_w
    for b in masks or ():
        if not b or b[3] - b[1] < 0.3:
            continue
        if (b[0] + b[2]) / 2 < 0.5 and b[2] <= hi:
            lo = max(lo, b[2])
        elif (b[0] + b[2]) / 2 >= 0.5 and b[0] - crop_w >= lo:
            hi = min(hi, b[0] - crop_w)
    return (lo, hi) if lo <= hi else (0.0, 1.0 - crop_w)


def action_path(times: list, cols, crop_w: float, start_x: float, masks: list = ()) -> list:
    """[(t, x)] smoothed and speed-limited left edges of the gameplay crop (shares of the width)."""
    lo, hi = allowed_range(crop_w, masks)
    pts, x = [], min(max(start_x, lo), hi)
    last_t = times[0] if times else 0.0
    for t, row in zip(times, cols, strict=False):
        target, motion = best_x(row, crop_w, lo, hi)
        if motion < MIN_MOTION:
            target = x  # still scene: stay
        if abs(target - x) < DEAD_ZONE:
            target = x
        step = MAX_SPEED * max(0.0, t - last_t) if pts else 1.0  # the first position is free
        x = x + max(-step, min(step, target - x))
        x = min(max(0.0, x), 1.0 - crop_w)
        pts.append((t, x))
        last_t = t
    return pts


def follow_segments(segs: list, video: Path, vw: int, vh: int, masks: list = ()) -> tuple:
    """
    Add keyframes to each segment's game window so it follows the action (source time = window file time).
    Returns (segments, note). Segments without enough motion are left as they are.
    """
    from streamclipper.reframe import MAX_KEYS, MIN_SEG

    out, moved = [], 0
    for seg in segs:
        seg = dict(seg)
        game = (seg.get("windows") or {}).get("game")
        a, b = float(seg["start"]), float(seg["end"])
        if not game or b - a < 2 * MIN_SEG:
            out.append(seg)
            continue
        cam = (seg.get("windows") or {}).get("cam")
        seg_masks = list(masks or ())
        if cam:
            seg_masks.append([cam[0] / vw, cam[1] / vh, (cam[0] + cam[2]) / vw, (cam[1] + cam[3]) / vh])
        times, cols = motion_columns(video, a, b, vw, vh, seg_masks)
        if not times:
            out.append(seg)
            continue
        gx, gy, gw, gh = game
        crop_w = gw / vw
        path = action_path(times, cols, crop_w, gx / vw, seg_masks)
        xs = [p[1] for p in path]
        if max(xs) - min(xs) < DEAD_ZONE and abs(xs[0] - gx / vw) < DEAD_ZONE:
            out.append(seg)
            continue
        step = max(KEY_EVERY, (b - a) / MAX_KEYS)
        keys, next_t = [], a
        for t, x in path:
            if t >= next_t and a + 1e-3 < t < b - 1e-3:
                win = dict(seg["windows"])
                win["game"] = [min(int(round(x * vw)), vw - gw), gy, gw, gh]
                keys.append({"t": round(t, 3), **{k: v for k, v in win.items() if v}})
                next_t = t + step
        if keys:
            first = dict(seg["windows"])
            first["game"] = [min(int(round(path[0][1] * vw)), vw - gw), gy, gw, gh]
            seg["windows"] = first
            seg["keys"] = keys[:MAX_KEYS]
            seg["ease"] = "smooth"
            moved += 1
        out.append(seg)
    note = f"gameplay follows the action in {moved} part(s)" if moved else "gameplay crop kept (no clear action)"
    return out, note
