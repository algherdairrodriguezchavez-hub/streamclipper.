"""
Per-clip facecam and gameplay crops, on top of the face-based detection in vision.py / options.clip_layout:

- cam fit: how tight the facecam panel frames the face ('standard' keeps the 5.3 padding, 'tight' and 'close'
  crop closer, inside the detected cam overlay, so the face is bigger in the panel);
- cam tracking: a clip is checked in parts, and when the cam moves, turns on/off or changes size inside the clip,
  it is cut into segments with their own crops (rendered by the reframe panel, see reframe.py);
- streamer profiles: a JSON file per streamer with the cam positions seen before (a hint when detection fails);
- layout overrides: a JSON file of per-clip crops (written by --preview-only, edited by hand) that win over
  detection for those clips only;
- preview images: the source frame with the crops drawn on it next to a mock of the vertical clip.

Boxes: 'norm' = [x0, y0, x1, y1] in 0..1 of the frame; 'px' = [x, y, w, h] in pixels of the clip window.
"""

import hashlib
import json
import time
from pathlib import Path

from streamclipper import faces as _faces
from streamclipper.layout import _fill_crop, _norm_to_px, gameplay_box_avoiding
from streamclipper.state import write_json_atomic
from streamclipper.util import ToolError, run_cmd

CAM_FITS = ("standard", "tight", "close")
CAM_TRACKS = ("off", "auto")
FIT_WIDTH = {"tight": 3.0, "close": 2.3}  # cam crop width in face widths (Haar-sized face box)
FIT_HEIGHT = 1.6  # the cam crop is at least this many face heights tall (forehead room and the chin)
CAM_PANEL_W = 1080  # every canvas is 1080 wide: the cam crop is scaled to this width
MAX_CAM_UPSCALE = 2.5  # a tighter crop is widened so the face is never enlarged more than this (blur)
CAM_ASPECT = {"9:16": 1080 / 672, "1:1": 1080 / 454}
GAME_ASPECT = {"9:16": 1080 / 1248, "1:1": 1080 / 626}
FULL_ASPECT = {"9:16": 1080 / 1920, "1:1": 1.0}
MOVE_IOU = 0.6  # two cam boxes with less overlap than this are different cam positions
TRACK_PART_S = 12.0  # a clip is checked in parts of about this length (2 to 4 parts)
REFINE_STEPS = 5  # binary-search steps for the moment the cam moves (part length / 32)
PROFILE_SCHEMA = 1


# ---------------------------------------------------------------- box helpers


def iou(a, b) -> float:
    """Overlap of two normalised boxes (0..1)."""
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def px_to_norm(box, vw: int, vh: int) -> list:
    x, y, w, h = (float(v) for v in box)
    return [round(x / vw, 5), round(y / vh, 5), round((x + w) / vw, 5), round((y + h) / vh, 5)]


def norm_to_px(box, vw: int, vh: int) -> list:
    return list(_norm_to_px(box, vw, vh))


def _inside(face, box, margin: float = 0.0) -> bool:
    fw, fh = face[2] - face[0], face[3] - face[1]
    return (
        face[0] - margin * fw >= box[0] - 1e-6
        and face[1] - margin * fh >= box[1] - 1e-6
        and face[2] + margin * fw <= box[2] + 1e-6
        and face[3] + margin * fh <= box[3] + 1e-6
    )


def fit_cam_box(face: list, cam_box: list, fit: str, ratio: str, vw: int, vh: int) -> list:
    """
    The facecam crop (normalised) for cam fit 'tight' or 'close': FIT_WIDTH face widths wide (at least
    FIT_HEIGHT face heights tall), with the cam panel's aspect, the face a little above the middle, and kept
    inside the detected cam overlay cam_box so no gameplay shows next to the face. 'standard' (or a crop that
    would cut the face) returns cam_box unchanged. face is the detector's box (BlazeFace or Haar).
    """
    if fit not in FIT_WIDTH or not face or not cam_box:
        return cam_box
    aspect = CAM_ASPECT.get(ratio, CAM_ASPECT["9:16"])
    f = _faces.haar_sized(face) if getattr(_faces.get_detector(), "name", "") == "MediaPipe" else list(face)
    fx0, fy0, fx1, fy1 = f[0] * vw, f[1] * vh, f[2] * vw, f[3] * vh
    fw, fh = fx1 - fx0, fy1 - fy0
    w = FIT_WIDTH[fit] * fw
    h = max(w / aspect, FIT_HEIGHT * fh)
    w = max(h * aspect, CAM_PANEL_W / MAX_CAM_UPSCALE)  # a small face: widen rather than blow up the pixels
    h = w / aspect
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2 + 0.1 * fh
    lim = [cam_box[0] * vw, cam_box[1] * vh, cam_box[2] * vw, cam_box[3] * vh]
    # never bigger than the overlay: shrink to it, keeping the aspect
    scale = min(1.0, (lim[2] - lim[0]) / w, (lim[3] - lim[1]) / h)
    w, h = w * scale, h * scale
    x0 = min(max(cx - w / 2, lim[0]), lim[2] - w)
    y0 = min(max(cy - h / 2, lim[1]), lim[3] - h)
    out = [round(x0 / vw, 5), round(y0 / vh, 5), round((x0 + w) / vw, 5), round((y0 + h) / vh, 5)]
    # the gate fails a face that touches the panel edge: keep a margin, else the standard box
    return out if _inside(f, out, 0.08) else cam_box


def game_box_for(face: list, corner: str, opts: dict, vw: int, vh: int, cam: list = None):
    """Gameplay crop (normalised) clear of the whole streamer (and of the cam overlay box `cam`, when given)
    for a corner cam, or None (see gameplay_box_avoiding). A centred crop that does not fit tries the side;
    with opts always_split the last resort is fallback_game_box (the streamer may show at its edge)."""
    from streamclipper.vision import person_box

    aspect = GAME_ASPECT.get(opts.get("ratio") or "9:16", GAME_ASPECT["9:16"])
    keep_out = union(person_box(face, corner), cam)
    anchor = opts.get("game_anchor") or "side"
    game = gameplay_box_avoiding(keep_out, vw, vh, aspect, anchor=anchor)
    if game is None and anchor == "center":
        game = gameplay_box_avoiding(keep_out, vw, vh, aspect, anchor="side")
    if game is None and opts.get("always_split"):
        game = fallback_game_box(keep_out, opts.get("ratio") or "9:16", vw, vh)
    return game


def fallback_game_box(keep_out, ratio: str, vw: int, vh: int) -> list:
    """Full-height gameplay crop with the game panel's aspect, centred, then moved away from keep_out as far
    as the frame allows (used only when no crop clear of the streamer exists: the split is always kept)."""
    aspect = GAME_ASPECT.get(ratio, GAME_ASPECT["9:16"])
    w = min(1.0, aspect * vh / vw)
    x0 = 0.5 - w / 2
    if keep_out:
        if (keep_out[0] + keep_out[2]) / 2 < 0.5:
            x0 = min(max(x0, keep_out[2]), 1.0 - w)
        else:
            x0 = max(min(x0, keep_out[0] - w), 0.0)
    return [round(x0, 5), 0.0, round(x0 + w, 5), 1.0]


def face_cam_box(face: list, opts: dict, vw: int, vh: int) -> list:
    """Cam panel crop around a face with no cam overlay (a big or full-screen cam): the tight fit inside
    the whole frame, or a box of the cam aspect centred on the face."""
    ratio = opts.get("ratio") or "9:16"
    fit = opts.get("cam_fit") if opts.get("cam_fit") in FIT_WIDTH else "tight"
    box = fit_cam_box(face, [0.0, 0.0, 1.0, 1.0], fit, ratio, vw, vh)
    if box != [0.0, 0.0, 1.0, 1.0]:
        return box
    aspect = CAM_ASPECT.get(ratio, CAM_ASPECT["9:16"])
    h = min(1.0, 3.2 * (face[3] - face[1]))
    w = min(1.0, h * vh * aspect / vw)
    h = w * vw / aspect / vh
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    x0 = min(max(cx - w / 2, 0.0), 1.0 - w)
    y0 = min(max(cy - h / 2, 0.0), 1.0 - h)
    return [round(x0, 5), round(y0, 5), round(x0 + w, 5), round(y0 + h, 5)]


def _corner_of_box(box: list) -> str:
    """The frame corner nearest a normalised box (for the streamer profile)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return ("top" if cy < 0.5 else "bottom") + ("_left" if cx < 0.5 else "_right")


def union(a, b):
    """The box around two normalised boxes (b may be None)."""
    if not b:
        return list(a)
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


# ---------------------------------------------------------------- cam tracking inside a clip


def _parts(s: float, e: float) -> list:
    n = max(2, min(4, int(round((e - s) / TRACK_PART_S))))
    step = (e - s) / n
    return [(s + k * step, s + (k + 1) * step) for k in range(n)]


def _part_box(res: dict, opts: dict, vw: int, vh: int) -> dict:
    """One part's decision: {kind, corner, face, cam (norm), game (norm)}."""
    kind = res.get("kind") if res else None
    if kind == "corner-cam" and res.get("cam_box"):
        cam = fit_cam_box(res["face_box"], res["cam_box"], opts.get("cam_fit") or "standard", opts["ratio"], vw, vh)
        game = None
        if opts.get("game_crop"):
            from streamclipper.util import box_to_norm

            game = box_to_norm(opts["game_crop"], vw, vh)
        else:
            game = game_box_for(res["face_box"], res["corner"], opts, vw, vh, res["cam_box"])
        if game is not None:
            return {"kind": "corner-cam", "corner": res["corner"], "face": res["face_box"], "cam": cam, "game": game}
    if kind == "full-cam" and opts.get("always_split") and res.get("face_box"):
        cam = face_cam_box(res["face_box"], opts, vw, vh)
        game = fallback_game_box(cam, opts.get("ratio") or "9:16", vw, vh)
        return {"kind": "full-cam", "corner": _corner_of_box(cam), "face": res["face_box"], "cam": cam, "game": game}
    return {"kind": kind or "no-cam", "corner": None, "face": None, "cam": None, "game": None}


def _same(a: dict, b: dict) -> bool:
    if (a["cam"] is None) != (b["cam"] is None):
        return False
    if a["cam"] is None:
        return True
    return a["corner"] == b["corner"] and iou(a["cam"], b["cam"]) >= MOVE_IOU


def _face_in(faces: list, box) -> bool:
    return any(box and _inside(f, box, 0.0) for f in (faces or []))


def _refine(video: Path, lo: float, hi: float, left: dict, right: dict, vw: int, vh: int, detect) -> float:
    """The time in [lo, hi] where the left cam state ends and the right one starts (binary search on single
    frames: the right state holds when the frame looks like it and not like the left state)."""

    def looks(st, other, faces):
        if st["cam"] is not None:
            return _face_in(faces, st["cam"])
        return not _face_in(faces, other["cam"])  # no cam: the other state's cam box is empty

    for _ in range(REFINE_STEPS):
        mid = (lo + hi) / 2
        got = detect(video, [mid], vw, vh)
        faces = got[0] if got else []
        if looks(right, left, faces) and not looks(left, right, faces):
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 3)


def track_cam(video: Path, s: float, e: float, vw: int, vh: int, opts: dict, classify=None, detect=None):
    """
    Check the clip [s, e] (window seconds) in 2-4 parts. Returns None when the cam stays put (the caller keeps
    its one-crop layout), else [{start, end, kind, corner, cam, game}] runs of parts with the same cam state, the
    cut between two runs refined to about 1/32 of a part. classify/detect: injected in tests.
    """
    from streamclipper import vision

    classify = classify or vision.classify_layout
    detect = detect or vision.detect_faces
    parts = _parts(s, e)
    states = []
    for a, b in parts:
        res = classify(video, a, b, vw, vh)
        if res is None:  # no face detector installed
            return None
        states.append(_part_box(res, opts, vw, vh))
    if all(_same(states[0], st) for st in states[1:]):
        return None
    runs = [dict(states[0], start=parts[0][0], end=parts[0][1])]
    for k in range(1, len(parts)):
        a, b = parts[k]
        st = states[k]
        if _same(runs[-1], st):
            runs[-1]["end"] = b
            continue
        pa, pb = parts[k - 1]
        cut = _refine(video, (pa + pb) / 2, (a + b) / 2, runs[-1], st, vw, vh, detect)
        cut = min(max(cut, runs[-1]["start"] + 0.5), b - 0.5)
        runs[-1]["end"] = cut
        runs.append(dict(st, start=cut, end=b))
    if len(runs) == 1:
        return None
    return runs


def runs_to_segments(runs: list, wdur: float, ratio: str, vw: int, vh: int) -> list:
    """
    Reframe segments (window time 0..wdur) for runs from track_cam: split layout with the cam and game crops
    where a cam was found, else a full-height centre crop. Crops are locked to the panel aspects.
    """
    from streamclipper import reframe

    segs = []
    for i, r in enumerate(runs):
        a = 0.0 if i == 0 else r["start"]
        b = wdur if i == len(runs) - 1 else runs[i + 1]["start"]
        if r["cam"] is not None:
            cam = reframe.lock_aspect(
                norm_to_px(r["cam"], vw, vh), reframe.window_aspects("split", ratio)["cam"], vw, vh
            )
            game = reframe.lock_aspect(
                norm_to_px(r["game"], vw, vh), reframe.window_aspects("split", ratio)["game"], vw, vh
            )
            layout, wins = "split", {"cam": cam, "game": game}
        else:
            asp = reframe.window_aspects("full", ratio)["game"]
            cw, ch, x0, y0 = _fill_crop((0, 0, vw, vh), vw, vh, asp)
            layout, wins = "full", {"cam": None, "game": reframe.lock_aspect([x0, y0, cw, ch], asp, vw, vh)}
        segs.append(
            {
                "start": round(a, 3),
                "end": round(b, 3),
                "layout": layout,
                "windows": wins,
                "keys": [],
                "ease": "smooth",
                "boundary": "cut",
            }
        )
    # a run shorter than the editor's minimum joins its neighbour
    out = []
    for sg in segs:
        if out and sg["end"] - sg["start"] < reframe.MIN_SEG:
            out[-1]["end"] = sg["end"]
            continue
        if out and out[-1]["end"] - out[-1]["start"] < reframe.MIN_SEG:
            sg["start"] = out[-1]["start"]
            out[-1] = sg
            continue
        out.append(sg)
    return out


# ---------------------------------------------------------------- streamer profiles


def load_profile(path) -> dict:
    """A streamer profile ({schema, name, cams: [{corner, box, seen, manual}], settings}); empty when missing."""
    if not path:
        return None
    p = Path(path)
    empty = {"schema": PROFILE_SCHEMA, "name": p.stem, "cams": [], "settings": {}, "path": str(p)}
    if not p.is_file():
        return empty
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ToolError(f"The streamer profile is not valid JSON: {p}", "fix the file or delete it") from None
    if not isinstance(data, dict) or not isinstance(data.get("cams", []), list):
        raise ToolError(f"The streamer profile has an unknown format: {p}", "delete it; it is rebuilt on the next run")
    data.setdefault("cams", [])
    data.setdefault("settings", {})
    data["path"] = str(p)
    return data


def learn(profile: dict, corner: str, box: list, manual: bool = False) -> None:
    """Count a cam position (normalised box) seen in a clip; a close match (IoU >= MOVE_IOU) is merged."""
    if profile is None or not box:
        return
    for c in profile["cams"]:
        if c.get("corner") == corner and iou(c["box"], box) >= MOVE_IOU:
            n = int(c.get("seen", 1))
            if manual or not c.get("manual"):
                c["box"] = [round((v * n + w) / (n + 1), 5) for v, w in zip(c["box"], box, strict=True)]
            c["seen"] = n + 1
            c["manual"] = bool(c.get("manual") or manual)
            return
    profile["cams"].append({"corner": corner, "box": [round(v, 5) for v in box], "seen": 1, "manual": bool(manual)})


def profile_candidates(profile: dict) -> list:
    """Known cam positions, manual corrections first, then the most seen."""
    if not profile:
        return []
    return sorted(profile["cams"], key=lambda c: (not c.get("manual"), -int(c.get("seen", 1))))


def save_profile(profile: dict, settings: dict = None) -> None:
    if not profile or not profile.get("path"):
        return
    data = {k: v for k, v in profile.items() if k != "path"}
    data["schema"] = PROFILE_SCHEMA
    if settings:
        data["settings"] = dict(data.get("settings") or {}, **settings)
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["cams"] = profile_candidates(data)[:8]
    write_json_atomic(Path(profile["path"]), data)


def profile_cam(profile: dict, video: Path, s: float, e: float, vw: int, vh: int, check=None):
    """
    The first known cam position of the profile that shows a face inside it in this clip (vision.check_face_box
    PASS), as {corner, cam (norm), face_hint}; None when none does or no face detector is installed.
    """
    from streamclipper import vision

    check = check or vision.check_face_box
    for c in profile_candidates(profile):
        px = norm_to_px(c["box"], vw, vh)
        got = check(video, s, max(0.5, e - s), px)
        if got and got[0] == "PASS":
            return {"corner": c.get("corner") or "top_left", "cam": c["box"]}
    return None


def corner_face(cam: list, corner: str) -> list:
    """A face-sized box inside a known cam box (for the person box that keeps the streamer out of the gameplay)."""
    x0, y0, x1, y1 = cam
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, y0 + 0.45 * h
    return [cx - 0.18 * w, cy - 0.25 * h, cx + 0.18 * w, cy + 0.25 * h]


# ---------------------------------------------------------------- layout overrides (per clip)


def load_overrides(path) -> dict:
    """{'clips': [...]} from a layout overrides file (see write_preview); empty when no path is given."""
    if not path:
        return {"clips": []}
    p = Path(path)
    if not p.is_file():
        raise ToolError(f"Layout overrides file not found: {p}", "run with --preview-only first, or fix the path")
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except ValueError as e:
        raise ToolError(
            f"The layout overrides file is not valid JSON ({e}).", "fix the file or make it again"
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        raise ToolError("The layout overrides file has no 'clips' list.", "make it again with --preview-only")
    return data


def overrides_key(path) -> str:
    """Fingerprint of the overrides file for the resume key (a changed file makes the clips again)."""
    if not path or not Path(path).is_file():
        return ""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def find_override(data: dict, clip: dict):
    """The manual entry for a clip (matched on its start time within 1.5 s, else on its rank), or None.
    An entry counts only when 'auto' is false."""
    for o in (data or {}).get("clips", []):
        if not isinstance(o, dict) or o.get("auto", True):
            continue
        if o.get("start") is not None and abs(float(o["start"]) - float(clip["start"])) <= 1.5:
            return o
    for o in (data or {}).get("clips", []):
        if isinstance(o, dict) and not o.get("auto", True) and o.get("start") is None and o.get("rank") == clip["rank"]:
            return o
    return None


def _check_px(box, what: str, vw: int, vh: int) -> list:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ToolError(f"The {what} box must be [x, y, width, height] in pixels.", "fix the layout overrides file")
    x, y, w, h = (int(round(float(v))) for v in box)
    if w < 64 or h < 64 or x < 0 or y < 0 or x + w > vw or y + h > vh:
        raise ToolError(
            f"The {what} box {x},{y},{w},{h} is too small or outside the {vw}x{vh} frame.",
            "fix the layout overrides file (boxes are in pixels of the source video)",
        )
    return [x, y, w, h]


def apply_override(o: dict, opts: dict, vw: int, vh: int) -> tuple:
    """(opts, cam_box, game_box, note) of a manual entry: layout 'split' needs cam and game boxes, 'full' a game box
    (or none: a centre crop). Boxes are pixels of the clip window/source frame."""
    layout = o.get("layout") or ("split" if o.get("cam") else "full")
    if layout not in ("split", "full"):
        raise ToolError(f"Unknown layout '{layout}' in the layout overrides.", "use split or full")
    game = px_to_norm(_check_px(o["game"], "gameplay", vw, vh), vw, vh) if o.get("game") else None
    if layout == "split":
        cam = px_to_norm(_check_px(o.get("cam"), "facecam", vw, vh), vw, vh)
        return dict(opts, layout="split"), cam, game, "layout: split (manual crop from the layout overrides)"
    return dict(opts, layout="full"), None, game, "layout: full (manual crop from the layout overrides)"


# ---------------------------------------------------------------- preview


def _drawbox(box_px, colour: str) -> str:
    x, y, w, h = box_px
    return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={colour}:t=8"


def _grid(vw: int, vh: int, labels: bool) -> list:
    """A light grid every 100 px of the source frame, with the pixel numbers every 200 px (when drawtext works)."""
    from streamclipper.paths import TEMPLATE_DIR

    out = ["drawgrid=w=100:h=100:t=1:color=yellow@0.35"]
    font = TEMPLATE_DIR / "fonts" / "Montserrat-Bold.ttf"
    if not labels or not font.is_file():
        return out
    ff = font.as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "")
    size = max(14, vh // 40)
    style = f"fontfile='{ff}':fontsize={size}:fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=3"
    out += [f"drawtext={style}:text='{x}':x={x + 4}:y=4" for x in range(200, vw, 200)]
    out += [f"drawtext={style}:text='{y}':x=4:y={y + 4}" for y in range(200, vh, 200)]
    return out


def preview_image(video: Path, t: float, vw: int, vh: int, layout: str, cam_px, game_px, ratio: str, out: Path) -> Path:
    """
    One JPG: the frame at t (window seconds) with a pixel grid, the facecam crop (red) and the gameplay crop (cyan)
    drawn on it, next to a mock of the vertical clip made from those crops. Also writes <out>_frame.jpg (the plain
    frame). The grid numbers need ffmpeg's drawtext; without it the grid is drawn without numbers.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    W = 1080
    H = 1920 if ratio == "9:16" else 1080
    cam_h = 672 if ratio == "9:16" else 454
    if layout == "split" and cam_px and game_px:
        cx, cy, cw, ch = cam_px
        gx, gy, gw, gh = game_px
        mock = (
            f"[b]crop={cw}:{ch}:{cx}:{cy},scale={W}:{cam_h}:force_original_aspect_ratio=increase,crop={W}:{cam_h}[cc];"
            f"[c]crop={gw}:{gh}:{gx}:{gy},scale={W}:{H - cam_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H - cam_h}[gg];[cc][gg]vstack=2,scale={W // 2}:{H // 2}[m]"
        )
    else:
        gx, gy, gw, gh = game_px or [0, 0, vw, vh]
        mock = (
            f"[b]crop={gw}:{gh}:{gx}:{gy},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"scale={W // 2}:{H // 2}[m];[c]nullsink"
        )
    draws = [d for d in (_drawbox(cam_px, "red") if cam_px else "", _drawbox(game_px, "cyan") if game_px else "") if d]
    last = None
    for labels in (True, False):
        chain = ",".join(_grid(vw, vh, labels) + draws + [f"scale=-2:{H // 2}"])
        fc = f"[0:v]split=4[a][b][c][d];[a]{chain}[f];{mock};[f][m]hstack=2,format=yuvj420p[v];[d]format=yuvj420p[raw]"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(video)]
        cmd += ["-filter_complex", fc, "-map", "[v]", "-frames:v", "1", "-q:v", "3", str(out)]
        cmd += ["-map", "[raw]", "-frames:v", "1", "-q:v", "3", str(out.with_name(out.stem + "_frame.jpg"))]
        try:
            run_cmd(cmd, "layout preview image")
            return out
        except ToolError as e:  # an ffmpeg build without drawtext (no libfreetype): try once without the numbers
            last = e
    raise last


def render_crops(layout: str, ratio: str, cam, game, vw: int, vh: int) -> dict:
    """The crops ([x, y, w, h] px) the panel render really uses for a one-crop layout (same rules as
    layout.render_split_panels / render_fill_panel), for the preview images."""
    from streamclipper.layout import frame_crop, grow_to_aspect

    cam_px = norm_to_px(cam, vw, vh) if cam else None
    game_px = norm_to_px(game, vw, vh) if game else None
    if layout == "split" and cam_px:
        cx, cy, cw, ch = grow_to_aspect(tuple(cam_px), vw, vh, CAM_ASPECT[ratio])
        if game_px:
            gw, gh, gx, gy = _fill_crop(tuple(game_px), vw, vh, GAME_ASPECT[ratio])
        elif cam_px[1] < vh // 4 and cam_px[1] + cam_px[3] < vh // 2:
            top = (cam_px[1] + cam_px[3]) & ~1
            gw, gh, gx, gy = _fill_crop((0, top, vw, vh - top), vw, vh, GAME_ASPECT[ratio])
        else:
            gw, gh, gx, gy = frame_crop(vw, vh, None, tuple(cam_px), GAME_ASPECT[ratio])
        return {"layout": "split", "cam": [cx, cy, cw, ch], "game": [gx, gy, gw, gh]}
    gw, gh, gx, gy = _fill_crop(tuple(game_px) if game_px else (0, 0, vw, vh), vw, vh, FULL_ASPECT[ratio])
    return {"layout": "full", "cam": None, "game": [gx, gy, gw, gh]}
