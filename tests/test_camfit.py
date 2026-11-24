"""Per-clip cam/gameplay crops: cam fit, centred gameplay, cam tracking, streamer profiles, overrides, preview."""

import json
import shutil
import subprocess

import pytest

from streamclipper import camfit, reframe
from streamclipper.cli import build_parser
from streamclipper.layout import gameplay_box_avoiding
from streamclipper.options import resolve_options
from streamclipper.util import ToolError
from streamclipper.vision import person_box

VW, VH = 1920, 1080
FACE = [0.05, 0.07, 0.11, 0.18]  # a top-left webcam face (normalised)
OVERLAY = [0.0, 0.0, 0.26, 0.36]  # the detected cam overlay around it


@pytest.fixture(autouse=True)
def _haar_faces(monkeypatch):
    """Face boxes in these tests are Haar-sized (no MediaPipe resize)."""
    monkeypatch.setattr(camfit._faces, "get_detector", lambda: None)


def _opts(**kw):
    o = {"layout": "split", "cam_crop": None, "game_crop": None, "ratio": "9:16", "preset": ""}
    o.update(kw)
    return o


def _px_aspect(box):
    return (box[2] - box[0]) * VW / ((box[3] - box[1]) * VH)


def test_centre_anchor_puts_the_gameplay_in_the_middle():
    pb = person_box(FACE, "top_left")
    aspect = 1080 / 1248
    side = gameplay_box_avoiding(pb, VW, VH, aspect)
    centre = gameplay_box_avoiding(pb, VW, VH, aspect, anchor="center")
    mid = lambda b: (b[0] + b[2]) / 2  # noqa: E731
    assert abs(mid(centre) - 0.5) < abs(mid(side) - 0.5)
    assert abs(mid(centre) - 0.5) < 0.02  # the person box ends well left of a centred crop
    assert centre[0] >= pb[2]  # never over the streamer
    assert centre[3] - centre[1] >= 0.99
    # a streamer in the middle still leaves no tall crop
    assert gameplay_box_avoiding([0.3, 0.2, 0.7, 1.0], VW, VH, aspect, anchor="center") is None


def test_tight_and_close_fit_frame_the_face_bigger_inside_the_overlay():
    assert camfit.fit_cam_box(FACE, OVERLAY, "standard", "9:16", VW, VH) == OVERLAY
    face_w = FACE[2] - FACE[0]
    prev = OVERLAY[2] - OVERLAY[0]
    for fit in ("tight", "close"):  # a 4K source: the face has enough pixels for a close crop
        box = camfit.fit_cam_box(FACE, OVERLAY, fit, "9:16", 2 * VW, 2 * VH)
        assert abs(_px_aspect(box) - 1080 / 672) < 0.02
        assert OVERLAY[0] <= box[0] and OVERLAY[1] <= box[1] and box[2] <= OVERLAY[2] and box[3] <= OVERLAY[3]
        assert camfit._inside(FACE, box, 0.08)
        width = box[2] - box[0]
        assert width < prev and face_w / width > 0.3  # the face fills more of the panel each step
        prev = width


def test_fit_never_enlarges_a_small_face_more_than_the_cap():
    for fit in ("tight", "close"):  # 1080p: a 3-face-width crop would be ~345 px, scaled x3.1 to 1080
        box = camfit.fit_cam_box(FACE, OVERLAY, fit, "9:16", VW, VH)
        assert 1080 / ((box[2] - box[0]) * VW) <= camfit.MAX_CAM_UPSCALE + 0.01
        assert camfit._inside(FACE, box, 0.08) and box[2] <= OVERLAY[2]


def test_sharpen_only_for_enlarged_crops():
    from streamclipper.layout import sharpen

    assert sharpen(1080, 1080) == "" and sharpen(950, 1080) == ""
    assert sharpen(432, 1080).startswith(",unsharp=5:5:0.")


def test_fit_keeps_the_overlay_when_a_closer_crop_would_cut_the_face():
    face = [0.0, 0.0, 0.08, 0.2]  # face against the frame corner
    assert camfit.fit_cam_box(face, [0.0, 0.0, 0.1, 0.22], "close", "9:16", VW, VH) == [0.0, 0.0, 0.1, 0.22]


def test_clip_layout_uses_cam_fit_and_game_anchor(monkeypatch):
    import streamclipper.options as options

    res = {"kind": "corner-cam", "corner": "top_left", "face_box": FACE, "cam_box": OVERLAY}
    monkeypatch.setattr(options, "classify_layout", lambda *a: dict(res))
    monkeypatch.setattr(options, "black_spans", lambda *a: [])
    meta = {"width": VW, "height": VH, "dur": 30.0}
    info = {}
    o, cam, game, note = options.clip_layout("x.mp4", meta, _opts(cam_fit="tight", game_anchor="center"), 0, 10, info)
    assert o["layout"] == "split" and info["corner"] == "top_left"
    assert cam != OVERLAY and cam[2] - cam[0] < OVERLAY[2] - OVERLAY[0]
    assert abs((game[0] + game[2]) / 2 - 0.5) < 0.02
    o, cam, game, note = options.clip_layout("x.mp4", meta, _opts(), 0, 10)
    assert cam == OVERLAY  # the defaults keep the 5.3 behaviour


def test_options_and_flags():
    args = build_parser().parse_args(["clip-stream", "--input", "s.mp4", "--out", "w"])
    o = resolve_options(args)
    assert (o["cam_fit"], o["game_anchor"], o["cam_track"]) == ("standard", "side", "off")
    args = build_parser().parse_args(
        [
            "clip-stream", "--input", "s.mp4", "--out", "w", "--cam-fit", "tight", "--game-anchor", "center",
            "--cam-track", "auto", "--streamer-profile", "p.json", "--layout-overrides", "o.json", "--preview-only",
        ]
    )  # fmt: skip
    o = resolve_options(args)
    assert (o["cam_fit"], o["game_anchor"], o["cam_track"]) == ("tight", "center", "auto")
    assert args.preview_only and args.streamer_profile == "p.json" and args.layout_overrides == "o.json"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["clip-stream", "--out", "w", "--cam-fit", "huge"])


def test_resume_key_is_unchanged_with_the_default_crop_options():
    from streamclipper.commands import _make_key

    args = build_parser().parse_args(["clip-stream", "--input", "s.mp4", "--out", "w"])
    o = resolve_options(args)
    old = {k: v for k, v in o.items() if k not in ("cam_fit", "game_anchor", "cam_track")}
    assert _make_key(o, args, "", True) == _make_key(old, args, "", True)
    assert _make_key(dict(o, cam_fit="tight"), args, "", True) != _make_key(o, args, "", True)


# ---- cam tracking ---------------------------------------------------------------------------------------------

LEFT = {"kind": "corner-cam", "corner": "top_left", "face_box": FACE, "cam_box": OVERLAY}
RIGHT_FACE = [0.89, 0.07, 0.95, 0.18]
RIGHT = {"kind": "corner-cam", "corner": "top_right", "face_box": RIGHT_FACE, "cam_box": [0.74, 0.0, 1.0, 0.36]}


def _fake(switch: float):
    """classify/detect stand-ins: the cam is top-left before `switch` seconds and top-right after."""

    def classify(video, a, b, vw, vh):
        return dict(LEFT if (a + b) / 2 < switch else RIGHT)

    def detect(video, times, vw, vh):
        return [[FACE] if t < switch else [RIGHT_FACE] for t in times]

    return classify, detect


def test_a_static_cam_keeps_one_crop():
    classify, detect = _fake(switch=999)
    assert camfit.track_cam("w.mp4", 1.0, 37.0, VW, VH, _opts(), classify, detect) is None


def test_a_cam_that_moves_inside_the_clip_gives_parts_with_their_own_crops():
    classify, detect = _fake(switch=20.0)
    runs = camfit.track_cam("w.mp4", 1.0, 37.0, VW, VH, _opts(game_anchor="center"), classify, detect)
    assert [r["corner"] for r in runs] == ["top_left", "top_right"]
    assert abs(runs[1]["start"] - 20.0) < 0.6  # the cut is refined to about half a second
    segs = camfit.runs_to_segments(runs, 38.0, "9:16", VW, VH)
    segs = reframe.validate_segments(segs, {"width": VW, "height": VH, "ratio": "9:16"}, 0.0, 38.0)
    assert [s["layout"] for s in segs] == ["split", "split"] and segs[0]["start"] == 0 and segs[-1]["end"] == 38.0
    cam0, cam1 = segs[0]["windows"]["cam"], segs[1]["windows"]["cam"]
    assert cam0[0] < VW / 2 < cam1[0]  # each part frames its own cam
    g1 = segs[1]["windows"]["game"]
    assert g1[0] + g1[2] <= cam1[0] + 2 or g1[1] >= cam1[1] + cam1[3]  # the gameplay stays clear of the moved cam


def test_a_cam_that_turns_off_gives_a_full_part():
    def classify(video, a, b, vw, vh):
        return dict(LEFT) if (a + b) / 2 < 15 else {"kind": "no-cam", "corner": None, "face_box": None}

    def detect(video, times, vw, vh):
        return [[FACE] if t < 15 else [] for t in times]

    runs = camfit.track_cam("w.mp4", 0.0, 36.0, VW, VH, _opts(), classify, detect)
    segs = camfit.runs_to_segments(runs, 36.0, "9:16", VW, VH)
    segs = reframe.validate_segments(segs, {"width": VW, "height": VH, "ratio": "9:16"}, 0.0, 36.0)
    assert [s["layout"] for s in segs] == ["split", "full"]


# ---- streamer profiles ----------------------------------------------------------------------------------------


def test_profile_learns_merges_and_round_trips(tmp_path):
    p = tmp_path / "streamers" / "ana.json"
    prof = camfit.load_profile(p)
    assert prof["cams"] == [] and prof["name"] == "ana"
    camfit.learn(prof, "top_left", [0.0, 0.0, 0.2, 0.3])
    camfit.learn(prof, "top_left", [0.0, 0.0, 0.21, 0.31])  # the same cam, a bit different
    camfit.learn(prof, "bottom_right", [0.8, 0.7, 1.0, 1.0], manual=True)
    assert len(prof["cams"]) == 2 and prof["cams"][0]["seen"] == 2
    camfit.save_profile(prof, {"cam_fit": "tight"})
    again = camfit.load_profile(p)
    assert again["settings"]["cam_fit"] == "tight"
    assert camfit.profile_candidates(again)[0]["manual"]  # manual corrections are tried first
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(ToolError):
        camfit.load_profile(bad)


def test_profile_cam_needs_a_face_inside():
    prof = {"cams": [{"corner": "top_left", "box": OVERLAY, "seen": 3}], "settings": {}}
    hit = camfit.profile_cam(prof, "w.mp4", 0, 10, VW, VH, check=lambda *a: ("PASS", "ok"))
    assert hit == {"corner": "top_left", "cam": OVERLAY}
    assert camfit.profile_cam(prof, "w.mp4", 0, 10, VW, VH, check=lambda *a: ("WARN", "no face")) is None


# ---- overrides and the per-clip decision -----------------------------------------------------------------------


def test_overrides_match_on_start_time_and_need_auto_false(tmp_path):
    data = {
        "clips": [
            {"rank": 1, "start": 100.0, "auto": True, "layout": "split", "cam": [0, 0, 400, 300]},
            {"rank": 2, "start": 250.4, "auto": False, "layout": "split", "cam": [0, 0, 400, 300],
             "game": [500, 0, 934, 1080]},
        ]
    }  # fmt: skip
    f = tmp_path / "o.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    loaded = camfit.load_overrides(f)
    assert camfit.find_override(loaded, {"rank": 1, "start": 100.0}) is None
    hit = camfit.find_override(loaded, {"rank": 7, "start": 251.0})  # ranks can change; the time matches
    assert hit["rank"] == 2 and camfit.overrides_key(f)
    o, cam, game, note = camfit.apply_override(hit, _opts(), VW, VH)
    assert o["layout"] == "split" and cam == camfit.px_to_norm([0, 0, 400, 300], VW, VH) and "manual" in note
    with pytest.raises(ToolError):
        camfit.apply_override({"layout": "split", "cam": [1800, 0, 400, 300]}, _opts(), VW, VH)
    with pytest.raises(ToolError):
        camfit.load_overrides(tmp_path / "missing.json")


def test_decide_layout_uses_the_override_then_detection_then_the_profile(monkeypatch, tmp_path):
    import streamclipper.commands as commands

    wmeta = {"width": VW, "height": VH, "dur": 40.0}
    clip = {"rank": 1, "start": 101.0, "end": 139.0}
    prof = camfit.load_profile(tmp_path / "p.json")
    ov = {"clips": [{"start": 101.0, "auto": False, "layout": "split", "cam": [0, 0, 480, 300]}]}
    o, cam, game, note, segs = commands.decide_clip_layout(
        clip, "w.mp4", wmeta, 100.0, _opts(), {}, {"profile": prof, "overrides": ov}
    )
    assert "manual" in note and segs is None and game is not None and prof["cams"][0]["manual"]

    def detected(src, meta, opts, s, e, info=None):
        info.update(LEFT)
        return opts, OVERLAY, [0.3, 0.0, 0.75, 1.0], "layout: corner-cam (top_left) -> split"

    monkeypatch.setattr(commands, "clip_layout", detected)
    prof = camfit.load_profile(tmp_path / "q.json")
    o, cam, game, note, segs = commands.decide_clip_layout(
        clip, "w.mp4", wmeta, 100.0, _opts(), {}, {"profile": prof, "overrides": {"clips": []}}
    )
    assert cam == OVERLAY and len(prof["cams"]) == 1  # a detected cam is remembered

    def missed(src, meta, opts, s, e, info=None):
        info.update({"kind": "no-cam", "corner": None, "face_box": None})
        return dict(opts, layout="full"), None, None, "layout: no-cam -> full (centre crop)"

    monkeypatch.setattr(commands, "clip_layout", missed)
    monkeypatch.setattr(camfit, "profile_cam", lambda *a, **k: {"corner": "top_left", "cam": OVERLAY})
    o, cam, game, note, segs = commands.decide_clip_layout(
        clip, "w.mp4", wmeta, 100.0, _opts(), {}, {"profile": prof, "overrides": {"clips": []}}
    )
    assert o["layout"] == "split" and cam == OVERLAY and "streamer profile" in note and game[0] >= OVERLAY[2] - 0.01


# ---- preview images --------------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_preview_image_draws_the_crops_and_a_mock(tmp_path):
    video = tmp_path / "w.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    crops = camfit.render_crops("split", "9:16", [0.0, 0.0, 0.25, 0.3], [0.3, 0.0, 0.6, 1.0], 640, 360)
    assert crops["layout"] == "split" and crops["cam"][2] / crops["cam"][3] == pytest.approx(1080 / 672, abs=0.03)
    out = camfit.preview_image(video, 1.0, 640, 360, "split", crops["cam"], crops["game"], "9:16", tmp_path / "p.jpg")
    assert out.is_file() and (tmp_path / "p_frame.jpg").is_file()
    full = camfit.render_crops("full", "9:16", None, None, 640, 360)
    camfit.preview_image(video, 1.0, 640, 360, "full", None, full["game"], "9:16", tmp_path / "f.jpg")
    assert (tmp_path / "f.jpg").is_file()


# ---- Colab helpers ---------------------------------------------------------------------------------------------


def _base_opts(**kw):
    o = {
        "prefix": "clip", "max_clips": 3, "min_score": 50, "llm": "none", "whisper_model": "auto", "device": "auto",
        "language": "auto", "render": True, "drafts": False, "resume": True,
    }  # fmt: skip
    o.update(kw)
    return o


def test_colab_argv_is_unchanged_with_the_defaults_and_passes_the_new_options(tmp_path):
    from streamclipper.colab import _clip_argv

    url = "https://www.youtube.com/watch?v=abc"
    plain = _clip_argv(url, tmp_path, _base_opts())
    for flag in (
        "--cam-fit",
        "--game-anchor",
        "--cam-track",
        "--min-dur",
        "--streamer-profile",
        "--preview-only",
        "--min-pause",
        "--visual-rank",
    ):
        assert flag not in plain
    argv = _clip_argv(
        url,
        tmp_path,
        _base_opts(
            cam_fit="tight", game_anchor="center", cam_track="auto", streamer_profile="p.json",
            layout_overrides="o.json", preview_only=True, min_dur=38, target_dur=42, max_dur=48,
            min_pause=999, visual_rank="qwen-3b",
        ),
    )  # fmt: skip
    joined = " ".join(argv)
    parts = ["--cam-fit tight", "--game-anchor center", "--cam-track auto", "--streamer-profile p.json"]
    parts += ["--layout-overrides o.json", "--preview-only", "--min-dur 38", "--target-dur 42", "--max-dur 48"]
    parts += ["--min-pause 999", "--visual-rank qwen-3b"]
    for part in parts:
        assert part in joined, part
    assert "--render" not in argv  # a preview renders nothing
    args = build_parser().parse_args(argv)
    assert args.cam_fit == "tight" and args.max_dur == 48 and args.min_pause == 999


def test_long_clips_raise_the_timeline_cap_and_new_windows_get_new_keys():
    from streamclipper.clipplan import LONG_TIMELINE_CAP, MAX_TIMELINE, fit_timeline
    from streamclipper.commands import _clip_key

    segs = [{"cut_start": 0.0, "cut_end": 85.0, "kind": "payoff"}]
    assert sum(p["cut_end"] - p["cut_start"] for p in fit_timeline(segs, [], 92.5)) == pytest.approx(85.0)
    assert MAX_TIMELINE == 59.5 and LONG_TIMELINE_CAP == 179.5
    a = {"start": 10.0, "end": 30.0, "segments": [1, 2]}
    assert _clip_key("k", a) == _clip_key("k", dict(a))
    assert _clip_key("k", a) != _clip_key("k", {**a, "end": 80.0})


def test_streamer_profile_path_is_a_safe_file_name(monkeypatch, tmp_path):
    from streamclipper import colab

    monkeypatch.setenv("STREAMCLIPPER_OUT", str(tmp_path))
    assert colab.streamer_profile_path("") == ""
    p = colab.streamer_profile_path("  El Rubius / GTA ")
    assert p.endswith("el_rubius_gta.json") and str(tmp_path) in p


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_correct_clip_sets_a_manual_crop_with_a_new_preview(tmp_path):
    from streamclipper.colab import correct_clip

    video = tmp_path / "w.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    ov = tmp_path / "layout_overrides.json"
    entry = {"rank": 2, "start": 101.0, "end": 102.5, "window_start": 100.0, "window": str(video)}
    entry.update(width=1280, height=720, auto=True, layout="split", cam=[0, 0, 300, 200], game=[400, 0, 622, 720])
    ov.write_text(json.dumps({"clips": [entry]}), encoding="utf-8")
    new = correct_clip(str(ov), 2, "0,0,320,200")
    assert (
        new["auto"] is False and new["game"][0] >= 320 and (tmp_path / "layout_preview" / "rank02_manual.jpg").is_file()
    )
    saved = camfit.load_overrides(ov)["clips"][0]
    assert camfit.find_override({"clips": [saved]}, {"rank": 9, "start": 101.2}) is not None
    with pytest.raises(ToolError):
        correct_clip(str(ov), 5, "0,0,320,200")
    with pytest.raises(ToolError):
        correct_clip(str(ov), 2, "0,0,320")


def test_always_split_keeps_the_face_on_top_and_the_game_below(monkeypatch):
    import streamclipper.options as options

    meta = {"width": VW, "height": VH, "dur": 30.0}
    monkeypatch.setattr(options, "black_spans", lambda *a: [])
    # a big cam lower-left: no crop clear of it exists, so 5.4 fell back to a face-only clip
    big = {
        "kind": "corner-cam",
        "corner": "bottom_left",
        "face_box": [0.3, 0.35, 0.45, 0.7],
        "cam_box": [0.0, 0.2, 0.7, 1.0],
    }
    monkeypatch.setattr(options, "classify_layout", lambda *a: dict(big))
    o, cam, game, note = options.clip_layout("x.mp4", meta, _opts(cam_fit="tight", game_anchor="center"), 0, 10)
    assert o["layout"] == "full" and cam is None  # the old behaviour: a face crop
    o, cam, game, note = options.clip_layout(
        "x.mp4", meta, _opts(cam_fit="tight", game_anchor="center", always_split=True), 0, 10
    )
    assert o["layout"] == "split" and cam is not None and game is not None
    assert game[3] - game[1] == 1.0 and game[0] >= 0.5  # full height, moved away from the streamer
    assert "person_box" not in o  # the streamer may show at its edge: no streamer_clear row
    # a big cam with no overlay border: the face crop on top, the rest of the picture below
    full = {"kind": "full-cam", "corner": None, "face_box": [0.42, 0.2, 0.58, 0.5], "cam_box": None}
    monkeypatch.setattr(options, "classify_layout", lambda *a: dict(full))
    o, cam, game, note = options.clip_layout("x.mp4", meta, _opts(always_split=True), 0, 10)
    assert o["layout"] == "split" and camfit._inside(full["face_box"], cam, 0.0) and game[3] == 1.0
    assert abs(_px_aspect(cam) - 1080 / 672) < 0.03


def test_fallback_game_box_moves_away_from_the_streamer():
    left = camfit.fallback_game_box([0.0, 0.3, 0.3, 1.0], "9:16", VW, VH)
    right = camfit.fallback_game_box([0.7, 0.3, 1.0, 1.0], "9:16", VW, VH)
    assert left[0] >= 0.3 and right[2] <= 0.7 and left[3] == right[3] == 1.0
    assert camfit.game_box_for([0.1, 0.5, 0.25, 0.75], "bottom_left", {"always_split": True, "game_anchor": "center"},
                               VW, VH, [0.0, 0.38, 0.5, 1.0])  # fmt: skip


def test_seam_captions_sit_on_the_cam_line():
    from streamclipper.layout import SEAM_DOWN, caption_anchor, layout_boxes

    lay = layout_boxes("9:16", "split")
    assert caption_anchor(lay, "seam") == (5, 672 + SEAM_DOWN)
    args = build_parser().parse_args(["clip-stream", "--input", "s.mp4", "--out", "w", "--caption-position", "seam",
                                      "--always-split"])  # fmt: skip
    assert args.caption_position == "seam" and args.always_split
