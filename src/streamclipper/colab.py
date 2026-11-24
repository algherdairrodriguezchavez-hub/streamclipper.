"""
Google Colab helpers. The notebook only sets form values and calls run_pipeline(), so the whole Colab run
can be tested on a normal PC. No web UI or servers: outputs go to a folder (Drive on Colab, or /content
without Drive, then downloaded as a zip). Drive share links only when the user ticks share_on_drive.
"""

import base64
import html
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from streamclipper.env import cuda_status
from streamclipper.env import gpu_info as _gpu_name_vram
from streamclipper.util import ToolError, fwd, set_ytdlp_cookies

BOT_BLOCK_MARKERS = ("sign in to confirm", "not a bot", "confirm you're not a bot", "confirm you are not a bot")
BOT_BLOCK_HINT = (
    "YouTube often blocks Colab servers. Two options: (1) download the VOD on your PC, upload it to Google Drive "
    "and use its Drive path as the source (e.g. /content/drive/MyDrive/my_stream.mp4); or (2) export a "
    "cookies.txt from a browser where you are signed in to YouTube, upload it to Drive and set the cookies path. "
    "A cookies file gives access to your account: never share it, and delete it from Drive after the run."
)
DRAFTS_ZIP = "capcut_drafts.zip"


def is_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def gpu_info() -> dict:
    """{'name', 'vram_mb', 'cuda', 'detail'}; cuda is True when faster-whisper can use the GPU."""
    name, vram = _gpu_name_vram()
    try:
        ok, detail = cuda_status()
    except Exception as e:  # never stop the notebook on a GPU probe
        ok, detail = False, str(e)
    return {"name": name, "vram_mb": vram, "cuda": bool(ok), "detail": detail}


def drive_out_dir(folder: str = "StreamClipper") -> Path:
    """$STREAMCLIPPER_OUT/<folder> when set, /content/drive/MyDrive/<folder> on Colab with Drive, else ./<folder>."""
    env_out = os.environ.get("STREAMCLIPPER_OUT")
    if env_out:
        return Path(env_out) / folder
    drive = Path("/content/drive/MyDrive")
    base = drive if is_colab() and drive.is_dir() else Path.cwd()
    return base / folder


LIVE_SYNC_FILES = ("clips.json", "clips.csv", "index.html", "live_summary.json")


def live_work_dir(folder: str) -> Path:
    """
    Local disk for a live run's rolling buffer, windows and panels: $STREAMCLIPPER_WORK/<folder> when set,
    /content/streamclipper_live/<folder> on Colab, else ./streamclipper_live/<folder>. Never the Drive
    mount: segments written and deleted there would go to Drive's trash and count against the quota.
    """
    env_work = os.environ.get("STREAMCLIPPER_WORK")
    if env_work:
        return Path(env_work) / folder
    base = Path("/content") if is_colab() and Path("/content").is_dir() else Path.cwd()
    return base / "streamclipper_live" / folder


def sync_live_outputs(work: Path, dest: Path) -> list:
    """
    Copy the finished results of a live run from local disk to Drive: <name>.mp4, <name>_thumb.jpg,
    clips.json, clips.csv, index.html and live_summary.json (top-level files only; buffer/, windows/,
    panels/ and half-written .part files stay local). A file is copied again only when it changed.
    Returns the copied file names.
    """
    work, dest = Path(work), Path(dest)
    if not work.is_dir():
        return []
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(work.iterdir()):
        name = p.name
        clip = name.endswith(".mp4") and not name.endswith(".part.mp4")
        if not p.is_file() or not (clip or name.endswith("_thumb.jpg") or name in LIVE_SYNC_FILES):
            continue
        st, target = p.stat(), dest / name
        if target.is_file() and target.stat().st_size == st.st_size and target.stat().st_mtime >= st.st_mtime:
            continue
        tmp = target.with_name(name + ".tmp")
        shutil.copy2(p, tmp)
        os.replace(tmp, target)
        copied.append(name)
    return copied


def is_bot_block(text: str) -> bool:
    low = str(text or "").lower()
    return any(m in low for m in BOT_BLOCK_MARKERS)


def zip_drafts(draft_root: Path, names: list, zip_path: Path) -> Path:
    """Zip the given draft folders (paths inside the zip start with the draft name)."""
    draft_root, zip_path = Path(draft_root), Path(zip_path)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name in names:
            d = draft_root / name
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(draft_root).as_posix())
    os.replace(tmp, zip_path)
    return zip_path


def _clip_argv(source: str, work: Path, o: dict) -> list:
    argv = ["clip-stream", "--out", str(work), "--prefix", o["prefix"]]
    if re.match(r"^https?://", source, re.I):
        argv += ["--url", source]
    else:
        src = Path(source)
        if not src.is_file():
            raise ToolError(
                f"Source video not found: {source}",
                "use a stream link, or the Drive path of an uploaded file (e.g. /content/drive/MyDrive/my_stream.mp4)",
            )
        argv += ["--input", str(src.resolve())]
    argv += ["--max-clips", str(int(o["max_clips"])), "--min-score", str(float(o["min_score"]))]
    argv += ["--llm", o["llm"], "--whisper-model", o["whisper_model"]]
    argv += ["--device", o["device"], "--language", o["language"]]
    for flag, key in (
        ("--ratio", "ratio"),
        ("--preset", "preset"),
        ("--caption-style", "caption_style"),
        ("--hook-style", "hook_style"),
        ("--llm-model", "llm_model"),
        ("--chat", "chat"),
    ):
        if o.get(key):
            argv += [flag, str(o[key])]
    # 5.3 effects and HUD: only passed when not the default, so older callers get the same argv
    if (o.get("zoom") or "off") != "off":
        argv += ["--zoom", str(o["zoom"])]
    if o.get("remove_fillers"):
        argv.append("--remove-fillers")
    if (o.get("enhance_speech") or "off") != "off":
        argv += ["--enhance-speech", str(o["enhance_speech"])]
    if o.get("brand_kit"):
        argv += ["--brand-kit", str(o["brand_kit"])]
    if (o.get("hud") or "auto") != "auto":
        argv += ["--hud", str(o["hud"])]
    # 5.4 per-clip crops and clip length: only passed when not the default, so older callers get the same argv
    for flag, key, default in (
        ("--cam-fit", "cam_fit", "standard"),
        ("--game-anchor", "game_anchor", "side"),
        ("--cam-track", "cam_track", "off"),
    ):
        if (o.get(key) or default) != default:
            argv += [flag, str(o[key])]
    for flag, key, default in (("--min-dur", "min_dur", 15.0), ("--target-dur", "target_dur", 30.0)):
        if float(o.get(key, default)) != default:
            argv += [flag, f"{float(o[key]):g}"]
    if float(o.get("max_dur", 60.0)) != 60.0:
        argv += ["--max-dur", f"{float(o['max_dur']):g}"]
    if float(o.get("min_pause", 0.55)) != 0.55:  # silences longer than this are cut (999 = keep every pause)
        argv += ["--min-pause", f"{float(o['min_pause']):g}"]
    if (o.get("caption_position") or "auto") != "auto":
        argv += ["--caption-position", str(o["caption_position"])]
    if o.get("always_split"):
        argv.append("--always-split")
    if (o.get("visual_rank") or "off") != "off":
        argv += ["--visual-rank", str(o["visual_rank"])]
    if o.get("streamer_profile"):
        argv += ["--streamer-profile", str(o["streamer_profile"])]
    if o.get("layout_overrides"):
        argv += ["--layout-overrides", str(o["layout_overrides"])]
    if o.get("preview_only"):
        argv.append("--preview-only")
    if o["render"] and not o.get("preview_only"):
        argv.append("--render")
    if o["drafts"] and not o.get("preview_only"):
        argv += ["--draft-root", str(o["draft_root"]), "--portable-drafts"]
    else:
        argv.append("--no-draft")
    if not o["resume"]:
        argv.append("--fresh")
    return argv


def run_pipeline(
    source: str,
    out_dir: str = "",
    preset: str = "",
    caption_style: str = "auto",
    hook_style: str = "bold-box",
    min_score: float = 50.0,
    max_clips: int = 10,
    ratio: str = "9:16",
    llm: str = "none",
    render: bool = True,
    drafts: bool = True,
    draft_root: str = "",
    cookies: str = "",
    resume: bool = True,
    whisper_model: str = "auto",
    device: str = "auto",
    language: str = "auto",
    prefix: str = "clip",
    llm_model: str = "",
    chat: str = "",
    zoom: str = "off",
    remove_fillers: bool = False,
    enhance_speech: str = "off",
    brand_kit: str = "",
    hud: str = "auto",
    cam_fit: str = "standard",
    game_anchor: str = "side",
    cam_track: str = "off",
    streamer_profile: str = "",
    layout_overrides: str = "",
    preview_only: bool = False,
    min_dur: float = 15.0,
    target_dur: float = 30.0,
    max_dur: float = 60.0,
    min_pause: float = 0.55,
    visual_rank: str = "off",
    caption_position: str = "auto",
    always_split: bool = False,
) -> dict:
    """
    Stream -> ranked clips (clips.json, clips.csv, index.html) -> MP4s + thumbnails (render) -> portable
    CapCut drafts zipped to <out_dir>/capcut_drafts.zip (drafts). Same code path as 'streamclipper clip-stream'.
    Progress is saved in <out_dir>; running it again with the same inputs resumes (resume=False starts over).
    max_clips = how many clips get an MP4/draft (0 = every clip above min_score; all clips go to clips.json).
    Effects (all off by default): zoom off|light|medium|strong, remove_fillers, enhance_speech off|light|strong,
    brand_kit = path of a brand kit JSON (e.g. on Drive). hud = auto|off|gta5|gta6|generic.
    Crops per clip: cam_fit standard|tight|close, game_anchor side|center, cam_track off|auto, streamer_profile =
    a JSON file of the streamer's cam positions (created/updated), layout_overrides = per-clip crops (the
    layout_overrides.json that preview_only=True writes, with "auto": false on the clips to correct).
    preview_only=True finds the clips and writes layout_preview/*.jpg + layout_overrides.json; renders nothing.
    min_dur/target_dur/max_dur = clip length in seconds (over 60 s the 59.5 s Shorts cap rises to max_dur, at
    most 180 s). min_pause = silences longer than this are cut (0.55 = podcast style; 3 = only long ones; 999 =
    keep the gameplay between lines). visual_rank = off|qwen-3b|qwen-7b: a local vision model (GPU) watches the
    best candidates and re-orders them. caption_position = auto|top|center|lower|seam (seam: captions and the
    title centred on the line between the facecam and the gameplay; the title shows first, alone).
    always_split=True keeps the facecam on top and the gameplay below on every clip with a face.
    None of these change the saved transcript.
    """
    from streamclipper.cli import build_parser  # late import: cli imports every command module
    from streamclipper.commands import clip_stream

    out = Path(out_dir or drive_out_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    droot = Path(draft_root).resolve() if draft_root else out / "capcut_drafts"
    if drafts:
        droot.mkdir(parents=True, exist_ok=True)
    opts = {
        "prefix": prefix,
        "max_clips": max_clips,
        "min_score": min_score,
        "ratio": ratio,
        "llm": llm or "none",
        "whisper_model": whisper_model or "auto",
        "device": device or "auto",
        "language": language or "auto",
        "preset": preset,
        "caption_style": caption_style,
        "hook_style": hook_style,
        "llm_model": llm_model,
        "chat": chat,
        "render": render,
        "drafts": drafts,
        "draft_root": droot,
        "resume": resume,
        "zoom": zoom or "off",
        "remove_fillers": bool(remove_fillers),
        "enhance_speech": enhance_speech or "off",
        "brand_kit": str(brand_kit or "").strip(),
        "hud": hud or "auto",
        "cam_fit": cam_fit or "standard",
        "game_anchor": game_anchor or "side",
        "cam_track": cam_track or "off",
        "streamer_profile": str(streamer_profile or "").strip(),
        "layout_overrides": str(layout_overrides or "").strip(),
        "preview_only": bool(preview_only),
        "min_dur": float(min_dur),
        "target_dur": float(target_dur),
        "max_dur": float(max_dur),
        "min_pause": float(min_pause),
        "visual_rank": visual_rank or "off",
        "caption_position": caption_position or "auto",
        "always_split": bool(always_split),
    }
    if opts["brand_kit"] and not Path(opts["brand_kit"]).is_file():
        raise ToolError("The brand kit file was not found.", "check the brand_kit path, or leave it empty")
    args = build_parser().parse_args(_clip_argv(str(source).strip(), out, opts))
    if cookies:
        if not Path(cookies).is_file():
            raise ToolError("The cookies file was not found.", "check the cookies path, or leave it empty")
        print("Using your cookies file for yt-dlp. It gives access to your account: never share it.")
        set_ytdlp_cookies(cookies)
    try:
        summary = clip_stream(args)
    except ToolError as e:
        if is_bot_block(f"{e} {e.hint}"):
            raise ToolError("YouTube asked to 'Sign in to confirm you're not a bot'.", BOT_BLOCK_HINT) from None
        raise
    finally:
        set_ytdlp_cookies(None)

    if preview_only:
        return {"out_dir": out, "preview": summary.get("preview") or [], "overrides": out / "layout_overrides.json"}
    result = collect_outputs(out, summary, droot if drafts else None)
    failed = [name for name, ok in summary["gate"] if not ok]
    if failed:
        raise ToolError(
            f"{len(failed)} MP4(s) failed the export gate: {', '.join(failed)} (see the tables above). "
            "The other outputs are saved.",
            f"check {fwd(out)}; re-run with other settings or skip those clips",
        )
    return result


def collect_outputs(out: Path, summary: dict, draft_root) -> dict:
    """Paths of everything a run produced, plus the drafts zip. Printed as a short list."""
    clips_json = out / "clips.json"
    doc = summary.get("doc") or {}
    plans = summary.get("plans") or []
    if summary.get("gate"):
        print("RENDER SUMMARY")
        for name, ok in summary["gate"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    mp4s = [out / f"{p['name']}.mp4" for p in plans if (out / f"{p['name']}.mp4").is_file()]
    thumbs = [m.with_name(f"{m.stem}_thumb.jpg") for m in mp4s if m.with_name(f"{m.stem}_thumb.jpg").is_file()]
    zip_path = None
    if draft_root is not None:
        names = [p.get("draft") for p in plans if p.get("draft") and (Path(draft_root) / p["draft"]).is_dir()]
        if names:
            zip_path = zip_drafts(Path(draft_root), names, out / DRAFTS_ZIP)
    result = {
        "out_dir": out,
        "clips_json": clips_json,
        "clips_csv": out / "clips.csv",
        "index_html": out / "index.html",
        "mp4s": mp4s,
        "thumbs": thumbs,
        "drafts_zip": zip_path,
        "clips": doc.get("clips") or [],
    }
    print("OUTPUTS")
    print(f"  folder      {fwd(out)}")
    print(f"  clips       {len(result['clips'])} in clips.json, clips.csv, index.html")
    print(f"  MP4s        {len(mp4s)} (+ {len(thumbs)} thumbnails)")
    if zip_path:
        print(f"  drafts zip  {fwd(zip_path)}  ->  on your PC: streamclipper import-draft --zip {DRAFTS_ZIP}")
    return result


ZIP_SUFFIXES = (".mp4", ".jpg", ".csv", ".html", ".json", ".zip")


def zip_outputs(out_dir, zip_path) -> Path:
    """The top-level results of a run (MP4s, thumbnails, clips.csv/json, index.html, drafts zip) in one zip."""
    out_dir, zip_path = Path(out_dir), Path(zip_path)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for p in sorted(out_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in ZIP_SUFFIXES or p.resolve() == zip_path.resolve():
                continue
            if p.name.endswith((".part.mp4", ".tmp")):
                continue
            stored = p.suffix.lower() in (".mp4", ".zip", ".jpg")  # already compressed
            zf.write(p, p.name, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    os.replace(tmp, zip_path)
    return zip_path


_COPY_JS = (
    "navigator.clipboard.writeText(this.parentElement.querySelector('.sc-t').innerText)"
    ".then(()=>{this.innerText='Copied'});return false;"
)


def _copy_line(label: str, text: str) -> str:
    if not text:
        return ""
    return (
        f"<div style='margin:4px 0'><b>{label}:</b> <span class='sc-t'>{html.escape(text)}</span> "
        f"<button onclick=\"{_COPY_JS}\" style='font-size:11px;padding:1px 8px;cursor:pointer'>Copy</button></div>"
    )


def results_html(result: dict, previews: int = 3, max_preview_mb: float = 15.0) -> str:
    """
    Result cards for IPython.display.HTML: thumbnail, an inline <video> for the first `previews` MP4s (base64,
    only files up to max_preview_mb so the notebook stays small), score, category, title/description/hashtags
    with Copy buttons and the Drive link when share_on_drive made one.
    """
    out = Path(result["out_dir"])
    cards, shown = [], 0
    for c in sorted(result.get("clips") or [], key=lambda c: c.get("rank", 0)):
        mp4 = out / c["mp4"] if c.get("mp4") else None
        media = ""
        if (
            mp4 is not None
            and shown < previews
            and mp4.is_file()
            and mp4.stat().st_size <= max_preview_mb * 1024 * 1024
        ):
            b64 = base64.b64encode(mp4.read_bytes()).decode("ascii")
            media = f"<video width='220' controls src='data:video/mp4;base64,{b64}'></video>"
            shown += 1
        elif c.get("thumb") and (out / c["thumb"]).is_file():
            b64 = base64.b64encode((out / c["thumb"]).read_bytes()).decode("ascii")
            media = f"<img width='220' src='data:image/jpeg;base64,{b64}' alt='thumbnail'>"
        score = max(0, min(100, int(float(c.get("score") or 0))))
        drive = str(c.get("drive_link") or "")
        link = (
            f"<div><a href='{html.escape(drive)}' target='_blank'>Drive link</a> "
            "(anyone with the link can see it)</div>"
            if drive.startswith("https://")
            else ""
        )
        cards.append(
            "<div style='display:inline-block;vertical-align:top;width:240px;margin:8px;padding:10px;"
            "border:1px solid #ccc;border-radius:10px;font-size:13px'>"
            f"{media}<div style='font-weight:bold;margin-top:6px'>#{html.escape(str(c.get('rank')))} &middot; "
            f"score {score} &middot; {html.escape(str(c.get('category') or ''))}</div>"
            f"<div style='height:6px;background:#eee;border-radius:3px'><div style='width:{score}%;height:6px;"
            "background:#8B5CF6;border-radius:3px'></div></div>"
            + _copy_line("Title", str(c.get("title") or c.get("hook_headline") or ""))
            + _copy_line("Description", str(c.get("description") or ""))
            + _copy_line("Hashtags", " ".join(c.get("hashtags") or []))
            + (f"<div style='color:#666'>{html.escape(str(c['mp4']))}</div>" if c.get("mp4") else "")
            + link
            + "</div>"
        )
    if not cards:
        return "<p>No clip passed the quality bar. Try a lower min_score.</p>"
    return "<div>" + "".join(cards) + "</div>"


def streamer_profile_path(name: str, folder: str = "StreamClipper/streamers") -> str:
    """Drive path of a streamer's profile JSON (MyDrive/StreamClipper/streamers/<name>.json); '' for no name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip()).strip("._")
    if not safe:
        return ""
    d = drive_out_dir(folder)
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{safe.lower()}.json")


def _parse_px(text: str, what: str):
    text = str(text or "").strip()
    if not text:
        return None
    try:
        vals = [int(round(float(v))) for v in text.replace(" ", "").split(",")]
    except ValueError:
        vals = []
    if len(vals) != 4:
        raise ToolError(f"The {what} box must be x,y,width,height in pixels, e.g. 0,0,480,300.", "fix the value")
    return vals


def correct_clip(
    overrides: str, rank: int, cam: str = "", game: str = "", layout: str = "split", ratio: str = "9:16"
) -> dict:
    """
    Set a manual crop for one clip of a layout_overrides.json written by the preview: cam/game as 'x,y,w,h' pixels
    of the source frame (empty game = chosen automatically clear of the streamer; layout 'full' needs no cam).
    Writes a new preview image of that clip and returns its entry. rank 0 or a missing clip raises ToolError.
    """
    from streamclipper import camfit
    from streamclipper.state import write_json_atomic

    data = camfit.load_overrides(overrides)
    entry = next((c for c in data["clips"] if isinstance(c, dict) and c.get("rank") == int(rank)), None)
    if entry is None:
        ranks = ", ".join(str(c.get("rank")) for c in data["clips"] if isinstance(c, dict))
        raise ToolError(f"Clip {rank} is not in the preview.", f"use one of: {ranks}")
    vw, vh = int(entry["width"]), int(entry["height"])
    cam_px, game_px = _parse_px(cam, "facecam"), _parse_px(game, "gameplay")
    if layout == "split" and cam_px is None:
        raise ToolError("A split layout needs the facecam box.", "fill in correct_cam, or choose layout full")
    new = dict(entry, auto=False, layout=layout, cam=cam_px if layout == "split" else None, game=game_px, segments=None)
    opts = {"ratio": ratio, "layout": layout, "game_anchor": "center"}
    _o, cam_n, game_n, _note = camfit.apply_override(new, dict(opts), vw, vh)
    if layout == "split" and game_n is None:
        corner = ("top" if (cam_n[1] + cam_n[3]) / 2 < 0.5 else "bottom") + (
            "_left" if (cam_n[0] + cam_n[2]) / 2 < 0.5 else "_right"
        )
        game_n = camfit.game_box_for(camfit.corner_face(cam_n, corner), corner, opts, vw, vh, cam_n)
        if game_n is None:
            raise ToolError("No gameplay area is left next to that facecam box.", "give correct_game too")
    crops = camfit.render_crops(layout, ratio, cam_n, game_n, vw, vh)
    new["cam"], new["game"] = crops["cam"], crops["game"]
    img = Path(overrides).parent / "layout_preview" / f"rank{int(rank):02d}_manual.jpg"
    t = (float(entry["start"]) + float(entry["end"])) / 2 - float(entry["window_start"])
    camfit.preview_image(Path(entry["window"]), t, vw, vh, layout, new["cam"], new["game"], ratio, img)
    new["images"], new["note"] = [fwd(img)], "manual crop (auto: false)"
    data["clips"] = [new if c is entry else c for c in data["clips"]]
    write_json_atomic(Path(overrides), data)
    return new


def preview_html(entries: list, max_width: int = 900) -> str:
    """Preview cards for IPython.display.HTML: each clip's crops drawn on the frame and the mock vertical clip."""
    parts = []
    for e in entries:
        head = f"<b>Clip {e.get('rank')}</b> ({_mmss_s(e.get('start'))}): {html.escape(str(e.get('note') or ''))}"
        if e.get("cam"):
            head += f"<br>facecam <code>{','.join(map(str, e['cam']))}</code>"
        if e.get("game"):
            head += f" &nbsp; gameplay <code>{','.join(map(str, e['game']))}</code>"
        imgs = []
        for p in e.get("images") or []:
            try:
                b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
            except OSError:
                continue
            imgs.append(f"<img src='data:image/jpeg;base64,{b64}' style='max-width:{max_width}px;width:100%'>")
        parts.append(f"<div style='margin:12px 0'>{head}<br>{''.join(imgs)}</div>")
    return "".join(parts) or "<p>No clips to preview.</p>"


def _mmss_s(t) -> str:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return "-"
    return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"
