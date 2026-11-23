"""
One link -> one project folder: the "automatic mode" of the Colab notebook (StreamClipper_Automatico.ipynb).

Every link gets its own folder, MyDrive/StreamClipper/proyectos/<date>_<channel>_<video id>/, so a new link never
reuses or overwrites the clips of another one. proyectos/proyectos.json is the index (one entry per folder) and
historial.csv a copy of it to open in Google Sheets. The same link again either continues its folder (the
saved transcript and clips are reused), or starts a new version (_v2, _v3 ...) that copies the transcript or
redoes everything.

A transcript made before this mode existed (a run in MyDrive/<output_folder>/analysis) is copied, never moved,
into the new folder when its state.json was made from the same video, so it is not transcribed twice.
"""

import csv
import io
import json
import re
import shutil
from datetime import datetime
from hashlib import sha1
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from streamclipper.state import write_json_atomic
from streamclipper.util import ToolError, fwd

BASE_FOLDER = "StreamClipper"
PROJECTS = "proyectos"
INDEX = "proyectos.json"
HISTORY = "historial.csv"
EXISTING = ("continuar", "nueva_version", "desde_cero")
COOKIE_SPOTS = ("privado/cookies.txt", "StreamClipper/privado/cookies.txt", "cookies.txt")
HISTORY_COLUMNS = ("fecha", "estado", "clips", "canal", "titulo", "enlace", "carpeta")
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URLISH = re.compile(r"^(https?://|www\.|[\w-]+(\.[\w-]+)*\.(com|tv|be)(/|$))", re.I)


def video_key(source: str) -> str:
    """Stable id of a video: the YouTube id, twitch v<id>, a file's name, else a short hash of the link."""
    s = str(source or "").strip()
    if not s:
        raise ToolError("No link given.", "paste a YouTube, Twitch or Kick link into 'enlace'")
    if not _URLISH.match(s):  # a video file (e.g. on Drive)
        return _safe(Path(s).stem, 40) or sha1(s.encode("utf-8")).hexdigest()[:10]
    u = urlparse(s if "://" in s else "https://" + s)
    host = u.netloc.lower().split(":")[0]
    parts = [p for p in u.path.split("/") if p]
    if host.endswith("youtu.be") and parts and _YT_ID.match(parts[0]):
        return parts[0]
    if "youtube" in host:
        v = parse_qs(u.query).get("v", [""])[0]
        if _YT_ID.match(v):
            return v
        if len(parts) >= 2 and parts[0] in ("live", "shorts", "embed", "v") and _YT_ID.match(parts[1]):
            return parts[1]
    if "twitch" in host and len(parts) >= 2 and parts[0] == "videos" and parts[1].isdigit():
        return "v" + parts[1]
    return sha1(s.encode("utf-8")).hexdigest()[:10]


def normalize_source(source: str) -> str:
    """The link as pasted, trimmed, with https:// added when it has no scheme (a file path stays as it is)."""
    s = str(source or "").strip().strip('"').strip("'")
    if _URLISH.match(s) and not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    return s


def _safe(text: str, limit: int = 30) -> str:
    return re.sub(r"[^\w-]+", "_", str(text or ""), flags=re.UNICODE).strip("_")[:limit].strip("_")


def base_dir() -> Path:
    from streamclipper.colab import drive_out_dir

    return drive_out_dir(BASE_FOLDER)


def find_cookies(drive_root: Path = None) -> str:
    """The cookies.txt in a usual Drive spot (MyDrive/privado/cookies.txt first), or ''."""
    root = Path(drive_root) if drive_root else base_dir().parent
    for rel in COOKIE_SPOTS:
        p = root / rel
        if p.is_file():
            return str(p)
    return ""


def load_index(base: Path) -> list:
    p = Path(base) / PROJECTS / INDEX
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in data.get("proyectos", []) if isinstance(e, dict)] if isinstance(data, dict) else []


def _row(e: dict) -> list:
    keys = ("creado", "estado", "clips", "canal", "titulo", "enlace", "carpeta")  # same order as HISTORY_COLUMNS
    return [e.get(k, 0 if k == "clips" else "") for k in keys]


def save_index(base: Path, entries: list) -> None:
    base = Path(base)
    write_json_atomic(base / PROJECTS / INDEX, {"schema": 1, "proyectos": entries})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HISTORY_COLUMNS)
    for e in sorted(entries, key=lambda x: x.get("creado", ""), reverse=True):
        w.writerow(_row(e))
    tmp = base / (HISTORY + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8-sig")  # utf-8-sig: Excel/Sheets read the accents right
    tmp.replace(base / HISTORY)


def _metadata(source: str, cookies: str = "") -> dict:
    """{'canal', 'titulo'} from yt-dlp (no download); {} when it cannot be read (bot check, offline, a file)."""
    if not re.match(r"^https?://", str(source), re.I):
        return {"canal": Path(source).stem, "titulo": Path(source).stem}
    from streamclipper.download import video_info
    from streamclipper.util import set_ytdlp_cookies

    try:
        set_ytdlp_cookies(cookies or None)
        info = video_info(source)
    except (ToolError, OSError):
        info = {}
    finally:
        set_ytdlp_cookies(None)
    canal = info.get("channel") or info.get("uploader") or info.get("uploader_id") or ""
    return {"canal": str(canal), "titulo": str(info.get("title") or "")} if info else {}


def _legacy_analysis(source: str, base: Path) -> tuple:
    """(analysis dir, the link it was made from) of an earlier run of the same video outside proyectos/."""
    key = video_key(source)
    roots = [Path(base)] + sorted(p for p in Path(base).parent.iterdir() if p.is_dir() and p != Path(base))
    for root in roots:
        st = root / "analysis" / "state.json"
        if not st.is_file():
            continue
        try:
            old = json.loads(st.read_text(encoding="utf-8")).get("source_key") or ""
        except (OSError, ValueError, AttributeError):
            continue
        if re.match(r"^https?://", old, re.I) and video_key(old) == key:
            return root / "analysis", old
    return None, ""


def _copy_analysis(src: Path, dest_project: Path) -> bool:
    dest = dest_project / "analysis"
    if dest.exists() or not src.is_dir():
        return False
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("*.part", "*.ytdl", "*.tmp"))
    return True


def prepare_project(source: str, base: Path = None, existing: str = "continuar", cookies: str = "",
                    now: datetime = None, metadata_fn=None) -> dict:  # fmt: skip
    """
    The folder for this link (created if needed) and its index entry:
    {dir, entry, new, reused} where reused names where the transcript came from ('' = none).
    existing: continuar (same folder), nueva_version (new folder, copies the transcript), desde_cero (new
    folder, nothing reused).
    """
    if existing not in EXISTING:
        raise ToolError(f"Unknown option si_ya_existe={existing!r}.", "use one of: " + ", ".join(EXISTING))
    source = normalize_source(source)
    base = Path(base) if base else base_dir()
    (base / PROJECTS).mkdir(parents=True, exist_ok=True)
    key = video_key(source)
    entries = load_index(base)
    same = [e for e in entries if e.get("clave") == key and (base / PROJECTS / e.get("carpeta", "")).is_dir()]
    if same and existing == "continuar":
        entry = max(same, key=lambda e: e.get("creado", ""))
        print(f"This link was processed before: continuing in {entry['carpeta']}")
        return {"dir": base / PROJECTS / entry["carpeta"], "entry": entry, "new": False, "reused": entry["carpeta"]}

    now = now or datetime.now()
    meta = (metadata_fn or _metadata)(source, cookies) or {}
    canal = _safe(meta.get("canal") or "", 24) or "canal"
    name = f"{now:%Y-%m-%d}_{canal}_{_safe(key, 40)}"
    if same:
        name += f"_v{len(same) + 1}"
    folder, n = name, 1
    while (base / PROJECTS / folder).exists():
        n += 1
        folder = f"{name}_{n}"
    proj = base / PROJECTS / folder
    proj.mkdir(parents=True)

    run_source, reused = source, ""
    if same and existing == "nueva_version":
        prev = max(same, key=lambda e: e.get("creado", ""))
        if _copy_analysis(base / PROJECTS / prev["carpeta"] / "analysis", proj):
            run_source, reused = prev.get("enlace") or source, prev["carpeta"]
    elif not same and existing != "desde_cero":
        old_dir, old_source = _legacy_analysis(source, base)
        if old_dir is not None and _copy_analysis(old_dir, proj):
            run_source, reused = old_source, fwd(old_dir.parent)
    if reused:
        print(f"Transcript copied from {reused} (the original stays where it was)")

    entry = {
        "clave": key,
        "enlace": run_source,
        "carpeta": folder,
        "canal": meta.get("canal") or "",
        "titulo": meta.get("titulo") or "",
        "creado": f"{now:%Y-%m-%d %H:%M}",
        "estado": "en proceso",
        "clips": 0,
    }
    entries.append(entry)
    save_index(base, entries)
    print(f"New project folder: {fwd(proj)}")
    return {"dir": proj, "entry": entry, "new": True, "reused": reused}


def _update_entry(base: Path, folder: str, **changes) -> None:
    entries = load_index(base)
    for e in entries:
        if e.get("carpeta") == folder:
            e.update(changes)
    save_index(base, entries)


def process_link(source: str, existing: str = "continuar", cookies: str = "", streamer: str = "",
                 base: Path = None, pipeline=None, **options) -> dict:  # fmt: skip
    """
    Paste a link, get the clips: prepare_project, then run_pipeline into that folder (render on, resume on).
    The streamer profile defaults to the channel name. Returns run_pipeline's result plus 'project'.
    options go to run_pipeline (max_clips, min_score, hook_style, cam_fit, min_dur ...).
    """
    from streamclipper.colab import run_pipeline, streamer_profile_path

    base = Path(base) if base else base_dir()
    cookies = cookies or find_cookies(base.parent)
    if cookies and not Path(cookies).is_file():
        raise ToolError("The cookies file was not found.", f"check the path {cookies}, or leave it empty")
    proj = prepare_project(source, base, existing, cookies)
    entry = proj["entry"]
    profile = streamer_profile_path(streamer or entry.get("canal") or "")
    try:
        result = (pipeline or run_pipeline)(
            source=entry["enlace"],
            out_dir=str(proj["dir"]),
            cookies=cookies,
            streamer_profile=profile,
            render=True,
            resume=True,
            **options,
        )
    except ToolError as e:
        _update_entry(base, entry["carpeta"], estado="error: " + str(e)[:120])
        raise
    mp4s = result.get("mp4s") or []
    _update_entry(base, entry["carpeta"], estado="listo", clips=len(mp4s))
    result["project"] = {**entry, "estado": "listo", "clips": len(mp4s), "dir": proj["dir"]}
    print(f"Done: {len(mp4s)} clip(s) in {fwd(proj['dir'])}")
    return result


def history_html(base: Path = None) -> str:
    """The project list as an HTML table for the notebook (newest first)."""
    base = Path(base) if base else base_dir()
    entries = sorted(load_index(base), key=lambda e: e.get("creado", ""), reverse=True)
    if not entries:
        return "<p>Todavía no hay proyectos. Procesa un enlace en la celda 2.</p>"
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in _row(e)) + "</tr>" for e in entries)
    head = "".join(f"<th>{c}</th>" for c in HISTORY_COLUMNS)
    return (
        f"<p>{len(entries)} proyecto(s) en {escape(fwd(base / PROJECTS))}</p>"
        f"<table style='border-collapse:collapse' border='1' cellpadding='4'><tr>{head}</tr>{rows}</table>"
    )
