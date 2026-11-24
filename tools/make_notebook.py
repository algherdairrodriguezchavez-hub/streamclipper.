"""
Build notebooks/StreamClipper_Colab.ipynb and notebooks/StreamClipper_Automatico.ipynb with nbformat,
so the notebooks are always valid JSON.

    python tools/make_notebook.py

Colab free-tier rules: only #@param forms and inline HTML output. No Gradio, Streamlit, tunnels or web servers;
Drive share links only when the user ticks share_on_drive.
"""

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from streamclipper.captions.styles import AUTO_STYLE, HOOK_STYLES, STYLES
from streamclipper.paths import PRESET_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "StreamClipper_Colab.ipynb"

INTRO = """# StreamClipper on Google Colab

Turn a long Twitch, YouTube or Kick stream into ranked vertical clips: ready-to-post MP4s with captions,
thumbnails, a `clips.csv`/`index.html` report and editable CapCut Desktop drafts (as a zip).

1. **Runtime > Change runtime type > T4 GPU** (free tier works; CPU also works, slower).
2. Run the cells from top to bottom. Fill in the form fields of cells 2, 5 and 6.
3. Everything is saved to your Google Drive in `MyDrive/StreamClipper/`. If Colab disconnects, run the cells
   again with the same inputs: the run resumes where it stopped. Untick `use_google_drive` in cell 4 to skip
   Drive: the files are then downloaded as a zip at the end (progress is not saved).

Drive access stays inside your own Google account. StreamClipper has no server.

Publishing to YouTube Shorts or Instagram Reels runs from your PC (`streamclipper publish` or the dashboard),
not from Colab: download the MP4s, then publish them there.

Clip only streams you have permission to clip (the streamer's permission or an official clipping program).
"""

CELL_GPU = """#@title 1. Check the GPU
import shutil
import subprocess

if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True
    )
    print("GPU:", out.stdout.strip())
    HAS_GPU = True
else:
    print("No GPU found. Speech recognition will run on the CPU, which is several times slower.")
    print("For a GPU: Runtime > Change runtime type > T4 GPU, then run this cell again.")
    HAS_GPU = False
"""

CELL_INSTALL = """#@title 2. Install StreamClipper
install_from = "github"  #@param ["github", "drive_zip"]
github_repo = "https://github.com/alokbaraii/streamclipper.git"  #@param {type:"string"}
drive_zip_path = "/content/drive/MyDrive/streamclipper.zip"  #@param {type:"string"}

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if shutil.which("ffmpeg") is None:
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)

# the gpu extra brings the CUDA 12 libraries faster-whisper needs (Colab's own CUDA may be another version)
EXTRAS = "asr,gpu,vision" if globals().get("HAS_GPU", False) else "asr,vision"
LOCAL_SRC = os.environ.get("STREAMCLIPPER_LOCAL_SRC", "")  # tests only: install from a local checkout
if LOCAL_SRC:
    project = Path("/tmp/streamclipper_src")
    shutil.rmtree(project, ignore_errors=True)
    shutil.copytree(LOCAL_SRC, project, ignore=shutil.ignore_patterns(".agents", ".git", "__pycache__"))
    target = f"{project}[{EXTRAS}]"
elif install_from == "github":
    target = f"streamclipper[{EXTRAS}] @ git+{github_repo}"
else:
    if (
        drive_zip_path.startswith("/content/drive")
        and not Path("/content/drive/MyDrive").is_dir()
        and not os.environ.get("STREAMCLIPPER_NO_DRIVE")
    ):
        from google.colab import drive

        drive.mount("/content/drive")
    src_dir = Path("/content/streamclipper_src")
    with zipfile.ZipFile(drive_zip_path) as zf:
        zf.extractall(src_dir)
    project = next(p.parent for p in src_dir.rglob("pyproject.toml"))
    target = f"{project}[{EXTRAS}]"
print("Installing", target, "...")
pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", target], capture_output=True, text=True)
if pip.returncode != 0:  # a notebook does not show the output of a child process: print it
    print(pip.stdout[-4000:], pip.stderr[-4000:])
    raise RuntimeError("pip install failed: see the lines above")
doctor = subprocess.run(["streamclipper", "doctor"], capture_output=True, text=True)
print(doctor.stdout, doctor.stderr)
"""

CELL_OLLAMA = """#@title 3. Optional: local LLM for titles, hooks and hashtags (needs the T4 GPU)
use_llm = True  #@param {type:"boolean"}
llm_model = "qwen2.5:7b"  #@param {type:"string"}

import shutil
import subprocess
import time
import urllib.request

LLM_READY = False
if use_llm and globals().get("HAS_GPU", False):
    if shutil.which("ollama") is None:
        subprocess.run(["apt-get", "-qq", "install", "-y", "zstd"], check=False)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    # Ollama runs as a local process inside this Colab machine only (localhost:11434); nothing is exposed.
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
            break
        except OSError:
            time.sleep(1)
    LLM_READY = subprocess.run(["ollama", "pull", llm_model]).returncode == 0
    print("LLM ready:", LLM_READY)
else:
    print("Skipped: clips are still found and scored; titles and hashtags stay empty without an LLM.")
"""

CELL_DRIVE = """#@title 4. Connect Google Drive (outputs and progress are saved there)
#@markdown Drive access stays inside your own Google account. StreamClipper has no server.
#@markdown Untick to work without Drive: outputs stay in /content and are downloaded as a zip at the end.
use_google_drive = True  #@param {type:"boolean"}
import os

if not use_google_drive or os.environ.get("STREAMCLIPPER_NO_DRIVE"):
    use_google_drive = False
    os.environ.setdefault("STREAMCLIPPER_OUT", "/content")
    print("Drive skipped: outputs go to", os.environ["STREAMCLIPPER_OUT"])
    print(
        "Without Google Drive your progress is not saved: if Colab disconnects, the run starts over. "
        "Your files are downloaded as a zip at the end."
    )
else:
    from google.colab import drive

    drive.mount("/content/drive")
    print("Drive access stays inside your own Google account. StreamClipper has no server.")
"""

CELL_INPUTS = """#@title 5. Your stream
#@markdown A YouTube/Twitch/Kick link, or the Drive path of a video you uploaded
#@markdown (e.g. `/content/drive/MyDrive/my_stream.mp4`). A Drive file avoids YouTube's bot check.
source = "https://www.youtube.com/watch?v=VIDEO_ID"  #@param {type:"string"}
#@markdown Optional, only if YouTube says "Sign in to confirm you're not a bot": the Drive path of a
#@markdown cookies.txt. **A cookies file gives access to your account. Never share it; delete it after use.**
cookies_path = ""  #@param {type:"string"}
output_folder = "StreamClipper"  #@param {type:"string"}

import os

source = os.environ.get("STREAMCLIPPER_TEST_INPUT") or source  # tests only: a local video instead of the link
"""

CELL_OPTIONS = """#@title 6. Options
#@markdown Caption styles (`auto` picks one per clip from its category):
#@markdown ![caption styles](https://raw.githubusercontent.com/alokbaraii/streamclipper/main/docs/media/styles_gallery.jpg)
preset = "streamer-split-9x16"  #@param {PRESETS}
caption_style = "auto"  #@param {STYLES}
hook_style = "bold-box"  #@param {HOOKS}
min_score = 50  #@param {type:"slider", min:0, max:100, step:5}
#@markdown How many of the best clips get an MP4 and a CapCut draft (0 = all above the minimum score).
#@markdown Every clip above the score is listed in clips.csv and index.html either way.
max_clips = 10  #@param {type:"integer"}
ratio = "9:16"  #@param ["9:16", "1:1"]
llm = "auto"  #@param ["auto", "ollama", "none"]
make_drafts = True  #@param {type:"boolean"}
#@markdown **Share on Drive** makes "anyone with the link can view" Drive links for the finished MP4s,
#@markdown index.html and clips.csv. Anyone with the link can see the clip. Default off.
share_on_drive = False  #@param {type:"boolean"}
#@markdown Game screen events (WASTED, BUSTED, mission passed, wanted stars): `auto` picks the game profile.
hud = "auto"  #@param ["auto", "off", "gta5", "gta6", "generic"]
#@markdown Editing effects (all off by default): punch-in zoom on the payoff, filler-word removal, speech cleanup.
zoom = "off"  #@param ["off", "light", "medium", "strong"]
remove_fillers = False  #@param {type:"boolean"}
enhance_speech = "off"  #@param ["off", "light", "strong"]
#@markdown **Facecam and gameplay crops, chosen per clip.** `cam_fit`: how close the facecam panel frames the face
#@markdown (`standard` = head and shoulders). `game_anchor`: `center` keeps the gameplay crop in the middle of the
#@markdown screen, clear of the streamer. `cam_track`: when the cam moves or turns off inside a clip, the clip is
#@markdown cut into parts with their own crops.
cam_fit = "tight"  #@param ["standard", "tight", "close"]
game_anchor = "center"  #@param ["side", "center"]
cam_track = True  #@param {type:"boolean"}
#@markdown Optional streamer name: their cam positions are saved in `MyDrive/StreamClipper/streamers/<name>.json`
#@markdown and help when a clip's cam is not found. Leave empty for no profile.
streamer = ""  #@param {type:"string"}
#@markdown Clip length in seconds (pauses are trimmed after this, so a clip ends a bit shorter).
min_dur = 15  #@param {type:"integer"}
target_dur = 30  #@param {type:"integer"}
max_dur = 60  #@param {type:"integer"}

LLM = ("ollama" if globals().get("LLM_READY", False) else "none") if llm == "auto" else llm
print("LLM:", LLM)
from streamclipper.colab import streamer_profile_path

PROFILE = streamer_profile_path(streamer) if globals().get("use_google_drive", True) else ""
if PROFILE:
    print("Streamer profile:", PROFILE)
"""

CELL_PREVIEW = """#@title 6b. Optional: preview the facecam and gameplay crops (renders nothing)
#@markdown Run once with `correct_clip = 0`: it finds the clips (the saved transcript is reused) and shows, per clip,
#@markdown the crops on the stream frame (red = facecam, cyan = gameplay) and the vertical clip they make.
#@markdown To fix one clip: set `correct_clip` to its number, type the boxes as `x,y,width,height` in pixels of the
#@markdown stream frame (read them off the yellow grid drawn on the frame), run again. Cell 7 then uses your boxes
#@markdown for that clip only; every other clip keeps its automatic crop.
correct_clip = 0  #@param {type:"integer"}
correct_layout = "split"  #@param ["split", "full"]
correct_cam = ""  #@param {type:"string"}
correct_game = ""  #@param {type:"string"}
from IPython.display import HTML, display

from streamclipper.colab import correct_clip as _correct, drive_out_dir, preview_html, run_pipeline
from streamclipper.util import ToolError

OUT_DIR = drive_out_dir(output_folder)
OVERRIDES = OUT_DIR / "layout_overrides.json"
try:
    if correct_clip:
        fixed = _correct(str(OVERRIDES), correct_clip, correct_cam, correct_game, correct_layout, ratio)
        display(HTML(preview_html([fixed])))
    else:
        prev = run_pipeline(
            source=source,
            out_dir=str(OUT_DIR),
            preset=preset,
            min_score=min_score,
            max_clips=max_clips,
            ratio=ratio,
            llm=LLM,
            llm_model=globals().get("llm_model", "") if LLM == "ollama" else "",
            cookies=cookies_path,
            hud=hud,
            cam_fit=cam_fit,
            game_anchor=game_anchor,
            cam_track="auto" if cam_track else "off",
            streamer_profile=PROFILE,
            preview_only=True,
            min_dur=min_dur,
            target_dur=target_dur,
            max_dur=max_dur,
        )
        display(HTML(preview_html(prev["preview"])))
except ToolError as e:
    print("ERROR:", e)
    if e.hint:
        print("Fix:", e.hint)
"""

CELL_RUN = """#@title 7. Find and make the clips (resumes after a disconnect)
from streamclipper.colab import drive_out_dir, run_pipeline
from streamclipper.drive_share import share_run
from streamclipper.util import ToolError

_missing = [cell for cell, name in (("5", "source"), ("6", "LLM")) if name not in globals()]
if _missing:
    result = None
    print(f"Run cell {' and '.join(_missing)} first, then run this cell again.")
else:
    OUT_DIR = drive_out_dir(output_folder)
    try:
        result = run_pipeline(
            source=source,
            out_dir=str(OUT_DIR),
            preset=preset,
            caption_style=caption_style,
            hook_style=hook_style,
            min_score=min_score,
            max_clips=max_clips,
            ratio=ratio,
            llm=LLM,
            llm_model=globals().get("llm_model", "") if LLM == "ollama" else "",
            render=True,
            drafts=make_drafts,
            cookies=cookies_path,
            resume=True,
            hud=hud,
            zoom=zoom,
            remove_fillers=remove_fillers,
            enhance_speech=enhance_speech,
            cam_fit=cam_fit,
            game_anchor=game_anchor,
            cam_track="auto" if cam_track else "off",
            streamer_profile=globals().get("PROFILE", ""),
            layout_overrides=str(OUT_DIR / "layout_overrides.json")
            if (OUT_DIR / "layout_overrides.json").is_file()
            else "",
            min_dur=min_dur,
            target_dur=target_dur,
            max_dur=max_dur,
        )
        if result and share_on_drive:
            share_run(result)
    except ToolError as e:
        result = None
        print("ERROR:", e)
        if e.hint:
            print("Fix:", e.hint)
"""

CELL_RESULTS = """#@title 8. Results and previews of the top 3 clips
from IPython.display import HTML, display

from streamclipper.colab import results_html

if result:
    display(HTML(results_html(result, previews=3)))
else:
    print("No results yet: see the error in cell 7.")
"""

CELL_OUTPUTS = """#@title 9. Your files (on Drive, or downloaded as a zip without Drive)
from pathlib import Path

from streamclipper.colab import is_colab, zip_outputs

if result:
    for p in sorted(Path(result["out_dir"]).glob("*")):
        if p.is_file() and p.suffix.lower() in (".mp4", ".jpg", ".csv", ".html", ".json", ".zip"):
            print(f"{p.stat().st_size / 1e6:8.1f} MB  {p}")
    print()
    print("Open index.html from Drive for the full ranked list.")
    if result["drafts_zip"]:
        print("CapCut drafts: download capcut_drafts.zip to your Windows PC, close CapCut, then run:")
        print("    streamclipper import-draft --zip capcut_drafts.zip")
    if not globals().get("use_google_drive", True):
        zp = zip_outputs(result["out_dir"], Path(result["out_dir"]).parent / "streamclipper_outputs.zip")
        print("Zip of your files:", zp)
        if is_colab():
            from google.colab import files

            files.download(str(zp))
"""

CELL_LIVE = """#@title 10. Live clipping (optional): clip a stream while it is still live
#@markdown Paste the link of a stream that is live right now (Twitch or YouTube Live; Kick is best effort).
#@markdown Cells 1, 2 and 4 must have run. Finished clips are copied to Drive while the stream runs
#@markdown (`MyDrive/StreamClipper/live/<channel>_<date>`); the rolling recording stays on the Colab disk
#@markdown (`/content/streamclipper_live`), so your Drive space is not used for it. Press the stop button
#@markdown to end early: the clips that are already queued are still finished.
live_url = ""  #@param {type:"string"}
stop_after_minutes = 60  #@param {type:"integer"}
min_score = 50  #@param {type:"slider", min:0, max:100, step:5}
max_clips_per_hour = 12  #@param {type:"integer"}
caption_style = "auto"  #@param {STYLES}

import datetime
import signal
import subprocess

from streamclipper.colab import drive_out_dir, live_work_dir, sync_live_outputs
from streamclipper.live.ingest import channel_name

if not live_url.strip():
    print("Paste a live stream link into live_url first.")
else:
    run_name = f"{channel_name(live_url)}_{datetime.date.today():%Y-%m-%d}"
    LIVE_OUT = drive_out_dir(f"StreamClipper/live/{run_name}")  # Drive: finished clips and the report only
    LIVE_WORK = live_work_dir(run_name)  # Colab disk: buffer/, windows/, panels/
    cmd = [
        "streamclipper", "live", "--url", live_url.strip(), "--out", str(LIVE_WORK),
        "--stop-after", str(int(stop_after_minutes)), "--min-score", str(min_score),
        "--max-clips-per-hour", str(int(max_clips_per_hour)), "--caption-style", caption_style,
        "--render", "--no-draft",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="")
            if line.startswith("CLIP "):  # a clip is finished: copy it to Drive now
                sync_live_outputs(LIVE_WORK, LIVE_OUT)
    except KeyboardInterrupt:
        print("Stopping: finishing the queued clips ...")
        proc.send_signal(signal.SIGINT)
        for line in proc.stdout:
            print(line, end="")
    proc.wait()
    sync_live_outputs(LIVE_WORK, LIVE_OUT)
    print("Clips and index.html are in", LIVE_OUT)
"""

OUTRO = """## Notes

- **YouTube bot check.** YouTube often blocks Colab servers with "Sign in to confirm you're not a bot".
  Upload the VOD to Drive and use its path, or use a cookies.txt (see cell 5; never share it).
- **Long streams.** An 8-10 hour stream works: speech recognition runs in 10-minute chunks and progress is
  saved after each chunk. The free tier can disconnect; run cells 2, 4, 5, 6 and 7 again to continue.
- **Kick and Twitch.** Links are passed to yt-dlp. Twitch chat replay is used when available; Kick has none.
"""


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [new_markdown_cell(INTRO)]
    choices = {
        "{PRESETS}": json.dumps(sorted(p.stem for p in PRESET_DIR.glob("*.json") if p.stem != "hook-grabber-9x16")),
        "{STYLES}": json.dumps([AUTO_STYLE] + list(STYLES)),
        "{HOOKS}": json.dumps(list(HOOK_STYLES)),
    }
    for src in (
        CELL_GPU,
        CELL_INSTALL,
        CELL_OLLAMA,
        CELL_DRIVE,
        CELL_INPUTS,
        CELL_OPTIONS,
        CELL_PREVIEW,
        CELL_RUN,
        CELL_RESULTS,
        CELL_OUTPUTS,
        CELL_LIVE,
    ):
        for key, val in choices.items():
            src = src.replace(key, val)
        cell = new_code_cell(src.rstrip("\n"))
        cell.metadata["cellView"] = "form"
        nb.cells.append(cell)
    nb.cells.append(new_markdown_cell(OUTRO))
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"  # stable ids: regenerating gives no git diff
    return nb


# ---------------------------------------------------------------------------------------------------------------
# Automatic mode (Spanish): one link -> one project folder, three cells. Same pipeline as the notebook above.
AUTO_OUT = ROOT / "notebooks" / "StreamClipper_Automatico.ipynb"
AUTO_REPO = "https://github.com/algherdairrodriguezchavez-hub/streamclipper.@claude/project-thread-ryl0jx"

AUTO_INTRO = """# StreamClipper automático: pega un enlace, recibe los clips

1. **Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU**.
2. Ejecuta la **celda 1** una vez por sesión (instala StreamClipper y conecta tu Google Drive).
3. Pega el enlace en la **celda 2** y pulsa ▶. Para otro enlace, cambia el enlace y pulsa ▶ otra vez.
4. La **celda 3** muestra la lista de todo lo procesado.
5. Si YouTube dice "confirma que no eres un bot", la **celda 4** hace un diagnóstico y dice la causa.

Cada enlace tiene su propia carpeta en `MyDrive/StreamClipper/proyectos/`: un enlace nuevo nunca usa ni
sobrescribe los clips de otro. Si Colab se desconecta, ejecuta la celda 1 y luego la 2 con el mismo enlace:
continúa donde se quedó, sin volver a transcribir.

YouTube suele pedir "confirma que no eres un bot" en Colab: guarda tu `cookies.txt` (de una cuenta secundaria) en
`MyDrive/privado/cookies.txt` y se usa automáticamente. Ese archivo da acceso a la cuenta: no lo compartas nunca.

Procesa solo directos con permiso del streamer (o de un programa oficial de clipping).
"""

AUTO_SETUP = """#@title 1. Preparar (una vez por sesión)
github_repo = "{AUTO_REPO}"  #@param {type:"string"}

import os
import shutil
import subprocess
import sys
from pathlib import Path

if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    )
    print("GPU:", out.stdout.strip())
    HAS_GPU = True
else:
    print("Sin GPU: funcionará, pero mucho más lento. Entorno de ejecución > Cambiar tipo > T4 GPU.")
    HAS_GPU = False

if shutil.which("ffmpeg") is None:
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)

# youtube = Deno + the PO Token plugin, which YouTube needs from cloud machines like Colab
EXTRAS = "asr,gpu,vision,youtube,visualrank" if HAS_GPU else "asr,vision,youtube"
LOCAL_SRC = os.environ.get("STREAMCLIPPER_LOCAL_SRC", "")  # tests only: install from a local checkout
if LOCAL_SRC:
    project = Path("/tmp/streamclipper_src")
    shutil.rmtree(project, ignore_errors=True)
    shutil.copytree(LOCAL_SRC, project, ignore=shutil.ignore_patterns(".agents", ".git", "__pycache__"))
    target = f"{project}[{EXTRAS}]"
else:
    target = f"streamclipper[{EXTRAS}] @ git+{github_repo}"
if globals().get("SC_INSTALLED") != target:
    print("Instalando StreamClipper (3 a 6 minutos) ...")
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", target], capture_output=True, text=True)
    if pip.returncode != 0:  # a notebook does not show the output of a child process: print it
        print(pip.stdout[-4000:], pip.stderr[-4000:])
        raise RuntimeError("La instalación falló: mira las líneas de arriba")
    SC_INSTALLED = target
else:
    print("StreamClipper ya está instalado en esta sesión.")

if os.environ.get("STREAMCLIPPER_NO_DRIVE"):
    os.environ.setdefault("STREAMCLIPPER_OUT", "/content")
else:
    from google.colab import drive

    drive.mount("/content/drive")

from streamclipper import ytaccess
from streamclipper.autorun import base_dir, find_cookies

print("Preparando el acceso a YouTube (JavaScript y PO Token, 1 a 3 minutos la primera vez) ...")
YT_STATUS = ytaccess.setup()
print("Tus proyectos se guardan en:", base_dir() / "proyectos")
COOKIES = find_cookies()
print("Cookies de YouTube:", COOKIES or "no encontradas (si YouTube pide verificar que no eres un bot, "
      "guarda tu cookies.txt en MyDrive/privado/cookies.txt)")
print("Listo. Ahora pega tu enlace en la celda 2.")
"""

AUTO_PROCESS = """#@title 2. Procesar un enlace (pega el enlace y pulsa ▶)
enlace = ""  #@param {type:"string"}
#@markdown Si este enlace ya se procesó: `continuar` sigue en su carpeta (reutiliza la transcripción y los clips),
#@markdown `nueva_version` crea otra carpeta (_v2) reutilizando la transcripción, `desde_cero` lo repite todo.
si_ya_existe = "continuar"  #@param ["continuar", "nueva_version", "desde_cero"]
#@markdown **Clips**
max_clips = 3  #@param {type:"integer"}
min_score = 40  #@param {type:"slider", min:0, max:100, step:5}
#@markdown Duración en segundos (Shorts y Reels aceptan hasta 180).
min_dur = 60  #@param {type:"integer"}
target_dur = 75  #@param {type:"integer"}
max_dur = 90  #@param {type:"integer"}
#@markdown Silencios: `mantener` deja el gameplay entre frases, `solo_largas` corta pausas de más de 3 s,
#@markdown `cortar` quita cada silencio (estilo podcast, clips mucho más cortos).
pausas = "mantener"  #@param ["mantener", "solo_largas", "cortar"]
#@markdown **Ver el video** con un modelo de visión en la GPU T4 (gratis, sin claves): elige los momentos con acción.
#@markdown `qwen-3b` es más rápido; `qwen-7b` juzga mejor y tarda más. `off` = solo audio y voz.
ver_video = "qwen-3b"  #@param ["off", "qwen-3b", "qwen-7b"]
#@markdown `hud = auto` también lee el HUD del juego (WASTED, estrellas). En GTA 6 es experimental y en un directo
#@markdown largo revisa miles de fotogramas y puede añadir mucho tiempo, por eso viene apagado.
hud = "off"  #@param ["off", "auto"]
#@markdown **Estilo**
caption_style = "auto"  #@param {STYLES}
hook_style = "clean"  #@param {HOOKS}
#@markdown `seam` = subtítulos y título en el centro, en la unión entre la cara y el gameplay (el título sale
#@markdown primero, solo, unos segundos; después los subtítulos).
caption_position = "seam"  #@param ["seam", "auto", "top", "center", "lower"]
#@markdown **Cámara y gameplay** (se eligen en cada clip). `streamer` vacío = se usa el nombre del canal.
#@markdown `siempre_dividido`: la cara del streamer arriba y el gameplay abajo en todos los clips.
siempre_dividido = True  #@param {type:"boolean"}
cam_fit = "tight"  #@param ["standard", "tight", "close"]
game_anchor = "center"  #@param ["side", "center"]
cam_track = True  #@param {type:"boolean"}
streamer = ""  #@param {type:"string"}
#@markdown Ruta de `cookies.txt` en Drive. Vacío = se busca sola en `MyDrive/privado/cookies.txt`.
cookies_path = ""  #@param {type:"string"}

import os

from IPython.display import HTML, display

if "SC_INSTALLED" not in globals():
    print("Primero ejecuta la celda 1 (Preparar) y después esta celda otra vez.")
elif not (os.environ.get("STREAMCLIPPER_TEST_INPUT") or enlace.strip()):
    print("Pega un enlace en el campo 'enlace' y pulsa ▶ otra vez.")
else:
    from streamclipper.autorun import process_link
    from streamclipper.colab import results_html
    from streamclipper.util import ToolError

    result = None
    try:
        result = process_link(
            os.environ.get("STREAMCLIPPER_TEST_INPUT") or enlace,  # tests only: a local video instead of the link
            existing=si_ya_existe,
            cookies=cookies_path.strip(),
            streamer=streamer.strip(),
            preset="streamer-split-9x16",
            caption_style=caption_style,
            hook_style=hook_style,
            min_score=min_score,
            max_clips=max_clips,
            ratio="9:16",
            llm="none",
            drafts=False,
            hud=hud,
            cam_fit=cam_fit,
            game_anchor=game_anchor,
            cam_track="auto" if cam_track else "off",
            min_dur=min_dur,
            target_dur=target_dur,
            max_dur=max_dur,
            min_pause={"mantener": 999, "solo_largas": 3, "cortar": 0.55}[pausas],
            visual_rank=ver_video,
            caption_position=caption_position,
            always_split=siempre_dividido,
        )
    except ToolError as e:
        print("ERROR:", e)
        if e.hint:
            print("Solución:", e.hint)
        if "not a bot" in str(e).lower():
            print("YouTube pide verificar que no eres un bot: guarda tu cookies.txt en MyDrive/privado/cookies.txt "
                  "(o escribe su ruta en cookies_path) y pulsa ▶ otra vez. Se continúa donde se quedó.")
    if result:
        print("Carpeta de este enlace:", result["project"]["dir"])
        display(HTML(results_html(result, previews=3)))
"""

AUTO_HISTORY = """#@title 3. Historial de enlaces procesados
from IPython.display import HTML, display

if "SC_INSTALLED" not in globals():
    print("Primero ejecuta la celda 1 (Preparar).")
else:
    from streamclipper.autorun import base_dir, history_html

    display(HTML(history_html()))
    print("También en Drive:", base_dir() / "historial.csv")
"""

AUTO_DIAGNOSE = """#@title 4. Diagnóstico de YouTube (si la celda 2 dice "not a bot")
#@markdown Hace dos descargas reales de 10 segundos (sin y con tus cookies) y dice la causa: cookies inválidas,
#@markdown falta JavaScript o PO Token, o bloqueo de la IP de Colab. No muestra el contenido de tus cookies.
enlace_prueba = ""  #@param {type:"string"}

if "SC_INSTALLED" not in globals():
    print("Primero ejecuta la celda 1 (Preparar).")
else:
    from streamclipper import ytaccess
    from streamclipper.autorun import find_cookies

    _link = enlace_prueba.strip() or globals().get("enlace", "").strip()
    if not _link:
        print("Pega un enlace de YouTube en 'enlace_prueba' y pulsa ▶.")
    else:
        DIAG = ytaccess.diagnose(_link, globals().get("cookies_path", "").strip() or find_cookies(),
                                 status=globals().get("YT_STATUS"))
"""


def build_auto() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [new_markdown_cell(AUTO_INTRO)]
    choices = {
        "{AUTO_REPO}": AUTO_REPO,
        "{STYLES}": json.dumps([AUTO_STYLE] + list(STYLES)),
        "{HOOKS}": json.dumps(list(HOOK_STYLES)),
    }
    for src in (AUTO_SETUP, AUTO_PROCESS, AUTO_HISTORY, AUTO_DIAGNOSE):
        for key, val in choices.items():
            src = src.replace(key, val)
        cell = new_code_cell(src.rstrip("\n"))
        cell.metadata["cellView"] = "form"
        nb.cells.append(cell)
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"auto-{i:02d}"
    return nb


def main():
    nb = build()
    nbformat.validate(nb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(OUT))
    print(f"Wrote {OUT}")
    auto = build_auto()
    nbformat.validate(auto)
    nbformat.write(auto, str(AUTO_OUT))
    print(f"Wrote {AUTO_OUT}")


if __name__ == "__main__":
    main()
