"""
GTA-specialised moment judge: a local vision-language model (Qwen2.5-VL, on the Colab T4) watches time-stamped
frames of each candidate clip (gameplay and the streamer's cam are both in the frame), reads what the streamer
says (with timestamps) and where the voice peaks, and rates the moment on several criteria (action, surprise,
humour, the streamer's reaction, spectacle, curiosity, story). It also names the event, marks the strongest
frame (the cold-open hook) and writes a short Spanish title.

Final score = 30% audio/voice score + 70% weighted criteria; routine gameplay (driving around, menus) is
penalised; clips under the quality bar are dropped (fewer, better clips) and the same kind of event close in
time is kept once. The model runs on the machine (no API key). Any failure (no GPU, no transformers, out of
memory, unreadable answers) keeps the audio ranking, with a notice. Answers are cached in
<work_dir>/vision_scores.json, so a re-run (Colab disconnect) does not rate again.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

from streamclipper.state import write_json_atomic
from streamclipper.util import run_cmd

MODELS = {
    "qwen-3b": "Qwen/Qwen2.5-VL-3B-Instruct",  # ~7.5 GB in fp16: fits the T4 next to nothing else
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",  # loaded in 4 bit (~6 GB): better judgement, about 2x slower
}
CHOICES = ("off", *MODELS)
VERSION = "v2"
TOP_N = 20  # candidates rated (the rest keep their audio score, after the rated ones)
FRAMES = 10  # frames per candidate: 7 spread over the clip + 3 around the voice peak
FRAME_W = 448  # frame width given to the model (keeps the prompt small)
TEXT_CHARS = 900  # transcript characters given per candidate
AUDIO_WEIGHT = 0.3  # final = 0.3 audio + 0.7 vision criteria
ROUTINE_FACTOR = 0.6  # routine gameplay keeps 60% of its score
QUALITY_MIN = 50  # rated clips under this final score are dropped (at least one clip is kept)
HOOK_MIN = 6  # the strongest moment must rate at least this (0-10) to open the clip as a cold open
SAME_EVENT_S = 180.0  # the same kind of event within this many seconds is kept once
CRITERIA = {  # weights of the 0-10 criteria (sum 1)
    "accion": 0.22,
    "sorpresa": 0.18,
    "humor": 0.14,
    "reaccion": 0.16,
    "espectacular": 0.12,
    "curiosidad": 0.10,
    "historia": 0.08,
}
EVENTS = (
    "persecucion", "tiroteo", "explosion", "choque", "robo", "mision", "huida", "npc_gracioso", "caos",
    "graficos", "descubrimiento", "reaccion", "charla", "conduccion", "menu", "otro",
)  # fmt: skip
ROUTINE_EVENTS = ("conduccion", "menu", "charla")
PROMPT = """Eres editor experto en clips virales de GTA 6 para TikTok, Reels y Shorts.
Te muestro {n} fotogramas en orden de un momento de {dur:.0f} s de un directo. En cada fotograma se ve el \
juego y, en un recuadro, la cámara del streamer (mira su cara: sorpresa, risa, grito).
Lo que dice el streamer (segundo entre corchetes): "{text}"
{audio}
Criterios (0 a 10):
- accion: persecución policial, tiroteo, explosión, choque, huida, robo, pelea, peligro.
- sorpresa: algo inesperado, absurdo o caótico.
- humor: situación o comentario gracioso, NPC raro, fallo divertido.
- reaccion: el streamer reacciona de verdad (grita, se ríe, se asombra) a lo que pasa en el juego.
- espectacular: gráficos, iluminación, realismo o un lugar impresionante del juego.
- curiosidad: hace que alguien quiera seguir mirando.
- historia: tiene inicio, desarrollo y un momento final claro.
Un auto que solo circula, caminar, menús, mapas, pantallas de carga o charla sin nada en pantalla es rutina \
(rutina = true, notas bajas). No inventes: valora solo lo que se ve y se oye.
"momento" = número del fotograma con el instante más fuerte; "gancho" = qué tan fuerte es ese instante (0-10).
"titulo" = título corto en español, 2 a 4 palabras, en mayúsculas, sin inventar \
(ej. "¡PERSECUCIÓN ÉPICA!").
Responde SOLO con JSON:
{{"evento": "<{events}>", "rutina": false, "accion": 0, "sorpresa": 0, "humor": 0, "reaccion": 0, \
"espectacular": 0, "curiosidad": 0, "historia": 0, "momento": 1, "gancho": 0, "titulo": "", \
"razon": "<frase corta>"}}"""


def _words_text(clip: dict) -> str:
    """The clip's transcript with a [second] mark every ~5 s (seconds from the clip start)."""
    s0, out, last = float(clip["start"]), [], -99.0
    for w in clip.get("words") or []:
        t = float(w.get("s", s0)) - s0
        if t - last >= 5.0:
            out.append(f"[{max(0, int(t))}s]")
            last = t
        out.append(str(w.get("w", "")))
    text = re.sub(r"\s+", " ", " ".join(out)).replace('"', "'").strip()
    return (text[:TEXT_CHARS] + " ...") if len(text) > TEXT_CHARS else (text or "(sin voz)")


def _audio_text(clip: dict) -> str:
    p = clip.get("peak")
    if p is None:
        return ""
    return f"La voz del streamer sube más fuerte en el segundo {max(0, int(float(p) - float(clip['start'])))}."


def _key(model: str, clip: dict) -> str:
    return f"{VERSION}|{model}|{clip['start']:.2f}|{clip['end']:.2f}"


def _num(v, lo=0.0, hi=10.0):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return None


def parse_answer(text: str, n_frames: int = FRAMES):
    """The model's answer -> {score (0-100), criteria, evento, rutina, momento, gancho, titulo, reason} or None.
    A bare {"score": x} (or 'score: x') answer is accepted too, as the overall score."""
    m = re.search(r"\{.*\}", text or "", re.S)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except ValueError:
            try:
                data = json.loads(re.sub(r",\s*}", "}", m.group(0).replace("'", '"')))
            except ValueError:
                data = None
    if not isinstance(data, dict):
        s = re.search(r'"?score"?\s*[:=]\s*(\d{1,3})', text or "")
        data = {"score": s.group(1)} if s else None
    if not data:
        return None
    crit = {k: _num(data.get(k)) for k in CRITERIA}
    if all(v is None for v in crit.values()):
        overall = _num(data.get("score"), 0, 100)
        if overall is None:
            return None
        crit = {k: overall / 10 for k in CRITERIA}
    crit = {k: (v if v is not None else 0.0) for k, v in crit.items()}
    evento = str(data.get("evento") or "otro").strip().lower()
    evento = evento if evento in EVENTS else "otro"
    rutina = data.get("rutina")
    rutina = (rutina.strip().lower() in ("true", "si", "sí", "1")) if isinstance(rutina, str) else bool(rutina)
    momento = _num(data.get("momento"), 1, max(1, n_frames))
    gancho = _num(data.get("gancho"))
    score = round(10 * sum(CRITERIA[k] * crit[k] for k in CRITERIA))
    return {
        "score": int(score),
        "criteria": {k: round(v, 1) for k, v in crit.items()},
        "evento": evento,
        "rutina": rutina or evento in ROUTINE_EVENTS,
        "momento": int(momento) if momento is not None else None,
        "gancho": gancho if gancho is not None else round(max(crit["accion"], crit["sorpresa"], crit["humor"]), 1),
        "titulo": _clean_title(data.get("titulo")),
        "reason": str(data.get("razon") or data.get("reason") or "").strip()[:200],
    }


def _clean_title(t) -> str:
    t = re.sub(r"\s+", " ", str(t or "")).strip().strip('"').upper()
    if not t or len(t) > 34 or len(t.split()) > 6 or not re.search(r"[A-ZÁÉÍÓÚÑ]", t):
        return ""
    return t


def blend(audio_score: float, vision_score: float, routine: bool = False) -> int:
    s = AUDIO_WEIGHT * float(audio_score) + (1 - AUDIO_WEIGHT) * float(vision_score)
    return round(s * (ROUTINE_FACTOR if routine else 1.0))


def frame_times(clip: dict, n: int = FRAMES) -> list:
    """7 times spread over the clip + 3 around the voice peak (p-1, p, p+1), sorted (source seconds)."""
    s, e = float(clip["start"]), float(clip["end"])
    k = max(1, n - 3)
    step = (e - s) / k
    ts = [s + step * (i + 0.5) for i in range(k)]
    p = clip.get("peak")
    if p is not None and s <= float(p) <= e:
        ts += [min(e - 0.05, max(s, float(p) + d)) for d in (-1.0, 0.0, 1.0)]
    return sorted(round(t, 2) for t in ts)[:n]


def extract_frames(video, times: list, out_dir: Path, offset: float = 0.0) -> list:
    """[(jpeg, t)] one per time (seconds in the source; offset = where video starts in the source). video is a
    file or a direct stream URL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t in times:
        p = out_dir / f"t{int(round(t * 100)):09d}.jpg"
        if not p.is_file():
            run_cmd(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t - offset):.3f}",
                 "-i", str(video), "-frames:v", "1", "-an", "-vf", f"scale={FRAME_W}:-2", "-q:v", "4", str(p)],
                "frame for the vision model",
            )  # fmt: skip
        if p.is_file():
            out.append((p, t))
    return out


_STREAM = {}


def stream_url(url: str) -> str:
    """The direct URL of the link's 360p video-only stream (one yt-dlp call per run), or '' when unavailable."""
    if url not in _STREAM:
        import subprocess

        from streamclipper.util import ytdlp_cmd

        got = ""
        try:
            fmt = "bv*[height<=360][vcodec^=avc1]/bv*[height<=360]/wv*"
            r = subprocess.run(
                ytdlp_cmd() + ["-f", fmt, "-g", "--no-playlist", "--no-warnings", url],
                capture_output=True, text=True, timeout=180,
            )  # fmt: skip
            lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("http")]
            got = lines[0] if r.returncode == 0 and lines else ""
        except Exception:  # noqa: BLE001 - any problem: per-window downloads instead
            got = ""
        _STREAM[url] = got
    return _STREAM[url]


def candidate_frames(clip: dict, src: Path, url: str, work_dir: Path) -> list:
    """[(jpeg, t)] of a candidate: from the source video; for a link, read straight from its 360p stream (seeks
    only the needed bytes), else from a 360p download of just that window."""
    from streamclipper.hud.frames import has_video

    s, e = float(clip["start"]), float(clip["end"])
    out_dir = work_dir / "vision" / f"{int(s * 1000)}_{int(e * 1000)}"
    times = frame_times(clip)
    if src is not None and has_video(src):
        return extract_frames(src, times, out_dir)
    if not url:
        return []
    direct = stream_url(url)
    if direct:
        try:
            frames = extract_frames(direct, times, out_dir)
            if len(frames) >= len(times) // 2:
                return frames
        except Exception:  # noqa: BLE001 - expired or refused stream URL: download the window
            _STREAM[url] = ""
    from streamclipper.download import download_slice

    slice_mp4 = out_dir / "low.mp4"
    if not slice_mp4.is_file():
        download_slice(url, s, e, slice_mp4, max_height=360)
    return extract_frames(slice_mp4, times, out_dir, offset=s)


class QwenJudge:
    """Qwen2.5-VL through transformers; fp16 on the T4 (bf16 where the GPU has it), 7B in 4 bit."""

    def __init__(self, choice: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU (in Colab: Entorno de ejecución > Cambiar tipo > T4 GPU)")
        mid = MODELS.get(choice, choice)  # a full Hugging Face id also works
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kw = {"torch_dtype": dtype, "device_map": "cuda", "attn_implementation": "sdpa"}
        if choice == "qwen-7b":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
            )
        print(f"Loading the vision model {mid} (first time: a 4-8 GB download) ...")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(mid, min_pixels=128 * 28 * 28, max_pixels=FRAME_W * FRAME_W)
        self.model = AutoModelForImageTextToText.from_pretrained(mid, **kw).eval()

    def ask(self, frames: list, prompt: str, labels: list = None) -> str:
        """frames: JPEG paths; labels: one text per frame shown just before it (e.g. 'Fotograma 3, segundo 12')."""
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in frames]
        content = []
        for i, _ in enumerate(images):
            if labels:
                content.append({"type": "text", "text": labels[i]})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=220, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def _ask(judge, frames: list, prompt: str, labels: list) -> str:
    try:
        return judge.ask(frames, prompt, labels)
    except TypeError:  # a judge without labels support
        return judge.ask(frames, prompt)


def apply_rating(c: dict, got: dict, times: list) -> None:
    """Write one rating into its clip: blended score, reason, Spanish title, the hook moment (peak)."""
    c["vision"] = dict(got, audio_score=c["score"])
    c["score"] = blend(c["score"], got["score"], got["rutina"])
    if got["reason"]:
        c["reason"] = f"{got['reason']} | {c['reason']}"
    if got["titulo"]:
        c["hook_headline"] = got["titulo"]
    m = got.get("momento")
    s, e = float(c["start"]), float(c["end"])
    if m and times and 1 <= m <= len(times):
        t = float(times[m - 1])
        if s + 1.5 <= t <= e - 0.5:
            c["vision"]["peak"] = round(t, 2)
            c["peak"] = round(t, 2)  # the cold open shows this instant
    c["tease_ok"] = float(got.get("gancho") or 0) >= HOOK_MIN and not got["rutina"]


def _peak(c: dict) -> float:
    return float(c.get("peak", c["start"]))


def select_rated(rated: list) -> list:
    """Rated clips by final score, without clips under QUALITY_MIN (at least one kept) and without the same
    kind of event close in time (repeats)."""
    out = []
    for c in sorted(rated, key=lambda c: -c["score"]):
        v = c.get("vision") or {}
        if out and c["score"] < QUALITY_MIN:
            continue
        ev = v.get("evento")
        if (
            ev
            and ev not in ("otro", "reaccion")
            and any(
                (o.get("vision") or {}).get("evento") == ev and abs(_peak(o) - _peak(c)) < SAME_EVENT_S for o in out
            )
        ):
            continue
        out.append(c)
    return out


MOTION_SLOTS = 6  # quiet visual moments (motion scan) judged even when their audio score is low


def _to_rate(clips: list) -> list:
    """The clips the model judges: the TOP_N best by score, with up to MOTION_SLOTS of them kept for visual
    moments (motion scan) that the audio score ranks low."""

    def is_motion(c):
        return "motion" in (c.get("sources") or [])

    extra = [c for c in clips[TOP_N:] if is_motion(c)][:MOTION_SLOTS]
    cut = TOP_N - len(extra)
    return clips[:cut] + [c for c in clips[cut:TOP_N] if is_motion(c)] + extra


def rerank(clips: list, src, url: str, work_dir: Path, choice: str, judge_factory=None) -> list:
    """
    The clips judged and re-ordered (rank renumbered 1..n): the top TOP_N are rated by the model, weak ones and
    repeats are dropped, each rated clip gets clip['vision'] (criteria, event, title, hook moment), a Spanish
    hook headline, a cold-open peak and tease_ok. Unchanged on any failure.
    """
    if choice in ("", "off", None) or not clips:
        return clips
    work_dir = Path(work_dir)
    cache_p = work_dir / "vision_scores.json"
    try:
        cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.is_file() else {}
    except ValueError:
        cache = {}
    rated = _to_rate(clips)
    todo = [c for c in rated if _key(choice, c) not in cache]
    judge = None
    try:
        if todo:
            judge = (judge_factory or QwenJudge)(choice)
            for i, c in enumerate(todo, start=1):
                frames = candidate_frames(c, src, url, work_dir)
                if not frames:
                    continue
                s0 = float(c["start"])
                labels = [f"Fotograma {k + 1} (segundo {max(0, int(t - s0))})" for k, (_, t) in enumerate(frames)]
                prompt = PROMPT.format(
                    n=len(frames), dur=c["end"] - c["start"], text=_words_text(c), audio=_audio_text(c),
                    events="|".join(EVENTS),
                )  # fmt: skip
                got = parse_answer(_ask(judge, [p for p, _ in frames], prompt, labels), len(frames))
                print(
                    f"  vision {i}/{len(todo)}: {_mmss(c['start'])} -> "
                    + (f"{got['score']} {got['evento']} '{got['titulo']}'" if got else "no answer")
                )
                if got:
                    got["times"] = [t for _, t in frames]
                    cache[_key(choice, c)] = got
                    write_json_atomic(cache_p, cache)
    except Exception as e:  # noqa: BLE001 - any model problem keeps the audio ranking
        print(f"Vision ranking skipped ({type(e).__name__}: {str(e)[:200]}); the audio ranking is kept.")
        if not any(_key(choice, c) in cache for c in rated):
            return clips
    finally:
        if judge is not None and hasattr(judge, "close"):
            judge.close()
    ok = [c for c in rated if _key(choice, c) in cache]
    if len(ok) < max(1, len(rated) // 2):
        print(f"Vision ranking: only {len(ok)} of {len(rated)} clips were rated; the audio ranking is kept.")
        return clips
    for c in ok:
        got = cache[_key(choice, c)]
        apply_rating(c, got, got.get("times") or [])
    kept = select_rated(ok)
    dropped = len(ok) - len(kept)
    out = kept  # clips the model did not judge are left out: fewer clips, but judged ones
    for r, c in enumerate(out, start=1):
        c["rank"] = r
    print(
        f"Vision ranking: {len(ok)} moments judged by {MODELS.get(choice, choice)}; {len(kept)} kept, "
        f"{dropped} dropped (score under {QUALITY_MIN}, routine or a repeat)."
    )
    return out


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
