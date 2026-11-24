"""
Visual re-ranking: a local vision-language model (Qwen2.5-VL, on the Colab T4) watches frames of each
candidate clip, reads what is said in it, and rates how good it is as a short. The final score blends the
audio/voice score with the model's rating, so the clips with real action on screen go first.

The model runs on the machine (no API key, nothing sent to another service). Any failure (no GPU, no
transformers, out of memory, unreadable answers) leaves the ranking as it was, with a notice.
Ratings are cached in <work_dir>/vision_scores.json, so a re-run (Colab disconnect) does not rate again.
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
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",  # loaded in 4 bit (~6 GB): better judgement, slower
}
CHOICES = ("off", *MODELS)
TOP_N = 24  # candidates rated (the rest keep their audio score, after the rated ones)
FRAMES = 12  # frames per candidate
FRAME_W = 448  # frame width given to the model (keeps the prompt small)
VISION_WEIGHT = 0.6  # final score = 0.4 audio + 0.6 vision
TEXT_CHARS = 700  # transcript characters given per candidate
PROMPT = (
    "Eres editor de clips virales de GTA para TikTok, Reels y Shorts. Estas {n} imágenes son fotogramas en orden "
    "de un momento de {dur:.0f} segundos de un directo (el streamer juega y comenta). Lo que dice el streamer: "
    '"{text}".\n'
    "Valora de 0 a 100 si este momento funciona como clip corto: acción clara en pantalla (persecución, choque, "
    "tiroteo, huida, fallo gracioso, algo inesperado), reacción del streamer y que se entienda sin contexto. "
    "Un momento con el juego en pausa, menús, pantalla de carga o solo charla vale poco.\n"
    'Responde SOLO con JSON: {{"score": <0-100>, "reason": "<una frase corta en español>"}}'
)


def _words_text(clip: dict) -> str:
    text = " ".join(w.get("w", "") for w in clip.get("words") or []).strip()
    text = re.sub(r"\s+", " ", text).replace('"', "'")
    return (text[:TEXT_CHARS] + " ...") if len(text) > TEXT_CHARS else (text or "(sin voz)")


def _key(model: str, clip: dict) -> str:
    return f"{model}|{clip['start']:.2f}|{clip['end']:.2f}"


def parse_answer(text: str):
    """{'score': 0-100, 'reason': str} from the model's answer, or None when it holds no score."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except ValueError:
            data = None
    if not isinstance(data, dict):
        s = re.search(r'"?score"?\s*[:=]\s*(\d{1,3})', text or "")
        if not s:
            return None
        data = {"score": s.group(1)}
    try:
        score = max(0, min(100, int(float(data.get("score")))))
    except (TypeError, ValueError):
        return None
    return {"score": score, "reason": str(data.get("reason") or "").strip()[:200]}


def blend(audio_score: float, vision_score: float) -> int:
    return round((1 - VISION_WEIGHT) * float(audio_score) + VISION_WEIGHT * float(vision_score))


def _frame_times(s: float, e: float, n: int = FRAMES) -> list:
    step = (e - s) / n
    return [s + step * (i + 0.5) for i in range(n)]


def extract_frames(video: Path, times: list, out_dir: Path, offset: float = 0.0) -> list:
    """One JPEG per time (seconds in the source; offset = where video starts in the source)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        p = out_dir / f"f{i:02d}.jpg"
        if not p.is_file():
            run_cmd(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t - offset):.3f}",
                 "-i", str(video), "-frames:v", "1", "-vf", f"scale={FRAME_W}:-2", "-q:v", "4", str(p)],
                "frame for the vision model",
            )  # fmt: skip
        if p.is_file():
            paths.append(p)
    return paths


def candidate_frames(clip: dict, src: Path, url: str, work_dir: Path) -> list:
    """Frames of a candidate: from the source video, or (a link) from a 360p download of just that window."""
    from streamclipper.hud.frames import has_video

    s, e = float(clip["start"]), float(clip["end"])
    out_dir = work_dir / "vision" / f"{int(s * 1000)}_{int(e * 1000)}"
    times = _frame_times(s, e)
    if src is not None and has_video(src):
        return extract_frames(src, times, out_dir)
    if not url:
        return []
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
        kw = {"torch_dtype": dtype, "device_map": "cuda"}
        if choice == "qwen-7b":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
            )
        print(f"Loading the vision model {mid} (first time: a 4-8 GB download) ...")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(mid, min_pixels=128 * 28 * 28, max_pixels=FRAME_W * FRAME_W)
        self.model = AutoModelForImageTextToText.from_pretrained(mid, **kw).eval()

    def ask(self, frames: list, prompt: str) -> str:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in frames]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def rerank(clips: list, src, url: str, work_dir: Path, choice: str, judge_factory=None) -> list:
    """
    The clips re-ordered by blended score (rank renumbered 1..n). The top TOP_N are rated by the model;
    each rated clip gets clip['vision'] = {score, reason, audio_score}. Unchanged on any failure.
    """
    if choice in ("", "off", None) or not clips:
        return clips
    work_dir = Path(work_dir)
    cache_p = work_dir / "vision_scores.json"
    try:
        cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.is_file() else {}
    except ValueError:
        cache = {}
    rated = clips[:TOP_N]
    todo = [c for c in rated if _key(choice, c) not in cache]
    judge = None
    try:
        if todo:
            judge = (judge_factory or QwenJudge)(choice)
            for i, c in enumerate(todo, start=1):
                frames = candidate_frames(c, src, url, work_dir)
                if not frames:
                    continue
                prompt = PROMPT.format(n=len(frames), dur=c["end"] - c["start"], text=_words_text(c))
                got = parse_answer(judge.ask(frames, prompt))
                print(f"  vision {i}/{len(todo)}: {_mmss(c['start'])} -> {got['score'] if got else 'no answer'}")
                if got:
                    cache[_key(choice, c)] = got
                    write_json_atomic(cache_p, cache)
    except Exception as e:  # noqa: BLE001 - any model problem keeps the audio ranking
        print(f"Vision ranking skipped ({type(e).__name__}: {str(e)[:200]}); the audio ranking is kept.")
        if not any(_key(choice, c) in cache for c in rated):
            return clips
    finally:
        if judge is not None and hasattr(judge, "close"):
            judge.close()
    n_ok = sum(1 for c in rated if _key(choice, c) in cache)
    if n_ok < max(1, len(rated) // 2):
        print(f"Vision ranking: only {n_ok} of {len(rated)} clips were rated; the audio ranking is kept.")
        return clips
    for c in rated:
        got = cache.get(_key(choice, c))
        if got:
            c["vision"] = {"score": got["score"], "reason": got["reason"], "audio_score": c["score"]}
            c["score"] = blend(c["score"], got["score"])
            if got["reason"]:
                c["reason"] = f"{got['reason']} | {c['reason']}"
    out = sorted(rated, key=lambda c: -c["score"]) + clips[len(rated) :]
    for r, c in enumerate(out, start=1):
        c["rank"] = r
    print(f"Vision ranking: {n_ok} clips rated by {MODELS.get(choice, choice)}; new order by blended score.")
    return out


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
