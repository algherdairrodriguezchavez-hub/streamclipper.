"""Visual re-ranking with a mocked vision model (the real Qwen2.5-VL needs a GPU and a model download)."""

import json
import subprocess

import pytest

from streamclipper import visual_rank
from streamclipper.cli import build_parser


def _clips(n=4):
    return [
        {"rank": i + 1, "start": 10.0 * i, "end": 10.0 * i + 6, "score": 80 - i, "reason": "voice peak",
         "words": [{"w": "mira", "s": 10.0 * i, "e": 10.0 * i + 0.4}]}
        for i in range(n)
    ]  # fmt: skip


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=40",
         "-f", "lavfi", "-i", "sine=duration=40", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(v)],
        check=True,
    )  # fmt: skip
    return v


class Judge:
    calls = 0

    def __init__(self, choice):
        self.choice = choice

    def ask(self, frames, prompt, labels=None):
        Judge.calls += 1
        assert len(frames) == visual_rank.FRAMES and "mira" in prompt and "GTA 6" in prompt
        assert labels and labels[0].startswith("Fotograma 1 (segundo")
        start = int(frames[0].parent.name.split("_")[0]) / 1000
        if start < 5:  # routine driving
            return '{"evento": "conduccion", "rutina": true, "accion": 2, "sorpresa": 1, "humor": 0, "reaccion": 1}'
        if start < 15:  # a chase, strong
            return (
                'Claro: {"evento": "persecucion", "rutina": false, "accion": 9, "sorpresa": 8, "humor": 3, '
                '"reaccion": 8, "espectacular": 7, "curiosidad": 8, "historia": 7, "momento": 6, "gancho": 9, '
                '"titulo": "¡Persecución épica!", "razon": "huye de la policía"}'
            )
        # the same kind of event again, close in time (a repeat), a bit weaker
        return (
            '{"evento": "persecucion", "accion": 7, "sorpresa": 6, "humor": 2, "reaccion": 6, "espectacular": 6, '
            '"curiosidad": 6, "historia": 6, "momento": 2, "gancho": 4, "titulo": "OTRA VEZ"}'
        )

    def close(self):
        pass


def test_parse_answer_and_blend():
    got = visual_rank.parse_answer(
        'x {"accion": 10, "sorpresa": 10, "humor": 10, "reaccion": 10, "espectacular": 10,'
        ' "curiosidad": 10, "historia": 10, "evento": "explosion", "titulo": "¡boom!"} y'
    )
    assert got["score"] == 100 and got["evento"] == "explosion" and got["titulo"] == "¡BOOM!" and not got["rutina"]
    assert visual_rank.parse_answer("score: 140")["score"] == 100  # a bare overall score still works
    assert visual_rank.parse_answer("no sé") is None
    assert visual_rank.parse_answer('{"evento": "menu", "accion": 5}')["rutina"]  # routine events
    assert visual_rank.blend(80, 30) == 45 and visual_rank.blend(80, 30, routine=True) == 27


def test_rerank_keeps_the_good_chase_drops_routine_and_repeats(tmp_path, video):
    Judge.calls = 0
    clips = _clips()
    for c in clips:
        c["peak"] = c["start"] + 3.0
    out = visual_rank.rerank(clips, video, "", tmp_path, "qwen-3b", judge_factory=Judge)
    assert [c["start"] for c in out] == [10.0]  # routine (0 s) dropped, later chases are repeats
    best = out[0]
    assert best["rank"] == 1 and best["hook_headline"] == "¡PERSECUCIÓN ÉPICA!" and best["tease_ok"]
    assert best["vision"]["evento"] == "persecucion" and best["vision"]["audio_score"] == 79
    assert best["peak"] == best["vision"]["peak"] and 11.5 <= best["peak"] <= 15.5  # frame 6 of the clip
    assert best["reason"].startswith("huye de la policía")
    assert Judge.calls == 4 and json.loads((tmp_path / "vision_scores.json").read_text())
    again = visual_rank.rerank(_clips(), video, "", tmp_path, "qwen-3b", judge_factory=Judge)
    assert Judge.calls == 4 and [c["start"] for c in again] == [10.0]  # cached


def test_a_broken_model_keeps_the_audio_ranking(tmp_path, video):
    def boom(choice):
        raise RuntimeError("CUDA out of memory")

    clips = _clips()
    assert visual_rank.rerank(clips, video, "", tmp_path, "qwen-7b", judge_factory=boom) == clips
    assert [c["rank"] for c in clips] == [1, 2, 3, 4] and "vision" not in clips[0]
    assert visual_rank.rerank(clips, video, "", tmp_path, "off") is clips


def test_flag_and_pipeline_argv():
    from streamclipper.colab import _clip_argv

    args = build_parser().parse_args(["clip-stream", "--input", "x.mp4", "--out", "o", "--visual-rank", "qwen-3b"])
    assert args.visual_rank == "qwen-3b"
    assert build_parser().parse_args(["clip-stream", "--input", "x.mp4", "--out", "o"]).visual_rank == "off"
    import inspect

    from streamclipper import colab

    assert inspect.signature(colab.run_pipeline).parameters["visual_rank"].default == "off"
    assert callable(_clip_argv)
