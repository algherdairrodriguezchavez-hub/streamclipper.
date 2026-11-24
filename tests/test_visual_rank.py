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

    def ask(self, frames, prompt):
        Judge.calls += 1
        assert len(frames) == visual_rank.FRAMES and "mira" in prompt
        start = int(frames[0].parent.name.split("_")[0]) / 1000
        return 'Claro: {"score": %d, "reason": "persecución"}' % (20 if start < 15 else 95)

    def close(self):
        pass


def test_parse_answer_and_blend():
    assert visual_rank.parse_answer('x {"score": 87, "reason": "choque"} y') == {"score": 87, "reason": "choque"}
    assert visual_rank.parse_answer("score: 140")["score"] == 100
    assert visual_rank.parse_answer("no sé") is None
    assert visual_rank.blend(80, 30) == 50


def test_rerank_puts_the_action_first_and_caches(tmp_path, video):
    Judge.calls = 0
    out = visual_rank.rerank(_clips(), video, "", tmp_path, "qwen-3b", judge_factory=Judge)
    assert [c["start"] for c in out] == [20.0, 30.0, 0.0, 10.0]
    assert [c["rank"] for c in out] == [1, 2, 3, 4]
    assert out[0]["vision"] == {"score": 95, "reason": "persecución", "audio_score": 78}
    assert out[0]["reason"].startswith("persecución")
    assert Judge.calls == 4 and json.loads((tmp_path / "vision_scores.json").read_text())
    again = visual_rank.rerank(_clips(), video, "", tmp_path, "qwen-3b", judge_factory=Judge)
    assert Judge.calls == 4 and [c["start"] for c in again] == [20.0, 30.0, 0.0, 10.0]  # cached


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
