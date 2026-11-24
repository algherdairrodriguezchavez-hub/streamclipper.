import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from streamclipper import autorun, colab, commands
from streamclipper.util import ToolError

A = "https://www.youtube.com/watch?v=AAAAAAAAAAA"
B = "https://youtu.be/BBBBBBBBBBB?si=xyz"


@pytest.fixture
def drive(tmp_path, monkeypatch):
    """MyDrive = tmp_path; a fake clip_stream that 'transcribes' unless analysis/state.json has the same link."""
    monkeypatch.setenv("STREAMCLIPPER_OUT", str(tmp_path))
    monkeypatch.setattr(autorun, "_metadata", lambda source, cookies="": {"canal": "Streamer Uno", "titulo": "GTA 6"})
    runs = []

    def fake(args):
        out = Path(args.out)
        st = out / "analysis" / "state.json"
        old = json.loads(st.read_text()) if st.is_file() else {}
        transcribed = old.get("source_key") != args.url
        st.parent.mkdir(parents=True, exist_ok=True)
        st.write_text(json.dumps({"source_key": args.url}))
        if transcribed:
            (st.parent / "transcript.json").write_text(args.url)
        (out / "clip_01.mp4").write_bytes(args.url.encode())
        runs.append({"out": out, "url": args.url, "transcribed": transcribed, "profile": args.streamer_profile})
        doc = {"clips": [{"rank": 1, "title": "x"}]}
        return {"doc": doc, "plans": [{"name": "clip_01"}], "gate": [("clip_01", True)]}

    monkeypatch.setattr(commands, "clip_stream", fake)
    return tmp_path, runs


def _run(link, **kw):
    return autorun.process_link(link, max_clips=3, drafts=False, **kw)


def test_video_key_and_normalize():
    assert autorun.video_key(A) == "AAAAAAAAAAA"
    assert autorun.video_key(B) == "BBBBBBBBBBB"
    assert autorun.video_key("https://www.youtube.com/live/CCCCCCCCCCC?feature=share") == "CCCCCCCCCCC"
    assert autorun.video_key("youtube.com/watch?v=AAAAAAAAAAA&t=30s") == "AAAAAAAAAAA"
    assert autorun.video_key("https://www.twitch.tv/videos/123456") == "v123456"
    assert autorun.video_key("/content/drive/MyDrive/mi directo.mp4") == "mi_directo"
    assert autorun.normalize_source(" youtube.com/watch?v=AAAAAAAAAAA ") == "https://youtube.com/watch?v=AAAAAAAAAAA"
    with pytest.raises(ToolError):
        autorun.video_key("  ")


def test_two_links_get_two_folders_and_never_share_results(drive):
    root, runs = drive
    ra, rb = _run(A), _run(B)
    pa, pb = ra["project"]["dir"], rb["project"]["dir"]
    assert pa != pb and pa.parent == pb.parent == root / "StreamClipper" / "proyectos"
    assert pa.name.endswith("_Streamer_Uno_AAAAAAAAAAA") and pb.name.endswith("_BBBBBBBBBBB")
    assert (pa / "clip_01.mp4").read_bytes() == A.encode()  # B did not overwrite A
    assert (pb / "clip_01.mp4").read_bytes() == B.encode()
    assert [r["transcribed"] for r in runs] == [True, True]
    assert runs[0]["profile"].endswith("streamers/streamer_uno.json")
    rows = list(csv.DictReader((root / "StreamClipper" / "historial.csv").open(encoding="utf-8-sig")))
    assert {r["enlace"] for r in rows} == {A, B} and {r["estado"] for r in rows} == {"listo"}
    assert all(r["clips"] == "1" for r in rows)


def test_same_link_continues_new_version_copies_transcript_from_scratch_does_not(drive):
    root, runs = drive
    first = _run(A)["project"]["dir"]
    again = _run(A)["project"]["dir"]
    assert again == first and runs[-1]["transcribed"] is False
    v2 = _run(A, existing="nueva_version")["project"]["dir"]
    assert v2 != first and v2.name.endswith("_v2") and runs[-1]["transcribed"] is False
    v3 = _run(A, existing="desde_cero")["project"]["dir"]
    assert v3 not in (first, v2) and runs[-1]["transcribed"] is True
    assert len(autorun.load_index(root / "StreamClipper")) == 3
    assert (first / "clip_01.mp4").is_file()  # nothing was deleted


def test_old_transcript_outside_proyectos_is_copied_not_moved(drive):
    root, runs = drive
    old = root / "StreamClipper" / "analysis"
    old.mkdir(parents=True)
    pasted_before = "https://www.youtube.com/watch?v=AAAAAAAAAAA&t=10s"
    (old / "state.json").write_text(json.dumps({"source_key": pasted_before}))
    (root / "StreamClipper" / "clip_01.mp4").write_bytes(b"old clip")
    res = _run(A)
    assert runs[-1]["transcribed"] is False and runs[-1]["url"] == pasted_before
    assert (old / "state.json").is_file() and (root / "StreamClipper" / "clip_01.mp4").read_bytes() == b"old clip"
    assert (res["project"]["dir"] / "analysis" / "state.json").is_file()
    _run(B)
    assert runs[-1]["transcribed"] is True  # another video never takes that transcript


def test_errors_are_recorded_and_cookies_are_found(drive, monkeypatch):
    root, _ = drive

    def boom(**kw):
        raise ToolError("YouTube asked to 'Sign in to confirm you're not a bot'.", "use cookies")

    with pytest.raises(ToolError):
        autorun.process_link(A, pipeline=boom)
    (entry,) = autorun.load_index(root / "StreamClipper")
    assert entry["estado"].startswith("error:")
    assert autorun.find_cookies(root) == ""
    (root / "privado").mkdir()
    (root / "privado" / "cookies.txt").write_text("# Netscape")
    assert autorun.find_cookies(root) == str(root / "privado" / "cookies.txt")
    seen = {}
    autorun.process_link(A, pipeline=lambda **kw: seen.update(kw) or {"mp4s": []})
    assert seen["cookies"] == str(root / "privado" / "cookies.txt") and seen["render"] and seen["resume"]
    html = autorun.history_html(root / "StreamClipper")
    assert "proyectos" in html and entry["creado"] in html


def test_prepare_project_names_are_unique_and_dated(tmp_path):
    meta = lambda s, c="": {}  # noqa: E731 - metadata unreadable (bot check): still a valid folder
    p = autorun.prepare_project(A, tmp_path, now=datetime(2026, 11, 23), metadata_fn=meta)
    assert p["dir"].name == "2026-11-23_canal_AAAAAAAAAAA" and p["new"]
    with pytest.raises(ToolError):
        autorun.prepare_project(A, tmp_path, existing="otra")


def test_auto_notebook_cells():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(Path(__file__).resolve().parents[1] / "notebooks" / "StreamClipper_Automatico.ipynb"), 4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) == 4
    assert "ytaccess.setup()" in code[0] and "ytaccess.diagnose(" in code[3] and ",youtube" in code[0]
    assert ",visualrank" in code[0] and "visual_rank=ver_video" in code[1] and "min_pause=" in code[1]
    assert "game_follow=" in code[1] and "always_split=siempre_dividido" in code[1]
    for i, src in enumerate(code, start=1):
        compile(src, f"cell {i}", "exec")
    assert "process_link(" in code[1] and 'enlace = ""  #@param' in code[1]
    assert "history_html()" in code[2]
    assert colab.run_pipeline  # the cells call the same pipeline as the main notebook
