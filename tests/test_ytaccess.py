import time

import pytest

from streamclipper import autorun, util, ytaccess
from streamclipper.util import ToolError

SECRET = "VERY-SECRET-COOKIE-VALUE"


def _cookies(tmp_path, header="# Netscape HTTP Cookie File", expires=None, names=("SAPISID", "LOGIN_INFO")):
    exp = int(expires if expires is not None else time.time() + 86400)
    rows = [header] + [f".youtube.com\tTRUE\t/\tTRUE\t{exp}\t{n}\t{SECRET}" for n in names]
    rows.append(f"#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t{exp}\t__Secure-3PSID\t{SECRET}")
    rows.append(f".google.com\tTRUE\t/\tTRUE\t{exp}\tNID\t{SECRET}")
    p = tmp_path / "cookies.txt"
    p.write_text("\n".join(rows) + "\n")
    return p


def test_check_cookies_reads_names_never_values(tmp_path):
    ck = ytaccess.check_cookies(_cookies(tmp_path))
    assert ck["exists"] and ck["format_ok"] and ck["youtube"] == 3 and not ck["expired"]
    assert set(ck["login"]) == {"SAPISID", "LOGIN_INFO", "__Secure-3PSID"}
    assert SECRET not in repr(ck)
    old = ytaccess.check_cookies(_cookies(tmp_path, expires=time.time() - 60))
    assert "SAPISID" in old["expired"]
    assert not ytaccess.check_cookies(_cookies(tmp_path, header="SAPISID=1"))["format_ok"]
    assert ytaccess.check_cookies(_cookies(tmp_path, names=()))["login"] == ["__Secure-3PSID"]
    assert not ytaccess.check_cookies(tmp_path / "missing.txt")["exists"]


@pytest.mark.parametrize(
    "text,cause",
    [
        ("WARNING: [youtube] The provided YouTube account cookies are no longer valid", "cookies"),
        ("WARNING: [youtube] No supported JavaScript runtime could be found", "js"),
        ("WARNING: [youtube] n challenge solving failed: Some formats may be missing", "js"),
        ("ERROR: [youtube] x: Sign in to confirm you're not a bot.", "ip"),
        ("ERROR: unable to download video data: HTTP Error 403: Forbidden", "pot"),
        ("ERROR: HTTP Error 429: Too Many Requests", "ip"),
        ("ERROR: [youtube] x: Video unavailable", "otro"),
        ("ERROR: Unable to connect to proxy (timed out)", "red"),
    ],
)
def test_classify(text, cause):
    assert ytaccess.classify(text) == cause


def test_verdict_tells_the_causes_apart():
    good = {"deno": "2.9.8", "ejs": "0.8.0", "pot_server": True}
    ck_ok = {"exists": True, "format_ok": True, "login": ["SAPISID"], "expired": []}
    bot = {"ok": False, "cause": "ip", "lines": []}
    ok = {"ok": True, "cause": "ok", "lines": []}
    assert ytaccess.verdict(good, ck_ok, bot, ok) == "ok"
    assert ytaccess.verdict(good, ck_ok, bot, bot) == "ip"
    assert ytaccess.verdict({**good, "pot_server": False}, ck_ok, bot, bot) == "pot"
    assert ytaccess.verdict({**good, "deno": ""}, ck_ok, bot, bot) == "js"
    assert ytaccess.verdict(good, {**ck_ok, "login": []}, bot, bot) == "cookies"
    assert ytaccess.verdict(good, {**ck_ok, "expired": ["SAPISID"]}, bot, bot) == "cookies"
    assert ytaccess.verdict(good, {**ck_ok, "format_ok": False}, bot, bot) == "cookies_formato"
    assert ytaccess.verdict(good, ck_ok, bot, {"ok": False, "cause": "cookies", "lines": []}) == "cookies"
    assert ytaccess.verdict(good, ck_ok, ok, bot) == "cookies"  # works only without cookies
    assert set(ytaccess.VERDICTS) >= {"ok", "cookies", "cookies_formato", "js", "pot", "ip", "otro"}


def test_ytdlp_gets_deno_next_to_python_when_not_on_path(tmp_path, monkeypatch):
    fake_py = tmp_path / "bin" / "python"
    fake_py.parent.mkdir()
    (fake_py.parent / "deno").write_text("")
    monkeypatch.setattr(util.sys, "executable", str(fake_py))
    monkeypatch.setattr(util.shutil, "which", lambda name: None)
    assert util._js_runtime_args() == ["--js-runtimes", f"deno:{fake_py.parent / 'deno'}"]
    monkeypatch.setattr(util.shutil, "which", lambda name: "/usr/bin/" + name)
    assert util._js_runtime_args() == []


def test_blocked_download_runs_the_diagnosis_and_keeps_the_project(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMCLIPPER_OUT", str(tmp_path))
    monkeypatch.setattr(autorun, "_metadata", lambda s, c="": {"canal": "C"})

    def blocked(**kw):
        raise ToolError("YouTube asked to 'Sign in to confirm you're not a bot'.", "use cookies")

    seen = []
    diag = lambda url, cookies: seen.append(url) or {"verdict": "ip"}  # noqa: E731
    url = "https://www.youtube.com/watch?v=AAAAAAAAAAA"
    with pytest.raises(ToolError) as ei:
        autorun.process_link(url, pipeline=blocked, diagnose_fn=diag)
    assert seen == [url] and ei.value.verdict == "ip"
    (entry,) = autorun.load_index(tmp_path / "StreamClipper")
    assert entry["estado"] == "error: YouTube (ip)"
    # the same link later continues in the same folder
    res = autorun.process_link(url, pipeline=lambda **kw: {"mp4s": []}, diagnose_fn=diag)
    assert res["project"]["carpeta"] == entry["carpeta"]
