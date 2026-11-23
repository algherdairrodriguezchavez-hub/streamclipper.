"""
YouTube access on Colab: what yt-dlp needs today and a diagnosis when a download is refused.

Setup (yt-dlp wiki, EJS and PO Token Guide):
- yt-dlp with the yt-dlp-ejs solver scripts (`yt-dlp[default]`) and a JavaScript runtime: Deno (pip `deno`).
- A PO Token provider: the bgutil-ytdlp-pot-provider plugin (pip) and its HTTP server on 127.0.0.1:4416, built
  with Deno from the tagged release. yt-dlp finds the server at that address without extra options.
No player client is forced: yt-dlp's own defaults are used.

diagnose() tells apart: invalid cookies, missing JavaScript runtime / PO token, and YouTube refusing this
machine's IP (Colab is a cloud IP). Cookie values are never read into messages: only names, counts and expiry.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from streamclipper.util import ToolError, ytdlp_cmd

BGUTIL_VERSION = "2.0.2"
BGUTIL_REPO = "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
POT_URL = "http://127.0.0.1:4416"
LOGIN_COOKIES = ("SAPISID", "__Secure-1PSID", "__Secure-3PSID", "LOGIN_INFO", "SID")
COOKIE_HEADERS = ("# Netscape HTTP Cookie File", "# HTTP Cookie File")
VERDICTS = {
    "ok": "Todo bien: YouTube permite descargar desde este Colab.",
    "cookies": "Cookies inválidas o caducadas: crea un cookies.txt nuevo (ventana de incógnito, cuenta "
    "secundaria) y reemplaza MyDrive/privado/cookies.txt.",
    "cookies_formato": "El archivo de cookies no tiene el formato correcto (debe empezar por "
    "'# Netscape HTTP Cookie File'): expórtalo otra vez con la extensión 'Get cookies.txt LOCALLY'.",
    "js": "Falta el runtime de JavaScript (Deno) o los scripts yt-dlp-ejs: ejecuta otra vez la celda 1.",
    "pot": "Falta el PO Token o YouTube lo rechazó (error 403 al descargar): ejecuta otra vez la celda 1 para "
    "reiniciar el servidor de PO Token y vuelve a probar.",
    "ip": "YouTube bloquea la IP de este Colab: con cookies válidas, JavaScript y PO Token sigue pidiendo "
    "'confirma que no eres un bot'. Prueba más tarde o con otra sesión de Colab (Entorno de ejecución > "
    "Desconectar y borrar); si sigue, hace falta descargar desde otra conexión.",
    "red": "Esta máquina no consigue conectar con YouTube (red o proxy): revisa la conexión de Colab y prueba otra "
    "vez.",
    "otro": "Error distinto a un bloqueo: revisa el enlace (video privado, borrado o con restricción de edad).",
}

_BOT = ("sign in to confirm", "not a bot")
_COOKIE_BAD = ("cookies are no longer valid", "cookies have been rotated", "provided youtube account cookies")
_NET_BAD = ("unable to connect to proxy", "connection refused", "name or service not known",
            "temporary failure in name resolution", "network is unreachable", "timed out", "timeout")  # fmt: skip
_IP_BAD = ("http error 429", "too many requests", "this content isn't available, try again later",
           "your ip", "rate-limited", "rate limited")  # fmt: skip


def _ver(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return ""


def deno_path() -> str:
    exe = shutil.which("deno")
    if exe:
        return exe
    near = Path(sys.executable).with_name("deno.exe" if os.name == "nt" else "deno")
    return str(near) if near.is_file() else ""


def pot_server_up(url: str = POT_URL, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url + "/ping", timeout=timeout) as r:  # noqa: S310 - localhost only
            return r.status == 200
    except OSError:
        return False


def _run(cmd, cwd=None, timeout=900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def start_pot_server(home: Path = None, log=print, wait_s: float = 60.0) -> bool:
    """Clone + build the bgutil server (Deno) once and start it in the background; True when it answers."""
    if pot_server_up():
        return True
    deno = deno_path()
    if not deno:
        log("PO Token: Deno is not installed, the server cannot start.")
        return False
    home = Path(home or Path.home() / ".cache" / "streamclipper" / f"bgutil-{BGUTIL_VERSION}")
    server = home / "server"
    if not (server / "src" / "main.ts").is_file():
        shutil.rmtree(home, ignore_errors=True)
        home.parent.mkdir(parents=True, exist_ok=True)
        r = _run(["git", "clone", "-q", "--depth", "1", "--branch", BGUTIL_VERSION, BGUTIL_REPO, str(home)])
        if r.returncode != 0:
            log("PO Token: could not download the server: " + (r.stderr or "").strip()[-300:])
            return False
    if not (server / "node_modules").is_dir():
        r = _run([deno, "install", "--allow-scripts=npm:canvas", "--frozen"], cwd=server)
        if r.returncode != 0 or not (server / "node_modules").is_dir():
            log("PO Token: could not build the server: " + (r.stderr or "").strip()[-300:])
            return False
    logf = open(home / "server.log", "ab")  # noqa: SIM115 - the server keeps it open
    subprocess.Popen(
        [deno, "run", "--allow-env", "--allow-net", "--allow-ffi=.", "--allow-read=.", "../src/main.ts"],
        cwd=server / "node_modules",
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # keeps running between notebook cells
    )
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        if pot_server_up():
            return True
        time.sleep(1.0)
    log(f"PO Token: the server did not answer; see {home / 'server.log'}")
    return False


def setup(home: Path = None, log=print) -> dict:
    """Check yt-dlp, yt-dlp-ejs, Deno and the bgutil plugin; start the PO Token server. Prints one line each."""
    st = {
        "yt_dlp": _ver("yt-dlp"),
        "ejs": _ver("yt-dlp-ejs"),
        "deno": "",
        "plugin": _ver("bgutil-ytdlp-pot-provider"),
        "pot_server": False,
    }
    d = deno_path()
    if d:
        r = _run([d, "--version"], timeout=30)
        st["deno"] = (r.stdout.split() or ["?", "?"])[1] if r.returncode == 0 else ""
    if st["deno"] and st["plugin"]:
        st["pot_server"] = start_pot_server(home, log)
    log(f"yt-dlp {st['yt_dlp'] or 'NO INSTALADO'} | yt-dlp-ejs {st['ejs'] or 'NO INSTALADO'} | "
        f"Deno {st['deno'] or 'NO INSTALADO'} | PO Token: "
        f"{'servidor activo' if st['pot_server'] else 'NO DISPONIBLE'}")  # fmt: skip
    return st


def check_cookies(path) -> dict:
    """Format and YouTube login cookies of a cookies.txt; never returns or prints a cookie value."""
    res = {"path": str(path or ""), "exists": False, "format_ok": False, "youtube": 0, "login": [], "expired": [],
           "crlf": False}  # fmt: skip
    p = Path(path) if path else None
    if not p or not p.is_file():
        return res
    res["exists"] = True
    raw = p.read_bytes()
    res["crlf"] = b"\r\n" in raw
    lines = raw.decode("utf-8", errors="replace").splitlines()
    res["format_ok"] = bool(lines) and lines[0].strip() in COOKIE_HEADERS
    now = time.time()
    for ln in lines:
        if ln.startswith("#HttpOnly_"):
            ln = ln[len("#HttpOnly_") :]
        elif not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) < 7 or "youtube.com" not in f[0]:
            continue
        res["youtube"] += 1
        name = f[5]
        if name in LOGIN_COOKIES and name not in res["login"]:
            res["login"].append(name)
        try:
            exp = int(float(f[4]))
        except ValueError:
            exp = 0
        if name in LOGIN_COOKIES and 0 < exp < now and name not in res["expired"]:
            res["expired"].append(name)
    return res


def classify(text: str) -> str:
    """ok | cookies | js | pot | ip | otro from yt-dlp output (lower-cased search)."""
    low = str(text or "").lower()
    if any(m in low for m in _COOKIE_BAD):
        return "cookies"
    if any(m in low for m in ("no supported javascript runtime", "challenge solving failed",
                              "signature solving failed", "only images are available")):  # fmt: skip
        return "js"
    if any(m in low for m in _BOT):
        return "ip"
    if "http error 403" in low or ("po token" in low and ("missing" in low or "required" in low)):
        return "pot"
    if any(m in low for m in _IP_BAD):
        return "ip"
    if any(m in low for m in _NET_BAD):
        return "red"
    return "otro"


def _keep(line: str) -> bool:
    """yt-dlp's ERROR/WARNING lines and its JS runtime / PO token debug lines (never the command line)."""
    low = line.strip().lower()
    if "--cookies" in low or "command-line config" in low:
        return False
    return low.startswith(("error", "warning")) or "js runtimes" in low or "[pot]" in low


def probe(url: str, cookies: str = "", seconds: int = 10, timeout: float = 240.0) -> dict:
    """Download the first seconds of the audio (a real request, like a run makes). {ok, cause, lines}."""
    from streamclipper.util import set_ytdlp_cookies

    with tempfile.TemporaryDirectory() as tmp:
        set_ytdlp_cookies(cookies or None)
        try:
            cmd = ytdlp_cmd()
        finally:
            set_ytdlp_cookies(None)
        cmd += ["-v", "--no-playlist", "--no-progress", "-f", "ba[ext=m4a]/ba/b", "--download-sections",
                f"*0-{seconds}", "-o", str(Path(tmp) / "probe.%(ext)s"), url]  # fmt: skip
        try:
            r = _run(cmd, timeout=timeout)
            out, code = (r.stdout or "") + "\n" + (r.stderr or ""), r.returncode
        except subprocess.TimeoutExpired:
            out, code = "timeout", 1
        got = [p for p in Path(tmp).glob("probe.*") if p.stat().st_size > 0 and not p.name.endswith(".part")]
    ok = code == 0 and bool(got)
    lines = [ln.strip()[:240] for ln in out.splitlines() if _keep(ln)][-12:]
    return {"ok": ok, "cause": "ok" if ok else classify(out), "lines": lines}


def verdict(st: dict, ck: dict, without: dict, with_cookies: dict) -> str:
    """One of VERDICTS from the setup status, the cookie check and the two probes."""
    if with_cookies and with_cookies["ok"]:
        return "ok"
    if not with_cookies and without and without["ok"]:
        return "ok"
    if not (st.get("deno") and st.get("ejs")):
        return "js"
    if ck.get("exists") and not ck.get("format_ok"):
        return "cookies_formato"
    if ck.get("exists") and (not ck.get("login") or ck.get("expired")):
        return "cookies"
    if without and without["ok"]:
        return "cookies"  # works without cookies but not with them
    causes = {p["cause"] for p in (without, with_cookies) if p}
    for c in ("cookies", "js", "pot", "red"):
        if c in causes:
            return c
    if "ip" in causes:
        return "ip" if st.get("pot_server") else "pot"
    return "otro"


def diagnose(url: str, cookies: str = "", status: dict = None, log=print) -> dict:
    """Step-by-step check, printed in Spanish. Returns {verdict, setup, cookies, sin_cookies, con_cookies}."""
    if not str(url or "").strip():
        raise ToolError("No link given.", "paste a YouTube link")
    st = status or setup(log=log)
    ck = check_cookies(cookies)
    if cookies:
        log(f"Cookies: {'formato correcto' if ck['format_ok'] else 'FORMATO INCORRECTO'}, {ck['youtube']} de "
            f"YouTube, sesión: {', '.join(ck['login']) or 'NINGUNA (no hay sesión iniciada)'}"
            + (f", CADUCADAS: {', '.join(ck['expired'])}" if ck["expired"] else ""))  # fmt: skip
    else:
        log("Cookies: no se encontró cookies.txt (MyDrive/privado/cookies.txt)")
    log("Probando una descarga real de 10 s sin cookies ...")
    without = probe(url)
    log(f"  sin cookies: {'OK' if without['ok'] else 'FALLA (' + without['cause'] + ')'}")
    with_c = None
    if cookies and ck["exists"]:
        log("Probando una descarga real de 10 s con tus cookies ...")
        with_c = probe(url, cookies)
        log(f"  con cookies: {'OK' if with_c['ok'] else 'FALLA (' + with_c['cause'] + ')'}")
    v = verdict(st, ck, without, with_c)
    log("")
    log("DIAGNÓSTICO: " + VERDICTS[v])
    last = (with_c or without)["lines"]
    if v != "ok" and last:
        log("Detalle de yt-dlp:")
        for ln in last[-6:]:
            log("  " + ln)
    return {"verdict": v, "setup": st, "cookies": ck, "sin_cookies": without, "con_cookies": with_c}


def report_json(rep: dict) -> str:
    """The diagnosis as JSON (no cookie values are in it), to paste into a chat."""
    return json.dumps(rep, ensure_ascii=False, indent=1, default=str)
