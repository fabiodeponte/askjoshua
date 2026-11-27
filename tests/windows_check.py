"""Checks a fresh Joshua install on Windows: the page, a chat with the model, the pages, code, the browser, stop and start.
Each result is printed as a GitHub annotation, so it can be read from the run's summary."""
import os, pathlib, subprocess, sys, time, traceback, urllib.request

HOME = pathlib.Path.home() / "AskJoshua"
APP = HOME / "joshua"
URL = "http://127.0.0.1:7860"
results = []


def note(ok, title, text=""):
    results.append(ok)
    text = str(text).replace("\r", "").replace("\n", "%0A")[:3000]
    print(f"::{'notice' if ok else 'error'} title={'OK' if ok else 'FAIL'} {title}::{text}", flush=True)


def up(url=URL, wait=90):
    for _ in range(wait):
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def check(title):
    def wrap(fn):
        try:
            got = fn()
            note(True, title, got or "")
        except Exception:
            note(False, title, traceback.format_exc()[-2500:])
        return fn
    return wrap


@check("Joshua answers on 127.0.0.1:7860")
def _():
    assert up(), "nothing answered in 90 seconds"


@check("Chat with the model, Models, Settings and Memory pages")
def _():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        p.on("pageerror", lambda e: errors.append(str(e)))
        p.goto(URL, wait_until="load")
        p.wait_for_timeout(5000)
        box = p.locator("#chatcol textarea").first
        box.fill("Reply with just the word PONG.")
        box.press("Enter")
        t0 = time.time()
        answer = ""
        while time.time() - t0 < 240:
            p.wait_for_timeout(3000)
            bots = p.locator("#chat .message-row.bot-row, #chat .message.bot")
            if bots.count() and "JOSHUA IS WORKING" not in (p.locator("#working").inner_text() if p.locator("#working").count() else ""):
                answer = bots.last.inner_text().strip()
                if answer:
                    break
        assert answer, "no answer from the model in 4 minutes"
        out = [f"answer in {time.time() - t0:.0f}s: {answer[:200]}"]
        for name, sel in (("Models", "table.mrank tr"), ("Settings", "#toolson label"), ("Memory", "#memtopic")):
            p.get_by_role("button", name=name).first.click()
            p.wait_for_timeout(8000 if name == "Models" else 3000)
            n = p.locator(sel).count()
            assert n, f"{name} page: nothing found for {sel}"
            out.append(f"{name}: {n} items")
            back = p.get_by_role("button", name="Back to chat")
            if back.count():
                back.first.click(force=True)
                p.wait_for_timeout(1000)
        p.screenshot(path=str(HOME / "shot.png"))
        b.close()
        assert not errors, "page errors: " + "; ".join(errors[:5])
        return "\n".join(out)


@check("Folder button: the chat's files panel, and Open folder opens Explorer")
def _():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        p.on("pageerror", lambda e: errors.append(str(e)))
        p.goto(URL, wait_until="load")
        p.wait_for_timeout(5000)
        p.locator("#convs label").first.click()
        p.wait_for_timeout(2500)
        fb = p.locator("button.fbtn")
        out = [f"folder buttons: {fb.count()}"]
        fb.first.click()
        p.wait_for_timeout(1500)
        out.append("data-files after click: " + str(p.evaluate("document.body.dataset.files")))
        out.append("files panel visible: " + str(p.locator("#chatopts").is_visible()))
        p.screenshot(path=str(HOME / "shot-folder.png"))
        opener = p.get_by_role("button", name="Open folder")
        out.append(f"Open folder buttons: {opener.count()}, visible: {opener.first.is_visible() if opener.count() else None}")
        if opener.count():
            opener.first.click()
            p.wait_for_timeout(5000)
        b.close()
    ps = "(New-Object -ComObject Shell.Application).Windows() | ForEach-Object { $_.LocationURL }"
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    out.append("Explorer windows: " + (r.stdout.strip() or "(none)") + (" " + r.stderr.strip()[:300] if r.stderr.strip() else ""))
    log = HOME / "AskJoshua" / "joshua.log"
    if log.exists():
        out.append("log tail: " + log.read_text(errors="replace")[-600:])
    assert not errors, "page errors: " + "; ".join(errors[:5])
    assert "AskJoshua" in r.stdout or "conversations" in r.stdout, "no Explorer window on the chat's folder\n" + "\n".join(out)
    return "\n".join(out)


@check("Documents: PDF, Word and LibreOffice, with accents and other alphabets")
def _():
    code = r"""
import agent, pathlib, tempfile, warnings
warnings.simplefilter("always")
md = "# Città, Łódź e 北京 — prova\n\nTesto con **grassetto** e *corsivo*: àèéìòù “virgolette” € → ≥ α Привет 中文\n\n- uno\n  - sotto\n- due\n\n| A | B |\n|---|---|\n| Zażółć | 你好 |\n"
agent.set_workspace(tempfile.mkdtemp())
for ext in ("pdf", "docx", "odt"):
    print(agent.write_file("t." + ext, md))
    back = agent.read_file("t." + ext)
    for word in ("Città", "Łódź", "Привет", "中文", "“virgolette”"):
        assert word in back or word in back.replace(" ", "").replace("\\n", ""), (ext, word, back[:300])  # PDF text comes back with a space where the font changes
print("all three read back with every character")
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=APP, capture_output=True, text=True, timeout=180, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout + r.stderr[-1500:]


@check("Interface language from Windows' settings")
def _():
    code = "import ctypes, locale, i18n; k = ctypes.windll.kernel32; print(k.GetUserDefaultUILanguage(), locale.windows_locale.get(k.GetUserDefaultUILanguage()), locale.windows_locale.get(k.GetUserDefaultLCID()), '->', i18n.system_language())"
    r = subprocess.run([sys.executable, "-c", code], cwd=APP, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "->" in r.stdout, r.stdout + r.stderr
    return r.stdout


@check("Transcription on this computer (a short English recording)")
def _():
    import tempfile
    ws = pathlib.Path(tempfile.mkdtemp())
    wav = ws / "test.wav"
    say = "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.SetOutputToWaveFile('" + str(wav) + "'); $s.Speak('The city council approved the budget with twenty two votes in favour and ten against.'); $s.Dispose()"
    subprocess.run(["powershell", "-NoProfile", "-Command", say], check=True, timeout=60)
    code = "import agent, sys; agent.set_workspace(sys.argv[1]); print(agent.transcribe('test.wav', 'en'))"
    r = subprocess.run([sys.executable, "-c", code, str(ws)], cwd=APP, capture_output=True, text=True, timeout=1500, encoding="utf-8")
    assert "budget" in r.stdout.lower(), r.stdout[-1500:] + r.stderr[-1500:]
    return r.stdout[:600]


@check("Code tool (run_python)")
def _():
    r = subprocess.run([sys.executable, "-c", "import agent; print(agent.run_python('print(6*7, \"àèì\")'))"], cwd=APP, capture_output=True, text=True, timeout=120)
    assert "42 àèì" in r.stdout, r.stdout + r.stderr
    return r.stdout


@check("Joshua's browser (Firefox)")
def _():
    r = subprocess.run([sys.executable, "-c", "import browser; print(browser.browser_open('https://example.com')[:400])"], cwd=APP, capture_output=True, text=True, timeout=900)
    assert "Example Domain" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    return r.stdout


@check("Stop and start again")
def _():
    subprocess.run(["cmd", "/c", str(HOME / "stop.cmd")], timeout=60)
    time.sleep(3)
    assert not up(wait=3), "still running after stop.ps1"
    subprocess.run(["cmd", "/c", str(HOME / "start.cmd"), "-NoOpen"], timeout=180)
    assert up(), "didn't start again"


@check("Start menu shortcut starts Joshua (with PowerShell scripts forbidden)")
def _():
    lnk = pathlib.Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Joshua.lnk"
    assert lnk.exists(), "no shortcut"
    subprocess.run(["cmd", "/c", str(HOME / "stop.cmd")], timeout=60)
    time.sleep(3)
    assert not up(wait=3), "still running after stop.cmd"
    os.environ["JOSHUA_STAY_HIDDEN"] = "1"  # start without opening a browser window
    subprocess.run(["powershell", "-NoProfile", "-Command", f"Start-Process '{lnk}'"], timeout=60)
    assert up(), "the Start menu shortcut didn't start Joshua"
    return "started from the shortcut"


@check("Update from the site, then restart by itself")
def _():
    import threading
    from gradio_client import Client
    log = HOME / "joshua.log"
    size = log.stat().st_size if log.exists() else 0
    got = []
    threading.Thread(target=lambda: got.append(Client(URL, verbose=False).predict(api_name="/do_update")), daemon=True).start()
    steps = []
    for k in range(60):  # 10 minutes: what is running, every 10 seconds
        time.sleep(10)
        running = subprocess.run(["tasklist", "/fi", "imagename eq pythonw.exe", "/fo", "csv", "/nh"], capture_output=True, text=True).stdout.count("pythonw")
        steps.append(f"{(k + 1) * 10}s: pythonw x{running}, page {'up' if up(wait=1) else 'down'}, answered: {bool(got)}")
        if got and up(wait=1) and k > 3:
            break
    new = log.read_bytes()[size:].decode("utf-8", "replace") if log.exists() else "(no log)"
    report = f"returned: {got}%0A" + "%0A".join(steps[-12:]) + "%0A--- log ---%0A" + new[-2000:]
    assert up(wait=5), report
    return report


print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
