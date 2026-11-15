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
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HOME / "stop.ps1")], timeout=60)
    time.sleep(3)
    assert not up(wait=3), "still running after stop.ps1"
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HOME / "start.ps1"), "-NoOpen"], timeout=180)
    assert up(), "didn't start again"


@check("Start menu shortcut")
def _():
    lnk = pathlib.Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Joshua.lnk"
    assert lnk.exists(), "no shortcut"


print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
