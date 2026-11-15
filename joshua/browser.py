"""Joshua's own browser: a Firefox window that the model can drive (open a page, read it, click, type, scroll).

It is a separate browser, not yours: none of your accounts or passwords are in it. Playwright drives it.
Playwright isn't thread-safe, and Joshua's tools run in different threads, so one thread owns the browser
and every command is handed to it through a queue.
"""
import pathlib, queue, re, subprocess, sys, threading

# Joshua's browser keeps its own profile here: sites you sign in to in its window stay signed in, also after a restart
PROFILE = pathlib.Path(__file__).parent / "browser-profile"

_jobs = queue.Queue()
_state = {"page": None, "engine": None}

# Clicks on buttons like these need you to say yes first, in the chat
RISKY = re.compile(r"\b(buy|pay|purchase|order|checkout|subscribe|send|post|publish|submit|delete|confirm|acquista|paga|ordina|invia|pubblica|elimina|conferma|abbonati)\b", re.I)

# Marks the buttons, links and fields on the page with numbers, and lists them
MARK = """() => {
  const sel = 'a[href], button, input:not([type=hidden]), select, textarea, [role=button], [role=link], [role=tab], [role=menuitem], [onclick], [contenteditable=true]';
  const out = []; let n = 0;
  document.querySelectorAll('[data-joshua]').forEach(e => e.removeAttribute('data-joshua'));
  for (const e of document.querySelectorAll(sel)) {
    const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    if (r.width < 2 || r.height < 2 || s.visibility === 'hidden' || s.display === 'none') continue;
    if (r.bottom < -200 || r.top > innerHeight * 3) continue;
    n += 1; e.setAttribute('data-joshua', n);
    const kind = e.tagName === 'A' ? 'link' : e.tagName === 'INPUT' ? 'field (' + (e.type || 'text') + ')' : e.tagName === 'TEXTAREA' ? 'text box' : e.tagName === 'SELECT' ? 'menu' : 'button';
    const name = (e.innerText || e.value || e.placeholder || e.getAttribute('aria-label') || e.title || e.name || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
    out.push('[' + n + '] ' + kind + (name ? ': ' + name : ''));
    if (n >= 150) break;
  }
  return out.join('\\n');
}"""


def _owner():
    """The one thread that owns the browser and runs every command."""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    ctx = None
    for engine in ("firefox", "chromium"):  # Firefox first; Chromium if Firefox won't start
        try:
            ctx = getattr(pw, engine).launch_persistent_context(str(PROFILE / engine), headless=False, viewport={"width": 1280, "height": 860})
            _state["engine"] = engine
            break
        except Exception:
            continue
    while True:
        fn, args, done = _jobs.get()
        try:
            if ctx is None:
                raise RuntimeError("No browser could start. Try again after Joshua installs it.")
            open_tabs = [p for p in ctx.pages if not p.is_closed()]
            # Joshua works in the tab you used last: open a page there yourself, sign in, then ask Joshua
            page = open_tabs[-1] if open_tabs else ctx.new_page()
            done.put((True, fn(page, *args)))
        except Exception as e:
            if "closed" in str(e).lower():  # you closed the window: start it again next time
                try:
                    ctx = getattr(pw, _state["engine"]).launch_persistent_context(str(PROFILE / _state["engine"]), headless=False, viewport={"width": 1280, "height": 860})
                    done.put((True, fn(ctx.pages[0] if ctx.pages else ctx.new_page(), *args)))
                    continue
                except Exception as e2:
                    e = e2
            done.put((False, str(e)))


def _install():
    """Playwright and its Firefox, the first time they're needed (about 90 MB)."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "playwright"], check=True, timeout=600)
    subprocess.run([sys.executable, "-m", "playwright", "install", "firefox"], check=True, timeout=900, capture_output=True)


_thread = None


def _run(fn, *args):
    global _thread
    if _thread is None or not _thread.is_alive():
        try:
            _install()
        except Exception as e:
            return f"Couldn't install the browser: {e}"
        _thread = threading.Thread(target=_owner, daemon=True)
        _thread.start()
    done = queue.Queue()
    _jobs.put((fn, args, done))
    try:
        ok, result = done.get(timeout=90)
    except queue.Empty:
        return "The browser didn't answer in time."
    return result if ok else f"Error: {result}"


def _view(page, note=""):
    """What the model sees after each step: where it is, the text, and the numbered things it can use."""
    page.wait_for_timeout(600)
    items = page.evaluate(MARK)
    text = page.evaluate("() => document.body ? document.body.innerText : ''")
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return (f"{note}Page: {page.title()}\nAddress: {page.url}\n\n--- Text ---\n{text[:6000]}"
            + ("\n[…the text goes on; use browser_scroll to see more]" if len(text) > 6000 else "")
            + f"\n\n--- Things you can use (give their number) ---\n{items or '(none)'}")


def browser_open(url: str) -> str:
    """Open a web page in Joshua's own browser window, to use it like a person: read it, click, fill in fields. Use this for sites that need clicking or typing; for just reading a page, read_page is faster.

    Args:
        url: the address to open, like https://example.com
    """
    if not re.match(r"https?://", url):
        url = "https://" + url
    return _run(lambda p, u: (p.goto(u, wait_until="domcontentloaded", timeout=45000), _view(p))[1], url)


def browser_read() -> str:
    """Read the page open in Joshua's browser again: its text and the numbered buttons, links and fields."""
    return _run(lambda p: _view(p))


def browser_click(number: int, confirmed: bool = False) -> str:
    """Click a button or link on the page, by the number browser_open or browser_read gave it.

    Args:
        number: the number of the button or link
        confirmed: true only after the user said yes in the chat to this exact click; needed for buttons that buy, pay, send, post, publish or delete
    """
    def go(p, n, ok):
        el = p.locator(f'[data-joshua="{n}"]').first
        label = (el.inner_text(timeout=3000) or el.get_attribute("value") or el.get_attribute("aria-label") or "").strip()
        if RISKY.search(label) and not ok:
            return f'Not clicked: "{label}" may buy, pay, send, post or delete something. Ask the user first; if they say yes, call browser_click again with confirmed=true.'
        el.scroll_into_view_if_needed(timeout=5000)
        el.click(timeout=8000)
        try:
            p.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        return _view(p, f'Clicked "{label[:60]}".\n')
    return _run(go, int(number), bool(confirmed))


def browser_type(number: int, text: str, press_enter: bool = False) -> str:
    """Type text into a field on the page, by its number. Never use it for passwords, card numbers or other secrets: those the user types.

    Args:
        number: the number of the field
        text: what to write in it
        press_enter: true to press Enter afterwards, for example to search
    """
    def go(p, n, txt, enter):
        el = p.locator(f'[data-joshua="{n}"]').first
        kind = (el.get_attribute("type") or "").lower()
        auto = (el.get_attribute("autocomplete") or "").lower()
        if kind == "password" or auto.startswith("cc-") or "password" in auto:
            return "Not typed: this is a password or payment field. The user must fill it in themselves, in the browser window."
        el.fill(txt, timeout=8000)
        if enter:
            el.press("Enter")
            try:
                p.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                pass
        return _view(p, "Typed.\n")
    return _run(go, int(number), str(text), bool(press_enter))


def browser_scroll(direction: str = "down") -> str:
    """Scroll the page down or up, to see more of it.

    Args:
        direction: down or up
    """
    return _run(lambda p, d: (p.mouse.wheel(0, 700 if d != "up" else -700), _view(p))[1], direction)


def browser_back() -> str:
    """Go back to the previous page."""
    return _run(lambda p: (p.go_back(wait_until="domcontentloaded", timeout=20000), _view(p))[1])


def show():
    """Open Joshua's browser window for you, so you can go to a site and sign in before asking Joshua to use it."""
    return _run(lambda p: (p.goto("https://duckduckgo.com") if p.url in ("", "about:blank") else None, p.bring_to_front(), "ok")[2])


TOOLS = [browser_open, browser_read, browser_click, browser_type, browser_scroll, browser_back]
