import datetime, difflib, json, os, pathlib, re, shutil, subprocess, sys, threading
import docx, lxml.html, ollama, requests, trafilatura
from langdetect import DetectorFactory, detect
DetectorFactory.seed = 0
from fpdf import FPDF
from pypdf import PdfReader
from ddgs import DDGS

CONFIG_FILE = pathlib.Path(__file__).parent / "config.json"
CONFIG = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
MODEL = CONFIG.get("model", "qwen3-vl")  # thinking version, used when deep thinking is on
FAST = CONFIG.get("fast", "qwen3-vl:8b-instruct")  # instruct version, the default
MAC = sys.platform == "darwin"
HERE = pathlib.Path(__file__).parent
# A font with accents and non-Latin letters for PDFs: Arial Unicode on a Mac, DejaVu Sans on Linux.
FONT = next((f for f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"] if os.path.exists(f)), None)
DEFAULT_WORK = (pathlib.Path(__file__).parent / "workspace").resolve()
CTX = threading.local()


def set_workspace(path, net=True, timeout=120, region="wt-wt"):
    CTX.work = pathlib.Path(path).resolve()
    CTX.work.mkdir(parents=True, exist_ok=True)
    CTX.net, CTX.timeout, CTX.region = net, timeout, region


REGIONS = {"it": "it-it", "en": "us-en", "fr": "fr-fr", "de": "de-de", "es": "es-es", "pt": "pt-pt", "pl": "pl-pl", "nl": "nl-nl", "ru": "ru-ru", "uk": "ua-uk", "zh-cn": "cn-zh", "zh-tw": "tw-tzh", "ja": "jp-jp", "ko": "kr-kr", "sv": "se-sv", "da": "dk-da", "no": "no-no", "fi": "fi-fi", "el": "gr-el", "tr": "tr-tr", "ar": "xa-ar", "cs": "cz-cs", "hu": "hu-hu", "ro": "ro-ro"}


LANGUAGES = {"it": "Italian", "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "pl": "Polish", "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "zh-cn": "Chinese", "zh-tw": "Chinese", "ja": "Japanese", "ko": "Korean", "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "el": "Greek", "tr": "Turkish", "ar": "Arabic", "cs": "Czech", "hu": "Hungarian", "ro": "Romanian"}


def lang_for(text, previous=None):
    """Language code of the user's message; short messages keep the previous one."""
    if len(text.split()) < 4:
        return previous
    try:
        return detect(text)
    except Exception:
        return previous


def region_for(text, previous="wt-wt"):
    """Search region matching the language of the user's message; short messages keep the previous one."""
    lang = lang_for(text)
    return REGIONS.get(lang, "wt-wt") if lang else previous


def _work():
    if not hasattr(CTX, "work"):
        set_workspace(DEFAULT_WORK)
    return CTX.work


def _path(name):
    p = (_work() / name).resolve()
    p.relative_to(_work())
    return p


def search_results(query, region=None, n=5):
    region = region or getattr(CTX, "region", "wt-wt")
    for backend in ("yahoo, duckduckgo, brave, startpage", "auto"):
        try:
            return DDGS().text(query, region=region, backend=backend, max_results=n)
        except Exception:
            continue
    return []


def web_search(query: str) -> str:
    """Search the web and return the top 5 results with title, link and snippet."""
    results = search_results(query)
    return str(results) if results else "No results found. Try rephrasing the query, or search again in a moment."


def _tables(raw):
    out = []
    for table in lxml.html.fromstring(raw).iter("table"):
        rows = [" | ".join(" ".join(t.strip() for t in cell.itertext() if t.strip()) for cell in row.iter("td", "th")) for row in table.iter("tr")]
        rows = [r for r in rows if r.strip(" |")]
        if len(rows) > 1:
            out.append("\n".join(rows[:80]))
    return "\n\n".join(out[:5])


def read_page(url: str) -> str:
    """Download a web page and return its main text, plus any tables on it."""
    raw = trafilatura.fetch_url(url)
    if not raw:
        return "Could not download the page."
    text = trafilatura.extract(raw, include_tables=False) or ""
    try:
        tables = _tables(raw)
    except Exception:
        tables = ""
    return ((text[:7000] or "No main text found.") + (f"\n\nTables on the page:\n{tables[:6000]}" if tables else ""))


def run_python(code: str) -> str:
    """Run Python code in a sandbox, with the workspace folder as the current directory, and return what it prints. It can only write files in the workspace folder."""
    work = _work()
    note = ""
    if MAC:  # macOS's built-in sandbox: the code may write only in the workspace (and temp files)
        net = "" if CTX.net else "(deny network*)"
        profile = f'(version 1)(allow default){net}(deny file-write*)(allow file-write* (subpath "{work}") (subpath "/private/var/folders") (literal "/dev/null"))'
        cmd = ["sandbox-exec", "-p", profile, sys.executable, "-c", code]
    elif shutil.which("bwrap"):  # Linux: bubblewrap shows the code a read-only copy of the system, with only the workspace writable
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", str(work), str(work),
               "--die-with-parent", "--unshare-pid"] + ([] if CTX.net else ["--unshare-net"]) + [sys.executable, "-c", code]
    else:
        cmd, note = [sys.executable, "-c", code], "\n(Note: bubblewrap is not installed, so this code ran without a sandbox.)"
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=CTX.timeout, env=dict(os.environ, MPLCONFIGDIR="/tmp"))
    except subprocess.TimeoutExpired:
        return f"Stopped: the code ran longer than {CTX.timeout} seconds."
    return ((r.stdout + r.stderr)[-4000:] or "(no output)") + note


def list_files(folder: str = ".") -> str:
    """List the files and subfolders in the workspace folder, or in a subfolder of it."""
    items = sorted(f"{x.name}/" if x.is_dir() else x.name for x in _path(folder).iterdir() if not x.name.startswith("."))
    return "\n".join(items[:300]) or "(empty)"


def install_package(name: str) -> str:
    """Install a Python package from PyPI so that code run with run_python can use it."""
    _work()
    if not CTX.net:
        return "Internet access is turned off, so packages can't be installed."
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", name], capture_output=True, text=True, timeout=600)
    return (r.stdout + r.stderr)[-2000:] or f"Installed {name}"


def read_file(name: str) -> str:
    """Read a file from the workspace folder: plain text, Word (.docx) or PDF (.pdf)."""
    p = _path(name)
    if p.suffix.lower() == ".docx":
        text = "\n".join(par.text for par in docx.Document(p).paragraphs)
    elif p.suffix.lower() == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
    else:
        text = p.read_text()
    return text[:20000]


def write_file(name: str, content: str) -> str:
    """Write a file in the workspace folder. A name ending in .docx creates a Word document and .pdf creates a PDF; in those, lines starting with '# ', '## ' or '### ' become headings."""
    p = _path(name)
    lines = [(len(m[1]), m[2]) if (m := re.match(r"(#{1,3}) (.*)", line)) else (0, line) for line in content.splitlines()]
    if p.suffix.lower() == ".docx":
        d = docx.Document()
        for level, text in lines:
            d.add_heading(text, level) if level else d.add_paragraph(text)
        d.save(p)
    elif p.suffix.lower() == ".pdf":
        pdf = FPDF()
        pdf.add_page()
        if FONT:
            pdf.add_font("U", fname=FONT)
        for level, text in lines:
            size = {0: 11, 1: 20, 2: 16, 3: 13}[level]
            pdf.set_font("U" if FONT else "Helvetica", size=size)
            text = text if FONT else text.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, size * 0.55, text, new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(p))
    else:
        p.write_text(content)
    return f"Saved {name}"


MEM = (pathlib.Path(__file__).parent / "memory").resolve()
MEM.mkdir(exist_ok=True)
MEM_LOCK = threading.Lock()


def _mem(topic):
    slug = re.sub(r"[^a-z0-9-]+", "-", (topic or "profile").lower()).strip("-") or "profile"
    return MEM / f"{slug}.md"


def memory_topics():
    return sorted(p.stem for p in MEM.glob("*.md"))


def remember(fact: str, topic: str = "profile") -> str:
    """Save a lasting fact about the user to long-term memory, so you know it in future chats. Use topic 'profile' for who the user is and how they like answers, or a short topic name (e.g. 'btc-model') for a specific project or subject."""
    fact = " ".join(fact.split())
    with MEM_LOCK:
        p = _mem(topic)
        old = p.read_text() if p.exists() else ""
        if fact.lower() in old.lower():
            return "Already in memory."
        p.write_text(old + f"- {fact}\n")
    return f"Saved to memory ({p.stem})."


def forget(text: str, topic: str = "profile") -> str:
    """Remove every line containing this text from a long-term memory topic."""
    with MEM_LOCK:
        p = _mem(topic)
        if not p.exists():
            return f"No memory topic '{topic}'."
        lines = p.read_text().splitlines()
        kept = [line for line in lines if text.lower() not in line.lower()]
        p.write_text("\n".join(kept) + "\n") if kept else p.unlink()
    return f"Removed {len(lines) - len(kept)} line(s)."


def read_memory(topic: str) -> str:
    """Read one long-term memory topic about the user."""
    p = _mem(topic)
    return p.read_text() if p.exists() else f"No memory topic '{topic}'. Topics: {', '.join(memory_topics()) or 'none'}."


def memory_prompt():
    profile = _mem("profile")
    text = profile.read_text().strip() if profile.exists() else ""
    others = [t for t in memory_topics() if t != "profile"]
    if not text and not others:
        return ""
    out = "\n\nWhat you remember about the user from earlier chats. These are facts about the USER, not about you: when they ask about themself, answer in the second person (in Italian with \"tu\"). Use them only when relevant, don't cite them as sources, and don't mention them unprompted:\n" + (text or "(nothing yet)")
    if others:
        out += f"\nOther memory topics you can open with read_memory: {', '.join(others)}."
    return out + "\nIf the user asks you to remember or forget something, use the remember or forget tool."


def get_datetime() -> str:
    """Return the current local date and time."""
    return datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M")


def look_at_image(source: str, question: str) -> str:
    """Look at an image, given as a URL or a file name in the workspace folder, and answer a question about it."""
    data = requests.get(source, timeout=15).content if source.startswith("http") else _path(source).read_bytes()
    return ollama.chat(getattr(CTX, "vision", FAST), messages=[{"role": "user", "content": question, "images": [data]}]).message.content


IMAGE_MODEL = "RunPod/FLUX.2-klein-4B-mflux-4bit"
MFLUX = HERE / "mflux-venv" / "bin" / "mflux-generate-flux2"  # Mac: mflux, built on Apple's MLX
IMAGE_PY = HERE / "image-venv" / "bin" / "python"  # Linux: diffusers on an NVIDIA card, via imagegen.py
IMAGES_OK = MFLUX.exists() if MAC else IMAGE_PY.exists()


def _draw(prompt, paths, w, h, out, timeout):
    """Run the image model: a new picture, or an edit of the pictures in 'paths'."""
    if not MAC:
        cmd = [str(IMAGE_PY), str(HERE / "imagegen.py"), "--prompt", prompt, "--width", str(w), "--height", str(h), "--out", str(out)] + (["--images", *paths] if paths else [])
    elif paths:
        cmd = [str(MFLUX_EDIT), "--model", IMAGE_MODEL, "--image-paths", *paths, "--prompt", prompt, "--width", str(w), "--height", str(h), "--steps", "4", "--output", str(out)]
    else:
        cmd = [str(MFLUX), "--model", IMAGE_MODEL, "--prompt", prompt, "--width", str(w), "--height", str(h), "--steps", "4", "--output", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def generate_image(prompt: str, name: str = "", width: int = 1024, height: int = 1024) -> str:
    """Create a new picture from a text description and save it as a PNG in the workspace folder. Write the prompt in English, describing subject, style, composition and lighting. Width and height are in pixels (multiples of 16, at most 1536). The picture is shown to the user automatically. Only for brand-new pictures: to change, combine or add to photos the user shared, use edit_image instead."""
    stem = re.sub(r"[^\w-]+", "-", name or prompt)[:40].strip("-") or "image"
    p = _path(f"{stem}-{datetime.datetime.now():%H%M%S}.png")
    w, h = (max(256, min(1536, int(v) // 16 * 16)) for v in (width, height))
    r = _draw(prompt, [], w, h, p, 900)
    if r.returncode or not p.exists():
        return "Error: image generation failed. " + (r.stderr or r.stdout)[-500:]
    CTX.last_image = str(p)
    return f"Saved image {p.name} and showed it to the user."


MFLUX_EDIT = HERE / "mflux-venv" / "bin" / "mflux-generate-flux2-edit"


def edit_image(instruction: str, images: str, name: str = "") -> str:
    """Edit or combine the user's own photos: add or remove people or objects, put people from one photo into another, change the background, season or style, and so on. It keeps the real people and places from the photos, unlike generate_image. It also works on pictures you generated or edited earlier in this chat, so the user can ask for changes step by step. 'images' is a comma-separated list of image file names in the workspace folder (photos the user shared, or files you saved earlier); put the main image to edit first. Write the instruction in English and refer to the photos as 'the first image', 'the second image'. Takes several minutes; the result is shown to the user automatically."""
    from PIL import Image
    names = images if isinstance(images, list) else str(images).split(",")
    files = sorted(f.name for f in _work().iterdir() if f.is_file())
    paths = []
    for n in [pathlib.Path(str(n).strip()).name for n in names if str(n).strip()][:4]:
        if n not in files:
            close = difflib.get_close_matches(n, files, 1, 0.5)
            if not close:
                return f"Error: there is no file called {n}. Files available: {', '.join(files)}"
            n = close[0]
        paths.append(str(_path(n)))
    if not paths:
        return "Error: give the file names of the photos to edit."
    with Image.open(paths[0]) as im:
        w, h = im.size
    scale = min(1.0, 1216 / max(w, h))
    w, h = (max(256, int(v * scale) // 16 * 16) for v in (w, h))
    stem = re.sub(r"[^\w-]+", "-", pathlib.Path(name).stem if name else "edited")[:40].strip("-") or "edited"
    out = _path(f"{stem}-{datetime.datetime.now():%H%M%S}.png")
    r = _draw(instruction, paths, w, h, out, 1800)
    if r.returncode or not out.exists():
        return "Error: image editing failed. " + (r.stderr or r.stdout)[-500:]
    CTX.last_image = str(out)
    return f"Saved edited image {out.name} and showed it to the user."

tools = {f.__name__: f for f in [web_search, read_page, run_python, install_package, list_files, read_file, write_file, get_datetime, look_at_image, generate_image, edit_image, remember, forget, read_memory]}
if not IMAGES_OK:  # image generation was not installed
    tools.pop("generate_image"), tools.pop("edit_image")
MEMORY_TOOLS = {"remember", "forget", "read_memory"}
messages = []
while __name__ == "__main__":
    messages.append({"role": "user", "content": input("\n> ")})
    while True:
        m = ollama.chat(FAST, messages=messages, tools=list(tools.values()), options={"num_ctx": 16384}).message
        messages.append({"role": "assistant", "content": m.content, "tool_calls": m.tool_calls})
        if not m.tool_calls:
            print(m.content)
            break
        for c in m.tool_calls:
            print(f"  [{c.function.name}] {dict(c.function.arguments)}")
            try:
                out = str(tools[c.function.name](**c.function.arguments))
            except Exception as e:
                out = f"Error: {e}"
            messages.append({"role": "tool", "tool_name": c.function.name, "content": out})
