import copy, html, inspect, json, os, pathlib, platform, re, secrets, shutil, subprocess, sys, threading, time, zipfile, urllib.parse
from concurrent.futures import ThreadPoolExecutor, wait
from itertools import zip_longest
import gradio as gr, ollama
import agent, apple, providers, tune_ui
from i18n import t, LANG, LANGS
import i18n
tr = t  # the same, for functions where t is a scheduled task
from agent import FAST, MODEL, tools

# Every part of Joshua calls ollama.chat. Calls for Apple's model go to apple.chat instead,
# which answers in the same shape, so nothing else needs to know which model it is talking to.
_ollama_chat = ollama.chat
ollama.chat = lambda model, *args, **kw: (apple.chat if model == apple.NAME else providers.chat if providers.is_api(model) else _ollama_chat)(model, *args, **kw)

ROOT = (pathlib.Path(__file__).parent / "conversations").resolve()
ROOT.mkdir(exist_ok=True)
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
ICONS = {"web_search": "🔎", "read_page": "📄", "run_python": "🐍", "read_file": "📂", "write_file": "💾", "get_datetime": "🕒", "look_at_image": "🖼️", "generate_image": "🎨", "edit_image": "🖌️", "goal_done": "✅", "give_up": "🛑", "install_package": "📦", "list_files": "🗂️", "remember": "🧠", "forget": "🧹", "read_memory": "📒"}
SKIP = (gr.skip(),) * 4

IMAGE_RULE = " When the user asks to change, combine or add to photos they shared, use edit_image with those photo files and never answer with an invented picture; use generate_image only for brand-new pictures." if "edit_image" in tools else " You cannot create or edit pictures on this computer: say so if asked."
SYSTEM = "You are Joshua, an AI assistant running on the user's computer; you have no job or personal life of your own. Today is {now}." + IMAGE_RULE + " Results returned by your tools are real and current. For any factual question (figures, people, events, prices, anything that may change), always use web_search before answering; never answer such questions from memory. Search snippets are often incomplete: if they don't clearly contain the answer, open the most relevant result with read_page before answering, and never fill gaps with guesses. Always address the user directly, in the second person, in the user's language. Files the user shares are in your workspace folder."
GOAL_RULES = " You are in goal mode: the user gives you a goal to accomplish, not a question. Work step by step with your tools until the goal is fully achieved, and check your own results (for example, read back files you wrote). Do not ask the user questions; make reasonable assumptions. When the goal is achieved and verified, call goal_done with a short summary. Only if it is truly impossible with your tools, call give_up with the reason."
NUDGE = "The goal is not marked as done yet. Review what you have done so far, fix anything missing or wrong, and continue working. Call goal_done only when the goal is fully achieved and verified."


def goal_done(summary: str) -> str:
    """Call this only when the goal is fully achieved and you have verified the result. Give a short summary of what was done."""
    return "Goal marked as done."


def give_up(reason: str) -> str:
    """Call this only if the goal is truly impossible to achieve with your tools. Explain why."""
    return "Stopped."


# ---------- available models ----------

CAPS = {}


def caps(name):
    if providers.is_api(name):  # all use tools; all but DeepSeek see images
        return {"tools", "vision"} if providers.sees_images(name) else {"tools"}
    if name not in CAPS:
        try:
            CAPS[name] = set(ollama.show(name).capabilities or [])
        except Exception:
            CAPS[name] = set()
    return CAPS[name]


CLOUD = set()  # models that run on Ollama's servers, not on this computer


def is_cloud(name):
    return providers.is_api(name) or name in CLOUD or name.split(":")[-1].endswith("cloud")


PAIR = "auto:qwen3-vl"  # one menu entry: the deep-thinking switch picks qwen3-vl or its faster instruct version


ADD = "+add"  # last entry of the model menu: opens the panels where you add cloud or local models


def models(add=False):
    try:
        listed = ollama.list().models
    except Exception:
        listed = []
    CLOUD.update(m.model for m in listed if getattr(m, "remote_host", None) or m.model.split(":")[-1].endswith("cloud"))
    names = [n for n in (m.model for m in listed) if "tools" in caps(n)]
    label = lambda n: "☁ " + n.removesuffix(":cloud").removesuffix("-cloud") if is_cloud(n) else "★ " + n.removeprefix("joshua-").removesuffix(":latest") if n.startswith("joshua-") else n.removesuffix(":latest")
    choices = [(label(n), n) for n in names if not (FAST in names and n in (FAST, MODEL, MODEL + ":latest"))]
    if FAST in names and (MODEL in names or MODEL + ":latest" in names):
        choices.insert(0, (agent.CONFIG.get("label", "qwen3-vl"), PAIR))
    if not apple.why_not():  # macOS 26 with Apple Intelligence on: free, and already on this Mac
        choices.append((apple.LABEL, apple.NAME))
    # the order: on this computer, then open-weight models in the cloud, then closed ones (Claude, ChatGPT, Gemini)
    local = [c for c in choices if not is_cloud(c[1])]
    cloud_open = [c for c in choices if is_cloud(c[1])] + providers.menu(open_weights=True)
    choices = local + cloud_open + providers.menu(open_weights=False)
    return (choices or [(FAST, FAST)]) + ([(t("＋ Add a model…"), ADD)] if add else [])


def default_model():
    local = [v for _, v in models() if not is_cloud(v)]
    return (local or [models()[0][1]])[0]  # start on a local model; cloud is always a choice you make


CLOUD_BADGE = "<div class='cloudbadge'>" + t("☁ Cloud model: messages go to Ollama's servers; everything stays stored here") + "</div>"


def badge(choice):
    if choice and providers.is_api(choice):
        return "<div class='cloudbadge'>" + t("🔑 {company}'s model: messages go to {company}'s servers; everything stays stored here", company=providers.company(choice)) + "</div>"
    return CLOUD_BADGE if choice and choice != ADD and is_cloud(resolve(choice, False)) else ""


CLOUD_HELP = ("**" + t("Stays here:") + "** " + t("your chats, files and memory notes.") + "  \n"
              + "**" + t("Goes to Ollama:") + "** " + t("the conversation, with the files and photos in it.") + "  \n"
              + "**" + t("Cost:") + "** " + t("pay per use, paid to Ollama, not to Joshua. Local models are free.") + "\n\n"
              + "<details><summary>" + t("Read more") + "</summary>\n\n" + t("""Bigger models running on Ollama's servers, for when this computer is too small or slow.
**What goes to the cloud:** for every message, the conversation so far, including the text of files and web pages Joshua reads, and your photos if you send them.
**What stays here:** your chats, files, memory notes and trained models are stored only on this computer. Memory notes are sent only if you allow it in 🧠 Memory.
Ollama says it never logs or trains on cloud prompts. That is a promise you are trusting, not something this computer can check.
**Cost:** new Ollama accounts pay per use: add a few dollars of credit on ollama.com (Gemma 4 31B costs well under a cent per message). Some older accounts still have a small free monthly allowance on a few models (marked "free plan" for those accounts). Local models are always free.""") + "\n\n</details>")


def ollama_bin():
    return shutil.which("ollama") or "/Applications/Ollama.app/Contents/Resources/ollama"


def cloud_catalog():
    """The cloud models Ollama offers today that can use tools, read from ollama.com (only model names, nothing of yours is sent)."""
    import requests
    try:
        names = [m["name"] for m in requests.get("https://ollama.com/api/tags", timeout=15).json()["models"]]
        info = {n: requests.post("https://ollama.com/api/show", json={"model": n}, timeout=15).json().get("capabilities", []) for n in names}
    except Exception:
        return gr.Dropdown(choices=[]), "<div class='cloudstat'>" + t("Joshua couldn't reach ollama.com. Check your internet connection, then close and reopen this panel to try again.") + "</div>"
    free = {"gemma4:31b"}  # free for older accounts on Ollama's legacy free plan, and among the cheapest per message (November 2026)
    choices = [(("👁 " if "vision" in c else "") + n + (" · " + t("cheapest") if n in free else ""), n) for n, c in info.items() if "tools" in c]
    choices.sort(key=lambda c: c[1] not in free)
    return gr.Dropdown(choices=choices, value=None), ""


def cloud_connect():
    """'ollama signin' links this computer to an Ollama account by registering its key; it prints a page to open."""
    import tempfile
    try:
        with tempfile.TemporaryFile("w+") as out:
            p = subprocess.Popen([ollama_bin(), "signin"], stdout=out, stderr=out, text=True)
            try:
                p.wait(8)  # it waits for the browser approval; we only need the link it prints
            except subprocess.TimeoutExpired:
                p.kill()
            out.seek(0)
            out = out.read()
    except Exception as e:
        return "<div class='cloudstat'>" + t("Couldn't run ollama signin: {error}", error=html.escape(str(e))) + "</div>"
    url = re.search(r"https://ollama\.com/connect\S+", out)
    if url:
        return "<div class='cloudstat'>" + t("Open <a href='{url}' target='_blank'>this page</a>, sign in to Ollama and approve this computer. Then come back and choose a cloud model.", url=html.escape(url[0])) + "</div>"
    return f"<div class='cloudstat'>{html.escape(out.strip() or t('This computer is already connected to Ollama.'))}</div>"


# ---------- local models: what ollama.com offers that fits this computer ----------

def memory_budget():
    """GB a local model may use: the graphics card's memory on a Linux PC with an NVIDIA card, otherwise the computer's."""
    if not agent.MAC:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            if out.strip():
                return int(out.split()[0]) / 1024
        except Exception:
            pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def _model_tags(name):
    """Downloadable versions of one model, read from its page on ollama.com: (tag, size in GB, sees images)."""
    import requests
    page = requests.get(f"https://ollama.com/library/{name}/tags", timeout=20).text
    found = {}
    for m in re.finditer(r'href="/library/(' + re.escape(name) + r':[\w.]+)"(.{0,1500}?)(?:href="/library/|$)', page, re.S):
        tag, info = m[1], m[2]
        size = re.search(r"([\d.]+)\s*GB", info)
        size_tag = re.fullmatch(r"(e?\d+(\.\d+)?[bm](-a\d+(\.\d+)?b)?|latest)", tag.split(":")[1])  # 8b, 0.6b, e4b, 30b-a3b: one entry per size
        if size and size_tag and "cloud" not in tag and tag not in found:
            found[tag] = (float(size[1]), "Image" in info)
    return [(t, gb, vision) for t, (gb, vision) in found.items()]


def _tool_models(order, limit):
    """Model names from ollama.com/library (sorted by popularity or newest) whose label says they can use tools."""
    import requests
    page = requests.get(f"https://ollama.com/library?sort={order}", timeout=20).text
    cards = re.split(r'(?=<a href="/library/[\w.-]+")', page)[1:]
    names = [re.match(r'<a href="/library/([\w.-]+)"', c)[1] for c in cards if re.search(r">\s*tools\s*</span>", c)]
    return names[:limit]


def local_catalog():
    """Models on ollama.com that can use tools, in the versions that fit this computer's memory, biggest first."""
    try:
        names = list(dict.fromkeys(_tool_models("popular", 30) + _tool_models("newest", 12)))
        with ThreadPoolExecutor(8) as pool:
            tags = [t for ts in pool.map(lambda n: _try(_model_tags, n), names) for t in ts]
    except Exception:
        return gr.Dropdown(choices=[]), "<div class='cloudstat'>" + t("Joshua couldn't reach ollama.com. Check your internet connection, then close and reopen this panel to try again.") + "</div>"
    sized = {(t.split(":")[0], gb) for t, gb, _ in tags if not t.endswith(":latest")}
    tags = [x for x in tags if not (x[0].endswith(":latest") and (x[0].split(":")[0], x[1]) in sized)]  # 'latest' is usually a copy of another size
    budget, free = memory_budget(), shutil.disk_usage(pathlib.Path.home()).free / 2**30
    have = {m.model for m in _try(lambda: ollama.list().models)}
    fits = sorted({(gb, t, v) for t, gb, v in tags if gb * 1.2 + 2 <= budget}, reverse=True)
    choices = [(("✓ " if t in have else "") + ("👁 " if v else "") + f"{t} · {gb:g} GB", t) for gb, t, v in fits]
    note = t("{n} versions fit this computer ({memory} GB of memory for models, {disk} GB free disk). Biggest first; 👁 = also sees images, ✓ = already here.", n=len(choices), memory=f"{budget:.0f}", disk=f"{free:.0f}")
    return gr.Dropdown(choices=choices, value=None), f"<div class='cloudstat'>{note}</div>"


def _try(f, *a):
    try:
        return f(*a)
    except Exception:
        return []


def local_download(tag):
    """Downloads a model with Ollama, showing progress as it goes."""
    if not tag:
        yield gr.skip(), "<div class='cloudstat'>" + t("Pick a model first.") + "</div>"
        return
    done = 0
    try:
        for p in ollama.pull(tag, stream=True):
            if p.total:
                done = int(100 * (p.completed or 0) / p.total)
            yield gr.skip(), "<div class='cloudstat'>" + t("Downloading {model}: {percent}%", model=html.escape(tag), percent=done) + f" · {html.escape(p.status or '')}</div>"
    except Exception as e:
        yield gr.skip(), "<div class='cloudstat'>" + t("The download didn't finish ({error}). Check your connection and try again.", error=html.escape(str(e))) + "</div>"
        return
    CAPS.pop(tag, None)
    ok = "tools" in caps(tag)
    yield gr.Dropdown(choices=models(True), value=tag if ok else gr.skip()), "<div class='cloudstat'>" + (t("{model} is ready and selected in the model menu.", model=html.escape(tag)) if ok else t("{model} is ready, but it can't use Joshua's tools, so it isn't in the menu.", model=html.escape(tag))) + "</div>"


def local_update():
    """Asks Ollama for the latest version of every local model: unchanged ones are skipped in seconds."""
    names = [m.model for m in ollama.list().models if not is_cloud(m.model)]
    for i, n in enumerate(names, 1):
        yield "<div class='cloudstat'>" + t("Checking {i} of {n}: {model}…", i=i, n=len(names), model=html.escape(n)) + "</div>"
        try:
            for _ in ollama.pull(n, stream=True):
                pass
        except Exception as e:
            yield f"<div class='cloudstat'>{html.escape(n)}: {html.escape(str(e))}</div>"
    CAPS.clear()
    yield "<div class='cloudstat'>" + t("All {n} local models are up to date.", n=len(names)) + "</div>"


# ---------- updates of Joshua itself ----------

VERSION = (pathlib.Path(__file__).parent / "VERSION").read_text().strip() if (pathlib.Path(__file__).parent / "VERSION").exists() else "0"
SITE = "https://ask.joshua.earth"


def check_update():
    """Asks ask.joshua.earth for the latest version number (nothing about you is sent)."""
    import requests
    try:
        latest = requests.get(f"{SITE}/version.txt", timeout=6).text.strip()
    except Exception:
        return gr.skip(), gr.skip()
    num = lambda v: tuple(int(x) for x in v.split(".") if x.isdigit())  # versions are dates plus a counter: 2026.11.08.3
    if re.fullmatch(r"[\d.]+", latest) and num(latest) > num(VERSION):
        return "<div class='updnote'>" + t("Joshua {new} is available (you have {old}).", new=html.escape(latest), old=html.escape(VERSION)) + "</div>", gr.Button(visible=True)
    return "", gr.Button(visible=False)


def do_update():
    """Downloads the new version over this one (your chats, files, memory and settings are left alone), then restarts."""
    import io, requests, tarfile
    here = pathlib.Path(__file__).parent
    try:
        data = requests.get(f"{SITE}/joshua.tar.gz", timeout=60).content
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            members = [m for m in t.getmembers() if m.isfile() and m.name.startswith("joshua/") and ".." not in m.name]
            for m in members:
                m.name = m.name.removeprefix("joshua/")
                (here / m.name).unlink(missing_ok=True)  # replace the file rather than write into it (it may be read-only)
                t.extract(m, here, filter="data")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(here / "requirements.txt")], timeout=600)
    except Exception as e:
        return "<div class='updnote'>" + tr("Update failed: {error}", error=html.escape(str(e))) + "</div>"
    if restart():
        return "<div class='updnote'>" + tr("Updated. Joshua is restarting: reload this page in a few seconds.") + "</div>"
    return "<div class='updnote'>" + tr("Updated. Quit and start Joshua again to use the new version.") + "</div>"


def restart():
    """Quits and starts Joshua again; False if it can't start itself (then you restart it by hand)."""
    if os.environ.get("JOSHUA_SERVICE"):  # running from login: just quit, and the system starts it again
        threading.Timer(1, lambda: os._exit(0)).start()
        return True
    start = pathlib.Path.home() / "AskJoshua" / "start.sh"
    if start.exists():  # start again in a moment, from a separate process, then quit this one
        subprocess.Popen(["/bin/sh", "-c", f"sleep 3; '{start}'"], env={**os.environ, "JOSHUA_NO_OPEN": "1"}, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        threading.Timer(1, lambda: os._exit(0)).start()
        return True
    return False


def set_language(code):
    """The whole interface is built in one language when Joshua starts, so a new one needs a restart."""
    if code == (i18n.chosen() if i18n.chosen() in LANGS else "auto"):
        return False
    i18n.choose(code)
    if restart():
        gr.Info(t("Restarting Joshua in the new language…"))
        return True
    gr.Info(t("Saved. Quit and start Joshua again to see the new language."))
    return False


def cloud_add_model(name):
    if not name:
        return gr.skip(), "<div class='cloudstat'>" + t("Pick a model first.") + "</div>"
    tag = name + ("-cloud" if ":" in name else ":cloud")  # Ollama's naming: gpt-oss:120b → gpt-oss:120b-cloud, kimi-k3 → kimi-k3:cloud
    r = subprocess.run([ollama_bin(), "pull", tag], capture_output=True, text=True, timeout=120)
    if r.returncode:
        return gr.skip(), "<div class='cloudstat'>" + t("Couldn't add {model}: {error}", model=html.escape(tag), error=html.escape((r.stderr or r.stdout)[-200:])) + "</div>"
    CAPS.pop(tag, None)
    return gr.Dropdown(choices=models(True), value=tag), "<div class='cloudstat'>" + t("Added ☁ {model}: it's selected in the model menu. Nothing was downloaded: it runs on Ollama's servers.", model=html.escape(name)) + "</div>"


KEYS_HELP = t("""Use Claude, ChatGPT, Gemini or DeepSeek inside Joshua, with all of Joshua's tools. Your chats, files and memory stay on this computer; the conversation goes to the company's servers, like any cloud model.

**You need an API key**, a password for programs, from the company's developer site. It is paid per use and separately from any chat subscription. The key is kept in providers.json on this computer, readable only by you.""")


def keys_status(provider):
    """What we know about this company's key, and its models if the key works."""
    info = providers.PROVIDERS[provider]
    link = f"<a href='{info['keys']}' target='_blank' rel='noopener'>" + t("Get a key from {company}", company=info["company"]) + "</a>"
    if not providers.settings()["keys"].get(provider):
        return gr.Dropdown(choices=[], value=None), "<div class='cloudstat'>" + t("No key yet.") + " " + link + "</div>"
    try:
        names = providers.catalog(provider)
    except Exception as e:
        return gr.Dropdown(choices=[], value=None), "<div class='cloudstat'>" + html.escape(str(e)) + " " + link + "</div>"
    return gr.Dropdown(choices=names, value=names[0] if names else None), "<div class='cloudstat'>" + t("Key works: {n} models available.", n=len(names)) + "</div>"


def keys_save(provider, key):
    if not key or not key.strip():
        return gr.skip(), "<div class='cloudstat'>" + t("Paste the key first.") + "</div>", gr.skip()
    providers.set_key(provider, key)
    pick, status = keys_status(provider)
    return pick, status, ""  # empty the box: the key is not shown again


def keys_add(provider, name):
    if not name:
        return gr.skip(), "<div class='cloudstat'>" + t("Pick a model first.") + "</div>"
    value = providers.add_model(provider, name)
    return gr.Dropdown(choices=models(True), value=value), "<div class='cloudstat'>" + t("Added {model}: it's selected in the model menu.", model=html.escape(providers.label(value))) + "</div>"


def share_memory_setting(value=None):
    """Whether cloud models may see the memory notes. Off unless you switch it on; stored in memory/settings.json."""
    f = agent.MEM / "settings.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    if value is not None:
        data["share_with_cloud"] = bool(value)
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(data))
    return data.get("share_with_cloud", False)


def resolve(choice, thinking):
    if choice == PAIR:
        return MODEL if thinking else FAST
    return choice


# ---------- conversation storage ----------

def fdir(cid):
    return ROOT / cid / "files"


def wdir(cid):
    project = load(cid).get("project")
    return pathlib.Path(project) if project else fdir(cid)


def load(cid):
    p = ROOT / cid / "chat.json"
    return json.loads(p.read_text()) if p.exists() else {"title": "", "updated": 0, "display": [], "llm": []}


def save(cid, conv):
    conv["updated"] = time.time()
    (ROOT / cid / "chat.json").write_text(json.dumps(conv, ensure_ascii=False, indent=1))


def new_conv():
    cid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    fdir(cid).mkdir(parents=True)
    save(cid, load(cid))
    return cid


def conv_choices():
    convs = [(load(p.name), p.name) for p in ROOT.iterdir() if (p / "chat.json").exists()]
    return [(("⏳ " if busy(cid) else "") + (c["title"] or t("New chat")), cid) for c, cid in sorted(convs, key=lambda x: -x[0]["updated"])]


def file_list(cid):
    if not cid or not wdir(cid).exists():
        return []
    return sorted(str(p) for p in wdir(cid).iterdir() if p.is_file() and not p.name.startswith("."))[:200]


# ---------- the agent loop (runs in a background thread per chat) ----------

class Job:
    def __init__(self, view):
        self.view, self.version, self.running, self.stop = list(view), 0, True, threading.Event()
        self.started = time.time()

    def publish(self, view):
        self.view = list(view)
        self.version += 1


JOBS = {}


def agent_turn(msgs, shown, toolset, model, thinking, stop, called):
    for _ in range(25):
        content, thought, calls, live = "", "", [], []
        mode = {"model": model, "think": thinking} if "thinking" in caps(model) else {"model": model}
        for chunk in ollama.chat(**mode, messages=msgs, tools=list(toolset.values()), stream=True, options={"num_ctx": 16384}):
            thought += chunk.message.thinking or ""
            content += chunk.message.content or ""
            calls += chunk.message.tool_calls or []
            live = [{"role": "assistant", "content": thought, "metadata": {"title": "💭 " + t("Thinking"), "status": "done" if content or calls else "pending"}}] if thought else []
            live += [{"role": "assistant", "content": content}] if content else []
            yield shown + live
            if stop.is_set():
                break
        shown += live
        msgs.append({"role": "assistant", "content": content, "tool_calls": [{"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}} for c in calls]})
        if thought and providers.is_api(model):  # DeepSeek wants its reasoning back with the tool results
            msgs[-1]["thinking"] = thought
        if not calls or stop.is_set():
            return
        for c in calls:
            name, args = c.function.name, dict(c.function.arguments)
            box = {"role": "assistant", "content": f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```", "metadata": {"title": f"{ICONS.get(name, '🔧')} {name}", "status": "pending"}}
            yield shown + [box]
            try:
                result = str(toolset[name](**args))
            except Exception as e:
                result = f"Error: {e}"
            box["content"] += f"\n**Result**\n```\n{result[:1500]}{'…' if len(result) > 1500 else ''}\n```"
            box["metadata"]["status"] = "done"
            shown.append(box)
            if name in ("generate_image", "edit_image") and getattr(agent.CTX, "last_image", None):
                shown.append({"role": "assistant", "content": {"path": agent.CTX.last_image}})
                agent.CTX.last_image = None
            called.append(name)
            yield shown
            msgs.append({"role": "tool", "tool_name": name, "content": result})
        if {"goal_done", "give_up"} & set(called) or stop.is_set():
            return


# ---------- research first: search, read and check before answering ----------

PLAN = """Today is {today}. Decide whether a good answer to the user's last message needs facts from the web: names, numbers, rankings, dates, news, prices, people, places, laws, or anything that may have changed. Greetings, opinions, creative writing, translation, coding, maths, and questions only about the user or about files in the chat don't need it.

Recent conversation:
{history}

User's last message: {text}

Reply in JSON. If a search is needed, give 1 to 3 short, specific web search queries in {language} that would find the exact answer (include names and places; only add a year when the question is about a specific time)."""
PLAN_SCHEMA = {"type": "object", "properties": {"search": {"type": "boolean"}, "queries": {"type": "array", "items": {"type": "string"}}}, "required": ["search", "queries"]}
USE_SOURCES = "\n\n---\nSources found on the web just now, for this question:\n\n{evidence}\n---\nAnswer my question using these sources and cite them like [1], only for facts that come from them. If they don't contain the answer, say so honestly instead of guessing. Write to me directly, in the second person, entirely in {language}."
CHECK = """Sources:

{evidence}

Answer to check:

{answer}

List every specific factual claim in the answer (names, numbers, dates, rankings, places) that the sources above do NOT clearly support, or that they contradict. Facts about the user themself don't count. Reply in JSON; use an empty list if every claim is supported."""
CHECK_SCHEMA = {"type": "object", "properties": {"unsupported": {"type": "array", "items": {"type": "string"}}}, "required": ["unsupported"]}
CONFIRM = """Source excerpts:

{lines}

Claim: {claim}

Do the source excerpts above clearly support this claim? Look carefully at names, numbers and rankings, including table rows. Reply in JSON."""
CONFIRM_SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"}}, "required": ["supported"]}
FIX = "A fact check against the sources found these problems in your answer:\n- {problems}\nRewrite your answer for me: correct these points using only what the sources say, cite them like [1], and if the sources don't say it, tell me you couldn't verify it. Don't mention the fact check. Write in {language}."


def fast_of(model):
    return FAST if model == MODEL else model


def ask_json(model, prompt, schema):
    think = {"think": False} if "thinking" in caps(model) else {}
    r = ollama.chat(model, messages=[{"role": "user", "content": prompt}], format=schema, options={"temperature": 0, "num_ctx": 16384}, **think)
    return json.loads(r.message.content)


def relevant(text, words, limit=2500):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if sum(map(len, lines)) <= limit:
        return "\n".join(lines)
    ranked = sorted(range(len(lines)), key=lambda i: (-sum(w in lines[i].lower() for w in words), i))
    keep, size = set(), 0
    for i in ranked:
        if size >= limit:
            break
        keep.add(i)
        size += len(lines[i])
    return "\n".join(lines[i] for i in sorted(keep))


def research(text, msgs, model, language, region, box, publish):
    """Returns numbered sources for the question, or None when no web facts are needed."""
    history = "\n".join(f"{m['role'].upper()}: {str(m.get('content') or '')[:300]}" for m in msgs[-7:-1] if m["role"] in ("user", "assistant") and m.get("content"))
    try:
        plan = ask_json(fast_of(model), PLAN.format(today=agent.get_datetime(), history=history or "(none)", text=text, language=language or "the user's language"), PLAN_SCHEMA)
    except Exception:
        return None
    if not plan.get("search"):
        return None
    queries = [q.strip() for q in plan.get("queries", []) if str(q).strip()][:3] or [text]
    box["content"] = t("Searching:") + " " + " · ".join(f"*{q}*" for q in queries)
    publish()
    with ThreadPoolExecutor(3) as ex:
        lists = list(ex.map(lambda q: agent.search_results(q, region), queries))
    results, seen = [], set()
    for group in zip_longest(*lists):
        for r in group:
            if r and r.get("href") and r["href"] not in seen:
                seen.add(r["href"])
                results.append(r)
    if not results:
        box["content"] += "\n\n" + t("No results found.")
        return None
    box["content"] += "\n\n" + t("Reading {n} pages…", n=min(3, len(results)))
    publish()
    pages = {}
    with ThreadPoolExecutor(3) as ex:
        futures = {ex.submit(agent.read_page, r["href"]): i for i, r in enumerate(results[:3])}
        done, _ = wait(futures, timeout=25)
        for f in done:
            try:
                pages[futures[f]] = f.result()
            except Exception:
                pass
    words = {w.lower() for w in re.findall(r"\w{4,}", " ".join(queries) + " " + text)}
    sources = []
    for i, r in enumerate(results[:6]):
        page = pages.get(i, "")
        main, _, tables = page.partition("Tables on the page:")
        body = (relevant(main, words, 1500) + ("\nTable:\n" + tables.strip()[:2000] if tables.strip() else "")) if len(page) > 200 else r.get("body", "")
        sources.append(f"[{i + 1}] {r.get('title', '')}\n{r['href']}\n{body}")
    box["content"] = t("Searched:") + " " + " · ".join(f"*{q}*" for q in queries) + "\n\n" + "\n".join(f"{i + 1}. [{r.get('title', r['href'])}]({r['href']})" + (" — " + t("read") if i in pages else "") for i, r in enumerate(results[:6]))
    return "\n\n".join(sources)[:12000]


def key_terms(claim):
    numbers = {re.sub(r"[.,\s]", "", n) for n in re.findall(r"\d[\d.,\s]*\d|\d", claim)}
    names = {w.lower() for w in re.findall(r"(?<![.!?]\s)(?<!^)\b[A-ZÀ-Ý][\wÀ-ÿ'’-]{2,}", claim)}
    return {n for n in numbers if len(n) > 1}, names


def invented(claim, evidence):
    """True when the claim contains a number or name that appears nowhere in the sources."""
    flat = re.sub(r"[.,\s]", "", evidence)
    numbers, names = key_terms(claim)
    return any(n not in flat for n in numbers) or any(n not in evidence.lower() for n in names)


def claim_lines(evidence, claim, limit=1800):
    numbers, names = key_terms(claim)
    lines = [line for line in evidence.splitlines() if line.strip()]
    strong = [line for line in lines if any(n in re.sub(r"[.,\s]", "", line) for n in numbers) or any(n in line.lower() for n in names)]
    out = "\n".join(strong)[:limit]
    return out if out else relevant(evidence, {w.lower() for w in re.findall(r"\w{4,}", claim)}, limit)


def last_answer(shown):
    for m in reversed(shown):
        if m["role"] == "assistant" and not m.get("metadata") and isinstance(m.get("content"), str):
            return m
    return None


def verify(evidence, msgs, shown, toolset, model, thinking, language, job):
    target = last_answer(shown)
    if not target or not target["content"].strip():
        return
    box = {"role": "assistant", "content": t("Checking the answer against the sources…"), "metadata": {"title": "🔍 " + t("Fact check"), "status": "pending"}}
    shown.append(box)
    job.publish(shown)
    try:
        flagged = [str(p) for p in ask_json(fast_of(model), CHECK.format(evidence=evidence, answer=target["content"]), CHECK_SCHEMA).get("unsupported", []) if str(p).strip()][:4]
    except Exception:
        flagged = []
    problems = []
    for claim in flagged:
        if not invented(claim, evidence):
            continue
        try:
            if not ask_json(fast_of(model), CONFIRM.format(lines=claim_lines(evidence, claim), claim=claim), CONFIRM_SCHEMA).get("supported"):
                problems.append(claim)
        except Exception:
            pass
    box["metadata"]["status"] = "done"
    if not problems:
        box["content"], box["metadata"]["title"] = t("Every claim in the answer is supported by the sources."), "✅ " + t("Checked against sources")
        return
    box["content"], box["metadata"]["title"] = t("Not supported by the sources:") + "\n" + "\n".join(f"- {p}" for p in problems), "⚠️ " + t("Fact check: corrected below")
    target["metadata"] = {"title": "📝 " + t("First draft (corrected below)"), "status": "done"}
    msgs.append({"role": "user", "content": FIX.format(problems="\n- ".join(problems), language=language or "the user's language")})
    for view in agent_turn(msgs, shown, toolset, model, thinking, job.stop, []):
        job.publish(view)


# ---------- swarm mode ----------

STYLES = ["Work methodically and double-check each step.", "Look for an angle the others might miss.", "Be skeptical: verify facts with your tools before stating them.", "Focus on the most practical, concrete answer.", "Think about what could go wrong or be misunderstood.", "Keep it simple and get to the core of the question."]
TEMPS = [0.6, 0.9, 0.7, 1.0, 0.8, 1.1]
SYNTH = "Your team of {n} agents worked on my last request independently. Their answers:\n\n{report}\n\nNow write the single best final answer for me. Agreement is not proof: if a fact isn't backed by a source the agents actually read, check it yourself with web_search and read_page. Combine their strongest points, resolve disagreements, and drop anything wrong. Write directly to me, in the second person, in the language of my request, as if you had answered alone: no notes about the agents or your process."


def swarm_agent(k, n, model, thinking, base, box, answers, stop, work, net, timeout, toolset, region):
    agent.set_workspace(work, net, int(timeout), region)
    agent.CTX.vision = model if "vision" in caps(model) else FAST
    msgs = base
    msgs[0] = {"role": "system", "content": msgs[0]["content"] + f" You are agent {k + 1} of {n} in a team answering the same request independently; a reviewer will compare your answers. {STYLES[k % len(STYLES)]} Back every fact with a tool result and say where it comes from. Only create files if the user asked for them. End with your complete answer, addressed to the user."}
    opts = {"num_ctx": 16384, "temperature": TEMPS[k % len(TEMPS)]}
    think = {"think": thinking} if "thinking" in caps(model) else {}
    log = []
    try:
        for _ in range(15):
            if stop.is_set():
                break
            m = ollama.chat(model, messages=msgs, tools=list(toolset.values()), options=opts, **think).message
            calls = m.tool_calls or []
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [{"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}} for c in calls]})
            if not calls:
                answers[k] = m.content or ""
                break
            for c in calls:
                name, args = c.function.name, dict(c.function.arguments)
                log.append(f"{ICONS.get(name, '🔧')} {name} `{json.dumps(args, ensure_ascii=False)[:100]}`")
                box["content"] = t("Working…") + "\n\n" + "\n".join(f"- {line}" for line in log)
                try:
                    result = str(toolset[name](**args))
                except Exception as e:
                    result = f"Error: {e}"
                msgs.append({"role": "tool", "tool_name": name, "content": result})
    except Exception as e:
        answers[k] = None
        log.append(f"⚠️ {e}")
    steps = "\n".join(f"- {line}" for line in log)
    box["content"] = (steps + "\n\n---\n\n" if steps else "") + (answers[k] or t("(no answer)"))
    box["metadata"]["status"] = "done"


def run_swarm(job, cid, conv, model, thinking, n, pool, net, timeout, toolset):
    shown, msgs = conv["display"], conv["llm"]
    work = conv.get("project") or fdir(cid)
    answers, threads = [None] * n, []
    for k in range(n):
        m = pool[k % len(pool)]
        box = {"role": "assistant", "content": tr("Working…"), "metadata": {"title": "🐝 " + tr("Agent {n}", n=k + 1) + f" · {m.removesuffix(':latest')}", "status": "pending"}}
        shown.append(box)
        t = threading.Thread(target=swarm_agent, args=(k, n, m, thinking, copy.deepcopy(msgs), box, answers, job.stop, work, net, timeout, toolset, conv.get("region", "wt-wt")), daemon=True)
        t.start()
        threads.append(t)
    while any(t.is_alive() for t in threads):
        job.publish(shown)
        time.sleep(0.5)
    job.publish(shown)
    if job.stop.is_set():
        return
    report = "\n\n".join(f"### Agent {k + 1} ({pool[k % len(pool)]})\n{a or '(no answer)'}" for k, a in enumerate(answers))
    judge = msgs + [{"role": "user", "content": SYNTH.format(n=n, report=report)}]
    for view in agent_turn(judge, shown, toolset, model, thinking, job.stop, []):
        job.publish(view)
    final = next((m["content"] for m in reversed(judge) if m["role"] == "assistant" and m["content"]), "")
    msgs.append({"role": "assistant", "content": final})


# ---------- long-term memory ----------

EXTRACT = """You keep long-term memory notes about the user of an AI assistant. The current notes:

{notes}

The latest exchange:
USER: {user}
ASSISTANT: {answer}

Does the USER message reveal lasting facts worth knowing in future chats: who they are, their work, ongoing projects, stable preferences about how they want answers? Rules: save only what the user explicitly stated, never guesses or things the assistant said; skip anything temporary, trivial, or already in the notes; never save passwords, ID or card numbers, or health details. Most exchanges contain nothing to save.

Reply with JSON only, like {{"save": [{{"topic": "profile", "fact": "Works as a journalist"}}]}}. Use topic "profile" for personal facts and preferences, or a short lowercase topic name for a specific project. Reply {{"save": []}} if there is nothing to save."""


def extract_memory(model, user, answer):
    try:
        notes = "\n\n".join(f"## {t}\n{agent.read_memory(t)}" for t in agent.memory_topics()) or "(empty)"
        prompt = EXTRACT.format(notes=notes[:6000], user=user[:3000], answer=answer[:2000])
        think = {"think": False} if "thinking" in caps(model) else {}
        r = ollama.chat(model, messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0}, **think)
        for item in json.loads(r.message.content).get("save", [])[:5]:
            if isinstance(item, dict) and str(item.get("fact", "")).strip():
                agent.remember(str(item["fact"]), str(item.get("topic") or "profile"))
    except Exception:
        pass


def worker(job, cid, conv, model, thinking, goal, rounds, net, timeout, swarm=0, pool=(), mem=True, user_text="", offline=False, private=False):
    net = net and not offline
    agent.set_workspace(conv.get("project") or fdir(cid), net, int(timeout), conv.get("region", "wt-wt"))
    agent.CTX.vision = model if "vision" in caps(model) else FAST
    shown, msgs = conv["display"], conv["llm"]
    base = {k: v for k, v in tools.items() if (mem and not private or k not in agent.MEMORY_TOOLS) and not (offline and k in ("web_search", "read_page"))}
    toolset = dict(base, goal_done=goal_done, give_up=give_up) if goal else base
    rounds = 0 if swarm else int(rounds) if goal else 1
    language = agent.LANGUAGES.get(conv.get("lang"))
    evidence = None
    try:
        if not offline and user_text:
            box = {"role": "assistant", "content": "", "metadata": {"title": "🔎 " + t("Looking online…"), "status": "pending"}}

            def show():  # the box appears only once Joshua has decided to search
                if not any(b is box for b in shown):
                    shown.append(box)
                job.publish(shown)

            evidence = research(user_text, msgs, model, language, conv.get("region", "wt-wt"), box, show)
            box["metadata"] = {"title": "🔎 " + t("Sources from the web"), "status": "done"}
            if evidence:
                asked, original = msgs[-1], msgs[-1]["content"]
                asked["content"] = original + USE_SOURCES.format(evidence=evidence, language=language or "my language")
            elif any(b is box for b in shown):
                shown.remove(box)
            job.publish(shown)
        if swarm:
            run_swarm(job, cid, conv, model, thinking, int(swarm), pool, net, timeout, base)
            if job.stop.is_set():
                shown.append({"role": "assistant", "content": t("You stopped this task. Send a message to continue."), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
        for r in range(1, rounds + 1):
            called = []
            for view in agent_turn(msgs, shown, toolset, model, thinking, job.stop, called):
                job.publish(view)
            save(cid, conv)
            if job.stop.is_set():
                shown.append({"role": "assistant", "content": t("You stopped this task. Send a message to continue."), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
                break
            if not goal or {"goal_done", "give_up"} & set(called):
                break
            if r == rounds:
                shown.append({"role": "assistant", "content": t("The goal wasn't marked as done after {n} rounds. Send a message to continue.", n=rounds), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
                break
            shown.append({"role": "assistant", "content": t("Not marked as done yet, continuing."), "metadata": {"title": "🔁 " + t("Round {i} of {n}", i=r + 1, n=rounds), "status": "done"}})
            msgs.append({"role": "user", "content": NUDGE})
            job.publish(shown)
        if evidence and not goal and not job.stop.is_set():
            verify(evidence, msgs, shown, base, model, thinking, language, job)
        final = last_answer(shown) if evidence else None
        if final:
            cited = set(re.findall(r"\[(\d+)\]", final["content"]))
            refs = [f"[{n}] [{t.strip()[:80]}]({u})" for n, t, u in re.findall(r"^\[(\d+)\] (.*)\n(https?://\S+)", evidence, re.M) if n in cited]
            if refs:
                final["content"] += "\n\n**" + t("Sources") + "**  \n" + "  \n".join(refs)
        if evidence:
            asked["content"] = original + "\n\n(Web sources consulted: " + ", ".join(re.findall(r"^https?://\S+", evidence, re.M)) + ")"
    except Exception as e:
        text = str(e)
        low = text.lower()
        if is_cloud(model) and not providers.is_api(model) and ("unauthorized" in low or "401" in low):
            text = t("This cloud model needs your computer to be connected to an Ollama account. Open **☁ Cloud models** in the right column and press **Connect**.")
        elif is_cloud(model) and not providers.is_api(model) and any(w in low for w in ("free usage", "free plan", "402", "payment", "billing", "credit", "subscription", "insufficient", "balance")):
            free = re.search(r"[Tt]ry ([\w.:-]+)", text)  # Ollama suggests a model that is in the free plan
            if free:  # Ollama names a model the account can use for free
                text = (t("Ollama refused this model for your account. **{model}** is in the free plan: add it from **☁ Cloud models** and pick it in the menu. Or add credit (pay as you go) or a plan on [ollama.com](https://ollama.com/settings). Local models keep working for free.", model=free[1])
                        + "\n\n" + t("Ollama's message:") + " " + text)
            else:  # no free usage on this account (not every account gets it), or it's used up
                text = (t("Ollama refused this request: your Ollama account has no cloud credit. New Ollama accounts pay per use (older ones may have a small free allowance, which may be used up). [💳 Add a few dollars of credit on ollama.com](https://ollama.com/dashboard): it pays Ollama for the computing power of its servers, not Joshua, which is free. Gemma 4 31B costs well under a cent per message. Or switch to a local model in the menu below: those are free and run on this computer.")
                        + "\n\n" + t("Ollama's message:") + " " + text)
        elif is_cloud(model) and not providers.is_api(model) and any(w in low for w in ("429", "quota", "usage limit", "rate limit", "too many requests")):
            text = t("Ollama's cloud refused this request: you've reached your plan's usage limit or too many requests are running. Wait a moment, check your usage on [ollama.com](https://ollama.com/settings), or switch to a local model.")
        shown.append({"role": "assistant", "content": text, "metadata": {"title": "⚠️ " + t("Something went wrong"), "status": "done"}})
    finally:
        save(cid, conv)
        job.running = False
        job.publish(shown)
    if mem and user_text and not job.stop.is_set():
        answer = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), "")
        # memory notes are always written by a model on this computer, even after a cloud answer
        local = FAST if is_cloud(model) else resolve(conv.get("model") or model, False) if conv.get("model") == PAIR else model
        if not is_cloud(local) or share_memory_setting():  # cloud-only installs: learn only if you let cloud models see your notes
            extract_memory(local, user_text, answer)


def busy(cid):
    return cid in JOBS and JOBS[cid].running


def send(message, cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem=False):
    message = message or {}
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        return gr.skip(), *SKIP
    if busy(cid):
        gr.Warning(t("Joshua is still answering here. You can wait, press Stop, or start a new chat."))
        return gr.skip(), *SKIP
    if model == ADD:  # the menu is still on "＋ Add a model…"
        model = default_model()
    choice, model = model, resolve(model, thinking)
    cloud = is_cloud(model)
    if cloud and offline:
        gr.Warning(t("Cloud models need the internet. Please choose a local model, or turn off No internet."))
        return gr.skip(), *SKIP
    if thinking and "thinking" not in caps(model):
        gr.Info(t("{model} can't do deep thinking, so it will answer normally.", model=model))
    cid = cid or new_conv()
    conv = load(cid)
    conv["model"] = choice
    if cloud and conv.get("cloud_notice") != model:  # say it once in the chat, each time a new cloud model is picked
        conv["cloud_notice"] = model
        where = (t("From here on this chat is answered by **{model}** on {company}'s servers: each message sends the conversation there, including the text of files and pages Joshua reads.", model=model, company=providers.company(model))
                 if providers.is_api(model) else t("From here on this chat is answered by **{model}** on Ollama's servers: each message sends the conversation there, including the text of files and pages Joshua reads.", model=model))
        conv["display"].append({"role": "assistant", "content": "☁ *" + where + ("" if cloudmem else " " + t("Your memory notes are not sent.")) + " " + t("Everything stays stored only on this computer.") + "*"})
    conv["lang"] = agent.lang_for(text, conv.get("lang"))
    conv["region"] = agent.REGIONS.get(conv["lang"], "wt-wt") if conv["lang"] else "wt-wt"
    language = agent.LANGUAGES.get(conv["lang"])
    work = pathlib.Path(conv.get("project") or fdir(cid))
    shown, msgs = conv["display"], conv["llm"]
    names, images = [], []
    for f in files:
        src = pathlib.Path(f if isinstance(f, str) else f["path"])
        dst = work / src.name
        shutil.copy(src, dst)
        names.append(src.name)
        shown.append({"role": "user", "content": {"path": str(dst)}})
        if dst.suffix.lower() in IMAGES and "vision" in caps(model):
            images.append(str(dst))
    if text:
        shown.append({"role": "user", "content": ("🐝 " if swarm else "🎯 " if goal else "") + text})
    conv["title"] = conv["title"] or (text or names[0])[:48]
    where = f" Your workspace is the user's project folder {work}; use list_files to explore it and edit files with care." if conv.get("project") else ""
    online = " You have no internet access in this chat: you cannot search or read web pages, so answer from what you know and say when something may be out of date." if offline else "" if net else " Code you run has no internet access."
    online += f" The user writes in {language}: always reply in {language}." if language else ""
    system = {"role": "system", "content": SYSTEM.format(now=agent.get_datetime()) + where + online + (GOAL_RULES if goal and not swarm else "") + (agent.memory_prompt() if mem and (cloudmem or not cloud) else "")}
    if msgs and msgs[0]["role"] == "system":
        msgs[0] = system
    else:
        msgs.insert(0, system)
    note = f"\n\n(Files saved in the workspace: {', '.join(names)})" if names else ""
    goal = goal and not swarm
    msgs.append({"role": "user", "content": ("GOAL: " if goal else "") + text + note, "images": images})
    pool = [model] + [m for m in dict.fromkeys(resolve(v, thinking) for _, v in models()) if m != model and (cloud or not is_cloud(m))]  # a local chat never borrows a cloud model
    save(cid, conv)
    JOBS[cid] = job = Job(shown)
    threading.Thread(target=worker, args=(job, cid, conv, model, thinking, goal, rounds, net, timeout, int(agents) if swarm else 0, pool, mem, text, offline, cloud and not cloudmem), daemon=True).start()
    return shown, None, cid, gr.Radio(choices=conv_choices(), value=cid), file_list(cid)


def stop_chat(cid):
    if busy(cid):
        JOBS[cid].stop.set()


# ---------- sidebar actions ----------

def files_label(cid):
    project = cid and load(cid).get("project")
    return gr.File(value=file_list(cid), label=t("Project: {name}", name=pathlib.Path(project).name) if project else t("Files in this chat"))


def open_conv(cid):
    conv = load(cid)
    view = JOBS[cid].view if busy(cid) else conv["display"]
    used = conv.get("model")
    return view, cid, files_label(cid), conv.get("project", ""), used if used in [v for _, v in models()] else gr.skip()


def new_chat():
    return [], None, gr.Radio(choices=conv_choices(), value=None), files_label(None), ""


def set_project(path, cid):
    if busy(cid):
        gr.Warning(t("Joshua is still working. You can change the folder once it has finished."))
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    path = (path or "").strip()
    if path:
        p = pathlib.Path(path).expanduser().resolve()
        if not p.is_dir():
            gr.Warning(t("Folder not found: {path}", path=p))
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        path = str(p)
    if not cid and not path:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    cid = cid or new_conv()
    conv = load(cid)
    conv["project"] = path
    conv["title"] = conv["title"] or (f"📁 {pathlib.Path(path).name}" if path else "")
    save(cid, conv)
    return files_label(cid), cid, gr.Radio(choices=conv_choices(), value=cid), path


def upload_files(paths, cid):
    cid = cid or new_conv()
    for p in map(pathlib.Path, paths or []):
        if p.parent != wdir(cid):
            shutil.copy(p, wdir(cid) / p.name)
    return file_list(cid), cid, gr.Radio(choices=conv_choices(), value=cid)


def open_folder(cid):
    """Shows the chat's files folder (or its project folder) in Finder, or the file manager on Linux."""
    if cid:
        wdir(cid).mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(wdir(cid))])


def zip_files(cid):
    """All the chat's files in one .zip, ready on the "Download all" button."""
    names = file_list(cid)
    if not names or sum(os.path.getsize(n) for n in names) > 500_000_000:  # nothing to zip, or too big to zip every time
        return gr.DownloadButton(visible=False)
    for old in (ROOT / cid).glob("*.zip"):
        old.unlink()
    title = re.sub(r"[^\w\- ]+", "", load(cid)["title"]).strip()[:40] or "files"
    out = ROOT / cid / (title + ".zip")  # beside the files folder, not inside it
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(name, pathlib.Path(name).name)
    return gr.DownloadButton(value=str(out), visible=True)


def retry(cid, *settings):
    """↻ under the last answer: forget that answer and ask the same question again, with the model chosen now."""
    if not cid or busy(cid):
        return gr.skip(), *SKIP
    conv = load(cid)
    llm, shown = conv["llm"], conv["display"]
    asked = [k for k, m in enumerate(llm) if m["role"] == "user" and m["content"] != NUDGE]
    typed = [k for k, m in enumerate(shown) if m["role"] == "user" and isinstance(m["content"], str)]
    if not asked or not typed:
        return gr.skip(), *SKIP
    text = llm[asked[-1]]["content"].split("\n\n---\nSources found on the web just now")[0]  # take off what Joshua added to the question
    text = re.sub(r"(\n\n\((Files saved in the workspace|Web sources consulted): [^\n]*\))+$", "", text).removeprefix("GOAL: ")
    del llm[asked[-1]:], shown[typed[-1]:]
    save(cid, conv)
    return send({"text": text}, cid, *settings)


def delete_file(cid, ev: gr.DeletedFileData):
    if cid:
        name = pathlib.Path(ev.file.orig_name or ev.file.path).name
        (wdir(cid) / name).unlink(missing_ok=True)
    return gr.skip()


def mem_view(topic=None):
    topics = agent.memory_topics() or ["profile"]
    topic = topic if topic in topics else ("profile" if "profile" in topics else topics[0])
    path = agent._mem(topic)
    return gr.Dropdown(choices=topics, value=topic), path.read_text() if path.exists() else ""


def mem_save(topic, text):
    if text is None:  # you cancelled clearing the notes
        return gr.skip(), gr.skip()
    with agent.MEM_LOCK:
        path = agent._mem(topic or "profile")
        path.write_text(text.strip() + "\n") if text.strip() else path.unlink(missing_ok=True)
    gr.Info(t("Memory saved.") if text.strip() else t("Deleted the '{topic}' notes.", topic=topic))
    return mem_view(topic)


STEPS = {"web_search": t("searching the web"), "read_page": t("reading a web page"), "run_python": t("running code"), "install_package": t("installing a package"),
         "list_files": t("browsing files"), "read_file": t("reading a file"), "write_file": t("writing a file"), "look_at_image": t("looking at an image"),
         "generate_image": t("generating an image (a minute or two)"), "edit_image": t("editing the image (several minutes)"), "remember": t("updating memory"),
         "forget": t("updating memory"), "read_memory": t("reading memory"), "get_datetime": t("checking the date"), "goal_done": t("wrapping up"), "give_up": t("wrapping up"),
         t("Looking online…"): t("researching: searching and reading sources"), t("Sources from the web"): t("writing the answer"), t("Fact check"): t("fact-checking the answer"), t("Thinking"): t("thinking")}


def status_html(cid):
    job = JOBS.get(cid)
    if not job or not job.running:
        return ""
    step, last = t("starting"), job.view[-1] if job.view else {}
    meta = last.get("metadata") or {} if isinstance(last, dict) else {}
    if meta.get("title"):
        name = meta["title"].split(" ", 1)[-1]
        step = t("running a swarm of agents") if name.startswith(t("Agent {n}", n="").strip()) else STEPS.get(name, name.lower())
    elif isinstance(last, dict) and last.get("role") == "assistant":
        step = t("writing the answer")
    secs = int(time.time() - job.started)
    return f"<div class='working'><span class='wdot'></span>{t('JOSHUA IS WORKING')} · {html.escape(step)} · {secs // 60}:{secs % 60:02d}</div>"


def machine():
    """The kind of computer, for the feedback form: chip and memory, nothing that identifies it."""
    try:
        if sys.platform == "darwin":
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
            return f"{chip}, {ram} GB, macOS {platform.mac_ver()[0]}"
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**30
        return f"Linux {platform.machine()}, {ram} GB"
    except Exception:
        return ""


# the beta-test form on the website, filled in with what Joshua knows: its version and the kind of computer (the model is added on click)
FEEDBACK = (f"{SITE}/{'' if LANG == 'en' else LANG + '/'}feedback.html?version={VERSION}&computer=" + urllib.parse.quote(machine()))


def rename_open(cid):
    if not cid:
        gr.Info(t("Open a chat first."))
        return gr.skip()
    return gr.Textbox(value=load(cid)["title"], visible=True)


def rename_save(cid, name):
    name = (name or "").strip()
    if cid and name:
        conv = load(cid)
        conv["title"] = name[:80]
        (ROOT / cid / "chat.json").write_text(json.dumps(conv, ensure_ascii=False, indent=1))  # not save(): a new name doesn't move the chat to the top
    return gr.Textbox(visible=False), gr.Radio(choices=conv_choices(), value=cid)


def delete_chat(cid):
    nothing = (gr.skip(),) * 5
    if not cid or not (ROOT / cid).exists():
        return nothing
    if busy(cid):
        gr.Warning(t("Joshua is still answering in this chat. Press Stop first, then you can delete it."))
        return nothing
    title = re.sub(r"[^\w -]+", "", load(cid)["title"] or "chat")[:40].strip() or "chat"
    dest = f"Joshua chat - {title} - {cid}"
    try:
        if agent.MAC:
            shutil.move(str(ROOT / cid), str(pathlib.Path.home() / ".Trash" / dest))
        else:  # the Linux desktop Trash: the files, plus a note saying where they came from
            trash = pathlib.Path.home() / ".local" / "share" / "Trash"
            (trash / "files").mkdir(parents=True, exist_ok=True), (trash / "info").mkdir(parents=True, exist_ok=True)
            (trash / "info" / f"{dest}.trashinfo").write_text(f"[Trash Info]\nPath={ROOT / cid}\nDeletionDate={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            shutil.move(str(ROOT / cid), str(trash / "files" / dest))
    except Exception:
        shutil.rmtree(ROOT / cid, ignore_errors=True)
    JOBS.pop(cid, None)
    gr.Info(t("Chat deleted (moved to the Trash)."))
    return new_chat()


SEEN = {}


def poll(cid, request: gr.Request):
    """Called every second by each open page: sends only what changed (chat view, chat list, files)."""
    seen = SEEN.setdefault(request.session_hash, {})
    out = []
    job = JOBS.get(cid)
    sig = (cid, job.version) if job else None
    out.append(job.view if job and seen.get("view") != sig else gr.skip())
    seen["view"] = sig
    choices = conv_choices()
    out.append(gr.Radio(choices=choices, value=cid) if seen.get("list") != choices else gr.skip())
    seen["list"] = choices
    cur = file_list(cid)
    fsig = (cid, tuple((p, os.path.getmtime(p)) for p in cur))
    out.append(cur if seen.get("files") != fsig else gr.skip())
    seen["files"] = fsig
    status = status_html(cid)
    out.append(status if seen.get("status") != status else gr.skip())
    seen["status"] = status
    return out


CSS = """
body:not([data-screen="green"]):not([data-screen="amber"]):not([data-screen="modern"]):not([data-screen="modern-night"]):not([data-screen="blue"]):not([data-screen="black"]):not([data-screen="white"]) {--bg: #dfdbc3; --bg-hi: #e9e5cf; --panel: #e7e3cc; --codebg: #d4cfb3; --line: rgba(90, 60, 40, .45); --cyan: #9c3b1c; --glow: transparent; --ink: #3b2322; --dim: #7a6250; --amber: #9c3b1c; --red: #b3261e; --codeink: #3b2322; --shadow: none}
body[data-screen="black"] {--bg: #000308; --bg-hi: #040a14; --panel: #050a14; --codebg: #02050a; --line: #8fe9ff; --cyan: #8fe9ff; --glow: rgba(80, 200, 255, .55); --ink: #a6e6ff; --dim: #5d8ea6; --amber: #ffb347; --red: #ff5a4e; --codeink: #59f08a; --shadow: 0 0 8px rgba(80, 200, 255, .55); --check: #ffb347}
body[data-screen="black"] {--bw: 2px; --panel-shadow: 0 0 8px var(--glow), inset 0 0 18px rgba(80, 200, 255, .12); --case: uppercase; --tglow: 0 0 3px rgba(120, 220, 255, .35); --flick: flick 3s infinite; --pbw: 2px; --pline: var(--amber); --pglow: 0 0 8px rgba(255, 179, 71, .35)}
body[data-screen="black"]::after {content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 9999; background: repeating-linear-gradient(to bottom, rgba(0, 0, 0, .22) 0, rgba(0, 0, 0, .22) 1px, transparent 1px, transparent 3px)}
body {--bw: 1px; --panel-shadow: 0 0 8px var(--glow); --case: none; --tglow: none; --flick: none; --pbw: 1px; --pline: color-mix(in srgb, var(--amber) 70%, transparent); --pglow: 0 0 5px color-mix(in srgb, var(--amber) 25%, transparent)}
body[data-screen="white"] {--bg: #ffffff; --bg-hi: #ffffff; --panel: #fbfbfb; --codebg: #f1f1f1; --line: rgba(0, 0, 0, .26); --cyan: #000000; --glow: transparent; --ink: #141414; --dim: #6b6b6b; --amber: #1f5fbf; --red: #c4271d; --codeink: #141414; --shadow: none}
body[data-screen="green"] {--bg: #0e1a14; --bg-hi: #1a3024; --panel: #13241b; --codebg: #09130e; --line: rgba(140, 255, 180, .42); --cyan: #8dffb4; --glow: rgba(120, 255, 170, .2); --ink: #d4f7df; --dim: #86b096; --amber: #ffd27a; --red: #ff6b5e; --codeink: #8dffb4; --shadow: 0 0 6px rgba(120, 255, 170, .35)}
body[data-screen="amber"] {--bg: #1c120c; --bg-hi: #2e1d13; --panel: #25170f; --codebg: #140c07; --line: rgba(255, 180, 100, .45); --cyan: #ffb866; --glow: rgba(255, 170, 80, .2); --ink: #ffe5c8; --dim: #bf9370; --amber: #9fe3ff; --red: #ff6b5e; --codeink: #ffd08a; --shadow: 0 0 6px rgba(255, 170, 80, .35)}
body[data-screen="blue"] {--bg: #08101b; --bg-hi: #10203a; --panel: #0d1625; --codebg: #060b14; --line: rgba(143, 233, 255, .5); --cyan: #8fe9ff; --glow: rgba(80, 200, 255, .26); --ink: #d6ecf5; --dim: #7fa3b5; --amber: #ffbf66; --red: #ff6b5e; --codeink: #6fe8a0; --shadow: 0 0 6px rgba(80, 200, 255, .4)}
:root body, :root body .dark {--green: #6fe8a0; --checkbox-background-color-selected: var(--check, var(--cyan)); --checkbox-border-color-selected: var(--check, var(--cyan)); --checkbox-border-color: var(--dim); --slider-color: var(--cyan); --button-primary-background-fill: var(--panel); --button-primary-text-color: var(--cyan); --button-primary-border-color: var(--line);
  --body-background-fill: var(--bg); --background-fill-primary: var(--panel); --background-fill-secondary: var(--codebg); --block-background-fill: var(--panel); --block-label-background-fill: var(--panel);
  --input-background-fill: var(--codebg); --body-text-color: var(--ink); --body-text-color-subdued: var(--dim); --border-color-primary: var(--line); --block-border-color: var(--line); --input-border-color: var(--line);
  --block-label-text-color: var(--cyan); --block-title-text-color: var(--cyan); --button-secondary-background-fill: var(--panel); --button-secondary-text-color: var(--ink); --checkbox-background-color: var(--codebg);
  --link-text-color: var(--amber); --color-accent-soft: var(--panel); --table-even-background-fill: var(--panel); --table-odd-background-fill: var(--codebg)}
body, .gradio-container {background: radial-gradient(ellipse at 50% 40%, var(--bg-hi) 0%, var(--bg) 80%) fixed !important; color: var(--ink)}
footer {display: none !important}
.gradio-container {max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 0 !important}
.gradio-container .main.fillable {padding-left: 12px !important; padding-right: 12px !important}

.gradio-container, .gradio-container *:not(svg):not(path) {font-family: 'VT323', ui-monospace, Menlo, monospace !important}
.gradio-container {font-size: 19px !important}
.gradio-container .prose, .gradio-container .prose * {font-size: 1.32rem !important; line-height: 1.32 !important}
.gradio-container .prose pre, .gradio-container .prose pre *, .gradio-container .prose code {font-size: 1.22rem !important; line-height: 1.25 !important}
.side label > span, #convs label, .side .label-wrap span, .side button {font-size: 1.2rem !important}
.side .info, .side [data-testid="block-info"] {font-size: 1.02rem !important; line-height: 1.15 !important}
#promptrow textarea {font-size: 1.3rem !important}
#chat .placeholder, #chat .placeholder * {font-size: 1.6rem !important; color: var(--ink); letter-spacing: .06em}
/* glowing wall-screen panels */
.side, #chat {background: color-mix(in srgb, var(--panel) 88%, transparent) !important; border: var(--bw) solid var(--line) !important; border-radius: 4px !important; box-shadow: var(--panel-shadow) !important}
.gradio-container * {text-shadow: var(--tglow)}
.side label > span, .side .label-wrap span, #convs label, #chat .placeholder * {text-transform: var(--case)}
#chat .placeholder {padding: 28px 36px !important; box-sizing: border-box}
#chat .bubble-wrap {padding: 14px 18px !important; box-sizing: border-box}
#promptrow .form {border: none !important; box-shadow: none !important; background: transparent !important}
/* clearer separators */
.opt {border: none !important; border-bottom: 1px dashed color-mix(in srgb, var(--line) 55%, transparent) !important; border-radius: 0 !important; padding-bottom: 10px !important}
#chat .message-row {margin: 12px 0 !important}
#chat .message-row + .message-row {border-top: 1px dashed color-mix(in srgb, var(--line) 45%, transparent); padding-top: 12px}
.message.bot, .message.user {padding: 12px 14px !important}
.prose hr {border: none !important; border-top: 1px dashed var(--line) !important; margin: 14px 0 !important}
/* code snippets: their own framed block */
.message.bot pre, .message.user pre, .prose pre {background: var(--codebg) !important; border: 1px solid var(--line) !important; border-left: 3px solid var(--cyan) !important; border-radius: 3px !important; padding: 12px 14px !important; margin: 14px 0 !important; overflow-x: auto; box-shadow: inset 0 0 10px rgba(0, 0, 0, .35) !important}
.message.bot pre code, .prose pre code {color: var(--codeink) !important}
.message.bot :not(pre) > code, .prose :not(pre) > code {background: var(--codebg) !important; border: 1px solid color-mix(in srgb, var(--line) 60%, transparent) !important; border-radius: 3px; padding: 1px 5px; color: var(--cyan) !important}
#screen, #textsize {background: transparent !important; border: none !important; box-shadow: none !important; padding: 4px 2px 0 !important}
#screen .wrap, #textsize .wrap, #kprov .wrap {gap: 4px; flex-wrap: wrap; border: none !important}
.side .info-text {font-size: 1.05rem !important; line-height: 1.1 !important; color: var(--dim) !important}
.side .label-wrap, .side .label-wrap * {color: var(--ink) !important}
textarea::placeholder {color: var(--dim) !important; opacity: .85}
.status {letter-spacing: .06em !important}
#screen label, #textsize label, #kprov label {padding: 1px 7px !important; font-size: 1rem !important; border: 1px solid var(--line) !important; border-radius: 2px !important; background: transparent !important; color: var(--dim) !important; box-shadow: none !important; text-transform: uppercase; letter-spacing: .06em}
#screen label.selected, #textsize label.selected, #kprov label.selected {color: var(--bg) !important; background: var(--cyan) !important}
#screen input, #textsize input, #kprov input {display: none}
.side {padding: 12px !important}
#brand {font-size: 3rem !important; letter-spacing: .1em; color: var(--ink); padding: 2px 4px 0; text-shadow: var(--shadow); line-height: 1}
#brand::after {color: var(--cyan)}
#brand::after {content: "█"; margin-left: 4px; animation: blink 1.1s steps(1) infinite}
@keyframes blink {50% {opacity: 0}}
.lights {display: flex; gap: 5px; padding: 2px 6px 10px}
.lights span {width: 14px; height: 5px; border-radius: 1px; box-shadow: 0 0 3px currentColor; background: currentColor; opacity: .85; animation: var(--flick)}
.lights span:nth-child(1) {color: var(--green)} .lights span:nth-child(2) {color: var(--cyan); animation-delay: .7s} .lights span:nth-child(3) {color: var(--amber); animation-delay: 1.3s} .lights span:nth-child(4) {color: var(--red); animation-delay: 2.1s} .lights span:nth-child(5) {color: #5aa0ff; animation-delay: .4s}
@keyframes flick {0%, 100% {opacity: 1} 45% {opacity: .35} 50% {opacity: 1}}
.status {font-size: .7rem; color: var(--dim); letter-spacing: .1em; padding: 0 6px 8px}
/* uppercase console labels */
#newchat, #stop {text-transform: uppercase; letter-spacing: .06em}
.side .info, .side [data-testid="block-info"] {color: var(--dim) !important; font-size: .78rem}
#newchat {justify-content: flex-start; border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
#newchat:hover {background: color-mix(in srgb, var(--cyan) 12%, transparent) !important}
#convs {background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important}
#convs .wrap {flex-direction: column; gap: 1px; border: none !important; box-shadow: none !important}
#convs label {width: 100%; padding: 6px 8px; border-radius: 2px; background: transparent; border: none; box-shadow: none; font-size: .82rem; cursor: pointer; color: var(--ink)}
#convs label span {overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-left: 0}
#convs label:hover {background: color-mix(in srgb, var(--cyan) 10%, transparent)}
#convs label.selected {background: color-mix(in srgb, var(--cyan) 18%, transparent); color: var(--ink)}
#convs label.selected::before {content: "> "; color: var(--amber)}
#convs input {display: none}
#convs, #convs * {outline: none !important}
#convs label.selected, #convs label:focus-within {border: none !important; box-shadow: none !important; outline: none !important}
/* chat: computer in cyan, you in amber */
.message.bot {font-size: .95rem; color: var(--ink) !important; background: color-mix(in srgb, var(--cyan) 4%, transparent) !important; border: 1px solid color-mix(in srgb, var(--line) 60%, transparent) !important; border-radius: 4px !important}
.message.user {background: color-mix(in srgb, var(--amber) 8%, transparent) !important; border: 1px solid color-mix(in srgb, var(--amber) 60%, transparent) !important; border-radius: 4px !important}
.message.user *, .message.bot * {border: none !important; background: transparent !important; box-shadow: none !important}
.message.user, .message.user * {color: var(--amber) !important}
.message.bot .prose {color: var(--ink) !important}
.prose a {color: var(--amber) !important}
.prose code, .prose pre {background: #02050a !important; color: var(--green) !important; border: 1px solid #12324a}
/* files panel */
#wfiles {font-size: .8rem}
body.noside #leftcol {display: none !important}
#chatcol {position: relative}
#chatopts, #panels {gap: 10px; padding: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important}
#filebtns {gap: 6px; flex-wrap: nowrap} #filebtns button {font-size: .76rem !important; min-width: 0 !important; padding: 4px 6px !important}
#wfiles .file-preview-holder {max-height: 40vh; overflow-y: auto}
#wfiles tr.file {display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 2px; background: transparent !important}
#wfiles tr.file:hover {background: color-mix(in srgb, var(--cyan) 10%, transparent) !important}
#wfiles td {padding: 0 !important}
#wfiles td.filename {flex: 1 1 auto; min-width: 0; width: auto !important; display: flex; justify-content: flex-start; text-align: left; overflow: hidden}
#wfiles td.filename .stem {flex: 0 1 auto; min-width: 0; width: auto !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap}
#wfiles td.filename .ext {flex: 0 0 auto}
#wfiles td.download {flex: 0 0 auto; min-width: 0 !important; width: auto !important; font-size: .72rem; opacity: .7}
#wfiles tr.file td:last-child {flex: 0 0 auto; min-width: 0 !important; width: auto !important}
#wfiles .label-clear-button {opacity: 0; transition: opacity .15s; font-size: 1rem; padding: 0 2px; color: var(--red)}
#wfiles tr.file:hover .label-clear-button {opacity: 1}
#wfiles button[aria-label="Clear"], #wfiles button[title="Clear"] {display: none}
/* prompt line */
#stop {min-width: 64px !important; max-width: 72px; font-size: 1.1rem !important; align-self: stretch; background: transparent !important; border: 1px solid var(--red) !important; color: var(--red) !important}
#stop:hover {background: rgba(255, 90, 78, .12) !important}
#promptrow textarea {color: var(--amber) !important; caret-color: var(--amber)}
#promptrow .block, #promptrow .multimodal-textbox {background: var(--panel) !important; border: var(--pbw) solid var(--pline) !important; border-radius: 4px !important; box-shadow: var(--pglow) !important}
#tools {font-size: .82rem; line-height: 1.8; opacity: .9}
#modelrow {justify-content: space-between; align-items: center; margin-top: -6px; padding-right: 86px}
#working {flex: 1 1 auto !important; min-width: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 0 0 4px !important}
#working .working {display: flex; align-items: center; gap: 8px; color: var(--cyan); font-size: 1.15rem; letter-spacing: .05em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
#working .progress-text, #working .eta-bar, #working .timer, #working .meta-text, #working .meta-text-center, #working .progress-level {display: none !important}
#working .wdot {flex: 0 0 auto; width: 9px; height: 15px; background: var(--cyan); animation: blink 1s steps(1) infinite}
#renchat {justify-content: flex-start}
#delchat {justify-content: flex-start; border: 1px dashed color-mix(in srgb, var(--red) 60%, transparent) !important; background: transparent !important; color: var(--red) !important; font-size: 1rem !important; margin-top: 6px}
#delchat:hover {background: color-mix(in srgb, var(--red) 10%, transparent) !important}
#model {max-width: 230px; background: transparent !important; border: none !important; box-shadow: none !important}
#model input {font-size: 1.1rem !important; text-align: right; padding-right: 20px !important; color: var(--dim); cursor: pointer; text-transform: uppercase; letter-spacing: .08em}
#model .wrap {background: transparent !important; border: none !important; box-shadow: none !important}
#tunebtn {justify-content: flex-start; border: 1px solid var(--cyan) !important; background: color-mix(in srgb, var(--cyan) 8%, transparent) !important; color: var(--cyan) !important; font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: .06em; margin-top: 6px}
#tunebtn:hover {background: color-mix(in srgb, var(--cyan) 10%, transparent) !important}
#tune {background: color-mix(in srgb, var(--panel) 88%, transparent) !important; border: var(--bw) solid var(--line) !important; border-radius: 4px !important; box-shadow: var(--panel-shadow) !important; padding: 14px 18px !important; gap: 10px}
#tunetitle {font-size: 2rem; letter-spacing: .08em; color: var(--cyan); text-shadow: var(--shadow)}
#tuneback {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
.ftstep {border: none !important; border-top: 1px dashed var(--line) !important; border-radius: 0 !important; background: transparent !important; box-shadow: none !important}
.ftstep > .label-wrap span {font-size: 1.45rem !important; color: var(--cyan) !important; letter-spacing: .04em}
#tune .prose, #tune .prose * {font-size: 1.25rem !important; line-height: 1.3 !important}
#tune .prose strong {color: var(--amber)}
.ftres, .ftcount, .ftstatus {font-size: 1.2rem; letter-spacing: .04em; color: var(--ink); padding: 6px 10px; border-left: 3px solid var(--cyan); background: var(--codebg)}
.ftres b {color: var(--cyan)}
.ftstatus.ok {border-left-color: var(--green); color: var(--green)} .ftstatus.bad {border-left-color: var(--red); color: var(--red)} .ftstatus.run {color: var(--cyan)}
.ftstatus .wdot {display: inline-block; width: 9px; height: 15px; margin-right: 8px; background: var(--cyan); vertical-align: -2px; animation: blink 1s steps(1) infinite}
#ftlog textarea {font-size: 1.05rem !important; line-height: 1.15 !important; color: var(--codeink) !important; background: var(--codebg) !important}
.ftans {min-height: 140px; background: var(--codebg) !important; border: 1px solid var(--line) !important; padding: 10px 12px !important}
#ftstop, #ftdel {border: 1px solid var(--red) !important; background: transparent !important; color: var(--red) !important}
#ftgrid {font-size: 1.1rem}
#tune button:not(#ftstop):not(#ftdel):not(.label-wrap) {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important; box-shadow: none !important}
#tune button.primary {background: color-mix(in srgb, var(--cyan) 14%, transparent) !important}
#tune button:hover {background: color-mix(in srgb, var(--cyan) 10%, transparent) !important}
#tune fieldset label {background: transparent !important; border: 1px solid var(--line) !important; color: var(--ink) !important; box-shadow: none !important}
#tune fieldset label.selected {background: var(--cyan) !important; color: var(--bg) !important}
#tune fieldset label.selected * {color: var(--bg) !important}
#tune .prose ol, #tune .prose ul {padding-left: 1.6em !important}
.ftstatus.ok {color: var(--ink) !important}
#tune .legend, #tune [class*=legend] {display: none !important}
#modelrow {flex-wrap: nowrap !important}
#modelrow > #working {flex: 0 1 auto !important; width: auto !important; min-width: 0 !important}
#modelrow > #cloudbadge {flex: 1 1 auto !important; width: auto !important; min-width: 0 !important}
#modelrow > #model {flex: 0 0 auto !important}
#cloudbadge .cloudbadge {font-size: 1rem; line-height: 1.2; color: var(--amber); padding: 2px 8px 0; text-align: right}
.cloudstat {font-size: 1.05rem; line-height: 1.3; padding: 6px 0; color: var(--ink)}
#localhelp, #localhelp *, #apihelp, #apihelp *, #schedhelp, #schedhelp * {font-size: 1.05rem !important; line-height: 1.3 !important}
.task {font-size: 1.15rem; line-height: 1.25; padding: 8px 0; border-bottom: 1px dashed var(--line); color: var(--ink)}
.taskp {color: var(--dim); font-size: 1rem; margin-top: 3px}
#schedbtn {justify-content: flex-start; border: 1px solid var(--cyan) !important; background: color-mix(in srgb, var(--cyan) 8%, transparent) !important; color: var(--cyan) !important; font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: .06em}
#schedpanel {background: color-mix(in srgb, var(--panel) 88%, transparent) !important; border: var(--bw) solid var(--line) !important; border-radius: 4px !important; box-shadow: var(--panel-shadow) !important; padding: 14px 18px !important; gap: 12px}
#schedtitle {font-size: 2rem; letter-spacing: .08em; color: var(--cyan); text-shadow: var(--shadow)}
#schedback {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
.schedsub {font-size: 1.2rem; letter-spacing: .08em; color: var(--cyan); padding-bottom: 4px; border-bottom: 1px dashed var(--line)}
#schedpanel .prose, #schedpanel .prose * {font-size: 1.25rem !important; line-height: 1.3 !important}
#schedpanel .prose strong {color: var(--amber)}
#schedel {border: 1px solid var(--red) !important; background: transparent !important; color: var(--red) !important}
#schedpanel label span, #schedpanel [data-testid="block-info"] {font-size: 1.15rem !important}
#schedpanel input, #schedpanel textarea {font-size: 1.2rem !important; color: var(--ink) !important}
#schedpanel button {font-size: 1.15rem !important}
#language {margin: 2px 0 10px !important; max-width: 220px}
#stop, #stop * {white-space: nowrap !important}
#chatcol:not(:has(#working .working)) #stop {display: none !important}
#wfiles button.upload-container {min-height: 0 !important; height: auto !important; padding: 10px 8px !important}
#wfiles button.upload-container .wrap {flex-direction: row !important; gap: 8px !important; min-height: 0 !important; height: auto !important}
#wfiles button.upload-container .icon-wrap {width: 18px !important; height: 18px !important; margin: 0 !important}
#wfiles button.upload-container .or:empty {display: none}
#schedel {background: transparent !important; border: none !important; box-shadow: none !important; text-decoration: underline; width: auto !important; align-self: flex-start; padding: 2px 0 !important; opacity: .85}
#modelrow #model::before {content: "@@MODEL@@"; color: var(--dim); font-size: .85rem; margin-right: 6px; align-self: center; white-space: nowrap}
#modelrow #model {display: flex !important; flex-wrap: nowrap !important; align-items: center; min-width: 240px !important}
#modelrow #model > .wrap, #modelrow #model > div:not([data-testid]) {flex: 1}
#model input {text-transform: none !important}
#language input {font-size: .9rem !important}
.cloudlinks {display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 4px 0 6px}
.cloudlinks a {color: var(--red) !important; text-decoration: underline; font-size: .95rem}
.cloudnote {font-size: .85rem; color: var(--dim); margin-bottom: 6px}
#schedpanel button.primary {background: var(--red) !important; color: var(--bg) !important; border: none !important; padding: 10px 28px !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button.primary {background: var(--ink) !important; border-radius: 10px !important}
#schedpanel button:not(#schedel):not(.primary) {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
#schedpanel fieldset label {background: transparent !important; border: 1px solid var(--line) !important; color: var(--ink) !important; box-shadow: none !important}
#schedpanel fieldset label.selected {background: var(--cyan) !important; color: var(--bg) !important}
#schedpanel fieldset label.selected * {color: var(--bg) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel label span, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel [data-testid="block-info"] {font-size: .85rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel input, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel textarea, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button {font-size: .92rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button:not(#schedel):not(.primary) {border: none !important; background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel fieldset label {border-radius: 999px !important; border-color: var(--line) !important; padding: 4px 12px !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel fieldset label.selected {background: var(--ink) !important; border-color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel fieldset label.selected * {color: var(--bg) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel {background: var(--panel) !important; border: none !important; border-radius: 16px !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedtitle {font-family: 'Newsreader', Georgia, serif !important; font-size: 1.7rem !important; color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .prose, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .prose *, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .task, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .task * {font-size: .95rem !important; line-height: 1.5 !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .prose strong {color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) .schedsub {font-size: .78rem !important; letter-spacing: .1em !important; color: var(--dim) !important; border-bottom: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedbtn {border: none !important; background: transparent !important; text-transform: none !important; font-size: .9rem !important; border-radius: 10px !important; letter-spacing: normal !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedbtn:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedback {border-radius: 10px !important; border: none !important; background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button {border-radius: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button.primary {background: var(--cyan) !important; color: var(--panel) !important; border-color: var(--cyan) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .block, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .form {border: none !important; box-shadow: none !important}
.task span {color: var(--dim)}
body[data-screen^="modern"]:not(#a):not(#b) #chatopts .task, body[data-screen^="modern"]:not(#a):not(#b) #chatopts .task * {font-size: .85rem !important; line-height: 1.45 !important}
body[data-screen^="modern"]:not(#a):not(#b) .task {border-bottom: none}
.cloudstat code {word-break: break-all}
body[data-screen^="modern"]:not(#a):not(#b) #chatopts .prose .cloudstat, body[data-screen^="modern"]:not(#a):not(#b) #chatopts .cloudstat * {font-size: .85rem !important; line-height: 1.45 !important}
.updnote {font-size: 1rem; line-height: 1.25; color: var(--amber); padding: 4px 4px 0}
#updbtn {border: 1px solid var(--amber) !important; background: transparent !important; color: var(--amber) !important}
body[data-screen^="modern"]:not(#a):not(#b) #localhelp, body[data-screen^="modern"]:not(#a):not(#b) #localhelp *, body[data-screen^="modern"]:not(#a):not(#b) #apihelp, body[data-screen^="modern"]:not(#a):not(#b) #apihelp *, body[data-screen^="modern"]:not(#a):not(#b) #schedhelp, body[data-screen^="modern"]:not(#a):not(#b) #schedhelp *, body[data-screen^="modern"]:not(#a):not(#b) .updnote {font-size: .85rem !important; line-height: 1.45 !important}
body[data-screen^="modern"]:not(#a):not(#b) #updbtn {border-radius: 10px !important; text-transform: none !important; border: none !important; background: var(--soft) !important; color: var(--cyan) !important}
.cloudstat a {color: var(--cyan)}
#cloudhelp, #cloudhelp *, #keyshelp, #keyshelp * {font-size: 1.05rem !important; line-height: 1.3 !important}
body[data-screen^="modern"]:not(#a):not(#b) #cloudbadge .cloudbadge {font-size: .85rem !important; color: var(--dim) !important; padding: 4px 8px 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #cloudhelp, body[data-screen^="modern"]:not(#a):not(#b) #cloudhelp *, body[data-screen^="modern"]:not(#a):not(#b) .cloudstat, body[data-screen^="modern"]:not(#a):not(#b) #keyshelp, body[data-screen^="modern"]:not(#a):not(#b) #keyshelp * {font-size: .85rem !important; line-height: 1.45 !important}
#model ul.options {font-size: .82rem; background: var(--panel) !important; border: 1px solid var(--cyan) !important}

/* ===== Modern look: calm, rounded, readable. Everything below applies only when a Modern screen is chosen.
   ':not(#a):not(#b)' matches always; it only makes these rules outrank the retro ones (Gradio adds a prefix that strengthens them). ===== */
body[data-screen="modern"] {--bg: #f6f4ef; --bg-hi: #f6f4ef; --panel: #fffdf9; --codebg: #f1eee7; --line: rgba(60, 50, 40, .13); --cyan: #2f6f62; --glow: transparent; --ink: #2a2622; --dim: #7b736a; --amber: #2f6f62; --red: #b5473a; --codeink: #2a2622; --shadow: none; --check: #2f6f62; --userbg: #ebe7de; --green: #2f6f62; --soft: rgba(47, 111, 98, .09)}
body[data-screen="modern-night"] {--bg: #1e1c19; --bg-hi: #1e1c19; --panel: #26231f; --codebg: #191714; --line: rgba(255, 240, 220, .11); --cyan: #7cc4b2; --glow: transparent; --ink: #ece6dc; --dim: #a39a8e; --amber: #7cc4b2; --red: #e07a6a; --codeink: #ece6dc; --shadow: none; --check: #7cc4b2; --userbg: #33302b; --green: #7cc4b2; --soft: rgba(124, 196, 178, .1)}
body[data-screen^="modern"]:not(#a):not(#b) {--bw: 1px; --panel-shadow: 0 1px 2px rgba(0, 0, 0, .04), 0 6px 24px rgba(0, 0, 0, .04); --case: none; --tglow: none; --flick: none; --pbw: 1px; --pline: var(--line); --pglow: 0 2px 12px rgba(0, 0, 0, .07)}
body[data-screen^="modern"], body[data-screen^="modern"]:not(#a):not(#b) .gradio-container {background: var(--bg) !important}
body[data-screen^="modern"]:not(#a):not(#b) .gradio-container, body[data-screen^="modern"]:not(#a):not(#b) .gradio-container *:not(svg):not(path) {font-family: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif !important; text-shadow: none !important; letter-spacing: normal !important}
body[data-screen^="modern"]:not(#a):not(#b) .gradio-container {font-size: 15px !important}
body[data-screen^="modern"]:not(#a):not(#b) .gradio-container .prose, body[data-screen^="modern"]:not(#a):not(#b) .gradio-container .prose * {font-size: 1rem !important; line-height: 1.62 !important}
body[data-screen^="modern"]:not(#a):not(#b) .gradio-container pre, body[data-screen^="modern"]:not(#a):not(#b) .gradio-container pre *, body[data-screen^="modern"]:not(#a):not(#b) .gradio-container code, body[data-screen^="modern"]:not(#a):not(#b) #ftlog textarea {font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace !important; font-size: .86rem !important; line-height: 1.5 !important}
body[data-screen^="modern"]:not(#a):not(#b) .prose pre, body[data-screen^="modern"]:not(#a):not(#b) .message.bot pre {border: 1px solid var(--line) !important; border-left: 1px solid var(--line) !important; border-radius: 10px !important; box-shadow: none !important; background: var(--codebg) !important}
body[data-screen^="modern"]:not(#a):not(#b) .message.bot :not(pre) > code, body[data-screen^="modern"]:not(#a):not(#b) .prose :not(pre) > code {color: var(--ink) !important; border-radius: 5px !important; border: 1px solid var(--line) !important}
/* panels and brand */
body[data-screen^="modern"]:not(#a):not(#b) .side, body[data-screen^="modern"]:not(#a):not(#b) #chat, body[data-screen^="modern"]:not(#a):not(#b) #tune {background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 16px !important; box-shadow: var(--panel-shadow) !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat {background: transparent !important; border: none !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #brand {font-family: 'Newsreader', Georgia, serif !important; font-size: 2rem !important; font-weight: 500 !important; color: var(--ink) !important; padding: 4px 6px 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #brand::after {content: "." !important; color: var(--cyan) !important; animation: none !important; margin-left: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) .lights {display: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #brand {text-transform: lowercase !important}
body[data-screen^="modern"]:not(#a):not(#b) #brand::first-letter {text-transform: uppercase !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .placeholder {display: flex !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; height: 100% !important}
body[data-screen^="modern"]:not(#a):not(#b) .side .prose .status, body[data-screen^="modern"]:not(#a):not(#b) .status {font-size: .78rem !important; color: var(--dim) !important; padding: 2px 6px 12px !important}
body[data-screen^="modern"]:not(#a):not(#b) .side label > span, body[data-screen^="modern"]:not(#a):not(#b) .side .label-wrap span, body[data-screen^="modern"]:not(#a):not(#b) .side button, body[data-screen^="modern"]:not(#a):not(#b) #convs label {font-size: .93rem !important; text-transform: none !important}
body[data-screen^="modern"]:not(#a):not(#b) .side .info, body[data-screen^="modern"]:not(#a):not(#b) .side .info-text, body[data-screen^="modern"]:not(#a):not(#b) .side [data-testid="block-info"] {font-size: .8rem !important; line-height: 1.35 !important}
body[data-screen^="modern"]:not(#a):not(#b) .opt {border-bottom: 1px solid var(--line) !important}
/* screen picker as soft pills */
body[data-screen^="modern"]:not(#a):not(#b) #screen label, body[data-screen^="modern"]:not(#a):not(#b) #textsize label, body[data-screen^="modern"]:not(#a):not(#b) #kprov label {border-radius: 999px !important; text-transform: none !important; font-size: .8rem !important; padding: 2px 10px !important; border-color: var(--line) !important}
body[data-screen^="modern"]:not(#a):not(#b) #screen label.selected, body[data-screen^="modern"]:not(#a):not(#b) #textsize label.selected, body[data-screen^="modern"]:not(#a):not(#b) #kprov label.selected {background: var(--ink) !important; color: var(--bg) !important; border-color: var(--ink) !important}
/* sidebar buttons and chat list */
body[data-screen^="modern"]:not(#a):not(#b) #newchat {border-radius: 10px !important; text-transform: none !important; border: 1px solid var(--line) !important; background: var(--panel) !important; color: var(--ink) !important; font-weight: 500 !important}
body[data-screen^="modern"]:not(#a):not(#b) #newchat:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs label {border-radius: 8px !important; padding: 7px 10px !important; color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs label:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs label.selected {background: var(--soft) !important; font-weight: 500 !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs label.selected::before {content: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #delchat, body[data-screen^="modern"]:not(#a):not(#b) #tunebtn {border-radius: 10px !important; text-transform: none !important; border-style: solid !important; font-size: .9rem !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #fbbtn {justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #renchat {border-color: transparent !important; color: var(--dim) !important; justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important; background: transparent !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #delchat {border-color: transparent !important; color: var(--dim) !important}
body[data-screen^="modern"]:not(#a):not(#b) #delchat:hover {color: var(--red) !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #tunebtn {border-color: var(--line) !important; color: var(--cyan) !important}
/* conversation: your messages in soft bubbles on the right, Joshua's as plain text */
body[data-screen^="modern"]:not(#a):not(#b) #chat .message-row + .message-row {border-top: none !important; padding-top: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) .message.bot {background: transparent !important; border: none !important; padding: 4px 6px !important}
body[data-screen^="modern"]:not(#a):not(#b) .message.user {background: var(--userbg) !important; border: none !important; border-radius: 18px !important; padding: 10px 16px !important}
body[data-screen^="modern"]:not(#a):not(#b) .message.user, body[data-screen^="modern"]:not(#a):not(#b) .message.user * {color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) .prose a {color: var(--cyan) !important; text-decoration: underline !important; text-underline-offset: 2px !important}
body[data-screen^="modern"]:not(#a):not(#b) .prose hr {border-top: 1px solid var(--line) !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .placeholder .prose {display: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .placeholder::before {content: "@@WHAT_HELP@@" !important; display: block !important; font-family: 'Newsreader', Georgia, serif !important; font-size: 2rem !important; color: var(--ink) !important; text-align: center !important; margin-top: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .placeholder::after {content: "@@ASK_HINT@@" !important; display: block !important; text-align: center !important; color: var(--dim) !important; font-size: 1rem !important; margin-top: 8px !important}
/* prompt box */
body[data-screen^="modern"]:not(#a):not(#b) #promptrow .block, body[data-screen^="modern"]:not(#a):not(#b) #promptrow .multimodal-textbox {border-radius: 18px !important; background: var(--panel) !important}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow textarea {color: var(--ink) !important; caret-color: var(--cyan) !important; font-size: 1rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #stop {border-radius: 14px !important; font-size: .9rem !important; text-transform: none !important; border-color: var(--line) !important; color: var(--dim) !important}
body[data-screen^="modern"]:not(#a):not(#b) #stop:hover {color: var(--red) !important; border-color: var(--red) !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #model input {text-transform: none !important; letter-spacing: normal !important; font-size: .85rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #working .working {font-size: .88rem !important; color: var(--dim) !important}
body[data-screen^="modern"]:not(#a):not(#b) #working .wdot {width: 8px !important; height: 8px !important; border-radius: 50% !important; background: var(--cyan) !important; animation: pulse 1.4s ease-in-out infinite !important}
@keyframes pulse {0%, 100% {opacity: .25; transform: scale(.8)} 50% {opacity: 1; transform: scale(1)}}
/* teach panel */
body[data-screen^="modern"]:not(#a):not(#b) #tunetitle {font-family: 'Newsreader', Georgia, serif !important; font-size: 1.7rem !important; color: var(--ink) !important; text-transform: none !important}
body[data-screen^="modern"]:not(#a):not(#b) .ftstep > .label-wrap span {font-size: 1.08rem !important; color: var(--ink) !important; font-weight: 600 !important}
body[data-screen^="modern"]:not(#a):not(#b) .ftstep {border-top: 1px solid var(--line) !important}
body[data-screen^="modern"]:not(#a):not(#b) #tune .prose strong {color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) .ftres, body[data-screen^="modern"]:not(#a):not(#b) .ftcount, body[data-screen^="modern"]:not(#a):not(#b) .ftstatus {font-size: .92rem !important; border-radius: 10px !important; border-left: none !important; padding: 10px 14px !important}
body[data-screen^="modern"]:not(#a):not(#b) #tune button {border-radius: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #tune button.primary {background: var(--cyan) !important; color: var(--panel) !important; border-color: var(--cyan) !important}
body[data-screen^="modern"]:not(#a):not(#b) #tuneback {border-radius: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) .ftans {border-radius: 10px !important}
/* fewer lines: spaces are told apart by tone and spacing, not borders */
body[data-screen^="modern"]:not(#a):not(#b) {--block-border-color: transparent !important; --block-border-width: 0px !important; --input-border-color: transparent !important; --border-color-primary: transparent !important; --block-label-border-color: transparent !important; --block-shadow: none !important; --side: #efebe3}
body[data-screen="modern-night"]:not(#a):not(#b) {--side: #23201c}
body[data-screen^="modern"]:not(#a):not(#b) .side, body[data-screen^="modern"]:not(#a):not(#b) #tune {border: none !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol {background: var(--side) !important}
body[data-screen^="modern"]:not(#a):not(#b) .side .block, body[data-screen^="modern"]:not(#a):not(#b) .side .form, body[data-screen^="modern"]:not(#a):not(#b) .side fieldset, body[data-screen^="modern"]:not(#a):not(#b) .side .accordion, body[data-screen^="modern"]:not(#a):not(#b) #tune .block, body[data-screen^="modern"]:not(#a):not(#b) #tune .form {border: none !important; box-shadow: none !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) .opt {border: none !important; padding-bottom: 4px !important}
body[data-screen^="modern"]:not(#a):not(#b) #wfiles, body[data-screen^="modern"]:not(#a):not(#b) #wfiles .wrap {background: var(--codebg) !important; border: none !important; border-radius: 12px !important}
body[data-screen^="modern"]:not(#a):not(#b) #wfiles .label-wrap, body[data-screen^="modern"]:not(#a):not(#b) #wfiles label[data-testid="block-label"] {border: none !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #newchat {border: none !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #newchat:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #tunebtn {border: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #tunebtn:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #screen label, body[data-screen^="modern"]:not(#a):not(#b) #textsize label, body[data-screen^="modern"]:not(#a):not(#b) #kprov label {border-color: transparent !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #screen label:not(.selected):hover, body[data-screen^="modern"]:not(#a):not(#b) #textsize label:not(.selected):hover, body[data-screen^="modern"]:not(#a):not(#b) #kprov label:not(.selected):hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .message.bot, body[data-screen^="modern"]:not(#a):not(#b) #chat .bot, body[data-screen^="modern"]:not(#a):not(#b) #chat .message-row.bot-row .message {border: none !important; box-shadow: none !important; outline: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow .block, body[data-screen^="modern"]:not(#a):not(#b) #promptrow .multimodal-textbox {border: 1px solid var(--line) !important}
body[data-screen^="modern"]:not(#a):not(#b) #stop {border: none !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #tune button:not(.primary) {border-color: var(--line) !important}
body[data-screen^="modern"]:not(#a):not(#b) .ftstep {border: none !important; padding-top: 6px !important}
body[data-screen^="modern"]:not(#a):not(#b) .ftres, body[data-screen^="modern"]:not(#a):not(#b) .ftcount, body[data-screen^="modern"]:not(#a):not(#b) .ftstatus, body[data-screen^="modern"]:not(#a):not(#b) .ftans {border: none !important; background: var(--codebg) !important}
body[data-screen^="modern"]:not(#a):not(#b) #ftlog textarea {border: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #tuneback {border: none !important; background: var(--soft) !important}
@media (prefers-reduced-motion: reduce) {body[data-screen^="modern"]:not(#a):not(#b) #working .wdot {animation: none}}
/* ---- app frame: the page never scrolls; the side column and the messages each scroll in their own place ---- */
html, body {height: 100%; overflow: hidden !important}
.gradio-container {height: 100vh !important; max-height: 100vh; min-height: 0 !important; padding: 0 !important; margin: 0 !important; max-width: none !important; overflow: hidden}
.gradio-container > .main, .gradio-container > .main > .wrap, main.contain, main.contain > .column {height: 100% !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; gap: 0 !important}
main.contain > .column > .row {height: 100%; flex-wrap: nowrap !important; align-items: stretch !important; gap: 12px !important; padding: 10px !important; box-sizing: border-box}
#leftcol {flex: 0 0 290px !important; width: 290px !important; min-width: 0 !important; max-height: 100%; overflow-y: auto !important; overscroll-behavior: contain}
#leftcol, #chatopts, #panels {flex-wrap: nowrap !important}
#leftcol > *, #chatopts > *, #panels > * {flex-shrink: 0 !important}
#chatcol, #tune, #schedpanel {height: 100%; min-width: 0 !important; min-height: 0}
#chatcol {display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important}
#tune, #schedpanel {overflow-y: auto !important}
#chat {flex: 1 1 auto !important; height: auto !important; min-height: 0 !important; max-height: none !important}
#chat > .wrapper {height: 100% !important}
#promptrow, #modelrow {flex: 0 0 auto !important}
#convs {max-height: none !important; overflow: visible !important}
* {scrollbar-width: thin; scrollbar-color: transparent transparent}
*:hover {scrollbar-color: color-mix(in srgb, var(--ink) 25%, transparent) transparent}
::-webkit-scrollbar {width: 8px; height: 8px}
::-webkit-scrollbar-thumb {background: transparent; border-radius: 8px}
*:hover::-webkit-scrollbar-thumb {background: color-mix(in srgb, var(--ink) 22%, transparent)}
#chat .message-buttons, #chat .message-buttons-right, #chat .message-buttons-left {opacity: 0; transition: opacity .15s}
#chat .message-row:hover + .message-buttons, #chat .message-buttons:hover, #chat .message-row:last-of-type ~ .message-buttons {opacity: 1}
#sidetoggle {position: absolute; z-index: 20; top: 10px; left: 10px; width: 32px !important; min-width: 0 !important; height: 32px; padding: 0 !important; font-size: 1rem !important; border: none !important; background: transparent !important; box-shadow: none !important; opacity: .55}
#sidetoggle:hover {opacity: 1; background: color-mix(in srgb, var(--ink) 8%, transparent) !important}
@media (max-width: 760px) {#leftcol {flex-basis: 240px !important; width: 240px !important}}

/* ---- Modern: the conversation in a calm reading column, the composer below it ---- */
body[data-screen^="modern"]:not(#a):not(#b) #chat {background: transparent !important; border: none !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .message-wrap {max-width: 46rem; margin: 0 auto !important; padding: 3rem 1.5rem 2rem !important; width: 100%; box-sizing: border-box}
body[data-screen^="modern"]:not(#a):not(#b) #chat .message-content, body[data-screen^="modern"]:not(#a):not(#b) #chat .message-content * {font-size: 1rem !important; line-height: 1.65 !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .user-row .message {border-radius: 18px !important; padding: .55rem 1rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .placeholder {max-width: 40rem; margin: 0 auto}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow, body[data-screen^="modern"]:not(#a):not(#b) #modelrow {width: 100% !important; max-width: 48rem; margin: 0 auto !important; box-sizing: border-box}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow {padding: 0 .5rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow .multimodal-textbox {border-radius: 22px !important; box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 6%, transparent), 0 4px 18px color-mix(in srgb, var(--ink) 6%, transparent) !important}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow textarea {font-size: 1rem !important; line-height: 1.5 !important; max-height: 40vh !important; overflow-y: auto !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelrow {padding: .35rem 1.25rem .5rem !important; justify-content: flex-start !important; gap: .75rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol {padding: 14px 10px !important; border-radius: 18px !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol .lights {display: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol .status {padding: 0 6px 6px !important; letter-spacing: 0 !important; text-transform: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #newchat, body[data-screen^="modern"]:not(#a):not(#b) #tunebtn, body[data-screen^="modern"]:not(#a):not(#b) #schedbtn, body[data-screen^="modern"]:not(#a):not(#b) #delchat {justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs label {padding: .42rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol {gap: 2px !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol > .html-container, body[data-screen^="modern"]:not(#a):not(#b) #leftcol > .block:has(#brand) {margin-bottom: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #brand {font-size: 1.6rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs {margin: 10px 0 4px !important}
#chat .top-panel {display: none !important}
body[data-screen^="modern"]:not(#a):not(#b) .side .label-wrap .icon {transform: scale(.65); opacity: .4}
body[data-screen^="modern"]:not(#a):not(#b) .side .label-wrap:hover .icon {opacity: .8}
body[data-screen^="modern"]:not(#a):not(#b) #filebtns button {background: transparent !important; border: 1px solid color-mix(in srgb, var(--ink) 12%, transparent) !important; border-radius: 10px !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #filebtns button:hover {background: color-mix(in srgb, var(--ink) 6%, transparent) !important}
body[data-screen^="modern"]:not(#a):not(#b) #chatopts .accordion, body[data-screen^="modern"]:not(#a):not(#b) #panels .accordion, body[data-screen^="modern"]:not(#a):not(#b) #settings {border-top: 1px solid color-mix(in srgb, var(--ink) 8%, transparent) !important; border-radius: 0 !important}

"""

_C = dict(
    body_background_fill="#08101b", body_text_color="#d6ecf5", body_text_color_subdued="#7fa3b5",
    background_fill_primary="#0d1625", background_fill_secondary="#111d30",
    block_background_fill="#0d1625", block_border_color="#24506b", block_label_background_fill="#0d1625", block_label_text_color="#8fe9ff",
    block_title_text_color="#8fe9ff", border_color_primary="#24506b", border_color_accent="#8fe9ff",
    input_background_fill="#0a1220", input_border_color="#24506b",
    color_accent="#ffb347", color_accent_soft="#0d1a2c", link_text_color="#ffb347",
    button_primary_background_fill="#0b2233", button_primary_background_fill_hover="#123247", button_primary_text_color="#8fe9ff", button_primary_border_color="#8fe9ff",
    button_secondary_background_fill="#081222", button_secondary_background_fill_hover="#0d1a2c", button_secondary_text_color="#a6e6ff", button_secondary_border_color="#1d4a66",
    checkbox_background_color="#02060d", checkbox_border_color="#5d8ea6", checkbox_background_color_selected="#ffb347", checkbox_border_color_selected="#ffb347",
    slider_color="#ffb347", table_even_background_fill="#050a14", table_odd_background_fill="#081222",
)
THEME = gr.themes.Base(primary_hue="cyan", neutral_hue="slate", font=gr.themes.GoogleFont("Share Tech Mono"), font_mono=gr.themes.GoogleFont("IBM Plex Mono"), radius_size="sm").set(**_C, **{k + "_dark": v for k, v in _C.items() if k + "_dark" in inspect.signature(gr.themes.Base.set).parameters})

# ---------- Joshua API: lets other programs on this computer (or your network) talk to Joshua ----------
# It speaks the OpenAI chat format that most apps and libraries understand, but every request runs the full
# Joshua agent here: tools, web research, memory, on the model you pick. Settings and key live in api.json.

API_FILE = pathlib.Path(__file__).parent / "api.json"
API_PORT = 7861


def api_settings(**change):
    data = json.loads(API_FILE.read_text()) if API_FILE.exists() else {}
    data.setdefault("key", "jsh-" + secrets.token_urlsafe(24))  # a password for programs; only this computer knows it
    data.setdefault("lan", False)  # False: only programs on this computer can connect
    data.update(change)
    API_FILE.write_text(json.dumps(data))
    API_FILE.chmod(0o600)
    return data


def _api_text(content):
    """OpenAI messages may hold text, or a list of parts (text and images)."""
    if isinstance(content, str):
        return content, []
    text, images = [], []
    for part in content or []:
        if part.get("type") == "text":
            text.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                images.append(url.split(",", 1)[1])  # Ollama takes the picture as base64
    return "\n".join(text), images


def api_answer(body):
    """Runs one Joshua answer for an API request and returns (model, answer)."""
    choice = body.get("model") or "joshua"
    model = resolve(default_model(), False) if choice in ("joshua", "default", "") else choice
    if model not in {m.model for m in ollama.list().models}:
        raise ValueError(f"Unknown model '{choice}'. Use 'joshua' (your default local model) or one from GET /v1/models.")
    cloud = is_cloud(model)
    save_chat = body.get("joshua_save", True)
    cid = new_conv() if save_chat else None
    work = fdir(cid) if cid else ROOT / "_api"
    work.mkdir(parents=True, exist_ok=True)
    agent.set_workspace(work, True, 120, "wt-wt")
    agent.CTX.vision = model if "vision" in caps(model) else FAST
    memory = agent.memory_prompt() if (not cloud or share_memory_setting()) else ""
    msgs = [{"role": "system", "content": SYSTEM.format(now=agent.get_datetime()) + memory}]
    shown = []
    for m in body.get("messages", []):
        text, images = _api_text(m.get("content"))
        if m.get("role") == "system":
            msgs[0]["content"] += "\n\n" + text  # the calling program's own instructions come after Joshua's
        elif m.get("role") in ("user", "assistant"):
            if images and "vision" not in caps(model):  # this model can't see: save the pictures so look_at_image can
                import base64
                names = []
                for b64 in images:
                    f = work / f"api-picture-{secrets.token_hex(3)}.jpg"
                    f.write_bytes(base64.b64decode(b64))
                    names.append(f.name)
                text += f"\n\n(Pictures saved in the workspace: {', '.join(names)})"
                images = []
            msgs.append({"role": m["role"], "content": text, **({"images": images} if images else {})})
            shown.append({"role": m["role"], "content": text})
    toolset = {k: v for k, v in tools.items() if not (cloud and not share_memory_setting() and k in agent.MEMORY_TOOLS)}
    stop = threading.Event()
    for _ in agent_turn(msgs, shown, toolset, model, False, stop, []):
        pass
    answer = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), "")
    if cid:  # the exchange is kept like any other chat, so you can see what programs asked
        conv = load(cid)
        conv.update(title="🔌 " + (next((m["content"] for m in shown if m["role"] == "user"), "API request"))[:46], model=model, llm=msgs, display=shown)
        save(cid, conv)
    return model, answer


def start_api():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    app = FastAPI(title="Joshua API")

    def denied(request):
        key = api_settings()["key"]
        return request.headers.get("authorization", "") != f"Bearer {key}"

    @app.get("/v1/models")
    def list_models(request: Request):
        if denied(request):
            return JSONResponse({"error": {"message": "Missing or wrong API key."}}, 401)
        ids = ["joshua"] + [m.model for m in ollama.list().models if "tools" in caps(m.model)]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "cloud" if is_cloud(i) else "this computer"} for i in ids]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if denied(request):
            return JSONResponse({"error": {"message": "Missing or wrong API key."}}, 401)
        body = await request.json()
        try:
            import anyio
            model, answer = await anyio.to_thread.run_sync(api_answer, body)
        except Exception as e:
            return JSONResponse({"error": {"message": str(e)}}, 400)
        rid, now = "chatcmpl-" + secrets.token_hex(8), int(time.time())
        if body.get("stream"):  # the whole answer in one piece, in the streaming format clients expect
            def events():
                yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]}) + "\n\n"
                yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": now, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return {"id": rid, "object": "chat.completion", "created": now, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}

    host = "0.0.0.0" if api_settings()["lan"] else "127.0.0.1"
    config = uvicorn.Config(app, host=host, port=API_PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()


def lan_address():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent: this only asks which network card would be used
        return s.getsockname()[0]
    except Exception:
        return "this-computer's-address"
    finally:
        s.close()


def api_panel():
    a = api_settings()
    host = lan_address() if a["lan"] else "127.0.0.1"
    url = f"http://{host}:{API_PORT}/v1"
    example = f'''curl {url}/chat/completions \\
  -H "Authorization: Bearer {a["key"]}" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "joshua", "messages": [{{"role": "user", "content": "What is the weather in Turin today?"}}]}}' '''
    return (f"<div class='cloudstat'><b>{t('Address')}</b> <code>{url}</code><br><b>{t('Key')}</b> <code>{a['key']}</code><br>"
            + (t("Other devices on your network can connect (with the key).") if a["lan"] else t("Only programs on this computer can connect."))
            + "</div>"), example


def api_lan(on):
    api_settings(lan=bool(on))
    gr.Info(t("Saved. Restart Joshua for this to take effect (stop.sh, then open Joshua again)."))
    return api_panel()


def api_new_key():
    api_settings(key="jsh-" + secrets.token_urlsafe(24))
    gr.Info(t("New key made: programs using the old one will be refused."))
    return api_panel()


API_HELP = t("""Lets other programs use Joshua: your scripts, a robot, apps that support "OpenAI-compatible" servers. Each request runs the full Joshua (tools, web research, memory) on this computer, and is saved as a 🔌 chat so you can see what programs asked.
Use model `joshua` for your default local model, or any name from `/v1/models`. Cloud models are used only if a program names one.""")


start_api()


# ---------- scheduled tasks: Joshua does something at a set time, on its own ----------
# Tasks live in schedules.json. A background thread checks every 30 seconds; each run becomes a ⏰ chat.
# They run only while Joshua is open and the computer is awake; a run missed meanwhile happens once, at the next check.

import datetime as dt

SCHED_FILE = pathlib.Path(__file__).parent / "schedules.json"
SCHED_LOCK = threading.Lock()
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def tasks_load():
    return json.loads(SCHED_FILE.read_text()) if SCHED_FILE.exists() else []


def tasks_save(tasks):
    SCHED_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=1))


def next_due(t, after):
    """The first time the task is due after the moment 'after'."""
    hh, mm = (int(x) for x in (t.get("time") or "08:00").split(":"))
    if t["when"] == "once":
        at = dt.datetime.fromisoformat(f"{t['date']}T{t['time']}")
        return at if at > after else None
    if t["when"] == "hours":
        return after + dt.timedelta(hours=int(t.get("every", 1)))
    day = after.replace(hour=hh, minute=mm, second=0, microsecond=0)
    for i in range(8):
        cand = day + dt.timedelta(days=i)
        if cand > after and (t["when"] == "daily" or DAYS[cand.weekday()] in t.get("days", [])):
            return cand
    return None


def due_time(t):
    ref = dt.datetime.fromisoformat(t.get("last_run") or t["created"])
    return next_due(t, ref)


def notify(title, text):
    """A system notification: the macOS one, or notify-send on Linux."""
    try:
        if agent.MAC:
            subprocess.run(["osascript", "-e", f"display notification {json.dumps(text)} with title {json.dumps(title)}"], timeout=10)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", title, text], timeout=10)
    except Exception:
        pass


def run_task(t):
    """Starts the task as a new chat, exactly as if you had typed it, and notifies when it's done."""
    model = t.get("model") or default_model()
    if model not in [v for _, v in models()]:
        model = default_model()
    cid = new_conv()  # named first: send() keeps an existing title
    conv = load(cid)
    conv["title"] = f"⏰ {t['name']} · {dt.datetime.now():%d/%m %H:%M}"
    save(cid, conv)
    send({"text": t["prompt"]}, cid, model, False, t.get("goal", False), 10, True, 120, False, 0, True, t.get("offline", False), share_memory_setting())

    def wait_and_notify():
        while busy(cid):
            time.sleep(5)
        if t.get("notify", True):
            answer = last_answer(load(cid)["display"]) or {}
            text = re.sub(r"[*#`\[\]]", "", str(answer.get("content", tr("Done."))))[:180]
            notify(f"Joshua · {t['name']}", text or tr("Done."))
    threading.Thread(target=wait_and_notify, daemon=True).start()


def scheduler():
    while True:
        try:
            now = dt.datetime.now()
            with SCHED_LOCK:
                tasks = tasks_load()
                due = [t for t in tasks if t.get("on", True) and (d := due_time(t)) and d <= now]
                for t in due:
                    t["last_run"] = now.isoformat(timespec="seconds")
                    if t["when"] == "once":
                        t["on"] = False
                if due:
                    tasks_save(tasks)
            for t in due:
                run_task(t)
        except Exception as e:
            print("scheduler:", e)
        time.sleep(30)


def describe(task):
    if task["when"] == "once":
        return t("once, {date} at {time}", date=task["date"], time=task["time"])
    if task["when"] == "daily":
        return t("every day at {time}", time=task["time"])
    if task["when"] == "weekly":
        return t("{days} at {time}", days=", ".join(t(d) for d in task.get("days", [])), time=task["time"])
    return t("every {n} hours", n=task.get("every", 1))


def tasks_view():
    rows = []
    for task in tasks_load():
        nxt = due_time(task) if task.get("on", True) else None
        last = " · " + t("last run {when}", when=f"{dt.datetime.fromisoformat(task['last_run']):%d/%m %H:%M}") if task.get("last_run") else ""
        state = (t("next: {when}", when=f"{t(DAYS[nxt.weekday()])} {nxt:%d/%m %H:%M}") if nxt else t("finished") if task["when"] == "once" else t("paused"))
        rows.append(f"<div class='task'><b>{html.escape(task['name'])}</b><br>{describe(task)}<br><span>{state}{last}</span>"
                    + f"<div class='taskp'>{html.escape(task['prompt'][:160])}</div></div>")
    names = [task["name"] for task in tasks_load()]
    warn = "<div class='cloudstat'>" + t("⚠ Keep Joshua open and the computer awake at those times.") + "</div>" if rows else ""
    return ("".join(rows) + warn or "<div class='cloudstat'>" + t("Your tasks will appear here once you save one.") + "</div>"), gr.Dropdown(choices=names, value=None), gr.Column(visible=bool(names))


def pause_label(name):
    task = next((x for x in tasks_load() if x["name"] == name), None)
    return gr.Button(t("Resume") if task and not task.get("on", True) else t("Pause"))


def task_add(name, prompt, when, date, at, days, every, offline, notify_on, model):
    name, prompt = (name or "").strip(), (prompt or "").strip()
    if not name or not prompt:
        gr.Warning(t("Give the task a name and write what Joshua should do."))
        return gr.skip(), gr.skip(), gr.skip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", at or ""):
        gr.Warning(t("Write the time as HH:MM, for example 07:30."))
        return gr.skip(), gr.skip(), gr.skip()
    at = f"{int(at.split(':')[0]):02d}:{at.split(':')[1]}"
    kind = when  # the radio's values are once / daily / weekly / hours
    if kind == "once":
        try:
            if dt.datetime.fromisoformat(f"{date}T{at}") <= dt.datetime.now():
                gr.Warning(t("That moment has already passed."))
                return gr.skip(), gr.skip(), gr.skip()
        except ValueError:
            gr.Warning(t("Write the date as YYYY-MM-DD, for example 2026-11-20."))
            return gr.skip(), gr.skip(), gr.skip()
    if kind == "weekly" and not days:
        gr.Warning(t("Pick at least one day."))
        return gr.skip(), gr.skip(), gr.skip()
    task = {"name": name, "prompt": prompt, "when": kind, "date": date, "time": at, "days": days, "every": int(every),
            "offline": bool(offline), "notify": bool(notify_on), "model": model or None,
            "created": dt.datetime.now().isoformat(timespec="seconds"), "on": True}
    with SCHED_LOCK:
        tasks = [t for t in tasks_load() if t["name"] != name]  # same name: replaced
        tasks_save(tasks + [task])
    gr.Info(t("Saved: {name}, {when}. It runs only while Joshua is open and the computer is on.", name=name, when=describe(task)))
    return tasks_view()


def task_fill(name):
    """Puts a task's settings in the form, so it can be changed and saved again."""
    t = next((t for t in tasks_load() if t["name"] == name), None)
    if not t:
        return (gr.skip(),) * 10
    when = t["when"]
    return (t["name"], t["prompt"], when, t.get("date") or time.strftime("%Y-%m-%d"), t.get("time", "07:30"), t.get("days", []), t.get("every", 6),
            t.get("offline", False), t.get("notify", True), t.get("model") or default_model())


def task_action(name, action):
    if not name:
        gr.Warning(tr("Pick a task first."))
        return gr.skip(), gr.skip(), gr.skip()
    with SCHED_LOCK:
        tasks = tasks_load()
        t = next((t for t in tasks if t["name"] == name), None)
        if t and action == "delete":
            tasks.remove(t)
        elif t and action == "pause":
            t["on"] = not t.get("on", True)
            if t["on"]:  # resuming: count from now, so it doesn't fire for the paused period
                t["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
        tasks_save(tasks)
    if t and action == "run":
        run_task(t)
        gr.Info(tr("Started {name}: it's in the chat list.", name=name))
    return tasks_view()


threading.Thread(target=scheduler, daemon=True).start()


def css_text(text):
    """A text for a CSS content: "..." rule."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


CSS = CSS.replace("@@WHAT_HELP@@", css_text(t("Hello. What would you like to do today?"))).replace("@@ASK_HINT@@", css_text(t("Everything you write here is stored only on this computer."))).replace("@@MODEL@@", css_text(t("Model:")))
JS_TEXT = lambda text: json.dumps(text, ensure_ascii=False).replace("'", "\\'")  # a text inside the js="..." handlers below


with gr.Blocks(title="Joshua", fill_width=True) as demo:
    cid = gr.State(None)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=270, elem_classes="side", elem_id="leftcol"):
            gr.HTML("<div id='brand'>JOSHUA</div><div class='lights'><span></span><span></span><span></span><span></span><span></span></div><div class='status'>" + t("LOCAL SYSTEM · ONLINE") + "<br>" + t("VERSION") + " " + VERSION + "</div>")
            updnote = gr.HTML("")
            updbtn = gr.Button("⬆  " + t("Update Joshua"), size="sm", visible=False, elem_id="updbtn")
            new = gr.Button("＋  " + t("New chat"), size="sm", elem_id="newchat")
            with gr.Column(elem_id="chatopts") as rightcol:  # the options for this chat; hidden while Teach or Scheduled tasks is open
                with gr.Accordion(t("This chat"), open=True, elem_id="thischat"):
                    thinking = gr.Checkbox(False, label="💭 " + t("Deep thinking"), elem_classes="opt", info=t("Slower, better on hard questions"))
                    goal = gr.Checkbox(False, label="🎯 " + t("Goal mode"), elem_classes="opt", info=t("Give a goal; it keeps working until done"))
                    rounds = gr.Slider(2, 30, value=10, step=1, label=t("Max rounds"), visible=False)
                    swarm = gr.Checkbox(False, label="🐝 " + t("Swarm"), elem_classes="opt", info=t("Several agents answer in parallel, then one combines the best answer"))
                    agents = gr.Slider(2, 6, value=5, step=1, label=t("Agents"), visible=False)
                    offline = gr.Checkbox(False, label="🚫 " + t("No internet"), elem_classes="opt", info=t("Joshua doesn't search or read the web, and code can't go online"))
                    files = gr.File(file_count="multiple", label=t("Files in this chat"), elem_id="wfiles", height=None)
                    with gr.Row(elem_id="filebtns"):
                        openf = gr.Button("📂 " + t("Open folder"), size="sm")
                        dlall = gr.DownloadButton("⬇ " + t("Download all (.zip)"), size="sm", visible=False)
            convs = gr.Radio(choices=conv_choices(), show_label=False, container=False, elem_id="convs")
            renbtn = gr.Button("✏  " + t("Rename this chat"), size="sm", elem_id="renchat")
            renbox = gr.Textbox(visible=False, show_label=False, max_lines=1, placeholder=t("New name, then press Enter"), elem_id="renbox")
            dele = gr.Button("🗑  " + t("Delete this chat"), size="sm", elem_id="delchat")
            sched = gr.Button("⏰  " + t("Scheduled tasks"), size="sm", elem_id="schedbtn")
            feedback = gr.Button("💬  " + t("Send feedback"), size="sm", elem_id="fbbtn")
            with gr.Column(elem_id="panels"):
                with gr.Accordion("🧠 " + t("Memory"), open=False) as mem_box:
                    mem = gr.Checkbox(True, label=t("Use memory"), info=t("Joshua learns lasting facts about you and uses them in every chat"))
                    mem_topic = gr.Dropdown(choices=["profile"], value="profile", label=t("Notes"), elem_id="memtopic")
                    mem_text = gr.Textbox(lines=8, max_lines=20, show_label=False, placeholder=t("Nothing remembered yet. Joshua adds notes as you chat, or you can type them here, one per line."))
                    mem_btn = gr.Button(t("Save notes"), size="sm")
                    cloudmem = gr.Checkbox(share_memory_setting(), label=t("Share notes with cloud models"), info=t("Off: cloud models never see your memory notes (Joshua still learns, on this computer)"))
                tune = gr.Button("🧪  " + t("Teach Joshua (fine-tune)"), size="sm", elem_id="tunebtn")
                with gr.Accordion("☁ " + t("Cloud models"), open=False, elem_id="cloudbox") as cloud_box:
                    gr.Markdown(CLOUD_HELP, elem_id="cloudhelp")
                    cloud_status = gr.HTML("")
                    connect = gr.Button(t("Connect this computer to Ollama"), size="sm")
                    gr.HTML("<div class='cloudlinks'><a href='https://ollama.com/dashboard' target='_blank' rel='noopener'>💳 " + t("Add credit on ollama.com") + "</a>"
                            "<a href='https://ollama.com/settings' target='_blank' rel='noopener'>" + t("See what you've used") + "</a></div>"
                            "<div class='cloudnote'>" + t("Credit is paid to Ollama for the computing power of its servers, which run the cloud models. Joshua itself is free: none of this money goes to Joshua, and local models never need credit.") + "</div>")
                    cloud_pick = gr.Dropdown([], label=t("Add a cloud model"), info=t("All of these can use Joshua's tools; 👁 ones can also see images"))
                    cloud_add = gr.Button(t("Add to the model menu"), size="sm")
                with gr.Accordion(providers.PANEL, open=False, elem_id="keysbox") as keys_box:
                    gr.Markdown(KEYS_HELP, elem_id="keyshelp")
                    k_provider = gr.Radio([(v["name"], k) for k, v in providers.PROVIDERS.items()], value="anthropic", show_label=False, elem_id="kprov")
                    k_key = gr.Textbox(type="password", label=t("API key"), placeholder=t("Paste the key here"), max_lines=1)
                    k_save = gr.Button(t("Save the key"), size="sm")
                    k_status = gr.HTML("")
                    k_pick = gr.Dropdown([], label=t("Add a model"))
                    k_add = gr.Button(t("Add to the model menu"), size="sm")
                with gr.Accordion("⬇ " + t("Local models"), open=False) as local_box:
                    gr.Markdown(t("Models that run on this computer: private and free, as fast as your hardware allows. The list comes from ollama.com and shows only versions that fit this computer's memory."), elem_id="localhelp")
                    local_status = gr.HTML("")
                    local_pick = gr.Dropdown([], label=t("Download a model"))
                    local_get = gr.Button(t("Download"), size="sm")
                    local_upd = gr.Button(t("Update my models"), size="sm")
                with gr.Accordion("🔌 API", open=False) as api_box:
                    gr.Markdown(API_HELP, elem_id="apihelp")
                    api_info = gr.HTML("")
                    api_example = gr.Code(language="shell", label=t("Try it in Terminal"), interactive=False)
                    api_lan_box = gr.Checkbox(api_settings()["lan"], label=t("Allow other devices on my network"), info=t("Off: only programs on this computer. Needs a restart of Joshua."))
                    api_key_btn = gr.Button(t("Make a new key"), size="sm")
                with gr.Accordion(t("Project & code settings"), open=False):
                    project = gr.Textbox(label=t("Project folder (optional)"), placeholder=t("e.g. ~/Projects/my-project"), info=t("The chat works directly in this folder. Press Enter to apply."), max_lines=1)
                    net = gr.Checkbox(True, label="🌐 " + t("Internet for code"), info=t("Lets code go online and install packages"))
                    timeout = gr.Slider(10, 600, value=120, step=10, label=t("Code time limit (seconds)"))
                with gr.Accordion(t("Tools"), open=False):
                    gr.Markdown("<div id='tools'>🔎 " + t("Web search") + "<br>📄 " + t("Read web pages") + "<br>🐍 " + t("Run Python (sandboxed)") + "<br>📦 " + t("Install packages") + "<br>🗂️ " + t("Browse folders") + "<br>📂 " + t("Read & 💾 write files (text, Word, PDF)") + "<br>🕒 " + t("Date and time") + "<br>🖼️ " + t("Look at images") + "<br>" + ("🎨 " + t("Generate images") + "<br>🖌️ " + t("Edit your photos") + "<br>" if "edit_image" in tools else "") + "🧠 " + t("Remember & forget") + "</div>")
            with gr.Accordion("⚙ " + t("Settings"), open=False, elem_id="settings"):
                screen = gr.Radio([(t("Novel"), "novel"), (t("Black"), "black"), (t("White"), "white"), (t("Green"), "green"), (t("Amber"), "amber"), (t("Blue"), "blue"), (t("Modern"), "modern"), (t("Modern night"), "modern-night")], value="modern", label=t("Look"), elem_id="screen")
                textsize = gr.Radio([("90%", "0.9"), ("100%", "1"), ("110%", "1.1"), ("125%", "1.25"), ("140%", "1.4")], value="1", label=t("Text size"), elem_id="textsize")
                language = gr.Dropdown([(t("Language: automatic"), "auto")] + [(name, code) for code, name in LANGS.items()], value=i18n.chosen() if i18n.chosen() in LANGS else "auto", label=t("Language"), elem_id="language")
        tune_panel, tune_back, tune_res = tune_ui.build(ROOT, conv_choices)
        with gr.Column(visible=False, scale=5, elem_id="schedpanel") as sched_panel:
            with gr.Row():
                gr.HTML("<div id='schedtitle'>" + t("SCHEDULED TASKS") + "</div>")
                sched_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_id="schedback")
            gr.Markdown(t("Joshua does a task on its own at the time you choose: a morning news summary, a weekly check of a website, a reminder written for you. Each run appears as a ⏰ chat in the list on the left.") + "\n\n" + t("**Only while Joshua is running.** Tasks can't run when the computer is off or asleep, or when Joshua is closed. A task missed in the meantime runs once, as soon as Joshua is open again."), elem_id="schedhelp")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_id="schedform"):
                    gr.HTML("<div class='schedsub'>" + t("NEW TASK, OR CHANGE ONE") + "</div>")
                    t_name = gr.Textbox(label=t("Name"), placeholder=t("Morning news"), max_lines=1)
                    t_prompt = gr.Textbox(label=t("What Joshua should do"), lines=5, placeholder=t("Find the three most important news stories in Piedmont today and summarise each in two lines, with sources."))
                    t_when = gr.Radio([(t("Once"), "once"), (t("Every day"), "daily"), (t("On some days"), "weekly"), (t("Every few hours"), "hours")], value="daily", label=t("When"))
                    with gr.Row():
                        t_date = gr.Textbox(label=t("Date (YYYY-MM-DD)"), value=time.strftime("%Y-%m-%d"), max_lines=1, visible=False)
                        t_time = gr.Textbox(label=t("Time (HH:MM)"), value="07:30", placeholder="07:30", max_lines=1)
                        t_every = gr.Slider(1, 24, value=6, step=1, label=t("Every how many hours"), visible=False)
                    t_days = gr.CheckboxGroup([(t(d), d) for d in DAYS], value=["Mon", "Tue", "Wed", "Thu", "Fri"], label=t("Days"), visible=False)
                    with gr.Row():
                        t_model = gr.Dropdown(choices=models(), value=default_model(), label=t("Model"))
                        with gr.Column():
                            t_offline = gr.Checkbox(False, label="🚫 " + t("No internet for this task"))
                            t_notify = gr.Checkbox(True, label="🔔 " + t("Notify me when it's done"))
                    t_add = gr.Button(t("Save task"), variant="primary")
                with gr.Column(scale=2, elem_id="schedlist"):
                    gr.HTML("<div class='schedsub'>" + t("YOUR TASKS") + "</div>")
                    task_list = gr.HTML("")
                    with gr.Column(visible=False) as t_controls:
                        t_pick = gr.Dropdown([], label=t("Choose a task"))
                        with gr.Row():
                            t_run = gr.Button("▶ " + t("Run now"), size="sm")
                            t_pause = gr.Button(t("Pause"), size="sm")
                        t_del = gr.Button(t("Delete this task"), size="sm", elem_id="schedel")
        with gr.Column(scale=4, elem_id="chatcol", elem_classes="chat-panel") as chatcol:
            sidetoggle = gr.Button("☰", size="sm", elem_id="sidetoggle", min_width=0)
            chatbot = gr.Chatbot(height="100%", show_label=False, elem_id="chat", placeholder="### " + t("Hello. What would you like to do today?") + "\n" + t("Everything you write here is stored only on this computer."))
            with gr.Row(equal_height=True, elem_id="promptrow"):
                box = gr.MultimodalTextbox(file_count="multiple", placeholder=t("Message Joshua, or drop files here…"), show_label=False, scale=12)
                stop = gr.Button("■ " + t("Stop"), scale=1, min_width=64, elem_id="stop")
            with gr.Row(elem_id="modelrow"):
                working = gr.HTML("", elem_id="working")
                cloudbadge = gr.HTML("", elem_id="cloudbadge")
                model = gr.Dropdown(choices=models(True), value=default_model(), show_label=False, container=False, elem_id="model", scale=0, min_width=200)

    box.submit(send, [box, cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem], [chatbot, box, cid, convs, files], concurrency_limit=None)
    model.change(badge, model, cloudbadge)

    # "＋ Add a model…" in the menu: open the two panels on the right, fill their lists, then put the menu back on the model you had
    last_model, adding = gr.State(default_model()), gr.State(False)

    def pick_model(choice, last):
        if choice != ADD:
            return gr.skip(), gr.skip(), choice, False
        gr.Info(t("Pick a model in ☁ Cloud models or ⬇ Local models, on the right."))
        return gr.Accordion(open=True), gr.Accordion(open=True), last, True

    scroll_js = "(c, l) => { if (c === '+add') setTimeout(() => document.getElementById('cloudbox')?.scrollIntoView({behavior: 'smooth', block: 'start'}), 300); return [c, l]; }"
    (model.change(pick_model, [model, last_model], [cloud_box, local_box, last_model, adding], js=scroll_js)
     .then(lambda a: cloud_catalog() if a else (gr.skip(), gr.skip()), adding, [cloud_pick, cloud_status])
     .then(lambda a: local_catalog() if a else (gr.skip(), gr.skip()), adding, [local_pick, local_status])
     .then(lambda a, last: (gr.Dropdown(value=last), False) if a else (gr.skip(), False), [adding, last_model], [model, adding]))
    cloudmem.change(share_memory_setting, cloudmem)
    cloud_box.expand(cloud_catalog, outputs=[cloud_pick, cloud_status])
    local_box.expand(local_catalog, outputs=[local_pick, local_status])
    sched.click(lambda: (gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False), gr.Dropdown(choices=models())), outputs=[sched_panel, chatcol, rightcol, tune_panel, t_model]).then(tasks_view, outputs=[task_list, t_pick, t_controls])
    sched_back.click(lambda: (gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=True), gr.Radio(choices=conv_choices())), outputs=[sched_panel, chatcol, rightcol, convs])
    t_pick.input(task_fill, t_pick, [t_name, t_prompt, t_when, t_date, t_time, t_days, t_every, t_offline, t_notify, t_model]).then(pause_label, t_pick, t_pause)
    t_when.change(lambda w: (gr.Textbox(visible=w == "once"), gr.Textbox(visible=w != "hours"), gr.CheckboxGroup(visible=w == "weekly"), gr.Slider(visible=w == "hours")), t_when, [t_date, t_time, t_days, t_every])
    t_add.click(task_add, [t_name, t_prompt, t_when, t_date, t_time, t_days, t_every, t_offline, t_notify, t_model], [task_list, t_pick, t_controls])
    t_run.click(lambda n: task_action(n, "run"), t_pick, [task_list, t_pick, t_controls])
    t_pause.click(lambda n: task_action(n, "pause"), t_pick, [task_list, t_pick, t_controls])
    t_del.click(lambda n: task_action(n, "delete"), t_pick, [task_list, t_pick, t_controls], js="(n) => (n && confirm(" + JS_TEXT(t("Delete '{name}'? It won't run again.")) + ".replace('{name}', n))) ? n : null")
    api_box.expand(api_panel, outputs=[api_info, api_example])
    api_lan_box.input(api_lan, api_lan_box, [api_info, api_example])
    api_key_btn.click(api_new_key, outputs=[api_info, api_example])
    local_get.click(local_download, local_pick, [model, local_status])
    local_upd.click(local_update, outputs=local_status)
    demo.load(check_update, outputs=[updnote, updbtn])
    updbtn.click(do_update, outputs=updnote)
    connect.click(cloud_connect, outputs=cloud_status)
    cloud_add.click(cloud_add_model, cloud_pick, [model, cloud_status])
    keys_box.expand(keys_status, k_provider, [k_pick, k_status])
    k_provider.change(keys_status, k_provider, [k_pick, k_status])
    k_save.click(keys_save, [k_provider, k_key], [k_pick, k_status, k_key])
    k_add.click(keys_add, [k_provider, k_pick], [model, k_status])
    mem_box.expand(mem_view, mem_topic, [mem_topic, mem_text])
    mem_topic.input(mem_view, mem_topic, [mem_topic, mem_text])
    mem_btn.click(mem_save, [mem_topic, mem_text], [mem_topic, mem_text],
                  js="(topic, text) => (text.trim() || confirm(" + JS_TEXT(t("Clear all the notes in '{topic}'? This can't be undone.")) + ".replace('{topic}', topic))) ? [topic, text] : [topic, null]")
    swarm.change(lambda s: (gr.Slider(visible=s), gr.Checkbox(value=False) if s else gr.skip()), swarm, [agents, goal])
    stop.click(stop_chat, cid, concurrency_limit=None)
    goal.change(lambda g: (gr.Slider(visible=g), gr.MultimodalTextbox(placeholder=t("Describe a goal for Joshua to accomplish…") if g else t("Message Joshua, or drop files here…")), gr.Checkbox(value=False) if g else gr.skip()), goal, [rounds, box, swarm])
    new.click(new_chat, outputs=[chatbot, cid, convs, files, project])
    convs.input(open_conv, convs, [chatbot, cid, files, project, model]).then(badge, model, cloudbadge)
    project.submit(set_project, [project, cid], [files, cid, convs, project])
    files.upload(upload_files, [files, cid], [files, cid, convs])
    files.change(zip_files, cid, dlall, show_progress="hidden")
    openf.click(open_folder, cid)
    chatbot.retry(retry, [cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem], [chatbot, box, cid, convs, files])
    files.delete(delete_file, cid, files)
    gr.Timer(1).tick(poll, cid, [chatbot, convs, files, working], show_progress="hidden", concurrency_limit=None)
    feedback.click(None, model, None, js="(m) => { window.open(" + json.dumps(FEEDBACK) + " + '&model=' + encodeURIComponent(m || ''), '_blank') }")
    renbtn.click(rename_open, cid, renbox)
    renbox.submit(rename_save, [cid, renbox], [renbox, convs])
    dele.click(delete_chat, convs, [chatbot, cid, convs, files, project], js="(c) => (c && confirm(" + JS_TEXT(t("Move this chat and its files to the Trash? You can restore them from there.")) + ")) ? c : null")
    demo.load(lambda: (gr.Radio(choices=conv_choices()), gr.Dropdown(choices=models(True), value=default_model())), outputs=[convs, model])
    tune.click(lambda: (gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False), "<div class='ftres'>" + t("Checking what your computer can do…") + "</div>"), outputs=[tune_panel, chatcol, rightcol, sched_panel, tune_res]).then(tune_ui.res_html, outputs=tune_res, show_progress="hidden")
    tune_back.click(lambda: (gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=True), gr.Dropdown(choices=models(True))), outputs=[tune_panel, chatcol, rightcol, model])
    reloading = gr.State(False)
    language.input(set_language, language, reloading).then(None, reloading, None, js="(r) => { if (r) setTimeout(() => location.reload(), 8000) }")
    # Gradio's own few labels follow the browser's language, not Joshua's: swap them for Joshua's translation as they appear
    gradio_texts = {k: t(k) for k in ("Drop File Here", "- or -", "Click to Upload", "Info", "Warning", "Error", "Clear")}
    chat_files = {"Drop File Here": t("Add files to this chat"), "- or -": "", "Click to Upload": ""}  # the file area in the right column
    if True:
        demo.load(None, None, None, js="""() => { const T = """ + json.dumps(gradio_texts, ensure_ascii=False) + """, F = """ + json.dumps(chat_files, ensure_ascii=False) + """;
          const fix = (root) => { const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n;
            while ((n = w.nextNode())) { const k = n.nodeValue.trim(), M = n.parentElement && n.parentElement.closest('#wfiles') ? F : T;
              if (k in M && M[k] !== k) n.nodeValue = n.nodeValue.replace(k, M[k]); } };
          fix(document.body); new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(x => x.nodeType === 1 ? fix(x) : x.nodeType === 3 && fix(x.parentNode || document.body)))).observe(document.body, {childList: true, subtree: true}); }""")
    screen.change(None, screen, None, js="(v) => { document.body.dataset.screen = v; try { localStorage.setItem('joshua-screen2', v) } catch (e) {} }")
    demo.load(None, None, screen, js="() => { let v = 'modern'; try { v = localStorage.getItem('joshua-screen2') || 'modern' } catch (e) {} document.body.dataset.screen = v; return v }")
    textsize.change(None, textsize, None, js="(v) => { document.documentElement.style.fontSize = (16 * v) + 'px'; try { localStorage.setItem('joshua-zoom', v) } catch (e) {} }")
    demo.load(None, None, textsize, js="() => { let v = '1'; try { v = localStorage.getItem('joshua-zoom') || '1' } catch (e) {} document.documentElement.style.fontSize = (16 * v) + 'px'; return v }")
    sidetoggle.click(None, None, None, js="() => { const h = document.body.classList.toggle('noside'); try { localStorage.setItem('joshua-noside', h ? '1' : '') } catch (e) {} }")
    demo.load(None, None, None, js="() => { try { if (localStorage.getItem('joshua-noside')) document.body.classList.add('noside') } catch (e) {} }")

ICON = (pathlib.Path(__file__).parent / "icon.png").resolve()
APP_HEAD = (f"<link rel='apple-touch-icon' href='/gradio_api/file={ICON}'>"  # the icon Safari uses for "Add to Dock"
            "<meta name='apple-mobile-web-app-title' content='Joshua'><meta name='application-name' content='Joshua'>")
SERVICE = os.environ.get("JOSHUA_SERVICE")  # set when Joshua runs in the background from login (launchd / systemd)
demo.launch(inbrowser=not (SERVICE or os.environ.get("JOSHUA_NO_OPEN")), favicon_path=str(ICON), allowed_paths=[str(ROOT), str(ICON), str(pathlib.Path.home())], css=CSS, theme=THEME, head=APP_HEAD + "<link href='https://fonts.googleapis.com/css2?family=VT323&family=Figtree:ital,wght@0,400;0,500;0,600;1,400&family=Newsreader:opsz,wght@6..72,500&family=Share+Tech+Mono&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap' rel='stylesheet'>")
