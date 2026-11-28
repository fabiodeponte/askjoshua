import functools, os, sys
os.environ.setdefault("GRADIO_RUN_HISTORY", "false")  # see LAUNCH_EXTRA
import jlog
jlog.setup()  # the log: AskJoshua/joshua.log, every line with its time
import copy, hashlib, html, inspect, json, os, pathlib, platform, re, secrets, shutil, subprocess, sys, threading, time, zipfile, urllib.parse
from concurrent.futures import ThreadPoolExecutor, wait
from itertools import zip_longest
import gradio as gr, ollama
import agent, apple, browser, hfcache, desks, podcast_ui, factcheck, falken, monitor, providers, ranking, reader, remote, scratch, teams, tune_ui
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
LOGO = "<g transform='translate(-4 3)'><path d='M54 44 V64 A15 15 0 0 1 24 64' stroke='#3f8a49' stroke-width='9' fill='none' stroke-linecap='round'/><path d='M54 44 C47 22 64 3 94 3 C96 28 78 49 54 44 Z' fill='#7fb86a'/><path d='M58 40 L86 11 M65 33 V20 M72 26 H84' stroke='#f3efdc' stroke-width='2.2' fill='none' stroke-linecap='round'/><circle cx='86' cy='11' r='2.8' fill='#f3efdc'/><circle cx='65' cy='20' r='2.8' fill='#f3efdc'/><circle cx='84' cy='26' r='2.8' fill='#f3efdc'/><circle cx='39' cy='64' r='8.5' fill='#2a6bb5'/><path d='M34.5 59.5 C37.5 57.5 41.5 58.5 41.5 61.5 C41.5 64.5 37.5 64.5 35.5 66.5 C33.5 64.5 32.5 61.5 34.5 59.5 Z' fill='#7fb86a'/></g>"  # the sprout J, the same drawing as the website's img/logo.svg
ICONS = {"web_search": "🔎", "read_page": "📄", "run_python": "🐍", "read_file": "📂", "write_file": "💾", "get_datetime": "🕒", "look_at_image": "🖼️", "generate_image": "🎨", "edit_image": "🖌️", "goal_done": "✅", "give_up": "🛑", "ask_member": "👥", "install_package": "📦", "list_files": "🗂️", "remember": "🧠", "forget": "🧹", "read_memory": "📒", "browser_open": "🧭", "browser_read": "🧭", "browser_click": "🖱️", "browser_type": "⌨️", "browser_scroll": "🧭", "browser_back": "🧭"}
SKIP = (gr.skip(),) * 4

IMAGE_RULE = " When the user asks to change, combine or add to photos they shared, use edit_image with those photo files and never answer with an invented picture; use generate_image only for brand-new pictures." if "edit_image" in tools else " You cannot create or edit pictures on this computer: say so if asked."
SYSTEM = "You are Joshua, an AI assistant running on the user's computer; you have no job or personal life of your own. Today is {now}." + IMAGE_RULE + " Results returned by your tools are real and current. For any factual question (figures, people, events, prices, anything that may change), always use web_search before answering; never answer such questions from memory. Search snippets are often incomplete: if they don't clearly contain the answer, open the most relevant result with read_page before answering, and never fill gaps with guesses. Always address the user directly, in the second person, in the user's language. Files the user shares are in your workspace folder."
GOAL_RULES = " You are in goal mode: the user gives you a goal to accomplish, not a question. Work step by step with your tools until the goal is fully achieved, and check your own results (for example, read back files you wrote). Do not ask the user questions; make reasonable assumptions. When the goal is achieved and verified, call goal_done with a short summary. Only if it is truly impossible with your tools, call give_up with the reason."
WRAP = "That's enough research: stop using tools now and write your complete final answer for me with what you have, saying what you couldn't check."
CHECK_GAPS = "Not done yet: checking your fact-check table found these gaps:\n{gaps}\nFix them (search and open sources as needed), then write the complete table again and call goal_done."
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
    return providers.is_api(name) or name in CLOUD or name.removesuffix(":latest").split(":")[-1].endswith("cloud")


PAIR = "auto:qwen3-vl"  # one menu entry: the deep-thinking switch picks qwen3-vl or its faster instruct version


ADD = "+add"  # last entry of the model menu: opens the panels where you add cloud or local models


def made_here(n, plain=None):
    """A model made by you, marked ★: taught on the Fine-tune page (joshua-…) or trained from zero and added to Joshua."""
    return n.startswith("joshua-") or n.removesuffix(":latest") in (scratch.in_joshua() if plain is None else plain)


def rename_model(old, new):
    """A new name for a model made by you. Ollama copies it under the new name (the weights are shared) and drops the
    old; the menu keeps its place. A model taught on the Fine-tune page keeps its joshua- prefix, which is its ★."""
    new, base = (new or "").strip(), old.removesuffix(":latest")
    if base in scratch.in_joshua():
        scratch.rename(base, new)
    else:
        new = new if new.startswith("joshua-") else "joshua-" + new
        if not scratch.NAME_OK.match(new):
            raise RuntimeError(t("Use letters, digits, dots, dashes or underscores for the name, like Wikipedia-Italian."))
        if new.lower() in {m.model.removesuffix(":latest").lower() for m in ollama.list().models} - {old.removesuffix(":latest").lower()}:
            raise RuntimeError(t("There is already a model called {name}.", name=new))
        scratch.move(old, new)  # through a temporary name when only the capitals change
    order = model_order()
    if old in order:
        order[order.index(old)] = new + ":latest"
        model_order(order)
    CAPS.pop(old, None)
    return new


def models(add=False):
    try:
        listed = ollama.list().models
    except Exception:
        listed = []
    CLOUD.update(m.model for m in listed if getattr(m, "remote_host", None) or m.model.split(":")[-1].endswith("cloud"))
    plain = scratch.in_joshua()  # models trained here from zero: no tools, but you can still talk to them
    names = [n for n in (m.model for m in listed) if "tools" in caps(n) or n.removesuffix(":latest") in plain]
    label = lambda n: "☁ " + n.removesuffix(":latest").removesuffix(":cloud").removesuffix("-cloud") if is_cloud(n) else "★ " + n.removeprefix("joshua-").removesuffix(":latest") if made_here(n, plain) else n.removesuffix(":latest")
    pair = FAST in names and (MODEL in names or MODEL + ":latest" in names)  # both halves of Joshua's model are here
    choices = [(label(n), n) for n in names if not (pair and n in (FAST, MODEL, MODEL + ":latest"))]
    if pair:
        own = agent.CONFIG.get("label", "qwen3-vl")
        choices.insert(0, ("☁ " + own.removesuffix(":cloud").removesuffix("-cloud") if is_cloud(FAST) else own, PAIR))  # Joshua's model can be a cloud one, chosen at install
    if not apple.why_not():  # macOS 26 with Apple Intelligence on: free, and already on this Mac
        choices.append((apple.LABEL, apple.NAME))
    # the order: on this computer, then open-weight models in the cloud, then closed ones (Claude, ChatGPT, Gemini)
    abc = lambda group: sorted(group, key=lambda c: re.sub(r"^\W+", "", c[0]).lower())  # alphabetical, ignoring ☁ ★ 🔑 🍎
    cl = lambda v: is_cloud(FAST) if v == PAIR else is_cloud(v)
    local = abc(c for c in choices if not cl(c[1]))
    cloud_open = abc([c for c in choices if cl(c[1])] + providers.menu(open_weights=True))
    cloud_open.sort(key=lambda c: not re.sub(r"^\W+", "", c[0]).lower().startswith("nemotron"))  # Nemotron first: the cheapest of the big ones
    choices = local + cloud_open + abc(providers.menu(open_weights=False))
    mine = model_order()  # the order you set on the Models page; models you haven't placed keep theirs, after
    choices.sort(key=lambda c: mine.index(c[1]) if c[1] in mine else len(mine))
    return (choices or [(FAST, FAST)]) + ([(t("＋ Add a model…"), ADD)] if add else [])


def model_order(value=None):
    return _settings("model_order", [], value)


def default_model():
    if model_order():  # you chose an order: the first is the default
        return models()[0][1]
    local = [v for _, v in models() if not is_cloud(v)]
    if PAIR in local:  # Joshua's own model stays the starting one, wherever the alphabet puts it
        return PAIR
    qwen = [v for v in local if v.lower().startswith("qwen")]
    if qwen:  # otherwise a Qwen model, when one is installed: the newest version first (qwen3.5 before qwen3)
        return sorted(qwen, key=lambda v: [int(x) if x.isdigit() else 0 for x in re.findall(r"\d+", v.split(":")[0])], reverse=True)[0]
    local.sort(key=lambda v: v == apple.NAME)  # Apple Intelligence only if there's nothing else here
    if local:
        return local[0]  # start on a local model; cloud is always a choice you make
    cloud = [v for _, v in models()]
    nemotron = [v for v in cloud if v.startswith("nemotron")]  # nothing local: Nemotron is the cloud model Joshua starts with
    return PAIR if PAIR in cloud else (nemotron or cloud)[0]


CLOUD_BADGE = "<div class='cloudbadge' title='" + html.escape(t("☁ Cloud model: messages go to Ollama's servers; everything stays stored here"), quote=True) + "'>☁ " + t("on Ollama's servers") + "</div>"


def money(x):
    return f"${x:.3f}".rstrip("0").rstrip(".") if x < 0.1 else f"${x:.2f}"


def badge(choice):
    """Under the message box: where the chosen model runs, what it can do (icons) and what it costs."""
    if not choice or choice == ADD:
        return ""
    tip = lambda icon, text: f"<span class='bicon' data-tip='{html.escape(text, quote=True)}' aria-label='{html.escape(text, quote=True)}'>{icon}</span>"  # data-tip: an explanation that shows at once on hover
    left = None  # dollars left on the Ollama account: shown for cloud models when you gave Joshua an API key
    if providers.is_api(choice):
        company = providers.company(choice)
        full = t("🔑 {company}'s model: messages go to {company}'s servers; everything stays stored here", company=company)
        place = "<span class='bplace' data-tip='" + html.escape(full, quote=True) + "'>🔑 " + html.escape(company) + "</span>"
        vision, audio, thinks, price = providers.sees_images(choice), False, False, ranking.closed_price(choice.split(" ", 1)[-1])
    elif choice == apple.NAME:
        place, vision, audio, thinks, price = "", False, False, False, None
    else:
        m = resolve(choice, True)
        c = caps(m)
        vision, audio, thinks = "vision" in c, "audio" in c, "thinking" in c
        place = ""  # the menu already shows ☁ before a cloud model's name
        price = ranking.cloud_price(m) if is_cloud(m) else None
        left = credit_left() if is_cloud(m) else None
    icons = tip("📄", t("Reads and writes text"))
    icons += tip("🖼️", t("Sees images: you can attach photos, screenshots and scanned pages")) if vision else ""
    icons += tip("🎧", t("Hears audio: you can attach recordings")) if audio else ""
    icons += tip("💭", t("Thinks before answering: slower, but better at hard questions")) if thinks else ""
    if price:
        cost = "<span class='bcost' data-tip='" + html.escape((t("Dollars per million tokens: first what you send (input), then what it writes (output). Paid to {company}.", company=providers.company(choice)) if providers.is_api(choice) else t("Dollars per million tokens: first what you send (input), then what it writes (output). Paid to Ollama, not to Joshua.")), quote=True) + "'>" + t("{inp} / {out} per M", inp=money(price[0]), out=money(price[2])) + "</span>"
    elif place or (choice != apple.NAME and not providers.is_api(choice) and is_cloud(resolve(choice, True))):
        cost = ""
    else:
        cost = "<span class='bcost'>" + t("free, local") + "</span>"
    credit = ("<span class='bcredit' data-tip='" + html.escape(t("What is left on your Ollama account: the credit included in your plan plus the credit you bought."), quote=True) + "'>" + t("{n} left", n=money(left)) + "</span>") if left is not None else ""
    return "<div class='cloudbadge'>" + "".join(x for x in (place, "<span class='bicons'>" + icons + "</span>", cost, credit) if x) + "</div>"


CLOUD_HELP = ("**" + t("Stays here:") + "** " + t("your chats, files and memory notes.") + "  \n"
              + "**" + t("Goes to Ollama:") + "** " + t("the conversation, with the files and photos in it.") + "  \n"
              + "**" + t("Cost:") + "** " + t("pay per use, with credit on your Ollama account.") + " " + t("Every message sends the whole chat again as input, so long chats cost more; cached input is the part already sent before.") + "\n\n"
              + "<details><summary>" + t("Read more") + "</summary>\n\n" + t("""Bigger models running on Ollama's servers, for when this computer is too small or slow.
**What goes to the cloud:** for every message, the conversation so far, including the text of files and web pages Joshua reads, and your photos if you send them.
**What stays here:** your chats, files, memory notes and trained models are stored only on this computer. Memory notes are sent only if you allow it in 🧠 Memory.
Ollama says it never logs or trains on cloud prompts. That is a promise you are trusting, not something this computer can check.
**Cost:** new Ollama accounts pay per use: add a few dollars of credit on ollama.com (Gemma 4 31B costs well under a cent per message). Some older accounts still have a small free monthly allowance on a few models (marked "free plan" for those accounts). Local models are always free.""") + "\n\n</details>")


def ollama_bin():
    import web_ollama
    return web_ollama.path()


def gb_text(nbytes):
    """A size people can read: 7.6 GB, or 24 MB for the small models made here (not 0.0 GB)."""
    return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{max(1, round(nbytes / 1e6))} MB"


def media(vision, audio=False, thinking=False):
    """What a model can do, said plainly: what it reads, and whether it can think at length before answering."""
    reads = t("text + images + audio") if vision and audio else t("text + images") if vision else t("text + audio") if audio else t("text only")
    return reads + (" · " + t("deep thinking") if thinking else "")


THINKERS = set()  # models on ollama.com labelled "thinking", seen while reading the library


def act_button(action, tag, label):
    """A button in a table row; the page script hands its action to row_action."""
    act = html.escape(action + ":" + tag, quote=True)
    danger = " danger" if action.startswith("rm") else ""  # it deletes something from the disk
    return f"<span class='mslot' data-slot='{act}'><button class='mact{danger}' data-act='{act}'>{html.escape(label)}</button></span>"


def cost_box(price=None, offpeak_note=""):
    """A small arrow next to the score; it opens a box with what the model costs per million tokens."""
    if not price:
        return ""
    money = lambda x: "—" if x is None else f"${x:,.4f}".rstrip("0").rstrip(".") if x < 0.1 else f"${x:,.2f}"
    rows = [(t("Input"), price[0]), (t("Cached input"), price[1]), (t("Output"), price[2])]
    box = "<table class='pbox'>" + "".join(f"<tr><td>{a}</td><td>{money(b)}</td>" + (f"<td>{money(price[3][i])}</td>" if len(price) > 3 else "") + "</tr>" for i, (a, b) in enumerate(rows)) + "</table>"
    if len(price) > 3:
        box = box.replace("<table class='pbox'>", "<table class='pbox'><tr><th></th><th>" + t("Normal") + "</th><th>" + t("Off-peak") + "</th></tr>", 1)
    note = t("Dollars per million tokens (about 750,000 words).")
    if price[0] is not None and price[2] is not None:  # a typical message: about 3,000 tokens of chat sent, 500 written back
        each = price[0] * 3000 / 1e6 + price[2] * 500 / 1e6
        note = "<b>" + t("About {cost} per typical message.", cost=money(each) if each >= 0.001 else "< $0.001") + "</b> " + note
    return (f"<details class='mcost'><summary title='{html.escape(t('Price'), quote=True)}'>{html.escape(t('Price'))}</summary><div class='pwrap'>{box}"
            f"<div class='pnote'>{note}{' ' + html.escape(t(offpeak_note)) if len(price) > 3 and offpeak_note else ''}</div></div></details>")


SCORE_TOP = 50  # the bars of every table share this top, so a short bar here is a short bar there too


def rank_table(rows, empty, size=None):
    """A ranked list: (name, size, can read, score or None, note). Best first; the bar shows the score.
    size: the heading of the second column (on disk, to download), or False to leave the column out."""
    if not rows:
        return f"<div class='cloudstat'>{empty}</div>"
    top = max([r[3] or 0 for r in rows] + [SCORE_TOP])
    score_tip = html.escape(t("Reasoning score from public tests (Artificial Analysis): higher is better, and it is the same scale in every table on this page."), quote=True)
    out = ["<table class='mrank'><tr><th>" + t("Model") + "</th>" + ("" if size is False else "<th>" + (size or t("Size")) + "</th>") + "<th>" + t("Abilities") + "</th><th title='" + score_tip + "'>" + t("Reasoning") + " ⓘ</th><th></th></tr>"]
    for name, msize, reads, score, note, *price in rows:
        bar = f"<span class='mscore'><span class='mbar'><i style='width:{100 * (score or 0) / top:.0f}%'></i></span> <span class='mnum'>{score:g}</span></span>" if score is not None else "<span class='mdim'>" + t("not rated") + "</span>"
        bar += cost_box(*price)
        out.append(f"<tr><td class='mname'>{html.escape(name)}</td>" + ("" if size is False else f"<td>{html.escape(msize)}</td>") + f"<td>{html.escape(reads)}</td><td class='mscorecell'>{bar}</td><td class='mnote'>{note}</td></tr>")
    return "".join(out) + "</table>"


def installed_html():
    """The models on this computer that Joshua can use, best first."""
    try:
        listed = {m.model: m for m in ollama.list().models if not is_cloud(m.model)}
    except Exception:
        listed = {}
    rows = []
    for n, m in listed.items():
        c = caps(n)
        if "tools" not in c:
            continue
        gb = (getattr(m, "size", 0) or 0) / 1e9
        name = ("★ " + n.removeprefix("joshua-") + " · " + t("taught by you")) if n.startswith("joshua-") else n.removesuffix(":latest")
        rows.append((name, gb_text(gb * 1e9), media("vision" in c, "audio" in c, "thinking" in c), ranking.local_score(n, gb), act_button("rm", n, t("Delete"))))
    plain = scratch.in_joshua()
    for n, m in listed.items():
        if n.removesuffix(":latest") in plain and "tools" not in caps(n):
            rows.append(("★ " + n.removesuffix(":latest") + " · " + t("built by you"), gb_text(getattr(m, "size", 0) or 0), media(False), None, act_button("rm", n, t("Delete"))))
    if not apple.why_not():
        rows.append((apple.LABEL, t("built into macOS"), media(False), ranking.data().get("apple", 1), ""))
    rows.sort(key=lambda r: -(r[3] or 0))
    return rank_table(rows, t("No local models yet: download one below."), t("On disk")) + other_models_html()


def other_models_html():
    """Models Joshua uses for other jobs than chatting, kept on this computer: today the speech model for transcription."""
    import transcribe as tr
    here = tr.model_dir()  # removed, it stays listed so it can be downloaded again
    size = f"{tr.size_gb():.1f} GB" if here else t("not downloaded")
    button = act_button("rmspeech", "whisper", t("Delete")) if here else act_button("getspeech", "whisper", t("Download"))
    return ("<h3 class='msub2'>" + t("Other models on this computer") + "</h3><table class='mrank'><tr><th>" + t("Model") + "</th><th>" + t("Size") + "</th><th>" + t("Used for") + "</th><th></th></tr>"
            + f"<tr><td class='mname'>🎙️ {tr.NAME}</td><td class='mdim'>{size}</td><td class='mdim'>" + t("transcribing audio and video, on this computer") + "</td><td class='mnote mbtns'>"
            + button + "</td></tr>" + image_row() + "</table>")


def image_row():
    """The image model, when image generation is installed: its size, and a button to free the space."""
    if not agent.IMAGES_OK:
        return ""
    p = agent.image_dir()
    size = f"{hfcache.size_gb(p):.1f} GB" if p else t("downloads with the first picture")
    return (f"<tr><td class='mname'>🎨 FLUX.2 klein 4B</td><td class='mdim'>{size}</td><td class='mdim'>" + t("creating and editing pictures, on this computer")
            + "</td><td class='mnote mbtns'>" + (act_button("rmimage", "flux", t("Delete")) if p else "") + "</td></tr>")


def cloud_catalog():
    """The cloud models Ollama offers today that can use tools, read from ollama.com (only model names, nothing of yours is sent)."""
    import requests
    try:
        names = [m["name"] for m in requests.get("https://ollama.com/api/tags", timeout=15).json()["models"]]
        info = {n: requests.post("https://ollama.com/api/show", json={"model": n}, timeout=15).json().get("capabilities", []) for n in names}
    except Exception:
        return "<div class='cloudstat'>" + t("Joshua couldn't reach ollama.com. Check your internet connection, then close and reopen this panel to try again.") + "</div>", ""
    free = {"gemma4:31b"}  # free for older accounts on Ollama's legacy free plan, and among the cheapest per message (November 2026)
    usable = sorted(((n, c) for n, c in info.items() if "tools" in c), key=lambda x: -(ranking.cloud_score(x[0]) or 0))
    have = set(CLOUD)
    added = lambda n: (n + ("-cloud" if ":" in n else ":cloud")) in have
    pnote = ranking.data().get("prices", {}).get("offpeak_cloud", "")
    rows = [(n, t("in the cloud"), media("vision" in c, "audio" in c, "thinking" in c), ranking.cloud_score(n), ("✓ " + t("in your menu") + " " + act_button("unmenu", n + ("-cloud" if ":" in n else ":cloud"), t("Remove from the menu")) if added(n) else act_button("add", n, t("Add"))) + (" · " + t("cheapest") if n in free else ""), ranking.cloud_price(n), pnote) for n, c in usable]
    rows.sort(key=lambda r: -(r[3] or 0))
    state = ("<div class='cloudstat cloudok'>✓ " + t("This computer is connected to Ollama: your cloud models work. Press the button only to connect a different account.") + "</div>") if have else ""
    return state, rank_table(rows, "", False)


def ollama_credit(key=None):
    """The credit left on the Ollama account, read with the API key you made on ollama.com (it stays on this computer)."""
    import requests
    key = (key or providers.settings()["keys"].get("ollama") or "").strip()
    if not key:
        return "<div class='cloudstat'>" + t("To see your credit here, make an API key on [ollama.com/settings/keys](https://ollama.com/settings/keys) and paste it below.").replace("[ollama.com/settings/keys](https://ollama.com/settings/keys)", "<a href='https://ollama.com/settings/keys' target='_blank' rel='noopener'>ollama.com/settings/keys</a>") + "</div>"
    try:
        r = requests.get("https://ollama.com/api/balance", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except Exception:
        return "<div class='cloudstat'>" + t("Joshua couldn't reach ollama.com to read your credit.") + "</div>"
    if r.status_code in (401, 403):
        return "<div class='cloudstat'>" + t("Ollama refused this API key. Make a new one on ollama.com and paste it again.") + "</div>"
    if not r.ok:
        return "<div class='cloudstat'>" + t("Ollama didn't say how much credit is left (error {code}).", code=r.status_code) + "</div>"
    data = r.json()
    inc, bought = data.get("included") or {}, (data.get("purchased") or {}).get("balance_usd")
    parts = []
    if "balance_usd" in inc and (inc.get("allowance_usd") or inc.get("balance_usd")):  # a plan that includes nothing has nothing to say
        until = (inc.get("period") or {}).get("until", "")[:10]
        parts.append(t("${left:.2f} of ${total:.2f} included in your plan, until {date}", left=inc["balance_usd"], total=inc.get("allowance_usd", 0), date=until))
    for name, label in (("session", t("this session")), ("weekly", t("this week"))):  # older plans count in percent
        if name in inc:
            parts.append(t("{n}% left {when}", n=inc[name].get("remaining_percent", 0), when=label))
    if bought is not None:
        parts.append(t("${n:.2f} of credit you bought", n=bought))
    return "<div class='cloudstat'>💳 <b>" + t("Ollama credit") + "</b><br>" + "<br>".join(html.escape(p) for p in parts) + "</div>"


_credit = {"at": 0, "key": "", "usd": None}


def credit_left():
    """Dollars left on the Ollama account (included in the plan + bought), or None without an API key.
    Asked at most once a minute: Ollama allows 10 requests a minute."""
    import requests
    key = (providers.settings()["keys"].get("ollama") or "").strip()
    if not key:
        return None
    if key == _credit["key"] and time.time() - _credit["at"] < 60:
        return _credit["usd"]
    usd = None
    try:
        r = requests.get("https://ollama.com/api/balance", headers={"Authorization": f"Bearer {key}"}, timeout=8)
        if r.ok:
            data = r.json()
            inc, bought = data.get("included") or {}, (data.get("purchased") or {}).get("balance_usd")
            if "balance_usd" in inc or bought is not None:
                usd = float(inc.get("balance_usd") or 0) + float(bought or 0)
    except Exception:
        pass
    _credit.update(at=time.time(), key=key, usd=usd)
    return usd


def save_ollama_key(key):
    if key and key.strip():
        providers.set_key("ollama", key)
    shown = ollama_credit()
    return "", shown, gr.Row(visible="💳" not in shown)  # the credit is there: no need for the field any more


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
    return agent.total_ram()


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
            found[tag] = (float(size[1]), "Image" in info, "Audio" in info)
    return [(t, gb, vision, audio) for t, (gb, vision, audio) in found.items()]


def _tool_models(order, limit):
    """Model names from ollama.com/library (sorted by popularity or newest) whose label says they can use tools."""
    import requests
    page = requests.get(f"https://ollama.com/library?sort={order}", timeout=20).text
    cards = re.split(r'(?=<a href="/library/[\w.-]+")', page)[1:]
    names = [re.match(r'<a href="/library/([\w.-]+)"', c)[1] for c in cards if re.search(r">\s*tools\s*</span>", c)]
    THINKERS.update(re.match(r'<a href="/library/([\w.-]+)"', c)[1] for c in cards if re.search(r">\s*thinking\s*</span>", c))
    return names[:limit]


def local_catalog():
    """Models on ollama.com that can use tools, in the versions that fit this computer's memory, most capable first."""
    try:
        names = list(dict.fromkeys(_tool_models("popular", 30) + _tool_models("newest", 12)))
        with ThreadPoolExecutor(8) as pool:
            tags = [t for ts in pool.map(lambda n: _try(_model_tags, n), names) for t in ts]
    except Exception:
        return "<div class='cloudstat'>" + t("Joshua couldn't reach ollama.com. Check your internet connection, then close and reopen this panel to try again.") + "</div>", ""
    sized = {(x[0].split(":")[0], x[1]) for x in tags if not x[0].endswith(":latest")}
    tags = [x for x in tags if not (x[0].endswith(":latest") and (x[0].split(":")[0], x[1]) in sized)]  # 'latest' is usually a copy of another size
    budget, free = memory_budget(), shutil.disk_usage(pathlib.Path.home()).free / 2**30
    have = {m.model for m in _try(lambda: ollama.list().models)}
    fits = sorted({x for x in tags if x[1] * 1.2 + 2 <= budget}, key=lambda x: (-ranking.local_score(x[0], x[1]), x[1]))
    rows = [(tg, f"{gb:g} GB", media(v, a, tg.split(":")[0] in THINKERS), ranking.local_score(tg, gb), "✓ " + t("installed") if tg in have else act_button("dl", tg, t("Download"))) for tg, gb, v, a in fits]
    note = t("{n} versions fit this computer ({memory} GB of memory for models, {disk} GB free disk). Most capable first.", n=len(rows), memory=f"{budget:.0f}", disk=f"{free:.0f}")
    if len(rows) <= 4:
        return f"<div class='cloudstat'>{note}</div>", rank_table(rows, "", t("Download"))
    return (f"<div class='cloudstat'>{note}</div>",
            "<h4 class='mrec'>" + t("Recommended for this computer: the most capable ones that fit") + "</h4>" + rank_table(rows[:3], "", t("Download"))
            + "<details class='mall'><summary>" + t("Show all {n}", n=len(rows)) + "</summary>" + rank_table(rows[3:], "", t("Download")) + "</details>")


def menu_order():
    """The menu as the Models page shows it: your order, or (until you set one) the default model first."""
    items = models()
    if not model_order():
        d = default_model()
        items.sort(key=lambda c: c[1] != d)
    return items


def menu_html():
    """The models in your model menu, in their order, with buttons to move them and to take them out."""
    items = menu_order()
    if not items:
        return ""
    try:
        sizes = {m.model: m.size or 0 for m in ollama.list().models}
    except Exception:
        sizes = {}
    rows = []
    for i, (label, v) in enumerate(items):
        kind = t("on this computer") if not is_cloud(resolve(v, False) if v == PAIR else v) and not providers.is_api(v) else t("in the cloud")
        parts = [n for n in (FAST, MODEL, MODEL + ":latest") if n in sizes and not is_cloud(n)] if v == PAIR else [v] if v in sizes else []
        used = sum(sizes[n] for n in parts)
        if used and kind == t("on this computer"):
            kind += " · " + gb_text(used)
        btns = "<span class='marrows'>" + (act_button("up", v, "↑") if i else "<span class='mspace'></span>") + (act_button("down", v, "↓") if i < len(items) - 1 else "<span class='mspace'></span>") + "</span>"
        if v not in (PAIR, apple.NAME) and made_here(v):
            btns += act_button("ren", v, t("Rename"))
        if v not in (PAIR, apple.NAME):
            local = not is_cloud(v) and not providers.is_api(v)  # a local model's button deletes it from the disk, so it says so (and asks first)
            btns += act_button("rm", v, t("Delete from computer")) if local else act_button("unmenu", v, t("Remove from the menu"))
        first = " <span class='mdefault'>" + t("default") + "</span>" if i == 0 else ""
        c = caps(resolve(v, False)) if v != apple.NAME else set()
        can = media("vision" in c, "audio" in c, "thinking" in c or (v == PAIR and "thinking" in caps(MODEL)))
        rows.append(f"<tr><td class='mname'>{html.escape(label)}{first}</td><td class='mdim'>{kind}</td><td class='mdim'>{html.escape(can)}</td><td class='mnote mbtns'>{btns}</td></tr>")
    head = "<tr><th>" + t("Model") + "</th><th>" + t("Where") + "</th><th>" + t("Abilities") + "</th><th>" + t("Order and actions") + "</th></tr>"
    return "<table class='mrank mmenu'>" + head + "".join(rows) + "</table>"


def move_model(v, step):
    order = [c for _, c in menu_order()]
    if v in order:
        i = order.index(v)
        j = max(0, min(len(order) - 1, i + step))
        order.insert(j, order.pop(i))
        model_order(order)


def row_action(act):
    """A Download or Add button pressed in one of the tables: 'dl:qwen3:8b' or 'add:kimi-k3'."""
    kind, _, tag = (act or "").partition(":")
    if kind == "dl":
        for m, status in local_download(tag):
            yield m, gr.skip(), status, gr.skip()
    elif kind in ("up", "down"):
        move_model(tag, -1 if kind == "up" else 1)
        yield gr.Dropdown(choices=models(True)), gr.skip(), gr.skip(), gr.skip()
    elif kind == "ren":
        old, _, new = tag.partition("|")
        try:
            new = rename_model(old, new)
            yield gr.Dropdown(choices=models(True)), "<div class='cloudstat'>" + t("{old} is now called {new}: you'll find it with this name in the model menu.", old=html.escape(old.removesuffix(":latest").removeprefix("joshua-")), new=html.escape(new.removeprefix("joshua-"))) + "</div>", gr.skip(), gr.skip()
        except Exception as e:
            yield gr.skip(), "<div class='cloudstat'>" + t("Couldn't rename {model}: {error}", model=html.escape(old), error=html.escape(str(e))) + "</div>", gr.skip(), gr.skip()
    elif kind == "unmenu" and providers.is_api(tag):
        providers.remove_model(tag)
        yield gr.Dropdown(choices=models(True)), gr.skip(), gr.skip(), gr.skip()
    elif kind in ("rm", "unmenu"):
        try:
            ollama.delete(tag)
            CAPS.pop(tag, None)
            yield gr.Dropdown(choices=models(True)), "<div class='cloudstat'>" + t("{model} was removed from this computer.", model=html.escape(tag)) + "</div>", gr.skip(), gr.skip()
        except Exception as e:
            yield gr.skip(), "<div class='cloudstat'>" + t("Couldn't remove {model}: {error}", model=html.escape(tag), error=html.escape(str(e))) + "</div>", gr.skip(), gr.skip()
    elif kind == "rmspeech":
        import transcribe as tr
        tr.remove()
        yield gr.skip(), "<div class='cloudstat'>" + t("The speech model was removed. It downloads again the next time you ask for a transcription, or with Download.") + "</div>", gr.skip(), gr.skip()
    elif kind == "rmimage":
        hfcache.remove(agent.image_dir())
        yield gr.skip(), "<div class='cloudstat'>" + t("The image model was removed. It downloads again the next time you ask for a picture.") + "</div>", gr.skip(), gr.skip()
    elif kind == "getspeech":
        import transcribe as tr
        yield gr.skip(), "<div class='cloudstat'>" + t("Downloading the speech model (about 1.6 GB)…") + "</div>", gr.skip(), gr.skip()
        try:
            tr.download()
            yield gr.skip(), "<div class='cloudstat'>" + t("The speech model is ready: Joshua can transcribe audio and video.") + "</div>", gr.skip(), gr.skip()
        except Exception as e:
            yield gr.skip(), "<div class='cloudstat'>" + t("Couldn't download the speech model: {error}", error=html.escape(str(e))) + "</div>", gr.skip(), gr.skip()
    elif kind == "add":
        m, status = cloud_add_model(tag)
        yield m, gr.skip(), gr.skip(), status


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
    done, shown, last = 0, None, 0.0
    try:
        for p in ollama.pull(tag, stream=True):
            if p.total:
                done = int(100 * (p.completed or 0) / p.total)
            line = (done, p.status)
            if line == shown or time.time() - last < 0.5:  # Ollama reports hundreds of times a second: sending each one froze the page
                continue
            shown, last = line, time.time()
            # the page script draws this as a progress bar in the model's own row
            yield gr.skip(), f"<span class='dlp' data-slot='{html.escape('dl:' + tag, quote=True)}' data-pct='{done}' data-label='{html.escape(t('Downloading {model}: {percent}%', model=tag, percent=done), quote=True)}'></span>"
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
jlog.started(VERSION)
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
        return ("<div class='updnote' title='" + html.escape(t("Joshua {new} is available (you have {old}).", new=latest, old=VERSION), quote=True) + "'>"
                + t("New version") + "</div>"), gr.Row(visible=True)
    return "", gr.Row(visible=False)


def do_update():
    """Downloads the new version over this one (your chats, files, memory and settings are left alone), then restarts."""
    import io, requests, tarfile
    here = pathlib.Path(__file__).parent
    jlog.note(f"update from {VERSION} started")
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
        return "<div class='updnote'>" + tr("Updated. Joshua is restarting: this page reloads by itself when it's ready (up to a minute).") + "</div>"
    return "<div class='updnote'>" + tr("Updated. Quit and start Joshua again to use the new version.") + "</div>"


def restart():
    """Quits and starts Joshua again; False if it can't start itself (then you restart it by hand)."""
    if os.environ.get("JOSHUA_SERVICE"):  # running from login: just quit, and the system starts it again
        threading.Timer(1, lambda: os._exit(0)).start()
        return True
    if agent.WIN:  # Windows: start.cmd starts it again in a moment, from a separate process
        start = pathlib.Path.home() / "AskJoshua" / "start.ps1"
        if start.exists():  # PowerShell runs start.ps1 as text (company rules may forbid .ps1 files); no double quotes, which cmd would misread
            script = str(start).replace("'", "''")
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command",
                              f"Start-Sleep 4; & ([scriptblock]::Create([IO.File]::ReadAllText('{script}')))"],
                             env={**os.environ, "JOSHUA_STAY_HIDDEN": "1"}, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            threading.Timer(1, lambda: os._exit(0)).start()
            return True
        return False
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
        return gr.Dropdown(choices=[], value=None, interactive=False), "<div class='cloudstat'>" + t("No key yet.") + " " + link + "</div>"
    try:
        names = providers.catalog(provider)
    except Exception as e:
        return gr.Dropdown(choices=[], value=None, interactive=False), "<div class='cloudstat'>" + html.escape(str(e)) + " " + link + "</div>"
    names.sort(key=lambda n: -(ranking.closed_score(n) or 0))
    return gr.Dropdown(choices=[(n + (f" · {ranking.closed_score(n):g}" if ranking.closed_score(n) else ""), n) for n in names], value=names[0] if names else None, interactive=bool(names)), "<div class='cloudstat'>" + t("Key works: {n} models available, most capable first.", n=len(names)) + "</div>"


def keys_save(provider, key):
    if not key or not key.strip():
        return gr.skip(), "<div class='cloudstat'>" + t("Paste the key first.") + "</div>", gr.skip()
    providers.set_key(provider, key)
    pick, status = keys_status(provider)
    return pick, status, ""  # empty the box: the key is not shown again


def closed_html():
    """The Claude, ChatGPT, Gemini and DeepSeek models in your menu, best first."""
    pnote = ranking.data().get("prices", {}).get("offpeak_closed", "")
    rows = [(providers.label(v).removeprefix("🔑 "), "☁", media(providers.sees_images(v)), ranking.closed_score(providers.split(v)[1]), "", ranking.closed_price(providers.split(v)[1]), pnote) for _, v in providers.menu(False)]
    rows.sort(key=lambda r: -(r[3] or 0))
    return rank_table(rows, t("None yet: add a key below, then pick a model."))


def keys_add(provider, name):
    if not name:
        return gr.skip(), "<div class='cloudstat'>" + t("Pick a model first.") + "</div>"
    value = providers.add_model(provider, name)
    return gr.Dropdown(choices=models(True), value=value), "<div class='cloudstat'>" + t("Added {model}: it's selected in the model menu.", model=html.escape(providers.label(value))) + "</div>"


def cloudmem_info(on):
    """The line under 'Share notes with cloud models' says what the box does as it is now."""
    return t("On: your notes are sent to cloud models with each message") if on else t("Off: cloud models never see your notes")


def share_memory_setting(value=None):
    """Whether cloud models may see the memory notes. Off unless you switch it on; stored in memory/settings.json."""
    f = agent.MEM / "settings.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    if value is not None:
        data["share_with_cloud"] = bool(value)
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(data))
    return data.get("share_with_cloud", False)


def _settings(key, default, value=None):
    """A setting kept in memory/settings.json."""
    f = agent.MEM / "settings.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    if value is not None:
        data[key] = value
        f.write_text(json.dumps(data))
    return data.get(key, default)


PORT = int(os.environ.get("GRADIO_SERVER_PORT", 7860))


def remote_setting(value=None):
    """Whether your other devices can reach Joshua through Tailscale."""
    return _settings("remote", False, None if value is None else bool(value))


def remote_html():
    """Where things stand with Tailscale, and what to do next."""
    now = remote.info()
    box = lambda body: "<div class='remotebox'>" + body + "</div>"
    if now["state"] == "missing":
        return box(t("Tailscale isn't installed on this computer.") + "<ol><li>" + t("Download it from <a href='https://tailscale.com/download' target='_blank' rel='noopener'>tailscale.com/download</a> and install it.") + "</li><li>"
                   + t("Open it and sign in (a free personal account is enough).") + "</li><li>" + t("Come back here and press Check again.") + "</li></ol>")
    if now["state"] == "stopped":
        return box(t("Tailscale is installed but not connected: open it from the menu bar and sign in, then press Check again."))
    if not remote_setting():
        return box(t("Tailscale is connected as <b>{name}</b>. Switch on access to reach Joshua from your other devices.", name=html.escape(now["name"] or now["ip"])))
    if not remote.listening():
        return box(t("Access is on, but Joshua couldn't listen on the Tailscale address. Press Check again in a moment."))
    urls = [u for u in (f"http://{now['name']}:{PORT}" if now["name"] else "", f"http://{now['ip']}:{PORT}") if u]
    copy = lambda u: f"<button type='button' class='keyshow' onclick=\"navigator.clipboard.writeText('{u}'); this.textContent = {html.escape(json.dumps(t('Copied')), quote=True)}\">{t('Copy')}</button>"
    links = "".join(f"<div class='rurl'><a href='{u}' target='_blank' rel='noopener'><code>{u}</code></a> {copy(u)}</div>" for u in urls)
    return box("<div class='remotegrid'><div>" + t("Joshua can be reached at:") + links + "<p>" + t("On each device you want to use: install Tailscale, sign in with the same account, then open this address in the browser. A phone can scan the code.") + "</p><p class='mdim'>"
               + t("This computer has to stay on and awake: in System Settings, turn on \"Prevent automatic sleeping when the display is off\".") + "</p></div><div class='remoteqr'>" + remote.qr_svg(urls[0]) + "</div></div>")


def learn_setting(value=None):
    """Whether Joshua adds memory notes from your chats by itself."""
    return _settings("learn", True, None if value is None else bool(value))


def tools_off(value=None):
    """The tools you switched off in Settings."""
    return _settings("tools_off", [], value)


TOOL_GROUPS = {"web_search": ("🔎 " + t("Search the web"), ["web_search"]), "read_page": ("📄 " + t("Read web pages"), ["read_page"]),
               "run_python": ("🐍 " + t("Run Python (in a sandbox)"), ["run_python"]), "install_package": ("📦 " + t("Install packages"), ["install_package"]),
               "list_files": ("🗂️ " + t("Browse folders"), ["list_files"]), "files": ("📂 " + t("Read and write files (text, Word, PDF)"), ["read_file", "write_file"]),
               "get_datetime": ("🕒 " + t("Check the date and time"), ["get_datetime"]), "look_at_image": ("🖼️ " + t("Look at images"), ["look_at_image"]),
               "transcribe": ("🎙️ " + t("Transcribe audio and video (on this computer)"), ["transcribe"]),
               **({"generate_image": ("🎨 " + t("Generate images"), ["generate_image"]), "edit_image": ("🖌️ " + t("Edit your photos"), ["edit_image"])} if "edit_image" in tools else {}),
               "browser": ("🧭 " + t("Use websites in its own browser (Firefox)"), sorted(agent.BROWSER_TOOLS)),
               "memory": ("🧠 " + t("Remember and forget"), sorted(agent.MEMORY_TOOLS))}


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
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"title": "", "updated": 0, "display": [], "llm": []}


def write_chat(cid, conv):
    """Write the chat whole, then swap it in: a chat read at the same moment is never half-written."""
    p = ROOT / cid / "chat.json"
    tmp = p.with_name(f"chat.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(conv, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save(cid, conv):
    conv["updated"] = time.time()
    write_chat(cid, conv)


def new_conv():
    cid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    fdir(cid).mkdir(parents=True)
    save(cid, load(cid))
    return cid


def cap(s):
    return s[:1].upper() + s[1:] if s else s


TITLE = "Write a title of 3 to 6 words that sums up this conversation, in {language}. Only the title: no quotes, no emoji, no final full stop.\n\nUser: {user}\n\nAssistant: {answer}"


def make_title(model, user, answer, language):
    try:
        r = ollama.chat(fast_of(model), messages=[{"role": "user", "content": TITLE.format(language=language or "the language of the user's message", user=user[:1500], answer=answer[:1500])}],
                        options={"temperature": 0.2, "num_predict": 40}, **think_mode(model, False))
        title = split_think(r.message.content or "")[0].strip().splitlines()[0].strip(" \"'“”«»*#.").strip()
        return cap(title[:60]) if 2 <= len(title) else ""
    except Exception:
        return ""


def conv_choices():
    convs = [(load(p.name), p.name) for p in ROOT.iterdir() if (p / "chat.json").exists()]
    order = sorted(convs, key=lambda x: (not x[0].get("pinned"), x[0].get("pin_order", 0), -x[0]["updated"]))  # pinned chats first, in your order; then the most recent
    # PIN is an invisible character: the page script sees it and draws the green pin
    icons = {k: d.get("icon", "") for k, d in desks.desks()}  # a chat at a desk shows the desk's icon
    return [((PIN if c.get("pinned") else "") + ("⏺ " if busy(cid) else "") + (icons.get(c.get("opts", {}).get("desk"), "") + " " if icons.get(c.get("opts", {}).get("desk")) else "")
             + (cap(c["title"]) or t("New chat")), cid) for c, cid in order]


PIN = "\u2063"


def pinned_ids():
    return [cid for label, cid in conv_choices() if label.startswith(PIN)]


def write_conv(cid, conv):
    write_chat(cid, conv)  # not save(): this doesn't move the chat to the top


def reorder_pins(cid, order):
    """The pinned chats dragged into a new order: 'order' lists their old positions, like '2,0,1'."""
    ids = pinned_ids()
    try:
        new = [ids[int(i)] for i in (order or "").split(",")]
    except (ValueError, IndexError):
        return gr.skip()
    if sorted(new) == sorted(ids):
        for n, c in enumerate(new, 1):
            conv = load(c)
            conv["pin_order"] = n
            write_conv(c, conv)
    return gr.Radio(choices=conv_choices(), value=cid)


OPTS = ("thinking", "goal", "swarm", "offline")


WAIT_AND_RELOAD = """() => { const t0 = Date.now(); let down = false;
  const tick = async () => { try { const r = await fetch(location.pathname, {cache: 'no-store'}); if (r.ok && down) { location.reload(); return; } } catch (e) { down = true; }
    if (Date.now() - t0 < 600000) setTimeout(tick, 2000); };
  setTimeout(tick, 1500); }"""
FOLDER_SVG = "<svg viewBox='0 0 24 24' width='22' height='22' fill='none' stroke='currentColor' stroke-width='1.7' stroke-linejoin='round'><path d='M3 6.5A1.5 1.5 0 0 1 4.5 5h4.3l2 2.2h8.7A1.5 1.5 0 0 1 21 8.7v9.8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z'/></svg>"
TEAM = 6  # the most agents a swarm can have


def team_menu(team=(), n=5):
    """The swarm's model menus: one per agent, 'Automatic' or a model of yours (the same one may be picked for several agents)."""
    choices = [(t("Automatic"), "")] + models()
    values = [v for _, v in choices]
    team = (list(team) + [""] * TEAM)[:TEAM]
    return [gr.Dropdown(choices=choices, value=v if v in values else "", visible=k < n) for k, v in enumerate(team)]


GENERAL_STARTERS = [  # the General desk's starter cards: everyday things a first-time user may not think of asking
    {"icon": "📄", "title": t("Summarise a file"), "prompt": t("Summarise the attached file in five short points: ")},
    {"icon": "💡", "title": t("Explain something simply"), "prompt": t("Explain this simply, as you would to a curious friend: ")},
    {"icon": "✉️", "title": t("Draft an email"), "prompt": t("Write a short, polite email to ")},
    {"icon": "🔎", "title": t("Check a fact on the web"), "prompt": t("Check on the web whether this is true, and tell me your sources: ")},
    {"icon": "🧭", "title": t("Plan something"), "prompt": t("Help me plan ")},
]


def starters_html(desk_id):
    """The empty chat's starter cards for a desk: each one puts a ready prompt in the message box."""
    if desk_id == "falken":  # WOPR's terminal: each game is a line you can click
        items = "".join(f"<button class='starter wline' data-prompt='{k}' data-send='1'>{k:>2}. {html.escape(g)}</button>" + ("<br>" if k == 14 else "")
                        for k, g in enumerate(falken.GAMES, 1))
        return ("<div class='wopr'>GREETINGS PROFESSOR FALKEN.<br><br>SHALL WE PLAY A GAME?<br><br>HOW ABOUT:<br><br>" + items
                + "<br>PLEASE ENTER YOUR SELECTION:<span class='wcur'>_</span></div>")
    d = desks.get(desk_id) if desk_id else {"starters": GENERAL_STARTERS}
    if not d:
        return ""
    cards = "".join("<button class='starter' data-prompt='" + html.escape(desks.text(s["prompt"], i18n.LANG), quote=True) + "'" + (" data-goal='1'" if s.get("goal") else "") + "><span class='sicon'>" + s.get("icon", "")
                    + "</span>" + html.escape(desks.text(s["title"], i18n.LANG)) + "</button>" for s in d.get("starters", []))
    return "<div class='starters'>" + cards + "</div>"


def desk_editor_choices():
    return [("＋ " + t("New desk"), "")] + [(desks.label(k, i18n.LANG) + ("" if d.get("mine") or not desks.edited(k) else "  · " + t("changed")), k) for k, d in desks.desks()]


def hidden_desk_choices():
    return [(desks.text(desks.default(k)["name"], i18n.LANG), k) for k in desks.hidden()]


def desk_editor_load(desk_id):
    """The editor shows one desk: what Joshua is told there, its tools and its tasks (in the language you see)."""
    d = desks.get(desk_id)
    if not d:  # a new desk
        return ("", "", "", [k for k in TOOL_GROUPS if k != "browser"], gr.Button(visible=False), gr.Button(visible=False), "", *desk_task_fields([]))
    rows = [[s.get("icon", ""), desks.text(s["title"], i18n.LANG), desks.text(s["prompt"], i18n.LANG)] for s in d.get("starters", [])]
    builtin = not d.get("mine")
    note = "<div class='cloudstat'>" + (t("Made by you") if not builtin else t("One of Joshua's own desks, changed by you: Restore the default brings back the original") if desks.edited(desk_id) else t("One of Joshua's own desks: if you remove it, you can bring it back later")) + "</div>"
    return (d.get("icon", ""), desks.text(d["name"], i18n.LANG), d.get("instructions", ""), [k for k in d.get("tools", []) if k in TOOL_GROUPS],
            gr.Button(visible=builtin and desks.edited(desk_id)), gr.Button(visible=True), note, *desk_task_fields(rows))


MAX_TASKS = 12


def desk_task_fields(rows):
    """The task lines of the desk editor: the values of every field, then which lines are shown."""
    rows = rows[:MAX_TASKS]
    values = [x for r in rows for x in r] + [""] * 3 * (MAX_TASKS - len(rows))
    return values + [gr.Row(visible=k < max(len(rows), 1)) for k in range(MAX_TASKS)]


def desk_task_rows(values):
    return [list(values[k:k + 3]) for k in range(0, 3 * MAX_TASKS, 3)]


def desk_task_add(*values):
    rows = [r for r in desk_task_rows(values) if any(x.strip() for x in r)]
    return desk_task_fields(rows + [["", "", ""]]) if len(rows) < MAX_TASKS else [gr.skip()] * (4 * MAX_TASKS)


def desk_task_remove(k, *values):
    rows = desk_task_rows(values)
    del rows[k]
    while rows and not any(x.strip() for x in rows[-1]):  # the hidden empty lines at the end aren't tasks
        rows.pop()
    return desk_task_fields(rows)


def _rows(values):
    return [{"icon": (r[0] or "").strip()[:4], "title": (r[1] or "").strip(), "prompt": (r[2] or "").strip()} for r in desk_task_rows(values) if (r[1] or "").strip()]


def desk_editor_save(desk_id, icon, name, instructions, tools, *cells):
    if not (name or "").strip() or not (instructions or "").strip():
        gr.Warning(t("Give the desk a name and write its instructions."))
        return (gr.skip(),) * (9 + 4 * MAX_TASKS)
    desk_id = desks.save(desk_id, name, icon or "", instructions, tools or [], _rows(cells), i18n.LANG)
    gr.Info(t("Saved: chats at this desk follow it from the next message."))
    return (gr.Dropdown(choices=desk_editor_choices(), value=desk_id), gr.Radio(choices=desk_choices())) + desk_editor_load(desk_id)


def desk_editor_reset(desk_id):
    desks.reset(desk_id)
    gr.Info(t("Joshua's own version is back."))
    return (gr.Dropdown(choices=desk_editor_choices(), value=desk_id), gr.Radio(choices=desk_choices())) + desk_editor_load(desk_id)


def desk_editor_remove(desk_id):
    if desk_id is None:  # you cancelled
        return (gr.skip(),) * (11 + 4 * MAX_TASKS)
    if desk_id:
        desks.remove(desk_id)
        gr.Info(t("Desk removed. Its chats stay, at no desk."))
    first = desk_editor_choices()[1][1] if len(desk_editor_choices()) > 1 else ""
    return ((gr.Dropdown(choices=desk_editor_choices(), value=first), gr.Radio(choices=desk_choices())) + desk_editor_load(first)
            + (gr.Row(visible=bool(desks.hidden())), gr.Dropdown(choices=hidden_desk_choices(), value=None)))


def desk_editor_back(desk_id):
    if desk_id:
        desks.unhide(desk_id)
    return (gr.Dropdown(choices=desk_editor_choices(), value=desk_id or gr.skip()), gr.Radio(choices=desk_choices())) + desk_editor_load(desk_id) + (
        gr.Row(visible=bool(desks.hidden())), gr.Dropdown(choices=hidden_desk_choices(), value=None))


def desk_choices():
    return [("💬 " + t("General"), "")] + [(desks.label(k, i18n.LANG), k) for k, _ in desks.desks()]


def chat_opts(cid):
    """This chat's own settings: deep thinking, goal mode, swarm, no internet, and the swarm's agents."""
    o = load(cid).get("opts", {}) if cid else {}
    n = int(o.get("agents", 5))
    desk = o.get("desk", "") if desks.get(o.get("desk")) else ""
    solo = not (cid and teams.get(load(cid).get("team")))  # in a team chat each member has its own settings: no desks, goal or swarm here
    return (tuple(gr.Checkbox(value=True if k == "thinking" else bool(o.get(k)), **({"visible": solo} if k in ("goal", "swarm") else {})) for k in OPTS)
            + (gr.Slider(value=n),) + tuple(team_menu(o.get("team", ()), n)) + (gr.Radio(value=desk, visible=solo), starters_html(desk)))


def save_opts(cid, *values):
    if cid and (ROOT / cid).exists():
        conv = load(cid)
        flags, (n, *team), desk = values[:len(OPTS)], values[len(OPTS):-1], values[-1]
        conv["opts"] = dict(zip(OPTS, map(bool, flags)), agents=int(n), team=[v or "" for v in team], desk=desk or "")
        write_chat(cid, conv)  # not save(): a setting doesn't move the chat to the top


def toggle_pin(cid):
    if cid and (ROOT / cid).exists():
        conv = load(cid)
        others = [load(c).get("pin_order", 0) for c in pinned_ids() if c != cid]
        conv["pinned"] = not conv.get("pinned")
        conv["pin_order"] = min(others, default=1) - 1  # a newly pinned chat goes to the top of the pinned ones
        write_conv(cid, conv)
    return gr.Radio(choices=conv_choices(), value=cid)


def file_list(cid):
    if not cid or not wdir(cid).exists():
        return []
    return sorted(str(p) for p in wdir(cid).iterdir() if p.is_file() and not p.name.startswith("."))[:200]


# ---------- the agent loop (runs in a background thread per chat) ----------

class Job:
    def __init__(self, view):
        self.view, self.version, self.running, self.stop = list(view), 0, True, threading.Event()
        self.started, self.ended = time.time(), None

    def publish(self, view):
        self.view = list(view)
        self.version += 1


def is_step(m):
    """A thinking box or a tool box: the working steps that are grouped on screen."""
    md = m.get("metadata") or {} if isinstance(m, dict) else {}
    title = md.get("title", "")
    if not title or not isinstance(m.get("content"), str) or "parent_id" in md:
        return False
    name = title.split(" ", 1)[-1].split(" ×")[0]
    return title == "💭 " + t("Thinking") or name in tools or name in ("goal_done", "give_up")


def wopr_view(view):
    """A Prof Falken chat is one terminal screen that changes in place: what you typed last, then WOPR's latest screen."""
    said = [m for m in view if m.get("role") == "user" and isinstance(m.get("content"), str)]
    answers = [k for k, m in enumerate(view) if m.get("role") == "assistant" and not m.get("metadata") and isinstance(m.get("content"), str) and m["content"].strip()]
    last_user = max((k for k, m in enumerate(view) if m.get("role") == "user"), default=-1)
    text = lambda c: c.strip().removeprefix("```wopr").removesuffix("```").strip("\n") if c.lstrip().startswith("```wopr") else c.strip().upper()
    waiting = not answers or answers[-1] < last_user  # WOPR is still answering (a chat move, not a game move)
    body = "..." if waiting else text(view[answers[-1]]["content"])
    typed = ("> " + said[-1]["content"].strip().upper() + "\n\n") if said else ""
    return [{"role": "assistant", "content": "```wopr\n" + typed + body + "\n```"}]


def present(view, live=False):
    """What the chat shows: runs of working steps become one box. While Joshua works, the box shows the latest step
    and opens on the steps one by one; once the answer comes, it becomes one text with every step, in order.
    Only the screen changes: the chat is saved as it is."""
    first = next((m for m in view if m.get("role") == "assistant" and not m.get("metadata") and isinstance(m.get("content"), str)), None)
    if first and first["content"].startswith("```wopr"):  # a chat that began as a game: one terminal screen
        return wopr_view(view)
    out, run = [], []

    def flush(at_end, after=None):
        thought_answer = None
        if run and not (live and at_end) and (after is None or after.get("role") == "user") and run[-1]["metadata"]["title"] == "💭 " + t("Thinking"):
            thought_answer = run.pop()  # the turn ended with thinking and no answer: that thinking was the answer (chats saved before the fix)
        if len(run) < 2:
            out.extend(run)
        elif live and at_end:  # still working: the latest step in the title, every step inside it
            gid = f"steps{len(out)}"
            out.append({"role": "assistant", "content": "", "metadata": {"title": "⏳ " + run[-1]["metadata"]["title"].split(" ", 1)[-1] + "…", "id": gid, "status": "done", "log": t("{n} steps", n=len(run))}})
            out.extend({**m, "metadata": {**m["metadata"], "parent_id": gid, "status": "done"}} for m in run)
        else:  # finished: one text with everything Joshua did
            text = "\n\n---\n\n".join(f"#### {m['metadata']['title'].split(' ', 1)[-1]}\n\n{m['content'].strip()}" for m in run)
            out.append({"role": "assistant", "content": text, "metadata": {"title": "🧭 " + t("How Joshua worked"), "status": "done", "log": t("{n} steps", n=len(run))}})
        run.clear()
        if thought_answer:
            out.append({"role": "assistant", "content": thought_answer["content"]})

    for m in view:
        if is_step(m):
            run.append(m)
        else:
            flush(False, m)
            out.append(m)
    flush(True)
    return out


JOBS = {}
QUEUE = {}  # chat id -> messages sent while Joshua was still working, sent in order when it finishes


def split_think(raw):
    """The answer and the reasoning, when a model puts its reasoning in the text: <think>…</think> answer, or just … </think> answer."""
    if "</think>" in raw:
        inline, answer = raw.split("</think>", 1)
        return answer.lstrip(), inline.replace("<think>", "").strip()
    if raw.lstrip().startswith("<think>"):  # still reasoning
        return "", raw.lstrip()[len("<think>"):].strip()
    return raw, ""


def link_files(text, work):
    """Turns the names of files in the chat's folder, where an answer mentions them, into links that open or download them."""
    work = pathlib.Path(work)
    if not text or not work.is_dir():
        return text
    found = [f for f in work.iterdir() if f.is_file() and not f.name.startswith(".")]
    found += [f for d in work.iterdir() if d.is_dir() and not d.name.startswith(".") for f in d.iterdir() if f.is_file()][:200]
    for f in sorted(found, key=lambda f: -len(f.name)):  # longest names first: "plan-v2.pdf" before "plan.pdf"
        if len(f.name) < 5 or "." not in f.name:
            continue
        url = "/gradio_api/file=" + urllib.parse.quote(f.resolve().as_posix(), safe="/:")
        text = re.sub(r"(?<![\w/\[(.-])`?" + re.escape(f.name) + r"`?(?![\w\]/-])", lambda m: f"[{f.name}]({url})", text)
    return text


def pictures():
    """Pictures in the workspace and its first-level folders, with when they were last changed."""
    work, found = getattr(agent.CTX, "work", None), {}
    if not work or not work.is_dir():
        return found
    for d in [work] + [x for x in work.iterdir() if x.is_dir() and not x.name.startswith(".")][:50]:
        for f in d.iterdir():
            if f.suffix.lower() in IMAGES and f.is_file():
                found[str(f)] = f.stat().st_mtime
    return found


def agent_turn(msgs, shown, toolset, model, thinking, stop, called, limit=25):
    for step in range(limit):
        last = step == limit - 1  # the last step has no tools: the model must answer with what it has
        if last:
            msgs.append({"role": "user", "content": WRAP})
        content, thought, calls, live = "", "", [], []
        raw, reasoned = "", ""
        mode = {"model": model, **think_mode(model, thinking)}
        for chunk in ollama.chat(**mode, messages=msgs, tools=[] if last else list(toolset.values()), stream=True, options={"num_ctx": 16384}):
            reasoned += chunk.message.thinking or ""
            raw += chunk.message.content or ""
            content, inline = split_think(raw)  # some models (glm) write their reasoning in the text, ending it with </think>
            thought = reasoned + ("\n\n" if reasoned and inline else "") + inline
            calls += chunk.message.tool_calls or []
            live = [{"role": "assistant", "content": thought, "metadata": {"title": "💭 " + t("Thinking"), "status": "done"}}] if thought else []
            live += [{"role": "assistant", "content": content}] if content else []
            yield shown + live
            if stop.is_set():
                break
        if not calls and not content.strip() and thought.strip() and not stop.is_set():
            # some local models (qwen3.5) write their whole final answer as "thinking" and leave the answer empty:
            # then that text is the answer, shown as one, not folded away among the steps
            content, live = thought.strip(), [{"role": "assistant", "content": thought.strip()}]
        shown += live
        msgs.append({"role": "assistant", "content": content, "tool_calls": [{"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}} for c in calls]})
        if thought and providers.is_api(model):  # DeepSeek wants its reasoning back with the tool results
            msgs[-1]["thinking"] = thought
        if not calls or stop.is_set() or last:
            return
        for c in calls:
            name, args = c.function.name, dict(c.function.arguments)
            box = {"role": "assistant", "content": f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```", "metadata": {"title": f"{ICONS.get(name, '🔧')} {name}", "status": "done"}}
            last = shown[-1] if shown else None  # the same tool called again right away goes in the same box: "🔎 web_search ×4"
            group = last if isinstance(last, dict) and isinstance(last.get("content"), str) and (last.get("metadata") or {}).get("title") == box["metadata"]["title"] else None
            yield shown if group else shown + [box]
            before = pictures()
            try:
                result = str(toolset[name](**args))
            except Exception as e:
                result = f"Error: {e}"
            box["content"] += f"\n**Result**\n```\n{result[:1500]}{'…' if len(result) > 1500 else ''}\n```"
            if group:
                group["content"] += "\n\n---\n\n" + box["content"]
                group["metadata"]["log"] = "×" + str(group["content"].count("\n\n---\n\n") + 1)
            else:
                shown.append(box)
            if name in ("generate_image", "edit_image") and getattr(agent.CTX, "last_image", None):
                shown.append({"role": "assistant", "content": {"path": agent.CTX.last_image}})
                before[agent.CTX.last_image] = float("inf")  # already shown
                agent.CTX.last_image = None
            new = [f for f, when in pictures().items() if when > before.get(f, 0)]  # pictures the tool made or changed (a chart from Python, a saved image…)
            shown += [{"role": "assistant", "content": {"path": f}} for f in sorted(new)[:8]]
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


def think_mode(model, thinking):
    """How to ask for deep thinking. Cloud models are never told think=False: some (glm) then write their reasoning into the answer."""
    if "thinking" not in caps(model) or (not thinking and is_cloud(model)):
        return {}
    return {"think": bool(thinking)}


def podcast_write(model, prompt):
    """A podcast script, piece by piece, from the model chosen on the Podcast page."""
    m = resolve(model, False)
    for chunk in ollama.chat(m, messages=[{"role": "user", "content": prompt}], stream=True, options={"num_ctx": 32768, "temperature": 0.7}, **think_mode(m, False)):
        yield chunk.message.content or ""


def ask_json(model, prompt, schema):
    r = ollama.chat(model, messages=[{"role": "user", "content": prompt}], format=schema, options={"temperature": 0, "num_ctx": 16384}, **think_mode(model, False))
    return json.loads(split_think(r.message.content)[0])


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
    box = {"role": "assistant", "content": t("Checking the answer against the sources…"), "metadata": {"title": "🔍 " + t("Fact check"), "status": "done"}}
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
SYNTH = "Your team of {n} agents worked on my last request independently. Their answers:\n\n{report}\n\nNow write the single best final answer for me. Agreement is not proof: if an important fact isn't backed by a source the agents actually read, check it yourself with web_search or read_page, but keep it to a few checks. Combine their strongest points, resolve disagreements, and drop anything wrong. Write directly to me, in the second person, in the language of my request, as if you had answered alone. Then end with a short section headed '{differ}' (translated into the language of my request): 3 to 5 bullets on where the agents' answers really differed (approach, conclusions, facts, depth), naming each agent by its number and model, and which view you kept and why."


SYNTH_GOAL = " This is a goal, not just a question: finish what is still missing with your tools, check the result, then call goal_done with a short summary (or give_up if it truly can't be done)."


def swarm_agent(k, n, model, thinking, base, box, answers, stop, work, net, timeout, toolset, region):
    agent.set_workspace(work, net, int(timeout), region)
    agent.CTX.vision = model if "vision" in caps(model) else FAST
    msgs = base
    goal = GOAL_RULES in msgs[0]["content"]
    msgs[0] = {"role": "system", "content": msgs[0]["content"].replace(GOAL_RULES, " The user gave a goal: work toward it with your tools, step by step, and end with what you did and the result.") + f" You are agent {k + 1} of {n} in a team answering the same request independently; a reviewer will compare your answers. {STYLES[k % len(STYLES)]} Back every fact with a tool result and say where it comes from. Only create files if the user asked for them. End with your complete answer, addressed to the user."}
    opts = {"num_ctx": 16384, "temperature": TEMPS[k % len(TEMPS)]}
    think = think_mode(model, thinking)
    log = []
    try:
        for step in range(15):
            if stop.is_set():
                break
            last = step == 14  # the last step has no tools, so every agent ends with an answer
            if last:
                msgs.append({"role": "user", "content": WRAP})
            for attempt in range(3):  # cloud servers sometimes answer "Internal Server Error": wait and try again
                try:
                    m = ollama.chat(model, messages=msgs, tools=[] if last else list(toolset.values()), options=opts, **think).message
                    break
                except Exception as e:
                    if attempt == 2 or stop.is_set():
                        raise
                    log.append(f"⚠️ {e} — " + t("trying again"))
                    time.sleep(5 * (attempt + 1))
            calls = [] if last else m.tool_calls or []
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


def run_swarm(job, cid, conv, model, thinking, n, pool, net, timeout, toolset, judge_tools=None, goal=False):
    shown, msgs = conv["display"], conv["llm"]
    work = conv.get("project") or fdir(cid)
    answers, threads = [None] * n, []
    for k in range(n):
        m = pool[k % len(pool)]
        box = {"role": "assistant", "content": tr("Working…"), "metadata": {"title": "🐝 " + tr("Agent {n}", n=k + 1) + f" · {m.removesuffix(':latest')}", "status": "done"}}
        shown.append(box)
        t = threading.Thread(target=swarm_agent, args=(k, n, m, thinking, copy.deepcopy(msgs), box, answers, job.stop, work, net, timeout, toolset, conv.get("region", "wt-wt")), daemon=True)
        t.start()
        threads.append(t)
    while any(t.is_alive() for t in threads):
        job.publish(shown)
        time.sleep(0.5)
    job.publish(shown)
    if job.stop.is_set():
        return []
    report = "\n\n".join(f"### Agent {k + 1} ({pool[k % len(pool)]})\n{a or '(no answer)'}" for k, a in enumerate(answers))
    judge = msgs + [{"role": "user", "content": SYNTH.format(n=n, report=report, differ="How the agents' answers differed") + (SYNTH_GOAL if goal else "")}]
    called = []
    for view in agent_turn(judge, shown, judge_tools or toolset, model, thinking, job.stop, called, limit=10):
        job.publish(view)
    final = next((m["content"] for m in reversed(judge) if m["role"] == "assistant" and m["content"]), "")
    msgs.append({"role": "assistant", "content": final})
    return called


# ---------- teams of agents ----------
# A team chat is answered by the team's lead. When the team has other members, the lead gets one more tool,
# ask_member: it hands a task to a member, who works on it alone (its own instructions, model, tools, goal mode)
# and returns its finished work to the lead. The lead checks it and writes the answer.

LEAD = "\n\nIn this chat you are {name}, {role}, the lead of the team \"{team}\".{purpose} Your own instructions:\n{instructions}"
CREW = "\n\nYour colleagues in the team:\n{crew}\nHand them the parts of the work that fit their roles with ask_member: give each one a complete task with everything they need, because they can't see this conversation. Then check what they return, fix or ask again if something is wrong or missing, and write the final answer for the user yourself. Do a part yourself only when no colleague fits it."
MEMBER = " In this task you are {name}, {role}, a member of the team \"{team}\". The team lead, {lead}, gives you the task below. The lead sees only your final reply, not your tool results, so put everything they need in it. Your instructions:\n{instructions}"


def installed_or(model, fallback):
    return model if model and model in {v for _, v in models()} else fallback


def member_tools(member, base):
    names = {n for g in member.get("tools", []) if g in TOOL_GROUPS for n in TOOL_GROUPS[g][1]}
    return {k: v for k, v in base.items() if k in names}


def member_run(member, task, team, job, shown, chat_model, thinking, base, language):
    """A member works on the lead's task alone, in a box of its own in the chat, and returns its final reply."""
    model = installed_or(member.get("model"), chat_model)
    goal = bool(member.get("goal"))
    toolset = member_tools(member, base)
    if goal:
        toolset.update(goal_done=goal_done, give_up=give_up)
    system = SYSTEM.format(now=agent.get_datetime()) + MEMBER.format(name=member["name"], role=member.get("role") or "-", team=team["name"], lead=teams.lead(team)["name"], instructions=member.get("instructions") or "-") \
        + (GOAL_RULES if goal else "") + (f" Always reply in {language}." if language else "")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": ("GOAL: " if goal else "") + task}]
    box = {"role": "assistant", "content": t("Working…"), "metadata": {"title": f"{member.get('icon') or '👤'} {member['name']} · {model.removesuffix(':latest')}", "status": "done"}}
    shown.append(box)
    job.publish(shown)
    log, answer, finished = [], "", False
    think = think_mode(model, thinking)
    for r in range(3 if goal else 1):  # in goal mode a member gets up to 3 rounds to finish its part
        for step in range(15):
            if job.stop.is_set():
                break
            last = step == 14
            if last:
                msgs.append({"role": "user", "content": WRAP})
            try:
                m = ollama.chat(model, messages=msgs, tools=[] if last or not toolset else list(toolset.values()), options={"num_ctx": 16384}, **think).message
            except Exception as e:
                log.append(f"⚠️ {e}")
                break
            calls = [] if last else m.tool_calls or []
            content = split_think(m.content or "")[0].strip() or ("" if calls else (m.thinking or "").strip())
            msgs.append({"role": "assistant", "content": content, "tool_calls": [{"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}} for c in calls]})
            if not calls:
                answer = content
                break
            for c in calls:
                name, args = c.function.name, dict(c.function.arguments)
                finished = finished or name in ("goal_done", "give_up")
                log.append(f"{ICONS.get(name, '🔧')} {name} `{json.dumps(args, ensure_ascii=False)[:100]}`")
                box["content"] = t("Working…") + "\n\n" + "\n".join(f"- {line}" for line in log)
                job.publish(shown)
                try:
                    result = str(toolset[name](**args)) if name in toolset else f"Error: no tool called {name}"
                except Exception as e:
                    result = f"Error: {e}"
                msgs.append({"role": "tool", "tool_name": name, "content": result})
        if not goal or finished or job.stop.is_set():
            break
        msgs.append({"role": "user", "content": NUDGE})
    steps = "\n".join(f"- {line}" for line in log)
    box["content"] = (steps + "\n\n---\n\n" if steps else "") + (answer or t("(no answer)"))
    job.publish(shown)
    return answer


def run_team(job, conv, chat_model, thinking, base, rounds, language):
    """The lead answers in the chat; the other members work when the lead asks them."""
    team = teams.get(conv.get("team"))
    shown, msgs = conv["display"], conv["llm"]
    head = teams.lead(team)
    model = installed_or(head.get("model"), chat_model)
    goal = bool(head.get("goal"))
    toolset = member_tools(head, base)
    crew = teams.others(team)

    def ask_member(name: str, task: str) -> str:
        """Give a task to a colleague in your team and get their finished work back. They can't see this conversation, so write a complete task with everything they need.

        Args:
            name: The colleague's name
            task: The complete task, with all the details and material the colleague needs
        """
        member = teams.find(team, name)
        if not member:
            return f"No colleague called {name}. Your colleagues: " + ", ".join(m["name"] for m in crew)
        answer = member_run(member, task, team, job, shown, chat_model, thinking, base, language)
        return f"{member['name']}'s work:\n\n{answer}" if answer else f"{member['name']} returned no answer."

    if crew:
        toolset["ask_member"] = ask_member
    if goal:
        toolset.update(goal_done=goal_done, give_up=give_up)
    msgs[0]["content"] += LEAD.format(name=head["name"], role=head.get("role") or "-", team=team["name"], purpose=f" The team's purpose: {team['purpose']}." if team.get("purpose") else "", instructions=head.get("instructions") or "-") \
        + (CREW.format(crew="\n".join(f"- {m['name']}: {m.get('role') or '-'}" for m in crew)) if crew else "") + (GOAL_RULES if goal else "")
    if goal and msgs[-1]["role"] == "user" and not msgs[-1]["content"].startswith("GOAL: "):
        msgs[-1]["content"] = "GOAL: " + msgs[-1]["content"]
    rounds = int(rounds) if goal else 1
    for r in range(1, rounds + 1):
        called = []
        for view in agent_turn(msgs, shown, toolset, model, thinking, job.stop, called):
            job.publish(view)
        if job.stop.is_set():
            shown.append({"role": "assistant", "content": t("You stopped this task. Send a message to continue."), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
            return
        if not goal or {"goal_done", "give_up"} & set(called):
            return
        if r == rounds:
            shown.append({"role": "assistant", "content": t("The goal wasn't marked as done after {n} rounds. Send a message to continue.", n=rounds), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
            return
        shown.append({"role": "assistant", "content": t("Not marked as done yet, continuing."), "metadata": {"title": "🔁 " + t("Round {i} of {n}", i=r + 1, n=rounds), "status": "done"}})
        msgs.append({"role": "user", "content": NUDGE})
        job.publish(shown)


def team_chat(team_id):
    """A new chat with a team: it starts with who is in it."""
    team = teams.get(team_id)
    if not team:
        gr.Warning(t("Pick a team first."))
        return None
    cid = new_conv()
    conv = load(cid)
    head = teams.lead(team)
    who = "\n".join(f"- {m.get('icon') or '👤'} **{m['name']}**" + (f" · {m['role']}" if m.get("role") else "") + (" · " + t("leads") if m is head else "") + (" · 🎯" if m.get("goal") else "") for m in team["members"])
    intro = f"{team.get('icon') or '👥'} **{team['name']}**" + (f"  \n{team['purpose']}" if team.get("purpose") else "") + "\n\n" + who + "\n\n" + (
        t("Write the team's task below. {lead} reads it, hands parts of the work to the others and writes the answer.", lead=head["name"]) if teams.others(team) else t("Write the task below: {lead} works on it.", lead=head["name"]))
    conv.update(team=team_id, title=f"{team.get('icon') or '👥'} {team['name']}", auto_title=True, display=[{"role": "assistant", "content": intro}])
    save(cid, conv)
    return cid


# ---------- long-term memory ----------

EXTRACT = """You keep long-term memory notes about the user of an AI assistant. The current notes:

{notes}

The latest exchange:
USER: {user}
ASSISTANT: {answer}

Does the USER message reveal lasting facts worth knowing in future chats: who they are, their work, ongoing projects, stable preferences about how they want answers? Rules: save only what the user explicitly stated, never guesses or things the assistant said; skip anything temporary, trivial, or already in the notes; never save passwords, ID or card numbers, or health details. Most exchanges contain nothing to save.

Reply with JSON only, like {{"save": [{{"topic": "profile", "fact": "<one short fact, in the user's language>"}}]}}. Use topic "profile" for personal facts and preferences, or a short lowercase topic name for a specific project. Reply {{"save": []}} if there is nothing to save."""


def notes_model(model):
    """Who writes the memory notes after a cloud answer: the fast local model if it's installed, else the smallest
    local model you have (the quickest), else the cloud model itself. (Before, it was always the fast model named in
    config.json, and when that model wasn't installed, no note was ever written.)"""
    try:
        plain = scratch.in_joshua()  # the tiny models trained here can't write notes: they'd always be the smallest
        local = {m.model: m.size or 0 for m in ollama.list().models if not is_cloud(m.model) and m.model.removesuffix(":latest") not in plain}
    except Exception:
        local = {}
    for name in (FAST, MODEL):
        if name in local:
            return name
    return min(local, key=local.get) if local else model


def extract_memory(model, user, answer):
    try:
        notes = "\n\n".join(f"## {t}\n{agent.read_memory(t)}" for t in agent.memory_topics()) or "(empty)"
        prompt = EXTRACT.format(notes=notes[:6000], user=user[:3000], answer=answer[:2000])
        r = ollama.chat(model, messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0}, **think_mode(model, False))
        for item in json.loads(split_think(r.message.content)[0]).get("save", [])[:5]:
            if isinstance(item, dict) and str(item.get("fact", "")).strip():
                agent.remember(str(item["fact"]), str(item.get("topic") or "profile"))
    except Exception as e:
        print("memory notes:", model, e)  # in joshua.log, so a failure can be seen


def worker(job, cid, conv, model, thinking, goal, rounds, net, timeout, swarm=0, pool=(), mem=True, user_text="", offline=False, private=False):
    jlog.note(f"answer started · chat {cid} · {model}" + (" · goal mode" if goal else "") + (f" · swarm of {swarm}" if swarm else "") + f" · message of {len(user_text or '')} characters")
    net = net and not offline
    agent.set_workspace(conv.get("project") or fdir(cid), net, int(timeout), conv.get("region", "wt-wt"))
    agent.CTX.vision = model if "vision" in caps(model) else FAST
    shown, msgs = conv["display"], conv["llm"]
    begin = len(shown)  # what this turn adds starts here
    off = {n for g in tools_off() for n in TOOL_GROUPS.get(g, ("", []))[1]}
    at_desk = desks.get(conv.get("opts", {}).get("desk"))
    if at_desk:  # a desk uses only its own tools: fewer tools, steadier small models
        off |= {n for g, (_, names) in TOOL_GROUPS.items() if g not in at_desk["tools"] for n in names}
    base = {k: v for k, v in tools.items() if k not in off and (mem and not private or k not in agent.MEMORY_TOOLS) and not (offline and (k in ("web_search", "read_page") or k in agent.BROWSER_TOOLS))}
    toolset = dict(base, goal_done=goal_done, give_up=give_up) if goal else base
    rounds = int(rounds) if goal else 0 if swarm else 1
    language = agent.LANGUAGES.get(conv.get("lang"))
    evidence = asked = None
    role = conv.pop("agent", None) or {}  # a monitor's run: what to search for, and what its sites and feeds said
    checking = goal and conv.get("opts", {}).get("desk") == "journalism" and factcheck.asked(user_text)  # the fact checker
    in_team = teams.get(conv.get("team"))
    plain = model.removesuffix(":latest") in scratch.in_joshua()
    try:
        if plain:  # a small model trained here from zero: no tools, no web, no memory; it just gets your words
            question = user_text or msgs[-1]["content"]
            lang = scratch.in_joshua().get(model.removesuffix(":latest"), {}).get("search")
            source = ""
            if lang:  # it learned to look things up: it writes what to search for, Joshua searches Wikipedia, it answers from that
                box = {"role": "assistant", "content": "", "metadata": {"title": "🔎 " + t("Searching Wikipedia…"), "status": "pending"}}
                shown.append(box)
                job.publish(shown)
                ask = lambda text, stop: ollama.generate(model, prompt=text, raw=True, options={"num_predict": 24, "stop": stop, "temperature": 0.3}).response
                framed, query, title = scratch.job.searched(lang, question, ask)
                box["metadata"] = {"title": "🔎 " + (t("Searched Wikipedia for “{query}”: {title}", query=query, title=title) if title else t("Searched Wikipedia for “{query}”: nothing found", query=query) if query
                                                   else t("Nothing to look up: it answers on its own")), "status": "done"}
                if title:
                    source = f"\n\n[{title}](https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))})"
            reply = {"role": "assistant", "content": ""}
            shown.append(reply)
            stream = ollama.generate(model, prompt=framed, raw=True, stream=True, options={"num_predict": 240, "stop": scratch.job.STOP}) if lang else \
                ollama.generate(model, prompt=question, stream=True, options={"num_predict": 300})
            for chunk in stream:
                reply["content"] += chunk.response
                job.publish(shown)
                if job.stop.is_set():
                    break
            reply["content"] = (reply["content"].strip() or "…") + source
            msgs.append({"role": "assistant", "content": reply["content"]})
            rounds, swarm, in_team = 0, 0, None
        if in_team:  # a team chat: the lead answers, and the members work when the lead asks them
            run_team(job, conv, model, thinking, base, rounds, language)
            rounds = 0
        if not plain and not in_team and not offline and user_text and "web_search" not in off:
            box = {"role": "assistant", "content": "", "metadata": {"title": "🔎 " + t("Looking online…"), "status": "done"}}

            def show():  # the box appears only once Joshua has decided to search
                if not any(b is box for b in shown):
                    shown.append(box)
                job.publish(shown)

            evidence = research(role.get("query") or user_text, msgs, model, language, conv.get("region", "wt-wt"), box, show)
            box["metadata"] = {"title": "🔎 " + t("Sources from the web"), "status": "done"}
            if evidence:
                asked, original = msgs[-1], msgs[-1]["content"]
                asked["content"] = original + USE_SOURCES.format(evidence=evidence, language=language or "my language")
            elif any(b is box for b in shown):
                shown.remove(box)
            job.publish(shown)
        if role.get("evidence"):  # the items read in a monitor's sites and feeds count as sources for the fact check
            evidence = (evidence + "\n\n" if evidence else "") + role["evidence"]
        first = 1
        if swarm:  # the team works first; in goal mode Joshua then keeps going, round after round, until the goal is done
            called = run_swarm(job, cid, conv, model, thinking, int(swarm), pool, net, timeout, base, toolset, goal)
            if job.stop.is_set():
                shown.append({"role": "assistant", "content": t("You stopped this task. Send a message to continue."), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
                rounds = 0
            elif goal and not {"goal_done", "give_up"} & set(called) and rounds > 1:
                first = 2
                shown.append({"role": "assistant", "content": t("Not marked as done yet, continuing."), "metadata": {"title": "🔁 " + t("Round {i} of {n}", i=2, n=rounds), "status": "done"}})
                msgs.append({"role": "user", "content": NUDGE})
                job.publish(shown)
            else:
                rounds = 0
        for r in range(first, rounds + 1):
            called = []
            for view in agent_turn(msgs, shown, toolset, model, thinking, job.stop, called):
                job.publish(view)
            save(cid, conv)
            if job.stop.is_set():
                shown.append({"role": "assistant", "content": t("You stopped this task. Send a message to continue."), "metadata": {"title": "⏹ " + t("Stopped"), "status": "done"}})
                break
            missing = factcheck.gaps((last_answer(shown) or {}).get("content", ""), user_text) if checking and "goal_done" in called else []
            if missing and r < rounds:  # Joshua said it's done, but the table doesn't pass the check: one more round, with what's missing
                shown.append({"role": "assistant", "content": "\n".join("- " + m for m in missing), "metadata": {"title": "🔁 " + t("Not finished: the check found gaps (round {i} of {n})", i=r + 1, n=rounds), "status": "done"}})
                msgs.append({"role": "user", "content": CHECK_GAPS.format(gaps="\n".join("- " + m for m in missing))})
                job.publish(shown)
                continue
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
        if evidence and asked:
            asked["content"] = original + "\n\n(Web sources consulted: " + ", ".join(re.findall(r"^https?://\S+", evidence, re.M)) + ")"
    except Exception as e:
        jlog.error(f"while answering in chat {cid} with {model}", e)
        text = str(e)
        low = text.lower()
        if is_cloud(model) and not providers.is_api(model) and ("unauthorized" in low or "401" in low):
            text = t("This cloud model needs your computer to be connected to an Ollama account. Open **Models** in the side menu, go to **Open-weight models · in the cloud** and press **Connect this computer to Ollama**.")
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
        for m in shown[begin:]:  # "saved as plan.pdf": the name becomes a link to the file
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and not m.get("metadata"):
                m["content"] = link_files(m["content"], conv.get("project") or fdir(cid))
        save(cid, conv)
        job.ended = time.time()
        job.running = False
        jlog.note(f"answer finished · chat {cid} · {int(job.ended - job.started) if getattr(job, 'started', None) else '?'} s")
        job.publish(shown)
    if conv.get("auto_title") and load(cid).get("auto_title") and user_text and not job.stop.is_set():  # the first answer is in: give the chat a title that sums it up
        answer = (last_answer(shown) or {}).get("content", "")
        title = make_title(FAST if plain else model, user_text, answer if isinstance(answer, str) else "", language) if answer else ""  # a tiny model can't write titles
        if title and in_team:  # a team's chats keep its icon in front
            title = (in_team.get("icon") or "👥") + " " + title
        if title:
            conv["title"], conv["auto_title"] = title, False
            save(cid, conv)
    if job.stop.is_set():
        QUEUE.pop(cid, None)  # Stop also cancels the messages waiting in line
    elif QUEUE.get(cid):
        try:
            send(*QUEUE[cid].pop(0))  # the next message in line, with the settings it was sent with
        except Exception as e:
            print("queued message failed:", e)
    if mem and learn_setting() and user_text and not job.stop.is_set():
        answer = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), "")
        # memory notes are written by a model on this computer when there is one, even after a cloud answer
        # (and after an answer from a small model trained here from zero, which can't write them, by a regular one)
        plain = model.removesuffix(":latest") in scratch.in_joshua()
        writer = notes_model(model) if is_cloud(model) or plain else resolve(conv.get("model") or model, False) if conv.get("model") == PAIR else model
        if writer and writer.removesuffix(":latest") not in scratch.in_joshua() and (not is_cloud(writer) or share_memory_setting()):  # no local model: learn only if you let cloud models see your notes
            extract_memory(writer, user_text, answer)


def busy(cid):
    return cid in JOBS and JOBS[cid].running


def send(message, cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem=False, desk="", *team):
    message = message or {}
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        return gr.skip(), *SKIP
    if busy(cid):  # like Claude: the message waits in line and goes when Joshua has finished
        QUEUE.setdefault(cid, []).append((message, cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem, desk, *team))
        gr.Info(t("Queued: Joshua will send it when it has finished."))
        return gr.skip(), None, gr.skip(), gr.skip(), gr.skip()
    if model == ADD:  # the menu is still on "＋ Add a model…"
        model = default_model()
    choice, model = model, resolve(model, thinking)
    picked = [resolve(v, thinking) if v and v != ADD else "" for v in (list(team) + [""] * TEAM)[:int(agents)]] if swarm else []
    cloud = is_cloud(model)
    if (cloud or any(is_cloud(m) for m in picked if m)) and offline:
        gr.Warning(t("Cloud models need the internet. Please choose a local model, or turn off No internet."))
        return gr.skip(), *SKIP
    cid = cid or new_conv()
    conv = load(cid)
    if teams.get(conv.get("team")):  # a team chat: each member has its own goal mode and tools; desks and swarm don't apply
        goal = swarm = False
        desk, picked = "", []
        team_cloud = any(is_cloud(installed_or(m.get("model"), model)) for m in teams.get(conv["team"])["members"])  # memory notes stay private then
    conv["model"] = choice
    conv["opts"] = dict(zip(OPTS, map(bool, (thinking, goal, swarm, offline))), agents=int(agents), team=[v or "" for v in (list(team) + [""] * TEAM)[:TEAM]], desk=desk or "")
    if cloud and conv.get("cloud_notice") != model:  # say it once in the chat, each time a new cloud model is picked
        conv["cloud_notice"] = model
        where = (t("From here on this chat is answered by **{model}** on {company}'s servers: each message sends the conversation there, including the text of files and pages Joshua reads.", model=model, company=providers.company(model))
                 if providers.is_api(model) else t("From here on this chat is answered by **{model}** on Ollama's servers: each message sends the conversation there, including the text of files and pages Joshua reads.", model=model))
        conv["display"].append({"role": "assistant", "content": "☁ *" + where + ("" if cloudmem else " " + t("Your memory notes are not sent.")) + " " + t("Everything stays stored only on this computer.") + "*"})
    if desk == "falken" and not conv["display"]:
        conv["ephemeral"] = True  # a game isn't kept: deleted when you leave it (forget_games)
    if desk == "falken" and text and not files:  # Prof Falken: a game move is played by falken.py, at once, without a model
        reply = falken.play(conv, text)
        if reply is not None:
            conv["display"] += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            conv["llm"] += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            if not conv["title"]:
                conv["title"] = "Shall we play a game?"
            save(cid, conv)
            return present(conv["display"]), None, cid, gr.Radio(choices=conv_choices(), value=cid), file_list(cid)
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
        shown.append({"role": "user", "content": ("🐝 " if swarm else "") + ("🎯 " if goal else "") + text})
    if not conv["title"]:  # for now the first words; after the first answer, a short title that sums up the chat
        conv["title"], conv["auto_title"] = cap((text or names[0])[:48]), True
    where = f" Your workspace is the user's project folder {work}; use list_files to explore it and edit files with care." if conv.get("project") else ""
    online = " You have no internet access in this chat: you cannot search or read web pages, so answer from what you know and say when something may be out of date." if offline else "" if net else " Code you run has no internet access."
    online += f" The user writes in {language}: always reply in {language}." if language else ""
    at_desk = desks.get(desk)
    system = {"role": "system", "content": SYSTEM.format(now=agent.get_datetime()) + where + online + (GOAL_RULES if goal else "") + (agent.memory_prompt() if mem and (cloudmem or not cloud) else "")
              + ("\n\n" + at_desk["instructions"] if at_desk else "")}
    if msgs and msgs[0]["role"] == "system":
        msgs[0] = system
    else:
        msgs.insert(0, system)
    note = f"\n\n(Files saved in the workspace: {', '.join(names)})" if names else ""
    msgs.append({"role": "user", "content": ("GOAL: " if goal else "") + text + note, "images": images})
    pool = [model] + [m for m in dict.fromkeys(resolve(v, thinking) for _, v in models()) if m != model and (cloud or not is_cloud(m))]  # "Automatic": a local chat never borrows a cloud model
    auto = iter(pool * TEAM)
    team = [m or next(auto) for m in picked]  # the agents you gave a model use it; the others take the menu's models in turn
    cloud = cloud or any(is_cloud(m) for m in team) or teams.get(conv.get("team")) is not None and team_cloud  # memory notes stay private if any agent runs in the cloud
    save(cid, conv)
    JOBS[cid] = job = Job(shown)
    threading.Thread(target=worker, args=(job, cid, conv, model, thinking, goal, rounds, net, timeout, int(agents) if swarm else 0, team, mem, text, offline, cloud and not cloudmem), daemon=True).start()
    return present(shown, True), None, cid, gr.Radio(choices=conv_choices(), value=cid), file_list(cid)


def stop_chat(cid):
    QUEUE.pop(cid, None)
    if busy(cid):
        JOBS[cid].stop.set()


# ---------- sidebar actions ----------

def files_label(cid):
    project = cid and load(cid).get("project")
    return gr.File(value=file_list(cid), label=t("Project: {name}", name=pathlib.Path(project).name) if project else t("Files in this chat"))


def forget_games(keep=None):
    """Prof Falken chats aren't kept: once you leave one (another chat, a new chat, or Joshua restarts), it's deleted."""
    for p in ROOT.iterdir():
        if p.name != keep and (p / "chat.json").exists() and not busy(p.name):
            try:
                if json.loads((p / "chat.json").read_text(encoding="utf-8")).get("ephemeral"):
                    shutil.rmtree(p, ignore_errors=True)
                    JOBS.pop(p.name, None)
            except Exception:
                pass


def open_conv(cid):
    forget_games(keep=cid)
    conv = load(cid)
    view = present(JOBS[cid].view, True) if busy(cid) else present(conv["display"])
    used = conv.get("model")
    return (view, cid, files_label(cid), conv.get("project", ""), used if used in [v for _, v in models()] else gr.skip()) + chat_opts(cid)


def new_chat():
    forget_games()
    return [], None, gr.Radio(choices=conv_choices(), value=None), files_label(None), "", *chat_opts(None)


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


def choose_project(current):
    """The computer's own folder window, opened by Joshua (it runs on this computer), so nobody has to type a path.
    On a Mac it's the Finder's; elsewhere Tk's, when Python has it. Cancelled: the field stays as it was."""
    start = pathlib.Path((current or "").strip()).expanduser()
    if sys.platform == "darwin":
        where = f" default location (POSIX file {json.dumps(str(start))})" if start.is_dir() else ""
        script = f"POSIX path of (choose folder with prompt {json.dumps(t('Choose the project folder for this chat'))}{where})"
        try:
            r = subprocess.run(["osascript", "-e", "activate", "-e", script], capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired):
            return gr.skip()
        return r.stdout.strip().rstrip("/") if r.returncode == 0 and r.stdout.strip() else gr.skip()
    try:
        import tkinter, tkinter.filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = tkinter.filedialog.askdirectory(initialdir=str(start) if start.is_dir() else str(pathlib.Path.home()), title=t("Choose the project folder for this chat"))
        root.destroy()
    except Exception:
        gr.Info(t("This computer can't open a folder window from Joshua: type the folder's path and press Enter."))
        return gr.skip()
    return chosen or gr.skip()


def upload_files(paths, cid):
    cid = cid or new_conv()
    for p in map(pathlib.Path, paths or []):
        if p.parent != wdir(cid):
            shutil.copy(p, wdir(cid) / p.name)
    return file_list(cid), cid, gr.Radio(choices=conv_choices(), value=cid)


def log_view():
    """The last lines of the log, the most recent at the bottom; errors marked."""
    lines = [l for l in jlog.tail(250) if not any(noise in l for noise in LOG_NOISE)]
    return "\n".join(lines) or t("Nothing in the log yet.")


LOG_NOISE = ("MallocStackLogging", "To create a public link", "Running on local URL")  # harmless lines from macOS and from Gradio's start: they only hide what matters  # harmless lines macOS prints for every program: they only hide what matters


def log_folder():
    folder = str(jlog.PATH.parent)
    try:
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
    except Exception as e:
        gr.Warning(str(e))


def client_log(text: str) -> str:
    """What goes wrong in the page (in the browser) is written in the log too."""
    jlog.note("page · " + (text or "")[:2000])
    return "ok"


def open_folder(cid):
    """Shows the chat's files folder (or its project folder) in Finder, or the file manager on Linux."""
    if cid:
        wdir(cid).mkdir(parents=True, exist_ok=True)
        if agent.WIN:
            os.startfile(str(wdir(cid)))
        else:
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
    asked = [k for k, m in enumerate(llm) if m["role"] == "user" and m["content"] not in (NUDGE, WRAP)]
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

    def label(name):  # "profile · 3 notes", so topics other than the one shown can be seen
        p = agent._mem(name)
        n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) if p.exists() else 0
        return (t("About you") if name == "profile" else name) + " · " + t("{n} notes", n=n)
    return gr.Dropdown(choices=[(label(x), x) for x in topics], value=topic, visible=len(topics) > 1), path.read_text() if path.exists() else ""  # one topic: nothing to choose


def notes_count():
    return sum(read.count("\n- ") + read.startswith("- ") for read in (agent._mem(x).read_text(encoding="utf-8") for x in agent.memory_topics()))


def learn_past(topic):
    """Reads the chats you've already had, one by one, and keeps the lasting facts about you, as after each reply.
    Each chat is read once (it's marked "learned"). Chats by programs, scheduled runs and games are skipped."""
    writer = notes_model(default_model())
    if is_cloud(writer) and not share_memory_setting():
        gr.Warning(t("There is no local model to read your chats, and your notes aren't shared with cloud models. Download a local model, or switch on 'Share notes with cloud models'."))
        yield gr.skip(), gr.skip(), gr.skip()
        return
    todo = []
    for p in ROOT.iterdir():
        if not (p / "chat.json").exists():
            continue
        c = load(p.name)
        if c.get("learned") or c.get("opts", {}).get("desk") == "falken" or str(c.get("title", ""))[:1] in ("⏰", "📡", "🔌"):
            continue
        said = [m["content"] for m in c.get("display", []) if m.get("role") == "user" and isinstance(m.get("content"), str)]
        if said:
            todo.append((p.name, "\n\n".join(said)))
    before, n = notes_count(), len(todo)
    if not n:
        yield t("All your chats have already been read."), *mem_view(topic)
        return
    for k, (cid, said) in enumerate(todo, 1):
        yield t("Reading your chats: {k} of {n} · {found} new notes so far", k=k, n=n, found=notes_count() - before), gr.skip(), gr.skip()
        extract_memory(writer, said[-3000:], "")
        c = load(cid)
        c["learned"] = True
        write_chat(cid, c)  # not save(): reading a chat doesn't move it to the top
    yield t("Done: {n} chats read, {found} new notes.", n=n, found=notes_count() - before), *mem_view(topic)


def mem_save(topic, text):
    if text is None:  # you cancelled clearing the notes
        return gr.skip(), gr.skip()
    with agent.MEM_LOCK:
        path = agent._mem(topic or "profile")
        path.write_text(text.strip() + "\n") if text.strip() else path.unlink(missing_ok=True)
    gr.Info(t("Memory saved.") if text.strip() else t("Deleted the '{topic}' notes.", topic=topic))
    return mem_view(topic)


STEPS = {**{k: t("using the browser") for k in agent.BROWSER_TOOLS}, "web_search": t("searching the web"), "read_page": t("reading a web page"), "run_python": t("running code"), "install_package": t("installing a package"),
         "list_files": t("browsing files"), "read_file": t("reading a file"), "write_file": t("writing a file"), "look_at_image": t("looking at an image"),
         "generate_image": t("generating an image (a minute or two)"), "edit_image": t("editing the image (several minutes)"), "remember": t("updating memory"),
         "forget": t("updating memory"), "read_memory": t("reading memory"), "get_datetime": t("checking the date"), "goal_done": t("wrapping up"), "give_up": t("wrapping up"),
         t("Looking online…"): t("researching: searching and reading sources"), t("Sources from the web"): t("writing the answer"), t("Fact check"): t("fact-checking the answer"), t("Thinking"): t("thinking")}


def status_html(cid):
    job = JOBS.get(cid)
    if not job:
        return ""
    if not job.running:  # finished: say so, with the time it took, until the next message
        secs = int((job.ended or time.time()) - job.started)
        label = "⏹ " + t("STOPPED") if job.stop.is_set() else "✓ " + t("DONE")
        return f"<div class='done'>{label} · {secs // 60}:{secs % 60:02d}</div>"
    step, last = t("starting"), job.view[-1] if job.view else {}
    meta = last.get("metadata") or {} if isinstance(last, dict) else {}
    if meta.get("title"):
        name = meta["title"].split(" ", 1)[-1]
        step = t("running a swarm of agents") if name.startswith(t("Agent {n}", n="").strip()) else STEPS.get(name, name.lower())
    elif isinstance(last, dict) and last.get("role") == "assistant":
        step = t("writing the answer")
    secs = int(time.time() - job.started)
    if step == t("starting") and secs >= 4:  # nothing has come back yet: usually the model is still being loaded into memory
        step = t("loading the model into memory: the first message can take up to a minute")
    waiting = QUEUE.get(cid) or []
    queued = f" · <span class='queued'>{t('{n} queued', n=len(waiting))}</span>" if waiting else ""
    return f"<div class='working'><span class='wdot'></span>{t('JOSHUA IS WORKING')} · <b class='clock'>{secs // 60}:{secs % 60:02d}</b> · {html.escape(step)}{queued}</div>"


def machine():
    """The kind of computer, for the feedback form: chip and memory, nothing that identifies it."""
    try:
        if sys.platform == "darwin":
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
            return f"{chip}, {ram} GB, macOS {platform.mac_ver()[0]}"
        ram = round(agent.total_ram())
        if agent.WIN:
            return f"Windows {platform.release()} {platform.machine()}, {ram} GB"
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
        conv["title"], conv["auto_title"] = name[:80], False  # your own name is never replaced
        write_chat(cid, conv)  # not save(): a new name doesn't move the chat to the top
    return gr.Textbox(visible=False), gr.Radio(choices=conv_choices(), value=cid)


def delete_chat(cid):
    nothing = (gr.skip(),) * (12 + TEAM)
    if not cid or not (ROOT / cid).exists():
        return nothing
    if busy(cid):
        gr.Warning(t("Joshua is still answering in this chat. Press Stop first, then you can delete it."))
        return nothing
    conv = load(cid)
    conv["deleted_at"] = time.time()
    write_chat(cid, conv)
    BIN.mkdir(exist_ok=True)
    shutil.rmtree(BIN / cid, ignore_errors=True)
    shutil.move(str(ROOT / cid), str(BIN / cid))
    JOBS.pop(cid, None)
    gr.Info(t("Chat moved to the bin: you can restore it there for {n} days.", n=BIN_DAYS))
    return new_chat()


SEEN = {}

# ---------- the bin: deleted chats wait here for 30 days, then they're gone ----------

BIN = ROOT / ".bin"  # inside the chats folder: it moves and is backed up with your chats, and updates never touch it
BIN_DAYS = 30


def bin_items():
    """The chats in the bin, the most recently deleted first: (id, title, deleted at)."""
    out = []
    for p in BIN.iterdir() if BIN.exists() else []:
        try:
            c = json.loads((p / "chat.json").read_text(encoding="utf-8"))
            out.append((p.name, cap(c.get("title") or t("New chat")), c.get("deleted_at") or p.stat().st_mtime))
        except Exception:
            pass
    return sorted(out, key=lambda x: -x[2])


def bin_purge():
    for cid, _, when in bin_items():
        if time.time() - when > BIN_DAYS * 86400:
            shutil.rmtree(BIN / cid, ignore_errors=True)


def bin_view():
    items = bin_items()
    act = lambda kind, cid, label, cls="": f"<button type='button' class='binact {cls}' data-act='{kind}:{html.escape(cid, quote=True)}'>{html.escape(label)}</button>"
    rows = "".join(f"<div class='task binrow'><div class='binmain'><b>{html.escape(title)}</b><br><span>{t('deleted {when}', when=time.strftime('%d/%m %H:%M', time.localtime(when)))}"
                   f" · {t('gone in {n} days', n=max(0, BIN_DAYS - int((time.time() - when) // 86400)))}</span></div>"
                   f"<div class='binbtns'>{act('restore', cid, '↩ ' + t('Restore'), 'primary')}{act('forever', cid, t('Delete for good'), 'red')}</div></div>" for cid, title, when in items)
    listing = rows or "<div class='cloudstat'>" + t("The bin is empty.") + "</div>"
    return listing, gr.Dropdown(choices=[(title, cid) for cid, title, _ in items], value=None), gr.Column(visible=bool(items))


def bin_restore(cid):
    if cid and (BIN / cid).exists():
        shutil.move(str(BIN / cid), str(ROOT / cid))
        conv = load(cid)
        conv.pop("deleted_at", None)
        write_chat(cid, conv)
        gr.Info(t("Restored: the chat is back in the list."))
    return bin_view() + (gr.Radio(choices=conv_choices()),)


def bin_row(act):
    """A button in one of the bin's rows: 'restore:<chat>' or 'forever:<chat>' (that one already confirmed in the page)."""
    kind, _, cid = (act or "").partition(":")
    if kind == "restore":
        return bin_restore(cid)
    if kind == "forever":
        return bin_forever(cid) + (gr.skip(),)
    return (gr.skip(),) * 4


def bin_forever(cid):
    if cid:
        shutil.rmtree(BIN / cid, ignore_errors=True)
    return bin_view()


def bin_empty(go):
    if go:
        shutil.rmtree(BIN, ignore_errors=True)
    return bin_view()


# ---------- the Teams page and its wizard ----------

ICON_CHOICES = list({svg: e for e, svg in reversed(json.loads((pathlib.Path(__file__).parent / "icons.json").read_text(encoding="utf-8"))["emoji"].items()) if svg != "busy"}.values())[::-1]  # one emoji for each icon Joshua draws


def icon_pick(label=None, value="", placeholder="", **kw):
    """An icon field: pick from the icons Joshua draws, or type any emoji."""
    return gr.Dropdown(ICON_CHOICES, value=value or None, label=label, allow_custom_value=True, filterable=True, **kw)

NEW_MEMBER = {"name": "", "icon": "", "role": "", "instructions": "", "model": "", "tools": [], "goal": False}
STEPS_WIZ = ("The team", "Members", "Lead", "Review")


def team_list_html():
    cards = []
    for team in teams.all():
        head = teams.lead(team)
        who = " · ".join(html.escape((m.get("icon") or "👤") + " " + m["name"]) + (" ★" if m is head else "") for m in team["members"])
        cards.append(f"<div class='task'><b>{html.escape((team.get('icon') or '👥') + ' ' + team['name'])}</b>"
                     + (f"<br><span>{html.escape(team['purpose'])}</span>" if team.get("purpose") else "") + f"<br><span>{who}</span></div>")
    return "".join(cards) or "<div class='cloudstat'>" + t("No teams yet. Press “New team” to create the first one.") + "</div>"


def team_page(pick=None):
    """The list of teams (the wizard closed)."""
    choices = [((tm.get("icon") or "👥") + " " + tm["name"], tm["id"]) for tm in teams.all()]
    return (team_list_html(), gr.Dropdown(choices=choices, value=pick if pick in {v for _, v in choices} else choices[0][1] if choices else None), gr.Column(visible=bool(choices)),
            gr.Column(visible=True), gr.Column(visible=False))


def team_delete(team_id):
    if team_id:
        teams.delete(team_id)
        gr.Info(t("Team deleted. Its chats stay in the list."))
    return team_page()


def wiz_render(state, step, sel):
    """Every field of the wizard, drawn from its state: which step is shown, and the member being edited."""
    members = state["members"]
    m = members[sel] if 0 <= sel < len(members) else dict(NEW_MEMBER, tools=[k for k in TOOL_GROUPS if k not in ("browser", "install_package")])
    head = "<div class='wsteps'>" + "".join(f"<span class='wstep{' on' if k + 1 == step else ' done' if k + 1 < step else ''}'>{k + 1}. {t(name)}</span>" for k, name in enumerate(STEPS_WIZ)) + "</div>"
    listing = t("Describe the first member below. A team can have just one. Next saves it for you.") if not members else t("Your members are listed here: pick one to change it. The form below adds another one; Next saves it for you.")
    lead = state.get("lead") if state.get("lead") in {x["id"] for x in members} else (members[0]["id"] if members else None)
    review = ""
    if step == 4:
        rows = "".join(f"<div class='task'><b>{html.escape((x.get('icon') or '👤') + ' ' + x['name'])}</b>" + (" · " + t("leads") if x["id"] == lead else "")
                       + f"<br><span>{html.escape(x.get('role') or '')}</span><br><span>"
                       + html.escape(" · ".join(filter(None, [x.get("model") or t("Same model as the chat"), "🎯 " + t("Goal mode") if x.get("goal") else "",
                                                              t("{n} tools", n=len(x.get("tools", [])))]))) + "</span></div>" for x in members)
        review = f"<div class='task'><b>{html.escape((state.get('icon') or '👥') + ' ' + state['name'])}</b><br><span>{html.escape(state.get('purpose') or '')}</span></div>" + rows
    return (state, step, sel,
            *(gr.Column(visible=step == k) for k in (1, 2, 3, 4)), head,
            gr.Button(visible=step > 1), gr.Button(visible=step < 4), gr.Button(visible=step == 4, value="✓ " + (t("Save team") if state.get("id") else t("Create team"))),
            gr.Textbox(value=state.get("name", ""), info=None, elem_classes=[]), state.get("icon") or "👥", state.get("purpose", ""),
            listing, gr.Radio(choices=[((x.get("icon") or "👤") + " " + x["name"], k) for k, x in enumerate(members)] + [("＋ " + t("New member"), -1)], value=sel),
            gr.Textbox(value=m["name"], info=None, elem_classes=[]), m.get("icon") or "👤", m.get("role", ""), m.get("instructions", ""),
            gr.Dropdown(choices=[(t("Same model as the chat"), "")] + models(), value=m.get("model", "")), m.get("tools", []), bool(m.get("goal")),
            gr.Button(value=t("Save member") if sel >= 0 else "＋ " + t("Save and add another member")), gr.Button(visible=sel >= 0),
            gr.Radio(choices=[((x.get("icon") or "👤") + " " + x["name"], x["id"]) for x in members], value=lead), review)


def wiz_collect(state, step, sel, name, icon, purpose, m_name, m_icon, m_role, m_instr, m_model, m_tools, m_goal, lead):
    """What you typed in the step you're on goes into the wizard's state."""
    state = copy.deepcopy(state)
    if step == 1:
        state.update(name=(name or "").strip(), icon=(icon or "").strip()[:4], purpose=(purpose or "").strip())
    elif step == 2 and (m_name or "").strip():
        member = {"name": m_name.strip(), "icon": (m_icon or "").strip()[:4], "role": (m_role or "").strip(), "instructions": (m_instr or "").strip(),
                  "model": m_model or "", "tools": list(m_tools or []), "goal": bool(m_goal)}
        if 0 <= sel < len(state["members"]):
            state["members"][sel] = dict(state["members"][sel], **member)
        elif len(state["members"]) < teams.MAX_MEMBERS:
            state["members"].append(dict(member, id=secrets.token_hex(3)))
            sel = len(state["members"]) - 1
    elif step == 3 and lead:
        state["lead"] = lead
    return state, sel


def wiz_open(team_id=None):
    team = teams.get(team_id)
    state = copy.deepcopy(team) if team else {"id": None, "name": "", "icon": "", "purpose": "", "lead": None, "members": []}
    return wiz_render(state, 1, -1) + (gr.Column(visible=False), gr.Column(visible=True))


def wiz_invalid(shown, i, message, value=""):
    """The wizard as drawn, with the name field at position i marked as missing and the reason written under it."""
    shown = list(shown)
    shown[i] = gr.Textbox(value=value, info="⚠ " + message, elem_classes=["invalid"])
    return tuple(shown)


def wiz_move(delta, *values):
    state, step = values[0], values[1]
    state, sel = wiz_collect(*values)
    if delta > 0 and step == 1 and not state["name"]:  # the message goes under the field that's missing, not in a corner
        return wiz_invalid(wiz_render(state, step, sel), 11, t("Give the team a name."))
    if delta > 0 and step == 2 and not state["members"]:
        return wiz_invalid(wiz_render(state, step, sel), 16, t("Add at least one member."), values[6] or "")
    step += delta
    if step == 3 and len(state["members"]) == 1:  # one member: it leads, nothing to choose
        step += delta
    return wiz_render(state, step, -1)


def wiz_pick(*values):
    """Another member picked in the list: the one you were editing is kept first."""
    state, sel = wiz_collect(*values[:-1])
    return wiz_render(state, 2, values[-1] if values[-1] is not None else -1)


def wiz_member_save(*values):
    state, sel = wiz_collect(*values)
    if not (values[6] or "").strip():
        return wiz_invalid(wiz_render(state, 2, sel), 16, t("Give the member a name."), "")
    return wiz_render(state, 2, -1)


def wiz_member_remove(*values):
    state, sel = copy.deepcopy(values[0]), values[2]
    if 0 <= sel < len(state["members"]):
        del state["members"][sel]
    return wiz_render(state, 2, -1)


def wiz_finish(*values):
    state, _ = wiz_collect(*values)
    if not state["name"] or not state["members"]:
        gr.Warning(t("Give the team a name and at least one member."))
        return (gr.skip(),) * 5
    team_id = teams.save(state)
    gr.Info(t("Team saved: pick it and press “Start a chat”."))
    return team_page(team_id)


def team_open_chat(team_id):
    new_id = team_chat(team_id)
    if not new_id:
        return (gr.skip(),) * (len(OPENED) + 4)
    forget_games(keep=new_id)
    return open_conv(new_id) + (gr.Radio(choices=conv_choices(), value=new_id), gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=True))


# ---------- the Reader ----------

def rd_title(title):
    """Titles in capitals (often a file's name) are shown in ordinary case."""
    return title[:1] + title[1:].lower() if title.isupper() and len(title) > 12 else title


def rd_library():
    """The library (no book open): your books, the last read first. A title opens the book; ✕ removes it."""
    rows = []
    for b in reader.books():
        pct = reader.percent(b.get("pos", 0), b["count"])
        src = b.get("file") or ""
        kind = (urllib.parse.urlparse(src).netloc.removeprefix("www.") if src.startswith("http") else pathlib.Path(src).suffix.lstrip(".").upper()) if src else t("Text")
        rows.append(f"<div class='rdrow' data-open='{b['id']}' role='button' tabindex='0' title='{html.escape(rd_title(b['title']), quote=True)}'><span class='rdplayicon' aria-hidden='true'>▶</span><div class='rdmain'><div class='rdname'>{html.escape(rd_title(b['title']))}</div>"
                    f"<div class='rdmeta'>{html.escape(kind)} · " + t("{p}% read", p=pct) + f"</div><div class='rdprogress'><span style='width:{pct}%'></span></div></div>"
                    f"<button class='rdx' data-remove='{b['id']}' title='" + html.escape(t("Remove from the library"), quote=True) + "'>✕</button></div>")
    listing = "<div class='rdlib'>" + "".join(rows) + "</div>" if rows else "<div class='rdempty'>" + t("No books yet.") + "</div>"
    return listing, gr.Column(visible=True), gr.Column(visible=False), ""


def rd_show(book_id):
    book = reader.get(book_id)
    if not book:
        return (gr.skip(),) * 4
    reader.set_pos(book_id, book.get("pos", 0))  # now the last opened
    book["title"] = rd_title(book["title"])
    return gr.skip(), gr.Column(visible=False), gr.Column(visible=True), RD_BAR + reader.page(book)  # one block, so the bar stays on top while you scroll


def rd_add(path):
    if not path:
        return (gr.skip(),) * 4
    try:
        book_id = reader.add(path, pathlib.Path(path).name)
    except Exception as e:
        gr.Warning(t("Joshua couldn't find text to read in this file.") + (f" ({e})" if str(e) != "no text" else ""))
        return (gr.skip(),) * 4
    return rd_show(book_id)


def rd_paste(value):
    """A web address or pasted text, added to the library and opened."""
    value = (value or "").strip()
    if not value:
        return (gr.skip(),) * 5
    try:
        book_id = reader.add_url(value) if reader.is_url(value) else reader.add_text(value)
    except Exception as e:
        gr.Warning((t("Joshua couldn't read that page.") if reader.is_url(value) else t("Joshua couldn't find text to read in this file.")) + (f" ({e})" if str(e) != "no text" else ""))
        return (gr.skip(),) * 5
    return rd_show(book_id) + ("",)


def rd_action(value):
    """A click in the library list: open:<id> or remove:<id>."""
    act, _, book_id = (value or "").partition(":")
    if act == "open":
        return rd_show(book_id)
    if act == "remove":
        reader.remove(book_id)
        return rd_library()
    return (gr.skip(),) * 4


def rd_save(value):
    """The browser reports each paragraph it finishes: that's your place in the book."""
    book_id, _, rest = (value or "").partition("|")
    pos, _, settings = rest.partition("|")
    if book_id and pos.isdigit():
        try:
            settings = json.loads(settings) if settings else None
        except ValueError:
            settings = None
        reader.set_pos(book_id, int(pos), settings)


# ---------- the Mac's own voices ----------
# Browsers show web pages only some of the Mac's voices: Firefox none of the Premium ones. So on a Mac Joshua speaks
# with macOS itself: 'say' turns a paragraph into a small audio file, and the page plays it. That works in any browser,
# and on the phone too. Elsewhere the browser's voices are used.

SAY = os.environ.get("JOSHUA_SAY") or (shutil.which("say") if sys.platform == "darwin" else None)  # JOSHUA_SAY: a stand-in for tests
TTS_DIR = ROOT / ".tts"  # recent audio, so a paragraph read twice is ready at once
NOVELTY = {"Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Good News", "Jester", "Junior", "Organ",
           "Superstar", "Ralph", "Trinoids", "Whisper", "Zarvox", "Fred", "Kathy"}


@functools.lru_cache(maxsize=1)
def mac_voices():
    """The Mac's voices for reading: name and language (en_US), without the novelty ones."""
    if not SAY:
        return []
    try:
        out = subprocess.run([SAY, "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Za-z0-9]+)\s+#", line)
        if m and re.sub(r"\s*\(.*\)$", "", m.group(1).strip()) not in NOVELTY:
            voices.append({"name": m.group(1).strip(), "lang": m.group(2).replace("_", "-")})
    return voices


def tts_voices() -> list:
    """For the page: the Mac's voices, or nothing (then the browser's are used)."""
    return mac_voices()


def tts(text: str, voice: str) -> str:
    """A paragraph spoken by the Mac, as the address of an audio file for the page."""
    names = {v["name"] for v in mac_voices()}
    text = (text or "").strip()[:4000]
    if not SAY or not text or voice not in names:
        return ""
    TTS_DIR.mkdir(exist_ok=True)
    key = hashlib.sha1((voice + "\n" + text).encode()).hexdigest()[:20]
    out = TTS_DIR / f"{key}.m4a"
    if not out.exists():
        src = TTS_DIR / f"{key}.txt"
        src.write_text(text, encoding="utf-8")  # read from a file: the text never becomes a command-line option
        tmp = TTS_DIR / f"{key}.part.m4a"
        try:
            subprocess.run([SAY, "-v", voice, "-f", str(src), "-o", str(tmp), "--file-format=m4af", "--data-format=aac"], timeout=120, check=True,
                           capture_output=True)
            tmp.replace(out)
        except Exception:
            return ""
        finally:
            src.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
        files = sorted(TTS_DIR.glob("*.m4a"), key=lambda f: f.stat().st_mtime)
        for f in files[:-400]:  # keep the latest 400 paragraphs
            f.unlink(missing_ok=True)
    return "/gradio_api/file=" + urllib.parse.quote(out.as_posix(), safe="/:")



SIMPLER = """Rewrite this {language} paragraph so it is easy to follow when heard read aloud. Write it in {language}.
Keep the meaning and every piece of information: add nothing, leave nothing out, no comments, no explanations.
Use short main clauses, one idea per sentence. Each sentence starts with its subject and verb.
Turn subordinate clauses into sentences of their own. Link them with a simple word (and, but, so, then) only where the original has that link.
No sentence may begin with a word like "although", "while", "if" or "since": state it plainly instead.
Say everything once: don't repeat a sentence and don't sum up at the end.
Keep the author's words wherever you can. Keep quotations exactly as they are, with the spoken quotation marks around them.
Reply with the rewritten {language} paragraph only.

Example (English):
Paragraph: Although the theory, which was first proposed in 1950 by researchers who worked at Bell Labs, was soon forgotten, it returned when computers became fast enough to test it.
Rewritten: The theory was first proposed in 1950. It came from researchers at Bell Labs. It was soon forgotten. But it returned when computers became fast enough to test it.

Paragraph: {text}
Rewritten in {language}:"""
LANG_NAMES = {"en": "English", "it": "Italian", "fr": "French", "de": "German", "pl": "Polish", "zh": "Chinese", "es": "Spanish", "pt": "Portuguese", "nl": "Dutch",
              "ru": "Russian", "ja": "Japanese", "uk": "Ukrainian", "ro": "Romanian", "sv": "Swedish", "ca": "Catalan", "cs": "Czech", "el": "Greek"}


def rd_text(book_id: str, i: int, mode: str) -> str:
    """The paragraph as it's read aloud: as written, without citations, or in simpler sentences."""
    book = reader.get(book_id)
    if not book or not 0 <= int(i) < len(book["paras"]):
        return ""
    text = book["paras"][int(i)]
    if mode == "raw":
        return text
    text = reader.clean(text, book.get("lang") or "")
    if mode != "simple" or len(text.split()) < 18:  # a short paragraph is easy already
        return text
    key = f"simple:{int(i)}"
    done = reader.easy_get(book_id, key)
    if done:
        return done
    model = default_model()
    try:
        language = LANG_NAMES.get((book.get("lang") or "")[:2], "the same language as the paragraph")  # named: otherwise models drift into English
        out = ollama.chat(model, messages=[{"role": "user", "content": SIMPLER.format(text=text, language=language)}], options={"num_ctx": 4096, "temperature": 0.2},
                          **think_mode(model, False)).message.content or ""
        out = split_think(out)[0].strip().strip('"«»“”')
    except Exception as e:
        print("simpler sentences:", e, flush=True)
        return text
    if not out or not 0.6 < len(out) / max(len(text), 1) < 1.6:  # far shorter or longer: something went wrong, read the paragraph as it is
        return text
    reader.easy_put(book_id, key, out)
    return out

LIVE_EVERY = 1.5  # seconds between redraws of a chat while Joshua writes it


def live_view(cid, request: gr.Request):
    """The chat Joshua is writing in, redrawn every few seconds (and once more at the end)."""
    seen = SEEN.setdefault(request.session_hash, {})
    job = JOBS.get(cid)
    sig = (cid, job.version) if job else None
    if not job or seen.get("view") == sig:
        return gr.skip()
    seen["view"] = sig
    return present(job.view, job.running)


def poll(cid, request: gr.Request):
    """Called every second by each open page: sends only what changed (chat list, files, status), and switches on
    the chat's own timer while Joshua writes in it. The chat isn't an output here: Gradio re-processes every output
    of an event, even unchanged ones, and with a long chat that kept the page busy several times a second."""
    seen = SEEN.setdefault(request.session_hash, {})
    out = []
    job = JOBS.get(cid)
    want = bool(job and (job.running or seen.get("view") != (cid, job.version)))  # until the final screen is shown
    out.append(gr.Timer(active=want) if seen.get("live") != want else gr.skip())
    seen["live"] = want
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
body:not([data-screen="green"]):not([data-screen="amber"]):not([data-screen="modern"]):not([data-screen="modern-night"]):not([data-screen="modern-paper"]):not([data-screen="modern-paper-night"]):not([data-screen="blue"]):not([data-screen="black"]):not([data-screen="white"]) {--bg: #dfdbc3; --bg-hi: #e9e5cf; --panel: #e7e3cc; --codebg: #d4cfb3; --line: rgba(90, 60, 40, .45); --cyan: #9c3b1c; --glow: transparent; --ink: #3b2322; --dim: #7a6250; --amber: #9c3b1c; --red: #b3261e; --codeink: #3b2322; --shadow: none}
body[data-screen="black"] {--bg: #000308; --bg-hi: #040a14; --panel: #050a14; --codebg: #02050a; --line: #8fe9ff; --cyan: #8fe9ff; --glow: rgba(80, 200, 255, .55); --ink: #a6e6ff; --dim: #5d8ea6; --amber: #ffb347; --red: #ff5a4e; --codeink: #59f08a; --shadow: 0 0 8px rgba(80, 200, 255, .55); --check: #ffb347}
body[data-screen="black"], body[data-screen="green"] {--bw: 2px; --panel-shadow: 0 0 8px var(--glow), inset 0 0 18px rgba(80, 200, 255, .12); --case: uppercase; --tglow: 0 0 3px rgba(120, 220, 255, .35); --flick: flick 3s infinite; --pbw: 2px; --pline: var(--amber); --pglow: 0 0 8px rgba(255, 179, 71, .35)}
body[data-screen="black"]::after, body[data-screen="green"]::after {content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 9999; background: repeating-linear-gradient(to bottom, rgba(0, 0, 0, .22) 0, rgba(0, 0, 0, .22) 1px, transparent 1px, transparent 3px)}
body {--bw: 1px; --panel-shadow: 0 0 8px var(--glow); --case: none; --tglow: none; --flick: none; --pbw: 1px; --pline: color-mix(in srgb, var(--amber) 70%, transparent); --pglow: 0 0 5px color-mix(in srgb, var(--amber) 25%, transparent)}
body[data-screen="white"] {--bg: #ffffff; --bg-hi: #ffffff; --panel: #fbfbfb; --codebg: #f1f1f1; --line: rgba(0, 0, 0, .26); --cyan: #000000; --glow: transparent; --ink: #141414; --dim: #6b6b6b; --amber: #1f5fbf; --red: #c4271d; --codeink: #141414; --shadow: none}
body[data-screen="green"] {--bg: #000300; --bg-hi: #010801; --panel: #020802; --codebg: #010501; --line: #33ff66; --cyan: #7dff8a; --glow: rgba(60, 255, 110, .5); --ink: #7dff8a; --dim: #3c9a4a; --amber: #d6ffb8; --red: #ff6b5e; --codeink: #7dff8a; --shadow: 0 0 8px rgba(60, 255, 110, .5); --check: #d6ffb8}  /* Black, in the green of WOPR's screens */
body[data-screen="green"] {--panel-shadow: 0 0 8px var(--glow), inset 0 0 18px rgba(60, 255, 110, .1); --tglow: 0 0 3px rgba(90, 255, 130, .35); --pglow: 0 0 8px rgba(214, 255, 184, .3)}
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
body {--brand: #3f8a49}  /* the logo's green; lighter on the dark screens */
body[data-screen="black"], body[data-screen="green"], body[data-screen="amber"], body[data-screen="blue"], body[data-screen="modern-night"], body[data-screen="modern-paper-night"] {--brand: #7fb86a}
#brand, body:not(#a):not(#b) #brand {font-family: 'VT323', monospace !important; font-size: 3.2rem !important; font-weight: 400 !important; letter-spacing: .1em; text-transform: uppercase !important; color: var(--brand) !important; padding: 26px 4px 0 18px !important; text-shadow: none; line-height: 1; white-space: nowrap}
body:not(#a):not(#b) #brand svg {display: inline-block !important; width: 5.3rem !important; height: 5.3rem !important; max-width: none !important; vertical-align: -.73rem !important; margin: -1.96rem -1.74rem 0 -.81rem !important}  /* the logo is the J, as tall as the letters */
#brand::after, body:not(#a):not(#b) #brand::after {content: none !important}  /* the cursor now shows when Joshua is working */
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
#convs label {position: relative}
body:not(#a):not(#b):not(#c) #convs label {padding-right: 34px !important}  /* room for the ⋯, so the name never runs under it */
#convs .cmore {position: absolute; right: 6px; top: 50%; transform: translateY(-50%); padding: 0 6px; border-radius: 6px; opacity: 0; color: var(--dim); font-weight: 700; line-height: 1.4}
#convs label:hover .cmore, #convs label.selected .cmore {opacity: 1}
#convs .cmore:hover {background: color-mix(in srgb, var(--ink) 12%, transparent); color: var(--ink)}
button.cmore.cur {background: transparent; border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px; color: var(--ink); cursor: pointer; font: inherit}
#renchat, #delchat {display: none !important}
#chatbar {justify-content: flex-end !important; align-items: center !important; gap: 6px !important; flex-wrap: wrap !important; min-height: 0 !important; padding: 0 4px !important; background: transparent !important; border: none !important; flex-grow: 0 !important}
#chatbar > * {flex: 0 0 auto !important; min-width: 0 !important; width: auto !important}
body:not(#a):not(#b):not(#c) #chatbar .opt {padding: 2px 10px !important; margin: 0 !important; width: auto !important; overflow: visible !important}
body:not(#a):not(#b):not(#c) #chatbar .opt, body:not(#a):not(#b):not(#c) #chatbar .opt * {cursor: pointer}
#chatbar .opt .info-text {display: none !important}
body:not(#a):not(#b):not(#c) #chatbar .opt {border: 1px solid color-mix(in srgb, var(--brand) 45%, transparent) !important; border-radius: 999px !important; padding: 2px 10px !important; background: color-mix(in srgb, var(--brand) 8%, transparent) !important}
#chatbar .opt:not(:has(input:checked)) {display: none !important}
/* goal and swarm under the message box: small switches, filled when on */
body:not(#a):not(#b):not(#c) #modelrow > .form:has(.modeopt) {display: flex !important; flex-direction: row !important; gap: 6px !important; flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important}
body:not(#a):not(#b):not(#c) #modelpick > #cloudbadge {flex: 1 1 auto !important; min-width: max-content !important; text-align: right}
body:not(#a):not(#b):not(#c) #modelrow .cloudbadge {white-space: nowrap !important; overflow: hidden; padding: 2px 0 0 !important}
body:not(#a):not(#b):not(#c) #modelrow {gap: 8px !important}
.cloudbadge > span + span::before {content: " · "; color: var(--dim)}
body:not(#a):not(#b):not(#c) #modelrow .modeopt {flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin: 0 !important; padding: 2px 11px !important; border: 1px solid color-mix(in srgb, var(--line) 80%, transparent) !important; border-radius: 999px !important; background: transparent !important; box-shadow: none !important; cursor: pointer; overflow: visible !important}
body:not(#a):not(#b):not(#c) #modelrow .modeopt:has(input:checked) {border-color: color-mix(in srgb, var(--brand) 55%, transparent) !important; background: color-mix(in srgb, var(--brand) 12%, transparent) !important}
#modelrow .modeopt input {display: none !important}
#modelrow .modeopt .info-text, #modelrow .modeopt [data-testid="block-info"], #modelrow .modeopt .info {display: none !important}
#modelrow .modeopt label, #modelrow .modeopt label span {font-size: .82rem !important; margin: 0 !important; padding: 0 !important; color: var(--dim); cursor: pointer}
#modelrow .modeopt:has(input:checked) label span {color: var(--brand)}  /* the ⋯ menu sets them; only the ones that are on show, as small labels */
#chatbar .opt input {display: none !important}
#chatbar .opt [data-testid="block-info"], #chatbar .opt .info {display: none !important}
#chatbar .opt label, #chatbar .opt label span {font-size: .82rem !important; margin: 0 !important; padding: 0 !important}
#chatbar .html-container, #chatbar .block {padding: 0 !important; border: none !important; background: transparent !important; box-shadow: none !important}
#chatbar button.cmore.cur {width: auto; border: none; font-size: 1.3rem; line-height: 1; padding: 2px 10px; color: var(--dim)}
#chatbar button.cmore.cur:hover {color: var(--ink); background: color-mix(in srgb, var(--ink) 8%, transparent)}
body:not([data-files="open"]) #chatopts {display: none !important}
#chatopts {flex-grow: 0 !important; margin: 0 4px 8px !important; padding: 10px 12px !important; border-radius: 12px; background: color-mix(in srgb, var(--brand) 6%, var(--bg)) !important}
#chatopts .fhead {font-weight: 600; color: var(--brand)}
.cloudbadge .bicon {margin: 0 1px}
body:not(#a):not(#b):not(#c) span.ji {display: inline-flex !important; width: 1.2em !important; height: 1.2em !important; vertical-align: -.22em !important; flex: none !important}
body:not(#a):not(#b):not(#c) #modelrow .modeopt label span.ji, body:not(#a):not(#b):not(#c) #deskpick label span.ji {margin: 0 .35em 0 0 !important; padding: 0 !important}
body:not(#a):not(#b):not(#c) #convs label span.ji {margin-right: .4em !important; opacity: .75}
#starters {flex: 0 0 auto !important; padding: 0 !important; background: transparent !important; border: none !important}
#chatcol:not(:has(#chat .placeholder)) #starters {display: none !important}  /* only on an empty chat */
.starters {display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 0 auto 10px; max-width: 760px}
button.starter {display: flex; align-items: center; gap: 8px; text-align: left; padding: 9px 14px; border-radius: 12px; border: 1px solid color-mix(in srgb, var(--line) 80%, transparent); background: color-mix(in srgb, var(--bg) 70%, white); color: var(--ink); font-size: .9rem; cursor: pointer}
button.starter:hover {border-color: var(--brand); background: color-mix(in srgb, var(--brand) 8%, transparent)}
button.starter .sicon {font-size: 1.1rem}
body:not(#a):not(#b):not(#c) #deskpick {flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; padding: 0 !important; border: none !important; background: transparent !important; box-shadow: none !important}
body:not(#a):not(#b):not(#c) #deskpick .wrap {display: flex !important; flex-wrap: nowrap !important; gap: 4px !important}
body:not(#a):not(#b):not(#c) #deskpick label {padding: 2px 10px !important; border-radius: 999px !important; border: 1px solid color-mix(in srgb, var(--line) 80%, transparent) !important; background: transparent !important; box-shadow: none !important; font-size: .82rem !important; min-height: 0 !important; cursor: pointer; color: var(--dim)}
body:not(#a):not(#b):not(#c) #deskpick label.selected, body:not(#a):not(#b):not(#c) #deskpick label:has(input:checked) {border-color: color-mix(in srgb, var(--brand) 55%, transparent) !important; background: color-mix(in srgb, var(--brand) 12%, transparent) !important; color: var(--brand)}
body:not(#a):not(#b):not(#c) #deskpick input {display: none !important}
body.narrow:not(#a):not(#b) #deskpick {order: 0; flex: 1 1 100% !important}
.cloudbadge .bcost, .cloudbadge .bcredit {white-space: nowrap}
.cloudbadge .bcredit {cursor: default}
.cloudbadge .bplace {cursor: default}
.cloudbadge .bplace + span::before {content: " "}
#fbtnwrap {flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; align-self: center !important; padding: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important}
#fbtnwrap .html-container {padding: 0 !important}
button.fbtn {display: flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 12px; border: none; background: transparent; color: var(--dim); cursor: pointer}
body:not(#a):not(#b) button.fbtn svg {width: 22px !important; height: 22px !important; min-width: 22px; flex: none; display: block}
button.fbtn:hover {color: var(--brand); background: color-mix(in srgb, var(--brand) 10%, transparent)}
body[data-files="open"] button.fbtn {color: var(--brand); background: color-mix(in srgb, var(--brand) 14%, transparent)}
.modepanel {flex-grow: 0 !important; margin: 0 4px 8px !important; padding: 10px 12px !important; border-radius: 12px; gap: 6px !important; border: none !important; box-shadow: none !important; background: color-mix(in srgb, var(--brand) 6%, var(--bg)) !important}
#swarmpanel .fhead {font-weight: 600; color: var(--brand)}
.modepanel .block {background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important}
#teamrow {flex-wrap: wrap !important; gap: 8px !important; background: transparent !important; border: none !important}
#teamrow > .teampick {flex: 1 1 150px !important}
.pinsvg {width: .95em; height: .95em; fill: var(--brand); vertical-align: -.12em; flex: none}
#convs label .pinsvg {margin-right: 6px}
#convs label.dropto {box-shadow: inset 0 2px 0 var(--brand) !important}
#convs label.pinned {cursor: grab}
.renfield {position: fixed; z-index: 1001; box-sizing: border-box; font: inherit; font-size: .9rem; padding: 2px 8px; border: 1.5px solid var(--brand); border-radius: 6px; background: var(--bg); color: var(--ink); outline: none}
#convs label span:not(.cmore) {flex: 1 1 auto !important; min-width: 0 !important; display: block !important; text-overflow: clip !important; -webkit-mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent); mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent)}  /* long names fade out, so no "…" next to the ⋯ */
#cmenu {position: fixed; z-index: 1000; min-width: 210px; padding: 6px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line); box-shadow: 0 8px 24px rgba(0, 0, 0, .18); display: flex; flex-direction: column}
#cmenu button {all: unset; padding: 7px 10px; border-radius: 6px; cursor: pointer; color: var(--ink); font-size: .92rem}
#cmenu button:hover {background: color-mix(in srgb, var(--cyan) 14%, transparent)}
#cmenu button b {color: var(--brand); float: right}
#cmenu button.danger {color: var(--red)}
#cmenu hr {border: none; border-top: 1px solid var(--line); margin: 4px 2px}
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
#modelrow {justify-content: space-between; align-items: center; margin-top: -6px; padding-right: 24px}
#working {flex: 1 1 auto !important; min-width: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 0 0 4px !important}
#working .working {display: flex; align-items: center; gap: 8px; color: var(--cyan); font-size: 1.15rem; letter-spacing: .05em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
#working .progress-text, #working .eta-bar, #working .timer, #working .meta-text, #working .meta-text-center, #working .progress-level {display: none !important}
#working .wdot {flex: 0 0 auto; width: 11px; height: 19px; background: repeating-linear-gradient(to bottom, var(--brand) 0 2px, color-mix(in srgb, var(--brand) 55%, #b8f0a0) 2px 3px); box-shadow: 0 0 6px color-mix(in srgb, var(--brand) 45%, transparent); animation: slowblink 2.6s ease-in-out infinite}  /* the logo's cursor: slowly blinking while Joshua works */
@keyframes slowblink {0%, 100% {opacity: 1} 50% {opacity: .15}}
#working .done {display: flex; align-items: center; gap: 8px; color: var(--brand); font-size: 1.15rem; letter-spacing: .05em; white-space: nowrap}
#working .working b.clock, #working .done {font-variant-numeric: tabular-nums}
#working .working b.clock {display: inline; font-weight: 600; color: var(--ink)}
#working .queued {color: var(--brand)}
#renchat {justify-content: flex-start}
#delchat {justify-content: flex-start; border: 1px dashed color-mix(in srgb, var(--red) 60%, transparent) !important; background: transparent !important; color: var(--red) !important; font-size: 1rem !important; margin-top: 6px}
#delchat:hover {background: color-mix(in srgb, var(--red) 10%, transparent) !important}
#model {max-width: 230px; background: transparent !important; border: none !important; box-shadow: none !important}
#model input {font-size: 1.1rem !important; text-align: right; padding-right: 20px !important; color: var(--dim); cursor: pointer; text-transform: uppercase; letter-spacing: .08em}
#model .wrap {background: transparent !important; border: none !important; box-shadow: none !important}
#tunebtn {justify-content: flex-start; border: 1px solid var(--cyan) !important; background: color-mix(in srgb, var(--cyan) 8%, transparent) !important; color: var(--cyan) !important; font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: .06em; margin-top: 6px}
#modelsbtn {justify-content: flex-start; border: 1px solid var(--cyan) !important; background: color-mix(in srgb, var(--cyan) 8%, transparent) !important; color: var(--cyan) !important; font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: .06em; margin-top: 6px}
#tunebtn:hover {background: color-mix(in srgb, var(--cyan) 10%, transparent) !important}
#modelsbtn:hover {background: color-mix(in srgb, var(--cyan) 10%, transparent) !important}
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
.ftstatus .wdot, .ftres .wdot {display: inline-block; width: 9px; height: 15px; margin-right: 8px; background: var(--cyan); vertical-align: -2px; animation: blink 1s steps(1) infinite}
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
/* the fine-tuning page: one step at a time */
#tune .ftlead {opacity: .75; margin: -4px 0 4px}
#tune .ftempty {opacity: .6; padding: 22px 2px}
.ftwarn {font-size: 1.2rem; padding: 6px 10px; margin-top: 4px; border-left: 3px solid #c0392b; background: var(--codebg); color: var(--ink)}
#tune #ftproject, #tune #ftnewrow, #tune .ftadd {gap: 8px !important; align-items: end !important}
#tune .ftpane {padding: 10px 2px 4px}
#tune details.ftmore {margin-top: 14px}
#tune details.ftmore summary {cursor: pointer; color: var(--brand); font-size: .92rem; list-style: none}
#tune details.ftmore summary::before {content: "▸ "; display: inline-block; transition: transform .15s}
#tune details.ftmore[open] summary::before {transform: rotate(90deg)}
#tune details.ftmore > :not(summary) {opacity: .9}
#tune details.ftmore p, #tune details.ftmore li {margin: .45em 0}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsteps {border: none !important; background: transparent !important; padding: 0 !important; margin-top: 8px}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsteps .wrap {display: flex !important; gap: 0 !important; flex-wrap: wrap !important; border-bottom: 1px solid var(--line)}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsteps label {border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; background: transparent !important;
  padding: 8px 14px !important; margin: 0 0 -1px !important; color: inherit !important; opacity: .65; box-shadow: none !important}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsteps label.selected {opacity: 1; border-bottom-color: var(--brand) !important; color: var(--brand) !important; font-weight: 600}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsteps label.selected * {color: var(--brand) !important}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsteps input {display: none !important}
body:not(#a):not(#b):not(#c):not(#d) #tune button {min-height: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b):not(#c):not(#d) #tune button.ftmain {background: var(--brand) !important; color: var(--panel) !important; border-color: var(--brand) !important}
body[data-screen^="modern"]:not(#a):not(#b):not(#c):not(#d) #tune .prose, body[data-screen^="modern"]:not(#a):not(#b):not(#c):not(#d) #tune .prose * {font-size: 1rem !important; line-height: 1.5 !important}
body[data-screen^="modern"]:not(#a):not(#b):not(#c):not(#d) #tunetitle {font-size: 1.15rem !important; letter-spacing: .06em}
#ftgrid .cell-menu-button {display: none !important}
#tune .ftchart {margin-top: 14px}
#tune .ftchart-head {font-size: .9rem; opacity: .8; margin-bottom: 4px}
#tune .ftchart-head .k1 {color: #4a7fb5; margin-left: 10px} #tune .ftchart-head .k2 {color: #e8892a; margin-left: 6px}
#tune .ftchart svg {width: 100%; height: auto; display: block}
#tune .ftchart .grid line {stroke: var(--line); stroke-width: 1}
#tune .ftchart text {fill: currentColor; opacity: .6; font-size: 11px}
#tune .ftchart polyline {fill: none; stroke-width: 2; stroke-linejoin: round; vector-effect: non-scaling-stroke}
#tune .ftchart .tr {stroke: #4a7fb5; fill: #4a7fb5} #tune .ftchart polyline.tr {fill: none}
#tune .ftchart .va {stroke: #e8892a; fill: #e8892a} #tune .ftchart polyline.va {fill: none}
#tune .ftchart .ol {stroke: #8a8f8c; fill: #8a8f8c} #tune .ftchart polyline.ol {fill: none; stroke-dasharray: 5 4}
#tune .ftphases {margin-top: 8px; display: flex; flex-direction: column; gap: 6px}
#tune .ftphase {padding: 6px 10px; border-left: 3px solid var(--line); font-size: .95rem}
#tune .ftphase b {display: block; font-weight: 600}
#tune .ftphase span {display: block; opacity: .8; font-size: .88rem; overflow-wrap: anywhere}
#tune .ftphase i {display: block; font-style: normal; opacity: .6; font-size: .84rem; margin-top: 2px; line-height: 1.45}
#tune .ftphase.done {border-left-color: var(--green)} #tune .ftphase.now {border-left-color: var(--cyan)} #tune .ftphase.todo {opacity: .5}
#tune .ftphase .wdot {display: inline-block; width: 8px; height: 13px; background: var(--cyan); vertical-align: -1px; animation: blink 1s steps(1) infinite}
#tune .ftwhere {margin-top: 4px; padding: 8px 10px; font-size: .88rem; line-height: 1.5; background: var(--codebg); overflow-wrap: anywhere}
#tune .ftchart-head .k3 {color: #8a8f8c; margin-left: 6px}
#tune .ftchart-last, #tune .ftchart-empty {font-size: .85rem; opacity: .65; margin-top: 2px}
#tune .ftchart-empty {padding: 18px 0}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftsize input {display: none !important}
#tune .ftsamples {margin-top: 10px} #tune .ftsample {font-size: .88rem; padding: 6px 10px; border-left: 2px solid var(--line); margin: 4px 0; white-space: pre-wrap; opacity: .9}
#tune .ftsample b {color: var(--cyan); font-weight: 600; margin-right: 6px}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftmode {margin: 4px 0 10px; border: none !important; background: transparent !important; padding: 0 !important}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftmode .wrap {display: flex !important; gap: 0 !important; flex-wrap: wrap !important; border-bottom: 1px solid var(--line)}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftmode label {border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; background: transparent !important;
  padding: 8px 14px !important; margin: 0 0 -1px !important; color: inherit !important; opacity: .65; box-shadow: none !important}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftmode label.selected {opacity: 1; border-bottom-color: var(--brand) !important; color: var(--brand) !important; font-weight: 600}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftmode label.selected * {color: var(--brand) !important}
body:not(#a):not(#b):not(#c):not(#d) #tune #ftmode input {display: none !important}
body:not(#a):not(#b):not(#c):not(#d) #tune {overflow-x: hidden !important; flex-wrap: nowrap !important}  /* allowed to wrap, a page taller than the window put its lower half in a second column, off screen */
#tune .ftpane .row, #tune #ftproject, #tune #ftnewrow {flex-wrap: wrap !important; row-gap: 8px !important; max-width: 100% !important}
#tune .ftpane .row > .form, #tune .ftpane .row > .block:not(button) {min-width: 0 !important}
body:not(#a):not(#b):not(#c):not(#d) #tune .ftpane button, body:not(#a):not(#b):not(#c):not(#d) #tune #ftproject button, body:not(#a):not(#b):not(#c):not(#d) #tune #ftnewrow button {flex: none !important; white-space: nowrap !important; width: auto !important; padding: 0 16px !important; height: 38px !important}
#tune .ftsave {align-items: center !important}
#tune .ftsave .ftcount {margin: 0 !important}
#modelrow {flex-wrap: wrap !important; row-gap: 4px !important}
#modelrow > #working {flex: 0 0 auto !important; width: auto !important; min-width: 0 !important}
#modelpick > #cloudbadge {flex: 1 1 0 !important; width: auto !important; min-width: 8ch !important}
#cloudbadge .cloudbadge {white-space: normal; overflow: visible; line-height: 1.25}  /* a narrow window wraps it onto a second line instead of cutting it */
#modelpick > #model {flex: 0 0 auto !important}
body:not(#a):not(#b):not(#c) #modelrow #model {min-width: 0 !important; width: auto !important}
body:not(#a):not(#b):not(#c) #modelrow #model input {field-sizing: content; min-width: 12ch; max-width: 30ch}  /* the menu is as wide as the model's name */
#cloudbadge .cloudbadge {font-size: 1rem; line-height: 1.2; color: var(--amber); padding: 2px 8px 0; text-align: right}
.cloudstat {font-size: 1.05rem; line-height: 1.3; padding: 6px 0; color: var(--ink)}
#localhelp, #localhelp *, #apihelp, #apihelp *, #schedhelp, #schedhelp * {font-size: 1.05rem !important; line-height: 1.3 !important}
.task {font-size: 1.15rem; line-height: 1.25; padding: 8px 0; border-bottom: 1px dashed var(--line); color: var(--ink)}
.taskp {color: var(--dim); font-size: 1rem; margin-top: 3px}
#schedbtn {justify-content: flex-start; border: 1px solid var(--cyan) !important; background: color-mix(in srgb, var(--cyan) 8%, transparent) !important; color: var(--cyan) !important; font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: .06em}
#schedpanel {background: color-mix(in srgb, var(--panel) 88%, transparent) !important; border: var(--bw) solid var(--line) !important; border-radius: 4px !important; box-shadow: var(--panel-shadow) !important; padding: 14px 18px !important; gap: 12px}
#modelspage {background: color-mix(in srgb, var(--panel) 88%, transparent) !important; border: var(--bw) solid var(--line) !important; border-radius: 4px !important; box-shadow: var(--panel-shadow) !important; padding: 14px 18px !important; gap: 12px}
#schedtitle {font-size: 2rem; letter-spacing: .08em; color: var(--cyan); text-shadow: var(--shadow)}
#schedback {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
#modelsback {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
.schedsub {font-size: 1.2rem; letter-spacing: .08em; color: var(--cyan); padding-bottom: 4px; border-bottom: 1px dashed var(--line)}
#schedpanel .prose, #schedpanel .prose * {font-size: 1.25rem !important; line-height: 1.3 !important}
#modelspage .prose, #modelspage .prose * {font-size: 1.25rem !important; line-height: 1.3 !important}
#schedpanel .prose strong {color: var(--amber)}
#modelspage .prose strong {color: var(--amber)}
#schedel {border: 1px solid var(--red) !important; background: transparent !important; color: var(--red) !important}
#schedpanel label span, #schedpanel [data-testid="block-info"] {font-size: 1.15rem !important}
#modelspage label span, #modelspage [data-testid="block-info"] {font-size: 1.15rem !important}
#schedpanel input, #schedpanel textarea {font-size: 1.2rem !important; color: var(--ink) !important}
#modelspage input, #modelspage textarea {font-size: 1.2rem !important; color: var(--ink) !important}
#schedpanel button {font-size: 1.15rem !important}
#modelspage button {font-size: 1.15rem !important}
#language {margin: 2px 0 10px !important; max-width: 220px}
#stop, #stop * {white-space: nowrap !important}
#chatcol:not(:has(#working .working)) #stop {display: none !important}
#wfiles button.upload-container {min-height: 0 !important; height: auto !important; padding: 10px 8px !important}
#wfiles button.upload-container .wrap {flex-direction: row !important; gap: 8px !important; min-height: 0 !important; height: auto !important}
#wfiles button.upload-container .icon-wrap {width: 18px !important; height: 18px !important; margin: 0 !important}
#wfiles button.upload-container .or:empty {display: none}
#schedel {background: transparent !important; border: none !important; box-shadow: none !important; text-decoration: underline; width: auto !important; align-self: flex-start; padding: 2px 0 !important; opacity: .85}
#modelrow #model::before {content: none}  /* no "Model:" label: the row is full */
#modelrow #model {display: flex !important; flex-wrap: nowrap !important; align-items: center; min-width: 240px !important}
#modelrow #model > .wrap, #modelrow #model > div:not([data-testid]) {flex: 1}
#model input {text-transform: none !important}
#language input {font-size: .9rem !important}
.cloudlinks {display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 4px 0 6px}
.cloudlinks a {color: var(--red) !important; text-decoration: underline; font-size: .95rem}
.cloudnote {font-size: .85rem; color: var(--dim); margin-bottom: 6px}
#schedpanel button.primary {background: var(--red) !important; color: var(--bg) !important; border: none !important; padding: 10px 28px !important}
#modelspage button.primary {background: var(--red) !important; color: var(--bg) !important; border: none !important; padding: 10px 28px !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button.primary {background: var(--ink) !important; border-radius: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage button.primary {background: var(--ink) !important; border-radius: 10px !important}
#schedpanel button:not(#schedel):not(.primary) {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
#modelspage button:not(#schedel):not(.primary) {border: 1px solid var(--cyan) !important; background: transparent !important; color: var(--cyan) !important}
#schedpanel fieldset label {background: transparent !important; border: 1px solid var(--line) !important; color: var(--ink) !important; box-shadow: none !important}
#modelspage fieldset label {background: transparent !important; border: 1px solid var(--line) !important; color: var(--ink) !important; box-shadow: none !important}
#schedpanel fieldset label.selected {background: var(--cyan) !important; color: var(--bg) !important}
#modelspage fieldset label.selected {background: var(--cyan) !important; color: var(--bg) !important}
#schedpanel fieldset label.selected * {color: var(--bg) !important}
#modelspage fieldset label.selected * {color: var(--bg) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel label span, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel [data-testid="block-info"] {font-size: .85rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage label span, body[data-screen^="modern"]:not(#a):not(#b) #modelspage [data-testid="block-info"] {font-size: .85rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel input, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel textarea, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button {font-size: .92rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage input, body[data-screen^="modern"]:not(#a):not(#b) #modelspage textarea, body[data-screen^="modern"]:not(#a):not(#b) #modelspage button {font-size: .92rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button:not(#schedel):not(.primary) {border: none !important; background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage button:not(#schedel):not(.primary) {border: none !important; background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel fieldset label {border-radius: 999px !important; border-color: var(--line) !important; padding: 4px 12px !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage fieldset label {border-radius: 999px !important; border-color: var(--line) !important; padding: 4px 12px !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel fieldset label.selected {background: var(--ink) !important; border-color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage fieldset label.selected {background: var(--ink) !important; border-color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel fieldset label.selected * {color: var(--bg) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage fieldset label.selected * {color: var(--bg) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel {background: var(--panel) !important; border: none !important; border-radius: 16px !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage {background: var(--panel) !important; border: none !important; border-radius: 16px !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedtitle {font-family: 'Newsreader', Georgia, serif !important; font-size: 1.7rem !important; color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .prose, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .prose *, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .task, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .task * {font-size: .95rem !important; line-height: 1.5 !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage .prose, body[data-screen^="modern"]:not(#a):not(#b) #modelspage .prose *, body[data-screen^="modern"]:not(#a):not(#b) #modelspage .task, body[data-screen^="modern"]:not(#a):not(#b) #modelspage .task * {font-size: .95rem !important; line-height: 1.5 !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .prose strong {color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage .prose strong {color: var(--ink) !important}
body[data-screen^="modern"]:not(#a):not(#b) .schedsub {font-size: .78rem !important; letter-spacing: .1em !important; color: var(--dim) !important; border-bottom: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedbtn {border: none !important; background: transparent !important; text-transform: none !important; font-size: .9rem !important; border-radius: 10px !important; letter-spacing: normal !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedbtn:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedback {border-radius: 10px !important; border: none !important; background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelsback {border-radius: 10px !important; border: none !important; background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button {border-radius: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage button {border-radius: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel button.primary {background: var(--cyan) !important; color: var(--panel) !important; border-color: var(--cyan) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage button.primary {background: var(--cyan) !important; color: var(--panel) !important; border-color: var(--cyan) !important}
body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .block, body[data-screen^="modern"]:not(#a):not(#b) #schedpanel .form {border: none !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelspage .block, body[data-screen^="modern"]:not(#a):not(#b) #modelspage .form {border: none !important; box-shadow: none !important}
.task span {color: var(--dim)}
body[data-screen^="modern"]:not(#a):not(#b) #chatopts .task, body[data-screen^="modern"]:not(#a):not(#b) #chatopts .task * {font-size: .85rem !important; line-height: 1.45 !important}
body[data-screen^="modern"]:not(#a):not(#b) .task {border-bottom: none}
.cloudstat code {word-break: break-all}
body[data-screen^="modern"]:not(#a):not(#b) #chatopts .prose .cloudstat, body[data-screen^="modern"]:not(#a):not(#b) #chatopts .cloudstat * {font-size: .85rem !important; line-height: 1.45 !important}
.updnote {font-size: .92rem; line-height: 1.3; color: var(--amber); padding: 0 4px}
#updrow {align-items: center !important; gap: 8px !important; margin: 2px 0 6px !important; flex-wrap: nowrap !important}
#updrow .updnote {white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
body:not(#a):not(#b):not(#c):not(#d) #updrow .updnote {font-size: .88rem !important; letter-spacing: 0 !important}
#updrow .updnote:first-line {font-weight: 500}
#updrow .updnote::before {content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: currentColor; margin-right: 8px; vertical-align: 1px}
body:not(#a):not(#b):not(#c):not(#d) #updbtn {min-height: 0 !important; height: 28px !important; padding: 0 12px !important; border-radius: 14px !important; font-size: .85rem !important;
  border: 1px solid currentColor !important; background: transparent !important; color: var(--amber) !important; text-transform: none !important; box-shadow: none !important}
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
body[data-screen^="modern"]:not(#a):not(#b) .lights {display: none !important}
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
body[data-screen^="modern"]:not(#a):not(#b) #delchat, body[data-screen^="modern"]:not(#a):not(#b) #modelsbtn {border-radius: 10px !important; text-transform: none !important; border-style: solid !important; font-size: .9rem !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #fbbtn {justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important; color: rgb(47, 111, 98) !important; margin-top: 5px !important}
body[data-screen^="modern"]:not(#a):not(#b) #fbbtn:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #renchat {border-color: transparent !important; color: var(--dim) !important; justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important; background: transparent !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #delchat {border-color: transparent !important; color: var(--dim) !important}
body[data-screen^="modern"]:not(#a):not(#b) #delchat:hover {color: var(--red) !important; background: transparent !important}
body[data-screen^="modern"]:not(#a):not(#b) #tunebtn {border-color: var(--line) !important; color: var(--cyan) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelsbtn {border-color: var(--line) !important; color: var(--cyan) !important}
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
body[data-screen^="modern"]:not(#a):not(#b) #working .done {font-size: .88rem !important}
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
body[data-screen^="modern"]:not(#a):not(#b) #modelsbtn {border: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #tunebtn:hover {background: var(--soft) !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelsbtn:hover {background: var(--soft) !important}
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
/* the working cursor keeps its slow fade even when the system asks for less motion: it is a change of light, not movement */
/* ---- app frame: the page never scrolls; the side column and the messages each scroll in their own place ---- */
html, body {height: 100%; overflow: hidden !important}
.gradio-container {height: 100vh !important; max-height: 100vh; min-height: 0 !important; padding: 0 !important; margin: 0 !important; max-width: none !important; overflow: hidden}
.gradio-container > .main, .gradio-container > .main > .wrap, main.contain, main.contain > .column {height: 100% !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; gap: 0 !important}
main.contain > .column > .row {height: 100%; flex-wrap: nowrap !important; align-items: stretch !important; gap: 12px !important; padding: 10px !important; box-sizing: border-box}
#leftcol {flex: 0 0 290px !important; width: 290px !important; min-width: 0 !important; max-height: 100%; overflow-y: auto !important; overscroll-behavior: contain}
#leftcol, #chatopts, #panels {flex-wrap: nowrap !important}
#navgap {height: 14px !important; min-height: 0 !important; padding: 0 !important; border: none !important; background: transparent !important}
#leftcol > *, #chatopts > *, #panels > * {flex-shrink: 0 !important}
#chatcol, #tune, #schedpanel, #modelspage {height: 100%; min-width: 0 !important; min-height: 0}
#chatcol, #tune, #modelspage {height: 100%; min-width: 0 !important; min-height: 0}
#chatcol {display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important}
#chatcol textarea {min-height: calc(1lh + 8px) !important}  /* Gradio measures the box by shrinking it to 1px for an instant: in Safari the chat jumps; this keeps it from shrinking */
#tune, #schedpanel, #modelspage {overflow-y: auto !important}
#schedpanel {flex-wrap: nowrap !important}  /* a tall form scrolls down instead of wrapping into a second column off to the right */
#schedpanel > * {flex-shrink: 0 !important}
#modelspage {flex-wrap: nowrap !important}  /* a tall column must scroll, not wrap into new columns to the right */
#modelspage > * {flex-shrink: 0 !important}
#modelspage h2.msec {margin: 0; font-size: 1.5rem; color: var(--cyan); letter-spacing: .04em}
#modelspage .mzone {border-radius: 14px !important; padding: 18px 20px 14px !important; margin: 14px 0 6px !important; border: none !important; flex-wrap: nowrap !important}
#modelspage .mzone > * {flex-shrink: 0 !important}
#modelspage .mz1 {background: color-mix(in srgb, #3f8a49 9%, var(--bg)) !important}
#modelspage .mz2 {background: color-mix(in srgb, #2a6bb5 9%, var(--bg)) !important}
#modelspage .mz3 {background: color-mix(in srgb, #9c6b3c 10%, var(--bg)) !important}
#modelspage .mzone .block, #modelspage .mzone .form, #modelspage .mzone .html-container {background: transparent !important; border: none !important; box-shadow: none !important}
#modelspage .mzhead {display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap}
#modelspage .mzscale {color: var(--dim); font-size: .9rem}
body:not(#a):not(#b) #modelspage .mzhead h2.msec, body:not(#a):not(#b) #setpage h2.msec {font-size: 1.55rem !important; line-height: 1.2 !important; font-weight: 600 !important}
body:not(#a):not(#b) #modelspage .mzone, body:not(#a):not(#b) #setpage .mzone {gap: 6px !important}
body:not(#a):not(#b) table.mrank tr {border: none !important}
#modelspage .mzscale::before {content: "↓ "}
.mslot .mbar.dl {width: 110px}
#modelspage h3.msub2 {margin: 14px 0 4px; font-size: 1.1rem; color: var(--ink)}
#modelspage .msub {color: var(--dim); font-size: 1rem; max-width: 70ch}
#ftbase .wrap:not(.default), #ftbase input {background: var(--panel) !important; color: var(--ink) !important}
table.mrank {width: 100%; border-collapse: collapse; font-size: .95rem; margin: 4px 0 8px}
table.mrank th {text-align: left; font-weight: 500; color: var(--dim); border-bottom: 1px solid var(--line); padding: 4px 8px}
table.mrank td {padding: 5px 8px; border-bottom: 1px solid color-mix(in srgb, var(--line) 50%, transparent); vertical-align: middle; color: var(--ink)}
table.mrank td.mname {font-weight: 600; word-break: break-word}
table.mmenu td.mbtns {text-align: right; white-space: nowrap}
details.mcost {display: inline-block; vertical-align: middle; margin-left: 6px; position: relative}
table.mrank td:has(details.mcost) {white-space: nowrap}
details.mcost summary {list-style: none; cursor: pointer; color: var(--brand); font-size: .8rem; padding: 0 4px; border-radius: 4px}
details.mcost summary::-webkit-details-marker {display: none}
details.mcost summary::before {content: "▸ $"}
details.mcost[open] summary::before {content: "▾ $"}
details.mcost summary:hover {background: color-mix(in srgb, var(--brand) 12%, transparent)}
details.mcost .pwrap {position: absolute; z-index: 60; top: 100%; left: 0; width: 300px; white-space: normal; box-shadow: 0 8px 20px rgba(0, 0, 0, .15); margin: 4px 0 0; padding: 8px 10px; border-radius: 8px; background: var(--bg); border: 1px solid color-mix(in srgb, var(--brand) 35%, transparent); max-width: 340px}
table.pbox {border-collapse: collapse; font-size: .85rem}
body:not(#a):not(#b):not(#c) table.pbox, body:not(#a):not(#b):not(#c) table.pbox tr, body:not(#a):not(#b):not(#c) table.pbox td, body:not(#a):not(#b):not(#c) table.pbox th {border: none !important; background: transparent !important}
table.pbox td, table.pbox th {padding: 1px 12px 1px 0 !important; border: none !important; font-weight: 400; color: var(--ink)}
table.pbox th {color: var(--dim); font-size: .78rem}
table.pbox td:not(:first-child) {font-variant-numeric: tabular-nums; text-align: right}
details.mcost .pnote {margin-top: 6px; font-size: .75rem; line-height: 1.35; color: var(--dim); white-space: normal}
table.mmenu .mslot {margin-left: 4px}
table.mmenu .mspace {display: inline-block; width: 38px}
.remotebox {line-height: 1.5}
.remotebox ol {margin: 6px 0 0 1.2em; padding: 0}
.remotegrid {display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap}
.remotegrid > div:first-child {flex: 1 1 280px}
.remotegrid code {font-size: 1rem}
.remoteqr svg {width: 148px; height: 148px; background: #fff; border-radius: 8px}
.mdefault {font-weight: 400; font-size: .78rem; color: var(--brand); border: 1px solid var(--brand); border-radius: 999px; padding: 0 7px; margin-left: 6px}
.mhidden {display: none !important}
button.mact {font: inherit; font-size: .85rem; padding: 2px 12px; border-radius: 6px; border: 1px solid var(--brand); color: var(--brand); background: transparent; cursor: pointer; white-space: nowrap}
button.mact:hover {background: var(--brand); color: var(--bg, #fff)}
button.mact:disabled {opacity: .6; cursor: wait}
input.mren {font: inherit; font-size: .85rem; padding: 2px 6px; width: 11em; border: 1px solid var(--brand); border-radius: 6px; margin-right: 4px; background: var(--bg, #fff); color: inherit}
table.mrank td.mnote, table.mrank .mdim {color: var(--dim); font-size: .85rem}
.mbar {display: inline-block; width: 90px; height: 7px; border-radius: 4px; background: color-mix(in srgb, var(--line) 60%, transparent); vertical-align: middle; margin-right: 6px; overflow: hidden}
.mbar i {display: block; height: 100%; background: var(--brand)}
#modelspage table.mrank, #modelspage table.mrank th, #modelspage table.mrank td {border-left: none !important; border-right: none !important; border-top: none !important; background: transparent !important}
#modelspage table.mrank {border: none !important; margin-bottom: 0 !important}

#tune, #modelspage {overflow-y: auto !important}
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
/* ---------- phones and narrow windows: the chat takes the whole screen, the side column slides over it from ☰ ---------- (Gradio rewrites rules inside @media, so a class on body marks a narrow screen instead) */
body.narrow:not(#a):not(#b) .gradio-container, body.narrow:not(#a):not(#b) .gradio-container > .main, body.narrow:not(#a):not(#b) .gradio-container .wrap.contain {padding: 0 !important; margin: 0 !important; max-width: 100vw !important}
body.narrow:not(#a):not(#b):not(.mside) #leftcol {display: none !important}
body.narrow.mside:not(#a):not(#b) #leftcol {display: flex !important; position: fixed !important; inset: 0 !important; z-index: 900 !important; width: 100vw !important; max-width: 100vw !important; flex-basis: auto !important; height: 100dvh !important; max-height: 100dvh !important; border-radius: 0 !important; padding: 56px 16px 24px !important; overflow-y: auto !important}
body.narrow.noside #leftcol {display: none}
body.narrow:not(#a):not(#b) #sidetoggle {position: fixed !important; top: 10px !important; left: 10px !important; z-index: 950 !important; font-size: 1.25rem !important; width: 40px !important; height: 40px !important; opacity: .85}
body.narrow:not(#a):not(#b) #chatcol, body.narrow:not(#a):not(#b) #modelspage, body.narrow:not(#a):not(#b) #schedpanel, body.narrow:not(#a):not(#b) #tune, body.narrow:not(#a):not(#b) #mempage, body.narrow:not(#a):not(#b) #setpage, body.narrow:not(#a):not(#b) #teampage, body.narrow:not(#a):not(#b) #readpage {min-width: 0 !important; width: 100vw !important; max-width: 100vw !important; flex: 1 1 100% !important; padding: 8px 10px 6px !important; border-radius: 0 !important}
body.narrow:not(#a):not(#b) #chatcol {height: 100dvh !important; max-height: 100dvh !important}
body.narrow:not(#a):not(#b) #chatbar {padding-left: 48px !important; min-height: 40px !important}
body.narrow:not(#a):not(#b) #chat {min-height: 0 !important}
body.narrow:not(#a):not(#b) #chat .message-row, body.narrow:not(#a):not(#b) #chat .message {max-width: 100% !important}
body.narrow:not(#a):not(#b) #chat .message-row.bot-row {margin-left: 0 !important; margin-right: 0 !important; padding-left: 0 !important; padding-right: 0 !important; width: 100% !important}
body.narrow:not(#a):not(#b) #chat .message-row.user-row {max-width: 85% !important}
body.narrow:not(#a):not(#b) #chat .message-wrap {padding-left: 4px !important; padding-right: 4px !important}
body.narrow:not(#a):not(#b) #chat .bubble-wrap {padding: 10px 6px !important}
body.narrow:not(#a):not(#b) #chat .placeholder h3, body.narrow:not(#a):not(#b) #chat .placeholder-content h3 {font-size: 1.5rem !important; line-height: 1.25 !important}
body.narrow:not(#a):not(#b) #promptrow {gap: 6px !important; flex-wrap: nowrap !important}
body.narrow:not(#a):not(#b) #promptrow > .form {min-width: 0 !important; flex: 1 1 auto !important}
body.narrow:not(#a):not(#b) #stop {min-width: 0 !important; flex: 0 0 auto !important; padding: 0 10px !important}
body.narrow:not(#a):not(#b) #modelrow {flex-wrap: wrap !important; padding-right: 0 !important; margin-top: 0 !important; gap: 4px 10px !important; row-gap: 2px !important}
body.narrow:not(#a):not(#b) #modelrow > #working {flex: 1 1 100% !important}
body.narrow:not(#a):not(#b) #modelpick > #cloudbadge {order: 2; flex: 1 1 0 !important; min-width: 0; font-size: .75rem; white-space: nowrap; text-align: right}
body.narrow:not(#a):not(#b) #cloudbadge .cloudbadge {white-space: nowrap !important; font-size: .75rem !important; padding: 0 2px !important}
body.narrow:not(#a):not(#b) #modelrow #model::before {content: none}
body.narrow:not(#a):not(#b) #modelrow > #working {order: 3}
body.narrow:not(#a):not(#b) #modelpick > #model {order: 1}
body.narrow:not(#a):not(#b) #modelrow > :not(.form):not(#deskpick) label, body.narrow:not(#a):not(#b) #modelrow > :not(.form):not(#deskpick) .label {display: none !important}
body.narrow:not(#a):not(#b) button.fbtn {width: 34px; height: 34px}
body.narrow:not(#a):not(#b) #starters .starters {gap: 6px} body.narrow:not(#a):not(#b) button.starter {padding: 7px 11px; font-size: .85rem}
body.narrow:not(#a):not(#b) #modelrow > .form:has(.modeopt) {order: 0; flex: 1 1 100% !important; justify-content: flex-start}
body.narrow:not(#a):not(#b) table.mrank, body.narrow:not(#a):not(#b) table.mrank tbody {display: block; width: 100%}
body.narrow:not(#a):not(#b) table.mrank thead, body.narrow:not(#a):not(#b) table.mrank tr:has(th) {display: none !important}
body.narrow:not(#a):not(#b) table.mrank tr {display: flex; flex-wrap: wrap; align-items: center; gap: 2px 10px; padding: 8px 0; border-bottom: 1px solid color-mix(in srgb, var(--line) 30%, transparent) !important}
body.narrow:not(#a):not(#b) table.mrank td {display: block; padding: 0 !important; border: none !important; white-space: normal}
body.narrow:not(#a):not(#b) table.mrank td:first-child {flex: 1 1 100%}
body.narrow:not(#a):not(#b) table.mrank td.mbtns {flex: 0 1 auto; margin-left: auto}
body.narrow:not(#a):not(#b) table.mrank tr {gap: 4px 8px}
body.narrow:not(#a):not(#b) #modelspage .mzone, body.narrow:not(#a):not(#b) #setpage .mzone {padding: 14px 12px !important}
body.narrow:not(#a):not(#b) #schedtitle {font-size: 1.2rem !important}
body.narrow:not(#a):not(#b) #cmenu {max-width: calc(100vw - 16px)}


/* ---- Modern: the conversation in a calm reading column, the composer below it ---- */
body[data-screen^="modern"]:not(#a):not(#b) #chat {background: transparent !important; border: none !important; box-shadow: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .message-wrap {max-width: none; margin: 0 auto !important; padding: 3rem 1.5rem 2rem !important; width: 100%; box-sizing: border-box}
body[data-screen^="modern"]:not(#a):not(#b) #chat .message-content, body[data-screen^="modern"]:not(#a):not(#b) #chat .message-content * {font-size: 1rem !important; line-height: 1.65 !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .user-row .message {border-radius: 18px !important; padding: .55rem 1rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #chat .placeholder {max-width: 40rem; margin: 0 auto}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow, body[data-screen^="modern"]:not(#a):not(#b) #modelrow {width: 100% !important; max-width: none; padding: 0 1.5rem !important; margin: 0 auto !important; box-sizing: border-box}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow {padding: 0 .5rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow .multimodal-textbox {border-radius: 22px !important; box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 6%, transparent), 0 4px 18px color-mix(in srgb, var(--ink) 6%, transparent) !important}
body[data-screen^="modern"]:not(#a):not(#b) #promptrow textarea {font-size: 1rem !important; line-height: 1.5 !important; max-height: 40vh !important; overflow-y: auto !important}
body[data-screen^="modern"]:not(#a):not(#b) #modelrow {padding: .35rem 1.25rem .5rem !important; justify-content: flex-start !important; gap: .75rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol {padding: 14px 10px !important; border-radius: 18px !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol .lights {display: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol .status {padding: 0 6px 6px !important; letter-spacing: 0 !important; text-transform: none !important}
body[data-screen^="modern"]:not(#a):not(#b) #newchat, body[data-screen^="modern"]:not(#a):not(#b) #tunebtn, body[data-screen^="modern"]:not(#a):not(#b) #schedbtn, body[data-screen^="modern"]:not(#a):not(#b) #delchat {justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #newchat, body[data-screen^="modern"]:not(#a):not(#b) #modelsbtn, body[data-screen^="modern"]:not(#a):not(#b) #schedbtn, body[data-screen^="modern"]:not(#a):not(#b) #delchat {justify-content: flex-start !important; padding: .45rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important; min-height: 0 !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs label {padding: .42rem .7rem !important; border-radius: 10px !important; font-size: .92rem !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol {gap: 2px !important}
body[data-screen^="modern"]:not(#a):not(#b) #leftcol > .html-container, body[data-screen^="modern"]:not(#a):not(#b) #leftcol > .block:has(#brand) {margin-bottom: 10px !important}
body[data-screen^="modern"]:not(#a):not(#b) #convs {margin: 10px 0 4px !important}
#chat .top-panel {display: none !important}
#jdown {display: none; position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.35); align-items: center; justify-content: center}
#jdown.on {display: flex}
#jdown > div {max-width: 30rem; margin: 16px; padding: 22px 24px; border-radius: 14px; background: var(--panel, #fff); color: var(--text, #222); box-shadow: 0 10px 40px rgba(0,0,0,.25); line-height: 1.5}
#jdown b {font-size: 1.1rem} #jdown p {margin: 8px 0 14px; opacity: .85} #jdown code {font-size: .85em; overflow-wrap: anywhere}
#jdown button {padding: 8px 16px; border-radius: 8px; border: 1px solid currentColor; background: transparent; color: var(--brand, inherit); cursor: pointer}
#logbox textarea {font-family: ui-monospace, Menlo, Consolas, monospace !important; font-size: .78rem !important; line-height: 1.4 !important}
#rdpos, #rdact {display: none !important}
#readpage .rdintro {opacity: .75; margin-bottom: 4px}
#readpage #rdadd {gap: 8px !important; align-items: stretch !important; flex-wrap: wrap !important}
#readpage #rdadd > * {min-height: 40px}
#readpage #rdadd button {min-height: 40px !important; white-space: nowrap}
#readpage #rdpaste {flex: 1 1 260px !important}
#readpage .rdlib {margin-top: 14px; border-top: 1px solid var(--line)}
#readpage .rdrow {display: flex; align-items: center; gap: 12px; padding: 12px 4px; border-bottom: 1px solid var(--line); cursor: pointer}
#readpage .rdrow:hover .rdname {color: var(--brand)}
#readpage .rdmain {flex: 1; min-width: 0}
#readpage .rdname {font-weight: 600; line-height: 1.3; overflow-wrap: anywhere}
#readpage .rdmeta {font-size: .85rem; opacity: .65; margin-top: 2px}
#readpage .rdprogress {height: 3px; border-radius: 2px; background: var(--line); margin-top: 7px; overflow: hidden}
#readpage .rdprogress span {display: block; height: 100%; background: var(--brand)}
#readpage .rdx {flex: none; width: 32px; height: 32px; border: none; background: transparent; color: inherit; opacity: .45; border-radius: 8px; cursor: pointer; font-size: .95rem}
#readpage .rdx:hover {opacity: 1; color: var(--red)}
body:not(#a):not(#b):not(#c):not(#d) #readpage button.rdx {background: transparent !important; border: none !important; box-shadow: none !important; min-width: 0 !important; padding: 0 !important}
body.narrow #readpage #rdadd > :has(#rdpaste), body.narrow #readpage #rdpaste {flex: 1 1 100% !important}
body.narrow #readpage #rdadd > button, body.narrow #readpage #rdadd > :has(button) {flex: 1 1 0 !important}
#readpage .rdempty {margin-top: 14px; opacity: .6}
#readpage #rdbar {position: sticky; top: 44px; z-index: 55; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 22px 0 8px; margin-top: -14px; background: var(--panel); border-bottom: 1px solid var(--line)}
#readpage #rdbar button {width: 38px; height: 34px; padding: 8px; border-radius: 8px; border: 1px solid var(--border-color-primary); background: transparent; color: inherit; cursor: pointer}
#readpage #rdbar button svg {width: 100%; height: 100%}
#readpage #rdbar button#rdplay {width: 46px; color: var(--brand)}
#readpage #rdbar select {height: 34px; border-radius: 8px; border: 1px solid var(--border-color-primary); background: transparent; color: inherit; padding: 0 6px}
#readpage #rdbar #rdvoice {max-width: 13rem}
body:not(#a):not(#b):not(#c):not(#d) #readpage #rdbar select {appearance: none; -webkit-appearance: none; height: 34px; border: 1px solid var(--line) !important; border-radius: 8px !important; padding: 0 26px 0 10px !important; background: transparent url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'%3E%3Cpath fill='%23888' d='M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z'/%3E%3C/svg%3E\") no-repeat right 8px center / 11px !important; color: inherit !important; font: inherit; font-size: .92rem; cursor: pointer}
#readpage #rdbar .rdopt {display: inline-flex; align-items: center; gap: 4px; opacity: .85}
#readpage #rdbar .rdopt input {width: 80px; accent-color: var(--brand)}
#readpage #rdprog {margin-left: 6px; font-variant-numeric: tabular-nums; opacity: .8}
#readpage #rdhint {margin-left: auto; opacity: .55; font-size: .85rem}
#readpage #rdhint.rdwarn {opacity: 1; color: var(--brand); font-weight: 600}
#readpage #rdbook {max-width: 44rem; margin: 6px auto 40vh; font-size: 1.08rem; line-height: 1.65}
#readpage #rdbook .rdtitle {font-size: 1.5rem; margin: 10px 0 18px}
#readpage #rdbook p {margin: 0 0 .9em; padding: 2px 8px; border-radius: 6px; cursor: pointer}
#readpage #rdbook p:hover {background: color-mix(in srgb, var(--brand) 6%, transparent)}
#readpage #rdbook p.rdnow {background: color-mix(in srgb, var(--brand) 15%, transparent)}
body:not(#a):not(#b):not(#c):not(#d) #rddel {color: var(--red) !important}
/* a field with the cursor in it keeps its look's colour (Gradio's own focus colour is the dark one of its base theme) */
input, textarea, .wrap {--input-background-fill-focus: var(--input-background-fill)}
#teampage .wsteps {display: flex; gap: 18px; flex-wrap: wrap; margin: 4px 0 10px; font-size: .92rem}
#teampage .wstep {color: var(--dim)}
#teampage .wstep.on {color: var(--brand); font-weight: 600}
#teampage .wstep.done {color: var(--text, inherit)}
#teampage #wiznav {margin-top: 12px; gap: 8px !important}
#teampage button {min-height: 0 !important}
#teampage > .column, #teampage .column {flex-grow: 0 !important}
body:not(#a):not(#b):not(#c):not(#d) #tmdel, body:not(#a):not(#b):not(#c):not(#d) #mremove {color: var(--red) !important}
#setpage .dkhead {margin: 10px 0 2px}
#setpage .dkrow {gap: 6px !important; margin: 0 !important; align-items: flex-start !important}
#setpage .dkrow textarea, #setpage .dkrow input {font-size: .9rem !important}
body:not(#a):not(#b):not(#c):not(#d) #setpage button.dkdel {background: transparent !important; border: none !important; box-shadow: none !important; color: var(--dim) !important; min-width: 0 !important}
body:not(#a):not(#b):not(#c):not(#d) #setpage button.dkdel:hover {color: var(--red) !important}
/* the desk editor: its buttons alike (Remove in red), and a readable selected row in the tasks table */
#dksave, #dkremove, #setpage button.dkbtn {min-height: 0 !important}
body:not(#a):not(#b):not(#c):not(#d) #dkremove {color: var(--red) !important}
body:not(#a):not(#b):not(#c):not(#d) #dkstarters :is(tr.selected, td.selected, .selected, [aria-selected="true"]) {background: color-mix(in srgb, var(--brand) 12%, transparent) !important; color: var(--ink) !important}
body:not(#a):not(#b):not(#c):not(#d) #dkstarters :is(tr.selected, td.selected, .selected, [aria-selected="true"]) * {color: var(--ink) !important}
body.narrow:not(#a):not(#b):not(#c):not(#d):not(#e) #modelrow > .form:has(.modeopt) {border-left: none !important; padding-left: 2px !important; margin-left: 0 !important}
/* ---------- under the message box: desks are one choice (a segmented control); goal and swarm are switches ---------- */
body:not(#a):not(#b):not(#c):not(#d) #deskpick .wrap {display: inline-flex !important; gap: 0 !important; padding: 2px !important; border: 1px solid color-mix(in srgb, var(--line) 85%, transparent) !important; border-radius: 999px !important; background: color-mix(in srgb, var(--ink) 4%, transparent) !important; width: auto !important}
body:not(#a):not(#b):not(#c):not(#d) #deskpick label {border: none !important; background: transparent !important; box-shadow: none !important; padding: 3px 11px !important; border-radius: 999px !important}
body:not(#a):not(#b):not(#c):not(#d) #deskpick label.selected, body:not(#a):not(#b):not(#c):not(#d) #deskpick label:has(input:checked) {background: var(--panel) !important; box-shadow: 0 1px 3px rgba(0, 0, 0, .14) !important}
body:not(#a):not(#b):not(#c):not(#d) #modelrow > .form:has(.modeopt) {border-left: 1px solid color-mix(in srgb, var(--line) 90%, transparent) !important; padding-left: 12px !important; margin-left: 6px !important; gap: 12px !important}
body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt {border: none !important; background: transparent !important; padding: 2px 0 !important; box-shadow: none !important}
body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt label {display: inline-flex !important; align-items: center !important; gap: 0 !important; position: relative !important}
body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt label::before {content: ""; flex: none; width: 26px; height: 15px; margin-right: 7px; border-radius: 999px; background: color-mix(in srgb, var(--ink) 20%, transparent); transition: background .15s}
body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt label::after {content: ""; position: absolute; left: 2px; top: 50%; width: 11px; height: 11px; margin-top: -5.5px; border-radius: 50%; background: #fff; box-shadow: 0 1px 2px rgba(0, 0, 0, .3); transition: left .15s}
body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt:has(input:checked) label::before {background: var(--brand)}
body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt:has(input:checked) label::after {left: 13px}
@media (prefers-reduced-motion: reduce) {body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt label::before, body:not(#a):not(#b):not(#c):not(#d) #modelrow .modeopt label::after {transition: none}}
/* the 🔊 button under answers, and the voice choices in Settings */
#chat button.speak {display: inline-flex; align-items: center; justify-content: center; background: transparent; border: none; color: var(--dim); cursor: pointer; padding: 4px; border-radius: 6px}
#chat button.speak svg {width: 15px; height: 15px}
#chat button.speak:hover {color: var(--ink); background: var(--soft)}
#chat button.speak.speaking {color: var(--brand) !important; animation: busyblink 1.2s ease-in-out infinite}
#voicepick {display: grid; gap: 6px; margin-top: 6px}
#voicepick .vrow {display: grid; grid-template-columns: 7rem 1fr auto; gap: 8px; align-items: center}
#voicepick select {appearance: auto !important; padding: 5px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); font: inherit}
#voicepick .vtest {background: var(--soft); border: none; border-radius: 8px; padding: 5px 10px; cursor: pointer; color: var(--ink)}
/* a chat Joshua is working on: a green dot that blinks next to its title */
body:not(#a):not(#b):not(#c):not(#d) #convs label span.ji-busy, body:not(#a):not(#b):not(#c):not(#d) #convs label span.ji-busy * {color: #2e8b3e !important; fill: #2e8b3e !important; opacity: 1}
body:not(#a):not(#b):not(#c):not(#d) #convs label span.ji-busy {animation: busyblink 1.2s ease-in-out infinite; -webkit-mask-image: none !important; mask-image: none !important}  /* the fade at the end of long titles must not touch the dot */
#convs .ji-busy {display: inline-flex !important; align-items: center; vertical-align: middle; margin-right: .35em !important}
#convs .ji-busy svg {width: .8em !important; height: .8em !important}
@keyframes busyblink {0%, 100% {opacity: 1} 50% {opacity: .3}}
@media (prefers-reduced-motion: reduce) {body:not(#a):not(#b):not(#c):not(#d) #convs label span.ji-busy {animation: none}}
/* ---------- Prof Falken: WOPR's green terminal (strong selectors: the themes style code blocks and fonts too) ---------- */
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr, body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr *, body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) #chat pre:has(> code.language-wopr), body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) #chat pre:has(> code.language-wopr) * {font-family: 'VT323', 'Menlo', 'Courier New', ui-monospace, monospace !important; color: #7dff8a !important; letter-spacing: .02em !important; text-shadow: 0 0 6px rgba(125, 255, 138, .45)}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr {background: #050805 !important; border-radius: 10px; max-width: 560px; margin: 0 auto 12px; padding: 18px 22px; text-align: left}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr, body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr * {font-size: 1.1rem !important; line-height: 1.3 !important; font-weight: 400 !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr .wline {display: block; background: none !important; border: none !important; padding: 0 !important; margin: 0 !important; cursor: pointer; text-align: left; white-space: pre; box-shadow: none !important; border-radius: 3px !important; min-height: 0 !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr .wline:hover {background: rgba(125, 255, 138, .15) !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) #chat pre:has(> code.language-wopr) {background: #050805 !important; border: none !important; border-radius: 10px !important; padding: 16px 20px !important; width: max-content; max-width: 100%}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) #chat pre:has(> code.language-wopr) code {background: transparent !important; font-size: 1.2rem !important; line-height: 1.2 !important; white-space: pre !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) #chat .code_wrap:has(code.language-wopr) button {display: none !important}
#chatcol:has(#starters .wopr) #chat .placeholder {visibility: hidden !important}  /* WOPR's greeting replaces Joshua's */
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr-in {display: flex; align-items: center; gap: 0; margin-top: 10px; font-family: 'VT323', 'Menlo', 'Courier New', ui-monospace, monospace !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr-in input {flex: 1; background: transparent !important; border: none !important; outline: none !important; box-shadow: none !important; padding: 0 !important; margin: 0 !important; color: #7dff8a !important; caret-color: #7dff8a; font-family: inherit !important; font-size: 1.2rem !important; text-transform: uppercase; min-width: 0}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) #chat pre:has(> code.language-wopr), body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr {position: relative !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr-voice {position: absolute !important; bottom: 12px; right: 12px; background: transparent !important; border: 1px solid rgba(125, 255, 138, .5) !important; color: #7dff8a !important; font-family: 'VT323', 'Menlo', monospace !important; font-size: .85rem !important; line-height: 1 !important; padding: 4px 7px !important; border-radius: 4px !important; cursor: pointer; letter-spacing: .05em; box-shadow: none !important; min-height: 0 !important; opacity: .75}
body:not(#a):not(#b):not(#c):not(#d):not(#e):not(#f):not(#g) .wopr-voice:hover {opacity: 1; background: rgba(125, 255, 138, .12) !important}
.wcur {animation: wblink 1s steps(1) infinite} @keyframes wblink {50% {opacity: 0}}
/* ---------- the side column: chats and menu items closer together ---------- */
body[data-screen^="modern"]:not(#a):not(#b):not(#c) #convs label {padding: .28rem .7rem !important}
body[data-screen^="modern"]:not(#a):not(#b):not(#c) #panels {gap: 2px !important}
body[data-screen^="modern"]:not(#a):not(#b):not(#c) #panels > button {margin: 0 !important; padding-top: .32rem !important; padding-bottom: .32rem !important}
body[data-screen^="modern"]:not(#a):not(#b):not(#c) #panels > #navgap {height: 10px !important; min-height: 10px !important}
/* ---------- under the message box: desks and modes on the left; the model, what it can do and its price on the right, on one line ---------- */
body:not(.narrow):not(#a):not(#b):not(#c) #modelrow {flex-wrap: wrap !important; align-items: center !important; column-gap: 6px !important; row-gap: 6px !important}
body:not(.narrow):not(#a):not(#b):not(#c) #modelrow > #deskpick {order: 0}
body:not(.narrow):not(#a):not(#b):not(#c) #modelrow > .form:has(.modeopt) {order: 1}
body:not(.narrow):not(#a):not(#b):not(#c) #modelrow > #working {order: 2}
body:not(.narrow):not(#a):not(#b):not(#c) #modelpick > #model {order: 3; flex: 0 1 auto !important; width: auto !important; min-width: 0 !important}
body:not(.narrow):not(#a):not(#b):not(#c) #modelpick > #cloudbadge {order: 4; flex: 0 0 auto !important; min-width: 0 !important; width: auto !important}
body:not(.narrow):not(#a):not(#b):not(#c) #cloudbadge, body:not(.narrow):not(#a):not(#b):not(#c) #cloudbadge .html-container, body:not(.narrow):not(#a):not(#b):not(#c) #cloudbadge .cloudbadge {overflow: visible !important}
body:not(.narrow):not(#a):not(#b):not(#c) #cloudbadge .cloudbadge, body:not(.narrow):not(#a):not(#b):not(#c) #cloudbadge .cloudbadge * {font-size: .85rem !important; color: var(--dim) !important}
body:not(.narrow):not(#a):not(#b):not(#c) #cloudbadge .cloudbadge {padding: 0 4px 0 0 !important; white-space: nowrap !important}
body:not(#a):not(#b):not(#c) #modelpick {flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; gap: 8px !important; flex-wrap: nowrap !important; align-items: center !important; padding: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important}
body:not(.narrow):not(#a):not(#b):not(#c) #modelrow > #modelpick {order: 3; margin-left: auto !important}
body.narrow:not(#a):not(#b):not(#c) #modelrow > #modelpick {order: 1; flex: 1 1 0 !important; min-width: 0 !important; gap: 4px !important}
body:not(.narrow):not(#a):not(#b):not(#c) #modelpick > #model input {font-size: .85rem !important}
/* the explanation of an icon or a price shows at once on hover, above it */
#cloudbadge [data-tip] {position: relative; cursor: default}
#cloudbadge [data-tip]:hover::after {content: attr(data-tip); position: absolute; bottom: calc(100% + 8px); right: -6px; z-index: 60; width: max-content; max-width: 280px; white-space: normal; text-align: left;
  background: #2a2622; color: #fff; font-size: .78rem; line-height: 1.35; font-weight: 400; padding: 7px 10px; border-radius: 8px; box-shadow: 0 4px 16px rgba(0, 0, 0, .18); pointer-events: none}
/* ---------- Paper: the look of a well-set document (like a PDF in macOS Preview) ----------
   Arial / Helvetica, near-black on white, a clear size step between headings and text, lines of 1.5,
   and tables drawn with thin grey rules and a pale header row. */
body[data-screen="modern-paper-night"] {--pp-strong: #fbf7f0; --pp-th: #34302b; --pp-rule: #4a453e; --pp-quote: #cfc8be; --pp-user: #ece6dc; --bg: #1e1c19; --bg-hi: #1e1c19; --panel: #26231f; --codebg: #191714; --line: rgba(255, 240, 220, .11); --cyan: #7cc4b2; --glow: transparent; --ink: #ece6dc; --dim: #a39a8e; --amber: #7cc4b2; --red: #e07a6a; --codeink: #ece6dc; --shadow: none; --check: #7cc4b2; --userbg: #33302b; --green: #7cc4b2; --soft: rgba(124, 196, 178, .1)}
body[data-screen="modern-paper"] {--pp-strong: #000; --pp-th: var(--pp-th); --pp-rule: var(--pp-rule); --pp-quote: #333; --pp-user: #111; --bg: #f6f4ef; --bg-hi: #f6f4ef; --panel: #fffdf9; --codebg: #f1eee7; --line: rgba(0, 0, 0, .12); --cyan: #24665a; --glow: transparent; --ink: #111111; --dim: #6b6b70; --amber: #24665a; --red: #b3261e; --codeink: #111111; --shadow: none; --check: #24665a; --userbg: #ebe7de; --green: #24665a; --soft: rgba(0, 0, 0, .05)}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) {--side: #efebe3}
body[data-screen="modern-paper-night"]:not(#a):not(#b):not(#c):not(#d) {--side: #23201c}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) .gradio-container, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) .gradio-container *:not(svg):not(path):not(code):not(pre):not(pre *):not(#brand):not(#brand *) {font-family: Arial, 'Helvetica Neue', Helvetica, 'Liberation Sans', sans-serif !important; -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose * {font-size: 1.06rem !important; line-height: 1.5 !important; color: var(--ink)}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message-content > .prose, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message-content > .prose * {opacity: 1 !important}  /* the answer in full black, not faded like the folded parts */
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose p {margin: .45em 0 .8em !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h1, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h1 * {font-size: 1.75rem !important; line-height: 1.2 !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h2, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h2 * {font-size: 1.5rem !important; line-height: 1.22 !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h3, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h3 * {font-size: 1.22rem !important; line-height: 1.25 !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h4, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose h4 * {font-size: 1.06rem !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose :is(h1, h2, h3, h4) {font-weight: 700 !important; color: var(--pp-strong) !important; letter-spacing: -.01em !important; margin: 1.15em 0 .4em !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose :is(h1, h2, h3, h4):first-child {margin-top: .2em !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose strong, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose b {font-weight: 700 !important; color: var(--pp-strong) !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose :is(ul, ol) {padding-left: 1.5em !important; margin: .3em 0 .9em !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose li {margin: .25em 0 !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose ul > li {list-style: disc !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose table {border-collapse: collapse !important; width: 100% !important; margin: .7em 0 1.4em !important; border: 1px solid var(--pp-rule) !important; display: table !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose :is(th, td) {border: 1px solid var(--pp-rule) !important; padding: .6em .85em !important; text-align: left !important; vertical-align: middle !important; background: transparent !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose th, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose th * {background: var(--pp-th) !important; font-weight: 700 !important; color: var(--pp-strong) !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose :is(th, td) p {margin: 0 !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose blockquote {border-left: 3px solid var(--pp-rule) !important; padding-left: 1em !important; color: var(--pp-quote) !important; margin: .6em 0 1em !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.bot .prose a {color: var(--cyan) !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.user, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #chat .message.user * {font-size: 1.02rem !important; line-height: 1.45 !important; color: var(--pp-user) !important}
body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #schedtitle, body[data-screen^="modern-paper"]:not(#a):not(#b):not(#c) #tunetitle {font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif !important; font-weight: 700 !important; letter-spacing: -.015em !important; color: var(--pp-strong) !important}
#chat button.jcopied {color: var(--brand) !important; transform: scale(1.15); transition: transform .15s}
/* ---------- phone: the message box and the switches under it take two short lines, not half the screen ---------- */
body.narrow:not(#a):not(#b):not(#c) #promptrow {padding: 0 !important; gap: 4px !important; align-items: center !important}
body.narrow:not(#a):not(#b):not(#c) #fbtnwrap {flex: 0 0 auto !important; width: auto !important; background: transparent !important; border: none !important; box-shadow: none !important}
body.narrow:not(#a):not(#b):not(#c) button.fbtn {width: 32px; height: 32px}
body.narrow:not(#a):not(#b):not(#c) #promptrow .multimodal-textbox {border-radius: 18px !important}
body.narrow:not(#a):not(#b):not(#c) #promptrow .input-container {padding: 2px 4px !important; min-height: 0 !important}
body.narrow:not(#a):not(#b):not(#c) #promptrow textarea {font-size: 16px !important; line-height: 1.35 !important; padding-top: 6px !important; padding-bottom: 6px !important}
body.narrow:not(#a):not(#b):not(#c) #promptrow textarea::placeholder {white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
body.narrow:not(#a):not(#b):not(#c) #modelrow {padding: 4px 2px 2px !important; gap: 4px 6px !important; row-gap: 4px !important; flex-wrap: wrap !important; align-items: center !important}
body.narrow:not(#a):not(#b):not(#c) #modelrow > * {margin: 0 !important}
body.narrow:not(#a):not(#b):not(#c) #deskpick {order: 0; flex: 0 0 auto !important}
body.narrow:not(#a):not(#b):not(#c) #deskpick {flex: 1 1 100% !important; max-width: 100% !important; min-width: 0 !important}
body.narrow:not(#a):not(#b):not(#c) #deskpick .wrap {gap: 3px !important; flex-wrap: nowrap !important; overflow-x: auto !important; scrollbar-width: none; -webkit-overflow-scrolling: touch}
body.narrow:not(#a):not(#b):not(#c) #deskpick .wrap::-webkit-scrollbar {display: none}
body.narrow:not(#a):not(#b):not(#c) #deskpick label {flex: 0 0 auto !important}
body.narrow:not(#a):not(#b):not(#c) #deskpick label {padding: 1px 6px !important; font-size: .74rem !important}
body.narrow:not(#a):not(#b):not(#c) #modelrow > .form:has(.modeopt) {order: 0; flex: 0 0 auto !important; gap: 3px !important}
body.narrow:not(#a):not(#b):not(#c) #modelrow .modeopt {padding: 1px 6px !important}
body.narrow:not(#a):not(#b):not(#c) #modelrow .modeopt label, body.narrow:not(#a):not(#b):not(#c) #modelrow .modeopt label span {font-size: .74rem !important}
body.narrow:not(#a):not(#b):not(#c) #modelpick > #model {order: 1; flex: 0 1 auto !important; min-width: 0 !important; width: auto !important; max-width: 48vw}
body.narrow:not(#a):not(#b):not(#c) #modelrow #model input {font-size: .8rem !important}
body.narrow:not(#a):not(#b):not(#c) #modelpick > #cloudbadge {order: 2; flex: 1 1 0 !important; min-width: 0 !important; text-align: left}
body.narrow:not(#a):not(#b):not(#c) #cloudbadge .html-container, body.narrow:not(#a):not(#b):not(#c) #working .html-container {padding: 0 !important}
body.narrow:not(#a):not(#b):not(#c) #cloudbadge .cloudbadge {font-size: .72rem !important; padding: 0 !important; white-space: nowrap !important; overflow: hidden; text-overflow: ellipsis}
body.narrow:not(#a):not(#b):not(#c) #modelrow > #working {order: 3; flex: 0 0 auto !important; font-size: .72rem}
body.narrow:not(#a):not(#b):not(#c) #chatcol {padding-left: 6px !important; padding-right: 6px !important}
body.narrow:not(#a):not(#b):not(#c) #chatcol > .row, body.narrow:not(#a):not(#b):not(#c) #chatcol .row:has(> #chatcol) {padding-left: 0 !important; padding-right: 0 !important}
body.narrow:not(#a):not(#b):not(#c) .row:has(> #chatcol) {padding: 0 !important}
body.narrow:not(#a):not(#b):not(#c) #promptrow .full-container {padding: 4px 6px !important}
body.narrow:not(#a):not(#b):not(#c) #starters .starters {gap: 5px; margin-bottom: 6px}
body.narrow:not(#a):not(#b):not(#c) button.starter {padding: 5px 10px; font-size: .8rem; border-radius: 10px}
body.narrow:not(#a):not(#b):not(#c) #modelrow > #working:not(:has(.prose *)):not(:has(.prose:not(:empty))) {display: none !important}
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


# Releases up to 2026.11.21.3.121 shipped one api.json by mistake, so every install that updated got the same key,
# readable by anyone in the public download. Only its fingerprint is kept here: a key that matches it is replaced.
SHIPPED_KEYS = {"6f5fb43c3a869308f23c9ba0f8ece6d5dd650dcdbda69a1668fa947d742cff9f"}


def api_settings(**change):
    data = json.loads(API_FILE.read_text()) if API_FILE.exists() else {}
    if hashlib.sha256(str(data.get("key", "")).encode()).hexdigest() in SHIPPED_KEYS:
        data.pop("key")  # the public one: this computer gets a key of its own
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
    key = html.escape(a["key"], quote=True)
    masked = html.escape(a["key"][:4] + "•" * 12 + a["key"][-4:])  # the key is a password: hidden until you ask to see it
    show = f"<button type='button' class='keyshow' onclick=\"const c = this.previousElementSibling; const on = c.dataset.shown === '1'; c.textContent = on ? c.dataset.masked : c.dataset.key; c.dataset.shown = on ? '' : '1'; this.textContent = on ? {html.escape(json.dumps(t('Show')), quote=True)} : {html.escape(json.dumps(t('Hide')), quote=True)}\">{t('Show')}</button>"
    copy = f"<button type='button' class='keyshow' onclick=\"navigator.clipboard.writeText('{key}'); this.textContent = {html.escape(json.dumps(t('Copied')), quote=True)}\">{t('Copy')}</button>"
    copyurl = f"<button type='button' class='keyshow' onclick=\"navigator.clipboard.writeText('{url}'); this.textContent = {html.escape(json.dumps(t('Copied')), quote=True)}\">{t('Copy')}</button>"
    return (f"<div class='cloudstat'><b>{t('Address')}</b> <code>{url}</code> {copyurl}<br><b>{t('Key')}</b> <code data-key='{key}' data-masked='{masked}'>{masked}</code> {show} {copy}<br>"
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
    if t.get("monitor"):
        return run_monitor(t, model)
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


def run_monitor(task, model):
    """A monitor's run: only what is new since the last run. Nothing new: no chat and no notification."""
    st = monitor.state(task["name"])
    seen, started = st["seen"], dt.datetime.now()
    after = monitor.since(task, st)
    items, failed = monitor.gather(task.get("sources") or [], set(seen), after)
    cid = new_conv()
    conv = load(cid)
    conv["title"] = f"📡 {task['name']} · {started:%d/%m %H:%M}"
    conv["agent"] = {"query": task["prompt"], "evidence": "\n\n".join(f"[{i['source']}] {i['title']}\n{i['link']}" for i in items)}
    save(cid, conv)
    send({"text": monitor.prompt(task, items, seen, after)}, cid, model, False, False, 10, True, 120, False, 0, True, False, share_memory_setting(), task.get("desk", ""))

    def finish():
        while busy(cid):
            time.sleep(5)
        answer = (last_answer(load(cid)["display"]) or {}).get("content", "")
        fresh = monitor.new_links(answer, seen)
        if not answer.strip():  # something went wrong: the chat stays, to show what
            return
        st.update(since=started.isoformat(timespec="seconds"), last=started.isoformat(timespec="seconds"), failed=failed)
        if monitor.NOTHING in answer or not fresh:
            st["result"] = 0
            monitor.save(task["name"], st)
            JOBS.pop(cid, None)
            shutil.rmtree(ROOT / cid, ignore_errors=True)  # nothing new: the run leaves no chat
            return
        n = sum(1 for line in answer.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s", line) and "http" in line) or len(fresh)
        st["seen"], st["result"] = seen + fresh, n
        monitor.save(task["name"], st)
        if task.get("notify", True):
            notify(f"Joshua · {task['name']}", tr("{n} new since the last check", n=n))
    threading.Thread(target=finish, daemon=True).start()


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
        if task.get("monitor"):
            st = monitor.state(task["name"])
            if st.get("last"):
                found = st.get("result", 0)
                last = " · " + t("last check {when}: {what}", when=f"{dt.datetime.fromisoformat(st['last']):%d/%m %H:%M}", what=t("{n} new", n=found) if found else t("nothing new"))
            if st.get("failed"):
                last += "<br>⚠ " + t("Couldn't read: {sites}", sites=html.escape(", ".join(st["failed"])))
        rows.append(f"<div class='task'><b>{'📡 ' if task.get('monitor') else ''}{html.escape(task['name'])}</b><br>{describe(task)}<br><span>{state}{last}</span>"
                    + f"<div class='taskp'>{html.escape(task['prompt'][:160])}</div></div>")
    names = [task["name"] for task in tasks_load()]
    warn = "<div class='cloudstat'>" + t("⚠ Keep Joshua open and the computer awake at those times.") + "</div>" if rows else ""
    return ("".join(rows) + warn or "<div class='cloudstat schedempty'>" + t("Your tasks will appear here once you save one.") + "</div>"), gr.Dropdown(choices=names, value=None), gr.Column(visible=bool(names))


def pause_label(name):
    task = next((x for x in tasks_load() if x["name"] == name), None)
    return gr.Button(t("Resume") if task and not task.get("on", True) else t("Pause"))


def task_add(name, prompt, when, date, at, days, every, offline, notify_on, model, watch=False, sources=""):
    name, prompt = (name or "").strip(), (prompt or "").strip()
    ok = gr.Textbox(info=None, elem_classes=[])  # a field that was marked as missing goes back to normal
    bad = lambda message: gr.Textbox(info="⚠ " + message, elem_classes=["invalid"])  # the reason goes under the field, not in a corner
    if not name or not prompt or not re.fullmatch(r"\d{1,2}:\d{2}", at or ""):
        return (gr.skip(), gr.skip(), gr.skip(), bad(t("Give the task a name.")) if not name else ok, bad(t("Write what Joshua should do.")) if not prompt else ok,
                bad(t("Write the time as HH:MM, for example 07:30.")) if not re.fullmatch(r"\d{1,2}:\d{2}", at or "") else ok)
    fields_ok = (ok, ok, ok)
    at = f"{int(at.split(':')[0]):02d}:{at.split(':')[1]}"
    kind = when  # the radio's values are once / daily / weekly / hours
    if kind == "once":
        try:
            if dt.datetime.fromisoformat(f"{date}T{at}") <= dt.datetime.now():
                gr.Warning(t("That moment has already passed."))
                return (gr.skip(),) * 6
        except ValueError:
            gr.Warning(t("Write the date as YYYY-MM-DD, for example 2026-11-20."))
            return (gr.skip(),) * 6
    if watch and offline:
        gr.Warning(t("A monitor needs the internet: turn off 'No internet for this task'."))
        return (gr.skip(),) * 6
    if kind == "weekly" and not days:
        gr.Warning(t("Pick at least one day."))
        return (gr.skip(),) * 6
    task = {"name": name, "prompt": prompt, "when": kind, "date": date, "time": at, "days": days, "every": int(every),
            "offline": bool(offline), "notify": bool(notify_on), "model": model or None,
            "monitor": bool(watch), "sources": [u.strip() for u in (sources or "").split() if u.strip().startswith(("http://", "https://"))],
            "created": dt.datetime.now().isoformat(timespec="seconds"), "on": True}
    with SCHED_LOCK:
        tasks = [t for t in tasks_load() if t["name"] != name]  # same name: replaced
        tasks_save(tasks + [task])
    gr.Info(t("Saved: {name}, {when}. It runs only while Joshua is open and the computer is on.", name=name, when=describe(task)))
    return tasks_view() + fields_ok


def sched_title(name):
    """The form's heading says whether you are making a new task or changing one of yours."""
    return "<div class='schedsub'>" + (t("Changing: {name}", name=html.escape(name)) if name else t("New task")) + "</div>"


def task_fill(name):
    """Puts a task's settings in the form, so it can be changed and saved again."""
    t = next((t for t in tasks_load() if t["name"] == name), None)
    if not t:
        return (gr.skip(),) * 12
    when = t["when"]
    return (t["name"], t["prompt"], when, t.get("date") or time.strftime("%Y-%m-%d"), t.get("time", "07:30"), t.get("days", []), t.get("every", 6),
            t.get("offline", False), t.get("notify", True), t.get("model") or default_model(), t.get("monitor", False), gr.Textbox(value="\n".join(t.get("sources", [])), visible=t.get("monitor", False)))


def task_action(name, action):
    if not name:
        gr.Warning(tr("Pick a task first."))
        return gr.skip(), gr.skip(), gr.skip()
    with SCHED_LOCK:
        tasks = tasks_load()
        t = next((t for t in tasks if t["name"] == name), None)
        if t and action == "delete":
            tasks.remove(t)
            monitor.forget(name)  # a monitor's memory of what it reported goes with it
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
forget_games()  # games left from the last session
bin_purge()  # chats deleted more than 30 days ago


def css_text(text):
    """A text for a CSS content: "..." rule."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


CSS = CSS.replace("@@WHAT_HELP@@", css_text(t("Hello. What would you like to do today?"))).replace("@@ASK_HINT@@", css_text(t("Everything you write here is stored only on this computer."))).replace("@@MODEL@@", css_text(t("Model:")))
CSS += "\n" + "\n".join(re.sub(r"#tune(?=[\s.,{:])", "#podcast", line).replace("#tunetitle", "#podtitle") for line in CSS.splitlines() if re.search(r"#tune(title)?(?=[\s.,{:])", line) and line.count("{") == 1 and line.rstrip().endswith("}"))  # whole one-line rules only  # the Podcast page looks like the fine-tuning one
CSS += "\n#podcast .podline {flex: 1} #podcast .podvoices {opacity: .8; margin: 4px 0 8px} #podscript textarea {font-family: 'IBM Plex Mono', monospace; font-size: .92rem; line-height: 1.55}"
CSS += "\nbody:not(#a):not(#b):not(#c):not(#d):not(#e) #podcast .podfile button {height: auto !important; min-height: 96px !important; width: 100% !important; white-space: normal !important}"  # the drop area is a button: not the small size of the page's buttons
CSS += "\n" + "\n".join(line.replace("#modelspage", pid) for pid in ("#mempage", "#setpage", "#binpage", "#teampage", "#readpage", "#podcast") for line in CSS.splitlines() if "#modelspage" in line and line.count("{") == 1)
CSS += "\n" + "\n".join(line.replace("#tunebtn", b) for b in ("#membtn", "#setbtn", "#binbtn", "#teambtn", "#readbtn", "#newmodelbtn", "#podbtn") for line in CSS.splitlines() if "#tunebtn" in line and line.count("{") == 1)  # Memory and Settings look like Teach
CSS += "\n#modelspage .mz0, #setpage .mz0 {background: color-mix(in srgb, var(--ink) 4%, var(--bg)) !important}"
CSS += """
/* ---------- getting around: one ☰ in the top-left corner on every page, a Back to chat that stays in view, the page links always on screen ---------- */
body:not(#a):not(#b):not(#c) #sidetoggle {position: fixed !important; z-index: 950 !important; top: 12px !important; left: 12px !important; width: 40px !important; height: 40px !important;
  font-size: 1.3rem !important; opacity: .85; border-radius: 10px !important; background: color-mix(in srgb, var(--panel) 85%, transparent) !important; color: var(--ink) !important}
body:not(#a):not(#b):not(#c) #sidetoggle:hover, body:not(#a):not(#b):not(#c) #sidetoggle:focus-visible {opacity: 1; background: color-mix(in srgb, var(--ink) 10%, var(--panel)) !important}
body:not(.noside):not(.narrow):not(#a):not(#b):not(#c) #sidetoggle {left: 256px !important}  /* with the side column open, the ☰ sits in its top-right corner, clear of the logo */
body.short:not(.narrow) #leftcol .status {display: none !important}  /* a short window: more room for the chats */
body.noside #chatbar, body.narrow #chatbar {padding-left: 52px !important; min-height: 44px !important}
body.noside:not(#a):not(#b):not(#c):not(#d):not(#e) .pagehead, body.narrow:not(#a):not(#b):not(#c):not(#d):not(#e) .pagehead {padding-left: 60px !important}  /* room for the ☰ */
body:not(#a):not(#b):not(#c):not(#d) .pagehead {position: sticky !important; top: -14px; z-index: 60;  /* above the field labels Gradio lifts to 40 */ align-items: center !important; flex-wrap: nowrap !important; gap: 10px !important; margin: -14px -18px 0 !important; padding: 14px 18px 8px !important;
  background: var(--panel) !important; border-bottom: 1px solid color-mix(in srgb, var(--line) 35%, transparent) !important}
.pagehead > * {flex: 0 1 auto !important; min-width: 0 !important}
.pagehead > :first-child {flex: 1 1 auto !important}
body:not(#a):not(#b):not(#c) .pagehead button.backbtn {display: inline-flex !important; white-space: nowrap; flex: 0 0 auto !important}
/* the side column: logo and New chat on top, the chats scroll in the middle, the page links stay at the bottom */
body:not(.narrow):not(.noside) #leftcol {height: calc(100vh - 20px) !important; max-height: calc(100vh - 20px) !important; overflow-y: auto !important; display: flex !important; flex-direction: column !important}
body.narrow.mside #leftcol {overflow: hidden !important; display: flex !important; flex-direction: column !important}
body:not(#a):not(#b):not(#c) #leftcol > #convs, body:not(#a):not(#b):not(#c) #leftcol > .form:has(#convs) {flex: 1 1 auto !important; min-height: 80px !important; overflow-y: auto !important; overscroll-behavior: contain}
body:not(#a):not(#b):not(#c) #leftcol > #panels {flex: 0 0 auto !important; margin-top: 6px !important; padding-top: 8px !important; border-top: 1px solid color-mix(in srgb, var(--line) 35%, transparent) !important}
body:not(#a):not(#b):not(#c):not(#d) #panels button {padding-top: .3rem !important; padding-bottom: .3rem !important; min-height: 0 !important}
body:not(#a):not(#b):not(#c) #navgap {height: 6px !important}
body:not(#a):not(#b):not(#c) #panels button[aria-current="page"] {background: var(--soft, color-mix(in srgb, var(--cyan) 14%, transparent)) !important; font-weight: 600 !important}
body:not(#a):not(#b):not(#c) #panels button:focus:not(:focus-visible):not([aria-current]) {background: transparent !important}
body.noside:not(.narrow) #chatbar {min-height: 52px !important}
body:not(.narrow):not(#a):not(#b):not(#c):not(#d):not(#e) #deskpick {flex: 0 1 auto !important; max-width: 100% !important}
body:not(.narrow):not(#a):not(#b):not(#c):not(#d):not(#e) #deskpick .wrap {flex-wrap: wrap !important; border-radius: 18px !important}  /* many desks: a second line, never cut off */
/* help under a field: readable, not light blue */
body:not(#a):not(#b):not(#c) :is(#schedpanel, #modelspage, #mempage, #setpage, #teampage, #tune, #podcast, #binpage, #readpage) .info-text {color: color-mix(in srgb, var(--ink) 68%, transparent) !important; font-size: .84rem !important; line-height: 1.35 !important}
/* placeholders are examples: lighter and in italics, so they never look like something already filled in */
body:not(#a):not(#b) :is(#schedpanel, #modelspage, #mempage, #setpage, #teampage, #tune, #podcast, #readpage) :is(input, textarea)::placeholder {opacity: .55 !important; font-style: italic}
/* a field the wizard is waiting for: red border, the reason in red under it */
body:not(#a):not(#b):not(#c) .invalid :is(input, textarea) {border: 1.5px solid var(--red, #b3261e) !important}
body:not(#a):not(#b):not(#c):not(#d):not(#e) .invalid :is(.info-text, [data-testid="block-info"], .info) {color: var(--red, #b3261e) !important; font-weight: 500}
body:not(#a):not(#b):not(#c):not(#d):not(#e) #teampage #twcancel {background: transparent !important; border: none !important; box-shadow: none !important; text-decoration: underline}
#readpage #rdtop {flex: 1 1 100%; display: flex; align-items: center; gap: 12px; min-width: 0}
body:not(#a):not(#b):not(#c):not(#d) #readpage #rdbar button#rdback {width: auto; height: 32px; padding: 0 12px; color: var(--brand); flex: 0 0 auto}
#readpage #rdname {font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0}
/* a long page gets a row of links to its parts */
.pagenav {display: flex; flex-wrap: wrap; gap: 6px 8px; margin: 2px 0 6px}
.pagenav a {padding: 4px 12px; border-radius: 999px; border: 1px solid color-mix(in srgb, var(--line) 80%, transparent); color: var(--brand, var(--cyan)) !important; text-decoration: none !important; font-size: .88rem}
.pagenav a:hover {background: color-mix(in srgb, var(--brand, var(--cyan)) 10%, transparent)}
/* model names on one or two lines, never one letter per line */
body:not(#a):not(#b):not(#c) table.mrank td.mname {min-width: 10.5rem; word-break: normal !important; overflow-wrap: anywhere}
body:not(#a):not(#b):not(#c) table.mrank .mdefault {display: inline-block; margin-top: 2px}
#modelspage #mintro details summary, #setpage details summary {cursor: pointer; color: var(--brand, var(--cyan))}
#okeyrow {align-items: center !important}
#ftres {min-height: 62px}  /* the computer check fills this in: the page doesn't jump when it does */
.ftidle {color: color-mix(in srgb, var(--ink) 60%, transparent); font-size: .9rem; padding: 4px 0}
#binpage .binrow {display: flex; align-items: center; gap: 12px; flex-wrap: wrap}
#binpage .binmain {flex: 1 1 14rem; min-width: 0}
#binpage .binbtns {display: flex; gap: 8px; flex: 0 0 auto}
#binpage button.binact {padding: 5px 12px; border-radius: 8px; border: 1px solid var(--line); background: transparent; color: var(--ink); cursor: pointer; font-size: .88rem}
#binpage button.binact.primary {background: var(--brand, var(--cyan)); border-color: transparent; color: #fff}
#binpage button.binact.red {color: var(--red, #b3261e)}
body:not(#a):not(#b):not(#c):not(#d) #binempty {margin-top: 14px; color: var(--red, #b3261e) !important}
#setpage .dkcols {display: grid; grid-template-columns: 90px 2fr 5fr 40px; gap: 6px; font-size: .8rem; color: color-mix(in srgb, var(--ink) 60%, transparent); margin: 2px 0 0}
body:not(#a):not(#b):not(#c):not(#d) #setpage .dkrow {flex-wrap: nowrap !important}  /* the ✕ stays at the end of its own row */
#dknote {flex: 1 1 12rem !important; min-width: 10rem !important; align-self: center}
button.keyshow {margin-left: 4px; padding: 1px 8px; border-radius: 6px; border: 1px solid var(--line); background: transparent; color: var(--ink); cursor: pointer; font-size: .82rem}
/* a slider's heading: its name on the left, the number and ↺ on the right, all on one line and nothing cut off */
body:not(#a):not(#b):not(#c) .head:has(> .tab-like-container) {display: flex !important; align-items: center !important; gap: 8px !important; flex-wrap: nowrap !important; overflow: visible !important}
body:not(#a):not(#b):not(#c) .head:has(> .tab-like-container) > label {flex: 1 1 auto !important; min-width: 0 !important}
body:not(#a):not(#b):not(#c) .tab-like-container {display: flex !important; align-items: center !important; gap: 4px !important; flex: 0 0 auto !important; height: auto !important; overflow: visible !important}
body:not(#a):not(#b):not(#c) .tab-like-container input[type=number] {width: 4.2em !important; height: 28px !important; padding: 2px 6px !important}
body:not(#a):not(#b):not(#c) .tab-like-container .reset-button {width: 28px !important; height: 28px !important; padding: 0 !important; overflow: visible !important; line-height: 1 !important}
.block:has(input[type=range]) {overflow: visible !important}
/* a dropdown's text never runs under its arrow */
body:not(#a):not(#b):not(#c) .wrap-inner:has(.icon-wrap) input {padding-right: 26px !important; text-overflow: ellipsis !important}
body:not(#a):not(#b):not(#c) .ftnewrow {align-items: flex-end !important}
body:not(#a):not(#b):not(#c):not(#d) .ftnewrow button {height: 42px !important; min-height: 0 !important; align-self: flex-end !important; margin-bottom: 4px}
#jtitle {display: none}
#jtitle.on {display: flex; align-items: center; gap: 8px; position: fixed; z-index: 940; top: 12px; left: 60px; height: 40px; max-width: calc(100vw - 180px)}
#jtitle .jnew {width: 40px; height: 40px; border-radius: 10px; border: none; background: color-mix(in srgb, var(--panel) 85%, transparent); color: var(--ink); font-size: 1.2rem; cursor: pointer; opacity: .8}
#jtitle .jnew:hover {opacity: 1; background: color-mix(in srgb, var(--ink) 10%, var(--panel))}
#jtitle .jname {font-weight: 600; font-size: .95rem; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
body.noside:not(.narrow) #chatbar {padding-left: 0 !important}
body.narrow:not(#a):not(#b) :is(#binpage, #podcast) {min-width: 0 !important; width: 100vw !important; max-width: 100vw !important; flex: 1 1 100% !important; padding: 8px 10px 6px !important; border-radius: 0 !important}
"""
# ---------- UX review of 28 November 2026: one set of rules, last, so it wins over the older ones above ----------
# (the long :not(#…) chains only raise the priority of a rule; they match every page)
X = "body:not(#x1):not(#x2):not(#x3):not(#x4):not(#x5):not(#x6)"
PAGES_SEL = ":is(#modelspage, #setpage, #mempage, #schedpanel, #binpage, #readpage, #teampage, #podcast, #tune, #chatopts)"
CSS += ("""
/* the side column: small headings over three groups; the chats get the room they need, with a scrollbar you can see */
X .navheadwrap, X #chatsheadwrap {padding: 0 !important; margin: 0 !important; min-height: 0 !important; border: none !important; background: transparent !important; box-shadow: none !important}
X :is(.navheadwrap, #chatsheadwrap) :is(.html-container, .prose) {padding: 0 !important; margin: 0 !important; min-height: 0 !important}
X #panels button {padding-top: .22rem !important; padding-bottom: .22rem !important}
body.short X #brand {zoom: .78; padding-top: 14px !important}
body.short X #leftcol > .block:has(#brand) {margin-bottom: 2px !important}
X .navhead {display: flex; align-items: center; gap: 6px; width: 100%; box-sizing: border-box; padding: 7px 10px 2px; margin: 0; font: inherit; font-size: .74rem !important; font-weight: 600; letter-spacing: .03em;
  color: color-mix(in srgb, var(--ink) 55%, transparent) !important; background: transparent !important; border: none !important; text-align: left; text-transform: none !important}
X button.navfold {cursor: pointer; border-radius: 8px !important}
X button.navfold:hover {color: var(--ink) !important; background: color-mix(in srgb, var(--ink) 6%, transparent) !important}
X button.navfold::after {content: "▸"; margin-left: auto; font-size: .8rem}
X button.navfold[aria-expanded="true"]::after {content: "▾"}
body:not(.navadv-open) #panels button.navadv:not([aria-current]) {display: none !important}
X #chatshead {padding-top: 2px; padding-bottom: 4px}
X #chatshead input.chatsearch {flex: 1 1 auto; min-width: 0; margin-left: 8px; padding: 3px 8px; font: inherit; font-size: .8rem; font-weight: 400; letter-spacing: 0; border-radius: 8px;
  border: 1px solid color-mix(in srgb, var(--line) 90%, transparent); background: var(--panel); color: var(--ink)}
X #leftcol > .form:has(#convs), X #leftcol > #convs {flex: 1 1 0 !important; min-height: 196px !important}
X #convs {overflow-y: auto !important; scrollbar-width: thin; scrollbar-color: color-mix(in srgb, var(--ink) 30%, transparent) transparent; scrollbar-gutter: stable}
#convs::-webkit-scrollbar {width: 8px}
#convs::-webkit-scrollbar-thumb {background: color-mix(in srgb, var(--ink) 28%, transparent); border-radius: 8px}
#convs::-webkit-scrollbar-track {background: transparent}
X #convs label.grpstart {margin-top: 18px !important}
X #convs label.grpstart:first-child {margin-top: 14px !important}
X #convs label.grpstart::after {content: attr(data-group); position: absolute; left: 10px; top: -17px; font-size: .7rem; font-weight: 600; letter-spacing: .03em; pointer-events: none;
  color: color-mix(in srgb, var(--ink) 50%, transparent); background: none !important}
X #convs label.jhide {display: none !important}
/* the chat's ⋯ never covers its name: the name stops before it */
X #convs label {padding-right: 34px !important}
X #convs label > span:not(.cmore):not(.ji):not(.pinsvg) {max-width: calc(100% - 4px) !important}
X #convs .cmore {background: var(--side, var(--panel)); right: 6px}
X #convs input[type=radio] {display: block !important; position: absolute !important; opacity: 0 !important; width: 1px !important; height: 1px !important; pointer-events: none}  /* hidden, but the keyboard still reaches the list */
X #convs label:focus-within {box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--brand) 55%, transparent) !important}

/* the model menu under the message box: wide enough for the names, solid, with where each model runs */
X #model .options:not(.option-list) {width: max-content !important; min-width: 300px !important; max-width: min(460px, 92vw) !important; left: auto !important; right: 14px !important;
  background: var(--panel) !important; opacity: 1 !important; border: 1px solid color-mix(in srgb, var(--line) 90%, transparent) !important; border-radius: 12px !important;
  box-shadow: 0 12px 32px rgba(0, 0, 0, .18) !important; padding: 6px !important; font-size: .9rem !important; z-index: 2147480001 !important}
X #model .options > ul.option-list {width: auto !important; min-width: 0 !important; max-width: none !important; border: none !important; box-shadow: none !important; padding: 0 !important; margin: 0 !important; background: transparent !important; position: static !important}
X #model .options li {width: auto !important; white-space: nowrap !important; display: flex !important; align-items: center; gap: 6px; padding: 7px 10px !important; border-radius: 8px; background: transparent}
X #model .options li:hover, X #model .options li.active {background: color-mix(in srgb, var(--brand) 10%, transparent) !important}
X #model .options li[data-kind]::after {content: attr(data-kind); margin-left: auto; padding-left: 18px; font-size: .74rem; color: color-mix(in srgb, var(--ink) 55%, transparent)}
X #model .options li.jadd {border-top: 1px solid color-mix(in srgb, var(--line) 90%, transparent); margin-top: 4px; border-radius: 0 0 8px 8px; color: var(--brand)}

/* the chat's bar: its files have a named button of their own, next to the ⋯ */
X #chatbarbtns {display: flex !important; align-items: center; gap: 4px; padding: 0 !important; border: none !important; background: transparent !important; box-shadow: none !important; min-width: 0 !important; flex: 0 0 auto !important}
X #chatbarbtns > div, X #chatbarbtns .prose {display: flex !important; align-items: center; gap: 4px}
X button.fpill {display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 12px; border-radius: 999px; border: 1px solid color-mix(in srgb, var(--line) 90%, transparent);
  background: transparent; color: var(--dim); cursor: pointer; font: inherit; font-size: .85rem}
X button.fpill:hover {color: var(--ink); background: color-mix(in srgb, var(--ink) 6%, transparent)}
X button.fpill svg {width: 15px; height: 15px}
body[data-files="open"] X button.fpill, X body[data-files="open"] button.fpill {color: var(--brand); border-color: color-mix(in srgb, var(--brand) 55%, transparent)}
/* the files panel floats over the chat instead of pushing it down, and has a ✕ */
X #chatopts {position: absolute !important; top: 52px; right: 18px; width: min(500px, calc(100% - 36px)) !important; z-index: 70; margin: 0 !important; padding: 14px 16px !important;
  border: 1px solid color-mix(in srgb, var(--line) 90%, transparent) !important; border-radius: 14px !important; background: var(--panel) !important; box-shadow: 0 14px 36px rgba(0, 0, 0, .18) !important}
X #chatopts .fhead {display: flex; align-items: center; gap: 8px}
X #chatopts .fclose {margin-left: auto; width: 30px; height: 30px; border-radius: 8px; border: none; background: transparent; color: var(--dim); cursor: pointer; font-size: 1rem}
X #chatopts .fclose:hover {background: color-mix(in srgb, var(--ink) 8%, transparent); color: var(--ink)}
X #projrow {align-items: flex-end !important}
X #projrow #pickproj {margin-bottom: 30px; align-self: flex-end}

/* the message box and what's under it always stay on screen, whatever is above them */
X #chatcol {overflow: hidden !important; min-height: 0 !important}
X #chatcol > .block:has(#chat), X #chatcol > #chat {flex: 1 1 0 !important; min-height: 0 !important}
X #chatcol:has(#starters .wopr) > .block:has(#chat), X #chatcol:has(#starters .wopr) > #chat {flex: 0 0 0 !important; height: 0 !important; min-height: 0 !important; overflow: hidden !important; border: none !important}
X #chatcol:has(#starters .wopr) #starters {flex: 1 1 0 !important; min-height: 0 !important; overflow-y: auto !important; padding: 8px 0 !important}
X #starters .wopr {max-height: none}
/* a desk's starter cards sit right under the greeting, not far below it */
X #chatcol:has(#starters .starters):has(#chat .placeholder) #chat .placeholder {justify-content: flex-end !important; padding-bottom: 12px !important}
X #chatcol:has(#starters .starters):has(#chat .placeholder) #starters {margin-top: -6px !important}
/* the desks never push the switches and the model onto more lines: on a smaller window they scroll sideways */
X #deskpick {flex: 0 1 auto !important; min-width: 0 !important; max-width: 100% !important}
X #deskpick .wrap {flex-wrap: nowrap !important; overflow-x: auto !important; scrollbar-width: none; max-width: 100%}
X #deskpick .wrap::-webkit-scrollbar {display: none}
X #deskpick label {flex: 0 0 auto !important; white-space: nowrap !important}
@media (max-width: 1180px) {X #modelrow {flex-wrap: wrap !important} X #modelrow > #deskpick {flex: 1 1 100% !important}}
/* what can the model do: icons a little bigger */
X #modelrow .bicon svg, X #modelrow .bicon .ji svg {width: 1.05em !important; height: 1.05em !important}
/* while Joshua works: the status shows just above the message box, and the ■ takes the place of the send arrow */
X #modelrow #working .working {display: none !important}
#jstatus {position: absolute; left: 50%; width: max-content; transform: translateX(-50%); z-index: 40; display: none; align-items: center; gap: 8px; max-width: calc(100% - 40px); padding: 6px 14px; border-radius: 999px;
  background: var(--panel); border: 1px solid color-mix(in srgb, var(--line) 90%, transparent); box-shadow: 0 6px 18px rgba(0, 0, 0, .1); font-size: .88rem; color: var(--dim);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none}
#jstatus.on {display: flex}
#jstatus b.clock {color: var(--ink); font-variant-numeric: tabular-nums}
#jstatus .wdot {flex: 0 0 auto; width: 8px; height: 8px; border-radius: 50%; background: var(--brand); animation: slowblink 1.6s ease-in-out infinite; box-shadow: none}
X #promptrow {position: relative !important}
X #promptrow #stop {position: absolute !important; right: 12px; top: 50%; transform: translateY(-50%); z-index: 6; width: 36px !important; height: 36px !important; min-width: 0 !important; max-width: none !important;
  padding: 0 !important; border-radius: 50% !important; font-size: 0 !important; background: var(--red, #b3261e) !important; border: none !important; color: #fff !important}
X #promptrow #stop::before {content: ""; width: 11px; height: 11px; border-radius: 2px; background: #fff; display: block; margin: auto}
X #chatcol:has(#working .working) #promptrow .submit-button {visibility: hidden !important}
/* under an answer: copy, try again and read aloud are always there, a little bigger */
X #chat .message-buttons, X #chat .message-buttons-row, X #chat [class*="message-buttons"] {opacity: 1 !important; visibility: visible !important}
X #chat .message-buttons button, X #chat .message-buttons .icon-button {opacity: .75; width: 28px !important; height: 28px !important}
X #chat .message-buttons button:hover {opacity: 1}
X #chat .message-buttons svg {width: 16px !important; height: 16px !important}

/* ---------- every page: one look for buttons, choices, titles and help ---------- */
X PAGES button.secondary:not(.backbtn):not(.mhidden):not(#twcancel):not(.ftmain) {background: transparent !important; border: 1px solid color-mix(in srgb, var(--brand) 45%, var(--line)) !important; color: var(--brand) !important; box-shadow: none !important}
X PAGES button.secondary:not(.backbtn):not(.mhidden):not(#twcancel):not(.ftmain):hover {background: color-mix(in srgb, var(--brand) 8%, transparent) !important}
X PAGES button.primary, X PAGES button.ftmain {background: var(--brand) !important; border: 1px solid var(--brand) !important; color: #fff !important}
X PAGES button.danger, X button.danger, X PAGES button.mact.danger, X #binpage button.binact.red {background: transparent !important; color: var(--red, #b3261e) !important; border: 1px solid color-mix(in srgb, var(--red, #b3261e) 50%, transparent) !important}
X PAGES button.danger:hover, X button.danger:hover, X #binpage button.binact.red:hover {background: color-mix(in srgb, var(--red, #b3261e) 8%, transparent) !important}
X PAGES > button:not(.mhidden), X PAGES .column > button:not(.mhidden), X PAGES .form > button:not(.mhidden) {width: auto !important; align-self: flex-start !important; min-width: 160px; flex-grow: 0 !important}
X #schedform > button {align-self: flex-start !important}
/* a choice of one: the chosen one filled; several ticks: tinted, with the tick */
X PAGES fieldset:not(#ftsteps) .wrap label {border-radius: 999px !important; border: 1px solid color-mix(in srgb, var(--line) 95%, transparent) !important; background: transparent !important; color: var(--ink) !important; box-shadow: none !important}
X PAGES fieldset:not(#ftsteps) .wrap label.selected:has(input[type=radio]) {background: var(--brand) !important; border-color: var(--brand) !important; color: #fff !important}
X PAGES fieldset:not(#ftsteps) .wrap label.selected:has(input[type=radio]) * {color: #fff !important}
X PAGES fieldset:not(#ftsteps) .wrap label.selected:has(input[type=checkbox]) {background: color-mix(in srgb, var(--brand) 12%, transparent) !important; border-color: color-mix(in srgb, var(--brand) 60%, transparent) !important; color: var(--ink) !important}
X PAGES fieldset:not(#ftsteps) .wrap label.selected:has(input[type=checkbox]) * {color: var(--ink) !important}
X PAGES input[type=radio]:not(:checked), X PAGES input[type=radio]:not(:checked):hover {background-color: var(--panel) !important; background-image: none !important; border-color: color-mix(in srgb, var(--ink) 40%, transparent) !important}
X PAGES input[type=radio]:not(:checked):hover {box-shadow: 0 0 0 2px color-mix(in srgb, var(--brand) 35%, transparent) !important}
/* page titles in sentence case, and one size of section title */
X :is(#schedtitle, #tunetitle, #podtitle) {text-transform: none !important; letter-spacing: -.01em !important}
X PAGES h2.msec {font-size: 1.2rem !important; line-height: 1.25 !important; font-weight: 650 !important; color: var(--ink) !important; letter-spacing: 0 !important; margin: 0 0 2px !important}
X #modelspage :is(.mz1, .mz2, .mz3) {background: color-mix(in srgb, var(--ink) 4%, var(--bg)) !important}
/* a field's name over it, help under it: readable */
X PAGES .info-text, X PAGES [data-testid="block-info"] {color: color-mix(in srgb, var(--ink) 74%, transparent) !important; font-size: .86rem !important}
X .fieldlabel {font-size: .86rem !important; font-weight: 600 !important; color: var(--ink); margin: 8px 0 4px}
/* the links to a page's parts: "On this page:", then plain links */
X .pagenav {align-items: baseline; gap: 4px 14px !important; margin: 4px 0 10px !important}
X .pagenav .pnlab {font-size: .85rem; color: color-mix(in srgb, var(--ink) 60%, transparent)}
X .pagenav a {padding: 0 !important; border: none !important; border-radius: 0 !important; text-decoration: underline !important; text-underline-offset: 3px; background: none !important}
/* folded explanations: the arrow on the left, then a space */
X :is(#tune, #podcast, #modelspage, #setpage) details > summary {list-style: none; cursor: pointer}
X :is(#tune, #podcast, #modelspage, #setpage) details > summary::-webkit-details-marker {display: none}
X :is(#tune, #podcast, #modelspage, #setpage) details:not(.mcost) > summary::before {content: "▸"; display: inline-block; width: 1.1em}
X :is(#tune, #podcast, #modelspage, #setpage) details[open]:not(.mcost) > summary::before {content: "▾"}
X #szapi .label-wrap {flex-direction: row-reverse !important; justify-content: flex-end !important; gap: 8px !important}

/* Models */
X table.mmenu td.mbtns {text-align: left !important; white-space: nowrap}
X table.mmenu .marrows {display: inline-flex; gap: 4px; width: 76px; margin-right: 6px; vertical-align: middle}
X table.mrank td.mscorecell {white-space: nowrap}
X table.mrank .mscore {display: inline-flex; align-items: center; gap: 6px; white-space: nowrap}
X table.mrank .mscore .mnum {min-width: 2.2em; text-align: left}
X h4.mrec {margin: 6px 0 4px; font-size: .95rem; font-weight: 600; color: var(--ink)}
X details.mall {margin-top: 8px}
X details.mall > summary {color: var(--brand); padding: 6px 0}
X .cloudok {color: var(--brand) !important}
X details.mcost summary::before {content: none !important}
X details.mcost summary {font-size: .78rem; padding: 1px 8px; border: 1px solid color-mix(in srgb, var(--brand) 40%, transparent); border-radius: 999px}
X details.mcost summary::after {content: " ▾"}
X details.mcost[open] summary::after {content: " ▴"}
X details.mcost .pwrap {position: static !important; width: auto !important; max-width: 320px !important; margin: 6px 0 2px !important; box-shadow: none !important}
X details.mcost[open] {display: block !important; margin-left: 0 !important; margin-top: 4px}
X details.mcost[open] > summary {display: inline-block}
X #okeyrow, X #kkeyrow, X #kaddrow {align-items: flex-end !important}
X :is(#okeyrow, #kkeyrow, #kaddrow) > button {margin-bottom: 4px}

/* Scheduled tasks: with none yet, the form has the whole width */
X #schedrow:has(.schedempty) > #schedlist {display: none !important}
X #schedrow:has(.schedempty) > #schedform {flex: 1 1 100% !important; max-width: 760px}
X .schedsub {text-transform: none !important; letter-spacing: 0 !important; font-size: 1rem !important; font-weight: 600 !important; color: var(--ink) !important}
X #schedopts {gap: 2px !important; padding: 0 !important; border: none !important; background: transparent !important}
X #schedform fieldset .wrap {flex-wrap: wrap !important}
X #schedform fieldset .wrap label {padding: 4px 12px !important}

/* Teams */
X .whenbox {margin: 6px 0 12px; padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--brand) 6%, var(--bg)); font-size: .9rem; line-height: 1.45}
X .whenbox ul {margin: 4px 0 0; padding-left: 18px}
X .namerow {align-items: flex-start !important}

/* Reader */
X .rdrow[role=button] {cursor: pointer}
X .rdrow[role=button]:focus-visible {outline: 2px solid var(--brand); outline-offset: 2px}
X .rdrow .rdplayicon {flex: 0 0 auto; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; margin-right: 10px; border-radius: 50%;
  background: color-mix(in srgb, var(--brand) 12%, transparent); color: var(--brand); font-size: .8rem}
X .rdrow:hover .rdplayicon {background: var(--brand); color: #fff}
X .rdrow .rdname {white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
X #rdbar .rdlab {display: inline-flex; flex-direction: column; gap: 1px}
X #rdbar .rdlab > span {font-size: .7rem !important; line-height: 1.1 !important; color: color-mix(in srgb, var(--ink) 60%, transparent) !important}
X #rdbar {box-shadow: 0 6px 12px -8px rgba(0, 0, 0, .25) !important; border-bottom: 1px solid color-mix(in srgb, var(--line) 90%, transparent) !important}

/* Podcast */
X #podcast .podspeaker {align-items: flex-end !important}
X #podcast .podspeaker > button {margin-bottom: 4px}
X #podcast input[type=range]:disabled {opacity: .4}
X #podcast .podvoices a {color: var(--brand)}

/* Memory: the notes one per row, each with Change and Delete; the whole text under "Edit as text" */
X #memlist .memrow {display: flex; align-items: flex-start; gap: 8px; padding: 7px 10px; border-bottom: 1px solid color-mix(in srgb, var(--line) 70%, transparent)}
X #memlist .memrow .memtxt {flex: 1 1 auto; min-width: 0; line-height: 1.4}
X #memlist .memrow input {flex: 1 1 auto; font: inherit; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: var(--ink)}
X #memlist .memrow button, X #memlist .memadd, X #memlist .memastext {flex: 0 0 auto !important; padding: 3px 10px !important; border-radius: 8px !important; font-size: .82rem !important; cursor: pointer;
  border: 1px solid color-mix(in srgb, var(--brand) 45%, var(--line)) !important; background: transparent !important; color: var(--brand) !important; width: auto !important; min-width: 0 !important; height: auto !important}
X #memlist .memrow button.memdel {color: var(--red, #b3261e) !important; border-color: color-mix(in srgb, var(--red, #b3261e) 45%, transparent) !important}
X #memlist .memtools {display: flex; gap: 8px; margin: 10px 0 4px; align-items: center}
X #memlist .memwarn {color: var(--red, #b3261e); font-size: .85rem}
X #memlist .memempty {padding: 8px 0; color: color-mix(in srgb, var(--ink) 65%, transparent)}
body:not(.memastext) #memtext {display: none !important}
X #memsave {margin-top: 6px}
X #pasthelp {margin-top: 18px !important}

/* Teach and Build */
body:not(.showtech) :is(.ftwhere, .ftarch) {display: none !important}
X .ftverdict {margin-top: 6px; padding: 6px 10px; border-radius: 8px; font-size: .9rem}
X .ftverdict.ok {background: color-mix(in srgb, var(--brand) 12%, transparent)}
X .ftverdict.warn {background: color-mix(in srgb, #d08a1e 16%, transparent)}
X .ftverdict.dim {background: color-mix(in srgb, var(--ink) 6%, transparent)}
X #tune .ftway {margin: 4px 0 10px !important; padding: 0 !important; border: none !important}
X #tune .ftmore > summary {margin-top: 6px}
X #tune #ftdel {margin-left: auto !important}

/* Bin */
X #binpage button.binact {height: 32px; padding: 0 14px}
/* the same Back to chat, and the same title size, on every page */
X button.backbtn {background: color-mix(in srgb, var(--ink) 6%, transparent) !important; border: none !important; box-shadow: none !important; color: var(--brand) !important}
X :is(#schedtitle, #tunetitle, #podtitle) {font-size: 1.3rem !important}
X #tune #ftdel.danger, X #tune button.danger {color: var(--red, #b3261e) !important; border: 1px solid color-mix(in srgb, var(--red, #b3261e) 50%, transparent) !important; background: transparent !important}
X PAGES button.mact:not(.danger) {background: transparent !important; border: 1px solid color-mix(in srgb, var(--brand) 45%, var(--line)) !important; color: var(--brand) !important}
X #teampage #twcancel {border: none !important; background: transparent !important; text-decoration: underline}

/* Settings */
X #setpage #szlook fieldset {margin-bottom: 14px !important}
X #setpage #browserhelp {padding: 0 !important; margin: 10px 0 6px !important}
X #setpage .dkrow {align-items: flex-start !important}
X #setpage .dkrow > button.dkdel {align-self: center}
X #setpage .dkrow input {text-overflow: ellipsis}
X #setpage .dkrow + .dkrow {margin-top: 4px}
X #dknote {align-self: center !important}
X button.logcopy {height: 32px; padding: 0 14px; border-radius: 8px; border: 1px solid color-mix(in srgb, var(--brand) 45%, var(--line)); background: transparent; color: var(--brand); cursor: pointer; font: inherit; font-size: .88rem}
X .logcopywrap {flex: 0 0 auto !important; min-width: 0 !important; padding: 0 !important; border: none !important; background: transparent !important}
X #logbox textarea {overscroll-behavior: auto !important}
X .rurl {display: flex; align-items: center; gap: 6px; margin: 2px 0}
""".replace("X ", X + " ").replace("PAGES", PAGES_SEL))


CSS += """
#cmenu button[role=menuitemcheckbox] {display: flex; align-items: center; gap: 10px}
#cmenu .swi {flex: none; position: relative; width: 28px; height: 16px; border-radius: 999px; background: color-mix(in srgb, var(--ink) 22%, transparent)}
#cmenu .swi::after {content: ""; position: absolute; left: 2px; top: 2px; width: 12px; height: 12px; border-radius: 50%; background: #fff; box-shadow: 0 1px 2px rgba(0, 0, 0, .3)}
#cmenu .swi.on {background: var(--brand)}
#cmenu .swi.on::after {left: 14px}
"""
JS_TEXT = lambda text: json.dumps(text, ensure_ascii=False).replace("'", "\\'")  # a text inside the js="..." handlers below


with gr.Blocks(title="Joshua", fill_width=True) as demo:
    cid = gr.State(None)
    sidetoggle = gr.Button("☰", size="sm", elem_id="sidetoggle", min_width=0)  # shows or hides the side menu; fixed in the top-left corner, on every page
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=270, elem_classes="side", elem_id="leftcol"):
            gr.HTML("<div id='brand' role='img' aria-label='Joshua'><svg viewBox='0 0 100 100' aria-hidden='true'>" + LOGO + "</svg>OSHUA</div><div class='lights'><span></span><span></span><span></span><span></span><span></span></div><div class='status'>" + t("LOCAL SYSTEM · ONLINE") + "<br>" + t("VERSION") + " " + VERSION + "</div>")
            with gr.Row(visible=False, elem_id="updrow", equal_height=True) as updrow:  # one quiet line when a new version is out
                updnote = gr.HTML("")
                updbtn = gr.Button(t("Update"), size="sm", scale=0, min_width=72, elem_id="updbtn")
            new = gr.Button("＋  " + t("New chat"), size="sm", elem_id="newchat")
            mact = gr.Textbox(elem_classes="mhidden", show_label=False, container=False)  # the hidden link between buttons drawn in HTML (tables, chat menus) and Python
            mact_go = gr.Button(elem_id="mactgo", elem_classes="mhidden")
            pin_go = gr.Button(elem_id="pingo", elem_classes="mhidden")
            tochat_go = gr.Button(elem_id="tochatgo", elem_classes="mhidden")  # clicked by the page script when you click the chat that is already open, to close the page in front of it
            ren_val = gr.Textbox(elem_classes="mhidden", show_label=False, container=False)
            ren_go = gr.Button(elem_id="rengo", elem_classes="mhidden")
            ord_val = gr.Textbox(elem_classes="mhidden", show_label=False, container=False)
            ord_go = gr.Button(elem_id="ordgo", elem_classes="mhidden")
            gr.HTML("<div class='navhead' id='chatshead'>" + t("Chats") + "</div>", elem_id="chatsheadwrap")  # the page script adds a search field next to it
            convs = gr.Radio(choices=conv_choices(), label=t("Chats"), show_label=False, container=False, elem_id="convs")
            renbtn = gr.Button("✏  " + t("Rename this chat"), size="sm", elem_id="renchat")
            renbox = gr.Textbox(visible=False, show_label=False, max_lines=1, placeholder=t("New name, then press Enter"), elem_id="renbox")
            dele = gr.Button("🗑  " + t("Delete this chat"), size="sm", elem_id="delchat")
            with gr.Column(elem_id="panels"):  # three small groups: things you make, Joshua itself, and training models (folded until you open it)
                gr.HTML("<div class='navhead'>" + t("Create") + "</div>", elem_classes="navheadwrap")
                sched = gr.Button("⏰  " + t("Scheduled tasks"), size="sm", elem_id="schedbtn")
                team_btn = gr.Button("👥  " + t("Teams"), size="sm", elem_id="teambtn")
                read_btn = gr.Button("🎧  " + t("Reader"), size="sm", elem_id="readbtn")
                pod_btn = gr.Button("🎙  " + t("Podcast"), size="sm", elem_id="podbtn")
                rd_pos = gr.Textbox("", elem_id="rdpos", show_label=False, container=False)  # hidden: where the browser reports your place in the book, from any page
                gr.HTML("<div class='navhead'>" + t("Joshua") + "</div>", elem_classes="navheadwrap")
                models_btn = gr.Button("🧩  " + t("Models"), size="sm", elem_id="modelsbtn")
                mem_btn_open = gr.Button("🧠  " + t("Memory"), size="sm", elem_id="membtn")  # what Joshua knows about you
                bin_btn = gr.Button("🗑  " + t("Bin"), size="sm", elem_id="binbtn")
                set_btn = gr.Button("⚙  " + t("Settings"), size="sm", elem_id="setbtn")
                feedback = gr.Button("💬  " + t("Send feedback") + " ↗", size="sm", elem_id="fbbtn")  # ↗: it opens a page on the website
                gr.HTML("<button type='button' class='navhead navfold' id='advfold' aria-expanded='false'>" + t("Advanced") + "</button>", elem_classes="navheadwrap")
                tune = gr.Button("🧪  " + t("Teach a model your style"), size="sm", elem_id="tunebtn", elem_classes="navadv")
                newmodel = gr.Button("📚  " + t("Build a model from scratch"), size="sm", elem_id="newmodelbtn", elem_classes="navadv")
        tune_panel, tune_back, tune_res, tune_refresh, tune_mode, tune_title = tune_ui.build(ROOT, conv_choices)
        pod_panel, pod_back, (pod_open, pod_outs) = podcast_ui.build(ROOT, lambda: models(), podcast_write)
        RD_BAR = ("<div id='rdbar'><div id='rdtop'><button type='button' id='rdback' onclick=\"document.querySelector('#rdlib').click()\">← " + html.escape(t("Library")) + "</button><span id='rdname'></span></div>"
                  "<button id='rdprev' title='" + html.escape(t("Previous paragraph")) + "'><svg viewBox='0 0 256 256' fill='currentColor' aria-hidden='true'><path d='M208,47.88V208.12a16,16,0,0,1-24.43,13.43L64,146.77V216a8,8,0,0,1-16,0V40a8,8,0,0,1,16,0v69.23L183.57,34.45A15.95,15.95,0,0,1,208,47.88Z'/></svg></button>"
                            "<button id='rdplay' title='" + html.escape(t("Read aloud") + " / " + t("Pause")) + "'><svg viewBox='0 0 256 256' fill='currentColor' aria-hidden='true'><path d='M240,128a15.74,15.74,0,0,1-7.6,13.51L88.32,229.65a16,16,0,0,1-16.2.3A15.86,15.86,0,0,1,64,216.13V39.87a15.86,15.86,0,0,1,8.12-13.82,16,16,0,0,1,16.2.3L232.4,114.49A15.74,15.74,0,0,1,240,128Z'/></svg></button>"
                            "<button id='rdnext' title='" + html.escape(t("Next paragraph")) + "'><svg viewBox='0 0 256 256' fill='currentColor' aria-hidden='true'><path d='M208,40V216a8,8,0,0,1-16,0V146.77L72.43,221.55A15.95,15.95,0,0,1,48,208.12V47.88A15.95,15.95,0,0,1,72.43,34.45L192,109.23V40a8,8,0,0,1,16,0Z'/></svg></button>"
                            "<label class='rdlab'><span>" + html.escape(t("Speed")) + "</span><select id='rdrate' title='" + html.escape(t("Speed")) + "'>" + "".join(f"<option value='{r}'>{r}×</option>" for r in ("0.8", "0.9", "1", "1.1", "1.25", "1.5", "1.75", "2")) + "</select></label>"
                            "<label class='rdlab'><span>" + html.escape(t("Voice")) + "</span><select id='rdvoice' title='" + html.escape(t("Voice")) + "'></select></label>"
                            "<label class='rdlab'><span>" + html.escape(t("Text read aloud")) + "</span><select id='rdmode' title='" + html.escape(t("Text read aloud")) + "'>" + "".join(f"<option value='{m}'>{html.escape(lab)}</option>" for m, lab in (("raw", t("As written")), ("clean", t("Without citations")), ("simple", t("Simpler sentences")))) + "</select></label>"
                            "<label class='rdopt' title='" + html.escape(t("Volume")) + "'>🔊 <input id='rdvol' type='range' min='0.1' max='1' step='0.05' value='1'></label>"
                            "<label class='rdlab'><span>" + html.escape(t("Pause between paragraphs")) + "</span><select id='rdgap' title='" + html.escape(t("Pause between paragraphs")) + "'>" + "".join(f"<option value='{g}'>{lab}</option>" for g, lab in (("0", t("No pause")), ("0.5", "½ s"), ("1", "1 s"), ("2", "2 s"))) + "</select></label>"
                            "<span id='rdprog'></span><span id='rdhint'>" + html.escape(t("Click a paragraph to read from there.")) + "</span></div>")
        with gr.Column(visible=False, scale=5, elem_id="readpage") as read_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Reader") + "</div>")
                read_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn")
            with gr.Column() as rd_libcol:
                gr.Markdown(t("Joshua reads books, documents and web pages aloud, and remembers where you stopped."), elem_classes="rdintro")
                with gr.Row(elem_id="rdadd", equal_height=True):
                    rd_text_in = gr.Textbox(show_label=False, placeholder=t("Paste a web address, or some text"), lines=1, max_lines=8, scale=5, container=False, elem_id="rdpaste")
                    rd_text_add = gr.Button(t("Add"), size="sm", scale=0, min_width=70)
                    rd_upload = gr.UploadButton("📄 " + t("Add a file"), file_types=reader.TYPES, type="filepath", size="sm", scale=0, min_width=130)
                rd_list = gr.HTML("")
                rd_act = gr.Textbox("", elem_id="rdact", show_label=False, container=False)  # hidden: what you clicked in the list
            with gr.Column(visible=False) as rd_view:
                rd_lib = gr.Button("← " + t("Library"), size="sm", scale=0, min_width=140, elem_id="rdlib", elem_classes="mhidden")  # pressed by the ← Library in the reading bar, which stays in view
                rd_book = gr.HTML("")
        with gr.Column(visible=False, scale=5, elem_id="teampage") as team_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Teams") + "</div>")
                team_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn")
            with gr.Column() as tm_listcol:
                gr.Markdown(t("A team is a group of agents, each with a role, its own instructions, model and tools. In a chat with a team you write to its lead: the lead hands parts of the work to the others, checks what they return and writes the answer. A team can also have a single member: an agent of its own."))
                gr.HTML("<div class='whenbox'><b>" + t("When to use what") + "</b><ul><li><b>" + t("Desk") + ":</b> " + t("a ready-made setup for one kind of work, chosen under the message box.") + "</li><li><b>"
                        + t("Goal mode") + ":</b> " + t("one agent keeps working on your goal until it's done.") + "</li><li><b>" + t("Swarm") + ":</b> " + t("several agents answer the same question; one combines the best answer.") + "</li><li><b>"
                        + t("Team") + ":</b> " + t("agents with different roles share out the work, and the lead writes the answer.") + "</li></ul></div>")
                tm_new = gr.Button("＋ " + t("New team"), size="sm", scale=0, min_width=180, elem_id="tmnew")
                tm_list = gr.HTML("")
                with gr.Column(visible=False) as tm_controls:
                    tm_pick = gr.Dropdown([], label=t("Choose a team"))
                    with gr.Row():
                        tm_go = gr.Button("💬 " + t("Start a chat"), size="sm", scale=0, min_width=160)
                        tm_edit = gr.Button("✏ " + t("Change"), size="sm", scale=0, min_width=160)
                        tm_del = gr.Button(t("Delete team"), size="sm", scale=0, min_width=160, elem_id="tmdel", elem_classes=["danger"])
            with gr.Column(visible=False, elem_id="teamwiz") as tm_wiz:
                tw_state, tw_step, tw_sel = gr.State({"members": []}), gr.State(1), gr.State(-1)
                tw_head = gr.HTML("", elem_id="wsteps")
                with gr.Column() as tw_s1:
                    gr.Markdown(t("**What is the team for?** For example: a newsroom that turns notes into finished articles, or a research group that prepares briefings."))
                    with gr.Row(equal_height=False, elem_classes="namerow"):
                        tw_name = gr.Textbox(label=t("Name"), placeholder=t("e.g. Newsroom"), max_lines=1)
                        tw_icon = icon_pick(t("Icon"), scale=0, min_width=110)
                    tw_purpose = gr.Textbox(label=t("Purpose"), placeholder=t("e.g. Turns my notes and sources into checked, finished articles."), lines=2)
                with gr.Column(visible=False) as tw_s2:
                    tw_mlist = gr.Markdown("")
                    tw_mpick = gr.Radio([], show_label=False, elem_id="twmpick")
                    with gr.Row(equal_height=False, elem_classes="namerow"):
                        m_name = gr.Textbox(label=t("Name"), placeholder=t("Reporter"), max_lines=1, scale=2)
                        m_icon = icon_pick(t("Icon"), scale=0, min_width=110)
                        m_role = gr.Textbox(label=t("Role"), placeholder=t("finds facts and sources"), max_lines=1, scale=3)
                    m_instr = gr.Textbox(label=t("Instructions"), placeholder=t("How this member works: what it does, how, and what it hands back."), lines=4)
                    with gr.Row():
                        m_model = gr.Dropdown([], label=t("Model"), scale=2)
                        m_goal = gr.Checkbox(False, label="🎯 " + t("Goal mode"), info=t("Keeps working on its part until it's done and checked"), scale=3)
                    m_tools = gr.CheckboxGroup([(label, k) for k, (label, _) in TOOL_GROUPS.items()], label=t("Tools"))
                    with gr.Row():
                        m_save = gr.Button("＋ " + t("Save and add another member"), size="sm", scale=0, min_width=240)
                        m_remove = gr.Button(t("Remove member"), size="sm", scale=0, min_width=160, visible=False, elem_id="mremove", elem_classes=["danger"])
                with gr.Column(visible=False) as tw_s3:
                    gr.Markdown(t("**Who leads?** The lead is the one you write to: it hands out the work and writes the answer. Pick the member whose role sums up the job, such as an editor-in-chief."))
                    tw_lead = gr.Radio([], show_label=False)
                with gr.Column(visible=False) as tw_s4:
                    gr.Markdown(t("**Check the team**, then save it. You can change it later."))
                    tw_review = gr.HTML("")
                with gr.Row(elem_id="wiznav"):
                    tw_cancel = gr.Button(t("Cancel"), size="sm", scale=0, min_width=120, elem_id="twcancel")
                    tw_prev = gr.Button("← " + t("Back"), size="sm", scale=0, min_width=120, visible=False)
                    tw_next = gr.Button(t("Next") + " →", size="sm", scale=0, min_width=120, variant="primary")
                    tw_done = gr.Button("✓ " + t("Create team"), size="sm", scale=0, min_width=160, visible=False, variant="primary")
        with gr.Column(visible=False, scale=5, elem_id="binpage") as bin_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Bin") + "</div>")
                bin_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn")
            gr.Markdown(t("Chats you delete wait here for {n} days with their files, then they're deleted for good. Restore one to put it back in the list.", n=BIN_DAYS))
            bin_list = gr.HTML("")
            bin_act = gr.Textbox("", elem_classes="mhidden", show_label=False, container=False)  # which row button was pressed
            bin_go = gr.Button(elem_id="bingo", elem_classes="mhidden")
            with gr.Column(visible=False) as bin_controls:  # each row has its own Restore and Delete for good; here only what concerns them all
                bin_pick = gr.Dropdown([], label=t("Choose a chat"), visible=False)
                with gr.Row():
                    bin_back_btn = gr.Button("↩ " + t("Restore"), size="sm", scale=0, min_width=160, variant="primary", visible=False)
                    bin_del = gr.Button(t("Delete for good"), size="sm", scale=0, min_width=160, visible=False)
                    bin_all = gr.Button("🗑 " + t("Empty the bin"), size="sm", scale=0, min_width=180, elem_id="binempty", elem_classes=["danger"])
        with gr.Column(visible=False, scale=5, elem_id="mempage") as mem_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Memory") + "</div>")
                mem_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn")
            gr.Markdown(t("""After each reply, a model on this computer reads your message and keeps any lasting fact about you: who you are, your work, your projects, how you like answers. Most messages contain nothing to keep. Joshua uses these notes in every chat. You can read, correct or delete them here."""), elem_id="memintro")
            with gr.Row():
                mem = gr.Checkbox(True, label=t("Use memory"), info=t("Joshua uses these notes in every chat"))
                learn = gr.Checkbox(learn_setting(), label=t("Learn from chats"), info=t("Joshua adds new notes as you chat"))
                cloudmem = gr.Checkbox(share_memory_setting(), label=t("Share notes with cloud models"), info=cloudmem_info(share_memory_setting()))
            mem_topic = gr.Dropdown(choices=[(t("About you"), "profile")], value="profile", label=t("Topic"), elem_id="memtopic")
            gr.HTML("<div id='memlist' data-empty='" + html.escape(t("Nothing remembered yet. Joshua adds notes as you chat, or you can add one below."), quote=True) + "' data-edit='" + html.escape(t("Change"), quote=True)
                    + "' data-del='" + html.escape(t("Delete"), quote=True) + "' data-add='" + html.escape("＋ " + t("Add a note"), quote=True) + "' data-astext='" + html.escape(t("Edit as text"), quote=True)
                    + "' data-unsaved='" + html.escape(t("Changed: press Save notes to keep it."), quote=True) + "'></div>")  # the notes one per row, drawn by the page script from the text below
            mem_text = gr.Textbox(lines=14, max_lines=40, show_label=False, placeholder=t("Nothing remembered yet. Joshua adds notes as you chat, or you can type them here, one per line."), elem_id="memtext")
            mem_btn = gr.Button(t("Save notes"), size="sm", variant="primary", elem_id="memsave")
            gr.Markdown(t("**Learn from my past chats** reads each chat you've had so far, once, with a model on this computer when you have one, and adds what it learns about you to these notes. It can take a few minutes."), elem_classes=["updnote"], elem_id="pasthelp")
            with gr.Row(elem_id="pastrow"):
                past_btn = gr.Button("📒 " + t("Learn from my past chats"), size="sm", scale=0, min_width=260)
                past_status = gr.Markdown("", elem_id="paststatus")
        with gr.Column(visible=False, scale=5, elem_id="setpage") as set_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Settings") + "</div>")
                set_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn")
            gr.HTML("<nav class='pagenav'><span class='pnlab'>" + t("On this page:") + "</span>" + "".join(f"<a href='#' onclick=\"document.getElementById('{z}').scrollIntoView({{behavior: 'smooth'}}); return false\">{html.escape(lab)}</a>"
                                                       for z, lab in (("szlook", t("Appearance")), ("szvoice", t("Voice")), ("sztools", t("Tools")), ("szdesks", t("Desks")), ("szremote", t("Access from anywhere")), ("szcode", t("Running code")), ("szlog", t("Log")), ("szapi", "API"))) + "</nav>")
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szlook"):
                gr.HTML("<h2 class='msec'>" + t("Appearance") + "</h2>")
                screen = gr.Radio([(t("Paper"), "modern-paper"), (t("Paper night"), "modern-paper-night"), (t("Novel"), "novel"), (t("Black"), "black"), (t("Green"), "green"), (t("Amber"), "amber")], value="modern-paper", label=t("Theme"), elem_id="screen")
                textsize = gr.Radio([("90%", "0.9"), ("100%", "1"), ("110%", "1.1"), ("125%", "1.25"), ("140%", "1.4")], value="1", label=t("Text size"), elem_id="textsize")
                language = gr.Dropdown([(t("Language: automatic"), "auto")] + [(name, code) for code, name in LANGS.items()], value=i18n.chosen() if i18n.chosen() in LANGS else "auto", label=t("Language"), elem_id="language")
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szvoice"):
                gr.HTML("<h2 class='msec'>" + t("Voice") + "</h2><div class='msub'>" + t("The 🔊 button under an answer reads it aloud, with a voice of this computer. Choose a voice for each language: the choice is kept on this computer.") + "</div>"
                        + "<div id='voicepick'>" + "".join(f"<label class='vrow'><span>{n}</span><select data-lang='{k}'></select><button type='button' class='vtest' data-lang='{k}'>▶</button></label>" for k, n in {'it': 'Italiano', 'en': 'English', 'fr': 'Français', 'de': 'Deutsch', 'pl': 'Polski', 'zh': '中文'}.items()) + "</div>")
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="sztools"):
                gr.HTML("<h2 class='msec'>" + t("Tools") + "</h2><div class='msub'>" + t("What Joshua can do while it answers. Switch off the ones you don't want it to use. The list is part of Joshua: new tools come with updates.") + "</div>")
                tools_on = gr.CheckboxGroup([(label, key) for key, (label, _) in TOOL_GROUPS.items()], value=[k for k in TOOL_GROUPS if k not in tools_off()], show_label=False, elem_id="toolson")
                gr.Markdown(t("**Joshua's browser** is a Firefox window of its own, separate from yours. To let Joshua work on a site where you need an account, open the window, go to the site and sign in yourself: the sign-in is kept there, also after a restart. Joshua never types passwords. Remember that with a cloud model, the pages Joshua reads are sent to that model."), elem_id="browserhelp")
                with gr.Row():
                    br_open = gr.Button("🧭 " + t("Open Joshua's browser"), size="sm", scale=0, min_width=240)
                    br_status = gr.HTML("")
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szdesks"):
                gr.HTML("<h2 class='msec'>" + t("Desks") + "</h2><div class='msub'>" + t("A desk is a ready-made setup for one kind of work: you choose it under the message box. Pick a desk to change its name, icon, instructions, tools and the prompts it suggests on an empty chat, or make a new one. Your changes are kept on this computer and updates never overwrite them.") + "</div>")
                dk_pick = gr.Dropdown(choices=desk_editor_choices(), value=desk_editor_choices()[1][1] if len(desk_editor_choices()) > 1 else "", label=t("Desk"), elem_id="dkpick")
                with gr.Row():
                    dk_icon = icon_pick(t("Icon"), scale=0, min_width=110)
                    dk_name = gr.Textbox("", label=t("Name"), placeholder=t("Cooking"), max_lines=1)
                dk_instr = gr.Textbox("", lines=8, max_lines=12, label=t("Instructions"), placeholder=t("You are working at the cooking desk. Suggest recipes with what I have at home, give quantities for the number of people I say, and keep it simple."), elem_classes="deskedit")
                dk_tools = gr.CheckboxGroup([(label, key) for key, (label, _) in TOOL_GROUPS.items()], value=[], label=t("Tools at this desk"), elem_id="dktools")
                gr.HTML("<div class='fhead dkhead'>" + t("Suggested prompts on an empty chat") + "</div><div class='dkcols'><span>" + t("Icon") + "</span><span>" + t("Button text") + "</span><span>" + t("Text put in the message box") + "</span></div>")
                dk_rows, dk_cells = [], []
                for k in range(MAX_TASKS):  # a fixed number of lines: the ones in use are shown
                    with gr.Row(visible=False, elem_classes="dkrow", equal_height=True) as row_:
                        c_icon = icon_pick(show_label=False, container=False, scale=0, min_width=90)
                        c_title = gr.Textbox("", show_label=False, placeholder=t("Button text"), max_lines=1, scale=2, container=False)
                        c_prompt = gr.Textbox("", show_label=False, placeholder=t("Text put in the message box"), lines=1, max_lines=4, scale=5, container=False)
                        c_del = gr.Button("✕", size="sm", scale=0, min_width=40, elem_classes="dkdel")
                    dk_rows.append(row_)
                    dk_cells += [c_icon, c_title, c_prompt]
                    row_.dk_del = c_del
                dk_add = gr.Button("＋ " + t("Add a suggestion"), size="sm", scale=0, min_width=180, elem_id="dkadd")
                with gr.Row():
                    dk_save = gr.Button(t("Save desk"), size="sm", scale=0, min_width=160, elem_id="dksave")
                    dk_reset = gr.Button(t("Restore the default"), size="sm", scale=0, min_width=200)
                    dk_remove = gr.Button(t("Remove desk"), size="sm", scale=0, min_width=160, elem_id="dkremove", elem_classes=["danger"])
                    dk_note = gr.HTML("", elem_id="dknote")
                with gr.Row(visible=bool(desks.hidden())) as dk_hiddenrow:
                    dk_hidden = gr.Dropdown(choices=hidden_desk_choices(), value=None, label=t("Removed desks"), scale=3)
                    dk_back = gr.Button(t("Bring it back"), size="sm", scale=0, min_width=160)
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szremote"):
                gr.HTML("<h2 class='msec'>" + t("Access from anywhere") + "</h2><div class='msub'>" + t("Use Joshua from your other computers, phone or tablet, wherever you are, through Tailscale: a free app that links your own devices in a private network. Nothing is opened to the internet.") + "</div>")
                remote_on = gr.Checkbox(remote_setting(), label=t("Let my devices on Tailscale reach Joshua"))
                remote_box = gr.HTML("")
                remote_check = gr.Button(t("Check again"), size="sm", scale=0, min_width=160)
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szcode"):
                gr.HTML("<h2 class='msec'>" + t("Running code") + "</h2>")
                net = gr.Checkbox(True, label="🌐 " + t("Internet for code"), info=t("Lets code go online and install packages"))
                timeout = gr.Slider(10, 600, value=120, step=10, label=t("Code time limit (seconds)"))
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szlog"):
                gr.HTML("<h2 class='msec'>" + t("Log") + "</h2><div class='msub'>" + t("What Joshua did and what went wrong, with the time. If something breaks, copy it and send it to whoever helps you.") + "</div>")
                log_box = gr.Textbox("", show_label=False, lines=12, max_lines=12, interactive=False, autoscroll=True, buttons=["copy"], elem_id="logbox")
                with gr.Row():
                    log_copy = gr.Button(t("Copy the log"), size="sm", scale=0, min_width=160)
                    log_refresh = gr.Button(t("Show the latest"), size="sm", scale=0, min_width=160)
                    log_open = gr.Button(t("Open the folder"), size="sm", scale=0, min_width=160)
                gr.HTML("<div class='msub'>" + html.escape(str(jlog.PATH)) + "</div>")
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="szapi"):
                gr.HTML("<h2 class='msec'>🔌 API</h2>")
                gr.Markdown(API_HELP, elem_id="apihelp")
                api_info = gr.HTML("")
                with gr.Accordion(t("Try it in Terminal"), open=False):  # it contains the key
                    api_example = gr.Code(language="shell", show_label=False, interactive=False)
                api_lan_box = gr.Checkbox(api_settings()["lan"], label=t("Allow other devices on my network"), info=t("Off: only programs on this computer. Needs a restart of Joshua."))
                api_key_btn = gr.Button(t("Make a new key"), size="sm", elem_classes=["danger"])

        with gr.Column(visible=False, scale=5, elem_id="modelspage") as models_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Models") + "</div>")
                models_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn", elem_id="modelsback")
            gr.HTML("<nav class='pagenav'><span class='pnlab'>" + t("On this page:") + "</span>" + "".join(f"<a href='#' onclick=\"document.getElementById('{z}').scrollIntoView({{behavior: 'smooth'}}); return false\">{html.escape(lab)}</a>"
                                                       for z, lab in (("mzmenu", t("Your model menu")), ("mzlocal", t("On this computer")), ("mzcloud", t("In the cloud")), ("mzapi", t("With your API key")))) + "</nav>")
            gr.Markdown(t("Choose which models appear in the menu under the chat. Models on this computer are free and private; models in the cloud are bigger, and paid.")
                        + "\n\n<details><summary>" + t("How models are ranked") + "</summary>\n\n" + t("Best reasoning first, in each group. The score measures reasoning only: public tests of logic, coding, knowledge and use of tools, all in text ([Artificial Analysis](https://artificialanalysis.ai/leaderboards/models), November 2026); for small models it is Joshua's estimate. Reading images or audio, and deep thinking, give no extra points: they are shown on their own, under Abilities. For everyday questions, smaller models are often enough.") + "\n\n</details>", elem_id="mintro")
            with gr.Column(elem_classes=["mzone", "mz0"], elem_id="mzmenu"):
                gr.HTML("<div class='mzhead'><h2 class='msec'>" + t("Your model menu") + "</h2></div><div class='msub'>" + t("The models you choose from under the chat, in this order. The first one is the default for new chats.") + "</div>")
                menu_list = gr.HTML(menu_html())
            with gr.Column(elem_classes=["mzone", "mz1"], elem_id="mzlocal"):
                gr.HTML("<div class='mzhead'><h2 class='msec'>" + t("Open-weight models · on this computer") + "</h2><span class='mzscale'>" + t("ranked from best to worst") + "</span></div><div class='msub'>" + t("On this computer. Private and free: nothing leaves your computer. As fast and as capable as your hardware allows.") + "</div>")
                gr.HTML("<h3 class='msub2'>" + t("Installed") + "</h3>")
                inst_status = gr.HTML("")
                installed = gr.HTML("")
                gr.HTML("<h3 class='msub2'>" + t("Available for this computer") + "</h3>")
                local_status = gr.HTML("")
                local_list = gr.HTML("<div class='cloudstat'>" + t("Reading the list from ollama.com…") + "</div>")
                local_upd = gr.Button(t("Update my models"), size="sm")
            with gr.Column(elem_classes=["mzone", "mz2"], elem_id="mzcloud"):
                gr.HTML("<div class='mzhead'><h2 class='msec'>" + t("Open-weight models · in the cloud") + "</h2><span class='mzscale'>" + t("ranked from best to worst") + "</span></div><div class='msub'>" + t("Open-weight models from many makers, run on Ollama's servers: much more capable than what fits a computer, paid with Ollama credit.") + "</div>")
                gr.Markdown(CLOUD_HELP, elem_id="cloudhelp")  # first what goes to the cloud, then connecting, then the models
                cloud_status = gr.HTML("")  # connected or not: above the button, so you know whether you need it
                connect = gr.Button("① " + t("Connect this computer to Ollama"), size="sm")
                gr.HTML("<div class='msub'>② " + t("Then add the models you want to the menu:") + "</div>")
                cloud_list = gr.HTML("")
                gr.HTML("<div class='cloudlinks'><a href='https://ollama.com/dashboard' target='_blank' rel='noopener'>💳 " + t("Add credit on ollama.com") + "</a>"
                        "<a href='https://ollama.com/settings' target='_blank' rel='noopener'>" + t("See what you've used") + "</a></div>"
                        "<div class='cloudnote'>" + t("Credit is paid to Ollama for the computing power of its servers, which run the cloud models. Joshua itself is free: none of this money goes to Joshua, and local models never need credit.") + "</div>")
                credit = gr.HTML("")
                with gr.Row(equal_height=False, elem_id="okeyrow", visible=not providers.settings()["keys"].get("ollama")) as okeyrow:  # once the credit shows, the key isn't asked again
                    o_key = gr.Textbox(type="password", label=t("Ollama key: only needed to show your credit here"), placeholder=t("Paste the key here"), max_lines=1, scale=3)
                    o_save = gr.Button(t("Show credit"), size="sm", scale=0, min_width=150)
            with gr.Column(elem_classes=["mzone", "mz3"], elem_id="mzapi"):
                gr.HTML("<div class='mzhead'><h2 class='msec'>" + t("Models with your own API key") + "</h2></div><div class='msub'>" + t("Claude, ChatGPT, Gemini and DeepSeek, run by their companies on their servers and paid through your own API key.") + "</div>")
                closed_list = gr.HTML("")
                gr.Markdown(KEYS_HELP, elem_id="keyshelp")
                gr.HTML("<div class='fieldlabel'>1 · " + t("Choose the company") + "</div>")
                k_provider = gr.Radio([(v["name"], k) for k, v in providers.PROVIDERS.items()], value="anthropic", label=t("Company"), show_label=False, elem_id="kprov")
                k_status = gr.HTML("")  # "No key yet. Get a key from …": before the field where the key goes
                with gr.Row(equal_height=False, elem_id="kkeyrow"):
                    k_key = gr.Textbox(type="password", label="2 · " + t("Paste the key"), placeholder=t("Paste the key here"), max_lines=1, scale=4)
                    k_save = gr.Button(t("Save the key"), size="sm", scale=0, min_width=150)
                with gr.Row(equal_height=False, elem_id="kaddrow"):
                    k_pick = gr.Dropdown([], label="3 · " + t("Choose a model"), interactive=False, scale=4)
                    k_add = gr.Button(t("Add to the model menu"), size="sm", scale=0, min_width=200)
        with gr.Column(visible=False, scale=5, elem_id="schedpanel") as sched_panel:
            with gr.Row(elem_classes="pagehead"):
                gr.HTML("<div id='schedtitle'>" + t("Scheduled tasks") + "</div>")
                sched_back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn", elem_id="schedback")
            gr.Markdown(t("Joshua does a task on its own at the time you choose: a morning news summary, a weekly check of a website, a reminder written for you. Each run appears as a ⏰ chat in your list of chats.") + "\n\n" + t("**Only while Joshua is running.** Tasks can't run when the computer is off or asleep, or when Joshua is closed. A task missed in the meantime runs once, as soon as Joshua is open again."), elem_id="schedhelp")
            with gr.Row(equal_height=False, elem_id="schedrow"):
                with gr.Column(scale=2, elem_id="schedlist"):
                    gr.HTML("<div class='schedsub'>" + t("Your tasks") + "</div>")
                    task_list = gr.HTML("")
                    with gr.Column(visible=False) as t_controls:
                        t_pick = gr.Dropdown([], label=t("Choose a task to change, run or delete"))
                        with gr.Row():
                            t_run = gr.Button("▶ " + t("Run now"), size="sm")
                            t_pause = gr.Button(t("Pause"), size="sm")
                        t_del = gr.Button(t("Delete this task"), size="sm", elem_id="schedel", elem_classes=["danger"])
                with gr.Column(scale=3, elem_id="schedform"):
                    sched_head = gr.HTML("<div class='schedsub'>" + t("New task") + "</div>")
                    t_name = gr.Textbox(label=t("Name"), placeholder=t("e.g. Morning news"), max_lines=1)
                    t_prompt = gr.Textbox(label=t("What Joshua should do"), lines=5, placeholder=t("e.g. Find the three most important news stories in Piedmont today and summarise each in two lines, with sources."))
                    t_watch = gr.Checkbox(False, label="📡 " + t("Only tell me what's new since last time"), info=t("It remembers what it already told you, checks your sites and feeds and searches the web. When there's nothing new, it stays silent."))
                    t_sources = gr.Textbox(label=t("Sites and RSS feeds to check, one per line (optional)"), lines=3, placeholder="https://www.ansa.it/sito/ansait_rss.xml\nhttps://www.reuters.com/world/", visible=False)
                    t_when = gr.Radio([(t("Once"), "once"), (t("Every day"), "daily"), (t("On some days"), "weekly"), (t("Every few hours"), "hours")], value="daily", label=t("When"))
                    with gr.Row():
                        t_date = gr.Textbox(label=t("Date (YYYY-MM-DD)"), value=time.strftime("%Y-%m-%d"), max_lines=1, visible=False)
                        t_time = gr.Textbox(label=t("Time (HH:MM)"), value="07:30", placeholder="07:30", max_lines=1, elem_id="ttime")
                        t_every = gr.Slider(1, 24, value=6, step=1, label=t("Repeat every how many hours"), visible=False)
                    t_days = gr.CheckboxGroup([(t(d), d) for d in DAYS], value=["Mon", "Tue", "Wed", "Thu", "Fri"], label=t("Days"), visible=False)
                    t_model = gr.Dropdown(choices=models(), value=default_model(), label=t("Model"))
                    with gr.Column(elem_id="schedopts"):
                        gr.HTML("<div class='fieldlabel'>" + t("Options") + "</div>")
                        t_offline = gr.Checkbox(False, label="🚫 " + t("No internet for this task"))
                        t_notify = gr.Checkbox(True, label="🔔 " + t("Notify me when it's done"))
                    t_add = gr.Button(t("Save task"), variant="primary")
        with gr.Column(scale=4, elem_id="chatcol", elem_classes="chat-panel") as chatcol:
            with gr.Row(elem_id="chatbar"):  # this chat's settings that are on, and its ⋯ menu
                thinking = gr.Checkbox(True, label="💭 " + t("Deep thinking"), elem_classes="opt", elem_id="o_thinking", visible=False)  # always on: models that can think, do
                offline = gr.Checkbox(False, label="🚫 " + t("No internet"), elem_classes="opt", elem_id="o_offline", info=t("Joshua doesn't search or read the web, and code can't go online"))
                gr.HTML("<button type='button' class='fbtn fpill' title='" + html.escape(t("This chat's files and project folder"), quote=True) + "'>" + FOLDER_SVG + "<span>" + t("Files") + "</span></button>"
                        "<button class='cmore cur' title='" + html.escape(t("Options for this chat"), quote=True) + "' aria-label='" + html.escape(t("Options for this chat"), quote=True) + "'>⋯</button>", elem_id="chatbarbtns")
            with gr.Column(elem_id="goalpanel", elem_classes="modepanel", visible=False) as goalpanel:
                rounds = gr.Slider(2, 30, value=10, step=1, label="🎯 " + t("Max rounds"))
            with gr.Column(elem_id="swarmpanel", elem_classes="modepanel", visible=False) as swarmpanel:  # the swarm's agents: how many, and which model each one uses
                gr.HTML("<div class='fhead'>🐝 " + t("Swarm agents") + "</div>")
                agents = gr.Slider(2, TEAM, value=5, step=1, label=t("Number of agents"))
                with gr.Row(elem_id="teamrow"):
                    team = [gr.Dropdown(choices=[(t("Automatic"), "")] + models(), value="", label=t("Agent {n}", n=k + 1), elem_classes="teampick", min_width=150, visible=k < 5) for k in range(TEAM)]
            with gr.Column(elem_id="chatopts") as rightcol:  # this chat's files, opened from the ⋯ menu
                gr.HTML("<div class='fhead'>📂 " + t("Files and folder of this chat") + "<button type='button' class='fclose' title='" + html.escape(t("Close"), quote=True) + "' aria-label='" + html.escape(t("Close"), quote=True) + "'>✕</button></div>")
                files = gr.File(file_count="multiple", label=t("Files in this chat"), elem_id="wfiles", height=None)
                with gr.Row(equal_height=False, elem_id="projrow"):
                    project = gr.Textbox(label=t("Project folder (optional)"), placeholder=t("e.g. ~/Projects/my-project"), info=t("The chat works directly in this folder. Choose it, or type its path and press Enter."), max_lines=1, scale=4)
                    pickproj = gr.Button(t("Choose folder…"), size="sm", scale=0, min_width=150, elem_id="pickproj")
                with gr.Row(elem_id="filebtns"):
                    openf = gr.Button("📂 " + (t("Show in Finder") if sys.platform == "darwin" else t("Show the folder")), size="sm")
                    dlall = gr.DownloadButton("⬇ " + t("Download all (.zip)"), size="sm", visible=False)
            chatbot = gr.Chatbot(height="100%", show_label=False, elem_id="chat", placeholder="### " + t("Hello. What would you like to do today?") + "\n" + t("Everything you write here is stored only on this computer."))
            starters = gr.HTML(starters_html(""), elem_id="starters")  # the desk's starter cards, on the empty chat
            with gr.Row(equal_height=True, elem_id="promptrow"):
                box = gr.MultimodalTextbox(file_count="multiple", placeholder=t("Message Joshua, or drop files here…"), show_label=False, scale=12)
                stop = gr.Button("■ " + t("Stop"), scale=1, min_width=64, elem_id="stop")
            with gr.Row(elem_id="modelrow"):  # under the message box: the desk, goal and swarm switches, what's happening, the model and what it can do
                desk = gr.Radio(desk_choices(), value="", label=t("Desk"), show_label=False, container=False, elem_id="deskpick")
                goal = gr.Checkbox(False, label="🎯 " + t("Goal mode"), elem_classes=["opt", "modeopt"], elem_id="o_goal", info=t("Give a goal; it keeps working until done"))
                swarm = gr.Checkbox(False, label="🐝 " + t("Swarm"), elem_classes=["opt", "modeopt"], elem_id="o_swarm", info=t("Several agents answer in parallel, then one combines the best answer"))
                working = gr.HTML("", elem_id="working")
                with gr.Row(elem_id="modelpick"):  # the model and its abilities and price stay together, on one line
                    model = gr.Dropdown(choices=models(True), value=default_model(), label=t("Model"), show_label=False, container=False, elem_id="model", scale=0, min_width=200, filterable=False)  # no typing: on a phone, a tap would open the keyboard and zoom the page
                    cloudbadge = gr.HTML(badge(default_model()), elem_id="cloudbadge")

    box.submit(send, [box, cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem, desk, *team], [chatbot, box, cid, convs, files], concurrency_limit=None)
    desk.change(starters_html, desk, starters, show_progress="hidden")
    model.change(badge, model, cloudbadge, show_progress="hidden")

    # "＋ Add a model…" in the menu: open the two panels on the right, fill their lists, then put the menu back on the model you had
    last_model, adding = gr.State(default_model()), gr.State(False)

    # One place decides what is on screen: the chat, or exactly one page. Every menu entry, Back button and chat in the list goes through show_page.
    ALL_PAGES = [models_panel, sched_panel, tune_panel, mem_panel, set_panel, bin_panel, team_panel, read_panel, pod_panel]
    VIEW = [chatcol, rightcol] + ALL_PAGES

    def show_page(panel=None):
        """Visibility for VIEW: the chat when panel is None, otherwise that page alone."""
        return (gr.Column(visible=panel is None), gr.Column(visible=panel is None)) + tuple(gr.Column(visible=p is panel) for p in ALL_PAGES)

    PAGES = VIEW
    show_models = lambda: show_page(models_panel)

    SHOWN = [installed, closed_list, credit, cloud_status, cloud_list, local_status, local_list]  # what the page showed last time, kept in cache/models.json
    MCACHE = pathlib.Path(__file__).parent / "cache" / "models.json"

    def last_shown():
        try:
            old = json.loads(MCACHE.read_text())
            return tuple(old.get(str(i), gr.skip()) for i in range(len(SHOWN)))
        except Exception:
            return (gr.skip(),) * len(SHOWN)

    def keep(f, outs):
        """Run a step of the page and remember what it showed, for next time."""
        def run(*a):
            got = f(*a)
            vals = got if isinstance(got, tuple) else (got,)
            try:
                old = json.loads(MCACHE.read_text()) if MCACHE.exists() else {}
                for o, v in zip(outs, vals):
                    i = next((i for i, c in enumerate(SHOWN) if c is o), None)
                    if i is not None and isinstance(v, str):
                        old[str(i)] = v
                MCACHE.parent.mkdir(exist_ok=True)
                MCACHE.write_text(json.dumps(old, ensure_ascii=False))
            except Exception:
                pass
            return got
        return run

    def open_models(e, gate=None):
        """The page fills in step by step: what's here first, then what ollama.com and the companies offer.
        gate: a State that must be True for this to run (the model menu's '＋ Add a model…')."""
        def step(f, n):
            if gate is None:
                return f, []
            return (lambda a, *x: f(*x) if a else ((gr.skip(),) * n if n > 1 else gr.skip())), [gate]
        # first what the page showed last time, at once; then each part is brought up to date
        for f, ins, outs in [(last_shown, [], SHOWN), (menu_html, [], [menu_list]),
                             (keep(lambda: (installed_html(), closed_html(), ollama_credit()), [installed, closed_list, credit]), [], [installed, closed_list, credit]),
                             (keys_status, [k_provider], [k_pick, k_status]),
                             (keep(cloud_catalog, [cloud_status, cloud_list]), [], [cloud_status, cloud_list]),
                             (keep(local_catalog, [local_status, local_list]), [], [local_status, local_list])]:
            fn, gates = step(f, len(outs))
            e = e.then(fn, gates + ins, outs, show_progress="hidden")
        return e

    def pick_model(choice, last):
        if choice != ADD:
            return (gr.skip(),) * len(PAGES) + (choice, False)
        return show_models() + (last, True)

    picked = open_models(model.change(pick_model, [model, last_model], PAGES + [last_model, adding]), gate=adding)
    picked.then(lambda a, last: (gr.Dropdown(value=last), False) if a else (gr.skip(), False), [adding, last_model], [model, adding])  # the menu goes back to the model you had
    open_models(models_btn.click(show_models, outputs=PAGES))
    models_back.click(lambda: show_page() + (gr.Dropdown(choices=models(True)),), outputs=VIEW + [model])
    cloudmem.input(lambda on: (share_memory_setting(on), gr.Checkbox(info=cloudmem_info(on)))[1], cloudmem, cloudmem, show_progress="hidden")
    o_save.click(save_ollama_key, o_key, [o_key, credit, okeyrow])
    o_key.submit(save_ollama_key, o_key, [o_key, credit, okeyrow])
    sched.click(lambda: show_page(sched_panel) + (gr.Dropdown(choices=models()),), outputs=VIEW + [t_model]).then(tasks_view, outputs=[task_list, t_pick, t_controls]).then(sched_title, t_pick, sched_head, show_progress="hidden")
    sched_back.click(lambda: show_page() + (gr.Radio(choices=conv_choices()),), outputs=VIEW + [convs])
    t_watch.change(lambda w: gr.Textbox(visible=w), t_watch, t_sources)
    t_pick.input(task_fill, t_pick, [t_name, t_prompt, t_when, t_date, t_time, t_days, t_every, t_offline, t_notify, t_model, t_watch, t_sources]).then(pause_label, t_pick, t_pause).then(sched_title, t_pick, sched_head, show_progress="hidden")
    t_when.change(lambda w: (gr.Textbox(visible=w == "once"), gr.Textbox(visible=w != "hours"), gr.CheckboxGroup(visible=w == "weekly"), gr.Slider(visible=w == "hours")), t_when, [t_date, t_time, t_days, t_every])
    t_add.click(task_add, [t_name, t_prompt, t_when, t_date, t_time, t_days, t_every, t_offline, t_notify, t_model, t_watch, t_sources], [task_list, t_pick, t_controls, t_name, t_prompt, t_time]).then(sched_title, t_pick, sched_head, show_progress="hidden")
    t_run.click(lambda n: task_action(n, "run"), t_pick, [task_list, t_pick, t_controls])
    t_pause.click(lambda n: task_action(n, "pause"), t_pick, [task_list, t_pick, t_controls])
    t_del.click(lambda n: task_action(n, "delete"), t_pick, [task_list, t_pick, t_controls], js="(n) => (n && confirm(" + JS_TEXT(t("Delete '{name}'? It won't run again.")) + ".replace('{name}', n))) ? n : null").then(sched_title, t_pick, sched_head, show_progress="hidden")
    DK = [dk_icon, dk_name, dk_instr, dk_tools, dk_reset, dk_remove, dk_note] + dk_cells + dk_rows
    dk_add.click(desk_task_add, dk_cells, dk_cells + dk_rows, show_progress="hidden")
    for k, row_ in enumerate(dk_rows):
        row_.dk_del.click(functools.partial(desk_task_remove, k), dk_cells, dk_cells + dk_rows, show_progress="hidden")
    dk_pick.input(desk_editor_load, dk_pick, DK)
    set_btn.click(desk_editor_load, dk_pick, DK)
    dk_save.click(desk_editor_save, [dk_pick, dk_icon, dk_name, dk_instr, dk_tools] + dk_cells, [dk_pick, desk] + DK)
    dk_reset.click(desk_editor_reset, dk_pick, [dk_pick, desk] + DK)
    dk_remove.click(desk_editor_remove, dk_pick, [dk_pick, desk] + DK + [dk_hiddenrow, dk_hidden],
                    js="(d) => (d && confirm(" + JS_TEXT(t("Remove this desk? Its chats stay, at no desk. One of Joshua's desks can be brought back later.")) + ")) ? d : null")
    dk_back.click(desk_editor_back, dk_hidden, [dk_pick, desk] + DK + [dk_hiddenrow, dk_hidden])

    for btn, panel, back in [(mem_btn_open, mem_panel, mem_back), (set_btn, set_panel, set_back), (bin_btn, bin_panel, bin_back), (team_btn, team_panel, team_back), (read_btn, read_panel, read_back), (pod_btn, pod_panel, pod_back)]:
        btn.click(functools.partial(show_page, panel), outputs=VIEW, show_progress="hidden")
        back.click(show_page, outputs=VIEW, show_progress="hidden")
    pod_btn.click(pod_open, outputs=pod_outs, show_progress="hidden")
    mem_btn_open.click(mem_view, mem_topic, [mem_topic, mem_text])
    bin_btn.click(bin_view, outputs=[bin_list, bin_pick, bin_controls])
    OPENED = [chatbot, cid, files, project, model, thinking, goal, swarm, offline, agents, *team, desk, starters]  # what opening a chat fills
    TM_PAGE = [tm_list, tm_pick, tm_controls, tm_listcol, tm_wiz]
    WIZ = [tw_state, tw_step, tw_sel, tw_s1, tw_s2, tw_s3, tw_s4, tw_head, tw_prev, tw_next, tw_done, tw_name, tw_icon, tw_purpose,
           tw_mlist, tw_mpick, m_name, m_icon, m_role, m_instr, m_model, m_tools, m_goal, m_save, m_remove, tw_lead, tw_review]
    W_IN = [tw_state, tw_step, tw_sel, tw_name, tw_icon, tw_purpose, m_name, m_icon, m_role, m_instr, m_model, m_tools, m_goal, tw_lead]
    team_btn.click(team_page, outputs=TM_PAGE)
    gr.api(tts_voices, api_name="voices", queue=False)
    gr.api(client_log, api_name="client_log", queue=False)
    log_refresh.click(log_view, outputs=log_box)
    log_copy.click(None, js="() => { const a = document.querySelector('#logbox textarea'); if (a) navigator.clipboard.writeText(a.value); }")
    log_open.click(log_folder)
    set_btn.click(log_view, outputs=log_box)
    gr.api(tts, api_name="tts", concurrency_limit=2)
    gr.api(rd_text, api_name="rd_text", concurrency_limit=2)
    RD = [rd_list, rd_libcol, rd_view, rd_book]
    read_btn.click(rd_library, outputs=RD)
    rd_lib.click(rd_library, outputs=RD)
    rd_upload.upload(rd_add, rd_upload, RD)
    rd_act.change(rd_action, rd_act, RD, show_progress="hidden")
    rd_text_add.click(rd_paste, rd_text_in, RD + [rd_text_in])
    rd_pos.change(rd_save, rd_pos, None, show_progress="hidden", queue=False)
    tm_new.click(wiz_open, outputs=WIZ + [tm_listcol, tm_wiz])
    tm_edit.click(wiz_open, tm_pick, WIZ + [tm_listcol, tm_wiz])
    tm_del.click(team_delete, tm_pick, TM_PAGE, js="(c) => (c && confirm(" + JS_TEXT(t("Delete this team? Its chats stay in the list.")) + ")) ? c : null")
    tw_cancel.click(lambda go: team_page() if go else (gr.skip(),) * 5, tw_step, TM_PAGE,
                    js="() => { const typed = [...document.querySelectorAll('#teamwiz input[type=text], #teamwiz textarea')].some(x => x.value.trim()); return !typed || confirm(" + JS_TEXT(t("Discard this team? What you typed will be lost.")) + "); }")
    tw_next.click(functools.partial(wiz_move, 1), W_IN, WIZ)
    tw_prev.click(functools.partial(wiz_move, -1), W_IN, WIZ)
    tw_mpick.input(wiz_pick, W_IN + [tw_mpick], WIZ)
    m_save.click(wiz_member_save, W_IN, WIZ)
    m_remove.click(wiz_member_remove, W_IN, WIZ)
    tw_done.click(wiz_finish, W_IN, TM_PAGE)
    tm_go.click(team_open_chat, tm_pick, OPENED + [convs, team_panel, chatcol, rightcol])
    bin_back_btn.click(bin_restore, bin_pick, [bin_list, bin_pick, bin_controls, convs])
    bin_del.click(bin_forever, bin_pick, [bin_list, bin_pick, bin_controls], js="(c) => (c && confirm(" + JS_TEXT(t("Delete this chat and its files for good? This can't be undone.")) + ")) ? c : null")
    bin_go.click(None, None, bin_act, js="() => window.joshuaBin || ''").then(bin_row, bin_act, [bin_list, bin_pick, bin_controls, convs], show_progress="hidden")
    demo.load(None, None, None, js="""() => { document.addEventListener('click', (e) => {  // Restore / Delete for good in a row of the bin
        const b = e.target.closest('button.binact'); if (!b) return;
        if (b.dataset.act.startsWith('forever:') && !confirm(""" + JS_TEXT(t("Delete this chat and its files for good? This can't be undone.")) + """)) return;
        window.joshuaBin = b.dataset.act; b.disabled = true; const g = document.querySelector('#bingo'); (g.tagName === 'BUTTON' ? g : g.querySelector('button')).click(); }) }""")
    bin_all.click(bin_empty, bin_all, [bin_list, bin_pick, bin_controls], js="() => confirm(" + JS_TEXT(t("Delete every chat in the bin for good? This can't be undone.")) + ")")
    set_btn.click(api_panel, outputs=[api_info, api_example])
    learn.change(learn_setting, learn)
    remote_on.change(lambda on: (remote_setting(on), remote.switch(on, PORT), remote_html())[2], remote_on, remote_box)
    remote_check.click(lambda: (remote.switch(remote_setting(), PORT), remote_html())[1], outputs=remote_box)
    set_btn.click(remote_html, outputs=remote_box)
    br_open.click(lambda: "" if browser.show() == "ok" else "<span class='cloudstat'>" + t("The browser couldn't open.") + "</span>", outputs=br_status)
    tools_on.change(lambda on: tools_off([k for k in TOOL_GROUPS if k not in on]), tools_on)
    api_lan_box.input(api_lan, api_lan_box, [api_info, api_example])
    api_key_btn.click(lambda go: api_new_key() if go else (gr.skip(), gr.skip()), api_key_btn, [api_info, api_example],
                      js="() => confirm(" + JS_TEXT(t("Make a new key? Programs using the old one will stop working until you give them the new one.")) + ")")
    mact_go.click(row_action, mact, [model, inst_status, local_status, cloud_status], js="() => window.joshuaAct || ''").then(
        menu_html, outputs=menu_list, show_progress="hidden").then(
        lambda: (installed_html(), local_catalog()[1], cloud_catalog()[1]), outputs=[installed, local_list, cloud_list], show_progress="hidden")
    local_upd.click(local_update, outputs=local_status)
    demo.load(check_update, outputs=[updnote, updrow])
    # after an update Joshua restarts, which can take a minute on a slow computer: the page waits, then reloads by itself
    updbtn.click(lambda: gr.Button(visible=False), outputs=updbtn).then(do_update, outputs=updnote).then(None, js=WAIT_AND_RELOAD)
    connect.click(cloud_connect, outputs=cloud_status)
    k_provider.change(keys_status, k_provider, [k_pick, k_status])
    k_save.click(keys_save, [k_provider, k_key], [k_pick, k_status, k_key])
    k_add.click(keys_add, [k_provider, k_pick], [model, k_status]).then(lambda: (closed_html(), cloud_catalog()[1]), outputs=[closed_list, cloud_list], show_progress="hidden")
    mem_topic.input(mem_view, mem_topic, [mem_topic, mem_text])
    past_btn.click(learn_past, mem_topic, [past_status, mem_topic, mem_text])
    mem_btn.click(mem_save, [mem_topic, mem_text], [mem_topic, mem_text],
                  js="(topic, text) => (text.trim() || confirm(" + JS_TEXT(t("Clear all the notes in '{topic}'? This can't be undone.")) + ".replace('{topic}', topic))) ? [topic, text] : [topic, null]")
    swarm.change(lambda s: gr.Column(visible=s), swarm, swarmpanel)
    agents.change(lambda n: [gr.Dropdown(visible=k < n) for k in range(TEAM)], agents, team, show_progress="hidden")
    stop.click(stop_chat, cid, concurrency_limit=None)
    goal.change(lambda g: (gr.Column(visible=g), gr.MultimodalTextbox(placeholder=t("Describe a goal for Joshua to accomplish…") if g else t("Message Joshua, or drop files here…"))), goal, [goalpanel, box])
    new.click(new_chat, outputs=[chatbot, cid, convs, files, project, thinking, goal, swarm, offline, agents, *team, desk, starters])
    convs.input(open_conv, convs, OPENED).then(badge, model, cloudbadge)
    for e in (convs.input, new.click, tochat_go.click):  # a chat or New chat in the side column closes whatever page is open
        e(show_page, outputs=VIEW, show_progress="hidden")
    for o in (thinking, goal, swarm, offline, agents, *team):
        o.input(save_opts, [cid, thinking, goal, swarm, offline, agents, *team, desk], show_progress="hidden")

    def desk_set(c, *values):  # a chat moved to another desk: save it, and show the new icon in the list
        save_opts(c, *values)
        return gr.Radio(choices=conv_choices(), value=c) if c else gr.skip()
    desk.input(desk_set, [cid, thinking, goal, swarm, offline, agents, *team, desk], convs, show_progress="hidden")
    project.submit(set_project, [project, cid], [files, cid, convs, project])
    pickproj.click(choose_project, project, project).then(lambda p, c: set_project(p, c) if p else (gr.skip(),) * 4, [project, cid], [files, cid, convs, project])
    files.upload(upload_files, [files, cid], [files, cid, convs])
    files.change(zip_files, cid, dlall, show_progress="hidden")
    openf.click(open_folder, cid)
    chatbot.retry(retry, [cid, model, thinking, goal, rounds, net, timeout, swarm, agents, mem, offline, cloudmem, desk, *team], [chatbot, box, cid, convs, files])
    files.delete(delete_file, cid, files)
    live = gr.Timer(LIVE_EVERY, active=False)  # redraws the chat Joshua is writing in: switched on and off by poll()
    live.tick(live_view, cid, chatbot, show_progress="hidden", concurrency_limit=None)
    gr.Timer(1).tick(poll, cid, [live, convs, files, working], show_progress="hidden", concurrency_limit=None)
    gr.Timer(60).tick(badge, model, cloudbadge, show_progress="hidden")  # the Ollama credit left goes down as you use cloud models
    feedback.click(None, model, None, js="(m) => { window.open(" + json.dumps(FEEDBACK) + " + '&model=' + encodeURIComponent(m || ''), '_blank') }")
    renbtn.click(rename_open, cid, renbox)
    pin_go.click(toggle_pin, cid, convs, show_progress="hidden")
    ord_go.click(None, None, ord_val, js="() => window.joshuaOrder || ''").then(reorder_pins, [cid, ord_val], convs, show_progress="hidden")
    ren_go.click(None, None, ren_val, js="() => window.joshuaRename || ''").then(rename_save, [cid, ren_val], [renbox, convs], show_progress="hidden")
    renbox.submit(rename_save, [cid, renbox], [renbox, convs])
    dele.click(delete_chat, convs, [chatbot, cid, convs, files, project, thinking, goal, swarm, offline, agents, *team, desk, starters], js="(c) => (c && confirm(" + JS_TEXT(t("Move this chat to the bin? You can restore it there for 30 days.")) + ")) ? c : null")
    demo.load(lambda: (gr.Radio(choices=conv_choices()), gr.Dropdown(choices=models(True), value=default_model())), outputs=[convs, model], show_progress="hidden")
    newmodel.click(lambda: show_page(tune_panel) + ("new", "<div id='tunetitle'>" + t("Build a model from scratch") + "</div>"),
                   outputs=VIEW + [tune_mode, tune_title]).then(tune_refresh[0], outputs=tune_refresh[1], show_progress="hidden")
    tune.click(lambda: show_page(tune_panel) + ("<div class='ftres'>" + t("Checking what your computer can do…") + "</div>", "teach", "<div id='tunetitle'>" + t("Teach a model your style") + "</div>"),
               outputs=VIEW + [tune_res, tune_mode, tune_title]).then(tune_ui.res_html, outputs=tune_res, show_progress="hidden").then(tune_refresh[0], outputs=tune_refresh[1], show_progress="hidden")
    tune_back.click(lambda: show_page() + (gr.Dropdown(choices=models(True)),), outputs=VIEW + [model])
    reloading = gr.State(False)
    language.input(set_language, language, reloading).then(None, reloading, None, js="(r) => { if (r) setTimeout(() => location.reload(), 8000) }")
    # Gradio's own few labels follow the browser's language, not Joshua's: swap them for Joshua's translation as they appear
    gradio_texts = {"Drop File Here": t("Drop a file here"), "- or -": t("or"), "Click to Upload": t("click to choose one"),  # in sentence case, like the rest of Joshua
                    **{k: t(k) for k in ("Info", "Warning", "Error", "Clear")}}
    chat_files = {"Drop File Here": t("Add files to this chat"), "- or -": "", "Click to Upload": ""}  # the file area in the right column
    if True:
        demo.load(None, None, None, js="""() => { const T = """ + json.dumps(gradio_texts, ensure_ascii=False) + """, F = """ + json.dumps(chat_files, ensure_ascii=False) + """;
          const fix = (root) => { const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n;
            while ((n = w.nextNode())) { const k = n.nodeValue.trim(), M = n.parentElement && n.parentElement.closest('#wfiles') ? F : T;
              if (k in M && M[k] !== k) n.nodeValue = n.nodeValue.replace(k, M[k]); } };
          fix(document.body); new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(x => x.nodeType === 1 ? fix(x) : x.nodeType === 3 && fix(x.parentNode || document.body)))).observe(document.body, {childList: true, subtree: true}); }""")
    screen.change(None, screen, None, js="(v) => { document.body.dataset.screen = v; try { localStorage.setItem('joshua-screen3', v) } catch (e) {} }")
    demo.load(None, None, None, js="""() => { new MutationObserver(() => document.querySelectorAll('.dlp').forEach(d => {  // a download's progress, drawn in its row
        const slot = document.querySelector('.mslot[data-slot="' + d.dataset.slot + '"]'); if (!slot) return;
        slot.innerHTML = "<span class='mbar dl' title='" + d.dataset.label + "'><i style='width:" + d.dataset.pct + "%'></i></span> " + d.dataset.pct + '%'; d.remove(); }))
        .observe(document.body, {childList: true, subtree: true}) }""")
    demo.load(None, None, None, js="""() => { document.addEventListener('click', (e) => {  // Download / Add in the Models tables
        const b = e.target.closest('button.mact'); if (!b || !b.dataset.act) return;
        if (b.dataset.act.startsWith('ren:')) {  // a new name for a model made by you, typed in its row (the Mac app has no prompt())
            const old = b.dataset.act.slice(4), now = old.replace(/:latest$/, '').replace(/^joshua-/, '');
            const slot = b.closest('.mslot'), keep = slot.innerHTML;
            slot.innerHTML = "<input class='mren' spellcheck='false'><button class='mact'>OK</button>";
            const box = slot.querySelector('input'); box.value = now; box.title = """ + JS_TEXT(t("New name for {model}:")) + """.replace('{model}', now);
            box.focus(); box.select();
            const go = () => { const name = box.value.trim();
                if (!name || name === now) { slot.innerHTML = keep; return; }
                window.joshuaAct = 'ren:' + old + '|' + name; slot.innerHTML = '…';
                const g = document.querySelector('#mactgo'); (g.tagName === 'BUTTON' ? g : g.querySelector('button')).click(); };
            slot.querySelector('button').addEventListener('click', (ev) => { ev.stopPropagation(); go(); });
            box.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') go(); if (ev.key === 'Escape') slot.innerHTML = keep; });
            return; }
        if (/^rm(speech|image)?:/.test(b.dataset.act) && !confirm(""" + JS_TEXT(t("Delete {model} from this computer? You can download it again later.")) + """.replace('{model}', b.dataset.act.split(':').slice(1).join(':')))) return;
        window.joshuaAct = b.dataset.act; b.disabled = true; b.textContent = '…';
        const g = document.querySelector('#mactgo'); (g.tagName === 'BUTTON' ? g : g.querySelector('button')).click(); }) }""")
    demo.load(None, None, None, js="""() => {  // the ⋯ menu of each chat (pin, rename, delete, its settings, its files), the green pin, dragging pinned chats
        const PIN = '\\u2063', ICON = "<svg class='pinsvg' viewBox='0 0 256 256'><path d='M235.33,104l-53.47,53.65c4.56,12.67,6.45,33.89-13.19,60A15.93,15.93,0,0,1,157,224c-.38,0-.75,0-1.13,0a16,16,0,0,1-11.32-4.69L96.29,171,53.66,213.66a8,8,0,0,1-11.32-11.32L85,159.71l-48.3-48.3A16,16,0,0,1,38,87.63c25.42-20.51,49.75-16.48,60.4-13.14L152,20.7a16,16,0,0,1,22.63,0l60.69,60.68A16,16,0,0,1,235.33,104Z'/></svg>";  // Phosphor push-pin (fill)
        const L = {thinking: """ + JS_TEXT("💭 " + t("Deep thinking")) + """, goal: """ + JS_TEXT("🎯 " + t("Goal mode")) + """, swarm: """ + JS_TEXT("🐝 " + t("Swarm")) + """, offline: """ + JS_TEXT("🚫 " + t("No internet")) + """,
                   internet: """ + JS_TEXT("🌐 " + t("Use the internet")) + """, pin: """ + JS_TEXT(t("Pin to the top")) + """, unpin: """ + JS_TEXT(t("Unpin")) + """, rename: """ + JS_TEXT("✏ " + t("Rename")) + """, del: """ + JS_TEXT("🗑 " + t("Delete")) + """, files: """ + JS_TEXT("📂 " + t("Files and folder of this chat")) + """};
        const OPT = ['offline'];
        const press = (id) => { const g = document.querySelector('#' + id); if (g) (g.tagName === 'BUTTON' ? g : g.querySelector('button, input')).click(); };
        const close = () => { const m = document.querySelector('#cmenu'); if (m) m.remove(); };
        const text = (l) => { const s = l.querySelector('span:not(.cmore)'); return s ? s.textContent : ''; };
        const open = (anchor, saved, top) => {  // top: the ⋯ over the chat (this chat's settings); otherwise a chat's ⋯ in the list
          close(); const m = document.createElement('div'); m.id = 'cmenu';
          const on = (k) => { const i = document.querySelector('#o_' + k + ' input'); return i && i.checked; };
          const sel = document.querySelector('#convs label.selected'); const pinned = sel && text(sel).startsWith(PIN);
          let h = '';
          if (saved && !top) h += "<button data-k='pin'>" + ICON + ' ' + (pinned ? L.unpin : L.pin) + "</button><button data-k='rename'>" + L.rename + "</button><button data-k='del' class='danger'>" + L.del + '</button>';
          if (top) { h += "<button data-k='files'>" + L.files + '</button>';  // this chat's own settings live in one place: the ⋯ over the chat
            for (const k of OPT) h += "<button data-k='" + k + "' role='menuitemcheckbox' aria-checked='" + (k === 'offline' ? !on(k) : on(k)) + "'><span class='swi" + ((k === 'offline' ? !on(k) : on(k)) ? ' on' : '') + "'></span>" + (k === 'offline' ? L.internet : L[k]) + '</button>'; }
          m.innerHTML = h; document.body.appendChild(m);
          const r = anchor.getBoundingClientRect(); m.style.left = Math.max(8, Math.min(r.left, innerWidth - m.offsetWidth - 8)) + 'px';
          m.style.top = Math.min(r.bottom + 4, innerHeight - m.offsetHeight - 8) + 'px';
          m.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; const k = b.dataset.k; close();
            if (OPT.includes(k)) press('o_' + k); else if (k === 'pin') press('pingo'); else if (k === 'rename') rename(); else if (k === 'del') press('delchat');
            else if (k === 'files') document.body.dataset.files = document.body.dataset.files ? '' : 'open'; });
        };
        const rename = () => {  // the chat's name becomes a text field: Enter saves, Esc cancels
          const l = document.querySelector('#convs label.selected'); if (!l) return;
          // the field floats over the name, outside the list: inside it, every click and key would also go to the list
          const old = text(l).replace(PIN, '').replace(/^⏺ /, ''), r = l.getBoundingClientRect();
          const f = document.createElement('input'); f.className = 'renfield'; f.value = old;
          Object.assign(f.style, {left: r.left + 'px', top: r.top + 'px', width: (r.width - 6) + 'px', height: r.height + 'px'});
          document.body.appendChild(f); let done = false;
          setTimeout(() => { f.focus(); f.select(); }, 50);  // after the menu has closed, or it takes the focus back
          const end = (save) => { if (done) return; done = true; const v = f.value.trim(); f.remove(); if (save && v && v !== old) { window.joshuaRename = v; press('rengo'); } };
          f.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') { e.preventDefault(); end(true); } else if (e.key === 'Escape') { e.preventDefault(); end(false); } });
          f.addEventListener('blur', () => setTimeout(() => end(true), 150));
        };
        let drag = null;
        const mark = () => document.querySelectorAll('#convs label').forEach(l => {
          if (!l.querySelector('.cmore')) { const d = document.createElement('span'); d.className = 'cmore'; d.textContent = '⋯'; l.appendChild(d); }
          const p = text(l).startsWith(PIN); if (l.classList.contains('pinned') !== p) l.classList.toggle('pinned', p); if (l.draggable !== p) l.draggable = p;
          if (p && !l.querySelector('.pinsvg')) l.insertAdjacentHTML('afterbegin', ICON); if (!p && l.querySelector('.pinsvg')) l.querySelectorAll('.pinsvg').forEach(x => x.remove());
          if (!l.dataset.dnd) { l.dataset.dnd = 1;  // drag a pinned chat over another pinned chat to change their order
            l.addEventListener('dragstart', (e) => { drag = l; e.dataTransfer.effectAllowed = 'move'; });
            l.addEventListener('dragover', (e) => { if (drag && l.classList.contains('pinned')) { e.preventDefault(); l.classList.add('dropto'); } });
            l.addEventListener('dragleave', () => l.classList.remove('dropto'));
            l.addEventListener('drop', (e) => { e.preventDefault(); l.classList.remove('dropto'); if (!drag || drag === l) return;
              const pins = [...document.querySelectorAll('#convs label.pinned')]; const order = pins.map((x, i) => i);
              const from = pins.indexOf(drag), to = pins.indexOf(l); order.splice(to, 0, order.splice(from, 1)[0]);
              window.joshuaOrder = order.join(','); drag = null; press('ordgo'); }); }
        });
        new MutationObserver(mark).observe(document.body, {childList: true, subtree: true, characterData: true}); mark();
        document.addEventListener('click', (e) => {
          if (e.target.closest('.fbtn')) { document.body.dataset.files = document.body.dataset.files ? '' : 'open'; return; }  // the folder left of the message box
          const sc = e.target.closest('button.starter');  // a desk's starter card: its prompt goes in the message box, ready to finish
          if (sc) { const g = document.querySelector('#o_goal input'); if (sc.dataset.goal && g && !g.checked) g.click();  // an agent's card (the fact checker) also turns on goal mode
            const ta = document.querySelector('#promptrow textarea'); ta.value = sc.dataset.prompt;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            if (sc.dataset.send) { setTimeout(() => { const b = document.querySelector('#promptrow button.submit-button'); if (b) b.click(); }, 150); return; }  // WOPR: a game starts on click
            ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); return; }
          const d = e.target.closest('.cmore');
          if (!d) { if (!e.target.closest('#cmenu')) close(); return; }
          if (d.classList.contains('cur')) { e.preventDefault(); open(d, !!document.querySelector('#convs label.selected'), true); return; }
          const already = d.closest('label').classList.contains('selected');  // the click also opens the chat; its settings arrive a moment later
          setTimeout(() => open(d, true, false), already ? 0 : 700); }, true);
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); }); }""")
    demo.load(None, None, None, js="""() => {  // Prof Falken: you type in WOPR's own screen, at a > prompt under the latest one; Enter sends it.
        // WOPR also speaks (on unless you switch it off): only its words, never boards, mazes or lists, with the browser's own voice.
        const synth = window.speechSynthesis;
        const voiceOn = () => { try { return localStorage.getItem('joshua-wopr-voice') !== '0' } catch (e) { return true } };  // on unless you switched it off
        const pickVoice = () => { const vs = synth ? synth.getVoices() : []; for (const n of ['Fred', 'Zarvox', 'Ralph', 'Albert', 'Microsoft David', 'Google US English'])
          { const v = vs.find(x => x.name.startsWith(n)); if (v) return v; } return vs.find(x => x.lang && x.lang.startsWith('en')) || null; };
        const words = (txt) => txt.split(/\\n\\s*\\n/).map(par => par.split('\\n').map(l => l.trim()).filter(l => l && !l.startsWith('>')
            && !/[|]|\\+-|---|^[+\\s-]+$|^\\d+[.:]\\s|^[1-8]\\s{2}|^a b c|^STRATEGY|^GAME \\d|^YOU:|^DEALER:|^\\.+$|SCORE:|VOICE (ON|OFF)/.test(l)).join(' '))
          .filter(Boolean).join('. ').replace(/[@_]|\\(([A-Z])(\\/[A-Z])?\\)/g, '').replace(/\\.\\.+/g, '.').replace(/\\s+\\./g, '.').replace(/([.?!:])\\./g, '$1').toLowerCase();
        const say = (txt) => { if (!synth || !voiceOn()) return; const w = words(txt); if (!w) return; synth.cancel();
          const u = new SpeechSynthesisUtterance(w); const v = pickVoice(); if (v) u.voice = v; u.rate = 0.9; u.pitch = 0.7; u.lang = 'en-US'; synth.speak(u); };
        let armed = false, lastGreet = null, lastSaid = '';
        const attach = () => {
          const blocks = document.querySelectorAll('#chat code.language-wopr'), last = blocks[blocks.length - 1];
          const greet = document.querySelector('#starters .wopr'), host = last ? last.parentElement : (greet && greet.offsetParent ? greet : null);
          document.querySelectorAll('.wopr-in, .wopr-voice').forEach(i => { if (!host || !host.contains(i)) i.remove(); });
          if (!host) return;
          const screen = last ? last.innerText : greet.innerText;
          if (!last && greet !== lastGreet) { lastGreet = greet; say(screen); }  // the greeting, when the desk opens
          else if (last && armed && screen !== lastSaid) { armed = false; say(screen); }  // WOPR's answer to what you typed
          lastSaid = screen;
          if (!host.querySelector('.wopr-voice')) {
            const b = document.createElement('button'); b.className = 'wopr-voice'; b.type = 'button';
            const show = () => { b.textContent = voiceOn() ? 'VOICE ON' : 'VOICE OFF'; b.title = voiceOn() ? 'WOPR speaks: click to mute' : 'Click to let WOPR speak'; };
            show();
            b.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); try { localStorage.setItem('joshua-wopr-voice', voiceOn() ? '0' : '1') } catch (x) {}
              show(); if (voiceOn()) say(host.innerText); else if (synth) synth.cancel(); });
            host.appendChild(b); }
          if (host.querySelector('.wopr-in')) return;
          const line = document.createElement('div'); line.className = 'wopr-in';
          line.innerHTML = '<span>&gt;&nbsp;</span><input type="text" spellcheck="false" autocomplete="off" aria-label="WOPR">';
          host.appendChild(line);
          const inp = line.querySelector('input');
          inp.addEventListener('keydown', (e) => {
            if (synth) synth.cancel();  // typing stops WOPR talking
            if (e.key !== 'Enter' || !inp.value.trim()) return;
            e.preventDefault(); e.stopPropagation();
            const ta = document.querySelector('#promptrow textarea'); ta.value = inp.value; ta.dispatchEvent(new Event('input', {bubbles: true}));
            inp.value = ''; armed = true;
            setTimeout(() => { const b = document.querySelector('#promptrow button.submit-button'); if (b) b.click(); }, 120); });
          setTimeout(() => inp.focus(), 60); };
        document.addEventListener('click', (e) => { if (e.target.closest('.wopr .wline')) armed = true; }, true);
        new MutationObserver(() => { clearTimeout(window.joshuaWopr); window.joshuaWopr = setTimeout(attach, 200); }).observe(document.body, {childList: true, subtree: true}); }""")
    demo.load(None, None, None, js="""() => {  // what goes wrong in the page goes to Joshua's log; if Joshua stops answering, the page says so
        let sent = 0;
        const report = (what) => { if (sent++ > 30) return; try {
            fetch('/gradio_api/call/client_log', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({data: [String(what).slice(0, 2000)]})})
              .then(r => r.json()).then(j => fetch('/gradio_api/call/client_log/' + j.event_id)).catch(() => {}); } catch (e) {} };
        window.addEventListener('error', (e) => report('error: ' + e.message + ' (' + (e.filename || '').split('/').pop() + ':' + (e.lineno || '') + ')'));
        window.addEventListener('unhandledrejection', (e) => report('error: ' + ((e.reason && (e.reason.stack || e.reason.message)) || e.reason)));
        const box = document.createElement('div'); box.id = 'jdown';
        box.innerHTML = '<div><b class="t1">""" + JS_TEXT(t("Joshua isn't answering")).strip("'\"") + """</b><p class="t2">""" + JS_TEXT(t("It may be busy with a long job, like a transcription, or it may have stopped. Wait a little; if it doesn't come back, restart Joshua. What happened is written in its log:")).strip("'\"") + """ <code>""" + html.escape(str(jlog.PATH)).replace("\\\\", "\\\\\\\\") + """</code></p><button>""" + JS_TEXT(t("Reload the page")).strip("'\"") + """</button></div>';
        box.querySelector('button').onclick = () => location.reload(); document.body.appendChild(box);
        let misses = 0, down = false;
        setInterval(async () => { let ok = false;
          try { const r = await fetch('/', {method: 'HEAD', cache: 'no-store', signal: AbortSignal.timeout(8000)}); ok = r.ok; } catch (e) {}
          if (ok) { misses = 0; if (down) { down = false; report('Joshua answered the page again after a pause'); box.classList.add('back');
              box.querySelector('.t1').textContent = """ + JS_TEXT(t("Joshua is answering again")) + """; box.querySelector('.t2').textContent = """ + JS_TEXT(t("Reload the page to carry on.")) + """; } }
          else if (++misses >= 3 && !down) { down = true; box.classList.remove('back'); box.classList.add('on'); } }, 5000); }""")
    demo.load(None, None, None, js="""() => {  // Joshua's voice, for 🔊 and the Reader: on a Mac the Mac speaks (with its Premium voices), elsewhere the browser
        const V = window.joshuaVoice = {mac: null, token: 0, el: new Audio()};
        const SILENT = (() => { const n = 2000, b = new Uint8Array(44 + n), d = new DataView(b.buffer), w = (o, t) => [...t].forEach((c, i) => b[o + i] = c.charCodeAt(0));
          w(0, 'RIFF'); d.setUint32(4, 36 + n, true); w(8, 'WAVEfmt '); d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true);
          d.setUint32(24, 8000, true); d.setUint32(28, 8000, true); d.setUint16(32, 1, true); d.setUint16(34, 8, true); w(36, 'data'); d.setUint32(40, n, true); b.fill(128, 44);
          return URL.createObjectURL(new Blob([b], {type: 'audio/wav'})); })();  // a quarter of a second of silence: Safari counts it as sound started by your click
        const call = async (name, data) => { try {  // Gradio's own way to call a Python function from the page: ask, then collect the answer
            const r = await fetch('/gradio_api/call/' + name, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({data})});
            const id = (await r.json()).event_id, txt = await (await fetch('/gradio_api/call/' + name + '/' + id)).text();
            const m = txt.match(/event: complete\\s*\\ndata: (.*)/); return m ? JSON.parse(m[1])[0] : null; } catch (e) { return null; } };
        V.call = call;
        V.ready = call('voices', []).then(v => { V.mac = v && v.length ? v : null; V.isReady = true; });
        const saved = (k) => { try { return localStorage.getItem('joshua-voice-' + k) } catch (e) { return null } };
        V.macVoice = (k, name) => { const mine = (V.mac || []).filter(v => v.lang.toLowerCase().startsWith(k));
          const v = (name && mine.find(v => v.name === name)) || mine.find(v => v.name === saved(k)) || mine.find(v => /premium/i.test(v.name)) || mine.find(v => /enhanced/i.test(v.name))
            || mine.find(v => /^(Samantha|Alice|Thomas|Anna|Zosia|Tingting)/.test(v.name)) || mine[0]; return v ? v.name : ''; };
        const made = new Map(), done = new Map();  // paragraphs on their way to audio, and those ready
        V.url = (text, k, name) => { const who = V.macVoice(k, name), key = who + '|' + text; if (!made.has(key)) { made.set(key, call('tts', [text, who]).then(u => { if (u) done.set(key, u); if (done.size > 60) done.delete(done.keys().next().value); return u; }));
            if (made.size > 60) made.delete(made.keys().next().value); } return made.get(key); };
        V.prefetch = (text, k, name) => { if (V.mac && text) V.url(text, k, name); };
        V.list = (k) => V.mac ? V.mac.filter(v => v.lang.toLowerCase().startsWith(k)).map(v => v.name)
          : (window.speechSynthesis ? speechSynthesis.getVoices().filter(v => v.lang && v.lang.toLowerCase().startsWith(k)).map(v => v.name) : []);
        V.current = (k, name) => V.mac ? V.macVoice(k, name) : ((name && window.speechSynthesis && speechSynthesis.getVoices().find(v => v.name === name)) || (window.joshuaVoiceFor ? window.joshuaVoiceFor(k) : null) || {}).name || '';
        V.unlock = () => { const a = V.el; if (a.paused) { a.src = SILENT; a.play().catch(() => {}); } };  // a sound started by your click lets the next ones play (Safari, phones)
        V.stop = () => { V.token++; V.el.onended = V.el.onerror = null; V.el.pause(); if (window.speechSynthesis) speechSynthesis.cancel(); };
        const pieces = (t) => { const out = []; (t.match(/[^.!?。！？;:]+[.!?。！？;:]*["'»”)\\]]*\\s*/g) || [t]).forEach(s => {
            while (s.length > 220) { let k = s.lastIndexOf(', ', 220); if (k < 80) k = s.lastIndexOf(' ', 220); if (k < 40) k = 220; out.push(s.slice(0, k + 1)); s = s.slice(k + 1); }
            if (s.trim()) out.push(s); }); return out; };  // browsers cut off long texts: short pieces
        V.say = (text, k, rate, onend, opts) => {  // says one piece of text, then calls onend (unless stopped meanwhile); opts: voice, volume
          opts = opts || {};
          const tok = ++V.token; let fired = false; const finish = () => { if (!fired && tok === V.token) { fired = true; if (onend) onend(); } };
          if (!V.isReady) { V.ready.then(() => { if (tok === V.token) V.say(text, k, rate, onend, opts); }); return; }
          if (V.mac) {
            const go = (url) => { if (tok !== V.token) return; if (!url) return finish();
              const a = V.el; a.onended = a.onerror = null; a.src = url; a.playbackRate = rate || 1; a.preservesPitch = true; a.volume = opts.volume == null ? 1 : opts.volume;
              a.onended = finish; a.onerror = () => { const b = opts.onblocked || V.onblocked; if (tok === V.token && b) b('error'); };
              a.play().catch(e => { if (tok !== V.token) return; const b = opts.onblocked || V.onblocked; if (b) b(e && e.name); else finish(); }); };
            const key = V.macVoice(k, opts.voice) + '|' + text;
            if (done.has(key)) return go(done.get(key));  // ready: it starts at once, still inside your click
            V.url(text, k, opts.voice).then(go); return; }
          const synth = window.speechSynthesis; if (!synth) return finish(); synth.cancel();
          const v = (opts.voice && synth.getVoices().find(x => x.name === opts.voice)) || (window.joshuaVoiceFor ? window.joshuaVoiceFor(k) : null), parts = pieces(text);
          parts.forEach((p, i) => { const u = new SpeechSynthesisUtterance(p.trim()); if (v) { u.voice = v; u.lang = v.lang; } else u.lang = k; u.rate = rate || 1;
            u.volume = opts.volume == null ? 1 : opts.volume;
            if (i === parts.length - 1) u.onend = finish; synth.speak(u); }); }; }""")
    demo.load(None, None, None, js="""() => {  // 🔊 under each answer: reads the answer aloud with a voice of this computer, in the answer's language
        const synth = window.speechSynthesis; if (!synth) return;
        const ICON = '<svg viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M154.64,26.61a6,6,0,0,0-6.32.65L77.94,82H32A14,14,0,0,0,18,96v64a14,14,0,0,0,14,14H77.94l70.38,54.74A6,6,0,0,0,158,224V32A6,6,0,0,0,154.64,26.61ZM30,160V96a2,2,0,0,1,2-2H74v68H32A2,2,0,0,1,30,160Zm116,51.73L86,165.07V90.93l60-46.66Zm50.53-108.85a38,38,0,0,1,0,50.24,6,6,0,1,1-9-7.94,26,26,0,0,0,0-34.37,6,6,0,0,1,9-7.93ZM246,128a77.86,77.86,0,0,1-19.86,52,6,6,0,1,1-8.94-8,66,66,0,0,0,0-88,6,6,0,1,1,8.94-8A77.86,77.86,0,0,1,246,128Z"/></svg>', TITLE = """ + JS_TEXT(t("Read aloud")) + """, SAMPLE = {it: 'Ciao, sono Joshua.', en: 'Hello, I am Joshua.', fr: 'Bonjour, je suis Joshua.', de: 'Hallo, ich bin Joshua.', pl: 'Cześć, jestem Joshua.', zh: '你好，我是Joshua。'};
        const PREFER = {it: ['Alice', 'Federica', 'Luca'], en: ['Samantha', 'Daniel', 'Microsoft Aria', 'Microsoft Jenny', 'Google US English'], fr: ['Thomas', 'Amélie', 'Audrey'], de: ['Anna', 'Petra', 'Markus'], pl: ['Zosia', 'Ewa', 'Krzysztof'], zh: ['Tingting', 'Ting-Ting', 'Lili']};
        const WORDS = {it: ' il la che di per non una sono è con anche gli delle del alla ', en: ' the and of to is that it you with for are this be on ', fr: ' le les et est des que une pour pas dans sur avec ', de: ' der die und das ist nicht ein zu mit den von sie ', pl: ' i w nie się na jest że to z do jak co ', };
        const lang = (txt) => { if ((txt.match(/[\\u4e00-\\u9fff]/g) || []).length > 20) return 'zh';
          const w = (' ' + txt.toLowerCase().replace(/[^\\p{L}\\s]/gu, ' ') + ' ').split(/\\s+/).slice(0, 400); let best = 'en', top = 0;
          for (const [k, list] of Object.entries(WORDS)) { const n = w.filter(x => x && list.includes(' ' + x + ' ')).length; if (n > top) { best = k; top = n; } } return best; };
        const voiceFor = (k) => { const vs = synth.getVoices(); let name = null; try { name = localStorage.getItem('joshua-voice-' + k) } catch (e) {}
          const mine = vs.filter(v => v.lang && v.lang.toLowerCase().startsWith(k));  // a Premium or Enhanced voice (a free download in the Mac's settings) sounds far more natural
          const best = mine.find(v => /premium/i.test(v.name)) || mine.find(v => /enhanced|neural|natural/i.test(v.name));
          return vs.find(v => v.name === name) || best || (PREFER[k] || []).map(p => vs.find(v => v.name.startsWith(p))).find(Boolean) || mine[0] || null; };
        const clean = (row) => { const c = row.cloneNode(true);
          const leaves = [...c.querySelectorAll('.message')].filter(m => !m.querySelector('.message') && !m.querySelector('.thought-group'));  // the answer, not the thinking, sources, steps
          leaves.forEach(m => m.querySelectorAll('pre, table, button, svg, .ji').forEach(x => x.remove()));  // nor code, tables, icons
          document.body.appendChild(c); c.style.cssText = 'position:fixed;left:-9999px;top:0';  // innerText needs the copy on the page
          const out = leaves.map(m => m.innerText); c.remove();
          return out.join('\\n').replace(/https?:\\/\\/\\S+/g, '').replace(/[*#_`>|]/g, ' ').replace(/\\[(\\d+)\\]/g, '').replace(/\\s+\\n/g, '\\n').trim(); };
        window.joshuaVoiceFor = (k) => voiceFor(k);
        let current = null;
        const stop = () => { window.joshuaVoice.stop(); if (current) current.classList.remove('speaking'); current = null; };
        const blocks = (t) => { const out = []; t.split(/\\n+/).forEach(p => { p = p.trim(); if (!p) return;  // paragraphs, joined up to ~600 characters
            if (out.length && out[out.length - 1].length + p.length < 600) out[out.length - 1] += '\\n' + p; else out.push(p); }); return out.length ? out : [t]; };
        const speak = (text, k, btn) => { if (window.jrPause) window.jrPause(); stop(); current = btn; if (btn) btn.classList.add('speaking');
          const V = window.joshuaVoice, parts = blocks(text); let i = 0; V.unlock();
          const next = () => { if (current !== btn) return; if (i >= parts.length) return stop(); if (parts[i + 1]) V.prefetch(parts[i + 1], k); V.say(parts[i++], k, 1, next, {onblocked: () => { if (current === btn) stop(); }}); };
          next(); };
        const add = () => document.querySelectorAll('#chat .message-buttons').forEach(bar => {
          if (bar.querySelector('.speak')) return;
          let row = bar.previousElementSibling; while (row && !row.querySelector('.message')) row = row.previousElementSibling;
          if (!row || !row.querySelector('.bot') || row.querySelector('code.language-wopr')) return;
          const b = document.createElement('button'); b.className = 'speak icon-button'; b.type = 'button'; b.title = TITLE; b.setAttribute('aria-label', TITLE); b.innerHTML = ICON;
          b.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); if (current === b) return stop(); const txt = clean(row); if (txt) speak(txt, lang(txt), b); });
          const copy = bar.querySelector("button[aria-label='Copy message']"); (copy ? copy.parentElement : bar).appendChild(b); });
        const fill = () => document.querySelectorAll('#voicepick select').forEach(sel => { const k = sel.dataset.lang, V = window.joshuaVoice;
          const names = V.mac ? V.mac.filter(v => v.lang.toLowerCase().startsWith(k)).map(v => v.name) : synth.getVoices().filter(v => v.lang && v.lang.toLowerCase().startsWith(k)).map(v => v.name);
          if (sel.dataset.n === String(names.length) + !!V.mac) return; sel.dataset.n = String(names.length) + !!V.mac;
          const cur = V.mac ? V.macVoice(k) : (voiceFor(k) || {}).name;
          sel.innerHTML = '<option value="">—</option>' + names.map(n => '<option' + (n === cur ? ' selected' : '') + '>' + n + '</option>').join(''); });
        window.joshuaVoice.ready.then(fill);
        document.addEventListener('change', (e) => { const sel = e.target.closest('#voicepick select'); if (!sel) return; try { localStorage.setItem('joshua-voice-' + sel.dataset.lang, sel.value) } catch (x) {} });
        document.addEventListener('click', (e) => { const t = e.target.closest('#voicepick .vtest'); if (!t) return; e.preventDefault(); speak(SAMPLE[t.dataset.lang], t.dataset.lang, null); }, true);
        synth.onvoiceschanged = fill;
        new MutationObserver(() => { clearTimeout(window.joshuaSpeak); window.joshuaSpeak = setTimeout(() => { add(); fill(); }, 250); }).observe(document.body, {childList: true, subtree: true}); }""")
    demo.load(None, None, None, js="""() => {  // the Reader: reads the open book paragraph by paragraph, and tells Joshua each paragraph it finishes
        const V = window.joshuaVoice;
        const PAUSE = '<svg viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M216,48V208a16,16,0,0,1-16,16H160a16,16,0,0,1-16-16V48a16,16,0,0,1,16-16h40A16,16,0,0,1,216,48ZM96,32H56A16,16,0,0,0,40,48V208a16,16,0,0,0,16,16H96a16,16,0,0,0,16-16V48A16,16,0,0,0,96,32Z"/></svg>';
        const R = window.jr = {id: null, i: 0, texts: [], lang: '', playing: false, token: 0, rate: 1, voice: '', volume: 1, gap: 0, mode: 'clean'};
        try { R.rate = +(localStorage.getItem('joshua-reader-rate') || 1) } catch (e) {}
        let PLAY = null;
        const book = () => document.querySelector('#rdbook');
        const para = (i) => { const b = book(); return b && b.dataset.id === R.id ? b.querySelector('p[data-i="' + i + '"]') : null; };
        const save = () => { const box = document.querySelector('#rdpos textarea, #rdpos input'); if (!box || !R.id) return;
          box.value = R.id + '|' + R.i + '|' + JSON.stringify({voice: R.voice, rate: R.rate, volume: R.volume, gap: R.gap, mode: R.mode}); box.dispatchEvent(new Event('input', {bubbles: true})); };
        const controls = () => {  // the book's own voice, speed, volume and pause, in the bar
          const vs = document.querySelector('#rdvoice'); if (vs) { const names = V.list(R.lang || 'en'), cur = V.current(R.lang || 'en', R.voice), sig = names.join('|') + '>' + cur;
            if (vs.dataset.sig !== sig) { vs.dataset.sig = sig; vs.innerHTML = names.map(n => '<option' + (n === cur ? ' selected' : '') + '>' + n + '</option>').join('') || '<option>—</option>'; } }
          const vol = document.querySelector('#rdvol'); if (vol && document.activeElement !== vol) vol.value = R.volume;
          const gp = document.querySelector('#rdgap'); if (gp) gp.value = String(R.gap);
          const md = document.querySelector('#rdmode'); if (md && md.value !== R.mode) md.value = R.mode; };
        const spoken = new Map(), ready = new Map();  // the text heard for each paragraph: without citations, or rewritten in simpler sentences by Joshua
        const text = (i) => { if (R.mode === 'raw' || i >= R.texts.length) return Promise.resolve(R.texts[i]);
          const key = R.id + '|' + R.mode + '|' + i; if (!spoken.has(key)) spoken.set(key, V.call('rd_text', [R.id, i, R.mode]).then(t => { t = t || R.texts[i]; ready.set(key, t); return t; }));
          return spoken.get(key); };
        const show = (scroll) => { const b = book(); if (b && b.dataset.id === R.id) {
            b.querySelectorAll('p.rdnow').forEach(p => p.classList.remove('rdnow')); const p = para(R.i);
            if (p) { p.classList.add('rdnow'); if (scroll) p.scrollIntoView({block: 'center', behavior: 'smooth'}); } }
          const n = R.texts.length, pr = document.querySelector('#rdprog'); if (pr && n) pr.textContent = """ + JS_TEXT(t("Paragraph {i} of {n} · {p}% read")) + """.replace('{i}', R.i + 1).replace('{n}', n).replace('{p}', Math.round(100 * R.i / Math.max(n - 1, 1)));
          const nm = document.querySelector('#rdname'), tt = document.querySelector('#rdbook .rdtitle'); if (nm && tt && nm.textContent !== tt.textContent) nm.textContent = tt.textContent;
          const pb = document.querySelector('#rdplay'); if (pb) { if (!PLAY) PLAY = pb.innerHTML; pb.innerHTML = R.playing ? PAUSE : PLAY; }
          const sel = document.querySelector('#rdrate'); if (sel && +sel.value !== R.rate) sel.value = String(R.rate); };
        const say = () => { if (!R.playing) return; if (R.i >= R.texts.length) { R.playing = false; R.i = R.texts.length - 1; show(false); return; }
          show(true); const tok = ++R.token, k = R.lang || 'en', i = R.i;
          [1, 2].forEach(d => { if (i + d < R.texts.length) text(i + d).then(t => { if (d === 1) V.prefetch(t, k, R.voice); }); });  // the next paragraphs get ready while this one is read
          const now = R.mode === 'raw' ? R.texts[i] : ready.get(R.id + '|' + R.mode + '|' + i);  // ready: no waiting, so the sound still counts as started by your click
          const speak = (t) => { if (tok !== R.token || !R.playing) return; V.say(t, k, R.rate, () => { if (tok !== R.token || !R.playing) return; R.i++; save(); setTimeout(() => { if (tok === R.token) say(); }, R.gap * 1000); },
                {voice: R.voice, volume: R.volume}); };
          if (now != null) speak(now); else text(i).then(speak); };
        V.onblocked = (why) => {  // the browser didn't let the sound start (Safari wants it to come from a click): pause, and ask for ▶
          if (!R.playing) return; window.jrPause(); const h = document.querySelector('#rdhint');
          if (h) { h.textContent = why === 'error' ? """ + JS_TEXT(t("This paragraph couldn't be played. Press ▶ to try again.")) + """ : """ + JS_TEXT(t("Press ▶ to start reading.")) + """; h.classList.add('rdwarn'); } };
        const play = () => { if (!R.texts.length) return; const h = document.querySelector('#rdhint.rdwarn'); if (h) { h.classList.remove('rdwarn'); h.textContent = """ + JS_TEXT(t("Click a paragraph to read from there.")) + """; }
          R.playing = true; R.token++; V.stop(); V.unlock(); say(); };
        window.jrPause = () => { if (!R.playing) return; R.playing = false; R.token++; V.stop(); save(); show(false); };
        const jump = (i) => { R.i = Math.max(0, Math.min(i, R.texts.length - 1)); R.token++; V.stop(); save(); if (R.playing) { V.unlock(); say(); } else show(true); };
        const act = (v) => { const box = document.querySelector('#rdact textarea, #rdact input'); if (!box) return;
          box.value = v; box.dispatchEvent(new Event('input', {bubbles: true})); setTimeout(() => { box.value = ''; box.dispatchEvent(new Event('input', {bubbles: true})); }, 400); };
        document.addEventListener('click', (e) => {
          const x = e.target.closest('.rdx[data-remove]'); if (x) { e.preventDefault(); e.stopPropagation(); if (confirm(""" + JS_TEXT(t("Remove this book from the library? Your file isn't touched.")) + """)) act('remove:' + x.dataset.remove); return; }
          const row = e.target.closest('.rdrow[data-open]'); if (row) { e.preventDefault(); return act('open:' + row.dataset.open); }
          if (e.target.closest('#rdplay')) { e.preventDefault(); return R.playing ? window.jrPause() : play(); }
          if (e.target.closest('#rdprev')) { e.preventDefault(); return jump(R.i - 1); }
          if (e.target.closest('#rdnext')) { e.preventDefault(); return jump(R.i + 1); }
          const p = e.target.closest('#rdbook p[data-i]'); if (p && !getSelection().toString()) { R.i = +p.dataset.i; R.playing = true; R.token++; V.stop(); V.unlock(); save(); say(); } }, true);
        const restart = () => { if (R.playing) { R.token++; V.stop(); V.unlock(); say(); } };
        document.addEventListener('change', (e) => {
          if (e.target.closest('#rdvoice')) { R.voice = e.target.value; save(); return restart(); }
          if (e.target.closest('#rdgap')) { R.gap = +e.target.value; return save(); }
          if (e.target.closest('#rdmode')) { R.mode = e.target.value; save(); return restart(); }
          if (e.target.closest('#rdvol')) return save(); });
        document.addEventListener('input', (e) => { if (!e.target.closest('#rdvol')) return; R.volume = +e.target.value; if (V.mac) V.el.volume = R.volume; });
        document.addEventListener('change', (e) => { if (!e.target.closest('#rdrate')) return; R.rate = +e.target.value; save();
          try { localStorage.setItem('joshua-reader-rate', String(R.rate)) } catch (x) {}
          if (R.playing && V.mac) V.el.playbackRate = R.rate; else if (R.playing) { R.token++; V.stop(); say(); } });
        setInterval(() => { const b = book(); if (!b) return;
          if (b.dataset.id !== R.id) {  // another book opened: start where you left it
            if (R.playing) window.jrPause();
            R.id = b.dataset.id; R.lang = b.dataset.lang || ''; R.i = +b.dataset.pos || 0; R.texts = [...b.querySelectorAll('p[data-i]')].map(p => p.textContent);
            let o = {}; try { o = JSON.parse(b.dataset.settings || '{}') } catch (x) {}
            R.voice = o.voice || ''; R.volume = o.volume == null ? 1 : +o.volume; R.gap = +o.gap || 0; if (o.rate) R.rate = +o.rate; R.mode = o.mode || 'clean';
            V.ready.then(controls);
            setTimeout(() => show(true), 150);
          } else { if (!b.querySelector('p.rdnow') || !document.querySelector('#rdprog').textContent) show(false); controls(); } }, 500);
        window.addEventListener('pagehide', () => { if (R.playing) save(); }); }""")
    demo.load(None, None, None, js="""() => { document.addEventListener('click', (e) => {  // a cost box closes when you click anywhere else; one open at a time
        document.querySelectorAll('details.mcost[open]').forEach(d => { if (!d.contains(e.target)) d.open = false; }); });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') document.querySelectorAll('details.mcost[open]').forEach(d => d.open = false); }); }""")
    demo.load(None, None, None, js="""() => { document.addEventListener('copy', (e) => {  // text selected in the chat is copied without the extra blank lines and spaces around it
        const s = getSelection(); if (!s.rangeCount || !s.anchorNode || !s.anchorNode.parentElement || !s.anchorNode.parentElement.closest('#chat')) return;
        e.clipboardData.setData('text/plain', s.toString().replace(/[ \\t]+\\n/g, '\\n').replace(/\\n{3,}/g, '\\n\\n').trim()); e.preventDefault(); })
      document.addEventListener('click', (e) => {  // the copy button under an answer copies only the answer: not the thinking, sources or fact check folded above and below it
        const b = e.target.closest("#chat button[aria-label='Copy message']"); if (!b) return;
        let row = b.closest('.message-buttons'); row = row && row.previousElementSibling;
        while (row && !row.querySelector('.message')) row = row.previousElementSibling;
        if (!row || !row.querySelector('.bot')) return;
        const text = [...row.querySelectorAll('.message')].filter(m => !m.querySelector('.thought-group')).map(m => m.innerText.trim()).filter(Boolean).join('\\n\\n');
        if (!text) return;
        e.stopPropagation(); e.preventDefault();
        const ta = document.createElement('textarea'); ta.value = text; ta.setAttribute('readonly', ''); ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
        document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
        b.classList.add('jcopied'); setTimeout(() => b.classList.remove('jcopied'), 1200); }, true) }""")
    demo.load(None, None, screen, js="() => { let v = 'modern-paper'; try { v = localStorage.getItem('joshua-screen3') || 'modern-paper' } catch (e) {} if (!['modern-paper', 'modern-paper-night', 'novel', 'black', 'green', 'amber'].includes(v)) v = 'modern-paper'; document.body.dataset.screen = v; return v }")
    textsize.change(None, textsize, None, js="(v) => { document.documentElement.style.fontSize = (16 * v) + 'px'; try { localStorage.setItem('joshua-zoom', v) } catch (e) {} }")
    demo.load(None, None, textsize, js="() => { let v = '1'; try { v = localStorage.getItem('joshua-zoom') || '1' } catch (e) {} document.documentElement.style.fontSize = (16 * v) + 'px'; return v }")
    demo.load(None, None, None, js="""() => {  // getting around: the ☰ has a name; clicking the open chat closes the page in front of it; old warnings go when you change page; the menu marks where you are
        const lab = """ + json.dumps(t("Show or hide the menu")) + """, tog = document.querySelector('#sidetoggle');
        if (tog) { tog.setAttribute('aria-label', lab); tog.title = lab; }
        const pageOpen = () => { const c = document.querySelector('#chatcol'); return !c || c.offsetParent === null };
        const clearToasts = () => document.querySelectorAll('.toast-wrap .toast-close, .toast-body .toast-close, [data-testid="toast-close"]').forEach(b => b.click());
        document.addEventListener('click', (e) => {
          if (e.target.closest('#convs .cmore')) return;  // the chat's ⋯ menu
          const l = e.target.closest('#convs label');
          if (l && l.classList.contains('selected') && pageOpen()) { const b = document.querySelector('#tochatgo'); if (b) b.click(); }
          if (l || e.target.closest('#newchat, #panels button, button.backbtn')) clearToasts();
        }, true);
        const PAGES = {modelspage: 'modelsbtn', schedpanel: 'schedbtn', mempage: 'membtn', setpage: 'setbtn', binpage: 'binbtn', teampage: 'teambtn', readpage: 'readbtn', podcast: 'podbtn'};
        const mark = () => { let here = null;
          for (const [pg, bt] of Object.entries(PAGES)) { const p = document.getElementById(pg); if (p && p.offsetParent !== null) here = bt; }
          const tn = document.getElementById('tune'); if (tn && tn.offsetParent !== null) here = (document.querySelector('#tunetitle') || {}).textContent === """ + json.dumps(t("Build a model from scratch")) + """ ? 'newmodelbtn' : 'tunebtn';
          document.querySelectorAll('#panels button').forEach(b => { const on = b.id === here; if (on !== b.hasAttribute('aria-current')) on ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current'); }); };
        setInterval(mark, 400);
        const TIP = """ + json.dumps({"desk": t("A desk is a ready-made setup for one kind of work, with its own instructions and tools."), "model": t("Model"), "message": t("Message"), "newchat": t("New chat")}) + """;
        const label = () => {  // explanations on hover, and names for screen readers, on controls that have none
          ['#o_goal', '#o_swarm'].forEach(s => { const o = document.querySelector(s), l = o && o.querySelector('label'), i = o && o.querySelector('.info-text'); if (l && i && !l.title) l.title = i.textContent; });
          const d = document.querySelector('#deskpick'); if (d && !d.title) d.title = TIP.desk;
          const m = document.querySelector('#model input'); if (m && !m.getAttribute('aria-label')) m.setAttribute('aria-label', TIP.model);
          const x = document.querySelector('#promptrow textarea'); if (x && !x.getAttribute('aria-label')) x.setAttribute('aria-label', TIP.message);
          const tm = document.querySelector('#ttime input'); if (tm && tm.type !== 'time') tm.type = 'time';  // the browser's own time picker; it still gives HH:MM
          let bar = document.querySelector('#jtitle');  // with the side menu hidden: the chat's name and a New chat button next to the ☰
          if (!bar) { bar = document.createElement('div'); bar.id = 'jtitle'; bar.innerHTML = "<button type='button' class='jnew' title='" + TIP.newchat + "' aria-label='" + TIP.newchat + "'>＋</button><span class='jname'></span>";
            bar.querySelector('button').addEventListener('click', () => { const n = document.querySelector('#newchat'); if (n) n.click(); }); document.body.appendChild(bar); }
          const sel = document.querySelector('#convs label.selected'); const name = sel ? sel.textContent.replace('\u2063', '').replace('⋯', '').trim() : '';
          const nm = bar.querySelector('.jname'); if (nm.textContent !== name) nm.textContent = name;
          bar.classList.toggle('on', document.body.classList.contains('noside') && !document.body.classList.contains('narrow') && !pageOpen()); };
        setInterval(label, 500);
        const tall = () => document.body.classList.toggle('short', innerHeight < 860); tall(); addEventListener('resize', tall); }""")
    sidetoggle.click(None, None, None, js="""() => {
        if (document.body.classList.contains('narrow')) { document.body.classList.toggle('mside'); return }  // a phone: the side column slides over the chat
        const h = document.body.classList.toggle('noside'); try { localStorage.setItem('joshua-noside', h ? '1' : '') } catch (e) {} }""")
    demo.load(None, None, None, js="""() => {  // a phone or a narrow window: the body gets the class 'narrow', and choosing a chat or a page closes the side column
        const fit = () => { const n = innerWidth <= 760; document.body.classList.toggle('narrow', n); if (n) document.body.classList.remove('noside'); else document.body.classList.remove('mside');
          const ta = document.querySelector('#promptrow textarea');  // the message box measured itself in the old layout: measure again
          if (ta) setTimeout(() => { ta.style.height = ''; ta.dispatchEvent(new Event('input', {bubbles: true})); }, 50); };
        fit(); addEventListener('resize', fit);
        document.addEventListener('click', (e) => {
          if (!document.body.classList.contains('narrow') || !document.body.classList.contains('mside')) return;
          if (e.target.closest('.cmore')) return;
          if (e.target.closest('#convs label, #newchat, #panels button, #membtn, #setbtn, #binbtn, #modelsbtn, #tunebtn, #newmodelbtn')) setTimeout(() => document.body.classList.remove('mside'), 150);
        }, true); }""")
    demo.load(None, None, None, js="() => { try { if (localStorage.getItem('joshua-noside')) document.body.classList.add('noside') } catch (e) {} }")
    DESK_TIPS = {"": t("General: everyday questions, with all of Joshua's tools"), "journalism": t("Journalism: fact-check, write news, transcribe interviews"),
                 "research": t("Research: find and read papers, analyse data, write academic text"), "teaching": t("Teaching: for teachers. Plan lessons, make exercises and quizzes, give feedback"),
                 "study": t("Study: for students. Learn a subject with explanations, summaries and practice questions"), "falken": t("Prof Falken: a hidden game computer from the 1980s, just for fun")}
    UX_TEXT = {"pinned": t("Pinned"), "recent": t("Recent"), "search": t("Search chats"), "sending": t("Sending…"), "working": t("JOSHUA IS WORKING"),
               "desks": {k: v for k, v in DESK_TIPS.items()}, "local": t("on this computer · free"), "cloud": t("in the cloud · paid"), "yours": t("made by you · free"), "key": t("your API key · paid")}
    demo.load(None, None, None, js="""() => {  // UX review, 28 November 2026: the side column, the model menu, the status above the message box, memory notes, and small helpers
        const T = """ + json.dumps(UX_TEXT, ensure_ascii=False) + """;
        const PIN = '\\u2063';
        // ---- the side column: Advanced folds; the chats can be searched; Pinned and Recent; the open chat stays in view
        try { if (localStorage.getItem('joshua-navadv') === '1') document.body.classList.add('navadv-open') } catch (e) {}
        const fold = () => { const b = document.querySelector('#advfold'); if (b) b.setAttribute('aria-expanded', document.body.classList.contains('navadv-open') ? 'true' : 'false'); };
        document.addEventListener('click', (e) => {
          if (e.target.closest('#advfold')) { const on = document.body.classList.toggle('navadv-open'); try { localStorage.setItem('joshua-navadv', on ? '1' : '') } catch (x) {} fold(); return; }
          const nb = e.target.closest('#panels button[id]');  // the page you open is marked at once, not when it has finished loading
          if (nb) { document.querySelectorAll('#panels button[aria-current]').forEach(b => b.removeAttribute('aria-current')); if (!/fbbtn/.test(nb.id)) nb.setAttribute('aria-current', 'page'); }
          if (e.target.closest('.fclose')) { document.body.dataset.files = ''; return; }
        }, true);
        let lastSel = null;
        const side = () => { fold();
          const head = document.querySelector('#chatshead');
          if (head && !head.querySelector('input')) { const q = document.createElement('input'); q.className = 'chatsearch'; q.type = 'search'; q.placeholder = T.search; q.setAttribute('aria-label', T.search);
            q.addEventListener('input', side); q.addEventListener('keydown', (e) => e.stopPropagation()); head.appendChild(q); }
          const q = ((head && head.querySelector('input')) || {}).value || '', words = q.toLowerCase().trim();
          const labels = [...document.querySelectorAll('#convs label')]; let firstPin = null, firstRecent = null; const anyPin = labels.some(l => l.textContent.startsWith(PIN) || l.classList.contains('pinned'));
          labels.forEach(l => { const name = l.textContent.replace(PIN, '').replace('⋯', '').trim();
            if (l.title !== name) l.title = name;
            const hide = !!words && !name.toLowerCase().includes(words); if (l.classList.contains('jhide') !== hide) l.classList.toggle('jhide', hide);
            const pinned = l.classList.contains('pinned');
            if (!hide && pinned && !firstPin) firstPin = l; if (!hide && !pinned && !firstRecent) firstRecent = l;
            l.classList.remove('grpstart'); });
          if (anyPin && firstPin) { firstPin.classList.add('grpstart'); firstPin.dataset.group = T.pinned; }
          if (anyPin && firstRecent) { firstRecent.classList.add('grpstart'); firstRecent.dataset.group = T.recent; }
          const sel = document.querySelector('#convs label.selected');
          if (sel && sel !== lastSel) { lastSel = sel; const box = document.querySelector('#convs'); if (box) { const r = sel.getBoundingClientRect(), b = box.getBoundingClientRect();
              if (r.top < b.top || r.bottom > b.bottom) sel.scrollIntoView({block: 'nearest'}); } }
          document.querySelectorAll('#deskpick label').forEach(l => { const i = l.querySelector('input'), tip = i && T.desks[i.value]; if (tip && l.title !== tip) l.title = tip; });
        };
        // ---- the model menu: where each model runs, and "Add a model" set apart
        const kinds = () => document.querySelectorAll('#model .options li').forEach(li => { if (li.dataset.kindDone) return; li.dataset.kindDone = 1;
          const txt = (li.getAttribute('aria-label') || li.textContent).trim();  // the label as written, before the page swaps emoji for icons
          if (/^[+＋]/.test(txt)) { li.classList.add('jadd'); return; }
          const k = li.querySelector('.ji-key') || txt.includes('🔑') ? T.key : li.querySelector('.ji-cloud') || txt.includes('☁') ? T.cloud : txt.includes('★') ? T.yours : T.local;
          li.dataset.kind = k; });
        // ---- while Joshua works: a status just above the message box, at once when you send
        let sentAt = 0;
        const pill = () => { const col = document.querySelector('#chatcol'); if (!col) return null; let p = document.querySelector('#jstatus');
          if (!p) { p = document.createElement('div'); p.id = 'jstatus'; p.setAttribute('role', 'status'); p.setAttribute('aria-live', 'polite'); col.appendChild(p); } return p; };
        const status = () => { const p = pill(); if (!p) return; const col = document.querySelector('#chatcol'), pr = document.querySelector('#promptrow');
          if (pr && col) p.style.bottom = Math.max(8, col.getBoundingClientRect().bottom - pr.getBoundingClientRect().top + 8) + 'px';
          const w = document.querySelector('#working .working');
          if (w) { const html = w.innerHTML.replace(T.working, '').replace(/^\\s*(<span class="wdot"><\\/span>)?\\s*·\\s*/, '$1'); if (p.dataset.h !== html) { p.dataset.h = html; p.innerHTML = html; } p.classList.add('on'); sentAt = 0; return; }
          if (sentAt && Date.now() - sentAt < 20000) { if (p.dataset.h !== 'sending') { p.dataset.h = 'sending'; p.innerHTML = "<span class='wdot'></span>" + T.sending; } p.classList.add('on'); return; }
          sentAt = 0; p.classList.remove('on'); p.dataset.h = ''; };
        const sent = () => { const ta = document.querySelector('#promptrow textarea'); if (ta && ta.value.trim() || document.querySelector('#promptrow .thumbnail-item, #promptrow .file-preview')) { sentAt = Date.now(); setTimeout(status, 30); } };
        document.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && e.target.closest && e.target.closest('#promptrow textarea')) sent(); }, true);
        document.addEventListener('click', (e) => { if (e.target.closest('#promptrow .submit-button, button.starter[data-send]')) sent(); }, true);
        // ---- memory: the notes one per row, with Change and Delete; the whole text under "Edit as text"
        const memlist = () => { const box = document.querySelector('#memlist'), ta = document.querySelector('#memtext textarea'); if (!box || !ta) return;
          if (box.dataset.src === ta.value && box.dataset.drawn) return; box.dataset.src = ta.value; box.dataset.drawn = 1;
          const lines = ta.value.split('\\n'), D = box.dataset, esc = (x) => x.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[c]);
          const notes = lines.map((l, i) => [l, i]).filter(([l]) => l.trim() && !/^#/.test(l.trim()));
          let h = notes.length ? '' : "<div class='memempty'>" + esc(D.empty) + '</div>';
          notes.forEach(([l, i]) => { h += "<div class='memrow' data-i='" + i + "'><span class='memtxt'>" + esc(l.replace(/^\\s*-\\s*/, '').replace(/^\\[stated\\]\\s*/, '')) + "</span><button type='button' class='memedit'>" + esc(D.edit) + "</button><button type='button' class='memdel'>" + esc(D.del) + '</button></div>'; });
          h += "<div class='memtools'><button type='button' class='memadd'>" + esc(D.add) + "</button><button type='button' class='memastext'>" + esc(D.astext) + "</button>" + (box.dataset.dirty ? "<span class='memwarn'>" + esc(D.unsaved) + '</span>' : '') + '</div>';
          box.innerHTML = h; };
        const setText = (v) => { const ta = document.querySelector('#memtext textarea'), box = document.querySelector('#memlist'); ta.value = v; ta.dispatchEvent(new Event('input', {bubbles: true})); box.dataset.dirty = 1; box.dataset.drawn = ''; memlist(); };
        document.addEventListener('click', (e) => { const box = e.target.closest('#memlist'); if (!box) return; const ta = document.querySelector('#memtext textarea'); if (!ta) return;
          const lines = ta.value.split('\\n'), row = e.target.closest('.memrow'), i = row ? +row.dataset.i : -1;
          if (e.target.closest('.memdel')) { lines.splice(i, 1); setText(lines.join('\\n')); return; }
          if (e.target.closest('.memastext')) { document.body.classList.toggle('memastext'); return; }
          const edit = (k, start) => { const r = k >= 0 ? box.querySelector(".memrow[data-i='" + k + "']") : null; const f = document.createElement('input'); f.value = start;
            const done = (save) => { const v = f.value.trim(); if (save && v) { if (k >= 0) lines[k] = '- ' + v; else lines.push('- ' + v); setText(lines.join('\\n')); } else { box.dataset.drawn = ''; memlist(); } };
            f.addEventListener('keydown', (ev) => { ev.stopPropagation(); if (ev.key === 'Enter') { ev.preventDefault(); done(true); } if (ev.key === 'Escape') done(false); });
            f.addEventListener('blur', () => done(true), {once: true});
            if (r) { r.querySelector('.memtxt').replaceWith(f); } else { const n = document.createElement('div'); n.className = 'memrow'; n.appendChild(f); box.querySelector('.memtools').before(n); }
            f.focus(); };
          if (e.target.closest('.memedit')) return edit(i, lines[i].replace(/^\\s*-\\s*/, ''));
          if (e.target.closest('.memadd')) return edit(-1, ''); });
        document.addEventListener('click', (e) => { if (e.target.closest('#memsave')) { const b = document.querySelector('#memlist'); if (b) { b.dataset.dirty = ''; b.dataset.drawn = ''; } } }, true);
        // ---- Teach and Build: the model's real name where it will be installed; technical details only when asked for
        const tune = () => { const n = document.querySelector('#ftproject input'), c = document.querySelector('.ftinstname');
          if (n && c) { const v = 'joshua-' + (n.value || '…'); if (c.textContent !== v) c.textContent = v; }
          const tech = document.querySelector('#ftshowtech input'); if (tech) document.body.classList.toggle('showtech', tech.checked); };
        // ---- the Reader: a book can be opened with the keyboard too
        document.addEventListener('keydown', (e) => { const r = e.target.closest && e.target.closest('.rdrow[role=button]'); if (r && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); r.click(); } });
        let lastBad = null;
        const bad = () => { const b = [...document.querySelectorAll('.invalid')].find(x => x.offsetParent); if (b && b !== lastBad) { b.scrollIntoView({block: 'center', behavior: 'smooth'}); const f = b.querySelector('input, textarea'); if (f) f.focus({preventScroll: true}); } lastBad = b || null; };
        const all = () => { side(); kinds(); status(); memlist(); tune(); bad(); };
        new MutationObserver(() => { clearTimeout(window.joshuaUx); window.joshuaUx = setTimeout(all, 120); }).observe(document.body, {childList: true, subtree: true, characterData: true});
        setInterval(status, 1000); all(); }""")

ICON = (pathlib.Path(__file__).parent / "icon.png").resolve()
LOOK_EARLY = ("<script>(function () {  // the saved look and text size, before anything is drawn: no flash of another look at start-up\n"
              "  var ok = ['modern-paper', 'modern-paper-night', 'novel', 'black', 'green', 'amber'], v = 'modern-paper', z = '1';\n"
              "  try { v = localStorage.getItem('joshua-screen3') || v; z = localStorage.getItem('joshua-zoom') || z } catch (e) {}\n"
              "  if (ok.indexOf(v) < 0) v = 'modern-paper';\n"
              "  document.documentElement.style.fontSize = (16 * z) + 'px';\n"
              "  var set = function () { if (document.body) { document.body.dataset.screen = v; return true } };\n"
              "  if (!set()) new MutationObserver(function (m, o) { if (set()) o.disconnect() }).observe(document.documentElement, {childList: true});\n"
              "})();</script>")
APP_HEAD = LOOK_EARLY + (f"<link rel='apple-touch-icon' href='/gradio_api/file={ICON}'>"  # the icon Safari uses for "Add to Dock"
            "<meta name='apple-mobile-web-app-title' content='Joshua'><meta name='application-name' content='Joshua'>")

# Icons: the interface is written with emoji (in labels, buttons, tool boxes); the page swaps each one for a drawn icon
# from Phosphor (light weight, MIT licence, icons.json). Only interface text is touched, never what you or the model write.
UI_ICONS = json.loads((pathlib.Path(__file__).parent / "icons.json").read_text(encoding="utf-8"))
ICON_JS = r"""(() => {
  const I = window.JOSHUA_ICONS, keys = Object.keys(I.emoji).sort((a, b) => b.length - a.length);
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), alt = keys.map(esc).join('|');
  const lead = new RegExp('^([\\s\\u2063]*)((?:(?:' + alt + ')\\uFE0F?\\s*)+)'), each = new RegExp('(' + alt + ')\\uFE0F?', 'g');
  const AREA = 'button, label, .title, h2, h3, .cloudbadge, .starter, .cmenu, th, td, .cloudstat, .fhead, #schedtitle, summary, li, .msub2, .ftitle';
  const made = new WeakMap();
  const fix = (node) => {
    const el = node.parentElement;
    if (!el || el.closest('.ji, .message-content, textarea, pre, code') || !el.closest(AREA)) return;
    const was = made.get(node);
    if (was && was.value === node.nodeValue) return;  // our own change, already done
    if (was) was.spans.forEach(x => x.remove());      // the text was set again: start over
    const m = node.nodeValue.match(lead);
    if (!m) return made.delete(node);
    const spans = [];
    for (const e of m[2].match(each)) {
      const name = I.emoji[e] || I.emoji[e.replace('\uFE0F', '')];
      if (!name) continue;
      const s = document.createElement('span'); s.className = 'ji ji-' + name; s.innerHTML = I.svg[name]; s._for = node;
      el.insertBefore(s, node); spans.push(s);
    }
    const value = m[1] + node.nodeValue.slice(m[0].length);
    made.set(node, {spans, value});
    node.nodeValue = value;
  };
  const walk = (root) => {
    if (root.nodeType === 3) return fix(root);
    if (root.nodeType !== 1 || root.closest?.('.ji')) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n;
    while ((n = w.nextNode())) fix(n);
  };
  let todo = new Set(), queued = false;
  const run = () => { queued = false; const list = [...todo]; todo = new Set(); list.forEach(walk); document.querySelectorAll('.ji').forEach(s => { if (!s._for || !s._for.isConnected) s.remove(); }); };
  new MutationObserver(ms => { for (const m of ms) { if (m.type === 'characterData') todo.add(m.target); else m.addedNodes.forEach(n => todo.add(n)); }
    if (!queued) { queued = true; requestAnimationFrame(run); } }).observe(document.documentElement, {subtree: true, childList: true, characterData: true});
  addEventListener('DOMContentLoaded', () => walk(document.body));
})();"""
ICON_HEAD = ("<script>window.JOSHUA_ICONS=" + json.dumps({"emoji": UI_ICONS["emoji"], "svg": UI_ICONS["svg"]}, ensure_ascii=False).replace("</", "<\\/") + ";</script>"
             "<script>" + ICON_JS + "</script>"
             "<style>.ji{display:inline-flex;width:1.2em;height:1.2em;margin-right:.42em;vertical-align:-.22em;flex:none}.ji svg{width:100%;height:100%}"
             ".cloudbadge .ji,.sicon .ji{margin-right:.18em}</style>")
SERVICE = os.environ.get("JOSHUA_SERVICE")  # set when Joshua runs in the background from login (launchd / systemd)
if remote_setting():
    remote.switch(True, PORT)  # your devices on Tailscale can reach Joshua again after a restart
    threading.Thread(target=remote.wake, daemon=True).start()  # and Tailscale is started, if it isn't running
# Gradio records every event (our once-a-second timers too) in a "run history" kept in the browser, rewriting it
# each time: with long chats this blocked the page for most of every second. Joshua doesn't need it.
LAUNCH_EXTRA = {"run_history": False} if "run_history" in inspect.signature(gr.Blocks.launch).parameters else {}
demo.launch(**LAUNCH_EXTRA, inbrowser=not (SERVICE or os.environ.get("JOSHUA_NO_OPEN")), favicon_path=str(ICON), allowed_paths=[str(ROOT), str(ICON), str(pathlib.Path.home())], css=CSS, theme=THEME, head=APP_HEAD + ICON_HEAD + "<link href='https://fonts.googleapis.com/css2?family=VT323&family=Figtree:ital,wght@0,400;0,500;0,600;1,400&family=Newsreader:opsz,wght@6..72,500&family=Share+Tech+Mono&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap' rel='stylesheet'>")
