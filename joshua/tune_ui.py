"""The 'Teach Joshua' panel: a step-by-step fine-tuning wizard. Built inside web.py's gr.Blocks."""
import html, json, pathlib
import gradio as gr, pandas as pd
from markdown_it import MarkdownIt
import finetune as ft
import scratch as sc
from i18n import t

INTRO = t("""
### What fine-tuning is
Joshua's brain is a model: billions of numbers learned from a huge amount of text. **Fine-tuning** nudges a few of those numbers
using *your* examples, so the model picks up a habit: a tone of voice, a format, a way of answering, your jargon.

**It teaches style, not facts.** Train it on 200 answers about your company and it will *sound* like it knows your company,
but it will invent the details. For facts, Joshua already has better tools: memory, your files, and the web.

**How it works here.** We use **LoRA**: instead of changing the whole model, training adds a thin layer of new numbers
(an *adapter*, a few tens of MB) on top of it. It takes 15–60 minutes, entirely on your computer.
Your examples never leave it.

**What you get.** A new model in Joshua's menu, named `joshua-<your name>`, that answers the way you taught it.
**You choose which model to teach** in step ①: any open model your computer has room for. Bigger ones learn better and keep
more of their skills, but take longer. The result works with text only, even if the original could also see images.

**The five steps:** ① name it · ② give examples · ③ train · ④ compare before and after · ⑤ install it into Joshua.
""")

STEP1 = t("""Each fine-tune is a **project** with its own examples and training. Give it a short name in lowercase,
like `formal-italian` or `newsroom-style`. The finished model will be called `joshua-<name>`.""")

STEP2 = t("""The model learns from **examples**: something you might say, and the exact answer you want back.
Quality beats quantity: **100–500 good examples** work far better than 5,000 sloppy ones. Every example should show the habit
you want, and the answers should be ones you'd be happy to get.

Three ways to add them, which you can mix:
- **Upload a file**: `.jsonl` (one `{"messages": [...]}` chat per line, the standard format), `.json`, or a `.csv` with two columns, question and answer.
- **From your chats**: pick conversations you liked; each of your messages becomes an example with Joshua's reply. Edit the replies until they're exactly right.
- **By hand**: write them one at a time below.

You can edit any cell in the table, and empty a row's cells to delete it. Press **Save examples** when you're done.""")

STEP3 = t("""Training shows the model your examples over and over, and each time adjusts the adapter a little so that
its answers get closer to yours. Three settings matter:

- **Steps**: how many small adjustments. One *pass* means every example has been seen once; 2–5 passes is normal.
- **Learning rate**: how big each adjustment is. Too small and nothing changes; too big and it forgets how to talk properly.
- **Layers**: how many of the model's 36 layers get an adapter. More layers learn more, use more memory.

The presets pick sensible values. While it trains, watch the chart: **training loss** measures how far its answers are
from yours on the examples it learns from; **validation loss** measures it on the 1 in 10 examples kept aside, which it never studies.
Both should go down. If training keeps falling but validation starts *rising*, it's memorising instead of learning:
stop, and train again with fewer steps.

**Don't overdo it.** Strong training on examples that all look the same can make the model answer *everything* that way,
and forget how to use Joshua's tools (search, files…). Vary your examples, start with **Light touch**, and step up only if the
habit doesn't stick. A validation loss close to zero is a warning sign, not a success.

Joshua's chat models are unloaded while training, so the computer's memory goes to the training. Chatting will be slow until it ends.""")

STEP4 = t("""Before installing, ask the same question to the model **before** and **after** training, side by side.
Use questions that are *not* in your examples: that's how you see whether it learned the habit or only memorised answers.""")

STEP5 = t("""Installing makes your trained model a normal Ollama model that Joshua can use, in three automatic steps:
1. **Merge**: the adapter is folded into a copy of the model.
2. **Convert**: that copy is turned into **GGUF**, the file format Ollama runs.
3. **Add to Ollama**: it gets Joshua's chat template, so it can still use tools, and a name: `joshua-<name>`.

While it works it needs a lot of free disk, about three times the model's full size (step ① says how much);
temporary files are deleted at the end.
Then pick it from the model menu under the prompt.""")

MD = MarkdownIt()
PRESETS = {"Light touch": (2, "2e-5", 8), "Normal": (3, "5e-5", 16), "Strong": (5, "1e-4", 16)}  # shown translated; the keys stay English


def res_html():
    """One line on what this computer can do."""
    r, missing = ft.resources(), ft.ready()
    fitting = [b for b in ft.bases() if ft.fits(b, r)[0]]
    ok = fitting and not missing
    parts = [("✓ " + t("Ready to train")) if ok else ("✗ " + t("Not ready yet")), html.escape(r["chip"]), t("{n} GB memory", n=r["ram"]), t("{n} GB free disk", n=r["free"])]
    if fitting:
        parts.append(t("the biggest model that fits: {name}", name=html.escape(fitting[-1]["label"])))
    extra = "".join(f"<div>✗ {html.escape(m)}</div>" for m in missing) + ("" if fitting else "<div>✗ " + t("No model fits: free some disk, or use a computer with more memory.") + "</div>")
    return f"<div class='ftres{'' if ok else ' bad'}'>" + " · ".join(parts) + extra + "</div>"


def why(short):
    kind, gb = short
    return t("needs about {n} GB of memory", n=gb) if kind == "ram" else t("needs {n} GB of free disk", n=gb)


def base_choices():
    """Every model you can teach, smallest first, saying whether it fits this computer and if not, why."""
    r, out = ft.resources(), []
    for b in sorted(ft.bases(), key=lambda b: b["params"]):
        ok, short = ft.fits(b, r)
        out.append((("✓ " if ok else "✗ ") + b["label"] + f" · {b['gb']:g} GB" + ("" if ok else " · " + why(short)), b["id"]))
    return out + [("＋ " + t("Another model from Hugging Face…"), "custom")]


def base_info(entry):
    n = ft.needs(entry)
    ok, short = ft.fits(entry)
    return ("<div class='ftres'>" + ("✓ " if ok else "✗ ") + html.escape(entry["label"]) + " · "
            + t("training needs about {ram} GB of memory and {disk} GB of free disk; installing the result needs {inst} GB of free disk for a while, and it then takes {final} GB.",
                ram=n["ram"], disk=n["train_disk"], inst=n["install_disk"], final=n["final"])
            + ("" if ok else "<br>✗ " + why(short)) + "</div>")


def table(rows):
    return pd.DataFrame(rows or [["", ""]], columns=[t("You say"), t("Joshua answers")])


def rows_of(df):
    if df is None:
        return []
    return [[str(q), str(a)] for q, a in df.values.tolist() if str(q).strip() and str(a).strip() and str(q) != "nan" and str(a) != "nan"]


def count_html(rows):
    n = len(rows)
    tip = (t("Add at least 20 to train.") if n < 20 else t("Enough to try; 100 or more gives steadier results.") if n < 100 else t("A good amount."))
    return f"<div class='ftcount'>{t('{n} EXAMPLES', n=n)} · {tip}</div>"


def loss_svg(train, val, old=()):
    """The loss chart, drawn here as a small SVG: training in blue, validation in orange, lower is better.
    In a second stage, `old` is the loss on the first texts, in grey: rising means the model is forgetting them."""
    pts = list(train) + list(val) + list(old)
    head = "<div class='ftchart'><div class='ftchart-head'>" + t("Loss, lower is better") + (f" <span class='k1'>■ {t('training')}</span> <span class='k2'>■ {t('validation')}</span>" if pts else "") + (f" <span class='k3'>■ {t('previous texts')}</span>" if old else "") + "</div>"
    if len(pts) < 2:
        return head + "<div class='ftchart-empty'>" + t("The chart appears when training starts.") + "</div></div>"
    W, H, L, B = 900, 230, 44, 26
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = 0, max(ys) * 1.08 or 1
    X = lambda x: L + (W - L - 10) * (x - x0) / max(x1 - x0, 1)
    Y = lambda y: 8 + (H - B - 8) * (1 - (y - y0) / (y1 - y0))
    line = lambda ps, cls: (f"<polyline class='{cls}' points='" + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in ps) + "'/>") if len(ps) > 1 else "".join(f"<circle class='{cls}' cx='{X(a):.1f}' cy='{Y(b):.1f}' r='3'/>" for a, b in ps)
    grid = "".join(f"<line x1='{L}' x2='{W - 10}' y1='{Y(v):.1f}' y2='{Y(v):.1f}'/><text x='{L - 6}' y='{Y(v) + 4:.1f}' text-anchor='end'>{v:.1f}</text>" for v in (y1 / 1.08 * k / 4 for k in range(5)))
    ticks = "".join(f"<text x='{X(v):.1f}' y='{H - 6}' text-anchor='{('start', 'middle', 'middle', 'middle', 'end')[k]}'>{int(v)}</text>" for k, v in enumerate(x0 + (x1 - x0) * k / 4 for k in range(5)))  # the end ones inside the chart
    last = f"{t('step')} {int(xs[-1] if not val else max(xs))} · " + (t("training") + f" {train[-1][1]:.2f}" if train else "") + (f" · {t('validation')} {val[-1][1]:.2f}" if val else "") + (f" · {t('previous texts')} {old[-1][1]:.2f}" if old else "")
    return (head + f"<svg viewBox='0 0 {W} {H}'><g class='grid'>{grid}</g><g class='ticks'>{ticks}</g>"
            + line(train, "tr") + line(val, "va") + line(old, "ol") + f"</svg><div class='ftchart-last'>{html.escape(last)}</div></div>")


def more(text, label=None):
    """The longer explanation of a step, folded away under a line you can open."""
    return gr.HTML("<details class='ftmore'><summary>" + html.escape(label or t("More about this step")) + "</summary>" + MD.render(text) + "</details>")


STEPS = [("1", "Model"), ("2", "Examples"), ("3", "Train"), ("4", "Compare"), ("5", "Install")]


NEW_INTRO = t("""A model trained from zero knows only what is in your texts. Joshua builds it in four steps:

1. **Reading.** Every text file in the folder (txt, md, PDF, EPUB, Word, OpenDocument, web pages) is read; repeated paragraphs are dropped, and 1 paragraph in 50 is kept aside to check the model on text it never studies.
2. **Vocabulary.** The model can only read numbers. Joshua learns 8,192 frequent pieces of your texts (whole words, parts of words, single letters) and gives each a number. The texts become one long list of numbers.
3. **Training.** The model starts as random numbers. Over and over, it sees 512 numbers of your text and guesses each next one; every wrong guess nudges its weights. The chart shows how wrong it still is (lower is better), and every so often it writes a sample, so you can watch it go from noise to words to sentences. You can stop and carry on later.
4. **Ollama.** Joshua converts it to Ollama's format, so other programs can use it too.

**How much text?** A model wants about 20 pieces of text for every parameter it has: 200 million for the 10M model, a billion for the 50M one. With less, it learns your texts nearly by heart and repeats them. **What it won't do:** answer questions, follow instructions or use tools. It continues text, in the style of what it read. That's why it doesn't appear in the chat's model menu.""")

STAGE2 = t("""Like a painting redone in the style of another painter: the earlier steps give the model its language and what it knows, this one its voice. Joshua takes the model from step 4 (or from step 3, if you skipped it) and trains it further on the new texts, with the same vocabulary and a gentler learning rate, so it bends rather than starts over.

**The balance.** Too little of the new texts and it still sounds like before; too much and it forgets what it knew. So Joshua keeps a copy of the model at 8 points along the way: in step 6 you can try each one and choose. The chart has a third line, in grey: how well the model still knows the texts of the previous step. When it rises, the model is forgetting them.

**Mixing.** A share of each training batch can still come from the texts of the previous step. It slows the forgetting: the model takes on the new voice while keeping more of what it knew.""")


WIKI_LANGS = [("Italiano", "it"), ("English", "en"), ("Français", "fr"), ("Deutsch", "de"), ("Español", "es"), ("Polski", "pl"),
              ("Português", "pt"), ("Nederlands", "nl"), ("中文", "zh")]


def when(h):
    return t("about {n} min", n=max(1, round(h * 60))) if h < 1 else t("about {n} h", n=round(h, 1) if h < 10 else round(h))


def data_line(d):
    return t("{docs} documents · {words} words · {tokens} pieces of text to learn from.", docs=f"{d.get('docs', d.get('files', 0)):,}", words=f"{d['words']:,}", tokens=f"{d['train_tokens']:,}")


def download_html(n, res=None):
    """Step 1: while it runs, the part being downloaded; then what was got, from where, to where."""
    out = ""
    ph = (res or {}).get("phases", {})
    p = ph.get("download")
    if p and not ph.get("downloaded"):
        out += ("<div class='ftphases'><div class='ftphase now'><b><span class=wdot></span> " + html.escape(t("Downloading Wikipedia from Hugging Face")) + "</b><span>"
                + html.escape(t("part {k} of {n}, {mb} MB · {words} words so far", k=p["part"], n=p["parts"], mb=p["mb"], words=f"{p['words']:,}")
                              + (" " + t("of {target}", target=f"{p['target']:,}") if p.get("target") else "") + " · " + t("from {src} · to {to}", src=p["src"], to=home(p["to"])))
                + "</span><i>" + html.escape(t(PHASE_WHY["download"])) + "</i></div></div>")
    got = sc.downloaded(n) if n else None
    if got and got.get("kind") == "wiki":
        out += "<div class='ftres'>" + html.escape(t("Downloaded {parts} of the {of} parts of Wikipedia ({lang}): {mb} MB, {words} words.", parts=got["parts"], of=got["of"], lang=got["lang"], mb=f"{got['mb']:,}", words=f"{got['words']:,}")) + "</div>"
        out += "<div class='ftwhere'>" + html.escape(t("From {src}", src=got["src"])) + "<br>" + html.escape(t("To {to}, Hugging Face's download folder, which keeps them: downloading them again is instant.", to=home(got["to"]))) + "<br>" + html.escape(t("Nothing has been read yet: step 2 does that.")) + "</div>"
    elif got:
        out += "<div class='ftres'>" + html.escape(t("{n} files, {mb} MB, in {folder}.", n=got["n"], mb=got["mb"], folder=home(got["folder"]))) + "</div>"
        out += "<div class='ftwhere'>" + html.escape(t("Nothing to download: step 2 reads them where they are, without changing them.")) + "</div>"
    elif n and sc.data(n):
        out += "<div class='ftres'>" + html.escape(t("These texts were got before this step existed; step 2 has already read them.")) + "</div>"
    elif not out:
        out = "<div class='ftempty'>" + t("Choose where the texts come from, then press Get the texts.") + "</div>"
    return out


def new_info(n):
    d = sc.data(n) if n else None
    if not d:
        return "<div class='ftempty'>" + (t("Press Tokenize.") if n and sc.downloaded(n) else t("First do step 1.")) + "</div>"
    best = sc.best_size(n)
    note = (" " + t("That is little text: the model will mostly learn to repeat it. More text helps more than a bigger model.")) if d["train_tokens"] < 20 * sc.params(sc.SIZES[0]) else ""
    return "<div class='ftres'>" + data_line(d) + " " + t("Suggested size: {size}.", size=best) + html.escape(note) + "</div>" + where_html(d.get("where"))


def home(p):
    return str(p).replace(str(__import__("pathlib").Path.home()), "~")


def where_html(w):
    """Where the files of a reading went, and what happens to them."""
    if not w:
        return ""
    out = [t("The numbers the model learns from: {file} ({mb} MB).", file=home(w["data"]), mb=f"{w.get('data_mb', 0):,}"),
           t("The extracted text was only temporary and has been deleted.")]
    if w.get("cache"):
        out.append(t("The downloaded Wikipedia parts ({gb} GB) stay in Hugging Face's download folder, {folder}, so reading again doesn't download them.", folder=home(w["cache"]), gb=w.get("cache_gb", 0)))
    if w.get("folder"):
        out.append(t("Your files in {folder} were only read, not changed.", folder=home(w["folder"])))
    return "<div class='ftwhere'>" + "<br>".join(html.escape(x) for x in out) + "</div>"


PHASE_WHY = {  # what each phase does, in a line, shown under it
    "download": "Wikipedia comes already cleaned into plain text, one row per article, in files of a few hundred MB.",
    "read": "Each file is read as plain text; paragraphs that appear twice are kept once.",
    "text": "The text goes into one plain file, a paragraph per line, so the next phases can read it in a stream: it never has to fit in memory.",
    "vocab": "This is the tokenizer. It starts from the 256 possible bytes, so it can write any character, then over and over merges the pair of pieces that appears most often into a new piece ('d'+'e' → 'de', 'de'+'lla' → 'della'), until it has 8,192. Frequent words end up as one piece, rare ones as several.",
    "numbers": "With that vocabulary, every paragraph is cut into pieces and each piece is replaced by its number, stored in 2 bytes. One paragraph in 50 goes to a separate file the model never trains on, to check how it does on text it hasn't seen.",
}


def phases_html(res):
    """Each phase of reading the texts, with how far it got and which files it reads and writes."""
    ph, stage = res.get("phases", {}), res.get("stage", 1)
    source = str(res.get("source", ""))
    first = [] if source.startswith("wikipedia:") else ["download"] if source.startswith("wikiqa:") else ["read"]  # Wikipedia was downloaded in step 1
    order = first + ["text"] + (["vocab"] if stage == 1 else []) + ["numbers"]
    started = [k for k in order if k in ph]
    rows = []
    for k in order:
        p = ph.get(k)
        state = "done" if p and (p.get("done") or started and order.index(started[-1]) > order.index(k)) else "now" if p else "todo"
        mark = {"done": "✓", "now": "<span class=wdot></span>", "todo": "○"}[state]
        if k == "download":
            label = t("Downloading Wikipedia from Hugging Face")
            detail = t("part {k} of {n}, {mb} MB · from {src} · to {to}", k=p["part"], n=p["parts"], mb=p["mb"], src=p["src"], to=home(p["to"])) if p else ""
        elif k == "read":
            label = t("Reading your files")
            detail = t("file {i} of {n}: {file}", i=p["i"], n=p["n"], file=p["file"]) if p else ""
        elif k == "text":
            label = t("Extracting the text")
            if p:
                detail = t("{docs} documents · {words} words", docs=f"{p['docs']:,}", words=f"{p['words']:,}") + (" " + t("of {target}", target=f"{p['target']:,}") if p.get("target") else "") + " · " + t("into {file}, a temporary file", file=home(p["file"]))
            else:
                detail = ""
        elif k == "vocab":
            label = t("Learning the vocabulary")
            detail = t("8,192 pieces, chosen from the first {words} words · saved in {file}", words=f"{p['words']:,}", file=home(p["file"])) if p else ""
        else:
            label = t("Turning the text into numbers")
            detail = t("{tokens} of about {estimate} pieces · saved in {file}", tokens=f"{p['tokens']:,}", estimate=f"{p['estimate']:,}", file=home(p["file"])) if p else ""
        why = PHASE_WHY.get(k)
        rows.append(f"<div class='ftphase {state}'><b>{mark} {html.escape(label)}</b>" + (f"<span>{html.escape(detail)}</span>" if detail else "")
                    + (f"<i>{html.escape(t(why))}</i>" if why else "") + "</div>")
    return "<div class='ftphases'>" + "".join(rows) + "</div>"  # where the files went is in the summary above, once done


def live_html(run):
    """The training's progress, right under its buttons."""
    if not run.train and not run.val:
        return "<div class='ftres'><span class=wdot></span> " + html.escape(t("Starting: loading the texts and building the model…")) + "</div>"
    step = (run.train or run.val)[-1][0]
    secs = int(__import__("time").time() - run.started)
    took = f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}" if secs >= 3600 else f"{secs // 60}:{secs % 60:02d}"
    left = None if run.done else run.left()
    msg = t("Step {i} of {n}", i=f"{step:,}", n=f"{run.total:,}") + " · " + took
    if left is not None:
        m = max(1, round(left / 60))
        msg += " · " + (t("about {n} min left", n=m) if m < 60 else t("about {h} h {m} min left", h=m // 60, m=m % 60))
    if run.done:
        msg = (t("Finished") if run.ok else t("Stopped")) + " · " + msg
    return "<div class='ftres'>" + ("" if run.done else "<span class=wdot></span> ") + html.escape(msg) + "</div>"


QA, TUNE = 2, 3  # on disk, step 4 (questions and answers) is stage 2 and step 5 (fine-tuning) is stage 3


def src(n, st):
    f = sc.pdir(n) / f"source{st}.json"
    return json.loads(f.read_text()) if n and f.exists() else {}


def valid(n, st):
    """The texts read for a step are its own: questions and answers for step 4, a folder for step 5."""
    return bool(n and sc.data(n, st)) and src(n, st).get("kind") == ("qa" if st == QA else "folder")


def tune_base(n):
    """Step 5 starts from the last copy of step 4, or from the pre-training if step 4 was skipped."""
    qa = [v for _, v in sc.versions(n) if v.startswith("stage2/")] if n and src(n, QA).get("kind") == "qa" else []
    return qa[-1] if qa else "model"


def step_info(n, size, st):
    """The box of step 4 (st=QA) or 5 (st=TUNE): where it starts from, and the texts read for it."""
    head = ("<div class='ftres'>" + html.escape(t("It starts from {v}.", v=version_label(n, "model" if st == QA else tune_base(n)))) + "</div>") if n and sc.trained(n) else ""
    if not valid(n, st):
        return head + "<div class='ftempty'>" + (t("Choose how many questions, then press Read them.") if st == QA else t("Choose the folder, then press Read them.")) + "</div>"
    d = sc.data(n, st)
    k = sc.steps(n, size, st)
    return head + "<div class='ftres'>" + data_line(d) + " " + t("This step takes {steps} steps, {when}.", steps=f"{k:,}", when=when(sc.hours(n, size, st))) + "</div>" + where_html(d.get("where"))


def source_of(kind, path, lang, millions, whole=False, qcount=50000):
    if kind == "qa":
        return sc.wikiqa(lang or "it", int(qcount or 50000))
    return sc.wiki(lang or "it", "all" if whole else (millions or 150)) if kind == "wiki" else str(__import__("pathlib").Path((path or "").strip()).expanduser())


def size_choices(n):
    if not (n and sc.data(n)):
        return [(s, s) for s in sc.SIZES]
    return [(f"{s} · {when(sc.hours(n, s))}", s) for s in sc.SIZES]


def arch_html(size, n=None):
    """The chosen size spelled out: what the model is made of."""
    import scratch_job as job
    d = sc.data(n) if n else None
    h, layers, heads, inter = job.SIZES[size]
    p = sc.params(size, d["vocab"] if d else job.VOCAB)
    return ("<div class='ftres'>" + t("{p} million parameters · {layers} layers · width {h} · {heads} attention heads · middle width {inter} · vocabulary of {vocab} pieces · it sees {ctx} pieces at a time",
            p=f"{p / 1e6:.1f}", layers=layers, h=h, heads=heads, inter=inter, vocab=f"{(d['vocab'] if d else job.VOCAB):,}", ctx=job.CTX)
            + (" · " + t("{n} steps", n=f"{sc.steps(n, size):,}") if d else "") + "</div>")


def stage_texts(n, stage):
    """What a stage was trained on, in a few words: Wikipedia, questions and answers, or the folder's name."""
    sr = src(n, stage)
    return {"wiki": t("Wikipedia"), "qa": t("questions and answers")}.get(sr.get("kind")) or (pathlib.Path(sr["path"]).name if sr.get("path") else "")


def version_label(n, v):
    """pre-training, questions and answers · step 2852, fine-tuning on testi-mattapedia · step 1150."""
    st = sc.job.stage_of(v)
    if st == 1:
        return t("pre-training")
    k = int(v.rsplit("-", 1)[1])
    if src(n, st).get("kind") == "qa":
        return t("questions and answers · step {n}", n=k)
    return t("fine-tuning on {what} · step {n}", what=stage_texts(n, st) or "?", n=k)


def version_choices(n):
    return [(version_label(n, v), v) for _, v in sc.versions(n)] if n else []


def versions_dd(n, value=None):
    vs = version_choices(n)
    vals = [v for _, v in vs]
    return gr.Dropdown(choices=vs, value=value if value in vals else (vals[-1] if vals else None))


def samples_html(samples):
    if not samples:
        return ""
    rows = "".join(f"<div class='ftsample'><b>{t('step')} {i}</b> {html.escape(text.strip())}</div>" for i, text in reversed(samples[-5:]))
    return "<div class='ftsamples'><div class='ftchart-head'>" + t("What it writes, newest first") + "</div>" + rows + "</div>"


REFRESH = []  # (function, outputs) to run each time the page is opened: the lists of models change while Joshua runs


def picker_state(choices):
    has = bool(choices)
    return gr.Radio(value="open" if has else "new", visible=has), gr.Row(visible=has), gr.Dropdown(choices=choices), gr.Row(visible=not has)


def pick_folder(current):
    """The Mac's own folder window, opened by Joshua (it runs on this Mac), so you don't have to type the path.
    'activate' brings the window to the front; if you cancel, the field stays as it was."""
    import pathlib, subprocess
    where = f" default location (POSIX file {json.dumps(current.strip())})" if current and pathlib.Path(current.strip()).expanduser().is_dir() else ""
    script = f"POSIX path of (choose folder with prompt {json.dumps(t('Choose the folder with the texts'))}{where})"
    try:
        r = subprocess.run(["osascript", "-e", "activate", "-e", script], capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired):
        return gr.skip()
    chosen = r.stdout.strip()
    return chosen.rstrip("/") if r.returncode == 0 and chosen else gr.skip()


def project_picker(label, choices, placeholder, elem=None):
    """Either open one you've made, or make a new one: one choice at a time, so it's clear which you're doing."""
    has = bool(choices)
    pick = gr.Radio([(t("Open one I've made"), "open"), (t("Make a new one"), "new")], value="open" if has else "new", show_label=False, elem_classes="ftpick", visible=has)  # none yet: just the name field
    with gr.Row(visible=has, equal_height=True, elem_id="ftproject") as openrow:
        name = gr.Dropdown(choices=choices, value=None, label=label, scale=3, elem_id=elem)
    with gr.Row(visible=not has, equal_height=True) as newrow:
        newname = gr.Textbox(show_label=False, placeholder=placeholder, max_lines=1, scale=3, container=False)
        create = gr.Button(t("Create"), size="sm", scale=0, min_width=110)
    pick.change(lambda p: (gr.Row(visible=p == "open"), gr.Row(visible=p == "new")), pick, [openrow, newrow], show_progress="hidden")
    return pick, openrow, name, newrow, newname, create


MAC_OK = ft.MAC and __import__("platform").machine() == "arm64"  # MLX runs on the Apple chip's graphics


def build_new():
    """The page part for a model trained from zero (scratch.py)."""
    import pathlib
    with gr.Column(visible=False) as col:
        gr.Markdown(t("A small language model of your own, starting from random numbers and trained only on texts you choose. It learns to continue text in their style."), elem_classes="ftlead")
        more(NEW_INTRO, t("How it works, and what it can't do"))
        if not MAC_OK:
            gr.HTML("<div class='ftempty'>" + t("This needs a Mac with Apple silicon (M1 or later): on other computers, without a big NVIDIA card, it would take weeks.") + "</div>")
            return col
        pick, openrow, name, newrow, newname, create = project_picker(t("Model"), sc.projects(), t("a short name in lowercase, like my-novels"))
        REFRESH.append((lambda: picker_state(sc.projects()), [pick, openrow, name, newrow]))

        def source_picker(stage):
            """Where a stage's texts come from: a folder on this computer, Wikipedia in a language, or (for a later
            stage) questions and answers made from Wikipedia."""
            kinds = [(t("A folder on this computer"), "folder"), (t("Wikipedia"), "wiki")] + ([(t("Questions and answers from Wikipedia"), "qa")] if stage > 1 else [])
            kind = gr.Radio(kinds, value="folder" if stage > 1 else "wiki", show_label=False, elem_classes="ftsrc")
            with gr.Row(equal_height=True, visible=stage > 1) as frow:
                path = gr.Textbox(show_label=False, placeholder=t("the folder's path, like /Users/you/Documents/my-texts"), max_lines=1, scale=4, container=False)
                browse = gr.Button(t("Browse…"), size="sm", scale=0, min_width=120)
            browse.click(pick_folder, path, path, show_progress="hidden")
            with gr.Row(equal_height=True, visible=stage == 1) as wrow:
                lang = gr.Dropdown(WIKI_LANGS, value="it", label=t("Language"), allow_custom_value=True, scale=2)
                millions = gr.Number(150, label=t("Million words"), info=t("The 10M model wants about 150, the 25M about 350, the 50M about 750."), minimum=1, maximum=3000, precision=0, scale=2)
                whole = gr.Checkbox(False, label=t("All of it"), info=t("The whole Wikipedia in that language: Italian is about 470 million words, 1.8 GB to download"), scale=1)
            with gr.Row(equal_height=True, visible=False) as qrow:
                qcount = gr.Number(50000, label=t("How many questions"), info=t("Each is 'Che cos'è …?' with the first paragraph of the article as the answer, in the language chosen above. A few tens of thousands are enough to learn the format."), minimum=100, maximum=1000000, precision=0)
            kind.change(lambda k: (gr.Row(visible=k == "folder"), gr.Row(visible=k in ("wiki", "qa")), gr.Number(visible=k == "wiki"), gr.Checkbox(visible=k == "wiki"), gr.Row(visible=k == "qa")),
                        kind, [frow, wrow, millions, whole, qrow], show_progress="hidden")
            whole.change(lambda w: gr.Number(interactive=not w), whole, millions, show_progress="hidden")
            return kind, path, lang, millions, whole, qcount

        with gr.Column(visible=False) as work:
            with gr.Column(elem_classes="ftpane"):
                gr.Markdown("**1 · " + t("Download") + "** " + t("Where the model learns its language and its knowledge from: Wikipedia, downloaded a part at a time, or a folder of your texts, every file inside it and its subfolders."))
                kind1, path1, lang1, mil1, whole1, _q1 = source_picker(1)
                with gr.Row():
                    getb = gr.Button("⬇ " + t("Get the texts"), size="sm", scale=0, min_width=170, elem_classes="ftmain")
                    stop1b = gr.Button("■ " + t("Stop"), size="sm", scale=0, min_width=100)
                dl = gr.HTML(download_html(None))
            with gr.Column(elem_classes="ftpane"):
                gr.Markdown("**2 · " + t("Tokenization") + "** " + t("The texts from step 1 are extracted into one plain file, a vocabulary of 8,192 pieces is learned from them, and every paragraph becomes a list of numbers: the form the model learns from."))
                with gr.Row():
                    readb = gr.Button(t("Tokenize"), size="sm", scale=0, min_width=130, elem_classes="ftmain")
                info = gr.HTML(new_info(None))
            with gr.Column(elem_classes="ftpane"):
                gr.Markdown("**3 · " + t("Pre-training") + "** " + t("The model learns to continue those texts, starting from random numbers. Bigger models learn more, but need more text and more time. You can stop and carry on later."))
                size = gr.Radio(size_choices(None), value=sc.SIZES[0], show_label=False, elem_id="ftsize")
                arch = gr.HTML(arch_html(sc.SIZES[0]))
                with gr.Row():
                    trainb = gr.Button("▶ " + t("Start training"), size="sm", scale=0, min_width=170, elem_classes="ftmain")
                    stopb = gr.Button("■ " + t("Stop"), size="sm", scale=0, min_width=100)
                live1 = gr.HTML("")
                plot = gr.HTML(loss_svg([], []))
                samples = gr.HTML("")
            with gr.Column(elem_classes="ftpane"):
                gr.Markdown("**4 · " + t("Questions and answers") + "** " + t("Optional, and it can be skipped. Questions made from Wikipedia ('Che cos'è …?', answered with the first paragraph of the article): the pre-trained model learns to answer instead of only continuing text."))
                with gr.Row(equal_height=True):
                    lang3 = gr.Dropdown(WIKI_LANGS, value="it", label=t("Language"), allow_custom_value=True, scale=2)
                    q3 = gr.Number(50000, label=t("How many questions"), info=t("A few tens of thousands are enough to learn the format."), minimum=100, maximum=1000000, precision=0, scale=2)
                mix3 = gr.Slider(0, 50, value=20, step=5, label=t("Share of each batch still from the pre-training texts (%)"), info=t("More keeps more of what it knew; less takes on the new texts faster."))
                with gr.Row():
                    read3b = gr.Button(t("Read them"), size="sm", scale=0, min_width=130, elem_classes="ftmain")
                    train3b = gr.Button("▶ " + t("Start training"), size="sm", scale=0, min_width=170, elem_classes="ftmain")
                    stop3b = gr.Button("■ " + t("Stop"), size="sm", scale=0, min_width=100)
                info3 = gr.HTML(step_info(None, sc.SIZES[0], QA))
                live3 = gr.HTML("")
                plot3 = gr.HTML(loss_svg([], []))
                samples3 = gr.HTML("")
            with gr.Column(elem_classes="ftpane"):
                gr.Markdown("**5 · " + t("Fine-tuning") + "** " + t("Optional, and it can be skipped. The model goes on learning from a folder of texts, the style of an author say, keeping what it knows. It starts from step 4, or from step 3 if you skipped it."))
                more(STAGE2, t("How it works"))
                with gr.Row(equal_height=True):
                    path4 = gr.Textbox(show_label=False, placeholder=t("the folder's path, like /Users/you/Documents/my-texts"), max_lines=1, scale=4, container=False)
                    browse4 = gr.Button(t("Browse…"), size="sm", scale=0, min_width=120)
                browse4.click(pick_folder, path4, path4, show_progress="hidden")
                mix4 = gr.Slider(0, 50, value=20, step=5, label=t("Share of each batch still from the previous step's texts (%)"), info=t("More keeps more of what it knew; less takes on the new texts faster."))
                with gr.Row():
                    read4b = gr.Button(t("Read them"), size="sm", scale=0, min_width=130, elem_classes="ftmain")
                    train4b = gr.Button("▶ " + t("Start training"), size="sm", scale=0, min_width=170, elem_classes="ftmain")
                    stop4b = gr.Button("■ " + t("Stop"), size="sm", scale=0, min_width=100)
                info4 = gr.HTML(step_info(None, sc.SIZES[0], TUNE))
                live4 = gr.HTML("")
                plot4 = gr.HTML(loss_svg([], []))
                samples4 = gr.HTML("")
            with gr.Column(elem_classes="ftpane"):
                gr.Markdown("**6 · " + t("Try it and add it to Joshua") + "** " + t("Choose a version, then write the beginning of a sentence for it to continue, or ask it a question."))
                which = gr.Dropdown(choices=[], value=None, label=t("Version"))
                askmode = gr.Radio([(t("Continue a text"), "text"), (t("Ask a question"), "question")], value="text", show_label=False, elem_classes="ftsrc",
                                   info=t("A question works once the model has done step 4."))
                with gr.Row(equal_height=True):
                    prompt = gr.Textbox(show_label=False, placeholder=t("The beginning of a sentence"), scale=4, container=False)
                    contb = gr.Button(t("Continue"), size="sm", scale=0, min_width=120, elem_classes="ftmain")
                out = gr.Markdown("", elem_classes="ftans", container=True)
                gr.Markdown(t("The chosen version goes into Joshua's model menu with this name, and into Ollama for other programs. In the chat it gets only your words: no web search, no tools, no memory."))
                with gr.Row(equal_height=True):
                    asname = gr.Textbox(show_label=False, placeholder=t("its name in the menu, like Wikipedia-Italian"), max_lines=1, scale=3, container=False)
                    instb = gr.Button("⬇ " + t("Add to Joshua"), size="sm", scale=0, min_width=170, elem_classes="ftmain")
                    delb = gr.Button(t("Delete this model"), size="sm", scale=0, min_width=170, elem_id="ftdel")
                with gr.Row(equal_height=True):
                    added = gr.Dropdown(choices=[], value=None, label=t("Already in Joshua"), scale=2)
                    newas = gr.Textbox(label=t("New name"), max_lines=1, scale=2)
                    renb = gr.Button(t("Rename"), size="sm", scale=0, min_width=120)
        seen = gr.State(None)
        armed = gr.State(None)

    def saved_source(n, stage):
        f = sc.pdir(n) / f"source{stage}.json"
        if f.exists():
            return json.loads(f.read_text())
        old = sc.pdir(n) / "folder.txt"  # models made before Wikipedia: the folder they were read from
        return {"kind": "folder", "path": old.read_text()} if stage == 1 and old.exists() else {"kind": "wiki" if stage == 1 else "folder"}

    def pane(n, st):
        """A step's chart and samples, if what is on disk for it is its own."""
        pr = (sc.progress(n, st) or {}) if valid(n, st) or (sc.pdir(n) / f"stage{st}").exists() and src(n, st).get("kind") == ("qa" if st == QA else "folder") else {}
        return loss_svg(pr.get("train", []), pr.get("val", []), pr.get("old", [])), samples_html(pr.get("samples", [])), pr

    N_OUT = 32

    def opened(n):
        n = (n or "").strip().lower()
        if not ft.NAME_RE.match(n):
            gr.Warning(t("Use 2–31 lowercase letters, digits or dashes, like 'newsroom-style'."))
            return (gr.skip(),) * N_OUT
        sc.pdir(n).mkdir(exist_ok=True)
        p = sc.progress(n) or {}
        chosen = p.get("size") or (sc.best_size(n) if sc.data(n) else sc.SIZES[0])
        s1, s3, s4 = saved_source(n, 1), src(n, QA), src(n, TUNE)
        plot3_, samples3_, p3 = pane(n, QA)
        plot4_, samples4_, p4 = pane(n, TUNE)
        return (gr.Dropdown(choices=sc.projects(), value=n), gr.Row(visible=False), gr.Column(visible=True),
                s1["kind"], gr.update(value=s1.get("path", "")), gr.update(value=s1.get("lang", "it")), gr.update(value=s1.get("millions", 150), interactive=not s1.get("whole")), gr.update(value=bool(s1.get("whole"))),
                download_html(n), new_info(n), gr.Radio(choices=size_choices(n), value=chosen), loss_svg(p.get("train", []), p.get("val", [])), samples_html(p.get("samples", [])), "", arch_html(chosen, n),
                gr.update(value=s3.get("lang") or (s1.get("lang", "it") if s1.get("kind") == "wiki" else "it")), gr.update(value=s3.get("qcount", 50000)), gr.update(value=round(p3.get("mix", 0.2) * 100)),
                step_info(n, chosen, QA), plot3_, samples3_,
                gr.update(value=s4.get("path", "") if s4.get("kind") == "folder" else ""), gr.update(value=round(p4.get("mix", 0.2) * 100)),
                step_info(n, chosen, TUNE), plot4_, samples4_,
                versions_dd(n), gr.Radio(value="open", visible=True), gr.Row(visible=True), "", "", "")

    def need(n):
        if not n or not sc.pdir(n).exists():
            gr.Warning(t("First open a model, or start a new one."))
            return False
        return True

    def read(n, kind, path, lang, millions, whole, stage, qcount=50000):
        if not need(n):
            return
        if kind == "folder" and not (path or "").strip() or kind == "folder" and not pathlib.Path(path.strip()).expanduser().is_dir():
            gr.Warning(t("That folder doesn't exist: copy its path from Finder (right-click it, hold ⌥ Option, Copy as Pathname)."))
            return
        if stage > 1 and not sc.data(n):
            gr.Warning(t("First do steps 1 and 2: the other steps use their vocabulary."))
            return
        if stage == 1 and sc.trained(n):
            gr.Info(t("Reading new texts starts the model over: the old one is deleted."))
        (sc.pdir(n) / f"source{stage}.json").write_text(json.dumps({"kind": kind, "path": path or "", "lang": lang or "it", "millions": int(millions or 150), "whole": bool(whole), "qcount": int(qcount or 50000)}))
        try:
            sc.prepare(n, source_of(kind, path, lang, int(millions or 150), whole, qcount), stage)
        except RuntimeError as e:
            gr.Warning(str(e))

    def start(n, s):
        if not need(n):
            return
        if not sc.data(n):
            gr.Warning(t("First do steps 1 and 2."))
            return
        if ft.ready():
            gr.Warning(" ".join(ft.ready()))
            return
        try:
            sc.train(n, s)
            gr.Info(t("Training. It writes a sample every so often: watch it learn below."))
        except RuntimeError as e:
            gr.Warning(str(e))

    def start_step(n, s, m, step):
        if not need(n):
            return
        if not sc.trained(n):
            gr.Warning(t("First do step 3, the pre-training."))
            return
        st = step
        if not valid(n, st):
            gr.Warning(t("First press Read them, above."))
            return
        try:
            sc.train(n, (sc.progress(n) or {}).get("size", s), st, round(m) / 100, "model" if st == QA else tune_base(n))
            gr.Info(t("Started. A copy of the model is kept at 8 points along the way: you can try each one in step 6."))
        except RuntimeError as e:
            gr.Warning(str(e))

    def replaces(n, step, what, armed_v):
        """Doing step 4 or 5 again replaces what it made (and, for step 4, step 5 built on it): that takes a second click.
        Carrying on with a training you stopped doesn't. Returns (go ahead?, new armed value)."""
        st = step
        pr = (sc.progress(n, st) or {}) if n else {}
        exists = bool(n) and (sc.pdir(n) / f"stage{st}").exists() and (what == "read" or pr.get("step", 0) >= pr.get("steps", 1))
        if not exists or armed_v == (n, step, what):
            return True, None
        later = st == QA and (sc.pdir(n) / f"stage{TUNE}").exists()
        gr.Warning(t("There is already a model trained here ({what}). Doing this again replaces it, and step 5 built on it. Click again to go ahead.", what=stage_texts(n, st) or "?") if later
                   else t("There is already a model trained here ({what}). Doing this again replaces it. Click again to go ahead.", what=stage_texts(n, st) or "?"))
        return False, (n, step, what)

    def read3(n, lang, q, armed_v):
        ok, armed_v = replaces(n, QA, "read", armed_v)
        if ok:
            read(n, "qa", "", lang, 150, False, QA, q)
        return armed_v

    def read4(n, path, armed_v):
        ok, armed_v = replaces(n, TUNE, "read", armed_v)
        if ok:
            read(n, "folder", path, "it", 150, False, TUNE)
        return armed_v

    def train3(n, s, m, armed_v):
        ok, armed_v = replaces(n, QA, "train", armed_v)
        if ok:
            start_step(n, s, m, QA)
        return armed_v

    def train4(n, s, m, armed_v):
        ok, armed_v = replaces(n, TUNE, "train", armed_v)
        if ok:
            start_step(n, s, m, TUNE)
        return armed_v

    def cont(n, text, v, mode):
        if not need(n):
            return
        if not sc.trained(n):
            gr.Warning(t("First do step 3, the pre-training."))
            return
        if mode == "question" and not text.strip():
            gr.Warning(t("Write a question."))
            return
        try:
            sc.try_out(n, text.strip() or "\n", v or "model", mode == "question")
        except RuntimeError as e:
            gr.Warning(str(e))

    def inst(n, v, as_name):
        if not need(n):
            return
        if not sc.trained(n):
            gr.Warning(t("First do step 3, the pre-training."))
            return
        try:
            sc.install(n, v or "model", (as_name or "").strip() or n)
        except RuntimeError as e:
            gr.Warning(str(e))

    def added_dd(n, value=None):
        """The versions of this model already added to Joshua, by their name in the menu."""
        mine = [k for k, v in sc.in_joshua().items() if v.get("model") == n] if n else []
        return gr.Dropdown(choices=mine, value=value if value in mine else (mine[0] if mine else None))

    def rename(n, old, new):
        if not old:
            gr.Warning(t("Choose the model to rename."))
            return gr.skip(), gr.skip()
        if ft.busy():
            gr.Warning(t("Wait for the running step to finish."))
            return gr.skip(), gr.skip()
        try:
            sc.rename(old, new)
        except Exception as e:
            gr.Warning(str(e))
            return gr.skip(), gr.skip()
        gr.Info(t("{old} is now called {new}: you'll find it with this name in the model menu.", old=old, new=new.strip()))
        return added_dd(n, new.strip()), ""

    def delete(n):
        if not n or not sc.pdir(n).exists():
            return gr.skip(), gr.skip()
        if ft.busy() and ft.RUN.name == n:
            gr.Warning(t("Stop the running step first."))
            return gr.skip(), gr.skip()
        sc.delete(n)
        return gr.Dropdown(choices=sc.projects(), value=None), gr.Column(visible=False)

    KINDS = ("downloading", "reading", "reading2", "pretraining", "pretraining2", "continue")
    TICK = ["dl", "info", "size", "plot", "samples", "out", "info3", "plot3", "samples3", "info4", "plot4", "samples4", "which", "live1", "live3", "live4"]

    def tick(seen_v, s):
        run = ft.RUN
        o = {k: gr.skip() for k in TICK}
        if run is None or run.kind not in KINDS or (id(run), run.version) == seen_v:
            # the training lines show time passing, so they refresh every few seconds even without news
            if run is not None and run.kind in ("pretraining", "pretraining2") and not run.done and int(__import__("time").time()) % 5 == 0:
                o["live1" if run.kind == "pretraining" else "live3" if (run.result or {}).get("stage") == QA else "live4"] = live_html(run)
            return (*o.values(), seen_v)
        sig, n, res = (id(run), run.version), run.name, run.result or {}
        size_now = (sc.progress(n) or {}).get("size", s)
        qa = res.get("stage") == QA  # step 4 (questions and answers), or else step 5 (fine-tuning)
        failed = "<div class='ftres bad'>" + html.escape(t("It stopped: see the console below for what went wrong.")) + "</div>"
        if run.kind == "downloading":
            o["dl"] = download_html(n, res) + (failed if run.done and not run.ok else "")
            if run.done:  # new texts: everything after starts over
                o.update(info=new_info(n), plot=loss_svg([], []), samples="", which=versions_dd(n), info3=step_info(n, s, QA), plot3=loss_svg([], []), samples3="",
                         info4=step_info(n, s, TUNE), plot4=loss_svg([], []), samples4="")
        elif run.kind == "reading":
            box = phases_html(res)  # each phase, with its files, under the button that started it
            if run.done and run.ok:
                box = new_info(n) + box
                o.update(size=gr.Radio(choices=size_choices(n), value=sc.best_size(n) if sc.data(n) else sc.SIZES[0]), plot=loss_svg([], []), samples="", which=versions_dd(n),
                         info3=step_info(n, s, QA), plot3=loss_svg([], []), samples3="", info4=step_info(n, s, TUNE), plot4=loss_svg([], []), samples4="")
            elif run.done:
                box += failed
            o["info"] = box
        elif run.kind == "reading2":
            box = phases_html(res)
            if run.done and run.ok:
                box = step_info(n, size_now, QA if qa else TUNE) + box
                o["which"] = versions_dd(n)
                o.update(plot3=loss_svg([], []), samples3="", info4=step_info(n, size_now, TUNE), plot4=loss_svg([], []), samples4="") if qa else o.update(plot4=loss_svg([], []), samples4="")
            elif run.done:
                box += failed
            o["info3" if qa else "info4"] = box
        elif run.kind == "pretraining":
            o["plot"] = loss_svg(run.train, run.val) if run.train else gr.skip()
            o["samples"] = samples_html(res.get("samples", []))
            o["which"] = versions_dd(n) if run.done or res.get("samples") else gr.skip()
            o["live1"] = live_html(run)
            if run.done:
                o.update(info3=step_info(n, size_now, QA), info4=step_info(n, size_now, TUNE))
        elif run.kind == "pretraining2":
            o["plot3" if qa else "plot4"] = loss_svg(run.train, run.val, res.get("old", [])) if run.train or res.get("old") else gr.skip()
            o["samples3" if qa else "samples4"] = samples_html(res.get("samples", []))
            o["which"] = versions_dd(n)
            o["live3" if qa else "live4"] = live_html(run)
            if qa and run.done:
                o["info4"] = step_info(n, size_now, TUNE)  # step 5 now starts from step 4's last copy
        elif run.kind == "continue" and res.get("text"):
            o["out"] = html.escape(res["text"])
        return (*o.values(), sig)

    def get(n, kind, path, lang, millions, whole):
        if not need(n):
            return
        if kind == "folder" and not ((path or "").strip() and pathlib.Path(path.strip()).expanduser().is_dir()):
            gr.Warning(t("That folder doesn't exist: copy its path from Finder (right-click it, hold ⌥ Option, Copy as Pathname)."))
            return
        if sc.trained(n):
            gr.Info(t("New texts start the model over: the old one is deleted."))
        (sc.pdir(n) / "source1.json").write_text(json.dumps({"kind": kind, "path": path or "", "lang": lang or "it", "millions": int(millions or 150), "whole": bool(whole)}))
        try:
            sc.download(n, source_of(kind, path, lang, int(millions or 150), whole))
        except RuntimeError as e:
            gr.Warning(str(e))

    def tokenize(n):
        if not need(n):
            return
        got = sc.downloaded(n)
        if not got:
            gr.Warning(t("First do step 1."))
            return
        try:
            sc.prepare(n, got["source"], 1)
        except RuntimeError as e:
            gr.Warning(str(e))

    name.focus(lambda: gr.Dropdown(choices=sc.projects()), None, name, show_progress="hidden")  # models made since the page loaded
    outs = [name, newrow, work, kind1, path1, lang1, mil1, whole1, dl, info, size, plot, samples, out, arch,
            lang3, q3, mix3, info3, plot3, samples3, path4, mix4, info4, plot4, samples4,
            which, pick, openrow, live1, live3, live4]
    assert len(outs) == N_OUT
    size.change(lambda s, n: (arch_html(s, n), step_info(n, s, QA), step_info(n, s, TUNE)), [size, name], [arch, info3, info4], show_progress="hidden")
    name.input(opened, name, outs)
    create.click(opened, newname, outs)
    newname.submit(opened, newname, outs)
    getb.click(get, [name, kind1, path1, lang1, mil1, whole1])
    readb.click(tokenize, name)
    trainb.click(start, [name, size])
    read3b.click(read3, [name, lang3, q3, armed], armed)
    train3b.click(train3, [name, size, mix3, armed], armed)
    read4b.click(read4, [name, path4, armed], armed)
    train4b.click(train4, [name, size, mix4, armed], armed)
    for b in (stop1b, stopb, stop3b, stop4b):
        b.click(lambda: ft.stop())
    contb.click(cont, [name, prompt, which, askmode])
    prompt.submit(cont, [name, prompt, which, askmode])
    askmode.change(lambda m: gr.Textbox(placeholder=t("A question, like: Che cos'è l'automobile?") if m == "question" else t("The beginning of a sentence")), askmode, prompt, show_progress="hidden")
    instb.click(inst, [name, which, asname])
    added.focus(lambda n, v: added_dd(n, v), [name, added], added, show_progress="hidden")  # models added since the page opened
    renb.click(rename, [name, added, newas], [added, newas])
    name.change(lambda n: added_dd(n), name, added, show_progress="hidden")
    delb.click(delete, name, [name, work], js="(n) => (n && confirm(" + json.dumps(t("Delete this model and its training? A copy added to Ollama stays."), ensure_ascii=False) + ")) ? n : null")
    gr.Timer(1).tick(tick, [seen, size], [dl, info, size, plot, samples, out, info3, plot3, samples3, info4, plot4, samples4, which, live1, live3, live4, seen], show_progress="hidden")
    return col


def build(conv_root, conv_choices):
    with gr.Column(visible=False, scale=5, elem_id="tune") as panel:
        with gr.Row():
            title = gr.HTML("<div id='tunetitle'>" + t("FINE-TUNING") + "</div>")
            back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_id="tuneback", elem_classes="backbtn")
        mode = gr.Radio([(t("Teach an existing model"), "teach"), (t("Train a new model from zero"), "new")], value="teach", show_label=False, elem_id="ftmode", visible=False)  # chosen by the two menu entries
        with gr.Column() as teachcol:
            gr.Markdown(t("Teach a model your style from examples, here on your computer. The result is a new model in Joshua's menu."), elem_classes="ftlead")
            res = gr.HTML()
            more(INTRO, t("What fine-tuning is, and what it can't do"))
            # the project: open one, or start a new one
            pick, openrow, name, newrow, newname, create = project_picker(t("Project"), ft.projects(), t("a short name in lowercase, like newsroom-style"), "ftname")
            REFRESH.append((lambda: picker_state(ft.projects()), [pick, openrow, name, newrow]))
            with gr.Column() as noproj:
                gr.HTML("<div class='ftempty'>" + t("Open a project, or start a new one: each project has its own examples and training.") + "</div>")
            with gr.Column(visible=False) as work:
                steps = gr.Radio([(f"{n} · {t(label)}", n) for n, label in STEPS], value="1", show_label=False, elem_id="ftsteps")
                with gr.Column(elem_classes="ftpane") as s1:
                    gr.Markdown(t("**Choose the model to teach.** ✓ fits this computer, ✗ doesn't (and why). Bigger models learn better but train more slowly."))
                    basepick = gr.Dropdown(base_choices(), value=ft.DEFAULT, show_label=False, elem_id="ftbase")
                    with gr.Row(visible=False, equal_height=True) as hfrow:
                        hfrepo = gr.Textbox(show_label=False, container=False, placeholder=t("its name on Hugging Face, like mlx-community/Llama-3.2-3B-Instruct-4bit") if ft.MAC else t("its name on Hugging Face, like meta-llama/Llama-3.2-3B-Instruct"), scale=4)
                        usehf = gr.Button(t("Use it"), size="sm", scale=0, min_width=100)
                    baseinfo = gr.HTML("")
                    more(STEP1)
                with gr.Column(visible=False, elem_classes="ftpane") as s2:
                    gr.Markdown(t("**Show the model what you want:** something you might say, and the exact answer you'd like back. 100–500 good examples work best."))
                    with gr.Row(equal_height=True, elem_classes="ftadd"):
                        upload = gr.UploadButton("📄 " + t("Add from a file"), file_types=[".jsonl", ".json", ".csv"], type="filepath", size="sm", scale=0, min_width=170)
                        chats = gr.Dropdown(choices=conv_choices(), multiselect=True, label=t("…or pick some of your chats"), scale=3, elem_id="ftchats")
                        addchats = gr.Button(t("Add from these chats"), size="sm", scale=0, min_width=170)
                    with gr.Row(equal_height=True, elem_classes="ftadd"):
                        q = gr.Textbox(show_label=False, placeholder=t("You say…"), lines=1, max_lines=6, scale=2, container=False)
                        a = gr.Textbox(show_label=False, placeholder=t("…and Joshua answers"), lines=1, max_lines=6, scale=3, container=False)
                        add = gr.Button(t("Add"), size="sm", scale=0, min_width=80)
                    grid = gr.Dataframe(table([]), headers=[t("You say"), t("Joshua answers")], column_count=(2, "fixed"), interactive=True, wrap=True, max_height=380,
                                        buttons=[], show_search="search", elem_id="ftgrid")
                    with gr.Row(elem_classes="ftsave"):
                        count = gr.HTML(count_html([]))
                        saveb = gr.Button(t("Save examples"), size="sm", scale=0, min_width=160, elem_classes="ftmain")
                    more(STEP2)
                with gr.Column(visible=False, elem_classes="ftpane") as s3:
                    gr.Markdown(t("**Train it.** Start with a light touch: you can always train again more strongly. It takes 15–60 minutes."))
                    preset = gr.Radio([(t(p), p) for p in PRESETS], value="Normal", show_label=False, elem_id="ftpreset")
                    advanced = gr.Checkbox(False, label=t("Advanced settings"))
                    with gr.Row(visible=False) as advrow:
                        iters = gr.Slider(50, 1500, value=300, step=10, label=t("Steps"))
                        lr = gr.Dropdown(["1e-5", "2e-5", "5e-5", "1e-4", "2e-4"], value="5e-5", label=t("Learning rate"))
                        layers = gr.Dropdown([4, 8, 16, 24], value=16, label=t("Layers"))
                    with gr.Row():
                        trainb = gr.Button("▶ " + t("Start training"), size="sm", scale=0, min_width=170, elem_classes="ftmain")
                        stopb = gr.Button("■ " + t("Stop"), size="sm", scale=0, min_width=100, elem_id="ftstop")
                    plot = gr.HTML(loss_svg([], []))
                    more(STEP3)
                with gr.Column(visible=False, elem_classes="ftpane") as s4:
                    gr.Markdown(t("**Compare.** Ask the model before and after training the same question, one that isn't in your examples."))
                    with gr.Row(equal_height=True):
                        tq = gr.Textbox(show_label=False, placeholder=t("A question that is not in your examples"), scale=4, container=False)
                        tryb = gr.Button(t("Ask both"), size="sm", scale=0, min_width=120, elem_classes="ftmain")
                    with gr.Row():
                        before = gr.Markdown("", label=t("Before"), elem_classes="ftans", container=True)
                        after = gr.Markdown("", label=t("After"), elem_classes="ftans", container=True)
                    more(STEP4)
                with gr.Column(visible=False, elem_classes="ftpane") as s5:
                    gr.Markdown(t("**Install it.** Your trained model becomes `joshua-<name>` in the model menu."))
                    with gr.Row():
                        instb = gr.Button("⬇ " + t("Install into Joshua"), size="sm", scale=0, min_width=190, elem_classes="ftmain")
                        delb = gr.Button(t("Delete this project"), size="sm", scale=0, min_width=170, elem_id="ftdel")
                    more(STEP5)
        newcol = build_new()
        status = gr.HTML("<div class='ftstatus'>" + t("Ready when you are") + "</div>")
        showlog = gr.Checkbox(False, label=t("Show the console"))
        with gr.Column(visible=False) as logbox:
            log = gr.Textbox(show_label=False, lines=10, max_lines=10, autoscroll=True, interactive=False, elem_id="ftlog")
        seen = gr.State(None)
        panes = [s1, s2, s3, s4, s5]

    # ----- handlers -----
    def open_project(n):
        n = (n or "").strip().lower()
        if not ft.NAME_RE.match(n):
            gr.Warning(t("Use 2–31 lowercase letters, digits or dashes, like 'newsroom-style'."))
            return (gr.skip(),) * 21
        rows = ft.load_examples(n) or ft.save_examples(n, [])
        meta = ft.pdir(n) / "training.json"
        df = plot_df(json.loads(meta.read_text())) if meta.exists() else gr.skip()
        b = ft.base(n)
        step = "1" if not rows else "3" if not ft.trained(n) and len(rows) >= 20 else "2" if len(rows) < 20 else "4"  # where you left off
        return (gr.Dropdown(choices=ft.projects(), value=n), table(rows), count_html(rows), gr.Dropdown(choices=conv_choices()), df,
                gr.Dropdown(choices=base_choices(), value=b["id"]), gr.Row(visible=b["id"] == "custom"), b["mac" if ft.MAC else "linux"] if b["id"] == "custom" else "", base_info(b),
                gr.Column(visible=False), gr.Column(visible=True), gr.Radio(value=step), *show_step(step), gr.Row(visible=False), "", gr.Radio(value="open", visible=True), gr.Row(visible=True))

    def choose(n, entry):
        """Saves the model this project teaches. A training made for another model can't be used with this one."""
        old = ft.base(n)
        ft.set_base(n, entry)
        if ft.trained(n) and repo_of(old) != repo_of(entry):
            import shutil
            shutil.rmtree(ft.pdir(n) / "adapters", ignore_errors=True)
            gr.Info(t("The previous training was for {model}: train again for this one (your examples are kept).", model=old["label"]))
        return base_info(entry)

    def pick_base(n, choice):
        if choice == "custom":
            return gr.Row(visible=True), gr.skip()
        entry = next((b for b in ft.bases() if b["id"] == choice), None)
        if not entry:
            return gr.skip(), gr.skip()
        if not n or not (ft.pdir(n) / "examples.json").exists():
            return gr.Row(visible=False), base_info(entry)
        return gr.Row(visible=False), choose(n, entry)

    def use_custom(n, name_on_hf):
        if not need(n):
            return gr.skip()
        try:
            entry = ft.custom(name_on_hf.strip())
        except Exception as e:
            gr.Warning(str(e))
            return gr.skip()
        return choose(n, entry)

    repo_of = lambda b: b["mac"] if ft.MAC else b["linux"]

    def need(n):
        if not n or not (ft.pdir(n) / "examples.json").exists():
            gr.Warning(t("First open a project, or start a new one."))
            return False
        return True

    def from_file(f, df):
        if not f:
            return gr.skip(), gr.skip()
        try:
            new = ft.parse_file(f)
        except Exception as e:
            gr.Warning(t("Couldn't read that file: {error}", error=e))
            return gr.skip(), gr.skip()
        gr.Info(t("Added {n} examples. Check them, then Save.", n=len(new)))
        rows = rows_of(df) + new
        return table(rows), count_html(rows)

    def from_chat_list(cids, df):
        new = ft.from_chats(conv_root, cids or [])
        gr.Info(t("Added {n} examples from {chats} chats. Edit the answers until they're exactly what you want, then Save.", n=len(new), chats=len(cids or [])))
        rows = rows_of(df) + new
        return table(rows), count_html(rows)

    def add_one(qq, aa, df):
        if not qq.strip() or not aa.strip():
            gr.Warning(t("Write both what you say and the answer you want."))
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        rows = rows_of(df) + [[qq, aa]]
        return table(rows), count_html(rows), "", ""

    def save_rows(n, df, p):
        if not need(n):
            return gr.skip(), gr.skip()
        rows = ft.save_examples(n, rows_of(df))
        gr.Info(t("Saved {n} examples.", n=len(rows)))
        return count_html(rows), iters_for(n, p)

    def iters_for(n, p):
        passes = PRESETS[p][0]
        return ft.suggest_iters(max(1, len(ft.load_examples(n)) * 9 // 10), ft.resources()["ram"], passes)

    def pick_preset(n, p):
        _, l, lay = PRESETS[p]
        return (iters_for(n, p) if n and ft.load_examples(n) else gr.skip()), l, lay

    def start_train(n, it, l, lay):
        if not need(n):
            return
        if ft.ready():
            gr.Warning(" ".join(ft.ready()))
            return
        rows = ft.load_examples(n)
        if len(rows) < 20:
            gr.Warning(t("You have {n} saved examples: add at least 20 and press Save.", n=len(rows)))
            return
        tr, va = ft.write_dataset(n, rows)
        try:
            ft.train(n, int(it), l, int(lay), ft.resources()["ram"])
            gr.Info(t("Training on {n} examples, checking on {m}. Follow it in the chart and console.", n=tr, m=va))
        except RuntimeError as e:
            gr.Warning(str(e))

    def start_try(n, question):
        if not need(n):
            return
        if not ft.trained(n):
            gr.Warning(t("Train the model first (step ③)."))
            return
        if not question.strip():
            gr.Warning(t("Write a question."))
            return
        try:
            ft.try_out(n, question)
        except RuntimeError as e:
            gr.Warning(str(e))

    def start_install(n):
        if not need(n):
            return
        if not ft.trained(n):
            gr.Warning(t("Train the model first (step ③)."))
            return
        try:
            ft.install(n)
        except RuntimeError as e:
            gr.Warning(str(e))

    def delete(n):
        if not n or not ft.pdir(n).exists():
            return (gr.skip(),) * 5
        if ft.busy() and ft.RUN.name == n:
            gr.Warning(t("Stop the running step first."))
            return (gr.skip(),) * 5
        ft.delete_project(n)
        gr.Info(t("Deleted the project '{name}'. An installed joshua-{name} model stays in the menu.", name=n))
        return gr.Dropdown(choices=ft.projects(), value=None), table([]), count_html([]), gr.Column(visible=True), gr.Column(visible=False)

    def plot_df(meta):
        return loss_svg(meta.get("train", []), meta.get("val", []))

    def tick(seen_v):
        run = ft.RUN
        if run is None:
            return (gr.skip(),) * 5 + (seen_v,)
        sig = (id(run), run.version)
        if sig == seen_v:
            return (gr.skip(),) * 5 + (seen_v,)
        secs = int(__import__("time").time() - run.started)
        colour = "ok" if run.done and run.ok else "bad" if run.done else "run"
        took = f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}" if secs >= 3600 else f"{secs // 60}:{secs % 60:02d}"
        left = None if run.done else run.left()
        if left is not None:
            m = max(1, round(left / 60))
            took += " · " + (t("about {n} min left", n=m) if m < 60 else t("about {h} h {m} min left", h=m // 60, m=m % 60))
        st = f"<div class='ftstatus {colour}'>{'' if run.done else '<span class=wdot></span>'}{run.kind.upper()} · {html.escape(run.name)} · {html.escape(run.status)} · {took}</div>"
        df = plot_df({"train": run.train, "val": run.val}) if run.kind == "training" else gr.skip()
        b = a_ = gr.skip()
        if run.kind == "test" and run.result:
            b, a_ = run.result.get("BEFORE", "…"), run.result.get("AFTER", "…")
        return st, "\n".join(run.log[-200:]), df, b, a_, sig

    def show_step(v):
        return [gr.Column(visible=str(k + 1) == str(v)) for k in range(5)]

    opened = [name, grid, count, chats, plot, basepick, hfrow, hfrepo, baseinfo, noproj, work, steps, *panes, newrow, newname, pick, openrow]
    name.focus(lambda: gr.Dropdown(choices=ft.projects()), None, name, show_progress="hidden")
    mode.change(lambda m: (gr.Column(visible=m == "teach"), gr.Column(visible=m == "new")), mode, [teachcol, newcol], show_progress="hidden")
    name.input(open_project, name, opened)
    create.click(open_project, newname, opened)
    newname.submit(open_project, newname, opened)
    steps.change(show_step, steps, panes, show_progress="hidden")
    basepick.input(pick_base, [name, basepick], [hfrow, baseinfo])
    usehf.click(use_custom, [name, hfrepo], baseinfo)
    hfrepo.submit(use_custom, [name, hfrepo], baseinfo)
    upload.upload(from_file, [upload, grid], [grid, count])
    addchats.click(from_chat_list, [chats, grid], [grid, count])
    add.click(add_one, [q, a, grid], [grid, count, q, a])
    grid.input(lambda df: count_html(rows_of(df)), grid, count)
    saveb.click(save_rows, [name, grid, preset], [count, iters])
    preset.change(pick_preset, [name, preset], [iters, lr, layers])
    advanced.change(lambda v: gr.Row(visible=v), advanced, advrow, show_progress="hidden")
    showlog.change(lambda v: gr.Column(visible=v), showlog, logbox, show_progress="hidden")
    trainb.click(start_train, [name, iters, lr, layers])
    stopb.click(lambda: ft.stop())
    tryb.click(start_try, [name, tq])
    instb.click(start_install, name)
    delb.click(delete, name, [name, grid, count, noproj, work], js="(n) => (n && confirm(" + json.dumps(t("Delete this project, its examples and its training? An installed model stays."), ensure_ascii=False) + ")) ? n : null")
    gr.Timer(1).tick(tick, seen, [status, log, plot, before, after, seen], show_progress="hidden")
    refresh = (lambda: tuple(x for fn, _ in REFRESH for x in fn()), [o for _, outs in REFRESH for o in outs])
    return panel, back, res, refresh, mode, title
