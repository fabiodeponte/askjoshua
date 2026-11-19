"""The 'Teach Joshua' panel: a step-by-step fine-tuning wizard. Built inside web.py's gr.Blocks."""
import html, json, pathlib
import gradio as gr, pandas as pd
from markdown_it import MarkdownIt
import finetune as ft
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


def loss_svg(train, val):
    """The loss chart, drawn here as a small SVG: training in blue, validation in orange, lower is better."""
    pts = list(train) + list(val)
    head = "<div class='ftchart'><div class='ftchart-head'>" + t("Loss, lower is better") + (f" <span class='k1'>■ {t('training')}</span> <span class='k2'>■ {t('validation')}</span>" if pts else "") + "</div>"
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
    ticks = "".join(f"<text x='{X(v):.1f}' y='{H - 6}' text-anchor='middle'>{int(v)}</text>" for v in (x0 + (x1 - x0) * k / 4 for k in range(5)))
    last = f"{t('step')} {int(xs[-1] if not val else max(xs))} · " + t("training") + f" {train[-1][1]:.2f}" + (f" · {t('validation')} {val[-1][1]:.2f}" if val else "")
    return (head + f"<svg viewBox='0 0 {W} {H}'><g class='grid'>{grid}</g><g class='ticks'>{ticks}</g>"
            + line(train, "tr") + line(val, "va") + f"</svg><div class='ftchart-last'>{html.escape(last)}</div></div>")


def more(text, label=None):
    """The longer explanation of a step, folded away under a line you can open."""
    return gr.HTML("<details class='ftmore'><summary>" + html.escape(label or t("More about this step")) + "</summary>" + MD.render(text) + "</details>")


STEPS = [("1", "Model"), ("2", "Examples"), ("3", "Train"), ("4", "Compare"), ("5", "Install")]


def build(conv_root, conv_choices):
    with gr.Column(visible=False, scale=5, elem_id="tune") as panel:
        with gr.Row():
            gr.HTML("<div id='tunetitle'>" + t("FINE-TUNING") + "</div>")
            back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_id="tuneback", elem_classes="backbtn")
        gr.Markdown(t("Teach a model your style from examples, here on your computer. The result is a new model in Joshua's menu."), elem_classes="ftlead")
        res = gr.HTML()
        more(INTRO, t("What fine-tuning is, and what it can't do"))
        # the project: open one, or start a new one
        with gr.Row(elem_id="ftproject", equal_height=True):
            name = gr.Dropdown(choices=ft.projects(), value=None, label=t("Project"), scale=3, elem_id="ftname")
            newp = gr.Button("＋ " + t("New project"), size="sm", scale=0, min_width=150)
        with gr.Row(visible=False, elem_id="ftnewrow", equal_height=True) as newrow:
            newname = gr.Textbox(show_label=False, placeholder=t("a short name in lowercase, like newsroom-style"), max_lines=1, scale=3, container=False)
            create = gr.Button(t("Create"), size="sm", scale=0, min_width=110)
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
            return (gr.skip(),) * 20
        rows = ft.load_examples(n) or ft.save_examples(n, [])
        meta = ft.pdir(n) / "training.json"
        df = plot_df(json.loads(meta.read_text())) if meta.exists() else gr.skip()
        b = ft.base(n)
        step = "1" if not rows else "3" if not ft.trained(n) and len(rows) >= 20 else "2" if len(rows) < 20 else "4"  # where you left off
        return (gr.Dropdown(choices=ft.projects(), value=n), table(rows), count_html(rows), gr.Dropdown(choices=conv_choices()), df,
                gr.Dropdown(choices=base_choices(), value=b["id"]), gr.Row(visible=b["id"] == "custom"), b["mac" if ft.MAC else "linux"] if b["id"] == "custom" else "", base_info(b),
                gr.Column(visible=False), gr.Column(visible=True), gr.Radio(value=step), *show_step(step), gr.Row(visible=False), "")

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
        st = f"<div class='ftstatus {colour}'>{'' if run.done else '<span class=wdot></span>'}{run.kind.upper()} · {html.escape(run.name)} · {html.escape(run.status)} · {secs // 60}:{secs % 60:02d}</div>"
        df = plot_df({"train": run.train, "val": run.val}) if run.kind == "training" else gr.skip()
        b = a_ = gr.skip()
        if run.kind == "test" and run.result:
            b, a_ = run.result.get("BEFORE", "…"), run.result.get("AFTER", "…")
        return st, "\n".join(run.log[-200:]), df, b, a_, sig

    def show_step(v):
        return [gr.Column(visible=str(k + 1) == str(v)) for k in range(5)]

    opened = [name, grid, count, chats, plot, basepick, hfrow, hfrepo, baseinfo, noproj, work, steps, *panes, newrow, newname]
    name.input(open_project, name, opened)
    newp.click(lambda: (gr.Row(visible=True), ""), outputs=[newrow, newname])
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
    return panel, back, res
