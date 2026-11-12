"""The 'Teach Joshua' panel: a step-by-step fine-tuning wizard. Built inside web.py's gr.Blocks."""
import html, json, pathlib
import gradio as gr, pandas as pd
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

PRESETS = {"Light touch": (2, "2e-5", 8), "Normal": (3, "5e-5", 16), "Strong": (5, "1e-4", 16)}  # shown translated; the keys stay English


def res_html():
    r, missing = ft.resources(), ft.ready()
    fitting = [b for b in ft.bases() if ft.fits(b, r)[0]]
    r["chip"] = r["chip"].replace("·", "<br>" + t("Graphics") + " ·") if not ft.MAC else r["chip"]
    rows = [t("Chip") + f" · {html.escape(r['chip'])}", t("Memory (RAM)") + f" · {r['ram']} GB", t("Free disk") + f" · {r['free']} GB",
            (("✓ " + t("{n} of {total} models fit; the biggest: {name}", n=len(fitting), total=len(ft.bases()), name=html.escape(fitting[-1]["label"])))
             if fitting else "✗ " + t("No model fits: free some disk, or use a computer with more memory."))]
    rows += [f"✗ {html.escape(m)}" for m in missing]
    verdict = t("READY TO TRAIN") if fitting and not missing else t("NOT READY YET")
    return f"<div class='ftres'><b>{verdict}</b><br>" + "<br>".join(rows) + "</div>"


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


def build(conv_root, conv_choices):
    with gr.Column(visible=False, scale=5, elem_id="tune") as panel:
        with gr.Row():
            gr.HTML("<div id='tunetitle'>" + t("TEACH JOSHUA · FINE-TUNING") + "</div>")
            back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_id="tuneback")
        with gr.Accordion("⓪  " + t("What this is, and whether your computer can do it"), open=True, elem_classes="ftstep"):
            gr.Markdown(INTRO)
            res = gr.HTML()
        with gr.Accordion("①  " + t("Name it"), open=True, elem_classes="ftstep"):
            gr.Markdown(STEP1)
            with gr.Row():
                name = gr.Dropdown(choices=ft.projects(), value=None, allow_custom_value=True, label=t("Project name"), scale=3)
                openp = gr.Button(t("Create / open"), scale=1)
            basepick = gr.Dropdown(base_choices(), value=ft.DEFAULT, label=t("Model to teach"), elem_id="ftbase", info=t("✓ fits this computer · ✗ doesn't, and why. Bigger learns better but trains more slowly."))
            with gr.Row(visible=False) as hfrow:
                hfrepo = gr.Textbox(show_label=False, placeholder=t("its name on Hugging Face, like mlx-community/Llama-3.2-3B-Instruct-4bit") if ft.MAC else t("its name on Hugging Face, like meta-llama/Llama-3.2-3B-Instruct"), scale=4)
                usehf = gr.Button(t("Use it"), scale=1)
            baseinfo = gr.HTML("")
        with gr.Accordion("②  " + t("Give examples"), open=False, elem_classes="ftstep") as acc2:
            gr.Markdown(STEP2)
            with gr.Row():
                upload = gr.File(file_types=[".jsonl", ".json", ".csv"], label=t("Upload a file"), height=110, scale=1)
                with gr.Column(scale=2):
                    chats = gr.Dropdown(choices=conv_choices(), multiselect=True, label=t("From your chats"))
                    addchats = gr.Button(t("Add from these chats"), size="sm")
            with gr.Row():
                q = gr.Textbox(label=t("You say"), lines=2, scale=2)
                a = gr.Textbox(label=t("Joshua answers"), lines=2, scale=3)
                add = gr.Button(t("Add"), scale=0, min_width=80)
            grid = gr.Dataframe(table([]), headers=[t("You say"), t("Joshua answers")], column_count=(2, "fixed"), interactive=True, wrap=True, max_height=420, elem_id="ftgrid")
            with gr.Row():
                count = gr.HTML(count_html([]))
                saveb = gr.Button(t("Save examples"), variant="primary", scale=0, min_width=180)
        with gr.Accordion("③  " + t("Train"), open=False, elem_classes="ftstep") as acc3:
            gr.Markdown(STEP3)
            preset = gr.Radio([(t(p), p) for p in PRESETS], value="Normal", label=t("Preset"))
            with gr.Row():
                iters = gr.Slider(50, 1500, value=300, step=10, label=t("Steps"))
                lr = gr.Dropdown(["1e-5", "2e-5", "5e-5", "1e-4", "2e-4"], value="5e-5", label=t("Learning rate"))
                layers = gr.Dropdown([4, 8, 16, 24], value=16, label=t("Layers"))
            with gr.Row():
                trainb = gr.Button("▶ " + t("Start training"), variant="primary")
                stopb = gr.Button("■ " + t("Stop"), elem_id="ftstop")
            plot = gr.LinePlot(pd.DataFrame({"step": [], "loss": [], "set": []}), x="step", y="loss", color="set", height=300, label=t("Loss, lower is better · blue = training · orange = validation"), color_map={t("training"): "#4a7fb5", t("validation"): "#e8892a"})
        with gr.Accordion("④  " + t("Compare before and after"), open=False, elem_classes="ftstep") as acc4:
            gr.Markdown(STEP4)
            with gr.Row():
                tq = gr.Textbox(label=t("A question that is not in your examples"), scale=4)
                tryb = gr.Button(t("Ask both"), scale=1)
            with gr.Row():
                before = gr.Markdown("", label=t("Before"), elem_classes="ftans", container=True)
                after = gr.Markdown("", label=t("After"), elem_classes="ftans", container=True)
        with gr.Accordion("⑤  " + t("Install it into Joshua"), open=False, elem_classes="ftstep") as acc5:
            gr.Markdown(STEP5)
            with gr.Row():
                instb = gr.Button("⬇ " + t("Install into Joshua"), variant="primary")
                delb = gr.Button("🗑 " + t("Delete this project"), elem_id="ftdel")
        status = gr.HTML("<div class='ftstatus'>" + t("Ready when you are") + "</div>")
        log = gr.Textbox(label=t("Console"), lines=10, max_lines=10, autoscroll=True, interactive=False, elem_id="ftlog")
        seen = gr.State(None)

    # ----- handlers -----
    def open_project(n):
        n = (n or "").strip().lower()
        if not ft.NAME_RE.match(n):
            gr.Warning(t("Use 2–31 lowercase letters, digits or dashes, like 'newsroom-style'."))
            return (gr.skip(),) * 11
        rows = ft.load_examples(n) or ft.save_examples(n, [])
        meta = ft.pdir(n) / "training.json"
        df = plot_df(json.loads(meta.read_text())) if meta.exists() else gr.skip()
        b = ft.base(n)
        return (gr.Dropdown(choices=ft.projects(), value=n), table(rows), count_html(rows), gr.Accordion(open=True), gr.Accordion(open=bool(rows)), gr.Dropdown(choices=conv_choices()), df,
                gr.Dropdown(choices=base_choices(), value=b["id"]), gr.Row(visible=b["id"] == "custom"), b["mac" if ft.MAC else "linux"] if b["id"] == "custom" else "", base_info(b))

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
            gr.Warning(t("First create or open a project in step ①."))
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
            return gr.skip(), gr.skip(), gr.skip()
        rows = ft.save_examples(n, rows_of(df))
        gr.Info(t("Saved {n} examples.", n=len(rows)))
        return count_html(rows), iters_for(n, p), gr.Accordion(open=len(rows) >= 20)

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
            return gr.skip(), gr.skip(), gr.skip()
        if ft.busy() and ft.RUN.name == n:
            gr.Warning(t("Stop the running step first."))
            return gr.skip(), gr.skip(), gr.skip()
        ft.delete_project(n)
        gr.Info(t("Deleted the project '{name}'. An installed joshua-{name} model stays in the menu.", name=n))
        return gr.Dropdown(choices=ft.projects(), value=None), table([]), count_html([])

    def plot_df(meta):
        pts = [(s, v, t("training")) for s, v in meta.get("train", [])] + [(s, v, t("validation")) for s, v in meta.get("val", [])]
        return pd.DataFrame(pts or [], columns=["step", "loss", "set"])

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

    opened = [name, grid, count, acc2, acc3, chats, plot, basepick, hfrow, hfrepo, baseinfo]
    openp.click(open_project, name, opened)
    name.select(open_project, name, opened)
    basepick.input(pick_base, [name, basepick], [hfrow, baseinfo])
    usehf.click(use_custom, [name, hfrepo], baseinfo)
    hfrepo.submit(use_custom, [name, hfrepo], baseinfo)
    upload.upload(from_file, [upload, grid], [grid, count])
    addchats.click(from_chat_list, [chats, grid], [grid, count])
    add.click(add_one, [q, a, grid], [grid, count, q, a])
    grid.input(lambda df: count_html(rows_of(df)), grid, count)
    saveb.click(save_rows, [name, grid, preset], [count, iters, acc3])
    preset.change(pick_preset, [name, preset], [iters, lr, layers])
    trainb.click(start_train, [name, iters, lr, layers])
    stopb.click(lambda: ft.stop())
    tryb.click(start_try, [name, tq])
    instb.click(start_install, name)
    delb.click(delete, name, [name, grid, count], js="(n) => (n && confirm(" + json.dumps(t("Delete this project, its examples and its training? An installed model stays."), ensure_ascii=False) + ")) ? n : null")
    gr.Timer(1).tick(tick, seen, [status, log, plot, before, after, seen], show_progress="hidden")
    return panel, back, res
