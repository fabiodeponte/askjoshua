"""The 'Teach Joshua' panel: a step-by-step fine-tuning wizard. Built inside web.py's gr.Blocks."""
import html, json, pathlib
import gradio as gr, pandas as pd
import finetune as ft

INTRO = """
### What fine-tuning is
Joshua's brain is a model: billions of numbers learned from a huge amount of text. **Fine-tuning** nudges a few of those numbers
using *your* examples, so the model picks up a habit: a tone of voice, a format, a way of answering, your jargon.

**It teaches style, not facts.** Train it on 200 answers about your company and it will *sound* like it knows your company,
but it will invent the details. For facts, Joshua already has better tools: memory, your files, and the web.

**How it works here.** We use **LoRA**: instead of changing the whole model, training adds a thin layer of new numbers
(an *adapter*, a few tens of MB) on top of it. It takes 15–60 minutes, entirely on your computer.
Your examples never leave it.

**What you get.** A new model in Joshua's menu, named `joshua-<your name>`. It's a small text model (4 billion parameters):
it can't see images, and it is less capable than the main one, but it will answer the way you taught it.

**The five steps:** ① name it · ② give examples · ③ train · ④ compare before and after · ⑤ install it into Joshua.
"""

STEP1 = """Each fine-tune is a **project** with its own examples and training. Give it a short name in lowercase,
like `formal-italian` or `newsroom-style`. The finished model will be called `joshua-<name>`."""

STEP2 = """The model learns from **examples**: something you might say, and the exact answer you want back.
Quality beats quantity: **100–500 good examples** work far better than 5,000 sloppy ones. Every example should show the habit
you want, and the answers should be ones you'd be happy to get.

Three ways to add them, which you can mix:
- **Upload a file**: `.jsonl` (one `{"messages": [...]}` chat per line, the standard format), `.json`, or a `.csv` with two columns, question and answer.
- **From your chats**: pick conversations you liked; each of your messages becomes an example with Joshua's reply. Edit the replies until they're exactly right.
- **By hand**: write them one at a time below.

You can edit any cell in the table, and empty a row's cells to delete it. Press **Save examples** when you're done."""

STEP3 = """Training shows the model your examples over and over, and each time adjusts the adapter a little so that
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

Joshua's chat models are unloaded while training, so the computer's memory goes to the training. Chatting will be slow until it ends."""

STEP4 = """Before installing, ask the same question to the model **before** and **after** training, side by side.
Use questions that are *not* in your examples: that's how you see whether it learned the habit or only memorised answers."""

STEP5 = """Installing makes your trained model a normal Ollama model that Joshua can use, in three automatic steps:
1. **Merge**: the adapter is folded into a copy of the model.
2. **Convert**: that copy is turned into **GGUF**, the file format Ollama runs.
3. **Add to Ollama**: it gets Joshua's chat template, so it can still use tools, and a name: `joshua-<name>`.

It needs about **18 GB of free disk** while it works; temporary files are deleted at the end and the model takes about 4.3 GB.
Then pick it from the model menu under the prompt."""

PRESETS = {"Light touch": (2, "2e-5", 8), "Normal": (3, "5e-5", 16), "Strong": (5, "1e-4", 16)}


def res_html():
    r, missing = ft.resources(), ft.ready()
    ok = r["ram"] >= 8 and r["free"] >= ft.BASE_GB + 2
    r["chip"] = r["chip"].replace("·", "<br>Graphics ·") if not ft.MAC else r["chip"]
    rows = [f"Chip · {html.escape(r['chip'])}", f"Memory · {r['ram']} GB {'✓' if r['ram'] >= 8 else '✗ needs 8 GB or more'}",
            f"Free disk · {r['free']} GB {'✓ enough to train' if r['free'] >= ft.BASE_GB + 2 else f'✗ training needs about {ft.BASE_GB + 2:.0f} GB'}"
            + ("" if r["free"] >= 18 else " · installing the result needs about 18 GB")]
    rows += [f"✗ {html.escape(m)}" for m in missing]
    verdict = "READY TO TRAIN" if ok and not missing else "NOT READY YET"
    return f"<div class='ftres'><b>{verdict}</b><br>" + "<br>".join(rows) + "</div>"


def table(rows):
    return pd.DataFrame(rows or [["", ""]], columns=["You say", "Joshua answers"])


def rows_of(df):
    if df is None:
        return []
    return [[str(q), str(a)] for q, a in df.values.tolist() if str(q).strip() and str(a).strip() and str(q) != "nan" and str(a) != "nan"]


def count_html(rows):
    n = len(rows)
    tip = ("Add at least 20 to train." if n < 20 else "Enough to try; 100 or more gives steadier results." if n < 100 else "A good amount.")
    return f"<div class='ftcount'>{n} EXAMPLES · {tip}</div>"


def build(conv_root, conv_choices):
    with gr.Column(visible=False, scale=5, elem_id="tune") as panel:
        with gr.Row():
            gr.HTML("<div id='tunetitle'>TEACH JOSHUA · FINE-TUNING</div>")
            back = gr.Button("← Back to chat", size="sm", scale=0, min_width=160, elem_id="tuneback")
        with gr.Accordion("⓪  What this is, and whether your computer can do it", open=True, elem_classes="ftstep"):
            gr.Markdown(INTRO)
            res = gr.HTML()
        with gr.Accordion("①  Name it", open=True, elem_classes="ftstep"):
            gr.Markdown(STEP1)
            with gr.Row():
                name = gr.Dropdown(choices=ft.projects(), value=None, allow_custom_value=True, label="Project name", scale=3)
                openp = gr.Button("Create / open", scale=1)
        with gr.Accordion("②  Give examples", open=False, elem_classes="ftstep") as acc2:
            gr.Markdown(STEP2)
            with gr.Row():
                upload = gr.File(file_types=[".jsonl", ".json", ".csv"], label="Upload a file", height=110, scale=1)
                with gr.Column(scale=2):
                    chats = gr.Dropdown(choices=conv_choices(), multiselect=True, label="From your chats")
                    addchats = gr.Button("Add from these chats", size="sm")
            with gr.Row():
                q = gr.Textbox(label="You say", lines=2, scale=2)
                a = gr.Textbox(label="Joshua answers", lines=2, scale=3)
                add = gr.Button("Add", scale=0, min_width=80)
            grid = gr.Dataframe(table([]), headers=["You say", "Joshua answers"], column_count=(2, "fixed"), interactive=True, wrap=True, max_height=420, elem_id="ftgrid")
            with gr.Row():
                count = gr.HTML(count_html([]))
                saveb = gr.Button("Save examples", variant="primary", scale=0, min_width=180)
        with gr.Accordion("③  Train", open=False, elem_classes="ftstep") as acc3:
            gr.Markdown(STEP3)
            preset = gr.Radio(list(PRESETS), value="Normal", label="Preset")
            with gr.Row():
                iters = gr.Slider(50, 1500, value=300, step=10, label="Steps")
                lr = gr.Dropdown(["1e-5", "2e-5", "5e-5", "1e-4", "2e-4"], value="5e-5", label="Learning rate")
                layers = gr.Dropdown([4, 8, 16, 24], value=16, label="Layers")
            with gr.Row():
                trainb = gr.Button("▶ Start training", variant="primary")
                stopb = gr.Button("■ Stop", elem_id="ftstop")
            plot = gr.LinePlot(pd.DataFrame({"step": [], "loss": [], "set": []}), x="step", y="loss", color="set", height=300, label="Loss, lower is better · blue = training · orange = validation", color_map={"training": "#4a7fb5", "validation": "#e8892a"})
        with gr.Accordion("④  Compare before and after", open=False, elem_classes="ftstep") as acc4:
            gr.Markdown(STEP4)
            with gr.Row():
                tq = gr.Textbox(label="A question that is not in your examples", scale=4)
                tryb = gr.Button("Ask both", scale=1)
            with gr.Row():
                before = gr.Markdown("", label="Before", elem_classes="ftans", container=True)
                after = gr.Markdown("", label="After", elem_classes="ftans", container=True)
        with gr.Accordion("⑤  Install it into Joshua", open=False, elem_classes="ftstep") as acc5:
            gr.Markdown(STEP5)
            with gr.Row():
                instb = gr.Button("⬇ Install into Joshua", variant="primary")
                delb = gr.Button("🗑 Delete this project", elem_id="ftdel")
        status = gr.HTML("<div class='ftstatus'>IDLE</div>")
        log = gr.Textbox(label="Console", lines=10, max_lines=10, autoscroll=True, interactive=False, elem_id="ftlog")
        seen = gr.State(None)

    # ----- handlers -----
    def open_project(n):
        n = (n or "").strip().lower()
        if not ft.NAME_RE.match(n):
            gr.Warning("Use 2–31 lowercase letters, digits or dashes, like 'newsroom-style'.")
            return (gr.skip(),) * 7
        rows = ft.load_examples(n) or ft.save_examples(n, [])
        meta = ft.pdir(n) / "training.json"
        df = plot_df(json.loads(meta.read_text())) if meta.exists() else gr.skip()
        return gr.Dropdown(choices=ft.projects(), value=n), table(rows), count_html(rows), gr.Accordion(open=True), gr.Accordion(open=bool(rows)), gr.Dropdown(choices=conv_choices()), df

    def need(n):
        if not n or not (ft.pdir(n) / "examples.json").exists():
            gr.Warning("First create or open a project in step ①.")
            return False
        return True

    def from_file(f, df):
        if not f:
            return gr.skip(), gr.skip()
        try:
            new = ft.parse_file(f)
        except Exception as e:
            gr.Warning(f"Couldn't read that file: {e}")
            return gr.skip(), gr.skip()
        gr.Info(f"Added {len(new)} examples. Check them, then Save.")
        rows = rows_of(df) + new
        return table(rows), count_html(rows)

    def from_chat_list(cids, df):
        new = ft.from_chats(conv_root, cids or [])
        gr.Info(f"Added {len(new)} examples from {len(cids or [])} chats. Edit the answers until they're exactly what you want, then Save.")
        rows = rows_of(df) + new
        return table(rows), count_html(rows)

    def add_one(qq, aa, df):
        if not qq.strip() or not aa.strip():
            gr.Warning("Write both what you say and the answer you want.")
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        rows = rows_of(df) + [[qq, aa]]
        return table(rows), count_html(rows), "", ""

    def save_rows(n, df, p):
        if not need(n):
            return gr.skip(), gr.skip(), gr.skip()
        rows = ft.save_examples(n, rows_of(df))
        gr.Info(f"Saved {len(rows)} examples.")
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
            gr.Warning(f"You have {len(rows)} saved examples: add at least 20 and press Save.")
            return
        tr, va = ft.write_dataset(n, rows)
        try:
            ft.train(n, int(it), l, int(lay), ft.resources()["ram"])
            gr.Info(f"Training on {tr} examples, checking on {va}. Follow it in the chart and console.")
        except RuntimeError as e:
            gr.Warning(str(e))

    def start_try(n, question):
        if not need(n):
            return
        if not ft.trained(n):
            gr.Warning("Train the model first (step ③).")
            return
        if not question.strip():
            gr.Warning("Write a question.")
            return
        try:
            ft.try_out(n, question)
        except RuntimeError as e:
            gr.Warning(str(e))

    def start_install(n):
        if not need(n):
            return
        if not ft.trained(n):
            gr.Warning("Train the model first (step ③).")
            return
        try:
            ft.install(n)
        except RuntimeError as e:
            gr.Warning(str(e))

    def delete(n):
        if not n or not ft.pdir(n).exists():
            return gr.skip(), gr.skip(), gr.skip()
        if ft.busy() and ft.RUN.name == n:
            gr.Warning("Stop the running step first.")
            return gr.skip(), gr.skip(), gr.skip()
        ft.delete_project(n)
        gr.Info(f"Deleted the project '{n}'. An installed joshua-{n} model stays in the menu.")
        return gr.Dropdown(choices=ft.projects(), value=None), table([]), count_html([])

    def plot_df(meta):
        pts = [(s, v, "training") for s, v in meta.get("train", [])] + [(s, v, "validation") for s, v in meta.get("val", [])]
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

    openp.click(open_project, name, [name, grid, count, acc2, acc3, chats, plot])
    name.select(open_project, name, [name, grid, count, acc2, acc3, chats, plot])
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
    delb.click(delete, name, [name, grid, count], js="(n) => (n && confirm('Delete this project, its examples and its training? An installed model stays.')) ? n : null")
    gr.Timer(1).tick(tick, seen, [status, log, plot, before, after, seen], show_progress="hidden")
    return panel, back, res
