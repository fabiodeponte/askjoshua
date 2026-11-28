"""The Podcast page: sources in, a script you can edit, then the audio with the voices, pauses and jingles you choose."""
import datetime, html, json, pathlib, re, shutil, threading, time
import gradio as gr
import podcast as pc
import i18n
from i18n import t

LANGUAGES = {"it": "Italian", "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "pl": "Polish",
             "nl": "Dutch", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "ru": "Russian", "uk": "Ukrainian"}
MAX_SPEAKERS = 4
DEFAULT_NAMES = ["Anna", "Marco", "Sofia", "Luca"]
JOBS = {}  # the audio being made: progress and result, read by the page every second


def ui_language():
    lang = i18n.chosen()
    return (i18n.system_language() if lang == "auto" else lang)[:2]


def voice_choices():
    return [(pc.label(v), v["id"]) for v in pc.voices()]


def default_voices(code, n):
    """Different voices in the podcast's language, women and men in turn; voices of other languages if there aren't enough."""
    vs = pc.voices()
    same = [v for v in vs if v["code"] == code]
    women, men = [v for v in same if "♀" in v["name"]], [v for v in same if "♀" not in v["name"]]
    turns = [x for pair in zip(women + [None] * len(men), men + [None] * len(women)) for x in pair if x]
    turns += [v for v in same if v not in turns] + [v for v in vs if v not in same]
    return [turns[i % len(turns)]["id"] if turns else None for i in range(n)]


def jingle_choices(folder):
    mine = sorted(p.name for p in pathlib.Path(folder).glob("*") if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aiff"))
    return [(t("None"), "")] + [(t(n) + " · " + t(d), n) for n, d in pc.JINGLES.items()] + [("🎵 " + n, n) for n in mine]


def voices_html():
    vs = pc.voices()
    by = {}
    for v in vs:
        by.setdefault(v["engine"], []).append(v)
    if not vs:
        return "<div class='ftempty'>" + t("No voices yet: install Kokoro below, or a Piper voice in your language.") + "</div>"
    return "<div class='podvoices'>" + " · ".join(f"<b>{html.escape(e)}</b> {len(l)}" for e, l in by.items()) + " " + t("voices on this computer") + "</div>"


def build(conv_root, model_choices, write):
    """conv_root: where Joshua keeps your data. model_choices(): the models menu. write(model, prompt): the model's
    answer, piece by piece."""
    home = pathlib.Path(conv_root) / "podcasts"
    jingles = home / "jingles"
    jingles.mkdir(parents=True, exist_ok=True)
    draft_file = home / "draft.json"

    with gr.Column(visible=False, scale=5, elem_id="podcast") as panel:
        with gr.Row(elem_classes="pagehead"):
            gr.HTML("<div id='podtitle'>" + t("Podcast") + "</div>")
            back = gr.Button("← " + t("Back to chat"), size="sm", scale=0, min_width=160, elem_classes="backbtn")
        gr.Markdown(t("A podcast from your sources: Joshua writes the dialogue, you edit it, then the voices on this computer read it."), elem_classes="ftlead")

        with gr.Column(elem_classes="ftpane"):
            gr.Markdown("**" + t("Your podcasts") + "** " + t("The ones you made, newest first, with their scripts."))
            with gr.Row(equal_height=True):
                past = gr.Dropdown([], label=t("Podcast"), scale=4)
                reopen = gr.Button("✎ " + t("Open its script"), size="sm", scale=0, min_width=170, interactive=False)  # until a podcast is chosen
            past_audio = gr.Audio(visible=False, show_label=False, type="filepath", interactive=False)


        with gr.Column(elem_classes="ftpane"):
            gr.Markdown("**1 · " + t("Sources") + "** " + t("Texts, links and files: the podcast uses only what is in them."))
            texts = gr.Textbox(label=t("Texts"), lines=5, placeholder=t("Paste one or more texts here"))
            links = gr.Textbox(label=t("Links"), lines=2, placeholder=t("One web address per line"))
            with gr.Row(equal_height=True):
                front = gr.Checkbox(False, label=t("These are news sites' front pages: read their latest articles"), scale=3)
                per_page = gr.Slider(1, 15, value=5, step=1, label=t("Articles per site"), scale=2, interactive=False)  # only for news front pages
            files = gr.File(label=t("Files (PDF, Word, LibreOffice, text)"), file_count="multiple", file_types=[".pdf", ".docx", ".odt", ".txt", ".md", ".html"], elem_classes="podfile")

        with gr.Column(elem_classes="ftpane"):
            gr.Markdown("**2 · " + t("The script") + "** " + t("Who speaks, for how long and how. Then Joshua writes the dialogue: change it as you like before making the audio."))
            with gr.Row():
                title = gr.Textbox(label=t("Title"), placeholder=t("e.g. Week in review"), max_lines=1, scale=3)
                lang = gr.Dropdown([(t("Same as the sources"), "auto")] + [(t(n), c) for c, n in LANGUAGES.items()], value="auto", label=t("Language"), scale=3)
            with gr.Row():
                minutes = gr.Slider(1, 30, value=5, step=1, label=t("Minutes"), scale=3)
                count = gr.Slider(1, MAX_SPEAKERS, value=2, step=1, label=t("Speakers"), scale=3)
            names, voices, samples, rows = [], [], [], []
            for i in range(MAX_SPEAKERS):
                with gr.Row(visible=i < 2, equal_height=False, elem_classes="podspeaker") as row:
                    names.append(gr.Textbox(DEFAULT_NAMES[i], label=t("Speaker {n}", n=i + 1), max_lines=1, scale=2))
                    voices.append(gr.Dropdown([], label=t("Voice"), scale=4))
                    samples.append(gr.Button("▶ " + t("Listen"), size="sm", scale=0, min_width=110))
                rows.append(row)
            gr.HTML("<div class='podvoices'><a href='#' onclick=\"document.getElementById('podmorevoices').scrollIntoView({behavior: 'smooth'}); return false\">" + t("More voices can be installed at the bottom of this page.") + "</a></div>")
            sample_audio = gr.Audio(visible=False, autoplay=True, show_label=False, elem_id="podsample")
            notes = gr.Textbox(label=t("Instructions"), lines=2, placeholder=t("Tone, audience, what to stress or leave out…"))
            model = gr.Dropdown(model_choices(), label=t("Model that writes"))
            with gr.Row():
                writeb = gr.Button("✍ " + t("Write the script"), size="sm", scale=0, min_width=190, elem_classes="ftmain", variant="primary")
            status = gr.HTML()
            script = gr.Textbox(label=t("Script"), lines=22, max_lines=60, buttons=["copy"], elem_id="podscript",
                                placeholder=t("Anna: Good morning and welcome…\nMarco: …\n[BREAK]\n\nOne line per turn: the speaker's name, a colon, what they say. [BREAK] is a pause between parts; [BREAK 3] lasts 3 seconds."))

        with gr.Column(elem_classes="ftpane"):
            gr.Markdown("**3 · " + t("The sound") + "** " + t("Pauses and music around the voices."))
            with gr.Row():
                gap = gr.Slider(0, 1.5, value=0.35, step=0.05, label=t("Pause between speakers (seconds)"))
                pause = gr.Slider(0.5, 6, value=1.5, step=0.5, label=t("Pause between sections (seconds)"))
                speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label=t("Speed"))
            with gr.Row(equal_height=False):  # each jingle with its own ▶ under it
                with gr.Column(scale=3):
                    intro = gr.Dropdown(jingle_choices(jingles), value="Morning", label=t("Opening jingle"))
                    hear_in = gr.Button("▶ " + t("Listen"), size="sm", scale=0, min_width=120)
                with gr.Column(scale=3):
                    brk = gr.Dropdown(jingle_choices(jingles), value="", label=t("Sound between sections"))
                    hear_brk = gr.Button("▶ " + t("Listen"), size="sm", scale=0, min_width=120)
                with gr.Column(scale=3):
                    outro = gr.Dropdown(jingle_choices(jingles), value="Calm", label=t("Closing jingle"))
                    hear_out = gr.Button("▶ " + t("Listen"), size="sm", scale=0, min_width=120)
            add_jingle = gr.File(label=t("Add your own jingle (MP3, WAV…)"), file_types=["audio"], elem_classes="podfile")
            bed = gr.Checkbox(False, label=t("The opening music goes on softly under the first voices"))
            with gr.Row():
                makeb = gr.Button("🎙 " + t("Make the audio"), size="sm", scale=0, min_width=190, elem_classes="ftmain", variant="primary")
            progress = gr.HTML()
            result = gr.Audio(visible=False, label=t("Your podcast"), type="filepath", interactive=False)

        with gr.Column(elem_classes="ftpane", elem_id="podmorevoices"):
            gr.Markdown("**" + t("More voices (optional)") + "** " + t("The voices installed on this computer, and more to add. They all work without internet once installed."))
            have = gr.HTML(voices_html())
            with gr.Row(equal_height=True):
                gr.Markdown(t("**Kokoro** (extra free voices): 54 natural voices in English, Italian, French, Spanish, Portuguese, Japanese, Chinese and Hindi. About 205 MB."), elem_classes="podline")
                kokoro_b = gr.Button("⬇ " + t("Install Kokoro"), size="sm", scale=0, min_width=170)
            with gr.Row(equal_height=True):
                plang = gr.Dropdown([], label=t("More free voices (Piper), in many more languages"), scale=2)
                pvoice = gr.Dropdown([], label=t("Voice"), scale=3)
                piper_b = gr.Button("⬇ " + t("Install this voice"), size="sm", scale=0, min_width=170)
            catalog_b = gr.Button(t("Show the Piper voices"), size="sm", scale=0, min_width=200)
            vstatus = gr.HTML()

    # ---------- what the controls do ----------

    def refresh_voices(code, n, *current):
        """The voice menus, after an install: what you picked stays, empty ones get a voice in the podcast's language."""
        choices = voice_choices()
        code = code if code in LANGUAGES else (ui_language() if ui_language() in LANGUAGES else "en")
        fits = {v["id"] for v in pc.voices() if v["code"] == code}  # a voice you picked stays if it speaks this language
        picks = default_voices(code, MAX_SPEAKERS)
        return [gr.Dropdown(choices=choices, value=cur if cur in fits or (cur and not fits) else picks[i]) for i, cur in enumerate(current)] + [voices_html()]

    def past_list(value=None):
        files = sorted(home.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
        choices = [(f"{p.stem} · {datetime.datetime.fromtimestamp(p.stat().st_mtime):%d %b %Y %H:%M}", p.name) for p in files]
        return gr.Dropdown(choices=choices, value=value)

    def play_past(name):
        return gr.Audio(value=str(home / name), visible=True) if name and (home / name).exists() else gr.Audio(visible=False)

    def open_script(name):
        f = (home / (name or "")).with_suffix(".txt")
        return f.read_text(encoding="utf-8") if name and f.exists() else gr.skip()

    def kokoro_button():
        return gr.Button("✓ " + t("Kokoro is installed.") if pc.kokoro_ready() else "⬇ " + t("Install Kokoro"), interactive=not pc.kokoro_ready())

    def show_speakers(n):
        return [gr.Row(visible=i < n) for i in range(MAX_SPEAKERS)]

    def sample(voice, name, code):
        if not voice:
            return gr.Audio(visible=False)
        text = {"it": "Ciao, sono {n}. Questa è la mia voce.", "fr": "Bonjour, je suis {n}. Voici ma voix.", "de": "Hallo, ich bin {n}. Das ist meine Stimme.",
                "es": "Hola, soy {n}. Esta es mi voz.", "pl": "Cześć, jestem {n}. To jest mój głos.", "zh": "你好，我是{n}。这是我的声音。"}.get(code, "Hello, I'm {n}. This is my voice.")
        out = home / ".samples" / (re.sub(r"\W+", "-", voice) + ".wav")
        pc.make(f"{name or 'X'}: " + text.format(n=name or ""), {name or "X": voice}, out, gap=0, speed=1.0)
        return gr.Audio(value=str(out), visible=True)

    def hear_jingle(choice):
        if not choice:
            return gr.Audio(visible=False)
        out = home / ".samples" / "jingle.wav"
        out.parent.mkdir(exist_ok=True)
        import soundfile as sf
        sf.write(out, pc.jingle(choice, jingles), pc.RATE)
        return gr.Audio(value=str(out), visible=True)

    def add_own(f):
        if not f:
            return gr.skip(), gr.skip(), gr.skip(), None
        name = re.sub(r"[^\w.-]+", "-", pathlib.Path(f).name)
        shutil.copy(f, jingles / name)
        choices = jingle_choices(jingles)
        return gr.Dropdown(choices=choices, value=name), gr.Dropdown(choices=choices), gr.Dropdown(choices=choices), None

    def write_script(texts_, links_, front_, per_page_, files_, title_, code, minutes_, n, model_, notes_, *names_):
        if not model_:
            yield gr.skip(), "<div class='ftempty'>" + t("Choose the model that writes.") + "</div>", gr.skip()
            return
        yield gr.skip(), "<div class='ftres'>" + t("Reading the sources…") + "</div>", gr.skip()
        try:
            sources, used = pc.gather(texts_ or "", links_ or "", [getattr(f, "name", f) for f in files_ or []], home / "sources", front=int(per_page_) if front_ else 0)
        except Exception as e:
            yield gr.skip(), "<div class='ftempty'>" + t("Couldn't read the sources: {error}", error=html.escape(str(e))) + "</div>", gr.skip()
            return
        if not sources.strip():
            yield gr.skip(), "<div class='ftempty'>" + t("Add at least one text, link or file.") + "</div>", gr.skip()
            return
        if code == "auto":  # the language most of the sources are written in
            import agent
            found = (agent.lang_for(re.sub(r"--- .* ---", "", sources)[:5000]) or "en")[:2]
            code = found if found in LANGUAGES else "en"
        ask = pc.prompt(sources, list(names_)[:int(n)], LANGUAGES[code], int(minutes_), title_ or "", notes_ or "")
        got = ""
        head = "<div class='ftres'>" + t("Writing, from: {sources}", sources=html.escape(", ".join(used))) + "</div>"
        for piece in write(model_, ask):
            got += piece
            yield got, head, code
        got = re.sub(r"<think>.*?</think>", "", got, flags=re.S).strip()
        speakers = pc.speakers_in(got)
        unknown = [s for s in speakers if s not in [x.strip() for x in names_[:int(n)]]]
        yield got, "<div class='ftres'>" + t("Done: about {m} min. Edit it as you like, then make the audio.", m=max(1, round(len(got.split()) / 150))) + "</div>" + (
            "<div class='ftempty'>" + t("Names in the script without a voice: {names}. Rename them, or give the speakers those names.", names=html.escape(", ".join(unknown))) + "</div>" if unknown else ""), code

    def make_audio(script_, title_, gap_, pause_, brk_, intro_, outro_, speed_, bed_, n, *rest):
        names_, voices_ = rest[:MAX_SPEAKERS], rest[MAX_SPEAKERS:]
        cast = {names_[i].strip(): voices_[i] for i in range(int(n)) if names_[i].strip() and voices_[i]}
        if not script_.strip():
            yield "<div class='ftempty'>" + t("Write or paste a script first.") + "</div>", gr.skip(), gr.skip()
            return
        missing = [s for s in pc.speakers_in(script_) if s not in cast]
        if missing:
            yield "<div class='ftempty'>" + t("Names in the script without a voice: {names}. Rename them, or give the speakers those names.", names=html.escape(", ".join(missing))) + "</div>", gr.skip(), gr.skip()
            return
        slug = re.sub(r"\W+", "-", (title_ or "podcast").lower()).strip("-")[:40] or "podcast"
        out = home / f"{slug}-{datetime.datetime.now():%Y%m%d-%H%M}.mp3"
        job = JOBS["now"] = {"done": 0.0, "error": None, "file": None, "start": time.time()}

        def run():
            try:
                job["file"], job["secs"] = pc.make(script_, cast, out, gap_, pause_, brk_ or None, intro_ or None, outro_ or None, jingles, speed_, bed_, progress=lambda x: job.update(done=x))
            except Exception as e:
                job["error"] = str(e)
        threading.Thread(target=run, daemon=True).start()
        while not job["file"] and not job["error"]:
            time.sleep(1)
            left = (time.time() - job["start"]) * (1 - job["done"]) / job["done"] if job["done"] > 0.02 else None
            yield ("<div class='ftres'>" + t("Reading the script aloud: {p}%", p=int(job["done"] * 100)) + (" · " + t("about {m} min left", m=max(1, round(left / 60))) if left else "") + "</div>"), gr.skip(), gr.skip()
        if job["error"]:
            yield "<div class='ftempty'>" + t("The audio failed: {error}", error=html.escape(job["error"])) + "</div>", gr.skip(), gr.skip()
            return
        yield ("<div class='ftres'>" + t("Ready: {m} min {s} s, saved as {file}", m=int(job["secs"] // 60), s=int(job["secs"] % 60), file=html.escape(str(job["file"]))) + "</div>",
               gr.Audio(value=str(job["file"]), visible=True), past_list(job["file"].name))

    def install_kokoro():
        state = {"p": 0.0, "done": False, "error": None}

        def run():
            try:
                pc.install_kokoro(lambda x: state.update(p=x))
            except Exception as e:
                state["error"] = str(e)
            state["done"] = True
        threading.Thread(target=run, daemon=True).start()
        while not state["done"]:
            time.sleep(1)
            yield "<div class='ftres'>" + t("Installing Kokoro: {p}%", p=int(state["p"] * 100)) + "</div>"
        yield "<div class='ftempty'>" + t("Couldn't install: {error}", error=html.escape(state["error"])) + "</div>" if state["error"] else "<div class='ftres'>" + t("Kokoro is installed.") + "</div>"

    def show_catalog():
        try:
            cat = pc.piper_catalog()
        except Exception as e:
            return gr.skip(), "<div class='ftempty'>" + t("Couldn't read the list of voices: {error}", error=html.escape(str(e))) + "</div>"
        langs = sorted({v["lang"] for v in cat})
        return gr.Dropdown(choices=langs, value=None), ""

    def piper_voices(language):
        cat = [v for v in pc.piper_catalog() if v["lang"] == language] if language else []
        return gr.Dropdown(choices=[(f"{v['name']} · {v['quality']} · {v['mb']} MB", v["key"]) for v in cat], value=cat[0]["key"] if cat else None)

    def install_piper(key):
        if not key:
            yield "<div class='ftempty'>" + t("Choose a language and a voice.") + "</div>"
            return
        state = {"p": 0.0, "done": False, "error": None}

        def run():
            try:
                pc.install_piper(key, lambda x: state.update(p=x))
            except Exception as e:
                state["error"] = str(e)
            state["done"] = True
        threading.Thread(target=run, daemon=True).start()
        while not state["done"]:
            time.sleep(1)
            yield "<div class='ftres'>" + t("Installing the voice: {p}%", p=int(state["p"] * 100)) + "</div>"
        yield "<div class='ftempty'>" + t("Couldn't install: {error}", error=html.escape(state["error"])) + "</div>" if state["error"] else "<div class='ftres'>" + t("The voice is installed.") + "</div>"

    # the draft is kept, so leaving the page or closing Joshua doesn't lose it
    fields = [texts, links, front, per_page, title, lang, minutes, count, notes, script, gap, pause, brk, speed, bed, intro, outro, *names, *voices]
    keys = ["texts", "links", "front", "per_page", "title", "lang", "minutes", "count", "notes", "script", "gap", "pause", "brk", "speed", "bed", "intro", "outro"] + \
           [f"name{i}" for i in range(MAX_SPEAKERS)] + [f"voice{i}" for i in range(MAX_SPEAKERS)]

    def save(*vals):
        draft_file.write_text(json.dumps(dict(zip(keys, vals)), ensure_ascii=False))

    def opened():
        """The page as you left it: the draft, the voices installed now, the models in the menu."""
        try:
            d = json.loads(draft_file.read_text())
        except Exception:
            d = {}
        code = d.get("lang") if d.get("lang") in LANGUAGES else (ui_language() if ui_language() in LANGUAGES else "en")
        choices, picks = voice_choices(), default_voices(code, MAX_SPEAKERS)
        ids = {c[1] for c in choices}
        jl = jingle_choices(jingles)
        out = []
        for k in keys:
            if k.startswith("voice"):
                v = d.get(k)
                out.append(gr.Dropdown(choices=choices, value=v if v in ids else picks[int(k[5:])]))
            elif k in ("intro", "outro", "brk"):
                out.append(gr.Dropdown(choices=jl, value=d.get(k, {"intro": "Morning", "outro": "Calm", "brk": ""}[k])))
            else:
                out.append(d[k] if k in d else gr.skip())
        return out + show_speakers(int(d.get("count", 2))) + [voices_html(), gr.Dropdown(choices=model_choices()), kokoro_button(), past_list()]
    open_outputs = fields + rows + [have, model, kokoro_b, past]

    count.change(show_speakers, count, rows, show_progress="hidden")
    vargs = [lang, count, *voices]
    for i in range(MAX_SPEAKERS):
        samples[i].click(sample, [voices[i], names[i], lang], sample_audio)
    hear_in.click(hear_jingle, intro, sample_audio)
    hear_out.click(hear_jingle, outro, sample_audio)
    hear_brk.click(hear_jingle, brk, sample_audio)
    past.change(play_past, past, past_audio, show_progress="hidden")
    past.change(lambda v: gr.Button(interactive=bool(v)), past, reopen, show_progress="hidden")
    reopen.click(open_script, past, script)
    add_jingle.upload(add_own, add_jingle, [intro, outro, brk, add_jingle])
    front.change(lambda f: gr.Slider(interactive=bool(f)), front, per_page, show_progress="hidden")  # Articles per site only means something for news front pages
    writeb.click(write_script, [texts, links, front, per_page, files, title, lang, minutes, count, model, notes, *names], [script, status, lang]).then(save, fields, None)
    makeb.click(save, fields, None).then(make_audio, [script, title, gap, pause, brk, intro, outro, speed, bed, count, *names, *voices], [progress, result, past])
    kokoro_b.click(install_kokoro, None, vstatus).then(refresh_voices, vargs, [*voices, have]).then(kokoro_button, None, kokoro_b)
    catalog_b.click(show_catalog, None, [plang, vstatus])
    plang.change(piper_voices, plang, pvoice, show_progress="hidden")
    piper_b.click(install_piper, pvoice, vstatus).then(refresh_voices, vargs, [*voices, have])
    lang.change(refresh_voices, vargs, [*voices, have], show_progress="hidden")
    script.blur(save, fields, None)
    return panel, back, (opened, open_outputs)
