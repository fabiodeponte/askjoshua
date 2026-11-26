"""Podcasts for Joshua: a dialogue written from your sources, read by voices on this computer, with jingles and pauses.
Voices come from three places: Kokoro (54 voices in 9 languages, one download), Piper (single voices in many more
languages, from a catalog) and, on a Mac, the voices of macOS itself. Everything runs on this computer."""
import json, os, pathlib, re, shutil, subprocess, sys, tempfile, urllib.request
import numpy as np

HERE = pathlib.Path(__file__).parent
VOICES = HERE / "voices"
RATE = 24000  # samples per second of the finished podcast
SAY = os.environ.get("JOSHUA_SAY") or (shutil.which("say") if sys.platform == "darwin" else None)

KOKORO = VOICES / "kokoro"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = {"kokoro-v1.0.fp16.onnx": 177, "voices-v1.0.bin": 28}  # MB
# the first letter of a Kokoro voice is its language: (code for Kokoro, name)
KOKORO_LANG = {"a": ("en-us", "English (US)"), "b": ("en-gb", "English (UK)"), "e": ("es", "Spanish"), "f": ("fr-fr", "French"),
               "h": ("hi", "Hindi"), "i": ("it", "Italian"), "j": ("ja", "Japanese"), "p": ("pt-br", "Portuguese (Brazil)"), "z": ("cmn", "Chinese")}

PIPER = VOICES / "piper"
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


def _need(*pkgs):
    """Installs the Python packages that are missing, the first time they are needed."""
    missing = []
    for pkg, mod in pkgs:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True, timeout=1800)


def _download(url, to, progress=None):
    to.parent.mkdir(parents=True, exist_ok=True)
    tmp = to.with_suffix(to.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total, got = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if progress and total:
                progress(got / total)
    tmp.replace(to)


# ---------- the voices on this computer ----------

def kokoro_ready():
    return all((KOKORO / f).exists() for f in KOKORO_FILES)


def mac_voices():
    if not SAY:
        return []
    out = subprocess.run([SAY, "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    found = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2,3})_([A-Za-z0-9]+)\s+#", line)
        if m:
            found.append({"id": "mac:" + m.group(1).strip(), "name": m.group(1).strip(), "code": m.group(2), "lang": m.group(2) + "-" + m.group(3), "engine": "macOS"})
    return found


def voices():
    """Every voice that can speak now: id ('kokoro:if_sara'), name, language code ('it') and language, engine."""
    found = []
    if kokoro_ready():
        for v in sorted(np.load(KOKORO / "voices-v1.0.bin").files):
            code, lang = KOKORO_LANG.get(v[0], ("en-us", "English"))
            found.append({"id": "kokoro:" + v, "name": v[3:].replace("_", " ").title() + (" ♀" if v[1] == "f" else " ♂"), "code": code[:2].replace("cm", "zh"), "lang": lang, "engine": "Kokoro"})
    for cfg in sorted(PIPER.glob("*.onnx.json")):
        info = json.loads(cfg.read_text())
        key = cfg.name.removesuffix(".onnx.json")
        lang = info.get("language", {})
        found.append({"id": "piper:" + key, "name": key.split("-")[1].replace("_", " ").title(), "code": lang.get("family", key[:2]),
                      "lang": lang.get("name_english", key[:5]) + (" (" + lang["country_english"] + ")" if lang.get("country_english") else ""), "engine": "Piper"})
    return found + mac_voices()


def label(v):
    return f"{v['name']} · {v['lang']} · {v['engine']}"


def install_kokoro(progress=None):
    """Kokoro's model and its 54 voices: about 205 MB."""
    _need(("kokoro-onnx", "kokoro_onnx"), ("soundfile", "soundfile"))
    for i, f in enumerate(KOKORO_FILES):
        if not (KOKORO / f).exists():
            _download(KOKORO_URL + f, KOKORO / f, progress and (lambda x, i=i: progress((i + x) / len(KOKORO_FILES))))


def piper_catalog():
    """The Piper voices you can install: name, language, quality, size. Read once from Hugging Face, then kept here."""
    cache = PIPER / "catalog.json"
    if not cache.exists():
        _download(PIPER_URL + "voices.json", cache)
    found = []
    for key, v in json.loads(cache.read_text()).items():
        lang = v.get("language", {})
        mb = sum(f.get("size_bytes", 0) for f in v.get("files", {}).values()) / 1e6
        found.append({"key": key, "name": v.get("name", key).replace("_", " ").title(), "quality": v.get("quality", ""),
                      "lang": lang.get("name_english", "") + (" (" + lang["country_english"] + ")" if lang.get("country_english") else ""),
                      "mb": round(mb), "files": [f for f in v.get("files", {}) if f.endswith((".onnx", ".onnx.json"))]})
    return sorted(found, key=lambda v: (v["lang"], v["name"], v["quality"]))


def install_piper(key, progress=None):
    _need(("piper-tts", "piper"), ("soundfile", "soundfile"))
    entry = next(v for v in piper_catalog() if v["key"] == key)
    for i, path in enumerate(sorted(entry["files"], key=lambda p: p.endswith(".json"))):  # the voice first, its settings last: a voice is listed once both are here
        _download(PIPER_URL + path, PIPER / pathlib.Path(path).name, progress and (lambda x, i=i: progress((i + x) / 2)))


def remove(voice_id):
    engine, name = voice_id.split(":", 1)
    if engine == "piper":
        for f in (PIPER / (name + ".onnx"), PIPER / (name + ".onnx.json")):
            f.unlink(missing_ok=True)
    elif engine == "kokoro":
        shutil.rmtree(KOKORO, ignore_errors=True)


# ---------- speaking ----------

_loaded = {}


def _resample(a, rate):
    if rate == RATE or not len(a):
        return a.astype(np.float32)
    n = int(len(a) * RATE / rate)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def speak(voice_id, text, speed=1.0):
    """One line of the script spoken by a voice, as sound at RATE samples per second."""
    engine, name = voice_id.split(":", 1)
    text = text.strip()
    if engine == "kokoro":
        if "kokoro" not in _loaded:
            _need(("kokoro-onnx", "kokoro_onnx"))
            from kokoro_onnx import Kokoro
            _loaded["kokoro"] = Kokoro(str(KOKORO / "kokoro-v1.0.fp16.onnx"), str(KOKORO / "voices-v1.0.bin"))
        a, rate = _loaded["kokoro"].create(text, voice=name, speed=speed, lang=KOKORO_LANG.get(name[0], ("en-us",))[0])
        return _resample(a, rate)
    if engine == "piper":
        if voice_id not in _loaded:
            _need(("piper-tts", "piper"))
            from piper import PiperVoice
            _loaded[voice_id] = PiperVoice.load(str(PIPER / (name + ".onnx")))
        v = _loaded[voice_id]
        from piper import SynthesisConfig
        chunks = [c.audio_float_array for c in v.synthesize(text, syn_config=SynthesisConfig(length_scale=1 / speed))]
        return _resample(np.concatenate(chunks) if chunks else np.zeros(0), v.config.sample_rate)
    if engine == "mac":
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            src, out = pathlib.Path(d) / "line.txt", pathlib.Path(d) / "line.aiff"
            src.write_text(text, encoding="utf-8")  # from a file: the text never becomes a command-line option
            subprocess.run([SAY, "-v", name, "-r", str(int(180 * speed)), "-f", str(src), "-o", str(out)], check=True, capture_output=True, timeout=300)
            a, rate = sf.read(out, dtype="float32", always_2d=True)
        return _resample(a.mean(axis=1), rate)
    raise ValueError(f"unknown voice {voice_id}")


# ---------- the script ----------
# One line per turn, "Name: what they say". A line [BREAK] is a pause between parts; [BREAK 3] a pause of 3 seconds.

BREAK = re.compile(r"^\[\s*(?:break|pausa|pause)\s*(\d+(?:[.,]\d+)?)?\s*s?\s*\]$", re.I)
TURN = re.compile(r"^\**\[?([^\]:*\n]{1,40}?)\]?\**\s*:\s*\**\s*(.+)$")  # also **Anna:** and [Anna]:


def parse(script):
    """The script as steps: ('say', name, text) and ('break', seconds or None)."""
    steps = []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        b = BREAK.match(line)
        if b:
            steps.append(("break", float(b.group(1).replace(",", ".")) if b.group(1) else None))
            continue
        m = TURN.match(line)
        if m:
            steps.append(("say", m.group(1).strip(), m.group(2).strip()))
        elif steps and steps[-1][0] == "say":  # a line without a name continues the turn before
            steps[-1] = ("say", steps[-1][1], steps[-1][2] + " " + line)
    return steps


def speakers_in(script):
    seen = []
    for s in parse(script):
        if s[0] == "say" and s[1] not in seen:
            seen.append(s[1])
    return seen


# ---------- jingles ----------

JINGLES = {"Morning": "bright notes going up", "Calm": "a soft chord", "News": "a pulse, like a newscast"}


def _tone(freq, secs, vol=0.25):
    t = np.arange(int(secs * RATE)) / RATE
    wave = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t) + 0.12 * np.sin(6 * np.pi * freq * t)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 2.2)  # a quick start, then it fades like a plucked string
    return (vol * wave * env).astype(np.float32)


def builtin_jingle(name):
    """Three short jingles made here, note by note, so a podcast can start with music without downloading any."""
    C = 261.63
    notes = {"Morning": [(0, 1), (0.18, 5 / 4), (0.36, 3 / 2), (0.54, 2), (0.9, 3 / 2), (0.9, 2), (0.9, 5 / 2)],  # (when, note as a ratio of C)
             "Calm": [(0, 1), (0.05, 6 / 5), (0.1, 3 / 2), (0.15, 15 / 8), (1.2, 4 / 3), (1.25, 5 / 3), (1.3, 2)],
             "News": [(i * 0.25, 2 if i % 4 == 0 else 3 / 2) for i in range(8)] + [(2.0, 1), (2.0, 3 / 2), (2.0, 2)]}[name]
    out = np.zeros(int(4 * RATE), np.float32)
    for at, ratio in notes:
        tone = _tone(C * ratio, 2.5, 0.18 if name != "News" else 0.14)
        i = int(at * RATE)
        out[i:i + len(tone)] += tone[:len(out) - i]
    end = np.nonzero(np.abs(out) > 0.003)[0]
    return out[:end[-1] + 1 if len(end) else 0]


def jingle(choice, folder):
    """A jingle as sound: one made here (by name), a file you added (in folder), or nothing."""
    if not choice:
        return np.zeros(0, np.float32)
    if choice in JINGLES:
        return builtin_jingle(choice)
    import soundfile as sf
    a, rate = sf.read(pathlib.Path(folder) / choice, dtype="float32", always_2d=True)
    a = _resample(a.mean(axis=1), rate)[: 30 * RATE]  # at most 30 seconds
    fade = min(len(a), RATE)  # the last second fades out
    a[len(a) - fade:] *= np.linspace(1, 0, fade)
    return a / max(1e-6, np.abs(a).max()) * 0.6


# ---------- the whole podcast ----------

def make(script, cast, out, gap=0.35, pause=1.5, sting=False, intro=None, outro=None, jingles_dir=None, speed=1.0, progress=None):
    """Reads the script with the voices in cast ({name: voice id}), with a gap between turns, a pause at each [BREAK]
    (with a short jingle if sting), the intro and outro jingles, and saves it as an MP3."""
    _need(("soundfile", "soundfile"))
    import soundfile as sf
    steps = parse(script)
    missing = sorted({s[1] for s in steps if s[0] == "say" and not cast.get(s[1])})
    if missing:
        raise ValueError("no voice for " + ", ".join(missing))
    silence = lambda s: np.zeros(int(s * RATE), np.float32)
    parts = []
    start = jingle(intro, jingles_dir)
    if len(start):
        parts += [start, silence(0.4)]
    says = sum(s[0] == "say" for s in steps)
    done = 0
    for s in steps:
        if s[0] == "break":
            parts.append(silence(s[1] if s[1] is not None else pause))
            if sting and s[1] is None:
                parts += [builtin_jingle("Calm")[: int(1.6 * RATE)] * np.linspace(1, 0, int(1.6 * RATE)), silence(0.3)]
        else:
            parts += [speak(cast[s[1]], s[2], speed), silence(gap)]
            done += 1
            if progress:
                progress(done / says)
    end = jingle(outro, jingles_dir)
    if len(end):
        parts += [silence(0.6), end]
    audio = np.concatenate(parts) if parts else silence(1)
    audio = audio / max(1.0, np.abs(audio).max() / 0.95)  # never louder than the loudest a file can hold
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, RATE, format="MP3") if out.suffix.lower() == ".mp3" else sf.write(out, audio, RATE)
    return out, len(audio) / RATE


# ---------- writing the script ----------

PROMPT = """You write the script of a podcast in {language}, from the sources below, for listeners who only hear it.
{who}
Length: about {words} words ({minutes} minutes).{title}
{notes}
Rules:
- Use only facts from the sources. Never invent facts, figures, names or quotations.
- Natural spoken dialogue: short sentences, the speakers ask each other questions, react, and take turns.
- Explain every acronym and every step a listener might not know. Few numbers: round them.
- When you name a news outlet, say what it is (for example "Reuters, the British news agency"). For politicians, say their role and country.
- Never say "the video" or "as you can see": listeners see nothing.
- No adjectives like "fascinating", "dramatic", "huge".
- Start with a greeting{greeting}. End with a short goodbye.
- Put a line [BREAK] on its own between one topic and the next.

Format: every line starts with the speaker's name and a colon, like this:
{example}
[BREAK]
Write only the script, nothing before or after it.

SOURCES
{sources}"""


def gather(texts, links, files, folder, budget=40000):
    """Every source as text, one after the other with its origin: the texts you pasted, the pages at the links,
    the files (PDF, Word, LibreOffice, plain text). Each gets an equal share of the budget, in characters."""
    import agent
    agent.set_workspace(folder)
    found = []
    if texts.strip():
        found.append(("Text", texts.strip()))
    for url in re.findall(r"https?://\S+", links or ""):
        found.append((url, agent.read_page(url)))
    for f in files or []:
        p = pathlib.Path(f)
        shutil.copy(p, agent._path(p.name))
        found.append((p.name, agent.read_file(p.name)))
    share = budget // max(1, len(found))
    return "\n\n".join(f"--- {name} ---\n{text[:share]}" for name, text in found), [name for name, _ in found]


def prompt(sources, names, language, minutes, title="", notes=""):
    names = [n.strip() for n in names if n.strip()] or ["Host"]
    who = ("One speaker: " + names[0] + ", who talks to the listener.") if len(names) == 1 else ("Speakers: " + ", ".join(names) + ".")
    example = "\n".join(f"{n}: …" for n in (names * 2)[:max(2, len(names))])
    return PROMPT.format(language=language, who=who, words=int(minutes * 150), minutes=minutes, title=f'\nTitle of the podcast: "{title}".' if title else "",
                         notes=("Instructions from the user: " + notes.strip()) if notes.strip() else "", greeting=f' that names "{title}"' if title else "",
                         example=example, sources=sources)
