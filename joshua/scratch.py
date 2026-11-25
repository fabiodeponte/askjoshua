"""A new model from zero: your texts or Wikipedia -> a vocabulary -> a small model trained on them -> optionally,
a second stage on other texts (the style of an author, say) -> into Ollama.
The heavy work runs in scratch_job.py, a separate process, one step at a time, through finetune's runner:
the page shows its status, console and chart the same way. Each model lives in newmodels/<name>/."""
import json, math, pathlib, re, shutil, sys

import finetune as ft
import scratch_job as job

HERE = pathlib.Path(__file__).parent
ROOT = HERE / "newmodels"
ROOT.mkdir(exist_ok=True)
JOB = [ft.PY, HERE / "scratch_job.py"]
SIZES = list(job.SIZES)


def projects():
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir())


def pdir(name):
    return ROOT / name


def _json(name, file):
    f = pdir(name) / file
    return json.loads(f.read_text()) if f.exists() else None


def data(name, stage=1):
    return _json(name, job.names(stage)["data"])


def progress(name, stage=1):
    return _json(name, job.names(stage)["progress"])


def trained(name):
    return (pdir(name) / "model" / "model.safetensors").exists()


def versions(name):
    """The versions you can try or build on: the first training, then each copy kept along every later stage."""
    d = pdir(name)
    out = [("after the first texts", "model")] if trained(name) else []
    for st in sorted((p for p in d.glob("stage*") if p.is_dir() and p.name[5:].isdigit()), key=lambda p: int(p.name[5:])):
        for p in sorted(st.glob("step-*")):
            out.append((f"stage {st.name[5:]}, step {int(p.name[5:])}", f"{st.name}/{p.name}"))
    return out


def stage_after(version):
    """Continuing from a version makes the next stage: from the first training, stage 2; from stage 2, stage 3."""
    return job.stage_of(version or "model") + 1


def last_stage(name):
    """The highest stage that has texts read for it (1 if none)."""
    return max([st for st in range(2, 10) if data(name, st)] or [1])


def params(size, vocab=job.VOCAB):
    h, layers, _, inter = job.SIZES[size]
    return vocab * h + layers * (4 * h * h + 3 * h * inter + 2 * h) + h


def steps(name, size, stage=1):
    """How long to train: about 20 tokens for every parameter (the rule of thumb for a model of a given size),
    but no more than 4 passes over your texts, after which it starts learning them by heart.
    The second stage only has to bend a trained model: 3 passes over the new texts, at most as long as the first."""
    d = data(name, stage)
    if stage > 1:
        return max(80, min(math.ceil(3 * d["train_tokens"] / (job.BATCH * job.CTX)), steps(name, size)))
    target = min(20 * params(size), 4 * d["train_tokens"])
    return max(200, math.ceil(target / (job.BATCH * job.CTX)))


def best_size(name):
    """The biggest size your texts can feed: one with more parameters than tokens/20 would be starved."""
    d = data(name)
    fit = [s for s in SIZES if params(s) * 20 <= d["train_tokens"] * 4]
    return fit[-1] if fit else SIZES[0]


SPEED = {"10M": 19500, "25M": 9300, "50M": 2500}  # tokens a second, measured on an M4 with 16 GB (Nov 2026)


def hours(name, size, stage=1):
    """The training time, from the speed measured on an M4; the console then shows the real time left."""
    return steps(name, size, stage) * job.BATCH * job.CTX / SPEED[size] / 3600


def _catch(run, tag, on):
    orig = run.say

    def say(line):
        if line.strip().startswith(tag):
            on(json.loads(line.strip()[len(tag):]))
            run.version += 1
        else:
            orig(line)
    run.say = say


def wiki(lang, millions):
    """millions of words, or "all" for the whole of that language's Wikipedia."""
    return f"wikipedia:{lang}:{millions}"


def wikiqa(lang, n):
    """n questions and answers made from Wikipedia articles: "Che cos'è X?" and the article's first paragraph."""
    return f"wikiqa:{lang}:{n}"


def wikisearch(lang, n):
    """n questions answered by looking them up: the question, what to search for, the start of the article, the answer."""
    return f"wikisearch:{lang}:{n}"


def searches(name, which):
    """The language of Wikipedia a version searches, if it went through the web-search step; otherwise None."""
    for st in range(2, job.stage_of(which) + 1):
        source = str((data(name, st) or {}).get("source", ""))
        if source.startswith("wikisearch:"):
            return source.split(":")[1]
    return None


def clear(name, stage):
    """Reading new texts for a stage makes what was built on the old ones meaningless. New first texts mean
    a new vocabulary, so everything goes; new second texts only replace the second stage."""
    d = pdir(name)
    for st in range(stage, 10) if stage > 1 else range(1, 10):
        for f in ("train", "val", "data", "progress"):
            (d / job.names(st)[f]).unlink(missing_ok=True)
        shutil.rmtree(d / f"stage{st}", ignore_errors=True)
    if stage == 1:
        (d / "tokenizer.json").unlink(missing_ok=True)
        shutil.rmtree(d / "model", ignore_errors=True)


def downloaded(name):
    """What step 1 got: download.json, or None."""
    return _json(name, "download.json")


def download(name, source):
    """Step 1: new texts mean a new model, so everything built on the old ones goes, then they're downloaded."""
    def target(run):
        clear(name, 1)
        (pdir(name) / "download.json").unlink(missing_ok=True)
        run.status = "getting the texts"
        run.result = {"phases": {}, "source": source, "stage": 1}
        _catch(run, "@@PHASE@@", lambda p: run.result["phases"].__setitem__(p["phase"], p))
        return ft._stream(run, JOB + ["download", pdir(name), source]) == 0 and downloaded(name) is not None
    return ft._start("downloading", name, target)


def prepare(name, source, stage=1):
    def target(run):
        clear(name, stage)
        run.status = "reading the texts and learning their vocabulary" if stage == 1 else "reading the texts for the second stage"
        run.result = {"phases": {}, "source": source, "stage": stage}
        _catch(run, "@@PHASE@@", lambda p: run.result["phases"].__setitem__(p["phase"], p))  # the page shows each phase
        return ft._stream(run, JOB + ["prepare", pdir(name), source, stage]) == 0 and data(name, stage) is not None
    return ft._start("reading" if stage == 1 else "reading2", name, target)


def train(name, size, stage=1, mix=0.0, base="model"):
    def target(run):
        ft.free_memory()
        n = steps(name, size, stage)
        p = progress(name, stage)
        same = p and p.get("size") == size and (stage == 1 or (p.get("mix") == mix and p.get("base", "model") == base)) and p.get("step", 0) < n
        run.train, run.val = (p["train"], p["val"]) if same else ([], [])
        run.result = {"samples": p.get("samples", []) if same else [], "old": p.get("old", []) if same else [], "stage": stage}
        _catch(run, "@@SAMPLE@@", lambda s: run.result["samples"].append(s))
        if stage > 1:
            _catch(run, "@@OLD@@", lambda o: run.result["old"].append(o))
        run.total = n
        run.status = f"building a {size} model" if stage == 1 else f"stage {stage}: continuing from {base}"
        return ft._stream(run, JOB + ["train", pdir(name), size, n, stage, mix, base]) == 0
    return ft._start("pretraining" if stage == 1 else "pretraining2", name, target)


def try_out(name, prompt, which="model", question=False):
    def target(run):
        ft.free_memory()
        run.status = "writing"
        run.result = {}
        _catch(run, "@@TRY@@", lambda t: run.result.update(text=t))
        lang = searches(name, which) if question else None  # a version that learned to search looks the question up first
        return ft._stream(run, JOB + ["try", pdir(name), prompt, which] + (["question", lang] if lang else ["question"] if question else [])) == 0 and "text" in run.result
    return ft._start("continue", name, target)


CONVERT = """
import sys
sys.path.insert(0, sys.argv[1])
pre = lambda self, tokenizer: "gpt-2"  # our own vocabulary is new to llama.cpp, but it splits words the way GPT-2 does
try:
    import conversion.base as cb
    cb.TextModel.get_vocab_base_pre = pre
except ImportError:
    pass
import convert_hf_to_gguf as c
if hasattr(c, "TextModel"):
    c.TextModel.get_vocab_base_pre = pre
sys.argv = [c.__file__] + sys.argv[2:]
c.main()
"""


NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # what Ollama accepts as a model name


def in_joshua():
    """The models made here and added to Joshua: name -> which model and version, and whether it answers questions.
    Joshua's chat lists them even though they can't use tools, and talks to them plainly (see web.worker)."""
    f = ROOT / "in_joshua.json"
    return json.loads(f.read_text()) if f.exists() else {}


def move(old, new):
    """Ollama's names ignore capitals: Mattapedia and MATTAPEDIA are the same model, so copying one onto the other and
    deleting the old would delete it. When only the capitals change, it goes through a temporary name."""
    import ollama
    if old.removesuffix(":latest").lower() == new.removesuffix(":latest").lower():
        tmp = new + "-renaming"
        ollama.copy(old, tmp)
        ollama.delete(old)
        ollama.copy(tmp, new)
        ollama.delete(tmp)
    else:
        ollama.copy(old, new)
        ollama.delete(old)


def rename(old, new):
    """A model added to Joshua gets a new name: Ollama copies it under the new one and drops the old (the weights are
    shared, so nothing is copied for real), and in_joshua.json follows."""
    import ollama
    new = (new or "").strip()
    if not NAME_OK.match(new):
        raise RuntimeError("Use letters, digits, dots, dashes or underscores for the name, like Wikipedia-Italian.")
    listed = in_joshua()
    if old not in listed:
        raise RuntimeError(f"{old} isn't one of the models added from here.")
    taken = {m.model.removesuffix(":latest").lower() for m in ollama.list().models} | {k.lower() for k in listed}
    if new.lower() in taken - {old.lower()}:
        raise RuntimeError(f"There is already a model called {new}.")
    move(old, new)
    listed[new] = listed.pop(old)
    (ROOT / "in_joshua.json").write_text(json.dumps(listed, indent=1))


def install(name, which="model", as_name=None):
    as_name = (as_name or f"joshua-new-{name}").strip()
    if not NAME_OK.match(as_name):
        raise RuntimeError("Use letters, digits, dots, dashes or underscores for the name, like Wikipedia-Italian.")

    def target(run):
        d = pdir(name)
        gguf = d / f"{name}.gguf"
        run.status = "1/2 converting it to the format Ollama uses (GGUF)"
        if ft._stream(run, [ft.PY, "-c", CONVERT, ft.LLAMA, d / which, "--outfile", gguf, "--outtype", "f16"]) != 0:
            return False
        run.status = "2/2 adding it to Ollama"
        # a version that went through a questions-and-answers stage gets the same frame in Ollama, so you only type
        # the question there too; otherwise there's no template: the model continues whatever text it is given
        upto = job.stage_of(which)
        answers = any(str((data(name, st) or {}).get("source", "")).startswith(("wikiqa:", "wikisearch:")) for st in range(2, upto + 1))
        lang = searches(name, which)  # Joshua's chat does the searching for it; Ollama alone gets the plain question frame
        frame = f"{job.END}\n{job.QUESTION} {{{{ .Prompt }}}}\n{job.ANSWER}" if answers else "{{ .Prompt }}"
        (d / "Modelfile").write_text(f'FROM ./{gguf.name}\nTEMPLATE """{frame}"""\nPARAMETER num_ctx {job.CTX}\n'
                                     f'PARAMETER stop "{job.END}"\nPARAMETER temperature 0.8\n')
        code = ft._stream(run, [ft.ollama_bin(), "create", as_name, "-f", "Modelfile"], cwd=d)
        gguf.unlink(missing_ok=True)
        if code == 0:
            listed = in_joshua()
            listed[as_name] = {"model": name, "version": which, "answers": answers} | ({"search": lang} if lang else {})
            (ROOT / "in_joshua.json").write_text(json.dumps(listed, indent=1))
            run.say(f"Done: {as_name} is in Joshua's model menu. In a terminal: ollama run {as_name} " + ('"Che cos\'è l\'automobile?"' if answers else '"The beginning of a sentence"'))
        return code == 0
    return ft._start("install", name, target)


def delete(name):
    shutil.rmtree(pdir(name), ignore_errors=True)
