"""A new model from zero: your texts or Wikipedia -> a vocabulary -> a small model trained on them -> optionally,
a second stage on other texts (the style of an author, say) -> into Ollama.
The heavy work runs in scratch_job.py, a separate process, one step at a time, through finetune's runner:
the page shows its status, console and chart the same way. Each model lives in newmodels/<name>/."""
import json, math, pathlib, shutil, sys

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
    """The versions you can try: the first training, then each copy kept along the second stage."""
    d = pdir(name)
    out = [("after the first texts", "model")] if trained(name) else []
    for p in sorted((d / "stage2").glob("step-*")) if (d / "stage2").exists() else []:
        out.append((f"second stage, step {int(p.name[5:])}", f"stage2/{p.name}"))
    return out


def params(size, vocab=job.VOCAB):
    h, layers, _, inter = job.SIZES[size]
    return vocab * h + layers * (4 * h * h + 3 * h * inter + 2 * h) + h


def steps(name, size, stage=1):
    """How long to train: about 20 tokens for every parameter (the rule of thumb for a model of a given size),
    but no more than 4 passes over your texts, after which it starts learning them by heart.
    The second stage only has to bend a trained model: 3 passes over the new texts, at most as long as the first."""
    d = data(name, stage)
    if stage == 2:
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


def clear(name, stage):
    """Reading new texts for a stage makes what was built on the old ones meaningless. New first texts mean
    a new vocabulary, so everything goes; new second texts only replace the second stage."""
    d = pdir(name)
    for st in (1, 2) if stage == 1 else (2,):
        for f in ("train", "val", "data", "progress"):
            (d / job.names(st)[f]).unlink(missing_ok=True)
    shutil.rmtree(d / "stage2", ignore_errors=True)
    if stage == 1:
        (d / "tokenizer.json").unlink(missing_ok=True)
        shutil.rmtree(d / "model", ignore_errors=True)


def prepare(name, source, stage=1):
    def target(run):
        clear(name, stage)
        run.status = "reading the texts and learning their vocabulary" if stage == 1 else "reading the texts for the second stage"
        run.result = {"phases": {}, "source": source, "stage": stage}
        _catch(run, "@@PHASE@@", lambda p: run.result["phases"].__setitem__(p["phase"], p))  # the page shows each phase
        return ft._stream(run, JOB + ["prepare", pdir(name), source, stage]) == 0 and data(name, stage) is not None
    return ft._start("reading" if stage == 1 else "reading2", name, target)


def train(name, size, stage=1, mix=0.0):
    def target(run):
        ft.free_memory()
        n = steps(name, size, stage)
        p = progress(name, stage)
        same = p and p.get("size") == size and (stage == 1 or p.get("mix") == mix) and p.get("step", 0) < n
        run.train, run.val = (p["train"], p["val"]) if same else ([], [])
        run.result = {"samples": p.get("samples", []) if same else [], "old": p.get("old", []) if same else [], "stage": stage}
        _catch(run, "@@SAMPLE@@", lambda s: run.result["samples"].append(s))
        if stage == 2:
            _catch(run, "@@OLD@@", lambda o: run.result["old"].append(o))
        run.total = n
        run.status = f"building a {size} model" if stage == 1 else "continuing on the second texts"
        return ft._stream(run, JOB + ["train", pdir(name), size, n, stage, mix]) == 0
    return ft._start("pretraining" if stage == 1 else "pretraining2", name, target)


def try_out(name, prompt, which="model"):
    def target(run):
        ft.free_memory()
        run.status = "writing"
        run.result = {}
        _catch(run, "@@TRY@@", lambda t: run.result.update(text=t))
        return ft._stream(run, JOB + ["try", pdir(name), prompt, which]) == 0 and "text" in run.result
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


def install(name, which="model"):
    def target(run):
        d = pdir(name)
        gguf = d / f"{name}.gguf"
        run.status = "1/2 converting it to the format Ollama uses (GGUF)"
        if ft._stream(run, [ft.PY, "-c", CONVERT, ft.LLAMA, d / which, "--outfile", gguf, "--outtype", "f16"]) != 0:
            return False
        run.status = "2/2 adding it to Ollama"
        # no chat template: the model continues whatever text it is given
        (d / "Modelfile").write_text(f'FROM ./{gguf.name}\nTEMPLATE """{{{{ .Prompt }}}}"""\nPARAMETER num_ctx {job.CTX}\n'
                                     f'PARAMETER stop "{job.END}"\nPARAMETER temperature 0.8\n')
        code = ft._stream(run, [ft.ollama_bin(), "create", f"joshua-new-{name}", "-f", "Modelfile"], cwd=d)
        gguf.unlink(missing_ok=True)
        if code == 0:
            run.say(f"Done. In a terminal: ollama run joshua-new-{name} \"The beginning of a sentence\"")
        return code == 0
    return ft._start("install", name, target)


def delete(name):
    shutil.rmtree(pdir(name), ignore_errors=True)
