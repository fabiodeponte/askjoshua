"""A new model from nothing: your texts -> a vocabulary -> a small model trained on them -> into Ollama.
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


def data(name):
    f = pdir(name) / "data.json"
    return json.loads(f.read_text()) if f.exists() else None


def progress(name):
    f = pdir(name) / "progress.json"
    return json.loads(f.read_text()) if f.exists() else None


def trained(name):
    return (pdir(name) / "model" / "model.safetensors").exists()


def params(size, vocab=job.VOCAB):
    h, layers, _, inter = job.SIZES[size]
    return vocab * h + layers * (4 * h * h + 3 * h * inter + 2 * h) + h


def steps(name, size):
    """How long to train: about 20 tokens for every parameter (the rule of thumb for a model of a given size),
    but no more than 4 passes over your texts, after which it starts learning them by heart."""
    d = data(name)
    target = min(20 * params(size), 4 * d["train_tokens"])
    return max(200, math.ceil(target / (job.BATCH * job.CTX)))


def best_size(name):
    """The biggest size your texts can feed: one with more parameters than tokens/20 would be starved."""
    d = data(name)
    fit = [s for s in SIZES if params(s) * 20 <= d["train_tokens"] * 4]
    return fit[-1] if fit else SIZES[0]


def hours(name, size):
    """A rough guess of the training time on Apple silicon (about 2 trillion operations a second);
    the console then shows the real time left."""
    secs = 6 * params(size) * job.BATCH * job.CTX * steps(name, size) / 2e12
    return secs / 3600


def _catch(run, tag, on):
    orig = run.say

    def say(line):
        if line.strip().startswith(tag):
            on(json.loads(line.strip()[len(tag):]))
            run.version += 1
        else:
            orig(line)
    run.say = say


def prepare(name, folder):
    def target(run):
        d = pdir(name)
        for f in ("train.bin", "val.bin", "data.json", "tokenizer.json", "progress.json"):
            (d / f).unlink(missing_ok=True)
        shutil.rmtree(d / "model", ignore_errors=True)  # a new vocabulary makes the old model meaningless
        run.status = "reading your texts and learning their vocabulary"
        return ft._stream(run, JOB + ["prepare", d, folder]) == 0 and data(name) is not None
    return ft._start("reading", name, target)


def train(name, size):
    def target(run):
        ft.free_memory()
        p = progress(name)
        run.train, run.val = (p["train"], p["val"]) if p and p.get("size") == size else ([], [])
        run.result = {"samples": p.get("samples", []) if p and p.get("size") == size else []}
        _catch(run, "@@SAMPLE@@", lambda s: run.result["samples"].append(s))
        run.status = f"building a {size} model"
        return ft._stream(run, JOB + ["train", pdir(name), size, steps(name, size)]) == 0
    return ft._start("pretraining", name, target)


def try_out(name, prompt):
    def target(run):
        ft.free_memory()
        run.status = "writing"
        run.result = {}
        _catch(run, "@@TRY@@", lambda t: run.result.update(text=t))
        return ft._stream(run, JOB + ["try", pdir(name), prompt]) == 0 and "text" in run.result
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


def install(name):
    def target(run):
        d = pdir(name)
        gguf = d / f"{name}.gguf"
        run.status = "1/2 converting it to the format Ollama uses (GGUF)"
        if ft._stream(run, [ft.PY, "-c", CONVERT, ft.LLAMA, d / "model", "--outfile", gguf, "--outtype", "f16"]) != 0:
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
