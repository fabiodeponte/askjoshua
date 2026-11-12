"""Fine-tuning for Joshua: examples -> LoRA training -> test -> install into Ollama.
On a Mac it trains with mlx-lm (Apple's MLX); on Linux with PyTorch + PEFT on an NVIDIA card (tune_cuda.py).

Every heavy step runs as a separate process (so the web page stays responsive and memory
is freed when it ends). The state of the running step lives in RUN, which the page polls.
"""
import csv, io, json, os, pathlib, platform, random, re, shutil, subprocess, sys, threading, time
import ollama, requests
import ranking

HERE = pathlib.Path(__file__).parent
ROOT = HERE / "finetune"
ROOT.mkdir(exist_ok=True)
LLAMA = HERE.parent / "llama.cpp"  # the installer puts llama.cpp's converter here
PY = sys.executable
CONFIG = json.loads((HERE / "config.json").read_text()) if (HERE / "config.json").exists() else {}
MAC = sys.platform == "darwin"
CUDA = [PY, HERE / "tune_cuda.py"]
DEFAULT = "qwen3-4b"  # the model Joshua taught before you could choose: a small text model that runs on 8 GB Macs


# ---------- which model to teach ----------

def bases():
    """The models you can teach, from models.json on ask.joshua.earth (updated without a new Joshua). Each has a
    version for Apple's MLX (mac), the original for PyTorch on Linux (linux), and its twin in Ollama's library (ollama),
    from which we borrow the chat template so the result can still use tools."""
    return ranking.data().get("tune") or json.loads((HERE / "models.json").read_text())["tune"]


def custom(repo):
    """Any model on Hugging Face: you give its name (on a Mac, an MLX version, like mlx-community/…-4bit)."""
    info = requests.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=15)
    if not info.ok:
        raise RuntimeError(f"Hugging Face doesn't know '{repo}'. Copy the name from the model's page, like mlx-community/Llama-3.2-3B-Instruct-4bit.")
    gb = sum((f.get("size") or 0) for f in info.json().get("siblings", []) if f["rfilename"].endswith(".safetensors")) / 1e9
    quant = re.search(r"(\d)[-_]?bit", repo, re.I)
    bits = int(quant[1]) if quant else 16
    return {"id": "custom", "label": repo, "mac": repo, "linux": repo, "ollama": None, "gb": round(gb, 1),
            "params": round(gb * 8 / bits, 1), "chatml": "qwen" in repo.lower()}


def base(name):
    """The model this project teaches (projects made before you could choose teach the original Qwen3 4B)."""
    f = pdir(name) / "base.json"
    chosen = json.loads(f.read_text()) if f.exists() else {"id": DEFAULT}
    if chosen.get("id") == "custom":
        return chosen
    return next((b for b in bases() if b["id"] == chosen["id"]), next(b for b in bases() if b["id"] == DEFAULT))


def set_base(name, entry):
    pdir(name).mkdir(parents=True, exist_ok=True)
    (pdir(name) / "base.json").write_text(json.dumps(entry if entry.get("id") == "custom" else {"id": entry["id"]}))


def repo(entry):
    return entry["mac"] if MAC else entry["linux"]


def needs(entry):
    """GB of memory and of free disk this model needs: to train, and to install the result (which unpacks it for a while)."""
    p, gb = entry["params"], entry["gb"]
    if MAC:  # MLX trains the 4-bit model; memory for it, its gradients and long examples
        ram, download = gb * 1.3 + 3, gb
    else:  # PyTorch loads the original and squeezes it to 4 bits on the card
        ram, download = p * 0.7 + 3, p * 2
    return {"ram": round(ram), "train_disk": round(download + 1), "install_disk": round(p * 3.2 + 1), "final": round(p * 1.1, 1)}


def fits(entry, res=None):
    """Whether this computer can teach the model, and if not, what is short: ("ram" or "disk", GB needed)."""
    res = res or resources()
    n = needs(entry)
    have = res["ram"] - (4 if MAC else 0) if MAC else (gpus()[0][1] if gpus() else 0)  # on a Mac the system keeps about 4 GB
    if have < n["ram"]:
        return False, ("ram", n["ram"])
    if res["free"] < n["train_disk"]:
        return False, ("disk", n["train_disk"])
    return True, None
SYSTEM = "You are Joshua, an AI assistant running on the user's computer."
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,30}$")


def ready():
    """What is missing for fine-tuning on this computer, as a list of sentences (empty = all good)."""
    missing = []
    if MAC and platform.machine() != "arm64":
        missing.append("Fine-tuning needs a Mac with Apple Silicon (M1 or later).")
    if MAC:
        import importlib.util
        if not importlib.util.find_spec("mlx_lm"):
            missing.append("The training tools (mlx-lm) are not installed: run the installer again and answer yes to fine-tuning.")
        elif subprocess.run([PY, "-c", "import mlx_lm"], capture_output=True).returncode:
            repair()  # a newer Hugging Face library can break the training tools: bring transformers up to date
            if subprocess.run([PY, "-c", "import mlx_lm"], capture_output=True).returncode:
                missing.append("The training tools (mlx-lm) don't start: run the installer again and answer yes to fine-tuning.")
    else:
        gpu = gpus()
        if not gpu:
            missing.append("On Linux, fine-tuning needs an NVIDIA graphics card with at least 8 GB of its own memory.")
        elif gpu[0][1] < 7.5:
            missing.append(f"Your {gpu[0][0]} has {gpu[0][1]:.0f} GB of memory; fine-tuning needs at least 8 GB.")
        r = subprocess.run([PY, "-c", "import torch, peft, bitsandbytes; assert torch.cuda.is_available()"], capture_output=True)
        if r.returncode:
            missing.append("The training tools (PyTorch, PEFT) are not installed or can't see the card: run the installer again and answer yes to fine-tuning.")
    if not (LLAMA / "convert_hf_to_gguf.py").exists():
        missing.append("The converter (llama.cpp) is missing: run the installer again and answer yes to fine-tuning.")
    return missing


def repair():
    subprocess.run([PY, "-m", "pip", "install", "-q", "-U", "transformers>=5"], capture_output=True, timeout=600)


def refresh_converter(run=None):
    """Downloads the newest llama.cpp converter (only its Python part): new model families need a recent one."""
    import tarfile, tempfile
    if run:
        run.status = "updating the converter (llama.cpp), which learns new model families every week"
    with tempfile.TemporaryDirectory() as tmp:
        data = requests.get("https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz", timeout=300).content
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(tmp, filter="data")
        shutil.rmtree(LLAMA, ignore_errors=True)
        shutil.move(pathlib.Path(tmp) / "llama.cpp-master", LLAMA)
    r = subprocess.run([PY, "-m", "pip", "install", "-q", "-U", str(LLAMA / "gguf-py")], capture_output=True, text=True, timeout=600)
    if run:
        run.say("Converter updated." if r.returncode == 0 else "Couldn't update the converter: " + r.stderr[-300:])
    return r.returncode == 0


def gpus():
    """NVIDIA cards as (name, GB of memory), from nvidia-smi; empty when there are none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return [(n.strip(), int(m) / 1024) for n, m in (l.rsplit(",", 1) for l in out.strip().splitlines())]
    except Exception:
        return []


def resources():
    if MAC:
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0) / 2**30
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    else:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        chip = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.machine())
        chip += "".join(f" · {n} ({m:.0f} GB)" for n, m in gpus()[:1])
    return {"ram": round(ram), "chip": chip or "unknown", "free": round(shutil.disk_usage(HERE).free / 2**30, 1)}


def trained(name):
    a = pdir(name) / "adapters"
    return (a / "adapters.safetensors").exists() or (a / "adapter_model.safetensors").exists()


def batch_size(ram):
    return 1 if ram < 12 else 2 if ram < 24 else 4


# ---------- projects and examples ----------

def projects():
    return sorted(p.name for p in ROOT.iterdir() if (p / "examples.json").exists())


def pdir(name):
    return ROOT / name


def load_examples(name):
    p = pdir(name) / "examples.json"
    return json.loads(p.read_text()) if p.exists() else []


def save_examples(name, rows):
    rows = [[str(q).strip(), str(a).strip()] for q, a in rows if str(q).strip() and str(a).strip()]
    pdir(name).mkdir(parents=True, exist_ok=True)
    (pdir(name) / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


def parse_file(path):
    """Examples from a .jsonl, .json or .csv file. Accepts {"messages": [...]}, question/answer, prompt/completion,
    or a CSV whose first two columns are question and answer."""
    text = pathlib.Path(path).read_text(encoding="utf-8-sig")
    rows = []
    if path.lower().endswith(".csv"):
        for r in csv.reader(io.StringIO(text)):
            if len(r) >= 2 and r[0].strip().lower() not in ("question", "domanda", "prompt", "input"):
                rows.append([r[0], r[1]])
        return rows
    items = json.loads(text) if path.lower().endswith(".json") else [json.loads(l) for l in text.splitlines() if l.strip()]
    for it in items:
        if "messages" in it:
            user = None
            for m in it["messages"]:
                if m["role"] == "user":
                    user = m["content"]
                elif m["role"] == "assistant" and user:
                    rows.append([user, m["content"]])
                    user = None
        else:
            q = it.get("question") or it.get("prompt") or it.get("input") or it.get("instruction")
            a = it.get("answer") or it.get("completion") or it.get("output") or it.get("response")
            if q and a:
                rows.append([q, a])
    return rows


def from_chats(conv_dir, cids):
    """Question/answer pairs from Joshua chats: each message you wrote, with Joshua's final reply to it."""
    rows = []
    for cid in cids:
        p = pathlib.Path(conv_dir) / cid / "chat.json"
        if not p.exists():
            continue
        user, answer = None, ""
        for m in json.loads(p.read_text())["llm"] + [{"role": "user", "content": None}]:
            if m["role"] == "user":
                if user and answer:
                    rows.append([user, re.sub(r"\n+Sources?:\n.*", "", answer, flags=re.S).strip()])
                user, answer = m["content"] if isinstance(m["content"], str) else None, ""
            elif m["role"] == "assistant" and m.get("content"):
                answer = m["content"]
    return [r for r in rows if not r[0].startswith(("Sources found on the web", "A fact check", "Your team of"))]


def chatml(question, answer=None):
    """The exact text the model sees in Ollama (the same 'ChatML' layout as its template). We write it out ourselves
    because the base model's own template adds empty <think></think> tags that the trained model would then learn."""
    text = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return text if answer is None else text + answer + "<|im_end|>\n"


RENDER = """
import sys, json
from transformers import AutoTokenizer
repo, system, src, dst = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(repo)
def chat(q, a=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}] + ([{"role": "assistant", "content": a}] if a is not None else [])
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=a is None)
    except Exception:  # some templates (Gemma's older ones) refuse a system message: put it in the first user message
        msgs = [{"role": "user", "content": system + "\\n\\n" + q}] + msgs[2:]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=a is None)
def plain(text):  # the trainer adds the start-of-text token itself: don't have it twice
    bos = tok.bos_token or ""
    return text[len(bos):] if bos and text.startswith(bos) else text
rows = json.load(open(src))
json.dump([plain(chat(q, a)) for q, a in rows], open(dst, "w"), ensure_ascii=False)
"""


def layout(entry, rows):
    """The exact text the model sees for each (question, answer); answer None = only the question, for a test."""
    if entry.get("chatml"):
        return [chatml(q, a) for q, a in rows]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = pathlib.Path(tmp) / "in.json", pathlib.Path(tmp) / "out.json"
        src.write_text(json.dumps(rows, ensure_ascii=False))
        r = subprocess.run([PY, "-c", RENDER, repo(entry), SYSTEM, src, dst], capture_output=True, text=True, timeout=600)
        if r.returncode:
            raise RuntimeError("Couldn't read this model's chat format: " + r.stderr.strip().splitlines()[-1][-300:])
        return json.loads(dst.read_text())


def write_dataset(name, rows):
    """mlx-lm wants a folder with train.jsonl and valid.jsonl, one chat per line. About 1 in 10 examples is held out
    for validation: the model never trains on them, so their loss shows whether it learned or just memorised."""
    rows = rows[:]
    random.Random(0).shuffle(rows)
    n_val = max(2, len(rows) // 10)
    data = pdir(name) / "data"
    data.mkdir(parents=True, exist_ok=True)
    texts = layout(base(name), rows)  # the base model's own chat format (ChatML for Qwen)
    for fname, part in (("valid.jsonl", texts[:n_val]), ("train.jsonl", texts[n_val:])):
        with open(data / fname, "w") as f:
            for text in part:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return len(rows) - n_val, n_val


def suggest_iters(n_train, ram, passes=3):
    return max(100, min(1000, round(passes * n_train / batch_size(ram) / 10) * 10))


# ---------- the running step (one at a time) ----------

class Run:
    def __init__(self, kind, name):
        self.kind, self.name, self.log, self.train, self.val = kind, name, [], [], []
        self.status, self.done, self.ok, self.proc, self.started, self.version = "starting", False, False, None, time.time(), 0
        self.result = None

    def say(self, line):
        line = ANSI.sub("", line).rstrip()
        if not line:
            return
        if self.log and re.match(r"^\S.*\d+%\|", line) and re.match(r"^\S.*\d+%\|", self.log[-1]):
            self.log[-1] = line  # download progress bars overwrite themselves
        else:
            self.log.append(line)
        del self.log[:-400]
        # mlx-lm prints either "Iter 10: Train loss 1.23, ..." (older) or a table row "  10    1.230 ▼   45   1.0k" (newer)
        m = re.search(r"Iter (\d+): Train loss ([\d.]+)", line) or re.match(r"^\s*(\d+)\s+([\d.]+)\s+[▼▲=•]?\s*\d", line)
        if m:
            self.train.append((int(m[1]), float(m[2])))
            self.status = f"training: step {m[1]}"
        m = re.search(r"Iter (\d+): Val loss ([\d.]+)", line) or re.match(r"^\s*(\d+)\s+val\s+([\d.]+)", line)
        if m:
            self.val.append((int(m[1]), float(m[2])))
        self.version += 1


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RUN = None
LOCK = threading.Lock()


def busy():
    return RUN is not None and not RUN.done


def free_memory():
    """Unload the chat models from Ollama, so training has the computer's memory to itself."""
    try:
        for m in ollama.ps().models:
            ollama.generate(model=m.model, prompt="", keep_alive=0)
    except Exception:
        pass


def _stream(run, cmd, cwd=None):
    run.say("$ " + " ".join(str(c) for c in cmd))
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", NO_COLOR="1", COLUMNS="110")
    run.proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    buf = b""
    while True:
        b = run.proc.stdout.read1(4096) if hasattr(run.proc.stdout, "read1") else run.proc.stdout.read(1)
        if not b:
            break
        buf += b
        *lines, buf = re.split(rb"[\r\n]", buf)
        for l in lines:
            run.say(l.decode(errors="replace"))
    run.say(buf.decode(errors="replace"))
    return run.proc.wait()


def _start(kind, name, target):
    global RUN
    with LOCK:
        if busy():
            raise RuntimeError(f"Another step ({RUN.kind}) is still running.")
        RUN = Run(kind, name)
    run = RUN

    def go():
        try:
            run.ok = target(run)
        except Exception as e:
            run.say(f"Error: {e}")
            run.ok = False
        run.status = "finished" if run.ok else ("stopped" if run.status == "stopping" else "failed")
        run.done = True
        run.version += 1
    threading.Thread(target=go, daemon=True).start()
    return run


def stop():
    if busy():
        RUN.status = "stopping"
        if RUN.proc and RUN.proc.poll() is None:
            RUN.proc.terminate()


def train(name, iters, lr, layers, ram):
    def target(run):
        free_memory()
        adapters = pdir(name) / "adapters"
        shutil.rmtree(adapters, ignore_errors=True)
        b = base(name)
        run.status = f"loading {b['label']} (the first time it downloads about {needs(b)['train_disk'] - 1} GB)"
        if not MAC:
            code = _stream(run, CUDA + ["train", repo(b), pdir(name) / "data", adapters, iters, lr, layers, batch_size(ram), max(20, iters // 10)])
        else:
            code = _stream(run, [PY, "-m", "mlx_lm", "lora", "--model", repo(b), "--train", "--data", pdir(name) / "data", "--adapter-path", adapters,
                                 "--iters", iters, "--learning-rate", lr, "--num-layers", layers, "--batch-size", batch_size(ram),
                                 "--steps-per-report", 10, "--steps-per-eval", max(20, iters // 10), "--val-batches", -1, "--max-seq-length", 2048,
                                 "--grad-checkpoint", "--save-every", max(50, iters // 4)])
        meta = {"iters": iters, "lr": lr, "layers": layers, "trained": time.time(), "train": run.train, "val": run.val}
        (pdir(name) / "training.json").write_text(json.dumps(meta))
        return code == 0 and trained(name)
    return _start("training", name, target)


TRY = """
import sys, json, mlx.core as mx
from mlx_lm import load, generate
base, adapters, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
for label, ad in (("BEFORE", None), ("AFTER", adapters)):
    model, tok = load(base, adapter_path=ad)
    text = generate(model, tok, prompt=prompt, max_tokens=500)
    print("@@" + label + "@@" + json.dumps(text), flush=True)
    del model; mx.clear_cache()
"""


def try_out(name, prompt):
    def target(run):
        free_memory()
        run.status = "writing both answers"
        out = {}
        orig = run.say

        def catch(line):
            m = re.match(r"@@(BEFORE|AFTER)@@(.*)", line.strip())
            if m:
                out[m[1]] = json.loads(m[2])
                run.result = out
                run.version += 1
            else:
                orig(line)
        run.say = catch
        b = base(name)
        code = _stream(run, ([PY, "-c", TRY] if MAC else CUDA + ["try"]) + [repo(b), pdir(name) / "adapters", layout(b, [[prompt, None]])[0]])
        return code == 0 and len(out) == 2
    return _start("test", name, target)


FALLBACK_TEMPLATE = """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""


def borrowed(entry):
    """The chat template (how Ollama lays out messages and tool calls) and settings (stop words, temperature) of the
    same model in Ollama's library: a few KB, no model download. None when the model has no twin there:
    Ollama then reads the template stored in the file itself, which usually works but may not handle tools."""
    tag = entry.get("ollama")
    if not tag:
        return (FALLBACK_TEMPLATE, {"stop": ["<|im_end|>"]}) if entry.get("chatml") else (None, {})
    try:
        name, version = tag.split(":")
        man = requests.get(f"https://registry.ollama.ai/v2/library/{name}/manifests/{version}", headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"}, timeout=15).json()
        blob = lambda kind: next((requests.get(f"https://registry.ollama.ai/v2/library/{name}/blobs/{l['digest']}", timeout=15).text for l in man["layers"] if l["mediaType"].endswith(kind)), None)
        params = blob(".params")
        return blob(".template"), (json.loads(params) if params else {})
    except Exception:
        return (FALLBACK_TEMPLATE, {"stop": ["<|im_end|>"]}) if entry.get("chatml") else (None, {})


def ollama_bin():
    return shutil.which("ollama") or "/Applications/Ollama.app/Contents/Resources/ollama"


def install(name):
    def target(run):
        work, b = pdir(name), base(name)
        fused, gguf = work / "fused", work / f"{name}.gguf"
        need = needs(b)["install_disk"]
        free = shutil.disk_usage(work).free / 2**30
        if free < need:
            run.say(f"Not enough free disk: this step needs about {need} GB for a while and you have {free:.1f} GB. Free some space and try again.")
            return False
        free_memory()
        run.status = "1/3 merging what it learned into the model"
        fuse = [PY, "-m", "mlx_lm", "fuse", "--model", repo(b), "--adapter-path", work / "adapters", "--save-path", fused, "--dequantize"] if MAC else CUDA + ["fuse", repo(b), work / "adapters", fused]
        if _stream(run, fuse) != 0:
            return False
        run.status = "2/3 converting it to the format Ollama uses (GGUF)"
        convert = [PY, LLAMA / "convert_hf_to_gguf.py", fused, "--outfile", gguf, "--outtype", "q8_0"]
        code = _stream(run, convert)
        if code != 0 and refresh_converter(run):  # most often a model family newer than our copy of the converter
            run.status = "2/3 converting it, with the updated converter"
            code = _stream(run, convert)
        shutil.rmtree(fused, ignore_errors=True)
        if code != 0:
            return False
        run.status = "3/3 adding it to Ollama"
        tpl, params = borrowed(b)
        lines = [f"FROM ./{gguf.name}"] + ([f'TEMPLATE """{tpl}"""'] if tpl else [])
        for key, value in params.items():
            for v in (value if isinstance(value, list) else [value]):
                lines.append(f"PARAMETER {key} {json.dumps(v) if isinstance(v, str) else v}")
        (work / "Modelfile").write_text("\n".join(lines) + "\n")
        code = _stream(run, [ollama_bin(), "create", f"joshua-{name}", "-f", "Modelfile"], cwd=work)
        gguf.unlink(missing_ok=True)
        if code == 0:
            run.say(f"Done. 'joshua-{name}' is now in Joshua's model menu.")
        return code == 0
    return _start("install", name, target)


def delete_project(name, model_too=False):
    shutil.rmtree(pdir(name), ignore_errors=True)
    if model_too:
        try:
            ollama.delete(f"joshua-{name}")
        except Exception:
            pass
