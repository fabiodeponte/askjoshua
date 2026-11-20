"""Training a small language model from zero, with Apple's MLX. Run by scratch.py as a separate process:

  python scratch_job.py prepare DIR SOURCE STAGE    texts -> numbers. SOURCE is a folder, or wikipedia:LANG:MILLIONS (words).
                                                    Stage 1 also learns the vocabulary (tokenizer.json); stage 2 reuses it.
  python scratch_job.py train DIR SIZE STEPS STAGE MIX
                                                    stage 1: random weights -> trained on train.bin
                                                    stage 2: the stage-1 model -> trained on train2.bin, with MIX (0-1) of
                                                    each batch still from train.bin; a copy is kept at 8 points along the way
  python scratch_job.py try DIR PROMPT WHICH        a version of the model continues a piece of text

Both trainings pick up where they stopped. Progress is printed in mlx-lm's format ("Iter 10: Train loss 1.234"),
so the page reads it like fine-tuning.
"""
import hashlib, json, pathlib, shutil, subprocess, sys, time

import numpy as np

HERE = pathlib.Path(__file__).parent
TYPES = {".txt", ".md", ".pdf", ".epub", ".docx", ".odt", ".html", ".htm", ".rtf"}
END = "<|end|>"  # marks where one document ends and the next begins
VOCAB = 8192
CTX = 512  # tokens the model sees at once
BATCH = 8  # windows per step: 8 × 512 = 4,096 tokens. On a 16 GB M4, 32 used 14 GB and was no faster per token
WIKI = ("wikimedia/wikipedia", "20231101")  # Wikipedia as plain text, one row per article, on Hugging Face
SNAPSHOTS = 8  # copies of the model kept along the second stage

# the three sizes: width of each layer, how many layers, attention heads, width of the middle of each layer
SIZES = {"10M": (256, 8, 4, 1024), "25M": (384, 10, 6, 1280), "50M": (512, 12, 8, 1792)}
LR = {"10M": 1e-3, "25M": 8e-4, "50M": 6e-4}
if __import__("os").environ.get("JOSHUA_SCRATCH_TEST"):  # tiny, for testing the whole path on a computer without Apple's chip
    SIZES, CTX, BATCH = {k: (64, 2, 2, 128) for k in SIZES}, 64, 4


def say(*a):
    print(*a, flush=True)


def names(stage):
    """The files of each stage: stage 1 is the first training, stage 2 the one that continues it on other texts."""
    s = "" if stage == 1 else "2"
    return {"train": f"train{s}.bin", "val": f"val{s}.bin", "data": f"data{s}.json", "progress": f"progress{s}.json",
            "model": "model" if stage == 1 else "stage2/model"}


# ---------- 1. the texts ----------

def from_folder(folder):
    """Each file in the folder (and its subfolders) as one document: a list of paragraphs. Repeats are dropped."""
    sys.path.insert(0, str(HERE))
    import reader
    reader.MAX_CHARS = 10**12  # the Reader stops long books early; here we want every word
    files = sorted(p for p in pathlib.Path(folder).expanduser().rglob("*") if p.suffix.lower() in TYPES and p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"No texts found in {folder} (it reads {', '.join(sorted(TYPES))}).")
    seen = set()
    for i, f in enumerate(files, 1):
        try:
            paras = reader.extract(f)[1]
        except Exception as e:
            say(f"skipped {f.name}: {e}")
            continue
        keep = []
        for p in paras:
            h = hashlib.sha1(p.strip().lower().encode()).digest()[:8]
            if len(p) > 1 and h not in seen:  # the same paragraph twice teaches nothing new
                seen.add(h)
                keep.append(p)
        say(f"read {i}/{len(files)} · {f.name}")
        yield keep


def need(package, module=None):
    try:
        __import__(module or package)
    except ImportError:
        say(f"installing {package}…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def from_wikipedia(lang, words):
    """Wikipedia in one language, article by article, until about `words` words. Downloaded a part at a time
    (each a few hundred MB) from Hugging Face, which keeps them, so preparing again doesn't download again."""
    need("pyarrow")
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    repo, date = WIKI
    parts = sorted(f.path for f in HfApi().list_repo_tree(repo, path_in_repo=f"{date}.{lang}", repo_type="dataset") if f.path.endswith(".parquet"))
    if not parts:
        raise SystemExit(f"No Wikipedia found for the language '{lang}'.")
    got = 0
    for k, part in enumerate(parts, 1):
        say(f"downloading Wikipedia ({lang}), part {k} of {len(parts)}…")
        path = hf_hub_download(repo, part, repo_type="dataset")
        for b in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000):
            for text in b.column(0).to_pylist():
                paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 1]
                got += sum(len(p.split()) for p in paras)
                yield paras
                if got >= words:
                    return
        say(f"{got:,} words so far")


def documents(source):
    if source.startswith("wikipedia:"):
        _, lang, millions = source.split(":")
        return from_wikipedia(lang, float(millions) * 1e6)
    return from_folder(source)


# ---------- 2. the vocabulary, and the texts as numbers ----------

def prepare(d, source, stage):
    """Three passes, so even a billion words never sit in memory at once:
    the texts go to a plain file (one paragraph per line, an empty line between documents);
    in stage 1 the vocabulary is learned from it; then it's turned into numbers, written to disk as it goes."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    f = names(stage)
    corpus = d / f"corpus{stage}.txt"
    words, docs = 0, 0
    with open(corpus, "w", encoding="utf-8") as out:
        for paras in documents(source):
            if paras:
                out.write("".join(" ".join(p.split()) + "\n" for p in paras) + "\n")
                words += sum(len(p.split()) for p in paras)
                docs += 1
                if docs % 20000 == 0:
                    say(f"{docs:,} documents · {words:,} words")
    if words < 50_000:
        corpus.unlink(missing_ok=True)
        raise SystemExit(f"Only {words:,} words: a model needs at least a few hundred thousand to learn anything.")

    def lines(limit=None):
        n = 0
        with open(corpus, encoding="utf-8") as src:
            for line in src:
                line = line.rstrip("\n")
                if line:
                    yield line
                    n += len(line.split())
                    if limit and n >= limit:
                        return

    if stage == 1:
        say("learning the vocabulary…")
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # starts from bytes: any character, any language
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(lines(40_000_000), trainers.BpeTrainer(  # 40 million words are plenty to choose 8,192 pieces
            vocab_size=VOCAB, special_tokens=[END], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        tok.save(str(d / "tokenizer.json"))
    else:
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))  # the same vocabulary: the model only knows these pieces
    end = np.array([tok.token_to_id(END)], np.uint16)

    say("turning the texts into numbers…")
    n_train = n_val = k = 0
    with open(corpus, encoding="utf-8") as src, open(d / f["train"], "wb") as tr, open(d / f["val"], "wb") as va:
        batch = []

        def flush():
            nonlocal n_train, n_val, k
            for enc in tok.encode_batch(["\n" + p for p in batch]):
                ids = np.array(enc.ids, np.uint16)
                k += 1
                if k % 50 == 0:  # one paragraph in 50 is kept aside, to check the model on text it never studies
                    ids.tofile(va)
                    n_val += len(ids)
                else:
                    ids.tofile(tr)
                    n_train += len(ids)
            batch.clear()

        for line in src:
            line = line.rstrip("\n")
            if not line:  # the end of a document
                flush()
                end.tofile(tr)
                n_train += 1
                continue
            batch.append(line)
            if len(batch) >= 5000:
                before = n_train
                flush()
                if n_train // 5_000_000 != before // 5_000_000:
                    say(f"{n_train:,} tokens")
        flush()
    corpus.unlink()
    label = source if source.startswith("wikipedia:") else pathlib.Path(source).name
    info = {"source": label, "docs": docs, "words": words, "train_tokens": n_train, "val_tokens": n_val, "vocab": tok.get_vocab_size()}
    (d / f["data"]).write_text(json.dumps(info))
    say(f"ready: {words:,} words → {n_train:,} tokens to learn from, {n_val:,} kept aside to check")


# ---------- 3. the model, and 4. training ----------

def config(size, vocab):
    hidden, layers, heads, inter = SIZES[size]
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": hidden, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": heads, "intermediate_size": inter, "vocab_size": vocab,
            "max_position_embeddings": CTX, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": True,
            "hidden_act": "silu", "torch_dtype": "float32", "bos_token_id": 0, "eos_token_id": 0}


def write_model(d, out, model, cfg):
    """The model in Hugging Face's layout, which mlx-lm, llama.cpp's converter and so Ollama all read."""
    from mlx.utils import tree_flatten
    import mlx.core as mx
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    shutil.copy(d / "tokenizer.json", out / "tokenizer.json")
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": END, "eos_token": END,
                                                           "model_max_length": CTX, "clean_up_tokenization_spaces": False}))


def sample(model, tok, prompt, n=60, temp=0.8):
    import mlx.core as mx
    ids = tok.encode(prompt).ids
    for _ in range(n):
        logits = model(mx.array([ids[-CTX:]]))[0, -1]
        nxt = int(mx.random.categorical(logits / temp))
        if nxt == tok.token_to_id(END):
            break
        ids.append(nxt)
    return tok.decode(ids)


def train(d, size, steps, stage=1, mix=0.0):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from functools import partial
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama
    from tokenizers import Tokenizer
    f = names(stage)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    cfg = config(size, tok.get_vocab_size())
    model = llama.Model(llama.ModelArgs(**{k: v for k, v in cfg.items() if k in llama.ModelArgs.__dataclass_fields__}))
    keys = ("train", "val", "old", "samples")
    done, meta = 0, {k: [] for k in keys}
    prog, out = d / f["progress"], d / f["model"]
    if prog.exists() and (out / "model.safetensors").exists():
        p = json.loads(prog.read_text())
        if p.get("size") == size and p["step"] < steps and p.get("mix", mix) == mix:
            model.load_weights(str(out / "model.safetensors"))
            done, meta = p["step"], {k: p.get(k, []) for k in keys}
            say(f"picking up at step {done} of {steps}")
    if stage == 2 and not done:
        if not (d / "model" / "model.safetensors").exists():
            raise SystemExit("Train the model on the first texts before continuing it on others.")
        model.load_weights(str(d / "model" / "model.safetensors"))  # where the first training ended
        shutil.rmtree(d / "stage2", ignore_errors=True)  # a new second stage replaces the old one
        prog.unlink(missing_ok=True)
        say("starting from the model trained on the first texts")
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    say(f"model: {n_params / 1e6:.1f} million parameters · {steps} steps of {BATCH * CTX:,} tokens")

    data = np.memmap(d / f["train"], dtype=np.uint16, mode="r")
    vdata = np.memmap(d / f["val"], dtype=np.uint16, mode="r")
    first = np.memmap(d / "train.bin", dtype=np.uint16, mode="r") if stage == 2 else None
    vfirst = np.memmap(d / "val.bin", dtype=np.uint16, mode="r") if stage == 2 else None
    if len(data) <= CTX + 1:
        raise SystemExit("Not enough text to fill even one window.")
    rng = np.random.default_rng(done + 1000 * stage)
    old_n = round(BATCH * mix) if stage == 2 else 0  # how many windows of each batch still come from the first texts

    def windows(src, n):
        start = rng.integers(0, len(src) - CTX - 1, n)
        return np.stack([src[s:s + CTX + 1] for s in start]).astype(np.int32)

    def batch(src, n=BATCH, old=0):
        w = windows(src, n - old) if not old else np.concatenate([windows(src, n - old), windows(first, old)])
        return mx.array(w[:, :-1]), mx.array(w[:, 1:])  # the target is the same text shifted by one token

    # the second stage starts from a trained model: a gentler learning rate, so it bends rather than breaks
    lr = LR[size] * (0.3 if stage == 2 else 1)
    warm = min(500 if stage == 1 else 100, steps // 10)
    sched = optim.join_schedules([optim.linear_schedule(lr / 100, lr, warm), optim.cosine_decay(lr, max(1, steps - warm), lr / 10)], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.1)
    opt.state["step"] = mx.array(done, mx.uint64)  # so the learning rate continues from the right point

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(m(x), y, reduction="mean")
    grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, g = grad(model, x, y)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    def check(src):
        return sum(loss_fn(model, *batch(src, 16)).item() for _ in range(10)) / 10 if len(src) > CTX + 1 else float("nan")

    # how often to check, save and write a sample: in stage 2, at 8 evenly spaced points, each kept as its own copy
    every = max(10, steps // SNAPSHOTS) if stage == 2 else max(50, min(1000, steps // 20))
    t0, start_step, run = time.time(), done, 0.0
    for i in range(done + 1, steps + 1):
        loss = step(*batch(data, old=old_n))
        mx.eval(state, loss)
        run += loss.item()
        if i % 10 == 0:
            left = (time.time() - t0) / (i - start_step) * (steps - i)
            say(f"Iter {i}: Train loss {run / 10:.3f}, about {int(left // 3600)} h {int(left % 3600 // 60)} min left")
            meta["train"].append((i, round(run / 10, 4)))
            run = 0.0
        if i % every == 0 or i == steps:
            v = check(vdata)
            say(f"Iter {i}: Val loss {v:.3f}")
            meta["val"].append((i, round(v, 4)))
            if stage == 2:  # how well it still knows the first texts: rising means it's forgetting them
                o = check(vfirst)
                meta["old"].append((i, round(o, 4)))
                say("@@OLD@@" + json.dumps([i, round(o, 4)]))
            text = sample(model, tok, "\n")
            meta["samples"] = (meta["samples"] + [(i, text)])[-30:]
            say("@@SAMPLE@@" + json.dumps([i, text]))
            write_model(d, out, model, cfg)
            if stage == 2:
                shutil.copytree(out, d / "stage2" / f"step-{i:06d}", dirs_exist_ok=True)
            prog.write_text(json.dumps({"step": i, "size": size, "steps": steps, "mix": mix, **meta}))
    say("finished")


def try_it(d, prompt, which="model"):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(str(d / which))
    text = generate(model, tok, prompt=prompt, max_tokens=200, sampler=make_sampler(temp=0.8))
    say("@@TRY@@" + json.dumps(prompt + text))


if __name__ == "__main__":
    cmd, d = sys.argv[1], pathlib.Path(sys.argv[2])
    if cmd == "prepare":
        prepare(d, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "train":
        train(d, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1, float(sys.argv[6]) if len(sys.argv) > 6 else 0.0)
    elif cmd == "try":
        try_it(d, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "model")
