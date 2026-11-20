"""Training a small language model from nothing, with Apple's MLX. Run by scratch.py as a separate process:

  python scratch_job.py prepare DIR FOLDER   your texts -> a vocabulary (tokenizer.json) and the texts as numbers (train.bin, val.bin)
  python scratch_job.py train DIR SIZE STEPS  a model with random weights -> trained on train.bin; picks up where it stopped
  python scratch_job.py try DIR PROMPT        the model continues a piece of text

It prints progress in mlx-lm's format ("Iter 10: Train loss 1.234"), so the page reads it like fine-tuning.
"""
import hashlib, json, math, pathlib, sys, time

import numpy as np

HERE = pathlib.Path(__file__).parent
TYPES = {".txt", ".md", ".pdf", ".epub", ".docx", ".odt", ".html", ".htm", ".rtf"}
END = "<|end|>"  # marks where one document ends and the next begins
VOCAB = 8192
CTX = 512  # tokens the model sees at once
BATCH = 8  # windows per step: 8 × 512 = 4,096 tokens. On a 16 GB M4, 32 used 14 GB and was no faster per token

# the three sizes: width of each layer, how many layers, attention heads, width of the middle of each layer
SIZES = {"10M": (256, 8, 4, 1024), "25M": (384, 10, 6, 1280), "50M": (512, 12, 8, 1792)}
LR = {"10M": 1e-3, "25M": 8e-4, "50M": 6e-4}
if __import__("os").environ.get("JOSHUA_SCRATCH_TEST"):  # tiny, for testing the whole path on a computer without Apple's chip
    SIZES, CTX, BATCH = {k: (64, 2, 2, 128) for k in SIZES}, 64, 4


def say(*a):
    print(*a, flush=True)


# ---------- 1. the texts, and 2. the vocabulary ----------

def prepare(d, folder):
    sys.path.insert(0, str(HERE))
    import reader
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    reader.MAX_CHARS = 10**12  # the Reader stops long books early; here we want every word
    files = sorted(p for p in pathlib.Path(folder).expanduser().rglob("*") if p.suffix.lower() in TYPES and p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"No texts found in {folder} (it reads {', '.join(sorted(TYPES))}).")
    docs, seen, words = [], set(), 0
    for i, f in enumerate(files, 1):
        try:
            paras = reader.extract(f)[1]
        except Exception as e:
            say(f"skipped {f.name}: {e}")
            continue
        keep = []
        for p in paras:
            h = hashlib.sha1(p.strip().lower().encode()).digest()
            if len(p) > 1 and h not in seen:  # the same paragraph twice teaches nothing new
                seen.add(h)
                keep.append(p)
        words += sum(len(p.split()) for p in keep)
        docs.append(keep)
        say(f"read {i}/{len(files)} · {f.name} · {words:,} words so far")
    if words < 50_000:
        raise SystemExit(f"Only {words:,} words: a model needs at least a few hundred thousand to learn anything.")

    say("learning the vocabulary…")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # starts from bytes: any character, any language
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator((p for doc in docs for p in doc), trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=[END], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok.save(str(d / "tokenizer.json"))
    end = tok.token_to_id(END)

    say("turning the texts into numbers…")
    train, val = [], []
    for doc in docs:
        for j in range(0, len(doc), 1000):
            chunk = doc[j:j + 1000]
            for k, enc in enumerate(tok.encode_batch(["\n" + p for p in chunk])):
                (val if (j + k) % 50 == 49 else train).append(np.array(enc.ids, np.uint16))  # one paragraph in 50 is kept aside
        train.append(np.array([end], np.uint16))
    train, val = np.concatenate(train), np.concatenate(val or [np.array([end], np.uint16)])
    train.tofile(d / "train.bin")
    val.tofile(d / "val.bin")
    info = {"files": len(files), "words": words, "train_tokens": int(len(train)), "val_tokens": int(len(val)), "vocab": tok.get_vocab_size()}
    (d / "data.json").write_text(json.dumps(info))
    say(f"ready: {words:,} words → {len(train):,} tokens to learn from, {len(val):,} kept aside to check")


# ---------- 3. the model, and 4. training ----------

def config(size, vocab):
    hidden, layers, heads, inter = SIZES[size]
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": hidden, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": heads, "intermediate_size": inter, "vocab_size": vocab,
            "max_position_embeddings": CTX, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": True,
            "hidden_act": "silu", "torch_dtype": "float32", "bos_token_id": 0, "eos_token_id": 0}


def save(d, model, cfg, step, size, meta):
    """The model in Hugging Face's layout, which mlx-lm, llama.cpp's converter and so Ollama all read."""
    from mlx.utils import tree_flatten
    import mlx.core as mx
    out = d / "model"
    out.mkdir(exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "tokenizer.json").write_text((d / "tokenizer.json").read_text())
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": END, "eos_token": END,
                                                           "model_max_length": CTX, "clean_up_tokenization_spaces": False}))
    (d / "progress.json").write_text(json.dumps({"step": step, "size": size, **meta}))


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


def train(d, size, steps):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from functools import partial
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    cfg = config(size, tok.get_vocab_size())
    model = llama.Model(llama.ModelArgs(**{k: v for k, v in cfg.items() if k in llama.ModelArgs.__dataclass_fields__}))
    done, meta = 0, {"train": [], "val": [], "samples": []}
    prog = d / "progress.json"
    if prog.exists() and (d / "model" / "model.safetensors").exists():
        p = json.loads(prog.read_text())
        if p.get("size") == size and p["step"] < steps:
            model.load_weights(str(d / "model" / "model.safetensors"))
            done, meta = p["step"], {k: p.get(k, []) for k in ("train", "val", "samples")}
            say(f"picking up at step {done} of {steps}")
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    say(f"model: {n_params / 1e6:.1f} million parameters · {steps} steps of {BATCH * CTX:,} tokens")

    data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    vdata = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
    if len(data) <= CTX + 1:
        raise SystemExit("Not enough text to fill even one window.")
    rng = np.random.default_rng(done)

    def batch(src, n=BATCH):
        start = rng.integers(0, len(src) - CTX - 1, n)
        w = np.stack([src[s:s + CTX + 1] for s in start]).astype(np.int32)
        return mx.array(w[:, :-1]), mx.array(w[:, 1:])  # the target is the same text shifted by one token

    lr = LR[size]
    warm = min(500, steps // 10)
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

    every = max(50, min(1000, steps // 20))  # check, save and write a sample this often
    t0, start_step, run = time.time(), done, 0.0
    for i in range(done + 1, steps + 1):
        loss = step(*batch(data))
        mx.eval(state, loss)
        run += loss.item()
        if i % 10 == 0:
            left = (time.time() - t0) / (i - start_step) * (steps - i)
            say(f"Iter {i}: Train loss {run / 10:.3f}, about {int(left // 3600)} h {int(left % 3600 // 60)} min left")
            meta["train"].append((i, round(run / 10, 4)))
            run = 0.0
        if i % every == 0 or i == steps:
            v = sum(loss_fn(model, *batch(vdata, 16)).item() for _ in range(10)) / 10 if len(vdata) > CTX + 1 else float("nan")
            say(f"Iter {i}: Val loss {v:.3f}")
            meta["val"].append((i, round(v, 4)))
            text = sample(model, tok, "\n")
            meta["samples"] = (meta["samples"] + [(i, text)])[-30:]
            say("@@SAMPLE@@" + json.dumps([i, text]))
            save(d, model, cfg, i, size, meta)
    say("finished")


def try_it(d, prompt):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(str(d / "model"))
    text = generate(model, tok, prompt=prompt, max_tokens=200, sampler=make_sampler(temp=0.8))
    say("@@TRY@@" + json.dumps(prompt + text))


if __name__ == "__main__":
    cmd, d = sys.argv[1], pathlib.Path(sys.argv[2])
    if cmd == "prepare":
        prepare(d, sys.argv[3])
    elif cmd == "train":
        train(d, sys.argv[3], int(sys.argv[4]))
    elif cmd == "try":
        try_it(d, sys.argv[3])
