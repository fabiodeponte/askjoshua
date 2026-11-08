"""Fine-tuning on Linux with an NVIDIA card: the same three jobs mlx-lm does on a Mac.
  tune_cuda.py train BASE DATA_DIR ADAPTER_DIR ITERS LR LAYERS BATCH EVAL_EVERY
  tune_cuda.py try   BASE ADAPTER_DIR PROMPT
  tune_cuda.py fuse  BASE ADAPTER_DIR OUT_DIR
It prints progress in mlx-lm's format ("Iter 10: Train loss 1.234"), so the page reads both the same way."""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

job, base = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(base)
GPU = torch.cuda.is_available()
DEV = "cuda" if GPU else "cpu"


def load(quantized):
    """QLoRA: the base model is loaded in 4 bits to fit small cards; the adapter itself trains in 16 bits."""
    if not quantized or not GPU:  # without a card (only for testing), plain 16-bit on the processor
        return AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"})
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(base, quantization_config=q, device_map={"": 0})


def ids(text):
    return tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(DEV)


if job == "train":
    data, out, iters, lr, layers, batch, every = sys.argv[3], sys.argv[4], *map(float, sys.argv[5:10])
    iters, layers, batch, every = int(iters), int(layers), int(batch), int(every)
    model = load(True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) if GPU else model
    n = model.config.num_hidden_layers
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM", layers_to_transform=list(range(n - layers, n)),
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    train, val = ([json.loads(l)["text"] for l in open(f"{data}/{f}")] for f in ("train.jsonl", "valid.jsonl"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    order, total = [], 0.0

    def val_loss():
        model.eval()
        with torch.no_grad():
            v = sum(model(input_ids=x, labels=x).loss.item() for x in map(ids, val)) / len(val)
        model.train()
        return v

    print(f"Iter 1: Val loss {val_loss():.3f}", flush=True)
    model.train()
    for it in range(1, iters + 1):
        loss = 0
        for _ in range(batch):  # one example at a time, gradients added up: same result as a batch, less memory
            if not order:
                order = random.sample(train, len(train))
            x = ids(order.pop())
            l = model(input_ids=x, labels=x).loss / batch
            l.backward()
            loss += l.item()
        opt.step(), opt.zero_grad()
        total += loss
        if it % 10 == 0:
            print(f"Iter {it}: Train loss {total / 10:.3f}", flush=True)
            total = 0.0
        if it % every == 0 or it == iters:
            print(f"Iter {it}: Val loss {val_loss():.3f}", flush=True)
    model.save_pretrained(out)
    print("Saved the adapter.", flush=True)

elif job == "try":
    adapters, prompt = sys.argv[3], sys.argv[4]
    model = load(True)
    for label in ("BEFORE", "AFTER"):
        if label == "AFTER":
            model = PeftModel.from_pretrained(model, adapters)
        x = ids(prompt)
        y = model.generate(x, max_new_tokens=500, do_sample=False)
        print(f"@@{label}@@" + json.dumps(tok.decode(y[0][x.shape[1]:], skip_special_tokens=True)), flush=True)

elif job == "fuse":
    adapters, out = sys.argv[3], sys.argv[4]
    model = PeftModel.from_pretrained(load(False), adapters).merge_and_unload()  # in ordinary memory, 16 bits: about 8 GB
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("Merged.", flush=True)
