"""Linux image generation for Joshua: FLUX.2 klein 4B with diffusers on an NVIDIA card.
New picture:  imagegen.py --prompt "..." --out pic.png
Edit photos:  imagegen.py --prompt "..." --images a.jpg b.jpg --out pic.png   (the photos are references)"""
import argparse, gc, os, torch
from diffusers import Flux2KleinPipeline
from PIL import Image

REPO = os.environ.get("JOSHUA_IMAGE_REPO", "black-forest-labs/FLUX.2-klein-4B")
GPU = "cuda" if torch.cuda.is_available() else "cpu"
a = argparse.ArgumentParser()
a.add_argument("--prompt", required=True); a.add_argument("--out", required=True)
a.add_argument("--width", type=int, default=1024); a.add_argument("--height", type=int, default=1024)
a.add_argument("--images", nargs="*", default=[])
a = a.parse_args()

# Two turns, so only one big model is in memory at a time (about 8 GB each):
# 1. the text model alone turns the prompt into numbers
enc = Flux2KleinPipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(GPU)
with torch.no_grad():
    embeds, _ = enc.encode_prompt(prompt=a.prompt, device=GPU)
del enc; gc.collect(); torch.cuda.empty_cache() if GPU == "cuda" else None
# 2. the image model alone draws from those numbers
pipe = Flux2KleinPipeline.from_pretrained(REPO, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16).to(GPU)
refs = {"image": [Image.open(p).convert("RGB") for p in a.images]} if a.images else {}
pipe(prompt_embeds=embeds, width=a.width, height=a.height, guidance_scale=1.0, num_inference_steps=4, **refs).images[0].save(a.out)
