"""Linux image generation for Joshua: FLUX.2 klein 4B with diffusers on an NVIDIA card.
New picture:  imagegen.py --prompt "..." --out pic.png
Edit photos:  imagegen.py --prompt "..." --images a.jpg b.jpg --out pic.png   (the photos are references)"""
import argparse, torch
from diffusers import Flux2KleinPipeline
from PIL import Image

a = argparse.ArgumentParser()
a.add_argument("--prompt", required=True); a.add_argument("--out", required=True)
a.add_argument("--width", type=int, default=1024); a.add_argument("--height", type=int, default=1024)
a.add_argument("--images", nargs="*", default=[])
a = a.parse_args()
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
if torch.cuda.get_device_properties(0).total_memory / 2**30 >= 20:
    pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload()  # keeps only the part in use on the card: fits about 12 GB, a bit slower
refs = {"image": [Image.open(p).convert("RGB") for p in a.images]} if a.images else {}
pipe(a.prompt, width=a.width, height=a.height, guidance_scale=1.0, num_inference_steps=4, **refs).images[0].save(a.out)
