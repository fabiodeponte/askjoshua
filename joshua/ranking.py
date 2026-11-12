"""How capable each model is, roughly, so the Models page can list them from best to least.

The scores live in models.json on ask.joshua.earth, so they can be updated when new models come out without a new
version of Joshua. A copy ships with Joshua for when there's no internet.
"""
import json, math, pathlib, re, time

HERE = pathlib.Path(__file__).parent
URL = "https://ask.joshua.earth/models.json"
_cache = {"at": 0, "data": None}


def data():
    if _cache["data"] and time.time() - _cache["at"] < 86400:  # read once a day
        return _cache["data"]
    try:
        import requests
        found = requests.get(URL, timeout=8).json()
    except Exception:
        found = json.loads((HERE / "models.json").read_text())
    _cache.update(at=time.time(), data=found)
    return found


def params(tag, gb=None):
    """Billions of parameters, from the tag (qwen3:8b → 8, qwen3:30b-a3b → 30, gemma3n:e4b → 4, smollm2:135m → 0.135) or the file size."""
    m = re.search(r":e?(\d+(?:\.\d+)?)([bm])", tag)
    if m:
        return float(m[1]) / (1000 if m[2] == "m" else 1)
    return gb / 0.6 if gb else None  # about 0.6 GB per billion parameters, at the size Ollama usually downloads


def local_score(tag, gb=None):
    """A family's score at its reference size, a bit lower for smaller versions and a bit higher for bigger ones."""
    family = data()["families"].get(tag.split(":")[0])
    p = params(tag, gb)
    if not family:  # a model the list doesn't know yet: only its size says something
        return round(0.3 * math.log2(1 + (p or 1)), 1)
    return round(family["score"] * ((p or family["size"]) / family["size"]) ** 0.35, 1)


def _match(name, table):
    """The longest name in the table that this model's name starts with or contains: glm-5.3-flash before glm-5.3."""
    name = name.lower()
    hits = [k for k in table if name.startswith(k) or k in name]
    return table[max(hits, key=len)] if hits else None


def cloud_score(name):
    return _match(name.removesuffix(":cloud").removesuffix("-cloud"), data()["cloud"])


def closed_score(model):
    return _match(model, data()["closed"])
