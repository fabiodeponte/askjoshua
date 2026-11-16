"""Desks: Joshua set up for one kind of work (journalism first). A desk = its instructions, the tools it uses,
and starter prompts for the empty chat. They come from desks.json on ask.joshua.earth, read once a day,
so desks improve without a new Joshua; the copy shipped with Joshua is used when offline."""
import json, pathlib, threading, time

HERE = pathlib.Path(__file__).parent
URL = "https://ask.joshua.earth/desks.json"
_cache = {"at": 0, "data": None}
_lock = threading.Lock()


def data():
    with _lock:
        if _cache["data"] and time.time() - _cache["at"] < 86400:
            return _cache["data"]
        try:
            import requests
            found = requests.get(URL, timeout=6).json()
            assert isinstance(found.get("desks"), dict)
        except Exception:
            found = json.loads((HERE / "desks.json").read_text(encoding="utf-8"))
        _cache.update(at=time.time(), data=found)
        return found


def desks():
    """The desks in their order, as (id, desk)."""
    d = data()["desks"]
    return sorted(d.items(), key=lambda kv: kv[1].get("order", 99))


def get(desk_id):
    return data()["desks"].get(desk_id or "")


def text(value, lang):
    """A desk's words in the interface language: a plain string, or {"en": …, "it": …}."""
    if isinstance(value, dict):
        return value.get(lang) or value.get("en") or next(iter(value.values()), "")
    return value or ""


def label(desk_id, lang):
    d = get(desk_id)
    return (d.get("icon", "") + " " + text(d["name"], lang)).strip() if d else ""
