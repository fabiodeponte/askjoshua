"""Desks: Joshua set up for one kind of work (journalism first). A desk = its instructions, the tools it uses,
and starter prompts for the empty chat. Everything stays on this computer: desks.json, shipped with Joshua, is the
default; what the user changes in Settings is kept apart in memory/desk-edits.json, so an update brings new desks
and better defaults without ever undoing the user's own words."""
import copy, json, pathlib

HERE = pathlib.Path(__file__).parent
DEFAULT = HERE / "desks.json"
EDITS = HERE / "memory" / "desk-edits.json"


def _read(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def data():
    """The default desks with the user's changes on top."""
    found = copy.deepcopy(_read(DEFAULT)) or {"desks": {}}
    for desk_id, change in _read(EDITS).items():
        if desk_id in found["desks"]:
            found["desks"][desk_id].update(change)
    return found


def desks():
    """The desks in their order, as (id, desk)."""
    return sorted(data()["desks"].items(), key=lambda kv: kv[1].get("order", 99))


def get(desk_id):
    return data()["desks"].get(desk_id or "")


def edited(desk_id):
    return desk_id in _read(EDITS)


def default_instructions(desk_id):
    return _read(DEFAULT).get("desks", {}).get(desk_id, {}).get("instructions", "")


def set_instructions(desk_id, text):
    """Keep the user's own instructions for a desk; the same words as the default count as no change."""
    edits = _read(EDITS)
    if text.strip() and text.strip() != default_instructions(desk_id).strip():
        edits[desk_id] = {"instructions": text.strip()}
    else:
        edits.pop(desk_id, None)
    EDITS.parent.mkdir(exist_ok=True)
    EDITS.write_text(json.dumps(edits, ensure_ascii=False, indent=1), encoding="utf-8")


def text(value, lang):
    """A desk's words in the interface language: a plain string, or {"en": …, "it": …}."""
    if isinstance(value, dict):
        return value.get(lang) or value.get("en") or next(iter(value.values()), "")
    return value or ""


def label(desk_id, lang):
    d = get(desk_id)
    return (d.get("icon", "") + " " + text(d["name"], lang)).strip() if d else ""
