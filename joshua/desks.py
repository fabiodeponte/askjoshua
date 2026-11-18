"""Desks: Joshua set up for one kind of work (journalism first). A desk = its instructions, the tools it uses,
and starter prompts for the empty chat. Everything stays on this computer: desks.json, shipped with Joshua, is the
default; what the user changes in Settings is kept apart in memory/desk-edits.json, so an update brings new desks
and better defaults without ever undoing the user's own words."""
import copy, json, pathlib, re

HERE = pathlib.Path(__file__).parent
DEFAULT = HERE / "desks.json"
EDITS = HERE / "memory" / "desk-edits.json"
MINE = HERE / "memory" / "desks-mine.json"  # desks you made yourself, in Settings


def _read(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def data():
    """The default desks with the user's changes on top."""
    found = copy.deepcopy(_read(DEFAULT)) or {"desks": {}}
    edits = _read(EDITS)
    for desk_id, change in edits.items():
        if desk_id in found["desks"] and isinstance(change, dict):
            found["desks"][desk_id].update(change)
    for desk_id in edits.get("_hidden", []):  # Joshua's desks you removed
        found["desks"].pop(desk_id, None)
    for desk_id, d in _read(MINE).items():
        found["desks"][desk_id] = dict(d, mine=True)
    return found


def mine():
    """Your own desks, as (id, desk)."""
    return [(k, d) for k, d in desks() if d.get("mine")]


def is_mine(desk_id):
    return desk_id in _read(MINE)


def save_mine(desk_id, name, icon, instructions, tools):
    """Creates or changes one of your desks; returns its id."""
    found = _read(MINE)
    if not desk_id:
        base = "my-" + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "desk")
        desk_id, n = base, 2
        while desk_id in found or desk_id in _read(DEFAULT).get("desks", {}):
            desk_id, n = f"{base}-{n}", n + 1
    order = found.get(desk_id, {}).get("order", 50 + len(found))
    found[desk_id] = {"order": order, "icon": icon.strip(), "name": name.strip(), "tools": list(tools), "instructions": instructions.strip(), "starters": []}
    MINE.parent.mkdir(exist_ok=True)
    MINE.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    return desk_id


def delete_mine(desk_id):
    found = _read(MINE)
    found.pop(desk_id, None)
    MINE.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")


def desks():
    """The desks in their order, as (id, desk)."""
    return sorted(data()["desks"].items(), key=lambda kv: kv[1].get("order", 99))


def get(desk_id):
    return data()["desks"].get(desk_id or "")


def edited(desk_id):
    return isinstance(_read(EDITS).get(desk_id), dict)


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


def _write(p, value):
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def default(desk_id):
    return _read(DEFAULT).get("desks", {}).get(desk_id)


def save(desk_id, name, icon, instructions, tools, starters, lang):
    """Saves a desk from Settings: yours entirely; for one of Joshua's, only what differs from its default (in the
    language shown), so the rest keeps its translations and gets better with updates. Returns the desk's id."""
    starters = [s for s in starters if s.get("title", "").strip() and s.get("prompt", "").strip()]
    if not desk_id or is_mine(desk_id):
        desk_id = save_mine(desk_id, name, icon, instructions, tools)
        found = _read(MINE)
        found[desk_id]["starters"] = starters
        _write(MINE, found)
        return desk_id
    d, edits = default(desk_id) or {}, _read(EDITS)
    shown = [{"icon": s.get("icon", ""), "title": text(s["title"], lang), "prompt": text(s["prompt"], lang)} for s in d.get("starters", [])]
    plain = [{k: s.get(k, "") for k in ("icon", "title", "prompt")} for s in starters]
    extras = {text(s["title"], lang): {k: v for k, v in s.items() if k not in ("icon", "title", "prompt")} for s in d.get("starters", [])}
    starters = [dict(extras.get(s["title"], {}), **s) for s in plain]  # a card that turns on goal mode keeps doing so
    change = {}
    if name.strip() and name.strip() != text(d.get("name"), lang):
        change["name"] = name.strip()
    if icon.strip() != d.get("icon", ""):
        change["icon"] = icon.strip()
    if instructions.strip() and instructions.strip() != d.get("instructions", "").strip():
        change["instructions"] = instructions.strip()
    if sorted(tools) != sorted(d.get("tools", [])):
        change["tools"] = list(tools)
    if plain != shown:
        change["starters"] = starters
    if change:
        edits[desk_id] = change
    else:
        edits.pop(desk_id, None)
    _write(EDITS, edits)
    return desk_id


def reset(desk_id):
    edits = _read(EDITS)
    edits.pop(desk_id, None)
    _write(EDITS, edits)


def remove(desk_id):
    """Yours: deleted. Joshua's: hidden, and can be brought back (hidden(), unhide())."""
    if is_mine(desk_id):
        return delete_mine(desk_id)
    edits = _read(EDITS)
    edits["_hidden"] = sorted(set(edits.get("_hidden", [])) | {desk_id})
    _write(EDITS, edits)


def hidden():
    return [k for k in _read(EDITS).get("_hidden", []) if default(k)]


def unhide(desk_id):
    edits = _read(EDITS)
    edits["_hidden"] = [k for k in edits.get("_hidden", []) if k != desk_id]
    _write(EDITS, edits)


def text(value, lang):
    """A desk's words in the interface language: a plain string, or {"en": …, "it": …}."""
    if isinstance(value, dict):
        return value.get(lang) or value.get("en") or next(iter(value.values()), "")
    return value or ""


def label(desk_id, lang):
    d = get(desk_id)
    return (d.get("icon", "") + " " + text(d["name"], lang)).strip() if d else ""
