"""Teams of agents: members with a role, their own instructions, model, tools and goal mode, one of them the lead.
Kept in memory/teams.json, on this computer; updates never touch it."""
import json
import pathlib
import re
import secrets

FILE = pathlib.Path(__file__).parent / "memory" / "teams.json"
MAX_MEMBERS = 8


def all():
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def get(team_id):
    return next((t for t in all() if t["id"] == team_id), None) if team_id else None


def _write(teams):
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(teams, ensure_ascii=False, indent=1), encoding="utf-8")


def new_id(name, taken=()):
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "team"
    return base if base not in taken else f"{base}-{secrets.token_hex(2)}"


def save(team):
    """Adds the team, or replaces the one with the same id."""
    teams = [t for t in all() if t["id"] != team.get("id")]
    team["id"] = team.get("id") or new_id(team["name"], {t["id"] for t in teams})
    for m in team["members"]:
        m["id"] = m.get("id") or secrets.token_hex(3)
    if team.get("lead") not in {m["id"] for m in team["members"]}:
        team["lead"] = team["members"][0]["id"]
    _write(teams + [team])
    return team["id"]


def delete(team_id):
    _write([t for t in all() if t["id"] != team_id])


def lead(team):
    return next(m for m in team["members"] if m["id"] == team["lead"])


def others(team):
    return [m for m in team["members"] if m["id"] != team["lead"]]


def find(team, name):
    """A member by name, as the lead writes it (case and spaces don't matter)."""
    key = re.sub(r"\W+", "", (name or "").lower())
    return next((m for m in others(team) if re.sub(r"\W+", "", m["name"].lower()) == key), None) \
        or next((m for m in others(team) if key and key in re.sub(r"\W+", "", (m["name"] + m.get("role", "")).lower())), None)
