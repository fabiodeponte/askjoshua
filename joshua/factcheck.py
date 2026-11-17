"""The fact checker: the journalism desk in goal mode, asked to check a text.

The goal is done only when Joshua's own code agrees: every claim in the table has one of the four
verdicts, and every claim that isn't Unverifiable has a source that isn't the text being checked.
"""
import re
import urllib.parse

# the four verdicts, in the six languages of Joshua; the last of each group needs no source
VERDICTS = {
    "true": "true", "misleading": "misleading", "false": "false", "unverifiable": "unverifiable",
    "vero": "true", "fuorviante": "misleading", "falso": "false", "non verificabile": "unverifiable",
    "vrai": "true", "trompeur": "misleading", "faux": "false", "invérifiable": "unverifiable",
    "wahr": "true", "irreführend": "misleading", "falsch": "false", "nicht überprüfbar": "unverifiable",
    "prawda": "true", "wprowadzające w błąd": "misleading", "fałsz": "false", "nie do zweryfikowania": "unverifiable",
    "属实": "true", "误导": "misleading", "不实": "false", "无法核实": "unverifiable",
}
ASKED = re.compile(r"fact.?check|verifica|vérifie|prüfe|sprawdź|核查", re.I)
LINK = re.compile(r"https?://[^\s)\]>|]+")


def host(url):
    return urllib.parse.urlsplit(url).netloc.lower().removeprefix("www.")


def asked(text):
    return bool(ASKED.search(text or ""))


def rows(answer):
    """The rows of the fact-check table: lines of a Markdown table, without its header and separator."""
    out = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"\|[\s:|-]+\|?", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        out.append(cells)
    return out[1:] if out else []  # the first line is the header


def verdict(cells):
    for c in cells:
        word = re.sub(r"[*_`✅❌⚠️❓🟢🟡🔴⚪]", "", c).strip().lower()
        if word in VERDICTS:
            return VERDICTS[word]
    return None


def gaps(answer, original):
    """What is still missing, as short notes for Joshua; an empty list means the check is done."""
    table = rows(answer)
    if not table:
        return ["There is no fact-check table yet: # | Claim | Verdict | Why | Sources."]
    own = {host(u) for u in LINK.findall(original or "")}  # the sites of the text being checked
    notes = []
    for k, cells in enumerate(table, 1):
        name = cells[0] if cells and cells[0] else str(k)
        v = verdict(cells)
        links = LINK.findall(" ".join(cells))
        if not v:
            notes.append(f"Claim {name}: it has no verdict; use exactly one of the four.")
        elif v != "unverifiable" and not links:
            notes.append(f"Claim {name}: it has no source link; open one and link it, or mark it as unverifiable.")
        elif v != "unverifiable" and own and all(host(u) in own for u in links):
            notes.append(f"Claim {name}: its only source is the text being checked; find an independent one, or mark it as unverifiable.")
    return notes
