"""Joshua's interface in the computer's language.

The texts are written in English in the code, wrapped in t("..."). Each translation lives in
locales/<language>.json, keyed by the English text, so adding a language means adding one file.
A text with no translation stays in English.

The language is the one chosen in Joshua (stored in memory/settings.json), or else the computer's:
on a Mac the first of System Settings → General → Language & Region, on Linux the LANG variable.
"""
import json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
LANGS = {"en": "English", "it": "Italiano", "fr": "Français", "de": "Deutsch", "pl": "Polski", "zh": "中文"}
SETTINGS = HERE / "memory" / "settings.json"


def chosen():
    """The language picked in Joshua's menu, or 'auto'."""
    try:
        return json.loads(SETTINGS.read_text()).get("language", "auto")
    except Exception:
        return "auto"


def choose(lang):
    data = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    data["language"] = lang
    SETTINGS.parent.mkdir(exist_ok=True)
    SETTINGS.write_text(json.dumps(data))


def system_language():
    """Two letters, like 'it', from the computer's own settings."""
    found = []
    if sys.platform == "darwin":
        try:  # prints e.g. ( "it-IT", "en-US" ): the user's languages, in order of preference
            out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True, timeout=5).stdout
            found += [w.strip(' "\n,') for w in out.strip("()\n ").split(",")]
        except Exception:
            pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):  # e.g. it_IT.UTF-8
        found.append(os.environ.get(var, ""))
    for code in found:
        code = code.lower()[:2]
        if code in LANGS:
            return code
    return "en"


LANG = chosen() if chosen() in LANGS else system_language()
TABLE = {}
if LANG != "en":
    try:
        TABLE = json.loads((HERE / "locales" / f"{LANG}.json").read_text(encoding="utf-8"))
    except Exception:
        TABLE = {}


def t(text, **values):
    """The text in the interface language; {name} placeholders are filled from values."""
    out = TABLE.get(text) or text
    return out.format(**values) if values else out
