"""Lists every text that has no translation yet, in the program and on the website.

  python3 check_translations.py

Program: every t("...") or tr("...") in joshua/*.py must be a key in joshua/locales/<lang>.json.
Website: every piece of the English pages (site/sitei18n.py extract) must be a key in site/tr/<lang>.json.
Run it after each change; it ends with "All translated." when nothing is missing.
"""
import ast, json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
LANGS = ["it", "fr", "de", "pl", "zh"]


def program_texts():
    found = {}
    for py in sorted((HERE / "joshua").glob("*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("t", "tr")
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                found.setdefault(node.args[0].value, f"{py.name}:{node.lineno}")
    return found


def site_texts():
    subprocess.run([sys.executable, str(HERE / "site" / "sitei18n.py"), "extract"], check=True, capture_output=True)
    texts = json.loads((HERE / "site" / "texts.json").read_text(encoding="utf-8"))
    skip = lambda s: not any(c.isalpha() for c in s.replace("Joshua", "").replace("JOSHUA", ""))  # names and codes only
    return [s for s in texts if not skip(s)]


missing = 0
program = program_texts()
for lang in LANGS:
    table = json.loads((HERE / "joshua" / "locales" / f"{lang}.json").read_text(encoding="utf-8"))
    for text, where in program.items():
        if text not in table:
            missing += 1
            print(f"program {lang}  {where}  {text[:90]!r}")

site = site_texts()
for lang in LANGS:
    table = json.loads((HERE / "site" / "tr" / f"{lang}.json").read_text(encoding="utf-8"))
    gaps = [s for s in site if s not in table]
    gaps += [s for s in site if table.get(s) == s and len(s.split()) > 6 and "<pre" not in s]  # "translated" into English by mistake
    for s in gaps:
        missing += 1
        print(f"website {lang}  {s[:90]!r}")

print("All translated." if not missing else f"{missing} missing.")
