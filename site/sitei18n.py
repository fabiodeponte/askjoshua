"""The website in several languages.

  python3 sitei18n.py extract            → texts.json: every piece of text on the English pages, in order
  python3 sitei18n.py build OUT_DIR      → OUT_DIR/<lang>/index.html, api.html, manifesto.html (English stays at the top)

A "piece" is a paragraph, heading, list item or table cell, kept whole with its links and bold text,
so it is translated as one sentence. Translations live in tr/<lang>.json, keyed by the English piece.
Code, terminal examples and commands are never translated.
"""
import json, re, sys, pathlib
from bs4 import BeautifulSoup, NavigableString, Comment

HERE = pathlib.Path(__file__).parent
PAGES = ["index", "features", "install", "api", "manifesto", "about"]
LANGS = {"en": "English", "it": "Italiano", "fr": "Français", "de": "Deutsch", "pl": "Polski", "zh": "中文"}
INLINE = {"a", "b", "strong", "em", "i", "code", "span", "br", "small", "kbd", "sup", "sub", "mark", "abbr"}
SKIP = {"pre", "script", "style", "code", "svg", "title"}
ATTRS = ["alt", "title", "aria-label", "placeholder", "data-text"]
SCRIPT_TEXTS = ["Copied", "Selected", "Copy"]


VERSION = re.compile(r"20\d\d\.\d\d\.\d\d\.\d+")


def norm(s):
    """One line, and the version number replaced by {VERSION}, so a new release doesn't need new translations."""
    return VERSION.sub("{VERSION}", re.sub(r"\s+", " ", s).strip())


def has_words(s):
    return re.search(r"[A-Za-z]{2,}", s) is not None


def is_block(el):
    return el.name not in INLINE


def leaf_block(el):
    return not any(is_block(d) for d in el.find_all(True))


def units(soup):
    """The elements whose contents are translated as one piece, in page order."""
    found = []
    for text in soup.find_all(string=True):
        if isinstance(text, Comment) or not has_words(text) or any(p.name in SKIP for p in text.parents):
            continue
        el, top_inline = text.parent, None
        while el is not None and el.name in INLINE:
            top_inline, el = el, el.parent
        if el is None:
            continue
        unit = el if leaf_block(el) else top_inline
        if unit is None:  # loose text directly inside a container: wrap it so it can be translated
            span = soup.new_tag("span")
            text.wrap(span)
            unit = span
        if not any(unit is f for f in found):  # identity: two cells with the same text are still two cells
            found.append(unit)
    return found


def attr_texts(soup):
    out = []
    for el in soup.find_all(True):
        if any(p.name in ("pre", "script") for p in el.parents):
            continue
        for a in ATTRS:
            if el.get(a) and has_words(el[a]):
                out.append((el, a))
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        out.append((meta, "content"))
    return out


def load(page):
    return BeautifulSoup((HERE / f"{page}.html").read_text(encoding="utf-8"), "html.parser")


def extract():
    texts = []
    for page in PAGES:
        soup = load(page)
        if soup.title:
            texts.append(norm(soup.title.string))
        texts += [norm(u.decode_contents()) for u in units(soup)]
        texts += [norm(el[a]) for el, a in attr_texts(soup)]
    texts += SCRIPT_TEXTS + ["Language"]
    texts = list(dict.fromkeys(t for t in texts if t))
    (HERE / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(texts), "pieces,", sum(len(t.split()) for t in texts), "words")


def switcher(lang, page):
    """The language menu in the header: a link to the same page in each language."""
    path = "" if page == "index" else f"{page}.html"
    links = []
    for code, name in LANGS.items():
        href = (f"/{path}" if code == "en" else f"/{code}/{path}")
        cur = " aria-current='true'" if code == lang else ""
        links.append(f"<a href='{href}' hreflang='{code}' lang='{code}' data-lang='{code}'{cur}>{name}</a>")
    return "<div class='langs' role='navigation' aria-label='{}'>" + "".join(links) + "</div>"


SWITCH_CSS = """
.langs { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: .9rem; margin: 6px 0 0; }
.langs a { color: inherit; opacity: .6; text-decoration: none; }
.langs a[aria-current] { opacity: 1; text-decoration: underline; text-underline-offset: 3px; }
.langs a:hover { opacity: 1; }
"""
# first visit: open the page in the browser's language; a language picked from the menu is remembered
SWITCH_JS = """<script>
(function () {
  var langs = %s, here = %s;
  document.querySelectorAll('.langs a').forEach(function (a) {
    a.addEventListener('click', function () { try { localStorage.setItem('lang', a.dataset.lang) } catch (e) {} });
  });
  var want; try { want = localStorage.getItem('lang') } catch (e) {}
  if (!want) {  // never picked: only the English pages follow the browser; a link to a translated page stays as it is
    if (here !== 'en') return;
    var nav = (navigator.language || 'en').slice(0, 2).toLowerCase(); want = langs.indexOf(nav) >= 0 ? nav : 'en';
  }
  if (want !== here && langs.indexOf(want) >= 0) {
    var a = document.querySelector('.langs a[data-lang="' + want + '"]');
    if (a) location.replace(a.getAttribute('href') + location.hash);
  }
})();
</script>"""


def build(out):
    out = pathlib.Path(out)
    for lang in LANGS:
        table = {} if lang == "en" else json.loads((HERE / "tr" / f"{lang}.json").read_text(encoding="utf-8"))
        version = (VERSION.search((HERE / "index.html").read_text()) or [""])[0]
        tr = lambda s: (table.get(norm(s)) or norm(s)).replace("{VERSION}", version)
        for page in PAGES:
            soup = load(page)
            if lang != "en":
                if soup.title:
                    soup.title.string = tr(soup.title.string)
                for u in units(soup):
                    new = tr(u.decode_contents())
                    u.clear()
                    for node in list(BeautifulSoup(new, "html.parser").contents):
                        u.append(node)
                for el, a in attr_texts(soup):
                    el[a] = tr(el[a])
                for s in soup.find_all("script"):
                    if s.string:
                        txt = s.string
                        for w in SCRIPT_TEXTS:
                            txt = txt.replace(f'"{w}"', json.dumps(tr(w), ensure_ascii=False))
                        s.string = txt
                for el in soup.find_all(src=True):  # pictures live once, at the top
                    if el["src"].startswith("img/"):
                        el["src"] = "../" + el["src"]
                for el in soup.find_all(href=True):
                    if el["href"].startswith("img/"):
                        el["href"] = "../" + el["href"]
            html_tag = soup.find("html")
            if html_tag:
                html_tag["lang"] = lang
            head = soup.find("head") or soup
            for code in LANGS:  # tells search engines about the other languages
                path = "" if page == "index" else f"{page}.html"
                link = soup.new_tag("link", rel="alternate", hreflang=code, href=f"https://ask.joshua.earth/{'' if code == 'en' else code + '/'}{path}")
                head.append(link)
            style = soup.find("style")
            style.append(SWITCH_CSS)
            nav = soup.find("nav")
            nav.append(BeautifulSoup(switcher(lang, page).replace("{}", tr("Language")), "html.parser"))
            body = soup.find("body") or soup
            body.append(BeautifulSoup(SWITCH_JS % (json.dumps(list(LANGS)), json.dumps(lang)), "html.parser"))
            dest = out / ("" if lang == "en" else lang) / f"{page}.html"
            dest.parent.mkdir(parents=True, exist_ok=True)
            opening = "" if soup.find("html") else f'<html lang="{lang}">\n'  # the page's language, for the browser, search engines and screen readers
            dest.write_text('<!doctype html>\n' + opening + '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n' + str(soup), encoding="utf-8")
    print("built", len(LANGS) * len(PAGES), "pages in", out)


if __name__ == "__main__":
    {"extract": lambda: extract(), "build": lambda: build(sys.argv[2])}[sys.argv[1]]()
