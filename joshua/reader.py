"""The Reader: books and documents read aloud, keeping your place.

A document is turned into a list of paragraphs once, when you add it, and kept in memory/reader/<id>.json
with the paragraph you got to. The id comes from the file's content, so adding the same file again finds your place."""
import hashlib
import html
import json
import pathlib
import re
import time
import zipfile

import docx
import lxml.etree
import lxml.html
from pypdf import PdfReader

LIB = pathlib.Path(__file__).parent / "memory" / "reader"
TYPES = [".pdf", ".epub", ".docx", ".odt", ".txt", ".md", ".html", ".htm", ".rtf"]
MAX_CHARS = 3_000_000


# ---------- turning a file into paragraphs ----------

def _clean(text):
    return re.sub(r"\s+", " ", text.replace("\xad", "")).strip()


def _blocks(text):
    """Paragraphs of plain text: separated by an empty line."""
    return [_clean(b) for b in re.split(r"\n\s*\n", text) if _clean(b)]


def _pdf_lines(page):
    """The lines of a PDF page from top to bottom, each with the gap above it. pypdf hands over the text in pieces
    with their position; pieces at the same height make one line."""
    pieces = []
    page.extract_text(visitor_text=lambda text, cm, tm, font, size: pieces.append((tm[5] * (cm[3] or 1) + cm[5], tm[4], text)) if text.strip() else None)
    rows = []
    for y, x, text in pieces:
        row = next((r for r in rows if abs(r[0] - y) < 2), None)
        if row:
            row[1].append((x, text))
        else:
            rows.append([y, [(x, text)]])
    rows.sort(key=lambda r: -r[0])
    out, last = [], None
    for y, parts in rows:
        line = re.sub(r"\s+", " ", "".join(t for _, t in sorted(parts, key=lambda p: p[0]))).strip()
        if line:
            out.append((line, (last - y) if last is not None else None))
            last = y
    return out


def _pdf(path):
    """PDF text comes as lines cut at the page's width: they're joined back into paragraphs. A new paragraph starts
    after a gap clearly wider than the usual one between lines (headings and paragraph spacing), or after a short line
    that ends a sentence. Page numbers and words split with a hyphen at the end of a line are mended."""
    paras, cur = [], ""
    for page in PdfReader(str(path)).pages:
        lines = [(l, g) for l, g in _pdf_lines(page) if not re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", l)]
        if not lines:
            continue
        gaps = sorted(g for _, g in lines if g)
        usual = gaps[len(gaps) // 2] if gaps else 0
        width = sorted(len(l) for l, _ in lines)[int(len(lines) * 0.8)]
        for k, (l, gap) in enumerate(lines):
            if cur and (gap is None and re.search(r"[.!?…:»”\"')]$", cur) or gap and usual and gap > usual * 1.35):
                paras.append(cur)
                cur = ""
            cur = cur[:-1] + l if cur.endswith("-") and l[:1].islower() else (cur + " " + l).strip()
            if len(l) < width * 0.75 and re.search(r"[.!?…:»”\"')]$", l):
                paras.append(cur)
                cur = ""
    if cur:
        paras.append(cur)
    return [_clean(p) for p in paras if _clean(p)]


BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dd", "dt", "figcaption", "td"}


def _html_paras(markup):
    """The paragraphs of a web page or an EPUB chapter, in order. Text outside paragraphs (a div with only text) counts too."""
    try:
        root = lxml.html.fromstring(markup)
    except Exception:
        return []
    for bad in root.xpath("//script|//style|//nav|//head"):
        bad.drop_tree()
    out = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.split("}")[-1].lower()
        inner = {c.tag.split("}")[-1].lower() for c in el.iterdescendants() if isinstance(c.tag, str)}
        if tag in BLOCK and not (inner & (BLOCK - {"td"})):
            out.append(_clean(el.text_content()))
        elif tag == "div" and not (inner & (BLOCK | {"div"})):
            out.append(_clean(el.text_content()))
    return [p for p in out if p]


def _epub(path):
    with zipfile.ZipFile(path) as z:
        container = lxml.etree.fromstring(z.read("META-INF/container.xml"))
        opf_path = container.xpath("//*[local-name()='rootfile']/@full-path")[0]
        opf = lxml.etree.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        title = (opf.xpath("//*[local-name()='title']/text()") or [""])[0].strip()
        items = {i.get("id"): i.get("href") for i in opf.xpath("//*[local-name()='manifest']/*[local-name()='item']")}
        paras = []
        for ref in opf.xpath("//*[local-name()='spine']/*[local-name()='itemref']/@idref"):
            href = items.get(ref)
            if not href:
                continue
            name = re.sub(r"[^/]+/\.\./", "", base + href.split("#")[0])
            try:
                paras += _html_paras(z.read(name))
            except KeyError:
                continue
    return title, paras


def _odt(path):
    """Headings and paragraphs in the order they're in the document."""
    from odf import opendocument, teletype
    doc = opendocument.load(str(path))
    out = []

    def walk(node):
        for child in node.childNodes:
            if getattr(child, "qname", (None, None))[1] in ("p", "h"):
                out.append(_clean(teletype.extractText(child)))
            elif child.nodeType == child.ELEMENT_NODE:
                walk(child)
    walk(doc.text)
    return [p for p in out if p]


def _rtf(text):
    text = re.sub(r"\{\\(?:fonttbl|colortbl|stylesheet|info|\*)(?:[^{}]|\{[^{}]*\})*\}", "", text)  # tables of fonts and colours aren't text
    text = re.sub(r"\\par[d]?\b", "\n\n", text)
    text = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode("cp1252", "replace"), text)
    text = re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", text)
    return _blocks(text)


def extract(path):
    """(title, paragraphs) of a document."""
    path = pathlib.Path(path)
    ext = path.suffix.lower()
    title = ""
    if ext == ".pdf":
        paras = _pdf(path)
        try:
            title = (PdfReader(str(path)).metadata or {}).get("/Title") or ""
        except Exception:
            pass
    elif ext == ".epub":
        title, paras = _epub(path)
    elif ext == ".docx":
        d = docx.Document(str(path))
        paras = [_clean(p.text) for p in d.paragraphs if _clean(p.text)]
        title = d.core_properties.title or ""
    elif ext == ".odt":
        paras = _odt(path)
    elif ext in (".html", ".htm"):
        paras = _html_paras(path.read_bytes())
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if ext == ".rtf":
            paras = _rtf(text)
        else:
            if ext == ".md":
                text = re.sub(r"^#+\s*|[*_`]|^\s*[-+]\s+|!?\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", text, flags=re.M)
            paras = _blocks(text)
    out, size = [], 0
    for p in paras:
        if size > MAX_CHARS:
            break
        out.append(p)
        size += len(p)
    title = str(title).strip()
    if not title or len(title) > 120 or re.fullmatch(r"(untitled|microsoft word.*|\W*)", title, flags=re.I):
        title = re.sub(r"[_-]+", " ", path.stem).strip()
    return title, out


def language(paras):
    try:
        from langdetect import detect
        return detect(" ".join(paras[:60])[:4000])[:2]
    except Exception:
        return ""


# ---------- the library ----------

def _file(book_id):
    return LIB / f"{re.sub(r'[^a-z0-9]', '', book_id)}.json"


def add(path, name=None):
    """Adds a document to the library and returns its id. The same file added again keeps its place."""
    path = pathlib.Path(path)
    book_id = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    if _file(book_id).exists():
        return book_id
    title, paras = extract(path)
    if not paras:
        raise ValueError("no text")
    LIB.mkdir(parents=True, exist_ok=True)
    book = {"id": book_id, "title": title, "file": name or path.name, "lang": language(paras), "paras": paras, "pos": 0,
            "added": time.time(), "opened": time.time()}
    _file(book_id).write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
    return book_id


def get(book_id):
    try:
        return json.loads(_file(book_id).read_text(encoding="utf-8"))
    except Exception:
        return None


def books():
    """The library, the last opened first, without the text."""
    out = []
    for f in LIB.glob("*.json") if LIB.exists() else []:
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        b["count"] = len(b.pop("paras"))
        out.append(b)
    return sorted(out, key=lambda b: -b.get("opened", 0))


def set_pos(book_id, pos):
    book = get(book_id)
    if book:
        book["pos"] = max(0, min(int(pos), len(book["paras"]) - 1))
        book["opened"] = time.time()
        _file(book_id).write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")


def remove(book_id):
    _file(book_id).unlink(missing_ok=True)


def percent(pos, count):
    return round(100 * pos / max(count - 1, 1)) if count > 1 else 0


def page(book):
    """The reading view: every paragraph, numbered, with the place to continue from. The browser does the reading."""
    paras = "".join(f"<p data-i='{i}'>{html.escape(p)}</p>" for i, p in enumerate(book["paras"]))
    return (f"<div id='rdbook' data-id='{book['id']}' data-lang='{html.escape(book.get('lang') or '')}' data-pos='{book.get('pos', 0)}' "
            f"data-count='{len(book['paras'])}'><h2 class='rdtitle'>{html.escape(book['title'])}</h2>{paras}</div>")
