import tempfile
import datetime, difflib, json, os, pathlib, re, shutil, subprocess, sys, threading, time
import docx, lxml.html, ollama, requests, trafilatura
from langdetect import DetectorFactory, detect
DetectorFactory.seed = 0
from fpdf import FPDF
from pypdf import PdfReader
from ddgs import DDGS

CONFIG_FILE = pathlib.Path(__file__).parent / "config.json"
CONFIG = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
MODEL = CONFIG.get("model", "qwen3-vl")  # thinking version, used when deep thinking is on
FAST = CONFIG.get("fast", "qwen3-vl:8b-instruct")  # instruct version, the default
MAC = sys.platform == "darwin"
WIN = sys.platform == "win32"


def total_ram():
    """The computer's memory in GB, on macOS, Linux and Windows."""
    if WIN:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                        ("pt", ctypes.c_ulonglong), ("pf", ctypes.c_ulonglong), ("vt", ctypes.c_ulonglong), ("vf", ctypes.c_ulonglong), ("ve", ctypes.c_ulonglong)]
        m = M(); m.len = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.total / 2**30
    if MAC:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0) / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
HERE = pathlib.Path(__file__).parent
# A font with accents and non-Latin letters for PDFs: Arial Unicode on a Mac, DejaVu Sans on Linux.
FONT = next((f for f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"] if os.path.exists(f)), None)
# Fonts for PDFs with bold and italics: Arial on Mac and Windows, DejaVu on Linux, plus a wide font for symbols they lack.
_FONT_DIR, _FONT_FILES = {
    "darwin": ("/System/Library/Fonts/Supplemental/", ["Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf", "Courier New.ttf"]),
    "win32": ("C:/Windows/Fonts/", ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf", "cour.ttf"]),
}.get(sys.platform, ("/usr/share/fonts/truetype/dejavu/", ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf", "DejaVuSansMono.ttf"]))
PDF_FONTS = [_FONT_DIR + f for f in _FONT_FILES] if all(os.path.exists(_FONT_DIR + f) for f in _FONT_FILES) else None  # regular, bold, italic, bold italic, code
# for letters and symbols the main font lacks: arrows and maths, Greek, Cyrillic, Chinese, Japanese, Korean
PDF_FALLBACKS = [f for f in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Apple Symbols.ttf",
                             "C:/Windows/Fonts/seguisym.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/YuGothM.ttc",
                             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                             "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"] if os.path.exists(f)]
DEFAULT_WORK = (pathlib.Path(__file__).parent / "workspace").resolve()
CTX = threading.local()


def set_workspace(path, net=True, timeout=120, region="wt-wt"):
    CTX.work = pathlib.Path(path).resolve()
    CTX.work.mkdir(parents=True, exist_ok=True)
    CTX.net, CTX.timeout, CTX.region = net, timeout, region


REGIONS = {"it": "it-it", "en": "us-en", "fr": "fr-fr", "de": "de-de", "es": "es-es", "pt": "pt-pt", "pl": "pl-pl", "nl": "nl-nl", "ru": "ru-ru", "uk": "ua-uk", "zh-cn": "cn-zh", "zh-tw": "tw-tzh", "ja": "jp-jp", "ko": "kr-kr", "sv": "se-sv", "da": "dk-da", "no": "no-no", "fi": "fi-fi", "el": "gr-el", "tr": "tr-tr", "ar": "xa-ar", "cs": "cz-cs", "hu": "hu-hu", "ro": "ro-ro"}


LANGUAGES = {"it": "Italian", "en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "pl": "Polish", "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "zh-cn": "Chinese", "zh-tw": "Chinese", "ja": "Japanese", "ko": "Korean", "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "el": "Greek", "tr": "Turkish", "ar": "Arabic", "cs": "Czech", "hu": "Hungarian", "ro": "Romanian"}


def lang_for(text, previous=None):
    """Language code of the user's message; short messages keep the previous one."""
    if len(text.split()) < 4:
        return previous
    try:
        return detect(text)
    except Exception:
        return previous


def region_for(text, previous="wt-wt"):
    """Search region matching the language of the user's message; short messages keep the previous one."""
    lang = lang_for(text)
    return REGIONS.get(lang, "wt-wt") if lang else previous


def _work():
    if not hasattr(CTX, "work"):
        set_workspace(DEFAULT_WORK)
    return CTX.work


def _path(name):
    p = (_work() / name).resolve()
    p.relative_to(_work())
    return p


def search_results(query, region=None, n=5):
    region = region or getattr(CTX, "region", "wt-wt")
    for backend in ("yahoo, duckduckgo, brave, startpage", "auto"):
        try:
            return DDGS().text(query, region=region, backend=backend, max_results=n)
        except Exception:
            continue
    return []


def web_search(query: str) -> str:
    """Search the web and return the top 5 results with title, link and snippet."""
    results = search_results(query)
    return str(results) if results else "No results found. Try rephrasing the query, or search again in a moment."


def _tables(raw):
    out = []
    for table in lxml.html.fromstring(raw).iter("table"):
        rows = [" | ".join(" ".join(t.strip() for t in cell.itertext() if t.strip()) for cell in row.iter("td", "th")) for row in table.iter("tr")]
        rows = [r for r in rows if r.strip(" |")]
        if len(rows) > 1:
            out.append("\n".join(rows[:80]))
    return "\n\n".join(out[:5])


def read_page(url: str) -> str:
    """Download a web page and return its main text, plus any tables on it."""
    raw = trafilatura.fetch_url(url)
    if not raw:
        return "Could not download the page."
    text = trafilatura.extract(raw, include_tables=False) or ""
    try:
        tables = _tables(raw)
    except Exception:
        tables = ""
    return ((text[:7000] or "No main text found.") + (f"\n\nTables on the page:\n{tables[:6000]}" if tables else ""))


def run_python(code: str) -> str:
    """Run Python code in a sandbox, with the workspace folder as the current directory, and return what it prints. It can only write files in the workspace folder."""
    work = _work()
    note = ""
    if MAC:  # macOS's built-in sandbox: the code may write only in the workspace (and temp files)
        net = "" if CTX.net else "(deny network*)"
        profile = f'(version 1)(allow default){net}(deny file-write*)(allow file-write* (subpath "{work}") (subpath "/private/var/folders") (literal "/dev/null"))'
        cmd = ["sandbox-exec", "-p", profile, sys.executable, "-c", code]
    elif shutil.which("bwrap"):  # Linux: bubblewrap shows the code a read-only copy of the system, with only the workspace writable
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", str(work), str(work),
               "--die-with-parent", "--unshare-pid"] + ([] if CTX.net else ["--unshare-net"]) + [sys.executable, "-c", code]
    elif WIN:  # Windows has no simple sandbox: the code runs as you, with a note saying so
        cmd, note = [sys.executable, "-c", code], "\n(Note: on Windows this code runs without a sandbox.)"
    else:
        cmd, note = [sys.executable, "-c", code], "\n(Note: bubblewrap is not installed, so this code ran without a sandbox.)"
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=CTX.timeout, env=dict(os.environ, MPLCONFIGDIR=tempfile.gettempdir(), PYTHONIOENCODING="utf-8"))
    except subprocess.TimeoutExpired:
        return f"Stopped: the code ran longer than {CTX.timeout} seconds."
    return ((r.stdout + r.stderr)[-4000:] or "(no output)") + note


def list_files(folder: str = ".") -> str:
    """List the files and subfolders in the workspace folder, or in a subfolder of it."""
    items = sorted(f"{x.name}/" if x.is_dir() else x.name for x in _path(folder).iterdir() if not x.name.startswith("."))
    return "\n".join(items[:300]) or "(empty)"


def install_package(name: str) -> str:
    """Install a Python package from PyPI so that code run with run_python can use it."""
    _work()
    if not CTX.net:
        return "Internet access is turned off, so packages can't be installed."
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", name], capture_output=True, text=True, timeout=600)
    return (r.stdout + r.stderr)[-2000:] or f"Installed {name}"


def read_file(name: str) -> str:
    """Read a file from the workspace folder: plain text, Word (.docx), LibreOffice (.odt) or PDF (.pdf)."""
    p = _path(name)
    if p.suffix.lower() == ".docx":
        text = "\n".join(par.text for par in docx.Document(p).paragraphs)
    elif p.suffix.lower() == ".odt":
        from odf.opendocument import load
        from odf import teletype, text as odftext
        from odf.namespaces import TEXTNS
        d = load(str(p))
        lines = []

        def walk(node):  # paragraphs and headings in the order they appear
            for child in node.childNodes:
                if getattr(child, "qname", None) in ((TEXTNS, "p"), (TEXTNS, "h")):
                    lines.append(teletype.extractText(child))
                else:
                    walk(child)
        walk(d.text)
        text = "\n".join(lines)
    elif p.suffix.lower() == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
    else:
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:  # an older file in the computer's own encoding (Windows: cp1252)
            text = raw.decode("cp1252", errors="replace")
    return text[:20000]


def write_file(name: str, content: str) -> str:
    """Write a file in the workspace folder. A name ending in .pdf, .docx (Word) or .odt (LibreOffice) creates a formatted document: write the content in Markdown (# headings, **bold**, *italics*, lists, tables, code blocks)."""
    p = _path(name)
    if p.suffix.lower() == ".docx":
        write_docx(p, content)
    elif p.suffix.lower() == ".odt":
        write_odt(p, content)
    elif p.suffix.lower() == ".pdf":
        write_pdf(p, content)
    else:
        p.write_text(content, encoding="utf-8")
    return f"Saved {name}"


def _clean(text):
    """Text ready for a document: unusual hyphens and spaces made plain, control characters (which break Word files) removed."""
    text = re.sub("[\u2010\u2011\u00ad]", "-", text)
    text = re.sub("[\u202f\u2009\u200a\u2007]", " ", text).replace("\u2060", "").replace("\ufeff", "")
    text = re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # nested list items indented by 2 spaces (as most models write them): Markdown wants 4
    return re.sub(r"^( +)(?=(?:[-*+]|\d+[.)]) )", lambda m: m[1] * 2 if len(m[1]) % 4 else m[1], text, flags=re.M)


def _html(content):
    import markdown
    return markdown.markdown(_clean(content), extensions=["tables", "sane_lists", "fenced_code"])


INK, GREEN = (40, 36, 32), (46, 107, 79)


def write_pdf(p, content):
    """A tidy PDF from Markdown: headings, bold and italics, lists, tables, code."""
    from fpdf.fonts import TextStyle
    html = _html(content)
    pdf = FPDF(format="A4")
    pdf.set_margins(22, 20, 22)
    pdf.set_auto_page_break(True, margin=18)
    family, mono = "Helvetica", "Courier"
    if PDF_FONTS:
        regular, bold, italic, both, code = PDF_FONTS
        for style, f in (("", regular), ("B", bold), ("I", italic), ("BI", both)):
            pdf.add_font("J", style, f)
        pdf.add_font("JM", "", code)
        backups = []
        for k, f in enumerate(PDF_FALLBACKS):
            try:
                pdf.add_font(f"JW{k}", "", f)
                backups.append(f"JW{k}")
            except Exception:
                pass
        if backups:
            pdf.set_fallback_fonts(backups, exact_match=False)
        family, mono = "J", "JM"
        html = re.sub("[\U0001F000-\U0001FAFF\u2600-\u26ff\u2700-\u27bf\ufe0f\u200d]", "", html)  # emoji: PDF fonts can't draw them in colour
    else:  # no font with accents found: the PDF's own fonts only know Western European letters
        html = html.encode("latin-1", "replace").decode("latin-1")
    pdf.set_title(next((l.lstrip("# ").strip() for l in content.splitlines() if l.strip()), p.stem)[:120])
    pdf.add_page()
    pdf.set_font(family, size=10.5)
    pdf.set_text_color(*INK)
    html = re.sub(r"<t([dh])>", r'<t\1 align="left">', html)  # table cells: ragged right, not stretched
    html = html.replace("<table>", "<br><table>")  # a little air above tables
    head = lambda size, color, top: TextStyle(font_family=family, font_style="B", font_size_pt=size, color=color, t_margin=top, b_margin=1)
    pdf.write_html(html, font_family=family, li_prefix_color=GREEN, ul_bullet_char="\u2022" if PDF_FONTS else "-",
                   table_line_separators=True,
                   tag_styles={"h1": head(20, GREEN, 0), "h2": head(15, GREEN, 4), "h3": head(12.5, INK, 3), "h4": head(11, INK, 2),
                               "p": TextStyle(font_family=family, font_size_pt=10.5, color=INK, t_margin=0, b_margin=2),
                               "code": TextStyle(font_family=mono, font_size_pt=9.5, color=(120, 60, 30)),
                               "blockquote": TextStyle(font_family=family, font_style="I", font_size_pt=10.5, color=(90, 84, 78), l_margin=8, t_margin=1, b_margin=2),
                               "pre": TextStyle(font_family=mono, font_size_pt=9.5, color=INK, t_margin=3, b_margin=3)})
    pdf.output(str(p))


def _blocks(content):
    """Markdown as a list of blocks, for Word and LibreOffice. Each piece of text is (text, bold, italic, code).
    ("h", level, text) ("p", text) ("li", depth, numbered, text) ("pre", lines) ("quote", text) ("hr",) ("table", rows of (header?, text))"""
    root = lxml.html.fragment_fromstring(_html(content), create_parent="div")
    out = []

    def text(el, bold=False, italic=False, code=False):
        parts = [(el.text, bold, italic, code)] if el.text else []
        for child in el:
            tag = child.tag if isinstance(child.tag, str) else ""
            if tag == "br":
                parts.append(("\n", bold, italic, code))
            elif tag not in ("ul", "ol"):  # nested lists come after their item
                parts += text(child, bold or tag in ("strong", "b"), italic or tag in ("em", "i"), code or tag == "code")
                href = child.get("href", "") if tag == "a" else ""
                if href.startswith("http") and href != child.text_content().strip():
                    parts.append((f" ({href})", bold, italic, False))
            if child.tail:
                parts.append((child.tail, bold, italic, code))
        return parts

    def lists(el, depth):
        for li in el.findall("li"):
            first = li.find("p")  # "loose" lists wrap each item in <p>
            out.append(("li", depth, el.tag == "ol", text(first if first is not None and not (li.text or "").strip() else li)))
            for sub in li:
                if sub.tag in ("ul", "ol"):
                    lists(sub, depth + 1)

    for el in root:
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            out.append(("h", min(int(tag[1]), 4), text(el)))
        elif tag in ("ul", "ol"):
            lists(el, 0)
        elif tag == "pre":
            out.append(("pre", el.text_content().rstrip("\n").split("\n")))
        elif tag == "blockquote":
            out.append(("quote", text(el, italic=True)))
        elif tag == "hr":
            out.append(("hr",))
        elif tag == "table":
            rows = [[(c.tag == "th", text(c, bold=c.tag == "th")) for c in r if c.tag in ("th", "td")] for r in el.iter("tr")]
            if any(rows):
                out.append(("table", rows))
        else:
            out.append(("p", text(el)))
    return out


def _title(content, p):
    return next((l.lstrip("# ").strip() for l in content.splitlines() if l.strip()), p.stem)[:120]


def write_docx(p, content):
    """A tidy Word document from Markdown, in the same style as the PDFs."""
    from docx.shared import Pt, RGBColor, Cm
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = docx.Document()
    for section in d.sections:
        section.left_margin = section.right_margin = Cm(2.2)
        section.top_margin = section.bottom_margin = Cm(2)
    normal = d.styles["Normal"]
    normal.font.name, normal.font.size, normal.font.color.rgb = "Calibri", Pt(11), RGBColor(*INK)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")  # Chinese, Japanese, Korean
    normal.paragraph_format.space_after, normal.paragraph_format.line_spacing = Pt(6), 1.15
    for level, size, color in ((1, 20, GREEN), (2, 15, GREEN), (3, 12.5, INK), (4, 11, INK)):
        h = d.styles[f"Heading {level}"]
        h.font.name, h.font.size, h.font.bold, h.font.italic, h.font.color.rgb = "Calibri", Pt(size), True, False, RGBColor(*color)
        for attr in ("w:asciiTheme", "w:hAnsiTheme"):  # Calibri, not the template's theme font
            h.element.rPr.rFonts.attrib.pop(qn(attr), None)
        h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(14 if level < 3 else 10), Pt(4)

    def put(par, parts):
        for t, bold, italic, code in parts:
            if t == "\n":
                par.add_run().add_break()
                continue
            r = par.add_run(t)
            r.bold, r.italic = bold or None, italic or None
            if code:
                r.font.name, r.font.size, r.font.color.rgb = "Consolas", Pt(9.5), RGBColor(120, 60, 30)
        return par

    for b in _blocks(content):
        kind = b[0]
        if kind == "h":
            put(d.add_heading(level=b[1]), b[2])
        elif kind == "p":
            put(d.add_paragraph(), b[1])
        elif kind == "li":
            style = ("List Number" if b[2] else "List Bullet") + (f" {min(b[1], 2) + 1}" if b[1] else "")
            put(d.add_paragraph(style=style), b[3]).paragraph_format.space_after = Pt(2)
        elif kind == "pre":
            par = d.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.5)
            put(par, [x for k, line in enumerate(b[1]) for x in ([("\n", 0, 0, 1)] if k else []) + [(line, False, False, True)]])
        elif kind == "quote":
            put(d.add_paragraph(), b[1]).paragraph_format.left_indent = Cm(1)
        elif kind == "hr":
            d.add_paragraph().add_run("—" * 3).font.color.rgb = RGBColor(160, 160, 160)
        elif kind == "table":
            cols = max(len(r) for r in b[1])
            t = d.add_table(rows=0, cols=cols)
            t.style = "Table Grid"
            for r in b[1]:
                cells = t.add_row().cells
                for k, (header, parts) in enumerate(r):
                    put(cells[k].paragraphs[0], parts)
                    if header:  # a light green header row
                        shd = OxmlElement("w:shd")
                        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E3EEE8")
                        cells[k]._tc.get_or_add_tcPr().append(shd)
            d.add_paragraph()
    d.core_properties.title = _title(content, p)
    d.save(p)


def write_odt(p, content):
    """A tidy LibreOffice Writer document (.odt) from Markdown, in the same style as the PDFs and Word documents."""
    from odf.opendocument import OpenDocumentText
    from odf.style import Style, TextProperties, ParagraphProperties, TableColumnProperties, TableCellProperties, ListLevelProperties
    from odf.text import H, P, Span, List, ListItem, ListStyle, ListLevelStyleBullet, ListLevelStyleNumber, LineBreak
    from odf.table import Table, TableColumn, TableRow, TableCell
    from odf.element import Text as Raw
    from odf.dc import Title
    hexa = lambda c: "#%02x%02x%02x" % c
    doc = OpenDocumentText()
    doc.meta.addElement(Title(text=_title(content, p)))

    def style(name, family="paragraph", text=None, para=None, parent=None, auto=False):
        st = Style(name=name, family=family, **({"parentstylename": parent} if parent else {}))
        if text:
            st.addElement(TextProperties(**text))
        if para:
            st.addElement(ParagraphProperties(**para))
        (doc.automaticstyles if auto else doc.styles).addElement(st)
        return st

    font = {"fontfamily": "Calibri", "fontfamilyasian": "Microsoft YaHei", "fontsize": "11pt", "color": hexa(INK)}
    body = style("Body", text=font, para={"marginbottom": "0.21cm", "lineheight": "115%"})
    heads = {lvl: style(f"Heading{lvl}", text={**font, "fontsize": f"{size}pt", "fontweight": "bold", "fontweightasian": "bold", "color": hexa(color)},
                        para={"margintop": f"{0.5 if lvl < 3 else 0.35}cm", "marginbottom": "0.14cm", "keepwithnext": "always"})
             for lvl, size, color in ((1, 20, GREEN), (2, 15, GREEN), (3, 12.5, INK), (4, 11, INK))}
    item = style("Item", text=font, para={"marginbottom": "0.07cm"})
    code_p = style("Code", text={**font, "fontfamily": "Consolas", "fontsize": "9.5pt"}, para={"marginleft": "0.5cm", "marginbottom": "0.21cm"})
    quote = style("Quote", text={**font, "fontstyle": "italic"}, para={"marginleft": "1cm", "marginbottom": "0.21cm"})
    spans = {}
    for b in (False, True):
        for i in (False, True):
            for c in (False, True):
                props = {}
                if b: props.update(fontweight="bold", fontweightasian="bold")
                if i: props.update(fontstyle="italic", fontstyleasian="italic")
                if c: props.update(fontfamily="Consolas", fontsize="9.5pt", color="#783c1e")
                if props:
                    spans[(b, i, c)] = style(f"T{int(b)}{int(i)}{int(c)}", "text", text=props, auto=True)
    bullets, numbers = ListStyle(name="Bullets"), ListStyle(name="Numbers")
    for lvl in range(1, 4):
        bl = ListLevelStyleBullet(level=lvl, bulletchar="•")
        bl.addElement(ListLevelProperties(spacebefore=f"{0.6 * lvl - 0.6}cm", minlabelwidth="0.6cm"))
        bullets.addElement(bl)
        nl = ListLevelStyleNumber(level=lvl, numformat="1", numsuffix=".")
        nl.addElement(ListLevelProperties(spacebefore=f"{0.6 * lvl - 0.6}cm", minlabelwidth="0.6cm"))
        numbers.addElement(nl)
    doc.automaticstyles.addElement(bullets); doc.automaticstyles.addElement(numbers)
    head_cell = style("HeadCell", "table-cell", auto=True)
    head_cell.addElement(TableCellProperties(backgroundcolor="#e3eee8", border="0.5pt solid #999999", padding="0.1cm"))
    cell = style("Cell", "table-cell", auto=True)
    cell.addElement(TableCellProperties(border="0.5pt solid #999999", padding="0.1cm"))

    def put(par, parts):
        for t, bold, italic, code in parts:
            if t == "\n":
                par.addElement(LineBreak())
            elif (bold, italic, code) in spans:
                par.addElement(Span(stylename=spans[(bold, italic, code)], text=t))
            else:
                par.addText(t)
        return par

    stack = []  # the open lists: (depth, numbered, List)
    for b in _blocks(content):
        if b[0] != "li":
            stack = []
        if b[0] == "h":
            doc.text.addElement(put(H(outlinelevel=b[1], stylename=heads[b[1]]), b[2]))
        elif b[0] == "p":
            doc.text.addElement(put(P(stylename=body), b[1]))
        elif b[0] == "li":
            depth, numbered, parts = b[1], b[2], b[3]
            while stack and (stack[-1][0] > depth or (stack[-1][0] == depth and stack[-1][1] != numbered)):
                stack.pop()
            if not stack or stack[-1][0] < depth:
                lst = List(stylename=numbers if numbered else bullets) if not stack else List()
                if stack:  # a nested list goes inside the last item of its parent
                    stack[-1][3].addElement(lst)
                else:
                    doc.text.addElement(lst)
                stack.append([depth, numbered, lst, None])
            li = ListItem()
            li.addElement(put(P(stylename=item), parts))
            stack[-1][2].addElement(li)
            stack[-1][3] = li
        elif b[0] == "pre":
            par = P(stylename=code_p)
            for k, line in enumerate(b[1]):
                if k:
                    par.addElement(LineBreak())
                par.addText(line)
            doc.text.addElement(par)
        elif b[0] == "quote":
            doc.text.addElement(put(P(stylename=quote), b[1]))
        elif b[0] == "hr":
            doc.text.addElement(P(stylename=body, text="—" * 3))
        elif b[0] == "table":
            cols = max(len(r) for r in b[1])
            t = Table()
            t.addElement(TableColumn(numbercolumnsrepeated=cols))
            for r in b[1]:
                row = TableRow()
                for header, parts in r + [(False, [])] * (cols - len(r)):
                    c = TableCell(stylename=head_cell if header else cell)
                    c.addElement(put(P(stylename=body), parts))
                    row.addElement(c)
                t.addElement(row)
            doc.text.addElement(t)
            doc.text.addElement(P(stylename=body))
    doc.save(str(p))


MEM = (pathlib.Path(__file__).parent / "memory").resolve()
MEM.mkdir(exist_ok=True)
MEM_LOCK = threading.Lock()


def _mem(topic):
    slug = re.sub(r"[^a-z0-9-]+", "-", (topic or "profile").lower()).strip("-") or "profile"
    return MEM / f"{slug}.md"


def memory_topics():
    return sorted(p.stem for p in MEM.glob("*.md"))


def remember(fact: str, topic: str = "profile") -> str:
    """Save a lasting fact about the user to long-term memory, so you know it in future chats. Use topic 'profile' for who the user is and how they like answers, or a short topic name (e.g. 'btc-model') for a specific project or subject."""
    fact = " ".join(fact.split())
    with MEM_LOCK:
        p = _mem(topic)
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if fact.lower() in old.lower():
            return "Already in memory."
        p.write_text(old + f"- {fact}\n", encoding="utf-8")
    return f"Saved to memory ({p.stem})."


def forget(text: str, topic: str = "profile") -> str:
    """Remove every line containing this text from a long-term memory topic."""
    with MEM_LOCK:
        p = _mem(topic)
        if not p.exists():
            return f"No memory topic '{topic}'."
        lines = p.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if text.lower() not in line.lower()]
        p.write_text("\n".join(kept) + "\n", encoding="utf-8") if kept else p.unlink()
    return f"Removed {len(lines) - len(kept)} line(s)."


def read_memory(topic: str) -> str:
    """Read one long-term memory topic about the user."""
    p = _mem(topic)
    return p.read_text(encoding="utf-8") if p.exists() else f"No memory topic '{topic}'. Topics: {', '.join(memory_topics()) or 'none'}."


def memory_prompt():
    profile = _mem("profile")
    text = profile.read_text().strip() if profile.exists() else ""
    others = [t for t in memory_topics() if t != "profile"]
    if not text and not others:
        return ""
    out = "\n\nWhat you remember about the user from earlier chats. These are facts about the USER, not about you: when they ask about themself, answer in the second person (in Italian with \"tu\"). Use them only when relevant, don't cite them as sources, and don't mention them unprompted:\n" + (text or "(nothing yet)")
    if others:
        out += f"\nOther memory topics you can open with read_memory: {', '.join(others)}."
    return out + "\nIf the user asks you to remember or forget something, use the remember or forget tool."


def get_datetime() -> str:
    """Return the current local date and time."""
    return datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M")


def look_at_image(source: str, question: str) -> str:
    """Look at an image, given as a URL or a file name in the workspace folder, and answer a question about it."""
    data = requests.get(source, timeout=15).content if source.startswith("http") else _path(source).read_bytes()
    return ollama.chat(getattr(CTX, "vision", FAST), messages=[{"role": "user", "content": question, "images": [data]}]).message.content


IMAGE_MODEL = "RunPod/FLUX.2-klein-4B-mflux-4bit"
MFLUX = HERE / "mflux-venv" / "bin" / "mflux-generate-flux2"  # Mac: mflux, built on Apple's MLX
IMAGE_PY = HERE / "image-venv" / "bin" / "python"  # Linux: diffusers on an NVIDIA card, via imagegen.py
IMAGES_OK = MFLUX.exists() if MAC else IMAGE_PY.exists()


def _draw(prompt, paths, w, h, out, timeout):
    """Run the image model: a new picture, or an edit of the pictures in 'paths'."""
    if not MAC:
        cmd = [str(IMAGE_PY), str(HERE / "imagegen.py"), "--prompt", prompt, "--width", str(w), "--height", str(h), "--out", str(out)] + (["--images", *paths] if paths else [])
    elif paths:
        cmd = [str(MFLUX_EDIT), "--model", IMAGE_MODEL, "--image-paths", *paths, "--prompt", prompt, "--width", str(w), "--height", str(h), "--steps", "4", "--output", str(out)]
    else:
        cmd = [str(MFLUX), "--model", IMAGE_MODEL, "--prompt", prompt, "--width", str(w), "--height", str(h), "--steps", "4", "--output", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def generate_image(prompt: str, name: str = "", width: int = 1024, height: int = 1024) -> str:
    """Create a new picture from a text description and save it as a PNG in the workspace folder. Write the prompt in English, describing subject, style, composition and lighting. Width and height are in pixels (multiples of 16, at most 1536). The picture is shown to the user automatically. Only for brand-new pictures: to change, combine or add to photos the user shared, use edit_image instead."""
    stem = re.sub(r"[^\w-]+", "-", name or prompt)[:40].strip("-") or "image"
    p = _path(f"{stem}-{datetime.datetime.now():%H%M%S}.png")
    w, h = (max(256, min(1536, int(v) // 16 * 16)) for v in (width, height))
    r = _draw(prompt, [], w, h, p, 900)
    if r.returncode or not p.exists():
        return "Error: image generation failed. " + (r.stderr or r.stdout)[-500:]
    CTX.last_image = str(p)
    return f"Saved image {p.name} and showed it to the user."


MFLUX_EDIT = HERE / "mflux-venv" / "bin" / "mflux-generate-flux2-edit"


def edit_image(instruction: str, images: str, name: str = "") -> str:
    """Edit or combine the user's own photos: add or remove people or objects, put people from one photo into another, change the background, season or style, and so on. It keeps the real people and places from the photos, unlike generate_image. It also works on pictures you generated or edited earlier in this chat, so the user can ask for changes step by step. 'images' is a comma-separated list of image file names in the workspace folder (photos the user shared, or files you saved earlier); put the main image to edit first. Write the instruction in English and refer to the photos as 'the first image', 'the second image'. Takes several minutes; the result is shown to the user automatically."""
    from PIL import Image
    names = images if isinstance(images, list) else str(images).split(",")
    files = sorted(f.name for f in _work().iterdir() if f.is_file())
    paths = []
    for n in [pathlib.Path(str(n).strip()).name for n in names if str(n).strip()][:4]:
        if n not in files:
            close = difflib.get_close_matches(n, files, 1, 0.5)
            if not close:
                return f"Error: there is no file called {n}. Files available: {', '.join(files)}"
            n = close[0]
        paths.append(str(_path(n)))
    if not paths:
        return "Error: give the file names of the photos to edit."
    with Image.open(paths[0]) as im:
        w, h = im.size
    scale = min(1.0, 1216 / max(w, h))
    w, h = (max(256, int(v * scale) // 16 * 16) for v in (w, h))
    stem = re.sub(r"[^\w-]+", "-", pathlib.Path(name).stem if name else "edited")[:40].strip("-") or "edited"
    out = _path(f"{stem}-{datetime.datetime.now():%H%M%S}.png")
    r = _draw(instruction, paths, w, h, out, 1800)
    if r.returncode or not out.exists():
        return "Error: image editing failed. " + (r.stderr or r.stdout)[-500:]
    CTX.last_image = str(out)
    return f"Saved edited image {out.name} and showed it to the user."

tools = {f.__name__: f for f in [web_search, read_page, run_python, install_package, list_files, read_file, write_file, get_datetime, look_at_image, generate_image, edit_image, remember, forget, read_memory]}
if not IMAGES_OK:  # image generation was not installed
    tools.pop("generate_image"), tools.pop("edit_image")
def transcribe(name: str, language: str = "") -> str:
    """Transcribe an audio or video file in the workspace (interview, press conference, voice note) on this computer, with timestamps. Nothing is sent anywhere. Saves the full transcript as a .md file next to it and returns it. language: optional two-letter code (it, en, fr…) if known; otherwise it is detected. The first use downloads the speech model (about 1.6 GB); an hour of audio takes from one to several minutes."""
    import transcribe as tr
    p = _path(name)
    if not p.exists():
        return f"There is no {name} in the workspace."
    if p.suffix.lower() not in tr.AUDIO:
        return f"{name} doesn't look like an audio or video file."
    t0 = time.time()
    try:
        segments, lang = tr.run(p, language)
    except Exception as e:
        return f"Couldn't transcribe {name}: {e}"
    lines = [f"[{tr._stamp(a)}] {txt}" for a, b, txt in segments if txt]
    out = p.with_name(p.stem + ".transcript.md")
    out.write_text(f"# {p.name}\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")
    took = int(time.time() - t0)
    head = f"Transcript of {name} ({tr._stamp(segments[-1][1]) if segments else '0:00'} of audio, language {lang}, done in {took // 60} min {took % 60} s), saved as {out.name}:\n\n"
    body = "\n".join(lines)
    return head + (body if len(body) < 15000 else body[:15000] + f"\n\n[… the rest is in {out.name}; read it with read_file]")



tools["transcribe"] = transcribe

import browser  # Joshua's own browser window, to use sites that need clicking and typing
tools.update({f.__name__: f for f in browser.TOOLS})
BROWSER_TOOLS = {f.__name__ for f in browser.TOOLS}
MEMORY_TOOLS = {"remember", "forget", "read_memory"}
messages = []
while __name__ == "__main__":
    messages.append({"role": "user", "content": input("\n> ")})
    while True:
        m = ollama.chat(FAST, messages=messages, tools=list(tools.values()), options={"num_ctx": 16384}).message
        messages.append({"role": "assistant", "content": m.content, "tool_calls": m.tool_calls})
        if not m.tool_calls:
            print(m.content)
            break
        for c in m.tool_calls:
            print(f"  [{c.function.name}] {dict(c.function.arguments)}")
            try:
                out = str(tools[c.function.name](**c.function.arguments))
            except Exception as e:
                out = f"Error: {e}"
            messages.append({"role": "tool", "tool_name": c.function.name, "content": out})
