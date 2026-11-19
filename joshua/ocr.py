"""Text from pictures of pages, for scanned PDFs that contain no text.

On a Mac it uses Apple's own text recognition (the one behind Live Text): on this computer, free, and it
recognises the language by itself. Elsewhere it uses Tesseract, if installed. What it reads is kept in
cache/ocr, so the same file is read only once."""
import hashlib
import io
import json
import pathlib
import re
import shutil
import subprocess
import sys

CACHE = pathlib.Path(__file__).parent / "cache" / "ocr"
MAC = sys.platform == "darwin"
TESS_LANGS = ["eng", "ita", "fra", "deu", "pol", "spa", "por", "chi_sim"]


def available():
    return MAC or bool(shutil.which("tesseract"))


def _need(module, package):
    """Imports a module, installing its package into Joshua's own Python the first time."""
    try:
        return __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True, capture_output=True)
        return __import__(module)


def _paragraphs(lines):
    """Lines (top, left, height, text) → text, with an empty line where the gap says a new paragraph starts."""
    lines = sorted(lines, key=lambda l: (round(l[0], 3), l[1]))
    if not lines:
        return ""
    heights = sorted(l[2] for l in lines)
    h = heights[len(heights) // 2] or 0.01
    out, prev = [], None
    for top, left, height, text in lines:
        if prev is not None and top - prev > h * 1.9:
            out.append("")
        out.append(text.strip())
        prev = top
    text = "\n".join(out)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # words split with a hyphen at the end of a line


def _mac(image):
    """Apple's text recognition on one page image (a PIL image)."""
    _need("objc", "pyobjc-core")
    Vision = _need("Vision", "pyobjc-framework-Vision")
    from Foundation import NSData
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)  # accurate
    request.setUsesLanguageCorrection_(True)
    if hasattr(request, "setAutomaticallyDetectsLanguage_"):
        request.setAutomaticallyDetectsLanguage_(True)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(str(error))
    lines = []
    for obs in request.results() or []:
        best = obs.topCandidates_(1)
        if best:
            box = obs.boundingBox()  # 0–1, measured from the bottom left
            lines.append((1 - box.origin.y - box.size.height, box.origin.x, box.size.height, str(best[0].string())))
    return _paragraphs(lines)


def _tesseract(image):
    have = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout.split()
    langs = "+".join(l for l in TESS_LANGS if l in have) or "eng"
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    out = subprocess.run(["tesseract", "-", "-", "-l", langs], input=buf.getvalue(), capture_output=True, timeout=300).stdout
    return out.decode("utf-8", "replace").strip()


def recognize(image):
    if MAC:
        try:
            return _mac(image)
        except Exception as e:
            if not shutil.which("tesseract"):
                raise RuntimeError(f"text recognition failed: {e}")
    if shutil.which("tesseract"):
        return _tesseract(image)
    raise RuntimeError("no text recognition on this computer (install Tesseract)")


def pdf_pages(path, pages, progress=None):
    """The text of the given pages (numbers from 0) of a scanned PDF, read from their pictures. Kept in a cache."""
    path = pathlib.Path(path)
    key = hashlib.sha1(path.read_bytes()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / f"{key}.json"
    try:
        done = json.loads(store.read_text(encoding="utf-8"))
    except Exception:
        done = {}
    todo = [p for p in pages if str(p) not in done]
    if todo:
        pdfium = _need("pypdfium2", "pypdfium2")
        pdf = pdfium.PdfDocument(str(path))
        for k, p in enumerate(todo):
            if progress:
                progress(k, len(todo))
            image = pdf[p].render(scale=200 / 72).to_pil()  # 200 dpi: scans are usually made at that
            done[str(p)] = recognize(image)
            if k % 10 == 9 or k == len(todo) - 1:
                store.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        pdf.close()
    return {p: done.get(str(p), "") for p in pages}


def pdf_text(path, texts, progress=None):
    """The text of each page: the PDF's own text where it has some, the pictures read where it has none."""
    empty = [i for i, t in enumerate(texts) if len((t or "").strip()) < 20]
    if not empty or not available():
        return texts
    read = pdf_pages(path, empty, progress)
    return [read.get(i, t) if i in read else t for i, t in enumerate(texts)]
