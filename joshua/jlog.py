"""Joshua's log: what it does and what goes wrong, with the time, in AskJoshua/joshua.log.

Everything Joshua prints goes there with a time in front, including Python's errors with their full detail,
crashes of the libraries underneath (faulthandler), and errors of the page in the browser. Settings → Log
shows the last lines. The file is kept below about 5 MB."""
import faulthandler
import os
import pathlib
import platform
import sys
import threading
import time
import traceback

PATH = pathlib.Path(os.environ.get("JOSHUA_LOG") or pathlib.Path(__file__).resolve().parent.parent / "joshua.log")
KEEP = 5_000_000


class _Stamped:
    """A stream that puts the time in front of every line."""

    def __init__(self, stream):
        self.stream, self.fresh, self.lock = stream, True, threading.Lock()

    def write(self, text):
        with self.lock:
            out = []
            for part in str(text).splitlines(True):
                if self.fresh:
                    out.append(time.strftime("%Y-%m-%d %H:%M:%S "))
                out.append(part)
                self.fresh = part.endswith("\n")
            self.stream.write("".join(out))
        return len(text)

    def flush(self):
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _trim():
    """Keeps the last ~1 MB when the file passes 5 MB (written in place: the Mac's start script holds it open)."""
    try:
        if PATH.exists() and PATH.stat().st_size > KEEP:
            tail = PATH.read_bytes()[-1_000_000:]
            with open(PATH, "r+b") as f:
                f.seek(0)
                f.write(b"[older lines removed]\n" + tail[tail.find(b"\n") + 1:])
                f.truncate()
    except OSError:
        pass


def setup():
    _trim()
    if os.environ.get("JOSHUA_LOG"):  # started without a window (Windows): everything goes to the file
        stream = open(PATH, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = sys.stderr = _Stamped(stream)
    elif not sys.stdout.isatty():  # Mac and Linux: the start script already sends it to the file
        sys.stdout, sys.stderr = _Stamped(sys.stdout), _Stamped(sys.stderr)
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)  # a crash in a library underneath (a segfault) is written too
    except Exception:
        pass
    sys.excepthook = lambda kind, err, tb: note("ERROR", "".join(traceback.format_exception(kind, err, tb)))
    threading.excepthook = lambda a: note("ERROR in a background task", "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))


def note(what, detail=""):
    """One event: a short line, and the detail (a traceback, say) under it."""
    print(f"▸ {what}" + (("\n" + str(detail).rstrip()) if detail else ""), flush=True)


def error(what, exc=None):
    note("ERROR " + what, traceback.format_exc() if exc is not None else "")


def started(version):
    note(f"Joshua {version} started · {platform.system()} {platform.release()} · Python {platform.python_version()} · {os.cpu_count()} processor cores · {memory()}")


def memory():
    """The computer's memory, total and free, as a short text."""
    try:
        if sys.platform == "win32":
            import ctypes

            class M(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
            m = M(); m.l = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return f"memory {m.total / 2**30:.0f} GB, {m.free / 2**30:.1f} GB free"
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return f"memory {total / 2**30:.0f} GB"
    except Exception:
        return "memory unknown"


def tail(lines=200):
    try:
        with open(PATH, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            return f.read().decode("utf-8", "replace").splitlines()[-lines:]
    except OSError:
        return []
