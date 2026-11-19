"""Transcription on this computer: interviews, press conferences, voice notes. Nothing is sent anywhere.
Whisper large-v3-turbo: on a Mac with Apple silicon through MLX (fast, on the graphics chip), elsewhere through
faster-whisper (on the processor; several minutes per hour of audio). Installed, with the model (about 1.6 GB),
the first time it's used."""
import platform, subprocess, sys, time

AUDIO = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac", ".wma", ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
APPLE = sys.platform == "darwin" and platform.machine() == "arm64"
_model = None


def _install():
    need = ["av"] + (["mlx-whisper"] if APPLE else ["faster-whisper"])
    missing = []
    for pkg in need:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True, timeout=1200)


def _audio(path):
    """The sound of any audio or video file as 16 kHz mono samples, decoded with PyAV (no ffmpeg needed)."""
    import av, numpy as np
    out = []
    with av.open(str(path)) as f:
        stream = next(s for s in f.streams if s.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in f.decode(stream):
            for r in resampler.resample(frame):
                out.append(r.to_ndarray()[0])
        for r in resampler.resample(None):
            out.append(r.to_ndarray()[0])
    return np.concatenate(out).astype("float32") if out else np.zeros(0, "float32")


def _stamp(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


SIZES = [("large-v3-turbo", 4.0), ("small", 1.5), ("base", 0.0)]  # model, GB of free memory it wants
USED = None


def free_gb():
    """Free memory in GB, or None if unknown."""
    import sys
    try:
        if sys.platform == "win32":
            import ctypes

            class M(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]
            m = M(); m.l = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.free / 2**30, 1)
        with open("/proc/meminfo") as f:
            info = dict(line.split(":", 1) for line in f)
        return round(int(info["MemAvailable"].split()[0]) / 2**20, 1)
    except Exception:
        return None


def threads():
    import os
    return max(1, (os.cpu_count() or 2) - 1)


def run(path, language=""):
    """Returns the segments as (start, end, text)."""
    global _model
    _install()
    sound = _audio(path)
    if not len(sound):
        raise ValueError("no sound found in this file")
    lang = (language or "").strip().lower()[:2] or None
    if APPLE:
        import mlx_whisper
        r = mlx_whisper.transcribe(sound, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language=lang, word_timestamps=False)
        return [(s["start"], s["end"], s["text"].strip()) for s in r["segments"]], r.get("language", lang)
    from faster_whisper import WhisperModel
    import jlog
    free = free_gb()
    sizes = [name for name, need in SIZES if free is None or free >= need] or [SIZES[-1][0]]
    last = None
    for name in sizes:  # the best model that fits in the free memory; if loading still fails for lack of memory, the next smaller one
        try:
            if _model is None or _model[0] != name:
                _model = None
                jlog.note(f"speech model {name} · {free if free is not None else '?'} GB of memory free")
                _model = (name, WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads()))  # one core stays free for the page
            segments, info = _model[1].transcribe(sound, language=lang, vad_filter=True, beam_size=5 if name == "large-v3-turbo" else 1)
            out = [(s.start, s.end, s.text.strip()) for s in segments]
            global USED
            USED = name
            return out, info.language
        except Exception as e:  # MemoryError, or the library's own "failed to allocate memory"
            if not ("memory" in str(e).lower() or "alloc" in str(e).lower() or isinstance(e, MemoryError)):
                raise
            jlog.note(f"speech model {name} didn't fit in memory: {e}; trying a smaller one")
            _model, last = None, e
            import gc
            gc.collect()
    raise RuntimeError(f"not enough free memory for any speech model ({last}). Close other programs and try again.")


REPO = "mlx-community/whisper-large-v3-turbo" if APPLE else "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
NAME = "Whisper large-v3-turbo"


def model_dir():
    """Where the speech model is stored (Hugging Face's cache), if it has been downloaded."""
    import os, pathlib
    home = pathlib.Path(os.environ.get("HF_HOME") or pathlib.Path.home() / ".cache" / "huggingface") / "hub"
    p = home / ("models--" + REPO.replace("/", "--"))
    return p if p.exists() else None


def size_gb():
    p = model_dir()
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9 if p else 0


def download():
    """Gets the speech model (and the tools that run it) now, instead of at the first transcription."""
    _install()
    from huggingface_hub import snapshot_download
    snapshot_download(REPO)


def remove():
    global _model
    import shutil
    p = model_dir()
    if p:
        shutil.rmtree(p)
    _model = None
