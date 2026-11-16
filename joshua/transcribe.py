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
    if _model is None:
        _model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    segments, info = _model.transcribe(sound, language=lang, vad_filter=True, beam_size=5)
    return [(s.start, s.end, s.text.strip()) for s in segments], info.language


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


def remove():
    global _model
    import shutil
    p = model_dir()
    if p:
        shutil.rmtree(p)
    _model = None
