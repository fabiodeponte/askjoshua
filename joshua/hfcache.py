"""Size and removal of a model in Hugging Face's cache. A model's files are in its 'blobs' folder; recent versions
of huggingface_hub keep the big ones in a shared folder (hub/blobs/…) and leave only a link to each in 'blobs'."""
import pathlib, shutil


def size_gb(p):
    """The model's size: each file in 'blobs', following the links to where the big files really are."""
    return sum(f.stat().st_size for f in (pathlib.Path(p) / "blobs").iterdir() if f.exists()) / 1e9


def remove(p):
    """Deletes the big files the links point to first, then the model's folder."""
    for f in (pathlib.Path(p) / "blobs").iterdir():
        if f.is_symlink():
            f.resolve().unlink(missing_ok=True)
    shutil.rmtree(p, ignore_errors=True)
