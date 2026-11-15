"""Where the ollama program is, on macOS, Linux and Windows."""
import os, pathlib, shutil


def path():
    found = shutil.which("ollama")
    if found:
        return found
    for p in ["/Applications/Ollama.app/Contents/Resources/ollama", pathlib.Path.home() / "Applications/Ollama.app/Contents/Resources/ollama",
              pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"]:
        if pathlib.Path(p).exists():
            return str(p)
    return "ollama"
