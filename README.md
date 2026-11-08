# Ask Joshua

**An AI assistant that lives on your own computer.** Your chats, files and memory stay with you; the models run on your machine, or, only if you choose, on Ollama's cloud.

Website and installer: **https://ask.joshua.earth** · Why it exists: [the manifesto](https://ask.joshua.earth/manifesto.html)

## Install

macOS (Apple Silicon) or Linux:

```bash
curl -fsSL https://ask.joshua.earth/install.sh | bash
```

The installer checks your chip, memory, graphics card and disk, estimates how fast each model would run, asks a few questions, then installs [Ollama](https://ollama.com), a private Python, Joshua and the models you chose. Everything lives in `~/AskJoshua`; `uninstall.sh` there removes it. On computers too small for local models (or if you prefer), it can install Joshua with cloud models only.

## What it does

- Chats with local open models (qwen3-vl 4B / 8B / 30B), with optional deep thinking, and optional Ollama cloud models, clearly marked.
- Searches the web and cites its sources, reads files and pictures, runs Python in a sandbox, keeps memory notes you can edit.
- Goal mode and a "swarm" of agents for harder tasks.
- Creates and edits pictures on your computer (FLUX.2 klein).
- Fine-tuning: teach it your style with your own examples, step by step, on your machine.
- Scheduled tasks, while the computer is on and Joshua is running.
- An OpenAI-compatible API for your own programs: see [the API how-to](https://ask.joshua.earth/api.html).

## How the code is organised

| File | What it does |
|---|---|
| `install.sh` | The installer: hardware check, questions, Ollama, Python, models, launchers, start-at-login service. Plain, commented shell. |
| `joshua/web.py` | The interface (Gradio), chats, model menu, cloud models, memory, API server, scheduler, self-update. |
| `joshua/agent.py` | The agent loop and its tools: web search, reading pages, files, sandboxed Python, pictures, memory. |
| `joshua/tune_ui.py`, `finetune.py` | The guided fine-tuning page and the steps behind it (mlx-lm on Mac). |
| `joshua/tune_cuda.py` | The same fine-tuning steps on Linux with an NVIDIA card (PyTorch + PEFT). |
| `joshua/imagegen.py` | Picture generation on Linux (diffusers); on Mac Joshua uses mflux. |
| `joshua/mac/Joshua.swift` | The small macOS app: a window with Joshua inside. `joshua/mac/Joshua` is the compiled version the installer uses. |
| `site/` | The website at ask.joshua.earth. |

To build the Mac app yourself:

```bash
cd joshua/mac && swiftc -O -target arm64-apple-macos13 Joshua.swift -o Joshua
```

## Your data

Nothing is collected. Chats, files, memory notes and settings are plain files in `~/AskJoshua/joshua/`. With a local model nothing leaves your computer except web searches Joshua makes for you. A chat with a cloud model sends its messages to Ollama's servers; memory notes are shared with cloud models only if you switch that on. Details: [where your data lives](https://ask.joshua.earth/#data).

## Related

Joshua comes from the same idea as the [Yattaphone](https://www.yattaphone.org), a phone you build yourself: your hardware, your data, your intelligence.

## Licence

[GNU Affero General Public License v3.0](LICENSE). You are free to use, study, change and share Joshua. If you distribute it, or offer a changed version as a service, you must share your source code under the same licence.
