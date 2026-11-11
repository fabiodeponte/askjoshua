#!/bin/bash
# Ask Joshua installer: a private AI assistant that runs entirely on your computer (macOS or Linux).
#
#   curl -fsSL https://ask.joshua.earth/install.sh | bash
#
# What it does, in order (nothing is installed outside ~/AskJoshua, except Ollama and the Joshua launcher):
#   1. checks your computer: chip, graphics card, memory, free disk
#   2. estimates how fast each model would run on it
#   3. asks a few questions (deep thinking? images? fine-tuning?) and picks the models that fit
#   4. installs Ollama (the engine that runs the models), Python, and Joshua
#   5. downloads the models
#   6. adds Joshua to your Applications folder (Mac) or applications menu (Linux)
# Run it again at any time to change your choices; it skips what is already there.

set -euo pipefail

JOSHUA_URL="${JOSHUA_URL:-https://ask.joshua.earth/joshua.tar.gz}"
HOME_DIR="$HOME/AskJoshua"
APP="$HOME_DIR/joshua"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"

bold=$'\e[1m'; dim=$'\e[2m'; green=$'\e[32m'; amber=$'\e[33m'; red=$'\e[31m'; off=$'\e[0m'
# JOSHUA_GUI is set by the "Install Joshua" app, which shows this script's output in a window and answers its questions
[ -n "${JOSHUA_GUI:-}" ] && { bold=""; dim=""; green=""; amber=""; red=""; off=""; }
say()  { printf "%s\n" "$*"; }
step() { printf "\n${bold}▸ %s${off}\n" "$*"; }
ok()   { printf "  ${green}✓${off} %s\n" "$*"; }
warn() { printf "  ${amber}!${off} %s\n" "$*"; }
die()  { printf "\n${red}✗ %s${off}\n" "$*"; exit 1; }
ask()  {  # ask "question" default(y/n) -> returns 0 for yes
  local q="$1" d="$2" a
  if [ -n "${JOSHUA_GUI:-}" ]; then  # the app shows the question with Yes and No buttons, and writes back y or n
    printf "\n@@ASK|%s|%s\n" "$d" "$q"; read -r a || a=""
  else
    printf "  %s %s " "$q" "$([ "$d" = y ] && echo "[Y/n]" || echo "[y/N]")"
    { read -r a </dev/tty; } 2>/dev/null || read -r a || a=""
  fi
  a="${a:-$d}"; [[ "$a" =~ ^[YySs] ]]
}

cat <<'EOF'

     ██╗ ██████╗ ███████╗██╗  ██╗██╗   ██╗ █████╗
     ██║██╔═══██╗██╔════╝██║  ██║██║   ██║██╔══██╗
     ██║██║   ██║███████╗███████║██║   ██║███████║
██   ██║██║   ██║╚════██║██╔══██║██║   ██║██╔══██║
╚█████╔╝╚██████╔╝███████║██║  ██║╚██████╔╝██║  ██║
 ╚════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
        your AI, on your computer, under your control
EOF

# ---------- 1. the computer ----------
step "Checking your computer"
OS=$(uname -s)
mkdir -p "$HOME_DIR"
FREE=$(( $(df -k "$HOME_DIR" | awk 'NR==2 {print $4}') / 1048576 ))
VRAM=0; GPU=""; LOCAL_NO=""
if [ "$OS" = Darwin ]; then
  [ "$(uname -m)" = arm64 ] || LOCAL_NO="it has an Intel processor, too slow for local models"
  MACOS=$(sw_vers -productVersion); MAJOR=${MACOS%%.*}
  [ "$MAJOR" -ge 13 ] || die "macOS $MACOS is too old: Joshua needs macOS 13 (Ventura) or later."
  CHIP=$(sysctl -n machdep.cpu.brand_string)
  RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  ok "$CHIP · ${RAM} GB memory · ${FREE} GB free disk · macOS $MACOS"
elif [ "$OS" = Linux ]; then
  case "$(uname -m)" in x86_64|aarch64) ;; *) die "Joshua needs a 64-bit PC (x86_64 or ARM64).";; esac
  CHIP=$(awk -F: '/model name/ {sub(/^ +/, "", $2); print $2; exit}' /proc/cpuinfo); CHIP=${CHIP:-$(uname -m)}
  RAM=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 + 1 ))
  if command -v nvidia-smi >/dev/null && LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) && [ -n "$LINE" ]; then
    GPU=${LINE%,*}; VRAM=$(( (${LINE##*,} + 512) / 1024 ))
  fi
  ok "$CHIP · ${RAM} GB memory · ${FREE} GB free disk"
  if [ -n "$GPU" ]; then ok "Graphics: $GPU with ${VRAM} GB"
  else warn "No NVIDIA graphics card found: models will run on the processor, which is much slower. Images and fine-tuning need an NVIDIA card."; fi
else
  die "Joshua runs on macOS and Linux. On Windows, install it inside WSL (Windows Subsystem for Linux)."
fi
[ "$RAM" -ge 8 ] || LOCAL_NO="it has ${RAM} GB of memory, and local models need at least 8"
[ "$FREE" -ge 8 ] || LOCAL_NO="${LOCAL_NO:+$LOCAL_NO, and }it has only ${FREE} GB of free disk, and local models need about 8"
[ "$FREE" -ge 3 ] || die "Joshua needs at least 3 GB of free disk, even with cloud models only. You have ${FREE} GB."

# ---------- 2. how fast is it? ----------
# How fast a model writes depends mostly on memory bandwidth (GB/s): every word means reading the whole model once.
# How fast it draws pictures or trains depends on the graphics cores (G: M4 Mac = 10).
if [ "$OS" = Darwin ]; then
  GEN=$(echo "$CHIP" | sed -nE 's/.*M([0-9]+).*/\1/p'); GEN=${GEN:-9}
  case "$CHIP" in *Ultra*) TIER=Ultra;; *Max*) TIER=Max;; *Pro*) TIER=Pro;; *) TIER="";; esac
  case "$GEN$TIER" in
    1) BW=68;; 1Pro|2Pro) BW=200;; 1Max|2Max|3Max|4Max) BW=400;; 1Ultra|2Ultra|3Ultra) BW=800;;
    2|3) BW=100;; 3Pro) BW=150;; 4) BW=120;; 4Pro) BW=273;; 5) BW=153;; 5Pro) BW=300;; 5Max) BW=550;;
    *Ultra) BW=800;; *Max) BW=500;; *Pro) BW=270;; *) BW=150;;
  esac
  case "$GEN" in 1) G=6;; 2) G=7;; 3) G=9;; 4) G=10;; 5) G=13;; *) G=13;; esac
  case "$TIER" in Pro) G=$(( G * 18 / 10 ));; Max) G=$(( G * 35 / 10 ));; Ultra) G=$(( G * 7 ));; esac
  GPU_BW=$BW
else
  BW=50                                  # ordinary PC memory, used when a model doesn't fit on the card
  GPU_BW=$(( VRAM >= 16 ? 700 : VRAM >= 10 ? 450 : 300 ))   # typical NVIDIA cards of that size
  G=$(( VRAM >= 16 ? 30 : VRAM >= 10 ? 20 : 12 ))
fi
fits() { [ "$OS" = Darwin ] || [ "$VRAM" -ge "$1" ]; }   # does a model needing $1 GB fit on the graphics card?
words() { local bw=$BW; fits "$2" && bw=$GPU_BW; echo $(( bw * 35 * 75 / 10000 / $1 )); }  # words/s, reading $1 GB per word (calibrated: M4 Mac + 8B = 5)
SPEED4=$(words 3 4); SPEED8=$(words 6 7); SPEED30=$(words 4 22)   # the 30B model only reads ~3B parameters per word
PIC_MIN=$(( (15 * 10 / G + 5) / 10 )); [ "$PIC_MIN" -lt 1 ] && PIC_MIN=1   # minutes per picture
TUNE_MIN=$(( 17 * 10 / G ))                                               # minutes to train on 300 examples
if [ "$OS" = Darwin ]; then ok "Estimated: ${BW} GB/s memory bandwidth"; fi

# ---------- 3. questions and model choice ----------
step "A few questions"
# Cloud models only: Joshua itself runs here, but every answer comes from a big model on Ollama's servers.
CLOUD_TEXT="Joshua runs here, and your chats, files and memory stay on this computer, but every message is answered by a large model on
  Ollama's servers. Nothing big to download, and it works on any computer. You need an internet connection and an Ollama account:
  new accounts pay per use, so add a few dollars of credit on ollama.com (Joshua sets up Gemma 4 31B, which costs well under
  a cent per message). Some older accounts still have a small free monthly allowance."
CLOUDONLY=n
if [ -n "$LOCAL_NO" ]; then
  say "  This computer can't run Joshua's models itself: $LOCAL_NO."
  say "  ${bold}It can still use cloud models only.${off} ${dim}$CLOUD_TEXT${off}"
  ask "Install Joshua with cloud models only?" y || die "Nothing was installed."
  CLOUDONLY=y
else
  say "  ${bold}Where should Joshua's answers come from?${off} Normally from models on this computer (private, free, works offline)."
  say "  ${dim}Or from cloud models only: $CLOUD_TEXT${off}"
  DEF=n; [ "$SPEED4" -lt 3 ] && DEF=y && say "  ${dim}Recommended here: local models would be very slow on this computer.${off}"
  ask "Use cloud models only?" $DEF && CLOUDONLY=y
fi
if [ "$CLOUDONLY" = y ]; then
  CLOUD=y; THINK=n; IMAGES=n; TUNE=n; MODEL_GB=0; LABEL="Gemma 4 31B (cloud)"
else
# The memory the model must fit in: on a Mac the whole memory; on a PC the graphics card's (7 GB+ runs the 8B, 22 GB+ the 30B)
if [ "$OS" = Darwin ]; then BIG=$RAM; else BIG=$(( VRAM >= 22 ? 32 : VRAM >= 7 ? 16 : 8 )); fi
if [ "$BIG" -lt 16 ]; then
  SIZE=4b
  say "  ${dim}On this computer Joshua uses the 4-billion-parameter model: it sees images and handles everyday questions, writing about ${SPEED4} words a second.${off}"
elif [ "$BIG" -lt 32 ]; then
  SIZE=8b
  if [ "$SPEED8" -lt 4 ]; then
    say "  ${dim}You could run the 8-billion-parameter model, but here it would write only about ${SPEED8} words a second."
    say "  The 4-billion one writes about ${SPEED4} a second and is a little less capable.${off}"
    ask "Use the faster 4B model? (recommended)" y && SIZE=4b
  else
    say "  ${dim}Joshua uses the 8-billion-parameter model, the best fit for this computer: about ${SPEED8} words a second.${off}"
  fi
else
  SIZE=8b
  say "  ${dim}You can run a much larger model: 30 billion parameters, about 20 GB to download."
  say "  It only uses 3 billion of them for each word, so it stays quick: about ${SPEED30} words a second (the 8B: about ${SPEED8}).${off}"
  if [ "$FREE" -ge 30 ]; then ask "Use the larger model? (recommended)" y && SIZE=30b
  else say "  ${dim}You have ${FREE} GB free, not enough for it with room to spare, so Joshua will use the 8B.${off}"; fi
fi
case "$SIZE" in 4b) LABEL="qwen3-vl 4b"; MODEL_GB=3; SPEED=$SPEED4;; 8b) LABEL="qwen3-vl 8b"; MODEL_GB=6; SPEED=$SPEED8;; 30b) LABEL="qwen3-vl 30b"; MODEL_GB=20; SPEED=$SPEED30;; esac
FAST_CANDIDATES="qwen3-vl:${SIZE}-instruct"
THINK_CANDIDATES="qwen3-vl:${SIZE}-thinking qwen3-vl:${SIZE}"
[ "$SIZE" = 30b ] && FAST_CANDIDATES="qwen3-vl:30b-a3b-instruct qwen3-vl:30b-instruct" && THINK_CANDIDATES="qwen3-vl:30b-a3b-thinking qwen3-vl:30b-thinking qwen3-vl:30b"

say ""
say "  ${bold}Deep thinking${off}: a second version of the model that reasons step by step before answering."
say "  ${dim}Better on hard questions, slower. You switch it on per message. Another ~${MODEL_GB} GB.${off}"
DEF=y
if [ "$SPEED" -lt 4 ]; then DEF=n; say "  ${dim}On this computer its reasoning can take a minute or more before the answer starts.${off}"; fi
THINK=n; ask "Install the deep-thinking version?" $DEF && THINK=y

IMAGES=n; IMG_GB=7
say ""
if { [ "$OS" = Darwin ] && [ "$RAM" -ge 16 ]; } || { [ "$OS" = Linux ] && [ "$VRAM" -ge 12 ]; }; then
  [ "$OS" = Linux ] && IMG_GB=20   # PyTorch with CUDA, plus the full-precision image model
  say "  ${bold}Images${off}: Joshua can create pictures and edit your photos, on your computer (FLUX.2 klein)."
  say "  ${dim}About ${IMG_GB} GB more. A picture takes about ${PIC_MIN} min here, editing a photo several times longer.${off}"
  DEF=y; [ "$PIC_MIN" -gt 3 ] && DEF=n
  ask "Install image generation?" $DEF && IMAGES=y
elif [ "$OS" = Darwin ]; then
  say "  ${dim}Image generation needs 16 GB of memory, so it's not offered on this Mac. Joshua can still look at images.${off}"
else
  say "  ${dim}Image generation needs an NVIDIA card with 12 GB or more, so it's not offered here. Joshua can still look at images.${off}"
fi

TUNE=n; TUNE_GB=5
say ""
if [ "$OS" = Darwin ] || [ "$VRAM" -ge 8 ]; then
  [ "$OS" = Linux ] && TUNE_GB=12   # PyTorch with CUDA, plus the full-precision base model
  say "  ${bold}Fine-tuning${off}: teach Joshua your style with your own examples, guided step by step."
  say "  ${dim}About ${TUNE_GB} GB of tools now, and ~18 GB free while you install a trained model."
  say "  Training on 300 examples takes about ${TUNE_MIN} minutes here.${off}"
  ask "Install the fine-tuning tools?" n && TUNE=y
else
  say "  ${dim}Fine-tuning needs an NVIDIA card with 8 GB or more on Linux, so it's not offered here.${off}"
fi

CLOUD=n
say ""
say "  ${bold}Cloud models${off} (optional): much bigger models that run on Ollama's servers instead of this computer."
say "  ${dim}Your chats, files and memory stay stored here, but each message in a cloud chat is sent to Ollama to be answered."
say "  Ollama says it doesn't log or train on them. You need an Ollama account; new accounts pay per use with credit you add on"
say "  ollama.com (Gemma 4 31B, which Joshua sets up, costs well under a cent per message). You choose per chat.${off}"
DEF=n; [ "$SPEED" -lt 4 ] && DEF=y && say "  ${dim}Recommended here: this computer is slow with local models.${off}"
ask "Connect Joshua to Ollama's cloud models?" $DEF && CLOUD=y
fi

NEED=$(( 2 + MODEL_GB )); [ "$CLOUDONLY" = y ] && NEED=1
[ "$THINK" = y ] && NEED=$(( NEED + MODEL_GB ))
[ "$IMAGES" = y ] && NEED=$(( NEED + IMG_GB ))
[ "$TUNE" = y ] && NEED=$(( NEED + TUNE_GB ))
say ""
if [ "$FREE" -lt $(( NEED + 3 )) ]; then
  die "This needs about ${NEED} GB of disk (plus a little room to breathe) and you have ${FREE} GB free. Free some space, or run again and say no to some options."
fi
if [ "$CLOUDONLY" = y ]; then ok "Plan: cloud models only (Gemma 4 31B on Ollama's servers) · about 1 GB"
else ok "Plan: ${LABEL} (~${SPEED} words/s)$([ "$THINK" = y ] && echo " + deep thinking")$([ "$CLOUD" = y ] && echo " + cloud models")$([ "$IMAGES" = y ] && echo " + images")$([ "$TUNE" = y ] && echo " + fine-tuning") · about ${NEED} GB"; fi
ask "Go ahead?" y || die "Nothing was installed."

# ---------- 4. Ollama, Python, Joshua ----------
step "Ollama (runs the models)"
OLLAMA=""
for c in "$(command -v ollama || true)" /Applications/Ollama.app/Contents/Resources/ollama "$HOME/Applications/Ollama.app/Contents/Resources/ollama"; do
  [ -n "$c" ] && [ -x "$c" ] && OLLAMA="$c" && break
done
if [ -n "$OLLAMA" ]; then
  ok "already installed"
elif [ "$OS" = Darwin ]; then
  TMP=$(mktemp -d)
  curl -fL --progress-bar https://ollama.com/download/Ollama-darwin.zip -o "$TMP/Ollama.zip"
  unzip -q "$TMP/Ollama.zip" -d "$TMP"
  DEST=/Applications; [ -w "$DEST" ] || { DEST="$HOME/Applications"; mkdir -p "$DEST"; }
  mv "$TMP/Ollama.app" "$DEST/"
  rm -rf "$TMP"
  OLLAMA="$DEST/Ollama.app/Contents/Resources/ollama"
  ok "installed in $DEST"
else
  say "  ${dim}Ollama's official Linux installer will ask for your password (it installs a system service).${off}"
  curl -fsSL https://ollama.com/install.sh | sh
  OLLAMA=$(command -v ollama) || die "Ollama didn't install. See https://ollama.com/download/linux"
fi
start_ollama() {
  curl -fs http://127.0.0.1:11434/api/version >/dev/null && return 0
  if [ "$OS" = Darwin ]; then open -a Ollama 2>/dev/null || ("$OLLAMA" serve >/dev/null 2>&1 &)
  else (nohup "$OLLAMA" serve >/dev/null 2>&1 &); fi
  for _ in $(seq 60); do curl -fs http://127.0.0.1:11434/api/version >/dev/null && return 0; sleep 1; done
  return 1
}
start_ollama || die "Ollama didn't start. Start it once by hand ('ollama serve'), then run this installer again."
ok "running"

if [ "$OS" = Linux ]; then
  step "Sandbox for Joshua's code"
  if command -v bwrap >/dev/null; then ok "bubblewrap is installed"
  else
    say "  ${dim}Joshua runs Python code in a sandbox (bubblewrap) so it can only write in the chat's folder.${off}"
    PKG=""; for p in apt-get dnf pacman zypper; do command -v $p >/dev/null && PKG=$p && break; done
    if [ -n "$PKG" ] && ask "Install bubblewrap now? (asks for your password)" y; then
      case $PKG in apt-get) sudo apt-get install -y bubblewrap;; dnf) sudo dnf install -y bubblewrap;; pacman) sudo pacman -S --noconfirm bubblewrap;; zypper) sudo zypper install -y bubblewrap;; esac
      ok "bubblewrap installed"
    else warn "without bubblewrap, Joshua's code runs without a sandbox; install the 'bubblewrap' package to add it"; fi
  fi
fi

step "Python"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null
  UV="$HOME/.local/bin/uv"
fi
[ -x "$UV" ] || die "Couldn't install uv (the Python installer)."
[ -x "$HOME_DIR/venv/bin/python" ] || "$UV" venv -q --seed --python 3.12 "$HOME_DIR/venv"
PY="$HOME_DIR/venv/bin/python"
"$UV" pip install -q --python "$PY" pip   # lets Joshua install packages for its own code
ok "Python $("$PY" -c 'import platform; print(platform.python_version())') in $HOME_DIR/venv"

step "Joshua"
mkdir -p "$APP"
if [ -n "$SRC_DIR" ] && [ -f "$SRC_DIR/joshua/web.py" ]; then
  cp -R "$SRC_DIR"/joshua/*.py "$SRC_DIR"/joshua/requirements.txt "$SRC_DIR"/joshua/VERSION "$SRC_DIR"/joshua/icon.png "$SRC_DIR"/joshua/mac "$SRC_DIR"/joshua/locales "$APP/"
else
  curl -fsSL "$JOSHUA_URL" | tar -xz -C "$APP" --strip-components=1
fi
"$UV" pip install -q --python "$PY" -r "$APP/requirements.txt"
ok "Joshua $(cat "$APP/VERSION" 2>/dev/null) installed in $APP"

# ---------- 5. models ----------
pull_first() {  # pull the first tag that exists; print its name
  for t in "$@"; do
    if "$OLLAMA" pull "$t" >&2; then echo "$t"; return 0; fi
    warn "$t is not available, trying the next one" >&2
  done
  return 1
}
FAST=""; CLOUDM=""
if [ "$CLOUDONLY" = n ]; then
step "Downloading the model (this is the long part)"
FAST=$(pull_first $FAST_CANDIDATES) || die "Couldn't download the model. Check your connection and run again."
ok "$FAST"
fi
MODEL="$FAST"
if [ "$THINK" = y ]; then
  MODEL=$(pull_first $THINK_CANDIDATES) || { warn "deep-thinking version not available; skipping"; MODEL="$FAST"; }
  ok "$MODEL"
fi

if [ "$CLOUD" = y ]; then
  step "Cloud models"
  say "  ${dim}Sign in to Ollama in the browser page that opens, and approve this computer. Come back here when it's done.${off}"
  "$OLLAMA" signin || warn "sign-in didn't finish; you can do it later from Joshua (☁ Cloud models → Connect)$([ "$CLOUDONLY" = y ] && echo ". Until then Joshua can't answer.")"
  CLOUDM=$(pull_first gemma4:31b-cloud glm-5.3-flash:cloud gpt-oss:120b-cloud) && ok "$CLOUDM added (nothing downloaded: it runs on Ollama's servers)" \
    || warn "couldn't add a cloud model now; add one later from Joshua (☁ Cloud models)"
  if [ "$CLOUDONLY" = y ]; then FAST=${CLOUDM:-gemma4:31b-cloud}; MODEL=$FAST; LABEL="$FAST"; fi
fi

if [ "$IMAGES" = y ]; then
  step "Image generation"
  if [ "$OS" = Darwin ]; then
    [ -x "$APP/mflux-venv/bin/python" ] || "$UV" venv -q --python 3.12 "$APP/mflux-venv"
    "$UV" pip install -q --python "$APP/mflux-venv/bin/python" "mflux==0.24.0"
    TEST=("$APP/mflux-venv/bin/mflux-generate-flux2" --model "RunPod/FLUX.2-klein-4B-mflux-4bit" --prompt "a small lighthouse at dusk" --width 256 --height 256 --steps 2 --output "$HOME_DIR/test-image.png")
  else
    [ -x "$APP/image-venv/bin/python" ] || "$UV" venv -q --python 3.12 "$APP/image-venv"
    "$UV" pip install -q --python "$APP/image-venv/bin/python" torch diffusers transformers accelerate safetensors pillow
    TEST=("$APP/image-venv/bin/python" "$APP/imagegen.py" --prompt "a small lighthouse at dusk" --width 256 --height 256 --out "$HOME_DIR/test-image.png")
  fi
  say "  ${dim}Downloading the image model and drawing a test picture…${off}"
  "${TEST[@]}" >/dev/null 2>&1 \
    && ok "working (test picture: $HOME_DIR/test-image.png)" \
    || warn "the test picture failed; Joshua will retry the download the first time you ask for an image"
fi

if [ "$OS" = Darwin ]; then TUNE_BASE="mlx-community/Qwen3-4B-Instruct-2507-4bit"; else TUNE_BASE="Qwen/Qwen3-4B-Instruct-2507"; fi
if [ "$TUNE" = y ]; then
  step "Fine-tuning tools"
  if [ "$OS" = Darwin ]; then
    [ "$MAJOR" -ge 14 ] || warn "macOS 14 or later is recommended for training (you have $MACOS)."
    "$UV" pip install -q --python "$PY" mlx-lm huggingface_hub
  else
    "$UV" pip install -q --python "$PY" torch transformers peft bitsandbytes accelerate huggingface_hub
  fi
  if [ ! -f "$HOME_DIR/llama.cpp/convert_hf_to_gguf.py" ]; then
    TMP=$(mktemp -d)
    curl -fsSL https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz | tar -xz -C "$TMP"
    rm -rf "$HOME_DIR/llama.cpp"
    mv "$TMP"/llama.cpp-master "$HOME_DIR/llama.cpp"   # we only use its Python converter; nothing is compiled
    rm -rf "$TMP"
  fi
  (cd "$HOME_DIR/llama.cpp/requirements" && "$UV" pip install -q --python "$PY" -r requirements-convert_hf_to_gguf.txt) \
    || "$UV" pip install -q --python "$PY" torch transformers sentencepiece protobuf numpy
  "$UV" pip install -q --python "$PY" "$HOME_DIR/llama.cpp/gguf-py"
  say "  ${dim}Downloading the model you will teach…${off}"
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$TUNE_BASE')" >/dev/null 2>&1 \
    || warn "couldn't download it now; Joshua will download it when you first train"
  ok "ready: open Joshua and press 🧪 Teach Joshua"
fi

cat > "$APP/config.json" <<EOF
{"model": "$MODEL", "fast": "$FAST", "label": "$LABEL", "tune_base": "$TUNE_BASE", "tune_template": "qwen3:4b-instruct-2507-q4_K_M"}
EOF

# ---------- 6. launcher ----------
step "Joshua launcher"
URL=http://127.0.0.1:7860
if [ "$OS" = Darwin ]; then
  START_OLLAMA="open -a Ollama 2>/dev/null || (\"$OLLAMA\" serve >/dev/null 2>&1 &)"
  SERVICE="$HOME/Library/LaunchAgents/earth.joshua.plist"
  KICK="launchctl bootstrap gui/\$(id -u) '$SERVICE' 2>/dev/null || launchctl kickstart gui/\$(id -u)/earth.joshua"
  HALT="launchctl bootout gui/\$(id -u)/earth.joshua 2>/dev/null"
  MACAPPS=/Applications; [ -w "$MACAPPS" ] || { MACAPPS="$HOME/Applications"; mkdir -p "$MACAPPS"; }  # the main Applications folder when we may write there
  OPEN_APP="open '$MACAPPS/Joshua.app'"; [ "$(uname -m)" = arm64 ] || OPEN_APP="open $URL"  # Intel: in the browser
else
  START_OLLAMA="(nohup \"$OLLAMA\" serve >/dev/null 2>&1 &)"
  SERVICE="$HOME/.config/systemd/user/joshua.service"
  KICK="systemctl --user start joshua"
  HALT="systemctl --user stop joshua 2>/dev/null"
  # its own window, without the browser's bars, when a Chromium-family browser is installed
  OPEN_APP='for b in chromium chromium-browser google-chrome google-chrome-stable brave-browser microsoft-edge; do command -v $b >/dev/null && { nohup $b --app='$URL' >/dev/null 2>&1 & exit 0; }; done; xdg-open '$URL
fi
cat > "$HOME_DIR/start.sh" <<EOF
#!/bin/bash
# Starts Joshua (and Ollama if needed), then opens it.
curl -fs http://127.0.0.1:11434/api/version >/dev/null || { $START_OLLAMA; sleep 4; }
if ! curl -fs $URL >/dev/null; then
  if [ -f "$SERVICE" ]; then  # Joshua runs from login: ask the system to start it, and wait for it
    $KICK
  else
    cd "$APP" && JOSHUA_NO_OPEN=1 nohup "$PY" "$APP/web.py" >> "$HOME_DIR/joshua.log" 2>&1 &
  fi
  for i in \$(seq 40); do sleep 1; curl -fs $URL >/dev/null && break; done
fi
[ -n "\$JOSHUA_NO_OPEN" ] || { $OPEN_APP; }
EOF
cat > "$HOME_DIR/stop.sh" <<EOF
#!/bin/bash
# Stops Joshua until you open it again (or, if it starts from login, until you next log in).
[ -f "$SERVICE" ] && $HALT
pkill -f "$APP/web.py" && echo "Joshua stopped." || echo "Joshua wasn't running."
EOF
if [ "$OS" = Darwin ]; then LAUNCHER="$MACAPPS/Joshua.app"; rm -rf "$HOME/Applications/Joshua.app"; else LAUNCHER="$HOME/.local/share/applications/joshua.desktop"; fi
cat > "$HOME_DIR/uninstall.sh" <<EOF
#!/bin/bash
# Removes Joshua. Your chats are in $APP/conversations: copy them first if you want to keep them.
read -r -p "Remove Joshua, its chats and its Python? [y/N] " a </dev/tty; [[ "\$a" =~ ^[Yy] ]] || exit 0
[ -f "$SERVICE" ] && { $HALT; systemctl --user disable joshua 2>/dev/null; rm -f "$SERVICE"; }
pkill -f "$APP/web.py" || true
rm -rf "$HOME_DIR" "$LAUNCHER"
read -r -p "Also delete the models from Ollama ($FAST $MODEL)? [y/N] " a </dev/tty
[[ "\$a" =~ ^[Yy] ]] && "$OLLAMA" rm $FAST $MODEL
echo "Done. Ollama itself is still installed; remove it separately if you don't need it."
EOF
chmod +x "$HOME_DIR"/{start,stop,uninstall}.sh

# Start at login: Joshua runs quietly in the background, so it opens at once and scheduled tasks can run.
AUTOSTART=
if ask "Start Joshua automatically when you log in? (opens at once; scheduled tasks can run)" y; then
  AUTOSTART=1
  if [ "$OS" = Darwin ]; then
    mkdir -p "$(dirname "$SERVICE")"
    cat > "$SERVICE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>earth.joshua</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$APP/web.py</string></array>
  <key>WorkingDirectory</key><string>$APP</string>
  <key>EnvironmentVariables</key><dict>
    <key>JOSHUA_SERVICE</key><string>1</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME_DIR/joshua.log</string>
  <key>StandardErrorPath</key><string>$HOME_DIR/joshua.log</string>
</dict></plist>
EOF
    pkill -f "$APP/web.py" || true
    launchctl bootout gui/$(id -u)/earth.joshua 2>/dev/null || true   # stop an older copy, if there is one
    launchctl bootstrap gui/$(id -u) "$SERVICE" || warn "couldn't start it now; it will start when you next log in"
  else
    mkdir -p "$(dirname "$SERVICE")"
    cat > "$SERVICE" <<EOF
[Unit]
Description=Joshua, your AI on your computer
[Service]
ExecStart=$PY $APP/web.py
WorkingDirectory=$APP
Environment=JOSHUA_SERVICE=1
StandardOutput=append:$HOME_DIR/joshua.log
StandardError=append:$HOME_DIR/joshua.log
Restart=always
RestartSec=10
[Install]
WantedBy=default.target
EOF
    pkill -f "$APP/web.py" || true
    { systemctl --user daemon-reload && systemctl --user enable --now joshua; } || { warn "couldn't set up the login service; open Joshua from the applications menu instead"; AUTOSTART=; }
  fi
  if [ -n "$AUTOSTART" ]; then ok "Joshua starts when you log in"; fi
fi

if [ "$OS" = Darwin ] && [ "$(uname -m)" != arm64 ]; then  # Intel Mac: Joshua.app is built for Apple Silicon, so a small launcher opens Joshua in the browser
  rm -rf "$LAUNCHER"
  osacompile -o "$LAUNCHER" -e "do shell script \"nohup '$HOME_DIR/start.sh' >/dev/null 2>&1 &\"" >/dev/null 2>&1 || warn "couldn't create the launcher; start Joshua with $HOME_DIR/start.sh"
  ok "Joshua is in your Applications folder ($LAUNCHER)"
elif [ "$OS" = Darwin ]; then  # Joshua.app: a window with Joshua inside, nothing else (source: joshua/mac/Joshua.swift)
  rm -rf "$LAUNCHER"; mkdir -p "$LAUNCHER/Contents/MacOS" "$LAUNCHER/Contents/Resources"
  cp "$APP/mac/Joshua" "$LAUNCHER/Contents/MacOS/Joshua"; cp "$APP/mac/Joshua.icns" "$LAUNCHER/Contents/Resources/"
  cat > "$LAUNCHER/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Joshua</string>
  <key>CFBundleDisplayName</key><string>Joshua</string>
  <key>CFBundleIdentifier</key><string>earth.joshua.app</string>
  <key>CFBundleExecutable</key><string>Joshua</string>
  <key>CFBundleIconFile</key><string>Joshua</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$(cat "$APP/VERSION")</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
EOF
  codesign --force -s - "$LAUNCHER" >/dev/null 2>&1 || true
  ok "Joshua is in your Applications folder: open it from there, or drag it to the Dock"
else
  mkdir -p "$(dirname "$LAUNCHER")"
  cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=Joshua
Comment=Your AI, on your computer
Exec=$HOME_DIR/start.sh
Icon=$APP/icon.png
Terminal=false
Categories=Utility;
EOF
  ok "Joshua is in your applications menu"
fi

step "Done"
if [ "$OS" = Darwin ]; then
  say "  Open ${bold}Joshua${off} from your Applications folder or Launchpad; drag it to the Dock to keep it there."
  say "  ${dim}Everything lives in $HOME_DIR · stop it with stop.sh · remove it with uninstall.sh${off}"
  open "$LAUNCHER" || true
else
  say "  Open ${bold}Joshua${off} from your applications. It opens at $URL"
  say "  ${dim}Everything lives in $HOME_DIR · stop it with stop.sh · remove it with uninstall.sh${off}"
  if ask "Start Joshua now?" y; then (nohup "$HOME_DIR/start.sh" >/dev/null 2>&1 &); fi
fi
