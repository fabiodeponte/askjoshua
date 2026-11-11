#!/bin/bash
# For testing the window only: talks like install.sh, installs nothing.
ask() { printf "\n@@ASK|%s|%s\n" "$2" "$1"; read -r a || a=""; a="${a:-$2}"; [[ "$a" =~ ^[Yy] ]]; }
printf "\n▸ Checking your computer\n"; sleep 1
echo "  ✓ Apple M4 · 16 GB memory · 120 GB free disk · macOS 26.5"
printf "\n▸ Choosing the model\n"; sleep 1
echo "  ✓ qwen3-vl 8B, about 22 words per second here"
echo ""
echo "  Images: Joshua can create pictures and edit your photos, on your computer (FLUX.2 klein)."
echo "  About 7 GB more. A picture takes about 1 min here, editing a photo several times longer."
ask "Install image generation?" y && echo "  (images: yes)" || echo "  (images: no)"
echo "  ✓ Plan: qwen3-vl 8B + images · about 13 GB"
ask "Go ahead?" y || { printf "\n✗ Nothing was installed.\n"; exit 1; }
printf "\n▸ Downloading the model\n"
for i in $(seq 0 10 100); do printf "\r  ######## %3d%%" $i; sleep 0.3; done; printf "\n"
printf "\n▸ Done\n"; echo "  ✓ Joshua is in your Applications folder"
