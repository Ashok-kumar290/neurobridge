#!/usr/bin/env bash
# ── NeuroBridge Ubuntu Setup Script ────────────────────────────────────────────
# Run: bash scripts/install_ubuntu.sh
#
# Sets up all dependencies for NeuroBridge on Ubuntu.

set -euo pipefail

echo "━━━ NeuroBridge: Ubuntu Setup ━━━"
echo ""

# ── System dependencies ───────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq \
  python3 python3-venv python3-pip \
  git curl build-essential \
  ripgrep fd-find sqlite3 \
  nodejs npm tmux

# ── Ollama ─────────────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "       Ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.com/install.sh | sh
fi

# ── Expert tools ───────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing expert bridges..."
if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code
fi
if ! command -v codex &>/dev/null; then
    npm install -g @openai/codex
fi

# ── Python venv ────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Setting up Python environment..."
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q

# ── Initialize ─────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Running neuro init..."
neuro init

echo ""
echo "━━━ Setup Complete! ━━━"
echo ""
echo "Next steps:"
echo "  1. Run the Ollama HDD migration:"
echo "     sudo bash scripts/migrate_ollama_to_hdd.sh"
echo ""
echo "  2. Pull models:"
echo "     ollama pull qwen2.5-coder:3b"
echo "     ollama pull qwen2.5-coder:7b"
echo ""
echo "  3. Check health:"
echo "     source .venv/bin/activate"
echo "     neuro config doctor"
