#!/usr/bin/env bash
# ── Migrate Ollama model storage from SSD to HDD ──────────────────────────────
# Run this script with: sudo bash scripts/migrate_ollama_to_hdd.sh
#
# What it does:
#   1. Stops Ollama service
#   2. Deletes ALL models from SSD (/usr/share/ollama/.ollama/models)
#   3. Adds OLLAMA_MODELS env var to systemd service → HDD path
#   4. Adds OLLAMA_MAX_LOADED_MODELS=1 for RAM safety (16 GB system)
#   5. Restarts Ollama pointing to HDD
#
# After running this, pull models with:
#   ollama pull qwen2.5-coder:3b
#   ollama pull qwen2.5-coder:7b

set -euo pipefail

HDD_MODELS="/media/seyominaoto/x/neurobridge/models"
SERVICE_FILE="/etc/systemd/system/ollama.service"

echo "━━━ NeuroBridge: Ollama HDD Migration ━━━"

# 1. Stop Ollama
echo "[1/5] Stopping Ollama service..."
systemctl stop ollama

# 2. Nuke SSD models
echo "[2/5] Removing ALL models from SSD..."
rm -rf /usr/share/ollama/.ollama/models/*
echo "       Cleared /usr/share/ollama/.ollama/models/"

# 3. Ensure HDD models directory exists
echo "[3/5] Preparing HDD models directory..."
mkdir -p "$HDD_MODELS"
chown -R ollama:ollama "$HDD_MODELS"

# 4. Patch systemd service (only if not already patched)
echo "[4/5] Patching Ollama systemd service..."
if grep -q "OLLAMA_MODELS" "$SERVICE_FILE"; then
    echo "       OLLAMA_MODELS already set, updating..."
    sed -i "s|Environment=\"OLLAMA_MODELS=.*\"|Environment=\"OLLAMA_MODELS=$HDD_MODELS\"|" "$SERVICE_FILE"
else
    echo "       Adding OLLAMA_MODELS=$HDD_MODELS"
    sed -i "/\[Service\]/a Environment=\"OLLAMA_MODELS=$HDD_MODELS\"" "$SERVICE_FILE"
fi

if grep -q "OLLAMA_MAX_LOADED_MODELS" "$SERVICE_FILE"; then
    echo "       OLLAMA_MAX_LOADED_MODELS already set."
else
    echo "       Adding OLLAMA_MAX_LOADED_MODELS=1"
    sed -i "/\[Service\]/a Environment=\"OLLAMA_MAX_LOADED_MODELS=1\"" "$SERVICE_FILE"
fi

# 5. Reload and restart
echo "[5/5] Reloading systemd and restarting Ollama..."
systemctl daemon-reload
systemctl start ollama
sleep 2

# Verify
echo ""
echo "━━━ Verification ━━━"
echo "Ollama status: $(systemctl is-active ollama)"
echo "Models directory: $HDD_MODELS"
echo "SSD models (should be empty):"
ls /usr/share/ollama/.ollama/models/ 2>/dev/null || echo "  (directory empty or removed)"
echo ""
echo "✅ Done! Now run:"
echo "   ollama pull qwen2.5-coder:3b"
echo "   ollama pull qwen2.5-coder:7b"
