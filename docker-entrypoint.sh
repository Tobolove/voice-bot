#!/usr/bin/env bash
# =============================================================================
# IndexTTS-2 Voice Bot - Docker Entrypoint
# =============================================================================
# This entrypoint script performs first-run setup (downloading model checkpoints
# if they are not already present) and then launches the Gradio application.
#
# The checkpoints directory is expected to be backed by a Docker volume so that
# the ~5GB download only happens once and persists across container restarts.
# =============================================================================

set -euo pipefail

CHECKPOINTS_DIR="/app/IndexTTS-2-Demo/checkpoints"
CONFIG_FILE="${CHECKPOINTS_DIR}/config.yaml"

echo "========================================"
echo "  IndexTTS-2 Voice Bot (Docker)"
echo "========================================"
echo ""

# -------------------------------------------------------------------------
# Step 1: Check / download model checkpoints
# -------------------------------------------------------------------------
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[*] Model checkpoints not found. Downloading from Hugging Face..."
    echo "    This will download ~5GB and may take 10-20 minutes on first run."
    echo ""

    huggingface-cli download \
        IndexTeam/IndexTTS-2 \
        --local-dir "${CHECKPOINTS_DIR}"

    echo ""
    echo "[*] Model download complete!"
else
    echo "[*] Model checkpoints found at: ${CHECKPOINTS_DIR}"
fi

echo ""

# -------------------------------------------------------------------------
# Step 2: Launch the application
# -------------------------------------------------------------------------
echo "[*] Starting Voice Bot on port ${GRADIO_SERVER_PORT:-7860}..."
echo ""

exec python /app/app.py "$@"
