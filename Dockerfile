# =============================================================================
# IndexTTS-2 Voice Bot - Docker Image
# =============================================================================
# Multi-stage build for an optimized TTS inference container with GPU support.
#
# Base image: NVIDIA CUDA 12.1 runtime with cuDNN 8 on Ubuntu 22.04
# Python: 3.11
# GPU: NVIDIA CUDA-compatible GPU required for inference
#
# Build:
#   docker compose build
#
# The model checkpoints (~5GB) are NOT baked into the image. They are
# automatically downloaded on first run and persisted in a Docker volume.
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    curl \
    && git lfs install \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (installed in dependency order for better layer caching)
# ---------------------------------------------------------------------------

# 1. PyTorch + torchaudio with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    torchaudio>=2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. IndexTTS-2-Demo dependencies
COPY IndexTTS-2-Demo/requirements.txt /tmp/indextts-requirements.txt
RUN pip install --no-cache-dir -r /tmp/indextts-requirements.txt \
    && rm /tmp/indextts-requirements.txt

# 3. App-level dependencies (many overlap with above, pip will skip duplicates)
COPY requirements.txt /tmp/app-requirements.txt
RUN pip install --no-cache-dir -r /tmp/app-requirements.txt \
    && rm /tmp/app-requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------

# Copy the IndexTTS-2-Demo source (checkpoints excluded via .dockerignore)
COPY IndexTTS-2-Demo/ /app/IndexTTS-2-Demo/

# Copy application files
COPY app.py /app/
COPY download_model.py /app/
COPY voice.mp3 /app/

# Create outputs directory
RUN mkdir -p /app/outputs

# Copy and prepare entrypoint
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

# Checkpoints volume - persists model weights across container restarts
VOLUME /app/IndexTTS-2-Demo/checkpoints

EXPOSE 7860

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--auto-load"]
