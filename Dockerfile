# =============================================================================
# IndexTTS-2 Voice Bot - Docker Image
# =============================================================================
# Self-contained build that clones the IndexTTS-2-Demo HF Space at build time,
# so this image can be built on any machine — no local setup required.
#
# Base image: NVIDIA CUDA 12.1 runtime with cuDNN 8 on Ubuntu 22.04
# Python: 3.10 (Ubuntu 22.04 default)
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
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Clone IndexTTS-2-Demo from Hugging Face Space (inference code + structure)
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo /app/IndexTTS-2-Demo \
    && rm -rf /app/IndexTTS-2-Demo/.git

# ---------------------------------------------------------------------------
# Python dependencies (installed in dependency order for better layer caching)
# ---------------------------------------------------------------------------

# 1. PyTorch + torchaudio with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    torchaudio>=2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. IndexTTS-2-Demo dependencies
#    Filter out packages that break the build:
#    - deepspeed: requires nvcc (CUDA devel image), optional for inference
#    - WeTextProcessing/wetext/pynini: requires OpenFst C++ lib, optional text normalization
#    - Cython: pinned version conflicts, already satisfied by build-essential
RUN grep -v -i -E "deepspeed|WeTextProcessing|wetext|pynini|Cython" /app/IndexTTS-2-Demo/requirements.txt > /tmp/indextts-requirements.txt \
    && pip install --no-cache-dir Cython \
    && pip install --no-cache-dir -r /tmp/indextts-requirements.txt \
    && rm /tmp/indextts-requirements.txt

# 3. App-level dependencies (many overlap with above, pip will skip duplicates)
COPY requirements.txt /tmp/app-requirements.txt
RUN pip install --no-cache-dir -r /tmp/app-requirements.txt \
    && rm /tmp/app-requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY app.py /app/
COPY download_model.py /app/

# Create directories
RUN mkdir -p /app/outputs

# Copy and prepare entrypoint (sed fixes Windows CRLF line endings)
COPY docker-entrypoint.sh /app/
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

# Checkpoints volume — persists model weights across container restarts
VOLUME /app/IndexTTS-2-Demo/checkpoints

EXPOSE 7860

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--auto-load"]
