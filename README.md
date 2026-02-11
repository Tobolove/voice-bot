# IndexTTS-2 Voice Bot

A simple interface for the IndexTTS-2 text-to-speech model with voice cloning and emotion control.

## Quick Start

### 1. Run Setup (Downloads ~5GB model)

**Double-click `setup.bat`** or run in PowerShell:

```powershell
.\setup.ps1
```

This will:
- Install git-xet (for large file downloads)
- Install Hugging Face CLI
- Clone the IndexTTS-2 Demo space
- Download the model weights (~5GB)
- Install Python dependencies

### 2. Run the App

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Usage

1. **Load Model**: Click "Load Model" in Model Settings (first time only)
2. **Upload Voice**: Upload your voice sample (3-10 seconds works best)
3. **Enter Text**: Type what you want the voice to say
4. **Generate**: Click "Generate Speech"
5. **Download**: Use the download button on the audio player

## Emotion Control

| Mode | Description |
|------|-------------|
| Same as Speaker | Uses emotion from your voice reference |
| Reference Audio | Upload separate audio for emotion |
| Emotion Vector | Manual 8-dimension sliders |
| Text Description | AI detects emotion from text |

### 8 Emotion Dimensions

- Happy, Angry, Sad, Afraid
- Disgusted, Melancholic, Surprised, Calm

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Emotion Alpha | 0.6 | Emotion intensity (0-1) |
| Temperature | 1.0 | Randomness (0.1-2.0) |
| Top P | 0.9 | Nucleus sampling |
| Top K | 50 | Vocabulary limit |

## Docker (Recommended for Remote Deployment)

Run the Voice Bot in a Docker container on any Linux machine with an NVIDIA GPU — no manual Python setup needed. Model checkpoints are automatically downloaded on first launch and persisted in a Docker volume.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU passthrough)

### Quick Start

```bash
# Build the image and start the container (first run downloads ~5GB model)
docker compose up --build

# Or run in the background
docker compose up --build -d
```

The web UI will be available at **http://\<your-ip\>:9871**.

### Docker Commands

| Command | Description |
|---------|-------------|
| `docker compose up --build` | Build and start (first time) |
| `docker compose up -d` | Start in background |
| `docker compose down` | Stop the container |
| `docker compose logs -f` | Follow live logs |
| `docker compose restart` | Restart the container |

### Details

| Setting | Value |
|---------|-------|
| Container name | `tts-voice-bot` |
| Host port | `9871` |
| Container port | `7860` |
| GPU | All available NVIDIA GPUs |
| Model storage | Docker volume `tts-voice-bot-checkpoints` |
| Generated audio | Mounted at `./outputs` on the host |

The container runs with `--auto-load` by default, so the TTS model is loaded into memory at startup and ready to use immediately.

### Removing Model Data

The model checkpoints are stored in a named Docker volume. To delete them and reclaim ~5GB of disk space:

```bash
docker compose down
docker volume rm tts-voice-bot-checkpoints
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works but slower)
- ~8GB VRAM for comfortable inference
- ~10GB disk space for model

## Manual Setup (Alternative)

If setup.bat doesn't work:

```powershell
# Install HF CLI
pip install "huggingface-hub[cli,hf_xet]"

# Clone the space
git clone https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo

# Download model
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=IndexTTS-2-Demo/checkpoints

# Install dependencies
pip install -r requirements.txt
cd IndexTTS-2-Demo && pip install -r requirements.txt

# Run
cd .. && python app.py
```

## Files

```
Voice_Bot/
├── app.py                # Main Gradio interface
├── setup.bat             # Windows setup (double-click)
├── setup.ps1             # PowerShell setup script
├── requirements.txt      # Python dependencies
├── voice.mp3             # Your voice sample
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Docker orchestration config
├── docker-entrypoint.sh  # Container startup script
├── .dockerignore         # Files excluded from Docker build
├── outputs/              # Generated audio files
└── IndexTTS-2-Demo/      # Cloned HF Space (after setup)
    ├── indextts/         # TTS module
    └── checkpoints/      # Model weights (~5GB)
```

## Credits

- [IndexTTS-2](https://github.com/index-tts/index-tts) by Bilibili
- Model: [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)
