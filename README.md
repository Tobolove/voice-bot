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
├── app.py              # Main Gradio interface
├── setup.bat           # Windows setup (double-click)
├── setup.ps1           # PowerShell setup script
├── requirements.txt    # Python dependencies
├── voice.mp3           # Your voice sample
├── outputs/            # Generated audio files
└── IndexTTS-2-Demo/    # Cloned HF Space (after setup)
    ├── indextts/       # TTS module
    └── checkpoints/    # Model weights
```

## Credits

- [IndexTTS-2](https://github.com/index-tts/index-tts) by Bilibili
- Model: [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)
