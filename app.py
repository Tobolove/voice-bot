"""
IndexTTS-2 Voice Bot Application

A comprehensive text-to-speech application using the IndexTTS-2 model from Bilibili.
This application provides a simple web interface for generating speech from text using
voice cloning and emotion control capabilities.

Features:
    - Zero-shot voice cloning from any reference audio
    - 8-dimensional emotion control (Happy, Angry, Sad, Fear, Disgust, Melancholy, Surprise, Calm)
    - Multiple emotion input modes (audio reference, vector, text-based)
    - Adjustable generation parameters (temperature, top_p, top_k, etc.)
    - Real-time audio preview and download

Author: Voice Bot
License: Apache-2.0
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union

import gradio as gr
import numpy as np

# Add the IndexTTS-2-Demo directory to path (from cloned Space)
POSSIBLE_PATHS = [
    Path(__file__).parent / "IndexTTS-2-Demo",  # Cloned from HF Space
    Path(__file__).parent / "index-tts",         # Alternative name
    Path(__file__).parent,                        # Current directory
]

for path in POSSIBLE_PATHS:
    if path.exists() and (path / "indextts").exists():
        sys.path.insert(0, str(path))
        print(f"Found indextts module at: {path}")
        break

# Global model instance
tts_model = None
MODEL_LOADED = False


def find_checkpoints_dir() -> Optional[Path]:
    """
    Find the checkpoints directory by searching multiple possible locations.

    This function searches for the model checkpoints in several standard locations
    to support different installation methods (direct download, cloned space, etc.).

    Returns:
        Optional[Path]: Path to checkpoints directory if found, None otherwise

    Locations searched (in order):
        1. ./checkpoints (direct download)
        2. ./IndexTTS-2-Demo/checkpoints (cloned HF Space)
        3. ./index-tts/checkpoints (alternative clone name)
    """
    base = Path(__file__).parent
    possible_dirs = [
        base / "checkpoints",
        base / "IndexTTS-2-Demo" / "checkpoints",
        base / "index-tts" / "checkpoints",
    ]

    for d in possible_dirs:
        if d.exists() and (d / "config.yaml").exists():
            return d
    return None


def check_model_files() -> Tuple[bool, str]:
    """
    Check if all required model files are present in the checkpoints directory.

    This function verifies that the IndexTTS-2 model has been properly downloaded
    by checking for the presence of the configuration file in the checkpoints
    directory.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if model files are found, False otherwise
            - str: Status message describing what was found or what's missing

    Example:
        >>> found, message = check_model_files()
        >>> if found:
        ...     print("Model ready!")
        ... else:
        ...     print(f"Missing: {message}")
    """
    checkpoints_dir = find_checkpoints_dir()

    if checkpoints_dir is None:
        return False, "Checkpoints directory not found. Run setup.bat first!"

    return True, f"Model files found at: {checkpoints_dir}"


def load_model(
    use_fp16: bool = False,
    use_cuda_kernel: bool = False,
    use_deepspeed: bool = False
) -> Tuple[bool, str]:
    """
    Load the IndexTTS-2 model into memory.

    This function initializes the IndexTTS2 model with the specified configuration
    options. The model is loaded globally and cached for subsequent inference calls.

    Args:
        use_fp16 (bool): Enable half-precision (FP16) inference for reduced VRAM
            usage and potentially faster inference. May have minimal quality impact.
            Default is False.
        use_cuda_kernel (bool): Enable BigVGAN custom CUDA kernels for accelerated
            vocoder inference. Only works on CUDA-compatible GPUs.
            Default is False.
        use_deepspeed (bool): Enable DeepSpeed optimization for inference. May
            improve or reduce performance depending on hardware configuration.
            Default is False.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if model loaded successfully, False otherwise
            - str: Status message describing the result

    Raises:
        ImportError: If the indextts package is not properly installed
        RuntimeError: If CUDA is requested but not available

    Example:
        >>> success, message = load_model(use_fp16=True)
        >>> if success:
        ...     print("Model loaded with FP16!")
        ... else:
        ...     print(f"Failed: {message}")
    """
    global tts_model, MODEL_LOADED

    try:
        from indextts.infer_v2 import IndexTTS2

        checkpoints_dir = find_checkpoints_dir()
        if checkpoints_dir is None:
            return False, "Checkpoints not found. Run setup.bat to download the model."

        config_path = checkpoints_dir / "config.yaml"

        print(f"Loading model from: {checkpoints_dir}")
        tts_model = IndexTTS2(
            cfg_path=str(config_path),
            model_dir=str(checkpoints_dir),
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )

        MODEL_LOADED = True
        return True, f"Model loaded successfully from {checkpoints_dir}!"

    except ImportError as e:
        return False, f"Import error: {str(e)}. Make sure you ran setup.bat first."
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def normalize_emotion_vector(emotion_values: List[float]) -> List[float]:
    """
    Normalize the emotion vector to ensure values are in valid range.

    This function takes raw emotion slider values and normalizes them to ensure
    they fall within the valid range [0.0, 1.0] expected by the model.

    Args:
        emotion_values (List[float]): List of 8 emotion intensity values
            in order: [happy, angry, sad, afraid, disgusted, melancholic,
            surprised, calm]

    Returns:
        List[float]: Normalized emotion vector with values clamped to [0.0, 1.0]

    Example:
        >>> raw_emotions = [0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]
        >>> normalized = normalize_emotion_vector(raw_emotions)
        >>> print(normalized)
        [0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5]
    """
    return [max(0.0, min(1.0, v)) for v in emotion_values]


def generate_speech(
    text: str,
    voice_file: str,
    emotion_mode: str,
    emotion_audio: Optional[str],
    emotion_alpha: float,
    happy: float,
    angry: float,
    sad: float,
    afraid: float,
    disgusted: float,
    melancholic: float,
    surprised: float,
    calm: float,
    emotion_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_text_tokens: int,
    interval_silence: int,
    use_random: bool,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], str]:
    """
    Generate speech from text using the IndexTTS-2 model.

    This is the main inference function that takes text input along with a voice
    reference file and various emotion/generation parameters to synthesize speech.
    The function supports multiple emotion control modes and extensive parameter
    customization.

    Args:
        text (str): The input text to synthesize into speech. Supports Chinese,
            English, and Japanese. Long texts will be automatically segmented.
        voice_file (str): Path to the reference audio file for voice cloning.
            Accepts WAV, MP3, FLAC, and other common audio formats.
        emotion_mode (str): The emotion control method to use. One of:
            - "Same as Speaker": Use the emotion from the voice reference
            - "Reference Audio": Use a separate audio file for emotion
            - "Emotion Vector": Use explicit 8-dimensional emotion values
            - "Text Description": Detect emotion from text or description
        emotion_audio (Optional[str]): Path to emotion reference audio file.
            Only used when emotion_mode is "Reference Audio".
        emotion_alpha (float): Emotion intensity multiplier (0.0 to 1.0).
            Higher values result in more pronounced emotional expression.
        happy (float): Happiness emotion intensity (0.0 to 1.0)
        angry (float): Anger emotion intensity (0.0 to 1.0)
        sad (float): Sadness emotion intensity (0.0 to 1.0)
        afraid (float): Fear emotion intensity (0.0 to 1.0)
        disgusted (float): Disgust emotion intensity (0.0 to 1.0)
        melancholic (float): Melancholy emotion intensity (0.0 to 1.0)
        surprised (float): Surprise emotion intensity (0.0 to 1.0)
        calm (float): Calmness emotion intensity (0.0 to 1.0)
        emotion_text (str): Text description for emotion (e.g., "happy and excited").
            Only used when emotion_mode is "Text Description".
        temperature (float): Sampling temperature (0.1 to 2.0). Higher values
            increase randomness in generation.
        top_p (float): Nucleus sampling parameter (0.0 to 1.0). Lower values
            make output more deterministic.
        top_k (int): Top-k sampling parameter (0 to 100). Limits vocabulary
            to top k tokens at each step.
        max_text_tokens (int): Maximum tokens per text segment (20 to 200).
            Controls how text is chunked for processing.
        interval_silence (int): Silence duration between segments in milliseconds.
        use_random (bool): Enable stochastic generation for varied outputs.
        progress (gr.Progress): Gradio progress tracker for UI updates.

    Returns:
        Tuple[Optional[str], str]: A tuple containing:
            - Optional[str]: Path to the generated audio file, or None on failure
            - str: Status message describing the result or error

    Raises:
        ValueError: If required inputs (text or voice_file) are missing
        RuntimeError: If the model is not loaded

    Example:
        >>> audio_path, status = generate_speech(
        ...     text="Hello, world!",
        ...     voice_file="reference.wav",
        ...     emotion_mode="Emotion Vector",
        ...     emotion_alpha=0.8,
        ...     happy=0.9,
        ...     calm=0.3,
        ...     # ... other parameters with defaults
        ... )
        >>> if audio_path:
        ...     print(f"Generated: {audio_path}")
    """
    global tts_model, MODEL_LOADED

    # Validate inputs
    if not text or not text.strip():
        return None, "Please enter some text to synthesize."

    if not voice_file:
        return None, "Please upload a voice reference file."

    if not MODEL_LOADED or tts_model is None:
        return None, "Model not loaded. Please load the model first."

    try:
        progress(0.1, desc="Preparing generation...")

        # Create output path
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        output_path = output_dir / f"generated_{timestamp}.wav"

        # Prepare generation kwargs
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        # Prepare emotion parameters based on mode
        emo_audio_prompt = None
        emo_vector = None
        use_emo_text = False
        emo_text_param = None

        if emotion_mode == "Reference Audio" and emotion_audio:
            emo_audio_prompt = emotion_audio
        elif emotion_mode == "Emotion Vector":
            emo_vector = normalize_emotion_vector([
                happy, angry, sad, afraid,
                disgusted, melancholic, surprised, calm
            ])
        elif emotion_mode == "Text Description":
            use_emo_text = True
            if emotion_text and emotion_text.strip():
                emo_text_param = emotion_text.strip()

        progress(0.3, desc="Generating speech...")

        # Run inference
        tts_model.infer(
            spk_audio_prompt=voice_file,
            text=text.strip(),
            output_path=str(output_path),
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emotion_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text_param,
            use_random=use_random,
            interval_silence=interval_silence,
            max_text_tokens_per_segment=max_text_tokens,
            verbose=True,
            **generation_kwargs
        )

        progress(1.0, desc="Complete!")

        if output_path.exists():
            return str(output_path), f"Speech generated successfully! Saved to: {output_path.name}"
        else:
            return None, "Generation completed but output file not found."

    except Exception as e:
        return None, f"Error during generation: {str(e)}"


def create_interface() -> gr.Blocks:
    """
    Create the Gradio web interface for the IndexTTS-2 Voice Bot.

    This function builds and returns the complete Gradio Blocks interface with
    all input components, parameter controls, and output displays. The interface
    is organized into logical sections for ease of use.

    Returns:
        gr.Blocks: The complete Gradio interface ready to be launched

    Interface Sections:
        1. Model Loading: Controls to load the TTS model with various options
        2. Input Section: Text input and voice reference upload
        3. Emotion Control: Multiple modes for controlling emotional expression
        4. Advanced Parameters: Generation settings like temperature, top_p, etc.
        5. Output Section: Audio playback and download

    Example:
        >>> interface = create_interface()
        >>> interface.launch(server_name="0.0.0.0", server_port=7860)
    """

    # Custom CSS for better styling
    css = """
    .main-title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .parameter-group {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    """

    with gr.Blocks(css=css, title="IndexTTS-2 Voice Bot") as interface:
        gr.Markdown(
            """
            # IndexTTS-2 Voice Bot

            Generate natural speech with voice cloning and emotion control.
            Upload a voice reference, enter your text, and customize the output!
            """
        )

        # Model Loading Section
        with gr.Accordion("Model Settings", open=False):
            with gr.Row():
                use_fp16 = gr.Checkbox(label="Use FP16 (Lower VRAM)", value=False)
                use_cuda_kernel = gr.Checkbox(label="Use CUDA Kernel", value=False)
                use_deepspeed = gr.Checkbox(label="Use DeepSpeed", value=False)

            with gr.Row():
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)

            load_btn.click(
                fn=load_model,
                inputs=[use_fp16, use_cuda_kernel, use_deepspeed],
                outputs=[gr.State(), model_status]
            )

        # Main Input Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5,
                    max_lines=10
                )

                voice_file = gr.Audio(
                    label="Voice Reference (Upload your voice sample)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

            with gr.Column(scale=1):
                gr.Markdown("### Emotion Control")

                emotion_mode = gr.Radio(
                    choices=[
                        "Same as Speaker",
                        "Reference Audio",
                        "Emotion Vector",
                        "Text Description"
                    ],
                    value="Same as Speaker",
                    label="Emotion Mode"
                )

                emotion_audio = gr.Audio(
                    label="Emotion Reference Audio",
                    type="filepath",
                    visible=False
                )

                emotion_alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Emotion Intensity (Alpha)"
                )

                # Emotion Vector Sliders
                with gr.Group(visible=False) as emotion_vector_group:
                    gr.Markdown("#### Emotion Vector (8 dimensions)")
                    with gr.Row():
                        happy = gr.Slider(0, 1, 0, step=0.05, label="Happy")
                        angry = gr.Slider(0, 1, 0, step=0.05, label="Angry")
                    with gr.Row():
                        sad = gr.Slider(0, 1, 0, step=0.05, label="Sad")
                        afraid = gr.Slider(0, 1, 0, step=0.05, label="Afraid")
                    with gr.Row():
                        disgusted = gr.Slider(0, 1, 0, step=0.05, label="Disgusted")
                        melancholic = gr.Slider(0, 1, 0, step=0.05, label="Melancholic")
                    with gr.Row():
                        surprised = gr.Slider(0, 1, 0, step=0.05, label="Surprised")
                        calm = gr.Slider(0, 1, 0.5, step=0.05, label="Calm")

                emotion_text = gr.Textbox(
                    label="Emotion Description",
                    placeholder="e.g., 'happy and excited' or 'sad and melancholic'",
                    visible=False
                )

        # Advanced Parameters
        with gr.Accordion("Advanced Parameters", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Temperature (higher = more random)"
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P (nucleus sampling)"
                )

            with gr.Row():
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top K"
                )
                max_text_tokens = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=120,
                    step=10,
                    label="Max Tokens per Segment"
                )

            with gr.Row():
                interval_silence = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=200,
                    step=50,
                    label="Silence Between Segments (ms)"
                )
                use_random = gr.Checkbox(
                    label="Enable Random Sampling",
                    value=False
                )

        # Generate Button and Output
        with gr.Row():
            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Speech",
                    type="filepath",
                    interactive=False
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )

        # Dynamic visibility based on emotion mode
        def update_emotion_controls(mode):
            """
            Update the visibility of emotion control components based on selected mode.

            Args:
                mode (str): The selected emotion control mode

            Returns:
                Tuple: Visibility states for emotion_audio, emotion_vector_group, emotion_text
            """
            return (
                gr.update(visible=(mode == "Reference Audio")),  # emotion_audio
                gr.update(visible=(mode == "Emotion Vector")),   # emotion_vector_group
                gr.update(visible=(mode == "Text Description"))  # emotion_text
            )

        emotion_mode.change(
            fn=update_emotion_controls,
            inputs=[emotion_mode],
            outputs=[emotion_audio, emotion_vector_group, emotion_text]
        )

        # Generate button click
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                voice_file,
                emotion_mode,
                emotion_audio,
                emotion_alpha,
                happy, angry, sad, afraid,
                disgusted, melancholic, surprised, calm,
                emotion_text,
                temperature,
                top_p,
                top_k,
                max_text_tokens,
                interval_silence,
                use_random
            ],
            outputs=[output_audio, status_text]
        )

        # Instructions
        gr.Markdown(
            """
            ---
            ### Instructions

            1. **Load Model**: Click "Load Model" in Model Settings (only needed once)
            2. **Upload Voice**: Upload a reference audio file (your voice sample)
            3. **Enter Text**: Type the text you want to synthesize
            4. **Adjust Emotions** (optional): Choose emotion mode and adjust settings
            5. **Generate**: Click "Generate Speech" and wait for the audio
            6. **Download**: Use the download button on the audio player

            ### Emotion Modes

            - **Same as Speaker**: Uses the emotion from your voice reference
            - **Reference Audio**: Upload a separate audio for emotion reference
            - **Emotion Vector**: Manually set 8 emotion intensities
            - **Text Description**: Let the model detect emotion from text

            ### Tips

            - Use 3-10 second voice samples for best results
            - Keep emotion alpha around 0.6 for natural results
            - Lower temperature (0.7-0.9) for more consistent output
            - The melancholy slider works well for natural, subdued speech
            """
        )

    return interface


def main():
    """
    Main entry point for the IndexTTS-2 Voice Bot application.

    This function performs the following steps:
        1. Checks for required model files
        2. Optionally auto-loads the model
        3. Creates and launches the Gradio web interface

    The application will be accessible at http://localhost:7860 by default.

    Command Line Arguments:
        --share: Create a public Gradio link for sharing
        --port: Specify a custom port number

    Example:
        >>> python app.py
        >>> # Or with sharing enabled:
        >>> python app.py --share
    """
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS-2 Voice Bot")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--auto-load", action="store_true", help="Auto-load model on startup")
    args = parser.parse_args()

    # Check model files
    found, message = check_model_files()
    print(f"Model check: {message}")

    if not found:
        print("\n" + "="*60)
        print("MODEL NOT FOUND - Please download the model first!")
        print("="*60)
        print("\nRun these commands to download:")
        print("  pip install huggingface-hub[cli,hf_xet]")
        print("  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints")
        print("\nOr manually download from:")
        print("  https://huggingface.co/IndexTeam/IndexTTS-2")
        print("="*60 + "\n")

    # Auto-load model if requested
    if args.auto_load and found:
        print("Auto-loading model...")
        success, msg = load_model()
        print(msg)

    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
