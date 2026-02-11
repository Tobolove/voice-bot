"""
Download IndexTTS-2 model from Hugging Face.

This script downloads the IndexTTS-2 model weights to the checkpoints directory.
The download is approximately 5GB and may take several minutes depending on
your internet connection speed.

Usage:
    python download_model.py

The model will be downloaded to:
    IndexTTS-2-Demo/checkpoints/
"""

import os
import sys

def main():
    """
    Download the IndexTTS-2 model weights from Hugging Face Hub.

    This function uses the huggingface_hub library to download all model files
    from the IndexTeam/IndexTTS-2 repository to the local checkpoints directory.
    Progress is displayed during download.

    Returns:
        None

    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails for any reason
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Run: pip install huggingface-hub")
        sys.exit(1)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Checkpoints directory path
    checkpoints_dir = os.path.join(script_dir, "IndexTTS-2-Demo", "checkpoints")

    print("=" * 60)
    print("IndexTTS-2 Model Downloader")
    print("=" * 60)
    print(f"\nDownloading to: {checkpoints_dir}")
    print("This will download approximately 5GB of model files.")
    print("Please wait...\n")

    try:
        # Download the model
        snapshot_download(
            repo_id="IndexTeam/IndexTTS-2",
            local_dir=checkpoints_dir,
            resume_download=True
        )

        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"\nModel files saved to: {checkpoints_dir}")
        print("\nYou can now run: python app.py")

    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
