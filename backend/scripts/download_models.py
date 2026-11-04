#!/usr/bin/env python3
"""
Download pre-trained model weights from Hugging Face Hub.

Run this script to download the required ONNX model files for the backend.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


def download_models():
    """Download model weights from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, HfApi

    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading models to: {models_dir}")

    # Model configurations
    models_to_download = [
        {
            "repo_id": "deepfake-detection/efficientnet-b0-faceforensics",
            "filename": "model.onnx",
            "local_filename": settings.EFFICIENTNET_MODEL_PATH,
            "description": "EfficientNet-B0 fine-tuned on FaceForensics++"
        },
        {
            "repo_id": "deepfake-detection/mesonet-faceforensics",
            "filename": "model.onnx",
            "local_filename": settings.MESONET_MODEL_PATH,
            "description": "MesoNet for deepfake detection"
        }
    ]

    success_count = 0

    for model_config in models_to_download:
        local_path = models_dir / model_config["local_filename"]

        if local_path.exists():
            print(f"✓ {model_config['local_filename']} already exists, skipping...")
            success_count += 1
            continue

        print(f"\nDownloading {model_config['description']}...")
        print(f"  From: {model_config['repo_id']}/{model_config['filename']}")

        try:
            downloaded_path = hf_hub_download(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"],
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )

            # Rename to expected filename if different
            downloaded_file = Path(downloaded_path)
            if downloaded_file.name != model_config["local_filename"]:
                target_path = models_dir / model_config["local_filename"]
                if target_path.exists():
                    target_path.unlink()
                downloaded_file.rename(target_path)
                print(f"  Renamed to: {model_config['local_filename']}")

            print(f"  ✓ Saved to: {local_path}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Failed to download: {e}")
            print(f"  Note: Model repo '{model_config['repo_id']}' may not exist yet.")
            print(f"  You may need to train your own models or use alternative sources.")

    print(f"\n{'='*50}")
    print(f"Download complete: {success_count}/{len(models_to_download)} models ready")

    if success_count < len(models_to_download):
        print("\nMissing models - the backend will use dummy predictors for testing.")
        print("For production, you need to:")
        print("  1. Train your own models on FaceForensics++ dataset")
        print("  2. Convert to ONNX format")
        print("  3. Place .onnx files in ./models/ directory")
        return 1

    return 0


def verify_models():
    """Verify that model files exist and are valid ONNX."""
    import onnx

    models_dir = settings.MODELS_DIR
    model_files = [
        settings.EFFICIENTNET_MODEL_PATH,
        settings.MESONET_MODEL_PATH
    ]

    print("Verifying model files...")
    all_valid = True

    for model_file in model_files:
        path = models_dir / model_file
        if not path.exists():
            print(f"  ✗ {model_file}: NOT FOUND")
            all_valid = False
            continue

        try:
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            print(f"  ✓ {model_file}: Valid ONNX model")
            print(f"    Inputs: {[i.name for i in model.graph.input]}")
            print(f"    Outputs: {[o.name for o in model.graph.output]}")
        except Exception as e:
            print(f"  ✗ {model_file}: Invalid ONNX - {e}")
            all_valid = False

    return all_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download/verify deepfake detection models")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing models")
    args = parser.parse_args()

    if args.verify_only:
        sys.exit(0 if verify_models() else 1)
    else:
        sys.exit(download_models())