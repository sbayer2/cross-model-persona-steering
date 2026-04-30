#!/usr/bin/env python3
"""
Download Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 models.
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model(repo_id, model_name):
    """Download a model from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading {model_name}...")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}\n")

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
            # Download model weights and tokenizer files
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.py", "*.txt", "*.bin"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        print(f"✅ {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def main():
    print("\n🚀 Persona Vector System - Model Downloader")
    print("=" * 60)
    print("\nThis will download:")
    print("  1. Llama-3.1-8B-Instruct (~16GB)")
    print("  2. Mistral-7B-Instruct-v0.3 (~14GB)")
    print("\nTotal space needed: ~30GB")
    print("=" * 60)

    # Check if user is logged in to HuggingFace
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"\n✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print("\n⚠️  Warning: You may need to login to HuggingFace")
        print("Run: huggingface-cli login")
        print(f"Error: {e}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Download models
    models = [
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3")
    ]

    results = {}
    for repo_id, name in models:
        success = download_model(repo_id, name)
        results[name] = success

    # Summary
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n🎉 All models downloaded successfully!")
        print("\nYou can now:")
        print("  1. Start the backend: cd backend && python main.py")
        print("  2. Open http://127.0.0.1:8000")
        print("  3. Select Llama or Mistral from the model dropdown")
    else:
        print("\n⚠️  Some downloads failed. Please check errors above.")

if __name__ == "__main__":
    main()
