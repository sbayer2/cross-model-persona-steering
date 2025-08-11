#!/usr/bin/env python3
"""
Download script for GPT-OSS 20B model.
Downloads the GGUF quantized version optimized for Metal acceleration.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def download_with_progress(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def main():
    print("🦙 GPT-OSS 20B Model Download")
    print("   Downloading GGUF quantized version optimized for Apple Silicon")
    print()
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model information
    model_filename = "openai_gpt-oss-20b-Q4_K_M.gguf"
    model_path = models_dir / model_filename
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists at: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024**3:.1f} GB")
        
        overwrite = input("\nOverwrite existing model? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("   Keeping existing model.")
            return
        else:
            print("   Removing existing model...")
            model_path.unlink()
    
    print(f"📥 Downloading {model_filename}...")
    print(f"   Destination: {model_path}")
    print(f"   Expected size: ~12 GB")
    print()
    
    try:
        # Download from HuggingFace Hub
        repo_id = "openai/gpt-oss-20b"
        
        print(f"🔍 Attempting to download from HuggingFace Hub: {repo_id}")
        
        # Try to download using hf_hub_download
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Successfully downloaded to: {downloaded_path}")
            
        except Exception as hf_error:
            print(f"❌ HuggingFace download failed: {hf_error}")
            print("\n🔄 Trying alternative download method...")
            
            # Alternative: direct download from known URL
            # Note: This is a placeholder - you'd need the actual download URL
            alt_url = "https://example.com/path/to/model.gguf"  # Replace with actual URL
            print(f"   URL: {alt_url}")
            
            # For now, provide manual instructions
            print("\n📋 Manual Download Instructions:")
            print("   1. Visit: https://huggingface.co/openai/gpt-oss-20b")
            print("   2. Download the GGUF quantized version")
            print(f"   3. Place the file at: {model_path}")
            print(f"   4. Ensure the filename is: {model_filename}")
            print()
            print("   Alternative repositories to check:")
            print("   - https://huggingface.co/TheBloke/gpt-oss-20b-GGUF")
            print("   - https://huggingface.co/Microsoft/gpt-oss-20b")
            return
        
        # Update the model path in models.py if needed
        update_model_path(model_path)
        
        print("\n✅ Download complete!")
        print(f"   Model location: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024**3:.1f} GB")
        print()
        print("🚀 You can now start the application with:")
        print("   cd backend && python main.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
        if model_path.exists() and model_path.stat().st_size == 0:
            model_path.unlink()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if model_path.exists():
            print(f"   Removing incomplete file: {model_path}")
            model_path.unlink()
        sys.exit(1)

def update_model_path(model_path):
    """Update the model path in models.py configuration."""
    models_py = Path("backend/models.py")
    
    if not models_py.exists():
        print("⚠️  backend/models.py not found, skipping path update")
        return
    
    try:
        # Read the current file
        with open(models_py, 'r') as f:
            content = f.read()
        
        # Update the GPT-OSS model path
        old_path_line = '"path": "/Users/sbm4_mac/Desktop/persona_vector_system_V4/models/openai_gpt-oss-20b-Q4_K_M.gguf"'
        new_path_line = f'"path": "{model_path.absolute()}"'
        
        if old_path_line in content:
            content = content.replace(old_path_line, new_path_line)
            
            with open(models_py, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated model path in {models_py}")
        else:
            print(f"⚠️  Could not find model path to update in {models_py}")
            print(f"   Please manually update the 'gpt-oss-20b' path to: {model_path.absolute()}")
            
    except Exception as e:
        print(f"⚠️  Error updating model path: {e}")
        print(f"   Please manually update backend/models.py with path: {model_path.absolute()}")

if __name__ == "__main__":
    main()
