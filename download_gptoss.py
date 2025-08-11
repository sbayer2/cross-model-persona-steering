#!/usr/bin/env python3
"""
Download script for GPT-OSS 20B GGUF model
Run this from the persona_vector_system_V4 directory after setting up the environment
"""

import os
import sys

try:
    from huggingface_hub import hf_hub_download
    print("✅ huggingface_hub found")
except ImportError:
    print("❌ huggingface_hub not found")
    print("Please run: pip install huggingface_hub")
    sys.exit(1)

def download_gptoss():
    """Download GPT-OSS 20B Q4_K_M GGUF model"""
    
    # Create models directory
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    print(f"📁 Models directory: {os.path.abspath(models_dir)}")
    
    print("🔄 Downloading GPT-OSS 20B Q4_K_M GGUF (11.67 GB)...")
    print("This may take 10-30 minutes depending on your internet speed...")
    
    try:
        file_path = hf_hub_download(
            repo_id="bartowski/openai_gpt-oss-20b-GGUF",
            filename="openai_gpt-oss-20b-Q4_K_M.gguf",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            resume_download=True  # Resume if interrupted
        )
        
        print(f"✅ Successfully downloaded to: {file_path}")
        print(f"📊 File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
        
        # Update the path in models.py
        models_py_path = "./backend/models.py"
        if os.path.exists(models_py_path):
            print(f"📝 Updating model path in {models_py_path}...")
            with open(models_py_path, 'r') as f:
                content = f.read()
            
            # Replace the placeholder path
            updated_content = content.replace(
                '"/path/to/gpt-oss-20b.gguf"',
                f'"{os.path.abspath(file_path)}"'
            )
            
            with open(models_py_path, 'w') as f:
                f.write(updated_content)
            
            print("✅ Updated model path in models.py")
        
        print("🚀 GPT-OSS 20B is ready for use!")
        return file_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    download_gptoss()