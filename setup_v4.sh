#!/bin/bash

echo "Setting up Persona Vector System V4 with GGUF support..."

# Create virtual environment
echo "Creating virtual environment..."
python3.12 -m venv venv
source venv/bin/activate

# Install PyTorch with Metal support for Apple Silicon
echo "Installing PyTorch with Metal support..."
pip install torch torchvision torchaudio

# Install llama-cpp-python with Metal support
echo "Installing llama-cpp-python with Metal support..."
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Install other requirements
echo "Installing remaining requirements..."
pip install -r requirements.txt

echo "V4 setup complete!"
echo ""
echo "To use GPT-OSS 20B:"
echo "1. Download a GGUF version of GPT-OSS 20B"
echo "2. Update the path in backend/models.py line 84"
echo "3. Run: python backend/main.py"