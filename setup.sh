#!/bin/bash

# Setup script for GPT-OSS Persona Vector System
# Optimized for macOS with Apple Silicon (M-series chips)

echo "🚀 Setting up GPT-OSS Persona Vector System..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup script is optimized for macOS. Please adapt for your system."
fi

# Create virtual environment with Python 3.12+ (required for latest transformers)
echo "📦 Creating virtual environment..."
python3.12 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip to latest version
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS (Metal Performance Shaders) support for Apple Silicon
echo "🔥 Installing PyTorch with Apple Silicon optimization..."
pip install torch torchvision torchaudio

# Install transformers and quantization libraries
echo "🤖 Installing transformers and quantization support..."
pip install transformers>=4.37.0
pip install bitsandbytes>=0.42.0
pip install accelerate>=0.25.0

# Install remaining requirements
echo "📚 Installing remaining dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/vectors
mkdir -p data/responses

# Create .env file template
echo "⚙️  Creating environment configuration..."
cat > .env << EOL
# GPT-OSS Persona Vector System Configuration

# Model Configuration
MODEL_CACHE_DIR=./model_cache
DEFAULT_MODEL_ID=gpt-oss-20b

# Quantization Settings (4-bit recommended for 24GB VRAM)
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
LOAD_IN_8BIT=false
LOAD_IN_4BIT=true

# Memory Optimization
MAX_BATCH_SIZE=1
GRADIENT_CHECKPOINTING=true
DEVICE_MAP=auto

# API Configuration
HOST=127.0.0.1
PORT=8000
RELOAD=true

# Logging
LOG_LEVEL=INFO
EOL

# Create CLAUDE.md documentation
echo "📖 Creating project documentation..."
cat > CLAUDE.md << 'EOL'
# CLAUDE.md - GPT-OSS Persona Vector System

This file provides guidance to Claude Code when working with the GPT-OSS version of the persona vector system.

## Project Overview

This is a specialized version of the Persona Vector System built for the gpt-oss:20b model. It extracts and manipulates persona vectors using 4-bit quantization to work efficiently on Apple Silicon Macs with 24GB unified memory.

## Key Differences from HF Model Version

- **Target Model**: gpt-oss:20b (20 billion parameter causal language model)
- **Quantization**: 4-bit quantization using bitsandbytes
- **Architecture**: Causal LM (decoder-only) vs encoder models
- **Memory Optimization**: Designed for 24GB unified memory constraints
- **Response Generation**: True text generation vs template-based responses

## Development Commands

### Setup and Installation
```bash
# Setup the environment (creates venv and installs dependencies)
chmod +x setup.sh
./setup.sh

# Manual setup if needed
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start the FastAPI server
cd backend
python main.py

# The application runs on http://127.0.0.1:8000
```

## Core System Mechanics

### Model Loading with Quantization
The system uses 4-bit quantization via bitsandbytes to fit the 20B parameter model in 24GB VRAM:
- `load_in_4bit=True` reduces memory from ~40GB to ~10GB
- `device_map="auto"` optimizes GPU memory allocation
- `torch_dtype=torch.bfloat16` for numerical stability

### Activation Extraction for Causal LM
Unlike encoder models, we hook decoder transformer blocks:
- Hooks on `model.transformer.h[i]` (GPT-style architecture)
- Extract hidden states from the last token position
- Process activations during text generation

### Vector Generation Process
1. Load gpt-oss:20b with 4-bit quantization
2. Generate responses using contrastive prompts
3. Extract activations from transformer layers during generation
4. Compute persona vectors as activation differences
5. Score effectiveness and normalize vectors

## Memory Management

### Apple Silicon Optimization
- Unified memory architecture shares 24GB between CPU and GPU
- MPS (Metal Performance Shaders) backend for PyTorch
- Gradient checkpointing to reduce activation memory
- Batch size limited to 1 for memory efficiency

### Quantization Strategy
- 4-bit quantization reduces model size by ~75%
- BitsAndBytesConfig with optimized parameters
- Dynamic loading/unloading of model components

## Supported Personality Traits

Same traits as HF version but with improved response quality:
- **silly**: Humorous vs serious behavior
- **superficial**: Surface-level vs deep analysis  
- **inattentive**: Poor vs excellent attention to detail

## Dependencies

Key additions for gpt-oss:20b support:
- bitsandbytes>=0.42.0 for quantization
- accelerate>=0.25.0 for model loading optimization
- optimum>=1.16.0 for additional optimizations
EOL

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the development server: cd backend && python main.py"
echo "3. Open http://127.0.0.1:8000 in your browser"
echo ""
echo "💡 The system is configured for gpt-oss:20b with 4-bit quantization"
echo "📊 Expected memory usage: ~10-12GB VRAM + 4-6GB system RAM"