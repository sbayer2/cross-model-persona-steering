#!/bin/bash
set -e

echo "🚀 Setting up Cross-Model Persona Vector Steering System V4"
echo "   Optimized for Apple Silicon with Metal acceleration"
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup script is optimized for macOS/Apple Silicon."
    echo "   For other systems, please install dependencies manually."
fi

# Check Python version
echo "🔍 Checking Python installation..."
if ! command -v python3.12 &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3.12+ required. Please install Python first."
        echo "   Recommended: brew install python@3.12"
        exit 1
    else
        PYTHON_CMD=python3
        echo "✅ Found Python 3 (using python3 command)"
    fi
else
    PYTHON_CMD=python3.12
    echo "✅ Found Python 3.12"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with Metal support for Apple Silicon
echo "🔥 Installing PyTorch with Metal acceleration..."
pip install torch torchvision torchaudio

# Install llama-cpp-python with Metal support
echo "🦙 Installing llama-cpp-python with Metal acceleration..."
echo "   This may take several minutes to compile..."
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Install other requirements
echo "📚 Installing remaining dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/vectors
mkdir -p data/responses
mkdir -p models
mkdir -p logs

# Download GPT-OSS model if desired
echo ""
echo "📥 Would you like to download the GPT-OSS 20B model now?"
echo "   Size: ~12GB, requires good internet connection"
read -p "Download now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⬇️  Downloading GPT-OSS 20B model..."
    python download_gptoss.py
else
    echo "⏭️  Skipping model download. Run 'python download_gptoss.py' later to download."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   1. source venv/bin/activate"
echo "   2. cd backend"
echo "   3. python main.py"
echo ""
echo "🌐 The application will be available at: http://127.0.0.1:8000"
echo ""
echo "📖 For more information, see README.md"
