# Cross-Model Persona Vector Steering System

A research implementation demonstrating cross-architecture persona vector transfer for controlling personality traits in large language models.

## Overview

This system implements persona vector extraction and steering based on the methodology described in Chen et al. (2024) "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)).

**Key contribution**: Cross-model architecture steering - extracting persona vectors from one model (Qwen2.5-7B-Instruct) and applying them to control behavior in a different architecture (GPT-OSS 20B).

## Architecture

- **Vector Generation Model**: Qwen2.5-7B-Instruct (HuggingFace Transformers)
- **Steering Target Model**: GPT-OSS 20B (GGUF via llama-cpp-python)
- **Backend**: FastAPI with real-time vector steering
- **Frontend**: Bootstrap-based web interface with visualization
- **Storage**: JSON-based vector and response persistence

## Supported Personality Traits

- **Silly vs. Serious**: Controls creative/playful vs. formal behavior
- **Superficial vs. Deep**: Influences depth of analysis and response detail  
- **Inattentive vs. Focused**: Affects attention to instructions and task completion

## Installation

### Prerequisites

- Python 3.12+
- Apple Silicon Mac (for Metal acceleration) or CUDA GPU
- 16GB+ RAM recommended for 20B model inference

### Setup

```bash
git clone https://github.com/sbayer2/cross-model-persona-steering.git
cd cross-model-persona-steering
chmod +x setup_v4.sh
./setup_v4.sh
```

### Manual Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
pip install -r requirements.txt
```

### Model Download

```bash
python download_gptoss.py
```

## Usage

1. Start the server:
```bash
source venv/bin/activate
cd backend
python main.py
```

2. Open http://127.0.0.1:8000

3. Generate persona vectors with Qwen2.5-7B-Instruct

4. Test steering with GPT-OSS 20B using various coefficients

## Methodology

### Vector Extraction
- Contrastive prompt pairs generate positive/negative trait responses
- PyTorch forward hooks extract activations from transformer layers
- Persona vectors computed as activation differences: `v = positive_activations - negative_activations`

### Cross-Model Steering
- Vectors extracted from Qwen2.5 applied to GPT-OSS via parameter modulation
- Steering coefficient influences temperature, top_p, and repetition_penalty
- Positive coefficients enhance traits, negative coefficients suppress them

## Research Findings

Cross-architecture persona vector transfer demonstrates:
- Personality representations that transcend model architectures
- Effective steering of reasoning patterns and cognitive processes
- Stable parameter-based steering approach for GGUF models

## File Structure

```
backend/
├── main.py              # FastAPI application
├── models.py            # Model loading and generation
├── persona_vectors.py   # Vector extraction and steering
├── prompts.py           # Trait definitions and prompts
├── static/              # Frontend assets
├── templates/           # HTML templates
└── data/
    ├── vectors/         # Generated persona vectors
    └── responses/       # Model responses
```

## Dependencies

- PyTorch 2.2.0+ (Metal/CUDA support)
- Transformers 4.37.0+
- llama-cpp-python 0.2.0+ (Metal compilation)
- FastAPI 0.104.1+
- sentence-transformers 2.3.1+

## Acknowledgments

This work builds upon:
- Chen, R., et al. (2024). "Persona Vectors: Monitoring and Controlling Character Traits in Language Models." arXiv:2507.21509
- OpenAI GPT-OSS 20B model
- Qwen2.5-7B-Instruct by Alibaba Cloud

## License

MIT License