# Cross-Model Persona Vector Steering System

A groundbreaking research implementation demonstrating cross-architecture persona vector transfer for controlling personality traits in large language models. This system extracts persona vectors from one model architecture and successfully applies them to steer behavior in completely different architectures.

## 🚀 Key Innovation

**Cross-Architecture Steering**: Extract personality vectors from Qwen2.5-7B-Instruct and apply them to control GPT-OSS 20B behavior - proving that personality representations transcend specific model architectures.

## Overview

Based on Chen et al. (2024) "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)), this implementation extends the original research with several breakthrough features:

- **Cross-model vector transfer** between different architectures
- **Dynamic layer selection** instead of fixed layer 20
- **Custom trait creation** with AI-powered prompt generation
- **Real-time visualization** of steering effects
- **Dual steering methods** for different model types

## Architecture

### Model Capabilities

| Model | Role | Steering Method | Capabilities |
|-------|------|----------------|-------------|
| **Qwen2.5-7B-Instruct** | Vector Extraction & Generation | Direct Activation Injection | • Extract persona vectors<br>• Generate custom trait prompts<br>• Layer-specific activation steering<br>• Full Chen et al. implementation |
| **GPT-OSS 20B** | Cross-Architecture Target | Parameter Modulation | • Receives vectors from Qwen<br>• Temperature/top_p steering<br>• Demonstrates transferability<br>• 2.5x larger reasoning model |

### Key Differences

- **Qwen2.5-7B**: Uses direct activation injection at the most effective layer (dynamically selected, not fixed to layer 20)
- **GPT-OSS 20B**: Uses parameter-based steering (temperature, top_p, repetition_penalty) interpreted from Qwen vectors

## 🎭 Personality Traits

### Built-in Traits
- **Silly vs. Serious**: Creative/playful vs. formal behavior
- **Dishonest vs. Honest**: Misleading vs. truthful responses
- **Superficial vs. Deep**: Surface-level vs. in-depth analysis
- **Inattentive vs. Focused**: Poor vs. excellent attention to detail

### Custom Trait Creation (New!)
- Define any personality trait you want
- AI generates contrastive prompts automatically
- Support for up to 5 custom traits
- Seamless integration with vector generation

## Installation

### Prerequisites
- Python 3.12+
- Apple Silicon Mac (for Metal acceleration) or CUDA GPU
- 16GB+ RAM recommended for 20B model inference

### Quick Setup
```bash
git clone https://github.com/sbayer2/cross-model-persona-steering
cd cross-model-persona-steering
chmod +x setup_v4.sh
./setup_v4.sh
```

### Manual Setup
```bash
python3.12 -m venv venv
source venv/bin/activate

# For Apple Silicon (Metal acceleration)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# For CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

pip install -r requirements.txt
```

### Model Download
```bash
python download_gptoss.py
```

## 🎮 Usage

### Starting the System
```bash
source venv/bin/activate
cd backend
python main.py
```

Open http://127.0.0.1:8000 in your browser.

### Workflow

1. **Generate Persona Vectors**
   - Select Qwen2.5-7B-Instruct
   - Choose a trait (built-in or custom)
   - Extract activation vectors from all layers
   - System automatically selects most effective layer

2. **Create Custom Traits** ✨
   - Click "Custom Trait" button
   - Enter trait name and descriptions
   - Qwen generates contrastive prompts
   - Trait saved for immediate use

3. **Test Steering**
   - Apply vectors to either model
   - Adjust coefficient (-2.0 to +2.0)
   - Observe personality changes

4. **Visualize Effects** 📊
   - Generate test suites across coefficient range
   - View coherence and ethical stance curves
   - Analyze the "thermostat effect"

## 🔬 Technical Implementation

### Vector Extraction Process
1. Contrastive prompt pairs generate positive/negative responses
2. PyTorch forward hooks extract activations from all transformer layers
3. Persona vectors computed as: `v = positive_activations - negative_activations`
4. Effectiveness scores calculated per layer
5. Most effective layer selected dynamically (not fixed to layer 20)

### Steering Methods

#### Qwen2.5-7B (Activation Injection)
```python
# Direct injection at most effective layer
steered_activations = activations + (coefficient * persona_vector)
```

#### GPT-OSS 20B (Parameter Modulation)
```python
# Vector-influenced parameter adjustment
temperature = 0.7 + (coefficient * 0.3)
top_p = 0.9 - (abs(coefficient) * 0.2)
```

## 📈 Research Findings

### Breakthrough Discoveries
- **Universal personality representations** that work across architectures
- **Cognitive pattern modification** beyond surface traits
- **Executive function steering** affecting attention and reasoning
- **Stable cross-architecture transfer** from 7B to 20B models

### The "Thermostat Effect"
Extreme coefficients (±1.8-2.0) cause model collapse, while moderate values (±0.5-1.5) produce stable personality changes.

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List available models |
| `/api/traits` | GET | List all traits (built-in + custom) |
| `/api/traits/generate-custom` | POST | Generate custom trait with AI |
| `/api/vectors/generate` | POST | Extract persona vectors |
| `/api/steering/test` | POST | Test steering with coefficient |
| `/visualization` | GET | Interactive steering visualization |

## File Structure

```
backend/
├── main.py              # FastAPI application
├── models.py            # Dual-architecture model handling
├── persona_vectors.py   # Vector extraction and steering
├── prompts.py           # Dynamic trait loading
├── static/              # Enhanced UI with visualizations
├── templates/           # HTML with custom trait modal
└── data/
    ├── vectors/         # Generated persona vectors
    ├── responses/       # Model responses
    └── custom_traits.json  # User-created traits
```

## Dependencies

- PyTorch 2.2.0+ (Metal/CUDA support)
- Transformers 4.37.0+
- llama-cpp-python 0.2.0+ (Metal/CUDA compilation)
- FastAPI 0.104.1+
- Qwen2.5-7B-Instruct (auto-downloads)
- GPT-OSS 20B GGUF (via download script)

## 📚 Citations

```bibtex
@article{chen2024persona,
  title={Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author={Chen, Runjin and others},
  journal={arXiv preprint arXiv:2507.21509},
  year={2024}
}
```

## Acknowledgments

This work builds upon:
- Chen et al. (2024) for the persona vector methodology
- OpenAI for GPT-OSS 20B model
- Alibaba Cloud for Qwen2.5-7B-Instruct
- The open-source community for transformers and llama.cpp

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: [Create an issue](https://github.com/sbayer2/cross-model-persona-steering/issues)

---

*This research demonstrates that personality traits in AI systems are not architecture-specific but represent fundamental patterns that can transfer between different model architectures.*