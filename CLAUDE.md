# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Cross-Model Persona Vector Steering System that extracts personality vectors from one language model and applies them to steer the behavior of different model architectures. The system demonstrates breakthrough cross-architecture personality transfer between Qwen2.5-7B-Instruct and GPT-OSS 20B, achieving true cognitive pattern modification rather than just surface-level trait expression.

## Architecture

### Backend Components
- **FastAPI Application** (`backend/main.py`): Web server with comprehensive API endpoints
  - `models.py`: Dual-architecture model support (HuggingFace + GGUF/llama-cpp-python)
  - `persona_vectors.py`: Vector extraction engine with PyTorch hooks and steering algorithms
  - `prompts.py`: Contrastive prompt pairs and evaluation questions (built-in + dynamic custom traits)
  
### Frontend Interface
- **Dynamic Web Interface**: Bootstrap-based responsive design with real-time visualization
  - `templates/index.html`: Main application interface with modals and controls
  - `static/js/main.js`: Full application class with CustomTraitManager and VizTestSuite
  - `static/js/visualization.js`: Interactive Chart.js thermostat effect visualization
  - `static/css/style.css`: Custom styling and responsive layout

### Data Management
- **Vector Storage**: JSON files in `data/vectors/` with effectiveness scores per layer
- **Response Cache**: `data/responses/` for model outputs and steering results
- **Custom Traits**: Browser sessionStorage for dynamic trait creation and test results

## Development Commands

### Setup and Installation
```bash
# Apple Silicon optimized setup with Metal acceleration
chmod +x setup_v4.sh
./setup_v4.sh

# Manual setup if needed
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download GPT-OSS 20B model (optional - for cross-model steering)
python download_gptoss.py
```

### Running the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start the FastAPI server
cd backend
python main.py

# Access the web interface
open http://127.0.0.1:8000
```

## Core System Mechanics

### Cross-Architecture Steering
The system's breakthrough feature is personality vector transfer between different model architectures:

**Vector Extraction** (Qwen2.5-7B-Instruct):
- PyTorch forward hooks extract activations from transformer layers during inference
- Contrastive prompting generates activation differences: `positive_activations - negative_activations`
- Dynamic layer selection based on effectiveness scores (not fixed to layer 20)

**Steering Application**:
- **HuggingFace Models**: Direct activation injection at optimal layer during forward pass
- **GGUF Models (GPT-OSS)**: Parameter-based steering via temperature/top_p modulation
- Cross-model compatibility layer enables Qwen vectors to steer GPT-OSS behavior

### Model Support

**Extraction Models** (HuggingFace Transformers):
- Qwen2.5-7B-Instruct (primary)
- GPT-2, DialoGPT-medium
- BERT, Sentence Transformers
- Any AutoModelForCausalLM compatible model

**Target Models**:
- **Same Architecture**: Direct activation injection
- **GGUF Models**: GPT-OSS 20B with Metal acceleration (parameter steering)
- **Cross-Architecture**: Universal personality representation transfer

### Personality Traits

**Built-in Traits**:
- `silly`: Humorous vs serious behavior patterns
- `superficial`: Surface-level vs deep analysis approaches  
- `inattentive`: Poor vs excellent attention to detail
- `dishonest`: Deceptive vs truthful response tendencies

**Custom Trait Creation**:
- Qwen2.5-7B powered prompt generation (temperature 0.6-0.8)
- User-defined personality dimensions with contrastive descriptions
- 5-trait cache limit with automatic oldest removal
- Dynamic integration with all system features

## Advanced Features

### Dynamic Visualization System
- **Real-time Thermostat Effect**: Interactive Chart.js visualization showing coefficient impact
- **5-Point Spectrum Analysis**: Response examples across personality range (0%, 25%, 50%, 75%, 100%)
- **Adaptive Trait Labeling**: Dynamic y-axis and tooltip text based on tested trait
- **Full Context Preservation**: Complete prompts and extended response text (500 chars)

### Test Suite Management
- **Batch Testing**: Automated coefficient sweeps with progress tracking
- **Coherence Analysis**: Repetition detection, gibberish filtering, completeness scoring
- **SessionStorage Caching**: Browser-based temporary storage for test results
- **Intelligent Navigation**: Smart button routing based on cached data availability

### Vector Effectiveness Calculation
```python
def calculate_vector_effectiveness(layer_position, total_layers):
    # Middle layers contain more semantic information for personality steering
    middle_layer = total_layers // 2
    distance_from_middle = abs(layer_position - middle_layer)
    max_distance = max(middle_layer, total_layers - middle_layer)
    return 1.0 - (distance_from_middle / max_distance)
```

## Key Implementation Details

### Dual-Path Model Loading
```python
# HuggingFace models for vector extraction
if model_id.endswith('.gguf'):
    model = _load_gguf_model(model_path)  # llama-cpp-python
else:
    model = _load_hf_model(model_id)      # transformers
```

### Cross-Model Vector Transfer
```python
# Parameter-based steering for GGUF models
if model_type == "gguf":
    temperature = base_temp + (steering_coefficient * temp_modifier)
    top_p = base_top_p - (abs(steering_coefficient) * top_p_modifier)
else:
    # Direct activation injection for HuggingFace models
    steered_activations = activations + (steering_coefficient * persona_vector)
```

### Custom Trait Integration
```python
def get_all_traits():
    """Combine built-in and custom traits dynamically"""
    built_in = load_built_in_traits()
    custom = load_custom_traits()
    return {**built_in, **custom}
```

## Research Contributions

### Breakthrough Discoveries
1. **Universal Personality Representations**: Personality traits transcend specific model architectures
2. **Cognitive Pattern Modification**: Steering affects executive function and attention control, not just surface traits
3. **Parameter-Based Cross-Architecture Steering**: Stable method for applying vectors across different model types
4. **Dynamic Layer Selection**: Optimal steering layers vary by trait and model architecture

### Performance Comparisons
- **Chen et al. (2024)**: Fixed layer 20 injection, same-model steering
- **This Implementation**: Dynamic layer selection, cross-architecture transfer, 2.5x larger target model

## File Structure
```
backend/
├── main.py                 # FastAPI application with comprehensive API
├── models.py              # Dual-architecture model loading and inference
├── persona_vectors.py     # Vector extraction and steering algorithms
├── prompts.py            # Built-in traits + dynamic custom trait loading
├── templates/
│   ├── index.html        # Main web interface with modals
│   └── visualization.html # Dynamic thermostat visualization page
└── static/
    ├── css/style.css     # Custom responsive styling
    └── js/
        ├── main.js       # Application logic with CustomTraitManager
        └── visualization.js # Interactive Chart.js visualizations
data/
├── vectors/              # Generated persona vectors with effectiveness scores
├── responses/            # Model response data and steering results
└── custom_traits.json    # User-defined traits (browser sessionStorage)
setup_v4.sh              # Apple Silicon setup with Metal acceleration
download_gptoss.py        # GPT-OSS 20B model downloader
requirements.txt          # Dependencies with specific versions
```

## Dependencies
- **Core**: PyTorch 2.1.0, Transformers 4.37.0, FastAPI 0.104.1
- **GGUF Support**: llama-cpp-python with Metal acceleration
- **Visualization**: Chart.js, Bootstrap 5
- **Vector Processing**: NumPy, sentence-transformers 2.3.1

## Memory Management
- Model caching system prevents repeated loading
- Memory optimization for Apple Silicon with Metal backend
- Efficient vector storage with JSON serialization
- Browser-based sessionStorage for test result caching

## Error Handling & Production Features
- Graceful degradation if activation extraction fails
- Circuit breaker pattern for robust generation
- Comprehensive logging throughout the system
- Utility scripts for production stability (fix_logger.py, fix_tokenizer.py)

## Research Impact
This system enables researchers to:
- Study personality transfer across different model architectures
- Investigate cognitive pattern modification in large language models
- Explore universal personality representations in neural networks
- Develop more sophisticated model steering techniques

The cross-architecture breakthrough suggests fundamental insights about how personality traits are encoded in transformer architectures, opening new research directions in model interpretability and control.