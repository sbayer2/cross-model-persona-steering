"""
Model loading and inference for multiple architectures:
- HuggingFace transformers (Qwen, GPT-2) 
- GGUF models via llama.cpp (GPT-OSS 20B)
Optimized for Apple Silicon Macs with Metal acceleration.
"""
import os
import json
import logging
import asyncio
import time
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import Accelerator

# GGUF support for GPT-OSS 20B
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Available models - focused on gpt-oss:20b
AVAILABLE_MODELS = {
    "qwen2.5-7b-instruct": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5-7B-Instruct (7B Parameters) - Research Paper Model",
        "quantization": "none",  # Disabled for Apple Silicon
        "max_length": 4096,
        "device_map": "auto"
    },
    "gpt2-medium": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer", 
        "path": "gpt2-medium",
        "description": "GPT-2 Medium (355M Parameters, 24 Layers)",
        "quantization": "none",
        "max_length": 1024,
        "device_map": "auto"
    },
    "gpt2": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "path": "gpt2", 
        "description": "GPT-2 Base (117M Parameters, 12 Layers) - Small",
        "quantization": "none",
        "max_length": 1024,
        "device_map": "auto"
    },
    "dialogpt-medium": {
        "model_type": "causal_lm",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "path": "microsoft/DialoGPT-medium", 
        "description": "DialoGPT Medium (355M) - Conversational Focused",
        "quantization": "none",
        "max_length": 1024,
        "device_map": "auto"
    },
    "gpt-oss-20b": {
        "model_type": "gguf",
        "model_class": "Llama",
        "tokenizer_class": "AutoTokenizer",
        "path": "/Users/sbm4_mac/Desktop/persona_vector_system_V4/models/openai_gpt-oss-20b-Q4_K_M.gguf",  # User will need to set this
        "tokenizer_path": "openai/gpt-oss-20b",  # For tokenizer compatibility
        "description": "GPT-OSS 20B (GGUF) - Metal Accelerated MXFP4",
        "quantization": "mxfp4",
        "max_length": 2048,
        "context_length": 4096,
        "gpu_layers": -1,  # Use all GPU layers on Metal
        "threads": 8
    }
}

# Store loaded models and tokenizers
_loaded_models = {}
_loaded_tokenizers = {}
_model_configs = {}

def _get_quantization_config():
    """Get BitsAndBytesConfig for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

def _get_model_config(model_id):
    """Get model configuration for a given model ID."""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_id} not available. Available: {list(AVAILABLE_MODELS.keys())}")
    
    return AVAILABLE_MODELS[model_id]

async def get_available_models():
    """Get list of available models."""
    return [
        {
            "id": model_id,
            "name": config["description"],
            "type": config["model_type"]
        }
        for model_id, config in AVAILABLE_MODELS.items()
        if config["model_type"] != "info"  # Filter out info-only entries
    ]

def _load_model(model_id):
    """
    Load a model and tokenizer supporting both HuggingFace and GGUF formats.
    """
    if model_id in _loaded_models:
        logger.info(f"Model {model_id} already loaded, returning cached version")
        return _loaded_models[model_id], _loaded_tokenizers[model_id]
    
    config = _get_model_config(model_id)
    model_type = config["model_type"]
    logger.info(f"Loading {model_id} (type: {model_type})...")
    
    if model_type == "gguf":
        # Handle GGUF models via llama.cpp
        return _load_gguf_model(model_id, config)
    else:
        # Handle HuggingFace models (causal_lm)
        return _load_hf_model(model_id, config)

def _load_gguf_model(model_id, config):
    """Load GGUF model via llama.cpp with Metal acceleration."""
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
    
    logger.info("Loading GGUF model with Metal acceleration...")
    
    # Load tokenizer from HuggingFace for compatibility
    tokenizer_path = config.get("tokenizer_path", config["path"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load GGUF model with Metal acceleration
    model = Llama(
        model_path=config["path"],
        n_ctx=config.get("context_length", 4096),
        n_gpu_layers=config.get("gpu_layers", -1),  # Use all GPU layers
        n_threads=config.get("threads", 8),
        verbose=False,
        use_mmap=True,
        use_mlock=True
    )
    
    _loaded_models[model_id] = model
    _loaded_tokenizers[model_id] = tokenizer
    
    logger.info(f"✅ Successfully loaded GGUF model {model_id}")
    return model, tokenizer

def _load_hf_model(model_id, config):
    """Load HuggingFace transformers model."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["path"],
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model (this may take a few minutes for first download)...")
    
    # Determine best device
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        logger.info("Using CUDA GPU")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU")
    
    try:
        if "qwen" in config["path"].lower() and torch.backends.mps.is_available():
            # Special handling for large Qwen models on Apple Silicon
            logger.info("Loading Qwen model with Apple Silicon optimizations...")
            model = AutoModelForCausalLM.from_pretrained(
                config["path"],
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None
            )
            model = model.to(device)
            logger.info(f"✅ Successfully loaded {config['path']} on {device}")
        else:
            # Simple loading for smaller models
            model = AutoModelForCausalLM.from_pretrained(
                config["path"],
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            logger.info(f"✅ Successfully loaded {config['path']} on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    _loaded_models[model_id] = model
    _loaded_tokenizers[model_id] = tokenizer
    
    logger.info(f"Successfully loaded {model_id}")
    return model, tokenizer

def _attach_activation_hooks(model, model_id):
    """
    Attach hooks to extract activations from transformer layers.
    
    For GPT-OSS (causal LM), we hook the transformer blocks.
    Architecture: model.transformer.h[i] contains the transformer layers.
    """
    activations = {}
    config = _get_model_config(model_id)
    
    def hook_fn(layer_name):
        def hook(module, input, output):
            # For causal LM, output is typically a tuple (hidden_states, ...)
            # or just hidden_states tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Extract activations from the last token position (sequence dimension)
            # Shape: [batch_size, sequence_length, hidden_size]
            if len(hidden_states.shape) == 3:
                # Take the last token's hidden states
                last_token_activations = hidden_states[:, -1, :]  # [batch_size, hidden_size]
                activations[layer_name] = last_token_activations.detach().clone().cpu().numpy()
            else:
                # Fallback: take the whole tensor
                activations[layer_name] = hidden_states.detach().clone().cpu().numpy()
                
        return hook
    
    hooks = []
    
    # Inspect the model structure first
    logger.info("Inspecting model architecture...")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
    
    # Try multiple possible layer access patterns for different architectures
    transformer_layers = None
    layer_path = None
    
    # Pattern 1: model.transformer.h (GPT-style)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h
        layer_path = "model.transformer.h"
    # Pattern 2: model.model.layers (Llama-style)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layers = model.model.layers
        layer_path = "model.model.layers"
    # Pattern 3: model.h (Direct access)
    elif hasattr(model, 'h'):
        transformer_layers = model.h
        layer_path = "model.h"
    # Pattern 4: model.layers (Direct layers)
    elif hasattr(model, 'layers'):
        transformer_layers = model.layers
        layer_path = "model.layers"
    
    if transformer_layers is not None:
        logger.info(f"Found {len(transformer_layers)} transformer layers at {layer_path}")
        
        for i, layer in enumerate(transformer_layers):
            hook = layer.register_forward_hook(hook_fn(f"layer_{i}"))
            hooks.append(hook)
            
        logger.info(f"Successfully hooked {len(hooks)} transformer layers")
    else:
        # Manual discovery as fallback
        logger.warning("Standard transformer layer paths not found, attempting manual discovery...")
        layer_count = 0
        
        for name, module in model.named_modules():
            # Look for transformer blocks/layers
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'h.']):
                # Skip embedding, norm, and other non-transformer layers
                if not any(skip in name.lower() for skip in ['embed', 'norm', 'ln', 'head', 'pooler']):
                    logger.info(f"Discovered potential layer: {name}")
                    hook = module.register_forward_hook(hook_fn(f"discovered_{name}"))
                    hooks.append(hook)
                    layer_count += 1
                    
                    # Limit to reasonable number of layers
                    if layer_count >= 50:
                        break
        
        logger.info(f"Discovered and hooked {layer_count} potential layers")
    
    return activations, hooks

def _detach_hooks(hooks):
    """Remove hooks from the model."""
    for hook in hooks:
        hook.remove()

def _generate_text_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens=150, persona_vectors=None, steering_coefficient=0.0):
    """
    Generate text response using either HuggingFace or GGUF models with optional persona vector steering.
    
    Args:
        model: The loaded model (HuggingFace or llama.cpp)
        tokenizer: The tokenizer for the model
        model_id: Model identifier to determine generation method
        system_prompt: System prompt to guide behavior
        user_prompt: User's question/prompt
        max_new_tokens: Maximum number of new tokens to generate
        persona_vectors: Dict of layer_name -> numpy vector for steering
        steering_coefficient: Strength of steering (-2.0 to 2.0)
        
    Returns:
        Generated response text
    """
    config = _get_model_config(model_id)
    model_type = config["model_type"]
    
    # Handle GGUF models differently from HuggingFace models
    if model_type == "gguf":
        return _generate_gguf_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens, persona_vectors, steering_coefficient)
    else:
        return _generate_hf_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens, persona_vectors, steering_coefficient)

def _generate_gguf_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens=150, persona_vectors=None, steering_coefficient=0.0):
    """Generate response using llama.cpp GGUF model."""
    # Construct the full prompt
    if "gpt-oss" in model_id.lower() or "gpt" in model_id.lower():
        # GPT-style format
        full_prompt = f"{system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"
    else:
        # Generic format
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    # Calculate generation parameters based on persona vectors
    # Lower base temperature for more consistent baseline behavior
    base_temp = 0.3
    base_top_p = 0.8
    base_rep_penalty = 1.05
    
    if persona_vectors and steering_coefficient != 0.0:
        # Use persona vector characteristics to influence generation
        target_layer = len(list(persona_vectors.keys())) // 2
        target_layer_name = f"layer_{target_layer}"
        
        if target_layer_name in persona_vectors:
            vector = persona_vectors[target_layer_name]
            if isinstance(vector, np.ndarray):
                magnitude_factor = min(np.linalg.norm(vector) / 100.0, 1.5)
                
                if steering_coefficient > 0:
                    # More creative/silly
                    temperature = min(base_temp * (1 + magnitude_factor * abs(steering_coefficient) * 0.5), 1.2)
                    top_p = min(base_top_p * (1 + abs(steering_coefficient) * 0.1), 0.95)
                    repetition_penalty = max(base_rep_penalty * (1 - abs(steering_coefficient) * 0.1), 0.9)
                else:
                    # More serious/focused
                    temperature = max(base_temp * (1 - magnitude_factor * abs(steering_coefficient) * 0.3), 0.3)
                    top_p = max(base_top_p * (1 - abs(steering_coefficient) * 0.1), 0.7)
                    repetition_penalty = min(base_rep_penalty * (1 + abs(steering_coefficient) * 0.1), 1.3)
                
                logger.info(f"GGUF persona steering: temp={temperature:.2f}, top_p={top_p:.2f}, rep_penalty={repetition_penalty:.2f}")
            else:
                temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
        else:
            temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
    else:
        temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
    
    try:
        # Clear any previous context to prevent cumulative effects
        if hasattr(model, 'reset'):
            model.reset()
        elif hasattr(model, '_ctx'):
            # Force context reset for llama.cpp
            try:
                model._ctx.reset()
            except:
                pass  # Context reset failed, continue anyway
        
        # Generate using llama.cpp API with fresh context
        response = model(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stop=["Human:", "User:", "\nHuman:", "\nUser:"],
            echo=False,
            # Force fresh generation, no context reuse
            seed=-1  # Random seed each time
        )
        
        # Extract the generated text
        if isinstance(response, dict) and 'choices' in response:
            generated_text = response['choices'][0]['text']
        else:
            generated_text = str(response)
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"GGUF generation error: {e}")
        raise e

def _generate_hf_response(model, tokenizer, model_id, system_prompt, user_prompt, max_new_tokens=150, persona_vectors=None, steering_coefficient=0.0):
    """Generate response using HuggingFace transformers model."""
    # Construct the full prompt based on model type
    if "qwen" in model_id.lower():
        # Qwen chat format
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in model_id.lower():
        # Llama chat format  
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        # Generic format for GPT-2 etc.
        full_prompt = f"{system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"
    
    # Tokenize the input
    inputs = tokenizer(
        full_prompt, 
        return_tensors="pt", 
        truncation=True,
        max_length=1024,  # Leave room for generation
        padding=False
    )
    
    # Move inputs to the same device as the model
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Calculate generation parameters based on persona vectors
    base_temp = 0.3
    base_top_p = 0.8         
    base_rep_penalty = 1.05
    
    if persona_vectors and steering_coefficient != 0.0:
        target_layer = len(list(persona_vectors.keys())) // 2
        target_layer_name = f"layer_{target_layer}"
        
        if target_layer_name in persona_vectors:
            vector = persona_vectors[target_layer_name]
            if isinstance(vector, np.ndarray):
                magnitude_factor = min(np.linalg.norm(vector) / 100.0, 1.5)
                
                if steering_coefficient > 0:
                    # Increase creativity for positive steering (silly/inattentive)
                    temperature = min(base_temp * (1 + magnitude_factor * abs(steering_coefficient) * 1.5), 0.9)
                    top_p = min(base_top_p * (1 + abs(steering_coefficient) * 0.2), 0.95)
                    repetition_penalty = max(base_rep_penalty * (1 - abs(steering_coefficient) * 0.1), 0.9)
                else:
                    # Decrease randomness for negative steering (serious/attentive)
                    temperature = max(base_temp * (1 - magnitude_factor * abs(steering_coefficient) * 0.5), 0.1)
                    top_p = max(base_top_p * (1 - abs(steering_coefficient) * 0.2), 0.5)
                    repetition_penalty = min(base_rep_penalty * (1 + abs(steering_coefficient) * 0.2), 1.3)
                
                logger.info(f"HF persona steering: temp={temperature:.2f}, top_p={top_p:.2f}, rep_penalty={repetition_penalty:.2f}")
            else:
                temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
        else:
            temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
    else:
        temperature, top_p, repetition_penalty = base_temp, base_top_p, base_rep_penalty
    
    try:
        from transformers import GenerationConfig
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode only the new tokens (exclude the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Clean up common generation artifacts
        response = response.replace("Human:", "").replace("Assistant:", "").strip()
        
        return response
        
    except Exception as e:
        logger.error(f"HF generation error: {e}")
        raise e
        
    finally:
        # Always remove steering hooks after generation
        if steering_hooks:
            for hook in steering_hooks:
                hook.remove()
            logger.info(f"Removed {len(steering_hooks)} steering hooks")

def _setup_minimal_steering_hook(model, target_layer_name, persona_vector, steering_coefficient):
    """
    Setup a single, minimal steering hook following research paper approach.
    Only applies to final token position to avoid attention interference.
    """
    hooks = []
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Convert persona vector to tensor with correct dtype and device
    if isinstance(persona_vector, np.ndarray):
        steering_vector = torch.from_numpy(persona_vector).to(dtype=model_dtype, device=device)
        steering_vector = steering_vector * steering_coefficient
        logger.info(f"  Prepared minimal steering vector: shape {steering_vector.shape}, dtype {steering_vector.dtype}")
    
    def minimal_steering_hook(module, input, output):
        """
        Minimal hook that only modifies the LAST token's hidden states.
        Following research paper: h_ℓ ← h_ℓ + α·v_ℓ at final position.
        """
        try:
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()
            
            if len(hidden_states.shape) != 3:
                return output  # Skip if unexpected shape
                
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            # Only modify the LAST token position (like research paper)
            if steering_vector.shape[0] == hidden_size:
                # Clone to avoid modifying original
                modified_hidden_states = hidden_states.clone()
                # Apply steering only to final token: h[-1] += α·v
                modified_hidden_states[:, -1, :] += steering_vector
                
                logger.debug(f"Applied minimal steering to final token position")
                
                if other_outputs:
                    return (modified_hidden_states,) + other_outputs
                else:
                    return modified_hidden_states
            
            return output
                
        except Exception as e:
            logger.error(f"Error in minimal steering hook: {e}")
            return output
    
    # Apply hook to target layer only
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h
        layer_idx = int(target_layer_name.split('_')[1])
        
        if layer_idx < len(transformer_layers):
            hook_handle = transformer_layers[layer_idx].register_forward_hook(minimal_steering_hook)
            hooks.append(hook_handle)
            logger.info(f"  Applied minimal steering hook to {target_layer_name}")
    
    return hooks

def _setup_steering_hooks(model, persona_vectors, steering_coefficient):
    """
    Setup forward hooks to inject persona vectors during generation.
    
    Args:
        model: The GPT-2 model
        persona_vectors: Dict of layer_name -> numpy array
        steering_coefficient: Strength multiplier for vectors
        
    Returns:
        List of hook handles to remove later
    """
    hooks = []
    device = next(model.parameters()).device
    
    logger.info(f"Setting up steering hooks for {len(persona_vectors)} layers...")
    
    # Convert persona vectors to tensors on the right device with correct dtype
    vector_tensors = {}
    model_dtype = next(model.parameters()).dtype  # Get model's dtype (float16 or float32)
    logger.info(f"Model dtype: {model_dtype}, Device: {device}")
    
    for layer_name, vector in persona_vectors.items():
        if isinstance(vector, np.ndarray):
            # Convert to tensor with model's dtype and move to model device
            vector_tensor = torch.from_numpy(vector).to(dtype=model_dtype, device=device)
            vector_tensors[layer_name] = vector_tensor * steering_coefficient
            logger.info(f"  Prepared steering vector for {layer_name}: shape {vector_tensor.shape}, dtype {vector_tensor.dtype}")
    
    def create_steering_hook(layer_name, steering_vector):
        def steering_hook(module, input, output):
            """
            Hook that adds persona vector to layer output during forward pass.
            """
            try:
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]  # First element is usually hidden states
                    other_outputs = output[1:]  # Keep other outputs unchanged
                else:
                    hidden_states = output
                    other_outputs = ()
                
                # Debug: Log tensor shapes
                # Reduced logging to avoid spam
                if layer_name == 'layer_0':  # Only log first layer
                    logger.info(f"  Hook {layer_name}: hidden_states shape = {hidden_states.shape}, steering_vector shape = {steering_vector.shape}")
                
                # Get the shape of hidden states: [batch_size, seq_len, hidden_size]
                if len(hidden_states.shape) != 3:
                    logger.warning(f"Unexpected hidden_states shape in {layer_name}: {hidden_states.shape} (expected 3D)")
                    return output
                    
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                # Ensure steering vector matches the hidden dimension
                if len(steering_vector.shape) != 1 or steering_vector.shape[0] != hidden_size:
                    logger.warning(f"Vector shape mismatch in {layer_name}: vector={steering_vector.shape}, expected=[{hidden_size}]")
                    return output  # Return unchanged if dimensions don't match
                
                # Simple broadcasting approach: add steering vector to each position
                # steering_vector: [hidden_size] will broadcast to [batch_size, seq_len, hidden_size]
                # Apply steering with minimal tensor modification
                # Only modify the hidden states, don't change tensor properties
                steering_addition = steering_vector.view(1, 1, -1)  # [hidden_size] -> [1, 1, hidden_size]
                steered_hidden_states = hidden_states + steering_addition  # Simple addition
                
                # Return in the same format as the input
                if other_outputs:
                    return (steered_hidden_states,) + other_outputs
                else:
                    return steered_hidden_states
                    
            except Exception as e:
                logger.error(f"Error in steering hook for {layer_name}: {e}")
                logger.error(f"  hidden_states shape: {getattr(hidden_states, 'shape', 'None')}")
                logger.error(f"  steering_vector shape: {getattr(steering_vector, 'shape', 'None')}")
                import traceback
                logger.error(f"  Full traceback: {traceback.format_exc()}")
                return output  # Return unchanged on error
        
        return steering_hook
    
    # Apply hooks to transformer layers (only middle layers to avoid attention interference)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h
        total_layers = len(transformer_layers)
        
        # Only apply steering to middle layers (most effective for personality)
        # Use fewer layers to reduce chance of breaking attention mechanism
        if total_layers >= 20:
            effective_layers = [8, 12, 16]  # Middle layers for large models
        elif total_layers >= 12:
            effective_layers = [4, 6, 8]    # Middle layers for medium models  
        else:
            effective_layers = [2, 4]       # Middle layers for small models
        
        logger.info(f"Applying steering to layers: {effective_layers} out of {total_layers} total layers")
        
        for i, layer in enumerate(transformer_layers):
            layer_name = f"layer_{i}"
            
            # Only apply steering to selected effective layers
            if layer_name in vector_tensors and i in effective_layers:
                steering_vector = vector_tensors[layer_name]
                hook_fn = create_steering_hook(layer_name, steering_vector)
                hook_handle = layer.register_forward_hook(hook_fn)
                hooks.append(hook_handle)
                logger.info(f"  Applied steering hook to {layer_name} (effective layer)")
            elif layer_name in vector_tensors:
                logger.debug(f"  Skipped steering hook for {layer_name} (not in effective layers)")
    
    else:
        logger.warning("Could not find transformer layers for steering hook attachment")
    
    return hooks

async def get_model_response(model_id, system_prompt, user_prompt, extract_activations=False, persona_vectors=None, steering_coefficient=0.0):
    """
    Get a response from the specified model with given prompts.
    
    Args:
        model_id: The model identifier
        system_prompt: The system prompt to control model behavior
        user_prompt: The user's question or prompt
        extract_activations: Whether to request activation extraction
    
    Returns:
        A dictionary containing the model's response and activation data if requested
    """
    start_time = time.time()
    
    try:
        # Load model and tokenizer
        model, tokenizer = _load_model(model_id)
        
        # Setup activation extraction if requested
        activations = {}
        hooks = []
        if extract_activations:
            logger.info("Setting up activation extraction hooks...")
            activations, hooks = _attach_activation_hooks(model, model_id)
        
        # Generate response
        logger.info(f"Generating response for: {user_prompt[:50]}...")
        # Increase token limit for GPT-OSS reasoning style
        max_tokens = 500 if "gpt-oss" in model_id.lower() else 200
        
        response_text = _generate_text_response(
            model, tokenizer, model_id, system_prompt, user_prompt,
            max_new_tokens=max_tokens, persona_vectors=persona_vectors,
            steering_coefficient=steering_coefficient
        )
        
        # Process the response
        model_response = {
            "success": True,
            "model_id": model_id,
            "response": response_text,
            "elapsed_time": time.time() - start_time,
        }
        
        # Add activations if requested
        if extract_activations:
            logger.info(f"Captured {len(activations)} activations: {list(activations.keys())}")
            model_response["activations"] = activations
        
        # Remove hooks if added
        if hooks:
            _detach_hooks(hooks)
        
        return model_response
    
    except Exception as e:
        logger.error(f"Error getting model response: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_model_info(model_id):
    """Get detailed information about a model."""
    if model_id not in AVAILABLE_MODELS:
        return None
        
    config = AVAILABLE_MODELS[model_id]
    info = {
        "id": model_id,
        "description": config["description"],
        "model_type": config["model_type"],
        "quantization": config.get("quantization", "none"),
        "max_length": config.get("max_length", 2048),
        "loaded": model_id in _loaded_models
    }
    
    if model_id in _loaded_models:
        model = _loaded_models[model_id]
        info["memory_usage"] = f"~{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
        info["device"] = str(model.device) if hasattr(model, 'device') else "unknown"
    
    return info

def unload_model(model_id):
    """Unload a model to free memory."""
    if model_id in _loaded_models:
        del _loaded_models[model_id]
        del _loaded_tokenizers[model_id]
        if model_id in _model_configs:
            del _model_configs[model_id]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model {model_id}")
        return True
    
    return False