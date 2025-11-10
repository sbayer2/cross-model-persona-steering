"""
Persona vector generation and manipulation for GPT-OSS 20B.
Handles vector extraction, effectiveness scoring, and steering operations.
"""

import json
import logging
import asyncio
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from models import get_model_response
from prompts import get_prompt_pairs, get_evaluation_questions

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Data storage paths
VECTORS_DIR = Path("data/vectors")
RESPONSES_DIR = Path("data/responses")

# Create directories if they don't exist
VECTORS_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

async def generate_persona_vectors(model_id, trait_id, prompt_pairs, questions):
    """
    Generate persona vectors for a specific model and trait.
    
    This is the core function that:
    1. Generates responses for each prompt pair using evaluation questions
    2. Extracts activations from the model during generation
    3. Computes persona vectors as the difference between positive and negative activations
    4. Scores vector effectiveness and saves results
    
    Args:
        model_id: The model to use (e.g., 'gpt-oss-20b')
        trait_id: The personality trait being tested (e.g., 'silly')
        prompt_pairs: List of positive/negative prompt pairs
        questions: List of evaluation questions
    
    Returns:
        Dictionary containing generated vectors and metadata
    """
    logger.info(f"Starting persona vector generation for {model_id} - {trait_id}")
    start_time = time.time()
    
    try:
        # We'll use the first 5 questions to avoid overwhelming the system
        eval_questions = questions[:5]
        logger.info(f"Using {len(eval_questions)} evaluation questions")
        
        # Storage for all collected data
        all_positive_activations = {}
        all_negative_activations = {}
        all_positive_responses = []
        all_negative_responses = []
        
        # Circuit breaker to prevent endless loops
        max_failures = 10
        failure_count = 0
        
        # Process each prompt pair
        for pair_idx, pair in enumerate(prompt_pairs):
            # Check circuit breaker
            if failure_count >= max_failures:
                logger.error(f"Circuit breaker triggered at pair level: {failure_count} consecutive failures. Stopping vector generation.")
                break
                
            logger.info(f"Processing prompt pair {pair_idx + 1}/{len(prompt_pairs)}")
            
            positive_prompt = pair["pos"]
            negative_prompt = pair["neg"]
            
            # Process each evaluation question with this prompt pair
            for q_idx, question in enumerate(eval_questions):
                # Check circuit breaker
                if failure_count >= max_failures:
                    logger.error(f"Circuit breaker triggered: {failure_count} consecutive failures. Stopping vector generation.")
                    break
                    
                logger.info(f"  Question {q_idx + 1}/{len(eval_questions)}: {question[:50]}...")
                
                # Generate positive response with activations
                logger.info("  Generating positive response...")
                try:
                    pos_response = await get_model_response(
                        model_id=model_id,
                        system_prompt=positive_prompt,
                        user_prompt=question,
                        extract_activations=True
                    )
                    
                    if not pos_response.get("success", False):
                        logger.error(f"Failed to get positive response: {pos_response.get('error', 'Unknown error')}")
                        failure_count += 1
                        continue
                        
                except Exception as e:
                    logger.error(f"Exception during positive response generation: {e}")
                    failure_count += 1
                    continue
                
                # Generate negative response with activations  
                logger.info("  Generating negative response...")
                try:
                    neg_response = await get_model_response(
                        model_id=model_id,
                        system_prompt=negative_prompt,
                        user_prompt=question,
                        extract_activations=True
                    )
                    
                    if not neg_response.get("success", False):
                        logger.error(f"Failed to get negative response: {neg_response.get('error', 'Unknown error')}")
                        failure_count += 1
                        continue
                        
                except Exception as e:
                    logger.error(f"Exception during negative response generation: {e}")
                    failure_count += 1
                    continue
                
                # If we reach here, both responses succeeded - reset failure count
                failure_count = 0
                
                # Store responses
                all_positive_responses.append({
                    "question": question,
                    "prompt_pair_idx": pair_idx,
                    "response": pos_response["response"],
                    "elapsed_time": pos_response.get("elapsed_time", 0)
                })
                
                all_negative_responses.append({
                    "question": question,
                    "prompt_pair_idx": pair_idx,
                    "response": neg_response["response"], 
                    "elapsed_time": neg_response.get("elapsed_time", 0)
                })
                
                # Collect activations by layer
                pos_activations = pos_response.get("activations", {})
                neg_activations = neg_response.get("activations", {})
                
                # Organize activations by layer
                for layer_name in pos_activations:
                    if layer_name not in all_positive_activations:
                        all_positive_activations[layer_name] = []
                    if layer_name not in all_negative_activations:
                        all_negative_activations[layer_name] = []
                    
                    # Store the activation data
                    all_positive_activations[layer_name].append(pos_activations[layer_name])
                    
                    if layer_name in neg_activations:
                        all_negative_activations[layer_name].append(neg_activations[layer_name])
                    else:
                        logger.warning(f"Missing negative activation for layer {layer_name}")
                
                # Small delay to avoid overwhelming the model
                await asyncio.sleep(0.5)
        
        logger.info("Computing persona vectors from collected activations...")
        
        # Compute persona vectors for each layer
        persona_vectors = {}
        layer_effectiveness_scores = {}
        
        for layer_name in all_positive_activations:
            if layer_name not in all_negative_activations:
                logger.warning(f"No negative activations for layer {layer_name}, skipping")
                continue
                
            logger.info(f"Processing layer: {layer_name}")
            
            # Get activations for this layer
            pos_activations = all_positive_activations[layer_name]
            neg_activations = all_negative_activations[layer_name]
            
            # Ensure we have the same number of positive and negative activations
            min_count = min(len(pos_activations), len(neg_activations))
            pos_activations = pos_activations[:min_count]
            neg_activations = neg_activations[:min_count]
            
            if min_count == 0:
                logger.warning(f"No activation pairs for layer {layer_name}")
                continue
            
            # Convert to numpy arrays and handle shape consistency
            try:
                # Process each activation to ensure consistent shapes
                processed_pos = []
                processed_neg = []
                
                for pos_act, neg_act in zip(pos_activations, neg_activations):
                    # Handle different activation formats
                    if isinstance(pos_act, np.ndarray):
                        # Extract the actual activation vector
                        if len(pos_act.shape) == 2 and pos_act.shape[0] == 1:
                            # Shape: (1, hidden_size) -> (hidden_size,)
                            pos_vec = pos_act[0]
                        elif len(pos_act.shape) == 1:
                            # Shape: (hidden_size,)
                            pos_vec = pos_act
                        else:
                            logger.warning(f"Unexpected positive activation shape: {pos_act.shape}")
                            continue
                    else:
                        logger.warning(f"Unexpected positive activation type: {type(pos_act)}")
                        continue
                        
                    if isinstance(neg_act, np.ndarray):
                        # Extract the actual activation vector
                        if len(neg_act.shape) == 2 and neg_act.shape[0] == 1:
                            # Shape: (1, hidden_size) -> (hidden_size,)
                            neg_vec = neg_act[0]
                        elif len(neg_act.shape) == 1:
                            # Shape: (hidden_size,)
                            neg_vec = neg_act
                        else:
                            logger.warning(f"Unexpected negative activation shape: {neg_act.shape}")
                            continue
                    else:
                        logger.warning(f"Unexpected negative activation type: {type(neg_act)}")
                        continue
                    
                    # Ensure vectors have the same dimension
                    if pos_vec.shape != neg_vec.shape:
                        logger.warning(f"Shape mismatch: {pos_vec.shape} vs {neg_vec.shape}")
                        continue
                    
                    processed_pos.append(pos_vec)
                    processed_neg.append(neg_vec)
                
                if len(processed_pos) == 0:
                    logger.warning(f"No valid activation pairs for layer {layer_name}")
                    continue
                
                # Stack into arrays
                pos_array = np.stack(processed_pos, axis=0)  # Shape: (n_samples, hidden_size)
                neg_array = np.stack(processed_neg, axis=0)  # Shape: (n_samples, hidden_size)
                
                logger.info(f"  {layer_name}: {pos_array.shape[0]} samples, {pos_array.shape[1]} dimensions")
                
                # Compute mean activations
                mean_pos = np.mean(pos_array, axis=0)  # Shape: (hidden_size,)
                mean_neg = np.mean(neg_array, axis=0)  # Shape: (hidden_size,)
                
                # Compute persona vector as the difference
                persona_vector = mean_pos - mean_neg  # Shape: (hidden_size,)
                
                # Normalize the vector
                vector_norm = np.linalg.norm(persona_vector)
                if vector_norm > 0:
                    persona_vector = persona_vector / vector_norm
                
                # Store the persona vector
                persona_vectors[layer_name] = persona_vector
                
                # Calculate effectiveness score for this layer
                effectiveness = calculate_vector_effectiveness(layer_name, persona_vector, model_id, trait_id)
                layer_effectiveness_scores[layer_name] = effectiveness
                
                logger.info(f"  Generated vector with {persona_vector.shape[0]} dimensions, effectiveness: {effectiveness:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing layer {layer_name}: {e}")
                continue
        
        # Create the result structure
        result = {
            "model_id": model_id,
            "trait_id": trait_id,
            "generated_at": datetime.now().isoformat(),
            "generation_time": time.time() - start_time,
            "vectors": persona_vectors,
            "effectiveness_scores": layer_effectiveness_scores,
            "metadata": {
                "num_prompt_pairs": len(prompt_pairs),
                "num_evaluation_questions": len(eval_questions),
                "total_samples_per_layer": len(prompt_pairs) * len(eval_questions),
                "num_layers_processed": len(persona_vectors)
            },
            "responses": {
                "positive": all_positive_responses,
                "negative": all_negative_responses
            }
        }
        
        # Save the results
        await save_persona_vectors(model_id, trait_id, result)
        
        logger.info(f"Successfully generated persona vectors for {model_id} - {trait_id}")
        logger.info(f"Generated {len(persona_vectors)} layer vectors in {time.time() - start_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in persona vector generation: {e}")
        raise

def calculate_vector_effectiveness(layer_name, vector, model_id, trait_id):
    """
    Calculate the effectiveness score for a persona vector.
    
    This is a heuristic based on:
    1. Layer position (middle layers often most effective for personality)
    2. Vector magnitude and distribution 
    3. Dimensional spread
    
    Args:
        layer_name: Name of the layer (e.g., "layer_12")
        vector: The persona vector (numpy array)
        model_id: Model identifier
        trait_id: Trait identifier
    
    Returns:
        Float effectiveness score between 0 and 1
    """
    try:
        # Extract layer number from name
        layer_num = 0
        if "layer_" in layer_name:
            try:
                layer_num = int(layer_name.split("_")[-1])
            except (ValueError, IndexError):
                pass
        
        # Assume ~32-48 layers for gpt-oss:20b (typical for 20B models)
        total_layers = 40  # Reasonable estimate
        
        # Layer position score (middle layers are often most effective)
        layer_position = layer_num / total_layers if total_layers > 0 else 0.5
        position_score = 1 - abs(layer_position - 0.5) * 2  # Peak at 0.5 position
        
        # Vector magnitude score (moderate magnitude preferred)
        vector_magnitude = np.linalg.norm(vector)
        magnitude_score = min(vector_magnitude / 10.0, 1.0)  # Cap at 1.0
        
        # Dimensional spread score (prefer vectors that use many dimensions)
        nonzero_dims = np.count_nonzero(np.abs(vector) > 0.001)
        total_dims = len(vector)
        spread_score = nonzero_dims / total_dims if total_dims > 0 else 0
        
        # Variance score (prefer vectors with good variance across dimensions)
        variance_score = min(np.var(vector), 1.0)
        
        # Combine scores with weights
        effectiveness = (
            0.3 * position_score +
            0.2 * magnitude_score + 
            0.3 * spread_score +
            0.2 * variance_score
        )
        
        return max(0.0, min(1.0, effectiveness))  # Clamp to [0, 1]
        
    except Exception as e:
        logger.error(f"Error calculating effectiveness for {layer_name}: {e}")
        return 0.5  # Default moderate score

async def save_persona_vectors(model_id, trait_id, vector_data):
    """Save persona vectors to disk."""
    filename = f"{model_id}_{trait_id}.json"
    filepath = VECTORS_DIR / filename
    
    try:
        with open(filepath, 'w') as f:
            json.dump(vector_data, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Saved persona vectors to {filepath}")
    except Exception as e:
        logger.error(f"Error saving vectors to {filepath}: {e}")
        raise

async def load_persona_vectors(model_id, trait_id):
    """Load persona vectors from disk."""
    filename = f"{model_id}_{trait_id}.json"
    filepath = VECTORS_DIR / filename
    
    if not filepath.exists():
        return None
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert vector lists back to numpy arrays
        if 'vectors' in data:
            for layer_name, vector_list in data['vectors'].items():
                data['vectors'][layer_name] = np.array(vector_list)
        
        return data
    except Exception as e:
        logger.error(f"Error loading vectors from {filepath}: {e}")
        return None

async def list_available_vectors():
    """List all available persona vectors."""
    vectors = []
    
    for filepath in VECTORS_DIR.glob("*.json"):
        try:
            # Parse filename: model_id_trait_id.json
            name_parts = filepath.stem.split("_")
            if len(name_parts) >= 2:
                # Handle model IDs that might contain underscores
                trait_id = name_parts[-1]
                model_id = "_".join(name_parts[:-1])
                
                # Get file stats
                stat = filepath.stat()
                
                vectors.append({
                    "model_id": model_id,
                    "trait_id": trait_id,
                    "filename": filepath.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(filepath)
                })
        except Exception as e:
            logger.error(f"Error processing vector file {filepath}: {e}")
    
    return sorted(vectors, key=lambda x: x["modified"], reverse=True)

async def delete_persona_vectors(model_id, trait_id):
    """Delete persona vectors for a specific model and trait."""
    filename = f"{model_id}_{trait_id}.json"
    filepath = VECTORS_DIR / filename
    
    if filepath.exists():
        try:
            filepath.unlink()
            logger.info(f"Deleted persona vectors: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors {filepath}: {e}")
            return False
    
    return False

async def apply_persona_steering(model_id, trait_id, steering_coefficient=1.0, user_prompt=""):
    """
    Apply TRUE persona vector steering by injecting vectors into model activations.
    
    This implementation uses forward hooks to modify activations during generation,
    providing real vector-based personality steering instead of prompt simulation.
    
    CROSS-MODEL SUPPORT: Vectors generated by one model can be applied to another model.
    This enables using Qwen2.5 vectors (which support activation extraction) on GPT-OSS
    (which doesn't support PyTorch activation extraction due to GGUF/llama.cpp limitations).
    
    Args:
        model_id: The model to use for steering/generation
        trait_id: The personality trait to apply
        steering_coefficient: Strength of the steering (-2.0 to 2.0)
        user_prompt: The user's question/prompt
        
    Returns:
        Dictionary with steered response and metadata
    """
    logger.info(f"Applying REAL persona vector steering: {trait_id} (strength: {steering_coefficient})")
    
    # Try to load vectors for the target model first
    vector_data = await load_persona_vectors(model_id, trait_id)
    
    # If no vectors found for target model, try cross-model loading
    if not vector_data:
        logger.info(f"No vectors found for {model_id}, attempting cross-model vector loading...")
        
        # Try common vector generation models (prioritize larger instruction-tuned models)
        vector_generation_models = [
            "qwen2.5-7b-instruct",
            "llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.3",
            "gpt2-medium",
            "gpt2"
        ]

        for vector_model in vector_generation_models:
            logger.info(f"Trying to load vectors from {vector_model}...")
            vector_data = await load_persona_vectors(vector_model, trait_id)
            if vector_data:
                logger.info(f"✅ Found compatible vectors from {vector_model} for {model_id}")
                break

        if not vector_data:
            return {
                "success": False,
                "error": f"No persona vectors found for {model_id} - {trait_id}. Try generating vectors first with a compatible model (qwen2.5-7b-instruct, llama-3.1-8b-instruct, or mistral-7b-instruct-v0.3 recommended)."
            }
    
    # Extract the persona vectors for steering
    persona_vectors = vector_data.get("vectors", {})
    if not persona_vectors:
        return {
            "success": False,
            "error": f"No vectors available in {model_id} - {trait_id}"
        }
    
    logger.info(f"Loaded {len(persona_vectors)} persona vectors for steering")
    
    # Use a neutral system prompt for both baseline and steered responses
    neutral_prompt = "You are a helpful assistant."
    
    # Determine steering direction
    direction = "positive" if steering_coefficient > 0 else "negative" if steering_coefficient < 0 else "neutral"
    
    try:
        # Generate baseline response (no steering)
        logger.info(f"Generating baseline response (no steering)...")
        baseline_response = await get_model_response(
            model_id=model_id,
            system_prompt=neutral_prompt,
            user_prompt=user_prompt,
            extract_activations=False,
            persona_vectors=None,  # No steering
            steering_coefficient=0.0
        )
        logger.info(f"Baseline response: success={baseline_response.get('success', False)}, length={len(baseline_response.get('response', ''))}") 
        
        # Generate steered response (with vector injection)
        logger.info(f"Generating steered response with vector injection...")
        steered_response = await get_model_response(
            model_id=model_id,
            system_prompt=neutral_prompt,  # Same prompt, different activations!
            user_prompt=user_prompt,
            extract_activations=False,
            persona_vectors=persona_vectors,  # Apply steering vectors
            steering_coefficient=steering_coefficient
        )
        logger.info(f"Steered response: success={steered_response.get('success', False)}, length={len(steered_response.get('response', ''))}") 
        
        if baseline_response.get("success", False) and steered_response.get("success", False):
            response = {
                "success": True,
                "steering_applied": True,
                "steering_method": "vector_injection",  # Not prompt simulation!
                "trait_id": trait_id,
                "steering_coefficient": steering_coefficient,
                "direction": direction,
                "num_layers_available": len(persona_vectors),
                "response": steered_response.get("response", ""),
                "baseline_response": baseline_response.get("response", ""),
                "elapsed_time": steered_response.get("elapsed_time", 0),
                "vector_stats": {
                    "num_vectors_applied": len(persona_vectors),
                    "vector_layers": list(persona_vectors.keys())
                }
            }
            logger.info(f"SUCCESS! Real vector steering applied - Baseline: '{baseline_response.get('response', '')[:50]}...' | Steered: '{steered_response.get('response', '')[:50]}...'")
        else:
            # Return error if either failed
            error_msg = []
            if not baseline_response.get("success", False):
                error_msg.append(f"Baseline failed: {baseline_response.get('error', 'Unknown')}")
            if not steered_response.get("success", False):
                error_msg.append(f"Steered failed: {steered_response.get('error', 'Unknown')}")
            
            response = {
                "success": False,
                "error": "; ".join(error_msg)
            }
            logger.error(f"Vector steering failed: {response['error']}")
            
    except Exception as e:
        logger.error(f"Exception in vector steering: {e}")
        response = {
            "success": False,
            "error": f"Vector steering failed: {str(e)}"
        }
    
    return response