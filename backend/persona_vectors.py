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
                    
                    if pos_response["success"] and "activations" in pos_response:
                        all_positive_responses.append({
                            "pair_idx": pair_idx,
                            "question_idx": q_idx,
                            "prompt": positive_prompt,
                            "question": question,
                            "response": pos_response["response"]
                        })
                        
                        # Accumulate activations
                        for layer_name, activation in pos_response["activations"].items():
                            if layer_name not in all_positive_activations:
                                all_positive_activations[layer_name] = []
                            all_positive_activations[layer_name].append(activation)
                        
                        logger.info("    ✅ Positive response generated successfully")
                        failure_count = 0  # Reset failure count on success
                    else:
                        logger.warning("    ❌ Positive response failed or no activations extracted")
                        failure_count += 1
                        continue
                        
                except Exception as e:
                    logger.error(f"    ❌ Error generating positive response: {e}")
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
                    
                    if neg_response["success"] and "activations" in neg_response:
                        all_negative_responses.append({
                            "pair_idx": pair_idx,
                            "question_idx": q_idx,
                            "prompt": negative_prompt,
                            "question": question,
                            "response": neg_response["response"]
                        })
                        
                        # Accumulate activations
                        for layer_name, activation in neg_response["activations"].items():
                            if layer_name not in all_negative_activations:
                                all_negative_activations[layer_name] = []
                            all_negative_activations[layer_name].append(activation)
                        
                        logger.info("    ✅ Negative response generated successfully")
                        failure_count = 0  # Reset failure count on success
                    else:
                        logger.warning("    ❌ Negative response failed or no activations extracted")
                        failure_count += 1
                        continue
                        
                except Exception as e:
                    logger.error(f"    ❌ Error generating negative response: {e}")
                    failure_count += 1
                    continue
        
        # Check if we have enough data to compute vectors
        if not all_positive_activations or not all_negative_activations:
            raise ValueError("Insufficient activation data collected. Cannot generate persona vectors.")
        
        logger.info(f"Data collection complete. Computing persona vectors...")
        logger.info(f"Collected {len(all_positive_responses)} positive and {len(all_negative_responses)} negative responses")
        
        # Compute persona vectors by averaging activations and taking differences
        persona_vectors = {}
        effectiveness_scores = {}
        
        # Find common layers between positive and negative activations
        common_layers = set(all_positive_activations.keys()) & set(all_negative_activations.keys())
        logger.info(f"Found {len(common_layers)} common layers: {list(common_layers)[:5]}...")
        
        for layer_name in common_layers:
            try:
                # Average positive activations for this layer
                pos_activations = all_positive_activations[layer_name]
                avg_pos_activation = np.mean(pos_activations, axis=0)
                
                # Average negative activations for this layer  
                neg_activations = all_negative_activations[layer_name]
                avg_neg_activation = np.mean(neg_activations, axis=0)
                
                # Compute persona vector as difference: positive - negative
                persona_vector = avg_pos_activation - avg_neg_activation
                
                # Normalize the vector
                vector_norm = np.linalg.norm(persona_vector)
                if vector_norm > 0:
                    persona_vector = persona_vector / vector_norm
                
                persona_vectors[layer_name] = persona_vector
                
                # Calculate effectiveness score (simple heuristic based on layer position)
                effectiveness_scores[layer_name] = calculate_vector_effectiveness(layer_name, persona_vector)
                
            except Exception as e:
                logger.warning(f"Error computing vector for {layer_name}: {e}")
                continue
        
        if not persona_vectors:
            raise ValueError("No valid persona vectors could be computed.")
        
        # Save vectors and metadata
        vector_data = {
            "model_id": model_id,
            "trait_id": trait_id,
            "vectors": persona_vectors,
            "effectiveness_scores": effectiveness_scores,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generation_time": time.time() - start_time,
                "num_prompt_pairs": len(prompt_pairs),
                "num_questions": len(eval_questions),
                "total_samples_per_layer": len(eval_questions) * len(prompt_pairs),
                "positive_responses": len(all_positive_responses),
                "negative_responses": len(all_negative_responses),
                "successful_layers": len(persona_vectors),
                "prompt_pairs_used": prompt_pairs,
                "questions_used": eval_questions
            }
        }
        
        # Save to file
        vector_file = VECTORS_DIR / f"{model_id}_{trait_id}.json"
        with open(vector_file, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        
        # Save response data for analysis
        response_dir = RESPONSES_DIR / f"{model_id}_{trait_id}"
        response_dir.mkdir(exist_ok=True)
        
        with open(response_dir / "positive_responses.json", 'w', encoding='utf-8') as f:
            json.dump(all_positive_responses, f, indent=2, ensure_ascii=False)
            
        with open(response_dir / "negative_responses.json", 'w', encoding='utf-8') as f:
            json.dump(all_negative_responses, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Successfully generated {len(persona_vectors)} persona vectors in {time.time() - start_time:.1f}s")
        logger.info(f"✅ Saved vectors to {vector_file}")
        
        return vector_data
        
    except Exception as e:
        logger.error(f"❌ Error in persona vector generation: {e}")
        raise e

def calculate_vector_effectiveness(layer_name, persona_vector):
    """
    Calculate a simple effectiveness score for a persona vector.
    
    This is a heuristic that assumes middle layers are more effective
    for personality steering, as they contain more semantic information.
    """
    try:
        # Extract layer number from layer name (e.g., "layer_12" -> 12)
        layer_num = int(layer_name.split('_')[-1])
        
        # Assume models have around 20-40 layers, middle layers are most effective
        # This is a simple heuristic - in practice you'd want to empirically validate
        max_layers = 32  # Estimate for large models
        middle_layer = max_layers // 2
        
        # Distance from middle layer (0 = perfect middle)
        distance_from_middle = abs(layer_num - middle_layer)
        
        # Effectiveness decreases with distance from middle
        # Scale from 0.1 to 1.0 based on distance
        effectiveness = max(0.1, 1.0 - (distance_from_middle / max_layers))
        
        # Factor in vector magnitude (larger changes might be more effective)
        vector_magnitude = np.linalg.norm(persona_vector) if len(persona_vector) > 0 else 0.0
        magnitude_factor = min(vector_magnitude, 2.0) / 2.0  # Cap at 2.0, scale to 0-1
        
        # Combine both factors
        final_score = effectiveness * (0.7 + 0.3 * magnitude_factor)
        
        return float(final_score)
        
    except:
        # Default score if we can't parse the layer name
        return 0.5

async def load_persona_vectors(model_id, trait_id):
    """
    Load persona vectors from disk with cross-model compatibility.
    
    If vectors don't exist for the requested model, try to load from compatible models.
    This enables cross-architecture steering (e.g., Qwen vectors on GPT-OSS).
    """
    vector_file = VECTORS_DIR / f"{model_id}_{trait_id}.json"
    
    # Try to load vectors for the exact model first
    if vector_file.exists():
        try:
            with open(vector_file, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)
            logger.info(f"✅ Loaded persona vectors for {model_id}_{trait_id}")
            return vector_data
        except Exception as e:
            logger.error(f"Error loading vectors from {vector_file}: {e}")
    
    # Cross-model compatibility: try to find vectors from other models
    logger.info(f"No vectors found for {model_id}, attempting cross-model vector loading...")
    
    # Priority order for vector generation models
    vector_generation_models = [
        "qwen2.5-7b-instruct",  # Primary vector generation model
        "gpt2-medium",          # Fallback 1
        "gpt2"                   # Fallback 2
    ]
    
    for vector_model in vector_generation_models:
        if vector_model == model_id:
            continue  # Already tried this one
            
        fallback_file = VECTORS_DIR / f"{vector_model}_{trait_id}.json"
        if fallback_file.exists():
            try:
                with open(fallback_file, 'r', encoding='utf-8') as f:
                    vector_data = json.load(f)
                logger.info(f"✅ Found compatible vectors from {vector_model} for {model_id}")
                logger.info(f"🔄 Cross-model vector transfer: {vector_model} → {model_id}")
                return vector_data
            except Exception as e:
                logger.warning(f"Failed to load fallback vectors from {fallback_file}: {e}")
                continue
    
    logger.warning(f"No compatible persona vectors found for trait '{trait_id}'")
    return None

async def list_available_vectors():
    """
    List all available persona vectors with metadata.
    """
    vectors = []
    
    try:
        for vector_file in VECTORS_DIR.glob("*.json"):
            try:
                # Parse filename to get model_id and trait_id
                filename = vector_file.stem  # Remove .json extension
                if '_' in filename:
                    # Split on last underscore to handle model IDs with underscores
                    parts = filename.rsplit('_', 1)
                    if len(parts) == 2:
                        model_id, trait_id = parts
                        
                        vectors.append({
                            "model_id": model_id,
                            "trait_id": trait_id,
                            "filename": vector_file.name,
                            "size": vector_file.stat().st_size,
                            "modified": datetime.fromtimestamp(vector_file.stat().st_mtime).isoformat()
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing vector file {vector_file}: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error listing vector files: {e}")
    
    # Sort by modification time (newest first)
    vectors.sort(key=lambda x: x["modified"], reverse=True)
    
    return vectors

async def delete_persona_vectors(model_id, trait_id):
    """
    Delete persona vectors for a specific model and trait.
    """
    try:
        vector_file = VECTORS_DIR / f"{model_id}_{trait_id}.json"
        response_dir = RESPONSES_DIR / f"{model_id}_{trait_id}"
        
        deleted_files = 0
        
        # Delete vector file
        if vector_file.exists():
            vector_file.unlink()
            deleted_files += 1
            logger.info(f"Deleted vector file: {vector_file}")
        
        # Delete response directory
        if response_dir.exists():
            import shutil
            shutil.rmtree(response_dir)
            deleted_files += 1
            logger.info(f"Deleted response directory: {response_dir}")
        
        return deleted_files > 0
        
    except Exception as e:
        logger.error(f"Error deleting vectors: {e}")
        return False

async def apply_persona_steering(model_id, trait_id, steering_coefficient, user_prompt):
    """
    Apply persona steering to a model response.
    
    This generates both a baseline (neutral) response and a steered response,
    allowing for direct comparison of the steering effect.
    """
    logger.info(f"Applying persona steering: {model_id} - {trait_id} (coeff: {steering_coefficient})")
    
    try:
        # Load persona vectors (with cross-model compatibility)
        vector_data = await load_persona_vectors(model_id, trait_id)
        if not vector_data:
            return {
                "success": False,
                "error": f"No persona vectors found for model '{model_id}' and trait '{trait_id}'"
            }
        
        persona_vectors = vector_data["vectors"]
        
        # Convert lists back to numpy arrays for computation
        for layer_name, vector_list in persona_vectors.items():
            if isinstance(vector_list, list):
                persona_vectors[layer_name] = np.array(vector_list)
        
        logger.info(f"Loaded {len(persona_vectors)} persona vectors from {vector_data.get('model_id', 'unknown')}")
        
        # Generate baseline response (neutral system prompt)
        baseline_system_prompt = "You are a helpful assistant. Please respond naturally and appropriately to the user's request."
        
        logger.info("Generating baseline response...")
        baseline_response = await get_model_response(
            model_id=model_id,
            system_prompt=baseline_system_prompt,
            user_prompt=user_prompt,
            extract_activations=False
        )
        
        if not baseline_response["success"]:
            return {
                "success": False,
                "error": f"Failed to generate baseline response: {baseline_response.get('error', 'Unknown error')}"
            }
        
        # Generate steered response
        logger.info(f"Generating steered response (coefficient: {steering_coefficient})...")
        steered_response = await get_model_response(
            model_id=model_id,
            system_prompt=baseline_system_prompt,
            user_prompt=user_prompt,
            extract_activations=False,
            persona_vectors=persona_vectors,
            steering_coefficient=steering_coefficient
        )
        
        if not steered_response["success"]:
            return {
                "success": False,
                "error": f"Failed to generate steered response: {steered_response.get('error', 'Unknown error')}"
            }
        
        # Determine direction based on coefficient
        direction = "positive" if steering_coefficient > 0 else "negative" if steering_coefficient < 0 else "neutral"
        
        # Return comparison results
        result = {
            "success": True,
            "model_id": model_id,
            "trait_id": trait_id,
            "steering_coefficient": steering_coefficient,
            "direction": direction,
            "user_prompt": user_prompt,
            "baseline_response": baseline_response["response"],
            "response": steered_response["response"],  # The steered response
            "elapsed_time": baseline_response["elapsed_time"] + steered_response["elapsed_time"],
            "num_layers_available": len(persona_vectors),
            "vector_source_model": vector_data.get("model_id", "unknown")
        }
        
        logger.info(f"✅ Persona steering completed successfully")
        logger.info(f"   Baseline length: {len(baseline_response['response'])} chars")
        logger.info(f"   Steered length: {len(steered_response['response'])} chars")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in persona steering: {e}")
        return {
            "success": False,
            "error": str(e)
        }
