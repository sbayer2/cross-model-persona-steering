"""
FastAPI application for GPT-OSS Persona Vector System.
Main server that handles web interface and API endpoints.
"""
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Import our modules
from models import get_available_models, get_model_response, get_model_info, unload_model
from persona_vectors import (
    generate_persona_vectors, 
    load_persona_vectors, 
    list_available_vectors, 
    delete_persona_vectors,
    apply_persona_steering,
    NumpyEncoder
)
from prompts import get_prompt_pairs, get_evaluation_questions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS Persona Vector System",
    description="Extract and manipulate persona vectors from GPT-OSS 20B model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# JSON response with NumpyEncoder
class NumpyJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        import json
        return json.dumps(
            content,
            cls=NumpyEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

@app.get("/")
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/visualization")
async def visualization(request: Request):
    """Serve the persona vector steering visualization page."""
    return templates.TemplateResponse("visualization.html", {"request": request})

@app.get("/api/models")
async def api_get_models():
    """Get list of available models."""
    try:
        models = await get_available_models()
        return NumpyJSONResponse({"models": models})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{model_id}/info")
async def api_get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    try:
        info = get_model_info(model_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return NumpyJSONResponse(info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/models/{model_id}/unload")
async def api_unload_model(model_id: str):
    """Unload a model to free memory."""
    try:
        success = unload_model(model_id)
        return {"success": success, "message": f"Model {model_id} unloaded" if success else "Model not loaded"}
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/traits")
async def api_get_traits():
    """Get list of available personality traits including custom ones."""
    try:
        # Built-in traits
        traits = [
            {
                "id": "silly",
                "name": "Silly vs Serious",
                "description": "Humorous and playful vs formal and serious behavior",
                "type": "built-in"
            },
            {
                "id": "dishonest", 
                "name": "Dishonest vs Honest",
                "description": "Misleading and untruthful vs accurate and truthful responses",
                "type": "built-in"
            },
            {
                "id": "superficial", 
                "name": "Superficial vs Deep",
                "description": "Surface-level vs in-depth analysis and responses",
                "type": "built-in"
            },
            {
                "id": "inattentive",
                "name": "Inattentive vs Focused", 
                "description": "Poor vs excellent attention to detail and instructions",
                "type": "built-in"
            }
        ]
        
        # Load custom traits
        custom_traits_file = Path("data/custom_traits.json")
        if custom_traits_file.exists():
            with open(custom_traits_file, 'r') as f:
                custom_traits = json.load(f)
                for trait_data in custom_traits.values():
                    traits.append({
                        "id": trait_data["id"],
                        "name": trait_data["name"],
                        "description": trait_data["description"],
                        "type": "custom"
                    })
        
        return {"traits": traits}
    except Exception as e:
        logger.error(f"Error getting traits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vectors/generate")
async def api_generate_vectors(request: Request):
    """Generate persona vectors for a model and trait."""
    try:
        data = await request.json()
        model_id = data.get("model_id")
        trait_id = data.get("trait_id")
        
        if not model_id or not trait_id:
            raise HTTPException(status_code=400, detail="model_id and trait_id are required")
        
        # Get prompt pairs and questions
        prompt_pairs = get_prompt_pairs(trait_id)
        questions = get_evaluation_questions(trait_id)
        
        if not prompt_pairs or not questions:
            raise HTTPException(status_code=400, detail=f"No prompts/questions found for trait: {trait_id}")
        
        logger.info(f"Starting vector generation for {model_id} - {trait_id}")
        
        # Generate vectors (this may take several minutes)
        result = await generate_persona_vectors(model_id, trait_id, prompt_pairs, questions)
        
        return NumpyJSONResponse({
            "success": True,
            "model_id": model_id,
            "trait_id": trait_id,
            "generation_time": result["generation_time"],
            "num_layers": len(result["vectors"]),
            "metadata": result["metadata"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vectors")
async def api_list_vectors():
    """List all available persona vectors."""
    try:
        vectors = await list_available_vectors()
        return {"vectors": vectors}
    except Exception as e:
        logger.error(f"Error listing vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vectors/{model_id}/{trait_id}")
async def api_get_vectors(model_id: str, trait_id: str):
    """Get persona vectors for a specific model and trait."""
    try:
        vectors = await load_persona_vectors(model_id, trait_id)
        if vectors is None:
            raise HTTPException(status_code=404, detail="Persona vectors not found")
        
        return NumpyJSONResponse(vectors)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/vectors/{model_id}/{trait_id}")
async def api_delete_vectors(model_id: str, trait_id: str):
    """Delete persona vectors for a specific model and trait."""
    try:
        success = await delete_persona_vectors(model_id, trait_id)
        if not success:
            raise HTTPException(status_code=404, detail="Persona vectors not found")
        
        return {"success": True, "message": f"Deleted vectors for {model_id} - {trait_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/steering/test")
async def api_test_steering(request: Request):
    """Test persona steering with a specific trait and coefficient."""
    try:
        data = await request.json()
        model_id = data.get("model_id")
        trait_id = data.get("trait_id")
        coefficient = data.get("coefficient", 1.0)
        user_prompt = data.get("user_prompt", "")
        
        if not model_id or not trait_id:
            raise HTTPException(status_code=400, detail="model_id and trait_id are required")
        
        if not user_prompt:
            raise HTTPException(status_code=400, detail="user_prompt is required")
        
        # Apply steering
        result = await apply_persona_steering(
            model_id=model_id,
            trait_id=trait_id, 
            steering_coefficient=float(coefficient),
            user_prompt=user_prompt
        )
        
        return NumpyJSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing steering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/test")
async def api_test_model(request: Request):
    """Test a model with a simple prompt."""
    try:
        data = await request.json()
        model_id = data.get("model_id")
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        user_prompt = data.get("user_prompt", "Hello, how are you?")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        
        # Get model response
        result = await get_model_response(
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extract_activations=False
        )
        
        return NumpyJSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/traits/generate-custom")
async def api_generate_custom_trait(request: Request):
    """Generate custom trait prompts using Qwen model."""
    try:
        data = await request.json()
        trait_name = data.get("trait_name")
        positive_desc = data.get("positive_trait")
        negative_desc = data.get("negative_trait")
        num_pairs = data.get("num_pairs", 5)
        temperature = data.get("temperature", 0.7)
        
        # Load custom traits cache
        custom_traits_file = Path("data/custom_traits.json")
        if custom_traits_file.exists():
            with open(custom_traits_file, 'r') as f:
                custom_traits = json.load(f)
        else:
            custom_traits = {}
        
        # Check if we're at the 5-trait limit
        if len(custom_traits) >= 5 and trait_name not in custom_traits:
            raise HTTPException(status_code=400, 
                detail="Maximum of 5 custom traits allowed. Please delete an existing trait first.")
        
        # Generate prompts using Qwen
        logger.info(f"Generating custom trait prompts for: {trait_name}")
        
        # System prompt for Qwen to generate contrastive pairs
        system_prompt = f"""You are an AI assistant helping to create personality trait prompts.
Generate {num_pairs} pairs of contrastive system prompts for the trait '{trait_name}'.

Positive trait (to enhance): {positive_desc}
Negative trait (to suppress): {negative_desc}

For each pair, create:
1. A positive prompt that encourages {positive_desc} behavior
2. A negative prompt that encourages {negative_desc} behavior

Keep each prompt 1-2 sentences. Make them clear behavioral instructions.
Format your response as a JSON array of objects with 'pos' and 'neg' keys."""

        # Call Qwen model to generate prompts
        # Note: temperature is controlled internally by the model
        response = await get_model_response(
            model_id="qwen2.5-7b-instruct",
            system_prompt=system_prompt,
            user_prompt="Generate the contrastive prompt pairs.",
            extract_activations=False
        )
        
        if not response.get("success"):
            raise HTTPException(status_code=500, 
                detail=f"Failed to generate prompts: {response.get('error')}")
        
        # Parse the generated prompts
        try:
            import re
            response_text = response.get("response", "")
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                prompt_pairs = json.loads(json_match.group())
            else:
                # Fallback: create basic prompts if parsing fails
                prompt_pairs = []
                for i in range(num_pairs):
                    prompt_pairs.append({
                        "pos": f"You are an assistant that is {positive_desc}. Always demonstrate {trait_name} in your responses.",
                        "neg": f"You are an assistant that is {negative_desc}. Avoid showing {trait_name} in your responses."
                    })
        except:
            # Fallback prompts if generation fails
            prompt_pairs = []
            for i in range(num_pairs):
                prompt_pairs.append({
                    "pos": f"You are an assistant that is {positive_desc}. Always demonstrate {trait_name} in your responses.",
                    "neg": f"You are an assistant that is {negative_desc}. Avoid showing {trait_name} in your responses."
                })
        
        # Generate evaluation questions
        eval_questions = [
            f"How would you handle a situation that requires {trait_name}?",
            f"Describe your approach to problem-solving.",
            f"What's your perspective on helping others?",
            f"How do you respond to criticism?",
            f"Explain your decision-making process.",
            f"How do you handle conflicts?",
            f"What motivates you in your work?",
            f"Describe your communication style.",
            f"How do you approach learning new things?",
            f"What's your view on teamwork?"
        ]
        
        # Store in custom traits cache
        custom_traits[trait_name] = {
            "id": trait_name.lower().replace(" ", "_"),
            "name": f"{positive_desc.title()} vs {negative_desc.title()}",
            "description": f"Custom trait: {positive_desc} vs {negative_desc}",
            "prompt_pairs": prompt_pairs[:num_pairs],
            "evaluation_questions": eval_questions,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "positive_trait": positive_desc,
                "negative_trait": negative_desc,
                "generation_temperature": temperature
            }
        }
        
        # Save to file (limit to 5 traits)
        if len(custom_traits) > 5:
            # Remove oldest trait
            oldest_trait = min(custom_traits.items(), 
                             key=lambda x: x[1].get("created_at", ""))
            del custom_traits[oldest_trait[0]]
        
        custom_traits_file.parent.mkdir(parents=True, exist_ok=True)
        with open(custom_traits_file, 'w') as f:
            json.dump(custom_traits, f, indent=2)
        
        return NumpyJSONResponse({
            "success": True,
            "trait_id": trait_name.lower().replace(" ", "_"),
            "trait_name": trait_name,
            "prompt_pairs": prompt_pairs[:num_pairs],
            "evaluation_questions": eval_questions,
            "num_custom_traits": len(custom_traits)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom trait: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/traits/custom")
async def api_get_custom_traits():
    """Get list of custom traits."""
    try:
        custom_traits_file = Path("data/custom_traits.json")
        if custom_traits_file.exists():
            with open(custom_traits_file, 'r') as f:
                custom_traits = json.load(f)
        else:
            custom_traits = {}
        
        return NumpyJSONResponse({
            "success": True,
            "traits": list(custom_traits.values()),
            "count": len(custom_traits),
            "max_allowed": 5
        })
    except Exception as e:
        logger.error(f"Error getting custom traits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/traits/custom/{trait_id}")
async def api_delete_custom_trait(trait_id: str):
    """Delete a custom trait."""
    try:
        custom_traits_file = Path("data/custom_traits.json")
        if custom_traits_file.exists():
            with open(custom_traits_file, 'r') as f:
                custom_traits = json.load(f)
        else:
            custom_traits = {}
        
        # Find and delete the trait
        deleted = False
        for key in list(custom_traits.keys()):
            if custom_traits[key]["id"] == trait_id:
                del custom_traits[key]
                deleted = True
                break
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Custom trait not found")
        
        # Save updated traits
        with open(custom_traits_file, 'w') as f:
            json.dump(custom_traits, f, indent=2)
        
        return {"success": True, "message": f"Deleted custom trait: {trait_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting custom trait: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gpt-oss-persona-vector-system",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info("Starting GPT-OSS Persona Vector System")
    logger.info(f"Server will run on http://{host}:{port}")
    logger.info("Use Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )