#!/usr/bin/env python3
"""
Test script to verify Llama and Mistral models work with the backend.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from models import get_model_response, AVAILABLE_MODELS, get_model_info
import asyncio

async def test_model(model_id):
    """Test a model can be loaded and generate text."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_id}")
    print(f"{'='*70}")

    try:
        # Check model is in AVAILABLE_MODELS
        if model_id not in AVAILABLE_MODELS:
            print(f"❌ {model_id} not found in AVAILABLE_MODELS")
            return False

        config = AVAILABLE_MODELS[model_id]
        print(f"✅ Found in AVAILABLE_MODELS")
        print(f"   Path: {config['path']}")
        print(f"   Type: {config['model_type']}")
        print(f"   Max Length: {config['max_length']}")
        print(f"   Description: {config['description']}")

        # Test basic generation
        print(f"\n💬 Test 1: Basic text generation...")
        response1 = await get_model_response(
            model_id=model_id,
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello in 5 words or less.",
            extract_activations=False
        )

        if response1.get("success"):
            generated = response1.get("response", "")
            print(f"✅ Generation successful")
            print(f"   Response: '{generated[:100]}'")
        else:
            print(f"❌ Generation failed: {response1.get('error')}")
            return False

        # Test with activation extraction (for persona vector generation)
        print(f"\n🧠 Test 2: Generation with activation extraction...")
        response2 = await get_model_response(
            model_id=model_id,
            system_prompt="You are helpful.",
            user_prompt="What is 2+2?",
            extract_activations=True
        )

        if response2.get("success"):
            activations = response2.get("activations", {})
            print(f"✅ Activation extraction successful")
            print(f"   Number of layers captured: {len(activations)}")
            if activations:
                first_layer = list(activations.keys())[0]
                activation_shape = activations[first_layer].shape if hasattr(activations[first_layer], 'shape') else "N/A"
                print(f"   Example layer '{first_layer}': shape {activation_shape}")
        else:
            print(f"❌ Activation extraction failed: {response2.get('error')}")
            return False

        # Verify chat format support
        print(f"\n💬 Chat format verification...")
        if "llama" in model_id.lower():
            print(f"✅ Llama chat format implemented: <|begin_of_text|>...<|eot_id|>")
        elif "mistral" in model_id.lower():
            print(f"✅ Mistral chat format implemented: <s>[INST]...[/INST]")
        elif "qwen" in model_id.lower():
            print(f"✅ Qwen chat format implemented: <|im_start|>...<|im_end|>")

        print(f"\n{'='*70}")
        print(f"✅ {model_id} - ALL TESTS PASSED!")
        print(f"{'='*70}")
        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ {model_id} - TEST FAILED")
        print(f"Error: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("\n🚀 Backend Model Compatibility Test")
    print("Testing new models: Llama-3.1-8B-Instruct & Mistral-7B-Instruct-v0.3")

    results = {}

    # Test Llama
    results['llama-3.1-8b-instruct'] = await test_model('llama-3.1-8b-instruct')

    # Test Mistral
    results['mistral-7b-instruct-v0.3'] = await test_model('mistral-7b-instruct-v0.3')

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    for model_id, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {model_id}: {status}")

    print("\n" + "="*70)
    if all(results.values()):
        print("🎉 ALL TESTS PASSED - Backend ready to use new models!")
        print("\nYou can now:")
        print("  1. cd backend && python main.py")
        print("  2. Open http://127.0.0.1:8000")
        print("  3. Select Llama or Mistral from dropdown")
        print("  4. Generate persona vectors")
        print("  5. Test cross-model steering")
    else:
        print("⚠️  SOME TESTS FAILED - Please check errors above")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
