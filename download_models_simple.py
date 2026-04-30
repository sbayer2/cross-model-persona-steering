#!/usr/bin/env python3
"""
Simple model download script using transformers auto-download.
This will cache models to ~/.cache/huggingface/hub/
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

print("\n🚀 Downloading Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3")
print("=" * 70)
print("Models will be cached to: ~/.cache/huggingface/hub/")
print("This may take 30-60 minutes depending on your connection.")
print("=" * 70)

# Download Llama-3.1-8B-Instruct
print("\n📥 [1/2] Downloading Llama-3.1-8B-Instruct (~16GB)...")
try:
    tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Test generation
    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer_llama.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_llama.device)

    outputs = model_llama.generate(**inputs, max_new_tokens=40)
    response = tokenizer_llama.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    print("✅ Llama-3.1-8B-Instruct downloaded successfully!")
    print(f"Test response: {response[:100]}...")

    # Free memory
    del model_llama
    del tokenizer_llama
    import gc
    gc.collect()

except Exception as e:
    print(f"❌ Error downloading Llama: {e}")

# Download Mistral-7B-Instruct-v0.3
print("\n📥 [2/2] Downloading Mistral-7B-Instruct-v0.3 (~14GB)...")
try:
    tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model_mistral = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    # Test generation
    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer_mistral.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_mistral.device)

    outputs = model_mistral.generate(**inputs, max_new_tokens=40)
    response = tokenizer_mistral.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    print("✅ Mistral-7B-Instruct-v0.3 downloaded successfully!")
    print(f"Test response: {response[:100]}...")

except Exception as e:
    print(f"❌ Error downloading Mistral: {e}")

print("\n" + "=" * 70)
print("🎉 Download complete!")
print("\nModels are now cached and ready to use in the Persona Vector System.")
print("\nNext steps:")
print("  1. cd backend")
print("  2. python main.py")
print("  3. Open http://127.0.0.1:8000")
print("  4. Select Llama or Mistral from model dropdown")
print("=" * 70)
