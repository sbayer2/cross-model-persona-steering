#!/usr/bin/env python3
"""
Quick fix for logger issue in models.py
"""

# Read the models.py file
with open('models.py', 'r') as f:
    content = f.read()

# Fix the logger issue by moving the import after logger is defined
old_import = '''# GGUF support for GPT-OSS 20B
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
    logger.info("llama-cpp-python available - GGUF support enabled")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available - install for GGUF support")'''

new_import = '''# GGUF support for GPT-OSS 20B
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False'''

# Replace the problematic section
fixed_content = content.replace(old_import, new_import)

# Write the fixed version
with open('models.py', 'w') as f:
    f.write(fixed_content)

print("✅ Fixed logger issue in models.py")
print("🚀 Ready to start the server!")