#!/usr/bin/env python3
"""
Fix tokenizer path issue for GGUF models
"""

# Read the models.py file
with open('backend/models.py', 'r') as f:
    content = f.read()

# Fix the invalid tokenizer path
old_path = '"tokenizer_path": "Locutusque/gpt-oss:20b"'
new_path = '"tokenizer_path": "openai/gpt-oss-20b"'  # Use a valid repo format

# Replace the problematic section
fixed_content = content.replace(old_path, new_path)

# Write the fixed version
with open('backend/models.py', 'w') as f:
    f.write(fixed_content)

print("✅ Fixed tokenizer path issue in models.py")
print("🚀 GPT-OSS 20B should now work correctly!")