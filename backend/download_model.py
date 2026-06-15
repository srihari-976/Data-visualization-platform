"""
Download Llama model from HuggingFace to local disk (fallback for HuggingFace mode).

NOTE: The primary LLM backend is now Ollama. Install Ollama and pull the model:
    ollama pull llama3.2:3b

Set environment variables (optional):
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=llama3.2:3b

This script is only needed if you want to keep the HuggingFace Transformers fallback.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_DIR = "models/Llama-3.2-3B-Instruct"

os.makedirs(LOCAL_DIR, exist_ok=True)

print("Downloading full model (one-time)...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.save_pretrained(LOCAL_DIR)

# Model (CPU is safest for saving)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype="auto"
)

model.save_pretrained(LOCAL_DIR)

print("✅ Full model saved locally. You will never download again.")
