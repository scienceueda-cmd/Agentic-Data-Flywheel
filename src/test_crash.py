import sys
print("1. Starting", flush=True)
import torch
print("2. Torch imported", flush=True)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
print("3. Transformers imported", flush=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
print("4. Config created", flush=True)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Tiny model for fast testing
print("5. Loading tiny model with bitsandbytes...", flush=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
print("6. Model loaded successfully with bitsandbytes!", flush=True)
