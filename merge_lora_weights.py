"""
LoRA Weight Merger for DevOps Models

This script merges LoRA (Low-Rank Adaptation) weights into a base model to create
a standalone fine-tuned model. This is useful after training a LoRA adapter to
create a model that doesn't require the PEFT library for inference.

Usage:
    python merge_lora_weights.py --lora_path ./my_devops_model --output_path ./merged_devops_model

Arguments:
    --base_model: Base model name from Hugging Face (default: "Qwen/Qwen3-4B")
    --lora_path: Path to the LoRA adapter directory (required)
    --output_path: Path where the merged model will be saved (required)

Example:
    # Merge a trained LoRA adapter with the default Qwen base model
    python merge_lora_weights.py --lora_path ./my_devops_model --output_path ./merged_devops_model
    
    # Use a different base model
    python merge_lora_weights.py --base_model "Qwen/Qwen3-4B" --lora_path ./my_adapter --output_path ./merged_model

Requirements:
    - torch
    - transformers
    - peft
    - Sufficient GPU memory or disk space for model loading and saving
"""

import torch
import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def merge_lora_weights(base_model_name, lora_model_path, output_path):
    """Merge LoRA weights into the base model"""
    
    print(f"Loading base model: {base_model_name}")
    # Load the base model directly on GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # Force everything on GPU
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from: {lora_model_path}")
    # Load the LoRA model
    model = PeftModel.from_pretrained(
        base_model, 
        lora_model_path
    )
    
    print("Merging LoRA weights...")
    # Merge the LoRA weights
    merged_model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
    
    print(f"Saving merged model to: {output_path}")
    # Save the merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Model merging completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B", help="Base model name")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA model")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for merged model")
    
    args = parser.parse_args()
    
    merge_lora_weights(args.base_model, args.lora_path, args.output_path)