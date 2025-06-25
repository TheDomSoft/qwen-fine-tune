#!/usr/bin/env python3
"""
Ultra-quick test - just one question to see if the model works
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Quick test function
def quick_test(question="How do I set up a CI/CD pipeline with GitHub Actions?"):
    print("🔄 Loading model...")
    
    # Load
    tokenizer = AutoTokenizer.from_pretrained("./my_devops_model")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, "./my_devops_model")
    model.eval()
    
    print("✅ Model loaded!")
    print(f"❓ Question: {question}")
    
    # Format and generate
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n💡 Response:\n{response.strip()}")

if __name__ == "__main__":
    quick_test()
