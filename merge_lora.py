#!/usr/bin/env python3
"""
Quick script to merge your specific LoRA checkpoint to full model
"""

from unsloth import FastLanguageModel
import os
import logging

logging.basicConfig(level=logging.INFO)

def quick_merge():
    # Your actual paths
    lora_path = "qwen3_fine_tune_output/checkpoint-20"
    output_path = "qwen3-devops-final-merged"
    
    print(f"🔄 Loading LoRA model from: {lora_path}")
    
    # Verify the checkpoint exists and has required files
    if not os.path.exists(lora_path):
        print(f"❌ Path not found: {lora_path}")
        return False
    
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(lora_path, f))]
    
    if missing:
        print(f"❌ Missing files: {missing}")
        print("📁 Files in directory:")
        for f in os.listdir(lora_path):
            print(f"  - {f}")
        return False
    
    print("✅ LoRA checkpoint found with required files")
    
    try:
        # Load the LoRA model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        
        print("✅ Model loaded successfully")
        
        # Merge and save as 16-bit model
        print(f"💾 Merging and saving to: {output_path}")
        model.save_pretrained_merged(
            output_path, 
            tokenizer, 
            save_method="merged_16bit"
        )
        
        print("✅ Model merged and saved successfully!")
        
        # Verify output
        if os.path.exists(output_path):
            print(f"📁 Output files:")
            for f in os.listdir(output_path):
                size = os.path.getsize(os.path.join(output_path, f)) / (1024*1024)
                print(f"  - {f}: {size:.1f} MB")
        
        print(f"\n🎉 Success! Your merged model is ready at: {output_path}")
        print(f"\n🚀 To serve with vLLM:")
        print(f"python -m vllm.entrypoints.openai.api_server --model {output_path} --host 0.0.0.0 --port 8000")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

if __name__ == "__main__":
    success = quick_merge()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you're in the right directory")
        print("2. Check that the checkpoint was saved properly during training")
        print("3. Verify you have enough disk space for the merged model")
