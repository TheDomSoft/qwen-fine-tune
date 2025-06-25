#!/usr/bin/env python3
"""
Gradio web interface for testing your MERGED model
This version loads a merged model (no PEFT required)
"""
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DevOpsAssistant:
    def __init__(self, model_path="./merged_devops_model"):
        print("Loading merged model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Fix pad token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the merged model directly (no PEFT needed)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ Merged model ready!")
    
    def respond(self, message, history):
        prompt = f"### Instruction:\n{message}\n\n### Response:\n"
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=2048, 
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(self.model.device),
                attention_mask=inputs.attention_mask.to(self.model.device),
                max_new_tokens=800,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response

# Initialize assistant with merged model
assistant = DevOpsAssistant(model_path="./merged_devops_model")  # Change this path to your merged model

# Create Gradio interface
demo = gr.ChatInterface(
    fn=assistant.respond,
    title="🚀 DevOps Assistant (Merged Fine-tuned Qwen)",
    description="Ask me anything about DevOps, CI/CD, Docker, Kubernetes, etc! (Using merged model)",
    examples=[
        "How do I set up a CI/CD pipeline with GitHub Actions?",
        "What are Docker best practices for production?",
        "How to troubleshoot high CPU usage in Kubernetes?",
        "Explain blue-green deployment strategy",
        "How do I secure a REST API?",
        "What's the difference between containers and VMs?",
        "How to implement monitoring for microservices?"
    ]
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True if you want a public link
    )
