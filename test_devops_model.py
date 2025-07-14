#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DevOps Model Testing Script
Test your fine-tuned Qwen3 DevOps model with comprehensive scenarios
"""

import torch
import json
import time
from typing import List, Dict, Optional
from unsloth import FastLanguageModel
import argparse
from pathlib import Path

class DevOpsModelTester:
    def __init__(self, model_path: str = "./qwen3-devops-model", base_model: str = "unsloth/Qwen3-4B"):
        """
        Initialize the DevOps model tester
        
        Args:
            model_path: Path to the fine-tuned model directory
            base_model: Base model name for fallback
        """
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model"""
        print(f"🔄 Loading model from: {self.model_path}")
        
        try:
            # Try to load fine-tuned model
            if Path(self.model_path).exists():
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    load_in_8bit=False,
                )
                print("✅ Fine-tuned model loaded successfully!")
            else:
                print(f"⚠️ Fine-tuned model not found at {self.model_path}")
                print(f"🔄 Loading base model: {self.base_model}")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.base_model,
                    max_seq_length=2048,
                    load_in_4bit=True,
                    load_in_8bit=False,
                )
                print("✅ Base model loaded for comparison!")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        enable_thinking: bool = False
    ) -> str:
        """
        Generate response for a given prompt
        
        Args:
            prompt: Input prompt/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            enable_thinking: Enable reasoning mode for Qwen3
            
        Returns:
            Generated response string
        """
        # Prepare conversation
        messages = [{"role": "user", "content": prompt}]
        
        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response, generation_time

    def test_devops_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Test model on multiple DevOps scenarios
        
        Args:
            scenarios: List of test scenarios with prompts and expected categories
            
        Returns:
            List of test results
        """
        results = []
        
        print(f"\n🧪 Testing {len(scenarios)} DevOps scenarios...")
        print("=" * 80)
        
        for i, scenario in enumerate(scenarios, 1):
            prompt = scenario["prompt"]
            category = scenario.get("category", "general")
            difficulty = scenario.get("difficulty", "unknown")
            
            print(f"\n📝 Test {i}/{len(scenarios)} - Category: {category} | Difficulty: {difficulty}")
            print(f"🔵 Question: {prompt}")
            
            # Generate response
            response, gen_time = self.generate_response(
                prompt,
                enable_thinking=scenario.get("enable_thinking", False)
            )
            
            print(f"🤖 DevOps Assistant: {response}")
            print(f"⏱️ Generation time: {gen_time:.2f}s")
            print("-" * 80)
            
            # Store results
            result = {
                "test_id": i,
                "prompt": prompt,
                "response": response,
                "category": category,
                "difficulty": difficulty,
                "generation_time": gen_time,
                "response_length": len(response),
                "contains_code": "```" in response
            }
            results.append(result)
        
        return results

    def evaluate_responses(self, results: List[Dict]) -> Dict:
        """
        Evaluate the quality of responses
        
        Args:
            results: Test results from test_devops_scenarios
            
        Returns:
            Evaluation metrics
        """
        print("\n📊 EVALUATION METRICS")
        print("=" * 50)
        
        total_tests = len(results)
        code_responses = sum(1 for r in results if r["contains_code"])
        avg_response_length = sum(r["response_length"] for r in results) / total_tests
        avg_gen_time = sum(r["generation_time"] for r in results) / total_tests
        
        # Category breakdown
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "avg_length": 0, "code_responses": 0}
            categories[cat]["count"] += 1
            categories[cat]["avg_length"] += result["response_length"]
            if result["contains_code"]:
                categories[cat]["code_responses"] += 1
        
        # Calculate averages
        for cat in categories:
            categories[cat]["avg_length"] /= categories[cat]["count"]
            categories[cat]["code_percentage"] = (categories[cat]["code_responses"] / categories[cat]["count"]) * 100
        
        metrics = {
            "total_tests": total_tests,
            "code_responses": code_responses,
            "code_percentage": (code_responses / total_tests) * 100,
            "avg_response_length": avg_response_length,
            "avg_generation_time": avg_gen_time,
            "categories": categories
        }
        
        # Print metrics
        print(f"Total tests: {total_tests}")
        print(f"Responses with code: {code_responses}/{total_tests} ({metrics['code_percentage']:.1f}%)")
        print(f"Average response length: {avg_response_length:.0f} characters")
        print(f"Average generation time: {avg_gen_time:.2f} seconds")
        
        print(f"\nCategory breakdown:")
        for cat, stats in categories.items():
            print(f"  {cat}: {stats['count']} tests, {stats['avg_length']:.0f} chars avg, {stats['code_percentage']:.1f}% with code")
        
        return metrics

    def save_results(self, results: List[Dict], metrics: Dict, output_file: str = "test_results.json"):
        """Save test results to JSON file"""
        output_data = {
            "model_path": self.model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "results": results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")

def get_comprehensive_test_scenarios() -> List[Dict]:
    """
    Get comprehensive test scenarios covering various DevOps areas
    """
    return [
        # Docker scenarios
        {
            "prompt": "How do I create a secure multi-stage Dockerfile for a Python Flask application?",
            "category": "docker",
            "difficulty": "intermediate"
        },
        {
            "prompt": "What are the best practices for optimizing Docker image size for a Node.js application?",
            "category": "docker", 
            "difficulty": "beginner"
        },
        {
            "prompt": "How do I implement proper logging and monitoring in Docker containers?",
            "category": "docker",
            "difficulty": "intermediate"
        },
        
        # Kubernetes scenarios
        {
            "prompt": "Create a Kubernetes deployment with horizontal pod autoscaling and proper resource limits",
            "category": "kubernetes",
            "difficulty": "intermediate"
        },
        {
            "prompt": "How do I implement a blue-green deployment strategy in Kubernetes?",
            "category": "kubernetes",
            "difficulty": "advanced"
        },
        {
            "prompt": "Design a Kubernetes network policy for microservices security",
            "category": "kubernetes",
            "difficulty": "advanced"
        },
        
        # CI/CD scenarios
        {
            "prompt": "Create a GitHub Actions workflow for a Python application with testing and deployment",
            "category": "cicd",
            "difficulty": "intermediate"
        },
        {
            "prompt": "How do I implement security scanning in my CI/CD pipeline?",
            "category": "cicd",
            "difficulty": "intermediate"
        },
        
        # Infrastructure as Code
        {
            "prompt": "Write Terraform configuration for an AWS ECS cluster with load balancer",
            "category": "iac",
            "difficulty": "advanced"
        },
        {
            "prompt": "How do I manage Terraform state files securely in a team environment?",
            "category": "iac",
            "difficulty": "intermediate"
        },
        
        # Monitoring and Observability
        {
            "prompt": "Set up Prometheus monitoring for a microservices architecture",
            "category": "monitoring",
            "difficulty": "advanced"
        },
        {
            "prompt": "How do I create effective alerts and dashboards for application monitoring?",
            "category": "monitoring",
            "difficulty": "intermediate"
        },
        
        # Security
        {
            "prompt": "Implement container security scanning in CI/CD pipeline",
            "category": "security",
            "difficulty": "intermediate"
        },
        {
            "prompt": "How do I secure secrets management in Kubernetes?",
            "category": "security",
            "difficulty": "advanced"
        },
        
        # Troubleshooting
        {
            "prompt": "Debug high CPU usage in a Kubernetes pod",
            "category": "troubleshooting",
            "difficulty": "intermediate",
            "enable_thinking": True  # Complex problem-solving
        },
        {
            "prompt": "How do I troubleshoot network connectivity issues between microservices?",
            "category": "troubleshooting",
            "difficulty": "advanced",
            "enable_thinking": True
        },
        
        # General DevOps
        {
            "prompt": "What are the key principles of implementing DevOps culture in an organization?",
            "category": "general",
            "difficulty": "beginner"
        },
        {
            "prompt": "How do I design a disaster recovery strategy for cloud-native applications?",
            "category": "general",
            "difficulty": "advanced",
            "enable_thinking": True
        }
    ]

def get_quick_test_scenarios() -> List[Dict]:
    """Get a smaller set of scenarios for quick testing"""
    return [
        {
            "prompt": "How do I optimize a Dockerfile for a Python application?",
            "category": "docker",
            "difficulty": "intermediate"
        },
        {
            "prompt": "Create a basic Kubernetes deployment with health checks",
            "category": "kubernetes",
            "difficulty": "beginner"
        },
        {
            "prompt": "Write a simple CI/CD pipeline for automated testing",
            "category": "cicd",
            "difficulty": "beginner"
        },
        {
            "prompt": "How do I troubleshoot a failing container?",
            "category": "troubleshooting",
            "difficulty": "intermediate",
            "enable_thinking": True
        }
    ]

def main():
    """Main function to run the testing"""
    parser = argparse.ArgumentParser(description="Test DevOps fine-tuned model")
    parser.add_argument("--model-path", default="./qwen3-devops-model/checkpoint-50", help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="unsloth/Qwen3-4B", help="Base model name")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer scenarios")
    parser.add_argument("--output", default="test_results.json", help="Output file for results")
    parser.add_argument("--categories", nargs="+", help="Filter by categories")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DevOpsModelTester(args.model_path, args.base_model)
    
    # Get test scenarios
    if args.quick:
        scenarios = get_quick_test_scenarios()
        print("🚀 Running quick test with 4 scenarios")
    else:
        scenarios = get_comprehensive_test_scenarios()
        print("🚀 Running comprehensive test with 18 scenarios")
    
    # Filter by categories if specified
    if args.categories:
        scenarios = [s for s in scenarios if s["category"] in args.categories]
        print(f"🔍 Filtered to {len(scenarios)} scenarios for categories: {args.categories}")
    
    # Run tests
    results = tester.test_devops_scenarios(scenarios)
    
    # Evaluate results
    metrics = tester.evaluate_responses(results)
    
    # Save results
    tester.save_results(results, metrics, args.output)
    
    print(f"\n✅ Testing completed! Check {args.output} for detailed results.")

if __name__ == "__main__":
    main()