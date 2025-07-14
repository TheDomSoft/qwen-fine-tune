#!/usr/bin/env python3
"""
Test your working Qwen3-4B DevOps model with proper formatting
"""

import requests
import json
import time
from datetime import datetime

API_BASE = "http://localhost:8000"
MODEL_NAME = "/app/model"

def clean_response(text):
    """Clean the model response by removing thinking tags"""
    # Remove <think> and </think> tags and content between them
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def test_model(question, max_tokens=500, temperature=0.7):
    """Test the model with a question"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        print(f"🤔 Asking: {question}")
        print("⏳ Generating response...")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            cleaned_content = clean_response(raw_content)
            
            usage = data.get("usage", {})
            
            print(f"✅ Response generated in {response_time:.1f}s")
            print(f"📊 Tokens: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total")
            print("🤖 Model Response:")
            print("-" * 60)
            print(cleaned_content)
            print("-" * 60)
            print()
            
            return {
                "success": True,
                "content": cleaned_content,
                "response_time": response_time,
                "tokens": usage
            }
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {"success": False, "error": str(e)}

def run_devops_test_suite():
    """Run a comprehensive test suite for DevOps scenarios"""
    
    print("🚀 Testing Qwen3-4B DevOps Fine-tuned Model")
    print("=" * 70)
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test questions covering different DevOps areas
    test_cases = [
        {
            "category": "🐳 Docker Optimization",
            "question": "How do I create a multi-stage Dockerfile for a Python web application that minimizes image size and follows security best practices?",
            "expected_keywords": ["multi-stage", "alpine", "non-root", "cache", "security"]
        },
        {
            "category": "☸️ Kubernetes Deployment", 
            "question": "Design a Kubernetes deployment with proper resource limits, health checks, and rolling updates for a high-availability web service.",
            "expected_keywords": ["deployment", "resources", "limits", "readiness", "liveness", "rolling"]
        },
        {
            "category": "🔄 CI/CD Pipeline",
            "question": "Create a GitHub Actions workflow that includes testing, security scanning, and automated deployment to staging and production environments.",
            "expected_keywords": ["github actions", "workflow", "testing", "security", "deployment", "staging"]
        },
        {
            "category": "🔒 Container Security",
            "question": "What are the essential security practices for containerized applications in production environments?",
            "expected_keywords": ["security", "container", "vulnerabilities", "scanning", "least privilege", "secrets"]
        },
        {
            "category": "📊 Monitoring & Observability",
            "question": "How do I implement comprehensive monitoring and logging for a microservices architecture using Prometheus and Grafana?",
            "expected_keywords": ["prometheus", "grafana", "monitoring", "metrics", "logs", "microservices"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📋 Test {i}/5: {test_case['category']}")
        print("=" * 70)
        
        result = test_model(test_case["question"], max_tokens=600, temperature=0.7)
        
        if result["success"]:
            # Check for expected keywords
            content_lower = result["content"].lower()
            found_keywords = [kw for kw in test_case["expected_keywords"] if kw.lower() in content_lower]
            keyword_score = len(found_keywords) / len(test_case["expected_keywords"])
            
            print(f"🎯 Keyword relevance: {keyword_score:.0%} ({len(found_keywords)}/{len(test_case['expected_keywords'])})")
            print(f"✅ Found keywords: {', '.join(found_keywords)}")
            
            if keyword_score < 0.5:
                missing = [kw for kw in test_case["expected_keywords"] if kw.lower() not in content_lower]
                print(f"⚠️ Missing keywords: {', '.join(missing)}")
            
            results.append({
                **result,
                "category": test_case["category"],
                "keyword_score": keyword_score
            })
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            results.append({
                **result,
                "category": test_case["category"]
            })
        
        print()
        if i < len(test_cases):
            print("⏳ Next test in 2 seconds...")
            time.sleep(2)
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = [r for r in results if r["success"]]
    total_tests = len(results)
    success_rate = len(successful_tests) / total_tests * 100
    
    if successful_tests:
        avg_response_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
        avg_keyword_score = sum(r.get("keyword_score", 0) for r in successful_tests) / len(successful_tests)
        total_tokens = sum(r["tokens"].get("total_tokens", 0) for r in successful_tests)
        
        print(f"✅ Success rate: {success_rate:.0f}% ({len(successful_tests)}/{total_tests})")
        print(f"⏱️ Average response time: {avg_response_time:.1f} seconds")
        print(f"🎯 Average keyword relevance: {avg_keyword_score:.0%}")
        print(f"🔢 Total tokens used: {total_tokens:,}")
        
        # Performance assessment
        if success_rate == 100 and avg_keyword_score >= 0.7:
            print("🎉 EXCELLENT: Your model is performing very well!")
        elif success_rate >= 80 and avg_keyword_score >= 0.5:
            print("👍 GOOD: Your model shows solid DevOps knowledge!")
        elif success_rate >= 60:
            print("👌 FAIR: Model is working but could use more training data")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Consider additional fine-tuning")
    else:
        print("❌ All tests failed - check your model deployment")
    
    return results

def interactive_mode():
    """Interactive testing mode"""
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 70)
    print("Ask your DevOps questions! Type 'quit' to exit.")
    print("💡 Try questions about Docker, Kubernetes, CI/CD, security, monitoring...")
    print()
    
    while True:
        try:
            question = input("❓ Your DevOps question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '']:
                break
            
            print()
            result = test_model(question, max_tokens=800, temperature=0.7)
            
            if not result["success"]:
                print(f"❌ Error: {result.get('error')}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main test runner"""
    print("🔍 Checking server health...")
    
    try:
        health_response = requests.get(f"{API_BASE}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server health check failed")
            return
    except:
        print("❌ Cannot connect to server. Make sure it's running at http://localhost:8000")
        return
    
    print("✅ Server is healthy")
    print()
    
    # Run automated tests
    results = run_devops_test_suite()
    
    # Ask for interactive mode
    if any(r["success"] for r in results):
        choice = input("🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode()
    
    print("👋 Testing completed!")

if __name__ == "__main__":
    main()