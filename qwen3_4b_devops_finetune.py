# -*- coding: utf-8 -*-
"""
Enhanced Qwen3 DevOps Fine-tuning with JSON Dataset Loader
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset, load_dataset
import pandas as pd
import json
import os
from typing import List, Dict
import random

# =============================================================================
# 1. MODEL SETUP
# =============================================================================

def setup_qwen_model(model_size="4B"):
    """Setup Qwen3 model with Unsloth optimizations"""
    
    model_name_map = {
        "4B": "unsloth/Qwen3-4B",
        "8B": "unsloth/Qwen3-8B", 
        "14B": "unsloth/Qwen3-14B"
    }
    
    model_name = model_name_map.get(model_size, "unsloth/Qwen3-4B")
    print(f"🔧 Loading model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

# =============================================================================
# 2. ENHANCED JSON DATASET LOADING
# =============================================================================

def load_devops_data_from_json(json_path="devops_data.json"):
    """
    Load DevOps data from JSON file with error handling and validation
    """
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        print("📄 Creating sample JSON file...")
        create_sample_json(json_path)
    
    try:
        with open(json_path, "r", encoding='utf-8') as file:
            devops_data = json.load(file)
        
        print(f"✅ Loaded {len(devops_data)} examples from {json_path}")
        
        # Validate data structure
        validated_data = validate_dataset(devops_data)
        return validated_data
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        raise

def validate_dataset(data):
    """
    Validate and clean dataset entries
    """
    validated = []
    
    for i, item in enumerate(data):
        # Check required fields
        if not isinstance(item, dict):
            print(f"⚠️ Skipping item {i}: not a dictionary")
            continue
            
        if "problem" not in item or "solution" not in item:
            print(f"⚠️ Skipping item {i}: missing 'problem' or 'solution' field")
            continue
            
        # Check for empty content
        if not item["problem"].strip() or not item["solution"].strip():
            print(f"⚠️ Skipping item {i}: empty problem or solution")
            continue
            
        # Add metadata if missing
        if "category" not in item:
            item["category"] = detect_category(item["problem"])
            
        if "difficulty" not in item:
            item["difficulty"] = detect_difficulty(item["solution"])
            
        validated.append(item)
    
    print(f"✅ Validated {len(validated)} examples (filtered {len(data) - len(validated)} invalid)")
    return validated

def detect_category(problem_text):
    """
    Auto-detect DevOps category from problem text
    """
    problem_lower = problem_text.lower()
    
    if any(word in problem_lower for word in ["docker", "dockerfile", "container"]):
        return "docker"
    elif any(word in problem_lower for word in ["kubernetes", "k8s", "pod", "deployment"]):
        return "kubernetes"
    elif any(word in problem_lower for word in ["ci/cd", "github actions", "pipeline", "jenkins"]):
        return "cicd"
    elif any(word in problem_lower for word in ["terraform", "cloudformation", "infrastructure"]):
        return "iac"
    elif any(word in problem_lower for word in ["monitoring", "prometheus", "grafana", "observability"]):
        return "monitoring"
    elif any(word in problem_lower for word in ["security", "secure", "vulnerability"]):
        return "security"
    else:
        return "general"

def detect_difficulty(solution_text):
    """
    Auto-detect difficulty level from solution complexity
    """
    solution_lower = solution_text.lower()
    
    # Count complexity indicators
    complexity_score = 0
    
    # Code blocks
    complexity_score += solution_text.count("```") // 2
    
    # Complex concepts
    complex_terms = ["multi-stage", "rbac", "network policy", "encryption", 
                    "monitoring", "prometheus", "terraform", "helm"]
    complexity_score += sum(1 for term in complex_terms if term in solution_lower)
    
    # Length indicator
    if len(solution_text) > 2000:
        complexity_score += 2
    elif len(solution_text) > 1000:
        complexity_score += 1
    
    if complexity_score >= 4:
        return "advanced"
    elif complexity_score >= 2:
        return "intermediate"
    else:
        return "beginner"

def create_sample_json(json_path):
    """
    Create a sample JSON file with DevOps examples
    """
    sample_data = [
        {
            "problem": "How do I create a multi-stage Dockerfile for a Node.js application that minimizes image size?",
            "solution": "Here's an optimized multi-stage Dockerfile for Node.js:\n\n```dockerfile\n# Build stage\nFROM node:18-alpine AS builder\nWORKDIR /app\nCOPY package*.json ./\nRUN npm ci --only=production && npm cache clean --force\n\n# Runtime stage\nFROM node:18-alpine AS runtime\nRUN addgroup -g 1001 -S nodejs && adduser -S -u 1001 nextjs -G nodejs\nWORKDIR /app\nCOPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules\nCOPY --chown=nextjs:nodejs . .\nUSER nextjs\nEXPOSE 3000\nCMD [\"npm\", \"start\"]\n```\n\nKey optimizations:\n- Multi-stage build reduces final image size\n- Alpine Linux for smaller base image\n- Non-root user for security\n- Only production dependencies\n- Clean npm cache to reduce size",
            "category": "docker",
            "difficulty": "intermediate"
        },
        {
            "problem": "Design a Kubernetes deployment with proper resource limits, health checks, and rolling updates for a web application",
            "solution": "Here's a production-ready Kubernetes deployment:\n\n```yaml\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: webapp-deployment\n  labels:\n    app: webapp\nspec:\n  replicas: 3\n  strategy:\n    type: RollingUpdate\n    rollingUpdate:\n      maxUnavailable: 1\n      maxSurge: 1\n  selector:\n    matchLabels:\n      app: webapp\n  template:\n    metadata:\n      labels:\n        app: webapp\n    spec:\n      containers:\n      - name: webapp\n        image: webapp:latest\n        ports:\n        - containerPort: 8080\n        resources:\n          requests:\n            memory: \"256Mi\"\n            cpu: \"250m\"\n          limits:\n            memory: \"512Mi\"\n            cpu: \"500m\"\n        livenessProbe:\n          httpGet:\n            path: /health\n            port: 8080\n          initialDelaySeconds: 30\n          periodSeconds: 10\n        readinessProbe:\n          httpGet:\n            path: /ready\n            port: 8080\n          initialDelaySeconds: 5\n          periodSeconds: 5\n        env:\n        - name: NODE_ENV\n          value: \"production\"\n```\n\nThis ensures:\n- Rolling updates with zero downtime\n- Proper resource management\n- Health monitoring\n- Multiple replicas for availability",
            "category": "kubernetes",
            "difficulty": "intermediate"
        },
        {
            "problem": "How do I optimize a Dockerfile for a Python application with security best practices?",
            "solution": "Here's a secure and optimized Python Dockerfile:\n\n```dockerfile\n# Build stage\nFROM python:3.12-slim AS builder\nWORKDIR /app\n\n# Install system dependencies\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n    build-essential \\\n    && rm -rf /var/lib/apt/lists/*\n\n# Copy requirements first for better caching\nCOPY requirements.txt .\nRUN pip install --no-cache-dir --user -r requirements.txt\n\n# Runtime stage\nFROM python:3.12-slim AS runtime\n\n# Create non-root user\nRUN groupadd -r appuser && useradd -r -g appuser appuser\n\n# Install only runtime dependencies\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n    curl \\\n    && rm -rf /var/lib/apt/lists/*\n\n# Copy Python packages from builder\nCOPY --from=builder /root/.local /home/appuser/.local\n\n# Set up application\nWORKDIR /app\nCOPY --chown=appuser:appuser . .\n\n# Switch to non-root user\nUSER appuser\n\n# Add user's local bin to PATH\nENV PATH=/home/appuser/.local/bin:$PATH\n\n# Health check\nHEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\\n    CMD curl -f http://localhost:8000/health || exit 1\n\nEXPOSE 8000\nCMD [\"python\", \"app.py\"]\n```\n\nKey security and optimization features:\n- Multi-stage build reduces final image size by ~60%\n- Slim base image instead of Alpine (better Python compatibility)\n- Non-root user for security\n- Layer caching optimization with requirements.txt first\n- Health check for container orchestration\n- Proper cleanup of package manager cache",
            "category": "docker",
            "difficulty": "intermediate"
        }
    ]
    
    with open(json_path, "w", encoding='utf-8') as file:
        json.dump(sample_data, file, indent=2, ensure_ascii=False)
    
    print(f"📄 Created sample JSON file: {json_path}")

def load_additional_datasets_from_json(additional_paths=[]):
    """
    Load additional DevOps datasets from multiple JSON files
    """
    all_data = []
    
    for path in additional_paths:
        if os.path.exists(path):
            try:
                data = load_devops_data_from_json(path)
                all_data.extend(data)
                print(f"✅ Loaded additional data from {path}")
            except Exception as e:
                print(f"⚠️ Could not load {path}: {e}")
    
    return all_data

def filter_dataset_by_criteria(data, categories=None, difficulties=None, min_length=None):
    """
    Filter dataset by various criteria
    """
    filtered = data.copy()
    
    if categories:
        filtered = [item for item in filtered if item.get("category") in categories]
        print(f"🔍 Filtered by categories {categories}: {len(filtered)} examples")
    
    if difficulties:
        filtered = [item for item in filtered if item.get("difficulty") in difficulties]
        print(f"🔍 Filtered by difficulties {difficulties}: {len(filtered)} examples")
    
    if min_length:
        filtered = [item for item in filtered if len(item["solution"]) >= min_length]
        print(f"🔍 Filtered by min length {min_length}: {len(filtered)} examples")
    
    return filtered

def create_combined_devops_dataset(
    primary_json="devops_data.json",
    additional_jsons=[],
    filter_categories=None,
    filter_difficulties=None,
    max_examples=None
):
    """
    Enhanced dataset creation with filtering and multiple sources
    """
    # Load primary dataset
    print("📊 Loading primary DevOps dataset...")
    primary_data = load_devops_data_from_json(primary_json)
    
    # Load additional datasets
    additional_data = load_additional_datasets_from_json(additional_jsons)
    
    # Combine data
    all_data = primary_data + additional_data
    print(f"📊 Combined dataset size: {len(all_data)} examples")
    
    # Apply filters
    if filter_categories or filter_difficulties:
        all_data = filter_dataset_by_criteria(
            all_data, 
            categories=filter_categories,
            difficulties=filter_difficulties
        )
    
    # Limit examples if specified
    if max_examples and len(all_data) > max_examples:
        all_data = random.sample(all_data, max_examples)
        print(f"🎲 Randomly sampled {max_examples} examples")
    
    # Convert to conversation format
    devops_conversations = []
    for item in all_data:
        conversation = [
            {"role": "user", "content": item["problem"]},
            {"role": "assistant", "content": item["solution"]}
        ]
        devops_conversations.append({"conversations": conversation})
    
    # Create dataset
    dataset = Dataset.from_list(devops_conversations)
    return dataset.shuffle(seed=3407)

# =============================================================================
# 3. DATASET ANALYSIS TOOLS
# =============================================================================

def analyze_dataset(json_path="devops_data.json"):
    """
    Analyze the dataset and provide statistics
    """
    data = load_devops_data_from_json(json_path)
    
    print("\n📊 DATASET ANALYSIS")
    print("=" * 50)
    print(f"Total examples: {len(data)}")
    
    # Category distribution
    categories = {}
    difficulties = {}
    lengths = []
    
    for item in data:
        cat = item.get("category", "unknown")
        diff = item.get("difficulty", "unknown")
        
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
        lengths.append(len(item["solution"]))
    
    print(f"\nCategories:")
    for cat, count in sorted(categories.items()):
        percentage = (count / len(data)) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    print(f"\nDifficulties:")
    for diff, count in sorted(difficulties.items()):
        percentage = (count / len(data)) * 100
        print(f"  {diff}: {count} ({percentage:.1f}%)")
    
    print(f"\nSolution lengths:")
    print(f"  Average: {sum(lengths) / len(lengths):.0f} characters")
    print(f"  Min: {min(lengths)} characters")
    print(f"  Max: {max(lengths)} characters")
    
    return data

# =============================================================================
# 4. ENHANCED TRAINING FUNCTION
# =============================================================================

def train_devops_model_enhanced(
    model_size="4B",
    primary_json="devops_data.json",
    additional_jsons=[],
    filter_categories=None,
    filter_difficulties=None,
    max_examples=None,
    **training_kwargs
):
    """
    Enhanced training function with dataset filtering options
    """
    print("🚀 Setting up enhanced Qwen3 DevOps fine-tuning...")
    
    # Setup model
    model, tokenizer = setup_qwen_model(model_size)
    
    # Create dataset with filters
    print("📊 Creating filtered DevOps dataset...")
    dataset = create_combined_devops_dataset(
        primary_json=primary_json,
        additional_jsons=additional_jsons,
        filter_categories=filter_categories,
        filter_difficulties=filter_difficulties,
        max_examples=max_examples
    )
    print(f"Final dataset size: {len(dataset)} examples")
    
    # Format conversations for training
    def formatting_function(examples):
        texts = []
        for conversation in examples["conversations"]:
            formatted = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(formatted)
        return {"text": texts}
    
    # Apply formatting
    train_dataset = dataset.map(
        formatting_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training configuration with defaults and overrides
    from trl import SFTTrainer, SFTConfig
    
    default_args = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "num_train_epochs": 3,
        "max_steps": 500,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "./qwen3-devops-model",
        "save_steps": 100,
        "save_total_limit": 2,
        "report_to": "none",
    }
    
    # Override with user-provided arguments
    default_args.update(training_kwargs)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(**default_args),
    )
    
    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Train the model
    print("🎯 Starting training...")
    trainer_stats = trainer.train()
    
    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds")
    print(f"Peak memory usage: {used_memory} GB")
    
    return model, tokenizer

# =============================================================================
# 5. USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Analyze existing dataset
    print("🔍 Analyzing dataset...")
    analyze_dataset("devops_data.json")
    
    # Example: Train only on Docker and Kubernetes examples
    print("\n🚀 Starting focused training...")
    model, tokenizer = train_devops_model_enhanced(
        model_size="4B",
        primary_json="devops_data.json",
        filter_categories=["docker", "kubernetes"],
        filter_difficulties=["intermediate", "advanced"],
        max_examples=100,  # Limit for quick testing
        num_train_epochs=1,
        max_steps=50
    )
    
    print("✅ Training completed!")