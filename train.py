import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, DatasetDict
import gc
import transformers

print(f"🚀 Starting Qwen DevOps Fine-tuning")
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# 1. Load and prepare dataset
print("\n📊 Loading and splitting the dataset...")
full_dataset = load_dataset("json", data_files="devops_data.json", split="train")
print(f"Total samples: {len(full_dataset)}")

# Split the dataset (90% train, 10% validation)
train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# 2. Load model and tokenizer
print("\n🤖 Loading model and tokenizer...")

# Use your proven quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Choose model size based on your preference
model_name = "Qwen/Qwen3-4B"

print(f"Loading: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model loaded: {model_name}")

# 3. Prepare model for k-bit training
print("\n⚙️ Preparing model for LoRA training...")
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank - higher = more parameters but better quality
    lora_alpha=32,  # LoRA scaling parameter
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print(f"📈 Trainable params: {trainable_params:,}")
print(f"📈 All params: {all_param:,}")
print(f"📈 Trainable%: {100 * trainable_params / all_param:.2f}%")

# 5. Data preprocessing
print("\n📝 Preparing data for instruction tuning...")

def format_devops_data(examples):
    """Format the data using Qwen's chat template"""
    formatted_texts = []
    
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
        output = examples["output"][i]
        
        # Create a conversation format
        if input_text and str(input_text).strip():
            conversation = [
                {"role": "system", "content": "You are a helpful DevOps assistant. Provide clear, practical, and accurate information about DevOps practices, tools, and methodologies."},
                {"role": "user", "content": f"{instruction}\n\nContext: {input_text}"},
                {"role": "assistant", "content": output}
            ]
        else:
            conversation = [
                {"role": "system", "content": "You are a helpful DevOps assistant. Provide clear, practical, and accurate information about DevOps practices, tools, and methodologies."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        formatted_texts.append(formatted_text)
    
    return {"text": formatted_texts}

# Apply formatting
print("Formatting training data...")
train_dataset = dataset['train'].map(format_devops_data, batched=True, remove_columns=dataset['train'].column_names)
eval_dataset = dataset['validation'].map(format_devops_data, batched=True, remove_columns=dataset['validation'].column_names)

print("Sample formatted conversation:")
print("=" * 50)
print(train_dataset[0]["text"][:400] + "...")
print("=" * 50)

# 6. Tokenization
def tokenize_function(examples):
    """Tokenize the formatted texts"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=2048,  # Adjust based on your data
        return_tensors=None,
    )
    # Add labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"✅ Tokenization complete")

# 7. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling
    pad_to_multiple_of=8,
)

# 8. Training arguments
print("\n🏋️ Setting up training configuration...")

training_args = TrainingArguments(
    # Output and logging
    output_dir="./qwen_devops_finetuned",
    logging_dir="./logs",
    logging_steps=5,
    
    # Training parameters
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
    
    # Learning schedule
    num_train_epochs=2,  # Start with 2 epochs
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Performance optimizations
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    
    # Miscellaneous
    report_to=None,  # Disable wandb
    seed=42,
    
    # Early stopping patience (optional)
    # early_stopping_patience=3,
)

print(f"📊 Training configuration:")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - Batch size: {training_args.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Learning rate: {training_args.learning_rate}")

# 9. Initialize trainer
print("\n🎯 Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 10. Start training
print("\n🚀 Starting fine-tuning...")
print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(eval_dataset)} samples")

try:
    # Print initial memory status
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"🔧 GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    # Start training
    training_output = trainer.train()
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Final train loss: {training_output.training_loss:.4f}")
    
    # Save the final model
    print("\n💾 Saving fine-tuned model...")
    trainer.save_model("./qwen_devops_final")
    tokenizer.save_pretrained("./qwen_devops_final")
    
    print("✅ Model saved to './qwen_devops_final'")
    
    # 11. Test the fine-tuned model
    print("\n🧪 Testing the fine-tuned model...")
    model.eval()
    
    test_cases = [
        "What is DevOps and why is it important?",
        "Explain the difference between CI and CD",
        "How do you implement infrastructure as code?",
        "What are the benefits of containerization in DevOps?"
    ]
    
    for i, test_question in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Question: {test_question}")
        
        # Format the test input using the same template
        test_conversation = [
            {"role": "system", "content": "You are a helpful DevOps assistant. Provide clear, practical, and accurate information about DevOps practices, tools, and methodologies."},
            {"role": "user", "content": test_question}
        ]
        
        test_input = tokenizer.apply_chat_template(
            test_conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract just the assistant's response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = generated_text.find("assistant\n") + len("assistant\n")
        response = generated_text[response_start:].strip()
        
        print(f"Response: {response}")
    
    print("\n🎉 Fine-tuning completed successfully!")
    print(f"📁 Model saved in: ./qwen_devops_final")
    
except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    
    # Print memory info for debugging
    if torch.cuda.is_available():
        print(f"🔧 GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")
    
    # Try to save partial progress
    try:
        print("💾 Attempting to save partial progress...")
        model.save_pretrained("./qwen_devops_partial")
        tokenizer.save_pretrained("./qwen_devops_partial")
        print("✅ Partial model saved to './qwen_devops_partial'")
    except Exception as save_error:
        print(f"❌ Could not save partial model: {save_error}")

print("\n🏁 Script completed!")