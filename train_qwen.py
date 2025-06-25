import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import argparse

def load_training_data(json_file_path):
    """Load training data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def split_dataset(dataset, test_size=0.2, random_state=42):
    """Split dataset into train and validation sets"""
    if len(dataset) < 2:
        print("Warning: Dataset too small to split. Using all data for training.")
        return dataset, None
    
    # Convert to list for splitting
    data_list = dataset.to_dict()
    
    # Create indices for splitting
    indices = list(range(len(dataset)))
    train_indices, eval_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Can't stratify with small datasets
    )
    
    # Create train and eval datasets
    train_data = {key: [data_list[key][i] for i in train_indices] for key in data_list.keys()}
    eval_data = {key: [data_list[key][i] for i in eval_indices] for key in data_list.keys()}
    
    train_dataset = Dataset.from_dict(train_data)
    eval_dataset = Dataset.from_dict(eval_data)
    
    print(f"Split dataset: {len(train_dataset)} training examples, {len(eval_dataset)} validation examples")
    
    return train_dataset, eval_dataset

def format_prompt(instruction, input_text="", output=""):
    """Format prompt for training"""
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return prompt

def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize the dataset"""
    prompts = []
    for i in range(len(examples['instruction'])):
        prompt = format_prompt(
            examples['instruction'][i],
            examples.get('input', [''] * len(examples['instruction']))[i],
            examples['output'][i]
        )
        prompts.append(prompt)
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    
    # Important: Copy input_ids to labels for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def setup_model_and_tokenizer(model_name="Qwen/Qwen3-4B"):
    """Setup model and tokenizer with LoRA configuration"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with proper configuration - avoid device_map="auto" to prevent meta device issues
    if torch.cuda.is_available():
        device_map = None  # Let trainer handle device placement
        torch_dtype = torch.float16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False  # Disable cache for gradient checkpointing
    )
    
    # Enable gradient checkpointing on the base model before applying LoRA
    model.gradient_checkpointing_enable()
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Important: Set model to training mode
    model.train()
    
    # Enable gradient computation for LoRA parameters
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(
    model, 
    tokenizer, 
    train_dataset,
    eval_dataset=None, 
    output_dir="./qwen3-devops-finetuned",
    num_epochs=5,
    batch_size=1,
    learning_rate=1e-4
):
    """Train the model"""
    
    # Disable gradient checkpointing if it causes issues
    use_gradient_checkpointing = False
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,  # Reduced from 200
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        eval_steps=50 if eval_dataset is not None else None,  # Reduced from 100
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_steps=100,  # Reduced from 500
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gradient_checkpointing else None,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to=None,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        logging_nan_inf_filter=True,
        label_names=["labels"]  # Explicitly set label names
    )
    
    # Custom data collator that properly handles labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Create custom training class to handle the gradient issue
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Override compute_loss to ensure proper gradient flow
            """
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    print("Starting training...")
    
    # Training with error handling
    try:
        trainer.train()
    except RuntimeError as e:
        if "does not require grad" in str(e):
            print("\nGradient checkpointing issue detected. Restarting training without gradient checkpointing...")
            
            # Disable gradient checkpointing
            training_args.gradient_checkpointing = False
            model.gradient_checkpointing_disable()
            
            # Recreate trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                processing_class=tokenizer,  # Use processing_class instead of tokenizer
            )
            
            # Retry training
            trainer.train()
        else:
            raise e
    
    print("Saving model...")
    trainer.save_model()
    
    # Save the tokenizer separately
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to {output_dir}")
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B for DevOps")
    parser.add_argument("--data_file", type=str, required=True, help="Path to JSON training data file")
    parser.add_argument("--output_dir", type=str, default="./qwen3-devops-finetuned", help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Base model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction of data to use for validation")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Using CPU for training.")
    
    # Load training data
    print(f"Loading training data from {args.data_file}")
    dataset = load_training_data(args.data_file)
    print(f"Loaded {len(dataset)} training examples")
    
    # Split dataset into train and validation
    train_dataset, eval_dataset = split_dataset(dataset, test_size=args.validation_split)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    
    # Tokenize datasets
    print("Tokenizing training dataset...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length), 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_eval_dataset = None
    if eval_dataset is not None:
        print("Tokenizing validation dataset...")
        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length), 
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    
    # Train model
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
