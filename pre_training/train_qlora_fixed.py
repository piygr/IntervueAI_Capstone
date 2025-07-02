#!/usr/bin/env python3
"""
QLoRA Training Script for Llama3 - Fixed Version
This script includes proper dependency checking and error handling.
"""

import torch
import sys
import os

# Check and install required dependencies
def check_and_install_dependencies():
    """Check if required packages are installed and provide installation instructions."""
    required_packages = {
        'peft': 'peft>=0.4.0',
        'trl': 'trl>=0.7.0', 
        'bitsandbytes': 'bitsandbytes>=0.41.0',
        'transformers': 'transformers>=4.30.0',
        'datasets': 'datasets>=2.12.0',
        'accelerate': 'accelerate>=0.20.0'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(install_name)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print("\n" + "="*60)
        print("MISSING DEPENDENCIES DETECTED")
        print("="*60)
        print("Please install the missing packages using one of these methods:")
        print("\nMethod 1: Install all missing packages at once:")
        print(f"pip install {' '.join(missing_packages)}")
        
        print("\nMethod 2: Install packages one by one:")
        for package in missing_packages:
            print(f"pip install {package}")
        
        print("\nMethod 3: If you're using conda:")
        print("conda install -c conda-forge peft trl bitsandbytes")
        
        print("\nAfter installation, run this script again.")
        sys.exit(1)
    
    print("\n✅ All required dependencies are installed!")

# Check dependencies first
check_and_install_dependencies()

# Now import the required packages
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTTrainer
    import torch
    from pdb import set_trace
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install the missing dependencies and try again.")
    sys.exit(1)

def main():
    """Main training function."""
    print("🚀 Starting QLoRA Training for Llama3")
    print("="*50)
    
    # Model and dataset configuration
    model_name = "meta-llama/Llama-3.2-1B"
    dataset_name = "Abirate/english_quotes"  # small text dataset for demonstration
    
    # Check if we have access to the model
    print(f"📋 Using model: {model_name}")
    print(f"📊 Using dataset: {dataset_name}")
    
    try:
        # Load tokenizer
        print("\n🔧 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"📝 Set pad_token to eos_token: {tokenizer.pad_token}")
        
        # Load dataset
        print("\n📚 Loading dataset...")
        dataset = load_dataset(dataset_name, split="train")
        print(f"📊 Dataset loaded: {len(dataset)} examples")
        
        # Preprocess dataset
        print("\n⚙️ Preprocessing dataset...")
        def preprocess(example):
            return tokenizer(
                example["quote"], 
                truncation=True, 
                padding="max_length", 
                max_length=512,
                return_tensors=None  # Return lists, not tensors
            )
        
        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        print(f"✅ Dataset preprocessed: {len(tokenized_dataset)} examples")
        
        # Load model
        print("\n🤖 Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully")
        
        # Prepare model for k-bit training
        print("\n🔧 Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        print("✅ Model prepared for k-bit training")
        
        # Configure QLoRA
        print("\n⚙️ Configuring QLoRA...")
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        print("✅ QLoRA configured successfully")
        
        # Print model info
        model.print_trainable_parameters()
        
        # Training arguments
        print("\n⚙️ Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="./qlora-llama3-1b",
            per_device_train_batch_size=2,  # Reduced for memory
            gradient_accumulation_steps=8,   # Increased to compensate
            logging_steps=10,
            num_train_epochs=3,
            save_strategy="epoch",
            fp16=True,
            learning_rate=2e-4,
            report_to="none",
            remove_unused_columns=False,  # Important for SFTTrainer
            dataloader_pin_memory=False,   # Reduce memory usage
        )
        
        # Trainer
        print("\n🏋️ Setting up trainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset,
            args=training_args,
            packing=False,
            max_seq_length=128,
        )
        
        print("\n🎯 Starting training...")
        print("="*50)
        
        # Start training
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {training_args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 