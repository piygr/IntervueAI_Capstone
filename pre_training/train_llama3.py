import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from models.llama3 import create_llama3_1b, create_llama3_3b

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def prepare_dataset():
    # Load the WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Combine all text chunks
    texts = []
    for split in ['train', 'validation', 'test']:
        texts.extend([text for text in dataset[split]['text'] if len(text.strip()) > 0])
    
    # Split into train and validation sets
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    
    return train_texts, val_texts

def train_model(model_type="custom-llama3-1b"):
    # Initialize wandb
    wandb.init(project="llama3-training", name=f"{model_type}-pretraining")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") # Using Llama-2 tokenizer
    logger.info(f"Creating {model_type} model...")
    if model_type == "custom-llama3-1b":
        model = create_llama3_1b()
    elif model_type == "custom-llama3-3b":
        model = create_llama3_3b()
    else:
         model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_type}") #loading given base model
    
    # Add special tokens if needed
    special_tokens = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare dataset
    train_texts, val_texts = prepare_dataset()
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    
    # Training arguments with overfitting prevention measures
    training_args = TrainingArguments(
        output_dir="./llama3_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,  # L2 regularization
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Mixed precision training
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_grad_norm=1.0,  # Gradient clipping
        report_to="wandb"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Add early stopping callback
    from transformers import EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    trainer.add_callback(early_stopping)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model("./llama3_final_model")
    tokenizer.save_pretrained("./llama3_final_model")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Llama3 model")
    parser.add_argument("--model_type", type=str, default="custom-llama3-1b", choices=["custom-llama3-1b", "custom-llama3-3b", "llama-2-7b-hf"],
                      help="model type to train (custom 1b, 3b or llama-2-7b-hf)")
    args = parser.parse_args()
    
    train_model(args.model_type) 