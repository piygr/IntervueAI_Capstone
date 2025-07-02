#!/usr/bin/env python3
"""
Training Script for Custom Llama3 Model with Causal Masking
This script uses your custom Llama3 implementation instead of Hugging Face models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
import sys
import os

# Add the current directory to the path so we can import our custom model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.llama3 import LLama3ForCausalLM, LLama3Config, create_llama3_1b
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the pre_training directory and the models folder exists.")
    sys.exit(1)

class SimpleTextDataset(Dataset):
    """Simple dataset for text training."""
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        print("Tokenizing dataset...")
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.encodings.append(encoding)
        print(f"Tokenized {len(self.encodings)} texts")
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings[idx].items()}
        return item

def create_simple_tokenizer():
    """Create a simple tokenizer for demonstration."""
    from transformers import PreTrainedTokenizerFast
    
    # Simple vocabulary
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
    }
    
    # Add basic characters and common words
    for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}"):
        vocab[char] = i + 4
    
    # Add some common words
    common_words = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with", "as", "for", "was", "on", "be", "at", "this", "by", "I", "you", "he", "she", "they", "we", "me", "him", "her", "them", "us"]
    for word in common_words:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Create tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=None,  # We'll create a simple one
        vocab_size=len(vocab),
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    
    # Set vocabulary
    tokenizer.vocab = vocab
    tokenizer.ids_to_tokens = {v: k for k, v in vocab.items()}
    
    return tokenizer

def main():
    """Main training function."""
    print("🚀 Training Custom Llama3 Model with Causal Masking")
    print("="*60)
    
    # Configuration
    config_type = "0.5B"  # Use the smallest model for faster training
    batch_size = 4
    max_length = 64
    num_epochs = 3
    learning_rate = 1e-4
    
    print(f"📋 Configuration:")
    print(f"   Model size: {config_type}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max length: {max_length}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    try:
        # Create tokenizer
        print("\n🔧 Creating tokenizer...")
        tokenizer = create_simple_tokenizer()
        print(f"✅ Tokenizer created with vocab size: {tokenizer.vocab_size}")
        
        # Load dataset
        print("\n📚 Loading dataset...")
        dataset = load_dataset("Abirate/english_quotes", split="train")
        print(f"📊 Dataset loaded: {len(dataset)} examples")
        
        # Prepare texts
        texts = []
        for example in dataset:
            text = example["quote"]
            if len(text) > 10:  # Only use texts with some content
                texts.append(text)
        
        print(f"📝 Prepared {len(texts)} texts for training")
        
        # Create dataset
        print("\n⚙️ Creating dataset...")
        train_dataset = SimpleTextDataset(texts[:100], tokenizer, max_length)  # Use first 100 for demo
        print(f"✅ Dataset created with {len(train_dataset)} examples")
        
        # Create model
        print("\n🤖 Creating model...")
        model = create_llama3_1b(config_type, {"vocab_size": tokenizer.vocab_size})
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"📱 Model moved to device: {device}")
        
        # Create dataloader
        print("\n📦 Creating dataloader...")
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"✅ DataLoader created with {len(dataloader)} batches")
        
        # Setup training
        print("\n⚙️ Setting up training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()
        
        # Training loop
        print("\n🎯 Starting training...")
        print("="*50)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # Use input_ids as labels for language modeling
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")
        
        print("\n✅ Training completed successfully!")
        
        # Save model
        print("\n💾 Saving model...")
        torch.save(model.state_dict(), f"custom_llama3_{config_type}_trained.pth")
        print(f"✅ Model saved to: custom_llama3_{config_type}_trained.pth")
        
        # Test inference
        print("\n🧪 Testing inference...")
        model.eval()
        with torch.no_grad():
            # Create a simple test input
            test_input = torch.randint(1, tokenizer.vocab_size, (1, 5)).to(device)
            print(f"Test input: {test_input[0].tolist()}")
            
            # Generate next token
            outputs = model(input_ids=test_input, attention_mask=None)  # Use causal masking
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            print(f"Predicted next token: {next_token.item()}")
        
        print("\n🎉 All done! Your custom Llama3 model with causal masking is working!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 