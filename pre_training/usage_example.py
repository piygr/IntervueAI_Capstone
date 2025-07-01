#!/usr/bin/env python3
"""
Usage example for the updated Llama3 model with causal masking.
This demonstrates how to use the model for both training and inference.
"""

import torch
from models.llama3 import LLama3ForCausalLM, LLama3Config

def inference_example():
    """Example of using the model for inference with causal masking."""
    print("="*50)
    print("INFERENCE EXAMPLE")
    print("="*50)
    
    # Create a small model
    config = LLama3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    
    model = LLama3ForCausalLM(config)
    model.eval()
    
    # Example 1: Inference without attention mask (uses causal masking)
    print("\n1. Inference without attention mask (automatic causal masking):")
    input_ids = torch.randint(1, 1000, (1, 10))  # Single sequence
    print(f"Input: {input_ids[0].tolist()}")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=None)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]  # Get logits for next token
        next_token = torch.argmax(next_token_logits).item()
        print(f"Predicted next token: {next_token}")
    
    # Example 2: Inference with attention mask
    print("\n2. Inference with attention mask:")
    # Create a padded sequence
    padded_input = torch.zeros(1, 10, dtype=torch.long)
    padded_input[0, :7] = input_ids[0, :7]  # Only first 7 tokens are valid
    attention_mask = (padded_input != 0).long()
    
    print(f"Padded input: {padded_input[0].tolist()}")
    print(f"Attention mask: {attention_mask[0].tolist()}")
    
    with torch.no_grad():
        outputs = model(input_ids=padded_input, attention_mask=attention_mask)
        logits = outputs.logits
        next_token_logits = logits[0, 6, :]  # Get logits at position 6 (last valid token)
        next_token = torch.argmax(next_token_logits).item()
        print(f"Predicted next token: {next_token}")

def training_example():
    """Example of using the model for training with proper masking."""
    print("\n" + "="*50)
    print("TRAINING EXAMPLE")
    print("="*50)
    
    # Create a small model
    config = LLama3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    
    model = LLama3ForCausalLM(config)
    model.train()
    
    # Example 1: Training without padding
    print("\n1. Training without padding (all sequences same length):")
    batch_size = 4
    seq_length = 8
    
    input_ids = torch.randint(1, 1000, (batch_size, seq_length))
    labels = input_ids.clone()
    
    # No attention mask needed - model uses causal masking
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=None)
    loss = outputs.loss
    print(f"Training loss: {loss.item():.4f}")
    
    # Example 2: Training with padded sequences
    print("\n2. Training with padded sequences:")
    # Create padded sequences
    padded_input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    padded_input_ids[0, :6] = input_ids[0, :6]  # 6 tokens + 2 padding
    padded_input_ids[1, :4] = input_ids[1, :4]  # 4 tokens + 4 padding
    padded_input_ids[2, :7] = input_ids[2, :7]  # 7 tokens + 1 padding
    padded_input_ids[3, :5] = input_ids[3, :5]  # 5 tokens + 3 padding
    
    # Create attention mask
    attention_mask = (padded_input_ids != 0).long()
    labels_padded = padded_input_ids.clone()
    
    print(f"Padded input shape: {padded_input_ids.shape}")
    print(f"Attention mask:\n{attention_mask}")
    
    outputs = model(input_ids=padded_input_ids, labels=labels_padded, attention_mask=attention_mask)
    loss = outputs.loss
    print(f"Training loss with padding: {loss.item():.4f}")
    
    # Example 3: Using the helper methods
    print("\n3. Using helper methods for attention mask creation:")
    
    # Create attention mask using the static method
    attention_mask_auto = LLama3ForCausalLM.create_attention_mask(padded_input_ids, pad_token_id=0)
    print(f"Auto-generated attention mask:\n{attention_mask_auto}")
    print(f"Matches manual mask: {torch.equal(attention_mask, attention_mask_auto)}")
    
    # Create causal mask
    causal_mask = LLama3ForCausalLM.create_causal_attention_mask(seq_length, padded_input_ids.device)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask (upper triangle should be True):\n{causal_mask[:5, :5]}")

def generation_example():
    """Example of using the model for text generation."""
    print("\n" + "="*50)
    print("GENERATION EXAMPLE")
    print("="*50)
    
    # Create a small model
    config = LLama3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    
    model = LLama3ForCausalLM(config)
    model.eval()
    
    # Simple greedy generation
    print("\nSimple greedy generation:")
    input_ids = torch.randint(1, 1000, (1, 5))  # Start with 5 tokens
    print(f"Initial sequence: {input_ids[0].tolist()}")
    
    max_length = 10
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(input_ids=generated, attention_mask=None)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            print(f"Generated token: {next_token.item()}, Sequence: {generated[0].tolist()}")

if __name__ == "__main__":
    print("Llama3 Model Usage Examples with Causal Masking")
    print("="*60)
    
    try:
        inference_example()
        training_example()
        generation_example()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✅ The model now properly implements causal masking:")
        print("   - When attention_mask=None, causal masking is automatically applied")
        print("   - When attention_mask is provided, it handles padding correctly")
        print("   - Loss calculation respects the attention mask for padded tokens")
        print("   - Helper methods are available for creating attention masks")
        print("   - The model works for both training and inference scenarios")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc() 