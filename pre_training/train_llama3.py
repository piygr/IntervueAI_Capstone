import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,   
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl, 
    TrainerState,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from utils import get_device
from pdb import set_trace 

# === Step 0: Evaluation class using trainerCallback ===
class PromptEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, max_new_tokens=50):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        # set_trace()
        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device

        logger.info(f"\n🔍 Prompt Evaluation @ step {state.global_step}")

        for prompt in self.prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                    use_cache=False,
                    # do_sample=False,
                    # num_beams=1,
                    # pad_token_id=self.tokenizer.pad_token_id,
                    # eos_token_id=self.tokenizer.eos_token_id,
                )
            # set_trace()    
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info(f"📝 Prompt: {prompt}")
            logger.info(f"🧠 Model Output: {decoded}")
            logger.info("-" * 50)

# Define compute_metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {'accuracy': accuracy}

# === Step 1: Import your custom model ===
from models.llama3 import LLama3Config, LLama3ForCausalLM, create_llama3_1b, create_llama3_3b

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Step 2: Define  Custom PyTorch dataset ===
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=512):
        # Join all texts into a single string
        full_text = "\n\n".join(texts)
        
        # Tokenize entire corpus
        tokenized = tokenizer(full_text, return_special_tokens_mask=False, return_attention_mask=False)

        # Split into blocks
        input_ids = tokenized["input_ids"]
        total_length = (len(input_ids) // block_size) * block_size
        input_ids = input_ids[:total_length]

        # Create blocks
        self.examples = [
            torch.tensor(input_ids[i:i+block_size], dtype=torch.long)
            for i in range(0, total_length, block_size)
        ]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx],
            # "attention_mask": torch.ones_like(self.examples[idx]),
            "labels": self.examples[idx].clone()
        }

# === Step 3: Define function to prepare dataset ===
def prepare_dataset():
    # Load the WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]") load the first 2% of training dataset
    
    # Combine all text chunks
    texts = []
    for split in ['train', 'validation', 'test']:
        texts.extend([text for text in dataset[split]['text'] if len(text.strip()) > 0])
    
    # Split into train and validation sets (90% train, 10% validation)
    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    
    # set_trace()
    return train_texts, val_texts

# === Step 4: Define function to train the model ===
def train_model(model_type="custom-llama3-1b"):
    SEED = 42
    device = get_device(seed=SEED)

    # === Step 5: Define model ===
    logger.info(f"Creating {model_type} model...")
    if model_type == "custom-llama3-1b":
        model = create_llama3_1b()
    elif model_type == "custom-llama3-3b":
        model = create_llama3_3b()
    else:
         model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_type}") #loading given base model

    # === Optional: Patch forward() if missing ===
    # def forward(self, input_ids, attention_mask=None, labels=None):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     logits = self.lm_head(outputs)
    #     loss = None
    #     if labels is not None:
    #         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id)
    #     return {"loss": loss, "logits": logits}

    # if not hasattr(model, "forward"):
    #     model.forward = forward.__get__(model)

    # === Step 6: Define tokenizer ===
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") # Using Llama-2 tokenizer
    # Add special tokens if needed
    special_tokens = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    }
    tokenizer.add_special_tokens(special_tokens)
    # set_trace()
    model.resize_token_embeddings(len(tokenizer))

    # === Step 7: Prepare dataset (call the function defined in step 2 & 3) ===
    train_texts, val_texts = prepare_dataset()
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # Batch padding handling
    

    # === Step 8: Initialize wandb ===
    os.environ["WANDB_MODE"] = "online"
    wandb_run = wandb.init(project="llama3-training", name=f"{model_type}-pretraining", 
    #                  config = {
    #     "model_type": model_type,
    #     "learning_rate": 2e-5,
    #     "epochs": 3,
    #     "batch_size": 4,
    #     "warmup_steps": 500,
    #     "weight_decay": 0.01,
    #     "gradient_accumulation_steps": 4
    # }
    )

    # === Step 9: Training setup ===
    training_args = TrainingArguments(
        seed=SEED,
        output_dir="./llama3_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=device.type == "cuda",  # Mixed precision training
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500, # Save model every 500 steps
        save_total_limit=2, # Save only the best model
        # evaluation_strategy="steps",
        eval_strategy="steps",
        eval_steps=500, # evaluation will happen every 500 steps
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        warmup_steps=500,
        weight_decay=0.01,  # L2 regularization
        learning_rate=2e-5,
        max_grad_norm=1.0,  # Gradient clipping
        report_to="wandb",
        # report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            PromptEvalCallback(
                tokenizer,
                prompts=[
                    "Once upon a time",
                    "The capital of France is"
                ],
                max_new_tokens=50
            )
        ],
        eval_dataset=val_dataset,
        # compute_metrics=compute_metrics,
    )
    
    # Add early stopping callback
    # early_stopping = EarlyStoppingCallback(
    #     early_stopping_patience=3,
    #     early_stopping_threshold=0.01
    # )
    # trainer.add_callback(early_stopping)

    # # Trigger internal setup (does not start training yet)
    # trainer._prepare_training()
    # # Print total steps
    # print(f"📊 Total training steps: {trainer.state.max_steps}")
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model("./llama3_final_model")
    tokenizer.save_pretrained("./llama3_final_model")
    
    logger.info("!!! FINISHED !!!")
    # Close wandb
    wandb_run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Llama3 model")
    parser.add_argument("--model_type", type=str, default="custom-llama3-1b", choices=["custom-llama3-1b", "custom-llama3-3b", "llama-2-7b-hf"],
                      help="model type to train (custom 1b, 3b or llama-2-7b-hf)")
    args = parser.parse_args()
    
    train_model(args.model_type) 