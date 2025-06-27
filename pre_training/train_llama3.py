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
from torchinfo import summary 
# === Step 1: Import your custom model ===
from models.llama3 import LLama3Config, LLama3ForCausalLM, create_llama3_1b, create_llama3_3b, loadLlamaModelWithoutWeights

# === Step 1: Evaluation class using trainerCallback ===
class PromptEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, max_new_tokens=50):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Step 2: Define  Custom PyTorch dataset ===
# class TextDataset(Dataset):
#     def __init__(self, texts, tokenizer, block_size=512):
#         # Join all texts into a single string
#         full_text = "\n\n".join(texts)
        
#         # Tokenize entire corpus
#         tokenized = tokenizer(full_text, return_special_tokens_mask=False, return_attention_mask=False)

#         # Split into blocks
#         input_ids = tokenized["input_ids"]
#         total_length = (len(input_ids) // block_size) * block_size
#         input_ids = input_ids[:total_length]

#         # Create blocks
#         self.examples = [
#             torch.tensor(input_ids[i:i+block_size], dtype=torch.long)
#             for i in range(0, total_length, block_size)
#         ]
    
#     def __len__(self):
#         return len(self.examples)
    
#     def __getitem__(self, idx):
#         return {
#             "input_ids": self.examples[idx],
#             # "attention_mask": torch.ones_like(self.examples[idx]),
#             "labels": self.examples[idx].clone()
#         }

# # === Step 3: Define function to prepare dataset ===
# def prepare_dataset():
#     # Load the WikiText-2 dataset
#     # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
#     # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]") load the first 2% of training dataset
    
#     # Combine all text chunks
#     texts = []
#     for split in ['train', 'validation', 'test']:
#         texts.extend([text for text in dataset[split]['text'] if len(text.strip()) > 0])
#     # Split into train and validation sets (90% train, 10% validation)
#     train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
#     # set_trace()
#     return train_texts, val_texts

# Step 2 & 3 
def prepare_dataset(tokenizer, block_size=512):
    # Loads the data set that contains "train", "validation", and "test" splits where each item is a dictionary with a "text" field.
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    def preprocess(example):
        return tokenizer(example["text"], truncation=False, add_special_tokens=False)

    # tokenize all splits without truncation or adding special tokens. 
    # Removes the original "text" column after tokenization.
    # The output (tokenized) contains: {"input_ids": [...]}
    tokenized = dataset.map(preprocess, batched=True, remove_columns=["text"])

    # set_trace()
    # group into blocks
    # example: tokens = [1, 2, 3, ..., 5120] 
    # becomes 
    # [
    #   {"input_ids": [1, ..., 512], "labels": [1, ..., 512]},
    #   {"input_ids": [513, ..., 1024], "labels": [513, ..., 1024]},
    #   ...
    # ]
    # 
    def group_texts(examples):
        concatenated_input_ids = sum(examples["input_ids"], [])
        concatenated_attention_mask = sum(examples["attention_mask"], [])
        total_length = (len(concatenated_input_ids) // block_size) * block_size
        result = {
            "input_ids": [concatenated_input_ids[i:i+block_size] for i in range(0, total_length, block_size)],
            "attention_mask": [concatenated_attention_mask[i:i+block_size] for i in range(0, total_length, block_size)]
        }
        # Creates labels that are equal to input_ids (next-token prediction).
        result["labels"] = list(result["input_ids"])
        return result

    # split the tokenized and grouped train split into 90% train and 10% validation.
    # Note - It turns out to be a common practice for large datasets 
    lm_datasets = tokenized.map(group_texts, batched=True)
    train_test = lm_datasets["train"].train_test_split(test_size=0.03, seed=42)

    return train_test["train"], train_test["test"]


# === Step 4: Define function to train the model ===
def train_model(model_type="custom-llama3-1b", config_type="0.5B", use_llama2_tokenizer=False):
    SEED = 42
    device = get_device(seed=SEED)

    # === Step 5: Define model ===
    logger.info(f"Creating {model_type} model...")
    input_config = {
        "max_position_embeddings": 2048, 
        "vocab_size": 32004 if use_llama2_tokenizer else 128256,
        "bos_token_id": 32000 if use_llama2_tokenizer else 128000,
        "eos_token_id": 32001 if use_llama2_tokenizer else 128001,
        "pad_token_id": 32002 if use_llama2_tokenizer else 128004
        }
    if model_type == "custom-llama3-1b":
        model = create_llama3_1b(config_type, input_config)
    elif model_type == "custom-llama3-3b":
        model = create_llama3_3b()
    else:
        #  model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_type}") #loading given base model
        model = loadLlamaModelWithoutWeights(model_type, config_type, input_config)

    logger.info(f"loaded model {model_type}")
    print(f"Model: {model}")
    summary(model, 
            input_size=(4, 512),
             dtypes=[torch.long])

    # === Step 6: Define tokenizer ===
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if (use_llama2_tokenizer):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") # Using Llama-2 tokenizer
        # Add special tokens as it doesn't have it
        special_tokens = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "ros_token": "<ROS>" # a random token
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        # model.resize_token_embeddings(len(tokenizer))

    set_trace()
    # === Step 7: Prepare dataset (call the function defined in step 2 & 3) ===
    train_dataset, val_dataset = prepare_dataset(tokenizer)

    # train_dataset = TextDataset(train_texts, tokenizer)
    # val_dataset = TextDataset(val_texts, tokenizer)
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
    parser.add_argument("--model_type", type=str, default="custom-llama3-1b", choices=["custom-llama3-1b", "custom-llama3-3b", "meta-llama/Llama-3.2-1B"],
                      help="model type to train (custom 1b, 3b or llama-2-7b-hf)")
    parser.add_argument("--config_type", type=str, default="0.5B", choices=["0.5B", "1B", "1.5B"],
                      help="config type with approx number of trainable parameters")
    parser.add_argument("--use_llama2_tokenizer", type=bool, default=False,
                      help="Whether to use llana2 type tokenizer with 32k vocab size compare to default llama2 tokenizer of vocab_size 128256")
    args = parser.parse_args()
    
    train_model(args.model_type, args.config_type, args.use_llama2_tokenizer) 