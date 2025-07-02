from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import torch
from pdb import set_trace
from utils import get_device
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_qlora")

# Model and dataset
model_name = "meta-llama/Llama-3.2-1B"
dataset_name = "Abirate/english_quotes"  # small text dataset for demonstration
device = get_device()
use_4bit = False
torch_dtype = torch.float32

try:
    import bitsandbytes as bnb
    # import from transformers import BitsAndBytesConfig

    if device.type != "cuda":
        logger.warning("⚠️ 4-bit not available on this system as not cuda. Falling back to float32.")
    elif not bnb.utils.is_4bit_available():
        logger.warning("⚠️ 4-bit not available on this system. Falling back to float32.")
    else:
        use_4bit = True
        torch_dtype = torch.bfloat16  # or torch.float16 depending on GPU
        logger.info("✅ Using bitsandbytes 4-bit quantization.")
except ImportError:
    logger.warning("⚠️ bitsandbytes not installed or incompatible. Using float32 fallback.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Needed for padding in causal LM
# tokenizer.pad_token = "<|finetune_right_pad_id|>"

logger.info(f"Loading dataset {dataset_name}")
# Load dataset and preprocess
dataset = load_dataset(dataset_name, split="train")

def preprocess(example):
    return tokenizer(example["quote"], truncation=True, padding="max_length", max_length=512)

logger.info("Tokenizing dataset")
tokenized_dataset = dataset.map(preprocess, batched=True)
# set_trace()

logger.info(f"Loading Model {model_name} with load_in_4bit as {use_4bit} and torch_dtype as {torch_dtype}")
# Load model in 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=use_4bit,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # ),
    torch_dtype=torch_dtype,
)

logger.info("Preparing Model for k bit training")
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure QLoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
logger.info(f"Configured PEFT model with Lora config {lora_config} ")


# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-llama3-1b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=device.type == "cuda",  # Mixed precision training
    learning_rate=2e-4,
    report_to="none",
    # remove_unused_columns=False,  # Important for SFTTrainer
    # dataloader_pin_memory=False,   # Reduce memory usage
)

# Trainer
trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    args=training_args,
    # packing=False,
    # max_seq_length=128,
)

logger.info(f"Starting Training ")
trainer.train()
logger.info(f"Trajning finished")
