import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import os
import logging
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import json
from pathlib import Path
from dataset_handler import DatasetHandler
from evaluate_model import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmolLM2Trainer:
    """Main trainer class for SmolLM2 model"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize dataset handler
        self.dataset_handler = DatasetHandler(self.config['dataset'])
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            "model_name": "microsoft/phi-2",
            "dataset": {
                "type": "huggingface",
                "name": "wikitext",
                "config": "wikitext-2-raw-v1"
            },
            "training": {
                "output_dir": "./smolLM2-trained",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-5,
                "max_steps": 1000,
                "save_steps": 100,
                "logging_steps": 10,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "max_length": 512,
                "early_stopping_patience": 3
            }
        }
    
    def prepare_dataset(self, tokenizer: Any) -> Dict[str, Any]:
        """Prepare and tokenize the dataset"""
        # Load and preprocess dataset
        dataset = self.dataset_handler.load_dataset()
        dataset = self.dataset_handler.preprocess_dataset(dataset)
        
        # Split dataset
        splits = self.dataset_handler.split_dataset(dataset)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config['training']['max_length']
            )
        
        tokenized_splits = {}
        for split_name, split_data in splits.items():
            tokenized_splits[split_name] = split_data.map(
                tokenize_function,
                batched=True,
                remove_columns=split_data.column_names
            )
        
        return tokenized_splits
    
    def train(self) -> None:
        """Main training function"""
        # Create output directory
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Initialize tokenizer and model
        logger.info("Initializing tokenizer and model")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare dataset
        datasets = self.prepare_dataset(tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            max_steps=self.config['training']['max_steps'],
            save_steps=self.config['training']['save_steps'],
            logging_steps=self.config['training']['logging_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            save_total_limit=2,
            fp16=self.config['training'].get('fp16', True),
            report_to="tensorboard",
            evaluation_strategy="steps",
            eval_steps=self.config['evaluation']['eval_steps'],
            load_best_model_at_end=self.config['evaluation']['save_best_model'],
            metric_for_best_model=self.config['evaluation']['metric_for_best_model'],
            greater_is_better=self.config['evaluation']['greater_is_better'],
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True),
            optim=self.config['training'].get('optim', 'adamw_torch'),
            lr_scheduler_type=self.config['training'].get('lr_scheduler_type', 'cosine'),
            max_grad_norm=self.config['training'].get('max_grad_norm', 1.0)
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config['training']['early_stopping_patience']
            )]
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Evaluate the model
        if 'test' in datasets:
            logger.info("Evaluating model on test set...")
            evaluator = ModelEvaluator(str(output_dir))
            test_data = {
                'texts': [example['text'] for example in datasets['test']],
                'prompts': [example['text'][:50] for example in datasets['test']],
                'references': [example['text'] for example in datasets['test']]
            }
            results = evaluator.evaluate_model(test_data)
            evaluator.save_results(results, output_dir / "evaluation_results.json")
            
            # Print evaluation results
            print("\nEvaluation Results:")
            print(f"Perplexity: {results['perplexity']:.2f}")
            print("\nROUGE Scores:")
            for metric, score in results['rouge'].items():
                print(f"{metric}: {score:.4f}")
            print(f"\nBLEU Score: {results['bleu']:.4f}")

def main():
    # Initialize trainer with config
    trainer = SmolLM2Trainer("pre_training/training_config.json")
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 