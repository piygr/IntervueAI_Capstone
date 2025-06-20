import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import numpy as np
from typing import Dict, List, Any
import json
import logging
from pathlib import Path
from tqdm import tqdm
from utils import get_device
from models.llama3 import LLama3ForCausalLM, LLama3Config
from safetensors.torch import load_file as safe_load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for language models"""
    
    def __init__(self, model_path: str):
        SEED = 42
        self.device = get_device(seed=SEED) 
        self.model_path = Path(model_path)
        logger.info(f"Loading model from {model_path}")
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     config=config
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        # Load custom model configuration and model
        config = LLama3Config.from_pretrained(model_path)
        self.model = LLama3ForCausalLM(config)
        
        # Load the trained weights from safetensors
        state_dict = safe_load_file(str(self.model_path / "model.safetensors"))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize metrics
        self.perplexity_metric = evaluate.load('perplexity')
        self.rouge_metric = evaluate.load('rouge')
        self.bleu_metric = evaluate.load('bleu')
    
    # Calculates perplexity, a measure of how "surprised" the model is by the dataset.
    def evaluate_perplexity(self, dataset: List[str]) -> float:
        """Calculate perplexity score"""
        encodings = self.tokenizer(dataset, return_tensors="pt", padding=True, truncation=True)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs["loss"]
        
        return torch.exp(loss).item()
    # Measures overlap of words and phrases between generated text and ground truth. Returns:
    # ROUGE-1, ROUGE-2, ROUGE-L
    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        result = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        return {k: float(v) for k, v in result.items()}
    
    # Measures n-gram precision for generated text. BLEU is common in translation
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score"""
        result = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        return result['bleu']
    
    def generate_responses(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """Generate responses for evaluation using greedy decoding"""
        responses = []
        self.model.eval()
        for prompt in tqdm(prompts, desc="Generating responses"):
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # outputs = self.model.generate(
            #     **inputs,
            #     max_length=max_length,
            #     num_return_sequences=1,
            #     temperature=0.7,
            #     do_sample=True,
            #     top_p=0.9
            # )
            # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            generated = input_ids
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(input_ids=generated)
                    logits = outputs["logits"]
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id], dim=-1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
            response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            responses.append(response)
        return responses
    
    def evaluate_model(self, test_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run all evaluations"""
        results = {}
        
        # Evaluate perplexity
        logger.info("Calculating perplexity...")
        results['perplexity'] = self.evaluate_perplexity(test_data['texts'])
        
        # Generate responses for ROUGE and BLEU
        logger.info("Generating responses for evaluation...")
        logger.info(f"prompts {test_data['prompts']}")
        predictions = self.generate_responses(test_data['prompts'])
        logger.info(f"predictions {predictions}")
        # Evaluate ROUGE
        logger.info("Calculating ROUGE scores...")
        results['rouge'] = self.evaluate_rouge(predictions, test_data['references'])
        
        # Evaluate BLEU
        logger.info("Calculating BLEU score...")
        results['bleu'] = self.evaluate_bleu(predictions, test_data['references'])
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {output_path}")

def main():
    # Example usage
    # model_path = "./smolLM2-trained"
    model_path = "/Users/gitesh.grover/Study/AI-ERA/IntervueAI_Capstone/pre_training/llama3_final_model"
    evaluator = ModelEvaluator(model_path)
    
    # Example test data
    test_data = {
        'texts': [
            "This is a test sentence for perplexity calculation.",
            "Another test sentence for evaluation."
        ],
        'prompts': [
            "Once upon a time",
            "The capital of France is"
        ],
        'references': [
            "Once upon a time there was a magical kingdom.",
            "The capital of France is Paris, a beautiful city."
        ]
    }
    
    # Run evaluation
    results = evaluator.evaluate_model(test_data)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print("\nROUGE Scores:")
    for metric, score in results['rouge'].items():
        print(f"{metric}: {score:.4f}")
    print(f"\nBLEU Score: {results['bleu']:.4f}")

if __name__ == "__main__":
    main() 