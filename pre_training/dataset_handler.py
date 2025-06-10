import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetHandler:
    """Handler for different types of datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = ['json', 'jsonl', 'txt', 'csv', 'huggingface']
    
    def load_dataset(self) -> Dataset:
        """Load dataset based on configuration"""
        dataset_type = self.config.get('dataset_type', 'huggingface')
        
        if dataset_type == 'huggingface':
            return self._load_huggingface_dataset()
        elif dataset_type == 'custom':
            return self._load_custom_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _load_huggingface_dataset(self) -> Dataset:
        """Load dataset from Hugging Face"""
        return load_dataset(
            self.config['dataset_name'],
            self.config['dataset_config']
        )
    
    def _load_custom_dataset(self) -> Dataset:
        """Load custom dataset from local files"""
        data_path = Path(self.config['dataset_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        file_format = data_path.suffix[1:]  # Remove the dot
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        if file_format == 'json':
            return self._load_json_dataset(data_path)
        elif file_format == 'jsonl':
            return self._load_jsonl_dataset(data_path)
        elif file_format == 'txt':
            return self._load_txt_dataset(data_path)
        elif file_format == 'csv':
            return self._load_csv_dataset(data_path)
    
    def _load_json_dataset(self, path: Path) -> Dataset:
        """Load JSON dataset"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_dict(data)
    
    def _load_jsonl_dataset(self, path: Path) -> Dataset:
        """Load JSONL dataset"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data)
    
    def _load_txt_dataset(self, path: Path) -> Dataset:
        """Load text dataset"""
        with open(path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        return Dataset.from_dict({'text': texts})
    
    def _load_csv_dataset(self, path: Path) -> Dataset:
        """Load CSV dataset"""
        return Dataset.from_csv(str(path))
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset based on configuration"""
        if 'preprocessing' not in self.config:
            return dataset
        
        preprocessing_steps = self.config['preprocessing']
        
        for step in preprocessing_steps:
            if step['type'] == 'filter':
                dataset = dataset.filter(
                    lambda x: eval(step['condition']),
                    desc=f"Filtering: {step['condition']}"
                )
            elif step['type'] == 'map':
                dataset = dataset.map(
                    lambda x: eval(step['function']),
                    desc=f"Mapping: {step['function']}"
                )
            elif step['type'] == 'shuffle':
                dataset = dataset.shuffle(seed=step.get('seed', 42))
            elif step['type'] == 'select':
                dataset = dataset.select(range(step['start'], step['end']))
        
        return dataset
    
    def split_dataset(self, dataset: Dataset) -> Dict[str, Dataset]:
        """Split dataset into train/validation/test sets"""
        split_config = self.config.get('split', {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1
        })
        
        splits = dataset.train_test_split(
            test_size=split_config['validation'] + split_config['test'],
            seed=self.config.get('seed', 42)
        )
        
        if split_config['test'] > 0:
            val_test = splits['test'].train_test_split(
                test_size=split_config['test'] / (split_config['validation'] + split_config['test']),
                seed=self.config.get('seed', 42)
            )
            return {
                'train': splits['train'],
                'validation': val_test['train'],
                'test': val_test['test']
            }
        else:
            return {
                'train': splits['train'],
                'validation': splits['test']
            } 