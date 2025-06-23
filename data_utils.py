"""
Data utilities for Vietnamese GEC with viGEC dataset
"""

import os
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
import logging
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)

class ViGECDataset(Dataset):
    """Vietnamese GEC Dataset for training"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 384,
        is_train: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['source']
        target = item['target']
        
        # Add task prefix for ViT5
        if hasattr(self.tokenizer, 'task_prefix'):
            source = self.tokenizer.task_prefix + source
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(),
            'source_text': source,
            'target_text': target
        }

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with positive/negative pairs"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 384
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['source']
        positive = item['positive']  # gold target
        negatives = item['negatives']  # list of negative samples
        
        # Add task prefix for ViT5
        if hasattr(self.tokenizer, 'task_prefix'):
            source = self.tokenizer.task_prefix + source
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize positive
        positive_encoding = self.tokenizer(
            positive,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize negatives
        negative_encodings = []
        for neg in negatives:
            neg_encoding = self.tokenizer(
                neg,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            negative_encodings.append(neg_encoding)
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'positive_ids': positive_encoding['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
            'negative_ids': torch.stack([neg['input_ids'].squeeze() for neg in negative_encodings]),
            'negative_attention_mask': torch.stack([neg['attention_mask'].squeeze() for neg in negative_encodings]),
            'source_text': source,
            'positive_text': positive,
            'negative_texts': negatives
        }

def normalize_text(text: str) -> str:
    """Normalize Vietnamese text to UTF-8 NFC"""
    # Normalize unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_text(text: str) -> str:
    """Clean and preprocess Vietnamese text"""
    # Normalize
    text = normalize_text(text)
    
    # Remove special characters but keep Vietnamese diacritics
    text = re.sub(r'[^\w\s\u00C0-\u1EF9\u0300-\u036F.,!?;:()"\'-]', '', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    text = re.sub(r'\s*([()"\'])\s*', r' \1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_vigec_dataset(
    dataset_name: str = "phuhuy-se1/viGEC",
    cache_dir: Optional[str] = None,
    train_subset_ratio: float = 1.0,
    validation_subset_ratio: float = 1.0,
    test_subset_ratio: float = 0.05
) -> Dict[str, List[Dict]]:
    """Load and preprocess viGEC dataset
    
    Args:
        dataset_name: HuggingFace dataset identifier
        cache_dir: Directory to cache the dataset
        train_subset_ratio: Ratio of training data to use (0.0-1.0, default 1.0)
        validation_subset_ratio: Ratio of validation data to use (0.0-1.0, default 1.0) 
        test_subset_ratio: Ratio of test data to use (0.0-1.0, default 0.05)
    
    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    
    console.print(f"[bold blue]Loading dataset: {dataset_name}[/bold blue]")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    processed_data = {}
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            console.print(f"[yellow]Processing {split} split...[/yellow]")
            
            split_data = []
            for item in track(dataset[split], description=f"Processing {split}"):
                # Fix column names - dataset uses 'incorrect_text' and 'correct_text'
                source = clean_text(item.get('incorrect_text', item.get('source', '')))
                target = clean_text(item.get('correct_text', item.get('target', '')))
                
                # Skip empty or very short texts
                if len(source.split()) < 3 or len(target.split()) < 3:
                    continue
                
                split_data.append({
                    'source': source,
                    'target': target,
                    'id': item.get('id', len(split_data))
                })
              # Apply subset ratio for faster processing if specified
            subset_ratios = {
                'train': train_subset_ratio,
                'validation': validation_subset_ratio,
                'test': test_subset_ratio
            }
            
            if split in subset_ratios and subset_ratios[split] < 1.0:
                import random
                random.seed(42)  # For reproducibility
                subset_size = int(len(split_data) * subset_ratios[split])
                split_data = random.sample(split_data, subset_size)
                console.print(f"[blue]Using {subset_size} samples ({subset_ratios[split]*100:.1f}%) from {split} set[/blue]")
            
            processed_data[split] = split_data
            console.print(f"[green]{split}: {len(split_data)} samples[/green]")
    
    return processed_data

def create_data_loaders(
    data: Dict[str, List[Dict]],
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 384,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Create data loaders for training"""
    
    data_loaders = {}
    
    for split, split_data in data.items():
        dataset = ViGECDataset(
            data=split_data,
            tokenizer=tokenizer,
            max_length=max_length,
            is_train=(split == 'train')
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        data_loaders[split] = data_loader
    
    return data_loaders

def get_model_and_tokenizer(model_name: str):
    """Get model and tokenizer for Vietnamese GEC"""
    
    console.print(f"[bold blue]Loading model: {model_name}[/bold blue]")
    
    if 'bartpho' in model_name.lower():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    elif 'vit5' in model_name.lower():
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add task prefix for ViT5
        if not hasattr(tokenizer, 'task_prefix'):
            tokenizer.task_prefix = "grammatical error correction: "
            console.print(f"[yellow]Added ViT5 task prefix: {tokenizer.task_prefix}[/yellow]")
        
    else:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
      # Add special tokens if needed
    special_tokens = ['<gec>', '</gec>']
    
    # Safe way to check vocabulary for different tokenizer types
    try:
        if hasattr(tokenizer, 'vocab'):
            # Standard tokenizers (BERT, etc.)
            vocab = tokenizer.vocab
        elif hasattr(tokenizer, 'get_vocab'):
            # SentencePiece tokenizers (BARTpho, etc.)
            vocab = tokenizer.get_vocab()
        else:
            # Fallback: assume all tokens are new
            vocab = {}
        
        new_tokens = [token for token in special_tokens if token not in vocab]
        
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
            console.print(f"[yellow]Added {len(new_tokens)} new tokens[/yellow]")
        else:
            console.print("[blue]No new tokens needed[/blue]")
            
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check vocabulary - {e}[/yellow]")
        # Skip adding special tokens if we can't check vocabulary
    
    return model, tokenizer

def save_processed_data(data: Dict[str, List[Dict]], output_dir: str):
    """Save processed data to disk"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split, split_data in data.items():
        output_path = os.path.join(output_dir, f"{split}.json")
        
        df = pd.DataFrame(split_data)
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        
        console.print(f"[green]Saved {split} data to {output_path}[/green]")

def load_processed_data(data_dir: str) -> Dict[str, List[Dict]]:
    """Load processed data from disk"""
    
    data = {}
    
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f"{split}.json")
        
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records')
            data[split] = df.to_dict('records')
            console.print(f"[green]Loaded {split}: {len(data[split])} samples[/green]")
    
    return data

def can_train_base_model() -> bool:
    """Check if system can train base model (GPU availability, memory, etc.)"""
    
    import torch
    
    if not torch.cuda.is_available():
        console.print("[red]❌ CUDA not available. Training will be very slow on CPU.[/red]")
        return False
    
    # Check GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    console.print(f"[blue]GPU Memory: {gpu_memory:.1f} GB[/blue]")
    
    if gpu_memory < 8:
        console.print("[red]❌ Insufficient GPU memory. Need at least 8GB for base model training.[/red]")
        return False
    
    # Check available disk space
    import shutil
    disk_space = shutil.disk_usage('.').free / 1e9
    console.print(f"[blue]Available disk space: {disk_space:.1f} GB[/blue]")
    
    if disk_space < 10:
        console.print("[red]❌ Insufficient disk space. Need at least 10GB for model checkpoints.[/red]")
        return False
    
    console.print("[green]✅ System ready for base model training![/green]")
    return True

def check_dataset_format(dataset_name: str = "phuhuy-se1/viGEC") -> bool:
    """Check if dataset has the expected format"""
    
    try:
        console.print("[yellow]Checking dataset format...[/yellow]")
        
        # Load a small sample
        dataset = load_dataset(dataset_name, split='train[:10]')
        
        # Check column names
        if len(dataset) == 0:
            console.print("[red]❌ Dataset is empty[/red]")
            return False
        
        first_item = dataset[0]
        expected_columns = ['incorrect_text', 'correct_text']
        fallback_columns = ['source', 'target']
        
        has_expected = all(col in first_item for col in expected_columns)
        has_fallback = all(col in first_item for col in fallback_columns)
        
        if has_expected:
            console.print("[green]✅ Dataset has expected columns: incorrect_text, correct_text[/green]")
            return True
        elif has_fallback:
            console.print("[yellow]⚠️ Dataset has fallback columns: source, target[/yellow]")
            return True
        else:
            console.print(f"[red]❌ Dataset missing expected columns. Available: {list(first_item.keys())}[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Error checking dataset: {e}[/red]")
        return False

if __name__ == "__main__":
    # Example usage
    console.print("[bold green]ViGEC Data Preparation Example[/bold green]")
    
    # Check system readiness
    if not can_train_base_model():
        console.print("[red]System not ready for training. Please check requirements.[/red]")
        exit(1)
    
    # Check dataset format
    if not check_dataset_format():
        console.print("[red]Dataset format check failed. Please verify dataset.[/red]")
        exit(1)
    
    # Load and process data
    data = load_vigec_dataset(test_subset_ratio=0.05)  # Use 5% of test set
    
    # Save processed data
    save_processed_data(data, "./data/processed")
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer("vinai/bartpho-syllable")
    
    # Create data loaders
    data_loaders = create_data_loaders(data, tokenizer, batch_size=8)
    
    console.print("[bold green]Data preparation completed![/bold green]")
