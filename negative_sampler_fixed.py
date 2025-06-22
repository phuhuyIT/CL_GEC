"""
Negative sample generator for contrastive learning
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from rich.console import Console
from rich.progress import track
from data_utils import load_processed_data, ViGECDataset, get_model_and_tokenizer
import random
from collections import defaultdict

console = Console()

class NegativeSampleGenerator:
    """Generate negative samples for contrastive learning"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = "auto"
    ):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() 
            else "cpu" if device == "auto" 
            else device
        )
        
        console.print(f"[yellow]Loading model from {model_path}[/yellow]")
        
        # Load model and tokenizer with proper setup (including ViT5 prefix)
        self.model, self.tokenizer = get_model_and_tokenizer(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        console.print(f"[green]Model loaded on {self.device}[/green]")
    
    def generate_beam_candidates(
        self,
        source_text: str,
        num_beams: int = 5,
        max_length: int = 384,
        temperature: float = 1.0
    ) -> List[str]:
        """Generate multiple candidates using beam search"""
        
        # Add task prefix for ViT5
        input_text = source_text
        if hasattr(self.tokenizer, 'task_prefix'):
            input_text = self.tokenizer.task_prefix + source_text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Generate with beam search
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=max_length,
                early_stopping=True,
                do_sample=False,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode candidates
        candidates = []
        for output in outputs:
            candidate = self.tokenizer.decode(
                output, skip_special_tokens=True
            ).strip()
            
            if candidate and candidate != source_text:
                candidates.append(candidate)
        
        return candidates
    
    def generate_sampling_candidates(
        self,
        source_text: str,
        num_samples: int = 5,
        max_length: int = 384,
        temperature: float = 1.2,
        top_p: float = 0.9
    ) -> List[str]:
        """Generate candidates using nucleus sampling"""
        
        # Add task prefix for ViT5
        input_text = source_text
        if hasattr(self.tokenizer, 'task_prefix'):
            input_text = self.tokenizer.task_prefix + source_text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        candidates = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            candidate = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            
            if candidate and candidate != source_text:
                candidates.append(candidate)
        
        return candidates
    
    def generate_rule_based_negatives(
        self,
        source_text: str,
        target_text: str
    ) -> List[str]:
        """Generate rule-based negative samples"""
        
        negatives = []
        
        # 1. Word order shuffling
        words = source_text.split()
        if len(words) > 2:
            random.shuffle(words)
            negatives.append(" ".join(words))
        
        # 2. Partial correction (apply some but not all corrections)
        source_words = source_text.split()
        target_words = target_text.split()
        
        if len(source_words) == len(target_words):
            # Apply random subset of corrections
            correction_indices = []
            for i, (s, t) in enumerate(zip(source_words, target_words)):
                if s != t:
                    correction_indices.append(i)
            
            if correction_indices:
                # Apply only half of the corrections
                partial_indices = random.sample(
                    correction_indices,
                    max(1, len(correction_indices) // 2)
                )
                
                partial_words = source_words.copy()
                for idx in partial_indices:
                    partial_words[idx] = target_words[idx]
                
                negatives.append(" ".join(partial_words))
        
        # 3. Over-correction (add unnecessary changes)
        # This is simple: just duplicate or modify some words
        words = target_text.split()
        if len(words) > 1:
            # Duplicate a random word
            idx = random.randint(0, len(words) - 1)
            words.insert(idx + 1, words[idx])
            negatives.append(" ".join(words))
        
        return negatives
    
    def create_negative_samples(
        self,
        data: List[Dict],
        num_negatives_per_sample: int = 3,
        methods: List[str] = ["beam", "sampling", "rule_based"],
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """Create negative samples for a dataset"""
        
        console.print(f"[yellow]Creating negative samples for {len(data)} examples[/yellow]")
        
        contrastive_data = []
        
        for item in track(data, description="Generating negatives"):
            source = item['source']
            target = item['target']
            
            negatives = []
            
            # Generate negatives using different methods
            if "beam" in methods:
                beam_negatives = self.generate_beam_candidates(
                    source, num_beams=5
                )
                # Filter out the target and take top candidates
                beam_negatives = [
                    neg for neg in beam_negatives 
                    if neg != target and neg != source
                ][:2]
                negatives.extend(beam_negatives)
            
            if "sampling" in methods:
                sampling_negatives = self.generate_sampling_candidates(
                    source, num_samples=3
                )
                # Filter out duplicates and target
                sampling_negatives = [
                    neg for neg in sampling_negatives 
                    if neg != target and neg != source and neg not in negatives
                ][:2]
                negatives.extend(sampling_negatives)
            
            if "rule_based" in methods:
                rule_negatives = self.generate_rule_based_negatives(
                    source, target
                )
                # Filter out duplicates
                rule_negatives = [
                    neg for neg in rule_negatives 
                    if neg not in negatives and neg != target and neg != source
                ][:1]
                negatives.extend(rule_negatives)
            
            # Ensure we have enough negatives
            if len(negatives) < num_negatives_per_sample:
                # Generate more using beam search
                additional = self.generate_beam_candidates(
                    source, num_beams=10
                )
                additional = [
                    neg for neg in additional 
                    if neg not in negatives and neg != target and neg != source
                ]
                negatives.extend(additional[:num_negatives_per_sample - len(negatives)])
            
            # Take only the required number
            negatives = negatives[:num_negatives_per_sample]
            
            if negatives:  # Only add if we have negatives
                contrastive_data.append({
                    'source': source,
                    'positive': target,  # gold target
                    'negatives': negatives,
                    'id': item.get('id', len(contrastive_data))
                })
        
        console.print(f"[green]Created {len(contrastive_data)} contrastive samples[/green]")
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(contrastive_data, f, ensure_ascii=False, indent=2)
            console.print(f"[green]Saved to {save_path}[/green]")
        
        return contrastive_data
    
    def analyze_negative_quality(
        self,
        contrastive_data: List[Dict],
        sample_size: int = 10
    ):
        """Analyze the quality of generated negative samples"""
        
        console.print(f"[yellow]Analyzing negative sample quality[/yellow]")
        
        # Sample random examples for analysis
        samples = random.sample(contrastive_data, min(sample_size, len(contrastive_data)))
        
        for i, item in enumerate(samples):
            console.print(f"\n[bold blue]Example {i+1}:[/bold blue]")
            console.print(f"[yellow]Source:[/yellow] {item['source']}")
            console.print(f"[green]Positive:[/green] {item['positive']}")
            
            for j, negative in enumerate(item['negatives']):
                console.print(f"[red]Negative {j+1}:[/red] {negative}")
        
        # Basic statistics
        avg_negatives = np.mean([len(item['negatives']) for item in contrastive_data])
        console.print(f"\n[blue]Average negatives per sample: {avg_negatives:.2f}[/blue]")

def main():
    """Test negative sample generation"""
    
    # Load some data
    data = load_processed_data("./data/processed")
    
    if 'train' not in data:
        console.print("[red]No training data found. Please run data preparation first.[/red]")
        return
    
    # Use small subset for testing
    test_data = data['train'][:100]
    
    # Initialize generator
    generator = NegativeSampleGenerator(
        model_path="vinai/bartpho-syllable",
        device="auto"
    )
    
    # Generate negative samples
    contrastive_data = generator.create_negative_samples(
        test_data,
        num_negatives_per_sample=3,
        methods=["beam", "sampling", "rule_based"],
        save_path="./data/contrastive/train_negatives.json"
    )
    
    # Analyze quality
    generator.analyze_negative_quality(contrastive_data, sample_size=5)

if __name__ == "__main__":
    main()
