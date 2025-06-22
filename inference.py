"""
Inference module with Contrastive Search for Vietnamese GEC
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import numpy as np
from rich.console import Console
from rich.progress import track
import json

console = Console()

class ContrastiveSearchDecoder:
    """Contrastive Search decoder implementation"""
    
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        alpha: float = 0.7,
        k: int = 5,
        device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha  # Balance between model confidence and contrastive penalty
        self.k = k  # Top-k candidates
        
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() 
            else "cpu" if device == "auto" 
            else device
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def contrastive_search_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generated_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[int, Tuple]:
        """Single step of contrastive search"""
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values
            
            # Get top-k tokens
            top_k_probs, top_k_indices = torch.topk(
                F.softmax(logits, dim=-1), 
                k=self.k, 
                dim=-1
            )
            
            # Calculate contrastive scores
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                # Calculate similarity penalty
                contrastive_scores = self._calculate_contrastive_penalty(
                    top_k_indices,
                    generated_tokens,
                    past_key_values
                )
                
                # Combine model confidence and contrastive penalty
                final_scores = self.alpha * top_k_probs + (1 - self.alpha) * contrastive_scores
            else:
                # First token: use only model confidence
                final_scores = top_k_probs
            
            # Select best token
            best_idx = torch.argmax(final_scores, dim=-1)
            next_token = top_k_indices.gather(-1, best_idx.unsqueeze(-1))
            
            return next_token.item(), past_key_values
    
    def _calculate_contrastive_penalty(
        self,
        candidate_tokens: torch.Tensor,
        generated_tokens: torch.Tensor,
        past_key_values: Tuple
    ) -> torch.Tensor:
        """Calculate contrastive penalty to encourage diversity"""
        
        batch_size, k = candidate_tokens.shape
        seq_len = generated_tokens.size(1)
        
        penalties = torch.zeros_like(candidate_tokens, dtype=torch.float)
        
        for i in range(k):
            candidate = candidate_tokens[:, i]
            
            # Create hypothetical sequence with this candidate
            candidate_seq = torch.cat([
                generated_tokens, 
                candidate.unsqueeze(-1)
            ], dim=1)
            
            # Calculate similarity with previous tokens
            if seq_len > 0:
                # Simple repetition penalty
                repetition_penalty = torch.sum(
                    generated_tokens == candidate.unsqueeze(-1), 
                    dim=1
                ).float()
                
                # Normalize by sequence length
                repetition_penalty = repetition_penalty / seq_len
                
                # Convert to penalty (higher repetition = lower score)
                penalties[:, i] = 1.0 - repetition_penalty
            else:
                penalties[:, i] = 1.0
        return penalties
    
    def generate(
        self,
        input_text: str,
        max_length: int = 384,
        min_length: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> str:
        """Generate text using contrastive search"""
        
        # Add task prefix for ViT5
        if hasattr(self.tokenizer, 'task_prefix'):
            input_text = self.tokenizer.task_prefix + input_text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Initialize generation
        batch_size = input_ids.size(0)
        generated_tokens = torch.empty(
            (batch_size, 0), 
            dtype=torch.long, 
            device=self.device
        )
        
        past_key_values = None
        current_input = input_ids
        current_attention_mask = attention_mask
        
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Generate tokens one by one
        for step in range(max_length):
            # Contrastive search step
            next_token, past_key_values = self.contrastive_search_step(
                input_ids=current_input,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                generated_tokens=generated_tokens
            )
            
            # Add generated token
            next_token_tensor = torch.tensor(
                [[next_token]], 
                device=self.device
            )
            generated_tokens = torch.cat([generated_tokens, next_token_tensor], dim=1)
            
            # Update input for next step
            current_input = next_token_tensor
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, 1), device=self.device)
            ], dim=1)
            
            # Check for early stopping
            if next_token == eos_token_id and generated_tokens.size(1) >= min_length:
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_tokens[0], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def batch_generate(
        self,
        input_texts: List[str],
        max_length: int = 384,
        batch_size: int = 8
    ) -> List[str]:
        """Generate text for multiple inputs"""
        
        results = []
        
        for i in track(range(0, len(input_texts), batch_size), description="Generating"):
            batch_texts = input_texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                generated = self.generate(text, max_length=max_length)
                batch_results.append(generated)
            
            results.extend(batch_results)
        
        return results

class GECInference:
    """Main inference class for Vietnamese GEC"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        use_contrastive_search: bool = True,
        contrastive_alpha: float = 0.7,
        contrastive_k: int = 5,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.use_contrastive_search = use_contrastive_search
        
        # Load model and tokenizer
        console.print(f"[yellow]Loading model from {model_path}[/yellow]")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path
        )
        
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() 
            else "cpu" if device == "auto" 
            else device
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize contrastive search decoder
        if use_contrastive_search:
            self.contrastive_decoder = ContrastiveSearchDecoder(
                model=self.model,
                tokenizer=self.tokenizer,
                alpha=contrastive_alpha,
                k=contrastive_k,
                device=self.device
            )
        
        console.print(f"[green]Model loaded on {self.device}[/green]")
        console.print(f"[blue]Using {'Contrastive Search' if use_contrastive_search else 'Beam Search'}[/blue]")
    def correct_text(
        self,
        text: str,
        max_length: int = 384,
        num_beams: int = 5,
        early_stopping: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """Correct a single text"""
        
        if self.use_contrastive_search:
            # Use contrastive search
            corrected = self.contrastive_decoder.generate(
                text, max_length=max_length
            )
        else:
            # Add task prefix for ViT5 in beam search
            input_text = text
            if hasattr(self.tokenizer, 'task_prefix'):
                input_text = self.tokenizer.task_prefix + text
            
            # Use standard beam search
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            corrected = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
        
        return corrected.strip()
    
    def correct_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """Correct multiple texts"""
        
        if self.use_contrastive_search:
            return self.contrastive_decoder.batch_generate(
                texts, 
                batch_size=batch_size,
                max_length=kwargs.get('max_length', 384)
            )
        else:
            results = []
            for i in track(range(0, len(texts), batch_size), description="Correcting"):
                batch_texts = texts[i:i + batch_size]
                
                # Add task prefix for ViT5
                if hasattr(self.tokenizer, 'task_prefix'):
                    batch_texts = [self.tokenizer.task_prefix + text for text in batch_texts]
                
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=kwargs.get('max_length', 384),
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=kwargs.get('max_length', 384),
                        num_beams=kwargs.get('num_beams', 5),
                        early_stopping=kwargs.get('early_stopping', True),
                        do_sample=kwargs.get('do_sample', False),
                        temperature=kwargs.get('temperature', 1.0),
                        top_p=kwargs.get('top_p', 0.9),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                batch_results = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                
                results.extend([text.strip() for text in batch_results])
            
            return results
    
    def correct_file(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 8,
        **kwargs
    ):
        """Correct texts from file"""
        
        console.print(f"[yellow]Correcting texts from {input_path}[/yellow]")
        
        # Read input file
        if input_path.endswith('.json'):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and isinstance(data[0], dict):
                # Assume format: [{"source": "...", ...}, ...]
                texts = [item['source'] for item in data]
            else:
                # Assume format: ["text1", "text2", ...]
                texts = data
        else:
            # Text file
            with open(input_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        # Correct texts
        corrected_texts = self.correct_batch(texts, batch_size=batch_size, **kwargs)
        
        # Save results
        if output_path.endswith('.json'):
            results = []
            for i, (original, corrected) in enumerate(zip(texts, corrected_texts)):
                results.append({
                    'id': i,
                    'source': original,
                    'corrected': corrected
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for corrected in corrected_texts:
                    f.write(corrected + '\n')
        
        console.print(f"[green]Results saved to {output_path}[/green]")
        console.print(f"[green]Corrected {len(texts)} texts[/green]")
    
    def interactive_correction(self):
        """Interactive text correction"""
        
        console.print("[bold green]Interactive Vietnamese GEC[/bold green]")
        console.print("Enter text to correct (type 'quit' to exit):")
        
        while True:
            try:
                text = input("\n> ").strip()
                
                if text.lower() == 'quit':
                    break
                
                if not text:
                    continue
                
                corrected = self.correct_text(text)
                
                console.print(f"[yellow]Original:[/yellow] {text}")
                console.print(f"[green]Corrected:[/green] {corrected}")
                
            except KeyboardInterrupt:
                break
        
        console.print("\n[blue]Goodbye![/blue]")

def main():
    """Main function for inference"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese GEC Inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--input_file", help="Input file to correct")
    parser.add_argument("--output_file", help="Output file for corrections")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--use_contrastive", action="store_true", default=True, help="Use contrastive search")
    parser.add_argument("--alpha", type=float, default=0.7, help="Contrastive search alpha")
    parser.add_argument("--k", type=int, default=5, help="Contrastive search k")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length")
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = GECInference(
        model_path=args.model_path,
        use_contrastive_search=args.use_contrastive,
        contrastive_alpha=args.alpha,
        contrastive_k=args.k
    )
    
    if args.interactive:
        # Interactive mode
        inference.interactive_correction()
    
    elif args.input_file and args.output_file:
        # File correction mode
        inference.correct_file(
            input_path=args.input_file,
            output_path=args.output_file,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
    
    else:
        # Single text correction
        text = input("Enter text to correct: ").strip()
        corrected = inference.correct_text(text, max_length=args.max_length)
        
        console.print(f"[yellow]Original:[/yellow] {text}")
        console.print(f"[green]Corrected:[/green] {corrected}")

if __name__ == "__main__":
    main()
