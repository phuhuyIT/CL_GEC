"""
Evaluator for Vietnamese GEC metrics
"""

import re
import string
from typing import List, Tuple, Dict, Any
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
from rich.console import Console

console = Console()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class F05Evaluator:
    """F0.5 evaluator for GEC following standard practice"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.smoothing = SmoothingFunction()
        
    def tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenize sentence for evaluation"""
        # Remove punctuation and convert to lowercase
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
        # Split by whitespace
        tokens = sentence.split()
        
        return tokens
    
    def calculate_f05(self, source: str, prediction: str, target: str) -> float:
        """Calculate F0.5 score for a single sentence"""
        
        # Tokenize
        source_tokens = set(self.tokenize_sentence(source))
        pred_tokens = set(self.tokenize_sentence(prediction))
        target_tokens = set(self.tokenize_sentence(target))
        
        # Calculate edits
        source_to_pred = pred_tokens - source_tokens  # Additions in prediction
        source_to_target = target_tokens - source_tokens  # Additions in target (gold)
        
        pred_to_source = source_tokens - pred_tokens  # Deletions in prediction
        target_to_source = source_tokens - target_tokens  # Deletions in target (gold)
        
        # True positives: correct additions and deletions
        tp_additions = len(source_to_pred & source_to_target)
        tp_deletions = len(pred_to_source & target_to_source)
        tp = tp_additions + tp_deletions
        
        # False positives: incorrect additions and deletions
        fp_additions = len(source_to_pred - source_to_target)
        fp_deletions = len(pred_to_source - target_to_source)
        fp = fp_additions + fp_deletions
        
        # False negatives: missed additions and deletions
        fn_additions = len(source_to_target - source_to_pred)
        fn_deletions = len(target_to_source - pred_to_source)
        fn = fn_additions + fn_deletions
        
        # Calculate F0.5
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
            
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
            
        if precision + recall == 0:
            f05 = 0.0
        else:
            # F0.5 weights precision higher than recall
            beta = 0.5
            f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            
        return f05

class GECEvaluator:
    """Comprehensive evaluator for GEC systems"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.f05_evaluator = F05Evaluator(tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=False
        )
        self.bleu_evaluator = evaluate.load("bleu")
        
    def evaluate_batch(
        self, 
        sources: List[str], 
        predictions: List[str], 
        targets: List[str]
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions"""
        
        assert len(sources) == len(predictions) == len(targets)
        
        results = {
            'f05_scores': [],
            'bleu_scores': [],
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        for source, pred, target in zip(sources, predictions, targets):
            # F0.5 score
            f05 = self.f05_evaluator.calculate_f05(source, pred, target)
            results['f05_scores'].append(f05)
            
            # BLEU score
            target_tokens = target.split()
            pred_tokens = pred.split()
            
            if len(pred_tokens) > 0 and len(target_tokens) > 0:
                bleu = sentence_bleu(
                    [target_tokens], 
                    pred_tokens,
                    smoothing_function=self.f05_evaluator.smoothing.method1
                )
            else:
                bleu = 0.0
            results['bleu_scores'].append(bleu)
            
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(target, pred)
            results['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
            results['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
            results['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)
            
            # Precision and Recall for edits
            precision, recall = self._calculate_edit_precision_recall(source, pred, target)
            results['precision_scores'].append(precision)
            results['recall_scores'].append(recall)
        
        # Aggregate results
        aggregated = {}
        for metric, scores in results.items():
            aggregated[metric.replace('_scores', '')] = np.mean(scores)
            aggregated[f"{metric.replace('_scores', '')}_std"] = np.std(scores)
        
        return aggregated
    
    def _calculate_edit_precision_recall(
        self, 
        source: str, 
        prediction: str, 
        target: str
    ) -> Tuple[float, float]:
        """Calculate precision and recall for edits"""
        
        # Tokenize
        source_tokens = self.f05_evaluator.tokenize_sentence(source)
        pred_tokens = self.f05_evaluator.tokenize_sentence(prediction)
        target_tokens = self.f05_evaluator.tokenize_sentence(target)
        
        # Calculate edits
        pred_edits = self._get_edits(source_tokens, pred_tokens)
        target_edits = self._get_edits(source_tokens, target_tokens)
        
        # True positives
        tp = len(pred_edits & target_edits)
        
        # Precision
        if len(pred_edits) == 0:
            precision = 1.0 if len(target_edits) == 0 else 0.0
        else:
            precision = tp / len(pred_edits)
        
        # Recall
        if len(target_edits) == 0:
            recall = 1.0
        else:
            recall = tp / len(target_edits)
        
        return precision, recall
    
    def _get_edits(self, source: List[str], target: List[str]) -> set:
        """Get edit operations between source and target"""
        
        # Simple edit detection based on token differences
        source_set = set(enumerate(source))
        target_set = set(enumerate(target))
        
        # Additions and deletions
        additions = target_set - source_set
        deletions = source_set - target_set
        
        edits = set()
        for pos, token in additions:
            edits.add(f"ADD_{pos}_{token}")
        for pos, token in deletions:
            edits.add(f"DEL_{pos}_{token}")
        
        return edits
    
    def calculate_ie_oe_ratio(
        self,
        sources: List[str],
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Calculate Input-preserving Edit (IE) vs Over-correction Edit (OE) ratio"""
        
        ie_count = 0  # Correct edits
        oe_count = 0  # Over-corrections
        total_edits = 0
        
        for source, pred, target in zip(sources, predictions, targets):
            source_tokens = set(self.f05_evaluator.tokenize_sentence(source))
            pred_tokens = set(self.f05_evaluator.tokenize_sentence(pred))
            target_tokens = set(self.f05_evaluator.tokenize_sentence(target))
            
            # Edits made by prediction
            pred_additions = pred_tokens - source_tokens
            pred_deletions = source_tokens - pred_tokens
            
            # Edits that should be made (target)
            target_additions = target_tokens - source_tokens
            target_deletions = source_tokens - target_tokens
            
            # Input-preserving edits (correct edits)
            ie_additions = pred_additions & target_additions
            ie_deletions = pred_deletions & target_deletions
            ie_count += len(ie_additions) + len(ie_deletions)
            
            # Over-correction edits (incorrect edits)
            oe_additions = pred_additions - target_additions
            oe_deletions = pred_deletions - target_deletions
            oe_count += len(oe_additions) + len(oe_deletions)
            
            total_edits += len(pred_additions) + len(pred_deletions)
        
        if total_edits == 0:
            ie_ratio = 0.0
            oe_ratio = 0.0
        else:
            ie_ratio = ie_count / total_edits
            oe_ratio = oe_count / total_edits
        
        return {
            'ie_ratio': ie_ratio,
            'oe_ratio': oe_ratio,
            'ie_count': ie_count,
            'oe_count': oe_count,
            'total_edits': total_edits
        }
    
    def generate_report(
        self,
        sources: List[str],
        predictions: List[str], 
        targets: List[str],
        output_path: str = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        console.print("[yellow]Generating evaluation report...[/yellow]")
        
        # Main metrics
        main_results = self.evaluate_batch(sources, predictions, targets)
        
        # IE/OE analysis
        ie_oe_results = self.calculate_ie_oe_ratio(sources, predictions, targets)
        
        # Combine results
        report = {
            'summary': {
                'num_samples': len(sources),
                'avg_f05': main_results['f05'],
                'avg_bleu': main_results['bleu'],
                'avg_precision': main_results['precision'],
                'avg_recall': main_results['recall'],
                'ie_oe_ratio': ie_oe_results['ie_ratio'] / max(ie_oe_results['oe_ratio'], 1e-8)
            },
            'detailed_metrics': main_results,
            'ie_oe_analysis': ie_oe_results,
            'examples': []
        }
        
        # Add examples
        for i in range(min(10, len(sources))):
            f05 = self.f05_evaluator.calculate_f05(sources[i], predictions[i], targets[i])
            
            report['examples'].append({
                'index': i,
                'source': sources[i],
                'prediction': predictions[i],
                'target': targets[i],
                'f05': f05
            })
        
        # Print summary
        console.print(f"[green]Evaluation Summary:[/green]")
        console.print(f"  Samples: {report['summary']['num_samples']}")
        console.print(f"  F0.5: {report['summary']['avg_f05']:.4f}")
        console.print(f"  BLEU: {report['summary']['avg_bleu']:.4f}")
        console.print(f"  Precision: {report['summary']['avg_precision']:.4f}")
        console.print(f"  Recall: {report['summary']['avg_recall']:.4f}")
        console.print(f"  IE/OE Ratio: {report['summary']['ie_oe_ratio']:.4f}")
        
        # Save report
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Report saved to {output_path}[/green]")
        
        return report

if __name__ == "__main__":
    # Example usage
    evaluator = GECEvaluator()
    
    # Test data
    sources = [
        "Tôi đi học trường đại học.",
        "Hôm nay tôi không đi làm.",
        "Cô ấy rất đẹp và thông minh."
    ]
    
    predictions = [
        "Tôi đi học ở trường đại học.",
        "Hôm nay tôi không đi làm.",
        "Cô ấy rất đẹp và thông minh."
    ]
    
    targets = [
        "Tôi đi học ở trường đại học.",
        "Hôm nay tôi không đi làm việc.",
        "Cô ấy rất đẹp và thông minh."
    ]
    
    # Evaluate
    report = evaluator.generate_report(sources, predictions, targets)
    
    console.print("[bold green]Evaluation completed![/bold green]")
