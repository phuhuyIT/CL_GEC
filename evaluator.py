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
from nltk.translate.gleu_score import sentence_gleu
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    evaluate = None

from rich.console import Console

console = Console()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    # Download required NLTK data for GLEU
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
        """Calculate F0.5 score for a single sentence using consistent edit detection"""
        
        # Use consistent edit detection method
        precision, recall = self.calculate_precision_recall(source, prediction, target)
        
        if precision + recall == 0:
            return 0.0
        
        # F0.5 weights precision higher than recall
        # Formula: F_beta = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        
        return f05
    
    def calculate_precision_recall(self, source: str, prediction: str, target: str) -> Tuple[float, float]:
        """Calculate precision and recall using proper edit alignment"""
        
        # Tokenize sequences
        source_tokens = self.tokenize_sentence(source)
        pred_tokens = self.tokenize_sentence(prediction)
        target_tokens = self.tokenize_sentence(target)
        
        # Use a more sophisticated edit distance calculation
        pred_edits = self._get_sequence_edits(source_tokens, pred_tokens)
        target_edits = self._get_sequence_edits(source_tokens, target_tokens)
        
        # Count matches
        tp = len(pred_edits & target_edits)
        fp = len(pred_edits - target_edits)
        fn = len(target_edits - pred_edits)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        
        return precision, recall
    
    def _get_sequence_edits(self, source: List[str], target: List[str]) -> set:
        """Get edit operations between source and target sequences"""
        
        # Use a simple but more accurate edit detection
        # This is a simplified version of edit distance
        
        edits = set()
        
        # Convert to sets for token-level comparison, but preserve some position info
        source_multiset = {}
        target_multiset = {}
        
        # Count occurrences
        for i, token in enumerate(source):
            source_multiset[token] = source_multiset.get(token, 0) + 1
            
        for i, token in enumerate(target):
            target_multiset[token] = target_multiset.get(token, 0) + 1
        
        # Find additions (tokens in target but not in source, or more in target)
        for token, target_count in target_multiset.items():
            source_count = source_multiset.get(token, 0)
            if target_count > source_count:
                for i in range(target_count - source_count):
                    edits.add(f"ADD_{token}_{i}")
        
        # Find deletions (tokens in source but not in target, or fewer in target)
        for token, source_count in source_multiset.items():
            target_count = target_multiset.get(token, 0)
            if source_count > target_count:
                for i in range(source_count - target_count):
                    edits.add(f"DEL_{token}_{i}")
        
        return edits

class GECEvaluator:
    """Comprehensive evaluator for GEC systems"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.f05_evaluator = F05Evaluator(tokenizer)
        
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=False
            )
        else:
            self.rouge_scorer = None
            console.print("[yellow]⚠️  ROUGE not available. Install rouge-score: pip install rouge-score[/yellow]")
        
        # Initialize BLEU evaluator if available
        if EVALUATE_AVAILABLE:
            self.bleu_evaluator = evaluate.load("bleu")
        else:
            self.bleu_evaluator = None
        
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
            'gleu_scores': [],
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
                # GLEU score (GLEU is more suitable for GEC than BLEU)
                gleu = sentence_gleu(
                    [target_tokens],
                    pred_tokens,
                    min_len=1,
                    max_len=4
                )
            else:
                bleu = 0.0
                gleu = 0.0
            results['bleu_scores'].append(bleu)
            results['gleu_scores'].append(gleu)
            
            # ROUGE scores
            if ROUGE_AVAILABLE and self.rouge_scorer:
                rouge_scores = self.rouge_scorer.score(target, pred)
                results['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
                results['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
                results['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)
            else:
                results['rouge1_f'].append(0.0)
                results['rouge2_f'].append(0.0)
                results['rougeL_f'].append(0.0)
            
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
        """Calculate precision and recall for edits using consistent method"""
        
        # Use the same method as F0.5 evaluator for consistency
        return self.f05_evaluator.calculate_precision_recall(source, prediction, target)
    
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
    
    def calculate_all_metrics(
        self,
        sources: List[str],
        predictions: List[str],
        targets: List[str],
        print_results: bool = True
    ) -> Dict[str, float]:
        """Calculate all metrics in one go with nice formatting"""
        
        console.print("[bold blue]📊 Calculating Comprehensive Metrics...[/bold blue]")
        
        # Get main metrics
        results = self.evaluate_batch(sources, predictions, targets)
        
        # Calculate additional metrics
        ie_oe_results = self.calculate_ie_oe_ratio(sources, predictions, targets)
        results.update(ie_oe_results)
        
        if print_results:
            self._print_metrics_table(results, len(sources))
        
        return results
    
    def _print_metrics_table(self, results: Dict[str, float], num_samples: int):
        """Print metrics in a nice table format"""
        
        from rich.table import Table
        
        table = Table(title=f"📈 Evaluation Results ({num_samples} samples)")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta")
        table.add_column("Percentage", style="green")
        table.add_column("Description", style="dim")
        
        # Core metrics
        if 'f05' in results:
            table.add_row(
                "F0.5", 
                f"{results['f05']:.4f}", 
                f"{results['f05']*100:.1f}%",
                "Edit-level F0.5 (precision-weighted)"
            )
        
        if 'precision' in results:
            table.add_row(
                "Precision", 
                f"{results['precision']:.4f}", 
                f"{results['precision']*100:.1f}%",
                "Edit precision"
            )
        
        if 'recall' in results:
            table.add_row(
                "Recall", 
                f"{results['recall']:.4f}", 
                f"{results['recall']*100:.1f}%",
                "Edit recall"
            )
        
        # Translation metrics
        if 'bleu' in results:
            table.add_row(
                "BLEU", 
                f"{results['bleu']:.4f}", 
                f"{results['bleu']*100:.1f}%",
                "BLEU score (n-gram overlap)"
            )
        
        if 'gleu' in results:
            table.add_row(
                "GLEU", 
                f"{results['gleu']:.4f}", 
                f"{results['gleu']*100:.1f}%",
                "GLEU score (better for GEC)"
            )
        
        # ROUGE metrics
        if 'rouge1_f' in results:
            table.add_row(
                "ROUGE-1", 
                f"{results['rouge1_f']:.4f}", 
                f"{results['rouge1_f']*100:.1f}%",
                "Unigram overlap"
            )
        
        if 'rouge2_f' in results:
            table.add_row(
                "ROUGE-2", 
                f"{results['rouge2_f']:.4f}", 
                f"{results['rouge2_f']*100:.1f}%",
                "Bigram overlap"
            )
        
        if 'rougeL_f' in results:
            table.add_row(
                "ROUGE-L", 
                f"{results['rougeL_f']:.4f}", 
                f"{results['rougeL_f']*100:.1f}%",
                "Longest common subsequence"
            )
        
        # Additional metrics
        if 'ie_ratio' in results:
            table.add_row(
                "IE Ratio", 
                f"{results['ie_ratio']:.4f}", 
                f"{results['ie_ratio']*100:.1f}%",
                "Input-preserving edit ratio"
            )
        
        console.print(table)
        
        # Summary
        if 'f05' in results:
            f05_score = results['f05']
            if f05_score >= 0.8:
                console.print(f"[bold green]🎉 Excellent performance! F0.5 = {f05_score:.3f}[/bold green]")
            elif f05_score >= 0.6:
                console.print(f"[green]👍 Good performance! F0.5 = {f05_score:.3f}[/green]")
            elif f05_score >= 0.4:
                console.print(f"[yellow]📊 Moderate performance. F0.5 = {f05_score:.3f}[/yellow]")
            else:
                console.print(f"[red]📉 Low performance. F0.5 = {f05_score:.3f}[/red]")
                console.print("[blue]💡 Consider checking model training, data preprocessing, or task prefix[/blue]")

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
