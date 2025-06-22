"""
Main evaluation script for Vietnamese GEC models
"""

import os
import json
import torch
from typing import List, Dict, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_processed_data
from evaluator import GECEvaluator
from inference import GECInference
import argparse

console = Console()

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "./data/processed",
        output_dir: str = "./evaluation_results"
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.evaluator = GECEvaluator()
        
        # Load inference engines for comparison
        self.inference_engines = {
            'beam_search': GECInference(
                model_path=model_path,
                use_contrastive_search=False
            ),
            'contrastive_search': GECInference(
                model_path=model_path,
                use_contrastive_search=True,
                contrastive_alpha=0.7,
                contrastive_k=5
            )
        }
        
        console.print(f"[green]Model evaluator initialized for {model_path}[/green]")
    
    def evaluate_on_test_set(
        self,
        max_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> Dict[str, Dict]:
        """Evaluate model on test set with different decoding strategies"""
        
        console.print("[yellow]Loading test data...[/yellow]")
        
        # Load test data
        data = load_processed_data(self.data_dir)
        test_data = data.get('test', data.get('validation', []))
        
        if not test_data:
            console.print("[red]No test data found![/red]")
            return {}
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        console.print(f"[blue]Evaluating on {len(test_data)} samples[/blue]")
        
        # Extract texts
        sources = [item['source'] for item in test_data]
        targets = [item['target'] for item in test_data]
        
        results = {}
        
        # Evaluate different decoding strategies
        for strategy_name, inference_engine in self.inference_engines.items():
            console.print(f"\n[yellow]Evaluating {strategy_name}...[/yellow]")
            
            # Generate predictions
            predictions = inference_engine.correct_batch(
                sources, 
                batch_size=batch_size
            )
            
            # Evaluate predictions
            eval_results = self.evaluator.evaluate_batch(
                sources, predictions, targets
            )
            
            # Calculate IE/OE ratios
            ie_oe_results = self.evaluator.calculate_ie_oe_ratio(
                sources, predictions, targets
            )
            
            # Combine results
            strategy_results = {
                **eval_results,
                **ie_oe_results,
                'predictions': predictions[:10],  # Save first 10 predictions
                'strategy': strategy_name
            }
            
            results[strategy_name] = strategy_results
            
            # Print summary
            self._print_strategy_summary(strategy_name, strategy_results)
        
        # Save detailed results
        self._save_evaluation_results(results, sources[:10], targets[:10])
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        return results
    
    def _print_strategy_summary(self, strategy_name: str, results: Dict):
        """Print summary for a strategy"""
        
        table = Table(title=f"{strategy_name.replace('_', ' ').title()} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Main metrics
        table.add_row("F0.5", f"{results['f05']:.4f}")
        table.add_row("BLEU", f"{results['bleu']:.4f}")
        table.add_row("ROUGE-1", f"{results['rouge1_f']:.4f}")
        table.add_row("ROUGE-L", f"{results['rougeL_f']:.4f}")
        table.add_row("Precision", f"{results['precision']:.4f}")
        table.add_row("Recall", f"{results['recall']:.4f}")
        table.add_row("IE Ratio", f"{results['ie_ratio']:.4f}")
        table.add_row("OE Ratio", f"{results['oe_ratio']:.4f}")
        table.add_row("IE/OE Ratio", f"{results['ie_ratio'] / max(results['oe_ratio'], 1e-8):.4f}")
        
        console.print(table)
    
    def _save_evaluation_results(
        self, 
        results: Dict[str, Dict], 
        sources: List[str], 
        targets: List[str]
    ):
        """Save detailed evaluation results"""
        
        # Create detailed results with examples
        detailed_results = {
            'evaluation_summary': {},
            'strategy_comparison': {},
            'examples': []
        }
        
        # Summary for each strategy
        for strategy, strategy_results in results.items():
            summary = {
                'f05': strategy_results['f05'],
                'bleu': strategy_results['bleu'],
                'rouge1_f': strategy_results['rouge1_f'],
                'rougeL_f': strategy_results['rougeL_f'],
                'precision': strategy_results['precision'],
                'recall': strategy_results['recall'],
                'ie_ratio': strategy_results['ie_ratio'],
                'oe_ratio': strategy_results['oe_ratio']
            }
            detailed_results['evaluation_summary'][strategy] = summary
        
        # Strategy comparison
        if len(results) > 1:
            strategies = list(results.keys())
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    comparison_key = f"{strategy1}_vs_{strategy2}"
                    
                    comparison = {
                        'f05_diff': results[strategy1]['f05'] - results[strategy2]['f05'],
                        'bleu_diff': results[strategy1]['bleu'] - results[strategy2]['bleu'],
                        'ie_oe_ratio_diff': (
                            results[strategy1]['ie_ratio'] / max(results[strategy1]['oe_ratio'], 1e-8) -
                            results[strategy2]['ie_ratio'] / max(results[strategy2]['oe_ratio'], 1e-8)
                        )
                    }
                    
                    detailed_results['strategy_comparison'][comparison_key] = comparison
        
        # Examples
        num_examples = min(10, len(sources))
        for i in range(num_examples):
            example = {
                'index': i,
                'source': sources[i],
                'target': targets[i]
            }
            
            for strategy, strategy_results in results.items():
                example[f'{strategy}_prediction'] = strategy_results['predictions'][i]
                
                # Calculate individual F0.5 for this example
                f05 = self.evaluator.f05_evaluator.calculate_f05(
                    sources[i], 
                    strategy_results['predictions'][i], 
                    targets[i]
                )
                example[f'{strategy}_f05'] = f05
            
            detailed_results['examples'].append(example)
        
        # Save to file
        output_path = os.path.join(self.output_dir, "detailed_evaluation_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Detailed results saved to {output_path}[/green]")
    
    def _generate_comparison_report(self, results: Dict[str, Dict]):
        """Generate comparison report and visualizations"""
        
        # Create comparison dataframe
        comparison_data = []
        
        for strategy, strategy_results in results.items():
            comparison_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'F0.5': strategy_results['f05'],
                'BLEU': strategy_results['bleu'],
                'ROUGE-1': strategy_results['rouge1_f'],
                'ROUGE-L': strategy_results['rougeL_f'],
                'Precision': strategy_results['precision'],
                'Recall': strategy_results['recall'],
                'IE Ratio': strategy_results['ie_ratio'],
                'OE Ratio': strategy_results['oe_ratio'],
                'IE/OE Ratio': strategy_results['ie_ratio'] / max(strategy_results['oe_ratio'], 1e-8)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        csv_path = os.path.join(self.output_dir, "strategy_comparison.csv")
        df.to_csv(csv_path, index=False)
        console.print(f"[green]Comparison table saved to {csv_path}[/green]")
        
        # Create visualizations
        self._create_visualizations(df)
        
        # Print best strategy
        best_f05_strategy = df.loc[df['F0.5'].idxmax(), 'Strategy']
        best_ie_oe_strategy = df.loc[df['IE/OE Ratio'].idxmax(), 'Strategy']
        
        console.print(f"\n[bold green]Best F0.5 Strategy: {best_f05_strategy}[/bold green]")
        console.print(f"[bold blue]Best IE/OE Ratio Strategy: {best_ie_oe_strategy}[/bold blue]")
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create evaluation visualizations"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GEC Model Evaluation Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Main metrics comparison
        main_metrics = ['F0.5', 'BLEU', 'Precision', 'Recall']
        df_main = df[['Strategy'] + main_metrics]
        df_main_melted = df_main.melt(id_vars=['Strategy'], var_name='Metric', value_name='Score')
        
        sns.barplot(data=df_main_melted, x='Metric', y='Score', hue='Strategy', ax=axes[0, 0])
        axes[0, 0].set_title('Main Evaluation Metrics')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: ROUGE scores
        rouge_metrics = ['ROUGE-1', 'ROUGE-L']
        df_rouge = df[['Strategy'] + rouge_metrics]
        df_rouge_melted = df_rouge.melt(id_vars=['Strategy'], var_name='Metric', value_name='Score')
        
        sns.barplot(data=df_rouge_melted, x='Metric', y='Score', hue='Strategy', ax=axes[0, 1])
        axes[0, 1].set_title('ROUGE Scores')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: IE/OE Analysis
        ie_oe_metrics = ['IE Ratio', 'OE Ratio']
        df_ie_oe = df[['Strategy'] + ie_oe_metrics]
        df_ie_oe_melted = df_ie_oe.melt(id_vars=['Strategy'], var_name='Metric', value_name='Ratio')
        
        sns.barplot(data=df_ie_oe_melted, x='Metric', y='Ratio', hue='Strategy', ax=axes[1, 0])
        axes[1, 0].set_title('Input-preserving vs Over-correction Analysis')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: IE/OE Ratio
        sns.barplot(data=df, x='Strategy', y='IE/OE Ratio', ax=axes[1, 1])
        axes[1, 1].set_title('IE/OE Ratio (Higher is Better)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "evaluation_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Visualization saved to {plot_path}[/green]")
    
    def evaluate_error_types(
        self,
        max_samples: Optional[int] = None
    ) -> Dict:
        """Analyze performance on different error types"""
        
        console.print("[yellow]Analyzing error types...[/yellow]")
        
        # Load test data
        data = load_processed_data(self.data_dir)
        test_data = data.get('test', data.get('validation', []))
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        # Categorize errors (simple heuristics)
        error_categories = {
            'no_change': [],  # source == target
            'minimal_change': [],  # 1-2 token differences
            'moderate_change': [],  # 3-5 token differences
            'major_change': []  # >5 token differences
        }
        
        for item in test_data:
            source_tokens = item['source'].split()
            target_tokens = item['target'].split()
            
            if source_tokens == target_tokens:
                error_categories['no_change'].append(item)
            else:
                # Simple edit distance approximation
                diff_count = abs(len(source_tokens) - len(target_tokens))
                common_tokens = set(source_tokens) & set(target_tokens)
                diff_count += len(set(source_tokens) | set(target_tokens)) - len(common_tokens)
                
                if diff_count <= 2:
                    error_categories['minimal_change'].append(item)
                elif diff_count <= 5:
                    error_categories['moderate_change'].append(item)
                else:
                    error_categories['major_change'].append(item)
        
        # Evaluate each category
        category_results = {}
        
        for category, items in error_categories.items():
            if not items:
                continue
            
            console.print(f"[blue]Evaluating {category}: {len(items)} samples[/blue]")
            
            sources = [item['source'] for item in items]
            targets = [item['target'] for item in items]
            
            # Use best performing strategy (contrastive search)
            predictions = self.inference_engines['contrastive_search'].correct_batch(sources)
            
            # Evaluate
            results = self.evaluator.evaluate_batch(sources, predictions, targets)
            ie_oe_results = self.evaluator.calculate_ie_oe_ratio(sources, predictions, targets)
            
            category_results[category] = {
                **results,
                **ie_oe_results,
                'sample_count': len(items)
            }
        
        # Save error analysis
        error_analysis_path = os.path.join(self.output_dir, "error_type_analysis.json")
        with open(error_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(category_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Error type analysis saved to {error_analysis_path}[/green]")
        
        # Print summary
        table = Table(title="Performance by Error Type")
        table.add_column("Error Type", style="cyan")
        table.add_column("Samples", style="yellow")
        table.add_column("F0.5", style="magenta")
        table.add_column("IE/OE Ratio", style="green")
        
        for category, results in category_results.items():
            ie_oe_ratio = results['ie_ratio'] / max(results['oe_ratio'], 1e-8)
            table.add_row(
                category.replace('_', ' ').title(),
                str(results['sample_count']),
                f"{results['f05']:.4f}",
                f"{ie_oe_ratio:.4f}"
            )
        
        console.print(table)
        
        return category_results

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese GEC Model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_dir", default="./data/processed", help="Data directory")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--error_analysis", action="store_true", help="Perform error type analysis")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    console.print("[bold green]Starting model evaluation...[/bold green]")
    
    # Main evaluation
    results = evaluator.evaluate_on_test_set(
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Error type analysis
    if args.error_analysis:
        error_results = evaluator.evaluate_error_types(
            max_samples=args.max_samples
        )
    
    console.print("[bold green]Evaluation completed![/bold green]")

if __name__ == "__main__":
    main()
