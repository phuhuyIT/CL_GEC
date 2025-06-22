"""
Main script to run the complete Vietnamese GEC training pipeline
"""

import argparse
import sys
import os
from rich.console import Console
from rich.progress import Progress
from config import get_config, print_config
from data_utils import (
    can_train_base_model, 
    check_dataset_format,
    load_vigec_dataset, 
    save_processed_data
)

console = Console()

def run_data_preparation(config):
    """Step 1: Data preparation"""
    console.print("\n[bold blue]📊 Step 1: Data Preparation[/bold blue]")
    
    # Check system readiness
    if not can_train_base_model():
        console.print("[red]❌ System not ready for training![/red]")
        return False
    
    # Check dataset format
    if not check_dataset_format(config.data.dataset_name):
        console.print("[red]❌ Dataset format check failed![/red]")
        return False
    
    # Load and process data
    console.print("[yellow]Loading and processing data...[/yellow]")
    data = load_vigec_dataset(
        dataset_name=config.data.dataset_name,
        cache_dir=config.data.cache_dir,
        test_subset_ratio=config.data.test_subset_ratio
    )
    
    # Save processed data
    save_processed_data(data, config.data_dir)
    
    console.print("[green]✅ Data preparation completed![/green]")
    return True

def run_base_training(config):
    """Step 2: Base model training"""
    console.print("\n[bold blue]🤖 Step 2: Base Model Training[/bold blue]")
    
    try:
        from base_trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model_name=config.model.model_name,
            data_dir=config.data_dir,
            output_dir=config.base_model_dir,
            hyperopt=config.hyperopt.enabled
        )
        
        trainer.train()
        
        console.print("[green]✅ Base model training completed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Base training failed: {e}[/red]")
        return False

def run_negative_sampling(config):
    """Step 3: Negative sample generation"""
    console.print("\n[bold blue]🎭 Step 3: Negative Sample Generation[/bold blue]")
    
    try:
        import subprocess
        
        base_model_path = os.path.join(config.base_model_dir, "final")
        
        if not os.path.exists(base_model_path):
            console.print(f"[red]❌ Base model not found at {base_model_path}[/red]")
            return False
        
        # Run negative sampler
        cmd = [
            "python", "negative_sampler.py",
            "--model_path", base_model_path,
            "--data_dir", config.data_dir,
            "--output_dir", config.contrastive_data_dir,
            "--batch_size", str(config.data.batch_size // 2)  # Smaller batch for generation
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✅ Negative sample generation completed![/green]")
            return True
        else:
            console.print(f"[red]❌ Negative sampling failed: {result.stderr}[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Negative sampling error: {e}[/red]")
        return False

def run_contrastive_training(config):
    """Step 4: Contrastive learning training"""
    console.print("\n[bold blue]🔄 Step 4: Contrastive Learning Training[/bold blue]")
    
    try:
        from contrastive_trainer import ContrastiveTrainer
        
        base_model_path = os.path.join(config.base_model_dir, "final")
        
        trainer = ContrastiveTrainer(
            base_model_path=base_model_path,
            contrastive_data_dir=config.contrastive_data_dir,
            output_dir=config.contrastive_model_dir,
            hyperopt=config.hyperopt.enabled
        )
        
        trainer.train()
        
        console.print("[green]✅ Contrastive learning training completed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Contrastive training failed: {e}[/red]")
        return False

def run_evaluation(config):
    """Step 5: Model evaluation"""
    console.print("\n[bold blue]📊 Step 5: Model Evaluation[/bold blue]")
    
    try:
        import subprocess
        
        contrastive_model_path = os.path.join(config.contrastive_model_dir, "final")
        
        if not os.path.exists(contrastive_model_path):
            console.print(f"[red]❌ Contrastive model not found at {contrastive_model_path}[/red]")
            return False
        
        # Run evaluation
        cmd = [
            "python", "evaluate_model.py",
            "--model_path", contrastive_model_path,
            "--data_dir", config.data_dir,
            "--output_dir", config.evaluation_dir,
            "--batch_size", str(config.data.batch_size),
            "--error_analysis"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✅ Model evaluation completed![/green]")
            return True
        else:
            console.print(f"[red]❌ Evaluation failed: {result.stderr}[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Evaluation error: {e}[/red]")
        return False

def run_quick_inference_test(config):
    """Quick inference test"""
    console.print("\n[bold blue]🔮 Quick Inference Test[/bold blue]")
    
    try:
        from inference import GECInference
        
        contrastive_model_path = os.path.join(config.contrastive_model_dir, "final")
        
        if not os.path.exists(contrastive_model_path):
            console.print("[yellow]⚠️ Contrastive model not found, using base model[/yellow]")
            contrastive_model_path = os.path.join(config.base_model_dir, "final")
        
        # Test inference
        gec = GECInference(
            model_path=contrastive_model_path,
            use_contrastive_search=config.inference.use_contrastive_search,
            contrastive_alpha=config.inference.contrastive_alpha,
            contrastive_k=config.inference.contrastive_k
        )
        
        # Test sentences
        test_sentences = [
            "Tôi đi học trường đại học.",
            "Hôm nay tôi không đi làm.",
            "Cô ấy rất đẹp và thông minh.",
            "Chúng ta phải học bài tập về nhà.",
            "Anh ấy làm việc ở công ty lớn."
        ]
        
        console.print("[yellow]Testing inference on sample sentences:[/yellow]")
        
        for i, sentence in enumerate(test_sentences):
            corrected = gec.correct_text(sentence)
            console.print(f"\n[cyan]Example {i+1}:[/cyan]")
            console.print(f"  [yellow]Original:[/yellow] {sentence}")
            console.print(f"  [green]Corrected:[/green] {corrected}")
        
        console.print("[green]✅ Inference test completed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Inference test failed: {e}[/red]")
        return False

def main():
    parser = argparse.ArgumentParser(description="Vietnamese GEC Training Pipeline")
    parser.add_argument(
        "--config", 
        default="bartpho_syllable",
        choices=["bartpho_syllable", "bartpho_word", "vit5_base", "vit5_large", "quick_test"],
        help="Configuration to use"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=["data", "base", "negative", "contrastive", "eval", "test", "all"],
        help="Steps to run"
    )
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation if already done")
    parser.add_argument("--skip-base", action="store_true", help="Skip base training if already done")
    parser.add_argument("--skip-negative", action="store_true", help="Skip negative sampling if already done")
    parser.add_argument("--quick", action="store_true", help="Use quick test configuration")
    
    args = parser.parse_args()
    
    # Get configuration
    config_name = "quick_test" if args.quick else args.config
    config = get_config(config_name)
    
    console.print("[bold green]🚀 Vietnamese GEC Training Pipeline[/bold green]")
    console.print(f"[blue]Using configuration: {config_name}[/blue]")
    
    # Print configuration
    print_config(config)
    
    # Determine steps to run
    if "all" in args.steps:
        steps_to_run = ["data", "base", "negative", "contrastive", "eval", "test"]
    else:
        steps_to_run = args.steps
    
    # Apply skip flags
    if args.skip_data and "data" in steps_to_run:
        steps_to_run.remove("data")
    if args.skip_base and "base" in steps_to_run:
        steps_to_run.remove("base")
    if args.skip_negative and "negative" in steps_to_run:
        steps_to_run.remove("negative")
    
    console.print(f"[blue]Steps to run: {', '.join(steps_to_run)}[/blue]")
    
    # Execute pipeline
    results = {}
    
    step_functions = {
        "data": run_data_preparation,
        "base": run_base_training,
        "negative": run_negative_sampling,
        "contrastive": run_contrastive_training,
        "eval": run_evaluation,
        "test": run_quick_inference_test
    }
    
    with Progress() as progress:
        task = progress.add_task("Pipeline Progress", total=len(steps_to_run))
        
        for step in steps_to_run:
            console.print(f"\n[bold cyan]Running: {step}[/bold cyan]")
            
            if step in step_functions:
                success = step_functions[step](config)
                results[step] = success
                
                if not success:
                    console.print(f"[red]❌ Step '{step}' failed. Stopping pipeline.[/red]")
                    break
            else:
                console.print(f"[red]❌ Unknown step: {step}[/red]")
                break
            
            progress.advance(task)
    
    # Summary
    console.print("\n[bold blue]📋 Pipeline Summary[/bold blue]")
    
    for step, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        console.print(f"  {step}: {status}")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    if successful_steps == total_steps:
        console.print("\n[bold green]🎉 Pipeline completed successfully![/bold green]")
        console.print(f"[green]All {total_steps} steps passed.[/green]")
        
        # Show final model location
        final_model_path = os.path.join(config.contrastive_model_dir, "final")
        console.print(f"[blue]📦 Final model saved at: {final_model_path}[/blue]")
        
    else:
        console.print(f"\n[bold yellow]⚠️ Pipeline partially completed.[/bold yellow]")
        console.print(f"[yellow]{successful_steps}/{total_steps} steps passed.[/yellow]")

if __name__ == "__main__":
    main()
