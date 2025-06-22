"""
Test script to verify the updated data_utils.py functionality
"""

import sys
import os
sys.path.append(os.getcwd())

from data_utils import (
    load_vigec_dataset, 
    get_model_and_tokenizer, 
    can_train_base_model,
    check_dataset_format,
    create_data_loaders
)
from rich.console import Console

console = Console()

def test_system_checks():
    """Test system readiness checks"""
    console.print("[bold blue]Testing System Checks[/bold blue]")
    
    # Test GPU and system readiness
    can_train = can_train_base_model()
    console.print(f"Can train base model: {can_train}")
    
    # Test dataset format check
    dataset_ok = check_dataset_format()
    console.print(f"Dataset format OK: {dataset_ok}")
    
    return can_train and dataset_ok

def test_vit5_model():
    """Test ViT5 model loading with prefix"""
    console.print("[bold blue]Testing ViT5 Model Loading[/bold blue]")
    
    try:
        # Test ViT5
        model, tokenizer = get_model_and_tokenizer("VietAI/vit5-base")
        
        # Check if prefix was added
        has_prefix = hasattr(tokenizer, 'task_prefix')
        console.print(f"ViT5 has task prefix: {has_prefix}")
        
        if has_prefix:
            console.print(f"Task prefix: '{tokenizer.task_prefix}'")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error loading ViT5: {e}[/red]")
        return False

def test_bartpho_model():
    """Test BARTpho model loading"""
    console.print("[bold blue]Testing BARTpho Model Loading[/bold blue]")
    
    try:
        # Test BARTpho
        model, tokenizer = get_model_and_tokenizer("vinai/bartpho-syllable")
        
        # Check that no prefix was added (should only be for ViT5)
        has_prefix = hasattr(tokenizer, 'task_prefix')
        console.print(f"BARTpho has task prefix: {has_prefix}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error loading BARTpho: {e}[/red]")
        return False

def test_data_loading():
    """Test data loading with subset"""
    console.print("[bold blue]Testing Data Loading[/bold blue]")
    
    try:
        # Load with small test subset
        data = load_vigec_dataset(test_subset_ratio=0.01)  # Use 1% for testing
        
        console.print("Data splits loaded:")
        for split, split_data in data.items():
            console.print(f"  {split}: {len(split_data)} samples")
            
            # Check first sample structure
            if split_data:
                sample = split_data[0]
                console.print(f"  Sample keys: {list(sample.keys())}")
                console.print(f"  Source: {sample['source'][:50]}...")
                console.print(f"  Target: {sample['target'][:50]}...")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return False

def test_data_loaders():
    """Test data loader creation"""
    console.print("[bold blue]Testing Data Loaders[/bold blue]")
    
    try:
        # Load small dataset
        data = load_vigec_dataset(test_subset_ratio=0.001)  # Very small for testing
        
        # Get model and tokenizer
        model, tokenizer = get_model_and_tokenizer("vinai/bartpho-syllable")
        
        # Create data loaders
        data_loaders = create_data_loaders(data, tokenizer, batch_size=2)
        
        console.print("Data loaders created:")
        for split, loader in data_loaders.items():
            console.print(f"  {split}: {len(loader)} batches")
            
            # Test one batch
            for batch in loader:
                console.print(f"  Batch shape - input_ids: {batch['input_ids'].shape}")
                console.print(f"  Batch shape - labels: {batch['labels'].shape}")
                break
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error creating data loaders: {e}[/red]")
        return False

if __name__ == "__main__":
    console.print("[bold green]🧪 Testing Updated Data Utils[/bold green]")
    
    all_tests = [
        test_system_checks,
        test_bartpho_model,
        # test_vit5_model,  # Skip ViT5 if not available
        test_data_loading,
        test_data_loaders,
    ]
    
    results = []
    for test_func in all_tests:
        try:
            result = test_func()
            results.append(result)
            console.print(f"✅ {test_func.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            console.print(f"❌ {test_func.__name__}: ERROR - {e}")
            results.append(False)
        
        console.print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    console.print(f"[bold {'green' if passed == total else 'yellow'}]Summary: {passed}/{total} tests passed[/bold {'green' if passed == total else 'yellow'}]")
    
    if passed == total:
        console.print("[bold green]🎉 All tests passed! System is ready.[/bold green]")
    else:
        console.print("[bold yellow]⚠️ Some tests failed. Check the output above.[/bold yellow]")
