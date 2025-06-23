#!/usr/bin/env python3
"""
Test script to verify BARTpho tokenizer fix
"""

import sys
import traceback
from rich.console import Console

console = Console()

def test_bartpho_loading():
    """Test loading BARTpho model and tokenizer"""
    console.print("🧪 Testing BARTpho loading...")
    
    try:
        from data_utils import get_model_and_tokenizer
        
        # Test BARTpho
        console.print("📦 Loading BARTpho...")
        model, tokenizer = get_model_and_tokenizer("vinai/bartpho-syllable")
        
        console.print("✅ BARTpho loaded successfully!")
        console.print(f"📊 Tokenizer type: {type(tokenizer).__name__}")
        console.print(f"📊 Model type: {type(model).__name__}")
        
        # Test tokenization
        test_text = "Tôi đi học trường đại học."
        tokens = tokenizer(test_text, return_tensors="pt")
        console.print(f"🔤 Test tokenization: {test_text}")
        console.print(f"🔢 Token IDs shape: {tokens['input_ids'].shape}")
        
        # Test vocab access methods
        console.print("🔍 Testing vocabulary access...")
        if hasattr(tokenizer, 'vocab'):
            console.print("✅ Has .vocab attribute")
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            console.print(f"✅ Has .get_vocab() method, vocab size: {len(vocab)}")
        else:
            console.print("⚠️ No standard vocab access method found")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error loading BARTpho: {e}")
        traceback.print_exc()
        return False

def test_vit5_loading():
    """Test loading ViT5 model and tokenizer"""
    console.print("🧪 Testing ViT5 loading...")
    
    try:
        from data_utils import get_model_and_tokenizer
        
        # Test ViT5
        console.print("📦 Loading ViT5...")
        model, tokenizer = get_model_and_tokenizer("VietAI/vit5-base")
        
        console.print("✅ ViT5 loaded successfully!")
        console.print(f"📊 Tokenizer type: {type(tokenizer).__name__}")
        console.print(f"📊 Model type: {type(model).__name__}")
        
        # Check task prefix
        if hasattr(tokenizer, 'task_prefix'):
            console.print(f"🏷️ Task prefix: {tokenizer.task_prefix}")
        
        # Test tokenization with prefix
        test_text = "Tôi đi học trường đại học."
        prefixed_text = tokenizer.task_prefix + test_text if hasattr(tokenizer, 'task_prefix') else test_text
        tokens = tokenizer(prefixed_text, return_tensors="pt")
        console.print(f"🔤 Test tokenization: {prefixed_text}")
        console.print(f"🔢 Token IDs shape: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error loading ViT5: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading with tokenizer"""
    console.print("🧪 Testing data loading...")
    
    try:
        from data_utils import load_vigec_dataset, create_data_loaders, get_model_and_tokenizer
        
        # Load small dataset
        console.print("📥 Loading small dataset sample...")
        data = load_vigec_dataset(test_subset_ratio=0.01)  # Only 1% for quick test
        
        # Test with BARTpho
        console.print("🔄 Testing with BARTpho...")
        model, tokenizer = get_model_and_tokenizer("vinai/bartpho-syllable")
        
        # Create small subset
        small_data = {}
        for split, split_data in data.items():
            small_data[split] = split_data[:5]  # Only 5 samples
        
        # Create data loaders
        data_loaders = create_data_loaders(
            small_data, 
            tokenizer, 
            batch_size=2, 
            num_workers=0  # No multiprocessing for testing
        )
        
        console.print("✅ Data loaders created successfully!")
        
        # Test one batch
        if 'train' in data_loaders:
            train_loader = data_loaders['train']
            batch = next(iter(train_loader))
            console.print(f"📦 Batch keys: {list(batch.keys())}")
            console.print(f"📦 Input shape: {batch['input_ids'].shape}")
            console.print(f"📦 Labels shape: {batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error in data loading: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    console.print("🚀 Running BARTpho fix verification tests...\n")
    
    tests = [
        ("BARTpho Model Loading", test_bartpho_loading),
        ("ViT5 Model Loading", test_vit5_loading),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*50}")
        console.print(f"🧪 Running: {test_name}")
        console.print(f"{'='*50}")
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            console.print(f"✅ {test_name}: PASSED")
        else:
            console.print(f"❌ {test_name}: FAILED")
    
    # Summary
    console.print(f"\n{'='*50}")
    console.print("📊 Test Summary")
    console.print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        console.print(f"{test_name}: {status}")
    
    console.print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        console.print("🎉 All tests passed! BARTpho fix is working correctly.")
        return True
    else:
        console.print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
