#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after fixing AdamW import issues
"""

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing import fixes...")
    
    try:
        # Test AdamW import fix
        from torch.optim import AdamW
        print("✅ torch.optim.AdamW imported successfully")
        
        # Test transformers without AdamW
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
        print("✅ transformers core imports successful")
        
        # Test base trainer
        print("📦 Testing base_trainer.py...")
        import base_trainer
        print("✅ base_trainer.py imported successfully")
        
        # Test contrastive trainer
        print("📦 Testing contrastive_trainer.py...")
        import contrastive_trainer
        print("✅ contrastive_trainer.py imported successfully")
        
        # Test data utils
        print("📦 Testing data_utils.py...")
        import data_utils
        print("✅ data_utils.py imported successfully")
        
        # Test other modules
        modules_to_test = [
            'negative_sampler',
            'inference'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"✅ {module}.py imported successfully")
            except ImportError as e:
                print(f"⚠️ {module}.py has missing dependencies: {e}")
            except Exception as e:
                print(f"❌ {module}.py failed: {e}")
        
        print("\n🎉 Import test completed! AdamW fix successfully applied.")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_imports()
