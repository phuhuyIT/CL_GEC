#!/usr/bin/env python3
"""
Install additional dependencies for comprehensive metrics
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install comprehensive metrics dependencies"""
    
    print("🔧 Installing Dependencies for Comprehensive Metrics")
    print("=" * 55)
    
    packages = [
        "rouge-score",
        "evaluate",
        "datasets",  # Required by evaluate
        "sacrebleu",  # Better BLEU implementation
    ]
    
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print(f"📊 Installation Summary: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        print("✅ You can now use comprehensive metrics including:")
        print("   • F0.5, Precision, Recall")
        print("   • BLEU, GLEU")
        print("   • ROUGE-1, ROUGE-2, ROUGE-L")
        print("   • Input-preserving Edit Ratio")
    else:
        print("⚠️  Some packages failed to install")
        print("💡 You can try installing them manually:")
        for package in packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()
