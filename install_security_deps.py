#!/usr/bin/env python3
"""
Script to install security dependencies for safe model loading
"""

import subprocess
import sys
from rich.console import Console

console = Console()

def install_package(package):
    """Install a package using pip"""
    try:
        console.print(f"[blue]Installing {package}...[/blue]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        console.print(f"[green]✅ {package} installed successfully[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Failed to install {package}: {e}[/red]")
        return False

def main():
    console.print("[bold blue]🔐 Installing security dependencies for safe model loading[/bold blue]")
    
    packages = [
        "safetensors",
        "torch>=2.6.0",  # Latest secure version
        "transformers[safetensors]",
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    if success_count == len(packages):
        console.print("\n[bold green]🎉 All security dependencies installed successfully![/bold green]")
        console.print("[green]Your model training is now secure and optimized for H100.[/green]")
    else:
        console.print(f"\n[yellow]⚠️  {success_count}/{len(packages)} packages installed successfully[/yellow]")
        console.print("[yellow]Some installations failed. Please check the errors above.[/yellow]")

if __name__ == "__main__":
    main()
