"""
Configuration file for Vietnamese GEC with Contrastive Learning
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "phuhuy-se1/viGEC"
    cache_dir: Optional[str] = None
    max_length: int = 384
    test_subset_ratio: float = 0.05  # Use 5% of test set for faster evaluation
    batch_size: int = 16
    num_workers: int = 4
    
@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "vinai/bartpho-syllable"  # or "VietAI/vit5-base"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    max_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_val: float = 1.0
    
@dataclass
class HyperoptConfig:
    """Hyperparameter optimization configuration"""
    enabled: bool = True
    n_trials: int = 30
    direction: str = "maximize"
    metric: str = "val_f05"
    pruning_enabled: bool = True
    
@dataclass
class ContrastiveConfig:
    """Contrastive learning configuration"""
    lambda_cl: float = 1.0  # Balance between CE and CL loss
    temperature: float = 0.25  # Contrastive loss temperature
    rdrop_alpha: float = 4.0  # R-Drop regularization strength
    max_epochs: int = 5
    num_negatives: int = 3
    
@dataclass
class InferenceConfig:
    """Inference configuration"""
    use_contrastive_search: bool = True
    contrastive_alpha: float = 0.7  # Balance between confidence and diversity
    contrastive_k: int = 5  # Top-k candidates
    beam_size: int = 5  # For beam search
    max_length: int = 384
    early_stopping: bool = True
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    data: DataConfig
    model: ModelConfig
    hyperopt: HyperoptConfig
    contrastive: ContrastiveConfig
    inference: InferenceConfig
    
    # Directories
    data_dir: str = "./data/processed"
    base_model_dir: str = "./models/base"
    contrastive_model_dir: str = "./models/contrastive"
    contrastive_data_dir: str = "./data/contrastive"
    evaluation_dir: str = "./evaluation_results"
    
    # Logging
    wandb_project: str = "vigec-gec"
    wandb_entity: Optional[str] = None
    
    # System
    device: str = "auto"
    precision: str = "16-mixed"  # Use mixed precision for faster training
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [
            self.data_dir, 
            self.base_model_dir, 
            self.contrastive_model_dir,
            self.contrastive_data_dir,
            self.evaluation_dir
        ]:
            os.makedirs(dir_path, exist_ok=True)

# Predefined configurations for different models
BARTPHO_SYLLABLE_CONFIG = TrainingConfig(
    data=DataConfig(batch_size=16),
    model=ModelConfig(
        model_name="vinai/bartpho-syllable",
        learning_rate=5e-5,
        max_epochs=10
    ),
    hyperopt=HyperoptConfig(n_trials=30),
    contrastive=ContrastiveConfig(),
    inference=InferenceConfig()
)

BARTPHO_WORD_CONFIG = TrainingConfig(
    data=DataConfig(batch_size=16),
    model=ModelConfig(
        model_name="vinai/bartpho-word",
        learning_rate=3e-5,
        max_epochs=10
    ),
    hyperopt=HyperoptConfig(n_trials=30),
    contrastive=ContrastiveConfig(),
    inference=InferenceConfig()
)

VIT5_BASE_CONFIG = TrainingConfig(
    data=DataConfig(batch_size=8),  # Smaller batch size for ViT5
    model=ModelConfig(
        model_name="VietAI/vit5-base",
        learning_rate=3e-5,
        max_epochs=8
    ),
    hyperopt=HyperoptConfig(n_trials=25),
    contrastive=ContrastiveConfig(max_epochs=4),
    inference=InferenceConfig()
)

VIT5_LARGE_CONFIG = TrainingConfig(
    data=DataConfig(batch_size=4),  # Even smaller batch size for large model
    model=ModelConfig(
        model_name="VietAI/vit5-large",
        learning_rate=2e-5,
        max_epochs=6
    ),
    hyperopt=HyperoptConfig(n_trials=20),
    contrastive=ContrastiveConfig(max_epochs=3),
    inference=InferenceConfig()
)

# Quick test configuration with smaller settings
QUICK_TEST_CONFIG = TrainingConfig(
    data=DataConfig(
        batch_size=4,
        test_subset_ratio=0.01  # Use only 1% for quick testing
    ),
    model=ModelConfig(
        model_name="vinai/bartpho-syllable",
        max_epochs=2
    ),
    hyperopt=HyperoptConfig(
        enabled=False,  # Skip hyperopt for quick test
        n_trials=5
    ),
    contrastive=ContrastiveConfig(max_epochs=1),
    inference=InferenceConfig()
)

def get_config(config_name: str = "bartpho_syllable") -> TrainingConfig:
    """Get configuration by name"""
    
    configs = {
        "bartpho_syllable": BARTPHO_SYLLABLE_CONFIG,
        "bartpho_word": BARTPHO_WORD_CONFIG,
        "vit5_base": VIT5_BASE_CONFIG,
        "vit5_large": VIT5_LARGE_CONFIG,
        "quick_test": QUICK_TEST_CONFIG
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

def print_config(config: TrainingConfig):
    """Print configuration summary"""
    
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    table = Table(title="Training Configuration")
    table.add_column("Component", style="cyan")
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="magenta")
    
    # Model settings
    table.add_row("Model", "Name", config.model.model_name)
    table.add_row("", "Learning Rate", f"{config.model.learning_rate}")
    table.add_row("", "Max Epochs", f"{config.model.max_epochs}")
    table.add_row("", "Label Smoothing", f"{config.model.label_smoothing}")
    
    # Data settings
    table.add_row("Data", "Batch Size", f"{config.data.batch_size}")
    table.add_row("", "Max Length", f"{config.data.max_length}")
    table.add_row("", "Test Subset", f"{config.data.test_subset_ratio*100:.1f}%")
    
    # Hyperopt settings
    table.add_row("Hyperopt", "Enabled", f"{config.hyperopt.enabled}")
    table.add_row("", "Trials", f"{config.hyperopt.n_trials}")
    
    # Contrastive settings
    table.add_row("Contrastive", "Lambda", f"{config.contrastive.lambda_cl}")
    table.add_row("", "Temperature", f"{config.contrastive.temperature}")
    table.add_row("", "R-Drop Alpha", f"{config.contrastive.rdrop_alpha}")
    
    # Inference settings
    table.add_row("Inference", "Contrastive Search", f"{config.inference.use_contrastive_search}")
    table.add_row("", "Alpha", f"{config.inference.contrastive_alpha}")
    table.add_row("", "K", f"{config.inference.contrastive_k}")
    
    console.print(table)

if __name__ == "__main__":
    # Example usage
    from rich.console import Console
    
    console = Console()
    
    console.print("[bold green]Available Configurations:[/bold green]")
    configs = ["bartpho_syllable", "bartpho_word", "vit5_base", "vit5_large", "quick_test"]
    
    for config_name in configs:
        console.print(f"\n[bold blue]{config_name}:[/bold blue]")
        config = get_config(config_name)
        print_config(config)
