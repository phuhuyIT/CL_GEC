"""
Base model trainer with hyperparameter optimization using Optuna
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)

# Handle PyTorch Lightning import with compatibility
try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
except ImportError as e:
    print(f"Warning: Lightning import issue: {e}")
    try:
        # Try older pytorch-lightning import
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
        print("Using pytorch_lightning (legacy)")
    except ImportError:
        raise ImportError("Cannot import PyTorch Lightning. Please install: pip install lightning>=2.0")

# Handle Optuna import
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
except ImportError:
    print("Warning: Optuna not available. Hyperparameter optimization disabled.")
    optuna = None
    PyTorchLightningPruningCallback = None

# Handle Wandb import
try:
    import wandb
except ImportError:
    print("Warning: Wandb not available. Logging disabled.")
    wandb = None

from rich.console import Console
from rich.progress import track
import numpy as np
from typing import Dict, List, Optional, Any
from data_utils import load_vigec_dataset, create_data_loaders, get_model_and_tokenizer
from evaluator import F05Evaluator
import warnings

console = Console()

def check_torch_security():
    """Check PyTorch version and setup secure loading"""
    torch_version = torch.__version__
    console.print(f"[blue]🔍 PyTorch version: {torch_version}[/blue]")
    
    # Parse version
    major, minor = map(int, torch_version.split('.')[:2])
    
    if major < 2 or (major == 2 and minor < 6):
        console.print("[yellow]⚠️  PyTorch version < 2.6 detected. Upgrading recommended for security.[/yellow]")
        console.print("[yellow]    Using safetensors for secure model loading where possible.[/yellow]")
        
        # Set environment variable to prefer safetensors
        os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "./cache")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable faster downloads
        
        # Suppress the warning if we can't avoid torch.load
        warnings.filterwarnings("ignore", message=".*torch.load.*")
        
    else:
        console.print("[green]✅ PyTorch version >= 2.6 - Security requirements met[/green]")
    
    return major, minor

def setup_secure_model_loading():
    """Setup secure model loading configurations"""
    try:
        # Try to import safetensors to check availability
        import safetensors
        console.print("[green]✅ Safetensors available - using secure model loading[/green]")
        
        # Set transformers to prefer safetensors
        os.environ["SAFETENSORS_FAST_GPU"] = "1"
        return True
    except ImportError:
        console.print("[yellow]⚠️  Safetensors not available. Consider installing: pip install safetensors[/yellow]")
        return False

# Check security setup
check_torch_security()
setup_secure_model_loading()

def setup_tensor_cores():
    """Setup optimal Tensor Core configuration for different GPU architectures"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        console.print(f"[blue]🖥️  Detected GPU: {device_name}[/blue]")
        
        # Modern GPUs with Tensor Cores (RTX 30/40/50 series, A100, H100, etc.)
        tensor_core_gpus = [
            'RTX 50', 'RTX 40', 'RTX 30', 'RTX 20','RTX PRO 60',  # Consumer RTX series
            'A100', 'A40', 'A30', 'A10',             # Data center A series
            'H100', 'H200',                          # Hopper architecture
            'V100',                                  # Volta architecture
            'T4'                                     # Tesla T4
        ]
        
        has_tensor_cores = any(gpu in device_name for gpu in tensor_core_gpus)
        
        if has_tensor_cores:
            # Use 'high' for newer architectures (RTX 40/50, H100) for best performance
            # Use 'medium' for older ones for balanced performance/precision
            if any(gpu in device_name for gpu in ['RTX 50', 'RTX 40', 'H100', 'H200', 'RTX PRO 60']):
                precision_mode = 'high'
                console.print("[green]🚀 Setting Tensor Core precision to 'high' for optimal performance[/green]")
            else:
                precision_mode = 'medium'
                console.print("[green]⚡ Setting Tensor Core precision to 'medium' for balanced performance[/green]")
            
            torch.set_float32_matmul_precision(precision_mode)
            console.print(f"[green]✅ Tensor Cores enabled with '{precision_mode}' precision[/green]")
        else:
            console.print("[yellow]⚠️  Tensor Cores not detected or not supported on this GPU[/yellow]")
    else:
        console.print("[yellow]⚠️  CUDA not available, running on CPU[/yellow]")

# Setup Tensor Cores at module import
setup_tensor_cores()

def get_optimal_precision():
    """Get optimal precision setting based on available hardware"""
    if not torch.cuda.is_available():
        return "32-true"
    
    device_name = torch.cuda.get_device_name()
    
    # RTX 50/40 series and H100 can handle bf16 very well
    if any(gpu in device_name for gpu in ['RTX 50', 'RTX 40', 'H100', 'H200']):
        # Check if bfloat16 is supported
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"  # Best for newest GPUs
        else:
            return "16-mixed"
    
    # For other modern GPUs with Tensor Cores, use 16-mixed
    tensor_core_gpus = ['RTX 30', 'RTX 20', 'A100', 'A40', 'A30', 'A10', 'V100', 'T4']
    if any(gpu in device_name for gpu in tensor_core_gpus):
        return "16-mixed"
    
    # For older GPUs without Tensor Cores
    return "32-true"

def is_interactive_environment():
    """Detect if running in interactive environment (Jupyter/Colab/Kaggle)"""
    try:
        # Check for Jupyter/IPython
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    
    # Check for Google Colab
    try:
        import google.colab
        return True
    except ImportError:
        pass
    
    # Check for Kaggle
    try:
        import kaggle
        return True
    except ImportError:
        pass
    
    # Check environment variables
    if any(env in os.environ for env in ['COLAB_GPU', 'KAGGLE_KERNEL_RUN_TYPE', 'JPY_PARENT_PID']):
        return True
    
    return False

def get_optimal_trainer_settings():
    """Get optimal trainer settings based on available hardware"""
    settings = {
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 1,
        'enable_checkpointing': True
    }
    
    if torch.cuda.is_available():
        is_interactive = is_interactive_environment()
        
        # For interactive environments, we need to be careful about CUDA initialization
        if is_interactive:
            # Use environment variables or lazy detection to avoid CUDA context issues
            device_count = torch.cuda.device_count()  # This is safe in most cases
            
            console.print(f"[blue]📓 Interactive environment detected[/blue]")
            console.print(f"[blue]🖥️  Available GPUs: {device_count}[/blue]")
            
            if device_count > 1:
                # For multi-GPU in interactive environments, use ddp_notebook strategy
                # PyTorch Lightning requires this for notebook compatibility
                console.print("[yellow]📔 Using ddp_notebook strategy for interactive multi-GPU[/yellow]")
                settings['strategy'] = 'ddp_notebook'  # Required for notebook environments
                settings['devices'] = device_count
                
                console.print(f"[green]🚀 Multi-GPU training enabled with {device_count} GPUs using DDP Notebook[/green]")
                console.print("[blue]💡 DDP Notebook is the only supported multi-GPU strategy in notebooks[/blue]")
                
                # Adjust batch size for multi-GPU
                console.print(f"[yellow]💡 Remember to scale your batch size for {device_count} GPUs[/yellow]")
                console.print(f"[yellow]   Effective batch size = batch_size × {device_count}[/yellow]")
            else:
                settings['devices'] = 1
                console.print(f"[blue]⚙️  Single GPU training in interactive environment[/blue]")
                
        else:
            # Script environment - safe to get detailed GPU info
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name() if device_count > 0 else "Unknown"
            
            console.print(f"[blue]🖥️  Available GPUs: {device_count}[/blue]")
            if device_count > 0:
                console.print(f"  GPU 0: {device_name}")
            
            # Multi-GPU strategy selection for script environments
            if device_count > 1:
                # Create DDP strategy with proper configuration
                try:
                    from lightning.pytorch.strategies import DDPStrategy
                    
                    ddp_strategy = DDPStrategy(
                        find_unused_parameters=False,  # Optimize for transformer models
                        gradient_as_bucket_view=True,  # Memory optimization
                    )
                    settings['strategy'] = ddp_strategy
                    
                except ImportError:
                    # Fallback to string strategy
                    console.print("[yellow]⚠️  Using fallback DDP strategy[/yellow]")
                    settings['strategy'] = 'ddp'
                
                settings['devices'] = device_count
                console.print(f"[green]🚀 Multi-GPU training enabled with {device_count} GPUs using DDP[/green]")
                
                # Adjust batch size for multi-GPU
                console.print(f"[yellow]💡 Remember to scale your batch size for {device_count} GPUs[/yellow]")
                console.print(f"[yellow]   Effective batch size = batch_size × {device_count}[/yellow]")
                
            else:
                settings['devices'] = 1
                console.print(f"[blue]⚙️  Single GPU training: {device_name}[/blue]")
        
        # Memory optimization settings
        if torch.cuda.device_count() > 1:
            settings['sync_batchnorm'] = True  # Synchronize batch norm across GPUs
    
    return settings

def calculate_optimal_batch_size(base_batch_size: int, device_count: int = None):
    """Calculate optimal batch size for multi-GPU training"""
    if device_count is None:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if device_count <= 1:
        return base_batch_size
    
    # For multi-GPU, we want each GPU to have reasonable batch size
    # Total effective batch size = per_gpu_batch_size * num_gpus
    per_gpu_batch_size = max(1, base_batch_size // device_count)
    total_batch_size = per_gpu_batch_size * device_count
    
    console.print(f"[blue]📊 Multi-GPU batch size calculation:[/blue]")
    console.print(f"  Base batch size: {base_batch_size}")
    console.print(f"  GPUs: {device_count}")
    console.print(f"  Per-GPU batch size: {per_gpu_batch_size}")
    console.print(f"  Total effective batch size: {total_batch_size}")
    
    return per_gpu_batch_size

def get_multi_gpu_config():
    """Get multi-GPU configuration info"""
    if not torch.cuda.is_available():
        return {'devices': 1, 'strategy': 'auto', 'num_gpus': 0}
    
    device_count = torch.cuda.device_count()
    is_interactive = is_interactive_environment()
    
    # Determine strategy based on environment
    if device_count > 1:
        if is_interactive:
            # Use ddp_notebook for interactive environments - required by PyTorch Lightning
            strategy = 'ddp_notebook'
        else:
            strategy = 'ddp'
    else:
        strategy = 'auto'
    
    config = {
        'num_gpus': device_count,
        'devices': device_count if device_count > 1 else 1,
        'strategy': strategy,
        'is_interactive': is_interactive,
        'gpu_names': [],  # Avoid early GPU info calls to prevent CUDA context issues
        'total_memory': 0  # Will be filled later when safe
    }
    
    if device_count > 1:
        env_type = "Interactive (Jupyter/Colab/Kaggle)" if is_interactive else "Script"
        strategy_name = "DDP Notebook" if is_interactive else "DDP"
        
        console.print(f"[bold green]🚀 Multi-GPU Setup Detected![/bold green]")
        console.print(f"  Environment: {env_type}")
        console.print(f"  Number of GPUs: {device_count}")
        console.print(f"  Strategy: {strategy_name}")
        console.print(f"  Expected speedup: {device_count}x (linear scaling)")
        
        if is_interactive:
            console.print("[blue]💡 Using DDP Spawn to avoid CUDA context conflicts[/blue]")
    
    return config

class SecureModelCheckpoint(ModelCheckpoint):
    """Secure ModelCheckpoint that prefers safetensors format"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_safetensors = True
        
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        """Override to use safetensors when possible"""
        try:
            # Try to save with safetensors format
            if self.use_safetensors and hasattr(trainer.lightning_module.model, 'save_pretrained'):
                # For transformers models, use safe_serialization
                model_dir = filepath.replace('.ckpt', '_safetensors')
                os.makedirs(model_dir, exist_ok=True)
                
                trainer.lightning_module.model.save_pretrained(
                    model_dir, 
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                
                # Save additional Lightning state
                lightning_state = {
                    'epoch': trainer.current_epoch,
                    'global_step': trainer.global_step,
                    'pytorch-lightning_version': trainer.lightning_module.__class__.__module__.split('.')[0],
                    'state_dict': {},  # Empty since model is saved separately
                    'lr_schedulers': [],
                    'epoch_loop.state_dict': {},
                    'optimizer_states': [],
                    'hparams_name': 'hparams',
                    'hyper_parameters': trainer.lightning_module.hparams,
                }
                
                # Save lightning state as json (safer than pickle)
                state_file = os.path.join(model_dir, 'lightning_state.json')
                with open(state_file, 'w') as f:
                    json.dump(lightning_state, f, indent=2, default=str)
                
                console.print(f"[green]✅ Model saved securely with safetensors: {model_dir}[/green]")
                
            else:
                # Fallback to default Lightning checkpoint
                super()._save_checkpoint(trainer, filepath)
                console.print(f"[yellow]⚠️  Fallback to standard checkpoint: {filepath}[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Safetensors save failed, using standard checkpoint: {e}[/yellow]")
            super()._save_checkpoint(trainer, filepath)

class GECLightningModule(L.LightningModule):
    """Lightning module for GEC with mT5 prefix support and multi-GPU optimization"""
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store model name for prefix detection
        self.model_name = model_name
        
        # Multi-GPU configuration
        self.multi_gpu_config = get_multi_gpu_config()
        
        # Check if mT5/T5 model
        self.is_mt5 = any(mt5_variant in model_name.lower() for mt5_variant in [
            'mt5', 't5', 'vit5'
        ])
        
        if self.is_mt5:
            self.prefix = "grammar: "
            console.print(f"[blue]🏷️ mT5 model detected, using prefix: '{self.prefix}'[/blue]")
        else:
            self.prefix = ""
        
        # Load model and tokenizer
        self.model, self.tokenizer = get_model_and_tokenizer(model_name)
        
        # Multi-GPU: Wrap model for better performance if needed
        if self.multi_gpu_config['num_gpus'] > 1:
            console.print(f"[green]🔧 Optimizing model for {self.multi_gpu_config['num_gpus']} GPUs[/green]")
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=label_smoothing
        )
        # Evaluator
        self.evaluator = F05Evaluator(self.tokenizer)
        
        # Track best metrics
        self.best_f05 = 0.0
        
        # Store validation outputs for epoch end processing
        self.validation_step_outputs = []
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Only do expensive generation and F0.5 calculation for a subset
        # Increased frequency for more accurate validation metrics
        do_generation = (
            batch_idx == 0 or  # Always do first batch for examples
            batch_idx % 10 == 0  # Every 10th batch (increased from 50) for better F0.5 monitoring
        )
        
        if do_generation:
            # Generate predictions for evaluation (much faster settings)
            with torch.no_grad():
                # Add task prefix for ViT5/mT5 models during validation
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Check if we need to add task prefix (for ViT5/mT5)
                if hasattr(self.tokenizer, 'task_prefix') and self.tokenizer.task_prefix:
                    # Get source texts and add prefix
                    source_texts = batch['source_text']
                    prefixed_texts = [self.tokenizer.task_prefix + text for text in source_texts]
                    
                    # Re-tokenize with prefix
                    prefixed_inputs = self.tokenizer(
                        prefixed_texts,
                        max_length=384,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    input_ids = prefixed_inputs['input_ids']
                    attention_mask = prefixed_inputs['attention_mask']
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=384,  
                    num_beams=5,     # Match inference settings
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode predictions and targets
            predictions = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            targets = batch['target_text']
            sources = batch['source_text']
            
            # Calculate F0.5 for this batch
            f05_scores = []
            for pred, target, source in zip(predictions, targets, sources):
                f05 = self.evaluator.calculate_f05(source, pred, target)
                f05_scores.append(f05)
            
            avg_f05 = float(np.mean(f05_scores))
            
            # Store outputs for epoch end processing
            output = {
                'val_loss': loss,
                'val_f05': avg_f05,
                'predictions': predictions[:3],  # Reduced from 5
                'targets': targets[:3],
                'sources': sources[:3],
                'has_generation': True
            }
        else:
            # Fast validation step - only compute loss
            output = {
                'val_loss': loss,
                'val_f05': None,  # Will be ignored in epoch end
                'has_generation': False            }
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if do_generation:
            self.log('val_f05', avg_f05, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        if not outputs:
            return
        
        # Only calculate F0.5 from outputs that have generation
        f05_outputs = [x for x in outputs if x.get('has_generation', False) and x['val_f05'] is not None]
        
        if f05_outputs:
            avg_f05 = torch.tensor([x['val_f05'] for x in f05_outputs]).mean()
            
            # For multi-GPU, we need to sync metrics across all processes
            if self.multi_gpu_config['num_gpus'] > 1:
                # All-reduce the metric across GPUs
                avg_f05 = self.all_reduce(avg_f05, reduce_op='mean')
            
            if avg_f05 > self.best_f05:
                self.best_f05 = avg_f05
                console.print(f"[green]New best F0.5: {self.best_f05:.4f}[/green]")
                
            # Log examples from the first output with generation (only on rank 0 for multi-GPU)
            if (self.multi_gpu_config['num_gpus'] <= 1 or self.global_rank == 0) and f05_outputs:
                first_gen_output = f05_outputs[0]
                for i, (src, pred, tgt) in enumerate(zip(
                    first_gen_output['sources'],
                    first_gen_output['predictions'], 
                    first_gen_output['targets']
                )):
                    # Handle different logger types
                    if hasattr(self.logger, 'experiment'):
                        if hasattr(self.logger.experiment, 'log'):
                            # Weights & Biases logger
                            self.logger.experiment.log({
                                f"example_{i}_source": src,
                                f"example_{i}_prediction": pred,
                                f"example_{i}_target": tgt
                            })
                        elif hasattr(self.logger.experiment, 'add_text'):
                            # TensorBoard logger
                            self.logger.experiment.add_text(
                                f"example_{i}", 
                                f"Source: {src}\nPrediction: {pred}\nTarget: {tgt}",
                                self.current_epoch
                            )
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8
        )
        
        # Linear warmup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(
        self,
        model_name: str,
        data_loaders: Dict[str, DataLoader],
        n_trials: int = 30,
        direction: str = "maximize",
        use_wandb: bool = True,
        dataset_name: str = "phuhuy-se1/viGEC",
        train_subset_ratio: float = 1.0,
        validation_subset_ratio: float = 1.0,
        test_subset_ratio: float = 0.05,
        search_space: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.data_loaders = data_loaders
        self.n_trials = n_trials
        self.direction = direction
        self.use_wandb = use_wandb
        self.dataset_name = dataset_name
        self.train_subset_ratio = train_subset_ratio
        self.validation_subset_ratio = validation_subset_ratio
        self.test_subset_ratio = test_subset_ratio
        self.search_space = search_space or {}
        
    def objective(self, trial) -> float:
        """Objective function for Optuna optimization"""
        
        if optuna is None:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")
        
        console.print(f"\n[bold cyan]🔍 Starting Trial {trial.number}[/bold cyan]")
        
        # Get search space parameters or use defaults
        search_space = self.search_space
        
        # Suggest hyperparameters with customizable search space
        lr_range = search_space.get('learning_rate', {'low': 1e-6, 'high': 1e-3, 'log': True})
        wd_range = search_space.get('weight_decay', {'low': 0.001, 'high': 0.1, 'log': True})
        ls_range = search_space.get('label_smoothing', {'low': 0.0, 'high': 0.3})
        bs_choices = search_space.get('batch_size', [16, 48, 96])
        wr_range = search_space.get('warmup_ratio', {'low': 0.05, 'high': 0.2})
        
        lr = trial.suggest_float('learning_rate', **lr_range)
        weight_decay = trial.suggest_float('weight_decay', **wd_range)
        label_smoothing = trial.suggest_float('label_smoothing', **ls_range)
        batch_size = trial.suggest_categorical('batch_size', bs_choices)
        warmup_ratio = trial.suggest_float('warmup_ratio', **wr_range)
        
        # Apply batch size dependent learning rate scaling (optional)
        # lr_scaled = lr * (batch_size / 16)  # Scale relative to base batch_size=16
        
        console.print(f"[yellow]📋 Trial {trial.number} parameters:[/yellow]")
        console.print(f"  Learning rate: {lr:.2e}")
        console.print(f"  Weight decay: {weight_decay:.4f}")
        console.print(f"  Label smoothing: {label_smoothing:.3f}")
        console.print(f"  Batch size: {batch_size}")
        console.print(f"  Warmup ratio: {warmup_ratio:.3f}")
        
        # Recreate data loaders with the suggested batch size
        # This is crucial - batch size affects convergence and memory usage
        try:
            console.print(f"[blue]🔄 Creating data loaders with batch size {batch_size}...[/blue]")
            from data_utils import create_data_loaders, get_model_and_tokenizer
            # Need to get tokenizer to recreate data loaders
            _, tokenizer = get_model_and_tokenizer(self.model_name)
            
            # Load data fresh for this trial with dataset parameters
            from data_utils import load_vigec_dataset
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
            
            # Create data loaders with trial-specific batch size
            trial_data_loaders = create_data_loaders(
                data=data,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
            console.print(f"[green]✅ Data loaders created successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to create data loaders for trial {trial.number}: {e}[/red]")
            return 0.0
          # Calculate steps based on actual data loader length
        train_steps_per_epoch = len(trial_data_loaders['train'])
        max_epochs = 3  # Coarse search with fewer epochs
        max_steps = train_steps_per_epoch * max_epochs
        warmup_steps = int(max_steps * warmup_ratio)
        
        console.print(f"[blue]📊 Training setup:[/blue]")
        console.print(f"  Steps per epoch: {train_steps_per_epoch}")
        console.print(f"  Max epochs: {max_epochs}")
        console.print(f"  Total steps: {max_steps}")
        console.print(f"  Warmup steps: {warmup_steps}")
        
        # Create model with all hyperparameters
        console.print(f"[blue]🤖 Creating model...[/blue]")
        model = GECLightningModule(
            model_name=self.model_name,
            learning_rate=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            warmup_steps=warmup_steps,
            max_steps=max_steps
        )
        console.print(f"[green]✅ Model created successfully[/green]")
        
        # Logger
        if self.use_wandb and wandb is not None:
            wandb_logger = WandbLogger(
                project="vigec-hyperopt",
                name=f"trial_{trial.number}",
                config={
                    **trial.params,
                    'max_steps': max_steps,
                    'warmup_steps': warmup_steps,
                    'train_steps_per_epoch': train_steps_per_epoch                }
            )
        else:
            # Use TensorBoard logger for hyperopt when wandb is disabled
            wandb_logger = TensorBoardLogger(
                save_dir="./logs", 
                name=f"trial_{trial.number}"
            )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_f05',
            patience=3,
            mode='max',
            verbose=True
        )
        
        checkpoint_callback = SecureModelCheckpoint(
            monitor='val_f05',
            mode='max',
            save_top_k=1,
            filename=f'trial_{trial.number}_best'
        )
        
        if PyTorchLightningPruningCallback is not None:
            pruning_callback = PyTorchLightningPruningCallback(
                trial, monitor='val_f05'
            )
            callbacks_list = [early_stopping, pruning_callback, checkpoint_callback]
        else:
            console.print("[yellow]⚠️  Optuna pruning not available, skipping pruning callback[/yellow]")
            callbacks_list = [early_stopping, checkpoint_callback]
          # Trainer with progress bars enabled for better visibility
        precision = get_optimal_precision()
        trainer_settings = get_optimal_trainer_settings()
        console.print(f"[blue]🎯 Using precision: {precision}[/blue]")
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=callbacks_list,
            enable_progress_bar=True,   # Enable progress bars for hyperopt visibility
            enable_model_summary=False, # Keep model summary disabled to reduce clutter
            accelerator='auto',
            precision=precision,
            **trainer_settings
        )
        
        try:
            console.print(f"[bold green]🚀 Starting training for Trial {trial.number}...[/bold green]")
            
            # Train with trial-specific data loaders
            trainer.fit(
                model,
                train_dataloaders=trial_data_loaders['train'],
                val_dataloaders=trial_data_loaders['validation']
            )
            
            # Get best metric
            best_f05 = model.best_f05
            console.print(f"[bold green]✅ Trial {trial.number} completed! Best F0.5: {best_f05:.4f}[/bold green]")
            
            # Clean up
            if self.use_wandb and wandb is not None:
                wandb.finish()
            
            return best_f05
            
        except Exception as e:
            console.print(f"[red]❌ Trial {trial.number} failed: {e}[/red]")
            if self.use_wandb and wandb is not None:
                wandb.finish()
            return 0.0
    
    def optimize(self, study_name: str = "vigec_hyperopt"):
        """Run hyperparameter optimization"""
        
        if optuna is None:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")
        
        console.print(f"[bold blue]Starting hyperparameter optimization with {self.n_trials} trials[/bold blue]")
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )
          # Progress tracking callback
        def progress_callback(study, trial):
            console.print(f"\n[bold yellow]📊 Trial {trial.number} Summary:[/bold yellow]")
            console.print(f"  F0.5 Score: {trial.value:.4f}")
            console.print(f"  Best so far: {study.best_value:.4f} (Trial {study.best_trial.number})")
            console.print(f"  Progress: {len(study.trials)}/{self.n_trials} trials completed")
            
            if len(study.trials) > 1:
                recent_trials = study.trials[-5:]  # Last 5 trials
                recent_scores = [t.value for t in recent_trials if t.value is not None]
                if recent_scores:
                    avg_recent = sum(recent_scores) / len(recent_scores)
                    console.print(f"  Recent average: {avg_recent:.4f}")
            
            console.print("-" * 60)
        
        # Optimize
        console.print(f"[bold green]🚀 Starting optimization process...[/bold green]")
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=[progress_callback]
        )
        
        # Print results
        console.print(f"[green]Best trial: {study.best_trial.number}[/green]")
        console.print(f"[green]Best F0.5: {study.best_value:.4f}[/green]")
        console.print("[green]Best parameters:[/green]")
        for key, value in study.best_params.items():
            console.print(f"  {key}: {value}")
        
        return study

def safe_load_checkpoint(checkpoint_path: str):
    """Safely load checkpoint with security considerations"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        # Try to load safetensors first
        if os.path.isdir(checkpoint_path):
            # Directory containing safetensors
            safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
            if safetensor_files:
                console.print(f"[green]✅ Loading secure safetensors checkpoint: {checkpoint_path}[/green]")
                return checkpoint_path
        
        # Check for .safetensors file
        if checkpoint_path.endswith('.safetensors'):
            console.print(f"[green]✅ Loading safetensors checkpoint: {checkpoint_path}[/green]")
            return checkpoint_path
            
        # For .ckpt files, warn about security
        if checkpoint_path.endswith('.ckpt'):
            major, minor = check_torch_security()
            if major < 2 or (major == 2 and minor < 6):
                console.print(f"[red]❌ Cannot load .ckpt file with PyTorch < 2.6 due to security restrictions[/red]")
                console.print(f"[yellow]💡 Solutions:[/yellow]")
                console.print(f"[yellow]  1. Upgrade PyTorch: pip install torch>=2.6[/yellow]")
                console.print(f"[yellow]  2. Use safetensors format instead[/yellow]")
                console.print(f"[yellow]  3. Convert checkpoint: torch.save(torch.load('{checkpoint_path}'), '{checkpoint_path}', _use_new_zipfile_serialization=False)[/yellow]")
                raise ValueError("Cannot load .ckpt file due to PyTorch security restrictions")
            
        console.print(f"[yellow]⚠️  Loading checkpoint (ensure it's from trusted source): {checkpoint_path}[/yellow]")
        return checkpoint_path
        
    except Exception as e:
        console.print(f"[red]❌ Error loading checkpoint: {e}[/red]")
        raise

class BaseTrainer:
    """Main trainer for base model"""
    
    def __init__(
        self,
        model_name: str,
        data_dir: str = "./data/processed",
        output_dir: str = "./models/base",
        hyperopt: bool = True,
        use_wandb: bool = True,
        dataset_name: str = "phuhuy-se1/viGEC",
        train_subset_ratio: float = 1.0,
        validation_subset_ratio: float = 1.0,
        test_subset_ratio: float = 0.05
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.hyperopt = hyperopt
        self.use_wandb = use_wandb
        self.dataset_name = dataset_name
        self.train_subset_ratio = train_subset_ratio
        self.validation_subset_ratio = validation_subset_ratio
        self.test_subset_ratio = test_subset_ratio        
        os.makedirs(output_dir, exist_ok=True)
        
    def _run_hyperopt(self, data_loaders, n_trials=10, study_name="vigec_hyperopt", search_space=None):
        """Internal method to run hyperparameter optimization"""
        optimizer = HyperparameterOptimizer(
            model_name=self.model_name,
            data_loaders=data_loaders,
            n_trials=n_trials,
            use_wandb=self.use_wandb,
            dataset_name=self.dataset_name,
            train_subset_ratio=self.train_subset_ratio,
            validation_subset_ratio=self.validation_subset_ratio,
            test_subset_ratio=self.test_subset_ratio,
            search_space=search_space
        )
        
        study = optimizer.optimize(study_name=study_name)
        
        # Save best parameters
        best_params_path = os.path.join(self.output_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
        console.print(f"[green]Best parameters saved to {best_params_path}[/green]")
            
        return study

    def _train_model(self, data_loaders, model_config, max_epochs, run_name="final_model"):
        """Internal method to train model with given config"""
        # Create model
        model = GECLightningModule(**model_config)        # Logger
        if self.use_wandb and wandb is not None:
            wandb_logger = WandbLogger(project="vigec-base-training", name=run_name)
        else:
            # Use TensorBoard logger when wandb is disabled
            wandb_logger = TensorBoardLogger(save_dir="./logs", name=run_name)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_f05', patience=5, mode='max', verbose=True),
            SecureModelCheckpoint(
                dirpath=self.output_dir,
                monitor='val_f05',
                mode='max',
                save_top_k=3,
                filename=f'model_{run_name}_{{epoch:02d}}_{{val_f05:.4f}}'
            )
        ]
        
        # Trainer
        precision = get_optimal_precision()
        trainer_settings = get_optimal_trainer_settings()
        console.print(f"[blue]🎯 Using precision: {precision}[/blue]")
        console.print(f"[dim]📋 Trainer settings: {list(trainer_settings.keys())}[/dim]")
        
        # Filter out any invalid trainer arguments
        valid_trainer_args = {
            'strategy', 'devices', 'enable_checkpointing', 'accumulate_grad_batches', 
            'sync_batchnorm', 'num_nodes', 'enable_progress_bar', 'enable_model_summary'
        }
        filtered_settings = {k: v for k, v in trainer_settings.items() if k in valid_trainer_args}
        
        if len(filtered_settings) != len(trainer_settings):
            removed_args = set(trainer_settings.keys()) - set(filtered_settings.keys())
            console.print(f"[yellow]⚠️  Filtered out invalid Trainer args: {removed_args}[/yellow]")
        
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            accelerator='auto',
            precision=precision,
            **filtered_settings
        )
        
        # Train
        trainer.fit(
            model,
            train_dataloaders=data_loaders['train'],
            val_dataloaders=data_loaders['validation']
        )
        
        # Save model with safetensors if possible
        try:
            output_path = os.path.join(self.output_dir, run_name)
            os.makedirs(output_path, exist_ok=True)
            
            # Save with safe serialization
            model.model.save_pretrained(
                output_path, 
                safe_serialization=True,
                max_shard_size="2GB"
            )
            model.tokenizer.save_pretrained(output_path)
            
            console.print(f"[green]✅ Model saved securely with safetensors: {output_path}[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Safetensors save failed, using standard format: {e}[/yellow]")
            # Fallback to standard save
            model.model.save_pretrained(os.path.join(self.output_dir, run_name))
            model.tokenizer.save_pretrained(os.path.join(self.output_dir, run_name))
        
        if self.use_wandb and wandb is not None:
            wandb.finish()
            
        return model
    
    def train(self, max_epochs: int = 10, batch_size: int = 16, search_space: Optional[Dict[str, Any]] = None):
        """Main training method with optional hyperparameter optimization and multi-GPU support"""
        console.print(f"[bold blue]Starting training for {self.model_name}[/bold blue]")
        
        # Get multi-GPU configuration
        multi_gpu_config = get_multi_gpu_config()
        
        # Adjust batch size for multi-GPU
        if multi_gpu_config['num_gpus'] > 1:
            original_batch_size = batch_size
            batch_size = calculate_optimal_batch_size(batch_size, multi_gpu_config['num_gpus'])
            console.print(f"[yellow]📊 Adjusted batch size for multi-GPU:[/yellow]")
            console.print(f"  Original: {original_batch_size} → Per-GPU: {batch_size}")
            console.print(f"  Total effective: {batch_size * multi_gpu_config['num_gpus']}")
        
        # Load data with proper data directory
        console.print("[yellow]Loading data...[/yellow]")
        try:
            # Use data_dir if provided, otherwise load from HuggingFace with dataset parameters
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                data_dir=self.data_dir,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load data with data_dir, falling back to HuggingFace: {e}[/yellow]")
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
        
        # Get tokenizer for data loading
        _, tokenizer = get_model_and_tokenizer(self.model_name)
        
        # Create data loaders
        data_loaders = create_data_loaders(
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
        best_params = None
        
        # Step 1: Hyperparameter optimization (if enabled)
        if self.hyperopt:
            if optuna is None:
                console.print("[yellow]⚠️  Optuna not available. Skipping hyperparameter optimization.[/yellow]")
                console.print("[yellow]    Install with: pip install optuna[/yellow]")
            else:
                console.print("[yellow]Running hyperparameter optimization...[/yellow]")
                if multi_gpu_config['num_gpus'] > 1:
                    console.print("[blue]🔧 Note: Hyperopt will use single GPU for faster trials[/blue]")
                study = self._run_hyperopt(data_loaders, n_trials=10, search_space=search_space)
                best_params = study.best_params
                console.print(f"[green]Hyperparameter optimization complete. Best F0.5: {study.best_value:.4f}[/green]")
        
        # Step 2: Final training with best parameters
        console.print("[yellow]Training final model...[/yellow]")
        
        if best_params:
            # Filter parameters that GECLightningModule accepts - include more parameters
            filtered_params = {
                k: best_params[k] 
                for k in ['learning_rate', 'weight_decay', 'label_smoothing', 'warmup_ratio'] 
                if k in best_params
            }
              # Handle warmup_ratio if present
            warmup_steps = int(len(data_loaders['train']) * max_epochs * 0.1)  # default
            if 'warmup_ratio' in filtered_params:
                warmup_steps = int(len(data_loaders['train']) * max_epochs * filtered_params.pop('warmup_ratio'))
            
            model_config = {
                'model_name': self.model_name,
                **filtered_params,
                'max_steps': len(data_loaders['train']) * max_epochs,
                'warmup_steps': warmup_steps
            }
        else:
            # Use default parameters
            model_config = {
                'model_name': self.model_name,
                'learning_rate': 5e-5,
                'weight_decay': 0.01,
                'label_smoothing': 0.1,
                'max_steps': len(data_loaders['train']) * max_epochs,
                'warmup_steps': int(len(data_loaders['train']) * max_epochs * 0.1)            }
        
        final_model = self._train_model(
            data_loaders, model_config, max_epochs, 
            run_name="final_model"
        )
        
        console.print(f"[green]Training complete! Model saved to {self.output_dir}[/green]")
        return final_model
    
    def train_with_params(self, params: Dict[str, Any], max_epochs: int = 10, batch_size: int = 16):
        """Train model with specific hyperparameters"""
        console.print(f"[bold blue]Training {self.model_name} with custom parameters[/bold blue]")
        
        # Load data with proper data directory
        try:
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                data_dir=self.data_dir,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load data with data_dir, falling back to HuggingFace: {e}[/yellow]")
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
        
        # Get tokenizer for data loading
        _, tokenizer = get_model_and_tokenizer(self.model_name)
        
        # Use batch_size from params if available, otherwise use the parameter
        effective_batch_size = params.get('batch_size', batch_size)
        
        # Create data loaders
        data_loaders = create_data_loaders(
            data=data,
            tokenizer=tokenizer,
            batch_size=effective_batch_size
        )
        
        # Filter parameters for GECLightningModule and handle warmup_ratio
        filtered_params = {
            k: params[k] 
            for k in ['learning_rate', 'weight_decay', 'label_smoothing'] 
            if k in params
        }
        
        # Handle warmup_ratio if present
        warmup_steps = int(len(data_loaders['train']) * max_epochs * 0.1)  # default
        if 'warmup_ratio' in params:
            warmup_steps = int(len(data_loaders['train']) * max_epochs * params['warmup_ratio'])
        
        model_config = {
            'model_name': self.model_name,
            **filtered_params,
            'max_steps': len(data_loaders['train']) * max_epochs,
            'warmup_steps': warmup_steps
        }
        
        model = self._train_model(
            data_loaders, model_config, max_epochs,
            run_name="custom_params_model"
        )
        
        console.print(f"[green]Training complete! Model saved to {self.output_dir}[/green]")
        return model
    
    def optimize_hyperparameters(self, n_trials: int = 30, batch_size: int = 16, search_space: Optional[Dict[str, Any]] = None):
        """Run only hyperparameter optimization"""
        
        if optuna is None:
            console.print("[red]❌ Optuna is required for hyperparameter optimization[/red]")
            console.print("[yellow]Install with: pip install optuna[/yellow]")
            return None
        
        console.print(f"[bold blue]Running hyperparameter optimization for {self.model_name}[/bold blue]")
        
        # Load data with proper data directory
        try:
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                data_dir=self.data_dir,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load data with data_dir, falling back to HuggingFace: {e}[/yellow]")
            data = load_vigec_dataset(
                dataset_name=self.dataset_name,
                train_subset_ratio=self.train_subset_ratio,
                validation_subset_ratio=self.validation_subset_ratio,
                test_subset_ratio=self.test_subset_ratio
            )
        
        # Get tokenizer for data loading
        _, tokenizer = get_model_and_tokenizer(self.model_name)
        
        # Create data loaders
        data_loaders = create_data_loaders(
            data=data,
            tokenizer=tokenizer,            batch_size=batch_size
        )
        
        study = self._run_hyperopt(
            data_loaders, n_trials=n_trials, 
            study_name="vigec_standalone_hyperopt",
            search_space=search_space
        )
        
        console.print(f"[green]Hyperparameter optimization complete![/green]")
        console.print(f"[green]Best trial: {study.best_trial.number}[/green]")
        console.print(f"[green]Best F0.5: {study.best_value:.4f}[/green]")
        
        return study
    
    def load_model_safely(self, checkpoint_path: str):
        """Load model from checkpoint safely"""
        try:
            safe_path = safe_load_checkpoint(checkpoint_path)
            
            if os.path.isdir(safe_path):
                # Load from safetensors directory
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                model = AutoModelForSeq2SeqLM.from_pretrained(safe_path)
                tokenizer = AutoTokenizer.from_pretrained(safe_path)
                
                console.print(f"[green]✅ Model loaded safely from: {safe_path}[/green]")
                return model, tokenizer
            else:
                # Load Lightning checkpoint
                model = GECLightningModule.load_from_checkpoint(safe_path)
                console.print(f"[green]✅ Lightning model loaded from: {safe_path}[/green]")
                return model.model, model.tokenizer
                
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise

if __name__ == "__main__":
    # Example usage
    trainer = BaseTrainer(
        model_name="vinai/bartpho-syllable",
        hyperopt=True
    )
    
    trainer.train()
