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
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import wandb
from rich.console import Console
from rich.progress import track
import numpy as np
from typing import Dict, List, Optional, Any
from data_utils import load_vigec_dataset, create_data_loaders, get_model_and_tokenizer
from evaluator import F05Evaluator

console = Console()

class GECLightningModule(L.LightningModule):
    """Lightning module for GEC base training"""
    
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
        
        # Load model and tokenizer
        self.model, self.tokenizer = get_model_and_tokenizer(model_name)
        
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
        
        # Only do expensive generation and F0.5 calculation for a small subset
        # This dramatically speeds up validation while still providing metrics
        do_generation = (
            batch_idx == 0 or  # Always do first batch for examples
            batch_idx % 50 == 0  # Every 50th batch for periodic F0.5 monitoring
        )
        
        if do_generation:
            # Generate predictions for evaluation (much faster settings)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=384,  
                    num_beams=3,     # Reduced from 5
                    early_stopping=True,
                    do_sample=False
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
            
            if avg_f05 > self.best_f05:
                self.best_f05 = avg_f05
                console.print(f"[green]New best F0.5: {self.best_f05:.4f}[/green]")
            
            # Log examples from the first output with generation
            first_gen_output = f05_outputs[0]
            for i, (src, pred, tgt) in enumerate(zip(
                first_gen_output['sources'],
                first_gen_output['predictions'], 
                first_gen_output['targets']
            )):
                if hasattr(self.logger, 'experiment'):
                    self.logger.experiment.log({
                        f"example_{i}_source": src,
                        f"example_{i}_prediction": pred,
                        f"example_{i}_target": tgt
                    })
        
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
        use_wandb: bool = True
    ):
        self.model_name = model_name
        self.data_loaders = data_loaders
        self.n_trials = n_trials
        self.direction = direction
        self.use_wandb = use_wandb
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
        
        # Recreate data loaders with the suggested batch size
        # This is crucial - batch size affects convergence and memory usage
        try:
            from data_utils import create_data_loaders, get_model_and_tokenizer
            # Need to get tokenizer to recreate data loaders
            _, tokenizer = get_model_and_tokenizer(self.model_name)
            
            # Load data fresh for this trial
            from data_utils import load_vigec_dataset
            data = load_vigec_dataset()
            
            # Create data loaders with trial-specific batch size
            trial_data_loaders = create_data_loaders(
                data=data,
                tokenizer=tokenizer,
                batch_size=batch_size
            )
        except Exception as e:
            console.print(f"[red]Failed to create data loaders for trial {trial.number}: {e}[/red]")
            return 0.0
        
        # Calculate steps based on actual data loader length
        train_steps_per_epoch = len(trial_data_loaders['train'])
        max_epochs = 5  # Coarse search with fewer epochs
        max_steps = train_steps_per_epoch * max_epochs
        warmup_steps = int(max_steps * warmup_ratio)
        
        # Create model with all hyperparameters
        model = GECLightningModule(
            model_name=self.model_name,
            learning_rate=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            warmup_steps=warmup_steps,
            max_steps=max_steps
        )
        
        # Logger
        if self.use_wandb:
            wandb_logger = WandbLogger(
                project="vigec-hyperopt",
                name=f"trial_{trial.number}",
                config={
                    **trial.params,
                    'max_steps': max_steps,
                    'warmup_steps': warmup_steps,
                    'train_steps_per_epoch': train_steps_per_epoch
                }
            )
        else:
            wandb_logger = None
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_f05',
            patience=3,
            mode='max',
            verbose=True
        )
        
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor='val_f05'
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f05',
            mode='max',
            save_top_k=1,
            filename=f'trial_{trial.number}_best'
        )
        
        # Trainer
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[early_stopping, pruning_callback, checkpoint_callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='auto',
            precision=precision
        )
        
        try:
            # Train with trial-specific data loaders
            trainer.fit(
                model,
                train_dataloaders=trial_data_loaders['train'],
                val_dataloaders=trial_data_loaders['validation']
            )
            
            # Get best metric
            best_f05 = model.best_f05
            
            # Clean up
            if self.use_wandb:
                wandb.finish()
            
            return best_f05
            
        except Exception as e:
            console.print(f"[red]Trial {trial.number} failed: {e}[/red]")
            if self.use_wandb:
                wandb.finish()
            return 0.0
    
    def optimize(self, study_name: str = "vigec_hyperopt") -> optuna.Study:
        """Run hyperparameter optimization"""
        
        console.print(f"[bold blue]Starting hyperparameter optimization with {self.n_trials} trials[/bold blue]")
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=[
                lambda study, trial: console.print(
                    f"[yellow]Trial {trial.number}: F0.5 = {trial.value:.4f}[/yellow]"
                )
            ]
        )
        
        # Print results
        console.print(f"[green]Best trial: {study.best_trial.number}[/green]")
        console.print(f"[green]Best F0.5: {study.best_value:.4f}[/green]")
        console.print("[green]Best parameters:[/green]")
        for key, value in study.best_params.items():
            console.print(f"  {key}: {value}")
        
        return study

class BaseTrainer:
    """Main trainer for base model"""
    
    def __init__(
        self,
        model_name: str,
        data_dir: str = "./data/processed",
        output_dir: str = "./models/base",
        hyperopt: bool = True,
        use_wandb: bool = True
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.hyperopt = hyperopt
        self.use_wandb = use_wandb
        
        os.makedirs(output_dir, exist_ok=True)
        
    def _run_hyperopt(self, data_loaders, n_trials=10, study_name="vigec_hyperopt"):
        """Internal method to run hyperparameter optimization"""
        optimizer = HyperparameterOptimizer(
            model_name=self.model_name,
            data_loaders=data_loaders,
            n_trials=n_trials,
            use_wandb=self.use_wandb
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
        model = GECLightningModule(**model_config)
        
        # Logger
        wandb_logger = WandbLogger(project="vigec-base-training", name=run_name) if self.use_wandb else None
          # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_f05', patience=5, mode='max', verbose=True),
            ModelCheckpoint(
                dirpath=self.output_dir,
                monitor='val_f05',
                mode='max',
                save_top_k=3,
                filename=f'model_{run_name}_{{epoch:02d}}_{{val_f05:.4f}}'
            )
        ]
        
        # Trainer
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            accelerator='auto',
            precision=precision,
            gradient_clip_val=1.0
        )
        
        # Train
        trainer.fit(
            model,
            train_dataloaders=data_loaders['train'],
            val_dataloaders=data_loaders['validation']
        )
        
        # Save model
        model.model.save_pretrained(os.path.join(self.output_dir, run_name))
        model.tokenizer.save_pretrained(os.path.join(self.output_dir, run_name))
        
        if self.use_wandb:
            wandb.finish()
            
        return model
    
    def train(self, max_epochs: int = 10, batch_size: int = 16):
        """Main training method with optional hyperparameter optimization"""
        console.print(f"[bold blue]Starting training for {self.model_name}[/bold blue]")
        
        # Load data with proper data directory
        console.print("[yellow]Loading data...[/yellow]")
        try:
            # Pass data_dir if the function supports it
            import inspect
            load_func_signature = inspect.signature(load_vigec_dataset)
            if 'data_dir' in load_func_signature.parameters:
                data = load_vigec_dataset(data_dir=self.data_dir)
            else:
                data = load_vigec_dataset()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not use data_dir parameter: {e}[/yellow]")
            data = load_vigec_dataset()
        
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
            console.print("[yellow]Running hyperparameter optimization...[/yellow]")
            study = self._run_hyperopt(data_loaders, n_trials=10)
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
                'warmup_steps': int(len(data_loaders['train']) * max_epochs * 0.1)
            }
        
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
            import inspect
            load_func_signature = inspect.signature(load_vigec_dataset)
            if 'data_dir' in load_func_signature.parameters:
                data = load_vigec_dataset(data_dir=self.data_dir)
            else:
                data = load_vigec_dataset()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not use data_dir parameter: {e}[/yellow]")
            data = load_vigec_dataset()
        
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
    
    def optimize_hyperparameters(self, n_trials: int = 30, batch_size: int = 16):
        """Run only hyperparameter optimization"""
        console.print(f"[bold blue]Running hyperparameter optimization for {self.model_name}[/bold blue]")
        
        # Load data with proper data directory
        try:
            import inspect
            load_func_signature = inspect.signature(load_vigec_dataset)
            if 'data_dir' in load_func_signature.parameters:
                data = load_vigec_dataset(data_dir=self.data_dir)
            else:
                data = load_vigec_dataset()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not use data_dir parameter: {e}[/yellow]")
            data = load_vigec_dataset()
        
        # Get tokenizer for data loading
        _, tokenizer = get_model_and_tokenizer(self.model_name)
        
        # Create data loaders
        data_loaders = create_data_loaders(
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size
        )
        
        study = self._run_hyperopt(
            data_loaders, n_trials=n_trials, 
            study_name="vigec_standalone_hyperopt"
        )
        
        console.print(f"[green]Hyperparameter optimization complete![/green]")
        console.print(f"[green]Best trial: {study.best_trial.number}[/green]")
        console.print(f"[green]Best F0.5: {study.best_value:.4f}[/green]")
        
        return study

if __name__ == "__main__":
    # Example usage
    trainer = BaseTrainer(
        model_name="vinai/bartpho-syllable",
        hyperopt=True
    )
    
    trainer.train()
