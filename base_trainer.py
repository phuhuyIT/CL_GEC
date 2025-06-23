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
        
        # Generate predictions for evaluation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=384,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode predictions and targets
        predictions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        targets = batch['target_text']
        sources = batch['source_text']
        
        # Calculate F0.5
        f05_scores = []
        for pred, target, source in zip(predictions, targets, sources):
            f05 = self.evaluator.calculate_f05(source, pred, target)
            f05_scores.append(f05)
        
        avg_f05 = np.mean(f05_scores)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f05', avg_f05, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'val_f05': avg_f05,
            'predictions': predictions[:5],  # Log first 5 predictions
            'targets': targets[:5],
            'sources': sources[:5]
        }
    
    def validation_epoch_end(self, outputs):
        avg_f05 = torch.stack([x['val_f05'] for x in outputs]).mean()
        
        if avg_f05 > self.best_f05:
            self.best_f05 = avg_f05
            console.print(f"[green]New best F0.5: {self.best_f05:.4f}[/green]")
        
        # Log examples
        if outputs:
            for i, (src, pred, tgt) in enumerate(zip(
                outputs[0]['sources'],
                outputs[0]['predictions'], 
                outputs[0]['targets']
            )):
                self.logger.experiment.log({
                    f"example_{i}_source": src,
                    f"example_{i}_prediction": pred,
                    f"example_{i}_target": tgt
                })
    
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
        
        # Calculate steps
        train_steps_per_epoch = len(self.data_loaders['train'])
        max_epochs = 5  # Coarse search with fewer epochs
        max_steps = train_steps_per_epoch * max_epochs
        warmup_steps = int(max_steps * warmup_ratio)
        
        # Create model
        model = GECLightningModule(
            model_name=self.model_name,
            learning_rate=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            warmup_steps=warmup_steps,
            max_steps=max_steps        )
        
        # Logger
        if self.use_wandb:
            wandb_logger = WandbLogger(
                project="vigec-hyperopt",
                name=f"trial_{trial.number}",
                config=trial.params
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
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[early_stopping, pruning_callback, checkpoint_callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='auto',
            precision='16-mixed' if torch.cuda.is_available() else 32
        )
        
        try:
            # Train
            trainer.fit(
                model,
                train_dataloaders=self.data_loaders['train'],
                val_dataloaders=self.data_loaders['validation']
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
        
    def train(self):
        """Full training pipeline"""
        
        console.print("[bold green]Starting base model training[/bold green]")
        
        # Load data
        console.print("[yellow]Loading data...[/yellow]")
        if os.path.exists(self.data_dir):
            from data_utils import load_processed_data
            data = load_processed_data(self.data_dir)
        else:
            data = load_vigec_dataset()
            from data_utils import save_processed_data
            save_processed_data(data, self.data_dir)
        
        # Get model and tokenizer
        model, tokenizer = get_model_and_tokenizer(self.model_name)
          # Create data loaders
        data_loaders = create_data_loaders(
            data, tokenizer, batch_size=16, max_length=384
        )
        
        best_params = None
        
        # Hyperparameter optimization
        if self.hyperopt:
            console.print("[yellow]Running hyperparameter optimization...[/yellow]")
            
            optimizer = HyperparameterOptimizer(
                model_name=self.model_name,
                data_loaders=data_loaders,
                n_trials=30,
                use_wandb=self.use_wandb
            )
            
            study = optimizer.optimize()
            best_params = study.best_params
            
            # Save best parameters
            with open(os.path.join(self.output_dir, "best_params.json"), "w") as f:
                json.dump(best_params, f, indent=2)
        
        # Final training with best parameters
        console.print("[yellow]Training final model with best parameters...[/yellow]")
        
        if best_params:
            # Use best parameters
            model_config = {
                'model_name': self.model_name,
                **best_params,
                'max_steps': len(data_loaders['train']) * 10,  # 10 epochs
                'warmup_steps': int(len(data_loaders['train']) * 10 * 0.1)
            }
        else:
            # Use default parameters
            model_config = {
                'model_name': self.model_name,
                'learning_rate': 5e-5,
                'weight_decay': 0.01,
                'label_smoothing': 0.1,
                'max_steps': len(data_loaders['train']) * 10,
                'warmup_steps': int(len(data_loaders['train']) * 10 * 0.1)
            }
          # Create final model
        final_model = GECLightningModule(**model_config)
        
        # Logger
        if self.use_wandb:
            wandb_logger = WandbLogger(
                project="vigec-base-training",
                name="final_model"
            )
        else:
            wandb_logger = None
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_f05',
            patience=5,
            mode='max',
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            monitor='val_f05',
            mode='max',
            save_top_k=3,
            filename='base_model_{epoch:02d}_{val_f05:.4f}'
        )
        
        # Trainer
        trainer = L.Trainer(
            max_epochs=10,
            logger=wandb_logger,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator='auto',
            precision='16-mixed' if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0
        )
        
        # Train
        trainer.fit(
            final_model,
            train_dataloaders=data_loaders['train'],
            val_dataloaders=data_loaders['validation']
        )
          # Save final model
        final_model.model.save_pretrained(os.path.join(self.output_dir, "final"))
        final_model.tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))
        
        console.print(f"[green]Base model training completed! Model saved to {self.output_dir}[/green]")
        
        if self.use_wandb:
            wandb.finish()
    
    def train_with_params(
        self,
        data: Dict[str, List[Dict]],
        max_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 5e-5
    ):
        """Training method that accepts parameters from Colab notebook"""
        console.print("[bold green]Starting base model training with custom parameters[/bold green]")
        
        # Get model and tokenizer
        model, tokenizer = get_model_and_tokenizer(self.model_name)
        
        # Create data loaders with custom batch size
        data_loaders = create_data_loaders(
            data, tokenizer, batch_size=batch_size, max_length=384
        )
        
        # Calculate steps
        train_steps_per_epoch = len(data_loaders['train'])
        max_steps = train_steps_per_epoch * max_epochs
        warmup_steps = int(max_steps * 0.1)  # 10% warmup
        
        # Model configuration
        model_config = {
            'model_name': self.model_name,
            'learning_rate': learning_rate,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'max_steps': max_steps,
            'warmup_steps': warmup_steps
        }
        
        # Create model
        final_model = GECLightningModule(**model_config)
        
        # Logger
        if self.use_wandb:
            wandb_logger = WandbLogger(
                project="vigec-base-training",
                name="custom_params_model"
            )
        else:
            wandb_logger = None
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_f05',
            patience=5,
            mode='max',
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            monitor='val_f05',
            mode='max',
            save_top_k=3,
            filename='base_model_{epoch:02d}_{val_f05:.4f}'
        )
        
        # Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator='auto',
            precision='16-mixed' if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0
        )
        
        # Train
        trainer.fit(
            final_model,
            train_dataloaders=data_loaders['train'],
            val_dataloaders=data_loaders['validation']
        )
        
        # Save final model
        final_model.model.save_pretrained(os.path.join(self.output_dir, "final"))
        final_model.tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))
        
        console.print(f"[green]Base model training completed! Model saved to {self.output_dir}[/green]")
        
        if self.use_wandb:
            wandb.finish()
    
    def optimize_hyperparameters(
        self,
        data: Dict[str, List[Dict]],
        n_trials: int = 10,
        timeout: int = 3600
    ):
        """Public method for hyperparameter optimization that the Colab notebook can call"""
        console.print(f"[bold blue]Starting hyperparameter optimization with {n_trials} trials[/bold blue]")
        
        # Get model and tokenizer
        model, tokenizer = get_model_and_tokenizer(self.model_name)
        
        # Create data loaders
        data_loaders = create_data_loaders(
            data, tokenizer, batch_size=16, max_length=384
        )
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            model_name=self.model_name,
            data_loaders=data_loaders,
            n_trials=n_trials,
            use_wandb=self.use_wandb
        )
        
        # Run optimization
        study = optimizer.optimize(study_name="vigec_colab_hyperopt")
        
        # Save best parameters
        with open(os.path.join(self.output_dir, "best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=2)
            
        console.print(f"[green]✅ Hyperparameter optimization completed![/green]")
        console.print(f"[green]Best F0.5: {study.best_value:.4f}[/green]")
        
        return study.best_params

if __name__ == "__main__":
    # Example usage
    trainer = BaseTrainer(
        model_name="vinai/bartpho-syllable",
        hyperopt=True
    )
    
    trainer.train()
