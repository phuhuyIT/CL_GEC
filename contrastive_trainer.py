"""
Contrastive Learning trainer for GEC
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.progress import track
from data_utils import ContrastiveDataset, get_model_and_tokenizer
from evaluator import GECEvaluator
import optuna

console = Console()

class ContrastiveLoss(nn.Module):
    """Contrastive loss for GEC with R-Drop regularization"""
    
    def __init__(self, temperature: float = 0.25, lambda_cl: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_cl = lambda_cl
        
    def forward(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        positive_labels: torch.Tensor,
        ce_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive loss
        
        Args:
            positive_logits: [batch_size, seq_len, vocab_size]
            negative_logits: [batch_size, num_negatives, seq_len, vocab_size]
            positive_labels: [batch_size, seq_len]
            ce_loss: Cross-entropy loss
        """
        
        batch_size, seq_len, vocab_size = positive_logits.shape
        num_negatives = negative_logits.shape[1]
        
        # Calculate probabilities
        positive_probs = F.log_softmax(positive_logits / self.temperature, dim=-1)
        negative_probs = F.log_softmax(negative_logits / self.temperature, dim=-1)
        
        # Get probabilities for target tokens
        positive_scores = torch.gather(
            positive_probs, 
            dim=-1, 
            index=positive_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len]
        
        # For negatives, we want to minimize their probability of generating the positive target
        negative_scores = torch.gather(
            negative_probs,
            dim=-1,
            index=positive_labels.unsqueeze(1).unsqueeze(-1).expand(-1, num_negatives, -1, -1)
        ).squeeze(-1)  # [batch_size, num_negatives, seq_len]
        
        # Mask out padding tokens
        mask = (positive_labels != -100).float()
        
        # Calculate contrastive loss
        positive_scores = (positive_scores * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        negative_scores = (negative_scores * mask.unsqueeze(1)).sum(dim=2) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # Contrastive loss: maximize positive, minimize negative
        contrastive_loss = -positive_scores.mean() + negative_scores.mean()
        
        # Total loss
        total_loss = ce_loss + self.lambda_cl * contrastive_loss
        
        return total_loss, contrastive_loss

class RDropLoss(nn.Module):
    """R-Drop regularization loss"""
    
    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """
        Calculate R-Drop loss between two forward passes
        """
        # KL divergence between two distributions
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        
        kl_loss1 = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction='none')
        kl_loss2 = F.kl_div(p2, F.softmax(logits1, dim=-1), reduction='none')
        
        # Average KL divergence
        rdrop_loss = (kl_loss1 + kl_loss2) / 2
        
        return self.alpha * rdrop_loss.mean()

class ContrastiveLightningModule(L.LightningModule):
    """Lightning module for contrastive learning"""
    
    def __init__(
        self,
        base_model_path: str,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        lambda_cl: float = 1.0,
        temperature: float = 0.25,
        rdrop_alpha: float = 4.0,
        warmup_steps: int = 500,
        max_steps: int = 5000,        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load base model with proper setup (including ViT5 prefix)
        console.print(f"[yellow]Loading base model from {base_model_path}[/yellow]")
        self.model, self.tokenizer = get_model_and_tokenizer(base_model_path)
        
        # Loss functions
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=label_smoothing
        )
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=temperature,
            lambda_cl=lambda_cl
        )
        
        self.rdrop_loss = RDropLoss(alpha=rdrop_alpha)
        # Evaluator
        self.evaluator = GECEvaluator(self.tokenizer)
        
        # Track metrics
        self.best_f05 = 0.0
        
        # Store validation outputs for epoch end processing
        self.validation_step_outputs = []
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        """Training step with contrastive learning and R-Drop"""
        
        # Unpack batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        positive_ids = batch['positive_ids']
        positive_attention_mask = batch['positive_attention_mask']
        negative_ids = batch['negative_ids']  # [batch_size, num_negatives, seq_len]
        negative_attention_mask = batch['negative_attention_mask']
        
        batch_size, num_negatives, seq_len = negative_ids.shape
        
        # Forward pass for positive (first pass for R-Drop)
        positive_outputs1 = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=positive_ids
        )
        
        # Forward pass for positive (second pass for R-Drop)
        positive_outputs2 = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=positive_ids
        )
        
        # Cross-entropy loss (average of two passes)
        ce_loss = (positive_outputs1.loss + positive_outputs2.loss) / 2
        
        # R-Drop loss
        rdrop_loss = self.rdrop_loss(
            positive_outputs1.logits,
            positive_outputs2.logits
        )
        
        # Generate logits for negatives
        negative_logits = []
        for i in range(num_negatives):
            neg_outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=negative_ids[:, i, :]
            )
            negative_logits.append(neg_outputs.logits)
        
        # Stack negative logits
        negative_logits = torch.stack(negative_logits, dim=1)  # [batch_size, num_negatives, seq_len, vocab_size]
        
        # Contrastive loss
        total_loss, cl_loss = self.contrastive_loss(
            positive_outputs1.logits,
            negative_logits,
            positive_ids,
            ce_loss
        )
        
        # Add R-Drop loss
        final_loss = total_loss + rdrop_loss
        
        # Logging
        self.log('train_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True)
        self.log('train_cl_loss', cl_loss, on_step=True, on_epoch=True)
        self.log('train_rdrop_loss', rdrop_loss, on_step=True, on_epoch=True)
        
        return final_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        positive_ids = batch['positive_ids']
        
        # Forward pass
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=positive_ids
        )
        
        val_loss = outputs.loss
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=384,
                num_beams=1,  # Use greedy decoding for speed
                early_stopping=True,
                do_sample=False
            )
        
        # Decode predictions
        predictions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        targets = batch['positive_text']
        sources = batch['source_text']
          # Calculate F0.5
        f05_scores = []
        for pred, target, source in zip(predictions, targets, sources):
            f05 = self.evaluator.f05_evaluator.calculate_f05(source, pred, target)
            f05_scores.append(f05)
        
        avg_f05 = np.mean(f05_scores)
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f05', avg_f05, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch end processing
        output = {
            'val_loss': val_loss,
            'val_f05': avg_f05,
            'predictions': predictions[:3],
            'targets': targets[:3],
            'sources': sources[:3]        }
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        """End of validation epoch"""
        outputs = self.validation_step_outputs
        
        if not outputs:
            return
            
        avg_f05 = torch.stack([torch.tensor(x['val_f05']) for x in outputs]).mean()
        
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
                if hasattr(self.logger, 'experiment'):
                    self.logger.experiment.log({
                        f"val_example_{i}_source": src,
                        f"val_example_{i}_prediction": pred,
                        f"val_example_{i}_target": tgt
                    })
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        
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

class ContrastiveTrainer:
    """Main trainer for contrastive learning"""
    
    def __init__(
        self,
        base_model_path: str,
        contrastive_data_dir: str = "./data/contrastive",
        output_dir: str = "./models/contrastive",
        hyperopt: bool = False
    ):
        self.base_model_path = base_model_path
        self.contrastive_data_dir = contrastive_data_dir
        self.output_dir = output_dir
        self.hyperopt = hyperopt
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_contrastive_data(self) -> Dict[str, List[Dict]]:
        """Load contrastive dataset"""
        
        console.print("[yellow]Loading contrastive data...[/yellow]")
        
        data = {}
        for split in ['train', 'validation']:
            file_path = os.path.join(self.contrastive_data_dir, f"{split}_contrastive.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[split] = json.load(f)
                console.print(f"[green]Loaded {split}: {len(data[split])} samples[/green]")
        
        return data
    
    def create_contrastive_dataloaders(
        self,
        data: Dict[str, List[Dict]],
        tokenizer: AutoTokenizer,
        batch_size: int = 8
    ) -> Dict[str, DataLoader]:
        """Create contrastive data loaders"""
        
        dataloaders = {}
        
        for split, split_data in data.items():
            dataset = ContrastiveDataset(
                data=split_data,
                tokenizer=tokenizer,
                max_length=384
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=4,
                pin_memory=True,
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
        
        return dataloaders
    
    def optimize_hyperparameters(
        self,
        dataloaders: Dict[str, DataLoader],
        n_trials: int = 20
    ) -> Dict:
        """Optimize contrastive learning hyperparameters"""
        
        def objective(trial):
            # Suggest parameters
            lambda_cl = trial.suggest_float('lambda_cl', 0.1, 2.0)
            temperature = trial.suggest_float('temperature', 0.1, 1.0)
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            rdrop_alpha = trial.suggest_float('rdrop_alpha', 1.0, 8.0)
            
            # Create model
            model = ContrastiveLightningModule(
                base_model_path=self.base_model_path,
                learning_rate=learning_rate,
                lambda_cl=lambda_cl,
                temperature=temperature,
                rdrop_alpha=rdrop_alpha,
                max_steps=len(dataloaders['train']) * 2  # 2 epochs for hyperopt
            )
            
            # Logger
            wandb_logger = WandbLogger(
                project="vigec-contrastive-hyperopt",
                name=f"trial_{trial.number}"
            )
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_f05',
                patience=2,
                mode='max'
            )
            
            # Trainer
            trainer = L.Trainer(
                max_epochs=2,
                logger=wandb_logger,
                callbacks=[early_stopping],
                enable_progress_bar=False,
                accelerator='auto',
                precision='16-mixed' if torch.cuda.is_available() else 32
            )
            
            try:
                trainer.fit(
                    model,
                    train_dataloaders=dataloaders['train'],
                    val_dataloaders=dataloaders['validation']
                )
                
                best_f05 = model.best_f05
                wandb.finish()
                return best_f05
                
            except Exception as e:
                console.print(f"[red]Trial {trial.number} failed: {e}[/red]")
                wandb.finish()
                return 0.0
        
        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            study_name="contrastive_hyperopt"
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train(self):
        """Full contrastive training pipeline"""
        
        console.print("[bold green]Starting contrastive learning training[/bold green]")
        
        # Load contrastive data
        data = self.load_contrastive_data()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        # Create data loaders
        dataloaders = self.create_contrastive_dataloaders(
            data, tokenizer, batch_size=8
        )
        
        best_params = {}
        
        # Hyperparameter optimization
        if self.hyperopt:
            console.print("[yellow]Optimizing contrastive learning hyperparameters...[/yellow]")
            best_params = self.optimize_hyperparameters(dataloaders)
            
            with open(os.path.join(self.output_dir, "best_cl_params.json"), "w") as f:
                json.dump(best_params, f, indent=2)
        
        # Final training
        console.print("[yellow]Training final contrastive model...[/yellow]")
        
        model_config = {
            'base_model_path': self.base_model_path,
            'learning_rate': best_params.get('learning_rate', 3e-5),
            'lambda_cl': best_params.get('lambda_cl', 1.0),
            'temperature': best_params.get('temperature', 0.25),
            'rdrop_alpha': best_params.get('rdrop_alpha', 4.0),
            'max_steps': len(dataloaders['train']) * 5  # 5 epochs
        }
        
        final_model = ContrastiveLightningModule(**model_config)
        
        # Logger
        wandb_logger = WandbLogger(
            project="vigec-contrastive-training",
            name="final_contrastive_model"
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_f05',
            patience=3,
            mode='max',
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            monitor='val_f05',
            mode='max',
            save_top_k=3,
            filename='contrastive_model_{epoch:02d}_{val_f05:.4f}'
        )
        
        # Trainer
        trainer = L.Trainer(
            max_epochs=5,
            logger=wandb_logger,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator='auto',
            precision='16-mixed' if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0
        )
        
        # Train
        trainer.fit(
            final_model,
            train_dataloaders=dataloaders['train'],
            val_dataloaders=dataloaders['validation']
        )
        
        # Save final model
        final_model.model.save_pretrained(os.path.join(self.output_dir, "final"))
        final_model.tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))
        
        console.print(f"[green]Contrastive training completed! Model saved to {self.output_dir}[/green]")
        
        wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to base model")
    parser.add_argument("--contrastive_data", default="./data/contrastive", help="Contrastive data directory")
    parser.add_argument("--output_dir", default="./models/contrastive", help="Output directory")
    parser.add_argument("--hyperopt", action="store_true", help="Run hyperparameter optimization")
    
    args = parser.parse_args()
    
    trainer = ContrastiveTrainer(
        base_model_path=args.base_model,
        contrastive_data_dir=args.contrastive_data,
        output_dir=args.output_dir,
        hyperopt=args.hyperopt
    )
    
    trainer.train()
