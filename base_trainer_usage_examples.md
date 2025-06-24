# Updated usage example for the improved BaseTrainer with dataset parameters

## Example 1: Basic training with dataset parameters
```python
# Configure training with dataset parameters
base_trainer = BaseTrainer(
    model_name="vinai/bartpho-syllable",
    data_dir="./data/processed",
    output_dir="./models/base_model",
    hyperopt=True,
    use_wandb=True,
    dataset_name="phuhuy-se1/viGEC-v2",  # Use version 2 of dataset
    train_subset_ratio=0.3,  # Use 30% of training data for faster training
    validation_subset_ratio=0.5,  # Use 50% of validation data  
    test_subset_ratio=0.1   # Use 10% of test data
)

# Train with custom search space for hyperparameter optimization
custom_search_space = {
    'learning_rate': {'low': 1e-5, 'high': 5e-4, 'log': True},
    'weight_decay': {'low': 0.001, 'high': 0.05, 'log': True},
    'label_smoothing': {'low': 0.0, 'high': 0.2},
    'batch_size': [16, 32, 48],  # Smaller batch sizes for limited GPU memory
    'warmup_ratio': {'low': 0.05, 'high': 0.15}
}

trained_model = base_trainer.train(
    max_epochs=5,
    batch_size=16,
    search_space=custom_search_space
)
```

## Example 2: Hyperparameter optimization only
```python
# Run only hyperparameter optimization with custom search space
study = base_trainer.optimize_hyperparameters(
    n_trials=20,
    batch_size=16,
    search_space=custom_search_space
)

print(f"Best F0.5: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")
```

## Example 3: Train with specific parameters (no hyperopt)
```python
# Train with specific parameters without hyperparameter optimization
base_trainer_no_hyperopt = BaseTrainer(
    model_name="vinai/bartpho-syllable",
    hyperopt=False,  # Disable hyperparameter optimization
    dataset_name="phuhuy-se1/viGEC-v2",
    train_subset_ratio=0.5,
    validation_subset_ratio=1.0,
    test_subset_ratio=0.1
)

# Custom parameters from previous optimization
best_params = {
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'label_smoothing': 0.1,
    'batch_size': 32,
    'warmup_ratio': 0.1
}

trained_model = base_trainer_no_hyperopt.train_with_params(
    params=best_params,
    max_epochs=10,
    batch_size=32
)
```
