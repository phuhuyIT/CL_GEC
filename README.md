# Vietnamese GEC with Contrastive Learning

A complete implementation of Vietnamese Grammatical Error Correction using Contrastive Learning, based on the research paper "Grammatical Error Correction with Contrastive Learning in Low-Error-Density Domains".

## 🎯 Overview

This project implements a comprehensive pipeline for training Vietnamese GEC models with:

- **Base Model Training**: Fine-tuning BARTpho/ViT5 with hyperparameter optimization
- **Negative Sample Generation**: Creating diverse negative samples for contrastive learning
- **Contrastive Learning**: Training with contrastive loss + R-Drop regularization
- **Contrastive Search**: Advanced inference strategy for better quality
- **Comprehensive Evaluation**: F0.5, BLEU, IE/OE analysis, and error type evaluation

## 📁 Project Structure

```
CL_GEC/
├── data_utils.py              # Data loading and preprocessing
├── base_trainer.py            # Base model training with hyperparameter optimization
├── negative_sampler.py        # Negative sample generation
├── contrastive_trainer.py     # Contrastive learning training
├── inference.py               # Inference with contrastive search
├── evaluator.py              # Evaluation metrics (F0.5, BLEU, IE/OE)
├── evaluate_model.py          # Complete model evaluation
├── run_colab.ipynb           # Google Colab notebook
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open `run_colab.ipynb` in Google Colab
2. Run all cells sequentially
3. The notebook will guide you through the complete pipeline

### Option 2: Local Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Data preparation**:
```python
from data_utils import load_vigec_dataset, save_processed_data

# Load and process data
data = load_vigec_dataset("phuhuy-se1/viGEC")
save_processed_data(data, "./data/processed")
```

3. **Base model training**:
```python
from base_trainer import BaseTrainer

trainer = BaseTrainer(
    model_name="vinai/bartpho-syllable",
    hyperopt=True
)
trainer.train()
```

4. **Generate negative samples**:
```bash
python negative_sampler.py --model_path ./models/base/final --data_dir ./data/processed --output_dir ./data/contrastive
```

5. **Contrastive learning training**:
```bash
python contrastive_trainer.py --base_model ./models/base/final --contrastive_data ./data/contrastive --output_dir ./models/contrastive --hyperopt
```

6. **Inference**:
```python
from inference import GECInference

gec = GECInference(
    model_path="./models/contrastive/final",
    use_contrastive_search=True
)

corrected = gec.correct_text("Tôi đi học trường đại học.")
```

7. **Evaluation**:
```bash
python evaluate_model.py --model_path ./models/contrastive/final --data_dir ./data/processed --output_dir ./evaluation_results --error_analysis
```

## 🔧 Configuration

### Base Training Parameters
- **Model**: BARTpho-syllable, BARTpho-word, ViT5-base, ViT5-large
- **Learning Rate**: Auto-optimized (1e-6 to 1e-3)
- **Label Smoothing**: 0.1
- **Max Length**: 384 tokens
- **Epochs**: 5-10

### Contrastive Learning Parameters
- **λ (lambda_cl)**: 1.0 (CE vs CL loss balance)
- **γ (temperature)**: 0.25 (contrastive temperature)
- **R-Drop α**: 4.0 (regularization strength)
- **Epochs**: 3-5

### Contrastive Search Parameters
- **α (alpha)**: 0.7 (confidence vs diversity)
- **k**: 5 (top-k candidates)
- **p**: 0.7 (nucleus sampling parameter)

## 📊 Evaluation Metrics

The system evaluates models using:

- **F0.5 Score**: Primary metric focusing on precision
- **BLEU Score**: Translation quality metric
- **ROUGE-1/L**: Text similarity metrics
- **IE/OE Ratio**: Input-preserving vs Over-correction analysis
- **Precision/Recall**: Edit-level accuracy

## 🎯 Supported Models

- **BARTpho**: `vinai/bartpho-syllable`, `vinai/bartpho-word`
- **ViT5**: `VietAI/vit5-base`, `VietAI/vit5-large`
- **Custom models**: Any Seq2Seq model compatible with Transformers

## 📈 Results

Expected improvements with contrastive learning:
- **F0.5 Score**: +2-5% improvement
- **Over-correction Reduction**: 20-40% reduction in OE ratio
- **Better Quality**: More natural and accurate corrections

## 🛠️ Advanced Usage

### Custom Hyperparameter Optimization

```python
from base_trainer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    model_name="vinai/bartpho-syllable",
    data_loaders=data_loaders,
    n_trials=50  # Increase for better optimization
)
study = optimizer.optimize()
```

### Batch Inference

```python
from inference import GECInference

gec = GECInference(model_path="./models/contrastive/final")

# Batch correction
texts = ["Text 1", "Text 2", "Text 3"]
corrected = gec.correct_batch(texts, batch_size=8)

# File correction
gec.correct_file("input.txt", "output.txt")
```

### Custom Evaluation

```python
from evaluator import GECEvaluator

evaluator = GECEvaluator()
report = evaluator.generate_report(
    sources=sources,
    predictions=predictions,
    targets=targets,
    output_path="custom_evaluation.json"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable fp16 training

2. **Slow Training**:
   - Use fp16 precision
   - Increase batch size if memory allows
   - Use DataLoader with multiple workers

3. **Poor Results**:
   - Increase hyperparameter optimization trials
   - Check data quality
   - Adjust contrastive learning parameters

### Memory Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **CPU**: 16GB+ RAM for data processing

## 📚 References

- **Paper**: "Grammatical Error Correction with Contrastive Learning in Low-Error-Density Domains"
- **Dataset**: [viGEC](https://huggingface.co/datasets/phuhuy-se1/viGEC)
- **BARTpho**: [VinAI Research](https://github.com/VinAIResearch/BARTpho)
- **ViT5**: [VietAI](https://huggingface.co/VietAI)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VinAI Research for BARTpho
- VietAI for ViT5
- Hugging Face for Transformers library
- The authors of the contrastive learning paper

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the Colab notebook for examples

---

**Happy Training! 🚀**