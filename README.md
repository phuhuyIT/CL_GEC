# Vietnamese GEC with Contrastive Learning

**Clean & Optimized**: Vietnamese Grammatical Error Correction using BARTpho/ViT5 with Contrastive Learning.

## 🚀 Quick Start (Google Colab)

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Upload the notebook**: Upload `run_colab.ipynb` to Colab
3. **Update the repository URL**: In the third cell, change `REPO_URL` to your GitHub repository URL
4. **Run all cells**: The notebook will handle everything automatically

## 📁 Project Structure

```
CL_GEC/
├── run_colab.ipynb          # 🎯 Main Colab notebook (START HERE)
├── data_utils.py            # 📊 Data loading and preprocessing
├── base_trainer.py          # 🏋️ Base model training with Optuna
├── negative_sampler.py      # 🎯 Negative sample generation
├── contrastive_trainer.py   # 🔥 Contrastive learning training
├── inference.py             # 🧪 Model inference
├── evaluator.py             # 📈 Model evaluation
├── config.py                # ⚙️ Configuration settings
├── main.py                  # 🖥️ Local training script
├── evaluate_model.py        # 📊 Standalone evaluation
├── README.md                # 📖 This file
└── IMPORT_FIXES.md          # 🔧 Documentation of fixes applied
```

## 🐛 Recent Fixes

- ✅ **AdamW Import Error**: Fixed `ImportError: cannot import name 'AdamW' from 'transformers'`
- ✅ **BARTpho Tokenizer**: Fixed `'BartphoTokenizer' object has no attribute 'vocab'` error
- ✅ **Clean Codebase**: Removed redundant files and code duplication
- ✅ **Colab Optimization**: Simple notebook that clones from GitHub

## 🔧 Dependencies

All dependencies are automatically installed in the Colab notebook:

- **Core**: PyTorch, Transformers, Datasets
- **Training**: PyTorch Lightning, Optuna, Wandb
- **Vietnamese NLP**: SentencePiece, underthesea
- **Evaluation**: NLTK, SacreBLEU, ROUGE

## 🎯 Pipeline Overview

1. **Setup**: Install dependencies and clone repository
2. **Data**: Load viGEC dataset and preprocess
3. **Base Training**: Fine-tune BARTpho/ViT5 on GEC task
4. **Negative Sampling**: Generate negative samples for contrastive learning
5. **Contrastive Training**: Apply contrastive loss + R-Drop regularization
6. **Evaluation**: Test model performance and export results

## ⚙️ Configuration

Key parameters can be adjusted in the Colab notebook:

```python
TRAINING_CONFIG = {
    "model_name": "vinai/bartpho-syllable",  # or "VietAI/vit5-base"
    "max_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "use_wandb": True,
    "run_optimization": False  # Set True for hyperparameter tuning
}
```

## 🎮 Models Supported

- **BARTpho**: `vinai/bartpho-syllable` (Recommended)
- **ViT5**: `VietAI/vit5-base`, `VietAI/vit5-large`

## 📊 Expected Results

- **Training Time**: 4-9 hours (depending on GPU and configuration)
- **Memory Requirements**: 8GB+ GPU memory for base models
- **Output**: Trained models, evaluation metrics, exportable results

## 🔍 Local Development

For local development (not recommended, use Colab instead):

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python main.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test in Colab
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Support

- **Issues**: Open a GitHub issue
- **Questions**: Check the Colab notebook comments
- **Fixes**: All known issues documented in `IMPORT_FIXES.md`

---

**Ready to train Vietnamese GEC models? Start with `run_colab.ipynb`! 🚀**