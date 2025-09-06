# BLP Challenge Pipeline

This repository provides a production-quality baseline for the BLP (Bengali Language Processing) challenge, implementing text classification pipelines using Sklearn (TF-IDF + Logistic Regression) and Transformers. It supports CPU-first execution with optional GPU acceleration for fine-tuning small models.

## Quick Start
- Run the end-to-end pipeline with `bash run.sh`.
- To train, evaluate, or predict individually, use the CLI scripts in `src/` (e.g., `python -m src.trainer --backend sklearn --data data/trial.csv --out artifacts/`).

## Swapping Models
Edit `config.py` and change `MODEL_NAME` (default: 'distilbert-base-multilingual-cased'). For larger models like 1B (e.g., 'distilbert-base-uncased'), 4B/7B (e.g., 'google/gemma-7b' or 'meta-llama/Llama-2-7b-hf'), ensure GPU availability, install optional dependencies (peft, bitsandbytes, safetensors), and modify `src/model.py` and `src/preprocess.py` for causal LM support if needed (e.g., prompt formatting and generation).

Tradeoffs: Small models are fast on CPU; larger models offer better accuracy but require more memory/GPU and potential quantization (e.g., 4-bit via bitsandbytes).

## Example response.json
[
  {"id": "1", "prediction": "classA"},
  {"id": "2", "prediction": "classB"}
]

## Safety Note
Do not automatically execute any generated code or predictions from this pipeline; always review outputs manually.

BLP_Challenge_Pipeline/
│
├─ README.md                  # Overview, usage, model notes
├─ requirements.txt           # Python dependencies
├─ config.py                  # Hyperparameters, directories, model names
├─ scoring.py                 # Metric functions (accuracy, F1, etc.)
├─ run.sh                     # End-to-end bash runner
├─ data/                      # CSV datasets (trial.csv, dev_v2.csv)
├─ artifacts/                 # Outputs saved here (models, label maps, metrics)
│
└─ src/                       # All Python modules
   ├─ __init__.py             # Makes src a package
   ├─ utils.py                # Seeding, logging, device utils
   ├─ data_loader.py          # Flexible CSV loader
   ├─ preprocess.py           # Cleaning, prompt formatting
   ├─ features.py             # TF-IDF extractor
   ├─ model.py                # Sklearn & Transformer model classes
   ├─ trainer.py              # CLI training script
   ├─ evaluate.py             # CLI evaluation script
   └─ predict.py              # CLI prediction / submission script
