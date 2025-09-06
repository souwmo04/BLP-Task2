# config.py
"""
Configuration tuned for CodeLlama-7B instruct with cautious defaults.
You must be authenticated (hf auth login) to access gated models like CodeLlama.
For low-VRAM local fine-tuning prefer:
  - small LoRA rank (LORA_R)
  - gradient accumulation to simulate larger batch size
  - bitsandbytes 4-bit quantization enabled
"""

SEED = 1337

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"

# Model: CodeLlama 7B instruct (gated). Must be accessible with hf auth token.
MODEL_NAME = "meta-llama/CodeLlama-7b-Instruct-hf"

# Auth: by default transformers will use saved HF token from `hf auth login`.
# If you prefer to explicitly set token string, export HF_TOKEN env var and pass it to loader.
USE_AUTH_TOKEN = True   # pass directly to from_pretrained (True uses saved token)

# Causal LM / tokenization
IS_CAUSAL = True
MAX_LENGTH = 1024

# Inference & training batch sizes (keep tiny for low VRAM)
BATCH_SIZE = 1                   # inference batch size
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # per-device train batch (LoRA)
GRAD_ACCUM_STEPS = 8             # effective batch size (accumulation)
EPOCHS_SMALL = 3
EPOCHS_FULL = 5

# Generation hyperparams
GEN_MAX_NEW_TOKENS = 512
GEN_TEMPERATURE = 0.2
GEN_TOP_P = 0.95
GEN_TOP_K = 50
GEN_NUM_BEAMS = 1

# LoRA / PEFT defaults (reduce r for low VRAM)
USE_PEFT = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Scikit-learn fallback (not used for code gen)
TFIDF_PARAMS = {'ngram_range': (1, 2), 'max_features': 5000}
LR_PARAMS = {'C': [0.1, 1, 10]}

# Training / logging
LOGGING_STEPS = 20
SAVE_TOTAL_LIMIT = 3
