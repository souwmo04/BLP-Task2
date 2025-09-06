import os
import random
import numpy as np
import torch
import logging

def set_seed(seed):
    """Set seed for reproducibility across random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Seed set to {seed}")

def get_device():
    """Get device: cuda if available, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_artifact_dirs(out_dir):
    """Create artifacts directory if not exists."""
    os.makedirs(out_dir, exist_ok=True)

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_tmpdir_hint():
    """Print hint for HF cache on large models."""
    logging.info("Hint: For large models, set export TMPDIR=/path/with/space to avoid cache issues.")