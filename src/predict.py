# src/predict.py
import argparse
import json
import os
import logging
import joblib
import torch

from src.utils import set_seed, setup_logging, get_device
from src.data_loader import load_csv
from src.preprocess import preprocess_texts, build_prompt
from src.model import TransformerModel
import config

def main():
    parser = argparse.ArgumentParser(description="Generate predictions (Bangla->Python) using a causal LM.")
    parser.add_argument('--data', required=True, help="Path to dev/test CSV")
    parser.add_argument('--out', required=True, help="Output JSON path (submission file)")
    parser.add_argument('--artifacts', required=True, help="Artifacts dir (trained model)")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="Generation batch size")
    args = parser.parse_args()

    setup_logging()
    set_seed(config.SEED)
    logging.info(f"Device: {get_device()}")

    # Load artifacts (we expect model/tokenizer saved in artifacts dir)
    model_dir = args.artifacts
    transformer = TransformerModel(model_name=model_dir, max_length=config.MAX_LENGTH, use_peft=config.USE_PEFT)

    # Load dev/test CSV
    df = load_csv(args.data, require_target=False)
    ids = df['id'].tolist()
    test_lists = df.get('test_list', [None]*len(df)).tolist()
    prompts = [build_prompt(p, t) for p, t in zip(df['prompt'], test_lists)]
    prompts = preprocess_texts(prompts)

    # Generate
    logging.info(f"Generating {len(prompts)} outputs (batch_size={args.batch_size})")
    predictions = transformer.generate_code(
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=config.GEN_MAX_NEW_TOKENS,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        top_k=config.GEN_TOP_K,
        num_beams=config.GEN_NUM_BEAMS
    )

    # Post-process: ensure returned strings are function definitions (best-effort strip)
    cleaned = []
    for pred in predictions:
        s = pred.strip()
        # If model returned extra commentary before function, try to find 'def ' and keep from there
        idx = s.find("\ndef ")
        if idx != -1:
            s = s[idx+1:].strip()
        elif s.startswith("def ") or s.startswith("async def "):
            pass
        else:
            # try to find 'def ' anywhere
            i = s.find("def ")
            if i != -1:
                s = s[i:].strip()
        cleaned.append(s)

    # Build competition submission: array of objects with integer id and response
    response = [{"id": int(i), "response": cleaned_j} for i, cleaned_j in zip(ids, cleaned)]

    # Validate
    if len(set([r['id'] for r in response])) != len(response):
        raise ValueError("Duplicate IDs in output")
    if not all(isinstance(r['response'], str) for r in response):
        raise ValueError("Responses must be strings")

    # Save
    out_path = args.out
    with open(out_path, 'w') as f:
        json.dump(response, f, indent=2)

    logging.info(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()
