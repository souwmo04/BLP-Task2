# src/trainer.py
import argparse
import json
import os
import logging
import numpy as np
from pathlib import Path
import torch

from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.utils import set_seed, setup_logging, make_artifact_dirs, get_device, print_tmpdir_hint
from src.data_loader import load_csv
from src.preprocess import preprocess_texts, build_prompt
from src.model import TransformerModel
import config

def prepare_causal_dataset(df, tokenizer, max_length):
    """
    Build dataset where each example is prompt + response. Labels are input_ids with prompt portion masked to -100.
    Returns a HuggingFace Dataset with input_ids, attention_mask, labels.
    """
    prompts = []
    targets = []
    for _, row in df.iterrows():
        instr = row['prompt']
        test_list = row.get('test_list', None)
        prompt_text = build_prompt(instr, test_list)
        # Ensure code/response column exists
        response = row.get('target') or row.get('response') or ""
        # Concat: prompt + response (we want model to learn to generate response)
        # include an explicit separator (tokenizer.eos_token will be used)
        full = prompt_text + " " + response + (tokenizer.eos_token or "")
        prompts.append(full)

    enc = tokenizer(prompts, truncation=True, padding=False, max_length=max_length)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Now create labels and mask the prompt portion: we need to find where the response starts for each example.
    labels = []
    for i, ids in enumerate(input_ids):
        # naive heuristic: find start of response by searching for the response string tokenization
        # safer: we can re-tokenize prompt and get its length
        prompt_only = build_prompt(df.iloc[i]['prompt'], df.iloc[i].get('test_list'))
        prompt_tokens = tokenizer(prompt_only, truncation=True, padding=False, max_length=max_length)["input_ids"]
        prompt_len = len(prompt_tokens)
        # labels: copy input ids then mask first prompt_len tokens with -100
        lab = ids.copy()
        for j in range(min(prompt_len, len(lab))):
            lab[j] = -100
        labels.append(lab)

    # Convert to Dataset
    ds = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return ds

def main():
    parser = argparse.ArgumentParser(description="Fine-tune causal LM for Bangla->Python code generation.")
    parser.add_argument('--data', required=True, help="Path to CSV (trial.csv with response)")
    parser.add_argument('--out', required=True, help="Output artifacts dir")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS_SMALL)
    args = parser.parse_args()

    setup_logging()
    set_seed(config.SEED)
    logging.info(f"Device: {get_device()}")
    print_tmpdir_hint()
    make_artifact_dirs(args.out)

    # load dataset (expect trial.csv includes 'response' or 'target' column)
    df = load_csv(args.data, require_target=True)

    # build prompt texts and preprocess
    df['prompt'] = df['prompt'].astype(str)
    df['test_list'] = df.get('test_list', [None] * len(df))

    # Initialize tokenizer & model
    transformer = TransformerModel(model_name=config.MODEL_NAME, max_length=config.MAX_LENGTH, use_peft=config.USE_PEFT)
    tokenizer = transformer.tokenizer

    # Prepare HF Dataset with label masking (prompt tokens -> -100)
    dataset = prepare_causal_dataset(df, tokenizer, config.MAX_LENGTH)

    # split train/val small split
    train_test = dataset.train_test_split(test_size=0.05, seed=config.SEED)
    train_ds = train_test['train']
    val_ds = train_test['test']

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        report_to="none",
        seed=config.SEED
    )

    trainer = Trainer(
        model=transformer.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()
    # Save final - both tokenizer and model (PEFT-aware)
    transformer.save(str(output_dir))

    # write metadata
    with open(output_dir / "config.json", "w") as f:
        json.dump({"backend": "transformer", "is_fine_tune": True, "model_name": config.MODEL_NAME}, f)

    logging.info("Training finished and artifacts saved.")

if __name__ == "__main__":
    main()
