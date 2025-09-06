#!/usr/bin/env bash
# run.sh - pipeline runner for transformer / sklearn
set -euo pipefail

BACKEND=${BACKEND:-transformer}
FORCE_TRAIN=${FORCE_TRAIN:-0}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-artifacts}
TRAIN_CSV=${TRAIN_CSV:-data/trial.csv}
DEV_CSV=${DEV_CSV:-data/dev_v2.csv}
OUT_JSON=${OUT_JSON:-submission.json}
ZIP_NAME=${ZIP_NAME:-submission.zip}
BATCH_SIZE=${BATCH_SIZE:-2}

GPU_AVAILABLE=$(python - <<'PY'
try:
    import torch
    print(1 if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)

echo "BACKEND = $BACKEND"
echo "FORCE_TRAIN = $FORCE_TRAIN"
if [ "$GPU_AVAILABLE" -eq 1 ]; then echo "GPU detected"; else echo "No GPU"; fi

# sanity check
for f in "$TRAIN_CSV" "$DEV_CSV"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f" >&2
        exit 2
    fi
done

mkdir -p "$ARTIFACTS_DIR"

# training
if [ "$BACKEND" = "sklearn" ]; then
    echo "=== Sklearn backend training ==="
    python -m src.trainer --backend sklearn --data "$TRAIN_CSV" --out "$ARTIFACTS_DIR"
else
    echo "=== Transformer (CodeLlama-7B) flow ==="
    MODEL_PRESENT=0
    if [ -d "$ARTIFACTS_DIR" ] && \
       ( [ -f "$ARTIFACTS_DIR/tokenizer.json" ] || [ -f "$ARTIFACTS_DIR/config.json" ] || [ -d "$ARTIFACTS_DIR/model" ] || [ -f "$ARTIFACTS_DIR/pytorch_model.bin" ] ); then
        MODEL_PRESENT=1
    fi

    if [ "$FORCE_TRAIN" = "1" ] || [ "$MODEL_PRESENT" = "0" ]; then
        echo "Training CodeLlama-7B -> $ARTIFACTS_DIR"
        if [ "$GPU_AVAILABLE" -eq 1 ]; then
            python -m src.trainer --data "$TRAIN_CSV" --out "$ARTIFACTS_DIR" --use_gpu
        else
            echo "WARNING: training on CPU may be very slow!"
            python -m src.trainer --data "$TRAIN_CSV" --out "$ARTIFACTS_DIR"
        fi
    else
        echo "Artifacts exist and FORCE_TRAIN=0 -> skipping training"
    fi
fi

# evaluation
echo "=== Evaluating ==="
python -m src.evaluate --data "$DEV_CSV" --artifacts "$ARTIFACTS_DIR" || echo "Evaluation skipped"

# predictions
echo "=== Generating predictions -> $OUT_JSON ==="
python -m src.predict --data "$DEV_CSV" --out "$OUT_JSON" --artifacts "$ARTIFACTS_DIR" --batch_size "$BATCH_SIZE"

if [ ! -f "$OUT_JSON" ]; then
    echo "ERROR: predictions not created" >&2
    exit 3
fi

echo "Zipping submission -> $ZIP_NAME"
if command -v zip >/dev/null 2>&1; then
    zip -j -q "$ZIP_NAME" "$OUT_JSON"
else
    tar -czf "${ZIP_NAME%.zip}.tar.gz" "$OUT_JSON"
fi

echo "Done."
