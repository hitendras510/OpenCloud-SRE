#!/usr/bin/env bash
# scripts/train_sft.sh
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Generate synthetic SFT dataset
# Step 2: Run supervised fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source venv/bin/activate 2>/dev/null || true
source .env 2>/dev/null || true

MODEL="${SFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET="data/sft_logs.jsonl"
OUTPUT="models/sft_model"
COUNT="${DATASET_COUNT:-100}"
EPOCHS="${SFT_EPOCHS:-2}"

echo "══════════════════════════════════════════"
echo "  OpenCloud-SRE — SFT Pipeline"
echo "  Model:   $MODEL"
echo "  Samples: $COUNT"
echo "  Epochs:  $EPOCHS"
echo "══════════════════════════════════════════"

echo ""
echo "► Step 1: Generating $COUNT synthetic incident logs …"
python -m training.sft.dataset_generator \
    --count  "$COUNT" \
    --output "$DATASET" \
    --model  "meta-llama/Meta-Llama-3-70B-Instruct"

echo ""
echo "► Step 2: Running SFT on $MODEL …"
python -m training.sft.train_sft \
    --dataset "$DATASET" \
    --model   "$MODEL" \
    --output  "$OUTPUT" \
    --epochs  "$EPOCHS"

echo ""
echo "✅  SFT complete! Model saved to: $OUTPUT"
