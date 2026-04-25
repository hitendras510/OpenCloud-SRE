#!/usr/bin/env bash
# scripts/train_sft.sh
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Generate synthetic SFT dataset
# Step 2: Run supervised fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# BUG-FIX: cd to repo root so python -m imports work from the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# BUG-FIX: source .env before venv so env vars are available
# (previously sourced after activate, .env vars like SFT_MODEL were never read)
if [ -f ".env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true
fi

source venv/bin/activate 2>/dev/null || true

MODEL="${SFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET="data/sft_logs.jsonl"
OUTPUT="models/sft_model"
COUNT="${DATASET_COUNT:-100}"
EPOCHS="${SFT_EPOCHS:-2}"
MAX_SEQ="${SFT_MAX_SEQ:-1024}"

echo "══════════════════════════════════════════"
echo "  OpenCloud-SRE — SFT Pipeline"
echo "  Model:    $MODEL"
echo "  Samples:  $COUNT"
echo "  Epochs:   $EPOCHS"
echo "  Max seq:  $MAX_SEQ"
echo "══════════════════════════════════════════"

# BUG-FIX: Check HF_TOKEN is set before making API calls
if [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠️  HF_TOKEN not set — dataset generation will use rule-based fallback."
fi

echo ""
echo "► Step 1: Generating $COUNT synthetic incident logs …"
python -m training.sft.dataset_generator \
    --count  "$COUNT" \
    --output "$DATASET" \
    --model  "meta-llama/Meta-Llama-3-70B-Instruct"

# BUG-FIX: Verify dataset was actually created before starting training
if [ ! -f "$DATASET" ]; then
    echo "❌  Dataset not found at $DATASET — dataset generation failed."
    exit 1
fi
SAMPLE_COUNT=$(wc -l < "$DATASET" | tr -d ' ')
echo "   Dataset ready: $SAMPLE_COUNT samples ✓"

echo ""
echo "► Step 2: Running SFT on $MODEL …"
python -m training.sft.train_sft \
    --dataset "$DATASET" \
    --model   "$MODEL" \
    --output  "$OUTPUT" \
    --epochs  "$EPOCHS" \
    --max-seq "$MAX_SEQ"

echo ""
echo "✅  SFT complete! Model saved to: $OUTPUT"
