#!/usr/bin/env bash
# scripts/train_rl.sh
# ─────────────────────────────────────────────────────────────────────────────
# GRPO Reinforcement Learning training.
# Prerequisites:
#   1. bash scripts/train_sft.sh   (or pre-trained model in models/sft_model)
#   2. uvicorn env.server:app --port 8000  (running in another terminal)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# BUG-FIX: cd to repo root so python -m imports work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# BUG-FIX: source .env before venv so env vars are actually available
if [ -f ".env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true
fi

source venv/bin/activate 2>/dev/null || true

MODEL="${RL_MODEL:-models/sft_model}"         # start from SFT checkpoint
ENV_URL="${ENV_URL:-http://localhost:8000}"
OUTPUT="models/rl_model"
EPOCHS="${RL_EPOCHS:-3}"
STEPS="${RL_STEPS:-50}"
GROUP_SIZE="${GROUP_SIZE:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-opencloud-sre-grpo}"

echo "══════════════════════════════════════════"
echo "  OpenCloud-SRE — GRPO RL Training"
echo "  Base model:  $MODEL"
echo "  Env server:  $ENV_URL"
echo "  Epochs:      $EPOCHS  |  Steps/epoch: $STEPS"
echo "  Group size:  $GROUP_SIZE"
echo "══════════════════════════════════════════"

# BUG-FIX: Check model directory exists before starting
if [ ! -d "$MODEL" ] && [ ! -f "$MODEL/config.json" ]; then
    # Allow HuggingFace Hub model IDs (no slash check needed for those)
    if [[ "$MODEL" != *"/"* ]] || [ -d "$MODEL" ]; then
        echo "ℹ️  Using HuggingFace Hub model: $MODEL"
    elif [ ! -d "$MODEL" ]; then
        echo "❌  Model not found: $MODEL"
        echo "    Run bash scripts/train_sft.sh first, or set RL_MODEL to a HF model ID."
        exit 1
    fi
fi

# Health-check the env server before starting
echo ""
echo "► Checking OpenEnv server at $ENV_URL …"
# BUG-FIX: retry up to 3 times with 2s delay (server may still be starting)
for attempt in 1 2 3; do
    if curl -sf "$ENV_URL/" > /dev/null 2>&1; then
        echo "   Server OK ✓"
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "❌  OpenEnv server not reachable at $ENV_URL after 3 attempts."
        echo "    Start it first:  uvicorn env.server:app --port 8000"
        exit 1
    fi
    echo "   Attempt $attempt failed — retrying in 2s …"
    sleep 2
done

echo ""
echo "► Starting GRPO training …"
python -m training.rl.grpo_trainer \
    --model         "$MODEL" \
    --env-url       "$ENV_URL" \
    --epochs        "$EPOCHS" \
    --steps         "$STEPS" \
    --group-size    "$GROUP_SIZE" \
    --output        "$OUTPUT" \
    --wandb-project "$WANDB_PROJECT"

echo ""
echo "✅  GRPO training complete! RL model saved to: $OUTPUT"
