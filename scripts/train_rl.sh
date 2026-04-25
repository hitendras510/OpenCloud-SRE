#!/usr/bin/env bash
# scripts/train_rl.sh
# ─────────────────────────────────────────────────────────────────────────────
# GRPO Reinforcement Learning training.
# Prerequisites:
#   1. bash scripts/train_sft.sh   (or pre-trained model in models/sft_model)
#   2. uvicorn env.server:app --port 8000  (running in another terminal)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source venv/bin/activate 2>/dev/null || true
source .env 2>/dev/null || true

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

# Health-check the env server before starting
echo ""
echo "► Checking OpenEnv server at $ENV_URL …"
if ! curl -sf "$ENV_URL/" > /dev/null; then
    echo "❌  OpenEnv server not reachable at $ENV_URL"
    echo "    Start it first:  uvicorn env.server:app --port 8000"
    exit 1
fi
echo "   Server OK ✓"

echo ""
echo "► Starting GRPO training …"
python -m training.rl.grpo_trainer \
    --model        "$MODEL" \
    --env-url      "$ENV_URL" \
    --epochs       "$EPOCHS" \
    --steps        "$STEPS" \
    --group-size   "$GROUP_SIZE" \
    --output       "$OUTPUT" \
    --wandb-project "$WANDB_PROJECT"

echo ""
echo "✅  GRPO training complete! RL model saved to: $OUTPUT"
