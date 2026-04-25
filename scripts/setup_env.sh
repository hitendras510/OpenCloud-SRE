#!/usr/bin/env bash
# scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot local environment setup for OpenCloud-SRE.
# Run once after cloning:  bash scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR="venv"

echo "╔══════════════════════════════════════╗"
echo "║   OpenCloud-SRE  —  Environment Setup ║"
echo "╚══════════════════════════════════════╝"

# 1. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "► Creating virtual environment in ./$VENV_DIR …"
    $PYTHON -m venv $VENV_DIR
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "► Upgrading pip …"
pip install --quiet --upgrade pip

echo "► Installing dependencies …"
pip install --quiet -r requirements.txt

# 2. Create .env if missing
if [ ! -f ".env" ]; then
    echo "► Creating .env from .env.example …"
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your HF_TOKEN before training."
else
    echo "► .env already exists — skipping."
fi

# 3. Ensure data directory exists
mkdir -p data models/sft_model models/rl_model

echo ""
echo "✅  Setup complete!"
echo ""
echo "Next steps:"
echo "  1.  Edit .env and add your HF_TOKEN"
echo "  2.  source venv/bin/activate"
echo "  3.  bash scripts/train_sft.sh       # Generate data + SFT warmup"
echo "  4.  bash scripts/train_rl.sh        # GRPO training"
echo "  5.  streamlit run ui/app.py          # Launch the War Room UI"
