#!/usr/bin/env bash
# scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot local environment setup for OpenCloud-SRE.
# Run once after cloning:  bash scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR="venv"

# BUG-FIX: Must run from repo root so relative paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "╔══════════════════════════════════════╗"
echo "║   OpenCloud-SRE  —  Environment Setup ║"
echo "╚══════════════════════════════════════╝"

# BUG-FIX: Check Python version before creating venv
PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MAJOR=3; REQUIRED_MINOR=10
if ! $PYTHON -c "import sys; sys.exit(0 if sys.version_info >= ($REQUIRED_MAJOR, $REQUIRED_MINOR) else 1)" 2>/dev/null; then
    echo "❌  Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "► Python $PYTHON_VERSION detected ✓"

# 1. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "► Creating virtual environment in ./$VENV_DIR …"
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "► Upgrading pip …"
pip install --quiet --upgrade pip

echo "► Installing dependencies …"
pip install --quiet -r requirements.txt

# BUG-FIX: bitsandbytes is needed for 4-bit QLoRA, install only if CUDA available
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "► CUDA detected — installing bitsandbytes for 4-bit QLoRA …"
    pip install --quiet "bitsandbytes>=0.43.0"
else
    echo "ℹ️  No CUDA — skipping bitsandbytes (CPU-only mode)."
fi

# 2. Create .env if missing
if [ ! -f ".env" ]; then
    echo "► Creating .env from .env.example …"
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your HF_TOKEN before training."
else
    echo "► .env already exists — skipping."
fi

# 3. Ensure data/model directories exist
mkdir -p data models/sft_model models/rl_model

# BUG-FIX: Make all scripts executable so they can be run without explicit `bash`
chmod +x scripts/*.sh
echo "► Script permissions set ✓"

echo ""
echo "✅  Setup complete!"
echo ""
echo "Next steps:"
echo "  1.  Edit .env and add your HF_TOKEN"
echo "  2.  source venv/bin/activate"
echo "  3.  bash scripts/train_sft.sh       # Generate data + SFT warmup"
echo "  4.  bash scripts/train_rl.sh        # GRPO training"
echo "  5.  streamlit run ui/app.py          # Launch the War Room UI"
