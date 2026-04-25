# ============================================================
# OpenCloud-SRE — Hugging Face Spaces GPU Training Dockerfile
# ============================================================
# Build/Run via HF Spaces (Dockerfile SDK).
# Push to trigger: git push space main
#
# GPU tier required: T4-small or better ($30 credit eligible).
# The container runs the full 3-stage pipeline automatically:
#   Stage 1 – Generate 100 synthetic SFT incident logs (Llama-3 via HF API)
#   Stage 2 – SFT warm-up on Qwen2.5-1.5B-Instruct (TRL / QLoRA)
#   Stage 3 – GRPO RL loop with anti-hacking reward function (3 epochs)
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── System setup ─────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential curl git wget \
    && rm -rf /var/lib/apt/lists/*

# Use python3 as python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ── Install Python deps (layer-cached before copying source) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Install Unsloth for 4-bit QLoRA (CUDA build) ─────────────────────────────
RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    bitsandbytes

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Runtime secrets (set these in HF Space Settings → Secrets) ───────────────
ENV HF_TOKEN=""
ENV WANDB_API_KEY=""
ENV WANDB_PROJECT="opencloud-sre-grpo"
ENV SFT_DATA_COUNT="100"
ENV RL_EPOCHS="3"
ENV RL_STEPS="50"

# ── Port (HF Spaces requires 7860) ───────────────────────────────────────────
EXPOSE 7860

# ── Entrypoint: run the full pipeline then launch the UI ─────────────────────
# The CMD is a shell script so each stage can fail gracefully.
CMD ["/bin/bash", "-c", "\
    set -e && \
    echo '🚀 Stage 1: Generating SFT dataset...' && \
    python -m training.sft.dataset_generator --count ${SFT_DATA_COUNT:-100} && \
    echo '🏋️ Stage 2: SFT warm-up...' && \
    python -m training.sft.train_sft \
        --dataset data/sft_logs.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output models/sft_model \
        --epochs 2 && \
    echo '🤖 Stage 3: Starting OpenEnv server...' && \
    uvicorn env.server:app --host 0.0.0.0 --port 8001 & \
    sleep 5 && \
    echo '🎯 Stage 3: GRPO RL training...' && \
    python -m training.rl.grpo_trainer \
        --model models/sft_model \
        --env-url http://localhost:8001 \
        --epochs ${RL_EPOCHS:-3} \
        --steps ${RL_STEPS:-50} \
        --output models/rl_model \
        --wandb-project ${WANDB_PROJECT:-opencloud-sre-grpo} && \
    echo '✅ Training complete! Launching UI dashboard...' && \
    streamlit run ui/app.py \
        --server.port 7860 \
        --server.address 0.0.0.0 \
        --server.headless true \
"]
