#!/usr/bin/env bash
# deployment/Dockerfile-equivalent for Hugging Face Spaces.
# Build: docker build -t opencloud-sre .
# Run:   docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN opencloud-sre

FROM python:3.10-slim

LABEL maintainer="OpenCloud-SRE Team"
LABEL description="OpenCloud-SRE — Autonomous SRE Governance Platform (OpenEnv FastAPI)"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose the OpenEnv FastAPI server port (HF Spaces default = 7860)
EXPOSE 7860

# Environment variable placeholders (override at runtime)
ENV HF_TOKEN=""
ENV WANDB_API_KEY=""
ENV PYTHONUNBUFFERED=1

# Start the OpenEnv server — the RL trainer connects to this
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]
