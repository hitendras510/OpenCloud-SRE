#!/bin/bash
set -e

# Activate Python virtual environment
source venv/bin/activate || true

echo "🔴 Starting Chaos Control Backend (OpenEnv API) on port 8000..."
uvicorn env.server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
sleep 3

echo "🖥️ Starting NEXUS Streamlit UI on port 7860..."
streamlit run ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true &
UI_PID=$!

echo "🧠 Starting GRPO Agent Training Loop..."
# Using the default model if not fine-tuned yet; adjust paths as needed.
python -m training.rl.grpo_trainer \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --env-url http://localhost:8000 \
    --epochs ${RL_EPOCHS:-3} \
    --steps ${RL_STEPS:-50} &
TRAINER_PID=$!

echo "✅ System is fully running. Press Ctrl+C to stop."
wait $UI_PID $SERVER_PID $TRAINER_PID
