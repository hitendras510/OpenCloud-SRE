#!/usr/bin/env bash
# scripts/run_demo.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch the complete OpenCloud-SRE demo stack:
#   Terminal 1: OpenEnv FastAPI server
#   Terminal 2: Streamlit War Room UI
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source venv/bin/activate 2>/dev/null || true
source .env 2>/dev/null || true

ENV_PORT="${ENV_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

echo "╔══════════════════════════════════════════╗"
echo "║   OpenCloud-SRE  —  Demo Stack Launcher  ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  OpenEnv server → http://localhost:$ENV_PORT"
echo "  War Room UI    → http://localhost:$UI_PORT"
echo ""
echo "Press Ctrl+C to stop both services."
echo ""

# Start env server in background
uvicorn env.server:app --host 0.0.0.0 --port "$ENV_PORT" --reload &
ENV_PID=$!
echo "► OpenEnv server started (PID $ENV_PID)"

sleep 2  # give server time to bind

# Start Streamlit
streamlit run ui/app.py --server.port "$UI_PORT" --server.headless true &
UI_PID=$!
echo "► Streamlit UI started (PID $UI_PID)"

# Wait for either to exit
wait $ENV_PID $UI_PID
