#!/usr/bin/env bash
# scripts/run_demo.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch the complete OpenCloud-SRE demo stack:
#   Process 1: OpenEnv FastAPI server
#   Process 2: Streamlit War Room UI
# ─────────────────────────────────────────────────────────────────────────────
# BUG-FIX: remove set -e — background processes exit with non-zero on Ctrl+C
# which caused immediate script death and orphaned processes.
set -uo pipefail

# BUG-FIX: cd to repo root so uvicorn finds env.server:app module
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# BUG-FIX: source .env before venv
if [ -f ".env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true
fi

source venv/bin/activate 2>/dev/null || true

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

# BUG-FIX: trap Ctrl+C to cleanly kill both background processes
cleanup() {
    echo ""
    echo "► Shutting down services …"
    kill "$ENV_PID" "$UI_PID" 2>/dev/null || true
    wait "$ENV_PID" "$UI_PID" 2>/dev/null || true
    echo "✅  All services stopped."
}
trap cleanup INT TERM

# Start env server in background (no --reload in demo mode — faster startup)
uvicorn env.server:app --host 0.0.0.0 --port "$ENV_PORT" &
ENV_PID=$!
echo "► OpenEnv server started (PID $ENV_PID)"

# BUG-FIX: Wait for env server to be ready before launching UI
echo "► Waiting for env server to be ready …"
for i in $(seq 1 15); do
    if curl -sf "http://localhost:$ENV_PORT/" > /dev/null 2>&1; then
        echo "   Server ready ✓"
        break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
        echo "❌  Env server failed to start within 15s."
        kill "$ENV_PID" 2>/dev/null || true
        exit 1
    fi
done

# Start Streamlit
streamlit run ui/app.py \
    --server.port "$UI_PORT" \
    --server.headless true \
    --server.address 0.0.0.0 &
UI_PID=$!
echo "► Streamlit UI started (PID $UI_PID)"
echo ""
echo "  ➜ Open: http://localhost:$UI_PORT"

# Wait for either process to exit
wait "$ENV_PID" "$UI_PID"
