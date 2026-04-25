"""
utils/constants.py
==================
Project-wide constants shared across env, training, and evaluation layers.
Single source of truth — import from here, not from individual modules.
"""
from __future__ import annotations

from typing import Final, List

# ── Environment ───────────────────────────────────────────────────────────────
VALID_ACTIONS: Final[List[str]] = [
    "throttle_traffic",
    "load_balance",
    "schema_failover",
    "cache_flush",
    "circuit_breaker",
    "restart_pods",
    "scale_out",
    "noop",
]

METRIC_NAMES: Final[List[str]] = [
    "Traffic_Load",
    "Database_Temperature",
    "Network_Health",
]

METRIC_MIN: Final[float] = 0.0
METRIC_MAX: Final[float] = 100.0

# Healthy / nominal baseline used by FAISS DNA Memory
NOMINAL_STATE: Final[List[float]] = [20.0, 30.0, 90.0]

# Critical thresholds (any breach → SLO violation)
CRITICAL_TRAFFIC_LOAD:   Final[float] = 85.0
CRITICAL_DB_TEMPERATURE: Final[float] = 80.0
CRITICAL_NETWORK_HEALTH: Final[float] = 30.0   # below this is critical

# SLO target for episode termination
SLO_SUCCESS_THRESHOLD: Final[float] = 0.95
SLO_FAILURE_THRESHOLD: Final[float] = 0.05

# ── DNA Memory ────────────────────────────────────────────────────────────────
DNA_HIGH_MATCH_THRESHOLD:   Final[float] = 8.0
DNA_MEDIUM_MATCH_THRESHOLD: Final[float] = 20.0

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_GRPO_MODEL:    Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SFT_MODEL:     Final[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_DATASET_MODEL: Final[str] = "meta-llama/Meta-Llama-3-70B-Instruct"

SFT_OUTPUT_DIR: Final[str] = "models/sft_model"
RL_OUTPUT_DIR:  Final[str] = "models/rl_model"

# ── Reward bounds ─────────────────────────────────────────────────────────────
REWARD_FLOOR: Final[float] = -150.0
REWARD_CEIL:  Final[float] =  130.0

# ── Routing paths ─────────────────────────────────────────────────────────────
ROUTING_FAST_PATH:   Final[str] = "fast_path"
ROUTING_MIDDLE_PATH: Final[str] = "middle_path"
ROUTING_SLOW_PATH:   Final[str] = "slow_path"

# ── API / Server ──────────────────────────────────────────────────────────────
ENV_SERVER_HOST: Final[str] = "0.0.0.0"
ENV_SERVER_PORT: Final[int] = 8000
ENV_SERVER_URL:  Final[str] = "http://localhost:8000"
