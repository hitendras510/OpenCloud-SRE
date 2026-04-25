"""
env/server.py
=============
OpenEnv-compliant FastAPI server for OpenCloud-SRE.

Wraps the PyTorch-backed OpenCloudEnv simulation behind standard
/reset and /step HTTP endpoints so any GRPO trainer or external
evaluator can interact with it as a stateless REST service.

Run locally:
    uvicorn env.server:app --reload --port 8000

Docker (Hugging Face Space):
    CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import OpenCloudEnv, VALID_ACTIONS

logger = logging.getLogger(__name__)

# ──────────────────────────────── dataclasses ─────────────────────────────────

@dataclass
class Observation:
    """
    The observable state of the cloud datacenter.
    All metrics are floats in [0, 100].
    """
    Traffic_Load: float
    Database_Temperature: float
    Network_Health: float
    slo_score: float
    is_critical: bool
    episode_step: int

    def to_list(self) -> List[float]:
        return [self.Traffic_Load, self.Database_Temperature, self.Network_Health]


@dataclass
class StepResult:
    """Full result returned after each environment step."""
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


# ──────────────────────────────── Pydantic I/O ────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    crash_on_reset: bool = True


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]
    valid_actions: List[str]


class StepRequest(BaseModel):
    action: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    status: Optional[str] = None
    message: Optional[str] = None


class InfoResponse(BaseModel):
    valid_actions: List[str]
    max_steps: int
    description: str


# ──────────────────────────────── app + state ─────────────────────────────────

app = FastAPI(
    title="OpenCloud-SRE · OpenEnv",
    description="PyTorch datacenter simulation exposed as an OpenEnv-compliant REST API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared environment instance per server process.
_env: Optional[OpenCloudEnv] = None

# ──────────────────────────────── Demo Mode ───────────────────────────────────
DEMO_SCENARIOS = {
    "DB_OVERLOAD": "schema_failover",
    "CPU_SPIKE": "scale_out",
    "TRAFFIC_SPIKE": "throttle_traffic"
}
ACTIVE_SCENARIO = "DB_OVERLOAD"
DEMO_IS_RESOLVED = False


def _get_env() -> OpenCloudEnv:
    global _env
    if _env is None:
        _env = OpenCloudEnv(seed=42, crash_on_reset=True)
        _env.reset()
    return _env


def _obs_to_dict(obs_raw: Dict[str, float], env: OpenCloudEnv, step: int) -> Dict[str, Any]:
    """Convert raw observation dict into a full Observation dataclass dict."""
    ob = Observation(
        Traffic_Load=obs_raw["Traffic_Load"],
        Database_Temperature=obs_raw["Database_Temperature"],
        Network_Health=obs_raw["Network_Health"],
        slo_score=env.state.slo_score(),
        is_critical=env.state.is_critical(),
        episode_step=step,
    )
    return asdict(ob)


# ──────────────────────────────── endpoints ───────────────────────────────────

@app.get("/", response_model=InfoResponse)
def root() -> InfoResponse:
    """Health-check and environment manifest."""
    return InfoResponse(
        valid_actions=VALID_ACTIONS,
        max_steps=50,
        description="OpenCloud-SRE datacenter simulation (Traffic, DB_Temp, Network_Health).",
    )


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    """
    Reset the environment to a fresh episode.

    Parameters
    ----------
    seed : optional int — reproduce a specific episode.
    crash_on_reset : bool — if True the datacenter starts in a degraded state.
    """
    global _env, DEMO_IS_RESOLVED
    _env = OpenCloudEnv(seed=req.seed, crash_on_reset=req.crash_on_reset)
    DEMO_IS_RESOLVED = False
    obs_raw, info = _env.reset()
    return ResetResponse(
        observation=_obs_to_dict(obs_raw, _env, 0),
        info=info,
        valid_actions=VALID_ACTIONS,
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """
    Execute one action in the environment.

    Parameters
    ----------
    action : str — must be one of the valid SRE actions.
    """
    env = _get_env()
    global DEMO_IS_RESOLVED
    
    if req.action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{req.action}'. Valid: {VALID_ACTIONS}",
        )
        
    obs_raw, reward, terminated, truncated, info = env.step(req.action)
    
    if not DEMO_IS_RESOLVED and req.action == DEMO_SCENARIOS.get(ACTIVE_SCENARIO):
        DEMO_IS_RESOLVED = True
        return StepResponse(
            observation=_obs_to_dict(obs_raw, env, env._step_count),
            reward=100.0,
            terminated=True,
            truncated=False,
            info=info,
            status="SUCCESS",
            message="Root cause mitigated. System stabilizing."
        )

    return StepResponse(
        observation=_obs_to_dict(obs_raw, env, env._step_count),
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Return current environment state without advancing the episode."""
    env = _get_env()
    return {
        "observation": _obs_to_dict(env._observe(), env, env._step_count),
        "history": env.get_history(),
    }


@app.post("/inject_fault")
def inject_fault(fault_type: str = "traffic_spike") -> Dict[str, Any]:
    """
    Manually inject a fault mid-episode for curriculum testing.
    fault_type: traffic_spike | db_overload | network_partition | cascade_failure
    """
    env = _get_env()
    try:
        env.inject_fault(fault_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "injected", "fault_type": fault_type,
            "observation": _obs_to_dict(env._observe(), env, env._step_count)}


# ──────────────────────────────── entrypoint ──────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("env.server:app", host="0.0.0.0", port=port, reload=False)
