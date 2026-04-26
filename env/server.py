import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Import the actual environment
from env.environment import OpenCloudEnv

app = FastAPI(title="OpenCloud-SRE OpenEnv Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard OpenEnv Instance
_env = OpenCloudEnv(crash_on_reset=True)

class StepRequest(BaseModel):
    action: str

class FaultRequest(BaseModel):
    fault_type: str
    value: float = 95.0

# ─── Standard OpenEnv API ─────────────────────────────────────────────────────

@app.post("/reset")
def reset(seed: Optional[int] = None):
    obs, info = _env.reset(seed=seed)
    return {"observation": obs, "info": info}

@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, terminated, truncated, info = _env.step(req.action)
        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return _env.state()

# ─── Legacy / UI Compatibility API ────────────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    """Backward compatibility for existing UI and grpo_trainer."""
    obs = _env._observe()
    status = "NOMINAL"
    if _env.state.is_critical():
        status = "CRITICAL"
    
    return {
        "status": status,
        "metrics": obs
    }

@app.post("/execute")
def execute_fix(req: StepRequest):
    """Backward compatibility for existing grpo_trainer."""
    obs, reward, terminated, truncated, info = _env.step(req.action)
    return {
        "message": f"Executed {req.action}",
        "state": {
            "status": "CRITICAL" if _env.state.is_critical() else "NOMINAL",
            "metrics": obs
        }
    }

@app.post("/inject-fault")
def inject_fault(req: FaultRequest):
    """Inject fault using the environment's native method."""
    try:
        _env.inject_fault(req.fault_type.lower())
        return {"message": f"Injected {req.fault_type}", "state": get_metrics()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("env.server:app", host="0.0.0.0", port=port, reload=False)
