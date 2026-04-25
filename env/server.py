from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

app = FastAPI(title="OpenCloud-SRE Live Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GlobalStateManager:
    def __init__(self):
        self.cpu = 50.0
        self.db_temp = 50.0
        self.latency = 5.0

    def get_state(self) -> Dict[str, Any]:
        status = "NOMINAL"
        if self.cpu >= 80.0 or self.db_temp >= 80.0 or self.latency >= 50.0:
            status = "CRITICAL"
        
        return {
            "status": status,
            "metrics": {
                "CPU": self.cpu,
                "DB_Temp": self.db_temp,
                "Latency": self.latency,
                # Including variants for app.py
                "Traffic_Load": self.cpu,
                "Database_Temperature": self.db_temp,
                "Network_Health": self.latency
            }
        }

    def inject_fault(self, fault_type: str, value: float):
        if fault_type == "CPU_SPIKE":
            self.cpu = value
        elif fault_type == "DB_DEADLOCK":
            self.db_temp = value
        elif fault_type == "NETWORK_PARTITION":
            self.latency = value

    def execute_fix(self, action: str):
        action = action.upper()
        if "DB" in action or action == "SCHEMA_FAILOVER":
            self.db_temp = max(50.0, self.db_temp - 40.0)
        elif "SCALE" in action or "CIRCUIT_BREAKER" in action or action == "SCALE_COMPUTE":
            self.cpu = max(50.0, self.cpu - 40.0)
        elif "ROUTE" in action or "THROTTLE" in action or "NETWORK" in action:
            self.latency = max(5.0, self.latency - 40.0)
            self.cpu = max(50.0, self.cpu - 20.0)
        else:
            self.cpu = max(50.0, self.cpu - 20.0)
            self.db_temp = max(50.0, self.db_temp - 20.0)
            self.latency = max(5.0, self.latency - 20.0)

state_manager = GlobalStateManager()

@app.get("/metrics")
def get_metrics():
    return state_manager.get_state()

class FaultRequest(BaseModel):
    fault_type: str
    value: float = 95.0

@app.post("/inject-fault")
def inject_fault(req: FaultRequest):
    state_manager.inject_fault(req.fault_type, req.value)
    return {"message": f"Injected {req.fault_type}", "state": state_manager.get_state()}

class ExecuteRequest(BaseModel):
    action: str

@app.post("/execute")
def execute_fix(req: ExecuteRequest):
    state_manager.execute_fix(req.action)
    return {"message": f"Executed {req.action}", "state": state_manager.get_state()}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("env.server:app", host="0.0.0.0", port=port, reload=False)
