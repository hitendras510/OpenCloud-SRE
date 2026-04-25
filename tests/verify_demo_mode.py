"""
tests/verify_demo_mode.py
=========================
QA Automation script to verify that the 'Deterministic Demo Mode' correctly
intercepts the right action, sets the resolution state, and gracefully halts
the RL training loop.
"""

import sys
import os

# Ensure project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient

from env.server import app, DEMO_SCENARIOS, ACTIVE_SCENARIO
import env.server
from training.rl.grpo_trainer import rollout, GRPOTrainConfig

# ── Mock Classes to avoid loading real LLMs ───────────────────────────────────

class DummyTokenizer:
    pad_token_id = 0
    def __call__(self, text, *args, **kwargs):
        import torch
        class DummyBatchEncoding(dict):
            def to(self, device): return self
        # Return dummy input IDs
        return DummyBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
        
    def decode(self, *args, **kwargs):
        # We need the model to output a JSON string with the CORRECT required fix
        required_fix = DEMO_SCENARIOS.get(ACTIVE_SCENARIO, "schema_failover")
        return f'{{"intent": "{required_fix}", "confidence": 0.95, "rationale": "Mitigating root cause."}}'

class DummyModel:
    device = "cpu"
    def generate(self, *args, **kwargs):
        import torch
        # Dummy generation
        return torch.tensor([[1, 2, 3, 4]])

class MockEnvClient:
    def __init__(self, client: TestClient):
        self.client = client
        
    def reset(self, seed=None):
        return self.client.post("/reset", json={"seed": seed, "crash_on_reset": True}).json()
        
    def step(self, action):
        return self.client.post("/step", json={"action": action}).json()

class MockEvaluator:
    def reset_episode(self): pass
    def score(self, *args, **kwargs):
        return {"total": 100.0}

# ── Test Execution ────────────────────────────────────────────────────────────

def run_tests():
    print("Running QA Automation Tests for Demo Mode...\n")
    
    client = TestClient(app)
    env_mock = MockEnvClient(client)
    
    assert ACTIVE_SCENARIO in DEMO_SCENARIOS, "Active scenario must be defined."
    required_fix = DEMO_SCENARIOS[ACTIVE_SCENARIO]
    print(f"[*] Simulating Scenario: {ACTIVE_SCENARIO}")
    print(f"[*] Required Fix Action: {required_fix}")
    
    # Reset DEMO state just in case
    env.server.DEMO_IS_RESOLVED = False
    
    # 1. Test Server Endpoint Directly
    print("\n[Test 1] Testing Server /step intercept logic...")
    _ = env_mock.reset()
    res = env_mock.step(required_fix)
    
    assert res.get("status") == "SUCCESS", f"Server did not return SUCCESS. Got: {res}"
    assert res.get("terminated") is True, "Server did not terminate the episode."
    print("  => PASSED: Server correctly forced 'is_resolved = True' and returned SUCCESS payload.")
    
    # Reset DEMO state for trainer test
    env.server.DEMO_IS_RESOLVED = False
    
    # 2. Test GRPO Trainer Rollout loop breakout
    print("\n[Test 2] Testing GRPO Trainer rollout breakout logic...")
    cfg = GRPOTrainConfig(max_seq_length=128, max_new_tokens=32, group_size=1)
    
    # The DummyTokenizer is hardcoded to return the `required_fix` action immediately
    records, final_obs = rollout(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        env=env_mock,
        evaluator=MockEvaluator(),
        cfg=cfg,
        valid_actions=[required_fix, "noop"]
    )
    
    assert final_obs.get("status") == "SUCCESS", "Rollout did not capture the SUCCESS status."
    assert len(records) == 1, f"Expected rollout to break immediately after 1 step, but took {len(records)} steps."
    print("  => PASSED: Multi-agent loop detected SUCCESS and gracefully terminated after exactly 1 step.")
    
    print("\n" + "="*60)
    print("✅ DEMO READY: Success Path Verified")
    print("  - Server intercept logic is bulletproof.")
    print("  - Multi-agent rollout loop breaks gracefully.")
    print("  - Zero wasted GPU compute after resolution.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_tests()
