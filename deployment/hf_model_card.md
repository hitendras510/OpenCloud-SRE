---
language: en
license: apache-2.0
tags:
  - sre
  - reinforcement-learning
  - grpo
  - langchain
  - openenv
  - cloud-operations
datasets:
  - custom (synthetic incident logs)
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
---

# OpenCloud-SRE — Autonomous SRE Governance Model

## Model Description

**OpenCloud-SRE** is a GRPO-trained language model that acts as an autonomous Site Reliability Engineer (SRE) for cloud datacenter incident response. It is part of the OpenCloud-SRE multi-agent governance platform built for the OpenEnv hackathon.

The model observes three infrastructure metrics — **Traffic Load**, **Database Temperature**, and **Network Health** — and selects the optimal remediation action to maximize the System Level Objective (SLO) score.

## Training Pipeline

| Stage | Method | Model |
|---|---|---|
| 1. Data Generation | HF InferenceClient (Llama-3-70B) | Synthetic SFT logs |
| 2. SFT Warmup | Causal LM fine-tuning | Qwen2.5-1.5B-Instruct |
| 3. RL Alignment | GRPO (Group Relative Policy Optimization) | SFT checkpoint |

## Reward Function

The model was trained with a **multi-component, anti-hacking reward function**:

| Component | Value | Type |
|---|---|---|
| Format Reward | +10 | Deterministic |
| State Recovery | -50 to +100 | Deterministic |
| Blast Radius Penalty | -50 | Deterministic |
| Repetition Penalty | -20 | Anti-hacking |
| Noop Abuse Penalty | -30 | Anti-hacking |
| Confidence Calibration | -15 | Anti-hacking |
| State Plausibility | -40 | Anti-hacking |
| LLM Reasoning Score | +10 to +20 | HF Llama-3-8B (slow path only) |

## Action Space

```
throttle_traffic | load_balance | schema_failover | cache_flush
circuit_breaker  | restart_pods | scale_out       | noop
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/opencloud-sre-model")
tok   = AutoTokenizer.from_pretrained("your-username/opencloud-sre-model")

prompt = """You are an autonomous SRE AI. Observe the datacenter state and respond with
ONLY a JSON object: {"intent": "<action>", "confidence": <0-1>, "rationale": "<one sentence>"}

Traffic_Load: 89.0
Database_Temperature: 72.0
Network_Health: 35.0
SLO: 0.213

Your JSON:"""

inp  = tok(prompt, return_tensors="pt")
out  = model.generate(**inp, max_new_tokens=128, temperature=0.1)
print(tok.decode(out[0], skip_special_tokens=True))
```

## OpenEnv Compliance

The training environment is deployed as a standard FastAPI REST server:
- `POST /reset` — reset episode
- `POST /step`  — execute action, returns next observation + reward

## Architecture

- **LangGraph** multi-agent orchestrator (3-tier routing)
- **DNA Memory** (FAISS) for fast-path incident recall
- **Shadow Consensus** for agent agreement checking
- **Blast Radius DDM** for safety filtering
- **Adaptive Trust Layer** for human-escrow decisions

## License

Apache 2.0
