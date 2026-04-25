"""
training/rl/rollout.py
=======================
Standalone rollout utilities for the GRPO trainer.
Extracted from grpo_trainer.py so they can be unit-tested and
reused independently (e.g. from a notebook or evaluation script).

All functions are pure (given a model, tokenizer, env client, and evaluator)
with no global state.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an autonomous SRE AI managing a cloud datacenter.\n"
    "Observe the state and respond with ONLY a valid JSON object:\n"
    '  {{"intent": "<action>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}\n'
    "Valid actions: {actions}"
)


def build_prompt(obs: Dict[str, Any], valid_actions: List[str]) -> str:
    """Build the LLM prompt from a raw environment observation dict."""
    s = obs.get("observation", obs)
    tl  = s.get("Traffic_Load", 50.0)
    db  = s.get("Database_Temperature", 50.0)
    nh  = s.get("Network_Health", 50.0)
    slo = s.get("slo_score", ((100 - tl) + (100 - db) + nh) / 300.0)

    system = _SYSTEM_PROMPT.format(actions=", ".join(valid_actions))
    return (
        f"{system}\n\n"
        f"Traffic_Load: {tl:.1f}\n"
        f"Database_Temperature: {db:.1f}\n"
        f"Network_Health: {nh:.1f}\n"
        f"SLO: {slo:.3f}\n\n"
        "Your JSON:"
    )


# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(
    text: str,
    valid_actions: List[str],
) -> Tuple[str, float, bool]:
    """
    Extract (action, confidence, is_valid) from a raw model completion.

    Falls back to ("noop", 0.0, False) on any parse error.
    """
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s == -1 or e == 0:
            return "noop", 0.0, False
        p = json.loads(text[s:e])
        action     = str(p.get("intent", "noop"))
        confidence = float(p.get("confidence", 0.5))
        if action in valid_actions:
            return action, confidence, True
        return "noop", confidence, False
    except Exception:
        return "noop", 0.0, False


# ── Single episode rollout ────────────────────────────────────────────────────

def run_episode(
    model: Any,
    tokenizer: Any,
    env_client: Any,              # OpenEnvClient
    evaluator: Any,               # MultiComponentEvaluator
    valid_actions: List[str],
    group_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    routing_path: str = "middle_path",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run one full episode and return step records for GRPO training.

    Each record contains:
      prompt, completions, rewards, advantages, component_rewards, obs_before, obs_after
    """
    import torch

    obs = env_client.reset(seed=seed or random.randint(0, 99999))
    evaluator.reset_episode()
    records = []
    done = False

    while not done:
        prompt = build_prompt(obs, valid_actions)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outs = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # FIX BUG-5
            )

        prompt_len  = inp["input_ids"].shape[1]  # FIX BUG-6: precompute once
        completions = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outs
        ]

        rewards, comp_rewards, next_obs_list = [], [], []

        for comp in completions:
            action, conf, _ = parse_action(comp, valid_actions)
            next_obs = env_client.step(action)
            rd = evaluator.score(
                comp, action, obs, next_obs,
                routing_path, [], conf,
            )
            rewards.append(rd["total"])
            comp_rewards.append(rd)
            next_obs_list.append(next_obs)

        # Group-relative advantages
        r_t = torch.tensor(rewards, dtype=torch.float32)
        std = r_t.std()
        if std < 1e-8:
            adv = [0.0] * len(rewards)  # all identical → skip gradient
        else:
            adv = ((r_t - r_t.mean()) / std.clamp(min=1e-8)).tolist()

        best_idx = rewards.index(max(rewards))
        best_obs = next_obs_list[best_idx]

        records.append({
            "prompt":            prompt,
            "prompt_len":        prompt_len,       # FIX BUG-6
            "completions":       completions,
            "rewards":           rewards,
            "advantages":        adv,
            "component_rewards": comp_rewards,
            "obs_before":        obs,
            "obs_after":         best_obs,
        })

        obs  = best_obs
        # FIX BUG-4: read terminated/truncated from top-level server response
        done = bool(best_obs.get("terminated", False)) or \
               bool(best_obs.get("truncated", False))

    return records


# ── GRPO gradient update ──────────────────────────────────────────────────────

def grpo_update(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    learning_rate: float = 5e-6,
    max_seq_length: int = 2048,
) -> float:
    """
    Manual GRPO gradient step over a batch of rollout records.

    Returns the mean loss over all (completion, advantage) pairs.
    """
    import torch
    from torch.optim import AdamW

    opt = AdamW(model.parameters(), lr=learning_rate)
    model.train()
    optimizer_zero = True
    total_loss = 0.0
    n = 0

    for rec in records:
        plen = rec.get("prompt_len")  # FIX BUG-6: use precomputed length
        for comp, adv_val in zip(rec["completions"], rec["advantages"]):
            if adv_val == 0.0:
                continue  # skip zero-advantage completions

            full = rec["prompt"] + comp
            enc  = tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(model.device)

            labels = enc["input_ids"].clone()
            if plen is not None:
                labels[:, :plen] = -100
            else:
                # Fallback: recompute prompt length (slower)
                plen = tokenizer(rec["prompt"], return_tensors="pt")["input_ids"].shape[1]
                labels[:, :plen] = -100

            loss = model(**enc, labels=labels).loss * adv_val
            loss.backward()
            total_loss += loss.item()
            n += 1

    if n > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    opt.zero_grad()
    return total_loss / max(n, 1)
