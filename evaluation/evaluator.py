"""
evaluation/evaluator.py
========================
Multi-component reward function for OpenCloud-SRE GRPO training.

Reward Components
-----------------
  format_reward          : +10   — valid parseable JSON micro-intent
  blast_radius_penalty   : -50   — action violates Dependency Matrix
  state_recovery_reward  : +50 to +100 — deterministic SLO / tensor improvement
  llm_reasoning_score    : +10 to +20  — GPT-4o-mini grades ChatOps quality
                                         (SLOW PATH only)

All components are logged independently to W&B so judges can audit
that the model isn't gaming a single soft score (reward hacking prevention).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── optional W&B ──────────────────────────────────────────────────────────────
try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False

# ── optional OpenAI (for llm_reasoning_score) ─────────────────────────────────
try:
    from openai import OpenAI as _OpenAI; _OPENAI = True
except ImportError:
    _OPENAI = False


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic Dependency Matrix
# Maps (action) → list of (condition_key, threshold, operator) tuples that
# flag a blast-radius violation. operator: "gt" = greater-than, "lt" = less-than
# ══════════════════════════════════════════════════════════════════════════════
_BLAST_MATRIX: Dict[str, List[Dict[str, Any]]] = {
    # circuit_breaker on already-degraded network → cascading isolation
    "circuit_breaker": [
        {"metric": "Network_Health",      "threshold": 20.0, "op": "lt"},
    ],
    # schema_failover while traffic is already throttled → split-brain risk
    "schema_failover": [
        {"metric": "Traffic_Load",        "threshold": 90.0, "op": "gt"},
    ],
    # restart_pods on critical DB temp → SPOF failure window
    "restart_pods": [
        {"metric": "Database_Temperature","threshold": 92.0, "op": "gt"},
    ],
    # cache_flush when network already partitioned
    "cache_flush": [
        {"metric": "Network_Health",      "threshold": 15.0, "op": "lt"},
    ],
}


def _check_blast_radius(action: str, obs_before: Dict[str, Any]) -> List[str]:
    """
    Return a list of violation strings if the action violates the DDM.
    Empty list = safe.
    """
    state = obs_before.get("observation", obs_before)
    rules = _BLAST_MATRIX.get(action, [])
    violations = []
    for rule in rules:
        val = state.get(rule["metric"], 50.0)
        if rule["op"] == "lt" and val < rule["threshold"]:
            violations.append(
                f"{action} is unsafe: {rule['metric']}={val:.1f} < {rule['threshold']}")
        elif rule["op"] == "gt" and val > rule["threshold"]:
            violations.append(
                f"{action} is unsafe: {rule['metric']}={val:.1f} > {rule['threshold']}")
    return violations


# ══════════════════════════════════════════════════════════════════════════════
# Format Reward
# ══════════════════════════════════════════════════════════════════════════════
_VALID_ACTIONS = [
    "throttle_traffic","load_balance","schema_failover","cache_flush",
    "circuit_breaker","restart_pods","scale_out","noop",
]

def _format_reward(completion: str) -> float:
    """
    +10 if the completion is a valid JSON micro-intent with correct keys.
    0   otherwise.
    """
    try:
        s, e = completion.find("{"), completion.rfind("}") + 1
        if s == -1 or e == 0: return 0.0
        p = json.loads(completion[s:e])
        if (isinstance(p.get("intent"), str) and p["intent"] in _VALID_ACTIONS
                and isinstance(p.get("confidence"), (int, float))
                and 0.0 <= float(p["confidence"]) <= 1.0
                and isinstance(p.get("rationale"), str)
                and len(p["rationale"]) > 5):
            return 10.0
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# State Recovery Reward (fully deterministic)
# ══════════════════════════════════════════════════════════════════════════════
def _state_recovery_reward(obs_before: Dict[str, Any],
                            obs_after:  Dict[str, Any]) -> float:
    """
    Reward based on improvement in the three PyTorch state tensor metrics.

    Score formula:
      slo_delta = slo_after − slo_before   ∈ [−1, +1]
      raw       = slo_delta × 100          ∈ [−100, +100]
      clamped   = clamp(raw, −50, +100)
      final     = clamped + 50             ∈ [0, +150] → then normalise to [50, 100] on positive

    Returns a value in [−50, +100].
    """
    def _slo(obs: Dict) -> float:
        s = obs.get("observation", obs)
        tl = s.get("Traffic_Load", 50.0)
        dt = s.get("Database_Temperature", 50.0)
        nh = s.get("Network_Health", 50.0)
        return ((100 - tl) + (100 - dt) + nh) / 300.0

    slo_before = obs_before.get("observation", {}).get("slo_score") or _slo(obs_before)
    slo_after  = obs_after.get("observation",  {}).get("slo_score") or _slo(obs_after)
    delta      = slo_after - slo_before            # ∈ [−1, +1]
    raw        = delta * 100.0                     # ∈ [−100, +100]

    if raw >= 0:
        # Map [0, +100] → [+50, +100]
        return 50.0 + raw * 0.5
    else:
        # Map [−100, 0] → [−50, 0]
        return max(-50.0, raw * 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# LLM Reasoning Score (SLOW PATH only, max +20)
# ══════════════════════════════════════════════════════════════════════════════
_REASONING_PROMPT = """You are an expert SRE evaluator.
Rate the quality of the following ChatOps negotiation transcript on a scale from 0 to 20.

Criteria:
  - Accuracy of root-cause analysis (0–8)
  - Justification quality for the chosen action (0–6)
  - Risk awareness / blast-radius consideration (0–6)

Output ONLY a JSON object: {{"score": <integer 0-20>, "reason": "<one sentence>"}}

Transcript:
{transcript}"""

def _llm_reasoning_score(
    chat_history: List[str],
    routing_path: str,
    openai_client: Optional[Any],
) -> float:
    """
    +10 to +20 via GPT-4o-mini only when routing_path == "slow_path".
    Returns 0.0 for non-slow paths or when no client is available.
    """
    if routing_path != "slow_path" or not openai_client or not chat_history:
        return 0.0
    transcript = "\n".join(chat_history[-8:])
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an SRE evaluation assistant."},
                {"role": "user",   "content": _REASONING_PROMPT.format(transcript=transcript)},
            ],
        )
        raw    = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        raw_score = float(parsed.get("score", 0))
        # Clamp to [0, 20], then scale to [+10, +20]
        clamped = max(0.0, min(20.0, raw_score))
        return 10.0 + clamped * 0.5   # maps [0,20] → [10,20]
    except Exception as exc:
        logger.warning("LLM reasoning score failed: %s", exc)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main Evaluator
# ══════════════════════════════════════════════════════════════════════════════
class MultiComponentEvaluator:
    """
    Thread-safe evaluator that computes all reward components and logs
    individual columns to W&B for full transparency.

    Usage:
        evaluator = MultiComponentEvaluator()
        reward_dict = evaluator.score(
            completion, action, obs_before, obs_after,
            routing_path, chat_history, confidence)
        total_reward = reward_dict["total"]
    """

    def __init__(self) -> None:
        self._openai_client: Optional[Any] = None
        if _OPENAI:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                self._openai_client = _OpenAI(api_key=api_key)

    # ── public API ────────────────────────────────────────────────────────────
    def score(
        self,
        completion:    str,
        action:        str,
        obs_before:    Dict[str, Any],
        obs_after:     Dict[str, Any],
        routing_path:  str,
        chat_history:  List[str],
        confidence:    float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all reward components and return a dict with individual
        scores plus the weighted 'total'.

        Parameters
        ----------
        completion   : raw model output string
        action       : parsed action string from the completion
        obs_before   : observation dict before the step
        obs_after    : observation dict after the step (from env /step response)
        routing_path : "fast_path" | "middle_path" | "slow_path"
        chat_history : list of recent agent chat strings
        confidence   : parsed confidence float from the model

        Returns
        -------
        dict with keys:
          format_reward, blast_radius_penalty, state_recovery_reward,
          llm_reasoning_score, total
        """
        # 1. Format reward
        fmt = _format_reward(completion)

        # 2. Blast radius penalty
        violations = _check_blast_radius(action, obs_before)
        blast = -50.0 if violations else 0.0
        if violations:
            logger.info("BLAST RADIUS VIOLATION: %s", violations)

        # 3. State recovery reward (deterministic)
        recovery = _state_recovery_reward(obs_before, obs_after)

        # 4. LLM reasoning score (slow path only)
        reasoning = _llm_reasoning_score(
            chat_history, routing_path, self._openai_client)

        total = fmt + blast + recovery + reasoning

        reward_dict = {
            "format_reward":         fmt,
            "blast_radius_penalty":  blast,
            "state_recovery_reward": recovery,
            "llm_reasoning_score":   reasoning,
            "total":                 total,
        }

        # Log individual columns to W&B
        if _WANDB and _is_wandb_active():
            wandb.log({f"eval/{k}": v for k, v in reward_dict.items()})

        return reward_dict

    def batch_score(
        self,
        completions:  List[str],
        actions:      List[str],
        obs_befores:  List[Dict],
        obs_afters:   List[Dict],
        routing_path: str,
        chat_history: List[str],
        confidences:  Optional[List[float]] = None,
    ) -> List[Dict[str, float]]:
        """Vectorised convenience wrapper over score()."""
        if confidences is None:
            confidences = [0.5] * len(completions)
        return [
            self.score(comp, act, ob, oa, routing_path, chat_history, conf)
            for comp, act, ob, oa, conf
            in zip(completions, actions, obs_befores, obs_afters, confidences)
        ]


# ── helpers ───────────────────────────────────────────────────────────────────
def _is_wandb_active() -> bool:
    try: return wandb.run is not None
    except Exception: return False


# ── legacy shim (keeps old call-sites working) ────────────────────────────────
class Evaluator(MultiComponentEvaluator):
    """Backwards-compatible alias."""
    pass
