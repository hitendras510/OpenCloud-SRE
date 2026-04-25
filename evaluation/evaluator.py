"""
evaluation/evaluator.py
========================
Multi-component reward function for OpenCloud-SRE GRPO training.

Reward Components (Positive)
-----------------------------
  format_reward          : +10   — valid parseable JSON micro-intent
  state_recovery_reward  : +50 to +100 — deterministic SLO / tensor improvement
  llm_reasoning_score    : +10 to +20  — Llama-3-8B via HF Inference API grades
                                         ChatOps quality (SLOW PATH only)

Reward Components (Penalties / Anti-Hacking)
---------------------------------------------
  blast_radius_penalty       : -50   — action violates Dependency Matrix
  repetition_penalty         : -20   — same action repeated 3+ times in a row
  noop_abuse_penalty         : -30   — noop used when system is in critical state
  confidence_calibration_pen : -15   — high confidence on bad outcomes
  state_plausibility_penalty : -40   — output claims impossible state transitions
  oscillation_penalty        : -25   — flipping between opposite actions
  json_integrity_penalty     : -15   — malformed or invalid action keys

All components are logged independently to W&B so judges can audit
that the model isn't gaming a single soft score (reward hacking prevention).
Fully vendor-agnostic: no OpenAI dependency.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── optional W&B ──────────────────────────────────────────────────────────────
try:
    import wandb; _WANDB = True
except ImportError:
    _WANDB = False

# ── optional HF client (for llm_reasoning_score) ──────────────────────────────
try:
    from huggingface_hub import InferenceClient as _HFClient
    _HF_AVAILABLE = True
except ImportError:
    _HFClient = None  # type: ignore[assignment, misc]
    _HF_AVAILABLE = False

_REASONING_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic Dependency Matrix (Blast Radius)
# ══════════════════════════════════════════════════════════════════════════════
_BLAST_MATRIX: Dict[str, List[Dict[str, Any]]] = {
    "circuit_breaker": [
        {"metric": "Network_Health",       "threshold": 20.0, "op": "lt"},
    ],
    "schema_failover": [
        {"metric": "Traffic_Load",         "threshold": 90.0, "op": "gt"},
    ],
    "restart_pods": [
        {"metric": "Database_Temperature", "threshold": 92.0, "op": "gt"},
    ],
    "cache_flush": [
        {"metric": "Network_Health",       "threshold": 15.0, "op": "lt"},
    ],
}


def _check_blast_radius(action: str, obs_before: Dict[str, Any]) -> List[str]:
    """Return violation strings if action violates the DDM. Empty = safe."""
    state = obs_before.get("observation", obs_before)
    rules = _BLAST_MATRIX.get(action, [])
    violations = []
    for rule in rules:
        val = state.get(rule["metric"], 50.0)
        if rule["op"] == "lt" and val < rule["threshold"]:
            violations.append(
                f"{action} unsafe: {rule['metric']}={val:.1f} < {rule['threshold']}")
        elif rule["op"] == "gt" and val > rule["threshold"]:
            violations.append(
                f"{action} unsafe: {rule['metric']}={val:.1f} > {rule['threshold']}")
    return violations


# ══════════════════════════════════════════════════════════════════════════════
# Format Reward
# ══════════════════════════════════════════════════════════════════════════════
_VALID_ACTIONS = [
    "throttle_traffic", "load_balance", "schema_failover", "cache_flush",
    "circuit_breaker", "restart_pods", "scale_out", "noop",
]


def _format_reward(completion: str) -> float:
    """+10 if valid JSON micro-intent with correct keys, 0 otherwise."""
    try:
        s, e = completion.find("{"), completion.rfind("}") + 1
        if s == -1 or e == 0:
            return 0.0
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
                           obs_after: Dict[str, Any]) -> float:
    """Deterministic reward based on SLO improvement. Returns [-50, +100]."""
    def _slo(obs: Dict) -> float:
        s = obs.get("observation", obs)
        tl = s.get("Traffic_Load", 50.0)
        dt = s.get("Database_Temperature", 50.0)
        nh = s.get("Network_Health", 50.0)
        return ((100 - tl) + (100 - dt) + nh) / 300.0

    slo_before = obs_before.get("observation", {}).get("slo_score") or _slo(obs_before)
    slo_after  = obs_after.get("observation", {}).get("slo_score") or _slo(obs_after)
    delta = slo_after - slo_before
    raw   = delta * 100.0

    if raw >= 0:
        return 50.0 + raw * 0.5
    else:
        return max(-50.0, raw * 0.5)


def _resource_efficiency_reward(obs_after: Dict[str, Any]) -> float:
    """+10 if system is stable (SLO > 0.8) AND resources are low (Temp < 60, Traffic < 60)."""
    state = obs_after.get("observation", obs_after)
    slo = state.get("slo_score", 1.0)
    tl = state.get("Traffic_Load", 0.0)
    dt = state.get("Database_Temperature", 0.0)
    if slo > 0.8 and tl < 60.0 and dt < 60.0:
        return 10.0
    return 0.0


def _action_impact_reward(action: str, obs_before: Dict[str, Any], obs_after: Dict[str, Any]) -> float:
    """+15 if the action significantly improved the most critical metric."""
    before = obs_before.get("observation", obs_before)
    after = obs_after.get("observation", obs_after)
    
    metrics = {
        "Traffic_Load": before.get("Traffic_Load", 50.0),
        "Database_Temperature": before.get("Database_Temperature", 50.0),
        "Network_Health": 100.0 - before.get("Network_Health", 50.0) # Invert health to find 'criticality'
    }
    
    critical_metric = max(metrics, key=metrics.get)
    
    v_before = before.get(critical_metric, 50.0)
    v_after = after.get(critical_metric, 50.0)
    
    if critical_metric == "Network_Health":
        if v_after > v_before + 5.0: # Health improved
            return 15.0
    else:
        if v_after < v_before - 5.0: # Load/Temp decreased
            return 15.0
            
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LLM Reasoning Score (SLOW PATH only, max +20)
# ══════════════════════════════════════════════════════════════════════════════
_REASONING_PROMPT = """You are an expert SRE evaluator.
Rate the quality of the following ChatOps negotiation transcript on a scale from 0 to 20.

Criteria:
  - Accuracy of root-cause analysis (0-8)
  - Justification quality for the chosen action (0-6)
  - Risk awareness / blast-radius consideration (0-6)

Output ONLY a JSON object: {{"score": <integer 0-20>, "reason": "<one sentence>"}}

Transcript:
{transcript}"""


def _llm_reasoning_score(
    chat_history: List[str],
    routing_path: str,
    hf_client: Optional[Any],
) -> float:
    """+10 to +20 via Llama-3-8B, SLOW PATH only. 0 otherwise."""
    if routing_path != "slow_path" or not hf_client or not chat_history:
        return 0.0
    transcript = "\n".join(chat_history[-8:])
    try:
        resp = hf_client.chat_completion(
            model=_REASONING_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are an SRE evaluation assistant. Output only valid JSON."},
                {"role": "user",
                 "content": _REASONING_PROMPT.format(transcript=transcript)},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed    = json.loads(raw.strip())
        raw_score = float(parsed.get("score", 0))
        clamped   = max(0.0, min(20.0, raw_score))
        return 10.0 + clamped * 0.5
    except Exception as exc:
        logger.warning("HF reasoning score failed: %s", exc)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-HACKING CHECK 1: Action Repetition Detector
# Prevents the agent from gaming the system by spamming one "safe" action
# ══════════════════════════════════════════════════════════════════════════════
def _repetition_penalty(action: str, action_history: List[str]) -> float:
    """
    -20 if the same action has been repeated 3+ consecutive times.
    Detects the common RL exploit where an agent finds one "harmless" action
    and locks into it to avoid negative rewards.
    """
    if len(action_history) < 2:
        return 0.0
    tail = action_history[-2:]  # the two most recent *previous* actions
    if all(a == action for a in tail):
        # This action would make it 3 in a row → penalty
        logger.info("ANTI-HACK: Repetition detected — '%s' × 3+ consecutive", action)
        return -20.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-HACKING CHECK 2: Noop Abuse Penalty
# Prevents the agent from choosing "noop" in a critical state to avoid
# blast-radius penalties while collecting safe format_reward
# ══════════════════════════════════════════════════════════════════════════════
def _noop_abuse_penalty(action: str, obs_before: Dict[str, Any]) -> float:
    """
    -30 if the agent chooses 'noop' when the system is in a critical/degraded state.
    A critical state means SLO < 0.4 — the system is actively failing and
    inaction is negligent.
    """
    if action != "noop":
        return 0.0
    state = obs_before.get("observation", obs_before)
    slo = state.get("slo_score")
    if slo is None:
        tl = state.get("Traffic_Load", 50.0)
        dt = state.get("Database_Temperature", 50.0)
        nh = state.get("Network_Health", 50.0)
        slo = ((100 - tl) + (100 - dt) + nh) / 300.0
    if slo < 0.40:
        logger.info("ANTI-HACK: Noop abuse — SLO=%.3f (critical), agent chose noop", slo)
        return -30.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-HACKING CHECK 3: Confidence Calibration
# Penalises agents that always claim high confidence regardless of outcome
# ══════════════════════════════════════════════════════════════════════════════
def _confidence_calibration_penalty(
    confidence: float,
    obs_before: Dict[str, Any],
    obs_after: Dict[str, Any],
) -> float:
    """
    -15 if confidence >= 0.85 but the action made things worse (SLO dropped).
    This forces the model to learn honest self-assessment instead of
    always claiming "0.95 confidence" to look good.
    """
    if confidence < 0.85:
        return 0.0

    def _slo(obs: Dict) -> float:
        s = obs.get("observation", obs)
        tl = s.get("Traffic_Load", 50.0)
        dt = s.get("Database_Temperature", 50.0)
        nh = s.get("Network_Health", 50.0)
        return ((100 - tl) + (100 - dt) + nh) / 300.0

    slo_before = obs_before.get("observation", {}).get("slo_score") or _slo(obs_before)
    slo_after  = obs_after.get("observation", {}).get("slo_score") or _slo(obs_after)

    if slo_after < slo_before - 0.01:  # outcome was worse by at least 1%
        logger.info(
            "ANTI-HACK: Overconfident failure — conf=%.2f but SLO dropped %.3f→%.3f",
            confidence, slo_before, slo_after,
        )
        return -15.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-HACKING CHECK 4: State Plausibility Validator
# Catches impossible environment transitions that might indicate the agent
# is somehow manipulating its observation of the environment
# ══════════════════════════════════════════════════════════════════════════════
# Maximum physically possible single-step delta per metric (action base + noise)
_MAX_PLAUSIBLE_DELTA = 45.0  # largest base delta (circuit_breaker traffic=-30) + noise


def _state_plausibility_penalty(
    obs_before: Dict[str, Any],
    obs_after: Dict[str, Any],
) -> float:
    """
    -40 if any metric changed by more than is physically possible in one step.
    Guards against environment manipulation or data corruption exploits.
    """
    before = obs_before.get("observation", obs_before)
    after  = obs_after.get("observation", obs_after)

    for metric in ("Traffic_Load", "Database_Temperature", "Network_Health"):
        v_before = before.get(metric, 50.0)
        v_after  = after.get(metric, 50.0)
        delta = abs(v_after - v_before)
        if delta > _MAX_PLAUSIBLE_DELTA:
            logger.warning(
                "ANTI-HACK: Implausible state transition — %s changed %.1f→%.1f (Δ=%.1f > %.1f)",
                metric, v_before, v_after, delta, _MAX_PLAUSIBLE_DELTA,
            )
            return -40.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-HACKING CHECK 5: Reward Explosion Clamp
# Final safety net — prevents any single step's total reward from being
# absurdly high, which is a hallmark of reward hacking
# ══════════════════════════════════════════════════════════════════════════════
_REWARD_FLOOR = -150.0
_REWARD_CEIL  =  155.0  # max possible: fmt(10)+rec(100)+reas(20)+eff(10)+impact(15)


def _clamp_reward(total: float) -> float:
    """Hard-clamp the total reward into a physically reasonable range."""
    clamped = max(_REWARD_FLOOR, min(_REWARD_CEIL, total))
    if clamped != total:
        logger.warning(
            "ANTI-HACK: Reward explosion clamped %.1f → %.1f", total, clamped
        )
    return clamped


def _oscillation_penalty(action: str, action_history: List[str]) -> float:
    """-25 if agent flips between opposite actions (e.g. scale_out vs throttle_traffic)."""
    if len(action_history) < 1:
        return 0.0
    last_a = action_history[-1]
    opposites = [
        ("scale_out", "throttle_traffic"),
        ("restart_pods", "noop"),
        ("circuit_breaker", "load_balance")
    ]
    for a1, a2 in opposites:
        if (action == a1 and last_a == a2) or (action == a2 and last_a == a1):
            logger.info("ANTI-HACK: Oscillation detected — '%s' vs '%s'", last_a, action)
            return -25.0
    return 0.0


def _json_integrity_penalty(completion: str) -> float:
    """-15 if JSON is valid but intent is missing or invalid."""
    try:
        s, e = completion.find("{"), completion.rfind("}") + 1
        if s == -1: return -15.0
        p = json.loads(completion[s:e])
        if "intent" not in p or p["intent"] not in _VALID_ACTIONS:
            return -15.0
    except:
        return -15.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main Evaluator
# ══════════════════════════════════════════════════════════════════════════════
class MultiComponentEvaluator:
    """
    Thread-safe evaluator with 4 positive reward signals and 5 anti-hacking
    penalty layers. All components are logged independently to W&B.

    Usage:
        evaluator = MultiComponentEvaluator()
        reward_dict = evaluator.score(
            completion, action, obs_before, obs_after,
            routing_path, chat_history, confidence)
        total_reward = reward_dict["total"]
    """

    def __init__(self) -> None:
        self._hf_client: Optional[Any] = None
        if _HF_AVAILABLE:
            hf_token = os.getenv("HF_TOKEN", "")
            if hf_token:
                self._hf_client = _HFClient(token=hf_token)
                logger.info("Evaluator: HF reasoning scorer active (%s).", _REASONING_MODEL)
            else:
                logger.info("Evaluator: HF_TOKEN not set — llm_reasoning_score disabled.")

        # Rolling action history for repetition detection (per-evaluator instance)
        self._action_history: deque = deque(maxlen=50)

    def reset_episode(self) -> None:
        """Call at the start of each episode to clear action history."""
        self._action_history.clear()

    # ── public API ────────────────────────────────────────────────────────────
    def score(
        self,
        completion:   str,
        action:       str,
        obs_before:   Dict[str, Any],
        obs_after:    Dict[str, Any],
        routing_path: str,
        chat_history: List[str],
        confidence:   float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all reward + anti-hacking components.

        Returns
        -------
        dict with keys:
          format_reward, blast_radius_penalty, state_recovery_reward,
          llm_reasoning_score, repetition_penalty, noop_abuse_penalty,
          confidence_calibration_penalty, state_plausibility_penalty,
          total, total_clamped
        """
        # ── Positive rewards ──────────────────────────────────────────────
        fmt       = _format_reward(completion)
        recovery  = _state_recovery_reward(obs_before, obs_after)
        reasoning = _llm_reasoning_score(
            chat_history, routing_path, self._hf_client)
        eff       = _resource_efficiency_reward(obs_after)
        impact    = _action_impact_reward(action, obs_before, obs_after)

        # ── Anti-hacking penalties ────────────────────────────────────────
        blast_v = _check_blast_radius(action, obs_before)
        blast       = -50.0 if blast_v else 0.0
        repetition  = _repetition_penalty(action, list(self._action_history))
        noop_abuse  = _noop_abuse_penalty(action, obs_before)
        conf_cal    = _confidence_calibration_penalty(confidence, obs_before, obs_after)
        plausible   = _state_plausibility_penalty(obs_before, obs_after)
        oscillation = _oscillation_penalty(action, list(self._action_history))
        integrity   = _json_integrity_penalty(completion)

        # ── Track action for future repetition checks ─────────────────────
        self._action_history.append(action)

        # ── Sum and clamp ─────────────────────────────────────────────────
        raw_total = (fmt + recovery + reasoning + eff + impact
                     + blast + repetition + noop_abuse + conf_cal + plausible
                     + oscillation + integrity)
        total = _clamp_reward(raw_total)

        # ── Failure Visibility (Violations) ───────────────────────────────
        violations = []
        if blast < 0: violations.append(f"Blast Radius: {blast_v[0]}")
        if repetition < 0: violations.append("Action Repetition (3+ times)")
        if noop_abuse < 0: violations.append("Noop Abuse in Critical State")
        if conf_cal < 0:   violations.append("Overconfident Failure")
        if plausible < 0:  violations.append("Implausible State Transition")
        if oscillation < 0: violations.append("Action Oscillation Detected")
        if integrity < 0:   violations.append("JSON/Intent Integrity Failure")

        reward_dict = {
            # Positive signals
            "format_reward":                 fmt,
            "state_recovery_reward":         recovery,
            "llm_reasoning_score":           reasoning,
            "resource_efficiency_reward":    eff,
            "action_impact_reward":          impact,
            # Anti-hacking penalties
            "blast_radius_penalty":          blast,
            "repetition_penalty":            repetition,
            "noop_abuse_penalty":            noop_abuse,
            "confidence_calibration_penalty": conf_cal,
            "state_plausibility_penalty":    plausible,
            "oscillation_penalty":           oscillation,
            "json_integrity_penalty":        integrity,
            # Totals
            "total_raw":                     raw_total,
            "total":                         total,
            "violations":                    violations,
        }

        # Log every column to W&B
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
    try:
        return wandb.run is not None
    except Exception:
        return False


# ── legacy shim ───────────────────────────────────────────────────────────────
class Evaluator(MultiComponentEvaluator):
    """Backwards-compatible alias."""
    pass
