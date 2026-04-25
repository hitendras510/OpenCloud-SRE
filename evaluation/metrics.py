"""
evaluation/metrics.py
======================
Deterministic episode-level SRE metrics computed from raw trajectory data.
Used by both the Streamlit UI (live display) and the GRPO trainer (W&B logging).

All functions are pure — no side effects, no external calls.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ── Per-step helpers ──────────────────────────────────────────────────────────

def slo_score(traffic: float, db_temp: float, net_health: float) -> float:
    """
    Scalar SLO health score in [0.0, 1.0].
    1.0 = perfect, 0.0 = fully degraded.
    """
    return ((100 - traffic) + (100 - db_temp) + net_health) / 300.0


def is_critical(traffic: float, db_temp: float, net_health: float) -> bool:
    """True if any metric breaches its critical threshold."""
    return traffic >= 85.0 or db_temp >= 80.0 or net_health <= 30.0


# ── Episode-level metrics ─────────────────────────────────────────────────────

def episode_summary(
    trajectory: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute aggregate metrics from a list of step records.

    Each step dict must contain:
      - Traffic_Load, Database_Temperature, Network_Health  (floats)
      - reward  (float)
      - action  (str)
      - [optional] slo_score (float) — computed if absent

    Returns
    -------
    dict with keys:
      mean_slo, max_slo, min_slo, final_slo,
      mean_reward, total_reward,
      slo_improvement,        # final_slo - initial_slo
      recovery_rate,          # fraction of steps with improving SLO
      critical_fraction,      # fraction of steps in critical state
      blast_radius_violations # int count from "blast_radius_penalty" key
    """
    if not trajectory:
        return {}

    slo_scores = [
        s.get("slo_score") or slo_score(
            s.get("Traffic_Load", 50.0),
            s.get("Database_Temperature", 50.0),
            s.get("Network_Health", 50.0),
        )
        for s in trajectory
    ]

    rewards         = [s.get("reward", 0.0) for s in trajectory]
    critical_steps  = sum(
        1 for s in trajectory
        if is_critical(
            s.get("Traffic_Load", 50.0),
            s.get("Database_Temperature", 50.0),
            s.get("Network_Health", 50.0),
        )
    )
    blast_violations = sum(
        1 for s in trajectory
        if s.get("blast_radius_penalty", 0.0) < 0
    )
    improving_steps = sum(
        1 for i in range(1, len(slo_scores))
        if slo_scores[i] > slo_scores[i - 1]
    )

    n = len(trajectory)
    return {
        "mean_slo":             round(sum(slo_scores) / n, 4),
        "max_slo":              round(max(slo_scores), 4),
        "min_slo":              round(min(slo_scores), 4),
        "final_slo":            round(slo_scores[-1], 4),
        "initial_slo":          round(slo_scores[0], 4),
        "slo_improvement":      round(slo_scores[-1] - slo_scores[0], 4),
        "mean_reward":          round(sum(rewards) / n, 4),
        "total_reward":         round(sum(rewards), 4),
        "recovery_rate":        round(improving_steps / max(n - 1, 1), 4),
        "critical_fraction":    round(critical_steps / n, 4),
        "blast_radius_violations": blast_violations,
        "steps":                n,
    }


def action_distribution(trajectory: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count how many times each action was taken in the episode."""
    counts: Dict[str, int] = {}
    for step in trajectory:
        a = step.get("action", "unknown")
        counts[a] = counts.get(a, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def rolling_slo(
    trajectory: List[Dict[str, Any]],
    window: int = 5,
) -> List[float]:
    """Return a smoothed rolling-average SLO over the episode."""
    scores = [
        s.get("slo_score") or slo_score(
            s.get("Traffic_Load", 50.0),
            s.get("Database_Temperature", 50.0),
            s.get("Network_Health", 50.0),
        )
        for s in trajectory
    ]
    out = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        out.append(round(sum(scores[start: i + 1]) / (i - start + 1), 4))
    return out
