"""
graph/routing_logic.py
=======================
Pure routing decision functions used by the LangGraph sre_graph.

These are extracted from sre_graph.py into their own module so they can be:
  - Unit-tested without spinning up LangGraph
  - Reused by the GRPO trainer's rollout loop to determine routing_path
  - Inspected by judges without reading 600+ lines of graph orchestration

Routing Tiers
-------------
  FAST  (fast_path)   — DNA Memory HIGH match  → immediate execution
  MID   (middle_path) — Medium match + consensus GREEN → standard execution
  SLOW  (slow_path)   — Low match OR consensus RED → ChatOps / human escrow
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.constants import (
    DNA_HIGH_MATCH_THRESHOLD,
    DNA_MEDIUM_MATCH_THRESHOLD,
    ROUTING_FAST_PATH,
    ROUTING_MIDDLE_PATH,
    ROUTING_SLOW_PATH,
)


# ── DNA Memory routing decision ───────────────────────────────────────────────

def decide_routing_from_dna(
    dna_hit: Dict[str, Any],
) -> str:
    """
    Determine the routing tier based on DNA memory distance.

    Parameters
    ----------
    dna_hit : dict
        Output of DNAMemory.query().to_dict() — must contain 'distance'.

    Returns
    -------
    str
        One of "fast_path", "middle_path", "slow_path".
    """
    distance = float(dna_hit.get("distance", 999.0))
    confidence = dna_hit.get("confidence", "Low Match")

    if confidence == "High Match" or distance < DNA_HIGH_MATCH_THRESHOLD:
        return ROUTING_FAST_PATH
    elif confidence == "Medium Match" or distance < DNA_MEDIUM_MATCH_THRESHOLD:
        return ROUTING_MIDDLE_PATH
    else:
        return ROUTING_SLOW_PATH


# ── Shadow Consensus routing decision ─────────────────────────────────────────

def decide_consensus(
    net_intent: str,
    db_intent: str,
    net_confidence: float,
    db_confidence: float,
) -> Tuple[str, str]:
    """
    Shadow Consensus: check if Network and DB controllers agree on a strategy.

    Returns
    -------
    Tuple[str, str]
        (consensus_status, recommended_action)
        consensus_status: "green" (agree) | "red" (conflict)
    """
    # Intent synergy map — compatible pairs
    _SYNERGIES: Dict[Tuple[str, str], str] = {
        ("throttle_traffic", "cache_flush"):    "throttle_traffic",
        ("throttle_traffic", "noop"):           "throttle_traffic",
        ("circuit_breaker",  "schema_failover"):"scale_out",
        ("circuit_breaker",  "restart_pods"):   "circuit_breaker",
        ("load_balance",     "cache_flush"):    "load_balance",
        ("load_balance",     "noop"):           "load_balance",
        ("scale_out",        "schema_failover"):"scale_out",
        ("scale_out",        "cache_flush"):    "scale_out",
        ("noop",             "noop"):           "noop",
    }

    key = (net_intent, db_intent)
    rev = (db_intent, net_intent)

    if key in _SYNERGIES:
        return "green", _SYNERGIES[key]
    if rev in _SYNERGIES:
        return "green", _SYNERGIES[rev]

    # If intents conflict, escalate to slow path unless both are high-confidence
    if net_confidence >= 0.80 and db_confidence >= 0.80:
        # Both are confident but disagree — default to the one with higher confidence
        winner = net_intent if net_confidence >= db_confidence else db_intent
        return "red", winner
    return "red", "noop"


# ── Blast radius gate ─────────────────────────────────────────────────────────

def should_escrow(
    action: str,
    blast_radius_violations: list,
    trust_score: float,
) -> bool:
    """
    Determine whether the action should be held in human escrow.

    An action is escrowed if:
      1. There are active blast-radius violations, OR
      2. The agent's trust score is below 0.60 (Adaptive Trust Layer)

    Parameters
    ----------
    action : str
        The proposed action.
    blast_radius_violations : list
        List of violation strings from _check_blast_radius().
    trust_score : float
        Agent's running trust score in [0, 1].

    Returns
    -------
    bool
        True = hold for human approval; False = auto-execute.
    """
    if blast_radius_violations:
        return True
    if trust_score < 0.60:
        return True
    return False
