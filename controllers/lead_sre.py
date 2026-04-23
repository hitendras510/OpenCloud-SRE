"""
controllers/lead_sre.py
========================
Lead SRE Orchestrator — Cognitive Incident Compression Architecture.

This node is the central governance gate of the LangGraph pipeline.
Every proposed action must pass through three sequential filters before
being approved for execution.

Filter 1 – Shadow Consensus Check (Compute Compression)
--------------------------------------------------------
  Reads micro-intents from the Network and DB agents stored in the
  message_bus state.  If they conflict → DEEP_NEGOTIATE (no LLM spend).
  If they align → derive a resolved action and proceed to Filter 2.

Filter 2 – Predictive Blast Radius Filter (Deterministic Safety)
-----------------------------------------------------------------
  A static dictionary maps every action to its known secondary impacts
  (e.g., schema_failover → temporary_read_latency).  If a proposed action
  triggers a CRITICAL secondary impact the node:
    • Rejects the action (BLAST_RADIUS_BLOCK).
    • Appends a "Blast Radius Warning" to the chat log so the sub-agents
      can see it in their next context window and propose a safer alternative.

Filter 3 – Adaptive Trust Layer (Escrow Execution)
----------------------------------------------------
  Checks the combined confidence score of the two sub-agents.
    confidence >= 0.90  AND  blast radius passed  →  AUTO_RESOLVE
    confidence <  0.90  AND  blast radius passed  →  HUMAN_ESCALATION
      (execution is paused; the UI 'Approve' button sets human_approved=True)

Routing Signals (stored in state["governance_signal"])
------------------------------------------------------
  AUTO_RESOLVE       – All clear; executor runs immediately.
  HUMAN_ESCALATION   – Awaiting human approval in the UI.
  DEEP_NEGOTIATE     – Intent conflict; ChatOps node invoked next.
  BLAST_RADIUS_BLOCK – Action rejected; agents must propose a safer intent.

LangGraph Integration
---------------------
  Use ``LeadSRENode().run_as_node`` or the ``make_lead_sre_node()`` factory.
  The node reads/writes the shared SREGraphState dict from message_bus.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graph.message_bus import (
    SREGraphState,
    NetworkIntent,
    DBIntent,
    RoutingPath,
    ConsensusStatus,
    GovernanceSignal,
    BlastRiskLevel,
    TrustDecision,
    append_chat,
)
from controllers.system_prompts import LEAD_SRE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── optional OpenAI (graceful no-op without API key) ─────────────────────────
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore[assignment, misc]

_DEFAULT_MODEL = "gpt-4o-mini"

# ─────────────────────────────────────────────────────────────────────────────
# FILTER 1 — SHADOW CONSENSUS: synergy matrix
# Maps (network_intent, db_intent) → (signal, resolved_action | None)
# ─────────────────────────────────────────────────────────────────────────────

# Routing signal constants (kept as plain strings for readability in logs)
_FAST = "FAST_EXECUTE"
_CONFLICT = "DEEP_NEGOTIATE"

_SYNERGY_MATRIX: Dict[Tuple[str, str], Tuple[str, Optional[str]]] = {
    # ── Clear-cut GREEN (single-domain actions) ───────────────────────────
    ("throttle",      "noop"):        (_FAST,     "throttle_traffic"),
    ("circuit_break", "noop"):        (_FAST,     "circuit_breaker"),
    ("load_balance",  "noop"):        (_FAST,     "load_balance"),
    ("scale_out",     "noop"):        (_FAST,     "scale_out"),
    ("noop",          "failover"):    (_FAST,     "schema_failover"),
    ("noop",          "cache_flush"): (_FAST,     "cache_flush"),
    ("noop",          "restart"):     (_FAST,     "restart_pods"),
    ("noop",          "noop"):        (_FAST,     "noop"),
    # ── Compound GREEN (compatible cross-domain pairs) ────────────────────
    ("throttle",      "failover"):    (_FAST,     "schema_failover"),
    ("throttle",      "cache_flush"): (_FAST,     "throttle_traffic"),
    ("throttle",      "restart"):     (_FAST,     "throttle_traffic"),
    ("scale_out",     "cache_flush"): (_FAST,     "scale_out"),
    ("scale_out",     "failover"):    (_FAST,     "scale_out"),
    ("load_balance",  "cache_flush"): (_FAST,     "cache_flush"),
    ("load_balance",  "restart"):     (_FAST,     "restart_pods"),
    # ── RED conflicts (dual-isolation / resource contention) ──────────────
    ("circuit_break", "failover"):    (_CONFLICT, None),
    ("circuit_break", "restart"):     (_CONFLICT, None),
    ("circuit_break", "cache_flush"): (_CONFLICT, None),
    ("load_balance",  "failover"):    (_CONFLICT, None),
    ("scale_out",     "restart"):     (_CONFLICT, None),
}

_NET_TO_ACTION: Dict[str, str] = {
    "throttle":      "throttle_traffic",
    "circuit_break": "circuit_breaker",
    "load_balance":  "load_balance",
    "scale_out":     "scale_out",
    "noop":          "noop",
}
_DB_TO_ACTION: Dict[str, str] = {
    "failover":    "schema_failover",
    "cache_flush": "cache_flush",
    "restart":     "restart_pods",
    "noop":        "noop",
}

# ─────────────────────────────────────────────────────────────────────────────
# FILTER 2 — BLAST RADIUS: secondary-impact catalogue
# Maps action_name → list of (impact_description, BlastRiskLevel)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SecondaryImpact:
    description: str
    risk: BlastRiskLevel
    # Services affected (used for UI display and prompt injection)
    affected_services: List[str] = field(default_factory=list)

# Any action whose max risk is CRITICAL will be blocked by the filter.
BLAST_RADIUS_MAP: Dict[str, List[SecondaryImpact]] = {
    "schema_failover": [
        SecondaryImpact(
            "Temporary read latency spike (5–30s) during replica promotion.",
            BlastRiskLevel.MEDIUM,
            ["read-replica", "reporting-service"],
        ),
        SecondaryImpact(
            "Auth service may lose DB connection during failover window.",
            BlastRiskLevel.HIGH,
            ["auth-service", "session-store"],
        ),
    ],
    "restart_pods": [
        SecondaryImpact(
            "Auth service dropout during rolling restart window.",
            BlastRiskLevel.HIGH,
            ["auth-service"],
        ),
        SecondaryImpact(
            "In-flight requests dropped; clients may see 502s.",
            BlastRiskLevel.MEDIUM,
            ["api-gateway"],
        ),
    ],
    "circuit_breaker": [
        SecondaryImpact(
            "Downstream dependent services receive fallback/empty responses.",
            BlastRiskLevel.MEDIUM,
            ["downstream-service", "mobile-api"],
        ),
        SecondaryImpact(
            "If applied during DB failover: dual isolation — total blackout.",
            BlastRiskLevel.CRITICAL,   # ← blocks execution if DB is also failing over
            ["all-services"],
        ),
    ],
    "throttle_traffic": [
        SecondaryImpact(
            "API gateway latency increases as token bucket refills.",
            BlastRiskLevel.LOW,
            ["api-gateway"],
        ),
    ],
    "load_balance": [
        SecondaryImpact(
            "Session affinity broken for sticky-session clients.",
            BlastRiskLevel.LOW,
            ["session-service"],
        ),
    ],
    "cache_flush": [
        SecondaryImpact(
            "Cold-cache miss storm hits primary DB for 30–120s.",
            BlastRiskLevel.HIGH,
            ["primary-db", "query-cache"],
        ),
    ],
    "scale_out": [
        SecondaryImpact(
            "Provisioning delay (60–300s) before new nodes accept traffic.",
            BlastRiskLevel.LOW,
            ["autoscaler"],
        ),
    ],
    "noop": [],  # No secondary impacts
}


def _evaluate_blast_radius(
    action: str,
    state_vec: List[float],
) -> Tuple[BlastRiskLevel, List[SecondaryImpact]]:
    """
    Evaluate the secondary impacts of *action* given the current system state.

    State-awareness: some impacts only become CRITICAL in specific contexts
    (e.g., circuit_breaker is only CRITICAL if DB is also in failover territory).

    Returns (max_risk_level, list_of_triggered_impacts).
    """
    traffic, db_temp, net_health = state_vec
    impacts = BLAST_RADIUS_MAP.get(action, [])

    triggered: List[SecondaryImpact] = []
    max_risk = BlastRiskLevel.NONE

    for impact in impacts:
        effective_risk = impact.risk

        # Context-aware risk escalation
        if action == "circuit_breaker" and impact.risk == BlastRiskLevel.CRITICAL:
            # Only truly CRITICAL if DB is simultaneously at failover threshold
            if db_temp < 80:
                effective_risk = BlastRiskLevel.MEDIUM  # downgrade if DB is healthy

        if action == "cache_flush" and db_temp > 85:
            # Cold-cache storm on an already overloaded DB is worse
            effective_risk = BlastRiskLevel.CRITICAL

        triggered.append(SecondaryImpact(
            description=impact.description,
            risk=effective_risk,
            affected_services=impact.affected_services,
        ))

        # Track the highest risk level seen
        risk_order = [
            BlastRiskLevel.NONE, BlastRiskLevel.LOW,
            BlastRiskLevel.MEDIUM, BlastRiskLevel.HIGH, BlastRiskLevel.CRITICAL,
        ]
        if risk_order.index(effective_risk) > risk_order.index(max_risk):
            max_risk = effective_risk

    return max_risk, triggered


def _format_blast_warning(
    action: str,
    impacts: List[SecondaryImpact],
    max_risk: BlastRiskLevel,
) -> str:
    """
    Build a "Blast Radius Warning" string that gets injected into the
    sub-agents' prompt context on the next iteration.
    """
    lines = [
        f"[BLAST RADIUS WARNING] Action '{action}' was REJECTED "
        f"(max_risk={max_risk.value.upper()}).",
        "Secondary impacts that triggered the block:",
    ]
    for imp in impacts:
        if imp.risk in (BlastRiskLevel.HIGH, BlastRiskLevel.CRITICAL):
            lines.append(
                f"  • [{imp.risk.value.upper()}] {imp.description} "
                f"(affects: {', '.join(imp.affected_services)})"
            )
    lines.append(
        "Please propose a safer alternative action that avoids these impacts."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 3 — ADAPTIVE TRUST LAYER
# ─────────────────────────────────────────────────────────────────────────────

# Confidence threshold for auto-execution without human approval
AUTO_RESOLVE_THRESHOLD: float = 0.90


def _compute_combined_confidence(
    net_intent: NetworkIntent,
    db_intent: DBIntent,
) -> float:
    """
    Combined confidence = weighted average of both agents' confidence scores.
    The agent with the higher individual confidence is weighted 60/40.
    """
    nc = net_intent["confidence"]
    dc = db_intent["confidence"]
    if nc >= dc:
        return round(nc * 0.60 + dc * 0.40, 4)
    return round(dc * 0.60 + nc * 0.40, 4)


# ─────────────────────────────────────────────────────────────────────────────
# LeadSRENode — the unified governance orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class LeadSRENode:
    """
    Central orchestrator implementing the three-filter governance pipeline.

    Parameters
    ----------
    use_llm : bool
        Enable GPT-4o arbitration for intent pairs not in the synergy matrix.
        Falls back to confidence-tiebreaker when False or no key available.
    model : str
        OpenAI model for LLM arbitration calls.
    auto_resolve_threshold : float
        Confidence floor for AUTO_RESOLVE (default 0.90).
    critical_blast_block : bool
        If True (default), CRITICAL blast radius rejects the action entirely.
        Set False to downgrade to HUMAN_ESCALATION for testing purposes.
    """

    def __init__(
        self,
        use_llm: bool = True,
        model: str = _DEFAULT_MODEL,
        auto_resolve_threshold: float = AUTO_RESOLVE_THRESHOLD,
        critical_blast_block: bool = True,
    ) -> None:
        self.model = model
        self.auto_resolve_threshold = auto_resolve_threshold
        self.critical_blast_block = critical_blast_block
        self._client: Optional[Any] = None

        if use_llm and _OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                self._client = _OpenAI(api_key=api_key)
                logger.info("LeadSRENode: LLM arbitration enabled (%s).", model)

    # =========================================================================
    # FILTER 1 — Shadow Consensus Check
    # =========================================================================

    def _shadow_consensus(
        self,
        net_intent: NetworkIntent,
        db_intent: DBIntent,
        state_vec: List[float],
        step: int,
    ) -> Tuple[str, Optional[str], str]:
        """
        Check intent alignment via the synergy matrix, then LLM, then fallback.

        Returns
        -------
        (signal, resolved_action_or_None, conflict_description_or_empty)
        """
        net_key = net_intent["intent"]
        db_key = db_intent["intent"]
        key = (net_key, db_key)

        # ── 1a. Deterministic synergy matrix lookup ───────────────────────
        if key in _SYNERGY_MATRIX:
            signal, action = _SYNERGY_MATRIX[key]
            if signal == _FAST:
                logger.info("Consensus GREEN: (%s,%s) → %s", net_key, db_key, action)
                return _FAST, action, ""
            conflict = (
                f"Intent conflict: network='{net_key}' vs db='{db_key}' "
                "would cause dual isolation or resource contention."
            )
            logger.info("Consensus RED: (%s,%s) → DEEP_NEGOTIATE", net_key, db_key)
            return _CONFLICT, None, conflict

        # ── 1b. Unknown pair — LLM arbitration ───────────────────────────
        if self._client is not None:
            try:
                signal, action, conflict = self._llm_arbitrate(
                    net_intent, db_intent, state_vec, step
                )
                return signal, action, conflict or ""
            except Exception as exc:
                logger.warning("LLM arbitration failed: %s — using tiebreaker.", exc)

        # ── 1c. Confidence tiebreaker fallback ────────────────────────────
        if net_intent["confidence"] >= db_intent["confidence"]:
            action = _NET_TO_ACTION.get(net_key, "noop")
        else:
            action = _DB_TO_ACTION.get(db_key, "noop")
        logger.info("Consensus tiebreak: action=%s", action)
        return _FAST, action, ""

    def _llm_arbitrate(
        self,
        net_intent: NetworkIntent,
        db_intent: DBIntent,
        state_vec: List[float],
        step: int,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Call GPT and parse consensus JSON. Returns (signal, action, conflict)."""
        payload = json.dumps({
            "current_state": {
                "Traffic_Load": state_vec[0],
                "Database_Temperature": state_vec[1],
                "Network_Health": state_vec[2],
            },
            "network_intent": net_intent,
            "db_intent": db_intent,
            "episode_step": step,
        })
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LEAD_SRE_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        status = str(parsed.get("consensus_status", "red")).lower()
        action = parsed.get("recommended_action")
        conflict = parsed.get("conflict_summary")
        if status == "green" and action and action != "ESCALATE":
            return _FAST, action, None
        return _CONFLICT, None, conflict

    # =========================================================================
    # FILTER 2 — Predictive Blast Radius
    # =========================================================================

    def _blast_radius_check(
        self,
        action: str,
        state_vec: List[float],
    ) -> Tuple[bool, BlastRiskLevel, List[SecondaryImpact], str]:
        """
        Evaluate secondary impacts of *action* in the current system state.

        Returns
        -------
        (passed, max_risk, triggered_impacts, warning_message_or_empty)
          passed=True  → action is cleared for Filter 3
          passed=False → action is BLOCKED; warning must be injected
        """
        max_risk, impacts = _evaluate_blast_radius(action, state_vec)

        if self.critical_blast_block and max_risk == BlastRiskLevel.CRITICAL:
            warning = _format_blast_warning(action, impacts, max_risk)
            logger.warning(
                "Blast Radius BLOCK: action='%s' max_risk=CRITICAL", action
            )
            return False, max_risk, impacts, warning

        # Non-critical risks pass (but are logged / shown in UI)
        warning = ""
        if max_risk in (BlastRiskLevel.HIGH, BlastRiskLevel.MEDIUM):
            impact_desc = "; ".join(i.description for i in impacts
                                    if i.risk in (BlastRiskLevel.HIGH, BlastRiskLevel.MEDIUM))
            warning = (
                f"[Blast Radius Advisory] action='{action}' risk={max_risk.value.upper()} "
                f"| {impact_desc}"
            )
        return True, max_risk, impacts, warning

    # =========================================================================
    # FILTER 3 — Adaptive Trust Layer (Escrow Execution)
    # =========================================================================

    def _trust_check(
        self,
        net_intent: NetworkIntent,
        db_intent: DBIntent,
    ) -> Tuple[TrustDecision, float, GovernanceSignal]:
        """
        Evaluate combined confidence and emit trust decision.

        Returns
        -------
        (trust_decision, combined_confidence, governance_signal)
        """
        combined = _compute_combined_confidence(net_intent, db_intent)
        if combined >= self.auto_resolve_threshold:
            logger.info(
                "Trust Layer: APPROVED (confidence=%.3f >= %.2f).",
                combined, self.auto_resolve_threshold,
            )
            return TrustDecision.APPROVED, combined, GovernanceSignal.AUTO_RESOLVE
        logger.info(
            "Trust Layer: ESCROWED (confidence=%.3f < %.2f) → HUMAN_ESCALATION.",
            combined, self.auto_resolve_threshold,
        )
        return TrustDecision.ESCROWED, combined, GovernanceSignal.HUMAN_ESCALATION

    # =========================================================================
    # LangGraph node entry-point
    # =========================================================================

    def run_as_node(self, state: SREGraphState) -> SREGraphState:
        """
        Execute all three governance filters and return the updated state.

        State keys written
        ------------------
        consensus_status    – GREEN | RED
        recommended_action  – resolved action string (or None if blocked)
        routing_path        – MIDDLE | SLOW
        governance_signal   – AUTO_RESOLVE | HUMAN_ESCALATION |
                              DEEP_NEGOTIATE | BLAST_RADIUS_BLOCK
        blast_radius_warnings – list of warning strings
        trust_decision      – APPROVED | ESCROWED
        human_approved      – preserved from incoming state (UI sets this)
        """
        # ── Extract inputs from shared state ─────────────────────────────
        net_intent: NetworkIntent = state.get("network_intent") or NetworkIntent(
            intent="noop", confidence=0.0, rationale="missing"
        )
        db_intent: DBIntent = state.get("db_intent") or DBIntent(
            intent="noop", confidence=0.0, rationale="missing"
        )
        vec: List[float] = state.get("current_state_tensor", [98.0, 95.0, 5.0])
        step: int = state.get("episode_step", 0)
        existing_warnings: List[str] = list(state.get("blast_radius_warnings") or [])

        # =================================================================
        # FILTER 1: Shadow Consensus
        # =================================================================
        consensus_signal, resolved_action, conflict_desc = self._shadow_consensus(
            net_intent, db_intent, vec, step
        )

        if consensus_signal == _CONFLICT:
            # Intents conflict — route to ChatOps; skip filters 2 & 3
            state = append_chat(
                state, role="lead_sre",
                content=(
                    f"[Filter 1 — Shadow Consensus] DEEP_NEGOTIATE | "
                    f"Conflict: {conflict_desc}"
                ),
            )
            return {  # type: ignore[return-value]
                **state,
                "consensus_status": ConsensusStatus.RED,
                "recommended_action": None,
                "routing_path": RoutingPath.SLOW,
                "governance_signal": GovernanceSignal.DEEP_NEGOTIATE,
                "blast_radius_warnings": existing_warnings,
                "trust_decision": TrustDecision.ESCROWED,
            }

        state = append_chat(
            state, role="lead_sre",
            content=(
                f"[Filter 1 — Shadow Consensus] GREEN | "
                f"Resolved action: '{resolved_action}'"
            ),
        )

        # =================================================================
        # FILTER 2: Predictive Blast Radius
        # =================================================================
        blast_passed, blast_risk, blast_impacts, blast_warning = (
            self._blast_radius_check(resolved_action or "noop", vec)
        )

        all_warnings = list(existing_warnings)
        if blast_warning:
            all_warnings.append(blast_warning)

        if not blast_passed:
            # Action is CRITICAL risk — block it; sub-agents will retry
            state = append_chat(
                state, role="lead_sre",
                content=(
                    f"[Filter 2 — Blast Radius] BLOCKED | "
                    f"action='{resolved_action}' risk=CRITICAL | "
                    f"{blast_warning}"
                ),
            )
            return {  # type: ignore[return-value]
                **state,
                "consensus_status": ConsensusStatus.GREEN,
                "recommended_action": None,
                "routing_path": RoutingPath.SLOW,
                "governance_signal": GovernanceSignal.BLAST_RADIUS_BLOCK,
                "blast_radius_warnings": all_warnings,
                "trust_decision": TrustDecision.ESCROWED,
            }

        state = append_chat(
            state, role="lead_sre",
            content=(
                f"[Filter 2 — Blast Radius] PASSED | "
                f"action='{resolved_action}' max_risk={blast_risk.value} | "
                f"{blast_warning or 'no advisory'}"
            ),
        )

        # =================================================================
        # FILTER 3: Adaptive Trust Layer
        # =================================================================
        trust_decision, combined_conf, gov_signal = self._trust_check(
            net_intent, db_intent
        )

        state = append_chat(
            state, role="lead_sre",
            content=(
                f"[Filter 3 — Trust Layer] {trust_decision.value.upper()} | "
                f"combined_confidence={combined_conf:.3f} | "
                f"threshold={self.auto_resolve_threshold} | "
                f"signal={gov_signal.value}"
            ),
        )

        return {  # type: ignore[return-value]
            **state,
            "consensus_status": ConsensusStatus.GREEN,
            "recommended_action": resolved_action,
            "routing_path": RoutingPath.MIDDLE,
            "governance_signal": gov_signal,
            "blast_radius_warnings": all_warnings,
            "trust_decision": trust_decision,
            # human_approved is not reset here — the UI manages it
        }


# ─────────────────────────── factory function ────────────────────────────────


def make_lead_sre_node(
    use_llm: bool = True,
    model: str = _DEFAULT_MODEL,
    auto_resolve_threshold: float = AUTO_RESOLVE_THRESHOLD,
    critical_blast_block: bool = True,
) -> Any:
    """
    Return a LangGraph-compatible callable bound to :class:`LeadSRENode`.

    Usage in ``sre_graph.py``::

        from controllers.lead_sre import make_lead_sre_node
        builder.add_node("shadow_consensus_node", make_lead_sre_node())
    """
    node = LeadSRENode(
        use_llm=use_llm,
        model=model,
        auto_resolve_threshold=auto_resolve_threshold,
        critical_blast_block=critical_blast_block,
    )
    return node.run_as_node


# ─────────────────────────── CLI smoke test ──────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from graph.message_bus import initial_state

    node = LeadSRENode(use_llm=False)

    print("\n=== Test 1: AUTO_RESOLVE (throttle + noop, high confidence) ===")
    s = initial_state([90.0, 40.0, 60.0])
    s["network_intent"] = NetworkIntent(intent="throttle", confidence=0.92, rationale="high traffic")
    s["db_intent"]      = DBIntent(intent="noop", confidence=0.91, rationale="db ok")
    result = node.run_as_node(s)
    print(f"  signal={result['governance_signal'].value}  action={result['recommended_action']}")
    print(f"  trust={result['trust_decision'].value}  blast_warnings={result['blast_radius_warnings']}")

    print("\n=== Test 2: HUMAN_ESCALATION (schema_failover, low confidence) ===")
    s2 = initial_state([30.0, 88.0, 70.0])
    s2["network_intent"] = NetworkIntent(intent="noop", confidence=0.55, rationale="net ok")
    s2["db_intent"]      = DBIntent(intent="failover", confidence=0.72, rationale="db hot")
    result2 = node.run_as_node(s2)
    print(f"  signal={result2['governance_signal'].value}  action={result2['recommended_action']}")
    print(f"  trust={result2['trust_decision'].value}")

    print("\n=== Test 3: BLAST_RADIUS_BLOCK (cache_flush with overloaded DB) ===")
    s3 = initial_state([50.0, 92.0, 65.0])
    s3["network_intent"] = NetworkIntent(intent="noop", confidence=0.91, rationale="net ok")
    s3["db_intent"]      = DBIntent(intent="cache_flush", confidence=0.93, rationale="db hot")
    result3 = node.run_as_node(s3)
    print(f"  signal={result3['governance_signal'].value}  action={result3['recommended_action']}")
    for w in result3['blast_radius_warnings']:
        print(f"  WARNING: {w[:80]}")

    print("\n=== Test 4: DEEP_NEGOTIATE (circuit_break + failover conflict) ===")
    s4 = initial_state([97.0, 90.0, 10.0])
    s4["network_intent"] = NetworkIntent(intent="circuit_break", confidence=0.95, rationale="critical")
    s4["db_intent"]      = DBIntent(intent="failover", confidence=0.93, rationale="critical")
    result4 = node.run_as_node(s4)
    print(f"  signal={result4['governance_signal'].value}  action={result4['recommended_action']}")
