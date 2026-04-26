"""
graph/sre_graph.py
==================
OpenCloud-SRE LangGraph state machine.

3-Tier Routing Architecture
----------------------------

  ┌─────────────────────────────────────────────────────────────┐
  │                      SRE Graph                              │
  │                                                             │
  │  START → [dna_memory_node]                                  │
  │                │                                            │
  │         ┌──────┴──────┐                                     │
  │     HIGH MATCH     LOW/MEDIUM                               │
  │    (FAST PATH)    (MIDDLE/SLOW)                             │
  │         │               │                                   │
  │  [executor_node]  [network_ctrl_node]                       │
  │         │               │                                   │
  │       END         [db_ctrl_node]                            │
  │                         │                                   │
  │                  [shadow_consensus_node]                     │
  │                         │                                   │
  │                  ┌──────┴──────┐                            │
  │               GREEN           RED                           │
  │           (MIDDLE PATH)   (SLOW PATH)                       │
  │                │               │                            │
  │         [executor_node]  [chatops_node]                     │
  │                │               │                            │
  │              END        [executor_node]                     │
  │                                │                            │
  │                              END                            │
  └─────────────────────────────────────────────────────────────┘

Each node is a pure function: ``(SREGraphState) → SREGraphState``.

LLM calls are made only in the MIDDLE and SLOW paths.  The FAST path is
purely deterministic (FAISS lookup + direct action execution).

The graph is designed to be runnable locally without any API key by
providing a ``mock_llm=True`` flag — the controllers then return
hard-coded intents derived from the current state tensor.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Literal, Optional

from langgraph.graph import StateGraph, END

from graph.message_bus import (
    SREGraphState,
    RoutingPath,
    ConsensusStatus,
    NetworkIntent,
    DBIntent,
    ComputeIntent,
    initial_state,
    append_chat,
)
from utils.dna_memory import DNAMemory, MatchConfidence
from env.environment import OpenCloudEnv, VALID_ACTIONS
from env.state_tensor import CloudStateTensor
from controllers.system_prompts import (
    NETWORK_CONTROLLER_SYSTEM_PROMPT,
    DATABASE_CONTROLLER_SYSTEM_PROMPT,
    LEAD_SRE_SYSTEM_PROMPT,
    CHATOPS_SYSTEM_PROMPT,
    COMPUTE_AGENT_SYSTEM_PROMPT,
    NETWORK_AGENT_SYSTEM_PROMPT,
    DATABASE_AGENT_SYSTEM_PROMPT,
    SHADOW_ARBITER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ───────────────────────────── optional LLM setup ─────────────────────────────

# ───────────────────────── optional HF LLM setup ─────────────────────────────
# BUG-FIX: Removed dead OpenAI import — project is 100% vendor-agnostic.
# The HF InferenceClient is used instead for live LLM calls.
try:
    from huggingface_hub import InferenceClient as _HFClient
    _HF_AVAILABLE = True
except ImportError:
    _HFClient = None  # type: ignore[assignment]
    _HF_AVAILABLE = False


def _get_hf_client() -> Optional[Any]:
    """Return a configured HF InferenceClient or None if unavailable/no token."""
    if not _HF_AVAILABLE or not _HFClient:
        return None
    token = os.getenv("HF_TOKEN", "")
    return _HFClient(token=token) if token else None


def _call_llm(
    client: Any,
    system_prompt: str,
    user_message: str,
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.1,
) -> str:
    """Make a single HF InferenceClient chat completion and return raw content."""
    resp = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=256,
        temperature=temperature,
    )
    return resp.choices[0].message.content or "{}"


# ──────────────────────────── mock intent helpers ─────────────────────────────

def _mock_network_intent(state_vec: list[float]) -> NetworkIntent:
    """
    Deterministic rule-based mock for the Network Controller.
    Used when no HF token is present (local dev / CI).
    """
    traffic, _, net_health = state_vec
    if traffic > 85:
        return NetworkIntent(thought_process="Mock logic: Critical traffic", observed_anomalies=["High Traffic"], verified_root_cause="Traffic Spike", action="circuit_breaker", risk_score=0.90)
    if traffic > 70:
        return NetworkIntent(thought_process="Mock logic: Elevated traffic", observed_anomalies=["Elevated Traffic"], verified_root_cause="Load Imbalance", action="throttle_traffic", risk_score=0.75)
    if net_health < 30:
        return NetworkIntent(thought_process="Mock logic: Low net health", observed_anomalies=["Low Network Health"], verified_root_cause="Capacity Limit", action="scale_out", risk_score=0.80)
    if traffic > 50:
        return NetworkIntent(thought_process="Mock logic: Moderate traffic", observed_anomalies=["Moderate Traffic"], verified_root_cause="Uneven Load", action="load_balance", risk_score=0.60)
    return NetworkIntent(thought_process="Mock logic: Nominal", observed_anomalies=[], verified_root_cause="Normal Operations", action="noop", risk_score=0.30)


def _mock_db_intent(state_vec: list[float]) -> DBIntent:
    """
    Deterministic rule-based mock for the Database Controller.
    """
    _, db_temp, _ = state_vec
    if db_temp > 85:
        return DBIntent(thought_process="Mock logic: DB temp critical", observed_anomalies=["High DB Temp"], verified_root_cause="DB Overload", action="schema_failover", risk_score=0.92)
    if db_temp > 70:
        return DBIntent(thought_process="Mock logic: Elevated DB temp", observed_anomalies=["Elevated DB Temp"], verified_root_cause="Cache Misses", action="cache_flush", risk_score=0.70)
    if db_temp > 55:
        return DBIntent(thought_process="Mock logic: Moderate DB load", observed_anomalies=["Moderate DB Load"], verified_root_cause="Minor Contention", action="cache_flush", risk_score=0.55)
    return DBIntent(thought_process="Mock logic: DB nominal", observed_anomalies=[], verified_root_cause="Normal Operations", action="noop", risk_score=0.25)


# ── Shadow Consensus mock helpers (rule-based, no LLM needed) ────────────────

def _mock_compute_agent_intent(state_vec: list[float]) -> ComputeIntent:
    """Deterministic Compute Agent: fires on CPU/Traffic spikes."""
    cpu = state_vec[0]  # Traffic_Load == CPU in state tensor
    if cpu > 85:
        return ComputeIntent(agent_role="Compute",
            diagnosis=f"CPU={cpu:.1f} critical — load-shedding required.",
            confidence_score=0.95, proposed_action="throttle_traffic")
    if cpu > 70:
        return ComputeIntent(agent_role="Compute",
            diagnosis=f"CPU={cpu:.1f} elevated — scale out recommended.",
            confidence_score=0.75, proposed_action="scale_out")
    return ComputeIntent(agent_role="Compute",
        diagnosis=f"CPU={cpu:.1f} nominal — deferring to other agents.",
        confidence_score=0.10, proposed_action="noop")


def _mock_network_agent_intent(state_vec: list[float]) -> ComputeIntent:
    """Deterministic Network Agent: fires on Latency/Network_Health failures."""
    net_health = state_vec[2]
    if net_health > 50:   # server Latency axis: high value = bad
        return ComputeIntent(agent_role="Network",
            diagnosis=f"Latency/Network={net_health:.1f} — severe partition detected.",
            confidence_score=0.99, proposed_action="schema_failover")
    if net_health > 30:
        return ComputeIntent(agent_role="Network",
            diagnosis=f"Network_Health={net_health:.1f} degraded — circuit break.",
            confidence_score=0.65, proposed_action="circuit_breaker")
    return ComputeIntent(agent_role="Network",
        diagnosis=f"Network_Health={net_health:.1f} nominal — deferring.",
        confidence_score=0.10, proposed_action="noop")


def _mock_database_agent_intent(state_vec: list[float]) -> ComputeIntent:
    """Deterministic Database Agent: fires on DB_Temp spikes."""
    db_temp = state_vec[1]
    if db_temp > 85:
        return ComputeIntent(agent_role="Database",
            diagnosis=f"DB_Temp={db_temp:.1f} — deadlock state assumed.",
            confidence_score=0.92, proposed_action="kill_long_queries")
    if db_temp > 65:
        return ComputeIntent(agent_role="Database",
            diagnosis=f"DB_Temp={db_temp:.1f} elevated — pod restart.",
            confidence_score=0.60, proposed_action="restart_pods")
    return ComputeIntent(agent_role="Database",
        diagnosis=f"DB_Temp={db_temp:.1f} nominal — deferring.",
        confidence_score=0.20, proposed_action="noop")


# Action mapping for Shadow Agent proposed_actions → env VALID_ACTIONS
_SHADOW_ACTION_MAP: Dict[str, str] = {
    "throttle_traffic": "throttle_traffic",
    "scale_out":        "scale_out",
    "schema_failover":  "schema_failover",
    "circuit_breaker":  "circuit_breaker",
    "kill_long_queries": "cache_flush",   # surgical DB proxy
    "restart_pods":     "restart_pods",
    "noop":             "noop",
}


_NETWORK_TO_ACTION: Dict[str, str] = {
    "throttle": "throttle_traffic",
    "circuit_break": "circuit_breaker",
    "scale_out": "scale_out",
    "load_balance": "load_balance",
    "noop": "noop",
}

_DB_TO_ACTION: Dict[str, str] = {
    "failover": "schema_failover",
    "cache_flush": "cache_flush",
    "restart": "restart_pods",
    "noop": "noop",
}

# Synergy table: (net_action, db_action) → resolved_env_action or ESCALATE
# Keys must match the 'action' field values emitted by NetworkIntent/DBIntent.
_SYNERGY_TABLE: Dict[tuple[str, str], str] = {
    ("throttle_traffic",  "noop"):            "throttle_traffic",
    ("circuit_breaker",   "noop"):            "circuit_breaker",
    ("load_balance",      "noop"):            "load_balance",
    ("scale_out",         "noop"):            "scale_out",
    ("noop",              "schema_failover"): "schema_failover",
    ("noop",              "cache_flush"):     "cache_flush",
    ("noop",              "restart_pods"):    "restart_pods",
    ("noop",              "noop"):            "noop",
    ("throttle_traffic",  "schema_failover"): "schema_failover",
    ("scale_out",         "cache_flush"):     "scale_out",
    ("load_balance",      "cache_flush"):     "cache_flush",
    ("circuit_breaker",   "schema_failover"): "ESCALATE",
    ("circuit_breaker",   "restart_pods"):    "ESCALATE",
    ("load_balance",      "schema_failover"): "ESCALATE",
}


# ──────────────────────────────── graph nodes ─────────────────────────────────


def dna_memory_node(state: SREGraphState, memory: DNAMemory) -> SREGraphState:
    """
    FAST PATH – Query the DNA Memory FAISS index.

    Sets routing_path to FAST if a High Match is found, otherwise MIDDLE.
    """
    vec = state["current_state_tensor"]
    hit = memory.query(vec)

    state = append_chat(
        state,
        role="dna_memory",
        content=(
            f"[DNA Memory] Confidence={hit.confidence.value} | "
            f"Distance={hit.distance:.2f} | Nearest={hit.matched_vector} | "
            f"Action={hit.matched_action}"
        ),
    )

    if hit.is_fast_path():
        logger.info("FAST PATH activated – cache hit: %s", hit.matched_action)
        return {
            **state,  # type: ignore[return-value]
            "routing_path": RoutingPath.FAST,
            "recommended_action": hit.matched_action,
            "dna_memory_hit": hit.to_dict(),
            "consensus_status": ConsensusStatus.GREEN,
        }
    else:
        return {
            **state,  # type: ignore[return-value]
            "routing_path": RoutingPath.MIDDLE,
            "dna_memory_hit": hit.to_dict(),
        }


def network_controller_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """MIDDLE PATH – Network Controller emits a micro-intent JSON."""
    vec = state["current_state_tensor"]

    if mock_llm or client is None:
        intent = _mock_network_intent(vec)
    else:
        user_msg = json.dumps({
            "current_state": {
                "Traffic_Load": vec[0],
                "Database_Temperature": vec[1],
                "Network_Health": vec[2],
            },
            "episode_step": state.get("episode_step", 0),
            "chat_history": state.get("chat_history", [])[-5:],
        })
        raw = _call_llm(client, NETWORK_CONTROLLER_SYSTEM_PROMPT, user_msg)
        parsed = json.loads(raw)
        intent = NetworkIntent(
            thought_process=parsed.get("thought_process", ""),
            observed_anomalies=parsed.get("observed_anomalies", []),
            verified_root_cause=parsed.get("verified_root_cause", ""),
            action=parsed.get("action", "noop"),
            risk_score=float(parsed.get("risk_score", 0.5)),
        )

    state = append_chat(
        state,
        role="network_ctrl",
        content=f"[Network Intent] {json.dumps(intent)}",
    )

    return {**state, "network_intent": intent}  # type: ignore[return-value]


def db_controller_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """MIDDLE PATH – Database Controller emits a micro-intent JSON."""
    vec = state["current_state_tensor"]

    if mock_llm or client is None:
        intent = _mock_db_intent(vec)
    else:
        user_msg = json.dumps({
            "current_state": {
                "Traffic_Load": vec[0],
                "Database_Temperature": vec[1],
                "Network_Health": vec[2],
            },
            "episode_step": state.get("episode_step", 0),
            "chat_history": state.get("chat_history", [])[-5:],
        })
        raw = _call_llm(client, DATABASE_CONTROLLER_SYSTEM_PROMPT, user_msg)
        parsed = json.loads(raw)
        intent = DBIntent(
            thought_process=parsed.get("thought_process", ""),
            observed_anomalies=parsed.get("observed_anomalies", []),
            verified_root_cause=parsed.get("verified_root_cause", ""),
            action=parsed.get("action", "noop"),
            risk_score=float(parsed.get("risk_score", 0.5)),
        )

    state = append_chat(
        state,
        role="db_ctrl",
        content=f"[DB Intent] {json.dumps(intent)}",
    )

    return {**state, "db_intent": intent}  # type: ignore[return-value]


def shadow_consensus_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """
    MIDDLE PATH – Lead SRE Shadow Consensus arbitration.

    Checks the synergy table first (deterministic, fast).
    Falls back to LLM only for cases not in the table.
    """
    net_intent: NetworkIntent = state.get("network_intent") or NetworkIntent(
        thought_process="missing", observed_anomalies=[], verified_root_cause="", action="noop", risk_score=0.0
    )
    db_intent: DBIntent = state.get("db_intent") or DBIntent(
        thought_process="missing", observed_anomalies=[], verified_root_cause="", action="noop", risk_score=0.0
    )

    net_key = net_intent.get("action", "noop")
    db_key = db_intent.get("action", "noop")
    lookup_key = (net_key, db_key)

    action = _SYNERGY_TABLE.get(lookup_key)

    if action is not None:
        # Deterministic path – no LLM needed
        if action == "ESCALATE":
            consensus = ConsensusStatus.RED
            conflict_summary = (
                f"Conflict: '{net_key}' vs '{db_key}' would cause dual isolation."
            )
            recommended = None
            routing = RoutingPath.SLOW
        else:
            consensus = ConsensusStatus.GREEN
            conflict_summary = None
            recommended = action
            routing = RoutingPath.MIDDLE
    else:
        # Unknown combination – use LLM arbitration or fallback mock
        if mock_llm or client is None:
            # Default: higher risk score controller wins
            if net_intent.get("risk_score", 0.0) >= db_intent.get("risk_score", 0.0):
                recommended = net_key
            else:
                recommended = db_key
            consensus = ConsensusStatus.GREEN
            conflict_summary = None
            routing = RoutingPath.MIDDLE
        else:
            vec = state["current_state_tensor"]
            user_msg = json.dumps({
                "current_state": {
                    "Traffic_Load": vec[0],
                    "Database_Temperature": vec[1],
                    "Network_Health": vec[2],
                },
                "network_intent": net_intent,
                "db_intent": db_intent,
                "episode_step": state.get("episode_step", 0),
            })
            raw = _call_llm(client, LEAD_SRE_SYSTEM_PROMPT, user_msg)
            parsed = json.loads(raw)
            status_str = parsed.get("consensus_status", "red").lower()
            if status_str == "green":
                consensus = ConsensusStatus.GREEN
                routing = RoutingPath.MIDDLE
            elif status_str == "retry":
                consensus = ConsensusStatus.RETRY
                routing = RoutingPath.MIDDLE
            else:
                consensus = ConsensusStatus.RED
                routing = RoutingPath.SLOW
            recommended = parsed.get("recommended_action")
            conflict_summary = parsed.get("conflict_summary")

    if consensus == ConsensusStatus.RETRY:
        state = append_chat(
            state,
            role="lead_sre",
            content="[PENALTY] RETRY TRIGGERED. High risk action proposed without a prior diagnostic step (e.g. ping, top). You have been penalized. Rethink and verify your hypothesis.",
        )
    else:
        state = append_chat(
            state,
            role="lead_sre",
            content=(
                f"[Shadow Consensus] {consensus.value.upper()} | "
                f"Action={recommended} | Conflict={conflict_summary}"
            ),
        )

    return {  # type: ignore[return-value]
        **state,
        "consensus_status": consensus,
        "recommended_action": recommended,
        "routing_path": routing,
    }


# ── Shadow Consensus agent nodes ─────────────────────────────────────────────

def compute_agent_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """Shadow Worker 1 — Compute domain specialist."""
    vec = state.get("current_state_tensor", [50.0, 50.0, 5.0])

    if mock_llm or client is None:
        intent = _mock_compute_agent_intent(vec)
    else:
        import json as _json
        cpu = vec[0]
        metrics = {"CPU": cpu, "Traffic_Load": vec[0], "DB_Temp": vec[1], "Latency": vec[2]}
        try:
            raw = _call_llm(client, COMPUTE_AGENT_SYSTEM_PROMPT, _json.dumps(metrics))
            p = _json.loads(raw)
            intent = ComputeIntent(agent_role=str(p.get("agent_role", "Compute")),
                diagnosis=str(p.get("diagnosis", "")),
                confidence_score=float(p.get("confidence_score", 0.1)),
                proposed_action=str(p.get("proposed_action", "noop")))
        except Exception as exc:
            logger.warning("compute_agent_node LLM failed: %s", exc)
            intent = _mock_compute_agent_intent(vec)

    state = append_chat(state, role="compute_agent",
        content=f"[Compute Agent] conf={intent['confidence_score']:.2f} | "
                f"action={intent['proposed_action']} | {intent['diagnosis']}")
    return {**state, "compute_intent": intent}  # type: ignore[return-value]


def network_agent_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """Shadow Worker 2 — Network domain specialist."""
    vec = state.get("current_state_tensor", [50.0, 50.0, 5.0])

    if mock_llm or client is None:
        intent = _mock_network_agent_intent(vec)
    else:
        import json as _json
        metrics = {"CPU": vec[0], "Traffic_Load": vec[0], "DB_Temp": vec[1], "Latency": vec[2]}
        try:
            raw = _call_llm(client, NETWORK_AGENT_SYSTEM_PROMPT, _json.dumps(metrics))
            p = _json.loads(raw)
            intent = ComputeIntent(agent_role=str(p.get("agent_role", "Network")),
                diagnosis=str(p.get("diagnosis", "")),
                confidence_score=float(p.get("confidence_score", 0.1)),
                proposed_action=str(p.get("proposed_action", "noop")))
        except Exception as exc:
            logger.warning("network_agent_node LLM failed: %s", exc)
            intent = _mock_network_agent_intent(vec)

    state = append_chat(state, role="network_agent",
        content=f"[Network Agent] conf={intent['confidence_score']:.2f} | "
                f"action={intent['proposed_action']} | {intent['diagnosis']}")
    return {**state, "network_agent_intent": intent}  # type: ignore[return-value]


def database_agent_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """Shadow Worker 3 — Database domain specialist."""
    vec = state.get("current_state_tensor", [50.0, 50.0, 5.0])

    if mock_llm or client is None:
        intent = _mock_database_agent_intent(vec)
    else:
        import json as _json
        metrics = {"CPU": vec[0], "Traffic_Load": vec[0], "DB_Temp": vec[1], "Latency": vec[2]}
        try:
            raw = _call_llm(client, DATABASE_AGENT_SYSTEM_PROMPT, _json.dumps(metrics))
            p = _json.loads(raw)
            intent = ComputeIntent(agent_role=str(p.get("agent_role", "Database")),
                diagnosis=str(p.get("diagnosis", "")),
                confidence_score=float(p.get("confidence_score", 0.1)),
                proposed_action=str(p.get("proposed_action", "noop")))
        except Exception as exc:
            logger.warning("database_agent_node LLM failed: %s", exc)
            intent = _mock_database_agent_intent(vec)

    state = append_chat(state, role="database_agent",
        content=f"[Database Agent] conf={intent['confidence_score']:.2f} | "
                f"action={intent['proposed_action']} | {intent['diagnosis']}")
    return {**state, "database_agent_intent": intent}  # type: ignore[return-value]


def shadow_debate_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """
    Shadow Arbiter — picks the winner of the 3-agent debate by confidence_score.

    Runs deterministically (no LLM needed) in mock mode.  In live mode, the
    HF Arbiter LLM validates and narrates the decision.
    """
    compute  = state.get("compute_intent")  or ComputeIntent(agent_role="Compute",  diagnosis="", confidence_score=0.0, proposed_action="noop")
    network  = state.get("network_agent_intent") or ComputeIntent(agent_role="Network",  diagnosis="", confidence_score=0.0, proposed_action="noop")
    database = state.get("database_agent_intent") or ComputeIntent(agent_role="Database", diagnosis="", confidence_score=0.0, proposed_action="noop")

    agents = [
        (network["confidence_score"],  "Network",  network),
        (database["confidence_score"], "Database", database),
        (compute["confidence_score"],  "Compute",  compute),
    ]
    agents.sort(key=lambda x: x[0], reverse=True)

    winning_score, winning_role, winner = agents[0]
    raw_action = winner["proposed_action"]
    resolved   = _SHADOW_ACTION_MAP.get(raw_action, "noop")
    if resolved not in VALID_ACTIONS:
        resolved = "noop"

    rationale = (
        f"{winning_role} Agent won with confidence={winning_score:.2f} — "
        f"proposed '{raw_action}' → executing '{resolved}'."
    )

    state = append_chat(state, role="shadow_arbiter",
        content=f"[Shadow Arbiter] WINNER={winning_role} | conf={winning_score:.2f} | "
                f"resolved_action={resolved} | {rationale}")

    return {  # type: ignore[return-value]
        **state,
        "recommended_action": resolved,
        "consensus_status":   ConsensusStatus.GREEN,
        "routing_path":       RoutingPath.MIDDLE,
        "winning_agent":      winning_role,
        "winning_confidence": winning_score,
        "shadow_rationale":   rationale,
    }


def chatops_node(
    state: SREGraphState,
    client: Optional[Any],
    mock_llm: bool,
) -> SREGraphState:
    """
    SLOW PATH – ChatOps deep-reasoning resolver.

    Activated only when shadow_consensus detects a RED conflict.
    Produces a resolved action via verbose LLM reasoning.
    """
    net_intent = state.get("network_intent")
    db_intent = state.get("db_intent")
    vec = state["current_state_tensor"]

    if mock_llm or client is None:
        # Mock: pick the higher-confidence intent's corresponding action
        # NetworkIntent / DBIntent use 'risk_score' (lower = more confident)
        if net_intent and db_intent:
            net_risk = float(net_intent.get("risk_score", 0.5))  # type: ignore[typeddict-item]
            db_risk  = float(db_intent.get("risk_score", 0.5))   # type: ignore[typeddict-item]
            if net_risk <= db_risk:   # lower risk_score = more confident
                resolved = net_intent.get("action", "noop")  # type: ignore[typeddict-item]
                if resolved not in VALID_ACTIONS:
                    resolved = "noop"
            else:
                resolved = db_intent.get("action", "noop")   # type: ignore[typeddict-item]
                if resolved not in VALID_ACTIONS:
                    resolved = "noop"
        else:
            resolved = "noop"
        risk = "medium"
        rationale = "Mock ChatOps: resolved by confidence tiebreaker."
    else:
        user_msg = json.dumps({
            "current_state": {
                "Traffic_Load": vec[0],
                "Database_Temperature": vec[1],
                "Network_Health": vec[2],
            },
            "network_intent": net_intent,
            "db_intent": db_intent,
            "conflict_summary": "Intent conflict detected by Shadow Consensus.",
            "chat_history": state.get("chat_history", [])[-8:],
            "episode_step": state.get("episode_step", 0),
        })
        raw = _call_llm(
            client,
            CHATOPS_SYSTEM_PROMPT,
            user_msg,
            temperature=0.3,   # slightly more creative for conflict resolution
        )
        parsed = json.loads(raw)
        resolved = parsed.get("resolved_action", "noop")
        risk = parsed.get("risk_level", "medium")
        rationale = parsed.get("resolution_rationale", "")

    if resolved not in VALID_ACTIONS:
        resolved = "noop"

    state = append_chat(
        state,
        role="chatops",
        content=(
            f"[ChatOps Resolution] Action={resolved} | Risk={risk} | "
            f"Rationale={rationale}"
        ),
    )

    return {  # type: ignore[return-value]
        **state,
        "recommended_action": resolved,
        "routing_path": RoutingPath.SLOW,
    }


def executor_node(state: SREGraphState, env: OpenCloudEnv) -> SREGraphState:
    """
    Terminal action node – applies the recommended_action to the environment.

    Updates current_state_tensor and slo_score.  Sets is_resolved if SLO ≥ 0.95.
    """
    action = state.get("recommended_action") or "noop"
    if action not in VALID_ACTIONS:
        logger.warning("Invalid action '%s', falling back to noop.", action)
        action = "noop"

    obs, reward, terminated, truncated, info = env.step(action)

    new_vec = [
        obs["Traffic_Load"],
        obs["Database_Temperature"],
        obs["Network_Health"],
    ]

    # BUG-FIX: use .get() with fallback — env.step() info dict may not always
    # include slo_score / is_critical if called on a terminated environment.
    slo = info.get("slo_score")
    if slo is None:
        tl, db, nh = new_vec
        slo = ((100 - tl) + (100 - db) + nh) / 300.0
    slo = float(slo)
    is_critical = bool(info.get("is_critical", False))
    is_resolved = slo >= 0.95
    step = state.get("episode_step", 0) + 1

    state = append_chat(
        state,
        role="executor",
        content=(
            f"[Executor] action={action} | reward={reward:.2f} | "
            f"SLO={slo:.3f} | critical={is_critical} | "
            f"state={[round(v, 1) for v in new_vec]}"
        ),
    )

    return {  # type: ignore[return-value]
        **state,
        "previous_state_tensor": state["current_state_tensor"],
        "current_state_tensor": new_vec,
        "slo_score": slo,
        "is_resolved": is_resolved,
        "episode_step": step,
    }


# ────────────────────────── routing conditions ────────────────────────────────

def _route_after_dna(state: SREGraphState) -> Literal["executor_node", "network_controller_node"]:
    """Branch after dna_memory_node: fast-path straight to executor, or middle path."""
    if state.get("routing_path") == RoutingPath.FAST:
        return "executor_node"
    return "network_controller_node"


def _route_after_consensus(
    state: SREGraphState,
) -> Literal["executor_node", "chatops_node", "network_controller_node"]:
    """Branch after shadow_consensus_node: GREEN → executor, RED → ChatOps, RETRY → network_controller_node."""
    status = state.get("consensus_status")
    if status == ConsensusStatus.GREEN:
        return "executor_node"
    elif status == ConsensusStatus.RETRY:
        return "network_controller_node"
    return "chatops_node"


# ───────────────────────── graph factory function ─────────────────────────────


def build_sre_graph(
    env: Optional[OpenCloudEnv] = None,
    memory: Optional[DNAMemory] = None,
    mock_llm: bool = True,
) -> StateGraph:
    """
    Compile and return the OpenCloud-SRE LangGraph state machine.

    Parameters
    ----------
    env:
        An initialised :class:`~env.environment.OpenCloudEnv`.
        Created with default settings if not provided.
    memory:
        A seeded :class:`~utils.dna_memory.DNAMemory` index.
        Created with default seed incidents if not provided.
    mock_llm:
        If True, all LLM nodes use deterministic rule-based mocks.
        Set to False and ensure HF_TOKEN is set for live LLM calls via HF InferenceClient.

    Returns
    -------
    StateGraph
        A compiled LangGraph graph.  Call ``.invoke(initial_state(...))``
        to run a single step, or ``.stream(...)`` for step-by-step output.
    """
    if env is None:
        env = OpenCloudEnv(seed=42, crash_on_reset=True)
    if memory is None:
        memory = DNAMemory()

    client = None if mock_llm else _get_hf_client()

    # ── bind env/memory/client into closures so nodes are (state→state) ──────
    def _dna(s: SREGraphState) -> SREGraphState:
        return dna_memory_node(s, memory)

    def _net(s: SREGraphState) -> SREGraphState:
        return network_controller_node(s, client, mock_llm)

    def _db(s: SREGraphState) -> SREGraphState:
        return db_controller_node(s, client, mock_llm)

    def _consensus(s: SREGraphState) -> SREGraphState:
        return shadow_consensus_node(s, client, mock_llm)

    def _chatops(s: SREGraphState) -> SREGraphState:
        return chatops_node(s, client, mock_llm)

    def _exec(s: SREGraphState) -> SREGraphState:
        return executor_node(s, env)

    # ── Shadow Consensus GRPO workers ─────────────────────────────────────
    def _compute_agent(s: SREGraphState) -> SREGraphState:
        return compute_agent_node(s, client, mock_llm)

    def _network_agent(s: SREGraphState) -> SREGraphState:
        return network_agent_node(s, client, mock_llm)

    def _database_agent(s: SREGraphState) -> SREGraphState:
        return database_agent_node(s, client, mock_llm)

    def _shadow_debate(s: SREGraphState) -> SREGraphState:
        return shadow_debate_node(s, client, mock_llm)

    # ── assemble the graph ─────────────────────────────────────────────────
    builder = StateGraph(SREGraphState)

    builder.add_node("dna_memory_node",         _dna)
    builder.add_node("network_controller_node",  _net)
    builder.add_node("db_controller_node",       _db)
    builder.add_node("shadow_consensus_node",    _consensus)
    builder.add_node("chatops_node",             _chatops)
    builder.add_node("executor_node",            _exec)
    # Shadow Consensus GRPO workers + arbiter
    builder.add_node("compute_agent_node",       _compute_agent)
    builder.add_node("network_agent_node",       _network_agent)
    builder.add_node("database_agent_node",      _database_agent)
    builder.add_node("shadow_debate_node",       _shadow_debate)

    builder.set_entry_point("dna_memory_node")

    builder.add_conditional_edges(
        "dna_memory_node",
        _route_after_dna,
        {
            "executor_node":          "executor_node",
            "network_controller_node": "network_controller_node",
        },
    )

    # Legacy MIDDLE path: network → db → shadow_consensus
    builder.add_edge("network_controller_node", "db_controller_node")
    # After legacy db_ctrl, run the three Shadow workers in series, then arbiter
    builder.add_edge("db_controller_node",      "compute_agent_node")
    builder.add_edge("compute_agent_node",       "network_agent_node")
    builder.add_edge("network_agent_node",       "database_agent_node")
    builder.add_edge("database_agent_node",      "shadow_debate_node")
    # Arbiter result goes straight to executor (GREEN by definition)
    builder.add_edge("shadow_debate_node",       "executor_node")

    # Keep the legacy consensus node reachable for the RETRY loop
    builder.add_conditional_edges(
        "shadow_consensus_node",
        _route_after_consensus,
        {
            "executor_node":          "executor_node",
            "chatops_node":           "chatops_node",
            "network_controller_node": "network_controller_node",
        },
    )

    builder.add_edge("chatops_node",   "executor_node")
    builder.add_edge("executor_node",  END)

    return builder.compile()


# ───────────────────────────────── CLI smoke test ─────────────────────────────

if __name__ == "__main__":
    import pprint

    logging.basicConfig(level=logging.INFO)

    env = OpenCloudEnv(seed=0, crash_on_reset=True)
    env.reset()

    graph = build_sre_graph(env=env, mock_llm=True)
    state = initial_state()

    print("=== OpenCloud-SRE 3-Tier Routing Smoke Test ===")
    for step in range(5):
        result = graph.invoke(state)
        state = result
        print(f"\n--- Step {step + 1} ---")
        pprint.pprint({
            "routing_path": state.get("routing_path"),
            "consensus_status": state.get("consensus_status"),
            "recommended_action": state.get("recommended_action"),
            "slo_score": round(state.get("slo_score", 0), 3),
            "is_resolved": state.get("is_resolved"),
            "state_vector": [round(v, 1) for v in state.get("current_state_tensor", [])],
        })
        if state.get("is_resolved"):
            print("✅ System recovered!")
            break
