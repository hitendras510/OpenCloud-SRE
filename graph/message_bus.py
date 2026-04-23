"""
graph/message_bus.py
=====================
LangGraph state schema for the OpenCloud-SRE multi-agent pipeline.

All nodes in the graph read from and write to a single shared state dict
of type :class:`SREGraphState`.  Using ``TypedDict`` gives full type-checker
support while remaining compatible with LangGraph's serialisation layer.

State lifecycle
---------------
  reset()  →  initial SREGraphState
  ↓  dna_memory_node  (fast-path check)
  ↓  network_controller_node
  ↓  db_controller_node
  ↓  shadow_consensus_node  (Lead SRE synergy check)
  ↓  [chatops_node if conflict]
  ↓  executor_node  (applies action to OpenCloudEnv)
  →  terminal / loop back
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

# ─────────────────────────────── enumerations ─────────────────────────────────


class RoutingPath(str, Enum):
    """Which of the three routing tiers is currently active."""
    FAST = "fast_path"       # DNA memory cache hit  → instant remediation
    MIDDLE = "middle_path"   # Shadow consensus      → synergy check
    SLOW = "slow_path"       # Intent conflict       → ChatOps deep-reason


class ConsensusStatus(str, Enum):
    """Traffic-light state of the Shadow Consensus layer."""
    GREEN = "green"    # Intents agree / synergy confirmed – safe to execute
    RED = "red"        # Intents conflict – escalate to ChatOps


class GovernanceSignal(str, Enum):
    """
    Unified routing signal emitted by LeadSRENode after all three
    governance filters have been applied.

      AUTO_RESOLVE       – All filters passed AND confidence ≥ 0.90.
                           Executor node runs immediately.
      HUMAN_ESCALATION   – Blast Radius passed BUT confidence < 0.90.
                           Execution is paused (escrow) until the UI
                           'Approve' button is clicked.
      DEEP_NEGOTIATE     – Shadow Consensus detected an intent conflict.
                           Routed to ChatOps for resolution.
      BLAST_RADIUS_BLOCK – Proposed action would cause a critical secondary
                           impact.  Intent rejected; agents must retry.
    """
    AUTO_RESOLVE = "AUTO_RESOLVE"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"
    DEEP_NEGOTIATE = "DEEP_NEGOTIATE"
    BLAST_RADIUS_BLOCK = "BLAST_RADIUS_BLOCK"


class BlastRiskLevel(str, Enum):
    """Severity of secondary impacts predicted by the Blast Radius filter."""
    NONE = "none"          # No secondary impacts
    LOW = "low"            # Minor degradation expected
    MEDIUM = "medium"      # Notable impact; monitor closely
    HIGH = "high"          # Significant risk; human review recommended
    CRITICAL = "critical"  # Execution blocked by governance filter


class TrustDecision(str, Enum):
    """
    Output of the Adaptive Trust Layer (Escrow Execution filter).

      APPROVED  – Confidence ≥ 0.90; action approved for auto-execution.
      ESCROWED  – Confidence < 0.90; action held pending human approval.
    """
    APPROVED = "approved"
    ESCROWED = "escrowed"


# ─────────────────────────────── message types ────────────────────────────────


class ChatMessage(TypedDict):
    """A single message in the ChatOps history."""
    role: str        # "network_ctrl" | "db_ctrl" | "lead_sre" | "chatops"
    content: str
    timestamp: str   # ISO-8601 string


class NetworkIntent(TypedDict):
    """Micro-intent JSON emitted by the Network Controller node."""
    intent: str           # e.g. "throttle" | "circuit_break" | "scale_out"
    confidence: float     # [0, 1]
    rationale: str        # one-sentence justification


class DBIntent(TypedDict):
    """Micro-intent JSON emitted by the Database Controller node."""
    intent: str           # e.g. "failover" | "cache_flush" | "restart"
    confidence: float
    rationale: str


# ───────────────────────────── primary state dict ─────────────────────────────


class SREGraphState(TypedDict, total=False):
    """
    Canonical shared state for the OpenCloud-SRE LangGraph pipeline.

    All keys are optional (``total=False``) so nodes can do partial updates
    using LangGraph's reducer pattern without clobbering unrelated fields.

    Fields
    ------
    current_state_tensor : List[float]
        The current [Traffic_Load, DB_Temperature, Network_Health] triple.
        Updated by the executor node after each action.

    previous_state_tensor : List[float]
        Snapshot of the state before the most recent action (for delta logging).

    chat_history : List[ChatMessage]
        Append-only log of all inter-agent messages this episode.

    network_intent : Optional[NetworkIntent]
        Latest micro-intent from the Network Controller.

    db_intent : Optional[DBIntent]
        Latest micro-intent from the Database Controller.

    routing_path : RoutingPath
        Which tier of the 3-tier routing logic is currently active.

    consensus_status : ConsensusStatus
        GREEN if the two controller intents are synergistic; RED otherwise.

    recommended_action : Optional[str]
        The final resolved action string to execute in the environment
        (one of env.environment.VALID_ACTIONS).

    dna_memory_hit : Optional[Dict]
        Serialised :class:`~utils.dna_memory.MemoryHit` if a fast-path
        cache hit was found; ``None`` otherwise.

    episode_step : int
        Monotonically increasing step counter within the current episode.

    is_resolved : bool
        True once the SLO score crosses 0.95 (system recovered).

    slo_score : float
        Most recent SLO score in [0, 1].

    error : Optional[str]
        Non-None if any node encountered a recoverable error.

    metadata : Dict[str, Any]
        Free-form bag for experiment tracking (e.g. model names, run IDs).
    """

    current_state_tensor: List[float]
    previous_state_tensor: List[float]
    chat_history: List[ChatMessage]
    network_intent: Optional[NetworkIntent]
    db_intent: Optional[DBIntent]
    routing_path: RoutingPath
    consensus_status: ConsensusStatus
    recommended_action: Optional[str]
    dna_memory_hit: Optional[Dict[str, Any]]
    episode_step: int
    is_resolved: bool
    slo_score: float
    error: Optional[str]
    metadata: Dict[str, Any]

    # ── Governance filter outputs (populated by LeadSRENode) ──────────────────

    governance_signal: GovernanceSignal
    """Unified routing decision after all three governance filters."""

    blast_radius_warnings: List[str]
    """
    List of secondary-impact strings flagged by the Blast Radius filter.
    Empty when no risks were identified.  Non-empty when the action was
    blocked or required human review.
    """

    trust_decision: TrustDecision
    """Adaptive Trust Layer decision: APPROVED or ESCROWED."""

    human_approved: bool
    """
    Set to True by the UI 'Approve' button when trust_decision == ESCROWED.
    The executor node checks this before proceeding.
    """


# ───────────────────────────── factory helpers ────────────────────────────────


def initial_state(
    state_vector: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SREGraphState:
    """
    Create a clean initial :class:`SREGraphState` for the start of an episode.

    Parameters
    ----------
    state_vector:
        Initial ``[Traffic_Load, DB_Temperature, Network_Health]``.
        Defaults to a crashed datacenter state if not provided.
    metadata:
        Optional experiment metadata dict (run IDs, model names, etc.).

    Returns
    -------
    SREGraphState
    """
    from env.state_tensor import CloudStateTensor   # lazy import avoids cycles

    if state_vector is None:
        state_vector = CloudStateTensor.crashed().as_list()

    return SREGraphState(
        current_state_tensor=state_vector,
        previous_state_tensor=state_vector,
        chat_history=[],
        network_intent=None,
        db_intent=None,
        routing_path=RoutingPath.MIDDLE,
        consensus_status=ConsensusStatus.RED,
        recommended_action=None,
        dna_memory_hit=None,
        episode_step=0,
        is_resolved=False,
        slo_score=0.0,
        error=None,
        metadata=metadata or {},
        # Governance filter defaults
        governance_signal=GovernanceSignal.DEEP_NEGOTIATE,
        blast_radius_warnings=[],
        trust_decision=TrustDecision.ESCROWED,
        human_approved=False,
    )


def append_chat(
    state: SREGraphState,
    role: str,
    content: str,
) -> SREGraphState:
    """
    Return a *new* state with *content* appended to ``chat_history``.

    This is the recommended pattern inside LangGraph node functions since
    TypedDict instances are technically mutable but we treat them as immutable
    for clarity.
    """
    import datetime

    msg = ChatMessage(
        role=role,
        content=content,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )
    updated_history = list(state.get("chat_history", [])) + [msg]
    return {**state, "chat_history": updated_history}  # type: ignore[return-value]
