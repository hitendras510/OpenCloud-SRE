"""
controllers/database_controller.py
====================================
Database SRE Controller — LangGraph node.

Responsibilities
----------------
  • Analyse the current :class:`~env.state_tensor.CloudStateTensor` with a
    focus ONLY on database metrics (Database_Temperature and Traffic_Load
    as it relates to query pressure).
  • Emit either:
      - A lightweight **DB micro-intent JSON** for the Shadow Consensus layer.
      - A **full diagnostic string** when ambiguity demands ChatOps escalation.
  • Never reason about network topology, BGP, or load balancers.

Integration
-----------
Mirrors the NetworkController interface exactly.  Accepts/returns
:class:`~graph.message_bus.SREGraphState`.  Use ``run_as_node(state)``
for LangGraph wiring or ``make_db_node()`` for the factory pattern.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Tuple

from graph.message_bus import (
    SREGraphState,
    DBIntent,
    append_chat,
)
from controllers.system_prompts import DATABASE_CONTROLLER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ──────────────────────────── optional LLM ───────────────────────────────────
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore[assignment, misc]

_DEFAULT_MODEL = "gpt-4o-mini"
_DEEP_REASON_CONFIDENCE_THRESHOLD = 0.50

# ─────────────────────────── rule-based fallback ─────────────────────────────


def _rule_based_db_intent(
    db_temp: float,
    traffic: float,
    chat_history: list,
) -> Tuple[DBIntent, bool]:
    """
    Deterministic DB intent for offline / no-LLM operation.

    Tracks recent restart attempts in chat_history to prevent repeated restarts.
    Returns (DBIntent, needs_deep_reasoning).
    """
    # Detect if restart was recently used
    recent_roles = [m.get("content", "") for m in chat_history[-6:]]
    recent_restart = any("restart" in c.lower() for c in recent_roles)

    if db_temp > 85:
        return (
            DBIntent(
                intent="failover",
                confidence=0.93,
                rationale=f"DB_Temperature={db_temp:.1f} critical; failover to standby required.",
            ),
            False,
        )
    if db_temp > 70:
        return (
            DBIntent(
                intent="cache_flush",
                confidence=0.75,
                rationale=f"DB_Temperature={db_temp:.1f} elevated; cache flush to relieve pressure.",
            ),
            False,
        )
    if db_temp > 55 and traffic > 75:
        return (
            DBIntent(
                intent="cache_flush",
                confidence=0.62,
                rationale=f"Combined DB_Temp={db_temp:.1f} + Traffic={traffic:.1f}; pre-emptive cache flush.",
            ),
            False,
        )
    if db_temp > 55 and not recent_restart:
        return (
            DBIntent(
                intent="restart",
                confidence=0.55,
                rationale=f"DB_Temperature={db_temp:.1f} moderately high; pod restart to clear connections.",
            ),
            False,
        )
    if db_temp < 40:
        return (
            DBIntent(
                intent="noop",
                confidence=0.20,
                rationale=f"DB_Temperature={db_temp:.1f} within nominal range.",
            ),
            False,
        )
    # Ambiguous zone
    return (
        DBIntent(
            intent="noop",
            confidence=0.38,
            rationale=f"Ambiguous: DB_Temp={db_temp:.1f}, Traffic={traffic:.1f}. Deep reasoning needed.",
        ),
        True,
    )


def _build_db_diagnostic(
    db_temp: float, traffic: float, net_health: float, step: int
) -> str:
    return (
        f"[DB Diagnostic / Step {step}] "
        f"DB_Temperature={db_temp:.1f} | Traffic_Load={traffic:.1f} | "
        f"Network_Health={net_health:.1f} (context). "
        f"DB signal is ambiguous — escalating to ChatOps."
    )


# ─────────────────────────── LLM call helper ─────────────────────────────────


def _call_llm_for_db_intent(
    client: Any,
    state_vec: list[float],
    chat_history: list,
    step: int,
    model: str,
) -> Tuple[DBIntent, bool]:
    user_msg = json.dumps({
        "current_state": {
            "Traffic_Load": state_vec[0],
            "Database_Temperature": state_vec[1],
            "Network_Health": state_vec[2],
        },
        "episode_step": step,
        "chat_history": chat_history[-5:],
    })

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DATABASE_CONTROLLER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)

    intent = DBIntent(
        intent=str(parsed.get("intent", "noop")),
        confidence=float(parsed.get("confidence", 0.5)),
        rationale=str(parsed.get("rationale", "")),
    )
    needs_deep = intent["confidence"] < _DEEP_REASON_CONFIDENCE_THRESHOLD
    return intent, needs_deep


# ─────────────────────── DatabaseController class ────────────────────────────


class DatabaseController:
    """
    Database SRE Controller.

    Parameters
    ----------
    model:
        OpenAI model for LLM-backed reasoning.
    use_llm:
        False → always use rule-based fallback.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        use_llm: bool = True,
    ) -> None:
        self.model = model
        self._client: Optional[Any] = None

        if use_llm and _OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                self._client = _OpenAI(api_key=api_key)

    # ───────────────────────── core analysis ─────────────────────────────────

    def analyse(
        self,
        state_vec: list[float],
        chat_history: list,
        step: int = 0,
    ) -> Tuple[DBIntent, bool, str]:
        """
        Analyse DB metrics and produce an intent.

        Returns
        -------
        Tuple[DBIntent, bool, str]
            intent, needs_deep_reasoning, diagnostic_string
        """
        traffic, db_temp, net_health = state_vec

        if self._client is not None:
            try:
                intent, needs_deep = _call_llm_for_db_intent(
                    self._client, state_vec, chat_history, step, self.model
                )
            except Exception as exc:
                logger.warning("DatabaseController LLM call failed: %s", exc)
                intent, needs_deep = _rule_based_db_intent(db_temp, traffic, chat_history)
        else:
            intent, needs_deep = _rule_based_db_intent(db_temp, traffic, chat_history)

        diagnostic = (
            _build_db_diagnostic(db_temp, traffic, net_health, step)
            if needs_deep
            else (
                f"[DB Intent] {intent['intent']} "
                f"(confidence={intent['confidence']:.2f}) | {intent['rationale']}"
            )
        )
        return intent, needs_deep, diagnostic

    # ──────────────────── LangGraph node interface ───────────────────────────

    def run_as_node(self, state: SREGraphState) -> SREGraphState:
        """LangGraph-compatible node: reads state → returns partial state update."""
        vec = state.get("current_state_tensor", [98.0, 95.0, 5.0])
        step = state.get("episode_step", 0)
        history = state.get("chat_history", [])

        intent, needs_deep, diagnostic = self.analyse(vec, history, step)

        state = append_chat(state, role="db_ctrl", content=diagnostic)

        return {
            **state,  # type: ignore[return-value]
            "db_intent": intent,
        }


# ──────────────────── module-level LangGraph node factory ────────────────────


def make_db_node(
    use_llm: bool = True,
    model: str = _DEFAULT_MODEL,
) -> Any:
    """
    Return a LangGraph-compatible callable bound to a :class:`DatabaseController`.

    Usage in ``sre_graph.py``::

        from controllers.database_controller import make_db_node
        builder.add_node("db_controller_node", make_db_node())
    """
    ctrl = DatabaseController(model=model, use_llm=use_llm)
    return ctrl.run_as_node
