"""
controllers/network_controller.py
===================================
Network SRE Controller — LangGraph node.

Responsibilities
----------------
  • Analyse the current :class:`~env.state_tensor.CloudStateTensor` with a
    focus ONLY on traffic and network-health metrics.
  • Emit either:
      - A lightweight **micro-intent JSON** (used by the Shadow Consensus layer
        when confidence is high and the signal is clear).
      - A **full diagnostic string** (used in the SLOW path when the signal is
        ambiguous and deep reasoning via ChatOps is needed).
  • Never reason about database internals — that is the DB controller's domain.

Integration
-----------
The node accepts and returns the shared :class:`~graph.message_bus.SREGraphState`
dict, making it a drop-in LangGraph node.  Call ``run_as_node(state)`` for
direct LangGraph wiring.

LLM Usage
---------
  • Uses ``gpt-4o-mini`` by default (fast + cheap for intent generation).
  • Falls back to :func:`_rule_based_intent` when no API key is present.
  • LLM is instructed to output **JSON only** (enforced by ``response_format``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from env.state_tensor import CloudStateTensor
from graph.message_bus import (
    SREGraphState,
    NetworkIntent,
    append_chat,
)
from controllers.system_prompts import NETWORK_CONTROLLER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ──────────────────────────── optional LLM ───────────────────────────────────
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore[assignment, misc]

_DEFAULT_MODEL = "gpt-4o-mini"

# ──────────────────────── diagnostic thresholds ──────────────────────────────

# When confidence drops below this, switch to full-diagnostic mode
_DEEP_REASON_CONFIDENCE_THRESHOLD = 0.50

# ─────────────────────────── rule-based fallback ─────────────────────────────


def _rule_based_intent(
    traffic: float, net_health: float
) -> Tuple[NetworkIntent, bool]:
    """
    Deterministic intent generator for offline / no-LLM operation.

    Returns (NetworkIntent, needs_deep_reasoning).
    """
    if traffic > 85:
        return (
            NetworkIntent(
                intent="circuit_break",
                confidence=0.92,
                rationale=f"Traffic_Load={traffic:.1f} is critically high; circuit breaker required.",
            ),
            False,
        )
    if traffic > 70:
        return (
            NetworkIntent(
                intent="throttle",
                confidence=0.78,
                rationale=f"Traffic_Load={traffic:.1f} elevated; throttle to prevent cascade.",
            ),
            False,
        )
    if net_health < 20:
        return (
            NetworkIntent(
                intent="scale_out",
                confidence=0.85,
                rationale=f"Network_Health={net_health:.1f} critically low; scale to restore capacity.",
            ),
            False,
        )
    if net_health < 40:
        return (
            NetworkIntent(
                intent="scale_out",
                confidence=0.65,
                rationale=f"Network_Health={net_health:.1f} degraded; scaling recommended.",
            ),
            False,
        )
    if traffic > 50:
        return (
            NetworkIntent(
                intent="load_balance",
                confidence=0.60,
                rationale=f"Traffic_Load={traffic:.1f} moderate; redistributing load.",
            ),
            False,
        )
    if traffic < 30 and net_health > 70:
        return (
            NetworkIntent(
                intent="noop",
                confidence=0.30,
                rationale="Network metrics within acceptable bounds.",
            ),
            False,
        )
    # Ambiguous — deep reason
    return (
        NetworkIntent(
            intent="noop",
            confidence=0.35,
            rationale=f"Ambiguous state: Traffic={traffic:.1f}, Health={net_health:.1f}. Deep reasoning needed.",
        ),
        True,
    )


def _build_diagnostic(
    traffic: float, net_health: float, db_temp: float, step: int
) -> str:
    """Full diagnostic text used in the SLOW path."""
    return (
        f"[Network Diagnostic / Step {step}] "
        f"Traffic_Load={traffic:.1f} | Network_Health={net_health:.1f} | "
        f"DB_Temperature={db_temp:.1f} (context only). "
        f"Signal is ambiguous — confidence below threshold. "
        f"Recommend ChatOps deep-negotiation before committing an action."
    )


# ─────────────────────────── LLM call helper ─────────────────────────────────


def _call_llm_for_intent(
    client: Any,
    state_vec: list[float],
    chat_history: list,
    step: int,
    model: str,
) -> Tuple[NetworkIntent, bool]:
    """
    Call the LLM and parse the NetworkIntent JSON.
    Returns (intent, needs_deep_reasoning).
    """
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
            {"role": "system", "content": NETWORK_CONTROLLER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)

    intent = NetworkIntent(
        intent=str(parsed.get("intent", "noop")),
        confidence=float(parsed.get("confidence", 0.5)),
        rationale=str(parsed.get("rationale", "")),
    )
    needs_deep = intent["confidence"] < _DEEP_REASON_CONFIDENCE_THRESHOLD
    return intent, needs_deep


# ──────────────────────── NetworkController class ────────────────────────────


class NetworkController:
    """
    Network SRE Controller.

    Parameters
    ----------
    model:
        OpenAI model name used for LLM-backed reasoning.
    use_llm:
        Set False to force rule-based mode (for tests / CI).
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
    ) -> Tuple[NetworkIntent, bool, str]:
        """
        Analyse the network layer and produce an intent.

        Parameters
        ----------
        state_vec:
            ``[Traffic_Load, DB_Temperature, Network_Health]``
        chat_history:
            Recent message history for LLM context.
        step:
            Current episode step number.

        Returns
        -------
        Tuple[NetworkIntent, bool, str]
            intent, needs_deep_reasoning, diagnostic_string
        """
        traffic, db_temp, net_health = state_vec

        if self._client is not None:
            try:
                intent, needs_deep = _call_llm_for_intent(
                    self._client, state_vec, chat_history, step, self.model
                )
            except Exception as exc:
                logger.warning("NetworkController LLM call failed: %s", exc)
                intent, needs_deep = _rule_based_intent(traffic, net_health)
        else:
            intent, needs_deep = _rule_based_intent(traffic, net_health)

        diagnostic = (
            _build_diagnostic(traffic, net_health, db_temp, step)
            if needs_deep
            else f"[Network Intent] {intent['intent']} (confidence={intent['confidence']:.2f})"
        )
        return intent, needs_deep, diagnostic

    # ──────────────────── LangGraph node interface ───────────────────────────

    def run_as_node(self, state: SREGraphState) -> SREGraphState:
        """
        LangGraph-compatible node function.

        Reads from ``SREGraphState``, runs the network analysis, and returns
        a partial state update with ``network_intent`` populated.
        """
        vec = state.get("current_state_tensor", [98.0, 95.0, 5.0])
        step = state.get("episode_step", 0)
        history = state.get("chat_history", [])

        intent, needs_deep, diagnostic = self.analyse(vec, history, step)

        state = append_chat(state, role="network_ctrl", content=diagnostic)

        return {
            **state,  # type: ignore[return-value]
            "network_intent": intent,
        }


# ──────────────────── module-level LangGraph node factory ────────────────────


def make_network_node(
    use_llm: bool = True,
    model: str = _DEFAULT_MODEL,
) -> Any:
    """
    Return a LangGraph-compatible callable bound to a :class:`NetworkController`.

    Usage in ``sre_graph.py``::

        from controllers.network_controller import make_network_node
        builder.add_node("network_controller_node", make_network_node())
    """
    ctrl = NetworkController(model=model, use_llm=use_llm)
    return ctrl.run_as_node
