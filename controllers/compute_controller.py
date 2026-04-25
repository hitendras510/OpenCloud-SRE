"""
controllers/compute_controller.py
==================================
Compute Reliability Agent — Shadow Consensus GRPO Worker #1.

Responsibilities
----------------
  • Analyse CPU and Traffic_Load metrics exclusively.
  • Emit a :class:`~graph.message_bus.ComputeIntent` with a ``confidence_score``
    that signals how urgently the Compute layer needs to act.
  • A LOW confidence_score (< 0.3) + "noop" means "I see nothing wrong here;
    let the Network or Database agent take the lead."
  • Pairs with NetworkAgentController and DatabaseAgentController.
    All three are run concurrently before the Shadow Arbiter node.

Integration
-----------
Use ``make_compute_agent_node()`` for the LangGraph factory pattern::

    from controllers.compute_controller import make_compute_agent_node
    builder.add_node("compute_agent_node", make_compute_agent_node())
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Tuple

from graph.message_bus import (
    SREGraphState,
    ComputeIntent,
    append_chat,
)
from controllers.system_prompts import COMPUTE_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ──────────────────────────── optional HF LLM ────────────────────────────────
try:
    from huggingface_hub import InferenceClient as _HFClient
    _HF_AVAILABLE = True
except ImportError:
    _HFClient = None  # type: ignore[assignment]
    _HF_AVAILABLE = False

_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# ─────────────────────────── rule-based fallback ─────────────────────────────

_CPU_CRITICAL   = 85.0   # Above this → high-confidence load-shedding
_CPU_ELEVATED   = 70.0   # Above this → moderate throttle
_TRAFFIC_HIGH   = 80.0   # Above this → scale_out recommended


def _rule_based_compute_intent(
    cpu: float,
    traffic: float,
) -> ComputeIntent:
    """
    Deterministic Compute Agent for offline / no-LLM operation.

    The confidence_score reflects how certain the agent is that the Compute
    domain is the root cause — mirrors the hackathon example exactly:
      - CPU spike → confidence ≈ 0.95
      - Traffic only → confidence ≈ 0.75
      - Nominal   → confidence ≈ 0.10  (yield to other agents)
    """
    if cpu > _CPU_CRITICAL:
        return ComputeIntent(
            agent_role="Compute",
            diagnosis=(
                f"CPU={cpu:.1f} is critically high — server is under severe compute "
                f"pressure. Load-shedding required immediately."
            ),
            confidence_score=0.95,
            proposed_action="throttle_traffic",
        )
    if cpu > _CPU_ELEVATED:
        return ComputeIntent(
            agent_role="Compute",
            diagnosis=(
                f"CPU={cpu:.1f} is elevated. Traffic_Load={traffic:.1f}. "
                f"Scale out to absorb demand before saturation."
            ),
            confidence_score=0.75,
            proposed_action="scale_out",
        )
    if traffic > _TRAFFIC_HIGH:
        return ComputeIntent(
            agent_role="Compute",
            diagnosis=(
                f"Traffic_Load={traffic:.1f} is high but CPU={cpu:.1f} is still "
                f"healthy. Proactive scale_out to prevent CPU ceiling."
            ),
            confidence_score=0.60,
            proposed_action="scale_out",
        )
    # Nominal — stay out of the debate
    return ComputeIntent(
        agent_role="Compute",
        diagnosis=(
            f"CPU={cpu:.1f} and Traffic_Load={traffic:.1f} are within normal bounds. "
            f"Compute layer is healthy — deferring to other agents."
        ),
        confidence_score=0.10,
        proposed_action="noop",
    )


# ─────────────────────────── LLM call helper ─────────────────────────────────

def _call_llm_for_compute_intent(
    client: Any,
    metrics: dict,
    model: str,
) -> ComputeIntent:
    """Call HF InferenceClient and parse a ComputeIntent from the response."""
    user_msg = json.dumps(metrics)

    resp = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": COMPUTE_AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)

    return ComputeIntent(
        agent_role=str(parsed.get("agent_role", "Compute")),
        diagnosis=str(parsed.get("diagnosis", "")),
        confidence_score=float(parsed.get("confidence_score", 0.1)),
        proposed_action=str(parsed.get("proposed_action", "noop")),
    )


# ─────────────────────── ComputeAgentController class ────────────────────────


class ComputeAgentController:
    """
    Compute Reliability Agent (Shadow Consensus GRPO Worker).

    Parameters
    ----------
    model:
        HuggingFace model ID for live LLM calls.
    use_llm:
        False → always use rule-based fallback (offline / CI safe).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        use_llm: bool = True,
    ) -> None:
        self.model = model
        self._client: Optional[Any] = None

        if use_llm and _HF_AVAILABLE and _HFClient:
            token = os.getenv("HF_TOKEN", "")
            if token:
                self._client = _HFClient(token=token)

    # ───────────────────────── core analysis ─────────────────────────────────

    def analyse(
        self,
        state_vec: list[float],
    ) -> ComputeIntent:
        """
        Analyse compute metrics and return a :class:`ComputeIntent`.

        Parameters
        ----------
        state_vec:
            ``[Traffic_Load, DB_Temperature, Network_Health]`` triple from
            the environment.  The ``env/server.py`` also exposes CPU as
            Traffic_Load — they're the same axis in the state tensor.
        """
        traffic, db_temp, net_health = state_vec
        # In the server.py GlobalStateManager, CPU == Traffic_Load
        cpu = traffic

        metrics = {
            "CPU": cpu,
            "Traffic_Load": traffic,
            "DB_Temp": db_temp,
            "Latency": net_health,  # server exposes latency via Network_Health axis
        }

        if self._client is not None:
            try:
                return _call_llm_for_compute_intent(self._client, metrics, self.model)
            except Exception as exc:
                logger.warning("ComputeAgentController LLM call failed: %s", exc)

        return _rule_based_compute_intent(cpu, traffic)

    # ──────────────────── LangGraph node interface ───────────────────────────

    def run_as_node(self, state: SREGraphState) -> SREGraphState:
        """LangGraph-compatible node: reads state → returns partial state update."""
        vec = state.get("current_state_tensor", [50.0, 50.0, 5.0])
        intent = self.analyse(vec)

        state = append_chat(
            state,
            role="compute_agent",
            content=(
                f"[Compute Agent] confidence={intent['confidence_score']:.2f} | "
                f"action={intent['proposed_action']} | {intent['diagnosis']}"
            ),
        )

        return {
            **state,  # type: ignore[return-value]
            "compute_intent": intent,
        }


# ──────────────────── module-level LangGraph node factory ────────────────────


def make_compute_agent_node(
    use_llm: bool = True,
    model: str = _DEFAULT_MODEL,
) -> Any:
    """
    Return a LangGraph-compatible callable bound to a :class:`ComputeAgentController`.

    Usage in ``sre_graph.py``::

        from controllers.compute_controller import make_compute_agent_node
        builder.add_node("compute_agent_node", make_compute_agent_node())
    """
    ctrl = ComputeAgentController(model=model, use_llm=use_llm)
    return ctrl.run_as_node
