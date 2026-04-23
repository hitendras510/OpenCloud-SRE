"""
env/fault_injection.py
=======================
ChaosMonkey – AI-driven fault injection engine for OpenCloud-SRE.

The ChaosMonkey uses GPT-4o (with a rule-based fallback) to *dynamically*
decide which infrastructure metric to degrade and by how much, simulating
realistic non-deterministic infrastructure failures.

Design
------
  • The LLM is prompted to act as a "Chaos Engineer" — it selects a fault
    category, the target metric(s), magnitude, and provides a realistic
    incident description string.
  • Output is strictly JSON for deterministic parsing.
  • A rule-based fallback generates statistically realistic faults without
    any LLM dependency (for CI / local-only runs).
  • All injected faults are recorded in a :class:`FaultLog` for post-episode
    evaluation and dataset generation.

Fault Taxonomy
--------------
  TRAFFIC_SPIKE       – Sudden ingress surge (e.g., flash-crowd, bot attack)
  DB_OVERLOAD         – Hot-key storm or runaway query filling the DB queue
  NETWORK_PARTITION   – Partial link failure or BGP route flap
  CASCADE_FAILURE     – Correlated multi-metric degradation
  MEMORY_LEAK_OOM     – Pod OOM kill triggering restart loop (traffic + health)
  REPLICA_LAG         – Replication delay causing read-replica overload (DB)
  HOT_KEY_STORM       – Concentrated cache misses on single key prefix (DB+traffic)
  BGP_FLAP            – Routing table churn causing intermittent packet loss (net)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from env.state_tensor import CloudStateTensor

logger = logging.getLogger(__name__)

# ─────────────────────────── optional OpenAI ─────────────────────────────────
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore[assignment, misc]


# ──────────────────────────────── enums / types ───────────────────────────────

class FaultCategory(str, Enum):
    TRAFFIC_SPIKE = "traffic_spike"
    DB_OVERLOAD = "db_overload"
    NETWORK_PARTITION = "network_partition"
    CASCADE_FAILURE = "cascade_failure"
    MEMORY_LEAK_OOM = "memory_leak_oom"
    REPLICA_LAG = "replica_lag"
    HOT_KEY_STORM = "hot_key_storm"
    BGP_FLAP = "bgp_flap"


# (traffic_delta_range, db_temp_delta_range, net_health_delta_range)
_FAULT_DELTA_RANGES: Dict[FaultCategory, Tuple[Tuple, Tuple, Tuple]] = {
    FaultCategory.TRAFFIC_SPIKE:     ((+35, +55), (-2, +5),  (-8, +2)),
    FaultCategory.DB_OVERLOAD:       ((+5,  +15), (+35, +55),(-5, +3)),
    FaultCategory.NETWORK_PARTITION: ((+8,  +20), (+2,  +8), (-45, -30)),
    FaultCategory.CASCADE_FAILURE:   ((+25, +40), (+25, +40),(-35, -20)),
    FaultCategory.MEMORY_LEAK_OOM:   ((+15, +30), (+10, +20),(-15, -5)),
    FaultCategory.REPLICA_LAG:       ((+5,  +12), (+20, +38),(-5,  +5)),
    FaultCategory.HOT_KEY_STORM:     ((+20, +35), (+25, +40),(-5,  +5)),
    FaultCategory.BGP_FLAP:          ((+5,  +15), (+2,  +8), (-40, -20)),
}

# ──────────────────────────── data structures ────────────────────────────────


@dataclass
class FaultEvent:
    """Record of a single chaos injection."""
    fault_id: str
    category: FaultCategory
    description: str                   # LLM-generated incident description
    delta_applied: Tuple[float, float, float]
    state_before: List[float]
    state_after: List[float]
    timestamp: str
    severity: str                      # "low" | "medium" | "high" | "critical"
    generated_by: str                  # "llm" | "rule_based"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "category": self.category.value,
            "description": self.description,
            "delta_applied": list(self.delta_applied),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "generated_by": self.generated_by,
        }

    def __str__(self) -> str:
        return (
            f"[ChaosMonkey/{self.severity.upper()}] {self.category.value} "
            f"| {self.description[:80]} "
            f"| Δ={[round(d,1) for d in self.delta_applied]}"
        )


# ─────────────────────────── LLM prompt ──────────────────────────────────────

_CHAOS_SYSTEM_PROMPT = """
You are a Chaos Engineering AI. Your job is to inject realistic infrastructure
failures into a simulated data-centre to test its autonomous recovery systems.

The data-centre has three metrics:
  Traffic_Load       (0–100): higher = more congested
  Database_Temperature (0–100): higher = more overloaded
  Network_Health     (0–100): lower = more degraded

Current state: {state_json}

Choose one fault to inject. Output ONLY valid JSON, no other text:
{{
  "category": "<one of: traffic_spike | db_overload | network_partition | cascade_failure | memory_leak_oom | replica_lag | hot_key_storm | bgp_flap>",
  "severity": "<low | medium | high | critical>",
  "traffic_delta": <float, positive means load INCREASES>,
  "db_temp_delta": <float, positive means temperature INCREASES>,
  "net_health_delta": <float, NEGATIVE means health DEGRADES>,
  "description": "<one sentence realistic incident description, max 25 words>"
}}

Rules:
- The deltas must be consistent with the chosen category.
- Severity must match the magnitude of deltas.
- critical severity: any delta magnitude > 40.
- high severity: any delta magnitude 25–40.
- medium: 10–25. low: < 10.
- Do NOT choose a fault that would have zero effect on the current state.
""".strip()


# ─────────────────────────── ChaosMonkey class ───────────────────────────────

class ChaosMonkey:
    """
    AI-driven chaos injection engine.

    Parameters
    ----------
    use_llm:
        If True, attempts to use GPT-4o for dynamic fault selection.
        Falls back to rule-based generation if no API key is available.
    model:
        OpenAI model to use for fault generation.
    seed:
        Random seed for reproducibility in rule-based mode.

    Example
    -------
    >>> monkey = ChaosMonkey(use_llm=False, seed=42)
    >>> state = CloudStateTensor.nominal()
    >>> event = monkey.inject(state)
    >>> print(event)
    """

    def __init__(
        self,
        use_llm: bool = True,
        model: str = "gpt-4o-mini",
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self._use_llm = use_llm
        self._client: Optional[Any] = None
        self._fault_log: List[FaultEvent] = []
        self._counter: int = 0

        if seed is not None:
            random.seed(seed)

        if use_llm and _OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                self._client = _OpenAI(api_key=api_key)
                logger.info("ChaosMonkey: LLM mode active (%s).", model)
            else:
                logger.warning(
                    "ChaosMonkey: OPENAI_API_KEY not set – using rule-based faults."
                )

    # ────────────────────────── public API ───────────────────────────────────

    def inject(
        self,
        state: CloudStateTensor,
        category_override: Optional[FaultCategory] = None,
    ) -> Tuple[FaultEvent, CloudStateTensor]:
        """
        Inject one fault into *state* and return the event + new state.

        Parameters
        ----------
        state:
            Current datacenter state tensor.
        category_override:
            If provided, forces this fault category regardless of LLM selection.
            Useful for targeted testing.

        Returns
        -------
        Tuple[FaultEvent, CloudStateTensor]
            The fault event record and the mutated state tensor.
        """
        self._counter += 1
        fault_id = f"chaos-{self._counter:04d}"
        before = state.as_list()

        if category_override is not None:
            delta, description, severity, source = self._rule_based_fault(
                state, force_category=category_override
            )
        elif self._client is not None:
            try:
                delta, description, severity, source, _ = self._llm_fault(state)
            except Exception as exc:
                logger.warning("LLM fault generation failed (%s) – falling back.", exc)
                delta, description, severity, source = self._rule_based_fault(state)
        else:
            delta, description, severity, source = self._rule_based_fault(state)

        # Apply delta to get new state
        delta_tensor = torch.tensor(list(delta), dtype=torch.float32)
        new_state = state.apply_delta(delta_tensor)
        after = new_state.as_list()

        # Determine category from description for rule-based path
        category = self._infer_category(description, delta)

        event = FaultEvent(
            fault_id=fault_id,
            category=category,
            description=description,
            delta_applied=delta,
            state_before=before,
            state_after=after,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            severity=severity,
            generated_by=source,
        )
        self._fault_log.append(event)
        logger.info("%s", event)
        return event, new_state

    def inject_into_env(
        self,
        env: Any,  # OpenCloudEnv – avoid circular import
        category_override: Optional[FaultCategory] = None,
    ) -> FaultEvent:
        """
        Inject a fault directly into an :class:`~env.environment.OpenCloudEnv`.

        Mutates ``env.state`` in place and returns the fault event.
        """
        event, new_state = self.inject(env.state, category_override)
        env.state = new_state
        return event

    def get_log(self) -> List[FaultEvent]:
        """Return a copy of all injected fault events this session."""
        return list(self._fault_log)

    def clear_log(self) -> None:
        """Clear fault history (call at episode reset)."""
        self._fault_log.clear()
        self._counter = 0

    def describe(self) -> str:
        return (
            f"ChaosMonkey(mode={'llm' if self._client else 'rule_based'}, "
            f"model={self.model}, injections={self._counter})"
        )

    # ────────────────────────── LLM path ─────────────────────────────────────

    def _llm_fault(
        self, state: CloudStateTensor
    ) -> Tuple[Tuple[float, float, float], str, str, str, str]:
        """
        Call GPT-4o to generate a dynamic fault.

        Returns (delta, description, severity, source="llm", raw_category)
        """
        state_json = json.dumps({
            "Traffic_Load": round(state.traffic_load, 1),
            "Database_Temperature": round(state.database_temperature, 1),
            "Network_Health": round(state.network_health, 1),
        })
        prompt = _CHAOS_SYSTEM_PROMPT.format(state_json=state_json)

        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            temperature=0.9,   # high creativity for diverse faults
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a chaos engineering AI."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        delta = (
            float(parsed.get("traffic_delta", 0.0)),
            float(parsed.get("db_temp_delta", 0.0)),
            float(parsed.get("net_health_delta", 0.0)),
        )
        description = str(parsed.get("description", "LLM-generated fault."))
        severity = str(parsed.get("severity", "medium"))
        category_str = str(parsed.get("category", "traffic_spike"))
        return delta, description, severity, "llm", category_str

    # ────────────────────────── rule-based path ───────────────────────────────

    def _rule_based_fault(
        self,
        state: CloudStateTensor,
        force_category: Optional[FaultCategory] = None,
    ) -> Tuple[Tuple[float, float, float], str, str, str]:
        """
        Deterministic rule-based fault generation.

        Weights fault categories based on current state to generate
        contextually plausible failures.

        Returns (delta, description, severity, source="rule_based")
        """
        if force_category is not None:
            category = force_category
        else:
            # Weight categories by current state stress level
            weights: Dict[FaultCategory, float] = {
                FaultCategory.TRAFFIC_SPIKE:     1.0,
                FaultCategory.DB_OVERLOAD:       1.0,
                FaultCategory.NETWORK_PARTITION: 1.0,
                FaultCategory.CASCADE_FAILURE:   0.5,
                FaultCategory.MEMORY_LEAK_OOM:   0.8,
                FaultCategory.REPLICA_LAG:       0.7,
                FaultCategory.HOT_KEY_STORM:     0.6,
                FaultCategory.BGP_FLAP:          0.6,
            }
            # Increase likelihood of already-stressed dimension getting worse
            if state.traffic_load > 70:
                weights[FaultCategory.TRAFFIC_SPIKE] *= 2.0
                weights[FaultCategory.CASCADE_FAILURE] *= 1.5
            if state.database_temperature > 70:
                weights[FaultCategory.DB_OVERLOAD] *= 2.0
                weights[FaultCategory.HOT_KEY_STORM] *= 1.8
            if state.network_health < 40:
                weights[FaultCategory.NETWORK_PARTITION] *= 2.0
                weights[FaultCategory.BGP_FLAP] *= 1.8

            categories = list(weights.keys())
            wts = [weights[c] for c in categories]
            category = random.choices(categories, weights=wts, k=1)[0]

        ranges = _FAULT_DELTA_RANGES[category]
        delta = (
            float(random.uniform(*ranges[0])),
            float(random.uniform(*ranges[1])),
            float(random.uniform(*ranges[2])),
        )

        max_magnitude = max(abs(d) for d in delta)
        severity = (
            "critical" if max_magnitude > 40
            else "high" if max_magnitude > 25
            else "medium" if max_magnitude > 10
            else "low"
        )

        _DESCRIPTIONS: Dict[FaultCategory, List[str]] = {
            FaultCategory.TRAFFIC_SPIKE: [
                "Flash-crowd event detected: ingress RPS spiked 400% above baseline.",
                "Bot swarm targeting /api/checkout: traffic surge overloading edge nodes.",
                "Marketing campaign launch triggered unexpected 5× traffic spike.",
            ],
            FaultCategory.DB_OVERLOAD: [
                "Runaway analytics query holding table lock on primary Postgres.",
                "Hot-key contention on user-session Redis causing DB fan-out storm.",
                "Connection pool exhausted: all 500 DB connections saturated.",
            ],
            FaultCategory.NETWORK_PARTITION: [
                "Partial AZ network partition: 30% of cross-zone packets dropped.",
                "Misconfigured security group blocking inter-service gRPC calls.",
                "Switch fabric error causing microsecond-level packet bursts.",
            ],
            FaultCategory.CASCADE_FAILURE: [
                "Cascading failure: DB overload → request queuing → traffic backup.",
                "Thundering herd after pod restart: cache cold start causing DB storm.",
                "Multi-AZ simultaneous degradation triggered by upstream DNS NXDOMAIN.",
            ],
            FaultCategory.MEMORY_LEAK_OOM: [
                "api-worker pods OOM-killed by kernel: restart loop initiated.",
                "Memory leak in session handler: pods evicted, traffic shedding.",
                "JVM heap exhaustion causing GC pause storm and pod evictions.",
            ],
            FaultCategory.REPLICA_LAG: [
                "Read replica 5s behind primary: read queries fanning out to primary.",
                "Replication slot bloat causing WAL disk fill on standby.",
                "Binary log delay exceeding 10s: replica unhealthy, read traffic failing.",
            ],
            FaultCategory.HOT_KEY_STORM: [
                "Single product-page key receiving 80% of all cache reads.",
                "Thundering herd on expired OAuth token key: cache miss storm.",
                "Viral content causing concentrated hot-key overload on Redis shard 3.",
            ],
            FaultCategory.BGP_FLAP: [
                "BGP session flapping on transit link: route withdrawals every 15s.",
                "AS path prepending misconfiguration causing routing instability.",
                "Upstream ISP maintenance causing intermittent BGP route churn.",
            ],
        }
        description = random.choice(_DESCRIPTIONS[category])
        return delta, description, severity, "rule_based"

    # ────────────────────────── helpers ──────────────────────────────────────

    @staticmethod
    def _infer_category(
        description: str,
        delta: Tuple[float, float, float],
    ) -> FaultCategory:
        """Heuristically infer the FaultCategory from a description string."""
        desc_lower = description.lower()
        for cat in FaultCategory:
            if cat.value.replace("_", " ") in desc_lower:
                return cat
        # Fallback: infer from dominant delta dimension
        traffic_d, db_d, net_d = delta
        if abs(net_d) > abs(traffic_d) and abs(net_d) > abs(db_d):
            return FaultCategory.NETWORK_PARTITION
        if abs(db_d) > abs(traffic_d):
            return FaultCategory.DB_OVERLOAD
        return FaultCategory.TRAFFIC_SPIKE

    def __repr__(self) -> str:
        return self.describe()
