"""
env/fault_injection.py
=======================
ChaosMonkey – Pure stochastic fault injection engine for OpenCloud-SRE.

Completely vendor-agnostic: no LLM calls. All fault selection and delta
calculation is done with weighted-random Python and PyTorch arithmetic,
making it fast enough to run inside a tight RL simulation loop.

Fault Taxonomy
--------------
  TRAFFIC_SPIKE       – Flash-crowd or bot-attack ingress surge
  DB_OVERLOAD         – Hot-key storm or runaway query filling the DB queue
  NETWORK_PARTITION   – Partial link failure or BGP route flap
  CASCADE_FAILURE     – Correlated multi-metric degradation
  MEMORY_LEAK_OOM     – Pod OOM kill triggering restart loop
  REPLICA_LAG         – Replication delay causing read-replica fan-out
  HOT_KEY_STORM       – Concentrated cache misses (DB + traffic)
  BGP_FLAP            – Routing table churn causing intermittent packet loss
"""

from __future__ import annotations

import datetime
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from env.state_tensor import CloudStateTensor

logger = logging.getLogger(__name__)


# ─────────────────────────── fault taxonomy ───────────────────────────────────

class FaultCategory(str, Enum):
    TRAFFIC_SPIKE     = "traffic_spike"
    DB_OVERLOAD       = "db_overload"
    NETWORK_PARTITION = "network_partition"
    CASCADE_FAILURE   = "cascade_failure"
    MEMORY_LEAK_OOM   = "memory_leak_oom"
    REPLICA_LAG       = "replica_lag"
    HOT_KEY_STORM     = "hot_key_storm"
    BGP_FLAP          = "bgp_flap"


# Delta ranges: (traffic_range, db_temp_range, net_health_range)
# All values are *additive* to the current tensor (positive = load increases / health degrades).
_FAULT_DELTA_RANGES: Dict[FaultCategory, Tuple[Tuple, Tuple, Tuple]] = {
    FaultCategory.TRAFFIC_SPIKE:     ((+35, +55), ( -2,  +5), ( -8,  +2)),
    FaultCategory.DB_OVERLOAD:       (( +5, +15), (+35, +55), ( -5,  +3)),
    FaultCategory.NETWORK_PARTITION: (( +8, +20), ( +2,  +8), (-45, -30)),
    FaultCategory.CASCADE_FAILURE:   ((+25, +40), (+25, +40), (-35, -20)),
    FaultCategory.MEMORY_LEAK_OOM:   ((+15, +30), (+10, +20), (-15,  -5)),
    FaultCategory.REPLICA_LAG:       (( +5, +12), (+20, +38), ( -5,  +5)),
    FaultCategory.HOT_KEY_STORM:     ((+20, +35), (+25, +40), ( -5,  +5)),
    FaultCategory.BGP_FLAP:          (( +5, +15), ( +2,  +8), (-40, -20)),
}

# Incident description pool (fully deterministic — no LLM needed)
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
        "Read replica 5 s behind primary: read queries fanning out to primary.",
        "Replication slot bloat causing WAL disk fill on standby.",
        "Binary log delay exceeding 10 s: replica unhealthy, read traffic failing.",
    ],
    FaultCategory.HOT_KEY_STORM: [
        "Single product-page key receiving 80% of all cache reads.",
        "Thundering herd on expired OAuth token key: cache miss storm.",
        "Viral content causing concentrated hot-key overload on Redis shard 3.",
    ],
    FaultCategory.BGP_FLAP: [
        "BGP session flapping on transit link: route withdrawals every 15 s.",
        "AS path prepending misconfiguration causing routing instability.",
        "Upstream ISP maintenance causing intermittent BGP route churn.",
    ],
}


# ─────────────────────────── data structures ──────────────────────────────────

@dataclass
class FaultEvent:
    """Record of a single chaos injection."""
    fault_id:      str
    category:      FaultCategory
    description:   str
    delta_applied: Tuple[float, float, float]
    state_before:  List[float]
    state_after:   List[float]
    timestamp:     str
    severity:      str   # "low" | "medium" | "high" | "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id":      self.fault_id,
            "category":      self.category.value,
            "description":   self.description,
            "delta_applied": list(self.delta_applied),
            "state_before":  self.state_before,
            "state_after":   self.state_after,
            "timestamp":     self.timestamp,
            "severity":      self.severity,
        }

    def __str__(self) -> str:
        return (
            f"[ChaosMonkey/{self.severity.upper()}] {self.category.value} "
            f"| {self.description[:80]} "
            f"| Δ={[round(d, 1) for d in self.delta_applied]}"
        )


# ─────────────────────────── ChaosMonkey ──────────────────────────────────────

class ChaosMonkey:
    """
    Pure stochastic chaos injection engine — no LLM, no external calls.

    Fault category selection is weighted by the current state tensor so
    that already-stressed dimensions are more likely to degrade further,
    producing realistic correlated failure scenarios.

    Parameters
    ----------
    seed : Optional[int]
        Set for reproducible episode replays.

    Example
    -------
    >>> monkey = ChaosMonkey(seed=42)
    >>> state  = CloudStateTensor.nominal()
    >>> event, new_state = monkey.inject(state)
    >>> print(event)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._fault_log: List[FaultEvent] = []
        self._counter:   int              = 0
        if seed is not None:
            random.seed(seed)

    # ── public API ────────────────────────────────────────────────────────────

    def inject(
        self,
        state: CloudStateTensor,
        category_override: Optional[FaultCategory] = None,
    ) -> Tuple[FaultEvent, CloudStateTensor]:
        """
        Inject one stochastic fault into *state*.

        Parameters
        ----------
        state : CloudStateTensor
            Current datacenter state.
        category_override : Optional[FaultCategory]
            Pin to a specific fault category (useful for curriculum testing).

        Returns
        -------
        Tuple[FaultEvent, CloudStateTensor]
            The fault record and the mutated state.
        """
        self._counter += 1
        fault_id = f"chaos-{self._counter:04d}"
        before   = state.as_list()

        category = category_override or self._select_category(state)
        delta, description, severity = self._sample_fault(state, category)

        delta_t   = torch.tensor(list(delta), dtype=torch.float32)
        new_state = state.apply_delta(delta_t)

        event = FaultEvent(
            fault_id=fault_id,
            category=category,
            description=description,
            delta_applied=delta,
            state_before=before,
            state_after=new_state.as_list(),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            severity=severity,
        )
        self._fault_log.append(event)
        logger.info("%s", event)
        return event, new_state

    def inject_into_env(
        self,
        env: Any,  # OpenCloudEnv — avoids circular import
        category_override: Optional[FaultCategory] = None,
    ) -> FaultEvent:
        """Inject a fault directly into an OpenCloudEnv, mutating env.state."""
        event, new_state = self.inject(env.state, category_override)
        env.state = new_state
        return event

    def get_log(self)   -> List[FaultEvent]: return list(self._fault_log)
    def clear_log(self) -> None:
        self._fault_log.clear(); self._counter = 0

    def __repr__(self) -> str:
        return f"ChaosMonkey(injections={self._counter}, mode=stochastic)"

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _select_category(state: CloudStateTensor) -> FaultCategory:
        """
        Weighted random category selection.
        Stressed metrics increase the probability of related faults.
        """
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
        if state.traffic_load > 70:
            weights[FaultCategory.TRAFFIC_SPIKE]   *= 2.0
            weights[FaultCategory.CASCADE_FAILURE] *= 1.5
        if state.database_temperature > 70:
            weights[FaultCategory.DB_OVERLOAD]   *= 2.0
            weights[FaultCategory.HOT_KEY_STORM] *= 1.8
        if state.network_health < 40:
            weights[FaultCategory.NETWORK_PARTITION] *= 2.0
            weights[FaultCategory.BGP_FLAP]          *= 1.8

        cats = list(weights.keys())
        wts  = [weights[c] for c in cats]
        return random.choices(cats, weights=wts, k=1)[0]

    @staticmethod
    def _sample_fault(
        state: CloudStateTensor,
        category: FaultCategory,
    ) -> Tuple[Tuple[float, float, float], str, str]:
        """
        Sample a random delta from the category's range, pick a description,
        and compute severity from delta magnitude.

        Returns (delta_tuple, description, severity_str).
        """
        ranges = _FAULT_DELTA_RANGES[category]

        # Add Gaussian jitter on top of the uniform range for realism
        base_delta = tuple(random.uniform(*r) for r in ranges)
        jitter     = tuple(random.gauss(0, abs(d) * 0.05) for d in base_delta)
        delta: Tuple[float, float, float] = tuple(  # type: ignore[assignment]
            b + j for b, j in zip(base_delta, jitter)
        )

        max_mag = max(abs(d) for d in delta)
        severity = (
            "critical" if max_mag > 40
            else "high"   if max_mag > 25
            else "medium" if max_mag > 10
            else "low"
        )

        description = random.choice(_DESCRIPTIONS[category])
        return delta, description, severity
