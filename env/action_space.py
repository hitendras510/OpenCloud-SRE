"""
env/action_space.py
====================
The Action Space — typed execution layer bridging LangGraph string-action
recommendations to concrete simulated infrastructure operations.

Each function returns an ActionResult (success bool + log string + delta hint).
All functions are registered in ACTION_REGISTRY and invoked via dispatch().
"""

from __future__ import annotations

import datetime
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────── result type ──────────────────────────────────

@dataclass
class ActionResult:
    """Immutable record of a single action execution attempt."""

    success: bool
    action_name: str
    target: str
    log_message: str
    delta_hint: Tuple[float, float, float]
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    latency_ms: float = field(default=0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_name": self.action_name,
            "target": self.target,
            "log_message": self.log_message,
            "delta_hint": list(self.delta_hint),
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"[{status}][{self.execution_id}] {self.action_name}→{self.target} "
            f"({self.latency_ms:.0f}ms) | {self.log_message}"
        )


# ─────────────────────────── audit log ────────────────────────────────────────

_AUDIT_LOG: List[ActionResult] = []


def get_audit_log() -> List[ActionResult]:
    return list(_AUDIT_LOG)


def clear_audit_log() -> None:
    _AUDIT_LOG.clear()


def _record(result: ActionResult) -> ActionResult:
    _AUDIT_LOG.append(result)
    logger.info("%s", result)
    return result


def _sim_latency(base_ms: float, jitter_ms: float = 20.0) -> float:
    return max(1.0, base_ms + random.gauss(0, jitter_ms))


# ──────────────────────────── action primitives ───────────────────────────────

def execute_throttle(
    target: str = "ingress-gateway",
    rate_limit_rps: int = 1000,
) -> ActionResult:
    """Apply rate-limiting to the ingress gateway of *target* service."""
    latency = _sim_latency(45.0)
    success = random.random() > 0.05
    if success:
        msg = (
            f"Rate-limit applied to '{target}': {rate_limit_rps} RPS cap. "
            "Traffic_Load dropping."
        )
        delta: Tuple[float, float, float] = (-20.0, -3.0, -2.0)
    else:
        msg = f"Throttle FAILED on '{target}': iptables rule conflict."
        delta = (0.0, 0.0, 0.0)
    return _record(ActionResult(
        success=success, action_name="throttle_traffic", target=target,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"rate_limit_rps": rate_limit_rps},
    ))


def execute_load_balance(
    target: str = "api-service",
    algorithm: str = "least_connections",
) -> ActionResult:
    """Redistribute traffic across healthy replicas of *target*."""
    latency = _sim_latency(30.0)
    success = random.random() > 0.08
    if success:
        msg = f"Load balancer updated for '{target}': algorithm={algorithm}."
        delta = (-12.0, +4.0, +5.0)
    else:
        msg = f"Load balance FAILED for '{target}': no healthy upstream."
        delta = (0.0, 0.0, -3.0)
    return _record(ActionResult(
        success=success, action_name="load_balance", target=target,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"algorithm": algorithm},
    ))


def execute_schema_failover(
    db_id: str = "primary-postgres",
    standby_id: str = "replica-01",
    promote_timeout_s: int = 30,
) -> ActionResult:
    """Promote *standby_id* to primary for the *db_id* cluster."""
    latency = _sim_latency(280.0, 50.0)
    success = random.random() > 0.10
    if success:
        msg = (
            f"Failover complete: '{db_id}' → '{standby_id}' promoted. "
            "DB_Temperature recovering."
        )
        delta = (-2.0, -25.0, +10.0)
    else:
        msg = (
            f"Failover ABORTED: '{standby_id}' not in sync "
            f"(lag > {promote_timeout_s}s)."
        )
        delta = (0.0, +5.0, -5.0)
    return _record(ActionResult(
        success=success, action_name="schema_failover", target=db_id,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"standby_id": standby_id, "promote_timeout_s": promote_timeout_s},
    ))


def execute_cache_flush(
    db_id: str = "redis-cluster",
    pattern: str = "*",
    flush_mode: str = "async",
) -> ActionResult:
    """Evict hot-key cache entries matching *pattern* on *db_id*."""
    latency = _sim_latency(80.0, 15.0)
    success = random.random() > 0.06
    if success:
        evicted = random.randint(10_000, 500_000)
        msg = (
            f"Cache flush on '{db_id}' (pattern='{pattern}'): "
            f"{evicted:,} keys evicted. DB_Temperature dropping."
        )
        delta = (+5.0, -18.0, +3.0)
    else:
        msg = f"Cache flush FAILED on '{db_id}': cluster unreachable."
        delta = (0.0, 0.0, 0.0)
    return _record(ActionResult(
        success=success, action_name="cache_flush", target=db_id,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"pattern": pattern, "flush_mode": flush_mode},
    ))


def execute_circuit_breaker(
    target: str = "downstream-service",
    threshold_pct: int = 50,
    half_open_timeout_s: int = 10,
) -> ActionResult:
    """Open the circuit breaker on *target* dependency."""
    latency = _sim_latency(15.0, 5.0)
    success = random.random() > 0.04
    if success:
        msg = (
            f"Circuit breaker OPEN on '{target}' "
            f"(error_threshold={threshold_pct}%). Traffic_Load plummeting."
        )
        delta = (-30.0, -5.0, -8.0)
    else:
        msg = f"Circuit breaker FAILED on '{target}': breaker service unreachable."
        delta = (0.0, 0.0, -5.0)
    return _record(ActionResult(
        success=success, action_name="circuit_breaker", target=target,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"threshold_pct": threshold_pct, "half_open_timeout_s": half_open_timeout_s},
    ))


def execute_restart_pod(
    service: str = "api-worker",
    strategy: str = "rolling",
    max_surge: int = 1,
) -> ActionResult:
    """Perform a rolling pod restart for *service*."""
    latency = _sim_latency(120.0, 30.0)
    success = random.random() > 0.07
    if success:
        pods = random.randint(2, 12)
        msg = (
            f"Rolling restart complete for '{service}': {pods} pods cycled "
            f"(strategy={strategy})."
        )
        delta = (-5.0, -15.0, -6.0)
    else:
        msg = (
            f"Pod restart STALLED for '{service}': PodDisruptionBudget violation. "
            "Scale out first."
        )
        delta = (0.0, +2.0, -3.0)
    return _record(ActionResult(
        success=success, action_name="restart_pods", target=service,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"strategy": strategy, "max_surge": max_surge},
    ))


def execute_scale_out(
    service: str = "api-service",
    additional_nodes: int = 3,
    node_type: str = "c5.xlarge",
) -> ActionResult:
    """Provision *additional_nodes* extra compute nodes for *service*."""
    latency = _sim_latency(350.0, 80.0)
    success = random.random() > 0.09
    if success:
        msg = (
            f"Scale-out complete for '{service}': +{additional_nodes}× "
            f"{node_type} nodes provisioned."
        )
        delta = (-15.0, -10.0, +15.0)
    else:
        msg = (
            f"Scale-out FAILED for '{service}': AZ capacity exhausted "
            f"for {node_type}."
        )
        delta = (0.0, 0.0, 0.0)
    return _record(ActionResult(
        success=success, action_name="scale_out", target=service,
        log_message=msg, delta_hint=delta, latency_ms=latency,
        metadata={"additional_nodes": additional_nodes, "node_type": node_type},
    ))


def execute_noop(reason: str = "observation") -> ActionResult:
    """Deliberate no-operation — state degrades naturally."""
    return _record(ActionResult(
        success=True, action_name="noop", target="system",
        log_message=f"No-op step (reason: {reason}). Passive degradation continues.",
        delta_hint=(+3.0, +2.0, -3.0), latency_ms=0.0,
        metadata={"reason": reason},
    ))


# ─────────────────────────── action registry ──────────────────────────────────

ACTION_REGISTRY: Dict[str, Callable[..., ActionResult]] = {
    "throttle_traffic": execute_throttle,
    "load_balance":     execute_load_balance,
    "schema_failover":  execute_schema_failover,
    "cache_flush":      execute_cache_flush,
    "circuit_breaker":  execute_circuit_breaker,
    "restart_pods":     execute_restart_pod,
    "scale_out":        execute_scale_out,
    "noop":             execute_noop,
}


def dispatch(
    action_name: str,
    target: Optional[str] = None,
    **kwargs: Any,
) -> ActionResult:
    """
    Execute the named action via the registry.

    This is the single entry-point for the executor node — it prevents agents
    from calling arbitrary code outside the registry.

    Parameters
    ----------
    action_name : str
        One of the keys in ACTION_REGISTRY.
    target : str, optional
        Override for the action's default target.
    **kwargs
        Additional keyword arguments forwarded to the action function.

    Raises
    ------
    ValueError
        If *action_name* is not registered.
    """
    if action_name not in ACTION_REGISTRY:
        raise ValueError(
            f"Unknown action '{action_name}'. "
            f"Registered: {sorted(ACTION_REGISTRY.keys())}"
        )
    fn = ACTION_REGISTRY[action_name]
    if target is not None:
        kwargs["target"] = target
    return fn(**kwargs)
