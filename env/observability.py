"""
env/observability.py
====================
Real-time observability layer for the OpenCloud-SRE environment.

Provides:
  - SnapshotCollector : rolling time-series buffer of metric snapshots
  - AlertEngine       : threshold-based alerting with deduplication
  - ObservabilityBus  : combines both into one object used by env/server.py

These are pure Python — no external dependencies beyond the standard library.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional


# ── Metric Snapshot ───────────────────────────────────────────────────────────

@dataclass
class MetricSnapshot:
    """One point-in-time observation of the datacenter state."""
    timestamp:            float          # unix epoch
    traffic_load:         float
    database_temperature: float
    network_health:       float
    slo_score:            float
    action_taken:         Optional[str]  = None
    episode_step:         int            = 0
    is_critical:          bool           = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":            datetime.fromtimestamp(
                                        self.timestamp, tz=timezone.utc
                                    ).isoformat(),
            "Traffic_Load":         round(self.traffic_load, 2),
            "Database_Temperature": round(self.database_temperature, 2),
            "Network_Health":       round(self.network_health, 2),
            "slo_score":            round(self.slo_score, 4),
            "action_taken":         self.action_taken,
            "episode_step":         self.episode_step,
            "is_critical":          self.is_critical,
        }


# ── Rolling time-series buffer ────────────────────────────────────────────────

class SnapshotCollector:
    """
    Rolling deque of MetricSnapshot objects.
    Used by the Streamlit UI to render live charts.

    Parameters
    ----------
    maxlen : int
        Maximum number of snapshots retained (default = 200 steps).
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._buffer: Deque[MetricSnapshot] = deque(maxlen=maxlen)

    def record(
        self,
        traffic: float,
        db_temp: float,
        net_health: float,
        slo: float,
        action: Optional[str] = None,
        step: int = 0,
    ) -> MetricSnapshot:
        snap = MetricSnapshot(
            timestamp=time.time(),
            traffic_load=traffic,
            database_temperature=db_temp,
            network_health=net_health,
            slo_score=slo,
            action_taken=action,
            episode_step=step,
            is_critical=(traffic >= 85.0 or db_temp >= 80.0 or net_health <= 30.0),
        )
        self._buffer.append(snap)
        return snap

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._buffer]

    def tail(self, n: int = 20) -> List[MetricSnapshot]:
        return list(self._buffer)[-n:]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ── Alert engine ──────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """A single threshold-breach alert."""
    level:       str       # "WARNING" | "CRITICAL"
    metric:      str
    value:       float
    threshold:   float
    message:     str
    timestamp:   float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level":     self.level,
            "metric":    self.metric,
            "value":     round(self.value, 2),
            "threshold": self.threshold,
            "message":   self.message,
            "timestamp": datetime.fromtimestamp(
                             self.timestamp, tz=timezone.utc
                         ).isoformat(),
            "acknowledged": self.acknowledged,
        }


_ALERT_RULES = [
    # (metric_key, threshold, operator, level, message_template)
    ("Traffic_Load",         85.0, "gte", "CRITICAL",
     "Traffic Load {val:.1f}% exceeds critical threshold of {thr:.0f}%."),
    ("Traffic_Load",         70.0, "gte", "WARNING",
     "Traffic Load {val:.1f}% approaching saturation."),
    ("Database_Temperature", 80.0, "gte", "CRITICAL",
     "DB Temperature {val:.1f}% is critical — risk of connection exhaustion."),
    ("Database_Temperature", 65.0, "gte", "WARNING",
     "DB Temperature {val:.1f}% is elevated."),
    ("Network_Health",       30.0, "lte", "CRITICAL",
     "Network Health {val:.1f}% is critical — partial partition likely."),
    ("Network_Health",       50.0, "lte", "WARNING",
     "Network Health {val:.1f}% is degraded."),
]


class AlertEngine:
    """
    Fires threshold-based alerts with cooldown deduplication.
    An alert of the same type won't re-fire within `cooldown_s` seconds.
    """

    def __init__(self, cooldown_s: float = 30.0) -> None:
        self._cooldown = cooldown_s
        self._last_fired: Dict[str, float] = {}
        self._active: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []

    def on_alert(self, fn: Callable[[Alert], None]) -> None:
        """Register a callback invoked on every new alert."""
        self._handlers.append(fn)

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate current metrics, fire new alerts, return those fired."""
        fired: List[Alert] = []
        now = time.time()
        for metric, threshold, op, level, tmpl in _ALERT_RULES:
            val = metrics.get(metric, 50.0)
            triggered = (
                (op == "gte" and val >= threshold) or
                (op == "lte" and val <= threshold)
            )
            key = f"{metric}:{level}"
            if triggered and (now - self._last_fired.get(key, 0)) > self._cooldown:
                alert = Alert(
                    level=level, metric=metric, value=val, threshold=threshold,
                    message=tmpl.format(val=val, thr=threshold),
                )
                self._last_fired[key] = now
                self._active.append(alert)
                fired.append(alert)
                for fn in self._handlers:
                    fn(alert)
        return fired

    def active_alerts(self) -> List[Dict[str, Any]]:
        # Trim old ones (> 5 min) for memory safety
        cutoff = time.time() - 300
        self._active = [a for a in self._active if a.timestamp > cutoff]
        return [a.to_dict() for a in self._active if not a.acknowledged]

    def acknowledge_all(self) -> None:
        for a in self._active:
            a.acknowledged = True


# ── Combined bus ──────────────────────────────────────────────────────────────

class ObservabilityBus:
    """
    Single object that exposes both SnapshotCollector and AlertEngine.
    Injected into the FastAPI server so /state and /alerts share one buffer.
    """

    def __init__(self, maxlen: int = 200, alert_cooldown_s: float = 30.0) -> None:
        self.snapshots = SnapshotCollector(maxlen=maxlen)
        self.alerts    = AlertEngine(cooldown_s=alert_cooldown_s)

    def ingest(
        self,
        metrics: Dict[str, float],
        action: Optional[str] = None,
        step: int = 0,
    ) -> None:
        """Record a snapshot and evaluate alert rules in one call."""
        slo = (
            (100 - metrics.get("Traffic_Load", 50))
            + (100 - metrics.get("Database_Temperature", 50))
            + metrics.get("Network_Health", 50)
        ) / 300.0
        self.snapshots.record(
            traffic=metrics.get("Traffic_Load", 50.0),
            db_temp=metrics.get("Database_Temperature", 50.0),
            net_health=metrics.get("Network_Health", 50.0),
            slo=slo,
            action=action,
            step=step,
        )
        self.alerts.evaluate(metrics)
