"""
env/state_tensor.py
====================
Defines the canonical PyTorch state representation for the OpenCloud-SRE
simulated data-centre environment.

The state is a rank-1 float32 tensor of shape (3,):
  Index 0 – Traffic_Load      (0.0 – 100.0)  higher  ⟹ more congested
  Index 1 – Database_Temperature (0.0 – 100.0) higher ⟹ more overloaded
  Index 2 – Network_Health    (0.0 – 100.0)  higher  ⟹ healthier

All values are clamped to [0, 100] after every mutation so downstream
consumers can safely treat the tensor as normalised without an extra pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, Tuple

import torch

# ─────────────────────────────────── constants ────────────────────────────────

METRIC_NAMES: Final[List[str]] = [
    "Traffic_Load",
    "Database_Temperature",
    "Network_Health",
]

METRIC_MIN: Final[float] = 0.0
METRIC_MAX: Final[float] = 100.0

# "Healthy" reference point used by the reward signal and FAISS memory
NOMINAL_STATE: Final[List[float]] = [20.0, 30.0, 90.0]

# Critical thresholds – crossing any of these triggers an SLO breach
CRITICAL_THRESHOLDS: Final[dict[str, float]] = {
    "Traffic_Load": 85.0,
    "Database_Temperature": 80.0,
    "Network_Health": 30.0,        # health < 30 is critical (inverted)
}

# ─────────────────────────────────── dataclass ────────────────────────────────


@dataclass
class CloudStateTensor:
    """
    A thin, typed wrapper around a :class:`torch.Tensor` that gives each
    dimension a human-readable name and enforces range clamping.

    Example
    -------
    >>> s = CloudStateTensor()
    >>> s.traffic_load = 95.0          # simulate a traffic spike
    >>> vec = s.as_tensor()            # retrieve the underlying tensor
    >>> print(s.is_critical())
    True
    """

    traffic_load: float = field(default=20.0)
    database_temperature: float = field(default=30.0)
    network_health: float = field(default=90.0)

    # Internal tensor – always kept in sync with the scalar fields
    _tensor: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sync_tensor()

    # ──────────────────────────────── private ─────────────────────────────────

    def _sync_tensor(self) -> None:
        """Rebuild the internal tensor from the three scalar fields."""
        raw = torch.tensor(
            [self.traffic_load, self.database_temperature, self.network_health],
            dtype=torch.float32,
        )
        self._tensor = torch.clamp(raw, METRIC_MIN, METRIC_MAX)
        # Clamp may have adjusted values – write back to scalars
        self.traffic_load = float(self._tensor[0])
        self.database_temperature = float(self._tensor[1])
        self.network_health = float(self._tensor[2])

    # ─────────────────────────────── public API ───────────────────────────────

    def as_tensor(self) -> torch.Tensor:
        """Return the current state as a float32 tensor of shape (3,)."""
        self._sync_tensor()
        return self._tensor.clone()

    def as_list(self) -> List[float]:
        """Return the current state as a plain Python list of three floats."""
        self._sync_tensor()
        return self._tensor.tolist()

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "CloudStateTensor":
        """Construct a :class:`CloudStateTensor` from an existing tensor.

        Parameters
        ----------
        t:
            A float32 tensor of shape (3,).  Values will be clamped.
        """
        if t.shape != (3,):
            raise ValueError(
                f"Expected tensor of shape (3,), got {tuple(t.shape)}"
            )
        values = t.float().tolist()
        return cls(
            traffic_load=values[0],
            database_temperature=values[1],
            network_health=values[2],
        )

    @classmethod
    def nominal(cls) -> "CloudStateTensor":
        """Return a healthy, baseline state tensor."""
        return cls(*NOMINAL_STATE)

    @classmethod
    def crashed(cls) -> "CloudStateTensor":
        """Return a maximally degraded state tensor (simulates a DC crash)."""
        return cls(
            traffic_load=98.0,
            database_temperature=95.0,
            network_health=5.0,
        )

    def apply_delta(self, delta: torch.Tensor) -> "CloudStateTensor":
        """
        Apply a (3,) delta tensor to the current state, returning a *new*
        :class:`CloudStateTensor`.  The original instance is not mutated.

        Parameters
        ----------
        delta:
            A float32 tensor of shape (3,) representing the change to apply.
        """
        if delta.shape != (3,):
            raise ValueError(
                f"Delta must have shape (3,), got {tuple(delta.shape)}"
            )
        new_tensor = self._tensor + delta.float()
        new_tensor = torch.clamp(new_tensor, METRIC_MIN, METRIC_MAX)
        return CloudStateTensor.from_tensor(new_tensor)

    def is_critical(self) -> bool:
        """Return True if *any* metric has breached its SLO threshold."""
        return (
            self.traffic_load >= CRITICAL_THRESHOLDS["Traffic_Load"]
            or self.database_temperature >= CRITICAL_THRESHOLDS["Database_Temperature"]
            or self.network_health <= CRITICAL_THRESHOLDS["Network_Health"]
        )

    def slo_score(self) -> float:
        """
        A scalar [0, 1] representing overall system health.
        1.0 means perfect; 0.0 means fully degraded.

        Computed as the normalised inverse distance from the nominal state.
        """
        nominal = torch.tensor(NOMINAL_STATE, dtype=torch.float32)
        dist = torch.norm(self._tensor - nominal).item()
        max_dist = torch.norm(
            torch.tensor([METRIC_MAX, METRIC_MAX, METRIC_MIN], dtype=torch.float32)
            - nominal
        ).item()
        return float(max(0.0, 1.0 - dist / max_dist))

    def named_metrics(self) -> dict[str, float]:
        """Return a dict mapping metric name → current value."""
        return {
            "Traffic_Load": self.traffic_load,
            "Database_Temperature": self.database_temperature,
            "Network_Health": self.network_health,
        }

    def compute_reward(self) -> float:
        """
        Scalar reward signal used by the RL training loop.

        Positive reward for healthy state, negative for SLO breach.
        """
        score = self.slo_score()
        breach_penalty = -10.0 if self.is_critical() else 0.0
        return score * 10.0 + breach_penalty

    def __repr__(self) -> str:
        return (
            f"CloudStateTensor("
            f"traffic_load={self.traffic_load:.1f}, "
            f"db_temp={self.database_temperature:.1f}, "
            f"net_health={self.network_health:.1f}, "
            f"slo_score={self.slo_score():.3f})"
        )
