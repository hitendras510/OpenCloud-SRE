"""
env/environment.py
==================
OpenCloudEnv – A Gym-compatible environment that simulates a crashed enterprise
data-centre.  State is tracked as a :class:`CloudStateTensor` backed by
PyTorch tensors.

Action space  (discrete strings)
---------------------------------
  throttle_traffic    – Reduces traffic load; minor network health cost
  load_balance        – Spreads traffic; slight temp increase
  schema_failover     – Activates standby DB; drops DB temperature rapidly
  cache_flush         – Clears hot-cache; reduces DB temperature, small traffic spike
  circuit_breaker     – Hard-kills incoming connections; big traffic drop, health hit
  restart_pods        – Rolling restart; temperature down, short health dip
  scale_out           – Adds nodes; traffic & temp decrease, improves health
  noop                – No action; natural degradation continues

Each action is modelled as a stochastic PyTorch delta vector:
  Δ = base_delta + N(0, noise_std)
clamped to [−100, 100] before being applied to the state tensor.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from env.state_tensor import CloudStateTensor, METRIC_MIN, METRIC_MAX

# ─────────────────────────────── type aliases ─────────────────────────────────

Observation = Dict[str, float]
Info = Dict[str, Any]

# ────────────────────────────── action catalogue ──────────────────────────────
# Each entry: (traffic_delta, db_temp_delta, net_health_delta)
# Values are the *expected* delta – Gaussian noise is added at runtime.

_ACTION_BASE_DELTAS: Dict[str, Tuple[float, float, float]] = {
    "throttle_traffic":  (-20.0,  -3.0,  -2.0),
    "load_balance":      (-12.0,  +4.0,  +5.0),
    "schema_failover":   ( -2.0, -25.0, +10.0),
    "cache_flush":       ( +5.0, -18.0,  +3.0),
    "circuit_breaker":   (-30.0,  -5.0,  -8.0),
    "restart_pods":      ( -5.0, -15.0,  -6.0),
    "scale_out":         (-15.0, -10.0, +15.0),
    "noop":              ( +3.0,  +2.0,  -3.0),   # passive degradation
}

_NOISE_STD: float = 3.0    # Gaussian noise std applied to every delta

VALID_ACTIONS: List[str] = list(_ACTION_BASE_DELTAS.keys())

# ────────────────────────────────── environment ───────────────────────────────


class OpenCloudEnv:
    """
    A minimal Gym-style environment for OpenCloud-SRE.

    Compatible with the standard ``reset() / step()`` interface so it can be
    wrapped by Gymnasium or used directly from the LangGraph controllers.

    Parameters
    ----------
    seed:
        Optional random seed for reproducibility.
    max_steps:
        Episode length before auto-termination.
    crash_on_reset:
        If True the environment always starts in a crashed (degraded) state.
        Set to False for curriculum-learning warm-up scenarios.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human", "json"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 50,
        crash_on_reset: bool = True,
    ) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.crash_on_reset = crash_on_reset

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
            random.seed(seed)

        # Initialise a valid state so type-checkers are happy
        self.state: CloudStateTensor = CloudStateTensor.nominal()
        self._step_count: int = 0
        self._episode_rewards: List[float] = []
        self._action_history: List[str] = []

    # ──────────────────────────── Gym interface ───────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Observation, Info]:
        """
        Reset the environment to the start of a new episode.

        Returns
        -------
        observation:
            A dict with keys matching :data:`~env.state_tensor.METRIC_NAMES`.
        info:
            Diagnostic dict with ``episode_rewards`` from the previous episode.
        """
        if seed is not None:
            self._rng.manual_seed(seed)
            random.seed(seed)

        info: Info = {
            "previous_episode_rewards": list(self._episode_rewards),
            "previous_episode_length": self._step_count,
        }

        self.state = (
            CloudStateTensor.crashed() if self.crash_on_reset
            else CloudStateTensor.nominal()
        )

        # Inject controlled randomness into the crash state so episodes differ
        jitter = torch.randn(3, generator=self._rng) * 5.0
        self.state = self.state.apply_delta(jitter)

        self._step_count = 0
        self._episode_rewards = []
        self._action_history = []

        return self._observe(), info

    def step(
        self, action: str
    ) -> Tuple[Observation, float, bool, bool, Info]:
        """
        Execute one action and advance the environment by one time-step.

        Parameters
        ----------
        action:
            One of the strings in :data:`VALID_ACTIONS`.

        Returns
        -------
        observation:
            Updated metric dict after the action.
        reward:
            Scalar reward signal.
        terminated:
            True if the episode ended because the system is fully recovered
            or fully failed.
        truncated:
            True if ``max_steps`` was reached.
        info:
            Diagnostic information including step count and SLO score.

        Raises
        ------
        ValueError:
            If *action* is not in :data:`VALID_ACTIONS`.
        """
        if action not in _ACTION_BASE_DELTAS:
            raise ValueError(
                f"Unknown action '{action}'. Valid actions: {VALID_ACTIONS}"
            )

        base = torch.tensor(_ACTION_BASE_DELTAS[action], dtype=torch.float32)
        noise = torch.randn(3, generator=self._rng) * _NOISE_STD
        delta = torch.clamp(base + noise, -100.0, 100.0)

        self.state = self.state.apply_delta(delta)
        self._step_count += 1
        self._action_history.append(action)

        reward = self.state.compute_reward()
        self._episode_rewards.append(reward)

        slo = self.state.slo_score()
        terminated = slo >= 0.95 or (self.state.is_critical() and slo < 0.05)
        truncated = self._step_count >= self.max_steps

        info: Info = {
            "step": self._step_count,
            "action": action,
            "delta": delta.tolist(),
            "slo_score": slo,
            "is_critical": self.state.is_critical(),
            "named_metrics": self.state.named_metrics(),
        }

        return self._observe(), reward, terminated, truncated, info

    # ──────────────────────── helper / render ─────────────────────────────────

    def _observe(self) -> Observation:
        """Return a plain-dict observation from the current state tensor."""
        return self.state.named_metrics()

    def state(self) -> Dict[str, Any]:
        """
        OpenEnv-standard state method.
        Provides access to current episode metadata and tracking variables.
        """
        return {
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "is_critical": self.state.is_critical(),
            "slo_score": self.state.slo_score(),
            "action_history": list(self._action_history),
            "episode_rewards": list(self._episode_rewards),
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """Print or return a human-readable snapshot of the current state."""
        snapshot = (
            f"[Step {self._step_count:03d}] "
            f"Traffic={self.state.traffic_load:5.1f} | "
            f"DB_Temp={self.state.database_temperature:5.1f} | "
            f"Net_Health={self.state.network_health:5.1f} | "
            f"SLO={self.state.slo_score():.3f} | "
            f"Critical={self.state.is_critical()}"
        )
        if mode == "human":
            print(snapshot)
            return None
        return snapshot

    def action_space_sample(self) -> str:
        """Return a uniformly random valid action."""
        return random.choice(VALID_ACTIONS)

    def inject_fault(self, fault_type: str = "traffic_spike") -> None:
        """
        Manually inject a fault into the running environment.

        Useful for testing controller resilience mid-episode.

        Parameters
        ----------
        fault_type:
            One of ``"traffic_spike"``, ``"db_overload"``, ``"network_partition"``,
            or ``"cascade_failure"``.
        """
        _FAULT_DELTAS: Dict[str, Tuple[float, float, float]] = {
            "traffic_spike":      (+40.0,  +5.0, -10.0),
            "db_overload":        ( +5.0, +45.0,  -5.0),
            "network_partition":  ( +10.0,  +3.0, -50.0),
            "cascade_failure":    (+30.0, +30.0, -40.0),
        }
        if fault_type not in _FAULT_DELTAS:
            raise ValueError(
                f"Unknown fault '{fault_type}'. "
                f"Valid: {list(_FAULT_DELTAS.keys())}"
            )
        delta = torch.tensor(_FAULT_DELTAS[fault_type], dtype=torch.float32)
        self.state = self.state.apply_delta(delta)

    def get_history(self) -> Dict[str, Any]:
        """Return episode history for logging and dataset generation."""
        return {
            "actions": list(self._action_history),
            "episode_rewards": list(self._episode_rewards),
            "final_state": self.state.as_list(),
            "final_slo_score": self.state.slo_score(),
            "steps": self._step_count,
        }

    def __repr__(self) -> str:
        return (
            f"OpenCloudEnv(step={self._step_count}, "
            f"state={self.state!r})"
        )
