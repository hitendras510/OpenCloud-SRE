"""
evaluation/wandb_logger.py
===========================
Thin W&B logging wrapper.
Decouples the evaluator and trainer from direct wandb imports so the
codebase works even when wandb is not installed (CI / lightweight envs).

Usage
-----
    from evaluation.wandb_logger import WandbLogger
    wlog = WandbLogger(project="opencloud-sre-grpo", config={"epochs": 3})
    wlog.log({"train/loss": 0.42, "eval/total": 87.3})
    wlog.finish()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandbLogger:
    """
    Thin wrapper around wandb.  Falls back to a no-op when wandb is not
    installed or WANDB_API_KEY is not set.

    Parameters
    ----------
    project : str
        W&B project name.
    config : dict
        Hyperparameter config to log at run init.
    tags : list[str]
        Optional run tags.
    """

    def __init__(
        self,
        project: str = "opencloud-sre",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and _WANDB_AVAILABLE
        self._run = None

        if self._enabled:
            try:
                self._run = _wandb.init(
                    project=project,
                    config=config or {},
                    tags=tags or [],
                    reinit=True,
                )
                logger.info("W&B run initialised: %s", project)
            except Exception as exc:
                logger.warning("W&B init failed (%s) — logging disabled.", exc)
                self._enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics to the current run."""
        if not self._enabled or self._run is None:
            return
        try:
            _wandb.log(metrics, step=step)
        except Exception as exc:
            logger.debug("W&B log failed: %s", exc)

    def log_table(self, key: str, data: list, columns: list) -> None:
        """Log a W&B Table artifact (e.g. reward breakdown per episode)."""
        if not self._enabled or self._run is None:
            return
        try:
            table = _wandb.Table(columns=columns, data=data)
            _wandb.log({key: table})
        except Exception as exc:
            logger.debug("W&B table log failed: %s", exc)

    def summary(self, key: str, value: Any) -> None:
        """Set a run summary metric (shown on the W&B run overview page)."""
        if not self._enabled or self._run is None:
            return
        try:
            _wandb.run.summary[key] = value  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("W&B summary failed: %s", exc)

    def finish(self) -> None:
        """Mark the run as complete."""
        if self._enabled and self._run is not None:
            try:
                _wandb.finish()
            except Exception:
                pass

    @property
    def active(self) -> bool:
        """True if W&B logging is live."""
        return self._enabled and self._run is not None
