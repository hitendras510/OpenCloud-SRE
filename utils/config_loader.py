"""
utils/config_loader.py
=======================
Loads training/config.yaml and merges with environment variable overrides.
Returns a plain dict so every subsystem can read config without coupling
to a specific config framework.

Usage
-----
    from utils.config_loader import load_config
    cfg = load_config()                     # reads training/config.yaml
    cfg = load_config("path/to/other.yaml") # custom path
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "training" / "config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the YAML config file and apply env-variable overrides.

    Env var override convention:
      OPENCLOUD_<SECTION>_<KEY>=value
      e.g. OPENCLOUD_TRAINING_EPOCHS=5 → cfg["training"]["epochs"] = 5

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. Install: pip install pyyaml"
        )

    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Apply env-var overrides
    prefix = "OPENCLOUD_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) == 2:
            section, sub_key = parts
            if section not in cfg:
                cfg[section] = {}
            # Attempt type coercion
            try:
                parsed: Any = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = val
            cfg[section][sub_key] = parsed

    return cfg
