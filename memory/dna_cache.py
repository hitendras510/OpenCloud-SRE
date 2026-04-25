"""
memory/dna_cache.py
====================
Kill Shot #3 – DNA 2.0 Knowledge Distillation

Implements "Memory Consolidation" — the mechanism that makes OpenCloud-SRE
smarter on every successfully resolved incident.

How it works
------------
  1. An incident is resolved via the SLOW PATH (ChatOps full reasoning).
  2. ``consolidate_slow_path_resolution()`` is called with:
       - the initial crash state vector [Traffic, DB_Temp, Net_Health]
       - the action that fixed it
  3. The function hashes the vector and writes it to the shared DNAMemory index.
  4. If the *exact* same crash recurs, the FAST PATH now catches it —
     skipping all LLM calls (O(1) cache hit).

This is the "Reflex over Reasoning" loop:
  Slow Path (1st encounter) → Memory Consolidation → Fast Path (all future encounters)

Usage
-----
  from memory.dna_cache import consolidate_slow_path_resolution, get_cache_stats

  # After a successful Slow Path resolution:
  consolidate_slow_path_resolution(
      state_vec=[95.0, 88.0, 12.0],
      successful_action="circuit_breaker",
      source_path="slow_path",
  )

  # Check how many incidents have been distilled:
  stats = get_cache_stats()
  print(stats["distilled_count"])
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ── path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dna_memory import DNAMemory, MatchConfidence

logger = logging.getLogger(__name__)

# ── persistence paths ─────────────────────────────────────────────────────────
_DISTILLED_LOG = ROOT / "data" / "distilled_incidents.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Module-level shared DNAMemory instance.
# The LangGraph graph uses utils.dna_memory.DNAMemory() directly.
# This module wraps it with a consolidation layer so we can inject new
# learnings at runtime without rebuilding the full graph.
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_DNA: Optional[DNAMemory] = None


def get_shared_dna() -> DNAMemory:
    """
    Return the module-level singleton DNAMemory instance.

    On first call it is initialised with the standard 20 seed incidents
    PLUS any previously distilled slow-path resolutions loaded from disk.
    """
    global _SHARED_DNA
    if _SHARED_DNA is None:
        _SHARED_DNA = DNAMemory()
        _load_distilled_incidents(_SHARED_DNA)
    return _SHARED_DNA


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation event dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsolidationEvent:
    """Record of a single slow-path resolution that was distilled into DNA memory."""
    state_vec: List[float]            # [Traffic, DB_Temp, Network_Health]
    successful_action: str            # e.g. "circuit_breaker"
    source_path: str                  # "slow_path" | "chatops"
    distilled_at: str                 # ISO-8601 timestamp
    cache_key: str                    # SHA-1 hex digest (from DNAMemory._hash)
    match_confidence_before: str      # confidence BEFORE distillation
    match_distance_before: float      # L2 distance BEFORE distillation


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def consolidate_slow_path_resolution(
    state_vec: List[float],
    successful_action: str,
    source_path: str = "slow_path",
    memory: Optional[DNAMemory] = None,
) -> ConsolidationEvent:
    """
    Distill a successful Slow Path resolution into the DNA Memory FAISS index.

    This function ONLY fires when an incident was resolved via the Slow Path
    (ChatOps deep negotiation). Do NOT call it for Fast or Middle Path resolutions
    — the system already "knew" those answers.

    Parameters
    ----------
    state_vec:
        The initial crashed state ``[Traffic_Load, DB_Temperature, Network_Health]``
        at the *start* of the incident (before any remediation was applied).
    successful_action:
        The action string that successfully resolved the incident
        (must be one of ``env.environment.VALID_ACTIONS``).
    source_path:
        For logging. Should always be ``"slow_path"`` or ``"chatops"``.
    memory:
        Optionally inject a specific DNAMemory instance (e.g. for testing).
        Defaults to the module-level shared singleton.

    Returns
    -------
    ConsolidationEvent
        A record of what was distilled (also persisted to JSONL log).

    Raises
    ------
    ValueError
        If ``state_vec`` is not a 3-element list.
    """
    if len(state_vec) != 3:
        raise ValueError(
            f"state_vec must have exactly 3 elements [Traffic, DB_Temp, Net_Health], "
            f"got {len(state_vec)}"
        )

    if source_path not in ("slow_path", "chatops"):
        logger.warning(
            "consolidate_slow_path_resolution called with source_path='%s'. "
            "Memory consolidation is only intended for slow-path incidents.",
            source_path,
        )

    mem = memory or get_shared_dna()

    # ── 1. Measure match confidence BEFORE distillation ───────────────────────
    pre_hit = mem.query(state_vec)
    conf_before = pre_hit.confidence.value
    dist_before = pre_hit.distance

    # ── 2. Add to FAISS index ─────────────────────────────────────────────────
    mem.add_incident(state_vec, successful_action)

    # ── 3. Verify: re-query to confirm a Fast Path hit was created ────────────
    post_hit = mem.query(state_vec)
    if post_hit.confidence != MatchConfidence.HIGH:
        logger.warning(
            "Distillation may have failed: post-insert confidence=%s (expected HIGH). "
            "vec=%s action=%s",
            post_hit.confidence.value, state_vec, successful_action,
        )
    else:
        logger.info(
            "✅ DNA Distillation SUCCESS: vec=%s → action=%s  "
            "(was %s dist=%.2f, now HIGH dist=%.2f)",
            state_vec, successful_action,
            conf_before, dist_before, post_hit.distance,
        )

    # ── 4. Build the event record ─────────────────────────────────────────────
    cache_key = post_hit.cache_key
    event = ConsolidationEvent(
        state_vec=list(state_vec),
        successful_action=successful_action,
        source_path=source_path,
        distilled_at=_iso_now(),
        cache_key=cache_key,
        match_confidence_before=conf_before,
        match_distance_before=round(dist_before, 4),
    )

    # ── 5. Persist to JSONL log for audit and cold-start replay ──────────────
    _persist_event(event)

    return event


def query_dna(
    state_vec: List[float],
    memory: Optional[DNAMemory] = None,
) -> Dict:
    """
    Convenience wrapper: query the shared DNA memory and return a plain dict.

    Parameters
    ----------
    state_vec:
        ``[Traffic_Load, DB_Temperature, Network_Health]``

    Returns
    -------
    dict with keys: confidence, distance, matched_action, is_fast_path
    """
    mem = memory or get_shared_dna()
    hit = mem.query(state_vec)
    return {
        "confidence": hit.confidence.value,
        "distance": round(hit.distance, 4),
        "matched_action": hit.matched_action,
        "is_fast_path": hit.is_fast_path(),
        "cache_key": hit.cache_key,
    }


def get_cache_stats(memory: Optional[DNAMemory] = None) -> Dict:
    """
    Return a summary of the current DNA Memory state.

    Returns
    -------
    dict with:
        total_vectors   – number of incidents in the FAISS index
        seed_count      – 20 hardcoded incidents (always present)
        distilled_count – slow-path incidents learned at runtime
        log_path        – absolute path to the JSONL audit log
    """
    mem = memory or get_shared_dna()
    total = len(mem._vectors)
    seed_count = 20   # matches SEED_INCIDENTS in utils/dna_memory.py
    distilled = max(0, total - seed_count)

    return {
        "total_vectors": total,
        "seed_count": seed_count,
        "distilled_count": distilled,
        "backend": "faiss" if hasattr(mem._index, "search") else "numpy",
        "log_path": str(_DISTILLED_LOG),
        "description": mem.describe(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iso_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _persist_event(event: ConsolidationEvent) -> None:
    """Append the ConsolidationEvent to the JSONL audit log."""
    _DISTILLED_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_DISTILLED_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")
    except OSError as exc:
        logger.error("Failed to persist distillation event: %s", exc)


def _load_distilled_incidents(mem: DNAMemory) -> None:
    """
    Replay persisted slow-path resolutions into *mem* on cold-start.

    This ensures that incidents learned in previous sessions are not lost
    when the process restarts.
    """
    if not _DISTILLED_LOG.exists():
        return

    loaded = 0
    try:
        with open(_DISTILLED_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    vec = rec.get("state_vec", [])
                    action = rec.get("successful_action", "noop")
                    if len(vec) == 3:
                        mem.add_incident(vec, action)
                        loaded += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Skipping malformed distillation record: %s", e)
    except OSError as exc:
        logger.error("Could not load distilled incidents: %s", exc)
        return

    if loaded:
        logger.info(
            "DNA cold-start: replayed %d distilled incident(s) from %s",
            loaded, _DISTILLED_LOG,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    mem = DNAMemory()   # fresh instance for isolated test

    print("\n=== DNA 2.0 Distillation Smoke Test ===\n")

    # A brand-new crash state NOT in the seed data
    novel_state = [72.0, 91.0, 18.0]  # high DB + bad network
    pre = query_dna(novel_state, memory=mem)
    print(f"BEFORE distillation: {pre}")

    # Simulate: this was resolved by 'restart_pods' via the Slow Path
    event = consolidate_slow_path_resolution(
        state_vec=novel_state,
        successful_action="restart_pods",
        source_path="slow_path",
        memory=mem,
    )
    print(f"\nDistillation event recorded: {asdict(event)}")

    post = query_dna(novel_state, memory=mem)
    print(f"\nAFTER distillation: {post}")

    assert post["is_fast_path"], "Expected a Fast Path hit after distillation!"
    print("\n[PASS] DNA 2.0 distillation verified -- Fast Path cache hit confirmed.")

    stats = get_cache_stats(memory=mem)
    print(f"\nCache stats: {stats}")
