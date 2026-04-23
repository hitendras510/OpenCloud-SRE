"""
utils/dna_memory.py
====================
DNA Memory System – a local FAISS-backed incident memory store.

Architecture
------------
  • A flat L2 FAISS index stores past incident vectors (3-D float32).
  • Each vector represents a (Traffic_Load, DB_Temperature, Network_Health)
    tuple at the moment a known-good remediation was applied.
  • At query time the L2 distance to the nearest neighbour determines
    whether the current incident is a "High Match" (fast-path cache hit)
    or a "Low Match" (requires full LangGraph reasoning).

The index is pre-seeded with 20 hardcoded historical incidents so the system
can function entirely offline during local development / hackathon demos.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────── optional FAISS ──────────────────────────────
try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu not installed – DNAMemory will fall back to brute-force "
        "numpy L2 search. Install with: pip install faiss-cpu"
    )

# ─────────────────────────────────── types ────────────────────────────────────


class MatchConfidence(str, Enum):
    HIGH = "High Match"    # L2 distance < HIGH_MATCH_THRESHOLD  → fast path
    MEDIUM = "Medium Match"  # distance < MEDIUM_MATCH_THRESHOLD → normal path
    LOW = "Low Match"      # distance >= MEDIUM_MATCH_THRESHOLD  → slow path


HIGH_MATCH_THRESHOLD: float = 8.0    # ~8-unit L2 radius in [0,100]³
MEDIUM_MATCH_THRESHOLD: float = 20.0

VECTOR_DIM: int = 3


@dataclass
class MemoryHit:
    """Result of a single FAISS lookup."""
    confidence: MatchConfidence
    distance: float
    matched_vector: List[float]
    matched_action: str
    cache_key: str

    def is_fast_path(self) -> bool:
        return self.confidence == MatchConfidence.HIGH

    def to_dict(self) -> Dict:
        return {
            "confidence": self.confidence.value,
            "distance": round(self.distance, 4),
            "matched_vector": self.matched_vector,
            "recommended_action": self.matched_action,
            "cache_key": self.cache_key,
        }


# ─────────────── hardcoded past incident vectors + remediation ────────────────
# Format: ([Traffic_Load, DB_Temperature, Network_Health], recommended_action)

SEED_INCIDENTS: List[Tuple[List[float], str]] = [
    # Traffic spikes
    ([92.0, 35.0, 70.0], "throttle_traffic"),
    ([88.0, 40.0, 65.0], "throttle_traffic"),
    ([95.0, 30.0, 72.0], "circuit_breaker"),
    ([97.0, 28.0, 60.0], "circuit_breaker"),
    # DB overheating
    ([30.0, 90.0, 75.0], "schema_failover"),
    ([25.0, 88.0, 80.0], "schema_failover"),
    ([40.0, 82.0, 68.0], "cache_flush"),
    ([35.0, 85.0, 70.0], "cache_flush"),
    # Network degradation
    ([45.0, 40.0, 15.0], "restart_pods"),
    ([50.0, 38.0, 10.0], "restart_pods"),
    ([60.0, 42.0, 20.0], "scale_out"),
    # Combined stress
    ([80.0, 75.0, 30.0], "scale_out"),
    ([85.0, 80.0, 25.0], "schema_failover"),
    ([90.0, 85.0, 15.0], "circuit_breaker"),
    # Near-nominal (light load balancing)
    ([55.0, 50.0, 55.0], "load_balance"),
    ([60.0, 55.0, 50.0], "load_balance"),
    ([65.0, 60.0, 45.0], "load_balance"),
    # Partial recovery states
    ([42.0, 62.0, 62.0], "cache_flush"),
    ([38.0, 58.0, 65.0], "noop"),
    ([22.0, 32.0, 88.0], "noop"),   # essentially healthy – no action
]


# ───────────────────────────────── DNA Memory ─────────────────────────────────


class DNAMemory:
    """
    FAISS-backed incident memory for the OpenCloud-SRE Shadow Consensus Layer.

    Usage
    -----
    >>> mem = DNAMemory()
    >>> hit = mem.query([92.0, 36.0, 68.0])
    >>> print(hit.confidence)          # MatchConfidence.HIGH
    >>> print(hit.matched_action)      # 'throttle_traffic'
    """

    def __init__(self, extra_incidents: Optional[List[Tuple[List[float], str]]] = None) -> None:
        self._actions: List[str] = []
        self._vectors: List[List[float]] = []
        self._index: Optional[object] = None

        # Populate from seed + any caller-supplied extras
        all_incidents = list(SEED_INCIDENTS)
        if extra_incidents:
            all_incidents.extend(extra_incidents)

        for vec, action in all_incidents:
            self._add(vec, action)

        self._build_index()

    # ─────────────────────────── internal helpers ─────────────────────────────

    def _add(self, vector: List[float], action: str) -> None:
        """Append a single incident vector without rebuilding the index."""
        if len(vector) != VECTOR_DIM:
            raise ValueError(f"Vector must have {VECTOR_DIM} dims, got {len(vector)}")
        self._vectors.append(list(vector))
        self._actions.append(action)

    def _build_index(self) -> None:
        """(Re)build the FAISS index from the current vector store."""
        mat = np.array(self._vectors, dtype=np.float32)

        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatL2(VECTOR_DIM)
            index.add(mat)  # type: ignore[attr-defined]
            self._index = index
        else:
            # Fallback: store as numpy matrix for brute-force search
            self._index = mat

        logger.debug("DNAMemory index built with %d vectors.", len(self._vectors))

    def _hash(self, vector: List[float]) -> str:
        """Deterministic short hash for a 3-D vector (used as cache key)."""
        raw = json.dumps([round(v, 2) for v in vector], separators=(",", ":"))
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _l2_search(self, query: np.ndarray) -> Tuple[float, int]:
        """Return (distance, index) of nearest neighbour."""
        if _FAISS_AVAILABLE and self._index is not None:
            distances, indices = self._index.search(  # type: ignore[attr-defined]
                query.reshape(1, VECTOR_DIM), k=1
            )
            return float(distances[0][0]), int(indices[0][0])
        else:
            # Brute-force numpy fallback
            mat: np.ndarray = self._index  # type: ignore[assignment]
            diffs = mat - query.reshape(1, VECTOR_DIM)
            l2_dists = np.sqrt((diffs ** 2).sum(axis=1))
            idx = int(np.argmin(l2_dists))
            return float(l2_dists[idx]), idx

    # ────────────────────────────── public API ────────────────────────────────

    def query(self, vector: List[float]) -> MemoryHit:
        """
        Look up the nearest past incident to *vector*.

        Parameters
        ----------
        vector:
            A 3-element list ``[Traffic_Load, DB_Temperature, Network_Health]``.

        Returns
        -------
        MemoryHit:
            Contains confidence level, L2 distance, and recommended action.
        """
        if len(vector) != VECTOR_DIM:
            raise ValueError(
                f"Query vector must have {VECTOR_DIM} elements, got {len(vector)}"
            )

        query_np = np.array(vector, dtype=np.float32)
        distance, nearest_idx = self._l2_search(query_np)
        matched_vec = self._vectors[nearest_idx]
        matched_action = self._actions[nearest_idx]

        if distance < HIGH_MATCH_THRESHOLD:
            confidence = MatchConfidence.HIGH
        elif distance < MEDIUM_MATCH_THRESHOLD:
            confidence = MatchConfidence.MEDIUM
        else:
            confidence = MatchConfidence.LOW

        hit = MemoryHit(
            confidence=confidence,
            distance=distance,
            matched_vector=matched_vec,
            matched_action=matched_action,
            cache_key=self._hash(vector),
        )

        logger.info(
            "DNAMemory query=%s → %s (dist=%.2f, action=%s)",
            [round(v, 1) for v in vector],
            confidence.value,
            distance,
            matched_action,
        )
        return hit

    def add_incident(self, vector: List[float], action: str) -> None:
        """
        Persist a new incident to the live index (online learning hook).

        Parameters
        ----------
        vector:
            State vector at incident time.
        action:
            The remediation action that successfully resolved the incident.
        """
        self._add(vector, action)
        self._build_index()   # Rebuild – acceptable for a small index

    def describe(self) -> str:
        """Human-readable summary of the current index."""
        return (
            f"DNAMemory | {len(self._vectors)} incidents indexed | "
            f"backend={'faiss' if _FAISS_AVAILABLE else 'numpy'}"
        )

    def __repr__(self) -> str:
        return self.describe()
