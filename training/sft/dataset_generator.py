"""
training/sft/dataset_generator.py
===================================
Synthetic SFT dataset generator for OpenCloud-SRE.

Generates 50 high-quality incident resolution logs that mimic the JSON output
of a perfectly operating LangGraph pipeline.  These are used as the Supervised
Fine-Tuning (SFT) seed dataset for the Unsloth/TRL training loop on hackathon
VMs.

Each log entry captures one complete episode step:
  • The initial crashed state
  • The 3-tier routing decision (path taken)
  • The inter-agent chat (network_ctrl, db_ctrl, lead_sre)
  • The final executed action and resulting metrics
  • A single "perfect" assistant turn that a fine-tuned model should replicate

Output format: JSON Lines (.jsonl), one episode per line.

Usage
-----
  export OPENAI_API_KEY=sk-...
  python -m training.sft.dataset_generator --output data/sft_logs.jsonl --count 50

Without an API key the script falls back to a rule-based generator that
produces structurally identical records for local testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────── optional OpenAI ─────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

# ────────────────────────────── constants ────────────────────────────────────

DEFAULT_MODEL = "gpt-4o"
DEFAULT_COUNT = 50
DEFAULT_OUTPUT = "data/sft_logs.jsonl"
BATCH_SIZE = 5            # generate this many records per API call
TEMPERATURE = 0.85        # higher = more diverse scenarios
REQUEST_DELAY_S = 1.5     # rate-limit courtesy delay between batches

# Routing path distribution (roughly realistic)
_PATH_WEIGHTS = {
    "fast_path": 0.25,
    "middle_path": 0.55,
    "slow_path": 0.20,
}

VALID_ACTIONS = [
    "throttle_traffic", "load_balance", "schema_failover",
    "cache_flush", "circuit_breaker", "restart_pods", "scale_out", "noop",
]

FAULT_SCENARIOS = [
    "traffic_spike",
    "db_overload",
    "network_partition",
    "cascade_failure",
    "hot_key_storm",
    "replica_lag",
    "BGP_flap",
    "memory_leak_OOM",
]

# ──────────────────────── GPT-4o generation prompt ───────────────────────────

GENERATION_SYSTEM_PROMPT = """
You are a senior Site Reliability Engineer generating a synthetic training dataset
for an AI incident command system called OpenCloud-SRE.

Generate exactly {batch_size} incident resolution log entries as a JSON array.

Each entry MUST follow this exact schema:
{{
  "id": <integer>,
  "timestamp": "<ISO-8601 string>",
  "fault_scenario": "<descriptive fault name>",
  "initial_state": {{
    "Traffic_Load": <float 0-100>,
    "Database_Temperature": <float 0-100>,
    "Network_Health": <float 0-100>
  }},
  "routing_path": "<fast_path | middle_path | slow_path>",
  "dna_memory_hit": {{
    "confidence": "<High Match | Medium Match | Low Match>",
    "distance": <float>,
    "recommended_action": "<action>"
  }},
  "network_intent": {{
    "intent": "<throttle | circuit_break | scale_out | load_balance | noop>",
    "confidence": <float 0-1>,
    "rationale": "<one sentence>"
  }},
  "db_intent": {{
    "intent": "<failover | cache_flush | restart | noop>",
    "confidence": <float 0-1>,
    "rationale": "<one sentence>"
  }},
  "consensus_status": "<green | red>",
  "executed_action": "<one of the 8 valid actions>",
  "post_action_state": {{
    "Traffic_Load": <float 0-100>,
    "Database_Temperature": <float 0-100>,
    "Network_Health": <float 0-100>
  }},
  "slo_score_before": <float 0-1>,
  "slo_score_after": <float 0-1>,
  "resolution_success": <true | false>,
  "agent_chat": [
    {{"role": "network_ctrl", "content": "<intent summary>"}},
    {{"role": "db_ctrl", "content": "<intent summary>"}},
    {{"role": "lead_sre", "content": "<consensus decision>"}},
    {{"role": "executor", "content": "<action taken + outcome>"}}
  ],
  "perfect_assistant_response": "<The ideal SRE response as a single coherent paragraph (3-5 sentences). Explain what happened, why the action was chosen, and what the outcome was. Write as if you are the Lead SRE briefing the on-call team.>"
}}

Rules:
- Make each scenario realistic and distinct (use all 8 fault types across the batch).
- slo_score_after should be higher than slo_score_before for successful resolutions.
- routing_path == "fast_path" means dna_memory_hit.confidence == "High Match".
- routing_path == "slow_path" means consensus_status == "red" and the conflict was resolved.
- fast_path entries have simpler agent_chat (just executor).
- Vary the fault scenarios, SLO improvements, and resolutions across entries.
- Start IDs from {start_id}.
- Output ONLY the JSON array, no other text.
""".strip()

# ─────────────────────────────── rule-based fallback ─────────────────────────


def _random_state(crashed: bool = True) -> Dict[str, float]:
    if crashed:
        return {
            "Traffic_Load": round(random.uniform(75, 99), 1),
            "Database_Temperature": round(random.uniform(70, 98), 1),
            "Network_Health": round(random.uniform(5, 35), 1),
        }
    return {
        "Traffic_Load": round(random.uniform(10, 40), 1),
        "Database_Temperature": round(random.uniform(15, 45), 1),
        "Network_Health": round(random.uniform(70, 95), 1),
    }


def _slo_score(state: Dict[str, float]) -> float:
    traffic_norm = (100 - state["Traffic_Load"]) / 100
    db_norm = (100 - state["Database_Temperature"]) / 100
    net_norm = state["Network_Health"] / 100
    return round((traffic_norm + db_norm + net_norm) / 3, 4)


def _rule_based_entry(entry_id: int) -> Dict[str, Any]:
    """Generate a single structurally valid log entry without an LLM."""
    random.seed(entry_id * 37)

    fault = random.choice(FAULT_SCENARIOS)
    initial = _random_state(crashed=True)
    path_weights = list(_PATH_WEIGHTS.values())
    paths = list(_PATH_WEIGHTS.keys())
    routing_path = random.choices(paths, weights=path_weights, k=1)[0]

    action = random.choice(VALID_ACTIONS)
    post = _random_state(crashed=False)
    slo_before = _slo_score(initial)
    slo_after = _slo_score(post)

    net_intents = ["throttle", "circuit_break", "scale_out", "load_balance", "noop"]
    db_intents = ["failover", "cache_flush", "restart", "noop"]
    net_intent_val = random.choice(net_intents)
    db_intent_val = random.choice(db_intents)
    consensus = "red" if (
        net_intent_val in ("circuit_break",) and db_intent_val in ("failover", "restart")
    ) else "green"

    agent_chat = [
        {"role": "network_ctrl", "content": f"[Network Intent] intent={net_intent_val} confidence=0.82"},
        {"role": "db_ctrl", "content": f"[DB Intent] intent={db_intent_val} confidence=0.76"},
        {"role": "lead_sre", "content": f"[Shadow Consensus] {consensus.upper()} → action={action}"},
        {"role": "executor", "content": f"[Executor] action={action} | SLO_before={slo_before:.3f} → SLO_after={slo_after:.3f}"},
    ]

    return {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fault_scenario": fault,
        "initial_state": initial,
        "routing_path": routing_path,
        "dna_memory_hit": {
            "confidence": "High Match" if routing_path == "fast_path" else "Low Match",
            "distance": round(random.uniform(2.0, 25.0), 2),
            "recommended_action": action,
        },
        "network_intent": {
            "intent": net_intent_val,
            "confidence": round(random.uniform(0.55, 0.95), 3),
            "rationale": f"Network metric suggests {net_intent_val} to stabilise load.",
        },
        "db_intent": {
            "intent": db_intent_val,
            "confidence": round(random.uniform(0.55, 0.95), 3),
            "rationale": f"Database temperature warrants {db_intent_val}.",
        },
        "consensus_status": consensus,
        "executed_action": action,
        "post_action_state": post,
        "slo_score_before": slo_before,
        "slo_score_after": slo_after,
        "resolution_success": slo_after > slo_before,
        "agent_chat": agent_chat,
        "perfect_assistant_response": (
            f"Detected a {fault} incident with Traffic_Load={initial['Traffic_Load']}, "
            f"DB_Temperature={initial['Database_Temperature']}, "
            f"Network_Health={initial['Network_Health']}. "
            f"The Shadow Consensus layer ({consensus}) resolved to execute '{action}' "
            f"based on network intent '{net_intent_val}' and DB intent '{db_intent_val}'. "
            f"Post-remediation the SLO score improved from {slo_before:.3f} to {slo_after:.3f}, "
            f"{'successfully recovering the system.' if slo_after > slo_before else 'indicating further intervention is required.'}"
        ),
    }


# ──────────────────────── GPT-4o batch generator ─────────────────────────────


def _generate_batch_via_llm(
    client: Any,
    batch_size: int,
    start_id: int,
    model: str,
) -> List[Dict[str, Any]]:
    """Call GPT-4o to generate a batch of log entries."""
    prompt = GENERATION_SYSTEM_PROMPT.format(
        batch_size=batch_size,
        start_id=start_id,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a JSON generation assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content or "[]"
    # GPT might wrap in {"entries": [...]} – unwrap if needed
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    for key in ("entries", "logs", "data", "records"):
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    raise ValueError(f"Unexpected GPT response shape: {list(parsed.keys())}")


# ─────────────────────────────── main logic ───────────────────────────────────


def generate_dataset(
    count: int = DEFAULT_COUNT,
    output_path: str = DEFAULT_OUTPUT,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
) -> Path:
    """
    Generate *count* synthetic incident logs and save to *output_path*.

    Parameters
    ----------
    count:
        Number of log entries to generate.
    output_path:
        Destination ``.jsonl`` file path (created if absent).
    model:
        OpenAI model to use (default: ``gpt-4o``).
    use_llm:
        If True and an API key is available, use GPT-4o.
        Otherwise use the rule-based fallback.

    Returns
    -------
    Path
        Absolute path to the written ``.jsonl`` file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = None
    if use_llm and _OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            client = OpenAI(api_key=api_key)
            logger.info("Using GPT-4o (%s) for generation.", model)
        else:
            logger.warning("OPENAI_API_KEY not set – falling back to rule-based generator.")
    elif not use_llm:
        logger.info("use_llm=False – using rule-based generator.")

    all_entries: List[Dict[str, Any]] = []
    entry_id = 1

    if client is not None:
        # Batch GPT-4o generation
        while len(all_entries) < count:
            remaining = count - len(all_entries)
            batch_size = min(BATCH_SIZE, remaining)
            logger.info(
                "Generating batch %d/%d via LLM (start_id=%d, size=%d)...",
                len(all_entries) // BATCH_SIZE + 1,
                (count + BATCH_SIZE - 1) // BATCH_SIZE,
                entry_id,
                batch_size,
            )
            try:
                batch = _generate_batch_via_llm(client, batch_size, entry_id, model)
                all_entries.extend(batch[:batch_size])
                entry_id += batch_size
                time.sleep(REQUEST_DELAY_S)
            except Exception as exc:
                logger.error("LLM batch failed (%s) – falling back to rules for this batch.", exc)
                for _ in range(batch_size):
                    all_entries.append(_rule_based_entry(entry_id))
                    entry_id += 1
    else:
        # Pure rule-based generation
        for i in range(1, count + 1):
            all_entries.append(_rule_based_entry(i))
            if i % 10 == 0:
                logger.info("Generated %d/%d entries.", i, count)

    # Truncate to exactly `count` entries
    all_entries = all_entries[:count]

    # Write JSONL
    with out.open("w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        "✅ Dataset saved: %s (%d entries, %.1f KB)",
        out.resolve(),
        len(all_entries),
        out.stat().st_size / 1024,
    )
    return out.resolve()


# ──────────────────────────────── CLI entry-point ─────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate synthetic SFT logs for OpenCloud-SRE fine-tuning."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output .jsonl file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Number of log entries to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM calls; use rule-based generation only (useful for CI).",
    )
    args = parser.parse_args()

    out_path = generate_dataset(
        count=args.count,
        output_path=args.output,
        model=args.model,
        use_llm=not args.no_llm,
    )
    print(f"\nDataset written to: {out_path}")


if __name__ == "__main__":
    main()
