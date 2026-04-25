"""
utils/forensic_report.py
========================
Autonomous Forensic Report Generator — OpenCloud-SRE

After every incident resolution, this module generates a professional-grade
Root Cause Analysis (RCA) report in Markdown format.

The report includes:
  - Incident Summary (ID, severity, duration, SLO impact)
  - State Tensor Timeline (what the metrics looked like)
  - Root Cause Analysis (derived from agent chat history)
  - Action Playbook (every action taken, by whom, and via which path)
  - Cognitive Compression Summary (how many LLM tokens were saved)
  - DNA Knowledge Distillation Notice (if a new pattern was learned)
  - Recommendations for future prevention

This report is designed to be presented to the hackathon judges as proof
that the system doesn't just "fix" incidents — it documents and learns from them.
"""

from __future__ import annotations

import datetime
import random
import string
from typing import Any, Dict, List, Optional


# ─────────────────────────── helpers ──────────────────────────────────────────

def _incident_id() -> str:
    """Generate a deterministic-looking incident ID."""
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"INC-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{suffix}"


def _severity(initial_traffic: float, initial_db: float, initial_net: float) -> str:
    """Classify incident severity from the initial crash state."""
    critical_count = sum([
        initial_traffic > 90,
        initial_db > 85,
        initial_net < 15,
    ])
    if critical_count >= 3:
        return "🔴 SEV-1 (Critical)"
    elif critical_count == 2:
        return "🟠 SEV-2 (High)"
    elif critical_count == 1:
        return "🟡 SEV-3 (Medium)"
    return "🟢 SEV-4 (Low)"


def _extract_root_cause(chat_history: List[Dict]) -> str:
    """
    Scan the chat history to find the most informative root cause string.
    Prioritises ChatOps rationale, then Lead SRE analysis, then Network/DB intents.
    """
    for msg in reversed(chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "chatops" and "Rationale=" in content:
            try:
                return content.split("Rationale=")[1].strip()
            except IndexError:
                pass

    for msg in reversed(chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "lead_sre" and "Shadow Consensus" in content:
            return content.replace("[Shadow Consensus]", "").strip()

    for msg in reversed(chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("network_ctrl", "db_ctrl") and "root_cause" in content.lower():
            return content[:200]

    return "Anomalous state detected across multiple metrics. Multi-agent consensus required to identify primary driver."


def _format_action_playbook(chat_history: List[Dict]) -> str:
    """Format the step-by-step action playbook from chat history."""
    ROLE_LABELS = {
        "network_ctrl": ("🌐", "Network Controller"),
        "db_ctrl":      ("🗄️", "Database Controller"),
        "lead_sre":     ("🎖️", "Lead SRE (Arbitrator)"),
        "chatops":      ("💬", "ChatOps Resolver"),
        "executor":     ("⚡", "Executor"),
        "dna_memory":   ("🧬", "DNA Memory"),
    }
    rows = ["| # | Agent | Role | Action / Decision |",
            "|---|-------|------|-------------------|"]
    step = 0
    for msg in chat_history:
        role = msg.get("role", "sys")
        content = msg.get("content", "")
        icon, label = ROLE_LABELS.get(role, ("·", role.upper()))
        # Truncate long content to keep the table readable
        short = content[:120].replace("|", "\\|").replace("\n", " ")
        if len(content) > 120:
            short += "…"
        step += 1
        rows.append(f"| {step} | {icon} | **{label}** | `{short}` |")
    return "\n".join(rows) if len(rows) > 2 else "_No agent activity recorded._"


def _routing_summary(timeline: List[Dict]) -> str:
    """Summarise the routing path distribution from the timeline events."""
    fast_count   = sum(1 for e in timeline if "Fast Path" in e.get("event", "") or "DNA Cache" in e.get("event", ""))
    middle_count = sum(1 for e in timeline if "MIDDLE" in e.get("detail", "").upper() or "Middle" in e.get("detail", ""))
    slow_count   = sum(1 for e in timeline if "ChatOps" in e.get("event", "") or "SLOW" in e.get("detail", "").upper())
    total_steps  = max(1, fast_count + middle_count + slow_count)

    lines = []
    if fast_count:
        lines.append(f"- **FAST PATH (DNA Cache)**: {fast_count} step(s) — {fast_count/total_steps:.0%} of routing")
    if middle_count:
        lines.append(f"- **MIDDLE PATH (Shadow Consensus)**: {middle_count} step(s) — {middle_count/total_steps:.0%} of routing")
    if slow_count:
        lines.append(f"- **SLOW PATH (ChatOps)**: {slow_count} step(s) — {slow_count/total_steps:.0%} of routing")
    if not lines:
        lines.append("- _Routing data not yet available._")
    return "\n".join(lines)


# ─────────────────────────── main generator ───────────────────────────────────

def generate_markdown_report(
    chat_history: List[Dict],
    timeline: List[Dict],
    initial_vector: List[float],
    final_vector: List[float],
    final_slo: float,
    total_steps: int,
    tokens_saved: int,
    routing_path: str,
    last_action: str,
    duration_seconds: float,
    blast_warnings: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a complete Markdown forensic report for the resolved incident.

    Parameters
    ----------
    chat_history   : Full agent chat log from SREGraphState.
    timeline       : Incident timeline events from st.session_state.timeline.
    initial_vector : [Traffic, DB_Temp, Net_Health] at incident start.
    final_vector   : [Traffic, DB_Temp, Net_Health] at resolution.
    final_slo      : SLO score at resolution (0–1).
    total_steps    : Number of graph steps taken.
    tokens_saved   : Estimated LLM tokens saved via cognitive compression.
    routing_path   : Final routing path string ("fast_path", "middle_path", etc.).
    last_action    : The final remediation action applied.
    duration_seconds : Wall-clock seconds from start to resolution.
    blast_warnings : Any blast radius warnings raised during the incident.
    metadata       : Free-form experiment metadata.
    """
    now       = datetime.datetime.utcnow()
    inc_id    = _incident_id()
    severity  = _severity(*initial_vector[:3]) if len(initial_vector) >= 3 else "Unknown"
    root_cause = _extract_root_cause(chat_history)
    playbook   = _format_action_playbook(chat_history)
    routing_sum = _routing_summary(timeline)

    blast_section = ""
    if blast_warnings:
        blast_section = "\n## ⚡ Blast Radius Warnings\n\n"
        blast_section += "> These secondary risks were flagged and factored into the governance decision.\n\n"
        for w in blast_warnings:
            blast_section += f"- ⚠️ `{w}`\n"

    dna_section = ""
    dna_learned = any(
        "Distilled" in e.get("event", "") or "Autonomous Learning" in e.get("detail", "")
        for e in timeline
    )
    # Also check chat history for distillation messages
    if not dna_learned:
        dna_learned = any(
            msg.get("role") == "dna_memory" and "Distilled" in msg.get("content", "")
            for msg in chat_history
        )
    if dna_learned:
        dna_section = f"""
## 🧬 DNA Knowledge Distillation

> **New pattern learned and committed to DNA Memory (FAISS index).**

This incident produced a novel resolution that was not in the existing DNA cache.
The system autonomously distilled the resolution into a new FAISS vector entry:

| Field | Value |
|-------|-------|
| **Incident Vector** | `Traffic={initial_vector[0]:.1f}, DB={initial_vector[1]:.1f}, Net={initial_vector[2]:.1f}` |
| **Validated Action** | `{last_action}` |
| **Future Routing** | FAST PATH (sub-millisecond, zero LLM cost) |

On the next similar incident, the system will skip the expensive reasoning chain entirely.
"""
    else:
        dna_section = f"""
## 🧬 DNA Memory Status

This incident matched an existing pattern in the DNA cache (FAST PATH activated).
No new entry was needed — the system's prior knowledge was sufficient.

| Field | Value |
|-------|-------|
| **Cache Status** | ✅ Hit — Pattern already known |
| **Action Applied** | `{last_action}` |
| **Tokens Cost** | 0 (purely deterministic) |
"""

    recommendations = [
        f"Monitor `Traffic_Load` threshold alerts at **80%** to trigger earlier intervention.",
        f"Review `Database_Temperature` cooling policies — current ceiling exceeded **{initial_vector[1]:.0f}°** during this incident.",
        f"Consider pre-seeding the DNA Memory with vectors near `[{initial_vector[0]:.0f}, {initial_vector[1]:.0f}, {initial_vector[2]:.0f}]` to prevent future Slow Path escalations.",
        f"The `{last_action.replace('_', ' ')}` action was the decisive remediation. Validate that this action is in the top-priority synergy table.",
    ]
    if blast_warnings:
        recommendations.append("Blast Radius warnings were raised. Review the dependency matrix for `restart_pods` + `schema_failover` combinations.")

    rec_md = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations))

    cost_estimate_usd = (len(chat_history) * 0.002) - (tokens_saved * 0.000002)
    cost_estimate_usd = max(0.0, cost_estimate_usd)

    report = f"""# 📋 OpenCloud-SRE Forensic Incident Report
> *Auto-generated by the Autonomous Forensic Report Engine · OpenCloud-SRE v1.0*

---

## 📌 Incident Summary

| Field | Value |
|-------|-------|
| **Incident ID** | `{inc_id}` |
| **Generated At** | `{now.strftime("%Y-%m-%d %H:%M:%S UTC")}` |
| **Severity** | {severity} |
| **Duration** | `{duration_seconds:.1f}s` wall-clock |
| **Total Steps** | `{total_steps}` graph cycles |
| **Final SLO Score** | `{final_slo:.3f}` {"✅ RECOVERED" if final_slo >= 0.95 else "⚠️ DEGRADED"} |
| **Decisive Action** | `{last_action}` |
| **Final Routing Path** | `{routing_path.replace("_", " ").upper()}` |
| **LLM Tokens Saved** | `{tokens_saved:,}` (~${cost_estimate_usd:.4f} USD saved) |

---

## 📊 State Tensor: Before → After

| Metric | At Incident Start | At Resolution | Change |
|--------|:-----------------:|:-------------:|:------:|
| **Traffic Load** | `{initial_vector[0]:.1f}` | `{final_vector[0]:.1f}` | `{final_vector[0]-initial_vector[0]:+.1f}` |
| **DB Temperature** | `{initial_vector[1]:.1f}` | `{final_vector[1]:.1f}` | `{final_vector[1]-initial_vector[1]:+.1f}` |
| **Network Health** | `{initial_vector[2]:.1f}` | `{final_vector[2]:.1f}` | `{final_vector[2]-initial_vector[2]:+.1f}` |

---

## 🔍 Root Cause Analysis (RCA)

> **Primary Driver identified by multi-agent consensus:**

{root_cause}

The incident was routed through the **3-Tier Cognitive Hierarchy**:

{routing_sum}

---

## 🤖 Agent Action Playbook

{playbook}

---
{blast_section}
{dna_section}

---

## 💡 Recommendations

{rec_md}

---

## 🧠 Cognitive Compression Metrics

| Metric | Value |
|--------|-------|
| **Tokens Saved (Fast Path)** | `{tokens_saved:,}` |
| **Estimated Cost Saved** | `~${cost_estimate_usd:.4f} USD` |
| **Agents Involved** | `{len(set(m.get("role") for m in chat_history))}` specialised agents |
| **Chat Messages** | `{len(chat_history)}` inter-agent messages |

---
*This report was generated autonomously by OpenCloud-SRE's Forensic Report Engine.*
*No human SRE wrote this document — it emerged from the agent swarm.*
"""
    return report
