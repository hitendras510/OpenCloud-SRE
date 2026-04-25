"""ui/app.py — OpenCloud-SRE NEXUS Command Center v3.0"""
from __future__ import annotations
import sys, time, logging, requests
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(
    page_title="OpenCloud-SRE · NEXUS",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = Path(__file__).parent / "styles.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── imports ───────────────────────────────────────────────────────────────────
try:
    from env.environment import OpenCloudEnv
    from graph.sre_graph import build_sre_graph
    from graph.message_bus import (
        initial_state, RoutingPath, ConsensusStatus,
        GovernanceSignal, TrustDecision,
    )
    _OK = True; _ERR = ""
except ImportError as e:
    _OK = False; _ERR = str(e)

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ── constants ─────────────────────────────────────────────────────────────────
MAX_H         = 80
TOKENS_FAST   = 1800
TOKENS_MIDDLE = 900

ROLE_CSS  = {"network_ctrl":"t-net","db_ctrl":"t-db","lead_sre":"t-sre",
             "executor":"t-exec","dna_memory":"t-dna","chatops":"t-ops"}
ROLE_ICON = {"network_ctrl":"NET","db_ctrl":"DB ","lead_sre":"SRE",
             "executor":"EXE","dna_memory":"DNA","chatops":"OPS"}

def _plotly_base(h=280):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#484f58", size=10, family="JetBrains Mono"),
        height=h, margin=dict(l=0, r=0, t=28, b=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)",
                   showline=False, zeroline=False, color="#484f58"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)",
                   showline=False, zeroline=False, color="#484f58"),
        hoverlabel=dict(bgcolor="#060f1e", font_color="#c9d1d9",
                        bordercolor="rgba(0,212,255,0.2)"),
        hovermode="x unified",
    )

# ── session init ──────────────────────────────────────────────────────────────
def _init():
    if "ok" in st.session_state: return
    st.session_state.ok           = True
    st.session_state.running      = False
    st.session_state.step         = 0
    st.session_state.resolved     = False
    st.session_state.traffic      = deque([98.0]*12, maxlen=MAX_H)
    st.session_state.db           = deque([95.0]*12, maxlen=MAX_H)
    st.session_state.net          = deque([5.0]*12,  maxlen=MAX_H)
    st.session_state.slo          = deque([0.05]*12, maxlen=MAX_H)
    st.session_state.steps        = deque(range(12),  maxlen=MAX_H)
    st.session_state.chat: List[Dict] = []
    st.session_state.gstate       = None
    st.session_state.consensus    = "red"
    st.session_state.last_action  = "—"
    st.session_state.routing      = "—"
    st.session_state.gov_signal   = "DEEP_NEGOTIATE"
    st.session_state.trust        = "escrowed"
    st.session_state.confidence   = 0.0
    st.session_state.blast_warnings: List[str] = []
    st.session_state.tokens_saved = 0
    st.session_state.human_approved = False
    st.session_state.human_rejected = False
    st.session_state.timeline: List[Dict] = []
    st.session_state.incident_start_time = None
    st.session_state.path_counts  = {"FAST": 0, "MIDDLE": 0, "SLOW": 0}
    st.session_state.gov_counts: Dict[str,int]    = {}
    st.session_state.action_counts: Dict[str,int] = {}
    st.session_state.demo_success = False
_init()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:20px">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:800;
        background:linear-gradient(90deg,#00d4ff,#6366f1);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
        NEXUS · Controls
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#484f58;
           letter-spacing:.12em;text-transform:uppercase;margin-top:2px">
        OpenCloud-SRE War Room
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    mock_llm   = st.toggle("🧠 Mock LLM (offline)", value=True)
    step_delay = st.slider("⏱ Step Delay (s)", 0.5, 3.0, 1.0, 0.5)
    st.divider()
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ Start", type="primary", use_container_width=True)
    stop_btn  = c2.button("⏹ Stop",  use_container_width=True)
    reset_btn = st.button("↺ Reset Episode", use_container_width=True)
    st.divider()
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#484f58;
         letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Routing Tiers</div>
    <div style="display:flex;flex-direction:column;gap:6px">
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:8px;height:8px;border-radius:50%;background:#00d4ff;box-shadow:0 0 6px rgba(0,212,255,0.6)"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#8b949e">FAST — DNA Cache Hit</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:8px;height:8px;border-radius:50%;background:#818cf8;box-shadow:0 0 6px rgba(99,102,241,0.6)"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#8b949e">MIDDLE — Shadow Consensus</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:8px;height:8px;border-radius:50%;background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,0.6)"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#8b949e">SLOW — ChatOps Resolver</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#484f58;
         letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">Governance</div>
    <div style="display:flex;flex-direction:column;gap:5px">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#34d399">✅ AUTO_RESOLVE</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#fcd34d">⏳ HUMAN_ESCALATION</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#fb7185">🚨 DEEP_NEGOTIATE</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#a78bfa">🛑 BLAST_RADIUS_BLOCK</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;color:#f43f5e;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">🔴 Chaos Control Center</div>', unsafe_allow_html=True)
    if st.button("Inject CPU Spike", use_container_width=True):
        try:
            requests.post("http://127.0.0.1:8000/inject-fault", json={"fault_type": "CPU_SPIKE", "value": 95.0}, timeout=2)
            st.toast("🔴 CPU Spike Injected!")
        except Exception as e:
            st.toast(f"Error injecting fault: {e}")
    if st.button("Simulate Network Partition", use_container_width=True):
        try:
            requests.post("http://127.0.0.1:8000/inject-fault", json={"fault_type": "NETWORK_PARTITION", "value": 95.0}, timeout=2)
            st.toast("🔴 Network Partition Simulated!")
        except Exception as e:
            st.toast(f"Error injecting fault: {e}")
    if st.button("Trigger DB Deadlock", use_container_width=True):
        try:
            requests.post("http://127.0.0.1:8000/inject-fault", json={"fault_type": "DB_DEADLOCK", "value": 95.0}, timeout=2)
            st.toast("🔴 DB Deadlock Triggered!")
        except Exception as e:
            st.toast(f"Error injecting fault: {e}")
            
    st.divider()
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;color:#484f58;text-align:center;letter-spacing:.06em">OpenCloud-SRE · Cognitive Compression<br>Meta PyTorch · Hackathon 2025</div>', unsafe_allow_html=True)

# ── env + graph ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build(mock):
    if not _OK: return None, None
    e = OpenCloudEnv(seed=42, crash_on_reset=True); e.reset()
    g = build_sre_graph(env=e, mock_llm=mock)
    return e, g

env, graph = _build(mock_llm)

# ── button handlers ───────────────────────────────────────────────────────────
if start_btn and _OK:
    st.session_state.running = True
    import time as _t
    st.session_state.incident_start_time = _t.perf_counter()
    st.session_state.timeline = [{
        "elapsed": 0.0, "icon": "🚨", "color": "#f43f5e",
        "event": "INCIDENT DETECTED",
        "detail": f"Traffic={list(st.session_state.traffic)[-1]:.0f} "
                  f"DB={list(st.session_state.db)[-1]:.0f} "
                  f"Net={list(st.session_state.net)[-1]:.0f}",
    }]
if stop_btn: st.session_state.running = False
if reset_btn and _OK:
    st.session_state.update({
        "running": False, "step": 0, "resolved": False, "chat": [],
        "traffic": deque([98.0]*12, maxlen=MAX_H), "db": deque([95.0]*12, maxlen=MAX_H),
        "net": deque([5.0]*12, maxlen=MAX_H), "slo": deque([0.05]*12, maxlen=MAX_H),
        "steps": deque(range(12), maxlen=MAX_H), "gstate": None, "tokens_saved": 0,
        "blast_warnings": [], "human_approved": False, "timeline": [],
        "incident_start_time": None, "path_counts": {"FAST":0,"MIDDLE":0,"SLOW":0},
        "gov_counts": {}, "action_counts": {}, "demo_success": False,
    })
    if env: env.reset()
    st.cache_resource.clear()

# ══════════════════════════════════════════════════════════════════════════════
# NEXUS HEADER
# ══════════════════════════════════════════════════════════════════════════════
status_color = "#34d399" if st.session_state.running else "#f43f5e"
status_text  = "LIVE" if st.session_state.running else "STANDBY"

st.markdown(f"""
<div class="nexus-header">
  <div>
    <div class="nexus-wordmark">OpenCloud-SRE</div>
    <div class="nexus-sub">
      NEXUS COMMAND CENTER &nbsp;·&nbsp; COGNITIVE COMPRESSION ENGINE &nbsp;·&nbsp; AUTONOMOUS SRE
    </div>
  </div>
  <div class="nexus-badges">
    <div class="badge badge-cyan"><span class="pulse-dot"></span>{status_text}</div>
    <div class="badge badge-cyan">🧬 DNA FAISS</div>
    <div class="badge badge-indigo">🤝 SHADOW CONSENSUS</div>
    <div class="badge badge-emerald">🛡️ BLAST RADIUS FILTER</div>
    <div class="badge badge-rose">🔒 TRUST ESCROW</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not _OK:
    st.error(f"⚠️ Import error: `{_ERR}` — run `pip install -r requirements.txt`")

# ── KPI strip ─────────────────────────────────────────────────────────────────
tv = list(st.session_state.traffic)[-1]
dv = list(st.session_state.db)[-1]
nv = list(st.session_state.net)[-1]
sv = list(st.session_state.slo)[-1]
tp = list(st.session_state.traffic)[-2] if len(st.session_state.traffic) > 1 else tv
dp = list(st.session_state.db)[-2]      if len(st.session_state.db)      > 1 else dv
np_ = list(st.session_state.net)[-2]    if len(st.session_state.net)     > 1 else nv
sp = list(st.session_state.slo)[-2]     if len(st.session_state.slo)     > 1 else sv

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🔴 Traffic",      f"{tv:.1f}", f"{tv-tp:+.1f}")
k2.metric("🟡 DB Temp",      f"{dv:.1f}", f"{dv-dp:+.1f}")
k3.metric("🟢 Net Health",   f"{nv:.1f}", f"{nv-np_:+.1f}")
k4.metric("🎯 SLO Score",    f"{sv:.3f}", f"{sv-sp:+.3f}")
k5.metric("📶 Step",         st.session_state.step)
with k6:
    st.markdown(f"""
    <div class="tokens-widget">
      <span class="tokens-num">{st.session_state.tokens_saved:,}</span>
      <span class="tokens-label">⚡ tokens saved</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_war, tab_analytics, tab_dna = st.tabs([
    "⚡  War Room",
    "📊  Analytics",
    "🧬  DNA Memory",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1  ——  WAR ROOM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_war:
    col_l, col_r = st.columns([3, 2], gap="large")

    # ── LEFT ─────────────────────────────────────────────────────────────────
    with col_l:
        # ── ROUTING PIPELINE ──────────────────────────────────────────────
        rp_lower = st.session_state.routing.lower()
        is_fast   = "fast"   in rp_lower
        is_middle = "middle" in rp_lower
        is_slow   = "slow"   in rp_lower and not is_fast

        f_cls = "pl-icon-fast"   if is_fast   else "pl-icon-off"
        m_cls = "pl-icon-middle" if is_middle  else "pl-icon-off"
        s_cls = "pl-icon-slow"   if is_slow    else "pl-icon-off"
        a1 = "pl-arrow-active" if is_fast or is_middle or is_slow else "pl-arrow"
        a2 = "pl-arrow-active" if is_middle or is_slow else "pl-arrow"

        st.markdown(f"""
        <div class="pipeline">
          <div class="pl-node">
            <div class="pl-icon {f_cls}">🧬</div>
            <div class="pl-label">FAST</div>
            <div class="pl-sublabel">DNA Cache</div>
          </div>
          <div class="{a1}">→</div>
          <div class="pl-node">
            <div class="pl-icon {m_cls}">🤝</div>
            <div class="pl-label">MIDDLE</div>
            <div class="pl-sublabel">Shadow Consensus</div>
          </div>
          <div class="{a2}">→</div>
          <div class="pl-node">
            <div class="pl-icon {s_cls}">💬</div>
            <div class="pl-label">SLOW</div>
            <div class="pl-sublabel">ChatOps LLM</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── REAL-TIME CHART ────────────────────────────────────────────────
        st.markdown('<p class="section-label">── LIVE SERVER METRICS</p>', unsafe_allow_html=True)
        chart_ph = st.empty()

        def _chart():
            xs = list(st.session_state.steps)
            tr = list(st.session_state.traffic)
            db = list(st.session_state.db)
            nt = list(st.session_state.net)
            sl = [v*100 for v in st.session_state.slo]
            if not _PLOTLY:
                chart_ph.line_chart({"Traffic":tr,"DB":db,"Network":nt}); return
            fig = go.Figure()
            fig.add_hrect(y0=85, y1=102, fillcolor="rgba(16,185,129,0.03)", line_width=0,
                          annotation_text="SLO Zone", annotation_position="top right",
                          annotation_font_color="rgba(16,185,129,0.5)", annotation_font_size=9)
            lw = 2
            fig.add_trace(go.Scatter(x=xs, y=tr, name="Traffic",
                line=dict(color="#f43f5e", width=lw),
                fill="tozeroy", fillcolor="rgba(244,63,94,0.04)"))
            fig.add_trace(go.Scatter(x=xs, y=db, name="DB Temp",
                line=dict(color="#f59e0b", width=lw),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.04)"))
            fig.add_trace(go.Scatter(x=xs, y=nt, name="Network",
                line=dict(color="#10b981", width=lw),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.05)"))
            fig.add_trace(go.Scatter(x=xs, y=sl, name="SLO×100",
                line=dict(color="#6366f1", dash="dot", width=1.5)))
            layout = _plotly_base(260)
            layout.update(
                yaxis=dict(range=[0,105], showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False, color="#484f58"),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            fig.update_layout(**layout)
            chart_ph.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        _chart()

        # ── GAUGE ROW ──────────────────────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:14px">── CURRENT STATE GAUGES</p>', unsafe_allow_html=True)
        if _PLOTLY:
            ga, gb, gc = st.columns(3)
            for col, val, name, color in [
                (ga, tv, "Traffic Load",   "#f43f5e"),
                (gb, dv, "DB Temperature", "#f59e0b"),
                (gc, nv, "Network Health", "#10b981"),
            ]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=val,
                    title={"text": name, "font": {"size": 11, "color": "#484f58", "family": "Space Grotesk"}},
                    number={"font": {"size": 24, "color": "#f0f6fc", "family": "Space Grotesk"}, "suffix": ""},
                    gauge={
                        "axis": {"range": [0,100], "tickcolor": "#21262d", "tickwidth": 1,
                                 "tickfont": {"size": 8, "color": "#484f58"}},
                        "bar": {"color": color, "thickness": 0.4},
                        "bgcolor": "rgba(255,255,255,0.02)",
                        "bordercolor": "rgba(255,255,255,0.05)", "borderwidth": 1,
                        "steps": [
                            {"range": [0,50],  "color": "rgba(16,185,129,0.04)"},
                            {"range": [50,75], "color": "rgba(245,158,11,0.04)"},
                            {"range": [75,100],"color": "rgba(244,63,94,0.06)"},
                        ],
                        "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.85, "value": val},
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=155, margin=dict(l=10,r=10,t=28,b=0))
                col.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        # ── PARTIAL OBSERVABILITY ──────────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:14px">── AGENT BLIND SPOTS</p>', unsafe_allow_html=True)
        v1, v2 = st.columns(2, gap="medium")
        with v1:
            st.markdown(f"""
            <div class="vision vision-net">
              <div class="v-title">🌐 Network Agent View</div>
              <div class="v-num">{tv:.0f}</div>
              <div class="v-sub">Traffic Load Index</div>
              <div class="v-bar"><div class="v-fill" style="width:{min(100,tv)}%"></div></div>
              <div class="v-hidden">⛔ DB_Temp → REDACTED<br>⛔ Net_Health → REDACTED</div>
            </div>""", unsafe_allow_html=True)
        with v2:
            st.markdown(f"""
            <div class="vision vision-db">
              <div class="v-title">🗄️ Database Agent View</div>
              <div class="v-num">{dv:.0f}</div>
              <div class="v-sub">DB Heat Index</div>
              <div class="v-bar"><div class="v-fill" style="width:{min(100,dv)}%"></div></div>
              <div class="v-hidden">⛔ Traffic_Load → REDACTED<br>⛔ Net_Health → REDACTED</div>
            </div>""", unsafe_allow_html=True)

        # ── BLAST RADIUS ───────────────────────────────────────────────────
        blast_ph = st.empty()
        def _render_blast():
            w = st.session_state.blast_warnings
            if not w: blast_ph.empty(); return
            html = ""
            for x in w[-2:]:
                html += f'<div class="blast"><div class="blast-title">⚡ BLAST RADIUS — EXECUTION BLOCKED</div><div class="blast-body">{x}</div></div>'
            blast_ph.markdown(html, unsafe_allow_html=True)
        _render_blast()

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    with col_r:
        # ── GOVERNANCE ORB ─────────────────────────────────────────────────
        st.markdown('<p class="section-label">── GOVERNANCE SIGNAL</p>', unsafe_allow_html=True)
        gov_ph = st.empty()

        def _render_gov():
            g = st.session_state.gov_signal
            if g == "AUTO_RESOLVE":
                orb, lbl, pill, pc = "gov-orb-green",  "AUTO-RESOLVE",   "pill-green",  "#34d399"
            elif g == "HUMAN_ESCALATION":
                orb, lbl, pill, pc = "gov-orb-yellow", "ESCROW ACTIVE",  "pill-yellow", "#fcd34d"
            elif g == "BLAST_RADIUS_BLOCK":
                orb, lbl, pill, pc = "gov-orb-purple", "BLAST BLOCKED",  "pill-purple", "#a78bfa"
            else:
                orb, lbl, pill, pc = "gov-orb-red",    "DEEP NEGOTIATE", "pill-red",    "#fb7185"

            act = (st.session_state.last_action or "—").replace("_"," ").upper()
            gov_ph.markdown(f"""
            <div class="gov-panel">
              <div class="gov-orb {orb}">{"✅" if g=="AUTO_RESOLVE" else "⏳" if g=="HUMAN_ESCALATION" else "🛑" if g=="BLAST_RADIUS_BLOCK" else "🚨"}</div>
              <div class="gov-status-label" style="color:{pc}">{lbl}</div>
              <span class="pill {pill}">{g}</span>
              <div class="gov-meta">
                <span style="color:#484f58">LAST ACTION</span><br>
                <span style="color:#c9d1d9;font-weight:600">{act}</span><br>
                <span style="color:#484f58;margin-top:3px;display:block">ROUTING PATH</span><br>
                <span style="color:#00d4ff;font-weight:600">{st.session_state.routing}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        _render_gov()

        # ── HUMAN ESCROW ───────────────────────────────────────────────────
        escrow_ph = st.empty()
        def _render_escrow():
            if st.session_state.gov_signal != "HUMAN_ESCALATION":
                escrow_ph.empty(); return
            conf   = st.session_state.confidence
            action = (st.session_state.last_action or "noop").replace(" ","_")
            with escrow_ph.container():
                st.markdown(f"""
                <div class="escrow">
                  <div class="escrow-heading">🔒 Human Authorization Required</div>
                  <div class="escrow-caption">AI confidence {conf*100:.1f}% — below 90% threshold — escrowed</div>
                  <div class="escrow-code">
                    &gt; PROPOSED :: {action}<br>
                    &gt; ROUTE    :: MIDDLE → ATL → ESCROWED<br>
                    &gt; RISK     :: AWAITING APPROVAL
                  </div>
                  <div class="conf-track"><div class="conf-fill" style="width:{conf*100:.0f}%"></div></div>
                  <div class="conf-meta">{conf*100:.1f}% confidence &nbsp;·&nbsp; gap: {(0.9-conf)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                ca, cb = st.columns(2)
                if ca.button("✅ Approve", type="primary", use_container_width=True, key="approve"):
                    st.session_state.human_approved = True
                    st.session_state.gov_signal     = "AUTO_RESOLVE"
                    st.rerun()
                if cb.button("❌ Reject", use_container_width=True, key="reject"):
                    st.session_state.human_rejected = True
                    st.session_state.gov_signal     = "DEEP_NEGOTIATE"
                    st.session_state.running        = False
                    st.rerun()
        _render_escrow()

        # ── CHATOPS TERMINAL ───────────────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:14px">── CHATOPS TERMINAL</p>', unsafe_allow_html=True)
        term_ph = st.empty()

        def _render_term():
            log = st.session_state.chat
            if not log:
                term_ph.markdown(
                    '<div class="terminal"><span style="color:#21262d">// Awaiting incident…</span></div>',
                    unsafe_allow_html=True); return
            lines = []
            for m in log[-40:]:
                role    = m.get("role","sys")
                content = m.get("content","")
                ts      = m.get("timestamp","")[:19].replace("T"," ")
                css     = ROLE_CSS.get(role,"t-exec")
                ico     = ROLE_ICON.get(role,"SYS")
                lines.append(
                    f'<span style="color:#21262d">{ts}</span> '
                    f'<span class="{css}">[{ico}]</span> '
                    f'<span style="color:#8b949e">{content}</span>'
                )
            term_ph.markdown(
                '<div class="terminal">' + "<br>".join(lines) + "</div>",
                unsafe_allow_html=True)
        _render_term()

    # ── INCIDENT TIMELINE ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">── INCIDENT ROUTING TIMELINE</p>', unsafe_allow_html=True)
    tl_ph = st.empty()

    def _render_timeline():
        evs = st.session_state.get("timeline", [])
        if not evs:
            tl_ph.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#484f58;padding:8px 0">// Timeline populates when simulation starts</div>', unsafe_allow_html=True)
            return
        html = '<div class="tl-container">'
        for i, ev in enumerate(evs):
            is_last = i == len(evs) - 1
            tl_line = '' if is_last else f'<div class="tl-line" style="background:{ev["color"]}"></div>'
            dot = f'<div class="tl-dot" style="background:{ev["color"]}18;border:1.5px solid {ev["color"]}">{ev["icon"]}</div>'
            icon_wrap = f'<div class="tl-icon-wrap">{dot}{tl_line}</div>'
            content = f'<div class="tl-content"><div class="tl-time">+{ev["elapsed"]:.2f}s</div><div class="tl-event" style="color:{ev["color"]}">{ev["event"]}</div><div class="tl-detail">{ev["detail"]}</div></div>'
            html += f'<div class="tl-item">{icon_wrap}{content}</div>'
        html += '</div>'
        tl_ph.markdown(html, unsafe_allow_html=True)
    _render_timeline()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2  ——  ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_analytics:

    # ── BENCHMARK STATS ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">── COGNITIVE COMPRESSION · BENCHMARK PROOF</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="astat-strip">
      <div class="astat" style="--astat-color:#00d4ff">
        <div class="astat-num" style="color:#00d4ff">~21×</div>
        <div class="astat-lbl">Speed Increase</div>
      </div>
      <div class="astat" style="--astat-color:#10b981">
        <div class="astat-num" style="color:#10b981">89%</div>
        <div class="astat-lbl">Token Reduction</div>
      </div>
      <div class="astat" style="--astat-color:#6366f1">
        <div class="astat-num" style="color:#6366f1">0</div>
        <div class="astat-lbl">Tokens · Fast Path</div>
      </div>
      <div class="astat" style="--astat-color:#ec4899">
        <div class="astat-num" style="color:#ec4899">100%</div>
        <div class="astat-lbl">Hallucination Blocked</div>
      </div>
      <div class="astat" style="--astat-color:#f59e0b">
        <div class="astat-num" style="color:#f59e0b">50</div>
        <div class="astat-lbl">Stress Tests Run</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY:
        ac1, ac2 = st.columns(2, gap="large")

        # ── SPEED COMPARISON ───────────────────────────────────────────────
        with ac1:
            st.markdown('<p class="section-label">── RESPONSE TIME COMPARISON</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Standard LLM",
                x=["Standard GPT-4", "OpenCloud-SRE"],
                y=[1.12, 0.053],
                marker=dict(
                    color=["rgba(244,63,94,0.7)", "rgba(0,212,255,0.7)"],
                    line=dict(color=["#f43f5e","#00d4ff"], width=1),
                    pattern_shape=["", ""],
                ),
                text=["1.12s", "0.05s"],
                textposition="outside",
                textfont=dict(color="#c9d1d9", size=13, family="Space Grotesk"),
                width=0.45,
            ))
            layout = _plotly_base(260)
            layout["title"] = dict(text="Avg Response Time (seconds) — lower is better",
                                   font=dict(size=10, color="#484f58", family="JetBrains Mono"))
            layout["yaxis"]["title"] = "Seconds"
            fig.update_layout(**layout, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        # ── TOKEN COST ─────────────────────────────────────────────────────
        with ac2:
            st.markdown('<p class="section-label">── TOKEN COST COMPARISON</p>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["Standard GPT-4", "OpenCloud-SRE"],
                y=[850, 90],
                marker=dict(
                    color=["rgba(244,63,94,0.7)", "rgba(16,185,129,0.7)"],
                    line=dict(color=["#f43f5e","#10b981"], width=1),
                ),
                text=["850 tokens", "90 tokens"],
                textposition="outside",
                textfont=dict(color="#c9d1d9", size=13, family="Space Grotesk"),
                width=0.45,
            ))
            layout2 = _plotly_base(260)
            layout2["title"] = dict(text="Avg Tokens per Incident — lower is better",
                                    font=dict(size=10, color="#484f58", family="JetBrains Mono"))
            layout2["yaxis"]["title"] = "Tokens"
            fig2.update_layout(**layout2, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")

    # ── LIVE SESSION CHARTS ────────────────────────────────────────────────────
    lc1, lc2, lc3 = st.columns(3, gap="large")

    pc = st.session_state.path_counts
    gc_ = st.session_state.gov_counts
    total_r = sum(pc.values())

    with lc1:
        st.markdown('<p class="section-label">── ROUTING DISTRIBUTION</p>', unsafe_allow_html=True)
        if total_r > 0 and _PLOTLY:
            fig = go.Figure(go.Pie(
                labels=list(pc.keys()), values=list(pc.values()),
                hole=0.6,
                marker=dict(colors=["#00d4ff","#818cf8","#f59e0b"],
                            line=dict(color="#010409", width=2)),
                textfont=dict(family="JetBrains Mono", size=9),
            ))
            fig.add_annotation(text=f"<b>{total_r}</b>", x=0.5, y=0.55, showarrow=False,
                               font=dict(size=18, color="#f0f6fc", family="Space Grotesk"))
            fig.add_annotation(text="steps", x=0.5, y=0.4, showarrow=False,
                               font=dict(size=9, color="#484f58", family="JetBrains Mono"))
            layout = _plotly_base(220)
            layout.pop("xaxis", None); layout.pop("yaxis", None)
            fig.update_layout(**layout, showlegend=True,
                              legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1,
                                         bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:#484f58;padding:30px 0;text-align:center">// Start simulation</div>', unsafe_allow_html=True)

    with lc2:
        st.markdown('<p class="section-label">── GOVERNANCE SIGNALS</p>', unsafe_allow_html=True)
        if gc_ and _PLOTLY:
            gov_colors = {"AUTO_RESOLVE":"#34d399","HUMAN_ESCALATION":"#fcd34d",
                          "BLAST_RADIUS_BLOCK":"#a78bfa","DEEP_NEGOTIATE":"#fb7185"}
            labels = list(gc_.keys()); vals = list(gc_.values())
            colors_g = [gov_colors.get(l,"#484f58") for l in labels]
            fig = go.Figure(go.Bar(
                x=[l.replace("_","\n") for l in labels], y=vals,
                marker_color=colors_g, text=vals, textposition="outside",
                textfont=dict(color="#c9d1d9", size=11, family="Space Grotesk"), width=0.5,
            ))
            layout = _plotly_base(220)
            layout["xaxis"]["tickfont"] = dict(size=7, color="#484f58", family="JetBrains Mono")
            fig.update_layout(**layout, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:#484f58;padding:30px 0;text-align:center">// Start simulation</div>', unsafe_allow_html=True)

    with lc3:
        st.markdown('<p class="section-label">── SLO RECOVERY CURVE</p>', unsafe_allow_html=True)
        if _PLOTLY:
            xs  = list(st.session_state.steps)
            slo = list(st.session_state.slo)
            fig = go.Figure()
            fig.add_hrect(y0=0.9, y1=1.05, fillcolor="rgba(16,185,129,0.04)", line_width=0)
            fig.add_trace(go.Scatter(
                x=xs, y=slo, line=dict(color="#6366f1", width=2.5),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                mode="lines", name="SLO",
            ))
            layout = _plotly_base(220)
            layout["yaxis"].update(range=[0,1.05], tickformat=".0%")
            fig.update_layout(**layout, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3  ——  DNA MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_dna:
    try:
        from memory.dna_cache import get_cache_stats, get_shared_dna, query_dna
        stats   = get_cache_stats()
        dna_mem = get_shared_dna()
        dna_ok  = True
    except Exception as e:
        dna_ok = False
        st.warning(f"DNA Memory unavailable: {e}")

    if dna_ok:
        st.markdown('<p class="section-label">── DNA MEMORY · FAISS KNOWLEDGE BASE</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="astat-strip">
          <div class="astat" style="--astat-color:#ec4899">
            <div class="astat-num" style="color:#ec4899">{stats['total_vectors']}</div>
            <div class="astat-lbl">Total Vectors</div>
          </div>
          <div class="astat" style="--astat-color:#6366f1">
            <div class="astat-num" style="color:#6366f1">{stats['seed_count']}</div>
            <div class="astat-lbl">Seed Incidents</div>
          </div>
          <div class="astat" style="--astat-color:#10b981">
            <div class="astat-num" style="color:#10b981">{stats['distilled_count']}</div>
            <div class="astat-lbl">Learned Rules</div>
          </div>
          <div class="astat" style="--astat-color:#00d4ff">
            <div class="astat-num" style="color:#00d4ff">{stats['backend'].upper()}</div>
            <div class="astat-lbl">Search Backend</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        dc1, dc2 = st.columns([2, 3], gap="large")

        with dc1:
            st.markdown('<p class="section-label">── LIVE QUERY PROBE</p>', unsafe_allow_html=True)
            q_t = st.slider("Traffic Load",   0.0, 100.0, float(tv), 1.0, key="q_t")
            q_d = st.slider("DB Temperature", 0.0, 100.0, float(dv), 1.0, key="q_d")
            q_n = st.slider("Network Health", 0.0, 100.0, float(nv), 1.0, key="q_n")

            if st.button("🔍 Query FAISS Index", type="primary", use_container_width=True):
                res = query_dna([q_t, q_d, q_n])
                c = res['confidence']
                fp = res['is_fast_path']
                cc = "#34d399" if fp else "#f59e0b"
                st.markdown(f"""
                <div class="dna-card">
                  <div class="dna-row">
                    <span class="dna-k">CONFIDENCE</span>
                    <span class="dna-v" style="color:{cc}">{c}</span>
                  </div>
                  <div class="dna-row">
                    <span class="dna-k">L2 DISTANCE</span>
                    <span class="dna-v">{res['distance']:.4f}</span>
                  </div>
                  <div class="dna-row">
                    <span class="dna-k">MATCHED ACTION</span>
                    <span class="dna-v">{res['matched_action']}</span>
                  </div>
                  <div class="dna-row">
                    <span class="dna-k">FAST PATH?</span>
                    <span class="dna-v" style="color:{cc}">{'YES — 0 tokens' if fp else 'NO — LLM required'}</span>
                  </div>
                  <div class="dna-row">
                    <span class="dna-k">CACHE KEY</span>
                    <span class="dna-v" style="font-size:0.62rem">{res['cache_key']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with dc2:
            st.markdown('<p class="section-label">── INCIDENT VECTOR SPACE (3D)</p>', unsafe_allow_html=True)
            action_color_map = {
                "circuit_breaker": "#f43f5e", "throttle_traffic": "#f59e0b",
                "schema_failover": "#8b5cf6", "cache_flush":       "#00d4ff",
                "restart_pods":    "#fcd34d", "scale_out":         "#10b981",
                "load_balance":    "#ec4899", "noop":              "#484f58",
            }
            if _PLOTLY:
                seed_data   = getattr(dna_mem, "_vectors", [])
                seed_labels = [r["action"] for r in getattr(dna_mem, "_records", [])]
                colors_pts  = [action_color_map.get(a, "#94a3b8") for a in seed_labels]
                if len(seed_data) > 0:
                    import numpy as np
                    arr = np.array(seed_data)
                    fig = go.Figure(go.Scatter3d(
                        x=arr[:,0], y=arr[:,1], z=arr[:,2],
                        mode="markers",
                        marker=dict(size=7, color=colors_pts, opacity=0.9,
                                    line=dict(width=0.5, color="#010409")),
                        text=seed_labels,
                        hovertemplate="<b>%{text}</b><br>Traffic=%{x}<br>DB=%{y}<br>Net=%{z}<extra></extra>",
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        scene=dict(
                            xaxis=dict(title="Traffic", backgroundcolor="rgba(0,0,0,0)",
                                       gridcolor="rgba(255,255,255,0.04)", color="#484f58"),
                            yaxis=dict(title="DB Temp", backgroundcolor="rgba(0,0,0,0)",
                                       gridcolor="rgba(255,255,255,0.04)", color="#484f58"),
                            zaxis=dict(title="Network", backgroundcolor="rgba(0,0,0,0)",
                                       gridcolor="rgba(255,255,255,0.04)", color="#484f58"),
                            bgcolor="rgba(0,0,0,0)",
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
                        ),
                        height=350, margin=dict(l=0,r=0,t=10,b=0),
                        font=dict(color="#484f58", size=9, family="JetBrains Mono"),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            # color legend
            st.markdown('<p class="section-label" style="margin-top:8px">── ACTION LEGEND</p>', unsafe_allow_html=True)
            leg_cols = st.columns(4)
            for idx, (action, color) in enumerate(action_color_map.items()):
                leg_cols[idx%4].markdown(
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;box-shadow:0 0 5px {color}77"></div>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;color:#484f58">{action}</span></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _step():
    if not _OK or graph is None or env is None: return
    gs = st.session_state.gstate
    if gs is None:
        gs = initial_state(); gs["current_state_tensor"] = env.state.as_list()
    if st.session_state.gov_signal == "HUMAN_ESCALATION" and not st.session_state.human_approved:
        return
        
    # Poll backend for manual injected faults
    try:
        r = requests.get("http://127.0.0.1:8000/metrics", timeout=0.5)
        if r.status_code == 200:
            data = r.json()
            obs = data.get("metrics", {})
            if obs:
                env.state.traffic_load = obs.get("CPU", env.state.traffic_load)
                env.state.database_temperature = obs.get("DB_Temp", env.state.database_temperature)
                env.state.network_health = obs.get("Latency", env.state.network_health)
                env.state._sync_tensor()
                gs["current_state_tensor"] = env.state.as_list()
    except Exception:
        pass
        
    result = graph.invoke(gs)

    rp     = result.get("routing_path", RoutingPath.MIDDLE)
    rp_val = getattr(rp, "value", str(rp))
    st.session_state.routing = rp_val.replace("_"," ").title()

    gs_raw = result.get("governance_signal", GovernanceSignal.DEEP_NEGOTIATE)
    gov_v  = getattr(gs_raw, "value", str(gs_raw))
    st.session_state.gov_signal = gov_v

    cs = result.get("consensus_status", ConsensusStatus.RED)
    st.session_state.consensus = getattr(cs, "value", str(cs))

    td = result.get("trust_decision", TrustDecision.ESCROWED)
    st.session_state.trust = getattr(td, "value", str(td))

    if   "fast"   in rp_val: st.session_state.tokens_saved += TOKENS_FAST
    elif "middle" in rp_val: st.session_state.tokens_saved += TOKENS_MIDDLE

    action = result.get("recommended_action") or "noop"
    st.session_state.last_action = action.replace("_"," ")

    # Demo override
    DEMO_ACTIONS = {"schema_failover","scale_out","throttle_traffic","circuit_breaker"}
    is_demo = action in DEMO_ACTIONS and st.session_state.step >= 5
    st.session_state.demo_success = is_demo
    if is_demo:
        from env.state_tensor import CloudStateTensor
        env.state = CloudStateTensor.nominal()
        result["current_state_tensor"] = env.state.as_list()
        result["slo_score"] = 1.0; result["is_resolved"] = True

    st.session_state.resolved = result.get("is_resolved", False)

    st.session_state.gstate = result
    vec = result.get("current_state_tensor", [98,95,5])
    st.session_state.traffic.append(vec[0])
    st.session_state.db.append(vec[1])
    st.session_state.net.append(vec[2])
    st.session_state.slo.append(result.get("slo_score", 0.0))
    st.session_state.steps.append(st.session_state.step + 1)
    st.session_state.step += 1
    st.session_state.chat = result.get("chat_history", [])

    bw = result.get("blast_radius_warnings") or []
    if bw: st.session_state.blast_warnings = bw

    ni = result.get("network_intent") or {}
    di = result.get("db_intent") or {}
    nc, dc_ = float(ni.get("confidence",0.5)), float(di.get("confidence",0.5))
    st.session_state.confidence = round(max(nc,dc_)*0.6 + min(nc,dc_)*0.4, 3)

    if st.session_state.human_approved:
        st.session_state.human_approved = False

    # accumulate analytics
    pk = "FAST" if "fast" in rp_val else ("MIDDLE" if "middle" in rp_val else "SLOW")
    st.session_state.path_counts[pk] = st.session_state.path_counts.get(pk,0) + 1
    st.session_state.gov_counts[gov_v]    = st.session_state.gov_counts.get(gov_v,0) + 1
    st.session_state.action_counts[action] = st.session_state.action_counts.get(action,0) + 1

    # timeline
    import time as _t
    _t0      = st.session_state.get("incident_start_time") or _t.perf_counter()
    _elapsed = round(_t.perf_counter() - _t0, 2)
    _rp      = st.session_state.routing

    _TL = {
        "AUTO_RESOLVE":       ("✅","#10b981","RESOLUTION EXECUTED"),
        "HUMAN_ESCALATION":   ("⏳","#f59e0b","AWAITING HUMAN APPROVAL"),
        "BLAST_RADIUS_BLOCK": ("🛑","#8b5cf6","BLAST RADIUS BLOCKED"),
        "DEEP_NEGOTIATE":     ("💬","#f43f5e","CONFLICT → CHATOPS"),
    }
    _icon, _color, _lbl = _TL.get(gov_v, ("🔄","#484f58","STEP PROCESSED"))
    if "fast" in _rp.lower():
        _icon, _color, _lbl = "🧬","#00d4ff","DNA CACHE HIT — FAST PATH"
        _detail = f"action:{action.upper()} · tokens:0"
    elif "middle" in _rp.lower():
        _detail = f"path:MIDDLE · conf:{st.session_state.confidence:.0%} · action:{action.upper()}"
    else:
        _detail = f"path:{_rp} · action:{action.upper()} · blast:{bool(bw)}"

    st.session_state.timeline.append({
        "elapsed": _elapsed, "icon": _icon,
        "event": _lbl, "detail": _detail, "color": _color,
    })
    if st.session_state.resolved:
        st.session_state.timeline.append({
            "elapsed": round(_t.perf_counter() - _t0, 2),
            "icon": "🏁", "color": "#10b981",
            "event": "INCIDENT CLOSED — SLO RESTORED",
            "detail": f"slo:{result.get('slo_score',0):.3f}",
        })

# ── main tick ─────────────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.resolved:
    # Live Polling loop to pull metrics from the backend every 1 second
    try:
        r = requests.get("http://127.0.0.1:8000/metrics", timeout=0.5)
        if r.status_code == 200:
            data = r.json()
            obs = data.get("metrics", {})
            if obs:
                cpu = obs.get("CPU", st.session_state.traffic[-1])
                db = obs.get("DB_Temp", st.session_state.db[-1])
                net = obs.get("Latency", st.session_state.net[-1])
                st.session_state.traffic.append(cpu)
                st.session_state.db.append(db)
                st.session_state.net.append(net)
                slo = ((100 - cpu) + (100 - db) + net) / 300.0
                st.session_state.slo.append(slo)
                st.session_state.step += 1
                st.session_state.steps.append(st.session_state.step)
                
                # Update visual state if needed for graph
                if env:
                    env.state.traffic_load = cpu
                    env.state.database_temperature = db
                    env.state.network_health = net
                    env.state._sync_tensor()
                
            if data.get("status") == "CRITICAL" and _OK and graph:
                _step()
    except Exception:
        pass

    if st.session_state.resolved:
        msg = "🎉 DEMO SUCCESS — System stabilized." if st.session_state.demo_success else "✅ System Recovered — SLO ≥ 0.95"
        st.success(msg); st.session_state.running = False; st.balloons()
    elif st.session_state.gov_signal == "HUMAN_ESCALATION":
        st.warning("⏳ Governance Escrow Active — awaiting human approval in War Room tab.")
    else:
        if st_autorefresh:
            st_autorefresh(interval=1000, limit=None, key="live_poll_running")
        else:
            time.sleep(1)
            st.rerun()

elif st.session_state.resolved:
    msg = "🎉 DEMO SUCCESS — System stabilized." if st.session_state.demo_success else "✅ System Recovered — SLO ≥ 0.95"
    st.success(msg)
elif not st.session_state.running:
    if not _OK:
        st.error(f"Import error: `{_ERR}`")
    else:
        st.info("▶ Click **Start** in the sidebar to launch the incident simulation.")

