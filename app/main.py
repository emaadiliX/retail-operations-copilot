import sys
import re
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from copilot_agents.orchestrator import run_pipeline  # noqa: E402
from copilot_agents.tracing import TraceLog  # noqa: E402
from copilot_agents.models import PipelineResult  # noqa: E402

STAGES = [
    ("Plan", "plan"),
    ("Research", "research"),
    ("Draft", "draft"),
    ("Verify", "verify"),
    ("Deliver", "deliver"),
]

EXAMPLE_QUERIES = [
    "What are the top supply chain visibility challenges for CPG companies, and what technologies are being adopted to address them?",
    "Summarize the state of retail returns in 2024 and recommend strategies to reduce return rates while improving customer satisfaction.",
    "What digital transformation strategies should a CPG company prioritize to stay competitive over the next 3-5 years?",
]


def init_session_state():
    for key, val in {
        "pipeline_result": None,
        "pipeline_trace": None,
        "pipeline_status": "idle",
        "pipeline_error": None,
        "user_request": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_stage_statuses():
    """Map each stage key to its current status string."""
    if st.session_state.get("pipeline_status") == "idle":
        return {k: "pending" for _, k in STAGES}
    trace = st.session_state.get("pipeline_trace")
    if not trace or not trace.entries:
        return {k: "pending" for _, k in STAGES}
    out = {}
    for _, key in STAGES:
        entry = next((e for e in trace.entries if e.stage == key), None)
        out[key] = entry.status if entry else "pending"
    return out


def inject_custom_css():
    # Google Fonts (Inter + Material Symbols)
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
        '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    st.markdown("""<style>
    /* ============================================================
       GLOBAL
       ============================================================ */
    html, body, .stApp, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: #101722 !important;
    }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    [data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #223149; border-radius: 10px; }

    /* ============================================================
       SIDEBAR  –  only color overrides, let Streamlit handle layout
       ============================================================ */
    section[data-testid="stSidebar"] {
        background: #0b101a !important;
        width: 21rem !important;
        min-width: 21rem !important;
        transform: none !important;
    }
    section[data-testid="stSidebar"] > div {
        background: #0b101a !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
        margin: 0.5rem 0 !important;
    }

    /* sidebar toggle arrows – brighter for dark theme */
    [data-testid="stHeader"] button svg {
        color: #94a3b8 !important;
    }
    [data-testid="stHeader"] button:hover svg {
        color: #e2e8f0 !important;
    }
    [data-testid="collapsedControl"] svg {
        color: #94a3b8 !important;
    }
    [data-testid="collapsedControl"]:hover svg {
        color: #e2e8f0 !important;
    }

    /* secondary buttons (example queries) */
    button[data-testid="stBaseButton-secondary"] {
        background: rgba(30,41,59,0.4) !important;
        border: 1px solid #1e293b !important;
        color: #94a3b8 !important;
        border-radius: 0.5rem !important;
        font-size: 0.8rem !important;
        text-align: left !important;
        transition: all 0.2s !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        height: 100% !important;
        padding: 0.75rem 1rem !important;
        line-height: 1.45 !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(60,131,246,0.1) !important;
        border-color: rgba(60,131,246,0.3) !important;
        color: #cbd5e1 !important;
    }

    /* equal-height columns for example queries */
    [data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) {
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > [data-testid="stColumn"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) > [data-testid="stColumn"] > div > div {
        flex: 1 !important;
        display: flex !important;
    }

    /* ============================================================
       MAIN CONTENT – dark theme overrides
       ============================================================ */
    [data-testid="stAppViewContainer"] {
        background: #101722 !important;
        color: #e2e8f0 !important;
    }

    /* text inputs */
    textarea, input[type="text"] {
        background: #0b101a !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e293b !important;
        border-radius: 0.5rem !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: #3c83f6 !important;
        box-shadow: 0 0 0 1px #3c83f6 !important;
    }
    .stTextArea label, .stTextInput label { color: #94a3b8 !important; }

    /* primary button */
    button[data-testid="stBaseButton-primary"] {
        background: #3c83f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(60,131,246,0.2) !important;
        transition: background 0.2s !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background: #2563eb !important;
    }

    /* metrics */
    div[data-testid="stMetric"] {
        background: rgba(30,41,59,0.5) !important;
        border: 1px solid #1e293b !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700 !important; }

    /* tabs */
    button[data-baseweb="tab"] { color: #64748b !important; background: transparent !important; font-weight: 500 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #3c83f6 !important; font-weight: 600 !important; }
    div[data-baseweb="tab-highlight"] { background-color: #3c83f6 !important; }
    div[data-baseweb="tab-border"] { background-color: #1e293b !important; }

    /* expanders */
    details { background: rgba(30,41,59,0.3) !important; border: 1px solid #1e293b !important; border-radius: 0.5rem !important; }
    details summary span { color: #e2e8f0 !important; }

    /* status widget */
    [data-testid="stStatusWidget"] { background: rgba(30,41,59,0.5) !important; border: 1px solid #1e293b !important; }

    /* dividers */
    hr { border-color: #1e293b !important; }

    /* markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #cbd5e1 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #f1f5f9 !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #64748b !important; }

    /* alerts */
    [data-testid="stAlert"] { border-radius: 0.5rem !important; }

    /* ============================================================
       COMPONENT CLASSES (used by render helpers)
       ============================================================ */
    .verdict-pass {
        display:inline-block; background:#22c55e; color:#fff !important;
        padding:0.4rem 1.2rem; border-radius:50px; font-weight:600;
        font-size:0.9rem; letter-spacing:0.03em;
    }
    .verdict-partial {
        display:inline-block; background:#f59e0b; color:#fff !important;
        padding:0.4rem 1.2rem; border-radius:50px; font-weight:600;
        font-size:0.9rem; letter-spacing:0.03em;
    }
    .verdict-fail {
        display:inline-block; background:#ef4444; color:#fff !important;
        padding:0.4rem 1.2rem; border-radius:50px; font-weight:600;
        font-size:0.9rem; letter-spacing:0.03em;
    }
    .finding-card {
        border-left:3px solid #3c83f6; padding:0.8rem 1rem;
        margin:0.5rem 0; background:rgba(60,131,246,0.06);
        border-radius:0 0.5rem 0.5rem 0;
    }
    .finding-card p { margin:0 0 0.4rem 0; color:#cbd5e1 !important; }
    .citation-pill {
        display:inline-block; background:rgba(60,131,246,0.15);
        color:#93bbfc !important; padding:0.12rem 0.55rem;
        border-radius:12px; font-size:0.72rem; font-weight:500;
    }
    .focus-pill {
        display:inline-block; background:rgba(60,131,246,0.12);
        color:#93bbfc !important; padding:0.2rem 0.65rem;
        border-radius:12px; font-size:0.78rem; font-weight:500;
        margin:0.12rem 0.2rem;
    }
    .claim-supported {
        border-left:3px solid #22c55e; padding:0.7rem 1rem;
        margin:0.4rem 0; background:rgba(34,197,94,0.06);
        border-radius:0 0.5rem 0.5rem 0;
    }
    .claim-partial {
        border-left:3px solid #f59e0b; padding:0.7rem 1rem;
        margin:0.4rem 0; background:rgba(245,158,11,0.06);
        border-radius:0 0.5rem 0.5rem 0;
    }
    .claim-unsupported {
        border-left:3px solid #ef4444; padding:0.7rem 1rem;
        margin:0.4rem 0; background:rgba(239,68,68,0.06);
        border-radius:0 0.5rem 0.5rem 0;
    }
    .trace-card {
        border:1px solid #1e293b; border-radius:0.5rem;
        padding:0.8rem 1rem; margin:0.4rem 0;
        background:rgba(30,41,59,0.3);
    }
    .trace-duration {
        display:inline-block; background:#3c83f6; color:#fff !important;
        padding:0.1rem 0.5rem; border-radius:6px;
        font-size:0.72rem; font-weight:600;
    }
    .email-card {
        border-left:3px solid #3c83f6; padding:1.1rem 1.3rem;
        background:rgba(30,41,59,0.4); border-radius:0 0.5rem 0.5rem 0;
        color:#cbd5e1 !important;
    }
    </style>""", unsafe_allow_html=True)


_STAGE_CFG = {
    "completed": ("\u2705", "#22c55e", "400"),   # green check
    "running":   ("\U0001F504", "#3c83f6", "600"),  # blue cycle
    "error":     ("\u274C", "#ef4444", "500"),    # red cross
    "pending":   ("\u25CB", "#475569", "400"),    # gray circle
}


def render_sidebar():
    with st.sidebar:
        # ---- Brand ----
        st.markdown(
            '<h3 style="margin:0 0 2px;color:#e2e8f0;">'
            '\U0001F916 Retail Copilot</h3>',
            unsafe_allow_html=True,
        )
        st.caption("MULTI-AGENT SYSTEM")

        st.divider()

        # ---- Nav ----
        st.markdown(
            '<p style="margin:0;padding:6px 10px;border-radius:6px;'
            'background:rgba(60,131,246,0.1);color:#3c83f6;'
            'font-weight:500;font-size:0.875rem;">'
            '\U0001F3E0 Dashboard</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ---- Workflow Phase ----
        st.markdown(
            '<p style="margin:0 0 8px;font-size:10px;font-weight:700;'
            'color:#94a3b8;text-transform:uppercase;letter-spacing:0.1em;">'
            'Workflow Phase</p>',
            unsafe_allow_html=True,
        )

        statuses = _get_stage_statuses()
        for label, key in STAGES:
            s = statuses[key]
            icon, color, weight = _STAGE_CFG.get(s, _STAGE_CFG["pending"])
            opacity = "opacity:0.5;" if s == "pending" else ""
            bar = ""
            if s in ("running", "error"):
                bar = (
                    f'<span style="display:inline-block;width:3px;height:100%;'
                    f'background:{color};border-radius:9999px;'
                    f'position:absolute;left:0;top:0;"></span>'
                )
            st.markdown(
                f'<p style="margin:0;padding:5px 8px;position:relative;{opacity}">'
                f'{bar}'
                f'<span style="color:{color};margin-right:6px;">{icon}</span>'
                f'<span style="color:{color};font-weight:{weight};'
                f'font-size:0.875rem;">{label}</span></p>',
                unsafe_allow_html=True,
            )

        # ---- Duration badge ----
        trace = st.session_state.get("pipeline_trace")
        if trace and trace.get_total_duration() > 0:
            dur = trace.get_total_duration()
            st.markdown(
                f'<p style="margin:8px 0 0;padding:5px 8px;font-size:0.75rem;'
                f'color:#94a3b8;background:rgba(60,131,246,0.08);'
                f'border-radius:6px;">'
                f'\u23F1 Completed in '
                f'<strong style="color:#e2e8f0;">{dur:.1f}s</strong></p>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ---- Footer ----
        st.markdown(
            '<p style="margin:0;font-size:0.75rem;color:#e2e8f0;font-weight:500;">'
            '\u2699\uFE0F GPT-4o-mini</p>'
            '<p style="margin:2px 0 0;font-size:10px;color:#64748b;">'
            '12 docs &middot; ChromaDB</p>',
            unsafe_allow_html=True,
        )


def render_verdict_badge(verdict: str):
    v = verdict.strip().upper()
    if v == "PASS":
        cls, label = "verdict-pass", "PASS — All Claims Verified"
    elif v == "PARTIAL":
        cls, label = "verdict-partial", "PARTIAL — Some Claims Need Review"
    else:
        cls, label = "verdict-fail", "FAIL — Unsupported Claims Detected"
    st.markdown(f'<span class="{cls}">{label}</span>', unsafe_allow_html=True)


def render_executive_summary(deliverable):
    wc = len(deliverable.executive_summary.split())
    st.markdown("#### Executive Summary")
    st.markdown(deliverable.executive_summary)
    st.caption(f"{wc} / 150 words")


def render_client_email(deliverable):
    st.markdown("#### Client-Ready Email")
    safe = _escape_html(deliverable.client_email)
    st.markdown(
        f'<div class="email-card">{safe}</div>', unsafe_allow_html=True)
    with st.expander("Copy-friendly plain text"):
        st.code(deliverable.client_email, language=None)


def render_action_items(deliverable):
    st.markdown("#### Action Items")
    if not deliverable.action_items:
        st.info("No action items generated.")
        return
    rows = []
    for i, item in enumerate(deliverable.action_items, 1):
        conf = item.confidence.strip().capitalize()
        dot = {"High": "\U0001F7E2", "Medium": "\U0001F7E1"}.get(
            conf.split()[0] if conf else "", "\U0001F534")
        rows.append({"#": i, "Action": item.action, "Owner": item.owner,
                     "Due Date": item.due_date, "Confidence": f"{dot} {conf}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_sources(deliverable):
    st.markdown("#### Sources & Citations")
    if not deliverable.sources:
        st.info("No sources recorded.")
        return
    for i, s in enumerate(deliverable.sources, 1):
        st.markdown(f"**{i}.** {s}")


def render_research_findings(research):
    st.markdown("#### Research Summary")
    st.info(research.summary)
    st.markdown(f"##### Findings ({len(research.findings)})")
    for f in research.findings:
        st.markdown(
            f'<div class="finding-card"><p>{_escape_html(f.finding)}</p>'
            f'<span class="citation-pill">{_escape_html(f.citation)}</span>'
            f'<br><small style="color:#64748b;font-style:italic;">{_escape_html(f.relevance)}</small></div>',
            unsafe_allow_html=True,
        )
    if research.gaps:
        st.markdown("##### Information Gaps")
        st.warning("The following was **not found in sources**:")
        for g in research.gaps:
            st.markdown(f"- {g}")
    if research.sources_used:
        st.markdown("##### All Sources Referenced")
        for i, s in enumerate(research.sources_used, 1):
            st.markdown(f"{i}. {s}")


def render_planning_details(plan):
    st.markdown("#### Execution Plan")
    st.markdown(f"**Task Summary:** {plan.task_summary}")
    st.markdown("##### Sub-Tasks")
    for i, t in enumerate(plan.sub_tasks, 1):
        st.markdown(f"{i}. {t}")
    st.markdown("##### Research Queries")
    for q in plan.research_queries:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Q:** {q.query}")
        with c2:
            st.caption(f"Purpose: {q.purpose}")
    if plan.focus_areas:
        st.markdown("##### Focus Areas")
        st.markdown(
            " ".join(
                f'<span class="focus-pill">{_escape_html(a)}</span>' for a in plan.focus_areas),
            unsafe_allow_html=True,
        )


def render_verification_details(verification):
    st.markdown("#### Verification Report")
    render_verdict_badge(verification.overall_verdict)
    st.markdown("")
    st.markdown(
        f"##### Claim Analysis ({len(verification.verified_claims)} claims)")
    for c in verification.verified_claims:
        v = c.verdict.strip().upper()
        if "NOT SUPPORTED" in v:
            cls, badge = "claim-unsupported", "NOT SUPPORTED"
        elif "PARTIALLY" in v:
            cls, badge = "claim-partial", "PARTIALLY SUPPORTED"
        else:
            cls, badge = "claim-supported", "SUPPORTED"
        srcs = ", ".join(
            c.supporting_sources) if c.supporting_sources else "None"
        st.markdown(
            f'<div class="{cls}"><strong>{_escape_html(c.claim)}</strong><br>'
            f'<span class="citation-pill">{badge}</span><br>'
            f'<small style="color:#64748b;">Sources: {_escape_html(srcs)}</small><br>'
            f'<small style="color:#64748b;font-style:italic;">{_escape_html(c.explanation)}</small></div>',
            unsafe_allow_html=True,
        )
    if verification.unsupported_claims:
        st.error("**Unsupported Claims Detected:**")
        for c in verification.unsupported_claims:
            st.markdown(f"- {c}")
    if verification.suggestions:
        st.info("**Verifier Suggestions:**")
        for s in verification.suggestions:
            st.markdown(f"- {s}")


def render_trace_log(trace):
    st.markdown("#### Agent Trace Log")
    icons = {"completed": "\u2705", "error": "\u274C",
             "running": "\u23F3", "pending": "\u23F8"}
    for i, e in enumerate(trace.entries, 1):
        ic = icons.get(e.status, "\u2753")
        st.markdown(
            f'<div class="trace-card"><strong>{ic} Step {i}: {e.agent_name}</strong>'
            f'&nbsp;&nbsp;<span class="citation-pill">{e.stage}</span>'
            f'&nbsp;&nbsp;<span class="trace-duration">{e.duration_seconds:.2f}s</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"Details — {e.agent_name}", expanded=False):
            if e.input_preview:
                st.markdown("**Input preview:**")
                st.text(e.input_preview[:300])
            if e.output_preview:
                st.markdown("**Output preview:**")
                st.text(e.output_preview[:400])
            if e.error_message:
                st.error(f"Error: {e.error_message}")
            if e.metadata:
                st.markdown("**Metadata:**")
                st.json(e.metadata)
    st.metric("Total Pipeline Duration", f"{trace.get_total_duration():.1f}s")
    with st.expander("Raw Trace JSON"):
        st.json(trace.to_list())


def render_summary_metrics(result, trace):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Time", f"{trace.get_total_duration():.1f}s")
    with c2:
        st.metric("Findings", str(len(result.research.findings)))
    with c3:
        st.metric("Claims Verified", str(
            len(result.verification.verified_claims)))
    with c4:
        st.metric("Action Items", str(
            len(result.final_deliverable.action_items)))


def _escape_html(text: str) -> str:
    """Escape HTML entities and convert newlines to <br>."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Retail Copilot",
        page_icon="\U0001F916",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_custom_css()
    render_sidebar()

    # ---- Header ----
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:between;padding:0.2rem 0 1rem;">'
        '<div>'
        '<h2 style="margin:0;font-size:1.25rem;font-weight:600;color:#f1f5f9 !important;">Workflow Dashboard</h2>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Input ----
    user_request = st.text_area(
        "Enter your business question",
        value=st.session_state.get("user_request", ""),
        height=100,
        placeholder="e.g., Analyze omnichannel retail challenges and recommend strategies for improving inventory accuracy...",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "\u25B6  Run Workflow",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state["pipeline_status"] == "running"),
        )

    # ---- Example Queries ----
    if st.session_state["pipeline_status"] == "idle":
        st.markdown(
            '<p style="margin:1.2rem 0 0.6rem;font-size:0.7rem;font-weight:600;'
            'color:#475569;text-transform:uppercase;letter-spacing:0.08em;">'
            'Try an example</p>',
            unsafe_allow_html=True,
        )
        eq_cols = st.columns(len(EXAMPLE_QUERIES))
        for i, q in enumerate(EXAMPLE_QUERIES):
            with eq_cols[i]:
                if st.button(q, key=f"eq_{i}", use_container_width=True):
                    st.session_state["user_request"] = q
                    st.rerun()

    # ---- Execute pipeline ----
    if run_clicked and user_request.strip():
        st.session_state["user_request"] = user_request.strip()
        st.session_state["pipeline_status"] = "running"
        st.session_state["pipeline_result"] = None
        st.session_state["pipeline_error"] = None

        with st.status("Running multi-agent analysis pipeline...", expanded=True):
            st.markdown(
                "**Stages:** Plan \u2192 Research \u2192 Draft \u2192 Verify \u2192 Deliver")
            st.markdown(
                "_Typically 30\u201390 seconds. Each agent is grounded in 12 retail & CPG source documents._")

            trace = TraceLog()
            try:
                result = run_pipeline(user_request.strip(), trace=trace)
                st.session_state["pipeline_result"] = result
                st.session_state["pipeline_trace"] = trace
                st.session_state["pipeline_status"] = "completed"
                st.session_state["pipeline_error"] = None
            except Exception as e:
                st.session_state["pipeline_status"] = "error"
                st.session_state["pipeline_error"] = str(e)
                st.session_state["pipeline_trace"] = trace

        # Rerun so sidebar workflow indicators update
        st.rerun()

    elif run_clicked and not user_request.strip():
        st.warning("Please enter a business question.")

    # ---- Error display ----
    if st.session_state["pipeline_status"] == "error":
        st.error(f"**Pipeline Error:** {st.session_state['pipeline_error']}")
        if st.session_state.get("pipeline_trace"):
            with st.expander("Partial Trace Log"):
                render_trace_log(st.session_state["pipeline_trace"])

    # ---- Results ----
    if st.session_state["pipeline_status"] == "completed" and st.session_state["pipeline_result"]:
        result: PipelineResult = st.session_state["pipeline_result"]
        trace: TraceLog = st.session_state["pipeline_trace"]

        render_summary_metrics(result, trace)
        st.divider()

        vcol, _ = st.columns([1, 2])
        with vcol:
            st.markdown("#### Verification Verdict")
            render_verdict_badge(result.verification.overall_verdict)

        st.divider()

        tabs = st.tabs([
            "Executive Summary", "Client Email", "Action Items",
            "Sources & Citations", "Research Findings", "Execution Plan",
            "Verification Details", "Agent Trace Log",
        ])
        with tabs[0]:
            render_executive_summary(result.final_deliverable)
        with tabs[1]:
            render_client_email(result.final_deliverable)
        with tabs[2]:
            render_action_items(result.final_deliverable)
        with tabs[3]:
            render_sources(result.final_deliverable)
        with tabs[4]:
            render_research_findings(result.research)
        with tabs[5]:
            render_planning_details(result.plan)
        with tabs[6]:
            render_verification_details(result.verification)
        with tabs[7]:
            render_trace_log(trace)


if __name__ == "__main__":
    main()
