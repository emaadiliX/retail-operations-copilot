import sys
import re
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from copilot_agents.orchestrator import run_pipeline, check_prompt_injection, check_input_relevance  # noqa: E402
from copilot_agents.tracing import TraceLog  # noqa: E402
from copilot_agents.models import PipelineResult  # noqa: E402

# pull in eval stuff so we can grade results and show test prompts in the UI
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from run_eval import TEST_PROMPTS, grade  # noqa: E402

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
        "eval_test": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_stage_statuses():
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


def load_css():
    css_path = Path(__file__).parent / "styles.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


_STAGE_ICONS = {
    "completed": "check_circle",
    "running": "sync",
    "error": "error",
    "pending": "radio_button_unchecked",
}


def _render_workflow_html(statuses):
    items = []
    for label, key in STAGES:
        status = statuses.get(key, "pending")
        icon = _STAGE_ICONS.get(status, "radio_button_unchecked")
        items.append(
            f'<div class="wf-item" data-status="{status}">'
            f'<div class="wf-bar"></div>'
            f'<span class="material-symbols-outlined wf-icon">{icon}</span>'
            f'<span class="wf-label">{label}</span></div>'
        )

    return (
        '<div class="wf-section">'
        '<div class="wf-heading">Workflow Phase</div>'
        '<div class="wf-list">'
        + "".join(items)
        + '</div></div>'
    )


def _on_stage_update(trace):
    """Re-render sidebar workflow phases on each stage transition."""
    st.session_state["pipeline_trace"] = trace
    placeholder = st.session_state.get("_workflow_placeholder")
    if placeholder:
        statuses = {}
        for _, key in STAGES:
            entry = next((e for e in trace.entries if e.stage == key), None)
            statuses[key] = entry.status if entry else "pending"
        placeholder.markdown(
            _render_workflow_html(statuses), unsafe_allow_html=True
        )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="sidebar-brand-icon">'
            '<span class="material-symbols-outlined">smart_toy</span></div>'
            '<div>'
            '<div class="sidebar-brand-title">Retail Copilot</div>'
            '<div class="sidebar-brand-subtitle">Multi-Agent System</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-nav">'
            '<div class="sidebar-nav-link">'
            '<span class="material-symbols-outlined">home</span>'
            'Home</div></div>',
            unsafe_allow_html=True,
        )

        workflow_placeholder = st.empty()
        statuses = _get_stage_statuses()
        workflow_placeholder.markdown(
            _render_workflow_html(statuses), unsafe_allow_html=True
        )
        st.session_state["_workflow_placeholder"] = workflow_placeholder

        trace = st.session_state.get("pipeline_trace")
        if trace and trace.get_total_duration() > 0:
            dur = trace.get_total_duration()
            st.markdown(
                f'<div class="sidebar-duration">'
                f'<div class="sidebar-duration-badge">'
                f'<span class="material-symbols-outlined">timer</span>'
                f'Completed in '
                f'<strong>{dur:.1f}s</strong></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="sidebar-footer">'
            '<div class="sidebar-footer-card">'
            '<div class="sidebar-avatar">JD</div>'
            '<div>'
            '<div class="sidebar-user-name">John Doe</div>'
            '<div class="sidebar-user-role">Project Lead</div>'
            '</div></div></div>',
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


def render_research_and_sources(research, deliverable):
    st.markdown("#### Research Summary")
    st.info(research.summary)

    st.markdown("#### Sources & Citations")
    if deliverable.sources:
        for i, s in enumerate(deliverable.sources, 1):
            st.markdown(f"**{i}.** {s}")
    else:
        st.info("No sources recorded.")

    st.markdown(f"#### Findings ({len(research.findings)})")
    for f in research.findings:
        st.markdown(
            f'<div class="finding-card"><p>{_escape_html(f.finding)}</p>'
            f'<span class="citation-pill">{_escape_html(f.citation)}</span>'
            f'<br><small class="text-muted-italic">{_escape_html(f.relevance)}</small></div>',
            unsafe_allow_html=True,
        )
    if research.gaps:
        st.markdown("#### Information Gaps")
        st.warning("The following was **not found in sources**:")
        for g in research.gaps:
            st.markdown(f"- {g}")


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
            f'<small class="text-muted">Sources: {_escape_html(srcs)}</small><br>'
            f'<small class="text-muted-italic">{_escape_html(c.explanation)}</small></div>',
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


def render_eval_checks(result, test_case):
    """Show pass/fail quality checks for an eval test prompt."""
    checks = grade(result, test_case)
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)

    if passed == total:
        st.success(f"Quality checks: {passed}/{total} passed")
    else:
        st.warning(f"Quality checks: {passed}/{total} passed")

    for name, ok, detail in checks:
        icon = "\u2705" if ok else "\u274C"
        label = name.replace("_", " ")
        if detail:
            st.markdown(f"{icon} **{label}** — {detail}")
        else:
            st.markdown(f"{icon} **{label}**")


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
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text.replace("\n", "<br>")


def main():
    st.set_page_config(
        page_title="Retail Copilot",
        page_icon="\U0001F916",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    load_css()
    render_sidebar()

    st.markdown(
        '<div class="page-header"><div>'
        '<h2 class="page-title">Workflow Dashboard</h2>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    user_request = st.text_area(
        "Enter your business question",
        value=st.session_state.get("user_request", ""),
        height=100,
        placeholder="e.g., Analyze omnichannel retail challenges and recommend strategies for improving inventory accuracy...",
        label_visibility="collapsed",
    ) or ""

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "\u25B6  Run Workflow",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state["pipeline_status"] == "running"),
        )

    if st.session_state["pipeline_status"] == "idle":
        st.markdown(
            '<p class="example-label">Try an example</p>',
            unsafe_allow_html=True,
        )
        eq_cols = st.columns(len(EXAMPLE_QUERIES))
        for i, q in enumerate(EXAMPLE_QUERIES):
            with eq_cols[i]:
                if st.button(q, key=f"eq_{i}", use_container_width=True):
                    st.session_state["user_request"] = q
                    st.session_state["eval_test"] = None
                    st.rerun()

        with st.expander("Eval test prompts (10 scenarios)"):
            for t in TEST_PROMPTS:
                label = t["id"].replace("_", " ").title()
                if st.button(f"{label}: {t['query'][:90]}...",
                             key=f"eval_{t['id']}", use_container_width=True):
                    st.session_state["user_request"] = t["query"]
                    st.session_state["eval_test"] = t
                    st.rerun()

    if run_clicked and user_request.strip():
        injection = check_prompt_injection(user_request.strip())
        if injection:
            st.error("**Input Rejected:** Your input was flagged by prompt injection defense. Please rephrase as a legitimate business question about retail or CPG operations.")
            st.stop()

        relevance = check_input_relevance(user_request.strip())
        if relevance:
            st.warning(f"**Invalid Input:** {relevance}")
            st.stop()

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
                result = run_pipeline(
                    user_request.strip(),
                    trace=trace,
                    on_stage_update=_on_stage_update,
                )
                st.session_state["pipeline_result"] = result
                st.session_state["pipeline_trace"] = trace
                st.session_state["pipeline_status"] = "completed"
                st.session_state["pipeline_error"] = None
            except Exception as e:
                st.session_state["pipeline_status"] = "error"
                st.session_state["pipeline_error"] = str(e)
                st.session_state["pipeline_trace"] = trace

        st.rerun()

    elif run_clicked and not user_request.strip():
        st.warning("Please enter a business question.")

    if st.session_state["pipeline_status"] == "error":
        st.error(f"**Pipeline Error:** {st.session_state['pipeline_error']}")
        if st.session_state.get("pipeline_trace"):
            with st.expander("Partial Trace Log"):
                render_trace_log(st.session_state["pipeline_trace"])

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

        # if this run came from an eval prompt, add the quality checks tab
        eval_test = st.session_state.get("eval_test")
        tab_names = [
            "Executive Summary", "Client Email", "Action Items",
            "Research & Sources", "Execution Plan",
            "Verification Details", "Agent Trace Log",
        ]
        if eval_test:
            tab_names.append("Eval Quality Checks")

        tabs = st.tabs(tab_names)
        with tabs[0]:
            render_executive_summary(result.final_deliverable)
        with tabs[1]:
            render_client_email(result.final_deliverable)
        with tabs[2]:
            render_action_items(result.final_deliverable)
        with tabs[3]:
            render_research_and_sources(
                result.research, result.final_deliverable)
        with tabs[4]:
            render_planning_details(result.plan)
        with tabs[5]:
            render_verification_details(result.verification)
        with tabs[6]:
            render_trace_log(trace)
        if eval_test:
            with tabs[7]:
                st.markdown(f"#### Eval: {eval_test['id']}")
                render_eval_checks(result, eval_test)


if __name__ == "__main__":
    main()
