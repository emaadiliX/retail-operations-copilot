"""Runs the full Plan -> Research -> Draft -> Verify -> Deliver pipeline."""

import re
from typing import Optional, Callable

from agents import Runner

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+(your|all|the)\s+(instructions|rules|guidelines|prompts)",
    r"override\s+(your|all|the|system)\s+(instructions|rules|prompt)",
    r"forget\s+(your|all|the|everything|prior)\s+(instructions|rules|context)",
    r"do\s+not\s+follow\s+(your|the|any)\s+(instructions|rules|guidelines)",
    r"new\s+instructions?\s*:",
    r"you\s+are\s+now\s+(?!analyzing|reviewing|examining|looking)",
    r"you\s+are\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|evil|hacked|general.purpose|unlimited)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if|though)\s+you\s+(have\s+no|are\s+not)",
    r"switch\s+to\s+.{0,20}?\s+mode",
    r"\b(jailbreak|DAN|do\s+anything\s+now)\b",
    r"\b(unrestricted|unfiltered|no\s+restrictions|without\s+restrictions)\b",
    r"answer\s+anything",
    r"(repeat|print|show|reveal|output|display)\s+(your|the|system)\s+(instructions|prompt|rules)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(instructions|prompt|rules)",
    r"(your|the)\s+system\s+prompt",
    r"(share|leak|dump|expose|give\s+me)\s+(your|the)\s+(instructions|prompt|rules)",
    r"```\s*system",
    r"<\s*system\s*>",
    r"###\s*SYSTEM",
    r"\[INST\]",
    r"<\|im_start\|>",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_prompt_injection(user_input: str) -> Optional[str]:
    """Return a description of the violation if prompt injection is detected, else None."""
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(user_input)
        if match:
            return f"Blocked input: detected prompt injection pattern ({match.group()!r})"
    return None

_MIN_INPUT_WORDS = 4

_OFFTOPIC_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|sup|greetings|good\s+(morning|afternoon|evening))\s*[!.?]*\s*$",
    r"^\s*what\s+is\s+my\s+name\s*[?!.]*\s*$",
    r"^\s*who\s+am\s+i\s*[?!.]*\s*$",
    r"^\s*tell\s+me\s+(a\s+)?(joke|story|riddle)",
    r"^\s*how\s+are\s+you\s*[?!.]*\s*$",
    r"^\s*what\s+can\s+you\s+do\s*[?!.]*\s*$",
    r"^\s*thank\s*(s|\s+you)\s*[!.]*\s*$",
    r"^\s*(yes|no|ok|okay|sure|nope|bye|goodbye)\s*[!.?]*\s*$",
    r"^\s*test(ing)?\s*[!.?]*\s*$",
]

_COMPILED_OFFTOPIC = [re.compile(p, re.IGNORECASE) for p in _OFFTOPIC_PATTERNS]


def check_input_relevance(user_input: str) -> Optional[str]:
    """Return a rejection message if the input is too short or clearly off-topic, else None."""
    stripped = user_input.strip()
    word_count = len(stripped.split())

    if word_count < _MIN_INPUT_WORDS:
        return (
            f"Input too short ({word_count} words). Please enter a detailed "
            "business question about retail or CPG operations (minimum 4 words)."
        )

    for pattern in _COMPILED_OFFTOPIC:
        if pattern.search(stripped):
            return (
                "Off-topic input detected. This system handles retail and CPG "
                "business questions only. Please enter a question about supply chain, "
                "inventory, omnichannel strategy, or similar topics."
            )

    return None

from .models import (
    ExecutionPlan,
    ResearchNotes,
    Deliverable,
    VerificationReport,
    PipelineResult,
)
from .planner import planner_agent, build_planner_prompt
from .researcher import researcher_agent, build_researcher_prompt
from .writer import writer_agent, build_writer_prompt
from .verifier import verifier_agent, build_verifier_prompt
from .tracing import TraceLog


def _warm_retrieval() -> int:
    from retrieval.indexing import get_chroma_client
    from retrieval.config import COLLECTION_NAME
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Retrieval layer ready: {count} chunks in '{COLLECTION_NAME}'")
        return count
    except Exception as e:
        print(f"WARNING: Could not verify retrieval layer: {e}")
        return 0


def _build_unsupported_disclaimer(unsupported_claims: list) -> str:
    """Return disclaimer text listing unsupported claims, or empty string if none."""
    if not unsupported_claims:
        return ""
    claims_list = "\n".join(f"  - {claim}" for claim in unsupported_claims)
    return (
        "\n\n[VERIFICATION NOTICE] The following claims were Not found in sources "
        "and may not be supported by the available evidence:\n"
        f"{claims_list}\n"
        "Please verify these points independently before acting on them.\n"
        "To strengthen these areas, consider providing additional source "
        "documents covering the above topics."
    )


def _collect_verified_sources(draft_sources: list, verified_claims: list) -> list:
    """Filter draft_sources to only those backing a supported claim."""
    verified_set: set = set()
    for claim in verified_claims:
        if claim.verdict.strip().upper() in ("SUPPORTED", "PARTIALLY SUPPORTED"):
            verified_set.update(claim.supporting_sources)

    if not verified_set:
        return draft_sources

    seen = set()
    filtered = []
    for s in draft_sources:
        if s in verified_set and s not in seen:
            seen.add(s)
            filtered.append(s)
    return filtered if filtered else draft_sources


def _fetch_chunk_texts(research: ResearchNotes) -> str:
    """Look up the actual chunk text for every citation in the research notes.

    This is a fast, local ChromaDB lookup (no LLM calls) that provides the
    verifier with ground-truth text to cross-check against claims.
    """
    from retrieval.retrieval import get_collection
    from retrieval.config import COLLECTION_NAME

    collection = get_collection(COLLECTION_NAME)
    if not collection:
        return ""

    citations = list(dict.fromkeys(f.citation for f in research.findings))
    parts = []
    failed = 0
    for citation in citations:
        try:
            results = collection.get(
                where={"citation": citation},
                include=["documents"],
            )
            if results["documents"]:
                parts.append(
                    f"## {citation}\n{results['documents'][0]}"
                )
            else:
                parts.append(f"## {citation}\nNOT FOUND in knowledge base.")
        except Exception as exc:
            failed += 1
            print(f"WARNING: chunk lookup failed for '{citation}': {exc}")
            continue
    if citations and failed == len(citations):
        print("WARNING: ALL chunk text lookups failed — verifier will run without ground-truth text")
    return "\n\n---\n\n".join(parts) if parts else ""


def _serialize(obj) -> str:
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=2)
    return str(obj)


def run_pipeline(
    user_request: str,
    trace: Optional[TraceLog] = None,
    on_stage_update: Optional[Callable[["TraceLog"], None]] = None,
) -> PipelineResult:
    """Run the full Plan -> Research -> Draft -> Verify -> Deliver pipeline."""

    if trace is None:
        trace = TraceLog()

    injection = check_prompt_injection(user_request)
    if injection:
        raise ValueError(injection)

    relevance = check_input_relevance(user_request)
    if relevance:
        raise ValueError(relevance)

    def _notify():
        if on_stage_update:
            on_stage_update(trace)

    trace.start_pipeline()
    chunk_count = _warm_retrieval()
    if chunk_count == 0:
        trace.end_pipeline()
        raise ValueError(
            "Knowledge base is empty or unavailable. "
            "Please ensure the ChromaDB index has been built before running the pipeline."
        )

    guard_entry = trace.begin("Input Guard", "guardrail", input_preview=user_request)
    trace.complete(
        guard_entry,
        output_preview="Input accepted",
        injection_check="passed",
        relevance_check="passed",
        kb_chunks=chunk_count,
    )
    _notify()

    try:
        # Stage 1 - Plan
        plan_entry = trace.begin("Planner Agent", "plan",
                                 input_preview=user_request)
        _notify()
        try:
            plan_result = Runner.run_sync(
                planner_agent,
                build_planner_prompt(user_request),
            )
            plan: ExecutionPlan = plan_result.final_output
            trace.complete(
                plan_entry,
                output_preview=plan.task_summary,
                sub_tasks=len(plan.sub_tasks),
                queries=len(plan.research_queries),
            )
            _notify()
        except Exception as e:
            trace.fail(plan_entry, str(e))
            _notify()
            raise RuntimeError(f"Planner Agent failed: {e}") from e

        # Stage 2 - Research
        research_input = build_researcher_prompt(_serialize(plan), user_request)
        research_entry = trace.begin(
            "Research Agent", "research", input_preview=plan.task_summary
        )
        _notify()
        try:
            research_result = Runner.run_sync(
                researcher_agent, research_input, max_turns=40
            )
            research: ResearchNotes = research_result.final_output

            finding_citations = list(
                dict.fromkeys(f.citation for f in research.findings)
            )
            existing = set(research.sources_used)
            for cit in finding_citations:
                if cit not in existing:
                    research.sources_used.append(cit)
                    existing.add(cit)

            trace.complete(
                research_entry,
                output_preview=research.summary,
                findings=len(research.findings),
                gaps=len(research.gaps),
                sources=len(research.sources_used),
            )
            _notify()
        except Exception as e:
            trace.fail(research_entry, str(e))
            _notify()
            raise RuntimeError(f"Research Agent failed: {e}") from e

        # Stage 3 - Draft
        writer_input = build_writer_prompt(_serialize(research), user_request)
        draft_entry = trace.begin(
            "Writer Agent", "draft", input_preview=research.summary
        )
        _notify()
        try:
            draft_result = Runner.run_sync(writer_agent, writer_input)
            draft: Deliverable = draft_result.final_output
            trace.complete(
                draft_entry,
                output_preview=draft.executive_summary,
                action_items=len(draft.action_items),
                sources=len(draft.sources),
            )
            _notify()
        except Exception as e:
            trace.fail(draft_entry, str(e))
            _notify()
            raise RuntimeError(f"Writer Agent failed: {e}") from e

        # Stage 4 - Verify
        chunk_texts = _fetch_chunk_texts(research)
        verify_input = build_verifier_prompt(
            _serialize(draft), _serialize(research), chunk_texts
        )
        verify_entry = trace.begin(
            "Verifier Agent", "verify", input_preview=draft.executive_summary
        )
        _notify()
        try:
            verify_result = Runner.run_sync(verifier_agent, verify_input)
            verification: VerificationReport = verify_result.final_output
            trace.complete(
                verify_entry,
                output_preview=verification.overall_verdict,
                claims_checked=len(verification.verified_claims),
                unsupported=len(verification.unsupported_claims),
                verdict=verification.overall_verdict,
                suggestions=len(verification.suggestions),
            )
            _notify()
        except Exception as e:
            trace.fail(verify_entry, str(e))
            _notify()
            raise RuntimeError(f"Verifier Agent failed: {e}") from e

        # Stage 5 - Deliver
        deliver_entry = trace.begin(
            "Delivery Agent", "deliver", input_preview=verification.overall_verdict
        )
        _notify()

        verdict = verification.overall_verdict.strip().upper()
        if verdict.startswith("PASS"):
            filtered_sources = _collect_verified_sources(
                draft.sources, verification.verified_claims
            )
            final_deliverable = Deliverable(
                executive_summary=draft.executive_summary,
                client_email=draft.client_email,
                action_items=draft.action_items,
                sources=filtered_sources if filtered_sources else draft.sources,
            )
            trace.complete(
                deliver_entry,
                output_preview="PASS - draft used as-is",
                verdict="PASS",
                corrections_applied=False,
                sources_kept=len(filtered_sources) if filtered_sources else len(draft.sources),
            )
            _notify()
        else:
            disclaimer = _build_unsupported_disclaimer(
                verification.unsupported_claims
            )
            used_fallback = False

            if verification.corrected_executive_summary is not None:
                final_summary = verification.corrected_executive_summary
            else:
                final_summary = draft.executive_summary + disclaimer
                used_fallback = True

            if verification.corrected_client_email is not None:
                final_email = verification.corrected_client_email
            else:
                final_email = draft.client_email + disclaimer
                used_fallback = True

            final_actions = (
                verification.corrected_action_items or draft.action_items
            )

            if used_fallback:
                final_sources = draft.sources
            else:
                final_sources = _collect_verified_sources(
                    draft.sources, verification.verified_claims
                )

            final_deliverable = Deliverable(
                executive_summary=final_summary,
                client_email=final_email,
                action_items=final_actions,
                sources=final_sources,
            )

            fallback_note = " (with disclaimer - verifier missing corrections)" if used_fallback else ""
            trace.complete(
                deliver_entry,
                output_preview=f"{verdict} - corrections applied{fallback_note}",
                verdict=verdict,
                corrections_applied=True,
                used_fallback=used_fallback,
                sources_kept=len(final_sources),
            )
            _notify()
    finally:
        trace.end_pipeline()

    return PipelineResult(
        plan=plan,
        research=research,
        draft=draft,
        verification=verification,
        final_deliverable=final_deliverable,
    )


def format_deliverable(deliverable: Deliverable) -> str:
    action_rows = []
    for i, item in enumerate(deliverable.action_items, 1):
        action_rows.append(
            f"| {i} | {item.action} | {item.owner} | {item.due_date} | {item.confidence} |"
        )
    actions_table = (
        "| # | Action | Owner | Due Date | Confidence |\n"
        "|---|--------|-------|----------|------------|\n"
        + "\n".join(action_rows)
    )

    sources_list = "\n".join(f"- {s}" for s in deliverable.sources)

    return (
        f"# Executive Summary\n\n{deliverable.executive_summary}\n\n"
        f"---\n\n"
        f"# Client Email\n\n{deliverable.client_email}\n\n"
        f"---\n\n"
        f"# Action Items\n\n{actions_table}\n\n"
        f"---\n\n"
        f"# Sources\n\n{sources_list}"
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    request = (
        "Analyze the key challenges and best practices for building "
        "omnichannel retail operations. Recommend actionable strategies "
        "for a mid-sized CPG company looking to improve inventory accuracy "
        "and fulfillment speed."
    )

    print(f"User request:\n{request}\n")
    print("Running pipeline...\n")

    log = TraceLog()
    result = run_pipeline(request, trace=log)

    print(log.format_for_display())
    print()
    print(format_deliverable(result.final_deliverable))
