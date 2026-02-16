"""
Writer Agent  -  third step in the copilot pipeline.

Takes the research notes from Stage 2 and produces the final structured
deliverable: Executive Summary, Client Email, Action List, and Sources.
"""

import time

from agents import Agent, Runner

from .models import Deliverable, ResearchNotes


WRITER_INSTRUCTIONS = """\
You are the Writer Agent for a Retail / CPG operations copilot.

You receive structured research notes (with citations) and produce a polished,
client-ready deliverable.

YOUR DELIVERABLE MUST CONTAIN EXACTLY THESE FOUR SECTIONS:

1. EXECUTIVE SUMMARY (max 150 words)
   - Concise overview of key findings and recommendations.
   - Every claim must reference a source from the research notes.
   - Written for a C-level audience.

2. CLIENT-READY EMAIL
   - Professional email format with Subject, Greeting, Body, and Closing.
   - Summarizes findings and recommends next steps.
   - Tone: professional, confident, data-driven.
   - Do NOT include inline citations in the email. Raw citations like
     "(DocumentName, Page N, Chunk M)" are internal artifacts and break
     professional tone. The email must read cleanly for an external client.
   - The email field must contain ONLY the email (Subject through Sign-off).
     Do NOT append Action Items, Sources, or any other section into the email.
     Those belong in their own separate fields.

3. ACTION ITEMS
   - 3-7 specific, actionable recommendations.
   - Each item must include:
     * action: a concrete, measurable step.
     * owner: a business ROLE (e.g., "Supply Chain Director",
       "VP of Merchandising", "Head of E-Commerce"). Never use personal
       names or generic terms like "Team" or "Management".
     * due_date: a future quarterly milestone in the format
       "Q3 2026" or "Q1 2027". Must be after today's date. Never use
       vague terms like "ASAP", "TBD", or "Immediately".
     * confidence: exactly one of High, Medium, or Low.
   - Confidence is based on how strongly the sources support the recommendation.
   - If evidence is weak, set confidence to Low and note the limitation.

4. SOURCES
   - List every unique citation from the research notes that appears in any
     finding you referenced. Do NOT omit citations from any document.
   - Use the full citation format: "DocumentName, Page N, Chunk M" - not just
     the document name.
   - Before finalizing, cross-check that every citation attached to a finding
     you used is present in this list.

CRITICAL RULES:
- ONLY use information from the research notes provided. NEVER add unsupported claims.
- "Not found in sources" must ONLY be used when the research notes' GAPS section
  explicitly lists something as missing. If a finding exists with a citation,
  use that citation - NEVER replace it with "Not found in sources." Do not
  generate "Not found in sources" on your own as a fallback.
- Maintain citation traceability throughout all sections.
- CITATION ACCURACY: Every inline citation you attach to a claim MUST be the
  exact same citation string that the Research Agent assigned to the finding you
  are drawing from. Do NOT reassign, swap, or merge citations across findings.
  If a sentence combines facts from multiple findings, list ALL of their original
  citations - do not pick just one.
- FINDING COVERAGE: The Executive Summary and Client Email must address EVERY
  research finding - not just a subset. Do not drop any findings or replace them
  with tangential details from the same chunk. Before finalizing, count the
  findings in the research notes and confirm each one is represented in your output.
- NO TANGENTIAL PROMOTION: Do not elevate minor or side details from a chunk into
  key claims. If a chunk's main point is about control towers but also mentions
  barcodes in passing, the control towers are the key finding - not the barcodes.
"""


writer_agent = Agent(  # type: ignore
    name="Writer Agent",
    instructions=WRITER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Deliverable,
)


def build_writer_prompt(research_json: str, user_request: str) -> str:
    from datetime import date
    today = date.today().isoformat()
    import json
    finding_count = len(json.loads(research_json).get("findings", []))
    return (
        "Using the research notes below, produce the final deliverable.\n\n"
        f"TODAY'S DATE: {today}\n"
        "Use this date to ensure all action item timelines are in the future.\n\n"
        f"ORIGINAL REQUEST:\n{user_request}\n\n"
        f"TOTAL FINDINGS: {finding_count} — your output must cover all {finding_count}.\n\n"
        f"RESEARCH NOTES:\n{research_json}"
    )


def run_writer(research: ResearchNotes, user_request: str) -> Deliverable:
    """Run the writer on research notes and return the final deliverable."""

    print("[Writer] Starting...")
    print(f"[Writer] Research summary: {research.summary[:120]}...")
    print(f"[Writer] Findings to work with: {len(research.findings)}")
    print(f"[Writer] Gaps flagged: {len(research.gaps)}")

    start = time.time()

    prompt = build_writer_prompt(research.model_dump_json(indent=2), user_request)

    result = Runner.run_sync(writer_agent, prompt)
    deliverable: Deliverable = result.final_output

    elapsed = round(time.time() - start, 2)

    print(f"[Writer] Done in {elapsed}s")
    print(f"[Writer] Action items: {len(deliverable.action_items)}")
    print(f"[Writer] Sources cited: {len(deliverable.sources)}")

    return deliverable


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from .planner import run_planner
    from .researcher import run_researcher

    test_request = (
        "What are the best practices for improving inventory accuracy "
        "in omnichannel retail operations?"
    )

    print("Step 1: Running the planner first...\n")
    plan = run_planner(test_request)

    print("\nStep 2: Running the researcher on that plan...\n")
    research = run_researcher(plan, test_request)

    print("\nStep 3: Running the writer on those research notes...\n")
    deliverable = run_writer(research, test_request)

    print("\nEXECUTIVE SUMMARY")
    print(deliverable.executive_summary)

    print("\nCLIENT EMAIL")
    print(deliverable.client_email)

    print("\nACTION ITEMS")
    for i, item in enumerate(deliverable.action_items, 1):
        print(f"  {i}. {item.action}")
        print(f"     Owner: {item.owner}")
        print(f"     Due: {item.due_date}")
        print(f"     Confidence: {item.confidence}")

    print("\nSOURCES")
    for s in deliverable.sources:
        print(f"  - {s}")
