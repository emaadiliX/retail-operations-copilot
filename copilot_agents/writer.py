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
   - Include inline citations where key data is referenced.

3. ACTION ITEMS
   - 3-7 specific, actionable recommendations.
   - Each item must include: action, suggested owner/role, suggested timeline,
     and confidence level (High / Medium / Low).
   - Confidence is based on how strongly the sources support the recommendation.
   - If evidence is weak, set confidence to Low and note the limitation.

4. SOURCES
   - List every unique citation from the research notes used in the deliverable.

CRITICAL RULES:
- ONLY use information from the research notes provided. NEVER add unsupported claims.
- If the research notes say "Not found in sources" for something, you must also
  state "Not found in sources" - do not fill the gap with your own knowledge.
- Maintain citation traceability throughout all sections.
"""


writer_agent = Agent(
    name="Writer Agent",
    instructions=WRITER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Deliverable,
)


def run_writer(research: ResearchNotes, user_request: str) -> Deliverable:
    """Run the writer on research notes and return the final deliverable."""

    print("[Writer] Starting...")
    print(f"[Writer] Research summary: {research.summary[:120]}...")
    print(f"[Writer] Findings to work with: {len(research.findings)}")
    print(f"[Writer] Gaps flagged: {len(research.gaps)}")

    start = time.time()

    prompt = (
        "Using the research notes below, produce the final deliverable.\n\n"
        f"ORIGINAL REQUEST:\n{user_request}\n\n"
        f"RESEARCH NOTES:\n{research.model_dump_json(indent=2)}"
    )

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
