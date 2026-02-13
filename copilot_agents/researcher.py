"""
Research Agent  -  second step in the copilot pipeline.

Takes the execution plan from the Planner and uses the search tools
to find relevant information from our retail/CPG knowledge base.
Produces structured research notes with citations.
"""

import time

from agents import Agent, Runner

from .models import ExecutionPlan, ResearchNotes
from .tools import search_retail_documents, multi_search_retail_documents


RESEARCHER_INSTRUCTIONS = """\
You are the Research Agent for a Retail / CPG operations copilot.

You have access to a knowledge base of retail and CPG documents including
whitepapers, strategy reports, and industry analyses. Use the provided
search tools to find relevant information.

WORKFLOW:
1. You will receive an execution plan with research queries and focus areas.
2. Use the search tools to retrieve information for each query.
3. Extract key facts, statistics, and insights from the retrieved sources.
4. Compile your findings into structured research notes.

CRITICAL RULES:
- ONLY use information found in the retrieved documents. NEVER fabricate data.
- ALWAYS include the exact citation for every finding (DocumentName, Page, Chunk).
- If a query returns no results, record it as a gap - say "Not found in sources."
- Do NOT add your own knowledge or assumptions beyond what the sources state.
- Prefer specific numbers, percentages, and quotes over vague summaries.

OUTPUT FORMAT:
Return structured ResearchNotes with:
- findings: list of {finding, citation, relevance} - one per key fact
- gaps: list of information that was needed but not found
- sources_used: all unique citations referenced
- summary: what was found and what is missing
"""


researcher_agent = Agent(
    name="Research Agent",
    instructions=RESEARCHER_INSTRUCTIONS,
    model="gpt-4o-mini",
    tools=[search_retail_documents, multi_search_retail_documents],
    output_type=ResearchNotes,
)


def run_researcher(plan: ExecutionPlan, user_request: str) -> ResearchNotes:
    """Run the researcher on an execution plan and return the research notes."""

    print("[Researcher] Starting...")
    print(f"[Researcher] Plan summary: {plan.task_summary}")
    print(f"[Researcher] Queries to run: {len(plan.research_queries)}")

    start = time.time()

    prompt = (
        "Execute the following research plan. Use the search tools to find "
        "information for each query.\n\n"
        f"EXECUTION PLAN:\n{plan.model_dump_json(indent=2)}\n\n"
        f"ORIGINAL USER REQUEST:\n{user_request}"
    )

    result = Runner.run_sync(researcher_agent, prompt, max_turns=25)
    research: ResearchNotes = result.final_output

    elapsed = round(time.time() - start, 2)

    print(f"[Researcher] Done in {elapsed}s")
    print(f"[Researcher] Findings: {len(research.findings)}")
    print(f"[Researcher] Gaps: {len(research.gaps)}")
    print(f"[Researcher] Sources used: {len(research.sources_used)}")

    return research


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from .planner import run_planner

    test_request = (
        "What are the best practices for improving inventory accuracy "
        "in omnichannel retail operations?"
    )

    print("Step 1: Running the planner first to get an execution plan...\n")
    plan = run_planner(test_request)

    print("\nStep 2: Running the researcher on that plan...\n")
    research = run_researcher(plan, test_request)

    print(f"\nSummary: {research.summary}\n")

    print("Findings:")
    for i, f in enumerate(research.findings, 1):
        print(f"  {i}. {f.finding}")
        print(f"     Citation: {f.citation}")
        print(f"     Relevance: {f.relevance}\n")

    if research.gaps:
        print("Gaps (not found in sources):")
        for gap in research.gaps:
            print(f"  - {gap}")

    print(f"\nSources used: {', '.join(research.sources_used)}")
