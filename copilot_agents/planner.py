"""
Planner Agent  -  first step in the copilot pipeline.

Takes the user's business request and breaks it down into smaller
sub-tasks and search queries for the Research Agent to work on.
"""

import time

from agents import Agent, Runner

from .models import ExecutionPlan


PLANNER_INSTRUCTIONS = """\
You are the Planner Agent for a Retail / CPG operations copilot.

Your job is to take a user's business request and break it down into a
structured execution plan that the other agents will follow.

RULES:
1. Read the user's request carefully and figure out the main question or goal.
2. Break it into 3-6 concrete sub-tasks that can each be researched on their own.
3. For each sub-task, write one or two specific search queries that the
   Research Agent will use to search our knowledge base of retail/CPG documents
   (whitepapers, strategy reports, supply-chain studies, omnichannel analyses).
4. Pick out the key focus areas or themes the research should cover.
5. Keep queries specific and grounded - do not write anything too vague or broad.
6. Use terminology that would actually appear in industry reports
   (e.g., "omnichannel fulfillment", "inventory accuracy", "supply chain visibility").

OUTPUT FORMAT:
Return a structured ExecutionPlan with:
- task_summary: one-sentence summary of the user's request
- sub_tasks: ordered list of sub-tasks
- research_queries: list of {query, purpose} objects
- focus_areas: key themes to cover
"""


planner_agent = Agent(
    name="Planner Agent",
    instructions=PLANNER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ExecutionPlan,
)


def run_planner(user_request: str) -> ExecutionPlan:
    """Run the planner on a user request and return the execution plan."""

    print("[Planner] Starting...")
    print(f"[Planner] Request: {user_request[:120]}...")

    start = time.time()

    prompt = (
        "Create an execution plan for the following business request:\n\n"
        + user_request
    )

    result = Runner.run_sync(planner_agent, prompt)
    plan: ExecutionPlan = result.final_output

    elapsed = round(time.time() - start, 2)

    print(f"[Planner] Done in {elapsed}s")
    print(f"[Planner] Summary: {plan.task_summary}")
    print(f"[Planner] Sub-tasks: {len(plan.sub_tasks)}")
    print(f"[Planner] Queries: {len(plan.research_queries)}")
    print(f"[Planner] Focus areas: {', '.join(plan.focus_areas)}")

    return plan


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_request = (
        "What are the best practices for improving inventory accuracy "
        "in omnichannel retail operations?"
    )

    plan = run_planner(test_request)

    print(f"\nSummary: {plan.task_summary}\n")

    print("Sub-tasks:")
    for i, task in enumerate(plan.sub_tasks, 1):
        print(f"  {i}. {task}")

    print("\nResearch queries:")
    for q in plan.research_queries:
        print(f"  - {q.query}")
        print(f"    Purpose: {q.purpose}")

    print(f"\nFocus areas: {', '.join(plan.focus_areas)}")
