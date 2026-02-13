"""
Verifier Agent  -  fourth step in the copilot pipeline.

Checks the Writer's deliverable against the research notes to catch
hallucinations, unsupported claims, and contradictions.
"""

import time

from agents import Agent, Runner

from .models import Deliverable, ResearchNotes, VerificationReport


VERIFIER_INSTRUCTIONS = """\
You are the Verifier Agent for a Retail / CPG operations copilot.

Your job is to ensure the deliverable is factually grounded in the
research notes provided. You receive:
  (a) The draft deliverable (Executive Summary, Client Email, Action Items, Sources).
  (b) The original research notes with findings and citations.

VERIFICATION PROCESS:
1. Extract every factual claim from the Executive Summary and Client Email.
2. For each claim, check whether it can be traced to a specific finding
   in the research notes.
3. Assign a verdict to each claim:
   - SUPPORTED: directly backed by at least one finding + citation.
   - PARTIALLY SUPPORTED: related finding exists but the claim adds
     detail or interpretation not in the source.
   - NOT SUPPORTED: no matching finding in the research notes -
     this is a potential hallucination.
4. Check Action Items: each recommendation's confidence should match
   the evidence strength in the research notes.

OUTPUT:
Return a VerificationReport with:
- overall_verdict: PASS / FAIL / PARTIAL
  - PASS    = all claims are SUPPORTED
  - FAIL    = any claim is NOT SUPPORTED
  - PARTIAL = some claims are only PARTIALLY SUPPORTED
- verified_claims: list of {claim, verdict, supporting_sources, explanation}
- unsupported_claims: list of claims that are NOT SUPPORTED
- suggestions: how to fix problems found

If the overall_verdict is FAIL or PARTIAL, you MUST also provide:
- corrected_executive_summary: rewrite with unsupported claims removed or
  replaced with "Not found in sources".
- corrected_client_email: same treatment.
- corrected_action_items: remove or lower confidence on unsupported items.

CRITICAL RULES:
- Be strict: if a claim cannot be traced to a specific research finding
  with a citation, mark it NOT SUPPORTED.
- "Not found in sources" is the required phrase for missing evidence.
- Do NOT approve vague or unverifiable statements.
- You do NOT have access to the original documents - only the research notes.
  Verify claims ONLY against the research findings provided.
"""


verifier_agent = Agent(
    name="Verifier Agent",
    instructions=VERIFIER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=VerificationReport,
)


def run_verifier(draft: Deliverable, research: ResearchNotes) -> VerificationReport:
    """Run the verifier on a draft deliverable and return the verification report."""

    print("[Verifier] Starting...")
    print(
        f"[Verifier] Claims to check from draft with {len(draft.action_items)} action items")
    print(f"[Verifier] Research findings available: {len(research.findings)}")

    start = time.time()

    prompt = (
        "Verify the following deliverable against the research notes.\n\n"
        f"DRAFT DELIVERABLE:\n{draft.model_dump_json(indent=2)}\n\n"
        f"RESEARCH NOTES (with citations):\n{research.model_dump_json(indent=2)}"
    )

    result = Runner.run_sync(verifier_agent, prompt)
    verification: VerificationReport = result.final_output

    elapsed = round(time.time() - start, 2)

    print(f"[Verifier] Done in {elapsed}s")
    print(f"[Verifier] Overall verdict: {verification.overall_verdict}")
    print(f"[Verifier] Claims checked: {len(verification.verified_claims)}")
    print(
        f"[Verifier] Unsupported claims: {len(verification.unsupported_claims)}")

    return verification


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from .planner import run_planner
    from .researcher import run_researcher
    from .writer import run_writer

    test_request = (
        "What are the best practices for improving inventory accuracy "
        "in omnichannel retail operations?"
    )

    print("Step 1: Running the planner first...\n")
    plan = run_planner(test_request)

    print("\nStep 2: Running the researcher on that plan...\n")
    research = run_researcher(plan, test_request)

    print("\nStep 3: Running the writer on those research notes...\n")
    draft = run_writer(research, test_request)

    print("\nStep 4: Running the verifier on that draft...\n")
    verification = run_verifier(draft, research)

    print(f"\nOverall verdict: {verification.overall_verdict}")

    print("\nVerified claims:")
    for i, claim in enumerate(verification.verified_claims, 1):
        print(f"  {i}. [{claim.verdict}] {claim.claim}")
        if claim.supporting_sources:
            print(f"     Sources: {', '.join(claim.supporting_sources)}")
        print(f"     Explanation: {claim.explanation}")

    if verification.unsupported_claims:
        print("\nUnsupported claims:")
        for claim in verification.unsupported_claims:
            print(f"  - {claim}")

    if verification.suggestions:
        print("\nSuggestions:")
        for s in verification.suggestions:
            print(f"  - {s}")
