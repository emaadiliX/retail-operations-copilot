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
  (c) ORIGINAL CHUNK TEXTS: the actual text from the knowledge base for each
      cited chunk. Use these to verify that claimed statistics actually appear
      in the cited chunk.

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
4. Check Action Items:
   - Each recommendation's confidence (High/Medium/Low) must match the
     evidence strength in the research notes.
   - Due dates must be future quarterly or yearly milestones (e.g., Q3 2026).
     Flag any that are vague ("ASAP", "TBD") or in the past.
   - Owners must be business roles (e.g., "Supply Chain Director"), not
     personal names or generic terms like "Team".

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
- corrected_executive_summary: rewrite with unsupported sentences REMOVED
  entirely. Do NOT replace them with the phrase "Not found in sources." — that
  phrase is only for the gaps list. Simply delete the unsupported sentence and
  ensure the remaining text reads naturally and flows as coherent prose.
- corrected_client_email: same treatment — remove unsupported sentences cleanly.
- corrected_action_items: remove or lower confidence on unsupported items.

CRITICAL RULES:
- Be strict: if a claim cannot be traced to a specific research finding
  with a citation, mark it NOT SUPPORTED.
- "Not found in sources" is the required phrase for missing evidence.
- Do NOT approve vague or unverifiable statements.
- Verify claims against the research findings AND the original chunk texts
  provided. The chunk texts are the ground truth from the knowledge base.
- Check SEMANTIC ACCURACY, not just topic overlap. If a finding says X is
  "an opportunity" but the claim says X is "a problem," or if a finding says
  "reduced costs" but the claim says "elevated costs," mark the claim
  PARTIALLY SUPPORTED or NOT SUPPORTED. Inversions, exaggerations, and
  misrepresentations of tone or direction count as inaccuracies.
- Watch for loose paraphrasing that changes meaning. For example,
  "lack of ubiquity for standards" is NOT the same as "lack of standards."
- CITATION CROSS-CHECK: For every inline citation in the Executive Summary
  and Client Email, verify that the citation string matches the citation on
  the research finding that supports the claim. If a claim says
  "(SourceA, Page 5, Chunk 0)" but the matching research finding has
  "(SourceA, Page 6, Chunk 1)", mark the claim PARTIALLY SUPPORTED and
  flag the citation mismatch in the explanation. Swapped, merged, or
  incorrectly re-assigned citations count as errors.
- FABRICATION CHECK: If a finding mentions specific technologies, trends, or
  details that are NOT present in the research notes at all, mark the claim
  NOT SUPPORTED. The Writer cannot introduce new content beyond the findings.
- FRAMING CHECK: If a finding says something "will need to" happen (aspiration)
  but the deliverable presents it as "case studies indicate" (proven evidence),
  mark it PARTIALLY SUPPORTED and flag the framing mismatch.
- SOURCE COMPLETENESS: Verify that every unique citation referenced in the
  findings is listed in the Sources section. Flag any missing citations.
- CHUNK TEXT CROSS-CHECK: For EVERY finding (not just those with statistics),
  read the corresponding chunk text in the ORIGINAL CHUNK TEXTS section and
  confirm the finding's TOPIC actually matches the chunk. If a finding claims
  "robotics and analytics improve inventory accuracy" but the cited chunk text
  discusses "lockers and pickup points," the finding is MISATTRIBUTED — mark it
  NOT SUPPORTED and flag the citation mismatch. Check that the key nouns and
  concepts in the finding actually appear in the cited chunk text. A finding
  whose topic does not match its cited chunk is worse than a framing issue —
  it means the citation points to the wrong page or chunk entirely.
"""


verifier_agent = Agent(
    name="Verifier Agent",
    instructions=VERIFIER_INSTRUCTIONS,
    model="gpt-4o",
    output_type=VerificationReport,
)


def build_verifier_prompt(draft_json: str, research_json: str,
                          chunk_texts: str = "") -> str:
    prompt = (
        "Verify the following deliverable against the research notes.\n\n"
        f"DRAFT DELIVERABLE:\n{draft_json}\n\n"
        f"RESEARCH NOTES (with citations):\n{research_json}"
    )
    if chunk_texts:
        prompt += f"\n\nORIGINAL CHUNK TEXTS (ground truth from knowledge base):\n{chunk_texts}"
    return prompt


def run_verifier(draft: Deliverable, research: ResearchNotes) -> VerificationReport:
    """Run the verifier on a draft deliverable and return the verification report."""
    from .orchestrator import _fetch_chunk_texts

    print("[Verifier] Starting...")
    print(
        f"[Verifier] Claims to check from draft with {len(draft.action_items)} action items")
    print(f"[Verifier] Research findings available: {len(research.findings)}")

    start = time.time()

    chunk_texts = _fetch_chunk_texts(research)
    prompt = build_verifier_prompt(
        draft.model_dump_json(indent=2), research.model_dump_json(indent=2),
        chunk_texts
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
