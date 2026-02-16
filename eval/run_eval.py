"""
Evaluation set for the Retail Operations Copilot.
10 test prompts that cover different retail/CPG topics from the knowledge base.
"""

from copilot_agents.tracing import TraceLog
from copilot_agents.orchestrator import run_pipeline, format_deliverable
from dotenv import load_dotenv
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()


TEST_PROMPTS = [
    {
        "id": "supply_chain_visibility",
        "query": "What are the biggest challenges in supply chain visibility for retail, and what technologies can help solve them?",
        "min_sources": 2,
        "min_actions": 2,
    },
    {
        "id": "omnichannel_strategy",
        "query": "How should a mid-sized retailer build an omnichannel strategy? What are the key success factors?",
        "min_sources": 2,
        "min_actions": 2,
    },
    {
        "id": "inventory_accuracy",
        "query": "What are the best practices for improving inventory accuracy in retail stores?",
        "min_sources": 2,
        "min_actions": 2,
    },
    {
        "id": "retail_returns",
        "query": "Analyze the current state of retail returns and recommend strategies to reduce return rates while keeping customers happy.",
        "min_sources": 1,
        "min_actions": 2,
    },
    {
        "id": "fulfillment_automation",
        "query": "What role does automation play in retail fulfillment, and what should companies invest in first?",
        "min_sources": 2,
        "min_actions": 2,
    },
    {
        "id": "cpg_digital_transformation",
        "query": "What does digital transformation look like for CPG companies, and what are the biggest risks involved?",
        "min_sources": 2,
        "min_actions": 2,
    },
    {
        "id": "global_retail_trends",
        "query": "What are the key trends shaping global retail in 2025 according to industry reports?",
        "min_sources": 2,
        "min_actions": 2,
    },
    {
        "id": "last_mile_delivery",
        "query": "How can retailers optimize last-mile delivery while keeping costs under control?",
        "min_sources": 1,
        "min_actions": 2,
    },
    {
        "id": "sustainability_gap",
        "query": "What sustainability initiatives are leading retailers adopting to reduce their carbon footprint?",
        "min_sources": 0,
        "min_actions": 1,
    },
    {
        "id": "complex_multi_part",
        "query": "Compare omnichannel strategies for CPG companies versus pure-play retailers, identify gaps in current approaches, and recommend a 12-month implementation roadmap with prioritized action items.",
        "min_sources": 3,
        "min_actions": 3,
    },
]

BAD_PHRASES = [
    "as an ai", "as a language model", "i cannot access",
    "i don't have access", "my training data", "my knowledge cutoff",
]


def grade(pipeline_result, test):
    d = pipeline_result.final_deliverable
    v = pipeline_result.verification
    results = []

    has_summary = len(d.executive_summary.strip()) > 0
    results.append(("has_summary", has_summary, ""))

    wc = len(d.executive_summary.split())
    results.append(("summary_under_150w", wc <= 150,
                   f"{wc} words" if wc > 150 else ""))

    results.append(("has_email", len(d.client_email.strip()) > 0, ""))

    n = len(d.action_items)
    results.append(("enough_actions", n >= test["min_actions"],
                    f"got {n}, need {test['min_actions']}" if n < test["min_actions"] else ""))

    fields_ok = all(
        a.action.strip() and a.owner.strip() and a.due_date.strip() and a.confidence.strip()
        for a in d.action_items
    )
    results.append(("action_fields_filled", fields_ok, ""))

    ns = len(d.sources)
    results.append(("enough_sources", ns >= test["min_sources"],
                    f"got {ns}, need {test['min_sources']}" if ns < test["min_sources"] else ""))

    verdict = v.overall_verdict.strip().upper()
    results.append(("valid_verdict", verdict in (
        "PASS", "FAIL", "PARTIAL"), verdict))

    text = (d.executive_summary + " " + d.client_email).lower()
    found = [p for p in BAD_PHRASES if p in text]
    results.append(("no_hallucination_phrases", len(found) == 0,
                    str(found) if found else ""))

    return results


def run_test(test, verbose=False):
    print(f"\n>> {test['id']}")
    print(f"   {test['query'][:80]}...")

    trace = TraceLog()
    t0 = time.time()

    try:
        result = run_pipeline(test["query"], trace=trace)
        dur = time.time() - t0
        checks = grade(result, test)
        passed = sum(1 for _, ok, _ in checks if ok)
        total = len(checks)

        if verbose:
            print(format_deliverable(result.final_deliverable))

    except Exception as e:
        dur = time.time() - t0
        print(f"   CRASHED: {e}")
        return {"id": test["id"], "passed": 0, "total": 8, "ok": False,
                "duration": dur, "issues": [f"pipeline error: {e}"]}

    issues = [f"{name} ({detail})" if detail else name
              for name, ok, detail in checks if not ok]

    for issue in issues:
        print(f"   FAILED: {issue}")

    if not issues:
        print(f"   all checks passed ({dur:.0f}s)")

    return {"id": test["id"], "passed": passed, "total": total,
            "ok": passed == total, "duration": dur, "issues": issues}


def main():
    parser = argparse.ArgumentParser(
        description="Run eval tests for the Retail Copilot.")
    parser.add_argument("--test", type=str, default=None,
                        help="Run one test by ID (e.g. inventory_accuracy)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full deliverable for each test")
    args = parser.parse_args()

    if args.test:
        matches = [t for t in TEST_PROMPTS if t["id"] == args.test]
        if not matches:
            ids = [t["id"] for t in TEST_PROMPTS]
            print(f"Unknown test '{args.test}'. Pick from: {', '.join(ids)}")
            sys.exit(1)
        tests = matches
    else:
        tests = TEST_PROMPTS

    print(f"Running {len(tests)} eval test(s)...")

    results = [run_test(t, verbose=args.verbose) for t in tests]

    n_passed = sum(1 for r in results if r["ok"])
    total_time = sum(r["duration"] for r in results)
    print(f"\n{'='*50}")
    print(f"Results: {n_passed}/{len(results)} passed")
    print(f"Total time: {total_time:.0f}s")

    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {status}  {r['id']} ({r['duration']:.0f}s)")
        for issue in r["issues"]:
            print(f"        ^ {issue}")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
