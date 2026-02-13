"""
Agents package for the Retail Operations Multi-Agent Copilot.

Implements the Plan -> Research -> Draft -> Verify -> Deliver pipeline
using the OpenAI Agents SDK.

Usage:
    from copilot_agents.orchestrator import run_pipeline, format_deliverable
    from copilot_agents.tracing import TraceLog

    trace = TraceLog()
    result = run_pipeline("your business request here", trace=trace)
    print(format_deliverable(result.final_deliverable))
    print(trace.format_for_display())
"""

__version__ = "0.1.0"
