"""Pydantic models that define the structured output for each agent in the pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field


# Planner Agent output models

class ResearchQuery(BaseModel):
    query: str = Field(description="Search query to run against the document store")
    purpose: str = Field(description="Why this query is needed")


class ExecutionPlan(BaseModel):
    task_summary: str = Field(description="One-sentence summary of what the user wants")
    sub_tasks: List[str] = Field(description="Ordered list of sub-tasks to complete")
    research_queries: List[ResearchQuery] = Field(
        description="Queries for the Research Agent to run"
    )
    focus_areas: List[str] = Field(
        description="Key themes the research should cover"
    )


# Research Agent output models

class ResearchFinding(BaseModel):
    finding: str = Field(description="A key fact, statistic, or insight from the documents")
    citation: str = Field(description="Source reference (DocumentName, Page/Chunk)")
    relevance: str = Field(description="How this finding relates to the task")


class ResearchNotes(BaseModel):
    findings: List[ResearchFinding] = Field(
        description="Grounded findings with citations"
    )
    gaps: List[str] = Field(
        description="Info that was needed but not found in sources"
    )
    sources_used: List[str] = Field(
        description="All unique source citations referenced"
    )
    summary: str = Field(
        description="Brief summary of what was found and what is missing"
    )


# Writer Agent output models

class ActionItem(BaseModel):
    action: str = Field(description="What needs to be done")
    owner: str = Field(description="Who is responsible")
    due_date: str = Field(description="Suggested timeline or deadline")
    confidence: str = Field(description="High, Medium, or Low based on source support")


class Deliverable(BaseModel):
    executive_summary: str = Field(
        description="Executive summary, max 150 words, based on research"
    )
    client_email: str = Field(
        description="Professional client-ready email with the findings"
    )
    action_items: List[ActionItem] = Field(
        description="Action items with owner, timeline, and confidence"
    )
    sources: List[str] = Field(
        description="All citations used in the deliverable"
    )


# Verifier Agent output models

class ClaimVerification(BaseModel):
    claim: str = Field(description="The claim being checked")
    verdict: str = Field(description="SUPPORTED / NOT SUPPORTED / PARTIALLY SUPPORTED")
    supporting_sources: List[str] = Field(
        description="Citations that back this claim"
    )
    explanation: str = Field(description="Why this verdict was given")


class VerificationReport(BaseModel):
    overall_verdict: str = Field(
        description="PASS if all claims supported, FAIL if any unsupported, PARTIAL otherwise"
    )
    verified_claims: List[ClaimVerification] = Field(
        description="Verification result for each claim"
    )
    unsupported_claims: List[str] = Field(
        description="Claims not backed by any source"
    )
    suggestions: List[str] = Field(
        description="Suggestions for fixing the deliverable"
    )
    corrected_executive_summary: Optional[str] = Field(
        default=None,
        description="Fixed executive summary with unsupported claims removed"
    )
    corrected_client_email: Optional[str] = Field(
        default=None,
        description="Fixed client email with unsupported claims removed"
    )
    corrected_action_items: Optional[List[ActionItem]] = Field(
        default=None,
        description="Fixed action items with unsupported ones removed"
    )


# Full pipeline output

class PipelineResult(BaseModel):
    plan: ExecutionPlan
    research: ResearchNotes
    draft: Deliverable
    verification: VerificationReport
    final_deliverable: Deliverable = Field(
        description="The final deliverable after verification corrections are applied"
    )
