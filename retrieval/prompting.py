"""Formats retrieval results into prompts for the LLM agents."""

from typing import List, Dict
from .retrieval import RetrievedChunk, retrieve_with_context


def format_context_for_agent(
    chunks: List[RetrievedChunk],
    include_scores: bool = False
) -> str:
    """Turn retrieved chunks into a structured markdown context string."""
    if not chunks:
        return "No relevant information found in the documents."

    context_parts = []
    context_parts.append("# Retrieved Information\n")
    context_parts.append(f"Found {len(chunks)} relevant sources:\n")

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"\n## Source {i}")
        context_parts.append(f"**Citation:** {chunk.citation}")

        if include_scores:
            context_parts.append(f"**Relevance Score:** {chunk.similarity_score:.3f}")

        context_parts.append(f"\n**Content:**\n{chunk.text}")
        context_parts.append("\n" + "-" * 70)

    return "\n".join(context_parts)


def format_citations(chunks: List[RetrievedChunk]) -> str:
    """Extract unique citations from chunks and return as a numbered list."""
    if not chunks:
        return "No sources"

    citations = list(set(chunk.citation for chunk in chunks))
    citation_lines = [f"{i}. {citation}" for i, citation in enumerate(citations, 1)]

    return "\n".join(citation_lines)


def create_grounded_response_prompt(
    query: str,
    context: str,
    citations: List[str]
) -> str:
    """Build a prompt that forces the agent to only use provided sources."""
    prompt = f"""You are a research assistant analyzing retail operations documents.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

AVAILABLE CITATIONS:
{chr(10).join(f"[{i}] {cite}" for i, cite in enumerate(citations, 1))}

INSTRUCTIONS:
1. Answer the query using ONLY information from the retrieved context above.
2. For every claim you make, cite the source using the citation format provided.
3. If the context doesn't contain enough information to fully answer the query, clearly state:
   "Not found in sources" and explain what information is missing.
4. Do not make assumptions or add information not present in the context.
5. Be specific and reference page numbers when possible.

Your response:"""

    return prompt


def create_research_agent_prompt(
    task: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.3
) -> Dict:
    """Retrieve relevant chunks and package them into a full prompt for the Research Agent."""
    results = retrieve_with_context(query, top_k=top_k, min_score=min_score)

    if not results["found"]:
        return {
            "found": False,
            "message": results["message"],
            "prompt": f"""RESEARCH TASK: {task}

SEARCH QUERY: {query}

RESULT: Not found in sources.

The available documents do not contain sufficient information to complete this research task.

Suggested actions:
1. Rephrase the query to search for related concepts
2. Break down the task into smaller, more specific questions
3. Consider that the information may not be available in the current document set

Please indicate what additional information would be needed to complete this task.""",
            "context": "",
            "citations": [],
            "chunks": []
        }

    context = format_context_for_agent(results["chunks"], include_scores=False)

    prompt = f"""You are a Research Agent specializing in retail operations.

RESEARCH TASK:
{task}

SEARCH QUERY USED:
{query}

{context}

INSTRUCTIONS FOR RESEARCH NOTES:
1. Extract key information relevant to the research task
2. Include specific facts, statistics, and insights from the sources
3. ALWAYS cite your sources using the provided citations
4. Organize information logically by theme or topic
5. Note any limitations or gaps in the available information
6. Do not add information not found in the sources

FORMAT YOUR RESEARCH NOTES AS:
- Use bullet points for key findings
- Include citations in square brackets after each point, e.g., [Source 1]
- Highlight any contradictions or uncertainties found
- End with a summary of what was found and what's missing (if anything)

Your research notes:"""

    return {
        "found": True,
        "message": results["message"],
        "prompt": prompt,
        "context": context,
        "citations": results["citations"],
        "chunks": results["chunks"],
        "chunks_by_document": results.get("chunks_by_document", {})
    }


def create_verification_prompt(
    claim: str,
    available_sources: List[RetrievedChunk]
) -> str:
    """Build a prompt for the Verifier Agent to check a claim against sources."""
    context = format_context_for_agent(available_sources, include_scores=False)

    prompt = f"""You are a Verification Agent. Your job is to check if claims are supported by source documents.

CLAIM TO VERIFY:
{claim}

AVAILABLE SOURCES:
{context}

VERIFICATION TASK:
1. Check if the claim is directly supported by the provided sources
2. Identify which source(s) support the claim (if any)
3. Determine if the claim contradicts any sources
4. Flag if the claim contains information not found in the sources

RESPOND WITH:
- VERDICT: [SUPPORTED / NOT SUPPORTED / PARTIALLY SUPPORTED / CONTRADICTED]
- SUPPORTING SOURCES: [List citations that support the claim]
- EXPLANATION: [Brief explanation of your verdict]
- MISSING INFORMATION: [What information is needed but not found, if any]

Your verification:"""

    return prompt


if __name__ == "__main__":
    print("Testing research agent prompt...")
    result = create_research_agent_prompt(
        task="Identify key strategies for improving omnichannel retail operations",
        query="omnichannel retail strategies best practices",
        top_k=3
    )
    print(f"Found: {result['found']}")
    print(f"Message: {result['message']}")
    if result["found"]:
        print(f"Citations: {len(result['citations'])}")
        print(f"Prompt preview:\n{result['prompt'][:500]}...")

    print("\nTesting verification prompt...")
    if result["found"] and result["chunks"]:
        v_prompt = create_verification_prompt(
            "Omnichannel retail requires integrating online and offline channels",
            result["chunks"][:2]
        )
        print(f"Verification prompt preview:\n{v_prompt[:500]}...")
    else:
        print("Skipped - no chunks from research test to verify against.")

    print("\nDone.")
