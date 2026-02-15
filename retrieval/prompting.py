"""Formats retrieval results into prompts for the LLM agents."""

from typing import List
from .retrieval import RetrievedChunk


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

    citations = list(dict.fromkeys(chunk.citation for chunk in chunks))
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


if __name__ == "__main__":
    from .retrieval import retrieve_with_context

    print("Testing format helpers...")
    results = retrieve_with_context(
        "omnichannel retail strategies best practices", top_k=3
    )
    print(f"Found: {results['found']}")
    print(f"Message: {results['message']}")
    if results["found"]:
        context = format_context_for_agent(results["chunks"], include_scores=True)
        citations = format_citations(results["chunks"])
        print(f"Context preview:\n{context[:500]}...")
        print(f"\nCitations:\n{citations}")
    print("\nDone.")
