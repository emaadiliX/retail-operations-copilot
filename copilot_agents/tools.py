"""Tool wrappers that let our agents search the retail knowledge base using @function_tool."""

from agents import function_tool

from retrieval.retrieval import retrieve_with_context, multi_query_retrieval
from retrieval.prompting import format_context_for_agent, format_citations


@function_tool
def search_retail_documents(query: str, top_k: int = 5) -> str:
    """Search the retail knowledge base for information related to a query."""
    results = retrieve_with_context(query, top_k=top_k, min_score=0.3)

    if not results["found"]:
        return "Not found in sources. The available documents do not contain information relevant to this query."

    context = format_context_for_agent(results["chunks"], include_scores=True)
    citations = format_citations(results["chunks"])

    return f"{context}\n\n## Citations\n{citations}"


@function_tool
def multi_search_retail_documents(queries: str) -> str:
    """Run multiple comma-separated search queries and combine the results. Use this when a topic needs info from different angles."""
    query_list = [q.strip() for q in queries.split(",") if q.strip()]

    if not query_list:
        return "No valid queries provided."

    results = multi_query_retrieval(query_list, top_k_per_query=3, min_score=0.3)

    if not results["found"]:
        return "Not found in sources. None of the queries returned relevant results from the document set."

    context = format_context_for_agent(results["chunks"], include_scores=True)
    citations = format_citations(results["chunks"])

    return (
        f"Combined results from {len(query_list)} queries:\n\n"
        f"{context}\n\n## Citations\n{citations}"
    )
