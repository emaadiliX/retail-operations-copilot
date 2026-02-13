"""Handles querying the vector store to find relevant document chunks."""

from typing import Any, List, Dict, Optional, cast
from dataclasses import dataclass
import chromadb

from .config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    MIN_SIMILARITY_SCORE,
    EMBEDDING_MODEL
)
from .indexing import get_chroma_client, generate_embeddings


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata and similarity score."""
    text: str
    document_name: str
    page_number: int
    chunk_index: int
    citation: str
    similarity_score: float
    metadata: Dict

    def __str__(self) -> str:
        return (
            f"[{self.citation}] (score: {self.similarity_score:.3f})\n"
            f"{self.text[:200]}..."
        )


_collection_cache: Dict[str, chromadb.Collection] = {}


def get_collection(collection_name: str = COLLECTION_NAME) -> Optional[chromadb.Collection]:
    """Get an existing ChromaDB collection by name. Cached after first fetch."""
    if collection_name in _collection_cache:
        return _collection_cache[collection_name]

    try:
        client = get_chroma_client()
        collection = client.get_collection(name=collection_name)
        _collection_cache[collection_name] = collection
        return collection
    except Exception as e:
        print(f"Error getting collection: {str(e)}")
        return None


def search_documents(
    query: str,
    top_k: int = TOP_K_RESULTS,
    min_score: float = MIN_SIMILARITY_SCORE,
    collection_name: str = COLLECTION_NAME
) -> List[RetrievedChunk]:
    """Search for relevant chunks by embedding similarity to the query."""
    collection = get_collection(collection_name)
    if not collection:
        print(f"ERROR: Collection '{collection_name}' not found. Have you built the index?")
        return []

    try:
        query_embedding = generate_embeddings([query], model=EMBEDDING_MODEL)[0]
    except Exception as e:
        print(f"ERROR: Failed to generate query embedding: {str(e)}")
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"ERROR: Search failed: {str(e)}")
        return []

    retrieved_chunks = []

    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]

    if not documents or not documents[0]:
        return []

    for i, (doc, metadata, distance) in enumerate(
        zip(documents[0], metadatas[0] if metadatas else [], distances[0] if distances else [])
    ):
        similarity_score = 1 - distance

        if similarity_score < min_score:
            continue

        chunk = RetrievedChunk(
            text=doc,
            document_name=cast(str, metadata.get("document_name", "Unknown")),
            page_number=cast(int, metadata.get("page_number", 0)),
            chunk_index=cast(int, metadata.get("chunk_index", 0)),
            citation=cast(str, metadata.get("citation", "No citation available")),
            similarity_score=similarity_score,
            metadata=cast(Dict[str, Any], metadata)
        )
        retrieved_chunks.append(chunk)

    return retrieved_chunks


def retrieve_with_context(
    query: str,
    top_k: int = TOP_K_RESULTS,
    min_score: float = MIN_SIMILARITY_SCORE
) -> Dict:
    """Retrieve chunks and format them with context for LLM consumption."""
    chunks = search_documents(query, top_k=top_k, min_score=min_score)

    if not chunks:
        return {
            "found": False,
            "message": "Not found in sources. The available documents do not contain information to answer this query.",
            "chunks": [],
            "context": "",
            "citations": []
        }

    chunks_by_doc = {}
    for chunk in chunks:
        doc_name = chunk.document_name
        if doc_name not in chunks_by_doc:
            chunks_by_doc[doc_name] = []
        chunks_by_doc[doc_name].append(chunk)

    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] {chunk.citation}\n{chunk.text}\n"
        )
        citations.append(chunk.citation)

    context = "\n---\n".join(context_parts)

    return {
        "found": True,
        "message": f"Found {len(chunks)} relevant chunks from {len(chunks_by_doc)} documents.",
        "chunks": chunks,
        "context": context,
        "citations": list(dict.fromkeys(citations)),
        "chunks_by_document": chunks_by_doc
    }


def multi_query_retrieval(
    queries: List[str],
    top_k_per_query: int = 3,
    min_score: float = MIN_SIMILARITY_SCORE
) -> Dict:
    """Run retrieval for multiple queries and combine deduplicated results."""
    all_chunks = []
    seen_ids = set()

    for query in queries:
        chunks = search_documents(query, top_k=top_k_per_query, min_score=min_score)

        for chunk in chunks:
            chunk_id = f"{chunk.document_name}_{chunk.page_number}_{chunk.chunk_index}"
            if chunk_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk_id)

    all_chunks.sort(key=lambda x: x.similarity_score, reverse=True)

    if not all_chunks:
        return {
            "found": False,
            "message": "Not found in sources.",
            "chunks": [],
            "context": "",
            "citations": []
        }

    context_parts = []
    citations = []
    for i, chunk in enumerate(all_chunks, 1):
        context_parts.append(f"[Source {i}] {chunk.citation}\n{chunk.text}\n")
        citations.append(chunk.citation)

    return {
        "found": True,
        "message": f"Found {len(all_chunks)} unique chunks across {len(queries)} queries.",
        "chunks": all_chunks,
        "context": "\n---\n".join(context_parts),
        "citations": list(dict.fromkeys(citations))
    }


def format_results_for_display(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as a readable string for printing."""
    if not chunks:
        return "No results found."

    output = [f"Found {len(chunks)} relevant chunks:\n"]

    for i, chunk in enumerate(chunks, 1):
        output.append(f"[{i}] {chunk.citation}")
        output.append(f"Similarity Score: {chunk.similarity_score:.3f}")
        output.append(f"Text Preview:")
        output.append(chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text)
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    test_queries = [
        "What are the key challenges in omnichannel retail operations?",
        "How can retailers improve inventory accuracy?",
        "What strategies are effective for managing returns?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve_with_context(query, top_k=3, min_score=0.3)

        if results["found"]:
            print(results["message"])
            for citation in results["citations"]:
                print(f"  - {citation}")

            if results["chunks"]:
                top = results["chunks"][0]
                print(f"  Top: {top.citation} (score: {top.similarity_score:.3f})")
                print(f"  {top.text[:200]}...")
        else:
            print(results["message"])

    multi_queries = ["supply chain visibility", "inventory management", "fulfillment automation"]
    print(f"\nMulti-query: {multi_queries}")
    multi_results = multi_query_retrieval(multi_queries, top_k_per_query=2, min_score=0.3)
    print(f"{multi_results['message']}")
    print(f"Unique chunks: {len(multi_results['chunks'])}")
