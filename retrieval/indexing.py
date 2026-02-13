"""Document indexing with ChromaDB vector store."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, cast
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI
from openai import OpenAI

from .config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DISTANCE_METRIC
)
from .ingestion import DocumentChunk, ingest_all_documents

load_dotenv()

# Cached clients so we don't recreate them on every call
_openai_client: Optional[OpenAI] = None
_chroma_client: Optional[ClientAPI] = None


def get_openai_client() -> OpenAI:
    """Get or create a cached OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def generate_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Generate embeddings for text using OpenAI API."""
    client = get_openai_client()

    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {str(e)}")


def batch_generate_embeddings(
    texts: List[str],
    batch_size: int = 100,
    model: str = EMBEDDING_MODEL
) -> List[List[float]]:
    """Generate embeddings in batches to handle API limits."""
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(
        f"Generating embeddings for {len(texts)} texts in {total_batches} batches...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(
            f"  Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")

        embeddings = generate_embeddings(batch, model)
        all_embeddings.extend(embeddings)

    print(f"Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def get_chroma_client(persist_directory: Path = CHROMA_DB_DIR) -> ClientAPI:
    """Get or create a cached ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def create_or_get_collection(
    client: ClientAPI,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False
) -> chromadb.Collection:
    """Create new collection or get existing one."""
    if reset:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": DISTANCE_METRIC,
            "description": "Retail operations documents with embeddings"
        }
    )

    return collection


def index_chunks(
    chunks: List[DocumentChunk],
    collection: chromadb.Collection,
    batch_size: int = 100
) -> None:
    """Index document chunks into ChromaDB with embeddings."""
    if not chunks:
        print("WARNING: No chunks to index!")
        return

    print(f"\nIndexing {len(chunks)} chunks into ChromaDB...")

    texts = [chunk.text for chunk in chunks]

    embeddings = batch_generate_embeddings(texts, batch_size=batch_size)

    ids = [chunk.chunk_id for chunk in chunks]
    metadatas: List[Dict[str, Any]] = [
        {
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "citation": chunk.get_citation(),
            **chunk.metadata
        }
        for chunk in chunks
    ]

    print(f"\nStoring chunks in ChromaDB...")
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch_num = i // batch_size + 1
        end_idx = min(i + batch_size, len(chunks))

        print(f"  Storing batch {batch_num}/{total_batches}...")

        collection.add(
            ids=ids[i:end_idx],
            embeddings=cast(Any, embeddings[i:end_idx]),
            documents=texts[i:end_idx],
            metadatas=cast(Any, metadatas[i:end_idx])
        )

    print(f"Indexed {len(chunks)} chunks into '{collection.name}'")


def build_index(reset: bool = False) -> Optional[chromadb.Collection]:
    """Build complete document index with ingestion and embedding."""
    print("Building document index...")

    print("\nIngesting documents...")
    chunks = ingest_all_documents()

    if not chunks:
        print("ERROR: No chunks were ingested. Cannot build index.")
        return None

    print(f"\nSetting up ChromaDB...")
    client = get_chroma_client()
    collection = create_or_get_collection(client, reset=reset)

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(
            f"Collection '{COLLECTION_NAME}' already has {existing_count} documents.")
        print("Use reset=True to rebuild the index.")
        return collection

    print(f"\nIndexing chunks...")
    index_chunks(chunks, collection)

    print(
        f"\nIndex complete: {collection.count()} documents in '{collection.name}'")
    print(f"Database location: {CHROMA_DB_DIR}")

    return collection


if __name__ == "__main__":
    collection = build_index(reset=True)
