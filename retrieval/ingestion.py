"""
Document Ingestion Module

Handles loading PDF documents, extracting text, and splitting into chunks.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

from pypdf import PdfReader

from .config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE
)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata for citations."""
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

    def get_citation(self) -> str:
        return f"{self.document_name}, Page {self.page_number}, Chunk {self.chunk_index}"


def generate_chunk_id(document_name: str, page_number: int, chunk_index: int) -> str:
    """Generate a unique ID for a chunk using MD5 hash."""
    id_string = f"{document_name}_{page_number}_{chunk_index}"
    hash_object = hashlib.md5(id_string.encode())
    return hash_object.hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end < text_length and text[end] != " ":
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end]

        if len(chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk.strip())

        start += chunk_size - chunk_overlap

    return chunks


def load_pdf(pdf_path: Path) -> List[tuple]:
    """Load a PDF file and extract text from each page."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                pages.append((page_num, text))

        return pages

    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path.name}: {str(e)}")


def process_document(pdf_path: Path) -> List[DocumentChunk]:
    """Process a single PDF document into chunks."""
    document_name = pdf_path.name
    chunks = []

    print(f"Processing: {document_name}")

    try:
        pages = load_pdf(pdf_path)

        for page_number, page_text in pages:
            text_chunks = chunk_text(page_text)

            for chunk_index, chunk_content in enumerate(text_chunks):
                chunk_id = generate_chunk_id(
                    document_name, page_number, chunk_index)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "file_path": str(pdf_path),
                        "total_chunks_on_page": len(text_chunks),
                        "chunk_size": len(chunk_content)
                    }
                )
                chunks.append(chunk)

        print(f"Extracted {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    except Exception as e:
        print(f"Error: {str(e)}")
        return []


def ingest_all_documents(data_dir: Path = DATA_DIR) -> List[DocumentChunk]:
    """Ingest all PDF documents from the data directory."""
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data directory")
        return []

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = process_document(pdf_path)
        all_chunks.extend(chunks)

    print(f"Processed {len(pdf_files)} documents, created {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_all_documents()
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk from '{sample.document_name}' (page {sample.page_number}):")
        print(sample.text[:200] + "..." if len(sample.text) > 200 else sample.text)
