"""
Configuration settings for the retrieval system.

This file centralizes all settings in order to easily adjust parameters
without modifying code in multiple places.
"""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

MIN_CHUNK_SIZE = 100

EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_DIMENSION = 1536

TOP_K_RESULTS = 5

MIN_SIMILARITY_SCORE = 0.5

# Name of the ChromaDB collection
COLLECTION_NAME = "retail_operations_docs"

DISTANCE_METRIC = "cosine"
