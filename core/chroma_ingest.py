"""
core/chroma_ingest.py
Loads pricing_data.csv into ChromaDB for RAG retrieval.
Run once before first use, and re-run whenever pricing data is updated.

Usage:
    python core/chroma_ingest.py
"""

import csv
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from openai import OpenAI
from core.config import settings
from core.logger import get_logger

log = get_logger("chroma_ingest", "system")

_ROOT = Path(__file__).parent.parent
_CSV_PATH = _ROOT / "knowledge_base" / "pricing_data.csv"


def ingest():
    log.info("Starting ChromaDB ingestion")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Connect to OpenAI for embeddings
    openai_client = OpenAI(api_key=settings.openai_api_key)

    # Load CSV
    if not _CSV_PATH.exists():
        log.error(f"pricing_data.csv not found at {_CSV_PATH}")
        sys.exit(1)

    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    log.info(f"Loaded {len(rows)} rows from pricing_data.csv")

    # Build documents for embedding
    documents, metadatas, ids = [], [], []

    for i, row in enumerate(rows):
        # Create a rich text description for semantic search
        doc_text = (
            f"{row['project_type']} {row['trade']} {row['category']} "
            f"{row['item']} unit:{row['unit']} "
            f"grade:{row['grade']} region:{row['region']} "
            f"rate:{row['rate_low']}-{row['rate_high']} {row['currency']}"
        )
        if row.get("notes"):
            doc_text += f" notes:{row['notes']}"

        documents.append(doc_text)
        metadatas.append({
            "project_type": row["project_type"],
            "trade":        row["trade"],
            "category":     row.get("category", ""),
            "item":         row["item"],
            "unit":         row["unit"],
            "rate_low":     float(row["rate_low"]),
            "rate_high":    float(row["rate_high"]),
            "grade":        row["grade"],
            "currency":     row["currency"],
            "region":       row["region"],
            "notes":        row.get("notes", ""),
        })
        ids.append(f"pricing_{i:04d}")

    # Generate embeddings in batches of 50
    batch_size = 50
    all_embeddings = []

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        log.info(f"Embedding batch {start//batch_size + 1} ({len(batch)} items)")
        response = openai_client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        all_embeddings.extend([r.embedding for r in response.data])

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    count = collection.count()
    log.info(f"ChromaDB ingestion complete — {count} items in collection")
    print(f"\n✅ Ingested {count} pricing records into ChromaDB")
    print(f"   Collection: {settings.chroma_collection_name}")
    print(f"   Path: {settings.chroma_db_path}\n")


if __name__ == "__main__":
    ingest()
