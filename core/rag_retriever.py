"""
core/rag_retriever.py
Retrieves relevant pricing records from ChromaDB for a QTO item.
Used by Agent #3 (Pricing Research).
"""

from __future__ import annotations
import chromadb
from openai import OpenAI
from core.config import settings
from core.logger import get_logger

log = get_logger("rag_retriever", "system")

_chroma_client = None
_collection = None
_openai_client = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def retrieve_rates(
    item_description: str,
    trade: str,
    project_type: str,
    grade: str,
    region: str,
    unit: str,
    top_k: int = None,
) -> list[dict]:
    """
    Retrieve top-k most relevant pricing records for a QTO item.

    Returns list of dicts with rate info and similarity metadata.
    """
    top_k = top_k or settings.rag_top_k
    collection = _get_collection()
    openai = _get_openai()

    # Build rich query string for semantic search
    query = (
        f"{project_type} {trade} {item_description} "
        f"unit:{unit} grade:{grade} region:{region}"
    )

    # Embed the query
    response = openai.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    query_embedding = response.data[0].embedding

    # Query ChromaDB with metadata filters
    where_filter = {"trade": {"$eq": trade}}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Fallback: query without filter if trade filter returns 0
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    hits = []
    if results and results["metadatas"]:
        for meta, dist, doc in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0],
        ):
            hits.append({
                "item":        meta.get("item", ""),
                "trade":       meta.get("trade", ""),
                "unit":        meta.get("unit", ""),
                "rate_low":    meta.get("rate_low", 0),
                "rate_high":   meta.get("rate_high", 0),
                "currency":    meta.get("currency", "SAR"),
                "region":      meta.get("region", ""),
                "grade":       meta.get("grade", ""),
                "notes":       meta.get("notes", ""),
                "similarity":  round(1 - dist, 3),
                "source_text": doc,
            })

    log.debug(f"RAG: '{item_description[:40]}' → {len(hits)} hits")
    return hits


def retrieve_batch(qto_items: list[dict], project_context: dict) -> dict:
    """
    Retrieve RAG results for all QTO items at once.
    Returns dict keyed by item_id.
    """
    results = {}
    for item in qto_items:
        try:
            hits = retrieve_rates(
                item_description=item.get("description", ""),
                trade=item.get("trade", ""),
                project_type=project_context.get("project_type", ""),
                grade=project_context.get("grade", "standard"),
                region=project_context.get("region", "KSA"),
                unit=item.get("unit", "nr"),
            )
            results[item["item_id"]] = hits
        except Exception as e:
            log.warning(f"RAG failed for {item.get('item_id')}: {e}")
            results[item["item_id"]] = []
    return results
