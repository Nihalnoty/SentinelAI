"""Retrieve potentially relevant maintenance documents for an incident."""

from pathlib import Path
from typing import Any

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents
from src.rag.retrieval import retrieve_chunks


def retrieve_maintenance_evidence(
    query: str,
    *,
    knowledge_dir: str | Path | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Retrieve potentially relevant maintenance-document chunks.

    This function performs retrieval only. Similarity scores do not prove
    that a chunk answers the query or establishes an incident's root cause.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if knowledge_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)

    chunks = chunk_documents(
        documents,
        chunk_size=300,
        overlap=40,
        sentence_aware=True,
    )

    retrieved = retrieve_chunks(
        query,
        chunks,
        top_k=top_k,
        min_similarity=0.0,
    )

    evidence = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "similarity_score": float(chunk["score"]),
            "evidence_type": "retrieved_document",
            "interpretation": (
                "Potentially relevant document evidence; "
                "not independently verified as an answer."
            ),
        }
        for chunk in retrieved
    ]

    return {
        "query": query.strip(),
        "documents_loaded": len(documents),
        "chunks_retrieved": len(evidence),
        "evidence": evidence,
        "answer_generated": False,
    }
