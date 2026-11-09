"""Retrieve potentially relevant maintenance documents for an incident."""

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents
from src.rag.retrieval import retrieve_chunks


SYNTHETIC_MARKERS = (
    "synthetic testing material",
    "synthetic test material",
    "synthetic software-testing material",
    "software-testing material",
    "test fixture disclaimer",
    "synthetic test fixture",
)


AUTHORITATIVE_CFG42_AUTHORITIES = {
    "primary_site_or_asset_documentation",
    "peer_reviewed_asset_specific_research",
    "dataset_owner_documentation",
}


REQUIRED_VERIFICATION_FIELDS = (
    "source_type",
    "publisher",
    "source_reference",
    "verified_by",
    "verified_at",
)


def load_source_registry(
    knowledge_dir: str | Path,
) -> dict[str, Any]:
    """Load nested or flat source registry formats."""

    registry_path = Path(knowledge_dir) / "source_registry.json"

    if not registry_path.is_file():
        return {}

    try:
        registry = json.loads(
            registry_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read source registry: {registry_path}"
        ) from exc

    if not isinstance(registry, dict):
        raise ValueError(
            "Source registry must contain a JSON object."
        )

    sources = registry.get("sources", registry)

    if not isinstance(sources, dict):
        raise ValueError(
            "Source registry must contain a valid sources mapping."
        )

    return sources


def classify_document_source(
    text: str,
    source: str,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Classify document provenance using content and registry metadata."""

    if not isinstance(registry, dict):
        registry = {}

    entry = registry.get(source)

    if not isinstance(entry, dict):
        entry = {}

    source_id = entry.get("source_id") or source
    source_type = entry.get("source_type")
    authority = entry.get("authority", "unknown")

    verification_status = entry.get(
        "verification_status",
        "unverified",
    )

    normalized_text = text.casefold()

    matched_markers = [
        marker
        for marker in SYNTHETIC_MARKERS
        if marker in normalized_text
    ]

    is_synthetic = (
        bool(matched_markers)
        or authority == "synthetic_test_material"
        or verification_status == "synthetic"
    )

    base_metadata = {
        "source_id": source_id,
        "title": entry.get("title"),
        "publisher": entry.get("publisher"),
        "source_type": source_type,
        "authority": authority,
        "source_reference": entry.get("source_reference"),
        "additional_reference": entry.get("additional_reference"),
        "verified_by": entry.get("verified_by"),
        "verified_at": entry.get("verified_at"),
        "limitations": entry.get("limitations", []),
    }

    # Synthetic content always overrides registry verification.
    if is_synthetic:
        return {
            **base_metadata,
            "verification_status": "synthetic",
            "source_quality": "synthetic_test_material",
            "is_synthetic": True,
            "source_quality_reason": (
                "The document is identified as synthetic testing material."
            ),
            "is_authoritative_for_cfg42": False,
        }

    # Validate all required provenance fields.
    metadata_complete = all(
        isinstance(entry.get(field), str)
        and bool(entry[field].strip())
        for field in REQUIRED_VERIFICATION_FIELDS
    )

    valid_date = False

    if metadata_complete:
        try:
            date.fromisoformat(entry["verified_at"])
            valid_date = True
        except ValueError:
            pass

    identity_status_checked = (
        isinstance(verification_status, str)
        and (
            verification_status.startswith("verified_")
            or verification_status.startswith("official_domain_")
        )
    )

    identity_verified = (
        identity_status_checked
        and metadata_complete
        and valid_date
    )

    if identity_verified:
        final_status = verification_status
        source_quality = "identity_verified"
    else:
        final_status = "unverified"
        source_quality = "unverified"

    is_authoritative = (
        identity_verified
        and authority in AUTHORITATIVE_CFG42_AUTHORITIES
    )

    return {
        **base_metadata,
        "verification_status": final_status,
        "source_quality": source_quality,
        "is_synthetic": False,
        "source_quality_reason": (
            "Source identity and registry metadata are present. "
            "This does not independently validate technical claims."
            if identity_verified
            else
            "The source is not independently verified or has "
            "incomplete verification metadata."
        ),
        "is_authoritative_for_cfg42": is_authoritative,
    }


def retrieve_maintenance_evidence(
    query: str,
    *,
    knowledge_dir: str | Path | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve potentially relevant maintenance-document chunks."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer")

    if knowledge_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)
    registry = load_source_registry(knowledge_dir)

    source_metadata = {
        document["source"]: classify_document_source(
            document["text"],
            document["source"],
            registry,
        )
        for document in documents
    }

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

    evidence = []

    for chunk in retrieved:
        metadata = source_metadata.get(
            chunk["source"],
            {
                "source_id": chunk["source"],
                "source_quality": "unverified",
                "is_synthetic": False,
                "verification_status": "unverified",
                "source_quality_reason": (
                    "Source quality could not be determined."
                ),
                "is_authoritative_for_cfg42": False,
            },
        )

        if metadata["is_synthetic"]:
            interpretation = (
                "Synthetic testing material; not an authoritative "
                "maintenance reference."
            )

        elif metadata["verification_status"] == "unverified":
            interpretation = (
                "Potentially relevant document evidence, but the source "
                "is not independently verified. Similarity does not "
                "establish correctness, technical authority or root cause."
            )

        else:
            interpretation = (
                "Potentially relevant document evidence. Similarity does "
                "not establish correctness, technical authority or root cause."
            )

        evidence.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "similarity_score": float(chunk["score"]),
            "evidence_type": "retrieved_document",
            **metadata,
            "interpretation": interpretation,
        })

    return {
        "query": query.strip(),
        "documents_loaded": len(documents),
        "chunks_retrieved": len(evidence),
        "evidence": evidence,
        "answer_generated": False,
    }