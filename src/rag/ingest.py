"""Registry-aware ingestion for SentinelAI knowledge-base documents."""

import json
import warnings
from datetime import date
from pathlib import Path
from typing import Any


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


def _load_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires pypdf. Install project requirements with: "
            "pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()

        if page_text:
            pages.append(f"[PDF page {page_number}]\n{page_text}")

    return "\n\n".join(pages)


def _load_registry(knowledge_dir: Path) -> dict[str, Any]:
    """Load nested or legacy flat source registry formats."""

    registry_path = knowledge_dir / "source_registry.json"

    if not registry_path.is_file():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8-sig") as file:
            registry_data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read source registry: {registry_path}"
        ) from exc

    if not isinstance(registry_data, dict):
        raise ValueError(
            "source_registry.json must contain a JSON object."
        )

    if "sources" in registry_data:
        registry = registry_data["sources"]
    else:
        registry = registry_data

    if not isinstance(registry, dict):
        raise ValueError(
            "Source registry must contain a valid sources mapping."
        )

    return registry


def _source_metadata(
    source_key: str,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Map registry metadata into explicit, validated provenance fields."""

    entry = registry.get(source_key)

    if not isinstance(entry, dict):
        entry = {}

    authority = entry.get("authority", "unknown")
    verification_status = entry.get(
        "verification_status",
        "unverified",
    )

    is_synthetic = (
        authority == "synthetic_test_material"
        or verification_status == "synthetic"
    )

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
        not is_synthetic
        and identity_status_checked
        and metadata_complete
        and valid_date
    )

    if is_synthetic:
        final_status = "synthetic"
        source_quality = "synthetic"
    elif identity_verified:
        final_status = verification_status
        source_quality = "identity_verified"
    else:
        final_status = "unverified"
        source_quality = "unverified"

    return {
        "source_id": entry.get("source_id") or source_key,
        "title": entry.get("title"),
        "publisher": entry.get("publisher"),
        "source_type": entry.get("source_type"),
        "authority": authority,
        "source_reference": entry.get("source_reference"),
        "verification_status": final_status,
        "verified_by": entry.get("verified_by"),
        "verified_at": entry.get("verified_at"),
        "limitations": entry.get("limitations", []),
        "is_synthetic": is_synthetic,
        "source_quality": source_quality,
        "source_quality_reason": (
            "The document is identified as synthetic testing material."
            if is_synthetic
            else
            "Source identity and required metadata are present. "
            "This does not independently validate every technical claim."
            if identity_verified
            else
            "Source is not independently verified or has incomplete "
            "verification metadata."
        ),
        "is_authoritative_for_cfg42": (
            identity_verified
            and authority in AUTHORITATIVE_CFG42_AUTHORITIES
        ),
    }


def load_documents(
    directory: str | Path,
    *,
    include_synthetic: bool = True,
    include_pdfs: bool = True,
) -> list[dict[str, Any]]:
    """Load TXT/PDF files and attach registry provenance."""

    knowledge_dir = Path(directory)

    if not knowledge_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge directory not found: {knowledge_dir}"
        )

    registry = _load_registry(knowledge_dir)

    patterns = ["*.txt"]

    if include_pdfs:
        patterns.append("*.pdf")

    paths = sorted(
        path
        for pattern in patterns
        for path in knowledge_dir.rglob(pattern)
        if path.name.lower() != "source_registry.json"
    )

    # Prefer the shallowest copy of duplicate filenames.
    selected_paths = []
    seen: dict[str, tuple[bytes, Path]] = {}

    for path in sorted(
        paths,
        key=lambda item: (
            len(item.relative_to(knowledge_dir).parts),
            str(item),
        ),
    ):
        raw = path.read_bytes()

        if path.name in seen:
            previous_bytes, previous_path = seen[path.name]

            if raw != previous_bytes:
                warnings.warn(
                    f"Ignoring deeper duplicate '{path}' because "
                    f"'{previous_path}' takes precedence.",
                    RuntimeWarning,
                )

            continue

        seen[path.name] = (raw, path)
        selected_paths.append(path)

    documents = []

    for path in selected_paths:
        metadata = _source_metadata(path.name, registry)

        if metadata["is_synthetic"] and not include_synthetic:
            continue

        if path.suffix.lower() == ".pdf":
            text = _load_pdf_text(path)
        else:
            text = path.read_text(
                encoding="utf-8-sig"
            ).strip()

        if not text:
            continue

        documents.append({
            **metadata,
            "source": path.name,
            "text": text,
            "character_count": len(text),
            "file_type": path.suffix.lower().lstrip("."),
        })

    return documents


def load_text_documents(
    directory: str | Path,
) -> list[dict[str, Any]]:
    """Backward-compatible TXT-only document loader."""

    return load_documents(
        directory,
        include_pdfs=False,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    candidates = [
        project_root / "data" / "knowledge_base",
        project_root / "knowledge_base",
    ]

    knowledge_dir = next(
        (path for path in candidates if path.is_dir()),
        candidates[0],
    )

    loaded = load_documents(knowledge_dir)

    print(f"Documents loaded: {len(loaded)}")

    for document in loaded:
        print(
            f"{document['source']} | "
            f"{document['file_type']} | "
            f"{document['authority']} | "
            f"{document['verification_status']} | "
            f"{document['character_count']} chars"
        )