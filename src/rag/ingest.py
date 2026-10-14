from collections import Counter
from pathlib import Path


def load_text_documents(directory: str | Path) -> list[dict]:
    """
    Load TXT documents recursively with stable, unambiguous source metadata.
    """
    knowledge_dir = Path(directory)

    if not knowledge_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge directory not found: {knowledge_dir}"
        )

    file_paths = sorted(knowledge_dir.rglob("*.txt"))

    filename_counts = Counter(
        file_path.name for file_path in file_paths
    )

    documents = []

    for file_path in file_paths:
        text = file_path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not text:
            continue

        if filename_counts[file_path.name] > 1:
            source = file_path.relative_to(
                knowledge_dir
            ).as_posix()
        else:
            source = file_path.name

        documents.append({
            "source": source,
            "text": text,
            "character_count": len(text),
        })

    return documents


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        print(f"\nSource: {document['source']}")
        print(f"Characters: {document['character_count']}")
        print(document["text"])
