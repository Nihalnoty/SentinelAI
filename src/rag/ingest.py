
from pathlib import Path


def load_text_documents(directory: str | Path) -> list[dict]:
    """
    Load local TXT documents and return their text and source metadata.
    """
    knowledge_dir = Path(directory)

    if not knowledge_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge directory not found: {knowledge_dir}"
        )

    documents = []

    for file_path in sorted(knowledge_dir.rglob("*.txt")):
        text = file_path.read_text(encoding="utf-8-sig").strip()

        if not text:
            continue

        documents.append({
            "source": file_path.name,
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
