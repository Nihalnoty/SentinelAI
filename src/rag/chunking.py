
from pathlib import Path

from src.rag.ingest import load_text_documents


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    """Split text into character-based chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[dict]:
    """Chunk documents while preserving source metadata."""
    results = []

    for document in documents:
        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(text_chunks):
            results.append({
                "source": document["source"],
                "chunk_index": index,
                "text": chunk,
            })

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)
    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nSource: {chunk['source']} | "
            f"Chunk: {chunk['chunk_index']}"
        )
        print(chunk["text"])
