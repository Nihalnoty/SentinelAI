
import re
from pathlib import Path

from src.rag.ingest import load_text_documents


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    """Split oversized text at word boundaries where possible."""
    words = text.split()
    pieces = []
    current = ""

    for word in words:
        # A single oversized token cannot be kept within the limit
        # without splitting that token.
        if len(word) > chunk_size:
            if current:
                pieces.append(current)
                current = ""

            for start in range(0, len(word), chunk_size):
                pieces.append(word[start:start + chunk_size])
            continue

        candidate = f"{current} {word}".strip() if current else word

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
    sentence_aware: bool = False,
) -> list[str]:
    """
    Split text into bounded chunks.

    Character mode preserves the original behavior.
    Sentence-aware mode groups complete sentences when possible,
    splits oversized sentences at word boundaries, and overlaps
    using whole words rather than arbitrary character fragments.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    if not isinstance(sentence_aware, bool):
        raise TypeError("sentence_aware must be a boolean")

    text = text.strip()

    if not text:
        return []

    if not sentence_aware:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])

            if end == len(text):
                break

            start = end - overlap

        return chunks

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(_split_long_text(sentence, chunk_size))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # Add whole-word overlap from the preceding chunk.
        overlap_words = []
        overlap_length = 0

        for word in reversed(current.split()):
            extra_length = len(word) + (1 if overlap_words else 0)

            if overlap_length + extra_length > overlap:
                break

            overlap_words.insert(0, word)
            overlap_length += extra_length

        overlap_text = " ".join(overlap_words)
        candidate = f"{overlap_text} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 100,
    overlap: int = 20,
    sentence_aware: bool = False,
) -> list[dict]:
    """Chunk documents while preserving source metadata."""
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    results = []

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise TypeError(
                f"Document at index {index} must be a dictionary"
            )

        source = document.get("source")
        text = document.get("text")

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"Document at index {index} has an invalid source"
            )

        if not isinstance(text, str):
            raise ValueError(
                f"Document at index {index} has invalid text"
            )

        text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            sentence_aware=sentence_aware,
        )

        for chunk_index, chunk in enumerate(text_chunks):
            results.append({
                "source": source,
                "chunk_index": chunk_index,
                "text": chunk,
            })

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)
    chunks = chunk_documents(documents, sentence_aware=True)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nSource: {chunk['source']} | "
            f"Chunk: {chunk['chunk_index']}"
        )
        print(chunk["text"])
