from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents


def retrieve_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> list[dict]:
    """Rank chunks by TF-IDF cosine similarity."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, document_vectors).flatten()
    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score <= 0 or score < min_similarity:
            continue

        result = dict(chunks[index])
        result["score"] = score
        results.append(result)

        if len(results) >= top_k:
            break

    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture_dir = project_root / "tests" / "fixtures"

    documents = load_text_documents(fixture_dir)
    chunks = chunk_documents(documents)

    queries = [
        "abnormal vibration",
        "What should be checked if the machine shakes?",
        "What is the maximum operating temperature?",
        "What is the recommended vibration limit in millimeters per second?",
        "What lubricant grade is required for this centrifuge?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        for threshold in (0.0, 0.2, 0.3, 0.4, 0.5):
            results = retrieve_chunks(
                query,
                chunks,
                top_k=3,
                min_similarity=threshold,
            )
            scores = [f"{item['score']:.4f}" for item in results]
            print(f"Threshold {threshold:.1f}: {scores or 'No results'}")


if __name__ == "__main__":
    main()
