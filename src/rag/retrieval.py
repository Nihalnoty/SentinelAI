
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents


def retrieve_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Rank document chunks by TF-IDF cosine similarity to a query.

    Each chunk must contain:
        - source: original filename
        - chunk_index: position within the document
        - text: chunk content
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    # Learn vocabulary and calculate TF-IDF vectors for the chunks.
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(texts)

    # Represent the user's query using the same vocabulary.
    query_vector = vectorizer.transform([query])

    # Calculate a similarity score between the query and each chunk.
    scores = cosine_similarity(query_vector, document_vectors).flatten()

    # Rank the chunks from highest similarity to lowest.
    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Don't return chunks that have no matching TF-IDF features.
        if score <= 0:
            continue

        result = dict(chunks[index])
        result["score"] = score
        results.append(result)

        if len(results) >= top_k:
            break

    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)
    chunks = chunk_documents(documents)

    query = "abnormal vibration"

    results = retrieve_chunks(query, chunks, top_k=3)

    print(f"Query: {query}")
    print(f"Matching chunks: {len(results)}")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {result['source']}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
