import argparse
import json
from pathlib import Path
from statistics import mean

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents
from src.rag.metrics import mean_reciprocal_rank, recall_at_k
from src.rag.retrieval import retrieve_chunks


def get_chunk_id(chunk: dict) -> str:
    """Create an ID for a document chunk."""
    return f"{chunk['source']}::{chunk['chunk_index']}"


def validate_evaluation_cases(
    evaluation_cases: list[dict],
    chunks: list[dict],
) -> None:
    """Validate evaluation queries and their relevant chunk IDs."""
    if not isinstance(evaluation_cases, list) or not evaluation_cases:
        raise ValueError("Evaluation dataset must be a non-empty list")

    available_ids = {get_chunk_id(chunk) for chunk in chunks}

    for index, case in enumerate(evaluation_cases):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index} must be an object")

        query = case.get("query")
        relevant_ids = case.get("relevant_chunk_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Evaluation case {index} must have a non-empty query"
            )

        if (
            not isinstance(relevant_ids, list)
            or not all(
                isinstance(chunk_id, str) and chunk_id.strip()
                for chunk_id in relevant_ids
            )
        ):
            raise ValueError(
                f"Evaluation case {index} must have a valid "
                "relevant_chunk_ids list"
            )

        unknown_ids = set(relevant_ids) - available_ids

        if unknown_ids:
            raise ValueError(
                f"Evaluation case {index} references unknown chunk IDs: "
                f"{sorted(unknown_ids)}. The evaluation labels may need "
                "to be updated for this chunking configuration."
            )


def print_score_summary(label: str, scores: list[float]) -> None:
    """Print descriptive statistics for retrieval confidence scores."""
    print(f"\n{label}")

    if not scores:
        print("No scores available.")
        return

    print(f"Queries with results: {len(scores)}")
    print(f"Mean: {mean(scores):.3f}")
    print(f"Minimum: {min(scores):.3f}")
    print(f"Maximum: {max(scores):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SentinelAI RAG retrieval."
    )

    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use the committed synthetic test document.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum cosine similarity required to return a chunk.",
    )
    parser.add_argument(
        "--sentence-aware",
        action="store_true",
        help="Use sentence-aware chunking for evaluation.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.min_similarity <= 1.0:
        parser.error("--min-similarity must be between 0 and 1")

    project_root = Path(__file__).resolve().parents[2]
    evaluation_file = (
        project_root / "evaluation" / "retrieval_queries.json"
    )

    if args.use_fixture:
        document_dir = project_root / "tests" / "fixtures"
        print("Document source: committed synthetic test fixture")
    else:
        document_dir = project_root / "data" / "knowledge_base"
        print("Document source: local knowledge base")

    chunking_mode = (
        "sentence-aware" if args.sentence_aware else "character-based"
    )

    print(f"Chunking mode: {chunking_mode}")
    print(f"Minimum similarity: {args.min_similarity:.2f}")

    documents = load_text_documents(document_dir)
    chunks = chunk_documents(
        documents,
        sentence_aware=args.sentence_aware,
    )

    if not chunks:
        raise ValueError(
            "No chunks were created. Check the selected document directory."
        )

    with evaluation_file.open("r", encoding="utf-8") as file:
        evaluation_cases = json.load(file)

    # Chunk indices can change between chunking modes. Stop rather than
    # calculating misleading metrics using labels for a different layout.
    validate_evaluation_cases(evaluation_cases, chunks)

    recall_scores = []
    reciprocal_ranks = []

    answerable_top_scores = []
    unanswerable_top_scores = []
    answerable_score_gaps = []
    unanswerable_score_gaps = []

    answerable_count = 0
    unanswerable_count = 0
    unanswerable_with_results = 0

    for case in evaluation_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        results = retrieve_chunks(
            query,
            chunks,
            top_k=3,
            min_similarity=args.min_similarity,
        )

        retrieved_ids = [get_chunk_id(result) for result in results]
        scores = [result["score"] for result in results]

        print(f"\nQuery: {query}")
        print(f"Relevant IDs: {sorted(relevant_ids)}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Similarity scores: {[round(s, 3) for s in scores]}")

        top_score = scores[0] if scores else None
        score_gap = (
            scores[0] - scores[1]
            if len(scores) >= 2
            else None
        )

        print(
            f"Top similarity score: {top_score:.3f}"
            if top_score is not None
            else "Top similarity score: unavailable (no results)"
        )
        print(
            f"Top-two score gap: {score_gap:.3f}"
            if score_gap is not None
            else "Top-two score gap: unavailable (fewer than 2 results)"
        )

        if relevant_ids:
            answerable_count += 1

            recall = recall_at_k(retrieved_ids, relevant_ids, k=3)
            mrr = mean_reciprocal_rank([retrieved_ids], [relevant_ids])

            recall_scores.append(recall)
            reciprocal_ranks.append(mrr)

            if top_score is not None:
                answerable_top_scores.append(top_score)
            if score_gap is not None:
                answerable_score_gaps.append(score_gap)

            print(f"Recall@3: {recall:.3f}")
            print(f"Reciprocal rank: {mrr:.3f}")

        else:
            unanswerable_count += 1

            if results:
                unanswerable_with_results += 1

            if top_score is not None:
                unanswerable_top_scores.append(top_score)
            if score_gap is not None:
                unanswerable_score_gaps.append(score_gap)

            print("Expected: no relevant evidence in the knowledge base")
            print(f"Returned any chunks: {'Yes' if results else 'No'}")

    print("\n--- Evaluation Summary ---")
    print(f"Queries evaluated: {len(evaluation_cases)}")
    print(f"Answerable queries: {answerable_count}")
    print(f"Unanswerable queries: {unanswerable_count}")

    if recall_scores:
        print(f"Mean Recall@3: {mean(recall_scores):.3f}")
        print(f"Mean Reciprocal Rank: {mean(reciprocal_ranks):.3f}")

    if unanswerable_count:
        print(
            "Unanswerable queries returning chunks: "
            f"{unanswerable_with_results}/{unanswerable_count}"
        )

    print("\n--- Retrieval Confidence Diagnostics ---")

    print_score_summary(
        "Answerable queries — top similarity scores",
        answerable_top_scores,
    )
    print_score_summary(
        "Unanswerable queries — top similarity scores",
        unanswerable_top_scores,
    )
    print_score_summary(
        "Answerable queries — top-two score gaps",
        answerable_score_gaps,
    )
    print_score_summary(
        "Unanswerable queries — top-two score gaps",
        unanswerable_score_gaps,
    )

    print(
        "\nNote: Similarity scores and score gaps are descriptive signals, "
        "not a validated answerability classifier."
    )


if __name__ == "__main__":
    main()
