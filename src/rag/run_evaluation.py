
import argparse
import json
from pathlib import Path

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents
from src.rag.metrics import mean_reciprocal_rank, recall_at_k
from src.rag.retrieval import retrieve_chunks


def get_chunk_id(chunk: dict) -> str:
    """Create a stable ID for a document chunk."""
    return f"{chunk['source']}::{chunk['chunk_index']}"


def validate_evaluation_cases(
    evaluation_cases: list[dict],
    chunks: list[dict],
) -> None:
    """Validate queries and their relevant chunk IDs."""
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
                f"{sorted(unknown_ids)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SentinelAI RAG retrieval."
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use the committed synthetic test document.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    evaluation_file = project_root / "evaluation" / "retrieval_queries.json"

    if args.use_fixture:
        document_dir = project_root / "tests" / "fixtures"
        print("Document source: committed synthetic test fixture")
    else:
        document_dir = project_root / "data" / "knowledge_base"
        print("Document source: local knowledge base")

    documents = load_text_documents(document_dir)
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created. Check the selected document directory."
        )

    with evaluation_file.open("r", encoding="utf-8") as file:
        evaluation_cases = json.load(file)

    validate_evaluation_cases(evaluation_cases, chunks)

    recall_scores = []
    reciprocal_ranks = []
    answerable_count = 0
    unanswerable_count = 0
    unanswerable_with_results = 0

    for case in evaluation_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        results = retrieve_chunks(query, chunks, top_k=3)
        retrieved_ids = [get_chunk_id(result) for result in results]

        print(f"\nQuery: {query}")
        print(f"Relevant IDs: {sorted(relevant_ids)}")
        print(f"Retrieved IDs: {retrieved_ids}")

        if relevant_ids:
            answerable_count += 1

            recall = recall_at_k(retrieved_ids, relevant_ids, k=3)
            mrr = mean_reciprocal_rank([retrieved_ids], [relevant_ids])

            recall_scores.append(recall)
            reciprocal_ranks.append(mrr)

            print(f"Recall@3: {recall:.3f}")
            print(f"Reciprocal rank: {mrr:.3f}")
        else:
            unanswerable_count += 1

            if results:
                unanswerable_with_results += 1

            print("Expected: no relevant evidence in the knowledge base")
            print(f"Returned any chunks: {'Yes' if results else 'No'}")

    print("\n--- Evaluation Summary ---")
    print(f"Queries evaluated: {len(evaluation_cases)}")
    print(f"Answerable queries: {answerable_count}")
    print(f"Unanswerable queries: {unanswerable_count}")

    if recall_scores:
        print(f"Mean Recall@3: {sum(recall_scores) / len(recall_scores):.3f}")
        print(
            "Mean Reciprocal Rank: "
            f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        )

    if unanswerable_count:
        print(
            "Unanswerable queries returning chunks: "
            f"{unanswerable_with_results}/{unanswerable_count}"
        )


if __name__ == "__main__":
    main()
