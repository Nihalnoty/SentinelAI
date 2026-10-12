
from pathlib import Path

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_text_documents
from src.rag.retrieval import retrieve_chunks


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    knowledge_dir = project_root / "data" / "knowledge_base"

    documents = load_text_documents(knowledge_dir)
    chunks = chunk_documents(documents)

    test_cases = [
        {
            "query": "abnormal vibration",
            "expected_terms": ["vibration"],
        },
        {
            "query": "mounting bolts",
            "expected_terms": ["mounting bolts"],
        },
        {
            "query": "bearing condition",
            "expected_terms": ["bearing condition"],
        },
        {
            "query": "quantum astronomy",
            "expected_terms": [],
        },
    ]

    passed = 0

    for case in test_cases:
        query = case["query"]
        expected_terms = case["expected_terms"]

        results = retrieve_chunks(query, chunks, top_k=3)

        # For this small demonstration, check whether expected
        # terms occur in the retrieved text.
        retrieved_text = " ".join(
            result["text"].lower() for result in results
        )

        if expected_terms:
            success = all(
                term in retrieved_text for term in expected_terms
            )
        else:
            success = len(results) == 0

        status = "PASS" if success else "FAIL"
        passed += int(success)

        print(f"\n[{status}] Query: {query}")
        print(f"Results returned: {len(results)}")

        for result in results:
            print(
                f"  Source: {result['source']} | "
                f"Chunk: {result['chunk_index']} | "
                f"Score: {result['score']:.4f}"
            )

    print(f"\nEvaluation summary: {passed}/{len(test_cases)} passed")


if __name__ == "__main__":
    main()
