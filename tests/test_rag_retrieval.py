
import pytest

from src.rag.retrieval import retrieve_chunks


SAMPLE_CHUNKS = [
    {
        "source": "maintenance.txt",
        "chunk_index": 0,
        "text": "Inspect the equipment for abnormal vibration.",
    },
    {
        "source": "temperature.txt",
        "chunk_index": 0,
        "text": "Monitor bearing temperature during operation.",
    },
]


def test_retrieval_returns_relevant_chunk():
    results = retrieve_chunks(
        "abnormal vibration",
        SAMPLE_CHUNKS,
        top_k=2,
    )

    assert len(results) > 0
    assert results[0]["source"] == "maintenance.txt"
    assert results[0]["score"] > 0


def test_retrieval_preserves_source_and_text():
    results = retrieve_chunks(
        "bearing temperature",
        SAMPLE_CHUNKS,
    )

    assert results[0]["source"] == "temperature.txt"
    assert "bearing temperature" in results[0]["text"]


def test_unrelated_query_returns_no_matches():
    results = retrieve_chunks(
        "quantum astronomy",
        SAMPLE_CHUNKS,
    )

    assert results == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        retrieve_chunks("   ", SAMPLE_CHUNKS)


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError):
        retrieve_chunks("vibration", SAMPLE_CHUNKS, top_k=0)


def test_empty_chunk_list_returns_no_results():
    assert retrieve_chunks("vibration", []) == []


def test_retrieval_can_return_chunks_for_unsupported_question():
    results = retrieve_chunks(
        "What is the maximum bearing temperature?",
        SAMPLE_CHUNKS,
        top_k=3,
    )

    # TF-IDF may return text because some words overlap,
    # even though the documents do not state a maximum temperature.
    assert results
