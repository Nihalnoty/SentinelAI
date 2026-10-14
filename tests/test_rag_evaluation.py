
import pytest

from src.rag.run_evaluation import (
    get_chunk_id,
    validate_evaluation_cases,
)


SAMPLE_CHUNKS = [
    {
        "source": "guide.txt",
        "chunk_index": 0,
        "text": "Abnormal vibration requires inspection.",
    },
    {
        "source": "guide.txt",
        "chunk_index": 1,
        "text": "Inspect mounting bolts and bearing condition.",
    },
]


def test_chunk_id_uses_source_and_index():
    assert get_chunk_id(SAMPLE_CHUNKS[0]) == "guide.txt::0"


def test_chunk_id_differs_for_different_chunks():
    assert get_chunk_id(SAMPLE_CHUNKS[0]) != get_chunk_id(SAMPLE_CHUNKS[1])


def test_chunk_id_requires_expected_fields():
    with pytest.raises(KeyError):
        get_chunk_id({"source": "guide.txt"})


def test_valid_evaluation_cases_are_accepted():
    cases = [
        {
            "query": "mounting bolts",
            "relevant_chunk_ids": ["guide.txt::1"],
        }
    ]

    validate_evaluation_cases(cases, SAMPLE_CHUNKS)


def test_unknown_chunk_id_is_rejected():
    cases = [
        {
            "query": "mounting bolts",
            "relevant_chunk_ids": ["missing.txt::0"],
        }
    ]

    with pytest.raises(ValueError, match="unknown chunk IDs"):
        validate_evaluation_cases(cases, SAMPLE_CHUNKS)


def test_empty_evaluation_dataset_is_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        validate_evaluation_cases([], SAMPLE_CHUNKS)


def test_empty_query_is_rejected():
    cases = [{"query": "  ", "relevant_chunk_ids": []}]

    with pytest.raises(ValueError, match="non-empty query"):
        validate_evaluation_cases(cases, SAMPLE_CHUNKS)


def test_empty_relevant_ids_are_accepted_for_unanswerable_query():
    cases = [
        {
            "query": "What is the maximum operating temperature?",
            "relevant_chunk_ids": [],
        }
    ]

    validate_evaluation_cases(cases, SAMPLE_CHUNKS)
