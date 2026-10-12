
import pytest

from src.rag.metrics import (
    mean_reciprocal_rank,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_at_k_finds_relevant_chunks():
    retrieved = ["chunk-a", "chunk-b", "chunk-c"]
    relevant = {"chunk-b", "chunk-c"}

    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k_finds_all_relevant_chunks():
    retrieved = ["chunk-a", "chunk-b", "chunk-c"]
    relevant = {"chunk-b", "chunk-c"}

    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_reciprocal_rank_for_second_result():
    retrieved = ["chunk-a", "chunk-b", "chunk-c"]

    assert reciprocal_rank(retrieved, {"chunk-b"}) == 0.5


def test_reciprocal_rank_is_zero_when_not_found():
    assert reciprocal_rank(["chunk-a"], {"chunk-z"}) == 0.0


def test_mean_reciprocal_rank():
    rankings = [
        ["chunk-a", "chunk-b"],
        ["chunk-c", "chunk-d"],
    ]
    relevant_sets = [
        {"chunk-b"},
        {"chunk-c"},
    ]

    assert mean_reciprocal_rank(rankings, relevant_sets) == 0.75


def test_recall_rejects_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["chunk-a"], {"chunk-a"}, k=0)
