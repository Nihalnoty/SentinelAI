
def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant chunks retrieved in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    top_k_ids = set(retrieved_ids[:k])
    found = top_k_ids.intersection(relevant_ids)

    return len(found) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_results: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """Average reciprocal rank across evaluation queries."""
    if len(ranked_results) != len(relevant_sets):
        raise ValueError(
            "ranked_results and relevant_sets must have equal lengths"
        )

    if not ranked_results:
        raise ValueError("At least one evaluation query is required")

    scores = [
        reciprocal_rank(results, relevant)
        for results, relevant in zip(ranked_results, relevant_sets)
    ]

    return sum(scores) / len(scores)
