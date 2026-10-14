from src.tools.rag_evidence_tool import retrieve_maintenance_evidence


def test_retrieves_maintenance_evidence_from_fixture(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    (knowledge_dir / "maintenance.txt").write_text(
        "Abnormal vibration: inspect mounting bolts and bearing "
        "condition. Follow site safety procedures.",
        encoding="utf-8",
    )

    result = retrieve_maintenance_evidence(
        "What should be inspected for abnormal vibration?",
        knowledge_dir=knowledge_dir,
    )

    assert result["documents_loaded"] == 1
    assert result["chunks_retrieved"] > 0
    assert result["answer_generated"] is False
    assert result["query"] == (
        "What should be inspected for abnormal vibration?"
    )

    evidence = result["evidence"][0]
    assert evidence["source"] == "maintenance.txt"
    assert evidence["text"]
    assert isinstance(evidence["similarity_score"], float)
    assert evidence["evidence_type"] == "retrieved_document"
    assert "not independently verified" in evidence["interpretation"]


def test_returns_empty_evidence_for_empty_knowledge_base(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    result = retrieve_maintenance_evidence(
        "What should be inspected?",
        knowledge_dir=knowledge_dir,
    )

    assert result["documents_loaded"] == 0
    assert result["chunks_retrieved"] == 0
    assert result["evidence"] == []
    assert result["answer_generated"] is False


def test_rejects_empty_query(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    try:
        retrieve_maintenance_evidence(
            "   ",
            knowledge_dir=knowledge_dir,
        )
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an empty query")


def test_rejects_non_positive_top_k(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    try:
        retrieve_maintenance_evidence(
            "abnormal vibration",
            knowledge_dir=knowledge_dir,
            top_k=0,
        )
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError for top_k=0")