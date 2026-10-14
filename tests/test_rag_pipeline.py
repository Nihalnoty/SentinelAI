import json

from src.rag.pipeline import ABSTENTION_MESSAGE, answer_question


SAMPLE_CHUNKS = [
    {
        "source": "maintenance.txt",
        "chunk_index": 0,
        "text": (
            "Inspect mounting bolts and bearing condition "
            "if abnormal vibration occurs."
        ),
    }
]


def test_answer_question_returns_cited_answer():
    def fake_model(prompt: str) -> str:
        assert "abnormal vibration" in prompt
        return json.dumps({
            "answerable": True,
            "answer": "Inspect the mounting bolts and bearing condition.",
            "citations": ["maintenance.txt::0"],
        })

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is True
    assert result["citations"] == ["maintenance.txt::0"]
    assert result["evidence"]


def test_abstains_when_no_chunks_are_retrieved():
    def unexpected_model_call(prompt: str) -> str:
        raise AssertionError("Model must not run without retrieved evidence")

    result = answer_question(
        "What is the rated power consumption?",
        [],
        generate_response=unexpected_model_call,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []


def test_abstains_when_model_reports_insufficient_evidence():
    def fake_model(prompt: str) -> str:
        return json.dumps({
            "answerable": False,
            "answer": "Insufficient evidence",
            "citations": [],
        })

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []
    assert result["evidence"]


def test_rejects_citations_not_in_retrieved_evidence():
    def fake_model(prompt: str) -> str:
        return json.dumps({
            "answerable": True,
            "answer": "The machine requires inspection.",
            "citations": ["unknown.txt::99"],
        })

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []


def test_rejects_answer_without_citations():
    def fake_model(prompt: str) -> str:
        return json.dumps({
            "answerable": True,
            "answer": "Inspect the mounting bolts.",
            "citations": [],
        })

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is False
    assert result["citations"] == []


def test_abstains_on_invalid_model_json():
    def fake_model(prompt: str) -> str:
        return "This is not valid JSON"

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []


def test_answer_question_abstains_when_provider_raises():
    from src.rag.pipeline import ABSTENTION_MESSAGE, answer_question

    chunks = [
        {
            "source": "centrifuge_maintenance.txt",
            "chunk_index": 0,
            "text": (
                "Inspect abnormal vibration and check bearing "
                "temperature during centrifuge maintenance."
            ),
        }
    ]

    def failing_provider(prompt: str) -> str:
        raise RuntimeError("Simulated provider outage")

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        chunks,
        generate_response=failing_provider,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []
    assert len(result["evidence"]) > 0



def test_abstains_when_model_returns_json_array():
    def fake_model(prompt: str) -> str:
        return json.dumps([])

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []


def test_abstains_when_answer_field_is_empty():
    def fake_model(prompt: str) -> str:
        return json.dumps({
            "answerable": True,
            "answer": "   ",
            "citations": ["maintenance.txt::0"],
        })

    result = answer_question(
        "What should be inspected for abnormal vibration?",
        SAMPLE_CHUNKS,
        generate_response=fake_model,
    )

    assert result["answerable"] is False
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["citations"] == []
