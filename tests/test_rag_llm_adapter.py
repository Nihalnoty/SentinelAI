import pytest

from src.rag.llm_adapter import (
    LLMConfigurationError,
    LLMProviderError,
    generate_response,
)


def test_generate_response_uses_provider_function():
    response = generate_response(
        "Answer using the evidence.",
        provider_function=lambda prompt: '{"answerable": false}',
    )

    assert response == '{"answerable": false}'


def test_generate_response_rejects_empty_prompt():
    with pytest.raises(ValueError, match="non-empty"):
        generate_response("  ", provider_function=lambda prompt: "response")


def test_generate_response_requires_provider_configuration(monkeypatch):
    monkeypatch.delenv("SENTINELAI_LLM_PROVIDER", raising=False)

    with pytest.raises(LLMConfigurationError, match="not configured"):
        generate_response("Test prompt")


def test_generate_response_reports_unimplemented_provider(monkeypatch):
    monkeypatch.setenv("SENTINELAI_LLM_PROVIDER", "example")

    with pytest.raises(LLMConfigurationError, match="not been implemented"):
        generate_response("Test prompt")


def test_generate_response_wraps_provider_errors():
    def failing_provider(prompt: str) -> str:
        raise RuntimeError("Internal provider detail")

    with pytest.raises(LLMProviderError, match="provider failed"):
        generate_response("Test prompt", provider_function=failing_provider)


def test_generate_response_rejects_empty_provider_response():
    with pytest.raises(LLMProviderError, match="empty or invalid"):
        generate_response(
            "Test prompt",
            provider_function=lambda prompt: "  ",
        )
