import pytest
import src.rag.llm_adapter as adapter
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
def test_gemini_429_stops_without_retry(monkeypatch):
    """A 429 quota error should stop immediately."""

    class FakeRateLimitError(Exception):
        code = 429

    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise FakeRateLimitError("Quota exceeded")

    fake_models = FakeModels()

    class FakeClient:
        models = fake_models

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    monkeypatch.setattr(
        adapter.genai,
        "Client",
        lambda **kwargs: FakeClient(),
    )

    with pytest.raises(LLMProviderError, match="rate limit"):
        adapter._generate_with_gemini("Test prompt")

    assert fake_models.calls == 1


def test_gemini_503_retries_and_succeeds(monkeypatch):
    """A temporary 503 error should retry before succeeding."""

    class FakeServerError(Exception):
        code = 503

    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1

            if self.calls < 3:
                raise FakeServerError("Temporary server error")

            return type(
                "FakeResponse",
                (),
                {"text": '{"answerable": true}'},
            )()

    fake_models = FakeModels()

    class FakeClient:
        models = fake_models

    delays = []

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    monkeypatch.setattr(
        adapter.genai,
        "Client",
        lambda **kwargs: FakeClient(),
    )

    monkeypatch.setattr(
        adapter.time,
        "sleep",
        lambda seconds: delays.append(seconds),
    )

    response = adapter._generate_with_gemini("Test prompt")

    assert response == '{"answerable": true}'
    assert fake_models.calls == 3
    assert delays == [2, 4]