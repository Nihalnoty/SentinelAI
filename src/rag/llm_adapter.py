import os
from collections.abc import Callable


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured."""


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider fails."""


def generate_response(
    prompt: str,
    *,
    provider_function: Callable[[str], str] | None = None,
) -> str:
    """
    Generate a response using an injected provider function.

    The provider function should accept a prompt and return the model's
    response as a string.

    Provider-specific API code will be connected separately once the
    team confirms which provider and credentials are available.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if provider_function is None:
        provider = os.getenv("SENTINELAI_LLM_PROVIDER")

        if not provider:
            raise LLMConfigurationError(
                "LLM provider is not configured. Set "
                "SENTINELAI_LLM_PROVIDER after choosing a provider."
            )

        raise LLMConfigurationError(
            f"Provider '{provider}' has been selected, but its adapter "
            "has not been implemented yet."
        )

    try:
        response = provider_function(prompt)
    except Exception as exc:
        raise LLMProviderError(
            "The configured LLM provider failed."
        ) from exc

    if not isinstance(response, str) or not response.strip():
        raise LLMProviderError(
            "The LLM provider returned an empty or invalid response."
        )

    return response
