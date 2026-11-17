import os
import re
import time
import logging
from pathlib import Path
from collections.abc import Callable

import httpx
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
)

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured."""


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider fails."""


def _load_project_environment() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"

    if env_file.is_file():
        load_dotenv(env_file, override=False)


_load_project_environment()


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MAX_ATTEMPTS = 2
GEMINI_MAX_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 2

OPENAI_TIMEOUT_SECONDS = 45.0
GEMINI_TIMEOUT_MS = 45_000


# --------------------------------------------------
# OPENAI GENERATION
# --------------------------------------------------

def _generate_with_openai(prompt: str) -> str:
    """Generate a JSON response using OpenAI Responses API."""

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not configured."
        )

    if not model:
        raise LLMConfigurationError(
            "OPENAI_MODEL is not configured."
        )

    try:
        client = OpenAI(
            api_key=api_key,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        )

    except Exception as exc:
        raise LLMProviderError(
            "Could not initialize the OpenAI client."
        ) from exc

    request_prompt = (
        prompt
        + "\n\nReturn your response as valid JSON only. "
        "Follow the JSON structure requested in the prompt."
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            response = client.responses.create(
                model=model,
                input=request_prompt,
                text={
                    "format": {
                        "type": "json_object"
                    }
                },
            )

            response_text = response.output_text

            if (
                not isinstance(response_text, str)
                or not response_text.strip()
            ):
                raise LLMProviderError(
                    "OpenAI returned an empty or invalid response."
                )

            return response_text.strip()

        except RateLimitError as exc:
            logger.error("OpenAI rate limit reached: %s", exc)

            raise LLMProviderError(
                f"OpenAI rate limit reached: {exc}"
            ) from exc

        except Exception as exc:

            retryable = (
                isinstance(
                    exc,
                    (
                        APIConnectionError,
                        APITimeoutError,
                    ),
                )
                or (
                    isinstance(exc, APIStatusError)
                    and exc.status_code in (500, 502, 503, 504)
                )
            )

            if attempt < MAX_ATTEMPTS and retryable:

                delay = INITIAL_RETRY_DELAY_SECONDS * (
                    2 ** (attempt - 1)
                )

                logger.warning(
                    "Temporary OpenAI failure on attempt %s/%s. "
                    "Retrying in %s seconds. Error: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                    exc,
                )

                time.sleep(delay)
                continue

            raise LLMProviderError(
                f"OpenAI failed after {attempt} attempt(s): {exc}"
            ) from exc

    raise LLMProviderError(
        "OpenAI failed after exhausting retry attempts."
    )


# --------------------------------------------------
# --------------------------------------------------
# GROQ GENERATION
# --------------------------------------------------

def _generate_with_groq(prompt: str) -> str:
    """Generate a JSON response using Groq's OpenAI-compatible API."""

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not api_key:
        raise LLMConfigurationError("GROQ_API_KEY is not configured.")

    if not model:
        raise LLMConfigurationError("GROQ_MODEL is not configured.")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        )
    except Exception as exc:
        raise LLMProviderError("Could not initialize the Groq client.") from exc

    request_prompt = (
        prompt
        + "\n\nReturn your response as valid JSON only. "
        "Follow the JSON structure requested in the prompt."
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request_prompt}],
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content

            if not isinstance(response_text, str) or not response_text.strip():
                raise LLMProviderError("Groq returned an empty or invalid response.")

            return response_text.strip()

        except RateLimitError as exc:
            error_text = str(exc).lower()
            headers = getattr(getattr(exc, "response", None), "headers", {})
            reset_header = headers.get("x-ratelimit-reset-tokens", "")
            retry_header = headers.get("retry-after", "")

            # Retry temporary token-per-minute limits only.
            is_tpm_limit = (
                "tokens per minute" in error_text
                or bool(reset_header)
            )

            if attempt < MAX_ATTEMPTS and is_tpm_limit:
                match = re.search(
                    r"try again in ([0-9.]+)s",
                    str(exc),
                    re.IGNORECASE,
                )

                delay = (
                    float(match.group(1)) if match
                    else float(retry_header) if retry_header
                    else 3.0
                )

                delay = max(0.0, delay) + 0.25

                logger.warning(
                    "Groq TPM limit reached. Retrying in %.2f seconds.",
                    delay,
                )
                time.sleep(delay)
                continue

            logger.error("Groq rate limit reached: %s", exc)
            raise LLMProviderError(
                f"Groq rate limit reached: {exc}"
            ) from exc

        except Exception as exc:
            retryable = (
                isinstance(exc, (APIConnectionError, APITimeoutError))
                or (
                    isinstance(exc, APIStatusError)
                    and exc.status_code in (500, 502, 503, 504)
                )
            )

            if attempt < MAX_ATTEMPTS and retryable:
                delay = INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Temporary Groq failure on attempt %s/%s. Retrying in %s seconds.",
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue

            raise LLMProviderError(
                f"Groq failed after {attempt} attempt(s): {exc}"
            ) from exc

    raise LLMProviderError("Groq failed after exhausting retry attempts.")


# GEMINI GENERATION (FALLBACK)
# --------------------------------------------------

def _generate_with_gemini(prompt: str) -> str:
    """Generate a JSON-formatted response using Gemini."""

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is not configured."
        )

    if not model:
        raise LLMConfigurationError(
            "GEMINI_MODEL is not configured."
        )

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=GEMINI_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=1
                ),
            ),
        )

    except Exception as exc:
        raise LLMProviderError(
            "Could not initialize the Gemini client."
        ) from exc

    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            response_text = response.text

            if (
                not isinstance(response_text, str)
                or not response_text.strip()
            ):
                raise LLMProviderError(
                    "Gemini returned an empty or invalid response."
                )

            return response_text.strip()

        except Exception as exc:

            status_code = getattr(exc, "code", None)

            if status_code is None:
                status_code = getattr(exc, "status_code", None)

            status = getattr(exc, "status", None)

            if (
                status_code == 429
                or (
                    isinstance(status, str)
                    and status.upper() == "RESOURCE_EXHAUSTED"
                )
            ):
                raise LLMProviderError(
                    f"Gemini rate limit or quota exhaustion: {exc}"
                ) from exc

            retryable = (
                isinstance(exc, httpx.TimeoutException)
                or status_code in (500, 502, 503, 504)
                or (
                    isinstance(status, str)
                    and status.upper() in {
                        "INTERNAL",
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    }
                )
            )

            if attempt < GEMINI_MAX_ATTEMPTS and retryable:

                delay = INITIAL_RETRY_DELAY_SECONDS * (
                    2 ** (attempt - 1)
                )

                logger.warning(
                    "Temporary Gemini failure. Retrying in %s seconds.",
                    delay,
                )

                time.sleep(delay)
                continue

            raise LLMProviderError(
                f"Gemini failed after {attempt} attempt(s): {exc}"
            ) from exc

    raise LLMProviderError(
        "Gemini failed after exhausting retry attempts."
    )


# --------------------------------------------------

# --------------------------------------------------
# LOCAL QWEN GENERATION
# --------------------------------------------------

def _generate_with_local_qwen(prompt: str) -> str:
    """Generate a response using the preloaded local Qwen model."""

    import torch

    local_model = globals().get("LOCAL_QWEN_MODEL")
    local_tokenizer = globals().get("LOCAL_QWEN_TOKENIZER")

    if local_model is None or local_tokenizer is None:
        raise LLMConfigurationError(
            "Local Qwen model or tokenizer is not loaded."
        )

    messages = [{"role": "user", "content": prompt}]

    encoded_inputs = local_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    device = next(local_model.parameters()).device

    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = local_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=local_tokenizer.eos_token_id,
        )

    response = local_tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    if not response.strip():
        raise LLMProviderError(
            "Local Qwen returned an empty response."
        )

    return response.strip()


# PUBLIC RESPONSE GENERATOR
# --------------------------------------------------

def generate_response(
    prompt: str,
    *,
    provider_function: Callable[[str], str] | None = None,
) -> str:
    """Generate a response using the configured LLM provider."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "prompt must be a non-empty string"
        )

    # Allow injected providers for testing.
    if provider_function is not None:

        try:
            response = provider_function(prompt)

        except Exception as exc:
            raise LLMProviderError(
                "The configured LLM provider failed."
            ) from exc

        if (
            not isinstance(response, str)
            or not response.strip()
        ):
            raise LLMProviderError(
                "The LLM provider returned an empty or invalid response."
            )

        return response.strip()

    provider = os.getenv("SENTINELAI_LLM_PROVIDER")

    if not provider:
        raise LLMConfigurationError(
            "LLM provider is not configured."
        )

    provider = provider.lower().strip()

    if provider == "openai":
        return _generate_with_openai(prompt)

    if provider == "groq":
        return _generate_with_groq(prompt)

    if provider == "gemini":
        return _generate_with_gemini(prompt)

    if provider == "local":
        return _generate_with_local_qwen(prompt)

    raise LLMConfigurationError(
        f"LLM provider '{provider}' has not been implemented."
    )