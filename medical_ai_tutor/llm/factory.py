"""Provider selection.

API keys are resolved through `config.get_secret()`, which reads
`local_settings.py` first and the environment second. No client reads
`os.environ` on its own, so writing a key in code works everywhere.
"""

from __future__ import annotations

from ..config import Config, get_secret
from .base import LLMClient, LLMError
from .mock_client import MockLLMClient

# Which secret each provider needs, for error messages and availability checks.
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _missing_key_message(provider: str, key_name: str) -> str:
    return (
        f"provider {provider!r} selected but {key_name} is not set.\n"
        f"Set it in medical_ai_tutor/local_settings.py:\n"
        f'    LLM_PROVIDER = "{provider}"\n'
        f'    {key_name} = "..."\n'
        f"or export {key_name} in your environment."
    )


def has_api_key(provider: str) -> bool:
    """Whether a provider is usable. Used by pytest to skip live tests."""
    key_name = PROVIDER_KEYS.get(provider.lower())
    return bool(key_name and get_secret(key_name))


def configured_provider(config: Config) -> str:
    return str(config.get("llm.provider", "mock") or "mock").lower()


def build_client(config: Config, provider: str | None = None) -> LLMClient:
    """Build the configured provider.

    Raises `LLMError` if the provider is selected but unusable; callers that
    prefer a graceful degrade use `build_client_or_mock`.
    """
    name = (provider or configured_provider(config)).lower()
    timeout = float(config.get("llm.request_timeout_s", 60))

    if name == "mock":
        return MockLLMClient()

    if name == "anthropic":
        from .anthropic_client import AnthropicClient

        key = get_secret("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError(_missing_key_message("anthropic", "ANTHROPIC_API_KEY"))
        return AnthropicClient(api_key=key, timeout=timeout)

    if name == "openai":
        from .openai_client import OpenAIClient

        key = get_secret("OPENAI_API_KEY")
        if not key:
            raise LLMError(_missing_key_message("openai", "OPENAI_API_KEY"))
        return OpenAIClient(
            api_key=key, base_url=get_secret("OPENAI_BASE_URL"), timeout=timeout
        )

    raise LLMError(f"unknown provider {name!r}; expected one of: mock, anthropic, openai")


def build_client_or_mock(
    config: Config, provider: str | None = None
) -> tuple[LLMClient, str | None]:
    """Best-effort provider construction. Returns (client, warning)."""
    try:
        return build_client(config, provider), None
    except Exception as exc:
        return MockLLMClient(), f"falling back to mock provider: {exc}"
