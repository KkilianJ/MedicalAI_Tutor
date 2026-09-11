"""Configuration loading.

Configuration comes from ``config.yaml`` with environment-variable overrides.
Loading is deliberately explicit and side-effect free so tests can construct a
configuration without touching the developer's environment.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Environment variable -> dotted config path.
ENV_OVERRIDES: dict[str, str] = {
    "TUTOR_LLM_PROVIDER": "llm.provider",
    "TUTOR_MODEL_CONTROLLER": "llm.models.controller",
    "TUTOR_MODEL_GENERATOR": "llm.models.generator",
    "TUTOR_MODEL_JUDGE": "llm.models.judge",
    "TUTOR_RETRIEVAL_TOP_K": "retrieval.top_k",
    "TUTOR_USE_EMBEDDINGS": "retrieval.use_embeddings",
    "TUTOR_MAX_TOOL_CALLS": "limits.max_tool_calls_per_turn",
    "TUTOR_MAX_REVISIONS": "limits.max_revisions_per_turn",
    "TUTOR_DB_PATH": "storage.db_path",
    "TUTOR_SEARCHABLE_DIR": "storage.searchable_dir",
    "TUTOR_EXERCISES_DIR": "storage.exercises_dir",
    "TUTOR_PROTECTED_DIR": "storage.protected_dir",
    "TUTOR_DEBUG_TRACING": "tracing.enabled",
}

# Settings that may be written directly in `local_settings.py`, mapped to the
# config path they override. Secrets (API keys) are deliberately NOT config
# paths: they are never written into the config tree, never serialised, and are
# read only through `get_secret()`.
LOCAL_CONFIG_OVERRIDES: dict[str, str] = {
    "LLM_PROVIDER": "llm.provider",
    "MODEL_CONTROLLER": "llm.models.controller",
    "MODEL_GENERATOR": "llm.models.generator",
    "MODEL_JUDGE": "llm.models.judge",
}

SECRET_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")

LOCAL_SETTINGS_MODULE = "medical_ai_tutor.local_settings"
# Populated when the file exists but cannot be imported.
LOCAL_SETTINGS_ERRORS: list[str] = []
# Placeholder values from the template; treated as "not configured".
_PLACEHOLDERS = {"sk-ant-...", "sk-...", "your-key-here", "..."}


@lru_cache(maxsize=1)
def local_settings() -> dict[str, str]:
    """Read `local_settings.py`, if it exists.

    Missing file, empty values and template placeholders all mean "not set", so
    a checkout without the file behaves exactly like one with an untouched copy.
    """
    try:
        module = importlib.import_module(LOCAL_SETTINGS_MODULE)
    except ModuleNotFoundError:
        return {}
    except Exception as exc:
        # A typo in local_settings.py must not take the whole app down with an
        # obscure traceback. The most common one by far is a value written
        # without quotes, which Python reads as an undefined name.
        hint = ""
        if isinstance(exc, (NameError, SyntaxError)):
            hint = (
                "  Values must be quoted strings, e.g.\n"
                '      OPENAI_API_KEY = "sk-..."\n'
                "  not\n"
                "      OPENAI_API_KEY = sk-..."
            )
        LOCAL_SETTINGS_ERRORS.append(
            f"local_settings.py could not be loaded ({type(exc).__name__}: {exc}).\n{hint}"
        )
        return {}

    found: dict[str, str] = {}
    for key, value in vars(module).items():
        if not key.isupper() or not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in _PLACEHOLDERS:
            found[key] = cleaned
    return found


# The undecorated-by-tests reference. Test fixtures replace the module-level
# `local_settings` to keep runs offline; anything that needs the real loader
# (including the test for its own error handling) reaches it through this.
_local_settings_impl = local_settings


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    """Populate os.environ from a `.env` file, without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Resolve a secret: local_settings.py, then the environment, then .env.

    This is the only way any component obtains an API key, so a key written in
    `local_settings.py` works everywhere the environment variable would.
    """
    from_local = local_settings().get(name)
    if from_local:
        return from_local

    _load_dotenv()
    from_env = os.environ.get(name)
    if from_env and from_env.strip():
        return from_env.strip()
    return default


def clear_caches() -> None:
    """Drop cached settings. Used by tests that change the environment."""
    LOCAL_SETTINGS_ERRORS.clear()
    local_settings.cache_clear()
    _load_dotenv.cache_clear()
    get_config.cache_clear()


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(raw: str, current: Any) -> Any:
    """Coerce an environment string to the type of the value it replaces."""
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"cannot interpret {raw!r} as a boolean")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        node = node[part]
    return node


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


# Model families, so a provider/model mismatch is caught at load time instead of
# surfacing as a 404 on every LLM call at runtime.
MODEL_FAMILY_PREFIXES = {
    "anthropic": ("claude",),
    "openai": ("gpt", "o1", "o3", "o4", "chatgpt"),
}
PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
}
MODEL_PATHS = ("llm.models.controller", "llm.models.generator", "llm.models.judge")


def _model_family(model: str) -> str | None:
    lowered = (model or "").lower()
    for family, prefixes in MODEL_FAMILY_PREFIXES.items():
        if lowered.startswith(prefixes):
            return family
    return None


def reconcile_models(data: dict[str, Any]) -> list[str]:
    """Swap in the provider's default model when the configured one is another
    provider's.

    Without this, selecting `openai` while `config.yaml` still names a Claude
    model produces a 404 on every call, and the runtime's own fallbacks then
    make the tutor look like it is merely being unhelpful.
    """
    provider = str(_get_path(data, "llm.provider") or "mock").lower()
    if provider not in PROVIDER_DEFAULT_MODELS:
        return []

    notes: list[str] = []
    for dotted in MODEL_PATHS:
        try:
            current = _get_path(data, dotted)
        except (KeyError, TypeError):
            continue
        family = _model_family(str(current))
        if family is not None and family != provider:
            replacement = PROVIDER_DEFAULT_MODELS[provider]
            _set_path(data, dotted, replacement)
            notes.append(
                f"{dotted}: {current!r} is a {family} model but the provider is "
                f"{provider!r}; using {replacement!r} instead"
            )
    return notes


class Config:
    """Read-only view over the configuration tree."""

    def __init__(
        self,
        data: dict[str, Any],
        project_root: Path | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self._data = data
        self.project_root = project_root or PROJECT_ROOT
        # Non-fatal configuration problems worth showing the user.
        self.notes: list[str] = list(notes or [])

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return _get_path(self._data, dotted)
        except (KeyError, TypeError):
            return default

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return dict(value) if isinstance(value, dict) else {}

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        raw = self.get(dotted)
        if raw is None:
            raise KeyError(f"no path configured at {dotted!r}")
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def as_dict(self) -> dict[str, Any]:
        return self._data

    def with_overrides(self, **dotted_values: Any) -> Config:
        """Return a copy with specific values replaced (used heavily by tests)."""
        import copy

        data = copy.deepcopy(self._data)
        for dotted, value in dotted_values.items():
            _set_path(data, dotted.replace("__", "."), value)
        notes = reconcile_models(data)
        return Config(data, self.project_root, notes)


def load_config(path: str | Path | None = None, apply_env: bool = True) -> Config:
    """Load configuration from YAML, then apply environment overrides."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if apply_env:
        _load_dotenv()
        for env_key, dotted in ENV_OVERRIDES.items():
            raw = os.environ.get(env_key)
            if raw is None or raw == "":
                continue
            try:
                current = _get_path(data, dotted)
            except (KeyError, TypeError):
                current = None
            _set_path(data, dotted, _coerce(raw, current))

        # `local_settings.py` is applied last: a value written in code is the
        # most explicit statement of intent, so it wins over the environment.
        for local_key, dotted in LOCAL_CONFIG_OVERRIDES.items():
            raw = local_settings().get(local_key)
            if not raw:
                continue
            try:
                current = _get_path(data, dotted)
            except (KeyError, TypeError):
                current = None
            _set_path(data, dotted, _coerce(raw, current))

    notes = reconcile_models(data) + list(LOCAL_SETTINGS_ERRORS)
    return Config(data, config_path.resolve().parent, notes)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide configuration singleton."""
    return load_config()
