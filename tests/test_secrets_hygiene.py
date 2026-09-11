"""Guards on writing an API key directly in the code.

Putting a real key in `local_settings.py` is supported and convenient. It is
only safe while that file stays out of version control, so that property is
asserted here rather than left to a README note.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from medical_ai_tutor.config import SECRET_NAMES, get_secret

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = PROJECT_ROOT / "medical_ai_tutor" / "local_settings.py"

# Key-shaped literals: a provider prefix followed by real key material. The
# template's "sk-ant-..." placeholder is far too short to match.
KEY_PATTERN = re.compile(r"\b(sk-ant-|sk-proj-|sk-or-|sk-)[A-Za-z0-9_\-]{20,}")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _in_git_repo() -> bool:
    return _git("rev-parse", "--is-inside-work-tree").returncode == 0


def test_local_settings_is_listed_in_gitignore():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "local_settings.py" in gitignore


@pytest.mark.skipif(not _in_git_repo(), reason="not a git checkout")
def test_git_actually_ignores_local_settings():
    """Not just listed — verified against git's own ignore resolution."""
    if not LOCAL_SETTINGS.exists():
        pytest.skip("local_settings.py not present")
    result = _git("check-ignore", "-q", str(LOCAL_SETTINGS))
    assert result.returncode == 0, "git does not ignore local_settings.py"


@pytest.mark.skipif(not _in_git_repo(), reason="not a git checkout")
def test_local_settings_is_not_tracked():
    tracked = _git("ls-files", "medical_ai_tutor/local_settings.py").stdout.strip()
    assert not tracked, "local_settings.py is tracked by git — a key written there would be committed"


@pytest.mark.skipif(not _in_git_repo(), reason="not a git checkout")
def test_no_tracked_file_contains_a_key_literal():
    """Scan everything git tracks for key-shaped strings."""
    listing = _git("ls-files")
    if listing.returncode != 0:
        pytest.skip("git ls-files unavailable")

    offenders: list[str] = []
    for name in listing.stdout.splitlines():
        path = PROJECT_ROOT / name
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if KEY_PATTERN.search(text):
            offenders.append(name)

    assert not offenders, f"API-key-shaped literals found in tracked files: {offenders}"


def test_example_template_carries_no_real_key():
    example = (PROJECT_ROOT / "medical_ai_tutor" / "local_settings.py").read_text(
        encoding="utf-8"
    )
    assert not KEY_PATTERN.search(example)
    for name in ("LLM_PROVIDER", *SECRET_NAMES):
        assert name in example, f"{name} is missing from the template"


def test_secret_resolution_prefers_local_settings_then_environment(monkeypatch):
    from medical_ai_tutor import config as config_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environment")
    config_module.local_settings.cache_clear()
    monkeypatch.setattr(
        config_module, "local_settings", lambda: {"ANTHROPIC_API_KEY": "from-code"}
    )
    assert get_secret("ANTHROPIC_API_KEY") == "from-code"

    monkeypatch.setattr(config_module, "local_settings", lambda: {})
    assert get_secret("ANTHROPIC_API_KEY") == "from-environment"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_secret("ANTHROPIC_API_KEY") is None
    assert get_secret("ANTHROPIC_API_KEY", "fallback") == "fallback"


def test_placeholder_values_are_treated_as_unset(tmp_path, monkeypatch):
    """An untouched template must behave exactly like a missing file."""
    from medical_ai_tutor import config as config_module

    module = type("M", (), {"ANTHROPIC_API_KEY": "sk-ant-...", "LLM_PROVIDER": "  "})
    monkeypatch.setattr(
        config_module.importlib, "import_module", lambda name: module
    )
    config_module.local_settings.cache_clear()
    try:
        assert config_module.local_settings() == {}
    finally:
        config_module.local_settings.cache_clear()
