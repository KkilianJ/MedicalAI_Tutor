"""Shared test fixtures.

Everything here runs offline: the mini textbook is ingested into a temporary
directory, and the mock LLM provider stands in for a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_ai_tutor.config import load_config  # noqa: E402
from medical_ai_tutor.llm.mock_client import MockLLMClient  # noqa: E402
from medical_ai_tutor.retrieval.ingest import run_ingestion  # noqa: E402
from medical_ai_tutor.tutor.orchestrator import build_orchestrator  # noqa: E402
from tests.fixtures import MINI_BOOK_PAGES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def mini_pages() -> list[dict[str, Any]]:
    return MINI_BOOK_PAGES


@pytest.fixture(scope="session")
def ingested(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the real ingestion pipeline over the synthetic book."""
    root = tmp_path_factory.mktemp("data")
    searchable = root / "searchable"
    exercises = root / "exercises"
    protected = root / "protected"
    manifest = run_ingestion(
        pdf_path=None,
        searchable_dir=searchable,
        exercises_dir=exercises,
        protected_dir=protected,
        doc_id="minibook",
        pages=MINI_BOOK_PAGES,
        chunk_config={"target_tokens": 120, "overlap_tokens": 20, "min_tokens": 20},
    )
    return {
        "root": root,
        "searchable": searchable,
        "exercises": exercises,
        "protected": protected,
        "manifest": manifest,
    }


@pytest.fixture
def config(ingested: dict[str, Any], tmp_path: Path):
    """Configuration pointing at the ingested fixture data and a temp database."""
    base = load_config(PROJECT_ROOT / "config.yaml", apply_env=False)
    return base.with_overrides(
        storage__db_path=str(tmp_path / "test.sqlite3"),
        storage__searchable_dir=str(ingested["searchable"]),
        storage__exercises_dir=str(ingested["exercises"]),
        storage__protected_dir=str(ingested["protected"]),
        llm__provider="mock",
    )


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def orchestrator(config, mock_client: MockLLMClient):
    return build_orchestrator(config, client=mock_client)


@pytest.fixture(autouse=True)
def _no_developer_credentials(request, monkeypatch):
    """Keep the default test run offline and free.

    `local_settings.py` deliberately outranks the environment, which is right for
    a developer but wrong for a test suite: it would override the mock provider
    the tests select and spend real credit on every run. Tests marked `live`
    opt back in, because hitting a real provider is the point of those.
    """
    if request.node.get_closest_marker("live"):
        # This is a generator fixture: it must still yield exactly once, or
        # pytest reports "did not yield a value" and every live test errors.
        yield
        return

    from medical_ai_tutor import config as config_module

    def empty() -> dict[str, str]:
        return {}

    empty.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(config_module, "local_settings", empty)
    config_module.get_config.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield
    config_module.get_config.cache_clear()
