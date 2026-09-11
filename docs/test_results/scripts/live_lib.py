"""Shared helpers for the live report runs."""
from __future__ import annotations

import sys, uuid, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # <repo>/docs/test_results/scripts -> <repo>
sys.path.insert(0, str(ROOT))

from medical_ai_tutor.config import load_config
from medical_ai_tutor.llm.factory import build_client, configured_provider
from medical_ai_tutor.tutor.orchestrator import build_orchestrator


def make_tutor(tag: str):
    config = load_config()
    provider = configured_provider(config)
    db = Path(tempfile.mkdtemp(prefix=f"live-{tag}-")) / "live.sqlite3"
    config = config.with_overrides(storage__db_path=str(db))
    tutor = build_orchestrator(config, client=build_client(config, provider))
    print(f"[live] provider={tutor.client.name} "
          f"controller={config.get('llm.models.controller')} "
          f"generator={config.get('llm.models.generator')} "
          f"judge={config.get('llm.models.judge')} "
          f"corpus={len(tutor.retriever.chunks)} chunks")
    return tutor


def session() -> str:
    return f"live-{uuid.uuid4().hex[:8]}"


def show(label, message, response, trace):
    print(f"\n  -- {label} " + "-" * 40)
    print(f"  STUDENT   : {message}")
    print(f"  intent    : {trace.diagnosis.intent.value} / {trace.diagnosis.response_quality.value}"
          f" (conf {trace.diagnosis.confidence})")
    print(f"  action    : {response.action.value}  ({trace.decision.reason_code})")
    print(f"  retrieval : used={bool(response.retrieved_section_ids)} "
          f"sections={response.retrieved_section_ids or []}")
    print(f"  citations : {response.citations or []}")
    print(f"  safety    : {response.safety_verdict.value} | revisions={response.revision_count} "
          f"| fallback={response.used_fallback}")
    print(f"  TUTOR     : {response.text}")
    print(f"  cost      : {trace.metrics.llm_calls} LLM calls, "
          f"{trace.metrics.input_tokens}->{trace.metrics.output_tokens} tokens, "
          f"{trace.metrics.total_latency_ms:.0f} ms")
