"""Provider abstraction: mock always, real providers only when configured."""

from __future__ import annotations

import json
import os

import pytest

from medical_ai_tutor.config import load_config
from medical_ai_tutor.llm.base import LLMError, LLMMessage, extract_json
from medical_ai_tutor.llm.factory import build_client, build_client_or_mock, has_api_key
from medical_ai_tutor.llm.mock_client import MockLLMClient
from medical_ai_tutor.llm.prompt_loader import KNOWN_PROMPTS, load_prompt
from medical_ai_tutor.state.models import LearnerDiagnosis, PedagogicalDecision


def test_every_prompt_template_exists_and_states_its_role():
    for name in KNOWN_PROMPTS:
        text = load_prompt(name)
        assert text.startswith(f"ROLE: {name}")
        assert "## Do NOT" in text  # every prompt states its boundaries


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert extract_json('{"nested": {"b": [1,2]}, "s": "}"}')["s"] == "}"
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_mock_is_deterministic():
    context = json.dumps(
        {"student_message": "What is interoperability?", "learner_state": {}, "exercise": {}}
    )
    results = []
    for _ in range(3):
        diagnosis, _ = MockLLMClient().structured(
            "ROLE: diagnosis", [LLMMessage("user", context)], LearnerDiagnosis, "m"
        )
        results.append(diagnosis.model_dump_json())
    assert len(set(results)) == 1


def test_mock_script_can_force_a_structured_output():
    client = MockLLMClient(
        script=[{"PedagogicalDecision": {"action": "QUIZ", "reason_code": "scripted"}}]
    )
    decision, _ = client.structured(
        "ROLE: pedagogical_decision",
        [LLMMessage("user", "{}")],
        PedagogicalDecision,
        "m",
    )
    assert decision.action.value == "QUIZ"
    assert decision.reason_code == "scripted"


def test_structured_output_repairs_once_then_gives_up():
    """The base client makes exactly one repair attempt, then fails loudly."""

    class AlwaysInvalid(MockLLMClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def complete(self, system, messages, model, temperature=0.2, max_tokens=1024):
            from medical_ai_tutor.llm.base import LLMResult

            self.attempts += 1
            return LLMResult(text='{"action": "NOT_A_REAL_ACTION"}', model=model)

    client = AlwaysInvalid()
    with pytest.raises(LLMError):
        # Bypass the mock's schema dispatch to exercise the base implementation.
        super(MockLLMClient, client).structured(
            "ROLE: pedagogical_decision",
            [LLMMessage("user", "{}")],
            PedagogicalDecision,
            "m",
        )
    assert client.attempts == 2  # original + exactly one repair


def test_factory_returns_the_mock_by_default():
    assert build_client(load_config(apply_env=False)).name == "mock"


def test_factory_falls_back_to_mock_without_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config(apply_env=False).with_overrides(llm__provider="anthropic")
    client, warning = build_client_or_mock(config)
    assert client.name == "mock"
    assert warning and "ANTHROPIC_API_KEY" in warning


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMError):
        build_client(load_config(apply_env=False).with_overrides(llm__provider="hal9000"))


# --------------------------------------------------------------------------- #
# Real-provider smoke test: opt-in, skipped by default
# --------------------------------------------------------------------------- #


@pytest.mark.live
@pytest.mark.skipif(
    not has_api_key("anthropic"),
    reason="no Anthropic key in local_settings.py or the environment",
)
def test_anthropic_structured_output_smoke():
    from medical_ai_tutor.llm.anthropic_client import AnthropicClient

    client = AnthropicClient()
    context = json.dumps(
        {
            "student_message": "I think interoperability just means sending a file.",
            "learner_state": {},
            "exercise": {},
            "known_concepts": ["interoperability"],
        }
    )
    diagnosis, result = client.structured(
        system=load_prompt("diagnosis"),
        messages=[LLMMessage("user", context)],
        schema=LearnerDiagnosis,
        model=load_config().get("llm.models.controller"),
        max_tokens=800,
    )
    assert isinstance(diagnosis, LearnerDiagnosis)
    assert result.input_tokens > 0


@pytest.mark.live
@pytest.mark.skipif(
    not has_api_key("openai"),
    reason="no OpenAI key in local_settings.py or the environment",
)
def test_openai_structured_output_smoke():
    from medical_ai_tutor.llm.openai_client import OpenAIClient

    client = OpenAIClient()
    diagnosis, _ = client.structured(
        system=load_prompt("diagnosis"),
        messages=[LLMMessage("user", json.dumps({"student_message": "what is an EHR?"}))],
        schema=LearnerDiagnosis,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        max_tokens=800,
    )
    assert isinstance(diagnosis, LearnerDiagnosis)


def test_provider_and_model_family_are_reconciled():
    """Selecting openai while config.yaml names a Claude model must not 404."""
    from medical_ai_tutor.config import load_config

    config = load_config(apply_env=False).with_overrides(llm__provider="openai")
    assert not config.get("llm.models.controller").startswith("claude")
    assert config.notes, "the substitution should be reported to the user"

    anthropic = load_config(apply_env=False).with_overrides(
        llm__provider="anthropic", llm__models__controller="gpt-4o"
    )
    assert anthropic.get("llm.models.controller").startswith("claude")


def test_matching_provider_and_model_are_left_alone():
    from medical_ai_tutor.config import load_config

    config = load_config(apply_env=False).with_overrides(
        llm__provider="openai", llm__models__controller="gpt-4o"
    )
    assert config.get("llm.models.controller") == "gpt-4o"


def test_a_broken_local_settings_file_does_not_crash_the_app(tmp_path, monkeypatch):
    """An unquoted key is the commonest typo; it must degrade, not explode."""
    from medical_ai_tutor import config as config_module

    class Broken:
        def __init__(self):
            raise NameError("name 'sk' is not defined")

    def explode(name):
        raise NameError("name 'sk' is not defined")

    # The autouse fixture stubs out `local_settings` to keep runs offline; this
    # test is specifically about the real loader's error handling.
    monkeypatch.setattr(config_module, "local_settings", config_module._local_settings_impl)
    config_module.clear_caches()
    monkeypatch.setattr(config_module.importlib, "import_module", explode)
    try:
        assert config_module.local_settings() == {}
        assert config_module.LOCAL_SETTINGS_ERRORS
        assert "quoted" in config_module.LOCAL_SETTINGS_ERRORS[0]
    finally:
        config_module.clear_caches()
