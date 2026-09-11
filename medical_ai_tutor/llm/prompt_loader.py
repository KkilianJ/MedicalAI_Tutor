"""Loads the prompt templates in `llm/prompts/`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

KNOWN_PROMPTS = (
    "diagnosis",
    "pedagogical_decision",
    "tutor_generation",
    "safety_judge",
    "revision",
)


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
