"""Layer A: deterministic leakage detection.

Pure, reproducible string analysis. It cannot be argued with by a prompt and it
runs before any model is asked for an opinion. It returns structured evidence —
which units were covered, how long the longest verbatim run was — not a bare
boolean, so the semantic judge and the reviser both have something to work with.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..state.models import DeterministicFinding

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", "→": " ",
}


def normalize(text: str) -> str:
    """Unicode, case, quote and punctuation normalisation.

    Defeats the cheap evasions: smart quotes, accents, odd dashes, casing, and
    inserted punctuation.
    """
    if not text:
        return ""
    for source, target in _QUOTES.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []


def ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def longest_common_run(a: list[str], b: list[str]) -> tuple[int, list[str]]:
    """Longest run of identical consecutive tokens, plus the run itself."""
    if not a or not b:
        return 0, []
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(b):
        positions.setdefault(token, []).append(index)

    best, best_span = 0, []
    for i in range(len(a)):
        for j in positions.get(a[i], ()):
            run = 0
            while i + run < len(a) and j + run < len(b) and a[i + run] == b[j + run]:
                run += 1
            if run > best:
                best, best_span = run, a[i : i + run]
    return best, best_span


class DeterministicDetector:
    """Configurable lexical leakage detector."""

    def __init__(
        self,
        ngram_size: int = 5,
        ngram_overlap_threshold: float = 0.16,
        phrase_min_words: int = 7,
        answer_unit_overlap_threshold: float = 0.5,
        max_verbatim_run: int = 12,
        single_unit_cover_threshold: float = 0.72,
    ) -> None:
        self.ngram_size = ngram_size
        self.ngram_overlap_threshold = ngram_overlap_threshold
        self.phrase_min_words = phrase_min_words
        self.answer_unit_overlap_threshold = answer_unit_overlap_threshold
        self.max_verbatim_run = max_verbatim_run
        self.single_unit_cover_threshold = single_unit_cover_threshold

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeterministicDetector:
        return cls(
            ngram_size=int(config.get("ngram_size", 5)),
            ngram_overlap_threshold=float(config.get("ngram_overlap_threshold", 0.16)),
            phrase_min_words=int(config.get("phrase_min_words", 7)),
            answer_unit_overlap_threshold=float(
                config.get("answer_unit_overlap_threshold", 0.5)
            ),
            max_verbatim_run=int(config.get("max_verbatim_run", 12)),
            single_unit_cover_threshold=float(config.get("single_unit_cover_threshold", 0.72)),
        )

    def check(
        self,
        candidate: str,
        solution_text: str,
        answer_units: list[str] | None = None,
        unit_labels: list[str] | None = None,
    ) -> DeterministicFinding:
        """Compare a candidate response against a protected solution."""
        candidate_tokens = normalized_tokens(candidate)
        solution_tokens = normalized_tokens(solution_text)
        answer_units = answer_units or []
        unit_labels = unit_labels or [f"unit_{i + 1}" for i in range(len(answer_units))]

        finding = DeterministicFinding()
        if not candidate_tokens or not solution_tokens:
            return finding

        # 1. n-gram overlap: how much of the solution appears in the candidate.
        solution_grams = ngrams(solution_tokens, self.ngram_size)
        candidate_grams = ngrams(candidate_tokens, self.ngram_size)
        shared = solution_grams & candidate_grams
        finding.ngram_overlap = round(len(shared) / max(1, len(solution_grams)), 4)

        # 2. Longest verbatim run.
        run_length, run_tokens = longest_common_run(candidate_tokens, solution_tokens)
        finding.max_verbatim_run = run_length
        if run_length >= self.phrase_min_words:
            finding.matched_phrases.append(" ".join(run_tokens[:20]))

        # 3. Per-answer-unit coverage: the measure that actually matters.
        covered: list[str] = []
        for label, unit in zip(unit_labels, answer_units, strict=False):
            unit_tokens = normalized_tokens(unit)
            if len(unit_tokens) < 3:
                continue
            unit_set = set(unit_tokens)
            candidate_set = set(candidate_tokens)
            lexical_cover = len(unit_set & candidate_set) / len(unit_set)
            unit_run, _ = longest_common_run(unit_tokens, candidate_tokens)
            phrase_cover = unit_run / len(unit_tokens)
            if (
                lexical_cover >= self.single_unit_cover_threshold
                or phrase_cover >= self.answer_unit_overlap_threshold
            ):
                covered.append(label)
        finding.covered_units = covered
        finding.unit_coverage = round(len(covered) / max(1, len(answer_units)), 4)

        # 4. Combine into a verdict with explicit reason codes.
        reasons: list[str] = []
        if finding.ngram_overlap >= self.ngram_overlap_threshold:
            reasons.append("ngram_overlap_above_threshold")
        if finding.max_verbatim_run >= self.max_verbatim_run:
            reasons.append("verbatim_run_above_threshold")
        if covered:
            reasons.append("answer_units_covered")
        if finding.unit_coverage >= self.answer_unit_overlap_threshold:
            reasons.append("unit_coverage_above_threshold")

        finding.reason_codes = reasons
        finding.leaked = bool(reasons)
        return finding
