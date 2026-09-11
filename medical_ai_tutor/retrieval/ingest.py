"""Textbook ingestion.

Turns the open-access Winter et al. textbook PDF into three physically separate
artefacts:

    data/searchable/corpus.jsonl   textbook sections + glossary + exercise questions
    data/exercises/exercises.json  exercise questions and public metadata
    data/protected/solutions.json  official solutions  <- NEVER indexed

The separation is enforced here, at write time, and re-verified by
`verify_separation()` and by an automated test. No component other than the
safety subsystem is given a path into `data/protected/`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bm25 import tokenize
from .chunking import Chunk, chunk_section, split_sentences

# --------------------------------------------------------------------------- #
# Page-level cleaning
# --------------------------------------------------------------------------- #

_RUNNING_HEADER = re.compile(
    r"^(?:\d+\s+\d+\s+[A-Z].*|\d+(?:\.\d+)*\s+.*\s+\d+|\d+\s+[A-Z][A-Za-z].*|"
    r"(?:Solutions to Exercises|Glossary|Index|Contents)(?:\s+\d+)?|\d+)$"
)
# Springer prints a copyright/citation footer on the first page of each part.
# It is not content and must not end up inside a section, an exercise or a
# solution's answer units.
_FOOTER_MARKERS = (
    "\u00a9 The Editor(s)",
    "The Author(s) 20",
    "et al., Health Information Systems",
    "Health Informatics, https://doi.org",
    "https://doi.org/10.1007",
)
_LICENSE_MARKERS = (
    "Open Access This chapter is licensed",
    "Open Access This book is licensed",
    "Creative Commons license",
    "creativecommons.org/licenses",
    "the copyright holder",
    "statutory regulation or exceeds the permitted use",
    "included in the chapter's Creative Commons",
    "included in the book's Creative Commons",
)
_SECTION_HEADING = re.compile(r"^(\d+(?:\.\d+){1,2})\s+(\S.{1,90})$")
_SOLUTION_HEADING = re.compile(r"^Exercise\s+(\d+\.\d+\.\d+)\s*(.*)$")
_CHAPTER_HEADING = re.compile(r"^Chapter\s+(\d+)$")
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\s*\n\s*([a-z])")
# A hyphen followed by a space mid-word is a line-wrap artifact once lines are
# joined ("dif- ferent"). A hanging hyphen before a conjunction is not
# ("patient- and provider-facing"), so those are left alone.
_HYPHEN_JOIN = re.compile(r"([a-z])-\s+(?!and\b|or\b|to\b|based\b)([a-z])")

_CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "health_information_system": ("health information system", "hospital information system"),
    "electronic_health_record": ("electronic health record", "ehr", "patient record"),
    "interoperability": ("interoperability", "hl7", "fhir", "standard"),
    "integration": ("integration", "communication server", "interface"),
    "architectural_styles": ("architectural style", "architecture", "best of breed", "monolithic"),
    "data_information_knowledge": ("data, information", "knowledge"),
    "information_management": ("information management", "cio", "governance", "strategic"),
    "data_quality": ("data quality", "data integrity", "redundancy"),
    "3lgm2": ("3lgm", "domain layer", "logical tool layer", "physical tool layer"),
    "evaluation": ("evaluation", "study design", "questionnaire", "quality of"),
    "data_protection": ("data protection", "privacy", "data security", "consent"),
    "application_systems": ("application system", "application component", "cpoe", "pacs", "ris"),
    "stakeholders": ("stakeholder", "requirement", "life situation"),
    "health_care_settings": ("nursing home", "medical office", "ambulatory", "health care setting"),
}


def _is_gibberish(line: str) -> bool:
    """Detect mangled figure text (rotated labels, matrices) from PDF extraction."""
    words = line.split()
    if len(words) < 6:
        return False
    singles = sum(1 for w in words if len(w) <= 2)
    return singles / len(words) > 0.5



def extract_bold_phrases(page: Any) -> list[str]:
    """Contiguous runs of bold words on a page, in reading order.

    Used to recover glossary head-words, which the PDF sets in bold. Failures
    are non-fatal: callers fall back to a textual heuristic.
    """
    try:
        words = page.extract_words(extra_attrs=["fontname"])
    except Exception:
        return []

    lines: dict[int, list[dict[str, Any]]] = {}
    for word in words:
        lines.setdefault(round(float(word.get("top", 0.0)) / 3.0), []).append(word)

    phrases: list[str] = []
    for key in sorted(lines):
        row = sorted(lines[key], key=lambda w: float(w.get("x0", 0.0)))
        run: list[str] = []
        for word in row:
            if "bold" in str(word.get("fontname", "")).lower():
                run.append(word["text"])
            elif run:
                phrases.append(" ".join(run))
                run = []
        if run:
            phrases.append(" ".join(run))
    return [p for p in phrases if p.strip()]


def clean_page(text: str, drop_header: bool = True) -> str:
    """Strip running headers, page numbers, licence boilerplate and figure noise."""
    lines = (text or "").split("\n")
    out: list[str] = []
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if drop_header and index == 0 and _RUNNING_HEADER.match(line):
            continue
        if line.isdigit():
            continue
        if any(marker in line for marker in _LICENSE_MARKERS):
            continue
        if any(marker in line for marker in _FOOTER_MARKERS):
            continue
        if line.startswith(("http://", "https://", "\u00a9")):
            continue
        if line.startswith("Fig.") or line.startswith("Table "):
            continue
        if _is_gibberish(line):
            continue
        out.append(line)
    return "\n".join(out)


def join_wrapped(text: str) -> str:
    """Undo PDF line wrapping, including hyphenated line breaks."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _HYPHEN_JOIN.sub(r"\1\2", text)
    paragraphs = re.split(r"\n\s*\n", text)
    joined = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in joined if p)


# --------------------------------------------------------------------------- #
# Extracted structures
# --------------------------------------------------------------------------- #


@dataclass
class Section:
    section_id: str
    title: str
    page: int
    text: str = ""
    chapter: str = ""
    kind: str = "textbook"


@dataclass
class ExerciseRecord:
    exercise_id: str
    title: str
    question: str
    chapter: str
    page: int
    concepts: list[str] = field(default_factory=list)


@dataclass
class SolutionRecord:
    exercise_id: str
    title: str
    solution_text: str
    answer_units: list[str]
    page: int


@dataclass
class GlossaryTerm:
    term: str
    definition: str
    page: int


def detect_concepts(text: str, limit: int = 4) -> list[str]:
    lowered = (text or "").lower()
    found = [cid for cid, needles in _CONCEPT_LEXICON.items() if any(n in lowered for n in needles)]
    return found[:limit]


def split_answer_units(solution_text: str, max_units: int = 12) -> list[str]:
    """Split an official solution into the discrete claims it is made of.

    Answer units are what the safety layer protects: the things the exercise
    exists to make the learner produce.
    """
    text = solution_text.strip()
    if not text:
        return []

    units: list[str] = []
    # Lettered/numbered/bulleted parts first, since solutions are often (a)/(b).
    parts = re.split(r"(?m)^(?:\(?[a-h]\)|[•\-•]|\d+\.)\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        candidates = parts
    else:
        candidates = split_sentences(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate.split()) < 5:
            continue
        if any(marker in candidate for marker in _FOOTER_MARKERS):
            continue
        units.append(candidate)
        if len(units) >= max_units:
            break
    return units or [text]


# --------------------------------------------------------------------------- #
# The ingestor
# --------------------------------------------------------------------------- #


class TextbookIngestor:
    """Parses the textbook PDF into separated artefacts."""

    def __init__(self, doc_id: str = "winter2023_his") -> None:
        self.doc_id = doc_id
        self.sections: list[Section] = []
        self.exercises: list[ExerciseRecord] = []
        self.solutions: list[SolutionRecord] = []
        self.glossary: list[GlossaryTerm] = []
        self.warnings: list[str] = []

    # -- PDF loading ------------------------------------------------------ #
    @staticmethod
    def read_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
        try:
            import pdfplumber  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "pdfplumber is required to ingest the PDF: pip install 'medical-ai-tutor[ingest]'"
            ) from exc

        pages: list[dict[str, Any]] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                pages.append(
                    {
                        "page": number,
                        "text": page.extract_text() or "",
                        "bold_phrases": extract_bold_phrases(page),
                    }
                )
        return pages

    # -- region detection -------------------------------------------------- #
    @staticmethod
    def find_regions(pages: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        """Locate body / solutions / glossary / index by their opening pages.

        Detected rather than hard-coded so a different printing still ingests.
        """
        body_start = solutions_start = glossary_start = index_start = None
        for page in pages:
            text = (page["text"] or "").lstrip()
            number = page["page"]
            if body_start is None and text.startswith("Chapter 1"):
                body_start = number
            if solutions_start is None and text.startswith("Solutions to Exercises"):
                solutions_start = number
            # Guarding on `number > body_start` rather than a fixed page number
            # keeps the table of contents from matching while still working on a
            # short document (such as the test fixture).
            after_body = body_start is not None and number > body_start
            if glossary_start is None and after_body and re.match(r"^Glossary\d*\b", text):
                glossary_start = number
            if index_start is None and after_body and re.match(r"^Index\b", text):
                index_start = number

        total = len(pages)
        body_start = body_start or 1
        solutions_start = solutions_start or total
        glossary_start = glossary_start or solutions_start
        index_start = index_start or total + 1
        return {
            "body": (body_start, solutions_start - 1),
            "solutions": (solutions_start, glossary_start - 1),
            "glossary": (glossary_start, index_start - 1),
            "index": (index_start, total),
        }

    # -- body -------------------------------------------------------------- #
    def parse_body(self, pages: list[dict[str, Any]], span: tuple[int, int]) -> None:
        """Split the chapters into sections, and split out exercise subsections."""
        current: Section | None = None
        chapter = ""
        chapter_title = ""
        exercise_section_ids: set[str] = set()
        pending_exercise: ExerciseRecord | None = None

        for page in pages:
            if not (span[0] <= page["page"] <= span[1]):
                continue
            body = clean_page(page["text"])
            for line in body.split("\n"):
                chapter_match = _CHAPTER_HEADING.match(line)
                if chapter_match:
                    chapter = chapter_match.group(1)
                    chapter_title = ""
                    continue

                heading = _SECTION_HEADING.match(line)
                if heading:
                    section_id, title = heading.group(1), heading.group(2).strip()
                    title = re.sub(r"\s+\d+$", "", title).strip()
                    depth = section_id.count(".")

                    if depth == 1:  # X.Y  - a real section
                        if pending_exercise:
                            self._finish_exercise(pending_exercise)
                            pending_exercise = None
                        current = Section(
                            section_id=section_id,
                            title=title,
                            page=page["page"],
                            chapter=chapter or section_id.split(".")[0],
                        )
                        self.sections.append(current)
                        if title.lower().startswith("exercise"):
                            exercise_section_ids.add(section_id)
                        continue

                    if depth == 2:  # X.Y.Z - subsection
                        parent = section_id.rsplit(".", 1)[0]
                        if parent in exercise_section_ids:
                            if pending_exercise:
                                self._finish_exercise(pending_exercise)
                            pending_exercise = ExerciseRecord(
                                exercise_id=section_id,
                                title=title,
                                question="",
                                chapter=chapter or section_id.split(".")[0],
                                page=page["page"],
                            )
                            current = None
                            continue
                        # Ordinary subsection: keep it inside the parent section
                        # but record the heading so retrieval can match on it.
                        if current is not None:
                            current.text += f"\n\n{title}\n"
                            continue

                if pending_exercise is not None:
                    pending_exercise.question += line + "\n"
                elif current is not None:
                    if line.startswith("Reference") or re.match(r"^\d+\.\s+\w+.*\(\d{4}", line):
                        continue  # reference list entries
                    current.text += line + "\n"
                elif chapter and not chapter_title:
                    chapter_title = line

        if pending_exercise:
            self._finish_exercise(pending_exercise)

        for section in self.sections:
            section.text = join_wrapped(section.text)

    def _finish_exercise(self, record: ExerciseRecord) -> None:
        record.question = join_wrapped(record.question).strip()
        if not record.question:
            self.warnings.append(f"exercise {record.exercise_id} has no question text")
        record.concepts = detect_concepts(f"{record.title} {record.question}")
        self.exercises.append(record)

    # -- solutions --------------------------------------------------------- #
    def parse_solutions(self, pages: list[dict[str, Any]], span: tuple[int, int]) -> None:
        current: SolutionRecord | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal current, buffer
            if current is None:
                return
            text = join_wrapped("\n".join(buffer)).strip()
            current.solution_text = text
            current.answer_units = split_answer_units(text)
            self.solutions.append(current)
            current, buffer = None, []

        for page in pages:
            if not (span[0] <= page["page"] <= span[1]):
                continue
            body = clean_page(page["text"])
            for line in body.split("\n"):
                if line.startswith("Chapter ") and ":" in line:
                    continue  # "Chapter 1: Introduction" divider
                heading = _SOLUTION_HEADING.match(line)
                if heading:
                    flush()
                    current = SolutionRecord(
                        exercise_id=heading.group(1),
                        title=heading.group(2).strip(),
                        solution_text="",
                        answer_units=[],
                        page=page["page"],
                    )
                    continue
                if current is not None:
                    buffer.append(line)
        flush()

    # -- glossary ---------------------------------------------------------- #
    def parse_glossary(self, pages: list[dict[str, Any]], span: tuple[int, int]) -> None:
        """Parse `Term  Definition` entries.

        Glossary head-words are set in bold in the source PDF, so the term
        boundary is read from the font rather than guessed from punctuation.
        Without font data (e.g. a fixture supplying plain text) the parser falls
        back to a conservative textual heuristic.
        """
        fallback_start = re.compile(
            r"^([A-Z0-9][A-Za-z0-9\u00b2\-/\u2019' ]{2,60}?(?:\([A-Z0-9]{2,8}\))?)\s+"
            r"((?:\u2192\s*)?[a-zA-Z\u201c][^\n]*)$"
        )
        current_term: str | None = None
        current_page = span[0]
        buffer: list[str] = []

        def flush() -> None:
            nonlocal current_term, buffer
            if current_term and buffer:
                definition = join_wrapped(" ".join(buffer)).strip()
                if len(definition.split()) >= 4:
                    self.glossary.append(
                        GlossaryTerm(term=current_term, definition=definition, page=current_page)
                    )
            current_term, buffer = None, []

        for page in pages:
            if not (span[0] <= page["page"] <= span[1]):
                continue
            bold = sorted(page.get("bold_phrases") or [], key=len, reverse=True)
            body = clean_page(page["text"])
            for line in body.split("\n"):
                if line.startswith("\u00a9") or "et al." in line or line.startswith("A. Winter"):
                    continue

                term: str | None = None
                remainder = ""
                for phrase in bold:
                    if len(phrase) >= 3 and line.startswith(phrase):
                        term, remainder = phrase, line[len(phrase) :].strip()
                        break

                if term is None and not bold:
                    match = fallback_start.match(line)
                    if match and not line[0].islower():
                        term, remainder = match.group(1).strip(), match.group(2).strip()

                if term is not None:
                    flush()
                    current_page = page["page"]
                    current_term = term.strip(" :\u2013-")
                    buffer = [remainder] if remainder else []
                elif current_term:
                    buffer.append(line)
        flush()

    # -- orchestration ----------------------------------------------------- #
    def ingest_pages(self, pages: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
        regions = self.find_regions(pages)
        self.parse_body(pages, regions["body"])
        self.parse_solutions(pages, regions["solutions"])
        self.parse_glossary(pages, regions["glossary"])
        return regions

    def ingest_pdf(self, pdf_path: str | Path) -> dict[str, tuple[int, int]]:
        return self.ingest_pages(self.read_pdf_pages(pdf_path))

    # -- artefact building -------------------------------------------------- #
    def build_searchable_chunks(
        self, target_tokens: int = 220, overlap_tokens: int = 40, min_tokens: int = 40
    ) -> list[Chunk]:
        """Build the searchable corpus.

        Includes textbook sections, glossary entries and exercise *questions*.
        Excludes, by construction, every solution record.
        """
        chunks: list[Chunk] = []

        for section in self.sections:
            if not section.text.strip():
                continue
            chunks.extend(
                chunk_section(
                    text=section.text,
                    doc_id=self.doc_id,
                    section_id=section.section_id,
                    section_title=section.title,
                    page=section.page,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    min_tokens=min_tokens,
                    kind="textbook",
                    metadata={"chapter": section.chapter},
                )
            )

        for index, term in enumerate(self.glossary):
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{self.doc_id}:glossary:{index:03d}:"
                        f"{re.sub(r'[^a-z0-9]+', '_', term.term.lower()).strip('_')}"
                    ),
                    doc_id=self.doc_id,
                    text=f"{term.term}: {term.definition}",
                    section_id="glossary",
                    section_title=f"Glossary — {term.term}",
                    page=term.page,
                    kind="glossary",
                    metadata={"term": term.term},
                )
            )

        for exercise in self.exercises:
            if not exercise.question.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{self.doc_id}:exercise:{exercise.exercise_id}",
                    doc_id=self.doc_id,
                    text=f"Exercise {exercise.exercise_id} {exercise.title}: {exercise.question}",
                    section_id=exercise.exercise_id,
                    section_title=f"Exercise {exercise.exercise_id} — {exercise.title}",
                    page=exercise.page,
                    kind="exercise",
                    metadata={"exercise_id": exercise.exercise_id, "chapter": exercise.chapter},
                )
            )
        return chunks

    def exercises_payload(self) -> list[dict[str, Any]]:
        solution_units = {s.exercise_id: len(s.answer_units) for s in self.solutions}
        return [
            {
                "exercise_id": e.exercise_id,
                "title": e.title,
                "question": e.question,
                "chapter": e.chapter,
                "page": e.page,
                "concepts": e.concepts,
                "protected_unit_count": solution_units.get(e.exercise_id, 0),
                "protected": e.exercise_id in solution_units,
            }
            for e in self.exercises
        ]

    def solutions_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "exercise_id": s.exercise_id,
                "title": s.title,
                "solution_text": s.solution_text,
                "answer_units": s.answer_units,
                "page": s.page,
            }
            for s in self.solutions
        ]


# --------------------------------------------------------------------------- #
# Writing and verification
# --------------------------------------------------------------------------- #


def write_corpus(chunks: Iterable[Chunk], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


ALLOWED_CHUNK_KINDS = frozenset({"textbook", "glossary", "exercise"})


def verify_provenance(
    chunks: list[Chunk], regions: dict[str, tuple[int, int]]
) -> list[str]:
    """The hard guarantee: no searchable chunk originates from a solution.

    Separation is structural — `build_searchable_chunks()` never reads the
    solution records at all — and this function proves it after the fact by
    checking every chunk's kind and source page.
    """
    violations: list[str] = []
    solutions_span = regions.get("solutions")

    for chunk in chunks:
        if chunk.kind not in ALLOWED_CHUNK_KINDS:
            violations.append(f"{chunk.chunk_id}: disallowed kind {chunk.kind!r}")
        if solutions_span and chunk.page is not None:
            low, high = solutions_span
            if low <= chunk.page <= high:
                violations.append(
                    f"{chunk.chunk_id}: page {chunk.page} lies inside the solutions region "
                    f"{low}-{high}"
                )
        if "solution" in (chunk.section_id or "").lower():
            violations.append(f"{chunk.chunk_id}: section id references a solution")
    return violations


def _longest_common_run(a: list[str], b: list[str]) -> int:
    """Longest run of identical consecutive tokens shared by two sequences."""
    if not a or not b:
        return 0
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(b):
        positions.setdefault(token, []).append(index)

    best = 0
    for i, token in enumerate(a):
        for j in positions.get(token, ()):
            if best and (i + best >= len(a) or j + best >= len(b)):
                continue
            run = 0
            while i + run < len(a) and j + run < len(b) and a[i + run] == b[j + run]:
                run += 1
            best = max(best, run)
    return best


def solution_overlap_report(
    chunks: list[Chunk], solutions: list[dict[str, Any]], ngram: int = 8
) -> list[dict[str, Any]]:
    """Diagnostic: how much wording each solution shares with the corpus.

    Some overlap is expected and legitimate — an official solution reuses the
    textbook's own definitions. What must never happen is a *substantial*
    reproduction of a solution inside an indexed chunk.
    """
    corpus_grams: dict[tuple[str, ...], list[str]] = {}
    corpus_tokens: dict[str, list[str]] = {}
    for chunk in chunks:
        tokens = tokenize(chunk.text)
        corpus_tokens[chunk.chunk_id] = tokens
        for i in range(max(0, len(tokens) - ngram + 1)):
            corpus_grams.setdefault(tuple(tokens[i : i + ngram]), []).append(chunk.chunk_id)

    report: list[dict[str, Any]] = []
    for solution in solutions:
        tokens = tokenize(solution.get("solution_text", ""))
        total = max(1, len(tokens) - ngram + 1)
        matched = 0
        candidates: set[str] = set()
        for i in range(max(0, len(tokens) - ngram + 1)):
            hits = corpus_grams.get(tuple(tokens[i : i + ngram]))
            if hits:
                matched += 1
                candidates.update(hits)

        worst_run, worst_chunk = 0, None
        for chunk_id in candidates:
            run = _longest_common_run(tokens, corpus_tokens[chunk_id])
            if run > worst_run:
                worst_run, worst_chunk = run, chunk_id

        report.append(
            {
                "exercise_id": solution.get("exercise_id"),
                "solution_tokens": len(tokens),
                "ngram_coverage": round(matched / total, 4),
                "max_verbatim_run": worst_run,
                "run_ratio": round(worst_run / max(1, len(tokens)), 4),
                "closest_chunk": worst_chunk,
            }
        )
    return report


def verify_no_solution_reproduction(
    report: list[dict[str, Any]],
    max_run: int = 40,
    max_run_ratio: float = 0.5,
    max_coverage: float = 0.5,
) -> list[str]:
    """Fail ingestion if any chunk substantially reproduces a solution.

    Thresholds sit well above the incidental overlap the textbook itself
    produces (observed maximum: a 27-token shared definition, ratio 0.04) and
    well below anything that would constitute a usable leaked answer.
    """
    violations: list[str] = []
    for entry in report:
        if entry["max_verbatim_run"] >= max_run:
            violations.append(
                f"solution {entry['exercise_id']}: {entry['max_verbatim_run']}-token verbatim run "
                f"in chunk {entry['closest_chunk']}"
            )
        elif entry["run_ratio"] >= max_run_ratio:
            violations.append(
                f"solution {entry['exercise_id']}: {entry['run_ratio']:.0%} of the solution appears "
                f"verbatim in chunk {entry['closest_chunk']}"
            )
        elif entry["ngram_coverage"] >= max_coverage:
            violations.append(
                f"solution {entry['exercise_id']}: {entry['ngram_coverage']:.0%} n-gram coverage "
                f"by the searchable corpus"
            )
    return violations


def run_ingestion(
    pdf_path: str | Path | None,
    searchable_dir: str | Path,
    exercises_dir: str | Path,
    protected_dir: str | Path,
    chunk_config: dict[str, Any] | None = None,
    doc_id: str = "winter2023_his",
    pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full pipeline. Returns a manifest; raises if separation is violated."""
    chunk_config = chunk_config or {}
    ingestor = TextbookIngestor(doc_id=doc_id)

    if pages is None:
        if pdf_path is None:
            raise ValueError("either pdf_path or pages must be provided")
        pages = ingestor.read_pdf_pages(pdf_path)
    regions = ingestor.ingest_pages(pages)

    chunks = ingestor.build_searchable_chunks(
        target_tokens=int(chunk_config.get("target_tokens", 220)),
        overlap_tokens=int(chunk_config.get("overlap_tokens", 40)),
        min_tokens=int(chunk_config.get("min_tokens", 40)),
    )
    exercises = ingestor.exercises_payload()
    solutions = ingestor.solutions_payload()

    provenance_violations = verify_provenance(chunks, regions)
    if provenance_violations:
        raise RuntimeError(
            "searchable corpus contains chunks of protected provenance; refusing to write:\n"
            + "\n".join(provenance_violations[:5])
        )

    overlap = solution_overlap_report(chunks, solutions)
    reproduction_violations = verify_no_solution_reproduction(overlap)
    if reproduction_violations:
        raise RuntimeError(
            "searchable corpus reproduces protected solution text; refusing to write:\n"
            + "\n".join(reproduction_violations[:5])
        )

    corpus_path = Path(searchable_dir) / "corpus.jsonl"
    exercises_path = Path(exercises_dir) / "exercises.json"
    solutions_path = Path(protected_dir) / "solutions.json"

    written = write_corpus(chunks, corpus_path)
    write_json(exercises, exercises_path)
    write_json(solutions, solutions_path)

    worst = max(overlap, key=lambda e: e["max_verbatim_run"], default=None)
    manifest = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path) if pdf_path else "(pages supplied directly)",
        "pages": len(pages),
        "regions": {k: list(v) for k, v in regions.items()},
        "sections": len(ingestor.sections),
        "glossary_terms": len(ingestor.glossary),
        "exercises": len(exercises),
        "solutions": len(solutions),
        "chunks": written,
        "chunk_kinds": {
            kind: sum(1 for c in chunks if c.kind == kind)
            for kind in sorted(ALLOWED_CHUNK_KINDS)
        },
        "exercises_without_solution": sorted(
            {e["exercise_id"] for e in exercises} - {s["exercise_id"] for s in solutions}
        ),
        "solutions_without_exercise": sorted(
            {s["exercise_id"] for s in solutions} - {e["exercise_id"] for e in exercises}
        ),
        "separation": {
            "provenance_verified": True,
            "reproduction_verified": True,
            "worst_overlap": worst,
        },
        "warnings": ingestor.warnings[:20],
        "paths": {
            "corpus": str(corpus_path),
            "exercises": str(exercises_path),
            "protected": str(solutions_path),
        },
    }
    write_json(manifest, Path(searchable_dir) / "manifest.json")
    return manifest
