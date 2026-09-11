"""Retrieval works offline, and never carries protected solutions."""

from __future__ import annotations

import json

from medical_ai_tutor.retrieval.bm25 import BM25Index, tokenize
from medical_ai_tutor.retrieval.chunking import Chunk, chunk_section
from medical_ai_tutor.retrieval.hybrid import HybridRetriever
from medical_ai_tutor.retrieval.ingest import (
    ALLOWED_CHUNK_KINDS,
    solution_overlap_report,
    verify_no_solution_reproduction,
    verify_provenance,
)


def _corpus(ingested) -> list[Chunk]:
    path = ingested["searchable"] / "corpus.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]


def test_ingestion_produces_all_three_artefacts(ingested):
    manifest = ingested["manifest"]
    assert manifest["sections"] > 0
    assert manifest["exercises"] == 2
    assert manifest["solutions"] == 2
    assert manifest["chunks"] > 0
    assert manifest["exercises_without_solution"] == []


def test_bm25_ranks_lexically_without_a_model():
    index = BM25Index().build(
        [
            ("a", "Interoperability lets application components exchange and use data."),
            ("b", "A communication server routes and transforms messages."),
            ("c", "Knowledge is general information beyond one individual case."),
        ]
    )
    assert index.search("what is interoperability", 1)[0][0] == "a"
    assert index.search("communication server routing", 1)[0][0] == "b"


def test_retrieval_finds_fixture_content(ingested):
    retriever = HybridRetriever.from_corpus_file(ingested["searchable"] / "corpus.jsonl")
    assert retriever.mode == "lexical"  # offline fallback, no model download
    hits = retriever.search("what is interoperability", top_k=3)
    assert hits
    assert any("interoperab" in hit.text.lower() for hit in hits)
    assert hits[0].citation()


def test_retrieval_is_deterministic(ingested):
    retriever = HybridRetriever.from_corpus_file(ingested["searchable"] / "corpus.jsonl")
    first = [hit.chunk_id for hit in retriever.search("communication server", top_k=3)]
    second = [hit.chunk_id for hit in retriever.search("communication server", top_k=3)]
    assert first == second


def test_near_duplicate_passages_are_deduplicated():
    text = "Interoperability is the ability of components to exchange and use data."
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id="d", text=text, section_id="1.3", section_title="I")
        for i in range(4)
    ]
    hits = HybridRetriever().index(chunks).search("interoperability exchange data", top_k=4)
    assert len(hits) == 1


def test_chunking_preserves_section_metadata():
    chunks = chunk_section(
        "First sentence about data. Second sentence about information. Third about knowledge.",
        doc_id="book",
        section_id="1.2",
        section_title="Data, Information, and Knowledge",
        page=3,
        target_tokens=8,
        overlap_tokens=2,
        min_tokens=2,
    )
    assert len(chunks) > 1
    assert all(c.section_id == "1.2" and c.page == 3 for c in chunks)
    assert all(c.chunk_id for c in chunks)


# --------------------------------------------------------------------------- #
# The separation guarantee
# --------------------------------------------------------------------------- #


def test_corpus_contains_no_protected_solution_records(ingested):
    """The required proof: nothing in the searchable corpus came from a solution."""
    chunks = _corpus(ingested)
    regions = {k: tuple(v) for k, v in ingested["manifest"]["regions"].items()}

    assert verify_provenance(chunks, regions) == []
    assert all(chunk.kind in ALLOWED_CHUNK_KINDS for chunk in chunks)

    solutions_low, solutions_high = regions["solutions"]
    assert all(
        chunk.page is None or not (solutions_low <= chunk.page <= solutions_high)
        for chunk in chunks
    )


def test_no_chunk_reproduces_a_solution(ingested):
    chunks = _corpus(ingested)
    with open(ingested["protected"] / "solutions.json", encoding="utf-8") as handle:
        solutions = json.load(handle)

    report = solution_overlap_report(chunks, solutions)
    assert verify_no_solution_reproduction(report) == []


def test_distinctive_solution_phrases_are_absent_from_the_corpus(ingested):
    """A stronger, content-level check on a phrase unique to a solution."""
    chunks = _corpus(ingested)
    with open(ingested["protected"] / "solutions.json", encoding="utf-8") as handle:
        solutions = json.load(handle)

    corpus_text = " ".join(" ".join(tokenize(chunk.text)) for chunk in chunks)
    for solution in solutions:
        tokens = tokenize(solution["solution_text"])
        # 12-gram: long enough that a shared definition cannot trigger it.
        grams = [" ".join(tokens[i : i + 12]) for i in range(0, max(1, len(tokens) - 12), 7)]
        assert not any(gram in corpus_text for gram in grams if gram)


def test_retrieval_never_returns_solution_text(ingested):
    """Query the retriever with the solutions themselves; results stay clean."""
    retriever = HybridRetriever.from_corpus_file(ingested["searchable"] / "corpus.jsonl")
    with open(ingested["protected"] / "solutions.json", encoding="utf-8") as handle:
        solutions = json.load(handle)

    for solution in solutions:
        for unit in solution["answer_units"]:
            for hit in retriever.search(unit, top_k=5):
                assert unit.strip() not in hit.text
