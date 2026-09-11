#!/usr/bin/env python3
"""Ingest the textbook into searchable / exercise / protected artefacts.

Searchable content and official solutions are written to physically separate
directories, and the split is verified before anything is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_ai_tutor.config import load_config  # noqa: E402
from medical_ai_tutor.retrieval.ingest import run_ingestion  # noqa: E402


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=config.path("storage.raw_dir") / "winter2023_his.pdf",
        help="path to the textbook PDF",
    )
    parser.add_argument("--doc-id", default="winter2023_his")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(
            f"error: {args.pdf} not found.\nRun `make fetch-data` first, or pass --pdf <path>.",
            file=sys.stderr,
        )
        return 1

    manifest = run_ingestion(
        pdf_path=args.pdf,
        searchable_dir=config.path("storage.searchable_dir"),
        exercises_dir=config.path("storage.exercises_dir"),
        protected_dir=config.path("storage.protected_dir"),
        chunk_config=config.get("retrieval.chunk", {}),
        doc_id=args.doc_id,
    )

    if args.quiet:
        print(json.dumps(manifest["chunk_kinds"]))
        return 0

    print("=" * 66)
    print("INGESTION COMPLETE")
    print("=" * 66)
    print(f"  pages                 {manifest['pages']}")
    print(f"  regions               {manifest['regions']}")
    print(f"  sections              {manifest['sections']}")
    print(f"  glossary terms        {manifest['glossary_terms']}")
    print(f"  exercises             {manifest['exercises']}")
    print(f"  official solutions    {manifest['solutions']}  (protected)")
    print(f"  searchable chunks     {manifest['chunks']}  {manifest['chunk_kinds']}")
    print()
    separation = manifest["separation"]
    print(f"  provenance verified   {separation['provenance_verified']}")
    print(f"  reproduction verified {separation['reproduction_verified']}")
    worst = separation.get("worst_overlap") or {}
    if worst:
        print(
            f"  closest overlap       solution {worst.get('exercise_id')}: "
            f"{worst.get('max_verbatim_run')} shared tokens "
            f"({worst.get('run_ratio', 0):.1%} of the solution)"
        )
    if manifest["exercises_without_solution"]:
        print(f"  exercises w/o solution {manifest['exercises_without_solution']}")
    if manifest["warnings"]:
        print(f"  warnings              {len(manifest['warnings'])}")
    print()
    for label, path in manifest["paths"].items():
        print(f"  {label:<10} {path}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
