#!/usr/bin/env python3
"""Fetch the open-access textbook.

    Alfred Winter, Elske Ammenwerth, Reinhold Haux, Michael Marschollek,
    Bianca Steiner, Franziska Jahn.
    Health Information Systems: Technological and Management Perspectives,
    3rd edition, Springer, 2023. DOI 10.1007/978-3-031-12310-8
    Open Access, CC BY 4.0.

If you already have the PDF locally, pass `--from-file` instead of downloading.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_ai_tutor.config import load_config  # noqa: E402

# Springer's canonical open-access download URL for this book.
CANONICAL_URL = "https://link.springer.com/content/pdf/10.1007/978-3-031-12310-8.pdf"
DOI = "10.1007/978-3-031-12310-8"


def download(url: str, destination: Path, timeout: float = 120.0) -> Path:
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with httpx.stream(
        "GET", url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "medical-ai-tutor/0.1"}
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(destination, "wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 16):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / total:6.1%}", end="", flush=True)
    print()
    return destination


def main() -> int:
    config = load_config()
    default_destination = config.path("storage.raw_dir") / "winter2023_his.pdf"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=CANONICAL_URL, help="source URL")
    parser.add_argument("--out", type=Path, default=default_destination, help="destination path")
    parser.add_argument(
        "--from-file", type=Path, help="copy an already-downloaded PDF instead of fetching"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"already present: {args.out} (use --force to replace)")
        return 0

    if args.from_file:
        source = args.from_file.expanduser()
        if not source.exists():
            print(f"error: {source} does not exist", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, args.out)
        print(f"copied {source} -> {args.out}")
    else:
        try:
            download(args.url, args.out)
        except Exception as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            print(
                f"\nFetch it manually from https://doi.org/{DOI} (Open Access, CC BY 4.0)\n"
                f"and place it at {args.out}, or re-run with --from-file <path>.",
                file=sys.stderr,
            )
            return 1

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"ready: {args.out} ({size_mb:.1f} MB)")
    print(f"source: https://doi.org/{DOI} — © the authors, CC BY 4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
