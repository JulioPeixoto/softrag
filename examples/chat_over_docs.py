"""Chat over a directory of documents, with citations.

Indexes a folder, then answers questions in a loop with streamed output and a
source list under each answer. Re-running it is cheap: unchanged files are
recognised and skipped, and changed ones are re-indexed in place.

Usage:
    python examples/chat_over_docs.py [directory] [--db kb.db]

With no API key or Ollama daemon available this still runs -- it falls back to
the built-in hash embedder and echoes the retrieved context instead of
generating -- which is a fine way to inspect what retrieval is actually finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import softrag


def index(rag: softrag.Rag, directory: Path) -> None:
    """Index a directory, reporting progress on one rewritten line."""

    def progress(source: str, done: int, total: int) -> None:
        name = Path(source).name[:40]
        print(f"\r  [{done}/{total}] {name:<42}", end="", flush=True)

    results = rag.add_directory(directory, on_progress=progress)
    print("\r" + " " * 60, end="\r")

    added = sum(r.chunks_added for r in results)
    skipped = sum(r.chunks_skipped for r in results)
    failed = [r for r in results if not r.ok]
    print(f"  {len(results)} files: {added} chunks added, {skipped} unchanged")
    for result in failed:
        print(f"  ! {result.source}: {result.error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="docs", help="directory to index")
    parser.add_argument("--db", default="kb.db", help="index file to use")
    parser.add_argument("--top-k", type=int, default=5, help="chunks per answer")
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 1

    with softrag.connect(args.db) as rag:
        print(f"indexing {directory} into {args.db}")
        index(rag, directory)

        stats = rag.stats()
        print(
            f"\n{stats.documents} chunks from {stats.sources} sources, "
            f"{stats.size_mb:.1f} MB on disk"
        )
        print("Ask a question, or press Ctrl-D to quit.\n")

        while True:
            try:
                question = input("? ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue

            stream = rag.query(question, top_k=args.top_k, stream=True)
            print()
            for delta in stream:
                print(delta, end="", flush=True)
            print()

            if stream.sources:
                print("\n  sources:")
                for hit in stream.hits:
                    preview = hit.text[:60].replace("\n", " ")
                    print(f"    {hit.score:.4f}  {hit.source}  {preview}...")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
