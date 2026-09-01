"""Benchmark softrag against a synthetic corpus.

Measures what actually matters for an embedded engine: how long ingestion takes,
how query latency scales with corpus size, how much disk a chunk costs, and
whether metadata filtering changes the shape of the curve.

The embedder is deliberately :class:`~softrag.HashEmbedder` -- it is fast and
local, so the numbers describe *softrag's* overhead rather than an embedding
API's latency. That is the only thing a benchmark of this library can honestly
claim to measure.

Usage:
    python benchmarks/bench.py                  # default sizes
    python benchmarks/bench.py --sizes 1000 10000 50000
    python benchmarks/bench.py --dimensions 768 --json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from softrag import EchoChatModel, HashEmbedder, Rag  # noqa: E402

WORDS = """
retrieval augmented generation embedding vector database sqlite index query
document chunk semantic search keyword hybrid ranking fusion reciprocal rank
model context window token latency throughput cache storage engine schema
migration transaction concurrency thread process memory disk network cluster
python rust javascript compiler runtime garbage collector allocation pointer
""".split()


@dataclass
class Timing:
    """Summary statistics for a repeated measurement, in milliseconds."""

    label: str
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def measure(
        cls, label: str, fn: Callable[[int], Any], *, runs: int, warmup: int = 3
    ) -> "Timing":
        for i in range(warmup):
            fn(i)
        gc.collect()
        durations: List[float] = []
        for i in range(runs):
            start = time.perf_counter()
            fn(i)
            durations.append((time.perf_counter() - start) * 1000)
        durations.sort()
        return cls(
            label=label,
            samples=runs,
            mean_ms=statistics.fmean(durations),
            p50_ms=durations[len(durations) // 2],
            p95_ms=durations[min(len(durations) - 1, int(len(durations) * 0.95))],
            min_ms=durations[0],
            max_ms=durations[-1],
        )


@dataclass
class SizeResult:
    """Everything measured at one corpus size."""

    chunks: int
    dimensions: int
    ingest_seconds: float
    chunks_per_second: float
    db_bytes: int
    bytes_per_chunk: float
    timings: List[Timing] = field(default_factory=list)


def make_document(rng: random.Random, words: int = 220) -> str:
    """Generate a paragraph-shaped document with a Zipf-ish word distribution."""
    body = []
    for _ in range(words):
        # Skewing toward the front of the vocabulary gives BM25 something
        # realistic to work with: a few common terms, many rare ones.
        index = min(int(rng.paretovariate(1.2)) - 1, len(WORDS) - 1)
        body.append(WORDS[index])
        if rng.random() < 0.06:
            body.append("\n")
    return " ".join(body)


def run_size(
    chunks: int, *, dimensions: int, queries: int, rng: random.Random
) -> SizeResult:
    """Build an index of roughly ``chunks`` chunks and time queries against it."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.db"
        rag = Rag(
            embed_model=HashEmbedder(dimensions),
            chat_model=EchoChatModel(),
            db_path=db_path,
            chunk_size=600,
            chunk_overlap=100,
        )

        documents = max(1, chunks // 3)
        payload = [
            (
                make_document(rng),
                {"bucket": i % 10, "year": 2020 + (i % 6), "kind": "synthetic"},
            )
            for i in range(documents)
        ]

        start = time.perf_counter()
        for i, (text, metadata) in enumerate(payload):
            rag.add_text(text, name=f"doc-{i}", metadata=metadata)
        ingest_seconds = time.perf_counter() - start

        actual = len(rag)
        rag.optimize()
        size = db_path.stat().st_size

        probes = [
            " ".join(rng.sample(WORDS, rng.randint(2, 6))) for _ in range(queries)
        ]

        timings = [
            Timing.measure(
                "search hybrid",
                lambda i: rag.search(probes[i % len(probes)], top_k=5),
                runs=queries,
            ),
            Timing.measure(
                "search vector",
                lambda i: rag.search(probes[i % len(probes)], top_k=5, mode="vector"),
                runs=queries,
            ),
            Timing.measure(
                "search keyword",
                lambda i: rag.search(probes[i % len(probes)], top_k=5, mode="keyword"),
                runs=queries,
            ),
            Timing.measure(
                "search + filter",
                lambda i: rag.search(
                    probes[i % len(probes)], top_k=5, where={"bucket": i % 10}
                ),
                runs=queries,
            ),
            Timing.measure(
                "search + mmr",
                lambda i: rag.search(
                    probes[i % len(probes)], top_k=5, diversity=0.5
                ),
                runs=queries,
            ),
        ]

        rag.close()
        return SizeResult(
            chunks=actual,
            dimensions=dimensions,
            ingest_seconds=ingest_seconds,
            chunks_per_second=actual / ingest_seconds if ingest_seconds else 0.0,
            db_bytes=size,
            bytes_per_chunk=size / actual if actual else 0.0,
            timings=timings,
        )


def render(results: List[SizeResult]) -> str:
    """Render results as aligned plain-text tables."""
    lines: List[str] = []
    lines.append("ingest")
    lines.append(f"{'chunks':>10} {'dim':>5} {'seconds':>9} {'chunks/s':>10} "
                 f"{'db size':>10} {'bytes/chunk':>12}")
    for result in results:
        lines.append(
            f"{result.chunks:>10,} {result.dimensions:>5} "
            f"{result.ingest_seconds:>9.2f} {result.chunks_per_second:>10,.0f} "
            f"{result.db_bytes / 1e6:>9.1f}M {result.bytes_per_chunk:>12,.0f}"
        )

    lines.append("")
    lines.append("query latency (ms)")
    labels = [t.label for t in results[0].timings] if results else []
    header = f"{'chunks':>10} " + " ".join(f"{label:>16}" for label in labels)
    lines.append(header)
    for result in results:
        cells = " ".join(f"{t.p50_ms:>10.2f} p95" for t in result.timings)
        lines.append(f"{result.chunks:>10,} {cells}")

    lines.append("")
    lines.append("query latency detail")
    for result in results:
        lines.append(f"  {result.chunks:,} chunks")
        for timing in result.timings:
            lines.append(
                f"    {timing.label:<18} p50 {timing.p50_ms:>7.2f}  "
                f"p95 {timing.p95_ms:>7.2f}  mean {timing.mean_ms:>7.2f}"
            )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 50_000],
        help="Approximate chunk counts to benchmark.",
    )
    parser.add_argument(
        "--dimensions", type=int, default=384, help="Embedding width to simulate."
    )
    parser.add_argument(
        "--queries", type=int, default=50, help="Queries timed per configuration."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument("--json", type=Path, help="Also write raw results here.")
    args = parser.parse_args(argv)

    results: List[SizeResult] = []
    for size in args.sizes:
        print(f"benchmarking {size:,} chunks...", file=sys.stderr, flush=True)
        results.append(
            run_size(
                size,
                dimensions=args.dimensions,
                queries=args.queries,
                rng=random.Random(args.seed),
            )
        )

    print(render(results))

    if args.json:
        payload: Dict[str, Any] = {
            "dimensions": args.dimensions,
            "results": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
