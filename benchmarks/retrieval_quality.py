"""Does hybrid retrieval actually beat either half of it?

softrag's central claim is that fusing dense and keyword retrieval retrieves
things neither finds alone. This script tests that claim instead of asserting
it, on a corpus built so that each failure mode is present by construction:

* **lexical queries** use a rare exact token -- a part number, an error code, an
  identifier. Dense retrieval is bad at these, because a rare string carries
  almost no semantic signal and lands nowhere useful in embedding space.
* **semantic queries** are paraphrases that share no content words with the
  document that answers them. BM25 is blind to these by definition: it matches
  terms, and there are no shared terms to match.
* **mixed queries** contain both an exact token and paraphrased intent.

A retriever that wins on one group and loses on the other is not better than
the alternative; it is differently blind. The number that matters is the
average across all three.

By default this runs with the dependency-free :class:`~softrag.HashEmbedder`,
which is a bag-of-words hash and therefore has no semantic ability at all. That
is deliberate for CI: it makes the lexical and mixed groups meaningful and keeps
the run instant. For an honest reading of the semantic group, pass a real model:

    python benchmarks/retrieval_quality.py --embedder local
    python benchmarks/retrieval_quality.py --embedder openai
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from softrag import HashEmbedder, Rag


@dataclass(frozen=True)
class Case:
    """One labelled query."""

    group: str
    query: str
    relevant: str


DOCUMENTS: dict[str, str] = {
    "invoice-policy": (
        "Refunds are processed within fourteen business days of an approved "
        "request. Customers on annual plans receive a prorated amount for the "
        "unused portion of the term. Approval requires the original order "
        "reference and is handled by the billing team."
    ),
    "error-codes": (
        "Error ERR_4417 indicates that the write-ahead log could not be "
        "checkpointed because a reader transaction is still open. Close the "
        "long-running reader and retry. Error ERR_4418 is the related case "
        "where the log file exceeded the configured size limit."
    ),
    "onboarding": (
        "New engineers are paired with a mentor for their first six weeks. The "
        "mentor reviews the first ten pull requests and runs a weekly check-in. "
        "Access to production systems is granted only after the security "
        "training module is complete."
    ),
    "hardware": (
        "The RX-9920 sensor module operates between minus twenty and sixty "
        "degrees Celsius. It draws 3.3 volts and reports readings over I2C at "
        "address 0x4A. Calibration drifts by roughly one percent per year."
    ),
    "deployment": (
        "Releases go out on Tuesdays and Thursdays. A change must sit in "
        "staging for at least one full business day before promotion. Rollback "
        "is performed by re-pointing the traffic router at the previous "
        "revision, which takes under a minute."
    ),
    "vacation": (
        "Time off accrues at one and a quarter days per month worked. Unused "
        "days carry over to the following year up to a maximum of ten. Requests "
        "longer than five consecutive days need approval from a second manager."
    ),
    "security": (
        "Credentials are rotated every ninety days. Service accounts use "
        "short-lived tokens issued by the identity provider rather than static "
        "secrets. Any credential committed to a repository must be revoked "
        "immediately, not merely removed from the history."
    ),
    "database": (
        "The primary datastore runs in write-ahead logging mode with one writer "
        "and many concurrent readers. Backups are taken by copying the file "
        "after a checkpoint. Point-in-time recovery is not supported."
    ),
}

CASES: Sequence[Case] = (
    # Rare exact tokens: keyword retrieval should carry these.
    Case("lexical", "ERR_4417", "error-codes"),
    Case("lexical", "RX-9920", "hardware"),
    Case("lexical", "what does ERR_4418 mean", "error-codes"),
    Case("lexical", "I2C address 0x4A", "hardware"),
    # Paraphrases sharing no content words with the answer.
    Case("semantic", "how long until I get my money back", "invoice-policy"),
    Case("semantic", "who helps someone who just joined the team", "onboarding"),
    Case("semantic", "when can we ship code to customers", "deployment"),
    Case("semantic", "can I keep days I did not use this year", "vacation"),
    Case("semantic", "how do we avoid leaving passwords lying around", "security"),
    # Both signals present.
    Case("mixed", "ERR_4417 checkpoint problem how do I fix it", "error-codes"),
    Case("mixed", "sensor RX-9920 temperature range", "hardware"),
    Case("mixed", "rollback procedure for a bad release", "deployment"),
    Case("mixed", "prorated refund on an annual plan", "invoice-policy"),
)


def recall_at_k(ranked: Sequence[str], relevant: str, k: int) -> float:
    """1.0 if the relevant source appears in the top ``k``, else 0.0."""
    return 1.0 if relevant in ranked[:k] else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: str) -> float:
    """1 / rank of the relevant source, or 0.0 if it never appears."""
    for index, source in enumerate(ranked, start=1):
        if source == relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: str, k: int) -> float:
    """nDCG with a single relevant document, so IDCG is 1."""
    for index, source in enumerate(ranked[:k], start=1):
        if source == relevant:
            return 1.0 / math.log2(index + 1)
    return 0.0


def build_embedder(kind: str) -> object:
    """Resolve the ``--embedder`` choice into a real embedder."""
    if kind == "hash":
        return HashEmbedder(256)
    if kind == "local":
        from softrag.providers.local import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    if kind == "openai":
        from softrag.providers.openai import OpenAIEmbedder

        return OpenAIEmbedder()
    raise SystemExit(f"unknown embedder {kind!r}")


def evaluate(rag: Rag, mode: str, *, top_k: int) -> dict[str, dict[str, float]]:
    """Score every case under one search mode, grouped by case kind."""
    per_group: dict[str, list[dict[str, float]]] = {}
    for case in CASES:
        hits = rag.search(case.query, top_k=top_k, mode=mode)
        # A source can occupy several of the top slots; rank by first appearance.
        ranked: list[str] = []
        for hit in hits:
            if hit.source not in ranked:
                ranked.append(hit.source)
        per_group.setdefault(case.group, []).append(
            {
                "recall@1": recall_at_k(ranked, case.relevant, 1),
                "recall@3": recall_at_k(ranked, case.relevant, 3),
                "mrr": reciprocal_rank(ranked, case.relevant),
                "ndcg@3": ndcg_at_k(ranked, case.relevant, 3),
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for group, rows in per_group.items():
        summary[group] = {
            metric: sum(row[metric] for row in rows) / len(rows) for metric in rows[0]
        }
    every = [row for rows in per_group.values() for row in rows]
    summary["ALL"] = {
        metric: sum(row[metric] for row in every) / len(every) for metric in every[0]
    }
    return summary


def render(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render mode-by-group results as a table."""
    modes = list(results)
    groups = ["lexical", "semantic", "mixed", "ALL"]
    metrics = ["recall@1", "recall@3", "mrr", "ndcg@3"]

    lines: list[str] = []
    for metric in metrics:
        lines.append(f"\n{metric}")
        lines.append(f"  {'mode':<10}" + "".join(f"{g:>12}" for g in groups))
        for mode in modes:
            cells = "".join(
                f"{results[mode].get(g, {}).get(metric, float('nan')):>12.3f}"
                for g in groups
            )
            lines.append(f"  {mode:<10}{cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedder",
        choices=["hash", "local", "openai"],
        default="hash",
        help="Which embedding backend to evaluate with.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Hits retrieved per query.")
    args = parser.parse_args(argv)

    embedder = build_embedder(args.embedder)
    rag = Rag(
        embed_model=embedder,
        db_path=":memory:",
        auto=False,
        chunk_size=400,
        chunk_overlap=80,
    )
    for name, text in DOCUMENTS.items():
        rag.add_text(text, name=name)

    results = {
        mode: evaluate(rag, mode, top_k=args.top_k)
        for mode in ("keyword", "vector", "hybrid")
    }

    print(
        f"embedder: {args.embedder}   documents: {len(DOCUMENTS)}   "
        f"queries: {len(CASES)}   top_k: {args.top_k}"
    )
    print(render(results))

    hybrid = results["hybrid"]["ALL"]["mrr"]
    best_single = max(results["keyword"]["ALL"]["mrr"], results["vector"]["ALL"]["mrr"])
    print(
        f"\nhybrid MRR {hybrid:.3f} vs best single-mode {best_single:.3f} "
        f"({'+' if hybrid >= best_single else ''}{(hybrid - best_single):.3f})"
    )
    rag.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
