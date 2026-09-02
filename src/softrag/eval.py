"""Retrieval evaluation: the numbers behind "did that change help?".

Tuning a RAG pipeline by reading a handful of answers is guesswork. Twenty
labelled questions and a metric turn it into a measurement: raise ``chunk_size``,
re-run, compare nDCG. This module is that measurement, with no dependencies
beyond the standard library.

The data model is the one ``trec_eval``, ``pytrec_eval`` and ``ranx`` all use:

* **qrels** -- ground truth, ``{query_id: {doc_id: relevance}}``. Relevance is an
  integer; 0 means not relevant, and anything above 0 means relevant, with
  larger values meaning *more* relevant for the graded metrics.
* **run** -- what the system returned, ``{query_id: {doc_id: score}}``. Only the
  ordering of the scores matters; magnitudes are never compared across queries.

Every metric is computed per query and then averaged over the queries that have
at least one relevant document -- a query nothing is relevant for has no
defined recall, so scoring it would only dilute the average.

    from softrag.eval import evaluate_engine, compare

    dataset = [{"query": "refund window?", "relevant": ["policy.md"]}]
    print(evaluate_engine(rag, dataset).summary())
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigurationError

log = logging.getLogger("softrag.eval")

__all__ = [
    "DEFAULT_METRICS",
    "EvalResult",
    "average_precision",
    "compare",
    "comparison_table",
    "evaluate",
    "evaluate_engine",
    "evaluate_run",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]

#: Ground truth: query id -> document id -> relevance grade.
Qrels = Mapping[str, Mapping[str, float]]
#: System output: query id -> document id -> score (larger is better).
Run = Mapping[str, Mapping[str, float]]
#: Per-query relevance, or a bare collection of relevant document ids.
Relevance = Mapping[str, float] | Iterable[str]

#: A reasonable default panel: one recall metric, one early-precision metric,
#: and one graded metric.
DEFAULT_METRICS: tuple[str, ...] = ("recall@5", "mrr", "ndcg@10")

#: Cutoff used when a metric spec omits ``@k``.
DEFAULT_K = 10

_CUTOFF_METRICS = ("recall", "precision", "hit_rate", "ndcg")
_FULL_METRICS = ("mrr", "map")
_ALIASES = {"hitrate": "hit_rate", "success": "hit_rate", "ap": "map"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def recall_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Fraction of the relevant documents that made it into the top ``k``.

    ``recall@k = |{relevant} ∩ {top k}| / |{relevant}|``

    The metric that matters most for RAG: a document the retriever never
    returned is a document the generator cannot use, no matter how good the
    prompt is.

    Args:
        ranked: Retrieved document ids, best first.
        relevant: Ground-truth relevance, as a mapping or a collection of ids.
        k: Cutoff rank.

    Returns:
        A value in ``[0, 1]``; 0.0 when nothing is relevant.
    """
    grades = _as_relevance(relevant)
    positives = {doc for doc, grade in grades.items() if grade > 0}
    if not positives or k <= 0:
        return 0.0
    found = sum(1 for doc in ranked[:k] if doc in positives)
    return found / len(positives)


def precision_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    ``precision@k = |{relevant} ∩ {top k}| / k``

    The denominator is ``k`` rather than the length of the result list, matching
    ``trec_eval``: a system that returned only two documents for ``k=10`` is
    penalised for the eight it did not return.

    Args:
        ranked: Retrieved document ids, best first.
        relevant: Ground-truth relevance, as a mapping or a collection of ids.
        k: Cutoff rank.

    Returns:
        A value in ``[0, 1]``.
    """
    grades = _as_relevance(relevant)
    positives = {doc for doc, grade in grades.items() if grade > 0}
    if not positives or k <= 0:
        return 0.0
    found = sum(1 for doc in ranked[:k] if doc in positives)
    return found / k


def hit_rate_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Whether *any* relevant document appears in the top ``k``.

    ``hit_rate@k = 1 if |{relevant} ∩ {top k}| > 0 else 0``

    Also called success@k. Averaged over queries it answers the blunt question a
    RAG pipeline actually cares about: how often did retrieval give the
    generator anything at all to work with?

    Args:
        ranked: Retrieved document ids, best first.
        relevant: Ground-truth relevance, as a mapping or a collection of ids.
        k: Cutoff rank.

    Returns:
        ``1.0`` or ``0.0``.
    """
    grades = _as_relevance(relevant)
    positives = {doc for doc, grade in grades.items() if grade > 0}
    if not positives or k <= 0:
        return 0.0
    return 1.0 if any(doc in positives for doc in ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: Relevance, k: int | None = None) -> float:
    """Reciprocal of the rank of the first relevant document.

    ``RR = 1 / rank_of_first_relevant``, with ranks counted from 1, and 0 when
    no relevant document was retrieved. Averaged over queries this is Mean
    Reciprocal Rank. It rewards putting one good document at the very top, which
    is exactly right when the generator only reads the first few chunks.

    Args:
        ranked: Retrieved document ids, best first.
        relevant: Ground-truth relevance, as a mapping or a collection of ids.
        k: Optional cutoff; ranks beyond it count as not found.

    Returns:
        A value in ``[0, 1]``.
    """
    grades = _as_relevance(relevant)
    positives = {doc for doc, grade in grades.items() if grade > 0}
    if not positives:
        return 0.0
    limited = ranked if k is None else ranked[:k]
    for rank, doc in enumerate(limited, start=1):
        if doc in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Normalised discounted cumulative gain, with graded relevance.

    ``DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)``

    ``IDCG@k`` is the same sum over the relevance grades sorted descending, i.e.
    the best DCG any ranking of the ground truth could achieve, and
    ``nDCG@k = DCG@k / IDCG@k``.

    Unlike recall or MRR this uses the *grades*, so a 3-star document ranked
    first genuinely beats a 1-star document ranked first. It is the metric to
    quote when relevance is not binary.

    Args:
        ranked: Retrieved document ids, best first.
        relevant: Ground-truth relevance. A bare collection of ids is treated as
            uniformly grade 1, which reduces this to the binary case.
        k: Cutoff rank.

    Returns:
        A value in ``[0, 1]``; 0.0 when no document has a positive grade.
    """
    grades = _as_relevance(relevant)
    if k <= 0 or not grades:
        return 0.0

    gains = [float(grades.get(doc, 0.0)) for doc in ranked[:k]]
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]

    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return _dcg(gains) / idcg


def average_precision(
    ranked: Sequence[str], relevant: Relevance, k: int | None = None
) -> float:
    """Precision averaged over the ranks where a relevant document appears.

    ``AP = (1 / R) * Σ_{i: doc_i relevant} precision@i``

    where ``R`` is the total number of relevant documents in the ground truth --
    including any the system never retrieved, so missing documents are properly
    penalised. Averaged over queries this is Mean Average Precision (MAP), the
    single number that best summarises a whole ranking rather than just its head.

    Args:
        ranked: Retrieved document ids, best first.
        relevant: Ground-truth relevance, as a mapping or a collection of ids.
        k: Optional cutoff; ranks beyond it are ignored.

    Returns:
        A value in ``[0, 1]``.
    """
    grades = _as_relevance(relevant)
    positives = {doc for doc, grade in grades.items() if grade > 0}
    if not positives:
        return 0.0

    limited = ranked if k is None else ranked[:k]
    found = 0
    total = 0.0
    for rank, doc in enumerate(limited, start=1):
        if doc in positives:
            found += 1
            total += found / rank
    return total / len(positives)


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of ``gains``, already in rank order."""
    return sum(gain / math.log2(i + 1) for i, gain in enumerate(gains, start=1))


def _as_relevance(relevant: Relevance) -> dict[str, float]:
    """Coerce ids-or-grades into ``{doc_id: grade}``."""
    if isinstance(relevant, Mapping):
        return {str(doc): float(grade) for doc, grade in relevant.items()}
    return {str(doc): 1.0 for doc in relevant}


# --------------------------------------------------------------------------- #
# Metric specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _MetricSpec:
    """A parsed ``"<metric>@<k>"`` string."""

    label: str
    base: str
    k: int | None

    def score(self, ranked: Sequence[str], relevant: Relevance) -> float:
        if self.base == "recall":
            return recall_at_k(ranked, relevant, self.k or DEFAULT_K)
        if self.base == "precision":
            return precision_at_k(ranked, relevant, self.k or DEFAULT_K)
        if self.base == "hit_rate":
            return hit_rate_at_k(ranked, relevant, self.k or DEFAULT_K)
        if self.base == "ndcg":
            return ndcg_at_k(ranked, relevant, self.k or DEFAULT_K)
        if self.base == "mrr":
            return mrr(ranked, relevant, self.k)
        return average_precision(ranked, relevant, self.k)


def _parse_metric(spec: str) -> _MetricSpec:
    """Parse ``"ndcg@10"`` into a :class:`_MetricSpec`.

    Args:
        spec: A metric name, optionally suffixed with ``@k``.

    Returns:
        The parsed spec.

    Raises:
        ConfigurationError: If the name is unknown or ``k`` is not a positive
            integer.
    """
    text = str(spec).strip().lower()
    base, _, cutoff = text.partition("@")
    base = _ALIASES.get(base.strip(), base.strip())

    if base not in _CUTOFF_METRICS and base not in _FULL_METRICS:
        valid = ", ".join([f"{name}@k" for name in _CUTOFF_METRICS] + list(_FULL_METRICS))
        raise ConfigurationError(f"Unknown metric {spec!r}. Valid metrics are: {valid}.")

    k: int | None = None
    if cutoff:
        try:
            k = int(cutoff)
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid cutoff in metric {spec!r}: {cutoff!r} is not an integer."
            ) from exc
        if k <= 0:
            raise ConfigurationError(
                f"Invalid cutoff in metric {spec!r}: k must be positive."
            )
    elif base in _CUTOFF_METRICS:
        k = DEFAULT_K

    label = f"{base}@{k}" if k is not None else base
    return _MetricSpec(label=label, base=base, k=k)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EvalResult:
    """Aggregated metrics plus the per-query breakdown behind them.

    The aggregate is what you report; the breakdown is what you debug with,
    since a mean of 0.6 hides the difference between "every query is mediocre"
    and "most are perfect and three are broken".

    Args:
        metrics: Metric label -> mean score over the evaluated queries.
        per_query: Query id -> metric label -> score.
        name: Optional label, used by :func:`compare` to tag each variant.

    Example:
        >>> result = evaluate_engine(rag, dataset)     # doctest: +SKIP
        >>> print(result.summary())                    # doctest: +SKIP
    """

    metrics: dict[str, float] = field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    name: str = ""

    @property
    def queries(self) -> int:
        """How many queries contributed to the aggregate."""
        return len(self.per_query)

    def __getitem__(self, metric: str) -> float:
        return self.metrics[metric]

    def worst(self, metric: str, *, n: int = 5) -> list[tuple[str, float]]:
        """The ``n`` lowest-scoring queries for ``metric``, worst first.

        Args:
            metric: A metric label present in :attr:`metrics`.
            n: How many queries to return.

        Returns:
            ``(query_id, score)`` pairs.

        Raises:
            ConfigurationError: If ``metric`` was not evaluated.
        """
        if metric not in self.metrics:
            raise ConfigurationError(
                f"Metric {metric!r} was not evaluated. Available: "
                f"{', '.join(self.metrics) or 'none'}."
            )
        scores = [(qid, row.get(metric, 0.0)) for qid, row in self.per_query.items()]
        scores.sort(key=lambda pair: (pair[1], pair[0]))
        return scores[:n]

    def summary(self, *, per_query: bool = False) -> str:
        """Render an aligned text table of the results.

        Args:
            per_query: Also list every query's scores under the aggregate.

        Returns:
            A printable, newline-separated table.
        """
        labels = list(self.metrics)
        title = self.name or "results"
        header = f"{title}  ({self.queries} queries)"
        if not labels:
            return f"{header}\n(no metrics)"

        width = max(len(label) for label in [*labels, "metric"])
        lines = [
            header,
            f"{'metric'.ljust(width)}  score",
            f"{'-' * width}  ------",
        ]
        lines += [f"{label.ljust(width)}  {self.metrics[label]:.4f}" for label in labels]

        if per_query and self.per_query:
            qid_width = max(len(qid) for qid in [*self.per_query, "query"])
            cells = [max(len(label), 6) for label in labels]
            lines.append("")
            lines.append(
                "  ".join(
                    ["query".ljust(qid_width)]
                    + [label.rjust(w) for label, w in zip(labels, cells, strict=True)]
                )
            )
            lines.append("  ".join(["-" * qid_width] + ["-" * w for w in cells]))
            for qid, row in self.per_query.items():
                lines.append(
                    "  ".join(
                        [qid.ljust(qid_width)]
                        + [
                            f"{row.get(label, 0.0):.4f}".rjust(w)
                            for label, w in zip(labels, cells, strict=True)
                        ]
                    )
                )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        scores = " ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        tag = f"{self.name!r} " if self.name else ""
        return f"<EvalResult {tag}queries={self.queries} {scores}>"


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    qrels: Qrels,
    run: Run,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Score a run against ground truth, ``trec_eval`` style.

    Args:
        qrels: Ground truth, ``{query_id: {doc_id: relevance}}``.
        run: System output, ``{query_id: {doc_id: score}}``. Queries missing from
            the run are scored as if they returned nothing, so a retriever that
            silently drops a query is penalised rather than ignored.
        metrics: Metric specs such as ``"recall@5"``, ``"ndcg@10"``, ``"mrr"``,
            ``"map"``, ``"precision@3"``, ``"hit_rate@10"``.

    Returns:
        Metric label -> mean score over the queries that have at least one
        relevant document.

    Raises:
        ConfigurationError: If a metric name is not recognised.

    Example:
        >>> qrels = {"q1": {"a": 1}}
        >>> run = {"q1": {"b": 2.0, "a": 1.0}}
        >>> round(evaluate(qrels, run, metrics=["mrr"])["mrr"], 3)
        0.5
    """
    return evaluate_run(qrels, run, metrics=metrics).metrics


def evaluate_run(
    qrels: Qrels,
    run: Run,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    name: str = "",
) -> EvalResult:
    """Like :func:`evaluate`, but keeping the per-query breakdown.

    Args:
        qrels: Ground truth, ``{query_id: {doc_id: relevance}}``.
        run: System output, ``{query_id: {doc_id: score}}``.
        metrics: Metric specs to compute.
        name: Optional label for the result.

    Returns:
        An :class:`EvalResult`.

    Raises:
        ConfigurationError: If a metric name is not recognised.
    """
    specs = [_parse_metric(spec) for spec in metrics]
    labels = [spec.label for spec in specs]

    per_query: dict[str, dict[str, float]] = {}
    for qid, grades in qrels.items():
        relevant = _as_relevance(grades)
        if not any(grade > 0 for grade in relevant.values()):
            # No relevant document: every metric is undefined, so skip the query
            # rather than average a meaningless zero into the result.
            log.debug("skipping query %s: no relevant documents in qrels", qid)
            continue
        ranked = _ranking(run.get(qid) or {})
        per_query[qid] = {spec.label: spec.score(ranked, relevant) for spec in specs}

    if not per_query:
        return EvalResult(metrics=dict.fromkeys(labels, 0.0), per_query={}, name=name)

    aggregates = {
        label: sum(row[label] for row in per_query.values()) / len(per_query)
        for label in labels
    }
    return EvalResult(metrics=aggregates, per_query=per_query, name=name)


def _ranking(scores: Mapping[str, float]) -> list[str]:
    """Turn ``{doc_id: score}`` into a deterministic best-first id list."""
    return [
        doc
        for doc, _ in sorted(scores.items(), key=lambda pair: (-float(pair[1]), pair[0]))
    ]


# --------------------------------------------------------------------------- #
# Evaluating a live engine
# --------------------------------------------------------------------------- #


def evaluate_engine(
    rag: Any,
    dataset: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    top_k: int = 10,
    name: str = "",
    **search_kwargs: Any,
) -> EvalResult:
    """Run a labelled dataset through ``rag.search()`` and score the results.

    This is the function that answers "did raising ``chunk_size`` help?". Build
    the smallest dataset you can stand to label -- twenty questions is already
    enough to see a real regression -- and re-run it after every change.

    Each entry needs a ``"query"`` and a ``"relevant"`` list. Relevant items are
    matched against both :attr:`Hit.source <softrag.types.Hit.source>` and
    :attr:`Hit.id <softrag.types.Hit.id>`, so labelling by file path or URL
    (usually what a human has to hand) works just as well as labelling by chunk
    id. Several chunks of the same source collapse to one entry, at the rank of
    the best-placed chunk.

    Documents are scored by their retrieved rank rather than by ``hit.score``,
    so the evaluated ordering is exactly the ordering the engine returned, even
    when a reranker or fusion step produced ties.

    Args:
        rag: Anything with a ``search(query, *, top_k=..., ...)`` method.
        dataset: Entries of ``{"query": str, "relevant": list[str]}``. An
            optional ``"id"`` labels the query; a mapping in ``"relevant"``
            supplies graded relevance, e.g. ``{"policy.md": 3, "faq.md": 1}``.
        metrics: Metric specs to compute.
        top_k: How many hits to retrieve per query. Keep it at or above the
            largest ``@k`` you evaluate, or those metrics are capped by it.
        name: Optional label for the result.
        **search_kwargs: Forwarded to ``rag.search()`` -- ``mode``, ``where``,
            ``diversity``, ``rerank`` and so on.

    Returns:
        An :class:`EvalResult`.

    Raises:
        ConfigurationError: If an entry is missing ``query`` or ``relevant``, or
            if a metric name is not recognised.

    Example:
        >>> dataset = [{"query": "refund window?", "relevant": ["policy.md"]}]
        >>> result = evaluate_engine(rag, dataset)          # doctest: +SKIP
        >>> print(result.summary())                         # doctest: +SKIP
    """
    qrels: dict[str, dict[str, float]] = {}
    run: dict[str, dict[str, float]] = {}

    for position, entry in enumerate(dataset):
        query, relevant, qid = _read_entry(entry, position)
        qrels[qid] = relevant

        hits = rag.search(query, top_k=top_k, **search_kwargs)
        scores: dict[str, float] = {}
        for rank, hit in enumerate(hits):
            key = _hit_key(hit, relevant)
            # First occurrence wins: a source is worth its best chunk's rank.
            scores.setdefault(key, float(len(hits) - rank))
        run[qid] = scores

    return evaluate_run(qrels, run, metrics=metrics, name=name)


def _read_entry(
    entry: Mapping[str, Any], position: int
) -> tuple[str, dict[str, float], str]:
    """Validate one dataset entry into ``(query, relevance, query_id)``."""
    if not isinstance(entry, Mapping):
        raise ConfigurationError(
            f"Dataset entry {position} must be a mapping with 'query' and "
            f"'relevant' keys, got {type(entry).__name__}."
        )
    query = entry.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ConfigurationError(
            f"Dataset entry {position} is missing a non-empty 'query' string."
        )
    if "relevant" not in entry:
        raise ConfigurationError(
            f"Dataset entry {position} ({query!r}) is missing 'relevant': the "
            "source identifiers or chunk ids that should have been retrieved."
        )
    relevant = _as_relevance(entry["relevant"])
    qid = str(entry.get("id") or f"q{position + 1}")
    return query, relevant, qid


def _hit_key(hit: Any, relevant: Mapping[str, float]) -> str:
    """Pick the identifier a hit should be judged under.

    Datasets are labelled with whatever the human had -- a file path, a URL, or
    a chunk id -- so both are tried against the ground truth before falling back
    to the source.
    """
    source = getattr(hit, "source", "") or ""
    doc_id = str(getattr(hit, "id", ""))
    if source and source in relevant:
        return source
    if doc_id and doc_id in relevant:
        return doc_id
    return source or doc_id


# --------------------------------------------------------------------------- #
# Comparing configurations
# --------------------------------------------------------------------------- #


def compare(
    rag: Any,
    dataset: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, EvalResult]:
    """Evaluate the same dataset under several ``search()`` configurations.

    The point is a controlled experiment: one index, one dataset, one metric
    panel, and only the search kwargs changing between runs.

    Args:
        rag: Anything with a ``search()`` method.
        dataset: As for :func:`evaluate_engine`.
        variants: Variant name -> kwargs passed to ``rag.search()``, for example
            ``{"hybrid": {"mode": "hybrid"}, "vector-only": {"mode": "vector"}}``.
        **kwargs: Shared settings applied to every variant -- ``metrics``,
            ``top_k``, and any search kwarg the variants do not override.

    Returns:
        Variant name -> :class:`EvalResult`, in the order the variants were
        given.

    Example:
        >>> results = compare(rag, dataset, {           # doctest: +SKIP
        ...     "hybrid": {"mode": "hybrid"},
        ...     "vector-only": {"mode": "vector"},
        ... })
        >>> print(comparison_table(results))            # doctest: +SKIP
    """
    results: dict[str, EvalResult] = {}
    for variant, overrides in variants.items():
        merged: dict[str, Any] = {**kwargs, **dict(overrides)}
        merged["name"] = variant
        log.debug("evaluating variant %s with %s", variant, merged)
        results[variant] = evaluate_engine(rag, dataset, **merged)
    return results


def comparison_table(results: Mapping[str, EvalResult]) -> str:
    """Render several :class:`EvalResult` objects side by side.

    Args:
        results: Variant name -> result, typically straight from :func:`compare`.

    Returns:
        A printable table with one row per metric and one column per variant.
    """
    if not results:
        return "(no results)"

    labels: list[str] = []
    for result in results.values():
        labels += [label for label in result.metrics if label not in labels]
    if not labels:
        return "(no metrics)"

    variants = list(results)
    metric_width = max(len(label) for label in [*labels, "metric"])
    columns = [max(len(variant), 6) for variant in variants]

    lines = [
        "  ".join(
            ["metric".ljust(metric_width)]
            + [v.rjust(w) for v, w in zip(variants, columns, strict=True)]
        ),
        "  ".join(["-" * metric_width] + ["-" * w for w in columns]),
    ]
    for label in labels:
        cells = []
        for variant, width in zip(variants, columns, strict=True):
            score = results[variant].metrics.get(label)
            cells.append(("-" if score is None else f"{score:.4f}").rjust(width))
        lines.append("  ".join([label.ljust(metric_width), *cells]))
    return "\n".join(lines)
