"""Second-stage rerankers.

First-stage retrieval optimises for recall: it is cheap, it runs over the whole
index, and it is happy to hand back forty candidates knowing five of them are
good. A reranker spends real compute on that shortlist to decide which five.

Everything here satisfies the :class:`~softrag.types.Reranker` protocol -- a
single ``rerank(query, hits, *, top_k)`` method -- so any of these can be passed
to ``Rag(reranker=...)`` or to ``rag.search(rerank=...)``.

A reranker is an *optimisation*. None of the ones below may raise because a
model returned nonsense, an API was down or a reply failed to parse: they log a
warning and fall back to the order first-stage retrieval already produced. A
slightly worse ranking is always better than a failed query.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from .errors import ConfigurationError, MissingDependencyError
from .providers import adapt_chat_model
from .providers.local import CrossEncoderReranker
from .types import ChatModel, Hit

log = logging.getLogger("softrag.rerank")

__all__ = [
    "ChainReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "DedupeReranker",
    "LLMReranker",
    "ScoreFusionReranker",
]

DEFAULT_COHERE_MODEL = "rerank-v3.5"

DEFAULT_LLM_PROMPT = """\
You are ranking search results for relevance to a query.

Query: {query}

Documents:
{documents}

Return the document numbers ordered from most to least relevant to the query,
as a JSON array of integers, for example [3, 1, 2]. Include every number exactly
once. Answer with the array and nothing else."""

_INTEGER = re.compile(r"-?\d+")


# --------------------------------------------------------------------------- #
# LLM reranking
# --------------------------------------------------------------------------- #


class LLMReranker:
    """Rerank with a chat model asked to order the candidates.

    Also known as listwise reranking: instead of scoring each document on its
    own, the model sees a batch of candidates together and returns a permutation.
    It needs no extra model download and reuses the chat backend the engine
    already has, at the cost of one LLM call per batch.

    Candidates are processed in batches of ``batch_size``. Each hit is scored
    ``1 / (1 + rank_within_its_batch)``, so with a single batch the model's
    ordering is reproduced exactly, and with several batches the batch winners
    interleave ahead of the batch runners-up.

    Every failure mode -- a backend error, a reply that is not a permutation, a
    reply that is not parseable at all -- degrades to the input order rather than
    raising.

    Args:
        chat_model: Any chat backend. Adapted through
            :func:`~softrag.providers.adapt_chat_model`, so a LangChain model, a
            softrag provider or a bare callable all work.
        batch_size: Candidates shown to the model per call.
        prompt: Template with ``{query}`` and ``{documents}`` fields. Defaults to
            :data:`DEFAULT_LLM_PROMPT`, which asks for a JSON array of positions.

    Example:
        >>> reranker = LLMReranker(chat_model)             # doctest: +SKIP
        >>> hits = rag.search("refund policy", rerank=reranker)   # doctest: +SKIP
    """

    __slots__ = ("_chat", "batch_size", "prompt")

    def __init__(
        self,
        chat_model: ChatModel | Any,
        *,
        batch_size: int = 10,
        prompt: str | None = None,
    ) -> None:
        self._chat = adapt_chat_model(chat_model)
        self.batch_size = max(1, batch_size)
        self.prompt = prompt or DEFAULT_LLM_PROMPT

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> list[Hit]:
        """Order ``hits`` by asking the model, and return the best ``top_k``.

        Args:
            query: The search text the candidates should be judged against.
            hits: Candidates from first-stage retrieval.
            top_k: How many hits to keep.

        Returns:
            At most ``top_k`` hits, best first. On any failure, the first
            ``top_k`` of ``hits`` unchanged.
        """
        if not hits or top_k <= 0:
            return []

        scored: list[tuple[float, int, Hit]] = []
        for start in range(0, len(hits), self.batch_size):
            batch = list(hits[start : start + self.batch_size])
            order = self._order_batch(query, batch)
            for rank, position in enumerate(order):
                hit = batch[position]
                scored.append((1.0 / (1.0 + rank), start + position, hit))

        # Stable on the fused score, then on the original position, so ties keep
        # whatever first-stage retrieval decided.
        scored.sort(key=lambda item: (-item[0], item[1]))
        out: list[Hit] = []
        for score, _, hit in scored[:top_k]:
            hit.score = score
            out.append(hit)
        return out

    def _order_batch(self, query: str, batch: Sequence[Hit]) -> list[int]:
        """Return 0-based positions of ``batch``, best first."""
        identity = list(range(len(batch)))
        if len(batch) == 1:
            return identity

        documents = "\n".join(
            f"[{i}] {_preview(hit.text)}" for i, hit in enumerate(batch, start=1)
        )
        try:
            reply = self._chat.complete(
                self.prompt.format(query=query, documents=documents)
            )
        except Exception as exc:
            log.warning("LLM reranking call failed (%s); keeping the original order", exc)
            return identity

        parsed = _parse_ranking(reply, len(batch))
        if parsed is None:
            log.warning(
                "could not parse a ranking out of the model reply %r; "
                "keeping the original order",
                _preview(str(reply), 120),
            )
            return identity
        return parsed

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LLMReranker(batch_size={self.batch_size})"


def _parse_ranking(reply: Any, n: int) -> list[int] | None:
    """Pull a permutation of ``0..n-1`` out of whatever the model said.

    Three shapes are accepted, in order:

    * JSON -- an array of integers, or of objects carrying an ``index``/``id``
      field, possibly wrapped in a markdown code fence;
    * a bare list of numbers, ``3, 1, 2`` or ``3 1 2``;
    * numbered lines, one document reference per line.

    All three degrade to the same fallback: scan the text for integers in
    ``1..n``, keep the first occurrence of each, and append whatever the model
    forgot in its original order. That makes a truncated or partial reply usable
    instead of fatal.

    Args:
        reply: The raw model output.
        n: How many documents were shown.

    Returns:
        0-based positions, best first, or ``None`` if nothing usable was found.
    """
    text = reply if isinstance(reply, str) else str(reply)
    if not text.strip():
        return None

    numbers = _from_json(text)
    if numbers is None:
        numbers = [int(match.group()) for match in _INTEGER.finditer(text)]

    seen: set[int] = set()
    order: list[int] = []
    for value in numbers:
        if 1 <= value <= n and value not in seen:
            seen.add(value)
            order.append(value - 1)
    if not order:
        return None
    ranked = set(order)
    order.extend(i for i in range(n) if i not in ranked)
    return order


def _from_json(text: str) -> list[int] | None:
    """Parse a JSON array of ranks out of ``text``, tolerating code fences."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, list):
        return None

    numbers: list[int] = []
    for item in payload:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            numbers.append(item)
        elif isinstance(item, float) and item.is_integer():
            numbers.append(int(item))
        elif isinstance(item, dict):
            for key in ("index", "id", "document", "doc", "rank"):
                value = item.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    numbers.append(value)
                    break
    return numbers or None


# --------------------------------------------------------------------------- #
# Hosted reranking
# --------------------------------------------------------------------------- #


class CohereReranker:
    """Rerank with Cohere's hosted Rerank API.

    A managed cross-encoder: accurate, multilingual, and no model to download or
    GPU to own. The trade-off against
    :class:`~softrag.providers.local.CrossEncoderReranker` is the usual one --
    every query leaves the machine, which for a local-first library is a choice
    rather than a default.

    Args:
        model: Cohere rerank model name.
        api_key: Overrides the ``COHERE_API_KEY`` environment variable.
        base_url: Point at a compatible or proxied endpoint.
        max_candidates: Hard cap on documents sent per call, to bound cost.

    Raises:
        MissingDependencyError: If the ``cohere`` package is not installed.
        ConfigurationError: If no API key can be found.

    Example:
        >>> rag = Rag(reranker=CohereReranker())           # doctest: +SKIP
    """

    __slots__ = ("_client", "max_candidates", "model")

    def __init__(
        self,
        model: str = DEFAULT_COHERE_MODEL,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        max_candidates: int = 100,
    ) -> None:
        try:
            import cohere
        except ImportError as exc:
            raise MissingDependencyError(
                "cohere", extra="rerank", feature="Cohere reranking"
            ) from exc

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ConfigurationError(
                "No Cohere API key found. Set COHERE_API_KEY or pass api_key=..."
            )

        self.model = model
        self.max_candidates = max(1, max_candidates)
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = cohere.ClientV2(**kwargs)

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> list[Hit]:
        """Score every hit through the API and return the best ``top_k``.

        Args:
            query: The search text.
            hits: Candidates from first-stage retrieval.
            top_k: How many hits to keep.

        Returns:
            At most ``top_k`` hits, best first, with ``hit.score`` replaced by
            Cohere's relevance score. On an API failure, the first ``top_k`` of
            ``hits`` unchanged.
        """
        if not hits or top_k <= 0:
            return []

        candidates = list(hits[: self.max_candidates])
        try:
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=[hit.text for hit in candidates],
                top_n=min(top_k, len(candidates)),
            )
        except Exception as exc:
            log.warning("Cohere reranking failed (%s); keeping the original order", exc)
            return list(hits[:top_k])

        out: list[Hit] = []
        for result in getattr(response, "results", []):
            index = getattr(result, "index", None)
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            hit = candidates[index]
            hit.score = float(getattr(result, "relevance_score", hit.score))
            out.append(hit)
        if not out:
            log.warning("Cohere returned no usable results; keeping the original order")
            return list(hits[:top_k])
        return out[:top_k]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CohereReranker(model={self.model!r})"


# --------------------------------------------------------------------------- #
# Score fusion
# --------------------------------------------------------------------------- #


class ScoreFusionReranker:
    """Re-fuse hybrid results by normalised score instead of by rank.

    softrag fuses dense and BM25 results with Reciprocal Rank Fusion, and RRF
    stays the default for one reason: it has nothing to tune. It throws away the
    incomparable magnitudes -- unbounded negative BM25 scores against cosine
    distances in ``[0, 2]`` -- and keeps only the ordering the two retrievers
    agree on, which works acceptably on any corpus without a labelled set.

    The cost of that robustness is information. RRF cannot tell a document that
    barely won its list from one that dominated it. Normalised score fusion
    keeps that margin: min-max (or z-score) normalise each score list, then take
    ``alpha * dense + (1 - alpha) * sparse``. Given a labelled evaluation set to
    tune ``alpha`` on, this reliably beats RRF by a few points of nDCG on that
    corpus -- and, tuned on the wrong corpus, it can just as reliably lose to it.

    So: start with RRF, build a small qrels file, use :mod:`softrag.eval` to
    sweep ``alpha``, and adopt this only if the numbers say so.

    Reads :attr:`~softrag.types.Hit.vector_distance` and
    :attr:`~softrag.types.Hit.bm25`, both populated by the retriever in hybrid
    mode. Either may be ``None`` when only one retriever found the document:
    under ``"minmax"`` a missing side scores 0 (the worst observed value), under
    ``"zscore"`` it scores 0 (the mean), and when one side is missing everywhere
    -- keyword-only or vector-only search -- the other side is used alone.

    Args:
        alpha: Weight of the dense side, in ``[0, 1]``. 1 is vector-only, 0 is
            BM25-only, 0.5 is an even split.
        normalize: ``"minmax"``, ``"zscore"`` or ``"none"``.

    Raises:
        ConfigurationError: If ``alpha`` is outside ``[0, 1]`` or ``normalize``
            is not one of the three strategies.

    Example:
        >>> hits = rag.search("q", rerank=ScoreFusionReranker(alpha=0.7))  # doctest: +SKIP
    """

    __slots__ = ("alpha", "normalize")

    def __init__(self, *, alpha: float = 0.5, normalize: str = "minmax") -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ConfigurationError(f"alpha must be in [0, 1], got {alpha!r}")
        if normalize not in ("minmax", "zscore", "none"):
            raise ConfigurationError(
                f"normalize must be 'minmax', 'zscore' or 'none', got {normalize!r}"
            )
        self.alpha = float(alpha)
        self.normalize = normalize

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> list[Hit]:
        """Recombine the two score channels and return the best ``top_k``.

        Args:
            query: Unused; kept for the :class:`~softrag.types.Reranker`
                protocol, since fusion works purely on stored scores.
            hits: Candidates carrying ``vector_distance`` and/or ``bm25``.
            top_k: How many hits to keep.

        Returns:
            At most ``top_k`` hits, best first, with ``hit.score`` set to the
            fused score.
        """
        if not hits or top_k <= 0:
            return []

        # Both channels flipped so that larger is better: cosine distance in
        # [0, 2] becomes a similarity, and FTS5's negative BM25 becomes positive.
        dense = [
            None if hit.vector_distance is None else 1.0 - hit.vector_distance / 2.0
            for hit in hits
        ]
        sparse = [None if hit.bm25 is None else -hit.bm25 for hit in hits]

        has_dense = any(value is not None for value in dense)
        has_sparse = any(value is not None for value in sparse)
        if not has_dense and not has_sparse:
            log.warning(
                "no vector_distance or bm25 on any hit; keeping the original order"
            )
            return list(hits[:top_k])

        alpha = self.alpha
        if not has_sparse:
            alpha = 1.0
        elif not has_dense:
            alpha = 0.0

        dense_norm = self._scale(dense)
        sparse_norm = self._scale(sparse)

        scored = [
            (alpha * dense_norm[i] + (1.0 - alpha) * sparse_norm[i], i, hit)
            for i, hit in enumerate(hits)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        out: list[Hit] = []
        for score, _, hit in scored[:top_k]:
            hit.score = score
            out.append(hit)
        return out

    def _scale(self, values: Sequence[float | None]) -> list[float]:
        """Normalise one score channel, mapping missing entries to 0.0."""
        present = [value for value in values if value is not None]
        if not present:
            return [0.0] * len(values)

        if self.normalize == "none":
            return [0.0 if value is None else value for value in values]

        if self.normalize == "zscore":
            mean = sum(present) / len(present)
            variance = sum((value - mean) ** 2 for value in present) / len(present)
            spread = variance**0.5
            if spread == 0:
                return [0.0] * len(values)
            return [0.0 if v is None else (v - mean) / spread for v in values]

        lo, hi = min(present), max(present)
        span = hi - lo
        if span == 0:
            # Every document scored the same: no signal, so no preference.
            return [0.0 if value is None else 1.0 for value in values]
        return [0.0 if v is None else (v - lo) / span for v in values]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ScoreFusionReranker(alpha={self.alpha}, normalize={self.normalize!r})"


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


class ChainReranker:
    """Apply several rerankers one after another.

    Every stage but the last is given the whole list it received, so it reorders
    and filters without truncating; only the final stage applies ``top_k``. That
    way a cheap filter can run in front of an expensive scorer without starving
    it of candidates.

    The usual shape is dedupe, then score:
    :class:`DedupeReranker` collapses the same paragraph appearing in three
    files, and the cross-encoder then spends its budget on distinct documents.

    Args:
        *rerankers: Stages, applied left to right. Passing none makes this a
            no-op that simply truncates to ``top_k``.

    Example:
        >>> chain = ChainReranker(DedupeReranker(), CrossEncoderReranker())  # doctest: +SKIP
    """

    __slots__ = ("rerankers",)

    def __init__(self, *rerankers: Any) -> None:
        self.rerankers = tuple(rerankers)

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> list[Hit]:
        """Pipe ``hits`` through every stage in order.

        Args:
            query: The search text, handed to each stage unchanged.
            hits: Candidates from first-stage retrieval.
            top_k: How many hits the last stage should keep.

        Returns:
            At most ``top_k`` hits, best first.
        """
        if not hits or top_k <= 0:
            return []

        current = list(hits)
        last = len(self.rerankers) - 1
        for i, reranker in enumerate(self.rerankers):
            stage_k = top_k if i == last else len(current)
            current = list(reranker.rerank(query, current, top_k=stage_k))
            if not current:
                break
        return current[:top_k]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ", ".join(repr(r) for r in self.rerankers)
        return f"ChainReranker({inner})"


class DedupeReranker:
    """Drop near-duplicate hits, keeping the best-ranked copy of each.

    The same paragraph turns up in a README, a docs page and a changelog more
    often than anyone expects, and three copies of one answer waste three of the
    five context slots. Similarity is character-trigram Jaccard --
    the size of the intersection over the size of the union of their sets of
    3-character shingles -- which is pure
    stdlib, needs no embeddings, and catches reformatted or lightly edited
    copies that exact-hash deduplication misses.

    Order is preserved: hits are walked best-first and a hit is dropped only if
    it is too similar to one already kept.

    Args:
        threshold: Jaccard similarity in ``[0, 1]`` at or above which a hit is
            considered a duplicate. 0.9 is near-identical text; lower it towards
            0.7 to also collapse paraphrases.

    Raises:
        ConfigurationError: If ``threshold`` is outside ``[0, 1]``.

    Example:
        >>> hits = rag.search("q", rerank=DedupeReranker(threshold=0.85))  # doctest: +SKIP
    """

    __slots__ = ("threshold",)

    def __init__(self, *, threshold: float = 0.9) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(f"threshold must be in [0, 1], got {threshold!r}")
        self.threshold = float(threshold)

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> list[Hit]:
        """Filter out near-duplicates and return the first ``top_k`` survivors.

        Args:
            query: Unused; deduplication is query-independent.
            hits: Candidates, already ordered best-first.
            top_k: How many hits to keep.

        Returns:
            At most ``top_k`` hits in their input order, duplicates removed.
        """
        if not hits or top_k <= 0:
            return []

        kept: list[Hit] = []
        signatures: list[set[str]] = []
        for hit in hits:
            grams = _trigrams(hit.text)
            if any(_jaccard(grams, other) >= self.threshold for other in signatures):
                log.debug("dropping near-duplicate hit from %s", hit.source or "?")
                continue
            kept.append(hit)
            signatures.append(grams)
            if len(kept) >= top_k:
                break
        return kept

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DedupeReranker(threshold={self.threshold})"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _trigrams(text: str) -> set[str]:
    """Character 3-shingles of ``text``, whitespace-collapsed and lowercased."""
    normalised = " ".join(text.lower().split())
    if len(normalised) < 3:
        return {normalised} if normalised else set()
    return {normalised[i : i + 3] for i in range(len(normalised) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two shingle sets; 0.0 when either is empty."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / (len(a) + len(b) - intersection)


def _preview(text: str, limit: int = 300) -> str:
    """One-line, length-capped view of ``text`` for prompts and log messages."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."
