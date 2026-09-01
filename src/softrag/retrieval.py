"""Retrieval: fusion, diversification and context expansion.

The default strategy is hybrid search -- dense vector similarity for meaning,
BM25 for exact terms -- combined with Reciprocal Rank Fusion.

Why rank fusion and not score fusion? BM25 scores are unbounded, negative and
corpus-dependent; cosine distances live in ``[0, 2]``. Adding or comparing them
directly, as naive hybrid implementations do, lets whichever scale happens to be
larger dominate the ranking for reasons unrelated to relevance. RRF throws the
magnitudes away and keeps only the ordering, which is the part both retrievers
actually agree on.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .store import Store
from .types import Hit, Where

log = logging.getLogger("softrag.retrieval")

__all__ = [
    "SearchMode",
    "RetrievalConfig",
    "Retriever",
    "reciprocal_rank_fusion",
    "maximal_marginal_relevance",
]

SearchMode = str  # "hybrid" | "vector" | "keyword"

#: The constant from the original RRF paper (Cormack et al., 2009). It damps the
#: influence of top ranks just enough that one retriever cannot veto the other.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    weights: Optional[Sequence[float]] = None,
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[int, float]]:
    """Fuse several ranked id lists into one.

    Each list contributes ``weight / (k + rank)`` to every id it contains, with
    ``rank`` counted from 1. Ids appearing in more than one list therefore rise
    above ids that only one retriever liked.

    Args:
        ranked_lists: The lists to fuse, each already ordered best-first.
        weights: Relative influence of each list. Defaults to equal weighting.
        k: Rank damping constant.

    Returns:
        ``(id, fused_score)`` pairs, best first.

    Example:
        >>> reciprocal_rank_fusion([[1, 2, 3], [3, 1]])[0][0]
        1
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"Got {len(ranked_lists)} ranked lists but {len(weights)} weights."
        )

    scores: Dict[int, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        if weight == 0:
            continue
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def maximal_marginal_relevance(
    query_vector: Sequence[float],
    candidates: Sequence[Tuple[int, Sequence[float]]],
    *,
    top_k: int,
    diversity: float = 0.3,
) -> List[int]:
    """Pick ``top_k`` ids that are relevant *and* different from each other.

    Greedy MMR: repeatedly take the candidate maximising
    ``(1 - diversity) * sim(query, c) - diversity * max sim(c, already_picked)``.
    This is the fix for the classic failure where the top five hits are five
    near-identical copies of the same paragraph.

    Args:
        query_vector: The embedded query.
        candidates: ``(id, vector)`` pairs to choose from.
        top_k: How many to select.
        diversity: 0 keeps pure relevance ordering, 1 maximises dissimilarity.

    Returns:
        The selected ids, in selection order.
    """
    if not candidates or top_k <= 0:
        return []
    diversity = min(max(diversity, 0.0), 1.0)

    query_norm = _normalize(query_vector)
    pool = [(doc_id, _normalize(vec)) for doc_id, vec in candidates]
    relevance = {doc_id: _dot(query_norm, vec) for doc_id, vec in pool}

    selected: List[int] = []
    remaining = dict(pool)
    # Running maximum similarity to anything already selected. Keeping it
    # incrementally means each round compares candidates against only the one
    # newly selected vector, which turns the naive O(n * k^2) into O(n * k).
    redundancy: Dict[int, float] = {doc_id: 0.0 for doc_id, _ in pool}

    while remaining and len(selected) < top_k:
        best_id: Optional[int] = None
        best_score = -math.inf
        for doc_id in remaining:
            score = (1 - diversity) * relevance[doc_id] - diversity * redundancy[doc_id]
            if score > best_score:
                best_score, best_id = score, doc_id
        if best_id is None:
            break
        selected.append(best_id)
        chosen = remaining.pop(best_id)
        for doc_id, vec in remaining.items():
            similarity = _dot(vec, chosen)
            if similarity > redundancy[doc_id]:
                redundancy[doc_id] = similarity

    return selected


def _normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return list(vec)
    return [v / norm for v in vec]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(slots=True)
class RetrievalConfig:
    """Knobs for a search.

    Args:
        mode: ``"hybrid"``, ``"vector"`` or ``"keyword"``.
        top_k: How many hits to return.
        candidates: How many candidates each retriever contributes before
            fusion. More candidates cost little and help fusion a lot; defaults
            to ``max(4 * top_k, 20)`` when left as ``None``.
        vector_weight: RRF weight of the dense list.
        keyword_weight: RRF weight of the BM25 list.
        rrf_k: RRF rank damping constant.
        diversity: MMR diversity in ``[0, 1]``; 0 disables MMR entirely.
        expand_context: Include this many neighbouring chunks around each hit.
    """

    mode: SearchMode = "hybrid"
    top_k: int = 5
    candidates: Optional[int] = None
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    rrf_k: int = DEFAULT_RRF_K
    diversity: float = 0.0
    expand_context: int = 0

    def resolved_candidates(self) -> int:
        if self.candidates is not None:
            return max(self.candidates, self.top_k)
        return max(4 * self.top_k, 20)


class Retriever:
    """Runs searches against a :class:`~softrag.store.Store`."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def search(
        self,
        query: str,
        query_vector: Optional[Sequence[float]],
        *,
        config: RetrievalConfig,
        where: Optional[Where] = None,
        source: Optional[str] = None,
    ) -> List[Hit]:
        """Retrieve the most relevant chunks for ``query``.

        Args:
            query: The raw query text, used for BM25.
            query_vector: The embedded query, used for dense search. ``None``
                forces keyword-only mode.
            config: Search parameters.
            where: Optional metadata filter.
            source: Optional exact source restriction.

        Returns:
            Hits ordered best-first, at most ``config.top_k`` of them (plus any
            neighbours added by ``expand_context``).
        """
        mode = config.mode
        if query_vector is None and mode != "keyword":
            log.debug("no query vector available, falling back to keyword search")
            mode = "keyword"

        n = config.resolved_candidates()
        vector_results: List[Tuple[int, float]] = []
        keyword_results: List[Tuple[int, float]] = []

        if mode in ("hybrid", "vector") and query_vector is not None:
            vector_results = self.store.search_vector(
                query_vector, k=n, where=where, source=source
            )
        if mode in ("hybrid", "keyword"):
            keyword_results = self.store.search_keyword(
                query, k=n, where=where, source=source
            )

        if mode == "vector":
            ordered = [(doc_id, 1.0 - dist / 2.0) for doc_id, dist in vector_results]
        elif mode == "keyword":
            ordered = _bm25_to_scores(keyword_results)
        else:
            fused = reciprocal_rank_fusion(
                [[i for i, _ in vector_results], [i for i, _ in keyword_results]],
                weights=[config.vector_weight, config.keyword_weight],
                k=config.rrf_k,
            )
            ordered = fused

        if not ordered:
            return []

        if config.diversity > 0 and query_vector is not None:
            ordered = self._diversify(query_vector, ordered, config)

        ordered = ordered[: config.top_k]
        hits = self._materialise(
            ordered, vector_results, keyword_results
        )
        if config.expand_context > 0:
            hits = self._expand(hits, radius=config.expand_context)
        return hits

    # -- internals ---------------------------------------------------------- #

    def _diversify(
        self,
        query_vector: Sequence[float],
        ordered: Sequence[Tuple[int, float]],
        config: RetrievalConfig,
    ) -> List[Tuple[int, float]]:
        scores = dict(ordered)
        vectors = self._load_vectors([doc_id for doc_id, _ in ordered])
        if not vectors:
            return list(ordered)
        picked = maximal_marginal_relevance(
            query_vector,
            [(doc_id, vec) for doc_id, vec in vectors.items()],
            top_k=config.top_k,
            diversity=config.diversity,
        )
        return [(doc_id, scores.get(doc_id, 0.0)) for doc_id in picked]

    def _load_vectors(self, ids: Sequence[int]) -> Dict[int, List[float]]:
        try:
            return self.store.vectors_for(ids)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not load vectors for MMR: %s", exc)
            return {}

    def _materialise(
        self,
        ordered: Sequence[Tuple[int, float]],
        vector_results: Sequence[Tuple[int, float]],
        keyword_results: Sequence[Tuple[int, float]],
    ) -> List[Hit]:
        distances = dict(vector_results)
        bm25 = dict(keyword_results)
        vector_ranks = {doc_id: i for i, (doc_id, _) in enumerate(vector_results, 1)}
        keyword_ranks = {doc_id: i for i, (doc_id, _) in enumerate(keyword_results, 1)}

        loaded = self.store.fetch([doc_id for doc_id, _ in ordered])
        hits: List[Hit] = []
        for doc_id, score in ordered:
            hit = loaded.get(doc_id)
            if hit is None:
                continue
            hit.score = score
            hit.vector_distance = distances.get(doc_id)
            hit.bm25 = bm25.get(doc_id)
            ranks = {}
            if doc_id in vector_ranks:
                ranks["vector"] = vector_ranks[doc_id]
            if doc_id in keyword_ranks:
                ranks["keyword"] = keyword_ranks[doc_id]
            hit.ranks = ranks
            hits.append(hit)
        return hits

    def _expand(self, hits: Sequence[Hit], *, radius: int) -> List[Hit]:
        """Widen each hit with its neighbouring chunks, without duplicating text."""
        seen: set[int] = set()
        expanded: List[Hit] = []
        for hit in hits:
            if hit.id in seen:
                continue
            window = self.store.neighbors(hit.source, hit.index, radius=radius)
            if not window:
                seen.add(hit.id)
                expanded.append(hit)
                continue
            merged = "\n".join(
                chunk.text for chunk in window if chunk.id not in seen or chunk.id == hit.id
            )
            for chunk in window:
                seen.add(chunk.id)
            expanded.append(
                Hit(
                    id=hit.id,
                    text=merged or hit.text,
                    score=hit.score,
                    source=hit.source,
                    index=hit.index,
                    metadata=hit.metadata,
                    vector_distance=hit.vector_distance,
                    bm25=hit.bm25,
                    ranks=hit.ranks,
                )
            )
        return expanded


def _bm25_to_scores(results: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Map FTS5's negative BM25 scores onto a friendlier 0..1 range.

    Only used for keyword-only mode, where there is nothing to fuse with; hybrid
    mode ignores magnitudes entirely.
    """
    if not results:
        return []
    raw = [-score for _, score in results]  # flip so larger is better
    lo, hi = min(raw), max(raw)
    span = hi - lo
    if span <= 0:
        return [(doc_id, 1.0) for doc_id, _ in results]
    return [
        (doc_id, (-score - lo) / span) for (doc_id, score) in results
    ]
