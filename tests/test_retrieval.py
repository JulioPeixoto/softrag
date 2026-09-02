"""Retrieval: rank fusion, MMR diversification, and the three search modes.

The fusion and MMR tests are pure functions with hand-checkable arithmetic --
that is the point, since these two are where a hybrid retriever silently goes
wrong. The engine-level tests then prove the wiring: that hybrid really does
find documents neither retriever finds alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from conftest import make_rag
from softrag import RetrievalConfig, maximal_marginal_relevance, reciprocal_rank_fusion
from softrag.retrieval import DEFAULT_RRF_K, _bm25_to_scores

# --------------------------------------------------------------------------- #
# Reciprocal rank fusion
# --------------------------------------------------------------------------- #


def test_the_documented_example_is_arithmetically_right():
    # id 1: 1/61 + 1/62, id 2: 1/62, id 3: 1/63 + 1/61.
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])
    scores = dict(fused)

    assert [doc_id for doc_id, _ in fused] == [1, 3, 2]
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)


def test_an_item_in_both_lists_outranks_an_item_in_one():
    # 7 is second in both lists; 1 is first in one list only.
    fused = reciprocal_rank_fusion([[1, 7], [9, 7]])
    assert fused[0][0] == 7


def test_weights_shift_the_ordering():
    lists = [[1, 2], [2, 1]]
    assert reciprocal_rank_fusion(lists, weights=[1.0, 5.0])[0][0] == 2
    assert reciprocal_rank_fusion(lists, weights=[5.0, 1.0])[0][0] == 1


def test_a_zero_weight_excludes_a_list_entirely():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]], weights=[1.0, 0.0])
    assert [doc_id for doc_id, _ in fused] == [1, 2]


def test_mismatched_weights_are_rejected():
    with pytest.raises(ValueError) as excinfo:
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])
    assert "2 ranked lists but 1 weights" in str(excinfo.value)


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_ties_break_deterministically_on_the_id():
    # Both ids sit at rank 1 in one list and rank 2 in the other, so the scores
    # are identical and only the id can order them.
    fused = reciprocal_rank_fusion([[9, 4], [4, 9]])
    assert [doc_id for doc_id, _ in fused] == [4, 9]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_the_damping_constant_is_honoured():
    small = dict(reciprocal_rank_fusion([[1]], k=1))
    default = dict(reciprocal_rank_fusion([[1]]))
    assert small[1] == pytest.approx(1 / 2)
    assert default[1] == pytest.approx(1 / (DEFAULT_RRF_K + 1))


# --------------------------------------------------------------------------- #
# Maximal marginal relevance
# --------------------------------------------------------------------------- #


def unit(*values: float) -> list[float]:
    return list(values)


#: Three near-identical vectors plus one pointing elsewhere.
NEAR_DUPLICATES = [
    (1, unit(1.0, 0.0, 0.0)),
    (2, unit(0.99, 0.01, 0.0)),
    (3, unit(0.98, 0.02, 0.0)),
    (4, unit(0.0, 1.0, 0.0)),
]


def test_zero_diversity_preserves_relevance_order():
    query = unit(1.0, 0.0, 0.0)
    candidates = [
        (4, unit(0.0, 1.0, 0.0)),
        (1, unit(1.0, 0.0, 0.0)),
        (2, unit(0.7, 0.7, 0.0)),
    ]

    picked = maximal_marginal_relevance(query, candidates, top_k=3, diversity=0.0)

    assert picked == [1, 2, 4]


def test_high_diversity_picks_the_dissimilar_candidate_second():
    query = unit(1.0, 0.0, 0.0)

    picked = maximal_marginal_relevance(query, NEAR_DUPLICATES, top_k=2, diversity=0.9)

    assert picked[0] == 1, "the most relevant candidate still comes first"
    assert picked[1] == 4, "the near-duplicates must be passed over for the outlier"


def test_low_diversity_keeps_the_near_duplicates():
    query = unit(1.0, 0.0, 0.0)
    picked = maximal_marginal_relevance(query, NEAR_DUPLICATES, top_k=2, diversity=0.05)
    assert picked == [1, 2]


def test_top_k_larger_than_the_pool_returns_the_whole_pool():
    picked = maximal_marginal_relevance(
        unit(1.0, 0.0, 0.0), NEAR_DUPLICATES, top_k=99, diversity=0.5
    )
    assert sorted(picked) == [1, 2, 3, 4]


def test_no_candidates_selects_nothing():
    assert maximal_marginal_relevance(unit(1.0, 0.0), [], top_k=5) == []


def test_a_non_positive_top_k_selects_nothing():
    assert maximal_marginal_relevance(unit(1.0, 0.0), NEAR_DUPLICATES, top_k=0) == []


def test_a_zero_vector_does_not_crash():
    candidates = [(1, unit(0.0, 0.0, 0.0)), (2, unit(1.0, 0.0, 0.0))]
    picked = maximal_marginal_relevance(
        unit(1.0, 0.0, 0.0), candidates, top_k=2, diversity=0.5
    )
    assert sorted(picked) == [1, 2]


def test_a_zero_query_vector_does_not_crash():
    picked = maximal_marginal_relevance(unit(0.0, 0.0, 0.0), NEAR_DUPLICATES, top_k=2)
    assert len(picked) == 2


def test_diversity_is_clamped_into_range():
    query = unit(1.0, 0.0, 0.0)
    assert maximal_marginal_relevance(query, NEAR_DUPLICATES, top_k=1, diversity=5.0)
    assert maximal_marginal_relevance(query, NEAR_DUPLICATES, top_k=1, diversity=-5.0)


# --------------------------------------------------------------------------- #
# RetrievalConfig
# --------------------------------------------------------------------------- #


def test_candidates_default_to_four_times_top_k_with_a_floor():
    assert RetrievalConfig(top_k=3).resolved_candidates() == 20
    assert RetrievalConfig(top_k=10).resolved_candidates() == 40


def test_an_explicit_candidate_count_is_never_below_top_k():
    assert RetrievalConfig(top_k=5, candidates=50).resolved_candidates() == 50
    assert RetrievalConfig(top_k=5, candidates=2).resolved_candidates() == 5


# --------------------------------------------------------------------------- #
# BM25 score normalisation
# --------------------------------------------------------------------------- #


def test_bm25_scores_are_mapped_onto_zero_to_one():
    scores = _bm25_to_scores([(1, -8.0), (2, -4.0), (3, -2.0)])
    assert scores[0] == (1, 1.0)
    assert scores[-1] == (3, 0.0)
    assert all(0.0 <= s <= 1.0 for _, s in scores)


def test_identical_bm25_scores_all_map_to_one():
    assert _bm25_to_scores([(1, -3.0), (2, -3.0)]) == [(1, 1.0), (2, 1.0)]


def test_no_bm25_results_map_to_nothing():
    assert _bm25_to_scores([]) == []


# --------------------------------------------------------------------------- #
# Engine-level retrieval
# --------------------------------------------------------------------------- #


class SemanticEmbedder:
    """A fake embedder with real synonymy, so vector search can beat keywords.

    Every text is mapped onto a one-hot vector over a handful of *concepts*;
    words that mean the same thing share a concept. That makes it possible to
    write a query which vector search can answer and BM25 cannot -- and, by
    choosing a concept the corpus does not share, the other way round too.
    """

    CONCEPTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "vehicle": ("car", "cars", "sedan", "automobile", "vehicle"),
        "cooking": ("braise", "ribs", "stock", "simmer", "recipe"),
        "release": ("version", "build", "changelog", "release"),
        "biology": ("mitochondria", "cell", "atp"),
    }
    dimensions = 5

    def _concept(self, text: str) -> int:
        tokens = {t.strip(".,").lower() for t in text.split()}
        for index, (_, words) in enumerate(self.CONCEPTS.items()):
            if tokens & set(words):
                return index
        return len(self.CONCEPTS)  # the "no known concept" axis

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        vector[self._concept(text)] = 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


#: Deliberately arranged so "automobile" is a vector-only hit and "zqxwv7" is a
#: keyword-only one.
SEMANTIC_CORPUS = [
    ("garage", "The sedan is parked in the garage overnight."),
    ("kitchen", "Braise the ribs slowly in stock."),
    ("notes", "Release build zqxwv7 appears in the audit trail."),
    ("lab", "Mitochondria power the cell."),
    # Occupies the catch-all axis, so a query with no known concept lands here
    # rather than on whichever document happens to sort first.
    ("misc", "Assorted trivia without any particular subject."),
]


@pytest.fixture
def semantic_rag():
    engine = make_rag(embed_model=SemanticEmbedder())
    for name, text in SEMANTIC_CORPUS:
        assert engine.add_text(text, name=name).ok
    try:
        yield engine
    finally:
        engine.close()


def test_every_mode_returns_something_sensible(corpus):
    for mode in ("hybrid", "vector", "keyword"):
        hits = corpus.search("refund policy", mode=mode)
        assert hits, mode
        assert hits[0].source == "handbook", mode


def test_vector_mode_reports_distances_and_keyword_mode_reports_bm25(corpus):
    vector_hit = corpus.search("refund policy", mode="vector")[0]
    keyword_hit = corpus.search("refund policy", mode="keyword")[0]

    assert vector_hit.vector_distance is not None
    assert vector_hit.ranks.get("vector") == 1
    assert keyword_hit.bm25 is not None
    assert keyword_hit.ranks.get("keyword") == 1


def test_vector_search_finds_what_keywords_cannot(semantic_rag):
    # "automobile" appears in no document, so BM25 has nothing to match.
    assert semantic_rag.search("automobile", mode="keyword", top_k=1) == []
    assert semantic_rag.search("automobile", mode="vector", top_k=1)[0].source == "garage"


def test_keyword_search_finds_what_vectors_cannot(semantic_rag):
    # "zqxwv7" carries no concept, so the query vector points at the catch-all
    # axis that no document occupies.
    assert semantic_rag.search("zqxwv7", mode="keyword", top_k=1)[0].source == "notes"
    assert semantic_rag.search("zqxwv7", mode="vector", top_k=1)[0].source == "misc"


def test_hybrid_finds_both(semantic_rag):
    sources = {hit.source for hit in semantic_rag.search("automobile zqxwv7", top_k=4)}
    assert {"garage", "notes"} <= sources


def test_source_restricts_the_result_set(corpus):
    hits = corpus.search("policy refund cell", source="biology", top_k=5)
    assert hits
    assert {hit.source for hit in hits} == {"biology"}


def test_where_restricts_the_result_set(corpus):
    hits = corpus.search("refund policy version cell", where={"year": 2019}, top_k=5)
    assert hits
    assert {hit.source for hit in hits} == {"biology"}


def test_a_filter_matching_nothing_returns_nothing(corpus):
    assert corpus.search("refund", where={"year": 1900}) == []


def test_searching_an_empty_index_returns_nothing(rag):
    for mode in ("hybrid", "vector", "keyword"):
        assert rag.search("anything at all", mode=mode) == []


def test_expand_context_attaches_neighbouring_chunks(make_engine):
    engine = make_engine(chunk_size=40, chunk_overlap=0)
    engine.add_text(
        "Alpha opening statement here.\n\n"
        "Beta the zqxwv7 marker sits here.\n\n"
        "Gamma closing statement here.",
        name="doc",
    )
    assert (
        len({row[0] for row in engine.store.db.execute("SELECT id FROM documents")}) >= 3
    )

    narrow = engine.search("zqxwv7", top_k=1)[0]
    wide = engine.search("zqxwv7", top_k=1, expand_context=1)[0]

    assert "zqxwv7" in narrow.text
    assert len(wide.text) > len(narrow.text)
    assert "Alpha" in wide.text and "Gamma" in wide.text
    assert wide.id == narrow.id


def test_expand_context_does_not_duplicate_shared_neighbours(make_engine):
    engine = make_engine(chunk_size=40, chunk_overlap=0)
    engine.add_text(
        "Alpha marker one here.\n\nBeta marker two here.\n\nGamma marker three here.",
        name="doc",
    )
    hits = engine.search("marker", top_k=3, expand_context=1)
    joined = "".join(hit.text for hit in hits)
    assert joined.count("Alpha") == 1
    assert joined.count("Beta") == 1
    assert joined.count("Gamma") == 1


def test_diversity_still_returns_top_k(corpus):
    hits = corpus.search("refund policy version", top_k=3, diversity=0.7)
    assert 0 < len(hits) <= 3
    assert len({hit.id for hit in hits}) == len(hits)


def test_candidates_can_be_widened_per_call(corpus):
    assert corpus.search("refund", candidates=50, top_k=2)


def test_weights_are_passed_through_to_fusion(make_engine):
    engine = make_engine(vector_weight=0.0, keyword_weight=1.0)
    engine.add_text("Reference identifier zqxwv7 lives here.", name="notes")
    engine.add_text("An unrelated paragraph about cooking.", name="kitchen")

    hits = engine.search("zqxwv7")

    assert [hit.source for hit in hits] == ["notes"]
