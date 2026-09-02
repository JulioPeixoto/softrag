"""Stopwords and the document-frequency cutoff.

Two mechanisms guard the same failure: a keyword query assembled out of words
that match everything. The fixed stopword list covers the small corpus, where
frequencies say nothing; the document-frequency cutoff covers the large one,
where they say everything. This module pins both, plus the deliberate fallback
between them -- when *every* term is too common the terms are kept rather than
the query being thrown away.
"""

from __future__ import annotations

import pytest

from softrag import stopwords as stopwords_module
from softrag.stopwords import ENGLISH, PORTUGUESE, STOPWORDS, is_stopword
from softrag.store import MAX_DOCUMENT_FREQUENCY, MIN_CORPUS_FOR_IDF, Store

DIM = 4


def vec(seed: float) -> list[float]:
    return [seed, 0.0, 0.0, 0.0]


@pytest.fixture
def store():
    s = Store(":memory:")
    try:
        yield s
    finally:
        s.close()


def seed(store: Store, source: str, texts) -> None:
    store.upsert_source(
        source, content_hash=source, characters=sum(len(t) for t in texts)
    )
    store.add_chunks(
        source,
        list(texts),
        [vec(float(i + 1)) for i in range(len(texts))],
        metadata=[{} for _ in texts],
    )


# --------------------------------------------------------------------------- #
# The word list
# --------------------------------------------------------------------------- #


def test_the_set_is_not_empty():
    assert len(STOPWORDS) > 100
    assert STOPWORDS == ENGLISH | PORTUGUESE


@pytest.mark.parametrize("token", ["the", "The", "THE", "ThE"])
def test_matching_is_case_insensitive(token):
    assert is_stopword(token)


@pytest.mark.parametrize(
    "token", ["a", "the", "and", "of", "to", "is", "was", "when", "which", "should"]
)
def test_english_stopwords_are_covered(token):
    assert is_stopword(token)


@pytest.mark.parametrize(
    "token", ["de", "que", "para", "com", "não", "você", "isso", "quando", "já", "são"]
)
def test_portuguese_stopwords_are_covered(token):
    assert is_stopword(token)


def test_accented_portuguese_words_are_matched_when_capitalised():
    assert is_stopword("Não")
    assert is_stopword("VOCÊ")


@pytest.mark.parametrize(
    "token",
    [
        "checkpoint",
        "refund",
        "mitochondria",
        "retrieval",
        "sqlite",
        "reembolso",
        "zqxwv7",
    ],
)
def test_domain_terms_are_not_stopwords(token):
    assert not is_stopword(token)


def test_the_documented_examples_hold():
    assert is_stopword("The") is True
    assert is_stopword("checkpoint") is False


def test_the_set_is_replaceable():
    """The module documents reassignment as the way to support another language."""
    original = stopwords_module.STOPWORDS
    try:
        stopwords_module.STOPWORDS = frozenset({"widget"})
        assert is_stopword("widget")
        assert not is_stopword("the")
    finally:
        stopwords_module.STOPWORDS = original
    assert is_stopword("the")


# --------------------------------------------------------------------------- #
# Stopwords through the store
# --------------------------------------------------------------------------- #


def test_a_query_of_only_stopwords_produces_no_match(store):
    seed(store, "doc", ["The refund policy allows returns within thirty days."])
    assert store.build_match("what is it that we can do") == ""


def test_a_stopword_only_query_contributes_no_keyword_hits(store):
    seed(store, "doc", ["The refund policy allows returns within thirty days."])
    assert store.search_keyword("what is it that we can do", k=10) == []


def test_a_query_with_no_words_at_all_produces_no_match(store):
    seed(store, "doc", ["some indexed body"])
    assert store.build_match("!!! ??? ...") == ""


def test_stopwords_are_stripped_and_content_terms_kept(store):
    seed(store, "doc", ["The refund policy allows returns within thirty days."])
    assert store.build_match("what is the refund policy") == '"refund" OR "policy"'


def test_content_terms_are_deduplicated_case_insensitively(store):
    seed(store, "doc", ["refund policy"])
    assert store.build_match("Refund refund REFUND policy") == '"Refund" OR "policy"'


def test_a_stopword_heavy_query_still_finds_the_right_document(store):
    seed(store, "handbook", ["The refund policy allows returns within thirty days."])
    seed(store, "biology", ["Mitochondria are the powerhouse of the cell."])

    hits = store.search_keyword("what is the refund policy that we have", k=10)

    assert [doc_id for doc_id, _ in hits]
    fetched = store.fetch([doc_id for doc_id, _ in hits])
    assert {hit.source for hit in fetched.values()} == {"handbook"}


# --------------------------------------------------------------------------- #
# The document-frequency cutoff
# --------------------------------------------------------------------------- #

#: Comfortably above MIN_CORPUS_FOR_IDF so frequencies are actually consulted.
CORPUS_SIZE = 12


def seed_frequency_corpus(store: Store) -> None:
    """Every chunk contains ``common``; exactly one also contains ``zqxwv7``."""
    texts = [f"common filler chunk number {i}" for i in range(CORPUS_SIZE)]
    texts[0] = "common filler chunk zqxwv7 number 0"
    seed(store, "corpus", texts)


def test_below_the_minimum_corpus_every_term_is_kept(store):
    seed(store, "small", [f"common chunk {i}" for i in range(MIN_CORPUS_FOR_IDF - 1)])
    assert store.count() < MIN_CORPUS_FOR_IDF
    assert store.build_match("common") == '"common"'


def test_a_term_in_every_chunk_is_dropped(store):
    seed_frequency_corpus(store)
    assert store.count() == CORPUS_SIZE

    cutoff = max(1, int(CORPUS_SIZE * MAX_DOCUMENT_FREQUENCY))
    assert store._document_frequency("common", CORPUS_SIZE) > cutoff
    assert store._document_frequency("zqxwv7", CORPUS_SIZE) <= cutoff

    assert store.build_match("common zqxwv7") == '"zqxwv7"'


def test_the_dropped_term_no_longer_drags_in_the_whole_corpus(store):
    seed_frequency_corpus(store)
    hits = store.search_keyword("common zqxwv7", k=50)
    assert len(hits) == 1


def test_when_every_term_is_too_common_they_are_kept_anyway(store):
    # Deliberate: the user searched for something real, it just fails to
    # separate the corpus. Answering with it beats answering with nothing.
    seed_frequency_corpus(store)

    assert store.build_match("common filler") == '"common" OR "filler"'
    assert store.search_keyword("common filler", k=50)


def test_the_fallback_never_resurrects_a_stopword(store):
    seed_frequency_corpus(store)
    match = store.build_match("the common filler of it")
    assert '"the"' not in match
    assert '"of"' not in match
    assert match == '"common" OR "filler"'


# --------------------------------------------------------------------------- #
# The document-frequency cache
# --------------------------------------------------------------------------- #


def test_frequencies_are_memoised_within_one_corpus_generation(store):
    seed_frequency_corpus(store)
    store.build_match("zqxwv7")
    assert store._df_generation == CORPUS_SIZE
    assert "zqxwv7" in store._df_cache

    # Poisoning the cache proves the second call reads it rather than the index.
    store._df_cache["zqxwv7"] = CORPUS_SIZE
    assert store.build_match("zqxwv7 common") == '"zqxwv7" OR "common"'


def test_the_cache_is_invalidated_when_the_corpus_grows(store):
    seed(store, "first", ["alpha marker chunk", "alpha marker chunk two", "beta chunk"])
    seed(store, "filler", [f"filler chunk {i}" for i in range(7)])
    assert store.count() == 10
    # alpha is in 2 of 10, beta in 1 of 10: both below the cutoff of 3.
    assert store.build_match("alpha beta") == '"alpha" OR "beta"'
    assert store._df_cache["alpha"] == 2

    # Ten more chunks, every one of them mentioning alpha.
    seed(store, "second", [f"alpha again chunk {i}" for i in range(10)])
    assert store.count() == 20

    # A stale cache would still see alpha at 2 and keep it; a fresh count sees
    # 12 of 20, far above the cutoff of 7.
    assert store.build_match("alpha beta") == '"beta"'
    assert store._df_generation == 20
    assert store._df_cache["alpha"] == 12


def test_the_cache_is_invalidated_when_the_corpus_shrinks(store):
    seed(store, "base", ["alpha marker chunk", "alpha marker chunk two", "beta chunk"])
    seed(store, "filler", [f"filler chunk {i}" for i in range(7)])
    seed(store, "extra", [f"alpha again chunk {i}" for i in range(10)])
    assert store.count() == 20
    assert store.build_match("alpha beta") == '"beta"'
    assert store._df_cache["alpha"] == 12

    store.delete_source("extra")

    # Back to 2 of 10, comfortably under the cutoff of 3, so alpha returns.
    assert store.count() == 10
    assert store.build_match("alpha beta") == '"alpha" OR "beta"'
    assert store._df_generation == 10
    assert store._df_cache["alpha"] == 2


def test_an_unknown_term_has_a_frequency_of_zero(store):
    seed_frequency_corpus(store)
    assert store._document_frequency("nonexistentterm", CORPUS_SIZE) == 0
    # A term matching nothing carries no signal either, so it is dropped, and
    # the fallback then keeps it because nothing else survived.
    assert store.build_match("nonexistentterm") == '"nonexistentterm"'


# --------------------------------------------------------------------------- #
# Through the engine
# --------------------------------------------------------------------------- #


def test_a_stopword_only_query_falls_to_the_dense_side(corpus):
    assert corpus.search("what is it that we can do", mode="keyword") == []
    # Hybrid still answers, because the vector half is unaffected.
    assert corpus.search("what is it that we can do", mode="hybrid")
