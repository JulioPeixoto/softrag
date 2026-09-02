"""Full-text search: query escaping, BM25 ordering and index synchronisation.

This is the regression-critical module. Two classes of bug live here:

* feeding raw user text to ``MATCH``, which makes FTS5 raise ``fts5: syntax
  error`` on ordinary questions containing ``NOT``, ``OR``, ``*`` or a quote;
* an external-content FTS index that drifts out of sync with its content table,
  which makes *every later query* fail with ``fts5: missing row N from content
  table`` -- long after the write that caused it.
"""

from __future__ import annotations

import sqlite3

import pytest

from softrag.store import Store, escape_fts_query

DIM = 4


def vec(*values: float) -> list:
    padded = list(values) + [0.0] * (DIM - len(values))
    return padded[:DIM]


@pytest.fixture
def store():
    s = Store(":memory:")
    try:
        yield s
    finally:
        s.close()


def seed(store: Store, source: str, texts, *, metadata=None):
    store.upsert_source(
        source, content_hash=source, characters=sum(len(t) for t in texts)
    )
    store.add_chunks(
        source,
        texts,
        [vec(float(i + 1)) for i in range(len(texts))],
        metadata=[metadata or {}] * len(texts),
    )


def match_rowids(store: Store, expression: str):
    """Run a raw MATCH so a syntax error surfaces instead of being swallowed."""
    return store.db.execute(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ?", (expression,)
    ).fetchall()


# --------------------------------------------------------------------------- #
# Escaping
# --------------------------------------------------------------------------- #

#: Every one of these has broken a naive MATCH implementation at some point.
HOSTILE_QUERIES = [
    pytest.param("the NOT thing", id="bare-NOT"),
    pytest.param("a AND b", id="bare-AND"),
    pytest.param("x OR y", id="bare-OR"),
    pytest.param('say "hi" there', id="embedded-quotes"),
    pytest.param("C++ and A*", id="operators-in-words"),
    pytest.param("cost: $5 (approx)", id="colon-dollar-parens"),
    pytest.param("NEAR(a b)", id="NEAR-call"),
    pytest.param("café", id="accented"),
    pytest.param("(((", id="only-parens"),
    pytest.param("***", id="only-stars"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace-only"),
    pytest.param("\U0001f600 \U0001f680", id="emoji"),
    pytest.param("x" * 5000, id="very-long"),
    pytest.param('" OR 1=1 --', id="injection-shaped"),
    pytest.param("^caret ^start", id="caret"),
    pytest.param("a-b-c", id="hyphenated"),
]


@pytest.mark.parametrize("raw", HOSTILE_QUERIES)
def test_escaped_query_is_either_valid_or_empty(store, raw):
    """Every input must produce a runnable MATCH, or "" meaning skip keyword search."""
    seed(store, "doc", ["the thing about a cafe and C++ and NEAR misses"])

    expression = escape_fts_query(raw)
    if expression == "":
        return  # documented contract: empty means "no searchable token"
    match_rowids(store, expression)  # must not raise


@pytest.mark.parametrize("raw", HOSTILE_QUERIES)
def test_search_keyword_never_raises(store, raw):
    seed(store, "doc", ["the thing about a cafe and C++"])
    assert isinstance(store.search_keyword(raw, k=5), list)


@pytest.mark.parametrize(
    "raw", ["", "   ", "(((", "***", "\U0001f600", "!!! ---", "\t\n"]
)
def test_queries_with_no_searchable_token_compile_to_empty(raw):
    assert escape_fts_query(raw) == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("the NOT thing", '"the" OR "NOT" OR "thing"'),
        ("a AND b", '"a" OR "AND" OR "b"'),
        ('say "hi"', '"say" OR "hi"'),
        ("C++ and A*", '"C" OR "and" OR "A"'),
    ],
)
def test_escaping_quotes_every_token_as_a_phrase(raw, expected):
    assert escape_fts_query(raw) == expected


def test_escaping_never_leaks_an_unbalanced_quote():
    expression = escape_fts_query('a " b "" c')
    assert expression.count('"') % 2 == 0


def test_prefix_mode_matches_a_partially_typed_word(store):
    seed(store, "doc", ["retrieval augmented generation"])
    expression = escape_fts_query("retriev", prefix=True)
    assert expression.endswith("*")
    assert match_rowids(store, expression)


def test_a_five_thousand_character_query_still_runs(store):
    seed(store, "doc", ["something short"])
    assert store.search_keyword("x" * 5000, k=5) == []
    assert store.search_keyword("short " * 800, k=5)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_bm25_ordering_puts_the_better_match_first(store):
    """fts5's bm25() is negative and *lower is better*; the store must respect that."""
    seed(
        store,
        "doc",
        [
            "quantum quantum quantum computing",
            "quantum computing is one topic among many other unrelated topics here",
            "cooking recipes for pasta and bread",
        ],
    )

    results = store.search_keyword("quantum", k=5)
    assert len(results) == 2  # the pasta chunk does not mention quantum

    ids = [doc_id for doc_id, _ in results]
    assert ids == [1, 2], "the denser, shorter match must rank first"

    scores = [score for _, score in results]
    assert all(score < 0 for score in scores), "fts5 bm25 scores are negative"
    assert scores == sorted(scores), "results must be ordered lowest (best) first"


def test_keyword_search_respects_k(store):
    seed(store, "doc", [f"common token number {i}" for i in range(10)])
    assert len(store.search_keyword("common", k=3)) == 3
    assert store.search_keyword("common", k=0) == []


def test_keyword_search_honours_a_metadata_filter(store):
    seed(store, "old", ["shared keyword here"], metadata={"year": 2020})
    seed(store, "new", ["shared keyword here"], metadata={"year": 2024})

    hits = store.search_keyword("shared", k=5, where={"year": 2024})
    assert len(hits) == 1
    assert store.fetch([hits[0][0]])[hits[0][0]].source == "new"


def test_keyword_search_honours_a_source_restriction(store):
    seed(store, "a", ["shared keyword here"])
    seed(store, "b", ["shared keyword here"])

    hits = store.search_keyword("shared", k=5, source="b")
    assert len(hits) == 1
    assert store.fetch([hits[0][0]])[hits[0][0]].source == "b"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def test_diacritics_are_folded_both_ways(store):
    """The tokenizer is configured with ``remove_diacritics 2``."""
    seed(store, "fr", ["le café est ouvert"])

    assert store.search_keyword("cafe", k=5), "unaccented query must find accented text"
    assert store.search_keyword("café", k=5), "accented query must still work"


def test_diacritic_folding_works_from_accented_text_to_plain(store):
    seed(store, "en", ["the cafe is open"])
    assert store.search_keyword("café", k=5)


def test_search_is_case_insensitive(store):
    seed(store, "doc", ["MiXeD CaSe Content"])
    assert store.search_keyword("mixed", k=5)
    assert store.search_keyword("CONTENT", k=5)


# --------------------------------------------------------------------------- #
# Index synchronisation -- the historically broken part
# --------------------------------------------------------------------------- #


def test_index_stays_in_sync_after_deleting_a_source(store):
    """The old design raised 'missing row N from content table' after a delete."""
    seed(store, "gone", ["alpha bravo", "charlie delta", "echo foxtrot"])
    seed(store, "kept", ["alpha zulu"])

    assert len(store.search_keyword("alpha", k=10)) == 2
    assert store.delete_source("gone") == 3

    surviving = store.search_keyword("alpha", k=10)
    assert len(surviving) == 1
    assert store.fetch([surviving[0][0]])[surviving[0][0]].source == "kept"

    # Terms that only existed in the deleted source are gone, not dangling.
    assert store.search_keyword("bravo", k=10) == []
    assert match_rowids(store, escape_fts_query("charlie")) == []


def test_index_stays_in_sync_after_delete_where(store):
    seed(store, "a", ["alpha bravo"], metadata={"year": 2020})
    seed(store, "b", ["alpha charlie"], metadata={"year": 2024})

    store.delete_where({"year": 2020})

    assert store.search_keyword("bravo", k=10) == []
    assert len(store.search_keyword("alpha", k=10)) == 1


def test_index_stays_in_sync_after_reset(store):
    seed(store, "a", ["alpha bravo", "charlie delta"])
    store.reset()

    assert store.search_keyword("alpha", k=10) == []
    assert match_rowids(store, escape_fts_query("alpha")) == []

    seed(store, "b", ["alpha again"])
    assert len(store.search_keyword("alpha", k=10)) == 1


def test_index_stays_in_sync_after_updating_a_document(store):
    """The AFTER UPDATE trigger must delete the old term set and insert the new."""
    seed(store, "doc", ["alpha bravo", "charlie delta"])
    store.db.execute("UPDATE documents SET text = 'alpha zulu' WHERE chunk_index = 0")

    assert store.search_keyword("bravo", k=10) == [], "stale terms must be dropped"
    assert store.search_keyword("zulu", k=10), "new terms must be indexed"
    assert len(store.search_keyword("alpha", k=10)) == 1


def test_delete_then_reinsert_the_same_text_stays_queryable(store):
    seed(store, "a", ["recurring paragraph text"])
    store.delete_source("a")
    seed(store, "a", ["recurring paragraph text"])

    hits = store.search_keyword("recurring", k=10)
    assert len(hits) == 1
    # The row really is reachable -- a desynchronised index throws here.
    assert store.fetch([hits[0][0]])[hits[0][0]].text == "recurring paragraph text"


def test_many_delete_and_reinsert_cycles_do_not_corrupt_the_index(store):
    for cycle in range(8):
        seed(store, "churn", [f"cycle {cycle} alpha", f"cycle {cycle} bravo"])
        assert store.search_keyword("alpha", k=10)
        store.delete_source("churn")
        assert store.search_keyword("alpha", k=10) == []

    seed(store, "final", ["alpha survives"])
    assert len(store.search_keyword("alpha", k=10)) == 1


def test_fts_integrity_check_passes_after_churn(store):
    seed(store, "a", ["alpha bravo", "charlie delta", "echo foxtrot"])
    store.delete_source("a")
    seed(store, "b", ["golf hotel"])
    store.db.execute("UPDATE documents SET text = 'india juliet' WHERE source = 'b'")
    store.optimize()

    # fts5's own consistency check between the index and its content table.
    store.db.execute(
        "INSERT INTO documents_fts(documents_fts, rank) VALUES ('integrity-check', 1)"
    )


def test_a_malformed_match_degrades_to_no_hits_rather_than_failing(store, monkeypatch):
    """Vector search must still answer even if keyword search blows up."""
    seed(store, "doc", ["alpha bravo"])
    monkeypatch.setattr(
        "softrag.store.escape_fts_query", lambda text, **kw: 'NEAR("a" "b"'
    )
    assert store.search_keyword("anything", k=5) == []


def test_raw_unescaped_input_really_would_have_failed(store):
    """Justifies escape_fts_query existing at all."""
    seed(store, "doc", ["the thing"])
    with pytest.raises(sqlite3.OperationalError):
        match_rowids(store, "the NOT")
