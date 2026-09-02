"""The SQLite storage layer: schema, dimensions, writes, deletes, stats."""

from __future__ import annotations

import sqlite3

import pytest

from softrag.errors import DimensionMismatchError, SchemaVersionError, StoreError
from softrag.store import SCHEMA_VERSION, Store, pack_vector, unpack_vector

DIM = 8


def vec(*values: float) -> list:
    """Pad a few leading values out to ``DIM`` so widths always agree."""
    padded = list(values) + [0.0] * (DIM - len(values))
    return padded[:DIM]


@pytest.fixture
def store():
    s = Store(":memory:")
    try:
        yield s
    finally:
        s.close()


def seed(store: Store, source: str, texts, *, metadata=None, content_hash="h"):
    store.upsert_source(
        source,
        content_hash=content_hash,
        characters=sum(len(t) for t in texts),
        metadata=metadata or {},
    )
    vectors = [vec(float(i + 1)) for i in range(len(texts))]
    return store.add_chunks(
        source, texts, vectors, metadata=[metadata or {}] * len(texts)
    )


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_schema_tables_are_created(store):
    names = {
        row[0]
        for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"documents", "sources", "softrag_meta"} <= names


def test_schema_creates_fts_and_triggers(store):
    objects = {
        row[0]
        for row in store.db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        )
    }
    assert "documents_fts" in objects
    # Without all three triggers the external-content FTS index desynchronises.
    assert {"documents_ai", "documents_ad", "documents_au"} <= objects


def test_pragma_user_version_is_set(store):
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopening_an_existing_database_keeps_its_content(tmp_path):
    path = tmp_path / "kb.db"
    first = Store(path)
    seed(first, "a", ["hello world"])
    first.close()

    second = Store(path)
    try:
        assert second.count() == 1
        assert second.dimensions == DIM
        assert second.has_source("a")
        # The vector table survived, so vector search works immediately.
        assert second.search_vector(vec(1.0), k=1)
    finally:
        second.close()


def test_wrong_user_version_raises_schema_version_error(tmp_path):
    path = tmp_path / "kb.db"
    Store(path).close()

    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version=99")
    raw.commit()
    raw.close()

    with pytest.raises(SchemaVersionError) as excinfo:
        Store(path)
    assert excinfo.value.found == 99
    assert excinfo.value.expected == SCHEMA_VERSION


def test_read_only_store_without_schema_raises(tmp_path):
    path = tmp_path / "empty.db"
    path.write_bytes(b"")
    with pytest.raises(StoreError, match="read-only"):
        Store(path, read_only=True)


# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #


def test_dimensions_are_unknown_before_the_first_write(store):
    assert store.dimensions is None
    assert store.stats().dimensions is None


def test_dimensions_are_learned_from_the_first_write(store):
    seed(store, "a", ["one chunk"])
    assert store.dimensions == DIM
    stored = store.db.execute(
        "SELECT value FROM softrag_meta WHERE key='dimensions'"
    ).fetchone()
    assert stored[0] == str(DIM)


def test_a_second_embedder_of_a_different_width_raises(store):
    seed(store, "a", ["one chunk"])
    with pytest.raises(DimensionMismatchError) as excinfo:
        store.add_chunks("a", ["other"], [[0.5] * (DIM + 8)])
    assert excinfo.value.expected == DIM
    assert excinfo.value.got == DIM + 8


def test_dimension_mismatch_survives_a_reopen(tmp_path):
    path = tmp_path / "kb.db"
    first = Store(path)
    seed(first, "a", ["one chunk"])
    first.close()

    second = Store(path)
    try:
        with pytest.raises(DimensionMismatchError):
            second.ensure_dimensions(DIM * 2)
    finally:
        second.close()


def test_inconsistent_widths_within_one_batch_raise(store):
    store.upsert_source("a", content_hash="h", characters=2)
    with pytest.raises(StoreError, match="inconsistent widths"):
        store.add_chunks("a", ["x", "y"], [vec(1.0), [0.1] * (DIM + 1)])


def test_zero_dimensions_is_rejected(store):
    with pytest.raises(StoreError, match="Invalid embedding width"):
        store.ensure_dimensions(0)


def test_querying_with_a_wrong_width_vector_raises(store):
    seed(store, "a", ["one chunk"])
    with pytest.raises(DimensionMismatchError):
        store.search_vector([0.1] * (DIM + 4), k=3)


# --------------------------------------------------------------------------- #
# Writes and dedup
# --------------------------------------------------------------------------- #


def test_add_chunks_returns_added_and_skipped(store):
    assert seed(store, "a", ["alpha", "beta"]) == (2, 0)
    # Re-adding the identical chunks under the same source is a no-op.
    assert seed(store, "a", ["alpha", "beta"]) == (0, 2)
    assert store.count() == 2


def test_add_chunks_rejects_a_length_mismatch(store):
    store.upsert_source("a", content_hash="h", characters=1)
    with pytest.raises(StoreError, match="one vector per input"):
        store.add_chunks("a", ["x", "y"], [vec(1.0)])


def test_add_chunks_with_nothing_to_do(store):
    assert store.add_chunks("a", [], []) == (0, 0)


def test_dedup_is_per_source_not_global(store):
    """The same text under two sources is stored twice, by design."""
    assert seed(store, "one", ["identical paragraph"]) == (1, 0)
    assert seed(store, "two", ["identical paragraph"]) == (1, 0)
    assert store.count() == 2

    # ...and deleting one source must not blank the other.
    assert store.delete_source("one") == 1
    assert store.count() == 1
    assert not store.has_source("one")
    assert store.has_source("two")
    assert [h.text for h in store.neighbors("two", 0, radius=1)] == [
        "identical paragraph"
    ]


def test_upsert_source_returns_the_previous_hash(store):
    assert store.upsert_source("a", content_hash="h1", characters=1) is None
    assert store.upsert_source("a", content_hash="h2", characters=1) == "h1"
    assert store.has_source("a", content_hash="h2")
    assert not store.has_source("a", content_hash="h1")


def test_source_chunk_count_is_maintained(store):
    seed(store, "a", ["one", "two", "three"])
    info = next(s for s in store.sources() if s.source == "a")
    assert info.chunks == 3


def test_metadata_round_trips_through_json(store):
    seed(store, "a", ["x"], metadata={"year": 2024, "tags": ["p", "q"], "ok": True})
    hit = store.neighbors("a", 0, radius=1)[0]
    assert hit.metadata == {"year": 2024, "tags": ["p", "q"], "ok": True}


# --------------------------------------------------------------------------- #
# Deletes
# --------------------------------------------------------------------------- #


def test_delete_source_removes_chunks_vectors_and_the_source_row(store):
    seed(store, "a", ["alpha", "beta"])
    seed(store, "b", ["gamma"])

    assert store.delete_source("a") == 2
    assert store.count() == 1
    assert store.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 1
    assert [s.source for s in store.sources()] == ["b"]


def test_delete_source_that_does_not_exist_is_a_no_op(store):
    seed(store, "a", ["alpha"])
    assert store.delete_source("nope") == 0
    assert store.count() == 1


def test_delete_where_removes_matching_chunks_only(store):
    seed(store, "a", ["alpha"], metadata={"year": 2020})
    seed(store, "b", ["beta"], metadata={"year": 2024})

    assert store.delete_where({"year": 2020}) == 1
    assert store.count() == 1
    # The now-empty source row is cleaned up too.
    assert [s.source for s in store.sources()] == ["b"]


def test_delete_where_matching_nothing_returns_zero(store):
    seed(store, "a", ["alpha"], metadata={"year": 2020})
    assert store.delete_where({"year": 1999}) == 0
    assert store.count() == 1


def test_reset_empties_the_index_but_keeps_the_schema(store):
    seed(store, "a", ["alpha", "beta"])
    store.reset()

    assert store.count() == 0
    assert store.sources() == []
    assert store.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
    # The vector width is deliberately retained, so the next write still fits.
    assert store.dimensions == DIM
    assert seed(store, "a", ["alpha"]) == (1, 0)


def test_optimize_leaves_the_index_queryable(tmp_path):
    store = Store(tmp_path / "kb.db")
    try:
        seed(store, "a", ["alpha bravo", "charlie delta"])
        store.delete_source("a")
        seed(store, "b", ["echo foxtrot"])
        store.optimize()

        assert store.count() == 1
        assert store.search_keyword("foxtrot", k=5)
        assert store.search_vector(vec(1.0), k=5)
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #


def test_count_and_has_source(store):
    assert store.count() == 0
    assert not store.has_source("a")

    seed(store, "a", ["alpha"], content_hash="abc")
    assert store.count() == 1
    assert store.has_source("a")
    assert store.has_source("a", content_hash="abc")
    assert not store.has_source("a", content_hash="different")


def test_stats_for_an_in_memory_store(store):
    seed(store, "a", ["alpha", "beta"])
    seed(store, "b", ["gamma"])

    stats = store.stats()
    assert stats.path == ":memory:"
    assert stats.documents == 3
    assert stats.sources == 2
    assert stats.dimensions == DIM
    assert stats.schema_version == SCHEMA_VERSION
    assert stats.size_bytes == 0


def test_stats_reports_a_real_size_on_disk(tmp_path):
    store = Store(tmp_path / "kb.db")
    try:
        seed(store, "a", ["alpha"])
        stats = store.stats()
        assert stats.size_bytes > 0
        assert stats.size_mb == pytest.approx(stats.size_bytes / (1024 * 1024))
    finally:
        store.close()


def test_neighbors_returns_the_window_around_a_chunk(store):
    seed(store, "a", ["zero", "one", "two", "three", "four"])

    window = store.neighbors("a", 2, radius=1)
    assert [hit.text for hit in window] == ["one", "two", "three"]
    assert [hit.index for hit in window] == [1, 2, 3]

    # The window is clipped at the document edges.
    assert [h.text for h in store.neighbors("a", 0, radius=2)] == ["zero", "one", "two"]


def test_neighbors_with_a_non_positive_radius_returns_nothing(store):
    seed(store, "a", ["zero", "one"])
    assert store.neighbors("a", 0, radius=0) == []
    assert store.neighbors("a", 0, radius=-1) == []


def test_neighbors_never_crosses_a_source_boundary(store):
    seed(store, "a", ["a0", "a1"])
    seed(store, "b", ["b0", "b1"])
    assert all(hit.source == "a" for hit in store.neighbors("a", 0, radius=5))


def test_fetch_returns_hits_keyed_by_id(store):
    seed(store, "a", ["alpha", "beta"])
    ids = [row[0] for row in store.db.execute("SELECT id FROM documents ORDER BY id")]

    loaded = store.fetch(ids)
    assert set(loaded) == set(ids)
    assert loaded[ids[0]].text == "alpha"
    assert store.fetch([]) == {}
    assert store.fetch([9999]) == {}


def test_sources_lists_everything_and_honours_limit(store):
    seed(store, "old", ["alpha"])
    seed(store, "new", ["beta"])

    assert {s.source for s in store.sources()} == {"old", "new"}
    assert len(store.sources(limit=1)) == 1


def test_sources_are_listed_most_recently_updated_first(store):
    seed(store, "old", ["alpha"])
    seed(store, "new", ["beta"])
    store.upsert_source("new", content_hash="h2", characters=1)

    assert next(s.source for s in store.sources()) == "new"


# --------------------------------------------------------------------------- #
# Vector search
# --------------------------------------------------------------------------- #


def test_search_vector_orders_by_distance(store):
    store.upsert_source("a", content_hash="h", characters=3)
    store.add_chunks(
        "a",
        ["north", "east", "diagonal"],
        [vec(1.0, 0.0), vec(0.0, 1.0), vec(0.9, 0.4)],
    )

    results = store.search_vector(vec(1.0, 0.0), k=3)
    assert [doc_id for doc_id, _ in results][:2] == [1, 3]
    assert results[0][1] < results[-1][1]


def test_search_vector_honours_a_metadata_filter(store):
    seed(store, "a", ["alpha"], metadata={"year": 2020})
    seed(store, "b", ["beta"], metadata={"year": 2024})

    only_new = store.search_vector(vec(1.0), k=5, where={"year": 2024})
    assert len(only_new) == 1
    assert store.fetch([only_new[0][0]])[only_new[0][0]].source == "b"


def test_search_vector_honours_a_source_restriction(store):
    seed(store, "a", ["alpha"])
    seed(store, "b", ["beta"])

    restricted = store.search_vector(vec(1.0), k=5, source="b")
    assert len(restricted) == 1
    assert store.fetch([restricted[0][0]])[restricted[0][0]].source == "b"


def test_search_vector_is_empty_before_any_vector_exists(store):
    assert store.search_vector([0.1] * DIM, k=5) == []


def test_search_vector_with_a_non_positive_k(store):
    seed(store, "a", ["alpha"])
    assert store.search_vector(vec(1.0), k=0) == []


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


def test_pack_and_unpack_round_trip():
    original = [0.5, -0.25, 1.0, 0.0]
    assert unpack_vector(pack_vector(original)) == pytest.approx(original)


def test_pack_vector_rejects_non_numbers():
    with pytest.raises(StoreError, match="could not be packed"):
        pack_vector(["not", "a", "number"])


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_close_is_idempotent(store):
    store.close()
    store.close()


def test_store_works_as_a_context_manager(tmp_path):
    with Store(tmp_path / "kb.db") as store:
        seed(store, "a", ["alpha"])
        assert store.count() == 1
    assert store._closed
