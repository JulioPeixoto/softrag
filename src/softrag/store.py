"""SQLite storage layer.

Everything softrag persists -- chunk text, metadata, the vector index and the
full-text index -- lives in one SQLite file. This module owns the schema, the
migrations and every SQL statement; nothing above it writes SQL.

Design notes
------------
*Vector search* uses ``sqlite-vec``'s ``vec0`` virtual table. Unfiltered queries
use its native KNN operator, which is the fast path. Filtered queries resolve
the filter first against the indexed ``documents`` table and then score only the
surviving rows exactly, so a metadata filter never silently costs recall the way
over-fetch-and-post-filter does.

*Keyword search* uses FTS5 with an external-content table kept in sync by
triggers. Without those triggers a delete leaves the index pointing at rows that
no longer exist and later queries fail outright, so they are not optional.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .errors import (
    DimensionMismatchError,
    SchemaVersionError,
    StoreError,
)
from .filters import compile_where
from .types import Hit, SourceInfo, Stats, Where

log = logging.getLogger("softrag.store")

__all__ = ["Store", "escape_fts_query", "pack_vector", "unpack_vector"]

SCHEMA_VERSION = 1
DEFAULT_PAGE_SIZE = 8192

#: vec0 rejects a KNN query asking for more than this many neighbours:
#: "k value in knn query too large, provided N and the limit is 4096".
VEC0_MAX_K = 4096

#: Above this many filter-matching rows, exact rescoring stops paying for
#: itself and we fall back to over-fetched KNN followed by filtering.
#:
#: The number comes from measurement rather than taste. Scoring a filtered
#: subset uses one point lookup per row (~8 us each on a 384-dimension index),
#: while vec0's native KNN costs a flat ~7 ms over 20k vectors no matter how
#: many neighbours are asked for. The two meet at roughly a thousand rows.
#: Note that a single ``doc_id IN (...)`` is *not* the alternative it looks
#: like: vec0 answers it with a full table scan, so it is slower than the
#: native KNN it was meant to avoid.
EXACT_FILTER_LIMIT = 1_000

#: How many extra candidates to pull when post-filtering is unavoidable.
POST_FILTER_OVERFETCH = 8

_WORD = re.compile(r"[\w']+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def pack_vector(vec: Sequence[float]) -> bytes:
    """Pack a float sequence into the compact binary form ``vec0`` stores.

    Args:
        vec: The embedding.

    Returns:
        Little-endian packed float32 bytes.

    Raises:
        StoreError: If the values are not all finite numbers.
    """
    try:
        return struct.pack(f"<{len(vec)}f", *vec)
    except (struct.error, TypeError) as exc:
        raise StoreError(
            f"Embedding could not be packed: {exc}. Expected a flat sequence of "
            f"floats, got {type(vec).__name__} of length {len(vec)}."
        ) from exc


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_vector`."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def escape_fts_query(text: str, *, prefix: bool = False) -> str:
    """Turn arbitrary user text into a syntactically valid FTS5 MATCH query.

    FTS5's query language treats ``AND``, ``OR``, ``NOT``, ``NEAR``, ``*``,
    ``:``, ``^``, ``(`` and ``"`` as syntax, so feeding raw user input to MATCH
    raises ``fts5: syntax error`` on ordinary questions such as *"the NOT
    thing"*. Quoting each token as a phrase neutralises all of it.

    Args:
        text: Raw user text.
        prefix: Also match words that merely start with the final token, which
            helps with partially typed queries.

    Returns:
        A safe MATCH expression, or ``""`` if the input holds no searchable
        token -- callers must treat empty as "skip keyword search".

    Example:
        >>> escape_fts_query('the NOT thing')
        '"the" OR "NOT" OR "thing"'
        >>> escape_fts_query('say "hi"')
        '"say" OR "hi"'
    """
    tokens = _WORD.findall(text)
    if not tokens:
        return ""
    quoted = ['"' + tok.replace('"', '""') + '"' for tok in tokens]
    if prefix:
        quoted[-1] = quoted[-1] + " *"
    return " OR ".join(quoted)


_clock_lock = threading.Lock()
_last_timestamp = ""


def _utcnow() -> str:
    """Current UTC time, guaranteed to increase between calls.

    Sources are ordered by ``updated_at``, so ties make "most recently updated
    first" arbitrary. Formatting to microseconds is not enough on its own:
    Windows' system clock advances in steps of roughly 15 ms, so a whole batch
    of writes reads back the identical timestamp. Nudging the value forward
    whenever the clock has not moved keeps the ordering strict.
    """
    global _last_timestamp
    now = datetime.now(timezone.utc)
    with _clock_lock:
        stamp = now.isoformat(timespec="microseconds")
        if stamp <= _last_timestamp:
            previous = datetime.fromisoformat(_last_timestamp)
            stamp = (previous + timedelta(microseconds=1)).isoformat(
                timespec="microseconds"
            )
        _last_timestamp = stamp
    return stamp


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class Store:
    """The SQLite-backed index.

    Args:
        path: Database file path. ``":memory:"`` gives an ephemeral index.
        dimensions: Vector width. Usually left as ``None`` and learned from the
            first embedding written; pass it to fail fast instead.
        read_only: Open without allowing writes or schema creation.
        timeout: Seconds to wait for a competing writer before giving up.
    """

    def __init__(
        self,
        path: str | os.PathLike = "softrag.db",
        *,
        dimensions: int | None = None,
        read_only: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self.read_only = read_only
        self._lock = threading.RLock()
        self._dimensions: int | None = dimensions
        self._vectors_ready = False
        self._closed = False

        self.db = self._connect(timeout)
        self._configure()
        self._load_extension()
        self._migrate()

    # -- lifecycle ---------------------------------------------------------- #

    def _connect(self, timeout: float) -> sqlite3.Connection:
        is_memory = str(self.path) == ":memory:"
        if not is_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._is_new = (
            is_memory or not self.path.exists() or self.path.stat().st_size == 0
        )
        try:
            return sqlite3.connect(
                str(self.path),
                timeout=timeout,
                check_same_thread=False,
                isolation_level=None,  # explicit transaction control
            )
        except sqlite3.Error as exc:
            raise StoreError(f"Could not open database {self.path}: {exc}") from exc

    def _configure(self) -> None:
        db = self.db
        # page_size only takes effect before the first table exists.
        if self._is_new:
            db.execute(f"PRAGMA page_size={DEFAULT_PAGE_SIZE}")
        if str(self.path) != ":memory:":
            db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("PRAGMA cache_size=-32000")  # ~32 MB

    def _load_extension(self) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - hard dependency
            raise StoreError(
                "sqlite-vec is required but not installed. Install softrag's "
                "dependencies with: pip install sqlite-vec"
            ) from exc
        try:
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
        except (AttributeError, sqlite3.Error) as exc:
            raise StoreError(
                "Could not load the sqlite-vec extension. Your Python was likely "
                "built against a SQLite without extension support "
                f"(sqlite {sqlite3.sqlite_version}). Original error: {exc}"
            ) from exc
        finally:
            with contextlib.suppress(AttributeError, sqlite3.Error):
                self.db.enable_load_extension(False)

        functions = {
            row[0] for row in self.db.execute("SELECT name FROM pragma_function_list")
        }
        missing = {"vec_distance_cosine", "vec_version"} - functions
        if missing:
            raise StoreError(
                f"sqlite-vec loaded but did not register {sorted(missing)}. "
                "The installed sqlite-vec build looks incompatible."
            )

    def close(self) -> None:
        """Close the underlying connection. Safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            try:
                self.db.close()
            finally:
                self._closed = True

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "closed" if self._closed else f"dim={self._dimensions}"
        return f"<Store {str(self.path)!r} {state}>"

    # -- schema ------------------------------------------------------------- #

    def _migrate(self) -> None:
        with self._lock:
            found = self.db.execute("PRAGMA user_version").fetchone()[0]
            has_tables = bool(
                self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
                ).fetchone()
            )

            if not has_tables:
                if self.read_only:
                    raise StoreError(
                        f"{self.path} has no softrag schema and the store is read-only."
                    )
                self._create_schema()
                self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                found = SCHEMA_VERSION

            if found != SCHEMA_VERSION:
                raise SchemaVersionError(found, SCHEMA_VERSION, str(self.path))

            self._dimensions = self._read_dimensions() or self._dimensions
            self._vectors_ready = bool(
                self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='vectors'"
                ).fetchone()
            )

    def _create_schema(self) -> None:
        log.debug("creating softrag schema v%d at %s", SCHEMA_VERSION, self.path)
        self.db.executescript(
            """
            BEGIN;

            CREATE TABLE softrag_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE sources (
                source       TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                characters   INTEGER NOT NULL DEFAULT 0,
                chunks       INTEGER NOT NULL DEFAULT 0,
                metadata     TEXT NOT NULL DEFAULT '{}',
                added_at     TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE documents (
                id          INTEGER PRIMARY KEY,
                source      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                hash        TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (source) REFERENCES sources(source) ON DELETE CASCADE
            );

            CREATE INDEX idx_documents_source ON documents(source, chunk_index);

            -- Re-adding an unchanged chunk of the same source is a no-op, while
            -- the identical paragraph appearing in a second file is still
            -- indexed under that file, so deleting one source cannot blank the
            -- other.
            CREATE UNIQUE INDEX idx_documents_dedup ON documents(source, hash);

            CREATE VIRTUAL TABLE documents_fts USING fts5(
                text,
                content='documents',
                content_rowid='id',
                tokenize="unicode61 remove_diacritics 2"
            );

            -- External-content FTS is only correct while these triggers exist:
            -- without them a DELETE leaves dangling index entries and every
            -- later query fails with "missing row N from content table".
            CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;

            CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO documents_fts(rowid, text) VALUES (new.id, new.text);
            END;

            COMMIT;
            """
        )
        now = _utcnow()
        self.db.executemany(
            "INSERT INTO softrag_meta(key, value) VALUES (?, ?)",
            [("created_at", now), ("distance_metric", "cosine")],
        )

    def _read_dimensions(self) -> int | None:
        row = self.db.execute(
            "SELECT value FROM softrag_meta WHERE key='dimensions'"
        ).fetchone()
        return int(row[0]) if row else None

    @property
    def dimensions(self) -> int | None:
        """Vector width of this index, or ``None`` before the first write."""
        return self._dimensions

    def ensure_dimensions(self, dimensions: int) -> None:
        """Pin the index to a vector width, creating the vector table once.

        The first embedding written decides the width. Later mismatches are a
        user error worth failing loudly on, since silently mixing widths yields
        an index that returns nonsense.

        Raises:
            DimensionMismatchError: If the index already uses a different width.
        """
        with self._lock:
            if self._dimensions is not None and self._dimensions != dimensions:
                raise DimensionMismatchError(self._dimensions, dimensions)
            if self._vectors_ready:
                return
            if dimensions <= 0:
                raise StoreError(f"Invalid embedding width: {dimensions}")

            self.db.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                    doc_id INTEGER PRIMARY KEY,
                    embedding float[{dimensions}] distance_metric=cosine,
                    source TEXT
                )
                """
            )
            self.db.execute(
                "INSERT OR REPLACE INTO softrag_meta(key, value) VALUES ('dimensions', ?)",
                (str(dimensions),),
            )
            self._dimensions = dimensions
            self._vectors_ready = True

    def check_embedder(self, fingerprint: str) -> str | None:
        """Record which embedding model built this index, or flag a change.

        Vector width already catches the obvious mistake of swapping a 384-
        dimension model for a 1536-dimension one. It cannot catch the quieter
        one: two different models of the *same* width produce vectors in
        unrelated spaces, so the index keeps answering and every neighbour it
        returns is meaningless. Comparing a model fingerprint catches that.

        This warns rather than raises, because a fingerprint can change for
        harmless reasons -- wrapping the same model in a different adapter, for
        instance -- and refusing to open an index over a naming detail would be
        worse than the risk.

        Args:
            fingerprint: Identifier from
                :func:`~softrag.providers.embedder_fingerprint`.

        Returns:
            The previously recorded fingerprint when it differs, otherwise
            ``None``.
        """
        with self._lock:
            row = self.db.execute(
                "SELECT value FROM softrag_meta WHERE key='embedder'"
            ).fetchone()
            previous = row[0] if row else None

            if previous is None:
                if not self.read_only:
                    self.db.execute(
                        "INSERT OR REPLACE INTO softrag_meta(key, value) "
                        "VALUES ('embedder', ?)",
                        (fingerprint,),
                    )
                return None

            if previous == fingerprint or self.count() == 0:
                return None

        log.warning(
            "This index was built with embedding model %r but %r is configured "
            "now. Vectors from different models are not comparable, so search "
            "results will be meaningless. Re-index into a new database file, or "
            "switch back to the original model.",
            previous,
            fingerprint,
        )
        return previous

    # -- writes ------------------------------------------------------------- #

    def upsert_source(
        self,
        source: str,
        *,
        content_hash: str,
        characters: int,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Record a source, returning the previous content hash if it existed."""
        with self._lock:
            row = self.db.execute(
                "SELECT content_hash FROM sources WHERE source = ?", (source,)
            ).fetchone()
            previous = row[0] if row else None
            now = _utcnow()
            payload = json.dumps(metadata or {}, ensure_ascii=False)
            if previous is None:
                self.db.execute(
                    "INSERT INTO sources(source, content_hash, characters, chunks, "
                    "metadata, added_at, updated_at) VALUES (?, ?, ?, 0, ?, ?, ?)",
                    (source, content_hash, characters, payload, now, now),
                )
            else:
                self.db.execute(
                    "UPDATE sources SET content_hash=?, characters=?, metadata=?, "
                    "updated_at=? WHERE source=?",
                    (content_hash, characters, payload, now, source),
                )
            return previous

    def add_chunks(
        self,
        source: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        *,
        metadata: Sequence[dict[str, Any]] | None = None,
        start_index: int = 0,
    ) -> tuple[int, int]:
        """Insert chunks with their embeddings.

        Chunks whose ``(source, hash)`` pair is already present are skipped, so
        re-adding an unchanged document costs nothing.

        Args:
            source: Stable identifier of the parent document.
            chunks: Chunk texts.
            embeddings: One embedding per chunk, same order.
            metadata: Optional per-chunk metadata.
            start_index: Chunk index to start numbering from.

        Returns:
            ``(added, skipped)`` counts.
        """
        if len(chunks) != len(embeddings):
            raise StoreError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings; "
                "the embedder must return one vector per input, in order."
            )
        if not chunks:
            return 0, 0

        width = len(embeddings[0])
        self.ensure_dimensions(width)
        for i, vec in enumerate(embeddings):
            if len(vec) != width:
                raise StoreError(
                    f"Embedding {i} has {len(vec)} dimensions but embedding 0 has "
                    f"{width}. The embedder returned inconsistent widths."
                )

        now = _utcnow()
        added = skipped = 0
        with self._lock, self._transaction():
            for offset, (text, vector) in enumerate(zip(chunks, embeddings, strict=True)):
                meta = dict(metadata[offset]) if metadata else {}
                digest = _hash_text(text)
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO documents"
                    "(source, chunk_index, text, hash, metadata, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        source,
                        start_index + offset,
                        text,
                        digest,
                        json.dumps(meta, ensure_ascii=False),
                        now,
                    ),
                )
                if cursor.rowcount == 0:
                    skipped += 1
                    continue
                self.db.execute(
                    "INSERT INTO vectors(doc_id, embedding, source) VALUES (?, ?, ?)",
                    (cursor.lastrowid, pack_vector(vector), source),
                )
                added += 1

            self.db.execute(
                "UPDATE sources SET chunks = (SELECT COUNT(*) FROM documents "
                "WHERE source = ?), updated_at = ? WHERE source = ?",
                (source, now, source),
            )
        return added, skipped

    def delete_source(self, source: str) -> int:
        """Remove a source and every chunk belonging to it.

        Returns:
            How many chunks were removed.
        """
        with self._lock, self._transaction():
            count = self.db.execute(
                "SELECT COUNT(*) FROM documents WHERE source = ?", (source,)
            ).fetchone()[0]
            if self._vectors_ready:
                self.db.execute("DELETE FROM vectors WHERE source = ?", (source,))
            self.db.execute("DELETE FROM documents WHERE source = ?", (source,))
            self.db.execute("DELETE FROM sources WHERE source = ?", (source,))
        return int(count)

    def delete_where(self, where: Where) -> int:
        """Remove every chunk matching a metadata filter.

        Returns:
            How many chunks were removed.
        """
        predicate, params = compile_where(where, column="d.metadata")
        with self._lock, self._transaction():
            ids = [
                row[0]
                for row in self.db.execute(
                    f"SELECT d.id FROM documents d WHERE {predicate}", params
                )
            ]
            if not ids:
                return 0
            self._delete_ids(ids)
            self.db.execute(
                "DELETE FROM sources WHERE source NOT IN (SELECT DISTINCT source FROM documents)"
            )
            # A partial delete leaves surviving sources with a stale chunk
            # count, which would then be wrong in sources() and in the CLI.
            self.db.execute(
                "UPDATE sources SET chunks = ("
                "  SELECT COUNT(*) FROM documents WHERE documents.source = sources.source"
                "), updated_at = ?",
                (_utcnow(),),
            )
        return len(ids)

    def _delete_ids(self, ids: Sequence[int]) -> None:
        # documents is an ordinary table, so a batched IN is the cheap way to
        # delete from it. vectors is a vec0 virtual table, where an IN forces a
        # full scan, so it gets one keyed delete per row instead.
        if self._vectors_ready:
            for doc_id in ids:
                self.db.execute("DELETE FROM vectors WHERE doc_id = ?", (doc_id,))
        for batch in _batched(ids, 500):
            placeholders = ",".join("?" for _ in batch)
            self.db.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", batch)

    def reset(self) -> None:
        """Drop all indexed content, keeping the schema and vector width."""
        with self._lock, self._transaction():
            if self._vectors_ready:
                self.db.execute("DELETE FROM vectors")
            self.db.execute("DELETE FROM documents")
            self.db.execute("DELETE FROM sources")

    def optimize(self) -> None:
        """Compact the FTS index and reclaim free pages.

        Worth calling after a large ingest or many deletions; never required for
        correctness.
        """
        with self._lock:
            self.db.execute(
                "INSERT INTO documents_fts(documents_fts) VALUES ('optimize')"
            )
            self.db.execute("PRAGMA optimize")
            self.db.execute("VACUUM")

    # -- reads -------------------------------------------------------------- #

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        where: Where | None = None,
        source: str | None = None,
    ) -> list[tuple[int, float]]:
        """Nearest neighbours by cosine distance.

        Without a filter this uses ``vec0``'s native KNN. With a filter the
        matching rows are resolved first and scored exactly, which keeps recall
        intact for selective filters instead of hoping over-fetch was enough.

        Args:
            query_vector: The embedded query.
            k: How many neighbours to return.
            where: Optional metadata filter.
            source: Optional exact source restriction (cheaper than ``where``).

        Returns:
            ``(document_id, distance)`` pairs, nearest first. Cosine distance is
            in ``[0, 2]``; smaller is more similar.
        """
        if not self._vectors_ready or k <= 0:
            return []
        blob = pack_vector(query_vector)
        if self._dimensions is not None and len(query_vector) != self._dimensions:
            raise DimensionMismatchError(self._dimensions, len(query_vector))

        with self._lock:
            if where is None:
                return self._native_knn(blob, k, source=source)

            candidates = self._filter_ids(where, source, limit=EXACT_FILTER_LIMIT + 1)
            if not candidates:
                return []
            if len(candidates) <= EXACT_FILTER_LIMIT:
                return self._exact_vector_scan(blob, candidates, k)
            return self._approximate_filtered(blob, where, source, k)

    def _native_knn(
        self, blob: bytes, k: int, *, source: str | None = None
    ) -> list[tuple[int, float]]:
        """vec0's own KNN operator: the fast path when nothing needs filtering."""
        capped = min(k, VEC0_MAX_K)
        if capped < k:
            log.warning(
                "vec0 caps KNN at %d neighbours; returning %d instead of the "
                "requested %d",
                VEC0_MAX_K,
                capped,
                k,
            )
        if source is None:
            sql = "SELECT doc_id, distance FROM vectors WHERE embedding MATCH ? AND k = ?"
            params: tuple[Any, ...] = (blob, capped)
        else:
            sql = (
                "SELECT doc_id, distance FROM vectors "
                "WHERE embedding MATCH ? AND k = ? AND source = ?"
            )
            params = (blob, capped, source)
        rows = self.db.execute(sql, params).fetchall()
        return [(int(i), float(d)) for i, d in rows if d is not None]

    def _exact_vector_scan(
        self, blob: bytes, ids: Sequence[int], k: int
    ) -> list[tuple[int, float]]:
        """Score a known set of rows exactly and keep the best ``k``.

        One point lookup per row. That looks wasteful next to a single
        ``doc_id IN (...)``, but vec0 answers an ``IN`` with a full table scan
        while an equality on the primary key is a real lookup, which makes the
        loop several times faster for the selective filters that reach here.
        """
        scored: list[tuple[int, float]] = []
        for doc_id in ids:
            row = self.db.execute(
                "SELECT doc_id, vec_distance_cosine(embedding, ?) AS distance "
                "FROM vectors WHERE doc_id = ?",
                (blob, doc_id),
            ).fetchone()
            if row is None or row[1] is None or math.isnan(row[1]):
                continue
            scored.append((int(row[0]), float(row[1])))
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]

    def _approximate_filtered(
        self, blob: bytes, where: Where | None, source: str | None, k: int
    ) -> list[tuple[int, float]]:
        """Fall back to over-fetched KNN, then drop non-matching rows.

        Only reached when the filter matches too many rows to rescore exactly,
        which also means it is unselective enough that over-fetching finds
        plenty of survivors.
        """
        predicate, params = compile_where(where, column="d.metadata")
        source_clause = " AND d.source = ?" if source else ""
        source_params = [source] if source else []
        overfetch = min(max(k * POST_FILTER_OVERFETCH, k), VEC0_MAX_K)
        rows = self.db.execute(
            f"""
            WITH knn AS (
                SELECT doc_id, distance FROM vectors
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT knn.doc_id, knn.distance
            FROM knn JOIN documents d ON d.id = knn.doc_id
            WHERE {predicate}{source_clause}
            ORDER BY knn.distance
            LIMIT ?
            """,
            (blob, overfetch, *params, *source_params, k),
        ).fetchall()
        return [(int(i), float(d)) for i, d in rows if d is not None]

    def vectors_for(self, ids: Sequence[int]) -> dict[int, list[float]]:
        """Load stored embeddings by document id.

        Point lookups again, for the same reason as :meth:`_exact_vector_scan`:
        an ``IN`` list would make vec0 scan the whole table.
        """
        if not ids or not self._vectors_ready:
            return {}
        out: dict[int, list[float]] = {}
        with self._lock:
            for doc_id in ids:
                row = self.db.execute(
                    "SELECT embedding FROM vectors WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if row is not None and row[0] is not None:
                    out[int(doc_id)] = unpack_vector(row[0])
        return out

    def search_keyword(
        self,
        query: str,
        *,
        k: int,
        where: Where | None = None,
        source: str | None = None,
    ) -> list[tuple[int, float]]:
        """Best BM25 matches for ``query``.

        Args:
            query: Raw user text; escaped internally, never passed through.
            k: How many matches to return.
            where: Optional metadata filter.
            source: Optional exact source restriction.

        Returns:
            ``(document_id, bm25_score)`` pairs, best first. FTS5's BM25 score is
            negative and more negative means a better match.
        """
        if k <= 0:
            return []
        match = escape_fts_query(query)
        if not match:
            return []

        predicate, params = compile_where(where, column="d.metadata")
        source_clause = " AND d.source = ?" if source else ""
        source_params = [source] if source else []
        with self._lock:
            try:
                rows = self.db.execute(
                    f"""
                    SELECT d.id, bm25(documents_fts) AS score
                    FROM documents_fts
                    JOIN documents d ON d.id = documents_fts.rowid
                    WHERE documents_fts MATCH ? AND {predicate}{source_clause}
                    ORDER BY score
                    LIMIT ?
                    """,
                    (match, *params, *source_params, k),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                # A malformed MATCH should degrade to "no keyword hits", never
                # take down a query that vector search could still answer.
                log.warning("keyword search failed for %r: %s", query, exc)
                return []
        return [(int(i), float(s)) for i, s in rows]

    def _filter_ids(
        self, where: Where | None, source: str | None, *, limit: int
    ) -> list[int]:
        predicate, params = compile_where(where, column="d.metadata")
        source_clause = " AND d.source = ?" if source else ""
        source_params = [source] if source else []
        rows = self.db.execute(
            f"SELECT d.id FROM documents d WHERE {predicate}{source_clause} LIMIT ?",
            (*params, *source_params, limit),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def fetch(self, ids: Sequence[int]) -> dict[int, Hit]:
        """Load documents by id, keyed by id (order is the caller's business)."""
        if not ids:
            return {}
        out: dict[int, Hit] = {}
        with self._lock:
            for batch in _batched(list(ids), 500):
                placeholders = ",".join("?" for _ in batch)
                rows = self.db.execute(
                    f"SELECT id, text, source, chunk_index, metadata FROM documents "
                    f"WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                for doc_id, text, source, index, metadata in rows:
                    out[int(doc_id)] = Hit(
                        id=int(doc_id),
                        text=text,
                        score=0.0,
                        source=source or "",
                        index=int(index),
                        metadata=json.loads(metadata) if metadata else {},
                    )
        return out

    def neighbors(self, source: str, index: int, *, radius: int = 1) -> list[Hit]:
        """Chunks adjacent to ``index`` within the same source.

        Used to widen a hit into its surrounding context without inflating the
        number of chunks that had to be ranked.
        """
        if radius <= 0:
            return []
        with self._lock:
            rows = self.db.execute(
                "SELECT id, text, source, chunk_index, metadata FROM documents "
                "WHERE source = ? AND chunk_index BETWEEN ? AND ? "
                "ORDER BY chunk_index",
                (source, index - radius, index + radius),
            ).fetchall()
        return [
            Hit(
                id=int(doc_id),
                text=text,
                score=0.0,
                source=src or "",
                index=int(idx),
                metadata=json.loads(meta) if meta else {},
            )
            for doc_id, text, src, idx, meta in rows
        ]

    def sources(self, *, limit: int | None = None) -> list[SourceInfo]:
        """Every source currently indexed, most recently updated first."""
        sql = (
            "SELECT source, chunks, characters, added_at, metadata FROM sources "
            "ORDER BY updated_at DESC, rowid DESC"
        )
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self.db.execute(sql, params).fetchall()
        return [
            SourceInfo(
                source=source,
                chunks=int(chunks),
                characters=int(characters),
                added_at=added_at,
                metadata=json.loads(metadata) if metadata else {},
            )
            for source, chunks, characters, added_at, metadata in rows
        ]

    def has_source(self, source: str, *, content_hash: str | None = None) -> bool:
        """Whether ``source`` is indexed, optionally with that exact content."""
        with self._lock:
            row = self.db.execute(
                "SELECT content_hash FROM sources WHERE source = ?", (source,)
            ).fetchone()
        if row is None:
            return False
        return content_hash is None or row[0] == content_hash

    def count(self) -> int:
        """Number of indexed chunks."""
        with self._lock:
            return int(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def stats(self) -> Stats:
        """A summary of the index."""
        with self._lock:
            documents = self.count()
            sources = int(self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        size = 0
        if str(self.path) != ":memory:" and self.path.exists():
            size = self.path.stat().st_size
            for suffix in ("-wal", "-shm"):
                sidecar = self.path.with_name(self.path.name + suffix)
                if sidecar.exists():
                    size += sidecar.stat().st_size
        return Stats(
            path=str(self.path),
            documents=documents,
            sources=sources,
            dimensions=self._dimensions,
            size_bytes=size,
            schema_version=SCHEMA_VERSION,
        )

    # -- transactions ------------------------------------------------------- #

    class _Transaction:
        def __init__(self, db: sqlite3.Connection) -> None:
            self.db = db
            self.owns = False

        def __enter__(self) -> Store._Transaction:
            if not self.db.in_transaction:
                self.db.execute("BEGIN IMMEDIATE")
                self.owns = True
            return self

        def __exit__(self, exc_type: object, *_: object) -> None:
            if not self.owns:
                return
            if exc_type is None:
                self.db.execute("COMMIT")
            else:
                self.db.execute("ROLLBACK")

    def _transaction(self) -> Store._Transaction:
        return Store._Transaction(self.db)


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _batched(items: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
