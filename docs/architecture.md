# Architecture

What is inside the `.db` file, and why it looks like that.

- [Layers](#layers)
- [The schema](#the-schema)
- [Why the FTS5 triggers are mandatory](#why-the-fts5-triggers-are-mandatory)
- [How vector search works](#how-vector-search-works)
- [Filtered vector search](#filtered-vector-search)
- [Connection settings](#connection-settings)
- [Schema versions and migrations](#schema-versions-and-migrations)
- [Storage cost](#storage-cost)

## Layers

```
              softrag.connect(...)
                      │
                  engine.py          Rag: ingest → chunk → embed → store; search → prompt → generate
             ┌────────┼────────┬─────────────┐
             │        │        │             │
       chunking.py  ingest/  providers/   retrieval.py    RRF, MMR, context expansion
       str→chunks   bytes→   anything→          │
                    text     protocol           │
                                                ▼
                                            store.py      the only module that writes SQL
                                                │
                                          SQLite + sqlite-vec + FTS5
```

The rule that keeps this honest: **`store.py` owns the schema, the migrations
and every SQL statement.** Nothing above it constructs SQL, and the one place
user values reach a query — the metadata filter — compiles to bound parameters
in `filters.py`, never to interpolated text.

## The schema

Five objects, all in one file.

```sql
CREATE TABLE softrag_meta (            -- created_at, distance_metric, dimensions
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sources (                 -- one row per indexed document
    source       TEXT PRIMARY KEY,     -- path, URL, or an explicit name
    content_hash TEXT NOT NULL,        -- SHA-256 of the extracted text
    characters   INTEGER NOT NULL DEFAULT 0,
    chunks       INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}',
    added_at     TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE documents (               -- one row per chunk
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL REFERENCES sources(source) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,      -- position within the source, from 0
    text        TEXT NOT NULL,
    hash        TEXT NOT NULL,         -- SHA-256 of the chunk text
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE INDEX        idx_documents_source ON documents(source, chunk_index);
CREATE UNIQUE INDEX idx_documents_dedup  ON documents(source, hash);

CREATE VIRTUAL TABLE documents_fts USING fts5(
    text,
    content='documents',               -- external content: no duplicated text
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE VIRTUAL TABLE vectors USING vec0(   -- created on the first embedding write
    doc_id    INTEGER PRIMARY KEY,
    embedding float[N] distance_metric=cosine,
    source    TEXT                     -- denormalised so source-scoped KNN is cheap
);
```

A few of these choices are load-bearing:

- **`documents` is the source of truth.** FTS5 stores no text of its own
  (external content) and `vectors` stores no text at all, so a chunk's text
  exists once.
- **`UNIQUE(source, hash)`** makes re-adding an unchanged chunk a no-op — that
  is what "re-ingest is free" is built on. It is scoped to the source on
  purpose: the same paragraph in two files is stored under each, so deleting
  one file can never blank the other.
- **`content_hash` on `sources`** is what lets `add()` decide in one comparison
  whether a document changed at all, before any embedding call is made.
- **`distance_metric=cosine` is set explicitly** because `vec0` defaults to L2,
  and L2 over unnormalised embeddings ranks by vector magnitude as much as by
  direction.
- **`remove_diacritics 2`** means `café` and `cafe` match in keyword search.

## Why the FTS5 triggers are mandatory

An external-content FTS5 table is an index over another table's rows; it holds
positions and doc-ids, not text. SQLite does not keep it in sync for you.
Without triggers, deleting a row from `documents` leaves index entries pointing
at rows that no longer exist — and FTS5 does not degrade gracefully there. The
*next* query fails outright:

```
sqlite3.DatabaseError: database disk image is malformed
fts5: missing row 42 from content table 'documents'
```

That is a whole-index failure caused by one delete, which is why deletion could
not be supported at all before the triggers existed. softrag creates all three
with the schema:

```sql
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
```

The `'delete'` command form is not optional either: it must be given the *old*
text so FTS5 can subtract exactly the terms it once added. This is also why you
should never `INSERT INTO documents` by hand through a raw connection unless you
are prepared to maintain the index yourself.

## How vector search works

Embeddings are stored as little-endian packed `float32` — 4 bytes per dimension —
in a `vec0` virtual table. An unfiltered search uses `vec0`'s native KNN
operator:

```sql
SELECT doc_id, distance FROM vectors
WHERE embedding MATCH ? AND k = ?
```

`vec0` scans the vectors and keeps the best `k` in a bounded heap. This is
*exact* nearest-neighbour search, not an approximate index: there is no
recall/latency trade-off to tune and no index build step, at the cost of work
linear in the number of chunks. That is the right trade for a file-sized index
and the wrong one at tens of millions of vectors.

Cosine distance is in `[0, 2]`; smaller is more similar. `mode="vector"` reports
`1 - distance/2`, so a score of 1.0 is identical and 0.0 is opposite.

Keyword search is a straightforward FTS5 `MATCH` with BM25 ordering, joined back
to `documents` for the metadata predicate. Raw user text never reaches `MATCH`:
each word is extracted and re-quoted as a phrase, because `AND`, `OR`, `NOT`,
`NEAR`, `*`, `:`, `^`, `(` and `"` are all FTS5 syntax and an ordinary question
containing the word "not" would otherwise be a syntax error. A malformed match
is caught and degraded to "no keyword hits" rather than failing a query that
vector search could still answer.

## Filtered vector search

This is where most embedded vector stores quietly lose recall. The naive
implementation is *post-filtering*: ask for `k * 8` approximate neighbours, drop
the ones that fail the filter, return what is left. When the filter is
selective — 40 matching chunks out of a million — the over-fetched neighbours
contain none of them, and you get an empty result for a query that had a perfect
answer.

softrag inverts the order when it can:

```
                    where= given?
                 ┌────── no ──────► vec0 native KNN            (fast path)
                 │
   search_vector ┤        yes
                 │          │
                 └──────────┴─► resolve the filter on documents (indexed, cheap)
                                        │
                              matching rows ≤ 1,000 ?
                                 ┌─── yes ───► vec_distance_cosine per row,
                                 │             sort, take k            (exact)
                                 └─── no  ───► KNN over-fetch k*8, then filter
                                                                  (approximate)
```

The filter resolves against `documents`, which is a normal indexed B-tree, so
finding the matching ids is cheap. Below the 1,000-row threshold you get the
true best `k` among the matching rows — a selective filter costs no recall at
all. Above it, the filter is not selective enough for over-fetching to miss
much, and exact rescoring would cost more than it is worth.

The threshold is measured, not guessed, and the shape of the measurement is
worth knowing because it is counter-intuitive. Exact rescoring uses **one point
lookup per row**, which looks wasteful next to collecting the ids and asking for
`doc_id IN (...)`. But `vec0` answers an `IN` with a full table scan
(`EXPLAIN QUERY PLAN` says `SCAN vectors VIRTUAL TABLE INDEX 0:1`), while an
equality on the primary key is a real lookup. On 20k 384-dimension vectors:

| strategy | 2,000 rows |
| --- | ---: |
| native KNN, k=20 (whole corpus) | 7.4 ms |
| `IN` batches of 500 | 29.7 ms |
| a single `IN` of 2,000 ids | 17.6 ms |
| 2,000 point lookups | 16.5 ms |
| 100 point lookups | 0.8 ms |

Point lookups win decisively while the filter is selective and lose to native
KNN somewhere around a thousand rows, which is where the threshold sits.

`source="handbook.pdf"` is a cheaper special case: `source` is a real column on
`vectors`, so it is pushed into the KNN query directly with no JSON extraction.

Metadata filters themselves compile to `json_extract(d.metadata, ?) op ?` with
every value bound as a parameter, so a filter built from user input cannot
inject SQL.

## Connection settings

Set on every connection, in `Store._configure`:

| Pragma                 | Value              | Why                                                        |
| ---------------------- | ------------------ | ---------------------------------------------------------- |
| `journal_mode`         | `WAL` (file DBs)   | Readers do not block the writer, and vice versa.            |
| `synchronous`          | `NORMAL`           | The usual WAL trade: durable across process crashes, a power loss can lose the last commits. |
| `foreign_keys`         | `ON`               | `ON DELETE CASCADE` from `sources` to `documents`.          |
| `busy_timeout`         | 30 s               | Wait for a competing writer instead of failing immediately. |
| `temp_store`           | `MEMORY`           | Sorting and temp b-trees stay off disk.                     |
| `cache_size`           | ~32 MB             | Page cache large enough for the working set of a mid-sized index. |
| `page_size`            | 8192 (new DBs)     | Chunk rows are ~1 KB; larger pages mean fewer of them. Only settable before the first table exists. |

Writes go through `BEGIN IMMEDIATE` and are serialised by a re-entrant lock, so
the connection is safe to share between threads (`check_same_thread=False`) and
`add_many()` can extract and embed concurrently without racing the writer.

`optimize()` runs FTS5's `'optimize'` command, `PRAGMA optimize` and `VACUUM`.
It is never required for correctness — call it after a large ingest or a lot of
deletions.

## Schema versions and migrations

The schema version lives in `PRAGMA user_version`, and the current version is
**1**.

- Opening a file with **no softrag schema** creates it (unless the store is
  read-only, which is an error rather than a silent empty index).
- Opening a file whose version **differs** raises `SchemaVersionError`, which
  says whether the file is newer than the library (upgrade softrag) or older
  (migrate or re-index).

There is only one version so far, so there is nothing to migrate *from* yet.
When version 2 arrives, the mechanism is in place: `_migrate()` reads
`user_version`, applies the steps, and stamps the new value.

The **embedding width** is separate from the schema version and is pinned by the
first vector written — stored in `softrag_meta` and baked into the `vectors`
table declaration (`float[N]`). Handing that index a differently sized embedding
raises `DimensionMismatchError` rather than corrupting it. See the
[FAQ](faq.md#what-happens-if-i-change-embedding-models).

## Storage cost

Per chunk, roughly:

| Component                   | Size                                       |
| --------------------------- | ------------------------------------------ |
| Chunk text                  | the text itself (~1 KB at default settings) |
| Embedding                   | `4 × dimensions` bytes                     |
| FTS5 index                  | ~30–50% of the text                        |
| Row overhead and metadata   | ~100–300 bytes                             |

So with `text-embedding-3-small` at its full 1536 dimensions, a 1 KB chunk costs
about 6 KB on disk, of which the vector is the largest part. Two levers if that
matters: shortened embeddings (`OpenAIEmbedder(dimensions=512)` → 2 KB per
vector) or a smaller local model (MiniLM's 384 dimensions → 1.5 KB). Both must
be chosen before the first ingest.

`rag.stats()` reports the real number, including the WAL sidecar:

```python
s = rag.stats()
print(s.documents, "chunks /", s.sources, "sources /", f"{s.size_mb:.1f} MB",
      "/ dim", s.dimensions, "/ schema v", s.schema_version)
```
