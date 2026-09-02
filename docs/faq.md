# FAQ

Practical questions about running softrag.

- [How big can an index get?](#how-big-can-an-index-get)
- [Is it thread safe?](#is-it-thread-safe)
- [Can several processes use one index?](#can-several-processes-use-one-index)
- [How do I back it up?](#how-do-i-back-it-up)
- [How do I move an index between machines?](#how-do-i-move-an-index-between-machines)
- [What happens if I change embedding models?](#what-happens-if-i-change-embedding-models)
- [Why is my retrieval bad?](#why-is-my-retrieval-bad)
- [Can I use it without any API key?](#can-i-use-it-without-any-api-key)
- [Can I query without generating?](#can-i-query-without-generating)
- [How do I change the prompt?](#how-do-i-change-the-prompt)
- [Which errors should I catch?](#which-errors-should-i-catch)
- [Can I use it with async code?](#can-i-use-it-with-async-code)
- [Why is the file bigger than my documents?](#why-is-the-file-bigger-than-my-documents)
- [Can I read the database with plain SQL?](#can-i-read-the-database-with-plain-sql)

## How big can an index get?

SQLite's own limits are not the constraint (a database can reach 281 TB).
Vector search is what sets the practical ceiling, because `vec0` does an exact
scan: search time grows linearly with the number of chunks.

| Chunks           | Experience                                                      |
| ---------------- | --------------------------------------------------------------- |
| up to ~100k      | Comfortable. Searches are milliseconds.                          |
| ~100k – 1M       | Fine. Latency is noticeable but small; raise `cache_size` if the file no longer fits in page cache. |
| 1M – 5M          | Works, but you are paying for a full scan per query. Consider narrowing with `source=`/`where=` filters, or shortened embeddings. |
| beyond ~10M      | Use a real vector database with an ANN index. softrag is the wrong tool here and does not pretend otherwise. |

At 1M chunks with 1536-dimensional vectors the file is roughly 6 GB, most of it
vectors. Smaller embeddings help proportionally — see
[storage cost](architecture.md#storage-cost).

## Is it thread safe?

Yes, within one process. The connection is opened with
`check_same_thread=False` and every write goes through a re-entrant lock and a
`BEGIN IMMEDIATE` transaction, so a single `Rag` object can be shared across
threads. That is exactly how `add_many()` works: extraction and embedding run in
a thread pool while writes serialise at the store.

Reads are concurrent. Writes serialise — with one writer at a time, throughput
is bounded by SQLite, not by softrag.

What is *not* safe is bypassing the store: writing to `documents` through
`rag.store.db` on your own leaves the FTS and vector tables inconsistent.

## Can several processes use one index?

For reading, yes. WAL mode lets many reader processes work against one file
while a writer proceeds, which makes the "one process builds the index, N
processes serve queries" pattern straightforward.

For writing, one at a time. A second writer waits up to the 30-second
`busy_timeout` and then raises. If you need many writers, funnel ingestion
through one process — the usual shape is a worker that owns writes and readers
that only query.

Network filesystems (NFS, SMB) are a known SQLite hazard: its locking primitives
are not reliable there. Keep the file on local storage.

## How do I back it up?

Copy the file.

```bash
cp kb.db kb-backup.db
```

The only caveat is WAL: while a database is open you may also see `kb.db-wal`
and `kb.db-shm` beside it, and a copy of `kb.db` alone can miss recent commits.
Two safe options:

```python
rag.close()          # checkpoints and removes the sidecars; then copy kb.db
```

```bash
sqlite3 kb.db ".backup kb-backup.db"     # consistent online backup, no downtime
```

Because an index is one file, everything you already do with files applies:
snapshot it, rsync it, put it in object storage, ship it in a container image,
version it with git-lfs if it is small enough to be worth it.

## How do I move an index between machines?

Copy the file and use the same embedding model on the other end. The vectors are
portable — plain little-endian float32 — and so is the schema; what is not
portable is the *meaning* of the vectors, which belongs to the model that
produced them. Querying with a different model gives you either a
`DimensionMismatchError` (if the width differs) or, worse, silently nonsense
rankings (if the widths happen to match).

Record what built the index, and check it on the other side:

```python
rag.add_text("...", metadata={"embed_model": "text-embedding-3-small"})
print(rag.stats().dimensions)     # the width the file is pinned to
```

The receiving machine needs `sqlite-vec` installed and a Python built with
SQLite extension loading. No server, no migration, no export step.

## What happens if I change embedding models?

The index is pinned to the width of the first vector ever written to it. Hand it
a different width and you get:

```
DimensionMismatchError: This database stores 1536-dimensional vectors but the
embedder returned 384 dimensions. You are most likely using a different
embedding model than the one the index was built with. Either switch back to the
original model or re-index into a new database file.
```

That is the *good* case — it fails loudly. The dangerous case is two different
models that happen to share a width (many are 768 or 1024). Nothing errors, and
retrieval quietly returns garbage, because query vectors from model B are being
compared against document vectors from model A. There is no way for softrag to
detect this, so the discipline is yours: one embedding model per database file.

Changing models means re-indexing:

```python
import softrag

old = softrag.connect("kb.db", embed_model=old_embedder)
new = softrag.connect("kb-v2.db", embed_model=new_embedder)

for info in old.sources():
    new.add(info.source, metadata=info.metadata)   # re-extract and re-embed

old.close(); new.close()
```

This works when your sources are still reachable (paths, URLs). If they are not,
read the chunk text straight out of the old index and re-embed it:

```python
rows = old.store.db.execute(
    "SELECT source, text, metadata FROM documents ORDER BY source, chunk_index"
).fetchall()
```

Chunk *size* is not pinned, only width — you can re-chunk any time by re-adding
the source with a different `chunker` or `chunk_size`.

## Why is my retrieval bad?

Run `rag.search()` on the failing question. It returns exactly what `query()`
would have shown the model, so it settles in one call whether the problem is
retrieval or generation. The full diagnostic checklist is in
[retrieval.md](retrieval.md#is-it-retrieval-or-generation); the short version,
in order of how often each is the cause:

1. **`HashEmbedder` is in use.** If `rag.stats().dimensions == 256` and you never
   configured an embedder, softrag warned once at startup and fell back to it.
   It has no notion of synonymy. Configure a real embedder and re-index.
2. **The content was never indexed.** Check `rag.sources()` and `len(rag)`, and
   check the `IngestResult` objects you ignored — a scanned PDF or a
   JavaScript-rendered page fails quietly into `result.error`.
3. **`candidates` is too low** for hybrid fusion to see the right document. Try
   `candidates=200`.
4. **The chunks are too big.** A 4000-character chunk dilutes its own embedding
   until it matches nothing in particular.
5. **The wrong mode.** Rare literal tokens want `mode="keyword"`; paraphrases
   want `mode="vector"`.

## Can I use it without any API key?

Two ways. Fully local and actually good:

```python
from softrag.providers.ollama import OllamaEmbedder, OllamaChat
rag = softrag.connect("kb.db",
    embed_model=OllamaEmbedder("nomic-embed-text"),
    chat_model=OllamaChat("llama3.2"))
```

Or dependency-free and deliberately mediocre, for tests and demos:

```python
rag = softrag.Rag(db_path=":memory:",
    embed_model=softrag.HashEmbedder(),
    chat_model=softrag.EchoChatModel())
```

`EchoChatModel` returns the prompt rather than generating, which makes it a
useful diagnostic: everything you see came out of the index.

## Can I query without generating?

Yes — that is `search()`, and it never touches a chat model. The chat model is
resolved lazily, so a retrieval-only engine that never configures one costs
nothing and errors only if you call `query()`.

```python
rag = softrag.Rag(db_path="kb.db", embed_model=my_embedder, auto=False)
hits = rag.search("query")          # fine
rag.query("query")                  # ConfigurationError
```

## How do I change the prompt?

The template needs `{context}` and `{question}` placeholders.

```python
import softrag
from softrag import DEFAULT_PROMPT

MY_PROMPT = """You are a terse support assistant. Answer only from the context.
Cite blocks as [1], [2]. If the context does not answer the question, say
"I don't know" and nothing else.

Context:
{context}

Question: {question}

Answer:"""

rag = softrag.connect("kb.db", prompt=MY_PROMPT)     # for every query
rag.query("...", prompt=MY_PROMPT)                   # just this one
print(DEFAULT_PROMPT)                                # the built-in one
```

Context blocks arrive numbered and attributed — `[1] (handbook.pdf)` — which is
what makes `[1]`-style citations possible and what lets the model say where
something came from. `answer.prompt` shows exactly what was sent.

## Which errors should I catch?

Everything softrag raises derives from `SoftragError`, so one `except` covers
the whole surface:

```
SoftragError
├── ConfigurationError          bad options, missing model, bad filter operator
│   └── MissingDependencyError  names the extra to install
├── StoreError                  SQLite layer
│   ├── SchemaVersionError      file written by an incompatible version
│   └── DimensionMismatchError  wrong embedding width for this index
├── IngestionError              could not turn input into text
│   ├── UnsupportedFormatError  no extractor for this extension
│   └── ExtractionError         extractor found, but it failed on this file
└── ProviderError               a backend misbehaved
    ├── EmbeddingError
    └── ChatError
```

```python
from softrag import SoftragError, IngestionError

for path in paths:
    try:
        rag.add(path)
    except IngestionError as exc:
        log.warning("skipping %s: %s", path, exc)
    except SoftragError:
        raise
```

In bulk methods this is already done for you: `add_many()` and
`add_directory()` default to `ignore_errors=True` and record failures in the
returned `IngestResult` list.

## Can I use it with async code?

There is no async API. SQLite operations are fast and local; the parts that
genuinely block are the embedding and chat calls, and those belong to your
backend. In an async application, run softrag calls in a thread:

```python
import asyncio

answer = await asyncio.to_thread(rag.query, "What changed?")
hits = await asyncio.to_thread(rag.search, "refunds", top_k=5)
```

The store is thread-safe, so this is fine from several tasks at once.

## Why is the file bigger than my documents?

Because it holds three representations: the chunk text, a full-text index over
it (~30–50% of the text), and a float32 vector per chunk (`4 × dimensions`
bytes). With 1536-dimensional embeddings the vectors alone are usually the
largest component — see [storage cost](architecture.md#storage-cost).

To shrink it: use shortened embeddings
(`OpenAIEmbedder(dimensions=512)`) or a smaller local model, reduce overlap, and
call `rag.optimize()` after large deletions to compact the FTS index and reclaim
free pages. `rag.stats().size_mb` reports the current size including the WAL
sidecar, so a number that looks inflated right after a big ingest usually
shrinks after a checkpoint.

## Can I read the database with plain SQL?

Yes, for reading. It is an ordinary SQLite file.

```bash
sqlite3 kb.db "SELECT source, COUNT(*) FROM documents GROUP BY source"
sqlite3 kb.db "SELECT source, chunk_index, substr(text,1,60) FROM documents LIMIT 5"
sqlite3 kb.db "PRAGMA user_version"     # schema version
```

Querying the `vectors` table needs the sqlite-vec extension loaded
(`.load vec0` in the shell, or use `rag.store.db`, which already has it).

Writing by hand is a different matter: inserting into `documents` outside the
store leaves the vector table without a row and, if you bypass the triggers, the
FTS index inconsistent. Use the Python API for writes.
