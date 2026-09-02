# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

A rewrite. 0.1.x was a single module whose hybrid retrieval could not work as
written and whose index was hard-coded to one embedding model's vector width.
This release replaces it with a layered engine, and cuts the required
dependencies from nine to one.

### Fixed

Each of these was reproduced before it was fixed.

- **Keyword search crashed on ordinary questions.** Query terms were joined
  with `" OR "` without escaping, so anything containing `AND`, `OR` or `NOT`
  as a word raised `fts5: syntax error near "NOT"`. Terms are now quoted as
  phrases.
- **Hybrid fusion mixed incomparable scales.** BM25 scores (negative and
  unbounded) were combined with cosine similarity (0..1) through
  `1.0/(bm25+1)`, which divides by zero at `bm25 = -1` and returns negative
  scores below it. Fusion is now Reciprocal Rank Fusion over ranks, which needs
  no calibration between the two.
- **The ranking was discarded anyway.** The outer
  `SELECT ... WHERE id IN (SELECT ... ORDER BY score)` had no `ORDER BY` of its
  own, so results came back in rowid order regardless of relevance.
- **Keyword noise outranked correct results.** OR-ing every token meant *"when
  can we ship code to customers"* matched whatever contained *can*, *we* and
  *to*. Stopwords and terms above a document-frequency cutoff are now dropped
  before the query is built.
- **The index was fixed at 1536 dimensions.** Any other embedding model
  silently produced a corrupt index. The width is now learned from the first
  embedding, and a mismatch raises `DimensionMismatchError`.
- **Vector search scanned and sorted every row in SQL.** It now uses `vec0`'s
  native KNN operator, with `distance_metric=cosine` set explicitly, since
  `vec0` defaults to L2.
- Filtered vector search was slower than unfiltered search, because `vec0`
  answers `doc_id IN (...)` with a full table scan. Measured at 20k chunks:
  75.6 ms before, 28.0 ms now.
- `RecursiveChunker` could return chunks larger than `chunk_size`, up to twice
  it after a hard split.
- `MarkdownChunker` repeated every heading inside its own section's chunks.

### Added

- `search()` — retrieval without a chat model, so you can see exactly what an
  answer would have been built from.
- Metadata filtering with a dict DSL: `$eq $ne $gt $gte $lt $lte $in $nin
  $like $contains $exists $and $or $not`, compiled to parameterised SQL.
- Deletion and updates: `delete(source=...)`, `delete(where=...)`, `reset()`.
  These were impossible before, because the FTS index had no sync triggers and
  a delete left it pointing at missing rows.
- Idempotent ingestion. Re-adding an unchanged source is free; re-adding
  changed content replaces the old chunks instead of duplicating them.
- `add_text()`, `add_directory()`, `add_many()` with threaded ingestion,
  `sources()`, `stats()`, `optimize()`, `len(rag)`, context-manager support.
- `AsyncRag`, mirroring `Rag` method for method.
- A CLI: `softrag add|search|query|ls|rm|stats|optimize|shell|doctor`, with
  JSON output and streamed answers. `--provider hash` runs it with no key, no
  network and no model download.
- Rerankers: cross-encoder, LLM, Cohere, score-fusion, chain and near-duplicate.
- `softrag.eval` — recall, precision, MRR, nDCG and MAP with a trec_eval-shaped
  interface, plus `compare()` for A/B-ing retrieval settings on your own corpus.
- Query transforms: HyDE, multi-query expansion, and Anthropic's contextual
  retrieval as an opt-in ingest stage.
- MMR diversity, context expansion to neighbouring chunks, and per-call search
  overrides.
- Chunking strategies: recursive (default), Markdown heading-aware, sentence,
  and any callable.
- Format support without third-party parsers: DOCX, PPTX, XLSX and EPUB through
  stdlib `zipfile`, HTML through `html.parser`, plus CSV, TSV, JSON, JSON Lines,
  Markdown and source code. Only PDF needs an extra.
- Backends: OpenAI, Anthropic, Ollama (over plain HTTP, no SDK) and
  sentence-transformers, plus adapters for LangChain objects, Chroma-style
  embedding functions and bare callables.
- A warning when an index is reopened with a different embedding model of the
  same vector width — the failure mode dimension checking cannot catch.
- `benchmarks/`, with throughput, latency and retrieval-quality measurements.
- A test suite, and CI across Python 3.10–3.13 on Linux, macOS and Windows.

### Changed

- **`query()` returns an `Answer`**, a `str` subclass carrying `.hits`,
  `.sources` and `.context`. It previously returned whatever the chat backend
  gave back, typically a LangChain message object that had to be unwrapped.
- **The core install has one dependency**, `sqlite-vec`. Everything else moved
  to extras: `openai`, `anthropic`, `local`, `rerank`, `cohere`, `files`,
  `web`, `cli`, `all`.
- Default chunk size is 1000 characters with 200 of overlap, up from 400/100.
- `Rag()` now works with no arguments, detecting backends from the environment.
- Minimum Python is 3.10, down from 3.12.
- Examples moved out of the package directory to `examples/`.

### Removed

- `langchain-text-splitters`, `langchain-openai`, `llama-index-readers-file`,
  `docx2txt`, `pymupdf`, `six` and `trafilatura` as required dependencies.
- The `dotenv` dependency, which was the abandoned placeholder package rather
  than `python-dotenv`.

### Migration from 0.1.x

`Rag(embed_model=..., chat_model=..., db_path=...)`, `add_file`, `add_web`,
`add_image` and `query` all still work. Three things to know:

1. **Re-index.** The schema changed and 0.1.x databases are not readable.
   Delete the old `.db` file and ingest again.
2. **`query()` returns a string now.** Code doing `rag.query(q).content` should
   drop the `.content`; `print(rag.query(q))` was already correct.
3. **Install the extras you use.** `pip install softrag` no longer pulls
   LangChain or OpenAI. Use `pip install 'softrag[openai]'`, or
   `'softrag[all]'` to get everything.

## [0.1.4]

Initial published releases.
