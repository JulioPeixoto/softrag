<div align="center">
  <img src="piriquito.png" width="150" alt="softrag mascot, a periquito"/>

  <h1>softrag</h1>

  <p><strong>The embedded RAG engine: hybrid retrieval in a single SQLite file, no server, no vendor account.</strong></p>

  <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="https://pypi.org/project/softrag/"><img src="https://img.shields.io/pypi/v/softrag.svg" alt="PyPI version"></a>
  </p>
</div>

SQLite is what you reach for when a database should be a file. softrag is that,
for retrieval: chunk text, metadata, the vector index and the full-text index
all live in one `.db` file you can copy, commit, ship, back up with `cp`, or
delete. The core install is one dependency (`sqlite-vec`); every model backend
is optional and pluggable, including fully local ones.

## Quickstart

```bash
pip install softrag
```

```python
import softrag

rag = softrag.connect("kb.db")          # picks a backend up from the environment
rag.add("handbook.pdf")                 # a file
rag.add("https://example.com/changelog")  # a URL
rag.add("Refunds are processed within 5 business days.")  # raw text

answer = rag.query("How long do refunds take?")
print(answer)            # Answer is a str, so this just works
print(answer.sources)    # ['handbook.pdf', ...]
```

`connect()` detects models from the environment (`OPENAI_API_KEY`, a running
Ollama daemon, an installed `sentence-transformers`) and falls back to a
dependency-free hash embedder with a loud warning, so the snippet above runs
even with nothing configured. To be explicit:

```python
from softrag.providers.openai import OpenAIChat, OpenAIEmbedder

rag = softrag.connect(
    "kb.db",
    embed_model=OpenAIEmbedder("text-embedding-3-small"),
    chat_model=OpenAIChat("gpt-4.1-mini"),
)
```

## Why softrag

|                          | softrag                       | Chroma                | LanceDB                  | LlamaIndex                    | raw sqlite-vec        |
| ------------------------ | ----------------------------- | --------------------- | ------------------------ | ----------------------------- | --------------------- |
| Infrastructure           | none — a file                 | embedded or server    | none — a directory       | none, but BYO store           | none — a file         |
| Core install             | 1 package (`sqlite-vec`)      | ~30 transitive deps   | ~10 transitive deps      | large framework + integrations| 1 package             |
| Hybrid dense + BM25      | built in, RRF-fused           | vectors only          | full-text index, opt-in  | via a retriever you assemble  | you write it          |
| Metadata filtering       | built in (`$gte`, `$in`, …)   | yes                   | SQL predicates           | yes, per store                | you write the SQL     |
| Works fully offline      | yes (Ollama / local models)   | yes                   | yes                      | depends on the pieces         | yes                   |
| Index portability        | one file, `cp` it             | a directory           | a directory              | n/a                           | one file              |
| Document extraction      | built in (PDF, Office, web…)  | no                    | no                       | yes, extensive                | no                    |

**Where softrag is the wrong tool.** It is a single-file embedded index, not a
distributed store: there is no sharding, no replication, no multi-writer
cluster. Vector search is exact (a full scan under `vec0`), which is excellent
up to a few hundred thousand chunks and the wrong shape at 50M+ vectors — reach
for a real vector database there. Concurrent readers are fine (WAL); concurrent
*writers* serialise on SQLite's write lock. And LlamaIndex covers far more
document connectors and agent patterns than softrag intends to.

**Where it fits.** A knowledge base that ships with your app. A CLI that answers
questions about a repo. A desktop tool that must work on a plane. A test suite
that needs a real index without standing up a service. Anything where "add a
vector database" is a bigger commitment than the feature is worth.

## How retrieval works

Dense vectors find text that *means* the right thing; BM25 finds text that
*says* the right thing — the error code, the product SKU, the surname. Neither
one alone is enough, so softrag runs both and fuses the two ranked lists with
**Reciprocal Rank Fusion**. RRF works on ranks, not scores, which matters: FTS5's
BM25 score is negative and unbounded while cosine distance is bounded in `[0, 2]`,
so adding or comparing them directly lets whichever scale happens to be larger
dominate for reasons unrelated to relevance. Throwing the magnitudes away and
keeping only the ordering needs no calibration at all.

```
query ──┬─► embed ─► vec0 KNN ────► [12, 4, 7, 3, …]  ┐
        │                                             ├─► RRF ─► top_k ─► (MMR) ─► (expand) ─► hits
        └─► FTS5 MATCH (BM25) ───► [4, 19, 12, 8, …]  ┘

              score(doc) = Σ  weight_i / (k + rank_i(doc))      k = 60
```

`k = 60` is the constant from the original RRF paper (Cormack et al., 2009). It
damps the influence of the very top ranks just enough that one retriever cannot
veto the other: a document ranked #1 by vectors and unranked by BM25 does not
automatically beat one both retrievers put in their top five.

Each retriever contributes `candidates` documents before fusion
(`max(4 * top_k, 20)` by default) — over-fetching is cheap and helps fusion a
lot. Optional passes follow: MMR for diversity, then neighbouring-chunk
expansion. See [docs/retrieval.md](docs/retrieval.md) for the full treatment.

## Core API

### Ingestion

```python
rag.add("report.pdf")                     # dispatches on the input: path, URL or text
rag.add_file("notes.docx", metadata={"team": "legal"})
rag.add_web("https://example.com/post", metadata={"year": 2025})
rag.add_text("Raw string content", name="policy-v3")
rag.add_image("architecture.png")         # captioned by a vision model, then indexed as text
rag.add_directory("./docs", pattern="**/*.md", exclude=("**/drafts/**",))
rag.add_many(["a.pdf", "b.md", "https://example.com"], max_workers=8)
```

Every `add*` call returns an `IngestResult` telling you what actually happened:

```python
result = rag.add("handbook.pdf")
print(result.chunks_added, result.chunks_skipped, result.chunks_deleted)
if not result:                            # falsy when result.error is set
    print(result.error)
```

Re-adding an unchanged source is a no-op. Re-adding *changed* content replaces
the old chunks by default; `on_change="skip"` keeps the old version and
`on_change="append"` keeps both.

### Retrieval

```python
hits = rag.search(
    "refund window",
    top_k=8,
    mode="hybrid",                        # or "vector" / "keyword"
    where={"team": "legal", "year": {"$gte": 2024}},
    diversity=0.4,                        # MMR: 0 = pure relevance, 1 = maximally different
    expand_context=1,                     # glue on the chunk before and after each hit
)

for hit in hits:
    print(f"{hit.score:.4f}  {hit.source}#{hit.index}  {hit.ranks}")
    print(hit.text[:200])
```

`search()` calls no chat model. Whatever it returns is exactly what a generated
answer would have been built from, which makes it the honest way to tell a
retrieval problem apart from a generation problem.

### Generation

```python
answer = rag.query("What changed in the refund policy?", top_k=5)
print(answer)               # Answer subclasses str
print(answer.sources)       # unique source ids, best-scoring first
print(answer.hits)          # the Hit objects behind it
print(answer.context)       # the context exactly as the model saw it
print(answer.prompt)        # the fully rendered prompt

# Streaming
stream = rag.query("Summarise the changelog", stream=True)
for delta in stream:
    print(delta, end="", flush=True)
print("\nSources:", stream.sources)   # hits are available before generation starts
```

The default prompt numbers each context block and asks for `[1]`-style
citations; override it per call with `prompt=` or globally with
`config.prompt`.

### Management

```python
for info in rag.sources():
    print(info.source, info.chunks, info.characters, info.added_at)

rag.delete(source="handbook.pdf")         # remove one document
rag.delete(where={"year": {"$lt": 2020}}) # remove by metadata filter
rag.reset()                               # empty the index, keep the file
rag.optimize()                            # compact FTS + VACUUM

stats = rag.stats()
print(stats.documents, stats.sources, stats.dimensions, f"{stats.size_mb:.1f} MB")

len(rag)                                  # chunk count
rag.close()                               # or use Rag as a context manager
```

## Metadata filtering

Filters are plain dicts, compiled to parameterised SQL over the JSON metadata
column — values are never interpolated into the query string.

```python
{"team": "legal"}                          # equality (bare value)
{"year": {"$eq": 2025}}                    # explicit equality
{"status": {"$ne": "draft"}}               # not equal
{"pages": {"$gt": 10}}                     # >
{"year": {"$gte": 2024}}                   # >=
{"score": {"$lt": 0.5}}                    # <
{"year": {"$lte": 2025}}                   # <=
{"ext": {"$in": [".md", ".rst"]}}          # membership
{"ext": {"$nin": [".log"]}}                # negated membership
{"title": {"$like": "%invoice%"}}          # SQL LIKE
{"tags": {"$contains": "urgent"}}          # element of a JSON array, or a substring
{"author": {"$exists": True}}              # key present / absent

{"$and": [{"team": "legal"}, {"year": 2025}]}
{"$or":  [{"year": 2024}, {"pinned": True}]}
{"$not": {"status": "archived"}}

{"year": {"$gte": 2024, "$lt": 2026}}      # two operators on one field = AND
{"doc.author.name": "Ada"}                 # dotted paths reach into nested metadata
```

Filters work on `search()`, `query()` and `delete()`. File ingestion attaches
`kind`, `filename`, `extension`, `bytes` and `path` automatically, so
`where={"extension": ".md"}` works with no extra bookkeeping.

## Model backends

With no models passed, `connect()` walks this order:

**Embeddings** — `OPENAI_API_KEY` → a reachable Ollama daemon → an installed
`sentence-transformers` → `HashEmbedder` (with a warning).
**Chat** — `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` → a reachable Ollama daemon →
`EchoChatModel` (with a warning).

`HashEmbedder` and `EchoChatModel` are deliberately terrible at their jobs but
need no network, no key and no download — they exist so examples and tests run
anywhere. `EchoChatModel` returns the prompt verbatim, which makes a bad answer
unambiguously a retrieval problem.

```python
# OpenAI (pip install 'softrag[openai]')
from softrag.providers.openai import OpenAIEmbedder, OpenAIChat
rag = softrag.connect("kb.db",
    embed_model=OpenAIEmbedder("text-embedding-3-small", dimensions=512),
    chat_model=OpenAIChat("gpt-4.1-mini", temperature=0))

# Anthropic (pip install 'softrag[anthropic]') — chat only; pair with any embedder
from softrag.providers.anthropic import AnthropicChat
rag = softrag.connect("kb.db", chat_model=AnthropicChat("claude-sonnet-5"))

# Ollama — fully local, no SDK, plain HTTP
from softrag.providers.ollama import OllamaEmbedder, OllamaChat
rag = softrag.connect("kb.db",
    embed_model=OllamaEmbedder("nomic-embed-text"),
    chat_model=OllamaChat("llama3.2"))

# sentence-transformers (pip install 'softrag[local]')
from softrag.providers.local import SentenceTransformerEmbedder, CrossEncoderReranker
rag = softrag.connect("kb.db",
    embed_model=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
    reranker=CrossEncoderReranker())   # pip install 'softrag[rerank]'
```

**Bring your own.** Anything with the right shape is accepted and adapted:
an object with `embed_query`/`embed_documents` (LangChain, softrag), an object
with `encode` (sentence-transformers), a Chroma-style function taking a list, or
a bare callable.

```python
rag = softrag.connect(
    "kb.db",
    embed_model=lambda text: my_model.vectorise(text),   # str -> list[float]
    chat_model=lambda prompt: my_llm.generate(prompt),   # str -> str
)

# LangChain objects work as-is (embed_query / invoke are both recognised)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
rag = softrag.connect("kb.db",
    embed_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    chat_model=ChatOpenAI(model="gpt-4.1-mini"))
```

See [docs/providers.md](docs/providers.md) to write your own in about ten lines.

## Installation extras

| Extra                       | Pulls in                    | You need it for                                        |
| --------------------------- | --------------------------- | ------------------------------------------------------ |
| *(none)*                    | `sqlite-vec`                | The engine, storage, hybrid search, text/HTML/Office/CSV/JSON ingest |
| `softrag[openai]`           | `openai`                    | OpenAI embeddings and chat, or any OpenAI-compatible server (vLLM, LM Studio) |
| `softrag[anthropic]`        | `anthropic`                 | Claude chat and vision                                 |
| `softrag[local]`            | `sentence-transformers`     | Local embeddings, no network after the model downloads |
| `softrag[rerank]`           | `sentence-transformers`     | `CrossEncoderReranker` as a second-stage reranker      |
| `softrag[files]`            | `pypdf`                     | Reading PDFs                                           |
| `softrag[web]`              | `httpx`, `trafilatura`      | Better URL fetching and boilerplate-free web extraction |
| `softrag[cli]`              | `rich`                      | The `softrag` command-line interface                   |
| `softrag[all]`              | everything above            | Trying things out                                      |

Ollama needs no extra at all — softrag talks to it over plain HTTP with the
standard library.

## Supported formats

| Kind      | Extensions                                                                   |
| --------- | ---------------------------------------------------------------------------- |
| Text      | `.txt` `.text` `.md` `.markdown` `.rst` `.org` `.log`                        |
| Markup    | `.html` `.htm` `.xhtml` `.xml`                                               |
| Documents | `.pdf` (needs `softrag[files]`), `.docx`, `.pptx`, `.xlsx`, `.xlsm`, `.epub` |
| Data      | `.csv` `.tsv` `.json` `.jsonl` `.ndjson`                                     |
| Code      | `.py` `.js` `.ts` `.go` `.rs` `.java` `.rb` `.c` `.cpp` `.sql` `.sh` `.toml` `.yaml` and ~25 more |
| Images    | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.bmp` via `add_image()` + a vision model |
| Web       | any URL via `add_web()`                                                      |

DOCX, PPTX, XLSX and EPUB are read with the standard library alone — they are
ZIP archives of XML, and softrag unzips them itself. Only PDF needs a
third-party parser. Unknown extensions that look like text are read as text, and
`softrag.ingest.EXTRACTORS[".rtf"] = my_extractor` teaches it a new format.

## What's included

Beyond the engine, the package ships a `softrag` command-line interface
(`softrag[cli]`) plus `softrag.rerank`, `softrag.eval` and `softrag.transforms`
modules for second-stage reranking, retrieval evaluation and query
transformation. Those are documented separately.

## Documentation

| Guide                                      | What it covers                                                     |
| ------------------------------------------ | ------------------------------------------------------------------ |
| [docs/quickstart.md](docs/quickstart.md)   | Install to first answer, including a fully offline path             |
| [docs/retrieval.md](docs/retrieval.md)     | Hybrid search, RRF, tuning, MMR, filters, debugging bad retrieval   |
| [docs/ingestion.md](docs/ingestion.md)     | Formats, chunking strategies, metadata, idempotent re-ingest        |
| [docs/providers.md](docs/providers.md)     | Every backend and how to write your own                             |
| [docs/architecture.md](docs/architecture.md) | The SQLite schema and why it looks like that                      |
| [docs/faq.md](docs/faq.md)                 | Scale, concurrency, backups, changing embedding models              |
| [examples/](examples/)                     | Runnable scripts, two of which need no API key                      |

## Contributing

Bug reports, format extractors and provider adapters are all welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, tests and PR conventions.

Release notes live in [CHANGELOG.md](CHANGELOG.md). Licensed under the MIT
License — see [LICENSE](LICENSE).

Created by [Julio Peixoto](https://github.com/JulioPeixoto).
