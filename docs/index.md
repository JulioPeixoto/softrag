# softrag documentation

softrag is an embedded RAG engine. Chunk text, metadata, the vector index and
the full-text index all live in one SQLite file; the core install is one
dependency; every model backend is pluggable, including fully local ones.

If you have five minutes, read the [quickstart](quickstart.md). If retrieval is
already returning the wrong things, skip to
[debugging retrieval](retrieval.md#is-it-retrieval-or-generation).

## Guides

| Guide                            | Read it when                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------- |
| [Quickstart](quickstart.md)      | You want a working index and a first answer, with or without an API key.         |
| [Retrieval](retrieval.md)        | You want to understand or tune what comes back: modes, RRF, MMR, filters.        |
| [Ingestion](ingestion.md)        | You are feeding documents in: formats, chunking, metadata, re-indexing.          |
| [Providers](providers.md)        | You are choosing a model backend, or writing your own adapter.                   |
| [Architecture](architecture.md)  | You want to know what is inside the `.db` file and why.                          |
| [FAQ](faq.md)                    | Practical operations: scale, threads, backups, changing embedding models.        |

## Runnable examples

The [`examples/`](../examples/) directory holds scripts you can run directly.
`quickstart.py` and `filtering.py` need no API key and no network.

## The 20-line tour

```python
import softrag

rag = softrag.connect("kb.db")

rag.add("handbook.pdf")                             # file
rag.add("https://example.com/changelog")            # URL
rag.add("Refunds take 5 business days.", metadata={"team": "support"})
rag.add_directory("./docs", pattern="**/*.md")      # a whole tree

hits = rag.search("refund window", top_k=5, where={"team": "support"})
for hit in hits:
    print(hit.score, hit.source, hit.text[:80])

answer = rag.query("How long do refunds take?")
print(answer)             # Answer is a str
print(answer.sources)     # provenance is one attribute away

print(rag.stats())
rag.close()
```

## API surface at a glance

| Area        | Names                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------- |
| Entry point | `connect`, `Rag`, `RagConfig`, `DEFAULT_PROMPT`                                            |
| Ingestion   | `add`, `add_text`, `add_file`, `add_web`, `add_image`, `add_directory`, `add_many`          |
| Retrieval   | `search`, `query`, `RetrievalConfig`, `reciprocal_rank_fusion`, `maximal_marginal_relevance`|
| Results     | `Answer`, `StreamingAnswer`, `Hit`, `Chunk`, `IngestResult`, `SourceInfo`, `Stats`          |
| Management  | `sources`, `delete`, `reset`, `optimize`, `stats`, `close`                                  |
| Chunking    | `RecursiveChunker`, `MarkdownChunker`, `SentenceChunker`, `by_separator`                    |
| Backends    | `HashEmbedder`, `EchoChatModel`, `adapt_embedder`, `adapt_chat_model`                       |
| Protocols   | `Embedder`, `ChatModel`, `Reranker`                                                         |
| Errors      | `SoftragError` and its subclasses (see [FAQ](faq.md#which-errors-should-i-catch))           |
| Storage     | `Store`                                                                                     |

Everything in that table is importable straight from `softrag`. Concrete
provider classes live one level down, in `softrag.providers.openai`,
`softrag.providers.anthropic`, `softrag.providers.ollama` and
`softrag.providers.local`.
