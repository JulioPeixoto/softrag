# Quickstart

From nothing to a first answer. Three paths, in increasing order of setup cost:
no API key at all, fully local with Ollama, and hosted with OpenAI.

## Install

```bash
pip install softrag
```

That is one dependency (`sqlite-vec`). Model backends and PDF support are
[extras](../README.md#installation-extras) you add when you need them.

Python 3.10 or newer is required, and your Python must have been built with
SQLite extension loading enabled — the stock python.org, Homebrew, `uv` and
Debian builds all are. If yours is not, softrag says so on the first connect
rather than failing mysteriously later.

## Path A — no API key, no network

Useful for a smoke test, for CI, and for understanding the shape of the library
before deciding on a model. `HashEmbedder` hashes character n-grams into a fixed
vector — retrieval quality is far below a real model, but it needs nothing.
`EchoChatModel` returns the prompt it was given instead of generating, which
means anything you see came out of the index.

```python
import softrag

rag = softrag.Rag(
    db_path=":memory:",
    embed_model=softrag.HashEmbedder(dimensions=256),
    chat_model=softrag.EchoChatModel(),
)

rag.add_text("Refunds are processed within 5 business days of approval.",
             name="refunds", metadata={"team": "support"})
rag.add_text("Shipping is free on orders above 50 EUR.",
             name="shipping", metadata={"team": "logistics"})

for hit in rag.search("how long do refunds take", top_k=2):
    print(f"{hit.score:.4f}  {hit.source}  {hit.text}")
```

```
0.0328  refunds  Refunds are processed within 5 business days of approval.
0.0161  shipping  Shipping is free on orders above 50 EUR.
```

Those scores are RRF fusion scores, not similarities — see
[retrieval.md](retrieval.md#reading-the-scores).

A complete version of this is [`examples/quickstart.py`](../examples/quickstart.py),
which runs as-is.

## Path B — fully local with Ollama

No API key, no data leaving the machine, no SDK: softrag talks to Ollama over
plain HTTP using the standard library.

```bash
# once
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

```python
import softrag
from softrag.providers.ollama import OllamaEmbedder, OllamaChat

rag = softrag.connect(
    "kb.db",
    embed_model=OllamaEmbedder("nomic-embed-text"),
    chat_model=OllamaChat("llama3.2", temperature=0),
)

rag.add_directory("./docs", pattern="**/*.md")

answer = rag.query("How does hybrid search work?")
print(answer)
print("Sources:", answer.sources)
```

If a daemon is already running when you call `softrag.connect("kb.db")` with no
models, softrag finds it by itself. See
[`examples/local_ollama.py`](../examples/local_ollama.py).

Prefer sentence-transformers for embeddings and Ollama only for generation? That
combination is [`examples/local_embeddings.py`](../examples/local_embeddings.py).

## Path C — OpenAI

```bash
pip install 'softrag[openai,files]'   # files = PDF support
export OPENAI_API_KEY=sk-...
```

```python
import softrag

rag = softrag.connect("kb.db")        # OPENAI_API_KEY is picked up automatically

rag.add("handbook.pdf")
rag.add("https://example.com/changelog")

answer = rag.query("What changed in the refund policy?")
print(answer)
for source in answer.sources:
    print(" -", source)
```

Being explicit costs one import and buys you control over model, dimensions and
temperature:

```python
from softrag.providers.openai import OpenAIEmbedder, OpenAIChat

rag = softrag.connect(
    "kb.db",
    embed_model=OpenAIEmbedder("text-embedding-3-small", dimensions=512),
    chat_model=OpenAIChat("gpt-4.1-mini", temperature=0),
)
```

Shortened embeddings (`dimensions=512` instead of the default 1536) make the
index roughly three times smaller and searches proportionally faster, at a small
accuracy cost. Decide before your first ingest: the width is baked into the
index. See [`examples/openai_rag.py`](../examples/openai_rag.py).

## Streaming

```python
stream = rag.query("Summarise the changelog", stream=True)
for delta in stream:
    print(delta, end="", flush=True)

print()
print("Sources:", stream.sources)     # retrieval finished before the first delta
```

`stream.hits` and `stream.sources` are populated immediately, because retrieval
happens before generation starts. `stream.text` holds whatever has been produced
so far, and `stream.collect()` drains the rest and hands you a plain `Answer`.

Backends without native streaming still work — softrag yields one delta
containing the whole completion.

## What you get back

```python
answer = rag.query("What is the refund window?")

str(answer)         # the generated text; Answer subclasses str
answer.sources      # ['handbook.pdf', 'policy.md'] — unique, best-scoring first
answer.hits         # the Hit objects, with scores and metadata
answer.context      # the retrieved text exactly as the model saw it
answer.question     # 'What is the refund window?'
answer.prompt       # the fully rendered prompt, useful when debugging
```

## Persisting and moving an index

```python
rag = softrag.connect("kb.db")       # created if absent
...
rag.close()
```

`kb.db` is the whole index. Copy it, commit it (if it is small), ship it inside
a container image, hand it to a colleague. WAL mode means you may also see
`kb.db-wal` and `kb.db-shm` beside it while the database is open; `close()`
cleans them up, and copying a closed database is a copy of everything.

## Next

- Tuning what comes back: [retrieval.md](retrieval.md)
- Feeding documents in: [ingestion.md](ingestion.md)
- Choosing or writing a backend: [providers.md](providers.md)
