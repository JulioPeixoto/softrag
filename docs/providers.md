# Providers

softrag never requires a particular AI framework. Whatever you already have —
a LangChain object, a sentence-transformers model, a bare function, one of the
small clients that ship here — is normalised into two tiny protocols and used
from there.

- [Auto-detection](#auto-detection)
- [The protocols](#the-protocols)
- [Built-in backends](#built-in-backends)
- [Bring your own](#bring-your-own)
- [Writing a reranker](#writing-a-reranker)
- [Error handling](#error-handling)

## Auto-detection

`softrag.connect(path)` with no models resolves them from the environment. API
keys come first because they need no download; local backends next; the
dependency-free fallbacks last, with a warning.

**Embeddings**

| Order | Condition                            | Chosen backend                                   |
| ----- | ------------------------------------ | ------------------------------------------------ |
| 1     | `OPENAI_API_KEY` is set              | `OpenAIEmbedder("text-embedding-3-small")`       |
| 2     | An Ollama daemon answers             | `OllamaEmbedder("nomic-embed-text")`             |
| 3     | `sentence-transformers` is installed | `SentenceTransformerEmbedder("all-MiniLM-L6-v2")`|
| 4     | otherwise                            | `HashEmbedder()` — warns                         |

**Chat**

| Order | Condition                   | Chosen backend                     |
| ----- | --------------------------- | ---------------------------------- |
| 1     | `ANTHROPIC_API_KEY` is set  | `AnthropicChat("claude-sonnet-5")` |
| 2     | `OPENAI_API_KEY` is set     | `OpenAIChat("gpt-4.1-mini")`       |
| 3     | An Ollama daemon answers    | `OllamaChat("llama3.2")`           |
| 4     | otherwise                   | `EchoChatModel()` — warns          |

Two details matter in practice. The Ollama probe is a 0.5-second HTTP request to
`$OLLAMA_HOST/api/tags` (default `http://localhost:11434`) — fast and silent, so
auto-detection never hangs a startup. And the chat model is resolved *lazily*,
on first use: a retrieval-only engine never needs one, so building one costs
nothing and configuring none is not an error until you call `query()`.

Turn detection off when a missing model should be a loud failure rather than a
silent downgrade:

```python
rag = softrag.Rag(db_path="kb.db", embed_model=my_embedder, auto=False)
rag.query("...")   # ConfigurationError: this engine has no chat model
```

## The protocols

Two duck-typed protocols, both plain Python:

```python
class Embedder(Protocol):
    def embed_query(self, text: str) -> Sequence[float]: ...
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

class ChatModel(Protocol):
    def complete(self, prompt: str) -> str: ...
```

Streaming is optional: a `stream(prompt) -> Iterator[str]` method is picked up
automatically when present, and backends without one still work — `query(stream=True)`
yields a single delta containing the whole completion.

`adapt_embedder()` recognises, in order of preference:

1. an object with `embed_query` / `embed_documents` (LangChain, softrag's own)
2. an object with `encode` (sentence-transformers)
3. a callable taking a **list** and returning a list of vectors (Chroma-style
   embedding functions — detected by probing once)
4. a callable taking a **string** and returning one vector

`adapt_chat_model()` recognises an object with `complete`, an object with
`invoke` (LangChain), or a plain callable. Return values are unwrapped for you:
a string, a `.content` string, Anthropic-style content blocks, or an object with
`.text` all come out as text.

Anything else raises `ConfigurationError` naming the shapes that would have
worked.

## Built-in backends

Concrete classes live one level down from `softrag.providers`, in a module per
vendor, so importing one never imports the others' SDKs.

### OpenAI — `pip install 'softrag[openai]'`

```python
from softrag.providers.openai import OpenAIEmbedder, OpenAIChat

embed = OpenAIEmbedder(
    "text-embedding-3-small",
    dimensions=512,      # shortened embeddings: smaller index, slight accuracy cost
    batch_size=128,
)
chat = OpenAIChat("gpt-4.1-mini", temperature=0.0, max_tokens=1024, system=None)
```

Both accept `api_key=` (overriding `OPENAI_API_KEY`) and `base_url=` (also read
from `OPENAI_BASE_URL`), which is how you point them at any OpenAI-compatible
server — vLLM, LM Studio, llama.cpp's server, an Azure gateway, a proxy:

```python
chat = OpenAIChat("local-model", base_url="http://localhost:8000/v1")
```

With a `base_url` set, an API key is optional. `OpenAIChat` supports streaming
and vision (`add_image()`).

### Anthropic — `pip install 'softrag[anthropic]'`

Chat and vision only; Anthropic serves no embedding model, so pair it with any
embedder.

```python
from softrag.providers.anthropic import AnthropicChat
from softrag.providers.openai import OpenAIEmbedder

rag = softrag.connect("kb.db",
    embed_model=OpenAIEmbedder(),
    chat_model=AnthropicChat("claude-sonnet-5", max_tokens=2048, temperature=0.0))
```

`max_tokens` is required by the Messages API, so it always has a value (2048 by
default). Streaming and vision are both supported.

### Ollama — no extra needed

Deliberately implemented over `urllib` rather than the `ollama` SDK: it adds no
dependency, and a fully offline softrag is the point of the project.

```python
from softrag.providers.ollama import OllamaEmbedder, OllamaChat, is_available, base_url

print(base_url(), is_available())     # honours OLLAMA_HOST

embed = OllamaEmbedder("nomic-embed-text", timeout=120.0, batch_size=32)
chat  = OllamaChat("llama3.2", timeout=300.0, temperature=0.0,
                   options={"num_ctx": 8192})   # extra Ollama options merged in
```

Error messages are translated into actions: a 404 becomes
`ollama pull <model>`, a connection failure becomes `ollama serve`. Vision works
with a vision model (`llava`, `llama3.2-vision`).

### sentence-transformers — `pip install 'softrag[local]'`

```python
from softrag.providers.local import SentenceTransformerEmbedder

embed = SentenceTransformerEmbedder(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",              # or "cuda" / "mps" / None to let the library choose
    batch_size=32,
    normalize=True,            # leave on: the index expects normalised vectors
    prompt_name=None,          # for E5/BGE/GTE-style asymmetric query prefixes
)
print(embed.dimensions)        # 384 for MiniLM-L6
```

Once the model is cached, this needs no network at all. Pair it with `OllamaChat`
for a stack that works on a plane.

### The dependency-free fallbacks

```python
from softrag import HashEmbedder, EchoChatModel

HashEmbedder(dimensions=256)   # hashes word unigrams + bigrams into a vector
EchoChatModel()                # returns the prompt verbatim
```

Neither is good at its job, and that is deliberate. They need no network, no key
and no download, which makes them right for tests, examples and offline smoke
checks. `EchoChatModel` in particular is a diagnostic: whatever it returns came
straight out of your index, so a bad "answer" is unambiguously a retrieval
problem.

`HashEmbedder` is *not* suitable for real retrieval. It has no notion of
synonymy — only literal token overlap survives the hash — so it mostly
duplicates what BM25 already does. If `rag.stats().dimensions` is 256 and you
never configured an embedder, you are accidentally using it.

## Bring your own

### A pair of functions

```python
import softrag

rag = softrag.connect(
    "kb.db",
    embed_model=lambda text: my_model.vectorise(text),    # str -> list[float]
    chat_model=lambda prompt: my_llm.generate(prompt),    # str -> str
)
```

That is the whole requirement. softrag adapts the callables, calls the embedder
once per text, and validates that what comes back is a non-empty sequence of
floats of consistent width.

To tell a Chroma-style list-taking function from a string-taking one, the
adapter probes your callable once with a one-element list at construction time.
If your function happens to accept a list *and* return something list-shaped, it
will be treated as a batch function — pass an object with explicit
`embed_query`/`embed_documents` methods when you want no ambiguity.

### A real embedder in ten lines

Implement the batch method when your backend has one — softrag will use it, and
one request per 64 chunks beats 64 requests.

```python
from typing import Sequence
import httpx

class MyEmbedder:
    def __init__(self, url: str, model: str) -> None:
        self.url, self.model = url, model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        r = httpx.post(self.url, json={"model": self.model, "input": list(texts)})
        r.raise_for_status()
        return [item["embedding"] for item in r.json()["data"]]   # same order as input
```

Two rules the adapter enforces, so get them right: one vector per input, **in
input order**, and every vector the same width. Both are checked, and a
violation raises `EmbeddingError` with a message saying which one broke.

### A chat model in ten lines

```python
from typing import Iterator

class MyChat:
    def complete(self, prompt: str) -> str:
        return my_client.generate(prompt).text

    def stream(self, prompt: str) -> Iterator[str]:      # optional
        for event in my_client.generate_stream(prompt):
            yield event.delta

    def describe_image(self, image_base64: str, *, mime_type: str,
                       prompt: str) -> str:              # optional, for add_image()
        return my_client.vision(image_base64, mime_type, prompt).text
```

### LangChain

LangChain objects satisfy the protocols as they are — `Embeddings` exposes
`embed_query`/`embed_documents` and chat models expose `invoke`:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

rag = softrag.connect("kb.db",
    embed_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    chat_model=ChatOpenAI(model="gpt-4.1-mini"))
```

The `AIMessage` a LangChain model returns is unwrapped to its text
automatically, which is why `query()` gives you a string and not a message
object.

### Testing an adapter

```python
from softrag import adapt_embedder, adapt_chat_model

e = adapt_embedder(my_thing)
vectors = e.embed_documents(["a", "b"])
assert len(vectors) == 2 and len(vectors[0]) == len(vectors[1])

c = adapt_chat_model(my_other_thing)
assert isinstance(c.complete("hello"), str)
```

If it survives that, it will work in the engine.

## Writing a reranker

A reranker is anything with:

```python
def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> list[Hit]: ...
```

Return at most `top_k` hits, best first, and set `hit.score` to your own score
if you want it visible downstream.

```python
from softrag import Hit

class KeywordBoostReranker:
    """Toy example: push hits containing the literal query to the top."""

    def rerank(self, query: str, hits, *, top_k: int) -> list[Hit]:
        needle = query.lower()
        ranked = sorted(hits, key=lambda h: needle in h.text.lower(), reverse=True)
        return list(ranked[:top_k])

rag = softrag.connect("kb.db", reranker=KeywordBoostReranker())
```

The built-in cross-encoder is `softrag.providers.local.CrossEncoderReranker`
(`pip install 'softrag[rerank]'`), which loads a local cross-encoder and scores
`(query, chunk)` pairs jointly. It is slower than the first-stage retrieval by
orders of magnitude, which is exactly right for a second stage over a few dozen
candidates — see [retrieval.md](retrieval.md#reranking) for how softrag widens
the candidate set automatically when a reranker is present.

A reranker that raises is not allowed to take down a query: `CrossEncoderReranker`
logs a warning and returns the original order.

## Error handling

Backend failures are wrapped so you can catch them by cause, not by vendor:

| Exception                | Raised when                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `EmbeddingError`         | The embedder failed, returned the wrong count, or returned non-floats. |
| `ChatError`              | The chat backend failed, including mid-stream.                     |
| `ConfigurationError`     | An object cannot be adapted, or a required model is missing.        |
| `MissingDependencyError` | An optional package is needed; the message names the extra to install. |
| `ProviderError`          | Base class of `EmbeddingError` and `ChatError`.                     |

```python
from softrag import ProviderError, MissingDependencyError

try:
    answer = rag.query("...")
except MissingDependencyError as exc:
    print(exc)          # "... requires the 'openai' package ... pip install 'softrag[openai]'"
except ProviderError as exc:
    print("backend failed:", exc)
```
