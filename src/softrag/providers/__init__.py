"""Model backends and the adapters that let almost anything be one.

softrag never requires a particular AI framework. Whatever you already have --
a LangChain ``Embeddings`` object, a sentence-transformers model, a bare
function, one of the small clients in this package -- is normalised here into
the :class:`~softrag.types.Embedder` and :class:`~softrag.types.ChatModel`
protocols the rest of the library speaks.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterator, List, Optional, Sequence

from ..errors import ChatError, ConfigurationError, EmbeddingError
from ..types import ChatModel, Embedder

log = logging.getLogger("softrag.providers")

__all__ = [
    "adapt_embedder",
    "adapt_chat_model",
    "auto_embedder",
    "auto_chat_model",
    "HashEmbedder",
    "EchoChatModel",
]


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


class _EmbedderAdapter:
    """Wraps a foreign embedding object in the :class:`Embedder` protocol."""

    __slots__ = ("_target", "_query_fn", "_batch_fn", "_name")

    def __init__(
        self,
        target: Any,
        query_fn: Callable[[str], Sequence[float]],
        batch_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]],
        name: str,
    ) -> None:
        self._target = target
        self._query_fn = query_fn
        self._batch_fn = batch_fn
        self._name = name

    def embed_query(self, text: str) -> List[float]:
        try:
            vector = self._query_fn(text)
        except Exception as exc:
            raise EmbeddingError(f"{self._name} failed to embed a query: {exc}") from exc
        return _as_floats(vector, self._name)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            if self._batch_fn is not None:
                vectors = self._batch_fn(list(texts))
            else:
                vectors = [self._query_fn(text) for text in texts]
        except Exception as exc:
            raise EmbeddingError(
                f"{self._name} failed to embed {len(texts)} documents: {exc}"
            ) from exc
        out = [_as_floats(v, self._name) for v in vectors]
        if len(out) != len(texts):
            raise EmbeddingError(
                f"{self._name} returned {len(out)} vectors for {len(texts)} inputs. "
                "An embedder must return exactly one vector per input, in order."
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Embedder {self._name}>"


def adapt_embedder(embedder: Any) -> Embedder:
    """Normalise ``embedder`` into something implementing :class:`Embedder`.

    Recognised shapes, in order of preference:

    * an object with ``embed_query`` / ``embed_documents`` (LangChain, softrag)
    * an object with ``encode`` (sentence-transformers)
    * an object with ``__call__`` taking a list (Chroma embedding functions)
    * a plain callable taking one string

    Args:
        embedder: The object to adapt.

    Returns:
        An adapter exposing the protocol.

    Raises:
        ConfigurationError: If no known shape matches.
    """
    if embedder is None:
        raise ConfigurationError(
            "No embedding model was provided. Pass embed_model=..., or use "
            "softrag.connect(...) to pick one up from the environment."
        )

    name = type(embedder).__name__

    query = getattr(embedder, "embed_query", None)
    batch = getattr(embedder, "embed_documents", None)
    if callable(query) or callable(batch):
        if not callable(query):
            def query(text: str, _batch=batch) -> Sequence[float]:  # type: ignore[misc]
                return _batch([text])[0]
        return _EmbedderAdapter(embedder, query, batch if callable(batch) else None, name)

    encode = getattr(embedder, "encode", None)
    if callable(encode):
        def _encode_one(text: str) -> Sequence[float]:
            return _first_row(encode(text))

        def _encode_many(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return _as_rows(encode(list(texts)))

        return _EmbedderAdapter(embedder, _encode_one, _encode_many, name)

    if callable(embedder):
        def _call_one(text: str) -> Sequence[float]:
            result = embedder(text)
            return _first_row(result) if _looks_nested(result) else result

        def _call_many(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            result = embedder(list(texts))
            return _as_rows(result)

        # Chroma-style functions take a list; plain ones take a string. Probe once.
        try:
            probe = embedder(["softrag probe"])
            if _looks_nested(probe):
                return _EmbedderAdapter(embedder, _call_one, _call_many, name)
        except Exception:
            pass
        return _EmbedderAdapter(embedder, _call_one, None, name)

    raise ConfigurationError(
        f"Cannot use {name} as an embedding model. Provide an object with "
        "embed_query()/embed_documents(), an object with encode(), or a callable "
        "that maps a string to a list of floats."
    )


class _ChatAdapter:
    """Wraps a foreign chat object in the :class:`ChatModel` protocol."""

    __slots__ = ("_complete", "_stream", "_name")

    def __init__(
        self,
        complete: Callable[[str], str],
        stream: Optional[Callable[[str], Iterator[str]]],
        name: str,
    ) -> None:
        self._complete = complete
        self._stream = stream
        self._name = name

    def complete(self, prompt: str) -> str:
        try:
            return _as_text(self._complete(prompt))
        except Exception as exc:
            raise ChatError(f"{self._name} failed to generate a response: {exc}") from exc

    def stream(self, prompt: str) -> Iterator[str]:
        if self._stream is None:
            yield self.complete(prompt)
            return
        try:
            for delta in self._stream(prompt):
                text = _as_text(delta)
                if text:
                    yield text
        except Exception as exc:
            raise ChatError(f"{self._name} failed mid-stream: {exc}") from exc

    @property
    def supports_streaming(self) -> bool:
        return self._stream is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ChatModel {self._name}>"


def adapt_chat_model(model: Any) -> ChatModel:
    """Normalise ``model`` into something implementing :class:`ChatModel`.

    Recognised shapes: an object with ``complete``, an object with ``invoke``
    (LangChain), or a plain callable. Streaming is picked up automatically from
    a ``stream`` method when present.

    Raises:
        ConfigurationError: If no known shape matches.
    """
    if model is None:
        raise ConfigurationError(
            "No chat model was provided. Pass chat_model=... to generate answers, "
            "or use rag.search(...) which needs no chat model at all."
        )

    name = type(model).__name__
    stream_fn = getattr(model, "stream", None)
    stream = stream_fn if callable(stream_fn) else None

    complete = getattr(model, "complete", None)
    if callable(complete):
        return _ChatAdapter(complete, stream, name)

    invoke = getattr(model, "invoke", None)
    if callable(invoke):
        return _ChatAdapter(invoke, stream, name)

    if callable(model):
        return _ChatAdapter(model, stream, name)

    raise ConfigurationError(
        f"Cannot use {name} as a chat model. Provide an object with complete() "
        "or invoke(), or a callable that maps a prompt string to text."
    )


# --------------------------------------------------------------------------- #
# Built-in fallbacks
# --------------------------------------------------------------------------- #


class HashEmbedder:
    """A deterministic, dependency-free embedder.

    It hashes character n-grams into a fixed-width vector -- essentially the
    hashing trick. Retrieval quality is far below a real embedding model, but it
    needs no network, no API key and no model download, which makes it ideal for
    tests, examples and offline smoke checks.

    Args:
        dimensions: Width of the produced vectors.

    Example:
        >>> embedder = HashEmbedder(dimensions=64)
        >>> len(embedder.embed_query("hello"))
        64
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ConfigurationError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> List[float]:
        import hashlib
        import math

        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        # Unigrams plus bigrams: enough signal to make ordering meaningful.
        grams = tokens + [
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:])
        ]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"HashEmbedder(dimensions={self.dimensions})"


class EchoChatModel:
    """A chat model that quotes its context instead of generating.

    Useful for testing retrieval in isolation: whatever it returns came straight
    from the index, so a bad answer is unambiguously a retrieval problem.
    """

    def complete(self, prompt: str) -> str:
        return prompt

    def stream(self, prompt: str) -> Iterator[str]:
        for line in prompt.splitlines(keepends=True):
            yield line

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "EchoChatModel()"


# --------------------------------------------------------------------------- #
# Environment auto-detection
# --------------------------------------------------------------------------- #


def auto_embedder(*, model: Optional[str] = None) -> Embedder:
    """Pick an embedding model from what the environment makes available.

    The order is deliberate -- API keys first because they need no download,
    then local backends, then the dependency-free fallback:

    1. ``OPENAI_API_KEY`` -> OpenAI embeddings
    2. ``VOYAGE_API_KEY`` -> Voyage embeddings
    3. ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` -> Gemini embeddings
    4. a reachable Ollama daemon -> Ollama embeddings
    5. ``sentence-transformers`` installed -> a local MiniLM
    6. :class:`HashEmbedder`, with a warning

    Args:
        model: Force a specific model name for whichever backend is chosen.

    Returns:
        A ready-to-use embedder.
    """
    from . import openai as openai_provider
    from . import ollama as ollama_provider
    from . import local as local_provider

    if os.getenv("OPENAI_API_KEY"):
        log.debug("auto-detected OpenAI embeddings")
        return openai_provider.OpenAIEmbedder(model=model or "text-embedding-3-small")

    if ollama_provider.is_available():
        log.debug("auto-detected a running Ollama daemon")
        return ollama_provider.OllamaEmbedder(model=model or "nomic-embed-text")

    if local_provider.is_available():
        log.debug("auto-detected sentence-transformers")
        return local_provider.SentenceTransformerEmbedder(
            model=model or "sentence-transformers/all-MiniLM-L6-v2"
        )

    log.warning(
        "No embedding backend detected, falling back to HashEmbedder. Retrieval "
        "quality will be poor. Set OPENAI_API_KEY, run Ollama, or install "
        "'softrag[local]' for real embeddings."
    )
    return HashEmbedder()


def auto_chat_model(*, model: Optional[str] = None) -> ChatModel:
    """Pick a chat model from what the environment makes available.

    1. ``ANTHROPIC_API_KEY`` -> Claude
    2. ``OPENAI_API_KEY`` -> GPT
    3. a reachable Ollama daemon -> a local model
    4. :class:`EchoChatModel`, with a warning

    Args:
        model: Force a specific model name for whichever backend is chosen.
    """
    from . import anthropic as anthropic_provider
    from . import openai as openai_provider
    from . import ollama as ollama_provider

    if os.getenv("ANTHROPIC_API_KEY"):
        log.debug("auto-detected Anthropic")
        return anthropic_provider.AnthropicChat(model=model or "claude-sonnet-5")

    if os.getenv("OPENAI_API_KEY"):
        log.debug("auto-detected OpenAI")
        return openai_provider.OpenAIChat(model=model or "gpt-4.1-mini")

    if ollama_provider.is_available():
        log.debug("auto-detected a running Ollama daemon")
        return ollama_provider.OllamaChat(model=model or "llama3.2")

    log.warning(
        "No chat backend detected, falling back to EchoChatModel, which returns "
        "the retrieved context instead of an answer. Set ANTHROPIC_API_KEY or "
        "OPENAI_API_KEY, or run Ollama, to generate real answers."
    )
    return EchoChatModel()


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #


def _as_floats(vector: Any, name: str) -> List[float]:
    """Coerce a backend's return value into a flat list of Python floats."""
    tolist = getattr(vector, "tolist", None)
    if callable(tolist):
        vector = tolist()
    try:
        out = [float(v) for v in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"{name} returned {type(vector).__name__} where a sequence of floats "
            "was expected."
        ) from exc
    if not out:
        raise EmbeddingError(f"{name} returned an empty embedding.")
    return out


def _as_rows(result: Any) -> List[Any]:
    tolist = getattr(result, "tolist", None)
    if callable(tolist):
        result = tolist()
    return list(result)


def _first_row(result: Any) -> Any:
    rows = _as_rows(result)
    if rows and _looks_nested(rows):
        return rows[0]
    return rows


def _looks_nested(result: Any) -> bool:
    try:
        rows = _as_rows(result)
    except TypeError:
        return False
    if not rows:
        return False
    first = rows[0]
    return isinstance(first, (list, tuple)) or hasattr(first, "__len__") and not isinstance(
        first, (str, bytes, float, int)
    )


def _as_text(value: Any) -> str:
    """Pull plain text out of whatever a chat backend returned."""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        # Anthropic-style content blocks.
        if isinstance(content, (list, tuple)):
            parts = []
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(str(text))
            return "".join(parts)
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    if value is None:
        return ""
    return str(value)
