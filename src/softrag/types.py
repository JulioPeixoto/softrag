"""Core data types and backend protocols.

The types here are the contract between the storage layer, the retrieval layer
and the user. They are deliberately plain: dataclasses and protocols, no
third-party base classes, so softrag stays dependency-light and any object with
the right shape can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [
    "Chunk",
    "Hit",
    "Answer",
    "StreamingAnswer",
    "IngestResult",
    "SourceInfo",
    "Stats",
    "Embedder",
    "ChatModel",
    "StreamingChatModel",
    "Reranker",
    "Extractor",
    "Where",
]

#: Metadata filter expression. Values may be scalars (equality) or a single-key
#: operator mapping such as ``{"$gte": 2020}``. See :mod:`softrag.filters`.
Where = Mapping[str, Any]


@dataclass(slots=True)
class Chunk:
    """A unit of indexable text plus the metadata travelling with it."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Position of this chunk inside its source document, starting at 0.
    index: int = 0
    #: Stable identifier of the document this chunk was cut from.
    source: str = ""

    def __len__(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class Hit:
    """A single retrieved chunk with the score that earned it its place."""

    id: int
    text: str
    score: float
    source: str = ""
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Cosine distance from the query vector, when vector search contributed.
    vector_distance: Optional[float] = None
    #: Raw BM25 score, when keyword search contributed. Lower is a better match.
    bm25: Optional[float] = None
    #: 1-based rank in each contributing list, keyed by "vector" / "keyword".
    ranks: Dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        preview = self.text[:60].replace("\n", " ")
        if len(self.text) > 60:
            preview += "..."
        where = f" source={self.source!r}" if self.source else ""
        return f"<Hit score={self.score:.4f}{where} text={preview!r}>"


class Answer(str):
    """The generated answer.

    Subclasses :class:`str` so the common case stays trivial::

        print(rag.query("..."))

    while the provenance remains one attribute away::

        answer = rag.query("...")
        for hit in answer.hits:
            print(hit.source, hit.score)
    """

    __slots__ = ("hits", "question", "prompt")

    hits: List[Hit]
    question: str
    prompt: str

    def __new__(
        cls,
        text: str,
        *,
        hits: Sequence[Hit] = (),
        question: str = "",
        prompt: str = "",
    ) -> "Answer":
        self = super().__new__(cls, text)
        self.hits = list(hits)
        self.question = question
        self.prompt = prompt
        return self

    @property
    def sources(self) -> List[str]:
        """Unique source identifiers behind this answer, best-scoring first."""
        seen: Dict[str, None] = {}
        for hit in self.hits:
            if hit.source:
                seen.setdefault(hit.source, None)
        return list(seen)

    @property
    def context(self) -> str:
        """The retrieved context exactly as it was shown to the model."""
        return "\n\n".join(hit.text for hit in self.hits)


class StreamingAnswer:
    """A lazily generated answer.

    Iterating yields text deltas. The retrieved :attr:`hits` are available
    immediately -- retrieval happens before generation -- and :attr:`text` holds
    whatever has been produced so far.
    """

    __slots__ = ("_stream", "hits", "question", "prompt", "_parts", "_done")

    def __init__(
        self,
        stream: Iterable[str],
        *,
        hits: Sequence[Hit] = (),
        question: str = "",
        prompt: str = "",
    ) -> None:
        self._stream = stream
        self.hits: List[Hit] = list(hits)
        self.question = question
        self.prompt = prompt
        self._parts: List[str] = []
        self._done = False

    def __iter__(self) -> Iterator[str]:
        if self._done:
            yield "".join(self._parts)
            return
        for delta in self._stream:
            self._parts.append(delta)
            yield delta
        self._done = True

    @property
    def text(self) -> str:
        """Text produced so far (the full answer once iteration finished)."""
        return "".join(self._parts)

    @property
    def sources(self) -> List[str]:
        seen: Dict[str, None] = {}
        for hit in self.hits:
            if hit.source:
                seen.setdefault(hit.source, None)
        return list(seen)

    def collect(self) -> Answer:
        """Drain the stream and return a complete :class:`Answer`."""
        for _ in self:
            pass
        return Answer(
            self.text, hits=self.hits, question=self.question, prompt=self.prompt
        )


@dataclass(slots=True)
class IngestResult:
    """What an ``add*`` call actually did."""

    source: str
    chunks_added: int = 0
    chunks_skipped: int = 0
    chunks_deleted: int = 0
    characters: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.error:
            return f"<IngestResult {self.source!r} FAILED: {self.error}>"
        return (
            f"<IngestResult {self.source!r} added={self.chunks_added} "
            f"skipped={self.chunks_skipped} deleted={self.chunks_deleted}>"
        )


@dataclass(slots=True)
class SourceInfo:
    """A document currently present in the index."""

    source: str
    chunks: int
    characters: int
    added_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Stats:
    """A summary of what a database holds."""

    path: str
    documents: int
    sources: int
    dimensions: Optional[int]
    size_bytes: int
    schema_version: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Backend protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors.

    A bare callable ``str -> list[float]`` also works: it is adapted internally.
    LangChain ``Embeddings`` objects satisfy this protocol as-is.
    """

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed a single string, optimised for the query side if applicable."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed a batch of strings. Order of the output must match the input."""
        ...


@runtime_checkable
class ChatModel(Protocol):
    """Anything that turns a prompt into text."""

    def complete(self, prompt: str) -> str:
        """Return the full completion for ``prompt``."""
        ...


@runtime_checkable
class StreamingChatModel(ChatModel, Protocol):
    """A chat backend that can emit incremental deltas."""

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield text deltas for ``prompt``."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate hits against the query, most relevant first."""

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> List[Hit]:
        ...


@runtime_checkable
class Extractor(Protocol):
    """Turns bytes of some format into plain text."""

    def extract(self, data: bytes, *, filename: str = "") -> str:
        ...
