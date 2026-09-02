"""softrag -- the embedded RAG engine.

One SQLite file holds your documents, their embeddings and a full-text index.
No server, no cluster, no vendor account: an index is a file you can copy,
commit, ship or delete.

    import softrag

    rag = softrag.connect("kb.db")
    rag.add("handbook.pdf")
    rag.add("https://example.com/changelog")

    answer = rag.query("What changed in the refund policy?")
    print(answer)
    print(answer.sources)

Retrieval is hybrid by default -- dense vectors for meaning, BM25 for exact
terms, combined with Reciprocal Rank Fusion -- and every model backend is
pluggable, including fully local ones.
"""

from __future__ import annotations

import logging

# Imported eagerly so the documented
# ``softrag.ingest.EXTRACTORS[".rtf"] = fn`` works after a bare
# ``import softrag``. It costs about 4 ms and pulls only standard-library
# modules. ``providers`` and ``stopwords`` bind themselves via the imports
# below.
from . import ingest
from .aengine import AsyncRag, AsyncStreamingAnswer, connect_async
from .chunking import (
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    by_separator,
)
from .engine import DEFAULT_PROMPT, Rag, RagConfig, connect
from .errors import (
    ChatError,
    ConfigurationError,
    DimensionMismatchError,
    EmbeddingError,
    ExtractionError,
    IngestionError,
    MissingDependencyError,
    ProviderError,
    SchemaVersionError,
    SoftragError,
    StoreError,
    UnsupportedFormatError,
)
from .eval import EvalResult, compare, evaluate, evaluate_engine
from .providers import EchoChatModel, HashEmbedder, adapt_chat_model, adapt_embedder
from .rerank import (
    ChainReranker,
    DedupeReranker,
    LLMReranker,
    ScoreFusionReranker,
)
from .retrieval import (
    RetrievalConfig,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from .store import Store
from .transforms import ContextualChunker, contextualize, expand_query, hyde
from .types import (
    Answer,
    ChatModel,
    Chunk,
    Embedder,
    Hit,
    IngestResult,
    Reranker,
    SourceInfo,
    Stats,
    StreamingAnswer,
)

__all__ = [
    # Entry points
    "Rag",
    "AsyncRag",
    "AsyncStreamingAnswer",
    "connect",
    "connect_async",
    "RagConfig",
    "DEFAULT_PROMPT",
    # Results
    "Answer",
    "StreamingAnswer",
    "Hit",
    "Chunk",
    "IngestResult",
    "SourceInfo",
    "Stats",
    # Protocols
    "Embedder",
    "ChatModel",
    "Reranker",
    # Retrieval
    "RetrievalConfig",
    "ChainReranker",
    "DedupeReranker",
    "LLMReranker",
    "ScoreFusionReranker",
    "reciprocal_rank_fusion",
    "maximal_marginal_relevance",
    "Store",
    # Chunking
    "RecursiveChunker",
    "MarkdownChunker",
    "SentenceChunker",
    "by_separator",
    # Evaluation and query transforms
    "EvalResult",
    "evaluate",
    "evaluate_engine",
    "compare",
    "hyde",
    "expand_query",
    "contextualize",
    "ContextualChunker",
    # Backends
    "HashEmbedder",
    "EchoChatModel",
    "adapt_embedder",
    "adapt_chat_model",
    "ingest",
    # Errors
    "SoftragError",
    "ConfigurationError",
    "MissingDependencyError",
    "StoreError",
    "SchemaVersionError",
    "DimensionMismatchError",
    "IngestionError",
    "UnsupportedFormatError",
    "ExtractionError",
    "ProviderError",
    "EmbeddingError",
    "ChatError",
    "__version__",
]


def _detect_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("softrag")
    except PackageNotFoundError:
        # Running from a source checkout that was never installed.
        return "0.0.0+local"


__version__ = _detect_version()

# A library should be silent unless the application asks otherwise.
logging.getLogger(__name__).addHandler(logging.NullHandler())
