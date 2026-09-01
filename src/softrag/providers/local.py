"""Fully local embedding and reranking with sentence-transformers.

Installed via ``pip install 'softrag[local]'``. Once the model is cached, an
engine built on these backends needs no network at all -- which, for a
local-first library, is the configuration everything else is measured against.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, List, Optional, Sequence

from ..errors import EmbeddingError, MissingDependencyError
from ..types import Hit

log = logging.getLogger("softrag.providers.local")

__all__ = ["SentenceTransformerEmbedder", "CrossEncoderReranker", "is_available"]

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def is_available() -> bool:
    """Whether ``sentence-transformers`` is importable.

    Checked via :mod:`importlib.util` rather than a real import, because
    importing sentence-transformers pulls in torch and costs seconds -- too slow
    for a detection probe.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


class SentenceTransformerEmbedder:
    """Embeddings from a local sentence-transformers model.

    Args:
        model: Model id or local path.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` to let the library
            choose.
        batch_size: Texts encoded per forward pass.
        normalize: L2-normalise the output vectors. Left on, since cosine
            distance over normalised vectors is what the index expects.
        prompt_name: Model-specific prompt for the query side, used by models
            trained with asymmetric query/document prefixes (E5, BGE, GTE).

    Example:
        >>> embedder = SentenceTransformerEmbedder()      # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        prompt_name: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MissingDependencyError(
                "sentence-transformers", extra="local", feature="Local embeddings"
            ) from exc

        self.model_name = model
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.prompt_name = prompt_name
        log.debug("loading sentence-transformers model %s", model)
        self.model = SentenceTransformer(model, device=device)

    @property
    def dimensions(self) -> int:
        """Width of the vectors this model produces."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed_query(self, text: str) -> List[float]:
        kwargs: dict[str, Any] = {}
        if self.prompt_name:
            kwargs["prompt_name"] = self.prompt_name
        return self._encode([text], **kwargs)[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def _encode(self, texts: Sequence[str], **kwargs: Any) -> List[List[float]]:
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
                **kwargs,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"{self.model_name} failed to encode {len(texts)} texts: {exc}"
            ) from exc
        return [[float(v) for v in row] for row in vectors]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SentenceTransformerEmbedder(model={self.model_name!r})"


class CrossEncoderReranker:
    """Rerank hits with a local cross-encoder.

    A bi-encoder embeds the query and each document separately, so it can index
    ahead of time but never sees the pair together. A cross-encoder scores
    ``(query, document)`` jointly, which is markedly more accurate and markedly
    slower -- exactly right as a second stage over a few dozen candidates.

    Args:
        model: Cross-encoder model id.
        device: Torch device, or ``None`` to let the library choose.
        batch_size: Pairs scored per forward pass.
        max_length: Token limit per pair.

    Example:
        >>> rag = Rag(reranker=CrossEncoderReranker())      # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise MissingDependencyError(
                "sentence-transformers", extra="rerank", feature="Cross-encoder reranking"
            ) from exc

        self.model_name = model
        self.batch_size = max(1, batch_size)
        log.debug("loading cross-encoder %s", model)
        self.model = CrossEncoder(model, max_length=max_length, device=device)

    def rerank(self, query: str, hits: Sequence[Hit], *, top_k: int) -> List[Hit]:
        """Score every hit against the query and return the best ``top_k``."""
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        try:
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )
        except Exception as exc:
            log.warning("reranking failed (%s); keeping the original order", exc)
            return list(hits[:top_k])

        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        out: List[Hit] = []
        for hit, score in ranked[:top_k]:
            hit.score = float(score)
            out.append(hit)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CrossEncoderReranker(model={self.model_name!r})"
