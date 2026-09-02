"""Shared fixtures and fakes for the softrag test suite.

Everything here is offline and deterministic: no network, no API keys, no model
downloads. The real backends are replaced by :class:`softrag.HashEmbedder`,
:class:`softrag.EchoChatModel` and the small fakes below.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from softrag import EchoChatModel, HashEmbedder, Rag

#: Vector width used everywhere unless a test needs a different one. Small
#: enough to keep the suite fast, wide enough that hashing collisions do not
#: dominate the rankings.
DIM = 64


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """A deterministic embedder with a controllable width.

    Vectors are token-hashed, so documents sharing words land near each other.
    That is enough to make ordering assertions meaningful without a real model.
    """

    def __init__(self, dimensions: int = DIM) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import math

        vector = [0.0] * self.dimensions
        for token in text.lower().split() or [""]:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


class CountingEmbedder:
    """Wraps a real embedder and records exactly how it was called.

    ``batch_calls`` counts :meth:`embed_documents` invocations, ``batch_sizes``
    records each batch's length, and ``texts`` accumulates every string that was
    actually embedded -- which is what batching and dedup assertions need.
    """

    def __init__(self, dimensions: int = DIM) -> None:
        self._inner = HashEmbedder(dimensions=dimensions)
        self.dimensions = dimensions
        self.query_calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
        self.texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._inner.embed_query(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        self.texts.extend(texts)
        return self._inner.embed_documents(texts)


class RecordingChatModel:
    """A chat model that records every prompt it was handed."""

    def __init__(self, reply: str = "RECORDED ANSWER") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    @property
    def prompt(self) -> str:
        """The most recent prompt, or ``""`` when never called."""
        return self.prompts[-1] if self.prompts else ""

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    def stream(self, prompt: str) -> Iterator[str]:
        self.prompts.append(prompt)
        for word in self.reply.split(" "):
            yield word + " "


class OrderingReranker:
    """A reranker that imposes a fixed source order and records its inputs."""

    def __init__(self, order: Sequence[str]) -> None:
        self.order = list(order)
        self.calls: list[tuple] = []

    def rerank(self, query: str, hits: Sequence[Any], *, top_k: int) -> list[Any]:
        self.calls.append((query, len(hits), top_k))
        rank = {source: i for i, source in enumerate(self.order)}
        return sorted(hits, key=lambda hit: rank.get(hit.source, len(rank)))[:top_k]


# --------------------------------------------------------------------------- #
# Engine fixtures
# --------------------------------------------------------------------------- #


def make_rag(db_path: str = ":memory:", **kwargs: Any) -> Rag:
    """Build an offline engine. Every test-owned engine goes through here."""
    kwargs.setdefault("embed_model", HashEmbedder(dimensions=DIM))
    kwargs.setdefault("chat_model", EchoChatModel())
    return Rag(db_path=db_path, **kwargs)


@pytest.fixture
def make_engine() -> Iterator[Any]:
    """Factory fixture: build offline engines that are closed at teardown."""
    engines: list[Rag] = []

    def factory(db_path: str = ":memory:", **kwargs: Any) -> Rag:
        engine = make_rag(db_path, **kwargs)
        engines.append(engine)
        return engine

    try:
        yield factory
    finally:
        for engine in engines:
            engine.close()


@pytest.fixture
def rag() -> Iterator[Rag]:
    """An in-memory engine with offline models."""
    engine = make_rag()
    try:
        yield engine
    finally:
        engine.close()


@pytest.fixture
def tmp_rag(tmp_path) -> Iterator[Rag]:
    """A file-backed engine living under ``tmp_path``."""
    engine = make_rag(str(tmp_path / "kb.db"))
    try:
        yield engine
    finally:
        engine.close()


@pytest.fixture
def recorder() -> RecordingChatModel:
    return RecordingChatModel()


@pytest.fixture
def recording_rag(recorder: RecordingChatModel) -> Iterator[Rag]:
    """An in-memory engine whose chat model captures the rendered prompt."""
    engine = make_rag(chat_model=recorder)
    try:
        yield engine
    finally:
        engine.close()


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #

#: A small, fixed corpus: ``(source, text, metadata)``. Kept deliberately tiny
#: and lexically distinct so both retrievers have something unambiguous to find.
CORPUS: list[tuple] = [
    (
        "handbook",
        "The refund policy allows returns within thirty days of purchase. "
        "Refunds are issued to the original payment method.",
        {"year": 2024, "kind": "policy", "tags": ["refund", "billing"], "public": True},
    ),
    (
        "changelog",
        "Version two introduced hybrid retrieval and the zqxwv7 build identifier. "
        "Older versions only supported keyword search.",
        {"year": 2024, "kind": "release", "tags": ["changelog"], "public": True},
    ),
    (
        "biology",
        "Mitochondria are the powerhouse of the cell, producing adenosine "
        "triphosphate through oxidative phosphorylation.",
        {"year": 2019, "kind": "science", "tags": ["cells"], "public": False},
    ),
    (
        "cooking",
        "To braise short ribs, sear them first and then simmer them slowly in "
        "stock with aromatic vegetables.",
        {"year": 2021, "kind": "recipe", "tags": ["food", "meat"], "public": True},
    ),
]


@pytest.fixture
def corpus(rag: Rag) -> Rag:
    """An engine preloaded with :data:`CORPUS`, one chunk per document."""
    for source, text, metadata in CORPUS:
        result = rag.add_text(text, name=source, metadata=metadata)
        assert result.ok, result.error
    return rag
