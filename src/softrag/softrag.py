"""Backwards-compatible import path.

``softrag.softrag`` was the single module the library used to live in. It now
re-exports the public API so imports written against the old layout keep
working; new code should import from :mod:`softrag` directly.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from .engine import Rag, RagConfig, connect
from .store import Store, pack_vector
from .types import Answer, Hit

#: Historical type aliases, kept so old annotations still resolve.
EmbedFn = Callable[[str], List[float]]
ChatFn = Callable[[str, Sequence[str]], str]

__all__ = [
    "Rag",
    "RagConfig",
    "connect",
    "Store",
    "Answer",
    "Hit",
    "EmbedFn",
    "ChatFn",
    "pack_vector",
]
