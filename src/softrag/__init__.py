"""SoftRAG library for local-first Retrieval-Augmented Generation."""

from .softrag import Rag, EmbedFn, ChatFn
from importlib.metadata import version

__version__ = version("softrag")
__all__ = ["Rag", "EmbedFn", "ChatFn"]
