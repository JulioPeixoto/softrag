"""Text chunking.

softrag ships its own splitters rather than depending on a framework: chunking
is a hundred lines of string handling, and owning it keeps ``pip install
softrag`` small and the behaviour predictable across versions.

Every chunker is a callable ``str -> list[str]``, so a plain function or lambda
is a drop-in replacement anywhere a chunker is accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

__all__ = [
    "Chunker",
    "RecursiveChunker",
    "MarkdownChunker",
    "SentenceChunker",
    "by_separator",
    "resolve_chunker",
]

#: Anything that cuts a document into pieces.
Chunker = Callable[[str], List[str]]

DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 200

#: Ordered from "most semantic" to "last resort". The splitter walks this list
#: and uses the first separator that actually breaks the text down far enough.
DEFAULT_SEPARATORS: Sequence[str] = (
    "\n\n\n",  # section break
    "\n\n",  # paragraph
    "\n",  # line
    ". ",  # sentence
    "? ",
    "! ",
    "; ",
    ", ",
    " ",  # word
    "",  # character
)

_SENTENCE_END = re.compile(r"(?<=[.!?])[\s\n]+(?=[A-Z0-9\"'(\[])")
_MD_HEADING = re.compile(r"^(#{1,6})\s+.*$", re.MULTILINE)


@dataclass(slots=True)
class RecursiveChunker:
    """Split text on progressively finer separators until chunks fit.

    This is the default strategy. It tries to keep semantically related text
    together by preferring paragraph boundaries over line breaks over words, and
    only ever falls back to hard character slicing when a single unbroken run of
    text is longer than ``chunk_size``.

    Args:
        chunk_size: Target maximum size of a chunk, in units of ``length``.
        chunk_overlap: How much of the tail of one chunk to repeat at the head of
            the next, so a fact spanning a boundary survives in at least one
            chunk. Must be smaller than ``chunk_size``.
        separators: Boundary strings to try, most semantic first.
        length: Size function. Defaults to character count; pass a tokenizer's
            length function to chunk by tokens instead.
        keep_separator: Keep the separator at the end of the preceding chunk.
            Keeps sentences readable; turn off for delimiter-style splitting.
        strip: Strip surrounding whitespace from each emitted chunk.

    Example:
        >>> chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        >>> chunker("alpha beta gamma delta epsilon")
        ['alpha beta gamma', 'gamma delta epsilon']
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: Sequence[str] = field(default=DEFAULT_SEPARATORS)
    length: Callable[[str], int] = len
    keep_separator: bool = True
    strip: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunks never advance."
            )

    def __call__(self, text: str) -> List[str]:
        return self.split(text)

    def split(self, text: str) -> List[str]:
        """Cut ``text`` into overlapping chunks."""
        if not text or not text.strip():
            return []
        pieces = self._split_recursive(text, list(self.separators))
        return self._merge(pieces)

    # -- internals ---------------------------------------------------------- #

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Break ``text`` into pieces that each fit, or cannot be broken further."""
        if self.length(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            return self._hard_split(text)

        separator, *rest = separators
        if separator == "":
            return self._hard_split(text)

        parts = text.split(separator)
        if len(parts) == 1:
            # This separator does not occur; try the next one without recursing.
            return self._split_recursive(text, rest)

        out: List[str] = []
        for i, part in enumerate(parts):
            if self.keep_separator and i < len(parts) - 1:
                part = part + separator
            if not part:
                continue
            if self.length(part) <= self.chunk_size:
                out.append(part)
            else:
                out.extend(self._split_recursive(part, rest))
        return out

    def _hard_split(self, text: str) -> List[str]:
        """Last resort: slice an unbreakable run at fixed width."""
        size = self.chunk_size
        return [text[i : i + size] for i in range(0, len(text), size)] or []

    def _merge(self, pieces: Sequence[str]) -> List[str]:
        """Greedily pack small pieces back up to ``chunk_size``, with overlap."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = self.length(piece)
            if current and current_len + piece_len > self.chunk_size:
                chunks.append(self._emit(current))
                current, current_len = self._carry_over(current)
            current.append(piece)
            current_len += piece_len

        if current:
            chunks.append(self._emit(current))

        return [c for c in chunks if c]

    def _carry_over(self, current: List[str]) -> tuple[List[str], int]:
        """Take the tail of the finished chunk to seed the next one."""
        if self.chunk_overlap == 0:
            return [], 0
        tail: List[str] = []
        tail_len = 0
        for piece in reversed(current):
            piece_len = self.length(piece)
            if tail_len + piece_len > self.chunk_overlap and tail:
                break
            tail.insert(0, piece)
            tail_len += piece_len
        return tail, tail_len

    def _emit(self, pieces: Sequence[str]) -> str:
        text = "".join(pieces)
        return text.strip() if self.strip else text


@dataclass(slots=True)
class MarkdownChunker:
    """Chunk Markdown along its heading structure.

    Sections are kept whole when they fit, and the heading trail (for example
    ``"# Guide > ## Install"``) is prepended to every chunk so an isolated chunk
    still says what it is about -- which measurably helps both keyword and
    vector retrieval.

    Args:
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap applied when a section must be split further.
        include_heading_trail: Prefix each chunk with its heading breadcrumb.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    include_heading_trail: bool = True

    def __call__(self, text: str) -> List[str]:
        return self.split(text)

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        inner = RecursiveChunker(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        sections = self._sections(text)
        if not sections:
            return inner.split(text)

        chunks: List[str] = []
        for trail, body in sections:
            body = body.strip()
            if not body:
                continue
            prefix = f"{trail}\n\n" if (trail and self.include_heading_trail) else ""
            budget = self.chunk_size - len(prefix)
            if budget <= 0 or len(body) <= budget:
                chunks.append(prefix + body)
                continue
            sized = RecursiveChunker(
                chunk_size=max(budget, 1), chunk_overlap=self.chunk_overlap
            )
            chunks.extend(prefix + part for part in sized.split(body))
        return chunks or inner.split(text)

    def _sections(self, text: str) -> List[tuple[str, str]]:
        """Return ``(heading_trail, body)`` pairs in document order."""
        matches = list(_MD_HEADING.finditer(text))
        if not matches:
            return []

        sections: List[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        trail: List[str] = []
        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(0).strip()
            del trail[level - 1 :]
            while len(trail) < level - 1:
                trail.append("")
            trail.append(heading)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.start() : end]
            sections.append((" > ".join(h for h in trail if h), body))
        return sections


@dataclass(slots=True)
class SentenceChunker:
    """Group whole sentences into chunks, overlapping by whole sentences.

    Useful when chunks are shown to a human (citations, snippets) and a chunk
    ending mid-sentence would look broken.

    Args:
        chunk_size: Target maximum chunk size in characters.
        overlap_sentences: How many trailing sentences to repeat in the next chunk.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap_sentences: int = 1

    def __call__(self, text: str) -> List[str]:
        return self.split(text)

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) + 1 > self.chunk_size:
                chunks.append(" ".join(current))
                keep = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current = list(keep)
                current_len = sum(len(s) + 1 for s in current)
            current.append(sentence)
            current_len += len(sentence) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks


def by_separator(separator: str, *, strip: bool = True) -> Chunker:
    """Build a chunker that splits on a literal delimiter.

    Args:
        separator: The delimiter to split on.
        strip: Strip whitespace and drop empty pieces.

    Returns:
        A callable suitable for the ``chunker`` argument.
    """

    def _split(text: str) -> List[str]:
        parts = text.split(separator)
        if not strip:
            return parts
        return [p.strip() for p in parts if p.strip()]

    return _split


def resolve_chunker(chunker: Chunker | str | None, **defaults: object) -> Chunker:
    """Normalise the many accepted chunker spellings into a callable.

    Accepts ``None`` (the default recursive chunker), a literal separator
    string, or any callable.

    Raises:
        TypeError: If ``chunker`` is none of those.
    """
    if chunker is None:
        return RecursiveChunker(**defaults)  # type: ignore[arg-type]
    if isinstance(chunker, str):
        return by_separator(chunker)
    if callable(chunker):
        return chunker
    raise TypeError(
        f"chunker must be None, a separator string, or a callable, "
        f"got {type(chunker).__name__}"
    )
