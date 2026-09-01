"""The engine: ingestion, retrieval and generation tied together.

:class:`Rag` is the one object most users ever touch. It owns a
:class:`~softrag.store.Store`, an embedder, an optional chat model, and the
policy that turns "here is a PDF" into "here is an answer with citations".
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from .chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunker,
    resolve_chunker,
)
from .errors import ConfigurationError, IngestionError
from .providers import adapt_chat_model, adapt_embedder, embedder_fingerprint
from .retrieval import RetrievalConfig, Retriever, SearchMode
from .store import Store
from .types import (
    Answer,
    ChatModel,
    Embedder,
    Hit,
    IngestResult,
    Reranker,
    SourceInfo,
    Stats,
    StreamingAnswer,
    Where,
)

log = logging.getLogger("softrag")

__all__ = ["Rag", "RagConfig", "connect", "DEFAULT_PROMPT"]

DEFAULT_PROMPT = """\
Answer the question using only the context below. Each context block is \
numbered; cite the blocks you relied on as [1], [2] and so on.

If the context does not contain the answer, say so plainly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

#: Input shapes accepted by :meth:`Rag.add`.
Source = Union[str, Path, bytes, bytearray]

ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class RagConfig:
    """Everything tunable about an engine, in one inspectable object.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Characters repeated between consecutive chunks.
        top_k: Default number of chunks retrieved per query.
        mode: Default search mode -- ``"hybrid"``, ``"vector"`` or ``"keyword"``.
        vector_weight: Relative influence of dense search during fusion.
        keyword_weight: Relative influence of BM25 during fusion.
        candidates: Candidates each retriever contributes before fusion.
        diversity: MMR diversity in ``[0, 1]``; 0 disables the diversity pass.
        expand_context: Neighbouring chunks to attach to each hit.
        prompt: Prompt template with ``{context}`` and ``{question}`` fields.
        embed_batch_size: Chunks embedded per backend call.
        max_workers: Threads used by :meth:`Rag.add_many`.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = 5
    mode: SearchMode = "hybrid"
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    candidates: Optional[int] = None
    diversity: float = 0.0
    expand_context: int = 0
    prompt: str = DEFAULT_PROMPT
    embed_batch_size: int = 64
    max_workers: int = 4

    def retrieval(self, **overrides: Any) -> RetrievalConfig:
        """Build a :class:`RetrievalConfig`, applying per-call overrides."""
        base = RetrievalConfig(
            mode=self.mode,
            top_k=self.top_k,
            candidates=self.candidates,
            vector_weight=self.vector_weight,
            keyword_weight=self.keyword_weight,
            diversity=self.diversity,
            expand_context=self.expand_context,
        )
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(base, **clean) if clean else base


class Rag:
    """A local-first retrieval-augmented generation engine.

    Everything -- chunk text, metadata, the vector index and the full-text
    index -- lives in a single SQLite file, so an index is a file you can copy,
    version, ship or delete.

    Args:
        embed_model: Any embedder. A LangChain ``Embeddings`` object, a
            sentence-transformers model, one of ``softrag.providers``, or a
            plain callable all work. Defaults to auto-detection.
        chat_model: Any chat backend, or ``None`` for a retrieval-only engine.
            Defaults to auto-detection.
        db_path: Where to keep the index. ``":memory:"`` for an ephemeral one.
        chunker: ``None`` for the default recursive chunker, a separator string,
            or any callable that maps text to a list of chunks.
        config: Tuning knobs. See :class:`RagConfig`.
        reranker: Optional second-stage reranker applied to every search.
        auto: Fill in missing models from the environment. Set ``False`` to make
            a missing model an error instead.

    Example:
        >>> rag = Rag(db_path=":memory:")             # doctest: +SKIP
        >>> rag.add("handbook.pdf")                   # doctest: +SKIP
        >>> answer = rag.query("What is the refund policy?")   # doctest: +SKIP
        >>> print(answer, answer.sources)             # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embed_model: Any = None,
        chat_model: Any = None,
        db_path: str | os.PathLike = "softrag.db",
        chunker: Chunker | str | None = None,
        config: Optional[RagConfig] = None,
        reranker: Optional[Reranker] = None,
        auto: bool = True,
        **overrides: Any,
    ) -> None:
        self.config = config or RagConfig()
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise ConfigurationError(
                    f"Unknown option {key!r}. Valid options: "
                    f"{', '.join(sorted(self.config.__slots__))}."
                )
            setattr(self.config, key, value)

        if embed_model is None:
            if not auto:
                raise ConfigurationError(
                    "embed_model is required when auto=False. Pass an embedder, "
                    "or leave auto=True to detect one from the environment."
                )
            from .providers import auto_embedder

            embed_model = auto_embedder()

        self.embedder: Embedder = adapt_embedder(embed_model)
        self._chat: Optional[ChatModel] = None
        self._chat_source = chat_model
        self._auto_chat = auto and chat_model is None
        if chat_model is not None:
            self._chat = adapt_chat_model(chat_model)

        self.store = Store(db_path)
        self.embedder_id = embedder_fingerprint(embed_model)
        #: Set when the index was built with a different embedding model than
        #: the one configured now. See :meth:`softrag.store.Store.check_embedder`.
        self.embedder_changed = self.store.check_embedder(self.embedder_id)
        self.retriever = Retriever(self.store)
        self.reranker = reranker
        self._chunker = resolve_chunker(
            chunker,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def chat_model(self) -> ChatModel:
        """The chat backend, resolved lazily so retrieval-only use costs nothing."""
        if self._chat is None:
            if not self._auto_chat:
                raise ConfigurationError(
                    "This engine has no chat model, so it can retrieve but not "
                    "generate. Pass chat_model=... to Rag(), or use rag.search()."
                )
            from .providers import auto_chat_model

            self._chat = adapt_chat_model(auto_chat_model())
        return self._chat

    def close(self) -> None:
        """Close the underlying database. Safe to call more than once."""
        self.store.close()

    def __enter__(self) -> "Rag":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        stats = self.store.stats()
        return (
            f"<Rag {stats.path!r} chunks={stats.documents} "
            f"sources={stats.sources} dim={stats.dimensions}>"
        )

    def __len__(self) -> int:
        return self.store.count()

    # -- ingestion ---------------------------------------------------------- #

    def add(
        self,
        source: Source,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        chunker: Chunker | str | None = None,
        on_change: str = "replace",
    ) -> IngestResult:
        """Index anything: a file path, a URL, a directory, or raw text.

        The kind of input is inferred, so a single call handles the common
        cases. Reach for :meth:`add_file`, :meth:`add_web` or :meth:`add_text`
        when you want to be explicit.

        Args:
            source: A path, a URL, a glob, a directory, raw text, or bytes.
            metadata: Metadata attached to every chunk, filterable at query time.
            name: Override the stored source identifier.
            chunker: Chunking strategy for this document only.
            on_change: What to do when this source is already indexed with
                different content -- ``"replace"`` (re-index), ``"skip"`` (leave
                the old version alone) or ``"append"`` (keep both).

        Returns:
            An :class:`~softrag.types.IngestResult` describing what changed.

        Raises:
            IngestionError: If the input cannot be turned into text.
        """
        if isinstance(source, (bytes, bytearray)):
            return self.add_file(source, metadata=metadata, name=name, chunker=chunker,
                                 on_change=on_change)

        text = str(source)
        if _looks_like_url(text):
            return self.add_web(text, metadata=metadata, name=name, chunker=chunker,
                                on_change=on_change)

        path = Path(text)
        if path.is_dir():
            raise IngestionError(
                f"{text!r} is a directory. Use rag.add_directory({text!r}) to index "
                "its contents, which lets you control the file pattern and "
                "recursion."
            )
        if any(ch in text for ch in "*?[") and not path.exists():
            raise IngestionError(
                f"{text!r} looks like a glob pattern. Use "
                f"rag.add_directory(base_dir, pattern={text!r}) instead."
            )
        if path.exists():
            return self.add_file(path, metadata=metadata, name=name, chunker=chunker,
                                 on_change=on_change)

        return self.add_text(text, metadata=metadata, name=name, chunker=chunker,
                             on_change=on_change)

    def add_text(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        chunker: Chunker | str | None = None,
        on_change: str = "replace",
    ) -> IngestResult:
        """Index a string directly.

        Args:
            text: The content to index.
            metadata: Metadata attached to every chunk.
            name: Source identifier. Defaults to a content-derived name, so the
                same text added twice is recognised as the same source.
            chunker: Chunking strategy for this document only.
            on_change: See :meth:`add`.
        """
        if not text or not text.strip():
            return IngestResult(source=name or "", error="empty content")
        identifier = name or f"text:{_digest(text)[:12]}"
        return self._ingest(identifier, text, metadata, chunker, on_change)

    def add_file(
        self,
        path: Source,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        chunker: Chunker | str | None = None,
        on_change: str = "replace",
    ) -> IngestResult:
        """Index a file: PDF, DOCX, Markdown, HTML, CSV, JSON, code or plain text.

        Args:
            path: A path, or raw bytes with ``name`` set so the format can be
                inferred from the extension.
            metadata: Extra metadata; file name, extension and size are added
                automatically.
            name: Override the stored source identifier.
            chunker: Chunking strategy for this document only.
            on_change: See :meth:`add`.

        Raises:
            IngestionError: If no extractor handles this format, or extraction
                fails.
        """
        from .ingest import extract_file

        text, detected, extra = extract_file(path, name=name)
        identifier = name or detected
        merged = {**extra, **(metadata or {})}
        return self._ingest(identifier, text, merged, chunker, on_change)

    def add_web(
        self,
        url: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        chunker: Chunker | str | None = None,
        on_change: str = "replace",
        timeout: float = 30.0,
    ) -> IngestResult:
        """Fetch a URL and index its main text content.

        Boilerplate -- navigation, footers, cookie banners -- is stripped when
        the optional ``trafilatura`` extractor is installed, and falls back to a
        built-in HTML-to-text pass otherwise.

        Args:
            url: The page to fetch.
            metadata: Extra metadata; the URL and page title are added
                automatically.
            name: Override the stored source identifier. Defaults to the URL.
            chunker: Chunking strategy for this document only.
            on_change: See :meth:`add`.
            timeout: Network timeout in seconds.
        """
        from .ingest import extract_web

        text, extra = extract_web(url, timeout=timeout)
        merged = {"url": url, **extra, **(metadata or {})}
        return self._ingest(name or url, text, merged, chunker, on_change)

    def add_image(
        self,
        path: Source,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        on_change: str = "replace",
    ) -> IngestResult:
        """Describe an image with the chat model and index the description.

        Images are made searchable by turning them into text: a vision-capable
        chat model writes a description, and that description is indexed like
        any other document, so one query searches text and images together.

        Args:
            path: Path to the image.
            metadata: Extra metadata; ``kind="image"`` and the path are added.
            name: Override the stored source identifier.
            prompt: Override the captioning instruction.
            on_change: See :meth:`add`.

        Raises:
            IngestionError: If the file is missing or captioning fails.
        """
        from .ingest import caption_image

        image_path = Path(str(path))
        if not image_path.exists():
            raise IngestionError(f"Image not found: {image_path}")

        caption = caption_image(image_path, self.chat_model, prompt=prompt)
        merged = {
            "kind": "image",
            "path": str(image_path),
            "filename": image_path.name,
            **(metadata or {}),
        }
        body = f"Image: {image_path.name}\n\n{caption}"
        return self._ingest(name or str(image_path), body, merged, None, on_change)

    def add_directory(
        self,
        directory: str | os.PathLike,
        *,
        pattern: str = "**/*",
        exclude: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = True,
        on_progress: Optional[ProgressCallback] = None,
        ignore_errors: bool = True,
    ) -> List[IngestResult]:
        """Index every supported file under a directory.

        Args:
            directory: The directory to walk.
            pattern: Glob pattern, relative to ``directory``.
            exclude: Glob patterns to skip, such as ``("**/node_modules/**",)``.
            metadata: Metadata attached to every file indexed.
            recursive: Walk subdirectories.
            on_progress: Called as ``(source, done, total)`` after each file.
            ignore_errors: Record per-file failures in the results instead of
                raising on the first one.

        Returns:
            One :class:`~softrag.types.IngestResult` per file attempted.
        """
        from .ingest import discover_files

        base = Path(directory)
        if not base.is_dir():
            raise IngestionError(f"Not a directory: {base}")

        files = discover_files(
            base, pattern=pattern, exclude=exclude, recursive=recursive
        )
        return self.add_many(
            files,
            metadata=metadata,
            on_progress=on_progress,
            ignore_errors=ignore_errors,
        )

    def add_many(
        self,
        sources: Iterable[Source],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
        ignore_errors: bool = True,
    ) -> List[IngestResult]:
        """Index many sources, extracting and embedding them concurrently.

        Extraction and embedding run in a thread pool because both are dominated
        by I/O -- disk reads and API calls. Writes are serialised by the store,
        so the database stays consistent.

        Args:
            sources: Paths, URLs or strings to index.
            metadata: Metadata attached to every source.
            max_workers: Thread count. Defaults to ``config.max_workers``.
            on_progress: Called as ``(source, done, total)`` after each item.
            ignore_errors: Record failures instead of raising.

        Returns:
            Results in input order.
        """
        items = list(sources)
        if not items:
            return []
        workers = max(1, max_workers or self.config.max_workers)
        results: List[IngestResult] = [IngestResult(source="")] * len(items)
        done = 0

        def work(index: int, item: Source) -> None:
            try:
                results[index] = self.add(item, metadata=metadata)
            except Exception as exc:
                if not ignore_errors:
                    raise
                log.warning("failed to index %s: %s", item, exc)
                results[index] = IngestResult(source=str(item), error=str(exc))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(work, i, item): item for i, item in enumerate(items)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                done += 1
                if on_progress:
                    on_progress(str(futures[future]), done, len(items))

        return results

    def _ingest(
        self,
        source: str,
        text: str,
        metadata: Optional[Dict[str, Any]],
        chunker: Chunker | str | None,
        on_change: str,
    ) -> IngestResult:
        """Shared path for every ``add*`` method."""
        if on_change not in ("replace", "skip", "append"):
            raise ConfigurationError(
                f"on_change must be 'replace', 'skip' or 'append', got {on_change!r}."
            )
        if not text or not text.strip():
            return IngestResult(source=source, error="no text could be extracted")

        content_hash = _digest(text)
        deleted = 0

        if self.store.has_source(source):
            if self.store.has_source(source, content_hash=content_hash):
                log.debug("%s is already indexed and unchanged", source)
                existing = next(
                    (s.chunks for s in self.store.sources() if s.source == source), 0
                )
                return IngestResult(
                    source=source,
                    chunks_skipped=existing,
                    characters=len(text),
                )
            if on_change == "skip":
                return IngestResult(source=source, characters=len(text))
            if on_change == "replace":
                deleted = self.store.delete_source(source)

        split = (
            resolve_chunker(
                chunker,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if chunker is not None
            else self._chunker
        )
        chunks = [c for c in split(text) if c and c.strip()]
        if not chunks:
            return IngestResult(source=source, error="chunking produced no content")

        self.store.upsert_source(
            source,
            content_hash=content_hash,
            characters=len(text),
            metadata=metadata or {},
        )

        start = 0 if on_change != "append" else _next_index(self.store, source)
        added = skipped = 0
        for batch in _batches(chunks, self.config.embed_batch_size):
            vectors = self.embedder.embed_documents(batch)
            batch_added, batch_skipped = self.store.add_chunks(
                source,
                batch,
                vectors,
                metadata=[dict(metadata or {}) for _ in batch],
                start_index=start,
            )
            added += batch_added
            skipped += batch_skipped
            start += len(batch)

        return IngestResult(
            source=source,
            chunks_added=added,
            chunks_skipped=skipped,
            chunks_deleted=deleted,
            characters=len(text),
        )

    # -- retrieval ---------------------------------------------------------- #

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        mode: Optional[SearchMode] = None,
        where: Optional[Where] = None,
        source: Optional[str] = None,
        candidates: Optional[int] = None,
        diversity: Optional[float] = None,
        expand_context: Optional[int] = None,
        rerank: Union[bool, Reranker, None] = None,
    ) -> List[Hit]:
        """Retrieve relevant chunks without calling a chat model.

        This is the honest way to evaluate an index: whatever comes back is
        exactly what a generated answer would have been built from.

        Args:
            query: The search text.
            top_k: How many chunks to return.
            mode: ``"hybrid"`` (default), ``"vector"`` or ``"keyword"``.
            where: Metadata filter, for example ``{"year": {"$gte": 2024}}``.
            source: Restrict to a single source identifier.
            candidates: Candidates each retriever contributes before fusion.
            diversity: MMR diversity in ``[0, 1]``; 0 disables it.
            expand_context: Neighbouring chunks to attach to each hit.
            rerank: ``True`` to use the engine's reranker, a reranker instance to
                use just this once, or ``False`` to skip it.

        Returns:
            Hits ordered best-first.
        """
        config = self.config.retrieval(
            top_k=top_k,
            mode=mode,
            candidates=candidates,
            diversity=diversity,
            expand_context=expand_context,
        )

        reranker = self._resolve_reranker(rerank)
        if reranker is not None:
            # A reranker only helps if it is given more to choose from.
            config = replace(
                config, top_k=max(config.resolved_candidates(), config.top_k)
            )

        query_vector = None
        if config.mode != "keyword":
            query_vector = self.embedder.embed_query(query)

        hits = self.retriever.search(
            query, query_vector, config=config, where=where, source=source
        )

        if reranker is not None and hits:
            final_k = top_k or self.config.top_k
            hits = reranker.rerank(query, hits, top_k=final_k)[:final_k]
        return hits

    def query(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        mode: Optional[SearchMode] = None,
        where: Optional[Where] = None,
        source: Optional[str] = None,
        stream: bool = False,
        prompt: Optional[str] = None,
        rerank: Union[bool, Reranker, None] = None,
        **search_kwargs: Any,
    ) -> Union[Answer, StreamingAnswer]:
        """Answer a question using retrieved context.

        Args:
            question: The question to answer.
            top_k: How many chunks to retrieve as context.
            mode: Search mode. See :meth:`search`.
            where: Metadata filter.
            source: Restrict retrieval to a single source.
            stream: Return a :class:`~softrag.types.StreamingAnswer` that yields
                deltas instead of waiting for the full response.
            prompt: Override the prompt template for this call. Must contain
                ``{context}`` and ``{question}``.
            rerank: See :meth:`search`.
            **search_kwargs: Forwarded to :meth:`search`.

        Returns:
            An :class:`~softrag.types.Answer` -- a ``str`` carrying ``.hits``
            and ``.sources`` -- or a :class:`~softrag.types.StreamingAnswer`.

        Example:
            >>> answer = rag.query("What changed in v2?")     # doctest: +SKIP
            >>> print(answer)                                 # doctest: +SKIP
            >>> print(answer.sources)                         # doctest: +SKIP
        """
        hits = self.search(
            question,
            top_k=top_k,
            mode=mode,
            where=where,
            source=source,
            rerank=rerank,
            **search_kwargs,
        )
        template = prompt or self.config.prompt
        rendered = template.format(context=format_context(hits), question=question)
        model = self.chat_model

        if not stream:
            return Answer(
                model.complete(rendered), hits=hits, question=question, prompt=rendered
            )

        streamer = getattr(model, "stream", None)
        source_stream: Iterator[str]
        if callable(streamer):
            source_stream = streamer(rendered)
        else:
            source_stream = iter([model.complete(rendered)])
        return StreamingAnswer(
            source_stream, hits=hits, question=question, prompt=rendered
        )

    def _resolve_reranker(
        self, rerank: Union[bool, Reranker, None]
    ) -> Optional[Reranker]:
        if rerank is False:
            return None
        if rerank is None or rerank is True:
            return self.reranker
        return rerank

    # -- management --------------------------------------------------------- #

    def sources(self, *, limit: Optional[int] = None) -> List[SourceInfo]:
        """List indexed sources, most recently updated first."""
        return self.store.sources(limit=limit)

    def delete(
        self, source: Optional[str] = None, *, where: Optional[Where] = None
    ) -> int:
        """Remove content from the index.

        Args:
            source: Delete everything belonging to this source.
            where: Delete every chunk matching this metadata filter.

        Returns:
            How many chunks were removed.

        Raises:
            ConfigurationError: If neither or both arguments are given -- an
                unconditional delete should be :meth:`reset`, explicitly.
        """
        if (source is None) == (where is None):
            raise ConfigurationError(
                "Pass exactly one of source=... or where=... . To empty the "
                "whole index, call rag.reset()."
            )
        if source is not None:
            return self.store.delete_source(source)
        return self.store.delete_where(where or {})

    def reset(self) -> None:
        """Delete every indexed document, keeping the file and its schema."""
        self.store.reset()

    def optimize(self) -> None:
        """Compact the indexes and reclaim disk space."""
        self.store.optimize()

    def stats(self) -> Stats:
        """Summarise what this index holds."""
        return self.store.stats()


def format_context(hits: Sequence[Hit]) -> str:
    """Render hits as numbered, attributed context blocks.

    Numbering is what makes ``[1]``-style citations possible, and naming the
    source in the block is what lets a model say *where* something came from.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        label = hit.source or "unknown source"
        blocks.append(f"[{number}] ({label})\n{hit.text}")
    return "\n\n".join(blocks) if blocks else "(no relevant documents were found)"


def connect(
    db_path: str | os.PathLike = "softrag.db",
    *,
    embed_model: Any = None,
    chat_model: Any = None,
    **kwargs: Any,
) -> Rag:
    """Open or create an index, detecting models from the environment.

    The shortest path to a working engine::

        rag = softrag.connect("kb.db")
        rag.add("handbook.pdf")
        print(rag.query("What is the refund policy?"))

    Args:
        db_path: Where the index lives.
        embed_model: Override the auto-detected embedder.
        chat_model: Override the auto-detected chat model.
        **kwargs: Forwarded to :class:`Rag`.
    """
    return Rag(
        db_path=db_path, embed_model=embed_model, chat_model=chat_model, **kwargs
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_url(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith(("http://", "https://"))


def _batches(items: Sequence[str], size: int) -> Iterator[List[str]]:
    size = max(1, size)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _next_index(store: Store, source: str) -> int:
    row = store.db.execute(
        "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM documents WHERE source = ?",
        (source,),
    ).fetchone()
    return int(row[0]) if row else 0
