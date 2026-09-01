"""Async API.

:class:`AsyncRag` mirrors :class:`~softrag.engine.Rag` method for method, so
moving between them costs nothing but an ``await``.

What it actually does, stated plainly: SQLite has no async driver, and neither
do most embedding SDKs, so the blocking work runs in a worker thread via
:func:`asyncio.to_thread`. That is not a marketing "async" -- it is the thing
that matters in an async application, because the event loop stays free to
serve other requests while an embedding call or a disk read is in flight.
Where a backend *does* offer native coroutines (``aembed_documents``,
``acomplete``, ``astream``), they are used directly instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

from .chunking import Chunker
from .engine import Rag, RagConfig, format_context
from .retrieval import SearchMode
from .types import (
    Answer,
    Hit,
    IngestResult,
    Reranker,
    SourceInfo,
    Stats,
    Where,
)

log = logging.getLogger("softrag.async")

__all__ = ["AsyncRag", "AsyncStreamingAnswer", "connect_async"]


class AsyncStreamingAnswer:
    """An answer streamed as it is generated.

    Iterate with ``async for`` to receive text deltas. The retrieved
    :attr:`hits` are available immediately, since retrieval finishes before
    generation starts.
    """

    __slots__ = ("_stream", "hits", "question", "prompt", "_parts", "_done")

    def __init__(
        self,
        stream: AsyncIterator[str],
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

    def __aiter__(self) -> "AsyncStreamingAnswer":
        return self

    async def __anext__(self) -> str:
        if self._done:
            raise StopAsyncIteration
        try:
            delta = await self._stream.__anext__()
        except StopAsyncIteration:
            self._done = True
            raise
        self._parts.append(delta)
        return delta

    @property
    def text(self) -> str:
        """Text produced so far."""
        return "".join(self._parts)

    @property
    def sources(self) -> List[str]:
        seen: Dict[str, None] = {}
        for hit in self.hits:
            if hit.source:
                seen.setdefault(hit.source, None)
        return list(seen)

    async def collect(self) -> Answer:
        """Drain the stream and return a complete :class:`Answer`."""
        async for _ in self:
            pass
        return Answer(
            self.text, hits=self.hits, question=self.question, prompt=self.prompt
        )


class AsyncRag:
    """The async twin of :class:`~softrag.engine.Rag`.

    Args:
        embed_model: Any embedder. Native ``aembed_documents`` is used when the
            object offers it.
        chat_model: Any chat backend, or ``None`` for retrieval only.
        db_path: Where the index lives.
        chunker: Chunking strategy.
        config: Tuning knobs. See :class:`~softrag.engine.RagConfig`.
        reranker: Optional second-stage reranker.
        auto: Fill in missing models from the environment.
        max_concurrency: Ceiling on simultaneous background operations, which
            keeps a bulk ingest from opening hundreds of connections to an
            embedding API at once.

    Example:
        >>> async def main():                                  # doctest: +SKIP
        ...     async with AsyncRag(db_path="kb.db") as rag:
        ...         await rag.add("handbook.pdf")
        ...         answer = await rag.query("What is the refund policy?")
        ...         print(answer, answer.sources)
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
        max_concurrency: int = 8,
        **overrides: Any,
    ) -> None:
        self._rag = Rag(
            embed_model=embed_model,
            chat_model=chat_model,
            db_path=db_path,
            chunker=chunker,
            config=config,
            reranker=reranker,
            auto=auto,
            **overrides,
        )
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def sync(self) -> Rag:
        """The underlying synchronous engine, for anything not mirrored here."""
        return self._rag

    @property
    def config(self) -> RagConfig:
        return self._rag.config

    @property
    def store(self) -> Any:
        return self._rag.store

    async def close(self) -> None:
        """Close the underlying database."""
        await asyncio.to_thread(self._rag.close)

    async def __aenter__(self) -> "AsyncRag":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<AsyncRag {self._rag!r}>"

    async def _run(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking engine call off the event loop, under the semaphore."""
        async with self._semaphore:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # -- ingestion ---------------------------------------------------------- #

    async def add(self, source: Any, **kwargs: Any) -> IngestResult:
        """Index anything. See :meth:`softrag.Rag.add`."""
        return await self._run(self._rag.add, source, **kwargs)

    async def add_text(self, text: str, **kwargs: Any) -> IngestResult:
        """Index a string. See :meth:`softrag.Rag.add_text`."""
        return await self._run(self._rag.add_text, text, **kwargs)

    async def add_file(self, path: Any, **kwargs: Any) -> IngestResult:
        """Index a file. See :meth:`softrag.Rag.add_file`."""
        return await self._run(self._rag.add_file, path, **kwargs)

    async def add_web(self, url: str, **kwargs: Any) -> IngestResult:
        """Fetch and index a URL. See :meth:`softrag.Rag.add_web`."""
        return await self._run(self._rag.add_web, url, **kwargs)

    async def add_image(self, path: Any, **kwargs: Any) -> IngestResult:
        """Caption and index an image. See :meth:`softrag.Rag.add_image`."""
        return await self._run(self._rag.add_image, path, **kwargs)

    async def add_many(
        self,
        sources: Iterable[Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        ignore_errors: bool = True,
        **kwargs: Any,
    ) -> List[IngestResult]:
        """Index many sources concurrently.

        Concurrency is bounded by ``max_concurrency`` so a large ingest does not
        overwhelm an embedding API. Results come back in input order.

        Args:
            sources: Paths, URLs or strings to index.
            metadata: Metadata attached to every source.
            ignore_errors: Record failures instead of raising.
            **kwargs: Forwarded to :meth:`add`.
        """
        items = list(sources)
        if not items:
            return []

        async def one(item: Any) -> IngestResult:
            try:
                return await self.add(item, metadata=metadata, **kwargs)
            except Exception as exc:
                if not ignore_errors:
                    raise
                log.warning("failed to index %s: %s", item, exc)
                return IngestResult(source=str(item), error=str(exc))

        return list(await asyncio.gather(*(one(item) for item in items)))

    async def add_directory(
        self, directory: str | os.PathLike, **kwargs: Any
    ) -> List[IngestResult]:
        """Index every supported file under a directory.

        Discovery is cheap and synchronous; the ingestion itself fans out
        through :meth:`add_many`.
        """
        from .ingest import discover_files
        from pathlib import Path

        base = Path(directory)
        files = await asyncio.to_thread(
            discover_files,
            base,
            pattern=kwargs.pop("pattern", "**/*"),
            exclude=kwargs.pop("exclude", ()),
            recursive=kwargs.pop("recursive", True),
        )
        return await self.add_many(files, **kwargs)

    # -- retrieval ---------------------------------------------------------- #

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        mode: Optional[SearchMode] = None,
        where: Optional[Where] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Hit]:
        """Retrieve relevant chunks. See :meth:`softrag.Rag.search`."""
        return await self._run(
            self._rag.search,
            query,
            top_k=top_k,
            mode=mode,
            where=where,
            source=source,
            **kwargs,
        )

    async def query(
        self,
        question: str,
        *,
        stream: bool = False,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Answer, AsyncStreamingAnswer]:
        """Answer a question using retrieved context.

        Args:
            question: The question to answer.
            stream: Return an :class:`AsyncStreamingAnswer` to iterate with
                ``async for`` instead of waiting for the full response.
            prompt: Override the prompt template for this call.
            **kwargs: Forwarded to :meth:`search`.

        Returns:
            An :class:`~softrag.types.Answer`, or an
            :class:`AsyncStreamingAnswer` when ``stream=True``.
        """
        hits = await self.search(question, **kwargs)
        template = prompt or self._rag.config.prompt
        rendered = template.format(context=format_context(hits), question=question)
        model = self._rag.chat_model

        if not stream:
            text = await self._complete(model, rendered)
            return Answer(text, hits=hits, question=question, prompt=rendered)

        return AsyncStreamingAnswer(
            self._stream(model, rendered),
            hits=hits,
            question=question,
            prompt=rendered,
        )

    async def _complete(self, model: Any, prompt: str) -> str:
        native = getattr(model, "acomplete", None)
        if callable(native):
            return str(await native(prompt))
        return str(await asyncio.to_thread(model.complete, prompt))

    async def _stream(self, model: Any, prompt: str) -> AsyncIterator[str]:
        """Yield deltas, preferring a native async stream when one exists."""
        native = getattr(model, "astream", None)
        if callable(native):
            async for delta in native(prompt):
                yield delta
            return

        sync_stream = getattr(model, "stream", None)
        if not callable(sync_stream):
            yield await self._complete(model, prompt)
            return

        # Bridge a blocking generator to the event loop through an unbounded
        # queue fed by call_soon_threadsafe. Unbounded matters: with a maxsize,
        # a consumer that stops iterating early would leave the worker thread
        # blocked on a full queue forever.
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def pump() -> None:
            try:
                for delta in sync_stream(prompt):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as exc:  # surfaced to the consumer below
                loop.call_soon_threadsafe(queue.put_nowait, _Error(exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        task = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, _Error):
                    raise item.exception
                yield item
        finally:
            # The producer always reaches its finally, so this cannot hang.
            await task

    # -- management --------------------------------------------------------- #

    async def sources(self, *, limit: Optional[int] = None) -> List[SourceInfo]:
        """List indexed sources."""
        return await self._run(self._rag.sources, limit=limit)

    async def delete(
        self, source: Optional[str] = None, *, where: Optional[Where] = None
    ) -> int:
        """Remove content from the index. See :meth:`softrag.Rag.delete`."""
        return await self._run(self._rag.delete, source, where=where)

    async def reset(self) -> None:
        """Delete every indexed document."""
        await self._run(self._rag.reset)

    async def optimize(self) -> None:
        """Compact the indexes and reclaim disk space."""
        await self._run(self._rag.optimize)

    async def stats(self) -> Stats:
        """Summarise what this index holds."""
        return await self._run(self._rag.stats)

    async def count(self) -> int:
        """Number of indexed chunks."""
        return await self._run(self._rag.store.count)


#: Sentinel marking the end of a bridged stream.
_DONE = object()


class _Error:
    """Carries an exception across the stream bridge."""

    __slots__ = ("exception",)

    def __init__(self, exception: BaseException) -> None:
        self.exception = exception


def connect_async(db_path: str | os.PathLike = "softrag.db", **kwargs: Any) -> AsyncRag:
    """Open or create an index for async use, detecting models from the environment.

    Args:
        db_path: Where the index lives.
        **kwargs: Forwarded to :class:`AsyncRag`.
    """
    return AsyncRag(db_path=db_path, **kwargs)
