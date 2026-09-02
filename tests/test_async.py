"""The async engine.

Coroutines are driven with :func:`asyncio.run` from ordinary sync tests, so the
suite needs no ``pytest-asyncio``. The test that matters most is the last one:
:class:`~softrag.AsyncRag` bridges a *blocking* generator onto the event loop
through a queue, and a bridge like that is exactly where an exception in the
worker thread turns into a consumer that waits forever.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from conftest import DIM, RecordingChatModel
from softrag import Answer, AsyncRag, EchoChatModel, HashEmbedder
from softrag.aengine import AsyncStreamingAnswer, connect_async
from softrag.errors import ChatError, ConfigurationError

CORPUS = {
    "handbook": "The refund policy allows returns within thirty days of purchase.",
    "changelog": "Version two introduced hybrid retrieval and the zqxwv7 identifier.",
    "biology": "Mitochondria are the powerhouse of the cell.",
}


def make_async_rag(**kwargs: Any) -> AsyncRag:
    kwargs.setdefault("embed_model", HashEmbedder(dimensions=DIM))
    kwargs.setdefault("chat_model", EchoChatModel())
    kwargs.setdefault("db_path", ":memory:")
    return AsyncRag(**kwargs)


def run(coro_factory, **kwargs: Any):
    """Build an engine, run one coroutine against it, and always close it."""

    async def main():
        rag = make_async_rag(**kwargs)
        try:
            return await coro_factory(rag)
        finally:
            await rag.close()

    return asyncio.run(main())


async def seed(rag: AsyncRag) -> None:
    for name, text in CORPUS.items():
        result = await rag.add_text(text, name=name)
        assert result.ok, result.error


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def test_add_indexes_a_document():
    async def scenario(rag: AsyncRag):
        result = await rag.add("a paragraph about quarterly revenue")
        return result, await rag.count()

    result, count = run(scenario)
    assert result.ok
    assert count == 1


def test_add_file_and_add_text_reach_the_same_store(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("the tax deadline moved to the fifteenth", encoding="utf-8")

    async def scenario(rag: AsyncRag):
        await rag.add_file(path)
        await rag.add_text("an unrelated inline note", name="inline")
        return await rag.count(), [s.source for s in await rag.sources()]

    count, sources = run(scenario)
    assert count == 2
    assert "inline" in sources


def test_add_many_returns_results_in_input_order():
    items = [f"document number {i} with its own distinct wording" for i in range(8)]

    async def scenario(rag: AsyncRag):
        return await rag.add_many(items)

    results = run(scenario)

    assert len(results) == len(items)
    assert all(r.ok for r in results)
    # add_many fans out concurrently, so completion order is not input order.
    # gather() has to put them back.
    assert [r.characters for r in results] == [len(item) for item in items]


def test_add_many_records_failures_instead_of_raising(tmp_path):
    bad = str(tmp_path / "nowhere" / "*.md")

    async def scenario(rag: AsyncRag):
        return await rag.add_many(["a fine document", bad])

    results = run(scenario)
    assert results[0].ok
    assert not results[1].ok


def test_add_many_can_raise_instead(tmp_path):
    bad = str(tmp_path / "nowhere" / "*.md")

    async def scenario(rag: AsyncRag):
        return await rag.add_many(["a fine document", bad], ignore_errors=False)

    with pytest.raises(Exception, match="glob"):
        run(scenario)


def test_add_many_on_nothing_does_nothing():
    assert run(lambda rag: rag.add_many([])) == []


def test_add_directory_walks_the_tree(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("first document body", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("second document body", encoding="utf-8")

    async def scenario(rag: AsyncRag):
        results = await rag.add_directory(tmp_path)
        return results, await rag.count()

    results, count = run(scenario)
    assert len(results) == 2
    assert count == 2


# --------------------------------------------------------------------------- #
# Retrieval and generation
# --------------------------------------------------------------------------- #


def test_search_returns_hits():
    async def scenario(rag: AsyncRag):
        await seed(rag)
        return await rag.search("refund policy", top_k=2)

    hits = run(scenario)
    assert hits
    assert hits[0].source == "handbook"


def test_search_honours_filters_and_source():
    async def scenario(rag: AsyncRag):
        await seed(rag)
        return (
            await rag.search("cell refund", source="biology"),
            await rag.search("anything", mode="keyword", top_k=1),
        )

    restricted, _ = run(scenario)
    assert {hit.source for hit in restricted} == {"biology"}


def test_query_returns_an_answer_with_provenance():
    recorder = RecordingChatModel()

    async def scenario(rag: AsyncRag):
        await seed(rag)
        return await rag.query("how long is the refund window?")

    answer = run(scenario, chat_model=recorder)

    assert isinstance(answer, Answer)
    assert answer == "RECORDED ANSWER"
    assert "handbook" in answer.sources
    assert "thirty days" in recorder.prompt
    assert "how long is the refund window?" in recorder.prompt


def test_a_custom_prompt_template_is_used():
    recorder = RecordingChatModel()

    async def scenario(rag: AsyncRag):
        await seed(rag)
        return await rag.query("what makes ATP?", prompt="C<{context}> Q<{question}>")

    run(scenario, chat_model=recorder)
    assert recorder.prompt.endswith("Q<what makes ATP?>")


def test_streaming_is_iterated_with_async_for():
    async def scenario(rag: AsyncRag):
        await seed(rag)
        streaming = await rag.query("refund policy", stream=True)
        assert isinstance(streaming, AsyncStreamingAnswer)
        assert streaming.sources
        deltas = [delta async for delta in streaming]
        return deltas, streaming.text

    deltas, text = run(scenario)
    assert len(deltas) > 1
    assert text == "".join(deltas)


def test_collect_drains_the_stream_into_an_answer():
    async def scenario(rag: AsyncRag):
        await seed(rag)
        streaming = await rag.query("refund policy", stream=True)
        return await streaming.collect()

    answer = run(scenario)
    assert isinstance(answer, Answer)
    assert "refund" in answer.lower()
    assert answer.hits


def test_a_native_astream_is_preferred():
    class NativeAsync:
        def complete(self, prompt: str) -> str:  # pragma: no cover - not used here
            return "sync"

        async def astream(self, prompt: str):
            yield "native "
            yield "async"

    async def scenario(rag: AsyncRag):
        await rag.add_text("body text", name="doc")
        streaming = await rag.query("body", stream=True)
        return [delta async for delta in streaming]

    # Regression: the adapter does not forward the backend's own coroutines, so
    # looking for astream on the adapter alone leaves the documented fast path
    # dead code.
    assert run(scenario, chat_model=NativeAsync()) == ["native ", "async"]


def test_a_native_acomplete_is_preferred():
    class NativeAsync:
        def complete(self, prompt: str) -> str:  # pragma: no cover - not used here
            return "sync path"

        async def acomplete(self, prompt: str) -> str:
            return "async path"

    async def scenario(rag: AsyncRag):
        await rag.add_text("body text", name="doc")
        return await rag.query("body")

    assert run(scenario, chat_model=NativeAsync()) == "async path"


def test_embedding_runs_in_a_worker_thread_and_frees_the_loop():
    """Embedding has no native async path -- it is offloaded, which is the point.

    ``AsyncRag`` shares its whole ingestion path with the synchronous engine, so
    a blocking embedder is called through :func:`asyncio.to_thread` rather than
    duplicated in a coroutine. What has to be true is the thing that matters in
    an async application: the embedder runs off the event loop, and the loop
    keeps serving other work while it is busy.
    """
    main_thread = threading.get_ident()

    class BlockingEmbedder:
        dimensions = DIM

        def __init__(self) -> None:
            self.inner = HashEmbedder(dimensions=DIM)
            self.threads: set[int] = set()
            self.entered = threading.Event()
            self.release = threading.Event()

        def embed_query(self, text: str) -> list[float]:
            return self.inner.embed_query(text)

        def embed_documents(self, texts) -> list[list[float]]:
            self.threads.add(threading.get_ident())
            self.entered.set()
            self.release.wait(timeout=10)
            return self.inner.embed_documents(texts)

    embedder = BlockingEmbedder()

    async def scenario(rag: AsyncRag):
        ingest = asyncio.ensure_future(rag.add_text("body text to embed", name="doc"))

        # The loop is still running while the embedder blocks: this only
        # completes if add_text() is not holding it hostage.
        await asyncio.wait_for(asyncio.to_thread(embedder.entered.wait, 10), timeout=10)
        ticks = 0
        while ticks < 5:
            await asyncio.sleep(0)
            ticks += 1
        embedder.release.set()

        return await asyncio.wait_for(ingest, timeout=10), ticks

    result, ticks = run(scenario, embed_model=embedder)

    assert result.ok
    assert ticks == 5, "the event loop was blocked while the embedder ran"
    assert embedder.threads and main_thread not in embedder.threads


# --------------------------------------------------------------------------- #
# The bridge: a failing sync stream must not hang the consumer
# --------------------------------------------------------------------------- #


class BreakingStream:
    """Streams a little, then fails -- the shape that deadlocks a naive bridge."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - not used here
        return "unused"

    def stream(self, prompt: str) -> Iterator[str]:
        yield "the answer begins"
        raise RuntimeError("the upstream connection dropped")


def test_an_exception_in_a_bridged_stream_reaches_the_consumer():
    async def scenario(rag: AsyncRag):
        await rag.add_text("body text", name="doc")
        streaming = await rag.query("body", stream=True)
        received: list[str] = []
        with pytest.raises(ChatError, match="upstream connection dropped"):
            async for delta in streaming:
                received.append(delta)
        return received

    # A timeout, not a bare run(): if the bridge swallowed the exception the
    # consumer would wait on an empty queue forever, and a hung test is a much
    # worse failure report than a failed assertion.
    async def guarded():
        rag = make_async_rag(chat_model=BreakingStream())
        try:
            return await asyncio.wait_for(scenario(rag), timeout=10)
        finally:
            await rag.close()

    assert asyncio.run(guarded()) == ["the answer begins"]


def test_a_stream_that_fails_immediately_still_raises():
    class FailsAtOnce:
        def complete(self, prompt: str) -> str:  # pragma: no cover - not used here
            return "unused"

        def stream(self, prompt: str) -> Iterator[str]:
            raise RuntimeError("nothing to give")
            yield ""  # pragma: no cover - unreachable, keeps this a generator

    async def guarded():
        rag = make_async_rag(chat_model=FailsAtOnce())
        try:
            streaming = await rag.query("body", stream=True)
            with pytest.raises(ChatError, match="nothing to give"):
                await asyncio.wait_for(streaming.collect(), timeout=10)
        finally:
            await rag.close()

    asyncio.run(guarded())


# --------------------------------------------------------------------------- #
# Lifecycle and management
# --------------------------------------------------------------------------- #


def test_the_async_context_manager_closes_the_store():
    async def scenario():
        async with make_async_rag() as rag:
            await rag.add_text("indexed inside the context manager", name="doc")
            assert await rag.count() == 1
            return rag
        # unreachable

    rag = asyncio.run(scenario())
    assert rag.store._closed


def test_count_stats_and_sources():
    async def scenario(rag: AsyncRag):
        await seed(rag)
        return await rag.count(), await rag.stats(), await rag.sources()

    count, stats, sources = run(scenario)
    assert count == 3
    assert stats.sources == 3
    assert stats.dimensions == DIM
    assert {info.source for info in sources} == set(CORPUS)


def test_sources_can_be_limited():
    async def scenario(rag: AsyncRag):
        await seed(rag)
        return await rag.sources(limit=1)

    assert len(run(scenario)) == 1


def test_delete_by_source_and_by_filter():
    async def scenario(rag: AsyncRag):
        await rag.add_text("first body", name="a", metadata={"keep": False})
        await rag.add_text("second body", name="b", metadata={"keep": True})
        removed_source = await rag.delete("a")
        removed_filter = await rag.delete(where={"keep": True})
        return removed_source, removed_filter, await rag.count()

    removed_source, removed_filter, count = run(scenario)
    assert removed_source == 1
    assert removed_filter == 1
    assert count == 0


def test_delete_still_refuses_an_ambiguous_call():
    async def scenario(rag: AsyncRag):
        return await rag.delete()

    with pytest.raises(ConfigurationError):
        run(scenario)


def test_reset_and_optimize(tmp_path):
    async def scenario(rag: AsyncRag):
        await seed(rag)
        await rag.reset()
        await rag.optimize()
        return await rag.count()

    assert run(scenario, db_path=str(tmp_path / "kb.db")) == 0


def test_the_sync_engine_and_config_are_reachable():
    async def scenario(rag: AsyncRag):
        return rag.sync, rag.config, rag.store

    engine, config, store = run(scenario, top_k=9)
    assert config.top_k == 9
    assert engine.store is store


def test_connect_async_builds_an_engine(tmp_path):
    async def scenario():
        rag = connect_async(
            tmp_path / "kb.db",
            embed_model=HashEmbedder(dimensions=DIM),
            chat_model=EchoChatModel(),
        )
        try:
            assert isinstance(rag, AsyncRag)
            await rag.add_text("body", name="doc")
            return await rag.count()
        finally:
            await rag.close()

    assert asyncio.run(scenario()) == 1


def test_max_concurrency_is_respected():
    async def scenario(rag: AsyncRag):
        return await rag.add_many([f"document {i} body text" for i in range(6)])

    results = run(scenario, max_concurrency=1)
    assert all(r.ok for r in results)
