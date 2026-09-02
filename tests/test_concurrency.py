"""Threaded access to one engine.

softrag hands a single ``sqlite3`` connection to every thread
(``check_same_thread=False``) and serialises the writes behind an ``RLock``, so
these tests are the ones that would catch the two failures that arrangement can
produce: ``sqlite3.ProgrammingError: SQLite objects created in a thread can only
be used in that same thread``, and an interleaved write that loses or duplicates
chunks.

Nothing here sleeps: threads are released together from a
:class:`threading.Barrier` so the overlap is real rather than hoped for.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from conftest import make_rag
from softrag import Rag
from softrag.store import Store

THREADS = 4
PER_THREAD = 6


def batch(prefix: str) -> list[str]:
    """Distinct texts, so every one of them is its own source and chunk."""
    return [f"{prefix} document {i} with body text zqxwv{i}" for i in range(PER_THREAD)]


def run_together(work, count: int = THREADS) -> list[Any]:
    """Run ``work(index)`` on ``count`` threads released at the same instant."""
    barrier = threading.Barrier(count)

    def entry(index: int) -> Any:
        barrier.wait(timeout=30)
        return work(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return [
            future.result(timeout=60)
            for future in [pool.submit(entry, i) for i in range(count)]
        ]


# --------------------------------------------------------------------------- #
# Concurrent ingestion
# --------------------------------------------------------------------------- #


def test_add_many_from_several_threads_indexes_everything_once(rag: Rag):
    results = run_together(lambda i: rag.add_many(batch(f"t{i}")))

    assert all(result.ok for group in results for result in group)
    assert len(rag) == THREADS * PER_THREAD
    assert len(rag.sources()) == THREADS * PER_THREAD

    stored = [row[0] for row in rag.store.db.execute("SELECT text FROM documents")]
    assert len(stored) == len(set(stored)), "a chunk was written twice"


def test_concurrent_adds_of_the_same_content_do_not_duplicate_it(rag: Rag):
    text = "one identical paragraph added from every thread at once"

    run_together(lambda i: rag.add_text(text, name="shared"))

    assert len(rag) == 1
    assert [info.source for info in rag.sources()] == ["shared"]


def test_single_adds_from_several_threads_stay_consistent(rag: Rag):
    def work(index: int) -> None:
        for position, text in enumerate(batch(f"t{index}")):
            assert rag.add_text(text, name=f"t{index}-{position}").ok

    run_together(work)

    assert len(rag) == THREADS * PER_THREAD
    counts = {info.source: info.chunks for info in rag.sources()}
    assert set(counts.values()) == {1}


# --------------------------------------------------------------------------- #
# Reads racing writes
# --------------------------------------------------------------------------- #


def test_searching_while_a_write_is_in_flight_never_raises(rag: Rag):
    rag.add_text("a seed document about refunds and returns", name="seed")
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(30):
                rag.add_text(f"written document {i} about refunds", name=f"w{i}")
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> int:
        seen = 0
        try:
            while not stop.is_set():
                for mode in ("hybrid", "vector", "keyword"):
                    rag.search("refunds returns", mode=mode, top_k=3)
                seen += 1
        except BaseException as exc:
            errors.append(exc)
        return seen

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(writer)] + [pool.submit(reader) for _ in range(3)]
        rounds = [future.result(timeout=60) for future in futures]

    assert not errors, errors
    assert sum(r for r in rounds if isinstance(r, int)) > 0
    assert len(rag) == 31


def test_deleting_while_searching_never_raises(rag: Rag):
    for i in range(20):
        rag.add_text(f"document {i} about refunds and returns", name=f"d{i}")

    errors: list[BaseException] = []
    stop = threading.Event()

    def deleter() -> None:
        try:
            for i in range(20):
                rag.delete(f"d{i}")
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def searcher() -> None:
        try:
            while not stop.is_set():
                rag.search("refunds", top_k=5)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(deleter), pool.submit(searcher), pool.submit(searcher)]
        for future in futures:
            future.result(timeout=60)

    assert not errors, errors
    assert len(rag) == 0


# --------------------------------------------------------------------------- #
# A file-backed index
# --------------------------------------------------------------------------- #


def test_a_file_backed_engine_stays_consistent_across_threads(tmp_path):
    engine = make_rag(str(tmp_path / "kb.db"))
    try:
        run_together(lambda i: engine.add_many(batch(f"t{i}")))

        assert len(engine) == THREADS * PER_THREAD
        found = run_together(lambda i: engine.search("zqxwv3", top_k=THREADS))
        assert all(hits for hits in found)
    finally:
        engine.close()

    # Reopening proves the writes were committed, not merely visible in-process.
    reopened = make_rag(str(tmp_path / "kb.db"))
    try:
        assert len(reopened) == THREADS * PER_THREAD
        assert len(reopened.sources()) == THREADS * PER_THREAD
    finally:
        reopened.close()


def test_the_store_itself_survives_threaded_reads_and_writes(tmp_path):
    store = Store(tmp_path / "kb.db")
    try:

        def work(index: int) -> None:
            source = f"s{index}"
            texts = [f"chunk {index}-{i} body" for i in range(PER_THREAD)]
            store.upsert_source(source, content_hash=source, characters=100)
            store.add_chunks(
                source,
                texts,
                [[float(index + 1), 0.0, 0.0, 0.0] for _ in texts],
                metadata=[{"thread": index} for _ in texts],
            )
            store.search_keyword("body chunk", k=5)
            store.count()

        run_together(work)

        assert store.count() == THREADS * PER_THREAD
        per_source = {info.source: info.chunks for info in store.sources()}
        assert per_source == {f"s{i}": PER_THREAD for i in range(THREADS)}
    finally:
        store.close()


def test_no_thread_affinity_error_leaks_out(rag: Rag):
    """The connection is opened with check_same_thread=False; prove it holds."""
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            rag.add_text(f"thread {index} content body", name=f"t{index}")
            rag.search("content body")
            rag.stats()
            rag.sources()
            len(rag)
        except sqlite3.ProgrammingError as exc:  # pragma: no cover - the bug
            errors.append(exc)

    run_together(work, count=8)

    assert not errors, errors


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_add_many_is_correct_at_any_worker_count(rag: Rag, workers):
    items = [f"document {i} with distinct body wording" for i in range(12)]
    results = rag.add_many(items, max_workers=workers)

    assert [result.characters for result in results] == [len(item) for item in items]
    assert len(rag) == 12
