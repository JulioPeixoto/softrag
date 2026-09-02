"""The engine: input dispatch, re-ingestion policy, generation and management.

These are the behaviours a user actually depends on -- that ``add`` guesses the
right thing, that re-adding a document twice is free, that changing it replaces
the old version rather than doubling it, and that an ``Answer`` still behaves
like the string everybody prints.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from conftest import DIM, FakeEmbedder, OrderingReranker, RecordingChatModel
from softrag import Answer, EchoChatModel, HashEmbedder, Rag, RagConfig, StreamingAnswer
from softrag.engine import DEFAULT_PROMPT, connect, format_context
from softrag.errors import ConfigurationError, IngestionError

# --------------------------------------------------------------------------- #
# add() dispatch
# --------------------------------------------------------------------------- #


def test_add_treats_a_bare_string_as_text(rag: Rag):
    result = rag.add("a paragraph of prose that is not a path and not a url")
    assert result.ok
    assert result.chunks_added == 1
    assert result.source.startswith("text:")


def test_add_reads_an_existing_path(rag: Rag, tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("the tax deadline moved to the fifteenth", encoding="utf-8")

    result = rag.add(str(path))

    assert result.ok
    assert result.source == str(path)
    assert rag.search("tax deadline")


def test_add_routes_http_urls_to_add_web(rag: Rag, monkeypatch):
    seen: list[str] = []

    def fake_extract_web(url: str, *, timeout: float = 30.0):
        seen.append(url)
        return "fetched page body about quarterly revenue", {"title": "Q2"}

    monkeypatch.setattr("softrag.ingest.extract_web", fake_extract_web)

    result = rag.add("https://example.com/changelog")

    assert seen == ["https://example.com/changelog"]
    assert result.source == "https://example.com/changelog"
    assert result.ok


def test_add_on_a_directory_points_at_add_directory(rag: Rag, tmp_path):
    with pytest.raises(IngestionError) as excinfo:
        rag.add(str(tmp_path))
    assert "add_directory" in str(excinfo.value)


def test_add_on_a_glob_points_at_add_directory(rag: Rag, tmp_path):
    with pytest.raises(IngestionError) as excinfo:
        rag.add(str(tmp_path / "*.md"))
    message = str(excinfo.value)
    assert "glob" in message
    assert "add_directory" in message


def test_add_accepts_bytes_when_named(rag: Rag):
    result = rag.add(b"# Title\n\nSome markdown body text.", name="inline.md")
    assert result.ok
    assert result.source == "inline.md"


# --------------------------------------------------------------------------- #
# add_text
# --------------------------------------------------------------------------- #


def test_add_text_uses_an_explicit_name(rag: Rag):
    result = rag.add_text("the refund window is thirty days", name="handbook")
    assert result.source == "handbook"
    assert [s.source for s in rag.sources()] == ["handbook"]


def test_add_text_derives_a_content_addressed_name(rag: Rag):
    first = rag.add_text("identical content")
    rag.reset()
    second = rag.add_text("identical content")
    assert first.source == second.source
    assert first.source.startswith("text:")


def test_add_text_rejects_blank_content(rag: Rag):
    result = rag.add_text("   \n  ")
    assert not result.ok
    assert result.error == "empty content"


# --------------------------------------------------------------------------- #
# Re-ingestion policy
# --------------------------------------------------------------------------- #


def test_readding_identical_content_is_a_no_op(rag: Rag):
    text = "Version two introduced hybrid retrieval and rank fusion."
    first = rag.add_text(text, name="changelog")
    assert first.chunks_added > 0

    again = rag.add_text(text, name="changelog")

    assert again.ok
    assert again.chunks_added == 0
    assert again.chunks_skipped > 0
    assert again.chunks_deleted == 0
    assert len(rag) == first.chunks_added


def test_changed_content_replaces_the_old_version(rag: Rag):
    rag.add_text("the refund window is thirty days", name="handbook")
    assert rag.search("thirty", mode="keyword")

    changed = rag.add_text("the refund window is ninety days", name="handbook")

    assert changed.chunks_deleted > 0
    assert changed.chunks_added > 0
    assert not rag.search("thirty", mode="keyword")
    assert rag.search("ninety", mode="keyword")
    assert len(rag) == changed.chunks_added


def test_on_change_skip_leaves_the_old_version_alone(rag: Rag):
    rag.add_text("the refund window is thirty days", name="handbook")

    result = rag.add_text(
        "the refund window is ninety days", name="handbook", on_change="skip"
    )

    assert result.ok
    assert result.chunks_added == 0
    assert result.chunks_deleted == 0
    assert rag.search("thirty", mode="keyword")
    assert not rag.search("ninety", mode="keyword")


def test_on_change_append_keeps_both_versions(rag: Rag):
    first = rag.add_text("the refund window is thirty days", name="handbook")

    second = rag.add_text(
        "the refund window is ninety days", name="handbook", on_change="append"
    )

    assert second.chunks_deleted == 0
    assert len(rag) == first.chunks_added + second.chunks_added
    assert rag.search("thirty", mode="keyword")
    assert rag.search("ninety", mode="keyword")


def test_appended_chunks_get_fresh_indices(rag: Rag):
    rag.add_text("alpha content", name="doc")
    rag.add_text("beta content", name="doc", on_change="append")
    indices = [
        row[0]
        for row in rag.store.db.execute(
            "SELECT chunk_index FROM documents WHERE source='doc' ORDER BY chunk_index"
        )
    ]
    assert indices == sorted(set(indices))
    assert len(indices) == 2


def test_the_next_chunk_index_continues_where_a_source_left_off(rag: Rag):
    assert rag.store.next_chunk_index("doc") == 0
    rag.add_text("alpha content", name="doc")
    assert rag.store.next_chunk_index("doc") == 1
    rag.add_text("beta content", name="doc", on_change="append")
    assert rag.store.next_chunk_index("doc") == 2
    assert rag.store.next_chunk_index("never-indexed") == 0


def test_unknown_on_change_is_a_configuration_error(rag: Rag):
    with pytest.raises(ConfigurationError):
        rag.add_text("content", name="doc", on_change="merge")


# --------------------------------------------------------------------------- #
# add_many / add_directory
# --------------------------------------------------------------------------- #


def test_add_many_returns_results_in_input_order(rag: Rag):
    sources = [f"document number {i} with distinct words" for i in range(6)]
    results = rag.add_many(sources)
    assert len(results) == len(sources)
    assert all(r.ok for r in results)
    assert len(rag) == 6


def test_add_many_records_a_failure_when_errors_are_ignored(rag: Rag, tmp_path):
    missing = tmp_path / "nested" / "*.md"
    results = rag.add_many(["a perfectly fine document", str(missing)])

    assert results[0].ok
    assert not results[1].ok
    assert results[1].error


def test_add_many_raises_when_errors_are_not_ignored(rag: Rag, tmp_path):
    missing = tmp_path / "nested" / "*.md"
    with pytest.raises(IngestionError):
        rag.add_many(["a perfectly fine document", str(missing)], ignore_errors=False)


def test_add_many_on_an_empty_iterable_does_nothing(rag: Rag):
    assert rag.add_many([]) == []


def test_add_many_reports_progress(rag: Rag):
    seen: list[tuple[str, int, int]] = []
    rag.add_many(
        ["first document text", "second document text"],
        on_progress=lambda *a: seen.append(a),
    )
    assert len(seen) == 2
    assert [done for _, done, _ in seen] == [1, 2]
    assert {total for _, _, total in seen} == {2}


def test_add_directory_honours_excludes(rag: Rag, tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "drop").mkdir()
    (tmp_path / "keep" / "a.md").write_text("kept document body", encoding="utf-8")
    (tmp_path / "drop" / "b.md").write_text("excluded document body", encoding="utf-8")

    results = rag.add_directory(tmp_path, exclude=("**/drop/**",))

    indexed = {r.source for r in results}
    assert any("a.md" in s for s in indexed)
    assert not any("b.md" in s for s in indexed)


def test_add_directory_rejects_a_file(rag: Rag, tmp_path):
    path = tmp_path / "single.txt"
    path.write_text("body", encoding="utf-8")
    with pytest.raises(IngestionError):
        rag.add_directory(path)


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


def test_answer_is_a_string_carrying_provenance(corpus: Rag):
    answer = corpus.query("what is the refund policy?")

    assert isinstance(answer, Answer)
    assert isinstance(answer, str)
    assert answer.hits
    assert "handbook" in answer.sources
    assert answer.context == "\n\n".join(hit.text for hit in answer.hits)
    assert answer.question == "what is the refund policy?"


def test_answer_sources_are_unique_and_ordered(corpus: Rag):
    corpus.add_text("a second refund paragraph", name="handbook", on_change="append")
    answer = corpus.query("refund", top_k=5)
    assert len(answer.sources) == len(set(answer.sources))


def test_streaming_query_yields_deltas_and_collects(corpus: Rag):
    streaming = corpus.query("what is the refund policy?", stream=True)

    assert isinstance(streaming, StreamingAnswer)
    deltas = list(streaming)
    assert len(deltas) > 1

    collected = streaming.collect()
    assert isinstance(collected, Answer)
    assert collected == "".join(deltas)
    assert collected.hits == streaming.hits


def test_streaming_falls_back_to_a_single_chunk_without_stream(corpus: Rag):
    class NoStream:
        def complete(self, prompt: str) -> str:
            return "one shot answer"

    corpus._chat = NoStream()
    streaming = corpus.query("refund", stream=True)
    assert list(streaming) == ["one shot answer"]


def test_the_prompt_contains_the_context_and_the_question(
    recording_rag: Rag, recorder: RecordingChatModel
):
    recording_rag.add_text(
        "The refund policy allows returns within thirty days.", name="handbook"
    )

    answer = recording_rag.query("how long is the refund window?")

    assert answer == "RECORDED ANSWER"
    prompt = recorder.prompt
    assert "how long is the refund window?" in prompt
    assert "thirty days" in prompt
    assert "(handbook)" in prompt


def test_a_custom_prompt_template_is_used(
    recording_rag: Rag, recorder: RecordingChatModel
):
    recording_rag.add_text("mitochondria produce ATP", name="biology")

    recording_rag.query("what makes ATP?", prompt="CTX<{context}>END Q<{question}>")

    assert recorder.prompt.startswith("CTX<")
    assert recorder.prompt.endswith("Q<what makes ATP?>")
    assert "mitochondria" in recorder.prompt


def test_format_context_numbers_blocks_and_names_sources(corpus: Rag):
    hits = corpus.search("refund", top_k=2)
    rendered = format_context(hits)
    assert rendered.startswith("[1] (")
    for hit in hits:
        assert hit.source in rendered


def test_format_context_says_so_when_nothing_was_found():
    assert "no relevant documents" in format_context([])


def test_the_default_prompt_has_both_fields():
    assert "{context}" in DEFAULT_PROMPT
    assert "{question}" in DEFAULT_PROMPT


# --------------------------------------------------------------------------- #
# Deletion and management
# --------------------------------------------------------------------------- #


def test_delete_with_neither_argument_is_a_configuration_error(corpus: Rag):
    with pytest.raises(ConfigurationError) as excinfo:
        corpus.delete()
    assert "reset" in str(excinfo.value)


def test_delete_with_both_arguments_is_a_configuration_error(corpus: Rag):
    with pytest.raises(ConfigurationError):
        corpus.delete("handbook", where={"kind": "policy"})


def test_delete_by_source(corpus: Rag):
    before = len(corpus)
    removed = corpus.delete("handbook")
    assert removed > 0
    assert len(corpus) == before - removed
    assert "handbook" not in {s.source for s in corpus.sources()}


def test_delete_by_filter(corpus: Rag):
    removed = corpus.delete(where={"kind": "science"})
    assert removed == 1
    assert "biology" not in {s.source for s in corpus.sources()}
    assert "handbook" in {s.source for s in corpus.sources()}


def test_reset_empties_the_index_but_keeps_the_schema(corpus: Rag):
    corpus.reset()
    assert len(corpus) == 0
    assert corpus.sources() == []
    assert corpus.add_text("fresh content", name="new").ok


def test_optimize_runs_and_keeps_the_index_usable(tmp_rag: Rag):
    tmp_rag.add_text("some indexed body text", name="doc")
    tmp_rag.optimize()
    assert tmp_rag.search("indexed")


def test_stats_summarise_the_index(corpus: Rag):
    stats = corpus.stats()
    assert stats.documents == len(corpus)
    assert stats.sources == 4
    assert stats.dimensions == DIM
    assert stats.schema_version == 1


def test_sources_can_be_limited(corpus: Rag):
    assert len(corpus.sources(limit=2)) == 2


def test_len_and_repr(corpus: Rag):
    assert len(corpus) == 4
    text = repr(corpus)
    assert "Rag" in text
    assert "chunks=4" in text


def test_the_engine_is_a_context_manager(tmp_path):
    with Rag(
        db_path=str(tmp_path / "kb.db"),
        embed_model=HashEmbedder(dimensions=DIM),
        chat_model=EchoChatModel(),
    ) as engine:
        engine.add_text("indexed inside the context manager", name="doc")
        assert len(engine) == 1
    assert engine.store._closed


def test_close_is_idempotent(tmp_path):
    engine = Rag(
        db_path=str(tmp_path / "kb.db"),
        embed_model=HashEmbedder(dimensions=DIM),
        chat_model=EchoChatModel(),
    )
    engine.close()
    engine.close()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_kwargs_override_the_config(make_engine):
    engine = make_engine(top_k=11, diversity=0.5, chunk_size=1234)
    assert engine.config.top_k == 11
    assert engine.config.diversity == 0.5
    assert engine.config.chunk_size == 1234


def test_an_unknown_option_is_a_configuration_error(make_engine):
    with pytest.raises(ConfigurationError) as excinfo:
        make_engine(top_kk=5)
    message = str(excinfo.value)
    assert "top_kk" in message
    assert "top_k" in message  # the message lists the valid options


def test_a_config_object_is_used_as_given(make_engine):
    engine = make_engine(config=RagConfig(top_k=2, mode="keyword"))
    assert engine.config.top_k == 2
    hits = engine.search("anything at all")
    assert hits == []


def test_retrieval_config_applies_only_non_none_overrides():
    config = RagConfig(top_k=7, diversity=0.4)
    resolved = config.retrieval(top_k=None, diversity=0.9)
    assert resolved.top_k == 7
    assert resolved.diversity == 0.9


def test_connect_is_a_thin_wrapper(tmp_path):
    engine = connect(
        tmp_path / "kb.db",
        embed_model=HashEmbedder(dimensions=DIM),
        chat_model=EchoChatModel(),
        top_k=3,
    )
    try:
        assert isinstance(engine, Rag)
        assert engine.config.top_k == 3
    finally:
        engine.close()


def test_a_retrieval_only_engine_searches_but_cannot_generate(make_engine):
    engine = make_engine(chat_model=None, auto=False)
    engine.add_text("mitochondria produce ATP", name="biology")

    assert engine.search("ATP")

    with pytest.raises(ConfigurationError) as excinfo:
        _ = engine.chat_model
    assert "rag.search" in str(excinfo.value)


def test_auto_false_without_an_embedder_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError):
        Rag(db_path=str(tmp_path / "kb.db"), auto=False)


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #


def _reranked_engine(make_engine, order: Sequence[str]):
    reranker = OrderingReranker(order)
    engine = make_engine(reranker=reranker, top_k=3)
    for source in ("handbook", "changelog", "biology", "cooking"):
        engine.add_text(
            f"a document about {source} and shared retrieval words", name=source
        )
    return engine, reranker


def test_a_custom_reranker_is_invoked_and_its_order_respected(make_engine):
    engine, reranker = _reranked_engine(make_engine, ["cooking", "biology", "changelog"])

    hits = engine.search("shared retrieval words")

    assert reranker.calls, "the reranker was never called"
    assert [hit.source for hit in hits][:2] == ["cooking", "biology"]
    assert len(hits) <= 3


def test_the_reranker_is_given_more_candidates_than_top_k(make_engine):
    engine, reranker = _reranked_engine(make_engine, ["cooking"])
    engine.search("shared retrieval words", top_k=1)
    _, candidate_count, final_k = reranker.calls[-1]
    assert final_k == 1
    assert candidate_count > 1


def test_rerank_false_skips_the_engines_reranker(make_engine):
    engine, reranker = _reranked_engine(make_engine, ["cooking", "biology"])

    hits = engine.search("shared retrieval words", rerank=False)

    assert reranker.calls == []
    assert hits


def test_a_per_call_reranker_overrides_the_engines(make_engine):
    engine, engine_reranker = _reranked_engine(make_engine, ["cooking"])
    once = OrderingReranker(["biology"])

    hits = engine.search("shared retrieval words", rerank=once)

    assert engine_reranker.calls == []
    assert once.calls
    assert hits[0].source == "biology"


# --------------------------------------------------------------------------- #
# Embedder fingerprinting
# --------------------------------------------------------------------------- #


def test_embedder_changed_is_none_on_a_fresh_index(make_engine):
    engine = make_engine()
    assert engine.embedder_changed is None


def test_reopening_with_a_different_model_of_the_same_width_is_flagged(tmp_path):
    path = str(tmp_path / "kb.db")
    first = Rag(
        db_path=path,
        embed_model=HashEmbedder(dimensions=DIM),
        chat_model=EchoChatModel(),
    )
    first.add_text("indexed with the hash embedder", name="doc")
    assert first.embedder_changed is None
    first.close()

    # Same width, different model: the width check cannot catch this, which is
    # exactly why the fingerprint exists.
    second = Rag(
        db_path=path,
        embed_model=FakeEmbedder(dimensions=DIM),
        chat_model=EchoChatModel(),
    )
    try:
        assert second.embedder_changed == f"HashEmbedder:{DIM}"
    finally:
        second.close()


def test_reopening_with_the_same_model_is_not_flagged(tmp_path):
    path = str(tmp_path / "kb.db")
    first = Rag(
        db_path=path, embed_model=HashEmbedder(dimensions=DIM), chat_model=EchoChatModel()
    )
    first.add_text("indexed with the hash embedder", name="doc")
    first.close()

    second = Rag(
        db_path=path, embed_model=HashEmbedder(dimensions=DIM), chat_model=EchoChatModel()
    )
    try:
        assert second.embedder_changed is None
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# Chunking hand-off
# --------------------------------------------------------------------------- #


def test_a_per_call_chunker_only_affects_that_document(make_engine):
    engine = make_engine(chunk_size=10_000)
    engine.add_text("alpha|beta|gamma", name="split", chunker="|")
    engine.add_text("delta|epsilon", name="whole")

    counts = {info.source: info.chunks for info in engine.sources()}
    assert counts["split"] == 3
    assert counts["whole"] == 1


def test_a_chunker_producing_nothing_is_reported(make_engine):
    def empty_chunker(text: str) -> list[str]:
        return ["   "]

    engine = make_engine(chunker=empty_chunker)
    result = engine.add_text("real content that the chunker throws away")
    assert not result.ok
    assert "chunking" in (result.error or "")


def test_documents_are_embedded_in_batches(make_engine):
    class Counting:
        dimensions = DIM

        def __init__(self) -> None:
            self.inner = HashEmbedder(dimensions=DIM)
            self.batch_sizes: list[int] = []

        def embed_query(self, text: str) -> list[float]:
            return self.inner.embed_query(text)

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            self.batch_sizes.append(len(texts))
            return self.inner.embed_documents(texts)

    embedder = Counting()
    engine = make_engine(
        embed_model=embedder, embed_batch_size=2, chunk_size=40, chunk_overlap=0
    )
    engine.add_text(" ".join(f"paragraph{i} filler words here" for i in range(8)))

    assert embedder.batch_sizes
    assert max(embedder.batch_sizes) <= 2


# --------------------------------------------------------------------------- #
# Streaming plumbing
# --------------------------------------------------------------------------- #


def test_streaming_answer_replays_after_it_is_drained(corpus: Rag):
    streaming = corpus.query("refund", stream=True)
    first = "".join(streaming)
    second = "".join(streaming)
    assert first == second
    assert streaming.text == first


def test_streaming_answer_exposes_sources_before_iteration(corpus: Rag):
    streaming = corpus.query("refund policy", stream=True)
    assert streaming.sources
    assert streaming.text == ""


class _SpyChat:
    """Records whether the engine asked for a stream or a completion."""

    def __init__(self) -> None:
        self.mode: list[str] = []

    def complete(self, prompt: str) -> str:
        self.mode.append("complete")
        return "done"

    def stream(self, prompt: str) -> Iterator[str]:
        self.mode.append("stream")
        yield "do"
        yield "ne"


def test_non_streaming_queries_never_open_a_stream(make_engine):
    chat = _SpyChat()
    engine = make_engine(chat_model=chat)
    engine.add_text("body text", name="doc")

    answer = engine.query("body")

    assert answer == "done"
    assert chat.mode == ["complete"]


def test_streaming_queries_use_the_stream(make_engine):
    chat = _SpyChat()
    engine = make_engine(chat_model=chat)
    engine.add_text("body text", name="doc")

    answer: Any = engine.query("body", stream=True)

    assert list(answer) == ["do", "ne"]
    assert chat.mode == ["stream"]
