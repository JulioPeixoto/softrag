"""Tests for softrag.rerank.

No network and no API keys: the LLM reranker is driven by scripted chat models,
including ones that return malformed output or raise, since falling back to the
input order instead of exploding is the contract.
"""

from __future__ import annotations

import importlib.util

import pytest

from softrag.errors import ConfigurationError, MissingDependencyError
from softrag.providers.local import CrossEncoderReranker as LocalCrossEncoderReranker
from softrag.rerank import (
    ChainReranker,
    CohereReranker,
    CrossEncoderReranker,
    DedupeReranker,
    LLMReranker,
    ScoreFusionReranker,
)
from softrag.types import Hit

# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #


def make_hits(n: int = 3, **fields: object) -> list[Hit]:
    """``n`` hits with distinguishable text, ids 0..n-1."""
    return [
        Hit(id=i, text=f"document number {i}", score=1.0 - i / 100, source=f"s{i}.md")
        for i in range(n)
    ]


class ScriptedChat:
    """Returns canned replies in order, then repeats the last one forever."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or [""]
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[index]


class ExplodingChat:
    """A backend that is simply down."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("backend on fire")


class RecordingReranker:
    """A stub stage that reverses hits and records the top_k it was given."""

    def __init__(self) -> None:
        self.seen_top_k: list[int] = []

    def rerank(self, query: str, hits, *, top_k: int) -> list[Hit]:
        self.seen_top_k.append(top_k)
        return list(reversed(list(hits)))[:top_k]


def ids(hits) -> list[int]:
    return [hit.id for hit in hits]


# --------------------------------------------------------------------------- #
# LLMReranker
# --------------------------------------------------------------------------- #


def test_llm_reranker_applies_a_json_permutation() -> None:
    hits = make_hits(3)
    reranker = LLMReranker(ScriptedChat("[3, 1, 2]"))
    assert ids(reranker.rerank("q", hits, top_k=3)) == [2, 0, 1]


def test_llm_reranker_accepts_a_json_code_fence() -> None:
    hits = make_hits(3)
    chat = ScriptedChat("Here you go:\n```json\n[2, 3, 1]\n```\n")
    assert ids(LLMReranker(chat).rerank("q", hits, top_k=3)) == [1, 2, 0]


def test_llm_reranker_accepts_json_objects() -> None:
    hits = make_hits(3)
    chat = ScriptedChat('[{"index": 3, "score": 0.9}, {"index": 2}, {"index": 1}]')
    assert ids(LLMReranker(chat).rerank("q", hits, top_k=3)) == [2, 1, 0]


def test_llm_reranker_accepts_a_bare_list_of_numbers() -> None:
    hits = make_hits(3)
    assert ids(LLMReranker(ScriptedChat("3, 1, 2")).rerank("q", hits, top_k=3)) == [
        2,
        0,
        1,
    ]


def test_llm_reranker_accepts_numbered_lines() -> None:
    hits = make_hits(3)
    chat = ScriptedChat("1. Document 3\n2. Document 1\n3. Document 2\n")
    # The leading list numbers are themselves in range, so the first integer of
    # each line wins; either way the parse must produce a valid permutation.
    out = ids(LLMReranker(chat).rerank("q", hits, top_k=3))
    assert sorted(out) == [0, 1, 2]


def test_llm_reranker_completes_a_partial_ranking() -> None:
    hits = make_hits(4)
    # The model only ranked two of the four; the rest keep their original order.
    out = ids(LLMReranker(ScriptedChat("[4, 2]")).rerank("q", hits, top_k=4))
    assert out == [3, 1, 0, 2]


def test_llm_reranker_ignores_out_of_range_numbers() -> None:
    hits = make_hits(3)
    out = ids(LLMReranker(ScriptedChat("[9, 2, 0, -1, 3]")).rerank("q", hits, top_k=3))
    assert out == [1, 2, 0]


@pytest.mark.parametrize(
    "reply",
    [
        "I am sorry, I cannot rank these documents.",
        "",
        "   \n  ",
        "{}",
        "[]",
        "[nope, nope]",
    ],
)
def test_llm_reranker_falls_back_on_malformed_output(reply: str) -> None:
    hits = make_hits(3)
    assert ids(LLMReranker(ScriptedChat(reply)).rerank("q", hits, top_k=3)) == [0, 1, 2]


def test_llm_reranker_falls_back_when_the_backend_raises() -> None:
    hits = make_hits(3)
    assert ids(LLMReranker(ExplodingChat()).rerank("q", hits, top_k=3)) == [0, 1, 2]


def test_llm_reranker_batches_and_interleaves_batch_winners() -> None:
    hits = make_hits(4)
    chat = ScriptedChat("[2, 1]", "[2, 1]")  # each batch of two, reversed
    out = ids(LLMReranker(chat, batch_size=2).rerank("q", hits, top_k=4))
    # Batch winners (1, 3) come first, then the runners-up (0, 2).
    assert out == [1, 3, 0, 2]
    assert len(chat.prompts) == 2


def test_llm_reranker_truncates_to_top_k_and_sets_scores() -> None:
    hits = make_hits(3)
    out = LLMReranker(ScriptedChat("[3, 2, 1]")).rerank("q", hits, top_k=2)
    assert ids(out) == [2, 1]
    assert out[0].score > out[1].score


def test_llm_reranker_handles_degenerate_inputs() -> None:
    reranker = LLMReranker(ScriptedChat("[1]"))
    assert reranker.rerank("q", [], top_k=5) == []
    assert reranker.rerank("q", make_hits(3), top_k=0) == []
    # A single candidate needs no LLM call at all.
    chat = ScriptedChat("[1]")
    assert ids(LLMReranker(chat).rerank("q", make_hits(1), top_k=5)) == [0]
    assert chat.prompts == []


def test_llm_reranker_prompt_is_customisable() -> None:
    chat = ScriptedChat("[1]")
    reranker = LLMReranker(chat, prompt="Q={query} D={documents}")
    reranker.rerank("needle", make_hits(2), top_k=2)
    assert chat.prompts[0].startswith("Q=needle D=[1] document number 0")


def test_llm_reranker_has_a_repr() -> None:
    assert "batch_size=7" in repr(LLMReranker(ScriptedChat("[1]"), batch_size=7))


# --------------------------------------------------------------------------- #
# CohereReranker
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    importlib.util.find_spec("cohere") is not None,
    reason="cohere is installed, so the missing-dependency path cannot be taken",
)
def test_cohere_reranker_requires_the_sdk() -> None:
    with pytest.raises(MissingDependencyError) as excinfo:
        CohereReranker()
    message = str(excinfo.value)
    assert "cohere" in message
    assert "softrag[rerank]" in message


@pytest.mark.skipif(
    importlib.util.find_spec("cohere") is None, reason="needs the cohere package"
)
def test_cohere_reranker_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        CohereReranker()


# --------------------------------------------------------------------------- #
# ScoreFusionReranker
# --------------------------------------------------------------------------- #


def fusion_hits() -> list[Hit]:
    """Three hits with hand-chosen scores.

    Dense similarity is ``1 - distance / 2``, so distances 0.0/1.0/2.0 give
    similarities 1.0/0.5/0.0 -> min-max normalised to 1.0/0.5/0.0.
    Sparse is ``-bm25``, so -1/-10/-5 give 1/10/5 -> normalised to
    0.0/1.0/0.4444.
    """
    hits = make_hits(3)
    for hit, distance, bm25 in zip(
        hits, (0.0, 1.0, 2.0), (-1.0, -10.0, -5.0), strict=True
    ):
        hit.vector_distance = distance
        hit.bm25 = bm25
    return hits


def test_score_fusion_alpha_one_is_dense_only() -> None:
    out = ScoreFusionReranker(alpha=1.0).rerank("q", fusion_hits(), top_k=3)
    assert ids(out) == [0, 1, 2]
    assert out[0].score == pytest.approx(1.0)


def test_score_fusion_alpha_zero_is_sparse_only() -> None:
    out = ScoreFusionReranker(alpha=0.0).rerank("q", fusion_hits(), top_k=3)
    assert ids(out) == [1, 2, 0]
    assert out[1].score == pytest.approx(4 / 9)


def test_score_fusion_convex_combination() -> None:
    # h0: 0.5*1.0 + 0.5*0.0    = 0.5
    # h1: 0.5*0.5 + 0.5*1.0    = 0.75
    # h2: 0.5*0.0 + 0.5*0.4444 = 0.2222
    out = ScoreFusionReranker(alpha=0.5).rerank("q", fusion_hits(), top_k=3)
    assert ids(out) == [1, 0, 2]
    assert out[0].score == pytest.approx(0.75)
    assert out[1].score == pytest.approx(0.5)


def test_score_fusion_treats_a_missing_channel_as_the_worst_value() -> None:
    hits = fusion_hits()
    hits[1].bm25 = None
    # Remaining sparse values 1 and 5 normalise to 0.0 and 1.0; h1 scores 0.0.
    out = ScoreFusionReranker(alpha=0.0).rerank("q", hits, top_k=3)
    assert ids(out)[0] == 2


def test_score_fusion_uses_the_only_available_channel() -> None:
    hits = fusion_hits()
    for hit in hits:
        hit.bm25 = None
    # alpha says "BM25 only", but there is no BM25 at all: use dense anyway.
    out = ScoreFusionReranker(alpha=0.0).rerank("q", hits, top_k=3)
    assert ids(out) == [0, 1, 2]


def test_score_fusion_without_any_scores_keeps_the_original_order() -> None:
    hits = make_hits(3)
    out = ScoreFusionReranker().rerank("q", hits, top_k=3)
    assert ids(out) == [0, 1, 2]


def test_score_fusion_handles_a_flat_channel() -> None:
    hits = make_hits(3)
    for hit in hits:
        hit.vector_distance = 0.4
        hit.bm25 = -2.0
    out = ScoreFusionReranker().rerank("q", hits, top_k=3)
    assert ids(out) == [0, 1, 2]


def test_score_fusion_zscore_normalisation() -> None:
    out = ScoreFusionReranker(alpha=1.0, normalize="zscore").rerank(
        "q", fusion_hits(), top_k=2
    )
    # Similarities 1.0/0.5/0.0 have mean 0.5, so the ordering is unchanged and
    # the top score is +1 standard deviation.
    assert ids(out) == [0, 1]
    assert out[0].score == pytest.approx(1.2247448, rel=1e-6)


def test_score_fusion_rejects_bad_configuration() -> None:
    with pytest.raises(ConfigurationError):
        ScoreFusionReranker(alpha=1.5)
    with pytest.raises(ConfigurationError):
        ScoreFusionReranker(normalize="softmax")


def test_score_fusion_has_a_repr() -> None:
    assert "alpha=0.7" in repr(ScoreFusionReranker(alpha=0.7))


# --------------------------------------------------------------------------- #
# ChainReranker
# --------------------------------------------------------------------------- #


def test_chain_applies_stages_in_order() -> None:
    first, second = RecordingReranker(), RecordingReranker()
    out = ChainReranker(first, second).rerank("q", make_hits(4), top_k=4)
    # Reversed twice is the identity.
    assert ids(out) == [0, 1, 2, 3]


def test_chain_gives_non_final_stages_the_whole_list() -> None:
    first, second = RecordingReranker(), RecordingReranker()
    ChainReranker(first, second).rerank("q", make_hits(5), top_k=2)
    assert first.seen_top_k == [5]
    assert second.seen_top_k == [2]


def test_chain_truncates_to_top_k() -> None:
    out = ChainReranker(RecordingReranker()).rerank("q", make_hits(5), top_k=2)
    assert ids(out) == [4, 3]


def test_empty_chain_is_a_truncating_no_op() -> None:
    out = ChainReranker().rerank("q", make_hits(4), top_k=2)
    assert ids(out) == [0, 1]


def test_chain_handles_degenerate_inputs() -> None:
    chain = ChainReranker(RecordingReranker())
    assert chain.rerank("q", [], top_k=3) == []
    assert chain.rerank("q", make_hits(2), top_k=0) == []


def test_chain_has_a_repr() -> None:
    assert repr(ChainReranker(DedupeReranker())).startswith("ChainReranker(Dedupe")


# --------------------------------------------------------------------------- #
# DedupeReranker
# --------------------------------------------------------------------------- #


def test_dedupe_drops_identical_text() -> None:
    hits = [
        Hit(id=0, text="The quick brown fox jumps over the lazy dog.", score=1.0),
        Hit(id=1, text="  the QUICK brown fox   jumps over the lazy dog.  ", score=0.9),
        Hit(id=2, text="SQLite stores the whole index in one portable file.", score=0.8),
    ]
    out = DedupeReranker().rerank("q", hits, top_k=5)
    assert ids(out) == [0, 2]


def test_dedupe_keeps_distinct_text() -> None:
    hits = [
        Hit(id=0, text="Refunds are issued within thirty days of purchase.", score=1.0),
        Hit(id=1, text="Shipping is free on orders above fifty euros.", score=0.9),
    ]
    assert ids(DedupeReranker().rerank("q", hits, top_k=5)) == [0, 1]


def test_dedupe_threshold_controls_strictness() -> None:
    hits = [
        Hit(id=0, text="Refunds are issued within thirty days of purchase.", score=1.0),
        Hit(id=1, text="Refunds are issued within thirty days of delivery.", score=0.9),
    ]
    # Near-identical, but not at 0.9 trigram overlap.
    assert ids(DedupeReranker(threshold=0.9).rerank("q", hits, top_k=5)) == [0, 1]
    assert ids(DedupeReranker(threshold=0.6).rerank("q", hits, top_k=5)) == [0]


def test_dedupe_respects_top_k_and_input_order() -> None:
    hits = make_hits(5)
    assert ids(DedupeReranker().rerank("q", hits, top_k=2)) == [0, 1]


def test_dedupe_handles_empty_text_and_degenerate_inputs() -> None:
    hits = [Hit(id=0, text="", score=1.0), Hit(id=1, text="", score=0.9)]
    assert ids(DedupeReranker().rerank("q", hits, top_k=5)) == [0, 1]
    assert DedupeReranker().rerank("q", [], top_k=5) == []
    assert DedupeReranker().rerank("q", hits, top_k=0) == []


def test_dedupe_rejects_a_bad_threshold() -> None:
    with pytest.raises(ConfigurationError):
        DedupeReranker(threshold=2.0)


def test_dedupe_has_a_repr() -> None:
    assert "threshold=0.8" in repr(DedupeReranker(threshold=0.8))


# --------------------------------------------------------------------------- #
# Re-exports
# --------------------------------------------------------------------------- #


def test_cross_encoder_is_re_exported_not_reimplemented() -> None:
    assert CrossEncoderReranker is LocalCrossEncoderReranker


def test_every_reranker_satisfies_the_protocol() -> None:
    from softrag.types import Reranker

    for reranker in (
        LLMReranker(ScriptedChat("[1]")),
        ScoreFusionReranker(),
        DedupeReranker(),
        ChainReranker(),
    ):
        assert isinstance(reranker, Reranker)
