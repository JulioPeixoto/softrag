"""Query- and ingest-time transforms.

Every transform here spends LLM calls, and every one of them promises the same
thing: a bad model reply degrades to the untransformed input rather than raising.
So the fakes below deliberately return malformed output -- preamble, numbered
lists, half-JSON, nothing at all -- and the tests assert the defensive parsing
holds and the ordering guarantees survive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from conftest import make_rag
from softrag import ContextualChunker, Rag, contextualize, expand_query, hyde
from softrag.transforms import (
    CONTEXTUAL_PROMPT,
    DEFAULT_EXPANSION_PROMPT,
    DEFAULT_HYDE_PROMPT,
    _parse_variants,
    multi_query_search,
)

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class CannedChat:
    """Returns a fixed reply and remembers every prompt it saw."""

    def __init__(self, reply: Any = "canned reply") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self.reply


class BrokenChat:
    """Fails on every call, the way a rate-limited backend does."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("the backend is unavailable")


class SelectivelyBrokenChat:
    """Fails only for prompts containing a marker, and answers everything else."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def complete(self, prompt: str) -> str:
        if self.marker in prompt:
            raise RuntimeError(f"cannot handle {self.marker}")
        return "SITUATED"


# --------------------------------------------------------------------------- #
# hyde
# --------------------------------------------------------------------------- #


def test_hyde_returns_the_generated_passage():
    chat = CannedChat("  Refunds are issued within thirty days of purchase.  ")

    passage = hyde("how long is the refund window?", chat)

    assert passage == "Refunds are issued within thirty days of purchase."
    assert "how long is the refund window?" in chat.prompts[0]
    assert chat.prompts[0].startswith(DEFAULT_HYDE_PROMPT[:20])


def test_hyde_accepts_a_custom_prompt():
    chat = CannedChat("passage")
    hyde("q", chat, prompt="WRITE A PASSAGE FOR: {question}")
    assert chat.prompts[0] == "WRITE A PASSAGE FOR: q"


def test_hyde_falls_back_to_the_question_when_generation_fails():
    assert hyde("the original question", BrokenChat()) == "the original question"


@pytest.mark.parametrize("reply", ["", "   \n  ", "\t"])
def test_hyde_falls_back_when_the_passage_is_blank(reply):
    assert hyde("the original question", CannedChat(reply)) == "the original question"


def test_hyde_stringifies_a_non_string_reply():
    class Weird:
        def __str__(self) -> str:
            return "stringified passage"

    assert hyde("q", CannedChat(Weird())) == "stringified passage"


# --------------------------------------------------------------------------- #
# expand_query
# --------------------------------------------------------------------------- #


def test_expansion_keeps_the_original_first():
    chat = CannedChat("first rewrite\nsecond rewrite\nthird rewrite")

    variants = expand_query("refund window?", chat, n=3)

    assert variants[0] == "refund window?"
    assert variants == [
        "refund window?",
        "first rewrite",
        "second rewrite",
        "third rewrite",
    ]
    assert "refund window?" in chat.prompts[0]
    assert (
        DEFAULT_EXPANSION_PROMPT.format(question="refund window?", n=3) == chat.prompts[0]
    )


def test_expansion_strips_preamble_and_list_markers():
    chat = CannedChat(
        "Here are three rewrites:\n"
        "1. How many days to return an item?\n"
        "- What is the return period?\n"
        "* Can I still send it back?\n"
    )

    variants = expand_query("refund window?", chat, n=3)

    assert variants[1:] == [
        "How many days to return an item?",
        "What is the return period?",
        "Can I still send it back?",
    ]


def test_expansion_reads_a_json_array_when_the_model_emits_one():
    chat = CannedChat('Sure thing!\n["alpha rewrite", "beta rewrite"]')
    assert expand_query("q", chat, n=3) == ["q", "alpha rewrite", "beta rewrite"]


def test_expansion_drops_a_rewrite_identical_to_the_question():
    chat = CannedChat("REFUND WINDOW?\ngenuinely different rewrite")
    assert expand_query("refund window?", chat, n=3) == [
        "refund window?",
        "genuinely different rewrite",
    ]


def test_expansion_never_returns_more_than_n_rewrites():
    chat = CannedChat("\n".join(f"rewrite {i}" for i in range(20)))
    assert len(expand_query("q", chat, n=2)) == 3  # the question plus two


def test_expansion_falls_back_to_the_question_alone_when_the_model_fails():
    assert expand_query("the original question", BrokenChat()) == [
        "the original question"
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "Here are the rewrites:",  # preamble only, nothing usable after it
    ],
)
def test_expansion_falls_back_when_nothing_can_be_parsed(reply):
    assert expand_query("the original question", CannedChat(reply)) == [
        "the original question"
    ]


def test_an_empty_json_array_means_no_rewrites():
    assert expand_query("the original question", CannedChat("[]")) == [
        "the original question"
    ]


def test_expansion_with_a_non_positive_n_never_calls_the_model():
    chat = CannedChat("should not be produced")
    assert expand_query("q", chat, n=0) == ["q"]
    assert expand_query("q", chat, n=-1) == ["q"]
    assert chat.prompts == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("a\nb", ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        ('["a", 5, null, "b"]', ["a", "b"]),  # non-strings are discarded
        ("[not valid json]", ["[not valid json]"]),  # falls through to line mode
        ("1) one\n2. two\n• three", ["one", "two", "three"]),
        ('"quoted"', ["quoted"]),
        ("", []),
    ],
)
def test_variant_parsing_is_defensive(reply, expected):
    assert _parse_variants(reply) == expected


# --------------------------------------------------------------------------- #
# multi_query_search
# --------------------------------------------------------------------------- #


@pytest.fixture
def searchable():
    engine = make_rag()
    for name, text in {
        "handbook": "The refund policy allows returns within thirty days.",
        "billing": "Refunds are credited to the original payment method.",
        "changelog": "Version two introduced hybrid retrieval.",
        "biology": "Mitochondria are the powerhouse of the cell.",
    }.items():
        assert engine.add_text(text, name=name).ok
    try:
        yield engine
    finally:
        engine.close()


def test_multi_query_search_returns_each_document_once(searchable: Rag):
    chat = CannedChat("refunds returns\nrefund credited\nreturns policy")

    hits = multi_query_search(searchable, "refund policy", n=3, top_k=4, chat_model=chat)

    assert hits
    ids = [hit.id for hit in hits]
    assert len(ids) == len(set(ids)), "fusion must not emit a document twice"
    assert len({hit.source for hit in hits}) == len(hits)


def test_multi_query_search_scores_are_fused_and_descending(searchable: Rag):
    chat = CannedChat("refunds returns\nrefund credited")
    hits = multi_query_search(searchable, "refund policy", n=2, top_k=4, chat_model=chat)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_multi_query_search_respects_top_k(searchable: Rag):
    chat = CannedChat("refunds returns\nrefund credited\nreturns policy")
    hits = multi_query_search(searchable, "refund policy", n=3, top_k=2, chat_model=chat)
    assert len(hits) <= 2


def test_multi_query_search_with_no_top_k_does_nothing(searchable: Rag):
    assert multi_query_search(searchable, "refund", top_k=0) == []


def test_multi_query_search_degrades_to_one_search_when_expansion_fails(searchable: Rag):
    plain = searchable.search("refund policy", top_k=3)
    fused = multi_query_search(
        searchable, "refund policy", n=3, top_k=3, chat_model=BrokenChat()
    )
    assert [hit.id for hit in fused] == [hit.id for hit in plain]


def test_multi_query_search_without_a_chat_model_still_searches():
    engine = make_rag(chat_model=None, auto=False)
    try:
        engine.add_text("The refund policy allows returns.", name="handbook")
        hits = multi_query_search(engine, "refund", top_k=3)
        assert [hit.source for hit in hits] == ["handbook"]
    finally:
        engine.close()


def test_multi_query_search_skips_a_variant_whose_search_explodes(searchable: Rag):
    class Flaky:
        def __init__(self, inner: Rag) -> None:
            self.inner = inner
            self.seen: list[str] = []

        def search(self, query: str, **kwargs: Any):
            self.seen.append(query)
            if "credited" in query:
                raise RuntimeError("index unavailable")
            return self.inner.search(query, **kwargs)

    flaky = Flaky(searchable)
    chat = CannedChat("refunds returns\nrefund credited")

    hits = multi_query_search(flaky, "refund policy", n=2, top_k=3, chat_model=chat)

    assert hits, "one failing variant must not empty the result"
    assert any("credited" in query for query in flaky.seen)


# --------------------------------------------------------------------------- #
# contextualize
# --------------------------------------------------------------------------- #

DOCUMENT = "ACME Corp Q2 2024 report, covering the quarter in three short notes."
CHUNKS = ["Revenue grew by 3%.", "Costs fell by 1%.", "Headcount was flat."]


def test_contextualize_prepends_the_blurb_and_keeps_order():
    chat = CannedChat("This chunk is from ACME Corp's Q2 2024 report")

    out = contextualize(DOCUMENT, CHUNKS, chat)

    assert len(out) == len(CHUNKS)
    for original, contextualised in zip(CHUNKS, out, strict=True):
        assert contextualised.endswith(original)
        assert contextualised.startswith("This chunk is from ACME")
        assert "\n\n" in contextualised


def test_contextualize_uses_the_published_prompt():
    chat = CannedChat("context")
    contextualize(DOCUMENT, CHUNKS[:1], chat)
    assert chat.prompts[0] == CONTEXTUAL_PROMPT.format(document=DOCUMENT, chunk=CHUNKS[0])


def test_contextualize_accepts_a_custom_prompt():
    chat = CannedChat("context")
    contextualize(DOCUMENT, CHUNKS[:1], chat, prompt="D={document} C={chunk}")
    assert chat.prompts[0] == f"D={DOCUMENT} C={CHUNKS[0]}"


def test_a_failing_chunk_falls_back_to_the_original_not_the_whole_batch():
    # Only the middle chunk's call fails; the other two must still be enriched.
    chat = SelectivelyBrokenChat("Costs fell by 1%.")

    out = contextualize(DOCUMENT, CHUNKS, chat)

    assert out[0] == f"SITUATED\n\n{CHUNKS[0]}"
    assert out[1] == CHUNKS[1], "the failing chunk is indexed as-is"
    assert out[2] == f"SITUATED\n\n{CHUNKS[2]}"


def test_an_empty_blurb_falls_back_to_the_original_chunk():
    assert contextualize(DOCUMENT, CHUNKS, CannedChat("   ")) == CHUNKS


def test_contextualizing_nothing_costs_nothing():
    chat = CannedChat("context")
    assert contextualize(DOCUMENT, [], chat) == []
    assert chat.prompts == []


def test_order_survives_a_wide_thread_pool():
    class Indexing:
        """Answers with the chunk it was given, so misordering is detectable."""

        def complete(self, prompt: str) -> str:
            body = prompt.split("<chunk>")[1].split("</chunk>")[0].strip()
            return f"ctx-for-{body}"

    chunks = [f"chunk number {i}" for i in range(16)]
    out = contextualize("document", chunks, Indexing(), max_workers=8)

    assert out == [f"ctx-for-{chunk}\n\n{chunk}" for chunk in chunks]


# --------------------------------------------------------------------------- #
# ContextualChunker
# --------------------------------------------------------------------------- #


def test_the_contextual_chunker_is_a_plain_callable():
    chunker = ContextualChunker("|", CannedChat("CTX"))
    assert chunker("alpha|beta") == ["CTX\n\nalpha", "CTX\n\nbeta"]


def test_the_contextual_chunker_on_empty_text_returns_nothing():
    assert ContextualChunker("|", CannedChat("CTX"))("") == []


def test_the_contextual_chunker_plugs_into_the_engine():
    chat = CannedChat("SECTION CONTEXT")
    engine = make_rag(chunker=ContextualChunker("\n\n", chat))
    try:
        engine.add_text("first paragraph body\n\nsecond paragraph body", name="doc")

        stored = [
            row[0]
            for row in engine.store.db.execute(
                "SELECT text FROM documents ORDER BY chunk_index"
            )
        ]
        assert stored == [
            "SECTION CONTEXT\n\nfirst paragraph body",
            "SECTION CONTEXT\n\nsecond paragraph body",
        ]
        # The blurb is indexed, so it is searchable alongside the chunk.
        assert engine.search("SECTION CONTEXT", mode="keyword")
    finally:
        engine.close()


def test_the_contextual_chunker_survives_a_broken_backend():
    engine = make_rag(chunker=ContextualChunker("\n\n", BrokenChat()))
    try:
        result = engine.add_text("first paragraph\n\nsecond paragraph", name="doc")
        assert result.ok
        assert result.chunks_added == 2
    finally:
        engine.close()


def test_the_contextual_chunker_reprs_its_inner_chunker():
    assert "ContextualChunker" in repr(ContextualChunker(None, CannedChat()))


def test_a_chunker_argument_may_be_a_sequence_of_strings():
    """The chunker contract is ``str -> list[str]``; nothing more is required."""

    def constant(text: str) -> Sequence[str]:
        return ["one", "two"]

    chunker = ContextualChunker(constant, CannedChat("CTX"))
    assert chunker("ignored") == ["CTX\n\none", "CTX\n\ntwo"]
