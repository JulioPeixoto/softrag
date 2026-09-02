"""Chunkers: recursive, markdown, sentence, and the resolution helpers."""

from __future__ import annotations

import itertools

import pytest

from softrag.chunking import (
    DEFAULT_SEPARATORS,
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    by_separator,
    resolve_chunker,
)

PARAGRAPHS = "\n\n".join(
    f"Paragraph {i} contains several words so that it has a realistic length."
    for i in range(20)
)


# --------------------------------------------------------------------------- #
# RecursiveChunker
# --------------------------------------------------------------------------- #


def test_the_documented_example_holds():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    assert chunker("alpha beta gamma delta epsilon") == [
        "alpha beta gamma",
        "gamma delta epsilon",
    ]


def test_text_shorter_than_chunk_size_is_one_chunk():
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
    assert chunker("a short document") == ["a short document"]


@pytest.mark.parametrize("size", [50, 120, 400])
def test_separable_text_respects_chunk_size(size):
    chunks = RecursiveChunker(chunk_size=size, chunk_overlap=10)(PARAGRAPHS)
    assert chunks
    assert all(len(chunk) <= size for chunk in chunks), [len(c) for c in chunks]


def test_overlap_is_actually_produced():
    chunker = RecursiveChunker(chunk_size=60, chunk_overlap=25)
    chunks = chunker(
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
    )
    assert len(chunks) > 1

    overlapping = 0
    for previous, current in itertools.pairwise(chunks):
        tail_words = previous.split()
        head_words = current.split()
        if set(tail_words) & set(head_words):
            overlapping += 1
    assert overlapping == len(chunks) - 1, "every boundary should share some text"


def test_zero_overlap_produces_no_repetition():
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=0)
    chunks = chunker("alpha bravo charlie delta echo foxtrot golf hotel india")
    assert len(chunks) > 1
    rejoined = "".join(chunks).replace(" ", "")
    assert rejoined == "alphabravocharliedeltaechofoxtrotgolfhotelindia"


def test_no_chunk_is_ever_empty_or_whitespace():
    messy = "alpha\n\n\n\n\n   \n\n bravo \n\n\n charlie\n\n\n\n"
    for size in (5, 12, 40, 500):
        chunks = RecursiveChunker(chunk_size=size, chunk_overlap=min(2, size - 1))(messy)
        assert all(chunk.strip() for chunk in chunks), chunks


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t \n"])
def test_empty_and_whitespace_input_yields_nothing(text):
    assert RecursiveChunker()(text) == []


def test_an_unbreakable_run_is_hard_split_and_fully_preserved():
    text = "z" * 10_000
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=0)(text)
    assert len(chunks) >= 100
    assert "".join(chunks) == text


def test_an_unbreakable_run_with_no_separators_at_all():
    chunker = RecursiveChunker(chunk_size=64, chunk_overlap=0, separators=())
    chunks = chunker("q" * 1000)
    assert all(len(chunk) <= 64 for chunk in chunks)
    assert "".join(chunks) == "q" * 1000


def test_overlap_not_smaller_than_chunk_size_raises():
    with pytest.raises(ValueError, match="must be smaller than"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="must be smaller than"):
        RecursiveChunker(chunk_size=100, chunk_overlap=250)


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveChunker(chunk_size=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        RecursiveChunker(chunk_size=100, chunk_overlap=-1)


def test_a_custom_length_function_changes_the_boundaries():
    words = " ".join(f"word{i}" for i in range(40))

    def word_count(text: str) -> int:
        return len(text.split())

    chunks = RecursiveChunker(chunk_size=8, chunk_overlap=2, length=word_count)(words)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 8 for chunk in chunks), [
        len(c.split()) for c in chunks
    ]


def test_keep_separator_false_drops_the_delimiters():
    chunker = RecursiveChunker(
        chunk_size=12, chunk_overlap=0, separators=(" ",), keep_separator=False
    )
    assert "".join(chunker("alpha bravo charlie delta")).count(" ") == 0


def test_strip_false_preserves_surrounding_whitespace():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, strip=False)
    assert any(chunk != chunk.strip() for chunk in chunker("  alpha   bravo  charlie  "))


def test_split_and_call_are_equivalent():
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=5)
    assert chunker(PARAGRAPHS) == chunker.split(PARAGRAPHS)


def test_the_default_separator_ladder_prefers_paragraphs_over_words():
    assert DEFAULT_SEPARATORS[0] == "\n\n\n"
    assert DEFAULT_SEPARATORS[-1] == ""
    text = "\n\n".join(["alpha bravo charlie"] * 4)
    chunks = RecursiveChunker(chunk_size=25, chunk_overlap=0)(text)
    assert all(chunk == "alpha bravo charlie" for chunk in chunks)


def test_hard_split_chunks_still_respect_chunk_size():
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=10)("z" * 10_000)
    assert all(len(chunk) <= 100 for chunk in chunks), sorted({len(c) for c in chunks})


# --------------------------------------------------------------------------- #
# MarkdownChunker
# --------------------------------------------------------------------------- #

MARKDOWN = """\
Some preamble before any heading at all.

# Guide

The guide body explains the basics.

## Install

Run the installer and follow the prompts.

## Configure

Edit the configuration file.

# Reference

The reference lists every option.
"""


def test_markdown_splits_on_headings():
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).split(MARKDOWN)
    assert len(chunks) == 5  # preamble + 4 headed sections
    assert any("installer" in chunk for chunk in chunks)
    assert any("reference lists" in chunk for chunk in chunks)


def test_markdown_keeps_a_preamble_before_the_first_heading():
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).split(MARKDOWN)
    assert chunks[0] == "Some preamble before any heading at all."


def test_markdown_prepends_the_heading_trail():
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).split(MARKDOWN)
    install = next(chunk for chunk in chunks if "installer" in chunk)
    assert install.startswith("# Guide > ## Install")


def test_markdown_trail_resets_at_the_next_top_level_heading():
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).split(MARKDOWN)
    reference = next(chunk for chunk in chunks if "reference lists" in chunk)
    assert reference.startswith("# Reference")
    assert "Guide" not in reference.splitlines()[0]


def test_markdown_can_omit_the_heading_trail():
    chunks = MarkdownChunker(
        chunk_size=500, chunk_overlap=50, include_heading_trail=False
    ).split(MARKDOWN)
    install = next(chunk for chunk in chunks if "installer" in chunk)
    assert not install.startswith("# Guide >")


def test_markdown_without_headings_falls_back_to_recursive_splitting():
    plain = " ".join(f"word{i}" for i in range(200))
    chunks = MarkdownChunker(chunk_size=100, chunk_overlap=10).split(plain)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_markdown_empty_input(text):
    assert MarkdownChunker().split(text) == []


def test_markdown_splits_a_section_that_is_too_long():
    long_section = "# Title\n\n" + " ".join(f"word{i}" for i in range(300))
    chunks = MarkdownChunker(chunk_size=120, chunk_overlap=10).split(long_section)
    assert len(chunks) > 1
    assert all(chunk.startswith("# Title") for chunk in chunks)


def test_markdown_is_callable():
    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
    assert chunker(MARKDOWN) == chunker.split(MARKDOWN)


def test_markdown_does_not_repeat_the_heading_twice():
    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=50).split(MARKDOWN)
    guide = next(chunk for chunk in chunks if "guide body" in chunk.lower())
    assert guide.count("# Guide") == 1, guide


# --------------------------------------------------------------------------- #
# SentenceChunker
# --------------------------------------------------------------------------- #

SENTENCES = (
    "First sentence here. Second sentence follows on. Third one arrives now. "
    "Fourth wraps everything up."
)


def test_sentences_are_never_split_in_the_middle():
    chunks = SentenceChunker(chunk_size=60, overlap_sentences=1).split(SENTENCES)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith((".", "!", "?")), chunk
        # No chunk may start mid-sentence either.
        assert chunk[0].isupper(), chunk


def test_sentence_overlap_repeats_whole_sentences():
    chunks = SentenceChunker(chunk_size=60, overlap_sentences=1).split(SENTENCES)
    for previous, current in itertools.pairwise(chunks):
        last_of_previous = previous.split(". ")[-1]
        assert current.startswith(last_of_previous.rstrip(".").split(". ")[-1][:10])


def test_sentence_overlap_can_be_disabled():
    chunks = SentenceChunker(chunk_size=60, overlap_sentences=0).split(SENTENCES)
    joined = " ".join(chunks)
    assert joined.count("Second sentence follows on.") == 1


def test_a_single_short_text_is_one_chunk():
    assert SentenceChunker(chunk_size=1000).split("Just one sentence.") == [
        "Just one sentence."
    ]


def test_a_sentence_longer_than_chunk_size_is_still_emitted_whole():
    long_sentence = "A " + "very " * 100 + "long sentence."
    chunks = SentenceChunker(chunk_size=50).split(long_sentence)
    assert chunks == [long_sentence.strip()]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_sentence_chunker_empty_input(text):
    assert SentenceChunker().split(text) == []


def test_sentence_chunker_is_callable():
    chunker = SentenceChunker(chunk_size=60)
    assert chunker(SENTENCES) == chunker.split(SENTENCES)


# --------------------------------------------------------------------------- #
# by_separator and resolve_chunker
# --------------------------------------------------------------------------- #


def test_by_separator_splits_and_strips():
    chunker = by_separator("---")
    assert chunker("alpha --- bravo ---   --- charlie") == ["alpha", "bravo", "charlie"]


def test_by_separator_without_stripping_keeps_empty_pieces():
    chunker = by_separator("---", strip=False)
    assert chunker("a---b") == ["a", "b"]
    assert chunker("a------b") == ["a", "", "b"]


def test_by_separator_on_text_without_the_separator():
    assert by_separator("|")("no separator here") == ["no separator here"]


def test_resolve_chunker_none_gives_the_default_recursive_chunker():
    chunker = resolve_chunker(None, chunk_size=42, chunk_overlap=7)
    assert isinstance(chunker, RecursiveChunker)
    assert chunker.chunk_size == 42
    assert chunker.chunk_overlap == 7


def test_resolve_chunker_string_gives_a_separator_chunker():
    chunker = resolve_chunker("\n---\n")
    assert callable(chunker)
    assert chunker("a\n---\nb") == ["a", "b"]


def test_resolve_chunker_passes_a_callable_through_unchanged():
    def custom(text: str):
        return [text.upper()]

    assert resolve_chunker(custom) is custom
    assert resolve_chunker(custom)("x") == ["X"]


def test_resolve_chunker_accepts_an_instantiated_chunker():
    chunker = SentenceChunker(chunk_size=30)
    assert resolve_chunker(chunker) is chunker


@pytest.mark.parametrize("bad", [42, 3.5, object(), ["a"], {"a": 1}])
def test_resolve_chunker_rejects_anything_else(bad):
    with pytest.raises(TypeError, match="chunker must be"):
        resolve_chunker(bad)
