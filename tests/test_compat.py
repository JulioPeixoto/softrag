"""The 0.1.x import path and call shape must keep working.

`softrag.softrag` was the single module the whole library lived in, and code
written against 0.1.x imports from it. These tests exist so that promise is
checked rather than assumed.
"""

from __future__ import annotations

import pytest

from softrag import EchoChatModel, HashEmbedder
from softrag.softrag import (
    Answer,
    ChatFn,
    EmbedFn,
    Hit,
    Rag,
    RagConfig,
    Store,
    connect,
    pack_vector,
)


def test_the_old_module_path_still_exports_the_engine():
    assert Rag is not None
    assert Store is not None
    assert connect is not None
    assert RagConfig is not None
    assert Answer is not None
    assert Hit is not None


def test_the_old_type_aliases_still_resolve():
    # 0.1.x annotated user code with these; they must stay importable even
    # though nothing in the library takes them as parameters any more.
    assert EmbedFn is not None
    assert ChatFn is not None


def test_pack_vector_is_still_importable_from_the_old_path():
    packed = pack_vector([1.0, 0.0, 0.0, 0.0])
    assert isinstance(packed, bytes)
    assert len(packed) == 16  # four float32 values


@pytest.fixture
def legacy_rag(tmp_path):
    """An engine built exactly the way 0.1.x documented it."""
    return Rag(
        embed_model=HashEmbedder(32),
        chat_model=EchoChatModel(),
        db_path=tmp_path / "legacy.db",
    )


def test_the_0_1_x_constructor_signature_still_works(legacy_rag):
    assert len(legacy_rag) == 0


def test_the_0_1_x_methods_still_exist(legacy_rag):
    for name in ("add_file", "add_web", "add_image", "query"):
        assert callable(getattr(legacy_rag, name)), name


def test_query_returns_a_plain_string_now(legacy_rag):
    """The one deliberate break, pinned so it cannot regress by accident.

    0.1.x returned whatever the chat backend gave back -- usually a LangChain
    message that callers had to unwrap with `.content`. It now returns an
    `Answer`, which *is* a `str`, so `print(rag.query(q))` keeps working while
    `.content` is gone.
    """
    legacy_rag.add_text("softrag keeps everything in one sqlite file", name="a")
    answer = legacy_rag.query("what does softrag keep?")

    assert isinstance(answer, str)
    assert not hasattr(answer, "content")
    assert answer.sources == ["a"]


def test_a_0_1_x_style_duck_typed_embedder_still_works(tmp_path):
    """0.1.x accepted any object with embed_query; that must still hold."""

    class LegacyEmbedder:
        def embed_query(self, text):
            return [float(len(text) % 7)] * 16

    rag = Rag(
        embed_model=LegacyEmbedder(),
        chat_model=EchoChatModel(),
        db_path=tmp_path / "duck.db",
    )
    rag.add_text("hello from a duck-typed embedder", name="duck")
    assert [hit.source for hit in rag.search("duck")] == ["duck"]
    rag.close()
