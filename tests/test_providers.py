"""Backend adapters: every shape softrag promises to accept, and the fingerprint.

The regression this module exists for is at the bottom of
:func:`softrag.providers.adapt_embedder`: deciding whether a callable takes one
string or a list must be read off its *signature*. Probing it -- calling it once
to see what comes back -- would spend a real API request during construction,
before the user has indexed anything.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, List  # noqa: UP035 - the legacy spelling is under test

import pytest

from softrag import EchoChatModel, HashEmbedder, adapt_chat_model, adapt_embedder
from softrag.errors import ChatError, ConfigurationError, EmbeddingError
from softrag.providers import _takes_a_batch, embedder_fingerprint

DIM = 4


def vector(seed: float = 1.0) -> list[float]:
    return [seed, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Embedder shapes
# --------------------------------------------------------------------------- #


class LangChainStyle:
    """``embed_query`` plus ``embed_documents``: LangChain, and softrag's own."""

    def embed_query(self, text: str) -> list[float]:
        return vector(1.0)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector(float(i + 1)) for i in range(len(texts))]


class BatchOnly:
    """Only ``embed_documents``; the single-text path has to be derived."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [vector(float(len(t))) for t in texts]


class SentenceTransformersStyle:
    """``encode`` returning a nested list, as sentence-transformers does."""

    def encode(self, texts: Any) -> list[list[float]]:
        if isinstance(texts, str):
            return [vector(1.0)]
        return [vector(float(i + 1)) for i in range(len(texts))]


class ChromaStyle:
    """A Chroma embedding function: ``__call__(self, input)``."""

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [vector(float(i + 1)) for i in range(len(input))]


def test_embed_query_and_embed_documents_are_used_directly():
    embedder = adapt_embedder(LangChainStyle())
    assert embedder.embed_query("hello") == vector(1.0)
    assert embedder.embed_documents(["a", "b"]) == [vector(1.0), vector(2.0)]


def test_batch_only_embedders_get_a_derived_query_method():
    target = BatchOnly()
    embedder = adapt_embedder(target)

    assert embedder.embed_query("abcd") == vector(4.0)
    assert target.batches == [["abcd"]], (
        "the single query must go through as a batch of one"
    )


def test_encode_is_recognised_and_unwrapped():
    embedder = adapt_embedder(SentenceTransformersStyle())
    assert embedder.embed_query("hello") == vector(1.0)
    assert embedder.embed_documents(["a", "b", "c"]) == [
        vector(1.0),
        vector(2.0),
        vector(3.0),
    ]


def test_a_chroma_style_callable_is_batched():
    embedder = adapt_embedder(ChromaStyle())
    assert embedder.embed_documents(["a", "b"]) == [vector(1.0), vector(2.0)]
    assert embedder.embed_query("a") == vector(1.0)


def test_a_plain_string_callable_is_accepted():
    def embed(text: str) -> list[float]:
        return vector(float(len(text)))

    embedder = adapt_embedder(embed)
    assert embedder.embed_query("abc") == vector(3.0)
    assert embedder.embed_documents(["a", "bb"]) == [vector(1.0), vector(2.0)]


def test_numpy_style_return_values_are_coerced():
    class ArrayLike:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    def embed(text: str) -> Any:
        return ArrayLike([1.0, 2.0, 3.0, 4.0])

    assert adapt_embedder(embed).embed_query("x") == [1.0, 2.0, 3.0, 4.0]


# --------------------------------------------------------------------------- #
# The regression: adapting must never call the backend
# --------------------------------------------------------------------------- #


class CountingCallable:
    """A callable embedder that refuses to be probed silently."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return vector()


class CountingBatchCallable:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, documents: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [vector() for _ in documents]


@pytest.mark.parametrize("factory", [CountingCallable, CountingBatchCallable])
def test_adapting_a_callable_never_calls_it(factory):
    # Probing would cost a real request on a paid backend, at construction time,
    # before the user has asked for anything at all.
    target = factory()

    adapted = adapt_embedder(target)

    assert target.calls == 0, "adapt_embedder probed the callable instead of reading it"
    adapted.embed_query("only now")
    assert target.calls == 1


def test_adapting_an_object_embedder_never_calls_it():
    class Recording:
        def __init__(self) -> None:
            self.calls = 0

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls += 1
            return [vector() for _ in texts]

    target = Recording()
    adapt_embedder(target)
    assert target.calls == 0


def single_text(text: str) -> list[float]:
    return vector()


def batch_of_texts(texts: List[str]) -> list[list[float]]:  # noqa: UP006
    return [vector() for _ in texts]


def unannotated(thing):
    return vector()


class CallableWithSequence:
    def __call__(self, documents: Sequence[str]) -> list[list[float]]:
        return [vector() for _ in documents]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (single_text, False),
        (batch_of_texts, True),
        (unannotated, False),
        (CallableWithSequence(), True),
        (CountingCallable(), False),
    ],
)
def test_batch_shape_is_read_from_the_signature(target, expected):
    assert _takes_a_batch(target) is expected


def test_an_unannotated_callable_is_treated_as_single_text():
    calls: list[Any] = []

    def embed(thing):
        calls.append(thing)
        return vector()

    adapt_embedder(embed).embed_documents(["a", "b"])

    assert calls == ["a", "b"], "the safe fallback is one call per text"


# --------------------------------------------------------------------------- #
# Embedder errors
# --------------------------------------------------------------------------- #


def test_no_embedder_at_all_is_a_configuration_error():
    with pytest.raises(ConfigurationError) as excinfo:
        adapt_embedder(None)
    assert "embed_model" in str(excinfo.value)


def test_an_unadaptable_object_is_a_configuration_error():
    with pytest.raises(ConfigurationError) as excinfo:
        adapt_embedder(object())
    assert "embed_query" in str(excinfo.value)


def test_the_wrong_number_of_vectors_is_an_embedding_error():
    class Miscounting:
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [vector()]  # one vector no matter how many inputs

    with pytest.raises(EmbeddingError) as excinfo:
        adapt_embedder(Miscounting()).embed_documents(["a", "b", "c"])

    assert "1 vectors for 3 inputs" in str(excinfo.value)


def test_a_garbage_return_value_is_an_embedding_error():
    def embed(text: str) -> Any:
        return "not a vector"

    with pytest.raises(EmbeddingError):
        adapt_embedder(embed).embed_query("x")


def test_an_empty_vector_is_an_embedding_error():
    def embed(text: str) -> Any:
        return []

    with pytest.raises(EmbeddingError) as excinfo:
        adapt_embedder(embed).embed_query("x")
    assert "empty embedding" in str(excinfo.value)


def test_a_raising_backend_is_wrapped_as_an_embedding_error():
    def embed(text: str) -> Any:
        raise RuntimeError("the API is down")

    with pytest.raises(EmbeddingError) as excinfo:
        adapt_embedder(embed).embed_query("x")
    assert "the API is down" in str(excinfo.value)


def test_embedding_no_documents_costs_nothing():
    target = CountingCallable()
    assert adapt_embedder(target).embed_documents([]) == []
    assert target.calls == 0


# --------------------------------------------------------------------------- #
# Chat model shapes
# --------------------------------------------------------------------------- #


class Block:
    """An Anthropic-style content block."""

    def __init__(self, text: str) -> None:
        self.text = text


def test_a_complete_method_is_used():
    class Model:
        def complete(self, prompt: str) -> str:
            return f"answered: {prompt}"

    assert adapt_chat_model(Model()).complete("hi") == "answered: hi"


def test_an_invoke_returning_a_string_is_used():
    class Model:
        def invoke(self, prompt: str) -> str:
            return "plain string reply"

    assert adapt_chat_model(Model()).complete("hi") == "plain string reply"


def test_an_invoke_returning_a_message_object_is_unwrapped():
    class Message:
        content = "langchain reply"

    class Model:
        def invoke(self, prompt: str) -> Message:
            return Message()

    assert adapt_chat_model(Model()).complete("hi") == "langchain reply"


def test_anthropic_style_content_blocks_are_joined():
    class Message:
        def __init__(self) -> None:
            self.content = [Block("first "), Block("second")]

    class Model:
        def invoke(self, prompt: str) -> Message:
            return Message()

    assert adapt_chat_model(Model()).complete("hi") == "first second"


def test_dict_content_blocks_are_joined():
    class Message:
        def __init__(self) -> None:
            self.content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    class Model:
        def invoke(self, prompt: str) -> Message:
            return Message()

    assert adapt_chat_model(Model()).complete("hi") == "ab"


def test_a_bare_list_of_content_blocks_is_joined():
    # Regression: some backends return the block list itself rather than an
    # object wrapping it. Without an explicit branch the answer degrades to a
    # Python repr like "[<Block object at 0x...>]".
    class Model:
        def invoke(self, prompt: str) -> list[Block]:
            return [Block("a"), Block("b")]

    assert adapt_chat_model(Model()).complete("hi") == "ab"


def test_a_bare_callable_is_accepted():
    assert adapt_chat_model(lambda prompt: f"echo {prompt}").complete("hi") == "echo hi"


def test_an_object_with_a_text_attribute_is_unwrapped():
    class Reply:
        text = "reply text"

    assert adapt_chat_model(lambda prompt: Reply()).complete("hi") == "reply text"


def test_no_chat_model_is_a_configuration_error():
    with pytest.raises(ConfigurationError) as excinfo:
        adapt_chat_model(None)
    assert "rag.search" in str(excinfo.value)


def test_an_unadaptable_chat_object_is_a_configuration_error():
    with pytest.raises(ConfigurationError) as excinfo:
        adapt_chat_model(object())
    assert "complete()" in str(excinfo.value)


def test_a_raising_chat_backend_is_wrapped_as_a_chat_error():
    class Model:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("rate limited")

    with pytest.raises(ChatError) as excinfo:
        adapt_chat_model(Model()).complete("hi")
    assert "rate limited" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_a_native_stream_is_used_and_empty_deltas_dropped():
    class Model:
        def complete(self, prompt: str) -> str:
            return "abc"

        def stream(self, prompt: str) -> Iterator[str]:
            yield "a"
            yield ""
            yield "bc"

    adapted = adapt_chat_model(Model())
    assert adapted.supports_streaming is True
    assert list(adapted.stream("hi")) == ["a", "bc"]


def test_a_model_without_stream_falls_back_to_one_chunk():
    class Model:
        def complete(self, prompt: str) -> str:
            return "the whole answer at once"

    adapted = adapt_chat_model(Model())
    assert adapted.supports_streaming is False
    assert list(adapted.stream("hi")) == ["the whole answer at once"]


def test_a_stream_that_breaks_midway_is_wrapped_as_a_chat_error():
    class Model:
        def complete(self, prompt: str) -> str:
            return "unused"

        def stream(self, prompt: str) -> Iterator[str]:
            yield "partial"
            raise RuntimeError("connection reset")

    adapted = adapt_chat_model(Model())
    with pytest.raises(ChatError) as excinfo:
        list(adapted.stream("hi"))
    assert "connection reset" in str(excinfo.value)


def test_the_echo_model_streams_its_prompt():
    model = EchoChatModel()
    assert model.complete("a\nb") == "a\nb"
    assert list(model.stream("a\nb")) == ["a\n", "b"]


# --------------------------------------------------------------------------- #
# HashEmbedder
# --------------------------------------------------------------------------- #


def test_hash_embedder_is_deterministic():
    a = HashEmbedder(dimensions=32)
    b = HashEmbedder(dimensions=32)
    assert a.embed_query("the refund policy") == b.embed_query("the refund policy")


def test_hash_embedder_honours_dimensions():
    for width in (8, 64, 256):
        assert len(HashEmbedder(dimensions=width).embed_query("hello world")) == width


def test_hash_embedder_output_is_l2_normalised():
    for text in ("hello", "a much longer sentence with plenty of tokens", "x"):
        vec = HashEmbedder(dimensions=64).embed_query(text)
        assert sum(v * v for v in vec) == pytest.approx(1.0)


def test_hash_embedder_normalises_empty_text_without_dividing_by_zero():
    vec = HashEmbedder(dimensions=8).embed_query("")
    assert vec == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_hash_embedder_batches_match_single_calls():
    embedder = HashEmbedder(dimensions=16)
    texts = ["alpha beta", "gamma delta"]
    assert embedder.embed_documents(texts) == [embedder.embed_query(t) for t in texts]


def test_different_texts_get_different_vectors():
    embedder = HashEmbedder(dimensions=64)
    assert embedder.embed_query("mitochondria") != embedder.embed_query("refunds")


def test_a_non_positive_width_is_rejected():
    with pytest.raises(ConfigurationError):
        HashEmbedder(dimensions=0)


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #


def test_the_documented_fingerprint_example_holds():
    assert embedder_fingerprint(HashEmbedder(dimensions=64)) == "HashEmbedder:64"


def test_a_fingerprint_is_stable_across_instances():
    assert embedder_fingerprint(HashEmbedder(dimensions=64)) == embedder_fingerprint(
        HashEmbedder(dimensions=64)
    )


def test_fingerprints_differ_between_models():
    class Other:
        dimensions = 64

    assert embedder_fingerprint(HashEmbedder(dimensions=64)) != embedder_fingerprint(
        Other()
    )
    assert embedder_fingerprint(HashEmbedder(dimensions=64)) != embedder_fingerprint(
        HashEmbedder(dimensions=128)
    )


def test_the_model_name_wins_over_the_width():
    class Named:
        model_name = "text-embedding-3-small"
        dimensions = 1536

    assert embedder_fingerprint(Named()) == "Named:text-embedding-3-small"


def test_a_model_exposing_nothing_falls_back_to_its_class_name():
    class Anonymous:
        pass

    assert embedder_fingerprint(Anonymous()) == "Anonymous"


def test_the_fingerprint_sees_through_the_adapter():
    raw = HashEmbedder(dimensions=64)
    assert embedder_fingerprint(adapt_embedder(raw)) == embedder_fingerprint(raw)


def test_the_fingerprint_ignores_callable_attributes():
    class Callables:
        def model(self) -> str:
            return "not a name"

        dimensions = 32

    assert embedder_fingerprint(Callables()) == "Callables:32"
