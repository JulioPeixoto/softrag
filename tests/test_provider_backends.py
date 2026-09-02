"""The vendor backends, exercised against fake SDKs.

These wrappers are the code most users actually run, and until now nothing ran
them: they need `openai`, `anthropic` or a live Ollama daemon, so the offline
suite skipped them entirely and a typo in a parameter name would have shipped.

The fakes here stand in for the real SDKs and for urllib, so the request each
backend builds and the response it unpacks are both checked without a network
or a key.
"""

from __future__ import annotations

import io
import json
import sys
import types
from typing import ClassVar

import pytest

from softrag.errors import ChatError, ConfigurationError, EmbeddingError

# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class _Embedding:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _EmbeddingResponse:
    def __init__(self, data):
        self.data = data


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _ChatResponse:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _StreamEvent:
    def __init__(self, content):
        self.choices = [_StreamChoice(content)]


class FakeOpenAI:
    """Records every call so the request shape can be asserted."""

    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.embedding_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.embeddings = types.SimpleNamespace(create=self._create_embeddings)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create_chat)
        )
        FakeOpenAI.instances.append(self)

    def _create_embeddings(self, **kwargs):
        self.embedding_calls.append(kwargs)
        rows = list(enumerate(kwargs["input"]))
        # Deliberately out of order: the API does not promise ordering, and the
        # wrapper is supposed to sort on `index`.
        return _EmbeddingResponse(
            [_Embedding(i, [float(i), float(len(t))]) for i, t in reversed(rows)]
        )

    def _create_chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_StreamEvent("Hel"), _StreamEvent("lo"), _StreamEvent(None)])
        return _ChatResponse("hello")


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances.clear()
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return module


def test_openai_embedder_batches_and_restores_order(fake_openai):
    from softrag.providers.openai import OpenAIEmbedder

    embedder = OpenAIEmbedder("text-embedding-3-small", batch_size=2)
    vectors = embedder.embed_documents(["a", "bb", "ccc"])

    assert len(vectors) == 3
    # Sorted back into input order despite the fake returning them reversed.
    assert vectors[0][0] == 0.0
    assert [v[1] for v in vectors] == [1.0, 2.0, 3.0]

    calls = FakeOpenAI.instances[0].embedding_calls
    assert [len(c["input"]) for c in calls] == [2, 1], "batch_size not honoured"
    assert all(c["model"] == "text-embedding-3-small" for c in calls)


def test_openai_embedder_passes_shortened_dimensions(fake_openai):
    from softrag.providers.openai import OpenAIEmbedder

    OpenAIEmbedder(dimensions=512).embed_query("x")
    assert FakeOpenAI.instances[0].embedding_calls[0]["dimensions"] == 512


def test_openai_embedder_omits_dimensions_when_unset(fake_openai):
    from softrag.providers.openai import OpenAIEmbedder

    OpenAIEmbedder().embed_query("x")
    assert "dimensions" not in FakeOpenAI.instances[0].embedding_calls[0]


def test_openai_embedder_never_sends_an_empty_string(fake_openai):
    """The API rejects an empty input, and a blank chunk is not worth a 400."""
    from softrag.providers.openai import OpenAIEmbedder

    OpenAIEmbedder().embed_documents(["", "   ", "real"])
    assert FakeOpenAI.instances[0].embedding_calls[0]["input"] == [" ", " ", "real"]


def test_openai_embedder_wraps_failures(fake_openai, monkeypatch):
    from softrag.providers.openai import OpenAIEmbedder

    embedder = OpenAIEmbedder()

    def boom(**kwargs):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(embedder._client.embeddings, "create", boom)
    with pytest.raises(EmbeddingError, match="429"):
        embedder.embed_query("x")


def test_openai_chat_builds_the_expected_request(fake_openai):
    from softrag.providers.openai import OpenAIChat

    chat = OpenAIChat("gpt-4.1-mini", temperature=0.0, max_tokens=64, system="be terse")
    assert chat.complete("hi") == "hello"

    call = FakeOpenAI.instances[0].chat_calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 64
    assert call["messages"][0] == {"role": "system", "content": "be terse"}
    assert call["messages"][1] == {"role": "user", "content": "hi"}


def test_openai_chat_streams_and_skips_empty_deltas(fake_openai):
    from softrag.providers.openai import OpenAIChat

    assert list(OpenAIChat().stream("hi")) == ["Hel", "lo"]
    assert FakeOpenAI.instances[0].chat_calls[0]["stream"] is True


def test_openai_chat_describes_an_image(fake_openai):
    from softrag.providers.openai import OpenAIChat

    assert OpenAIChat().describe_image("QUJD", mime_type="image/png", prompt="what?")
    content = FakeOpenAI.instances[0].chat_calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_openai_without_a_key_or_base_url_is_a_configuration_error(
    fake_openai, monkeypatch
):
    from softrag.providers.openai import OpenAIEmbedder

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()


def test_a_base_url_alone_is_enough_for_a_compatible_server(fake_openai, monkeypatch):
    from softrag.providers.openai import OpenAIEmbedder

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAIEmbedder(base_url="http://localhost:8000/v1")
    assert FakeOpenAI.instances[0].init_kwargs["base_url"] == "http://localhost:8000/v1"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class _Block:
    def __init__(self, text):
        self.text = text


class _AnthropicMessage:
    def __init__(self, blocks):
        self.content = blocks


class _StreamContext:
    def __init__(self, parts):
        self.text_stream = iter(parts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeAnthropic:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls: list[dict] = []
        self.messages = types.SimpleNamespace(create=self._create, stream=self._stream)
        FakeAnthropic.instances.append(self)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _AnthropicMessage([_Block("he"), _Block("llo")])

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return _StreamContext(["he", "llo"])


@pytest.fixture
def fake_anthropic(monkeypatch):
    FakeAnthropic.instances.clear()
    module = types.ModuleType("anthropic")
    module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return module


def test_anthropic_joins_content_blocks(fake_anthropic):
    from softrag.providers.anthropic import AnthropicChat

    assert AnthropicChat().complete("hi") == "hello"


def test_anthropic_always_sends_max_tokens(fake_anthropic):
    """The Messages API rejects a request without it, so it can never be absent."""
    from softrag.providers.anthropic import AnthropicChat

    AnthropicChat(max_tokens=128, system="be terse").complete("hi")
    call = FakeAnthropic.instances[0].calls[0]
    assert call["max_tokens"] == 128
    assert call["system"] == "be terse"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_streams(fake_anthropic):
    from softrag.providers.anthropic import AnthropicChat

    assert list(AnthropicChat().stream("hi")) == ["he", "llo"]


def test_anthropic_describes_an_image(fake_anthropic):
    from softrag.providers.anthropic import AnthropicChat

    assert AnthropicChat().describe_image("QUJD", mime_type="image/png", prompt="what?")
    content = FakeAnthropic.instances[0].calls[0]["messages"][0]["content"]
    assert content[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "QUJD",
    }


def test_anthropic_without_a_key_is_a_configuration_error(fake_anthropic, monkeypatch):
    from softrag.providers.anthropic import AnthropicChat

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicChat()


def test_anthropic_wraps_failures(fake_anthropic, monkeypatch):
    from softrag.providers.anthropic import AnthropicChat

    chat = AnthropicChat()
    monkeypatch.setattr(
        chat._client.messages,
        "create",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("overloaded")),
    )
    with pytest.raises(ChatError, match="overloaded"):
        chat.complete("hi")


# --------------------------------------------------------------------------- #
# Ollama (plain HTTP, so urllib is what gets faked)
# --------------------------------------------------------------------------- #


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_http(monkeypatch):
    """Capture the request each Ollama call builds and script the reply."""
    sent: list[dict] = []
    replies: list[bytes] = []

    def urlopen(request, timeout=None):
        payload = getattr(request, "data", None)
        sent.append(
            {
                "url": request.full_url if hasattr(request, "full_url") else request,
                "body": json.loads(payload) if payload else None,
            }
        )
        return _Response(replies.pop(0) if replies else b"{}")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return types.SimpleNamespace(sent=sent, replies=replies)


def test_ollama_embedder_uses_the_batch_endpoint(fake_http):
    from softrag.providers.ollama import OllamaEmbedder

    fake_http.replies.append(
        json.dumps({"embeddings": [[1.0, 2.0], [3.0, 4.0]]}).encode()
    )
    vectors = OllamaEmbedder("nomic-embed-text").embed_documents(["a", "b"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    request = fake_http.sent[0]
    assert request["url"].endswith("/api/embed")
    assert request["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_ollama_embedder_explains_an_empty_response(fake_http):
    from softrag.providers.ollama import OllamaEmbedder

    fake_http.replies.append(b"{}")
    with pytest.raises(EmbeddingError, match="embedding model"):
        OllamaEmbedder().embed_query("x")


def test_ollama_chat_sends_the_expected_payload(fake_http):
    from softrag.providers.ollama import OllamaChat

    fake_http.replies.append(json.dumps({"message": {"content": "hello"}}).encode())
    assert OllamaChat("llama3.2", system="be terse").complete("hi") == "hello"

    body = fake_http.sent[0]["body"]
    assert body["model"] == "llama3.2"
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["options"]["temperature"] == 0.0


def test_ollama_chat_streams_newline_delimited_json(fake_http):
    from softrag.providers.ollama import OllamaChat

    fake_http.replies.append(
        b'{"message":{"content":"he"}}\n{"message":{"content":"llo"}}\n{"done":true}\n'
    )
    assert list(OllamaChat().stream("hi")) == ["he", "llo"]


def test_ollama_reports_a_missing_model_with_the_pull_command(monkeypatch):
    import urllib.error

    from softrag.providers.ollama import OllamaChat

    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(ChatError, match="ollama pull missing-model"):
        OllamaChat("missing-model").complete("hi")


def test_ollama_reports_an_unreachable_daemon_with_the_serve_command(monkeypatch):
    import urllib.error

    from softrag.providers.ollama import OllamaEmbedder

    def urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(EmbeddingError, match="ollama serve"):
        OllamaEmbedder().embed_query("x")


def test_ollama_base_url_honours_the_environment(monkeypatch):
    from softrag.providers import ollama

    monkeypatch.setenv("OLLAMA_HOST", "example.test:1234")
    assert ollama.base_url() == "http://example.test:1234"

    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.internal/")
    assert ollama.base_url() == "https://ollama.internal"


def test_is_available_is_false_when_nothing_answers(monkeypatch):
    from softrag.providers import ollama

    def urlopen(request, timeout=None):
        raise OSError("nope")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert ollama.is_available() is False
