"""OpenAI embedding and chat backends.

Thin wrappers over the official ``openai`` SDK. They exist so ``pip install
'softrag[openai]'`` is all it takes to get a working engine, without dragging in
a whole orchestration framework.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Iterator, List, Optional, Sequence

from ..errors import ChatError, ConfigurationError, EmbeddingError, MissingDependencyError

__all__ = ["OpenAIEmbedder", "OpenAIChat"]

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"


def _client(api_key: Optional[str], base_url: Optional[str], feature: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MissingDependencyError("openai", extra="openai", feature=feature) from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key and not base_url:
        raise ConfigurationError(
            "No OpenAI API key found. Set OPENAI_API_KEY, or pass api_key=..., or "
            "point base_url=... at an OpenAI-compatible server."
        )
    kwargs: dict[str, Any] = {"api_key": key or "not-needed"}
    if base_url or os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
    return OpenAI(**kwargs)


class OpenAIEmbedder:
    """Embeddings from OpenAI or any OpenAI-compatible endpoint.

    Args:
        model: Embedding model name.
        api_key: Overrides ``OPENAI_API_KEY``.
        base_url: Point at a compatible server (vLLM, LM Studio, Azure gateways).
        dimensions: Ask for shortened embeddings, supported by the v3 models.
            Smaller vectors mean a smaller, faster index at some accuracy cost.
        batch_size: Inputs per request.

    Example:
        >>> embedder = OpenAIEmbedder(dimensions=512)   # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 128,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        self._client = _client(api_key, base_url, "OpenAI embeddings")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            try:
                response = self._client.embeddings.create(**kwargs)
            except Exception as exc:
                raise EmbeddingError(
                    f"OpenAI embeddings failed for {len(batch)} inputs: {exc}"
                ) from exc
            # The API may return items out of order; index is authoritative.
            ordered = sorted(response.data, key=lambda item: item.index)
            out.extend(list(item.embedding) for item in ordered)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OpenAIEmbedder(model={self.model!r})"


class OpenAIChat:
    """Chat completions from OpenAI or any OpenAI-compatible endpoint.

    Args:
        model: Chat model name.
        api_key: Overrides ``OPENAI_API_KEY``.
        base_url: Point at a compatible server.
        temperature: Sampling temperature. Low values suit grounded answering.
        max_tokens: Cap on the response length.
        system: System prompt.
    """

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system
        self._client = _client(api_key, base_url, "OpenAI chat")

    def _messages(self, prompt: str) -> List[dict]:
        messages: List[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _kwargs(self) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                messages=self._messages(prompt), **self._kwargs()
            )
        except Exception as exc:
            raise ChatError(f"OpenAI chat failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                messages=self._messages(prompt), stream=True, **self._kwargs()
            )
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:
            raise ChatError(f"OpenAI chat failed mid-stream: {exc}") from exc

    def describe_image(
        self, image_base64: str, *, mime_type: str, prompt: str
    ) -> str:
        """Caption an image. Used by :meth:`softrag.Rag.add_image`."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise ChatError(f"OpenAI vision failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OpenAIChat(model={self.model!r})"
