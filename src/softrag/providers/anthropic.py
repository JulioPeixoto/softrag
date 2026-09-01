"""Anthropic (Claude) chat backend.

Anthropic does not serve an embedding model, so this module provides chat and
vision only. Pair it with any embedder -- OpenAI, Ollama or a local model.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional

from ..errors import ChatError, ConfigurationError, MissingDependencyError

__all__ = ["AnthropicChat"]

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 2048


class AnthropicChat:
    """Chat completions from Claude.

    Args:
        model: Model id.
        api_key: Overrides ``ANTHROPIC_API_KEY``.
        max_tokens: Cap on the response length. Required by the API, so it
            always has a value.
        temperature: Sampling temperature. Low values suit grounded answering.
        system: System prompt.

    Example:
        >>> chat = AnthropicChat(model="claude-sonnet-5")    # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError(
                "anthropic", extra="anthropic", feature="Claude chat"
            ) from exc

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or pass "
                "api_key=... to AnthropicChat."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self._client = anthropic.Anthropic(api_key=key)

    def _kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.system:
            kwargs["system"] = self.system
        return kwargs

    def complete(self, prompt: str) -> str:
        try:
            message = self._client.messages.create(
                messages=[{"role": "user", "content": prompt}], **self._kwargs()
            )
        except Exception as exc:
            raise ChatError(f"Anthropic chat failed: {exc}") from exc
        return _text_of(message)

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                messages=[{"role": "user", "content": prompt}], **self._kwargs()
            ) as stream:
                yield from stream.text_stream
        except Exception as exc:
            raise ChatError(f"Anthropic chat failed mid-stream: {exc}") from exc

    def describe_image(self, image_base64: str, *, mime_type: str, prompt: str) -> str:
        """Caption an image. Used by :meth:`softrag.Rag.add_image`."""
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise ChatError(f"Anthropic vision failed: {exc}") from exc
        return _text_of(message)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AnthropicChat(model={self.model!r})"


def _text_of(message: Any) -> str:
    """Join the text blocks of a Messages API response."""
    parts: List[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "".join(parts)
