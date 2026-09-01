"""Ollama backends, over plain HTTP.

Deliberately implemented with :mod:`urllib` rather than the ``ollama`` SDK: it
adds no dependency, and a fully local, fully offline softrag is the point of the
project. If the SDK is installed it is not used -- there is nothing it would add
here.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..errors import ChatError, EmbeddingError

log = logging.getLogger("softrag.providers.ollama")

__all__ = ["OllamaEmbedder", "OllamaChat", "is_available", "base_url"]

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "llama3.2"


def base_url() -> str:
    """The Ollama endpoint, honouring ``OLLAMA_HOST``."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def is_available(*, timeout: float = 0.5) -> bool:
    """Whether an Ollama daemon is reachable right now.

    Kept fast and quiet: this runs during auto-detection, where a slow or noisy
    probe would be worse than no probe at all.
    """
    try:
        with urllib.request.urlopen(f"{base_url()}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def _post(path: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_stream(
    path: str, payload: Dict[str, Any], *, timeout: float
) -> Iterator[Dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue


def _explain(exc: Exception, model: str) -> str:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
        return (
            f"Ollama has no model named {model!r}. Pull it first:  ollama pull {model}"
        )
    if isinstance(exc, urllib.error.URLError):
        return (
            f"Could not reach Ollama at {base_url()}. Is it running? "
            f"Start it with:  ollama serve"
        )
    return str(exc)


class OllamaEmbedder:
    """Embeddings from a local Ollama model.

    Args:
        model: Model name, for example ``"nomic-embed-text"`` or ``"mxbai-embed-large"``.
        timeout: Per-request timeout in seconds.
        batch_size: Inputs per request. Ollama's batch endpoint accepts a list.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        timeout: float = 120.0,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            try:
                response = _post(
                    "/api/embed",
                    {"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Ollama embeddings failed: {_explain(exc, self.model)}"
                ) from exc
            vectors = response.get("embeddings")
            if not vectors:
                raise EmbeddingError(
                    f"Ollama returned no embeddings for model {self.model!r}. "
                    "Confirm it is an embedding model, not a chat model."
                )
            out.extend([[float(v) for v in vector] for vector in vectors])
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OllamaEmbedder(model={self.model!r})"


class OllamaChat:
    """Chat completions from a local Ollama model.

    Args:
        model: Model name.
        timeout: Per-request timeout in seconds.
        temperature: Sampling temperature.
        system: System prompt.
        options: Extra Ollama options, merged into the request.
    """

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        timeout: float = 300.0,
        temperature: float = 0.0,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.system = system
        self.options = dict(options or {})

    def _payload(self, prompt: str, *, stream: bool) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.temperature, **self.options},
        }

    def complete(self, prompt: str) -> str:
        try:
            response = _post(
                "/api/chat", self._payload(prompt, stream=False), timeout=self.timeout
            )
        except Exception as exc:
            raise ChatError(f"Ollama chat failed: {_explain(exc, self.model)}") from exc
        return str(response.get("message", {}).get("content", ""))

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            for event in _post_stream(
                "/api/chat", self._payload(prompt, stream=True), timeout=self.timeout
            ):
                delta = event.get("message", {}).get("content")
                if delta:
                    yield delta
                if event.get("done"):
                    break
        except Exception as exc:
            raise ChatError(
                f"Ollama chat failed mid-stream: {_explain(exc, self.model)}"
            ) from exc

    def describe_image(self, image_base64: str, *, mime_type: str, prompt: str) -> str:
        """Caption an image with a vision model such as ``llava`` or ``llama3.2-vision``."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt, "images": [image_base64]}
            ],
            "stream": False,
        }
        try:
            response = _post("/api/chat", payload, timeout=self.timeout)
        except Exception as exc:
            raise ChatError(
                f"Ollama vision failed: {_explain(exc, self.model)}"
            ) from exc
        return str(response.get("message", {}).get("content", ""))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OllamaChat(model={self.model!r})"
