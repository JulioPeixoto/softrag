"""Turning inputs into indexable text.

This package answers one question for every kind of input -- file, URL, image --
"what is the text?" -- and returns it along with the metadata worth keeping.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import ExtractionError, IngestionError, UnsupportedFormatError
from .formats import EXTRACTORS, extension_for, html_to_text

log = logging.getLogger("softrag.ingest")

__all__ = [
    "DEFAULT_EXCLUDES",
    "EXTRACTORS",
    "IMAGE_EXTENSIONS",
    "caption_image",
    "discover_files",
    "extract_file",
    "extract_web",
]

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})

#: Directories nobody wants in a knowledge base.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    "**/.git/**",
    "**/.hg/**",
    "**/.svn/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/.next/**",
    "**/.idea/**",
    "**/*.egg-info/**",
)

#: Files larger than this are skipped by directory walks, which is almost always
#: what you want: a 200 MB log is not a document.
MAX_AUTO_FILE_BYTES = 32 * 1024 * 1024

FileInput = str | os.PathLike | bytes | bytearray


def extract_file(
    data: FileInput, *, name: str | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Extract text from a file path or raw bytes.

    Args:
        data: A filesystem path, or the file's bytes.
        name: Filename used to pick the extractor when ``data`` is bytes, and to
            name the source. Required for bytes in a non-plain-text format.

    Returns:
        ``(text, source_identifier, metadata)``.

    Raises:
        IngestionError: If the file is missing or empty.
        UnsupportedFormatError: If no extractor handles the extension.
        ExtractionError: If the extractor fails on this particular file.
    """
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        filename = name or ""
        source = name or f"bytes:{len(raw)}"
        metadata: dict[str, Any] = {"kind": "bytes", "bytes": len(raw)}
    else:
        path = Path(os.fspath(data))
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        if path.is_dir():
            raise IngestionError(
                f"{path} is a directory. Use rag.add_directory() to index it."
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise IngestionError(f"Could not read {path}: {exc}") from exc
        filename = path.name
        source = name or str(path)
        metadata = {
            "kind": "file",
            "filename": path.name,
            "extension": path.suffix.lower(),
            "bytes": len(raw),
            "path": str(path.resolve()),
        }

    if not raw:
        raise IngestionError(f"{source} is empty.")

    extension = extension_for(filename)
    if extension in IMAGE_EXTENSIONS:
        raise UnsupportedFormatError(
            f"{source} is an image. Use rag.add_image() instead, which captions "
            "it with a vision model so it becomes searchable."
        )

    extractor = EXTRACTORS.get(extension)
    if extractor is None:
        if _looks_like_text(raw):
            log.debug("no extractor for %r, treating %s as plain text", extension, source)
            extractor = EXTRACTORS[".txt"]
        else:
            raise UnsupportedFormatError(
                f"No extractor for {extension or 'files without an extension'} "
                f"({source}). Supported: {', '.join(sorted(set(EXTRACTORS)))}. "
                "Register your own with softrag.ingest.EXTRACTORS[ext] = fn."
            )

    text = extractor(raw, filename=filename)
    if not text.strip():
        raise ExtractionError(
            f"No text could be extracted from {source}. If it is a scanned PDF, "
            "it has no text layer and needs OCR first."
        )
    return text, source, metadata


def extract_web(url: str, *, timeout: float = 30.0) -> tuple[str, dict[str, Any]]:
    """Fetch a URL and reduce it to its main text.

    ``trafilatura`` is used when installed, since it is much better at throwing
    away navigation and boilerplate. Otherwise softrag falls back to its own
    HTML-to-text pass, which needs no dependencies.

    Args:
        url: The page to fetch.
        timeout: Network timeout in seconds.

    Returns:
        ``(text, metadata)``, where metadata carries the title when one is found.

    Raises:
        IngestionError: If the page cannot be fetched or holds no text.
    """
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ImportError:
        trafilatura = None  # type: ignore[assignment]

    if trafilatura is not None:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded, include_comments=False, include_tables=True
                )
                if text and text.strip():
                    metadata: dict[str, Any] = {"kind": "web", "url": url}
                    title = _trafilatura_title(trafilatura, downloaded)
                    if title:
                        metadata["title"] = title
                    return text.strip(), metadata
            log.debug("trafilatura returned nothing for %s, falling back", url)
        except Exception as exc:
            log.debug("trafilatura failed on %s (%s), falling back", url, exc)

    html, content_type = _fetch(url, timeout=timeout)
    if "html" not in content_type and "xml" not in content_type:
        text = html.strip()
        return text, {"kind": "web", "url": url, "content_type": content_type}

    text, title = html_to_text(html)
    if not text.strip():
        raise IngestionError(
            f"{url} returned a page with no readable text. It may be rendered "
            "entirely by JavaScript, which softrag does not execute."
        )
    metadata = {"kind": "web", "url": url}
    if title:
        metadata["title"] = title
    return text, metadata


def _trafilatura_title(module: Any, downloaded: Any) -> str:
    try:
        metadata = module.extract_metadata(downloaded)
    except Exception:
        return ""
    title = getattr(metadata, "title", None) if metadata else None
    return title or ""


def _fetch(url: str, *, timeout: float) -> tuple[str, str]:
    """Fetch a URL with httpx when available, urllib otherwise."""
    headers = {
        "User-Agent": "softrag/1.0 (+https://github.com/JulioPeixoto/softrag)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }
    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        httpx = None  # type: ignore[assignment]

    if httpx is not None:
        try:
            response = httpx.get(
                url, timeout=timeout, follow_redirects=True, headers=headers
            )
            response.raise_for_status()
            return response.text, response.headers.get("content-type", "").lower()
        except Exception as exc:
            raise IngestionError(f"Could not fetch {url}: {exc}") from exc

    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "").lower()
    except urllib.error.HTTPError as exc:
        raise IngestionError(
            f"Could not fetch {url}: HTTP {exc.code} {exc.reason}"
        ) from exc
    except Exception as exc:
        raise IngestionError(f"Could not fetch {url}: {exc}") from exc

    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        return raw.decode(charset, errors="replace"), content_type
    except LookupError:
        return raw.decode("utf-8", errors="replace"), content_type


DEFAULT_CAPTION_PROMPT = (
    "Describe this image for a search index. Cover the subject, setting, notable "
    "objects, colours, any visible text verbatim, and what is happening. Be "
    "specific and factual. Write a single dense paragraph with no preamble."
)


def caption_image(path: Path, chat_model: Any, *, prompt: str | None = None) -> str:
    """Describe an image so it can be retrieved by text search.

    Args:
        path: The image file.
        chat_model: A vision-capable chat backend.
        prompt: Override the captioning instruction.

    Returns:
        The description.

    Raises:
        ExtractionError: If the model cannot caption the image.
    """
    import base64

    extension = path.suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        log.warning("%s has an unexpected image extension %r", path, extension)

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(extension, "image/jpeg")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    instruction = prompt or DEFAULT_CAPTION_PROMPT

    describe = getattr(chat_model, "describe_image", None)
    if callable(describe):
        try:
            return str(describe(encoded, mime_type=mime, prompt=instruction)).strip()
        except Exception as exc:
            raise ExtractionError(f"Could not caption {path}: {exc}") from exc

    # Fall back to the LangChain multimodal message shape, which the common
    # chat wrappers understand.
    try:
        from langchain_core.messages import HumanMessage  # type: ignore[import-not-found]
    except ImportError:
        raise ExtractionError(
            f"The configured chat model cannot describe images. Use one of "
            f"softrag's providers (softrag.providers.OpenAIChat, AnthropicChat), "
            f"or install 'softrag[langchain]' to use a LangChain vision model. "
            f"Could not caption {path}."
        ) from None

    message = HumanMessage(
        content=[
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    )
    invoke = getattr(chat_model, "invoke", None)
    if not callable(invoke):
        raise ExtractionError(
            f"The configured chat model has no invoke() and cannot describe images. "
            f"Could not caption {path}."
        )
    try:
        response = invoke([message])
    except Exception as exc:
        raise ExtractionError(f"Could not caption {path}: {exc}") from exc

    content = getattr(response, "content", response)
    return str(content).strip()


def discover_files(
    base: Path,
    *,
    pattern: str = "**/*",
    exclude: Sequence[str] = (),
    recursive: bool = True,
    max_bytes: int = MAX_AUTO_FILE_BYTES,
) -> list[Path]:
    """Find indexable files under ``base``.

    Directories that never belong in a knowledge base -- ``.git``,
    ``node_modules``, virtualenvs, build output -- are skipped by default, as
    are files with no registered extractor and files above ``max_bytes``.

    Args:
        base: The directory to walk.
        pattern: Glob pattern relative to ``base``.
        exclude: Extra glob patterns to skip, added to the defaults.
        recursive: Walk subdirectories.
        max_bytes: Skip files larger than this.

    Returns:
        Matching paths, sorted for a stable ingestion order.
    """
    if not recursive and pattern.startswith("**/"):
        pattern = pattern[3:]

    patterns = tuple(DEFAULT_EXCLUDES) + tuple(exclude)
    found: list[Path] = []
    for path in sorted(base.glob(pattern)):
        if not path.is_file():
            continue
        as_posix = path.as_posix()
        if any(fnmatch.fnmatch(as_posix, p) for p in patterns):
            continue
        if path.suffix.lower() not in EXTRACTORS:
            continue
        try:
            if path.stat().st_size > max_bytes:
                log.debug("skipping %s: larger than %d bytes", path, max_bytes)
                continue
        except OSError:
            continue
        found.append(path)
    return found


def _looks_like_text(raw: bytes, *, sample: int = 4096) -> bool:
    """Heuristic: does this look like text rather than a binary blob?"""
    head = raw[:sample]
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        printable = sum(1 for byte in head if 32 <= byte < 127 or byte in (9, 10, 13))
        return printable / max(len(head), 1) > 0.9
