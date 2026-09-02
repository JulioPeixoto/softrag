"""The ``softrag`` command line.

The CLI is a thin, honest wrapper around :class:`softrag.Rag`: every command
maps onto one or two library calls, so anything you can do here you can also do
in three lines of Python.

It has no dependencies of its own. ``rich`` is used for tables, colour and
progress bars when it happens to be installed (``pip install 'softrag[cli]'``),
and every code path degrades to :func:`print` when it is not.

Run ``softrag --help`` for the command list, or ``softrag doctor`` when
something looks wrong.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__, providers
from .engine import Rag
from .errors import ConfigurationError, IngestionError, SoftragError
from .types import Hit, IngestResult, SourceInfo

log = logging.getLogger("softrag.cli")

__all__ = ["build_parser", "main"]

#: Backends selectable with ``--provider``.
PROVIDERS: tuple[str, ...] = (
    "auto",
    "openai",
    "anthropic",
    "ollama",
    "local",
    "hash",
)

#: Environment variable consulted when ``--db`` is not given.
DB_ENV_VAR = "SOFTRAG_DB"

DEFAULT_DB = "softrag.db"

#: Characters of chunk text shown per hit unless ``--full`` is passed.
PREVIEW_CHARS = 220

#: Optional packages reported by ``softrag doctor``.
OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("sentence_transformers", "local"),
    ("pypdf", "files"),
    ("httpx", "web"),
    ("trafilatura", "web"),
    ("rich", "cli"),
)

#: Keys ``softrag doctor`` checks for. Only presence is ever reported.
API_KEY_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VOYAGE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def rich_available() -> bool:
    """Whether ``rich`` can be imported.

    Checked with :mod:`importlib.util` so the import cost is only paid when the
    package is actually there.
    """
    return importlib.util.find_spec("rich") is not None


class Printer:
    """Console output that upgrades itself when ``rich`` is installed.

    Every method works identically without ``rich``; the difference is colour,
    box-drawing and column alignment. Text is never interpreted as markup, so a
    chunk containing ``[bold]`` prints as-is instead of exploding.

    Args:
        plain: Force the dependency-free path, whatever is installed.
    """

    def __init__(self, *, plain: bool = False) -> None:
        self._out: Any = None
        self._err: Any = None
        if not plain and rich_available():
            from rich.console import Console

            self._out = Console(highlight=False, markup=False, emoji=False)
            self._err = Console(stderr=True, highlight=False, markup=False, emoji=False)

    @property
    def fancy(self) -> bool:
        """Whether the rich-backed path is in use."""
        return self._out is not None

    def line(self, text: str = "", *, style: str | None = None) -> None:
        """Print one line to stdout, optionally styled."""
        if self._out is not None:
            self._out.print(text, style=style)
        else:
            print(text)

    def segments(self, *parts: tuple[str, str | None]) -> None:
        """Print one line assembled from ``(text, style)`` pairs."""
        if self._out is not None:
            from rich.text import Text

            line = Text()
            for text, style in parts:
                line.append(text, style=style)
            self._out.print(line)
        else:
            print("".join(text for text, _ in parts))

    def error(self, text: str) -> None:
        """Print one line to stderr, in red when possible."""
        if self._err is not None:
            self._err.print(text, style="red")
        else:
            print(text, file=sys.stderr)

    def table(
        self,
        columns: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        title: str | None = None,
        right_align: Sequence[int] = (),
    ) -> None:
        """Print a table, boxed with ``rich`` and space-aligned without it.

        Args:
            columns: Column headers.
            rows: Row values, already stringified.
            title: Optional caption above the table.
            right_align: Indices of columns holding numbers.
        """
        if self._out is not None:
            from rich.table import Table

            table = Table(title=title, header_style="bold", title_justify="left")
            for index, column in enumerate(columns):
                table.add_column(
                    column,
                    justify="right" if index in right_align else "left",
                    overflow="fold",
                )
            for row in rows:
                table.add_row(*row)
            self._out.print(table)
            return

        if title:
            print(title)
        widths = [len(column) for column in columns]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        header = "  ".join(
            _pad(column, widths[index], index in right_align)
            for index, column in enumerate(columns)
        )
        print(header)
        print("  ".join("-" * width for width in widths))
        for row in rows:
            print(
                "  ".join(
                    _pad(cell, widths[index], index in right_align)
                    for index, cell in enumerate(row)
                )
            )


def _pad(text: str, width: int, right: bool) -> str:
    return text.rjust(width) if right else text.ljust(width)


class _Progress:
    """Per-file progress reporting for ``softrag add``.

    Uses a ``rich`` bar on an interactive terminal and one line per file
    everywhere else, which keeps logs and CI output readable.
    """

    def __init__(self, printer: Printer, total: int, *, quiet: bool) -> None:
        self._printer = printer
        self._total = total
        self._quiet = quiet
        self._done = 0
        self._lock = threading.Lock()
        self._bar: Any = None
        self._task: Any = None
        self._interactive = printer.fancy and sys.stdout.isatty()

    def __enter__(self) -> _Progress:
        if self._quiet or not self._interactive:
            return self
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._bar.start()
        self._task = self._bar.add_task("indexing", total=self._total)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._bar is not None:
            self._bar.stop()

    def advance(self, label: str, result: IngestResult) -> None:
        """Record one finished file."""
        with self._lock:
            self._done += 1
            position = self._done
        shown = display_source(label, width=40 if self._bar is not None else 56)
        if self._bar is not None:
            self._bar.update(self._task, advance=1, description=shown)
            return
        if result.error:
            self._printer.error(
                f"[{position}/{self._total}] {shown} ... FAILED: {result.error}"
            )
        elif not self._quiet:
            count = result.chunks_added
            self._printer.line(
                f"[{position}/{self._total}] {shown} ... "
                f"{count} chunk{'' if count == 1 else 's'}"
            )


# --------------------------------------------------------------------------- #
# Argument helpers
# --------------------------------------------------------------------------- #


def coerce_value(raw: str) -> Any:
    """Turn a ``--metadata`` string into the type it looks like.

    ``true``/``false`` become booleans, digits become numbers, and everything
    else stays a string. This is what makes ``--metadata year=2024`` filterable
    with ``--where '{"year": {"$gte": 2024}}'`` instead of comparing strings.

    Args:
        raw: The raw right-hand side of a ``KEY=VALUE`` pair.

    Returns:
        The coerced value.

    Example:
        >>> coerce_value("2024"), coerce_value("true"), coerce_value("x")
        (2024, True, 'x')
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return raw


def parse_metadata(pairs: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` options into a metadata dictionary.

    Args:
        pairs: Raw ``--metadata`` values.

    Returns:
        The parsed mapping, empty when nothing was passed.

    Raises:
        ConfigurationError: If an entry has no ``=``.
    """
    metadata: dict[str, Any] = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ConfigurationError(
                f"--metadata expects KEY=VALUE, got {pair!r}. "
                "For example: --metadata team=platform --metadata year=2024"
            )
        metadata[key.strip()] = coerce_value(value)
    return metadata


def parse_where(raw: str | None) -> dict[str, Any] | None:
    """Parse a ``--where`` filter expression.

    Args:
        raw: A JSON object, or ``None``.

    Returns:
        The decoded filter, or ``None``.

    Raises:
        ConfigurationError: If the text is not a JSON object.
    """
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"--where is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(
            '--where must be a JSON object, for example \'{"year": {"$gte": 2024}}\'.'
        )
    return value


def resolve_db_path(args: argparse.Namespace) -> str:
    """Decide which database file to use.

    Precedence is ``--db``, then ``$SOFTRAG_DB``, then ``softrag.db`` in the
    working directory.
    """
    return getattr(args, "db", None) or os.environ.get(DB_ENV_VAR) or DEFAULT_DB


def _looks_like_url(text: str) -> bool:
    return text.strip().lower().startswith(("http://", "https://"))


def display_source(source: str, *, width: int = 60) -> str:
    """Shorten a source identifier so one hit fits on one line.

    Absolute paths inside the working directory are shown relative to it, which
    is both shorter and how the user typed them. Anything still too long keeps
    its tail -- the filename is what identifies a document, so a path loses its
    head and a URL loses its query string.

    Only display is affected: ``softrag ls`` and every ``--json`` output print
    identifiers in full, so what you copy back into ``softrag rm`` always works.

    Args:
        source: The stored source identifier.
        width: Maximum characters to show.

    Returns:
        The shortened label.
    """
    text = source
    if not _looks_like_url(text):
        try:
            candidate = Path(text)
            if candidate.is_absolute():
                relative = os.path.relpath(candidate, Path.cwd())
                if not relative.startswith(".."):
                    text = relative
        except (OSError, ValueError):
            pass
    if len(text) <= width:
        return text
    if _looks_like_url(text):
        return text[: width - 3] + "..."
    return "..." + text[-(width - 3) :]


def _snippet(text: str, *, full: bool) -> str:
    """Collapse a chunk to a single readable preview line."""
    if full:
        return text.strip()
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS].rstrip() + "..."


def human_bytes(size: int) -> str:
    """Format a byte count for people.

    Example:
        >>> human_bytes(2048)
        '2.0 KB'
    """
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #


def build_embedder(provider: str, model: str | None) -> Any:
    """Resolve ``--provider`` / ``--embed-model`` into an embedder.

    ``None`` is returned for the fully automatic case so the engine keeps its
    own lazy detection, which is what users expect from a bare ``softrag add``.

    Args:
        provider: One of :data:`PROVIDERS`.
        model: Model name override, or ``None`` for the backend default.

    Returns:
        An embedder, or ``None`` to let the engine choose.

    Raises:
        ConfigurationError: If the provider cannot supply embeddings.
    """
    if provider == "auto":
        return providers.auto_embedder(model=model) if model else None
    if provider == "hash":
        # No download, no key, no network -- for trying the CLI out and for
        # tests. Retrieval quality is poor by construction.
        return providers.HashEmbedder()
    if provider == "openai":
        from .providers.openai import OpenAIEmbedder

        return OpenAIEmbedder(model or "text-embedding-3-small")
    if provider == "ollama":
        from .providers.ollama import OllamaEmbedder

        return OllamaEmbedder(model or "nomic-embed-text")
    if provider == "local":
        from .providers.local import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    if provider == "anthropic":
        # Anthropic ships no embedding endpoint, so embeddings still come from
        # auto-detection. Saying so beats a confusing failure later.
        log.debug("provider=anthropic supplies chat only; auto-detecting embeddings")
        return providers.auto_embedder(model=model) if model else None
    raise ConfigurationError(f"Unknown provider {provider!r}.")


def build_chat_model(provider: str, model: str | None) -> Any:
    """Resolve ``--provider`` / ``--chat-model`` into a chat backend.

    As with :func:`build_embedder`, ``None`` means "let the engine detect one
    lazily", which keeps retrieval-only commands free of model construction.

    Raises:
        ConfigurationError: If the provider cannot supply a chat model.
    """
    if provider == "auto":
        return providers.auto_chat_model(model=model) if model else None
    if provider == "hash":
        # Pairs with the hash embedder: echoes the retrieved context instead of
        # generating, so what you see is exactly what retrieval found.
        return providers.EchoChatModel()
    if provider == "anthropic":
        from .providers.anthropic import AnthropicChat

        return AnthropicChat(model or "claude-sonnet-5")
    if provider == "openai":
        from .providers.openai import OpenAIChat

        return OpenAIChat(model or "gpt-4.1-mini")
    if provider == "ollama":
        from .providers.ollama import OllamaChat

        return OllamaChat(model or "llama3.2")
    if provider == "local":
        from .providers import ollama as ollama_provider

        if not ollama_provider.is_available():
            raise ConfigurationError(
                "--provider local has no chat backend: sentence-transformers only "
                "does embeddings. Start an Ollama daemon for local generation, or "
                "use `softrag search` which needs no chat model at all."
            )
        from .providers.ollama import OllamaChat

        return OllamaChat(model or "llama3.2")
    raise ConfigurationError(f"Unknown provider {provider!r}.")


def open_engine(args: argparse.Namespace, *, chat: bool = False) -> Rag:
    """Open the index described by ``args``.

    Commands that never generate text pass ``chat=False``, which leaves
    ``rag.chat_model`` untouched -- no API client is constructed, no key is
    required, and no model is downloaded.

    Args:
        args: Parsed arguments carrying the engine flags.
        chat: Whether this command may need to generate text.

    Returns:
        A ready engine. The caller owns it and should close it.
    """
    provider = getattr(args, "provider", "auto") or "auto"
    overrides: dict[str, Any] = {}
    for flag in ("chunk_size", "chunk_overlap"):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[flag] = value
    workers = getattr(args, "workers", None)
    if workers is not None:
        overrides["max_workers"] = workers

    return Rag(
        db_path=resolve_db_path(args),
        embed_model=build_embedder(provider, getattr(args, "embed_model", None)),
        chat_model=(
            build_chat_model(provider, getattr(args, "chat_model", None))
            if chat
            else None
        ),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


def _expand_sources(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Turn the raw ``add`` arguments into concrete ``(kind, value)`` jobs.

    Directories are walked here rather than inside the engine so that
    ``--on-change`` and ``--workers`` apply uniformly to every file.

    Raises:
        IngestionError: If a source is neither a URL, ``-``, nor an existing path.
    """
    from .ingest import discover_files

    jobs: list[tuple[str, str]] = []
    for raw in args.sources:
        if raw == "-":
            jobs.append(("stdin", "-"))
            continue
        if _looks_like_url(raw):
            jobs.append(("url", raw))
            continue
        path = Path(raw)
        if path.is_dir():
            found = discover_files(
                path,
                pattern=args.pattern,
                exclude=tuple(args.exclude or ()),
            )
            if not found:
                log.warning("no indexable files under %s (pattern %r)", raw, args.pattern)
            jobs.extend(("file", str(item)) for item in found)
            continue
        if not path.exists():
            raise IngestionError(
                f"File not found: {raw}. Pass a file, a directory, a URL, or '-' "
                "to read text from stdin."
            )
        jobs.append(("file", str(path)))
    return jobs


def cmd_add(args: argparse.Namespace) -> int:
    """Index files, directories, URLs or stdin."""
    printer = Printer()
    metadata = parse_metadata(args.metadata)
    jobs = _expand_sources(args)
    if not jobs:
        printer.error("error: nothing to index.")
        return 1

    stdin_text: str | None = None
    if any(kind == "stdin" for kind, _ in jobs):
        stdin_text = sys.stdin.read()
        if not stdin_text.strip():
            raise IngestionError("Nothing was piped in on stdin.")

    started = time.perf_counter()
    results: list[IngestResult] = [IngestResult(source="")] * len(jobs)

    with open_engine(args) as rag:

        def work(index: int, kind: str, value: str) -> None:
            try:
                if kind == "stdin":
                    results[index] = rag.add_text(
                        stdin_text or "",
                        metadata=metadata,
                        name=args.name or "stdin",
                        on_change=args.on_change,
                    )
                else:
                    results[index] = rag.add(
                        value, metadata=metadata, on_change=args.on_change
                    )
            except SoftragError as exc:
                results[index] = IngestResult(source=value, error=str(exc))
            except Exception as exc:
                log.debug("unexpected failure on %s", value, exc_info=True)
                results[index] = IngestResult(source=value, error=str(exc))

        workers = max(1, args.workers or rag.config.max_workers)
        with (
            _Progress(printer, len(jobs), quiet=args.quiet) as progress,
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            futures = {
                pool.submit(work, index, kind, value): (index, value)
                for index, (kind, value) in enumerate(jobs)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                index, value = futures[future]
                progress.advance(value, results[index])

    elapsed = time.perf_counter() - started
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    added = sum(r.chunks_added for r in ok)
    skipped = sum(r.chunks_skipped for r in ok)

    printer.line(
        f"Indexed {len(ok)} file{'' if len(ok) == 1 else 's'} in {elapsed:.2f}s: "
        f"{added} chunks added, {skipped} skipped."
    )
    if failed:
        printer.error(f"{len(failed)} failed:")
        for result in failed:
            printer.error(f"  {result.source}: {result.error}")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def _hit_payload(hit: Hit, rank: int) -> dict[str, Any]:
    """Render a hit as a JSON-safe dictionary."""
    return {
        "rank": rank,
        "score": hit.score,
        "source": hit.source,
        "index": hit.index,
        "text": hit.text,
        "metadata": hit.metadata,
        "vector_distance": hit.vector_distance,
        "bm25": hit.bm25,
    }


def _print_hits(printer: Printer, hits: Sequence[Hit], *, full: bool) -> None:
    """Print search hits as a numbered, scored list."""
    for rank, hit in enumerate(hits, start=1):
        printer.segments(
            (f"{rank:>2}. ", "dim"),
            (f"{hit.score:.4f}", "bold cyan"),
            ("  ", None),
            (display_source(hit.source) if hit.source else "(unnamed)", "green"),
            (f"  #{hit.index}", "dim"),
        )
        printer.line(f"    {_snippet(hit.text, full=full)}")
        printer.line()


def cmd_search(args: argparse.Namespace) -> int:
    """Retrieve chunks without calling a chat model."""
    where = parse_where(args.where)
    with open_engine(args) as rag:
        hits = rag.search(
            args.query,
            top_k=args.top_k,
            mode=args.mode,
            where=where,
            source=args.source,
        )

    if args.json:
        payload = {
            "query": args.query,
            "count": len(hits),
            "results": [_hit_payload(hit, rank) for rank, hit in enumerate(hits, 1)],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    printer = Printer()
    if not hits:
        printer.line("No matching chunks.")
        return 0
    printer.line(
        f"{len(hits)} result{'' if len(hits) == 1 else 's'} for {args.query!r}",
        style="bold",
    )
    printer.line()
    _print_hits(printer, hits, full=args.full)
    return 0


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #


def _load_prompt(path: str | None) -> str | None:
    """Read a prompt template from disk and check it has the required fields."""
    if path is None:
        return None
    try:
        template = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Could not read --prompt-file {path}: {exc}") from exc
    missing = [f for f in ("{context}", "{question}") if f not in template]
    if missing:
        raise ConfigurationError(
            f"The prompt in {path} is missing {' and '.join(missing)}. A template "
            "must contain both so the retrieved context and the question can be "
            "substituted in."
        )
    return template


def _print_sources(printer: Printer, hits: Sequence[Hit]) -> None:
    """Print the compact provenance list shown under an answer."""
    seen: dict[str, float] = {}
    for hit in hits:
        label = hit.source or "(unnamed)"
        seen[label] = max(seen.get(label, 0.0), hit.score)
    if not seen:
        return
    printer.line()
    printer.line("sources:", style="bold")
    for rank, (label, score) in enumerate(seen.items(), start=1):
        printer.segments(
            (f"  [{rank}] ", "dim"),
            (display_source(label), "green"),
            (f"  {score:.4f}", "dim"),
        )


def cmd_query(args: argparse.Namespace) -> int:
    """Answer a question with retrieval-augmented generation."""
    where = parse_where(args.where)
    prompt = _load_prompt(args.prompt_file)
    stream = not args.no_stream and not args.json

    with open_engine(args, chat=True) as rag:
        answer = rag.query(
            args.question,
            top_k=args.top_k,
            mode=args.mode,
            where=where,
            source=args.source,
            stream=stream,
            prompt=prompt,
        )

        if args.json:
            payload = {
                "question": args.question,
                "answer": str(answer),
                "sources": list(answer.sources),
                "hits": [_hit_payload(h, r) for r, h in enumerate(answer.hits, 1)],
            }
            print(json.dumps(payload, indent=2, default=str))
            return 0

        printer = Printer()
        if stream:
            for delta in answer:
                sys.stdout.write(delta)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            printer.line(str(answer))

        if args.show_sources:
            _print_sources(printer, answer.hits)
    return 0


# --------------------------------------------------------------------------- #
# ls / rm / stats / optimize
# --------------------------------------------------------------------------- #


def _source_payload(info: SourceInfo) -> dict[str, Any]:
    return {
        "source": info.source,
        "chunks": info.chunks,
        "characters": info.characters,
        "added_at": info.added_at,
        "metadata": info.metadata,
    }


def cmd_ls(args: argparse.Namespace) -> int:
    """List the sources present in an index."""
    with open_engine(args) as rag:
        sources = rag.sources(limit=args.limit)

    if args.json:
        print(json.dumps([_source_payload(s) for s in sources], indent=2, default=str))
        return 0

    printer = Printer()
    if not sources:
        printer.line("The index is empty. Add something with `softrag add <file>`.")
        return 0
    printer.table(
        ("SOURCE", "CHUNKS", "CHARS", "ADDED"),
        [
            (
                info.source,
                str(info.chunks),
                str(info.characters),
                str(info.added_at)[:19],
            )
            for info in sources
        ],
        right_align=(1, 2),
    )
    printer.line()
    printer.line(
        f"{len(sources)} source{'' if len(sources) == 1 else 's'}, "
        f"{sum(s.chunks for s in sources)} chunks.",
        style="dim",
    )
    return 0


def _confirm(question: str, *, assume_yes: bool) -> bool:
    """Ask before something irreversible, unless ``--yes`` was passed."""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise ConfigurationError(
            f"{question} Re-run with --yes to confirm, since stdin is not a terminal."
        )
    reply = input(f"{question} [y/N] ").strip().lower()
    return reply in ("y", "yes")


def cmd_rm(args: argparse.Namespace) -> int:
    """Delete sources, filtered chunks, or the whole index."""
    where = parse_where(args.where)
    if where is not None and not where:
        raise ConfigurationError(
            "An empty --where filter matches every chunk. Use --all --yes if that "
            "is really what you meant."
        )
    if args.all and (args.sources or where is not None):
        raise ConfigurationError("--all removes everything; drop the other arguments.")
    if args.sources and where is not None:
        raise ConfigurationError(
            "Pass sources or --where, not both: they would delete different things."
        )
    if not args.all and not args.sources and where is None:
        raise ConfigurationError(
            "Nothing to remove. Pass one or more sources, --where JSON, or --all."
        )

    printer = Printer()
    with open_engine(args) as rag:
        if args.all:
            total = len(rag)
            if not _confirm(
                f"Delete all {total} chunks from {rag.store.path}?",
                assume_yes=args.yes,
            ):
                printer.line("Aborted.")
                return 0
            rag.reset()
            printer.line(f"Removed {total} chunks.")
            return 0

        if where is not None:
            removed = rag.delete(where=where)
            printer.line(f"Removed {removed} chunks matching the filter.")
            return 0

        removed = 0
        missing: list[str] = []
        for source in args.sources:
            count = rag.delete(source=source)
            if count == 0:
                missing.append(source)
            removed += count
        printer.line(
            f"Removed {removed} chunks from {len(args.sources) - len(missing)} source(s)."
        )
        for source in missing:
            printer.error(f"  not indexed: {source}")
    return 1 if missing else 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Summarise what an index holds."""
    with open_engine(args) as rag:
        stats = rag.stats()

    if args.json:
        print(
            json.dumps(
                {
                    "path": stats.path,
                    "size_bytes": stats.size_bytes,
                    "chunks": stats.documents,
                    "sources": stats.sources,
                    "dimensions": stats.dimensions,
                    "schema_version": stats.schema_version,
                },
                indent=2,
            )
        )
        return 0

    printer = Printer()
    printer.table(
        ("FIELD", "VALUE"),
        [
            ("path", stats.path),
            ("size on disk", human_bytes(stats.size_bytes)),
            ("chunks", str(stats.documents)),
            ("sources", str(stats.sources)),
            (
                "embedding dimensions",
                str(stats.dimensions) if stats.dimensions else "-- (nothing indexed)",
            ),
            ("schema version", str(stats.schema_version)),
        ],
    )
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Compact the indexes and reclaim disk space."""
    printer = Printer()
    with open_engine(args) as rag:
        before = rag.stats().size_bytes
        started = time.perf_counter()
        rag.optimize()
        elapsed = time.perf_counter() - started
        after = rag.stats().size_bytes
    saved = before - after
    printer.line(
        f"Optimized in {elapsed:.2f}s: {human_bytes(before)} -> {human_bytes(after)}"
        + (f" (saved {human_bytes(saved)})" if saved > 0 else "")
    )
    return 0


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #

SHELL_HELP = """\
Type a question and press Enter to get an answer from the index.

  \\help            show this message
  \\search <query>  retrieve chunks without calling the chat model
  \\add <source>    index a file, directory or URL
  \\ls              list indexed sources
  \\stats           show index statistics
  \\quit            leave (Ctrl-D also works)
"""


def _shell_meta(rag: Rag, printer: Printer, line: str) -> bool:
    """Run one ``\\``-prefixed meta command. Returns ``False`` to quit."""
    command, _, rest = line[1:].strip().partition(" ")
    rest = rest.strip()
    command = command.lower()

    if command in ("quit", "q", "exit"):
        return False
    if command in ("help", "h", "?"):
        printer.line(SHELL_HELP)
    elif command == "search":
        if not rest:
            printer.error("usage: \\search <query>")
        else:
            hits = rag.search(rest)
            if hits:
                _print_hits(printer, hits, full=False)
            else:
                printer.line("No matching chunks.")
    elif command == "add":
        if not rest:
            printer.error("usage: \\add <file|directory|url>")
        else:
            result = rag.add(rest)
            if result.ok:
                printer.line(
                    f"{display_source(result.source)}: {result.chunks_added} chunks "
                    f"added, {result.chunks_skipped} skipped."
                )
            else:
                printer.error(f"failed: {result.error}")
    elif command == "ls":
        sources = rag.sources()
        if not sources:
            printer.line("The index is empty.")
        else:
            printer.table(
                ("SOURCE", "CHUNKS", "CHARS", "ADDED"),
                [
                    (s.source, str(s.chunks), str(s.characters), str(s.added_at)[:19])
                    for s in sources
                ],
                right_align=(1, 2),
            )
    elif command == "stats":
        stats = rag.stats()
        printer.line(
            f"{stats.path}: {stats.documents} chunks from {stats.sources} sources, "
            f"{human_bytes(stats.size_bytes)}, dim={stats.dimensions or '--'}"
        )
    else:
        printer.error(f"unknown command \\{command}. Try \\help.")
    return True


def cmd_shell(args: argparse.Namespace) -> int:
    """Run an interactive question-and-answer session."""
    printer = Printer()
    where = parse_where(args.where)

    with open_engine(args, chat=True) as rag:
        stats = rag.stats()
        printer.line(
            f"softrag {__version__} -- {stats.path} "
            f"({stats.documents} chunks, {stats.sources} sources)",
            style="bold",
        )
        printer.line(
            "Ask a question, or \\help for commands. Ctrl-D to quit.", style="dim"
        )
        printer.line()

        while True:
            try:
                line = input("softrag> ").strip()
            except EOFError:
                printer.line()
                break
            except KeyboardInterrupt:
                printer.line()
                printer.line("(interrupted -- \\quit to leave)", style="dim")
                continue

            if not line:
                continue
            try:
                if line.startswith("\\"):
                    if not _shell_meta(rag, printer, line):
                        break
                    continue
                answer = rag.query(
                    line, top_k=args.top_k, mode=args.mode, where=where, stream=True
                )
                for delta in answer:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                sys.stdout.write("\n")
                if args.show_sources:
                    _print_sources(printer, answer.hits)
                printer.line()
            except KeyboardInterrupt:
                printer.line()
                printer.line("(cancelled)", style="dim")
            except SoftragError as exc:
                printer.error(f"error: {str(exc).splitlines()[0]}")

    printer.line("bye.")
    return 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def _package_version(module: str) -> str | None:
    """Installed version of ``module``, or ``None`` when it is absent."""
    if importlib.util.find_spec(module) is None:
        return None
    from importlib.metadata import PackageNotFoundError, version

    for candidate in (module, module.replace("_", "-")):
        try:
            return version(candidate)
        except PackageNotFoundError:
            continue
    return "installed"


def _check_sqlite_vec() -> tuple[bool, str]:
    """Try to load sqlite-vec into a throwaway connection."""
    try:
        import sqlite_vec
    except ImportError:
        return False, "not installed (pip install sqlite-vec)"
    db = sqlite3.connect(":memory:")
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        version = db.execute("SELECT vec_version()").fetchone()[0]
        return True, f"loaded, vec_version {version}"
    except Exception as exc:
        return False, f"failed to load: {exc}"
    finally:
        db.close()


def _predicted_backends() -> tuple[str, str]:
    """Name the embedder and chat model auto-detection would pick.

    Determined by inspecting the environment rather than by constructing the
    backends, so ``doctor`` never downloads a model or spends an API call.
    """
    from .providers import ollama as ollama_provider

    ollama_up = ollama_provider.is_available()
    if os.getenv("OPENAI_API_KEY"):
        embedder = "OpenAIEmbedder (text-embedding-3-small)"
    elif ollama_up:
        embedder = "OllamaEmbedder (nomic-embed-text)"
    elif importlib.util.find_spec("sentence_transformers") is not None:
        embedder = "SentenceTransformerEmbedder (all-MiniLM-L6-v2)"
    else:
        embedder = "HashEmbedder (fallback -- poor retrieval quality)"

    if os.getenv("ANTHROPIC_API_KEY"):
        chat = "AnthropicChat (claude-sonnet-5)"
    elif os.getenv("OPENAI_API_KEY"):
        chat = "OpenAIChat (gpt-4.1-mini)"
    elif ollama_up:
        chat = "OllamaChat (llama3.2)"
    else:
        chat = "EchoChatModel (fallback -- echoes context, does not generate)"
    return embedder, chat


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report on the environment softrag is running in."""
    printer = Printer()
    rows: list[tuple[str, str]] = [
        ("python", f"{sys.version.split()[0]} ({sys.platform})"),
        ("softrag", __version__),
        ("sqlite", sqlite3.sqlite_version),
    ]
    vec_ok, vec_detail = _check_sqlite_vec()
    rows.append(("sqlite-vec", vec_detail))

    for module, extra in OPTIONAL_PACKAGES:
        version = _package_version(module)
        rows.append((module, version if version else f"not installed (softrag[{extra}])"))

    from .providers import ollama as ollama_provider

    rows.append(
        (
            "ollama daemon",
            "reachable" if ollama_provider.is_available() else "not reachable",
        )
    )
    # Only ever the words SET / not set -- never any part of a key.
    for name in API_KEY_VARS:
        rows.append((name, "SET" if os.getenv(name) else "not set"))

    printer.table(("CHECK", "RESULT"), [(name, value) for name, value in rows])
    printer.line()

    embedder, chat = _predicted_backends()
    if not vec_ok:
        printer.error(
            "verdict: sqlite-vec will not load, so no index can be opened at all."
        )
        return 1
    printer.line(
        f"verdict: ready -- embedder={embedder}, chat={chat}",
        style="bold green",
    )
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "add": cmd_add,
    "search": cmd_search,
    "query": cmd_query,
    "ls": cmd_ls,
    "rm": cmd_rm,
    "stats": cmd_stats,
    "optimize": cmd_optimize,
    "shell": cmd_shell,
    "doctor": cmd_doctor,
}

EPILOG = """\
examples:
  softrag add handbook.pdf ./docs https://example.com/changelog
  softrag search "refund policy" --top-k 3
  softrag query "What changed in v2?" --show-sources
  softrag ls --json | jq '.[].source'
  softrag doctor

The index is a single SQLite file (default: softrag.db, or $SOFTRAG_DB).
"""


def _engine_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Attach the flags every command shares.

    They are declared twice -- once on the top-level parser and once on each
    subcommand -- so both ``softrag --db kb.db search x`` and
    ``softrag search x --db kb.db`` work. The subcommand copies default to
    ``SUPPRESS`` so an unset flag never clobbers the top-level value.
    """
    hidden = argparse.SUPPRESS
    group = parser.add_argument_group("engine options")
    group.add_argument(
        "--db",
        metavar="PATH",
        default=hidden if suppress else None,
        help=f"index file to use (default: ${DB_ENV_VAR} or {DEFAULT_DB})",
    )
    group.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=hidden if suppress else "auto",
        help=(
            "model backend (default: auto, detected from the environment). "
            "'anthropic' supplies chat only; embeddings stay auto-detected"
        ),
    )
    group.add_argument(
        "--embed-model",
        metavar="NAME",
        default=hidden if suppress else None,
        help="embedding model name for the chosen provider",
    )
    group.add_argument(
        "--chat-model",
        metavar="NAME",
        default=hidden if suppress else None,
        help="chat model name for the chosen provider",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        default=hidden if suppress else False,
        help="verbose logging and full tracebacks on any error",
    )


def _search_flags(parser: argparse.ArgumentParser, *, full: bool) -> None:
    """Attach the retrieval flags shared by search, query and shell."""
    group = parser.add_argument_group("retrieval options")
    group.add_argument(
        "--top-k",
        type=int,
        metavar="N",
        help="number of chunks to retrieve (default: 5)",
    )
    group.add_argument(
        "--mode",
        choices=("hybrid", "vector", "keyword"),
        help="hybrid fuses vectors and BM25 (default); vector or keyword use one",
    )
    group.add_argument(
        "--where",
        metavar="JSON",
        help='metadata filter, e.g. \'{"year": {"$gte": 2024}}\'',
    )
    group.add_argument(
        "--source",
        metavar="NAME",
        help="restrict retrieval to a single indexed source",
    )
    group.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON on stdout and nothing else",
    )
    if full:
        group.add_argument(
            "--full",
            action="store_true",
            help="print whole chunks instead of a truncated preview",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete argument parser.

    Returns:
        A parser whose ``command`` attribute selects a handler in
        :data:`COMMANDS`.
    """
    parser = argparse.ArgumentParser(
        prog="softrag",
        description=(
            "The embedded RAG engine: index documents into one SQLite file and "
            "search or question them from the command line."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"softrag {__version__}",
        help="print the installed softrag version and exit",
    )
    _engine_flags(parser, suppress=False)

    common = argparse.ArgumentParser(add_help=False)
    _engine_flags(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # -- add ---------------------------------------------------------------- #
    add = subparsers.add_parser(
        "add",
        parents=[common],
        help="index files, directories, URLs or stdin",
        description=(
            "Index one or more sources. Directories are walked, URLs are fetched "
            "and stripped of boilerplate, and '-' reads text from stdin."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  softrag add notes.md ./docs https://example.com/faq\n"
            "  cat report.txt | softrag add - --name report\n"
            "  softrag add ./src --pattern '**/*.py' --metadata lang=python\n"
        ),
    )
    add.add_argument(
        "sources",
        nargs="+",
        metavar="SOURCE",
        help="file, directory, URL, or '-' for stdin",
    )
    add.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="metadata attached to every chunk; repeatable. Numbers and "
        "true/false are converted to their real types",
    )
    add.add_argument(
        "--name",
        metavar="NAME",
        help="source identifier for stdin content (default: 'stdin')",
    )
    add.add_argument(
        "--chunk-size", type=int, metavar="N", help="target chunk size in characters"
    )
    add.add_argument(
        "--chunk-overlap",
        type=int,
        metavar="N",
        help="characters repeated between consecutive chunks",
    )
    add.add_argument(
        "--pattern",
        default="**/*",
        metavar="GLOB",
        help="glob applied inside directories (default: %(default)s)",
    )
    add.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="glob to skip inside directories; repeatable. Build directories, "
        "virtualenvs and VCS metadata are excluded already",
    )
    add.add_argument(
        "--on-change",
        choices=("replace", "skip", "append"),
        default="replace",
        help="what to do when a source is already indexed with different "
        "content (default: %(default)s)",
    )
    add.add_argument(
        "--workers", type=int, metavar="N", help="threads used to index (default: 4)"
    )
    add.add_argument(
        "--quiet",
        action="store_true",
        help="hide per-file progress; the final summary is still printed",
    )

    # -- search ------------------------------------------------------------- #
    search = subparsers.add_parser(
        "search",
        parents=[common],
        help="retrieve chunks without calling a chat model",
        description=(
            "Retrieval only: whatever comes back is exactly what an answer "
            "would have been built from. No chat model is constructed."
        ),
    )
    search.add_argument("query", metavar="QUERY", help="text to search for")
    _search_flags(search, full=True)

    # -- query -------------------------------------------------------------- #
    query = subparsers.add_parser(
        "query",
        parents=[common],
        help="answer a question using the index",
        description=(
            "Retrieve context and generate an answer. The answer streams as it "
            "is produced, followed by the sources it drew on."
        ),
    )
    query.add_argument("question", metavar="QUESTION", help="the question to answer")
    _search_flags(query, full=False)
    generation = query.add_argument_group("generation options")
    generation.add_argument(
        "--no-stream",
        action="store_true",
        help="wait for the whole answer instead of streaming it",
    )
    generation.add_argument(
        "--show-sources",
        dest="show_sources",
        action="store_true",
        default=True,
        help="list the sources behind the answer (default)",
    )
    generation.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        help="print only the answer",
    )
    generation.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="prompt template containing {context} and {question}",
    )

    # -- ls ----------------------------------------------------------------- #
    listing = subparsers.add_parser(
        "ls",
        parents=[common],
        help="list the sources in an index",
        description="Show every indexed source, most recently updated first.",
    )
    listing.add_argument(
        "--json", action="store_true", help="print a JSON array on stdout"
    )
    listing.add_argument("--limit", type=int, metavar="N", help="show at most N sources")

    # -- rm ----------------------------------------------------------------- #
    remove = subparsers.add_parser(
        "rm",
        parents=[common],
        help="delete sources from an index",
        description=(
            "Delete by source identifier, by metadata filter, or everything. "
            "Identifiers are the ones shown by `softrag ls`."
        ),
    )
    remove.add_argument(
        "sources", nargs="*", metavar="SOURCE", help="source identifiers to delete"
    )
    remove.add_argument(
        "--where", metavar="JSON", help="delete every chunk matching this filter"
    )
    remove.add_argument(
        "--all", action="store_true", help="empty the index; requires --yes"
    )
    remove.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    # -- stats -------------------------------------------------------------- #
    stats = subparsers.add_parser(
        "stats",
        parents=[common],
        help="show index size, counts and schema version",
        description="Summarise what an index file holds and what it costs on disk.",
    )
    stats.add_argument(
        "--json", action="store_true", help="print a JSON object on stdout"
    )

    # -- optimize ----------------------------------------------------------- #
    subparsers.add_parser(
        "optimize",
        parents=[common],
        help="compact the indexes and reclaim disk space",
        description=(
            "Merge the full-text index segments and VACUUM the database. Worth "
            "running after deleting a lot of content."
        ),
    )

    # -- shell -------------------------------------------------------------- #
    shell = subparsers.add_parser(
        "shell",
        parents=[common],
        help="interactive question-and-answer session",
        description=(
            "An interactive prompt over one index. Type a question for an "
            "answer, or a \\-prefixed meta command such as \\search or \\ls."
        ),
    )
    shell.add_argument("--top-k", type=int, metavar="N", help="chunks to retrieve")
    shell.add_argument(
        "--mode", choices=("hybrid", "vector", "keyword"), help="search mode"
    )
    shell.add_argument("--where", metavar="JSON", help="metadata filter for questions")
    shell.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        default=True,
        help="do not list sources under each answer",
    )

    # -- doctor ------------------------------------------------------------- #
    subparsers.add_parser(
        "doctor",
        parents=[common],
        help="diagnose the environment",
        description=(
            "Check the interpreter, SQLite, sqlite-vec, optional extras, a local "
            "Ollama daemon and which API keys are present. Key values are never "
            "printed -- only whether each variable is set."
        ),
    )

    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _configure_logging(debug: bool) -> None:
    """Send library logs to stderr, leaving stdout free for real output."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("softrag")
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if debug else logging.WARNING)
    root.propagate = False


def _report(exc: Exception) -> None:
    """Print an expected failure as a clean ``error:`` line on stderr.

    Multi-line messages keep their remaining lines, indented: those lines are
    the ``pip install`` hints that make a missing-dependency error actionable.
    """
    message = str(exc).strip() or exc.__class__.__name__
    first, _, rest = message.partition("\n")
    printer = Printer()
    printer.error(f"error: {first}")
    for line in rest.splitlines():
        if line.strip():
            printer.error(f"  {line.strip()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments without the program name. Defaults to :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` for an expected failure, ``2`` for a usage
        error, ``130`` when interrupted.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors land here
        return int(exc.code) if exc.code is not None else 0

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    debug = bool(getattr(args, "debug", False))
    _configure_logging(debug)

    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except SoftragError as exc:
        if debug:
            raise
        _report(exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
