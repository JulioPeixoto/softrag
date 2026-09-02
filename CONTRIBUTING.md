# Contributing to softrag

Thanks for helping. This document covers how to get set up, what the code
expects of a change, and how to extend the two things people most often want to
extend: file formats and model backends.

## Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JulioPeixoto/softrag
cd softrag

uv venv
uv pip install -e . pytest pytest-cov ruff mypy
```

Then, with the environment active (`source .venv/bin/activate`, or
`.venv\Scripts\activate` on Windows):

```bash
pytest                                   # the suite: fast, offline, no API keys
pytest --cov=softrag --cov-report=term-missing
ruff check src tests benchmarks          # lint
ruff format src tests benchmarks         # format
mypy src/softrag                         # types (not yet strict-clean)
```

The whole suite runs in seconds and needs no network, no API key and no model
download. Keep it that way: a test that reaches the network is a test that will
one day fail for a reason unrelated to your change. Use `softrag.HashEmbedder`
and `softrag.EchoChatModel`, or a fake of your own.

## Project layout

```
src/softrag/
  engine.py      Rag: ingestion, retrieval and generation policy
  aengine.py     AsyncRag, the async mirror of Rag
  store.py       every SQL statement, the schema and its migrations
  retrieval.py   rank fusion, MMR, context expansion
  chunking.py    the splitters
  filters.py     the metadata filter DSL, compiled to parameterised SQL
  stopwords.py   terms dropped when building a keyword query
  rerank.py      second-stage rerankers
  eval.py        retrieval metrics
  transforms.py  HyDE, multi-query, contextual retrieval
  cli.py         the command line interface
  types.py       dataclasses and backend protocols
  errors.py      the exception hierarchy
  providers/     OpenAI, Anthropic, Ollama, sentence-transformers, adapters
  ingest/        extraction: formats.py has the per-format extractors
benchmarks/      throughput, latency and retrieval-quality measurement
docs/            the guides
```

One rule worth stating: **only `store.py` writes SQL.** Everything above it
asks the store for what it needs. If a change has you writing a query outside
that module, the store is missing a method.

## What a good change looks like

- **Reproduce before you fix.** For a bug, the ideal first commit is a failing
  test. Several fixes in 0.2.0 came from exactly that, and the test is what
  keeps them fixed.
- **Measure before you optimise.** `benchmarks/bench.py` exists so performance
  claims can be checked. The filtered-search fix in 0.2.0 looked obviously
  right and was three times slower than what it replaced; the benchmark is what
  caught it.
- **Say why in comments, not what.** The code says what it does. A comment
  earns its place by explaining a decision that is not obvious — a measured
  threshold, a workaround for a library's behaviour, a tradeoff taken
  deliberately.
- Google-style docstrings with `Args:`, `Returns:` and `Raises:` on anything
  public, and an `Example:` where it helps.
- Type annotations everywhere. `from __future__ import annotations` at the top.
- Errors come from `softrag.errors` and say how to fix the problem, not just
  what went wrong.
- No `print()` in library code. Use `logging.getLogger("softrag.<module>")`.
  The CLI prints; nothing else does.

## Adding a file format

Extractors live in `src/softrag/ingest/formats.py`. One is a function taking
bytes and returning text:

```python
def extract_rtf(data: bytes, *, filename: str = "") -> str:
    """Rich Text Format documents."""
    ...
```

Register it:

```python
EXTRACTORS[".rtf"] = extract_rtf
```

Users can do the same at runtime without touching the library:

```python
from softrag.ingest import EXTRACTORS
EXTRACTORS[".rtf"] = my_extractor
```

Prefer the standard library where it can do the job — DOCX, PPTX, XLSX and EPUB
are all handled with `zipfile` and a little XML, which is why they need no
dependency. If a format genuinely needs a parser, import it inside the function
and raise `MissingDependencyError` naming the extra to install.

## Adding a model backend

Nothing needs to subclass anything. A backend satisfies `Embedder` if it has
`embed_query` and `embed_documents`, and `ChatModel` if it has `complete`:

```python
class MyEmbedder:
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

rag = Rag(embed_model=MyEmbedder())
```

`providers/adapt_embedder` also accepts objects with `encode`, Chroma-style
callables, and plain functions, so most existing objects work unchanged. Add a
`stream()` method for incremental generation and a `describe_image()` method to
support `add_image`.

To ship a backend in the library, put it in `src/softrag/providers/`, import
the SDK inside `__init__` rather than at module level, raise
`MissingDependencyError` when it is absent, and add the extra to
`pyproject.toml`.

## Changing retrieval

Retrieval changes need evidence, not argument. Run:

```bash
python benchmarks/retrieval_quality.py                    # instant
python benchmarks/retrieval_quality.py --embedder local   # real embeddings
```

and include the before-and-after numbers in the pull request. A change that
helps one query group and quietly hurts another is not an improvement, which is
why that script reports lexical, semantic and mixed queries separately.

## Commits and pull requests

Conventional commits: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `refactor:`,
`style:`, `chore:`. The body matters more than the subject — say what was
wrong, what evidence you have, and what you decided. Keep pull requests
focused; several small ones beat one large one.

Before opening a PR:

```bash
pytest && ruff check src tests benchmarks && ruff format --check src tests benchmarks
```

CI runs the suite on Python 3.10 through 3.13 on Linux, plus 3.12 on macOS and
Windows.

## Reporting bugs

Include the softrag version, Python version, operating system, and the output
of `softrag doctor`. A short snippet that reproduces the problem is worth more
than a description of it.
