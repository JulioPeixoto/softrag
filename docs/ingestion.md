# Ingestion

Getting documents into the index: what softrag can read, how it cuts text into
chunks, what metadata travels with them, and what happens when you index the
same thing twice.

- [The `add` family](#the-add-family)
- [Supported formats](#supported-formats)
- [Chunking](#chunking)
- [Choosing a chunk size](#choosing-a-chunk-size)
- [Metadata](#metadata)
- [Re-indexing and `on_change`](#re-indexing-and-on_change)
- [Bulk ingest](#bulk-ingest)
- [Adding a format](#adding-a-format)

## The `add` family

```python
rag.add(source, *, metadata=None, name=None, chunker=None, on_change="replace")
```

`add()` infers what you gave it: a `http(s)://` string is fetched as a web page,
an existing path is read as a file, `bytes` are treated as a file (pass `name=`
so the extension picks the extractor), and anything else is indexed as raw text.

The explicit forms do exactly one thing each, and are what you want in library
code where guessing is a liability:

| Method                       | Input                       | Notes                                                    |
| ---------------------------- | --------------------------- | -------------------------------------------------------- |
| `add_text(text, ...)`        | A string                    | `name` defaults to a content hash, so the same text twice is the same source. |
| `add_file(path_or_bytes, …)` | A path, or bytes + `name`   | Extractor chosen by extension.                           |
| `add_web(url, …, timeout=30)`| A URL                       | Boilerplate stripped; title captured into metadata.      |
| `add_image(path, …)`         | An image path               | Captioned by a vision model, then indexed as that text.  |
| `add_directory(dir, …)`      | A directory                 | Walks, filters, and ingests concurrently.                |
| `add_many(sources, …)`       | An iterable of any of these | Threaded; results in input order.                        |

Passing a directory or a glob string to `add()` is an error with a message
pointing at `add_directory()` — silently guessing a recursion policy would be
worse than asking.

### What you get back

```python
result = rag.add("handbook.pdf")

result.source          # 'handbook.pdf'
result.chunks_added    # newly indexed chunks
result.chunks_skipped  # chunks already present, byte-identical
result.chunks_deleted  # old chunks removed because the content changed
result.characters      # length of the extracted text
result.error           # None on success
result.ok              # not result.error
bool(result)           # same as .ok
```

`add_directory()` and `add_many()` return a list of these, one per file, and by
default record per-file failures instead of raising on the first one. Check
them:

```python
results = rag.add_directory("./docs")
for r in results:
    if not r:
        print("FAILED", r.source, r.error)
print(sum(r.chunks_added for r in results), "chunks indexed")
```

Ignoring these is the single most common reason for "my document is not in the
index".

### Images

`add_image()` asks a vision-capable chat model to describe the image, then
indexes the description like any other document — so one query searches text and
images together. It uses the engine's chat model, and needs one that can see:
`OpenAIChat`, `AnthropicChat`, an Ollama vision model such as `llava` or
`llama3.2-vision`, or a LangChain multimodal model.

```python
rag.add_image("architecture.png", metadata={"doc": "design-review"})
rag.add_image("chart.jpg", prompt="Transcribe every axis label and data value.")
```

The stored text is `Image: <filename>` followed by the caption, and metadata
gets `kind="image"` plus the path — so `where={"kind": "image"}` restricts a
search to images.

## Supported formats

| Kind      | Extensions                                                                        | Needs                |
| --------- | --------------------------------------------------------------------------------- | -------------------- |
| Text      | `.txt` `.text` `.md` `.markdown` `.rst` `.org` `.log`                             | —                    |
| Markup    | `.html` `.htm` `.xhtml` `.xml`                                                    | —                    |
| Office    | `.docx` `.pptx` `.xlsx` `.xlsm`                                                   | —                    |
| Books     | `.epub`                                                                           | —                    |
| PDF       | `.pdf`                                                                            | `softrag[files]`     |
| Data      | `.csv` `.tsv` `.json` `.jsonl` `.ndjson`                                          | —                    |
| Code      | `.py` `.pyi` `.js` `.jsx` `.ts` `.tsx` `.java` `.kt` `.go` `.rs` `.c` `.h` `.cpp` `.hpp` `.cs` `.rb` `.php` `.swift` `.scala` `.sh` `.bash` `.zsh` `.sql` `.r` `.jl` `.lua` `.pl` `.vim` `.toml` `.yaml` `.yml` `.ini` `.cfg` `.conf` `.env` `.dockerfile` `.tf` | — |
| Images    | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.bmp` — via `add_image()`                   | a vision chat model  |
| Web       | any URL — via `add_web()`                                                         | `softrag[web]` optional |

**DOCX, PPTX, XLSX and EPUB need no third-party dependency.** They are ZIP
archives of XML, and softrag unzips and reads them with the standard library:
Word documents including footnotes and endnotes, PowerPoint one section per
slide, Excel one line per row with shared strings resolved, EPUB chapter by
chapter. Only PDF genuinely needs a parser — `pypdf` is tried first, `pymupdf`
second, so whichever you already have is used.

Some format notes worth knowing before you are surprised by them:

- **CSV/TSV** is rendered as `column: value; column: value` records rather than
  raw rows. Repeating the header on every line costs space but makes each chunk
  self-describing — a bare row of numbers matches no query.
- **JSON/JSONL** is flattened to `path: value` lines (`user.address.city: Lisbon`),
  and empty values are dropped.
- **HTML** has `script`, `style`, `svg`, `head`, `template` and `iframe` removed,
  and block elements become line breaks.
- **Web pages** use `trafilatura` when installed, which is much better at
  discarding navigation and cookie banners, and fall back to the built-in
  HTML-to-text pass otherwise. Pages rendered entirely by JavaScript yield
  nothing — softrag does not run a browser.
- **Scanned PDFs** have no text layer and raise `ExtractionError`. OCR them
  first.
- **Legacy `.doc`** is not supported; convert to `.docx`.

Unknown extensions that sniff as text are read as plain text. Everything else
raises `UnsupportedFormatError` naming the extensions that are supported.

## Chunking

A chunker is any callable `str -> list[str]`. That is the whole protocol, so a
lambda is a drop-in replacement anywhere a chunker is accepted.

```python
import softrag
from softrag import RecursiveChunker, MarkdownChunker, SentenceChunker, by_separator

rag = softrag.connect("kb.db", chunker=MarkdownChunker(chunk_size=800))  # engine default
rag.add("notes.md", chunker=SentenceChunker())                           # this document only
rag.add("records.txt", chunker="\n---\n")                                # a literal separator
rag.add("weird.txt", chunker=lambda t: t.split("\n\n"))                  # anything callable
```

### `RecursiveChunker` — the default

Splits on progressively finer separators until the pieces fit, preferring
section breaks over paragraphs over lines over sentences over words, and only
slicing mid-word when a single unbroken run exceeds `chunk_size`. Then it packs
the pieces back up to `chunk_size` with `chunk_overlap` characters carried over.

```python
RecursiveChunker(chunk_size=1000, chunk_overlap=200)
```

Overlap exists so a fact spanning a boundary survives intact in at least one
chunk. It must be smaller than `chunk_size` (otherwise chunks would never
advance, and the constructor says so).

`length` takes a size function, so you can chunk by tokens instead of
characters:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
RecursiveChunker(chunk_size=256, chunk_overlap=32, length=lambda t: len(enc.encode(t)))
```

### `MarkdownChunker`

Splits along heading structure, keeps sections whole when they fit, and prefixes
every chunk with its heading breadcrumb (`# Guide > ## Install`). That prefix is
the point: an isolated chunk that still says what it is about retrieves
measurably better with both BM25 and vectors. Falls back to recursive splitting
for documents with no headings.

```python
MarkdownChunker(chunk_size=1000, chunk_overlap=200, include_heading_trail=True)
```

### `SentenceChunker`

Groups whole sentences and overlaps by whole sentences. Use it when chunks are
shown to a human — citations, snippets, quoted context — where a chunk ending
mid-sentence looks broken.

```python
SentenceChunker(chunk_size=1000, overlap_sentences=1)
```

### `by_separator`

For structured text with an explicit record delimiter — log entries, exported
transcripts, one-record-per-block dumps.

```python
by_separator("\n=== ")     # equivalent to passing the string directly
```

### Custom

```python
def chunk_by_function(source: str) -> list[str]:
    """One chunk per top-level Python definition."""
    import re
    parts = re.split(r"\n(?=(?:def |class |@))", source)
    return [p for p in parts if p.strip()]

rag.add("module.py", chunker=chunk_by_function)
```

Empty and whitespace-only chunks are dropped by the engine, so a chunker need
not filter them itself.

## Choosing a chunk size

There is no universally right answer, but the trade-off is stable: a chunk is
the unit of both retrieval and dilution. Large chunks carry more surrounding
context but their embedding averages over more topics, so they match nothing
specifically. Small chunks embed sharply but arrive at the model without enough
context to be useful.

| Content                                   | Chunk size      | Overlap    | Chunker             |
| ----------------------------------------- | --------------- | ---------- | ------------------- |
| FAQs, short answers, product entries      | 300 – 500       | 50         | `RecursiveChunker`  |
| Prose docs, handbooks, articles           | 800 – 1200      | 150 – 200  | `MarkdownChunker`   |
| Technical reference, API docs             | 500 – 800       | 100        | `MarkdownChunker`   |
| Legal, policies, contracts                | 1000 – 1500     | 200 – 300  | `SentenceChunker`   |
| Source code                               | 1000 – 2000     | 100        | custom, by symbol   |
| Log lines, records, transcripts           | one per record  | 0          | `by_separator`      |

Rules of thumb:

- Start at the 1000/200 default and only move when `search()` shows a problem.
- Chunks above ~2000 characters usually dilute; below ~200 usually lack context.
- Overlap of 10–20% of `chunk_size` is a reasonable default; more is mostly
  wasted index.
- If chunks look right but hits are too narrow, prefer `expand_context=1` at
  query time over raising `chunk_size` — it gives the model more text without
  blurring the embeddings.

Chunk size is not baked into the file: change it and re-ingest, and the changed
content replaces the old chunks. (Unlike embedding *width*, which is —
see [FAQ](faq.md#what-happens-if-i-change-embedding-models).)

## Metadata

Anything you pass as `metadata=` is attached to every chunk of that document and
is filterable at query time.

```python
rag.add("handbook.pdf", metadata={"team": "legal", "year": 2025, "tags": ["policy"]})
rag.search("holiday", where={"team": "legal", "year": {"$gte": 2024}})
```

Values must be JSON-serialisable. Nested objects work and are reachable with
dotted paths (`{"doc.author.name": "Ada"}`).

Ingestion adds its own metadata, which is available to filters for free:

| Source        | Automatic metadata                                             |
| ------------- | -------------------------------------------------------------- |
| `add_file`    | `kind="file"`, `filename`, `extension`, `bytes`, `path`         |
| bytes         | `kind="bytes"`, `bytes`                                         |
| `add_web`     | `kind="web"`, `url`, `title` (when the page has one)            |
| `add_image`   | `kind="image"`, `path`, `filename`                              |

Your keys win on collision — `metadata={"kind": "manual"}` overrides the
automatic `kind`.

```python
# "Search only the Markdown files, only in this year's docs."
rag.search("onboarding", where={"extension": ".md", "year": 2025})
```

## Re-indexing and `on_change`

Sources are identified by their `source` string — a path, a URL, or whatever you
passed as `name=`. softrag stores a content hash per source, so:

**Unchanged content is free.** Re-adding a source whose text hashes identically
does nothing at all: no extraction, no embedding calls, no writes. The result
reports the existing chunks as `chunks_skipped`.

```python
rag.add("handbook.pdf")   # <IngestResult 'handbook.pdf' added=42 skipped=0 deleted=0>
rag.add("handbook.pdf")   # <IngestResult 'handbook.pdf' added=0 skipped=42 deleted=0>
```

This is what makes `rag.add_directory("./docs")` safe to run on every startup:
only what actually changed costs anything.

**Changed content is governed by `on_change`:**

| `on_change`  | Behaviour                                                                 |
| ------------ | ------------------------------------------------------------------------- |
| `"replace"`  | Default. Delete the old chunks, index the new ones. The index reflects the current document. |
| `"skip"`     | Leave the old version alone. Useful for pinned snapshots and cheap crawls. |
| `"append"`   | Keep both, numbering the new chunks after the old. Useful for append-only sources such as logs. |

```python
rag.add("handbook.pdf")                       # v1 indexed:  added=42
# ... the file is edited ...
rag.add("handbook.pdf")                       # added=39 deleted=42
rag.add("handbook.pdf", on_change="skip")     # added=0 deleted=0 — old version kept
rag.add("changelog.md", on_change="append")   # new chunks appended after the old
```

There is a second layer of deduplication below this: a chunk is unique on
`(source, hash)`, so an identical paragraph appearing twice in one document is
stored once, while the same paragraph in two different files is stored under
each — deleting one source can never blank the other.

To take a source out:

```python
rag.delete(source="handbook.pdf")             # returns the number of chunks removed
rag.delete(where={"year": {"$lt": 2020}})     # or by filter
```

## Bulk ingest

```python
results = rag.add_directory(
    "./knowledge",
    pattern="**/*.md",                     # glob, relative to the directory
    exclude=("**/drafts/**", "**/*.tmp"),  # added to the built-in exclusions
    recursive=True,
    metadata={"corpus": "internal"},
    on_progress=lambda src, done, total: print(f"[{done}/{total}] {src}"),
    ignore_errors=True,
)
```

Directory walks skip what nobody wants in a knowledge base without being asked:
`.git`, `.hg`, `.svn`, `node_modules`, `__pycache__`, `.venv`/`venv`, `.tox`,
`.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `dist`, `build`, `target`,
`.next`, `.idea` and `*.egg-info`. Files with no registered extractor are
skipped, as are files above 32 MB — a 200 MB log is not a document.

`add_many()` is the same machinery over an arbitrary iterable, and mixes kinds
freely:

```python
results = rag.add_many(
    ["handbook.pdf", "https://example.com/changelog", "./notes.md"],
    max_workers=8,
    metadata={"batch": "2025-09"},
)
```

Extraction and embedding run in a thread pool, because both are dominated by
I/O — disk reads and API calls. Database writes are serialised by the store, so
concurrency never corrupts the index. `max_workers` defaults to 4; raise it for
network-bound work (URLs, hosted embedding APIs), leave it low for
CPU-bound local models where threads only contend.

Results come back in input order regardless of completion order, so
`zip(sources, results)` is valid.

## Adding a format

`EXTRACTORS` maps a lowercase extension to a callable
`(bytes, *, filename: str) -> str`. Registering one teaches every `add*` method
and the directory walker at once.

```python
from softrag.ingest import EXTRACTORS
from striprtf.striprtf import rtf_to_text

def extract_rtf(data: bytes, *, filename: str = "") -> str:
    return rtf_to_text(data.decode("utf-8", errors="replace"))

EXTRACTORS[".rtf"] = extract_rtf

rag.add("memo.rtf")          # just works, and so does add_directory
```

Raise `softrag.ExtractionError` for a file your extractor cannot handle; the
engine turns it into a failed `IngestResult` rather than aborting a whole
directory walk.
