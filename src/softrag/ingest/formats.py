"""Format extractors.

Each extractor turns bytes into plain text. Wherever the standard library can
do the job -- and for the Open XML and EPUB families it can, since they are ZIP
archives of XML -- softrag does it without a third-party dependency. Formats
that genuinely need a parser (PDF above all) degrade to a clear error naming the
extra to install.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from html import unescape
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional

from ..errors import ExtractionError, MissingDependencyError

log = logging.getLogger("softrag.ingest")

__all__ = [
    "html_to_text",
    "extract_text",
    "extract_html",
    "extract_pdf",
    "extract_docx",
    "extract_pptx",
    "extract_xlsx",
    "extract_epub",
    "extract_csv",
    "extract_json",
    "EXTRACTORS",
    "extension_for",
]

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    """Collapse runaway whitespace without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _decode(data: bytes) -> str:
    """Decode bytes, trying the encodings that actually show up in the wild."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML to readable text, dropping non-content elements."""

    SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
    BLOCK = {
        "p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3",
        "h4", "h5", "h6", "blockquote", "pre", "table", "header", "footer",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title: str = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return _clean("".join(self.parts))


def html_to_text(html: str) -> tuple[str, str]:
    """Convert an HTML document to text.

    Args:
        html: The document source.

    Returns:
        ``(text, title)``. The title is ``""`` when the document has none.
    """
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - malformed markup
        log.debug("HTML parsing degraded to tag stripping: %s", exc)
        stripped = re.sub(r"<[^>]+>", " ", html)
        return _clean(unescape(stripped)), ""
    return parser.text(), parser.title


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #


def extract_text(data: bytes, *, filename: str = "") -> str:
    """Plain text, Markdown, source code and anything else already readable."""
    return _clean(_decode(data))


def extract_html(data: bytes, *, filename: str = "") -> str:
    """HTML documents."""
    text, _ = html_to_text(_decode(data))
    return text


def extract_pdf(data: bytes, *, filename: str = "") -> str:
    """PDF documents, page by page.

    Tries ``pypdf`` first and ``pymupdf`` second, so whichever the user already
    has installed is used.
    """
    try:
        import pypdf
    except ImportError:
        pypdf = None  # type: ignore[assignment]

    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = _clean("\n\n".join(pages))
            if text:
                return text
            log.debug("pypdf found no text layer in %s, trying pymupdf", filename)
        except Exception as exc:
            log.debug("pypdf failed on %s (%s), trying pymupdf", filename, exc)

    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[import-not-found, no-redef]
        except ImportError:
            pymupdf = None  # type: ignore[assignment]

    if pymupdf is not None:
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                pages = [page.get_text() for page in document]
            return _clean("\n\n".join(pages))
        except Exception as exc:
            raise ExtractionError(f"Could not read PDF {filename or '<bytes>'}: {exc}") from exc

    raise MissingDependencyError("pypdf", extra="files", feature="Reading PDF files")


# -- Open XML (DOCX / PPTX / XLSX) and EPUB: ZIP archives of XML ------------- #

_XML_TAG = re.compile(r"<[^>]+>")
_DOCX_BREAK = re.compile(r"</w:p>|<w:br[^>]*/>|<w:tab[^>]*/>", re.IGNORECASE)
_PPTX_BREAK = re.compile(r"</a:p>|<a:br[^>]*/>", re.IGNORECASE)


def _xml_to_text(xml: str, breaks: re.Pattern[str]) -> str:
    """Turn Open XML markup into text by honouring its paragraph boundaries."""
    xml = breaks.sub("\n", xml)
    return _clean(unescape(_XML_TAG.sub("", xml)))


def _open_zip(data: bytes, kind: str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtractionError(
            f"This does not look like a valid {kind} file (it is not a readable "
            f"ZIP archive): {exc}"
        ) from exc


def extract_docx(data: bytes, *, filename: str = "") -> str:
    """Word documents, including headers, footnotes and table cells."""
    with _open_zip(data, "DOCX") as archive:
        names = set(archive.namelist())
        if "word/document.xml" not in names:
            raise ExtractionError(
                f"{filename or 'This file'} is a ZIP but has no word/document.xml, "
                "so it is not a DOCX. Legacy .doc files are not supported; convert "
                "them to .docx first."
            )
        parts = [_xml_to_text(archive.read("word/document.xml").decode("utf-8"), _DOCX_BREAK)]
        for extra in ("word/footnotes.xml", "word/endnotes.xml"):
            if extra in names:
                text = _xml_to_text(archive.read(extra).decode("utf-8"), _DOCX_BREAK)
                if text:
                    parts.append(text)
    return _clean("\n\n".join(p for p in parts if p))


def extract_pptx(data: bytes, *, filename: str = "") -> str:
    """PowerPoint decks, one section of text per slide."""
    with _open_zip(data, "PPTX") as archive:
        slides = sorted(
            (n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.findall(r"\d+", n)[-1]),
        )
        if not slides:
            raise ExtractionError(f"{filename or 'This file'} contains no slides.")
        blocks = []
        for number, name in enumerate(slides, start=1):
            text = _xml_to_text(archive.read(name).decode("utf-8"), _PPTX_BREAK)
            if text:
                blocks.append(f"Slide {number}\n{text}")
    return _clean("\n\n".join(blocks))


def extract_xlsx(data: bytes, *, filename: str = "") -> str:
    """Spreadsheets, rendered as one line per row.

    Values are resolved through the shared-string table, which is where Excel
    keeps most cell text.
    """
    with _open_zip(data, "XLSX") as archive:
        names = set(archive.namelist())
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            raw = archive.read("xl/sharedStrings.xml").decode("utf-8")
            shared = [
                _clean(unescape(_XML_TAG.sub("", item)))
                for item in re.findall(r"<si>(.*?)</si>", raw, re.DOTALL)
            ]

        sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not sheets:
            raise ExtractionError(f"{filename or 'This file'} contains no worksheets.")

        blocks: List[str] = []
        for name in sheets:
            raw = archive.read(name).decode("utf-8")
            lines: List[str] = []
            for row in re.findall(r"<row[^>]*>(.*?)</row>", raw, re.DOTALL):
                cells: List[str] = []
                for cell in re.findall(r"<c[^>]*>.*?</c>|<c[^>]*/>", row, re.DOTALL):
                    value_match = re.search(r"<v>(.*?)</v>", cell, re.DOTALL)
                    if not value_match:
                        inline = re.search(r"<is>(.*?)</is>", cell, re.DOTALL)
                        cells.append(
                            _clean(unescape(_XML_TAG.sub("", inline.group(1))))
                            if inline
                            else ""
                        )
                        continue
                    value = unescape(value_match.group(1))
                    if 't="s"' in cell:
                        index = int(value)
                        value = shared[index] if index < len(shared) else ""
                    cells.append(value)
                if any(cell.strip() for cell in cells):
                    lines.append(" | ".join(cells).strip())
            if lines:
                blocks.append("\n".join(lines))
    return _clean("\n\n".join(blocks))


def extract_epub(data: bytes, *, filename: str = "") -> str:
    """EPUB books, in spine order where the manifest allows it."""
    with _open_zip(data, "EPUB") as archive:
        names = archive.namelist()
        documents = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))]
        if not documents:
            raise ExtractionError(f"{filename or 'This file'} contains no readable chapters.")
        documents.sort()
        chapters = []
        for name in documents:
            text, _ = html_to_text(_decode(archive.read(name)))
            if text:
                chapters.append(text)
    return _clean("\n\n".join(chapters))


def extract_csv(data: bytes, *, filename: str = "") -> str:
    """Delimited data, rendered as ``column: value`` records.

    Repeating the header on every row costs a little space but makes each chunk
    self-describing, which is what retrieval needs -- a bare row of numbers
    matches nothing.
    """
    text = _decode(data)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # type: ignore[assignment]

    reader = csv.reader(io.StringIO(text), dialect)
    try:
        header = next(reader)
    except StopIteration:
        return ""

    lines: List[str] = []
    for row in reader:
        pairs = [
            f"{name.strip()}: {value.strip()}"
            for name, value in zip(header, row)
            if value and value.strip()
        ]
        if pairs:
            lines.append("; ".join(pairs))
    if not lines:
        return _clean(" | ".join(header))
    return _clean("\n".join(lines))


def extract_json(data: bytes, *, filename: str = "") -> str:
    """JSON and JSON Lines, flattened to ``path: value`` lines."""
    text = _decode(data).strip()
    if not text:
        return ""

    documents: List[object] = []
    try:
        documents = [json.loads(text)]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not documents:
            raise ExtractionError(
                f"{filename or 'This file'} is neither valid JSON nor JSON Lines."
            )

    blocks = ["\n".join(_flatten(doc)) for doc in documents]
    return _clean("\n\n".join(b for b in blocks if b))


def _flatten(value: object, prefix: str = "") -> List[str]:
    """Render nested JSON as flat, readable ``path: value`` lines."""
    if isinstance(value, dict):
        lines: List[str] = []
        for key, item in value.items():
            lines.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for index, item in enumerate(value):
            lines.extend(_flatten(item, f"{prefix}[{index}]"))
        return lines
    if value is None or value == "":
        return []
    return [f"{prefix}: {value}" if prefix else str(value)]


#: Extension to extractor. Extend it to teach softrag a new format::
#:
#:     from softrag.ingest.formats import EXTRACTORS
#:     EXTRACTORS[".rtf"] = my_rtf_extractor
EXTRACTORS: Dict[str, Callable[..., str]] = {
    ".txt": extract_text,
    ".text": extract_text,
    ".md": extract_text,
    ".markdown": extract_text,
    ".rst": extract_text,
    ".org": extract_text,
    ".log": extract_text,
    ".html": extract_html,
    ".htm": extract_html,
    ".xhtml": extract_html,
    ".xml": extract_html,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".pptx": extract_pptx,
    ".xlsx": extract_xlsx,
    ".xlsm": extract_xlsx,
    ".epub": extract_epub,
    ".csv": extract_csv,
    ".tsv": extract_csv,
    ".json": extract_json,
    ".jsonl": extract_json,
    ".ndjson": extract_json,
}

#: Source files are read as text; listed explicitly so directory walks can tell
#: code apart from binaries without sniffing every file.
CODE_EXTENSIONS = (
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh",
    ".bash", ".zsh", ".sql", ".r", ".jl", ".lua", ".pl", ".vim", ".toml",
    ".yaml", ".yml", ".ini", ".cfg", ".conf", ".env", ".dockerfile", ".tf",
)
for _extension in CODE_EXTENSIONS:
    EXTRACTORS.setdefault(_extension, extract_text)


def extension_for(filename: str) -> str:
    """Lowercased extension of ``filename``, including the dot."""
    _, _, suffix = filename.rpartition(".")
    return f".{suffix.lower()}" if suffix and suffix != filename else ""
