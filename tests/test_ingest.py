"""Ingestion: format extractors, dispatch by extension, and directory discovery.

Every fixture here is built at runtime under ``tmp_path`` -- including the DOCX
and PPTX archives, which are just ZIPs of XML and so can be assembled with the
standard library. No binary blobs are committed to the repository, and the
Open XML extractors are still exercised for real rather than mocked away.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from softrag.errors import ExtractionError, IngestionError, UnsupportedFormatError
from softrag.ingest import (
    DEFAULT_EXCLUDES,
    discover_files,
    extract_file,
)
from softrag.ingest.formats import (
    EXTRACTORS,
    extension_for,
    extract_csv,
    extract_docx,
    extract_html,
    extract_json,
    extract_pptx,
    extract_text,
    html_to_text,
)

# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Quarterly refund policy</w:t></w:r></w:p>
    <w:p><w:r><w:t>Refunds are issued within thirty days.</w:t></w:r></w:p>
  </w:body>
</w:document>"""

FOOTNOTES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p><w:r><w:t>Footnote zqxwv7 about the policy.</w:t></w:r></w:p>
</w:footnotes>"""

SLIDE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <a:p><a:r><a:t>{title}</a:t></a:r></a:p>
    <a:p><a:r><a:t>{body}</a:t></a:r></a:p>
  </p:spTree></p:cSld>
</p:sld>"""


def write_docx(path: Path, *, with_footnotes: bool = False) -> Path:
    """A DOCX is a ZIP whose text lives in ``word/document.xml``."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", DOCUMENT_XML)
        if with_footnotes:
            archive.writestr("word/footnotes.xml", FOOTNOTES_XML)
    return path


def write_pptx(path: Path, slides: int = 2) -> Path:
    """A PPTX is a ZIP whose text lives in ``ppt/slides/slideN.xml``."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        for number in range(1, slides + 1):
            archive.writestr(
                f"ppt/slides/slide{number}.xml",
                SLIDE_XML.format(
                    title=f"Slide title {number}", body=f"Slide body number {number}"
                ),
            )
    return path


# --------------------------------------------------------------------------- #
# Plain text and Markdown
# --------------------------------------------------------------------------- #


def test_plain_text_round_trips(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("first line\n\n\n\n  second line  ", encoding="utf-8")

    text, source, metadata = extract_file(path)

    assert text == "first line\n\nsecond line"
    assert source == str(path)
    assert metadata["extension"] == ".txt"
    assert metadata["filename"] == "note.txt"
    assert metadata["bytes"] > 0


def test_markdown_is_kept_verbatim(tmp_path):
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\n- one\n- two\n", encoding="utf-8")
    text, _, _ = extract_file(path)
    assert "# Title" in text
    assert "- one" in text


def test_latin1_bytes_are_decoded_without_raising():
    assert "café" in extract_text("café au lait".encode("cp1252"))


def test_bytes_are_accepted_with_a_name():
    text, source, metadata = extract_file(b"inline body text", name="inline.txt")
    assert text == "inline body text"
    assert source == "inline.txt"
    assert metadata["kind"] == "bytes"


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

HTML = """<!doctype html>
<html>
  <head>
    <title>Refund Policy</title>
    <style>body { color: salmon; }</style>
  </head>
  <body>
    <script>var tracker = "should never be indexed";</script>
    <h1>Refunds</h1>
    <p>Returns are accepted within thirty days.</p>
    <noscript>enable javascript</noscript>
  </body>
</html>"""


def test_html_drops_script_and_style_content(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML, encoding="utf-8")

    text, _, _ = extract_file(path)

    assert "Returns are accepted within thirty days." in text
    assert "should never be indexed" not in text
    assert "salmon" not in text
    assert "enable javascript" not in text


def test_html_to_text_captures_the_title():
    # Regression: <title> lives inside <head>, and <head> is skipped, so an
    # extractor that tests its skip depth before its title flag discards the
    # title of every real document while a stray <title> outside <head> keeps
    # working -- which is exactly what hides the bug.
    text, title = html_to_text(HTML)
    assert title == "Refund Policy"
    # The title lives in <head>, which is skipped, so it must not leak into body
    # text and be indexed twice.
    assert "Refund Policy" not in text


def test_a_title_outside_head_is_captured():
    """The half of the title handling that does work, pinned as a regression."""
    text, title = html_to_text("<html><title>Loose Title</title><body>body</body></html>")
    assert title == "Loose Title"
    assert text == "body"


def test_html_without_a_title_reports_an_empty_one():
    text, title = html_to_text("<p>bare paragraph</p>")
    assert title == ""
    assert text == "bare paragraph"


def test_block_elements_become_line_breaks():
    text = extract_html(b"<p>one</p><p>two</p>")
    assert [line for line in text.splitlines() if line] == ["one", "two"]


# --------------------------------------------------------------------------- #
# CSV / TSV
# --------------------------------------------------------------------------- #


def test_csv_is_rendered_as_column_value_pairs(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,role\nAda,engineer\nGrace,admiral\n", encoding="utf-8")

    text, _, _ = extract_file(path)

    assert text.splitlines() == [
        "name: Ada; role: engineer",
        "name: Grace; role: admiral",
    ]


def test_tsv_uses_the_tab_delimiter(tmp_path):
    path = tmp_path / "people.tsv"
    path.write_text("name\trole\nAda\tengineer\n", encoding="utf-8")
    text, _, _ = extract_file(path)
    assert text == "name: Ada; role: engineer"


def test_csv_empty_cells_are_dropped():
    assert extract_csv(b"a,b\n1,\n") == "a: 1"


def test_a_header_only_csv_still_yields_the_header():
    assert extract_csv(b"alpha,beta\n") == "alpha | beta"


def test_an_empty_csv_yields_nothing():
    assert extract_csv(b"") == ""


# --------------------------------------------------------------------------- #
# JSON / JSON Lines
# --------------------------------------------------------------------------- #


def test_json_is_flattened_to_path_value_lines(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "service": {"name": "billing", "region": "eu-west-1"},
                "replicas": [{"id": 1}, {"id": 2}],
                "enabled": True,
            }
        ),
        encoding="utf-8",
    )

    text, _, _ = extract_file(path)

    assert "service.name: billing" in text
    assert "service.region: eu-west-1" in text
    assert "replicas[0].id: 1" in text
    assert "replicas[1].id: 2" in text
    assert "enabled: True" in text


def test_json_lines_become_one_block_per_record(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "created"}\n{"event": "deleted"}\n', encoding="utf-8")

    text, _, _ = extract_file(path)

    assert text == "event: created\n\nevent: deleted"


def test_json_lines_tolerate_a_broken_line():
    text = extract_json(b'{"a": 1}\nnot json at all {{\n{"b": 2}\n')
    assert "a: 1" in text
    assert "b: 2" in text


def test_empty_json_values_are_omitted():
    text = extract_json(
        json.dumps({"kept": "yes", "blank": "", "missing": None}).encode()
    )
    assert text == "kept: yes"


def test_neither_json_nor_json_lines_is_an_extraction_error():
    with pytest.raises(ExtractionError) as excinfo:
        extract_json(b"<<< not json >>>", filename="broken.json")
    assert "JSON Lines" in str(excinfo.value)


def test_an_empty_json_document_yields_nothing():
    assert extract_json(b"   ") == ""


# --------------------------------------------------------------------------- #
# Open XML: DOCX and PPTX
# --------------------------------------------------------------------------- #


def test_docx_text_is_extracted(tmp_path):
    path = write_docx(tmp_path / "policy.docx")

    text, source, metadata = extract_file(path)

    assert "Quarterly refund policy" in text
    assert "Refunds are issued within thirty days." in text
    assert "<w:" not in text
    assert metadata["extension"] == ".docx"
    assert source == str(path)


def test_docx_paragraphs_are_kept_apart(tmp_path):
    text = extract_docx(write_docx(tmp_path / "policy.docx").read_bytes())
    lines = [line for line in text.splitlines() if line]
    assert lines[0] == "Quarterly refund policy"
    assert lines[1] == "Refunds are issued within thirty days."


def test_docx_footnotes_are_included(tmp_path):
    path = write_docx(tmp_path / "policy.docx", with_footnotes=True)
    assert "zqxwv7" in extract_docx(path.read_bytes())


def test_a_zip_without_a_word_document_is_not_a_docx(tmp_path):
    path = tmp_path / "fake.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "this is just a zip")

    with pytest.raises(ExtractionError) as excinfo:
        extract_file(path)

    message = str(excinfo.value)
    assert "word/document.xml" in message
    assert ".doc" in message  # it names the legacy format it cannot read


def test_a_file_that_is_not_a_zip_at_all_is_reported_clearly(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"definitely not a zip archive")

    with pytest.raises(ExtractionError) as excinfo:
        extract_file(path)

    assert "ZIP" in str(excinfo.value)


def test_pptx_text_is_extracted_slide_by_slide(tmp_path):
    path = write_pptx(tmp_path / "deck.pptx", slides=3)

    text, _, _ = extract_file(path)

    assert "Slide 1" in text
    assert "Slide 3" in text
    assert "Slide body number 2" in text
    assert "<a:" not in text


def test_pptx_slides_are_ordered_numerically_not_lexically(tmp_path):
    path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        for number in (10, 2, 1):
            archive.writestr(
                f"ppt/slides/slide{number}.xml",
                SLIDE_XML.format(title=f"marker{number}", body="body"),
            )

    text = extract_pptx(path.read_bytes())

    assert text.index("marker1") < text.index("marker2") < text.index("marker10")


def test_a_zip_with_no_slides_is_not_a_pptx(tmp_path):
    path = tmp_path / "empty.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")

    with pytest.raises(ExtractionError) as excinfo:
        extract_file(path)
    assert "no slides" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Dispatch and failure modes
# --------------------------------------------------------------------------- #


def test_an_unknown_extension_on_text_is_read_as_text(tmp_path):
    path = tmp_path / "notes.zzz"
    path.write_text("perfectly readable prose", encoding="utf-8")
    text, _, _ = extract_file(path)
    assert text == "perfectly readable prose"


def test_an_unknown_extension_on_binary_is_unsupported(tmp_path):
    path = tmp_path / "blob.zzz"
    path.write_bytes(bytes(range(256)) * 8)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        extract_file(path)

    message = str(excinfo.value)
    assert ".zzz" in message
    assert "EXTRACTORS" in message  # it says how to register your own


def test_a_file_with_no_extension_falls_back_to_text(tmp_path):
    path = tmp_path / "LICENSE"
    path.write_text("MIT licence body", encoding="utf-8")
    assert extract_file(path)[0] == "MIT licence body"


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".gif", ".webp"])
def test_an_image_points_at_add_image(tmp_path, suffix):
    path = tmp_path / f"photo{suffix}"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    with pytest.raises(UnsupportedFormatError) as excinfo:
        extract_file(path)

    assert "add_image" in str(excinfo.value)


def test_a_missing_file_is_an_ingestion_error(tmp_path):
    with pytest.raises(IngestionError) as excinfo:
        extract_file(tmp_path / "nowhere.txt")
    assert "File not found" in str(excinfo.value)


def test_a_directory_is_an_ingestion_error(tmp_path):
    with pytest.raises(IngestionError) as excinfo:
        extract_file(tmp_path)
    assert "add_directory" in str(excinfo.value)


def test_an_empty_file_is_an_ingestion_error(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(IngestionError) as excinfo:
        extract_file(path)
    assert "empty" in str(excinfo.value)


def test_a_file_holding_only_whitespace_is_an_extraction_error(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  \t ", encoding="utf-8")
    with pytest.raises(ExtractionError) as excinfo:
        extract_file(path)
    assert "No text could be extracted" in str(excinfo.value)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.PDF", ".pdf"),
        ("archive.tar.gz", ".gz"),
        ("Makefile", ""),
        ("", ""),
        (".gitignore", ".gitignore"),
    ],
)
def test_extension_for(filename, expected):
    assert extension_for(filename) == expected


def test_every_registered_extractor_is_callable():
    assert EXTRACTORS
    assert all(callable(fn) for fn in EXTRACTORS.values())
    assert ".py" in EXTRACTORS  # source files are indexable


# --------------------------------------------------------------------------- #
# discover_files
# --------------------------------------------------------------------------- #


@pytest.fixture
def tree(tmp_path) -> Path:
    """A directory laid out like a real project, junk directories included."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)

    (tmp_path / "README.md").write_text("readme body", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text("guide body", encoding="utf-8")
    (tmp_path / "docs" / "notes.txt").write_text("notes body", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "node_modules" / "left-pad" / "index.js").write_text(
        "x", encoding="utf-8"
    )
    (tmp_path / ".git" / "COMMIT_EDITMSG.txt").write_text("wip", encoding="utf-8")
    (tmp_path / "src" / "__pycache__" / "app.py").write_text("cached", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_discovery_finds_the_real_documents(tree):
    names = {path.name for path in discover_files(tree)}
    assert names == {"README.md", "guide.md", "notes.txt", "app.py"}


def test_discovery_skips_the_junk_directories(tree):
    found = {path.as_posix() for path in discover_files(tree)}
    assert not any("node_modules" in path for path in found)
    assert not any("/.git/" in path for path in found)
    assert not any("__pycache__" in path for path in found)


def test_discovery_skips_files_with_no_extractor(tree):
    assert not any(path.suffix == ".png" for path in discover_files(tree))


def test_discovery_honours_max_bytes(tree):
    (tree / "docs" / "huge.txt").write_text("x" * 5000, encoding="utf-8")

    small = {path.name for path in discover_files(tree, max_bytes=1000)}
    large = {path.name for path in discover_files(tree, max_bytes=100_000)}

    assert "huge.txt" not in small
    assert "huge.txt" in large


def test_discovery_honours_a_custom_pattern(tree):
    names = {path.name for path in discover_files(tree, pattern="**/*.md")}
    assert names == {"README.md", "guide.md"}


def test_discovery_honours_extra_excludes(tree):
    names = {path.name for path in discover_files(tree, exclude=("**/docs/**",))}
    assert names == {"README.md", "app.py"}


def test_discovery_without_recursion_stays_at_the_top_level(tree):
    names = {path.name for path in discover_files(tree, recursive=False)}
    assert names == {"README.md"}


def test_discovery_order_is_sorted_and_stable(tree):
    first = discover_files(tree)
    assert first == sorted(first)
    assert first == discover_files(tree)


def test_discovery_of_an_empty_directory_finds_nothing(tmp_path):
    assert discover_files(tmp_path) == []


def test_the_default_excludes_cover_the_usual_suspects():
    joined = " ".join(DEFAULT_EXCLUDES)
    for name in ("node_modules", ".git", "__pycache__", ".venv", "dist"):
        assert name in joined
