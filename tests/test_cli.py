"""Tests for the ``softrag`` command line.

Every test drives :func:`softrag.cli.main` directly, so exit codes and stdout
are exercised exactly as a shell would see them. Model auto-detection is
monkeypatched to the dependency-free backends, which keeps the suite offline,
deterministic and free of API keys.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from softrag import providers
from softrag.cli import coerce_value, human_bytes, main, parse_metadata, parse_where
from softrag.errors import ConfigurationError


@pytest.fixture(autouse=True)
def offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every auto-detected model to a local, deterministic stand-in."""
    monkeypatch.setattr(
        providers, "auto_embedder", lambda **_: providers.HashEmbedder(dimensions=64)
    )
    monkeypatch.setattr(
        providers, "auto_chat_model", lambda **_: providers.EchoChatModel()
    )
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def db(tmp_path: Path) -> str:
    """Path to a fresh index file."""
    return str(tmp_path / "index.db")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """A small corpus with distinguishable content."""
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "refunds.md").write_text(
        "# Refund policy\n\nRefunds are issued within 30 days of purchase. "
        "Contact support to start a refund request.\n",
        encoding="utf-8",
    )
    (directory / "shipping.md").write_text(
        "# Shipping\n\nOrders ship within two business days by courier.\n",
        encoding="utf-8",
    )
    return directory


def run(*argv: str) -> int:
    """Invoke the CLI the way a shell would."""
    return main(list(argv))


def stdout_json(capsys: pytest.CaptureFixture[str]) -> Any:
    """Parse captured stdout as JSON, asserting nothing else was printed."""
    captured = capsys.readouterr()
    return json.loads(captured.out)


def shows(haystack: str, needle: str) -> bool:
    """Whether ``needle`` appears in ``haystack``, ignoring line wrapping.

    Rich wraps and folds long values -- a temp path in a table cell is routinely
    broken across lines -- so whitespace is squeezed out of both sides before
    comparing. Without this, these assertions would pass or fail depending on
    how long ``tmp_path`` happens to be.
    """
    return "".join(needle.split()) in "".join(haystack.split())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_coerce_value_types() -> None:
    assert coerce_value("2024") == 2024
    assert coerce_value("1.5") == pytest.approx(1.5)
    assert coerce_value("true") is True
    assert coerce_value("False") is False
    assert coerce_value("platform") == "platform"


def test_parse_metadata_builds_typed_mapping() -> None:
    assert parse_metadata(["team=platform", "year=2024", "draft=false"]) == {
        "team": "platform",
        "year": 2024,
        "draft": False,
    }


def test_parse_metadata_rejects_missing_equals() -> None:
    with pytest.raises(ConfigurationError):
        parse_metadata(["nope"])


def test_parse_where_requires_an_object() -> None:
    assert parse_where('{"year": 2024}') == {"year": 2024}
    assert parse_where(None) is None
    with pytest.raises(ConfigurationError):
        parse_where("[1, 2]")
    with pytest.raises(ConfigurationError):
        parse_where("{not json}")


def test_human_bytes() -> None:
    assert human_bytes(512) == "512 B"
    assert human_bytes(2048) == "2.0 KB"


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #


def test_no_arguments_prints_help_and_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run() == 0
    out = capsys.readouterr().out
    assert "usage: softrag" in out
    assert "doctor" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--version") == 0
    assert capsys.readouterr().out.startswith("softrag ")


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--help") == 0
    out = capsys.readouterr().out
    for command in (
        "add",
        "search",
        "query",
        "ls",
        "rm",
        "stats",
        "optimize",
        "shell",
        "doctor",
    ):
        assert command in out


@pytest.mark.parametrize(
    "command", ["add", "search", "query", "ls", "rm", "stats", "optimize", "doctor"]
)
def test_subcommand_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(command, "--help") == 0
    assert f"usage: softrag {command}" in capsys.readouterr().out


def test_unknown_command_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("frobnicate") == 2
    assert "invalid choice" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# add / search / ls / stats / rm round trip
# --------------------------------------------------------------------------- #


def test_round_trip(db: str, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("add", str(docs), "--db", db) == 0
    added = capsys.readouterr().out
    assert "Indexed 2 files" in added
    assert "chunks added" in added

    assert run("search", "refund policy", "--db", db) == 0
    found = capsys.readouterr().out
    assert shows(found, "refunds.md")

    assert run("ls", "--db", db) == 0
    listed = capsys.readouterr().out
    assert shows(listed, "refunds.md") and shows(listed, "shipping.md")
    assert "2 sources" in listed

    assert run("stats", "--db", db) == 0
    stats = capsys.readouterr().out
    assert shows(stats, "schema version")
    assert "64" in stats  # HashEmbedder dimensions

    assert run("rm", str(docs / "refunds.md"), "--db", db) == 0
    assert "Removed" in capsys.readouterr().out

    assert run("ls", "--db", db) == 0
    remaining = capsys.readouterr().out
    assert not shows(remaining, "refunds.md")
    assert shows(remaining, "shipping.md")


def test_add_single_file_and_metadata(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        "add",
        str(docs / "refunds.md"),
        "--db",
        db,
        "--metadata",
        "team=support",
        "--metadata",
        "year=2024",
    )
    assert code == 0
    capsys.readouterr()

    assert run("ls", "--db", db, "--json") == 0
    rows = stdout_json(capsys)
    assert rows[0]["metadata"]["team"] == "support"
    assert rows[0]["metadata"]["year"] == 2024


def test_add_reads_stdin(
    db: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    monkeypatch.setattr(
        "sys.stdin", io.StringIO("Piped knowledge about capybaras and rivers.")
    )
    assert run("add", "-", "--db", db, "--name", "piped") == 0
    capsys.readouterr()

    assert run("ls", "--db", db, "--json") == 0
    rows = stdout_json(capsys)
    assert rows[0]["source"] == "piped"


def test_add_respects_pattern_and_exclude(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--pattern", "**/shipping.md") == 0
    capsys.readouterr()
    assert run("ls", "--db", db, "--json") == 0
    rows = stdout_json(capsys)
    assert len(rows) == 1 and "shipping.md" in rows[0]["source"]


def test_add_quiet_still_prints_the_summary(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    out = capsys.readouterr().out
    assert out.strip().startswith("Indexed 2 files")


def test_add_twice_skips_unchanged_content(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db) == 0
    capsys.readouterr()
    assert run("add", str(docs), "--db", db) == 0
    out = capsys.readouterr().out
    assert "0 chunks added" in out


def test_db_environment_variable(
    db: str,
    docs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SOFTRAG_DB", db)
    assert run("add", str(docs / "shipping.md")) == 0
    capsys.readouterr()
    assert run("stats", "--json") == 0
    assert stdout_json(capsys)["path"] == db


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #


def test_search_json_is_pure_json(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()

    assert run("search", "refunds", "--db", db, "--json", "--top-k", "2") == 0
    payload = stdout_json(capsys)
    assert payload["query"] == "refunds"
    assert payload["count"] == len(payload["results"]) <= 2
    first = payload["results"][0]
    assert {"rank", "score", "source", "text", "metadata"} <= set(first)


def test_ls_and_stats_json(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()

    assert run("ls", "--db", db, "--json") == 0
    rows = stdout_json(capsys)
    assert len(rows) == 2
    assert {"source", "chunks", "characters", "added_at"} <= set(rows[0])

    assert run("stats", "--db", db, "--json") == 0
    stats = stdout_json(capsys)
    assert stats["sources"] == 2
    assert stats["dimensions"] == 64
    assert stats["schema_version"] == 1


def test_ls_limit(db: str, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    assert run("ls", "--db", db, "--json", "--limit", "1") == 0
    assert len(stdout_json(capsys)) == 1


def test_search_full_prints_whole_chunk(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs / "refunds.md"), "--db", db, "--quiet") == 0
    capsys.readouterr()
    assert run("search", "refund", "--db", db, "--full") == 0
    assert shows(capsys.readouterr().out, "Contact support")


def test_search_on_empty_index_is_not_an_error(
    db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("search", "anything", "--db", db) == 0
    assert shows(capsys.readouterr().out, "No matching chunks")


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def test_where_filters_search(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run("add", str(docs / "refunds.md"), "--db", db, "--metadata", "year=2024") == 0
    )
    assert (
        run("add", str(docs / "shipping.md"), "--db", db, "--metadata", "year=2019") == 0
    )
    capsys.readouterr()

    assert run("search", "policy", "--db", db, "--json", "--where", '{"year": 2024}') == 0
    payload = stdout_json(capsys)
    assert payload["count"] >= 1
    assert all("refunds.md" in hit["source"] for hit in payload["results"])

    assert (
        run(
            "search",
            "policy",
            "--db",
            db,
            "--json",
            "--where",
            '{"year": {"$gte": 2020}}',
        )
        == 0
    )
    assert all("refunds.md" in hit["source"] for hit in stdout_json(capsys)["results"])


def test_source_flag_restricts_search(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    target = str(docs / "shipping.md")
    assert run("search", "policy", "--db", db, "--json", "--source", target) == 0
    payload = stdout_json(capsys)
    assert all(hit["source"] == target for hit in payload["results"])


def test_bad_where_is_a_clean_error(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("search", "x", "--db", db, "--where", "[1]") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_rm_with_where(db: str, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        run("add", str(docs / "refunds.md"), "--db", db, "--metadata", "year=2024") == 0
    )
    assert (
        run("add", str(docs / "shipping.md"), "--db", db, "--metadata", "year=2019") == 0
    )
    capsys.readouterr()

    assert run("rm", "--db", db, "--where", '{"year": 2019}') == 0
    assert "Removed" in capsys.readouterr().out

    assert run("search", "shipping courier", "--db", db, "--json") == 0
    assert all("shipping" not in hit["source"] for hit in stdout_json(capsys)["results"])


# --------------------------------------------------------------------------- #
# rm safety
# --------------------------------------------------------------------------- #


def test_rm_without_arguments_fails_cleanly(
    db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("rm", "--db", db) == 1
    assert shows(capsys.readouterr().err, "Nothing to remove")


def test_rm_all_requires_yes(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()

    # stdin is not a terminal under pytest, so the CLI must refuse rather than
    # block on a prompt nobody can answer.
    assert run("rm", "--all", "--db", db) == 1
    assert shows(capsys.readouterr().err, "--yes")

    assert run("rm", "--all", "--yes", "--db", db) == 0
    assert "Removed" in capsys.readouterr().out

    assert run("stats", "--db", db, "--json") == 0
    assert stdout_json(capsys)["chunks"] == 0


def test_rm_rejects_sources_and_where_together(
    db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("rm", "a.md", "--where", '{"year": 2024}', "--db", db) == 1
    assert shows(capsys.readouterr().err, "not both")


def test_rm_refuses_an_empty_filter(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("rm", "--where", "{}", "--db", db) == 1
    assert shows(capsys.readouterr().err, "matches every chunk")


def test_rm_unknown_source_reports_it(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    assert run("rm", "does-not-exist.md", "--db", db) == 1
    assert shows(capsys.readouterr().err, "not indexed")


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #


def test_query_streams_and_lists_sources(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()

    assert run("query", "How long do refunds take?", "--db", db) == 0
    out = capsys.readouterr().out
    # EchoChatModel returns the rendered prompt, so the context must be in it.
    assert shows(out, "Refunds are issued within 30 days")
    assert shows(out, "sources:")
    assert shows(out, "refunds.md")


def test_query_no_sources_and_no_stream(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    assert run("query", "refunds", "--db", db, "--no-stream", "--no-sources") == 0
    out = capsys.readouterr().out
    assert "sources:" not in out


def test_query_json(db: str, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    assert run("query", "refunds", "--db", db, "--json") == 0
    payload = stdout_json(capsys)
    assert payload["question"] == "refunds"
    assert payload["answer"]
    assert payload["sources"]


def test_query_prompt_file(
    db: str, docs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("CUSTOM {context} :: {question}", encoding="utf-8")
    assert run("query", "refunds", "--db", db, "--prompt-file", str(prompt)) == 0
    assert shows(capsys.readouterr().out, "CUSTOM")


def test_query_prompt_file_must_have_placeholders(
    db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = tmp_path / "bad.txt"
    prompt.write_text("no placeholders here", encoding="utf-8")
    assert run("query", "x", "--db", db, "--prompt-file", str(prompt)) == 1
    err = capsys.readouterr().err
    assert shows(err, "{context}")
    assert "Traceback" not in err


def test_query_missing_prompt_file(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("query", "x", "--db", db, "--prompt-file", "nope.txt") == 1
    assert shows(capsys.readouterr().err, "Could not read")


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_missing_file_is_a_clean_error(
    db: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", "definitely-not-here.txt", "--db", db) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: File not found")
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_missing_file_indexes_nothing(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # One bad path among good ones fails before anything is written.
    assert run("add", str(docs / "refunds.md"), "missing.md", "--db", db) == 1
    capsys.readouterr()
    assert run("stats", "--db", db, "--json") == 0
    assert stdout_json(capsys)["chunks"] == 0


def test_bad_metadata_is_a_clean_error(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("add", str(docs), "--db", db, "--metadata", "oops") == 1
    err = capsys.readouterr().err
    assert shows(err, "KEY=VALUE")
    assert "Traceback" not in err


def test_debug_flag_reraises(db: str) -> None:
    with pytest.raises(ConfigurationError):
        run("search", "x", "--db", db, "--where", "[1]", "--debug")


def test_keyboard_interrupt_exits_130(
    db: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_args: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setitem(
        __import__("softrag.cli", fromlist=["COMMANDS"]).COMMANDS, "stats", boom
    )
    assert run("stats", "--db", db) == 130


# --------------------------------------------------------------------------- #
# optimize / doctor
# --------------------------------------------------------------------------- #


def test_optimize(db: str, docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()
    assert run("optimize", "--db", db) == 0
    assert shows(capsys.readouterr().out, "Optimized")


def test_doctor_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("doctor") == 0
    out = capsys.readouterr().out
    assert shows(out, "sqlite-vec")
    assert shows(out, "verdict:")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert shows(out, name)


def test_doctor_never_prints_a_key_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-do-not-leak-this-value-12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    assert run("doctor") == 0
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "SET" in captured.out


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #


def test_search_never_touches_the_chat_model(
    db: str,
    docs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(**_: Any) -> Any:
        raise AssertionError("a retrieval-only command must not resolve a chat model")

    monkeypatch.setattr(providers, "auto_chat_model", explode)
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    for command in (
        ("search", "refunds", "--db", db),
        ("ls", "--db", db),
        ("stats", "--db", db),
        ("optimize", "--db", db),
        ("rm", "--all", "--yes", "--db", db),
    ):
        assert run(*command) == 0
    capsys.readouterr()


def test_embed_model_flag_is_forwarded(
    db: str,
    docs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str | None] = []

    def spy(*, model: str | None = None) -> Any:
        seen.append(model)
        return providers.HashEmbedder(dimensions=64)

    monkeypatch.setattr(providers, "auto_embedder", spy)
    assert run("add", str(docs), "--db", db, "--quiet", "--embed-model", "tiny") == 0
    capsys.readouterr()
    assert seen == ["tiny"]


def test_engine_flags_work_before_the_command(
    db: str, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("--db", db, "add", str(docs), "--quiet") == 0
    capsys.readouterr()
    assert run("--db", db, "ls", "--json") == 0
    assert len(stdout_json(capsys)) == 2


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #


def _fake_input(lines: Sequence[str]) -> Any:
    """An ``input`` replacement that replays ``lines`` then raises EOF."""
    iterator = iter(lines)

    def read(_prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return read


def test_shell_meta_commands(
    db: str,
    docs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "builtins.input",
        _fake_input(
            ["\\help", "\\ls", "\\stats", "\\search refunds", "\\bogus", "\\quit"]
        ),
    )
    assert run("shell", "--db", db) == 0
    captured = capsys.readouterr()
    assert shows(captured.out, "\\search <query>")
    assert shows(captured.out, "refunds.md")
    assert shows(captured.out, "chunks from")
    assert shows(captured.err, "unknown command")
    assert captured.out.rstrip().endswith("bye.")


def test_shell_answers_questions_and_handles_eof(
    db: str,
    docs: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("add", str(docs), "--db", db, "--quiet") == 0
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", _fake_input(["", "How long do refunds take?"]))
    assert run("shell", "--db", db) == 0
    out = capsys.readouterr().out
    assert shows(out, "Refunds are issued within 30 days")
    assert shows(out, "sources:")


def test_shell_survives_ctrl_c(
    db: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    replies = iter(["interrupt", "\\quit"])

    def read(_prompt: str = "") -> str:
        value = next(replies)
        if value == "interrupt":
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr("builtins.input", read)
    assert run("shell", "--db", db) == 0
    assert shows(capsys.readouterr().out, "\\quit to leave")
