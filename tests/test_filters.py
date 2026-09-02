"""The metadata filter DSL: compilation to SQL, and end-to-end filtering.

Two things matter here. First, that every operator compiles to the right
predicate. Second -- and more important -- that no user value ever reaches the
SQL *string*: values must always travel as bind parameters, or a filter becomes
an injection vector.
"""

from __future__ import annotations

import pytest

from softrag.errors import ConfigurationError
from softrag.filters import SQL_TRUE, compile_where


def sql_of(where):
    return compile_where(where)[0]


def params_of(where):
    return compile_where(where)[1]


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


def test_empty_filter_matches_everything():
    assert compile_where(None) == (SQL_TRUE, [])
    assert compile_where({}) == (SQL_TRUE, [])


def test_bare_value_means_equality():
    assert compile_where({"a": 1}) == ("json_extract(d.metadata, ?) = ?", ["$.a", 1])


@pytest.mark.parametrize(
    ("op", "operator"),
    [
        ("$eq", "="),
        ("$ne", "!="),
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ],
)
def test_comparison_operators(op, operator):
    sql, params = compile_where({"year": {op: 2024}})
    assert sql == f"json_extract(d.metadata, ?) {operator} ?"
    assert params == ["$.year", 2024]


def test_in_and_nin():
    sql, params = compile_where({"kind": {"$in": ["pdf", "docx"]}})
    assert sql == "json_extract(d.metadata, ?) IN (?, ?)"
    assert params == ["$.kind", "pdf", "docx"]

    sql, params = compile_where({"kind": {"$nin": ["pdf"]}})
    assert sql == "json_extract(d.metadata, ?) NOT IN (?)"
    assert params == ["$.kind", "pdf"]


def test_empty_in_matches_nothing_and_empty_nin_matches_everything():
    assert compile_where({"kind": {"$in": []}}) == ("0", [])
    assert compile_where({"kind": {"$nin": []}}) == (SQL_TRUE, [])


def test_like():
    sql, params = compile_where({"title": {"$like": "%invoice%"}})
    assert sql == "json_extract(d.metadata, ?) LIKE ?"
    assert params == ["$.title", "%invoice%"]


def test_exists():
    assert compile_where({"author": {"$exists": True}}) == (
        "json_extract(d.metadata, ?) IS NOT NULL",
        ["$.author"],
    )
    assert compile_where({"author": {"$exists": False}}) == (
        "json_extract(d.metadata, ?) IS NULL",
        ["$.author"],
    )


def test_contains_checks_both_array_membership_and_substrings():
    sql, params = compile_where({"tags": {"$contains": "urgent"}})
    assert "json_each" in sql
    assert "LIKE" in sql
    assert params == ["$.tags", "urgent", "$.tags", "%urgent%"]


def test_multiple_operators_on_one_field_are_a_conjunction():
    sql, params = compile_where({"year": {"$gte": 2020, "$lt": 2030}})
    assert sql == (
        "(json_extract(d.metadata, ?) >= ? AND json_extract(d.metadata, ?) < ?)"
    )
    assert params == ["$.year", 2020, "$.year", 2030]


def test_multiple_fields_are_a_conjunction():
    sql, params = compile_where({"a": 1, "b": 2})
    assert sql.startswith("(") and " AND " in sql
    assert params == ["$.a", 1, "$.b", 2]


# --------------------------------------------------------------------------- #
# Boolean composition
# --------------------------------------------------------------------------- #


def test_and():
    sql, params = compile_where({"$and": [{"a": 1}, {"b": 2}]})
    assert sql == (
        "(json_extract(d.metadata, ?) = ? AND json_extract(d.metadata, ?) = ?)"
    )
    assert params == ["$.a", 1, "$.b", 2]


def test_or():
    sql, _ = compile_where({"$or": [{"a": 1}, {"b": 2}]})
    assert " OR " in sql


def test_not():
    sql, params = compile_where({"$not": {"a": 1}})
    assert sql == "NOT (json_extract(d.metadata, ?) = ?)"
    assert params == ["$.a", 1]


def test_empty_and_or_or_matches_everything():
    assert compile_where({"$and": []}) == (SQL_TRUE, [])
    assert compile_where({"$or": []}) == (SQL_TRUE, [])


def test_deeply_nested_composition():
    where = {
        "$or": [
            {"$and": [{"year": {"$gte": 2020}}, {"kind": "pdf"}]},
            {"$not": {"public": True}},
        ]
    }
    sql, params = compile_where(where)
    assert sql.count("(") == sql.count(")")
    assert params == ["$.year", 2020, "$.kind", "pdf", "$.public", 1]


# --------------------------------------------------------------------------- #
# Values and field paths
# --------------------------------------------------------------------------- #


def test_booleans_bind_as_json_integers():
    assert params_of({"ok": True}) == ["$.ok", 1]
    assert params_of({"ok": False}) == ["$.ok", 0]


def test_none_compiles_to_is_null():
    # "= NULL" is never true in SQL, so a null filter must become IS NULL --
    # and then there is no value left to bind.
    sql, params = compile_where({"a": None})
    assert sql == "json_extract(d.metadata, ?) IS NULL"
    assert params == ["$.a"]


def test_lists_bind_as_canonical_json_text():
    assert params_of({"tags": ["b", "a"]}) == ["$.tags", '["b","a"]']
    assert params_of({"tags": {"$in": [["a"], ["b"]]}}) == [
        "$.tags",
        '["a"]',
        '["b"]',
    ]


def test_a_dict_value_must_go_through_an_explicit_operator():
    """A bare dict is read as an operator mapping, as in the MongoDB DSL."""
    with pytest.raises(ConfigurationError):
        compile_where({"meta": {"b": 1, "a": 2}})
    assert params_of({"meta": {"$eq": {"b": 1, "a": 2}}}) == [
        "$.meta",
        '{"a":2,"b":1}',
    ]


def test_dotted_field_paths_become_nested_json_paths():
    assert params_of({"meta.nested.deep": "v"})[0] == "$.meta.nested.deep"


def test_field_names_needing_quoting_are_quoted_in_the_path():
    assert params_of({"weird key!": 1})[0] == '$."weird key!"'
    assert params_of({'has"quote': 1})[0] == '$."has\\"quote"'


def test_an_explicit_json_path_is_passed_through():
    assert params_of({"$.already[0].a": 1})[0] == "$.already[0].a"


def test_a_custom_metadata_column_is_honoured():
    sql, _ = compile_where({"a": 1}, column="x.meta")
    assert sql == "json_extract(x.meta, ?) = ?"


# --------------------------------------------------------------------------- #
# Parameterisation -- the security-relevant property
# --------------------------------------------------------------------------- #

INJECTION_VALUES = [
    "'; DROP TABLE documents; --",
    '" OR 1=1 --',
    "\\'; DELETE FROM sources; --",
    "100 OR 1=1",
    "%'; --",
]


@pytest.mark.parametrize("evil", INJECTION_VALUES)
@pytest.mark.parametrize(
    "build",
    [
        lambda v: {"a": v},
        lambda v: {"a": {"$eq": v}},
        lambda v: {"a": {"$ne": v}},
        lambda v: {"a": {"$like": v}},
        lambda v: {"a": {"$in": [v]}},
        lambda v: {"a": {"$nin": [v]}},
        lambda v: {"a": {"$contains": v}},
        lambda v: {"$or": [{"a": v}, {"b": v}]},
        lambda v: {"$not": {"a": v}},
        lambda v: {v: 1},
    ],
    ids=[
        "bare",
        "eq",
        "ne",
        "like",
        "in",
        "nin",
        "contains",
        "or",
        "not",
        "as-field-name",
    ],
)
def test_user_values_never_appear_in_the_sql_string(build, evil):
    sql, params = compile_where(build(evil))
    assert evil not in sql
    # Field names are rendered into a JSON *path*, which itself is a bind param.
    assert not any(token in sql for token in ("DROP", "DELETE", "1=1"))
    assert all(isinstance(p, (str, int, float, type(None))) for p in params)


def test_the_number_of_placeholders_matches_the_number_of_params():
    for where in [
        {"a": 1},
        {"a": {"$in": [1, 2, 3]}},
        {"a": {"$contains": "x"}},
        {"a": {"$exists": True}},
        {"$and": [{"a": 1}, {"b": {"$lt": 2}}]},
        {"a": {"$gte": 1, "$lt": 10}},
    ]:
        sql, params = compile_where(where)
        assert sql.count("?") == len(params), where


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [
        {"$nope": 1},
        {"a": {"$nope": 1}},
        {"$and": "not a list"},
        {"$or": {"a": 1}},
        {"$and": [1]},
        {"$not": "not a mapping"},
        {"a": {"$in": "abc"}},
        {"a": {"$nin": 5}},
    ],
)
def test_malformed_filters_raise_configuration_error(bad):
    with pytest.raises(ConfigurationError):
        compile_where(bad)


def test_the_error_names_the_supported_operators():
    with pytest.raises(ConfigurationError, match=r"\$gte"):
        compile_where({"a": {"$nope": 1}})


# --------------------------------------------------------------------------- #
# End to end, through the engine
# --------------------------------------------------------------------------- #


@pytest.fixture
def filtered(make_engine):
    """An engine whose four documents differ only in their metadata."""
    engine = make_engine()
    rows = [
        (
            "a",
            {
                "year": 2020,
                "kind": "pdf",
                "tags": ["x", "y"],
                "ok": True,
                "nested": {"deep": "v"},
                "author": "ada",
            },
        ),
        ("b", {"year": 2024, "kind": "pdf", "tags": ["z"], "ok": False}),
        ("c", {"year": 2024, "kind": "html", "tags": [], "ok": True}),
        ("d", {"year": 2019, "kind": "md"}),
    ]
    for name, metadata in rows:
        engine.add_text(
            f"shared searchable document body {name}", name=name, metadata=metadata
        )
    return engine


def found(engine, where):
    hits = engine.search("shared searchable document", where=where, top_k=10)
    return sorted({hit.source for hit in hits})


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        (None, ["a", "b", "c", "d"]),
        ({}, ["a", "b", "c", "d"]),
        ({"year": 2024}, ["b", "c"]),
        ({"year": {"$eq": 2024}}, ["b", "c"]),
        ({"year": {"$ne": 2024}}, ["a", "d"]),
        ({"year": {"$gt": 2020}}, ["b", "c"]),
        ({"year": {"$gte": 2020}}, ["a", "b", "c"]),
        ({"year": {"$lt": 2020}}, ["d"]),
        ({"year": {"$lte": 2020}}, ["a", "d"]),
        ({"year": {"$gte": 2020, "$lt": 2024}}, ["a"]),
        ({"kind": {"$in": ["pdf", "md"]}}, ["a", "b", "d"]),
        ({"kind": {"$in": []}}, []),
        ({"kind": {"$nin": ["pdf"]}}, ["c", "d"]),
        ({"kind": {"$nin": []}}, ["a", "b", "c", "d"]),
        ({"kind": {"$like": "p%"}}, ["a", "b"]),
        ({"tags": {"$contains": "y"}}, ["a"]),
        ({"author": {"$exists": True}}, ["a"]),
        ({"author": {"$exists": False}}, ["b", "c", "d"]),
        ({"ok": True}, ["a", "c"]),
        ({"ok": False}, ["b"]),
        ({"nested.deep": "v"}, ["a"]),
        ({"$and": [{"year": 2024}, {"kind": "pdf"}]}, ["b"]),
        ({"$or": [{"year": 2019}, {"kind": "html"}]}, ["c", "d"]),
        ({"$not": {"year": 2024}}, ["a", "d"]),
        ({"$or": [{"$and": [{"year": 2024}, {"ok": True}]}, {"year": 2019}]}, ["c", "d"]),
        ({"year": 2024, "ok": True}, ["c"]),
        ({"tags": [1, 2]}, []),
    ],
)
def test_end_to_end_filtering(filtered, where, expected):
    assert found(filtered, where) == expected


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_filters_apply_in_every_search_mode(filtered, mode):
    hits = filtered.search(
        "shared searchable document", where={"year": 2024}, mode=mode, top_k=10
    )
    assert {hit.source for hit in hits} <= {"b", "c"}
    assert hits, f"{mode} search returned nothing at all"


def test_an_injection_shaped_filter_value_deletes_nothing(filtered):
    before = len(filtered)
    assert found(filtered, {"kind": "'; DROP TABLE documents; --"}) == []
    assert len(filtered) == before


def test_a_bad_filter_raises_before_touching_the_database(filtered):
    with pytest.raises(ConfigurationError):
        filtered.search("anything", where={"$nope": 1})


def test_delete_where_uses_the_same_dsl(filtered):
    assert filtered.delete(where={"year": {"$lt": 2024}}) == 2
    assert found(filtered, None) == ["b", "c"]


def test_filtering_on_an_explicit_null_value(filtered):
    filtered.add_text("shared searchable document e", name="e", metadata={"year": None})
    assert found(filtered, {"year": None}) == ["e"]
