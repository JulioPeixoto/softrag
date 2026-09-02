"""Metadata filter DSL.

Filters are plain dictionaries, in the style users already know from MongoDB and
Chroma, compiled down to a parameterised SQL predicate over the JSON metadata
column. Nothing is ever interpolated into the SQL string, so filter values are
safe by construction.

    {"source": "handbook.pdf"}                    # equality
    {"year": {"$gte": 2020}}                      # comparison
    {"kind": {"$in": ["pdf", "docx"]}}            # membership
    {"title": {"$like": "%invoice%"}}             # SQL LIKE
    {"tags": {"$contains": "urgent"}}             # element of a JSON array
    {"author": {"$exists": True}}                 # key is present
    {"$or": [{"year": 2024}, {"pinned": True}]}   # boolean composition
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import ConfigurationError

__all__ = ["SQL_TRUE", "compile_where"]

SQL_TRUE = "1"

_COMPARISON = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}

_LOGICAL = {"$and": "AND", "$or": "OR"}


def compile_where(
    where: Mapping[str, Any] | None, *, column: str = "d.metadata"
) -> tuple[str, list[Any]]:
    """Compile a filter mapping into ``(sql_predicate, params)``.

    The predicate is always safe to drop into a ``WHERE`` clause and evaluates to
    true when ``where`` is empty, so callers never need a special case.

    Args:
        where: The filter expression, or ``None`` for "match everything".
        column: SQL expression yielding the JSON metadata object.

    Returns:
        A tuple of the SQL fragment and the ordered bind parameters.

    Raises:
        ConfigurationError: If the expression uses an unknown operator or has a
            shape that cannot be interpreted.
    """
    if not where:
        return SQL_TRUE, []
    sql, params = _compile_mapping(where, column)
    return sql, params


def _compile_mapping(where: Mapping[str, Any], column: str) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in where.items():
        if key.startswith("$."):
            # A raw JSON path, the escape hatch for anything the dotted syntax
            # cannot express (array indexing, most usefully). Unambiguous
            # against the operators, which are "$and"/"$or"/"$not".
            sub_sql, sub_params = _compile_field(key, value, column)
        elif key in _LOGICAL:
            sub_sql, sub_params = _compile_logical(key, value, column)
        elif key == "$not":
            inner_sql, inner_params = _compile_operand(value, column)
            sub_sql, sub_params = f"NOT ({inner_sql})", inner_params
        elif key.startswith("$"):
            raise ConfigurationError(
                f"Unknown top-level filter operator {key!r}. "
                f"Expected one of: $and, $or, $not, or a metadata field name."
            )
        else:
            sub_sql, sub_params = _compile_field(key, value, column)
        clauses.append(sub_sql)
        params.extend(sub_params)
    if not clauses:
        return SQL_TRUE, []
    if len(clauses) == 1:
        return clauses[0], params
    return "(" + " AND ".join(clauses) + ")", params


def _compile_logical(op: str, value: Any, column: str) -> tuple[str, list[Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(
            f"{op} expects a list of filter expressions, got {type(value).__name__}."
        )
    if not value:
        return SQL_TRUE, []
    parts: list[str] = []
    params: list[Any] = []
    for operand in value:
        sql, sub_params = _compile_operand(operand, column)
        parts.append(sql)
        params.extend(sub_params)
    joiner = f" {_LOGICAL[op]} "
    return "(" + joiner.join(parts) + ")", params


def _compile_operand(operand: Any, column: str) -> tuple[str, list[Any]]:
    if not isinstance(operand, Mapping):
        raise ConfigurationError(
            f"Expected a filter expression (a dict), got {type(operand).__name__}."
        )
    return _compile_mapping(operand, column)


def _compile_field(field: str, value: Any, column: str) -> tuple[str, list[Any]]:
    path = _json_path(field)
    extract = f"json_extract({column}, ?)"

    if not isinstance(value, Mapping):
        # Bare value means equality. JSON booleans round-trip as 0/1 integers.
        # None is the exception: "= NULL" is never true in SQL, so a filter for
        # a JSON null has to become an IS NULL test to match anything.
        if value is None:
            return f"{extract} IS NULL", [path]
        return f"{extract} = ?", [path, _bind(value)]

    if len(value) != 1:
        # {"$gte": 1, "$lt": 10} is a conjunction; compile each half.
        parts: list[str] = []
        params: list[Any] = []
        for op, operand in value.items():
            sql, sub = _compile_field(field, {op: operand}, column)
            parts.append(sql)
            params.extend(sub)
        return "(" + " AND ".join(parts) + ")", params

    ((op, operand),) = value.items()

    if op in _COMPARISON:
        if operand is None and op in ("$eq", "$ne"):
            test = "IS NULL" if op == "$eq" else "IS NOT NULL"
            return f"{extract} {test}", [path]
        return f"{extract} {_COMPARISON[op]} ?", [path, _bind(operand)]

    if op in ("$in", "$nin"):
        if not isinstance(operand, Sequence) or isinstance(operand, (str, bytes)):
            raise ConfigurationError(
                f"{op} expects a list, got {type(operand).__name__}."
            )
        if not operand:
            # An empty set matches nothing ($in) or everything ($nin).
            return ("0", []) if op == "$in" else (SQL_TRUE, [])
        placeholders = ", ".join("?" for _ in operand)
        negate = "NOT " if op == "$nin" else ""
        return (
            f"{extract} {negate}IN ({placeholders})",
            [path, *(_bind(v) for v in operand)],
        )

    if op == "$like":
        return f"{extract} LIKE ?", [path, operand]

    if op == "$exists":
        test = "IS NOT NULL" if operand else "IS NULL"
        return f"{extract} {test}", [path]

    if op == "$contains":
        # Matches an element of a JSON array, or a substring of a text value.
        return (
            f"(EXISTS (SELECT 1 FROM json_each({column}, ?) WHERE json_each.value = ?)"
            f" OR {extract} LIKE ?)",
            [path, _bind(operand), path, f"%{operand}%"],
        )

    raise ConfigurationError(
        f"Unknown filter operator {op!r} for field {field!r}. Supported: "
        "$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $like, $contains, $exists."
    )


def _json_path(field: str) -> str:
    """Build a JSON path for a field name, supporting dotted nesting.

    A name already written as a JSON path (``"$.items[0].sku"``) is passed
    through untouched, which is the only way to reach an array element.
    """
    if field.startswith("$."):
        return field
    parts = field.split(".")
    rendered = "$"
    for part in parts:
        if part.isidentifier():
            rendered += f".{part}"
        else:
            escaped = part.replace('"', '\\"')
            rendered += f'."{escaped}"'
    return rendered


def _bind(value: Any) -> Any:
    """Convert a Python value to the form ``json_extract`` will compare against."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float, str)) or value is None:
        return value
    # Lists and dicts compare against their canonical JSON text.
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
