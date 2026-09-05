"""Exact-type validators for decoded artifact values."""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import cast

from heliostune.errors import SchemaError


def exact_object(value: object, *, context: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SchemaError(f"{context} must be an object")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise SchemaError(f"{context} object keys must be strings")
    return cast(dict[str, object], result)


def exact_fields(
    value: object,
    *,
    required: Collection[str],
    context: str,
) -> dict[str, object]:
    result = exact_object(value, context=context)
    expected = set(required)
    actual = set(result)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing!r}")
        if unknown:
            details.append(f"unknown fields {unknown!r}")
        raise SchemaError(f"{context} has {' and '.join(details)}")
    return result


def exact_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{context} must be a boolean")
    return value


def exact_int(
    value: object,
    *,
    context: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise SchemaError(f"{context} must be an integer")
    result = value
    if minimum is not None and result < minimum:
        raise SchemaError(f"{context} must be at least {minimum}")
    return result


def finite_float(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in {int, float}:
        raise SchemaError(f"{context} must be a number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise SchemaError(f"{context} must be finite")
    if strictly_positive and result <= 0:
        raise SchemaError(f"{context} must be positive")
    if minimum is not None and result < minimum:
        raise SchemaError(f"{context} must be at least {minimum}")
    return result


def optional_finite_float(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float | None:
    if value is None:
        return None
    return finite_float(
        value,
        context=context,
        minimum=minimum,
        strictly_positive=strictly_positive,
    )


def integer_pair(value: object, *, context: str) -> tuple[int, int]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be a two-element integer array")
    result = cast(list[object], value)
    if len(result) != 2:
        raise SchemaError(f"{context} must be a two-element integer array")
    return (
        exact_int(result[0], context=f"{context}[0]", minimum=0),
        exact_int(result[1], context=f"{context}[1]", minimum=0),
    )


def nonblank_string(value: object, *, context: str) -> str:
    if type(value) is not str:
        raise SchemaError(f"{context} must be a string")
    result = value
    if not result or result != result.strip():
        raise SchemaError(f"{context} must be nonblank with no surrounding whitespace")
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SchemaError(f"{context} must be valid Unicode") from exc
    return result


def optional_nonblank_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return nonblank_string(value, context=context)
