"""Audited pure reference analyzer for offline replay."""

from __future__ import annotations

import hashlib
import json

from heliostune.errors import SchemaError

_INPUT_ROLE = "analysis_input"
_OUTPUT_ROLE = "analysis_summary"
_MIN_INTEGER = -(1 << 63)
_MAX_INTEGER = (1 << 63) - 1
_MIN_VALUES = 1
_MAX_VALUES = 4096


def _reject_constant(value: str) -> object:
    raise SchemaError(f"non-finite JSON constant {value!r} is not permitted")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _decode_input(payload: bytes) -> list[int]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError("analysis_input is not valid UTF-8") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"analysis_input is not valid JSON at column {exc.colno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise SchemaError(f"analysis_input is not valid JSON: {exc}") from exc

    if type(decoded) is not dict or set(decoded) != {"values"}:
        raise SchemaError("analysis_input must contain exactly the key 'values'")
    values = decoded["values"]
    if type(values) is not list:
        raise SchemaError("analysis_input values must be an array")
    if not _MIN_VALUES <= len(values) <= _MAX_VALUES:
        raise SchemaError("analysis_input values must contain between 1 and 4096 integers")
    for value in values:
        if type(value) is not int:
            raise SchemaError("analysis_input values must contain only exact integers")
        if not _MIN_INTEGER <= value <= _MAX_INTEGER:
            raise SchemaError("analysis_input integer is outside the signed 64-bit range")
    if _canonical_json_bytes(decoded) != payload:
        raise SchemaError("analysis_input bytes are not canonical JSON")
    return values


def analyze(inputs: tuple[tuple[str, bytes], ...]) -> tuple[tuple[str, bytes], ...]:
    """Return the canonical integer summary for one canonical input payload."""

    if type(inputs) is not tuple or len(inputs) != 1:
        raise SchemaError("reference analyzer requires exactly one input")
    item = inputs[0]
    if type(item) is not tuple or len(item) != 2:
        raise SchemaError("reference analyzer input must be a (role, bytes) tuple")
    role, payload = item
    if type(role) is not str or role != _INPUT_ROLE:
        raise SchemaError("reference analyzer input role must be 'analysis_input'")
    if type(payload) is not bytes:
        raise SchemaError("reference analyzer input payload must be bytes")

    values = _decode_input(payload)
    total = sum(values)
    if not _MIN_INTEGER <= total <= _MAX_INTEGER:
        raise SchemaError("analysis_input sum is outside the signed 64-bit range")
    output = {
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "sum": total,
    }
    return ((_OUTPUT_ROLE, _canonical_json_bytes(output)),)
