from __future__ import annotations

import ast
from pathlib import Path

import pytest

from heliostune._reference_analyzer import analyze
from heliostune.errors import SchemaError

_CANONICAL_INPUT = b'{\n  "values": [\n    -7,\n    0,\n    12\n  ]\n}\n'
_EXPECTED_OUTPUT = (
    b"{\n"
    b'  "count": 3,\n'
    b'  "input_sha256": "269889f8364c96e03c00d43df71b75329c1f41fec88d55a2c38b4ac74ec714e4",\n'
    b'  "maximum": 12,\n'
    b'  "minimum": -7,\n'
    b'  "sum": 5\n'
    b"}\n"
)


def _invoke(payload: bytes) -> tuple[tuple[str, bytes], ...]:
    return analyze((("analysis_input", payload),))


def test_exact_vector_is_deterministic_and_byte_exact() -> None:
    expected = (("analysis_summary", _EXPECTED_OUTPUT),)

    assert _invoke(_CANONICAL_INPUT) == expected
    assert _invoke(_CANONICAL_INPUT) == expected


def test_signed_64_bit_boundaries_are_exact_integers() -> None:
    payload = b'{\n  "values": [\n    -9223372036854775808,\n    9223372036854775807\n  ]\n}\n'
    expected = (
        b"{\n"
        b'  "count": 2,\n'
        b'  "input_sha256": "4df0bf436dbc9b972152c1d4268a66ed3af0d3adb4371dcbd35679092ca4aa1e",\n'
        b'  "maximum": 9223372036854775807,\n'
        b'  "minimum": -9223372036854775808,\n'
        b'  "sum": -1\n'
        b"}\n"
    )

    assert _invoke(payload) == (("analysis_summary", expected),)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not JSON",
        b'{\n  "values": [1]\n',
        b'\xff{\n  "values": [\n    1\n  ]\n}\n',
        b'{\n  "values": [\n    NaN\n  ]\n}\n',
        b'{\n  "values": [\n    Infinity\n  ]\n}\n',
    ],
)
def test_malformed_inputs_are_rejected(payload: bytes) -> None:
    with pytest.raises(SchemaError):
        _invoke(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{\n  "values": [\n    1\n  ],\n  "values": [\n    2\n  ]\n}\n',
        b'{\n  "extra": 0,\n  "values": [\n    1\n  ]\n}\n',
        b'{\n  "values": [\n    1\n  ],\n  "extra": 0\n}\n',
        b'{\n  "values": [\n    1\n  ]\n}\n\n',
        b'{"values":[1]}',
        b'{\r\n  "values": [\r\n    1\r\n  ]\r\n}\r\n',
        b'{\n  "values": [\n    -0\n  ]\n}\n',
    ],
)
def test_duplicate_unknown_and_noncanonical_bytes_are_rejected(payload: bytes) -> None:
    with pytest.raises(SchemaError):
        _invoke(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"[]\n",
        b"{}\n",
        b'{\n  "values": null\n}\n',
        b'{\n  "values": "1"\n}\n',
        b'{\n  "values": [\n    true\n  ]\n}\n',
        b'{\n  "values": [\n    1.0\n  ]\n}\n',
        b'{\n  "values": [\n    "1"\n  ]\n}\n',
        b'{\n  "values": [\n    null\n  ]\n}\n',
        b'{\n  "values": [\n    {}\n  ]\n}\n',
    ],
)
def test_wrong_json_types_are_rejected(payload: bytes) -> None:
    with pytest.raises(SchemaError):
        _invoke(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{\n  "values": [\n    -9223372036854775809\n  ]\n}\n',
        b'{\n  "values": [\n    9223372036854775808\n  ]\n}\n',
    ],
)
def test_out_of_range_values_are_rejected(payload: bytes) -> None:
    with pytest.raises(SchemaError, match="signed 64-bit range"):
        _invoke(payload)


def test_empty_and_overlong_arrays_are_rejected() -> None:
    with pytest.raises(SchemaError, match="between 1 and 4096"):
        _invoke(b'{\n  "values": []\n}\n')

    values = b",\n".join(b"    0" for _ in range(4097))
    payload = b'{\n  "values": [\n' + values + b"\n  ]\n}\n"
    with pytest.raises(SchemaError, match="between 1 and 4096"):
        _invoke(payload)


def test_exact_maximum_count_is_accepted() -> None:
    values = b",\n".join(b"    0" for _ in range(4096))
    payload = b'{\n  "values": [\n' + values + b"\n  ]\n}\n"

    role, output = _invoke(payload)[0]

    assert role == "analysis_summary"
    assert b'  "count": 4096,\n' in output
    assert b'  "minimum": 0,\n' in output
    assert b'  "maximum": 0,\n' in output
    assert b'  "sum": 0\n' in output


def test_sum_overflow_and_underflow_are_rejected() -> None:
    overflow = b'{\n  "values": [\n    1,\n    9223372036854775807\n  ]\n}\n'
    underflow = b'{\n  "values": [\n    -9223372036854775808,\n    -1\n  ]\n}\n'

    with pytest.raises(SchemaError, match="sum is outside"):
        _invoke(overflow)
    with pytest.raises(SchemaError, match="sum is outside"):
        _invoke(underflow)


@pytest.mark.parametrize(
    "inputs",
    [
        (),
        (("wrong_role", _CANONICAL_INPUT),),
        (("analysis_input", _CANONICAL_INPUT), ("analysis_input", _CANONICAL_INPUT)),
        (("other", b"payload"), ("analysis_input", _CANONICAL_INPUT)),
        (("analysis_input", _CANONICAL_INPUT), ("other", b"payload")),
        [["analysis_input", _CANONICAL_INPUT]],
        (("analysis_input", bytearray(_CANONICAL_INPUT)),),
    ],
)
def test_input_role_count_order_and_tuple_types_are_strict(inputs: object) -> None:
    with pytest.raises(SchemaError):
        analyze(inputs)  # type: ignore[arg-type]


def test_module_imports_and_calls_are_statically_cpu_pure() -> None:
    source = Path(__file__).parents[1] / "src" / "heliostune" / "_reference_analyzer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported_roots: set[str] = set()
    forbidden_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    }
    forbidden_methods = {
        "connect",
        "fork",
        "getaddrinfo",
        "open",
        "popen",
        "read",
        "read_bytes",
        "read_text",
        "recv",
        "send",
        "sleep",
        "system",
        "time",
        "urandom",
        "write",
        "write_bytes",
        "write_text",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_methods

    assert imported_roots == {"__future__", "hashlib", "heliostune", "json"}
    assert imported_roots.isdisjoint(
        {"ctypes", "importlib", "random", "socket", "subprocess", "time", "torch", "triton"}
    )
