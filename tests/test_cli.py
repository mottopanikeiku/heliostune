from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import pytest

import heliostune.cli as cli
from heliostune.artifacts import write_json_atomic
from heliostune.errors import SchemaError


@pytest.mark.parametrize(
    "arguments",
    [
        ["compare", "rows.jsonl", "--source", "L4", "--target", "A10", "--seeds", "0"],
        [
            "compare",
            "rows.jsonl",
            "--source",
            "L4",
            "--target",
            "A10",
            "--transfer-strength",
            "NaN",
        ],
        [
            "compare-multisource",
            "rows.jsonl",
            "--sources",
            "L4, A10",
            "--target",
            "T4",
        ],
        [
            "compare-multisource",
            "rows.jsonl",
            "--sources",
            "L4,L4",
            "--target",
            "T4",
        ],
        ["demo", "--seed", "-1"],
    ],
)
def test_argparse_rejects_bad_numeric_and_list_values(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(arguments)
    assert raised.value.code == 2
    assert ": error:" in capsys.readouterr().err


class _FakeParser:
    def __init__(self, handler: Callable[[argparse.Namespace], int]) -> None:
        self._handler = handler

    def parse_args(self, _argv: object) -> argparse.Namespace:
        return argparse.Namespace(handler=self._handler)


def _raising_handler(error: BaseException) -> Callable[[argparse.Namespace], int]:
    def handler(_args: argparse.Namespace) -> int:
        raise error

    return handler


@pytest.mark.parametrize(
    "error",
    [
        SchemaError("bad schema"),
        OSError("disk failed"),
        json.JSONDecodeError("bad json", "{", 1),
    ],
)
def test_main_formats_only_expected_user_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    monkeypatch.setattr(cli, "build_parser", lambda: _FakeParser(_raising_handler(error)))

    assert cli.main([]) == 2
    assert capsys.readouterr().err.startswith("heliostune: error: ")


@pytest.mark.parametrize("error", [ValueError("bug"), TypeError("bug"), KeyboardInterrupt()])
def test_main_does_not_catch_programmer_faults_or_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(cli, "build_parser", lambda: _FakeParser(_raising_handler(error)))
    with pytest.raises(
        type(error), match="bug" if not isinstance(error, KeyboardInterrupt) else None
    ):
        cli.main([])


def test_output_collision_is_rejected_before_input_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "summary.json"
    output.write_text("existing", encoding="utf-8")

    def unexpected_read(_path: Path) -> object:
        raise AssertionError("input was accessed before output collision rejection")

    monkeypatch.setattr(cli, "read_measurements", unexpected_read)
    result = cli.main(
        [
            "compare",
            str(tmp_path / "missing.jsonl"),
            "--source",
            "L4",
            "--target",
            "A10",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    assert "refusing to replace existing output" in capsys.readouterr().err


def test_invalid_report_is_normalized_before_output_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "bad-summary.json"
    output = tmp_path / "report.html"
    write_json_atomic(
        source,
        {
            "source_gpu": "L4",
            "target_gpu": "A10",
            "workloads": 1,
            "configs": 1,
            "methods": {"new_method": []},
        },
    )

    assert cli.main(["report", str(source), "--output", str(output)]) == 2
    assert not output.exists()
    assert "heliostune: error:" in capsys.readouterr().err


def test_demo_stages_complete_strict_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"

    assert (
        cli.main(
            [
                "demo",
                "--output-dir",
                str(output_dir),
                "--max-budget",
                "1",
                "--seeds",
                "1",
            ]
        )
        == 0
    )

    assert (output_dir / "measurements.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    report = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in report
    assert "control-panel" in report


def test_version_is_installed_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"heliostune {version('heliostune')}"
