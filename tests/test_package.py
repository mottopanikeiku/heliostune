from __future__ import annotations

import importlib.metadata
import importlib.resources
import subprocess
import sys

import heliostune


def test_root_api_and_version_come_from_installed_metadata() -> None:
    assert heliostune.__version__ == importlib.metadata.version("heliostune") == "0.6.0.dev0"
    assert set(heliostune.__all__) == {
        "DEFAULT_CONFIGS",
        "DEFAULT_WORKLOADS",
        "HardwareProfile",
        "KernelConfig",
        "Measurement",
        "Workload",
        "__version__",
        "read_jsonl",
        "read_measurements",
        "write_jsonl",
        "write_measurements_atomic",
    }
    assert all(hasattr(heliostune, name) for name in heliostune.__all__)


def test_css_and_typing_marker_are_packaged_resources() -> None:
    package = importlib.resources.files("heliostune")
    css = package.joinpath("report.css").read_text(encoding="utf-8")
    assert ".control-panel" in css
    assert package.joinpath("py.typed").is_file()


def test_root_import_keeps_replay_and_report_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, heliostune; "
                "assert 'heliostune.replay' not in sys.modules; "
                "assert 'heliostune.report' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""
