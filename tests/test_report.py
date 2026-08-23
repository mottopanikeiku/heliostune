from pathlib import Path

from heliostune.report import render_report


def test_report_is_offline_and_escapes_dynamic_text(tmp_path: Path) -> None:
    output = tmp_path / "index.html"
    summary = {
        "source_gpu": "source<script>",
        "target_gpu": "target",
        "workloads": 2,
        "configs": 3,
        "methods": {
            "transfer_thompson": [
                {
                    "budget": 1,
                    "mean_fraction_oracle": 0.9,
                    "ci95_low": 0.8,
                    "ci95_high": 0.95,
                }
            ],
            "cold_thompson": [
                {
                    "budget": 1,
                    "mean_fraction_oracle": 0.7,
                    "ci95_low": 0.6,
                    "ci95_high": 0.8,
                }
            ],
        },
    }

    render_report(summary, output)
    document = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in document
    assert "source&lt;script&gt;" in document
    assert "source<script>" not in document
    assert "https://" not in document
    assert "<svg" in document
    assert 'type="application/json"' in document
