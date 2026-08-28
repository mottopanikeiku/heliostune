"""Self-contained HTML reporting for HeliosTune tuning summaries."""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from heliostune.artifacts import write_text_atomic
from heliostune.report_model import ReportData, normalize_report_summary

_CHART_WIDTH = 960
_CHART_HEIGHT = 410
_PLOT_LEFT = 72
_PLOT_RIGHT = 22
_PLOT_TOP = 20
_PLOT_BOTTOM = 58
_SERIES_COUNT = 10
_DASH_PATTERNS = (None, "9 5", "3 5", "12 4 3 4", "6 4 2 4", "2 3")
_MARKERS = ("circle", "diamond", "square")
_MISSING = "Not reported"


def _styles() -> str:
    return files("heliostune").joinpath("report.css").read_text(encoding="utf-8")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _human_label(value: Any) -> str:
    text = re.sub(r"[_-]+", " ", str(value)).strip()
    return text[:1].upper() + text[1:] if text else "Value"


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10_000 or magnitude < 0.001:
        return f"{value:.4g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_budget(value: float) -> str:
    return str(int(value)) if value.is_integer() else _format_number(value)


def _format_value(value: Any, key: str | None = None) -> str:
    if value is None or value == "":
        return _MISSING
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if (
        key == "compute_capability"
        and isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        return ".".join(str(part) for part in value)
    if isinstance(value, float):
        rendered = _format_number(value)
        return f"{rendered} GB" if key in {"memory_gb", "total_memory_gb"} else rendered
    if isinstance(value, Mapping):
        return "; ".join(
            f"{_human_label(item_key)}: {_format_value(item_value, str(item_key))}"
            for item_key, item_value in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(_format_value(item) for item in value) or _MISSING
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, Mapping) else None
    return None


def _record_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        record_keys = {
            "m",
            "n",
            "k",
            "family",
            "block_m",
            "block_n",
            "block_k",
            "num_warps",
            "num_stages",
        }
        if record_keys.intersection(value):
            return [value]
        records: list[Mapping[str, Any]] = []
        for name, item in value.items():
            mapped = _as_mapping(item)
            if mapped is None:
                records.append({"name": name, "value": item})
            else:
                records.append({"name": name, **mapped})
        return records
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = []
        for index, item in enumerate(value, start=1):
            mapped = _as_mapping(item)
            records.append(mapped if mapped is not None else {"name": index, "value": item})
        return records
    return []


def _item_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{location} must be finite")
    return number


def _normalise_methods(value: Any) -> dict[str, list[dict[str, float]]]:
    if not isinstance(value, Mapping):
        raise TypeError("summary['methods'] must be a mapping")
    methods: dict[str, list[dict[str, float]]] = {}
    for raw_name, raw_points in value.items():
        name = str(raw_name)
        if not name:
            raise ValueError("method names must not be empty")
        if name in methods:
            raise ValueError(f"duplicate method name after string conversion: {name!r}")
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise TypeError(f"summary['methods'][{name!r}] must be a sequence")
        points: list[dict[str, float]] = []
        seen_budgets: set[float] = set()
        for index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, Mapping):
                raise TypeError(f"point {index} for method {name!r} must be a mapping")
            location = f"method {name!r}, point {index}"
            try:
                budget = _finite_number(raw_point["budget"], f"{location} budget")
                mean = _finite_number(
                    raw_point["mean_fraction_oracle"], f"{location} mean_fraction_oracle"
                )
                low = _finite_number(raw_point["ci95_low"], f"{location} ci95_low")
                high = _finite_number(raw_point["ci95_high"], f"{location} ci95_high")
            except KeyError as exc:
                raise ValueError(f"{location} is missing {exc.args[0]!r}") from exc
            if budget < 0:
                raise ValueError(f"{location} budget must be non-negative")
            if low > high:
                raise ValueError(f"{location} has ci95_low greater than ci95_high")
            if budget in seen_budgets:
                raise ValueError(f"method {name!r} has duplicate budget {_format_budget(budget)}")
            seen_budgets.add(budget)
            points.append({"budget": budget, "mean": mean, "low": low, "high": high})
        methods[name] = sorted(points, key=lambda point: point["budget"])
    return methods


def _method_display_labels(
    summary: Mapping[str, Any], methods: Mapping[str, list[dict[str, float]]]
) -> dict[str, str]:
    supplied = _as_mapping(summary.get("method_labels"))
    labels: dict[str, str] = {}
    for name in methods:
        value = supplied.get(name) if supplied is not None else None
        labels[name] = str(value).strip() if value not in (None, "") else _human_label(name)
    return labels


def _data_status(summary: Mapping[str, Any]) -> tuple[str, str]:
    raw_kind = summary.get("data_kind")
    kind = str(raw_kind).strip().lower() if raw_kind not in (None, "") else ""
    if kind == "synthetic":
        return (
            "Synthetic data · not hardware measurements",
            "Generated benchmark values; device identities and curves are evidence about the replay protocol, not measured GPU performance.",
        )
    if kind == "measured":
        return (
            "Measured benchmark data",
            "Reported as hardware measurements; the hardware, timing protocol, source cost, and study boundaries below define their scope.",
        )
    if kind:
        return (
            f"Data kind · {_human_label(raw_kind)}",
            "The summary supplies a data classification other than synthetic or measured; measurement status is not inferred.",
        )
    return (
        "Measurement status not reported",
        "The summary does not identify whether values are synthetic or measured; hardware claims cannot be established from this report alone.",
    )


def _safe_json(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _nice_budget_ticks(maximum: float) -> list[float]:
    if maximum <= 0:
        return [0.0]
    rough_step = maximum / 5
    magnitude = 10 ** math.floor(math.log10(rough_step))
    residual = rough_step / magnitude
    if residual <= 1:
        step = magnitude
    elif residual <= 2:
        step = 2 * magnitude
    elif residual <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    ticks: list[float] = []
    tick = 0.0
    while tick < maximum:
        ticks.append(tick)
        tick += step
    ticks.append(maximum if not math.isclose(ticks[-1], maximum) else ticks[-1])
    return ticks


def _marker_svg(kind: str, x: float, y: float, color: str, label: str) -> str:
    safe_label = _escape(label)
    if kind == "diamond":
        shape = f'<polygon points="{x:.2f},{y - 5:.2f} {x + 5:.2f},{y:.2f} {x:.2f},{y + 5:.2f} {x - 5:.2f},{y:.2f}" />'
    elif kind == "square":
        shape = f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" />'
    else:
        shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" />'
    return f'<g class="series-point" fill="{color}"><title>{safe_label}</title>{shape}</g>'


def _render_chart(methods: Mapping[str, list[dict[str, float]]], labels: Mapping[str, str]) -> str:
    all_points = [point for points in methods.values() for point in points]
    if not all_points:
        return '<div class="empty-state">No budget-efficiency points were supplied.</div>'

    max_budget = max(point["budget"] for point in all_points)
    observed_high = max(max(point["high"], point["mean"]) for point in all_points)
    y_max = max(1.0, math.ceil(observed_high * 4) / 4)
    plot_width = _CHART_WIDTH - _PLOT_LEFT - _PLOT_RIGHT
    plot_height = _CHART_HEIGHT - _PLOT_TOP - _PLOT_BOTTOM

    def x_position(budget: float) -> float:
        ratio = budget / max_budget if max_budget else 0.5
        return _PLOT_LEFT + ratio * plot_width

    def y_position(value: float) -> float:
        return _PLOT_TOP + (1 - value / y_max) * plot_height

    elements = [
        f'<svg viewBox="0 0 {_CHART_WIDTH} {_CHART_HEIGHT}" role="img" aria-labelledby="curve-title curve-desc">',
        '<title id="curve-title">Budget efficiency by tuning method</title>',
        '<desc id="curve-desc">Mean fraction of the held-out reference by target evaluation budget. Shaded regions show supplied 95 percent confidence intervals.</desc>',
        f'<defs><clipPath id="plot-clip"><rect x="{_PLOT_LEFT}" y="{_PLOT_TOP}" width="{plot_width}" height="{plot_height}" /></clipPath></defs>',
    ]

    for tick_index in range(5):
        value = y_max * tick_index / 4
        y = y_position(value)
        elements.append(
            f'<line class="chart-grid" x1="{_PLOT_LEFT}" x2="{_CHART_WIDTH - _PLOT_RIGHT}" y1="{y:.2f}" y2="{y:.2f}" />'
        )
        elements.append(
            f'<text class="chart-text" x="{_PLOT_LEFT - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.0%}</text>'
        )

    for budget in _nice_budget_ticks(max_budget):
        x = x_position(budget)
        elements.append(
            f'<line class="chart-grid" x1="{x:.2f}" x2="{x:.2f}" y1="{_PLOT_TOP}" y2="{_CHART_HEIGHT - _PLOT_BOTTOM}" />'
        )
        elements.append(
            f'<text class="chart-text" x="{x:.2f}" y="{_CHART_HEIGHT - _PLOT_BOTTOM + 25}" text-anchor="middle">{_escape(_format_budget(budget))}</text>'
        )

    oracle_y = y_position(1.0)
    elements.extend(
        [
            f'<line class="oracle-line" x1="{_PLOT_LEFT}" x2="{_CHART_WIDTH - _PLOT_RIGHT}" y1="{oracle_y:.2f}" y2="{oracle_y:.2f}" />',
            f'<text class="chart-text" x="{_CHART_WIDTH - _PLOT_RIGHT - 4}" y="{oracle_y - 8:.2f}" text-anchor="end">reference parity</text>',
            f'<line class="chart-axis" x1="{_PLOT_LEFT}" x2="{_PLOT_LEFT}" y1="{_PLOT_TOP}" y2="{_CHART_HEIGHT - _PLOT_BOTTOM}" />',
            f'<line class="chart-axis" x1="{_PLOT_LEFT}" x2="{_CHART_WIDTH - _PLOT_RIGHT}" y1="{_CHART_HEIGHT - _PLOT_BOTTOM}" y2="{_CHART_HEIGHT - _PLOT_BOTTOM}" />',
            f'<text class="chart-label" x="{_PLOT_LEFT + plot_width / 2:.2f}" y="{_CHART_HEIGHT - 18}" text-anchor="middle">Target evaluation budget</text>',
            f'<text class="chart-label" x="22" y="{_PLOT_TOP + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {_PLOT_TOP + plot_height / 2:.2f})">Fraction of held-out reference</text>',
            '<g clip-path="url(#plot-clip)">',
        ]
    )

    for series_index, (name, points) in enumerate(methods.items()):
        if not points:
            continue
        display_name = labels[name]
        color = f"var(--series-{series_index % _SERIES_COUNT})"
        dash = _DASH_PATTERNS[(series_index // _SERIES_COUNT + series_index) % len(_DASH_PATTERNS)]
        marker = _MARKERS[series_index % len(_MARKERS)]
        upper = [(x_position(point["budget"]), y_position(point["high"])) for point in points]
        lower = [
            (x_position(point["budget"]), y_position(point["low"])) for point in reversed(points)
        ]
        band_path = " ".join(
            [f"M {upper[0][0]:.2f} {upper[0][1]:.2f}"]
            + [f"L {x:.2f} {y:.2f}" for x, y in upper[1:] + lower]
            + ["Z"]
        )
        elements.append(f'<path class="series-band" d="{band_path}" fill="{color}" />')
        mean_points = [(x_position(point["budget"]), y_position(point["mean"])) for point in points]
        line_path = " ".join(
            [f"M {mean_points[0][0]:.2f} {mean_points[0][1]:.2f}"]
            + [f"L {x:.2f} {y:.2f}" for x, y in mean_points[1:]]
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<path class="series-line" d="{line_path}" stroke="{color}"{dash_attribute} />'
        )
        for point, (x, y) in zip(points, mean_points, strict=True):
            label = (
                f"{display_name}: budget {_format_budget(point['budget'])}, "
                f"mean {_format_number(point['mean'])}, 95% CI "
                f"{_format_number(point['low'])} to {_format_number(point['high'])}"
            )
            elements.append(_marker_svg(marker, x, y, color, label))

    elements.extend(["</g>", "</svg>"])
    return "".join(elements)


def _render_legend(methods: Mapping[str, list[dict[str, float]]], labels: Mapping[str, str]) -> str:
    items = []
    for series_index, (name, points) in enumerate(methods.items()):
        color = f"var(--series-{series_index % _SERIES_COUNT})"
        dash = _DASH_PATTERNS[(series_index // _SERIES_COUNT + series_index) % len(_DASH_PATTERNS)]
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        endpoint = (
            f"{_format_number(points[-1]['mean'])} at budget {_format_budget(points[-1]['budget'])}"
            if points
            else "no points"
        )
        items.append(
            "<li>"
            '<svg viewBox="0 0 42 14" width="42" height="14" aria-hidden="true">'
            f'<line x1="1" x2="41" y1="7" y2="7" stroke="{color}" stroke-width="3"{dash_attribute} />'
            f'<circle cx="21" cy="7" r="3.5" fill="{color}" />'
            "</svg>"
            f'<span><strong>{_escape(labels[name])}</strong><span class="legend-endpoint">{_escape(endpoint)}</span></span>'
            "</li>"
        )
    return f'<ul class="method-legend">{"".join(items)}</ul>' if items else ""


def _method_by_role(
    summary: Mapping[str, Any], methods: Mapping[str, list[dict[str, float]]], role: str
) -> str | None:
    explicit = summary.get(f"{role}_method")
    if explicit is not None and str(explicit) in methods:
        return str(explicit)
    tokens = (
        ("transfer", "warm", "posterior", "source")
        if role == "transfer"
        else ("cold", "scratch", "uninformed", "no transfer")
    )
    for name in methods:
        lowered = name.lower().replace("_", " ").replace("-", " ")
        if any(token in lowered for token in tokens):
            return name
    return None


def _method_label(value: Any, labels: Mapping[str, str], fallback: str) -> str:
    if value in (None, ""):
        return fallback
    key = str(value)
    return labels.get(key, _human_label(key))


def _paired_primary_metric(
    summary: Mapping[str, Any],
) -> tuple[bool, Mapping[str, Any] | None]:
    primary_metrics = _as_mapping(summary.get("primary_metrics"))
    metric_key = "paired_parhelion_vs_primary_auc_delta"
    if primary_metrics is not None and metric_key in primary_metrics:
        return True, _as_mapping(primary_metrics.get(metric_key))

    headline = _as_mapping(summary.get("headline"))
    headline_key = "paired_auc_delta_vs_primary"
    if headline is not None and headline_key in headline:
        return True, _as_mapping(headline.get(headline_key))
    return False, None


def _descriptive_strongest_legacy(summary: Mapping[str, Any]) -> Any:
    for container_name in ("headline", "primary_metrics"):
        container = _as_mapping(summary.get(container_name))
        if container is not None and "descriptive_target_strongest_legacy_method" in container:
            return container.get("descriptive_target_strongest_legacy_method")
    return None


def _supplied_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_auc_delta(value: Any) -> str:
    number = _supplied_number(value)
    if number is None:
        return _format_value(value)
    return f"{number:+.4f} AUC ({number * 100:+.2f} pp)"


def _format_auc_interval(low: Any, high: Any) -> str:
    low_number = _supplied_number(low)
    high_number = _supplied_number(high)
    if low_number is None or high_number is None:
        return f"[{_format_value(low)}, {_format_value(high)}]"
    return (
        f"[{low_number:+.4f}, {high_number:+.4f}] AUC "
        f"([{low_number * 100:+.2f}, {high_number * 100:+.2f}] pp)"
    )


def _render_primary_evidence(
    summary: Mapping[str, Any],
    labels: Mapping[str, str],
    transfer: str | None,
) -> str:
    _, evidence = _paired_primary_metric(summary)
    if evidence is None:
        return (
            '<div class="empty-state">A paired frozen primary-comparator endpoint was not supplied. '
            "The target endpoint comparison is not substituted with a transfer-versus-cold claim.</div>"
        )

    comparator_key = evidence.get("comparator")
    if comparator_key in (None, ""):
        comparator_key = summary.get("primary_comparator")
    transfer_key = summary.get("transfer_method", transfer)
    transfer_label = _method_label(transfer_key, labels, "Transfer method")
    comparator_label = _method_label(comparator_key, labels, _MISSING)

    mean_delta = evidence.get("mean_auc_delta")
    ci_low = evidence.get("ci95_low")
    ci_high = evidence.get("ci95_high")
    paired_seeds = evidence.get("paired_seeds")
    degrees_of_freedom = evidence.get("degrees_of_freedom")
    mean_number = _supplied_number(mean_delta)
    delta_class = "delta negative" if mean_number is not None and mean_number < 0 else "delta"
    supported_value = evidence.get("superiority_supported")
    supported = supported_value is True
    if supported:
        claim = evidence.get("claim")
        supplied_claim = (
            str(claim).strip()
            if claim not in (None, "") and str(claim).strip()
            else "The supplied paired interval supports the frozen comparison."
        )
        evidence_status = f"<strong>Superiority supported.</strong> {_escape(supplied_claim)}"
    else:
        evidence_status = (
            "<strong>Superiority was not demonstrated.</strong> "
            "The supplied two-sided 95% Student-t interval does not support a superiority claim."
        )

    if supported_value is True:
        supported_display = "true"
    elif supported_value is False:
        supported_display = "false"
    else:
        supported_display = _format_value(supported_value)

    rows = (
        ("Transfer method", transfer_label),
        ("Frozen primary comparator", comparator_label),
        ("Paired mean fraction-reference AUC delta", _format_auc_delta(mean_delta)),
        ("Two-sided 95% Student-t CI", _format_auc_interval(ci_low, ci_high)),
        (
            "Paired seed / df count",
            f"{_format_value(paired_seeds)} paired seeds / {_format_value(degrees_of_freedom)} df",
        ),
        ("superiority_supported", supported_display),
    )
    table_rows = "".join(
        "<tr>"
        f'<td class="method-name">{_escape(label)}</td>'
        f'<td class="provenance-value">{_escape(value)}</td>'
        "</tr>"
        for label, value in rows
    )

    descriptive_key = _descriptive_strongest_legacy(summary)
    descriptive_note = ""
    if descriptive_key not in (None, ""):
        descriptive_label = _method_label(descriptive_key, labels, _MISSING)
        descriptive_note = (
            '<div class="disclosure primary-evidence-note">'
            "<strong>Descriptive target-side strongest baseline only</strong>"
            f"<p>{_escape(descriptive_label)} is reported as target-selected descriptive context. "
            f"It does not replace the frozen primary comparator, {_escape(comparator_label)}, "
            "in this evidence panel.</p></div>"
        )

    return (
        '<div class="comparison-lead">'
        f'<span class="{delta_class}">{_escape(_format_auc_delta(mean_delta))}</span>'
        '<span class="delta-context">'
        f"<strong>{_escape(transfer_label)}</strong> versus frozen primary comparator "
        f"<strong>{_escape(comparator_label)}</strong>. {evidence_status}"
        "</span></div>"
        '<div class="table-wrap"><table>'
        "<caption>Frozen paired primary endpoint; values and inference fields are rendered as supplied.</caption>"
        '<thead><tr><th scope="col">Evidence field</th>'
        '<th scope="col">Reported value</th></tr></thead>'
        f"<tbody>{table_rows}</tbody></table></div>"
        f"{descriptive_note}"
    )


def _shared_comparison(
    methods: Mapping[str, list[dict[str, float]]], transfer: str | None, cold: str | None
) -> list[tuple[float, dict[str, float], dict[str, float]]]:
    if transfer is None or cold is None or transfer == cold:
        return []
    transfer_points = {point["budget"]: point for point in methods[transfer]}
    cold_points = {point["budget"]: point for point in methods[cold]}
    return [
        (budget, transfer_points[budget], cold_points[budget])
        for budget in sorted(transfer_points.keys() & cold_points.keys())
    ]


def _render_comparison(
    methods: Mapping[str, list[dict[str, float]]],
    labels: Mapping[str, str],
    transfer: str | None,
    cold: str | None,
) -> str:
    comparisons = _shared_comparison(methods, transfer, cold)
    if not comparisons:
        if transfer is None or cold is None:
            reason = (
                "A transfer and cold-start method pair could not be identified from the supplied method "
                "names. Provide transfer_method and cold_method in the summary to make this comparison explicit."
            )
        else:
            reason = (
                f"{labels[transfer]} and {labels[cold]} have no matching reported budgets, so a "
                "like-for-like delta would be misleading."
            )
        return f'<div class="empty-state">{_escape(reason)}</div>'
    assert transfer is not None and cold is not None

    budget, transfer_point, cold_point = comparisons[-1]
    delta = (transfer_point["mean"] - cold_point["mean"]) * 100
    delta_class = "delta negative" if delta < 0 else "delta"
    direction = "higher" if delta >= 0 else "lower"
    rows = []
    for row_budget, transfer_row, cold_row in comparisons:
        row_delta = (transfer_row["mean"] - cold_row["mean"]) * 100
        rows.append(
            "<tr>"
            f'<td class="numeric">{_escape(_format_budget(row_budget))}</td>'
            f'<td class="numeric">{_escape(_format_number(transfer_row["mean"]))}</td>'
            f'<td class="numeric">{_escape(_format_number(cold_row["mean"]))}</td>'
            f'<td class="numeric">{row_delta:+.2f} pp</td>'
            "</tr>"
        )
    return (
        '<div class="comparison-lead">'
        f'<span class="{delta_class}">{delta:+.1f} pp</span>'
        '<span class="delta-context">'
        f"<strong>{_escape(labels[transfer])}</strong> is {direction} than <strong>{_escape(labels[cold])}</strong> "
        f"at their largest shared budget ({_escape(_format_budget(budget))}). Deltas compare supplied means only."
        "</span></div>"
        '<div class="table-wrap"><table>'
        "<caption>Like-for-like transfer comparison at shared target budgets</caption>"
        '<thead><tr><th scope="col">Budget</th>'
        f'<th scope="col">{_escape(labels[transfer])}</th><th scope="col">{_escape(labels[cold])}</th>'
        '<th scope="col">Transfer delta</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _gpu_name(value: Any, fallback: str) -> str:
    mapped = _as_mapping(value)
    if mapped is not None:
        for key in ("device_name", "name", "gpu"):
            if mapped.get(key):
                return str(mapped[key])
        return fallback
    return str(value) if value not in (None, "") else fallback


def _hardware_facts(summary: Mapping[str, Any], role: str) -> tuple[str, Mapping[str, Any]]:
    gpu_value = summary.get(f"{role}_gpu")
    name = _gpu_name(gpu_value, _MISSING)
    facts: dict[str, Any] = {}
    identifiers: set[str] = set()
    gpu_mapping = _as_mapping(gpu_value)
    if gpu_mapping is not None:
        facts.update(gpu_mapping)
        identifiers.update(
            str(gpu_mapping[key])
            for key in ("gpu", "device_name", "name")
            if gpu_mapping.get(key) not in (None, "")
        )
    elif gpu_value not in (None, ""):
        identifiers.add(str(gpu_value))

    hardware = summary.get("hardware")
    if isinstance(hardware, Mapping):
        role_hardware = _as_mapping(hardware.get(role))
        if role_hardware is not None:
            facts.update(role_hardware)
    elif isinstance(hardware, Sequence) and not isinstance(hardware, (str, bytes)):
        for item in hardware:
            candidate = _as_mapping(item)
            if candidate is None:
                continue
            candidate_ids = {
                str(candidate[key])
                for key in ("gpu", "device_name", "name")
                if candidate.get(key) not in (None, "")
            }
            if identifiers & candidate_ids:
                facts.update(candidate)
                break

    explicit_hardware = _as_mapping(summary.get(f"{role}_hardware"))
    if explicit_hardware is not None:
        facts.update(explicit_hardware)
    if facts:
        name = _gpu_name(facts, name)
    return name, facts


def _source_hardware_profiles(
    summary: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    source_hardware = _as_mapping(summary.get("source_hardware"))
    declared_value = summary.get("source_gpus")
    if declared_value is None and source_hardware is not None:
        declared_value = source_hardware.get("gpus")

    declared_names: list[str] = []
    if isinstance(declared_value, Sequence) and not isinstance(declared_value, (str, bytes)):
        declared_names = [
            _gpu_name(value, f"Source {index}")
            for index, value in enumerate(declared_value, start=1)
        ]
    elif declared_value not in (None, ""):
        declared_names = [_gpu_name(declared_value, "Source 1")]

    profiles_value = None if source_hardware is None else source_hardware.get("profiles")
    profiles: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(profiles_value, Sequence) and not isinstance(profiles_value, (str, bytes)):
        for index, value in enumerate(profiles_value):
            mapped = _as_mapping(value)
            facts: Mapping[str, Any] = mapped if mapped is not None else {"profile": value}
            fallback = (
                declared_names[index] if index < len(declared_names) else f"Source {index + 1}"
            )
            profiles.append((_gpu_name(facts, fallback), facts))
        for index in range(len(profiles), len(declared_names)):
            profiles.append((declared_names[index], {}))
    elif isinstance(profiles_value, Mapping):
        for profile_name, value in profiles_value.items():
            mapped = _as_mapping(value)
            facts = mapped if mapped is not None else {"profile": value}
            profiles.append((_gpu_name(facts, str(profile_name)), facts))
    elif declared_names:
        profiles.extend((name, {}) for name in declared_names)
    return profiles


def _render_fact_list(facts: Mapping[str, Any]) -> str:
    preferred = (
        "gpu",
        "device_name",
        "compute_capability",
        "multiprocessor_count",
        "total_memory_gb",
        "memory_gb",
    )
    ordered_keys = [key for key in preferred if key in facts]
    ordered_keys.extend(
        key
        for key in facts
        if key not in ordered_keys and not isinstance(facts[key], (Mapping, list, tuple))
    )
    rows = []
    for key in ordered_keys:
        if key == "device_name":
            continue
        rows.append(
            "<div>"
            f"<dt>{_escape(_human_label(key))}</dt>"
            f"<dd>{_escape(_format_value(facts[key], key))}</dd>"
            "</div>"
        )
    if not rows:
        rows.append(
            "<div><dt>Additional facts</dt><dd>Not supplied; memory, SM count, and compute capability cannot be verified.</dd></div>"
        )
    return f'<dl class="fact-list">{"".join(rows)}</dl>'


def _render_hardware(summary: Mapping[str, Any]) -> str:
    sheets = []
    source_profiles = _source_hardware_profiles(summary)
    if source_profiles:
        profile_count = len(source_profiles)
        for index, (name, facts) in enumerate(source_profiles, start=1):
            role_label = (
                f"Source hardware · profile {index} of {profile_count}"
                if profile_count > 1
                else "Source hardware"
            )
            sheets.append(
                '<article class="panel hardware-sheet source">'
                f'<span class="hardware-role">{_escape(role_label)}</span>'
                f'<h3 class="hardware-name">{_escape(name)}</h3>'
                f"{_render_fact_list(facts)}"
                "</article>"
            )
    else:
        source_name, source_facts = _hardware_facts(summary, "source")
        sheets.append(
            '<article class="panel hardware-sheet source">'
            '<span class="hardware-role">Source hardware</span>'
            f'<h3 class="hardware-name">{_escape(source_name)}</h3>'
            f"{_render_fact_list(source_facts)}"
            "</article>"
        )

    target_name, target_facts = _hardware_facts(summary, "target")
    sheets.append(
        '<article class="panel hardware-sheet target">'
        '<span class="hardware-role">Target hardware · evaluation domain</span>'
        f'<h3 class="hardware-name">{_escape(target_name)}</h3>'
        f"{_render_fact_list(target_facts)}"
        "</article>"
    )
    return f'<div class="hardware-grid">{"".join(sheets)}</div>'


def _field(record: Mapping[str, Any], key: str) -> str:
    return _escape(_format_value(record.get(key), key))


def _render_experiment_matrix(workloads: Any, configs: Any, experiment: Any = None) -> str:
    workload_records = _record_list(workloads)
    config_records = _record_list(configs)
    config_count = _item_count(configs)
    experiment_mapping = _as_mapping(experiment) or {}
    panels = []

    def experiment_keys(key: str) -> list[Any]:
        value = experiment_mapping.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return []

    workload_has_dimensions = any(
        {"family", "m", "n", "k"}.intersection(record) for record in workload_records
    )
    if workload_has_dimensions:
        rows = []
        for index, workload in enumerate(workload_records, start=1):
            rows.append(
                "<tr>"
                f'<td class="numeric">{index:02d}</td>'
                f'<td class="method-name">{_field(workload, "family")}</td>'
                f'<td class="numeric">{_field(workload, "m")}</td>'
                f'<td class="numeric">{_field(workload, "n")}</td>'
                f'<td class="numeric">{_field(workload, "k")}</td>'
                f'<td class="numeric">{_escape(str(config_count) if config_count is not None else _MISSING)}</td>'
                "</tr>"
            )
        panels.append(
            f'<details class="data-disclosure"><summary>Workload matrix · {len(rows)} rows</summary>'
            '<div class="table-wrap"><table><caption>Workload dimensions and candidate-set coverage</caption>'
            '<thead><tr><th scope="col">ID</th><th scope="col">Family</th><th scope="col">M</th>'
            '<th scope="col">N</th><th scope="col">K</th><th scope="col">Candidate configs</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div></details>"
        )
    else:
        workload_keys = [
            record.get("value", record.get("name"))
            for record in workload_records
            if record.get("value", record.get("name")) not in (None, "")
        ] or experiment_keys("workload_keys")
        if workload_keys:
            identifier_rows = "".join(
                "<tr>"
                f'<td class="numeric">{index:02d}</td>'
                f'<td class="identifier">{_escape(_format_value(value))}</td>'
                "</tr>"
                for index, value in enumerate(workload_keys, start=1)
            )
            panels.append(
                f'<details class="data-disclosure"><summary>Workload identifiers · {len(workload_keys)} rows</summary>'
                '<div class="table-wrap"><table><caption>Complete workload identifiers supplied by the experiment</caption>'
                '<thead><tr><th scope="col">ID</th><th scope="col">Workload key</th></tr></thead>'
                f"<tbody>{identifier_rows}</tbody></table></div></details>"
            )
        else:
            count = _item_count(workloads)
            message = (
                f"The summary reports {count} workloads, but their definitions were not supplied."
                if count is not None
                else "No workload definitions were supplied."
            )
            panels.append(f'<div class="empty-state">{_escape(message)}</div>')

    config_has_parameters = any(
        {"block_m", "block_n", "block_k", "num_warps", "num_stages"}.intersection(record)
        for record in config_records
    )
    if config_has_parameters:
        rows = []
        for index, config in enumerate(config_records, start=1):
            rows.append(
                "<tr>"
                f'<td class="numeric">{index:02d}</td>'
                f'<td class="numeric">{_field(config, "block_m")}</td>'
                f'<td class="numeric">{_field(config, "block_n")}</td>'
                f'<td class="numeric">{_field(config, "block_k")}</td>'
                f'<td class="numeric">{_field(config, "num_warps")}</td>'
                f'<td class="numeric">{_field(config, "num_stages")}</td>'
                f'<td class="numeric">{_field(config, "group_m")}</td>'
                "</tr>"
            )
        panels.append(
            f'<details class="data-disclosure"><summary>Launch configurations · {len(rows)} rows</summary>'
            '<div class="table-wrap"><table><caption>Manual Triton launch candidates</caption>'
            '<thead><tr><th scope="col">ID</th><th scope="col">Block M</th><th scope="col">Block N</th>'
            '<th scope="col">Block K</th><th scope="col">Warps</th><th scope="col">Stages</th>'
            '<th scope="col">Group M</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div></details>"
        )
    else:
        config_keys = [
            record.get("value", record.get("name"))
            for record in config_records
            if record.get("value", record.get("name")) not in (None, "")
        ] or experiment_keys("config_keys")
        if config_keys:
            identifier_rows = "".join(
                "<tr>"
                f'<td class="numeric">{index:02d}</td>'
                f'<td class="identifier">{_escape(_format_value(value))}</td>'
                "</tr>"
                for index, value in enumerate(config_keys, start=1)
            )
            panels.append(
                f'<details class="data-disclosure"><summary>Launch configuration identifiers · {len(config_keys)} rows</summary>'
                '<div class="table-wrap"><table><caption>Complete launch-configuration identifiers supplied by the experiment</caption>'
                '<thead><tr><th scope="col">ID</th><th scope="col">Configuration key</th></tr></thead>'
                f"<tbody>{identifier_rows}</tbody></table></div></details>"
            )
        else:
            count = _item_count(configs)
            message = (
                f"The summary reports {count} configurations, but their launch parameters were not supplied."
                if count is not None
                else "No launch configuration definitions were supplied."
            )
            panels.append(f'<div class="empty-state">{_escape(message)}</div>')

    return f'<div class="matrix-stack">{"".join(panels)}</div>'


def _render_raw_table(
    methods: Mapping[str, list[dict[str, float]]],
    labels: Mapping[str, str],
    caption: str = "Values used to draw the budget-efficiency figure",
) -> str:
    rows = []
    for name, points in methods.items():
        for point in points:
            rows.append(
                "<tr>"
                f'<td class="method-name">{_escape(labels[name])}</td>'
                f'<td class="numeric">{_escape(_format_budget(point["budget"]))}</td>'
                f'<td class="numeric">{_escape(_format_number(point["mean"]))}</td>'
                f'<td class="numeric">{_escape(_format_number(point["low"]))}</td>'
                f'<td class="numeric">{_escape(_format_number(point["high"]))}</td>'
                "</tr>"
            )
    if not rows:
        return '<div class="empty-state">No raw method values were supplied.</div>'
    return (
        f'<div class="table-wrap"><table><caption>{_escape(caption)}</caption>'
        '<thead><tr><th scope="col">Method</th><th scope="col">Budget</th>'
        '<th scope="col">Mean fraction of held-out reference</th><th scope="col">CI95 low</th>'
        '<th scope="col">CI95 high</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_fold_source_table(fold: Mapping[str, Any]) -> str:
    visible = _as_mapping(fold.get("visible_bank0_source_observations_by_gpu")) or {}
    excluded = _as_mapping(fold.get("excluded_exact_target_shapes_by_gpu")) or {}
    sources = list(dict.fromkeys([*(str(key) for key in visible), *(str(key) for key in excluded)]))
    if not sources:
        return (
            '<div class="empty-state fold-source-table">'
            "No per-source visibility or exact-shape exclusion counts were supplied for this fold."
            "</div>"
        )

    visible_by_name = {str(key): value for key, value in visible.items()}
    excluded_by_name = {str(key): value for key, value in excluded.items()}
    rows = "".join(
        "<tr>"
        f'<td class="method-name">{_escape(source)}</td>'
        f'<td class="numeric">{_escape(_format_value(visible_by_name.get(source)))}</td>'
        f'<td class="numeric">{_escape(_format_value(excluded_by_name.get(source)))}</td>'
        "</tr>"
        for source in sources
    )
    return (
        '<div class="table-wrap panel-table fold-source-table"><table>'
        "<caption>Per-source fold visibility and exact-target-shape exclusion audit</caption>"
        '<thead><tr><th scope="col">Source GPU</th>'
        '<th scope="col">Visible bank-0 source rows</th>'
        '<th scope="col">Excluded exact target shapes</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _render_fold_results(summary: Mapping[str, Any]) -> str:
    value = summary.get("fold_results")
    if value is None:
        return ""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("summary['fold_results'] must be a sequence")

    panels = []
    for index, raw_fold in enumerate(value, start=1):
        fold = _as_mapping(raw_fold)
        if fold is None:
            raise TypeError(f"summary['fold_results'][{index - 1}] must be a mapping")
        fold_methods = _normalise_methods(fold.get("methods", {}))
        fold_labels = _method_display_labels(summary, fold_methods)
        family = _format_value(fold.get("heldout_model"))
        target_workloads = _format_value(fold.get("target_workloads"))
        row_count = sum(len(points) for points in fold_methods.values())
        heading_id = f"fold-result-{index}"
        method_table = _render_raw_table(
            fold_methods,
            fold_labels,
            f"Complete supplied method values for held-out family {family}",
        )
        panels.append(
            f'<article class="panel fold-audit" aria-labelledby="{heading_id}">'
            '<header class="fold-heading"><div>'
            f'<p class="micro-label">Fold {index:02d} · held-out family</p>'
            f'<h3 id="{heading_id}">{_escape(family)}</h3></div>'
            '<p class="fold-target-count">'
            f"<strong>{_escape(target_workloads)}</strong>"
            "<span>Target workloads</span></p></header>"
            f"{_render_fold_source_table(fold)}"
            '<details class="data-disclosure">'
            f"<summary>Complete method-by-budget numeric table · {row_count} rows</summary>"
            f"{method_table}</details></article>"
        )
    return f'<div class="fold-stack">{"".join(panels)}</div>' if panels else ""


def _flatten_details(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    details: list[tuple[str, Any]] = []
    for key, item in value.items():
        label = f"{prefix} / {_human_label(key)}" if prefix else _human_label(key)
        if isinstance(item, Mapping):
            details.extend(_flatten_details(item, label))
        else:
            details.append((label, item))
    return details


def _render_release_provenance(value: Any) -> str:
    if value is None:
        return ""
    release = _as_mapping(value)
    if release is None:
        raise TypeError("summary['release_provenance'] must be a mapping")

    stages = (
        (
            "01 · Freeze",
            (
                ("Algorithm commit", "algorithm_commit"),
                ("Freeze commit", "freeze_commit"),
                ("Freeze SHA-256", "freeze_sha256"),
            ),
        ),
        (
            "02 · Collection",
            (
                ("Sole H100 run", "sole_h100_run"),
                ("Raw H100 SHA-256", "raw_h100_sha256"),
            ),
        ),
        (
            "03 · Release",
            (
                ("Final archive SHA-256", "final_archive_sha256"),
                ("Post-run manifest path", "post_run_manifest_path"),
            ),
        ),
    )
    items = []
    for stage, fields in stages:
        facts = "".join(
            "<div>"
            f"<dt>{_escape(label)}</dt>"
            f"<dd>{_escape(_format_value(release.get(key)))}</dd>"
            "</div>"
            for label, key in fields
        )
        items.append(
            "<li>"
            f'<span class="custody-stage">{_escape(stage)}</span>'
            f'<dl class="custody-facts">{facts}</dl>'
            "</li>"
        )
    return (
        '<aside class="custody-panel" aria-labelledby="release-provenance-title">'
        '<p class="micro-label">caller-supplied release provenance</p>'
        '<h3 id="release-provenance-title">Release provenance</h3>'
        '<p class="scope-note">not independently authenticated by summary</p>'
        f'<ol class="custody-chain">{"".join(items)}</ol></aside>'
    )


def _render_reproducibility(
    summary: Mapping[str, Any], methods: Mapping[str, list[dict[str, float]]]
) -> str:
    release_panel = _render_release_provenance(summary.get("release_provenance"))
    workload_count = _item_count(summary.get("workloads"))
    config_count = _item_count(summary.get("configs"))
    status_label, _ = _data_status(summary)
    details: list[tuple[str, Any]] = [
        ("Data classification", status_label),
        ("Reported metric", "mean_fraction_oracle with supplied 95% confidence bounds"),
        ("Method count", len(methods)),
        ("Reported curve points", sum(len(points) for points in methods.values())),
        ("Workloads", workload_count if workload_count is not None else _MISSING),
        ("Configurations", config_count if config_count is not None else _MISSING),
    ]

    experiment = _as_mapping(summary.get("experiment"))
    if experiment is not None:
        compact_experiment = {
            key: value
            for key, value in experiment.items()
            if key not in {"workload_keys", "config_keys"}
        }
        details.extend(_flatten_details(compact_experiment, "Experiment"))

    reproducibility = summary.get("reproducibility")
    if isinstance(reproducibility, Mapping):
        details.extend(_flatten_details(reproducibility, "Run"))
    elif reproducibility not in (None, ""):
        details.append(("Reproducibility notes", reproducibility))

    known_fields = (
        "generated_at",
        "seed",
        "seeds",
        "repetitions",
        "model_families",
        "measurement_banks",
        "max_budget",
        "transfer_strength",
        "warmup_iterations",
        "measurement_iterations",
        "timing_protocol",
        "correctness_protocol",
        "commit",
        "software_versions",
    )
    for field in known_fields:
        if field in summary:
            details.append((_human_label(field), summary[field]))

    rows = "".join(
        "<tr>"
        f'<td class="method-name">{_escape(label)}</td>'
        f'<td class="provenance-value">{_escape(_format_value(value))}</td>'
        "</tr>"
        for label, value in details
    )
    metadata_supplied = (
        experiment is not None
        or isinstance(reproducibility, Mapping)
        or any(field in summary for field in known_fields)
        or bool(release_panel)
    )
    note = (
        ""
        if metadata_supplied
        else (
            '<div class="empty-state">No run metadata was supplied. Seed, repetition count, software versions, '
            "timing protocol, correctness protocol, and source revision cannot be verified from this report.</div>"
        )
    )
    table = (
        '<div class="table-wrap panel-table"><table>'
        "<caption>Protocol, aggregation, and run metadata supplied with this report</caption>"
        '<thead><tr><th scope="col">Field</th><th scope="col">Reported value</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )
    return f"{release_panel}{table}{note}"


def _source_cost(summary: Mapping[str, Any]) -> Any:
    for key in ("source_cost", "source_collection_cost", "source_tuning_cost"):
        if key in summary:
            return summary[key]
    costs = summary.get("costs")
    if isinstance(costs, Mapping) and "source" in costs:
        return costs["source"]
    return None


def _render_cost_disclosure(summary: Mapping[str, Any]) -> str:
    cost = _source_cost(summary)
    if cost is None:
        return (
            '<div class="disclosure"><strong>Source collection cost is not included.</strong>'
            "<p>The payload does not report the evaluations, runtime, or monetary cost used to produce the "
            "source posterior. Transfer curves therefore describe target-budget efficiency only and must not "
            "be read as end-to-end cost savings.</p></div>"
        )
    if isinstance(cost, Mapping):
        facts = _render_fact_list(cost)
        return (
            '<div class="disclosure"><strong>Reported source collection cost</strong>'
            "<p>This cost is disclosed separately from the target evaluation budgets in the curves.</p>"
            f"{facts}</div>"
        )
    return (
        '<div class="disclosure"><strong>Reported source collection cost</strong>'
        f"<p>{_escape(_format_value(cost))}. This value is separate from target evaluation budgets.</p></div>"
    )


def _render_budget_disclosure(summary: Mapping[str, Any]) -> str:
    details: dict[str, Any] = {}
    experiment = _as_mapping(summary.get("experiment"))
    if experiment is not None:
        for key in (
            "target_budget_unit",
            "adaptation_scope",
            "budget_disclosure",
            "cross_workload_feedback",
        ):
            if experiment.get(key) not in (None, ""):
                details[f"experiment_{key}"] = experiment[key]

    target_cost_value = summary.get("target_collection_cost")
    if target_cost_value is None:
        costs = _as_mapping(summary.get("costs"))
        if costs is not None:
            target_cost_value = costs.get("target")
    target_cost = _as_mapping(target_cost_value)
    if target_cost is not None:
        for key in (
            "simulated_online_queries_per_live_method_per_workload",
            "budget_b_formula",
            "adaptation_scope",
            "disclosure",
        ):
            if target_cost.get(key) not in (None, ""):
                details[f"target_cost_{key}"] = target_cost[key]
    elif target_cost_value not in (None, ""):
        details["target_collection_cost"] = target_cost_value

    if not details:
        return ""
    return (
        '<div class="disclosure"><strong>Supplied target-budget and posterior scope</strong>'
        "<p>These statements are reproduced from the experiment and cost metadata.</p>"
        f"{_render_fact_list(details)}</div>"
    )


def _render_limitations(summary: Mapping[str, Any]) -> str:
    limitations = [
        "Fraction of the held-out reference is an aggregate efficiency measure; without workload-level distributions, it can hide regressions on individual matrix shapes.",
        "Confidence intervals communicate the supplied sampling uncertainty but do not by themselves establish independence, causal transfer benefit, or performance outside this matrix.",
        "Conclusions are bounded to the reported GPUs, workloads, candidate launch configurations, and FP16 Triton matmul implementation.",
    ]
    if str(summary.get("data_kind", "")).strip().lower() == "synthetic":
        limitations.insert(
            0,
            "Synthetic values do not establish measured latency, throughput, or transfer behavior on the named hardware.",
        )
    supplied = summary.get("limitations")
    if isinstance(supplied, str) and supplied.strip():
        limitations.append(supplied.strip())
    elif isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        limitations.extend(str(item) for item in supplied if str(item).strip())
    return (
        '<ul class="limitations">'
        + "".join(f"<li>{_escape(item)}</li>" for item in limitations)
        + "</ul>"
    )


def _declared_online_budget(summary: Mapping[str, Any]) -> float | None:
    primary_metrics = _as_mapping(summary.get("primary_metrics"))
    headline = _as_mapping(summary.get("headline"))
    candidates = (
        None if primary_metrics is None else primary_metrics.get("primary_budget"),
        summary.get("max_budget"),
        None if headline is None else headline.get("budget"),
    )
    for candidate in candidates:
        number = _supplied_number(candidate)
        if number is not None and number >= 0:
            return number
    return None


def _headline_values(
    summary: Mapping[str, Any],
    methods: Mapping[str, list[dict[str, float]]],
    labels: Mapping[str, str],
) -> list[tuple[str, str]]:
    endpoints = [(name, points[-1]) for name, points in methods.items() if points]
    if endpoints:
        best_name, best_point = max(endpoints, key=lambda item: item[1]["mean"])
        best_value = f"{best_point['mean']:.1%}"
        best_label = (
            f"Best endpoint · {labels[best_name]} at budget {_format_budget(best_point['budget'])}"
        )
    else:
        best_value, best_label = _MISSING, "Best reported endpoint"

    online_budget = _declared_online_budget(summary)
    if online_budget is not None:
        budget_value = _format_budget(online_budget)
        budget_label = "Declared online target budget"
    elif endpoints:
        largest_point_budget = max(
            point["budget"] for points in methods.values() for point in points
        )
        budget_value = _format_budget(largest_point_budget)
        budget_label = "Largest target budget reported"
    else:
        budget_value, budget_label = _MISSING, "Largest target budget reported"

    headline = _as_mapping(summary.get("headline"))
    transfer = _method_by_role(summary, methods, "transfer")
    transfer_key = summary.get("transfer_method", transfer)
    transfer_name = _method_label(transfer_key, labels, "Transfer")
    paired_metric_present, paired_metric = _paired_primary_metric(summary)
    if paired_metric is not None:
        comparator_key = paired_metric.get("comparator")
        if comparator_key in (None, ""):
            comparator_key = summary.get("primary_comparator")
        comparator_name = _method_label(comparator_key, labels, _MISSING)
        paired_mean = _supplied_number(paired_metric.get("mean_auc_delta"))
        paired_low = _supplied_number(paired_metric.get("ci95_low"))
        paired_high = _supplied_number(paired_metric.get("ci95_high"))
        transfer_value = f"{paired_mean * 100:+.1f} pp AUC" if paired_mean is not None else _MISSING
        if paired_low is not None and paired_high is not None:
            interval = f"[{paired_low * 100:+.1f}, {paired_high * 100:+.1f}] pp"
            transfer_label = (
                f"{transfer_name} − frozen {comparator_name}; two-sided 95% Student-t CI {interval}"
            )
        else:
            transfer_label = f"{transfer_name} − frozen {comparator_name}; paired primary AUC delta"
    elif paired_metric_present:
        transfer_value = _MISSING
        transfer_label = (
            f"{transfer_name}; frozen paired primary comparison not reported "
            "(cold endpoint not substituted)"
        )
    else:
        strongest_legacy = None if headline is None else headline.get("strongest_legacy_method")
        auc_delta = None if headline is None else headline.get("auc_delta_vs_strongest_legacy")
        if (
            isinstance(strongest_legacy, str)
            and strongest_legacy in labels
            and isinstance(auc_delta, (int, float))
        ):
            transfer_value = f"{float(auc_delta) * 100:+.1f} pp AUC"
            transfer_label = (
                f"{transfer_name} − {labels[strongest_legacy]} across the reported budget curve"
            )
        else:
            cold = _method_by_role(summary, methods, "cold")
            comparisons = _shared_comparison(methods, transfer, cold)
            if comparisons:
                comparison_budget, transfer_point, cold_point = comparisons[-1]
                delta = (transfer_point["mean"] - cold_point["mean"]) * 100
                transfer_value = f"{delta:+.1f} pp"
                transfer_label = (
                    f"{transfer_name} − {labels.get(cold or '', 'cold start')} "
                    f"at shared budget {_format_budget(comparison_budget)}"
                )
            else:
                transfer_value = _MISSING
                transfer_label = "Transfer versus cold start at a shared target budget"

    workload_count = _item_count(summary.get("workloads"))
    config_count = _item_count(summary.get("configs"))
    if workload_count is not None and config_count is not None:
        surface_value = f"{workload_count} W / {config_count} C"
    else:
        surface_value = _MISSING
    return [
        (transfer_value, transfer_label),
        (best_value, best_label),
        (budget_value, budget_label),
        (surface_value, "Reported workloads / launch configs"),
    ]


def _signal_band(values: Sequence[tuple[str, str]]) -> str:
    return (
        '<div class="result-strip" aria-label="Primary reported results">'
        + "".join(
            '<div class="result">'
            f'<span class="result-value">{_escape(value)}</span>'
            f'<span class="result-label">{_escape(label)}</span>'
            "</div>"
            for value, label in values
        )
        + "</div>"
    )


def _section(index: str, title: str, copy: str, content: str) -> str:
    heading_id = f"section-{re.sub(r'[^a-z0-9]+', '-', index.lower()).strip('-')}"
    return (
        f'<section class="section" aria-labelledby="{heading_id}">'
        '<header class="section-head">'
        f'<p class="section-number">{_escape(index)}</p>'
        "<div>"
        f'<h2 id="{heading_id}">{_escape(title)}</h2>'
        f'<p class="section-copy">{_escape(copy)}</p>'
        "</div></header>"
        f"{content}</section>"
    )


def _render_control_panel(data: ReportData) -> str:
    controls = [
        method for method in data.methods if method.role in {"zero_query", "external", "exhaustive"}
    ]
    if not controls:
        return ""
    cards: list[str] = []
    for method in controls:
        if not method.points:
            continue
        endpoint = method.points[-1]
        if method.role == "exhaustive":
            budget = f"Endpoint at {_format_budget(endpoint.budget)} measured configurations"
        elif method.role == "external":
            budget = "External implementation control · no tuner action-space budget"
        else:
            budget = "Zero-query control · no target policy probes"
        uncertainty = endpoint.uncertainty
        cards.append(
            '<article class="metric-card control-card">'
            f'<p class="eyebrow">{_escape(method.role.replace("_", " "))}</p>'
            f"<h3>{_escape(method.label)}</h3>"
            f'<p class="metric-card-value">{_escape(_format_number(endpoint.mean))}</p>'
            f"<p>{_escape(budget)}</p>"
            f'<p class="fine-print">{_escape(uncertainty.interval_method)}; '
            f"n={uncertainty.n} {_escape(uncertainty.sampling_unit)}; conditional on "
            f"{_escape(uncertainty.conditional_on)}.</p>"
            "</article>"
        )
    if not cards:
        return ""
    return (
        '<section class="control-panel" aria-labelledby="control-panel-title">'
        '<header><p class="eyebrow">Separate action spaces</p>'
        '<h3 id="control-panel-title">Zero-query, external, and exhaustive controls</h3></header>'
        f'<div class="metric-grid">{"".join(cards)}</div></section>'
    )


def _render_uncertainty_notes(data: ReportData) -> str:
    items: list[str] = []
    for method in data.methods:
        if method.role != "sequential" or not method.points:
            continue
        uncertainty = method.points[-1].uncertainty
        items.append(
            "<li>"
            f"<strong>{_escape(method.label)}</strong>: "
            f"{_escape(uncertainty.interval_method)} over n={uncertainty.n} "
            f"{_escape(uncertainty.sampling_unit)}; conditional on "
            f"{_escape(uncertainty.conditional_on)}."
            "</li>"
        )
    return (
        '<details class="data-disclosure uncertainty-disclosure">'
        "<summary>Interval estimands and conditioning</summary>"
        f'<ul class="audit-list">{"".join(items)}</ul></details>'
        if items
        else ""
    )


def render_report(summary: Mapping[str, Any], output_path: str | Path) -> None:
    """Normalize once, then atomically render a complete offline research report."""
    data = normalize_report_summary(summary)
    summary = data.raw_summary
    methods = {
        method.key: [
            {
                "budget": point.budget,
                "mean": point.mean,
                "low": point.uncertainty.low,
                "high": point.uncertainty.high,
            }
            for point in method.points
        ]
        for method in data.methods
    }
    labels = {method.key: method.label for method in data.methods}
    roles = {method.key: method.role for method in data.methods}
    sequential_methods = {
        key: points for key, points in methods.items() if roles[key] == "sequential"
    }
    source_profiles = _source_hardware_profiles(summary)
    if source_profiles:
        source_name = " + ".join(name for name, _ in source_profiles)
        source_role_label = (
            f"Source contexts · {len(source_profiles)}"
            if len(source_profiles) > 1
            else "Source context"
        )
    else:
        source_name, _ = _hardware_facts(summary, "source")
        source_role_label = "Source context"
    target_name, _ = _hardware_facts(summary, "target")
    transfer = _method_by_role(summary, methods, "transfer")
    cold = _method_by_role(summary, methods, "cold")
    data_label, data_copy = _data_status(summary)
    methodology = summary.get(
        "methodology",
        "Methods are evaluated over a target-side configuration budget. Curves report the supplied mean fraction of a held-out reference and 95% confidence interval; higher is better and 1.0 marks reference parity.",
    )
    if not isinstance(methodology, str):
        methodology = _format_value(methodology)

    experiment = _as_mapping(summary.get("experiment")) or {}
    workload_count = _item_count(summary.get("workloads"))
    config_count = _item_count(summary.get("configs"))
    surface = (
        f"{workload_count} workloads × {config_count} configurations"
        if workload_count is not None and config_count is not None
        else _MISSING
    )
    budget_unit = experiment.get("target_budget_unit", "Target-side configuration evaluations")
    metric = experiment.get("aggregation", "Mean fraction of held-out reference")
    curve_points = sum(len(points) for points in methods.values())

    chart = (
        '<figure class="chart-frame sequential-chart">'
        '<div class="chart-scroll">'
        f"{_render_chart(sequential_methods, labels)}"
        "</div>"
        f"{_render_legend(sequential_methods, labels)}"
        "<figcaption>Only methods that pay target probes appear on this online-budget axis. "
        "The horizontal reference-parity line is not a queried policy. Lines connect supplied "
        "means; bands encode validated uncertainty bounds.</figcaption>"
        "</figure>"
        f"{_render_uncertainty_notes(data)}"
        f"{_render_control_panel(data)}"
        f'<details class="data-disclosure"><summary>Complete numeric results · {curve_points} rows</summary>'
        f"{_render_raw_table(methods, labels)}</details>"
    )
    paired_metric_present, _ = _paired_primary_metric(summary)
    if paired_metric_present:
        comparison_title = "Frozen primary-comparison evidence"
        comparison_copy = (
            "The supplied paired AUC delta is tied to the frozen comparator. "
            "The target-selected strongest legacy method remains descriptive only."
        )
        comparison_content = _render_primary_evidence(summary, labels, transfer)
    else:
        comparison_title = "Transfer versus cold start"
        comparison_copy = (
            "Only matching target budgets are compared. A positive percentage-point delta means "
            "the identified transfer method has the higher supplied mean."
        )
        comparison_content = _render_comparison(methods, labels, transfer, cold)

    cost_and_limits = (
        '<div class="split-grid"><div class="matrix-stack">'
        f"{_render_cost_disclosure(summary)}"
        f"{_render_budget_disclosure(summary)}</div>"
        '<div class="panel"><h3>Interpretation boundaries</h3>'
        f"{_render_limitations(summary)}</div></div>"
    )
    fold_content = _render_fold_results(summary)
    if fold_content:
        fold_section = _section(
            "03",
            "Fold-level evidence audit",
            (
                "Every held-out family exposes its target count, source rows visible to bank 0, "
                "exact-target-shape exclusions, and all supplied method points. This reporting-only "
                "decomposition does not alter the frozen primary comparator; the target-selected "
                "strongest legacy method remains descriptive only."
            ),
            fold_content,
        )
        hardware_index, scope_index, provenance_index, limits_index = "04", "05", "06", "07"
    else:
        fold_section = ""
        hardware_index, scope_index, provenance_index, limits_index = "03", "04", "05", "06"
    provenance_copy = (
        (
            "Reported run metadata, caller-supplied release provenance, and experiment protocol "
            "are inventoried separately from facts that are absent from the summary. Release "
            "provenance is not independently authenticated by summary."
        )
        if summary.get("release_provenance") is not None
        else (
            "Reported run metadata and experiment protocol are inventoried separately from "
            "facts that are absent from the summary."
        )
    )

    serialized_chart_data = {
        name: [
            {
                "budget": point["budget"],
                "mean_fraction_oracle": point["mean"],
                "ci95_low": point["low"],
                "ci95_high": point["high"],
            }
            for point in points
        ]
        for name, points in methods.items()
    }
    data_blob = _safe_json(serialized_chart_data)
    signals = _signal_band(_headline_values(summary, methods, labels))
    source_target = f"{source_name} → {target_name}"
    title = f"HeliosTune transfer report · {source_target}"

    document = (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'none'; base-uri 'none'; form-action 'none'\">"
        f"<title>{_escape(title)}</title>"
        f"<style>{_styles()}</style></head><body>"
        '<a class="skip-link" href="#main">Skip to results</a>'
        '<div class="page"><header class="report-header">'
        '<div class="utility-rail">'
        '<span class="project-lockup">HeliosTune // transfer study</span>'
        f'<span class="data-flag">{_escape(data_label)}</span>'
        "</div>"
        '<div class="identity-layout"><div>'
        '<p class="kicker">GPU kernel autotuning evidence record</p>'
        "<h1>Autotuning transfer across GPU targets</h1>"
        f'<div class="route" aria-label="Transfer direction from {_escape(source_name)} to {_escape(target_name)}">'
        f'<div class="route-node"><span>{_escape(source_role_label)}</span>'
        f"<strong>{_escape(source_name)}</strong></div>"
        '<span class="route-arrow" aria-hidden="true">→</span>'
        '<div class="route-node target"><span>Target context · evaluation domain</span>'
        f"<strong>{_escape(target_name)}</strong></div></div>"
        "</div>"
        '<aside class="method-note" aria-label="Methodology summary">'
        '<p class="micro-label">Methodology</p>'
        f'<p class="method-copy">{_escape(methodology)}</p>'
        "</aside></div>"
        '<dl class="meta-strip">'
        "<div><dt>Target metric</dt>"
        f"<dd>{_escape(_format_value(metric))} · higher is better</dd></div>"
        "<div><dt>Curve inventory</dt>"
        f"<dd>{len(methods)} methods · {curve_points} points</dd></div>"
        "<div><dt>Experiment surface</dt>"
        f"<dd>{_escape(surface)}</dd></div>"
        "<div><dt>Budget unit</dt>"
        f"<dd>{_escape(_format_value(budget_unit))}</dd></div>"
        "</dl>"
        f'<p class="status-copy"><strong>{_escape(data_label)}.</strong> {_escape(data_copy)}</p>'
        "</header>"
        f'<main id="main">{signals}'
        f"{_section('01', 'Budget-efficiency curves', 'The online chart contains only methods that pay target probes. Zero-query, external, and exhaustive controls remain in a separate panel; reference parity is horizontal.', chart)}"
        f"{_section('02', comparison_title, comparison_copy, comparison_content)}"
        f"{fold_section}"
        f"{_section(hardware_index, 'Hardware context', 'Each supplied source profile is shown separately from the target evaluation domain. Facts are rendered exactly as supplied.', _render_hardware(summary))}"
        f"{_section(scope_index, 'Experiment scope', 'The workload and launch-configuration inventories bound the reported reference and every curve. Large inventories are contained in local, expandable tables.', _render_experiment_matrix(summary['workloads'], summary['configs'], summary.get('experiment')))}"
        f"{_section(provenance_index, 'Protocol and provenance', provenance_copy, _render_reproducibility(summary, methods))}"
        f"{_section(limits_index, 'Cost, budget scope, and limitations', 'Source acquisition, supplied target-budget accounting, posterior scope, and interpretation boundaries remain separate from the reported curves.', cost_and_limits)}"
        "</main>"
        '<footer class="footer"><span><strong>HeliosTune</strong> / GPU autotuning transfer report</span>'
        "<span>Offline HTML · inline SVG · no network requests</span></footer></div>"
        f'<script type="application/json" id="heliostune-chart-data">{data_blob}</script>'
        "</body></html>"
    )

    write_text_atomic(output_path, document)
