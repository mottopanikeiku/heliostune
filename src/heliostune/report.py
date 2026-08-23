"""Self-contained HTML reporting for HeliosTune tuning summaries."""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

_STYLES = """
:root {
  color-scheme: light;
  --canvas: #f1f3f0;
  --paper: #fafbf9;
  --paper-muted: #f5f7f4;
  --ink: #18211c;
  --ink-strong: #0d1711;
  --muted: #55615a;
  --subtle: #737f78;
  --line: #d9dedb;
  --line-strong: #b9c2bc;
  --accent: #1f5d99;
  --accent-soft: #e6eef6;
  --note: #eef0ec;
  --negative: #8f3f35;
  --series-0: #1f5d99;
  --series-1: #b85c2d;
  --series-2: #287460;
  --series-3: #6e5b9c;
  --series-4: #85651c;
  --series-5: #3f6f86;
  --series-6: #a04658;
  --series-7: #4c6a3d;
  --series-8: #715047;
  --series-9: #52657d;
  --font-sans: "IBM Plex Sans", Aptos, "Segoe UI", "Helvetica Neue", sans-serif;
  --font-mono: "IBM Plex Mono", "SFMono-Regular", "Cascadia Code", Consolas, monospace;
  --text-2xs: 0.6875rem;
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-md: 1rem;
  --text-lg: 1.125rem;
  --text-title: clamp(1.6rem, 2.5vw, 2.15rem);
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-7: 3rem;
  --space-8: 4rem;
  --radius-sm: 0.2rem;
  --radius-md: 0.35rem;
  --page-max: 80rem;
  --chart-min: 44rem;
  --focus-ring: 0 0 0 0.1875rem rgba(31, 93, 153, 0.24);
}

* { box-sizing: border-box; }
html { background: var(--canvas); }
body {
  margin: 0;
  min-width: 0;
  background: var(--canvas);
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: var(--text-md);
  line-height: 1.5;
}
a {
  color: var(--accent);
  text-decoration-thickness: 0.0625rem;
  text-underline-offset: var(--space-1);
}
a:focus-visible, summary:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}
.skip-link {
  position: fixed;
  inset: var(--space-3) auto auto var(--space-3);
  z-index: 20;
  padding: var(--space-2) var(--space-3);
  background: var(--ink-strong);
  color: var(--paper);
  transform: translateY(-160%);
}
.skip-link:focus { transform: translateY(0); }
.page {
  width: 100%;
  max-width: var(--page-max);
  margin: 0 auto;
  padding: 0 var(--space-6) var(--space-7);
}
.report-header {
  padding: var(--space-5) 0 var(--space-4);
  border-top: var(--space-1) solid var(--accent);
}
.utility-rail {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding-bottom: var(--space-3);
  border-bottom: 1px solid var(--line);
  color: var(--subtle);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.project-lockup { color: var(--ink-strong); font-weight: 700; }
.data-flag {
  color: var(--accent);
  font-weight: 700;
  text-align: right;
}
.identity-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(18rem, 0.65fr);
  gap: var(--space-6);
  align-items: end;
  padding: var(--space-5) 0 var(--space-4);
}
.identity-layout > * { min-width: 0; }
.kicker, .micro-label, .section-number, .hardware-role {
  margin: 0;
  color: var(--subtle);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
h1, h2, h3 {
  margin-top: 0;
  color: var(--ink-strong);
  letter-spacing: -0.015em;
}
h1 {
  max-width: 34ch;
  margin: var(--space-2) 0 var(--space-4);
  font-size: var(--text-title);
  font-weight: 680;
  line-height: 1.12;
}
h2 {
  margin-bottom: var(--space-2);
  font-size: clamp(1.25rem, 2vw, 1.55rem);
  font-weight: 680;
  line-height: 1.2;
}
h3 {
  margin-bottom: var(--space-3);
  font-size: var(--text-md);
  font-weight: 680;
}
.route {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: var(--space-3);
  align-items: center;
  max-width: 48rem;
}
.route-node {
  min-width: 0;
  padding: var(--space-2) 0;
  border-block: 1px solid var(--line-strong);
}
.route-node span {
  display: block;
  color: var(--subtle);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.route-node strong {
  display: block;
  overflow-wrap: anywhere;
  color: var(--ink-strong);
  font-size: var(--text-sm);
  font-weight: 650;
}
.route-arrow {
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: var(--text-lg);
}
.method-note {
  padding-left: var(--space-4);
  border-left: 2px solid var(--accent);
}
.method-note p { margin: 0; }
.method-note .method-copy {
  margin-top: var(--space-2);
  color: var(--muted);
  font-size: var(--text-sm);
}
.meta-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 0;
  border-block: 1px solid var(--line-strong);
}
.meta-strip div {
  min-width: 0;
  padding: var(--space-3) var(--space-4);
  border-left: 1px solid var(--line);
}
.meta-strip div:first-child { border-left: 0; }
.meta-strip dt {
  color: var(--subtle);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.meta-strip dd {
  margin: var(--space-1) 0 0;
  overflow-wrap: anywhere;
  color: var(--ink-strong);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
}
.status-copy {
  margin: var(--space-3) 0 0;
  color: var(--muted);
  font-size: var(--text-xs);
}
.result-strip {
  display: grid;
  grid-template-columns: minmax(14rem, 1.35fr) repeat(3, minmax(0, 1fr));
  margin: var(--space-5) 0 var(--space-6);
  border-block: 1px solid var(--line-strong);
}
.result {
  min-width: 0;
  padding: var(--space-3) var(--space-4);
  border-left: 1px solid var(--line);
}
.result:first-child {
  border-left: var(--space-1) solid var(--accent);
  background: var(--accent-soft);
}
.result-value {
  display: block;
  margin-bottom: var(--space-1);
  overflow-wrap: anywhere;
  color: var(--ink-strong);
  font-family: var(--font-mono);
  font-size: var(--text-lg);
  font-weight: 700;
  line-height: 1.25;
}
.result:first-child .result-value { color: var(--accent); font-size: 1.35rem; }
.result-label { display: block; color: var(--muted); font-size: var(--text-xs); }
.section {
  min-width: 0;
  padding: var(--space-6) 0;
  border-top: 1px solid var(--line);
}
.section-head {
  display: grid;
  grid-template-columns: 7rem minmax(0, 1fr);
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}
.section-head > * { min-width: 0; }
.section-copy {
  max-width: 78ch;
  margin: 0;
  color: var(--muted);
  font-size: var(--text-sm);
}
.panel {
  min-width: 0;
  padding: var(--space-4);
  border: 1px solid var(--line);
  background: var(--paper);
  border-radius: var(--radius-md);
}
.panel + .panel { margin-top: var(--space-4); }
.chart-frame {
  min-width: 0;
  margin: 0;
  border: 1px solid var(--line);
  background: var(--paper);
}
.chart-scroll {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overscroll-behavior-inline: contain;
  padding: var(--space-2) var(--space-2) 0;
  scrollbar-color: var(--line-strong) var(--paper-muted);
}
.chart-scroll svg {
  display: block;
  width: 100%;
  min-width: var(--chart-min);
  height: auto;
}
.chart-grid {
  stroke: var(--line);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}
.chart-axis {
  stroke: var(--line-strong);
  stroke-width: 1.2;
  vector-effect: non-scaling-stroke;
}
.chart-text {
  fill: var(--muted);
  font-family: var(--font-mono);
  font-size: 12px;
}
.chart-label {
  fill: var(--ink);
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 650;
}
.oracle-line {
  stroke: var(--subtle);
  stroke-dasharray: 3 6;
  vector-effect: non-scaling-stroke;
}
.series-line {
  fill: none;
  stroke-width: 2.4;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}
.series-band { stroke: none; opacity: 0.11; }
.series-point {
  stroke: var(--paper);
  stroke-width: 1.6;
  vector-effect: non-scaling-stroke;
}
.method-legend {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
  gap: var(--space-2) var(--space-4);
  margin: 0;
  padding: var(--space-3) var(--space-4);
  border-top: 1px solid var(--line);
  list-style: none;
}
.method-legend li {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--space-2);
  align-items: center;
  min-width: 0;
  color: var(--muted);
  font-size: var(--text-xs);
}
.method-legend svg { flex: 0 0 auto; }
.method-legend strong {
  display: block;
  overflow-wrap: anywhere;
  color: var(--ink);
  font-weight: 650;
}
.legend-endpoint {
  display: block;
  color: var(--subtle);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
}
figcaption {
  padding: var(--space-3) var(--space-4);
  border-top: 1px solid var(--line);
  color: var(--subtle);
  font-size: var(--text-xs);
}
.data-disclosure {
  min-width: 0;
  margin-top: var(--space-3);
  border: 1px solid var(--line);
  background: var(--paper);
}
.data-disclosure summary {
  cursor: pointer;
  padding: var(--space-3) var(--space-4);
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: 650;
}
.data-disclosure[open] summary { border-bottom: 1px solid var(--line); }
.table-wrap {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overscroll-behavior-inline: contain;
  scrollbar-color: var(--line-strong) var(--paper-muted);
}
.panel-table {
  border: 1px solid var(--line);
  background: var(--paper);
}
table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}
caption {
  padding: var(--space-3) var(--space-4);
  color: var(--muted);
  font-size: var(--text-xs);
  text-align: left;
}
th, td {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--paper-muted);
  color: var(--ink-strong);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
td { color: var(--muted); font-size: var(--text-xs); }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: var(--accent-soft); color: var(--ink); }
.numeric {
  text-align: right;
  font-family: var(--font-mono);
  white-space: nowrap;
}
.method-name { color: var(--ink); font-weight: 650; }
.identifier {
  min-width: 24rem;
  max-width: 44rem;
  overflow-wrap: anywhere;
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
}
.provenance-value {
  min-width: 18rem;
  max-width: 48rem;
  overflow-wrap: anywhere;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  white-space: normal;
}
.hardware-grid, .split-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-4);
}
.hardware-sheet { border-top: 2px solid var(--accent); }
.hardware-sheet.target { border-top-color: var(--ink); }
.hardware-name {
  margin: var(--space-1) 0 var(--space-3);
  overflow-wrap: anywhere;
  font-family: var(--font-mono);
  font-size: var(--text-md);
  font-weight: 650;
}
.fact-list { margin: 0; }
.fact-list div {
  display: grid;
  grid-template-columns: minmax(8rem, 0.75fr) minmax(0, 1fr);
  gap: var(--space-3);
  padding: var(--space-2) 0;
  border-top: 1px solid var(--line);
}
.fact-list dt { color: var(--subtle); font-size: var(--text-xs); }
.fact-list dd {
  margin: 0;
  overflow-wrap: anywhere;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
}
.comparison-lead {
  display: grid;
  grid-template-columns: minmax(8rem, auto) minmax(0, 1fr);
  gap: var(--space-4);
  align-items: center;
  padding: var(--space-4);
  border-left: var(--space-1) solid var(--accent);
  background: var(--accent-soft);
}
.delta {
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1;
}
.delta.negative { color: var(--negative); }
.delta-context { max-width: 62ch; color: var(--muted); font-size: var(--text-sm); }
.comparison-lead + .table-wrap { margin-top: var(--space-3); border: 1px solid var(--line); }
.disclosure {
  padding: var(--space-4);
  border-left: var(--space-1) solid var(--line-strong);
  background: var(--note);
  color: var(--muted);
  font-size: var(--text-sm);
}
.disclosure strong { color: var(--ink-strong); }
.disclosure p { margin: var(--space-2) 0 0; }
.empty-state {
  padding: var(--space-4);
  border: 1px dashed var(--line-strong);
  color: var(--muted);
  background: var(--paper-muted);
  font-size: var(--text-sm);
}
.matrix-stack { display: grid; gap: var(--space-3); }
.limitations {
  margin: 0;
  padding-left: var(--space-5);
  color: var(--muted);
  font-size: var(--text-sm);
}
.limitations li + li { margin-top: var(--space-2); }
.footer {
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-5) 0;
  border-top: 1px solid var(--line-strong);
  color: var(--subtle);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
}
.footer strong { color: var(--ink); }

@media (max-width: 62rem) {
  .identity-layout { grid-template-columns: 1fr; gap: var(--space-4); }
  .method-note { max-width: 52rem; }
  .result-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .result:nth-child(odd) { border-left: 0; }
  .result:first-child { border-left: var(--space-1) solid var(--accent); }
  .result:nth-child(n + 3) { border-top: 1px solid var(--line); }
}
@media (max-width: 44rem) {
  .page { padding-inline: var(--space-4); }
  .report-header { padding-top: var(--space-4); }
  .utility-rail { align-items: flex-start; }
  .identity-layout { padding-block: var(--space-4); }
  .meta-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .meta-strip div:nth-child(odd) { border-left: 0; }
  .meta-strip div:nth-child(n + 3) { border-top: 1px solid var(--line); }
  .result-strip { grid-template-columns: 1fr; margin-bottom: var(--space-5); }
  .result { border-left: 0; border-top: 1px solid var(--line); }
  .result:first-child { grid-column: auto; border-top: 0; border-left: var(--space-1) solid var(--accent); }
  .section { padding-block: var(--space-5); }
  .section-head { grid-template-columns: 1fr; gap: var(--space-1); }
  .hardware-grid, .split-grid { grid-template-columns: 1fr; }
  .comparison-lead { grid-template-columns: 1fr; gap: var(--space-2); }
  .footer { flex-direction: column; }
}
@media (max-width: 25rem) {
  .utility-rail { flex-direction: column; gap: var(--space-2); }
  .data-flag { text-align: left; }
  .route { gap: var(--space-2); }
  .meta-strip { grid-template-columns: 1fr; }
  .meta-strip div, .meta-strip div:nth-child(odd) { border-left: 0; border-top: 1px solid var(--line); }
  .meta-strip div:first-child { border-top: 0; }
  .fact-list div { grid-template-columns: 1fr; gap: var(--space-1); }
}
@media print {
  body, html { background: var(--paper); }
  .page { max-width: none; padding: 0; }
  .chart-scroll, .table-wrap { overflow: visible; }
  .chart-scroll svg { min-width: 0; }
  details:not([open]) > *:not(summary) { display: block; }
  .section, .panel, .chart-frame, table { break-inside: avoid; }
}
"""


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
    for role in ("source", "target"):
        name, facts = _hardware_facts(summary, role)
        sheets.append(
            f'<article class="panel hardware-sheet {role}">'
            f'<span class="hardware-role">{role} hardware</span>'
            f'<h3 class="hardware-name">{_escape(name)}</h3>'
            f"{_render_fact_list(facts)}"
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
            rows = "".join(
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
                f"<tbody>{rows}</tbody></table></div></details>"
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
            rows = "".join(
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
                f"<tbody>{rows}</tbody></table></div></details>"
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
    methods: Mapping[str, list[dict[str, float]]], labels: Mapping[str, str]
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
        '<div class="table-wrap"><table><caption>Values used to draw the budget-efficiency figure</caption>'
        '<thead><tr><th scope="col">Method</th><th scope="col">Budget</th>'
        '<th scope="col">Mean fraction of held-out reference</th><th scope="col">CI95 low</th>'
        '<th scope="col">CI95 high</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _flatten_details(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    details: list[tuple[str, Any]] = []
    for key, item in value.items():
        label = f"{prefix} / {_human_label(key)}" if prefix else _human_label(key)
        if isinstance(item, Mapping):
            details.extend(_flatten_details(item, label))
        else:
            details.append((label, item))
    return details


def _render_reproducibility(
    summary: Mapping[str, Any], methods: Mapping[str, list[dict[str, float]]]
) -> str:
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
    return f"{table}{note}"


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
        max_budget = max(point["budget"] for points in methods.values() for point in points)
        budget_value = _format_budget(max_budget)
        budget_label = "Largest target budget reported"
    else:
        best_value, best_label = _MISSING, "Best reported endpoint"
        budget_value, budget_label = _MISSING, "Largest target budget reported"

    headline = _as_mapping(summary.get("headline"))
    strongest_legacy = None if headline is None else headline.get("strongest_legacy_method")
    auc_delta = None if headline is None else headline.get("auc_delta_vs_strongest_legacy")
    if (
        isinstance(strongest_legacy, str)
        and strongest_legacy in labels
        and isinstance(auc_delta, (int, float))
    ):
        transfer_value = f"{float(auc_delta) * 100:+.1f} pp AUC"
        transfer_label = (
            f"{labels.get('transfer_thompson', 'Transfer')} − "
            f"{labels[strongest_legacy]} across the reported budget curve"
        )
    else:
        transfer = _method_by_role(summary, methods, "transfer")
        cold = _method_by_role(summary, methods, "cold")
        comparisons = _shared_comparison(methods, transfer, cold)
        if comparisons:
            comparison_budget, transfer_point, cold_point = comparisons[-1]
            delta = (transfer_point["mean"] - cold_point["mean"]) * 100
            transfer_value = f"{delta:+.1f} pp"
            transfer_label = (
                f"{labels.get(transfer or '', 'Transfer')} − "
                f"{labels.get(cold or '', 'cold start')} at shared budget "
                f"{_format_budget(comparison_budget)}"
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


def render_report(summary: Mapping[str, Any], output_path: str | Path) -> None:
    """Render a complete offline research report from an aggregate tuning summary.

    Required summary keys are ``source_gpu``, ``target_gpu``, ``workloads``, ``configs``,
    and ``methods``. Each method maps to points containing ``budget``,
    ``mean_fraction_oracle``, ``ci95_low``, and ``ci95_high``.
    """

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    missing = [
        key
        for key in ("source_gpu", "target_gpu", "workloads", "configs", "methods")
        if key not in summary
    ]
    if missing:
        raise ValueError(f"summary is missing required keys: {', '.join(missing)}")

    methods = _normalise_methods(summary["methods"])
    labels = _method_display_labels(summary, methods)
    source_name, _ = _hardware_facts(summary, "source")
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
        '<figure class="chart-frame">'
        '<div class="chart-scroll">'
        f"{_render_chart(methods, labels)}"
        "</div>"
        f"{_render_legend(methods, labels)}"
        "<figcaption>Lines connect reported means; bands encode the supplied 95% confidence bounds. "
        "Values are plotted as supplied and are not recomputed by this renderer.</figcaption>"
        "</figure>"
        f'<details class="data-disclosure"><summary>Complete numeric results · {curve_points} rows</summary>'
        f"{_render_raw_table(methods, labels)}</details>"
    )
    cost_and_limits = (
        '<div class="split-grid">'
        f"{_render_cost_disclosure(summary)}"
        '<div class="panel"><h3>Interpretation boundaries</h3>'
        f"{_render_limitations(summary)}</div></div>"
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
        f"<title>{_escape(title)}</title>"
        f"<style>{_STYLES}</style></head><body>"
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
        '<div class="route-node"><span>Source context</span>'
        f"<strong>{_escape(source_name)}</strong></div>"
        '<span class="route-arrow" aria-hidden="true">→</span>'
        '<div class="route-node"><span>Target context</span>'
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
        f"{_section('01', 'Budget-efficiency curves', 'All supplied methods share the same target-budget and held-out-reference axes. The legend aligns each human-readable method label with its terminal value.', chart)}"
        f"{_section('02', 'Transfer versus cold start', 'Only matching target budgets are compared. A positive percentage-point delta means the identified transfer method has the higher supplied mean.', _render_comparison(methods, labels, transfer, cold))}"
        f"{_section('03', 'Hardware context', 'Source and target facts are separated and shown exactly as supplied. A synthetic report identifies simulated device context without presenting it as measured hardware.', _render_hardware(summary))}"
        f"{_section('04', 'Experiment scope', 'The workload and launch-configuration inventories bound the reported reference and every curve. Large inventories are contained in local, expandable tables.', _render_experiment_matrix(summary['workloads'], summary['configs'], summary.get('experiment')))}"
        f"{_section('05', 'Protocol and provenance', 'Reported run metadata and experiment protocol are inventoried separately from facts that are absent from the summary.', _render_reproducibility(summary, methods))}"
        f"{_section('06', 'Source cost and limitations', 'Target-budget efficiency is not end-to-end efficiency. Source acquisition and interpretation boundaries remain explicit.', cost_and_limits)}"
        "</main>"
        '<footer class="footer"><span><strong>HeliosTune</strong> / GPU autotuning transfer report</span>'
        "<span>Offline HTML · inline SVG · no network requests</span></footer></div>"
        f'<script type="application/json" id="heliostune-chart-data">{data_blob}</script>'
        "</body></html>"
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
