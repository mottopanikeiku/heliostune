"""Build the quartet report and README figure from the stored summaries."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from html import escape
from pathlib import Path
from typing import Any, cast

from heliostune.tooling.artifacts import read_json

_REPO = Path(__file__).resolve().parents[1]
SUMMARY = "results/action-expansion-summary.json"
DIAGNOSIS = "results/kernel-bank-diagnosis.json"
OUTPUT = "site/index.html"
REPOSITORY = "https://github.com/mottopanikeiku/quartet"
BLOB = f"{REPOSITORY}/blob/main/"
EXPANSION_VIEW = "results/action-expansion-raw.json [NVIDIA H100 80GB HBM3]"
H100_ARCHIVE = "benchmarks/data/h100-measurements.jsonl.zst"
H100_ARCHIVE_VIEW = f"{H100_ARCHIVE} [NVIDIA H100 80GB HBM3]"
ORIGINAL_GPUS = (
    ("T4", "benchmarks/data/t4-measurements.jsonl.zst", "Tesla T4"),
    ("A10", "benchmarks/data/measurements.jsonl.zst", "NVIDIA A10"),
    ("L4", "benchmarks/data/measurements.jsonl.zst", "NVIDIA L4"),
    ("H100", H100_ARCHIVE, "NVIDIA H100 80GB HBM3"),
)
ARCHIVE: tuple[tuple[str, str, str], ...] = (
    (
        "transfer.html",
        "L4 + A10 + T4 → H100 transfer",
        "Cold Thompson sampling beat cross-GPU transfer; the selected posterior transfer strength was zero.",
    ),
    (
        "parhelion-v2-addendum.html",
        "Transfer ablations",
        "A follow-up comparison of retrieval anchors and posterior transfer.",
    ),
    ("v1.html", "L4 → A10 transfer", "The original single-source comparison."),
    ("a10-to-l4.html", "A10 → L4 transfer", "The reverse direction."),
    (
        "parhelion-v3-engineering.html",
        "H200 transfer",
        "A later run with more configurations; A100 parameter selection mixed PCIe and SXM devices.",
    ),
    (
        "hopper-h100-engineering.html",
        "H100 persistent and skinny-GEMV kernels",
        "An earlier single-pass comparison.",
    ),
    (
        "h100-precision-probe.html",
        "H100 FP16 reduction",
        "Fixed configurations under different arithmetic settings.",
    ),
    (
        "fusion-remote-exploratory.html",
        "H100 fusion experiments",
        "RMSNorm and gated-MLP measurements.",
    ),
    ("native-rmsnorm-h100.html", "H100 Triton RMSNorm", "A separate operator comparison."),
)
ARCHIVE_DATA: tuple[tuple[str, str], ...] = (
    ("parhelion-h100-final.json", "H100 transfer summary"),
    ("parhelion-t4-selection.json", "T4 selection summary"),
    ("a10-to-l4.json", "v1 A10 → L4 summary"),
    ("l4-to-a10.json", "v1 L4 → A10 summary"),
)
_SHORT_MODEL = {
    "granite-3.1-8b": "Granite",
    "mistral-7b": "Mistral",
    "phi-3-mini": "Phi",
    "qwen2.5-7b": "Qwen",
}
_ARMS = (("old", "Old 36"), ("new", "New actions"), ("union", "Expanded union"))
_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'none'; "
    "base-uri 'none'; form-action 'none'"
)
_CSS = """
:root { --paper: #f4f0e7; --surface: #fbf8f1; --ink: #142a33; --muted: #53656a;
  --line: #c9c0ae; --accent: #a23b2a; --accent-soft: #f1d9cf; --teal: #176b67;
  --teal-soft: #d9e8e3; --warn: #76591b; --warn-soft: #eee2bd; color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); line-height: 1.6;
  font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif; }
a { color: var(--teal); text-underline-offset: 0.2rem; }
a:focus-visible { outline: 0.2rem solid var(--accent); outline-offset: 0.2rem; }
.skip { position: absolute; left: 1rem; top: -5rem; background: var(--ink); color: var(--surface); padding: 0.5rem 1rem; }
.skip:focus { top: 1rem; }
.page { width: min(100% - 2rem, 78rem); margin-inline: auto; }
header { padding: 2.5rem 0 2rem; border-bottom: 0.0625rem solid var(--line); }
.eyebrow, .badge { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.badge { display: inline-block; margin: 0 0.5rem 0.5rem 0; padding: 0.15rem 0.6rem; border: 0.0625rem solid var(--accent);
  border-radius: 0.25rem; background: var(--accent-soft); color: var(--accent); }
.badge.inference { border-color: var(--warn); background: var(--warn-soft); color: var(--warn); vertical-align: middle; }
h1, h2, h3 { font-family: Charter, Georgia, serif; line-height: 1.12; }
h1 { font-size: clamp(2rem, 5vw, 3.6rem); max-width: 20ch; margin: 0.5rem 0; letter-spacing: -0.02em; }
h2 { font-size: 1.7rem; margin: 3rem 0 0.75rem; }
h3 { font-size: 1.2rem; margin: 2rem 0 0.5rem; }
.lede { max-width: 48rem; font-size: 1.125rem; color: var(--muted); }
nav ul { display: flex; flex-wrap: wrap; gap: 0.25rem 1.25rem; list-style: none; padding: 0; margin: 1.5rem 0 0; }
.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 0.75rem; margin: 2rem 0 0; }
.strip div { background: var(--surface); border: 0.0625rem solid var(--line); border-radius: 0.5rem; padding: 0.9rem 1rem; }
.strip strong { display: block; font-family: Charter, Georgia, serif; font-size: 2rem; line-height: 1.1; }
.strip span { color: var(--muted); font-size: 0.875rem; }
p, li { max-width: 52rem; }
.note { color: var(--muted); font-size: 0.9rem; }
.callout { background: var(--teal-soft); border-left: 0.25rem solid var(--teal); padding: 0.75rem 1rem; max-width: 52rem; }
.callout.inference { background: var(--warn-soft); border-color: var(--warn); }
.wrap { overflow-x: auto; margin: 1rem 0; background: var(--surface); border: 0.0625rem solid var(--line); border-radius: 0.5rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 0.875rem; }
caption { text-align: left; padding: 0.75rem 1rem 0.25rem; color: var(--muted); }
th, td { padding: 0.45rem 0.7rem; border-bottom: 0.0625rem solid var(--line); text-align: right; vertical-align: top; }
th { color: var(--muted); font-size: 0.8rem; font-weight: 700; }
.text { text-align: left; }
td.text:first-child { white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
dl { display: grid; grid-template-columns: minmax(0, 16rem) minmax(0, 1fr); gap: 0.35rem 1.25rem; max-width: 62rem; }
dt { color: var(--muted); }
dt, dd { min-width: 0; overflow-wrap: anywhere; }
dd { margin: 0; }
code { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85em; overflow-wrap: anywhere; }
figure { margin: 1.5rem 0; }
figcaption { color: var(--muted); font-size: 0.875rem; max-width: 52rem; }
svg { max-width: 100%; height: auto; background: var(--surface); border: 0.0625rem solid var(--line); border-radius: 0.5rem; }
footer { margin: 4rem 0 2rem; padding-top: 1rem; border-top: 0.0625rem solid var(--line); color: var(--muted); font-size: 0.875rem; }
@media (max-width: 40rem) {
  dl { grid-template-columns: minmax(0, 1fr); gap: 0 0; }
  dd { margin-bottom: 0.6rem; }
}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _repo_link(path: str, text: str | None = None) -> str:
    return f'<a href="{_e(BLOB + path)}"><code>{_e(text or path)}</code></a>'


def _ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}×"


def _us(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def bound_pct(value: float, *, upper: bool, digits: int = 2) -> str:
    """Format a fraction as a percentage bound that never understates its side.

    "At most"/"within" bounds round up and "at least" bounds round down. Rounding
    uses the float's exact binary value, so no decimal artefact can cross the bound.
    """
    if not math.isfinite(value):
        raise ValueError("a percentage bound must be finite")
    # A double's exact decimal expansion has at most 767 significant digits.
    with localcontext(prec=1100):
        step = Decimal(1).scaleb(-digits)
        percent = Decimal(value) * 100
        rounded = percent.quantize(step, ROUND_CEILING if upper else ROUND_FLOOR)
    return f"{rounded}%"


def _span(values: Sequence[float], render: Callable[[float], str]) -> str:
    low, high = render(min(values)), render(max(values))
    return low if low == high else f"{low}–{high}"


def _table(
    table_id: str,
    caption: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    text_columns: int = 1,
) -> str:
    def cell(tag: str, index: int, value: str, scope: str = "") -> str:
        css = ' class="text"' if index < text_columns else ""
        return f"<{tag}{scope}{css}>{value}</{tag}>"

    head = "".join(cell("th", i, _e(h), ' scope="col"') for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows
    )
    return (
        f'<div class="wrap"><table id="{_e(table_id)}"><caption>{_e(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _dl(items: Iterable[tuple[str, str]]) -> str:
    return "<dl>" + "".join(f"<dt>{_e(k)}</dt><dd>{v}</dd>" for k, v in items) + "</dl>"


def _view(diagnosis: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [d for d in diagnosis["datasets"] if d["view"] == name]
    if len(matches) != 1:
        raise ValueError(f"diagnosis must contain exactly one view {name!r}")
    return cast(Mapping[str, Any], matches[0])


def _by_regime(group: Mapping[str, Any], regime: str) -> Mapping[str, Any]:
    matches = [r for r in group["by_regime"] if r["regime"] == regime]
    if len(matches) != 1:
        raise ValueError(f"missing regime {regime!r}")
    return cast(Mapping[str, Any], matches[0])


def _full_pairs(dataset: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        p["full_roster"]
        for p in dataset["bank_pairs"]
        if p["full_roster"]["overall"]["named_count"]
    ]


def _consistent(summary: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> None:
    raw = [i for i in diagnosis["inputs"] if i["path"] == "results/action-expansion-raw.json"]
    if len(raw) != 1 or raw[0]["sha256"] != summary["input_sha256"]:
        raise ValueError("diagnosis and summary were built from different raw measurements")
    overall = diagnosis["expansion"]["overall"]
    for arm, _ in _ARMS:
        if (
            overall[f"{arm}_torch_ratio"]["named"]
            != summary["overall"]["arms"][arm]["geometric_mean_to_torch"]
        ):
            raise ValueError(f"diagnosis and summary disagree on the {arm} geometric mean")


def _wins(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    workloads = {w["workload_key"]: w for w in summary["workloads"]}
    return [workloads[w["workload_key"]] for w in summary["largest_wins"]["new"]]


def _gpu_section(diagnosis: Mapping[str, Any]) -> str:
    rows = []
    for label, path, device in ORIGINAL_GPUS:
        dataset = _view(diagnosis, f"{path} [{device}]")
        (pair,) = [
            p for p in dataset["bank_pairs"] if p["selection_bank"] == 1 and p["scoring_bank"] == 2
        ]
        overall = pair["full_roster"]["overall"]
        rows.append(
            [
                label,
                str(overall["named_count"]),
                _ratio(overall["policy_torch_ratio"]["named"]),
                f"{(overall['policy_oracle_ratio']['named'] - 1) * 100:.2f}%",
            ]
        )
    table = _table(
        "gpus",
        "Original 36 configurations; selected on pass 1, scored on pass 2",
        ("GPU", "Workloads", "Triton / torch", "Selection overhead"),
        rows,
    )
    return f"""
<section id="gpus" aria-labelledby="gpus-title">
<h2 id="gpus-title">Four GPUs, the same 96 workloads</h2>
<p>The suite covers attention and FFN projections from Granite, Mistral, Phi and Qwen at
M = 1, 7, 31, 96, 257 and 1024. There are 84 distinct shapes. Ratios are geometric means;
lower is faster. Selection overhead compares the fixed pass-1 choice with the fastest
configuration measured on pass 2.</p>
{table}
<p>The original kernels are close to torch on A10 and L4. On H100 and T4, selecting
configurations more accurately leaves most of the latency gap untouched. I expanded the
H100 kernel set to address that gap.</p>
</section>"""


def _result_section(summary: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    arms = overall["arms"]
    view = diagnosis["expansion"]["overall"]
    rows = [
        [
            _e(label),
            str(arms[arm]["comparable_win_count"]),
            str(arms[arm]["comparable_tie_count"]),
            str(arms[arm]["comparable_loss_count"]),
            _ratio(view[f"{arm}_torch_ratio"]["named"]),
            _ratio(view[f"{arm}_torch_ratio"]["unique_shape"]),
            _us(view[f"{arm}_us"]["named"]),
            _us(view[f"{arm}_us"]["unique_shape"]),
        ]
        for arm, label in _ARMS
    ]
    rows.append(
        [
            "torch.matmul",
            "—",
            "—",
            "—",
            _ratio(1.0),
            _ratio(1.0),
            _us(view["torch_us"]["named"]),
            _us(view["torch_us"]["unique_shape"]),
        ]
    )
    table = _table(
        "headline",
        f"Bank-2 scoring of bank-1 selections: {overall['named_workload_count']} named "
        f"workloads, {overall['unique_shape_count']} unique shapes, one H100",
        (
            "Arm",
            "Wins vs torch",
            "Exact ties",
            "Losses",
            "Latency / torch (named)",
            "Latency / torch (unique shape)",
            "Mean µs (named)",
            "Mean µs (unique shape)",
        ),
        rows,
    )
    wins = _wins(summary)
    win_rows = [
        [
            _e(f"{_SHORT_MODEL[w['model']]} / {w['projection']}"),
            _e("({},{},{})".format(*w["shape"])),
            _e(w["selected"]["new"]["family"]),
            f"{w['new_ms'] * 1000:.3f}",
            f"{w['torch_ms'] * 1000:.3f}",
            f"{w['new_to_torch']:.6f}",
        ]
        for w in wins
    ]
    wins_table = _table(
        "wins",
        "Every nominal new-action win, smallest ratio first; the exact median tie is not a win",
        ("Model / projection", "(M,N,K)", "Family", "New µs", "Torch µs", "New / torch"),
        win_rows,
        text_columns=3,
    )
    comparison = overall["old_new_comparison"]
    return f"""
<section id="result" aria-labelledby="result-title">
<h2 id="result-title">Closing the H100 gap</h2>
<p>The new actions reduce geometric-mean latency by
{(1 - comparison["geometric_mean_new_to_old"]) * 100:.1f}% and improve
{comparison["new_faster_count"]}/{overall["named_workload_count"]} workloads. New/old latency is
<strong>{comparison["geometric_mean_new_to_old"]:.3f}</strong>.
Each ratio divides the selected candidate's median by torch's median in the same scoring pass.
“Named” weights each model projection equally; “unique shape” first averages projections
sharing an (M,N,K) shape. The microsecond columns are arithmetic means. All of this is one H100
session, and {sum(w["new_to_torch"] > 0.99 for w in wins)} of the {len(wins)} wins are under 1%.</p>
{table}
{wins_table}
</section>"""


def _split_section(summary: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> str:
    raw = _view(diagnosis, EXPANSION_VIEW)
    banks = {b["bank"]: b for b in raw["banks"]}
    same = raw["bank_pairs"][0]["common_actions_only"]["overall"]["winner_same"]
    agreed = round(same["named"] * same["named_covered"])
    actions = summary["action_set"]
    split = sum(a["family"] == "split_k" for a in actions["new"])
    return f"""
<section id="split" aria-labelledby="split-title">
<h2 id="split-title">Selection and scoring</h2>
<p>Pass 1 times all {len(actions["old"])} original configurations,
{len(actions["new"]) - split} new ordinary and persistent actions, and {split} split-K actions
when M ≤ 31 ({banks[1]["observed_cells"]:,} candidate measurements).
For each workload, it selects the lowest valid median separately from the old set,
the new set and their union, breaking ties by configuration key. Pass 2 times only those fixed
choices ({banks[2]["observed_cells"]} candidate measurements). Both passes time torch once per
workload. The JSON files call these passes banks 1 and 2.
See the {_repo_link("docs/action-expansion.md", "measurement method")}.</p>
<p>The combined set's pass-1 choice also wins the old-versus-new comparison on pass 2 for
{agreed}/{same["named_covered"]} workloads. This page measures exhaustive selection;
Thompson sampling has not been rerun on the expanded set.</p>
</section>"""


def _contract_section(summary: Mapping[str, Any]) -> str:
    hardware, software = summary["hardware"], summary["software"]
    protocol, precision = summary["protocol"], summary["precision"]
    capability = ".".join(map(str, hardware["compute_capability"]))
    quantiles = "/".join(f"p{round(q * 100)}" for q in protocol["quantiles"])
    items = [
        (
            "Device",
            _e(
                f"{hardware['device_name']}, {hardware['multiprocessor_count']} SMs, "
                f"compute capability {capability}"
            ),
        ),
        (
            "Software",
            _e(
                f"Torch {hardware['torch_version']}, Triton {software['triton']}, "
                f"CUDA {hardware['cuda_version']}"
            ),
        ),
        ("Operands", "FP16 A[M,K] @ B[K,N] with FP16 output"),
        ("Reference", _e(protocol["reference"])),
        ("Tolerance", _e(f"atol={protocol['atol']}, rtol={protocol['rtol']}")),
        (
            "Torch precision flags",
            _e(
                f"allow_tf32={precision['allow_tf32']}, allow_fp16_reduced_precision_reduction="
                f"{precision['allow_fp16_reduced_precision_reduction']}, "
                f"float32_matmul_precision={precision['float32_matmul_precision']}"
            ),
        ),
        ("Timed call", _e(protocol["timed_calls"])),
        (
            "Timing",
            _e(f"{protocol['warmup_ms']} ms warmup, {protocol['rep_ms']} ms timing; {quantiles}"),
        ),
        (
            "Order and seeds",
            _e(
                f"workload order: {protocol['workload_order_seed']}; config order: "
                f"{protocol['config_order_seed']}; tensors: {protocol['tensor_seed']}"
            ),
        ),
        ("Selection rule", _e(protocol["selection"])),
        (
            "Measured source",
            "The original action-expansion collection; collector and package hashes are in the "
            f"{_repo_link('results/action-expansion-raw.json', 'raw artifact')}",
        ),
        ("Failure rows", _e(f"{summary['failure_row_count']}; a failed row would stay unscored")),
    ]
    return f"""
<section id="contract" aria-labelledby="contract-title">
<h2 id="contract-title">Numerical and timing setup</h2>
<p>Each candidate must pass the numerical comparison before timing.
Torch's reduced-precision FP16 reduction flag was enabled.</p>
{_dl(items)}
</section>"""


def _regime_chart(regimes: Sequence[Mapping[str, Any]]) -> str:
    series = (
        ("old", "#1261a0", "Old 36"),
        ("new", "#b54c00", "New"),
        ("union", "#176b67", "Union"),
    )
    values = [r[f"{a}_torch_ratio"]["named"] for r in regimes for a, _, _ in series]
    low, high = min(1.0, *values) / 1.08, max(values) * 1.07
    left, right, top, row = 120, 700, 58, 54
    height = top + row * len(regimes) + 30

    def x(value: float) -> float:
        return left + math.log(value / low) / math.log(high / low) * (right - left)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="{height}" '
        f'viewBox="0 0 720 {height}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Geometric-mean latency over torch by M</title>',
        '<desc id="chart-desc">Log-scale dot plot of named-workload geometric means for the old, '
        "new and union arms on bank 2. The dark vertical line marks parity with torch; lower is "
        "faster.</desc>",
        '<rect width="100%" height="100%" fill="#fbf8f1"/>',
        '<g font-family="sans-serif" font-size="12" fill="#142a33">',
    ]
    for i, (_, color, label) in enumerate(series):
        parts.append(
            f'<circle cx="{130 + i * 110}" cy="22" r="5" fill="{color}"/>'
            f'<text x="{140 + i * 110}" y="26">{label}</text>'
        )
    tick = math.ceil(low * 4) / 4
    while tick <= high:
        position = x(tick)
        stroke = "#142a33" if tick == 1 else "#c9c0ae"
        parts.append(
            f'<line x1="{position:.1f}" y1="{top - 12}" x2="{position:.1f}" y2="{height - 22}" '
            f'stroke="{stroke}"/><text x="{position:.1f}" y="{height - 6}" '
            f'text-anchor="middle">{tick:.2f}×</text>'
        )
        tick += 0.25
    for i, regime in enumerate(regimes):
        y = top + i * row + 16
        m = ", ".join(map(str, regime["m_values"]))
        parts.append(f'<text x="12" y="{y + 4}">M = {m}</text>')
        for j, (arm, color, label) in enumerate(series):
            ratio = regime[f"{arm}_torch_ratio"]["named"]
            parts.append(
                f'<circle cx="{x(ratio):.1f}" cy="{y - 12 + j * 12}" r="5" fill="{color}">'
                f"<title>{label}, M = {m}: {ratio:.3f}× torch</title></circle>"
            )
    parts.extend(["</g>", "</svg>"])
    return "".join(parts)


def _regime_section(diagnosis: Mapping[str, Any]) -> str:
    expansion = diagnosis["expansion"]
    regimes = expansion["by_regime"]
    pairs = _full_pairs(_view(diagnosis, H100_ARCHIVE_VIEW))
    rows = []
    for r in regimes:
        archived = [_by_regime(p, r["regime"]) for p in pairs]
        outcome = r["new_vs_torch"]
        rows.append(
            [
                _e(f"M={', '.join(map(str, r['m_values']))} ({r['regime']})"),
                f"{r['named_count']} / {r['shape_count']}",
                _us(r["torch_us"]["named"]),
                _us(r["old_us"]["named"]),
                _us(r["new_us"]["named"]),
                _ratio(r["old_torch_ratio"]["named"]),
                _ratio(r["new_torch_ratio"]["named"]),
                _ratio(r["new_torch_ratio"]["unique_shape"]),
                f"{outcome['wins']} / {outcome['ties']} / {outcome['losses']}",
                _e(", ".join(f"{k} {v}" for k, v in r["new_winner_families"].items())),
                _span([a["oracle_torch_ratio"]["named"] for a in archived], _ratio),
                _span([a["policy_oracle_ratio"]["named"] for a in archived], _ratio),
            ]
        )
    table = _table(
        "regimes",
        "Bank-2 results by M; the last two columns re-read the original 36-action H100 archive",
        (
            "Regime",
            "Named / shapes",
            "Torch µs",
            "Old µs",
            "New µs",
            "Old / torch",
            "New / torch (named)",
            "New / torch (shape)",
            "New W / T / L",
            "Selected new families",
            "Old-bank oracle / torch",
            "Earlier-bank pick / oracle",
        ),
        rows,
        text_columns=2,
    )
    split = [r for r in regimes if "split_k" in r["new_winner_families"]]
    rest = [r for r in regimes if "split_k" not in r["new_winner_families"]]
    split_count = sum(r["new_winner_families"]["split_k"] for r in split)
    split_total = sum(r["named_count"] for r in split)

    def ratios(group: Sequence[Mapping[str, Any]]) -> str:
        return ", ".join(
            f"M={r['m_values'][0]}: {_ratio(r['new_torch_ratio']['named'])}" for r in group
        )

    return f"""
<section id="regimes" aria-labelledby="regimes-title">
<h2 id="regimes-title">H100 results by M</h2>
<p>For M ≤ 31, where split-K applies, bank 1 picked split-K for {split_count}/{split_total} workloads. There,
new/torch is {_e(ratios(split))}. At the larger M values without split-K it is {_e(ratios(rest))}. Bank 1 picked
persistent TMA for {expansion["new_winner_families"].get("persistent", 0)} workloads, and it beat torch on none of
them.</p>
<figure>{_regime_chart(regimes)}
<figcaption>Geometric-mean latency relative to torch, grouped by M.
<a href="h100-by-m.svg">Download the figure</a>.</figcaption></figure>
{table}
<p class="note">The last two columns use {_repo_link(H100_ARCHIVE, "the original H100 archive")},
which predates this collection and uses a different correctness reference. Each entry spans its
{len(pairs)} selection/scoring pairs. “Oracle” is the fastest of the original 36 actions on
the scoring pass. “Pick / oracle” compares the earlier pass's choice with that minimum.</p>
</section>"""


def _bank_section(diagnosis: Mapping[str, Any]) -> str:
    shared: dict[str, int] = {}
    for pair in diagnosis["shared_cells"]:
        for view in pair["views"]:
            shared[view] = shared.get(view, 0) + 1
    multi, single, picks, oracles = [], [], [], []
    for d in diagnosis["datasets"]:
        name = f"{_repo_link(d['path'])}<br>{_e(d['hardware']['device_name'])}"
        failures = sum(b["failure_cells"] for b in d["banks"])
        cells = sum(b["observed_cells"] for b in d["banks"])
        full = [p["overall"] for p in _full_pairs(d)]
        if full:
            pick = [f["policy_oracle_ratio"]["named"] for f in full]
            oracle = [f["oracle_torch_ratio"]["named"] for f in full]
            picks += pick
            oracles += oracle
            multi.append(
                [
                    name,
                    str(len(d["banks"])),
                    f"{cells:,}",
                    str(failures),
                    _span(pick, _ratio),
                    _span(oracle, _ratio),
                    _span([f["winner_same"]["named"] for f in full], _pct),
                    str(shared.get(d["view"], 0)),
                ]
            )
        elif not d["bank_pairs"]:
            bank = d["banks"][0]
            single.append(
                [
                    name,
                    str(bank["bank"]),
                    f"{cells:,}",
                    str(failures),
                    str(bank["observed_workloads"]),
                    _ratio(bank["overall"]["oracle_torch_ratio"]["named"]),
                    str(shared.get(d["view"], 0)),
                ]
            )
    multi_table = _table(
        "bank-pairs",
        "Every stored multi-bank archive whose banks share the full action set; ranges are over that view's bank pairs",
        (
            "View",
            "Banks",
            "Measurements",
            "Failed",
            "Earlier-bank pick / oracle",
            "Oracle / torch",
            "Same winner",
            "Overlapping archives",
        ),
        multi,
    )
    single_table = _table(
        "single-bank",
        "Single-bank views: fastest action in the same bank only, so nothing is held out",
        (
            "View",
            "Bank",
            "Measurements",
            "Failed",
            "Workloads",
            "In-sample oracle / torch",
            "Overlapping archives",
        ),
        single,
    )
    worst = max(picks) - 1
    return f"""
<section id="bank" aria-labelledby="bank-title">
<h2 id="bank-title">Selection loss across the stored measurements</h2>
<p>For each pair of timing passes, I select the fastest valid action on the earlier pass and
score it on the later pass. The comparison is the later pass's own minimum.
Across {len(multi)} multi-pass archives, the selection is at most {bound_pct(worst, upper=True)} slower
in geometric-mean latency. The same-pass minimum ranges from {_ratio(min(oracles))} to
{_ratio(max(oracles))} torch. Where the gap is large, the kernel set is the main constraint.</p>
<p>This comparison uses exhaustive selection. Probe-limited Thompson sampling and transfer
results are in the <a href="transfer.html">original tuning report</a>.</p>
{multi_table}
{single_table}
<p class="note">Some archives contain the same measurements with different hardware labels or
combined acquisition sessions. {len(diagnosis["shared_cells"])} archive pairs overlap; they are
listed separately rather than pooled. Full per-shape results and failure counts are in
{_repo_link(DIAGNOSIS)}.</p>
</section>"""


def _dossier_section(diagnosis: Mapping[str, Any]) -> str:
    dossier = diagnosis["dossier"]
    predictions = dossier["predictions"]
    rows = []
    for p in predictions:
        if not p["bank1_selected"]:
            continue
        measured = p["bank2_measurement"]
        balance = p["persistent_tile_balance"]
        rows.append(
            [
                _e(f"M={p['shape'][0]}"),
                _e(p["arm"]),
                f"<code>{_e(p['config_key'])}</code>",
                f"{p['matmul_programs']:,} ({p['programs_per_sm']:.2f}/SM)",
                str(p["kernel_launches"]),
                _pct(p["masked_row_fraction"]),
                _pct(p["useful_dot_fraction"]),
                "—" if balance is None else _pct(balance),
                f"{p['workspace_bytes'] / 1024:,.0f}",
                f"{p['workspace_adjusted_intensity_flops_per_byte']:,.1f}",
                _us(measured["median_us"]),
                _ratio(measured["ratio_to_torch"]),
            ]
        )
    shape = predictions[0]["shape"]
    table = _table(
        "dossier",
        f"Qwen2.5-7B FFN-down (N={shape[1]}, K={shape[2]}): bank-1-selected old and new actions, "
        f"{dossier['sm_count']} SMs",
        (
            "M",
            "Arm",
            "Action",
            "Programs",
            "Launches",
            "Masked rows",
            "Useful dot work",
            "Persistent balance",
            "Workspace KiB",
            "FLOP / logical byte",
            "Measured µs",
            "Measured / torch",
        ),
        rows,
        text_columns=3,
    )
    masked: dict[int, float] = {}
    for p in predictions:
        if p["arm"] == "new":
            m = p["shape"][0]
            masked[m] = min(p["masked_row_fraction"], masked.get(m, 1.0))
    smallest_bm = min(
        p["config"]["block_m"]
        for p in predictions
        if p["arm"] == "new" and p["family"] != "split_k"
    )
    split_bm = sorted({p["config"]["block_m"] for p in predictions if p["family"] == "split_k"})
    padding = "; ".join(f"M={m}: {_pct(v)}" for m, v in sorted(masked.items()))
    sparse = "; ".join(
        f"M={p['shape'][0]}: {p['matmul_programs']} programs, so at least "
        f"{p['sm_count'] - p['matmul_programs']} SMs get none"
        for p in predictions
        if p["arm"] == "new" and p["bank1_selected"] and p["matmul_programs"] < p["sm_count"]
    )
    origins = {
        "old / tile": "Grouped tl.load/tl.dot matmul. The four new tile actions use larger launch parameters.",
        "persistent": "Adapted from Triton 3.4.0's persistent-matmul tutorial; warp specialization and epilogue subtiling are disabled.",
        "split_k": "Tensor-core partial tiles, an FP32 workspace and a separate reduction with one FP16 conversion.",
    }
    sources = "".join(
        f"<li><strong>{_e(p['family'])}</strong> ({_repo_link(p['source'])}): "
        f"{_e(origins[p['family']])}</li>"
        for p in dossier["provenance"]
    )
    formulas = _dl(
        (name.replace("_", " "), f"<code>{_e(text)}</code>")
        for name, text in dossier["formulas"].items()
    )
    return f"""
<section id="dossier" aria-labelledby="dossier-title">
<h2 id="dossier-title">Kernel layout</h2>
<p class="callout inference">Inference: these program counts, padding fractions and workspace sizes
come from the source configurations, not profiler measurements.</p>
<h3>Implementations</h3>
<ul>{sources}</ul>
<h3>Selected actions next to their predictions</h3>
{table}
<p>The smallest masked-row fraction among all applicable new actions on this projection is {_e(padding)}. Every
ordinary and persistent new action has BM ≥ {smallest_bm}; only split-K, for M ≤ 31, uses
BM = {_e(", ".join(map(str, split_bm)))}. Some selected new actions launch fewer matmul programs than the
{dossier["sm_count"]} SMs: {_e(sparse)}. Profiler counters would be needed to measure the cost of
padding, occupancy and memory traffic.</p>
<p class="note">{_repo_link(DIAGNOSIS)} contains all {len(predictions)} configurations under
<code>dossier</code>. {_repo_link("tests/test_kernel_dossier.py")} checks the formulas.</p>
<h3>Formulas</h3>
{formulas}
</section>"""


def _archive_section() -> str:
    reports = "".join(
        f'<li><a href="{_e(href)}">{_e(title)}</a>: {_e(text)}</li>'
        for href, title, text in ARCHIVE
    )
    data = " · ".join(f'<a href="{_e(href)}">{_e(text)}</a>' for href, text in ARCHIVE_DATA)
    return f"""
<section id="archive" aria-labelledby="archive-title">
<h2 id="archive-title">Other reports</h2>
<p>Older studies, kept unchanged because the release manifests hash them.</p>
<ul>{reports}</ul>
<p class="note">Data published with the transfer reports: {data}.</p>
</section>"""


def render(
    summary: Mapping[str, Any], diagnosis: Mapping[str, Any], digests: Mapping[str, str]
) -> str:
    """Return the complete landing page; identical inputs give identical bytes."""
    _consistent(summary, diagnosis)
    overall = summary["overall"]
    arms = overall["arms"]
    new = arms["new"]
    strip = "".join(
        f"<div><strong>{_e(value)}</strong><span>{_e(label)}</span></div>"
        for value, label in (
            (
                f"{overall['named_workload_count']}",
                "LLM-shaped workloads on T4, A10, L4 and H100",
            ),
            (
                f"{(1 - overall['old_new_comparison']['geometric_mean_new_to_old']) * 100:.0f}%",
                "lower H100 latency after expanding the kernel set",
            ),
            (f"{new['geometric_mean_to_torch']:.2f}×", "new H100 actions / torch latency"),
            (
                f"{new['comparable_win_count']} / {overall['named_workload_count']}",
                "H100 workloads faster than torch",
            ),
        )
    )
    inputs = "".join(
        f"<li>{_repo_link(path)} <code>sha256:{_e(digest)}</code></li>"
        for path, digest in digests.items()
    )
    sections = "\n".join(
        (
            _gpu_section(diagnosis),
            _result_section(summary, diagnosis),
            _split_section(summary, diagnosis),
            _contract_section(summary),
            _regime_section(diagnosis),
            _bank_section(diagnosis),
            _dossier_section(diagnosis),
            _archive_section(),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta http-equiv="Content-Security-Policy" content="{_CSP}">
<title>quartet · Triton matmul tuning across four GPUs</title>
<style>{_CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to results</a>
<div class="page">
<header>
<p class="eyebrow">quartet · <a href="{REPOSITORY}">source</a></p>
<h1>Triton matmul tuning across four GPUs</h1>
<p class="lede">A Triton autotuner measured against torch.matmul on T4, A10, L4 and H100. In the
original four-GPU comparison, the configuration picked on one timing pass was within 0.7% of the
fastest of the 36 on a second pass, yet Triton on the H100 still trailed torch. Ten new kernels cut
its H100 latency by 27%.</p>
<div class="strip">{strip}</div>
<nav aria-label="Sections"><ul>
<li><a href="#gpus">Four GPUs</a></li><li><a href="#result">H100 expansion</a></li>
<li><a href="#split">Selection</a></li><li><a href="#contract">Setup</a></li>
<li><a href="#regimes">By M</a></li><li><a href="#bank">Selection loss</a></li>
<li><a href="#dossier">Kernel layout</a></li><li><a href="#archive">Other reports</a></li>
</ul></nav>
</header>
<main id="main">
{sections}
</main>
<footer>
<p>Generated by {_repo_link("scripts/build_site.py")} from:</p>
<ul>{inputs}</ul>
<p>Regenerate with <code>uv run python scripts/diagnose_kernel_bank.py</code>, then
<code>uv run python scripts/build_site.py</code>; both accept <code>--check</code>. See also the
{_repo_link("README.md", "README")}, {_repo_link("results/action-expansion-raw.json", "raw measurements")} and
{_repo_link("src/heliostune/action_configs.py", "action manifest")}. MIT license.</p>
</footer>
</div>
</body>
</html>
"""


def build(repo: Path = _REPO) -> str:
    summary = cast(Mapping[str, Any], read_json(repo / SUMMARY))
    diagnosis = cast(Mapping[str, Any], read_json(repo / DIAGNOSIS))
    digests = {path: _sha256(repo / path) for path in (SUMMARY, DIAGNOSIS)}
    return render(summary, diagnosis, digests)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_REPO / OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="fail unless --output is byte-identical"
    )
    args = parser.parse_args(argv)
    page = build()
    diagnosis = cast(Mapping[str, Any], read_json(_REPO / DIAGNOSIS))
    figure = _regime_chart(diagnosis["expansion"]["by_regime"]) + "\n"
    outputs = (
        (args.output, page.encode("utf-8")),
        (args.output.with_name("h100-by-m.svg"), figure.encode("utf-8")),
    )
    for path, payload in outputs:
        if args.check:
            # Compare raw bytes: text mode would hide CRLF or other newline rewrites.
            if not path.is_file() or path.read_bytes() != payload:
                raise SystemExit(f"{path} is stale; rerun scripts/build_site.py")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        print(f"{path.name}: {len(payload)} bytes; {'byte-exact' if args.check else 'written'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
