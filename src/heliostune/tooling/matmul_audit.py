"""Descriptive torch-versus-action-set audit of an existing three-bank timing matrix."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from html import escape
from statistics import median
from typing import Any

from heliostune.configs import Workload
from heliostune.replay import BenchmarkTable


@dataclass(frozen=True, slots=True)
class AuditRow:
    workload: Workload
    reference_config: str
    reference_ms: float
    torch_ms: float
    reference_p20_ms: float | None
    reference_p80_ms: float | None
    torch_p20_ms: float | None
    torch_p80_ms: float | None
    best_bank2_config: str
    best_bank2_ms: float

    @property
    def ratio(self) -> float:
        """Reference latency / torch latency; above one favors torch."""
        return self.reference_ms / self.torch_ms

    @property
    def saved_us(self) -> float:
        return (self.reference_ms - self.torch_ms) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload.key,
            **asdict(self),
            "reference_over_torch": self.ratio,
            "best_bank2_over_torch": self.best_bank2_ms / self.torch_ms,
            "torch_saved_us": self.saved_us,
        }


def _latency(value: float | None) -> float:
    # BenchmarkTable validates this already; keep the narrowing explicit for typing.
    if value is None:
        raise ValueError("validated matrix unexpectedly has no latency")
    return value


def audit_rows(table: BenchmarkTable, gpu: str) -> tuple[AuditRow, ...]:
    """Use exactly the replay's reference selection, tie-break and torch deduplication."""
    table.validate_matrix(gpu, (0, 1, 2))
    configs = table.configs(gpu)
    rows = []
    for workload in table.workloads(gpu):
        reference = table.reference_config(gpu, workload)
        scored = table.get(gpu, workload, reference, 2)
        best = min(
            (table.get(gpu, workload, config, 2) for config in configs),
            key=lambda cell: (_latency(cell.latency_ms), cell.config.key),
        )
        rows.append(
            AuditRow(
                workload=workload,
                reference_config=reference.key,
                reference_ms=_latency(scored.latency_ms),
                torch_ms=scored.torch_latency_ms,
                reference_p20_ms=scored.latency_p20_ms,
                reference_p80_ms=scored.latency_p80_ms,
                torch_p20_ms=scored.torch_latency_p20_ms,
                torch_p80_ms=scored.torch_latency_p80_ms,
                best_bank2_config=best.config.key,
                best_bank2_ms=_latency(best.latency_ms),
            )
        )
    return tuple(rows)


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def summarize_rows(rows: Sequence[AuditRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("audit requires at least one workload")
    ratios = [row.ratio for row in rows]
    separated = [
        row for row in rows if row.reference_p20_ms is not None and row.torch_p80_ms is not None
    ]
    return {
        "workloads": len(rows),
        "unique_shapes": len({(r.workload.m, r.workload.n, r.workload.k) for r in rows}),
        "torch_wins_reference": sum(r.ratio > 1 for r in rows),
        "triton_wins_reference": sum(r.ratio < 1 for r in rows),
        "ties_reference": sum(r.ratio == 1 for r in rows),
        "reference_over_torch_geomean": geometric_mean(ratios),
        "reference_over_torch_median": median(ratios),
        "reference_over_torch_min": min(ratios),
        "reference_over_torch_max": max(ratios),
        "torch_saved_us_median": median([r.saved_us for r in rows]),
        "torch_saved_us_min": min(r.saved_us for r in rows),
        "torch_saved_us_max": max(r.saved_us for r in rows),
        "torch_wins_best_bank2": sum(r.best_bank2_ms > r.torch_ms for r in rows),
        "triton_wins_best_bank2": sum(r.best_bank2_ms < r.torch_ms for r in rows),
        "ties_best_bank2": sum(r.best_bank2_ms == r.torch_ms for r in rows),
        "best_bank2_over_torch_geomean": geometric_mean(
            [r.best_bank2_ms / r.torch_ms for r in rows]
        ),
        "spread_comparisons_available": len(separated),
        "torch_p80_below_reference_p20": sum(
            r.torch_p80_ms is not None
            and r.reference_p20_ms is not None
            and r.torch_p80_ms < r.reference_p20_ms
            for r in rows
        ),
    }


def gpu_overview(table: BenchmarkTable) -> list[dict[str, Any]]:
    """Summarize each device separately; never pool acquisition stages."""
    return [
        {
            "gpu": gpu,
            "hardware": table.hardware(gpu).to_dict(),
            "configs": len(table.configs(gpu)),
            "banks": list(table.banks(gpu)),
            **summarize_rows(audit_rows(table, gpu)),
        }
        for gpu in table.gpus
    ]


def build_audit(table: BenchmarkTable, gpu: str = "H100") -> dict[str, Any]:
    rows = audit_rows(table, gpu)
    by_m: dict[int, list[AuditRow]] = defaultdict(list)
    by_model: dict[str, list[AuditRow]] = defaultdict(list)
    by_projection: dict[str, list[AuditRow]] = defaultdict(list)
    by_shape: dict[tuple[int, int, int], list[AuditRow]] = defaultdict(list)
    for row in rows:
        w = row.workload
        by_m[w.m].append(row)
        by_model[w.model].append(row)
        by_projection[w.projection].append(row)
        by_shape[w.m, w.n, w.k].append(row)
    folds = [summarize_rows(by_model[model]) for model in sorted(by_model)]
    return {
        "schema_version": 1,
        "analysis_kind": "descriptive audit of committed measurements; no new timing or policy replay",
        "target_gpu": gpu,
        "hardware": table.hardware(gpu).to_dict(),
        "configs": len(table.configs(gpu)),
        "banks": list(table.banks(gpu)),
        "definitions": {
            "reference": "minimum bank-1 latency, ties by config key; that config scored on bank 2",
            "ratio": "bank-2 reference latency / bank-2 torch latency; above 1 favors torch",
            "torch_saved_us": "1000 * (bank-2 reference latency_ms - bank-2 torch latency_ms)",
            "wins": "strict comparison of stored median latencies, not a significance test",
            "aggregation": "geometric mean of workload ratios within each displayed group",
            "equal_fold_mean": "arithmetic mean of per-model geometric means, matching the published torch curve",
            "shape_duplicates": "distinct model/projection workloads kept; shape groups geometrically average them, not independent devices",
            "best_bank2": "minimum of all curated config latencies on scoring bank 2; optimistic in-sample diagnostic, not an independently selected deployable policy",
            "spread": "count of torch p80 < reference p20; within-run central spreads, not confidence intervals or paired tests",
        },
        "overall": {
            **summarize_rows(rows),
            "reference_over_torch_equal_fold_mean": math.fsum(
                fold["reference_over_torch_geomean"] for fold in folds
            )
            / len(folds),
        },
        "by_m": [{"m": m, **summarize_rows(by_m[m])} for m in sorted(by_m)],
        "by_model": [
            {"model": model, **summarize_rows(by_model[model])} for model in sorted(by_model)
        ],
        "by_projection": [
            {"projection": p, **summarize_rows(by_projection[p])} for p in sorted(by_projection)
        ],
        "by_shape": [
            {"m": m, "n": n, "k": k, **summarize_rows(by_shape[m, n, k])}
            for m, n, k in sorted(by_shape)
        ],
        "workloads": [row.to_dict() for row in rows],
        "limits": [
            "One fixed corpus and collected GPU instances; no new-device uncertainty estimate.",
            "Torch timings are duplicated across config rows and counted once per workload/bank.",
            "The curated reference is not a hardware ceiling; no causal hardware bottleneck is identified.",
            "Bank-2 best-config selection is optimistic and cannot be used as a held-out policy score.",
            "Kernel profiling, matched stricter accumulation and deployment cost are not measured by this audit.",
        ],
    }


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#18212b">',
    ]


def _text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def render_by_m(audit: dict[str, Any]) -> str:
    groups = audit["by_m"]
    records = audit["workloads"]
    height = 180 + 58 * len(groups)
    lines = _svg_start(
        980,
        height,
        "H100: where torch beats the curated Triton reference",
        "Each dot is one workload. Diamonds are group geometric means. "
        "Reference selected on bank 1; both latencies scored on bank 2. "
        "Logarithmic ratio axis; above one favors torch. Ranges are workload spread, not uncertainty.",
    )
    lines += [
        _text(24, 32, "H100: where torch beats the curated Triton reference", size=21),
        _text(
            24,
            57,
            "Bank-1 selection, bank-2 medians · one dot per workload · diamond = geometric mean",
        ),
    ]
    low = min(-1, math.floor(math.log2(min(r["reference_over_torch"] for r in records))))
    high = max(1, math.ceil(math.log2(max(r["reference_over_torch"] for r in records))))

    def x(ratio: float) -> float:
        return 150 + 580 * (math.log2(ratio) - low) / (high - low)

    for power in range(low, high + 1):
        xpos = x(2.0**power)
        color = "#18212b" if power == 0 else "#dce2e8"
        lines.append(
            f'<path d="M {xpos:.2f} 92 V {height - 72}" stroke="{color}" '
            f'stroke-dasharray="{("none" if power == 0 else "3 4")}"/>'
        )
        lines.append(_text(xpos, height - 49, f"{2.0**power:g}×", anchor="middle"))
    lines.append(_text(764, 90, "torch wins / workloads", size=13))
    for index, group in enumerate(groups):
        y = 117 + index * 58
        lines.append(_text(24, y + 5, f"M = {group['m']}"))
        lines.append(
            f'<path d="M {x(group["reference_over_torch_min"]):.2f} {y} H '
            f'{x(group["reference_over_torch_max"]):.2f}" stroke="#90a4b5" stroke-width="2"/>'
        )
        subset = [r for r in records if r["workload"]["m"] == group["m"]]
        for j, record in enumerate(subset):
            ratio = record["reference_over_torch"]
            color = "#b04a27" if ratio > 1 else "#176f91"
            tip = (
                f"{record['workload_key']}: {ratio:.4f}×; "
                f"reference {1000 * record['reference_ms']:.3f} µs, "
                f"torch {1000 * record['torch_ms']:.3f} µs"
            )
            lines.append(
                f'<circle cx="{x(ratio):.2f}" cy="{y + (j % 5 - 2) * 3}" r="3.5" '
                f'fill="{color}" fill-opacity="0.75"><title>{escape(tip)}</title></circle>'
            )
        cx = x(group["reference_over_torch_geomean"])
        lines.append(
            f'<path d="M {cx:.2f} {y - 7} l 6 7 l -6 7 l -6 -7 Z" fill="#18212b" stroke="#fff"/>'
        )
        lines.append(
            _text(
                764,
                y + 5,
                f"{group['torch_wins_reference']}/{group['workloads']} · "
                f"{group['reference_over_torch_geomean']:.3f}×",
            )
        )
    lines += [
        _text(
            440,
            height - 22,
            "Reference latency / torch.matmul latency (log₂ axis)",
            anchor="middle",
        ),
        _text(754, height - 22, "<1 Triton wins · >1 torch wins", size=12),
        "</g>",
        "</svg>",
        "",
    ]
    return "\n".join(lines)


def render_shapes(audit: dict[str, Any]) -> str:
    shapes = audit["by_shape"]
    ms = sorted({s["m"] for s in shapes})
    nks = sorted({(s["n"], s["k"]) for s in shapes})
    lookup = {(s["m"], s["n"], s["k"]): s for s in shapes}
    width = 280 + 98 * len(ms)
    height = 185 + 35 * len(nks)
    lines = _svg_start(
        width,
        height,
        "H100 torch advantage by matrix shape",
        "Rows give N and K; columns give M for A[M,K] @ B[K,N]. "
        "Each cell is the geometric mean reference/torch ratio for that shape. "
        "Duplicate shapes across model families are averaged, not extra devices. "
        "Orange favors torch, blue favors Triton. Cell ratios are rounded to two decimals; "
        "tooltips give six decimals. A displayed 1.00 need not be a tie.",
    )
    lines += [
        _text(20, 32, "H100 torch advantage by matrix shape", size=21),
        _text(20, 57, "Bank-2 reference / torch medians · reference selected on bank 1"),
        _text(
            20,
            80,
            "Repeated shapes across models: geometric mean; not independent devices",
            size=12,
        ),
        _text(20, 109, "N × K", size=13),
    ]
    for j, m in enumerate(ms):
        lines.append(_text(279 + j * 98, 109, f"M = {m}", size=13, anchor="middle"))
    for i, (n, k) in enumerate(nks):
        y = 121 + i * 35
        lines.append(_text(20, y + 21, f"{n} × {k}"))
        for j, m in enumerate(ms):
            x = 232 + j * 98
            group = lookup.get((m, n, k))
            if group is None:
                lines.append(_text(x + 47, y + 21, "—", anchor="middle"))
                continue
            ratio = group["reference_over_torch_geomean"]
            strength = min(1.0, abs(math.log2(ratio)) / 3)
            base = (190, 77, 36) if ratio > 1 else (30, 114, 150)
            rgb = tuple(round(248 + strength * (component - 248)) for component in base)
            color = "#" + "".join(f"{component:02x}" for component in rgb)
            tip = (
                f"M={m}, N={n}, K={k}: {ratio:.6f}×; "
                f"torch wins {group['torch_wins_reference']}/{group['workloads']} workloads; "
                f"range {group['reference_over_torch_min']:.6f}–{group['reference_over_torch_max']:.6f}×"
            )
            lines.append(
                f'<rect x="{x}" y="{y}" width="94" height="31" rx="3" '
                f'fill="{color}"><title>{escape(tip)}</title></rect>'
            )
            ink = "#fff" if strength > 0.65 else "#18212b"
            lines.append(
                f'<text x="{x + 47}" y="{y + 21}" text-anchor="middle" '
                f'font-size="14" fill="{ink}">{ratio:.2f}×</text>'
            )
    lines += [
        _text(
            20, height - 32, "<1×: Triton reference wins (blue) · >1×: torch wins (orange)", size=13
        ),
        _text(
            20,
            height - 12,
            "Color saturates at ⅛× and 8×. Rounded 1.00× need not be a tie; tooltips give six decimals.",
            size=12,
        ),
        "</g>",
        "</svg>",
        "",
    ]
    return "\n".join(lines)
