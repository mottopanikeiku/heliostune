"""The Pages landing page is regenerated from committed results and links only real files."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
from decimal import Decimal, localcontext
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

import pytest
from hypothesis import given
from hypothesis import strategies as st

_REPO = Path(__file__).resolve().parents[1]
_SITE = _REPO / "site"


def _load() -> ModuleType:
    name = "_test_build_site"
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts/build_site.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BUILDER = _load()
_PAGE = (_SITE / "index.html").read_text(encoding="utf-8")
_SUMMARY: dict[str, Any] = json.loads((_REPO / _BUILDER.SUMMARY).read_text(encoding="utf-8"))
_DIAGNOSIS: dict[str, Any] = json.loads((_REPO / _BUILDER.DIAGNOSIS).read_text(encoding="utf-8"))


class _Page(HTMLParser):
    """Collect links, ids, scripts and table cell text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.tags: set[str] = set()
        self.tables: dict[str, list[list[str]]] = {}
        self._table: str | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.tags.add(tag)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        for attribute in ("href", "src"):
            if values.get(attribute) is not None:
                self.links.append(str(values[attribute]))
        if tag == "table":
            self._table = str(values["id"])
            self.tables[self._table] = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            self.tables[self._table].append(self._row)
            self._row = None
        elif tag == "table":
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _parse(page: str = _PAGE) -> _Page:
    parser = _Page()
    parser.feed(page)
    return parser


def _rows(table: str) -> dict[str, list[str]]:
    return {row[0]: row[1:] for row in _parse().tables[table][1:]}


def test_committed_page_is_byte_identical_to_a_fresh_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _BUILDER.build() == _PAGE
    output = tmp_path / "index.html"
    assert _BUILDER.main(["--output", str(output)]) == 0
    assert output.read_bytes() == (_SITE / "index.html").read_bytes()
    assert _BUILDER.main(["--output", str(output), "--check"]) == 0
    assert "byte-exact" in capsys.readouterr().out
    output.write_text(_PAGE.replace("1.174", "1.000", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="stale"):
        _BUILDER.main(["--output", str(output), "--check"])


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda b: b.replace(b"\n", b"\r\n"), id="crlf"),
        pytest.param(lambda b: b.replace(b"\n", b"\r\n", 1), id="one-crlf"),
        pytest.param(lambda b: b + b"\n", id="trailing-newline"),
        pytest.param(lambda b: b + b" ", id="trailing-space"),
        pytest.param(lambda b: b"\xef\xbb\xbf" + b, id="utf8-bom"),
        pytest.param(lambda b: b[:-1], id="truncated"),
        # Text mode reads a final "\r" as "\n", so only a raw-byte check sees this.
        pytest.param(lambda b: b[:-1] + b"\r", id="final-cr"),
        pytest.param(lambda b: b + b"\r", id="trailing-cr"),
    ],
)
def test_check_compares_raw_bytes(tmp_path: Path, mutate: Any) -> None:
    committed = (_SITE / "index.html").read_bytes()
    output = tmp_path / "index.html"
    output.write_bytes(mutate(committed))
    assert output.read_bytes() != committed
    with pytest.raises(SystemExit, match="stale"):
        _BUILDER.main(["--output", str(output), "--check"])


def test_check_rejects_a_missing_page(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="stale"):
        _BUILDER.main(["--output", str(tmp_path / "absent.html"), "--check"])


@pytest.mark.parametrize(
    ("value", "upper", "expected"),
    [
        (0.0013095868, True, "0.14%"),  # H100 archive worst bank pair
        (0.0082674012, True, "0.83%"),  # every multi-bank view
        (0.0013095868, False, "0.13%"),
        (0.25, True, "25.00%"),  # exactly representable: no spurious step up
        (0.0, True, "0.00%"),
    ],
)
def test_bound_pct_rounds_towards_the_safe_side(value: float, upper: bool, expected: str) -> None:
    assert _BUILDER.bound_pct(value, upper=upper) == expected


@given(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=4),
)
def test_bound_pct_never_understates_an_upper_or_overstates_a_lower_bound(
    value: float, digits: int
) -> None:
    with localcontext(prec=1100):
        exact = Decimal(value) * 100  # exact: no context rounding
    upper = Decimal(_BUILDER.bound_pct(value, upper=True, digits=digits).removesuffix("%"))
    lower = Decimal(_BUILDER.bound_pct(value, upper=False, digits=digits).removesuffix("%"))
    step = Decimal(1).scaleb(-digits)
    assert lower <= exact <= upper
    assert upper - lower <= step


def test_bound_pct_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _BUILDER.bound_pct(math.nan, upper=True)


def test_page_states_the_rounded_up_selection_bound() -> None:
    picks = [
        p["full_roster"]["overall"]["policy_oracle_ratio"]["named"]
        for d in _DIAGNOSIS["datasets"]
        for p in d["bank_pairs"]
        if p["full_roster"]["overall"]["named_count"]
    ]
    bound = _BUILDER.bound_pct(max(picks) - 1, upper=True)
    assert bound == "0.83%"
    assert f"at most {bound} slower" in _PAGE


def test_render_is_deterministic_and_binds_input_digests() -> None:
    digests = {_BUILDER.SUMMARY: "a" * 64, _BUILDER.DIAGNOSIS: "b" * 64}
    first = _BUILDER.render(_SUMMARY, _DIAGNOSIS, digests)
    again = _BUILDER.render(copy.deepcopy(_SUMMARY), copy.deepcopy(_DIAGNOSIS), dict(digests))
    assert first == again
    assert f"sha256:{'a' * 64}" in first and f"sha256:{'b' * 64}" in first


def test_mismatched_summary_and_diagnosis_are_rejected() -> None:
    digests = {_BUILDER.SUMMARY: "a" * 64, _BUILDER.DIAGNOSIS: "b" * 64}
    stale = copy.deepcopy(_SUMMARY)
    stale["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different raw measurements"):
        _BUILDER.render(stale, _DIAGNOSIS, digests)
    changed = copy.deepcopy(_DIAGNOSIS)
    changed["expansion"]["overall"]["new_torch_ratio"]["named"] = 1.0
    with pytest.raises(ValueError, match="disagree on the new geometric mean"):
        _BUILDER.render(_SUMMARY, changed, digests)


def test_headline_table_matches_the_summary_and_diagnosis() -> None:
    arms = _SUMMARY["overall"]["arms"]
    view = _DIAGNOSIS["expansion"]["overall"]
    rows = _rows("headline")
    for arm, label in (("old", "Old 36"), ("new", "New actions"), ("union", "Expanded union")):
        assert rows[label] == [
            str(arms[arm]["comparable_win_count"]),
            str(arms[arm]["comparable_tie_count"]),
            str(arms[arm]["comparable_loss_count"]),
            f"{arms[arm]['geometric_mean_to_torch']:.3f}×",
            f"{view[f'{arm}_torch_ratio']['unique_shape']:.3f}×",
            f"{view[f'{arm}_us']['named']:,.1f}",
            f"{view[f'{arm}_us']['unique_shape']:,.1f}",
        ]
    assert rows["torch.matmul"][-2:] == [
        f"{view['torch_us']['named']:,.1f}",
        f"{view['torch_us']['unique_shape']:,.1f}",
    ]
    ratio = _SUMMARY["overall"]["old_new_comparison"]["geometric_mean_new_to_old"]
    assert f"<strong>{ratio:.3f}</strong>" in _PAGE


def test_wins_table_lists_every_summary_win() -> None:
    table = _parse().tables["wins"][1:]
    workloads = {w["workload_key"]: w for w in _SUMMARY["workloads"]}
    wins = [workloads[w["workload_key"]] for w in _SUMMARY["largest_wins"]["new"]]
    assert len(table) == _SUMMARY["overall"]["arms"]["new"]["comparable_win_count"] == len(wins)
    for row, win in zip(table, wins, strict=True):
        assert row[1] == "({},{},{})".format(*win["shape"])
        assert row[2] == win["selected"]["new"]["family"]
        assert row[3:] == [
            f"{win['new_ms'] * 1000:.3f}",
            f"{win['torch_ms'] * 1000:.3f}",
            f"{win['new_to_torch']:.6f}",
        ]


def test_regime_table_matches_summary_counts_and_diagnosis_means() -> None:
    table = _parse().tables["regimes"][1:]
    regimes = _DIAGNOSIS["expansion"]["by_regime"]
    by_m = {m["m"]: m for m in _SUMMARY["by_m"]}
    assert len(table) == len(regimes) == len(by_m)
    for row, regime in zip(table, regimes, strict=True):
        (m,) = regime["m_values"]
        arms = by_m[m]["arms"]["new"]
        assert row[0] == f"M={m} ({regime['regime']})"
        assert row[1] == f"{by_m[m]['named_workload_count']} / {by_m[m]['unique_shape_count']}"
        assert row[6] == f"{arms['geometric_mean_to_torch']:.3f}×"
        assert row[7] == f"{regime['new_torch_ratio']['unique_shape']:.3f}×"
        assert row[8] == (
            f"{arms['comparable_win_count']} / {arms['comparable_tie_count']} / "
            f"{arms['comparable_loss_count']}"
        )
        families = [
            w["selected"]["new"]["family"] for w in _SUMMARY["workloads"] if w["shape"][0] == m
        ]
        assert row[9] == ", ".join(f"{f} {families.count(f)}" for f in sorted(set(families)))


def test_bank_table_reports_every_multi_bank_view_without_pooling() -> None:
    table = _parse().tables["bank-pairs"][1:]
    multi = [
        d
        for d in _DIAGNOSIS["datasets"]
        if any(p["full_roster"]["overall"]["named_count"] for p in d["bank_pairs"])
    ]
    assert len(table) == len(multi)
    for row, dataset in zip(table, multi, strict=True):
        assert row[0] == f"{dataset['path']} {dataset['hardware']['device_name']}"
        assert row[2] == f"{sum(b['observed_cells'] for b in dataset['banks']):,}"
    single = _parse().tables["single-bank"][1:]
    assert len(single) == sum(not d["bank_pairs"] for d in _DIAGNOSIS["datasets"])
    # The expansion's finalist-only pair is never shown as a full-roster pair.
    assert not any(row[0].startswith("results/") for row in table)


def test_dossier_table_shows_only_bank1_finalists_with_their_measurements() -> None:
    table = _parse().tables["dossier"][1:]
    selected = [p for p in _DIAGNOSIS["dossier"]["predictions"] if p["bank1_selected"]]
    assert len(table) == len(selected)
    for row, p in zip(table, selected, strict=True):
        assert row[2] == p["config_key"]
        assert row[5] == f"{p['masked_row_fraction'] * 100:.1f}%"
        assert row[-1] == f"{p['bank2_measurement']['ratio_to_torch']:.3f}×"
    assert "Inference" in _PAGE and "not profiler measurements" in _PAGE


def test_local_links_and_fragments_resolve() -> None:
    page = _parse()
    external = set()
    for link in page.links:
        parts = urlsplit(link)
        if not parts.scheme:
            target = parts.path
            if target:
                assert (_SITE / target).is_file(), link
            if parts.fragment:
                assert target == "" and parts.fragment in page.ids, link
            continue
        assert parts.scheme == "https", link
        if link.startswith(_BUILDER.BLOB):
            assert (_REPO / link.removeprefix(_BUILDER.BLOB)).is_file(), link
        elif link.startswith(f"{_BUILDER.REPOSITORY}/commit/"):
            assert re.fullmatch(r"[0-9a-f]{40}", link.rsplit("/", 1)[1]), link
        else:
            external.add(link)
    assert external == {_BUILDER.REPOSITORY}


def test_every_published_site_file_is_reachable_from_the_landing_page() -> None:
    linked = {urlsplit(link).path for link in _parse().links if not urlsplit(link).scheme}
    published = {p.name for p in _SITE.iterdir() if p.is_file() and p.name != "index.html"}
    assert published <= linked
    assert "transfer.html" in linked


def test_page_is_static_offline_and_carries_a_strict_policy() -> None:
    page = _parse()
    assert "script" not in page.tags and "iframe" not in page.tags and "img" not in page.tags
    assert "script-src 'none'" in _PAGE and "default-src 'none'" in _PAGE
    assert "<svg" in _PAGE and "http://" not in _PAGE.replace("http://www.w3.org/2000/svg", "")
    for marker in ("Not policy replay", "unidentifiable", "MEAS", "no new timings"):
        assert marker in _PAGE


def test_historical_landing_page_is_preserved_byte_for_byte() -> None:
    baseline = json.loads(
        (_REPO / "benchmarks/historical-artifact-baseline.json").read_text(encoding="utf-8")
    )
    frozen = baseline["present_artifacts"]["site/index.html"]
    transfer = (_SITE / "transfer.html").read_bytes()
    assert len(transfer) == frozen["bytes"]
    assert hashlib.sha256(transfer).hexdigest() == frozen["sha256"]
    assert hashlib.sha256(_PAGE.encode()).hexdigest() != frozen["sha256"]
