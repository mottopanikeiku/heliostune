"""Strict standalone reports for the Hopper, precision, and fusion remote studies."""

from __future__ import annotations

import html
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from urllib.parse import quote, urlsplit

from heliostune.artifacts import write_text_atomic
from heliostune.errors import SchemaError
from heliostune.schema import HardwareProfile
from heliostune.validation import (
    exact_bool,
    exact_fields,
    exact_int,
    exact_object,
    finite_float,
    nonblank_string,
    optional_finite_float,
)

HOPPER_STUDY_ID: Literal["hopper-h100-engineering-benchmark"] = "hopper-h100-engineering-benchmark"
PRECISION_STUDY_ID: Literal["h100-fp16-reduction-probe"] = "h100-fp16-reduction-probe"
FUSION_REMOTE_STUDY_ID: Literal["fusion-remote-h100-exploratory"] = "fusion-remote-h100-exploratory"
FUSION_REMOTE_SCHEMA = "heliostune.fusion-remote-exploratory.summary/1"
FUSION_REMOTE_RAW_PATH = "benchmarks/data/fusion-remote-exploratory.json.zst"
FUSION_REMOTE_SUMMARY_PATH = "benchmarks/results/fusion-remote-exploratory-summary.json"
FUSION_REMOTE_MANIFEST_PATH = "benchmarks/fusion-remote-exploratory-manifest.json"
ENGINEERING_STUDY_IDS = frozenset({HOPPER_STUDY_ID, PRECISION_STUDY_ID, FUSION_REMOTE_STUDY_ID})

_HEX = frozenset("0123456789abcdef")
_NUMERIC_REL_TOLERANCE = 1e-12
_NUMERIC_ABS_TOLERANCE = 1e-12

_FUSION_UNRESOLVED_SEQUENCE = (
    "retrieval returned 401; client then requested cancellation; "
    "terminal provider outcome/cancellation success remained unresolved"
)


@dataclass(frozen=True, slots=True)
class ClaimClassification:
    candidate_role: str
    claim_kind: str
    comparator_role: str
    decision: str
    evidence_class: str
    inferential: bool
    limitations: tuple[str, ...]
    reference_role: str
    scope: str


@dataclass(frozen=True, slots=True)
class AttemptAccounting:
    attempted: int
    completed: int
    failed: int
    retried: int


@dataclass(frozen=True, slots=True)
class RowAccounting:
    expected: int
    failed: int
    omitted: int
    published: int


@dataclass(frozen=True, slots=True)
class RateSource:
    checked_at_utc: str
    path: str
    sha256: str
    url: str


@dataclass(frozen=True, slots=True)
class PublishedRateEstimate:
    amount_usd: float
    classification: str
    gpu_rate_usd_per_second: float
    limitations: str
    rate_source: RateSource
    time_basis: str


@dataclass(frozen=True, slots=True)
class ExcludedCall:
    actual_h100_cost_usd: float | None
    actual_h100_cost_unknown_reason: str
    call_ids: tuple[str, ...]
    role: str


@dataclass(frozen=True, slots=True)
class CostAccounting:
    actual_h100_cost_usd: float | None
    actual_h100_cost_unknown_reason: str
    covered_call_ids: tuple[str, ...]
    excluded_calls: tuple[ExcludedCall, ...]
    published_rate_estimate: PublishedRateEstimate
    scope: str


@dataclass(frozen=True, slots=True)
class CollectionAccounting:
    attempts: AttemptAccounting
    cost: CostAccounting
    elapsed_seconds: float
    rows: RowAccounting


@dataclass(frozen=True, slots=True)
class PublishedRaw:
    path: str
    sha256: str
    uncompressed_sha256: str
    bytes: int
    rows: int
    schema: str | None
    schema_version: int | None


@dataclass(frozen=True, slots=True)
class PublishedJournal:
    path: str
    sha256: str
    records: int


@dataclass(frozen=True, slots=True)
class ModalPublication:
    app_url: str
    app_id: str
    app_url_provenance: str
    call_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Publication:
    raw: PublishedRaw
    journal: PublishedJournal
    manifest_path: str
    head_commit: str
    source_sha256: str
    wheel_sha256: str
    hardware: HardwareProfile
    runtime: tuple[tuple[str, str], ...]
    modal: ModalPublication


@dataclass(frozen=True, slots=True)
class EvidenceFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RawEvidenceFile:
    path: str
    compressed_sha256: str
    decompressed_sha256: str
    schema: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class CommittedEvidence:
    attempt_journal: EvidenceFile
    manifest: EvidenceFile
    raw: RawEvidenceFile
    summary: EvidenceFile


@dataclass(frozen=True, slots=True)
class PrecisionMetrics:
    archive_baseline_torch_over_best_triton: float
    paired_strict_slowdown: float
    torch_reduced_over_torch_strict_median: float
    torch_reduced_over_triton_median: float
    torch_strict_over_triton_median: float


@dataclass(frozen=True, slots=True)
class PrecisionThresholds:
    baseline_absolute_tolerance: float
    baseline_relative_tolerance: float
    meaningful_paired_effect: float


@dataclass(frozen=True, slots=True)
class PrecisionFinding:
    accuracy_regression: bool
    baseline_agrees: bool
    classification: str
    conclusion: str
    metrics: PrecisionMetrics
    parity_authorized: bool
    thresholds: PrecisionThresholds
    committed_evidence: CommittedEvidence | None


@dataclass(frozen=True, slots=True)
class HopperProtocol:
    bank: int
    gpu: str
    ratio: str
    row_count: int
    workload_count: int


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    block_k: int
    block_m: int
    block_n: int
    num_stages: int
    num_warps: int
    split_k: int | None
    epilogue_subtile: bool | None
    group_m: int | None
    warp_specialize: bool | None

    def display(self) -> str:
        common = (
            f"BM={self.block_m}, BN={self.block_n}, BK={self.block_k}, "
            f"warps={self.num_warps}, stages={self.num_stages}"
        )
        if self.split_k is not None:
            return f"{common}, split-K={self.split_k}"
        return (
            f"{common}, group-M={self.group_m}, "
            f"epilogue-subtile={_bool(self.epilogue_subtile is True)}, "
            f"warp-specialize={_bool(self.warp_specialize is True)}"
        )


RegimeName = Literal["hopper_gemm", "skinny_gemv"]


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    archive_ratio: float
    best_candidate_ms: float
    best_config: CandidateConfig
    best_config_key: str
    correct: bool
    regime: RegimeName
    torch_ms: float
    torch_over_best_candidate: float
    workload_key: str


@dataclass(frozen=True, slots=True)
class RegimeResult:
    name: RegimeName
    all_selected_correct: bool
    decision: str
    geometric_mean_speedup: float
    maximum_speedup: float
    median_speedup: float
    minimum_speedup: float
    percent_at_least_five_percent_faster: float
    workload_count: int
    workloads_at_least_five_percent_faster: int


@dataclass(frozen=True, slots=True)
class CostScreen:
    evaluated_independently_by_regime: bool
    geometric_mean_speedup_threshold: float
    required_fraction_at_least_five_percent_faster: float
    speedup_threshold_for_win: float


@dataclass(frozen=True, slots=True)
class ContextualBaseline:
    display_value: str
    frozen_value: float
    role: str


@dataclass(frozen=True, slots=True)
class HopperSummary:
    analysis_status: str
    candidate_selection: tuple[CandidateSelection, ...]
    claim: str
    claim_classification: ClaimClassification
    collection_accounting: CollectionAccounting
    contextual_baseline: ContextualBaseline
    cost_screen: CostScreen
    evidence_scope: str
    global_decision: str
    limitations: tuple[str, ...]
    precision_finding: PrecisionFinding
    protocol: HopperProtocol
    publication: Publication
    regimes: tuple[RegimeResult, ...]
    schema_version: int
    study_id: Literal["hopper-h100-engineering-benchmark"]
    three_bank_collection_performed: bool


@dataclass(frozen=True, slots=True)
class PrecisionProtocol:
    arm_order: str
    arm_order_seed: str
    arms: tuple[str, ...]
    banks: tuple[int, ...]
    gpu: str
    modal_selector: str
    probe: str
    quantiles: tuple[float, ...]
    reference: str
    rep_ms: int
    retry_policy: str
    role: str
    schema_version: int
    statistic: str
    tensor_seed: str
    tensor_seed_protocol: str
    warmup_ms: int
    workload_count: int


@dataclass(frozen=True, slots=True)
class PrecisionWorkload:
    archive_ratio: float
    banks: tuple[int, ...]
    torch_reduced_error: float
    torch_strict_error: float
    triton_error: float
    k: int
    torch_reduced_ms: float
    torch_strict_ms: float
    triton_ms: float
    m: int
    n: int
    torch_reduced_over_torch_strict: float
    torch_reduced_over_triton: float
    torch_strict_over_triton: float
    workload_key: str


@dataclass(frozen=True, slots=True)
class PrecisionSummary:
    analysis_status: str
    claim_classification: ClaimClassification
    collection_accounting: CollectionAccounting
    evidence_scope: str
    limitations: tuple[str, ...]
    precision_finding: PrecisionFinding
    protocol: PrecisionProtocol
    publication: Publication
    row_count: int
    schema_version: int
    study_id: Literal["h100-fp16-reduction-probe"]
    workload_count: int
    workloads: tuple[PrecisionWorkload, ...]


@dataclass(frozen=True, slots=True)
class FusionRemoteApp:
    app_id: str
    app_url: str
    artifact_binding: str
    identity_provenance: str


@dataclass(frozen=True, slots=True)
class FusionRemoteCall:
    function_call_id: str
    identity_provenance: str


@dataclass(frozen=True, slots=True)
class FusionRemoteAttempt:
    app: FusionRemoteApp
    attempt_id: str
    call: FusionRemoteCall
    head_commit: str
    journal_states: tuple[str, ...]
    source_sha256: str
    status: str
    suite_id: str
    suite_path: str
    terminal_detail: str | None
    wheel_filename: str
    wheel_sha256: str


@dataclass(frozen=True, slots=True)
class FusionRemoteHardware:
    compute_capability: tuple[int, int]
    cuda_version: str
    device_name: str
    gpu: str
    multiprocessor_count: int
    torch_version: str
    total_memory_gb: float
    triton_version: str


@dataclass(frozen=True, slots=True)
class FusionCompileMetrics:
    arm_id: str
    backend_invoked: bool
    callable_distinct: bool
    eager_fallback: bool
    first_call_ns: int
    status: str
    wrapper_create_ns: int


@dataclass(frozen=True, slots=True)
class FusionCorrectnessMetrics:
    close: bool
    finite: bool
    input_storage_unchanged: bool
    max_abs_error: float
    output_disjoint: bool
    status: str


@dataclass(frozen=True, slots=True)
class FusionTimingMetrics:
    median_ms: float
    repetitions: int
    status: str
    warmups: int


@dataclass(frozen=True, slots=True)
class FusionDescriptiveRatios:
    candidate_to_reference_median: float
    interpretation: str
    reference_to_candidate_median: float
    superiority_tested: bool


@dataclass(frozen=True, slots=True)
class FusionCompletedMetrics:
    candidate_distinction: str
    candidate_reference_arithmetic: str
    compile: FusionCompileMetrics
    candidate_correctness: FusionCorrectnessMetrics
    reference_correctness: FusionCorrectnessMetrics
    ratios: FusionDescriptiveRatios
    candidate_timing: FusionTimingMetrics
    reference_timing: FusionTimingMetrics


@dataclass(frozen=True, slots=True)
class FusionCompletedResult:
    attempt_id: str
    claim_scope: str
    fusion_claim: bool
    hardware: FusionRemoteHardware
    metrics: FusionCompletedMetrics
    publication_eligible: bool
    suite_id: str


@dataclass(frozen=True, slots=True)
class FusionRemoteCounts:
    attempts: int
    completed: int
    failed: int
    unresolved: int


@dataclass(frozen=True, slots=True)
class FusionRemoteClaims:
    analysis: str
    completed_correctness_timing_compile_metrics: str
    fusion: str
    performance: str
    superiority: str


@dataclass(frozen=True, slots=True)
class FusionRemoteMethodology:
    analysis_status: str
    design: str
    fusion_claim: bool
    performance_inference: str
    publication_eligible: bool
    report_status: str
    superiority_claim: bool


@dataclass(frozen=True, slots=True)
class FusionProviderAccounting:
    actual_cost_usd: float | None
    client_authorized_spawns: int
    cost_status: str
    provider_attempts_observable: bool
    provider_physical_attempts: int | None
    total_gpu_seconds: float | None


@dataclass(frozen=True, slots=True)
class FusionRemoteSummary:
    attempts: tuple[FusionRemoteAttempt, ...]
    claim_classification: FusionRemoteClaims
    completed_results: tuple[FusionCompletedResult, ...]
    counts: FusionRemoteCounts
    limitations: tuple[str, ...]
    methodology: FusionRemoteMethodology
    provider_accounting: FusionProviderAccounting
    publication_eligible: bool
    schema: str
    study_id: Literal["fusion-remote-h100-exploratory"]


EngineeringSummary = HopperSummary | PrecisionSummary | FusionRemoteSummary


def _array(value: object, *, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


def _strings(value: object, *, context: str, nonempty: bool = True) -> tuple[str, ...]:
    raw = _array(value, context=context)
    if nonempty and not raw:
        raise SchemaError(f"{context} must not be empty")
    result = tuple(
        nonblank_string(item, context=f"{context}[{index}]") for index, item in enumerate(raw)
    )
    if len(set(result)) != len(result):
        raise SchemaError(f"{context} must not contain duplicates")
    return result


def _ints(value: object, *, context: str) -> tuple[int, ...]:
    return tuple(
        exact_int(item, context=f"{context}[{index}]", minimum=0)
        for index, item in enumerate(_array(value, context=context))
    )


def _numbers(value: object, *, context: str) -> tuple[float, ...]:
    return tuple(
        finite_float(item, context=f"{context}[{index}]")
        for index, item in enumerate(_array(value, context=context))
    )


def _literal(value: object, expected: str, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if result != expected:
        raise SchemaError(f"{context} must be {expected!r}")
    return result


def _integer_literal(value: object, expected: int, *, context: str) -> int:
    result = exact_int(value, context=context, minimum=0)
    if result != expected:
        raise SchemaError(f"{context} must be {expected}")
    return result


def _digest(value: object, *, context: str, length: int = 64) -> str:
    result = nonblank_string(value, context=context)
    if len(result) != length or any(character not in _HEX for character in result):
        raise SchemaError(f"{context} must be a {length}-character lowercase hexadecimal digest")
    return result


def _repository_path(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    path = PurePosixPath(result)
    if (
        "\\" in result
        or path.is_absolute()
        or path.as_posix() != result
        or not path.parts
        or path.parts[0] != "benchmarks"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SchemaError(f"{context} must be a normalized repository-relative benchmarks path")
    return result


def _https_url(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    parsed = urlsplit(result)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise SchemaError(
            f"{context} must be an absolute HTTPS URL without credentials or fragment"
        )
    return result


def _unit_number(value: object, *, context: str) -> float:
    result = finite_float(value, context=context, minimum=0.0)
    if result > 1.0:
        raise SchemaError(f"{context} must be at most 1")
    return result


def _parse_claim(value: object, *, expected_decision: str) -> ClaimClassification:
    context = "claim_classification"
    data = exact_fields(
        value,
        required=(
            "candidate_role",
            "claim_kind",
            "comparator_role",
            "decision",
            "evidence_class",
            "inferential",
            "limitations",
            "reference_role",
            "scope",
        ),
        context=context,
    )
    claim_kind = _literal(data["claim_kind"], "descriptive", context=f"{context}.claim_kind")
    evidence_class = _literal(
        data["evidence_class"], "exploratory", context=f"{context}.evidence_class"
    )
    inferential = exact_bool(data["inferential"], context=f"{context}.inferential")
    if inferential:
        raise SchemaError(
            f"{context}.inferential must be false for exploratory descriptive evidence"
        )
    return ClaimClassification(
        candidate_role=nonblank_string(data["candidate_role"], context=f"{context}.candidate_role"),
        claim_kind=claim_kind,
        comparator_role=nonblank_string(
            data["comparator_role"], context=f"{context}.comparator_role"
        ),
        decision=_literal(data["decision"], expected_decision, context=f"{context}.decision"),
        evidence_class=evidence_class,
        inferential=inferential,
        limitations=_strings(data["limitations"], context=f"{context}.limitations"),
        reference_role=nonblank_string(data["reference_role"], context=f"{context}.reference_role"),
        scope=nonblank_string(data["scope"], context=f"{context}.scope"),
    )


def _parse_attempts(value: object) -> AttemptAccounting:
    context = "collection_accounting.attempts"
    data = exact_fields(
        value,
        required=("attempted", "completed", "failed", "retried"),
        context=context,
    )
    result = AttemptAccounting(
        attempted=exact_int(data["attempted"], context=f"{context}.attempted", minimum=0),
        completed=exact_int(data["completed"], context=f"{context}.completed", minimum=0),
        failed=exact_int(data["failed"], context=f"{context}.failed", minimum=0),
        retried=exact_int(data["retried"], context=f"{context}.retried", minimum=0),
    )
    if result.completed + result.failed != result.attempted:
        raise SchemaError(f"{context} completed plus failed must equal attempted")
    if result.retried > result.attempted:
        raise SchemaError(f"{context}.retried must not exceed attempted")
    return result


def _parse_rate_source(value: object) -> RateSource:
    context = "collection_accounting.cost.published_rate_estimate.rate_source"
    data = exact_fields(
        value,
        required=("checked_at_utc", "path", "sha256", "url"),
        context=context,
    )
    return RateSource(
        checked_at_utc=nonblank_string(data["checked_at_utc"], context=f"{context}.checked_at_utc"),
        path=_repository_path(data["path"], context=f"{context}.path"),
        sha256=_digest(data["sha256"], context=f"{context}.sha256"),
        url=_https_url(data["url"], context=f"{context}.url"),
    )


def _parse_excluded_call(value: object, *, index: int) -> ExcludedCall:
    context = f"collection_accounting.cost.excluded_calls[{index}]"
    data = exact_fields(
        value,
        required=(
            "actual_h100_cost_unknown_reason",
            "actual_h100_cost_usd",
            "call_ids",
            "role",
        ),
        context=context,
    )
    actual = optional_finite_float(
        data["actual_h100_cost_usd"],
        context=f"{context}.actual_h100_cost_usd",
        minimum=0.0,
    )
    if actual is not None:
        raise SchemaError(f"{context}.actual_h100_cost_usd must be null when cost is unknown")
    return ExcludedCall(
        actual_h100_cost_usd=actual,
        actual_h100_cost_unknown_reason=nonblank_string(
            data["actual_h100_cost_unknown_reason"],
            context=f"{context}.actual_h100_cost_unknown_reason",
        ),
        call_ids=_strings(data["call_ids"], context=f"{context}.call_ids"),
        role=nonblank_string(data["role"], context=f"{context}.role"),
    )


def _parse_cost(value: object) -> CostAccounting:
    context = "collection_accounting.cost"
    data = exact_fields(
        value,
        required=(
            "actual_h100_cost_unknown_reason",
            "actual_h100_cost_usd",
            "covered_call_ids",
            "excluded_calls",
            "published_rate_estimate",
            "scope",
        ),
        context=context,
    )
    estimate_context = f"{context}.published_rate_estimate"
    estimate = exact_fields(
        data["published_rate_estimate"],
        required=(
            "amount_usd",
            "classification",
            "gpu_rate_usd_per_second",
            "limitations",
            "rate_source",
            "time_basis",
        ),
        context=estimate_context,
    )
    classification = _literal(
        estimate["classification"], "estimated", context=f"{estimate_context}.classification"
    )
    actual = optional_finite_float(
        data["actual_h100_cost_usd"],
        context=f"{context}.actual_h100_cost_usd",
        minimum=0.0,
    )
    if actual is not None:
        raise SchemaError(f"{context}.actual_h100_cost_usd must be null when cost is unknown")
    reason = nonblank_string(
        data["actual_h100_cost_unknown_reason"],
        context=f"{context}.actual_h100_cost_unknown_reason",
    )
    covered_call_ids = _strings(data["covered_call_ids"], context=f"{context}.covered_call_ids")
    excluded_calls = tuple(
        _parse_excluded_call(item, index=index)
        for index, item in enumerate(
            _array(data["excluded_calls"], context=f"{context}.excluded_calls")
        )
    )
    excluded_call_ids = tuple(
        call_id for excluded in excluded_calls for call_id in excluded.call_ids
    )
    if len(set(excluded_call_ids)) != len(excluded_call_ids):
        raise SchemaError(f"{context}.excluded_calls call_ids must not contain duplicates")
    if set(covered_call_ids) & set(excluded_call_ids):
        raise SchemaError(f"{context} covered_call_ids and excluded call_ids must be disjoint")
    return CostAccounting(
        actual_h100_cost_usd=actual,
        actual_h100_cost_unknown_reason=reason,
        covered_call_ids=covered_call_ids,
        excluded_calls=excluded_calls,
        published_rate_estimate=PublishedRateEstimate(
            amount_usd=finite_float(
                estimate["amount_usd"], context=f"{estimate_context}.amount_usd", minimum=0.0
            ),
            classification=classification,
            gpu_rate_usd_per_second=finite_float(
                estimate["gpu_rate_usd_per_second"],
                context=f"{estimate_context}.gpu_rate_usd_per_second",
                strictly_positive=True,
            ),
            limitations=nonblank_string(
                estimate["limitations"], context=f"{estimate_context}.limitations"
            ),
            rate_source=_parse_rate_source(estimate["rate_source"]),
            time_basis=nonblank_string(
                estimate["time_basis"], context=f"{estimate_context}.time_basis"
            ),
        ),
        scope=nonblank_string(data["scope"], context=f"{context}.scope"),
    )


def _parse_accounting(value: object) -> CollectionAccounting:
    context = "collection_accounting"
    data = exact_fields(
        value,
        required=("attempts", "cost", "elapsed_seconds", "rows"),
        context=context,
    )
    row_context = f"{context}.rows"
    row_data = exact_fields(
        data["rows"],
        required=("expected", "failed", "omitted", "published"),
        context=row_context,
    )
    rows = RowAccounting(
        expected=exact_int(row_data["expected"], context=f"{row_context}.expected", minimum=0),
        failed=exact_int(row_data["failed"], context=f"{row_context}.failed", minimum=0),
        omitted=exact_int(row_data["omitted"], context=f"{row_context}.omitted", minimum=0),
        published=exact_int(row_data["published"], context=f"{row_context}.published", minimum=0),
    )
    if rows.failed + rows.omitted + rows.published != rows.expected:
        raise SchemaError(f"{row_context} failed plus omitted plus published must equal expected")
    return CollectionAccounting(
        attempts=_parse_attempts(data["attempts"]),
        cost=_parse_cost(data["cost"]),
        elapsed_seconds=finite_float(
            data["elapsed_seconds"], context=f"{context}.elapsed_seconds", minimum=0.0
        ),
        rows=rows,
    )


def _parse_publication(
    value: object,
    *,
    expected_raw_schema: str | None = None,
) -> Publication:
    context = "publication"
    data = exact_fields(
        value,
        required=(
            "raw",
            "journal",
            "manifest_path",
            "head_commit",
            "source_sha256",
            "wheel_sha256",
            "hardware",
            "runtime",
            "modal",
        ),
        context=context,
    )
    raw_context = f"{context}.raw"
    raw_fields = ["path", "sha256", "uncompressed_sha256", "bytes", "rows"]
    if expected_raw_schema is not None:
        raw_fields.extend(("schema", "schema_version"))
    raw_data = exact_fields(data["raw"], required=raw_fields, context=raw_context)
    journal_context = f"{context}.journal"
    journal_data = exact_fields(
        data["journal"],
        required=("path", "sha256", "records"),
        context=journal_context,
    )
    modal_context = f"{context}.modal"
    modal_data = exact_fields(
        data["modal"],
        required=("app_url", "app_id", "app_url_provenance", "call_ids"),
        context=modal_context,
    )
    runtime_data = exact_object(data["runtime"], context=f"{context}.runtime")
    runtime = tuple(
        sorted(
            (
                nonblank_string(key, context=f"{context}.runtime key"),
                nonblank_string(value, context=f"{context}.runtime[{key!r}]"),
            )
            for key, value in runtime_data.items()
        )
    )
    if not runtime:
        raise SchemaError(f"{context}.runtime must not be empty")
    publication = Publication(
        raw=PublishedRaw(
            path=_repository_path(raw_data["path"], context=f"{raw_context}.path"),
            sha256=_digest(raw_data["sha256"], context=f"{raw_context}.sha256"),
            uncompressed_sha256=_digest(
                raw_data["uncompressed_sha256"],
                context=f"{raw_context}.uncompressed_sha256",
            ),
            bytes=exact_int(raw_data["bytes"], context=f"{raw_context}.bytes", minimum=1),
            rows=exact_int(raw_data["rows"], context=f"{raw_context}.rows", minimum=1),
            schema=(
                _literal(
                    raw_data["schema"],
                    expected_raw_schema,
                    context=f"{raw_context}.schema",
                )
                if expected_raw_schema is not None
                else None
            ),
            schema_version=(
                _integer_literal(
                    raw_data["schema_version"],
                    2,
                    context=f"{raw_context}.schema_version",
                )
                if expected_raw_schema is not None
                else None
            ),
        ),
        journal=PublishedJournal(
            path=_repository_path(journal_data["path"], context=f"{journal_context}.path"),
            sha256=_digest(journal_data["sha256"], context=f"{journal_context}.sha256"),
            records=exact_int(
                journal_data["records"], context=f"{journal_context}.records", minimum=1
            ),
        ),
        manifest_path=_repository_path(data["manifest_path"], context=f"{context}.manifest_path"),
        head_commit=_digest(data["head_commit"], context=f"{context}.head_commit", length=40),
        source_sha256=_digest(data["source_sha256"], context=f"{context}.source_sha256"),
        wheel_sha256=_digest(data["wheel_sha256"], context=f"{context}.wheel_sha256"),
        hardware=HardwareProfile.from_dict(data["hardware"]),
        runtime=runtime,
        modal=ModalPublication(
            app_url=_https_url(modal_data["app_url"], context=f"{modal_context}.app_url"),
            app_id=nonblank_string(modal_data["app_id"], context=f"{modal_context}.app_id"),
            app_url_provenance=_literal(
                modal_data["app_url_provenance"],
                "operator_recorded",
                context=f"{modal_context}.app_url_provenance",
            ),
            call_ids=_strings(modal_data["call_ids"], context=f"{modal_context}.call_ids"),
        ),
    )
    return publication


def _parse_evidence_file(value: object, *, context: str) -> EvidenceFile:
    data = exact_fields(value, required=("path", "sha256"), context=context)
    return EvidenceFile(
        path=_repository_path(data["path"], context=f"{context}.path"),
        sha256=_digest(data["sha256"], context=f"{context}.sha256"),
    )


def _parse_committed_evidence(value: object) -> CommittedEvidence:
    context = "precision_finding.committed_evidence"
    data = exact_fields(
        value,
        required=("attempt_journal", "manifest", "raw", "summary"),
        context=context,
    )
    raw_context = f"{context}.raw"
    raw_data = exact_fields(
        data["raw"],
        required=(
            "compressed_sha256",
            "decompressed_sha256",
            "path",
            "schema",
            "schema_version",
        ),
        context=raw_context,
    )
    return CommittedEvidence(
        attempt_journal=_parse_evidence_file(
            data["attempt_journal"], context=f"{context}.attempt_journal"
        ),
        manifest=_parse_evidence_file(data["manifest"], context=f"{context}.manifest"),
        raw=RawEvidenceFile(
            path=_repository_path(raw_data["path"], context=f"{raw_context}.path"),
            compressed_sha256=_digest(
                raw_data["compressed_sha256"], context=f"{raw_context}.compressed_sha256"
            ),
            decompressed_sha256=_digest(
                raw_data["decompressed_sha256"],
                context=f"{raw_context}.decompressed_sha256",
            ),
            schema=_literal(
                raw_data["schema"],
                "h100-precision-probe-raw-v2",
                context=f"{raw_context}.schema",
            ),
            schema_version=_integer_literal(
                raw_data["schema_version"],
                2,
                context=f"{raw_context}.schema_version",
            ),
        ),
        summary=_parse_evidence_file(data["summary"], context=f"{context}.summary"),
    )


def _parse_precision_finding(value: object, *, embedded_evidence: bool) -> PrecisionFinding:
    context = "precision_finding"
    required = {
        "accuracy_regression",
        "baseline_agrees",
        "classification",
        "conclusion",
        "metrics",
        "parity_authorized",
        "thresholds",
    }
    if embedded_evidence:
        required.add("committed_evidence")
    data = exact_fields(value, required=required, context=context)
    metric_context = f"{context}.metrics"
    metrics = exact_fields(
        data["metrics"],
        required=(
            "archive_baseline_torch_over_best_triton",
            "paired_strict_slowdown",
            "torch_reduced_over_torch_strict_median",
            "torch_reduced_over_triton_median",
            "torch_strict_over_triton_median",
        ),
        context=metric_context,
    )
    threshold_context = f"{context}.thresholds"
    thresholds = exact_fields(
        data["thresholds"],
        required=(
            "baseline_absolute_tolerance",
            "baseline_relative_tolerance",
            "meaningful_paired_effect",
        ),
        context=threshold_context,
    )
    accuracy_regression = exact_bool(
        data["accuracy_regression"], context=f"{context}.accuracy_regression"
    )
    baseline_agrees = exact_bool(data["baseline_agrees"], context=f"{context}.baseline_agrees")
    parity_authorized = exact_bool(
        data["parity_authorized"], context=f"{context}.parity_authorized"
    )
    if accuracy_regression:
        raise SchemaError(f"{context}.accuracy_regression must be false")
    if not baseline_agrees:
        raise SchemaError(f"{context}.baseline_agrees must be true")
    if parity_authorized:
        raise SchemaError(f"{context}.parity_authorized must be false")
    return PrecisionFinding(
        accuracy_regression=accuracy_regression,
        baseline_agrees=baseline_agrees,
        classification=_literal(
            data["classification"], "does not explain", context=f"{context}.classification"
        ),
        conclusion=nonblank_string(data["conclusion"], context=f"{context}.conclusion"),
        metrics=PrecisionMetrics(
            archive_baseline_torch_over_best_triton=finite_float(
                metrics["archive_baseline_torch_over_best_triton"],
                context=f"{metric_context}.archive_baseline_torch_over_best_triton",
                strictly_positive=True,
            ),
            paired_strict_slowdown=finite_float(
                metrics["paired_strict_slowdown"],
                context=f"{metric_context}.paired_strict_slowdown",
            ),
            torch_reduced_over_torch_strict_median=finite_float(
                metrics["torch_reduced_over_torch_strict_median"],
                context=f"{metric_context}.torch_reduced_over_torch_strict_median",
                strictly_positive=True,
            ),
            torch_reduced_over_triton_median=finite_float(
                metrics["torch_reduced_over_triton_median"],
                context=f"{metric_context}.torch_reduced_over_triton_median",
                strictly_positive=True,
            ),
            torch_strict_over_triton_median=finite_float(
                metrics["torch_strict_over_triton_median"],
                context=f"{metric_context}.torch_strict_over_triton_median",
                strictly_positive=True,
            ),
        ),
        parity_authorized=parity_authorized,
        thresholds=PrecisionThresholds(
            baseline_absolute_tolerance=finite_float(
                thresholds["baseline_absolute_tolerance"],
                context=f"{threshold_context}.baseline_absolute_tolerance",
                minimum=0.0,
            ),
            baseline_relative_tolerance=finite_float(
                thresholds["baseline_relative_tolerance"],
                context=f"{threshold_context}.baseline_relative_tolerance",
                minimum=0.0,
            ),
            meaningful_paired_effect=finite_float(
                thresholds["meaningful_paired_effect"],
                context=f"{threshold_context}.meaningful_paired_effect",
                minimum=0.0,
            ),
        ),
        committed_evidence=(
            _parse_committed_evidence(data["committed_evidence"]) if embedded_evidence else None
        ),
    )


def _parse_candidate(value: object, *, index: int) -> CandidateSelection:
    context = f"candidate_selection[{index}]"
    data = exact_fields(
        value,
        required=(
            "archive_torch_over_bank1_selected_bank2_scored_triton",
            "best_candidate_ms",
            "best_config",
            "best_config_key",
            "correct",
            "regime",
            "torch_ms",
            "torch_over_best_candidate",
            "workload_key",
        ),
        context=context,
    )
    raw_regime = nonblank_string(data["regime"], context=f"{context}.regime")
    if raw_regime not in {"hopper_gemm", "skinny_gemv"}:
        raise SchemaError(f"{context}.regime is not a supported engineering regime")
    regime = cast(RegimeName, raw_regime)
    config_context = f"{context}.best_config"
    if regime == "skinny_gemv":
        config = exact_fields(
            data["best_config"],
            required=("block_k", "block_m", "block_n", "num_stages", "num_warps", "split_k"),
            context=config_context,
        )
    else:
        config = exact_fields(
            data["best_config"],
            required=(
                "block_k",
                "block_m",
                "block_n",
                "epilogue_subtile",
                "group_m",
                "num_stages",
                "num_warps",
                "warp_specialize",
            ),
            context=config_context,
        )
    result = CandidateSelection(
        archive_ratio=finite_float(
            data["archive_torch_over_bank1_selected_bank2_scored_triton"],
            context=f"{context}.archive_torch_over_bank1_selected_bank2_scored_triton",
            strictly_positive=True,
        ),
        best_candidate_ms=finite_float(
            data["best_candidate_ms"],
            context=f"{context}.best_candidate_ms",
            strictly_positive=True,
        ),
        best_config=CandidateConfig(
            block_k=exact_int(config["block_k"], context=f"{config_context}.block_k", minimum=1),
            block_m=exact_int(config["block_m"], context=f"{config_context}.block_m", minimum=1),
            block_n=exact_int(config["block_n"], context=f"{config_context}.block_n", minimum=1),
            num_stages=exact_int(
                config["num_stages"], context=f"{config_context}.num_stages", minimum=1
            ),
            num_warps=exact_int(
                config["num_warps"], context=f"{config_context}.num_warps", minimum=1
            ),
            split_k=(
                exact_int(config["split_k"], context=f"{config_context}.split_k", minimum=1)
                if regime == "skinny_gemv"
                else None
            ),
            epilogue_subtile=(
                exact_bool(config["epilogue_subtile"], context=f"{config_context}.epilogue_subtile")
                if regime == "hopper_gemm"
                else None
            ),
            group_m=(
                exact_int(config["group_m"], context=f"{config_context}.group_m", minimum=1)
                if regime == "hopper_gemm"
                else None
            ),
            warp_specialize=(
                exact_bool(config["warp_specialize"], context=f"{config_context}.warp_specialize")
                if regime == "hopper_gemm"
                else None
            ),
        ),
        best_config_key=nonblank_string(
            data["best_config_key"], context=f"{context}.best_config_key"
        ),
        correct=exact_bool(data["correct"], context=f"{context}.correct"),
        regime=regime,
        torch_ms=finite_float(
            data["torch_ms"], context=f"{context}.torch_ms", strictly_positive=True
        ),
        torch_over_best_candidate=finite_float(
            data["torch_over_best_candidate"],
            context=f"{context}.torch_over_best_candidate",
            strictly_positive=True,
        ),
        workload_key=nonblank_string(data["workload_key"], context=f"{context}.workload_key"),
    )
    expected_ratio = result.torch_ms / result.best_candidate_ms
    if not math.isclose(
        result.torch_over_best_candidate,
        expected_ratio,
        rel_tol=_NUMERIC_REL_TOLERANCE,
        abs_tol=_NUMERIC_ABS_TOLERANCE,
    ):
        raise SchemaError(
            f"{context}.torch_over_best_candidate must equal torch_ms / best_candidate_ms"
        )
    return result


def _parse_regime(name: RegimeName, value: object) -> RegimeResult:
    context = f"regimes.{name}"
    data = exact_fields(
        value,
        required=(
            "all_selected_correct",
            "decision",
            "geometric_mean_speedup",
            "maximum_speedup",
            "median_speedup",
            "minimum_speedup",
            "percent_at_least_five_percent_faster",
            "workload_count",
            "workloads_at_least_five_percent_faster",
        ),
        context=context,
    )
    result = RegimeResult(
        name=name,
        all_selected_correct=exact_bool(
            data["all_selected_correct"], context=f"{context}.all_selected_correct"
        ),
        decision=_literal(data["decision"], "STOP", context=f"{context}.decision"),
        geometric_mean_speedup=finite_float(
            data["geometric_mean_speedup"],
            context=f"{context}.geometric_mean_speedup",
            strictly_positive=True,
        ),
        maximum_speedup=finite_float(
            data["maximum_speedup"],
            context=f"{context}.maximum_speedup",
            strictly_positive=True,
        ),
        median_speedup=finite_float(
            data["median_speedup"],
            context=f"{context}.median_speedup",
            strictly_positive=True,
        ),
        minimum_speedup=finite_float(
            data["minimum_speedup"],
            context=f"{context}.minimum_speedup",
            strictly_positive=True,
        ),
        percent_at_least_five_percent_faster=_unit_number(
            data["percent_at_least_five_percent_faster"],
            context=f"{context}.percent_at_least_five_percent_faster",
        ),
        workload_count=exact_int(
            data["workload_count"], context=f"{context}.workload_count", minimum=1
        ),
        workloads_at_least_five_percent_faster=exact_int(
            data["workloads_at_least_five_percent_faster"],
            context=f"{context}.workloads_at_least_five_percent_faster",
            minimum=0,
        ),
    )
    if result.workloads_at_least_five_percent_faster > result.workload_count:
        raise SchemaError(
            f"{context}.workloads_at_least_five_percent_faster exceeds workload_count"
        )
    expected_fraction = result.workloads_at_least_five_percent_faster / result.workload_count
    if not math.isclose(
        result.percent_at_least_five_percent_faster,
        expected_fraction,
        rel_tol=_NUMERIC_REL_TOLERANCE,
        abs_tol=_NUMERIC_ABS_TOLERANCE,
    ):
        raise SchemaError(
            f"{context}.percent_at_least_five_percent_faster must equal "
            "workloads_at_least_five_percent_faster / workload_count"
        )
    return result


def _parse_hopper(data: dict[str, object]) -> HopperSummary:
    context = "Hopper engineering summary"
    fields = exact_fields(
        data,
        required=(
            "analysis_status",
            "candidate_selection",
            "claim",
            "claim_classification",
            "collection_accounting",
            "contextual_baseline",
            "cost_screen",
            "evidence_scope",
            "global_decision",
            "limitations",
            "precision_finding",
            "protocol",
            "publication",
            "regimes",
            "schema_version",
            "study_id",
            "three_bank_collection_performed",
        ),
        context=context,
    )
    claim_classification = _parse_claim(fields["claim_classification"], expected_decision="stopped")
    limitations = _strings(fields["limitations"], context="limitations")
    if limitations != claim_classification.limitations:
        raise SchemaError("limitations must exactly match claim_classification.limitations")
    candidates = tuple(
        _parse_candidate(item, index=index)
        for index, item in enumerate(
            _array(fields["candidate_selection"], context="candidate_selection")
        )
    )
    protocol_context = "protocol"
    protocol_data = exact_fields(
        fields["protocol"],
        required=("bank", "gpu", "ratio", "row_count", "workload_count"),
        context=protocol_context,
    )
    protocol = HopperProtocol(
        bank=_integer_literal(protocol_data["bank"], 0, context=f"{protocol_context}.bank"),
        gpu=_literal(protocol_data["gpu"], "H100", context=f"{protocol_context}.gpu"),
        ratio=_literal(
            protocol_data["ratio"],
            "torch median milliseconds / selected candidate median milliseconds",
            context=f"{protocol_context}.ratio",
        ),
        row_count=_integer_literal(
            protocol_data["row_count"], 3008, context=f"{protocol_context}.row_count"
        ),
        workload_count=_integer_literal(
            protocol_data["workload_count"], 96, context=f"{protocol_context}.workload_count"
        ),
    )
    if len(candidates) != protocol.workload_count:
        raise SchemaError("candidate_selection length must equal protocol.workload_count")
    keys = tuple(item.workload_key for item in candidates)
    if len(set(keys)) != len(keys):
        raise SchemaError("candidate_selection workload_key values must be unique")
    regimes_data = exact_fields(
        fields["regimes"], required=("hopper_gemm", "skinny_gemv"), context="regimes"
    )
    regimes = (
        _parse_regime("hopper_gemm", regimes_data["hopper_gemm"]),
        _parse_regime("skinny_gemv", regimes_data["skinny_gemv"]),
    )
    for regime in regimes:
        observed = sum(item.regime == regime.name for item in candidates)
        if observed != regime.workload_count:
            raise SchemaError(
                f"regimes.{regime.name}.workload_count does not match candidate_selection"
            )
    baseline_context = "contextual_baseline"
    baseline_data = exact_fields(
        fields["contextual_baseline"],
        required=("display_value", "frozen_value", "role"),
        context=baseline_context,
    )
    screen_context = "cost_screen"
    screen_data = exact_fields(
        fields["cost_screen"],
        required=(
            "evaluated_independently_by_regime",
            "geometric_mean_speedup_threshold",
            "required_fraction_at_least_five_percent_faster",
            "speedup_threshold_for_win",
        ),
        context=screen_context,
    )
    accounting = _parse_accounting(fields["collection_accounting"])
    _literal(
        accounting.cost.scope,
        "engineering_timing_call_only",
        context="collection_accounting.cost.scope",
    )
    publication = _parse_publication(fields["publication"])
    if publication.hardware.gpu != protocol.gpu:
        raise SchemaError("publication.hardware.gpu must match protocol.gpu")
    if publication.raw.rows != accounting.rows.published:
        raise SchemaError("publication.raw.rows must match collection_accounting.rows.published")
    journal_records = (
        accounting.attempts.attempted + accounting.attempts.completed + accounting.attempts.failed
    )
    if publication.journal.records != journal_records:
        raise SchemaError("publication.journal.records must match attempt journal record count")
    if accounting.rows.expected != protocol.row_count:
        raise SchemaError("collection_accounting.rows.expected must match protocol.row_count")
    three_bank_collection = exact_bool(
        fields["three_bank_collection_performed"], context="three_bank_collection_performed"
    )
    if three_bank_collection:
        raise SchemaError("three_bank_collection_performed must be false after the frozen STOP")
    return HopperSummary(
        analysis_status=_literal(
            fields["analysis_status"], "post_hoc_exploratory", context="analysis_status"
        ),
        candidate_selection=candidates,
        claim=_literal(fields["claim"], "No superiority claim is made.", context="claim"),
        claim_classification=claim_classification,
        collection_accounting=accounting,
        contextual_baseline=ContextualBaseline(
            display_value=nonblank_string(
                baseline_data["display_value"], context=f"{baseline_context}.display_value"
            ),
            frozen_value=finite_float(
                baseline_data["frozen_value"],
                context=f"{baseline_context}.frozen_value",
                strictly_positive=True,
            ),
            role=nonblank_string(baseline_data["role"], context=f"{baseline_context}.role"),
        ),
        cost_screen=CostScreen(
            evaluated_independently_by_regime=exact_bool(
                screen_data["evaluated_independently_by_regime"],
                context=f"{screen_context}.evaluated_independently_by_regime",
            ),
            geometric_mean_speedup_threshold=finite_float(
                screen_data["geometric_mean_speedup_threshold"],
                context=f"{screen_context}.geometric_mean_speedup_threshold",
                strictly_positive=True,
            ),
            required_fraction_at_least_five_percent_faster=_unit_number(
                screen_data["required_fraction_at_least_five_percent_faster"],
                context=f"{screen_context}.required_fraction_at_least_five_percent_faster",
            ),
            speedup_threshold_for_win=finite_float(
                screen_data["speedup_threshold_for_win"],
                context=f"{screen_context}.speedup_threshold_for_win",
                strictly_positive=True,
            ),
        ),
        evidence_scope=nonblank_string(fields["evidence_scope"], context="evidence_scope"),
        global_decision=_literal(fields["global_decision"], "STOP", context="global_decision"),
        limitations=limitations,
        precision_finding=_parse_precision_finding(
            fields["precision_finding"], embedded_evidence=True
        ),
        protocol=protocol,
        publication=publication,
        regimes=regimes,
        schema_version=_integer_literal(fields["schema_version"], 1, context="schema_version"),
        study_id=HOPPER_STUDY_ID,
        three_bank_collection_performed=three_bank_collection,
    )


def _parse_precision_protocol(value: object) -> PrecisionProtocol:
    context = "protocol"
    data = exact_fields(
        value,
        required=(
            "arm_order",
            "arm_order_seed",
            "arms",
            "banks",
            "gpu",
            "modal_selector",
            "probe",
            "quantiles",
            "reference",
            "rep_ms",
            "retry_policy",
            "role",
            "schema_version",
            "statistic",
            "tensor_seed",
            "tensor_seed_protocol",
            "warmup_ms",
            "workload_count",
        ),
        context=context,
    )
    result = PrecisionProtocol(
        arm_order=nonblank_string(data["arm_order"], context=f"{context}.arm_order"),
        arm_order_seed=nonblank_string(data["arm_order_seed"], context=f"{context}.arm_order_seed"),
        arms=_strings(data["arms"], context=f"{context}.arms"),
        banks=_ints(data["banks"], context=f"{context}.banks"),
        gpu=_literal(data["gpu"], "H100", context=f"{context}.gpu"),
        modal_selector=nonblank_string(data["modal_selector"], context=f"{context}.modal_selector"),
        probe=_literal(data["probe"], PRECISION_STUDY_ID, context=f"{context}.probe"),
        quantiles=_numbers(data["quantiles"], context=f"{context}.quantiles"),
        reference=nonblank_string(data["reference"], context=f"{context}.reference"),
        rep_ms=exact_int(data["rep_ms"], context=f"{context}.rep_ms", minimum=1),
        retry_policy=nonblank_string(data["retry_policy"], context=f"{context}.retry_policy"),
        role=nonblank_string(data["role"], context=f"{context}.role"),
        schema_version=_integer_literal(
            data["schema_version"], 2, context=f"{context}.schema_version"
        ),
        statistic=nonblank_string(data["statistic"], context=f"{context}.statistic"),
        tensor_seed=nonblank_string(data["tensor_seed"], context=f"{context}.tensor_seed"),
        tensor_seed_protocol=nonblank_string(
            data["tensor_seed_protocol"], context=f"{context}.tensor_seed_protocol"
        ),
        warmup_ms=exact_int(data["warmup_ms"], context=f"{context}.warmup_ms", minimum=1),
        workload_count=_integer_literal(
            data["workload_count"], 96, context=f"{context}.workload_count"
        ),
    )
    if result.arms != ("torch_reduced", "torch_strict", "triton"):
        raise SchemaError("protocol.arms must list torch_reduced, torch_strict, and triton")
    if result.banks != (0, 1, 2):
        raise SchemaError("protocol.banks must be exactly [0, 1, 2]")
    if result.quantiles != (0.2, 0.5, 0.8):
        raise SchemaError("protocol.quantiles must be exactly [0.2, 0.5, 0.8]")
    return result


def _parse_precision_workload(value: object, *, index: int) -> PrecisionWorkload:
    context = f"workloads[{index}]"
    data = exact_fields(
        value,
        required=(
            "archive_torch_over_best_triton",
            "banks",
            "errors",
            "k",
            "latencies_ms",
            "m",
            "n",
            "ratios",
            "workload_key",
        ),
        context=context,
    )
    errors_context = f"{context}.errors"
    errors = exact_fields(
        data["errors"],
        required=("torch_reduced", "torch_strict", "triton"),
        context=errors_context,
    )
    latency_context = f"{context}.latencies_ms"
    latencies = exact_fields(
        data["latencies_ms"],
        required=("torch_reduced", "torch_strict", "triton"),
        context=latency_context,
    )
    ratio_context = f"{context}.ratios"
    ratios = exact_fields(
        data["ratios"],
        required=(
            "torch_reduced_over_torch_strict",
            "torch_reduced_over_triton",
            "torch_strict_over_triton",
        ),
        context=ratio_context,
    )
    banks = _ints(data["banks"], context=f"{context}.banks")
    if banks != (0, 1, 2):
        raise SchemaError(f"{context}.banks must be exactly [0, 1, 2]")
    return PrecisionWorkload(
        archive_ratio=finite_float(
            data["archive_torch_over_best_triton"],
            context=f"{context}.archive_torch_over_best_triton",
            strictly_positive=True,
        ),
        banks=banks,
        torch_reduced_error=finite_float(
            errors["torch_reduced"], context=f"{errors_context}.torch_reduced", minimum=0.0
        ),
        torch_strict_error=finite_float(
            errors["torch_strict"], context=f"{errors_context}.torch_strict", minimum=0.0
        ),
        triton_error=finite_float(
            errors["triton"], context=f"{errors_context}.triton", minimum=0.0
        ),
        k=exact_int(data["k"], context=f"{context}.k", minimum=1),
        torch_reduced_ms=finite_float(
            latencies["torch_reduced"],
            context=f"{latency_context}.torch_reduced",
            strictly_positive=True,
        ),
        torch_strict_ms=finite_float(
            latencies["torch_strict"],
            context=f"{latency_context}.torch_strict",
            strictly_positive=True,
        ),
        triton_ms=finite_float(
            latencies["triton"], context=f"{latency_context}.triton", strictly_positive=True
        ),
        m=exact_int(data["m"], context=f"{context}.m", minimum=1),
        n=exact_int(data["n"], context=f"{context}.n", minimum=1),
        torch_reduced_over_torch_strict=finite_float(
            ratios["torch_reduced_over_torch_strict"],
            context=f"{ratio_context}.torch_reduced_over_torch_strict",
            strictly_positive=True,
        ),
        torch_reduced_over_triton=finite_float(
            ratios["torch_reduced_over_triton"],
            context=f"{ratio_context}.torch_reduced_over_triton",
            strictly_positive=True,
        ),
        torch_strict_over_triton=finite_float(
            ratios["torch_strict_over_triton"],
            context=f"{ratio_context}.torch_strict_over_triton",
            strictly_positive=True,
        ),
        workload_key=nonblank_string(data["workload_key"], context=f"{context}.workload_key"),
    )


def _parse_precision(data: dict[str, object]) -> PrecisionSummary:
    context = "precision probe summary"
    fields = exact_fields(
        data,
        required=(
            "analysis_status",
            "claim_classification",
            "collection_accounting",
            "evidence_scope",
            "limitations",
            "precision_finding",
            "protocol",
            "publication",
            "row_count",
            "schema_version",
            "study_id",
            "workload_count",
            "workloads",
        ),
        context=context,
    )
    claim_classification = _parse_claim(
        fields["claim_classification"], expected_decision="supported"
    )
    limitations = _strings(fields["limitations"], context="limitations")
    if limitations != claim_classification.limitations:
        raise SchemaError("limitations must exactly match claim_classification.limitations")
    protocol = _parse_precision_protocol(fields["protocol"])
    workloads = tuple(
        _parse_precision_workload(item, index=index)
        for index, item in enumerate(_array(fields["workloads"], context="workloads"))
    )
    workload_count = _integer_literal(fields["workload_count"], 96, context="workload_count")
    row_count = _integer_literal(fields["row_count"], 288, context="row_count")
    if len(workloads) != workload_count or workload_count != protocol.workload_count:
        raise SchemaError("workloads length and workload counts must agree")
    if row_count != workload_count * len(protocol.banks):
        raise SchemaError("row_count must equal workload_count times bank count")
    keys = tuple(item.workload_key for item in workloads)
    if len(set(keys)) != len(keys):
        raise SchemaError("workloads workload_key values must be unique")
    accounting = _parse_accounting(fields["collection_accounting"])
    publication = _parse_publication(
        fields["publication"],
        expected_raw_schema="h100-precision-probe-raw-v2",
    )
    if accounting.rows.published != row_count or publication.raw.rows != row_count:
        raise SchemaError("published row counts must agree with row_count")
    if accounting.rows.expected != row_count:
        raise SchemaError("collection_accounting.rows.expected must agree with row_count")
    _literal(
        accounting.cost.scope,
        "precision_probe_calls_only",
        context="collection_accounting.cost.scope",
    )
    journal_records = (
        accounting.attempts.attempted + accounting.attempts.completed + accounting.attempts.failed
    )
    if publication.journal.records != journal_records:
        raise SchemaError("publication.journal.records must match attempt journal record count")
    if publication.hardware.gpu != protocol.gpu:
        raise SchemaError("publication.hardware.gpu must match protocol.gpu")
    return PrecisionSummary(
        analysis_status=_literal(
            fields["analysis_status"], "post_hoc_exploratory", context="analysis_status"
        ),
        claim_classification=claim_classification,
        collection_accounting=accounting,
        evidence_scope=nonblank_string(fields["evidence_scope"], context="evidence_scope"),
        limitations=limitations,
        precision_finding=_parse_precision_finding(
            fields["precision_finding"], embedded_evidence=False
        ),
        protocol=protocol,
        publication=publication,
        row_count=row_count,
        schema_version=_integer_literal(fields["schema_version"], 1, context="schema_version"),
        study_id=PRECISION_STUDY_ID,
        workload_count=workload_count,
        workloads=workloads,
    )


def _boolean_literal(value: object, expected: bool, *, context: str) -> bool:
    result = exact_bool(value, context=context)
    if result is not expected:
        raise SchemaError(f"{context} must be {str(expected).lower()}")
    return result


def _null_literal(value: object, *, context: str) -> None:
    if value is not None:
        raise SchemaError(f"{context} must be null")


def _parse_fusion_app(value: object, *, context: str) -> FusionRemoteApp:
    fields = exact_fields(
        value,
        required=("app_id", "app_url", "artifact_binding", "identity_provenance"),
        context=context,
    )
    app_id = nonblank_string(fields["app_id"], context=f"{context}.app_id")
    app_suffix = app_id.removeprefix("ap-")
    if (
        app_suffix == app_id
        or not app_suffix.isascii()
        or not app_suffix.isalnum()
        or len(app_suffix) < 20
    ):
        raise SchemaError(f"{context}.app_id must be a Modal app ID")
    app_url = _https_url(fields["app_url"], context=f"{context}.app_url")
    parsed_url = urlsplit(app_url)
    if (
        parsed_url.netloc != "modal.com"
        or parsed_url.path != f"/apps/mottopanikeiku/main/{app_id}"
        or parsed_url.query
    ):
        raise SchemaError(f"{context}.app_url must identify {app_id} on modal.com")
    return FusionRemoteApp(
        app_id=app_id,
        app_url=app_url,
        artifact_binding=_literal(
            fields["artifact_binding"], "none", context=f"{context}.artifact_binding"
        ),
        identity_provenance=_literal(
            fields["identity_provenance"],
            "operator_recorded",
            context=f"{context}.identity_provenance",
        ),
    )


def _parse_fusion_call(value: object, *, context: str) -> FusionRemoteCall:
    fields = exact_fields(
        value,
        required=("function_call_id", "identity_provenance"),
        context=context,
    )
    call_id = nonblank_string(fields["function_call_id"], context=f"{context}.function_call_id")
    call_suffix = call_id.removeprefix("fc-")
    if (
        call_suffix == call_id
        or not call_suffix.isascii()
        or not call_suffix.isalnum()
        or call_suffix != call_suffix.upper()
        or len(call_suffix) < 20
    ):
        raise SchemaError(f"{context}.function_call_id must be a Modal FunctionCall ID")
    return FusionRemoteCall(
        function_call_id=call_id,
        identity_provenance=_literal(
            fields["identity_provenance"],
            "artifact_bound_remote_journal",
            context=f"{context}.identity_provenance",
        ),
    )


def _parse_fusion_attempt(value: object, *, index: int) -> FusionRemoteAttempt:
    context = f"attempts[{index}]"
    fields = exact_fields(
        value,
        required=(
            "app",
            "attempt_id",
            "call",
            "head_commit",
            "journal_states",
            "source_sha256",
            "status",
            "suite_id",
            "suite_path",
            "terminal_detail",
            "wheel_filename",
            "wheel_sha256",
        ),
        context=context,
    )
    status = nonblank_string(fields["status"], context=f"{context}.status")
    if status not in {"completed", "unresolved"}:
        raise SchemaError(f"{context}.status must be completed or unresolved")
    attempt_id = nonblank_string(fields["attempt_id"], context=f"{context}.attempt_id")
    if not attempt_id.endswith(f"-{status}"):
        raise SchemaError(f"{context}.attempt_id must end with its status")
    suite_id = nonblank_string(fields["suite_id"], context=f"{context}.suite_id")
    suite_paths = {
        "gated-mlp-epilogue-reference": "benchmarks/suites/gated-mlp-epilogue-v1.json",
        "residual-rmsnorm-reference": "benchmarks/suites/residual-rmsnorm-v1.json",
    }
    if suite_id not in suite_paths:
        raise SchemaError(f"{context}.suite_id is not supported by the fusion remote report")
    suite_path = _repository_path(fields["suite_path"], context=f"{context}.suite_path")
    if suite_path != suite_paths[suite_id]:
        raise SchemaError(f"{context}.suite_path must match suite_id")
    journal_states = _strings(fields["journal_states"], context=f"{context}.journal_states")
    if fields["terminal_detail"] is None:
        terminal_detail = None
    else:
        terminal_detail = nonblank_string(
            fields["terminal_detail"], context=f"{context}.terminal_detail"
        )
    if status == "unresolved":
        expected_states = (
            "intent",
            "spawned",
            "retrieval_started",
            "cancellation_requested",
            "unresolved",
        )
        if journal_states != expected_states:
            raise SchemaError(f"{context}.journal_states contradict unresolved status")
        expected_detail = "RemoteError: AuthError(\"Received :status = '401'\")"
        if terminal_detail != expected_detail:
            raise SchemaError(f"{context}.terminal_detail must record the retained 401 error")
        if suite_id != "gated-mlp-epilogue-reference":
            raise SchemaError(f"{context}.suite_id contradicts the retained unresolved attempts")
    else:
        if journal_states != ("intent", "spawned", "retrieval_started", "completed"):
            raise SchemaError(f"{context}.journal_states contradict completed status")
        if terminal_detail is not None:
            raise SchemaError(f"{context}.terminal_detail must be null for completed status")
    return FusionRemoteAttempt(
        app=_parse_fusion_app(fields["app"], context=f"{context}.app"),
        attempt_id=attempt_id,
        call=_parse_fusion_call(fields["call"], context=f"{context}.call"),
        head_commit=_digest(fields["head_commit"], context=f"{context}.head_commit", length=40),
        journal_states=journal_states,
        source_sha256=_digest(fields["source_sha256"], context=f"{context}.source_sha256"),
        status=status,
        suite_id=suite_id,
        suite_path=suite_path,
        terminal_detail=terminal_detail,
        wheel_filename=_literal(
            fields["wheel_filename"],
            "heliostune-0.4.1-py3-none-any.whl",
            context=f"{context}.wheel_filename",
        ),
        wheel_sha256=_digest(fields["wheel_sha256"], context=f"{context}.wheel_sha256"),
    )


def _parse_fusion_hardware(value: object, *, context: str) -> FusionRemoteHardware:
    fields = exact_fields(
        value,
        required=(
            "compute_capability",
            "cuda_version",
            "device_name",
            "gpu",
            "multiprocessor_count",
            "torch_version",
            "total_memory_gb",
            "triton_version",
        ),
        context=context,
    )
    compute_capability = _ints(
        fields["compute_capability"], context=f"{context}.compute_capability"
    )
    if compute_capability != (9, 0):
        raise SchemaError(f"{context}.compute_capability must be [9, 0]")
    return FusionRemoteHardware(
        compute_capability=(compute_capability[0], compute_capability[1]),
        cuda_version=nonblank_string(fields["cuda_version"], context=f"{context}.cuda_version"),
        device_name=nonblank_string(fields["device_name"], context=f"{context}.device_name"),
        gpu=_literal(fields["gpu"], "H100", context=f"{context}.gpu"),
        multiprocessor_count=exact_int(
            fields["multiprocessor_count"],
            context=f"{context}.multiprocessor_count",
            minimum=1,
        ),
        torch_version=nonblank_string(fields["torch_version"], context=f"{context}.torch_version"),
        total_memory_gb=finite_float(
            fields["total_memory_gb"],
            context=f"{context}.total_memory_gb",
            strictly_positive=True,
        ),
        triton_version=nonblank_string(
            fields["triton_version"], context=f"{context}.triton_version"
        ),
    )


def _parse_fusion_compile(value: object, *, context: str) -> FusionCompileMetrics:
    fields = exact_fields(
        value,
        required=(
            "arm_id",
            "backend_invoked",
            "callable_distinct",
            "eager_fallback",
            "first_call_ns",
            "status",
            "wrapper_create_ns",
        ),
        context=context,
    )
    wrapper_create_ns = exact_int(
        fields["wrapper_create_ns"], context=f"{context}.wrapper_create_ns", minimum=1
    )
    first_call_ns = exact_int(
        fields["first_call_ns"], context=f"{context}.first_call_ns", minimum=1
    )
    if first_call_ns <= wrapper_create_ns:
        raise SchemaError(f"{context}.first_call_ns must exceed wrapper_create_ns")
    return FusionCompileMetrics(
        arm_id=nonblank_string(fields["arm_id"], context=f"{context}.arm_id"),
        backend_invoked=_boolean_literal(
            fields["backend_invoked"], True, context=f"{context}.backend_invoked"
        ),
        callable_distinct=_boolean_literal(
            fields["callable_distinct"], True, context=f"{context}.callable_distinct"
        ),
        eager_fallback=_boolean_literal(
            fields["eager_fallback"], False, context=f"{context}.eager_fallback"
        ),
        first_call_ns=first_call_ns,
        status=_literal(
            fields["status"],
            "compiled_and_first_call_completed",
            context=f"{context}.status",
        ),
        wrapper_create_ns=wrapper_create_ns,
    )


def _parse_fusion_correctness(value: object, *, context: str) -> FusionCorrectnessMetrics:
    fields = exact_fields(
        value,
        required=(
            "close",
            "finite",
            "input_storage_unchanged",
            "max_abs_error",
            "output_disjoint",
            "status",
        ),
        context=context,
    )
    return FusionCorrectnessMetrics(
        close=_boolean_literal(fields["close"], True, context=f"{context}.close"),
        finite=_boolean_literal(fields["finite"], True, context=f"{context}.finite"),
        input_storage_unchanged=_boolean_literal(
            fields["input_storage_unchanged"],
            True,
            context=f"{context}.input_storage_unchanged",
        ),
        max_abs_error=finite_float(
            fields["max_abs_error"], context=f"{context}.max_abs_error", minimum=0.0
        ),
        output_disjoint=_boolean_literal(
            fields["output_disjoint"], True, context=f"{context}.output_disjoint"
        ),
        status=_literal(fields["status"], "passed", context=f"{context}.status"),
    )


def _parse_fusion_timing(value: object, *, context: str) -> FusionTimingMetrics:
    fields = exact_fields(
        value,
        required=("median_ms", "repetitions", "status", "warmups"),
        context=context,
    )
    return FusionTimingMetrics(
        median_ms=finite_float(
            fields["median_ms"], context=f"{context}.median_ms", strictly_positive=True
        ),
        repetitions=_integer_literal(fields["repetitions"], 50, context=f"{context}.repetitions"),
        status=_literal(fields["status"], "passed", context=f"{context}.status"),
        warmups=_integer_literal(fields["warmups"], 10, context=f"{context}.warmups"),
    )


def _parse_fusion_ratios(
    value: object,
    *,
    context: str,
    candidate_timing: FusionTimingMetrics,
    reference_timing: FusionTimingMetrics,
) -> FusionDescriptiveRatios:
    fields = exact_fields(
        value,
        required=(
            "candidate_to_reference_median",
            "interpretation",
            "reference_to_candidate_median",
            "superiority_tested",
        ),
        context=context,
    )
    candidate_to_reference = finite_float(
        fields["candidate_to_reference_median"],
        context=f"{context}.candidate_to_reference_median",
        strictly_positive=True,
    )
    reference_to_candidate = finite_float(
        fields["reference_to_candidate_median"],
        context=f"{context}.reference_to_candidate_median",
        strictly_positive=True,
    )
    expected_candidate_to_reference = candidate_timing.median_ms / reference_timing.median_ms
    expected_reference_to_candidate = reference_timing.median_ms / candidate_timing.median_ms
    if not math.isclose(
        candidate_to_reference,
        expected_candidate_to_reference,
        rel_tol=_NUMERIC_REL_TOLERANCE,
        abs_tol=_NUMERIC_ABS_TOLERANCE,
    ):
        raise SchemaError(
            f"{context}.candidate_to_reference_median must equal candidate median / reference median"
        )
    if not math.isclose(
        reference_to_candidate,
        expected_reference_to_candidate,
        rel_tol=_NUMERIC_REL_TOLERANCE,
        abs_tol=_NUMERIC_ABS_TOLERANCE,
    ):
        raise SchemaError(
            f"{context}.reference_to_candidate_median must equal reference median / candidate median"
        )
    if not math.isclose(
        candidate_to_reference * reference_to_candidate,
        1.0,
        rel_tol=_NUMERIC_REL_TOLERANCE,
        abs_tol=_NUMERIC_ABS_TOLERANCE,
    ):
        raise SchemaError(f"{context} ratio directions must be reciprocal")
    return FusionDescriptiveRatios(
        candidate_to_reference_median=candidate_to_reference,
        interpretation=_literal(
            fields["interpretation"],
            "ratio_of_returned_medians_only",
            context=f"{context}.interpretation",
        ),
        reference_to_candidate_median=reference_to_candidate,
        superiority_tested=_boolean_literal(
            fields["superiority_tested"], False, context=f"{context}.superiority_tested"
        ),
    )


def _parse_fusion_completed_result(value: object, *, index: int) -> FusionCompletedResult:
    context = f"completed_results[{index}]"
    fields = exact_fields(
        value,
        required=(
            "attempt_id",
            "claim_scope",
            "fusion_claim",
            "hardware",
            "metrics",
            "publication_eligible",
            "suite_id",
        ),
        context=context,
    )
    metrics_context = f"{context}.metrics"
    metrics_fields = exact_fields(
        fields["metrics"],
        required=(
            "candidate_distinction",
            "candidate_reference_arithmetic",
            "compile",
            "correctness",
            "descriptive_ratios",
            "timing",
        ),
        context=metrics_context,
    )
    correctness_fields = exact_fields(
        metrics_fields["correctness"],
        required=("candidate", "reference"),
        context=f"{metrics_context}.correctness",
    )
    timing_fields = exact_fields(
        metrics_fields["timing"],
        required=("candidate", "reference"),
        context=f"{metrics_context}.timing",
    )
    candidate_timing = _parse_fusion_timing(
        timing_fields["candidate"], context=f"{metrics_context}.timing.candidate"
    )
    reference_timing = _parse_fusion_timing(
        timing_fields["reference"], context=f"{metrics_context}.timing.reference"
    )
    compile_metrics = _parse_fusion_compile(
        metrics_fields["compile"], context=f"{metrics_context}.compile"
    )
    suite_id = nonblank_string(fields["suite_id"], context=f"{context}.suite_id")
    expected_arm_ids = {
        "gated-mlp-epilogue-reference": "mlp-candidate",
        "residual-rmsnorm-reference": "rmsnorm-candidate",
    }
    if suite_id not in expected_arm_ids:
        raise SchemaError(f"{context}.suite_id is not supported by the fusion remote report")
    if compile_metrics.arm_id != expected_arm_ids[suite_id]:
        raise SchemaError(f"{metrics_context}.compile.arm_id must match suite_id")
    return FusionCompletedResult(
        attempt_id=nonblank_string(fields["attempt_id"], context=f"{context}.attempt_id"),
        claim_scope=_literal(
            fields["claim_scope"], "measured_fact_only", context=f"{context}.claim_scope"
        ),
        fusion_claim=_boolean_literal(
            fields["fusion_claim"], False, context=f"{context}.fusion_claim"
        ),
        hardware=_parse_fusion_hardware(fields["hardware"], context=f"{context}.hardware"),
        metrics=FusionCompletedMetrics(
            candidate_distinction=_literal(
                metrics_fields["candidate_distinction"],
                "fullgraph_inductor_compilation_only",
                context=f"{metrics_context}.candidate_distinction",
            ),
            candidate_reference_arithmetic=_literal(
                metrics_fields["candidate_reference_arithmetic"],
                "candidate_reference_identical",
                context=f"{metrics_context}.candidate_reference_arithmetic",
            ),
            compile=compile_metrics,
            candidate_correctness=_parse_fusion_correctness(
                correctness_fields["candidate"],
                context=f"{metrics_context}.correctness.candidate",
            ),
            reference_correctness=_parse_fusion_correctness(
                correctness_fields["reference"],
                context=f"{metrics_context}.correctness.reference",
            ),
            ratios=_parse_fusion_ratios(
                metrics_fields["descriptive_ratios"],
                context=f"{metrics_context}.descriptive_ratios",
                candidate_timing=candidate_timing,
                reference_timing=reference_timing,
            ),
            candidate_timing=candidate_timing,
            reference_timing=reference_timing,
        ),
        publication_eligible=_boolean_literal(
            fields["publication_eligible"],
            False,
            context=f"{context}.publication_eligible",
        ),
        suite_id=suite_id,
    )


def _parse_fusion_remote(data: dict[str, object]) -> FusionRemoteSummary:
    context = "fusion remote exploratory summary"
    fields = exact_fields(
        data,
        required=(
            "attempts",
            "claim_classification",
            "completed_results",
            "counts",
            "limitations",
            "methodology",
            "provider_accounting",
            "publication_eligible",
            "schema",
            "study_id",
        ),
        context=context,
    )
    attempts = tuple(
        _parse_fusion_attempt(item, index=index)
        for index, item in enumerate(_array(fields["attempts"], context="attempts"))
    )
    if len(attempts) != 4:
        raise SchemaError("attempts must contain exactly four retained remote attempts")
    attempt_ids = tuple(attempt.attempt_id for attempt in attempts)
    app_ids = tuple(attempt.app.app_id for attempt in attempts)
    call_ids = tuple(attempt.call.function_call_id for attempt in attempts)
    if len(set(attempt_ids)) != 4:
        raise SchemaError("attempts.attempt_id values must be unique")
    if len(set(app_ids)) != 4:
        raise SchemaError("attempts app IDs must be unique")
    if len(set(call_ids)) != 4:
        raise SchemaError("attempts FunctionCall IDs must be unique")
    if len({attempt.head_commit for attempt in attempts}) != 4:
        raise SchemaError("attempts must preserve four distinct historical HEAD commits")
    if len({attempt.source_sha256 for attempt in attempts}) != 4:
        raise SchemaError("attempts must preserve four distinct historical source digests")
    if len({attempt.wheel_sha256 for attempt in attempts}) != 4:
        raise SchemaError("attempts must preserve four distinct historical wheel digests")
    statuses = tuple(attempt.status for attempt in attempts)
    if statuses.count("completed") != 2 or statuses.count("unresolved") != 2:
        raise SchemaError("attempts must contain two completed and two unresolved statuses")
    gated_attempts = tuple(
        attempt for attempt in attempts if attempt.suite_id == "gated-mlp-epilogue-reference"
    )
    rmsnorm_attempts = tuple(
        attempt for attempt in attempts if attempt.suite_id == "residual-rmsnorm-reference"
    )
    if (
        len(gated_attempts) != 3
        or sum(attempt.status == "unresolved" for attempt in gated_attempts) != 2
        or sum(attempt.status == "completed" for attempt in gated_attempts) != 1
        or len(rmsnorm_attempts) != 1
        or rmsnorm_attempts[0].status != "completed"
    ):
        raise SchemaError("attempt suite and status accounting contradicts the retained evidence")

    completed_results = tuple(
        _parse_fusion_completed_result(item, index=index)
        for index, item in enumerate(
            _array(fields["completed_results"], context="completed_results")
        )
    )
    if len(completed_results) != 2:
        raise SchemaError("completed_results must contain exactly two returned results")
    completed_attempts = tuple(attempt for attempt in attempts if attempt.status == "completed")
    if tuple(result.attempt_id for result in completed_results) != tuple(
        attempt.attempt_id for attempt in completed_attempts
    ):
        raise SchemaError("completed_results must exactly follow the completed attempts")
    for result, attempt in zip(completed_results, completed_attempts, strict=True):
        if result.suite_id != attempt.suite_id:
            raise SchemaError("completed_results suite IDs must match their attempts")
    if len({result.suite_id for result in completed_results}) != 2:
        raise SchemaError("completed_results must cover one result for each retained suite")
    if completed_results[0].hardware != completed_results[1].hardware:
        raise SchemaError("completed_results hardware profiles must agree")

    claims_data = exact_fields(
        fields["claim_classification"],
        required=(
            "analysis",
            "completed_correctness_timing_compile_metrics",
            "fusion",
            "performance",
            "superiority",
        ),
        context="claim_classification",
    )
    claims = FusionRemoteClaims(
        analysis=_literal(
            claims_data["analysis"], "exploratory", context="claim_classification.analysis"
        ),
        completed_correctness_timing_compile_metrics=_literal(
            claims_data["completed_correctness_timing_compile_metrics"],
            "supported_only_as_measured_fact",
            context="claim_classification.completed_correctness_timing_compile_metrics",
        ),
        fusion=_literal(claims_data["fusion"], "not_tested", context="claim_classification.fusion"),
        performance=_literal(
            claims_data["performance"],
            "descriptive",
            context="claim_classification.performance",
        ),
        superiority=_literal(
            claims_data["superiority"],
            "not_tested",
            context="claim_classification.superiority",
        ),
    )

    counts_data = exact_fields(
        fields["counts"],
        required=("attempts", "completed", "failed", "unresolved"),
        context="counts",
    )
    counts = FusionRemoteCounts(
        attempts=exact_int(counts_data["attempts"], context="counts.attempts", minimum=0),
        completed=exact_int(counts_data["completed"], context="counts.completed", minimum=0),
        failed=exact_int(counts_data["failed"], context="counts.failed", minimum=0),
        unresolved=exact_int(counts_data["unresolved"], context="counts.unresolved", minimum=0),
    )
    observed_counts = FusionRemoteCounts(
        attempts=len(attempts),
        completed=statuses.count("completed"),
        failed=0,
        unresolved=statuses.count("unresolved"),
    )
    if counts != observed_counts:
        raise SchemaError("counts must exactly match attempt statuses")

    expected_limitations = (
        "Two gated-MLP calls ended unresolved after cancellation requests; no result receipts exist for them.",
        "The two completed calls are single returned observations collected without a prespecified comparative analysis plan.",
        "Reference/candidate ratios are descriptive for the returned medians only; no uncertainty or superiority test was performed.",
        "The candidate and reference arithmetic are not evidence of kernel fusion; every returned environment explicitly records fusion_claim=false.",
        "Modal provider physical starts and restarts are unobservable, so provider attempt count, total GPU time, and actual cost are unknown.",
        "The retained evidence has no attestation and is not publication eligible.",
    )
    limitations = _strings(fields["limitations"], context="limitations")
    if limitations != expected_limitations:
        raise SchemaError("limitations must exactly preserve the fusion remote disclosures")

    methodology_data = exact_fields(
        fields["methodology"],
        required=(
            "analysis_status",
            "design",
            "fusion_claim",
            "performance_inference",
            "publication_eligible",
            "report_status",
            "superiority_claim",
        ),
        context="methodology",
    )
    methodology = FusionRemoteMethodology(
        analysis_status=_literal(
            methodology_data["analysis_status"],
            "post_hoc_exploratory",
            context="methodology.analysis_status",
        ),
        design=_literal(
            methodology_data["design"],
            "four retained remote attempts analyzed after execution",
            context="methodology.design",
        ),
        fusion_claim=_boolean_literal(
            methodology_data["fusion_claim"], False, context="methodology.fusion_claim"
        ),
        performance_inference=_literal(
            methodology_data["performance_inference"],
            "not_tested",
            context="methodology.performance_inference",
        ),
        publication_eligible=_boolean_literal(
            methodology_data["publication_eligible"],
            False,
            context="methodology.publication_eligible",
        ),
        report_status=_literal(
            methodology_data["report_status"],
            "not_created",
            context="methodology.report_status",
        ),
        superiority_claim=_boolean_literal(
            methodology_data["superiority_claim"],
            False,
            context="methodology.superiority_claim",
        ),
    )

    provider_data = exact_fields(
        fields["provider_accounting"],
        required=(
            "actual_cost_usd",
            "client_authorized_spawns",
            "cost_status",
            "provider_attempts_observable",
            "provider_physical_attempts",
            "total_gpu_seconds",
        ),
        context="provider_accounting",
    )
    _null_literal(provider_data["actual_cost_usd"], context="provider_accounting.actual_cost_usd")
    _null_literal(
        provider_data["provider_physical_attempts"],
        context="provider_accounting.provider_physical_attempts",
    )
    _null_literal(
        provider_data["total_gpu_seconds"], context="provider_accounting.total_gpu_seconds"
    )
    provider = FusionProviderAccounting(
        actual_cost_usd=None,
        client_authorized_spawns=_integer_literal(
            provider_data["client_authorized_spawns"],
            4,
            context="provider_accounting.client_authorized_spawns",
        ),
        cost_status=_literal(
            provider_data["cost_status"], "unknown", context="provider_accounting.cost_status"
        ),
        provider_attempts_observable=_boolean_literal(
            provider_data["provider_attempts_observable"],
            False,
            context="provider_accounting.provider_attempts_observable",
        ),
        provider_physical_attempts=None,
        total_gpu_seconds=None,
    )

    return FusionRemoteSummary(
        attempts=attempts,
        claim_classification=claims,
        completed_results=completed_results,
        counts=counts,
        limitations=limitations,
        methodology=methodology,
        provider_accounting=provider,
        publication_eligible=_boolean_literal(
            fields["publication_eligible"], False, context="publication_eligible"
        ),
        schema=_literal(fields["schema"], FUSION_REMOTE_SCHEMA, context="schema"),
        study_id=FUSION_REMOTE_STUDY_ID,
    )


def parse_engineering_summary(value: object) -> EngineeringSummary:
    """Parse one supported strict study summary and reject every other study."""
    data = exact_object(value, context="engineering report summary")
    study_id = nonblank_string(data.get("study_id"), context="engineering report study_id")
    if study_id == HOPPER_STUDY_ID:
        return _parse_hopper(data)
    if study_id == PRECISION_STUDY_ID:
        return _parse_precision(data)
    if study_id == FUSION_REMOTE_STUDY_ID:
        return _parse_fusion_remote(data)
    raise SchemaError(f"unsupported engineering report study_id {study_id!r}")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_number(value: float, *, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bool(value: bool) -> str:
    return "Yes" if value else "No"


def _repo_href(path: str, output_path: Path) -> str:
    normalized = _repository_path(path, context="report artifact path")
    repository_root = Path.cwd().resolve()
    source = (repository_root / normalized).resolve()
    try:
        source.relative_to(repository_root)
    except ValueError as error:
        raise SchemaError("report artifact path resolves outside the repository") from error
    relative = Path(os.path.relpath(source, start=output_path.resolve().parent)).as_posix()
    return quote(relative, safe="/")


def _link(url: str, label: str) -> str:
    return f'<a href="{_escape(url)}">{_escape(label)}</a>'


def _section(section_id: str, eyebrow: str, title: str, introduction: str, body: str) -> str:
    return (
        f'<section id="{_escape(section_id)}" class="report-section">'
        '<div class="section-heading">'
        f'<p class="eyebrow">{_escape(eyebrow)}</p><h2>{_escape(title)}</h2>'
        f"<p>{_escape(introduction)}</p></div>{body}</section>"
    )


def _facts(items: Iterable[tuple[str, str]], *, class_name: str = "facts") -> str:
    return (
        f'<dl class="{_escape(class_name)}">'
        + "".join(
            f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"
            for label, value in items
        )
        + "</dl>"
    )


def _table(
    headers: tuple[str, ...],
    rows: Iterable[tuple[str, ...]],
    *,
    caption: str,
    table_class: str = "",
) -> str:
    classes = "data-table" + (f" {table_class}" if table_class else "")
    heading = "".join(f'<th scope="col">{_escape(item)}</th>' for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        '<div class="table-wrap" tabindex="0" role="region" '
        f'aria-label="{_escape(caption)}"><table class="{_escape(classes)}">'
        f"<caption>{_escape(caption)}</caption><thead><tr>{heading}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _claim_section(claim: ClaimClassification, *, no_superiority: str) -> str:
    facts = _facts(
        (
            ("Evidence class", claim.evidence_class),
            ("Claim kind", claim.claim_kind),
            ("Inferential claim", _bool(claim.inferential)),
            ("Decision", claim.decision),
            ("Candidate role", claim.candidate_role),
            ("Comparator role", claim.comparator_role),
            ("Reference role", claim.reference_role),
            ("Scope", claim.scope),
        ),
        class_name="facts facts-wide",
    )
    return _section(
        "claims",
        "Claim discipline",
        "Methodology-compatible typed claims",
        "This legacy-shaped publication uses an explicit exploratory claim taxonomy. It is not a Methodology v1 EvidenceBundle and must not be presented as one.",
        f'<div class="notice"><strong>{_escape(no_superiority)}</strong></div>{facts}',
    )


def _accounting_section(accounting: CollectionAccounting, output_path: Path) -> str:
    attempts = accounting.attempts
    rows = accounting.rows
    cost = accounting.cost
    estimate = cost.published_rate_estimate
    actual = (
        f"${_format_number(cost.actual_h100_cost_usd)}"
        if cost.actual_h100_cost_usd is not None
        else "Unknown"
    )
    covered_calls = ", ".join(cost.covered_call_ids)
    excluded_calls = (
        "; ".join(
            f"{excluded.role}: {', '.join(excluded.call_ids)}" for excluded in cost.excluded_calls
        )
        or "None"
    )
    body = (
        '<div class="split">'
        + _facts(
            (
                ("Attempts", str(attempts.attempted)),
                ("Completed", str(attempts.completed)),
                ("Failed", str(attempts.failed)),
                ("Retried", str(attempts.retried)),
                ("Elapsed journal span", f"{_format_number(accounting.elapsed_seconds)} s"),
                ("Rows published / expected", f"{rows.published} / {rows.expected}"),
                ("Rows failed / omitted", f"{rows.failed} / {rows.omitted}"),
            )
        )
        + _facts(
            (
                ("Actual H100 cost", actual),
                ("Cost scope", cost.scope),
                ("Covered call IDs", covered_calls),
                ("Excluded calls", excluded_calls),
                ("Why actual cost is unknown", cost.actual_h100_cost_unknown_reason),
                ("Published-rate estimate", f"${_format_number(estimate.amount_usd)}"),
                ("Estimate classification", estimate.classification),
                ("GPU rate", f"${_format_number(estimate.gpu_rate_usd_per_second)} / second"),
                ("Estimate time basis", estimate.time_basis),
                ("Estimate limitation", estimate.limitations),
            )
        )
        + "</div>"
        '<p class="source-note">Rate source: '
        + _link(estimate.rate_source.url, estimate.rate_source.url)
        + f" · checked {_escape(estimate.rate_source.checked_at_utc)} · "
        + _link(
            _repo_href(estimate.rate_source.path, output_path),
            estimate.rate_source.path,
        )
        + f" · SHA-256 <code>{_escape(estimate.rate_source.sha256)}</code></p>"
    )
    return _section(
        "accounting",
        "Operational record",
        "Attempts and cost accounting",
        "Observed collection attempts are separated from an explicitly estimated GPU line item. Unknown billing data is not inferred.",
        body,
    )


def _hardware_publication_section(
    publication: Publication, protocol_rows: Iterable[tuple[str, str]]
) -> str:
    hardware = publication.hardware
    hardware_facts = (
        ("Requested GPU", hardware.gpu),
        ("Observed device", hardware.device_name),
        ("Compute capability", ".".join(str(item) for item in hardware.compute_capability)),
        ("Multiprocessors", str(hardware.multiprocessor_count)),
        ("Total memory", f"{_format_number(hardware.total_memory_gb)} GiB"),
        ("CUDA", hardware.cuda_version or "Not reported"),
        ("PyTorch", hardware.torch_version or "Not reported"),
        ("Triton", hardware.triton_version or "Not reported"),
    )
    runtime_rows = publication.runtime or (("Runtime", "Not reported"),)
    body = (
        '<div class="split">'
        + _facts(hardware_facts)
        + _facts(runtime_rows)
        + "</div>"
        + _table(("Protocol field", "Value"), protocol_rows, caption="Collection protocol")
    )
    return _section(
        "hardware",
        "Execution context",
        "Hardware, runtime, and protocol",
        "The observed hardware identity is distinct from the recorded runtime and study protocol. These facts bound the report; they do not broaden its claim scope.",
        body,
    )


def _publication_section(
    publication: Publication,
    related: CommittedEvidence | None,
    output_path: Path,
) -> str:
    raw = publication.raw
    journal = publication.journal
    local_rows = [
        (
            "Published raw archive",
            raw.path,
            f"compressed {raw.sha256}; uncompressed {raw.uncompressed_sha256}",
        ),
        ("Attempt journal", journal.path, journal.sha256),
        (
            "Publication manifest",
            publication.manifest_path,
            "Binds the raw archive, attempt journal, and summary; does not bind this report.",
        ),
        (
            "Canonical research catalog",
            "benchmarks/research-artifact-manifest.json",
            "Binds the generated report digest.",
        ),
    ]
    linked_rows = "".join(
        "<tr>"
        f'<th scope="row">{_escape(label)}</th>'
        f"<td>{_link(_repo_href(path, output_path), path)}</td>"
        f"<td><code>{_escape(digest)}</code></td>"
        "</tr>"
        for label, path, digest in local_rows
    )
    related_html = ""
    if related is not None:
        related_rows = (
            (
                "Precision attempt journal",
                related.attempt_journal.path,
                related.attempt_journal.sha256,
            ),
            ("Precision manifest", related.manifest.path, related.manifest.sha256),
            (
                "Precision raw archive",
                related.raw.path,
                f"compressed {related.raw.compressed_sha256}; decompressed {related.raw.decompressed_sha256}",
            ),
            ("Precision summary", related.summary.path, related.summary.sha256),
        )
        related_html = (
            '<h3>Related precision evidence</h3><div class="table-wrap" tabindex="0" '
            'role="region" aria-label="Related precision evidence"><table class="data-table">'
            '<thead><tr><th scope="col">Artifact</th><th scope="col">Path</th>'
            '<th scope="col">SHA-256</th></tr></thead><tbody>'
            + "".join(
                "<tr>"
                f'<th scope="row">{_escape(label)}</th>'
                f"<td>{_link(_repo_href(path, output_path), path)}</td>"
                f"<td><code>{_escape(digest)}</code></td>"
                "</tr>"
                for label, path, digest in related_rows
            )
            + "</tbody></table></div>"
        )
    calls = ", ".join(publication.modal.call_ids)
    body = (
        '<div class="table-wrap" tabindex="0" role="region" aria-label="Published artifact provenance">'
        '<table class="data-table"><caption>Published artifacts and digests</caption><thead><tr>'
        '<th scope="col">Artifact</th><th scope="col">Path</th><th scope="col">Digest or digest note</th>'
        f"</tr></thead><tbody>{linked_rows}</tbody></table></div>"
        + _facts(
            (
                ("Raw archive bytes", str(raw.bytes)),
                ("Raw archive rows", str(raw.rows)),
                ("Journal records", str(journal.records)),
                ("Repository HEAD", publication.head_commit),
                ("Collector source SHA-256", publication.source_sha256),
                ("Wheel SHA-256", publication.wheel_sha256),
                ("Modal app ID", publication.modal.app_id),
                ("Modal URL provenance", publication.modal.app_url_provenance),
                ("Modal call IDs", calls),
            ),
            class_name="facts facts-wide provenance-facts",
        )
        + '<p class="source-note">Operator-recorded Modal application: '
        + _link(publication.modal.app_url, publication.modal.app_url)
        + ". This link identifies the application record; artifact digests authenticate committed bytes.</p>"
        + related_html
    )
    return _section(
        "provenance",
        "Chain of custody",
        "Publication provenance",
        "The linked publication manifest binds the raw archive, attempt journal, and summary. The canonical research catalog separately binds this generated report digest.",
        body,
    )


def _limitations_section(limitations: tuple[str, ...]) -> str:
    items = "".join(f"<li>{_escape(item)}</li>" for item in limitations)
    return _section(
        "limitations",
        "Interpretation boundary",
        "Limitations",
        "These limitations are part of the typed claim record, not optional caveats.",
        f'<ol class="limitations">{items}</ol>',
    )


def _hopper_body(summary: HopperSummary, output_path: Path) -> str:
    screen = summary.cost_screen
    regime_rows = (
        (
            regime.name,
            regime.decision,
            str(regime.workload_count),
            _format_number(regime.geometric_mean_speedup),
            _format_number(regime.median_speedup),
            _format_number(regime.minimum_speedup),
            _format_number(regime.maximum_speedup),
            _percent(regime.percent_at_least_five_percent_faster),
            f"{regime.workloads_at_least_five_percent_faster}/{regime.workload_count}",
            _bool(regime.all_selected_correct),
        )
        for regime in summary.regimes
    )
    decision_body = (
        '<div class="decision stop"><p class="decision-label">Frozen engineering gate</p>'
        f'<p class="decision-value">{_escape(summary.global_decision)}</p>'
        "<p>Neither regime met the frozen screen. Three-bank selection/scoring collection was not performed.</p></div>"
        '<div class="notice direction"><strong>Ratio direction:</strong> '
        f"{_escape(summary.protocol.ratio)}. Values above 1 mean torch was slower and the selected candidate was faster; values below 1 mean torch was faster.</div>"
        '<div class="notice warning"><strong>Same-bank limitation:</strong> candidates were selected and scored on bank 0. Selection optimism is possible; this is not held-out superiority evidence.</div>'
        + _facts(
            (
                (
                    "Geometric-mean threshold",
                    _format_number(screen.geometric_mean_speedup_threshold),
                ),
                (
                    "Required workload win fraction",
                    _percent(screen.required_fraction_at_least_five_percent_faster),
                ),
                ("Per-workload win threshold", _format_number(screen.speedup_threshold_for_win)),
                (
                    "Regimes evaluated independently",
                    _bool(screen.evaluated_independently_by_regime),
                ),
                ("Three-bank collection performed", _bool(summary.three_bank_collection_performed)),
                ("Contextual archived ratio", summary.contextual_baseline.display_value),
                ("Contextual ratio role", summary.contextual_baseline.role),
            ),
            class_name="facts facts-wide",
        )
        + _table(
            (
                "Regime",
                "Decision",
                "Workloads",
                "Geomean torch / candidate",
                "Median",
                "Min",
                "Max",
                "≥5% faster",
                "Wins",
                "Correct",
            ),
            regime_rows,
            caption="Engineering regime gate results",
        )
    )
    candidate_rows = (
        (
            str(index),
            item.workload_key,
            item.regime,
            item.best_config_key,
            item.best_config.display(),
            _format_number(item.torch_ms),
            _format_number(item.best_candidate_ms),
            _format_number(item.torch_over_best_candidate),
            _format_number(item.archive_ratio),
            _bool(item.correct),
        )
        for index, item in enumerate(summary.candidate_selection, start=1)
    )
    selection_body = _table(
        (
            "#",
            "Workload",
            "Regime",
            "Selected configuration",
            "Launch parameters",
            "Torch ms",
            "Candidate ms",
            "Torch / candidate",
            "Archived contextual ratio",
            "Correct",
        ),
        candidate_rows,
        caption=f"Complete per-workload post-hoc selections ({len(summary.candidate_selection)} workloads)",
        table_class="selection-table",
    )
    finding = summary.precision_finding
    precision_body = (
        f'<div class="decision diagnostic"><p class="decision-label">Related diagnostic</p>'
        f'<p class="decision-value small">{_escape(finding.classification)}</p>'
        f"<p>{_escape(finding.conclusion)}</p></div>"
        + _facts(
            (
                (
                    "Reduced / strict torch median",
                    _format_number(finding.metrics.torch_reduced_over_torch_strict_median),
                ),
                (
                    "Strict / Triton median",
                    _format_number(finding.metrics.torch_strict_over_triton_median),
                ),
                (
                    "Reduced / Triton median",
                    _format_number(finding.metrics.torch_reduced_over_triton_median),
                ),
                ("Accuracy regression", _bool(finding.accuracy_regression)),
                ("Archived baseline agrees", _bool(finding.baseline_agrees)),
                ("Parity authorized", _bool(finding.parity_authorized)),
            ),
            class_name="facts facts-wide",
        )
    )
    protocol_rows = (
        ("GPU", summary.protocol.gpu),
        ("Bank", str(summary.protocol.bank)),
        ("Ratio", summary.protocol.ratio),
        ("Workloads", str(summary.protocol.workload_count)),
        ("Rows", str(summary.protocol.row_count)),
    )
    return (
        _section(
            "decision",
            "Primary result",
            "Engineering gate decision",
            "The gate is an exploratory cost screen over one bank. A STOP result ends this branch; it does not establish a general performance ordering.",
            decision_body,
        )
        + _claim_section(summary.claim_classification, no_superiority=summary.claim)
        + _section(
            "selections",
            "Complete audit table",
            "Per-workload selections",
            "Every post-hoc selected candidate is shown. The archived ratio is contextual bank-1-selection/bank-2-scoring evidence and is not the comparator used by this same-bank screen.",
            selection_body,
        )
        + _section(
            "precision-finding",
            "Related diagnostic",
            "FP16 reduction finding",
            "This diagnostic cannot revise the Hopper STOP and is not a confirmatory Parhelion endpoint.",
            precision_body,
        )
        + _accounting_section(summary.collection_accounting, output_path)
        + _hardware_publication_section(summary.publication, protocol_rows)
        + _publication_section(
            summary.publication,
            summary.precision_finding.committed_evidence,
            output_path,
        )
        + _limitations_section(summary.limitations)
    )


def _precision_body(summary: PrecisionSummary, output_path: Path) -> str:
    finding = summary.precision_finding
    metrics = finding.metrics
    metric_rows = (
        (
            "Reduced torch / strict torch",
            _format_number(metrics.torch_reduced_over_torch_strict_median),
            "Below 1 favors reduced; above 1 favors strict",
        ),
        (
            "Strict torch / Triton",
            _format_number(metrics.torch_strict_over_triton_median),
            "Below 1 means strict torch is faster",
        ),
        (
            "Reduced torch / Triton",
            _format_number(metrics.torch_reduced_over_triton_median),
            "Below 1 means reduced torch is faster",
        ),
        (
            "Archived torch / best Triton",
            _format_number(metrics.archive_baseline_torch_over_best_triton),
            "Contextual frozen bank-1-selected, bank-2-scored result",
        ),
        (
            "Paired strict slowdown",
            _format_number(metrics.paired_strict_slowdown),
            "Compared with the frozen meaningful-effect threshold",
        ),
    )
    decision_body = (
        '<div class="decision diagnostic"><p class="decision-label">Exploratory diagnostic</p>'
        f'<p class="decision-value small">{_escape(finding.classification)}</p>'
        f"<p>{_escape(finding.conclusion)}</p></div>"
        '<div class="notice direction"><strong>Ratio direction:</strong> each numerator latency is divided by its denominator latency. Ratios below 1 favor the numerator; ratios above 1 favor the denominator.</div>'
        + _table(
            ("Metric", "Median ratio", "Interpretation"),
            metric_rows,
            caption="Strict and reduced precision ratios",
        )
        + _facts(
            (
                ("Accuracy regression", _bool(finding.accuracy_regression)),
                ("Archived baseline agrees", _bool(finding.baseline_agrees)),
                ("Parity authorized", _bool(finding.parity_authorized)),
                (
                    "Meaningful paired effect threshold",
                    _format_number(finding.thresholds.meaningful_paired_effect),
                ),
                (
                    "Baseline relative tolerance",
                    _percent(finding.thresholds.baseline_relative_tolerance),
                ),
                (
                    "Baseline absolute tolerance",
                    _format_number(finding.thresholds.baseline_absolute_tolerance),
                ),
            ),
            class_name="facts facts-wide",
        )
        + '<div class="notice warning"><strong>Comparator limitation:</strong> the archived comparator selected Triton configurations on bank 1 and scored bank 2, while this probe reports three-bank medians. The protocols are not interchangeable.</div>'
    )
    protocol = summary.protocol
    protocol_rows = (
        ("GPU selector", f"{protocol.gpu} · {protocol.modal_selector}"),
        ("Banks", ", ".join(str(item) for item in protocol.banks)),
        ("Arms", ", ".join(protocol.arms)),
        ("Arm order", protocol.arm_order),
        ("Arm-order seed", protocol.arm_order_seed),
        ("Reference", protocol.reference),
        ("Statistic", protocol.statistic),
        ("Quantiles", ", ".join(_format_number(item) for item in protocol.quantiles)),
        ("Warmup / repetition", f"{protocol.warmup_ms} ms / {protocol.rep_ms} ms"),
        ("Tensor seed", protocol.tensor_seed),
        ("Tensor seed protocol", protocol.tensor_seed_protocol),
        ("Retry policy", protocol.retry_policy),
        ("Protocol role", protocol.role),
        ("Workloads / rows", f"{summary.workload_count} / {summary.row_count}"),
    )
    return (
        _section(
            "decision",
            "Primary result",
            "Precision diagnostic conclusion",
            "The strict/reduced comparison is a paired exploratory diagnostic. It tests one proposed explanation; it does not create a superiority or parity claim.",
            decision_body,
        )
        + _claim_section(
            summary.claim_classification,
            no_superiority="No superiority claim is made; the supported decision applies only to the typed does-not-explain diagnostic.",
        )
        + _accounting_section(summary.collection_accounting, output_path)
        + _hardware_publication_section(summary.publication, protocol_rows)
        + _publication_section(summary.publication, None, output_path)
        + _limitations_section(summary.limitations)
    )


def _fusion_remote_body(summary: FusionRemoteSummary, output_path: Path) -> str:
    attempt_rows = tuple(
        (
            attempt.attempt_id,
            attempt.suite_id,
            attempt.status,
            f"{attempt.app.app_id} · operator-recorded; artifact binding: none",
            f"{attempt.call.function_call_id} · artifact-bound remote journal",
            (
                _FUSION_UNRESOLVED_SEQUENCE
                if attempt.status == "unresolved"
                else "completed receipt returned"
            ),
        )
        for attempt in summary.attempts
    )
    attempt_table = _table(
        ("Attempt", "Suite", "Status", "Modal app", "FunctionCall", "Retained lifecycle record"),
        attempt_rows,
        caption="Four retained remote attempts and identity provenance",
        table_class="selection-table",
    )
    status_body = (
        '<div class="decision diagnostic">'
        '<p class="decision-label">Receipt classification</p>'
        '<p class="decision-value small">Exploratory</p>'
        "<p>Four retained client-authorized attempts: two gated-MLP calls retained this sequence: "
        f"{_FUSION_UNRESOLVED_SEQUENCE}. One gated-MLP and one residual-RMSNorm call returned "
        "completed receipts. Completed metrics are measured facts only.</p></div>"
        '<div class="notice warning"><strong>No fusion or superiority claim.</strong> '
        "Candidate and reference arithmetic are identical. The candidate distinction is "
        "full-graph Inductor compilation only; every completed environment records "
        "<code>fusion_claim=false</code>.</div>"
    )

    correctness_rows: list[tuple[str, ...]] = []
    compile_rows: list[tuple[str, ...]] = []
    timing_rows: list[tuple[str, ...]] = []
    ratio_rows: list[tuple[str, ...]] = []
    for result in summary.completed_results:
        metrics = result.metrics
        for arm, correctness in (
            ("candidate", metrics.candidate_correctness),
            ("reference", metrics.reference_correctness),
        ):
            correctness_rows.append(
                (
                    result.suite_id,
                    arm,
                    correctness.status,
                    _bool(correctness.close),
                    _bool(correctness.finite),
                    _bool(correctness.input_storage_unchanged),
                    _bool(correctness.output_disjoint),
                    _format_number(correctness.max_abs_error),
                )
            )
        compile_rows.append(
            (
                result.suite_id,
                metrics.compile.arm_id,
                metrics.compile.status,
                _bool(metrics.compile.backend_invoked),
                _bool(metrics.compile.callable_distinct),
                _bool(metrics.compile.eager_fallback),
                _format_number(metrics.compile.wrapper_create_ns / 1_000_000),
                _format_number(metrics.compile.first_call_ns / 1_000_000),
            )
        )
        for arm, timing in (
            ("candidate", metrics.candidate_timing),
            ("reference", metrics.reference_timing),
        ):
            timing_rows.append(
                (
                    result.suite_id,
                    arm,
                    timing.status,
                    str(timing.warmups),
                    str(timing.repetitions),
                    _format_number(timing.median_ms),
                )
            )
        ratio_rows.append(
            (
                result.suite_id,
                _format_number(metrics.ratios.candidate_to_reference_median),
                _format_number(metrics.ratios.reference_to_candidate_median),
                metrics.ratios.interpretation,
                _bool(metrics.ratios.superiority_tested),
            )
        )
    observations = (
        _table(
            (
                "Suite",
                "Arm",
                "Status",
                "Close",
                "Finite",
                "Input unchanged",
                "Output disjoint",
                "Max abs. error",
            ),
            correctness_rows,
            caption="Returned correctness observations",
            table_class="selection-table",
        )
        + _table(
            (
                "Suite",
                "Compile arm",
                "Status",
                "Backend invoked",
                "Callable distinct",
                "Eager fallback",
                "Wrapper create (ms)",
                "First call (ms)",
            ),
            compile_rows,
            caption="Returned compile observations",
            table_class="selection-table",
        )
        + _table(
            ("Suite", "Arm", "Status", "Warmups", "Raw samples", "Median (ms)"),
            timing_rows,
            caption="Returned timing observations",
        )
    )

    raw_href = _repo_href(FUSION_REMOTE_RAW_PATH, output_path)
    summary_href = _repo_href(FUSION_REMOTE_SUMMARY_PATH, output_path)
    manifest_href = _repo_href(FUSION_REMOTE_MANIFEST_PATH, output_path)
    interpretation = (
        '<div class="notice direction"><strong>Ratio direction.</strong> '
        "<code>candidate / reference</code> is candidate median divided by reference median; "
        "a value below 1 means the candidate returned the lower median. "
        "<code>reference / candidate</code> is its reciprocal. Both are descriptive ratios "
        "of returned medians, not a superiority test.</div>"
        + _table(
            (
                "Suite",
                "Candidate / reference",
                "Reference / candidate",
                "Interpretation",
                "Superiority tested",
            ),
            ratio_rows,
            caption="Descriptive returned-median ratios in both directions",
        )
        + '<div class="notice"><strong>Raw-sample stability boundary.</strong> '
        "Each timing row summarizes 50 retained raw samples after 10 warmups. The raw archive "
        "preserves every sample sequence, but no stability threshold, variance estimate, "
        "uncertainty interval, or comparative analysis plan was prespecified. "
        f"{_link(raw_href, 'Inspect the compressed raw evidence')}.</div>"
    )

    hardware = summary.completed_results[0].hardware
    hardware_facts = _facts(
        (
            ("GPU", f"{hardware.device_name} · {hardware.gpu}"),
            (
                "Compute capability / SMs",
                f"{hardware.compute_capability[0]}.{hardware.compute_capability[1]} / "
                f"{hardware.multiprocessor_count}",
            ),
            ("CUDA / torch", f"{hardware.cuda_version} / {hardware.torch_version}"),
            (
                "Triton / memory",
                f"{hardware.triton_version} / {_format_number(hardware.total_memory_gb)} GB",
            ),
        ),
        class_name="facts facts-wide",
    )
    provenance_rows = tuple(
        (
            attempt.attempt_id,
            attempt.head_commit,
            attempt.source_sha256,
            attempt.wheel_filename,
            attempt.wheel_sha256,
        )
        for attempt in summary.attempts
    )
    provenance = (
        hardware_facts + '<div class="notice warning"><strong>Historical build boundary.</strong> '
        "The four attempts came from four different historical HEAD commits, source digests, "
        "and wheel digests. Results must remain attached to their own attempt; they are not "
        "measurements of one interchangeable build.</div>"
        + _table(
            ("Attempt", "HEAD", "Source SHA-256", "Wheel", "Wheel SHA-256"),
            provenance_rows,
            caption="Historical source and wheel bindings by attempt",
            table_class="selection-table",
        )
    )

    provider = summary.provider_accounting
    accounting = (
        _facts(
            (
                ("Client-authorized spawns", str(provider.client_authorized_spawns)),
                ("Provider physical attempts", "Unknown / unobservable"),
                ("Provider starts or restarts", "Unknown / unobservable"),
                ("Total GPU time", "Unknown"),
                ("Actual cost", "Unknown"),
                ("Attestation", "None present"),
                ("Cost status", provider.cost_status),
                ("Publication eligible", _bool(summary.publication_eligible)),
            ),
            class_name="facts facts-wide",
        )
        + '<div class="notice warning"><strong>Receipt and publication boundary.</strong> '
        "Both completed receipts record <code>publication_eligible=false</code>. Provider "
        "physical starts and restarts, provider attempt count, total GPU time, and actual cost "
        "are unknown; no attestation is present.</div>"
    )
    artifact_links = (
        '<p class="source-note">Evidence files: '
        f"{_link(raw_href, 'compressed raw evidence')} · "
        f"{_link(summary_href, 'strict source summary')} · "
        f"{_link(manifest_href, 'publication manifest')}</p>"
    )
    publication = (
        _facts(
            (
                ("Summary schema", summary.schema),
                ("Analysis status", summary.methodology.analysis_status),
                ("Performance inference", summary.methodology.performance_inference),
                ("Source report status", summary.methodology.report_status),
                ("Fusion claim", _bool(summary.methodology.fusion_claim)),
                ("Superiority claim", _bool(summary.methodology.superiority_claim)),
                ("Publication eligible", _bool(summary.methodology.publication_eligible)),
                ("Completed result scope", "Measured fact only"),
            ),
            class_name="facts facts-wide provenance-facts",
        )
        + artifact_links
        + '<div class="notice"><strong>Binding status.</strong> The immutable source summary '
        "and manifest record <code>report_status=not_created</code>. This deterministic page "
        "renders that source without changing it and does not create a fusion, superiority, "
        "attestation, or publication-eligibility claim.</div>"
    )
    corrected_limitations = (
        "Two gated-MLP calls retained this sequence: "
        f"{_FUSION_UNRESOLVED_SEQUENCE}. No result receipts exist for them.",
        *summary.limitations[1:],
    )
    return (
        _section(
            "status",
            "Receipt status",
            "Exploratory remote evidence",
            "A deterministic synthesis of four retained H100 remote attempts. It separates unresolved lifecycle evidence from two completed measured observations.",
            status_body,
        )
        + _section(
            "attempts",
            "Lifecycle",
            "Four attempts: two completed, two unresolved",
            "Modal app identities are operator-recorded and have no artifact binding. FunctionCall identities are bound by each retained remote journal.",
            attempt_table,
        )
        + _section(
            "observations",
            "Measured facts",
            "Correctness, compile, and timing",
            "Only the two completed receipts contribute observation rows. No result is imputed for either unresolved 401 attempt.",
            observations,
        )
        + _section(
            "interpretation",
            "Reading the numbers",
            "Raw-sample stability and ratio direction",
            "Returned medians are displayed in both ratio directions with the limits required to interpret them.",
            interpretation,
        )
        + _section(
            "provenance",
            "Environment and bytes",
            "H100 environment and historical builds",
            "The two completed environments agree, while each attempt remains bound to distinct historical source and wheel bytes.",
            provenance,
        )
        + _section(
            "accounting",
            "Unknowns",
            "Provider accounting and attestation",
            "Client-authorized calls are countable; provider execution lifecycle and spend are not observable from the retained evidence.",
            accounting,
        )
        + _section(
            "publication",
            "Evidence access",
            "Raw, summary, and manifest",
            "Repository-relative links expose the immutable evidence without promoting these receipts to publication eligibility.",
            publication,
        )
        + _limitations_section(corrected_limitations)
    )


def _styles() -> str:
    return """
:root {
  --paper: #f4f0e7;
  --surface: #fbf8f1;
  --ink: #142a33;
  --muted: #53656a;
  --line: #c9c0ae;
  --accent: #a23b2a;
  --accent-soft: #f1d9cf;
  --teal: #176b67;
  --teal-soft: #d9e8e3;
  --warning: #76591b;
  --warning-soft: #eee2bd;
  --shadow: 0 0.5rem 1.5rem rgb(20 42 51 / 0.08);
  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-7: 3rem;
  --space-8: 4rem;
  --measure: 76rem;
  --reading: 48rem;
  --line-thin: 0.0625rem;
  --line-heavy: 0.25rem;
  --type-xs: 0.75rem;
  --type-sm: 0.875rem;
  --type-base: 1rem;
  --type-md: 1.125rem;
  --type-lg: 1.5rem;
  --type-xl: clamp(2.25rem, 6vw, 4.75rem);
  color-scheme: light;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
  font-size: var(--type-base);
  line-height: 1.6;
}
a { color: var(--teal); text-decoration-thickness: var(--line-thin); text-underline-offset: var(--space-1); }
a:hover { text-decoration-thickness: calc(var(--line-thin) * 2); }
a:focus-visible, [tabindex="0"]:focus-visible { outline: var(--line-heavy) solid var(--accent); outline-offset: var(--space-1); }
.skip-link { position: absolute; left: var(--space-4); top: -5rem; padding: var(--space-2) var(--space-4); background: var(--ink); color: var(--surface); z-index: 2; }
.skip-link:focus { top: var(--space-4); }
.page { width: min(100% - var(--space-6), var(--measure)); margin-inline: auto; }
.report-header { padding: var(--space-6) 0 var(--space-8); border-bottom: var(--line-thin) solid var(--line); }
.utility { display: flex; gap: var(--space-4); align-items: center; justify-content: space-between; flex-wrap: wrap; }
.lockup, .badge, .eyebrow { font-size: var(--type-xs); font-weight: 750; letter-spacing: 0.11em; text-transform: uppercase; }
.badge { display: inline-flex; padding: var(--space-1) var(--space-3); border: var(--line-thin) solid var(--accent); border-radius: var(--radius-sm); background: var(--accent-soft); color: var(--accent); }
.kicker { margin: var(--space-7) 0 var(--space-2); color: var(--accent); font-weight: 700; }
h1, h2, h3 { font-family: Charter, Georgia, serif; line-height: 1.08; }
h1 { max-width: 16ch; margin: 0; font-size: var(--type-xl); letter-spacing: -0.035em; }
.lede { max-width: var(--reading); margin: var(--space-5) 0 0; font-size: var(--type-md); color: var(--muted); }
.header-meta { margin-top: var(--space-6); }
.report-section { padding: var(--space-7) 0; border-bottom: var(--line-thin) solid var(--line); }
.section-heading { display: grid; grid-template-columns: minmax(9rem, 0.35fr) minmax(0, 1fr); gap: var(--space-5); margin-bottom: var(--space-6); }
.section-heading .eyebrow { grid-row: 1 / span 2; margin: var(--space-2) 0 0; color: var(--accent); }
.section-heading h2 { margin: 0; font-size: clamp(1.75rem, 4vw, 2.75rem); }
.section-heading > p:last-child { max-width: var(--reading); margin: 0; color: var(--muted); }
h3 { margin: var(--space-6) 0 var(--space-3); font-size: var(--type-lg); }
.decision { display: grid; grid-template-columns: minmax(10rem, 0.35fr) minmax(0, 1fr); gap: var(--space-4) var(--space-6); padding: var(--space-5); border-left: var(--line-heavy) solid var(--accent); background: var(--accent-soft); box-shadow: var(--shadow); }
.decision-label { margin: 0; font-size: var(--type-xs); font-weight: 750; letter-spacing: 0.1em; text-transform: uppercase; }
.decision-value { grid-row: 2; margin: 0; font-family: Charter, Georgia, serif; font-size: clamp(3rem, 10vw, 7rem); line-height: 0.8; color: var(--accent); }
.decision-value.small { font-size: clamp(2rem, 7vw, 4rem); }
.decision > p:last-child { grid-column: 2; grid-row: 1 / span 2; align-self: center; margin: 0; font-size: var(--type-md); }
.decision.diagnostic { border-color: var(--teal); background: var(--teal-soft); }
.decision.diagnostic .decision-value { color: var(--teal); }
.notice { margin: var(--space-5) 0; padding: var(--space-4); border: var(--line-thin) solid var(--line); border-radius: var(--radius-sm); background: var(--surface); }
.notice.direction { border-left: var(--line-heavy) solid var(--teal); }
.notice.warning { border-left: var(--line-heavy) solid var(--warning); background: var(--warning-soft); }
.split { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-5); }
.facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--line-thin); margin: var(--space-5) 0; padding: var(--line-thin); background: var(--line); border-radius: var(--radius-sm); overflow: hidden; }
.facts > div { min-width: 0; padding: var(--space-3) var(--space-4); background: var(--surface); }
.facts dt { margin-bottom: var(--space-1); color: var(--muted); font-size: var(--type-xs); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.facts dd { margin: 0; overflow-wrap: anywhere; }
.facts-wide { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.provenance-facts dd { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-size: var(--type-sm); }
.table-wrap { overflow-x: auto; margin: var(--space-5) 0; border: var(--line-thin) solid var(--line); border-radius: var(--radius-sm); background: var(--surface); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
caption { padding: var(--space-3) var(--space-4); text-align: left; color: var(--muted); font-size: var(--type-sm); font-weight: 700; }
th, td { padding: var(--space-3) var(--space-4); border-top: var(--line-thin) solid var(--line); text-align: left; vertical-align: top; }
thead th { border-top: 0; background: var(--ink); color: var(--surface); font-size: var(--type-xs); letter-spacing: 0.04em; text-transform: uppercase; }
tbody th { font-weight: 700; }
tbody tr:nth-child(even) { background: var(--paper); }
.selection-table { min-width: 88rem; font-size: var(--type-sm); }
code { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-size: var(--type-sm); overflow-wrap: anywhere; }
.source-note { color: var(--muted); font-size: var(--type-sm); overflow-wrap: anywhere; }
.limitations { max-width: var(--reading); margin: 0; padding-left: var(--space-5); }
.limitations li { padding: var(--space-2) 0 var(--space-2) var(--space-2); }
.footer { display: flex; justify-content: space-between; gap: var(--space-4); padding: var(--space-5) 0 var(--space-7); color: var(--muted); font-size: var(--type-sm); }
@media (max-width: 52rem) {
  .section-heading, .decision, .split { grid-template-columns: 1fr; }
  .section-heading .eyebrow, .decision-value, .decision > p:last-child { grid-column: 1; grid-row: auto; }
  .facts, .facts-wide { grid-template-columns: 1fr; }
  .footer { flex-direction: column; }
}
@media print {
  :root { --paper: #f7f3eb; --surface: #fdfaf3; --ink: #172a31; }
  .page { width: 100%; }
  a { color: inherit; }
  .table-wrap { overflow: visible; }
}
""".strip()


def _render_document(summary: EngineeringSummary, output_path: Path) -> str:
    if isinstance(summary, HopperSummary):
        title = "Hopper H100 engineering gate"
        badge = "Exploratory engineering gate"
        lede = (
            "A one-bank post-hoc screen of selected Triton candidates against torch.matmul. "
            "The frozen result is STOP; no superiority claim is made."
        )
        scope = summary.evidence_scope
        rows = summary.protocol.row_count
        workloads = summary.protocol.workload_count
        body = _hopper_body(summary, output_path)
        kicker = "Legacy-shaped publication · methodology-compatible typed claims"
        header_facts = _facts(
            (
                ("Study ID", summary.study_id),
                ("Schema", str(summary.schema_version)),
                ("Evidence scope", scope),
                ("Workloads / published rows", f"{workloads} / {rows}"),
            ),
            class_name="facts facts-wide header-meta",
        )
    elif isinstance(summary, PrecisionSummary):
        title = "H100 FP16 reduction diagnostic"
        badge = "Exploratory engineering diagnostic"
        lede = (
            "A paired three-bank diagnostic comparing strict and reduced FP16 reduction. "
            + summary.precision_finding.conclusion
        )
        scope = summary.evidence_scope
        rows = summary.row_count
        workloads = summary.workload_count
        body = _precision_body(summary, output_path)
        kicker = "Legacy-shaped publication · methodology-compatible typed claims"
        header_facts = _facts(
            (
                ("Study ID", summary.study_id),
                ("Schema", str(summary.schema_version)),
                ("Evidence scope", scope),
                ("Workloads / published rows", f"{workloads} / {rows}"),
            ),
            class_name="facts facts-wide header-meta",
        )
    else:
        title = "H100 fusion remote receipts"
        badge = "Exploratory receipt status"
        lede = (
            "Four retained remote attempts: two unresolved gated-MLP 401 outcomes and two "
            "completed measured observations. No fusion or superiority claim is made."
        )
        body = _fusion_remote_body(summary, output_path)
        kicker = "Remote receipt synthesis · measured facts only"
        header_facts = _facts(
            (
                ("Study ID", summary.study_id),
                ("Schema", summary.schema),
                ("Attempt status", "2 unresolved / 2 completed"),
                ("Publication eligible", _bool(summary.publication_eligible)),
            ),
            class_name="facts facts-wide header-meta",
        )
    return (
        "<!doctype html>"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'none'; base-uri 'none'; form-action 'none'\">"
        f"<title>{_escape(title)} · HeliosTune</title><style>{_styles()}</style></head><body>"
        '<a class="skip-link" href="#main">Skip to report</a><div class="page">'
        '<header class="report-header"><div class="utility">'
        '<span class="lockup">HeliosTune // engineering evidence</span>'
        f'<span class="badge">{_escape(badge)}</span></div>'
        f'<p class="kicker">{_escape(kicker)}</p>'
        f'<h1>{_escape(title)}</h1><p class="lede">{_escape(lede)}</p>{header_facts}</header>'
        f'<main id="main">{body}</main><footer class="footer">'
        "<span><strong>HeliosTune</strong> / engineering evidence record</span>"
        "<span>Standalone HTML · no scripts · no network dependencies</span>"
        "</footer></div></body></html>"
    )


def render_engineering_report(summary: object, output_path: str | Path) -> None:
    """Strictly parse and atomically render one supported engineering study."""
    parsed = parse_engineering_summary(summary)
    destination = Path(output_path)
    write_text_atomic(destination, _render_document(parsed, destination))
