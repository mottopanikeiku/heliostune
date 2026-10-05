"""Immutable validated report data shared by every HTML renderer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from heliostune.errors import SchemaError

MethodRole = Literal["sequential", "zero_query", "external", "exhaustive", "reference"]
ReadOnlyMapping: TypeAlias = Mapping[str, object]
_ROLES = {"sequential", "zero_query", "external", "exhaustive", "reference"}
_HISTORICAL_ROLES: Mapping[str, MethodRole] = MappingProxyType(
    {
        "static": "zero_query",
        "static_multisource": "zero_query",
        "torch": "external",
        "random": "sequential",
        "nearest_shape": "sequential",
        "single_source_nearest": "sequential",
        "multisource_retrieval": "sequential",
        "cold": "sequential",
        "cold_thompson": "sequential",
        "transfer": "sequential",
        "transfer_thompson": "sequential",
        "pooled_source_thompson": "sequential",
        "parhelion_thompson": "sequential",
        "parhelion_no_forced_anchor": "sequential",
        "parhelion_no_transfer": "sequential",
        "anchored_cold_thompson": "sequential",
        "official_triton_config_exhaustive": "exhaustive",
        "exhaustive": "exhaustive",
        "heldout_reference": "reference",
    }
)
_STOCHASTIC_METHODS = {
    "random",
    "cold",
    "cold_thompson",
    "transfer",
    "transfer_thompson",
    "pooled_source_thompson",
    "parhelion_thompson",
    "parhelion_no_forced_anchor",
    "parhelion_no_transfer",
    "anchored_cold_thompson",
}


@dataclass(frozen=True, slots=True)
class ReportUncertainty:
    estimand: str
    sampling_unit: str
    n: int
    conditional_on: str
    interval_method: str
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ReportCurvePoint:
    budget: float
    mean: float
    uncertainty: ReportUncertainty


@dataclass(frozen=True, slots=True)
class ReportMethod:
    key: str
    label: str
    role: MethodRole
    points: tuple[ReportCurvePoint, ...]


@dataclass(frozen=True, slots=True)
class ReportPrimaryEvidence:
    method: str
    comparator: str
    mean_delta: float
    uncertainty: ReportUncertainty
    paired_seeds: int
    degrees_of_freedom: int
    superiority_supported: bool
    claim: str | None


@dataclass(frozen=True, slots=True)
class ReportHardware:
    gpu: str
    device_name: str
    facts: ReadOnlyMapping


@dataclass(frozen=True, slots=True)
class ReportFold:
    heldout_model: str
    target_workloads: int | None
    facts: ReadOnlyMapping


@dataclass(frozen=True, slots=True)
class ReportCosts:
    source: ReadOnlyMapping
    target: ReadOnlyMapping


@dataclass(frozen=True, slots=True)
class ReportProvenance:
    facts: ReadOnlyMapping


@dataclass(frozen=True, slots=True)
class ReportData:
    source_label: str
    target_label: str
    max_budget: int
    methods: tuple[ReportMethod, ...]
    primary: tuple[ReportPrimaryEvidence, ...]
    hardware: tuple[ReportHardware, ...]
    folds: tuple[ReportFold, ...]
    costs: ReportCosts
    provenance: ReportProvenance
    limitations: tuple[str, ...]
    raw_summary: ReadOnlyMapping


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be an object")
    if any(type(key) is not str for key in value):
        raise SchemaError(f"{context} keys must be strings")
    return value


def _string(value: object, *, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SchemaError(f"{context} must be nonblank with no surrounding whitespace")
    return value


def _number(value: object, *, context: str) -> float:
    if type(value) not in {int, float}:
        raise SchemaError(f"{context} must be numeric")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise SchemaError(f"{context} must be finite")
    return result


def _positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise SchemaError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise SchemaError(f"{context} must be a non-negative integer")
    return value


def _freeze(value: object) -> object:
    if type(value) is float and not math.isfinite(value):
        raise SchemaError("report contains a non-finite numeric value")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _label(key: str, labels: Mapping[str, object]) -> str:
    supplied = labels.get(key)
    if supplied is not None:
        return _string(supplied, context=f"method label {key!r}")
    return key.replace("_", " ").replace("-", " ").title()


def _role(key: str, declared: Mapping[str, object]) -> MethodRole:
    raw = declared.get(key)
    if raw is None:
        try:
            return _HISTORICAL_ROLES[key]
        except KeyError as exc:
            raise SchemaError(
                f"new report method {key!r} must declare one role from {sorted(_ROLES)!r}"
            ) from exc
    role = _string(raw, context=f"method role {key!r}")
    if role not in _ROLES:
        raise SchemaError(f"unknown method role {role!r} for {key!r}")
    return cast(MethodRole, role)


def _structured_uncertainty(
    value: object,
    *,
    context: str,
    mean: float,
) -> ReportUncertainty:
    data = _mapping(value, context=context)
    low = _number(data.get("low"), context=f"{context} low")
    high = _number(data.get("high"), context=f"{context} high")
    if not low <= mean <= high:
        raise SchemaError(f"{context} must satisfy low <= mean <= high")
    return ReportUncertainty(
        estimand=_string(data.get("estimand"), context=f"{context} estimand"),
        sampling_unit=_string(data.get("sampling_unit"), context=f"{context} sampling_unit"),
        n=_positive_int(data.get("n"), context=f"{context} n"),
        conditional_on=_string(data.get("conditional_on"), context=f"{context} conditional_on"),
        interval_method=_string(data.get("interval_method"), context=f"{context} interval_method"),
        low=low,
        high=high,
    )


def _historical_uncertainty(
    point: Mapping[str, object],
    *,
    method: str,
    mean: float,
    summary: Mapping[str, object],
) -> ReportUncertainty:
    low = _number(point.get("ci95_low"), context=f"method {method!r} ci95_low")
    high = _number(point.get("ci95_high"), context=f"method {method!r} ci95_high")
    if not low <= mean <= high:
        raise SchemaError(f"method {method!r} interval must satisfy low <= mean <= high")
    stochastic = method in _STOCHASTIC_METHODS
    folds = summary.get("folds")
    fold_count = len(folds) if isinstance(folds, Sequence) else 1
    n = _positive_int(
        summary.get("seeds", 1) if stochastic else max(1, fold_count),
        context=f"method {method!r} uncertainty n",
    )
    return ReportUncertainty(
        estimand="mean fraction of held-out reference at the declared target-probe budget",
        sampling_unit="paired policy seed" if stochastic else "equal-weight model-family fold",
        n=n,
        conditional_on="the fixed benchmark matrix, workload corpus, and declared folds",
        interval_method=(
            "historical supplied normal 95% interval"
            if stochastic
            else "historical supplied fold-variation interval"
        ),
        low=low,
        high=high,
    )


def _normalize_methods(summary: Mapping[str, object]) -> tuple[ReportMethod, ...]:
    methods = _mapping(summary.get("methods"), context="report methods")
    if not methods:
        raise SchemaError("report methods must not be empty")
    labels = _mapping(summary.get("method_labels", {}), context="method_labels")
    declared_roles = _mapping(summary.get("method_roles", {}), context="method_roles")
    normalized: list[ReportMethod] = []
    for key, raw_points in methods.items():
        method_key = _string(key, context="method key")
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise SchemaError(f"method {method_key!r} points must be a sequence")
        points: list[ReportCurvePoint] = []
        seen_budgets: set[float] = set()
        for index, raw_point in enumerate(raw_points):
            point = _mapping(raw_point, context=f"method {method_key!r} point {index}")
            budget = _number(
                point.get("budget"), context=f"method {method_key!r} point {index} budget"
            )
            if budget <= 0 or budget in seen_budgets:
                raise SchemaError(f"method {method_key!r} budgets must be positive and unique")
            seen_budgets.add(budget)
            mean = _number(
                point.get("mean_fraction_oracle", point.get("mean")),
                context=f"method {method_key!r} point {index} mean",
            )
            uncertainty = (
                _structured_uncertainty(
                    point["uncertainty"],
                    context=f"method {method_key!r} point {index} uncertainty",
                    mean=mean,
                )
                if "uncertainty" in point
                else _historical_uncertainty(
                    point,
                    method=method_key,
                    mean=mean,
                    summary=summary,
                )
            )
            points.append(ReportCurvePoint(budget, mean, uncertainty))
        normalized.append(
            ReportMethod(
                key=method_key,
                label=_label(method_key, labels),
                role=_role(method_key, declared_roles),
                points=tuple(sorted(points, key=lambda point: point.budget)),
            )
        )
    return tuple(normalized)


def _primary_metric(summary: Mapping[str, object]) -> Mapping[str, object] | None:
    primary_metrics = summary.get("primary_metrics")
    if isinstance(primary_metrics, Mapping):
        value = primary_metrics.get("paired_parhelion_vs_primary_auc_delta")
        if value is not None:
            return _mapping(value, context="primary paired evidence")
    headline = summary.get("headline")
    if isinstance(headline, Mapping):
        value = headline.get("paired_auc_delta_vs_primary")
        if value is not None:
            return _mapping(value, context="headline paired evidence")
    return None


def _normalize_primary(
    summary: Mapping[str, object],
    methods: tuple[ReportMethod, ...],
) -> tuple[ReportPrimaryEvidence, ...]:
    metric = _primary_metric(summary)
    if metric is None:
        return ()
    method_keys = {method.key for method in methods}
    comparator = _string(
        metric.get("comparator", summary.get("primary_comparator")),
        context="primary comparator",
    )
    if comparator not in method_keys:
        raise SchemaError(f"primary comparator {comparator!r} is not a rendered method")
    method = str(summary.get("transfer_method", "parhelion_thompson"))
    if method not in method_keys:
        method = "transfer_thompson" if "transfer_thompson" in method_keys else "transfer"
    if method not in method_keys:
        raise SchemaError("primary method is not a rendered method")
    mean = _number(metric.get("mean_auc_delta"), context="primary mean_auc_delta")
    paired_seeds = _positive_int(metric.get("paired_seeds"), context="primary paired_seeds")
    degrees = _nonnegative_int(
        metric.get("degrees_of_freedom"), context="primary degrees_of_freedom"
    )
    if degrees != paired_seeds - 1:
        raise SchemaError("primary degrees_of_freedom must equal paired_seeds - 1")
    if "uncertainty" in metric:
        uncertainty = _structured_uncertainty(
            metric["uncertainty"], context="primary uncertainty", mean=mean
        )
    else:
        low = _number(metric.get("ci95_low"), context="primary ci95_low")
        high = _number(metric.get("ci95_high"), context="primary ci95_high")
        if not low <= mean <= high:
            raise SchemaError("primary interval must satisfy low <= mean <= high")
        uncertainty = ReportUncertainty(
            estimand="paired Parhelion minus frozen-comparator fraction-reference AUC",
            sampling_unit="paired policy seed",
            n=paired_seeds,
            conditional_on="the fixed benchmark matrix, workload corpus, and campaign",
            interval_method="two-sided 95% Student-t interval",
            low=low,
            high=high,
        )
    if uncertainty.n != paired_seeds:
        raise SchemaError("primary uncertainty n must equal paired_seeds")
    supported = metric.get("superiority_supported")
    if type(supported) is not bool:
        raise SchemaError("primary superiority_supported must be a boolean")
    if supported != (uncertainty.low > 0):
        raise SchemaError("primary superiority_supported must equal (ci95_low > 0)")
    claim_value = metric.get("claim")
    claim = None if claim_value is None else _string(claim_value, context="primary claim")
    return (
        ReportPrimaryEvidence(
            method=method,
            comparator=comparator,
            mean_delta=mean,
            uncertainty=uncertainty,
            paired_seeds=paired_seeds,
            degrees_of_freedom=degrees,
            superiority_supported=supported,
            claim=claim,
        ),
    )


def _normalize_hardware(summary: Mapping[str, object]) -> tuple[ReportHardware, ...]:
    if "hardware" in summary:
        raw = summary["hardware"]
    else:
        collected: list[object] = []
        source_hardware = summary.get("source_hardware")
        if isinstance(source_hardware, Mapping):
            source_profiles = source_hardware.get("profiles", ())
            if isinstance(source_profiles, Sequence) and not isinstance(
                source_profiles, (str, bytes)
            ):
                collected.extend(source_profiles)
        target_hardware = summary.get("target_hardware")
        if isinstance(target_hardware, Mapping):
            collected.append(target_hardware)
        raw = collected
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SchemaError("report hardware must be a sequence")
    profiles: list[ReportHardware] = []
    for index, value in enumerate(raw):
        profile = _mapping(value, context=f"hardware profile {index}")
        gpu = _string(profile.get("gpu"), context=f"hardware profile {index} gpu")
        device = _string(
            profile.get("device_name", gpu),
            context=f"hardware profile {index} device_name",
        )
        profiles.append(
            ReportHardware(
                gpu=gpu,
                device_name=device,
                facts=cast(ReadOnlyMapping, _freeze(profile)),
            )
        )
    return tuple(profiles)


def _normalize_folds(summary: Mapping[str, object]) -> tuple[ReportFold, ...]:
    raw = summary.get("folds", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SchemaError("report folds must be a sequence")
    folds: list[ReportFold] = []
    for index, value in enumerate(raw):
        fold = _mapping(value, context=f"fold {index}")
        if "methods" in fold:
            fold_summary = dict(summary)
            fold_summary["methods"] = fold["methods"]
            _normalize_methods(fold_summary)
        heldout = _string(fold.get("heldout_model"), context=f"fold {index} heldout_model")
        target_raw = fold.get("target_workloads")
        target = (
            None
            if target_raw is None
            else _nonnegative_int(target_raw, context=f"fold {index} target_workloads")
        )
        folds.append(ReportFold(heldout, target, cast(ReadOnlyMapping, _freeze(fold))))
    return tuple(folds)


def _validate_scope(summary: Mapping[str, object]) -> None:
    for key in ("workloads", "configs"):
        if key not in summary:
            raise SchemaError(f"report is missing required field {key!r}")
        value = summary[key]
        if type(value) is int:
            if value <= 0:
                raise SchemaError(f"report {key} count must be positive")
        elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                raise SchemaError(f"report {key} inventory must not be empty")
        else:
            raise SchemaError(f"report {key} must be a positive count or inventory")


def normalize_report_summary(summary: Mapping[str, object]) -> ReportData:
    """Validate and normalize historical or role-declared report input exactly once."""
    source = _string(summary.get("source_gpu"), context="report source_gpu")
    target = _string(summary.get("target_gpu"), context="report target_gpu")
    _validate_scope(summary)
    methods = _normalize_methods(summary)
    raw_max_budget = summary.get("max_budget")
    if raw_max_budget is None:
        observed_max = max(point.budget for method in methods for point in method.points)
        if not observed_max.is_integer():
            raise SchemaError("inferred report max_budget must be an integer")
        max_budget = int(observed_max)
    else:
        max_budget = _positive_int(raw_max_budget, context="report max_budget")
    limitations_raw = summary.get("limitations", ())
    if not isinstance(limitations_raw, Sequence) or isinstance(limitations_raw, (str, bytes)):
        raise SchemaError("report limitations must be a sequence")
    limitations = tuple(
        _string(value, context=f"limitation {index}") for index, value in enumerate(limitations_raw)
    )
    source_cost = _mapping(summary.get("source_cost", {}), context="source_cost")
    target_cost = _mapping(
        summary.get("target_collection_cost", {}), context="target_collection_cost"
    )
    provenance = _mapping(summary.get("provenance", {}), context="provenance")
    primary = _normalize_primary(summary, methods)
    hardware = _normalize_hardware(summary)
    folds = _normalize_folds(summary)
    frozen = _freeze(summary)
    if not isinstance(frozen, Mapping):
        raise SchemaError("normalized raw summary must be a mapping")
    return ReportData(
        source_label=source,
        target_label=target,
        max_budget=max_budget,
        methods=methods,
        primary=primary,
        hardware=hardware,
        folds=folds,
        costs=ReportCosts(
            source=cast(ReadOnlyMapping, _freeze(source_cost)),
            target=cast(ReadOnlyMapping, _freeze(target_cost)),
        ),
        provenance=ReportProvenance(cast(ReadOnlyMapping, _freeze(provenance))),
        limitations=limitations,
        raw_summary=frozen,
    )


__all__ = [
    "MethodRole",
    "ReportCosts",
    "ReportCurvePoint",
    "ReportData",
    "ReportFold",
    "ReportHardware",
    "ReportMethod",
    "ReportPrimaryEvidence",
    "ReportProvenance",
    "ReportUncertainty",
    "normalize_report_summary",
]
