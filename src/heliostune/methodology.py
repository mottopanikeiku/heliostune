"""Strict methodology protocol, evidence-bundle, and closed-file v1 verification.

The filesystem verifier closes only the byte inventory and the lifecycle/count
claims it can establish locally.  Its typed result makes all deferred checks
explicit rather than implying publication eligibility.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from heliostune.artifacts import read_json, strict_json_dumps, strict_json_loads
from heliostune.errors import ArtifactError, SchemaError
from heliostune.validation import (
    exact_fields,
    exact_int,
    finite_float,
    nonblank_string,
)

EvidenceClass = Literal["exploratory", "engineering_gate", "confirmatory"]
ClaimKind = Literal[
    "descriptive",
    "superiority",
    "inferiority",
    "noninferiority",
    "equivalence",
    "scoped_exhaustive_dominance",
    "transfer_benefit",
]
Direction = Literal["higher", "lower"]
StoppingRule = Literal["none", "fixed_n", "confidence_sequence"]
RetryPolicy = Literal["none", "pre_measurement_infrastructure"]
LifecycleState = Literal[
    "DRAFT",
    "RESOLVED",
    "FROZEN",
    "DISPATCHED",
    "SEALED",
    "VERIFIED",
    "ANALYZED",
    "PUBLISHED",
]
Outcome = Literal["pending", "completed", "failed", "aborted"]
Attestation = Literal["none", "self_attested_backend", "provider_signed"]
OfflineReproduction = Literal["not_checked", "partial", "complete"]
ProtocolDigestRole = Literal[
    "plugin",
    "workloads",
    "candidates",
    "comparators",
    "splits",
    "numerics",
    "timing",
    "analyzer",
    "expected_cells",
    "environment_predicate",
    "failure_policy",
    "paid_plan",
    "parent_protocol",
]
CellIdentityRole = Literal["expected_cells", "terminal_cells"]
AttemptTransitionStatus = Literal["pending", "running", "success", "failure"]

PROTOCOL_DIGEST_ROLES: tuple[ProtocolDigestRole, ...] = (
    "plugin",
    "workloads",
    "candidates",
    "comparators",
    "splits",
    "numerics",
    "timing",
    "analyzer",
    "expected_cells",
    "environment_predicate",
    "failure_policy",
    "paid_plan",
    "parent_protocol",
)
CELL_IDENTITY_ROLES: tuple[CellIdentityRole, ...] = ("expected_cells", "terminal_cells")
_ATTEMPT_CHAIN_SCHEMA = "heliostune.attempt-chain/1"
_SELECTED_SUITE_SCHEMA = "heliostune.selected-suite/1"
_ATTEMPT_CHAIN_INITIAL_HEAD = hashlib.sha256(b"").hexdigest()


def plugin_suite_role(index: int) -> str:
    """Return the reserved artifact role for a plugin suite inventory index."""

    exact_int(index, context="plugin suite index", minimum=0)
    return f"plugin_suite_{index}"


def plugin_suite_path(index: int) -> str:
    """Return the required flat bundle path for a plugin suite inventory index."""

    return f"{plugin_suite_role(index)}.json"


def attempt_chain_descriptor_bytes() -> bytes:
    """Return the exact descriptor selecting chained attempt-journal parsing."""

    return strict_json_dumps({"schema": _ATTEMPT_CHAIN_SCHEMA}).encode("utf-8")


def selected_suite_descriptor_bytes(index: int) -> bytes:
    """Return the exact descriptor binding the selected suite inventory index."""

    exact_int(index, context="selected suite plugin_suite_index", minimum=0)
    return strict_json_dumps(
        {"schema": _SELECTED_SUITE_SCHEMA, "plugin_suite_index": index}
    ).encode("utf-8")


def _canonical_attempt_row(
    cell_id: object,
    status: object,
    predecessor_sha256: str,
    *,
    context: str,
) -> bytes:
    parsed_cell_id = nonblank_string(cell_id, context=f"{context} cell_id")
    parsed_status = _enum(
        status,
        allowed={"pending", "running", "success", "failure"},
        context=f"{context} status",
    )
    return (
        strict_json_dumps(
            {
                "cell_id": parsed_cell_id,
                "predecessor_sha256": predecessor_sha256,
                "status": parsed_status,
            },
            compact=True,
        ).encode("utf-8")
        + b"\n"
    )


def encode_attempt_journal(
    transitions: Iterable[Mapping[str, object]],
) -> tuple[bytes, str]:
    """Encode transitions into canonical predecessor-linked JSONL bytes."""

    rows: list[bytes] = []
    head = _ATTEMPT_CHAIN_INITIAL_HEAD
    for index, transition in enumerate(transitions, start=1):
        data = exact_fields(
            transition,
            required=("cell_id", "status"),
            context=f"attempt transition {index}",
        )
        row = _canonical_attempt_row(
            data["cell_id"],
            data["status"],
            head,
            context=f"attempt transition {index}",
        )
        rows.append(row)
        head = _sha256(row)
    return b"".join(rows), head


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})(?P<fraction>\.\d+)?"
    r"(?P<offset>Z|[+-]\d{2}:\d{2})\Z"
)


def _enum(value: object, *, allowed: Collection[str], context: str) -> str:
    result = nonblank_string(value, context=context)
    if result not in allowed:
        raise SchemaError(f"unknown {context} {result!r}")
    return result


def _optional_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return nonblank_string(value, context=context)


def _digest(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if _DIGEST_RE.fullmatch(result) is None:
        raise SchemaError(f"{context} must be a lowercase 64-hex SHA-256 digest")
    return result


def _optional_digest(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _digest(value, context=context)


def _timestamp(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    match = _RFC3339_RE.fullmatch(result)
    if match is None:
        raise SchemaError(f"{context} must be an RFC3339 timestamp")
    iso_value = result[:-1] + "+00:00" if result.endswith("Z") else result
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise SchemaError(f"{context} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaError(f"{context} must be an RFC3339 timestamp")
    return result


def _relative_path(value: object, *, context: str) -> str:
    result = nonblank_string(value, context=context)
    if "\\" in result or "\x00" in result:
        raise SchemaError(f"{context} must be a normalized POSIX relative path")
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or result != path.as_posix()
        or any(part in {"", ".", ".."} for part in result.split("/"))
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise SchemaError(f"{context} must be normalized, relative, and non-escaping")
    return result


def _object_array(value: object, *, context: str) -> list[object]:
    if type(value) is not list:
        raise SchemaError(f"{context} must be an array")
    return cast(list[object], value)


def _optional_number(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float | None:
    if value is None:
        return None
    return finite_float(
        value,
        context=context,
        minimum=minimum,
        strictly_positive=strictly_positive,
    )


def _require_exact_instance(value: object, expected: type[object], *, context: str) -> None:
    if type(value) is not expected:
        raise SchemaError(f"{context} must be a {expected.__name__}")


_CLAIM_FIELDS = (
    "claim_id",
    "kind",
    "candidate_id",
    "comparator_id",
    "reference_id",
    "estimand_ast_sha256",
    "units",
    "direction",
    "scope_sha256",
    "population_sha256",
    "delta",
    "alpha",
    "multiplicity_family",
    "stopping",
)
_CLAIM_KINDS = {
    "descriptive",
    "superiority",
    "inferiority",
    "noninferiority",
    "equivalence",
    "scoped_exhaustive_dominance",
    "transfer_benefit",
}
_DIRECTIONS = {"higher", "lower"}
_STOPPING_RULES = {"none", "fixed_n", "confidence_sequence"}


@dataclass(frozen=True, slots=True)
class ClaimSpec:
    claim_id: str
    kind: ClaimKind
    candidate_id: str
    comparator_id: str
    reference_id: str | None
    estimand_ast_sha256: str
    units: str
    direction: Direction
    scope_sha256: str
    population_sha256: str
    delta: float | None
    alpha: float | None
    multiplicity_family: str | None
    stopping: StoppingRule

    def __post_init__(self) -> None:
        nonblank_string(self.claim_id, context="claim claim_id")
        _enum(self.kind, allowed=_CLAIM_KINDS, context="claim kind")
        candidate = nonblank_string(self.candidate_id, context="claim candidate_id")
        comparator = nonblank_string(self.comparator_id, context="claim comparator_id")
        reference = _optional_string(self.reference_id, context="claim reference_id")
        if candidate == comparator:
            raise SchemaError("claim candidate_id and comparator_id must differ")
        if reference is not None and reference in {candidate, comparator}:
            raise SchemaError("claim reference_id must identify a distinct role")
        _digest(self.estimand_ast_sha256, context="claim estimand_ast_sha256")
        nonblank_string(self.units, context="claim units")
        _enum(self.direction, allowed=_DIRECTIONS, context="claim direction")
        _digest(self.scope_sha256, context="claim scope_sha256")
        _digest(self.population_sha256, context="claim population_sha256")
        delta = _optional_number(self.delta, context="claim delta", minimum=0)
        alpha = _optional_number(self.alpha, context="claim alpha", strictly_positive=True)
        family = _optional_string(self.multiplicity_family, context="claim multiplicity_family")
        stopping = _enum(self.stopping, allowed=_STOPPING_RULES, context="claim stopping")

        if alpha is not None and alpha >= 0.5:
            raise SchemaError("claim alpha must be less than 0.5")
        if self.kind == "descriptive":
            if delta is not None or alpha is not None or family is not None:
                raise SchemaError(
                    "descriptive claim must have null delta, alpha, and multiplicity_family"
                )
            if stopping != "none":
                raise SchemaError("descriptive claim stopping must be 'none'")
        else:
            if delta is None:
                raise SchemaError(f"{self.kind} claim requires delta")
            if self.kind in {"noninferiority", "equivalence"} and delta <= 0:
                raise SchemaError(f"{self.kind} claim requires a positive delta")
            if alpha is None:
                raise SchemaError(f"{self.kind} claim requires alpha")
            if family is None:
                raise SchemaError(f"{self.kind} claim requires multiplicity_family")
            if stopping == "none":
                raise SchemaError(f"{self.kind} claim requires a frozen stopping rule")
        if self.kind == "scoped_exhaustive_dominance" and reference is None:
            raise SchemaError("scoped_exhaustive_dominance claim requires reference_id")

    @classmethod
    def from_dict(cls, value: object) -> ClaimSpec:
        data = exact_fields(value, required=_CLAIM_FIELDS, context="claim")
        return cls(
            claim_id=nonblank_string(data["claim_id"], context="claim claim_id"),
            kind=cast(
                ClaimKind,
                _enum(data["kind"], allowed=_CLAIM_KINDS, context="claim kind"),
            ),
            candidate_id=nonblank_string(data["candidate_id"], context="claim candidate_id"),
            comparator_id=nonblank_string(data["comparator_id"], context="claim comparator_id"),
            reference_id=_optional_string(data["reference_id"], context="claim reference_id"),
            estimand_ast_sha256=_digest(
                data["estimand_ast_sha256"], context="claim estimand_ast_sha256"
            ),
            units=nonblank_string(data["units"], context="claim units"),
            direction=cast(
                Direction,
                _enum(data["direction"], allowed=_DIRECTIONS, context="claim direction"),
            ),
            scope_sha256=_digest(data["scope_sha256"], context="claim scope_sha256"),
            population_sha256=_digest(data["population_sha256"], context="claim population_sha256"),
            delta=_optional_number(data["delta"], context="claim delta", minimum=0),
            alpha=_optional_number(data["alpha"], context="claim alpha", strictly_positive=True),
            multiplicity_family=_optional_string(
                data["multiplicity_family"], context="claim multiplicity_family"
            ),
            stopping=cast(
                StoppingRule,
                _enum(
                    data["stopping"],
                    allowed=_STOPPING_RULES,
                    context="claim stopping",
                ),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "comparator_id": self.comparator_id,
            "reference_id": self.reference_id,
            "estimand_ast_sha256": self.estimand_ast_sha256,
            "units": self.units,
            "direction": self.direction,
            "scope_sha256": self.scope_sha256,
            "population_sha256": self.population_sha256,
            "delta": self.delta,
            "alpha": self.alpha,
            "multiplicity_family": self.multiplicity_family,
            "stopping": self.stopping,
        }


_PLUGIN_FIELDS = ("id", "version", "artifact_sha256")


@dataclass(frozen=True, slots=True)
class Plugin:
    id: str
    version: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        nonblank_string(self.id, context="plugin id")
        nonblank_string(self.version, context="plugin version")
        _digest(self.artifact_sha256, context="plugin artifact_sha256")

    @classmethod
    def from_dict(cls, value: object) -> Plugin:
        data = exact_fields(value, required=_PLUGIN_FIELDS, context="protocol plugin")
        return cls(
            id=nonblank_string(data["id"], context="plugin id"),
            version=nonblank_string(data["version"], context="plugin version"),
            artifact_sha256=_digest(data["artifact_sha256"], context="plugin artifact_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "version": self.version, "artifact_sha256": self.artifact_sha256}


_SEMANTIC_FIELDS = (
    "workloads_sha256",
    "candidates_sha256",
    "comparators_sha256",
    "splits_sha256",
    "numerics_sha256",
    "timing_sha256",
)


@dataclass(frozen=True, slots=True)
class Semantic:
    workloads_sha256: str
    candidates_sha256: str
    comparators_sha256: str
    splits_sha256: str
    numerics_sha256: str
    timing_sha256: str

    def __post_init__(self) -> None:
        for name in _SEMANTIC_FIELDS:
            _digest(getattr(self, name), context=f"semantic {name}")

    @classmethod
    def from_dict(cls, value: object) -> Semantic:
        data = exact_fields(value, required=_SEMANTIC_FIELDS, context="protocol semantic")
        return cls(
            workloads_sha256=_digest(data["workloads_sha256"], context="semantic workloads_sha256"),
            candidates_sha256=_digest(
                data["candidates_sha256"], context="semantic candidates_sha256"
            ),
            comparators_sha256=_digest(
                data["comparators_sha256"], context="semantic comparators_sha256"
            ),
            splits_sha256=_digest(data["splits_sha256"], context="semantic splits_sha256"),
            numerics_sha256=_digest(data["numerics_sha256"], context="semantic numerics_sha256"),
            timing_sha256=_digest(data["timing_sha256"], context="semantic timing_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _SEMANTIC_FIELDS}


_ANALYSIS_FIELDS = ("analyzer_sha256", "claims")


@dataclass(frozen=True, slots=True)
class Analysis:
    analyzer_sha256: str
    claims: tuple[ClaimSpec, ...]

    def __post_init__(self) -> None:
        _digest(self.analyzer_sha256, context="analysis analyzer_sha256")
        if type(self.claims) is not tuple:
            raise SchemaError("analysis claims must be a tuple")
        for claim in self.claims:
            _require_exact_instance(claim, ClaimSpec, context="analysis claim")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise SchemaError("analysis claim_id values must be unique")

    @classmethod
    def from_dict(cls, value: object) -> Analysis:
        data = exact_fields(value, required=_ANALYSIS_FIELDS, context="protocol analysis")
        claims = tuple(
            ClaimSpec.from_dict(item)
            for item in _object_array(data["claims"], context="analysis claims")
        )
        return cls(
            analyzer_sha256=_digest(data["analyzer_sha256"], context="analysis analyzer_sha256"),
            claims=claims,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "analyzer_sha256": self.analyzer_sha256,
            "claims": [claim.to_dict() for claim in self.claims],
        }


_EXECUTION_FIELDS = (
    "executor_api",
    "expected_cells_sha256",
    "expected_cell_count",
    "environment_predicate_sha256",
    "failure_policy_sha256",
    "retry_policy",
    "max_physical_attempts",
    "wall_limit_s",
    "paid_plan_sha256",
)
_RETRY_POLICIES = {"none", "pre_measurement_infrastructure"}


@dataclass(frozen=True, slots=True)
class Execution:
    executor_api: str
    expected_cells_sha256: str
    expected_cell_count: int
    environment_predicate_sha256: str
    failure_policy_sha256: str
    retry_policy: RetryPolicy
    max_physical_attempts: int
    wall_limit_s: int
    paid_plan_sha256: str | None

    def __post_init__(self) -> None:
        nonblank_string(self.executor_api, context="execution executor_api")
        _digest(self.expected_cells_sha256, context="execution expected_cells_sha256")
        exact_int(
            self.expected_cell_count,
            context="execution expected_cell_count",
            minimum=0,
        )
        _digest(
            self.environment_predicate_sha256,
            context="execution environment_predicate_sha256",
        )
        _digest(self.failure_policy_sha256, context="execution failure_policy_sha256")
        retry = _enum(
            self.retry_policy,
            allowed=_RETRY_POLICIES,
            context="execution retry_policy",
        )
        attempts = exact_int(
            self.max_physical_attempts,
            context="execution max_physical_attempts",
            minimum=1,
        )
        exact_int(self.wall_limit_s, context="execution wall_limit_s", minimum=1)
        _optional_digest(self.paid_plan_sha256, context="execution paid_plan_sha256")
        if retry == "none" and attempts != 1:
            raise SchemaError("retry_policy 'none' requires max_physical_attempts equal to 1")
        if retry == "pre_measurement_infrastructure" and attempts < 2:
            raise SchemaError(
                "pre_measurement_infrastructure retry requires max_physical_attempts at least 2"
            )

    @classmethod
    def from_dict(cls, value: object) -> Execution:
        data = exact_fields(value, required=_EXECUTION_FIELDS, context="protocol execution")
        return cls(
            executor_api=nonblank_string(data["executor_api"], context="execution executor_api"),
            expected_cells_sha256=_digest(
                data["expected_cells_sha256"], context="execution expected_cells_sha256"
            ),
            expected_cell_count=exact_int(
                data["expected_cell_count"],
                context="execution expected_cell_count",
                minimum=0,
            ),
            environment_predicate_sha256=_digest(
                data["environment_predicate_sha256"],
                context="execution environment_predicate_sha256",
            ),
            failure_policy_sha256=_digest(
                data["failure_policy_sha256"], context="execution failure_policy_sha256"
            ),
            retry_policy=cast(
                RetryPolicy,
                _enum(
                    data["retry_policy"],
                    allowed=_RETRY_POLICIES,
                    context="execution retry_policy",
                ),
            ),
            max_physical_attempts=exact_int(
                data["max_physical_attempts"],
                context="execution max_physical_attempts",
                minimum=1,
            ),
            wall_limit_s=exact_int(
                data["wall_limit_s"], context="execution wall_limit_s", minimum=1
            ),
            paid_plan_sha256=_optional_digest(
                data["paid_plan_sha256"], context="execution paid_plan_sha256"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "executor_api": self.executor_api,
            "expected_cells_sha256": self.expected_cells_sha256,
            "expected_cell_count": self.expected_cell_count,
            "environment_predicate_sha256": self.environment_predicate_sha256,
            "failure_policy_sha256": self.failure_policy_sha256,
            "retry_policy": self.retry_policy,
            "max_physical_attempts": self.max_physical_attempts,
            "wall_limit_s": self.wall_limit_s,
            "paid_plan_sha256": self.paid_plan_sha256,
        }


_PROTOCOL_FIELDS = (
    "schema",
    "study_id",
    "revision",
    "created_at",
    "evidence_class",
    "parent_protocol_sha256",
    "plugin",
    "semantic",
    "analysis",
    "execution",
)
_EVIDENCE_CLASSES = {"exploratory", "engineering_gate", "confirmatory"}


@dataclass(frozen=True, slots=True)
class ProtocolV1:
    schema: Literal["heliostune.protocol/1"]
    study_id: str
    revision: int
    created_at: str
    evidence_class: EvidenceClass
    parent_protocol_sha256: str | None
    plugin: Plugin
    semantic: Semantic
    analysis: Analysis
    execution: Execution

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "heliostune.protocol/1":
            raise SchemaError("protocol schema must be 'heliostune.protocol/1'")
        nonblank_string(self.study_id, context="protocol study_id")
        exact_int(self.revision, context="protocol revision", minimum=1)
        _timestamp(self.created_at, context="protocol created_at")
        evidence_class = _enum(
            self.evidence_class,
            allowed=_EVIDENCE_CLASSES,
            context="protocol evidence_class",
        )
        _optional_digest(
            self.parent_protocol_sha256,
            context="protocol parent_protocol_sha256",
        )
        _require_exact_instance(self.plugin, Plugin, context="protocol plugin")
        _require_exact_instance(self.semantic, Semantic, context="protocol semantic")
        _require_exact_instance(self.analysis, Analysis, context="protocol analysis")
        _require_exact_instance(self.execution, Execution, context="protocol execution")
        if evidence_class in {"exploratory", "engineering_gate"}:
            inferential = [
                claim.claim_id for claim in self.analysis.claims if claim.kind != "descriptive"
            ]
            if inferential:
                raise SchemaError(
                    f"{evidence_class} protocol cannot contain inferential claims {inferential!r}"
                )
        if (
            evidence_class in {"engineering_gate", "confirmatory"}
            and self.execution.expected_cell_count == 0
        ):
            raise SchemaError(f"{evidence_class} protocol requires at least one expected cell")

    @classmethod
    def from_dict(cls, value: object) -> ProtocolV1:
        data = exact_fields(value, required=_PROTOCOL_FIELDS, context="methodology protocol")
        schema = nonblank_string(data["schema"], context="protocol schema")
        if schema != "heliostune.protocol/1":
            raise SchemaError("protocol schema must be 'heliostune.protocol/1'")
        return cls(
            schema="heliostune.protocol/1",
            study_id=nonblank_string(data["study_id"], context="protocol study_id"),
            revision=exact_int(data["revision"], context="protocol revision", minimum=1),
            created_at=_timestamp(data["created_at"], context="protocol created_at"),
            evidence_class=cast(
                EvidenceClass,
                _enum(
                    data["evidence_class"],
                    allowed=_EVIDENCE_CLASSES,
                    context="protocol evidence_class",
                ),
            ),
            parent_protocol_sha256=_optional_digest(
                data["parent_protocol_sha256"],
                context="protocol parent_protocol_sha256",
            ),
            plugin=Plugin.from_dict(data["plugin"]),
            semantic=Semantic.from_dict(data["semantic"]),
            analysis=Analysis.from_dict(data["analysis"]),
            execution=Execution.from_dict(data["execution"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "study_id": self.study_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "evidence_class": self.evidence_class,
            "parent_protocol_sha256": self.parent_protocol_sha256,
            "plugin": self.plugin.to_dict(),
            "semantic": self.semantic.to_dict(),
            "analysis": self.analysis.to_dict(),
            "execution": self.execution.to_dict(),
        }


_BINDING_FIELDS = ("path", "sha256", "bytes")


@dataclass(frozen=True, slots=True)
class ProtocolBinding:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _relative_path(self.path, context="bundle protocol path")
        _digest(self.sha256, context="bundle protocol sha256")
        exact_int(self.bytes, context="bundle protocol bytes", minimum=1)

    @classmethod
    def from_dict(cls, value: object) -> ProtocolBinding:
        data = exact_fields(value, required=_BINDING_FIELDS, context="bundle protocol")
        return cls(
            path=_relative_path(data["path"], context="bundle protocol path"),
            sha256=_digest(data["sha256"], context="bundle protocol sha256"),
            bytes=exact_int(data["bytes"], context="bundle protocol bytes", minimum=1),
        )

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


_LIFECYCLE_FIELDS = ("state", "outcome")
_LIFECYCLE_STATES = {
    "DRAFT",
    "RESOLVED",
    "FROZEN",
    "DISPATCHED",
    "SEALED",
    "VERIFIED",
    "ANALYZED",
    "PUBLISHED",
}
_OUTCOMES = {"pending", "completed", "failed", "aborted"}


@dataclass(frozen=True, slots=True)
class Lifecycle:
    state: LifecycleState
    outcome: Outcome

    def __post_init__(self) -> None:
        state = _enum(self.state, allowed=_LIFECYCLE_STATES, context="lifecycle state")
        outcome = _enum(self.outcome, allowed=_OUTCOMES, context="lifecycle outcome")
        preterminal_states = {"DRAFT", "RESOLVED", "FROZEN", "DISPATCHED"}
        if state in preterminal_states and outcome != "pending":
            raise SchemaError(f"lifecycle state {state!r} requires outcome 'pending'")
        if state not in preterminal_states and outcome == "pending":
            raise SchemaError(f"lifecycle state {state!r} requires a terminal outcome")

    @classmethod
    def from_dict(cls, value: object) -> Lifecycle:
        data = exact_fields(value, required=_LIFECYCLE_FIELDS, context="bundle lifecycle")
        return cls(
            state=cast(
                LifecycleState,
                _enum(data["state"], allowed=_LIFECYCLE_STATES, context="lifecycle state"),
            ),
            outcome=cast(
                Outcome,
                _enum(data["outcome"], allowed=_OUTCOMES, context="lifecycle outcome"),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {"state": self.state, "outcome": self.outcome}


_ATTEMPTS_FIELDS = (
    "path",
    "sha256",
    "hash_chain_head",
    "logical",
    "physical",
    "terminal",
    "orphaned",
)


@dataclass(frozen=True, slots=True)
class AttemptsSummary:
    path: str
    sha256: str
    hash_chain_head: str
    logical: int
    physical: int
    terminal: int
    orphaned: int

    def __post_init__(self) -> None:
        _relative_path(self.path, context="bundle attempts path")
        _digest(self.sha256, context="bundle attempts sha256")
        _digest(self.hash_chain_head, context="bundle attempts hash_chain_head")
        logical = exact_int(self.logical, context="bundle attempts logical", minimum=0)
        physical = exact_int(self.physical, context="bundle attempts physical", minimum=0)
        terminal = exact_int(self.terminal, context="bundle attempts terminal", minimum=0)
        orphaned = exact_int(self.orphaned, context="bundle attempts orphaned", minimum=0)
        if physical < logical:
            raise SchemaError("bundle physical attempts cannot be fewer than logical attempts")
        if terminal > logical:
            raise SchemaError("bundle terminal attempts cannot exceed logical attempts")
        if terminal + orphaned > physical:
            raise SchemaError(
                "bundle terminal and orphaned attempts cannot exceed physical attempts"
            )

    @classmethod
    def from_dict(cls, value: object) -> AttemptsSummary:
        data = exact_fields(value, required=_ATTEMPTS_FIELDS, context="bundle attempts")
        return cls(
            path=_relative_path(data["path"], context="bundle attempts path"),
            sha256=_digest(data["sha256"], context="bundle attempts sha256"),
            hash_chain_head=_digest(
                data["hash_chain_head"], context="bundle attempts hash_chain_head"
            ),
            logical=exact_int(data["logical"], context="bundle attempts logical", minimum=0),
            physical=exact_int(data["physical"], context="bundle attempts physical", minimum=0),
            terminal=exact_int(data["terminal"], context="bundle attempts terminal", minimum=0),
            orphaned=exact_int(data["orphaned"], context="bundle attempts orphaned", minimum=0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "hash_chain_head": self.hash_chain_head,
            "logical": self.logical,
            "physical": self.physical,
            "terminal": self.terminal,
            "orphaned": self.orphaned,
        }


_COVERAGE_FIELDS = ("expected_cells", "terminal_cells", "successes", "failures")


@dataclass(frozen=True, slots=True)
class Coverage:
    expected_cells: int
    terminal_cells: int
    successes: int
    failures: int

    def __post_init__(self) -> None:
        expected = exact_int(
            self.expected_cells, context="bundle coverage expected_cells", minimum=0
        )
        terminal = exact_int(
            self.terminal_cells, context="bundle coverage terminal_cells", minimum=0
        )
        successes = exact_int(self.successes, context="bundle coverage successes", minimum=0)
        failures = exact_int(self.failures, context="bundle coverage failures", minimum=0)
        if terminal > expected:
            raise SchemaError("bundle terminal cells cannot exceed expected cells")
        if successes + failures != terminal:
            raise SchemaError("bundle coverage successes plus failures must equal terminal_cells")

    @classmethod
    def from_dict(cls, value: object) -> Coverage:
        data = exact_fields(value, required=_COVERAGE_FIELDS, context="bundle coverage")
        return cls(
            expected_cells=exact_int(
                data["expected_cells"], context="bundle coverage expected_cells", minimum=0
            ),
            terminal_cells=exact_int(
                data["terminal_cells"], context="bundle coverage terminal_cells", minimum=0
            ),
            successes=exact_int(data["successes"], context="bundle coverage successes", minimum=0),
            failures=exact_int(data["failures"], context="bundle coverage failures", minimum=0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_cells": self.expected_cells,
            "terminal_cells": self.terminal_cells,
            "successes": self.successes,
            "failures": self.failures,
        }


_ARTIFACT_FIELDS = ("role", "path", "media_type", "bytes", "sha256")


@dataclass(frozen=True, slots=True)
class Artifact:
    role: str
    path: str
    media_type: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        nonblank_string(self.role, context="bundle artifact role")
        _relative_path(self.path, context="bundle artifact path")
        nonblank_string(self.media_type, context="bundle artifact media_type")
        exact_int(self.bytes, context="bundle artifact bytes", minimum=0)
        _digest(self.sha256, context="bundle artifact sha256")

    @classmethod
    def from_dict(cls, value: object) -> Artifact:
        data = exact_fields(value, required=_ARTIFACT_FIELDS, context="bundle artifact")
        return cls(
            role=nonblank_string(data["role"], context="bundle artifact role"),
            path=_relative_path(data["path"], context="bundle artifact path"),
            media_type=nonblank_string(data["media_type"], context="bundle artifact media_type"),
            bytes=exact_int(data["bytes"], context="bundle artifact bytes", minimum=0),
            sha256=_digest(data["sha256"], context="bundle artifact sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "media_type": self.media_type,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


_PROVENANCE_FIELDS = ("attestation", "offline_reproduction")
_ATTESTATIONS = {"none", "self_attested_backend", "provider_signed"}
_OFFLINE_REPRODUCTIONS = {"not_checked", "partial", "complete"}


@dataclass(frozen=True, slots=True)
class Provenance:
    attestation: Attestation
    offline_reproduction: OfflineReproduction

    def __post_init__(self) -> None:
        _enum(self.attestation, allowed=_ATTESTATIONS, context="provenance attestation")
        _enum(
            self.offline_reproduction,
            allowed=_OFFLINE_REPRODUCTIONS,
            context="provenance offline_reproduction",
        )

    @classmethod
    def from_dict(cls, value: object) -> Provenance:
        data = exact_fields(value, required=_PROVENANCE_FIELDS, context="bundle provenance")
        return cls(
            attestation=cast(
                Attestation,
                _enum(
                    data["attestation"],
                    allowed=_ATTESTATIONS,
                    context="provenance attestation",
                ),
            ),
            offline_reproduction=cast(
                OfflineReproduction,
                _enum(
                    data["offline_reproduction"],
                    allowed=_OFFLINE_REPRODUCTIONS,
                    context="provenance offline_reproduction",
                ),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attestation": self.attestation,
            "offline_reproduction": self.offline_reproduction,
        }


_SIGNATURE_FIELDS = ("scheme", "signer", "subject_sha256", "signature")


@dataclass(frozen=True, slots=True)
class Signature:
    scheme: str
    signer: str
    subject_sha256: str
    signature: str

    def __post_init__(self) -> None:
        nonblank_string(self.scheme, context="bundle signature scheme")
        nonblank_string(self.signer, context="bundle signature signer")
        _digest(self.subject_sha256, context="bundle signature subject_sha256")
        nonblank_string(self.signature, context="bundle signature signature")

    @classmethod
    def from_dict(cls, value: object) -> Signature:
        data = exact_fields(value, required=_SIGNATURE_FIELDS, context="bundle signature")
        return cls(
            scheme=nonblank_string(data["scheme"], context="bundle signature scheme"),
            signer=nonblank_string(data["signer"], context="bundle signature signer"),
            subject_sha256=_digest(
                data["subject_sha256"], context="bundle signature subject_sha256"
            ),
            signature=nonblank_string(data["signature"], context="bundle signature signature"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "signer": self.signer,
            "subject_sha256": self.subject_sha256,
            "signature": self.signature,
        }


_BUNDLE_FIELDS = (
    "schema",
    "bundle_id",
    "created_at",
    "protocol",
    "lifecycle",
    "attempts",
    "coverage",
    "artifacts",
    "provenance",
    "signatures",
)


@dataclass(frozen=True, slots=True)
class EvidenceBundleV1:
    schema: Literal["heliostune.bundle/1"]
    bundle_id: str
    created_at: str
    protocol: ProtocolBinding
    lifecycle: Lifecycle
    attempts: AttemptsSummary
    coverage: Coverage
    artifacts: tuple[Artifact, ...]
    provenance: Provenance
    signatures: tuple[Signature, ...]

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != "heliostune.bundle/1":
            raise SchemaError("bundle schema must be 'heliostune.bundle/1'")
        nonblank_string(self.bundle_id, context="bundle bundle_id")
        _timestamp(self.created_at, context="bundle created_at")
        _require_exact_instance(self.protocol, ProtocolBinding, context="bundle protocol")
        _require_exact_instance(self.lifecycle, Lifecycle, context="bundle lifecycle")
        _require_exact_instance(self.attempts, AttemptsSummary, context="bundle attempts")
        _require_exact_instance(self.coverage, Coverage, context="bundle coverage")
        if type(self.artifacts) is not tuple:
            raise SchemaError("bundle artifacts must be a tuple")
        if type(self.signatures) is not tuple:
            raise SchemaError("bundle signatures must be a tuple")
        for artifact in self.artifacts:
            _require_exact_instance(artifact, Artifact, context="bundle artifact")
        for signature in self.signatures:
            _require_exact_instance(signature, Signature, context="bundle signature")
        _require_exact_instance(self.provenance, Provenance, context="bundle provenance")

        paths = [self.protocol.path, self.attempts.path]
        paths.extend(artifact.path for artifact in self.artifacts)
        if len(paths) != len(set(paths)):
            raise SchemaError("bundle paths must be unique")
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise SchemaError("bundle artifact roles must be unique")
        if len(self.signatures) != len(set(self.signatures)):
            raise SchemaError("bundle signatures must be unique")

    @classmethod
    def from_dict(cls, value: object) -> EvidenceBundleV1:
        data = exact_fields(value, required=_BUNDLE_FIELDS, context="evidence bundle")
        schema = nonblank_string(data["schema"], context="bundle schema")
        if schema != "heliostune.bundle/1":
            raise SchemaError("bundle schema must be 'heliostune.bundle/1'")
        return cls(
            schema="heliostune.bundle/1",
            bundle_id=nonblank_string(data["bundle_id"], context="bundle bundle_id"),
            created_at=_timestamp(data["created_at"], context="bundle created_at"),
            protocol=ProtocolBinding.from_dict(data["protocol"]),
            lifecycle=Lifecycle.from_dict(data["lifecycle"]),
            attempts=AttemptsSummary.from_dict(data["attempts"]),
            coverage=Coverage.from_dict(data["coverage"]),
            artifacts=tuple(
                Artifact.from_dict(item)
                for item in _object_array(data["artifacts"], context="bundle artifacts")
            ),
            provenance=Provenance.from_dict(data["provenance"]),
            signatures=tuple(
                Signature.from_dict(item)
                for item in _object_array(data["signatures"], context="bundle signatures")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "protocol": self.protocol.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "attempts": self.attempts.to_dict(),
            "coverage": self.coverage.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "provenance": self.provenance.to_dict(),
            "signatures": [signature.to_dict() for signature in self.signatures],
        }


def load_protocol_v1(path: str | Path) -> ProtocolV1:
    """Read and strictly parse one ``heliostune.protocol/1`` document."""

    return ProtocolV1.from_dict(read_json(path))


def load_bundle_v1(path: str | Path) -> EvidenceBundleV1:
    """Read and strictly parse one ``heliostune.bundle/1`` root document."""

    return EvidenceBundleV1.from_dict(read_json(path))


@dataclass(frozen=True, slots=True)
class VerifiedProtocol:
    """A strict protocol together with the identity of its exact file bytes."""

    protocol: ProtocolV1
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class VerificationLimitations:
    """Per-control disclosure; ``checked`` means this verifier performed it."""

    protocol_ancestry: Literal["not_checked"] = "not_checked"
    evidence_nonpromotion: Literal["not_checked"] = "not_checked"
    semantic_content_beyond_digests: Literal["not_checked"] = "not_checked"
    plugin_suite_custody: Literal["not_checked", "checked"] = "not_checked"
    attempt_journal_hash_chain: Literal["not_checked", "checked"] = "not_checked"
    attempt_reconciliation: Literal["not_checked", "checked"] = "not_checked"
    claim_eligibility: Literal["not_checked"] = "not_checked"
    analyzer_replay: Literal["not_checked"] = "not_checked"
    provenance_tier_derivation: Literal["not_checked"] = "not_checked"
    signature_cryptography: Literal["not_checked"] = "not_checked"
    catalog_membership: Literal["not_checked"] = "not_checked"
    offline_reproduction: Literal["not_checked"] = "not_checked"


@dataclass(frozen=True, slots=True)
class VerifiedBundle:
    """A bundle whose local referenced-byte inventory is closed.

    This is intentionally not a publication-eligibility result.  Callers must
    inspect ``limitations`` and must not infer the deferred controls from the
    bundle's declared lifecycle or provenance tiers.
    """

    bundle: EvidenceBundleV1
    protocol: VerifiedProtocol
    root_path: Path
    root_sha256: str
    root_bytes: int
    referenced_paths: tuple[Path, ...]
    limitations: VerificationLimitations

    @property
    def publication_eligible(self) -> bool:
        """Whether every mandatory publication control was checked."""

        controls = (
            self.limitations.protocol_ancestry,
            self.limitations.evidence_nonpromotion,
            self.limitations.semantic_content_beyond_digests,
            self.limitations.plugin_suite_custody,
            self.limitations.attempt_journal_hash_chain,
            self.limitations.attempt_reconciliation,
            self.limitations.claim_eligibility,
            self.limitations.analyzer_replay,
            self.limitations.provenance_tier_derivation,
            self.limitations.signature_cryptography,
            self.limitations.catalog_membership,
            self.limitations.offline_reproduction,
        )
        return all(control == "checked" for control in controls)


def _read_verified_file(path: Path, *, context: str) -> bytes:
    if not path.is_file():
        raise ArtifactError(f"{context} is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read {context} {path}: {exc}") from exc


def _resolve_file(path: str | Path, *, context: str) -> Path:
    source = Path(path)
    try:
        directory = source.parent.resolve(strict=True)
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise ArtifactError(f"cannot resolve {context} {source}: {exc}") from exc
    try:
        resolved.relative_to(directory)
    except ValueError as exc:
        raise ArtifactError(f"{context} path escapes its directory: {source}") from exc
    if not resolved.is_file():
        raise ArtifactError(f"{context} is not a regular file: {resolved}")
    return resolved


class _BundleDirectoryReader:
    """Read one bundle tree through a pinned directory descriptor."""

    @staticmethod
    def _require_support() -> None:
        if (
            not hasattr(os, "O_DIRECTORY")
            or not hasattr(os, "O_NOFOLLOW")
            or os.open not in os.supports_dir_fd
        ):
            raise ArtifactError("descriptor-pinned bundle verification is unsupported")

    def __init__(self, root_manifest_path: str | Path) -> None:
        self._require_support()
        source = Path(root_manifest_path)
        self.root_relative_path = _relative_path(source.name, context="bundle root path")
        try:
            self.directory = source.parent.resolve(strict=True)
            self._directory_fd = os.open(
                self.directory,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise ArtifactError(f"cannot open bundle directory for {source}: {exc}") from exc
        self._occupied: set[tuple[int, int]] = set()

    @classmethod
    def from_directory_fd(
        cls,
        directory_fd: int,
        root_relative_path: str,
        *,
        diagnostic_directory: str | Path | None,
    ) -> _BundleDirectoryReader:
        cls._require_support()
        root = _relative_path(root_relative_path, context="bundle root path")
        directory = (
            Path(diagnostic_directory)
            if diagnostic_directory is not None
            else Path(f"<bundle-directory-fd-{directory_fd}>")
        )
        try:
            pinned_fd = os.dup(directory_fd)
        except OSError as exc:
            raise ArtifactError(f"cannot duplicate bundle directory descriptor: {exc}") from exc
        try:
            identity = os.fstat(pinned_fd)
            if not stat.S_ISDIR(identity.st_mode):
                raise ArtifactError("bundle directory descriptor does not refer to a directory")
        except (ArtifactError, OSError) as exc:
            os.close(pinned_fd)
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"cannot inspect bundle directory descriptor: {exc}") from exc
        reader = cls.__new__(cls)
        reader.root_relative_path = root
        reader.directory = directory
        reader._directory_fd = pinned_fd
        reader._occupied = set()
        return reader

    def close(self) -> None:
        os.close(self._directory_fd)

    def read(self, relative_path: str, *, context: str) -> tuple[Path, bytes]:
        normalized = _relative_path(relative_path, context=f"{context} path")
        parts = normalized.split("/")
        current_fd = self._directory_fd
        opened_directories: list[int] = []
        file_fd: int | None = None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        try:
            for component in parts[:-1]:
                current_fd = os.open(
                    component,
                    flags | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=current_fd,
                )
                opened_directories.append(current_fd)
            file_fd = os.open(
                parts[-1],
                flags | getattr(os, "O_NONBLOCK", 0),
                dir_fd=current_fd,
            )
            identity = os.fstat(file_fd)
            if not stat.S_ISREG(identity.st_mode):
                raise ArtifactError(f"{context} is not a regular file: {normalized!r}")
            key = (identity.st_dev, identity.st_ino)
            if key in self._occupied:
                raise ArtifactError(
                    f"{context} uses a file identity already used by the closed bundle: "
                    f"{normalized!r}"
                )
            chunks: list[bytes] = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            self._occupied.add(key)
            return self.directory / normalized, b"".join(chunks)
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError(
                f"cannot open {context} {normalized!r}; symlinks are forbidden and a path that "
                "escapes the bundle directory is rejected: "
                f"{exc}"
            ) from exc
        finally:
            if file_fd is not None:
                os.close(file_fd)
            for descriptor in reversed(opened_directories):
                os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_file_identity(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    context: str,
) -> None:
    actual_bytes = len(payload)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ArtifactError(
            f"{context} byte count mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    actual_sha256 = _sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ArtifactError(
            f"{context} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )


def _parse_protocol_bytes(payload: bytes, *, source: Path) -> ProtocolV1:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactError(f"cannot decode protocol artifact {source} as UTF-8: {exc}") from exc
    return ProtocolV1.from_dict(strict_json_loads(text, source=source))


def _parse_bundle_bytes(payload: bytes, *, source: Path) -> EvidenceBundleV1:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactError(f"cannot decode bundle root {source} as UTF-8: {exc}") from exc
    return EvidenceBundleV1.from_dict(strict_json_loads(text, source=source))


def _protocol_digest_bindings(protocol: ProtocolV1) -> dict[ProtocolDigestRole, str]:
    bindings: dict[ProtocolDigestRole, str] = {
        "plugin": protocol.plugin.artifact_sha256,
        "workloads": protocol.semantic.workloads_sha256,
        "candidates": protocol.semantic.candidates_sha256,
        "comparators": protocol.semantic.comparators_sha256,
        "splits": protocol.semantic.splits_sha256,
        "numerics": protocol.semantic.numerics_sha256,
        "timing": protocol.semantic.timing_sha256,
        "analyzer": protocol.analysis.analyzer_sha256,
        "expected_cells": protocol.execution.expected_cells_sha256,
        "environment_predicate": protocol.execution.environment_predicate_sha256,
        "failure_policy": protocol.execution.failure_policy_sha256,
    }
    if protocol.execution.paid_plan_sha256 is not None:
        bindings["paid_plan"] = protocol.execution.paid_plan_sha256
    if protocol.parent_protocol_sha256 is not None:
        bindings["parent_protocol"] = protocol.parent_protocol_sha256
    return bindings


def _parse_cell_ids(
    payload: bytes,
    *,
    role: CellIdentityRole,
    source: Path,
) -> tuple[str, ...]:
    try:
        text = payload.decode("utf-8")
        value = strict_json_loads(text, source=source)
        items = _object_array(value, context=f"{role} artifact")
        cell_ids = tuple(
            nonblank_string(item, context=f"{role} artifact cell ID") for item in items
        )
    except (UnicodeError, SchemaError) as exc:
        raise ArtifactError(f"invalid {role} artifact {source}: {exc}") from exc
    if len(cell_ids) != len(set(cell_ids)):
        raise ArtifactError(f"{role} artifact cell IDs must be unique")
    return cell_ids


def _parse_attempt_transitions(
    payload: bytes,
    *,
    source: Path,
    chained: bool,
) -> tuple[
    tuple[str, ...],
    dict[str, AttemptTransitionStatus],
    dict[str, AttemptTransitionStatus],
    str,
]:
    rows: list[tuple[str, bytes | None]] = []
    if chained:
        if payload and (not payload.endswith(b"\n") or b"\r" in payload):
            raise ArtifactError(
                f"attempts journal {source} must use LF-terminated rows without carriage returns"
            )
        for line_number, encoded in enumerate(
            payload[:-1].split(b"\n") if payload else [],
            start=1,
        ):
            try:
                line = encoded.decode("utf-8")
            except UnicodeError as exc:
                raise ArtifactError(
                    f"cannot decode attempts journal {source}:{line_number} as UTF-8: {exc}"
                ) from exc
            rows.append((line, encoded))
    else:
        try:
            text = payload.decode("utf-8")
        except UnicodeError as exc:
            raise ArtifactError(f"cannot decode attempts journal {source} as UTF-8: {exc}") from exc
        rows.extend((line, None) for line in text.splitlines())

    states: dict[str, AttemptTransitionStatus] = {}
    allowed: dict[AttemptTransitionStatus | None, set[AttemptTransitionStatus]] = {
        None: {"pending"},
        "pending": {"running", "success", "failure"},
        "running": {"success", "failure"},
        "success": set(),
        "failure": set(),
    }
    head = _ATTEMPT_CHAIN_INITIAL_HEAD
    for line_number, (line, encoded_line) in enumerate(rows, start=1):
        if not line.strip():
            raise ArtifactError(f"attempts journal {source}:{line_number} contains a blank line")
        try:
            value = strict_json_loads(line, source=source, line_number=line_number)
            required = (
                ("cell_id", "predecessor_sha256", "status") if chained else ("cell_id", "status")
            )
            data = exact_fields(
                value,
                required=required,
                context=f"attempt transition line {line_number}",
            )
            cell_id = nonblank_string(
                data["cell_id"], context=f"attempt transition line {line_number} cell_id"
            )
            status = cast(
                AttemptTransitionStatus,
                _enum(
                    data["status"],
                    allowed={"pending", "running", "success", "failure"},
                    context=f"attempt transition line {line_number} status",
                ),
            )
            if chained:
                predecessor = _digest(
                    data["predecessor_sha256"],
                    context=f"attempt transition line {line_number} predecessor_sha256",
                )
                canonical = _canonical_attempt_row(
                    cell_id,
                    status,
                    predecessor,
                    context=f"attempt transition line {line_number}",
                )
                if encoded_line is None:
                    raise AssertionError("chained attempt row lost its exact bytes")
                actual_row = encoded_line + b"\n"
                if actual_row != canonical:
                    raise ArtifactError(f"attempts journal {source}:{line_number} is not canonical")
                if predecessor != head:
                    raise ArtifactError(
                        f"attempt chain predecessor mismatch at {source}:{line_number}: "
                        f"expected {head}, found {predecessor}"
                    )
                head = _sha256(actual_row)
        except SchemaError as exc:
            raise ArtifactError(f"invalid attempts journal {source}: {exc}") from exc
        previous = states.get(cell_id)
        if status not in allowed[previous]:
            raise ArtifactError(
                f"invalid attempt transition for cell {cell_id!r}: {previous!r} -> {status!r}"
            )
        states[cell_id] = status

    terminal = {
        cell_id: status for cell_id, status in states.items() if status in {"success", "failure"}
    }
    return tuple(states), terminal, states, head


def _parse_selected_suite_descriptor(payload: bytes, *, source: Path) -> int:
    try:
        value = strict_json_loads(payload.decode("utf-8"), source=source)
        data = exact_fields(
            value,
            required=("schema", "plugin_suite_index"),
            context="selected suite descriptor",
        )
        schema = nonblank_string(data["schema"], context="selected suite descriptor schema")
        if schema != _SELECTED_SUITE_SCHEMA:
            raise SchemaError(
                f"selected suite descriptor schema must be {_SELECTED_SUITE_SCHEMA!r}"
            )
        index = exact_int(
            data["plugin_suite_index"],
            context="selected suite descriptor plugin_suite_index",
            minimum=0,
        )
    except (UnicodeError, SchemaError) as exc:
        raise ArtifactError(f"invalid selected suite descriptor {source}: {exc}") from exc
    if payload != selected_suite_descriptor_bytes(index):
        raise ArtifactError(f"selected suite descriptor {source} is not canonical")
    return index


def _verify_plugin_suite_custody(
    protocol: ProtocolV1,
    artifacts_by_role: Mapping[str, Artifact],
    payloads_by_role: Mapping[str, bytes],
    paths_by_role: Mapping[str, Path],
) -> Literal["not_checked", "checked"]:
    reserved = {
        role
        for role in artifacts_by_role
        if role.startswith("plugin_suite") or role.startswith("selected_suite")
    }
    if not reserved:
        return "not_checked"
    if "selected_suite" not in reserved:
        raise ArtifactError("plugin suite custody is missing required role 'selected_suite'")

    suite_indexes: list[int] = []
    for role in reserved - {"selected_suite"}:
        match = re.fullmatch(r"plugin_suite_(0|[1-9][0-9]*)", role)
        if match is None:
            raise ArtifactError(f"malformed reserved plugin suite role {role!r}")
        suite_indexes.append(int(match.group(1)))
    suite_indexes.sort()
    if not suite_indexes or suite_indexes != list(range(len(suite_indexes))):
        raise ArtifactError(
            "plugin suite custody roles must be nonempty and contiguous from index 0"
        )

    expected_reserved = {
        "selected_suite",
        *(plugin_suite_role(index) for index in suite_indexes),
    }
    if reserved != expected_reserved:
        raise ArtifactError("plugin suite custody contains missing or extra reserved roles")

    selected_artifact = artifacts_by_role["selected_suite"]
    if (
        selected_artifact.path != "selected_suite.json"
        or selected_artifact.media_type != "application/json"
    ):
        raise ArtifactError(
            "selected_suite must use path 'selected_suite.json' and media type 'application/json'"
        )
    selected_index = _parse_selected_suite_descriptor(
        payloads_by_role["selected_suite"],
        source=paths_by_role["selected_suite"],
    )
    if selected_index >= len(suite_indexes):
        raise ArtifactError("selected suite descriptor index is outside the plugin suite inventory")

    suite_artifacts: list[tuple[Path, bytes]] = []
    for index in suite_indexes:
        role = plugin_suite_role(index)
        artifact = artifacts_by_role[role]
        if artifact.path != plugin_suite_path(index) or artifact.media_type != "application/json":
            raise ArtifactError(
                f"{role} must use path {plugin_suite_path(index)!r} and media type "
                "'application/json'"
            )
        suite_artifacts.append((paths_by_role[role], payloads_by_role[role]))

    from heliostune.scope import verify_plugin_inventory

    verified = verify_plugin_inventory(
        paths_by_role["plugin"],
        payloads_by_role["plugin"],
        suite_artifacts,
    )
    if verified.plugin.plugin_id != protocol.plugin.id:
        raise ArtifactError("protocol plugin ID does not match inventoried plugin ID")
    if str(verified.plugin.version) != protocol.plugin.version:
        raise ArtifactError(
            "protocol plugin version is not the canonical decimal inventoried plugin version"
        )
    return "checked"


def _attempt_chain_is_enabled(
    artifacts_by_role: Mapping[str, Artifact],
    payloads_by_role: Mapping[str, bytes],
    paths_by_role: Mapping[str, Path],
) -> bool:
    reserved = {role for role in artifacts_by_role if role.startswith("attempt_chain")}
    if not reserved:
        return False
    if reserved != {"attempt_chain"}:
        raise ArtifactError("attempt chain custody contains malformed reserved roles")
    artifact = artifacts_by_role["attempt_chain"]
    if artifact.path != "attempt_chain.json" or artifact.media_type != "application/json":
        raise ArtifactError(
            "attempt_chain must use path 'attempt_chain.json' and media type 'application/json'"
        )
    payload = payloads_by_role["attempt_chain"]
    if payload != attempt_chain_descriptor_bytes():
        raise ArtifactError(f"invalid attempt chain descriptor {paths_by_role['attempt_chain']}")
    return True


def _verify_declared_closure(
    bundle: EvidenceBundleV1,
    protocol: ProtocolV1,
    *,
    expected_ids: tuple[str, ...],
    terminal_ids: tuple[str, ...],
    journal_ids: tuple[str, ...],
    terminal_statuses: dict[str, AttemptTransitionStatus],
    chained: bool,
) -> None:
    closed_states = {"SEALED", "VERIFIED", "ANALYZED", "PUBLISHED"}
    if bundle.lifecycle.state not in closed_states:
        raise ArtifactError(f"bundle lifecycle state {bundle.lifecycle.state!r} is not closed")

    expected_set = set(expected_ids)
    terminal_set = set(terminal_ids)
    if not terminal_set <= expected_set:
        raise ArtifactError("terminal_cells contains a cell not present in expected_cells")
    if not set(journal_ids) <= expected_set:
        raise ArtifactError("attempt journal contains a cell not present in expected_cells")
    if set(terminal_statuses) != terminal_set:
        raise ArtifactError("attempt journal terminal cell IDs do not match terminal_cells")
    if chained and journal_ids and len(terminal_statuses) != len(journal_ids):
        raise ArtifactError("a chained closed attempt journal cannot end in a live state")

    coverage = bundle.coverage
    expected_count = len(expected_ids)
    terminal_count = len(terminal_ids)
    if protocol.execution.expected_cell_count != expected_count:
        raise ArtifactError(
            "expected_cells artifact count does not match protocol expected_cell_count: "
            f"{expected_count} != {protocol.execution.expected_cell_count}"
        )
    if coverage.expected_cells != expected_count:
        raise ArtifactError(
            "bundle coverage expected_cells does not match parsed expected_cells: "
            f"{coverage.expected_cells} != {expected_count}"
        )
    if coverage.terminal_cells != terminal_count:
        raise ArtifactError(
            "bundle coverage terminal_cells does not match parsed terminal_cells: "
            f"{coverage.terminal_cells} != {terminal_count}"
        )
    if bundle.attempts.logical != len(journal_ids):
        raise ArtifactError(
            "bundle attempts logical does not match journal cell IDs: "
            f"{bundle.attempts.logical} != {len(journal_ids)}"
        )
    if bundle.attempts.terminal != terminal_count:
        raise ArtifactError(
            "bundle attempts terminal does not match parsed terminal_cells: "
            f"{bundle.attempts.terminal} != {terminal_count}"
        )

    attempts = bundle.attempts
    execution = protocol.execution
    if execution.retry_policy == "none":
        if execution.max_physical_attempts != 1 or attempts.physical != attempts.logical:
            raise ArtifactError(
                "retry_policy 'none' requires physical attempts to equal logical attempts "
                "and max_physical_attempts to equal 1"
            )
    elif attempts.physical > attempts.logical * execution.max_physical_attempts:
        raise ArtifactError(
            "bundle physical attempts exceed the retry-policy bound: "
            f"{attempts.physical} > "
            f"{attempts.logical} * {execution.max_physical_attempts}"
        )

    successes = sum(status == "success" for status in terminal_statuses.values())
    failures = sum(status == "failure" for status in terminal_statuses.values())
    if coverage.successes != successes or coverage.failures != failures:
        raise ArtifactError(
            "bundle coverage success/failure counts do not match attempt journal terminal statuses"
        )

    complete = terminal_set == expected_set
    if bundle.lifecycle.outcome == "completed" and not complete:
        raise ArtifactError("a completed bundle must close every expected cell")
    if protocol.evidence_class != "exploratory" and not complete:
        raise ArtifactError(f"{protocol.evidence_class} evidence must close every expected cell")
    if protocol.evidence_class == "exploratory" and not complete:
        expected_journal_prefix = expected_ids[: len(journal_ids)]
        if journal_ids != expected_journal_prefix:
            raise ArtifactError(
                "incomplete exploratory attempt journal must retain an expected-cell prefix"
            )
        if terminal_ids != expected_ids[:terminal_count]:
            raise ArtifactError(
                "incomplete exploratory terminal cells must retain an expected-cell prefix"
            )


def verify_protocol_v1(path: str | Path) -> VerifiedProtocol:
    """Strictly parse a protocol and bind its exact filesystem bytes."""

    resolved = _resolve_file(path, context="protocol artifact")
    payload = _read_verified_file(resolved, context="protocol artifact")
    return VerifiedProtocol(
        protocol=_parse_protocol_bytes(payload, source=resolved),
        path=resolved,
        sha256=_sha256(payload),
        bytes=len(payload),
    )


def _verify_bundle_with_reader(reader: _BundleDirectoryReader) -> VerifiedBundle:
    """Verify a bundle through an already-pinned directory reader."""
    try:
        root_path, root_payload = reader.read(
            reader.root_relative_path,
            context="bundle root",
        )
        bundle = _parse_bundle_bytes(root_payload, source=root_path)

        protocol_path, protocol_payload = reader.read(
            bundle.protocol.path,
            context="bundle protocol",
        )
        _require_file_identity(
            protocol_payload,
            expected_sha256=bundle.protocol.sha256,
            expected_bytes=bundle.protocol.bytes,
            context="bundle protocol",
        )
        protocol = VerifiedProtocol(
            protocol=_parse_protocol_bytes(protocol_payload, source=protocol_path),
            path=protocol_path,
            sha256=_sha256(protocol_payload),
            bytes=len(protocol_payload),
        )

        attempts_path, attempts_payload = reader.read(
            bundle.attempts.path,
            context="bundle attempts",
        )
        _require_file_identity(
            attempts_payload,
            expected_sha256=bundle.attempts.sha256,
            expected_bytes=None,
            context="bundle attempts",
        )

        artifacts_by_role: dict[str, Artifact] = {}
        payloads_by_role: dict[str, bytes] = {}
        paths_by_role: dict[str, Path] = {}
        artifact_paths: list[Path] = []
        for artifact in bundle.artifacts:
            artifacts_by_role[artifact.role] = artifact
            artifact_path, artifact_payload = reader.read(
                artifact.path,
                context=f"bundle artifact role {artifact.role!r}",
            )
            _require_file_identity(
                artifact_payload,
                expected_sha256=artifact.sha256,
                expected_bytes=artifact.bytes,
                context=f"bundle artifact role {artifact.role!r}",
            )
            payloads_by_role[artifact.role] = artifact_payload
            paths_by_role[artifact.role] = artifact_path
            artifact_paths.append(artifact_path)

        digest_bindings = _protocol_digest_bindings(protocol.protocol)
        for role, expected_sha256 in digest_bindings.items():
            bound_artifact = artifacts_by_role.get(role)
            if bound_artifact is None:
                raise ArtifactError(f"bundle is missing protocol digest role {role!r}")
            if bound_artifact.sha256 != expected_sha256:
                raise ArtifactError(
                    f"bundle artifact role {role!r} does not match its protocol SHA-256"
                )
        for optional_role in ("paid_plan", "parent_protocol"):
            if optional_role in artifacts_by_role and optional_role not in digest_bindings:
                raise ArtifactError(
                    f"bundle artifact role {optional_role!r} has no protocol digest to bind"
                )
        if "terminal_cells" not in artifacts_by_role:
            raise ArtifactError("bundle is missing required artifact role 'terminal_cells'")

        custody = _verify_plugin_suite_custody(
            protocol.protocol,
            artifacts_by_role,
            payloads_by_role,
            paths_by_role,
        )
        chained = _attempt_chain_is_enabled(
            artifacts_by_role,
            payloads_by_role,
            paths_by_role,
        )
        expected_ids = _parse_cell_ids(
            payloads_by_role["expected_cells"],
            role="expected_cells",
            source=paths_by_role["expected_cells"],
        )
        terminal_ids = _parse_cell_ids(
            payloads_by_role["terminal_cells"],
            role="terminal_cells",
            source=paths_by_role["terminal_cells"],
        )
        journal_ids, terminal_statuses, final_states, final_head = _parse_attempt_transitions(
            attempts_payload,
            source=attempts_path,
            chained=chained,
        )
        if chained and bundle.attempts.hash_chain_head != final_head:
            raise ArtifactError(
                "attempt chain final head mismatch: "
                f"expected {bundle.attempts.hash_chain_head}, found {final_head}"
            )
        _verify_declared_closure(
            bundle,
            protocol.protocol,
            expected_ids=expected_ids,
            terminal_ids=terminal_ids,
            journal_ids=journal_ids,
            terminal_statuses=terminal_statuses,
            chained=chained,
        )
        attempts = bundle.attempts
        execution = protocol.protocol.execution
        reconciled = (
            execution.retry_policy == "none"
            and execution.max_physical_attempts == 1
            and attempts.physical == attempts.logical
            and attempts.orphaned == 0
            and len(final_states) == attempts.logical
            and all(status in {"success", "failure"} for status in final_states.values())
        )
        return VerifiedBundle(
            bundle=bundle,
            protocol=protocol,
            root_path=root_path,
            root_sha256=_sha256(root_payload),
            root_bytes=len(root_payload),
            referenced_paths=(protocol_path, attempts_path, *artifact_paths),
            limitations=VerificationLimitations(
                plugin_suite_custody=custody,
                attempt_journal_hash_chain="checked" if chained else "not_checked",
                attempt_reconciliation="checked" if reconciled else "not_checked",
            ),
        )
    finally:
        reader.close()


def verify_bundle_v1(root_manifest_path: str | Path) -> VerifiedBundle:
    """Verify a bundle by resolving and pinning its containing directory once."""

    return _verify_bundle_with_reader(_BundleDirectoryReader(root_manifest_path))


def verify_bundle_v1_from_directory_fd(
    directory_fd: int,
    root_relative_path: str = "bundle.json",
    *,
    diagnostic_directory: str | Path | None = None,
) -> VerifiedBundle:
    """Verify solely through a duplicate of an already-open bundle directory fd.

    ``diagnostic_directory`` is never resolved or opened; it only labels paths
    in the returned inventory and any diagnostics.
    """

    descriptor = exact_int(directory_fd, context="bundle directory descriptor", minimum=0)
    return _verify_bundle_with_reader(
        _BundleDirectoryReader.from_directory_fd(
            descriptor,
            root_relative_path,
            diagnostic_directory=diagnostic_directory,
        )
    )


__all__ = [
    "Analysis",
    "Artifact",
    "CELL_IDENTITY_ROLES",
    "AttemptsSummary",
    "ClaimSpec",
    "CellIdentityRole",
    "Coverage",
    "EvidenceBundleV1",
    "Execution",
    "Lifecycle",
    "PROTOCOL_DIGEST_ROLES",
    "Plugin",
    "ProtocolBinding",
    "ProtocolDigestRole",
    "ProtocolV1",
    "Provenance",
    "Semantic",
    "Signature",
    "VerificationLimitations",
    "VerifiedBundle",
    "VerifiedProtocol",
    "attempt_chain_descriptor_bytes",
    "encode_attempt_journal",
    "load_bundle_v1",
    "load_protocol_v1",
    "plugin_suite_path",
    "plugin_suite_role",
    "selected_suite_descriptor_bytes",
    "verify_bundle_v1",
    "verify_bundle_v1_from_directory_fd",
    "verify_protocol_v1",
]
