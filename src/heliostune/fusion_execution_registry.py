"""Frozen remote execution bindings for authenticated fusion suites."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import SchemaError


@dataclass(frozen=True, slots=True)
class FusionExecutionSpec:
    """Immutable suite, plugin, and Modal API identity for one executable suite."""

    suite_sha256: str
    suite_id: str
    suite_revision: int
    plugin_id: str
    plugin_version: int
    plugin_sha256: str
    modal_executor_api: str


_REFERENCE_PLUGIN_SHA256 = "9d696f135a5e62ef622a88d85a7bb03e8fa76bddd0bf57ebf20b2eb4c1d1edc1"

FUSION_EXECUTION_REGISTRY: Mapping[str, FusionExecutionSpec] = MappingProxyType(
    {
        spec.suite_sha256: spec
        for spec in (
            FusionExecutionSpec(
                suite_sha256="407487a6aa7dc157dcd4aa7bcab698168813bf0a79916d70d91163dc384fe8a8",
                suite_id="gated-mlp-epilogue-reference",
                suite_revision=1,
                plugin_id="fusion-reference-plugin",
                plugin_version=1,
                plugin_sha256=_REFERENCE_PLUGIN_SHA256,
                modal_executor_api="heliostune.modal_fusion_executor/1",
            ),
            FusionExecutionSpec(
                suite_sha256="a318a59bca434b97d073e0ae76f827814213c0a68b0c4263b19c81f98be8f9ee",
                suite_id="residual-rmsnorm-reference",
                suite_revision=1,
                plugin_id="fusion-reference-plugin",
                plugin_version=1,
                plugin_sha256=_REFERENCE_PLUGIN_SHA256,
                modal_executor_api="heliostune.modal_fusion_executor/1",
            ),
            FusionExecutionSpec(
                suite_sha256="23f7397f2adee93cd9f7919aaf075c0f8b5e92cd6d4257ce4c54197d3c98035f",
                suite_id="residual-rmsnorm-triton",
                suite_revision=1,
                plugin_id="fusion-triton-rmsnorm-plugin",
                plugin_version=1,
                plugin_sha256="ce4a497113adf1ee82ed995fb4ba671a8a1664d756321499d91187056ca0d815",
                modal_executor_api="heliostune.modal_fusion_executor/2",
            ),
        )
    }
)


def fusion_execution_spec(suite_sha256: str) -> FusionExecutionSpec:
    """Resolve an exact frozen suite digest or reject it closed."""

    if (
        type(suite_sha256) is not str
        or len(suite_sha256) != 64
        or any(character not in "0123456789abcdef" for character in suite_sha256)
    ):
        raise SchemaError("fusion execution suite SHA-256 must be a 64-character lowercase digest")
    try:
        return FUSION_EXECUTION_REGISTRY[suite_sha256]
    except KeyError as exc:
        raise SchemaError(
            f"fusion execution is closed to unsupported suite SHA-256 {suite_sha256}"
        ) from exc


__all__ = ["FUSION_EXECUTION_REGISTRY", "FusionExecutionSpec", "fusion_execution_spec"]
