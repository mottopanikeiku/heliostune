from __future__ import annotations

from typing import Any

import pytest

from heliostune.action_configs import (
    NON_SPLIT_CANDIDATES,
    SPLIT_K_CANDIDATES,
    ActionCandidate,
)


@pytest.fixture
def gpu_modules() -> tuple[Any, Any]:
    torch: Any = pytest.importorskip("torch")
    triton: Any = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not torch.__version__.startswith("2.8.0") or triton.__version__ != "3.4.0":
        pytest.skip("the pinned PyTorch 2.8/Triton 3.4 stack is unavailable")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("these tensor-core candidates require Ampere or newer")
    from heliostune import action_expansion

    return torch, action_expansion


def _assert_matches_reference(torch: Any, output: Any, a: Any, b: Any) -> None:
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = torch.mm(a, b, out_dtype=torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert output.shape == (a.shape[0], b.shape[1])
    assert output.dtype == torch.float16
    assert output.device == a.device
    assert bool(torch.isfinite(output).all().item())
    torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("candidate", SPLIT_K_CANDIDATES, ids=lambda c: c.key)
@pytest.mark.parametrize("m", (1, 7, 31))
@pytest.mark.parametrize("k", (17, 1543))
def test_split_k_masks_edges_and_empty_splits(
    gpu_modules: tuple[Any, Any], candidate: ActionCandidate, m: int, k: int
) -> None:
    torch, expansion = gpu_modules
    torch.manual_seed(17)
    a = torch.rand((m, k), device="cuda", dtype=torch.float16)
    b = torch.rand((k, 131), device="cuda", dtype=torch.float16)
    output = expansion.launch(a, b, candidate)
    _assert_matches_reference(torch, output, a, b)


@pytest.mark.parametrize("candidate", SPLIT_K_CANDIDATES, ids=lambda c: c.key)
def test_split_k_supports_strided_inputs(
    gpu_modules: tuple[Any, Any], candidate: ActionCandidate
) -> None:
    torch, expansion = gpu_modules
    torch.manual_seed(31)
    a = torch.rand((31, 514), device="cuda", dtype=torch.float16)[:, ::2]
    b = torch.rand((514, 134), device="cuda", dtype=torch.float16)[::2, ::2]
    assert not a.is_contiguous() and not b.is_contiguous()
    output = expansion.launch(a, b, candidate)
    _assert_matches_reference(torch, output, a, b)


@pytest.mark.parametrize("candidate", NON_SPLIT_CANDIDATES, ids=lambda c: c.key)
@pytest.mark.parametrize("m", (1, 7, 31, 257))
def test_frozen_tile_and_persistent_candidates_match_reference(
    gpu_modules: tuple[Any, Any], candidate: ActionCandidate, m: int
) -> None:
    torch, expansion = gpu_modules
    if candidate.family == "persistent" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("persistent TMA correctness is exercised on Hopper")
    torch.manual_seed(257)
    a = torch.rand((m, 256), device="cuda", dtype=torch.float16)
    b = torch.rand((256, 256), device="cuda", dtype=torch.float16)
    output = expansion.launch(a, b, candidate)
    _assert_matches_reference(torch, output, a, b)


@pytest.mark.parametrize(
    ("family", "entrypoint"),
    (("tile", "matmul"), ("persistent", "hopper_matmul"), ("split_k", "_split_k_matmul")),
)
def test_launch_propagates_failures_without_another_implementation(
    monkeypatch: pytest.MonkeyPatch, family: str, entrypoint: str
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from heliostune import action_expansion

    candidate = next(c for c in (*NON_SPLIT_CANDIDATES, *SPLIT_K_CANDIDATES) if c.family == family)
    failure = RuntimeError("requested kernel failed")

    def fail(*args: object) -> None:
        raise failure

    def forbidden(*args: object) -> None:
        raise AssertionError("another implementation must not run")

    for name in ("matmul", "hopper_matmul", "_split_k_matmul"):
        monkeypatch.setattr(action_expansion, name, fail if name == entrypoint else forbidden)
    with pytest.raises(RuntimeError) as raised:
        action_expansion.launch(None, None, candidate)  # type: ignore[arg-type]
    assert raised.value is failure
