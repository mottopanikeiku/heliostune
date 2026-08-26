from collections.abc import Callable, Sequence
from typing import Any

def do_bench(
    function: Callable[[], Any],
    *,
    warmup: float = ...,
    rep: float = ...,
    quantiles: Sequence[float] | None = ...,
) -> Any: ...
