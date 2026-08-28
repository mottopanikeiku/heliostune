from collections.abc import Callable
from typing import Any, TypeVar

from triton import language as language
from triton import testing as testing

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., Any])

__version__: str

def jit(function: _FunctionT) -> Any: ...
def cdiv(left: Any, right: Any) -> Any: ...
