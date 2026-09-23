# See ../__init__.pyi -- keeps autograd.numpy untyped, so that
# ``surpyval.np`` behaves as it did before the ArrayBox stub existed.

from typing import Any

def __getattr__(name: str) -> Any: ...
