# Everything in autograd except ArrayBox stays untyped.
#
# Supplying any stub for a package makes mypy consider the whole package
# described, so ``ignore_missing_imports`` no longer covers it and every
# other autograd attribute becomes an error. This ``__getattr__`` keeps
# the rest of the package Any, which is what it was before, while
# ``numpy.numpy_boxes`` gets a real class.

from typing import Any

def __getattr__(name: str) -> Any: ...
