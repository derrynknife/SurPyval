"""The import of PyTorch, made only inside the helpers.

PyTorch is not a dependency of SurPyval, not even an optional extra:
these helpers are for people who already train their own networks with
it. ``import surpyval`` and ``import surpyval.torch`` never import it;
each helper calls :func:`torch` on its first line, which refuses with
what to install when it is absent.
"""

from __future__ import annotations

import importlib
from typing import Any


def torch(what: str) -> Any:
    """The ``torch`` module, or an ImportError saying to install it."""
    try:
        # (the installed PyTorch: this package is ``surpyval.torch``)
        return importlib.import_module("torch")
    except ImportError as err:
        raise ImportError(
            f"{what}: the torch helpers need PyTorch, which is not "
            "installed. SurPyval does not install it for you: "
            "`pip install torch` (see pytorch.org for the build for your "
            "machine; the CPU build is `pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu`)."
        ) from err
