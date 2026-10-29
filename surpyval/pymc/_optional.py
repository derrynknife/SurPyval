"""Imports of PyMC and pytensor, made only inside the helpers.

PyMC is not a dependency of SurPyval, not even an optional extra: these
helpers are for people who already use it. ``import surpyval`` and
``import surpyval.pymc`` never import it; each helper calls one of these
on its first line, which refuses with what to install when it is absent.
"""

from __future__ import annotations

import importlib
from typing import Any


def _module(name: str, what: str, install: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as err:
        raise ImportError(
            f"{what} needs {install.split()[0]}, which is not installed. "
            f"SurPyval does not install it for you: install pymc to use "
            f"this (`pip install {install}` or "
            f"`conda install -c conda-forge {install}`)."
        ) from err


def pytensor_tensor(what: str) -> Any:
    """``pytensor.tensor``, or an ImportError saying to install PyMC
    (which brings pytensor)."""
    return _module("pytensor.tensor", what, "pymc")


def pymc(what: str) -> Any:
    """The ``pymc`` module, or an ImportError saying to install it."""
    return _module("pymc", what, "pymc")


def arviz(what: str) -> Any:
    """The ``arviz`` module (installed with PyMC), or an ImportError."""
    return _module("arviz", what, "pymc")


def pytensor_assert(message: str) -> Any:
    """pytensor's ``Assert`` op with ``message``, checked when the graph
    runs."""
    raise_op = _module("pytensor.raise_op", "This check", "pymc")
    return raise_op.Assert(message)
