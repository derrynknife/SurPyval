"""The ``to_dict`` of the test planning results: native Python values,
strict JSON (an infinite or missing number is ``None``, recorded under
``"non_finite"``, as every SurPyval ``to_dict`` writes it)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def _plain(value: Any) -> Any:
    """``value`` with numpy arrays and scalars, tuples and nested dicts
    made native: lists, floats, ints and dicts."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def result_dict(result: Any, values: dict[str, Any]) -> dict[str, Any]:
    """``{"result": <class name>, **values}``, native and strict JSON."""
    from surpyval.serialisation import encode_non_finite

    return encode_non_finite(
        {"result": type(result).__name__, **_plain(values)}
    )


def dataclass_dict(result: Any) -> dict[str, Any]:
    """A result dataclass's fields, as :func:`result_dict` writes them."""
    values = {
        f.name: getattr(result, f.name) for f in dataclasses.fields(result)
    }
    return result_dict(result, values)
