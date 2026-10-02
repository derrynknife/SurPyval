"""The input check every degradation fit and prediction shares (#352)."""

from typing import Any, overload

import numpy as np
import numpy.typing as npt


@overload
def validate_xy(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    i: None = None,
    *,
    i_name: str = "i",
) -> tuple[npt.NDArray, npt.NDArray, None]: ...


@overload
def validate_xy(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    i: npt.ArrayLike,
    *,
    i_name: str = "i",
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]: ...


def validate_xy(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    i: Any = None,
    *,
    i_name: str = "i",
) -> tuple[npt.NDArray, npt.NDArray, "npt.NDArray | None"]:
    """``(x, y, i)`` as one-dimensional arrays: the measurement times and
    values as floats, and the unit labels ``i`` (named ``i_name`` in the
    messages; the destructive model's censoring flags ``c``) as given, or
    ``None`` without them.

    Refuses, in order, with a ``ValueError``: an array that is not one
    dimensional, arrays of different lengths (giving them), no
    measurements, and a missing or infinite time or value (naming the
    array). The degradation path, process and destructive fits and the
    single-trajectory predictions each carried a copy of these checks,
    with different wording and some of them missing.
    """
    arrays = {
        "x": np.atleast_1d(np.asarray(x, dtype=float)),
        "y": np.atleast_1d(np.asarray(y, dtype=float)),
    }
    if i is not None:
        arrays[i_name] = np.atleast_1d(np.asarray(i))
    names = list(arrays)
    listed = (
        "{} and {}".format(*names)
        if len(names) == 2
        else "{}, {}, and {}".format(*names)
    )
    if any(a.ndim != 1 for a in arrays.values()):
        raise ValueError("{} must be one dimensional".format(listed))
    lengths = [len(a) for a in arrays.values()]
    if len(set(lengths)) > 1:
        raise ValueError(
            "{} must have the same length; got {}".format(
                listed,
                (
                    "{} and {}" if len(lengths) == 2 else "{}, {}, and {}"
                ).format(*lengths),
            )
        )
    if lengths[0] == 0:
        raise ValueError("{} must not be empty".format(listed))
    for name in ("x", "y"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError("{} must contain only finite values".format(name))
    return arrays["x"], arrays["y"], arrays.get(i_name)
