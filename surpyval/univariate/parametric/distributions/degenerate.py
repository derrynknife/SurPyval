from typing import Any

import numpy.typing as npt

"""The two degenerate lifetime distributions.

``InstantlyOccurs`` is the point mass at zero (every unit has already
failed) and ``NeverOccurs`` the point mass at infinity (no unit ever
fails). They are the limits of distributions in the ordinary catalogue —
``FixedEventProbability`` at ``p = 1`` / ``p = 0``, or ``ExactEventTime``
at ``T = 0`` / ``T = inf`` — kept as their own named models because they
arise as boundary cases in composed models: the survival-tree leaves use
``NeverOccurs`` for a node with no events, and mixtures or renewal
compositions can degenerate the same way.

They are stateless (no parameters, nothing to fit), so the *class* is
the model: every method is a classmethod and the classes serialise by
name alone.
"""

import numpy as np

from surpyval.distribution import Distribution
from surpyval.serialisation import stamp_schema


class NeverOccurs(Distribution):
    """The event never occurs: ``R(x) = 1`` everywhere (mass at +inf)."""

    name = "NeverOccurs"

    @classmethod
    def sf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.ones_like(x).astype(float)

    @classmethod
    def ff(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros_like(x).astype(float)

    @classmethod
    def df(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros_like(x).astype(float)

    @classmethod
    def hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros_like(x).astype(float)

    @classmethod
    def Hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros_like(x).astype(float)

    @classmethod
    def qf(cls, u: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.full_like(np.asarray(u, dtype=float), np.inf)

    @classmethod
    def mean(cls, *args: Any, **kwargs: Any) -> float:
        return np.inf

    @classmethod
    def random(cls, size: int, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.ones(size) * np.inf

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return stamp_schema({"model": cls.name})

    @classmethod
    def from_dict(cls, model_dict: dict[str, Any]) -> type["Distribution"]:
        return cls


class InstantlyOccurs(Distribution):
    """The event has already occurred: ``F(x) = 1`` everywhere (mass at 0)."""

    name = "InstantlyOccurs"

    @classmethod
    def sf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros_like(x).astype(float)

    @classmethod
    def ff(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.ones_like(x).astype(float)

    @classmethod
    def df(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        # Point mass at zero: the "density" is the degenerate spike there.
        x = np.asarray(x, dtype=float)
        return np.where(x == 0, np.inf, 0.0)

    @classmethod
    def hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.full_like(np.asarray(x, dtype=float), np.inf)

    @classmethod
    def Hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.full_like(x, np.inf, dtype=float)

    @classmethod
    def qf(cls, u: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros_like(np.asarray(u, dtype=float))

    @classmethod
    def mean(cls, *args: Any, **kwargs: Any) -> float:
        return 0.0

    @classmethod
    def random(cls, size: int, *args: Any, **kwargs: Any) -> npt.NDArray:
        return np.zeros(size)

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return stamp_schema({"model": cls.name})

    @classmethod
    def from_dict(cls, model_dict: dict[str, Any]) -> type["Distribution"]:
        return cls
