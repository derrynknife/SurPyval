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

import json
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.distribution import Distribution
from surpyval.serialisation import (
    checked_from_dict,
    read_model_dict,
    require_model_tag,
    stamp_schema,
)

# The serialisation of the two classes. They are the model themselves
# (every method is a classmethod), so the instance-method ``to_json`` of
# ``SerialisableMixin`` does not fit; these are its classmethod
# counterparts. ``to_json`` used to be missing altogether, although the
# package reader ``surpyval.from_json`` was registered for both classes.
# Not being mixin users, their ``from_dict`` takes the shared reader
# checks (schema, ...) from ``checked_from_dict`` explicitly.


def _degenerate_to_dict(cls: type[Any]) -> dict[str, Any]:
    return stamp_schema({"model": cls.name})


def _degenerate_from_dict(
    cls: type[Any], model_dict: dict[str, Any]
) -> type[Distribution]:
    # The tag is the whole of the model, so check it: any dictionary at
    # all used to "restore" as whichever class was asked.
    require_model_tag(model_dict, cls.name, f"the {cls.name} distribution")
    return cls


def _degenerate_to_json(
    cls: type[Distribution], fp: str | os.PathLike
) -> None:
    with open(fp, "w+") as f:
        json.dump(_degenerate_to_dict(cls), f, allow_nan=False)


def _degenerate_from_json(
    cls: type[Distribution], fp: str | os.PathLike
) -> type[Distribution]:
    with open(fp, "r") as f:
        model_dict = json.load(f)
    if not isinstance(model_dict, dict):
        raise ValueError(
            "Expected a serialised model dict, got "
            f"{type(model_dict).__name__}"
        )
    return read_model_dict(cls, model_dict)


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
        return _degenerate_to_dict(cls)

    @classmethod
    @checked_from_dict
    def from_dict(cls, model_dict: dict[str, Any]) -> type["Distribution"]:
        return _degenerate_from_dict(cls, model_dict)

    @classmethod
    def to_json(cls, fp: str | os.PathLike) -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        _degenerate_to_json(cls, fp)

    @classmethod
    def from_json(cls, fp: str | os.PathLike) -> type["Distribution"]:
        """Load the distribution from a JSON file written by
        :meth:`to_json`."""
        return _degenerate_from_json(cls, fp)


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
        return _degenerate_to_dict(cls)

    @classmethod
    @checked_from_dict
    def from_dict(cls, model_dict: dict[str, Any]) -> type["Distribution"]:
        return _degenerate_from_dict(cls, model_dict)

    @classmethod
    def to_json(cls, fp: str | os.PathLike) -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        _degenerate_to_json(cls, fp)

    @classmethod
    def from_json(cls, fp: str | os.PathLike) -> type["Distribution"]:
        """Load the distribution from a JSON file written by
        :meth:`to_json`."""
        return _degenerate_from_json(cls, fp)
