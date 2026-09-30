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

import os
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.distribution import Distribution
from surpyval.serialisation import (
    checked_from_dict,
    read_json,
    read_model_dict,
    require_model_tag,
    stamp_schema,
    write_json,
)
from surpyval.utils.shapes import keeps_query_shape

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
    cls: type[Distribution], fp: str | os.PathLike | None
) -> str | None:
    return write_json(_degenerate_to_dict(cls), fp)


def _degenerate_from_json(
    cls: type[Distribution], fp: str | os.PathLike
) -> type[Distribution]:
    model_dict = read_json(fp)
    if not isinstance(model_dict, dict):
        raise ValueError(
            "Expected a serialised model dict, got "
            f"{type(model_dict).__name__}"
        )
    return read_model_dict(cls, model_dict)


def _constant(x: npt.ArrayLike, value: float) -> npt.NDArray:
    """``value`` at every point of ``x``, and NaN where ``x`` is NaN: a
    missing query is answered as missing (principle 3; these returned the
    constant there, #382)."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), np.nan, np.full_like(x, value))


class NeverOccurs(Distribution):
    """The event never occurs: ``R(x) = 1`` everywhere (mass at +inf).

    Stateless: the class itself is the model, with nothing to fit.

    Examples
    --------
    >>> from surpyval import NeverOccurs
    >>> NeverOccurs.sf([1, 10, 100])
    array([1., 1., 1.])
    >>> NeverOccurs.qf([0.5])
    array([inf])
    >>> NeverOccurs.mean()
    inf
    """

    name = "NeverOccurs"

    @classmethod
    @keeps_query_shape
    def sf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 1.0)

    @classmethod
    @keeps_query_shape
    def ff(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 0.0)

    @classmethod
    @keeps_query_shape
    def df(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 0.0)

    @classmethod
    @keeps_query_shape
    def hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 0.0)

    @classmethod
    @keeps_query_shape
    def Hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 0.0)

    @classmethod
    @keeps_query_shape
    def qf(cls, p: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(p, np.inf)

    @classmethod
    def mean(cls, *args: Any, **kwargs: Any) -> float:
        return np.inf

    @classmethod
    def random(
        cls,
        size: int,
        *args: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        # A point mass: random_state is taken for the common signature.
        return np.ones(size) * np.inf

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return _degenerate_to_dict(cls)

    @classmethod
    @checked_from_dict
    def from_dict(cls, model_dict: dict[str, Any]) -> type["Distribution"]:
        return _degenerate_from_dict(cls, model_dict)

    @classmethod
    def to_json(cls, fp: str | os.PathLike | None = None) -> str | None:
        """Write :meth:`to_dict` to ``fp`` as JSON, or return the JSON
        text without ``fp``."""
        return _degenerate_to_json(cls, fp)

    @classmethod
    def from_json(cls, fp: str | os.PathLike) -> type["Distribution"]:
        """Load the distribution from a JSON file written by
        :meth:`to_json`."""
        return _degenerate_from_json(cls, fp)


class InstantlyOccurs(Distribution):
    """The event has already occurred: ``F(x) = 1`` everywhere (mass at 0).

    Stateless: the class itself is the model, with nothing to fit.

    Examples
    --------
    >>> from surpyval import InstantlyOccurs
    >>> InstantlyOccurs.ff([0, 1, 10])
    array([1., 1., 1.])
    >>> InstantlyOccurs.sf([0, 1, 10])
    array([0., 0., 0.])
    >>> InstantlyOccurs.mean()
    0.0
    """

    name = "InstantlyOccurs"

    @classmethod
    @keeps_query_shape
    def sf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 0.0)

    @classmethod
    @keeps_query_shape
    def ff(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, 1.0)

    @classmethod
    @keeps_query_shape
    def df(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        # Point mass at zero: the "density" is the degenerate spike there.
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), np.nan, np.where(x == 0, np.inf, 0.0))

    @classmethod
    @keeps_query_shape
    def hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, np.inf)

    @classmethod
    @keeps_query_shape
    def Hf(cls, x: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(x, np.inf)

    @classmethod
    @keeps_query_shape
    def qf(cls, p: npt.ArrayLike, *args: Any, **kwargs: Any) -> npt.NDArray:
        return _constant(p, 0.0)

    @classmethod
    def mean(cls, *args: Any, **kwargs: Any) -> float:
        return 0.0

    @classmethod
    def random(
        cls,
        size: int,
        *args: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        # A point mass: random_state is taken for the common signature.
        return np.zeros(size)

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return _degenerate_to_dict(cls)

    @classmethod
    @checked_from_dict
    def from_dict(cls, model_dict: dict[str, Any]) -> type["Distribution"]:
        return _degenerate_from_dict(cls, model_dict)

    @classmethod
    def to_json(cls, fp: str | os.PathLike | None = None) -> str | None:
        """Write :meth:`to_dict` to ``fp`` as JSON, or return the JSON
        text without ``fp``."""
        return _degenerate_to_json(cls, fp)

    @classmethod
    def from_json(cls, fp: str | os.PathLike) -> type["Distribution"]:
        """Load the distribution from a JSON file written by
        :meth:`to_json`."""
        return _degenerate_from_json(cls, fp)
