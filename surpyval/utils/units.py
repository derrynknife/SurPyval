"""The unit a model's data was in (#845): a label, for display and
storage only.

``fit``, ``fit_from_df`` and ``from_params`` take ``unit=`` -- any unit,
not only of time (``"hours"``, ``"km"``, ``"cycles"``, ``"pump starts"``)
-- and the model keeps it as ``model.unit``, shows it in its ``repr`` and
on the axes of its plots, and saves it in ``to_dict``. A unit pint knows
is stored by its canonical name, so the same unit always reads the same
(``"hrs"``, ``"hours"`` and ``"Hour"`` are all ``"hour"``); anything else
is kept as given. Nothing is converted or checked: a tool that combines
models (RePyability, say) converts with pint from these names.

It is ``unit``, not ``units``: in SurPyval "units" are the items measured
(a degradation model's ``units`` are its unit ids).

pint is imported only when a unit is given, so ``import surpyval`` does
not load it (#470).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

__all__ = [
    "normalise_unit",
    "unit_symbol",
    "axis_label",
    "label_axes",
    "set_unit",
    "takes_unit",
]

# Counts that reliability data are measured in, which pint does not know
# or knows as something else: its ``cycle`` is an angle (a turn). Each is
# a dimension of its own, so ``"cycles"`` reads as ``"cycle"`` and a
# kilocycle is a thousand of them.
_COUNTS = (
    "cycle = [cycle] = cyc",
    "start = [start]",
    "demand = [demand]",
    "landing = [landing]",
    "flight = [flight]",
    "mission = [mission]",
    "operation = [operation] = op",
    "actuation = [actuation]",
)
_COUNT_NAMES = frozenset(d.split("=")[0].strip() for d in _COUNTS)

# Symbols that read better on an axis than pint's: a year is "a" (annum).
_SYMBOLS = {"year": "yr"}


@functools.lru_cache(maxsize=1)
def _registry() -> Any:
    try:
        import pint
    except ImportError as err:  # pragma: no cover - a declared dependency
        raise ImportError(
            "unit= needs pint, which SurPyval depends on: pip install pint"
        ) from err
    registry = pint.UnitRegistry(on_redefinition="ignore")
    for definition in _COUNTS:
        registry.define(definition)
    return registry


@functools.lru_cache(maxsize=256)
def _parse(text: str) -> Any:
    """The pint unit ``text`` names, or None. Tried as written, then in
    lower case for a word written with capitals (``"Hours"``): pint's
    names are case sensitive, and ``"Mm"`` and ``"mm"`` are different
    units, so only a failed spelling is retried."""
    registry = _registry()
    for candidate in dict.fromkeys((text, text.lower())):
        try:
            parsed = registry.parse_units(candidate)
        except Exception:
            continue
        if str(parsed) != "dimensionless":
            return parsed
    return None


def normalise_unit(unit: Any) -> str | None:
    """``unit`` as a model stores it: pint's canonical name for a unit it
    knows, else the text as given (stripped, with runs of spaces made
    one); None for None (unknown).

    Examples
    --------
    >>> from surpyval.utils.units import normalise_unit
    >>> normalise_unit("hrs"), normalise_unit("Hours"), normalise_unit("h")
    ('hour', 'hour', 'hour')
    >>> normalise_unit("kilometres"), normalise_unit("cycles")
    ('kilometer', 'cycle')
    >>> normalise_unit("  pump   starts ")
    'pump starts'
    """
    if unit is None:
        return None
    if hasattr(unit, "dimensionality") and not isinstance(unit, str):
        # A pint unit (or a quantity's unit) passed directly
        unit = str(getattr(unit, "units", unit))
    if not isinstance(unit, str):
        raise TypeError(
            "unit must be text naming the unit of the data, such as "
            f"'hours', 'km' or 'cycles'; got {type(unit).__name__}."
        )
    text = " ".join(unit.split())
    if not text:
        raise ValueError(
            "unit must name the unit of the data, such as 'hours', 'km' "
            "or 'cycles'; leave it out (None) when the unit is unknown."
        )
    parsed = _parse(text)
    return text if parsed is None else str(parsed)


def unit_symbol(unit: str | None) -> str | None:
    """The short form of ``unit`` for an axis: pint's symbol (``"h"``,
    ``"km"``, ``"kWh"``, ``"°C"``; ``"yr"`` for a year), the plural of a
    count (``"cycles"``), or the text as stored for a unit pint does not
    know.

    Examples
    --------
    >>> from surpyval.utils.units import unit_symbol
    >>> unit_symbol("hour"), unit_symbol("kilometer"), unit_symbol("cycle")
    ('h', 'km', 'cycles')
    >>> unit_symbol("pump starts")
    'pump starts'
    """
    if unit is None:
        return None
    parsed = _parse(unit)
    if parsed is None:
        return unit
    name = str(parsed)
    if name in _SYMBOLS:
        return _SYMBOLS[name]
    if _is_count(parsed):
        return name + "s"
    return f"{parsed:~P}"


def _is_count(parsed: Any) -> bool:
    """Whether the pint unit ``parsed`` is one of the counts of
    ``_COUNTS`` (or a multiple of one: a kilocycle)."""
    dims = dict(parsed.dimensionality)
    return len(dims) == 1 and next(iter(dims)).strip("[]") in _COUNT_NAMES


def axis_label(unit: str | None, default: str = "Time") -> str:
    """The label of an axis that measures the data's ``unit``:
    ``default`` alone where the unit is unknown, and otherwise named by
    what the unit measures -- ``"Time (h)"``, ``"Distance (km)"``, and
    ``"Usage (cycles)"`` for a count, an energy or a unit pint does not
    know (``"Usage (pump starts)"``).

    Examples
    --------
    >>> from surpyval.utils.units import axis_label
    >>> axis_label(None), axis_label("hour"), axis_label("kilometer")
    ('Time', 'Time (h)', 'Distance (km)')
    >>> axis_label("cycle"), axis_label("pump starts")
    ('Usage (cycles)', 'Usage (pump starts)')
    """
    if unit is None:
        return default
    parsed = _parse(unit)
    what = "Usage"
    if parsed is not None:
        dims = dict(parsed.dimensionality)
        if dims == {"[time]": 1}:
            what = "Time"
        elif dims == {"[length]": 1}:
            what = "Distance"
    return f"{what} ({unit_symbol(unit)})"


def label_axes(ax: Any, unit: str | None, ylabel: str) -> None:
    """Label a plot's axes where they have no label yet: the time axis by
    :func:`axis_label` and the other ``ylabel``, so a caller's own labels
    (or an earlier plot's on the same axes) stay."""
    if not ax.get_xlabel():
        ax.set_xlabel(axis_label(unit))
    if not ax.get_ylabel():
        ax.set_ylabel(ylabel)


def set_unit(model: Any, unit: Any) -> Any:
    """Store ``unit`` (normalised) on ``model`` and return it."""
    model.unit = normalise_unit(unit)
    return model


R = TypeVar("R")


def takes_unit(method: Callable[..., R]) -> Callable[..., R]:
    """Give a ``fit``, ``fit_from_df`` or ``from_params`` an optional
    keyword ``unit=``, stored on the model it returns (see
    :func:`normalise_unit`; checked before the fit runs). Its signature
    shows the keyword, and its return type is kept for a type checker
    (which cannot be told of one added keyword, so it is told of any).
    Applied to a method that already has the keyword, it is returned as
    it is."""
    if getattr(method, "_takes_unit", False):
        return method
    signature = inspect.signature(method)
    if "unit" in signature.parameters:
        return method

    @functools.wraps(method)
    def wrapper(*args: Any, unit: Any = None, **kwargs: Any) -> R:
        canonical = normalise_unit(unit)
        model = method(*args, **kwargs)
        if canonical is not None and model is not None:
            held = getattr(model, "unit", None)
            if held is not None and held != canonical:
                # The data (a SurpyvalData or EventHistoryData with a
                # unit) gave the model its unit (#945)
                raise ValueError(
                    f"The data's ages are in {held!r}, but unit={unit!r} "
                    "was given to the fit: a model is in the unit of its "
                    "data (no conversion is made), so leave unit= out, or "
                    "convert the data."
                )
            try:
                setattr(model, "unit", canonical)
            except AttributeError:
                pass
        return model

    params = list(signature.parameters.values())
    keyword = inspect.Parameter(
        "unit", inspect.Parameter.KEYWORD_ONLY, default=None
    )
    at = len(params)
    if params and params[-1].kind is inspect.Parameter.VAR_KEYWORD:
        at -= 1
    params.insert(at, keyword)
    setattr(wrapper, "__signature__", signature.replace(parameters=params))
    setattr(wrapper, "_takes_unit", True)
    return wrapper
