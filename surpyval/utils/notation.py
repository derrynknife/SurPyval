"""A small notation for writing survival data the way it is described.

Each observation is one object, and observations combine with ``*`` (a
count) and ``/`` (a truncation window)::

    >>> from surpyval.utils.notation import (
    ...     IC, LC, RC, O, Observations, Truncation)
    >>> data = Observations(
    ...     [3 * O(10), RC(20), LC(3), IC(5, 10) / Truncation(0, 40)])
    >>> data
    Observations, 4 rows (6 units):
      3 × failed at 10
      1 × still running at 20
      1 × failed before 3
      1 × failed between 5 and 10, observed in (0, 40]

``O`` failed at a time, ``RC`` still running at a time (right censored),
``LC`` failed some time before a time (left censored) and ``IC`` failed
some time between two times (interval censored). ``n * obs`` is ``n``
units alike (a whole number), and ``obs / Truncation(tl, tr)`` says the
unit was only observed within ``(tl, tr]``: late entry at ``tl``, right
truncation at ``tr``, the same window as everywhere else in SurPyval.
The rules are :class:`~surpyval.SurpyvalData`'s, checked when the
observation is built: a time is finite, a point lies in ``(tl, tr]``
(a suspension strictly before ``tr``), and an interval lies inside its
window (``tl <= lower``, ``upper <= tr``). An interval open at one end is
one-sided censoring, as in ``SurpyvalData``: ``IC(5, inf)`` is
``RC(5)`` and ``IC(-inf, 5)`` is ``LC(5)``.

The same data can be written as text, one entry per observation, in the
shorthand of R's ``Surv`` printout::

    >>> Observations.parse("3*10, 20+, <3, [5, 10] | (0, 40]")
    Observations, 4 rows (6 units):
      3 × failed at 10
      1 × still running at 20
      1 × failed before 3
      1 × failed between 5 and 10, observed in (0, 40]

``"10"`` failed at 10, ``"20+"`` still running at 20, ``"<3"`` (or
``"3-"``) failed before 3, ``"[5, 10]"`` failed between 5 and 10,
``"3*10"`` (or ``"3×10"``) three units that failed at 10, and
``" | (tl, tr]"`` after an entry its observation window (``inf`` and
``-inf`` for an open side). :meth:`Observations.text` writes the text
back, which ``parse`` reads as the same observations.

``Observations.surpyval_data()`` (or ``SurpyvalData.from_notation``)
gives the :class:`~surpyval.SurpyvalData` every fitter takes, and every
fitter that takes a ``SurpyvalData`` takes an ``Observations`` too
(``Weibull.fit(Observations.parse("10, 20+"))``);
``SurpyvalData.to_notation()`` goes back. ``Observations.xcnt()`` gives
the arrays themselves (``x``, ``c``, ``n``, ``tl``, ``tr``), with
censoring coded as everywhere in SurPyval: 0 failed, 1 right censored,
-1 left censored, 2 interval censored.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from numbers import Number
from typing import Any, SupportsFloat, cast

import numpy as np

__all__ = [
    "Truncation",
    "SurvivalObservation",
    "Observed",
    "RightCensored",
    "LeftCensored",
    "IntervalCensored",
    "O",
    "RC",
    "LC",
    "IC",
    "Observations",
    "Item",
    "Items",
]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _time(value: object, what: str) -> float:
    """A number of time (infinities allowed), or a ValueError naming
    ``what``."""
    if isinstance(value, bool) or not isinstance(value, Number):
        raise ValueError(f"{what} must be a number, got {value!r}.")
    out = float(cast(SupportsFloat, value))
    if math.isnan(out):
        raise ValueError(f"{what} must be a number, not NaN.")
    return out


def _finite_time(value: object) -> float:
    """The time of a point observation: a finite number."""
    out = _time(value, "The time")
    if not math.isfinite(out):
        raise ValueError(
            f"The time must be finite, got {_fmt(out)}: an infinite time "
            "carries no information. For a unit still running at x write "
            "RC(x); for one failed some time before x, LC(x)."
        )
    return out


def _count(value: object) -> int | None:
    """A count: a whole number, 1 or more; None for what is not a number
    (so the operator can say NotImplemented)."""
    if isinstance(value, bool) or not isinstance(value, Number):
        return None
    k = float(cast(SupportsFloat, value))
    if not math.isfinite(k) or k < 1 or k % 1:
        raise ValueError(
            f"A count must be a whole number, 1 or more, got {_fmt(k)}: it "
            "says how many units were alike. For a weight (each unit "
            "standing for part of a population), give w= to SurpyvalData "
            "instead."
        )
    return int(k)


def _fmt(value: float) -> str:
    """A time for people to read (six significant figures)."""
    return (
        f"{value:g}"
        if math.isfinite(value)
        else ("inf" if value > 0 else "-inf")
    )


def _exact(value: float) -> str:
    """A time as text that reads back as the same float."""
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    out = repr(float(value))
    return out[:-2] if out.endswith(".0") else out


def _plural(k: float, word: str) -> str:
    return f"{k:g} {word}{'' if k == 1 else 's'}"


def _checked_covariates(value: Any) -> dict | None:
    """Covariates ``{name: value}`` as a dict of finite floats, in the
    order given; None where there are none."""
    if value is None:
        return None
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict()  # a pandas Series
    if not isinstance(value, Mapping):
        raise ValueError(
            "covariates are named: give a dict {name: value}, e.g. "
            f"covariates={{'temp': 60}}; got {value!r}."
        )
    if not value:
        raise ValueError(
            "covariates={} names no covariate: leave it out, or give "
            "{name: value}."
        )
    out = {}
    for name, v in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"A covariate's name must be text; got {name!r}.")
        if isinstance(v, (bool, np.bool_)) or not isinstance(
            v, (Number, np.number)
        ):
            raise ValueError(
                f"Covariate {name!r} must be a number, got {v!r} (code a "
                "category as 0/1 columns)."
            )
        number = float(v)  # type: ignore[arg-type]
        if not math.isfinite(number):
            raise ValueError(
                f"Covariate {name!r} must be finite, got {_fmt(number)}."
            )
        out[name] = number
    return out


def _covariate_matrix(
    covariates: list, labels: list[str]
) -> tuple[list | None, np.ndarray | None]:
    """The covariate names and one row of values per entry (an
    observation or an item, named by ``labels``), refused where they do
    not all have the same names; ``(None, None)`` where none has any."""
    if all(c is None for c in covariates):
        return None, None
    without = [lab for lab, c in zip(labels, covariates) if c is None]
    if without:
        raise ValueError(
            "Covariates are given for some and not others: "
            + ", ".join(without[:5])
            + (f" and {len(without) - 5} more" if len(without) > 5 else "")
            + f" {'has' if len(without) == 1 else 'have'} none. Give every "
            "one the same covariates, or none."
        )
    names = list(covariates[0])
    for label, cov in zip(labels, covariates):
        missing = [n for n in names if n not in cov]
        extra = [n for n in cov if n not in names]
        if missing or extra:
            said = []
            if missing:
                said.append("is missing " + ", ".join(map(repr, missing)))
            if extra:
                said.append(
                    "has "
                    + ", ".join(map(repr, extra))
                    + f", which {labels[0]} has not"
                )
            raise ValueError(
                "Every one needs the same covariates: "
                f"{labels[0]} has {', '.join(map(repr, names))}, and "
                f"{label} " + " and ".join(said) + ". Give each the same "
                "names."
            )
    return names, np.array([[c[n] for n in names] for c in covariates])


# ---------------------------------------------------------------------------
# The window and the observations
# ---------------------------------------------------------------------------


class Truncation:
    """The window ``(left, right]`` a unit was observed in.

    ``left`` is the time it entered observation (late entry, left
    truncation) and ``right`` the time observation stopped for good
    (right truncation). ``None`` (or ``-inf`` / ``inf``) means no
    truncation on that side.
    """

    __slots__ = ("left", "right")

    def __init__(
        self, left: float | None = None, right: float | None = None
    ) -> None:
        lo = (
            -math.inf
            if left is None
            else _time(left, "The left edge of a truncation")
        )
        hi = (
            math.inf
            if right is None
            else _time(right, "The right edge of a truncation")
        )
        if lo >= hi:
            raise ValueError(
                f"The left edge of a truncation ({_fmt(lo)}) must be below "
                f"the right edge ({_fmt(hi)})."
            )
        if lo == math.inf or hi == -math.inf:
            raise ValueError(
                "A truncation's left edge cannot be inf, nor its right "
                "edge -inf: no unit could be seen in that window."
            )
        self.left: float = lo
        self.right: float = hi

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Truncation) and (self.left, self.right) == (
            other.left,
            other.right,
        )

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __repr__(self) -> str:
        return f"Truncation({_fmt(self.left)}, {_fmt(self.right)})"

    def text(self) -> str:
        """The window as ``parse`` reads it, ``"(2, 50]"``."""
        return f"({_exact(self.left)}, {_exact(self.right)}]"

    def _words(self) -> str:
        return f"({_fmt(self.left)}, {_fmt(self.right)}]"


# Why a time outside its window is refused: the same reasons SurpyvalData
# gives (surpyval.utils.data_formats), so the two never disagree.
_AT_LEFT_EDGE = (
    "a value at or before its own left truncation time has no observation "
    "window: the window (tl, tr] starts just after tl"
)
_INTERVAL_BELOW = (
    "an interval cannot start below its own left truncation time. In "
    "inspection data a failure found at the first inspection after the "
    "records begin starts at its last good inspection, before the records "
    "do: its left truncation time is that inspection (so tl <= lower), not "
    "the date the records begin. Do not move the interval's start up to "
    "tl, which biases the fit"
)
_ABOVE_RIGHT = (
    "a value after its right truncation time could not have been seen: "
    "the window (tl, tr] ends at tr"
)
_SUSPENDED_AT_RIGHT = (
    "a right censored value must be strictly less than its right "
    "truncation time: censored at tr, the event would have to lie after "
    "tr and at or before it"
)


class SurvivalObservation:
    """One observation (or ``n`` alike): what happened, how many, and the
    window it was observed in. Use :class:`Observed`,
    :class:`RightCensored`, :class:`LeftCensored` or
    :class:`IntervalCensored` (``O``, ``RC``, ``LC``, ``IC``)."""

    __slots__ = ("n", "truncation", "event_type", "covariates", "terminal")
    #: SurPyval's censoring code for this kind of observation.
    code: int = 0

    def __init__(
        self,
        event_type: Any = None,
        covariates: Any = None,
        terminal: bool = False,
    ) -> None:
        self.n: int = 1
        self.truncation: Truncation | None = None
        #: What failed (a failure mode or cause), or None.
        self.event_type: Any = self._checked_type(event_type)
        #: The unit's fixed covariates, ``{name: value}``, or None.
        self.covariates: dict | None = _checked_covariates(covariates)
        #: Whether this event ended its unit's history (in an
        #: :class:`Item`'s history: ``a = 1``).
        self.terminal: bool = self._checked_terminal(terminal)

    def _checked_type(self, value: Any) -> Any:
        from surpyval.utils.item_history import item_id, missing_type

        if missing_type(value):
            return None
        value = item_id(value)
        if isinstance(value, bool) or not isinstance(value, (str, Number)):
            raise ValueError(
                "event_type is what failed, as text or a number "
                f"(event_type='bearing'); got {value!r}."
            )
        if self.code == 1:
            raise ValueError(
                f"A unit {self._words()} has not failed, so it has no "
                "event_type: leave it out."
            )
        return value

    def _checked_terminal(self, value: Any) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"terminal is True or False; got {value!r}.")
        if value and self.code == 1:
            raise ValueError(
                f"RC({_fmt(self._bounds()[0])}) is the end of watching, not "
                "an event, so it cannot end the history (terminal=True): "
                "put terminal=True on the event that ended it, and leave "
                "the RC out."
            )
        return bool(value)

    def _extras(self) -> list[str]:
        """What the observation carries besides its time, in words."""
        out = []
        if self.event_type is not None:
            out.append(f"type {self.event_type!r}")
        if self.covariates:
            out.append(
                ", ".join(f"{k} = {v:g}" for k, v in self.covariates.items())
            )
        if self.terminal:
            out.append("ended its history")
        return out

    def _key(self) -> tuple:
        cov = (
            None if self.covariates is None else tuple(self.covariates.items())
        )
        return (
            type(self),
            self._bounds(),
            self.n,
            self.truncation,
            self.event_type,
            cov,
            self.terminal,
        )

    # The time(s) this observation records, for the window check and xcnt.
    def _bounds(self) -> tuple[float, float]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _copy(self) -> SurvivalObservation:
        new = object.__new__(type(self))
        for cls in type(self).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                setattr(new, slot, getattr(self, slot))
        return new

    def _outside(self, t: Truncation) -> str | None:
        """Why this observation cannot lie in the window ``t``; None where
        it can. Points lie in ``(tl, tr]``; overridden for the rest."""
        x = self._bounds()[0]
        if x <= t.left:
            return _AT_LEFT_EDGE
        if x > t.right:
            return _ABOVE_RIGHT
        return None

    def __truediv__(self, t: Truncation) -> SurvivalObservation:
        if not isinstance(t, Truncation):
            return NotImplemented
        reason = self._outside(t)
        if reason is not None:
            raise ValueError(
                f"{self._words()} lies outside its observation window "
                f"{t._words()}: {reason}. Check the times or the window."
            )
        new = self._copy()
        new.truncation = t
        return new

    def __mul__(self, n: object) -> SurvivalObservation:
        k = _count(n)
        if k is None:
            return NotImplemented
        new = self._copy()
        new.n = self.n * k
        return new

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SurvivalObservation):
            return False
        mine, theirs = self._key(), other._key()
        # Covariates compare as dicts: the order they were given in is not
        # part of what they say.
        return (
            mine[:5] == theirs[:5]
            and self.covariates == other.covariates
            and self.terminal == other.terminal
        )

    def __hash__(self) -> int:
        key = self._key()
        cov = None if key[5] is None else frozenset(key[5])
        return hash((*key[:5], cov, key[6]))

    def _words(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _entry(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def text(self) -> str:
        """The observation as ``Observations.parse`` reads it:
        ``"3*10 | (2, 50]"``. Its event type, covariates and terminal flag
        are not part of the text (they stay in Python)."""
        n = f"{self.n}*" if self.n != 1 else ""
        t = f" | {self.truncation.text()}" if self.truncation else ""
        return f"{n}{self._entry()}{t}"

    def _line(self) -> str:
        t = (
            f", observed in {self.truncation._words()}"
            if self.truncation
            else ""
        )
        extras = self._extras()
        more = f" ({'; '.join(extras)})" if extras else ""
        return f"{self._words()}{t}{more}"

    def __repr__(self) -> str:
        n = f"{self.n} × " if self.n != 1 else ""
        return f"{n}{self._line()}"


class _Point(SurvivalObservation):
    __slots__ = ("x",)

    def __init__(
        self,
        x: float,
        *,
        event_type: Any = None,
        covariates: Any = None,
        terminal: bool = False,
    ) -> None:
        self.x = _finite_time(x)
        super().__init__(event_type, covariates, terminal)

    def _bounds(self) -> tuple[float, float]:
        return (self.x, self.x)


class Observed(_Point):
    """A unit that failed at ``x``.

    Every observation takes, by keyword, ``event_type`` (what failed: a
    failure mode or cause, for the competing-risks models; not on a
    suspension), ``covariates`` (``{name: value}``, the unit's fixed
    covariates) and ``terminal`` (in an :class:`Item`'s history, the
    event that ended it for good)."""

    __slots__ = ()
    code = 0

    def _words(self) -> str:
        return f"failed at {_fmt(self.x)}"

    def _entry(self) -> str:
        return _exact(self.x)


class RightCensored(_Point):
    """A unit still running at ``x`` (a suspension)."""

    __slots__ = ()
    code = 1

    def _outside(self, t: Truncation) -> str | None:
        if self.x <= t.left:
            return _AT_LEFT_EDGE
        if self.x >= t.right:
            return _SUSPENDED_AT_RIGHT
        return None

    def _words(self) -> str:
        return f"still running at {_fmt(self.x)}"

    def _entry(self) -> str:
        return f"{_exact(self.x)}+"


class LeftCensored(_Point):
    """A unit that failed some time before ``x``."""

    __slots__ = ()
    code = -1

    def _words(self) -> str:
        return f"failed before {_fmt(self.x)}"

    def _entry(self) -> str:
        return f"<{_exact(self.x)}"


class IntervalCensored(SurvivalObservation):
    """A unit that failed some time between ``left`` and ``right``.

    An interval open at one end is one-sided censoring, as
    :class:`~surpyval.SurpyvalData` reads it: ``IntervalCensored(5, inf)``
    is ``RightCensored(5)`` and ``IntervalCensored(-inf, 5)`` is
    ``LeftCensored(5)``; ``(-inf, inf)`` carries no information and is
    refused."""

    __slots__ = ("left", "right")
    code = 2

    def __new__(cls, left: float, right: float, **extra: Any) -> Any:
        lo = _time(left, "The start of an interval")
        hi = _time(right, "The end of an interval")
        if lo >= hi:
            raise ValueError(
                f"An interval's start ({_fmt(lo)}) must be below its "
                f"end ({_fmt(hi)}); "
                "for a failure seen at one time use O(x)."
            )
        if lo == -math.inf and hi == math.inf:
            raise ValueError(
                "An interval from -inf to inf carries no information: give "
                "the times the unit was last seen working and first seen "
                "failed."
            )
        if hi == math.inf:
            return RightCensored(lo, **extra)
        if lo == -math.inf:
            return LeftCensored(hi, **extra)
        return super().__new__(cls)

    def __init__(
        self,
        left: float,
        right: float,
        *,
        event_type: Any = None,
        covariates: Any = None,
        terminal: bool = False,
    ) -> None:
        self.left: float = float(left)
        self.right: float = float(right)
        super().__init__(event_type, covariates, terminal)

    def _bounds(self) -> tuple[float, float]:
        return (self.left, self.right)

    def _outside(self, t: Truncation) -> str | None:
        if self.left < t.left:
            return _INTERVAL_BELOW
        if self.right > t.right:
            return _ABOVE_RIGHT
        return None

    def _words(self) -> str:
        return f"failed between {_fmt(self.left)} and {_fmt(self.right)}"

    def _entry(self) -> str:
        return f"[{_exact(self.left)}, {_exact(self.right)}]"


O = Observed
RC = RightCensored
LC = LeftCensored
IC = IntervalCensored


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_COUNT = r"(?:(?P<n>\d+(?:\.\d+)?)\s*[*×x]\s*)?"
_NUM = r"[-+]?(?:inf|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
_PATTERNS = (
    (
        re.compile(rf"^{_COUNT}\[\s*(?P<a>{_NUM})\s*,\s*(?P<b>{_NUM})\s*\]$"),
        "ic",
    ),
    (re.compile(rf"^{_COUNT}<\s*(?P<a>{_NUM})$"), "lc"),
    (re.compile(rf"^{_COUNT}(?P<a>{_NUM})\s*-$"), "lc"),
    (re.compile(rf"^{_COUNT}(?P<a>{_NUM})\s*\+$"), "rc"),
    (re.compile(rf"^{_COUNT}(?P<a>{_NUM})$"), "o"),
)
_WINDOW = re.compile(rf"^\(\s*(?P<a>{_NUM})\s*,\s*(?P<b>{_NUM})\s*\]$")
_POINTS: dict[str, type[_Point]] = {
    "o": Observed,
    "rc": RightCensored,
    "lc": LeftCensored,
}
_HOW = (
    "Write 10 (failed at 10), 20+ (still running at 20), <3 or 3- (failed "
    "before 3), [5, 10] (failed between 5 and 10), 3*10 for three alike, "
    "and add | (tl, tr] for the window an entry was observed in "
    "(12 | (2, 50], or (2, inf] for late entry alone)."
)


def _split(text: str) -> list[str]:
    """Split on commas, semicolons and new lines, keeping the comma of an
    interval ``[a, b]`` or a window ``(a, b]`` inside it."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch in ",\n;" and depth <= 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_window(entry: str, text: str) -> Truncation:
    m = _WINDOW.match(text.strip())
    if not m:
        raise ValueError(
            f"Can't read the window in {entry!r}: write it as (tl, tr], "
            "e.g. 12 | (2, 50], with inf or -inf for an open side."
        )
    return Truncation(float(m.group("a")), float(m.group("b")))


def _parse_one(entry: str) -> SurvivalObservation:
    body, bar, window = entry.partition("|")
    body = body.strip()
    for pattern, kind in _PATTERNS:
        m = pattern.match(body)
        if not m:
            continue
        a = float(m.group("a"))
        out: SurvivalObservation = (
            IntervalCensored(a, float(m.group("b")))
            if kind == "ic"
            else _POINTS[kind](a)
        )
        if m.group("n"):
            out = out * float(m.group("n"))
        return out / _parse_window(entry, window) if bar else out
    raise ValueError(f"Can't read {entry!r}. {_HOW}")


def _unit_into(unit: str | None, kwargs: dict, what: str) -> None:
    """Put the notation's ``unit`` into the ``SurpyvalData`` keywords,
    refusing a different ``unit=`` given there."""
    if unit is None:
        return
    given = kwargs.get("unit")
    if given is not None and _checked_unit(given) != unit:
        raise ValueError(
            f"The {what} are in {unit!r} and unit= says {given!r}: leave "
            "unit= out, or convert the times first."
        )
    kwargs["unit"] = unit


def _checked_unit(unit: Any) -> str | None:
    from surpyval.utils.units import normalise_unit

    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise ValueError(
            "'unit' is the name of the unit the times are in ('hours', "
            f"'km', 'cycles'); got {unit!r}."
        )
    return normalise_unit(unit)


# ---------------------------------------------------------------------------
# A set of observations
# ---------------------------------------------------------------------------


class Observations:
    """A list of observations, ready for any fitter.

    ``Observations([3 * O(10), RC(20)], unit="hours")``; ``obs + other``
    joins two; ``obs / Truncation(tl, tr)`` gives every observation the
    same window (one that already has its own window keeps it);
    ``surpyval_data()`` and ``xcnt()`` give what the fitters take, and
    every fitter that takes a :class:`~surpyval.SurpyvalData` takes an
    ``Observations`` as it is. ``text()`` writes it as text, which
    :meth:`parse` reads back. Two are equal where they hold the same
    observations in the same order, with the same unit.
    """

    __slots__ = ("observations", "unit")

    def __init__(
        self,
        observations: Iterable[SurvivalObservation] = (),
        unit: str | None = None,
    ) -> None:
        observations = list(observations)
        wrong = [
            o for o in observations if not isinstance(o, SurvivalObservation)
        ]
        if wrong:
            raise ValueError(
                "Observations takes O, RC, LC and IC observations; "
                f"got {wrong[0]!r}. "
                "Use Observations.parse() to read text."
            )
        self.observations = observations
        #: The unit of the times (``"hours"``), kept with the data; None
        #: where unknown.
        self.unit: str | None = _checked_unit(unit)

    @classmethod
    def parse(cls, text: str, unit: str | None = None) -> Observations:
        """Read observations from text: entries separated by commas,
        semicolons or new lines (``"3*10, 20+, <3, [5, 10]"``), each
        with its window after a bar where it has one
        (``"12 | (2, 50]"``). ``unit`` is the unit of the times.

        Examples
        --------
        >>> from surpyval.utils.notation import Observations
        >>> obs = Observations.parse("2*10, 20+, [5, inf], 12 | (2, 50]")
        >>> obs.text()
        '2*10, 20+, 5+, 12 | (2, 50]'
        >>> Observations.parse(obs.text()) == obs
        True
        """
        if not isinstance(text, str):
            raise ValueError(
                'Observations.parse takes a string, e.g. "10, 20+, [5, 10]".'
            )
        return cls((_parse_one(e) for e in _split(text)), unit=unit)

    @classmethod
    def _from_surpyval_data(cls, data: Any) -> Observations:
        """One observation per row of ``data`` (a ``SurpyvalData``), with
        its count and window; see ``SurpyvalData.to_notation``."""
        x = np.asarray(data.x, dtype=float)
        lo, hi = (x[:, 0], x[:, 1]) if x.ndim == 2 else (x, x)
        out: list[SurvivalObservation] = []
        names = data.covariate_names
        Z = None if data.Z is None else np.asarray(data.Z, dtype=float)
        e = data.e
        for k, code in enumerate(np.asarray(data.c).tolist()):
            extra: dict[str, Any] = {}
            if Z is not None:
                extra["covariates"] = dict(zip(names, Z[k].tolist()))
            if e is not None and e[k] is not None:
                extra["event_type"] = e[k]
            obs: SurvivalObservation
            if code == 2:
                obs = IntervalCensored(lo[k], hi[k], **extra)
            else:
                kind = {0: "o", 1: "rc", -1: "lc"}[code]
                obs = _POINTS[kind](lo[k], **extra)
            n = int(data.n[k])
            if n != 1:
                obs = obs * n
            tl, tr = float(data.tl[k]), float(data.tr[k])
            if math.isfinite(tl) or math.isfinite(tr):
                obs = obs / Truncation(tl, tr)
            out.append(obs)
        return cls(out, unit=getattr(data, "unit", None))

    def text(self) -> str:
        """The observations as text, one entry each, which :meth:`parse`
        reads back as the same observations (exactly: every time is
        written in full). The unit is not part of the text; give it to
        ``parse`` again. Nor are event types, covariates and terminal
        flags, which the text leaves out so that it stays readable: they
        stay in Python, and ``parse(obs.text())`` is then the observations
        without them.

        Examples
        --------
        >>> from surpyval.utils.notation import O, RC, Observations
        >>> obs = Observations([O(10, event_type="bearing"), RC(20)])
        >>> obs.text()
        '10, 20+'
        """
        return ", ".join(o.text() for o in self.observations)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observations):
            return NotImplemented
        return (
            self.observations == other.observations and self.unit == other.unit
        )

    __hash__ = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[SurvivalObservation]:
        return iter(self.observations)

    def __getitem__(self, index: Any) -> Any:
        got = self.observations[index]
        return (
            Observations(got, unit=self.unit)
            if isinstance(index, slice)
            else got
        )

    def __add__(self, other: object) -> Observations:
        if isinstance(other, SurvivalObservation):
            other = Observations([other], unit=self.unit)
        if not isinstance(other, Observations):
            return NotImplemented
        units = {u for u in (self.unit, other.unit) if u is not None}
        if len(units) > 1:
            raise ValueError(
                f"These observations are in {self.unit!r} and those in "
                f"{other.unit!r}: convert one set's times first (and give "
                "it the other's unit), then join them."
            )
        return Observations(
            self.observations + other.observations,
            unit=next(iter(units), None),
        )

    def __truediv__(self, t: Truncation) -> Observations:
        if not isinstance(t, Truncation):
            return NotImplemented
        return Observations(
            (o if o.truncation else o / t for o in self.observations),
            unit=self.unit,
        )

    @property
    def units(self) -> int:
        """How many units in all (the counts added up)."""
        return int(sum(o.n for o in self.observations))

    def xcnt(self) -> dict[str, np.ndarray]:
        """The arrays SurPyval takes, one row per observation: ``x`` (two
        columns, lower and upper, equal except for intervals), ``c``, ``n``,
        ``tl`` and ``tr`` (``-inf`` / ``inf`` where there is no truncation).
        Every array has one entry per observation, so a truncation always
        sits on its own row."""
        if not self.observations:
            raise ValueError("There are no observations to convert.")
        x = np.array([o._bounds() for o in self.observations], dtype=float)
        c = np.array([o.code for o in self.observations], dtype=int)
        n = np.array([o.n for o in self.observations], dtype=int)
        tl = np.array(
            [
                o.truncation.left if o.truncation else -np.inf
                for o in self.observations
            ]
        )
        tr = np.array(
            [
                o.truncation.right if o.truncation else np.inf
                for o in self.observations
            ]
        )
        return {"x": x, "c": c, "n": n, "tl": tl, "tr": tr}

    def surpyval_data(self, **kwargs: Any) -> Any:
        """The :class:`~surpyval.SurpyvalData` every fitter takes
        (``Weibull.fit(obs.surpyval_data())``; ``Weibull.fit(obs)`` does
        this itself). ``kwargs`` go to ``SurpyvalData`` (``unit=``, for
        example, where the observations have none)."""
        from surpyval.utils.surpyval_data import SurpyvalData

        _unit_into(self.unit, kwargs, "observations")
        ended = [k for k, o in enumerate(self.observations) if o.terminal]
        if ended:
            raise ValueError(
                f"Observation {ended[0]} "
                f"({self.observations[ended[0]]._words()}) "
                "says it ended its unit's history (terminal=True), which "
                "only a history of several events can say: in "
                "Observations each failure is a unit's one lifetime. Write "
                "the unit's events as an Item, Item('P1', O(120), O(800, "
                "terminal=True)), in Items([...])."
            )
        a = self.xcnt()
        x = [
            row[0] if code != 2 else list(row)
            for row, code in zip(a["x"], a["c"])
        ]
        truncated = np.isfinite(a["tl"]).any() or np.isfinite(a["tr"]).any()
        labels = [
            f"observation {k} ({o._words()})"
            for k, o in enumerate(self.observations)
        ]
        names, Z = _covariate_matrix(
            [o.covariates for o in self.observations], labels
        )
        types = [o.event_type for o in self.observations]
        e = None
        if any(t is not None for t in types):
            e = np.empty(len(types), dtype=object)
            e[:] = types
        extra: dict[str, Any] = {}
        if Z is not None:
            extra.update(Z=Z, covariate_names=names)
        if e is not None:
            extra["e"] = e
        return SurpyvalData(
            x=x,
            c=a["c"],
            n=a["n"],
            tl=a["tl"] if truncated else None,
            tr=a["tr"] if truncated else None,
            **extra,
            **kwargs,
        )

    def __repr__(self) -> str:
        rows = len(self.observations)
        head = (
            f"Observations, {_plural(rows, 'row')} "
            f"({_plural(self.units, 'unit')})"
        )
        if self.unit is not None:
            head += f", times in {self.unit}"
        if not rows:
            return head
        shown = self.observations[:20]
        lines = [f"  {o.n} × {o._line()}" for o in shown]
        if rows > len(shown):
            lines.append(f"  … and {rows - len(shown)} more")
        return head + ":\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Histories: several events per unit
# ---------------------------------------------------------------------------


def _event_key(o: SurvivalObservation) -> tuple:
    """Time order within a history: by the event's upper end, the end of
    watching (RC) after any event at its age."""
    lo, hi = o._bounds()
    return (hi, o.code == 1, lo)


def _checked_readings(item: Any, value: Any) -> dict | None:
    """Readings as ``{name: {age: value}}``, ``None`` naming the one
    unnamed series; None where there are none."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f"Item {item!r}: readings are a dict, {{age: value}} for one "
            "series or {name: {age: value}} for several named ones "
            "(readings={'vibration': {0: 1.2, 100: 1.9}})."
        )
    nested = [isinstance(v, Mapping) for v in value.values()]
    if all(nested):
        bad = [k for k in value if not isinstance(k, str) or not k.strip()]
        if bad:
            raise ValueError(
                f"Item {item!r}: a named series of readings needs text for "
                f"its name; got {bad[0]!r}."
            )
        series = {name: dict(v) for name, v in value.items()}
    elif not any(nested):
        series = {None: dict(value)}
    else:
        raise ValueError(
            f"Item {item!r}: readings mix named series ({{name: {{age: "
            "value}}}) and single readings ({age: value}): give one or the "
            "other."
        )
    from surpyval.utils.measurements import Measurements

    for name, readings in series.items():
        # Checked now, where the item was written, with the
        # Measurements' own rules
        Measurements({item: readings}, name=name)
    return series


class Item:
    """One unit's history: its events in time order, the window it was
    watched in, and what describes it.

    ``Item("P1", O(120), O(310), RC(1000))`` failed at 120 and 310 and was
    still running at 1000, when watching stopped. The end of watching is
    an ``RC`` (at most one, after every event); without one, watching
    ended at the last event. ``Item(...) / Truncation(tl)`` is late entry:
    the unit came under observation at ``tl``, and nothing before is
    known. (Right truncation, ``Truncation(tl, tr)``, means the unit is in
    the data because it failed by ``tr``; it is never the end of watching,
    which is the ``RC``.) Events are given as ``O``, ``LC`` and ``IC``:

    - ``O(x)`` a failure at ``x``; ``3 * O(120)`` three failures at once.
    - ``IC(a, b)`` a failure found between two inspections;
      ``2 * IC(a, b)`` two.
    - ``LC(x)`` failures at unknown times before ``x``, the first look
      (at most one, before every other event); ``2 * LC(x)`` two.
    - ``O(800, terminal=True)`` the event that ended the history for good
      (scrapped, written off: ``a = 1``), the last event, with no ``RC``
      after it.

    An event takes its ``event_type`` (what failed); the unit's fixed
    covariates go on the item, ``covariates={name: value}``, not on its
    events.

    Parameters
    ----------
    id : hashable
        The unit's id.
    *events : O, RC, LC or IC
        What happened to it, in any order (kept in time order).
    covariates : dict, optional
        ``{name: value}``, constant over its history.
    readings : dict, optional
        Readings taken on it over its life: ``{age: value}`` for one
        unnamed series (the joint models' ``markers``), or ``{name: {age:
        value}}`` for several named ones; they become the data's
        :class:`~surpyval.utils.measurements.Measurements`.
    time_varying_covariates : dict, StepSchedule or CovariatePath, optional
        Covariates that change over its history, passed through as they
        are (the fits expand them): ``StepSchedule.from_changepoints(...)``,
        a ``CovariatePath`` or ``{age: values}``, as
        :class:`~surpyval.utils.item_history.ItemHistory` takes them.

    The history is checked as an
    :class:`~surpyval.utils.item_history.ItemHistory` is (``.history``),
    and a collection of items, :class:`Items`, gives the data.

    Examples
    --------
    >>> from surpyval.utils.notation import Item, O, RC, Truncation
    >>> pump = Item("P1", O(120), O(310), RC(1000)) / Truncation(0)
    >>> pump
    Item 'P1', observed in (0, inf]:
      failed at 120
      failed at 310
      still running at 1000
    >>> pump.text()
    'P1: 120, 310, 1000+ | (0, inf]'
    """

    __slots__ = (
        "id",
        "events",
        "truncation",
        "covariates",
        "readings",
        "time_varying_covariates",
        "history",
    )

    def __init__(
        self,
        id: Any,
        *events: SurvivalObservation,
        covariates: Any = None,
        readings: Any = None,
        time_varying_covariates: Any = None,
    ) -> None:
        from surpyval.utils.item_history import checked_id

        item = checked_id(id)
        self.id: Any = item
        for k, ev in enumerate(events):
            if not isinstance(ev, SurvivalObservation):
                raise ValueError(
                    f"Item {item!r}: its events are O, RC, LC and IC "
                    f"observations; event {k} is {ev!r}."
                    + (
                        " Give them one by one: Item(id, O(1), O(2))."
                        if isinstance(ev, (list, tuple, Observations))
                        else ""
                    )
                )
            if ev.truncation is not None:
                raise ValueError(
                    f"Item {item!r}: the window belongs to the item, not "
                    f"to one event ({ev!r}): write "
                    f"Item(...) / Truncation(tl)."
                )
            if ev.covariates is not None:
                raise ValueError(
                    f"Item {item!r}: covariates belong to the item, fixed "
                    f"over its history, not to one event ({ev!r}): write "
                    "Item(id, ..., covariates={...}); covariates that "
                    "change go in time_varying_covariates."
                )
        self.events: tuple = tuple(sorted(events, key=_event_key))
        self.truncation: Truncation | None = None
        self.covariates: dict | None = _checked_covariates(covariates)
        self.readings: dict | None = _checked_readings(item, readings)
        self.time_varying_covariates: Any = time_varying_covariates
        self._check_kinds()
        self.history: Any = self._history()

    def _check_kinds(self) -> None:
        item = self.id
        ends = [e for e in self.events if e.code == 1]
        if len(ends) > 1:
            raise ValueError(
                f"Item {item!r} has {len(ends)} ends of watching (RC): give "
                "one, the age it was last seen. Watched in separate "
                "stretches, give it as rows with gaps (SurpyvalData(..., "
                "i=..., tl=...))."
            )
        if ends and self.events[-1] is not ends[0]:
            later = self.events[-1]
            raise ValueError(
                f"Item {item!r}: the end of watching, {ends[0]!r}, is "
                f"before an event ({later!r}): RC is the age it was last "
                "seen, at or after every event."
            )
        firsts = [e for e in self.events if e.code == -1]
        if len(firsts) > 1:
            raise ValueError(
                f"Item {item!r} has {len(firsts)} LC events: LC(x) is the "
                "failures before the first look, at most one (2 * LC(x) "
                "for two failures before x); later ones found at an "
                "inspection are IC(last good, found)."
            )
        terminal = [e for e in self.events if e.terminal]
        if not terminal:
            return
        if len(terminal) > 1 or terminal[0].n > 1:
            raise ValueError(
                f"Item {item!r}: one event ends a history (terminal=True); "
                f"it has {sum(e.n for e in terminal)}."
            )
        if ends:
            raise ValueError(
                f"Item {item!r} ended at a terminal event "
                f"({terminal[0]!r}), so it was not watched after it: leave "
                f"out the RC ({ends[0]!r})."
            )
        if self.events[-1] is not terminal[0]:
            raise ValueError(
                f"Item {item!r}: the terminal event ({terminal[0]!r}) "
                f"ended its history, so it is its last event; "
                f"{self.events[-1]!r} comes after it."
            )

    def _history(self) -> Any:
        """The :class:`ItemHistory` of the events: its checks are the
        history's checks."""
        from surpyval.utils.item_history import ItemHistory

        observed, observed_types = [], []
        intervals, interval_counts, interval_types = [], [], []
        fields: dict[str, Any] = {}
        for ev in self.events:
            if ev.code == 0:
                observed += [ev._bounds()[0]] * ev.n
                observed_types += [ev.event_type] * ev.n
            elif ev.code == 2:
                intervals.append(list(ev._bounds()))
                interval_counts.append(ev.n)
                interval_types.append(ev.event_type)
            elif ev.code == -1:
                fields["left_censored"] = ev._bounds()[0]
                fields["left_censored_counts"] = ev.n
                if ev.event_type is not None:
                    fields["left_censored_event_types"] = ev.event_type
            else:
                fields["right_censored"] = ev._bounds()[0]
        typed = any(e.event_type is not None for e in self.events)
        if observed:
            fields["observed"] = observed
            if typed:
                fields["observed_event_types"] = observed_types
        if intervals:
            fields["interval_censored"] = intervals
            fields["interval_censored_counts"] = interval_counts
            if typed:
                fields["interval_censored_event_types"] = interval_types
        t = self.truncation
        if t is not None:
            if math.isfinite(t.left):
                fields["left_truncation"] = t.left
            if math.isfinite(t.right):
                fields["right_truncation"] = t.right
        return ItemHistory(
            self.id,
            covariates=self.covariates,
            time_varying_covariates=self.time_varying_covariates,
            **fields,
        )

    def __truediv__(self, t: Truncation) -> Item:
        if not isinstance(t, Truncation):
            return NotImplemented
        new = object.__new__(Item)
        for slot in Item.__slots__:
            setattr(new, slot, getattr(self, slot))
        new.truncation = t
        new.history = new._history()
        return new

    def _rows(self) -> list[tuple]:
        """``(x, c, n, event type, terminal)`` per row, in time order: a
        failure at an age repeated once per failure (each row one event),
        a count of failures in an interval or before a first look on one
        row."""
        rows = []
        for ev in self.events:
            lo, hi = ev._bounds()
            x: Any = [lo, hi] if ev.code == 2 else lo
            if ev.code == 0:
                rows += [(x, 0, 1, ev.event_type, ev.terminal)] * ev.n
            else:
                rows.append((x, ev.code, ev.n, ev.event_type, ev.terminal))
        return rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Item):
            return NotImplemented
        return (
            self.id == other.id
            and self.events == other.events
            and self.truncation == other.truncation
            and self.covariates == other.covariates
            and self.readings == other.readings
            and self.history == other.history
        )

    __hash__ = None  # type: ignore[assignment]

    def text(self) -> str:
        """The history as one line, ``"P1: 120, 310, [400, 450], 1000+ |
        (0, inf]"``, which :meth:`Items.parse` reads back as the same
        item where its id is text. Covariates, event types, terminal
        flags, readings and time-varying covariates are not part of the
        text (they stay in Python, so the text stays readable)."""
        name = str(self.id)
        if not name.strip() or any(ch in name for ch in ":\n;"):
            raise ValueError(
                f"Item {self.id!r}: an id written as text cannot be blank "
                "or hold ':', ';' or a new line; rename it first."
            )
        body = ", ".join(e.text() for e in self.events)
        t = f" | {self.truncation.text()}" if self.truncation else ""
        return f"{name}: {body}{t}"

    def __repr__(self) -> str:
        head = f"Item {self.id!r}"
        if self.truncation:
            head += f", observed in {self.truncation._words()}"
        lines = [f"  {e!r}" for e in self.events]
        if self.covariates:
            lines.append(
                "  covariates: "
                + ", ".join(f"{k} = {v:g}" for k, v in self.covariates.items())
            )
        if self.readings:
            for name, series in self.readings.items():
                label = "readings" if name is None else f"readings {name!r}"
                lines.append(f"  {label}: {_plural(len(series), 'reading')}")
        if self.time_varying_covariates is not None:
            lines.append("  time-varying covariates")
        return head + ":\n" + "\n".join(lines)


def _parse_item(line: str) -> Item:
    name, colon, body = line.partition(":")
    if not colon or not name.strip():
        raise ValueError(
            f"Can't read the history {line!r}: write the id, a colon and "
            "its events, e.g. P1: 120, 310, [400, 450], 1000+ | (0, inf]."
        )
    window = None
    if "|" in body:
        body, _, after = body.rpartition("|")
        if "|" in body:
            raise ValueError(
                f"Can't read the history {line!r}: a history has one "
                "window, the item's, after its last event (| (tl, tr])."
            )
        window = _parse_window(line, after)
    events = [_parse_one(e) for e in _split(body)]
    item = Item(name.strip(), *events)
    return item / window if window is not None else item


class Items:
    """Several units' histories (:class:`Item`), ready for the recurrent
    fitters and as :class:`~surpyval.SurpyvalData`.

    ``Items([pump, fan], unit="hours")``; ids must differ.
    :meth:`surpyval_data` gives the data, one row per event (and one for
    each end of watching), each row's unit ``i`` and terminal flag ``a``;
    the recurrent fitters, and any fitter that takes a ``SurpyvalData``,
    take an ``Items`` as it is. ``text()`` writes one history per line,
    which :meth:`parse` reads back.

    Examples
    --------
    >>> from surpyval.utils.notation import Item, Items, O, RC
    >>> fleet = Items.parse('''
    ...     P1: 120, 310, [400, 450], 1000+
    ...     P2: 3*200, 900 | (100, inf]
    ... ''')
    >>> data = fleet.surpyval_data()
    >>> data.i
    ['P1', 'P1', 'P1', 'P1', 'P2', 'P2', 'P2', 'P2']
    >>> data.a
    array([0, 0, 0, 0, 0, 0, 0, 0])
    >>> from surpyval.recurrent import CrowAMSAA
    >>> CrowAMSAA.fit(fleet).params.round(3)
    array([108.293,   0.656])
    """

    __slots__ = ("items", "unit")

    def __init__(self, items: Iterable[Item] = (), unit: str | None = None):
        if isinstance(items, Item):
            raise ValueError(
                f"Items takes a list of Item; got one Item ({items.id!r}): "
                "give [item]."
            )
        items = list(items)
        wrong = [i for i in items if not isinstance(i, Item)]
        if wrong:
            raise ValueError(
                f"Items takes Item histories; got {wrong[0]!r}. For units "
                "with one observation each, use Observations([...]); for "
                "text, Items.parse()."
            )
        ids = [i.id for i in items]
        seen: set = set()
        twice = []
        for i in ids:
            if i in seen and i not in twice:
                twice.append(i)
            seen.add(i)
        if twice:
            raise ValueError(
                "Each item's id must differ, and "
                + ", ".join(map(repr, twice))
                + f" {'is' if len(twice) == 1 else 'are'} given more than "
                "once: put a unit's events in one Item, or rename the ids."
            )
        self.items: list = items
        #: The unit of the ages (``"hours"``), kept with the data.
        self.unit: str | None = _checked_unit(unit)

    @classmethod
    def parse(cls, text: str, unit: str | None = None) -> Items:
        """Read histories from text, one per line: the id, a colon and its
        events in the grammar of :meth:`Observations.parse`, with the
        item's window last after a bar (``"P1: 120, 310, [400, 450],
        1000+ | (0, inf]"``). ``3*120`` is three failures at 120, ``1000+``
        the end of watching. Ids are read as text.

        Examples
        --------
        >>> from surpyval.utils.notation import Items
        >>> fleet = Items.parse("P1: 120, 310, 1000+\\nP2: 50 | (10, inf]")
        >>> print(fleet.text())
        P1: 120, 310, 1000+
        P2: 50 | (10, inf]
        >>> Items.parse(fleet.text()) == fleet
        True
        """
        if not isinstance(text, str):
            raise ValueError(
                'Items.parse takes a string, e.g. "P1: 120, 310, 1000+".'
            )
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return cls((_parse_item(ln) for ln in lines), unit=unit)

    def text(self) -> str:
        """One history per line (:meth:`Item.text`), which :meth:`parse`
        reads back; covariates, event types, terminal flags, readings and
        time-varying covariates are left out, and so is the unit."""
        return "\n".join(i.text() for i in self.items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Items):
            return NotImplemented
        return self.items == other.items and self.unit == other.unit

    __hash__ = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def __getitem__(self, index: Any) -> Any:
        got = self.items[index]
        return Items(got, unit=self.unit) if isinstance(index, slice) else got

    def __add__(self, other: object) -> Items:
        if isinstance(other, Item):
            other = Items([other], unit=self.unit)
        if not isinstance(other, Items):
            return NotImplemented
        units = {u for u in (self.unit, other.unit) if u is not None}
        if len(units) > 1:
            raise ValueError(
                f"These items are in {self.unit!r} and those in "
                f"{other.unit!r}: convert one set's ages first, then join "
                "them."
            )
        return Items(self.items + other.items, unit=next(iter(units), None))

    def surpyval_data(self, **kwargs: Any) -> Any:
        """The :class:`~surpyval.SurpyvalData` of the histories, the
        primary form for several rows per unit: one row per event in time
        order (a failure repeated at one age is a row per failure; a
        count in an interval or before a first look one row), then the
        end of watching (``c = 1``) where given, each row's unit ``i``,
        the item's late entry ``tl`` on its first row and its right
        truncation on every row, its covariates (``Z``, named), the
        events' types (``e``), the readings (``m``) and the time-varying
        covariates, passed through. The terminal flag ``a`` is 1 on a
        terminal event and 0 on every other row; where no item has more
        than one row and none is terminal, the data has no ``a`` (it is
        lifetime data, as ``SurpyvalData.from_items`` gives). ``kwargs``
        go to ``SurpyvalData``."""
        from surpyval.utils.surpyval_data import SurpyvalData

        if not self.items:
            raise ValueError("There are no items to convert.")
        _unit_into(self.unit, kwargs, "items' ages")
        x: list = []
        c, n, i, tl, tr, e, a = [], [], [], [], [], [], []
        several = False
        for item in self.items:
            rows = item._rows()
            several = several or len(rows) > 1
            t = item.truncation or Truncation()
            for k, (xk, ck, nk, ek, ak) in enumerate(rows):
                x.append(xk)
                c.append(ck)
                n.append(nk)
                i.append(item.id)
                tl.append(t.left if k == 0 else -math.inf)
                tr.append(t.right)
                e.append(ek)
                a.append(1 if ak else 0)
        extra: dict[str, Any] = {}
        if np.isfinite(tl).any() or np.isfinite(tr).any():
            extra.update(tl=tl, tr=tr)
        labels = [f"item {item.id!r}" for item in self.items]
        names, Z = _covariate_matrix(
            [item.covariates for item in self.items], labels
        )
        if Z is not None:
            counts = [len(item._rows()) for item in self.items]
            extra.update(Z=np.repeat(Z, counts, axis=0), covariate_names=names)
        if any(t is not None for t in e):
            types = np.empty(len(e), dtype=object)
            types[:] = e
            extra["e"] = types
        if several or any(a):
            extra["a"] = a
        m = self._measurements()
        if m is not None:
            extra["m"] = m
        data = SurpyvalData(x, c, n, i=i, **extra, **kwargs)
        varying = [item.history.time_varying_covariates for item in self.items]
        if any(v is not None for v in varying):
            missing = [
                repr(item.id)
                for item, v in zip(self.items, varying)
                if v is None
            ]
            if missing:
                raise ValueError(
                    "Time-varying covariates are given for some items and "
                    "not others: " + ", ".join(missing[:5]) + " "
                    f"{'has' if len(missing) == 1 else 'have'} none. Give "
                    "every item its schedule, or none."
                )
            data.time_varying_covariates = {
                item.id: v for item, v in zip(self.items, varying)
            }
        return data

    def _measurements(self) -> Any:
        """The readings as one store ``{i: {x: value}}`` (one unnamed
        series) or ``{name: {i: {x: value}}}``; None where there are
        none."""
        given = [item for item in self.items if item.readings]
        if not given:
            return None
        named_ = {item.id: None in item.readings for item in given}
        if len(set(named_.values())) > 1:
            unnamed = [repr(k) for k, v in named_.items() if v]
            raise ValueError(
                "Items "
                + ", ".join(unnamed[:5])
                + " give one unnamed series of readings and others name "
                "theirs: name every series ({name: {age: value}}) or give "
                "each item one unnamed series."
            )
        if None in given[0].readings:
            return {item.id: item.readings[None] for item in given}
        out: dict = {}
        for item in given:
            for name, series in item.readings.items():
                out.setdefault(name, {})[item.id] = series
        return out

    def event_history_data(self) -> Any:
        """The histories as an
        :class:`~surpyval.utils.event_history_data.EventHistoryData`, a
        bridge while that class remains: :meth:`surpyval_data` is the
        primary form, which every recurrent fitter takes. Terminal flags
        and readings, which it has no place for, are refused."""
        from surpyval.utils.event_history_data import EventHistoryData

        ended = [
            repr(item.id)
            for item in self.items
            if any(ev.terminal for ev in item.events)
        ]
        if ended:
            raise ValueError(
                "EventHistoryData has no terminal flag, and items "
                + ", ".join(ended[:5])
                + " end at a terminal event: use surpyval_data(), which "
                "keeps it (a)."
            )
        if any(item.readings for item in self.items):
            raise ValueError(
                "EventHistoryData has no readings: use surpyval_data(), "
                "which keeps them (m)."
            )
        return EventHistoryData.from_items(
            [item.history for item in self.items], unit=self.unit
        )

    def __repr__(self) -> str:
        head = f"Items, {_plural(len(self.items), 'item')}"
        if self.unit is not None:
            head += f", ages in {self.unit}"
        if not self.items:
            return head
        shown = self.items[:10]
        lines = [repr(item) for item in shown]
        if len(self.items) > len(shown):
            lines.append(f"… and {len(self.items) - len(shown)} more")
        return head + ":\n" + "\n".join(lines)
