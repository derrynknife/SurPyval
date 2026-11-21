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
from collections.abc import Iterable, Iterator
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

    __slots__ = ("n", "truncation")
    #: SurPyval's censoring code for this kind of observation.
    code: int = 0

    def __init__(self) -> None:
        self.n: int = 1
        self.truncation: Truncation | None = None

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
        return (
            type(self) is type(other)
            and self._bounds() == other._bounds()
            and self.n == other.n
            and self.truncation == other.truncation
        )

    def __hash__(self) -> int:
        return hash((type(self), self._bounds(), self.n, self.truncation))

    def _words(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _entry(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def text(self) -> str:
        """The observation as ``Observations.parse`` reads it:
        ``"3*10 | (2, 50]"``."""
        n = f"{self.n}*" if self.n != 1 else ""
        t = f" | {self.truncation.text()}" if self.truncation else ""
        return f"{n}{self._entry()}{t}"

    def _line(self) -> str:
        t = (
            f", observed in {self.truncation._words()}"
            if self.truncation
            else ""
        )
        return f"{self._words()}{t}"

    def __repr__(self) -> str:
        n = f"{self.n} × " if self.n != 1 else ""
        return f"{n}{self._line()}"


class _Point(SurvivalObservation):
    __slots__ = ("x",)

    def __init__(self, x: float) -> None:
        super().__init__()
        self.x = _finite_time(x)

    def _bounds(self) -> tuple[float, float]:
        return (self.x, self.x)


class Observed(_Point):
    """A unit that failed at ``x``."""

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

    def __new__(cls, left: float, right: float) -> Any:
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
            return RightCensored(lo)
        if lo == -math.inf:
            return LeftCensored(hi)
        return super().__new__(cls)

    def __init__(self, left: float, right: float) -> None:
        super().__init__()
        self.left: float = float(left)
        self.right: float = float(right)

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
        for k, code in enumerate(np.asarray(data.c).tolist()):
            obs: SurvivalObservation
            if code == 2:
                obs = IntervalCensored(lo[k], hi[k])
            else:
                kind = {0: "o", 1: "rc", -1: "lc"}[code]
                obs = _POINTS[kind](lo[k])
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
        ``parse`` again."""
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

        if self.unit is not None:
            given = kwargs.get("unit")
            if given is not None and _checked_unit(given) != self.unit:
                raise ValueError(
                    f"The observations are in {self.unit!r} and unit= "
                    f"says {given!r}: leave unit= out, or convert the "
                    "times first."
                )
            kwargs["unit"] = self.unit
        a = self.xcnt()
        x = [
            row[0] if code != 2 else list(row)
            for row, code in zip(a["x"], a["c"])
        ]
        truncated = np.isfinite(a["tl"]).any() or np.isfinite(a["tr"]).any()
        return SurpyvalData(
            x=x,
            c=a["c"],
            n=a["n"],
            tl=a["tl"] if truncated else None,
            tr=a["tr"] if truncated else None,
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
