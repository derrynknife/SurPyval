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
units alike, and ``obs / Truncation(tl, tr)`` says the unit was only
observed within ``(tl, tr]``: late entry at ``tl``, right truncation at
``tr``, the same window as everywhere else in SurPyval (an event exactly
at ``tl`` is outside it and refused).

The same data can be written as text, one entry per observation, in the
shorthand of R's ``Surv`` printout::

    >>> Observations.parse("3*10, 20+, <3, [5, 10]")
    Observations, 4 rows (6 units):
      3 × failed at 10
      1 × still running at 20
      1 × failed before 3
      1 × failed between 5 and 10

``"10"`` failed at 10, ``"20+"`` still running at 20, ``"<3"`` (or
``"3-"``) failed before 3, ``"[5, 10]"`` failed between 5 and 10, and
``"3*10"`` (or ``"3×10"``) three units that failed at 10.

``Observations.surpyval_data()`` gives the :class:`~surpyval.SurpyvalData`
every fitter takes (``Weibull.fit(data.surpyval_data())``), and
``Observations.xcnt()`` the arrays themselves (``x``, ``c``, ``n``, ``tl``,
``tr``), with censoring coded as everywhere in SurPyval: 0 failed, 1
right censored, -1 left censored, 2 interval censored.
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


def _time(value: object, what: str) -> float:
    """A number of time, or a ValueError naming ``what``."""
    if isinstance(value, bool) or not isinstance(value, Number):
        raise ValueError(f"{what} must be a number, got {value!r}.")
    out = float(cast(SupportsFloat, value))
    if math.isnan(out):
        raise ValueError(f"{what} must be a number, not NaN.")
    return out


def _fmt(value: float) -> str:
    return (
        f"{value:g}"
        if math.isfinite(value)
        else ("inf" if value > 0 else "-inf")
    )


class Truncation:
    """The window ``(left, right]`` a unit was observed in.

    ``left`` is the time it entered observation (late entry, left
    truncation) and ``right`` the time observation stopped for good
    (right truncation). ``None`` means no truncation on that side.
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
        return f"({_fmt(self.left)}, {_fmt(self.right)}]"


class SurvivalObservation:
    """One observation (or ``n`` alike): what happened, how many, and the
    window it was observed in. Use :class:`Observed`,
    :class:`RightCensored`, :class:`LeftCensored` or
    :class:`IntervalCensored` (``O``, ``RC``, ``LC``, ``IC``)."""

    __slots__ = ("n", "truncation")
    #: SurPyval's censoring code for this kind of observation.
    code: int = 0

    def __init__(self) -> None:
        self.n: float = 1.0
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

    def _fits(self, t: Truncation) -> bool:
        low, high = self._bounds()
        if self.code == 2:
            # An interval needs some part of it inside the window (tl, tr].
            return high > t.left and low < t.right
        # A point (failed, censored) must lie in (tl, tr].
        return t.left < low <= t.right

    def __truediv__(self, t: Truncation) -> SurvivalObservation:
        if not isinstance(t, Truncation):
            return NotImplemented
        if not self._fits(t):
            raise ValueError(
                f"{self._words()} lies outside its observation window "
                f"{t.text()}: a unit is only seen inside its window (an "
                "event exactly at the "
                "left edge is outside it). Check the times or the window."
            )
        new = self._copy()
        new.truncation = t
        return new

    def __mul__(self, n: float) -> SurvivalObservation:
        if isinstance(n, bool) or not isinstance(n, Number):
            return NotImplemented
        k = float(n)
        if not math.isfinite(k) or k <= 0:
            raise ValueError(f"A count must be a positive number, got {k:g}.")
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

    def __repr__(self) -> str:
        n = f"{self.n:g} × " if self.n != 1 else ""
        t = (
            f", observed in {self.truncation.text()}"
            if self.truncation
            else ""
        )
        return f"{n}{self._words()}{t}"


class _Point(SurvivalObservation):
    __slots__ = ("x",)

    def __init__(self, x: float) -> None:
        super().__init__()
        self.x = _time(x, "The time")

    def _bounds(self) -> tuple[float, float]:
        return (self.x, self.x)


class Observed(_Point):
    """A unit that failed at ``x``."""

    __slots__ = ()
    code = 0

    def _words(self) -> str:
        return f"failed at {_fmt(self.x)}"


class RightCensored(_Point):
    """A unit still running at ``x`` (a suspension)."""

    __slots__ = ()
    code = 1

    def _words(self) -> str:
        return f"still running at {_fmt(self.x)}"


class LeftCensored(_Point):
    """A unit that failed some time before ``x``."""

    __slots__ = ()
    code = -1

    def _words(self) -> str:
        return f"failed before {_fmt(self.x)}"


class IntervalCensored(SurvivalObservation):
    """A unit that failed some time between ``left`` and ``right``."""

    __slots__ = ("left", "right")
    code = 2

    def __init__(self, left: float, right: float) -> None:
        super().__init__()
        lo = _time(left, "The start of an interval")
        hi = _time(right, "The end of an interval")
        if lo >= hi:
            raise ValueError(
                f"An interval's start ({_fmt(lo)}) must be below its "
                f"end ({_fmt(hi)}); "
                "for a failure seen at one time use O(x)."
            )
        self.left: float = lo
        self.right: float = hi

    def _bounds(self) -> tuple[float, float]:
        return (self.left, self.right)

    def _words(self) -> str:
        return f"failed between {_fmt(self.left)} and {_fmt(self.right)}"


O = Observed
RC = RightCensored
LC = LeftCensored
IC = IntervalCensored

_COUNT = r"(?:(?P<n>\d+(?:\.\d+)?)\s*[*×x]\s*)?"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
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


def _split(text: str) -> list[str]:
    """Split on commas and new lines, keeping an interval's comma inside it."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch in ",\n;" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_one(entry: str) -> SurvivalObservation:
    for pattern, kind in _PATTERNS:
        m = pattern.match(entry)
        if not m:
            continue
        a = float(m.group("a"))
        out: SurvivalObservation
        if kind == "ic":
            out = IntervalCensored(a, float(m.group("b")))
        else:
            points: dict[str, type[_Point]] = {
                "o": Observed,
                "rc": RightCensored,
                "lc": LeftCensored,
            }
            out = points[kind](a)
        return out * float(m.group("n")) if m.group("n") else out
    raise ValueError(
        f"Can't read {entry!r}. Write 10 (failed at 10), "
        "20+ (still running at 20), <3 or 3- (failed before 3), "
        "[5, 10] (failed between 5 and 10), and 3*10 for three alike."
    )


class Observations:
    """A list of observations, ready for any fitter.

    ``Observations([3 * O(10), RC(20)])``; ``obs + other`` joins two;
    ``obs / Truncation(tl, tr)`` gives every observation the same window
    (one that already has its own window keeps it); ``surpyval_data()`` and
    ``xcnt()`` give what the fitters take.
    """

    __slots__ = ("observations",)

    def __init__(
        self, observations: Iterable[SurvivalObservation] = ()
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

    @classmethod
    def parse(cls, text: str) -> Observations:
        """Read observations from text: entries separated by commas, semicolons
        or new lines (``"3*10, 20+, <3, [5, 10]"``)."""
        if not isinstance(text, str):
            raise ValueError(
                'Observations.parse takes a string, e.g. "10, 20+, [5, 10]".'
            )
        return cls(_parse_one(e) for e in _split(text))

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[SurvivalObservation]:
        return iter(self.observations)

    def __getitem__(self, index: Any) -> Any:
        got = self.observations[index]
        return Observations(got) if isinstance(index, slice) else got

    def __add__(self, other: object) -> Observations:
        if isinstance(other, SurvivalObservation):
            other = Observations([other])
        if not isinstance(other, Observations):
            return NotImplemented
        return Observations(self.observations + other.observations)

    def __truediv__(self, t: Truncation) -> Observations:
        if not isinstance(t, Truncation):
            return NotImplemented
        return Observations(
            o if o.truncation else o / t for o in self.observations
        )

    @property
    def units(self) -> float:
        """How many units in all (the counts added up)."""
        return float(sum(o.n for o in self.observations))

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
        n = np.array([o.n for o in self.observations], dtype=float)
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
        (``Weibull.fit(obs.surpyval_data())``). ``kwargs`` go to
        ``SurpyvalData`` (``unit=``, for example)."""
        from surpyval.utils.surpyval_data import SurpyvalData

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
        plural = "s" if rows != 1 else ""
        head = f"Observations, {rows} row{plural} ({self.units:g} units)"
        if not rows:
            return head
        shown = self.observations[:20]
        lines = [
            f"  {o.n:g} × {o._words()}"
            + (f", observed in {o.truncation.text()}" if o.truncation else "")
            for o in shown
        ]
        if rows > len(shown):
            lines.append(f"  … and {rows - len(shown)} more")
        return head + ":\n" + "\n".join(lines)
