r"""Population hazard rates: the background a relative survival analysis
measures against.

A rate table holds the hazard of the event (death, say) in a reference
population, by attained age and, optionally, calendar year and a
stratifying variable (sex, region). A unit diagnosed (or installed) at
age :math:`a` in calendar year :math:`y` has at follow-up time :math:`t`
the population hazard of attained age :math:`a + t` in year :math:`y +
t`: its age and the calendar move along together as it is followed. Its
expected (population) cumulative hazard is

.. math::
    \Lambda_P(t) = \int_0^t \lambda_P(a + s, y + s)\, ds ,

computed exactly here, band by band, as the unit crosses the table's age
and year bands. Life tables usually give the annual probability of death
:math:`q`; with times in years the hazard of that year is
:math:`-\log(1 - q)`.

The table has no unit of time: its ages, years and follow-up times must
be in one unit, and its rates per that unit (per person-year with times
in years, per day with times in days).
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)

#: The largest number of (unit, band edge) pairs held at once.
_CHUNK = 2_000_000


def _edges(values: Any, name: str) -> npt.NDArray:
    try:
        out = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"'{name}' must be numbers: the lower edge of each band, in "
            f"increasing order; got {values!r}."
        ) from None
    if out.ndim != 1 or out.size == 0:
        raise ValueError(
            f"'{name}' must be a 1-D array of the lower edge of each band, "
            f"in increasing order; got shape {out.shape}."
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"'{name}' must be finite; got {out}.")
    if np.any(np.diff(out) <= 0):
        raise ValueError(
            f"'{name}' must be strictly increasing (each the lower edge of "
            f"a band); got {out}."
        )
    return out


def _label(value: Any) -> Any:
    """A stratum label as a plain Python value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(
        value, (str, numbers.Integral, numbers.Real)
    ):
        raise ValueError(
            "A stratum label must be a string or a number; got "
            f"{value!r} ({type(value).__name__})."
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _key(value: Any) -> Any:
    """A stratum label given with the data, as the table keys it."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class RateTable(SerialisableMixin):
    r"""
    Population hazard rates by attained age, and optionally by calendar
    year and stratum: the expected hazards of a relative survival
    analysis (:class:`surpyval.RelativeSurvival`,
    :class:`surpyval.ExcessHazard`).

    Each band runs from its edge up to the next: an age band from
    ``ages[k]`` to ``ages[k + 1]``, the last open above, and the first
    taken to cover attained ages below it too; the same for ``years``.
    A unit's attained age and calendar year advance with its follow-up
    time, so its expected cumulative hazard is the integral of the rates
    along that diagonal path through the table (see the module).

    Parameters
    ----------
    rates : array_like or dict
        The hazard rates, per unit of time, at least 0: shape
        ``(len(ages),)``, or ``(len(ages), len(years))`` with ``years``.
        With strata, a ``dict`` of such arrays keyed by stratum label
        (``{"female": ..., "male": ...}``).
    ages : array_like
        The lower edge of each age band, strictly increasing.
    years : array_like, optional
        The lower edge of each calendar-year band, strictly increasing.
        Without it the rates depend on age alone.

    Notes
    -----
    Ages, years and the follow-up times given to the fits must be in the
    same unit, and the rates per that unit: per person-year with times in
    years. A life table's annual probability of death :math:`q` is the
    rate :math:`-\log(1 - q)` per year.

    Examples
    --------
    A table by age and sex, the rates rising 9% a year of age:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> ages = np.arange(0, 101)
    >>> table = sp.RateTable(
    ...     {"F": 5e-5 * np.exp(0.09 * ages), "M": 8e-5 * np.exp(0.09 * ages)},
    ...     ages=ages,
    ... )
    >>> table
    RateTable: 101 age band(s) (0 to 100+), 2 strata ('F', 'M')
    >>> float(table.sf(10, age=70, stratum="F").round(4))  # expected survival
    0.6557
    """

    ages: npt.NDArray
    years: npt.NDArray | None
    strata: list | None
    rates: npt.NDArray

    def __init__(
        self,
        rates: Any,
        ages: npt.ArrayLike,
        years: npt.ArrayLike | None = None,
    ) -> None:
        self.ages = _edges(ages, "ages")
        self.years = None if years is None else _edges(years, "years")
        shape: tuple[int, ...] = (self.ages.size,)
        if self.years is not None:
            shape = shape + (self.years.size,)
        if isinstance(rates, dict):
            if not rates:
                raise ValueError(
                    "'rates' is an empty dict: give one array of rates per "
                    "stratum, keyed by its label."
                )
            labels = [_label(k) for k in rates]
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"'rates' has two strata of one label: {labels}."
                )
            self.strata = labels
            arrays = [self._rate_array(v, shape, k) for k, v in rates.items()]
            stacked = np.stack(arrays)
        else:
            self.strata = None
            stacked = self._rate_array(rates, shape, None)[None]
        if self.years is None:
            stacked = stacked[..., None]
        self.rates = stacked

    @staticmethod
    def _rate_array(
        values: Any, shape: tuple[int, ...], stratum: Any
    ) -> npt.NDArray:
        where = "" if stratum is None else f" of stratum {stratum!r}"
        try:
            out = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"The rates{where} must be numbers; got {values!r}."
            ) from None
        if out.shape != shape:
            axes = "(ages,)" if len(shape) == 1 else "(ages, years)"
            years = f" and {shape[1]} year band(s)" if len(shape) > 1 else ""
            raise ValueError(
                f"The rates{where} have shape {out.shape}; a table with "
                f"{shape[0]} age band(s){years} needs shape {shape}, one "
                f"rate per band {axes}."
            )
        if not np.all(np.isfinite(out)) or np.any(out < 0):
            raise ValueError(
                f"The rates{where} must be finite and at least 0 (hazards "
                "per unit of time)."
            )
        return out

    # -- reading it ----------
    def __repr__(self) -> str:
        parts = [
            f"{self.ages.size} age band(s) ({self.ages[0]:g} to "
            f"{self.ages[-1]:g}+)"
        ]
        if self.years is not None:
            parts.append(
                f"{self.years.size} year band(s) ({self.years[0]:g} to "
                f"{self.years[-1]:g}+)"
            )
        if self.strata is not None:
            shown = ", ".join(repr(s) for s in self.strata[:6])
            more = ", ..." if len(self.strata) > 6 else ""
            parts.append(f"{len(self.strata)} strata ({shown}{more})")
        return "RateTable: " + ", ".join(parts)

    @property
    def by_year(self) -> bool:
        """
        Whether the rates depend on the calendar year.

        Examples
        --------
        >>> import surpyval as sp
        >>> sp.RateTable([0.01], ages=[0]).by_year
        False
        """
        return self.years is not None

    def stratum_index(self, stratum: Any, size: int) -> npt.NDArray:
        """
        The row of the table for each of ``size`` units: ``stratum`` is
        one label for every unit or one per unit (refused where the table
        has no strata, and required where it has).

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable({"F": [0.01], "M": [0.02]}, ages=[0])
        >>> table.stratum_index(["M", "F", "M"], 3)
        array([1, 0, 1])
        """
        if self.strata is None:
            if stratum is not None:
                raise ValueError(
                    "This rate table has no strata, so 'stratum' must be "
                    "left out (or build the table with a dict of rates per "
                    "stratum)."
                )
            return np.zeros(size, dtype=int)
        if stratum is None:
            raise ValueError(
                "This rate table is stratified ({} strata: {}): give each "
                "unit's 'stratum'.".format(len(self.strata), self.strata)
            )
        values = np.asarray(stratum, dtype=object)
        if values.ndim == 0:
            values = np.full(size, values.item(), dtype=object)
        values = values.ravel()
        if values.size != size:
            raise ValueError(
                f"'stratum' has {values.size} values for {size} units; give "
                "one label, or one per unit."
            )
        lookup = {label: k for k, label in enumerate(self.strata)}
        out = np.empty(size, dtype=int)
        for i, value in enumerate(values):
            try:
                out[i] = lookup[_key(value)]
            except (KeyError, TypeError):
                raise ValueError(
                    f"'stratum' {value!r} (unit {i}) is not a stratum of the "
                    f"rate table: {self.strata}."
                ) from None
        return out

    def _rate(
        self, s: npt.NDArray, age: npt.NDArray, year: npt.NDArray
    ) -> npt.NDArray:
        """The rate of row ``s`` at attained ``age`` and ``year``."""
        ia = np.clip(
            np.searchsorted(self.ages, age, side="right") - 1,
            0,
            self.ages.size - 1,
        )
        if self.years is None:
            iy = np.zeros_like(ia)
        else:
            iy = np.clip(
                np.searchsorted(self.years, year, side="right") - 1,
                0,
                self.years.size - 1,
            )
        return self.rates[s, ia, iy]

    def _path(
        self,
        age: npt.NDArray,
        year: npt.NDArray,
        s: npt.NDArray,
        horizon: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The expected cumulative hazard of each unit along its path, as
        ``(nodes, H)``: ``(n, B)`` arrays, each row the follow-up times
        from 0 to ``horizon`` at which the unit crosses a band edge
        (repeated at ``horizon`` to fill the row) and the cumulative
        hazard there. It is linear between the nodes, so ``np.interp``
        of a row gives it exactly at any time up to ``horizon``."""
        n = age.size
        cuts = [np.zeros((n, 1))]
        top = float(np.max(horizon)) if n else 0.0
        for edges, start in ((self.ages, age), (self.years, year)):
            if edges is None or edges.size < 2:
                continue
            cut = edges[None, 1:] - start[:, None]
            used = np.any((cut > 0) & (cut < top), axis=0)
            if np.any(used):
                cuts.append(cut[:, used])
        cuts.append(horizon[:, None])
        nodes = np.clip(np.concatenate(cuts, axis=1), 0.0, horizon[:, None])
        nodes.sort(axis=1)
        mid = 0.5 * (nodes[:, :-1] + nodes[:, 1:])
        rate = self._rate(s[:, None], age[:, None] + mid, year[:, None] + mid)
        H = np.zeros_like(nodes)
        np.cumsum(rate * np.diff(nodes, axis=1), axis=1, out=H[:, 1:])
        return nodes, H

    def _path_chunks(self) -> int:
        """How many units' paths to build at once."""
        width = self.ages.size + (0 if self.years is None else self.years.size)
        return max(1, _CHUNK // (width + 2))

    def _inputs(
        self, x: Any, age: Any, year: Any, stratum: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, tuple]:
        try:
            xs = np.asarray(x, dtype=float)
            ages = np.asarray(age, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "'x' and 'age' must be numbers (times and ages in the "
                "table's unit)."
            ) from None
        if self.years is None:
            if year is not None:
                raise ValueError(
                    "This rate table has no calendar-year bands, so 'year' "
                    "must be left out."
                )
            years = np.zeros(())
        else:
            if year is None:
                raise ValueError(
                    "This rate table is by calendar year: give each unit's "
                    "'year' (the calendar time at which its follow-up "
                    "starts, in the table's unit)."
                )
            try:
                years = np.asarray(year, dtype=float)
            except (TypeError, ValueError):
                raise ValueError(
                    f"'year' must be numbers; got {year!r}."
                ) from None
        labels = np.asarray(stratum, dtype=object)
        try:
            shape = np.broadcast_shapes(
                xs.shape, ages.shape, years.shape, labels.shape
            )
        except ValueError:
            raise ValueError(
                "'x', 'age', 'year' and 'stratum' must have one shape (or be "
                f"single values); got {np.shape(x)}, {np.shape(age)}, "
                f"{np.shape(year)} and {np.shape(stratum)}."
            ) from None
        t, a, y = (
            np.ravel(np.broadcast_to(v, shape)) for v in (xs, ages, years)
        )
        s = self.stratum_index(
            (
                None
                if stratum is None
                else np.broadcast_to(labels, shape).ravel()
            ),
            t.size,
        )
        if np.any(t < 0):
            raise ValueError(
                "'x' must be at least 0: the follow-up time since the "
                "unit's age and year were given."
            )
        return t, a, y, s, shape

    def Hf(
        self,
        x: npt.ArrayLike,
        age: npt.ArrayLike,
        year: npt.ArrayLike | None = None,
        stratum: Any = None,
    ) -> Any:
        """
        The expected cumulative hazard over the follow-up ``(0, x]`` of a
        unit aged ``age`` in calendar ``year`` (and ``stratum``) at time 0:
        the integral of the rates along its path through the table.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.02], ages=[0, 50])
        >>> table.Hf([5, 10], age=45)  # five years at 0.01, then 0.02
        array([0.05, 0.15])
        """
        t, a, y, s, shape = self._inputs(x, age, year, stratum)
        result = self._H(t, a, y, s).reshape(shape)
        return result[()] if result.ndim == 0 else result

    def _H(
        self,
        t: npt.NDArray,
        a: npt.NDArray,
        y: npt.NDArray,
        s: npt.NDArray,
    ) -> npt.NDArray:
        """The expected cumulative hazard to the follow-up times ``t`` of
        units of ages ``a``, years ``y`` and table rows ``s`` (flat
        arrays; NaN where a value is missing)."""
        out = np.full(t.size, np.nan)
        idx = np.flatnonzero(np.isfinite(t) & np.isfinite(a) & np.isfinite(y))
        step = self._path_chunks()
        for start in range(0, idx.size, step):
            part = idx[start : start + step]
            _, H = self._path(a[part], y[part], s[part], t[part])
            out[part] = H[:, -1]
        return out

    def hf(
        self,
        x: npt.ArrayLike,
        age: npt.ArrayLike,
        year: npt.ArrayLike | None = None,
        stratum: Any = None,
    ) -> Any:
        """
        The population hazard at follow-up time ``x`` of a unit aged
        ``age`` in calendar ``year`` (and ``stratum``) at time 0: the rate
        of attained age ``age + x`` in year ``year + x``.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.02], ages=[0, 50])
        >>> table.hf([4, 6], age=45)
        array([0.01, 0.02])
        """
        t, a, y, s, shape = self._inputs(x, age, year, stratum)
        out = np.full(t.size, np.nan)
        ok = np.isfinite(t) & np.isfinite(a) & np.isfinite(y)
        out[ok] = self._rate(s[ok], a[ok] + t[ok], y[ok] + t[ok])
        result = out.reshape(shape)
        return result[()] if result.ndim == 0 else result

    def sf(
        self,
        x: npt.ArrayLike,
        age: npt.ArrayLike,
        year: npt.ArrayLike | None = None,
        stratum: Any = None,
    ) -> Any:
        """
        The expected (population) survival over ``(0, x]``,
        ``exp(-Hf(x, ...))``, of a unit aged ``age`` in calendar ``year``
        (and ``stratum``) at time 0.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.02], ages=[0, 50])
        >>> float(table.sf(10, age=45).round(4))
        0.8607
        """
        return np.exp(-self.Hf(x, age, year, stratum))

    # -- building and saving ----------
    @classmethod
    def from_df(
        cls,
        df: Any,
        age_col: str,
        rate_col: str,
        year_col: str | None = None,
        stratum_col: str | None = None,
    ) -> RateTable:
        """
        A rate table from a long DataFrame: one row per age band (per year
        band and stratum, where those columns are named), with the band's
        lower edge and its rate. Every combination of age, year and
        stratum must appear exactly once.

        Examples
        --------
        >>> import pandas as pd
        >>> import surpyval as sp
        >>> df = pd.DataFrame({
        ...     "age": [0, 50, 0, 50],
        ...     "sex": ["F", "F", "M", "M"],
        ...     "rate": [0.010, 0.020, 0.015, 0.030],
        ... })
        >>> sp.RateTable.from_df(df, "age", "rate", stratum_col="sex")
        RateTable: 2 age band(s) (0 to 50+), 2 strata ('F', 'M')
        """
        for name, col in (
            ("age_col", age_col),
            ("rate_col", rate_col),
            ("year_col", year_col),
            ("stratum_col", stratum_col),
        ):
            if col is not None and col not in df.columns:
                raise ValueError(
                    f"{name}={col!r} is not a column of the DataFrame; its "
                    f"columns are {list(df.columns)}."
                )
        ages = np.unique(df[age_col].to_numpy(dtype=float))
        years = (
            None
            if year_col is None
            else np.unique(df[year_col].to_numpy(dtype=float))
        )
        keys = [age_col] + ([] if year_col is None else [year_col])
        what = "age" if year_col is None else "age and year"
        groups: list[tuple[Any, Any]]
        if stratum_col is None:
            groups = [(None, df)]
        else:
            labels = list(dict.fromkeys(df[stratum_col].tolist()))
            groups = [(lab, df[df[stratum_col] == lab]) for lab in labels]
        tables = {}
        expected = ages.size * (1 if years is None else years.size)
        for label, part in groups:
            where = "" if label is None else f" in stratum {label!r}"
            if part.duplicated(keys).any():
                raise ValueError(
                    f"The DataFrame repeats an {what}{where}: give each "
                    "band once."
                )
            if len(part) != expected:
                raise ValueError(
                    f"The DataFrame has {len(part)} rows{where} for "
                    f"{expected} bands: every {what} needs a rate."
                )
            if years is None:
                order = part.sort_values(age_col)
                tables[label] = order[rate_col].to_numpy(dtype=float)
            else:
                grid = part.pivot(
                    index=age_col, columns=year_col, values=rate_col
                )
                tables[label] = grid.loc[ages, years].to_numpy(dtype=float)
        rates: Any = tables[None] if stratum_col is None else tables
        return cls(rates, ages, years)

    def to_dict(self) -> dict:
        """
        The table as a plain, JSON-ready ``dict``; :meth:`from_dict`
        rebuilds it.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.02], ages=[0, 50])
        >>> sp.RateTable.from_dict(table.to_dict())
        RateTable: 2 age band(s) (0 to 50+)
        """
        rates = self.rates if self.years is not None else self.rates[..., 0]
        out: dict[str, Any] = {
            "model": "RateTable",
            "ages": self.ages.tolist(),
            "years": None if self.years is None else self.years.tolist(),
            "strata": None if self.strata is None else to_native(self.strata),
            "rates": (rates[0] if self.strata is None else rates).tolist(),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> RateTable:
        """
        Rebuild a table from :meth:`to_dict`'s dictionary.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable({"a": [0.01], "b": [0.02]}, ages=[0])
        >>> sp.RateTable.from_dict(table.to_dict()).strata
        ['a', 'b']
        """
        require_model_tag(model_dict, "RateTable", "a rate table")
        strata = model_dict.get("strata")
        rates: Any = model_dict["rates"]
        if strata is not None:
            rates = dict(zip(strata, rates))
        return cls(rates, model_dict["ages"], model_dict.get("years"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RateTable):
            return NotImplemented
        same_years = (self.years is None and other.years is None) or (
            self.years is not None
            and other.years is not None
            and np.array_equal(self.years, other.years)
        )
        return bool(
            np.array_equal(self.ages, other.ages)
            and same_years
            and self.strata == other.strata
            and np.array_equal(self.rates, other.rates)
        )

    __hash__ = None  # type: ignore[assignment]
