"""Person-years (exposure) tables: :func:`person_years`, the total time at
risk, the number of events and the number of units in each cell of a
cross-classification by fixed factors and by time bands a unit moves
through while it is followed (attained age, calendar period, time since
entry), with each cell's event rate and its exact (Garwood) confidence
bounds, rate ratios between cells, and the expected events and observed
to expected ratios given a table of reference rates.

It imports no plotting library.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import alpha_ci_error

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["person_years", "PersonYears", "poisson_rate_bounds"]

#: Column names a factor or time scale may not take.
_RESERVED = (
    "units",
    "exposure",
    "events",
    "rate",
    "rate_lower",
    "rate_upper",
    "expected",
    "oe_ratio",
    "oe_lower",
    "oe_upper",
)


def poisson_rate_bounds(
    events: npt.ArrayLike,
    exposure: npt.ArrayLike,
    alpha_ci: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""
    The exact (Garwood) two-sided confidence bounds of a Poisson rate,
    ``events / exposure``.

    The lower bound is the rate at which seeing ``events`` or more has
    probability ``alpha_ci / 2``, the upper the rate at which seeing
    ``events`` or fewer has that probability:

    .. math::

        \frac{\chi^2_{2d,\ \alpha/2}}{2T}, \qquad
        \frac{\chi^2_{2d + 2,\ 1 - \alpha/2}}{2T},

    with a lower bound of 0 when ``d = 0`` (Garwood 1936; R's
    ``poisson.test``). Their coverage is at least ``1 - alpha_ci`` for
    every true rate.

    Parameters
    ----------
    events : array like
        The number of events (whole or not: a count given with weights
        ``n`` that are not whole is used as it is).
    exposure : array like
        The total time at risk, in any unit; the bounds are per that unit.
    alpha_ci : float, optional
        The bounds are a two-sided ``1 - alpha_ci`` interval. Default
        0.05.

    Returns
    -------
    lower, upper : ndarray
        The bounds, ``nan`` where the exposure is 0.

    Examples
    --------
    >>> from surpyval import poisson_rate_bounds
    >>> lower, upper = poisson_rate_bounds(3, 1000.0)
    >>> print(round(float(lower), 6), round(float(upper), 6))
    0.000619 0.008767
    """
    from scipy import stats

    if not (isinstance(alpha_ci, (int, float)) and 0 < alpha_ci < 1):
        raise alpha_ci_error(alpha_ci)
    d = np.asarray(events, dtype=float)
    T = np.asarray(exposure, dtype=float)
    if np.any(d < 0) or np.any(T < 0):
        raise ValueError(
            "'events' and 'exposure' must not be negative; got events from "
            "{:g} and exposure from {:g}.".format(np.min(d), np.min(T))
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        low = np.where(
            d > 0, stats.chi2.ppf(alpha_ci / 2, 2 * np.maximum(d, 1)) / 2, 0.0
        )
        high = stats.chi2.ppf(1 - alpha_ci / 2, 2 * d + 2) / 2
        lower = np.where(T > 0, low / np.where(T > 0, T, 1.0), np.nan)
        upper = np.where(T > 0, high / np.where(T > 0, T, 1.0), np.nan)
    return lower, upper


def person_years(
    x: Any,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
    *,
    factors: Mapping[str, Any] | None = None,
    timescales: Mapping[str, Any] | None = None,
    rates: Any = None,
    per: float = 1.0,
    alpha_ci: float = 0.05,
) -> PersonYears:
    r"""
    The exposure (person-years) table of right censored, possibly
    late-entry data: in each cell of a cross-classification, the total
    time the units were followed there, the number of events there and
    the number of units that spent time there, with the event rate and
    its exact confidence bounds. As R's ``survival::pyears``.

    The cells are formed by *factors*, fixed for each unit (a site, a
    model, a sex), and by *time scales* cut into bands (attained age,
    calendar period, time since entry, operating hours since overhaul).
    A unit moves through the bands of a time scale as it is followed: a
    pump followed from 3,000 to 7,000 operating hours, with bands every
    2,000 hours, spends 1,000 hours in the 2,000-4,000 band, 2,000 in the
    4,000-6,000 band and 1,000 in the 6,000-8,000 band; its event (or its
    censoring) is in the band it is in when it happens. R's ``tcut`` does
    this; a band read from each unit's value at the start
    (``cut(age, ...)``) would put all its time in one band.

    Parameters
    ----------
    x : array like
        The time each unit's follow-up ends (its event or its censoring),
        on the time scale of the data: time since each unit's origin
        (installation, birth, diagnosis, the start of a study). The rows
        are matched to the factors and start values by position, so a
        :class:`~surpyval.SurpyvalData` (which sorts its rows) is
        refused; pass the arrays.
    c : array like, optional
        The censoring flags: 0 an event, 1 right censored (followed to
        ``x`` with no event). Left (-1) and interval (2) censored rows
        are refused: their exposure ends at an unknown time.
    n : array like, optional
        The count of each row: ``n = k`` is ``k`` identical units. Default
        1.
    tl : array like or float, optional
        Each unit's entry time (late entry, left truncation), on the same
        scale as ``x``: the unit is followed on ``(tl, x]``. Default 0.
    factors : dict, optional
        ``{name: labels}``: a fixed classification of the units, one label
        per row (strings, numbers or a ``pandas.Categorical``, whose
        category order is kept; otherwise the levels are sorted).
    timescales : dict, optional
        ``{name: (start, breaks)}`` (or ``{name: {"start": ..., "breaks":
        ...}}``): a time scale that runs with the follow-up. ``start`` is
        each unit's value on it at time 0 of ``x`` (a number, or one per
        row): its value at time ``t`` is ``start + t``. Age at time 0 for
        attained age; the calendar date at time 0 for calendar period; 0
        for the time scale of ``x`` itself; ``-tl`` for time since entry.
        ``breaks`` are the increasing band edges (the last may be ``inf``);
        the bands are ``(b_j, b_{j+1}]``, so an event exactly on an edge
        is in the band that ends there, as in R. Time before the first
        edge or after the last is *off the table*: counted in
        :attr:`PersonYears.offtable`, with a warning.
    rates : DataFrame, dict, float or None, optional
        Reference rates, to give each cell's expected events and the
        observed to expected ratio (a standardised incidence or mortality
        ratio). A single rate for every cell, or a table: a
        ``pandas.DataFrame`` (or a dict of columns) with a column
        ``"rate"`` and one column for each of the factors and time scales
        it varies by (a subset of them). A time scale's column holds each
        band's lower edge, as the table's does. Every cell must find one
        rate. The rates are in the units of the table's rates: events per
        ``per`` units of time.
    per : float, optional
        The rates are per ``per`` units of the time of ``x``: ``per=1000``
        gives failures per 1,000 hours when ``x`` is in hours. Default 1.
        The exposure stays in the units of ``x``.
    alpha_ci : float, optional
        The bounds are two-sided ``1 - alpha_ci`` intervals. Default 0.05.

    Returns
    -------
    PersonYears
        The table (:attr:`PersonYears.table`), one row per cell, every
        combination of the levels and bands (cells with no time
        included), with the columns:

        - one per factor (its label) and one per time scale (its band's
          lower edge), followed by ``<name>_end`` (the upper edge) for
          each time scale;
        - ``units``: the units (the sum of ``n``) that spent time in the
          cell; a unit is counted in each cell it passes through;
        - ``exposure``: the total time at risk in the cell, in the units
          of ``x``;
        - ``events``: the events in the cell;
        - ``rate``, ``rate_lower``, ``rate_upper``: ``events / exposure``
          times ``per``, with its exact Garwood bounds (``nan`` with no
          exposure);
        - with ``rates``: ``expected`` (the reference rate times the
          exposure, divided by ``per``), ``oe_ratio`` (``events /
          expected``) and its exact bounds ``oe_lower`` and ``oe_upper``
          (the Garwood bounds of the events divided by the expected).

    Notes
    -----
    Each unit's follow-up ``(tl, x]`` is cut where any of its time scales
    crosses a band edge, and each piece is added to the cell it lies in.
    The event rate of a cell is the maximum likelihood estimate of a
    constant hazard there, ``d / T``. Its exact bounds treat ``d`` as
    Poisson with mean ``lambda T`` (Garwood 1936): with ``chi2_{k, p}``
    the ``p`` quantile of a chi-square on ``k`` degrees of freedom,

    .. math::

        \left[\frac{\chi^2_{2d,\ \alpha/2}}{2T},\
        \frac{\chi^2_{2d+2,\ 1-\alpha/2}}{2T}\right].

    With ``rates``, the expected events of a cell are ``E = r T``, and
    the observed to expected ratio ``d / E`` has the bounds above times
    ``T / E`` (Breslow and Day 1987, section 2.3), ``E`` taken as known.
    The rate table's resolution is the table's: a reference rate that
    changes within a cell is approximated by one value there (R's
    ``pyears`` integrates a finer rate table along each unit's path);
    cut the time scales at the rate table's edges for the exact expected
    count, and add up the cells.

    References
    ----------
    Garwood, F. (1936), "Fiducial limits for the Poisson distribution",
    Biometrika 28, 437-442.

    Breslow, N. E. and Day, N. E. (1987), *Statistical Methods in Cancer
    Research, Volume II: The Design and Analysis of Cohort Studies*,
    IARC, chapter 2.

    Berry, G. (1983), "The analysis of mortality by the subject-years
    method", Biometrics 39, 173-184.

    Examples
    --------
    Four pumps at two sites, followed from installation (hours), with
    bands of operating age:

    >>> import surpyval
    >>> x = [1500, 4200, 2500, 6000]
    >>> c = [0, 1, 0, 0]
    >>> site = ["A", "A", "B", "B"]
    >>> py = surpyval.person_years(
    ...     x, c,
    ...     factors={"site": site},
    ...     timescales={"age": (0, [0, 2000, 4000, 8000])},
    ...     per=1000,
    ... )
    >>> cols = ["site", "age", "units", "exposure", "events", "rate"]
    >>> py.table[cols].round(3)
      site     age  units  exposure  events   rate
    0    A     0.0      2    3500.0       1  0.286
    1    A  2000.0      1    2000.0       0  0.000
    2    A  4000.0      1     200.0       0  0.000
    3    B     0.0      2    4000.0       0  0.000
    4    B  2000.0      2    2500.0       1  0.400
    5    B  4000.0      1    2000.0       1  0.500

    Site B's failure rate over site A's, with its exact bounds:

    >>> ratio = py.rate_ratio({"site": "B"}, {"site": "A"})
    >>> print(round(ratio["ratio"], 3), round(ratio["upper"], 1))
    1.341 79.1
    """
    import pandas as pd

    if not (isinstance(alpha_ci, (int, float)) and 0 < alpha_ci < 1):
        raise alpha_ci_error(alpha_ci)
    if (
        isinstance(per, bool)
        or not isinstance(per, (int, float, np.integer, np.floating))
        or not (np.isfinite(per) and per > 0)
    ):
        raise ValueError(
            "'per' must be a positive number, the time the rates are per "
            "(e.g. per=1000 for events per 1,000 hours); got {!r}.".format(per)
        )
    x_, c_, n_, entry = _data(x, c, n, tl)
    m = x_.size
    dims = _dimensions(factors, timescales, m)

    # Every point where a time scale crosses a band edge, inside (entry, x).
    cuts = [entry[:, None], x_[:, None]]
    for d in dims:
        if d["kind"] == "scale":
            t = d["breaks"][None, :] - d["start"][:, None]
            inside = (t > entry[:, None]) & (t < x_[:, None])
            cuts.append(np.where(inside, t, np.nan))
    points = np.sort(np.concatenate(cuts, axis=1), axis=1)
    lo, hi = points[:, :-1], points[:, 1:]
    length = hi - lo
    keep = np.isfinite(length) & (length > 0)
    row = np.broadcast_to(np.arange(m)[:, None], lo.shape)[keep]
    length = length[keep]
    middle = ((lo + hi) / 2)[keep]

    shape = tuple(d["size"] for d in dims)
    ncell = int(np.prod(shape)) if shape else 1
    seg_code, seg_on = _cells(dims, row, middle, side="right")
    end_code, end_on = _cells(dims, np.arange(m), x_, side="left")

    exposure = np.bincount(
        seg_code[seg_on], weights=(n_[row] * length)[seg_on], minlength=ncell
    )
    is_event = c_ == 0
    events = np.bincount(
        end_code[end_on & is_event],
        weights=n_[end_on & is_event],
        minlength=ncell,
    )
    pairs = np.unique(row[seg_on] * ncell + seg_code[seg_on])
    units = np.bincount(
        pairs % ncell, weights=n_[pairs // ncell], minlength=ncell
    )
    off_exposure = float((n_[row] * length)[~seg_on].sum())
    off_events = float(n_[~end_on & is_event].sum())
    in_table = np.zeros(m, dtype=bool)
    in_table[row[seg_on]] = True

    columns: dict[str, Any] = {}
    grid = np.unravel_index(np.arange(ncell), shape) if shape else ()
    for d, idx in zip(dims, grid):
        if d["kind"] == "factor":
            columns[d["name"]] = pd.Categorical.from_codes(
                idx, categories=d["levels"]
            )
        else:
            columns[d["name"]] = d["breaks"][idx]
    for d, idx in zip(dims, grid):
        if d["kind"] == "scale":
            columns[d["name"] + "_end"] = d["breaks"][idx + 1]
    columns["units"] = _whole(units)
    columns["exposure"] = exposure
    columns["events"] = _whole(events)
    table = pd.DataFrame(columns)
    table = _add_rates(table, per, alpha_ci)
    if rates is not None:
        reference = _reference_rates(rates, table, dims)
        table = _add_expected(table, reference, per, alpha_ci)

    if off_exposure > 0 or off_events > 0:
        scales = [d for d in dims if d["kind"] == "scale"]
        warnings.warn(
            "{:g} of the {:g} units of exposure, and {:g} of the {:g} "
            "events, fall outside the bands of the time scales ({}), and "
            "are not in the table; they are in .offtable. Start the first "
            "band lower or end the last at inf to count them.".format(
                off_exposure,
                off_exposure + float(exposure.sum()),
                off_events,
                off_events + float(events.sum()),
                ", ".join(d["name"] for d in scales),
            ),
            stacklevel=2,
        )
    return PersonYears(
        table=table,
        factors=[d["name"] for d in dims if d["kind"] == "factor"],
        timescales=[d["name"] for d in dims if d["kind"] == "scale"],
        per=float(per),
        alpha_ci=float(alpha_ci),
        offtable={"exposure": off_exposure, "events": _scalar(off_events)},
        units=_scalar(float(n_[in_table].sum())),
    )


class PersonYears:
    """
    An exposure (person-years) table, from :func:`person_years`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per cell; see :func:`person_years` for the columns.
    factors, timescales : list of str
        The names of the fixed factors and of the time scales, in the
        order of the table's columns.
    per : float
        The rates are events per ``per`` units of time.
    alpha_ci : float
        The bounds are two-sided ``1 - alpha_ci`` intervals.
    offtable : dict
        ``{"exposure": ..., "events": ...}``: the time and the events that
        fell outside every band of a time scale (R's ``offtable``).
    units : float
        The units with any time in the table, each counted once.

    Examples
    --------
    >>> import surpyval
    >>> py = surpyval.person_years(
    ...     [10.0, 20.0, 30.0, 40.0], [0, 0, 1, 0],
    ...     factors={"group": ["a", "a", "b", "b"]},
    ... )
    >>> py.margins()[["exposure", "events", "rate"]]
       exposure  events      rate
    0     100.0       3  0.03
    >>> ratio = py.rate_ratio({"group": "a"}, {"group": "b"})
    >>> round(ratio["ratio"], 4)
    4.6667
    """

    def __init__(
        self,
        table: pd.DataFrame,
        factors: list[str],
        timescales: list[str],
        per: float,
        alpha_ci: float,
        offtable: dict,
        units: float,
    ) -> None:
        self.table = table
        self.factors = list(factors)
        self.timescales = list(timescales)
        self.per = per
        self.alpha_ci = alpha_ci
        self.offtable = offtable
        self.units = units

    @property
    def dimensions(self) -> list[str]:
        """The names of the table's dimensions, factors first, in the
        order of the call."""
        return [c for c in self.table.columns if c in self._dims]

    @property
    def _dims(self) -> set:
        return set(self.factors) | set(self.timescales)

    @property
    def has_expected(self) -> bool:
        """Whether the table was given reference rates."""
        return "expected" in self.table.columns

    def margins(self, by: Any = None) -> pd.DataFrame:
        """
        The table added up over every dimension not in ``by``: the
        exposure, events (and expected events), with the rates (and
        observed to expected ratios) and their exact bounds recomputed.

        A unit passes through several cells, so the counts of units do not
        add up and are not given; :attr:`units` is the count for the whole
        table, and :func:`person_years` called with only the dimensions in
        ``by`` gives them per cell.

        Parameters
        ----------
        by : str or list of str, optional
            The factors and time scales to keep. Default none: one row,
            the whole table.

        Returns
        -------
        pandas.DataFrame
            One row per combination of the kept dimensions.
        """
        import pandas as pd

        keep = self._by(by)
        sums = ["exposure", "events"] + (
            ["expected"] if self.has_expected else []
        )
        if keep:
            ends = [k + "_end" for k in keep if k in self.timescales]
            out = (
                self.table.groupby(keep + ends, observed=False, sort=False)[
                    sums
                ]
                .sum()
                .reset_index()
            )
        else:
            out = pd.DataFrame({k: [self.table[k].sum()] for k in sums})
        out["events"] = _whole(out["events"].to_numpy(dtype=float))
        out = _add_rates(out, self.per, self.alpha_ci)
        if self.has_expected:
            out = _add_expected(out, None, self.per, self.alpha_ci)
        return out

    def rate_ratio(
        self, cells: Mapping[str, Any], reference: Mapping[str, Any]
    ) -> dict:
        """
        The ratio of the event rate in ``cells`` to that in ``reference``,
        with its exact confidence bounds and the exact test of equal rates.

        Each of ``cells`` and ``reference`` picks cells by their labels,
        ``{"site": "B"}`` or ``{"site": "B", "age": 2000}`` (a time scale
        by its band's lower edge); a dimension left out is added up over.
        Given the total ``N = d_1 + d_0`` events in the two, ``d_1`` is
        binomial with probability ``rho T_1 / (rho T_1 + T_0)``: the
        Clopper-Pearson bounds of that probability give the bounds of the
        ratio ``rho``, and the exact binomial test of ``T_1 / (T_1 +
        T_0)`` gives the p-value, as R's ``poisson.test`` with two counts
        (the conditional test, Breslow and Day 1987, section 2.4).

        Parameters
        ----------
        cells, reference : dict
            The labels picking the two sets of cells; they must not share
            a cell.

        Returns
        -------
        dict
            ``ratio``, ``lower``, ``upper`` (the ``1 - alpha_ci`` bounds),
            ``p_value`` (two-sided), and each side's ``events`` and
            ``exposure``. With no events in either, the ratio is ``nan``
            with bounds ``[0, inf]`` and ``p_value`` 1.
        """
        a = self._select(cells, "cells")
        b = self._select(reference, "reference")
        if np.any(a & b):
            raise ValueError(
                "'cells' and 'reference' pick {:d} cells in common; pick two "
                "sets that do not overlap (for example {{'site': 'A'}} and "
                "{{'site': 'B'}}).".format(int(np.sum(a & b)))
            )
        d1 = float(self.table["events"][a].sum())
        d0 = float(self.table["events"][b].sum())
        T1 = float(self.table["exposure"][a].sum())
        T0 = float(self.table["exposure"][b].sum())
        if not (T1 > 0 and T0 > 0):
            raise ValueError(
                "The {} have no exposure: there is no rate to compare. Pick "
                "cells where the units spent time.".format(
                    "cells" if not T1 > 0 else "reference cells"
                )
            )
        ratio, lower, upper, p = _rate_ratio(d1, T1, d0, T0, self.alpha_ci)
        return {
            "ratio": ratio,
            "lower": lower,
            "upper": upper,
            "p_value": p,
            "events": d1,
            "exposure": T1,
            "reference_events": d0,
            "reference_exposure": T0,
        }

    def _by(self, by: Any) -> list[str]:
        if by is None:
            return []
        keep = [by] if isinstance(by, str) else list(by)
        unknown = [k for k in keep if k not in self._dims]
        if unknown:
            raise ValueError(
                "'by' names {} that the table does not have; its dimensions "
                "are {}.".format(unknown, self.dimensions)
            )
        return [k for k in self.dimensions if k in keep]

    def _select(self, labels: Any, name: str) -> npt.NDArray:
        if not isinstance(labels, Mapping) or not labels:
            raise ValueError(
                "'{}' must be a dict of labels picking cells, e.g. "
                "{{'site': 'B'}}; got {!r}.".format(name, labels)
            )
        mask = np.ones(len(self.table), dtype=bool)
        for k, v in labels.items():
            if k not in self._dims:
                raise ValueError(
                    "'{}' names {!r}, which is not a dimension of the table; "
                    "they are {}.".format(name, k, self.dimensions)
                )
            col = self.table[k]
            hit = np.asarray(col == v, dtype=bool)
            if not hit.any():
                shown = (
                    list(col.cat.categories)
                    if k in self.factors
                    else sorted(set(col.tolist()))
                )
                raise ValueError(
                    "'{}': {!r} is not a level of {!r}; the levels are {} "
                    "(a time scale's band is named by its lower "
                    "edge).".format(name, v, k, shown)
                )
            mask &= hit
        return mask

    def to_dict(self) -> dict:
        """The table as a dict of native Python values, strict JSON: a
        missing or infinite number is written as ``None`` and recorded
        under ``"non_finite"``.

        Returns
        -------
        dict
            The settings and the table, column by column.
        """
        from surpyval.serialisation import encode_non_finite, to_native

        columns = {}
        levels = {}
        for k in self.table.columns:
            col = self.table[k]
            if k in self.factors:
                levels[k] = [to_native(v) for v in col.cat.categories]
                columns[k] = [int(v) for v in col.cat.codes]
            else:
                columns[k] = to_native(col.to_numpy())
        out = {
            "result": type(self).__name__,
            "schema": 1,
            "factors": list(self.factors),
            "timescales": list(self.timescales),
            "levels": levels,
            "per": float(self.per),
            "alpha_ci": float(self.alpha_ci),
            "units": to_native(self.units),
            "offtable": {k: to_native(v) for k, v in self.offtable.items()},
            "columns": list(self.table.columns),
            "table": columns,
        }
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> PersonYears:
        """The table :meth:`to_dict` wrote.

        Parameters
        ----------
        result_dict : dict
            A dict written by :meth:`to_dict`.

        Returns
        -------
        PersonYears
        """
        import pandas as pd

        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                "Must create a PersonYears from a dict its to_dict wrote "
                "(with 'result': 'PersonYears')."
            )
        for key in ("factors", "timescales", "per", "alpha_ci", "table"):
            if key not in result_dict:
                raise ValueError(
                    "The dict has no {!r}; it was not written by "
                    "PersonYears.to_dict.".format(key)
                )
        d = decode_non_finite(dict(result_dict))
        data: dict[str, Any] = {}
        for k in d["columns"]:
            v = d["table"][k]
            if k in d["factors"]:
                data[k] = pd.Categorical.from_codes(
                    v, categories=d["levels"][k]
                )
            elif k in ("units", "events"):
                data[k] = _whole(np.asarray(v, dtype=float))
            else:
                data[k] = np.asarray(v, dtype=float)
        return cls(
            table=pd.DataFrame(data, columns=d["columns"]),
            factors=d["factors"],
            timescales=d["timescales"],
            per=float(d["per"]),
            alpha_ci=float(d["alpha_ci"]),
            offtable=dict(d["offtable"]),
            units=d["units"],
        )

    def __repr__(self) -> str:
        level = 100 * (1 - self.alpha_ci)
        total = self.margins().iloc[0]
        out = (
            "Exposure table: {:g} units, {:g} units of time, {:g} events; "
            "rates per {:g}, {:g}% exact bounds\n".format(
                self.units,
                total["exposure"],
                total["events"],
                self.per,
                level,
            )
        )
        if self.has_expected:
            out += (
                "Observed / expected: {:g} / {:.4g} = {:.4g} [{:.4g}, "
                "{:.4g}]\n".format(
                    total["events"],
                    total["expected"],
                    total["oe_ratio"],
                    total["oe_lower"],
                    total["oe_upper"],
                )
            )
        if self.offtable["exposure"] > 0 or self.offtable["events"] > 0:
            out += "Off the table: {:g} time, {:g} events\n".format(
                self.offtable["exposure"], self.offtable["events"]
            )
        return out + self.table.to_string(index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(v: float) -> float | int:
    return int(v) if float(v).is_integer() else float(v)


def _whole(v: npt.NDArray) -> npt.NDArray:
    """Counts as integers where they are whole (they are unless ``n``
    is not)."""
    v = np.asarray(v, dtype=float)
    if np.all(v == np.round(v)):
        return v.astype(int)
    return v


def _data(
    x: Any, c: Any, n: Any, tl: Any
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Times, flags, counts and entry times, in the order given (the
    factors and start values are matched to the rows by position),
    refusing what a table of exposure cannot hold."""
    if isinstance(x, SurpyvalData):
        raise TypeError(
            "person_years takes arrays, x, c, n and tl, in the order of the "
            "factors and the time scales' start values; a SurpyvalData "
            "holds its rows sorted, so they could not be matched. Pass "
            "data.x, data.c, data.n and data.tl with labels in that order, "
            "or the original arrays."
        )
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 2 and x_arr.shape[1] == 2:
        raise ValueError(
            "x has two columns (interval-censored rows); " + _INTERVAL_FIX
        )
    x_arr = np.atleast_1d(x_arr)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(
            "x must be a 1-D array of the times follow-up ends, with at "
            "least one row; got shape {}.".format(x_arr.shape)
        )
    m = x_arr.size
    if not np.isfinite(x_arr).all():
        raise ValueError(
            "x is missing or infinite for {:d} rows; every unit needs the "
            "time its follow-up ends. Leave those rows out.".format(
                int((~np.isfinite(x_arr)).sum())
            )
        )
    flags = np.zeros(m) if c is None else np.asarray(c, dtype=float)
    flags = _per_row(flags, m, "c")
    if not np.isin(flags, (-1, 0, 1, 2)).all():
        raise ValueError(
            "c must be 0 (event) or 1 (right censored); got values {}.".format(
                sorted(set(flags[~np.isin(flags, (-1, 0, 1, 2))].tolist()))
            )
        )
    bad = (flags == -1) | (flags == 2)
    if bad.any():
        raise ValueError(
            "{:d} rows are left (c=-1) or interval (c=2) censored; ".format(
                int(bad.sum())
            )
            + _INTERVAL_FIX
        )
    counts = np.ones(m) if n is None else np.asarray(n, dtype=float)
    counts = _per_row(counts, m, "n")
    if not (np.isfinite(counts).all() and (counts > 0).all()):
        raise ValueError(
            "n must be positive counts, one per row; {:d} are not.".format(
                int((~(np.isfinite(counts) & (counts > 0))).sum())
            )
        )
    entry = np.zeros(m) if tl is None else np.asarray(tl, dtype=float)
    entry = _per_row(entry, m, "tl")
    entry = np.where(np.isneginf(entry), 0.0, entry)
    if not np.isfinite(entry).all():
        raise ValueError(
            "tl is missing or +inf for {:d} rows; give each unit its entry "
            "time (0 if followed from the origin).".format(
                int((~np.isfinite(entry)).sum())
            )
        )
    short = x_arr <= entry
    if short.any():
        raise ValueError(
            "{:d} rows end at or before their entry (x <= tl, or x <= 0 "
            "with no 'tl'), from x = {:g}; follow-up runs from 'tl' "
            "(default 0) to 'x'. Give the entry time in 'tl' if the times "
            "are not from 0.".format(
                int(short.sum()), float(x_arr[short].min())
            )
        )
    return x_arr, flags.astype(int), counts, entry


_INTERVAL_FIX = (
    "person_years ends each unit's exposure at its event or censoring "
    "time, and puts the event in the band it is in then, so it needs "
    "those times. If a failure was found at an inspection, record it as an "
    "event (c=0) at the inspection, as the exposure is counted to there; "
    "or estimate the survival with surpyval.Turnbull.fit."
)


def _per_row(values: Any, m: int, what: str) -> npt.NDArray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.full(m, arr)
    arr = arr.ravel()
    if arr.size != m:
        raise ValueError(
            "{} has {:d} values; it needs one per row of x ({:d}).".format(
                what, arr.size, m
            )
        )
    return arr


def _check_name(name: Any, seen: set) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(
            "The names of factors and time scales must be non-empty "
            "strings; got {!r}.".format(name)
        )
    if name in _RESERVED or name.endswith("_end") or name in seen:
        raise ValueError(
            "{!r} cannot name a factor or time scale: it is {} Rename "
            "it.".format(
                name,
                (
                    "used twice."
                    if name in seen
                    else "a column of the table (or ends in '_end')."
                ),
            )
        )
    seen.add(name)
    return name


def _dimensions(factors: Any, timescales: Any, m: int) -> list[dict]:
    """The factors and time scales, validated, as dicts."""
    import pandas as pd

    dims: list[dict] = []
    seen: set = set()
    for arg, given in (("factors", factors), ("timescales", timescales)):
        if given is not None and not isinstance(given, Mapping):
            raise ValueError(
                "'{}' must be a dict {{name: ...}}; got {}.".format(
                    arg, type(given).__name__
                )
            )
    for name, labels in (factors or {}).items():
        _check_name(name, seen)
        if isinstance(labels, (pd.Categorical, pd.Series)) and isinstance(
            getattr(labels, "dtype", None), pd.CategoricalDtype
        ):
            cat = pd.Categorical(labels)
        else:
            raw = _per_row(labels, m, f"factors[{name!r}]")
            cat = pd.Categorical(list(raw))
        if len(cat) == 1 and m > 1:
            cat = pd.Categorical(list(cat) * m, categories=cat.categories)
        if len(cat) != m:
            raise ValueError(
                "factors[{!r}] has {:d} labels; it needs one per row of x "
                "({:d}).".format(name, len(cat), m)
            )
        missing = np.asarray(cat.codes) < 0
        if missing.any():
            raise ValueError(
                "factors[{!r}] is missing for {:d} rows; give them a label "
                "of their own (e.g. 'unknown') or leave the rows "
                "out.".format(name, int(missing.sum()))
            )
        dims.append(
            {
                "kind": "factor",
                "name": name,
                "codes": np.asarray(cat.codes, dtype=int),
                "levels": cat.categories,
                "size": len(cat.categories),
            }
        )
    for name, spec in (timescales or {}).items():
        _check_name(name, seen)
        if isinstance(spec, Mapping):
            if "breaks" not in spec:
                raise ValueError(
                    "timescales[{!r}] needs 'breaks', the band edges; "
                    "e.g. {{'start': age, 'breaks': [40, 50, 60]}}.".format(
                        name
                    )
                )
            start, breaks = spec.get("start", 0.0), spec["breaks"]
        elif isinstance(spec, tuple) and len(spec) == 2:
            start, breaks = spec
        else:
            raise ValueError(
                "timescales[{!r}] must be (start, breaks): each unit's "
                "value on the scale at time 0 and the band edges, e.g. "
                "(age_at_start, [40, 50, 60, 70]); use (0, breaks) for "
                "bands of x itself.".format(name)
            )
        b = np.asarray(breaks, dtype=float).ravel()
        if (
            b.size < 2
            or np.isnan(b).any()
            or not np.all(np.diff(b) > 0)
            or not np.isfinite(b[:-1]).all()
        ):
            raise ValueError(
                "timescales[{!r}]: the breaks must be at least two "
                "increasing edges, finite but for a last inf, e.g. "
                "[0, 10, 20, inf]; got {}.".format(name, list(b))
            )
        s = _per_row(start, m, f"timescales[{name!r}] start")
        try:
            s = s.astype(float)
        except (TypeError, ValueError):
            raise ValueError(
                "timescales[{!r}]: start must be numbers, each unit's value "
                "on the scale at time 0 (convert dates to a number of "
                "years or days in the unit of x).".format(name)
            ) from None
        if not np.isfinite(s).all():
            raise ValueError(
                "timescales[{!r}]: start is missing or infinite for {:d} "
                "rows; give every unit its value at time 0, or leave the "
                "rows out.".format(name, int((~np.isfinite(s)).sum()))
            )
        dims.append(
            {
                "kind": "scale",
                "name": name,
                "start": s,
                "breaks": b,
                "size": b.size - 1,
            }
        )
    return dims


def _cells(
    dims: list[dict],
    row: npt.NDArray,
    t: npt.NDArray,
    side: Literal["left", "right"],
) -> tuple[npt.NDArray, npt.NDArray]:
    """The flat cell index of each (row, time), and whether it is on the
    table. ``side="left"`` puts a value on an edge in the band ending
    there (an event), ``"right"`` in the one starting there."""
    on = np.ones(row.size, dtype=bool)
    idx = []
    for d in dims:
        if d["kind"] == "factor":
            idx.append(d["codes"][row])
        else:
            v = d["start"][row] + t
            j = np.searchsorted(d["breaks"], v, side=side) - 1
            on &= (j >= 0) & (j < d["size"])
            idx.append(np.clip(j, 0, d["size"] - 1))
    if not dims:
        return np.zeros(row.size, dtype=int), on
    shape = tuple(d["size"] for d in dims)
    return np.ravel_multi_index(tuple(idx), shape), on


def _add_rates(
    table: pd.DataFrame, per: float, alpha_ci: float
) -> pd.DataFrame:
    T = table["exposure"].to_numpy(dtype=float)
    d = table["events"].to_numpy(dtype=float)
    lower, upper = poisson_rate_bounds(d, T, alpha_ci)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["rate"] = np.where(T > 0, d / np.where(T > 0, T, 1.0), np.nan)
    table["rate"] = table["rate"] * per
    table["rate_lower"] = lower * per
    table["rate_upper"] = upper * per
    return table


def _add_expected(
    table: pd.DataFrame,
    reference: npt.NDArray | None,
    per: float,
    alpha_ci: float,
) -> pd.DataFrame:
    if reference is not None:
        table["expected"] = reference * table["exposure"].to_numpy() / per
    E = table["expected"].to_numpy(dtype=float)
    d = table["events"].to_numpy(dtype=float)
    lower, upper = poisson_rate_bounds(d, E, alpha_ci)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["oe_ratio"] = np.where(
            E > 0, d / np.where(E > 0, E, 1.0), np.nan
        )
    table["oe_lower"] = lower
    table["oe_upper"] = upper
    return table


def _reference_rates(
    rates: Any, table: pd.DataFrame, dims: list[dict]
) -> npt.NDArray:
    """The reference rate of each cell of ``table``."""
    import pandas as pd

    if isinstance(rates, (int, float, np.integer, np.floating)) and not (
        isinstance(rates, bool)
    ):
        value = float(rates)
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(
                f"'rates' must be a rate of at least 0; got {rates!r}."
            )
        return np.full(len(table), value)
    if isinstance(rates, Mapping):
        rates = pd.DataFrame(dict(rates))
    if not isinstance(rates, pd.DataFrame):
        raise ValueError(
            "'rates' must be a number, a dict of columns or a "
            "pandas.DataFrame with a 'rate' column; got {}.".format(
                type(rates).__name__
            )
        )
    if "rate" not in rates.columns:
        raise ValueError(
            "'rates' needs a 'rate' column (events per 'per' units of "
            "time); its columns are {}.".format(list(rates.columns))
        )
    names = [d["name"] for d in dims]
    keys = [k for k in rates.columns if k != "rate"]
    unknown = [k for k in keys if k not in names]
    if unknown:
        raise ValueError(
            "'rates' has columns {} that are not factors or time scales of "
            "the table ({}); a rate table varies by some of those and has a "
            "'rate' column.".format(unknown, names)
        )
    r = rates["rate"].to_numpy(dtype=float)
    if not np.all(np.isfinite(r) & (r >= 0)):
        raise ValueError(
            "'rates' has rates that are missing, negative or infinite; "
            "every rate must be a number of at least 0."
        )
    if not keys:
        if len(rates) != 1:
            raise ValueError(
                "'rates' has no columns to match the cells by but {:d} "
                "rows; give one rate, or columns naming the factors and "
                "time scales it varies by.".format(len(rates))
            )
        return np.full(len(table), r[0])
    left = pd.DataFrame(
        {
            k: (
                table[k].astype(object)
                if isinstance(table[k].dtype, pd.CategoricalDtype)
                else table[k]
            )
            for k in keys
        }
    )
    right = rates[keys + ["rate"]].copy()
    for k in keys:
        if isinstance(right[k].dtype, pd.CategoricalDtype):
            right[k] = right[k].astype(object)
    if right.duplicated(keys).any():
        raise ValueError(
            "'rates' gives {:d} cells more than one rate (rows repeated on "
            "{}); give each combination one rate.".format(
                int(right.duplicated(keys).sum()), keys
            )
        )
    merged = left.merge(right, on=keys, how="left", sort=False)
    missing = merged["rate"].isna().to_numpy()
    if missing.any():
        first = merged.loc[missing, keys].iloc[0].to_dict()
        raise ValueError(
            "'rates' has no rate for {:d} cells, the first {}; give a rate "
            "for every combination of {} (a time scale by its band's lower "
            "edge).".format(int(missing.sum()), first, keys)
        )
    return merged["rate"].to_numpy(dtype=float)


def _rate_ratio(
    d1: float, T1: float, d0: float, T0: float, alpha_ci: float
) -> tuple[float, float, float, float]:
    """The exact conditional (Clopper-Pearson) bounds of a ratio of two
    Poisson rates and the exact binomial test of equal rates."""
    from scipy import stats

    N = d1 + d0
    if N == 0:
        return float("nan"), 0.0, float("inf"), 1.0
    a = alpha_ci / 2
    p_lo = stats.beta.ppf(a, d1, d0 + 1) if d1 > 0 else 0.0
    p_hi = stats.beta.ppf(1 - a, d1 + 1, d0) if d0 > 0 else 1.0
    scale = T0 / T1

    def odds(p: float) -> float:
        return float("inf") if p >= 1 else p / (1 - p) * scale

    ratio = odds(d1 / N)
    p0 = T1 / (T1 + T0)
    if float(d1).is_integer() and float(d0).is_integer():
        p_value = float(stats.binomtest(int(d1), int(N), p0).pvalue)
    else:
        p_value = float("nan")
    return ratio, odds(float(p_lo)), odds(float(p_hi)), p_value
