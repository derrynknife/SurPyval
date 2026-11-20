"""The data of a relative survival fit: times, censoring and counts, each
unit's age, calendar year and stratum at time 0 (to read its expected
hazard from the rate table), and the covariates of a regression."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.data_formats import refuse_time_values
from surpyval.utils.warnings import caller_stacklevel

from .rate_table import RateTable

#: The refusal of a SurpyvalData by the relative survival fits.
REFUSAL = (
    "The relative survival fits take arrays, not a SurpyvalData: each "
    "unit's age (and calendar year and stratum) at time 0 must line up "
    "with its row, and a SurpyvalData groups and sorts its rows. Pass x, c "
    "and n with age=, year= and stratum= (one per row), or the columns to "
    "fit_from_df."
)


@dataclass
class RelativeData:
    """The checked data: one row per group of identical units."""

    x: npt.NDArray
    c: npt.NDArray
    n: npt.NDArray
    age: npt.NDArray
    year: npt.NDArray
    s: npt.NDArray
    Z: npt.NDArray | None

    @property
    def size(self) -> int:
        return int(self.x.size)


def _numbers(values: Any, name: str, size: int) -> npt.NDArray:
    refuse_time_values(values, name)
    try:
        out = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"'{name}' must be numbers, one per row of x (or one for "
            f"every row); got {values!r}."
        ) from None
    if out.ndim == 0:
        out = np.full(size, float(out))
    if out.ndim != 1 or out.size != size:
        raise ValueError(
            f"'{name}' has shape {np.shape(values)}; give one value per row "
            f"of x ({size}), or one for every row."
        )
    return out


def _missing_label(values: npt.NDArray) -> npt.NDArray:
    out = np.zeros(values.size, dtype=bool)
    for i, value in enumerate(values):
        if value is None:
            out[i] = True
            continue
        try:
            out[i] = bool(np.isnan(value))
        except TypeError:
            out[i] = False
    return out


def _times(x: Any) -> npt.NDArray:
    refuse_time_values(x, "x")
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"'x' must be numbers (times); got {x!r}.") from None
    if x_arr.ndim != 1:
        raise ValueError(
            "'x' must be 1-D, one follow-up time per row; got shape "
            f"{x_arr.shape}. Interval-censored times are not taken by the "
            "relative survival fits."
        )
    if x_arr.size == 0:
        raise ValueError("'x' is empty: there is nothing to fit.")
    if not np.all(np.isfinite(x_arr)):
        bad = np.flatnonzero(~np.isfinite(x_arr))
        raise ValueError(
            f"'x' must be finite: {bad.size} row(s) are missing (NaN) or "
            f"infinite (the first is row {bad[0]}). Give every row its "
            "time."
        )
    if np.any(x_arr < 0):
        raise ValueError(
            "'x' must be at least 0: the follow-up time since the date at "
            "which each unit's age and year are given."
        )
    return x_arr


def _censoring_and_counts(
    c: Any, n: Any, size: int
) -> tuple[npt.NDArray, npt.NDArray]:
    c_arr = np.zeros(size) if c is None else _numbers(c, "c", size)
    if not np.all(np.isin(c_arr, (0, 1))):
        raise ValueError(
            "'c' must be 0 (an event) or 1 (right censored): the relative "
            "survival fits take exact and right-censored times, not left "
            "(-1) or interval (2) censoring."
        )
    n_arr = np.ones(size) if n is None else _numbers(n, "n", size)
    if not np.all(np.isfinite(n_arr)) or np.any(n_arr <= 0):
        raise ValueError("'n' must be positive counts (whole numbers).")
    if np.any(n_arr != np.round(n_arr)):
        raise ValueError(
            "'n' must be whole numbers: it counts identical units."
        )
    return c_arr, n_arr


def _years(rate_table: RateTable, year: Any, size: int) -> npt.NDArray:
    if rate_table.years is None:
        if year is not None:
            raise ValueError(
                "The rate table has no calendar-year bands, so 'year' must "
                "be left out."
            )
        return np.zeros(size)
    if year is None:
        raise ValueError(
            "The rate table is by calendar year: give each unit's 'year' "
            "at time 0 (in the table's unit, e.g. 2011.5)."
        )
    return _numbers(year, "year", size)


def _labels(stratum: Any, size: int) -> npt.NDArray | None:
    if stratum is None:
        return None
    labels = np.asarray(stratum, dtype=object)
    if labels.ndim == 0:
        labels = np.full(size, labels.item(), dtype=object)
    labels = labels.reshape(-1)
    if labels.size != size:
        raise ValueError(
            f"'stratum' has {labels.size} values; give one per row of x "
            f"({size}), or one for every row."
        )
    return labels


def _covariates(Z: Any, size: int) -> npt.NDArray | None:
    if Z is None:
        return None
    try:
        Z_arr = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "'Z' must be numbers: one row of covariates per row of x."
        ) from None
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    if Z_arr.ndim != 2 or Z_arr.shape[0] != size or Z_arr.shape[1] == 0:
        raise ValueError(
            f"'Z' has shape {np.shape(Z)}; give one row of covariates per "
            f"row of x ({size} rows), one column per covariate."
        )
    return Z_arr


def check_data(
    x: Any,
    c: Any,
    n: Any,
    rate_table: Any,
    age: Any,
    year: Any,
    stratum: Any,
    Z: Any = None,
) -> RelativeData:
    """Check and clean the data of a relative survival fit.

    A row with a missing age, year, stratum or covariate is dropped, with
    one warning; a missing or infinite time is refused.
    """
    if not isinstance(rate_table, RateTable):
        raise ValueError(
            "'rate_table' must be a surpyval.RateTable (the population "
            "hazards by age, and by year and stratum where they apply); "
            f"got {type(rate_table).__name__}."
        )
    x_arr = _times(x)
    size = x_arr.size
    c_arr, n_arr = _censoring_and_counts(c, n, size)
    if age is None:
        raise ValueError(
            "'age' is required: each unit's age at time 0, in the rate "
            "table's unit, to read its expected hazard."
        )
    age_arr = _numbers(age, "age", size)
    year_arr = _years(rate_table, year, size)
    labels = _labels(stratum, size)
    Z_arr = _covariates(Z, size)
    missing = ~np.isfinite(age_arr) | ~np.isfinite(year_arr)
    if labels is not None:
        missing |= _missing_label(labels)
    if Z_arr is not None:
        missing |= ~np.all(np.isfinite(Z_arr), axis=1)
    dropped = int(missing.sum())
    if dropped == size:
        raise ValueError(
            "Every row has a missing age, year, stratum or covariate: "
            "there is nothing to fit."
        )
    if dropped:
        warnings.warn(
            f"Dropped {dropped} of {size} rows with a missing (NaN) or "
            "infinite age, year, stratum or covariate value.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    keep = ~missing
    s = rate_table.stratum_index(
        None if labels is None else labels[keep], int(keep.sum())
    )
    x_arr, c_arr, n_arr = x_arr[keep], c_arr[keep], n_arr[keep]
    if np.any((x_arr == 0) & (c_arr == 0)):
        raise ValueError(
            "An event at time 0 has no expected hazard to compare with: "
            "every event time must be above 0."
        )
    if not np.any(c_arr == 0):
        raise ValueError(
            "There are no events (every row is right censored): there is "
            "nothing to estimate the excess hazard from."
        )
    return RelativeData(
        x=x_arr,
        c=c_arr.astype(int),
        n=n_arr,
        age=age_arr[keep],
        year=year_arr[keep],
        s=s,
        Z=None if Z_arr is None else Z_arr[keep],
    )


def expected_on_grid(
    table: RateTable, data: RelativeData, grid: npt.NDArray
) -> Iterator[tuple[int, int, npt.NDArray]]:
    """Each unit's expected cumulative hazard at the grid times up to its
    own time: (i, m, H) for each unit i, H its cumulative hazard at
    grid[:m] (grid[m - 1] being its own time)."""
    stop = np.searchsorted(grid, data.x, side="right")
    step = table._path_chunks()
    for start in range(0, data.size, step):
        part = np.arange(start, min(start + step, data.size))
        nodes, H = table._path(
            data.age[part], data.year[part], data.s[part], data.x[part]
        )
        for k, i in enumerate(part):
            m = int(stop[i])
            yield int(i), m, np.interp(grid[:m], nodes[k], H[k])
