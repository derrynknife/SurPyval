"""Semi-parametric regression for repeated events.

Four models that leave the baseline unspecified, as a Cox model does, and
differ in what they assume about an item's second and later events:

- :data:`AndersenGill`: one baseline intensity for every event, on the
  item's age (time since the start of its observation). An item is at
  risk of its next event from the moment of the last one.
- :data:`PrenticeWilliamsPeterson`: a baseline of its own for each event
  number, and an item is at risk of its ``k``-th event only after its
  ``(k - 1)``-th. The clock is the item's age (``time_scale="total"``)
  or the time since its previous event (``time_scale="gap"``).
- :data:`WeiLinWeissfeld`: a marginal model of the time, from the start,
  to each item's first, second, ... event, each with its own baseline;
  an item is at risk of every one of them from the start.

Each is the Cox partial likelihood on its own layout of the rows (the
risk interval of each row, and its stratum), so the fit is
:data:`~surpyval.CoxPH`'s, with its tie handling, aliasing and check of
the maximum; there is no second likelihood here. What the layouts share
is that one item contributes several rows, which are not independent,
so the coefficients' covariance is the sandwich with each item a
cluster (Lin and Wei 1989), as R's ``coxph(... + cluster(id))``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.serialisation import (
    SerialisableMixin,
    dict_numbers,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._aliasing import (
    covariate_columns,
    dataframe_covariates,
)
from surpyval.univariate.regression._summary import (
    coefficient_names,
    coefficient_table,
)
from surpyval.utils.dataframe import (
    check_columns,
    frame_column,
    require_frame,
)
from surpyval.utils.deprecation import ambiguous_item_ids
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.recurrent_data import (
    accepts_recurrent_surpyval_data,
    notes_text,
    takes_terminal_column,
)
from surpyval.utils.units import label_axes, takes_unit
from surpyval.utils.validation import check_alpha_ci

ANDERSEN_GILL = "Andersen-Gill"
PWP_TOTAL = "Prentice-Williams-Peterson, total time"
PWP_GAP = "Prentice-Williams-Peterson, gap time"
WLW = "Wei-Lin-Weissfeld"

#: How each kind is refitted, for the refusals of a restored model.
_FITTER_NAME = {
    ANDERSEN_GILL: "AndersenGill.fit",
    PWP_TOTAL: "PrenticeWilliamsPeterson.fit",
    PWP_GAP: 'PrenticeWilliamsPeterson.fit(..., time_scale="gap")',
    WLW: "WeiLinWeissfeld.fit",
}

#: What each kind's cumulative baseline is, for the axis of ``plot``.
_CURVE = {
    ANDERSEN_GILL: "Expected number of events",
    PWP_TOTAL: "Cumulative hazard of the k-th event, by age",
    PWP_GAP: "Cumulative hazard of the k-th event, by time since the last",
    WLW: "Cumulative hazard of the k-th event",
}


def _vector(values: Any, name: str, nrow: int | None = None) -> npt.NDArray:
    out = np.asarray(values)
    if out.ndim != 1:
        raise ValueError(
            f"`{name}` must be one-dimensional, one value per row; got an "
            f"array of shape {out.shape}."
        )
    if nrow is not None and out.shape[0] != nrow:
        raise ValueError(
            f"`{name}` must have one value per row ({nrow}); got "
            f"{out.shape[0]}."
        )
    return out


def _times(values: Any, name: str, nrow: int | None = None) -> npt.NDArray:
    try:
        out = np.asarray(_vector(values, name, nrow), dtype=float)
    except (TypeError, ValueError) as error:
        if "one" in str(error) and "`" in str(error):
            raise
        raise ValueError(f"`{name}` must be numbers (times).") from None
    if not np.all(np.isfinite(out)):
        raise ValueError(
            "`{}` has {} missing or infinite value(s). A row is part of "
            "its item's history, so it cannot be dropped: give every row "
            "a finite time, or remove the whole item.".format(
                name, int((~np.isfinite(out)).sum())
            )
        )
    return out


def _flags(c: Any, nrow: int) -> npt.NDArray:
    if c is None:
        return np.zeros(nrow, dtype=int)
    raw = _times(c, "c", nrow)
    if not np.isin(raw, (0, 1)).all():
        raise ValueError(
            "`c` must be 0 (the row ends in an event) or 1 (it does not: "
            "the end of an item's observation); got {}. A partial "
            "likelihood has no term for a left-censored (-1) or counted / "
            "interval (2) row: fit those with ProportionalIntensityNHPP."
            "".format(sorted(set(raw[~np.isin(raw, (0, 1))].tolist())))
        )
    return raw.astype(int)


def _covariates(Z: Any, nrow: int) -> npt.NDArray:
    try:
        out = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "`Z` must be numbers, one row of covariates per row of the "
            "data; encode a category as 0 / 1 columns."
        ) from None
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[0] != nrow:
        raise ValueError(
            f"`Z` must have one row of covariates per row of the data "
            f"({nrow}); got an array of shape {np.shape(Z)}."
        )
    if out.shape[1] == 0:
        raise ValueError("`Z` must have at least one covariate column.")
    if not np.all(np.isfinite(out)):
        raise ValueError(
            "`Z` has {} row(s) with a missing or infinite covariate. A "
            "row is part of its item's history, so it cannot be dropped: "
            "fill the covariate in, or remove the whole item.".format(
                int((~np.isfinite(out).all(axis=1)).sum())
            )
        )
    return out


def _items(i: Any, nrow: int) -> npt.NDArray:
    if i is None:
        return np.zeros(nrow, dtype=int)
    out = _vector(i, "i", nrow)
    if pd.isna(out).any():
        raise ValueError(
            "`i` has {} missing item label(s); every row must say which "
            "item it belongs to.".format(int(pd.isna(out).sum()))
        )
    return out


def _item_codes(i: npt.NDArray) -> npt.NDArray:
    """An integer per row, the same for the rows of one item."""
    return pd.factorize(i, sort=False)[0]


def rows_from_event_times(
    x: Any, Z: Any, i: Any = None, c: Any = None, n: Any = None
) -> dict[str, npt.NDArray]:
    """The start-stop rows of an event log.

    ``x`` holds each item's event times and, flagged ``c = 1``, the end
    of its observation; ``Z`` the covariates in force up to each row's
    time. An item is observed from 0, so each row becomes the interval
    from its item's previous row (0 for the first) to its own time.
    Returns ``i``, ``xl``, ``xr``, ``c`` and ``Z``, sorted by item and
    time.

    Examples
    --------
    >>> from surpyval.recurrent.regression.recurrent_cox import (
    ...     rows_from_event_times,
    ... )
    >>> rows = rows_from_event_times(
    ...     [5, 9, 12, 7, 12], [[0], [0], [0], [1], [1]],
    ...     i=[1, 1, 1, 2, 2], c=[0, 0, 1, 0, 1],
    ... )
    >>> rows["xl"]
    array([0., 5., 9., 0., 7.])
    >>> rows["xr"]
    array([ 5.,  9., 12.,  7., 12.])
    """
    x = _times(x, "x")
    nrow = x.shape[0]
    if nrow == 0:
        raise ValueError("`x` is empty; there are no rows to fit.")
    Z, i, c = _covariates(Z, nrow), _items(i, nrow), _flags(c, nrow)
    if n is not None:
        counts = _times(n, "n", nrow)
        if not np.all(counts == 1):
            raise ValueError(
                "`n` must be 1 on every row: each row is one event or one "
                "end of observation of one item. Give each event its own "
                "row, and each item its own label in `i`."
            )
    if (x < 0).any():
        raise ValueError(
            "`x` has {} negative time(s); an item is observed from 0. "
            "For an item that enters later, give start-stop rows to "
            "fit_tvc.".format(int((x < 0).sum()))
        )
    codes = _item_codes(i)
    # Within an item by time, an event before a censoring at its time.
    order = np.lexsort((c, x, codes))
    x, Z, i, c, codes = x[order], Z[order], i[order], c[order], codes[order]
    first = np.r_[True, codes[1:] != codes[:-1]]
    xl = np.where(first, 0.0, np.r_[0.0, x[:-1]])
    empty = xl == x
    if (empty & (c == 0)).any():
        raise ValueError(
            "{} event(s) fall at the time of their item's previous row "
            "(or at 0). A partial likelihood needs each event at a later "
            "time than the one before: separate the times, or record "
            "the events as one.".format(int((empty & (c == 0)).sum()))
        )
    # An end of observation at the last event's time adds no time at risk
    keep = ~empty
    return {
        "i": i[keep],
        "xl": xl[keep],
        "xr": x[keep],
        "c": c[keep],
        "Z": Z[keep],
    }


def _checked_rows(
    i: Any, xl: Any, xr: Any, c: Any, Z: Any
) -> dict[str, npt.NDArray]:
    """Start-stop rows as given to ``fit_tvc``, checked and sorted by
    item and time."""
    xr = _times(xr, "xr")
    nrow = xr.shape[0]
    if nrow == 0:
        raise ValueError("`xr` is empty; there are no rows to fit.")
    xl = _times(xl, "xl", nrow)
    Z, i, c = _covariates(Z, nrow), _items(i, nrow), _flags(c, nrow)
    if (xr <= xl).any():
        raise ValueError(
            "{} row(s) have `xr <= xl`; each row is an interval (xl, xr] "
            "of positive length.".format(int((xr <= xl).sum()))
        )
    codes = _item_codes(i)
    order = np.lexsort((xr, xl, codes))
    i, xl, xr, c, Z = i[order], xl[order], xr[order], c[order], Z[order]
    codes = codes[order]
    same = codes[1:] == codes[:-1]
    overlap = same & (xl[1:] < xr[:-1])
    if overlap.any():
        raise ValueError(
            "{} row(s) overlap their item's previous row; an item's rows "
            "must not overlap (each starts at or after the end of the "
            "one before).".format(int(overlap.sum()))
        )
    return {"i": i, "xl": xl, "xr": xr, "c": c, "Z": Z}


def _event_numbers(codes: npt.NDArray, c: npt.NDArray) -> npt.NDArray:
    """For rows sorted by item and time: 1 + the number of the item's
    events before the row (the event the row is at risk of)."""
    events = (c == 0).astype(int)
    before = np.cumsum(events) - events
    first = np.r_[True, codes[1:] != codes[:-1]]
    start = np.maximum.accumulate(np.where(first, before, 0))
    return before - start + 1


def _check_max_events(max_events: Any) -> int | None:
    if max_events is None:
        return None
    if (
        isinstance(max_events, bool)
        or not isinstance(max_events, (int, np.integer))
        or max_events < 1
    ):
        raise ValueError(
            "`max_events` must be a whole number of at least 1 (the "
            f"number of events per item to model), or None; got "
            f"{max_events!r}."
        )
    return int(max_events)


def _marginal_layout(
    rows: dict[str, npt.NDArray], codes: npt.NDArray, max_events: int | None
) -> dict[str, Any]:
    """The Wei-Lin-Weissfeld layout: for each item and each event number
    ``k`` up to the largest, the time from 0 to its ``k``-th event, or
    its end of observation if it had fewer."""
    xl, xr, c, Z = rows["xl"], rows["xr"], rows["c"], rows["Z"]
    first = np.r_[True, codes[1:] != codes[:-1]]
    gap = ~first & (xl != np.r_[0.0, xr[:-1]])
    if (first & (xl != 0)).any() or gap.any():
        raise ValueError(
            "The Wei-Lin-Weissfeld model takes each item as observed "
            "without a break from time 0: {} item(s) start later or have "
            "a gap. Use AndersenGill or PrenticeWilliamsPeterson, whose "
            "rows carry their own risk intervals.".format(
                len(set(codes[(first & (xl != 0)) | gap].tolist()))
            )
        )
    varies = ~first & (Z != np.vstack([Z[:1], Z[:-1]])).any(axis=1)
    if varies.any():
        raise ValueError(
            "The Wei-Lin-Weissfeld model needs covariates that are "
            "constant within an item (each item is at risk of every "
            "event number from the start): {} item(s) have covariates "
            "that change. Use AndersenGill or PrenticeWilliamsPeterson "
            "for covariates that change over time.".format(
                len(set(codes[varies].tolist()))
            )
        )
    number = _event_numbers(codes, c)
    counts = np.bincount(codes, weights=(c == 0))
    largest = int(counts.max()) if max_events is None else max_events
    last = np.r_[codes[1:] != codes[:-1], True]
    end, z_item = xr[last], Z[last]
    n_items = end.shape[0]
    x = np.tile(end, (largest, 1)).T
    flag = np.ones((n_items, largest), dtype=int)
    event = (c == 0) & (number <= largest)
    x[codes[event], number[event] - 1] = xr[event]
    flag[codes[event], number[event] - 1] = 0
    return {
        "x": x.ravel(),
        "tl": None,
        "c": flag.ravel(),
        "Z": np.repeat(z_item, largest, axis=0),
        "strata": np.tile(np.arange(1, largest + 1), n_items),
        "cluster": np.repeat(np.arange(n_items), largest),
    }


def _layout(
    kind: str, rows: dict[str, npt.NDArray], max_events: int | None
) -> dict[str, Any]:
    """The rows as the Cox fit of ``kind`` takes them: ``x``, ``tl``,
    ``c``, ``Z``, ``strata`` (``None`` for one baseline) and each row's
    ``cluster`` (its item)."""
    codes = _item_codes(rows["i"])
    if kind == WLW:
        return _marginal_layout(rows, codes, max_events)
    xl, xr, c = rows["xl"], rows["xr"], rows["c"]
    out: dict[str, Any] = {
        "x": xr,
        "tl": xl,
        "c": c,
        "Z": rows["Z"],
        "strata": None,
        "cluster": codes,
    }
    if kind != ANDERSEN_GILL:
        number = _event_numbers(codes, c)
        out["strata"] = number
        if kind == PWP_GAP:
            # The clock restarts at each event: time since the item's
            # previous event, or since its first row began.
            first = np.r_[True, codes[1:] != codes[:-1]]
            restart = first | np.r_[False, c[:-1] == 0]
            origin = np.where(first, xl, np.r_[0.0, xr[:-1]])
            position = np.maximum.accumulate(
                np.where(restart, np.arange(xl.shape[0]), 0)
            )
            origin = origin[position]
            out["x"], out["tl"] = xr - origin, xl - origin
        if max_events is not None:
            keep = number <= max_events
            out = {k: (v if v is None else v[keep]) for k, v in out.items()}
    if not np.any(out["tl"]):
        out["tl"] = None
    return out


class RecurrentCoxModel(SerialisableMixin):
    """
    A fitted semi-parametric regression for repeated events: the result
    of :data:`AndersenGill`, :data:`PrenticeWilliamsPeterson` or
    :data:`WeiLinWeissfeld`.

    ``params`` are the log rate (hazard) ratios, one per covariate, named
    by ``parameter_names``. :meth:`covariance`, :meth:`standard_errors`
    and :meth:`summary` use the robust (sandwich) covariance with each
    item a cluster, which stays valid when an item's events are
    correlated; :meth:`naive_covariance` is the model-based one.
    :meth:`cif` is the cumulative baseline scaled by the covariates.

    ``cox`` is the Cox model fitted to the layout's rows, a
    :class:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel`:
    its residuals, ``diagnose()`` and concordance are those of that
    fit, row by row.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import AndersenGill
    >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
    >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
    >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    >>> model = AndersenGill.fit(x, Z, i=i, c=c)
    >>> model.kind
    'Andersen-Gill'
    >>> model.params.round(4)
    array([0.6931])
    >>> model.standard_errors().round(4)
    array([0.2632])
    >>> model.cif([6, 12], [1]).round(4)
    array([1.6667, 3.1111])
    """

    # What the fit made of the data it was given (from a SurpyvalData:
    # how the terminal flags a were read), shown by the repr; empty
    # for a fit from arrays.
    notes: tuple = ()

    #: Which model: ``"Andersen-Gill"``, ``"Prentice-Williams-Peterson,
    #: total time"``, ``"Prentice-Williams-Peterson, gap time"`` or
    #: ``"Wei-Lin-Weissfeld"``.
    kind: str
    #: The Cox model fitted to the layout's rows.
    cox: Any
    #: The event numbers with a baseline of their own (the strata), or
    #: ``None`` for the Andersen-Gill model's single baseline.
    strata: list | None
    #: The number of items, and of their events the fit used.
    n_items: int
    n_events: int
    #: ``max_events`` as given to the fit.
    max_events: int | None = None
    _robust: npt.NDArray

    # -- what the fit found -------------------------------------------------

    @property
    def params(self) -> npt.NDArray:
        """The coefficients: the log rate ratio of each covariate
        (``nan`` for an aliased one)."""
        return np.asarray(self.cox.params, dtype=float)

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``: the covariates' column names, else
        ``coef_0``, ``coef_1``, ..."""
        return list(self.cox.parameter_names)

    @property
    def feature_names(self) -> list[str] | None:
        """The covariates' column names, or ``None`` for an array."""
        return self.cox.feature_names

    @feature_names.setter
    def feature_names(self, names: list[str] | None) -> None:
        self.cox.feature_names = names

    @property
    def aliased(self) -> npt.NDArray:
        """The positions of the coefficients the data cannot determine."""
        return np.asarray(self.cox.aliased, dtype=int)

    @property
    def maximum(self) -> str:
        """What the fit reached (``"verified"``, ``"unverified"`` or
        ``"no finite maximum"``), as its warnings say."""
        return self.cox.maximum

    @property
    def tie_method(self) -> str:
        """How tied event times were handled."""
        return self.cox.tie_method

    @property
    def log_likelihood(self) -> float:
        """The partial log-likelihood at the fit."""
        return self.cox.log_likelihood

    def neg_ll(self) -> float:
        """
        The negative partial log-likelihood at the fit.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> round(model.neg_ll(), 4)
        3.3651
        """
        return self.cox.neg_ll()

    def aic(self) -> float:
        """
        Akaike's criterion from the partial likelihood, as R's
        ``AIC(coxph)``.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> round(model.aic(), 4)
        8.7301
        """
        return self.cox.aic()

    def aic_c(self) -> float:
        """
        The small-sample AIC from the partial likelihood, its sample size
        the number of events.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> round(model.aic_c(), 4)
        10.0635
        """
        return self.cox.aic_c()

    def bic(self) -> float:
        """
        The Bayesian criterion from the partial likelihood, its sample
        size the number of events.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> round(model.bic(), 4)
        np.float64(8.3396)
        """
        return self.cox.bic()

    # -- uncertainty --------------------------------------------------------

    def covariance(self) -> npt.NDArray:
        """
        The coefficients' robust covariance: the sandwich with each item
        a cluster (R's ``coxph(... + cluster(id))$var``). It stays valid
        when an item's events are correlated, which the model-based
        covariance (:meth:`naive_covariance`) does not. An aliased
        coefficient's row and column are ``nan``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> model.covariance().round(4)
        array([[0.0693]])
        """
        return self._robust.copy()

    def naive_covariance(self) -> npt.NDArray:
        """
        The coefficients' model-based covariance, the inverse of the
        partial likelihood's information (R's ``naive.var``). It takes
        every row as independent given the covariates, so it is right
        only when the model accounts for all of the dependence between
        an item's events.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> model.naive_covariance().round(4)
        array([[0.5]])
        """
        return np.array(self.cox.covariance(), dtype=float)

    def standard_errors(self) -> npt.NDArray:
        """
        The coefficients' robust standard errors, the square roots of
        the diagonal of :meth:`covariance`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> model.standard_errors().round(4)
        array([0.2632])
        """
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(self._robust))

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The coefficient table, with the columns of ``CoxPH``'s: the
        coefficient, the rate ratio ``exp(coef)``, the robust standard
        error ``se(coef)`` with the Wald interval, ``z`` and ``p`` that
        follow from it, and the model-based standard error beside it as
        ``naive se(coef)``.

        Parameters
        ----------
        alpha_ci : float, optional
            The total tail probability of the intervals. Default 0.05.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> model.summary()[["coef", "se(coef)", "naive se(coef)"]].round(4)
                     coef  se(coef)  naive se(coef)
        covariate
        coef_0     0.6931    0.2632          0.7071
        """
        check_alpha_ci(alpha_ci)
        beta = self.params
        table = coefficient_table(
            coefficient_names(self.cox, beta.size),
            beta,
            self.standard_errors(),
            alpha_ci,
        )
        with np.errstate(invalid="ignore"):
            naive = np.sqrt(np.diag(self.naive_covariance()))
        table.insert(3, "naive se(coef)", naive)
        return table

    # -- prediction ---------------------------------------------------------

    def _stratum(self, stratum: Any) -> Any:
        if self.strata is None:
            if stratum is not None:
                raise ValueError(
                    "`stratum` is for the models with a baseline per "
                    "event number; the Andersen-Gill model has one "
                    "baseline. Leave it out."
                )
            return None
        if stratum is None or stratum not in self.strata:
            raise ValueError(
                "`stratum` must be the event number whose baseline to "
                f"use, one of {self.strata}; got {stratum!r}."
            )
        return stratum

    def cif(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, stratum: Any = None
    ) -> npt.NDArray:
        """
        The cumulative baseline at ``x``, scaled by the covariates:
        :math:`\\Lambda_0(x)\\, e^{\\beta' Z}`.

        For the Andersen-Gill model this is the expected number of
        events by age ``x`` of an item with covariates ``Z``. For the
        models with a baseline per event number it is the cumulative
        hazard of event number ``stratum``: by age given the events
        before it (Prentice-Williams-Peterson, total time), by time
        since the previous event (gap time), or of the time from the
        start to that event (Wei-Lin-Weissfeld). It is 0 before the
        first event time and holds its last value after the last.

        Parameters
        ----------
        x : array-like
            The times.
        Z : array-like
            One covariate row (used for every time) or one per time.
        stratum : optional
            The event number, for a model with a baseline for each.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> model.cif([6, 12], [0]).round(4)
        array([0.8333, 1.5556])
        """
        return self.cox.Hf(x, Z, self._stratum(stratum))

    def sf(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, stratum: Any = None
    ) -> npt.NDArray:
        """
        The probability that event number ``stratum`` has not happened by
        ``x``, :math:`e^{-\\mathrm{cif}}`: by ``x`` from the start for
        the Wei-Lin-Weissfeld model, and by ``x`` after the previous
        event for the gap-time Prentice-Williams-Peterson model.

        Refused for the two models on the age scale after earlier
        events, where the exponential of the cumulative baseline is not
        the probability of anything observable: the Andersen-Gill model
        (use :meth:`cif`, the expected number of events) and the
        total-time Prentice-Williams-Peterson model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import WeiLinWeissfeld
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12, 6, 10, 12, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
        >>> Z = np.array(
        ...     [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
        ... )
        >>> model = WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=2)
        >>> model.sf([4, 8], [0], stratum=1).round(4)
        array([0.883 , 0.4155])
        """
        if self.kind not in (WLW, PWP_GAP):
            raise ValueError(
                f"sf is not defined for the {self.kind} model: on an "
                "item's age, after its earlier events, the exponential "
                "of the cumulative baseline is not a probability of "
                "anything observable. Use cif (for the Andersen-Gill "
                "model the expected number of events), or fit "
                "WeiLinWeissfeld for the time to each event number."
            )
        return np.exp(-self.cif(x, Z, stratum))

    def cif_cb(self, *args: Any, **kwargs: Any) -> npt.NDArray:
        """
        Confidence bounds on :meth:`cif` are not available: refused,
        saying what to use.

        The robust covariance here is of the coefficients; a bound on
        the curve needs the robust variance of the baseline as well,
        which is not built. ``model.cox.cb(x, Z, on="Hf")`` gives the
        model-based bound, valid when the model accounts for all of the
        dependence between an item's events; :meth:`summary` gives the
        robust intervals of the rate ratios.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> model.cif_cb([6], [0])
        Traceback (most recent call last):
        ...
        NotImplementedError: Robust confidence bounds on cif are not available: the sandwich covariance is of the coefficients, not of the baseline. Use summary() for the robust intervals of the rate ratios, or model.cox.cb(x, Z, on="Hf") for the model-based bound on the curve, which takes an item's events as independent given its covariates.
        """  # noqa: E501
        raise NotImplementedError(
            "Robust confidence bounds on cif are not available: the "
            "sandwich covariance is of the coefficients, not of the "
            "baseline. Use summary() for the robust intervals of the "
            'rate ratios, or model.cox.cb(x, Z, on="Hf") for the '
            "model-based bound on the curve, which takes an item's "
            "events as independent given its covariates."
        )

    # -- checks and pictures ------------------------------------------------

    def diagnose(self, *args: Any, **kwargs: Any) -> Any:
        """
        The diagnostics of the Cox fit to the layout's rows
        (``model.cox.diagnose``, which takes the arguments): the
        proportional-hazards tests, influence and the rest, row by row.
        Their p-values take the rows as independent, so with strongly
        correlated events read them as a screen, not a test.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> type(model.diagnose()).__name__
        'Diagnosis'
        """
        if not hasattr(self.cox, "_fit_data"):
            # A model restored by from_dict keeps the coefficients,
            # covariances and baseline, not the layout's rows; the Cox
            # model's own message would point at CoxPH.fit, which is not
            # how this model is refitted.
            raise ValueError(
                "The diagnostics need the data the model was fitted to "
                "(the layout's rows), which a model restored with "
                "from_dict / from_json does not carry. Refit on the data "
                "with {}.".format(
                    _FITTER_NAME.get(self.kind, "the recurrent fitter")
                )
            )
        return self.cox.diagnose(*args, **kwargs)

    def plot(self, ax: Any = None) -> Any:
        """
        Plot the cumulative baseline (:meth:`cif` at ``Z = 0``, or at the
        covariate means for a fit with ``center=True``) as a step
        function, one line per event number for a model with a baseline
        for each.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. Default: the current axes.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> ax = model.plot()
        """
        from surpyval import plotting

        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        for stratum in [None] if self.strata is None else self.strata:
            bx, _, bH0 = self.cox._baseline_arrays(stratum)
            ax.step(
                np.r_[0.0, bx],
                np.r_[0.0, bH0],
                where="post",
                label=(
                    "Baseline"
                    if stratum is None
                    else f"Event number {stratum}"
                ),
            )
        label_axes(ax, self.unit, _CURVE[self.kind])
        plotting.style_axes(ax)
        plotting.style_legend(ax)
        return ax

    # -- printing and saving ------------------------------------------------

    def __repr__(self) -> str:
        from surpyval.univariate.regression._summary import coefficient_repr

        lines = [
            "Recurrent-Event Regression SurPyval Model",
            "=========================================",
            f"Kind                : {self.kind}",
            "Parameterization    : Semi-Parametric",
            "Data                : {} items, {} events".format(
                self.n_items, self.n_events
            ),
            f"Tie method          : {self.tie_method}",
        ]
        if self.strata is not None:
            lines.append(
                "Baselines           : one per event number, "
                f"{self.strata[0]} to {self.strata[-1]}"
            )
        lines.append(
            "Coefficients        : exp(coef) is the rate ratio; robust "
            "standard errors (items as clusters); Wald 95% intervals"
        )
        table = coefficient_table(
            coefficient_names(self.cox, self.params.size),
            self.params,
            self.standard_errors(),
        )
        lines.append(coefficient_repr(table))
        return "\n".join(lines) + notes_text(self)

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, strict-JSON dictionary: the Cox
        fit of the layout (``"cox"``, its own dictionary), the robust
        covariance and what kind of model it is. The restored model gives
        the same coefficients, covariances, ``summary()`` and
        predictions; as for a restored Cox model, the diagnostics need
        the data and a refit.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> from surpyval.recurrent.regression import RecurrentCoxModel
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> restored = RecurrentCoxModel.from_dict(model.to_dict())
        >>> bool(restored.cif(6, [1]) == model.cif(6, [1]))
        True
        """
        out = {
            "model": "RecurrentCoxModel",
            "kind": self.kind,
            "parameterization": "Semi-Parametric",
            "cox": self.cox.to_dict(),
            "robust_covariance": np.asarray(
                self._robust, dtype=float
            ).tolist(),
            "n_items": int(self.n_items),
            "n_events": int(self.n_events),
            "max_events": self.max_events,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> RecurrentCoxModel:
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> from surpyval.recurrent.regression import RecurrentCoxModel
        >>> model = AndersenGill.fit(
        ...     [5, 9, 12, 3, 6, 8, 12], [0, 0, 0, 1, 1, 1, 1],
        ...     i=[1, 1, 1, 2, 2, 2, 2], c=[0, 0, 1, 0, 0, 0, 1],
        ... )
        >>> RecurrentCoxModel.from_dict(model.to_dict()).kind
        'Andersen-Gill'
        """
        from surpyval.univariate.regression import (
            semi_parametric_regression_model as spr,
        )

        require_model_tag(
            model_dict,
            "RecurrentCoxModel",
            "a semi-parametric recurrent-event regression",
        )
        kind = model_dict["kind"]
        if kind not in _CURVE:
            raise ValueError(
                "The serialised 'kind' must be one of {}; got {!r}.".format(
                    sorted(_CURVE), kind
                )
            )
        out = cls()
        out.kind = kind
        out.cox = spr.SemiParametricRegressionModel.from_dict(
            model_dict["cox"]
        )
        out._robust = dict_numbers(model_dict, "robust_covariance", True)
        out.n_items = int(model_dict["n_items"])
        out.n_events = int(model_dict["n_events"])
        out.max_events = model_dict.get("max_events")
        out.strata = _strata_of(out.cox)
        return out


def _strata_of(cox: Any) -> list | None:
    labels = getattr(cox, "strata_labels", None)
    if labels is None:
        return None
    return sorted(int(label) for label in labels)


def _fit_layout(
    kind: str,
    rows: dict[str, npt.NDArray],
    max_events: int | None,
    tie_method: str,
    tol: float,
    center: bool,
    names: list[str] | None,
) -> RecurrentCoxModel:
    from surpyval.univariate.regression.proportional_hazards import CoxPH

    layout = _layout(kind, rows, _check_max_events(max_events))
    if not (layout["c"] == 0).any():
        raise ValueError(
            "The data have no events (every row has c = 1), so there is "
            "nothing to fit."
        )
    cox = CoxPH.fit(
        layout["x"],
        layout["Z"],
        c=layout["c"],
        tl=layout["tl"],
        strata=layout["strata"],
        tie_method=tie_method,
        tol=tol,
        center=center,
    )
    if names is not None:
        cox.feature_names = [str(name) for name in names]
    model = RecurrentCoxModel()
    model.kind = kind
    model.cox = cox
    model.max_events = max_events
    model.strata = _strata_of(cox)
    model.n_items = int(np.unique(layout["cluster"]).shape[0])
    model.n_events = int((layout["c"] == 0).sum())
    size = np.asarray(cox.params).shape[0]
    try:
        robust = np.array(
            cox.robust_covariance(layout["cluster"]), dtype=float
        )
    except (ValueError, np.linalg.LinAlgError):
        # No finite maximum (the fit has warned): no covariance either
        robust = np.full((size, size), np.nan)
    # A coefficient with no finite estimate (the fit has warned) has no
    # standard error, model-based or robust.
    lost = np.isnan(np.asarray(cox.standard_errors(), dtype=float))
    robust[lost, :] = np.nan
    robust[:, lost] = np.nan
    model._robust = robust
    return model


def _frame_names(Z: Any) -> list[str] | None:
    if isinstance(Z, pd.DataFrame):
        return [str(name) for name in Z.columns]
    return None


class _RecurrentCoxFitter(FitterRepr):
    """What the three fitters share: the two array entry points and the
    two DataFrame ones, each a layout of the rows handed to ``CoxPH``."""

    fitter_kind = "semi-parametric recurrent-event regression fitter"
    #: The model fitted, or for a fitter with several the one each
    #: option selects.
    _kind: str
    fit: Any

    def _resolve_kind(self, options: dict) -> str:
        return self._kind

    def _fit_rows(
        self,
        rows: dict[str, npt.NDArray],
        names: list[str] | None,
        max_events: int | None,
        tie_method: str,
        tol: float,
        center: bool,
        options: dict,
    ) -> RecurrentCoxModel:
        kind = self._resolve_kind(options)
        return _fit_layout(
            kind, rows, max_events, tie_method, tol, center, names
        )

    def _frame(
        self,
        fit: Any,
        df: pd.DataFrame,
        columns: dict[str, str | None],
        Z_cols: str | list[str],
        fit_options: dict[str, Any],
    ) -> RecurrentCoxModel:
        df = require_frame(df)
        for name in (*columns, "Z"):
            if name in fit_options:
                suffix = "_cols" if name == "Z" else "_col"
                raise TypeError(
                    f"{fit.__name__}_from_df() got an unexpected keyword "
                    f"argument '{name}'; name the column with "
                    f"'{name}{suffix}'"
                )
        arrays: dict[str, Any] = {}
        for arg, column in columns.items():
            if column is None:
                continue
            arrays[arg] = frame_column(
                df, column, f"{arg}_col", time=arg in ("x", "xl", "xr")
            )
        names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        from surpyval.utils.covariates import numeric_columns

        check_columns(df, Z_cols=names)
        arrays["Z"] = numeric_columns(df, names)
        with covariate_columns(names):
            model = fit(**arrays, **fit_options)
        model.cox.feature_names = [str(name) for name in names]
        return model

    @ambiguous_item_ids(
        "a regression's fit_from_df(df, x_col, Z_cols, c_col)", column=True
    )
    @takes_unit
    @takes_terminal_column
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        Z_cols: str | list[str],
        i_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        **fit_options: Any,
    ) -> RecurrentCoxModel:
        """
        :meth:`fit` from the columns of a DataFrame: the column names in
        place of the arrays, every other option passed on unchanged.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import AndersenGill
        >>> log = pd.DataFrame({
        ...     "hours": [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12],
        ...     "truck": [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4],
        ...     "c": [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        ...     "load": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        ... })
        >>> model = AndersenGill.fit_from_df(
        ...     log, x_col="hours", Z_cols="load", i_col="truck", c_col="c"
        ... )
        >>> model.parameter_names
        ['load']
        """
        columns: dict[str, str | None] = {
            "x": x_col,
            "i": i_col,
            "c": c_col,
            "n": n_col,
        }
        return self._frame(self.fit, df, columns, Z_cols, fit_options)


class _StartStopFrame(_RecurrentCoxFitter):
    """``fit_tvc_from_df`` for the fitters with a start-stop form."""

    fit_tvc: Any

    @takes_unit
    def fit_tvc_from_df(
        self,
        df: pd.DataFrame,
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str],
        **fit_options: Any,
    ) -> RecurrentCoxModel:
        """
        :meth:`fit_tvc` from the columns of a DataFrame: the column
        names in place of the arrays, every other option passed on
        unchanged.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import AndersenGill
        >>> rows = pd.DataFrame({
        ...     "id": [1, 1, 1, 2, 2, 2, 2, 3, 3],
        ...     "start": [0, 5, 9, 0, 3, 6, 8, 0, 7],
        ...     "stop": [5, 9, 12, 3, 6, 8, 12, 7, 12],
        ...     "c": [0, 0, 1, 0, 0, 0, 1, 0, 1],
        ...     "rx": [0, 0, 0, 1, 1, 1, 1, 0, 0],
        ... })
        >>> model = AndersenGill.fit_tvc_from_df(
        ...     rows, i_col="id", xl_col="start", xr_col="stop",
        ...     c_col="c", Z_cols="rx",
        ... )
        >>> model.params.round(4)
        array([0.6931])
        """
        columns: dict[str, str | None] = {
            "i": i_col,
            "xl": xl_col,
            "xr": xr_col,
            "c": c_col,
        }
        return self._frame(self.fit_tvc, df, columns, Z_cols, fit_options)


@singleton_fitter
class AndersenGill(_StartStopFrame):
    """
    The Andersen-Gill model: a proportional-intensity model for repeated
    events with the baseline left unspecified,

    .. math::
        \\lambda(t \\mid Z) = \\lambda_0(t)\\, e^{\\beta' Z},

    with ``t`` the item's age. Every event shares the one baseline, and
    an item is at risk again from the moment of each event, so the model
    says that the covariates carry everything about an item that matters:
    its earlier events do not change its rate (give a count of them as a
    covariate that changes over time, with ``fit_tvc``, if they may).
    It is the semi-parametric relative of
    :data:`~surpyval.recurrent.ProportionalIntensityNHPP`.

    The reported standard errors are robust, each item a cluster, so
    they hold when an item's events are more alike than the covariates
    explain; read that way the model is one of the *mean* number of
    events (Lin, Wei, Yang and Ying 2000). The fit is R's
    ``coxph(Surv(start, stop, event) ~ ... + cluster(id))``.

    ``AndersenGill`` is an instance of this class; its fits return a
    :class:`~surpyval.recurrent.regression.recurrent_cox.RecurrentCoxModel`.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import AndersenGill
    >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
    >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
    >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    >>> model = AndersenGill.fit(x, Z, i=i, c=c)
    >>> model.params.round(4)
    array([0.6931])
    """

    _kind = ANDERSEN_GILL

    @ambiguous_item_ids("a regression's fit(x, Z, c, n)")
    @dataframe_covariates
    @takes_unit
    @accepts_recurrent_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        i: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> RecurrentCoxModel:
        """
        Fit the Andersen-Gill model to an event log.

        Parameters
        ----------
        x : array-like
            Each item's event times and, flagged ``c = 1``, the end of
            its observation. An item is observed from 0 (for a later
            entry, or a break in observation, use :meth:`fit_tvc`); one
            whose last row is an event is observed up to that event.
            Or a ``SurpyvalData`` with each row's unit ``i``, which
            holds the data: ``fit(data)``, with the other data
            arguments left out. Its rows are read as each unit's
            history (gaps as ``windows``, covariates that change
            within a unit by ``fit_tvc`` where the model has one),
            and the model's ``notes`` say what was made of the
            terminal flags ``a``.
        Z : array-like
            The covariates, one row per row of ``x``: those in force
            over the interval that ends at the row's time. They may
            change from one row of an item to the next. A missing value
            is refused: a row is part of its item's history.
        i : array-like, optional
            The item each row belongs to. Default: one item.
        c : array-like, optional
            0 for an event, 1 for the end of an item's observation.
            Default: every row an event.
        n : array-like, optional
            Must be 1 on every row; a row is one event of one item.
        tie_method : str, optional
            How tied event times are handled, as ``CoxPH.fit``:
            ``"efron"`` (default), ``"breslow"``, ``"exact"`` or
            ``"kalbfleisch-prentice"``.
        tol : float, optional
            The convergence tolerance of the Newton-Raphson search, as
            ``CoxPH.fit``.
        center : bool, optional
            ``False`` (default) reports the baseline at ``Z = 0``;
            ``True`` at the covariate means, as ``CoxPH.fit``.

        Returns
        -------
        RecurrentCoxModel

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import AndersenGill
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        >>> model = AndersenGill.fit(x, Z, i=i, c=c)
        >>> model.params.round(4)
        array([0.6931])
        """
        rows = rows_from_event_times(x, Z, i, c, n)
        return self._fit_rows(
            rows, _frame_names(Z), None, tie_method, tol, center, {}
        )

    @dataframe_covariates
    @takes_unit
    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> RecurrentCoxModel:
        """
        Fit the Andersen-Gill model to start-stop rows: each row an
        interval ``(xl, xr]`` of item ``i``'s history with the
        covariates ``Z`` in force over it, ending in an event
        (``c = 0``) or not (``c = 1``). An item's rows must not overlap;
        it is not at risk before its first row (a delayed entry) or in a
        gap between rows.

        Parameters
        ----------
        i, xl, xr, c, Z : array-like
            One value (one row of ``Z``) per interval.
        tie_method : str, optional
            How tied event times are handled, as ``CoxPH.fit``:
            ``"efron"`` (default), ``"breslow"``, ``"exact"`` or
            ``"kalbfleisch-prentice"``.
        tol : float, optional
            The convergence tolerance of the Newton-Raphson search, as
            ``CoxPH.fit``.
        center : bool, optional
            ``False`` (default) reports the baseline at ``Z = 0``;
            ``True`` at the covariate means, as ``CoxPH.fit``.

        Returns
        -------
        RecurrentCoxModel

        Examples
        --------
        >>> from surpyval.recurrent import AndersenGill
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3]
        >>> xl = [0, 5, 9, 0, 3, 6, 8, 0, 7]
        >>> xr = [5, 9, 12, 3, 6, 8, 12, 7, 12]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1]
        >>> Z = [0, 0, 0, 1, 1, 1, 1, 0, 0]
        >>> AndersenGill.fit_tvc(i, xl, xr, c, Z).params.round(4)
        array([0.6931])
        """
        rows = _checked_rows(i, xl, xr, c, Z)
        return self._fit_rows(
            rows, _frame_names(Z), None, tie_method, tol, center, {}
        )


_TIME_SCALES = {"total": PWP_TOTAL, "gap": PWP_GAP}


@singleton_fitter
class PrenticeWilliamsPeterson(_StartStopFrame):
    """
    The Prentice-Williams-Peterson conditional model: an item is at risk
    of its ``k``-th event only once it has had ``k - 1``, and each event
    number has a baseline of its own (a stratum), with shared
    coefficients:

    .. math::
        \\lambda_k(t \\mid Z) = \\lambda_{0k}(t)\\, e^{\\beta' Z}.

    Use it when an event changes the item, so that the rate of a second
    event is not that of a first: a repair that leaves a machine worse
    (or better) than before, a relapse that makes the next more likely.
    ``time_scale="total"`` measures ``t`` as the item's age (R's
    ``coxph(Surv(start, stop, event) ~ ... + strata(enum) +
    cluster(id))``); ``time_scale="gap"`` restarts the clock at each
    event, so ``t`` is the time since the last one (``Surv(stop - start,
    event)``).

    ``PrenticeWilliamsPeterson`` is an instance of this class; its fits
    return a
    :class:`~surpyval.recurrent.regression.recurrent_cox.RecurrentCoxModel`,
    whose ``cif`` takes the event number as ``stratum``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import PrenticeWilliamsPeterson
    >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12, 6, 10, 12, 12]
    >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
    >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
    >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    >>> model = PrenticeWilliamsPeterson.fit(x, Z, i=i, c=c, max_events=2)
    >>> model.strata
    [1, 2]
    >>> model.params.round(4)
    array([1.363])
    """

    _kind = PWP_TOTAL

    def _resolve_kind(self, options: dict) -> str:
        time_scale = options["time_scale"]
        if time_scale not in _TIME_SCALES:
            raise ValueError(
                "`time_scale` must be 'total' (the item's age) or 'gap' "
                f"(the time since its previous event); got {time_scale!r}."
            )
        return _TIME_SCALES[time_scale]

    @ambiguous_item_ids("a regression's fit(x, Z, c, n)")
    @dataframe_covariates
    @takes_unit
    @accepts_recurrent_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        i: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        time_scale: str = "total",
        max_events: int | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> RecurrentCoxModel:
        """
        Fit the Prentice-Williams-Peterson model to an event log.

        Parameters
        ----------
        x, Z, i, c, n : array-like
            The event log, as :data:`AndersenGill`'s ``fit`` takes it:
            event and end-of-observation times, the covariates in force
            up to each row, the item of each row, and the flags (0 an
            event, 1 the end of an item's observation).
            Or a ``SurpyvalData`` with each row's unit ``i``, which
            holds the data: ``fit(data)``, with the other data
            arguments left out. Its rows are read as each unit's
            history (gaps as ``windows``, covariates that change
            within a unit by ``fit_tvc`` where the model has one),
            and the model's ``notes`` say what was made of the
            terminal flags ``a``.
        time_scale : str, optional
            ``"total"`` (default): the clock is the item's age.
            ``"gap"``: it restarts at each event.
        max_events : int, optional
            Model each item's first ``max_events`` events only: what
            follows an item's ``max_events``-th event is left out. The
            later event numbers are reached by few items, and a baseline
            estimated from a handful of them is poorly determined.
            Default ``None``: every event.
        tie_method : str, optional
            How tied event times are handled, as ``CoxPH.fit``:
            ``"efron"`` (default), ``"breslow"``, ``"exact"`` or
            ``"kalbfleisch-prentice"``.
        tol : float, optional
            The convergence tolerance of the Newton-Raphson search, as
            ``CoxPH.fit``.
        center : bool, optional
            ``False`` (default) reports the baseline at ``Z = 0``;
            ``True`` at the covariate means, as ``CoxPH.fit``.

        Returns
        -------
        RecurrentCoxModel

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import PrenticeWilliamsPeterson
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12, 6, 10, 12, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
        >>> Z = np.array(
        ...     [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
        ... )
        >>> model = PrenticeWilliamsPeterson.fit(
        ...     x, Z, i=i, c=c, time_scale="gap", max_events=2
        ... )
        >>> model.kind
        'Prentice-Williams-Peterson, gap time'
        >>> model.params.round(4)
        array([1.6117])
        """
        rows = rows_from_event_times(x, Z, i, c, n)
        return self._fit_rows(
            rows,
            _frame_names(Z),
            max_events,
            tie_method,
            tol,
            center,
            {"time_scale": time_scale},
        )

    @dataframe_covariates
    @takes_unit
    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        time_scale: str = "total",
        max_events: int | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> RecurrentCoxModel:
        """
        Fit the Prentice-Williams-Peterson model to start-stop rows, as
        :data:`AndersenGill`'s ``fit_tvc`` takes them: each row an
        interval ``(xl, xr]`` of item ``i``'s history with the
        covariates ``Z`` in force over it, ending in an event
        (``c = 0``) or not (``c = 1``). The event number a row is at
        risk of is one more than the item's events in its earlier rows;
        with ``time_scale="gap"`` a row's interval is measured from the
        item's previous event (from the start of its first row before
        any).

        Parameters
        ----------
        i, xl, xr, c, Z : array-like
            One value (one row of ``Z``) per interval.
        time_scale : str, optional
            ``"total"`` (default) or ``"gap"``.
        max_events : int, optional
            Model each item's first ``max_events`` events only: what
            follows an item's ``max_events``-th event is left out. The
            later event numbers are reached by few items, and a baseline
            estimated from a handful of them is poorly determined.
            Default ``None``: every event.
        tie_method : str, optional
            How tied event times are handled, as ``CoxPH.fit``:
            ``"efron"`` (default), ``"breslow"``, ``"exact"`` or
            ``"kalbfleisch-prentice"``.
        tol : float, optional
            The convergence tolerance of the Newton-Raphson search, as
            ``CoxPH.fit``.
        center : bool, optional
            ``False`` (default) reports the baseline at ``Z = 0``;
            ``True`` at the covariate means, as ``CoxPH.fit``.

        Returns
        -------
        RecurrentCoxModel

        Examples
        --------
        >>> from surpyval.recurrent import PrenticeWilliamsPeterson
        >>> i = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6]
        >>> xl = [0, 5, 9, 0, 3, 6, 0, 7, 0, 2, 4, 0, 6, 10, 0]
        >>> xr = [5, 9, 12, 3, 6, 12, 7, 12, 2, 4, 12, 6, 10, 12, 12]
        >>> c = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        >>> Z = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
        >>> model = PrenticeWilliamsPeterson.fit_tvc(i, xl, xr, c, Z)
        >>> model.params.round(4)
        array([1.363])
        """
        rows = _checked_rows(i, xl, xr, c, Z)
        return self._fit_rows(
            rows,
            _frame_names(Z),
            max_events,
            tie_method,
            tol,
            center,
            {"time_scale": time_scale},
        )


@singleton_fitter
class WeiLinWeissfeld(_RecurrentCoxFitter):
    """
    The Wei-Lin-Weissfeld marginal model: the time from the start to an
    item's first event, to its second, to its third, ... are each given
    a Cox model, with a baseline per event number and shared
    coefficients,

    .. math::
        \\lambda_k(t \\mid Z) = \\lambda_{0k}(t)\\, e^{\\beta' Z},

    and every item is at risk of every event number from time 0 (an
    item is "at risk" of its third event before it has had its first).
    It makes no statement about how one event leads to the next, and
    answers "how do the covariates change the time until the ``k``-th
    event, counted from the start?": the question of a trial that
    randomises at time 0. The fit is R's ``coxph(Surv(stop, event) ~
    ... + strata(enum) + cluster(id))`` on one row per item and event
    number.

    ``WeiLinWeissfeld`` is an instance of this class; its fits return a
    :class:`~surpyval.recurrent.regression.recurrent_cox.RecurrentCoxModel`,
    whose ``cif`` and ``sf`` take the event number as ``stratum``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import WeiLinWeissfeld
    >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12, 6, 10, 12, 12]
    >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
    >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
    >>> Z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    >>> model = WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=2)
    >>> model.params.round(4)
    array([1.6878])
    """

    _kind = WLW

    @ambiguous_item_ids("a regression's fit(x, Z, c, n)")
    @dataframe_covariates
    @takes_unit
    @accepts_recurrent_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        i: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        max_events: int | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> RecurrentCoxModel:
        """
        Fit the Wei-Lin-Weissfeld model to an event log.

        Parameters
        ----------
        x, Z, i, c, n : array-like
            The event log, as :data:`AndersenGill`'s ``fit`` takes it,
            with covariates that are constant within an item (refused
            otherwise: every event number's clock starts at 0).
            Or a ``SurpyvalData`` with each row's unit ``i``, which
            holds the data: ``fit(data)``, with the other data
            arguments left out. Its rows are read as each unit's
            history (gaps as ``windows``, covariates that change
            within a unit by ``fit_tvc`` where the model has one),
            and the model's ``notes`` say what was made of the
            terminal flags ``a``.
        max_events : int, optional
            The number of event numbers modelled, each with its own
            baseline: an item with fewer events is censored, at the end
            of its observation, for the rest, and events after an
            item's ``max_events``-th are left out. Default ``None``:
            the largest number of events any item had.
        tie_method : str, optional
            How tied event times are handled, as ``CoxPH.fit``:
            ``"efron"`` (default), ``"breslow"``, ``"exact"`` or
            ``"kalbfleisch-prentice"``.
        tol : float, optional
            The convergence tolerance of the Newton-Raphson search, as
            ``CoxPH.fit``.
        center : bool, optional
            ``False`` (default) reports the baseline at ``Z = 0``;
            ``True`` at the covariate means, as ``CoxPH.fit``.

        Returns
        -------
        RecurrentCoxModel

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import WeiLinWeissfeld
        >>> x = [5, 9, 12, 3, 6, 8, 12, 7, 12, 2, 4, 9, 12, 6, 10, 12, 12]
        >>> i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
        >>> c = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
        >>> Z = np.array(
        ...     [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
        ... )
        >>> model = WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=2)
        >>> model.strata
        [1, 2]
        >>> model.params.round(4)
        array([1.6878])
        """
        rows = rows_from_event_times(x, Z, i, c, n)
        return self._fit_rows(
            rows, _frame_names(Z), max_events, tie_method, tol, center, {}
        )
