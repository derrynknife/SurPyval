"""Time-varying-covariate fitting for the parametric regression families.

For a proportional-hazards, additive-hazards or proportional-odds model the
hazard at time ``t`` depends only on ``t`` and the covariate value *at*
``t``, so the cumulative hazard is *additive over disjoint time intervals*:
each constant-covariate segment ``(xl, xr]`` contributes the increment
``H(xr | Z_seg) - H(xl | Z_seg)`` of the constant-covariate model,

- PH: ``[H0(xr) - H0(xl)] * exp(Z_seg' beta)``,
- AH: ``[H0(xr) - H0(xl)] + (Z_seg' beta) * (xr - xl)``,
- PO: ``log[S(xl | Z_seg) / S(xr | Z_seg)]``, as the PO hazard
  ``h0 / (F0 + phi S0)`` with ``phi = exp(Z_seg' beta)`` integrates to
  ``H(t | z) = -log S(t | z)``.

A subject observed with a time-varying covariate therefore factorises
exactly into one *left-truncated* (delayed-entry) observation per
constant-covariate interval -- entering at ``xl`` and exiting at ``xr``. This
is the same episode-splitting identity the Cox partial likelihood uses; here
it lets the ordinary parametric MLE ``fit`` (which already accepts truncation
``t``) fit start-stop data with no new likelihood. The mixin therefore just
reshapes the time-varying-covariate data and calls ``fit``.

It is mixed into the fitters whose cumulative hazard is additive over
intervals (``ProportionalHazardsFitter``, ``AdditiveHazardsFitter``,
``ProportionalOddsFitter``). It is *not* correct for accelerated failure time,
which must accumulate an "accelerated age" across intervals; that family has
its own likelihood (``aft_tvc_fit``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.univariate.regression._penalty_argument import (
    penalised_entry,
    penalty_argument,
)

from ._aliasing import covariate_columns
from .regression_data import design_matrix_from_df

if TYPE_CHECKING:
    import pandas as pd

    from .parametric_regression_model import ParametricRegressionModel


def fit_tvc_df(
    fit: Any,
    df: pd.DataFrame,
    columns: dict[str, str],
    Z_cols: str | list[str] | None,
    formula: str | None,
    n_col: str | None,
    **kwargs: Any,
) -> Any:
    """The body of every time-varying ``*_from_df`` fit (PH, AH, PO, AFT
    and Cox): call the array fit ``fit`` with the columns of ``df`` that
    ``columns`` names (``{argument: column}``), the covariates ``Z`` from
    ``Z_cols`` or ``formula`` (:func:`design_matrix_from_df`, every row
    kept: a TVC fit refuses a missing covariate rather than drop part of a
    subject's path), the counts ``n`` from ``n_col``, and ``kwargs``; the
    model records ``feature_names``, the ``formula`` (as given) and its
    encoding, so it predicts from a DataFrame with the same design."""
    Z, names, spec = design_matrix_from_df(df, Z_cols, formula)
    arrays = {arg: df[col].to_numpy() for arg, col in columns.items()}
    n = None if n_col is None else df[n_col].to_numpy()
    with covariate_columns(names, Z, spec):
        model = fit(**arrays, Z=Z, n=n, **kwargs)
    model.feature_names = names
    model.formula = formula
    model._model_spec = spec
    return model


def start_stop_rows(
    ident: npt.NDArray,
    xl: npt.NDArray,
    xr: npt.NDArray,
    c: npt.NDArray,
    Z: npt.ArrayLike,
    n: npt.NDArray,
) -> dict[str, Any]:
    """The start-stop rows of a time-varying fit as ``handle_tvc`` gives
    them: each row's subject ``i``, entry ``xl``, exit ``xr``, ``c``,
    ``Z`` and ``n``. A fitted model keeps them (:func:`mark_tvc_fit`):
    ``diagnose()`` sums its residuals over them and refits the other
    models to them."""
    return {"i": ident, "xl": xl, "xr": xr, "c": c, "Z": Z, "n": n}


def mark_tvc_fit(model: Any, rows: dict[str, Any]) -> None:
    """Mark a parametric ``model`` fitted to the start-stop ``rows``
    (:func:`start_stop_rows`): ``is_tvc``, each row's subject, and the
    rows as given."""
    model.is_tvc = True
    model.tvc_subject_ids = rows["i"]
    model._tvc_fit_rows = rows


def _count_subjects(
    model: Any, first: npt.NDArray, counts: npt.NDArray, n: npt.NDArray
) -> None:
    """The sample sizes of a time-varying model's ``bic`` and ``aic_c``:
    they count observed failures, and fall back (with none) to subjects,
    each weighted by its last interval's count (as the AFT time-varying
    fit does), not interval rows: splitting a subject's follow-up into
    more intervals leaves the likelihood unchanged and must leave the
    criteria unchanged too. ``first`` and ``counts`` are each subject's
    first row and number of rows."""
    model.n_subjects = int(first.shape[0])
    model._ic_n_total = float(n[first + counts - 1].sum())


def _entered_at(fitter: Any, tl: npt.NDArray) -> npt.NDArray:
    """The entry of each start-stop row for the fit: its ``xl``, as
    given (#952). A subject's first row at 0 entered at 0, as at any
    other time, and one at -inf is not truncated. For a baseline that
    starts at 0 the two are the same (survival to 0 is certain), so an
    entry at or before the start of its support is no truncation there.
    For one defined below 0 (Normal, Gumbel, Logistic) an entry at 0
    conditions on surviving to 0; followed since -inf, the subject's
    first covariate holds since then."""
    support_lo = float(fitter.support[0])
    if support_lo > -np.inf:
        return np.where(tl <= support_lo, -np.inf, tl)
    return tl


def _check_additive_from_zero(
    ident: npt.NDArray,
    first: npt.NDArray,
    counts: npt.NDArray,
    tl: npt.NDArray,
    x: npt.NDArray,
) -> None:
    """Refuse, for additive hazards on a baseline defined below 0, a
    subject followed since -inf whose covariate changes before 0: the
    model measures the covariate's effect from 0, which its rows, each
    measured from -inf, cannot give (#952)."""
    early = (counts > 1) & np.isneginf(tl[first]) & (x[first] < 0)
    if early.any():
        u = ident[first[np.flatnonzero(early)[0]]]
        raise ValueError(
            f"Subject {u!r} is followed since -inf and its covariate "
            "changes before 0: an additive hazards model on a baseline "
            "defined below 0 measures the covariate's effect from 0, so a "
            "subject followed since -inf needs one covariate value across "
            "0 (a first row ending at or after 0). Give its entry as a "
            "finite xl to fit the changes before 0."
        )


def subject_units(
    x: npt.NDArray,
    tl: npt.NDArray,
    Z: npt.NDArray,
    n: npt.NDArray,
    ident: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """One covariate row and one count per subject of start-stop rows
    ``(tl, x]``, for a penalised fit's ``N`` and standardisation, as
    ``CoxPH.fit_tvc_penalised`` takes them: each subject's covariates
    averaged over its follow-up, each row weighted by its length (so
    splitting a row into several with the same covariates leaves it
    unchanged), and the count ``n`` of its last row. A subject followed
    since ``-inf`` (a first row of infinite length) has that row's
    covariates."""
    ident = np.asarray(ident)
    Z = np.asarray(Z, dtype=float)
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        length = x - np.asarray(tl, dtype=float)
    labels, inverse = np.unique(ident, return_inverse=True)
    k = labels.size
    endless = ~np.isfinite(length)
    if endless.any():
        # Only such a row counts for its subject.
        has = np.bincount(inverse, weights=endless, minlength=k) > 0
        length = np.where(endless, 1.0, np.where(has[inverse], 0.0, length))
    total = np.bincount(inverse, weights=length, minlength=k)
    Z_units = (
        np.column_stack(
            [
                np.bincount(inverse, weights=length * Z[:, j], minlength=k)
                for j in range(Z.shape[1])
            ]
        ).reshape(k, Z.shape[1])
        / total[:, None]
    )
    last = np.zeros(k, dtype=int)
    order = np.lexsort((x, inverse))
    last[inverse[order]] = order
    return Z_units, np.asarray(n, dtype=float)[last]


class TVCFitMixin:
    """Adds ``fit_tvc`` (start-stop and timeline, array and DataFrame) to a
    parametric regression fitter whose cumulative hazard is additive over time
    intervals (PH, AH, PO). Requires the host class to provide a
    ``fit(x, Z, c, n, t, ...)`` method that accepts truncation ``t`` as a
    ``[tl, tr]`` matrix."""

    @penalty_argument("_fit_tvc_penalised")
    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> ParametricRegressionModel:
        """
        Fit the model to time-varying covariates in start-stop format.

        Each row is one observation interval ``(xl, xr]`` of subject ``i`` on
        which the covariate row ``Z`` is constant; ``c`` is ``0`` (event at
        ``xr``) only on the interval that ends at the subject's event and ``1``
        (right-censored) otherwise -- surpyval's censoring convention. The
        intervals are validated and mapped to left-truncated observations
        (``t = [xl, inf]``), then fitted with the ordinary parametric MLE,
        so the
        fit is identical to the equivalent non-time-varying data and its
        log-likelihood is that of ``sf_tvc`` / ``hf`` along each subject's
        covariate path. Extra keyword arguments (``init``, ``fixed``,
        ``center``) are passed through to ``fit``.

        A subject's rows must not overlap, it may have at most one event,
        and that event must be on its last interval; gaps between its
        intervals (not at risk) and delayed entry (a first ``xl > 0``) are
        allowed.

        A subject's first row enters at its ``xl``, as any other row does
        (#952). For a baseline that starts at 0, entering at 0 is no
        truncation. For one defined below 0 (``Normal``, ``Gumbel``,
        ``Logistic``) a first ``xl`` of 0 means the subject entered at 0,
        and the fit conditions on its surviving to 0; give ``-inf`` for a
        subject followed since ``-inf``, its first covariate row holding
        since then.

        Parameters
        ----------
        i : array_like
            Subject identifier of each interval row.
        xl, xr : array_like
            The open-closed interval ``(xl, xr]`` of each row.
        c : array_like
            Status at ``xr``: ``0`` for the subject's event, ``1`` for a
            right-censored interval end (a covariate change or the end of
            follow-up).
        Z : array_like
            The covariate row in force on each interval (a 1-D array is a
            single covariate).
        n : array_like, optional
            Count weight of each row. Defaults to 1.
        **kwargs
            Passed to ``fit`` (``init``, ``fixed``, ``center``).

        Returns
        -------
        ParametricRegressionModel
            The fitted model, with ``is_tvc`` set. Evaluate it along a
            covariate path with ``sf_tvc`` / ``Hf_tvc``.

        Examples
        --------
        Units fail at rate 0.5 until a stress switches on at a random time,
        and at ``0.5 e`` after it; a unit that outlives its switch
        contributes two rows:

        >>> import numpy as np
        >>> from surpyval import WeibullPH
        >>> rng = np.random.default_rng(0)
        >>> n = 300
        >>> switch = rng.uniform(0.3, 1.5, n)
        >>> t_low = rng.exponential(2.0, n)
        >>> t_high = switch + rng.exponential(2.0 / np.e, n)
        >>> T = np.where(t_low > switch, t_high, t_low)
        >>> one = T <= switch
        >>> i = np.r_[np.arange(n), np.flatnonzero(~one)]
        >>> xl = np.r_[np.zeros(n), switch[~one]]
        >>> xr = np.r_[np.where(one, T, switch), T[~one]]
        >>> c = np.r_[np.where(one, 0, 1), np.zeros((~one).sum(), dtype=int)]
        >>> Z = np.r_[np.zeros(n), np.ones((~one).sum())]
        >>> model = WeibullPH.fit_tvc(i, xl, xr, c, Z)
        >>> model.params.round(3)  # alpha, beta (shape), coef_0
        array([2.059, 0.992, 0.995])

        The survival of a unit whose stress switches on at time 1:

        >>> model.sf_tvc([1.0, 2.0], [[0.0], [1.0]], xl=[0.0, 1.0]).round(4)
        array([0.6136, 0.1661])
        """
        # Local import avoids a circular import at package load (the
        # proportional_hazards package imports this mixin).
        from .proportional_hazards.tvc import handle_tvc

        x, c_arr, n_arr, tl, Z_arr, ident = handle_tvc(i, xl, xr, c, Z, n)
        # handle_tvc returns the rows grouped by subject in entry order.
        _, first, counts = np.unique(
            ident, return_index=True, return_counts=True
        )
        # The rows as given, in the order of handle_tvc
        # (model.data._source_rows maps the fitted rows to them).
        given = start_stop_rows(ident, tl, x, c_arr, Z_arr, n_arr)
        if (
            float(self.support[0]) == -np.inf  # type: ignore[attr-defined]
            and getattr(self, "fitter_kind", "") == "additive hazards fitter"
        ):
            _check_additive_from_zero(ident, first, counts, tl, x)
        tl = _entered_at(self, tl)
        t = np.column_stack([tl, np.full(tl.shape[0], np.inf)])
        model = self.fit(  # type: ignore[attr-defined]
            x=x, Z=Z_arr, c=c_arr, n=n_arr, t=t, **kwargs
        )
        mark_tvc_fit(model, given)
        _count_subjects(model, first, counts, n_arr)
        return model

    fit_tvc_penalised = penalised_entry(
        "_fit_tvc_penalised",
        "call fit_tvc(..., penalizer=...); its default was penalizer=0.1",
    )

    def _fit_tvc_penalised(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        penalizer: float | npt.ArrayLike = 0.1,
        l1_ratio: float = 0.0,
        standardise: bool = True,
        tol: float = 1e-10,
        w: npt.ArrayLike | float | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the model to time-varying covariates in start-stop format (as
        :meth:`fit_tvc`) with a ridge, lasso or elastic-net penalty on the
        coefficients (as ``fit_penalised``).

        The rows are validated and fitted as the delayed-entry rows
        :meth:`fit_tvc` fits, by ``fit_penalised``'s objective, with ``N``
        the number of *subjects* (the distinct ``i``, each weighted by the
        ``n`` of its last row), not of rows, and the covariates
        standardised over the subjects: each subject's covariate is its
        average over its follow-up, each row weighted by its length ``xr -
        xl``. Splitting a subject's follow-up into more rows with the same
        covariates leaves the likelihood unchanged (the cumulative hazard
        is additive over the rows), and so, with ``N`` and the scale per
        subject, it leaves the fit unchanged too, as for
        ``CoxPH.fit_tvc_penalised``. The model has no covariance,
        standard errors, bounds or tests (see ``fit_penalised``).

        The parametric additive hazards models (``WeibullAH``, ...) have
        no penalised fit, with or without time-varying covariates, and
        refuse: their coefficients are rates, so a penalty on them would
        depend on the unit of time.

        Parameters
        ----------
        i, xl, xr, c, Z, n : array_like
            As for :meth:`fit_tvc`.
        penalizer, l1_ratio, standardise, tol
            As for ``fit_penalised``.
        w : None
            Weights are refused here: give them to ``fit_penalised`` on
            rows without time-varying covariates, or to the unpenalised
            :meth:`fit_tvc`.

        Returns
        -------
        ParametricRegressionModel
            The fitted model, with ``is_tvc`` and ``penalty`` set.

        Examples
        --------
        Units fail at rate 0.5 until a stress switches on at a random
        time, and at ``0.5 e`` after it; two other covariates do nothing:

        >>> import numpy as np
        >>> from surpyval import WeibullPH
        >>> rng = np.random.default_rng(0)
        >>> n = 300
        >>> switch = rng.uniform(0.3, 1.5, n)
        >>> t_low = rng.exponential(2.0, n)
        >>> t_high = switch + rng.exponential(2.0 / np.e, n)
        >>> T = np.where(t_low > switch, t_high, t_low)
        >>> one = T <= switch
        >>> i = np.r_[np.arange(n), np.flatnonzero(~one)]
        >>> xl = np.r_[np.zeros(n), switch[~one]]
        >>> xr = np.r_[np.where(one, T, switch), T[~one]]
        >>> c = np.r_[np.where(one, 0, 1), np.zeros((~one).sum(), dtype=int)]
        >>> stress = np.r_[np.zeros(n), np.ones((~one).sum())]
        >>> Z = np.column_stack([stress, rng.normal(size=(n, 2))[i]])
        >>> model = WeibullPH.fit_tvc(
        ...     i, xl, xr, c, Z, penalizer=0.05, l1_ratio=1
        ... )
        >>> model.params.round(3)  # alpha, beta (shape), coefficients
        array([1.928, 1.032, 0.895, 0.087, 0.   ])
        """
        from ._penalised import refuse_weights
        from ._penalised_fit import fit_parametric_penalised
        from .proportional_hazards.tvc import handle_tvc

        kind = getattr(self, "_penalised_kind", None)
        if kind is None:
            raise ValueError(
                "A penalised fit_tvc is not available for the parametric "
                "additive hazards models, which have no penalised fit: "
                "their coefficients are rates (an excess hazard per unit of "
                "time), so a penalty on their size would depend on the unit "
                "of time. Fit the unpenalised model with fit_tvc, or a "
                "penalised proportional hazards (WeibullPH.fit_tvc(..., "
                "penalizer=...)) or Cox (CoxPH.fit_tvc(..., penalizer=...)) "
                "model."
            )
        refuse_weights(w)
        x, c_arr, n_arr, tl, Z_arr, ident = handle_tvc(i, xl, xr, c, Z, n)
        _, first, counts = np.unique(
            ident, return_index=True, return_counts=True
        )
        given = start_stop_rows(ident, tl, x, c_arr, Z_arr, n_arr)
        Z_arr = np.asarray(Z_arr, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        units = subject_units(x, tl, Z_arr, n_arr, ident)
        # Each row enters at its xl, as for fit_tvc (#952).
        tl = _entered_at(self, tl)
        t = np.column_stack([tl, np.full(tl.shape[0], np.inf)])
        model = fit_parametric_penalised(
            self,
            kind,
            self._penalised_link(),  # type: ignore[attr-defined]
            x,
            Z_arr,
            c_arr,
            n_arr,
            t,
            penalizer,
            l1_ratio,
            standardise,
            tol,
            units=units,
        )
        mark_tvc_fit(model, given)
        _count_subjects(model, first, counts, n_arr)
        return model

    fit_tvc_timeline_penalised = penalised_entry(
        "_fit_tvc_timeline_penalised",
        "call fit_tvc_timeline(..., penalizer=...); its default was "
        "penalizer=0.1",
    )

    def _fit_tvc_timeline_penalised(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        penalizer: float | npt.ArrayLike = 0.1,
        l1_ratio: float = 0.0,
        standardise: bool = True,
        tol: float = 1e-10,
        w: npt.ArrayLike | float | None = None,
    ) -> ParametricRegressionModel:
        """
        :meth:`fit_tvc_timeline` with a ridge, lasso or elastic-net
        penalty: the timeline is expanded to start-stop rows and fitted by
        :meth:`fit_tvc_penalised`, so it gives the same fit as the
        equivalent start-stop data.

        Parameters
        ----------
        i, x, Z, c, n : array_like
            As for :meth:`fit_tvc_timeline`.
        penalizer, l1_ratio, standardise, tol, w
            As for :meth:`fit_tvc_penalised`.

        Returns
        -------
        ParametricRegressionModel
            The fitted model, with ``is_tvc`` and ``penalty`` set.
        """
        from .proportional_hazards.tvc import handle_tvc_timeline

        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_tvc_timeline(i, x, Z, c, n)
        return self._fit_tvc_penalised(
            i_ss,
            xl,
            xr,
            c_ss,
            Z_ss,
            n=n_ss,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            standardise=standardise,
            tol=tol,
            w=w,
        )

    @penalty_argument("_fit_tvc_timeline_penalised")
    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> ParametricRegressionModel:
        """
        Fit the model from a covariate *timeline* (the ``xicnt``-style input).

        Each subject's rows give its covariate history: a value ``Z`` takes
        effect at time ``x`` and holds until the subject's next row, with the
        terminal event / censoring on the last row's ``c`` (``0`` event, ``1``
        censored). The timeline is expanded to start-stop intervals and
        fitted as :meth:`fit_tvc`, so the fit is identical to the
        equivalent start-stop data.

        Parameters
        ----------
        i : array_like
            Subject identifier of each timeline row.
        x : array_like
            The time each row's covariate value takes effect, strictly
            increasing within a subject: the first is the entry time, the
            last the event or censoring time.
        Z : array_like
            The covariate row in force from ``x``; the value on a subject's
            last row is ignored.
        c : array_like
            Status, read from each subject's last row only (``0`` event,
            ``1`` censored).
        n : array_like, optional
            Count weight of each subject, read from its last row.
        **kwargs
            Passed to ``fit`` (``init``, ``fixed``, ``center``).

        Returns
        -------
        ParametricRegressionModel
            The fitted model, with ``is_tvc`` set.
        """
        from .proportional_hazards.tvc import handle_tvc_timeline

        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_tvc_timeline(i, x, Z, c, n)
        return self.fit_tvc(i_ss, xl, xr, c_ss, Z_ss, n_ss, **kwargs)

    def fit_tvc_from_df(
        self,
        df: pd.DataFrame,
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str] | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        **kwargs: Any,
    ) -> ParametricRegressionModel:
        """Fit start-stop time-varying-covariate data from a DataFrame.

        ``i_col``, ``xl_col``, ``xr_col``, ``c_col`` and ``n_col`` name the
        columns passed to :meth:`fit_tvc` as ``i``, ``xl``, ``xr``, ``c`` and
        ``n``. The covariates are ``Z_cols``, a column name or a list of
        them, or instead ``formula``, a ``formulaic`` formula as in
        ``fit_from_df``, which codes categorical (e.g. ``"yes"`` / ``"no"``)
        columns; give exactly one. The model records ``feature_names``
        (and the ``formula`` and its encoding), so it predicts from a
        DataFrame with the same design. Other keyword arguments go to
        ``fit`` (``init``, ``fixed``, ``center``). Returns the fitted
        ``ParametricRegressionModel``.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import WeibullPH
        >>> df = pd.DataFrame({
        ...     "id": [0, 0, 1, 2, 2, 3, 4, 4, 5, 6],
        ...     "start": [0, 2, 0, 0, 1, 0, 0, 3, 0, 0],
        ...     "stop": [2, 5, 3, 1, 4, 6, 3, 7, 2, 8],
        ...     "c": [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        ...     "dose": ["low", "high", "low", "low", "high", "low", "low",
        ...              "high", "high", "low"],
        ... })
        >>> model = WeibullPH.fit_tvc_from_df(
        ...     df, "id", "start", "stop", "c", formula="dose"
        ... )
        >>> model.feature_names
        ['dose[T.low]']
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        return fit_tvc_df(
            self.fit_tvc,
            df,
            {"i": i_col, "xl": xl_col, "xr": xr_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            **kwargs,
        )

    def fit_tvc_timeline_from_df(
        self,
        df: pd.DataFrame,
        i_col: str,
        x_col: str,
        Z_cols: str | list[str] | None,
        c_col: str,
        n_col: str | None = None,
        formula: str | None = None,
        **kwargs: Any,
    ) -> ParametricRegressionModel:
        """Fit a covariate timeline from a DataFrame.

        ``i_col``, ``x_col``, ``c_col`` and ``n_col`` name the columns
        passed to :meth:`fit_tvc_timeline` as ``i``, ``x``, ``c`` and ``n``.
        The covariates are ``Z_cols``, a column name or a list of them, or
        instead (pass ``Z_cols=None``) ``formula``, as for
        :meth:`fit_tvc_from_df`; the model records ``feature_names`` (and
        the ``formula`` and its encoding). Other keyword arguments go to
        ``fit`` (``init``, ``fixed``, ``center``). Returns the fitted
        ``ParametricRegressionModel``.
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        return fit_tvc_df(
            self.fit_tvc_timeline,
            df,
            {"i": i_col, "x": x_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            **kwargs,
        )
