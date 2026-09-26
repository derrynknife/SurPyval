"""Time-varying-covariate fitting for the parametric regression families.

For a proportional-hazards or additive-hazards model the cumulative hazard is
*additive over disjoint time intervals*:

- PH: ``H(t) = sum_seg [H0(xr) - H0(xl)] * exp(Z_seg' beta)``
- AH: ``H(t) = [H0(xr) - H0(xl)] + (Z_seg' beta) * (xr - xl)`` summed over
  segments,

so a subject observed with a time-varying covariate factorises exactly into
one *left-truncated* (delayed-entry) observation per constant-covariate
interval -- entering at ``xl`` and exiting at ``xr``. This is the same
episode-splitting identity the Cox partial likelihood uses; here it lets the
ordinary parametric MLE ``fit`` (which already accepts truncation ``t``) fit
start-stop data with no new likelihood. The mixin therefore just reshapes the
time-varying-covariate data and calls ``fit``.

It is mixed into the fitters whose cumulative hazard is additive over
intervals (``ProportionalHazardsFitter``, ``AdditiveHazardsFitter``). It is
*not* correct for accelerated failure time (which must accumulate an
"accelerated age" across intervals) or proportional odds (no additive
structure), so those fitters do not expose it.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

    from .parametric_regression_model import ParametricRegressionModel


class TVCFitMixin:
    """Adds ``fit_tvc`` (start-stop and timeline, array and DataFrame) to a
    parametric regression fitter whose cumulative hazard is additive over time
    intervals. Requires the host class to provide a ``fit(x, Z, c, n, t, ...)``
    method that accepts truncation ``t`` as a ``[tl, tr]`` matrix."""

    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> "ParametricRegressionModel":
        """
        Fit the model to time-varying covariates in start-stop format.

        Each row is one observation interval ``(xl, xr]`` of subject ``i`` on
        which the covariate row ``Z`` is constant; ``c`` is ``0`` (event at
        ``xr``) only on the interval that ends at the subject's event and ``1``
        (right-censored) otherwise -- surpyval's censoring convention. The
        intervals are validated and mapped to left-truncated observations
        (``t = [xl, inf]``), then fitted with the ordinary parametric MLE, so
        the fit is identical to the equivalent non-time-varying data. Extra
        keyword arguments (``init``, ``fixed``) are passed through to ``fit``.

        A subject's rows must not overlap, it may have at most one event,
        and that event must be on its last interval; gaps between its
        intervals (not at risk) and delayed entry (a first ``xl > 0``) are
        allowed.

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
            Passed to ``fit`` (``init``, ``fixed``).

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
        >>> model.params.round(3)  # alpha, beta (shape), beta_0
        array([2.059, 0.992, 0.995])

        The survival of a unit whose stress switches on at time 1:

        >>> model.sf_tvc([1.0, 2.0], [[0.0], [1.0]], xl=[0.0, 1.0]).round(4)
        array([0.6136, 0.1661])
        """
        # Local import avoids a circular import at package load (the
        # proportional_hazards package imports this mixin).
        from .proportional_hazards.tvc import handle_tvc

        x, c_arr, n_arr, tl, Z_arr, ident = handle_tvc(i, xl, xr, c, Z, n)
        t = np.column_stack([tl, np.full(tl.shape[0], np.inf)])
        model = self.fit(  # type: ignore[attr-defined]
            x=x, Z=Z_arr, c=c_arr, n=n_arr, t=t, **kwargs
        )
        model.is_tvc = True
        # aic_c counts subjects, each weighted by its last interval's count
        # (as the AFT time-varying fit does), not interval rows: splitting
        # a subject's follow-up into more intervals leaves the likelihood
        # unchanged and must leave aic_c unchanged too. handle_tvc returns
        # the rows grouped by subject in entry order.
        _, first, counts = np.unique(
            ident, return_index=True, return_counts=True
        )
        model.n_subjects = int(first.shape[0])
        model._ic_n_total = float(n_arr[first + counts - 1].sum())
        return model

    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> "ParametricRegressionModel":
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
            Passed to ``fit`` (``init``, ``fixed``).

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
        df: "pd.DataFrame",
        id_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str],
        n_col: str | None = None,
        **kwargs: Any,
    ) -> "ParametricRegressionModel":
        """Fit start-stop time-varying-covariate data from a DataFrame.

        ``id_col``, ``xl_col``, ``xr_col``, ``c_col`` and ``n_col`` name the
        columns passed to :meth:`fit_tvc` as ``i``, ``xl``, ``xr``, ``c`` and
        ``n``; ``Z_cols`` is a column name or a list of them, recorded on the
        model as ``feature_names`` so it predicts from a DataFrame. Other
        keyword arguments go to ``fit`` (``init``, ``fixed``). Returns the
        fitted ``ParametricRegressionModel``.
        """
        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        model = self.fit_tvc(
            df[id_col].to_numpy(),
            df[xl_col].to_numpy(),
            df[xr_col].to_numpy(),
            df[c_col].to_numpy(),
            df[cols].to_numpy(),
            None if n_col is None else df[n_col].to_numpy(),
            **kwargs,
        )
        model.feature_names = cols
        return model

    def fit_tvc_timeline_from_df(
        self,
        df: "pd.DataFrame",
        id_col: str,
        time_col: str,
        Z_cols: str | list[str],
        c_col: str,
        n_col: str | None = None,
        **kwargs: Any,
    ) -> "ParametricRegressionModel":
        """Fit a covariate timeline from a DataFrame.

        ``id_col``, ``time_col``, ``c_col`` and ``n_col`` name the columns
        passed to :meth:`fit_tvc_timeline` as ``i``, ``x``, ``c`` and ``n``;
        ``Z_cols`` is a column name or a list of them, recorded on the model
        as ``feature_names``. Other keyword arguments go to ``fit``
        (``init``, ``fixed``). Returns the fitted
        ``ParametricRegressionModel``.
        """
        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        model = self.fit_tvc_timeline(
            df[id_col].to_numpy(),
            df[time_col].to_numpy(),
            df[cols].to_numpy(),
            df[c_col].to_numpy(),
            None if n_col is None else df[n_col].to_numpy(),
            **kwargs,
        )
        model.feature_names = cols
        return model
