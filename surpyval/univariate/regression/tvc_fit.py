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
        """
        # Local import avoids a circular import at package load (the
        # proportional_hazards package imports this mixin).
        from .proportional_hazards.tvc import handle_tvc

        x, c_arr, n_arr, tl, Z_arr, _ = handle_tvc(i, xl, xr, c, Z, n)
        t = np.column_stack([tl, np.full(tl.shape[0], np.inf)])
        model = self.fit(  # type: ignore[attr-defined]
            x=x, Z=Z_arr, c=c_arr, n=n_arr, t=t, **kwargs
        )
        model.is_tvc = True
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
        censored). The timeline is expanded to start-stop intervals (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc_timeline`) and fitted as :meth:`fit_tvc`.
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
        """Fit start-stop time-varying-covariate data from a DataFrame; see
        :meth:`fit_tvc`."""
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
        """Fit a covariate timeline from a DataFrame; see
        :meth:`fit_tvc_timeline`."""
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
