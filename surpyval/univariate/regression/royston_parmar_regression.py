"""Royston-Parmar (flexible parametric) regression: a restricted cubic
spline baseline in log time with covariates.

The baseline is the Royston-Parmar model of
:mod:`surpyval.univariate.parametric.royston_parmar`, a restricted cubic
spline :math:`s(\\log t; \\gamma)` on a link scale, and the covariates
shift it (Royston and Parmar 2002):

* ``scale="hazard"``: :math:`\\log H(t \\mid Z) = s(\\log t; \\gamma) +
  \\beta' Z`, proportional hazards; :math:`e^{\\beta}` are hazard ratios.
  It is ``PH`` on the spline baseline, and R's
  ``flexsurvspline(..., scale = "hazard")`` and ``rstpm2::stpm2``.
* ``scale="odds"``: the survival odds :math:`S / F` multiplied by
  :math:`e^{\\beta' Z}`, so :math:`\\operatorname{logit} F(t \\mid Z) =
  s(\\log t; \\gamma) - \\beta' Z`: ``PO`` on the spline baseline, whose
  :math:`e^{\\beta}` are survival-odds ratios as for every SurPyval
  proportional odds model (flexsurv's odds-scale coefficients are these
  with the sign changed).

A covariate shift on the normal (probit) scale is neither proportional
hazards nor proportional odds, so ``scale="normal"`` is refused here; the
univariate ``RoystonParmar`` fits it without covariates.

The knots are placed as the univariate model places them, at quantiles of
the log event times (exactly observed failures; else the ends of the
interval and left censored rows), the boundary knots at the extremes, and
are then held: the bounds, the bootstrap and the likelihood-ratio searches
are conditional on them, as R's are.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.scipy.special import expit

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    ParametricFitter,
)
from surpyval.univariate.parametric.royston_parmar import (
    _SCALES,
    RoystonParmar,
    RoystonParmarModel,
    _place_knots,
    _rcs_basis,
    _rcs_deriv,
)
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.surpyval_data import SurpyvalData, accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option

from ._aliasing import dataframe_covariates
from .parametric_regression_model import ParametricRegressionModel
from .regression_data import DataFrameRegressionMixin, truncation_window

#: The scales a spline regression takes (the univariate model's less the
#: normal, whose covariate shift is neither PH nor PO).
REGRESSION_SCALES = ("hazard", "odds")


class SplineBaseline(ParametricFitter):
    """The Royston-Parmar baseline on fixed ``knots`` (in log time) and
    ``scale``, as a distribution of the spline coefficients
    ``gamma_0, ..., gamma_m`` that the ``PH`` (hazard scale) and ``PO``
    (odds scale) families take. Built by :class:`RoystonParmarRegression`
    for each fit; not for direct use."""

    def __init__(self, knots: npt.ArrayLike, scale: str) -> None:
        check_option("scale", scale, REGRESSION_SCALES)
        self.knots = np.asarray(knots, dtype=float)
        self.scale = scale
        k = self.knots.size
        names = [f"gamma_{i}" for i in range(k)]
        super().__init__(
            name="RoystonParmar",
            k=k,
            bounds=((None, None),) * k,
            support=(0, np.inf),
            parameter_names=names,
            param_map={nm: i for i, nm in enumerate(names)},
            plot_x_scale="log",
        )
        # The Wald band's scale (utils.linalg.cb_link): the spline's own.
        self._cb_link = "loglog" if scale == "hazard" else "logit"

    def __repr__(self) -> str:
        return "RoystonParmar spline baseline ({} scale, {} knots)".format(
            self.scale, self.knots.size
        )

    def __reduce__(self) -> Any:
        return (SplineBaseline, (self.knots, self.scale))

    # -- the spline --------------------------------------------------------

    def _eta_and_slope(self, x: Numeric, gamma: tuple) -> tuple[Any, Any]:
        """``(s(log x), s'(log x))`` at ``x``, in its shape."""
        x = np.asarray(x, dtype=float)
        lx = np.log(x.reshape(-1))
        g = anp.array(list(gamma))
        eta = anp.dot(_rcs_basis(lx, self.knots), g)
        slope = anp.dot(_rcs_deriv(lx, self.knots), g)
        return anp.reshape(eta, x.shape), anp.reshape(slope, x.shape)

    def Hf(self, x: Numeric, *gamma: Boxable) -> Boxable:
        eta, _ = self._eta_and_slope(x, gamma)
        if self.scale == "hazard":
            return anp.exp(eta)
        return anp.logaddexp(0.0, eta)

    def hf(self, x: Numeric, *gamma: Boxable) -> Boxable:
        eta, slope = self._eta_and_slope(x, gamma)
        x = np.asarray(x, dtype=float)
        if self.scale == "hazard":
            return anp.exp(eta) * slope / x
        return expit(eta) * slope / x

    def sf(self, x: Numeric, *gamma: Boxable) -> Boxable:
        return anp.exp(-self.Hf(x, *gamma))

    def ff(self, x: Numeric, *gamma: Boxable) -> Boxable:
        return -anp.expm1(-self.Hf(x, *gamma))

    def df(self, x: Numeric, *gamma: Boxable) -> Boxable:
        return self.hf(x, *gamma) * self.sf(x, *gamma)

    def log_sf(self, x: Numeric, *gamma: Boxable) -> Boxable:
        return -self.Hf(x, *gamma)

    def log_ff(self, x: Numeric, *gamma: Boxable) -> Boxable:
        eta, _ = self._eta_and_slope(x, gamma)
        if self.scale == "hazard":
            # log(1 - exp(-e^eta)), accurate in both tails
            H = anp.exp(eta)
            return anp.where(
                H < 0.6931,
                anp.log(-anp.expm1(-H)),
                anp.log1p(-anp.exp(-H)),
            )
        return -anp.logaddexp(0.0, -eta)

    def log_df(self, x: Numeric, *gamma: Boxable) -> Boxable:
        eta, slope = self._eta_and_slope(x, gamma)
        lx = np.log(np.asarray(x, dtype=float))
        if self.scale == "hazard":
            return eta + anp.log(slope) - lx - anp.exp(eta)
        return eta - 2 * anp.logaddexp(0.0, eta) + anp.log(slope) - lx

    def qf(self, u: Numeric, *gamma: Boxable) -> Boxable:
        model = self._univariate(gamma)
        return model.qf(u)

    def _univariate(self, gamma: tuple) -> RoystonParmarModel:
        model = RoystonParmarModel()
        model.scale = self.scale
        model.knots = self.knots
        model.params = np.array([float(g) for g in gamma], dtype=float)
        return model

    def mpp_x_transform(self, x: Any) -> Any:
        return np.log(x)

    def fit_from_surpyval_data(self, data: SurpyvalData, **kwargs: Any) -> Any:
        """The univariate Royston-Parmar fit on these knots: the start of
        a regression fit."""
        model = RoystonParmar.fit(
            x=data.x,
            c=data.c,
            n=data.n,
            t=data.t,
            knots=self.knots,
            scale=self.scale,
        )
        return SimpleNamespace(params=np.asarray(model.params, dtype=float))

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        return self.fit_from_surpyval_data(data).params


def regression_knots(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    t: npt.ArrayLike | None,
    df: int,
) -> npt.NDArray:
    """The knots (log time) for ``df``: quantiles of the log event times
    of the rows the fit uses (those with every covariate finite), each
    time counted ``n`` times, as flexsurv places them (the exactly
    observed failures; else the ends of the interval and left censored
    rows)."""
    data = SurpyvalData(x=x, c=c, n=n, t=t, group_and_sort=False)
    Z_arr = np.asarray(Z, dtype=float).reshape(len(data), -1)
    finite = np.isfinite(Z_arr).all(axis=1)
    if not finite.all() and finite.any():
        data = data[finite]
    event = np.repeat(data.x_o, data.n_o.astype(int))
    if event.size == 0:
        n_i = data.n_i.astype(int)
        event = np.concatenate(
            [
                np.repeat(data.x_il, n_i),
                np.repeat(data.x_ir, n_i),
                np.repeat(data.x_l, data.n_l.astype(int)),
            ]
        )
        event = event[np.isfinite(event) & (event > 0)]
    n_internal = max(int(df) - 1, 0)
    distinct = np.unique(event).size
    if distinct < n_internal + 2:
        raise ValueError(
            "A Royston-Parmar fit with df={} needs at least {} distinct "
            "event times to place its knots; the data have {}. Reduce "
            "`df` or pass `knots`.".format(df, n_internal + 2, distinct)
        )
    knots = _place_knots(event, n_internal)
    if np.unique(knots).size != knots.size:
        raise ValueError(
            "The knots placed at quantiles of the event times coincide "
            "(tied event times); reduce `df` or pass `knots`."
        )
    return knots


class RoystonParmarRegression(FitterRepr, DataFrameRegressionMixin):
    """
    Royston-Parmar (flexible parametric) regression: a restricted cubic
    spline baseline in log time, with ``df`` degrees of freedom (``df -
    1`` internal knots), and covariates acting as proportional hazards
    (``scale="hazard"``) or proportional odds (``scale="odds"``).

    The model is

    .. math::
        \\log H(t \\mid Z) = s(\\log t; \\gamma) + \\beta' Z

    on the hazard scale, and :math:`\\operatorname{logit} F(t \\mid Z) =
    s(\\log t; \\gamma) - \\beta' Z` on the odds scale (the survival odds
    multiplied by :math:`e^{\\beta' Z}`, as ``PO``). ``df=1`` on the
    hazard scale is the Weibull proportional hazards model. The fitted
    model is a
    :class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`
    whose baseline parameters are the spline coefficients ``gamma_j``,
    with every bound (Wald, likelihood ratio and bootstrap, conditional on
    the knots), ``summary``, ``fit_from_df`` and saving.

    Parameters
    ----------
    df : int, optional
        Degrees of freedom of the spline: the number of terms beyond the
        intercept. Default 3. Compare a few by AIC.
    scale : {"hazard", "odds"}, optional
        How the covariates act. Default ``"hazard"``.
    knots : array_like, optional
        Explicit knots in *log* time, the boundary knots included,
        instead of the quantiles of the event times.

    Examples
    --------
    >>> from surpyval import RoystonParmarPH
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> df["censored"] = 1 - df["arrest"]
    >>> model = RoystonParmarPH.fit_from_df(
    ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    ... )
    >>> model.params[-3:].round(4)
    array([-0.3488, -0.067 ,  0.0974])
    """

    fitter_kind = "flexible parametric (spline) regression fitter"

    def __init__(
        self,
        df: int = 3,
        scale: str = "hazard",
        knots: npt.ArrayLike | None = None,
    ) -> None:
        if scale == "normal":
            raise ValueError(
                "scale='normal' is not available with covariates: a shift "
                "of the probit of F is neither proportional hazards nor "
                "proportional odds. Use scale='hazard' or 'odds', "
                "LogNormalAFT for a log-normal model with covariates, or "
                "RoystonParmar.fit(..., scale='normal') without them."
            )
        check_option("scale", scale, _SCALES)
        if int(df) != df or df < 1:
            raise ValueError(
                f"`df` must be a whole number at least 1; got {df!r}."
            )
        self.df = int(df)
        self.scale = scale
        self.knots = None if knots is None else np.asarray(knots, float)
        self.name = "RoystonParmar{}".format(
            "PH" if scale == "hazard" else "PO"
        )

    def __repr__(self) -> str:
        return "{}: {} (df={}, {} scale)".format(
            self.name, self.fitter_kind, self.df, self.scale
        )

    def family(self, knots: npt.ArrayLike) -> Any:
        """The ``PH`` (hazard scale) or ``PO`` (odds scale) fitter on the
        spline baseline with these ``knots``."""
        from .proportional_hazards import PH
        from .proportional_odds import PO

        baseline = SplineBaseline(knots, self.scale)
        return (PH if self.scale == "hazard" else PO)(baseline)

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit by maximum likelihood: the knots are placed on the data (or
        taken from the fitter), then the spline coefficients and the
        covariate coefficients are fitted together, by the ``PH`` or
        ``PO`` family's fit.

        Parameters
        ----------
        x, Z, c, n, t, tl, tr : array_like
            As for ``WeibullPH.fit``: every kind of censoring (``x`` an
            ``(N, 2)`` array for intervals) and left and right
            truncation.
        init : array_like, optional
            Initial values: the spline coefficients, then the covariate
            coefficients. Default the univariate fit's and zeros.
        fixed : dict, optional
            Parameters to hold, by name (``"gamma_1"``, a coefficient's
            column name or ``"coef_j"``).
        center : bool, optional
            As for ``WeibullPH.fit``.

        Returns
        -------
        ParametricRegressionModel
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import RoystonParmarRegression
        >>> rng = np.random.default_rng(2)
        >>> Z = rng.binomial(1, 0.5, (300, 1))
        >>> x = 10 * rng.weibull(1.5, 300) * np.exp(-0.4 * Z[:, 0])
        >>> model = RoystonParmarRegression(df=2).fit(x, Z)
        >>> model.parameter_names
        ['gamma_0', 'gamma_1', 'gamma_2', 'coef_0']
        """
        t = truncation_window(x, t, tl, tr)
        knots = self.knots
        if knots is None:
            knots = regression_knots(x, Z, c, n, t, self.df)
        return self.family(knots).fit(
            x, Z, c=c, n=n, t=t, init=init, fixed=fixed, center=center
        )


#: Royston-Parmar proportional hazards regression with three degrees of
#: freedom; ``RoystonParmarRegression(df=..., scale=...)`` for others.
RoystonParmarPH = RoystonParmarRegression(3, "hazard")
#: Royston-Parmar proportional odds regression with three degrees of
#: freedom.
RoystonParmarPO = RoystonParmarRegression(3, "odds")
