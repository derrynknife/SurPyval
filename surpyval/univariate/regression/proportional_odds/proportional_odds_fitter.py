from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import (
    LogLinearPhi,
    MirroredDistributionAttrs,
    assemble_regression_model,
    finish_search,
    free_coefficients,
    keep_information,
    make_objective,
    mirror_distribution,
    optimise_nm_tnc,
    prepare_regression_fit,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from ..tvc_fit import TVCFitMixin

_TINY = float(np.finfo(float).tiny)


class ProportionalOddsFitter(
    MirroredDistributionAttrs, TVCFitMixin, DataFrameRegressionMixin
):
    """
    Proportional Odds model fitter using :math:`\\phi = e^{\\beta' Z}` as
    the multiplier of the survival odds :math:`O(x) = S(x) / F(x)`:

    .. math::
        O(x \\mid Z) = \\phi\\, O_0(x), \\qquad
        S(x \\mid Z) = \\frac{\\phi S_0(x)}{F_0(x) + \\phi S_0(x)}, \\qquad
        h(x \\mid Z) = \\frac{h_0(x)}{F_0(x) + \\phi S_0(x)}.

    A positive beta coefficient means higher covariate values increase the
    survival odds (protective effect -- longer life). This is the opposite
    sign to the PH and AFT fitters, where a positive coefficient shortens
    life; negate the coefficients to compare them.

    The hazard depends only on the time and the current covariate, so a
    fitted model is evaluated exactly along a step covariate path by
    ``sf_tvc`` / ``Hf_tvc`` (a sum of per-segment cumulative-hazard
    increments), and ``fit_tvc`` (with the timeline and DataFrame variants)
    fits start-stop time-varying-covariate data exactly by splitting each
    subject into one delayed-entry row per constant-covariate interval, as
    for the proportional and additive hazards fitters.

    Use the pre-built instances (``LogisticPO``, ``WeibullPO``, ...) or the
    ``PO`` factory.
    """

    def __init__(self, distribution: Any) -> None:
        mirror_distribution(self, distribution)
        self.Hf_dist = distribution.Hf
        self.hf_dist = distribution.hf
        self.sf_dist = distribution.sf
        self.ff_dist = distribution.ff
        self.df_dist = distribution.df
        self.log_sf_dist = distribution.log_sf

    def _phi(self, Z: Numeric, *phi_params: Boxable) -> Boxable:
        return LogLinearPhi.phi(Z, *phi_params)

    def sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Survival function :math:`\\phi S_0 / (F_0 + \\phi S_0)` at ``x`` for
        covariates ``Z``; ``params`` are the distribution parameters
        followed by the covariate coefficients.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return phi * S0 / (F0 + phi * S0)

    def ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Failure (CDF) function :math:`F_0 / (F_0 + \\phi S_0)` at ``x`` for
        covariates ``Z``; ``params`` as for :meth:`sf`.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return F0 / (F0 + phi * S0)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate :math:`h_0 / (F_0 + \\phi S_0)` at ``x`` for covariates
        ``Z``; ``params`` as for :meth:`sf`.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        h0 = self.hf_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return h0 / (F0 + phi * S0)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard :math:`-\\ln S(x \\mid Z)` at ``x`` for covariates
        ``Z``; ``params`` as for :meth:`sf`.

        Evaluated as :math:`\\ln(1 + F_0 / (\\phi S_0))`, which keeps full
        relative precision where the cumulative hazard is small (#528), and
        where :math:`\\phi S_0` is below the normal range as
        :math:`H_0(x) - \\ln\\phi + \\ln(1 + (\\phi - 1) S_0)`, which stays
        finite there (``-log(sf)`` is ``inf``, and a difference of two such
        values along a time-varying path would be ``nan``).
        """
        return -self.log_sf(x, Z, *params)

    def df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Density :math:`\\phi f_0 / (F_0 + \\phi S_0)^2` at ``x`` for
        covariates ``Z``; ``params`` as for :meth:`sf`.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        f0 = self.df_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        denom = F0 + phi * S0
        return phi * f0 / (denom * denom)

    def log_sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Log of the survival function, :math:`-H(x \\mid Z)`; see :meth:`Hf`.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        # S = phi S0 / (F0 + phi S0) = 1 / (1 + F0 / (phi S0)). The old
        # log(phi) + log(S0) - log(F0 + phi S0) cancelled to about 1e-16
        # absolute where H is small: 20 % wrong at H = 4e-16 (#528).
        scaled = phi * S0
        # Where phi S0 is below the normal range (or 0), F0 / (phi S0) can
        # overflow; there H is large and nothing cancels: log S0 from the
        # baseline's log_sf, which stays finite where S0 is 0, and
        # log(F0 + phi S0) = log1p((phi - 1) S0).
        normal = scaled >= _TINY
        # A denominator of 1 in the branch not taken keeps the value (and
        # autograd's derivative) free of inf.
        safe = np.where(normal, scaled, 1.0)
        return np.where(
            normal,
            -np.log1p(F0 / safe),
            np.log(phi)
            + self.log_sf_dist(x, *dist_params)
            - np.log1p((phi - 1.0) * S0),
        )

    def log_ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return np.log(F0) - np.log(F0 + phi * S0)

    def log_df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        f0 = self.df_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        denom = F0 + phi * S0
        return np.log(phi) + np.log(f0) - 2.0 * np.log(denom)

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

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
    ) -> ParametricRegressionModel:
        """
        Fit the proportional odds model by maximum likelihood.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The covariate matrix, one row per observation (a 1-D array is
            read as a single covariate). Rows with a missing or infinite
            covariate are dropped, with a warning.
        c : array_like, optional
            The censoring indicators (0 observed, 1 right, -1 left, 2
            interval). Defaults to all observed.
        n : array_like, optional
            The count of observations at each time. Defaults to 1.
        t : array_like, optional
            Truncation bounds: an (N, 2) array of the left and right
            truncation times of each observation.
        init : array_like, optional
            Initial parameter values: the distribution parameters followed
            by the covariate coefficients.
        fixed : dict, optional
            Parameters to hold fixed, by name (a distribution parameter
            such as ``"beta"``, or a coefficient ``"beta_0"``, ...).
        center : bool, optional
            ``False`` (the default) reports the baseline at ``Z = 0``.
            ``True`` reports the baseline at the covariate means (stored as
            ``model.center``) instead: the fit runs on ``Z - center``, and
            ``init`` and ``fixed`` are read there too. Use it for
            covariates far from 0 (a year, a date), where the baseline at
            ``Z = 0`` cannot be represented or fitted, which the default
            fit refuses with a ``ValueError`` saying so.

        Returns
        -------

        ParametricRegressionModel
            The fitted model.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Logistic, LogisticPO
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Logistic.random(100, 10, 2) + 2.0 * Z[:, 0]
        >>> model = LogisticPO.fit(x, Z)
        >>> model.params.round(3)
        array([9.708, 2.337, 0.918])
        """
        data, prep = prepare_regression_fit(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            LogLinearPhi.phi_bounds,
            LogLinearPhi.make_param_map,
            kind="Proportional Odds",
            center=center,
        )
        (
            init_t,
            bounds,
            pmap,
            transform,
            inv_trans,
            const,
            fixed,
            centring,
        ) = prep

        with np.errstate(all="ignore"):

            fun = make_objective(self, data, inv_trans, const)

            res = optimise_nm_tnc(fun, init_t, quiet=True)

        params = inv_trans(const(res.x))
        reg_model = LogLinearPhi(LogLinearPhi.NAME_EXP, pmap)

        model = assemble_regression_model(
            self,
            "Proportional Odds",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
            centring=centring,
        )
        # After the model is built (which may refuse the data), one
        # warning for what the search found (#392).
        no_maximum, derivatives = finish_search(
            fun, res, free_coefficients(self, fixed, pmap), init_t
        )
        # The exact information for the model's covariance (#392).
        keep_information(
            model, no_maximum, derivatives, inv_trans, const, res.x, centring
        )
        return model


def PO(distribution: Any) -> "ProportionalOddsFitter":
    """
    Create a Proportional Odds fitter for the given distribution.

    Uses exp(beta'Z) as the odds multiplier — the standard parameterisation
    for proportional odds survival models.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Logistic``,
        ``LogLogistic``).

    Returns
    -------
    ProportionalOddsFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Logistic
    >>> from surpyval import PO
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Logistic.random(100, 10, 2) + 2.0 * Z[:, 0]
    >>> model = PO(Logistic).fit(x, Z=Z)
    >>> model.params.round(3)
    array([9.708, 2.337, 0.918])
    """
    return ProportionalOddsFitter(distribution)
