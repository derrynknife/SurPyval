"""
Parametric additive hazards regression.

Where the semi-parametric ``AdditiveHazards`` (Lin & Ying) leaves the baseline
hazard unspecified, this fits a fully *parametric* additive hazards model,

.. math::
    h(x \\mid Z) = h_0(x;\\,\\theta) + \\beta' Z

with a parametric baseline hazard :math:`h_0(x;\\theta)` (Weibull,
Exponential, ...) estimated jointly with the risk-difference coefficients
:math:`\\beta` by maximum likelihood. Because the covariate effect is additive
(a constant integrates to :math:`x\\,\\beta'Z`), the cumulative hazard is

.. math::
    H(x \\mid Z) = H_0(x;\\,\\theta) + x\\,\\beta' Z .

This is the additive-scale companion to the parametric proportional hazards
models (``WeibullPH`` and friends), and gives a smooth, extrapolatable
version of what the semi-parametric ``AdditiveHazards`` estimates.

A caveat inherent to additive hazards: nothing constrains
:math:`h_0(x;\\theta) + \\beta' Z > 0`. The likelihood needs :math:`\\log h`
for every observed event, so if the fitted hazard is driven non-positive at
an observed time the log-likelihood becomes ``nan`` and the fit fails rather
than returning a silently invalid model. This is deliberate: an additive
hazards model that cannot keep the hazard positive over the data is not a
valid description of it. When covariate effects are strongly protective a
proportional hazards model, whose exponential form keeps the hazard positive
by construction, is the safer choice.
"""

import autograd.numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from ..tvc_fit import TVCFitMixin


class _AdditiveReg:
    # Lightweight namespace for the fitted model's ``reg_model`` attribute;
    # the model repr reads ``.name``.
    name: str
    phi_param_map: object


class AdditiveHazardsFitter(TVCFitMixin, DataFrameRegressionMixin):
    def __init__(self, name, dist):
        self.name = name
        self.dist = dist
        self.k_dist = len(dist.param_names)
        self.bounds = dist.bounds
        self.support = dist.support
        self.param_names = dist.param_names
        self.param_map = {v: i for i, v in enumerate(dist.param_names)}
        self.Hf_dist = dist.Hf
        self.hf_dist = dist.hf

    # -- covariate-aware distribution functions (x, Z, *params) -----------

    def _beta_Z(self, Z, beta):
        return np.dot(Z, np.array(beta))

    def hf(self, x, Z, *params):
        dist_params = np.array(params[: self.k_dist])
        beta = params[self.k_dist :]
        return self.hf_dist(x, *dist_params) + self._beta_Z(Z, beta)

    def Hf(self, x, Z, *params):
        # H(x | Z) = H_0(x) + integral_0^x beta'Z ds = H_0(x) + x * beta'Z.
        dist_params = np.array(params[: self.k_dist])
        beta = params[self.k_dist :]
        return self.Hf_dist(x, *dist_params) + x * self._beta_Z(Z, beta)

    def sf(self, x, Z, *params):
        return np.exp(-self.Hf(x, Z, *params))

    def ff(self, x, Z, *params):
        return 1 - np.exp(-self.Hf(x, Z, *params))

    def df(self, x, Z, *params):
        return self.hf(x, Z, *params) * np.exp(-self.Hf(x, Z, *params))

    def log_df(self, x, Z, *params):
        # log h - H. When the additive hazard is driven non-positive the log
        # is nan; the optimiser then rejects that point (the fit fails rather
        # than returning an invalid model).
        return np.log(self.hf(x, Z, *params)) - self.Hf(x, Z, *params)

    def log_sf(self, x, Z, *params):
        return -self.Hf(x, Z, *params)

    def log_ff(self, x, Z, *params):
        return np.log(self.ff(x, Z, *params))

    # mpp transforms are the identity (probability plotting is not used for
    # these models, but the interface is kept consistent with the other
    # regression fitters).
    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_inv_y_transform(self, y, *params):
        return y

    def neg_ll(self, data, *params):
        return regression_neg_ll(self, data, *params)

    def random(self, size, Z, *params):
        """
        Draw ``size`` samples for a single covariate vector ``Z`` by
        numerically inverting the (monotone) cumulative hazard. Requires the
        additive hazard to stay positive over the sampled range.
        """
        dist_params = np.array(params[: self.k_dist])
        beta = np.array(params[self.k_dist :])
        Z = np.asarray(Z, dtype=float).ravel()
        bz = float(np.dot(Z, beta))
        target = -np.log(np.random.uniform(0, 1, size))

        def cum_haz(xv):
            return self.Hf_dist(xv, *dist_params) + xv * bz

        lo = np.zeros(size)
        hi = np.ones(size)
        for _ in range(200):
            below = cum_haz(hi) < target
            if not np.any(below):
                break
            hi = np.where(below, hi * 2.0, hi)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            below = cum_haz(mid) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        x = 0.5 * (lo + hi)
        Z_out = np.ones_like(x)[:, None] * Z
        return x.flatten(), Z_out.flatten()

    # -- factory ----------------------------------------------------------

    @staticmethod
    def create(distribution):
        """
        Create a parametric additive hazards fitter for the given
        distribution.

        Parameters
        ----------
        distribution : ParametricFitter
            A surpyval parametric distribution (e.g. ``Weibull``,
            ``Exponential``).

        Returns
        -------
        AdditiveHazardsFitter
            A configured fitter with a ``.fit(x, Z, ...)`` method.
        """
        return AdditiveHazardsFitter(f"{distribution.name}AH", distribution)

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the parametric additive hazards model by maximum likelihood.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The covariate matrix (one row per observation).
        c : array_like, optional
            The censoring indicators (0 observed, 1 right, -1 left, 2
            interval).
        n : array_like, optional
            The count of observations at each time.
        t : array_like, optional
            The truncation matrix.
        init : array_like, optional
            Initial parameter values (baseline parameters followed by the
            covariate coefficients).
        fixed : dict, optional
            Parameters to hold fixed, by name.

        Returns
        -------

        ParametricRegressionModel
            The fitted model. Raises if the additive hazard cannot be kept
            positive over the data (the log-likelihood is then non-finite).

        Examples
        --------

        >>> from surpyval import WeibullAH
        >>> model = WeibullAH.fit(x=x, Z=Z, c=c)
        """
        data = SurpyvalData(x, c, n, t, group_and_sort=False)
        data.add_covariates(Z)

        if fixed is None:
            fixed = {}

        assert data.Z is not None  # set by add_covariates above
        n_cov = data.Z.shape[1]
        beta_param_map = {
            "beta_" + str(i): self.k_dist + i for i in range(n_cov)
        }
        param_map = {**self.param_map, **beta_param_map}
        bounds = (*self.bounds, *(((None, None),) * n_cov))

        if init is None or len(init) == 0:  # type: ignore[arg-type]
            ps = self.dist.fit_from_surpyval_data(data).params
            init = np.array([*ps, *np.zeros(n_cov)])
        else:
            init = np.array(init)

        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            x, bounds, fixed, param_map
        )
        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            def true_neg_ll(params):
                return self.neg_ll(data, *inv_trans(const(params)))

            def fun(params):
                # Where the additive hazard goes non-positive the log-
                # likelihood is genuinely -inf; return a large finite penalty
                # (not a solver constraint) so the derivative-free optimiser
                # stays in the region where the model is valid rather than
                # stalling on nan gradients. The initial guess (beta = 0) has
                # the strictly-positive baseline hazard, so it is valid.
                val = true_neg_ll(params)
                return val if np.isfinite(val) else 1e15

            res = minimize(fun, init, method="Nelder-Mead")
            res = minimize(fun, res.x, method="TNC")

            params = inv_trans(const(res.x))

            # The penalty above can leave the optimiser at the edge of the
            # valid region; recompute the unpenalised likelihood and fail if
            # it is not finite (the additive hazard could not be kept
            # positive at the optimum).
            final_neg_ll = float(true_neg_ll(res.x))
        if not np.isfinite(final_neg_ll):
            raise ValueError(
                "The additive hazards fit could not keep the hazard "
                "h_0(x) + beta'Z positive at every observed time. A "
                "proportional hazards model (e.g. {}PH) keeps the hazard "
                "positive by construction and may be more appropriate for "
                "this data.".format(self.dist.name)
            )

        reg_model = _AdditiveReg()
        reg_model.name = "Additive [beta'Z]"
        reg_model.phi_param_map = beta_param_map

        model = ParametricRegressionModel()
        model.distribution_param_map = self.param_map
        model.phi_param_map = beta_param_map
        model.model = self
        model.reg_model = reg_model
        model.kind = "Additive Hazard"
        model.distribution = self.dist
        model.params = np.array(params)
        model.res = res
        model._neg_ll = final_neg_ll
        model.fixed = fixed
        model.k_dist = self.k_dist
        model.k = len(bounds)
        model.data = data

        return model
