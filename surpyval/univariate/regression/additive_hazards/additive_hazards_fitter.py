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
at every observed failure, so the optimiser only accepts parameters that keep
the hazard positive there: a non-positive hazard at a failure is a barrier,
not a failure of the fit. When the data would prefer a negative hazard -- a
strongly protective covariate -- the fit ends pressed against that barrier,
with the hazard nearly zero at one failure, ``beta`` held there and the
baseline distorted to compensate. Such a fit is returned with a warning (the
fit raises only if the optimiser cannot end at a finite likelihood at all).
Positivity is not checked between the observed times. When covariate effects
are strongly protective a proportional hazards model, whose exponential form
keeps the hazard positive by construction, is the safer choice.
"""

import warnings
from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
    MirroredDistributionAttrs,
    assemble_regression_model,
    make_objective,
    mirror_distribution,
    prepare_regression_fit,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from ..tvc_fit import TVCFitMixin


class _AdditiveReg:
    # Lightweight namespace for the fitted model's ``reg_model`` attribute;
    # the model repr reads ``.name``.
    name: str
    phi_param_map: object


class AdditiveHazardsFitter(
    MirroredDistributionAttrs,
    HazardIdentitiesMixin,
    TVCFitMixin,
    DataFrameRegressionMixin,
):
    """
    Parametric additive hazards fitter: the covariates add a constant
    risk difference to a parametric baseline hazard,

    .. math::
        h(x \\mid Z) = h_0(x) + \\beta' Z, \\qquad
        H(x \\mid Z) = H_0(x) + x\\, \\beta' Z.

    Use the pre-built instances (``WeibullAH``, ``ExponentialAH``, ...) or
    the ``AH`` factory. Nothing keeps the hazard positive except the
    likelihood itself, which needs ``log h`` at every observed failure:
    the fit keeps the hazard positive at the failures, and when a strongly
    protective covariate pushes it to that limit the fit ends on the
    boundary -- the hazard nearly zero at one failure, the baseline
    distorted -- and warns. A proportional hazards model, which keeps the
    hazard positive by construction, is then the safer choice.
    """

    def __init__(self, name: str, dist: Any) -> None:
        self.name = name
        mirror_distribution(self, dist)
        self.Hf_dist = dist.Hf
        self.hf_dist = dist.hf

    # -- covariate-aware distribution functions (x, Z, *params) -----------

    def _beta_Z(self, Z: Numeric, beta: "tuple[Boxable, ...]") -> Boxable:
        return np.dot(Z, np.array(beta))

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate :math:`h_0(x) + \\beta' Z` at ``x`` for covariates
        ``Z``; ``params`` are the distribution parameters followed by the
        coefficients.
        """
        dist_params = np.array(params[: self.k_dist])
        beta = params[self.k_dist :]
        return self.hf_dist(x, *dist_params) + self._beta_Z(Z, beta)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard :math:`H_0(x) + x \\beta' Z` at ``x`` for
        covariates ``Z``; ``params`` as for :meth:`hf`.
        """
        # H(x | Z) = H_0(x) + integral_0^x beta'Z ds = H_0(x) + x * beta'Z.
        dist_params = np.array(params[: self.k_dist])
        beta = params[self.k_dist :]
        return self.Hf_dist(x, *dist_params) + x * self._beta_Z(Z, beta)

    # sf/ff/df and the log identities come from HazardIdentitiesMixin;
    # when the additive hazard is driven non-positive its log_df is nan
    # and the optimiser rejects that point (see the module docstring).

    # mpp transforms are the identity (probability plotting is not used for
    # these models, but the interface is kept consistent with the other
    # regression fitters).
    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    def random(
        self, size: int, Z: npt.ArrayLike, *params: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` samples for each covariate row of ``Z`` by numerically
        inverting the (monotone) cumulative hazard. Requires the additive
        hazard to stay positive over the sampled range.

        Returns the draws and a 2-D array of the covariate row each was
        drawn at, row by row -- the same contract as the proportional
        hazards ``random``.
        """
        dist_params = np.array(params[: self.k_dist])
        beta = np.array(params[self.k_dist :])
        # One row per covariate vector, as in the PH sampler. This used to
        # ravel ``Z`` into a single vector, so several rows failed with a
        # shape mismatch and one row came back as a 1-D ``Z``.
        Z_arr = np.atleast_2d(np.asarray(Z, dtype=float))
        x = []
        Z_out = []
        for row in Z_arr:
            x.append(
                self._invert_cumulative_hazard(size, row, dist_params, beta)
            )
            Z_out.append(np.tile(row, (size, 1)))
        return np.concatenate(x), np.vstack(Z_out)

    def _invert_cumulative_hazard(
        self,
        size: int,
        row: npt.NDArray,
        dist_params: npt.NDArray,
        beta: npt.NDArray,
    ) -> npt.NDArray:
        """``size`` draws at one covariate row: the times at which
        ``H_0(x) + x beta'Z`` reaches ``-log U``, found by bracketing and
        bisection."""
        bz = float(np.dot(row, beta))
        target = -np.log(np.random.uniform(0, 1, size))

        def cum_haz(xv: npt.NDArray) -> npt.NDArray:
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
        return 0.5 * (lo + hi)

    # -- positivity boundary ------------------------------------------------

    #: The share of the information about ``beta`` a single failure must
    #: carry, and the fraction of its baseline hazard its hazard must be
    #: below, for the fit to count as held at the positivity boundary (see
    #: ``_warn_if_on_positivity_boundary``).
    BOUNDARY_INFORMATION_SHARE = 0.5
    BOUNDARY_HAZARD_FRACTION = 0.25

    def _warn_if_on_positivity_boundary(
        self, data: SurpyvalData, params: npt.NDArray
    ) -> None:
        """Warn when the fit is held at the positivity boundary.

        Each failure contributes ``log h`` to the likelihood, so a hazard
        driven towards zero at one of them is a barrier: with a strongly
        protective covariate the optimiser stops against it, with ``beta``
        pinned by the one failure whose hazard it may not cross and the
        baseline bent to compensate. That result has a finite likelihood
        and used to be returned silently.

        The barrier shows in the observed information. ``H`` is linear in
        ``beta``, so the information about ``beta`` is the sum over failures
        of ``Z Z' / h^2``; at an interior optimum it is spread over many
        failures, while at the barrier the failure whose hazard is nearly
        zero supplies most of it. In simulations boundary fits put well over
        half of it on one failure, and fits to genuinely additive data with
        a positive hazard a small fraction (below a third at a hundred
        failures, falling as the sample grows). In a handful of failures
        one of them can carry half the information anyway, so the failure
        must also have had most of its baseline hazard cancelled -- the
        boundary fits simulated kept at most a seventh of it.
        """
        event = np.asarray(data.c) == 0
        x = np.asarray(data.x)
        x = (x[:, 0] if x.ndim == 2 else x)[event]
        Z = np.asarray(data.Z)[event]
        with np.errstate(all="ignore"):
            h = np.asarray(self.hf(x, Z, *params), dtype=float)
            h0 = np.asarray(
                self.hf_dist(x, *params[: self.k_dist]), dtype=float
            )
            info = np.where(h > 0, (Z**2).sum(axis=1) / h**2, 0.0)
        # Only a failure whose hazard the covariates pushed well below the
        # baseline can be pressed against zero.
        protected = (h > 0) & (h < self.BOUNDARY_HAZARD_FRACTION * h0)
        total = info.sum()
        if not np.any(protected) or not np.isfinite(total) or total <= 0:
            return
        i = int(np.argmax(np.where(protected, info, -np.inf)))
        if info[i] / total > self.BOUNDARY_INFORMATION_SHARE:
            warnings.warn(
                "The additive hazards fit ended on the positivity boundary: "
                "the fitted hazard h_0(x) + beta'Z at the observed failure "
                "x = {:.4g} is {:.3g}, {:.2%} of the baseline hazard there, "
                "and that one failure carries {:.0%} of the information "
                "about beta. The covariate effect is too protective for the "
                "additive model to fit without the hazard nearly vanishing, "
                "so beta sits at the boundary and the baseline is "
                "distorted. A proportional hazards model (e.g. {}PH) keeps "
                "the hazard positive by construction.".format(
                    x[i], h[i], h[i] / h0[i], info[i] / total, self.dist.name
                ),
                stacklevel=3,
            )

    # -- factory ----------------------------------------------------------

    @staticmethod
    def create(distribution: Any) -> "AdditiveHazardsFitter":
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
            The covariate matrix (one row per observation). Rows with a
            missing or infinite covariate are dropped, with a warning.
        c : array_like, optional
            The censoring indicators (0 observed, 1 right, -1 left, 2
            interval).
        n : array_like, optional
            The count of observations at each time.
        t : array_like, optional
            Truncation bounds: an (N, 2) array of the left and right
            truncation times of each observation.
        init : array_like, optional
            Initial parameter values (baseline parameters followed by the
            covariate coefficients).
        fixed : dict, optional
            Parameters to hold fixed, by name.

        Returns
        -------

        ParametricRegressionModel
            The fitted model. Warns if the fit ends on the positivity
            boundary (the hazard nearly zero at an observed failure), and
            raises if the optimiser cannot reach a finite likelihood.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullAH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> c = np.zeros(100)
        >>> model = WeibullAH.fit(x=x, Z=Z, c=c)
        >>> model.params.round(3)
        array([9.332, 1.851, 0.086])
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
        )
        init, bounds, pmap, transform, inv_trans, const, fixed = prep

        with np.errstate(all="ignore"):

            true_neg_ll = make_objective(self, data, inv_trans, const)

            def fun(params: npt.NDArray) -> Boxable:
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
        self._warn_if_on_positivity_boundary(data, params)

        reg_model = _AdditiveReg()
        reg_model.name = "Additive [beta'Z]"
        reg_model.phi_param_map = pmap

        return assemble_regression_model(
            self,
            "Additive Hazard",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
            neg_ll=final_neg_ll,
        )
