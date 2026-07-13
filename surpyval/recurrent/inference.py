import warnings

import numpy as np
from scipy.stats import norm


def numerical_hessian(func, x):
    """
    Central finite-difference Hessian of a scalar ``func`` at ``x``. Used to
    approximate the observed Fisher information from the negative
    log-likelihood of the renewal models, which are fitted with a derivative
    -free optimiser.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    # Step scaled to each parameter; cube-root of machine epsilon is the usual
    # choice for a second-derivative central difference.
    step = (np.finfo(float).eps ** (1.0 / 3.0)) * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ei[i] = step[i]
            ej = np.zeros(n)
            ej[j] = step[j]
            H[i, j] = H[j, i] = (
                func(x + ei + ej)
                - func(x + ei - ej)
                - func(x - ei + ej)
                + func(x - ei - ej)
            ) / (4.0 * step[i] * step[j])
    return H


def delta_method_std_errors(func, mle, cov):
    """
    Standard errors of the (possibly vector-valued) function ``func`` of the
    parameters, evaluated at the MLE, via the delta method with a
    central-difference Jacobian: ``se_i = sqrt(J_i' cov J_i)``.
    """
    mle = np.asarray(mle, dtype=float)
    step = (np.finfo(float).eps ** (1.0 / 3.0)) * np.maximum(np.abs(mle), 1e-2)
    cols = []
    for i in range(mle.size):
        ei = np.zeros(mle.size)
        ei[i] = step[i]
        cols.append(
            (
                np.asarray(func(mle + ei), dtype=float)
                - np.asarray(func(mle - ei), dtype=float)
            )
            / (2.0 * step[i])
        )
    J = np.stack(cols, axis=-1)
    var = np.einsum("...i,ij,...j->...", J, cov, J)
    with np.errstate(invalid="ignore"):
        return np.sqrt(var)


def _bound_signs(alpha_ci, bound):
    """
    The one-sided tail probability and the signs of the normal quantile for
    each requested bound: ``[-1, 1]`` (lower, upper) for two-sided bounds,
    a single sign otherwise.
    """
    if bound == "two-sided":
        return alpha_ci / 2.0, np.array([-1.0, 1.0])
    elif bound == "lower":
        return alpha_ci, np.array([-1.0])
    elif bound == "upper":
        return alpha_ci, np.array([1.0])
    raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")


def log_transformed_cb(estimate, se, alpha_ci=0.05, bound="two-sided"):
    """
    Log-transformed normal confidence bounds ``est * exp(+/- z * se / est)``
    for a positive curve (the same construction as the exponential Greenwood
    bounds on the nonparametric MCF). Where the estimate is zero (e.g. a CIF
    at ``x = 0``) both bounds are zero.

    For two-sided bounds the last axis holds ``[lower, upper]``; one-sided
    bounds are returned with the shape of ``estimate``.
    """
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    alpha, signs = _bound_signs(alpha_ci, bound)
    z = norm.ppf(1.0 - alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(estimate > 0, se / estimate, 0.0)
    cb = estimate[..., None] * np.exp(signs * z * ratio[..., None])
    if bound == "two-sided":
        return cb
    return cb[..., 0]


class LikelihoodInferenceMixin:
    """
    Likelihood-based inference for fitted recurrent-event models.

    The fitting routine must set ``_neg_ll`` (the negative log-likelihood in
    natural parameter space), ``_mle`` (the fitted parameter vector in that
    same space) and ``_n_obs`` (the number of events contributing to the
    likelihood). Models built with ``fit_from_parameters`` (or by a non-
    likelihood method such as MSE) carry no likelihood and these methods raise.

    This is shared by every fitted recurrent model that has a likelihood: the
    renewal / imperfect-repair models (``RenewalModel``), the parametric
    intensity models (``ParametricRecurrenceModel``) and the proportional-
    intensity regression models (``ProportionalIntensityModel``). Only the
    parameter labelling differs between them, so subclasses supply
    :meth:`_parameter_names`; everything numeric (log-likelihood, AIC, BIC,
    covariance, standard errors) is computed here from ``_neg_ll``/``_mle``/
    ``_n_obs`` alone.

    Standard errors come from inverting a numerical Hessian of the negative
    log-likelihood (the observed Fisher information). When a parameter sits on
    a boundary (e.g. a repair parameter driven to its limit) the asymptotic
    normal approximation does not hold and the corresponding standard error is
    returned as NaN with a warning.
    """

    def _check_fitted(self):
        if not hasattr(self, "_neg_ll"):
            raise ValueError(
                "Inference is only available for models fitted from data; "
                "fit_from_parameters does not compute a likelihood."
            )

    def _parameter_names(self):
        """
        Names of the entries of ``_mle``, in order. Subclasses override this to
        label their parameters (e.g. the renewal models prepend the restoration
        parameter, the regression models append the covariate coefficients).
        """
        raise NotImplementedError

    @property
    def parameter_names(self):
        self._check_fitted()
        return list(self._parameter_names())

    @property
    def log_likelihood(self):
        self._check_fitted()
        return -float(self._neg_ll(self._mle))

    @property
    def aic(self):
        self._check_fitted()
        k = self._mle.size
        return 2.0 * k - 2.0 * self.log_likelihood

    @property
    def bic(self):
        self._check_fitted()
        k = self._mle.size
        return k * np.log(self._n_obs) - 2.0 * self.log_likelihood

    def covariance(self):
        """
        Approximate parameter covariance matrix, ordered to match
        :attr:`parameter_names`. Computed as the inverse of the numerical
        Hessian of the negative log-likelihood at the MLE.
        """
        self._check_fitted()
        H = numerical_hessian(self._neg_ll, self._mle)
        n = self._mle.size
        if not np.all(np.isfinite(H)):
            warnings.warn(
                "Hessian could not be evaluated (the optimum may be at a "
                "parameter boundary); covariance is unavailable."
            )
            return np.full((n, n), np.nan)
        try:
            return np.linalg.inv(H)
        except np.linalg.LinAlgError:
            warnings.warn("Hessian is singular; covariance is unavailable.")
            return np.full((n, n), np.nan)

    def standard_errors(self):
        """
        Standard errors of the fitted parameters (the square roots of the
        diagonal of :meth:`covariance`), ordered to match
        :attr:`parameter_names`. Entries are NaN where the variance is
        non-positive, which typically indicates a boundary optimum.
        """
        var = np.diag(self.covariance())
        with np.errstate(invalid="ignore"):
            se = np.sqrt(var)
        if np.any(~(var > 0)):
            warnings.warn(
                "Some parameter variances are non-positive (the optimum may "
                "be at a boundary); their standard errors are NaN."
            )
        return se

    def _parameter_bounds(self):
        """
        Natural-space ``(lower, upper)`` bounds for each entry of ``_mle``,
        ordered to match :attr:`parameter_names`. Subclasses override this so
        :meth:`param_cb` can pick a transform that keeps the confidence bounds
        inside the parameter's support; the default is unbounded.
        """
        return [(None, None)] * self._mle.size

    def param_cb(self, name, alpha_ci=0.05, bound="two-sided"):
        """
        Confidence bound(s) on a fitted parameter, mirroring the univariate
        ``Parametric.param_cb`` API.

        Wald bounds from the observed information, computed on a transformed
        scale chosen from the parameter's bounds so the result respects its
        support: log scale for one-sided-bounded parameters (e.g. a positive
        rate), logit scale for interval-bounded parameters (e.g. a repair
        efficiency in ``(0, 1)``), and the natural scale for unbounded ones.

        Parameters
        ----------

        name : str
            The parameter to bound; one of :attr:`parameter_names`.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.

        Returns
        -------

        numpy array
            The confidence bound(s) on the parameter.
        """
        self._check_fitted()
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        idx = names.index(name)
        p_hat = float(self._mle[idx])
        var = float(self.covariance()[idx, idx])
        lower, upper = self._parameter_bounds()[idx]

        alpha, signs = _bound_signs(alpha_ci, bound)
        offsets = signs * norm.ppf(1.0 - alpha) * np.sqrt(var)

        if lower is not None and upper is not None:
            # Bounds on the generalised logit keep the result in (lower,
            # upper).
            width = upper - lower
            frac = (p_hat - lower) / width
            u_hat = np.log(frac / (1.0 - frac))
            du = offsets / (width * frac * (1.0 - frac))
            return lower + width / (1.0 + np.exp(-(u_hat + du)))
        elif lower is not None:
            # Bounds on log(p - lower) keep the result above ``lower``.
            return lower + (p_hat - lower) * np.exp(offsets / (p_hat - lower))
        elif upper is not None:
            # Bounds on log(upper - p) keep the result below ``upper``.
            return upper - (upper - p_hat) * np.exp(-offsets / (upper - p_hat))
        return p_hat + offsets
