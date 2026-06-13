import warnings

import numpy as np


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


class RenewalInferenceMixin:
    """
    Likelihood-based inference for fitted renewal models.

    The fitting routine must set ``_neg_ll`` (the negative log-likelihood in
    natural parameter space), ``_mle`` (the fitted ``[q, *dist_params]``
    vector) and ``_n_obs`` (the number of events contributing to the
    likelihood). Models built with ``fit_from_parameters`` carry no likelihood
    and these methods raise.

    Standard errors come from inverting a numerical Hessian of the negative
    log-likelihood (the observed Fisher information). When a parameter sits on
    a boundary (notably the restoration factor ``q`` driven to 0) the
    asymptotic normal approximation does not hold and the corresponding
    standard error is returned as NaN with a warning.
    """

    def _check_fitted(self):
        if not hasattr(self, "_neg_ll"):
            raise ValueError(
                "Inference is only available for models fitted from data; "
                "fit_from_parameters does not compute a likelihood."
            )

    @property
    def parameter_names(self):
        self._check_fitted()
        return ["q", *self.model.dist.param_names]

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
        Approximate parameter covariance matrix, ordered as
        ``[q, *dist_params]``. Computed as the inverse of the numerical Hessian
        of the negative log-likelihood at the MLE.
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
        Standard errors of ``[q, *dist_params]`` (the square roots of the
        diagonal of :meth:`covariance`). Entries are NaN where the variance is
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
