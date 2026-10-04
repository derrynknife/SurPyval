import warnings
from typing import Any, Callable

import numpy as np

# The finite-difference Hessian and the bound-sign helper live in
# ``surpyval.utils.linalg`` -- shared with the parametric-regression
# bounds machinery, which used to carry verbatim copies of them (the
# drift-prone pattern that produced #288).
from surpyval.univariate.information_criteria import (
    corrected_aic,
    ic_sample_size,
)
from surpyval.utils.linalg import numerical_hessian, wald_bound_on_support
from surpyval.utils.validation import alpha_ci_error
from surpyval.utils.warnings import warn_no_covariance


def check_alpha_ci(alpha_ci: Any) -> None:
    """
    Refuse an ``alpha_ci`` outside (0, 1) in a recurrent bound method
    (``cif_cb``, ``iif_cb``, ``mtbf_cb``, ``mcf_cb``, ``param_cb`` and the
    plots that draw them), with the package's one message. Outside (0, 1)
    the bounds came back reversed (``alpha_ci=1.5``), equal (1) or NaN
    (below 0), silently (#647).
    """
    if not (
        isinstance(alpha_ci, (int, float, np.integer, np.floating))
        and not isinstance(alpha_ci, bool)
        and 0 < alpha_ci < 1
    ):
        raise alpha_ci_error(alpha_ci)


def bic_sample_size(data: Any) -> float:
    """
    The sample size ``n`` of every recurrent model's BIC: the number of
    observed events -- exactly observed (``c=0``), left-censored (``c=-1``,
    events before ``x``) and interval-censored (``c=2``, events within
    ``[xl, xr]``), each row weighted by the number of events ``n`` it
    holds -- or, with none, the (weighted) number of rows. End-of-
    observation (``c=1``) rows are not events. This is the rule of every
    SurPyval BIC (:func:`surpyval.univariate.information_criteria.
    ic_sample_size`); the recurrent models used to count exact events only,
    and returned NaN without one.
    """
    return ic_sample_size(data.c, data.n)


def require_data(model: Any, what: str) -> None:
    """
    Raise an informative ``ValueError`` when ``model`` carries no data.

    Only a model fitted from data keeps it: one built from parameters
    (``from_params`` / ``fit_from_parameters``) never had any, and one
    restored with ``from_dict`` / ``from_json`` does not store it. Every
    method that reads the fitted data (residuals, trend tests, goodness of
    fit, plots against the data) calls this first, so none of them fails
    with a bare ``AttributeError``.
    """
    if getattr(model, "data", None) is None:
        raise ValueError(
            "{} requires a model fitted from data; models built from "
            "parameters (from_params / fit_from_parameters) or restored "
            "with from_dict / from_json carry no data.".format(what)
        )


class LikelihoodInferenceMixin:
    """
    Likelihood-based inference for fitted recurrent-event models.

    The fitting routine must set ``_neg_ll`` (the negative log-likelihood in
    natural parameter space), ``_mle`` (the fitted parameter vector in that
    same space) and ``_n_obs`` (BIC's sample size, from
    :func:`bic_sample_size`). Models built with
    ``fit_from_parameters`` (or by a non-likelihood method such as MSE) carry
    no likelihood and these methods raise.

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

    A ``nan`` entry of ``_mle`` is a parameter that was not estimated: an
    aliased coefficient of a proportional-intensity regression (#502). It
    enters the likelihood as 0, is not counted in AIC and BIC, and has no
    variance (``nan``); the information is that of the other parameters.
    """

    # Supplied by the fitting routine (see the class docstring); declared
    # here so the checker knows their types on the host class.
    _neg_ll: Callable
    _mle: np.ndarray
    _n_obs: float
    _fitter: Any

    def _check_fitted(self) -> None:
        if not hasattr(self, "_neg_ll"):
            if getattr(self, "how", None) == "MSE":
                reason = (
                    "a how='MSE' fit minimises squared error on the MCF "
                    "and has no likelihood; refit with how='MLE'."
                )
            elif getattr(self, "how", None) == "MLE":
                # A fit restored with from_dict / from_json still says how
                # it was fitted (#663).
                reason = (
                    "this model was restored with from_dict / from_json, "
                    "and restored models carry no data, so no likelihood; "
                    "refit it to the data for bounds and standard errors."
                )
            else:
                reason = (
                    "models built from parameters (from_params or "
                    "fit_from_parameters) or restored with from_dict / "
                    "from_json carry no data, so no likelihood."
                )
            raise ValueError(
                "Likelihood inference is only available for models fitted "
                "from data by maximum likelihood: " + reason
            )

    def _check_has_data(self, what: str) -> None:
        require_data(self, what)

    def _estimated(self) -> np.ndarray:
        """Which entries of ``_mle`` were estimated: all but an aliased
        coefficient's ``nan`` (#502)."""
        return ~np.isnan(np.asarray(self._mle, dtype=float))

    def _mle_values(self) -> np.ndarray:
        """``_mle`` with a parameter that was not estimated as 0, the value
        the likelihood takes it at."""
        mle = np.asarray(self._mle, dtype=float)
        return np.where(self._estimated(), mle, 0.0)

    def _parameter_names(self) -> list:
        """
        Names of the entries of ``_mle``, in order. Subclasses override this to
        label their parameters (e.g. the renewal models prepend the restoration
        parameter, the regression models append the covariate coefficients).
        """
        raise NotImplementedError

    @property
    def parameter_names(self) -> list:
        """
        The names of the model's parameters, in the order used by
        :meth:`covariance`, :meth:`standard_errors` and :meth:`param_cb`.
        A model built from parameters has them too.
        """
        return list(self._parameter_names())

    @property
    def log_likelihood(self) -> float:
        """
        The maximised log-likelihood of the fit. Raises ``ValueError`` for a
        model with no likelihood (built from parameters, or fitted by MSE).
        """
        self._check_fitted()
        return -float(self._neg_ll(self._mle_values()))

    def neg_ll(self) -> float:
        """
        The negative of the maximised log-likelihood, ``-log_likelihood``,
        as on every other fitted model (#572).
        """
        return -self.log_likelihood

    def aic(self) -> float:
        """
        Akaike's information criterion, :math:`2k - 2\\ln L`, with ``k`` the
        number of fitted parameters. Lower is better. Call it,
        ``model.aic()``, as on every other fitted model.

        .. versionchanged:: 0.23
           ``aic`` is a method, as on every other model (#572); it was a
           property.
        """
        self._check_fitted()
        k = int(self._estimated().sum())
        return float(2.0 * k - 2.0 * self.log_likelihood)

    def bic(self) -> float:
        """
        The Bayesian information criterion, :math:`k \\ln n - 2\\ln L`,
        with ``n`` the number of observed events the model was fitted to:
        exact, left- and interval-censored, the last two adding the
        number of events they hold. End-of-observation rows do not add to
        it, and with no observed event it is the number of rows -- the
        rule of BIC everywhere in SurPyval (see :func:`bic_sample_size`).
        Lower is better. Call it, ``model.bic()``, as on every other
        fitted model.

        .. versionchanged:: 0.23
           ``bic`` is a method, as on every other model (#572); it was a
           property.
        """
        self._check_fitted()
        k = int(self._estimated().sum())
        return float(k * np.log(self._n_obs) - 2.0 * self.log_likelihood)

    def aic_c(self) -> float:
        """
        The small-sample corrected AIC, ``aic() + (2k^2 + 2k) / (n - k -
        1)``, with the ``k`` of :meth:`aic` and the ``n`` of :meth:`bic`;
        ``nan`` where ``n <= k + 1`` (``corrected_aic``), as on every
        other model (#605).
        """
        self._check_fitted()
        k = int(self._estimated().sum())
        return corrected_aic(self.aic(), k, self._n_obs)

    def covariance(self) -> np.ndarray:
        """
        Approximate parameter covariance matrix, ordered to match
        :attr:`parameter_names`. Computed as the inverse of the numerical
        Hessian of the negative log-likelihood at the MLE. A parameter
        that was not estimated (an aliased coefficient, #502) has a ``nan``
        row and column.
        """
        self._check_fitted()
        full = self._mle_values()
        n = full.size
        free = self._estimated()
        if free.all():
            H = numerical_hessian(self._neg_ll, full)
        else:

            def neg_ll_free(values: np.ndarray) -> float:
                params = full.copy()
                params[free] = values
                return self._neg_ll(params)

            H = numerical_hessian(neg_ll_free, full[free])
        out = np.full((n, n), np.nan)
        if not np.all(np.isfinite(H)):
            warn_no_covariance()
            return out
        try:
            out[np.ix_(free, free)] = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            warn_no_covariance()
        return out

    def standard_errors(self) -> np.ndarray:
        """
        Standard errors of the fitted parameters (the square roots of the
        diagonal of :meth:`covariance`), ordered to match
        :attr:`parameter_names`. Entries are NaN where the variance is
        non-positive, which typically indicates a boundary optimum, and for
        a parameter that was not estimated (an aliased coefficient).
        """
        var = np.diag(self.covariance())
        with np.errstate(invalid="ignore"):
            se = np.sqrt(var)
        if np.any(~(var > 0) & self._estimated()):
            warnings.warn(
                "Some parameter variances are non-positive (the optimum may "
                "be at a boundary); their standard errors are NaN."
            )
        return se

    def _parameter_bounds(self) -> list:
        """
        Natural-space ``(lower, upper)`` bounds for each entry of ``_mle``,
        ordered to match :attr:`parameter_names`. Subclasses override this so
        :meth:`param_cb` can pick a transform that keeps the confidence bounds
        inside the parameter's support; the default is unbounded.
        """
        return [(None, None)] * self._mle.size

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
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
        check_alpha_ci(alpha_ci)
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
        return wald_bound_on_support(
            p_hat, var, lower, upper, alpha_ci, bound, name=name
        )
