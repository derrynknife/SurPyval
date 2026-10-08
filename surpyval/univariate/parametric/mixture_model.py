from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from autograd import grad, hessian, jacobian, value_and_grad
from autograd.scipy.special import logsumexp as ag_logsumexp
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from scipy.special import ndtri as z

from surpyval import Distribution
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.univariate.parametric.fitters import OPTIMUM_GTOL
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import UnivariateDataFrameMixin
from surpyval.utils.deprecation import MadePrivate
from surpyval.utils.linalg import (
    bound_signs,
    safe_inv,
    sf_link_bound,
    standard_errors_of,
    wald_bound_on_support,
    warn_wald_undefined,
)
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.removed_names import (
    MIXTURE_EM_ATTRIBUTES,
    removed_arguments,
    removed_attributes,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import (
    BOUNDS,
    CB_ON,
    check_alpha_ci,
    check_option,
    no_covariance_error,
    option_error,
    warn_outside_unit_interval,
)
from surpyval.utils.warnings import caller_stacklevel

from .probability_plotting import (
    adjust_heuristic,
    draw_probability_plot,
    probability_plot_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# The log-likelihood floor of one observation under one component: far
# below any log-likelihood an observation the component can explain has,
# and finite, so the EM objective stays finite (see
# ``MixtureModel._component_log_likelihood``).
LOG_FLOOR = -1e4


#: The values of ``MixtureModel.fit``'s ``em`` option.
EM_METHODS = ("plain", "squarem")

# What a fit keeps while it runs, and drops when it ends.
_FIT_CACHES = ("_Q_jac_cache", "_Q_value_and_grad_cache", "_log_resp_cache")


class _NonFiniteGradient(ArithmeticError):
    """An autograd gradient that is not finite (see ``_finite_gradient``)."""


def _finite_gradient(jac: Callable[..., Any]) -> Callable[..., Any]:
    """``jac``, raising ``_NonFiniteGradient`` where it is not finite, which
    L-BFGS-B would otherwise follow to the bounds."""

    def checked(x: npt.NDArray) -> npt.NDArray:
        g = np.asarray(jac(x), dtype=float)
        if not np.all(np.isfinite(g)):
            raise _NonFiniteGradient
        return g

    return checked


class _FitMethod:
    """``MixtureModel.fit`` as both an instance and a class method (#482).

    On a model (``MixtureModel(dist, m).fit(x)``) it fits in place, as it
    always has, and returns the model. On the class
    (``MixtureModel.fit(x, dist=Weibull, m=2)``) it builds the model from
    ``dist`` and ``m`` and fits it, the ``Dist.fit(x)`` form of every other
    fitter.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        functools.update_wrapper(self, func)  # type: ignore[arg-type]

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is not None:
            return functools.partial(self.func, obj)
        func = self.func

        @functools.wraps(func)
        def fit(
            x: npt.ArrayLike | None = None,
            c: npt.ArrayLike | None = None,
            n: npt.ArrayLike | None = None,
            t: npt.ArrayLike | None = None,
            tl: npt.ArrayLike | None = None,
            tr: npt.ArrayLike | None = None,
            xl: npt.ArrayLike | None = None,
            xr: npt.ArrayLike | None = None,
            *,
            dist: Any = None,
            m: int = 2,
            em: str = "plain",
        ) -> Any:
            if isinstance(x, objtype):
                # ``MixtureModel.fit(model, x, ...)``: the unbound call of
                # the instance method, which worked before #482.
                return func(x, c, n, t, tl, tr, xl, xr, em=em)
            if dist is None:
                raise ValueError(
                    "MixtureModel.fit needs `dist`, the distribution of "
                    "every component, e.g. "
                    "MixtureModel.fit(x, dist=surpyval.Weibull, m=2)"
                )
            return func(
                objtype(dist=dist, m=m), x, c, n, t, tl, tr, xl, xr, em=em
            )

        # Keep the docstring but show this signature (with ``dist`` and
        # ``m``), not the instance method's.
        del fit.__wrapped__
        return fit


class MixtureModel(
    InformationCriteriaMixin,
    UnivariateDataFrameMixin,
    SerialisableMixin,
    Distribution,
):
    """
    A class for creating a Mixture Model fitter.

    This class implements a Mixture Model, which is a probabilistic model that
    combines multiple probability distributions to model complex data. Models
    can be fit with either the Expectation Maximisation (EM) algorithm or
    Maximum Likelihood Estimation (MLE). The EM algorithm is the default and
    is based on the paper found `here \
    <https://www.sciencedirect.com/science/article/pii/S0307904X12002545>`__.

    Parameters
    ----------

    dist : surpyval distribution
        The distribution to be used in the mixture model. Must be a
        surpyval distribution.

    m : int, optional
        The number of sub-distributions to be used in the mixture model.
        Defaults to 2.

    Examples
    --------
    Fit in one call, like any other fitter:

    >>> import surpyval as surv
    >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
    >>> wmm.w.round(3)
    array([0.618, 0.382])

    or build the (unfitted) model first and fit it, which returns the
    same model:

    >>> surv.MixtureModel(dist=surv.Weibull, m=2)
    Unfitted Parametric Mixture SurPyval Model (Weibull, m = 2)
    """

    # How ``fit`` runs the EM iterations (its ``em`` option), and
    # whether the M-step is solved to full precision (SQUAREM's).
    _em_method = "plain"
    _exact_m_step = False

    # The EM responsibilities (one row per component, one column per
    # observation) are internal to the fit, as the EM steps are (#605); the
    # public name ``p`` means a distribution's own parameter elsewhere, and
    # ``lfp_p`` the limited-failure proportion (#626).
    p = MadePrivate("_resp")

    @property
    def parameter_names(self) -> list[str]:
        """The names of the columns of ``params``: the component
        distribution's ``parameter_names``. ``params`` has one row per
        component (``m`` rows); the weights are ``w``."""
        return list(self.dist.parameter_names)

    def __init__(self, dist: Any, m: int = 2) -> None:
        self.m = m
        self.dist = dist
        # These are None until ``fit`` runs and arrays afterwards, so the
        # honest annotation is the union -- and every use is downstream of
        # a fit. Narrowing them to the fitted type would be a lie before
        # the fit; narrowing to Optional would need an assert at each of
        # the thirty-odd uses without making anything safer, because
        # calling a predict method on an unfitted model is a contract
        # error the AttributeError already reports.
        self.data: Any = None
        self.params: Any = None
        self.w: Any = None
        self._resp: Any = None
        # The observed-data negative log-likelihood at the current
        # parameters, which the EM iteration tracks: the fitted one after
        # a fit (``neg_ll()``, ``log_likelihood``).
        self._neg_ll: Any = None
        #: What the fit reached, one of ``MAXIMUM_STATES``
        #: (``surpyval.utils.no_maximum``), as its warnings say:
        #: ``"verified"`` (a zero gradient and a positive-definite Hessian),
        #: ``"unverified"`` or ``"no finite maximum"`` (a component collapsed
        #: onto a point mass); ``"unknown"`` before a fit, or for a model
        #: restored from a dict saved without it.
        self.maximum: str = "unknown"

    # -- model comparison (#572) --------------------------------------------

    def _ic_k(self) -> int:
        """The number of free parameters: ``k`` per component and the
        ``m - 1`` free weights (they sum to one)."""
        return int(self.m * self.dist.k + self.m - 1)

    def _ic_sample_size_from_data(self) -> float:
        if self.data is None:
            raise ValueError("Must have been fit with data")
        return ic_sample_size(self.data.c, self.data.n)

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood of the fit, ``-neg_ll()``, as on
        a parametric model.

        .. versionchanged:: 0.23
           It was a method, ``log_likelihood(params)``, giving one
           component's log-likelihood of each observation.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> round(wmm.log_likelihood, 4)
        -48.7105
        >>> round(wmm.aic(), 4), wmm.aic() == 2 * 5 - 2 * wmm.log_likelihood
        (107.4211, True)
        """
        return float(-self.neg_ll())

    @log_likelihood.setter
    def log_likelihood(self, value: float) -> None:
        # As every model's: it records the negative log-likelihood.
        mixin: Any = InformationCriteriaMixin
        mixin.log_likelihood.fset(self, value)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted mixture model to a plain, JSON-serialisable dict.

        Stores the base distribution's name, the number of components ``m``,
        the per-component parameters and the mixing weights, so the reloaded
        model reproduces ``sf``/``ff``/``df``/``mean``/``random`` exactly. The
        fitted data and EM responsibilities are not stored; the fitted
        negative log-likelihood and the sample size of the information
        criteria are, so the restored model's :meth:`neg_ll`,
        ``log_likelihood``, :meth:`aic`, :meth:`aic_c` and :meth:`bic` are
        the fitted model's; and so is the parameters' covariance
        (``"theta_covariance"``, in the coordinates :meth:`covariance`
        describes), where the fit has one, so the restored model's
        :meth:`standard_errors`, :meth:`param_cb`, :meth:`cb` and
        :meth:`quantile_cb` are the fitted model's too.
        """
        from .parametric import is_custom_distribution

        out = {
            "model": "MixtureModel",
            "dist": self.dist.name,
            "m": int(self.m),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "w": np.asarray(self.w, dtype=float).tolist(),
            **maximum_entry(self.maximum),
        }
        # What the information criteria need, as ``Parametric`` stores it
        if self._neg_ll is not None:
            out["_neg_ll"] = float(self._neg_ll)
        ic_n = self._ic_sample_size_or_none()
        if ic_n is not None:
            out["ic_n"] = ic_n
        # The covariance of the search coordinates, so the restored
        # model's standard errors and Wald bounds are the fitted model's
        try:
            theta_cov = self._theta_covariance()
        except ValueError:
            pass
        else:
            out["theta_covariance"] = np.asarray(theta_cov, float).tolist()
        if is_custom_distribution(self.dist):
            # Resolved through the CustomDistribution registry on reading
            out["custom"] = True
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "MixtureModel":
        """Rebuild a mixture model from a :meth:`to_dict` dictionary.

        The restored model evaluates the mixture (``sf``, ``ff``, ``df``,
        ``hf``, ``qf``, ``cs``, ``mean``, ``random``) exactly, and has the
        fitted model's covariance where the dict stores it, but holds no
        data, so :meth:`plot` and :meth:`get_plot_data` raise. A mixture of a
        ``CustomDistribution`` or a ``Discretize`` distribution is read
        back as described in ``Parametric.from_dict``.
        """
        from .parametric import resolve_distribution

        require_model_tag(model_dict, "MixtureModel", "a mixture model")
        dist = resolve_distribution(
            model_dict["dist"], bool(model_dict.get("custom", False))
        )
        out = cls(dist=dist, m=int(model_dict["m"]))
        out.params = np.array(model_dict["params"], dtype=float)
        out.w = np.array(model_dict["w"], dtype=float)
        out.maximum = restored_maximum(model_dict)
        if "_neg_ll" in model_dict:
            out._neg_ll = float(model_dict["_neg_ll"])
        out._ic_n = cls._restored_ic_n(model_dict)
        if "theta_covariance" in model_dict:
            out._theta_cov = np.array(
                model_dict["theta_covariance"], dtype=float
            )
        return out

    def __repr__(self) -> str:
        if self.params is not None:
            param_string = "\n".join(
                [
                    f"{name:>10}: {p}"
                    for p, name in zip(
                        self.params.T, self.dist.parameter_names
                    )
                ]
            )
            weight_string = ",\n\t".join([str(w) for w in self.w])
            # Truncated data is fitted by direct maximisation, not EM; a
            # model restored by ``from_dict`` does not know how it was fit.
            fitted_by = {True: "MLE", False: "EM", None: "-"}[
                getattr(self, "_truncated", None)
            ]
            out = (
                "Parametric Mixture SurPyval Model"
                "\n================================="
                f"\nDistribution        : {self.dist.name}"
                f"\nSub-Distributions   : {self.m}"
                f"\nFitted by           : {fitted_by}"
                f"{self._data_repr()}"
                f"\nWeights             : \n\t{weight_string}"
                f"\nParameters          :\n{param_string}"
            )

            return out
        else:
            return (
                "Unfitted Parametric Mixture SurPyval Model "
                f"({self.dist.name}, m = {self.m})"
            )

    def _data_repr(self) -> str:
        """The "Data" line of the printout (#508): the units the model was
        fitted to, weighted by ``n``, by kind of censoring and truncation;
        nothing for a model restored without its data."""
        data = self.data
        if data is None or getattr(data, "c", None) is None:
            return ""
        lower, upper = self.dist.support
        return "\nData                : " + data_summary(
            data.c, data.n, data.tl, data.tr, lower, upper, x=data.x
        )

    def _likelihood(self, params: Any) -> Any:
        """Per-observation likelihood of one component (no count powers:
        counts ``n`` enter the log-likelihood as multipliers -- raising the
        per-component likelihood to ``n`` *before* mixing is wrong, since
        ``sum_i w_i f_i^n != (sum_i w_i f_i)^n`` (#254)."""
        self._require_fit_data("_likelihood()")
        data = self.data
        like_o = self.dist.df(data.x_o, *params)
        like_r = self.dist.sf(data.x_r, *params)
        like_l = self.dist.ff(data.x_l, *params)
        like_i = self.dist.ff(data.x_ir, *params) - self.dist.ff(
            data.x_il, *params
        )
        like = np.zeros(len(self.data.x))
        # The observation masks, not the raw codes: a censored row with a
        # finite truncation bound on its censored side is an interval
        # (#310, #544).
        like[data.mask_o] = like_o
        like[data.mask_r] = like_r
        like[data.mask_l] = like_l
        like[data.mask_i] = like_i
        return like

    def _component_log_likelihood(self, params: Any) -> Any:
        """Per-observation log-likelihood of one component, floored at
        ``LOG_FLOOR``.

        Formed from the distribution's log functions rather than as the
        log of :meth:`_likelihood`: a density or interval probability that
        underflows to 0 made ``log`` return -inf, a responsibility times
        -inf made the M-step objective infinite, and the optimiser
        stopped after one step (a two-Weibull mixture on interval data
        stalled 250 log-likelihood units short, with no warning). The
        floor keeps an observation a component cannot explain at a finite,
        heavily penalised value instead.
        """
        self._require_fit_data("_component_log_likelihood()")
        data = self.data
        dist = self.dist
        # Each kind of row in one piece, put back in the rows' order by
        # indexing rather than assignment, so that autograd can
        # differentiate it (the EM's M-step and the polish take its
        # gradient, #506). The kinds are the observation masks, not the raw
        # codes: a right-censored row with a finite ``tr`` is the interval
        # [x, tr] and a left-censored row with a finite ``tl`` is [tl, x]
        # (#310), and grouping by code lost them (#544).
        pieces = []
        with np.errstate(all="ignore"):
            if data.mask_o.any():
                pieces.append(dist.log_df(data.x_o, *params))
            if data.mask_r.any():
                pieces.append(dist.log_sf(data.x_r, *params))
            if data.mask_l.any():
                pieces.append(dist.log_ff(data.x_l, *params))
            if data.mask_i.any():
                window = dist.ff(data.x_ir, *params) - self._ff_lower(params)
                positive = window > 0
                pieces.append(
                    np.where(
                        positive,
                        np.log(np.where(positive, window, 1.0)),
                        LOG_FLOOR,
                    )
                )
            out = np.concatenate(pieces)[self._row_order()]
            out = np.where(np.isnan(out), LOG_FLOOR, out)
        return np.maximum(out, LOG_FLOOR)

    def _ff_lower(self, params: Any) -> Any:
        """One component's CDF at the interval rows' lower ends, exactly 0
        at or below the support's lower edge without evaluating it there.

        The value is the same, but the derivative is not: a Weibull's
        ``(0 / alpha) ** beta`` has a NaN gradient in ``alpha`` for
        ``beta < 1`` (``inf * 0``), so with one ``[0, 1]`` interval row the
        M-step fell back to finite differences and the polish stopped
        after one evaluation, unable to verify (or reach) the maximum
        (#582). Those rows are evaluated at their upper end instead (a
        point inside the support, whose value is discarded), as
        ``ParametricFitter.ll_interval_or_truncated`` does.
        """
        data = self.data
        lower = float(self.dist.support[0])
        if np.isnan(lower):
            return self.dist.ff(data.x_il, *params)
        inside = data.x_il > lower
        if inside.all():
            return self.dist.ff(data.x_il, *params)
        safe = np.where(inside, data.x_il, data.x_ir)
        return np.where(inside, self.dist.ff(safe, *params), 0.0)

    def _row_order(self) -> npt.NDArray:
        """The index that puts the rows grouped by kind (exact, right,
        left, interval, as :meth:`_component_log_likelihood` builds
        them) back in
        the data's order."""
        data = self.data
        masks = (data.mask_o, data.mask_r, data.mask_l, data.mask_i)
        grouped = np.concatenate([np.flatnonzero(mask) for mask in masks])
        return np.argsort(grouped)

    def _log_resp(self, w: npt.NDArray, params: Any) -> Any:
        """``log w_i + log L_i`` for every component (rows) and
        observation (columns)."""
        with np.errstate(divide="ignore"):
            log_w = np.log(w)
        return np.array(
            [
                log_w[i] + self._component_log_likelihood(params[i])
                for i in range(self.m)
            ]
        )

    def _window_prob(self, params_i: npt.NDArray) -> Any:
        """One component's probability of landing in each observation's
        truncation window ``(tl, tr]`` -- the per-component piece of the
        truncation correction."""
        tl, tr = self.data.tl, self.data.tr
        lo = 0.0
        fin = np.isfinite(tl)
        if fin.any():
            lo = np.where(
                fin, self.dist.ff(np.where(fin, tl, 0.0), *params_i), 0.0
            )
        hi = 1.0
        fin = np.isfinite(tr)
        if fin.any():
            hi = np.where(
                fin, self.dist.ff(np.where(fin, tr, 0.0), *params_i), 1.0
            )
        return hi - lo

    def neg_ll_of(self, w: npt.NDArray, params: Any) -> Any:
        """Observed negative log-likelihood of the mixture: counts multiply
        in the log domain, and truncated observations are conditioned on
        their window through the mixture probability of the window."""
        self._require_fit_data("neg_ll_of()")
        return self._neg_ll_from(self._log_resp(w, params), w, params)

    def _neg_ll_from(self, log_r: Any, w: Any, params: Any) -> Any:
        """:meth:`neg_ll_of` from ``log_r = _log_resp(w, params)``."""
        # log-sum-exp over the components, so the mixture density of an
        # observation is not lost to underflow in any one of them. In
        # autograd's functions, so the polish can differentiate it (#506).
        with np.errstate(all="ignore"):
            ll = np.sum(self.data.n * ag_logsumexp(log_r, axis=0))
            if self._truncated:
                win = 0.0
                for i in range(self.m):
                    win = win + w[i] * self._window_prob(params[i])
                ll = ll - np.sum(self.data.n * np.log(win))
        return -ll

    def _Q(self, params: Any) -> Any:
        """EM M-step objective: the (negative) expected complete-data
        log-likelihood over the component labels -- counts times
        responsibilities times each component's log-likelihood."""
        self._require_fit_data("_Q()")
        params = params.reshape(self.m, self.dist.k)
        total = 0.0
        for i in range(self.m):
            # Finite by construction (see _component_log_likelihood), so a
            # zero responsibility contributes exactly 0 and none inf.
            loglike = self._component_log_likelihood(params[i])
            total -= np.sum(self.data.n * self._resp[i] * loglike)
        return total

    def _expectation(self) -> Any:
        """EM E-step: set each observation's responsibilities ``_resp`` (the
        probability it belongs to each component, given the current fit)
        and the count-weighted mixing weights ``w``."""
        # Normalised in the log domain: dividing likelihoods that had all
        # underflowed to 0 gave 0/0 responsibilities (and the overflow
        # and invalid-value warnings of a discrete mixture).
        cached = self.__dict__.pop("_log_resp_cache", None)
        if (
            cached is not None
            and cached[0] is self.w
            and cached[1] is self.params
        ):
            # What the last EM iteration computed for its likelihood, at
            # these same weights and parameters (#589).
            log_r = cached[2]
        else:
            log_r = self._log_resp(self.w, self.params)
        with np.errstate(all="ignore"):
            self._resp = np.exp(log_r - logsumexp(log_r, axis=0))
        # Mixing weights are count-weighted responsibility totals.
        self.w = (self._resp * self.data.n).sum(axis=1) / self.data.n.sum()

    def _maximisation(self) -> Any:
        """EM M-step: refit every component's parameters by minimising
        :meth:`_Q` with the current responsibilities held fixed, on its
        exact (autograd) gradient (#506); finite differences of it were
        60% of a fit's time.

        ``Q`` and its gradient come from one autograd pass
        (``value_and_grad``), which computes the value anyway; scipy
        asking for them separately evaluated ``Q`` twice at every point,
        and ``Q`` at the start is the search's first evaluation (#589).
        Both are the same numbers as before, so the step is too.
        """
        bounds = self.dist.bounds * self.m
        x0 = self.params.ravel()
        jac = self._Q_jac()
        res = None
        with np.errstate(all="ignore"):
            if jac is not None:
                # A bound of 0 (a scale or shape) held just inside: the
                # gradient there is 0 / 0.
                inner = [(1e-10 if lo == 0 else lo, hi) for lo, hi in bounds]
                value_and_jac = self._Q_value_and_grad()
                at_x0: list = []

                def fun(p: npt.NDArray) -> Any:
                    q, g = value_and_jac(p)
                    g = np.asarray(g, dtype=float)
                    if not np.all(np.isfinite(g)):
                        raise _NonFiniteGradient
                    if not at_x0 and np.array_equal(p, x0):
                        at_x0.append(q)
                    return q, g

                try:
                    res = minimize(
                        fun,
                        x0,
                        jac=True,
                        bounds=inner,
                        options=self._m_step_options(),
                    )
                except _NonFiniteGradient:
                    res = None
                # Where the gradient breaks down (a component heading for
                # a point mass, whose shape runs off to 1e4 and beyond) or
                # the step makes Q worse, finite differences as before.
                if res is not None:
                    q0 = at_x0[0] if at_x0 else self._Q(x0)
                    slack = 1e-8 * max(1.0, abs(q0))
                    if not (
                        np.all(np.isfinite(res.x)) and res.fun <= q0 + slack
                    ):
                        res = None
            if res is None:
                res = minimize(self._Q, x0, bounds=bounds)
        self.params = res.x.reshape(self.m, self.dist.k)

    def _Q_jac(self) -> "Callable[..., Any] | None":
        """The gradient of :meth:`_Q` by autograd, or ``None`` (finite
        differences) for a distribution autograd cannot differentiate."""
        # Kept for the fit in progress (``fit`` drops it: a closure would
        # stop the model being pickled); False where autograd fails.
        cached = self.__dict__.get("_Q_jac_cache")
        if cached is not None:
            return cached or None
        jac = grad(self._Q)
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = np.asarray(jac(self.params.ravel()), dtype=float)
            usable = bool(np.all(np.isfinite(g)))
        except Exception:
            usable = False
        self._Q_jac_cache = jac if usable else False
        return jac if usable else None

    def _m_step_options(self) -> dict:
        """L-BFGS-B's options in the M-step: scipy's defaults for plain
        EM, tight ones for SQUAREM (see :meth:`_squarem_steps`)."""
        if self._exact_m_step:
            return {"ftol": 1e-15, "gtol": 1e-10}
        return {}

    def _Q_value_and_grad(self) -> Callable[..., Any]:
        """:meth:`_Q` and its autograd gradient in one pass, kept for the
        fit in progress like :meth:`_Q_jac`."""
        cached = self.__dict__.get("_Q_value_and_grad_cache")
        if cached is None:
            cached = value_and_grad(self._Q)
            self._Q_value_and_grad_cache = cached
        return cached

    def _em_iteration(self) -> Any:
        """One EM iteration (:meth:`_expectation` then
        :meth:`_maximisation`), after which ``_neg_ll`` holds the observed
        negative log-likelihood."""
        self._expectation()
        self._maximisation()
        # Convergence is tracked on the observed likelihood, not the
        # M-step objective. Its log-responsibilities are the next E-step's.
        log_r = self._log_resp(self.w, self.params)
        self._log_resp_cache = (self.w, self.params, log_r)
        self._neg_ll = self._neg_ll_from(log_r, self.w, self.params)

    def _em(
        self, tol: float = 1e-10, max_iter: int = 1000, budget: int = 20
    ) -> "str | None":
        """Fit by EM, polished by direct maximum likelihood (#506).

        EM moves linearly, and on a censored mixture it can crawl along a
        flat direction of the likelihood for all ``max_iter`` iterations
        (it stops when two iterations' negative log-likelihoods are within
        ``tol``), warning that it had not converged at what was already
        the maximum, after 17 s. So after ``budget`` iterations, or
        sooner where it converges, its answer is polished by BFGS on the
        observed likelihood with its exact gradient (``_polish``), and
        accepted when that is a verified maximum: a zero gradient and a
        positive-definite Hessian. Only if it is not does EM go on to
        ``max_iter``, polished again. Returns ``None`` for a verified
        maximum, else why it is not (for ``warn_unverified``, which
        :meth:`fit` gives unless the likelihood has no finite maximum).

        A mixture's likelihood has more than one maximum, so this short
        run is made from two starts where they differ (#582): the current
        weights and parameters (:meth:`_initialise_params`), and the split
        of :meth:`_failure_split_start`. The better is kept
        (:meth:`_better_start`) and, only if it is not verified, run on:
        by SQUAREM where ``fit`` was given ``em="squarem"``
        (:meth:`_squarem_steps`; #589). The short run is plain EM either
        way, so a fit verified after it is the same with either option.
        """
        verified, converged = self._em_from_starts(tol, budget)
        if verified:
            return None
        if not converged:
            converged = self._em_steps(
                tol, max_iter - budget, self._em_method == "squarem"
            )
            if self._polish():
                return None
        if not converged:
            return "EM reached its iteration limit"
        return "EM converged where the likelihood is not a verified maximum"

    def _em_from_starts(self, tol: float, budget: int) -> tuple[bool, bool]:
        """Up to ``budget`` EM iterations then the polish, from each start
        in turn, leaving the model at the better end point (see
        :meth:`_em`); whether that is a verified maximum, and whether its
        EM run converged."""
        starts = [(self.w, self.params)]
        other = self._failure_split_start()
        if other is not None and not np.allclose(other[1], self.params):
            starts.append(other)
        best: tuple | None = None
        for w, params in starts:
            self.w, self.params = w, params
            converged = self._em_steps(tol, budget)
            verified = self._polish()
            end = (verified, float(self._neg_ll), converged)
            if best is None or self._better_start(end, best[0]):
                best = (end, self.w, self.params, self._resp)
        assert best is not None
        (verified, loglike, converged), self.w, self.params, self._resp = best
        self._neg_ll = loglike
        return verified, converged

    def _better_start(self, end: tuple, best: tuple) -> bool:
        """Whether the end point ``end`` of a start, ``(verified,
        negative log-likelihood, converged)``, beats ``best``: a verified
        maximum beats one that is not (a likelihood that grows without
        bound as a component collapses onto a point mass is higher, and
        no answer), and otherwise the higher likelihood wins, by more than
        1e-6 per observation where both are verified -- so that two
        polishes of the same maximum, which agree to about that, keep the
        first start's (the fit as it was before #582)."""
        if end[0] != best[0]:
            return bool(end[0])
        margin = 1e-6 * float(np.sum(self.data.n)) if end[0] else 0.0
        return end[1] < best[1] - margin

    def _failure_split_start(self) -> "tuple[npt.NDArray, Any] | None":
        """A second EM start (#582): the failures, by count, cut into
        ``m`` consecutive blocks, each component fitted to one block, and
        every survivor (right-censored row) given to the last.

        :meth:`_initialise_params` cuts the *rows* into blocks, survivors
        and all. On field data -- a few early failures, a long tail of
        survivors, counts per row -- that mixes the early failures with
        survivors, and the fit went to a different maximum: a 3%
        defective sub-population plus wear-out fitted as 26% with a
        1,000-year life. Here the first component starts on the earliest
        failures, with their share of the units as its weight, and the
        last component holds the survivors, which is the usual reliability
        shape (infant mortality plus wear-out). ``None`` where a block
        cannot be fitted (fewer than ``k + 1`` distinct failure rows for
        it).
        """
        data, m, k = self.data, self.m, self.dist.k
        failed = np.flatnonzero(data.c != 1)
        if len(failed) < m * (k + 1):
            return None
        cum = np.cumsum(data.n[failed])
        blocks = []
        start = 0
        for i in range(1, m):
            end = int(np.searchsorted(cum, cum[-1] * i / m)) + 1
            end = max(end, start + k + 1)
            blocks.append(failed[start:end])
            start = end
        if len(failed) - start < k + 1:
            return None
        survivors = np.flatnonzero(data.c == 1)
        blocks.append(np.concatenate([failed[start:], survivors]))
        params = np.zeros((m, k))
        w = np.zeros(m)
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, rows in enumerate(blocks):
                    rows = np.sort(rows)
                    params[i] = self.dist.fit(
                        x=data.x[rows], c=data.c[rows], n=data.n[rows]
                    ).params
                    w[i] = data.n[rows].sum()
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(params)):
            return None
        return w / w.sum(), params

    def _em_steps(
        self, tol: float, max_iter: int, accelerate: bool = False
    ) -> bool:
        """Up to ``max_iter`` EM iterations; whether two in a row came
        within ``tol`` of each other in the negative log-likelihood.
        ``accelerate`` runs them by :meth:`_squarem_steps`."""
        if max_iter < 1:
            return False
        if accelerate:
            self._exact_m_step = True
            try:
                return self._squarem_steps(tol, max_iter)
            finally:
                self._exact_m_step = False
        self._em_iteration()
        f0 = self._neg_ll
        for _ in range(max_iter - 1):
            self._em_iteration()
            f1 = self._neg_ll
            if np.abs(f0 - f1) <= tol:
                return True
            f0 = f1
        return False

    def _squarem_steps(self, tol: float, max_iter: int) -> bool:
        """Up to ``max_iter`` EM iterations accelerated by SQUAREM (#589;
        Varadhan and Roland, 2008, scheme S3); whether two EM iterations
        in a row came within ``tol`` of each other, as for plain EM.

        Each cycle takes two EM steps from ``theta0`` (in the coordinates
        of :meth:`_pack`), ``theta1`` and ``theta2``, and from
        ``r = theta1 - theta0`` and ``v = theta2 - theta1 - r``
        extrapolates to ``theta0 - 2 a r + a**2 v`` with
        ``a = -|r| / |v|`` (at most -1; ``a = -1`` is ``theta2``), then
        takes one EM step from there. Where that ends below ``theta2`` in
        likelihood (or cannot be evaluated), the cycle ends at ``theta2``
        instead, so no cycle does worse than two plain EM steps. Where EM
        crawls along a flat direction, the extrapolation takes the many
        small steps at once.

        The M-steps are solved to full precision here (``ftol`` 1e-15,
        ``gtol`` 1e-10, against scipy's 2.2e-9 and 1e-5): the
        extrapolation assumes the steps are those of one smooth map, and
        steps that stop at scipy's tolerances wander by more than EM's
        own progress near the maximum. With them, plain EM on a censored
        two-Weibull mixture also stalled 1.3e-4 below the maximum after
        1000 iterations; this reaches it in about 40.
        """
        used = 0
        while used < max_iter:
            theta0 = self._pack(self.w, self.params)
            self._em_iteration()
            f1, used = self._neg_ll, used + 1
            if used >= max_iter:
                return False
            theta1 = self._pack(self.w, self.params)
            self._em_iteration()
            f2, used = self._neg_ll, used + 1
            if np.abs(f1 - f2) <= tol:
                return True
            if used >= max_iter:
                return False
            state2 = (self.w, self.params, f2)
            theta2 = self._pack(self.w, self.params)
            r = theta1 - theta0
            v = theta2 - theta1 - r
            norm_v = float(np.sqrt(np.sum(v**2)))
            if not (np.all(np.isfinite(v)) and norm_v > 0):
                continue
            a = min(-float(np.sqrt(np.sum(r**2))) / norm_v, -1.0)
            if a == -1.0:
                continue
            try:
                with np.errstate(all="ignore"):
                    self.w, self.params = (
                        np.asarray(z, dtype=float)
                        for z in self._unpack(theta0 - 2 * a * r + a**2 * v)
                    )
                    self._em_iteration()
                ok = bool(np.isfinite(self._neg_ll) and self._neg_ll <= f2)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                ok = False
            used += 1
            if not ok:
                self.w, self.params, self._neg_ll = state2
                self.__dict__.pop("_log_resp_cache", None)
        return False

    def _pack(self, w: npt.NDArray, params: npt.NDArray) -> npt.NDArray:
        """The weights and parameters as the unconstrained vector the
        polish searches: ``m - 1`` log-ratios of the weights to the last
        one, then each component's parameters, a bounded one as the log
        of its distance from the bound (or the logit between two)."""
        w = np.asarray(w, dtype=float)
        free = []
        # A weight of 0 or a parameter on its bound maps to an infinite
        # coordinate, which the caller checks for.
        with np.errstate(all="ignore"):
            logits = np.log(w[:-1]) - np.log(w[-1])
            for row in np.asarray(params, dtype=float):
                for value, (lo, hi) in zip(row, self.dist.bounds):
                    if lo is not None and hi is not None:
                        u = (value - lo) / (hi - lo)
                        free.append(np.log(u) - np.log1p(-u))
                    elif lo is not None:
                        free.append(np.log(value - lo))
                    elif hi is not None:
                        free.append(np.log(hi - value))
                    else:
                        free.append(value)
        return np.concatenate([logits, np.array(free, dtype=float)])

    def _unpack(self, theta: Any) -> Any:
        """The inverse of :meth:`_pack` (autograd-differentiable): the
        weights by softmax, and the parameters back on their scale."""
        m, k = self.m, self.dist.k
        logits = np.concatenate([theta[: m - 1], np.zeros(1)])
        log_w = logits - ag_logsumexp(logits)
        rows = []
        for i in range(m):
            row = []
            for j, (lo, hi) in enumerate(self.dist.bounds):
                u = theta[m - 1 + i * k + j]
                if lo is not None and hi is not None:
                    row.append(lo + (hi - lo) / (1.0 + np.exp(-u)))
                elif lo is not None:
                    row.append(lo + np.exp(u))
                elif hi is not None:
                    row.append(hi - np.exp(u))
                else:
                    row.append(u)
            rows.append(row)
        return np.exp(log_w), np.array(rows)

    def _polish(self) -> bool:
        """Direct maximum likelihood from the current weights and
        parameters (#506): BFGS on the observed negative log-likelihood,
        in the unconstrained coordinates of :meth:`_pack` and on its
        autograd gradient, the answer kept where it is better. Whether
        the result is a verified maximum (``is_local_minimum``: a zero
        gradient and a positive-definite Hessian)."""
        from .fitters import is_local_minimum, preconditioned_bfgs

        def fun(theta: Any) -> Any:
            w, params = self._unpack(theta)
            return self.neg_ll_of(w, params)

        n_obs = float(np.sum(self.data.n))
        jac, hess = grad(fun), hessian(fun)
        x0 = self._pack(self.w, self.params)
        if not np.all(np.isfinite(x0)):
            # A weight of 0, or a parameter on its bound: no interior
            # point to polish from.
            return False
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                f0 = float(fun(x0))
                res = preconditioned_bfgs(fun, x0, (), jac, obj_scale=n_obs)
            except Exception:
                return False
            x = x0
            if np.all(np.isfinite(res.x)) and res.fun <= f0:
                x = res.x
                self.w, self.params = self._unpack(x)
                self.w = np.asarray(self.w, dtype=float)
                self.params = np.asarray(self.params, dtype=float)
                self._neg_ll = float(res.fun)
            try:
                return is_local_minimum(fun, jac, hess, x, obj_scale=n_obs)
            except Exception:
                return False

    def _initialise_params(self) -> Any:
        """The EM starting point: cut the (sorted) data into ``m``
        consecutive blocks, fit one component to each block, and weight
        the components equally."""
        splits_x = np.array_split(self.data.x, self.m)
        splits_c = np.array_split(self.data.c, self.m)
        splits_n = np.array_split(self.data.n, self.m)
        params = np.zeros(shape=(self.m, self.dist.k))

        for i in range(self.m):
            params[i, :] = self.dist.fit(
                x=splits_x[i], c=splits_c[i], n=splits_n[i]
            ).params
        self.params = params
        self.w = np.ones(shape=(self.m)) / self.m

    @_FitMethod
    def fit(
        self,
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        *,
        em: str = "plain",
    ) -> Any:
        """
        Fit the mixture to data.

        Call it on the class, ``MixtureModel.fit(x, ..., dist=Weibull,
        m=2)``, to build and fit a model in one step, as with every other
        fitter; or on a model built with ``MixtureModel(dist, m)``, which
        it fits in place. Either way it returns the fitted model, with the
        ``params`` (one row per sub-distribution) and mixing weights ``w``.
        Untruncated data is fitted by the EM algorithm; truncated data by
        direct maximisation of the truncated likelihood.

        Parameters
        ----------

        x : array like, optional
            Array of observations of the random variables. If x is
            :code:`None`, xl and xr must be provided.
        c : array like, optional
            Array of censoring flag. -1 is left censored, 0 is observed, 1 is
            right censored, and 2 is intervally censored. If not provided
            will assume all values are observed.
        n : array like, optional
            Array of counts for each x. If data is provided as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.
        t : 2D-array like, optional
            2D array like of the left and right values at which the
            respective observation was truncated. If not provided it assumes
            that no truncation occurs.

        tl : array like or scalar, optional
            Values of left truncation for observations. If it is a scalar
            value assumes each observation is left truncated at the value.
            If an array, it is the respective 'late entry' of the observation

        tr : array like or scalar, optional
            Values of right truncation for observations. If it is a scalar
            value assumes each observation is right truncated at the value.
            If an array, it is the respective right truncation value for each
            observation

        xl : array like, optional
            Array like of the left array for 2-dimensional input of x. This
            is useful for data that is all intervally censored. Must be used
            with the :code:`xr` input.

        xr : array like, optional
            Array like of the right array for 2-dimensional input of x. This
            is useful for data that is all intervally censored. Must be used
            with the :code:`xl` input.
        dist : surpyval distribution
            The distribution of every component. Keyword only, and only on
            the class call (a model already has its ``dist``).
        m : int, optional
            The number of components (default 2). Keyword only, and only
            on the class call.
        em : {'plain', 'squarem'}, optional
            How the EM iterations run (keyword only). ``'plain'`` (the
            default) is the EM algorithm itself; ``'squarem'`` accelerates
            it with SQUAREM (Varadhan and Roland, 2008), which extrapolates
            along the path two EM steps take and keeps the result only
            where the likelihood is no lower than after those two steps.
            It needs far fewer iterations where EM crawls (a fit that
            cannot verify its maximum after the first 20, and runs on
            towards 1000), and ends at the same verified maximum where
            both reach one, to the precision of its verification; it may,
            like any change of path, reach a different local maximum.
            Truncated data is fitted directly, not by EM, and ignores it.

        Returns
        -------

        MixtureModel
            The fitted model (on a model, the model itself).

        Warns
        -----
        UserWarning
            "No finite maximum" when a component has collapsed onto a
            point mass (every row it explains is consistent with one time),
            where a mixture's likelihood grows without bound.

        Examples
        --------

        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17 ,17, 18, 19]
        >>> # A Weibull Mixture Model with 2 sub-distributions
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm
        Parametric Mixture SurPyval Model
        =================================
        Distribution        : Weibull
        Sub-Distributions   : 2
        Fitted by           : EM
        Data                : 17 units: 17 events at 15 unique times
        Weights             :
                0.6184891886499861,
                0.381510811350014
        Parameters          :
             alpha: [ 6.32508961 17.37701969]
              beta: [ 1.83105154 12.01392721]
        """

        check_option("em", em, EM_METHODS)
        data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)

        # Count observations from the validated data so ``xl``/``xr``-only
        # input works (#254), and weigh by counts.
        if data.n.sum() < self.m * (self.dist.k + 1):
            raise ValueError("More parameters than data points")

        self.data = data
        self._em_method = em
        # A refit in place: the criteria are recomputed from the new fit
        self._ic_n = None
        for name in ("_aic", "_aic_c", "_bic", "_theta_cov"):
            self.__dict__.pop(name, None)
        self._truncated = bool(np.isfinite(data.t).any())
        self._resp = np.ones(shape=(self.m, len(self.data.x))) / self.m

        self._initialise_params()

        if self._truncated:
            # The truncation correction couples the components through the
            # mixture window probability, so the label-based EM does not
            # apply; maximise the observed truncated likelihood directly,
            # warm-started from the split-fit initialisation (#254), and
            # polish and verify its answer as the EM path does (#560).
            self._direct_mle()
            unverified = (
                None
                if self._polish()
                else "L-BFGS-B's answer is not a verified maximum"
            )
        else:
            try:
                unverified = self._em()
            finally:
                # Closures and arrays of the fit in progress (a closure
                # would stop the model being pickled)
                for name in _FIT_CACHES:
                    self.__dict__.pop(name, None)
        # One warning: a component collapsed onto a point mass has no
        # finite maximum, which is also why its search was not verified.
        if self._warn_if_point_mass() or self._warn_if_lfp_limit():
            self.maximum = "no finite maximum"
        elif unverified is not None:
            self.maximum = "unverified"
            warn_unverified("The mixture fit", unverified, "check the fit")
        else:
            self.maximum = "verified"
        return self

    def _warn_if_lfp_limit(self) -> bool:
        """Warn when a component has run off past the data (#650), and
        say whether one did.

        A component whose failures all lie beyond the last observation
        explains no failure: it only holds back a share of the units that
        never fail within the data, as a limited-failure proportion does.
        The likelihood then keeps rising as it moves further out, towards
        the limit where it never fails -- the other components with a
        limited-failure proportion -- which no member of the mixture
        reaches: a two-Weibull fit came back with a second component of
        scale 33,561 (the largest observation 1,150), shape 5.2 and weight
        0.09, as a verified maximum, 2e-8 below that limit.

        A component is that, here, when the likelihood with it replaced
        by one that never fails (survival 1 at every time) is at least as
        high as the fit's, to the verification's tolerance. One that
        explains a single failure loses its density there, which is
        never within that. Truncated data are not checked: a component
        that never fails has no probability in a window.
        """
        if self._truncated or self.m < 2:
            return False
        data = self.data
        never = np.where(data.mask_r, 0.0, LOG_FLOOR)
        with np.errstate(all="ignore"):
            log_r = self._log_resp(self.w, self.params)
            nll = float(self._neg_ll_from(log_r, self.w, self.params))
            for i in range(self.m):
                limit = np.array(log_r)
                limit[i] = np.log(self.w[i]) + never
                nll_limit = float(
                    self._neg_ll_from(limit, self.w, self.params)
                )
                if not nll_limit <= nll + OPTIMUM_GTOL * max(1.0, abs(nll)):
                    continue
                params = ", ".join(
                    f"{name} = {value:.4g}"
                    for name, value in zip(
                        self.dist.parameter_names, self.params[i]
                    )
                )
                warn_no_maximum(
                    f"mixture component {i} ({params}, weight "
                    f"{self.w[i]:.3g}) explains no failure: its failures "
                    "all lie beyond the data, and the likelihood keeps "
                    "increasing as it moves further out, towards the other "
                    "components with a limited-failure proportion of "
                    f"{1 - self.w[i]:.3g}, which no mixture reaches",
                    "Its parameters are where the search stopped and "
                    "describe nothing in the data",
                    "the data hold units that do not fail within them: fit "
                    "one component fewer with lfp=True (for two components, "
                    "dist.fit(x, c, lfp=True))",
                )
                return True
        return False

    def _warn_if_point_mass(self) -> bool:
        """Warn when a component has collapsed onto a point mass (#392),
        and say whether it did.

        A mixture's likelihood grows without bound as one component
        concentrates on a single time (a spike explaining a cluster of
        tied values while the others explain the rest), so a fit that
        heads there has no finite maximum and stops wherever the search
        gives up: with 10 of 20 rows at 3.0 a Weibull component came back
        with beta 8955, in silence.

        The criterion is the univariate one (``_point_mass_region``,
        which refuses such data for a single distribution), applied to
        the rows each component still explains -- those it gives a
        positive responsibility, to double precision. When one time is
        consistent with every one of them, the component is a spike. A
        component of an ordinary fit explains rows at several distinct
        times, and is never flagged.
        """
        region_of = getattr(self.dist, "_point_mass_region", None)
        if region_of is None:
            return False
        data = self.data
        log_r = self._log_resp(self.w, self.params)
        with np.errstate(all="ignore"):
            resp = np.exp(log_r - logsumexp(log_r, axis=0))
        for i in range(self.m):
            rows = (resp[i] > 0) & (data.n > 0)
            if not rows.any():
                continue
            explained = SurpyvalData(
                x=data.x[rows], c=data.c[rows], n=data.n[rows], t=data.t[rows]
            )
            region = region_of(explained, False, False)
            if region is None:
                continue
            params = ", ".join(
                f"{name} = {value:.4g}"
                for name, value in zip(
                    self.dist.parameter_names, self.params[i]
                )
            )
            explained_rows = int(data.n[rows].sum())
            warn_no_maximum(
                f"mixture component {i} ({params}) has collapsed onto a "
                f"point mass: a single time {region} is consistent with "
                f"every one of the {explained_rows} rows it explains, and "
                "a mixture's likelihood grows without bound as a "
                "component concentrates on one time",
                "The reported parameters and weights are meaningless",
                "a component is a point mass: the data hold a cluster of "
                "tied values; model that cluster separately, or fit fewer "
                "components",
            )
            return True
        return False

    def _direct_mle(self) -> Any:
        """Directly maximise the observed (truncation-corrected) negative
        log-likelihood over the mixing weights (via softmax logits) and the
        component parameters."""
        k = self.dist.k

        def unpack(theta: npt.NDArray) -> Any:
            logits = np.append(theta[: self.m - 1], 0.0)
            logits = logits - logits.max()
            w = np.exp(logits)
            w = w / w.sum()
            params = theta[self.m - 1 :].reshape(self.m, k)
            return w, params

        def obj(theta: npt.NDArray) -> Any:
            w, params = unpack(theta)
            return self.neg_ll_of(w, params)

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * (
            self.m - 1
        )
        for _ in range(self.m):
            for lo, hi in self.dist.bounds:
                lo_s = None if lo is None else (1e-10 if lo == 0 else lo)
                bounds.append((lo_s, hi))

        x0 = np.concatenate([np.zeros(self.m - 1), self.params.ravel()])
        with np.errstate(all="ignore"):
            res = minimize(obj, x0, bounds=bounds)
        self.w, self.params = unpack(res.x)
        self._neg_ll = float(res.fun)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        The mean of the fitted mixture, :math:`\sum_{j} w_{j} E[X_{j}]`.

        Returns
        -------
        float
            The weighted sum of the component means.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> round(float(wmm.mean()), 4)
        9.8294
        """
        mean = 0
        for i in range(self.m):
            mean += self.w[i] * self.dist.mean(*self.params[i])
        return mean

    def random(
        self,
        size: int,
        *args: Any,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Draw random samples from the fitted mixture.

        The number drawn from each component is multinomial with the
        mixing weights ``w``, and the draws are shuffled together.

        Parameters
        ----------
        size : int
            The number of samples to draw.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a draw of its own, which neither depends
            on nor advances numpy's global stream (an int is
            ``np.random.default_rng(seed)``). ``None`` (the default)
            draws from numpy's global stream, so ``np.random.seed``
            reproduces it.

        Returns
        -------
        numpy array
            ``size`` values from the mixture, in random order.
        """
        # numpy's global stream (None), or one generator for every step.
        rng = None if random_state is None else as_generator(random_state)
        draw: Any = np.random if rng is None else rng
        sizes = draw.multinomial(size, self.w)
        rvs = np.zeros(size)
        s_last = 0
        for i, s in enumerate(sizes):
            rvs[s_last : s + s_last] = self.dist.random(
                s, *self.params[i, :], random_state=rng
            )
            s_last += s
        # Shuffles the data (inplace) so that the data is random
        draw.shuffle(rvs)
        return rvs

    @keeps_query_shape
    def df(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The probability density function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the probability density function will be
            evaluated.

        Returns
        -------

        array like
            The probability density function evaluated at x.
        """
        x = np.asarray(x, dtype=float)
        df = np.zeros_like(x)
        for i in range(self.m):
            df += self.w[i] * self.dist.df(x, *self.params[i])
        return df

    @keeps_query_shape
    def ff(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The cumulative density function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the cumulative density function will be
            evaluated.

        Returns
        -------

        array like
            The cumulative density function evaluated at x.
        """
        F = np.zeros_like(x)
        for i in range(self.m):
            F = F + self.w[i] * self.dist.ff(x, *self.params[i])
        return F

    @keeps_query_shape
    def sf(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The survival function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the survival function will be evaluated.

        Returns
        -------

        array like
            The survival function evaluated at x.

        Notes
        -----
        The components' survival functions are summed, weighted, as
        ``ff`` and ``df`` sum theirs, rather than taken as ``1 - ff``:
        each keeps its precision far in the upper tail, so their sum does
        too, where ``1 - ff`` would round to 0 once the survival is below
        about 1e-16.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.sf([5, 15]).round(4)
        array([0.7043, 0.3264])
        """
        x = np.asarray(x, dtype=float)
        S = np.zeros_like(x)
        for i in range(self.m):
            S = S + self.w[i] * self.dist.sf(x, *self.params[i])
        return S

    @keeps_query_shape
    def Hf(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The cumulative hazard function of the fitted mixture, ``-log
        sf(x)``.

        It keeps its precision at both ends: where the mixture has failed
        less than half it is ``-log1p(-ff(x))`` (``ff`` is accurate where
        it is small), elsewhere ``-log sf(x)`` from the summed survival,
        and where that survival underflows to 0 it is the log-sum-exp of
        the components' log survivals, so it stays finite far in the tail.

        Parameters
        ----------

        x : array like
            The values at which the cumulative hazard will be evaluated.

        Returns
        -------

        array like
            The cumulative hazard evaluated at x.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.Hf([5, 15]).round(4)
        array([0.3505, 1.1196])
        """
        x = np.asarray(x, dtype=float)
        F = np.asarray(self.ff(x), dtype=float)
        S = np.asarray(self.sf(x), dtype=float)
        with np.errstate(all="ignore"):
            H = np.where(F < 0.5, -np.log1p(-F), -np.log(S))
            gone = (S == 0) & np.isfinite(x)
            if np.any(gone):
                xs = x[gone]
                log_w = np.log(self.w)[:, None]
                log_s = np.array(
                    [self.dist.log_sf(xs, *row) for row in self.params]
                )
                H[gone] = -logsumexp(log_w + log_s, axis=0)
        return H

    @removed_arguments("0.23", X="'given'")
    def cs(self, x: Any, given: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The conditional survival function of the fitted model.

        .. versionchanged:: 0.22
           The time already survived is ``given`` (it was ``X``), the
           name the regression models' ``sf_tvc(..., given=)`` uses.

        .. versionchanged:: 0.23
           ``X`` is removed.

        Parameters
        ----------

        x : array like
            The values at which the conditional survival function will be
            evaluated.

        given : array like
            The values at which the item is known to have survived to.

        Returns
        -------

        array like
            The conditional survival function evaluated at x given given:
            ``exp(-(Hf(x + given) - Hf(given)))``, from the cumulative
            hazard, so it stays exact where ``sf(given)`` underflows;
            ``nan`` only where ``Hf(given)`` is itself infinite.
        """
        # As arrays: ``x + given`` on a list concatenated (or raised) rather
        # than adding.
        x = np.asarray(x, dtype=float)
        given = np.asarray(given, dtype=float)
        H_given = np.asarray(self.Hf(given), dtype=float)
        H_end = np.asarray(self.Hf(x + given), dtype=float)
        with np.errstate(invalid="ignore"):
            out = np.exp(H_given - H_end)
        return np.where(np.isinf(H_given), np.nan, out)[()]

    @keeps_query_shape
    def hf(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The hazard function of the fitted mixture, ``df(x) / sf(x)``.

        Where the mixture's survival underflows to 0 (far in its upper
        tail) the ratio is taken from the components' log densities and
        log survivals instead, so it stays finite there.

        Parameters
        ----------

        x : array like
            The values at which the hazard function will be evaluated.

        Returns
        -------

        array like
            The hazard function evaluated at x.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.hf([5, 15]).round(4)
        array([0.1091, 0.1435])
        """
        x = np.asarray(x, dtype=float)
        sf = self.sf(x)
        with np.errstate(all="ignore"):
            hf = self.df(x) / sf
            gone = (sf == 0) & np.isfinite(x)
            if np.any(gone):
                xs = x[gone]
                log_w = np.log(self.w)[:, None]
                log_f = np.array(
                    [self.dist.log_df(xs, *row) for row in self.params]
                )
                log_s = np.array(
                    [self.dist.log_sf(xs, *row) for row in self.params]
                )
                hf[gone] = np.exp(
                    logsumexp(log_w + log_f, axis=0)
                    - logsumexp(log_w + log_s, axis=0)
                )
        return hf

    @keeps_query_shape
    def qf(self, p: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The quantile function of the fitted mixture: the time by which a
        fraction ``p`` has failed (the B10 life is ``qf(0.1)``).

        A mixture's failure function has no closed-form inverse, so the
        quantile is found numerically: it lies between the smallest and
        the largest of the components' quantiles at ``p``, and is the root
        of ``ff(t) = p`` between them (for a discrete distribution, the
        smallest ``t`` with ``ff(t) >= p``). ``qf(0)`` and ``qf(1)`` are
        the ends of the support. A probability outside [0, 1] gives NaN,
        with one warning, as the other models' ``qf`` (it is most often a
        percentage given for a probability: ``qf(10)`` for the B10 life);
        NaN gives NaN.

        Parameters
        ----------

        p : array like
            The probabilities, in [0, 1], at which to find the quantiles.

        Returns
        -------

        array like
            The quantiles, shaped as ``p``.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.qf([0.1, 0.5]).round(4)
        array([2.4519, 8.3201])
        """
        u = np.asarray(p, dtype=float)
        outside = warn_outside_unit_interval(u)
        q = np.full(u.shape, np.nan)
        with np.errstate(all="ignore"):
            comp = np.array(
                [
                    np.asarray(self.dist.qf(np.clip(u, 0, 1), *row), float)
                    for row in self.params
                ]
            ).reshape((self.m,) + u.shape)
        lo, hi = comp.min(axis=0), comp.max(axis=0)
        ok = ~outside & ~np.isnan(u)
        q = np.where(ok & (u == 0), lo, q)
        q = np.where(ok & (u == 1), hi, q)
        discrete = getattr(self.dist, "discrete", False)
        for i in np.flatnonzero(ok & (u > 0) & (u < 1)):
            a, b, target = float(lo[i]), float(hi[i]), float(u[i])
            if discrete:
                q[i] = self._discrete_quantile(target, a, b)
            elif a == b:
                q[i] = a
            else:
                q[i] = brentq(
                    lambda t: float(self.ff(t)) - target,
                    a,
                    b,
                    xtol=1e-14 * max(1.0, abs(a)),
                    rtol=4 * np.finfo(float).eps,
                )
        return q

    def _discrete_quantile(self, target: float, a: float, b: float) -> float:
        """The smallest integer ``k`` in ``[a, b]`` with ``ff(k) >=
        target``, by bisection (``ff(b) >= target`` there)."""
        a, b = float(np.floor(a)), float(np.ceil(b))
        while b - a > 1:
            mid = float(np.floor((a + b) / 2))
            if float(self.ff(mid)) >= target:
                b = mid
            else:
                a = mid
        return a if float(self.ff(a)) >= target else b

    # -- inference (#651) --------------------------------------------------

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows (and columns) of :meth:`covariance`, in
        its order, which :meth:`param_cb` takes: each component's
        parameters suffixed by the component's index (``alpha_0``,
        ``beta_0``, ``alpha_1``, ...: the rows of ``params`` in turn),
        then the weights ``w_0``, ..., ``w_{m-1}``.

        Examples
        --------
        >>> import surpyval as surv
        >>> surv.MixtureModel(surv.Weibull, 2).covariance_names
        ['alpha_0', 'beta_0', 'alpha_1', 'beta_1', 'w_0', 'w_1']
        """
        names = [
            f"{name}_{i}"
            for i in range(self.m)
            for name in self.dist.parameter_names
        ]
        return names + [f"w_{i}" for i in range(self.m)]

    def _theta_covariance(self) -> npt.NDArray:
        """The inverse of the observed information in the unconstrained
        coordinates of :meth:`_pack` (the weights' log-ratios to the last
        one, then each component's parameters on the scale the polish
        searches), at the fitted maximum; computed once and kept."""
        cached = self.__dict__.get("_theta_cov")
        if cached is not None:
            return cached
        if self.params is None:
            raise ValueError("covariance() needs a fitted mixture")
        if self.data is None:
            raise no_covariance_error(
                "the dict this mixture was restored from has none, and the "
                "mixture does not carry the data to compute it from"
            )
        if self.maximum == "no finite maximum":
            raise no_covariance_error(
                "the fit has no finite maximum, so there is no observed "
                "information to invert"
            )
        theta = self._pack(self.w, self.params)
        if not np.all(np.isfinite(theta)):
            raise no_covariance_error(
                "a weight is 0 or a parameter is on its bound, where the "
                "likelihood is not regular"
            )

        def fun(th: Any) -> Any:
            w, params = self._unpack(th)
            return self.neg_ll_of(w, params)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = np.asarray(hessian(fun)(theta), dtype=float)
            if not np.all(np.isfinite(info)):
                # autograd's second derivative can be NaN where the first
                # is finite: central differences of the gradient instead.
                jac = grad(fun)
                steps = 1e-5 * np.maximum(np.abs(theta), 1.0)
                cols = []
                for j, step in enumerate(steps):
                    e = np.zeros_like(theta)
                    e[j] = step
                    up = np.asarray(jac(theta + e), dtype=float)
                    down = np.asarray(jac(theta - e), dtype=float)
                    cols.append((up - down) / (2 * step))
                info = np.array(cols).T
        info = (info + info.T) / 2
        if not np.all(np.isfinite(info)):
            raise no_covariance_error(
                "the observed information is not finite at the fit"
            )
        cov = safe_inv(info)
        self._theta_cov = cov
        return cov

    def _natural(self, theta: Any) -> Any:
        """The parameters and weights, in the order of
        :attr:`covariance_names`, at the coordinates ``theta``."""
        w, params = self._unpack(theta)
        return np.concatenate([np.ravel(params), w])

    def covariance(self) -> npt.NDArray:
        """
        The covariance of the fitted parameters and weights, in the order
        of :attr:`covariance_names`: each component's parameters in turn
        (the rows of ``params``), then the ``m`` weights.

        It is the inverse of the observed information of the mixture's
        log-likelihood at the fitted maximum, taken in the coordinates
        the fit's final direct maximisation searches -- the weights as
        the ``m - 1`` log-ratios ``log(w_j / w_{m-1})`` (a softmax), each
        parameter with a bound as the log of its distance from it (the
        logit between two bounds) -- and carried to the parameters and
        weights by the delta method. As the weights sum to one, their
        block is singular: each of its rows sums to 0.

        Raises a ``ValueError`` where there is none: a mixture restored
        from a dict that does not store it (it carries no data), a fit
        with no finite maximum (a component collapsed onto a point mass),
        or one with a weight of 0 or a parameter on its bound.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.covariance().shape
        (6, 6)
        >>> bool(np.allclose(wmm.covariance()[4:, 4:].sum(axis=1), 0))
        True
        """
        cov_theta = self._theta_covariance()
        theta = self._pack(self.w, self.params)
        jac = np.atleast_2d(jacobian(self._natural)(theta))
        return jac @ cov_theta @ jac.T

    def standard_errors(self) -> npt.NDArray:
        """
        The standard errors of the fitted parameters and weights, the
        square roots of the diagonal of :meth:`covariance`, in the order
        of :attr:`covariance_names`; ``nan`` where a variance is not
        positive. Raises the ``ValueError`` of :meth:`covariance` where
        the model has none.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.standard_errors().round(3)
        array([1.619, 0.589, 0.733, 5.275, 0.143, 0.143])
        """
        return standard_errors_of(self.covariance())

    @staticmethod
    def _check_wald(method: "str | None", bound: str) -> None:
        """A mixture has Wald bounds only."""
        if method is not None:
            check_option(
                "method",
                method,
                ("wald",),
                "A mixture has Wald (delta-method) bounds only",
            )
        check_option("bound", bound, BOUNDS)

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: "str | None" = None,
    ) -> npt.NDArray:
        """
        The Wald confidence bound on one component's parameter or weight.

        As ``Parametric.param_cb``'s Wald bound: from the standard error
        of :meth:`standard_errors`, on a scale chosen from the
        parameter's support so the interval stays in it -- the log of a
        positive parameter, the logit of a weight -- or its own scale
        for an unbounded one. Where the variance is not positive the
        bound is ``nan``, with a warning saying why.

        Parameters
        ----------
        name : str
            The parameter, as :attr:`covariance_names` names it: a
            component's parameter suffixed by its index (``"alpha_0"``,
            ``"beta_1"``), or a weight (``"w_0"``).
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : str, optional
            ``"two-sided"`` (the default), ``"upper"`` or ``"lower"``.
        method : str, optional
            ``"wald"``, the only bound a mixture has (``None``, the
            default, is it).

        Returns
        -------
        numpy array
            ``[lower, upper]`` for a two-sided bound, else the one bound.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.param_cb("alpha_1").round(3)
        array([15.998, 18.875])
        >>> wmm.param_cb("w_0", bound="lower").round(3)
        array([0.375])
        """
        check_alpha_ci(alpha_ci)
        self._check_wald(method, bound)
        names = self.covariance_names
        if name not in names:
            raise option_error("name", name, names)
        idx = names.index(name)
        n_params = self.m * self.dist.k
        lower: "float | None"
        upper: "float | None"
        if idx < n_params:
            p_hat = float(np.ravel(self.params)[idx])
            lower, upper = self.dist.bounds[idx % self.dist.k]
        else:
            p_hat = float(self.w[idx - n_params])
            lower, upper = 0.0, 1.0
        var = float(self.covariance()[idx, idx])
        return wald_bound_on_support(
            p_hat, var, lower, upper, alpha_ci, bound, name
        )

    def _mixture_fn(self, fn: str, x: Any, theta: Any) -> Any:
        """The mixture's ``fn`` (``"sf"``, ``"ff"`` or ``"df"``) at ``x``
        for the coordinates ``theta`` (autograd-differentiable)."""
        w, params = self._unpack(theta)
        out = 0.0
        for i in range(self.m):
            out = out + w[i] * getattr(self.dist, fn)(x, *params[i])
        return out

    def _delta_var(self, func: Callable[..., Any]) -> Any:
        """The delta-method variance of ``func(theta)`` at the fit, in the
        coordinates of :meth:`_theta_covariance`."""
        cov = self._theta_covariance()
        theta = self._pack(self.w, self.params)
        with np.errstate(all="ignore"):
            jac = np.atleast_2d(jacobian(func)(theta))
        var = np.einsum("ij,jk,ik->i", jac, cov, jac)
        # Rounding can leave a zero variance a hair below zero.
        scale = np.einsum("ij,jk,ik->i", abs(jac), abs(cov), abs(jac))
        return np.where((var < 0) & (var >= -1e-10 * scale), 0.0, var)

    def _delta_var_each(
        self, func_at: Callable[[Any], Callable[..., Any]], n: int
    ) -> Any:
        """``_delta_var`` of a function of ``n`` points, each point's
        variance its own: ``func_at(idx)`` is the function at the points
        ``idx``. A gradient that is not finite at one point (far in a
        tail) is ``0 * nan`` in every other point's reverse pass, so it
        made every variance ``nan`` (#710, as #652 for one distribution);
        those points are differentiated again one at a time, and only a
        point that overflows keeps its ``nan``."""
        var = np.array(self._delta_var(func_at(slice(None))), dtype=float)
        bad = ~np.isfinite(var)
        if n > 1 and bad.any():
            for i in np.flatnonzero(bad):
                var[i] = self._delta_var(func_at(slice(i, i + 1)))[0]
        return var

    def _sd(self, var: Any, x: Any, on: str) -> Any:
        """``sqrt(var)``, ``nan`` with one warning where there is none:
        where the variance is negative the covariance is not positive
        definite; where it is not finite while the covariance is, the
        function's derivatives overflowed at that ``x`` (far in a tail),
        and the message names that ``x`` rather than blaming the
        covariance (#710)."""
        var = np.asarray(var, dtype=float)
        bad = ~(var >= 0)
        where = np.broadcast_to(np.atleast_1d(x), var.shape)
        # A missing x is NaN in silence.
        warn = bad & ~np.isnan(where)
        if np.any(warn):
            overflow = warn & ~np.isfinite(var)
            if not np.all(np.isfinite(self._theta_covariance())):
                overflow = np.zeros_like(warn)
            singular = warn & ~overflow
            if np.any(singular):
                warn_wald_undefined(
                    f"{on} at x = {where[singular].tolist()}",
                    "its delta-method variance is negative or not finite, "
                    "so the parameter covariance is not positive definite",
                    stacklevel=caller_stacklevel(),
                )
            if np.any(overflow):
                warn_wald_undefined(
                    f"{on} at x = {where[overflow].tolist()}",
                    f"the derivatives of {on} with respect to the "
                    "parameters overflow there (that x is too far in the "
                    "mixture's tail for double precision), so it has no "
                    "delta-method variance; the bounds at the other x are "
                    "computed on their own",
                    stacklevel=caller_stacklevel(),
                )
        return np.sqrt(np.where(bad, np.nan, var))

    def _log_mixture(self, fn: str, x: Any, theta: Any) -> Any:
        """The log of the mixture's ``fn`` (``"sf"`` or ``"df"``) at
        ``x`` for the coordinates ``theta``: the log-sum-exp of the
        components' weighted logs (autograd-differentiable), finite far
        in the upper tail, where the sum itself underflows to 0."""
        w, params = self._unpack(theta)
        log_fn = getattr(self.dist, "log_" + fn)
        terms = [np.log(w[i]) + log_fn(x, *params[i]) for i in range(self.m)]
        return ag_logsumexp(np.stack(terms), axis=0)

    def _tail_fn(self, fn: str, x: Any, theta: Any, head: bool) -> Any:
        """The mixture's ``Hf`` or ``hf`` at ``x`` for the coordinates
        ``theta`` (autograd-differentiable), computed as is accurate there:
        where less than half has failed (``head``) ``Hf`` is
        ``-log1p(-ff)`` and ``hf`` is ``df / sf``; above, both are from the
        log-sum-exp of the components' weighted log survivals (and log
        densities), finite where the survival underflows and ``df / sf``
        is 0 / 0 (#710)."""
        if head:
            if fn == "Hf":
                return -np.log1p(-self._mixture_fn("ff", x, theta))
            return self._mixture_fn("df", x, theta) / self._mixture_fn(
                "sf", x, theta
            )
        log_sf = self._log_mixture("sf", x, theta)
        if fn == "Hf":
            return -log_sf
        return np.exp(self._log_mixture("df", x, theta) - log_sf)

    def _tail_var(self, fn: str, t: Any, g: Any, log: bool) -> Any:
        """The delta-method variance, each time's own, of ``fn``
        (``"Hf"`` or ``"hf"``; of its log with ``log``) at the times
        ``t``, where its fitted value ``g`` is positive and finite; 0
        elsewhere (the bound is the estimate there)."""
        var = np.zeros(t.shape)
        positive = (g > 0) & np.isfinite(g)
        head = np.asarray(self.ff(t), dtype=float) < 0.5
        for part in (head, ~head):
            idx = np.flatnonzero(part & positive)
            if idx.size == 0:
                continue
            ts, is_head = t[idx], bool(head[idx[0]])

            def func_at(sel: Any, ts: Any = ts, hd: bool = is_head) -> Any:
                tsel = ts[sel]

                def func(th: Any) -> Any:
                    value = self._tail_fn(fn, tsel, th, hd)
                    return np.log(value) if log else value

                return func

            var[idx] = self._delta_var_each(func_at, idx.size)
        return var

    @keeps_query_shape
    def cb(
        self,
        x: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        r"""
        Wald confidence bounds of the ``on`` function at ``x``, by the
        delta method from :meth:`covariance`.

        As ``Parametric.cb``'s Wald bounds: those on ``sf``, ``ff`` and
        ``Hf`` are one bound on the log cumulative hazard, ``log(-log
        sf)`` (the "log-log" scale), which keeps them within their range;
        those on ``hf`` and ``df`` are on the log scale. Each ``x``'s
        bound is computed on its own. Where the delta-method variance is
        negative (the covariance is not positive definite) a bound is
        ``nan``, with a warning; where the function's derivatives overflow
        at an ``x`` far in the tail, that ``x``'s bound alone is ``nan``,
        with a warning naming it (#710). The ``Hf`` bound is formed from
        ``Hf`` itself and the ``hf`` bound from the mixture's own hazard,
        so both stay finite, and contain the estimate, past where ``sf``
        underflows.

        Parameters
        ----------
        x : array like or scalar
            The times at which to bound the function.
        on : ('sf', 'ff', 'Hf', 'hf', 'df'), optional
            The function to bound (``'R'`` and ``'F'`` are ``'sf'`` and
            ``'ff'``). Defaults to ``'sf'``.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : ('two-sided', 'upper', 'lower'), optional
            Defaults to two-sided.
        method : 'wald', optional
            The only bound a mixture has.

        Returns
        -------
        numpy array
            The bound(s), shaped as ``x``; a two-sided bound adds a last
            ``[lower, upper]`` axis.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.cb([5, 15]).round(4)
        array([[0.4687, 0.8503],
               [0.1344, 0.5354]])
        """
        check_alpha_ci(alpha_ci)
        self._check_wald(method, bound)
        on = {"R": "sf", "F": "ff"}.get(on, on)
        check_option("on", on, CB_ON)
        t = np.atleast_1d(np.asarray(x, dtype=float))
        if t.size == 0:
            return np.empty((0, 2) if bound == "two-sided" else (0,))
        theta = self._pack(self.w, self.params)
        alpha, signs = bound_signs(alpha_ci, bound)
        k = signs * z(1 - alpha)
        if on == "Hf":
            # The sf band's scale, log(-log sf), is log Hf: the bound on
            # it from Hf itself stays finite, and contains the estimate,
            # far past where sf underflows (#710; it was [inf, inf]).
            H = np.asarray(self.Hf(t), dtype=float)
            with np.errstate(all="ignore"):
                var = self._tail_var("Hf", t, H, log=True)
                sd = self._sd(var, t, on)
                out = H[:, None] * np.exp(k * sd[:, None])
            out = np.where(((H == 0) | np.isinf(H))[:, None], H[:, None], out)
            return out if bound == "two-sided" else out[:, 0]
        if on in ("sf", "ff"):
            S = np.asarray(self._mixture_fn("sf", t, theta), dtype=float)
            F = np.asarray(self._mixture_fn("ff", t, theta), dtype=float)
            # The smaller of S and F, each accurate where it is small;
            # Var S = Var F, taken relative to it.
            left = F < 0.5
            small = np.where(left, F, S)
            unit = np.where(small > 0, small, 1.0)

            def func_at(idx: Any) -> Callable[..., Any]:
                ti, left_i, unit_i = t[idx], left[idx], unit[idx]

                def func(th: Any) -> Any:
                    value = np.where(
                        left_i,
                        -self._mixture_fn("ff", ti, th),
                        self._mixture_fn("sf", ti, th),
                    )
                    return value / unit_i

                return func

            with np.errstate(all="ignore"):
                var = self._delta_var_each(func_at, t.size)
            # Where S or F is below the normal range the bound is the edge
            # it is at (sf_link_bound), whatever its variance.
            var = np.where(small < np.finfo(float).tiny, 0.0, var)
            sd = unit * self._sd(var, t, on)
            return sf_link_bound(
                S, sd, alpha_ci, bound, "loglog", ff_hat=F, on=on
            )

        with np.errstate(all="ignore"):
            if on == "hf":
                # The mixture's own hazard: df / sf is 0 / 0 where sf
                # underflows (#710).
                g = np.asarray(self.hf(t), dtype=float)
                var = self._tail_var("hf", t, g, log=False)
            else:
                g = np.asarray(self._mixture_fn("df", t, theta), dtype=float)
                # Where the density is 0 or infinite the bound is the
                # estimate, whatever its variance.
                var = np.zeros(t.shape)
                idx = np.flatnonzero((g > 0) & np.isfinite(g))
                if idx.size:
                    ts = t[idx]

                    def density_at(sel: Any) -> Callable[..., Any]:
                        ti = ts[sel]
                        return lambda th: self._mixture_fn("df", ti, th)

                    var[idx] = self._delta_var_each(density_at, idx.size)
            sd = self._sd(var, t, on)
            out = g[:, None] * np.exp(k * (sd / g)[:, None])
        out = np.where((g == 0)[:, None], 0.0, out)
        return out if bound == "two-sided" else out[:, 0]

    @keeps_query_shape
    def quantile_cb(
        self,
        p: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        r"""
        Wald confidence bounds on the quantile ``qf(p)``, the B-life at
        ``p`` (the B10 life is ``p = 0.1``).

        As ``Parametric.quantile_cb``'s Wald bound: the delta method on
        the log of the quantile above the support's start (the quantile
        itself for a distribution on the whole line), with the
        quantile's gradient implicit from ``ff(t) = p``, ``dt/dtheta =
        -(dF/dtheta) / f(t)``.

        Parameters
        ----------
        p : array like or scalar
            The probabilities, in (0, 1), whose quantiles are bounded;
            outside it the bound is ``nan``, with one warning (#626).
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : ('two-sided', 'upper', 'lower'), optional
            Defaults to two-sided.
        method : 'wald', optional
            The only bound a mixture has.

        Returns
        -------
        numpy array
            The bound(s), shaped as ``p``; a two-sided bound adds a last
            ``[lower, upper]`` axis.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> wmm.quantile_cb(0.1).round(3)
        array([1.203, 4.997])
        """
        check_alpha_ci(alpha_ci)
        self._check_wald(method, bound)
        probs = np.atleast_1d(np.asarray(p, dtype=float))
        if probs.size == 0:
            return np.empty((0, 2) if bound == "two-sided" else (0,))
        # As qf: NaN, with one warning, outside (0, 1) (#626).
        outside = warn_outside_unit_interval(
            probs, "quantile_cb", closed=False
        )
        ok = ~outside & ~np.isnan(probs)
        if not ok.all():
            out = np.full(
                probs.shape + ((2,) if bound == "two-sided" else ()), np.nan
            )
            if ok.any():
                out[ok] = self.quantile_cb(probs[ok], alpha_ci, bound, method)
            return out
        t = np.asarray(self.qf(probs), dtype=float)
        dens = np.asarray(self.df(t), dtype=float)
        # The variance of F(t) at the fixed t, over f(t)^2: that of t.
        with np.errstate(all="ignore"):
            var = self._delta_var(lambda th: self._mixture_fn("ff", t, th)) / (
                dens**2
            )
        sd = self._sd(var, probs, "qf")
        lower = float(self.dist.support[0])
        alpha, signs = bound_signs(alpha_ci, bound)
        k = signs * z(1 - alpha)
        with np.errstate(all="ignore"):
            if np.isfinite(lower):
                out = lower + (t - lower)[:, None] * np.exp(
                    k * (sd / (t - lower))[:, None]
                )
            else:
                out = t[:, None] + k * sd[:, None]
        return out if bound == "two-sided" else out[:, 0]

    def _require_fit_data(self, what: str) -> None:
        # The likelihood pieces also run mid-fit, before ``params`` is
        # set, so only the data is required; on a restored mixture they
        # used to fail with ``AttributeError: 'NoneType' ... 'n'``.
        if self.data is None:
            self._require_data(what)

    def _require_data(self, what: str) -> None:
        if self.params is None:
            raise ValueError(f"{what} needs a fitted mixture")
        if self.data is None:
            raise ValueError(
                f"{what} needs the data the mixture was fitted to, which a "
                "mixture restored with from_dict does not carry (to_dict "
                "stores only the parameters and weights)."
            )

    def get_plot_data(self, heuristic: str = "Nelson-Aalen") -> Any:
        """The plotting positions and fitted curve that :meth:`plot`
        draws, computed from the fitted data with ``heuristic``.

        As for :meth:`Parametric.get_plot_data`: ``x_`` and ``F`` are
        every row of the plotting positions, suspensions included,
        ``failed`` is a boolean mask of the rows that record a failure
        (the points :meth:`plot` draws), and ``x_censored`` holds the
        suspension times."""
        self._require_data("get_plot_data()")
        return probability_plot_data(
            dist=self.dist,
            ff=self.ff,
            x=self.data.x,
            c=self.data.c,
            n=self.data.n,
            t=self.data.t,
            heuristic=heuristic,
            params=self.params,
        )

    def plot(
        self,
        heuristic: str = "Nelson-Aalen",
        ax: Any = None,
        show_censored: bool = False,
        color: Any = None,
        label: "str | None" = None,
        **kwargs: Any,
    ) -> Axes:
        """
        A method to do a probability plot.

        The points are the failures, at their plotting positions; a
        suspension (right-censored unit) has no point of its own, as for
        :meth:`Parametric.plot`.

        Parameters
        ----------
        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', \
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probability plot will
            be calculated.

        ax: matplotlib.axes.Axes, optional
            The axis onto which the plot will be created. Optional, if not
            provided a new axes will be created.

        show_censored : bool, optional
            Mark the suspension (right-censored) times with ticks along
            the time axis. Defaults to False.

        color : matplotlib color, optional
            The colour of the points and the fitted line. By default the
            next colour of the axes' colour cycle, so that models plotted
            on the same axes differ.

        label : str, optional
            The legend label of the fitted line.

        **kwargs
            Other keyword arguments for the fitted line (a
            ``matplotlib.lines.Line2D``).

        Returns
        -------
        matplotlib.axes.Axes
            a matplotlib axes containing the plot; the x label is "Time"
            unless the axes already have one.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax = wmm.plot(ax=ax, label="two Weibulls")
        >>> ax.get_legend_handles_labels()[1]
        ['two Weibulls']
        >>> plt.close(fig)
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        if self.params is None:
            raise ValueError("Can't plot model that failed to fit")
        self._require_data("plot()")

        heuristic = adjust_heuristic(self.data.c, self.data.t, heuristic)

        d = self.get_plot_data(heuristic=heuristic)

        return draw_probability_plot(
            ax,
            d,
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params),
            title=f"{self.dist.name} Mixture Probability Plot",
            show_censored=show_censored,
            color=color,
            label=label,
            **kwargs,
        )


# The EM steps' public names before v0.24 say where they went (#653).
for _old, _removed in removed_attributes(MIXTURE_EM_ATTRIBUTES).items():
    setattr(MixtureModel, _old, _removed)
