"""
Fitter for shared-frailty proportional-hazards models.

The frailty enters as a random multiplier on the hazard shared within a group,
so the conditional cumulative hazard of an observation is ``u * eta * H0(t)``
with ``eta = exp(beta'Z)``. Integrating the frailty out of a group's
likelihood leaves, with ``D`` the number of events in the group and ``H`` the
sum of ``eta * H0`` over its observations,

.. math::
    \\sum_{\\text{events}} \\log(h_0 \\, \\eta)
    + \\log \\int u^D e^{-u H} \\, dG_\\theta(u).

A Gamma frailty of mean 1 and variance ``theta`` (the default family)
integrates out in closed form,

.. math::
    - \\tfrac{1}{\\theta}\\log\\theta - \\log\\Gamma(\\tfrac{1}{\\theta})
    + \\log\\Gamma(D + \\tfrac{1}{\\theta})
    - (D + \\tfrac{1}{\\theta}) \\log(H + \\tfrac{1}{\\theta}),

with posterior frailty ``(D + 1/theta) / (H + 1/theta)``. A log-normal
frailty (``family="lognormal"``: ``u = exp(w)``, ``w ~ N(0, theta)``) has no
closed form; its integral is computed by adaptive Gauss-Hermite quadrature
(``families.py``). Only observed (``c = 0``) and right-censored (``c = 1``)
data are supported -- the integral relies on that split. The ordinary
parametric MLE is not touched; this module maximises its own marginal
likelihood on a fresh optimiser.
"""

import warnings
from types import SimpleNamespace
from typing import Any, Callable

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pandas as pd
from autograd.extend import defvjp, primitive
from autograd.scipy.special import gammaln as _ad_gammaln
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.univariate.parametric.fitters import at_boundary_maximum
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils import (
    _caller_stacklevel,
    check_covariate_rows,
    finite_covariate_mask,
    xcnt_handler,
)
from surpyval.utils.covariates import coefficient_floor, coefficient_names
from surpyval.utils.fitter_repr import FitterRepr, baseline_name
from surpyval.utils.linalg import numerical_hessian
from surpyval.utils.surpyval_data import SurpyvalData

from .._aliasing import covariate_columns, expand, fit_columns
from .._fit_skeleton import (
    _gradient,
    alias_coefficients,
    check_baseline_support,
    finish_search,
    natural_information,
    optimise_ph,
    require_finite_fit,
)
from .._kinds import PROPORTIONAL_HAZARD
from ..proportional_hazards.cox_likelihood import strata_labels
from ..regression_data import design_matrix_from_df
from .families import (
    check_family,
    lognormal_log_integral,
    lognormal_posterior_mean,
)
from .frailty_model import FrailtyModel


def _make_transforms(dist: Any, k_dist: int) -> tuple[
    Callable[[npt.NDArray, int], npt.NDArray],
    Callable[..., npt.NDArray],
]:
    """Per-parameter (natural <-> unconstrained) maps for the optimiser.

    Baseline parameters follow the distribution's bounds (log for a positive
    parameter, logit for a ``(0, 1)`` parameter, identity otherwise); the
    coefficients are unconstrained; ``theta`` is positive, so log.
    """
    forms = []
    for low, high in dist.bounds[:k_dist]:
        if low == 0 and high is None:
            forms.append("log")
        elif (low, high) == (0, 1):
            forms.append("logit")
        else:
            forms.append("id")

    def to_unc(nat: npt.NDArray, n_beta: int) -> npt.NDArray:
        out = []
        for i, f in enumerate(forms):
            v = nat[i]
            if f == "log":
                out.append(np.log(v))
            elif f == "logit":
                out.append(np.log(v / (1 - v)))
            else:
                out.append(v)
        out.extend(nat[k_dist : k_dist + n_beta])  # beta: identity
        out.append(np.log(nat[-1]))  # theta: log
        return np.array(out, dtype=float)

    def to_nat(unc: npt.NDArray, n_beta: int, ops: Any = None) -> npt.NDArray:
        # ``ops`` as for ``_neg_ll_natural``.
        xp = (ops or _NUMPY).np
        out = []
        for i, f in enumerate(forms):
            v = unc[i]
            if f == "log":
                out.append(xp.exp(v))
            elif f == "logit":
                out.append(1.0 / (1.0 + xp.exp(-v)))
            else:
                out.append(v)
        out.extend(unc[k_dist + i] for i in range(n_beta))
        out.append(xp.exp(unc[-1]))
        return xp.array(out)

    return to_unc, to_nat


def _log_rising_ratio(
    D: npt.NDArray, theta: float, ops: Any = None
) -> npt.NDArray:
    """
    ``log Gamma(D + 1/theta) - log Gamma(1/theta) - D log(1/theta)``,
    computed without cancellation.

    Written directly, the three terms are each of size about
    ``(1/theta) log(1/theta)`` while their difference is at most of order
    ``D^2 theta``, so for a small ``theta`` round-off swamps the answer. For
    an integer ``D`` the ratio is the product ``prod_{k<D} (1 + k theta)``,
    so its log is a sum of ``log1p`` terms; a non-integer ``D`` (fractional
    weights) uses the gamma functions, or their Stirling series once
    ``theta`` is small enough for the gamma functions to cancel. ``ops`` as
    for ``FrailtyFitter._neg_ll_natural``.
    """
    ops = ops or _NUMPY
    xp = ops.np
    D = np.asarray(D, dtype=float)
    integer = np.isclose(D, np.round(D), rtol=0.0, atol=1e-9)
    # Written without assignment into an array, so that autograd can
    # differentiate it in theta: each form is evaluated where it is used
    # (the integer one at 0 on the other rows), and the rows pick theirs.
    d_int = np.where(integer, np.round(D), 0.0).astype(int)
    k = np.arange(max(int(d_int.max(initial=0)), 0), dtype=float)
    cumulative = xp.concatenate([np.zeros(1), xp.cumsum(xp.log1p(k * theta))])
    out = cumulative[d_int]
    if (~integer).any():
        d = D
        if theta < 1e-6:
            # the Stirling series in theta = 1/a (Bernoulli polynomials),
            # where the gamma functions below would cancel
            d2 = d * (d - 1.0)
            other = (
                d2 / 2.0 * theta
                - d2 * (2.0 * d - 1.0) / 12.0 * theta**2
                + d2**2 / 12.0 * theta**3
            )
        else:
            it = 1.0 / theta
            other = ops.gammaln(d + it) - ops.gammaln(it) - d * xp.log(it)
        out = xp.where(integer, out, other)
    return out


def _group_frailty_ll(
    D: npt.NDArray,
    H: npt.NDArray,
    theta: float,
    ops: Any = None,
    family: str = "gamma",
) -> npt.NDArray:
    """
    Each group's frailty term of the marginal log-likelihood,
    ``log E[u^D exp(-u H)]``. For the gamma frailty it is

    .. math::
        -\\tfrac{1}{\\theta}\\log\\theta - \\log\\Gamma(\\tfrac{1}{\\theta})
        + \\log\\Gamma(D + \\tfrac{1}{\\theta})
        - (D + \\tfrac{1}{\\theta}) \\log(H + \\tfrac{1}{\\theta}),

    rearranged so that it stays accurate as ``theta -> 0``: it equals
    ``log_rising_ratio(D, theta) - (D + 1/theta) log1p(H theta)``, which
    tends to ``-H`` -- the no-frailty (proportional-hazards) contribution.
    The log-normal frailty's has no closed form and is computed by
    quadrature (``families.lognormal_log_integral``); it tends to ``-H``
    too. ``ops`` as for ``FrailtyFitter._neg_ll_natural``.
    """
    xp = (ops or _NUMPY).np
    if family == "lognormal":
        return lognormal_log_integral(D, H, theta, ops)
    if theta * max(xp.max(H), np.max(D), 1.0) ** 2 < _EPS:
        # theta is too small to change any group's term by more than
        # rounding (each correction to the no-frailty value -H is of order
        # theta H^2, theta D H or theta D^2), and the terms below divide by
        # it: the limit itself. (Its derivatives in beta must stay finite
        # there too, for the fit's check of its answer, #392.)
        return -1.0 * H
    return _log_rising_ratio(D, theta, ops) - (
        D * xp.log1p(H * theta) + xp.log1p(H * theta) / theta
    )


_EPS = float(np.finfo(float).eps)


def _settle_on_zero_variance(fun: Callable, res: Any) -> Any:
    """Carry a frailty variance that the gradient search left on its way to
    the boundary at 0 the rest of the way there.

    ``theta`` is searched as ``log theta``, whose boundary is at minus
    infinity, and the gradient in ``log theta`` is ``theta`` times the one
    in ``theta``: on data with no detectable frailty BFGS stops at a small
    ``theta`` (1e-7 to 1e-6) once that product is below its tolerance,
    with the likelihood still rising towards ``theta = 0`` by about
    ``theta`` times its slope there (up to 1e-5 nats). Nelder-Mead, used
    before (#515), kept going to ``theta`` of 1e-9 or less. Steps of a
    factor ``e^10`` in ``theta`` carry on while the likelihood rises; once
    ``theta`` is small enough for every group term to be its no-frailty
    limit (``_group_frailty_ll``) nothing changes and the steps stop. A
    ``theta`` at an interior maximum is left where it is, since the first
    step lowers the likelihood.
    """
    u = np.array(res.x, dtype=float)
    best = float(res.fun)
    for _ in range(10):
        trial = u.copy()
        trial[-1] -= 10.0
        f = float(fun(trial))
        if not (np.isfinite(f) and f < best):
            break
        u, best = trial, f
    res.x, res.fun = u, best
    return res


def _at_zero_variance(fun: Callable, res: Any, n_obs: float) -> bool:
    """Whether the fit is a maximum on the boundary ``theta = 0``, where
    the model is the one without frailty, so that ``theta`` is left out of
    the check of its gradient and Hessian (``judge_search``'s ``held``).

    ``theta`` is searched as ``log theta``, whose boundary is at minus
    infinity, where the Hessian is singular (see ``at_boundary_maximum``).
    The fit is there when the likelihood does not change, to rounding, as
    ``theta`` moves further towards 0 (a factor ``e^10``, as
    ``_settle_on_zero_variance`` steps), and a maximum there when moving
    ``theta`` off it, to ``1e-6`` (its natural unit being 1), does not
    raise the likelihood.
    """
    u = np.array(res.x, dtype=float)
    toward, away = u.copy(), u.copy()
    toward[-1] -= 10.0
    away[-1] = np.log(1e-6)
    return at_boundary_maximum(fun, u, toward, away, 1e-6, n_obs)


@primitive
def _group_sum(
    values: npt.NDArray, inv: npt.NDArray, n_groups: int
) -> npt.NDArray:
    """Each group's sum of ``values`` (``inv`` the group of each), as
    ``np.bincount``, which autograd cannot differentiate on its own."""
    return np.bincount(inv, weights=values, minlength=n_groups)


# The derivative of a group's sum in each of its values is 1: a gradient
# with respect to the sums spreads back to each value from its group.
defvjp(_group_sum, lambda ans, values, inv, n_groups: lambda g: g[inv])

# The functions the likelihood is written in: numpy's for the search, which
# evaluates it thousands of times, and autograd's, for the fit's check of
# its answer (#392), which differentiates it. They compute the same values;
# autograd's wrappers only cost time where nothing is differentiated.
_NUMPY = SimpleNamespace(
    np=np,
    gammaln=gammaln,
    group_sum=lambda v, inv, n: np.bincount(inv, weights=v, minlength=n),
)
_AUTOGRAD = SimpleNamespace(np=anp, gammaln=_ad_gammaln, group_sum=_group_sum)


def grouped_data(x: Any, Z: Any, c: Any, n: Any, groups: Any) -> tuple[
    npt.NDArray,
    "npt.NDArray | None",
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """The data of a shared-frailty fit, checked: ``(x, Z, c, w, labels,
    inv)``, with ``Z`` ``None`` when not given, ``w`` the counts, and
    ``labels`` the distinct groups and ``inv`` each row's index into them.

    Only observed (c=0) and right-censored (c=1) rows are taken. Rows with
    a missing or infinite covariate, or a missing group label, are dropped
    with a warning; at least one event and two groups are required.
    """
    # Through the data handler first, in the caller's row order: the
    # documented ragged form ``[10, [11, 13], ...]`` is not a
    # rectangular array, and ``np.asarray(x, dtype=float)`` on it raised
    # a raw numpy error.
    x_h, c_h, n_h, _ = xcnt_handler(x, c, n, group_and_sort=False)
    c = np.asarray(c_h, dtype=int).ravel()
    if not np.all(np.isin(c, (0, 1))):
        raise ValueError(
            "Frailty fitting supports only observed (c=0) and "
            "right-censored (c=1) data."
        )
    x = np.asarray(x_h, dtype=float)
    if x.ndim == 2:
        # Two columns with no interval row: xl == xr on every row.
        x = x[:, 0]
    n_obs = x.shape[0]
    w = np.asarray(n_h, dtype=float).ravel()
    if groups is None:
        raise ValueError("'groups' (a cluster label per row) is required.")
    # Read element by element: a ``None`` label in an array of numbers
    # made ``np.unique`` raise a TypeError, and a NaN label was kept as
    # a group of its own (#388).
    groups, missing = strata_labels(
        np.asarray(groups, dtype=object).ravel().tolist()
    )
    if groups.shape[0] != n_obs:
        raise ValueError(
            "'groups' has {} label(s) but there are {} observations; "
            "give one group label per row.".format(groups.shape[0], n_obs)
        )

    if Z is not None:
        Zm = np.atleast_2d(np.asarray(Z, dtype=float))
        if Zm.shape[0] != n_obs and Zm.shape[1] == n_obs:
            # A single covariate given as a row.
            Zm = Zm.T
        check_covariate_rows(Zm, n_obs)
        keep = finite_covariate_mask(Zm)
        if not keep.all():
            x, c, w, groups, missing, Zm = (
                a[keep] for a in (x, c, w, groups, missing, Zm)
            )
            n_obs = x.shape[0]
    if missing.any():
        # A row without a group has no frailty to share: it is dropped,
        # as a row with a missing stratum label is in a stratified Cox
        # model.
        if missing.all():
            raise ValueError(
                "Every group label is missing; there is nothing to fit."
            )
        warnings.warn(
            "Dropped {} of {} rows with a missing group label.".format(
                int(missing.sum()), n_obs
            ),
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
        keep = ~missing
        x, c, w, groups = (a[keep] for a in (x, c, w, groups))
        if Z is not None:
            Zm = Zm[keep]
        n_obs = x.shape[0]

    if int((c == 0).sum()) == 0:
        raise ValueError("At least one event (c=0) is required.")

    labels, inv = np.unique(groups, return_inverse=True)
    n_groups = labels.shape[0]
    if n_groups < 2:
        raise ValueError(
            "The frailty variance is not identifiable from a single "
            "group; at least two groups are required."
        )

    return x, (Zm if Z is not None else None), c, w, labels, inv


class FrailtyFitter(FitterRepr):
    """Configured fitter for a shared-frailty PH model on one distribution."""

    #: The ``repr`` (#614)
    fitter_kind = "shared frailty fitter"

    def _repr_details(self) -> "list[str]":
        return [*baseline_name(self), self.family + " frailty"]

    def __init__(self, name: str, dist: Any, family: str = "gamma") -> None:
        family = check_family(family)
        self.name = name
        self.dist = dist
        self.family = family
        self.k_dist = len(dist.parameter_names)

    @staticmethod
    def create(distribution: Any, family: str = "gamma") -> "FrailtyFitter":
        return FrailtyFitter(
            f"{distribution.name}Frailty", distribution, family
        )

    # -- likelihood --------------------------------------------------------

    def _neg_ll_natural(
        self,
        nat: npt.NDArray,
        x: npt.NDArray,
        c: npt.NDArray,
        w: npt.NDArray,
        eta_Z: npt.NDArray,
        inv: npt.NDArray,
        n_beta: int,
        ops: Any = None,
    ) -> float:
        """Marginal negative log-likelihood in natural parameters.

        ``eta_Z`` is the covariate matrix (n_obs x n_beta); ``inv`` maps each
        observation to its group index; ``w`` are observation weights.
        ``ops`` holds the functions it is computed with: numpy's (the
        default) or, to be differentiated, autograd's (``_AUTOGRAD``).
        """
        ops = ops or _NUMPY
        xp = ops.np
        dist_params = nat[: self.k_dist]
        beta = nat[self.k_dist : self.k_dist + n_beta]
        theta = nat[-1]

        H0 = self.dist.Hf(x, *dist_params)
        h0 = self.dist.hf(x, *dist_params)
        eta = xp.exp(xp.dot(eta_Z, beta)) if n_beta else np.ones_like(x)

        event = c == 0
        ll = xp.sum(w[event] * (xp.log(h0[event]) + xp.log(eta[event])))

        n_groups = inv.max() + 1
        D = np.bincount(inv, weights=w * event, minlength=n_groups)
        H = ops.group_sum(w * eta * H0, inv, n_groups)
        ll = ll + xp.sum(_group_frailty_ll(D, H, theta, ops, self.family))
        return -ll

    # -- fit ---------------------------------------------------------------

    @dataframe_covariates
    def fit(
        self,
        x: Any,
        Z: Any = None,
        c: Any = None,
        n: Any = None,
        groups: Any = None,
        init: Any = None,
    ) -> FrailtyModel:
        """Fit the shared-frailty model.

        Parameters
        ----------
        x : array_like
            Observed times.
        Z : array_like, optional
            Covariates ``(n_obs, p)``. Omit for a frailty model with no
            covariates (a pure random-effects survival model). Rows with a
            missing or infinite covariate are dropped (with their group
            labels), with a warning.
        c : array_like, optional
            Censoring flags: ``0`` event, ``1`` right-censored (the only two
            supported). Defaults to all events.
        n : array_like, optional
            Observation weights / counts. Defaults to 1.
        groups : array_like
            The group (cluster) label of each observation. Required. Rows
            with a missing label (``None``, ``NaN`` or pandas ``NA``) are
            dropped, with a warning.
        init : array_like, optional
            Optional initial natural parameters ``[*dist, *beta, theta]``.

        Returns
        -------
        FrailtyModel
            The fitted model: the baseline ``dist_params``, the coefficients
            ``beta``, the frailty variance ``theta`` and each group's
            posterior frailty.

        Examples
        --------
        Thirty groups of six units, each group with its own gamma frailty
        (mean 1, variance 0.5):

        >>> import numpy as np
        >>> from surpyval import WeibullFrailty
        >>> rng = np.random.default_rng(4)
        >>> groups = np.repeat(np.arange(30), 6)
        >>> u = rng.gamma(2.0, 0.5, 30)[groups]
        >>> Z = rng.binomial(1, 0.5, (180, 1))
        >>> H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
        >>> x = 10 * H**0.5  # Weibull baseline, alpha 10 and beta 2
        >>> model = WeibullFrailty.fit(x, Z=Z, groups=groups)
        >>> model.beta.round(3), round(model.theta, 3)
        (array([0.399]), 0.432)
        """
        x, Zm, c, w, labels, inv = grouped_data(x, Z, c, n, groups)
        # The times inside the baseline's support, as for the other
        # parametric regressions (#565)
        check_baseline_support(
            self, SurpyvalData(x, c, w, None, group_and_sort=False)
        )
        n_obs = x.shape[0]
        n_groups = labels.shape[0]
        Zc = np.zeros((n_obs, 0)) if Zm is None else Zm
        n_beta = Zc.shape[1]
        feature_names = None
        # Coefficients the data cannot determine are aliased (#476): a
        # constant column where the baseline's scale is the intercept, or
        # a linear combination of the others. The fit runs on the other
        # columns, and the model reports them as nan.
        p_all = n_beta
        # Each coefficient named by its column, or coef_j (#614)
        coefs = coefficient_names(
            n_beta, fit_columns(), [*self.dist.parameter_names, "theta"]
        )
        aliased = getattr(
            alias_coefficients(
                self,
                PROPORTIONAL_HAZARD,
                Zc,
                w,
                {},
                {name: j for j, name in enumerate(coefs)},
            ),
            "aliased",
            (),
        )
        kept = np.array(
            [j for j in range(n_beta) if coefs[j] not in aliased],
            dtype=int,
        )
        if len(aliased):
            Zc = Zc[:, kept]
            n_beta = kept.size

        to_unc, to_nat = _make_transforms(self.dist, self.k_dist)

        if init is None:
            base = self.dist.fit(x=x, c=c, n=w).params
            init_nat = np.concatenate(
                [np.asarray(base, float), np.zeros(n_beta), [0.5]]
            )
        else:
            init_nat = np.asarray(init, dtype=float).ravel()
            n_params = self.k_dist + p_all + 1
            if init_nat.shape[0] != n_params:
                raise ValueError(
                    "`init` has {} value(s) but the model has {} parameters: "
                    "the {} distribution parameter(s), {} coefficient(s) and "
                    "theta, in that order.".format(
                        init_nat.shape[0], n_params, self.k_dist, p_all
                    )
                )
            if len(aliased):
                init_nat = np.concatenate(
                    [
                        init_nat[: self.k_dist],
                        init_nat[self.k_dist + kept],
                        init_nat[-1:],
                    ]
                )

        def obj_unc(u: npt.NDArray) -> float:
            nat = to_nat(u, n_beta)
            return self._neg_ll_natural(nat, x, c, w, Zc, inv, n_beta)

        def obj_traced(u: npt.NDArray) -> Any:
            nat = to_nat(u, n_beta, _AUTOGRAD)
            return self._neg_ll_natural(
                nat, x, c, w, Zc, inv, n_beta, _AUTOGRAD
            )

        u0 = to_unc(init_nat, n_beta)
        # Each coefficient searched and judged in its own covariate's
        # units (#577)
        floor = coefficient_floor(
            u0.size, [(self.k_dist + i, i) for i in range(n_beta)], Zc
        )
        res = None
        with np.errstate(all="ignore"):
            # The gradient ladder first, on the likelihood's exact gradient,
            # as for the AFT and PO fits (#499): Nelder-Mead took 1800 to
            # 4000 evaluations and BFGS then differenced the gradient, 44.6 s
            # for a gamma baseline at 10 000 rows (#515). Its answer is kept
            # when it is a verified optimum; otherwise the derivative-free
            # search below runs as before.
            if _gradient(obj_traced, u0) is not None:
                fast = optimise_ph(obj_traced, u0, quiet=True, floor=floor)
                if np.isfinite(fast.fun) and not getattr(
                    fast, "stopped_short", False
                ):
                    res = _settle_on_zero_variance(obj_unc, fast)
                    converged = True
            if res is None:
                res = minimize(
                    obj_unc,
                    u0,
                    method="Nelder-Mead",
                    options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
                )
                polished = minimize(obj_unc, res.x, method="BFGS")
                # BFGS often stops on "precision loss" at the optimum
                # Nelder-Mead already found; keep whichever is better, and
                # say so only if neither converged.
                converged = bool(res.success or polished.success)
                if np.isfinite(polished.fun) and (
                    polished.fun <= res.fun or not np.isfinite(res.fun)
                ):
                    res = polished
        require_finite_fit(float(res.fun))
        # One warning: a coefficient with no finite maximum (a level with no
        # events, #392), or else a search that did not reach a verified
        # maximum -- with theta left out of that check where it is a
        # maximum on its boundary, 0.
        res.stopped_short = not converged
        n_weighted = float(np.sum(w))
        held = (
            (res.x.size - 1,)
            if _at_zero_variance(obj_unc, res, n_weighted)
            else ()
        )
        verdict = finish_search(
            obj_traced,
            res,
            [
                (self.k_dist + i, int(kept[i]) if len(aliased) else i)
                for i in range(n_beta)
            ],
            u0,
            n_weighted,
            held=held,
            floor=floor,
        )
        res = verdict.res
        no_maximum, derivatives = verdict.no_maximum, verdict.derivatives
        nat = to_nat(res.x, n_beta)

        dist_params = nat[: self.k_dist]
        beta = nat[self.k_dist : self.k_dist + n_beta]
        theta = float(nat[-1])

        # Posterior (empirical-Bayes) frailty per group.
        H0 = self.dist.Hf(x, *dist_params)
        eta = np.exp(Zc @ beta) if n_beta else np.ones_like(x)
        D = np.bincount(inv, weights=w * (c == 0), minlength=n_groups)
        H = np.bincount(inv, weights=w * eta * H0, minlength=n_groups)
        if self.family == "lognormal":
            post = lognormal_posterior_mean(D, H, theta)
        else:
            # (D + 1/theta) / (H + 1/theta), written to stay finite (and
            # tend to 1) as theta -> 0
            post = (1.0 + D * theta) / (1.0 + H * theta)

        # Covariance of the natural parameters: the inverse of the exact
        # Hessian the check computed (#392), converted from the search
        # space; a numerical one where there is none (no maximum, or a
        # Hessian that is not positive definite, as with the variance at
        # its limit of 0).
        parameter_names = list(self.dist.parameter_names)
        parameter_names += [coefs[j] for j in kept] if len(aliased) else coefs
        parameter_names += ["theta"]

        def nll_nat(v: npt.NDArray) -> float:
            return self._neg_ll_natural(v, x, c, w, Zc, inv, n_beta)

        exact = None
        if not no_maximum:
            exact = natural_information(
                derivatives, lambda u: to_nat(u, n_beta, _AUTOGRAD), res.x
            )
        covariance = None
        with np.errstate(all="ignore"):
            try:
                if exact is not None:
                    Hmat = exact
                else:
                    steps = 1e-5 * np.maximum(np.abs(nat), 1.0)
                    Hmat = numerical_hessian(nll_nat, nat, step=steps)
                cov = np.linalg.inv(Hmat)
                if np.all(np.isfinite(cov)):
                    covariance = cov
            except np.linalg.LinAlgError:
                covariance = None

        if len(aliased):
            # Back to one entry per column of Z, nan where aliased.
            beta = expand(beta, kept, p_all)
            parameter_names = list(self.dist.parameter_names)
            parameter_names += coefs
            parameter_names += ["theta"]
            if covariance is not None:
                where = np.r_[
                    np.arange(self.k_dist),
                    self.k_dist + kept,
                    self.k_dist + p_all,
                ]
                full = np.full((len(parameter_names),) * 2, np.nan)
                full[np.ix_(where, where)] = covariance
                covariance = full

        model = FrailtyModel()
        model.family = self.family
        model.dist = self.dist
        model.dist_params = np.asarray(dist_params, float)
        model.beta = np.asarray(beta, float)
        model.theta = theta
        model.k_dist = self.k_dist
        model.feature_names = feature_names
        model.group_labels = list(labels)
        model.frailties = {str(lab): float(u) for lab, u in zip(labels, post)}
        model._covariance = covariance
        model.parameter_names = parameter_names
        model.k = len(parameter_names) - len(aliased)
        model.n_obs = n_obs
        model.n_events = int((c == 0).sum())
        model.n_events_weighted = float(w[c == 0].sum())
        model.n_obs_weighted = float(w.sum())
        model.n_groups = n_groups
        model._neg_ll = float(res.fun)
        model.maximum = verdict.maximum
        # (the columns of Z whose coefficients were estimated)
        model._fit_data = {"x": x, "c": c, "w": w, "Z": Zc, "inv": inv}
        return model

    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        group_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        formula: "str | None" = None,
        init: Any = None,
    ) -> FrailtyModel:
        """Fit from a :class:`pandas.DataFrame` naming the columns.

        Either ``Z_cols`` or ``formula`` describes the covariates (or neither,
        for a no-covariate frailty model); ``group_col`` names the cluster
        column. Covariate names / the formula transformer are retained so the
        fitted model predicts from raw-covariate DataFrames.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col : str
            The column of times.
        group_col : str
            The column of group (cluster) labels; rows with a missing label
            are dropped, with a warning.
        Z_cols : str or list of str, optional
            The covariate columns.
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        formula : str, optional
            A formula (formulaic syntax) for the covariates, instead of
            ``Z_cols``.
        init : array_like, optional
            As for :meth:`fit`.

        Returns
        -------
        FrailtyModel
            The fitted model.
        """
        x = df[x_col].values
        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values
        groups = df[group_col].values

        feature_names = None
        model_spec = None
        if Z_cols is None and formula is None:
            Z = None
        else:
            Z, feature_names, model_spec = design_matrix_from_df(
                df, Z_cols=Z_cols, formula=formula
            )

        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(x, Z=Z, c=c, n=n, groups=groups, init=init)
        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec
        return model
