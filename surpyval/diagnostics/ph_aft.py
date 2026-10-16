"""``diagnose()`` for the parametric proportional hazards and accelerated
failure time models (#869).

Two families, one module: :data:`PH_FAMILY` for a
``ParametricRegressionModel`` of kind ``"Proportional Hazard"``
(``WeibullPH``, ``LogNormalPH``, ..., ``PH(dist)``) and :data:`AFT_FAMILY`
for one of kind ``"Accelerated Failure Time"`` (``WeibullAFT``, ...,
``AFT(dist)``), both fitted with the ``exp(beta'Z)`` link to one row per
unit (``fit`` / ``fit_from_df``). They share the machinery below: every
test is a likelihood-ratio test of the fitted model against a larger one,
whose likelihood is the package's own (``regression_neg_ll``, with every
kind of censoring and truncation the fit took), maximised here from the
fitted parameters.

The checks of a proportional hazards model
------------------------------------------
``proportional_hazards``
    For each covariate, the likelihood-ratio test (1 df) of a coefficient
    that steps at the median failure time ``tau``: the model with hazard
    ``h0(t) exp(beta'Z + [t > tau] (g0 + gamma Z_j))`` against the one with
    ``gamma = 0``; and the joint ``GLOBAL`` test of a step in every
    coefficient at once (one df per covariate). Both models have the step
    ``g0`` in the level of the baseline at ``tau``, which makes the test
    independent of where a covariate's origin is (principle 6): without
    it, a step in the coefficient of ``Z_j`` is also a step in the hazard
    of every unit with ``Z_j`` far from 0, which the fitted baseline must
    absorb, and a fading effect of a covariate far from 0 (R's Karnofsky
    score) goes unseen. The cumulative hazard has a closed form for every
    baseline, so the test is exact for the model, whatever the censoring
    and truncation. ``tau`` is the lower weighted median of the failure
    times (an interval-censored failure counts at its midpoint, a
    left-censored one at its time), a cut chosen without the covariates.
    Drawn as each covariate's coefficient in each quarter of the failure
    times (the model with a step in it, and in the baseline, at each
    quartile), with its Wald interval, against the fitted coefficient.
``functional_form``
    For each covariate with three or more distinct values, the
    likelihood-ratio test of adding its square (as the model's covariates
    enter it, about the model's ``center``): whether it acts linearly on
    the log hazard. Drawn as the martingale residuals against the
    covariate, with a lowess trend.
``influence``
    For each covariate, the largest ``|dfbeta| / se``: the one-step
    change ``I^-1 U_i`` of the coefficient when one unit is dropped (the
    observed information ``I`` and the unit's score ``U_i``; a row with
    count ``n`` is ``n`` units), in standard errors. It warns when one
    unit moves a coefficient by more than one standard error; a screen,
    never a fail.
``outliers``
    The deviance residuals of the martingale residuals ``delta - H(x | Z)
    + H(tl | Z)``: a row is an outlier when its ``|deviance residual|``
    exceeds the two-sided Bonferroni cutoff ``z_{1 - alpha / (2 N)}``,
    ``N`` the number of units, as for the Cox family. It warns when any
    row is flagged; never a fail.
``baseline_distribution``
    Two screens, never fails: the AIC of the model against that of each
    other built-in baseline (``ExponentialPH``, ``NormalPH``,
    ``WeibullPH``, ``GumbelPH``, ``LogisticPH``, ``LogNormalPH``,
    ``GammaPH``) refitted to the same data, as ``fit_best`` compares
    distributions; and against the accelerated failure time model of the
    same baseline (not for the Weibull and the exponential, whose PH and
    AFT models are the same model). Each warns when the other model's AIC
    is more than :data:`AIC_MARGIN` (2) lower; its ``statistic`` is the
    fitted model's AIC less the other's (positive: the other is lower).
    A candidate whose fit is not a verified maximum is set aside, as
    ``fit_best`` sets it aside, and a note says so.
    Drawn as the Cox-Snell residuals ``H(x | Z)`` on exponential
    probability paper, which a model whose baseline fits puts on the line
    ``y = x``, and the AIC of each candidate. As for a Cox model, the
    Cox-Snell residuals have no useful test of their own.

The checks of an accelerated failure time model
-----------------------------------------------
``error_distribution``
    Drawn as the standardised residuals ``e^{beta'Z} x`` (the time on the
    baseline's scale, which the model says follows the baseline
    distribution) on the baseline's probability paper, with the fitted
    baseline's line. The same two screens as the PH family's
    ``baseline_distribution``: the AIC against the other built-in error
    distributions (``WeibullAFT``, ``LogNormalAFT``, ...) and against the
    proportional hazards model of the same baseline.
``common_shape``
    For each categorical covariate (two to five distinct values, or the
    dummy columns a formula codes one factor with), the likelihood-ratio
    test of one shape for every level against a shape per level (``L - 1``
    df for ``L`` levels): the model with the shape parameter (the
    Weibull's ``beta``, the LogNormal's ``sigma``, ...; see :data:`SHAPE`)
    times ``exp(delta_l)`` in level ``l``. Drawn as the standardised
    residuals of each level on the baseline's paper: one line for all is
    a common shape.
``functional_form``
    As for the PH family, of log time: the likelihood-ratio test of adding
    a covariate's square, and the martingale residuals against it.

The covariates' kinds are the Cox family's: two distinct values is
*binary*, three to five *categorical*, more *continuous*.
"""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2, norm

from surpyval.univariate.regression._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
)
from surpyval.univariate.regression._likelihood import regression_neg_ll

from .calibration import calibration_advice, calibration_check
from .cox import CATEGORICAL_LEVELS, INFLUENCE_LIMIT, _fmt, _source
from .registry import AdviceRule, Check, Context, Family, Panel, register

#: Another model whose AIC is more than this below the fitted model's
#: makes the AIC screens warn (Burnham and Anderson's rule of thumb: a
#: difference of 2 or less is no real evidence either way).
AIC_MARGIN = 2.0
#: The built-in baselines, each with its ``<name>PH`` and ``<name>AFT``
#: fitter: the candidates of the AIC comparisons.
BASELINES = (
    "Exponential",
    "Normal",
    "Weibull",
    "Gumbel",
    "Logistic",
    "LogNormal",
    "Gamma",
)
#: The shape parameter of each built-in baseline that has one, the one
#: the ``common_shape`` check lets vary by level: the parameter left when
#: the acceleration has taken the scale (the exponential has none).
SHAPE = {
    "Weibull": "beta",
    "LogNormal": "sigma",
    "Gamma": "alpha",
    "Normal": "sigma",
    "Gumbel": "sigma",
    "Logistic": "sigma",
}
#: The baselines whose proportional hazards and accelerated failure time
#: models are the same model (the Weibull family's: a power of time times
#: a scale).
SAME_PH_AFT = ("Weibull", "Exponential", "Rayleigh")
#: The number of time windows of the proportional-hazards plot.
WINDOWS = 4

_NUMBERS = {3: "three", 4: "four", 5: "five", 6: "six"}

_PH = "Proportional Hazard"
_AFT = "Accelerated Failure Time"


# ---------------------------------------------------------------------------
# The fitted model, the data and the likelihood
# ---------------------------------------------------------------------------
def _base(ctx: Context) -> SimpleNamespace:
    """What every check uses: the fitter, the data, the covariates as the
    parameters are relative to (``Z - center``), the parameters and which
    of them were estimated."""

    def compute() -> SimpleNamespace:
        m = ctx.model
        data = m.data
        raw = np.asarray(data.Z, dtype=float)
        if raw.ndim == 1:
            raw = raw[:, None]
        Z = raw
        if m.center is not None and np.any(m.center):
            Z = raw - np.asarray(m.center, dtype=float)
        params = np.asarray(m._eval_params(), dtype=float)
        names = list(m.parameter_names)
        free = np.array([name not in m.fixed for name in names])
        bounds = list(m.model.bounds) + [(None, None)] * (
            len(params) - m.k_dist
        )
        x = np.asarray(data.x, dtype=float)
        lo = x if x.ndim == 1 else x[:, 0]
        hi = x if x.ndim == 1 else x[:, -1]
        return SimpleNamespace(
            model=m,
            fitter=m.model,
            dist=m.distribution,
            name=m.distribution.name,
            data=data,
            raw=raw,
            Z=Z,
            params=params,
            free=free,
            bounds=bounds,
            k_dist=int(m.k_dist),
            n=np.asarray(data.n, dtype=float),
            c=np.asarray(data.c),
            lo=lo,
            hi=hi,
            tl=np.asarray(data.tl, dtype=float),
            tr=np.asarray(data.tr, dtype=float),
        )

    return ctx.cached("base", compute)


def _view(b: SimpleNamespace, Z: Any, weights: Any = None) -> SimpleNamespace:
    """The model's data as ``regression_neg_ll`` reads it, with the
    covariates ``Z`` (one row per row of the data) and the counts times
    ``weights`` (which may be an autograd box). A left truncation at or
    below the start of the baseline's support, with no right truncation,
    truncates nothing (its probability is 1) and is left out: its
    derivatives are ``0 * inf`` at 0 (a Weibull's ``(0 / alpha)^beta``)."""
    data = b.data
    lower = float(b.fitter.support[0])
    truncated = data.truncated_mask & ~(
        (data.tl <= lower) & ~np.isfinite(data.tr)
    )
    n = np.asarray(data.n, dtype=float)
    if weights is not None:
        n = n * weights
    return SimpleNamespace(
        x_o=data.x_o,
        x_r=data.x_r,
        x_l=data.x_l,
        x_il=data.x_il,
        x_ir=data.x_ir,
        x_tl=data.tl[truncated],
        x_tr=data.tr[truncated],
        Z_o=Z[data.mask_o],
        Z_r=Z[data.mask_r],
        Z_l=Z[data.mask_l],
        Z_i=Z[data.mask_i],
        Z_t=Z[truncated],
        n_o=n[data.mask_o],
        n_r=n[data.mask_r],
        n_l=n[data.mask_l],
        n_i=n[data.mask_i],
        n_t=n[truncated],
    )


def _neg_ll(model: Any, view: Any) -> Callable[[Any], Any]:
    """The negative log-likelihood of ``model`` (anything with ``log_df``,
    ``log_sf``, ``log_ff`` and ``ff`` in ``(x, Z, *params)``) on
    ``view``, as a function of the parameter vector."""

    def f(theta: Any) -> Any:
        return regression_neg_ll(
            model, view, *[theta[i] for i in range(len(theta))]
        )

    return f


def _transform(bounds: list) -> tuple[Callable, Callable]:
    """Maps between the parameters within ``bounds`` and the whole line,
    for the search: the identity, a log (one bound) or a logit (two)."""
    import autograd.numpy as anp

    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], float)
    low, high = np.isfinite(lo), np.isfinite(hi)
    both, only_lo, only_hi = low & high, low & ~high, ~low & high
    lo0, hi0 = np.where(low, lo, 0.0), np.where(high, hi, 0.0)
    width = np.where(both, hi0 - lo0, 1.0)

    def to_u(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            u = np.array(theta, dtype=float)
            u = np.where(only_lo, np.log(theta - lo0), u)
            u = np.where(only_hi, np.log(hi0 - theta), u)
            p = (theta - lo0) / width
            u = np.where(both, np.log(p) - np.log1p(-p), u)
        return u

    def from_u(u: Any) -> Any:
        e = anp.exp(anp.where(only_lo | only_hi, u, 0.0))
        s = 1.0 / (1.0 + anp.exp(-anp.where(both, u, 0.0)))
        out = anp.where(only_lo, lo0 + e, u)
        out = anp.where(only_hi, hi0 - e, out)
        return anp.where(both, lo0 + width * s, out)

    return to_u, from_u


class _Fit(SimpleNamespace):
    """A maximum found by :func:`_maximise`: ``theta``, ``value`` (the
    negative log-likelihood there) and ``ok`` (a verified maximum)."""

    theta: np.ndarray
    value: float
    ok: bool


def _maximise(
    f: Callable[[Any], Any], start: Any, free: Any, bounds: list
) -> _Fit:
    """Minimise the negative log-likelihood ``f`` over the parameters
    marked ``free`` (the others held at ``start``), within ``bounds``,
    from ``start``: BFGS, then if need be Newton's method with a trust
    region, on the exact (autograd) gradient and Hessian, in the
    transformed space of :func:`_transform`. ``ok`` is True when the
    search ends at a zero gradient with a positive-definite Hessian."""
    import autograd.numpy as anp
    from autograd import grad, hessian
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=float)
    idx = np.flatnonzero(free)
    if not idx.size:
        return _Fit(theta=start, value=float(f(start)), ok=True)
    to_u, from_u = _transform([bounds[i] for i in idx])
    E = np.zeros((start.size, idx.size))
    E[idx, np.arange(idx.size)] = 1.0
    held = np.where(np.asarray(free), 0.0, start)

    def full(u: Any) -> Any:
        return held + anp.dot(E, from_u(u))

    def obj(u: Any) -> Any:
        return f(full(u))

    g, h = grad(obj), hessian(obj)
    u0 = to_u(start[idx])
    best = None
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # BFGS on the exact gradient first, started with the exact
        # Hessian: from near the maximum it needs a few cheap steps, where
        # each Newton step needs a Hessian (a dozen gradients' work); the
        # Hessian is taken again at the end, to verify the maximum.
        # Newton's method with a trust region if BFGS does not get there.
        for method in ("BFGS", "trust-exact"):
            try:
                if method == "BFGS":
                    options: dict = {"gtol": 1e-7}
                    curvature0 = np.asarray(h(u0), dtype=float)
                    if np.all(np.isfinite(curvature0)) and np.all(
                        np.linalg.eigvalsh(curvature0) > 0
                    ):
                        # Started with the exact curvature, BFGS takes a
                        # few Newton-like steps instead of dozens.
                        inverse = np.linalg.inv(curvature0)
                        options["hess_inv0"] = 0.5 * (inverse + inverse.T)
                    res = minimize(
                        obj, u0, jac=g, method=method, options=options
                    )
                else:
                    res = minimize(obj, u0, jac=g, hess=h, method=method)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                continue
            value = float(res.fun)
            if not np.isfinite(value):
                continue
            # trust-exact returns the gradient and Hessian at its answer
            gradient = np.asarray(
                res.jac if res.get("jac") is not None else g(res.x),
                dtype=float,
            )
            try:
                curvature = np.asarray(
                    res.hess if res.get("hess") is not None else h(res.x),
                    dtype=float,
                )
                definite = bool(np.all(np.linalg.eigvalsh(curvature) > 0))
            except (ValueError, np.linalg.LinAlgError):
                definite = False
            ok = (
                bool(np.all(np.isfinite(gradient)))
                and float(np.max(np.abs(gradient), initial=0.0))
                < 1e-4 * max(1.0, abs(value))
                and definite
            )
            fit = _Fit(
                theta=np.asarray(full(res.x), dtype=float), value=value, ok=ok
            )
            if (
                best is None
                or (fit.ok and not best.ok)
                or (fit.ok == best.ok and fit.value < best.value)
            ):
                best = fit
            if ok:
                break
    if best is None:
        return _Fit(theta=start, value=float("nan"), ok=False)
    return best


def _covariance(f: Callable[[Any], Any], theta: Any, free: Any) -> np.ndarray:
    """The inverse of the observed information (the exact Hessian of
    ``f`` in the free parameters at ``theta``); ``nan`` where it is not
    positive definite."""
    from autograd import hessian

    idx = np.flatnonzero(free)
    theta = np.asarray(theta, dtype=float)

    def sub(v: Any) -> Any:
        import autograd.numpy as anp

        E = np.zeros((theta.size, idx.size))
        E[idx, np.arange(idx.size)] = 1.0
        return f(np.where(np.asarray(free), 0.0, theta) + anp.dot(E, v))

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = np.asarray(hessian(sub)(theta[idx]), dtype=float)
    try:
        if not np.all(np.isfinite(H)) or np.any(np.linalg.eigvalsh(H) <= 0):
            raise np.linalg.LinAlgError
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(H.shape, np.nan)


def _lr_test(
    f: Callable[[Any], Any],
    start: Any,
    free_null: Any,
    free_alt: Any,
    bounds: list,
    null: "_Fit | None" = None,
) -> dict:
    """The likelihood-ratio test of the model with the parameters
    ``free_null`` free (the extra ones held at ``start``, 0) against the
    one with ``free_alt`` free. Both are maximised here (the null only if
    its maximum ``null`` is not given, in the layout of ``f``'s
    parameters: several tests share it), the larger from the smaller's
    maximum, so the statistic is never negative."""
    df = int(np.sum(free_alt) - np.sum(free_null))
    if null is None:
        null = _maximise(f, start, free_null, bounds)
    if not null.ok:
        return {
            "statistic": np.nan,
            "df": df,
            "p": np.nan,
            "fit": None,
            "reason": "the fitted model's maximum could not be confirmed",
        }
    alt = _maximise(f, null.theta, free_alt, bounds)
    if not alt.ok:
        return {
            "statistic": np.nan,
            "df": df,
            "p": np.nan,
            "fit": None,
            "reason": "the larger model's fit did not reach a verified "
            "maximum",
        }
    stat = max(0.0, 2.0 * (null.value - alt.value))
    return {
        "statistic": stat,
        "df": df,
        "p": float(chi2.sf(stat, df)),
        "fit": alt,
        "null": null,
        "reason": "",
    }


def _in_layout(null: _Fit, extra: int, at: "int | None" = None) -> _Fit:
    """The maximum ``null`` with ``extra`` zeros inserted at position
    ``at`` of its parameters (at the end by default): the same maximum
    in the parameters of a larger model."""
    theta = np.asarray(null.theta, dtype=float)
    at = theta.size if at is None else at
    theta = np.concatenate([theta[:at], np.zeros(extra), theta[at:]])
    return _Fit(theta=theta, value=null.value, ok=null.ok)


def _fitted_null(ctx: Context) -> _Fit:
    """The fitted model's maximum, confirmed (found again from the fitted
    parameters with the exact derivatives), which the tests of a
    quadratic term and of a shape per level share."""

    def compute() -> _Fit:
        b = _base(ctx)
        f = _neg_ll(b.fitter, _view(b, b.Z))
        return _maximise(f, b.params, b.free, b.bounds)

    return ctx.cached("fitted_null", compute)


def _extend(b: SimpleNamespace, extra: int) -> tuple:
    """The fitted parameters with ``extra`` zeros after them, which of
    them are free under the null (the fitted ones) and the alternative
    (and the extra ones), and their bounds."""
    start = np.concatenate([b.params, np.zeros(extra)])
    null = np.concatenate([b.free, np.zeros(extra, dtype=bool)])
    alt = np.concatenate([b.free, np.ones(extra, dtype=bool)])
    bounds = b.bounds + [(None, None)] * extra
    return start, null, alt, bounds


# ---------------------------------------------------------------------------
# The larger models
# ---------------------------------------------------------------------------
class _SteppedPH(HazardIdentitiesMixin):
    """A proportional hazards model whose hazard steps at the times
    ``cuts``: in the window after the ``w``-th cut it is the fitted
    model's times ``exp(g0 + gamma' Z_J)``, a step ``g0`` in the baseline
    and steps ``gamma`` in the coefficients of the covariates ``J``. The
    extra parameters are, window by window, ``g0`` then ``gamma``. Built
    on the fitter's own ``Hf`` and ``hf``, so that with every step 0 it is
    the fitted model to the last bit.

    The baseline's step makes the test of the coefficients' steps
    independent of each covariate's origin: without it a step in the
    coefficient of ``Z_j`` is also a step of ``c gamma`` in the level of
    the hazard of a unit at ``Z_j = c``, which the fitted baseline then
    has to absorb (on R's veteran data, a step in the Karnofsky score's
    coefficient tested without it scores 0.17, with it 13)."""

    def __init__(self, fitter: Any, k: int, cuts: Any, J: Any) -> None:
        self.fitter = fitter
        self.k = k
        self.cuts = np.asarray(cuts, dtype=float)
        self.J = list(J)

    def _multipliers(self, Z: Any, params: tuple) -> list:
        """``exp(g0 + gamma' Z_J)`` of each window after the first."""
        import autograd.numpy as anp

        extra = params[self.k :]
        m = 1 + len(self.J)
        ZJ = Z[:, self.J]
        out = []
        for w in range(self.cuts.size):
            block = extra[w * m : (w + 1) * m]
            eta = block[0]
            if self.J:
                eta = eta + anp.dot(ZJ, anp.array(list(block[1:])))
            out.append(anp.exp(eta))
        return out

    def Hf(self, x: Any, Z: Any, *params: Any) -> Any:
        base = params[: self.k]
        multipliers = self._multipliers(Z, params)
        total: Any = 0.0
        previous: Any = 0.0
        edges = list(self.cuts) + [np.inf]
        for w, edge in enumerate(edges):
            upto = x if not np.isfinite(edge) else np.minimum(x, edge)
            H = self.fitter.Hf(upto, Z, *base)
            piece = H - previous
            if w:
                piece = multipliers[w - 1] * piece
            total = total + piece
            previous = H
        return total

    def hf(self, x: Any, Z: Any, *params: Any) -> Any:
        base = params[: self.k]
        multipliers = self._multipliers(Z, params)
        window = np.searchsorted(self.cuts, np.asarray(x), side="left")
        multiplier: Any = 1.0
        for w in range(1, self.cuts.size + 1):
            multiplier = multiplier + (window == w) * (
                multipliers[w - 1] - 1.0
            )
        return self.fitter.hf(x, Z, *base) * multiplier


class _GroupShapeAFT(HazardIdentitiesMixin):
    """An accelerated failure time model whose shape parameter (position
    ``s`` of the baseline's) is its fitted value times ``exp(G delta)``,
    ``G`` the level dummies, the columns of ``Z`` after the first ``p``."""

    def __init__(self, dist: Any, k_dist: int, p: int, s: int) -> None:
        self.dist = dist
        self.k_dist = k_dist
        self.p = p
        self.s = s

    def _rows(self, Z: Any, params: tuple) -> tuple:
        import autograd.numpy as anp

        k, p = self.k_dist, self.p
        beta = anp.array(list(params[k : k + p]))
        delta = anp.array(list(params[k + p :]))
        phi = anp.exp(anp.dot(Z[:, :p], beta))
        shape = params[self.s] * anp.exp(anp.dot(Z[:, p:], delta))
        dist_params = [shape if i == self.s else params[i] for i in range(k)]
        return phi, dist_params

    def Hf(self, x: Any, Z: Any, *params: Any) -> Any:
        phi, dist_params = self._rows(Z, params)
        return self.dist.Hf(phi * x, *dist_params)

    def hf(self, x: Any, Z: Any, *params: Any) -> Any:
        phi, dist_params = self._rows(Z, params)
        return phi * self.dist.hf(phi * x, *dist_params)


# ---------------------------------------------------------------------------
# What the checks share
# ---------------------------------------------------------------------------
def _levels(b: SimpleNamespace, j: int) -> int:
    return int(np.unique(b.raw[:, j]).size)


def _kind(b: SimpleNamespace, j: int) -> str:
    levels = _levels(b, j)
    if levels <= 2:
        return "binary"
    if levels <= CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _failed(b: SimpleNamespace) -> np.ndarray:
    """The rows that failed (exact, left or interval censored)."""
    return b.c != 1


def _failure_times(b: SimpleNamespace) -> np.ndarray:
    """Each failed row's time: an exact one's time, an interval-censored
    one's midpoint, a left-censored one's time."""
    mid = np.where(b.c == 2, 0.5 * (b.lo + b.hi), b.lo)
    return mid[_failed(b)]


def _weighted_quantile(t: np.ndarray, w: np.ndarray, q: float) -> float:
    """The smallest ``t`` at which the weighted fraction of ``t`` at or
    below it reaches ``q`` (the lower weighted quantile)."""
    order = np.argsort(t, kind="stable")
    t, w = t[order], w[order]
    share = np.cumsum(w) / w.sum()
    return float(t[np.searchsorted(share, q - 1e-12)])


def _cut(ctx: Context) -> float:
    """``tau``, the time of the step in the proportional-hazards test:
    the lower weighted median of the failure times."""

    def compute() -> float:
        b = _base(ctx)
        t = _failure_times(b)
        if t.size == 0:
            return float("nan")
        return _weighted_quantile(t, b.n[_failed(b)], 0.5)

    return ctx.cached("cut", compute)


def _fitter_name(b: SimpleNamespace, kind: str) -> str:
    """How the advice names the fitter of ``kind`` (``"PH"`` or ``"AFT"``)
    with the model's baseline: ``WeibullPH``, or ``PH(<name>)`` for a
    baseline that has no fitter of its own."""
    if b.name in BASELINES:
        return b.name + kind
    return "{}({})".format(kind, b.name)


def _interval_rows(b: SimpleNamespace) -> np.ndarray:
    """The rows with a martingale residual: exact or right censored, not
    right truncated."""
    return np.isin(b.c, (0, 1)) & ~np.isfinite(b.tr)


def _Hf(b: SimpleNamespace, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.asarray(b.fitter.Hf(x, b.Z, *b.params), dtype=float)


def _martingale(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, residuals)``: the martingale residual ``delta - H(x | Z)
    + H(tl | Z)`` of each row of :func:`_interval_rows`."""

    def compute() -> tuple[np.ndarray, np.ndarray]:
        b = _base(ctx)
        rows = _interval_rows(b)
        H = _Hf(b, b.lo)
        entered = np.isfinite(b.tl)
        H_tl = np.where(entered, _Hf(b, np.where(entered, b.tl, b.lo)), 0.0)
        m = (b.c == 0).astype(float) - (H - H_tl)
        return rows, m

    return ctx.cached("martingale", compute)


def _note(ctx: Context, text: str) -> None:
    """Add ``text`` to the report's notes, once."""
    if text not in ctx.notes:
        ctx.notes.append(text)


def _note_residual_rows(ctx: Context) -> None:
    b = _base(ctx)
    left_out = int((~_interval_rows(b)).sum())
    if left_out:
        _note(
            ctx,
            "The martingale and deviance residuals (functional_form plots, "
            "outliers) are of the exact and right-censored rows; {} "
            "left-censored, interval-censored or right-truncated row{} "
            "{} none and {} left out of them. The tests use every "
            "row.".format(
                left_out,
                "" if left_out == 1 else "s",
                "has" if left_out == 1 else "have",
                "is" if left_out == 1 else "are",
            ),
        )


# ---------------------------------------------------------------------------
# Proportional hazards (the PH family)
# ---------------------------------------------------------------------------
def _stepped(ctx: Context, J: list, cuts: Any) -> tuple:
    """The negative log-likelihood of :class:`_SteppedPH` with steps of
    the covariates ``J`` at ``cuts``, its start (the fitted parameters,
    every step 0), which parameters are free under the null (the fitted
    ones and the baseline's steps) and the alternative (and the
    coefficients' steps), and the bounds."""
    b = _base(ctx)
    windows = len(np.atleast_1d(cuts))
    model = _SteppedPH(b.fitter, b.params.size, cuts, J)
    f = _neg_ll(model, _view(b, b.Z))
    start, _, alt, bounds = _extend(b, windows * (1 + len(J)))
    null = alt.copy()
    for w in range(windows):
        first = b.params.size + w * (1 + len(J))
        null[first + 1 : first + 1 + len(J)] = False
    return f, (start, null, alt, bounds)


def _step_test(ctx: Context, J: list) -> dict:
    def compute() -> dict:
        b = _base(ctx)
        tau = _cut(ctx)
        failed = _failure_times(b)
        if not np.isfinite(tau) or not np.any(failed > tau):
            return {
                "statistic": np.nan,
                "df": len(J),
                "p": np.nan,
                "fit": None,
                "reason": "no failure after the median failure time",
            }
        f, (start, null, alt, bounds) = _stepped(ctx, J, [tau])
        shared = _step_null(ctx)
        return _lr_test(
            f, start, null, alt, bounds, null=_in_layout(shared, len(J))
        )

    return ctx.cached("step_{}".format(J), compute)


def _step_null(ctx: Context) -> _Fit:
    """The null of every step test: the fitted model with a step in the
    baseline at ``tau``, maximised once."""

    def compute() -> _Fit:
        f, (start, null, _, bounds) = _stepped(ctx, [], [_cut(ctx)])
        return _maximise(f, start, null, bounds)

    return ctx.cached("step_null", compute)


def _ph_rows(ctx: Context) -> list[dict]:
    b = _base(ctx)
    tau = _cut(ctx)
    rows = []
    for j, name in ctx.checked:
        res = _step_test(ctx, [j])
        p = res["p"]
        if not np.isfinite(p):
            detail = "the test could not be computed: " + res["reason"]
        elif p < 2 * ctx.alpha:
            gamma = res["fit"].theta[-1]
            beta = res["fit"].theta[b.k_dist + j]
            detail = "its coefficient {} after t = {} (from {} to {})".format(
                "rises" if gamma > 0 else "falls",
                _fmt(tau),
                _fmt(beta),
                _fmt(beta + gamma),
            )
        else:
            detail = (
                "no evidence that its coefficient changes at t = {}".format(
                    _fmt(tau)
                )
            )
        rows.append(
            {
                "test": "likelihood-ratio test of a step at the median",
                "covariates": (name,),
                "statistic": res["statistic"],
                "df": res["df"],
                "p": p,
                "detail": detail,
                "_kind": _kind(b, j),
                "_tau": tau,
            }
        )
    every = list(range(len(ctx.names)))
    res = _step_test(ctx, every)
    rows.append(
        {
            "test": "likelihood-ratio test of a step at the median (GLOBAL)",
            "covariates": tuple(ctx.names),
            "statistic": res["statistic"],
            "df": res["df"],
            "p": res["p"],
            "detail": (
                "the joint test of every covariate, at t = {}".format(
                    _fmt(tau)
                )
                if np.isfinite(res["p"])
                else "the test could not be computed: " + res["reason"]
            ),
            "_global": True,
            "_tau": tau,
        }
    )
    return rows


def _window_cuts(ctx: Context) -> np.ndarray:
    """The cuts of the plot's windows: the quartiles of the failure
    times, less any that would leave a window without a failure."""
    b = _base(ctx)
    t = _failure_times(b)
    w = b.n[_failed(b)]
    cuts: list[float] = []
    for q in np.arange(1, WINDOWS) / WINDOWS:
        cut = _weighted_quantile(t, w, q)
        lower = cuts[-1] if cuts else -np.inf
        if (
            cut > lower
            and np.any(t > cut)
            and np.any((t > lower) & (t <= cut))
        ):
            cuts.append(cut)
    return np.array(cuts)


def _window_estimates(ctx: Context, j: int) -> tuple:
    """Covariate ``j``'s coefficient in each window of
    :func:`_window_cuts` (the model with a step in it, and in the
    baseline, at each cut) and its standard error: ``nan`` where the
    model's fit is not a verified maximum."""

    def compute() -> tuple:
        b = _base(ctx)
        cuts = _window_cuts(ctx)
        values = np.full(cuts.size + 1, np.nan)
        ses = np.full(cuts.size + 1, np.nan)
        if not cuts.size:
            return values, ses
        f, (start, _, alt, bounds) = _stepped(ctx, [j], cuts)
        fit = _maximise(f, start, alt, bounds)
        if not fit.ok:
            return values, ses
        cov = _covariance(f, fit.theta, alt)
        idx = list(np.flatnonzero(alt))
        for w in range(cuts.size + 1):
            a = np.zeros(len(idx))
            a[idx.index(b.k_dist + j)] = 1.0
            if w:
                a[idx.index(b.params.size + 2 * (w - 1) + 1)] = 1.0
            values[w] = float(a @ fit.theta[idx])
            ses[w] = float(np.sqrt(a @ cov @ a))
        return values, ses

    return ctx.cached("windows_{}".format(j), compute)


def _ph_panels(ctx: Context) -> list[Panel]:
    b = _base(ctx)
    cuts = _window_cuts(ctx)
    t = _failure_times(b)
    p_values = {
        r["covariates"][0]: r["p"]
        for r in _ph_rows(ctx)
        if not r.get("_global")
    }
    panels = []
    for j, name in ctx.checked:

        def draw(ax: Any, options: Any, j: int = j) -> None:
            from surpyval.plotting import PALETTE

            values, ses = _window_estimates(ctx, j)
            z = norm.isf(options.alpha / 2)
            windows = np.arange(values.size)
            for w in windows:
                ax.plot(
                    [w, w + 1],
                    [values[w], values[w]],
                    color=PALETTE[0],
                    lw=2,
                    label="per window" if w == 0 else None,
                )
            shown = np.isfinite(ses)
            ax.errorbar(
                windows[shown] + 0.5,
                values[shown],
                yerr=z * ses[shown],
                color=PALETTE[0],
                capsize=3,
                lw=0.8,
                ls="none",
            )
            ax.axhline(
                b.params[b.k_dist + j],
                color=PALETTE[1],
                ls="--",
                label="fitted",
            )
            edges = np.concatenate([[t.min()], cuts, [t.max()]])
            ax.set_xticks(np.arange(edges.size))
            ax.set_xticklabels(["{:.3g}".format(v) for v in edges])
            ax.set_xlabel("time (windows of equal failures)")
            ax.set_ylabel("coefficient")

        panels.append(
            Panel("{}: p = {} (step)".format(name, _fmt(p_values[name])), draw)
        )
    return panels


# ---------------------------------------------------------------------------
# Functional form (both families)
# ---------------------------------------------------------------------------
def _form_columns(ctx: Context) -> list[tuple[int, str]]:
    b = _base(ctx)
    return [(j, n) for j, n in ctx.checked if _kind(b, j) != "binary"]


def _square_test(ctx: Context, j: int) -> dict:
    def compute() -> dict:
        b = _base(ctx)
        z = b.Z[:, j]
        scale = float(np.std(z)) or 1.0
        Z = np.column_stack([b.Z, (z / scale) ** 2])
        f = _neg_ll(b.fitter, _view(b, Z))
        start, null, alt, bounds = _extend(b, 1)
        shared = _in_layout(_fitted_null(ctx), 1)
        return _lr_test(f, start, null, alt, bounds, null=shared)

    return ctx.cached("square_{}".format(j), compute)


def _form_rows_of(scale: str) -> Callable[[Context], list[dict]]:
    def rows_of(ctx: Context) -> list[dict]:
        b = _base(ctx)
        skipped = [n for j, n in ctx.checked if _kind(b, j) == "binary"]
        if skipped:
            _note(
                ctx,
                "Functional form is tested for covariates with three or "
                "more distinct values; {} {} two, and the effect of a "
                "two-valued covariate is linear by construction.".format(
                    ", ".join(skipped), "has" if len(skipped) == 1 else "have"
                ),
            )
        if _form_columns(ctx):
            _note_residual_rows(ctx)
        rows = []
        for j, name in _form_columns(ctx):
            res = _square_test(ctx, j)
            p = res["p"]
            if not np.isfinite(p):
                detail = "the test could not be computed: " + res["reason"]
            elif p < 2 * ctx.alpha:
                detail = "its effect on {} is not linear".format(scale)
            else:
                detail = "no evidence against a linear effect"
            rows.append(
                {
                    "test": "likelihood-ratio test of a quadratic term",
                    "covariates": (name,),
                    "statistic": res["statistic"],
                    "df": res["df"],
                    "p": p,
                    "detail": detail,
                }
            )
        return rows

    return rows_of


def _form_panels(ctx: Context) -> list[Panel]:
    from .smooth import lowess

    b = _base(ctx)
    rows, m = _martingale(ctx)
    panels = []
    for j, name in _form_columns(ctx):

        def draw(ax: Any, options: Any, j: int = j, name: str = name) -> None:
            from surpyval.plotting import INK, PALETTE

            z = b.raw[rows, j]
            grid = np.linspace(z.min(), z.max(), 60)
            ax.scatter(z, m[rows], s=10, color=INK["muted"], alpha=0.6)
            ax.plot(grid, lowess(z, m[rows], grid), color=PALETTE[0])
            ax.axhline(0.0, color=INK["axis"], lw=0.8)
            ax.set_xlabel(name)
            ax.set_ylabel("martingale residual")

        if rows.any():
            panels.append(Panel("{}: functional form".format(name), draw))
    return panels


# ---------------------------------------------------------------------------
# Influence and outliers (the PH family)
# ---------------------------------------------------------------------------
def _dfbeta(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """``(dfbeta, se)``: each unit's one-step change ``I^-1 U_i`` of each
    coefficient (one row per data row; a row with count ``n`` is one of
    ``n`` identical units), and the coefficients' standard errors, both
    ``nan`` where the information is not positive definite."""

    def compute() -> tuple[np.ndarray, np.ndarray]:
        from autograd import grad, jacobian

        b = _base(ctx)
        idx = np.flatnonzero(b.free)
        theta = b.params

        def gradient(w: Any) -> Any:
            f = _neg_ll(b.fitter, _view(b, b.Z, w))
            return grad(f)(theta)[idx]

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # d(grad of -loglik)/d(weight of row i) is -n_i u_i, u_i the
            # score of one unit of row i: the likelihood is linear in the
            # weights, so this is exact.
            J = np.asarray(jacobian(gradient)(np.ones(b.n.size)), dtype=float)
        scores = -J.T / b.n[:, None]
        f = _neg_ll(b.fitter, _view(b, b.Z))
        cov = _covariance(f, theta, b.free)
        coef = [list(idx).index(b.k_dist + j) for j in range(len(ctx.names))]
        dfbeta = (scores @ cov)[:, coef]
        se = np.sqrt(np.diag(cov))[coef]
        return dfbeta, se

    return ctx.cached("dfbeta", compute)


def _influence_rows(ctx: Context) -> list[dict]:
    dfbeta, se = _dfbeta(ctx)
    scaled = np.abs(dfbeta / se[None, :])
    rows: list[dict] = []
    for j, name in ctx.checked:
        col = scaled[:, j]
        if not np.all(np.isfinite(col)):
            rows.append(
                {
                    "test": "largest |dfbeta| / se",
                    "covariates": (name,),
                    "verdict": "warn",
                    "detail": "the screen could not be computed: the "
                    "observed information is not positive definite",
                }
            )
            continue
        worst = int(np.argmax(col))
        above = np.flatnonzero(col > INFLUENCE_LIMIT)
        if above.size:
            detail = (
                "row {} of model.data moves the coefficient by {} standard "
                "errors; {} row{} above {:g}".format(
                    worst,
                    _fmt(col[worst]),
                    above.size,
                    "" if above.size == 1 else "s",
                    INFLUENCE_LIMIT,
                )
            )
        else:
            detail = (
                "no row moves the coefficient by more than {:g} standard "
                "error (largest {}, row {} of model.data)".format(
                    INFLUENCE_LIMIT, _fmt(col[worst]), worst
                )
            )
        rows.append(
            {
                "test": "largest |dfbeta| / se",
                "covariates": (name,),
                "statistic": float(col[worst]),
                "verdict": "warn" if above.size else "ok",
                "detail": detail,
                "_row": worst,
                "_size": float(col[worst]),
            }
        )
    return rows


def _influence_panels(ctx: Context) -> list[Panel]:
    dfbeta, se = _dfbeta(ctx)
    scaled = dfbeta / se[None, :]
    panels = []
    for j, name in ctx.checked:

        def draw(ax: Any, options: Any, j: int = j) -> None:
            from surpyval.plotting import INK, PALETTE

            rows = np.arange(scaled.shape[0])
            ax.vlines(rows, 0, scaled[:, j], color=PALETTE[0], lw=0.8)
            for level in (-INFLUENCE_LIMIT, INFLUENCE_LIMIT):
                ax.axhline(level, color=INK["secondary"], ls="--", lw=0.8)
            ax.set_xlabel("row")
            ax.set_ylabel("dfbeta / se")

        panels.append(Panel("{}: influence".format(name), draw))
    return panels


def _deviance(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    rows, m = _martingale(ctx)
    b = _base(ctx)
    delta = (b.c == 0).astype(float)
    with np.errstate(all="ignore"):
        inner = m + np.where(delta > 0, delta * np.log(delta - m), 0.0)
        dev = np.sign(m) * np.sqrt(np.maximum(-2.0 * inner, 0.0))
    return rows, dev


def _outlier_cutoff(ctx: Context) -> float:
    b = _base(ctx)
    rows, _ = _martingale(ctx)
    n = float(b.n[rows].sum())
    return float(norm.isf(ctx.alpha / (2.0 * n)))


def _outlier_rows(ctx: Context) -> list[dict]:
    rows, dev = _deviance(ctx)
    _note_residual_rows(ctx)
    if not rows.any():
        _note(
            ctx,
            "Outliers are not screened: no row is exact or right censored "
            "without right truncation, the rows with a deviance residual.",
        )
        return []
    cutoff = _outlier_cutoff(ctx)
    absolute = np.where(rows, np.abs(dev), 0.0)
    flagged = np.flatnonzero(rows & (absolute > cutoff))
    if flagged.size:
        shown = ", ".join(str(int(i)) for i in flagged[:10])
        more = (
            ""
            if flagged.size <= 10
            else " and {} more".format(flagged.size - 10)
        )
        detail = "row{} {}{} of model.data above {}".format(
            "" if flagged.size == 1 else "s", shown, more, _fmt(cutoff)
        )
    else:
        detail = "no |deviance residual| above {} (largest {})".format(
            _fmt(cutoff), _fmt(absolute.max())
        )
    return [
        {
            "test": "deviance residuals, Bonferroni cutoff",
            "covariates": (),
            "statistic": float(absolute.max()),
            "verdict": "warn" if flagged.size else "ok",
            "detail": detail,
            "_rows": flagged.tolist(),
            "_cutoff": cutoff,
        }
    ]


def _outlier_panels(ctx: Context) -> list[Panel]:
    rows, dev = _deviance(ctx)
    if not rows.any():
        return []
    b = _base(ctx)
    eta = b.Z @ b.params[b.k_dist :]
    cutoff = _outlier_cutoff(ctx)

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.scatter(eta[rows], dev[rows], s=10, color=PALETTE[0], alpha=0.6)
        for level in (-cutoff, cutoff):
            ax.axhline(level, color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("linear predictor")
        ax.set_ylabel("deviance residual")

    return [Panel("outliers", draw)]


# ---------------------------------------------------------------------------
# The AIC comparisons (both families)
# ---------------------------------------------------------------------------
def _fitter(name: str, kind: str, dist: Any) -> Any:
    """The fitter of ``kind`` (``"PH"`` or ``"AFT"``) with the baseline
    named ``name`` (``dist`` itself where it is not built in)."""
    import surpyval

    if name in BASELINES:
        return getattr(surpyval, name + kind)
    return getattr(surpyval, kind)(dist)


def _refit(ctx: Context, fitter: Any) -> tuple[float, str, str]:
    """``(aic, status, reason)`` of ``fitter`` refitted to the model's
    data: status ``"ranked"`` (a verified maximum), ``"set aside"`` (not
    one) or ``"failed"``."""
    b = _base(ctx)
    data = b.data
    kwargs: dict = {"c": data.c, "n": data.n, "t": data.t}
    if b.model._has_center():
        kwargs["center"] = True
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = fitter.fit(data.x, b.raw, **kwargs)
            aic = float(fitted.aic())
    except Exception as error:  # noqa: BLE001 -- a candidate may fail
        text = str(error).strip().splitlines()
        return (
            float("nan"),
            "failed",
            text[0][:100] if text else type(error).__name__,
        )
    if fitted.maximum != "verified" or not np.isfinite(aic):
        return (
            aic,
            "set aside",
            "not a verified maximum ({})".format(fitted.maximum),
        )
    return aic, "ranked", ""


def _aic_candidates(ctx: Context, kind: str) -> list[dict]:
    """The model and the candidates of its own kind (``"PH"`` or
    ``"AFT"``), and the other kind's model with its baseline (except
    where they are the same), each ``{"label", "baseline", "kind", "aic",
    "status", "reason"}``."""

    def compute() -> list[dict]:
        b = _base(ctx)
        other = "AFT" if kind == "PH" else "PH"
        out = [
            {
                "label": _fitter_name(b, kind),
                "baseline": b.name,
                "kind": kind,
                "aic": float(b.model.aic()),
                "status": "fitted",
                "reason": "",
            }
        ]
        for name in BASELINES:
            if name == b.name:
                continue
            aic, status, reason = _refit(ctx, _fitter(name, kind, None))
            out.append(
                {
                    "label": name + kind,
                    "baseline": name,
                    "kind": kind,
                    "aic": aic,
                    "status": status,
                    "reason": reason,
                }
            )
        if b.name not in SAME_PH_AFT:
            aic, status, reason = _refit(ctx, _fitter(b.name, other, b.dist))
            out.append(
                {
                    "label": _fitter_name(b, other),
                    "baseline": b.name,
                    "kind": other,
                    "aic": aic,
                    "status": status,
                    "reason": reason,
                }
            )
        return out

    return ctx.cached("aic_" + kind, compute)


def _gap(delta: float) -> str:
    """Another model's AIC against the fitted model's, ``delta`` the
    fitted model's less the other's, in words."""
    if delta == 0:
        return "level with it"
    return "{} {}".format(_fmt(abs(delta)), "lower" if delta > 0 else "higher")


def _aic_rows_of(kind: str) -> Callable[[Context], list[dict]]:
    other = "AFT" if kind == "PH" else "PH"
    word = "baselines" if kind == "PH" else "error distributions"

    def rows_of(ctx: Context) -> list[dict]:
        b = _base(ctx)
        fixed = [name for name in b.model.fixed]
        if fixed:
            _note(
                ctx,
                "The AIC comparisons are left out: the fit holds {} fixed, "
                "which the other models cannot hold.".format(", ".join(fixed)),
            )
            return []
        candidates = _aic_candidates(ctx, kind)
        own = candidates[0]
        others = [
            c
            for c in candidates[1:]
            if c["kind"] == kind and c["status"] == "ranked"
        ]
        skipped = [c for c in candidates[1:] if c["status"] != "ranked"]
        if skipped:
            _note(
                ctx,
                "AIC comparison: {} left out ({}).".format(
                    ", ".join(c["label"] for c in skipped),
                    "; ".join(
                        "{}: {}".format(c["label"], c["reason"])
                        for c in skipped
                    ),
                ),
            )
        rows = []
        if others:
            best = min(others, key=lambda c: c["aic"])
            delta = own["aic"] - best["aic"]
            if delta > AIC_MARGIN:
                detail = "{} fits better: its AIC is {} lower".format(
                    best["label"], _fmt(delta)
                )
            else:
                detail = (
                    "no other {} has an AIC more than {:g} lower (the best "
                    "of them, {}, is {})".format(
                        word[:-1], AIC_MARGIN, best["label"], _gap(delta)
                    )
                )
            rows.append(
                {
                    "test": "AIC against the other {}".format(word),
                    "covariates": (),
                    "statistic": delta,
                    "verdict": "warn" if delta > AIC_MARGIN else "ok",
                    "detail": detail,
                    "_aic": "baselines",
                    "_best": best["baseline"],
                    "_label": best["label"],
                    "_delta": delta,
                }
            )
        twin = [
            c
            for c in candidates[1:]
            if c["kind"] == other and c["status"] == "ranked"
        ]
        if b.name in SAME_PH_AFT:
            if kind == "PH":  # the AFT family's notes say so themselves
                _note(
                    ctx,
                    "A {name} proportional hazards model is also an "
                    "accelerated failure time model ({ph} and {aft} fit the "
                    "same model), so the two are not compared.".format(
                        name=b.name,
                        ph=_fitter_name(b, "PH"),
                        aft=_fitter_name(b, "AFT"),
                    ),
                )
        elif twin:
            delta = own["aic"] - twin[0]["aic"]
            if delta > AIC_MARGIN:
                detail = "{} fits better: its AIC is {} lower".format(
                    twin[0]["label"], _fmt(delta)
                )
            else:
                detail = (
                    "{}'s AIC is not more than {:g} lower (it is {})".format(
                        twin[0]["label"], AIC_MARGIN, _gap(delta)
                    )
                )
            rows.append(
                {
                    "test": "AIC against the {} model, same baseline".format(
                        other
                    ),
                    "covariates": (),
                    "statistic": delta,
                    "verdict": "warn" if delta > AIC_MARGIN else "ok",
                    "detail": detail,
                    "_aic": "kind",
                    "_label": twin[0]["label"],
                    "_delta": delta,
                }
            )
        return rows

    return rows_of


def _aic_panel(ctx: Context, kind: str) -> Panel:
    candidates = [
        c
        for c in _aic_candidates(ctx, kind)
        if c["status"] in ("fitted", "ranked")
    ]

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        best = min(c["aic"] for c in candidates)
        order = sorted(candidates, key=lambda c: c["aic"])
        y = np.arange(len(order))
        colours = [
            PALETTE[1] if c["status"] == "fitted" else PALETTE[0]
            for c in order
        ]
        ax.barh(y, [c["aic"] - best for c in order], color=colours)
        ax.set_yticks(y)
        ax.set_yticklabels([c["label"] for c in order])
        ax.invert_yaxis()
        ax.axvline(AIC_MARGIN, color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("AIC - lowest AIC")

    return Panel("AIC by model", draw)


# ---------------------------------------------------------------------------
# Probability plots (both families)
# ---------------------------------------------------------------------------
def _residual_data(ctx: Context, kind: str) -> dict:
    """The residuals of each row with their censoring: for a PH model the
    Cox-Snell residuals ``H(x | Z)`` (a unit exponential sample, truncated
    at ``H(tl | Z)`` and ``H(tr | Z)``), for an AFT model the standardised
    residuals ``e^{beta'Z} x`` (a sample of the baseline, truncated at
    ``e^{beta'Z} tl`` and ``e^{beta'Z} tr``)."""

    def compute() -> dict:
        b = _base(ctx)
        if kind == "PH":

            def g(v: np.ndarray) -> np.ndarray:
                finite = np.isfinite(v)
                safe = np.where(finite, v, b.lo)
                out = _Hf(b, safe)
                return np.where(finite, out, np.where(v > 0, np.inf, -np.inf))

        else:
            phi = np.exp(b.Z @ b.params[b.k_dist :])

            def g(v: np.ndarray) -> np.ndarray:
                with np.errstate(invalid="ignore"):
                    return np.where(np.isfinite(v), phi * v, v)

        return {
            "lo": g(b.lo),
            "hi": g(b.hi),
            "c": b.c,
            "n": b.n,
            "tl": g(b.tl),
            "tr": g(b.tr),
        }

    return ctx.cached("residuals_" + kind, compute)


def _plotting_points(res: dict, rows: Any = None) -> tuple:
    """The nonparametric estimate of the residuals' distribution (the
    Kaplan-Meier estimate, or Turnbull's for left- or interval-censored or
    right-truncated residuals) at its jumps: ``(x, F)``, ``F`` the middle
    of each jump."""
    import surpyval

    keep = np.ones(res["c"].size, dtype=bool) if rows is None else rows
    lo, hi, c, n = (
        res["lo"][keep],
        res["hi"][keep],
        res["c"][keep],
        res["n"][keep],
    )
    tl, tr = res["tl"][keep], res["tr"][keep]
    if not np.any(c != 1):
        return np.array([]), np.array([])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entry = tl if np.any(np.isfinite(tl)) else None
        if np.all(np.isin(c, (0, 1))) and not np.any(np.isfinite(tr)):
            fit = surpyval.KaplanMeier.fit(lo, c=c, n=n, tl=entry)
        else:
            fit = surpyval.Turnbull.fit(
                x=np.column_stack([lo, hi]),
                c=c,
                n=n,
                tl=tl,
                tr=tr,
                turnbull_estimator="Kaplan-Meier",
            )
    x, F = np.asarray(fit.x, dtype=float), np.asarray(fit.F, dtype=float)
    previous = np.concatenate([[0.0], F[:-1]])
    jumps = F > previous + 1e-12
    return x[jumps], 0.5 * (previous + F)[jumps]


def _paper(kind: str, b: SimpleNamespace) -> tuple:
    """``(dist, params, x label)`` of the probability paper: the unit
    exponential for a PH model's Cox-Snell residuals, the fitted baseline
    for an AFT model's standardised ones."""
    import surpyval

    if kind == "PH":
        return surpyval.Exponential, (1.0,), "Cox-Snell residual"
    return (
        b.dist,
        tuple(b.params[: b.k_dist]),
        "standardised residual e^(b'Z) x",
    )


def _draw_probability(
    ax: Any, ctx: Context, kind: str, groups: "list | None" = None
) -> None:
    from surpyval.plotting import INK, PALETTE

    b = _base(ctx)
    dist, params, label = _paper(kind, b)
    res = _residual_data(ctx, kind)
    groups = groups or [("", None)]
    xs = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, (name, rows) in enumerate(groups):
            x, F = _plotting_points(res, rows)
            if not x.size:
                continue
            xs.append(x)
            ax.scatter(
                dist.mpp_x_transform(x),
                dist.mpp_y_transform(F, *params),
                s=10,
                color=PALETTE[k % len(PALETTE)],
                alpha=0.7,
                label=name or None,
            )
        if xs:
            every = np.concatenate(xs)
            grid = np.linspace(every.min(), every.max(), 100)
            if dist.support[0] >= 0:
                grid = grid[grid > 0]
            F0 = np.asarray(dist.ff(grid, *params), dtype=float)
            inside = (F0 > 0) & (F0 < 1)
            ax.plot(
                dist.mpp_x_transform(grid[inside]),
                dist.mpp_y_transform(F0[inside], *params),
                color=INK["secondary"],
                ls="--",
                lw=0.8,
                label="fitted" if groups[0][0] else None,
            )
    transformed = np.asarray(dist.mpp_x_transform(np.array([2.0])))
    log_x = not np.allclose(transformed, 2.0)
    ax.set_xlabel(("log " if log_x else "") + label)
    ax.set_ylabel("{} probability scale".format(dist.name))
    if groups[0][0]:
        ax.legend(fontsize="small")


def _distribution_panels_of(kind: str) -> Callable[[Context], list[Panel]]:
    def panels_of(ctx: Context) -> list[Panel]:
        b = _base(ctx)
        title = (
            "Cox-Snell residuals (exponential paper)"
            if kind == "PH"
            else "standardised residuals ({} paper)".format(b.name)
        )
        panels = [
            Panel(title, lambda ax, options: _draw_probability(ax, ctx, kind))
        ]
        if not b.model.fixed:
            panels.append(_aic_panel(ctx, kind))
        return panels

    return panels_of


# ---------------------------------------------------------------------------
# Common shape (the AFT family)
# ---------------------------------------------------------------------------
_DUMMY = re.compile(r"\[T\.(.*)\]$")


def _level_name(name: str) -> str:
    """The level a formula's dummy column codes: ``b`` for
    ``C(arm)[T.b]``."""
    match = _DUMMY.search(name)
    return name if match is None else match.group(1)


def _factors(ctx: Context) -> list[dict]:
    """The categorical covariates to test, each ``{"name", "columns",
    "labels", "level"}``: ``level`` the level of each row (0 the
    reference), ``labels`` the levels' names. A formula's dummy columns of
    one variable are one factor; any other column with two to
    :data:`~.cox.CATEGORICAL_LEVELS` distinct values is its own."""

    def compute() -> list[dict]:
        b = _base(ctx)
        out = []
        dummies: dict[str, list[int]] = {}
        single = []
        for j, name in enumerate(ctx.names):
            column = b.raw[:, j]
            if _DUMMY.search(name) and set(np.unique(column)) <= {0.0, 1.0}:
                dummies.setdefault(_source(name), []).append(j)
            else:
                single.append(j)
        for var, cols in dummies.items():
            block = b.raw[:, cols]
            if np.any(block.sum(axis=1) > 1):
                # Not the treatment coding of one factor: each column is
                # its own covariate.
                single.extend(cols)
                continue
            if not any(j in ctx.columns for j in cols):
                continue
            code = np.where(
                block.sum(axis=1) == 0, 0, 1 + np.argmax(block, axis=1)
            )
            present = np.unique(code)
            labels = [
                (
                    "{} reference".format(var)
                    if k == 0
                    else "{}={}".format(
                        var, _level_name(ctx.names[cols[k - 1]])
                    )
                )
                for k in present
            ]
            out.append(
                {
                    "name": var,
                    "columns": tuple(ctx.names[j] for j in cols),
                    "labels": labels,
                    "level": np.searchsorted(present, code),
                }
            )
        for j in single:
            if j not in ctx.columns or _kind(b, j) == "continuous":
                continue
            values, level = np.unique(b.raw[:, j], return_inverse=True)
            out.append(
                {
                    "name": ctx.names[j],
                    "columns": (ctx.names[j],),
                    "labels": [
                        "{}={:g}".format(ctx.names[j], v) for v in values
                    ],
                    "level": np.asarray(level).reshape(-1),
                }
            )
        return out

    return ctx.cached("factors", compute)


def _shape_test(ctx: Context, factor: dict) -> dict:
    def compute() -> dict:
        b = _base(ctx)
        s = b.model.distribution_param_map[SHAPE[b.name]]
        level = factor["level"]
        L = len(factor["labels"])
        failed = _failed(b)
        for k in range(L):
            times = np.unique(_failure_times(b)[level[failed] == k])
            if times.size < 2:
                return {
                    "statistic": np.nan,
                    "df": L - 1,
                    "p": np.nan,
                    "fit": None,
                    "reason": "level {} has fewer than two distinct "
                    "failure times".format(factor["labels"][k]),
                }
        G = (level[:, None] == np.arange(1, L)[None, :]).astype(float)
        model = _GroupShapeAFT(b.dist, b.k_dist, b.Z.shape[1], s)
        f = _neg_ll(model, _view(b, np.column_stack([b.Z, G])))
        start, null, alt, bounds = _extend(b, L - 1)
        shared = _in_layout(_fitted_null(ctx), L - 1)
        return _lr_test(f, start, null, alt, bounds, null=shared)

    return ctx.cached("shape_{}".format(factor["name"]), compute)


def _shape_applies(ctx: Context) -> bool:
    b = _base(ctx)
    if b.name not in SHAPE:
        _note(
            ctx,
            "Common shape is not tested: the {} baseline has no shape "
            "parameter{}.".format(
                b.name,
                "" if b.name == "Exponential" else " the check knows of",
            ),
        )
        return False
    if SHAPE[b.name] in b.model.fixed:
        _note(
            ctx,
            "Common shape is not tested: the fit holds the shape ({}) "
            "fixed.".format(SHAPE[b.name]),
        )
        return False
    return True


def _shape_rows(ctx: Context) -> list[dict]:
    b = _base(ctx)
    if not _shape_applies(ctx):
        return []
    factors = _factors(ctx)
    tested = {c for f in factors for c in f["columns"]}
    skipped = [n for _, n in ctx.checked if n not in tested]
    if skipped:
        _note(
            ctx,
            "Common shape is tested for categorical covariates (two to {} "
            "distinct values, or a formula's factor); {} {} more.".format(
                _NUMBERS.get(CATEGORICAL_LEVELS, CATEGORICAL_LEVELS),
                ", ".join(skipped),
                "has" if len(skipped) == 1 else "have",
            ),
        )
    param = SHAPE[b.name]
    s = b.model.distribution_param_map[param]
    rows = []
    for factor in factors:
        res = _shape_test(ctx, factor)
        p = res["p"]
        if not np.isfinite(p):
            detail = "the test could not be computed: " + res["reason"]
        else:
            theta = res["fit"].theta
            delta = np.concatenate([[0.0], theta[b.params.size :]])
            shapes = theta[s] * np.exp(delta)
            by_level = ", ".join(
                "{} ({})".format(_fmt(v), label)
                for v, label in zip(shapes, factor["labels"])
            )
            if p < 2 * ctx.alpha:
                detail = "the shape ({}) differs by level: {}".format(
                    param, by_level
                )
            else:
                detail = (
                    "no evidence that the shape ({}) differs by level: "
                    "{}".format(param, by_level)
                )
        rows.append(
            {
                "test": "likelihood-ratio test of a shape per level",
                "covariates": factor["columns"],
                "statistic": res["statistic"],
                "df": res["df"],
                "p": p,
                "detail": detail,
                "_factor": factor["name"],
            }
        )
    return rows


def _shape_panels(ctx: Context) -> list[Panel]:
    b = _base(ctx)
    if b.name not in SHAPE or SHAPE[b.name] in b.model.fixed:
        return []
    panels = []
    for factor in _factors(ctx):
        groups = [
            (label, factor["level"] == k)
            for k, label in enumerate(factor["labels"])
        ]

        def draw(ax: Any, options: Any, groups: list = groups) -> None:
            _draw_probability(ax, ctx, "AFT", groups)

        panels.append(
            Panel("{}: residuals by level".format(factor["name"]), draw)
        )
    return panels


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    if getattr(model, "is_tvc", False):
        raise NotImplementedError(
            "diagnose() is not yet available for a parametric regression "
            "fitted to time-varying covariates (fit_tvc, fit_tvc_timeline): "
            "its checks need one row per unit; tracked in #869. Diagnose "
            "the model fitted with fit (or fit_from_df) to each unit's "
            "covariates at the start."
        )
    if (
        getattr(model, "_restored", False)
        or getattr(model, "data", None) is None
    ):
        raise ValueError(
            "diagnose() needs the data the model was fitted to, which a "
            "model restored with from_dict / from_json does not carry. Call "
            "it on the fitted model, or refit."
        )
    link = getattr(getattr(model, "reg_model", None), "name", None)
    if link not in (LogLinearPhi.NAME_E, LogLinearPhi.NAME_EXP):
        raise NotImplementedError(
            "diagnose() is not yet available for a parametric regression "
            "with the covariate link {!r}: its checks are for the "
            "exp(beta'Z) link of WeibullPH, PH(dist), WeibullAFT and "
            "AFT(dist); tracked in #869.".format(link)
        )
    if model.maximum == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one has "
            "none (model.maximum is 'no finite maximum', as the fit warned: "
            "its likelihood keeps rising as a parameter runs off, most "
            "often because a covariate, or a combination of them, "
            "separates the failures from the survivors). Drop or merge the "
            "separating covariate, or choose a baseline that the data "
            "determine, and refit."
        )
    if model.maximum != "verified":
        raise ValueError(
            "diagnose() needs a fit at a verified maximum of its likelihood "
            "(its tests compare it with larger models), and this one's "
            "model.maximum is {!r}, as the fit warned. Refit from another "
            "start (init=), or with center=True for covariates far from "
            "0.".format(model.maximum)
        )
    aliased = model.aliased
    if np.size(aliased):
        raise ValueError(
            "diagnose() is not available for a model with aliased "
            "coefficients (column(s) {} of Z, whose coefficients are nan, "
            "#476). Refit without those columns.".format(
                np.asarray(aliased).tolist()
            )
        )


def _ph_notes(ctx: Context) -> list[str]:
    b = _base(ctx)
    return [
        "Proportional hazards: the likelihood-ratio test of a step in each "
        "coefficient at t = {} (the median failure time), and in every "
        "coefficient at once (GLOBAL); d.plot('proportional_hazards') "
        "draws each coefficient in each quarter of the failure "
        "times.".format(_fmt(_cut(ctx))),
        "Baseline distribution: the AIC against each other built-in "
        "baseline ({}PH) refitted to the same data warns when one is more "
        "than {:g} lower; the Cox-Snell residuals are drawn on exponential "
        "paper (d.plot('baseline_distribution')) and, as for a Cox model, "
        "have no useful test of their own.".format(
            "PH, ".join(n for n in BASELINES if n != b.name), AIC_MARGIN
        ),
        "Influence is the one-step change I^-1 U_i of each coefficient "
        "when a unit is dropped, not a refit without it.",
    ]


def _aft_notes(ctx: Context) -> list[str]:
    b = _base(ctx)
    notes = [
        "Error distribution: the standardised residuals e^(beta'Z) x, the "
        "time on the baseline's scale, are drawn on {name} paper "
        "(d.plot('error_distribution')); the AIC against each other "
        "built-in error distribution refitted to the same data warns when "
        "one is more than {margin:g} lower.".format(
            name=b.name, margin=AIC_MARGIN
        ),
    ]
    if b.name in SHAPE:
        notes.append(
            "Common shape: the likelihood-ratio test of one {} for every "
            "level of a categorical covariate against one per "
            "level.".format(SHAPE[b.name])
        )
    if b.name in SAME_PH_AFT:
        notes.append(
            "A {name} accelerated failure time model is also a proportional "
            "hazards model: {ph}.fit on the same data fits the same model, "
            "and its diagnose() tests proportional hazards.".format(
                name=b.name, ph=_fitter_name(b, "PH")
            )
        )
    return notes


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _dist_is(
    name: "str | None", same: "bool | None" = None
) -> Callable[[Context], bool]:
    """Whether the model's baseline is the built-in ``name`` (``None``:
    one that is not built in, and, if ``same`` is given, one whose PH and
    AFT models are (True) or are not (False) the same model)."""

    def test(ctx: Context) -> bool:
        baseline = ctx.model.distribution.name
        if name is None:
            return baseline not in BASELINES and (
                same is None or (baseline in SAME_PH_AFT) == same
            )
        return baseline == name

    return test


def _aft_instead(name: "str | None") -> str:
    """The AFT model the PH advice names: the same baseline's, except for
    the Weibull and the exponential, whose AFT model is their PH model."""
    if name in SAME_PH_AFT:
        return "LogNormalAFT"
    if name is None:
        return "AFT"
    return name + "AFT"


def _advised(ctx: Context) -> dict:
    """How the advice names the model's own fitter (``own``, e.g.
    ``WeibullPH``, or ``PH(Rayleigh)`` for a baseline with no fitter of
    its own) and the AFT model it suggests (``aft``)."""
    b = _base(ctx)
    aft = "LogNormalAFT" if b.name in SAME_PH_AFT else _fitter_name(b, "AFT")
    return {"own": _fitter_name(b, "PH"), "aft": aft}


def _ph_rules(name: "str | None", same: bool = False) -> list[AdviceRule]:
    """The PH family's advice for a model with the baseline ``name``
    (``None``: one not built in, whose PH and AFT models are the same
    model if ``same``)."""
    on_dist = _dist_is(name, same if name is None else None)
    own = "PH" if name is None else name + "PH"
    tvc = ("PH",) if name is None else (own + ".fit_tvc",)
    if name is None:
        aft_feature = ("LogNormalAFT.fit",) if same else ("AFT",)
    else:
        aft_feature = (_aft_instead(name) + ".fit",)
    fit = ("PH",) if name is None else (own + ".fit",)
    from_df = ("PH",) if name is None else (own + ".fit_from_df(formula=)",)

    def stepping(row: dict, ctx: Context, stratify: bool) -> str:
        cov = row["covariates"][0]
        text = (
            "The effect of {cov} changes with time (it fails the test of a "
            "step in its coefficient at t = {tau}). Give it a time-varying "
            "effect: split each unit's follow-up at t = {tau} and fit "
            "{own}.fit_tvc on the pieces, with one column per period holding "
            "{cov} in that period and 0 elsewhere, so that each period has "
            "its own coefficient. Or fit an accelerated failure time model, "
            "whose hazard ratios change with time, such as {aft}.fit, and "
            "compare.".format(cov=cov, tau=_fmt(row["_tau"]), **_advised(ctx))
        )
        if stratify:
            var = _source(cov)
            text += (
                " Or, as {var} has few levels, fit a Cox model "
                "stratified on it (CoxPH.fit(..., strata=<the {var} "
                "column>), leaving {var} out of Z), which gives each level "
                "its own baseline hazard.".format(var=var)
            )
        return text

    def failed_alone(row: dict, ctx: Context) -> bool:
        if not row.get("_global"):
            return False
        return not any(
            _step_test(ctx, [j])["p"] < ctx.alpha for j, _ in ctx.checked
        )

    return [
        AdviceRule(
            "proportional_hazards",
            lambda row, ctx: stepping(row, ctx, True),
            tvc + aft_feature + ("CoxPH.fit(strata=)",),
            when=lambda row, ctx: on_dist(ctx)
            and row.get("_kind") in ("binary", "categorical"),
        ),
        AdviceRule(
            "proportional_hazards",
            lambda row, ctx: stepping(row, ctx, False),
            tvc + aft_feature,
            when=lambda row, ctx: on_dist(ctx)
            and row.get("_kind") == "continuous",
        ),
        AdviceRule(
            "proportional_hazards",
            lambda row, ctx: (
                "The joint (GLOBAL) test of a step in every coefficient at "
                "t = {tau} fails though no single covariate's does. The "
                "covariates whose coefficients move most across the windows "
                "of d.plot('proportional_hazards') are the likeliest cause: "
                "give them a time-varying effect ({own}.fit_tvc), or fit an "
                "accelerated failure time model ({aft}.fit).".format(
                    tau=_fmt(row["_tau"]), **_advised(ctx)
                )
            ),
            tvc + aft_feature,
            when=lambda row, ctx: on_dist(ctx) and failed_alone(row, ctx),
        ),
        AdviceRule(
            "functional_form",
            lambda row, ctx: _form_message(
                row, _advised(ctx)["own"], "the log hazard"
            ),
            from_df,
            when=lambda row, ctx: on_dist(ctx),
        ),
        AdviceRule(
            "influence",
            lambda row, ctx: (
                "Row {row} of model.data (the fitted rows, sorted) has a "
                "large influence on the coefficient of {cov}: dropping it "
                "would move the coefficient by about {size} standard "
                "errors. Check that row, and refit ({own}.fit) without it "
                "to see how much the conclusions depend on it.".format(
                    row=row["_row"],
                    cov=row["covariates"][0],
                    size=_fmt(row["_size"]),
                    own=_advised(ctx)["own"],
                )
            ),
            fit,
            when=lambda row, ctx: on_dist(ctx) and "_row" in row,
            on=("warn",),
        ),
    ]


def _form_message(row: dict, own: str, scale: str) -> str:
    name = row["covariates"][0]
    return (
        "{name} does not act linearly on {scale} (adding its square "
        "improves the fit; see d.plot('functional_form')). Transform it, "
        "for example with a log in a formula, {own}.fit_from_df(..., "
        'formula="np.log({name}) + ..."), or bin it (with pandas.cut, '
        "say) and fit the bins as a categorical covariate, "
        'formula="C({name}_band) + ...".'.format(
            name=name, scale=scale, own=own
        )
    )


def _outlier_message(row: dict, ctx: Context) -> str:
    rows = row["_rows"]
    shown = ", ".join(str(i) for i in rows[:10])
    if len(rows) > 10:
        shown += " and {} more".format(len(rows) - 10)
    return (
        "The model predicts row{s} {shown} of model.data (the fitted rows, "
        "sorted) poorly (|deviance residual| above {cut}). Check {them} "
        "for recording errors.".format(
            s="" if len(rows) == 1 else "s",
            shown=shown,
            cut=_fmt(row["_cutoff"]),
            them="it" if len(rows) == 1 else "them",
        )
    )


def _better_baseline_rule(kind: str, check: str, name: str) -> AdviceRule:
    """Refit with the baseline ``name``, when the AIC screen of the family
    ``kind`` found it markedly better."""
    word = "baseline" if kind == "PH" else "error distribution"

    def message(row: dict, ctx: Context) -> str:
        return (
            "Another {word} fits markedly better: {label}'s AIC is {delta} "
            "lower than this model's (see d.plot('{check}')). Refit with "
            "{label}.fit and diagnose it.".format(
                word=word,
                label=row["_label"],
                delta=_fmt(row["_delta"]),
                check=check,
            )
        )

    def when(row: dict, ctx: Context) -> bool:
        return row.get("_aic") == "baselines" and row.get("_best") == name

    return AdviceRule(
        check, message, (name + kind + ".fit",), when=when, on=("warn",)
    )


def _other_kind_rule(kind: str, check: str, name: "str | None") -> AdviceRule:
    """Fit the other kind of model (PH for an AFT model, AFT for a PH
    one) with the baseline ``name`` (``None``: one not built in), when
    the AIC screen found it markedly better."""
    other = "AFT" if kind == "PH" else "PH"
    meaning = (
        "the covariates may rescale time rather than multiply the hazard"
        if kind == "PH"
        else "the covariates may multiply the hazard rather than rescale time"
    )
    on_dist = _dist_is(name)

    def message(row: dict, ctx: Context) -> str:
        return (
            "The {other} model with the same baseline, {label}, has an AIC "
            "{delta} lower: {meaning}. Fit {label}.fit and diagnose "
            "it.".format(
                other=(
                    "accelerated failure time"
                    if other == "AFT"
                    else "proportional hazards"
                ),
                label=row["_label"],
                delta=_fmt(row["_delta"]),
                meaning=meaning,
            )
        )

    def when(row: dict, ctx: Context) -> bool:
        return row.get("_aic") == "kind" and on_dist(ctx)

    feature = (other,) if name is None else (name + other + ".fit",)
    return AdviceRule(check, message, feature, when=when, on=("warn",))


def _aic_rules(kind: str, check: str) -> list[AdviceRule]:
    """The advice of the AIC screens of the family ``kind``: refit with
    the better baseline (one rule per built-in baseline), or with the
    other kind of model."""
    rules = [_better_baseline_rule(kind, check, name) for name in BASELINES]
    rules.extend(
        _other_kind_rule(kind, check, name)
        for name in list(BASELINES) + [None]
        if name not in SAME_PH_AFT
    )
    return rules


def _shape_rule(name: str, param: str) -> AdviceRule:
    """Fit each level on its own, or a stratified Cox model, when the
    shape ``param`` of the baseline ``name`` differs by level."""
    on_dist = _dist_is(name)

    def message(row: dict, ctx: Context) -> str:
        return (
            "The shape of the {name} ({param}) differs between the levels "
            "of {var} (it fails the test of one shape for every level), and "
            "an accelerated failure time model has one shape for all. Fit "
            "each level on its own, {name}AFT.fit on that level's rows (or "
            "{name}.fit where nothing else varies), so that each has its own "
            "shape; or fit a Cox model stratified on {var}, CoxPH.fit(..., "
            "strata=<the {var} column>) (from a DataFrame, "
            'CoxPH.fit_from_df(..., strata_col="{var}")), which gives each '
            "level its own baseline hazard.".format(
                name=name, param=param, var=row["_factor"]
            )
        )

    return AdviceRule(
        "common_shape",
        message,
        (
            name + "AFT.fit",
            name + ".fit",
            "CoxPH.fit(strata=)",
            "CoxPH.fit_from_df(strata_col=)",
        ),
        when=lambda row, ctx: on_dist(ctx),
    )


def _aft_form_rule(name: "str | None") -> AdviceRule:
    """Transform a covariate that does not act linearly on log time."""
    on_dist = _dist_is(name)
    feature = (
        ("AFT",) if name is None else (name + "AFT.fit_from_df(formula=)",)
    )
    return AdviceRule(
        "functional_form",
        lambda row, ctx: _form_message(
            row, _fitter_name(_base(ctx), "AFT"), "log time"
        ),
        feature,
        when=lambda row, ctx: on_dist(ctx),
    )


def _kind_of(kind: str) -> Callable[[Any], bool]:
    def applies(model: Any) -> bool:
        from surpyval.univariate.regression.parametric_regression_model import (  # noqa: E501
            ParametricRegressionModel,
        )

        return (
            isinstance(model, ParametricRegressionModel)
            and getattr(model, "kind", None) == kind
        )

    return applies


def _names(model: Any) -> list[str]:
    """The covariates, as ``summary()`` names the coefficients."""
    return list(model._coefficient_names())


#: The parametric proportional hazards family.
PH_FAMILY = register(
    Family(
        name="Parametric proportional hazards",
        options=("t0",),
        issue=869,
        applies=_kind_of(_PH),
        validate=_validate,
        covariates=_names,
        checks=(
            Check(
                "proportional_hazards",
                "Each covariate multiplies the hazard by a factor that does "
                "not change with time.",
                rows=_ph_rows,
                panels=_ph_panels,
            ),
            Check(
                "functional_form",
                "Each covariate acts linearly on the log hazard.",
                rows=_form_rows_of("the log hazard"),
                panels=_form_panels,
            ),
            Check(
                "influence",
                "No single unit decides a coefficient.",
                rows=_influence_rows,
                panels=_influence_panels,
            ),
            Check(
                "outliers",
                "No row is far worse predicted than the model allows.",
                rows=_outlier_rows,
                panels=_outlier_panels,
            ),
            Check(
                "baseline_distribution",
                "The baseline distribution fits: no other fits markedly "
                "better, and the Cox-Snell residuals follow the unit "
                "exponential.",
                rows=_aic_rows_of("PH"),
                panels=_distribution_panels_of("PH"),
            ),
            calibration_check(),
        ),
        advice=tuple(rule for name in BASELINES for rule in _ph_rules(name))
        + tuple(_ph_rules(None, same=False))
        + tuple(_ph_rules(None, same=True))
        + (AdviceRule("outliers", _outlier_message, (), on=("warn",)),)
        + tuple(_aic_rules("PH", "baseline_distribution"))
        + calibration_advice("PH", BASELINES, SAME_PH_AFT),
        notes=_ph_notes,
    )
)

#: The accelerated failure time family.
AFT_FAMILY = register(
    Family(
        name="Accelerated failure time",
        options=("t0",),
        issue=869,
        applies=_kind_of(_AFT),
        validate=_validate,
        covariates=_names,
        checks=(
            Check(
                "error_distribution",
                "The time on the baseline's scale, e^(beta'Z) x, follows "
                "the baseline distribution, and no other fits markedly "
                "better.",
                rows=_aic_rows_of("AFT"),
                panels=_distribution_panels_of("AFT"),
            ),
            Check(
                "common_shape",
                "Every level of a categorical covariate has the same shape: "
                "the covariates only rescale time.",
                rows=_shape_rows,
                panels=_shape_panels,
            ),
            Check(
                "functional_form",
                "Each covariate acts linearly on log time.",
                rows=_form_rows_of("log time"),
                panels=_form_panels,
            ),
            calibration_check(),
        ),
        advice=tuple(_aic_rules("AFT", "error_distribution"))
        + tuple(_shape_rule(name, param) for name, param in SHAPE.items())
        + tuple(_aft_form_rule(name) for name in list(BASELINES) + [None])
        + calibration_advice("AFT", BASELINES, SAME_PH_AFT),
        notes=_aft_notes,
    )
)
