"""``diagnose()`` for the shared-frailty models (#872).

Two families, one module: :data:`PARAMETRIC_FAMILY` for a
:class:`~surpyval.FrailtyModel` (``WeibullFrailty``, ...,
``Frailty(dist, family=...)``) and :data:`COX_FAMILY` for a
:class:`~surpyval.CoxFrailtyModel` (``CoxFrailty``). A shared-frailty
model is a proportional hazards model *given the frailty*,

.. math::
    h(t \\mid Z, u) = u \\, h_0(t) \\, e^{\\beta' Z},

with ``u`` shared by the rows of a group and drawn from a gamma (mean 1,
variance ``theta``) or a log-normal (``log u ~ N(0, theta)``)
distribution. Its assumptions are those of the conditional model and of
the frailty, so the checks below test those; they are not the checks of a
proportional hazards model refitted without the frailty, which would be
wrong for this model (see the notes).

The checks
----------
``frailty_variance`` (both models)
    The likelihood-ratio test of ``theta = 0``, no frailty: twice the
    frailty fit's log-likelihood less that of the same model without the
    frailty fitted to the same rows. For a :class:`~surpyval.FrailtyModel`
    that null model is the parametric proportional hazards fit of the
    same baseline, ``<Dist>PH.fit(x, Z, c, n)`` (``PH(dist).fit`` for a
    baseline that is not built in; ``<Dist>.fit(x, c, n)`` without
    covariates), whose likelihood is the frailty model's at ``theta =
    0``. For a :class:`~surpyval.CoxFrailtyModel` it is the Cox partial
    likelihood of ``CoxPH.fit(x, Z, c, n, tie_method=...)`` on the same
    rows, which ``CoxFrailty.fit`` computes as it fits (and stores as
    ``log_likelihood_no_frailty``), against the integrated ("I-")
    likelihood, as R's ``coxph(... + frailty(id, dist = "gamma"))``.
    ``theta = 0`` is on the boundary of its space, so under the null the
    statistic is a 50:50 mixture of a point mass at 0 and a chi-square on
    one degree of freedom (Self and Liang 1987): ``p = chi2_1.sf(T) / 2``
    for ``T > 0``, and 1 for ``T = 0`` (a fit with ``theta`` at 0). A
    screen, never a fail: the model holds either way. It is ``ok`` when
    ``p < alpha`` (the groups differ: the frailty is needed) and warns
    when ``p >= alpha`` (no evidence of a frailty: the model without it
    will do) or when the test cannot be computed. Drawn as each group's
    posterior frailty.
``frailty_family`` (parametric)
    A screen: the AIC of the fit against that of the same model with the
    other frailty family (``family="gamma"`` or ``"lognormal"``)
    refitted to the same rows. The two have the same number of
    parameters, so the difference is twice that of their
    log-likelihoods. It warns when the other's AIC is more than
    :data:`~surpyval.diagnostics.ph_aft.AIC_MARGIN` (2) lower; the
    ``statistic`` is the fitted model's AIC less the other's.
``proportional_hazards`` (parametric)
    For each covariate, the likelihood-ratio test (1 df) of a step in its
    coefficient *given the frailty* at the median failure time ``tau``:
    the frailty model with conditional hazard ``u h0(t) exp(beta'Z + [t >
    tau] (g0 + gamma Z_j))`` against ``gamma = 0``, each maximised over
    the marginal likelihood (the frailty integrated out, ``theta``
    re-estimated); and with two or more covariates the joint ``GLOBAL``
    test of a step in every coefficient. As in the parametric PH family
    the step ``g0`` of the baseline makes the test independent of a
    covariate's origin, and ``tau`` is the lower count-weighted median of
    the failure times.
``functional_form`` (parametric)
    For each covariate with three or more distinct values, the
    likelihood-ratio test (1 df) of adding its square to the conditional
    linear predictor, over the marginal likelihood. Drawn as the
    conditional martingale residuals ``delta - u_g exp(beta'Z) H0(x)``
    (each group's posterior frailty ``u_g``) against the covariate, with
    a lowess trend.
``baseline_distribution`` (parametric)
    A screen: the AIC of the fit against that of each other built-in
    baseline (:data:`~surpyval.diagnostics.ph_aft.BASELINES`) with the
    same frailty family refitted to the same rows, as the parametric PH
    family compares baselines. It warns when the best other's AIC is
    more than 2 lower. A candidate whose refit is not a verified maximum
    is set aside, and a note says so. Drawn as the AIC of each candidate.
``overall_fit`` (both)
    A plot only: the Nelson-Aalen cumulative hazard of the marginal
    Cox-Snell residuals ``H(x | Z)`` (the frailty integrated out) against
    the unit exponential's, the line ``y = x``. Each row's marginal
    residual is unit exponential when the model holds, whatever the
    frailties; the rows of a group are not independent, which leaves the
    curve's expectation as it is but widens its scatter, so it has no
    test here.

The tests of the conditional model (``proportional_hazards`` and
``functional_form``) maximise both of their models here, with the model's
own marginal likelihood (the gamma frailty's closed form, the
log-normal's adaptive Gauss-Hermite quadrature,
:func:`~surpyval.univariate.regression.frailty.families.lognormal_log_integral`),
the larger from the smaller's maximum, so that the statistic is never
negative.

A :class:`~surpyval.CoxFrailtyModel` does not keep the group of each row
it was fitted to, so its conditional model (proportional hazards and
functional form given the frailties) cannot be checked, and the Cox
family's checks of ``CoxPH`` refitted to the rows would test the marginal
model, which a frailty makes non-proportional; its report has the
``frailty_variance`` row and the ``overall_fit`` plot, and its notes say
why. Influence and outlier screens are left out for both models: the
residuals depend on each group's shrunken frailty estimate and the rows
of a group are not independent.
"""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2

from .cox import CATEGORICAL_LEVELS
from .ph_aft import AIC_MARGIN, BASELINES
from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The frailty families a parametric frailty model can have.
FRAILTY_FAMILIES = ("gamma", "lognormal")
#: The baselines with a gamma-frailty fitter built in (``WeibullFrailty``).
BUILT_IN_FRAILTY = ("Exponential", "Weibull", "LogNormal", "Gamma")
#: A boundary statistic at or below this is 0 (``theta`` on its boundary,
#: ``p = 1``): rounding, not evidence.
ZERO_STATISTIC = 1e-7
# Below these the frailty terms are their Taylor series in theta, where
# the closed form (gamma) or the quadrature (log-normal) loses digits to
# cancellation.
_GAMMA_SERIES = 1e-5
_LOGNORMAL_SERIES = 1e-8

_PARAMETRIC = "Parametric shared frailty"
_COX = "Cox shared frailty"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return "{:.3g}".format(value)


def _source(name: str) -> str:
    """The variable a formula-coded column comes from (``sex`` for
    ``C(sex)[T.2]`` or ``sex[T.2]``), else the name itself."""
    match = re.match(r"^C\(\s*([A-Za-z_][\w.]*)[^)]*\)\[", name)
    if match is None:
        match = re.match(r"^([A-Za-z_][\w.]*)\[", name)
    return match.group(1) if match else name


def _note(ctx: Context, text: str) -> None:
    if text not in ctx.notes:
        ctx.notes.append(text)


def _quiet(fit: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _is_cox(model: Any) -> bool:
    from surpyval.univariate.regression.frailty import CoxFrailtyModel

    return isinstance(model, CoxFrailtyModel)


def _names(model: Any) -> list[str]:
    """The coefficients' names, as ``summary()`` names them."""
    k = int(model.k_dist)
    return list(model.parameter_names[k : k + np.size(model.beta)])


def _data(ctx: Context) -> SimpleNamespace:
    """The rows the model was fitted to: ``x``, ``c``, ``w`` (counts),
    ``Z`` and, for a parametric model, ``inv`` (each row's group)."""

    def compute() -> SimpleNamespace:
        d = ctx.model._fit_data
        x = np.asarray(d["x"], dtype=float)
        Z = np.asarray(d["Z"], dtype=float).reshape(x.size, -1)
        w = np.asarray(d["w"] if "w" in d else d["n"], dtype=float)
        inv = d.get("inv")
        return SimpleNamespace(
            x=x,
            c=np.asarray(d["c"]).astype(int),
            w=w,
            Z=Z,
            inv=None if inv is None else np.asarray(inv, dtype=int),
        )

    return ctx.cached("data", compute)


def _kind(ctx: Context, j: int) -> str:
    """``"binary"``, ``"categorical"`` or ``"continuous"``, by the number
    of distinct values of covariate ``j`` (as the Cox family counts)."""
    levels = np.unique(_data(ctx).Z[:, j]).size
    if levels <= 2:
        return "binary"
    if levels <= CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _dist_name(model: Any) -> str:
    return str(model.dist.name)


def frailty_fitter_name(model: Any, family: "str | None" = None) -> str:
    """How the advice names the fitter of a parametric frailty model with
    ``model``'s baseline and frailty ``family`` (``model``'s own by
    default): ``WeibullFrailty`` for a built-in gamma one, else
    ``Frailty(Weibull, family='lognormal')``.

    Examples
    --------
    >>> from surpyval import FrailtyModel, Weibull
    >>> from surpyval.diagnostics.frailty import frailty_fitter_name
    >>> model = FrailtyModel()
    >>> model.dist = Weibull
    >>> frailty_fitter_name(model), frailty_fitter_name(model, "lognormal")
    ('WeibullFrailty', "Frailty(Weibull, family='lognormal')")
    """
    family = model.family if family is None else family
    name = _dist_name(model)
    if family == "gamma" and name in BUILT_IN_FRAILTY:
        return name + "Frailty"
    return "Frailty({}, family={!r})".format(name, family)


def null_fitter_name(model: Any) -> str:
    """The fitter of the model without the frailty, as the advice and the
    ``frailty_variance`` row name it: ``CoxPH`` for a Cox frailty model;
    for a parametric one ``<Dist>PH`` (``PH(<Dist>)`` for a baseline
    that is not built in), or the distribution itself without covariates.

    Examples
    --------
    >>> from surpyval import FrailtyModel, Weibull
    >>> from surpyval.diagnostics.frailty import null_fitter_name
    >>> model = FrailtyModel()
    >>> model.dist = Weibull
    >>> model.beta = [0.5]
    >>> null_fitter_name(model)
    'WeibullPH'
    """
    if _is_cox(model):
        return "CoxPH"
    name = _dist_name(model)
    if not np.size(model.beta):
        return name
    if name in BASELINES:
        return name + "PH"
    return "PH({})".format(name)


# ---------------------------------------------------------------------------
# The zero-variance test
# ---------------------------------------------------------------------------
def _boundary_p(statistic: float) -> float:
    """The p-value of a boundary likelihood-ratio statistic under the
    50:50 mixture of chi-square(0) and chi-square(1)."""
    if not np.isfinite(statistic):
        return float("nan")
    if statistic <= ZERO_STATISTIC:
        return 1.0
    return float(0.5 * chi2.sf(statistic, 1))


def _parametric_null(model: Any, data: SimpleNamespace) -> tuple[float, str]:
    """``(log-likelihood, maximum)`` of the parametric model without the
    frailty refitted to ``data``."""
    import surpyval

    name = _dist_name(model)
    if not np.size(model.beta):
        fitted = _quiet(model.dist.fit, data.x, c=data.c, n=data.w)
        ll = -float(fitted.neg_ll())
        return ll, "verified"
    fitter = (
        getattr(surpyval, name + "PH")
        if name in BASELINES
        else surpyval.PH(model.dist)
    )
    fitted = _quiet(fitter.fit, data.x, data.Z, c=data.c, n=data.w)
    return -float(fitted.neg_ll()), str(fitted.maximum)


def zero_variance_test(model: Any) -> dict:
    """
    The likelihood-ratio test of no frailty (``theta = 0``) of a fitted
    shared-frailty model, the ``frailty_variance`` row of its
    ``diagnose()``.

    The null model is the same model without the frailty fitted to the
    same rows (:func:`null_fitter_name` names its fitter); the statistic
    is twice the difference of the log-likelihoods (the frailty fit's
    marginal likelihood, or its integrated likelihood for a Cox frailty
    model, against the Cox partial likelihood), and its p-value that of
    the 50:50 mixture of chi-square(0) and chi-square(1), since ``theta =
    0`` is on the boundary.

    Parameters
    ----------
    model : FrailtyModel or CoxFrailtyModel
        A model fitted with ``fit`` or ``fit_from_df`` (it keeps the rows
        it was fitted to).

    Returns
    -------
    dict
        ``statistic``, ``df`` (1), ``p``, ``loglik`` (the frailty fit's),
        ``loglik_null``, ``null`` (the null model's fitter) and
        ``reason`` (why the test could not be computed, else ``""``).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import CoxFrailty
    >>> from surpyval.datasets import load_kidney
    >>> from surpyval.diagnostics.frailty import zero_variance_test
    >>> df = load_kidney()
    >>> Z = np.column_stack([df["age"], df["sex"] == 2])
    >>> model = CoxFrailty.fit(
    ...     df["time"], Z=Z, c=1 - df["status"], groups=df["id"]
    ... )
    >>> res = zero_variance_test(model)
    >>> res["null"], round(res["statistic"], 4), round(res["p"], 5)
    ('CoxPH', 5.4119, 0.00999)
    """
    data = getattr(model, "_fit_data", None)
    if data is None:
        raise ValueError(
            "The test needs the rows the model was fitted to, which a model "
            "restored with from_dict does not keep; refit it."
        )
    out = {
        "df": 1,
        "null": null_fitter_name(model),
        "loglik": float("nan"),
        "loglik_null": float("nan"),
        "statistic": float("nan"),
        "p": float("nan"),
        "reason": "",
    }
    if _is_cox(model):
        loglik = float(model.log_likelihood)
        null = float(model.log_likelihood_no_frailty)
        maximum = "verified"
    else:
        d = SimpleNamespace(
            x=np.asarray(data["x"], dtype=float),
            c=np.asarray(data["c"]).astype(int),
            w=np.asarray(data["w"], dtype=float),
            Z=np.asarray(data["Z"], dtype=float).reshape(
                np.size(data["x"]), -1
            ),
        )
        loglik = -float(model.neg_ll())
        try:
            null, maximum = _parametric_null(model, d)
        except Exception as error:  # noqa: BLE001 -- reported in the row
            text = str(error).strip().splitlines()
            out["reason"] = (
                "the model without frailty could not be fitted: "
                + (text[0][:80] if text else type(error).__name__)
            )
            return out
    out["loglik"], out["loglik_null"] = loglik, null
    if maximum != "verified" or not np.isfinite(null):
        out["reason"] = (
            "the fit without frailty is not a verified maximum "
            "({})".format(maximum)
        )
        return out
    statistic = max(0.0, 2.0 * (loglik - null))
    if statistic <= ZERO_STATISTIC:
        statistic = 0.0
    out["statistic"] = statistic
    out["p"] = _boundary_p(statistic)
    return out


def _variance_fixed(model: Any) -> bool:
    """Whether ``theta`` was given (``CoxFrailty.fit(theta=)``), not
    estimated: ``k`` then counts the coefficients only."""
    if not _is_cox(model):
        return False
    return int(model.k) == int(np.isfinite(np.asarray(model.beta)).sum())


def _variance_rows(ctx: Context) -> list[dict]:
    model = ctx.model
    if _variance_fixed(model):
        _note(
            ctx,
            "The frailty variance was fixed (theta={:g}), not estimated, so "
            "it is not tested.".format(float(model.theta)),
        )
        return []
    res = ctx.cached("zero_variance", lambda: zero_variance_test(model))
    p = res["p"]
    if not np.isfinite(p):
        verdict, detail = "warn", "the test could not be computed: " + (
            res["reason"] or "a log-likelihood is not finite"
        )
    elif p < ctx.alpha:
        verdict = "ok"
        detail = "the groups differ: theta = {} is significant".format(
            _fmt(float(model.theta))
        )
    else:
        verdict = "warn"
        if res["statistic"] == 0:
            detail = "theta is on its boundary, 0: no frailty is detected"
        else:
            detail = "no evidence of a frailty (theta = {})".format(
                _fmt(float(model.theta))
            )
    return [
        {
            "test": "boundary likelihood-ratio test against {} (50:50 "
            "chi2(0):chi2(1))".format(res["null"]),
            "covariates": (),
            "statistic": res["statistic"],
            "df": 1,
            "p": p,
            "verdict": verdict,
            "detail": detail,
            "_p": p,
            "_null": res["null"],
        }
    ]


def _variance_panels(ctx: Context) -> list[Panel]:
    frailties = np.sort(
        np.asarray(list(ctx.model.frailties.values()), dtype=float)
    )

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.scatter(
            np.arange(frailties.size), frailties, s=12, color=PALETTE[0]
        )
        ax.axhline(1.0, color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("group (sorted)")
        ax.set_ylabel("posterior frailty")

    return [Panel("frailties: theta = {}".format(_fmt(ctx.model.theta)), draw)]


# ---------------------------------------------------------------------------
# The marginal likelihood of a larger model (parametric frailty)
# ---------------------------------------------------------------------------
def _group_sum_primitive() -> Callable:
    from autograd.extend import defvjp, primitive

    @primitive
    def group_sum(values: Any, inv: np.ndarray, size: int) -> np.ndarray:
        return np.bincount(inv, weights=values, minlength=size)

    defvjp(group_sum, lambda ans, values, inv, size: lambda g: g[inv])
    return group_sum


_GROUP_SUM: list = []


def _group_sum(values: Any, inv: np.ndarray, size: int) -> Any:
    if not _GROUP_SUM:
        _GROUP_SUM.append(_group_sum_primitive())
    return _GROUP_SUM[0](values, inv, size)


def _gamma_term(D: np.ndarray, H: Any, theta: Any) -> Any:
    """Each group's ``log E[u^D exp(-u H)]`` for a gamma frailty of mean
    1 and variance ``theta``: the closed form, or its series in ``theta``
    to second order near 0, where the closed form cancels."""
    import autograd.numpy as anp
    from autograd.scipy.special import gammaln
    from autograd.tracer import getval

    if float(getval(theta)) < _GAMMA_SERIES:
        first = D * (D - 1.0) / 2.0 - D * H + H**2 / 2.0
        second = -D * (D - 1.0) * (2.0 * D - 1.0) / 12.0 - H**3 / 3.0
        second = second + D * H**2 / 2.0
        return -H + theta * first + theta**2 * second
    a = 1.0 / theta
    rising = gammaln(D + a) - gammaln(a) - D * anp.log(a)
    return rising - (D + a) * anp.log1p(H * theta)


def _lognormal_term(D: np.ndarray, H: Any, theta: Any) -> Any:
    """Each group's ``log E[u^D exp(-u H)]`` for ``log u ~ N(0, theta)``:
    the package's quadrature, or its series to first order near 0."""
    from autograd.tracer import getval

    from surpyval.univariate.regression.frailty.families import (
        lognormal_log_integral,
    )

    if float(getval(theta)) < _LOGNORMAL_SERIES:
        return -H + theta * ((D - H) ** 2 - D) / 2.0
    return lognormal_log_integral(D, H, theta)


class _Marginal:
    """The marginal negative log-likelihood of a parametric frailty model
    with extra terms in its conditional linear predictor, as a function of
    the natural parameters ``[*baseline, *beta, theta, *extra]``.

    ``always`` (``n x a``) are columns that enter at every time (a
    covariate's square), with the first ``a`` extra coefficients; with a
    cut ``tau``, the next extra coefficient is the step ``g0`` of the log
    baseline after ``tau`` and the rest the steps of the coefficients of
    the columns ``steps`` (``n x s``) after it."""

    def __init__(
        self,
        model: Any,
        data: SimpleNamespace,
        always: "np.ndarray | None" = None,
        tau: "float | None" = None,
        steps: "np.ndarray | None" = None,
    ) -> None:
        self.dist = model.dist
        self.family = model.family
        self.k = int(model.k_dist)
        self.p = data.Z.shape[1]
        self.data = data
        n = data.x.size
        self.always = np.zeros((n, 0)) if always is None else always
        self.tau = tau
        self.steps = np.zeros((n, 0)) if steps is None else steps
        self.n_extra = self.always.shape[1] + (
            0 if tau is None else 1 + self.steps.shape[1]
        )
        self.event = data.c == 0
        self.size = int(data.inv.max()) + 1
        self.D = np.bincount(
            data.inv, weights=data.w * self.event, minlength=self.size
        )
        if tau is not None:
            self.late = data.x > tau
            self.x_early = np.minimum(data.x, tau)

    def __call__(self, nat: Any) -> Any:
        import autograd.numpy as anp

        d = self.data
        k, p = self.k, self.p
        params = [nat[i] for i in range(k)]
        beta = nat[k : k + p]
        theta = nat[k + p]
        extra = nat[k + p + 1 :]
        a = self.always.shape[1]
        lin = anp.dot(d.Z, beta) if p else 0.0 * d.x
        if a:
            lin = lin + anp.dot(self.always, extra[:a])
        H0 = self.dist.Hf(d.x, *params)
        log_h = anp.log(self.dist.hf(d.x, *params)) + lin
        if self.tau is None:
            cum = anp.exp(lin) * H0
        else:
            shift = extra[a] + (
                anp.dot(self.steps, extra[a + 1 :])
                if self.steps.shape[1]
                else 0.0
            )
            H_tau = self.dist.Hf(self.x_early, *params)
            cum = anp.exp(lin) * (
                H_tau + anp.where(self.late, anp.exp(shift) * (H0 - H_tau), 0)
            )
            log_h = log_h + anp.where(self.late, shift, 0.0)
        events = anp.sum(anp.where(self.event, d.w * log_h, 0.0))
        H = _group_sum(d.w * cum, d.inv, self.size)
        if self.family == "lognormal":
            groups = _lognormal_term(self.D, H, theta)
        else:
            groups = _gamma_term(self.D, H, theta)
        return -(events + anp.sum(groups))


class _Fit(SimpleNamespace):
    """A maximum found by :func:`_maximise`: ``theta`` (the natural
    parameters), ``value`` (the negative log-likelihood) and ``ok``."""

    theta: np.ndarray
    value: float
    ok: bool


def _maximise(f: _Marginal, start: np.ndarray) -> _Fit:
    """Minimise ``f`` over its parameters from ``start``, with the
    baseline's positive parameters on their log, the coefficients on
    their covariates' scale and ``theta >= 0``: Newton's method on the
    exact (autograd) gradient and Hessian, with step halving and
    ``theta`` held at 0 where the likelihood falls off that boundary;
    where it does not get there (from far off, or where the Hessian is
    not positive definite), L-BFGS-B, then Newton again. ``ok`` when the
    gradient is zero (to ``1e-6`` relative) with a positive-definite
    Hessian, ``theta`` aside where it is held at 0."""
    import autograd.numpy as anp
    from autograd import grad, hessian
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=float)
    j_theta = f.k + f.p
    log_scale = np.zeros(start.size, dtype=bool)
    for i, (low, high) in enumerate(f.dist.bounds[: f.k]):
        log_scale[i] = low == 0 and high is None
    columns = [f.data.Z, np.ones((f.data.x.size, 1)), f.always]
    if f.tau is not None:
        columns.append(f.late[:, None].astype(float))
        columns.append(f.steps * f.late[:, None])
    spread = np.concatenate([np.std(c, axis=0) for c in columns])
    spread[f.p] = 1.0  # theta
    scale = np.ones(start.size)
    scale[f.k :] = np.where(spread > 0, spread, 1.0)

    def full(u: Any) -> Any:
        return anp.where(log_scale, anp.exp(u), u / scale)

    def obj(u: Any) -> Any:
        return f(full(u))

    g, h = grad(obj), hessian(obj)

    def newton(u: np.ndarray, value: float) -> tuple:
        """Newton's steps from ``u``: ``(u, value, ok)``."""
        ok = False
        for _ in range(50):
            try:
                gradient = np.asarray(g(u), dtype=float)
                curvature = np.asarray(h(u), dtype=float)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                break
            active = np.ones(u.size, dtype=bool)
            if u[j_theta] <= 0 and gradient[j_theta] >= 0:
                active[j_theta] = False
            if not (
                np.all(np.isfinite(gradient[active]))
                and np.all(np.isfinite(curvature))
            ):
                break
            sub = curvature[np.ix_(active, active)]
            try:
                definite = bool(np.all(np.linalg.eigvalsh(sub) > 0))
            except np.linalg.LinAlgError:
                definite = False
            size = float(np.max(np.abs(gradient[active]), initial=0.0))
            ok = definite and size < 1e-6 * max(1.0, abs(value))
            if not definite or size < 1e-10 * max(1.0, abs(value)):
                break
            step = np.zeros(u.size)
            step[active] = -np.linalg.solve(sub, gradient[active])
            moved = False
            for _ in range(40):
                trial = u + step
                trial[j_theta] = max(trial[j_theta], 0.0)
                new = float(obj(trial))
                if np.isfinite(new) and new <= value:
                    moved = new < value
                    u, value = trial, new
                    break
                step = step / 2.0
            if not moved:
                break
        return u, value, ok

    first = start.copy()
    first[j_theta] = max(first[j_theta], 0.0)
    with np.errstate(all="ignore"):
        u = np.where(log_scale, np.log(np.abs(first)), first * scale)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(obj(u))
        if not np.isfinite(value):
            return _Fit(theta=start, value=float("nan"), ok=False)
        u, value, ok = newton(u, value)
        if not ok:
            bounds: "list[tuple[float | None, None]]" = [(None, None)] * u.size
            bounds[j_theta] = (0.0, None)
            try:
                res = minimize(
                    lambda v: float(obj(v)),
                    u,
                    jac=lambda v: np.asarray(g(v), dtype=float),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
                )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                return _Fit(theta=start, value=float("nan"), ok=False)
            if np.isfinite(res.fun) and res.fun <= value:
                u, value, ok = newton(np.asarray(res.x, dtype=float), res.fun)
    return _Fit(theta=np.asarray(full(u), dtype=float), value=value, ok=ok)


def _start(model: Any) -> np.ndarray:
    """The fitted natural parameters ``[*baseline, *beta, theta]``."""
    return np.concatenate(
        [
            np.asarray(model.dist_params, dtype=float),
            np.asarray(model.beta, dtype=float),
            [float(model.theta)],
        ]
    )


def _null_fit(ctx: Context) -> _Fit:
    """The fitted model's maximum, confirmed with :func:`_maximise` on
    the likelihood of :class:`_Marginal` (from the fitted parameters):
    the null of every likelihood-ratio test of the conditional model."""

    def compute() -> _Fit:
        f = _Marginal(ctx.model, _data(ctx))
        start = _start(ctx.model)
        found = _maximise(f, start)
        at_fit = float(f(start))
        if np.isfinite(at_fit) and (
            not np.isfinite(found.value) or at_fit < found.value
        ):
            return _Fit(theta=start, value=at_fit, ok=found.ok)
        return found

    return ctx.cached("null_fit", compute)


def _lr(ctx: Context, f: _Marginal, null: "_Fit | None" = None) -> dict:
    """The likelihood-ratio test of a null model against the larger model
    ``f``: by default the fitted model, with every extra coefficient of
    ``f`` 0; else ``null``, a maximum of ``f``'s first parameters (the
    rest 0 under it). The larger model is maximised from the null's
    maximum, so the statistic is never negative."""
    null = _null_fit(ctx) if null is None else null
    df = f.k + f.p + 1 + f.n_extra - null.theta.size
    nan = {"statistic": np.nan, "df": df, "p": np.nan, "fit": None}
    if not null.ok:
        return dict(
            nan, reason="the smaller model's maximum could not be confirmed"
        )
    start = np.concatenate([null.theta, np.zeros(df)])
    # theta off its boundary, so that the search sees whether the larger
    # model wants a frailty
    j = f.k + f.p
    start[j] = max(start[j], 1e-4)
    alt = _maximise(f, start)
    if not alt.ok:
        return dict(
            nan,
            reason="the larger model's fit did not reach a verified maximum",
        )
    stat = max(0.0, 2.0 * (null.value - alt.value))
    return {
        "statistic": stat,
        "df": df,
        "p": float(chi2.sf(stat, df)),
        "fit": alt,
        "reason": "",
    }


# ---------------------------------------------------------------------------
# Proportional hazards given the frailty (parametric)
# ---------------------------------------------------------------------------
def _cut(ctx: Context) -> float:
    """The lower count-weighted median of the failure times."""
    d = _data(ctx)
    event = d.c == 0
    t, w = d.x[event], d.w[event]
    order = np.argsort(t, kind="stable")
    t, w = t[order], w[order]
    cumulative = np.cumsum(w)
    return float(t[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _step_null(ctx: Context) -> _Fit:
    """The fitted model with a step ``g0`` in its log baseline at the
    cut, maximised: the null of the tests of a step in a coefficient."""

    def compute() -> _Fit:
        f = _Marginal(ctx.model, _data(ctx), tau=_cut(ctx))
        res = _lr(ctx, f)
        if res["fit"] is None:
            return _Fit(theta=np.zeros(0), value=float("nan"), ok=False)
        fit: _Fit = res["fit"]
        return fit

    return ctx.cached("step_null", compute)


def _step_test(ctx: Context, J: list[int]) -> dict:
    """The test of a step at the cut in the coefficients ``J`` (and in
    the baseline, under both models)."""

    def compute() -> dict:
        d = _data(ctx)
        f = _Marginal(ctx.model, d, tau=_cut(ctx), steps=d.Z[:, J])
        null = _step_null(ctx)
        if not null.ok:
            return {
                "statistic": np.nan,
                "df": len(J),
                "p": np.nan,
                "fit": None,
                "reason": "the model with a step in the baseline did not "
                "reach a verified maximum",
            }
        return _lr(ctx, f, null)

    return ctx.cached("step_{}".format(J), compute)


def _ph_rows(ctx: Context) -> list[dict]:
    if not ctx.names:
        return []
    tau = _cut(ctx)
    rows = []
    for j, name in ctx.checked:
        res = _step_test(ctx, [j])
        p = res["p"]
        if not np.isfinite(p):
            detail = "the test could not be computed: " + res["reason"]
        elif p < 2 * ctx.alpha:
            gamma = float(res["fit"].theta[-1])
            detail = (
                "given the frailty, its coefficient {} after t = {}".format(
                    "rises" if gamma > 0 else "falls", _fmt(tau)
                )
            )
        else:
            detail = "no evidence that its coefficient changes with time"
        rows.append(
            {
                "test": "likelihood-ratio test of a step at t = {}".format(
                    _fmt(tau)
                ),
                "covariates": (name,),
                "statistic": res["statistic"],
                "df": res["df"],
                "p": p,
                "detail": detail,
                "_kind": _kind(ctx, j),
            }
        )
    if len(ctx.names) > 1:
        res = _step_test(ctx, list(range(len(ctx.names))))
        rows.append(
            {
                "test": "likelihood-ratio test of a step at t = {} "
                "(GLOBAL)".format(_fmt(tau)),
                "covariates": tuple(ctx.names),
                "statistic": res["statistic"],
                "df": res["df"],
                "p": res["p"],
                "detail": (
                    "the joint test of every covariate"
                    if np.isfinite(res["p"])
                    else "the test could not be computed: " + res["reason"]
                ),
                "_global": True,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Functional form given the frailty (parametric)
# ---------------------------------------------------------------------------
def _form_columns(ctx: Context) -> list[tuple[int, str]]:
    return [(j, n) for j, n in ctx.checked if _kind(ctx, j) != "binary"]


def _form_rows(ctx: Context) -> list[dict]:
    d = _data(ctx)
    skipped = [n for j, n in ctx.checked if _kind(ctx, j) == "binary"]
    if skipped:
        _note(
            ctx,
            "Functional form is tested for covariates with three or more "
            "distinct values; {} {} two, and the effect of a two-valued "
            "covariate is linear by construction.".format(
                ", ".join(skipped), "has" if len(skipped) == 1 else "have"
            ),
        )
    rows = []
    for j, name in _form_columns(ctx):

        def compute(j: int = j) -> dict:
            square = d.Z[:, j] ** 2
            spread = float(np.std(square)) or 1.0
            f = _Marginal(ctx.model, d, always=(square / spread)[:, None])
            return _lr(ctx, f)

        res = ctx.cached("square_{}".format(j), compute)
        p = res["p"]
        if not np.isfinite(p):
            detail = "the test could not be computed: " + res["reason"]
        elif p < 2 * ctx.alpha:
            detail = (
                "given the frailty, its effect on the log hazard is not linear"
            )
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


def _conditional_martingale(ctx: Context) -> np.ndarray:
    """``delta - u_g exp(beta'Z) H0(x)`` of each row, ``u_g`` its group's
    posterior frailty."""
    model = ctx.model
    d = _data(ctx)
    labels = [str(g) for g in model.group_labels]
    u = np.array([model.frailties[labels[i]] for i in d.inv], dtype=float)
    eta = np.exp(d.Z @ np.asarray(model.beta, dtype=float))
    H0 = model.dist.Hf(d.x, *model.dist_params)
    return (d.c == 0).astype(float) - u * eta * H0


def _form_panels(ctx: Context) -> list[Panel]:
    from .smooth import lowess

    Z = _data(ctx).Z
    martingale = _conditional_martingale(ctx)
    panels = []
    for j, name in _form_columns(ctx):

        def draw(ax: Any, options: Any, j: int = j, name: str = name) -> None:
            from surpyval.plotting import INK, PALETTE

            z = Z[:, j]
            grid = np.linspace(z.min(), z.max(), 60)
            ax.scatter(z, martingale, s=10, color=INK["muted"], alpha=0.6)
            ax.plot(grid, lowess(z, martingale, grid), color=PALETTE[0])
            ax.axhline(0.0, color=INK["axis"], lw=0.8)
            ax.set_xlabel(name)
            ax.set_ylabel("martingale residual, given the frailty")

        panels.append(Panel("{}: functional form".format(name), draw))
    return panels


# ---------------------------------------------------------------------------
# The AIC screens (parametric)
# ---------------------------------------------------------------------------
def _refit(ctx: Context, dist: Any, family: str) -> dict:
    """The model with baseline ``dist`` and frailty ``family`` refitted to
    the same rows: ``{"aic", "status", "reason"}``, status ``"ranked"``
    (a verified maximum), ``"set aside"`` or ``"failed"``."""
    import surpyval

    d = _data(ctx)
    fitter = surpyval.Frailty(dist, family=family)
    try:
        fitted = _quiet(
            fitter.fit,
            d.x,
            d.Z if d.Z.shape[1] else None,
            c=d.c,
            n=d.w,
            groups=d.inv,
        )
        aic = float(fitted.aic())
    except Exception as error:  # noqa: BLE001 -- a candidate may fail
        text = str(error).strip().splitlines()
        return {
            "aic": float("nan"),
            "status": "failed",
            "reason": text[0][:100] if text else type(error).__name__,
        }
    if fitted.maximum != "verified" or not np.isfinite(aic):
        return {
            "aic": aic,
            "status": "set aside",
            "reason": "not a verified maximum ({})".format(fitted.maximum),
        }
    return {"aic": aic, "status": "ranked", "reason": ""}


def _gap(delta: float) -> str:
    if delta == 0:
        return "level with it"
    return "{} {}".format(_fmt(abs(delta)), "lower" if delta > 0 else "higher")


def _family_rows(ctx: Context) -> list[dict]:
    model = ctx.model
    other = [f for f in FRAILTY_FAMILIES if f != model.family][0]
    res = ctx.cached("family", lambda: _refit(ctx, model.dist, other))
    label = frailty_fitter_name(model, other)
    own = float(model.aic())
    if res["status"] != "ranked":
        return [
            {
                "test": "AIC against the {} frailty".format(other),
                "covariates": (),
                "statistic": float("nan"),
                "verdict": "warn",
                "detail": "the {} refit could not be compared: {}".format(
                    other, res["reason"]
                ),
            }
        ]
    delta = own - res["aic"]
    if delta > AIC_MARGIN:
        detail = "{} fits better: its AIC is {} lower".format(
            label, _fmt(delta)
        )
    else:
        detail = (
            "the {} frailty's AIC is not more than {:g} lower (it is "
            "{})".format(other, AIC_MARGIN, _gap(delta))
        )
    return [
        {
            "test": "AIC against the {} frailty".format(other),
            "covariates": (),
            "statistic": delta,
            "verdict": "warn" if delta > AIC_MARGIN else "ok",
            "detail": detail,
            "_other": other,
            "_delta": delta,
        }
    ]


def _baseline_candidates(ctx: Context) -> list[dict]:
    def compute() -> list[dict]:
        import surpyval

        model = ctx.model
        own = _dist_name(model)
        out = [
            {
                "label": frailty_fitter_name(model),
                "baseline": own,
                "aic": float(model.aic()),
                "status": "fitted",
                "reason": "",
            }
        ]
        for name in BASELINES:
            if name == own:
                continue
            res = _refit(ctx, getattr(surpyval, name), model.family)
            out.append(
                dict(
                    res,
                    baseline=name,
                    label="Frailty({}, family={!r})".format(
                        name, model.family
                    ),
                )
            )
        return out

    return ctx.cached("baselines", compute)


def _baseline_rows(ctx: Context) -> list[dict]:
    candidates = _baseline_candidates(ctx)
    own = candidates[0]
    others = [c for c in candidates[1:] if c["status"] == "ranked"]
    skipped = [c for c in candidates[1:] if c["status"] != "ranked"]
    if skipped:
        _note(
            ctx,
            "Baseline comparison: {} left out ({}).".format(
                ", ".join(c["baseline"] for c in skipped),
                "; ".join(
                    "{}: {}".format(c["baseline"], c["reason"])
                    for c in skipped
                ),
            ),
        )
    if not others:
        return []
    best = min(others, key=lambda c: c["aic"])
    delta = own["aic"] - best["aic"]
    if delta > AIC_MARGIN:
        detail = "the {} baseline fits better: its AIC is {} lower".format(
            best["baseline"], _fmt(delta)
        )
    else:
        detail = (
            "no other baseline has an AIC more than {:g} lower (the best of "
            "them, {}, is {})".format(
                AIC_MARGIN, best["baseline"], _gap(delta)
            )
        )
    return [
        {
            "test": "AIC against the other baselines",
            "covariates": (),
            "statistic": delta,
            "verdict": "warn" if delta > AIC_MARGIN else "ok",
            "detail": detail,
            "_best": best["baseline"],
            "_delta": delta,
        }
    ]


def _baseline_panels(ctx: Context) -> list[Panel]:
    candidates = [
        c
        for c in _baseline_candidates(ctx)
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
        ax.set_yticklabels([c["baseline"] for c in order])
        ax.invert_yaxis()
        ax.axvline(AIC_MARGIN, color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("AIC - lowest AIC")

    return [
        Panel("AIC by baseline ({} frailty)".format(ctx.model.family), draw)
    ]


# ---------------------------------------------------------------------------
# Overall fit (both)
# ---------------------------------------------------------------------------
def _overall_panels(ctx: Context) -> list[Panel]:
    model = ctx.model
    d = _data(ctx)
    with np.errstate(all="ignore"):
        if d.Z.shape[1]:
            r = np.asarray(model.Hf(d.x, d.Z), dtype=float)
        else:
            r = np.asarray(model.Hf(d.x), dtype=float) * np.ones(d.x.size)
    c, n = d.c, d.w
    order = np.argsort(r, kind="stable")
    r, c, n = r[order], c[order], n[order]
    uniq, first = np.unique(r, return_index=True)
    at_risk = n[::-1].cumsum()[::-1][first]
    deaths = np.bincount(
        np.searchsorted(uniq, r), weights=n * (c == 0), minlength=uniq.size
    )
    H = np.cumsum(deaths / at_risk)
    keep = deaths > 0

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.step(uniq[keep], H[keep], where="post", color=PALETTE[0])
        top = float(max(uniq[keep].max(), H[keep].max())) if keep.any() else 1
        ax.plot([0, top], [0, top], color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("marginal Cox-Snell residual")
        ax.set_ylabel("cumulative hazard")

    return [Panel("overall fit (marginal Cox-Snell)", draw)]


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    cox = _is_cox(model)
    fitter = "CoxFrailty" if cox else frailty_fitter_name(model)
    if getattr(model, "_fit_data", None) is None:
        raise ValueError(
            "diagnose() needs the rows the model was fitted to, which a "
            "model restored with from_dict does not keep. Refit it, with "
            "{}.fit or {}.fit_from_df, and diagnose the fit.".format(
                fitter, fitter
            )
        )
    maximum = getattr(model, "maximum", "verified")
    if maximum == "no finite maximum":
        raise ValueError(
            "diagnose() needs a fit with a finite maximum, and this one has "
            "none (model.maximum is 'no finite maximum', as the fit "
            "warned): its tests compare the fit with other models, which "
            "means nothing at no maximum. Drop or merge the covariate that "
            "separates the events, or fix theta, and refit."
        )
    if maximum != "verified":
        raise ValueError(
            "diagnose() needs a fit that reached a verified maximum, and "
            "this one did not (model.maximum is {!r}, as the fit warned): "
            "its tests compare the fit with other models. Refit it (with "
            "init= from a nearby fit for {}).".format(maximum, fitter)
        )
    aliased = np.flatnonzero(np.isnan(np.asarray(model.beta, dtype=float)))
    if aliased.size:
        names = _names(model)
        raise ValueError(
            "diagnose() needs every coefficient estimated, and {} {} "
            "aliased (the data cannot determine {}: a constant column, or "
            "a combination of the others). Drop {} from Z and "
            "refit.".format(
                ", ".join(names[j] for j in aliased),
                "is" if aliased.size == 1 else "are",
                "it" if aliased.size == 1 else "them",
                "it" if aliased.size == 1 else "them",
            )
        )


_MIXTURE = (
    "theta = 0 is on the boundary, so the p-value is that of the 50:50 "
    "mixture of chi2(0) and chi2(1) (half the chi2(1) p-value). A screen: "
    "it warns when the frailty is not significant."
)


def _variance_note(ctx: Context) -> str:
    if _is_cox(ctx.model):
        return (
            "Frailty variance: the likelihood-ratio test of theta = 0, twice "
            "the integrated log-likelihood (R's I-likelihood) less the "
            "partial log-likelihood of CoxPH.fit(x, Z, c, n, "
            "tie_method={!r}) on the same rows, the model without the "
            "frailty (which CoxFrailty.fit computes as it fits, "
            "log_likelihood_no_frailty); ".format(ctx.model.tie_method)
            + _MIXTURE
        )
    return (
        "Frailty variance: the likelihood-ratio test of theta = 0 against "
        "{}.fit, the same model without the frailty, fitted here to the "
        "same rows; ".format(null_fitter_name(ctx.model)) + _MIXTURE
    )


def _parametric_notes(ctx: Context) -> list[str]:
    name = null_fitter_name(ctx.model)
    notes = [
        _variance_note(ctx),
        "Proportional hazards and functional form are tested given the "
        "frailty: each larger model's marginal likelihood (the frailty "
        "integrated out, theta re-estimated) against the fit's. The "
        "checks of {name}.fit(...).diagnose() on these data would test "
        "the marginal model instead, which a frailty makes "
        "non-proportional (the hazard ratio fades as the frailer units "
        "fail), so they are not used.".format(name=name),
        "Influence and outliers are not screened: the residuals of a "
        "frailty model depend on each group's estimated (shrunken) "
        "frailty, and the rows of a group are not independent, so the "
        "one-row-at-a-time screens and their Bonferroni cutoff do not "
        "apply.",
        "Overall fit is a plot only (d.plot('overall_fit')): the marginal "
        "Cox-Snell residuals of a group are dependent, so their curve "
        "has no simple test.",
    ]
    return notes


def _cox_notes(ctx: Context) -> list[str]:
    notes = [] if _variance_fixed(ctx.model) else [_variance_note(ctx)]
    return notes + [
        "Frailty family: CoxFrailty fits the gamma frailty only, so there "
        "is no other family to compare it with (the parametric "
        "Frailty(dist, family='lognormal') has one).",
        "Proportional hazards, functional form, influence and outliers "
        "are not checked: given the frailty they need each row's group, "
        "which a CoxFrailtyModel does not keep, and the Cox family's "
        "checks of CoxPH.fit(...) on these data would test the marginal "
        "model, which a frailty makes non-proportional (the hazard ratio "
        "fades as the frailer units fail). Where the frailty is not "
        "significant, CoxPH.fit(...).diagnose() checks the model without "
        "it.",
        "Overall fit is a plot only (d.plot('overall_fit')): the marginal "
        "Cox-Snell residuals of a group are dependent, so their curve "
        "has no simple test.",
    ]


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _not_significant(row: Any, ctx: Context) -> bool:
    p = row.get("_p") if row else None
    return p is not None and np.isfinite(p) and p >= ctx.alpha


def _drop_message(row: dict, ctx: Context) -> str:
    model = ctx.model
    name = null_fitter_name(model)
    if _is_cox(model):
        how = (
            "CoxPH.fit(x, Z, c, n, tie_method={!r}) (from a DataFrame, "
            "CoxPH.fit_from_df)".format(model.tie_method)
        )
    elif not np.size(model.beta):
        how = "{}.fit(x, c, n)".format(name)
    else:
        how = "{}.fit(x, Z, c, n) (from a DataFrame, {}.fit_from_df)".format(
            name, name
        )
    return (
        "The frailty is not significant (the test of theta = 0 has p = "
        "{p}): the groups "
        "share no detectable frailty. Drop it and fit the same model "
        "without it, {how}, the model this one is at theta = 0; its own "
        "diagnose() then checks it.".format(p=_fmt(row["_p"]), how=how)
    )


def _drops_to(target: Callable[[str], bool]) -> Callable[[Any, Context], bool]:
    """The ``when`` of a rule that advises dropping the frailty: the
    frailty is not significant and the model without it is the one whose
    fitter's name ``target`` accepts."""

    def when(row: Any, ctx: Context) -> bool:
        return target(null_fitter_name(ctx.model)) and _not_significant(
            row, ctx
        )

    return when


def _drop_rules() -> list[AdviceRule]:
    """One rule per model without frailty the advice can name, so that
    each rule's ``features`` are the ones its message names."""

    def rule(features: tuple[str, ...], target: Callable) -> AdviceRule:
        return AdviceRule(
            "frailty_variance",
            _drop_message,
            features,
            when=_drops_to(target),
            on=("warn",),
        )

    rules = [rule(("CoxPH.fit", "CoxPH.fit_from_df"), "CoxPH".__eq__)]
    for name in BASELINES:
        rules.append(
            rule(
                (name + "PH.fit", name + "PH.fit_from_df"),
                (name + "PH").__eq__,
            )
        )
        rules.append(rule((name + ".fit",), name.__eq__))
    rules.append(rule(("PH",), lambda fitter: fitter.startswith("PH(")))
    return rules


def _family_message(row: dict, ctx: Context) -> str:
    other = row["_other"]
    return (
        "The {other} frailty fits better: its AIC is {gap} lower on the "
        "same rows, with as many parameters. Refit with "
        "Frailty({dist}, family={other!r}).fit(x, Z, c, n, "
        "groups=...).".format(
            other=other, gap=_fmt(row["_delta"]), dist=_dist_name(ctx.model)
        )
    )


def _baseline_message(row: dict, ctx: Context) -> str:
    return (
        "The {best} baseline fits better: its AIC is {gap} lower on the "
        "same rows, with the same frailty. Refit with Frailty({best}, "
        "family={family!r}).fit(x, Z, c, n, groups=...).".format(
            best=row["_best"],
            gap=_fmt(row["_delta"]),
            family=ctx.model.family,
        )
    )


def _ph_name(row: dict) -> str:
    return row["covariates"][0]


def _stratify_message(row: dict, ctx: Context) -> str:
    name = _ph_name(row)
    var = _source(name)
    return (
        "The effect of {name} changes with time even given the frailty "
        "(it fails the proportional-hazards test). The frailty fitters "
        "have no strata or time-varying effects. Fit {fitter} to each "
        "level of {var} on its own ({fitter}.fit on its rows), which gives "
        "each level its own baseline; or stratify on {var} in a Cox model "
        "of the marginal hazard, CoxPH.fit(..., strata=<the {var} "
        "column>), leaving {var} out of Z, with standard errors clustered "
        "by group, summary(robust=True, cluster=<each row's group>) of "
        "that fit.".format(
            name=name, var=var, fitter=frailty_fitter_name(ctx.model)
        )
    )


def _time_varying_message(row: dict, ctx: Context) -> str:
    return (
        "The effect of {name} changes with time even given the frailty "
        "(it fails the proportional-hazards test). The frailty fitters "
        "have no time-varying effects; give it one in a Cox model of the "
        "marginal hazard: split each unit's follow-up at one or more "
        "times and fit CoxPH.fit_tvc on the pieces, with one column per "
        "period holding {name} in that period and 0 elsewhere, and "
        "standard errors clustered by group, summary(robust=True, "
        "cluster=<each piece's group>) of that fit.".format(name=_ph_name(row))
    )


def _global_only(row: dict, ctx: Context) -> bool:
    if not row.get("_global"):
        return False
    failed = [
        j for j, _ in ctx.checked if _step_test(ctx, [j])["p"] < ctx.alpha
    ]
    return not failed


def _global_message(row: dict, ctx: Context) -> str:
    return (
        "The joint (GLOBAL) proportional-hazards test fails given the "
        "frailty, though no single covariate does. The frailty fitters "
        "have no time-varying effects: a Cox model of the marginal hazard "
        "stratified on the likeliest covariates (CoxPH.fit(..., "
        "strata=...)) or with time-varying effects (CoxPH.fit_tvc), with "
        "standard errors clustered by group (summary(robust=True, "
        "cluster=...)), relaxes it."
    )


def _form_message(row: dict, ctx: Context) -> str:
    name = row["covariates"][0]
    return (
        "{name} does not act linearly on the log hazard given the frailty "
        "(its square is significant; see d.plot('functional_form')). "
        "Transform it, for example with a log in a formula, "
        "{fitter}.fit_from_df(..., group_col=..., "
        'formula="np.log({name}) + ..."), or bin it (with pandas.cut, '
        'say) and fit the bins as a categorical covariate, formula="C('
        '{name}_band) + ...".'.format(
            name=name, fitter=frailty_fitter_name(ctx.model)
        )
    )


def _is_parametric(model: Any) -> bool:
    from surpyval.univariate.regression.frailty import FrailtyModel

    return isinstance(model, FrailtyModel)


_VARIANCE_CHECK = Check(
    "frailty_variance",
    "The groups share frailties: the frailty variance theta is not 0.",
    rows=_variance_rows,
    panels=_variance_panels,
)
_OVERALL_CHECK = Check(
    "overall_fit",
    "The marginal Cox-Snell residuals behave as a censored sample of the "
    "unit exponential.",
    panels=_overall_panels,
)

#: The parametric shared-frailty family (``FrailtyModel``).
PARAMETRIC_FAMILY = register(
    Family(
        name=_PARAMETRIC,
        issue=872,
        applies=_is_parametric,
        validate=_validate,
        covariates=_names,
        checks=(
            _VARIANCE_CHECK,
            Check(
                "frailty_family",
                "The frailty family fits: the other family's AIC is not "
                "markedly lower.",
                rows=_family_rows,
            ),
            Check(
                "proportional_hazards",
                "Given the frailty, each covariate multiplies the hazard by "
                "a factor that does not change with time.",
                rows=_ph_rows,
            ),
            Check(
                "functional_form",
                "Given the frailty, each covariate acts linearly on the log "
                "hazard.",
                rows=_form_rows,
                panels=_form_panels,
            ),
            Check(
                "baseline_distribution",
                "The baseline distribution fits: no other built-in "
                "baseline's AIC is markedly lower.",
                rows=_baseline_rows,
                panels=_baseline_panels,
            ),
            _OVERALL_CHECK,
        ),
        advice=(
            *_drop_rules(),
            AdviceRule(
                "frailty_family",
                _family_message,
                ("Frailty(family=)",),
                when=lambda row, ctx: "_other" in row,
                on=("warn",),
            ),
            AdviceRule(
                "proportional_hazards",
                _stratify_message,
                (
                    "FrailtyFitter.fit",
                    "CoxPH.fit(strata=)",
                    "summary(robust=)",
                ),
                when=lambda row, ctx: row.get("_kind")
                in ("binary", "categorical"),
            ),
            AdviceRule(
                "proportional_hazards",
                _time_varying_message,
                ("CoxPH.fit_tvc", "summary(robust=)"),
                when=lambda row, ctx: row.get("_kind") == "continuous",
            ),
            AdviceRule(
                "proportional_hazards",
                _global_message,
                ("CoxPH.fit(strata=)", "CoxPH.fit_tvc", "summary(robust=)"),
                when=_global_only,
            ),
            AdviceRule(
                "functional_form",
                _form_message,
                ("FrailtyFitter.fit_from_df(formula=)",),
            ),
            AdviceRule(
                "baseline_distribution",
                _baseline_message,
                ("Frailty(family=)",),
                when=lambda row, ctx: "_best" in row,
                on=("warn",),
            ),
        ),
        notes=_parametric_notes,
    )
)

#: The Cox shared-frailty family (``CoxFrailtyModel``).
COX_FAMILY = register(
    Family(
        name=_COX,
        issue=872,
        applies=_is_cox,
        validate=_validate,
        covariates=_names,
        checks=(_VARIANCE_CHECK, _OVERALL_CHECK),
        advice=tuple(
            r for r in _drop_rules() if r.features[0].startswith("CoxPH")
        ),
        notes=_cox_notes,
    )
)
