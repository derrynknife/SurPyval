"""The derivatives a fit takes agree with finite differences (#562).

A fit that differentiates its likelihood -- for the search, for the check
that its answer is a maximum, or for the covariance -- is only as good as
those derivatives. For every registered model that takes them (see
``DIFFERENTIATED`` in ``registry_families.py``), at the fitted parameters
and in the space the fitter differentiates in, the gradient and Hessian
of the negative log-likelihood it differentiates agree with Richardson
finite differences of the same function (``_helpers.richardson_*``):

- univariate parametric MLE (with offset, zero inflation and LFP; the
  causes of a parametric competing-risks model): the fitter's search
  objective in its unbounded search space, by autograd; a closed-form fit
  in its parameters, whose covariance is autograd's Hessian there. And
  the delta-method gradients behind ``cb``, of ``sf`` and ``ff`` in the
  parameters, by autograd;
- parametric regression (PH, AFT, PO, AH, accelerated life): the search
  objective in the transformed search space (rebuilt as the fit builds
  it, and checked to give the fit's likelihood), by autograd;
- frailty and Fine-Gray: the objective, point and derivatives the fitter
  itself passes through ``search_derivatives``, captured from a refit;
- Cox: the analytic score and information of the partial likelihood that
  its Newton-Raphson fit takes;
- the HPP and proportional-intensity HPP: their log-rate search space, by
  autograd; the mixture model: its unconstrained polishing space;
- copulas: the h-functions ``du``, ``dv`` and the density ``pdf`` (closed
  forms, or autograd's derivatives of ``cdf``), which the censored
  likelihood is made of, against differences of ``cdf``;
- degradation paths: the analytic Jacobian of the path in its parameters
  (the per-unit fit and covariance), against differences of the path.

Errors are measured in units the fit cares about: each parameter is
scaled by ``s = 1/sqrt(H_ii)`` (about its standard error), so a gradient
error is ``|dg_i| s_i`` and a Hessian error ``|dH_ij| s_i s_j``, the
latter relative to the unit diagonal of the scaled Hessian. The finite
differences step ``STEP`` of a standard error, so their error is about
``STEP**4`` from truncation and ``eps |f| / STEP**2`` from rounding: on
these fixtures the largest disagreement of a correct derivative is
4e-9 (Hessian) and 2e-10 (gradient), against a tolerance ``TOL`` of
1e-6; the likelihoods through the incomplete gamma and beta functions'
numerical shape derivatives get 1e-5 (see ``NUMERICAL_SHAPE``). A wrong
derivative shows well above that: the accelerated-life ``where`` bug of
#555 was 4e-5, a NaN is infinite. The delta-method gradients are scaled
the same way (the change of the probability per standard error: at most
5e-9 here), and the copulas' derivatives are compared absolutely (at
most 3e-11).

The property reuses the fitted models (it is not a refit property),
except for the frailty and Fine-Gray cases, refitted to capture the
derivatives their fitters take.
"""

from unittest import mock

import numpy as np
import pytest
from autograd import hessian, jacobian

from surpyval.tests._helpers import (
    richardson_gradient,
    richardson_hessian,
    richardson_jacobian,
)
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    DIFFERENTIATED,
    NOT_DIFFERENTIATED,
    cases_for,
    fitted,
    refit,
)

# The finite-difference step, in standard errors of each parameter.
STEP = 1e-2
# The largest scaled error of a correct derivative on the fixtures is
# below 1e-8 (see the module docstring); a wrong one is 1e-5 or more.
TOL = 1e-6
# Likelihoods through a shape derivative of the regularised incomplete
# gamma or beta function, which ``utils/autograd_gamma_compat.py`` takes
# by central differences of step eps**(1/3) |a|: their second
# derivatives are good to about eps**(1/3) = 6e-6 of the function, and
# these Hessians agree to 2e-6 (Gamma, Beta4).
NUMERICAL_SHAPE = ("Gamma", "NegativeBinomial", "Beta4")
TOL_NUMERICAL_SHAPE = 1e-5


def _tolerance(case):
    if case.name.startswith(NUMERICAL_SHAPE):
        return TOL_NUMERICAL_SHAPE
    return TOL


def _scales(H, at):
    """``1/sqrt(H_ii)`` where the curvature is positive, else 1; at most
    ``max(|at_i|, 1)``, so that a parameter the likelihood hardly depends
    on (an offset at the first failure, a zero-inflation near 0) is not
    stepped across its whole range."""
    d = np.diag(np.asarray(H, dtype=float))
    ok = np.isfinite(d) & (d > 0)
    s = np.where(ok, 1.0 / np.sqrt(np.where(ok, d, 1.0)), 1.0)
    return np.minimum(s, np.maximum(np.abs(at), 1.0))


def _objective_errors(f, at, g, H, tol=TOL):
    """The scaled errors of the gradient ``g`` and Hessian ``H`` of ``f``
    at ``at`` against Richardson differences (infinite where ``g`` or
    ``H`` is not finite).

    The differences step ``STEP`` standard errors, and a tenth of that
    where they do not agree there: next to a singularity of the
    likelihood their truncation error grows (Beta4's lower bound sits
    three steps below the first observation, where the Hessian is
    3e-2 off at the larger step and 2e-6 at the smaller). A wrong
    derivative disagrees at every step."""
    at = np.asarray(at, dtype=float)
    g = np.asarray(g, dtype=float).ravel()
    H = np.atleast_2d(np.asarray(H, dtype=float))
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        return np.inf, np.inf
    s = _scales(H, at)
    best = (np.inf, np.inf)
    for step in (STEP, STEP / 10):
        with np.errstate(all="ignore"):
            g_fd = richardson_gradient(f, at, step * s)
            H_fd = richardson_hessian(f, at, step * s)
        err = (
            float(np.max(np.abs(g - g_fd) * s)),
            float(np.max(np.abs(H - H_fd) * np.outer(s, s))),
        )
        if max(err) < max(best):
            best = err
        if max(best) <= tol:
            break
    return best


def _autograd(f):
    """The fitters' derivatives of ``f``: autograd's gradient and
    Hessian."""
    jac, hess = jacobian(f), hessian(f)

    def derivatives(at):
        with np.errstate(all="ignore"):
            return jac(at), hess(at)

    return derivatives


# ---------------------------------------------------------------------------
# What each fitter differentiates: (label, f, point, derivatives(point) ->
# (gradient, Hessian), the likelihood the fit reports there or None).
# ---------------------------------------------------------------------------
def _parametric(model, label=""):
    from surpyval.univariate.parametric.fitters.mle import (
        _negative_log_likelihood,
    )

    res = getattr(model, "res", None)
    u = None if res is None else np.asarray(res.x, dtype=float)
    # The search maps a parameter with one bound by a log below its start
    # and linearly above (``adj_relu``): a search coordinate of exactly 0,
    # the start, where the search did not move it (the LogNormal's sigma,
    # whose start is its answer), is the switch, where the map has no
    # second derivative and differences across it are wrong by O(step).
    # There, as for a closed-form fit (whose covariance is autograd's
    # Hessian in the parameters, ``fitters/closed_form.py``), the
    # likelihood is checked in the parameters.
    if u is not None and model.fitting_info and not np.any(u == 0.0):
        fun = _negative_log_likelihood(model)
        args = (model.offset, model.lfp, model.zi, True)

        def f(u):
            return fun(u, *args)

        return [(label + " (search space)", f, u, _autograd(f), model._neg_ll)]

    k = len(model.params)

    def f_natural(phi):
        # (*params, p?, f0?) with the offset held, as the covariance is
        p = phi[k] if model.lfp else model.p
        f0 = phi[-1] if model.zi else model.f0
        return model.dist._neg_ll_func(
            model.surv_data, *phi[:k], model.gamma, f0, p
        )

    at = np.asarray(model._cb_context().phi_hat, dtype=float)
    return [
        (
            label + " (parameters)",
            f_natural,
            at,
            _autograd(f_natural),
            model._neg_ll,
        )
    ]


def _regression(model, label=""):
    from surpyval.univariate.parametric.fitters import bounds_convert
    from surpyval.univariate.regression._fit_skeleton import centred_copy

    # The point the search ran at: the centred fit's parameters where the
    # fit moved its baseline to Z = 0 afterwards (prepare_regression_fit).
    if model._fit_centring is not None:
        p_hat, center = model._fit_centring[:2]
    else:
        p_hat, center = model._eval_params(), model.center
    p_hat = np.asarray(p_hat, dtype=float)
    data = model.data
    if center is not None and np.any(center):
        data = centred_copy(data, center)
    names = list(model.parameter_names)
    held = model._held()
    fixed = {n: p_hat[i] for i, n in enumerate(names) if n in held}
    transform, inv_trans, const, _, not_fixed = bounds_convert(
        None,
        model._parameter_bounds(),
        fixed,
        {n: i for i, n in enumerate(names)},
    )
    fitter = model.model

    def f(t):
        return fitter.neg_ll(data, *inv_trans(const(t)))

    with np.errstate(all="ignore"):
        at = np.asarray(transform(p_hat), dtype=float)[not_fixed]
    return [(label + " (search space)", f, at, _autograd(f), model._neg_ll)]


def _spied(case, label=""):
    """The objective, point and derivatives the fitter passes through
    ``search_derivatives`` first (at its answer), from a refit."""
    # Fine-Gray and the frailty fits reach it through _fit_skeleton
    # (judge_search), so one spy there sees every fit.
    from surpyval.univariate.regression import _fit_skeleton

    original = _fit_skeleton.search_derivatives
    calls = []

    def spy(neg_ll, x):
        out = original(neg_ll, x)
        calls.append((neg_ll, np.array(x, dtype=float), out))
        return out

    with mock.patch.object(_fit_skeleton, "search_derivatives", spy):
        refit(case, case.data())
    assert calls, "the fit took no derivatives"
    neg_ll, at, out = calls[0]
    assert out is not None, "autograd could not differentiate the fit"
    H, g = out
    return [
        (
            label + " (as the fit took them)",
            neg_ll,
            at,
            lambda _: (g, H),
            None,
        )
    ]


def _cox(model, label=""):
    def derivatives(at):
        score, H = model.jac(at)
        return np.atleast_1d(score), np.atleast_2d(H)

    at = np.asarray(model.params, dtype=float)
    return [(label + " (analytic)", model.neg_ll, at, derivatives, None)]


def _competing_parametric(model, label=""):
    out = []
    for cause, m in model.models.items():
        out += _parametric(m, f"{label}cause {cause}")
    return out


def _hpp(model, label=""):
    # The fit searches log(rate), then the coefficients, on the fitter's
    # own likelihood (``_neg_ll`` is the same in the rate itself, but not
    # differentiable).
    fitter = getattr(model, "_fitter", model.dist)
    f = fitter.create_negll_func(model.data)
    at = np.asarray(model.res.x, dtype=float)
    return [(label + " (search space)", f, at, _autograd(f), None)]


def _mixture(model, label=""):
    def f(theta):
        return model.neg_ll_of(*model._unpack(theta))

    at = np.asarray(model._pack(model.w, model.params), dtype=float)
    return [(label + " (search space)", f, at, _autograd(f), None)]


OBJECTIVES = {
    "surpyval.Parametric": lambda case: _parametric(fitted(case)),
    "surpyval.ParametricRegressionModel": lambda case: _regression(
        fitted(case)
    ),
    "surpyval.FrailtyModel": _spied,
    "surpyval.univariate.competing_risks.regression.fine_gray"
    ".FineGrayModel": _spied,
    "surpyval.SemiParametricRegressionModel": lambda case: _cox(fitted(case)),
    "surpyval.univariate.competing_risks.ParametricCompetingRisks": (
        lambda case: _competing_parametric(fitted(case))
    ),
    "surpyval.recurrent.parametric.parametric_recurrence"
    ".ParametricRecurrenceModel": lambda case: _hpp(fitted(case)),
    "surpyval.recurrent.regression.proportional_intensity"
    ".ProportionalIntensityModel": lambda case: _hpp(fitted(case)),
    "surpyval.MixtureModel": lambda case: _mixture(fitted(case)),
}


@pytest.mark.parametrize(
    "case",
    cases_for("derivatives", where=lambda c: c.model_class in OBJECTIVES),
)
def test_likelihood_derivatives_agree_with_finite_differences(case):
    tol = _tolerance(case)
    failures = []
    for label, f, at, derivatives, reported in OBJECTIVES[case.model_class](
        case
    ):
        if reported is not None and np.isfinite(reported):
            # The rebuilt objective is the fit's: it gives the fit's
            # likelihood at the fit's point.
            with np.errstate(all="ignore"):
                value = float(f(at))
            np.testing.assert_allclose(value, reported, rtol=1e-9, atol=0)
        g, H = derivatives(at)
        err_g, err_H = _objective_errors(f, at, g, H, tol)
        if not (err_g <= tol and err_H <= tol):
            failures.append(
                f"{label}: gradient error {err_g:.2g}, Hessian error "
                f"{err_H:.2g} (scaled; tolerance {tol:g})"
            )
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------
# The delta-method gradients behind a parametric model's cb.
# ---------------------------------------------------------------------------
def _has_covariance(case):
    if case.model_class != "surpyval.Parametric":
        return False
    return getattr(fitted(case), "cov_matrix", None) is not None


@pytest.mark.parametrize(
    "case", cases_for("derivatives", needs=("sf",), where=_has_covariance)
)
def test_delta_method_gradients_agree_with_finite_differences(case):
    model = fitted(case)
    ctx = model._cb_context()
    se = np.sqrt(np.clip(np.diag(ctx.cov), 0, None))
    free = se > 0
    x = np.asarray(case.x, dtype=float)
    for name in ("sf", "ff"):
        full = getattr(model, f"_cb_full_{name}")

        def func(phi):
            return full(x, phi, ctx)

        with np.errstate(all="ignore"):
            J = np.atleast_2d(jacobian(func)(ctx.phi_hat))
            J_fd = richardson_jacobian(
                func, ctx.phi_hat, STEP * np.where(free, se, 1.0)
            )
        # The change of the probability per standard error of each
        # parameter (the delta method's terms).
        err = np.abs(J - J_fd)[:, free] * se[free]
        assert np.max(err, initial=0.0) <= TOL, (name, np.max(err))


# ---------------------------------------------------------------------------
# Copulas: the h-functions and the density against differences of cdf.
# ---------------------------------------------------------------------------
_GRID = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
_H = 1e-4


def _is_copula(case):
    return case.model_class == "surpyval.multivariate.CopulaModel"


@pytest.mark.parametrize("case", cases_for("derivatives", where=_is_copula))
def test_copula_derivatives_agree_with_finite_differences(case):
    model = fitted(case)
    copula, params = model.copula, tuple(model.params)
    u, v = (a.ravel() for a in np.meshgrid(_GRID, _GRID))

    def d(fun, wrt):
        # Richardson central difference of ``fun`` in u (0) or v (1).
        def central(h):
            if wrt == 0:
                return (fun(u + h, v) - fun(u - h, v)) / (2 * h)
            return (fun(u, v + h) - fun(u, v - h)) / (2 * h)

        return (4 * central(_H / 2) - central(_H)) / 3

    def cdf(a, b):
        return np.asarray(copula.cdf(a, b, *params), dtype=float)

    def du(a, b):
        return np.asarray(copula.du(a, b, *params), dtype=float)

    with np.errstate(all="ignore"):
        pairs = {
            "du": (du(u, v), d(cdf, 0)),
            "dv": (copula.dv(u, v, *params), d(cdf, 1)),
            "pdf": (copula.pdf(u, v, *params), d(du, 1)),
        }
    for name, (value, fd) in pairs.items():
        value = np.asarray(value, dtype=float)
        np.testing.assert_allclose(value, fd, rtol=TOL, atol=TOL, err_msg=name)


# ---------------------------------------------------------------------------
# Degradation paths: the analytic Jacobian against differences of the path.
# ---------------------------------------------------------------------------
def _is_path(case):
    return case.model_class == "surpyval.degradation.DegradationModel"


@pytest.mark.parametrize("case", cases_for("derivatives", where=_is_path))
def test_path_jacobian_agrees_with_finite_differences(case):
    model = fitted(case)
    path = model.path_model
    x = np.unique(np.asarray(model.x, dtype=float))
    for theta in np.atleast_2d(model.path_params):
        theta = np.asarray(theta, dtype=float)
        J = np.asarray(path.jacobian(x, *theta), dtype=float)
        with np.errstate(all="ignore"):
            J_fd = richardson_jacobian(
                lambda p: path.path(x, *p),
                theta,
                1e-4 * np.maximum(np.abs(theta), 1e-2),
            )
        scale = np.max(np.abs(J_fd), axis=0, initial=0.0)
        np.testing.assert_array_less(
            np.max(np.abs(J - J_fd), axis=0), TOL * np.maximum(scale, 1.0)
        )


def test_every_differentiated_class_has_a_check():
    checked = set(OBJECTIVES) | {
        "surpyval.multivariate.CopulaModel",
        "surpyval.degradation.DegradationModel",
    }
    assert checked == set(DIFFERENTIATED)
    # and the exceptions are cases of those classes
    for name in NOT_DIFFERENTIATED:
        assert CASE_BY_NAME[name].model_class in DIFFERENTIATED, name
