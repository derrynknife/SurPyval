"""Every maximum-likelihood fit reports and verifies its maximum
(principles 12 and 13).

A fit that maximises a likelihood -- a parametric distribution, a
mixture, a parametric or semi-parametric regression, a frailty model, a
competing-risks model, a recurrence process, a copula, a degradation
process or destructive degradation model -- records what it
reached as its model's ``maximum``, one of
``surpyval.utils.no_maximum.MAXIMUM_STATES``. For every registered case
whose fit is a likelihood maximisation, and for its time-varying
covariate fit where its fitter has one, the property checks that

- the fitted model's ``maximum`` is what a fit can reach: ``"verified"``,
  ``"unverified"`` or ``"no finite maximum"`` (``"unknown"`` is only for a
  model restored from a dictionary saved without it, ``"not applicable"``
  for parameters that do not come from a likelihood);
- the fit warned that it "did not reach a verified maximum" exactly when
  ``maximum`` is ``"unverified"`` (``warn_unverified``; a few fitters say
  so in their own words, :data:`OWN_WORDS`), and "No finite maximum"
  exactly when it is ``"no finite maximum"`` (``warn_no_maximum``);
- where ``maximum`` is ``"verified"``, an independent check at the
  reported parameters passes: the gradient of the model's own negative
  log-likelihood, scaled per observation and by each parameter's size,
  is ~0 and its Hessian positive definite (``is_local_minimum``). The
  check is made in the unconstrained space the fitter searches (see each
  ``_search_*`` function): the univariate fit's own search vector and
  transforms, the regression fits' (``bounds_convert``), and for the
  others the log of a positive parameter, the logit of a probability and
  a coefficient as it is. A parameter on a boundary of its space where
  that is the maximum -- a frailty variance of 0, where the model is the
  one without frailty -- is held out of the check, and the likelihood
  must not rise as it moves off the boundary instead. A covariate
  coefficient is measured in its own covariate's units
  (``coefficient_floor``): its least unit is ``1 / range(Z_j)``, so that
  the check means the same whatever units a covariate is recorded in.

The fixture's fit is checked, and so is the starved fit of the
convergence property (``Case.starve``), which reaches the other states:
it must say what it reached in the same way. So is the fit with every
covariate in millionths of its units (#577), where a coefficient's
gradient is a millionth of what it was at its start of 0: below an
absolute tolerance, which a search and a check in fixed units met at
once. Cases whose estimate is
not a likelihood maximisation are excluded with the reason
(``registry_families.NOT_A_LIKELIHOOD_FIT`` and the case's
``exclude``).
"""

import warnings
from typing import Any, Callable, NamedTuple

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import hessian, jacobian

import surpyval as sp
from surpyval.tests.conformance import leaks
from surpyval.tests.conformance.registry import CASES, tvc_path
from surpyval.univariate.parametric.fitters import (
    OPTIMUM_GTOL,
    bounds_convert,
    is_local_minimum,
    search_floor,
)
from surpyval.utils.covariates import coefficient_floor
from surpyval.utils.no_maximum import MAXIMUM_STATES

UNVERIFIED = "did not reach a verified maximum"
NO_MAXIMUM = "No finite maximum"
FITTED = ("verified", "unverified", "no finite maximum")

# Fitters that say a search did not reach a verified maximum in their own
# words, each saying why (the state is "unverified" all the same).
OWN_WORDS = {
    # The univariate MLE that fell back to its starting point
    "MLE Failed": "the univariate search returned its start",
    # The parametric additive hazards fit held by the positivity barrier
    "ended on the positivity boundary": "the additive hazards barrier",
    # An accelerated life fit with fewer stress levels than parameters
    "is not identifiable": "a ridge of accelerated life parameters",
}

# Cases whose verified maximum has no gradient to check, and why.
NOT_DIFFERENTIABLE = {
    "Uniform": (
        "the maximum is on the sample's extremes, the support's ends, "
        "where the likelihood is not differentiable"
    ),
    "Binomial": "the number of trials is an integer parameter",
    "FixedEventProbability": (
        "a closed-form share of events, of a distribution with no hazard "
        "for the general likelihood to read"
    ),
    "ExactEventTime": (
        "the likelihood is flat between the censoring times on either "
        "side of the estimate (a point mass)"
    ),
}


def _said(fit: Callable[[], Any]) -> tuple[Any, list[str]]:
    """``fit()`` and the deliberate warnings it gave."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with leaks.deliberate() as found:
            model = fit()
    return model, [message for _, message in found]


def _check(case, model, said, data):
    """The property for ``model``, fitted with the warnings ``said``;
    ``data`` is the case's fixture, or ``None`` for a starved fit, whose
    data only the starve knows (a family whose check needs the fixture --
    the model keeps neither its data nor its likelihood -- is then not
    checked independently)."""
    state = getattr(model, "maximum", None)
    assert state in MAXIMUM_STATES, f"maximum is {state!r}"
    assert state in FITTED, (
        f"a likelihood fit reports maximum={state!r}: it should say "
        f"whether it reached a verified maximum, one of {FITTED}"
    )
    unverified = any(
        UNVERIFIED in m or any(w in m for w in OWN_WORDS) for m in said
    )
    no_maximum = any(m.startswith(NO_MAXIMUM) for m in said)
    assert unverified == (state == "unverified"), (
        f"maximum={state!r} but the fit "
        f"{'warned' if unverified else 'did not warn'} that it did not "
        f"reach a verified maximum: {said}"
    )
    assert no_maximum == (state == "no finite maximum"), (
        f"maximum={state!r} but the fit "
        f"{'warned' if no_maximum else 'did not warn'} 'No finite "
        f"maximum': {said}"
    )
    if state != "verified" or case.name in NOT_DIFFERENTIABLE:
        return
    searches = SEARCHES[type(model).__name__](model, data)
    for search in searches or ():
        assert search.verified(), (
            "maximum='verified', but the reported parameters are not a "
            "verified maximum of the model's likelihood: " + search.why()
        )


def _params(key, where=None, slow=False):
    """The cases of the property, marked with their known failures of
    ``key`` ("maximum", "maximum[starved]" or "maximum[tvc]"), and
    ``slow`` where the case's refits are, or for every case."""
    out = []
    for case in CASES:
        if not case.applies("maximum") or key in case.exclude:
            continue
        if where is not None and not where(case):
            continue
        marks = []
        if slow or case.is_slow("maximum"):
            marks.append(pytest.mark.slow)
        if key in case.xfail:
            marks.append(
                pytest.mark.xfail(strict=True, reason=case.xfail[key])
            )
        out.append(pytest.param(case, id=case.name, marks=marks))
    return out


@pytest.mark.parametrize("case", _params("maximum"))
def test_a_fit_says_what_it_reached(case):
    data = case.data()
    model, said = _said(lambda: case.fit(data))
    _check(case, model, said, data)


# The starved fits are the convergence property's, which the pull-request
# job runs; their states are checked in the full suite (two thirds of this
# module's time).
@pytest.mark.parametrize(
    "case",
    _params("maximum[starved]", lambda c: c.starve is not None, slow=True),
)
def test_a_starved_fit_says_what_it_reached(case):
    try:
        model, said = _said(lambda: case.starve(case.data()))
    except ValueError:
        return  # refused, saying why (the convergence property)
    _check(case, model, said, None)


def _tvc(case):
    return case.paths.get("fit_tvc") or tvc_path(case)


@pytest.mark.parametrize(
    "case", _params("maximum[tvc]", lambda c: _tvc(c) is not None)
)
def test_a_tvc_fit_says_what_it_reached(case):
    data = case.data()
    model, said = _said(lambda: _tvc(case)(data))
    _check(case, model, said, data)


#: The factor the covariates are multiplied by in the small-scale fit.
SMALL_SCALE = 1e-6


def _small_scale(case):
    data = case.data()
    Z = np.asarray(data[case.covariates], dtype=float)
    return {**data, case.covariates: Z * SMALL_SCALE}


# A covariate in millionths of its units: an Arrhenius 1/T in kelvin spans
# 3e-4, and a WeibullPH fit to it stopped at its start and reported a
# verified maximum (#577). The time-varying fit takes the same search.
@pytest.mark.parametrize(
    "case",
    _params("maximum[small-scale]", lambda c: c.covariates is not None),
)
def test_577_a_small_scale_covariate_fit_says_what_it_reached(case):
    data = _small_scale(case)
    model, said = _said(lambda: case.fit(data))
    _check(case, model, said, data)


@pytest.mark.parametrize(
    "case",
    _params(
        "maximum[small-scale tvc]",
        lambda c: c.covariates is not None and _tvc(c) is not None,
        slow=True,
    ),
)
def test_577_a_small_scale_covariate_tvc_fit_says_what_it_reached(case):
    data = _small_scale(case)
    model, said = _said(lambda: _tvc(case)(data))
    _check(case, model, said, data)


# ---------------------------------------------------------------------------
# The independent check: each family's negative log-likelihood, in the
# space its fitter searches, at the reported parameters
# ---------------------------------------------------------------------------
class Search(NamedTuple):
    """A negative log-likelihood ``fun`` in a search space, the point
    ``x`` the model reports there, and the scales of ``is_local_minimum``
    (``n_obs`` observations, ``floor`` per component). ``held`` are
    components on a boundary of their space, where the likelihood falls
    as they move off it (checked by the family's function), and ``jac``
    and ``hess`` the derivatives where autograd does not take them."""

    fun: Callable
    x: Any
    n_obs: float
    floor: Any = 1.0
    held: tuple = ()
    jac: "Callable | None" = None
    hess: "Callable | None" = None
    name: str = ""

    def in_covariate_units(self, coefs, Z):
        """This search with each coefficient's least unit its covariate's
        (``coefficient_floor``; ``coefs`` its ``(position, column)``
        pairs), as the fits search and judge it (#577)."""
        floor = np.broadcast_to(
            np.asarray(self.floor, dtype=float), np.shape(self.x)
        )
        units = coefficient_floor(np.size(self.x), coefs, Z)
        return self._replace(floor=np.maximum(floor, units))

    def _parts(self):
        x = np.asarray(self.x, dtype=float)
        keep = [i for i in range(x.size) if i not in self.held]
        jac, hess = self.jac, self.hess
        if jac is None or hess is None:
            jac, hess = _derivatives(self.fun, x)
        floor = np.broadcast_to(np.asarray(self.floor, float), x.shape)
        return x, keep, jac, hess, floor

    def verified(self) -> bool:
        x, keep, jac, hess, floor = self._parts()
        with np.errstate(all="ignore"):
            if not self.held:
                return is_local_minimum(
                    self.fun, jac, hess, x, floor=floor, obj_scale=self.n_obs
                )
            # The derivatives in the other components, at ``x``
            g = np.asarray(jac(x), float)[keep]
            H = np.atleast_2d(np.asarray(hess(x), float))[np.ix_(keep, keep)]
            return is_local_minimum(
                self.fun,
                lambda _: g,
                lambda _: H,
                x[keep],
                floor=floor[keep],
                obj_scale=self.n_obs,
            )

    def why(self) -> str:
        x, keep, jac, hess, floor = self._parts()
        with np.errstate(all="ignore"):
            g = np.asarray(jac(x), float)[keep]
            H = np.atleast_2d(np.asarray(hess(x), float))[np.ix_(keep, keep)]
        scale = np.maximum(np.abs(x[keep]), floor[keep])
        eig = np.linalg.eigvalsh(0.5 * (H + H.T) * np.outer(scale, scale))
        return "{}at {}: scaled gradient {}, scaled Hessian {}".format(
            f"{self.name} " if self.name else "",
            np.round(x, 6).tolist(),
            np.round(scale * g / self.n_obs, 8).tolist(),
            "eigenvalues {}".format(np.round(eig / self.n_obs, 8).tolist()),
        )


def _derivatives(fun, x):
    """``(jac, hess)`` of ``fun``: autograd's, or central differences
    (``numerical_gradient``, ``numerical_hessian``, in steps relative to
    each component) for a likelihood autograd cannot differentiate, or
    whose autograd derivatives are not finite at ``x``."""
    from surpyval.utils.linalg import numerical_gradient, numerical_hessian

    jac, hess = jacobian(fun), hessian(fun)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = np.all(np.isfinite(np.asarray(jac(x), float))) and np.all(
                np.isfinite(np.asarray(hess(x), float))
            )
    except Exception:
        ok = False
    if ok:
        return jac, hess
    steps = 1e-5 * np.maximum(np.abs(np.asarray(x, float)), 1.0)
    return (
        lambda v: numerical_gradient(fun, v, 1e-2 * steps),
        lambda v: numerical_hessian(fun, v, steps),
    )


def _canonical(neg_ll, params, bounds, n_obs, held=(), name=""):
    """``neg_ll`` of the natural parameters as a :class:`Search` in the
    unconstrained space the fitters without one of their own search: the
    log of the distance from a one-sided bound, the logit between two, a
    free parameter as it is."""
    params = np.asarray(params, dtype=float)

    def to_natural(u):
        out = []
        for k, (lo, hi) in enumerate(bounds):
            if lo is not None and hi is not None:
                out.append(lo + (hi - lo) / (1.0 + anp.exp(-u[k])))
            elif lo is not None:
                out.append(lo + anp.exp(u[k]))
            elif hi is not None:
                out.append(hi - anp.exp(u[k]))
            else:
                out.append(u[k])
        return out  # (a list: a parameter is a scalar, as a fit gives it)

    u = []
    for value, (lo, hi) in zip(params, bounds):
        with np.errstate(all="ignore"):
            if lo is not None and hi is not None:
                f = (value - lo) / (hi - lo)
                u.append(np.log(f) - np.log1p(-f))
            elif lo is not None:
                u.append(np.log(value - lo))
            elif hi is not None:
                u.append(np.log(hi - value))
            else:
                u.append(value)
    return Search(
        lambda v: neg_ll(to_natural(v)),
        np.array(u),
        n_obs,
        held=held,
        name=name,
    )


def _search_parametric(model, data, name=""):
    """The univariate fit's own search: ``res.x`` mapped by its
    ``fitting_info``; a closed form in the canonical space."""
    dist = model.dist
    offset, lfp, zi = model.offset, model.lfp, model.zi

    def split(params):
        gamma, f0, p = 0.0, 0.0, 1.0
        if offset:
            gamma, *params = params
        if zi:
            *params, f0 = params
        if lfp:
            *params, p = params
        return params, gamma, f0, p

    surv_data = getattr(model, "surv_data", None)
    if surv_data is None and data is None:
        return None
    if surv_data is None:
        surv_data = sp.SurpyvalData(
            data["x"], data.get("c"), data.get("n"), data.get("t")
        )
    info = getattr(model, "fitting_info", None) or {}
    if "inv_trans" in info and getattr(model, "res", None) is not None:
        inv, const = info["inv_trans"], info["const"]
        x = np.asarray(model.res.x, dtype=float)
        reported = [model.gamma] if offset else []
        reported += list(model.params)
        reported += [model.p] if lfp else []
        reported += [model.f0] if zi else []
        np.testing.assert_allclose(
            np.asarray(inv(const(x)), float),
            reported,
            rtol=1e-12,
            err_msg="the search vector is not at the reported parameters",
        )

        def fun(u):
            params, gamma, f0, p = split(inv(const(u)))
            return dist._neg_ll_func(surv_data, *params, gamma, f0, p)

        def natural(values):
            params, gamma, f0, p = split(values)
            return dist._neg_ll_func(surv_data, *params, gamma, f0, p)

        n_obs = float(np.sum(surv_data.n))
        free = [i for i in range(len(reported)) if i not in info["fixed_idx"]]
        held = _on_range_ends(
            natural, np.array(reported, float), model.bounds, free, n_obs
        )
        if held and not info["fixed_idx"]:
            # (a fit with fixed parameters is refitted only with them)
            _no_higher_off_the_ends(model, surv_data, reported, held, free)
        return [
            Search(
                fun,
                x,
                n_obs,
                search_floor(model),
                held=tuple(held),
                name=name,
            )
        ]

    # A closed form: no search vector of its own
    bounds = list(model.bounds)
    params = list(model.params)

    def neg_ll(v):
        return dist._neg_ll_func(surv_data, *v, 0.0, 0.0, 1.0)

    return [_canonical(neg_ll, params, bounds, float(np.sum(surv_data.n)))]


def _on_range_ends(neg_ll, values, bounds, free, n_obs):
    """The positions in the search vector (``free``: the natural parameter
    of each) of the parameters on an end of a range bounded at both (a
    limited-failure ``p`` of 1, a zero-inflation ``f0`` of 0), where the
    likelihood (``neg_ll`` of the natural ``values``) stops depending on
    them: the same, to rounding, a millionth of the way closer. Each must
    be a maximum there, the likelihood not rising off the end by more
    than the verification's tolerance (``OPTIMUM_GTOL`` per observation)
    a millionth of the range into it. Searched as a scaled arctanh, such
    a parameter reaches its end in floating point, where its gradient and
    curvature are zero or rounding, so the search vector's check cannot
    judge it (#579)."""
    f = float(neg_ll(values))
    level = 1e-12 * max(abs(f), 1.0)
    held = []
    for k, i in enumerate(free):
        lo, hi = bounds[i]
        if lo is None or hi is None:
            continue
        width = float(hi) - float(lo)
        for bound, inward in ((lo, 1.0), (hi, -1.0)):
            toward, away = values.copy(), values.copy()
            toward[i] = bound + (values[i] - bound) * 1e-6
            away[i] = bound + inward * 1e-6 * width
            with np.errstate(all="ignore"):
                if not abs(float(neg_ll(toward)) - f) <= level:
                    continue
                rise = (f - float(neg_ll(away))) / (1e-6 * width) / n_obs
            assert rise < OPTIMUM_GTOL, (
                f"parameter {i} is on the end {bound} of its range, but the "
                f"likelihood rises off it ({rise:.3g} per observation)"
            )
            held.append(k)
            break
    return held


def _no_higher_off_the_ends(model, surv_data, reported, held, free):
    """A fit with a parameter on an end of its range (``held``, see
    :func:`_on_range_ends`) is refitted from the reported parameters with
    each such parameter a tenth of its range into it: the likelihood the
    refit reaches must not be higher. The default start of a Weibull with
    ``lfp=True`` on interval-censored counts ran ``p`` to 1, a point the
    likelihood rises off, though by less than the tolerance there: a
    search off the end found a maximum 0.84 higher (#579)."""
    start = np.array(reported, dtype=float)
    for k in held:
        i = free[k]
        lo, hi = model.bounds[i]
        near_hi = abs(start[i] - hi) < abs(start[i] - lo)
        start[i] = hi - 0.1 * (hi - lo) if near_hi else lo + 0.1 * (hi - lo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = model.dist.fit_from_surpyval_data(
            surv_data,
            offset=model.offset,
            lfp=model.lfp,
            zi=model.zi,
            init=start,
        )
    gap = model.neg_ll() - refit.neg_ll()
    assert gap <= 1e-6 * max(1.0, abs(model.neg_ll())), (
        f"maximum='verified' with parameter(s) {[free[k] for k in held]} on "
        f"an end of their range, but a fit started off it reaches a "
        f"log-likelihood {gap:.4g} higher"
    )


def _search_mixture(model, data):
    """The mixture's own search space (``_pack``) and likelihood."""

    def fun(theta):
        return model.neg_ll_of(*model._unpack(theta))

    x = model._pack(model.w, model.params)
    return [Search(fun, x, float(np.sum(model.data.n)))]


def _search_regression(model, data):
    """``bounds_convert``'s space (the regression fits search in it, with
    units of 1), at the parameters the covariance is computed at: the
    centred fit's, where the baseline was moved to ``Z = 0`` (#463)."""
    names = list(model.parameter_names)
    held = model._held()
    free = [i for i, nm in enumerate(names) if nm not in held]
    if model._fit_centring is not None:
        p_hat, center = model._fit_centring[:2]
    else:
        p_hat, center = model._eval_params(), model.center
    p_hat = np.asarray(p_hat, dtype=float)
    fitted = model.data
    if center is not None and np.any(center):
        from surpyval.univariate.regression._fit_skeleton import centred_copy

        fitted = centred_copy(fitted, center)
    bounds = [model._parameter_bounds()[i] for i in free]
    pmap = {str(i): i for i in range(len(free))}
    to_search, to_natural, _, _, _ = bounds_convert(None, bounds, None, pmap)

    def fun(u):
        full = list(p_hat)
        for k, value in zip(free, to_natural(u)):
            full[k] = value
        return model.model.neg_ll(fitted, *full)

    tvc = getattr(model.model, "_tvc", None)
    if tvc is not None and model.is_tvc:
        # The accumulated-age likelihood: one term per subject
        n_obs = float(np.sum(tvc["weight"]))
    else:
        n_obs = float(np.sum(fitted.n))
    search = Search(fun, to_search(p_hat[free]), n_obs)
    coefs = _coefficients(names, free)
    return [search.in_covariate_units(coefs, model.data.Z)]


def _coefficients(names, free):
    """``(position, column)`` of each coefficient ``beta_<column>`` among
    the parameters ``names`` at the positions ``free``."""
    out = []
    for k, i in enumerate(free):
        head, _, column = names[i].rpartition("_")
        if head == "beta" and column.isdigit():
            out.append((k, int(column)))
    return out


def _search_frailty(model, data):
    """The marginal likelihood of the fitter, on the case's data, in the
    canonical space; a frailty variance at 0 is held out, the likelihood
    flat towards 0 and not rising away from it."""
    from surpyval.univariate.regression.frailty.frailty_fitter import (
        _AUTOGRAD,
        grouped_data,
    )

    if data is None:
        return None
    fitter = sp.Frailty(model.dist, family=model.family)
    x, Z, c, w, _, inv = grouped_data(
        data["x"], data["Z"], data["c"], data["n"], data["groups"]
    )
    n_beta = model.beta.size
    nat = np.r_[model.dist_params, model.beta, model.theta]
    bounds = [*model.dist.bounds, *[(None, None)] * n_beta, (0, None)]
    n_obs = float(np.sum(w))

    def neg_ll(v):
        return fitter._neg_ll_natural(
            anp.array(v), x, c, w, Z, inv, n_beta, _AUTOGRAD
        )

    search = _canonical(neg_ll, nat, bounds, n_obs).in_covariate_units(
        [(model.dist_params.size + j, j) for j in range(n_beta)], Z
    )
    f = search.fun(search.x)
    toward = search.x.copy()
    toward[-1] -= 10.0
    if abs(search.fun(toward) - f) <= 1e-12 * max(abs(f), 1.0):
        # On the boundary: the likelihood must not rise off it
        away = search.x.copy()
        away[-1] = np.log(1e-6)
        rise = (f - search.fun(away)) / 1e-6 / n_obs
        assert rise < OPTIMUM_GTOL, (
            "the frailty variance is at 0, but the likelihood rises as it "
            f"moves off it (by {rise:.3g} per observation per unit)"
        )
        search = search._replace(held=(search.x.size - 1,))
    return [search]


def _search_cox(model, data):
    """The partial likelihood's score and information (``model.jac``, of
    the centred fit) at the coefficients; aliased ones left out."""
    beta = np.asarray(model.res.x, dtype=float)
    kept = np.flatnonzero(~np.isnan(np.asarray(model.beta, float)))
    n_events = _cox_events(model, data)

    def jac(b):
        full = beta.copy()
        full[kept] = b
        return np.atleast_1d(model.jac(full)[0])[kept]

    def hess(b):
        full = beta.copy()
        full[kept] = b
        return np.atleast_2d(model.jac(full)[1])[np.ix_(kept, kept)]

    search = Search(
        lambda b: 0.0,
        beta[kept],
        n_events,
        jac=jac,
        hess=hess,
    )
    if data is None or data.get("Z") is None:
        return [search]  # (a starved fit: its data only the starve knows)
    Z = np.asarray(data["Z"], dtype=float).reshape(len(data["x"]), -1)
    return [search.in_covariate_units(_columns(kept), Z[:, kept])]


def _columns(kept):
    """``(position, column)`` of coefficients that are all searched, in
    the order of ``kept``'s columns (of a ``Z`` restricted to them)."""
    return [(k, k) for k in range(np.size(kept))]


def _cox_events(model, data):
    # The weighted events at each time of the baseline
    return max(float(np.sum(model.d)), 1.0)


def _search_cox_frailty(model, data):
    """The profile (integrated) likelihood in ``log theta``, from refits
    at fixed ``theta``; at ``theta = 0`` (no frailty, the Cox model) the
    Cox fit's partial likelihood, and the profile not rising off 0."""
    if data is None:
        return None
    fit = {k: data[k] for k in ("x", "Z", "c", "n", "groups")}
    n_obs = float(np.sum(data["n"]))
    tie_method = model.tie_method
    if model.theta == 0:
        cox = sp.CoxPH.fit(
            data["x"], data["Z"], data["c"], data["n"], tie_method=tie_method
        )
        off = sp.CoxFrailty.fit(**fit, theta=1e-6, tie_method=tie_method)
        rise = (off.log_likelihood - model.log_likelihood) / 1e-6 / n_obs
        assert rise < OPTIMUM_GTOL, (
            "theta is 0, but the profile likelihood rises as it moves off "
            f"it (by {rise:.3g} per observation per unit)"
        )
        return _search_cox(cox, data)

    def profile(log_theta):
        refit = sp.CoxFrailty.fit(
            **fit, theta=float(np.exp(log_theta)), tie_method=tie_method
        )
        return -refit.log_likelihood

    u = float(np.log(model.theta))
    h = 1e-3

    def jac(v):
        v = float(np.ravel(v)[0])
        return np.array([(profile(v + h) - profile(v - h)) / (2 * h)])

    def hess(v):
        v = float(np.ravel(v)[0])
        return np.array(
            [[(profile(v + h) - 2 * profile(v) + profile(v - h)) / h**2]]
        )

    return [Search(profile, np.array([u]), n_obs, jac=jac, hess=hess)]


def _search_proportional_odds(model, data):
    """The profile likelihood in ``gamma = -beta``, the baseline solved at
    each point (the fit's inner problem), by central differences."""
    from surpyval.univariate.regression._fit_skeleton import covariate_center
    from surpyval.univariate.regression.proportional_odds import (
        proportional_odds as po,
    )

    fd = model._fit_data
    x, c, n, Z, tl = fd["x"], fd["c"], fd["n"], fd["Z"], fd["tl"]
    lik = po._POLikelihood(x, c, n, tl, Z - covariate_center(Z, n))
    start = lik.start(x, n, tl)

    def fun(gamma):
        _, der = po._inner(lik, np.asarray(gamma, float), start, 1e-12)
        return -float(der["value"])

    gamma = -np.asarray(model.beta, dtype=float)
    n_events = float(n[c == 0].sum())
    search = _numerical(fun, gamma, n_events)
    return [search.in_covariate_units(_columns(gamma), Z)]


def _numerical(fun, x, n_obs):
    """A :class:`Search` of a likelihood written in plain numpy, by
    central differences (see :func:`_derivatives`)."""
    from surpyval.utils.linalg import numerical_gradient, numerical_hessian

    x = np.asarray(x, dtype=float)
    steps = 1e-5 * np.maximum(np.abs(x), 1.0)
    return Search(
        fun,
        x,
        n_obs,
        jac=lambda v: numerical_gradient(fun, v, 1e-2 * steps),
        hess=lambda v: numerical_hessian(fun, v, steps),
    )


def _search_fine_gray(model, data, name=""):
    """The weighted partial likelihood the fit kept (``_objective``, of
    the centred covariates), at the coefficients it did not alias."""
    if data is None:
        return None
    neg_ll = model._objective
    kept = ~np.isnan(np.asarray(model.beta, dtype=float))
    beta = np.asarray(model.coefficients, dtype=float)[kept]
    e = np.asarray(data["e"], dtype=object)
    c = np.asarray(data.get("c", np.zeros(e.size)))
    n = np.asarray(data.get("n", np.ones(e.size)), dtype=float)
    is_cause = np.array([v == model.cause for v in e])
    events = max(float(n[(c == 0) & is_cause].sum()), 1.0)
    Z = np.asarray(data["Z"], dtype=float).reshape(e.size, -1)[:, kept]
    search = Search(neg_ll, beta, events, name=name)
    return [search.in_covariate_units(_columns(beta), Z)]


def _search_crph(model, data):
    if data is None:
        return None
    if model.model == "Fine-Gray":
        return [
            s
            for event, fg in model._fg_models.items()
            for s in _search_fine_gray(fg, data, name=f"cause {event!r}")
        ]
    from surpyval.univariate.competing_risks.labels import label_mask

    out = []
    for event, i in model.event_idx_map.items():
        c_e = np.where(label_mask(np.asarray(data["e"]), event), 0, 1)
        cox = sp.CoxPH.fit(
            data["x"], data["Z"], c_e, data.get("n"), tie_method="efron"
        )
        cox.res.x = np.where(
            np.isnan(model.betas[i]), 0.0, np.asarray(model.betas[i], float)
        )
        cox.beta = np.asarray(model.betas[i], float)
        out += [
            s._replace(name=f"cause {event!r}")
            for s in _search_cox(cox, {**data, "c": c_e})
        ]
    return out


def _search_parametric_competing_risks(model, data):
    if data is None:
        return None
    out = []
    for k in model.causes:
        e = np.asarray(data["e"], dtype=object)
        c = np.asarray(data.get("c", np.zeros(e.size)))
        c_k = np.where(np.array([v == k for v in e]) & (c == 0), 0, 1)
        out += _search_parametric(
            model.models[k], {**data, "c": c_k}, name=f"cause {k!r}"
        )
    return out


def _on_bounds(neg_ll, x, bounds, n_obs):
    """The components of ``x`` on a bound of their space where the
    likelihood stops depending on them (it is the same, to rounding, a
    millionth of the way closer); each must be a maximum there, the
    likelihood not rising ``1e-6`` off the bound."""
    x = np.asarray(x, dtype=float)
    f = float(neg_ll(x))
    held = []
    for j, (lo, hi) in enumerate(bounds):
        for bound, inward in ((lo, 1.0), (hi, -1.0)):
            if bound is None:
                continue
            toward, away = x.copy(), x.copy()
            toward[j] = bound + (x[j] - bound) * 1e-6
            if abs(float(neg_ll(toward)) - f) > 1e-12 * max(abs(f), 1.0):
                continue
            away[j] = bound + inward * 1e-6
            rise = (f - float(neg_ll(away))) / 1e-6 / n_obs
            assert rise < OPTIMUM_GTOL, (
                f"parameter {j} is on its bound {bound}, but the likelihood "
                f"rises off it ({rise:.3g} per observation)"
            )
            held.append(j)
            break
    return held


def _search_recurrence(model, data, name=""):
    """The process's likelihood in its natural parameters (``_neg_ll``,
    which the likelihood inference reads), in the canonical space, at the
    parameters it estimated; a parameter on a bound of its space (an ARA
    repair efficiency of 1, a Kijima ``q`` of 0) held out where it is a
    maximum there (:func:`_on_bounds`)."""
    mle = np.asarray(model._mle, dtype=float)
    bounds = list(model._parameter_bounds())
    bounds += [(None, None)] * (mle.size - len(bounds))
    estimated = np.flatnonzero(~np.isnan(mle))
    with np.errstate(all="ignore"):
        on_bounds = _on_bounds(
            lambda v: model._neg_ll(np.where(np.isnan(mle), 0.0, v)),
            mle,
            bounds,
            float(model._n_obs),
        )
    free = np.array([i for i in estimated if i not in on_bounds], dtype=int)
    if free.size == 0:
        return []

    def neg_ll(v):
        # the free parameters from ``v``; those on a bound at their value,
        # and an aliased coefficient (nan) at the 0 it predicts with
        full = list(np.where(np.isnan(mle), 0.0, mle))
        for k, i in enumerate(free):
            full[i] = v[k]
        return model._neg_ll(anp.array(full))

    search = _canonical(
        neg_ll,
        mle[free],
        [bounds[i] for i in free],
        float(model._n_obs),
        name=name,
    )
    n_base = len(model._parameter_bounds())
    Z = getattr(getattr(model, "data", None), "Z", None)
    if Z is None or not np.size(Z):
        return [search]
    coefs = [(k, i - n_base) for k, i in enumerate(free) if i >= n_base]
    return [search.in_covariate_units(coefs, Z)]


def _search_cause_specific_nhpp(model, data):
    return [
        s
        for cause in model.event_types
        for s in _search_recurrence(model.models[cause], data, f"{cause!r}")
    ]


def _search_copula(model, data):
    """The two-stage (IFM) estimate: each margin's own fit, and the
    copula's likelihood in its parameters, the margins as fitted, in the
    space the fit searches (``_bounds_transforms``). A parameter on a
    bound of its family where the likelihood is highest (the AMH's
    ``theta = 1``, a Clayton at its independence end) is checked there
    instead (:func:`_on_bounds`), the others in the canonical space."""
    copula, fitted = model.copula, model.data
    out = []
    for d, margin in enumerate(model.margins):
        if getattr(margin, "maximum", None) == "verified":
            out += _search_parametric(margin, None, name=f"margin {d}")
    if not copula.parameter_names:
        return out
    dims = [
        copula._prepare_dim(model.margins[d], *fitted.dimension(d))
        for d in range(fitted.D)
    ]
    to_unbounded, to_bounded = copula._bounds_transforms()
    n_obs = float(np.sum(fitted.n))
    theta = np.asarray(model.params, float)

    def neg_ll(params):
        return float(copula.neg_ll(params, dims, fitted.n))

    with np.errstate(all="ignore"):
        held = _on_bounds(neg_ll, theta, list(copula.bounds), n_obs)
    if held:
        free = [j for j in range(theta.size) if j not in held]
        if not free:
            return out

        def reduced(v):
            full = list(theta)
            for k, j in enumerate(free):
                full[j] = v[k]
            return neg_ll(np.array(full, dtype=float))

        bounds = [copula.bounds[j] for j in free]
        search = _canonical(reduced, theta[free], bounds, n_obs)
        return out + [search._replace(name="copula")]

    def fun(phi):
        return copula.neg_ll(to_bounded(phi), dims, fitted.n)

    x = np.asarray(to_unbounded(theta), float)
    return out + [Search(fun, x, n_obs, name="copula")]


def _search_royston_parmar(model, data):
    """The spline coefficients' likelihood the fit kept (``_objective``);
    they are searched as they are."""
    return [_numerical(model._objective, model.params, float(model.n))]


def _increments(data):
    """The pooled increments ``(dt, dy)`` of degradation readings, each
    unit's in time order."""
    x, y, i = (np.asarray(data[k]) for k in ("x", "y", "i"))
    dts, dys = [], []
    for unit in np.unique(i):
        order = np.argsort(x[i == unit])
        dts.append(np.diff(x[i == unit][order]))
        dys.append(np.diff(y[i == unit][order]))
    return np.concatenate(dts), np.concatenate(dys)


def _search_wiener(model, data):
    """The Gaussian likelihood of the increments, ``N(mu dt, sigma^2
    dt)``, in ``(mu, log sigma)``."""
    if data is None or model.is_accelerated:
        return None
    dt, dy = _increments(data)

    def neg_ll(p):
        mu, sigma = p
        var = sigma**2 * dt
        return anp.sum(
            0.5 * anp.log(2 * np.pi * var) + (dy - mu * dt) ** 2 / (2 * var)
        )

    bounds = [(None, None), (0.0, None)]
    return [_canonical(neg_ll, model.params, bounds, float(dt.size))]


def _search_gamma_process(model, data):
    """The likelihood of the increments, ``Gamma(alpha dt, beta)``
    densities (a zero increment censored at the smallest positive one, the
    fit's default resolution), in ``(log alpha, log beta)``."""
    from scipy.special import gammainc, gammaln

    if data is None or model.is_accelerated:
        return None
    dt, dy = _increments(data)
    pos = dy > 0
    resolution = dy[pos].min()

    def neg_ll(p):
        alpha, beta = (float(v) for v in p)
        k = alpha * dt
        ll = np.sum(
            k[pos] * np.log(beta)
            + (k[pos] - 1) * np.log(dy[pos])
            - beta * dy[pos]
            - gammaln(k[pos])
        )
        ll += np.sum(np.log(gammainc(k[~pos], beta * resolution)))
        return -ll

    bounds = [(0.0, None), (0.0, None)]
    return [_canonical(neg_ll, model.params, bounds, float(dt.size))]


def _search_destructive(model, data):
    """The likelihood of the measurements the model kept, ``dist(beta0 +
    beta1 phi(x), sigma)`` (censored ones by their survival or CDF), in
    ``(beta0, beta1, log sigma)``."""
    x, y, c = (model.data[k] for k in ("x", "y", "c"))
    phi, dist = model._phi(x), model.distribution

    def neg_ll(p):
        b0, b1, sigma = (float(v) for v in p)
        loc = b0 + b1 * phi
        parts = [
            dist.log_df(y[c == 0], loc[c == 0], sigma),
            dist.log_sf(y[c == 1], loc[c == 1], sigma),
            dist.log_ff(y[c == -1], loc[c == -1], sigma),
        ]
        return -sum(float(np.sum(part)) for part in parts)

    params = [*model.beta, model.sigma]
    bounds = [(None, None), (None, None), (0.0, None)]
    return [_canonical(neg_ll, params, bounds, float(x.size))]


SEARCHES: dict[str, Callable] = {
    "Parametric": _search_parametric,
    "MixtureModel": _search_mixture,
    "RoystonParmarModel": _search_royston_parmar,
    "ParametricRegressionModel": _search_regression,
    "FrailtyModel": _search_frailty,
    "CoxFrailtyModel": _search_cox_frailty,
    "SemiParametricRegressionModel": _search_cox,
    "ProportionalOddsModel": _search_proportional_odds,
    "FineGrayModel": _search_fine_gray,
    "CompetingRisksProportionalHazards": _search_crph,
    "ParametricCompetingRisks": _search_parametric_competing_risks,
    "ParametricRecurrenceModel": _search_recurrence,
    "ProportionalIntensityModel": _search_recurrence,
    "RenewalModel": _search_recurrence,
    "CauseSpecificNHPP": _search_cause_specific_nhpp,
    "CopulaModel": _search_copula,
    "WienerProcessModel": _search_wiener,
    "GammaProcessModel": _search_gamma_process,
    "DestructiveDegradationModel": _search_destructive,
}


def test_every_family_has_its_check():
    # (a family whose fits are all known failures has none yet)
    missing = sorted(
        {
            case.model_class.rpartition(".")[2]
            for case in CASES
            if case.applies("maximum") and "maximum" not in case.xfail
        }
        - set(SEARCHES)
    )
    assert not missing, (
        "give these model classes their likelihood in SEARCHES: " f"{missing}"
    )
