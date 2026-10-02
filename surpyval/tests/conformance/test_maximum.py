"""Every maximum-likelihood fit reports and verifies its maximum
(principles 12 and 13).

A fit that maximises a likelihood -- a parametric distribution, a
mixture, a parametric or semi-parametric regression, a frailty model, a
competing-risks model, a recurrence process, a copula -- records what it
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
  must not rise as it moves off the boundary instead.

The fixture's fit is checked, and so is the starved fit of the
convergence property (``Case.starve``), which reaches the other states:
it must say what it reached in the same way. Cases whose estimate is
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
from surpyval.tests.conformance.registry import CASES, cases_for, tvc_path
from surpyval.univariate.parametric.fitters import (
    OPTIMUM_GTOL,
    bounds_convert,
    is_local_minimum,
    search_floor,
)
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


def _params(key, where=None):
    """The cases of the property, marked with their known failures of
    ``key`` ("maximum", "maximum[starved]" or "maximum[tvc]")."""
    out = []
    for case in CASES:
        if not case.applies("maximum") or key in case.exclude:
            continue
        if where is not None and not where(case):
            continue
        marks = [pytest.mark.slow] if case.is_slow("maximum") else []
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


@pytest.mark.parametrize(
    "case", _params("maximum[starved]", lambda c: c.starve is not None)
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

    def _parts(self):
        x = np.asarray(self.x, dtype=float)
        keep = [i for i in range(x.size) if i not in self.held]
        jac = self.jac or jacobian(self.fun)
        hess = self.hess or hessian(self.fun)
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

        return [
            Search(
                fun,
                x,
                float(np.sum(surv_data.n)),
                search_floor(model),
                name=name,
            )
        ]

    # A closed form: no search vector of its own
    bounds = list(model.bounds)
    params = list(model.params)

    def neg_ll(v):
        return dist._neg_ll_func(surv_data, *v, 0.0, 0.0, 1.0)

    return [_canonical(neg_ll, params, bounds, float(np.sum(surv_data.n)))]


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
    return [Search(fun, to_search(p_hat[free]), n_obs)]


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

    search = _canonical(neg_ll, nat, bounds, n_obs)
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

    return [
        Search(
            lambda b: 0.0,
            beta[kept],
            n_events,
            jac=jac,
            hess=hess,
        )
    ]


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
        rise = (off.loglik - model.loglik) / 1e-6 / n_obs
        assert rise < OPTIMUM_GTOL, (
            "theta is 0, but the profile likelihood rises as it moves off "
            f"it (by {rise:.3g} per observation per unit)"
        )
        return _search_cox(cox, data)

    def profile(log_theta):
        refit = sp.CoxFrailty.fit(
            **fit, theta=float(np.exp(log_theta)), tie_method=tie_method
        )
        return -refit.loglik

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
    return [_numerical(fun, gamma, n_events)]


def _numerical(fun, x, n_obs):
    """A :class:`Search` of a likelihood autograd cannot differentiate,
    by central differences (``numerical_gradient``, ``numerical_hessian``,
    in steps relative to each component)."""
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
    return [Search(neg_ll, beta, events, name=name)]


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


def _search_recurrence(model, data, name=""):
    """The process's likelihood in its natural parameters (``_neg_ll``,
    which the likelihood inference reads), in the canonical space, at the
    parameters it estimated."""
    mle = np.asarray(model._mle, dtype=float)
    free = np.flatnonzero(~np.isnan(mle))
    bounds = list(model._parameter_bounds())
    bounds += [(None, None)] * (mle.size - len(bounds))

    def neg_ll(v):
        full = [mle[i] if np.isnan(mle[i]) else None for i in range(mle.size)]
        k = 0
        for i in range(mle.size):
            if full[i] is None:
                full[i] = v[k]
                k += 1
            else:
                full[i] = 0.0
        return model._neg_ll(anp.array(full))

    return [
        _canonical(
            neg_ll,
            mle[free],
            [bounds[i] for i in free],
            float(model._n_obs),
            name=name,
        )
    ]


def _search_cause_specific_nhpp(model, data):
    return [
        s
        for cause in model.event_types
        for s in _search_recurrence(model.models[cause], data, f"{cause!r}")
    ]


def _search_copula(model, data):
    """The copula's likelihood in its parameters, the margins as fitted
    (the two-stage estimate), in ``bounds_convert``'s space, which the fit
    searches."""
    copula, fitted = model.copula, model.data
    dims = [
        copula._prepare_dim(model.margins[d], *fitted.dimension(d))
        for d in range(fitted.D)
    ]
    to_unbounded, to_bounded = copula._bounds_transforms()

    def fun(phi):
        return copula.neg_ll(to_bounded(phi), dims, fitted.n)

    x = np.asarray(to_unbounded(np.asarray(model.params, float)), float)
    return [Search(fun, x, float(np.sum(fitted.n)))]


def _search_royston_parmar(model, data):
    """The spline coefficients' likelihood the fit kept (``_objective``);
    they are searched as they are."""
    return [_numerical(model._objective, model.params, float(model.n))]


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
}


def test_every_family_has_its_check():
    missing = sorted(
        {
            p.values[0].model_class.rpartition(".")[2]
            for p in cases_for("maximum")
        }
        - set(SEARCHES)
        - {"WienerProcessModel", "GammaProcessModel"}
        - {"DestructiveDegradationModel"}
    )
    assert not missing, (
        "give these model classes their likelihood in SEARCHES: " f"{missing}"
    )
