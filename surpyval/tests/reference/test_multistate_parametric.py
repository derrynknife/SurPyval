"""The parametric multi-state models (``ParametricMultiState``,
``ParametricMultiStateRegression``) against R ``flexsurv``.

The values are in ``data/r_multistate_parametric.json``, written by
``scripts/reference/reference_r_multistate_parametric.R`` (R 4.6.2,
flexsurv 2.3.2, deSolve 1.42), on the fixtures of the non-parametric
comparison: ``survival::mgus2`` as illness-death spells and 300 simulated
units of an illness-death model with recovery (delayed entry, ties).

What is compared:

- each transition's fit (``flexsurvreg(Surv(Tstart, Tstop, status) ~ 1,
  dist = d)`` on the transition's spells, Weibull, log-normal and
  exponential): the maximised log-likelihood (SurPyval's at least as
  high, flexsurv's within 1e-7 relative of it: on the simulated data
  flexsurv's optimiser stops up to 1.2e-5 short) and the parameters, to
  a twentieth of their standard errors;
- the transition probabilities ``P(0, t)`` (``pmatrix.fs``, deSolve at
  ``rtol = atol = 1e-12``) and ``P(s, t)`` (from them, as ``P(0, s)^-1
  P(0, t)``): with flexsurv's parameters put into SurPyval's transitions,
  so that only the solutions of the forward equations are compared, to
  1e-9 absolute; and from SurPyval's own fits, which carry the two
  optimisers' differences, to 1e-3;
- a Weibull proportional-hazards regression of each transition on sex:
  the log-likelihoods, coefficients and ``P(0, t | male)``;
- flexsurv's bounds (``ci = TRUE``: quantiles of ``P(0, t)`` over 1000
  parameter draws, a simulation) against SurPyval's delta-method bounds:
  the two are different approximations, so the check is that
  SurPyval's estimate is inside flexsurv's interval and the widths agree
  to 30%.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval import Exponential, LogNormal, Weibull
from surpyval.multistate import (
    ParametricMultiState,
    ParametricMultiStateRegression,
    TransitionMatrix,
)

from ._data import fixture, reference

SOURCE = "r_multistate_parametric"
STATES = {
    "mgus2_multistate": ("mgus", "pcm", "death"),
    "multistate": ("healthy", "ill", "dead"),
}
# The transitions in the order of the R script's numbering
R_TRANSITIONS = {
    "mgus2_multistate": [("mgus", "pcm"), ("mgus", "death"), ("pcm", "death")],
    "multistate": [
        ("healthy", "ill"),
        ("healthy", "dead"),
        ("ill", "healthy"),
        ("ill", "dead"),
    ],
}
SHORT = {"mgus2_multistate": "mgus2", "multistate": "sim"}
DISTS = {"weibull": Weibull, "lnorm": LogNormal, "exp": Exponential}


def _values(ref_id):
    return reference(SOURCE, ref_id)["values"]


def _spells(name):
    d = fixture(name)
    labels = STATES[name]
    return d, {
        "x": d["x"],
        "tl": d["tl"],
        "i": d["id"].astype(int),
        "state": [labels[int(v) - 1] for v in d["state"]],
        "e": [None if np.isnan(v) else labels[int(v) - 1] for v in d["e"]],
    }


def _transitions(name):
    return TransitionMatrix(R_TRANSITIONS[name], states=STATES[name])


def _r_params(dist, values):
    """flexsurv's parameters as SurPyval's (Weibull: scale, shape)."""
    p = np.atleast_1d(np.asarray(values, dtype=float))
    return p[::-1] if dist == "weibull" else p


def _matrices(v, times, prefix, states):
    out = np.empty((len(times), 3, 3))
    for a, sa in enumerate(states):
        for b, sb in enumerate(states):
            out[:, a, b] = v[f"{prefix}_{sa}_{sb}"]
    return out


@pytest.fixture(scope="module")
def fits():
    out = {}
    for name in STATES:
        _, sp = _spells(name)
        for dist, fitter in DISTS.items():
            out[name, dist] = ParametricMultiState.fit(
                **sp, transitions=_transitions(name), dist=fitter
            )
    return out


CASES = [(name, dist) for name in STATES for dist in DISTS]


@pytest.mark.parametrize("name, dist", CASES)
def test_each_transition_is_flexsurvs_fit(fits, name, dist):
    model = fits[name, dist]
    v = _values(f"flexsurv_{SHORT[name]}_{dist}")
    for h, pair in enumerate(R_TRANSITIONS[name]):
        m = model.models[pair]
        # The same maximum: SurPyval's at least as high (to rounding), and
        # flexsurv's within 1e-7 of it (on the simulated data its
        # optimiser stops up to 1.2e-5 short, 3e-8 relative).
        ll, r_ll = -m.neg_ll(), v["loglik"][h]
        assert ll >= r_ll - 1e-9 * abs(r_ll), pair
        assert ll - r_ll < 1e-7 * abs(r_ll), pair
        se = np.sqrt(np.diag(m.covariance()))
        diff = np.abs(m.params - _r_params(dist, v["params"][h]))
        assert np.all(diff < 0.05 * se), (pair, diff / se)


@pytest.mark.parametrize("name, dist", CASES)
def test_forward_equations_are_flexsurvs(fits, name, dist):
    """With flexsurv's parameters, SurPyval's solution of the forward
    equations is deSolve's to 1e-9 (both solvers' tolerances are far
    below it): P(0, t), and P(s, t) from a later start."""
    v = _values(f"flexsurv_{SHORT[name]}_{dist}")
    model = fits[name, dist]
    states = STATES[name]
    for h, pair in enumerate(R_TRANSITIONS[name]):
        model.models[pair] = model.models[pair].with_params(
            _r_params(dist, v["params"][h])
        )
    model._cache = {}
    try:
        P = model.transition_probabilities(v["times"], s=0.0)
        assert_allclose(P, _matrices(v, v["times"], "P", states), atol=1e-9)
        Ps = model.transition_probabilities(v["times_s"], s=v["s"])
        assert_allclose(
            Ps, _matrices(v, v["times_s"], "Ps", states), atol=1e-9
        )
    finally:
        _, sp = _spells(name)
        fits[name, dist] = ParametricMultiState.fit(
            **sp, transitions=_transitions(name), dist=DISTS[dist]
        )


@pytest.mark.parametrize("name, dist", CASES)
def test_transition_probabilities_from_the_fits(fits, name, dist):
    """From SurPyval's own fits, which differ from flexsurv's by the two
    optimisers' tolerances (on the simulated data's Weibull, flexsurv's
    ill -> healthy fit stops 1.2e-5 below SurPyval's maximum)."""
    v = _values(f"flexsurv_{SHORT[name]}_{dist}")
    P = fits[name, dist].transition_probabilities(v["times"], s=0.0)
    assert_allclose(P, _matrices(v, v["times"], "P", STATES[name]), atol=1e-3)


def test_weibull_proportional_hazards_regression():
    name = "mgus2_multistate"
    d, sp = _spells(name)
    v = _values("flexsurv_mgus2_weibullPH_male")
    model = ParametricMultiStateRegression.fit(
        sp["x"],
        d["male"],
        sp["e"],
        sp["state"],
        tl=sp["tl"],
        i=sp["i"],
        transitions=_transitions(name),
    )
    states = STATES[name]
    for h, pair in enumerate(R_TRANSITIONS[name]):
        m = model.models[pair]
        shape, m_scale, coef = (float(u) for u in v["params"][h])
        assert_allclose(m.neg_ll(), -v["loglik"][h], rtol=1e-8)
        se = np.sqrt(np.diag(m.covariance()))
        # flexsurv's weibullPH: H(t) = m t^shape, so alpha = m^(-1/shape)
        r = np.array([m_scale ** (-1.0 / shape), shape, coef])
        assert np.all(np.abs(m.params - r) < 0.01 * se), pair
    P = model.transition_probabilities(v["times"], [1.0], s=0.0)
    assert_allclose(P, _matrices(v, v["times"], "P", states), atol=1e-3)
    # with flexsurv's parameters, the solution is deSolve's
    for h, pair in enumerate(R_TRANSITIONS[name]):
        shape, m_scale, coef = (float(u) for u in v["params"][h])
        model.models[pair] = model.models[pair].with_params(
            [m_scale ** (-1.0 / shape), shape, coef]
        )
    model._cache = {}
    P = model.transition_probabilities(v["times"], [1.0], s=0.0)
    assert_allclose(P, _matrices(v, v["times"], "P", states), atol=1e-9)


def test_bounds_against_flexsurvs_simulated_intervals(fits):
    """flexsurv's intervals are quantiles over simulated parameters;
    SurPyval's are delta-method intervals on the log(-log) scale.
    SurPyval's estimate is inside flexsurv's interval, and the widths
    agree to 30% (from
    0.73 to 0.96 of flexsurv's, the narrowest for the rare state pcm
    starting in pcm, whose distribution is the most skewed)."""
    model = fits["mgus2_multistate", "weibull"]
    v = _values("flexsurv_mgus2_weibull_ci")
    states = STATES["mgus2_multistate"]
    for k, t in enumerate(v["times"]):
        for a in ("mgus", "pcm"):
            for b in states:
                if (a, b) == ("pcm", "mgus"):
                    continue
                lo, hi = v[f"L_{a}_{b}"][k], v[f"U_{a}_{b}"][k]
                p = model.occupancy(t, b, s=0.0, from_state=a)
                cb = model.cb(t, b, s=0.0, from_state=a)
                assert lo <= p <= hi, (t, a, b)
                ratio = (cb[1] - cb[0]) / (hi - lo)
                assert 0.7 < ratio < 1.3, (t, a, b, ratio)
