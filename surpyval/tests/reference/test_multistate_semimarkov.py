"""The semi-Markov ("clock reset") parametric multi-state models, and the
expected length of stay of both clocks, against R ``flexsurv``.

The values are in ``data/r_multistate_semimarkov.json``, written by
``scripts/reference/reference_r_multistate_semimarkov.R`` (R 4.6.2,
flexsurv 2.3.2, deSolve 1.42), on the fixtures of the parametric
comparison: ``survival::mgus2`` as illness-death spells and 300 simulated
units of an illness-death model with recovery.

What is compared:

- each transition's clock-reset fit (``flexsurvreg(Surv(Tstop - Tstart,
  status) ~ 1)``, Weibull, log-normal, exponential, and a Weibull
  proportional-hazards regression on sex): the maximised log-likelihoods
  (to 1e-7 relative, SurPyval's at least as high) and the parameters (to
  a twentieth of their standard errors);
- the transition probabilities of the clock-reset Weibull models with
  flexsurv's parameters put into SurPyval's transitions. flexsurv has no
  exact method for them, only simulation (``pmatrix.simfs``, 10^6 paths
  each), so they agree to the simulation's error: within five Monte Carlo
  standard errors, ``sqrt(p (1 - p) / M)``, plus the 5e-7 of flexsurv's
  printing to six decimals. (The exact check of the convolution is the
  closed form in ``tests/multistate/test_semimarkov.py``.) The cyclic
  structure of the simulated data (recovery) is included;
- the expected length of stay from the first state, against
  ``totlos.simfs`` (10^6 paths): within five times the largest Monte
  Carlo standard error a time in ``(0, t)`` can have, ``t / (2
  sqrt(M))``;
- the clock-forward expected length of stay against ``totlos.fs`` (the
  forward equations extended by the integral of ``P``, deSolve at ``rtol
  = atol = 1e-12``), with flexsurv's parameters: to 1e-7 of ``t``.
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

SOURCE = "r_multistate_semimarkov"
STATES = {
    "mgus2_multistate": ("mgus", "pcm", "death"),
    "multistate": ("healthy", "ill", "dead"),
}
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
M = 1e6


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


def _matrices(v, times, states):
    out = np.empty((len(times), 3, 3))
    for a, sa in enumerate(states):
        for b, sb in enumerate(states):
            out[:, a, b] = v[f"P_{sa}_{sb}"]
    return out


def _fit(name, dist, clock="reset"):
    _, sp = _spells(name)
    return ParametricMultiState.fit(
        **sp, transitions=_transitions(name), dist=DISTS[dist], clock=clock
    )


def _with_r_params(model, name, dist, v):
    for h, pair in enumerate(R_TRANSITIONS[name]):
        model.models[pair] = model.models[pair].with_params(
            _r_params(dist, v["params"][h])
        )
    model._cache = {}
    return model


def _within_simulation(P, R):
    se = np.sqrt(np.clip(R * (1 - R), 0, None) / M)
    assert np.all(np.abs(P - R) <= 5 * se + 5e-7), np.max(
        np.abs(P - R) / (se + 1e-12)
    )


CASES = [(name, dist) for name in STATES for dist in DISTS]


@pytest.mark.parametrize("name, dist", CASES)
def test_each_transition_is_flexsurvs_reset_fit(name, dist):
    model = _fit(name, dist)
    v = _values(f"flexsurv_reset_{SHORT[name]}_{dist}")
    for h, pair in enumerate(R_TRANSITIONS[name]):
        m = model.models[pair]
        ll, r_ll = -m.neg_ll(), v["loglik"][h]
        assert ll >= r_ll - 1e-9 * abs(r_ll), pair
        assert ll - r_ll < 1e-7 * abs(r_ll), pair
        se = np.sqrt(np.diag(m.covariance()))
        diff = np.abs(m.params - _r_params(dist, v["params"][h]))
        assert np.all(diff < 0.05 * se), (pair, diff / se)


@pytest.mark.parametrize("name", list(STATES))
def test_probabilities_are_flexsurvs_simulation(name):
    v = _values(f"flexsurv_reset_{SHORT[name]}_weibull")
    model = _with_r_params(_fit(name, "weibull"), name, "weibull", v)
    P = model.transition_probabilities(v["times"], s=0.0)
    _within_simulation(P, _matrices(v, v["times"], STATES[name]))


@pytest.mark.parametrize("name", list(STATES))
def test_length_of_stay_is_flexsurvs_simulation(name):
    v = _values(f"flexsurv_reset_{SHORT[name]}_weibull")
    model = _with_r_params(_fit(name, "weibull"), name, "weibull", v)
    t = v["los_t"]
    first = STATES[name][0]
    los = np.array(
        [
            model.length_of_stay(t, b, s=0.0, from_state=first)
            for b in STATES[name]
        ]
    )
    assert_allclose(los.sum(), t, rtol=1e-5)
    assert np.all(np.abs(los - v["los"]) <= 5 * t / (2 * np.sqrt(M)))


@pytest.mark.parametrize("name", list(STATES))
def test_forward_length_of_stay_is_flexsurvs(name):
    v = _values(f"flexsurv_forward_los_{SHORT[name]}")
    model = _with_r_params(
        _fit(name, "weibull", clock="forward"), name, "weibull", v
    )
    states = STATES[name]
    for k, t in enumerate(v["times"]):
        R = np.array(v[f"los_{k + 1}"])
        for a, sa in enumerate(states[:2]):
            los = [
                model.length_of_stay(t, b, s=0.0, from_state=sa)
                for b in states
            ]
            assert_allclose(los, R[a], atol=1e-7 * t)


def test_weibull_proportional_hazards_regression_reset():
    name = "mgus2_multistate"
    d, sp = _spells(name)
    v = _values("flexsurv_reset_mgus2_weibullPH_male")
    model = ParametricMultiStateRegression.fit(
        sp["x"],
        d["male"],
        sp["e"],
        sp["state"],
        tl=sp["tl"],
        i=sp["i"],
        transitions=_transitions(name),
        clock="reset",
    )
    for h, pair in enumerate(R_TRANSITIONS[name]):
        m = model.models[pair]
        shape, m_scale, coef = (float(u) for u in v["params"][h])
        assert_allclose(m.neg_ll(), -v["loglik"][h], rtol=1e-7)
        se = np.sqrt(np.diag(m.covariance()))
        r = np.array([m_scale ** (-1.0 / shape), shape, coef])
        assert np.all(np.abs(m.params - r) < 0.05 * se), pair
        model.models[pair] = m.with_params(r)
    model._cache = {}
    P = model.transition_probabilities(v["times"], [1.0], s=0.0)
    _within_simulation(P, _matrices(v, v["times"], STATES[name]))
