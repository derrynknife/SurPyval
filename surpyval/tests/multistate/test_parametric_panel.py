"""``ParametricPanelMultiState``: a distribution per transition fitted to
states seen only at visits, on the forward clock (non-homogeneous
Markov) and the reset clock (semi-Markov, progressive models)."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_ivp

from surpyval import Exponential, Weibull
from surpyval.multistate import (
    PanelMultiState,
    ParametricMultiState,
    ParametricPanelMultiState,
    TransitionMatrix,
)
from surpyval.multistate._data import check_spells
from surpyval.multistate._interval import IntervalLikelihood
from surpyval.multistate.panel import _check_panel
from surpyval.multistate.parametric import _fitters, _lows
from surpyval.tests.multistate._inspected import NAMES, TRANSITIONS, visits

TM = TransitionMatrix(TRANSITIONS)
PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}


def _visits(n, seed, clock="forward", gap=1.0, pars=PARS):
    ids, times, states, _ = visits(n, seed, pars, clock, gap)
    s = np.array([NAMES[int(v)] for v in states], dtype=object)
    obstype = np.where(s == "dead", "exact_entry", "panel")
    return dict(x=np.array(times), s=s, i=np.array(ids), obstype=obstype)


def _models(pars, tm=TM):
    return [Weibull.from_params(list(pars[p])) for p in tm.transitions]


def _lik(pdata, clock, dist=Weibull):
    return ParametricPanelMultiState._likelihood(
        pdata, _fitters(dist, pdata.tm, False), clock
    )


def _H(par, t):
    a, b = par
    return (np.maximum(t, 0.0) / a) ** b


def _h(par, t):
    a, b = par
    return b / a * (t / a) ** (b - 1)


@pytest.fixture(scope="module")
def small():
    return _visits(80, 7071)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_exponential_is_panel_multistate(small, clock):
    """With constant hazards both clocks are PanelMultiState's
    time-homogeneous Markov model (a progressive model: on the reset
    clock the time in a state does not matter)."""
    markov = PanelMultiState.fit(transitions=TM, **small)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ParametricPanelMultiState.fit(
            transitions=TM, dist=Exponential, clock=clock, **small
        )
    assert model.maximum == "verified"
    assert_allclose(model.log_likelihood, markov.log_likelihood, rtol=1e-10)
    assert_allclose(model.params, markov.params, rtol=1e-5)
    assert_allclose(
        model.standard_errors(), markov.standard_errors(), rtol=1e-4
    )
    assert model.n_obs == markov.n_obs
    assert_allclose(model.bic(), markov.bic(), rtol=1e-9)
    assert_allclose(
        model.occupancy([2.0, 5.0], "ill", from_state="healthy", s=0.0),
        markov.occupancy([2.0, 5.0], "ill", from_state="healthy", s=0.0),
        rtol=1e-5,
    )


def _P_forward(pars, t0, t1):
    """P(t0, t1) of the illness-death model on the forward clock, by an
    ODE solver."""
    names = ["healthy", "ill", "dead"]

    def rhs(t, p):
        P = p.reshape(3, 3)
        Q = np.zeros((3, 3))
        for (a, b), par in pars.items():
            Q[names.index(a), names.index(b)] = _h(par, t)
        Q -= np.diag(Q.sum(axis=1))
        return (P @ Q).ravel()

    sol = solve_ivp(rhs, (t0, t1), np.eye(3).ravel(), rtol=1e-12, atol=1e-14)
    return sol.y[:, -1].reshape(3, 3)


def test_forward_likelihood_is_the_product_of_transition_probabilities():
    """Each gap's factor is an element of P(t_{j-1}, t_j) (by an ODE
    solver here); an exactly timed death from a state not seen adds the
    hazard into death from each state; a censored visit sums the states
    it allows."""
    pars = {
        ("healthy", "ill"): (4.0, 1.5),
        ("healthy", "dead"): (12.0, 1.2),
        ("ill", "dead"): (5.0, 2.0),
    }
    x = [0.5, 1.5, 3.0, 0.5, 2.0, 4.5, 1.0, 2.5]
    s = [
        "healthy",
        "ill",
        "ill",
        "healthy",
        "healthy",
        "dead",
        "healthy",
        "alive",
    ]
    i = [1, 1, 1, 2, 2, 2, 3, 3]
    obstype = ["panel"] * 5 + ["exact_entry"] + ["panel"] * 2
    pdata = _check_panel(
        x,
        s,
        i,
        None,
        obstype,
        TM,
        {"alive": ["healthy", "ill"]},
        None,
        None,
    )
    lik = _lik(pdata, "forward")
    models = _models(pars)
    lik.plan(models)
    P = _P_forward
    h_id = _h(pars[("ill", "dead")], 4.5)
    h_hd = _h(pars[("healthy", "dead")], 4.5)
    P1 = P(pars, 0.5, 1.5)[0, 1] * P(pars, 1.5, 3.0)[1, 1]
    P24 = P(pars, 2.0, 4.5)
    P2 = P(pars, 0.5, 2.0)[0, 0] * (P24[0, 0] * h_hd + P24[0, 1] * h_id)
    P3 = P(pars, 1.0, 2.5)[0, :2].sum()
    assert_allclose(lik.neg_ll(models), -np.log(P1 * P2 * P3), rtol=1e-8)


def test_reset_likelihood_by_quadrature():
    """On the reset clock: a death at a known time from a state not seen
    (healthy, then dead: directly, or ill in between at an unknown time),
    and a death seen at a visit after the onset of illness was found at
    one, against scipy's adaptive quadrature. One shape is below 1, so
    that hazard is infinite at entry."""
    pars = {
        ("healthy", "ill"): (3.0, 0.8),
        ("healthy", "dead"): (10.0, 1.3),
        ("ill", "dead"): (4.0, 1.7),
    }
    hi, hd, idd = (pars[p] for p in TM.transitions)

    def Sh(d):
        return np.exp(-_H(hi, d) - _H(hd, d))

    def Si(d):
        return np.exp(-_H(idd, d))

    x = [0, 1, 2.5, 0, 1, 3]
    s = ["healthy", "healthy", "dead", "healthy", "ill", "dead"]
    i = [1, 1, 1, 2, 2, 2]
    obstype = ["panel", "panel", "exact_entry", "panel", "panel", "panel"]
    pdata = _check_panel(x, s, i, None, obstype, TM, None, None, None)
    lik = _lik(pdata, "reset")

    opts = dict(epsabs=0, epsrel=1e-12, limit=200)
    # unit 1: healthy at 0 and 1, dead at exactly 2.5
    via_ill = quad(
        lambda v: Sh(v) * _h(hi, v) * Si(2.5 - v) * _h(idd, 2.5 - v),
        1.0,
        2.5,
        **opts,
    )[0]
    L1 = Sh(2.5) * _h(hd, 2.5) + via_ill
    # unit 2: ill at 1 (fell ill in (0, 1]), dead by 3
    L2 = quad(
        lambda u: Sh(u) * _h(hi, u) * (Si(1 - u) - Si(3 - u)), 0, 1, **opts
    )[0]
    assert_allclose(lik.neg_ll(_models(pars)), -np.log(L1 * L2), rtol=1e-7)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_moves_one_at_a_time_are_the_interval_censored_fit(clock):
    """Where no state can be passed through unseen (a chain, at most one
    move between visits), the panel likelihood is that of the spells
    with each move interval censored (ParametricMultiState)."""
    tm = TransitionMatrix([("new", "worn"), ("worn", "failed")])
    pars = {("new", "worn"): (2.5, 1.4), ("worn", "failed"): (2.0, 0.9)}
    x = [0, 1, 2, 3, 0, 1.5, 3, 0, 2, 0, 1, 2.5, 0.5, 2, 3.5]
    s = [
        "new",
        "new",
        "worn",
        "failed",
        "new",
        "worn",
        "worn",
        "new",
        "new",
        "new",
        "worn",
        "failed",
        "new",
        "worn",
        "failed",
    ]
    i = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
    pdata = _check_panel(x, s, i, None, None, tm, None, None, None)
    lik = _lik(pdata, clock)
    models = _models(pars, tm)
    if clock == "forward":
        lik.plan(models)
    spells = check_spells(
        [
            [1, 2],
            [2, 3],
            [0, 1.5],
            [3, 3],
            [2, 2],
            [0, 1],
            [1, 2.5],
            [0.5, 2],
            [2, 3.5],
        ],
        [
            "worn",
            "failed",
            "worn",
            None,
            None,
            "worn",
            "failed",
            "worn",
            "failed",
        ],
        ["new", "worn", "new", "worn", "new", "new", "worn", "new", "worn"],
        [0, 2, 0, 1.5, 0, 0, 1, 0.5, 2],
        [1, 1, 2, 2, 3, 4, 4, 5, 5],
        None,
        tm,
        intervals=True,
    )
    fitters = _fitters(Weibull, tm, False)
    interval = IntervalLikelihood(
        spells,
        clock,
        ParametricMultiState._row_hazard(None),
        _lows(fitters),
    )
    assert_allclose(lik.neg_ll(models), interval.neg_ll(models), rtol=1e-8)


def test_weibull_fit_and_predictions():
    data = _visits(150, 7072)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ParametricPanelMultiState.fit(transitions=TM, **data)
    assert model.maximum == "verified"
    truth = np.concatenate([PARS[p] for p in TM.transitions])
    z = (model.params - truth) / model.standard_errors()
    assert np.all(np.abs(z) < 3.5)
    occ = model.occupancy([1.0, 4.0, 8.0], "ill")
    cb = model.cb([1.0, 4.0, 8.0], "ill")
    assert np.all((cb[..., 0] < occ) & (occ < cb[..., 1]))
    assert model.aic() == pytest.approx(2 * 6 + 2 * model.neg_ll())
    assert "states seen at visits" in repr(model)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_round_trip(small, clock):
    model = ParametricPanelMultiState.fit(
        transitions=TM, dist=Exponential, clock=clock, **small
    )
    back = ParametricPanelMultiState.from_dict(model.to_dict())
    assert_allclose(back.params, model.params)
    assert_allclose(back.covariance(), model.covariance())
    assert back.neg_ll() == model.neg_ll()
    assert_allclose(
        back.occupancy([1.0, 3.0], "dead"), model.occupancy([1.0, 3.0], "dead")
    )


def test_fit_from_df(small):
    import pandas as pd

    df = pd.DataFrame(
        {
            "t": small["x"],
            "st": small["s"],
            "u": small["i"],
            "o": small["obstype"],
        }
    )
    a = ParametricPanelMultiState.fit_from_df(
        df,
        "t",
        "st",
        "u",
        obstype_col="o",
        transitions=TM,
        dist=Exponential,
    )
    b = ParametricPanelMultiState.fit(
        transitions=TM, dist=Exponential, **small
    )
    assert_allclose(a.params, b.params)


def test_refusals(small):
    # (a model with a way back is taken on the reset clock:
    # test_parametric_panel_cycles.py)
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="max_moves"):
            ParametricPanelMultiState.fit(
                transitions=TM, clock="reset", max_moves=bad, **small
            )
    with pytest.raises(ValueError, match="truncation_tol"):
        ParametricPanelMultiState.fit(
            transitions=TM, clock="reset", truncation_tol=0.0, **small
        )
    with pytest.raises(ValueError, match="no allowed transitions lead"):
        ParametricPanelMultiState.fit(
            x=[0, 1, 0, 2],
            s=["ill", "healthy", "healthy", "ill"],
            i=[1, 1, 2, 2],
            transitions=TM,
        )
    with pytest.raises(ValueError, match="clock"):
        ParametricPanelMultiState.fit(transitions=TM, clock="back", **small)
    with pytest.raises(ValueError, match="transitions"):
        ParametricPanelMultiState.fit(**small)
