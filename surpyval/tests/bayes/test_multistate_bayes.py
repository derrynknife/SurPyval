"""Bayesian parametric multi-state models
(``ParametricMultiState.fit_bayes``): a composition of each
transition's ``fit_bayes``.

The references: each transition's posterior is its own fit's exactly;
a two-state model (alive to dead) is the univariate fit, its occupation
probability the univariate model's survival at the same draws; and the
occupation probabilities at each draw are the maximum-likelihood model's
machinery at that draw's parameters.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.multistate import ParametricMultiState
from surpyval.multistate import bayes as ms_bayes

SPELLS = dict(
    x=[2, 5, 3, 4, 6, 1, 7, 3, 8, 2, 4, 9, 5, 6],
    e=[
        "ill",
        "dead",
        "dead",
        None,
        "ill",
        "dead",
        None,
        "ill",
        None,
        "dead",
        "ill",
        None,
        "dead",
        None,
    ],
    state=[
        "healthy",
        "ill",
        "healthy",
        "healthy",
        "healthy",
        "ill",
        "ill",
        "healthy",
        "ill",
        "healthy",
        "healthy",
        "ill",
        "ill",
        "healthy",
    ],
    tl=[0, 2, 0, 0, 0, 0, 6, 0, 3, 0, 0, 4, 4, 0],
)


@pytest.fixture(scope="module")
def illness():
    return ParametricMultiState.fit_bayes(**SPELLS, dist=sp.Exponential)


def test_each_transition_is_its_own_fit(illness):
    x = np.asarray(SPELLS["x"], float)
    tl = np.asarray(SPELLS["tl"], float)
    state = np.asarray(SPELLS["state"], object)
    e = np.asarray(SPELLS["e"], object)
    rows = state == "ill"
    c = np.where(e[rows] == "dead", 0, 1)
    # (an entry at 0 truncates nothing, as for the fit)
    tl_ill = np.where(tl[rows] <= 0, -np.inf, tl[rows])
    alone = sp.Exponential.fit_bayes(x[rows], c, tl=tl_ill)
    got = illness.models[("ill", "dead")]
    np.testing.assert_array_equal(got.posterior.points, alone.posterior.points)
    np.testing.assert_array_equal(
        got.posterior.weights, alone.posterior.weights
    )
    assert set(illness.method.values()) == {"grid"}


def test_two_states_are_the_univariate_fit():
    x = sp.Weibull.random(40, 100, 1.5, random_state=3)
    c = (x > 150).astype(int)
    x = np.minimum(x, 150.0)
    e = np.where(c == 1, None, "dead").astype(object)
    two = ParametricMultiState.fit_bayes(x, e, state=["alive"] * 40)
    uni = sp.Weibull.fit_bayes(x, c)
    fit = two.models[("alive", "dead")]
    np.testing.assert_array_equal(fit.posterior.points, uni.posterior.points)
    np.testing.assert_array_equal(fit.posterior.weights, uni.posterior.weights)
    t = np.array([25.0, 50.0, 100.0, 140.0])
    # The occupation of 'alive' is the survival at each draw: its mean is
    # the univariate model's mean survival over the same draws ...
    models = uni.draw_models(ms_bayes.DRAWS, random_state=ms_bayes.SEED)
    mean_sf = np.mean([m.sf(t) for m in models], axis=0)
    np.testing.assert_allclose(two.occupancy(t, "alive"), mean_sf, atol=1e-7)
    np.testing.assert_allclose(
        two.occupancy(t, "dead"), 1 - mean_sf, atol=1e-7
    )
    # ... within Monte Carlo error of the exact predictive survival.
    sd = np.std([m.sf(t) for m in models], axis=0)
    assert np.all(
        np.abs(two.occupancy(t, "alive") - uni.sf(t))
        < 4 * sd / np.sqrt(ms_bayes.DRAWS)
    )
    # and the interval is the univariate one's, to Monte Carlo error
    lo, hi = two.occupancy_cri(100.0, "alive")
    ref = uni.cri(100.0)
    assert lo == pytest.approx(ref[0], abs=0.01)
    assert hi == pytest.approx(ref[1], abs=0.01)


def test_draws_are_the_fitted_machinery(illness):
    draws = illness._draw_models()
    mle = illness.mle
    t = np.array([2.0, 5.0, 8.0])
    vals, _ = illness._values("P", t, None, None)
    p0 = mle._p0(mle.start, None)
    for i in (0, 123, 999):
        P = mle._matrices(draws[i], np.zeros(0), mle.start, t, "P")
        np.testing.assert_allclose(vals[i], p0 @ P, atol=1e-12)
    # each draw's occupation probabilities sum to 1
    np.testing.assert_allclose(vals.sum(axis=-1), 1.0, atol=1e-8)


def test_predictions(illness):
    t = [2.0, 5.0, 8.0]
    occ = illness.occupancy(t, "dead")
    band = illness.occupancy_cri(t, "dead")
    assert band.shape == (3, 2)
    assert np.all((band[:, 0] < occ) & (occ < band[:, 1]))
    total = sum(illness.occupancy(t, s) for s in illness.states)
    np.testing.assert_allclose(total, 1.0, atol=1e-8)
    # near the maximum-likelihood estimate (a weak prior's posterior
    # mean is not the estimate, but close to it)
    assert np.all(np.abs(occ - illness.mle.occupancy(t, "dead")) < 0.1)
    P = illness.transition_probabilities(5.0)
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P.sum(axis=-1), 1.0, atol=1e-8)
    los = illness.length_of_stay(8.0, "healthy")
    lo, hi = illness.length_of_stay_cri(8.0, "healthy")
    assert lo < los < hi
    upper = illness.occupancy_cri(5.0, "dead", bound="upper")
    assert upper == pytest.approx(
        illness.occupancy_cri(5.0, "dead", alpha_ci=0.1)[1]
    )
    assert np.isnan(illness.occupancy(-1.0, "dead"))
    assert illness.param_cri(("ill", "dead"), "failure_rate").shape == (2,)
    with pytest.raises(ValueError, match="No transition"):
        illness.param_cri(("dead", "ill"), "failure_rate")


def test_clock_reset():
    model = ParametricMultiState.fit_bayes(
        **SPELLS, dist=sp.Exponential, clock="reset"
    )
    model.draws = 200
    occ = model.occupancy([2.0, 5.0], "ill", from_state="healthy")
    lo, hi = model.occupancy_cri(5.0, "ill", from_state="healthy")
    assert lo < occ[1] < hi


def test_priors_a_transition():
    prior = {"failure_rate": sp.Gamma.from_params([2.0, 10.0])}
    model = ParametricMultiState.fit_bayes(
        **SPELLS, dist=sp.Exponential, priors={("ill", "dead"): prior}
    )
    assert model.method[("ill", "dead")] == "conjugate"
    assert model.method[("healthy", "ill")] == "grid"


def test_saved_and_loaded(illness):
    back = ms_bayes.BayesianMultiState.from_dict(illness.to_dict())
    np.testing.assert_allclose(
        back.occupancy([2.0, 5.0], "dead"),
        illness.occupancy([2.0, 5.0], "dead"),
    )
    loaded = sp.from_dict(illness.to_dict())
    assert type(loaded).__name__ == "BayesianMultiState"
