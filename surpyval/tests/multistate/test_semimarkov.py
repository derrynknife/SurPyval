"""Semi-Markov ("clock reset") parametric multi-state models: the fit on
the times spent in each state, the occupation probabilities and expected
lengths of stay from the convolution of the renewal equations (against a
closed form, the Markov special case and a simulation of a model with
recovery), the delta-method bounds, the refusals and serialisation."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from surpyval import Exponential, Normal, Weibull, WeibullPH
from surpyval.multistate import (
    ParametricMultiState,
    ParametricMultiStateRegression,
    TransitionMatrix,
)

STATES = ("healthy", "ill", "dead")
ID = TransitionMatrix.illness_death(STATES)
REC = TransitionMatrix.illness_death(STATES, recovery=True)


def _simulate(n, params, cens, seed, Z=None, effect=None):
    """Spells of a clock-reset multi-state model with Weibull hazards of
    the time in the state, ``params[(from, to)] = (alpha, beta)`` (with
    recovery where ``params`` has it), each unit censored at a uniform
    time in ``(0, cens)``; with ``Z`` (one per unit) each hazard is
    multiplied by ``exp(effect[pair] * z)``."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        z = 0.0 if Z is None else float(Z[unit])
        c = rng.uniform(0, cens)
        t, state = 0.0, "healthy"
        while state != "dead":
            draws = {}
            for (a, b), (alpha, beta) in params.items():
                if a != state:
                    continue
                mult = np.exp((0.0 if effect is None else effect[a, b]) * z)
                draws[b] = alpha * (-np.log(rng.uniform()) / mult) ** (
                    1 / beta
                )
            to = min(draws, key=draws.get)
            end = t + draws[to]
            if end >= c:
                rows.append((unit, t, c, state, None, z))
                break
            rows.append((unit, t, end, state, to, z))
            t, state = end, to
    cols = list(zip(*rows))
    return {
        "i": np.array(cols[0]),
        "tl": np.array(cols[1], dtype=float),
        "x": np.array(cols[2], dtype=float),
        "state": np.array(cols[3], dtype=object),
        "e": np.array(cols[4], dtype=object),
        "Z": np.array(cols[5], dtype=float)[:, None],
    }


ID_PARAMS = {
    ("healthy", "ill"): (10.0, 0.8),
    ("healthy", "dead"): (20.0, 1.5),
    ("ill", "dead"): (5.0, 1.3),
}
REC_PARAMS = {
    ("healthy", "ill"): (4.0, 1.4),
    ("healthy", "dead"): (15.0, 1.0),
    ("ill", "healthy"): (3.0, 0.8),
    ("ill", "dead"): (6.0, 1.6),
}


def _spells(d):
    return {k: d[k] for k in ("x", "e", "state", "tl", "i")}


@pytest.fixture(scope="module")
def id_data():
    return _simulate(400, ID_PARAMS, 30.0, 1)


@pytest.fixture(scope="module")
def id_model(id_data):
    return ParametricMultiState.fit(
        **_spells(id_data), transitions=ID, clock="reset"
    )


@pytest.fixture(scope="module")
def rec_data():
    return _simulate(400, REC_PARAMS, 20.0, 2)


def _set(model, params):
    for pair, p in params.items():
        model.models[pair] = model.models[pair].with_params(list(p))
    model._cache = {}
    return model


# -- the fit ------------------------------------------------------------------


def test_each_transition_is_fitted_on_the_time_in_its_state(id_data):
    model = ParametricMultiState.fit(
        **_spells(id_data), transitions=ID, clock="reset"
    )
    d = id_data
    ill = d["state"] == "ill"
    dur = d["x"][ill] - d["tl"][ill]
    c = np.where(d["e"][ill] == "dead", 0, 1)
    alone = Weibull.fit(dur, c=c)
    assert_allclose(
        model.models["ill", "dead"].params, alone.params, rtol=1e-8
    )
    # the first state starts at 0: the same as the clock forward
    fwd = ParametricMultiState.fit(**_spells(d), transitions=ID)
    assert_allclose(
        model.models["healthy", "ill"].params,
        fwd.models["healthy", "ill"].params,
    )
    assert "Clock: reset" in repr(model)


def test_recovers_the_simulated_parameters(rec_data):
    model = ParametricMultiState.fit(
        **_spells(rec_data), transitions=REC, clock="reset"
    )
    for pair, p in REC_PARAMS.items():
        m = model.models[pair]
        se = np.sqrt(np.diag(m.covariance()))
        assert np.all(np.abs(m.params - np.array(p)) < 4 * se), pair


# -- the probabilities --------------------------------------------------------


def _closed_form(t, params):
    """Illness-death with clock-reset Weibull hazards: P(healthy at t) is
    S_h(t), P(ill at t) the integral of f_hi(v) S_i(t - v), and the
    expected times their integrals."""

    def H(u, pair):
        a, b = params[pair]
        return (u / a) ** b

    def h(u, pair):
        a, b = params[pair]
        return b / a * (u / a) ** (b - 1)

    def S_h(u):
        return np.exp(-H(u, ("healthy", "ill")) - H(u, ("healthy", "dead")))

    def S_i(u):
        return np.exp(-H(u, ("ill", "dead")))

    def ill(u):
        return quad(
            lambda v: h(v, ("healthy", "ill")) * S_h(v) * S_i(u - v),
            0,
            u,
            limit=200,
            epsabs=1e-13,
            epsrel=1e-12,
        )[0]

    los_h = quad(S_h, 0, t, epsabs=1e-12)[0]
    los_i = quad(ill, 0, t, epsabs=1e-11, limit=200)[0]
    return S_h(t), ill(t), los_h, los_i


@pytest.mark.parametrize("t", [0.3, 2.0, 7.5, 20.0, 45.0])
def test_illness_death_is_the_closed_form(id_model, t):
    """To the solver's stated accuracy (1e-6 in a probability; 1e-6 of
    the time in an expected length of stay), between the grid's nodes,
    and past the data (45)."""
    model = ParametricMultiState.from_dict(id_model.to_dict())
    _set(model, ID_PARAMS)
    healthy, ill, los_h, los_i = _closed_form(t, ID_PARAMS)
    occ = [model.occupancy(t, b, s=0.0, from_state="healthy") for b in STATES]
    assert_allclose(occ[:2], [healthy, ill], atol=1e-6)
    assert_allclose(sum(occ), 1.0, atol=1e-6)
    los = [
        model.length_of_stay(t, b, s=0.0, from_state="healthy") for b in STATES
    ]
    assert_allclose(los[:2], [los_h, los_i], atol=1e-6 * max(t, 1.0))
    assert_allclose(sum(los), t, atol=1e-6 * max(t, 1.0))


def test_exponential_sojourns_are_the_markov_model(rec_data):
    """With constant hazards the clock does not matter: the reset model's
    convolution gives the forward equations' matrix exponential, with
    recovery (a cycle)."""
    reset = ParametricMultiState.fit(
        **_spells(rec_data), transitions=REC, dist=Exponential, clock="reset"
    )
    fwd = ParametricMultiState.fit(
        **_spells(rec_data), transitions=REC, dist=Exponential
    )
    for pair in REC.transitions:
        fwd.models[pair] = reset.models[pair]
    fwd._cache = {}
    times = [0.5, 3.0, 9.0, 25.0]
    assert_allclose(
        reset.transition_probabilities(times, s=0.0),
        fwd.transition_probabilities(times, s=0.0),
        atol=1e-6,
    )
    for b in STATES:
        assert_allclose(
            reset.length_of_stay(times, b, s=0.0, from_state="ill"),
            fwd.length_of_stay(times, b, s=0.0, from_state="ill"),
            atol=1e-6 * max(times),
        )


def test_recovery_is_the_simulation(rec_data):
    """A model with recovery and Weibull hazards of the time in each
    state, against 200 000 simulated paths (seeded): within four Monte
    Carlo standard errors."""
    model = ParametricMultiState.fit(
        **_spells(rec_data), transitions=REC, clock="reset"
    )
    _set(model, REC_PARAMS)
    times = np.array([1.0, 4.0, 10.0])
    paths = 200_000
    rng = np.random.default_rng(12)
    state = np.zeros(paths, dtype=int)  # 0 healthy, 1 ill, 2 dead
    t = np.zeros(paths)
    where = np.zeros((times.size, paths), dtype=int)
    stay = np.zeros((3, paths))
    index = {s: k for k, s in enumerate(STATES)}
    horizon = times[-1]
    alive = np.ones(paths, dtype=bool)
    while np.any(alive):
        draws = np.full((3, paths), np.inf)
        for (a, b), (alpha, beta) in REC_PARAMS.items():
            sel = alive & (state == index[a])
            u = rng.uniform(size=paths)
            draws[index[b], sel] = alpha * (-np.log(u[sel])) ** (1 / beta)
        to = np.argmin(draws, axis=0)
        end = t + np.min(draws, axis=0)
        for k, tt in enumerate(times):
            hit = alive & (t <= tt) & (end > tt)
            where[k, hit] = state[hit]
        np.add.at(
            stay,
            (state[alive], np.flatnonzero(alive)),
            (np.minimum(end, horizon) - t)[alive],
        )
        t = np.where(alive, end, t)
        state = np.where(alive, to, state)
        alive = alive & (state != 2) & (t < horizon)
    for k, tt in enumerate(times):
        dead = t <= tt
        where[k, dead & (state == 2)] = 2
    for k, tt in enumerate(times):
        for b in STATES:
            p_sim = np.mean(where[k] == index[b])
            p = model.occupancy(tt, b, s=0.0, from_state="healthy")
            se = np.sqrt(p_sim * (1 - p_sim) / paths)
            assert abs(p - p_sim) < 4 * se + 1e-9, (tt, b, p, p_sim)
    for b in ("healthy", "ill"):
        L = model.length_of_stay(horizon, b, s=0.0, from_state="healthy")
        values = stay[index[b]]
        se = values.std() / np.sqrt(paths)
        assert abs(L - values.mean()) < 4 * se, (b, L, values.mean())


def test_a_prediction_does_not_depend_on_the_other_times(id_model):
    one = id_model.occupancy(7.3, "ill")
    many = id_model.occupancy([0.1, 7.3, 60.0], "ill")
    # The same calculation, but numpy's vectorised loops can round the
    # last bit differently from the scalar one on x86-64 (one ulp seen).
    assert_allclose(one, many[1], rtol=1e-14, atol=0)
    assert np.isnan(id_model.occupancy(-1.0, "ill"))


def test_a_later_start_shifts_the_clock(id_model):
    """On the clock reset the hazards depend only on the time in the
    state, so a unit entering 'ill' at s = 5 is, at s + u, where one
    entering it at 0 is at u."""
    a = id_model.transition_probabilities([2.0, 8.0], s=0.0)
    b = id_model.transition_probabilities([7.0, 13.0], s=5.0)
    assert_allclose(a, b, atol=1e-12)
    with pytest.raises(ValueError, match="from_state"):
        id_model.occupancy(10.0, "ill", s=5.0)
    assert np.isfinite(
        id_model.occupancy(10.0, "ill", s=5.0, from_state="ill")
    )


# -- bounds -------------------------------------------------------------------


def test_delta_method_matches_simulation_of_the_parameters(id_model):
    """The delta-method standard errors of an occupation probability and
    an expected length of stay against the spread of the same over 300
    draws of the parameters from their estimated normal distribution
    (seeded), to 15%."""
    model = id_model
    t = 10.0
    se_p = model.occupancy_se(t, "ill")
    se_l = model.length_of_stay_se(t, "ill")
    rng = np.random.default_rng(5)
    pairs = list(ID.transitions)
    base = {p: model.models[p] for p in pairs}
    draws_p, draws_l = [], []
    # (the grid step stays the fitted model's, as in the delta method)
    steps = {k: v for k, v in model._cache.items() if k[0] == "step"}
    try:
        for _ in range(300):
            for p in pairs:
                m = base[p]
                v = rng.multivariate_normal(m._estimates(), m.covariance())
                model.models[p] = m.with_params(v)
            model._cache = dict(steps)
            draws_p.append(model.occupancy(t, "ill"))
            draws_l.append(model.length_of_stay(t, "ill"))
    finally:
        for p in pairs:
            model.models[p] = base[p]
        model._cache = {}
    assert_allclose(se_p, np.std(draws_p), rtol=0.15)
    assert_allclose(se_l, np.std(draws_l), rtol=0.15)


def test_bounds_conventions(id_model):
    t = [3.0, 10.0]
    p = id_model.occupancy(t, "ill")
    cb = id_model.cb(t, "ill")
    assert np.all((cb[:, 0] < p) & (p < cb[:, 1]))
    assert_allclose(
        id_model.cb(t, "ill", bound="lower", alpha_ci=0.025), cb[:, 0]
    )
    L = id_model.length_of_stay(t, "ill")
    lcb = id_model.length_of_stay_cb(t, "ill")
    assert np.all((lcb[:, 0] < L) & (L < lcb[:, 1]))
    assert np.all((lcb[:, 0] > 0) & (lcb[:, 1] < np.array(t)))
    assert_allclose(
        id_model.length_of_stay_cb(t, "ill", bound="upper", alpha_ci=0.025),
        lcb[:, 1],
    )
    assert_allclose(id_model.length_of_stay_cb(0.0, "ill"), [0.0, 0.0])


# -- regression ---------------------------------------------------------------


def test_regression_recovers_the_effects():
    rng = np.random.default_rng(3)
    Z = rng.binomial(1, 0.5, 500).astype(float)
    effect = {
        ("healthy", "ill"): 0.7,
        ("healthy", "dead"): 0.0,
        ("ill", "dead"): -0.6,
    }
    d = _simulate(500, ID_PARAMS, 30.0, 4, Z=Z, effect=effect)
    model = ParametricMultiStateRegression.fit(
        d["x"],
        d["Z"],
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["i"],
        transitions=ID,
        clock="reset",
    )
    for h, pair in enumerate(ID.transitions):
        se = np.sqrt(np.diag(model.models[pair].covariance()))[-1]
        assert abs(model.betas[h, 0] - effect[pair]) < 4 * se, pair
    # at z = 0 the predictions are those of the model without covariates
    # with each transition's baseline put in
    base = ParametricMultiState.fit(
        **_spells(d), transitions=ID, clock="reset"
    )
    for pair in ID.transitions:
        alpha, beta, _ = model.models[pair].params
        base.models[pair] = base.models[pair].with_params([alpha, beta])
    base._cache = {}
    assert_allclose(
        model.occupancy([2.0, 9.0], [0.0], "ill"),
        base.occupancy([2.0, 9.0], "ill"),
        atol=2e-6,
    )
    assert np.isfinite(model.length_of_stay_cb(9.0, [1.0], "ill")).all()


# -- refusals and serialisation -----------------------------------------------


def test_refusals(id_data):
    d = _spells(id_data)
    with pytest.raises(ValueError, match="dist.*Normal|0, inf"):
        ParametricMultiState.fit(
            **d, transitions=ID, dist=Normal, clock="reset"
        )
    tm = TransitionMatrix([("healthy", "ill")], states=("healthy", "ill"))
    with pytest.raises(ValueError, match="positive"):
        ParametricMultiState.fit(
            [0.0, 2.0, 3.0],
            ["ill", "ill", None],
            ["healthy"] * 3,
            transitions=tm,
            dist=Exponential,
            clock="reset",
        )


@pytest.mark.parametrize("regression", [False, True])
def test_json_round_trip(id_data, regression):
    if regression:
        model = ParametricMultiStateRegression.fit(
            id_data["x"],
            id_data["Z"] + np.arange(id_data["x"].size)[:, None] % 2,
            id_data["e"],
            id_data["state"],
            tl=id_data["tl"],
            transitions=ID,
            model=WeibullPH,
            clock="reset",
        )
        args = ([4.0, 9.0], [1.0], "ill")
    else:
        model = ParametricMultiState.fit(
            **_spells(id_data), transitions=ID, clock="reset"
        )
        args = ([4.0, 9.0], "ill")
    back = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
    assert back.clock == "reset"
    assert_allclose(back.occupancy(*args), model.occupancy(*args), rtol=1e-12)
    assert_allclose(back.cb(*args), model.cb(*args), rtol=1e-10)
    assert_allclose(
        back.length_of_stay(*args), model.length_of_stay(*args), rtol=1e-12
    )
