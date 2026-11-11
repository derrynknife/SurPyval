"""The parametric multi-state models (``ParametricMultiState`` and
``ParametricMultiStateRegression``): exact cases, the accuracy of the
forward equations, competing risks as a special case, recovery on
simulated Markov data, the delta-method bounds and the refusals."""

import json
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.linalg import expm

import surpyval
from surpyval import (
    Exponential,
    ExponentialPH,
    LogNormal,
    Weibull,
    WeibullAFT,
    WeibullPH,
)
from surpyval.multistate import (
    MultiState,
    ParametricMultiState,
    ParametricMultiStateRegression,
    TransitionMatrix,
)
from surpyval.univariate.competing_risks import ParametricCompetingRisks

ID = TransitionMatrix.illness_death(("healthy", "ill", "dead"))
REC = TransitionMatrix.illness_death(("healthy", "ill", "dead"), recovery=True)


def _simulate(n, params, cens, seed, Z=None, effect=None):
    """Spells of a clock-forward illness-death model with Weibull
    hazards ``params[(from, to)] = (alpha, beta)`` (no recovery),
    censored at uniform times in ``(0, cens)``. With covariates ``Z``
    (one per unit) each hazard is multiplied by ``exp(effect[pair] *
    z)``."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        z = 0.0 if Z is None else float(Z[unit])
        mult = {
            k: np.exp((0.0 if effect is None else effect[k]) * z)
            for k in params
        }

        def draw(pair, after):
            a, b = params[pair]
            # P(T > t | T > after) = exp(-(H(t) - H(after)) * mult)
            H = (after / a) ** b - np.log(rng.uniform()) / mult[pair]
            return a * H ** (1 / b)

        c = rng.uniform(0, cens)
        t_ill = draw(("healthy", "ill"), 0.0)
        t_dead = draw(("healthy", "dead"), 0.0)
        first = min(t_ill, t_dead)
        if first >= c:
            rows.append((unit, 0.0, c, "healthy", None, z))
            continue
        if t_dead < t_ill:
            rows.append((unit, 0.0, t_dead, "healthy", "dead", z))
            continue
        rows.append((unit, 0.0, t_ill, "healthy", "ill", z))
        t_d2 = draw(("ill", "dead"), t_ill)
        if t_d2 >= c:
            rows.append((unit, t_ill, c, "ill", None, z))
        else:
            rows.append((unit, t_ill, t_d2, "ill", "dead", z))
    cols = list(zip(*rows))
    return {
        "i": np.array(cols[0]),
        "tl": np.array(cols[1], dtype=float),
        "x": np.array(cols[2], dtype=float),
        "state": np.array(cols[3], dtype=object),
        "e": np.array(cols[4], dtype=object),
        "Z": np.array(cols[5], dtype=float)[:, None],
    }


def _closed_form(params, s, t):
    """P(s, t) of the illness-death model (no recovery) with Weibull
    hazards, by its integral form: P_hh = exp(-(H12 + H13)), P_ii =
    exp(-H23), P_hi = int_s^t P_hh(s, u) h12(u) P_ii(u, t) du (by
    adaptive quadrature to 1e-13)."""
    (a1, b1), (a2, b2), (a3, b3) = (
        params[("healthy", "ill")],
        params[("healthy", "dead")],
        params[("ill", "dead")],
    )

    def H(a, b, u):
        return (u / a) ** b

    def stay_h(u, v):
        return np.exp(
            -(H(a1, b1, v) - H(a1, b1, u)) - (H(a2, b2, v) - H(a2, b2, u))
        )

    def stay_i(u, v):
        return np.exp(-(H(a3, b3, v) - H(a3, b3, u)))

    def integrand(u):
        h12 = b1 / a1 * (u / a1) ** (b1 - 1)
        return stay_h(s, u) * h12 * stay_i(u, t)

    p_hi = quad(integrand, s, t, epsabs=1e-14, epsrel=1e-13, limit=200)[0]
    p_hh = stay_h(s, t)
    p_ii = stay_i(s, t)
    return np.array(
        [[p_hh, p_hi, 1 - p_hh - p_hi], [0.0, p_ii, 1 - p_ii], [0, 0, 1.0]]
    )


def _with_params(model, params):
    """``model`` with its Weibull transitions set to ``params``."""
    for pair, (a, b) in params.items():
        model.models[pair] = Weibull.from_params([a, b])
    model._cache = {}
    return model


def _template():
    """A model fitted to a few simulated units, whose transitions the
    tests set (``_with_params``)."""
    d = _simulate(50, TRUE, 20.0, 1)
    return ParametricMultiState.fit(
        d["x"], d["e"], d["state"], tl=d["tl"], transitions=ID
    )


TRUE = {
    ("healthy", "ill"): (8.0, 1.6),
    ("healthy", "dead"): (15.0, 0.7),
    ("ill", "dead"): (5.0, 2.2),
}


@pytest.fixture(scope="module")
def sim():
    return _simulate(1500, TRUE, 20.0, seed=20261112)


@pytest.fixture(scope="module")
def fitted(sim):
    return ParametricMultiState.fit(
        sim["x"],
        sim["e"],
        sim["state"],
        tl=sim["tl"],
        i=sim["i"],
        transitions=ID,
    )


# -- an exact case: constant hazards ------------------------------------------


def test_exponential_transitions_are_the_matrix_exponential():
    """With exponential transitions each rate is its events over its time
    at risk (the MLE in closed form), and P(s, t) = expm(Q (t - s))."""
    d = {
        "x": [2.0, 5.0, 3.0, 4.0, 6.0, 1.0, 7.0, 9.0],
        "e": ["ill", "dead", "dead", None, "ill", "healthy", None, "dead"],
        "state": ["healthy", "ill", "healthy", "healthy", "healthy", "ill",
                  "ill", "healthy"],
        "tl": [0.0, 2.0, 0.0, 0.0, 0.0, 0.5, 6.0, 1.0],
    }  # fmt: skip
    model = ParametricMultiState.fit(**d, transitions=REC, dist=Exponential)
    # by hand: time at risk in healthy 2+3+4+6+8 = 23, in ill 3+0.5+1 = 4.5
    rates = {
        ("healthy", "ill"): 2 / 23,
        ("healthy", "dead"): 2 / 23,
        ("ill", "dead"): 1 / 4.5,
        ("ill", "healthy"): 1 / 4.5,
    }
    Q = np.zeros((3, 3))
    for (a, b), r in rates.items():
        assert_allclose(model.models[(a, b)].params[0], r, rtol=1e-6)
        Q[REC.index(a), REC.index(b)] = model.models[(a, b)].params[0]
    Q[np.diag_indices(3)] = -Q.sum(axis=1)
    for s, t in ((0.0, 2.0), (1.0, 7.5), (3.0, 30.0)):
        assert_allclose(
            model.transition_probabilities(t, s=s),
            expm(Q * (t - s)),
            atol=1e-12,
        )
    assert_allclose(
        model.transition_probabilities(1.0, s=1.0), np.eye(3), atol=1e-15
    )


# -- the forward equations against the closed form ----------------------------


@pytest.mark.parametrize("s", [-1.0, 0.0, 3.0])
def test_forward_equations_reach_their_stated_accuracy(fitted, s):
    """Weibull hazards of different shapes (one infinite at 0) do not
    commute, so P(s, t) is not a matrix exponential; the integral form
    gives it to quadrature accuracy. The solver's rows are within the
    stated 1e-8."""
    model = _with_params(
        _template(),
        TRUE,
    )
    times = np.array([s + 0.01, s + 0.5, 4.0, 10.0, 25.0, 60.0])
    exact_from = max(s, 0.0)  # (no hazard before 0)
    P = model.transition_probabilities(times, s=s)
    for k, t in enumerate(times):
        exact = _closed_form(TRUE, exact_from, max(t, exact_from))
        err = np.abs(P[k] - exact).sum(axis=1).max()
        assert err < 1e-8, (t, err)


def test_forward_equations_hold_chapman_kolmogorov(fitted):
    P = fitted.transition_probabilities
    assert_allclose(P(12.0, s=0.0), P(5.0, s=0.0) @ P(12.0, s=5.0), atol=1e-8)


# -- competing risks as a special case ----------------------------------------


def test_competing_risks_are_a_special_case():
    """With one transient state the occupation probabilities of the
    absorbing states are ParametricCompetingRisks' cumulative incidences
    (the same fits, so to the solver's accuracy)."""
    rng = np.random.default_rng(7)
    ta = rng.weibull(1.8, 400) * 10
    tb = rng.weibull(0.8, 400) * 20
    c = rng.uniform(0, 25, 400)
    x = np.minimum.reduce([ta, tb, c])
    e = np.where(x == c, None, np.where(ta < tb, "a", "b"))
    cr = ParametricCompetingRisks.fit(x, e, dist=Weibull)
    tm = TransitionMatrix.competing_risks(["a", "b"], initial="alive")
    ms = ParametricMultiState.fit(
        x, e, ["alive"] * x.size, transitions=tm, dist=Weibull
    )
    assert_allclose(ms.models[("alive", "a")].params, cr.models["a"].params)
    times = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 50.0])
    for k in ("a", "b"):
        assert_allclose(ms.occupancy(times, k), cr.cif(times, k), atol=1e-8)
    assert_allclose(ms.occupancy(times, "alive"), cr.sf(times), atol=1e-10)


# -- recovery on simulated Markov data ----------------------------------------


def test_recovers_the_simulated_parameters(fitted):
    """Each transition's parameters within three standard errors of the
    truth (1500 units)."""
    est = fitted.params
    se = fitted.standard_errors()
    truth = np.concatenate([TRUE[k] for k in ID.transitions])
    z = np.abs(est - truth) / se
    assert np.all(z < 3), z


def test_recovers_the_occupation_probabilities(fitted):
    """The true occupation probabilities are within the 99% bounds, and
    the fit is close to the Aalen-Johansen estimate on the same data."""
    times = np.array([2.0, 5.0, 10.0, 15.0])
    aj = MultiState._from_data(fitted._data)
    for state in ID.states:
        truth = np.array([_closed_form(TRUE, 0.0, t)[0, ID.index(state)]
                          for t in times])  # fmt: skip
        cb = fitted.cb(times, state, from_state="healthy", alpha_ci=0.01)
        assert np.all((cb[:, 0] <= truth) & (truth <= cb[:, 1])), state
        assert_allclose(
            fitted.occupancy(times, state),
            aj.occupancy(times, state),
            atol=0.03,
        )


def test_recovers_covariate_effects():
    effect = {k: v for k, v in zip(ID.transitions, (0.7, -0.4, 0.0))}
    rng = np.random.default_rng(31)
    Z = rng.binomial(1, 0.5, 1500).astype(float)
    d = _simulate(1500, TRUE, 20.0, seed=32, Z=Z, effect=effect)
    model = ParametricMultiStateRegression.fit(
        d["x"],
        d["Z"],
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["i"],
        transitions=ID,
    )
    assert model.maximum == "verified"
    for pair in ID.transitions:
        m = model.models[pair]
        coef, se = m.params[-1], np.sqrt(np.diag(m.covariance()))[-1]
        assert abs(coef - effect[pair]) < 3 * se, (pair, coef, se)


# -- the delta-method bounds --------------------------------------------------


def test_delta_method_matches_simulation_of_the_parameters(fitted):
    """The delta-method standard error of an occupation probability is
    the standard deviation of the probability over parameters drawn from
    their asymptotic normal distribution (300 draws, so within 15%)."""
    times = np.array([3.0, 8.0])
    se = fitted.occupancy_se(times, "ill", from_state="healthy")
    rng = np.random.default_rng(5)
    draws = rng.multivariate_normal(
        fitted._estimates(), fitted.covariance(), 300
    )
    values = []
    for v in draws:
        m = _with_params(
            _template(),
            {k: tuple(v[2 * h : 2 * h + 2])
             for h, k in enumerate(ID.transitions)},
        )  # fmt: skip
        m._data = fitted._data
        values.append(m.occupancy(times, "ill", from_state="healthy"))
    sd = np.std(values, axis=0, ddof=1)
    assert_allclose(se, sd, rtol=0.15)


def test_bounds_conventions(fitted):
    p = fitted.occupancy([4.0, 9.0], "ill")
    cb = fitted.cb([4.0, 9.0], "ill")
    assert cb.shape == (2, 2)
    assert np.all((cb[:, 0] < p) & (p < cb[:, 1]))
    # a one-sided bound at alpha is a side of the two-sided at 2 alpha
    two = fitted.cb([4.0, 9.0], "ill", alpha_ci=0.1)
    assert_allclose(fitted.cb([4.0, 9.0], "ill", bound="lower"), two[:, 0])
    normal = fitted.cb([4.0, 9.0], "ill", bound_type="normal")
    se = fitted.occupancy_se([4.0, 9.0], "ill")
    assert_allclose(normal[:, 1] - p, 1.959963984540054 * se)
    # an absorbing state reached by nobody at s has no uncertainty there
    assert_allclose(fitted.cb(0.0, "dead", from_state="healthy"), [0, 0])


def test_extrapolates_past_the_data(fitted):
    """Unlike the Aalen-Johansen estimate, the curves go on past the last
    observation, with bounds."""
    late = fitted.last * 2
    p = fitted.occupancy(late, "dead")
    assert 0 < p < 1
    assert np.all(np.isfinite(fitted.cb(late, "dead")))


# -- predictions and the API --------------------------------------------------


def test_occupancy_sums_to_one_and_starts(fitted):
    t = np.array([0.0, 1.0, 7.0, 40.0])
    total = sum(fitted.occupancy(t, s) for s in ID.states)
    assert_allclose(total, 1.0, atol=1e-10)
    assert fitted.occupancy(0.0, "healthy") == pytest.approx(1.0)
    assert np.isnan(fitted.occupancy(2.0, "ill", s=3.0))
    P = fitted.transition_probabilities([[1.0, 2.0], [3.0, 4.0]])
    assert P.shape == (2, 2, 3, 3)


def test_from_state_and_a_later_start(fitted):
    P = fitted.transition_probabilities(9.0, s=4.0)
    assert_allclose(
        fitted.occupancy(9.0, "dead", s=4.0, from_state="ill"),
        P[1, 2],
    )


def test_cumulative_hazard_and_hazard_are_the_transitions(fitted):
    m = fitted.models[("ill", "dead")]
    assert_allclose(
        fitted.cumulative_hazard([1, 3], "ill", "dead"), m.Hf([1, 3])
    )
    assert_allclose(fitted.hazard([1, 3], "ill", "dead"), m.hf([1, 3]))
    with pytest.raises(ValueError):
        fitted.cumulative_hazard(1.0, "dead", "ill")


def test_neg_ll_and_information_criteria_add_over_transitions(fitted):
    models = [fitted.models[k] for k in ID.transitions]
    assert fitted.neg_ll() == pytest.approx(sum(m.neg_ll() for m in models))
    assert fitted.aic() == pytest.approx(sum(m.aic() for m in models))
    assert fitted.maximum == "verified"
    assert len(fitted.parameter_names) == fitted.params.size == 6
    assert fitted.covariance().shape == (6, 6)
    assert fitted.covariance()[0, 2] == 0


def test_a_distribution_per_transition(sim):
    dist = {
        ("healthy", "ill"): Weibull,
        ("healthy", "dead"): Exponential,
        ("ill", "dead"): LogNormal,
    }
    model = ParametricMultiState.fit(
        sim["x"],
        sim["e"],
        sim["state"],
        tl=sim["tl"],
        transitions=ID,
        dist=dist,
    )
    assert model.models[("healthy", "dead")].dist.name == "Exponential"
    assert model.parameter_names[2] == "healthy -> dead: failure_rate"
    total = sum(model.occupancy(10.0, s) for s in ID.states)
    assert total == pytest.approx(1.0)


def test_regression_with_other_fitters(sim):
    Z = (np.arange(sim["x"].size) % 2).astype(float)
    model = ParametricMultiStateRegression.fit(
        sim["x"],
        Z,
        sim["e"],
        sim["state"],
        tl=sim["tl"],
        transitions=ID,
        model={
            ("healthy", "ill"): WeibullAFT,
            ("healthy", "dead"): ExponentialPH,
            ("ill", "dead"): WeibullPH,
        },
    )
    p = model.occupancy([5.0, 5.0], [[0.0], [1.0]], "ill")
    assert np.all((0 < p) & (p < 1))
    assert model.cb(5.0, [1.0], "ill").shape == (2,)
    assert model.summary().index.names[0] == "transition"


def test_fit_from_df_is_fit(sim):
    import pandas as pd

    df = pd.DataFrame({k: sim[k] for k in ("x", "e", "state", "tl", "i")})
    df["z"] = (np.arange(len(df)) % 2).astype(float)
    a = ParametricMultiState.fit_from_df(
        df, "x", "e", "state", tl_col="tl", i_col="i", transitions=ID
    )
    b = ParametricMultiState.fit(
        sim["x"],
        sim["e"],
        sim["state"],
        tl=sim["tl"],
        i=sim["i"],
        transitions=ID,
    )
    assert_allclose(a.params, b.params)
    r = ParametricMultiStateRegression.fit_from_df(
        df, "x", ["z"], "e", "state", tl_col="tl", transitions=ID
    )
    assert r.parameter_names[2] == "healthy -> ill: z"


# -- refusals -----------------------------------------------------------------


def test_clock_reset_is_refused_with_the_reason(sim):
    with pytest.raises(ValueError, match="semi-Markov"):
        ParametricMultiState.fit(
            sim["x"], sim["e"], sim["state"], tl=sim["tl"], clock="reset"
        )
    with pytest.raises(ValueError, match="clock"):
        ParametricMultiState.fit(
            sim["x"], sim["e"], sim["state"], tl=sim["tl"], clock="gap"
        )


def test_refusals(sim):
    with pytest.raises(ValueError, match="no events"):
        ParametricMultiState.fit(
            [1.0, 2.0],
            ["dead", None],
            ["healthy", "healthy"],
            transitions=ID,
        )
    with pytest.raises(ValueError, match="no fitter"):
        ParametricMultiState.fit(
            sim["x"],
            sim["e"],
            sim["state"],
            tl=sim["tl"],
            transitions=ID,
            dist={("healthy", "ill"): Weibull},
        )
    with pytest.raises(ValueError, match="parametric distribution"):
        ParametricMultiState.fit(
            sim["x"], sim["e"], sim["state"], tl=sim["tl"], dist=WeibullPH
        )
    with pytest.raises(ValueError, match="parametric regression"):
        ParametricMultiStateRegression.fit(
            sim["x"],
            sim["Z"],
            sim["e"],
            sim["state"],
            tl=sim["tl"],
            model=Weibull,
        )
    with pytest.raises(ValueError, match="missing or infinite"):
        Z = sim["Z"].copy()
        Z[3] = np.nan
        ParametricMultiStateRegression.fit(
            sim["x"], Z, sim["e"], sim["state"], tl=sim["tl"]
        )


def test_a_start_before_the_support(fitted):
    """Before the distributions' support (0) the hazards are 0, so a
    prediction from s < 0 is the one from 0; an infinite start is
    refused."""
    assert_allclose(
        fitted.occupancy(5.0, "ill", s=-1.0, from_state="healthy"),
        fitted.occupancy(5.0, "ill", s=0.0, from_state="healthy"),
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="finite"):
        fitted.occupancy(5.0, "ill", s=np.inf)


# -- serialisation ------------------------------------------------------------


@pytest.mark.parametrize("regression", [False, True])
def test_json_round_trip(sim, regression):
    if regression:
        model = ParametricMultiStateRegression.fit(
            sim["x"],
            (np.arange(sim["x"].size) % 3).astype(float),
            sim["e"],
            sim["state"],
            tl=sim["tl"],
            i=sim["i"],
            transitions=ID,
        )
        args = ([2.0, 6.0], [1.0])
    else:
        model = ParametricMultiState.fit(
            sim["x"],
            sim["e"],
            sim["state"],
            tl=sim["tl"],
            i=sim["i"],
            transitions=ID,
        )
        args = ([2.0, 6.0],)
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert type(restored) is type(model)
    for state in ID.states:
        assert_allclose(
            restored.occupancy(*args, state), model.occupancy(*args, state)
        )
        assert_allclose(restored.cb(*args, state), model.cb(*args, state))
    assert restored.maximum == model.maximum
    assert restored.neg_ll() == pytest.approx(model.neg_ll())


def test_repr(fitted):
    text = repr(fitted)
    assert "healthy -> ill: Weibull" in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "Clock: forward" in text
