"""The multi-state models: a case checked by hand, the special cases that
other estimators already give (competing risks, Kaplan-Meier), recovery
of known Markov models from simulated data, and the data checks.

The comparisons with R (``survival``, ``mstate``) are in
``surpyval/tests/reference/test_multistate.py``.
"""

import json
import pickle
import warnings
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

import surpyval as sp
from surpyval.multistate import (
    MultiState,
    MultiStateProportionalHazards,
    TransitionMatrix,
    from_stacked,
    to_stacked,
)
from surpyval.univariate.nonparametric.kaplan_meier import greenwood_variance

# ---------------------------------------------------------------------------
# By hand
# ---------------------------------------------------------------------------
# Unit 1 moves a -> b at 1 and back at 2, then is censored at 5; unit 2
# moves a -> b at 3 and is censored at 6; unit 3 is censored at 4.
HAND = dict(
    x=[1, 2, 5, 3, 6, 4],
    e=["b", "a", None, "b", None, None],
    state=["a", "b", "a", "a", "b", "a"],
    tl=[0, 1, 2, 0, 3, 0],
    i=[1, 1, 1, 2, 2, 3],
)


def test_hand_checked_transition_probabilities():
    model = MultiState.fit(**HAND)
    # t = 1: three in a, one leaves (1/3); t = 2: the one in b returns
    # (1/1); t = 3: three in a, one leaves (1/3).
    f = Fraction
    want = {
        0.5: [[1, 0], [0, 1]],
        1.5: [[f(2, 3), f(1, 3)], [0, 1]],
        2.5: [[1, 0], [1, 0]],
        3.5: [[f(2, 3), f(1, 3)], [f(2, 3), f(1, 3)]],
    }
    for t, P in want.items():
        assert_allclose(
            model.transition_probabilities(t),
            np.array(P, dtype=float),
            rtol=1e-15,
            atol=1e-15,
        )
    assert_allclose(
        model.cumulative_hazard([1, 3], "a", "b"), [1 / 3, 2 / 3], rtol=1e-15
    )
    assert_allclose(model.cumulative_hazard(2, "b", "a"), 1.0)


def test_hand_checked_variances_at_the_first_transition():
    model = MultiState.fit(**HAND)
    # One of r = 3 moves: Greenwood d (r - d) / r^3 = 2/27; Aalen's
    # increment variance d / r^2 = 1/9, which mstate's "aalen" recursion
    # weights by P_aa after the step, (2/3)^2, so 4/81; the jackknife: the
    # mover's influence (1 - 1/3) / 3, the others' -1/9 each, so 4/81 +
    # 1/81 + 1/81 = 2/27.
    for variance, var in (
        ("greenwood", 2 / 27),
        ("aalen", 4 / 81),
        ("robust", 2 / 27),
    ):
        se = model.occupancy_se(1.5, "b", from_state="a", variance=variance)
        assert_allclose(se, np.sqrt(var), rtol=1e-14)
    # Before any transition the probability, and its variance, are exact.
    assert model.occupancy_se(0.5, "b", variance="greenwood") == 0.0
    assert_allclose(model.cb(0.5, "b"), [0.0, 0.0])


def test_times_before_s_are_nan_and_s_is_the_identity():
    model = MultiState.fit(**HAND)
    p = model.occupancy([0.5, 1.5, 3.0, 3.5], "b", s=1.5, from_state="a")
    assert np.isnan(p[0])
    assert_allclose(p[1:], [0.0, 1 / 3, 1 / 3])
    assert np.all(np.isnan(model.transition_probabilities(1.0, s=1.5)))


def test_occupancies_add_up_to_one():
    model = MultiState.fit(**HAND)
    t = np.linspace(0, 6, 25)
    total = sum(model.occupancy(t, s) for s in model.states)
    assert_allclose(total, 1.0, rtol=0, atol=1e-15)


def test_bounds_past_the_data_are_nan_with_a_warning():
    model = MultiState.fit(**HAND)
    with pytest.warns(UserWarning, match="past the last observed value 6"):
        b = model.cb([3.5, 7.0], "b")
    assert np.all(np.isfinite(b[0])) and np.all(np.isnan(b[1]))
    # The estimate holds its last value.
    assert model.occupancy(7.0, "b") == model.occupancy(6.0, "b")


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------
def _competing_risks_data():
    rng = np.random.default_rng(3)
    size = 200
    x = np.round(rng.exponential(5, size), 1) + 0.1
    e = rng.choice(np.array(["a", "b", None], dtype=object), size)
    tl = np.round(x * rng.uniform(0, 0.6, size), 1)
    return x, e, tl


def test_competing_risks_is_the_cumulative_incidence():
    # One transient state and an absorbing state per cause: the
    # occupation probabilities are CompetingRisks' cumulative incidences
    # (and its Kaplan-Meier survival), with delayed entry and ties; the
    # robust variance is CompetingRisks' with weights of 1.
    x, e, tl = _competing_risks_data()
    cr = sp.CompetingRisks.fit(x, e, tl=tl, how="Kaplan-Meier")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crw = sp.CompetingRisks.fit(
            x, e, tl=tl, how="Kaplan-Meier", w=np.ones(x.size)
        )
    tm = TransitionMatrix.competing_risks(["a", "b"], initial="alive")
    model = MultiState.fit(x, e, ["alive"] * x.size, tl=tl, transitions=tm)
    q = np.linspace(0.5, 20, 40)
    for cause in ("a", "b"):
        assert_allclose(
            model.occupancy(q, cause), cr.cif(q, cause), rtol=0, atol=1e-15
        )
        width = crw.cb(q, cause, bound_type="normal")
        se = (width[:, 1] - width[:, 0]) / (2 * 1.959963984540054)
        assert_allclose(
            model.occupancy_se(q, cause, variance="robust"),
            se,
            rtol=1e-9,
            atol=1e-15,
        )
    assert_allclose(model.occupancy(q, "alive"), cr.sf(q), atol=1e-15)


def test_one_transition_is_kaplan_meier_with_greenwood():
    # Two states and one transition: the occupation of the first is the
    # Kaplan-Meier survival, and the Greenwood-type variance is
    # Greenwood's.
    rng = np.random.default_rng(4)
    x = np.round(rng.weibull(1.5, 80) * 10, 1) + 0.1
    died = rng.uniform(size=80) < 0.7
    e = np.where(died, "dead", None)
    model = MultiState.fit(x, e, ["alive"] * 80)
    km = sp.KaplanMeier.fit(x, c=np.where(died, 0, 1))
    q = np.linspace(1, 15, 15)
    assert_allclose(model.occupancy(q, "alive"), km.sf(q), atol=1e-15)
    idx = np.searchsorted(km.x, q, side="right") - 1
    var_H = greenwood_variance(km.r, km.d)
    want = km.sf(q) * np.sqrt(np.where(idx < 0, 0.0, var_H[idx]))
    assert_allclose(
        model.occupancy_se(q, "alive", variance="greenwood"),
        want,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Recovery of known Markov models
# ---------------------------------------------------------------------------
STATES = ("healthy", "ill", "dead")
PAIRS = [("healthy", "ill"), ("healthy", "dead"), ("ill", "healthy")]
PAIRS += [("ill", "dead")]
RATES = np.array([0.3, 0.1, 0.4, 0.3])


def _simulate(size, seed, beta=None):
    """Spells of an illness-death model with recovery and constant
    hazards ``RATES`` (times ``exp(beta * z)`` with ``z`` 0 or 1), half
    the units entering late (at a uniform time in (0, 1)), censored at a
    uniform time in (2, 8)."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(4) if beta is None else np.asarray(beta)
    rows = []
    for i in range(size):
        z = float(rng.integers(0, 2))
        rates = RATES * np.exp(beta * z)
        entry = rng.uniform(0, 1) if i % 2 else 0.0
        cens = rng.uniform(2, 8)
        t, s = 0.0, "healthy"
        while True:
            out = [k for k, (a, _) in enumerate(PAIRS) if a == s]
            total = rates[out].sum()
            nt = t + rng.exponential(1 / total)
            if nt >= cens:
                if cens > entry:
                    rows.append((i, max(t, entry), cens, s, None, z))
                break
            k = rng.choice(out, p=rates[out] / total)
            if nt > entry:
                rows.append((i, max(t, entry), nt, s, PAIRS[k][1], z))
            t, s = nt, PAIRS[k][1]
            if s == "dead":
                break
    cols = list(zip(*rows))
    return {
        "id": list(cols[0]),
        "tl": np.array(cols[1]),
        "x": np.array(cols[2]),
        "state": list(cols[3]),
        "e": list(cols[4]),
        "z": np.array(cols[5]),
    }


def _true_P(t, beta=None, z=0.0):
    beta = np.zeros(4) if beta is None else np.asarray(beta)
    Q = np.zeros((3, 3))
    for (a, b), rate in zip(PAIRS, RATES * np.exp(beta * z)):
        Q[STATES.index(a), STATES.index(b)] = rate
    Q -= np.diag(Q.sum(axis=1))
    return expm(Q * t)


TM = TransitionMatrix(PAIRS, states=STATES)


def test_recovers_a_markov_model():
    d = _simulate(3000, seed=20261112)
    model = MultiState.fit(
        d["x"], d["e"], d["state"], tl=d["tl"], i=d["id"], transitions=TM
    )
    times = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
    for a, start in enumerate(STATES[:2]):
        for b, state in enumerate(STATES):
            p = model.occupancy(times, state, s=0.0, from_state=start)
            truth = np.array([_true_P(t)[a, b] for t in times])
            se = model.occupancy_se(times, state, s=0.0, from_state=start)
            robust = model.occupancy_se(
                times, state, s=0.0, from_state=start, variance="robust"
            )
            assert np.all(np.abs(p - truth) <= 4 * se), (start, state)
            # The Greenwood-type and jackknife variances estimate the same
            ratio = robust / se
            assert np.all((ratio > 0.85) & (ratio < 1.15)), ratio


def test_cox_recovers_the_effects_and_the_probabilities():
    beta = np.array([0.7, 0.0, -0.5, 0.4])
    d = _simulate(3000, seed=20261113, beta=beta)
    model = MultiStateProportionalHazards.fit(
        d["x"],
        d["z"][:, None],
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=TM,
    )
    est, se = model.params, model.standard_errors()
    assert np.all(np.abs(est - beta) <= 4 * se), (est, se)
    times = np.array([1.0, 2.0, 4.0, 6.0])
    for z in (0.0, 1.0):
        P = model.transition_probabilities(times, [z], s=0.0)
        for b, state in enumerate(STATES):
            se_p = model.occupancy_se(
                times, [z], state, s=0.0, from_state="healthy"
            )
            truth = np.array([_true_P(t, beta, z)[0, b] for t in times])
            assert np.all(np.abs(P[:, 0, b] - truth) <= 4 * se_p), (z, b)


def test_transition_specific_effects_are_separate_cox_models():
    # Each covariate expanded per transition, stratified by transition:
    # the coefficients and baselines of a Cox model of each transition.
    d = _simulate(400, seed=7, beta=[0.5, 0.2, -0.3, 0.1])
    model = MultiStateProportionalHazards.fit(
        d["x"],
        d["z"][:, None],
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=TM,
        tie_method="breslow",
    )
    state = np.array(d["state"], dtype=object)
    e = np.array(d["e"], dtype=object)
    for h, (a, b) in enumerate(PAIRS):
        rows = state == a
        c = np.where(e[rows] == b, 0, 1)
        cox = sp.CoxPH.fit(
            d["x"][rows],
            d["z"][rows][:, None],
            c,
            tl=d["tl"][rows],
            tie_method="breslow",
        )
        assert_allclose(model.betas[h], cox.beta, rtol=1e-6)
        t = np.array([1.0, 3.0, 5.0])
        assert_allclose(
            model.cumulative_hazard(t, [0.0], a, b),
            cox.Hf(t, [0.0]),
            rtol=1e-6,
        )


def test_a_shared_effect_is_one_coefficient():
    d = _simulate(400, seed=8, beta=[0.3, 0.3, 0.3, 0.3])
    model = MultiStateProportionalHazards.fit(
        d["x"],
        np.column_stack([d["z"], np.array(d["id"]) % 3 == 0]),
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=TM,
        shared=[0],
    )
    assert model.parameter_names[0] == "z0"
    assert len(model.params) == 1 + 4
    assert_allclose(model.betas[:, 0], model.params[0])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def test_long_and_stacked_forms_round_trip():
    tm = TransitionMatrix(PAIRS, states=STATES)
    d = _simulate(30, seed=1)
    stacked = to_stacked(d["x"], d["e"], d["state"], tm, tl=d["tl"], i=d["id"])
    # One row per transition out of each spell's state
    out = {"healthy": 2, "ill": 2}
    assert len(stacked) == sum(out[s] for s in d["state"])
    assert stacked["status"].sum() == sum(v is not None for v in d["e"])
    back = from_stacked(
        stacked["id"],
        stacked["from"],
        stacked["to"],
        stacked["tl"],
        stacked["x"],
        stacked["status"],
    )
    a = MultiState.fit(**back, transitions=tm)
    b = MultiState.fit(
        d["x"], d["e"], d["state"], tl=d["tl"], i=d["id"], transitions=tm
    )
    q = np.linspace(0, 8, 9)
    for state in STATES:
        assert_allclose(a.occupancy(q, state), b.occupancy(q, state))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"tl": [0, 2, 5, 0, 3, 0]}, "at or after their exit time"),
        ({"tl": [0, 0.5, 2, 0, 3, 0]}, "overlap in time"),
        ({"e": ["b", "b", None, "b", None, None]}, "the same state"),
        ({"state": ["a", "b", "a", None, "b", "a"]}, "missing values"),
        ({"x": [1, 2, np.nan, 3, 6, 4]}, "missing"),
        ({"n": [1, 2, 1, 1, 1, 1]}, "differs between the rows of one"),
    ],
)
def test_bad_data_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        MultiState.fit(**{**HAND, **change})


def test_transitions_the_data_break_are_refused():
    with pytest.raises(ValueError, match="does not allow"):
        MultiState.fit(**HAND, transitions=[("a", "b")])
    with pytest.raises(ValueError, match="have no events"):
        MultiStateProportionalHazards.fit(
            **{k: v for k, v in HAND.items()},
            Z=np.ones((6, 1)),
            transitions=[("a", "b"), ("b", "a"), ("a", "c")],
        )


def test_missing_covariates_are_refused():
    Z = np.array([[0.0], [0.0], [np.nan], [1.0], [1.0], [0.0]])
    with pytest.raises(ValueError, match="missing or infinite covariates"):
        MultiStateProportionalHazards.fit(Z=Z, **HAND)


@pytest.mark.filterwarnings("ignore:Covariate column")
def test_variance_choices():
    model = MultiState.fit(**HAND)
    with pytest.raises(ValueError, match="variance"):
        model.occupancy_se(1.5, "b", variance="bootstrap")
    reg = MultiStateProportionalHazards.fit(
        Z=np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [0.0]]), **HAND
    )
    with pytest.raises(ValueError, match="no robust"):
        reg.occupancy_se(1.5, [0.0], "b", variance="robust")


# ---------------------------------------------------------------------------
# The transition matrix
# ---------------------------------------------------------------------------
def test_transition_matrix_shapes():
    tm = TransitionMatrix.illness_death(recovery=True)
    assert tm.n_states == 3 and tm.n_transitions == 4
    assert tm.absorbing == ("dead",)
    assert tm.transient == ("healthy", "ill")
    again = TransitionMatrix.from_matrix(tm.matrix >= 0, tm.states)
    # (read row by row: the same transitions, numbered in that order)
    assert set(again.transitions) == set(tm.transitions)
    assert TransitionMatrix.from_dict(json.loads(json.dumps(tm.to_dict())))
    with pytest.raises(ValueError, match="into itself"):
        TransitionMatrix([("a", "a")])
    with pytest.raises(ValueError, match="twice"):
        TransitionMatrix([("a", "b"), ("a", "b")])
    with pytest.raises(ValueError, match="Unknown state"):
        tm.index("gone")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:Covariate column")
def test_models_round_trip_and_pickle():
    model = MultiState.fit(**HAND, n=[1, 1, 1, 2, 2, 1])
    back = sp.from_json(model.to_json())
    q = np.linspace(0, 6, 13)
    for state in model.states:
        for variance in ("greenwood", "robust"):
            assert_allclose(
                back.occupancy_se(q, state, variance=variance),
                model.occupancy_se(q, state, variance=variance),
            )
    again = pickle.loads(pickle.dumps(model))
    assert_allclose(again.occupancy(q, "b"), model.occupancy(q, "b"))
    reg = MultiStateProportionalHazards.fit(
        Z=np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [0.5]]), **HAND
    )
    back = sp.from_dict(json.loads(json.dumps(reg.to_dict())))
    assert_allclose(back.params, reg.params)
    assert_allclose(
        back.occupancy(q, [0.5], "b"), reg.occupancy(q, [0.5], "b")
    )
