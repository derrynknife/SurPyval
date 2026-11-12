"""The panel-data multi-state models: ``PanelMultiState`` and
``PanelMultiStateRegression``.

The checks against R ``msm`` are in
``surpyval/tests/reference/test_multistate_panel.py`` and the coverage of
the bounds in ``surpyval/tests/calibration/test_coverage_multistate_panel.py``.
Here: the exact derivatives against finite differences, closed forms the
model must reduce to, the refusals, and serialisation.
"""

import json
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

import surpyval as sp
from surpyval.multistate import (
    PanelMultiState,
    PanelMultiStateRegression,
    TransitionMatrix,
)
from surpyval.multistate import panel as pm
from surpyval.tests.conformance.registry_fixtures import panel_data
from surpyval.utils.linalg import numerical_gradient

ILLNESS = [("healthy", "ill"), ("healthy", "dead"), ("ill", "dead")]


def _simulate(rng, n, Q, states, gap=(0.5, 1.5), end=(4.0, 8.0), z=None):
    """Panel visits of a Markov chain with intensity matrix ``Q`` (a
    function of the unit's covariate where ``z`` is given), from state 0,
    deaths (the last state) timed exactly."""
    rows = []
    K = len(states)
    for unit in range(n):
        zu = None if z is None else float(z(rng))
        Qu = Q(zu) if callable(Q) else Q
        t, state, path = 0.0, 0, [(0.0, 0)]
        stop = rng.uniform(*end)
        while state != K - 1:
            rate = -Qu[state, state]
            t += rng.exponential(1 / rate)
            if t >= stop:
                break
            p = np.clip(Qu[state], 0, None)
            state = int(rng.choice(K, p=p / p.sum()))
            path.append((t, state))
        death = path[-1][0] if path[-1][1] == K - 1 else None
        v = 0.0
        while v < stop:
            if death is not None and v >= death:
                rows.append((unit, death, states[-1], zu))
                break
            now = [s for u, s in path if u <= v][-1]
            rows.append((unit, v, states[now], zu))
            v += rng.uniform(*gap)
        else:
            if death is not None and death < stop:
                rows.append((unit, death, states[-1], zu))
    i, x, s, zz = (np.array(c, dtype=object) for c in zip(*rows))
    return x.astype(float), s, i.astype(int), zz


@pytest.fixture(scope="module")
def illness_data():
    rng = np.random.default_rng(20261113)
    Q = np.array([[-0.5, 0.35, 0.15], [0.0, -0.4, 0.4], [0, 0, 0]])
    return _simulate(rng, 120, Q, ("healthy", "ill", "dead"))


# -- derivatives --------------------------------------------------------------


def _likelihood(model):
    return model._likelihood()


@pytest.mark.parametrize("with_Z", [False, True])
def test_gradient_is_that_of_the_likelihood(with_Z):
    """The forward recursion's exact gradient (block-triangular matrix
    exponentials for the derivatives of P, the product rule through each
    gap) against central differences, on data with every kind of row:
    snapshots, exactly timed deaths and censored states; and with
    covariates and period effects."""
    cls = PanelMultiStateRegression if with_Z else PanelMultiState
    d = panel_data(with_Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls.fit(**d, pci=[2.0])
    lik = _likelihood(model)
    rng = np.random.default_rng(1)
    for theta in (model._theta, model._theta + rng.normal(0, 0.3, 1)):
        theta = np.asarray(theta, float)
        assert_allclose(
            lik.jac(theta),
            numerical_gradient(lik.fun, theta, 1e-6 * np.ones(theta.size)),
            rtol=1e-6,
            atol=1e-6,
        )


def test_gradient_with_exact_moves():
    """Rows with ``obstype='exact'`` (a move timed exactly, the state
    held until it) have their own derivative path."""
    x = [0, 1.0, 2.5, 0, 0.7, 2.0, 3.0, 0, 1.5, 4.0]
    s = ["a", "b", "c", "a", "a", "b", "b", "a", "b", "c"]
    i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    ot = ["panel", "exact", "panel", "panel", "panel", "exact", "exact"]
    ot += ["panel", "panel", "exact"]
    model = PanelMultiState.fit(
        x, s, i, obstype=ot, transitions=[("a", "b"), ("b", "c")]
    )
    lik = _likelihood(model)
    theta = model._theta + 0.2
    assert_allclose(
        lik.jac(theta), numerical_gradient(lik.fun, theta), rtol=1e-6
    )


@pytest.mark.parametrize("from_state", [None, "healthy"])
def test_occupancy_se_is_the_delta_method(illness_data, from_state):
    """The standard error from the exact derivatives of P equals the
    delta method by finite differences of the occupancy through
    ``with_params``."""
    x, s, i, _ = illness_data
    model = PanelMultiState.fit(x, s, i, transitions=ILLNESS)
    times = np.array([0.5, 2.0, 5.0])
    cov = model.covariance()
    est = model.params

    def occ(p):
        return model.with_params(p).occupancy(times, "ill", 0.0, from_state)

    G = np.empty((times.size, est.size))
    for k in range(est.size):
        h = 1e-6 * est[k]
        e = np.zeros(est.size)
        e[k] = h
        G[:, k] = (occ(est + e) - occ(est - e)) / (2 * h)
    se = np.sqrt(np.einsum("ip,pq,iq->i", G, cov, G))
    assert_allclose(
        model.occupancy_se(times, "ill", 0.0, from_state), se, rtol=1e-6
    )


# -- closed forms -------------------------------------------------------------


def test_exactly_observed_moves_give_the_occurrence_rates():
    """Where every move is seen when it happens (obstype 'exact'), the
    likelihood is that of exponential stays and each intensity is its
    number of moves over the time spent in its 'from' state."""
    rng = np.random.default_rng(5)
    q = {("a", "b"): 0.4, ("a", "c"): 0.1, ("b", "c"): 0.3}
    rows = []
    counts = dict.fromkeys(q, 0)
    time_in = {"a": 0.0, "b": 0.0}
    for unit in range(60):
        t, st = 0.0, "a"
        rows.append((unit, 0.0, "a", "panel"))
        end = 6.0
        while st != "c":
            out = [(k, r) for k, r in q.items() if k[0] == st]
            total = sum(r for _, r in out)
            dt = rng.exponential(1 / total)
            if t + dt >= end:
                time_in[st] += end - t
                rows.append((unit, end, st, "exact"))
                break
            time_in[st] += dt
            t += dt
            k = rng.choice(len(out), p=[r / total for _, r in out])
            counts[out[k][0]] += 1
            st = out[k][0][1]
            rows.append((unit, t, st, "exact"))
    i, x, s, ot = zip(*rows)
    model = PanelMultiState.fit(x, s, i, obstype=ot, transitions=list(q))
    expected = [counts[k] / time_in[k[0]] for k in q]
    assert_allclose(model.params, expected, rtol=1e-7)
    # the information is the Poisson one: var(q) = q^2 / n
    assert_allclose(
        model.standard_errors(),
        np.array(expected) / np.sqrt([counts[k] for k in q]),
        rtol=1e-4,
    )


def test_two_states_is_the_interval_censored_exponential():
    """Two states, alive and dead, seen at visits: the likelihood is the
    exponential's, a death found between two visits interval censored
    and a unit alive at its last visit right censored."""
    rng = np.random.default_rng(11)
    rows, xl, xr, c = [], [], [], []
    for unit in range(80):
        life = rng.exponential(4.0)
        visits = np.cumsum(rng.uniform(0.5, 2.0, 8))
        rows.append((unit, 0.0, "alive"))
        last = 0.0
        for v in visits:
            if v >= life:
                rows.append((unit, v, "dead"))
                xl.append(last)
                xr.append(v)
                c.append(2)
                break
            rows.append((unit, v, "alive"))
            last = v
        else:
            xl.append(last)
            xr.append(last)
            c.append(1)
    i, x, s = zip(*rows)
    model = PanelMultiState.fit(x, s, i, transitions=[("alive", "dead")])
    expo = sp.Exponential.fit(x=np.column_stack([xl, xr]), c=c)
    assert_allclose(model.params[0], expo.params[0], rtol=1e-6)
    assert_allclose(model.log_likelihood, -expo.neg_ll(), rtol=1e-9)


def test_transition_probabilities_are_the_matrix_exponential(illness_data):
    x, s, i, _ = illness_data
    model = PanelMultiState.fit(x, s, i, transitions=ILLNESS)
    Q = model.intensity_matrix()
    assert_allclose(Q.sum(axis=1), 0.0, atol=1e-14)
    P = model.transition_probabilities([1.0, 3.0], s=0.5)
    assert_allclose(P[0], expm(Q * 0.5), atol=1e-12)
    assert_allclose(P[1], expm(Q * 2.5), atol=1e-12)
    assert np.all(np.isnan(model.transition_probabilities(0.2, s=0.5)))
    # occupancies add up to 1, and from a state are a row of P
    occ = [model.occupancy(3.0, st, 0.5, "healthy") for st in model.states]
    assert_allclose(occ, P[1][0], atol=1e-12)
    assert_allclose(model.mean_sojourn("healthy"), -1 / Q[0, 0])


def test_pci_probabilities_chain_the_periods(illness_data):
    """With intensities that change at a cut, P(s, t) is the product of
    each period's matrix exponential."""
    x, s, i, _ = illness_data
    model = PanelMultiState.fit(x, s, i, transitions=ILLNESS, pci=[3.0])
    Q0 = model.intensity_matrix()
    Q1 = model.intensity_matrix(at=4.0)
    ratio = Q1[[0, 0, 1], [1, 2, 2]] / Q0[[0, 0, 1], [1, 2, 2]]
    assert_allclose(np.log(ratio), model.params[3:], rtol=1e-10)
    P = model.transition_probabilities(5.0, s=1.0)
    assert_allclose(P, expm(Q0 * 2.0) @ expm(Q1 * 2.0), atol=1e-12)
    with pytest.raises(ValueError, match="pci"):
        model.mean_sojourn("healthy")


def test_regression_predicts_with_the_intensity_ratios():
    rng = np.random.default_rng(3)

    def Q(z):
        b = np.exp(0.7 * z)
        return np.array(
            [[-0.5 * b, 0.35 * b, 0.15 * b], [0, -0.4, 0.4], [0, 0, 0]]
        )

    x, s, i, z = _simulate(
        rng, 150, Q, ("healthy", "ill", "dead"), z=lambda r: r.integers(2)
    )
    model = PanelMultiStateRegression.fit(
        x, z.astype(float), s, i, transitions=ILLNESS
    )
    Q1 = model.intensity_matrix([1.0])
    Q0 = model.intensity_matrix([0.0])
    assert_allclose(Q1[0, 1] / Q0[0, 1], np.exp(model.betas[0, 0]))
    assert_allclose(
        model.transition_probabilities(2.0, [1.0], s=0.0), expm(Q1 * 2.0)
    )
    # a missing covariate gives NaN for its row only
    occ = model.occupancy([2.0, 2.0], [[1.0], [np.nan]], "ill", s=0.0)
    assert np.isfinite(occ[0]) and np.isnan(occ[1])
    # shared: one coefficient for every transition
    shared = PanelMultiStateRegression.fit(
        x, z.astype(float), s, i, transitions=ILLNESS, shared=0
    )
    assert shared.parameter_names[-1] == "z0"
    assert np.ptp(shared.betas) == 0
    assert shared.log_likelihood <= model.log_likelihood + 1e-8


def test_a_constant_covariate_is_aliased(illness_data):
    x, s, i, _ = illness_data
    Z = np.column_stack([np.ones(x.size), np.arange(x.size) % 2])
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = PanelMultiStateRegression.fit(x, Z, s, i, transitions=ILLNESS)
    assert list(model.aliased) == [0]
    assert np.all(np.isnan(model.betas[:, 0]))
    assert np.all(np.isfinite(model.betas[:, 1]))


def test_a_move_the_data_do_not_support_has_no_finite_maximum():
    """Every death is seen after an 'ill' visit, so the direct move from
    'healthy' to 'dead' has its maximum at an intensity of 0."""
    x = [0, 1, 2, 0, 1, 2, 3, 0, 2, 3, 0, 1]
    s = ["healthy", "ill", "dead", "healthy", "healthy", "ill", "dead"]
    s += ["healthy", "ill", "dead", "healthy", "healthy"]
    i = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = PanelMultiState.fit(x, s, i, transitions=ILLNESS)
    assert model.maximum == "no finite maximum"


# -- refusals -----------------------------------------------------------------


def _fit(**kw):
    base = dict(
        x=[0, 1, 2, 0, 1],
        s=["healthy", "ill", "dead", "healthy", "ill"],
        i=[1, 1, 1, 2, 2],
        transitions=ILLNESS,
    )
    base.update(kw)
    with warnings.catch_warnings():
        # (a fit this small has no finite maximum for healthy -> dead)
        warnings.simplefilter("ignore")
        return PanelMultiState.fit(**base)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"transitions": None}, "'transitions' is needed"),
        ({"s": ["healthy", "ill", "gone", "healthy", "ill"]}, "neither"),
        ({"s": ["healthy", None, "dead", "healthy", "ill"]}, "missing"),
        ({"x": [0, 1, np.nan, 0, 1]}, "missing or infinite"),
        ({"x": [0, 1, 1, 0, 1]}, "two rows at time"),
        (
            {"s": ["ill", "healthy", "dead", "healthy", "ill"]},
            "cannot explain",
        ),
        ({"exact_entry": "ill"}, "not an absorbing state"),
        ({"obstype": ["panel"] * 4 + ["sometimes"]}, "obstype"),
        (
            {"censored": {"ill": ["healthy"]}},
            "also a state",
        ),
        (
            {
                "s": ["alive", "ill", "dead", "healthy", "ill"],
                "censored": {"alive": ["healthy", "ill"]},
            },
            "first visit must see one state",
        ),
        ({"i": [1, 1, 1, 2, 3]}, None),
        ({"pci": [2.0, 1.0]}, "increasing"),
    ],
)
def test_refusals_say_what_to_do(kw, match):
    if match is None:
        # a unit seen once adds nothing, and is accepted
        _fit(**kw)
        return
    with pytest.raises(ValueError, match=match):
        _fit(**kw)


def test_a_unit_with_two_visits_is_needed():
    with pytest.raises(ValueError, match="two visits"):
        PanelMultiState.fit(
            [0, 0], ["healthy", "ill"], [1, 2], transitions=ILLNESS
        )


def test_sojourn_of_an_absorbing_state_is_refused(illness_data):
    x, s, i, _ = illness_data
    model = PanelMultiState.fit(x, s, i, transitions=ILLNESS)
    with pytest.raises(ValueError, match="absorbing"):
        model.mean_sojourn("dead")
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.param_cb("healthy -> ill")


def test_censored_state_sums_over_its_states():
    """A visit coded with a set of states contributes the sum of the
    probabilities of those states (and a set of one state is that
    state)."""
    d = dict(
        x=[0, 1, 2, 0, 1, 3],
        i=[1, 1, 1, 2, 2, 2],
        transitions=ILLNESS,
    )
    plain = PanelMultiState.fit(
        s=["healthy", "ill", "dead", "healthy", "ill", "ill"], **d
    )
    coded = PanelMultiState.fit(
        s=["healthy", "ill", "dead", "healthy", "ill", "sick"],
        censored={"sick": "ill"},
        **d,
    )
    assert_allclose(coded.params, plain.params, rtol=1e-8)
    assert_allclose(coded.log_likelihood, plain.log_likelihood, rtol=1e-12)


# -- serialisation and the rest -----------------------------------------------


@pytest.mark.parametrize("with_Z", [False, True])
def test_round_trip_predicts_and_bounds_the_same(with_Z):
    cls = PanelMultiStateRegression if with_Z else PanelMultiState
    d = panel_data(with_Z)
    with warnings.catch_warnings():
        # (a period effect of so few visits has no finite maximum)
        warnings.simplefilter("ignore")
        model = cls.fit(**d, pci=[2.5])
    back = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert type(back) is cls
    args = ([1.0, 3.0],) + (([0.5],) if with_Z else ())
    assert_allclose(
        back.occupancy(*args, "ill"), model.occupancy(*args, "ill")
    )
    assert_allclose(back.cb(*args, "ill"), model.cb(*args, "ill"))
    assert_allclose(back.params, model.params)
    assert_allclose(back.covariance(), model.covariance())
    assert back.maximum == model.maximum
    assert back.log_likelihood == model.log_likelihood
    assert repr(back) == repr(model)


def test_rescaling_time_rescales_the_intensities(illness_data):
    x, s, i, _ = illness_data
    a = PanelMultiState.fit(x, s, i, transitions=ILLNESS)
    b = PanelMultiState.fit(x * 24.0, s, i, transitions=ILLNESS)
    assert_allclose(b.params, a.params / 24.0, rtol=1e-6)
    assert_allclose(
        b.occupancy(48.0, "ill"), a.occupancy(2.0, "ill"), rtol=1e-6
    )


def test_summary_and_param_cb(illness_data):
    x, s, i, _ = illness_data
    model = PanelMultiState.fit(x, s, i, transitions=ILLNESS)
    table = model.summary()
    assert list(table.index) == model.parameter_names
    lo, hi = model.param_cb(model.parameter_names[0], alpha_ci=0.1)
    q, se = model.params[0], model.standard_errors()[0]
    z = 1.6448536269514722
    assert_allclose([lo, hi], q * np.exp(np.array([-z, z]) * se / q))
    assert model.param_cb(
        model.parameter_names[0], 0.05, "lower"
    ) == pytest.approx(model.param_cb(model.parameter_names[0], 0.1)[0])


def test_data_frame_paths_match():
    import pandas as pd

    d = panel_data(True)
    df = pd.DataFrame(
        {"t": d["x"], "st": d["s"], "unit": d["i"], "z": d["Z"][:, 0]}
    )
    opts = dict(
        transitions=d["transitions"],
        censored=d["censored"],
        exact_entry=d["exact_entry"],
    )
    a = PanelMultiStateRegression.fit(**d)
    b = PanelMultiStateRegression.fit_from_df(
        df, "t", "z", "st", "unit", **opts
    )
    assert b.covariate_names == ["z"]
    assert_allclose(a.params, b.params, rtol=1e-9)


def test_engine_pieces_split_at_cuts():
    owner, pos, dt, period = pm._pieces(
        np.array([0.0, 1.0, 2.5]),
        np.array([1.0, 3.5, 2.9]),
        np.array([2.0, 3.0]),
    )
    assert owner.tolist() == [0, 1, 1, 1, 2]
    assert pos.tolist() == [0, 0, 1, 2, 0]
    assert_allclose(dt, [1.0, 1.0, 1.0, 0.5, 0.4])
    assert period.tolist() == [0, 0, 1, 2, 1]


def test_transitions_as_a_matrix_object():
    tm = TransitionMatrix(ILLNESS, states=["healthy", "ill", "dead"])
    model = _fit(transitions=tm)
    assert model.transitions == tm
