"""Drawing the next failure from a unit's state, stepping the state on, and
rebuilding a fitted model at other parameters, for the renewal models
(``RenewalModel``) and the intensity models (``ParametricRecurrenceModel``).

- ``next_failure_qf`` inverts ``next_failure_sf`` from any state, and
  matches the closed forms where there is one (a Weibull's conditional
  quantile, an HPP's exponential, a Crow-AMSAA power law);
- ``next_failure_random`` is ``next_failure_qf`` at uniform draws;
- ``states`` / ``advance`` follow each family's restoration rule, and a
  loop of draws and advances reproduces the whole-history simulators
  (``time_terminated_simulation_data``, ``count_terminated_simulation_data``)
  in distribution;
- ``with_params`` keeps the structure, round-trips ``_estimates()``,
  refuses a wrong length, and its draws spread as the delta method says.

Every seed, sample size and tolerance was fixed before the results were
seen.
"""

import time

import numpy as np
import pytest
from scipy import stats

from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)
from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted
from surpyval.tests.conformance.registry_families import predictions

# (fitter, options, true restoration parameter), as test_unit_prediction.
FAMILIES = {
    "GRP Kijima I": (GeneralizedRenewal, {"kijima": "i"}, 0.4),
    "GRP Kijima II": (GeneralizedRenewal, {"kijima": "ii"}, 0.4),
    "G1": (GeneralizedOneRenewal, {}, 0.1),
    "ARA m=2": (ARA, {"m": 2}, 0.6),
    "ARA m=inf": (ARA, {"m": np.inf}, 0.6),
    "ARI m=1": (ARI, {"m": 1}, 0.4),
    "ARI m=inf": (ARI, {"m": np.inf}, 0.4),
}

PROBS = np.array([1e-6, 0.01, 0.3, 0.5, 0.9, 0.999, 1 - 1e-9])


def _truth(key):
    fitter, options, restoration = FAMILIES[key]
    if fitter is ARI:
        return ARI.fit_from_parameters(
            [20.0, 1.8], restoration, m=options["m"], baseline=CrowAMSAA
        )
    return fitter.fit_from_parameters([20.0, 1.8], restoration, **options)


def _some_states(model):
    """A new unit, one 30 into its life, and two with failure
    histories."""
    return model.states(
        failures=[[], [], [12.0, 25.0, 31.0], [5.0, 40.0]],
        now=[0.0, 30.0, 33.5, 70.0],
    )


# -- next_failure_qf -------------------------------------------------------


@pytest.mark.parametrize("key", FAMILIES)
def test_qf_inverts_sf_from_every_state(key):
    model = _truth(key)
    states = _some_states(model)
    q = model.next_failure_qf(PROBS, states=states)
    assert q.shape == (4, PROBS.size)
    assert np.all(np.isfinite(q)) and np.all(np.diff(q, axis=1) > 0)
    sf = np.stack(
        [
            model.next_failure_sf(q[r], states=states.take([r]))[0]
            for r in range(4)
        ]
    )
    # On the cumulative-hazard scale, so the tails are checked too.
    np.testing.assert_allclose(
        -np.log(sf), np.broadcast_to(-np.log1p(-PROBS), sf.shape), rtol=1e-6
    )


def test_qf_is_the_weibull_conditional_quantile():
    alpha, beta = 20.0, 1.8
    model = GeneralizedRenewal.fit_from_parameters([alpha, beta], 0.4)
    states = model.states(failures=[[], [10.0], [10.0, 30.0]], now=[7, 15, 40])
    # Kijima I: v = q * last failure + time since it.
    v = np.array([7.0, 0.4 * 10 + 5, 0.4 * 30 + 10])
    energy = -np.log1p(-PROBS)
    expected = (
        alpha * ((v[:, None] / alpha) ** beta + energy) ** (1 / beta)
        - v[:, None]
    )
    np.testing.assert_allclose(
        model.next_failure_qf(PROBS, states=states), expected, rtol=1e-9
    )


def test_qf_of_g1_is_on_the_next_gaps_scale():
    alpha, beta, q = 20.0, 1.8, 0.1
    model = GeneralizedOneRenewal.fit_from_parameters([alpha, beta], q)
    states = model.states(failures=[[10.0, 30.0]], now=[34.0])
    scale = (1 + q) ** 2
    s = 4.0 / scale
    energy = -np.log1p(-PROBS)
    expected = scale * (
        alpha * ((s / alpha) ** beta + energy) ** (1 / beta) - s
    )
    np.testing.assert_allclose(
        model.next_failure_qf(PROBS, states=states)[0], expected, rtol=1e-9
    )


def test_qf_of_ari_inverts_the_reduced_intensity():
    model = ARI.fit_from_parameters([20.0, 1.8], 0.4, m=1)
    states = model.states(failures=[[10.0, 30.0]], now=[34.0])
    t, R = 34.0, 0.4 * float(model.model.iif(30.0))
    x = model.next_failure_qf(PROBS, states=states)[0]
    gained = model.model.cif(t + x) - model.model.cif(t) - R * x
    np.testing.assert_allclose(gained, -np.log1p(-PROBS), rtol=1e-8)


def test_qf_ends_shapes_and_refusals():
    model = _truth("GRP Kijima I")
    assert model.next_failure_qf(0.5, age=[0.0, 6.0]).shape == (2,)
    assert model.next_failure_qf([0.5], age=[0.0, 6.0]).shape == (2, 1)
    out = model.next_failure_qf([0.0, 1.0, np.nan], age=[3.0])
    np.testing.assert_array_equal(out, [[0.0, np.inf, np.nan]])
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        assert np.isnan(model.next_failure_qf(1.5, age=[3.0])).all()
    with pytest.raises(ValueError, match="not both"):
        model.next_failure_qf(0.5, age=[1.0], states=model.states(age=[1.0]))
    with pytest.raises(TypeError, match="UnitStates"):
        model.next_failure_qf(0.5, states=[1.0])
    with pytest.raises(ValueError, match="whole number"):
        model.next_failure_random(-1, age=[1.0])
    with pytest.raises(ValueError, match="no data|fitted from data"):
        model.next_failure_qf(0.5)


@pytest.mark.parametrize("key", FAMILIES)
def test_random_is_qf_at_uniform_draws(key):
    model = _truth(key)
    states = _some_states(model)
    draws = model.next_failure_random(5, states=states, random_state=11)
    u = np.random.default_rng(11).random((4, 5))
    for r in range(4):
        np.testing.assert_allclose(
            draws[r],
            model.next_failure_qf(u[r], states=states.take([r]))[0],
            rtol=1e-12,
        )
    assert model.next_failure_random(0, states=states).shape == (4, 0)


# -- states and advance ----------------------------------------------------


@pytest.mark.parametrize("key", FAMILIES)
def test_advance_is_the_state_of_the_longer_history(key):
    model = _truth(key)
    start = model.states(failures=[[12.0, 25.0], []], now=[31.0, 8.0])
    after = model.advance(start, [4.0, 2.5])
    direct = model.states(failures=[[12.0, 25.0, 35.0], [10.5]])
    ran = model.advance(start, [4.0, 2.5], failed=False)
    ran_direct = model.states(failures=[[12.0, 25.0], []], now=[35.0, 10.5])
    for got, want in ((after, direct), (ran, ran_direct)):
        for field in ("now", "since_failure", "after_repair", "scale"):
            np.testing.assert_allclose(
                getattr(got, field), getattr(want, field), rtol=1e-12
            )
        np.testing.assert_allclose(
            got.age, want.age, rtol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(got.reduction, want.reduction, rtol=1e-12)
    # The state given is not changed.
    np.testing.assert_array_equal(start.now, [31.0, 8.0])
    mixed = model.advance(start, [4.0, 2.5], failed=[True, False])
    np.testing.assert_array_equal(mixed.failure_count, [3, 0])


def test_advance_follows_each_restoration_rule():
    # One failure at 10 of a new unit, then 5 more without one.
    def after(model):
        state = model.advance(model.states(age=[0.0]), 10.0)
        return model.advance(state, 5.0, failed=False)

    grp1 = after(GeneralizedRenewal.fit_from_parameters([20, 1.8], 0.4))
    assert grp1.age[0] == pytest.approx(0.4 * 10 + 5)
    grp2 = after(
        GeneralizedRenewal.fit_from_parameters([20, 1.8], 0.4, kijima="ii")
    )
    assert grp2.age[0] == pytest.approx(0.4 * 10 + 5)
    ara = after(ARA.fit_from_parameters([20, 1.8], 0.6, m=1))
    assert ara.age[0] == pytest.approx((1 - 0.6) * 10 + 5)
    g1 = after(GeneralizedOneRenewal.fit_from_parameters([20, 1.8], 0.1))
    assert g1.scale[0] == pytest.approx(1.1)
    assert g1.age[0] == pytest.approx(5 / 1.1)
    ari_model = ARI.fit_from_parameters([20, 1.8], 0.4, m=1)
    ari = after(ari_model)
    assert ari.reduction[0] == pytest.approx(0.4 * ari_model.model.iif(10.0))


def test_states_default_is_the_fitted_items_and_refusals():
    truth = _truth("ARA m=2")
    sim = truth.time_terminated_simulation_data(80.0, items=6, random_state=3)
    model = ARA.fit(sim.x, sim.i, sim.c, m=2)
    data = model.data
    failures, now = [], []
    for item in np.unique(data.i):
        mask = data.i == item
        failures.append(data.x[mask][data.c[mask] == 0])
        now.append(data.x[mask].max())
    given = model.states(failures=failures, now=now)
    np.testing.assert_allclose(given.age, model.states().age, rtol=1e-12)
    np.testing.assert_allclose(
        model.next_failure_qf([0.1, 0.9], states=given),
        model.next_failure_qf([0.1, 0.9]),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="not both"):
        model.states(age=[1.0], failures=[[]])
    with pytest.raises(ValueError, match="now goes with failures"):
        model.states(now=[1.0])
    with pytest.raises(ValueError, match="last failure"):
        model.states(failures=[[5.0]], now=[4.0])
    with pytest.raises(ValueError, match="finite"):
        model.states(failures=[[np.inf]])
    with pytest.raises(ValueError, match="finite"):
        model.advance(given, np.inf)
    with pytest.raises(ValueError, match="negative"):
        model.advance(given, -1.0)


# -- a loop of draws and advances is the whole-history simulation ----------

N_LOOP = 2000
HORIZON = 60.0


def _loop_until(model, horizon, rng):
    """Every event time of N_LOOP new units up to ``horizon``, unit by
    unit, by drawing each next failure and advancing the state."""
    states = model.states(age=np.zeros(N_LOOP))
    events = [[] for _ in range(N_LOOP)]
    running = np.arange(N_LOOP)
    while running.size:
        part = states.take(running)
        gap = model.next_failure_random(1, states=part, random_state=rng)[:, 0]
        nxt = part.now + gap
        inside = nxt <= horizon
        for r, t in zip(running[inside], nxt[inside]):
            events[r].append(t)
        states = _put(
            states,
            running[inside],
            model.advance(part, gap).take(np.flatnonzero(inside)),
        )
        running = running[inside]
    return events


def _put(states, rows, new):
    """``states`` with the units ``rows`` replaced by ``new``."""
    out = states.take(np.arange(states.now.size))
    for k, r in enumerate(rows):
        out.failures[r] = new.failures[k]
    for field in (
        "now",
        "since_failure",
        "after_repair",
        "age",
        "scale",
        "reduction",
    ):
        getattr(out, field)[rows] = getattr(new, field)
    return out


def _simulated_events(data):
    x, i, c = np.asarray(data.x), np.asarray(data.i), np.asarray(data.c)
    return [np.sort(x[(i == u) & (c == 0)]) for u in np.unique(i)]


@pytest.mark.parametrize("key", FAMILIES)
def test_loop_reproduces_time_terminated_simulation(key):
    model = _truth(key)
    looped = _loop_until(model, HORIZON, np.random.default_rng(1001))
    sim = model.time_terminated_simulation_data(
        HORIZON, items=N_LOOP, random_state=2002
    )
    whole = _simulated_events(sim)
    counts_a = np.array([len(e) for e in looped])
    counts_b = np.array([len(e) for e in whole])
    # Counts: a chi-squared test of the two count distributions.
    top = int(max(counts_a.max(), counts_b.max()))
    table = np.array(
        [
            np.bincount(counts_a, minlength=top + 1),
            np.bincount(counts_b, minlength=top + 1),
        ]
    )
    table = table[:, table.sum(axis=0) >= 10]
    assert stats.chi2_contingency(table).pvalue > 1e-3
    # The first, second and third failure times.
    for k in range(3):
        a = [e[k] for e in looped if len(e) > k]
        b = [e[k] for e in whole if len(e) > k]
        assert stats.ks_2samp(a, b).pvalue > 1e-3, k


@pytest.mark.parametrize(
    "key", ["GRP Kijima II", "G1", "ARA m=inf", "ARI m=1"]
)
def test_loop_reproduces_count_terminated_simulation(key):
    model = _truth(key)
    events = 3
    sim = model.count_terminated_simulation_data(
        events, items=N_LOOP, random_state=3003
    )
    whole = _simulated_events(sim)
    states = model.states(age=np.zeros(N_LOOP))
    rng = np.random.default_rng(4004)
    times = []
    for _ in range(events + 1):
        gap = model.next_failure_random(1, states=states, random_state=rng)
        states = model.advance(states, gap[:, 0])
        times.append(states.now.copy())
    for k in range(events + 1):
        b = [e[k] for e in whole]
        assert stats.ks_2samp(times[k], b).pvalue > 1e-3, k


# -- the intensity models ---------------------------------------------------


def test_hpp_next_failure_is_exponential_whatever_the_age():
    model = HPP.from_params([0.1])
    ages = np.array([0.0, 1e4])
    q = model.next_failure_qf(PROBS, age=ages)
    exact = np.broadcast_to(-np.log1p(-PROBS) / 0.1, q.shape)
    # To the documented accuracy: the double's rounding of the age.
    np.testing.assert_allclose(
        q, exact, rtol=1e-9, atol=4 * np.finfo(float).eps * ages.max()
    )
    np.testing.assert_allclose(q[0], exact[0], rtol=1e-12)
    draws = model.next_failure_random(4000, age=[250.0], random_state=5)[0]
    assert stats.kstest(draws, "expon", args=(0, 10.0)).pvalue > 1e-3


def test_crow_amsaa_next_failure_is_the_power_law():
    alpha, beta = 10.0, 1.5
    model = CrowAMSAA.from_params([alpha, beta])
    t = np.array([0.0, 10.0, 200.0])
    energy = -np.log1p(-PROBS)
    expected = (
        alpha * ((t[:, None] / alpha) ** beta + energy) ** (1 / beta)
        - t[:, None]
    )
    q = model.next_failure_qf(PROBS, age=t)
    np.testing.assert_allclose(q, expected, rtol=1e-9)
    np.testing.assert_allclose(
        -np.log(
            np.stack(
                [model.next_failure_sf(q[r], age=[t[r]])[0] for r in range(3)]
            )
        ),
        np.broadcast_to(energy, q.shape),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        model.next_failure_hf([0.0, 5.0], age=[10.0]),
        [[model.iif(10.0), model.iif(15.0)]],
    )


@pytest.mark.parametrize("name", ["HPP", "CrowAMSAA", "Duane", "CoxLewis"])
def test_intensity_loop_reproduces_time_terminated_simulation(name):
    model = fitted(CASE_BY_NAME[name])
    horizon = 40.0
    age = np.zeros(N_LOOP)
    rng = np.random.default_rng(5005)
    looped = [[] for _ in range(N_LOOP)]
    running = np.arange(N_LOOP)
    while running.size:
        gap = model.next_failure_random(1, age=age[running], random_state=rng)[
            :, 0
        ]
        nxt = age[running] + gap
        inside = nxt <= horizon
        for r, t in zip(running[inside], nxt[inside]):
            looped[r].append(t)
        age[running[inside]] = nxt[inside]
        running = running[inside]
    sim = model.time_terminated_simulation_data(
        horizon, items=N_LOOP, random_state=6006
    )
    whole = _simulated_events(sim)
    a = np.array([len(e) for e in looped])
    b = np.array([len(e) for e in whole])
    top = int(max(a.max(), b.max()))
    table = np.array(
        [np.bincount(a, minlength=top + 1), np.bincount(b, minlength=top + 1)]
    )
    table = table[:, table.sum(axis=0) >= 10]
    assert stats.chi2_contingency(table).pvalue > 1e-3
    first_a = [e[0] for e in looped if e]
    first_b = [e[0] for e in whole if len(e)]
    assert stats.ks_2samp(first_a, first_b).pvalue > 1e-3


def test_cox_lewis_falling_intensity_may_never_fail_again():
    model = CoxLewis.from_params([np.log(0.1), -0.05])
    # cif(inf) = 0.1 / 0.05 = 2; at age 0 the chance of no event ever is
    # exp(-2), so a p above 1 - exp(-2) is never reached.
    q = model.next_failure_qf([0.5, 0.9], age=[0.0])
    assert np.isfinite(q[0, 0]) and q[0, 1] == np.inf


def test_intensity_default_units_are_the_fitted_items():
    model = fitted(CASE_BY_NAME["CrowAMSAA"])
    data = model.data
    ends = [data.x[data.i == u].max() for u in np.unique(data.i)]
    np.testing.assert_allclose(
        model.next_failure_sf([1.0, 5.0]),
        model.next_failure_sf([1.0, 5.0], age=ends),
    )


# -- with_params ------------------------------------------------------------

RENEWAL_CASES = [
    "GeneralizedRenewal",
    "GeneralizedRenewal[kijima ii]",
    "GeneralizedOneRenewal",
    "ARA",
    "ARI",
]
INTENSITY_CASES = ["HPP", "CrowAMSAA", "Duane", "CoxLewis"]


@pytest.mark.parametrize("name", RENEWAL_CASES + INTENSITY_CASES)
def test_with_params_of_the_estimates_reproduces_the_registry_predictions(
    name,
):
    case = CASE_BY_NAME[name]
    model = fitted(case)
    rebuilt = model.with_params(model._estimates())
    assert type(rebuilt) is type(model)
    assert rebuilt.covariance_names == model.covariance_names
    assert len(model.covariance_names) == model.covariance().shape[0]
    want, got = predictions(case, model), predictions(case, rebuilt)
    assert want.keys() == got.keys()
    for key in want:
        np.testing.assert_allclose(got[key], want[key], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        rebuilt.next_failure_qf([0.1, 0.5, 0.9], age=[0.0, 7.0]),
        model.next_failure_qf([0.1, 0.5, 0.9], age=[0.0, 7.0]),
        rtol=1e-12,
    )
    # A model built from parameters: no data, no likelihood.
    assert rebuilt.how == "from_params"
    with pytest.raises(ValueError, match="Likelihood inference"):
        rebuilt.covariance()
    k = len(model.covariance_names)
    for wrong in (np.r_[model._estimates(), 1.0], model._estimates()[:-1]):
        with pytest.raises(ValueError, match=f"{k} values"):
            model.with_params(wrong)
    with pytest.raises(ValueError, match="covariance_names"):
        model.with_params(np.atleast_2d(model._estimates()))


def test_renewal_with_params_keeps_the_structure():
    model = ARI.fit_from_parameters([1.0, 1.5], 0.6, m=3, baseline=Duane)
    other = model.with_params([0.3, 1.2, 1.4])
    assert other.m == 3 and other.model.dist is Duane
    assert other.rho == 0.3 and list(other.model.params) == [1.2, 1.4]
    grp = GeneralizedRenewal.fit_from_parameters([10, 2], 0.5, kijima="ii")
    assert grp.with_params([0.1, 9, 2]).kijima_type == "ii"
    with pytest.raises(ValueError, match="q"):
        grp.with_params([-0.1, 9, 2])
    ara = ARA.fit_from_parameters([10, 2], 0.5, m=np.inf)
    assert ara.with_params([0.2, 9, 2]).m == np.inf
    with pytest.raises(ValueError, match="rho"):
        ara.with_params([1.2, 9, 2])
    fit = fitted(CASE_BY_NAME["GeneralizedRenewal"])
    fit_unit = GeneralizedRenewal.fit(
        fit.data.x, fit.data.i, fit.data.c, unit="hours"
    )
    assert fit_unit.with_params(fit_unit._estimates()).unit == "hour"


def test_intensity_with_params_refuses_outside_the_range():
    model = CrowAMSAA.from_params([10, 1.5])
    with pytest.raises(ValueError, match="beta = -0.5 is outside"):
        model.with_params([10, -0.5])
    with pytest.raises(ValueError, match="lambda = -0.1 is outside"):
        HPP.from_params([0.1]).with_params([-0.1])
    assert CoxLewis.from_params([0.1, -0.2]).with_params([0.2, 0.3]).params[
        1
    ] == pytest.approx(0.3)


def test_intensity_with_params_draws_agree_with_the_delta_method():
    truth = CrowAMSAA.from_params([5.0, 1.4])
    sim = truth.time_terminated_simulation_data(100.0, items=8, random_state=7)
    model = CrowAMSAA.fit(sim.x, sim.i, sim.c)
    t = np.array([20.0, 100.0, 300.0])
    draws = np.random.default_rng(8).multivariate_normal(
        model._estimates(), model.covariance(), size=4000
    )
    cif = np.array([model.with_params(d).cif(t) for d in draws])
    lo, hi = model.cif_cb(t, alpha_ci=0.05).T
    z = stats.norm.ppf(0.975)
    # cif_cb is a log-transformed Wald bound: its log-scale SE.
    se_log = np.log(hi / lo) / (2 * z)
    se_delta = model.cif(t) * se_log
    np.testing.assert_allclose(cif.std(axis=0), se_delta, rtol=0.1)


def test_renewal_with_params_draws_agree_with_the_delta_method():
    truth = GeneralizedRenewal.fit_from_parameters([20.0, 1.8], 0.4)
    sim = truth.time_terminated_simulation_data(
        150.0, items=25, random_state=9
    )
    model = GeneralizedRenewal.fit(sim.x, sim.i, sim.c)
    theta, cov = model._estimates(), model.covariance()
    # q is 3.7 standard errors from its edge at 0: the few draws below it
    # (about 1 in 10,000) are left out.
    age = [0.0, 30.0]

    def sf(params):
        return model.with_params(params).next_failure_sf(10.0, age=age)

    # The delta method by central differences.
    step = 1e-5 * np.maximum(np.abs(theta), 1.0)
    grad = np.array(
        [
            (
                sf(theta + np.eye(3)[j] * step[j])
                - sf(theta - np.eye(3)[j] * step[j])
            )
            / (2 * step[j])
            for j in range(3)
        ]
    )
    se_delta = np.sqrt(np.einsum("ji,jk,ki->i", grad, cov, grad))
    draws = np.random.default_rng(10).multivariate_normal(theta, cov, 4000)
    draws = draws[draws[:, 0] >= 0]
    assert draws.shape[0] > 3990
    spread = np.array([sf(d) for d in draws]).std(axis=0)
    np.testing.assert_allclose(spread, se_delta, rtol=0.1)


def test_with_params_of_a_refit_gives_the_refit_predictions():
    truth = _truth("ARA m=2")
    sim = truth.time_terminated_simulation_data(
        80.0, items=10, random_state=12
    )
    refit = ARA.fit(sim.x, sim.i, sim.c, m=2)
    rebuilt = truth.with_params(refit._estimates())
    for age in ([0.0], [15.0, 40.0]):
        np.testing.assert_allclose(
            rebuilt.next_failure_sf([1.0, 10.0], age=age),
            refit.next_failure_sf([1.0, 10.0], age=age),
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        rebuilt.mcf([10.0, 50.0], items=50, random_state=1),
        refit.mcf([10.0, 50.0], items=50, random_state=1),
        rtol=1e-12,
    )


def test_a_draw_is_cheap(capsys):
    # The timing for the record (not asserted beyond a generous ceiling):
    # one draw for each of 1000 units, and one advance of them.
    for key in ("GRP Kijima I", "ARA m=2", "ARI m=1"):
        model = _truth(key)
        states = model.states(age=np.full(1000, 10.0))
        start = time.perf_counter()
        gap = model.next_failure_random(1, states=states, random_state=1)
        drawn = time.perf_counter()
        model.advance(states, gap[:, 0])
        done = time.perf_counter()
        with capsys.disabled():
            print(
                f"\n{key}: draw {1e3 * (drawn - start):.2f} ms, advance "
                f"{1e3 * (done - drawn):.2f} ms for 1000 units"
            )
        assert done - start < 10.0
