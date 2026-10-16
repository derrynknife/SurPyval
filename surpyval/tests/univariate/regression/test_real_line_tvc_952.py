"""#952: on a baseline defined below 0 (Normal, Gumbel, Logistic) a
covariate path must cover every time the answer depends on: back to -inf
for proportional hazards and odds, from 0 (and any earlier time asked
about) for accelerated failure time and additive hazards, which measure
the covariate's effect from 0. In a fit, a first row at -inf is followed
since -inf and one at 0 entered at 0."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    CoxPH,
    CovariatePath,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)

INF = np.inf


def _data(seed=1, n=300):
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, (n, 1)).astype(float)
    x = rng.normal(3, 2, n) - 0.8 * Z[:, 0]
    return x, Z


def _fit(name):
    x, Z = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, name).fit(x, Z)


def _sf(m, t, z):
    return float(np.ravel(m.sf(np.array([t]), [z]))[0])


def _Hf(m, t, z):
    return float(np.ravel(m.Hf(np.array([t]), [z]))[0])


# Z = 1 since -inf, 0 on [-5, 0), 1 from 0.
BEFORE = StepSchedule.from_changepoints([-INF, -5, 0], [[1.0], [0.0], [1.0]])


@pytest.mark.parametrize("name", ["NormalPH", "NormalPO", "GumbelPH"])
def test_ph_and_po_need_the_covariate_back_to_minus_inf(name):
    m = _fit(name)
    with pytest.raises(ValueError, match="back to -inf"):
        m.sf_tvc([2.0], StepSchedule.from_changepoints([0], [[1.0]]))
    with pytest.raises(ValueError, match="back to -inf"):
        m.sf_tvc([2.0], [[1.0]], xl=[-5.0])
    # What it is before 0 counts: H is the sum over the segments from -inf.
    for t in (-1.0, 2.0, 4.0):
        H = _Hf(m, -5, 1) + _Hf(m, min(t, 0), 0) - _Hf(m, -5, 0)
        if t > 0:
            H += _Hf(m, t, 1) - _Hf(m, 0, 1)
        np.testing.assert_allclose(m.sf_tvc([t], BEFORE), np.exp(-H))
    const = StepSchedule.constant([1.0])
    np.testing.assert_allclose(
        m.sf_tvc([-1.0, 2.0], const), m.sf(np.array([-1.0, 2.0]), [1.0])
    )


@pytest.mark.parametrize("name", ["NormalAFT", "NormalAH", "LogisticAFT"])
def test_aft_and_ah_measure_the_effect_from_zero(name):
    m = _fit(name)
    from_0 = StepSchedule.from_changepoints([0], [[1.0]])
    # From 0 is enough for times from 0: what it was before 0 is not used.
    np.testing.assert_allclose(
        m.sf_tvc([2.0, 4.0], from_0), m.sf(np.array([2.0, 4.0]), [1.0])
    )
    np.testing.assert_allclose(
        m.sf_tvc([2.0, 4.0], BEFORE), m.sf(np.array([2.0, 4.0]), [1.0])
    )
    # A time before 0 needs the covariate between it and 0.
    with pytest.raises(ValueError, match="earliest time asked about"):
        m.sf_tvc([-1.0], from_0)
    np.testing.assert_allclose(m.sf_tvc([-1.0], BEFORE), _sf(m, -1.0, 0.0))


@pytest.mark.parametrize("name", ["NormalPH", "NormalPO"])
def test_a_path_serves_ph_and_po_given_an_age(name):
    m = _fit(name)
    ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
    with pytest.raises(ValueError, match="back to -inf"):
        m.sf_tvc([2.0], ramp)
    # A formula has a value at every time: it is integrated from -inf
    # (#957).
    np.testing.assert_allclose(
        m.sf_tvc([2.0], CovariatePath.from_callable(lambda t: 0 * t + 1.0)),
        m.sf(np.array([2.0]), [1.0]),
        rtol=1e-9,
    )
    # Given an age from 0, only the path after it counts.
    got = m.sf_tvc([4.0, 6.0], ramp, given=2.0)
    steps = StepSchedule.from_changepoints([-INF, 2.0], [[0.0], [0.2]])
    flat = CovariatePath.from_points([0, 2, 2], [0.0, 0.0, 0.2])
    np.testing.assert_allclose(
        m.sf_tvc([4.0, 6.0], flat, given=2.0),
        m.sf_tvc([4.0, 6.0], steps, given=2.0),
    )
    assert np.all((got > 0) & (got < 1))


@pytest.mark.parametrize("name", ["NormalAFT", "NormalAH"])
def test_a_path_serves_aft_and_ah_from_zero(name):
    m = _fit(name)
    ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
    assert np.all(np.isfinite(m.sf_tvc([0.0, 2.0], ramp)))
    with pytest.raises(ValueError, match="before 0"):
        m.sf_tvc([-1.0, 2.0], ramp)
    # At and before an age given, survival is certain whatever the path.
    np.testing.assert_array_equal(
        m.sf_tvc([-1.0, 1.0], ramp, given=1.0), [1.0, 1.0]
    )


def test_the_mean_needs_the_covariate_back_to_minus_inf():
    m = _fit("NormalPH")
    const = StepSchedule.constant([1.0])
    np.testing.assert_allclose(m.mean_tvc(const), np.ravel(m.mean([1.0]))[0])
    with pytest.raises(ValueError, match="back to -inf"):
        m.mean_tvc(StepSchedule.from_changepoints([0], [[1.0]]))
    # Along a schedule that changes before 0, summed from -inf (#957).
    sf = lambda t: float(m.sf_tvc([t], BEFORE)[0])  # noqa: E731
    from scipy.integrate import quad

    ref = (
        quad(sf, 0, np.inf, epsrel=1e-11, limit=500)[0]
        - quad(lambda t: 1 - sf(t), -np.inf, 0, epsrel=1e-11, limit=500)[0]
    )
    np.testing.assert_allclose(m.mean_tvc(BEFORE), ref, rtol=1e-8)
    with pytest.raises(ValueError, match="not provided"):
        _fit("NormalAH").mean_tvc(BEFORE)
    ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
    with pytest.raises(ValueError, match="back to -inf"):
        m.mean_tvc(ramp)
    assert np.isfinite(m.mean_tvc(ramp, given=2.0))


def _rows(n=200, seed=2):
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, (n, 1)).astype(float)
    x = rng.normal(4, 2, n) - 0.8 * Z[:, 0]
    x = np.where(x <= 0.2, 0.2 + rng.uniform(0, 1, n), x)
    return np.arange(n), x, Z


@pytest.mark.parametrize(
    "name", ["NormalPH", "NormalPO", "NormalAH", "NormalAFT", "WeibullPH"]
)
def test_a_first_row_at_minus_inf_is_followed_since_then(name):
    F = getattr(sp, name)
    i, x, Z = _rows()
    c = np.zeros(len(x), int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = F.fit(x, Z)
        since = F.fit_tvc(i, np.full(len(x), -INF), x, c, Z)
        at_0 = F.fit_tvc(i, np.zeros(len(x)), x, c, Z)
        truncated = F.fit(x, Z, tl=np.zeros(len(x)))
    np.testing.assert_allclose(since.params, plain.params, rtol=1e-3)
    # A first row at 0 entered at 0: survival to 0 is conditioned on.
    np.testing.assert_allclose(at_0.params, truncated.params, rtol=1e-3)
    if name != "WeibullPH":
        assert not np.allclose(at_0.params, since.params, rtol=1e-6)


def test_only_a_first_row_is_followed_since_minus_inf():
    with pytest.raises(ValueError, match="only a subject's first row"):
        sp.NormalPH.fit_tvc(
            [1, 1], [-INF, -INF], [1.0, 2.0], [1, 0], [[0.0], [1.0]]
        )


def test_additive_hazards_need_one_value_across_zero_from_minus_inf():
    i = np.array([1, 1, 2])
    xl = np.array([-INF, -1.0, -INF])
    xr = np.array([-1.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="one covariate value across"):
        sp.NormalAH.fit_tvc(i, xl, xr, [1, 0, 0], [[0.0], [1.0], [1.0]])


def test_aft_refuses_an_exit_before_zero_on_a_real_line_baseline():
    with pytest.raises(ValueError, match="leaves observation"):
        sp.NormalAFT.fit_tvc(
            [1, 2], [-INF, 0.0], [-1.0, 3.0], [0, 0], [[0.0], [1.0]]
        )


def _items(left_truncation=None, start=-INF):
    i, x, Z = _rows(80)
    return (
        SurpyvalData.from_items(
            [
                ItemLifetime(
                    k,
                    observed=float(t),
                    left_truncation=left_truncation,
                    time_varying_covariates={start: [float(z[0])]},
                )
                for k, t, z in zip(i, x, Z)
            ]
        ),
        x,
        Z,
    )


def test_records_are_followed_since_minus_inf_on_a_real_line_baseline():
    data, x, Z = _items()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.NormalPH.fit(data)
        plain = sp.NormalPH.fit(x, Z)
    np.testing.assert_allclose(model.params, plain.params, rtol=1e-3)
    late, _, _ = _items(start=0.0)
    with pytest.raises(ValueError, match="no left_truncation"):
        sp.NormalPH.fit(late)
    # Under observation from 0, as it says.
    entered, _, _ = _items(left_truncation=0.0, start=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.NormalPH.fit(entered)
        truncated = sp.NormalPH.fit(x, Z, tl=np.zeros(len(x)))
    np.testing.assert_allclose(model.params, truncated.params, rtol=1e-3)
    # On a baseline that starts at 0, from 0 is enough.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.WeibullPH.fit(_items(start=0.0)[0])


def test_cox_needs_the_covariate_before_a_jump_below_zero():
    x, Z = _data(3, 120)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CoxPH.fit(x, Z)
    first = float(np.min(x))
    assert first < 0
    with pytest.raises(ValueError, match="before it"):
        m.sf_tvc([2.0], StepSchedule.from_changepoints([0], [[1.0]]))
    np.testing.assert_allclose(
        m.sf_tvc([2.0], StepSchedule.constant([1.0])),
        m.sf(np.array([2.0]), [1.0]),
    )
    # The value before each jump counts.
    lower = StepSchedule.from_changepoints([-INF, 0], [[0.0], [1.0]])
    assert (
        m.sf_tvc([2.0], lower)[0]
        != m.sf_tvc([2.0], StepSchedule.constant([1.0]))[0]
    )
    # Conditioned on an age past the early jumps, from 0 is enough.
    assert np.isfinite(
        m.sf_tvc(
            [2.0], CovariatePath.from_points([0, 5], [0.0, 1.0]), given=1.0
        )
    ).all()


def test_a_schedule_can_be_held_since_minus_inf():
    s = StepSchedule.constant([2.0])
    assert s.start == -INF and repr(s) == "StepSchedule.constant([2.0])"
    s = StepSchedule.from_changepoints([-INF, 5], [[0.0], [1.0]])
    assert repr(s) == (
        "StepSchedule.from_changepoints([-np.inf, 5.0], [[0.0], [1.0]])"
    )
    assert eval(repr(s)) == s
    item = ItemLifetime(
        "a", observed=3.0, time_varying_covariates={-INF: [0.0], 1: [1.0]}
    )
    assert item.time_varying_covariates == StepSchedule.from_changepoints(
        [-INF, 1], [[0.0], [1.0]]
    )
    with pytest.raises(ValueError, match="first and last"):
        StepSchedule([-INF, 1.0], [[0.0]], period=2.0)
