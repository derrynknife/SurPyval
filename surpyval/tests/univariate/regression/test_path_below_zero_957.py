"""#957: on a baseline defined below 0 (Normal, Gumbel, Logistic) a
CovariatePath is integrated below 0 too: back to each time asked about
before 0, and, for proportional hazards and odds, back to -inf."""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import surpyval as sp
from surpyval import CovariatePath, StepSchedule


def _fit(name):
    rng = np.random.default_rng(1)
    Z = rng.binomial(1, 0.5, (300, 1)).astype(float)
    x = rng.normal(3, 2, 300) - 0.8 * Z[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, name).fit(x, Z)


def _sine(u):
    return 0.5 + 0.5 * np.sin(np.asarray(u, dtype=float) / 2)


SINE = CovariatePath.from_callable(_sine)
TIMES = np.array([-2.0, 0.0, 1.5, 4.0])


def _hazard(m, u, z):
    return float(np.ravel(m.hf(np.array([u]), [z]))[0])


def _quad(f, a, b):
    return quad(f, a, b, epsabs=0, epsrel=1e-12, limit=500)[0]


@pytest.mark.parametrize("name", ["NormalPH", "LogisticPO", "GumbelPH"])
def test_ph_and_po_integrate_a_formula_from_minus_inf(name):
    m = _fit(name)
    ref = [
        np.exp(-_quad(lambda u: _hazard(m, u, _sine(u)), -np.inf, t))
        for t in TIMES
    ]
    np.testing.assert_allclose(m.sf_tvc(TIMES, SINE), ref, rtol=1e-10)


def test_additive_hazards_integrate_back_to_a_time_before_zero():
    m = _fit("NormalAH")
    H0 = float(np.ravel(m.Hf(np.array([0.0]), [0.0]))[0])
    ref = [
        np.exp(-(H0 + _quad(lambda u: _hazard(m, u, _sine(u)), 0, t)))
        for t in TIMES
    ]
    np.testing.assert_allclose(m.sf_tvc(TIMES, SINE), ref, rtol=1e-10)


def test_aft_ages_back_to_a_time_before_zero():
    m = _fit("NormalAFT")
    params, center = m._eval_params(), m.center

    def rate(u):
        z = m._centred(np.array([[_sine(u)]]), center)
        return float(m._tvc_rate(z, params)[0])

    psi = np.array([_quad(rate, 0, t) for t in TIMES])
    ref = np.exp(-m._aft_H0(psi))
    np.testing.assert_allclose(m.sf_tvc(TIMES, SINE), ref, rtol=1e-10)


@pytest.mark.parametrize(
    "name", ["NormalPH", "LogisticPO", "NormalAH", "NormalAFT"]
)
def test_given_an_age_before_zero(name):
    m = _fit(name)
    x = np.array([-0.5, 1.0, 3.0])
    np.testing.assert_allclose(
        m.sf_tvc(x, SINE, given=-1.0),
        m.sf_tvc(x, SINE) / m.sf_tvc(-1.0, SINE),
        rtol=1e-9,
    )


@pytest.mark.parametrize("name", ["NormalPH", "NormalAFT", "NormalAH"])
def test_a_constant_formula_gives_cb(name):
    m = _fit(name)
    flat = CovariatePath.from_callable(lambda u: 0 * np.asarray(u) + 1.0)
    np.testing.assert_allclose(
        m.cb_tvc(TIMES, flat), m.cb(TIMES, [1.0]), rtol=1e-6
    )


@pytest.mark.parametrize("name", ["NormalPH", "LogisticPO", "NormalAFT"])
def test_the_mean_along_a_formula(name):
    m = _fit(name)

    def sf(t):
        return float(m.sf_tvc([t], SINE)[0])

    ref = _quad(sf, 0, np.inf) - _quad(lambda t: 1 - sf(t), -np.inf, 0)
    np.testing.assert_allclose(m.mean_tvc(SINE), ref, rtol=1e-8)


def test_a_path_through_points_starts_at_its_first_point():
    late = CovariatePath.from_points([-3, 5], [0.0, 1.0])
    aft = _fit("NormalAFT")
    assert np.isfinite(aft.sf_tvc([-2.0, 1.0], late)).all()
    with pytest.raises(ValueError, match="starts at -3"):
        aft.sf_tvc([-4.0], late)
    ph = _fit("NormalPH")
    with pytest.raises(ValueError, match="back to -inf"):
        ph.sf_tvc([1.0], late)
    # Given an age at or after its start, it is enough.
    assert np.isfinite(ph.sf_tvc([1.0], late, given=-2.0)).all()


def test_a_hazard_that_does_not_die_away_going_back_is_refused():
    m = _fit("LogisticPH")
    beta, sigma = m.params[-1], m.params[1]
    k = np.sign(beta) * 3 / (abs(beta) * sigma)
    grows = CovariatePath.from_callable(lambda u: -k * np.asarray(u))
    with pytest.raises(ValueError, match="does not die away"):
        m.sf_tvc([1.0], grows)


def test_a_schedule_and_its_path_agree_before_zero():
    m = _fit("NormalPH")
    steps = StepSchedule.from_changepoints(
        [-np.inf, -2.0, 1.0], [[1.0], [0.0], [1.0]]
    )
    jumps = CovariatePath.from_callable(
        lambda u: np.where(
            (np.asarray(u) > -2) & (np.asarray(u) <= 1), 0.0, 1.0
        ),
        breakpoints=[-2.0, 1.0],
    )
    np.testing.assert_allclose(
        m.sf_tvc(TIMES, jumps), m.sf_tvc(TIMES, steps), rtol=1e-10
    )
