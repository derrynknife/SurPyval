import numpy as np
import pytest
from scipy.integrate import quad

from surpyval.recurrent import CoxLewis


@pytest.mark.parametrize(
    "alpha,beta", [(0.5, 0.2), (-1.0, 0.5), (1.0, -0.1), (0.0, 1.0)]
)
def test_cif_is_integral_of_iif(alpha, beta):
    # The cumulative intensity must be the integral of the instantaneous
    # intensity from 0 to x, i.e. cif(x) == \int_0^x iif(s) ds.
    for x in [0.5, 1.0, 3.0, 7.0]:
        expected, _ = quad(lambda s: CoxLewis.iif(s, alpha, beta), 0.0, x)
        assert np.isclose(CoxLewis.cif(x, alpha, beta), expected)


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.2), (-1.0, 0.5), (0.0, 1.0)])
def test_cif_is_zero_at_origin(alpha, beta):
    # A cumulative intensity (expected count) must start at zero.
    assert np.isclose(CoxLewis.cif(0.0, alpha, beta), 0.0)


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.2), (-1.0, 0.5), (0.0, 1.0)])
def test_iif_is_log_linear_intensity(alpha, beta):
    # The Cox-Lewis model is defined by a log-linear intensity.
    x = np.array([0.0, 1.0, 2.5, 5.0])
    assert np.allclose(CoxLewis.iif(x, alpha, beta), np.exp(alpha + beta * x))
    assert np.allclose(CoxLewis.log_iif(x, alpha, beta), alpha + beta * x)


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.2), (-1.0, 0.5), (0.0, 1.0)])
def test_inv_cif_inverts_cif(alpha, beta):
    N = np.array([0.5, 1.0, 4.0, 10.0])
    x = CoxLewis.inv_cif(N, alpha, beta)
    assert np.allclose(CoxLewis.cif(x, alpha, beta), N)


def test_inv_cif_beyond_asymptote_is_inf():
    # For an improving system (beta < 0) the cumulative intensity is bounded
    # above by exp(alpha) / -beta; counts at or beyond that asymptote are
    # never reached, so their inverse must be inf -- not NaN or a negative
    # time.
    alpha, beta = 0.0, -0.5
    asymptote = np.exp(alpha) / -beta  # = 2 expected events, ever
    N = np.array([1.0, asymptote, asymptote + 1.0])
    x = CoxLewis.inv_cif(N, alpha, beta)
    assert np.isfinite(x[0]) and x[0] > 0
    assert np.isinf(x[1]) and np.isinf(x[2])
    assert not np.isnan(x).any()


def test_time_terminated_simulation_with_improving_system():
    # An improving system generates finitely many events; simulation must
    # right-censor each sequence at T with valid (finite, non-NaN) event
    # times instead of spinning out NaNs.
    model = CoxLewis.fit(np.array([0.5, 1.0, 2.0, 4.0]), tl=0, tr=10)
    model.params = np.array([0.0, -0.5])
    data = model.time_terminated_simulation_data(T=100.0, items=5, seed=1)
    assert np.isfinite(data.x).all()
    assert (data.x >= 0).all()
    assert (data.x <= 100.0).all()
    # every sequence ends right-censored at the window close
    for item in np.unique(data.i):
        assert data.c[data.i == item][-1] == 1


def test_fit_recovers_log_linear_intensity():
    # Events simulated from a known log-linear intensity should recover the
    # generating parameters (alpha, beta) via the intensity, not an offset
    # reparameterisation.
    rng = np.random.default_rng(0)
    alpha, beta = 0.0, 0.3
    # Thinning on [0, T]: homogeneous candidates at the max rate, kept w.p.
    # iif(t)/iif(T).
    T = 20.0
    lam_max = np.exp(alpha + beta * T)
    n_cand = rng.poisson(lam_max * T)
    cand = np.sort(rng.uniform(0, T, n_cand))
    keep = rng.uniform(0, 1, n_cand) < np.exp(alpha + beta * cand) / lam_max
    events = cand[keep]

    model = CoxLewis.fit(events, tl=0.0, tr=T)
    assert np.isclose(model.params[0], alpha, atol=0.5)
    assert np.isclose(model.params[1], beta, atol=0.1)
