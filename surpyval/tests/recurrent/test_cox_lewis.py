import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize

from surpyval.recurrent import CoxLewis, CrowAMSAA, ProportionalIntensityNHPP
from surpyval.tests._helpers import no_warnings
from surpyval.tests.conformance.registry import recurrent_data


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
    data = model.time_terminated_simulation_data(
        T=100.0, items=5, random_state=1
    )
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


# ---------------------------------------------------------------------------
# The Cox-Lewis intensity at zero and tiny ``beta``.
# ---------------------------------------------------------------------------


def test_cox_lewis_zero_and_tiny_beta():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = CoxLewis.from_params([0.5, 0.0])
        assert np.allclose(flat.cif([1, 2]), np.exp(0.5) * np.array([1, 2]))
        assert np.allclose(
            flat.inv_cif([1, 2]), np.array([1, 2]) / np.exp(0.5)
        )
        tiny = CoxLewis.from_params([0.0, 1e-14])
        assert tiny.cif(1.0) == pytest.approx(1.0, rel=1e-12)
        assert tiny.inv_cif(1.0) == pytest.approx(1.0, rel=1e-12)
    # unchanged away from zero
    model = CoxLewis.from_params([0.2, 0.3])
    assert model.cif(2.0) == pytest.approx(
        np.exp(0.2) / 0.3 * (np.exp(0.6) - 1.0)
    )


# ---------------------------------------------------------------------------
# Count termination (#386) and the least-squares fit (#419).
# ---------------------------------------------------------------------------


# -- #386: count termination of a falling intensity ----------------------
@pytest.mark.parametrize("seed", [0, 7, None])
def test_count_termination_of_a_bounded_count_is_refused(seed):
    # The fixture's fit falls (beta = -0.021): cif(inf) = 6.04 events ever,
    # so a sequence has fewer than 4 with probability 0.148. Seed 7 drew
    # one and failed on its infinite event time ("Event times 'x' must be
    # finite"); other seeds returned a sample. Now every seed says why.
    model = no_warnings(CoxLewis.fit, **recurrent_data())
    assert model.params[1] < 0
    with pytest.raises(ValueError, match=r"cif\(inf\) = 6\.04.*0\.148"):
        model.count_terminated_simulation(3, items=2, random_state=seed)
    with pytest.raises(ValueError, match="time_terminated_simulation"):
        model.count_terminated_simulation_data(3, items=2, random_state=seed)


def test_count_termination_of_a_bounded_count_with_covariates():
    d = recurrent_data(with_Z=True)
    model = no_warnings(ProportionalIntensityNHPP.fit, **d, baseline=CoxLewis)
    with pytest.raises(ValueError, match=r"cif\(inf\)"):
        model.count_terminated_simulation(3, Z=[0.5], random_state=1)


@pytest.mark.parametrize("params", [(0.0, 0.1), (0.0, 0.0), (-1.0, 1e-9)])
def test_count_termination_of_an_unbounded_count(params):
    # A rising or constant intensity reaches any count.
    model = CoxLewis.from_params(params)
    data = no_warnings(
        model.count_terminated_simulation_data, 3, items=5, random_state=2
    )
    assert np.all(np.isfinite(data.x))
    assert np.all(np.bincount(data.i)[1:] == 4)


def test_time_termination_of_a_bounded_count_still_works():
    model = no_warnings(CoxLewis.fit, **recurrent_data())
    sim = no_warnings(
        model.time_terminated_simulation_data, 60.0, items=5, random_state=7
    )
    assert np.all(np.isfinite(sim.x))


# -- #419: the least-squares fit -----------------------------------------
def _mcf_least_squares(data):
    """The least-squares objective, minimised independently: a tight
    Nelder-Mead from a grid of starts."""
    x, r, d = data.to_xrd()
    mcf = np.cumsum(d / r)

    def sse(p):
        with np.errstate(all="ignore"):
            v = np.sum((CoxLewis.cif(x, *p) - mcf) ** 2)
        return v if np.isfinite(v) else 1e300

    best = None
    for a0 in (-4.0, -2.0, 0.0):
        for b0 in (-0.05, 0.0, 0.05):
            res = minimize(
                sse,
                [a0, b0],
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000},
            )
            if best is None or res.fun < best.fun:
                best = res
    return best


def _sample():
    model = no_warnings(CoxLewis.fit, **recurrent_data())
    return model.time_terminated_simulation_data(
        60.0, items=40, random_state=1
    )


def test_least_squares_fit_of_a_sample_from_the_model():
    # From the all-ones start (a rate growing e-fold per unit time, cif(60)
    # = 1e26) BFGS stopped at alpha, beta = 0.98, -0.0099: cif(55) = 113.4
    # against 4.40 by MLE and 4.14 true, with no warning.
    data = _sample()
    mse = no_warnings(CoxLewis.fit_from_recurrent_data, data, how="MSE")
    mle = no_warnings(CoxLewis.fit_from_recurrent_data, data)
    best = _mcf_least_squares(data)
    np.testing.assert_allclose(mse.params, best.x, rtol=1e-3)
    assert mse.res.fun <= best.fun * (1 + 1e-6)
    np.testing.assert_allclose(mse.cif(55.0), mle.cif(55.0), rtol=0.05)


def test_least_squares_fit_of_the_fixture():
    # It was alpha, beta = -15.1, 0.263 (cif(60) = 2.1 against an MCF of
    # 4.67), stopped by BFGS's "precision loss".
    d = recurrent_data()
    mse = no_warnings(CoxLewis.fit, **d, how="MSE")
    best = _mcf_least_squares(mse.data)
    np.testing.assert_allclose(mse.params, best.x, rtol=1e-3)
    np.testing.assert_allclose(mse.params, [-1.856, -0.0305], atol=2e-3)


@pytest.mark.parametrize("how", ["MSE", "MLE"])
def test_cox_lewis_fit_does_not_depend_on_the_unit(how):
    # Time in hours rather than days: the same curve (principle 6).
    d = recurrent_data()
    days = no_warnings(CoxLewis.fit, **d, how=how)
    hours = no_warnings(CoxLewis.fit, **{**d, "x": d["x"] * 24.0}, how=how)
    t = np.array([5.0, 30.0, 60.0])
    np.testing.assert_allclose(hours.cif(t * 24.0), days.cif(t), rtol=1e-3)


def test_least_squares_fit_that_stops_early_is_finished():
    # BFGS's "precision loss" stop is now followed by a Nelder-Mead
    # search, as the likelihood's is, so a converged fit does not warn;
    # the other intensity models are unchanged.
    mse = no_warnings(CoxLewis.fit, **recurrent_data(), how="MSE")
    assert mse.res.success
    ca = no_warnings(CrowAMSAA.fit, **recurrent_data(), how="MSE")
    assert ca.res.success


# ---------------------------------------------------------------------------
# The log-intercept is unbounded below (#286).
# ---------------------------------------------------------------------------


class TestCoxLewisBounds:
    def test_negative_log_intercept_recovered(self):
        # 286: the (0, None) bound pinned alpha at 0 for baseline rates
        # below one event per time unit.
        np.random.seed(8)
        xs, iis, cs = [], [], []
        for it in range(100):
            t = 0.0
            while True:
                u = np.random.uniform()
                inc = (
                    np.log(1 - 0.05 * np.log(u) / np.exp(-1.0 + 0.05 * t))
                    / 0.05
                )
                t += inc
                if t > 30.0:
                    break
                xs.append(t)
                iis.append(it)
                cs.append(0)
            xs.append(30.0)
            iis.append(it)
            cs.append(1)
        m = CoxLewis.fit(x=xs, i=iis, c=cs)
        assert m.params[0] == pytest.approx(-1.0, abs=0.15)
        assert m.params[1] == pytest.approx(0.05, abs=0.01)
