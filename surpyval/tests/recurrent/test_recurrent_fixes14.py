"""Cox-Lewis count termination (#386) and least-squares fit (#419)."""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

from surpyval.recurrent import CoxLewis, CrowAMSAA, ProportionalIntensityNHPP
from surpyval.tests.conformance.registry import recurrent_data


def _silent(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(*args, **kwargs)


# -- #386: count termination of a falling intensity ----------------------
@pytest.mark.parametrize("seed", [0, 7, None])
def test_count_termination_of_a_bounded_count_is_refused(seed):
    # The fixture's fit falls (beta = -0.021): cif(inf) = 6.04 events ever,
    # so a sequence has fewer than 4 with probability 0.148. Seed 7 drew
    # one and failed on its infinite event time ("Event times 'x' must be
    # finite"); other seeds returned a sample. Now every seed says why.
    model = _silent(CoxLewis.fit, **recurrent_data())
    assert model.params[1] < 0
    with pytest.raises(ValueError, match=r"cif\(inf\) = 6\.04.*0\.148"):
        model.count_terminated_simulation(3, items=2, random_state=seed)
    with pytest.raises(ValueError, match="time_terminated_simulation"):
        model.count_terminated_simulation_data(3, items=2, random_state=seed)


def test_count_termination_of_a_bounded_count_with_covariates():
    d = recurrent_data(with_Z=True)
    model = _silent(ProportionalIntensityNHPP.fit, **d, dist=CoxLewis)
    with pytest.raises(ValueError, match=r"cif\(inf\)"):
        model.count_terminated_simulation(3, Z=[0.5], random_state=1)


@pytest.mark.parametrize("params", [(0.0, 0.1), (0.0, 0.0), (-1.0, 1e-9)])
def test_count_termination_of_an_unbounded_count(params):
    # A rising or constant intensity reaches any count.
    model = CoxLewis.from_params(params)
    data = _silent(
        model.count_terminated_simulation_data, 3, items=5, random_state=2
    )
    assert np.all(np.isfinite(data.x))
    assert np.all(np.bincount(data.i)[1:] == 4)


def test_time_termination_of_a_bounded_count_still_works():
    model = _silent(CoxLewis.fit, **recurrent_data())
    sim = _silent(
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
    model = _silent(CoxLewis.fit, **recurrent_data())
    return model.time_terminated_simulation_data(
        60.0, items=40, random_state=1
    )


def test_least_squares_fit_of_a_sample_from_the_model():
    # From the all-ones start (a rate growing e-fold per unit time, cif(60)
    # = 1e26) BFGS stopped at alpha, beta = 0.98, -0.0099: cif(55) = 113.4
    # against 4.40 by MLE and 4.14 true, with no warning.
    data = _sample()
    mse = _silent(CoxLewis.fit_from_recurrent_data, data, how="MSE")
    mle = _silent(CoxLewis.fit_from_recurrent_data, data)
    best = _mcf_least_squares(data)
    np.testing.assert_allclose(mse.params, best.x, rtol=1e-3)
    assert mse.res.fun <= best.fun * (1 + 1e-6)
    np.testing.assert_allclose(mse.cif(55.0), mle.cif(55.0), rtol=0.05)


def test_least_squares_fit_of_the_fixture():
    # It was alpha, beta = -15.1, 0.263 (cif(60) = 2.1 against an MCF of
    # 4.67), stopped by BFGS's "precision loss".
    d = recurrent_data()
    mse = _silent(CoxLewis.fit, **d, how="MSE")
    best = _mcf_least_squares(mse.data)
    np.testing.assert_allclose(mse.params, best.x, rtol=1e-3)
    np.testing.assert_allclose(mse.params, [-1.856, -0.0305], atol=2e-3)


@pytest.mark.parametrize("how", ["MSE", "MLE"])
def test_cox_lewis_fit_does_not_depend_on_the_unit(how):
    # Time in hours rather than days: the same curve (principle 6).
    d = recurrent_data()
    days = _silent(CoxLewis.fit, **d, how=how)
    hours = _silent(CoxLewis.fit, **{**d, "x": d["x"] * 24.0}, how=how)
    t = np.array([5.0, 30.0, 60.0])
    np.testing.assert_allclose(hours.cif(t * 24.0), days.cif(t), rtol=1e-3)


def test_least_squares_fit_that_stops_early_is_finished():
    # BFGS's "precision loss" stop is now followed by a Nelder-Mead
    # search, as the likelihood's is, so a converged fit does not warn;
    # the other intensity models are unchanged.
    mse = _silent(CoxLewis.fit, **recurrent_data(), how="MSE")
    assert mse.res.success
    ca = _silent(CrowAMSAA.fit, **recurrent_data(), how="MSE")
    assert ca.res.success
