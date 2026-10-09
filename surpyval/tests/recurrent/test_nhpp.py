"""Fitting and simulating the parametric NHPP intensity models (HPP,
Crow-AMSAA, Duane, Cox-Lewis): degenerate data, starts and checks.
"""

import warnings

import numpy as np
import pytest

from surpyval.recurrent import (
    HPP,
    CoxLewis,
    CrowAMSAA,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)


def test_nhpp_simulation_does_not_underflow():
    # exp(-cif) underflowed once the expected count passed about 745.
    model = CrowAMSAA.from_params([10, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = model.time_terminated_simulation(100, items=200, random_state=1)
    assert np.isclose(sim.mcf(100), 1000, rtol=0.05)


# --- degenerate NHPP data -----------------------------------------------


@pytest.mark.parametrize("model", ["CrowAMSAA", "HPP", "PI-HPP", "PI-NHPP"])
def test_all_censored_data_is_a_clear_error(model):
    x, i, c, Z = [5, 6], [1, 2], [1, 1], [[0], [1]]
    with pytest.raises(ValueError, match="no events"):
        if model == "CrowAMSAA":
            CrowAMSAA.fit(x, i, c)
        elif model == "HPP":
            HPP.fit(x, i, c)
        elif model == "PI-HPP":
            ProportionalIntensityHPP.fit(x, Z, i, c)
        else:
            ProportionalIntensityNHPP.fit(x, Z, i, c, baseline=CrowAMSAA)


def test_power_law_event_at_time_zero_is_a_clear_error():
    with pytest.raises(ValueError, match="event at t = 0"):
        CrowAMSAA.fit([0, 3, 5], [1, 1, 1], [0, 0, 1])
    # the constant-rate model is fine there
    assert np.isclose(HPP.fit([0, 3, 5], [1, 1, 1], [0, 0, 1]).params[0], 0.4)


def test_power_law_rejects_times_before_zero():
    x, i, c = [-3, -1, 2, 4], [1] * 4, [0, 0, 0, 1]
    with pytest.raises(ValueError, match="outside it"):
        CrowAMSAA.fit(x, i, c, tl=-5)
    # models defined on the whole line take the negative window
    assert np.isclose(HPP.fit(x, i, c, tl=-5).params[0], 3 / 9)
    CoxLewis.fit(x, i, c, tl=-5)


def test_single_failure_truncated_event_is_a_clear_error():
    with pytest.raises(ValueError, match="single event"):
        CrowAMSAA.fit([5])
    # a window closed after the event identifies the power law
    CrowAMSAA.fit([5, 10], c=[0, 1])
    # and the one-parameter HPP is fine
    assert np.isclose(HPP.fit([5]).params[0], 0.2)


def test_nhpp_rejects_bad_how_and_init():
    with pytest.raises(ValueError, match="'how' must be"):
        CrowAMSAA.fit([1, 2, 3, 4], how="bad")
    with pytest.raises(ValueError, match="init must have 2 values"):
        CrowAMSAA.fit([1, 2, 3, 4], init=[1])


def test_hpp_init_must_be_a_positive_rate():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="positive"):
            HPP.fit([1, 2, 3, 4], init=[0])
        with pytest.raises(ValueError, match="positive"):
            ProportionalIntensityHPP.fit(
                [1, 2, 3], [[0], [1], [0]], [1, 2, 3], init=[0, 0]
            )


# ---------------------------------------------------------------------------
# An MSE fit has no likelihood, and says so.
# ---------------------------------------------------------------------------


def test_mse_fit_says_why_it_has_no_likelihood():
    from surpyval.recurrent import CrowAMSAA

    x = [1.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    model = CrowAMSAA.fit(x, c=[0] * 7 + [1], how="MSE")
    assert "MSE" in repr(model)
    with pytest.raises(ValueError, match="how='MSE'"):
        model.aic()


def test_665_crow_amsaa_closed_form_mle():
    # Every item observed from 0 to a common end: the MIL-HDBK-189C closed
    # form, to full precision (the search agreed only to about 1e-5).
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 100, 15))
    beta = 15 / np.log(100 / t).sum()
    expected = [100 * (1 / 15) ** (1 / beta), beta]
    # Time terminated, by a c=1 row or by tr
    via_row = CrowAMSAA.fit(np.r_[t, 100], c=np.r_[np.zeros(15), 1])
    np.testing.assert_allclose(via_row.params, expected, rtol=1e-14)
    assert via_row.maximum == "verified"
    via_tr = CrowAMSAA.fit(t, tr=100.0)
    np.testing.assert_allclose(via_tr.params, expected, rtol=1e-14)
    # Failure terminated
    failure = CrowAMSAA.fit(t)
    beta = 15 / np.log(t[-1] / t).sum()
    np.testing.assert_allclose(failure.params[1], beta, rtol=1e-14)
    # Two systems to a common T: the expected count at T is N / k.
    x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
    i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    model = CrowAMSAA.fit(x, i=i, c=c)
    assert model.cif(60) == pytest.approx(4.5, rel=1e-14)
    # Unequal ends are searched, as before.
    searched = CrowAMSAA.fit(x[:-1] + [50], i=i, c=c)
    assert searched.maximum == "verified"
