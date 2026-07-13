"""Residual diagnostics, trend-test delegation and the Cramer-von Mises
goodness-of-fit test on fitted parametric recurrent models."""

import numpy as np
import pytest

from surpyval import Exponential
from surpyval.recurrent import HPP, CoxLewis, CrowAMSAA, laplace, mil_hdbk_189c
from surpyval.recurrent.diagnostics import cvm_statistic


def _events():
    np.random.seed(1)
    return Exponential.random(40, 1e-2).cumsum()


def _multi_item_data():
    # Item 1 closes with an explicit censoring row; items 2 and 3 are
    # failure-truncated (observed to their last event).
    x = [5, 9, 13, 16, 18, 20, 6, 10, 13, 15, 17, 12, 15, 19]
    i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    return x, i, c


def test_hpp_cumulative_hazard_residuals_are_rescaled_gaps():
    # For an HPP the time-rescaling transform is linear, so the residuals
    # are exactly lambda times the interarrival gaps.
    x = _events()
    model = HPP.fit(x)
    gaps = np.diff(np.concatenate([[0.0], x]))
    assert np.allclose(model.residuals(), model.params[0] * gaps)


def test_pit_residuals_transform_of_cumulative_hazard():
    model = HPP.fit(_events())
    e = model.residuals()
    pit = model.residuals(kind="pit")
    assert np.allclose(pit, 1.0 - np.exp(-e))
    assert np.all((pit >= 0) & (pit <= 1))


def test_martingale_residuals_sum_to_zero_at_hpp_mle():
    # The HPP MLE is total events / total exposure, which forces the summed
    # observed-minus-expected counts to zero exactly.
    x, i, c = _multi_item_data()
    model = HPP.fit(x, i=i, c=c)
    mart = model.residuals(kind="martingale")
    assert mart.shape == (3,)
    assert np.isclose(mart.sum(), 0.0, atol=1e-8)


def test_residuals_with_delayed_entry():
    # With left truncation the first interval starts at the entry time, so
    # residuals use cif differences from tl, not from 0.
    model = CrowAMSAA.fit([12, 15, 19, 21], i=[1] * 4, c=[0, 0, 0, 1], tl=10.0)
    e = model.residuals()
    x = np.array([12.0, 15.0, 19.0])
    expected = np.diff(model.cif(np.concatenate([[10.0], x])))
    assert np.allclose(e, expected)


def test_residuals_validation():
    model = HPP.fit(_events())
    with pytest.raises(ValueError, match="kind"):
        model.residuals(kind="nope")
    no_data = CrowAMSAA.from_params([1000.0, 1.2])
    with pytest.raises(ValueError, match="fitted from data"):
        no_data.residuals()
    interval = CrowAMSAA.fit(
        [[0, 5], [5, 10], [10, 15]], c=[2, 2, 2], n=[3, 2, 2]
    )
    with pytest.raises(ValueError, match="exact event times"):
        interval.residuals()


def test_trend_test_matches_standalone_failure_truncated():
    # A failure-truncated single system: the model method must reproduce
    # the standalone test's statistic exactly.
    x = _events()
    model = HPP.fit(x)
    for name, func in (("laplace", laplace), ("mil_hdbk_189c", mil_hdbk_189c)):
        res = model.trend_test(test=name)
        direct = func(x)
        assert np.isclose(res.statistic, direct.statistic)
        assert res.p_value == direct.p_value


def test_trend_test_matches_standalone_mixed_windows():
    # Mixed truncation: item 1 is time-truncated at its censoring row while
    # items 2 and 3 are failure-truncated. The model method must agree with
    # a hand-built standalone call using the same windows.
    x, i, c = _multi_item_data()
    model = CrowAMSAA.fit(x, i=i, c=c)
    res = model.trend_test()
    direct = laplace(
        [5, 9, 13, 16, 18, 6, 10, 13, 15, 12, 15],
        i=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        T={1: 20.0, 2: 17.0, 3: 19.0},
    )
    assert np.isclose(res.statistic, direct.statistic)
    assert res.n_events == direct.n_events
    assert res.n_systems == direct.n_systems


def test_trend_test_validation():
    model = HPP.fit(_events())
    with pytest.raises(ValueError, match="`test` must be"):
        model.trend_test(test="nope")
    truncated = CrowAMSAA.fit(
        [12, 15, 19, 21], i=[1] * 4, c=[0, 0, 0, 1], tl=10.0
    )
    with pytest.raises(ValueError, match="observation from time 0"):
        truncated.trend_test()


def test_cvm_statistic_minimum_at_uniform_quantiles():
    # The statistic attains its floor 1/(12M) when the sample sits exactly
    # on the uniform plotting positions.
    m = 10
    u = (2 * np.arange(1, m + 1) - 1) / (2 * m)
    assert np.isclose(cvm_statistic(u), 1.0 / (12 * m))
    # Any other sample is strictly larger.
    assert cvm_statistic(np.linspace(0.8, 0.99, m)) > 1.0 / (12 * m)


def test_cramer_von_mises_reproducible_and_calibrated():
    # A correctly-specified HPP should not be rejected, and the same seed
    # must reproduce the same p-value.
    model = HPP.fit(_events())
    gof = model.cramer_von_mises(n_boot=50, seed=1)
    gof2 = model.cramer_von_mises(n_boot=50, seed=1)
    assert gof.p_value == gof2.p_value
    assert 0 < gof.p_value <= 1
    assert gof.p_value > 0.05
    assert gof.n_boot == 50
    assert gof.n_systems == 1
    assert "p-value" in repr(gof)


def test_cramer_von_mises_rejects_misspecified_model():
    # Events from a strongly increasing intensity, fitted with a constant
    # one: the test must reject at its bootstrap resolution floor.
    t = np.linspace(1, 40, 40)
    x = (t / 40) ** (1.0 / 3.0) * 4000.0
    model = HPP.fit(x)
    gof = model.cramer_von_mises(n_boot=50, seed=1)
    assert gof.p_value <= 0.02


def test_cramer_von_mises_multi_item_and_cox_lewis():
    x, i, c = _multi_item_data()
    model = CrowAMSAA.fit(x, i=i, c=c)
    gof = model.cramer_von_mises(n_boot=30, seed=0)
    # Failure-truncated items drop their final event from the statistic.
    assert gof.n_events == 11
    assert gof.n_systems == 3
    assert 0 < gof.p_value <= 1

    rng = np.random.default_rng(0)
    T = 20.0
    lam_max = np.exp(0.3 * T)
    cand = np.sort(rng.uniform(0, T, rng.poisson(lam_max * T)))
    keep = rng.uniform(0, 1, cand.size) < np.exp(0.3 * cand) / lam_max
    cox = CoxLewis.fit(cand[keep], tl=0.0, tr=T)
    gof = cox.cramer_von_mises(n_boot=30, seed=1)
    assert gof.p_value > 0.05


def test_cramer_von_mises_requires_likelihood_fit():
    x = _events()
    mse = CrowAMSAA.fit(x, how="MSE")
    with pytest.raises(ValueError, match="fitted from data"):
        mse.cramer_von_mises()
    no_data = CrowAMSAA.from_params([1000.0, 1.2])
    with pytest.raises(ValueError, match="fitted from data"):
        no_data.cramer_von_mises()
