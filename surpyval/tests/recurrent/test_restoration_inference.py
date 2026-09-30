"""The restoration factor is printed with its uncertainty (#513).

A generalized renewal fit printed ``q`` as a bare number, so a fit to
minimal-repair data (true ``q = 1``) reported ``q = 2.63`` -- "every
repair leaves the truck worse than before it failed" -- although its 95%
interval ran from 0.094 to 73.4. The model now prints every parameter
with its standard error and Wald interval, says when the restoration
parameter is not determined by the data, and ``repair_test`` compares the
fit with minimal repair.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import recurrent as rc


def _trucks():
    # The issue's eight haul trucks under minimal repair (power law).
    rows = []
    rng = np.random.default_rng(8)
    for k in range(8):
        T = rng.uniform(6000, 12000)
        N = rng.poisson(2e-4 * T**1.35)
        ts = np.sort(T * rng.random(N) ** (1 / 1.35))
        rows += [(h, k, 0) for h in ts] + [(T, k, 1)]
    return map(np.array, zip(*rows))


def _renewals(n_items=30, n_events=10, seed=0):
    # As good as new (q = 0): each gap a fresh Weibull(10, 3) life.
    rng = np.random.default_rng(seed)
    gaps = 10.0 * rng.weibull(3.0, (n_items, n_events))
    x = np.cumsum(gaps, axis=1).ravel()
    i = np.repeat(np.arange(n_items), n_events)
    return x, i, np.zeros(x.size, int)


def test_undetermined_q_is_flagged_with_its_interval():
    x, i, c = _trucks()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    text = " ".join(repr(model).split())
    assert "q is not determined by these data" in text
    table = model.summary()
    assert table.loc["q", "estimate"] == pytest.approx(2.626, abs=1e-3)
    assert table.loc["q", "se"] == pytest.approx(4.463, rel=1e-2)
    np.testing.assert_allclose(
        table.loc["q", ["lower 95%", "upper 95%"]], [0.0939, 73.44], rtol=1e-2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb = model.param_cb("q")
    np.testing.assert_allclose(
        table.loc["q", ["lower 95%", "upper 95%"]], cb, rtol=1e-12
    )


def test_repair_test_against_minimal_repair():
    x, i, c = _trucks()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    test = model.repair_test()
    # q = 1 with a Weibull is the power-law NHPP: the restricted maximum is
    # Crow-AMSAA's.
    crow = rc.CrowAMSAA.fit(x, i, c)
    assert test.log_likelihood_minimal == pytest.approx(
        crow.log_likelihood, abs=1e-4
    )
    assert test.statistic == pytest.approx(0.398, abs=2e-3)
    assert test.p_value == pytest.approx(0.528, abs=2e-3)
    assert test.minimal == 1.0


def test_repair_test_rejects_minimal_repair_for_renewal_data():
    x, i, c = _renewals()
    model = rc.GeneralizedRenewal.fit(x, i, c)
    assert "not determined" not in repr(model)
    test = model.repair_test()
    assert test.p_value < 1e-6


def test_a_determined_q_is_not_flagged():
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    model = rc.GeneralizedRenewal.fit(x, dist=sp.Weibull)
    text = repr(model)
    assert "Wald 95% intervals" in text
    assert "Note" not in text


def test_q_at_its_edge_says_so():
    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
    model = rc.GeneralizedRenewal.fit(x, i, c)
    assert "at the edge of its range" in " ".join(repr(model).split())
    assert np.isnan(model.summary().loc["q", "se"])


def test_ara_rho_uses_its_own_scale():
    x, i, c = _renewals(20, 8, seed=1)
    model = rc.ARA.fit(x, i, c)
    # renewals: as good as new is rho = 1, the other edge of its range
    assert model.rho == pytest.approx(1.0, abs=1e-6)
    assert "rho = 1 is at the edge" in " ".join(repr(model).split())
    x, i, c = _trucks()
    table = rc.ARA.fit(x, i, c).summary()
    assert np.isnan(table.loc["rho", "se"])  # rho -> 0 on minimal repair
    # rho = 0 is on the edge: the p-value is halved (chi-bar-squared)
    test = model.repair_test()
    assert test.minimal == 0.0
    assert test.p_value < 1e-6


def test_no_minimal_repair_for_g1_and_no_likelihood_from_parameters():
    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    g1 = rc.GeneralizedOneRenewal.fit(x)
    assert "Wald 95% intervals" in repr(g1)
    with pytest.raises(ValueError, match="no minimal-repair value"):
        g1.repair_test()
    given = rc.GeneralizedRenewal.fit_from_parameters([10, 2], 0.2)
    assert "Restoration Factor  : 0.2" in repr(given)
    with pytest.raises(ValueError, match="Likelihood inference"):
        given.repair_test()
