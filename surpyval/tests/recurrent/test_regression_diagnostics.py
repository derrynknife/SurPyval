"""Residual and trend-test diagnostics for the proportional-intensity
recurrent regression models.

Each item's cumulative intensity is the baseline scaled by its covariate
factor ``exp(Z'beta)``, so the time-rescaling residuals are computed per item
with the item's own CIF. These tests check that the per-item plumbing is
exactly ``model.cif(., Z_item)``, that the residual shapes and identities
hold, and that the trend test delegates to the standalone statistic.
"""

import numpy as np
import pytest

from surpyval.recurrent import (
    CrowAMSAA,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
    laplace,
)


def _two_group_data():
    # Three items per covariate group; item windows close explicitly (c = 1).
    x = [
        9,
        14,
        18,
        20,
        7,
        12,
        16,
        19,
        20,
        5,
        9,
        13,
        16,
        18,
        20,
        6,
        10,
        13,
        15,
        17,
        19,
        20,
    ]
    i = [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    c = [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    Z = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).reshape(-1, 1)
    return x, i, c, Z


def _fitted():
    x, i, c, Z = _two_group_data()
    return ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)


def test_cumulative_hazard_residuals_are_per_item_cif_gaps():
    # The strongest check: each item's residuals equal the differences of its
    # own covariate-scaled CIF over [entry, *events].
    model = _fitted()
    x, i, c, Z = _two_group_data()
    x, i, c = np.array(x, float), np.array(i), np.array(c)
    Z = Z.astype(float)

    expected = []
    for item in np.unique(i):
        mask = i == item
        events = np.sort(x[mask][c[mask] == 0])
        Z_item = Z[mask][0]
        cif = model.cif(np.concatenate([[0.0], events]), Z_item)
        expected.append(np.diff(cif))
    expected = np.concatenate(expected)

    assert np.allclose(model.residuals("cumulative_hazard"), expected)
    # one residual per observed event
    assert model.residuals().size == int((c == 0).sum())


def test_pit_residuals_transform_of_cumulative_hazard():
    model = _fitted()
    e = model.residuals("cumulative_hazard")
    pit = model.residuals("pit")
    assert np.allclose(pit, 1.0 - np.exp(-e))
    assert np.all((pit >= 0) & (pit <= 1))


def test_martingale_residuals_are_count_minus_expected():
    model = _fitted()
    x, i, c, Z = _two_group_data()
    x, i, c = np.array(x, float), np.array(i), np.array(c)
    Z = Z.astype(float)

    expected = []
    for item in np.unique(i):
        mask = i == item
        events = x[mask][c[mask] == 0]
        Z_item = Z[mask][0]
        close = float(x[mask].max())  # explicit c = 1 close
        exp_count = float(model.cif(close, Z_item) - model.cif(0.0, Z_item))
        expected.append(events.size - exp_count)

    assert np.allclose(model.residuals("martingale"), expected)
    # one martingale residual per item
    assert model.residuals("martingale").size == np.unique(i).size


def test_covariate_scaling_enters_the_residuals():
    # A larger covariate scales the CIF by exp(Z'beta): the martingale
    # expected counts of the Z=1 group differ from what the baseline alone
    # (Z=0) would predict, unless the coefficient is exactly zero.
    model = _fitted()
    assert not np.isclose(model.coeffs[0], 0.0)
    # cif for the two groups differs by the exp(beta) factor
    t = np.array([10.0, 15.0])
    ratio = model.cif(t, np.array([1.0])) / model.cif(t, np.array([0.0]))
    assert np.allclose(ratio, np.exp(model.coeffs[0]))


def test_trend_test_delegates_to_standalone_statistic():
    model = _fitted()
    result = model.trend_test(test="laplace")
    # Rebuild the pooled (event, item, window) inputs the delegate uses and
    # compare against the standalone Laplace test directly.
    x, i, c, _ = _two_group_data()
    x, i, c = np.array(x, float), np.array(i), np.array(c)
    xs, items, T = [], [], {}
    for idx, item in enumerate(np.unique(i)):
        mask = i == item
        events = np.sort(x[mask][c[mask] == 0])
        xs.extend(events)
        items.extend([idx] * events.size)
        T[idx] = float(x[mask].max())
    direct = laplace(xs, i=items, T=T, alternative="two-sided")
    assert np.isclose(result.statistic, direct.statistic)
    assert np.isclose(result.p_value, direct.p_value)


def test_residuals_reject_interval_censored_data():
    # Interval-censored rows cannot be placed in time, so the residuals must
    # refuse them rather than silently mislead.
    xl = [5, 9, 13]
    xr = [6, 10, 14]
    x = np.array([xl, xr]).T
    Z = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
    model = ProportionalIntensityNHPP.fit(
        x, Z, i=[1, 1, 1], c=[2, 2, 2], dist=CrowAMSAA
    )
    with pytest.raises(ValueError, match="exact event times"):
        model.residuals()


def test_bad_kind_raises():
    model = _fitted()
    with pytest.raises(ValueError, match="cumulative_hazard"):
        model.residuals(kind="nonsense")


def test_hpp_proportional_intensity_diagnostics():
    # The homogeneous proportional-intensity model exposes the same
    # diagnostics.
    x, i, c, Z = _two_group_data()
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    assert model.residuals("cumulative_hazard").size == int(
        (np.array(c) == 0).sum()
    )
    assert model.residuals("martingale").size == np.unique(i).size
    assert np.all(
        (model.residuals("pit") >= 0) & (model.residuals("pit") <= 1)
    )
