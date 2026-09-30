"""Greenwood and Nelson-Aalen variances snap d / r elementwise (#498).

Both used a Python loop of ``_snap`` over every time point, which was 85%
of a 100,000-row Kaplan-Meier fit. The vectorised ``_snap_array`` must give
exactly the same variance, including on the round-off cases the snap exists
for: Turnbull EM expected counts one ulp off a whole number, d == r, a zero
count, and 0 / 0 with no one at risk.
"""

import numpy as np
import pytest

from surpyval.univariate.nonparametric import (
    greenwood_variance,
    nelson_aalen_variance,
)
from surpyval.univariate.nonparametric.fleming_harrington import _snap


def _greenwood_loop(r, d):
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    with np.errstate(all="ignore"):
        var = d / (r * (r - d))
        q = np.array([_snap(v) for v in np.where(d == 0, 0.0, d / r)])
        var = np.where(q == 1, np.nan, np.where(q == 0, 0.0, var))
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def _nelson_aalen_loop(r, d):
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    with np.errstate(all="ignore"):
        var = d / r**2
        q = np.array([_snap(v) for v in np.where(d == 0, 0.0, d / r)])
        var = np.where(q == 0, 0.0, var)
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def _cases():
    rng = np.random.default_rng(498)
    r = np.arange(200, 0, -1).astype(float)
    d = rng.binomial(3, 0.3, 200).astype(float)
    yield "integer counts", r, d
    # Turnbull-style fractional expected counts carrying round-off.
    rf = rng.uniform(1, 50, 60)
    df = rf * rng.uniform(0, 1, 60)
    df[::7] = 0.0
    df[3] = rf[3] * (1 + 2e-16)  # d == r up to round-off
    df[5] = 1e-16  # no event up to round-off
    yield "fractional with round-off", rf, df
    # A step with no one at risk and no event (0 / 0, #425), and d == r.
    yield "0/0 and d == r", np.array([5.0, 3.0, 0.0, 2.0]), np.array(
        [1.0, 0.0, 0.0, 2.0]
    )


@pytest.mark.parametrize("label,r,d", list(_cases()))
def test_greenwood_matches_elementwise_snap(label, r, d):
    np.testing.assert_array_equal(
        greenwood_variance(r, d), _greenwood_loop(r, d)
    )


@pytest.mark.parametrize("label,r,d", list(_cases()))
def test_nelson_aalen_matches_elementwise_snap(label, r, d):
    np.testing.assert_array_equal(
        nelson_aalen_variance(r, d), _nelson_aalen_loop(r, d)
    )
