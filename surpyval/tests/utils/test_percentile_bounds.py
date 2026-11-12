"""The rule the percentile bootstrap bounds read their levels from the
resamples with (``utils.linalg.percentile_bounds``, used by the
Kaplan-Meier, degradation, destructive-degradation and Buckley-James
bootstraps, and ``bootstrap_quantile`` for the diagnostic plots' lowess
band).

The bound at level ``p`` of ``n`` resamples is the ``(n + 1) p``-th
smallest, interpolated (Davison and Hinkley 1997, section 5.2.2), the same
rule as the BC/BCa bounds of the parametric models (``bca_bounds``). The
old rule, ``np.quantile``'s default ``(n - 1) p``, read every tail level
nearer the middle; the first two tests fail on it.
"""

import numpy as np
import pytest

from surpyval.univariate.regression._bootstrap import bca_bounds
from surpyval.utils.linalg import (
    bootstrap_position,
    bootstrap_quantile,
    percentile_bounds,
)


def test_evenly_spaced_resamples_give_the_levels_exactly():
    # 199 resamples at k / 200: the (n + 1) p-th is exactly p. The old
    # (n - 1) p rule gave 0.0545 and 0.9455 for a 90% interval.
    draws = np.arange(1, 200)[:, None] / 200.0
    lo, hi = percentile_bounds(draws, 0.1)[0]
    assert lo == pytest.approx(0.05, abs=1e-12)
    assert hi == pytest.approx(0.95, abs=1e-12)
    assert percentile_bounds(draws, 0.1, "lower")[0] == pytest.approx(0.1)
    assert percentile_bounds(draws, 0.1, "upper")[0] == pytest.approx(0.9)


def test_bounds_sit_at_their_levels_on_average():
    # 4,000 sets of 200 uniform resamples (one set per column): the 95%
    # bounds average 0.025 and 0.975. The old rule averaged about 0.0295
    # and 0.9703 (an interval about 2% too narrow).
    rng = np.random.default_rng(20261112)
    draws = rng.uniform(size=(200, 4000))
    bounds = percentile_bounds(draws, 0.05)
    lo, hi = bounds[:, 0].mean(), bounds[:, 1].mean()
    # Each mean's standard error is about 0.00017.
    assert lo == pytest.approx(0.025, abs=0.001)
    assert hi == pytest.approx(0.975, abs=0.001)


def test_position_and_quantile_are_one_rule():
    # Indexing the sorted resamples at ``bootstrap_position`` (as
    # ``bca_bounds`` does) gives ``bootstrap_quantile``, ends included.
    rng = np.random.default_rng(3)
    n = 57
    draws = np.sort(rng.normal(size=n))
    for p in [0.001, 0.01, 0.025, 0.1, 0.5, 0.9, 0.975, 0.99, 0.999]:
        pos = float(bootstrap_position(p, n))
        lo = int(np.clip(np.floor(pos), 0, n - 1))
        hi = min(lo + 1, n - 1)
        frac = max(pos - lo, 0.0)
        by_hand = draws[lo] + frac * (draws[hi] - draws[lo])
        assert bootstrap_quantile(draws, p) == pytest.approx(by_hand)


def test_percentile_and_bca_bounds_agree_with_no_correction():
    # Resamples symmetric about the estimate (no bias correction) and no
    # acceleration: the BCa bounds are the percentile bounds.
    draws = np.arange(1, 200)[:, None] / 200.0
    est = np.array([0.5])
    for alpha_ci in (0.05, 0.1, 0.2):
        bca = bca_bounds(draws, est, np.zeros(1), alpha_ci, "two-sided")
        pct = percentile_bounds(draws, alpha_ci)
        assert np.allclose(bca, pct, atol=1e-12)


def test_nan_skip():
    draws = np.arange(1, 200, dtype=float)[:, None] / 200.0
    with_nan = np.vstack([draws, [[np.nan]]])
    assert np.isnan(bootstrap_quantile(with_nan, 0.05)[0])
    assert bootstrap_quantile(with_nan, 0.05, nan_skip=True)[0] == (
        pytest.approx(0.05)
    )
