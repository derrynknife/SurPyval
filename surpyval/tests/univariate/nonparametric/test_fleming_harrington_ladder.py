"""The Fleming-Harrington tie ladder (``fh_h``/``fh_var_h``).

The ladder used to be a per-event Python ``while`` loop. The Turnbull
EM feeds it *fractional expected* event counts which, under heavy
truncation, grow without bound between iterations -- the loop then
effectively (or with ``d = inf``, literally) never returned, hanging
any Turnbull fit with truncation (and with it the documentation
builds). The ladder is now evaluated in closed form (digamma /
trigamma harmonic sums) beyond a small exact loop, so its cost is O(1)
in the event count. These tests pin:

- exact agreement with the original term-by-term ladder for ordinary
  integer and fractional tie counts;
- instant, principled behaviour on pathological counts (huge or
  infinite ``d`` and ladders that exhaust the risk set give ``inf``,
  i.e. a diverging hazard, rather than looping or producing garbage);
- that the formerly hanging Turnbull fits now terminate quickly.
"""

import time

import numpy as np
import pytest

from surpyval import Turnbull
from surpyval.univariate.nonparametric.fleming_harrington import (
    fh_h,
    fh_var_h,
)


def _ladder_h(r_i, d_i):
    out = 0.0
    while d_i > 1:
        out += 1.0 / r_i
        r_i -= 1
        d_i -= 1
    return out + d_i / r_i


def _ladder_var(r_i, d_i):
    out = 0.0
    while d_i > 1:
        out += 1.0 / r_i**2
        r_i -= 1
        d_i -= 1
    return out + d_i / r_i**2


def test_matches_term_by_term_ladder():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d = rng.uniform(0.01, 30.0)
        if rng.random() < 0.5:
            d = float(int(d) + 1)  # integer tie counts too
        r = d + rng.uniform(0.5, 100.0)
        assert np.isclose(fh_h(r, d), _ladder_h(r, d), rtol=1e-12)
        assert np.isclose(fh_var_h(r, d), _ladder_var(r, d), rtol=1e-12)


def test_closed_form_matches_ladder_for_large_counts():
    for d in (100.3, 5000.0, 20000.7):
        r = d + 7.25
        assert np.isclose(fh_h(r, d), _ladder_h(r, d), rtol=1e-9)
        assert np.isclose(fh_var_h(r, d), _ladder_var(r, d), rtol=1e-9)


def test_pathological_counts_return_instantly():
    start = time.time()
    assert fh_h(10.0, 1e15) == np.inf  # ladder exhausts the risk set
    assert fh_h(10.0, np.inf) == np.inf
    assert fh_var_h(0.3, 2.0) == np.inf
    assert np.isfinite(fh_h(1e16, 1e12))  # huge but valid: closed form
    assert time.time() - start < 1.0


def test_small_counts_unchanged():
    assert fh_h(5.0, 1.0) == pytest.approx(0.2)
    assert fh_h(5.0, 0.5) == pytest.approx(0.1)
    assert fh_h(5.0, 2.0) == pytest.approx(1 / 5 + 1 / 4)
    assert np.isnan(fh_h(5.0, np.nan))


def test_truncated_turnbull_terminates():
    # This is the (formerly hanging) docs example: a small, heavily
    # censored sample with two-sided truncation. Its NPMLE is
    # non-identifiable, so the estimate is degenerate (the EM may or
    # may not flag non-convergence) -- but the fit must terminate
    # promptly rather than hang.
    import warnings

    x = [1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12]
    c = [1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0]
    n = [1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1]
    tl = [0, 0, 0, 0, 0, 2, 3, 3, 1, 1, 5]
    tr = [np.inf, np.inf, 10, 10, 10, 10, np.inf, np.inf, np.inf, 15, 15]
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Turnbull.fit(x=x, c=c, n=n, tl=tl, tr=tr)
    assert time.time() - start < 30.0
    R = np.asarray(model.R, dtype=float)
    assert np.all((R >= -1e-9) & (R <= 1 + 1e-9))


def test_left_truncated_turnbull_is_accurate_on_healthy_data():
    # A well-sized left-truncated sample: the truncated NPMLE should
    # recover the underlying survival function (and must not hang).
    rng = np.random.default_rng(1)
    t_true = 10 * rng.weibull(2.0, 200)
    tl = rng.uniform(0, 4, 200)
    keep = t_true > tl * 1.15
    x, tlk = t_true[keep], tl[keep]
    xi = [[v * 0.9, v * 1.1] if i % 3 == 0 else v for i, v in enumerate(x)]
    c = np.where(np.arange(x.size) % 3 == 0, 2, 0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tol non-convergence is fine
        model = Turnbull.fit(x=xi, c=c, tl=tlk)
    # true S(8.3) for Weibull(10, 2) is exp(-(0.83)^2) ~= 0.502
    s_med = float(np.interp(8.3, model.x, model.R))
    assert abs(s_med - 0.502) < 0.1
