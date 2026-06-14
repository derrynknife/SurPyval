"""Unit tests for the helpers extracted from ``fit_from_surpyval_data``.

These exercise ``_clamp_truncation_to_support``, ``_initial_guess`` and
``_set_support`` directly, without running a full optimisation, so each
piece of the fit pipeline can be checked in isolation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from surpyval import (
    Beta,
    Beta4,
    Exponential,
    Normal,
    Uniform,
    Weibull,
)


# ---------------------------------------------------------------------------
# _clamp_truncation_to_support
# ---------------------------------------------------------------------------


def test_clamp_truncation_half_line():
    # Weibull support is [0, inf): the left bound is finite (clamped at 0)
    # while the right bound is infinite (left untouched).
    t = np.array([[-5.0, np.inf], [-1.0, 10.0], [3.0, np.inf]])
    tl, tr = Weibull._clamp_truncation_to_support(t)
    assert np.array_equal(tl, [0.0, 0.0, 3.0])
    assert np.array_equal(tr, [np.inf, 10.0, np.inf])


def test_clamp_truncation_finite_both_edges():
    # Beta support is [0, 1]: both edges are finite and clamp.
    t = np.array([[-0.2, 2.0], [0.1, 0.9], [-1.0, 1.5]])
    tl, tr = Beta._clamp_truncation_to_support(t)
    assert np.array_equal(tl, [0.0, 0.1, 0.0])
    assert np.array_equal(tr, [1.0, 0.9, 1.0])


def test_clamp_truncation_unbounded_is_noop():
    # Normal support is (-inf, inf): nothing is clamped.
    t = np.array([[-1e9, 1e9], [-3.0, 4.0]])
    tl, tr = Normal._clamp_truncation_to_support(t)
    assert np.array_equal(tl, t[:, 0])
    assert np.array_equal(tr, t[:, 1])


# ---------------------------------------------------------------------------
# _initial_guess
# ---------------------------------------------------------------------------


def test_initial_guess_returns_one_value_per_parameter():
    np.random.seed(0)
    x = Weibull.random(500, 10.0, 3.0)
    c = np.zeros_like(x)
    n = np.ones_like(x)

    init = Weibull._initial_guess(
        x, c, n, offset=False, zi=False, lfp=False, heuristic="Nelson-Aalen"
    )

    assert len(init) == Weibull.k
    assert np.all(np.isfinite(init))
    # The seed should be in the right ballpark for the generating params.
    assert init[0] == pytest.approx(10.0, rel=0.5)


def test_initial_guess_lfp_appends_a_bounded_p_seed():
    np.random.seed(1)
    x = Weibull.random(500, 10.0, 3.0)
    c = np.zeros_like(x)
    n = np.ones_like(x)

    init = Weibull._initial_guess(
        x, c, n, offset=False, zi=False, lfp=True, heuristic="Nelson-Aalen"
    )

    assert len(init) == Weibull.k + 1
    # The limited-failure seed is min(0.6, max_F) and so lies in (0, 0.6].
    assert 0.0 < init[-1] <= 0.6


def test_initial_guess_zi_appends_zero_fraction():
    np.random.seed(2)
    x = np.concatenate([np.zeros(50), Weibull.random(450, 10.0, 3.0)])
    c = np.zeros_like(x)
    n = np.ones_like(x)

    init = Weibull._initial_guess(
        x, c, n, offset=False, zi=True, lfp=False, heuristic="Nelson-Aalen"
    )

    assert len(init) == Weibull.k + 1
    # The zero-inflation seed is the observed fraction of exact zeros.
    assert init[-1] == pytest.approx(50 / x.size)


def test_initial_guess_offset_seeds_gamma_below_min():
    np.random.seed(3)
    x = Weibull.random(500, 10.0, 3.0) + 7.0
    c = np.zeros_like(x)
    n = np.ones_like(x)

    init = Weibull._initial_guess(
        x, c, n, offset=True, zi=False, lfp=False, heuristic="Nelson-Aalen"
    )

    # Offset distributions carry gamma as the leading parameter; the seed
    # is one below the smallest observation.
    assert len(init) == Weibull.k + 1
    assert init[0] == pytest.approx(x.min() - 1.0)


def test_initial_guess_interval_data_uses_midpoint():
    # 2D x (interval censored): the helper imputes the midpoint and must
    # still return a finite seed of the right length.
    np.random.seed(4)
    centres = Weibull.random(300, 10.0, 3.0)
    x = np.vstack([centres - 0.5, centres + 0.5]).T
    c = np.full(centres.shape, 2)
    n = np.ones(centres.shape)

    init = Weibull._initial_guess(
        x, c, n, offset=False, zi=False, lfp=False, heuristic="Nelson-Aalen"
    )

    assert len(init) == Weibull.k
    assert np.all(np.isfinite(init))


# ---------------------------------------------------------------------------
# _set_support
# ---------------------------------------------------------------------------


def _make_model(params, gamma=0.0):
    return SimpleNamespace(params=np.asarray(params, dtype=float), gamma=gamma)


def test_set_support_half_line():
    model = _make_model([10.0, 3.0])
    Weibull._set_support(model, offset=False)
    assert np.array_equal(model.support, [0.0, np.inf])


def test_set_support_offset_uses_gamma():
    model = _make_model([10.0, 3.0], gamma=4.5)
    Weibull._set_support(model, offset=True)
    assert np.array_equal(model.support, [4.5, np.inf])


def test_set_support_finite_both_edges():
    model = _make_model([2.0, 5.0])
    Beta._set_support(model, offset=False)
    assert np.array_equal(model.support, [0.0, 1.0])


def test_set_support_unbounded():
    model = _make_model([0.0, 1.0])
    Normal._set_support(model, offset=False)
    assert np.array_equal(model.support, [-np.inf, np.inf])


def test_set_support_data_dependent_reads_nominated_params():
    # Beta4 declares support (nan, nan) and reads the bounds from params
    # 2 and 3 (``a`` and ``b``).
    model = _make_model([1.5, 2.5, 3.0, 8.0])
    Beta4._set_support(model, offset=False)
    assert np.array_equal(model.support, [3.0, 8.0])


def test_set_support_exponential_single_param():
    model = _make_model([0.5])
    Exponential._set_support(model, offset=False)
    assert np.array_equal(model.support, [0.0, np.inf])


def test_set_support_matches_from_params():
    # _set_support is shared with from_params; the two must agree.
    model = _make_model([2.0, 1.5, 3.0, 8.0])
    Beta4._set_support(model, offset=False)
    via_from_params = Beta4.from_params([2.0, 1.5, 3.0, 8.0])
    assert np.array_equal(model.support, via_from_params.support)


def test_set_support_uniform_workaround():
    # Uniform keeps the (-inf, inf) declaration rather than NaN; the helper
    # must reproduce that rather than read the fitted a/b.
    model = _make_model([2.0, 7.0])
    Uniform._set_support(model, offset=False)
    assert np.array_equal(model.support, [-np.inf, np.inf])
