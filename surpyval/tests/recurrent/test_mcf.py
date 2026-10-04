"""The non-parametric mean cumulative function (``NonParametricCounting``):
the estimate, its Lawless-Nadeau variance and bounds, and the risk set.
"""

import numpy as np
import pytest

from surpyval.recurrent import NonParametricCounting
from surpyval.utils.recurrent_event_data import RecurrentEventData


def test_linear_mcf_bounds_rise_from_zero_like_the_mcf():
    model = NonParametricCounting.fit(
        [2, 4, 6, 3, 5], [1, 1, 1, 2, 2], [0, 0, 1, 0, 1]
    )
    query = np.array([0.0, 1.0, 2.0])
    mcf = model.mcf(query, interp="linear")
    cb = model.mcf_cb(query, interp="linear")
    assert np.allclose(cb[0], [0.0, 0.0])
    assert np.all(cb[:, 0] <= mcf) and np.all(mcf <= cb[:, 1])
    upper = model.mcf_cb(query, interp="linear", bound="upper")
    assert upper[0] == 0 and np.all(upper >= mcf)
    # before the origin both are undefined
    assert np.isnan(model.mcf(-1.0, interp="linear")).all()
    assert np.isnan(model.mcf_cb(-1.0, interp="linear")).all()


def test_mcf_is_defined_at_negative_times_inside_a_negative_tl():
    model = NonParametricCounting.fit(
        [-3, -1, 2, 4], [1, 1, 1, 1], [0, 0, 0, 1], tl=-5
    )
    assert np.allclose(model.mcf([-4, -3, 0]), [0, 1, 2])
    assert np.isnan(model.mcf(-6)).all()
    assert np.allclose(model.mcf_cb([-4])[0], [0, 0])
    restored = NonParametricCounting.from_dict(model.to_dict())
    assert np.allclose(restored.mcf([-4, -3, 0]), [0, 1, 2])


def test_recurrent_event_data_to_xrd_keeps_integer_counts():
    data = RecurrentEventData(
        np.array([1, 2, 1]), np.array([1, 1, 2]), np.zeros(3), np.ones(3, int)
    )
    _, r, d = data.to_xrd()
    assert d.dtype.kind == "i" and r.dtype.kind == "i"


# ---------------------------------------------------------------------------
# The normal bounds and the robust (Lawless-Nadeau) variance.
# ---------------------------------------------------------------------------


def _heterogeneous_fleet(n_items=30, T=10.0, seed=0):
    rng = np.random.default_rng(seed)
    x, i, c = [], [], []
    for k in range(n_items):
        rate = rng.gamma(0.5, 2.0) * 0.5
        t = 0.0
        while True:
            t += rng.exponential(1 / rate)
            if t > T:
                break
            x.append(t), i.append(k), c.append(0)
        x.append(T), i.append(k), c.append(1)
    return np.array(x), np.array(i), np.array(c)


def test_mcf_normal_bounds_are_estimate_plus_minus_z_se():
    x, i, c = _heterogeneous_fleet()
    model = NonParametricCounting.fit(x, i, c)
    t = np.array([5.0])
    k = np.searchsorted(model.x, 5.0, side="right") - 1
    se = np.sqrt(model.var[k])
    lower, upper = model.mcf_cb(t, bound_type="normal")[0]
    assert lower == pytest.approx(model.mcf_hat[k] - 1.959964 * se)
    assert upper == pytest.approx(model.mcf_hat[k] + 1.959964 * se)


def test_mcf_variance_is_robust_to_heterogeneous_items():
    # Lawless-Nadeau: close to the sampling variance even when items have
    # very different rates; the per-step variance was ~8 times too small
    estimates, variances = [], []
    for seed in range(150):
        x, i, c = _heterogeneous_fleet(seed=seed)
        model = NonParametricCounting.fit(x, i, c)
        k = np.searchsorted(model.x, 7.0, side="right") - 1
        estimates.append(model.mcf_hat[k])
        variances.append(model.var[k])
    ratio = np.mean(variances) / np.var(estimates)
    assert 0.75 < ratio < 1.3


def test_mcf_variance_matches_the_per_step_form_for_single_events():
    # one event per item, no ties: no within-item covariance to add, so
    # the robust variance reduces to the per-step one
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    i = np.array([1, 2, 3, 4, 5])
    c = np.zeros(5, dtype=int)
    model = NonParametricCounting.fit(x, i, c)
    naive = NonParametricCounting.from_xrd(model.x, model.r, model.d)
    np.testing.assert_allclose(model.var, naive.var)


# ---------------------------------------------------------------------------
# The per-step variance of ``from_xrd`` with ties.
# ---------------------------------------------------------------------------


def test_from_xrd_per_step_variance_with_ties():
    # Two events among three items at risk: d (r - d) / r^3 = 2/27. The
    # old formula centred on 1/r and gave 1/9.
    model = NonParametricCounting.from_xrd([1.0], [3], [2])
    np.testing.assert_allclose(model.var, [2 / 27])


def test_from_xrd_single_event_steps_unchanged():
    r = np.array([5, 4, 3])
    model = NonParametricCounting.from_xrd([1.0, 2.0, 3.0], r, [1, 1, 1])
    np.testing.assert_allclose(model.var, np.cumsum((r - 1) / r**3))


def test_from_xrd_more_events_than_at_risk_gives_nan_variance():
    # The triple cannot say how three events shared out over two items.
    model = NonParametricCounting.from_xrd([1.0, 2.0], [4, 2], [1, 3])
    assert model.var[0] == pytest.approx(3 / 64)
    assert np.isnan(model.var[1])


# ---------------------------------------------------------------------------
# ``mcf_cb`` selects by query position before masking, and
# returns ``[lower, upper]`` (#285).
# ---------------------------------------------------------------------------


class TestMCFConfidenceBounds:
    @staticmethod
    def _model():
        # The items' event counts differ at the queried times: with the
        # Lawless-Nadeau variance, items that have had the same number of
        # events so far give zero variance (and zero-width bounds).
        return NonParametricCounting.fit(
            x=[4, 5, 6, 8, 10, 12], i=[1, 1, 2, 1, 1, 2], c=[0] * 6
        )

    def test_two_sided_off_grid_queries(self):
        # 285: masks used to zero the whole upper-bound column and crash
        # when queries outnumbered the two bound rows.
        m = self._model()
        out = np.asarray(m.mcf_cb(np.array([1.0, 5.5, 6.5, 20.0])))
        assert out.shape == (4, 2)
        # below the first time: [0, 0]; above the last: NaN
        assert np.all(out[0] == 0.0)
        assert np.all(np.isnan(out[3]))
        # in-range rows are finite with lower < upper
        for row in out[1:3]:
            assert np.all(np.isfinite(row))
            assert row[0] < row[1]

    def test_more_queries_than_bound_rows_no_crash(self):
        m = self._model()
        out = np.asarray(m.mcf_cb(np.array([1.0, 5.5, 20.0])))
        assert out.shape == (3, 2)

    def test_one_sided_out_of_range(self):
        m = self._model()
        low = np.asarray(
            m.mcf_cb(np.array([1.0, 6.5, 20.0]), bound="lower"), dtype=float
        )
        assert low[0] == 0.0
        assert np.isfinite(low[1])
        assert np.isnan(low[2])

    def test_column_order_lower_upper(self):
        # 285: two-sided order used to be [upper, lower], inconsistent
        # with the parametric cif_cb.
        m = self._model()
        out = np.asarray(m.mcf_cb(np.array([8.5])))
        assert out[0, 0] < out[0, 1]


def test_666_fitted_mcf_repr_says_what_it_is():
    model = NonParametricCounting.fit(
        [2, 4, 6, 3, 5], [1, 1, 1, 2, 2], [0, 0, 1, 0, 1]
    )
    text = repr(model)
    assert "object at" not in text
    assert "Mean cumulative function" in text
    assert "2 items: 3 events at 3 unique times" in text
