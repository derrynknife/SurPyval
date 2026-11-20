import numpy as np
import pytest

from surpyval.recurrent import HPP, CrowAMSAA, ProportionalIntensityHPP
from surpyval.recurrent.parametric.nhpp_fitter import nhpp_log_likelihood
from surpyval.utils.recurrent_utils import handle_xicn


def test_hpp_interval_censoring_matches_analytic_rate():
    # Interval-censored counts: n_k events in [a_k, b_k]. The HPP rate MLE is
    # total events / total exposure. Here 6 events over a width-15 window.
    x = [[0, 5], [5, 10], [10, 15]]
    c = [2, 2, 2]
    n = [2, 3, 1]
    i = [1, 1, 1]
    model = HPP.fit(x, i, c=c, n=n)
    assert np.isclose(model.params[0], 6.0 / 15.0)


def test_crow_amsaa_interval_censoring_fits():
    x = [[0, 5], [5, 10], [10, 20]]
    c = [2, 2, 2]
    n = [1, 2, 2]
    i = [1, 1, 1]
    model = CrowAMSAA.fit(x, i, c=c, n=n)
    assert np.all(np.isfinite(model.params))


def test_interval_left_must_not_exceed_right():
    # A reversed interval [5, 1] is invalid regardless of input type. The
    # check is shared with the univariate handler via coerce_xcnt_x.
    with pytest.raises(ValueError, match="less than or equal to right"):
        handle_xicn(np.array([[5.0, 1.0]]), np.array([1]), c=np.array([2]))
    with pytest.raises(ValueError, match="less than or equal to right"):
        handle_xicn([[5.0, 1.0]], [1], c=[2])


def test_single_interval_row_does_not_crash():
    # A single interval row must not trip the old row-0-vs-row-1 check.
    data = handle_xicn([[1.0, 5.0]], [1], c=[2], n=[2])
    assert data.x.shape == (1, 2)


def test_interval_censoring_respects_truncation_window():
    # An interval beginning below the left-truncation bound is rejected.
    x = np.array([[1.0, 5.0], [6.0, 9.0]])
    with pytest.raises(ValueError, match="before its observation began"):
        HPP.fit(
            x, np.array([1, 1]), c=np.array([2, 2]), n=np.array([1, 1]), tl=3.0
        )


# ---------------------------------------------------------------------------
# A left-censored count with a left truncation covers the
# window from entry.
# ---------------------------------------------------------------------------


def test_hpp_left_censored_count_covers_the_window_from_entry():
    # three events in (5, 10]: rate 3 / 5, not 3 / 10
    model = HPP.fit([10], [1], [-1], n=[3], tl=[5])
    assert np.isclose(model.params[0], 0.6)


def test_nhpp_left_censored_count_uses_cif_from_entry():
    data = handle_xicn([10, 20], [1, 1], [-1, 1], n=[3, 1], tl=5)
    neg_ll = CrowAMSAA.create_negll_func(data)
    alpha, beta = 8.0, 1.3

    def cif(t):
        return (t / alpha) ** beta

    lam = cif(10) - cif(5)
    expected = 3 * np.log(lam) - lam - np.log(6) - (cif(20) - cif(10))
    assert np.isclose(-neg_ll(np.array([alpha, beta])), expected)


# ---------------------------------------------------------------------------
# Degenerate interval pairs are exact times (#288).
# ---------------------------------------------------------------------------


class TestNHPPDegenerateIntervals:
    def test_degenerate_pairs_match_1d(self):
        # 288: x_prev typo cancelled the exposure term for 2-D input
        # without interval rows.
        np.random.seed(5)
        u = np.sort(np.random.uniform(size=40))
        t_ev = 100.0 * u ** (1 / 3)
        m1 = CrowAMSAA.fit(
            x=t_ev, i=np.ones(40, dtype=int), c=np.zeros(40, dtype=int)
        )
        m2 = CrowAMSAA.fit(
            x=np.column_stack([t_ev, t_ev]), i=np.ones(40, dtype=int)
        )
        np.testing.assert_allclose(m1.params, m2.params, rtol=1e-8)


# ---------------------------------------------------------------------------
# The HPP and the proportional-intensity HPP take the sums their likelihood
# needs from the data once, before the search; with a constant rate it is
# the general NHPP likelihood, term for term.
# ---------------------------------------------------------------------------


def _every_piece_of_the_likelihood():
    """Rows of every kind the NHPP likelihood splits: exact events, right
    censoring, a left-censored count, interval counts, and items followed
    to a right-truncation time with (item 4) and without (item 5) a
    failure."""
    t = [2, 5, 8, 4, 6, 9, 0, 3, 7, 1, 2.5, 3]
    x = np.array([[v, v] for v in t], dtype=float)
    x[6], x[7] = [0.0, 3.0], [3.0, 7.0]
    i = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5])
    c = np.array([0, 0, 1, -1, 0, 1, 2, 2, 0, 0, 0, 1])
    n = np.array([1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1])
    tr = np.array([np.inf] * 9 + [4.0, 4.0, 5.0])
    Z = np.array([0.5, 0.5, 0.5, -1, -1, -1, 2, 2, 2, 0, 0, 1.0])
    return x, i, c, n, tr, Z.reshape(-1, 1)


def test_hpp_likelihood_is_the_nhpp_likelihood_at_a_constant_rate():
    x, i, c, n, tr, _ = _every_piece_of_the_likelihood()
    data = handle_xicn(x, i, c, n, tr=tr)
    s = data.split_for_nhpp_likelihood()
    assert all(s[key].size for key in ("x_left", "x_i_l", "sel_to", "gap_to"))
    neg_ll = HPP.create_negll_func(data)
    for rate in (0.05, 0.4, 3.0):
        expected = nhpp_log_likelihood(
            lambda t: rate * t, lambda t: np.log(rate) + 0.0 * t, s
        )
        assert np.isclose(-neg_ll(np.array([np.log(rate)])), expected)


def test_pi_hpp_likelihood_is_the_pi_nhpp_likelihood_at_a_constant_rate():
    x, i, c, n, tr, Z = _every_piece_of_the_likelihood()
    data = handle_xicn(x, i, c, n, tr=tr, Z=Z)
    s = data.split_for_nhpp_likelihood()
    neg_ll = ProportionalIntensityHPP.create_negll_func(data)
    masks = {k: s["mask_" + k] for k in ("o", "right", "left", "i")}
    for rate, beta in ((0.4, -0.3), (1.5, 0.8)):
        eta = {k: data.Z[m] @ [beta] for k, m in masks.items()}
        eta["sel"] = data.Z[s["sel_idx"]] @ [beta]
        eta["gap"] = data.Z[s["gap_idx"]] @ [beta]
        expected = nhpp_log_likelihood(
            lambda t: rate * t, lambda t: np.log(rate) + 0.0 * t, s, eta
        )
        got = -neg_ll(np.array([np.log(rate), beta]))
        assert np.isclose(got, expected)
