import numpy as np
import pytest

from surpyval.tests._helpers import no_warnings
from surpyval.univariate.nonparametric.plotting_positions import (
    plotting_positions,
)


def test_filliben_heuristic():
    # Values from the Filliben (1975) estimate:
    # F[0] = 1 - 0.5**(1/N), F[-1] = 0.5**(1/N),
    # F[i] = (i + 1 - 0.3175) / (N + 0.365) otherwise.
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    _, _, _, F = plotting_positions(x, heuristic="Filliben")
    expected = [
        0.08299596,
        0.20113568,
        0.32068141,
        0.44022714,
        0.55977286,
        0.67931859,
        0.79886432,
        0.91700404,
    ]
    assert np.allclose(F, expected, atol=1e-7)


def test_ecdf_adj_heuristic_is_accepted():
    x = np.array([1.0, 2, 3, 4, 5])
    _, _, _, F = plotting_positions(x, heuristic="ECDF_Adj")
    expected = (np.arange(1, 6) - 0) / (5 + 1)
    assert np.allclose(F, expected, atol=1e-12)


def test_unknown_heuristic_rejected():
    with pytest.raises(ValueError):
        plotting_positions(np.array([1.0, 2, 3]), heuristic="NotAMethod")


def test_blom_heuristic_unchanged():
    x = np.array([1.0, 2, 3, 4, 5])
    _, _, _, F = plotting_positions(x, heuristic="Blom")
    expected = (np.arange(1, 6) - 0.375) / (5 + 0.25)
    assert np.allclose(F, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# ``'Benard'`` plotting positions use Benard's
# (i - 0.3) / (N + 0.4).
# ---------------------------------------------------------------------------


class TestBenard:
    def test_benard_is_the_median_rank_approximation(self):
        x = np.arange(1.0, 11.0)
        N = x.size
        _, _, _, F = plotting_positions(x, heuristic="Benard")
        i = np.arange(1, N + 1)
        np.testing.assert_allclose(F, (i - 0.3) / (N + 0.4))
        _, _, _, F_median = plotting_positions(x, heuristic="Median")
        np.testing.assert_allclose(F, F_median)


# ---------------------------------------------------------------------------
# Filliben's end points with censoring; Modal needs two items.
# ---------------------------------------------------------------------------


def test_filliben_end_points_go_to_extreme_ranks_only():
    x = [1, 2, 3, 4, 5]
    N = 5
    complete = plotting_positions(x, heuristic="Filliben")[3]
    assert complete[0] == pytest.approx(1 - 0.5 ** (1 / N))
    assert complete[-1] == pytest.approx(0.5 ** (1 / N))

    # Last item censored: the failure at rank 4 keeps the interior value
    # and the censored row carries it forward (it used to get 0.8706).
    last = plotting_positions(x, c=[0, 0, 0, 0, 1], heuristic="Filliben")[3]
    interior = (4 - 0.3175) / (N + 0.365)
    np.testing.assert_allclose(last[3:], [interior, interior])

    # First item censored: 0 before the first failure, not NaN.
    first = plotting_positions(x, c=[1, 0, 0, 0, 0], heuristic="Filliben")[3]
    assert first[0] == 0
    assert np.isfinite(first).all()


def test_modal_needs_two_items():
    with pytest.raises(ValueError, match="at least two"):
        no_warnings(plotting_positions, [5.0], heuristic="Modal")


# The Lieblein-Zelen ball bearings (millions of revolutions), 23 failures.
BEARINGS = np.array(
    [
        17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96,
        54.12, 55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64,
        105.12, 105.84, 127.92, 128.04, 173.40,
    ]
)  # fmt: skip


def test_exact_median_ranks_are_the_beta_medians():
    # The median of the i-th of N uniform order statistics, Beta(i,
    # N - i + 1): for N = 5 the middle one is 0.5 and the first is
    # 1 - 0.5**(1/5).
    from scipy.stats import beta

    x = np.arange(1.0, 6.0)
    _, _, _, F = plotting_positions(x, heuristic="Exact")
    np.testing.assert_allclose(
        F, beta.ppf(0.5, np.arange(1, 6), 5 - np.arange(1, 6) + 1), rtol=1e-12
    )
    assert F[2] == pytest.approx(0.5, abs=1e-15)
    assert F[0] == pytest.approx(1 - 0.5 ** (1 / 5), rel=1e-12)


@pytest.mark.parametrize("name", ["exact", "Exact median", "EXACT-MEDIAN"])
def test_exact_median_ranks_by_any_of_their_names(name):
    _, _, _, F = plotting_positions(BEARINGS, heuristic=name)
    _, _, _, G = plotting_positions(BEARINGS, heuristic="Exact")
    np.testing.assert_array_equal(F, G)


def test_exact_median_ranks_match_weibullr_with_suspensions():
    # WeibullR 1.2.4, getPPP(failures, suspensions) (ppos = "beta",
    # Johnson's adjusted ranks): every fourth bearing suspended at its
    # time.
    c = np.zeros(BEARINGS.size, int)
    c[np.arange(2, BEARINGS.size, 4)] = 1
    _, _, d, F = plotting_positions(BEARINGS, c, heuristic="Exact")
    weibullr = [
        0.0296872472502, 0.0719057007042, 0.116610702314, 0.161418769236,
        0.206263712459, 0.25376506044, 0.30127641045, 0.348793438214,
        0.399969324534, 0.451147353169, 0.502326323037, 0.559191752394,
        0.616055745811, 0.672916499962, 0.741140859152, 0.809345114987,
        0.877494600142,
    ]  # fmt: skip
    np.testing.assert_allclose(F[d == 1], weibullr, rtol=1e-10)
    # a suspension carries the previous failure's position
    np.testing.assert_array_equal(F[2], F[1])


@pytest.mark.parametrize(
    "case, expected",
    [
        # WeibullR 1.2.4, MRRw2p (rank regression on X, exact median ranks)
        ("complete", (80.9470163809, 2.25501242274)),
        ("stopped at 110", (77.7857958302, 2.41114215068)),
        ("interleaved", (90.9707880768, 2.19540667674)),
    ],
)
def test_exact_median_rank_regression_matches_weibullr(case, expected):
    import surpyval as sp

    x, c = BEARINGS, np.zeros(BEARINGS.size, int)
    if case == "stopped at 110":
        c = (BEARINGS > 110).astype(int)
        x = np.where(c == 1, 110.0, BEARINGS)
    elif case == "interleaved":
        c = c.copy()
        c[np.arange(2, BEARINGS.size, 4)] = 1
    model = sp.Weibull.fit(x, c, how="MPP", rr="x", heuristic="Exact")
    np.testing.assert_allclose(model.params, expected, rtol=1e-9)


def test_median_and_benard_are_the_approximation_of_the_exact_ranks():
    # Benard's (i - 0.3) / (N + 0.4) is within half a percent of the
    # exact median rank here, but not equal to it.
    _, _, _, exact = plotting_positions(BEARINGS, heuristic="Exact")
    _, _, _, benard = plotting_positions(BEARINGS, heuristic="Median")
    assert np.max(np.abs(benard - exact)) < 0.002
    assert not np.allclose(benard, exact, rtol=1e-6)
