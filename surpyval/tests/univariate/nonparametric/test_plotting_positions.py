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
