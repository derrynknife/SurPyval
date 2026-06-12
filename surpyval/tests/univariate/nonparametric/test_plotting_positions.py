import numpy as np
import pytest

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
