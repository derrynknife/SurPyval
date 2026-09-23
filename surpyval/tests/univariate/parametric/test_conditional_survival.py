"""
Numerical coverage for ``cs``, the conditional survival function.

Eleven distributions used to carry their own ``cs``, each with the same
one-line body as ``ParametricFitter.cs`` and a docstring example that
pinned its numbers. The bodies were duplication -- and duplication that
had already rotted, since Gamma's docstring stated the *exponential*
survival function above a body that computed the ratio correctly. The
overrides are gone; the numbers they pinned are here, so deleting the
docstrings did not delete the only per-distribution check on ``cs``.

Expected values are the ones those docstrings recorded, which the
doctest run verified on every supported interpreter.
"""

import numpy as np
import pytest

from surpyval import (
    Beta,
    Exponential,
    ExpoWeibull,
    Gamma,
    LogLogistic,
    LogNormal,
    Normal,
    Rayleigh,
    Uniform,
    Weibull,
)

X = np.array([1, 2, 3, 4, 5])

# (distribution, x, X, params, expected) -- lifted from the docstrings
# the overrides used to carry.
CASES = [
    (
        Beta,
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        0.4,
        (3, 4),
        [0.6315219, 0.32921811, 0.12946429, 0.03115814, 0.00233319],
    ),
    (
        ExpoWeibull,
        X,
        1,
        (3, 4, 1.2),
        [
            8.77367129e-01,
            4.25451775e-01,
            5.09266354e-02,
            5.37452200e-04,
            1.35732908e-07,
        ],
    ),
    (
        Exponential,
        X,
        5,
        (3,),
        [
            4.97870684e-02,
            2.47875218e-03,
            1.23409804e-04,
            6.14421235e-06,
            3.05902321e-07,
        ],
    ),
    (
        Gamma,
        X,
        5,
        (3, 4),
        [
            2.59402488e-02,
            6.39048747e-04,
            1.51519143e-05,
            3.48776510e-07,
            7.79933496e-09,
        ],
    ),
    (
        LogLogistic,
        X,
        5,
        (3, 4),
        [0.51270879, 0.28444803, 0.16902083, 0.10629329, 0.07003273],
    ),
    (
        LogNormal,
        X,
        5,
        (3, 4),
        [0.97287811, 0.9496515, 0.92933892, 0.91129122, 0.89505592],
    ),
    (
        Normal,
        X,
        5,
        (3, 4),
        [0.73452116, 0.51421702, 0.34242113, 0.2165286, 0.1298356],
    ),
    (
        Rayleigh,
        X,
        5,
        (3,),
        [0.54274748, 0.26359714, 0.11455884, 0.04455143, 0.01550385],
    ),
    (
        Uniform,
        X,
        4,
        (0, 10),
        [0.83333333, 0.66666667, 0.5, 0.33333333, 0.16666667],
    ),
    (
        Weibull,
        X,
        5,
        (3, 4),
        [
            2.52537548e-04,
            3.00394073e-10,
            2.45288508e-19,
            1.48999440e-32,
            5.42544000e-51,
        ],
    ),
]

IDS = [c[0].name for c in CASES]


@pytest.mark.parametrize("dist, x, cond, params, expected", CASES, ids=IDS)
def test_cs_matches_the_documented_values(dist, x, cond, params, expected):
    # The docstrings printed eight decimal places, so a value like
    # 0.01550385 pins the result to about 2e-7 relative -- atol carries
    # the fixed-decimal entries, rtol the ones in scientific notation.
    got = np.asarray(dist.cs(x, cond, *params), dtype=float)
    np.testing.assert_allclose(got, np.array(expected), rtol=1e-6, atol=5e-9)


@pytest.mark.parametrize("dist, x, cond, params, expected", CASES, ids=IDS)
def test_cs_equals_the_survival_ratio(dist, x, cond, params, expected):
    # The property the inherited implementation encodes. Exponential is
    # included deliberately: its override returns sf(x) on the strength
    # of memorylessness, and this is what checks that shortcut is the
    # same function the others compute the long way.
    got = np.asarray(dist.cs(x, cond, *params), dtype=float)
    ratio = np.asarray(
        dist.sf(x + cond, *params) / dist.sf(cond, *params), dtype=float
    )
    np.testing.assert_allclose(got, ratio, rtol=1e-9)


def test_cs_at_zero_is_one():
    # Surviving a further nothing is certain, whatever has been survived.
    for dist, _, cond, params, _ in CASES:
        got = np.asarray(dist.cs(0.0, cond, *params), dtype=float)
        np.testing.assert_allclose(got, 1.0, rtol=1e-9, atol=1e-12)


def test_exponential_cs_is_memoryless():
    # The reason Exponential keeps an override. Conditioning on any
    # amount of prior survival leaves the distribution unchanged.
    x = np.array([0.5, 1.0, 2.0, 4.0])
    base = np.asarray(Exponential.sf(x, 3), dtype=float)
    for cond in (0.0, 1.0, 10.0, 100.0):
        got = np.asarray(Exponential.cs(x, cond, 3), dtype=float)
        np.testing.assert_allclose(got, base, rtol=1e-12)


def test_discrete_distributions_inherit_a_working_cs():
    # These never defined cs and reach the base implementation. Before
    # the base gained one they raised AttributeError.
    from surpyval import Geometric, NegativeBinomial, Poisson

    for dist, params in (
        (Poisson, (3.0,)),
        (Geometric, (0.3,)),
        (NegativeBinomial, (2.0, 0.4)),
    ):
        got = np.asarray(dist.cs(np.array([1, 2, 3]), 2, *params), dtype=float)
        assert got.shape == (3,)
        assert np.all(np.isfinite(got))
        assert np.all((got >= 0) & (got <= 1 + 1e-12))
