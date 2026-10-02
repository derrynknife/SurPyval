"""The incomplete beta's tails where the continued fraction does not
converge (#520).

``_beta_logs`` takes a small tail from its own continued fraction. Where
that tail's x is near 1, far above the fraction's region (x below
(a + 1) / (a + b + 2)), as it is when the other side is near 1 because
its first shape is tiny (a NegativeBinomial ``r`` of 1e-172, which a
search from a far start reached), the fraction ran to its 100,000-term
limit, 4.6 s a call, and returned a value 40% off. The other side's
power series is used there now. The references are mpmath's, at 400
digits.
"""

import time

import numpy as np
import pytest
from autograd import grad

import surpyval as sp
from surpyval.utils import autograd_gamma_compat as ag

R_TINY = 2.34763792e-172


@pytest.mark.parametrize(
    "k, p, reference",
    [
        (2.0, 1e-17, -391.54985924175463101),
        (6.0, 1e-17, -391.58408272342430333),
        (2.0, 1e-300, -388.65486008996170135),
    ],
)
def test_negative_binomial_tail_at_a_tiny_r(k, p, reference):
    # log R(k) = log(1 - I_p(r, k)): about r ln(1 / p), from the power
    # series of I_p(r, k), whose log is of size r. The continued fraction
    # of the upper tail, at 1 - p = 1, gave -392.05 for -391.55.
    start = time.perf_counter()
    got = sp.NegativeBinomial.log_sf(np.array([k]), R_TINY, p)[0]
    assert time.perf_counter() - start < 1.0
    assert got == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize(
    "a, b, x, lower, upper",
    [
        # log I_x(a, b) and log(1 - I_x(a, b)) by mpmath
        (R_TINY, 3.0, 1e-17, -8.83743564517727e-171, -391.563054152802),
        (R_TINY, 3.0, 1e-300, -1.61816936662187e-169, -388.655585226308),
        (1e-5, 3.0, 1e-6, -0.00012315518807906274592, -9.0021268831183564),
    ],
)
def test_the_logs_where_the_fraction_does_not_converge(a, b, x, lower, upper):
    assert ag.betaincln(a, b, x) == pytest.approx(lower, rel=1e-12)
    assert ag.betainccln(a, b, x) == pytest.approx(upper, rel=1e-12)


def test_the_gradient_there_matches_mpmath():
    # d/da and d/db are central differences of the value (``_make_dab_
    # primitives``), d/dx analytical; mpmath's derivatives at 60 digits.
    a, b, x = 1e-5, 3.0, 1e-6
    assert grad(ag.betainccln, 0)(a, b, x) == pytest.approx(
        99993.893112071685214, rel=1e-7
    )
    assert grad(ag.betainccln, 1)(a, b, x) == pytest.approx(
        -0.03206588240171898434, rel=1e-7
    )
    assert grad(ag.betainccln, 2)(a, b, x) == pytest.approx(
        -81193.203421695007418, rel=1e-12
    )


def test_an_unconverged_fraction_is_nan():
    # It used to return wherever it had got to (#473).
    out = ag.beta_cf(np.array([3.0, 3.0]), R_TINY, np.array([1.0, 0.1]), 50)
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_the_front_factor_series_meets_the_gammas():
    # ln Gamma(a + b) - ln Gamma(b) - ln Gamma(1 + a): the Taylor series
    # just below 1e-4 min(1, b) agrees with the gammas (exactly 0 at b = 1,
    # where Gamma(1 + a) / Gamma(1 + a) is 1), and at a tiny ``a`` it is
    # a (psi(b) + euler), where the gammas give rounding.
    b = np.array([0.5, 3.0, 50.0])
    a = 0.99e-4 * np.minimum(1.0, b)
    np.testing.assert_allclose(
        ag._log_beta_front(a, b),
        ag.log_gamma_ratio(b, a) - ag._lngamma1p(a),
        rtol=1e-9,
    )
    assert abs(ag._log_beta_front(np.array([1e-5]), np.array([1.0]))) < 1e-20
    np.testing.assert_allclose(
        ag._log_beta_front(np.full(3, 1e-300), b),
        1e-300 * (ag._sc_digamma(b) + ag._EULER),
        rtol=1e-15,
    )
