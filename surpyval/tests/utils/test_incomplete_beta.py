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
    assert got == pytest.approx(reference, rel=1e-12, abs=0)


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
    assert ag.betaincln(a, b, x) == pytest.approx(lower, rel=1e-12, abs=0)
    assert ag.betainccln(a, b, x) == pytest.approx(upper, rel=1e-12, abs=0)


def test_the_gradient_there_matches_mpmath():
    # d/da and d/db are analytic (``_beta_log_shape_grad``, #621) where
    # the continued fraction converges, and here, where it does not, the
    # central differences of the value; d/dx analytical; mpmath's
    # derivatives at 60 digits.
    a, b, x = 1e-5, 3.0, 1e-6
    assert grad(ag.betainccln, 0)(a, b, x) == pytest.approx(
        99993.893112071685214, rel=1e-7, abs=0
    )
    assert grad(ag.betainccln, 1)(a, b, x) == pytest.approx(
        -0.03206588240171898434, rel=1e-7, abs=0
    )
    assert grad(ag.betainccln, 2)(a, b, x) == pytest.approx(
        -81193.203421695007418, rel=1e-12, abs=0
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


# #621: d/da and d/db of log I_x(a, b) and log(1 - I_x(a, b)), against
# mpmath (60 digits: central differences of the tails' quadrature, after
# t = w^(1/a), in steps of 1e-15). The five-point differences of the value
# they replace were 1e-12 to 2e-4 off.
SHAPE_DERIVATIVES = [
    (2.0, 5.0, 0.3, False, -0.43533097273881922, 0.17150846301807766),
    (2.0, 5.0, 0.3, True, 0.60073964721434123, -0.23667494393873236),
    (0.3, 0.7, 1e-06, True, 0.19617473297022222, -0.0088657720656168828),
    (40.0, 60.0, 0.95, True, 0.87343217543626623, -2.4987940165753201),
    (7.5, 0.4, 1 - 1e-6, False, -0.00055392755277876375, 0.1173325871737577),
    (
        667370.7909956266,
        0.010331615383403254,
        1.8780123823566156e-14,
        False,
        -31.605978810714483,
        110.76173355730035,
    ),
    (
        0.05317286695406665,
        731067.9885594342,
        0.9999999999889172,
        True,
        32.801838947925024,
        -25.225627932874331,
    ),
]


@pytest.mark.parametrize("a, b, x, upper, d_a, d_b", SHAPE_DERIVATIVES)
def test_621_shape_derivatives_against_mpmath(a, b, x, upper, d_a, d_b):
    f = ag.betainccln if upper else ag.betaincln
    assert grad(f, 0)(a, b, x) == pytest.approx(d_a, rel=1e-13, abs=0)
    assert grad(f, 1)(a, b, x) == pytest.approx(d_b, rel=1e-13, abs=0)


def test_621_shape_derivatives_of_arrays_and_edges():
    # Elementwise over an array, 0 at the edges (where the log is 0 or
    # -inf whatever the shapes), and the same at a repeated call (the
    # derivatives in a and b come from one pass)
    x = np.array([0.0, 0.3, 0.95, 1.0])
    g_a, g_b = ag._beta_log_shape_grad(40.0, 60.0, x, True)
    assert g_a[0] == g_b[0] == g_a[-1] == g_b[-1] == 0.0
    assert g_a[2] == pytest.approx(0.87343217543626623, rel=1e-13)
    again = ag._beta_log_shape_grad(40.0, 60.0, x, True)
    np.testing.assert_array_equal(again[0], g_a)
    single = ag._beta_log_shape_grad(40.0, 60.0, 0.3, True)
    assert single[1] == g_b[1]


@pytest.mark.parametrize("y", [1.0, 4.0, 10.0, 0.5, 37.0])
def test_665_log_gamma_ratio_gradient_at_whole_numbers(y):
    # d/dy of ln G(y + a) - ln G(y) is digamma(y + a) - digamma(y). At
    # y = 1, ..., 10 the shifted argument met its floor of 10 exactly, and
    # autograd's maximum split the gradient there in half: a
    # NegativeBinomial fit from r = 4 could not move.
    from scipy.special import digamma

    a = 2.5
    expected = digamma(y + a) - digamma(y)
    assert grad(ag.log_gamma_ratio, 0)(y, a) == pytest.approx(
        expected, rel=1e-12
    )
