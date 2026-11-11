"""``scale_free_hessian``: a finite-difference Hessian whose evaluation
points come from the function, not from the size of the parameters."""

import numpy as np
import pytest

from surpyval.utils.linalg import (
    delta_method_se,
    numerical_hessian,
    scale_free_hessian,
)

A = np.array([[2.0, 3.0, 0.5], [3.0, 10.0, -1.0], [0.5, -1.0, 4.0]])


def _quadratic(v):
    return 7.0 + 0.5 * v @ A @ v


def test_exact_for_a_quadratic():
    got = scale_free_hessian(_quadratic, np.zeros(3))
    np.testing.assert_allclose(got, A, rtol=1e-8)


@pytest.mark.parametrize("scales", [(1.0, 1e-6, 1e6), (1e-9, 1.0, 3e-4)])
def test_a_rescaled_parameter_rescales_the_hessian_and_nothing_else(scales):
    # f(v) written in w = v / s: the Hessian is S A S exactly.
    s = np.array(scales)
    got = scale_free_hessian(lambda w: _quadratic(w * s), np.zeros(3))
    np.testing.assert_allclose(got, A * np.outer(s, s), rtol=1e-8)


def test_a_shifted_parameter_changes_nothing():
    # Away from 0 the plain rule's step grows with |x|; this one's does not.
    shift = np.array([1e6, -40.0, 3e-5])
    got = scale_free_hessian(lambda v: _quadratic(v - shift), shift)
    np.testing.assert_allclose(got, A, rtol=1e-6)


def _poisson_power_law(log_scale):
    """The negative log-likelihood of a power-law process, cif = b t^a,
    observed to 60 * k with events at 3, 9, 20, 35, 56 (times k), in
    (a, b); its Hessian in closed form is below."""
    t = np.array([3.0, 9.0, 20.0, 35.0, 56.0]) * log_scale
    end = 60.0 * log_scale

    def neg_ll(v):
        a, b = v
        return b * end**a - (
            np.log(a * b) * t.size + (a - 1) * np.log(t).sum()
        )

    def hessian(v):
        a, b = v
        n, log_end = t.size, np.log(end)
        return np.array(
            [
                [b * end**a * log_end**2 + n / a**2, end**a * log_end],
                [end**a * log_end, n / b**2],
            ]
        )

    return neg_ll, hessian, t.size, end


@pytest.mark.parametrize("k", [1.0, 730.0, 730.0 * 3600.0, 1.0 / 12.0])
def test_power_law_process_in_any_time_unit(k):
    """At the maximum (a = n / sum(log(end / t)), b = n / end^a) the
    Hessian is known in closed form. In hours b is 3e-4 and in seconds
    1e-6, where the plain rule's step, floored at 1e-2 of a unit, is far
    too long."""
    neg_ll, hessian, n, end = _poisson_power_law(k)
    t = np.array([3.0, 9.0, 20.0, 35.0, 56.0]) * k
    a = n / np.log(end / t).sum()
    v = np.array([a, n / end**a])
    exact = np.sqrt(np.diag(np.linalg.inv(hessian(v))))
    got = np.sqrt(np.diag(np.linalg.inv(scale_free_hessian(neg_ll, v))))
    np.testing.assert_allclose(got, exact, rtol=2e-6)
    # The standard error of the exponent does not depend on the unit.
    months = _poisson_power_law(1.0)
    v1 = np.array([a, n / 60.0**a])
    ref = np.sqrt(np.diag(np.linalg.inv(scale_free_hessian(months[0], v1))))
    assert got[0] == pytest.approx(ref[0], rel=2e-6)


def test_a_boundary_or_a_saddle_falls_back_to_the_plain_rule():
    # No rise on one side of x (a boundary: nan beyond it)
    def one_sided(v):
        return np.nan if v[0] < 0 else v[0] + v[1] ** 2

    x = np.array([0.0, 0.0])
    np.testing.assert_array_equal(
        scale_free_hessian(one_sided, x), numerical_hessian(one_sided, x)
    )

    # Not a minimum: the first Hessian is not positive definite
    def saddle(v):
        return v[0] ** 2 + 4 * v[0] * v[1] + v[1] ** 2

    np.testing.assert_array_equal(
        scale_free_hessian(saddle, x), numerical_hessian(saddle, x)
    )
    # Not finite at x
    np.testing.assert_array_equal(
        scale_free_hessian(lambda v: np.inf, x),
        numerical_hessian(lambda v: np.inf, x),
    )


# -- delta_method_se(scale_free=True) ---------------------------------------


def _cox_lewis_cif(x):
    # exp(a) * expm1(b x) / b: a slope b of 1e-8 per second is ordinary.
    def f(theta):
        a, b = theta
        return np.exp(a) * np.expm1(b * x) / b

    return f


@pytest.mark.parametrize("k", [1.0, 730.0, 730.0 * 3600.0])
def test_delta_method_steps_scale_with_the_standard_errors(k):
    """The same curve and covariance written in another time unit
    (``x' = k x``, ``a' = a - log k``, ``b' = b / k``) give the same
    standard error of the curve, to rounding."""
    a, b = -1.95, -0.024
    cov = np.array([[0.33, -0.009], [-0.009, 4.1e-4]])
    x = np.array([10.0, 30.0, 59.0])
    ref = delta_method_se(_cox_lewis_cif(x), np.array([a, b]), cov)
    J = np.diag([1.0, 1.0 / k])
    got = delta_method_se(
        _cox_lewis_cif(x * k),
        np.array([a - np.log(k), b / k]),
        J @ cov @ J.T,
        scale_free=True,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-8)


def test_delta_method_default_step_is_unchanged_and_not_unit_free():
    # The default rule's floor (6e-8 of a unit) is longer than a slope
    # of 9e-9: in seconds its answer is far off. scale_free leaves the
    # default alone where it is not asked for.
    k = 730.0 * 3600.0
    cov = np.array([[0.33, -0.009], [-0.009, 4.1e-4]])
    J = np.diag([1.0, 1.0 / k])
    x = np.array([59.0])
    ref = delta_method_se(
        _cox_lewis_cif(x), np.array([-1.95, -0.024]), cov, scale_free=True
    )
    plain = delta_method_se(
        _cox_lewis_cif(x * k),
        np.array([-1.95 - np.log(k), -0.024 / k]),
        J @ cov @ J.T,
    )
    assert abs(plain[0] / ref[0] - 1) > 1e-2


def test_delta_method_scale_free_falls_back_where_a_variance_is_zero():
    # A held parameter (zero variance) keeps the default step; its
    # column is multiplied by zero either way.
    def f(theta):
        return np.array([theta[0] * 2.0 + theta[1] ** 2])

    cov = np.array([[0.25, 0.0], [0.0, 0.0]])
    got = delta_method_se(f, np.array([1.0, 3.0]), cov, scale_free=True)
    np.testing.assert_allclose(got, [1.0], rtol=1e-9)
