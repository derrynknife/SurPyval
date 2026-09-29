"""Gradients of the incomplete gamma/beta primitives have the shape of the
argument they are taken with respect to, whatever the broadcasting."""

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad, hessian

from surpyval.utils.autograd_gamma_compat import (
    betainc,
    betaincln,
    gammainc,
    gammainccln,
    gammaincln,
)

A = np.array([0.7, 1.5, 2.5, 4.0])
X01 = np.array([0.1, 0.3, 0.6, 0.9])
X = np.array([0.4, 1.2, 2.0, 3.5])


def _fd(f, v, h=1e-6):
    return (f(v + h) - f(v - h)) / (2 * h)


@pytest.mark.parametrize("func", [gammainc, gammaincln, gammainccln])
def test_gamma_family_scalar_argument_against_array_partner(func):
    # scalar x, array a (the NegativeBinomial-style call that used to fail)
    g = grad(lambda x: anp.sum(func(A, x)))(1.3)
    assert np.shape(g) == ()
    assert g == pytest.approx(_fd(lambda x: np.sum(func(A, x)), 1.3), rel=1e-5)
    # scalar a, array x
    g = grad(lambda a: anp.sum(func(a, X)))(1.7)
    assert np.shape(g) == ()
    assert g == pytest.approx(_fd(lambda a: np.sum(func(a, X)), 1.7), rel=1e-4)
    # array a, array x: elementwise
    g = grad(lambda a: anp.sum(func(a, X)))(A)
    assert g.shape == A.shape


@pytest.mark.parametrize("func", [betainc, betaincln])
def test_beta_family_scalar_argument_against_array_partners(func):
    g = grad(lambda p: anp.sum(func(A, A + 1.0, p)))(0.4)
    assert np.shape(g) == ()
    assert g == pytest.approx(
        _fd(lambda p: np.sum(func(A, A + 1.0, p)), 0.4), rel=1e-5
    )
    for k in (0, 1):

        def f(s, k=k):
            args = [A, A + 1.0, X01]
            args[k] = s
            return anp.sum(func(*args))

        g = grad(f)(1.9)
        assert np.shape(g) == ()
        assert g == pytest.approx(_fd(f, 1.9), rel=1e-4)


def test_second_derivatives_keep_their_shapes():
    # Hessian through the traced shape-derivative with a scalar parameter
    # against array data, and a mixed parameter/x pair
    def f(v):
        return anp.sum(betainc(v[0], A + 1.0, v[1] * X01))

    h = hessian(f)(np.array([1.5, 0.8]))
    assert h.shape == (2, 2)
    assert np.all(np.isfinite(h))
    assert h[0, 1] == pytest.approx(h[1, 0], rel=1e-3, abs=1e-8)
