"""What a parametric fit costs: the likelihood evaluations it makes.

#593: the optimisers were given the likelihood and its gradient
separately, and at each point evaluated the likelihood once for its value
and again in the gradient's autograd pass, which computes the value
anyway. They now take both from one pass (``Gradient``), and reach the
same point.
"""

import numpy as np
from autograd import jacobian
from autograd.tracer import Box

import surpyval as sp
from surpyval.univariate.parametric import parametric_fitter
from surpyval.univariate.parametric.fitters import (
    Gradient,
    minimize_with_gradient,
    preconditioned_bfgs,
)


def _unboxed(value):
    while isinstance(value, Box):
        value = value._value
    return float(value)


class _Counted:
    """A Rosenbrock-like objective that records each point it is
    evaluated at, and whether in an autograd pass."""

    def __init__(self):
        self.calls = []

    def __call__(self, u):
        self.calls.append((isinstance(u, Box), tuple(_unboxed(v) for v in u)))
        return (1.0 - u[0]) ** 2 + 5.0 * (u[1] - u[0] ** 2) ** 2


def test_593_gradient_is_the_jacobian():
    fun = _Counted()
    x = np.array([0.3, -1.2])
    value, grad = Gradient(fun).value_and_grad(x)
    assert value == fun(x)
    assert np.array_equal(grad, jacobian(fun)(x))
    assert np.array_equal(Gradient(fun)(x), jacobian(fun)(x))


def test_593_gradient_keeps_the_last_point():
    fun = _Counted()
    gradient = Gradient(fun)
    x = np.array([0.3, -1.2])
    first = gradient.value_and_grad(x)
    first[1][0] = 99.0  # (a caller changing its copy)
    again = gradient(x)
    assert len(fun.calls) == 1
    assert again[0] != 99.0


def test_593_bfgs_takes_value_and_gradient_in_one_pass():
    # The same path to the same point, with no evaluation outside the
    # gradient's pass
    x0 = np.array([-1.0, 2.0])
    separate = _Counted()
    old = preconditioned_bfgs(
        separate, x0, jac=jacobian(separate), obj_scale=1.0
    )
    together = _Counted()
    new = preconditioned_bfgs(
        together, x0, jac=Gradient(together), obj_scale=1.0
    )
    assert np.array_equal(new.x, old.x)
    assert new.fun == old.fun
    assert new.nit == old.nit
    assert not [c for c in together.calls if not c[0]]
    assert len(together.calls) < len(separate.calls)


def test_593_tnc_takes_value_and_gradient_in_one_pass():
    x0 = np.array([-1.0, 2.0])
    together = _Counted()
    new = minimize_with_gradient(
        together, x0, (), Gradient(together), method="TNC"
    )
    separate = _Counted()
    old = minimize_with_gradient(
        separate, x0, (), jacobian(separate), method="TNC"
    )
    assert np.array_equal(new.x, old.x)
    assert not [c for c in together.calls if not c[0]]


def test_593_a_fit_evaluates_each_point_once(monkeypatch):
    # The fit used to evaluate the likelihood plainly at every point the
    # gradient's pass evaluated it at (7 of 7 points on these data). The
    # one left is the start, which the initial guess evaluates.
    calls = []
    original = parametric_fitter.ParametricFitter._neg_ll_func

    def counted(self, data, *params):
        boxed = any(isinstance(p, Box) for p in params)
        calls.append((boxed, tuple(_unboxed(p) for p in params)))
        return original(self, data, *params)

    monkeypatch.setattr(
        parametric_fitter.ParametricFitter, "_neg_ll_func", counted
    )
    x = sp.Weibull.random(1000, 10, 2, random_state=1)
    model = sp.Weibull.fit(x)
    plain = {point for boxed, point in calls if not boxed}
    in_pass = {point for boxed, point in calls if boxed}
    assert len(plain & in_pass) <= 1
    assert model.maximum == "verified"
