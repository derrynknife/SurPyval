"""Derivatives through ``np.where`` with broadcast arguments (#562).

``surpyval/utils/autograd_where_compat.py`` re-registers autograd's
derivative rules for ``where`` so that the cotangent is reduced to each
argument's shape. These tests check first and second derivatives through
a broadcast ``where`` against finite differences, and that autograd on
its own still gets them wrong (so the patch can be dropped when it does
not).
"""

import json
import subprocess
import sys

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad, hessian, make_jvp

import surpyval  # noqa: F401  (registers the rule)
from surpyval.tests._helpers import richardson_gradient, richardson_hessian

MASK = np.array([[True, False, True], [False, True, True]])
OTHER = np.arange(6.0).reshape(2, 3) / 4 + 0.5


def _smooth(w):
    # Not a polynomial, so that a wrong second derivative cannot hide in
    # a zero third one.
    return anp.sum(anp.exp(0.3 * w) * w**2)


# Each case: the free parameters and how they enter ``where`` (as x, as
# y, or both), broadcast against the (2, 3) condition.
CASES = {
    # one value against every entry
    "scalar": (
        np.array([0.7]),
        lambda t: anp.where(MASK, t[0], OTHER),
    ),
    # one value per column, as a (3,) row and as a (1, 3) row
    "row": (
        np.array([0.4, 1.1, -0.6]),
        lambda t: anp.where(MASK, t, OTHER),
    ),
    "row_2d": (
        np.array([0.4, 1.1, -0.6]),
        lambda t: anp.where(MASK, OTHER, anp.reshape(t, (1, 3))),
    ),
    # one value per row
    "column": (
        np.array([0.9, -0.3]),
        lambda t: anp.where(MASK, anp.reshape(t, (2, 1)), OTHER),
    ),
    # every entry its own
    "full": (
        np.linspace(-0.5, 1.0, 6),
        lambda t: anp.where(MASK, anp.reshape(t, (2, 3)), OTHER),
    ),
    # a scalar in one branch, a row in the other, the condition a row
    "mixed": (
        np.array([0.7, 0.2, -0.4, 1.3]),
        lambda t: anp.where(MASK[0], t[0], t[1:] * t[0]),
    ),
    # the accelerated-life substitution before #555: a (1,) life put in
    # one slot of a (2,) parameter row
    "substitution": (
        np.array([1.3, 0.4]),
        lambda t: anp.where(
            np.array([True, False]),
            anp.exp(t[1] * np.array([0.7])),
            anp.array([t[0], t[0]]),
        ),
    ),
}


def _fd_gradient(f, t):
    return richardson_gradient(f, t, 1e-4)


def _fd_hessian(f, t):
    return richardson_hessian(f, t, 1e-3)


@pytest.mark.parametrize("case", sorted(CASES))
def test_562_gradient_through_a_broadcast_where(case):
    t, inner = CASES[case]

    def f(v):
        return _smooth(inner(v))

    g = grad(f)(t)
    assert g.shape == t.shape
    np.testing.assert_allclose(g, _fd_gradient(f, t), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("case", sorted(CASES))
def test_562_hessian_through_a_broadcast_where(case):
    t, inner = CASES[case]

    def f(v):
        return _smooth(inner(v))

    H = hessian(f)(t)
    assert H.shape == (t.size, t.size)
    # Richardson-extrapolated second differences at h = 1e-3: error
    # O(h**4) ~ 1e-12 times the fourth derivative, plus rounding
    # ~ 1e-16 / h**2 = 1e-10.
    np.testing.assert_allclose(H, _fd_hessian(f, t), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("case", sorted(CASES))
def test_562_forward_mode_through_a_broadcast_where(case):
    t, inner = CASES[case]

    def f(v):
        return _smooth(inner(v))

    direction = np.linspace(1.0, -0.5, t.size)
    _, directional = make_jvp(f)(t)(direction)
    np.testing.assert_allclose(
        directional, _fd_gradient(f, t) @ direction, rtol=1e-8
    )


def test_562_forward_mode_tangent_has_the_output_shape():
    # A scalar against a (3,) condition and a (2, 3) other branch: the
    # tangent must be (2, 3), or a sum straight after counts each column
    # once instead of once per row (autograd 1.9.1 gives 2, not 4).
    def f(t):
        return anp.sum(anp.where(MASK[0], t, OTHER))

    _, directional = make_jvp(f)(0.5)(1.0)
    assert directional == 4.0


def test_562_where_values_and_python_types_are_unchanged():
    # Only the derivative rules are replaced: values, and autograd's
    # plain-Python result for plain-Python inputs, are as before.
    np.testing.assert_array_equal(
        anp.where(MASK, 1.0, OTHER), np.where(MASK, 1.0, OTHER)
    )
    assert anp.where([True, False], [1.0, 2.0], [3.0, 4.0]) == [1.0, 4.0]
    np.testing.assert_array_equal(anp.where(MASK)[1], np.where(MASK)[1])


# Plain autograd, in a fresh interpreter that never imports surpyval.
_UPSTREAM = r"""
import json
import autograd.numpy as anp
import numpy as np
from autograd import grad, hessian

out = {}
mask = np.array([True, False, True])
other = np.array([1.0, 2.0, 3.0])
try:
    g = grad(lambda t: anp.sum(anp.where(mask, t, other) ** 3))(2.0)
    out["scalar"] = np.asarray(g).tolist()
except Exception as e:
    out["scalar"] = type(e).__name__

m2 = np.array([[True, False, True], [False, True, True]])
try:
    g = grad(lambda t: anp.sum(anp.where(m2, t[None, :], 1.0) ** 3))(
        np.array([1.0, 2.0, 3.0])
    )
    out["row"] = np.asarray(g).tolist()
except Exception as e:
    out["row"] = type(e).__name__

def al(t):
    life = anp.exp(t[1] * np.array([0.7]))
    p = anp.where(np.array([True, False]), life, anp.array([t[0], t[0]]))
    return anp.log(p[0]) * p[1] ** 2 + p[0] ** 2 * p[1]

out["al_hessian"] = hessian(al)(np.array([1.3, 0.4])).tolist()
print(json.dumps(out))
"""


def test_562_autograd_still_needs_the_patch():
    # Fails when autograd differentiates a broadcast ``where`` correctly
    # itself: then surpyval/utils/autograd_where_compat.py (and its
    # import in surpyval/__init__.py) can go.
    run = subprocess.run(
        [sys.executable, "-c", _UPSTREAM],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(run.stdout)
    fixed = []
    # d/dt sum(where(mask, t, other)**3) at t = 2 is 2 * 3 * 2**2 = 24.
    if out["scalar"] == 24.0:
        fixed.append("scalar")
    if out["row"] == [3.0, 12.0, 54.0]:
        fixed.append("row")
    # The (1, 1) entry of the substitution's Hessian: 4.4607 (finite
    # differences and the patched rule); autograd 1.9.1 gives 7.519.
    if abs(out["al_hessian"][1][1] - 4.460713530754) < 1e-8:
        fixed.append("al_hessian")
    assert not fixed, (
        f"autograd now differentiates a broadcast where correctly "
        f"({fixed}): drop surpyval/utils/autograd_where_compat.py"
    )
