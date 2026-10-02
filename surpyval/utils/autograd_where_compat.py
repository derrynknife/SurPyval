"""
A broadcast-aware derivative rule for ``autograd.numpy.where``.

autograd (up to 1.9.1 at least) differentiates ``np.where(c, x, y)`` by
masking the output's cotangent, ``where(c, g, 0)`` for ``x`` and
``where(c, 0, g)`` for ``y``, without reducing it back to the shape of
``x`` or ``y``. When ``x`` or ``y`` is broadcast (a scalar, a row or a
column against a larger condition), the cotangent it receives has the
output's shape, not its own:

- the gradient with respect to a broadcast argument comes back with the
  wrong shape, or ``grad`` raises (``array is not broadcastable to
  correct shape``, ``cannot reshape array of size 3 into shape ()``);
- where a later rule happens to sum the extra entries away, the first
  derivative is right but the second derivative through it is silently
  wrong (a LogNormal ``Power`` accelerated-life model's ``Hf`` second
  derivative was 13% off, #555; #562).

Importing this module (``import surpyval`` does) re-registers both rules
of ``where`` with the cotangent reduced to each argument's shape
(``unbroadcast_f``, as autograd's own arithmetic rules do); the
condition gets none. The rules are written in ``autograd.numpy``, so
they are themselves differentiable and Hessians through ``where`` are
right. The forward-mode rules are re-registered too, so that a tangent
has the output's shape. Values are unchanged: only derivatives through a
broadcast ``where`` differ.

``surpyval/tests/utils/test_autograd_where.py`` checks that autograd
itself still gets this wrong, so the patch can be dropped once it does
not.
"""

from typing import Any, Callable

import autograd.numpy as anp
from autograd.extend import defjvp, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f


def _masked(c: Any, g: Any, take: bool) -> Any:
    """``g`` where ``c`` is ``take``, zero elsewhere (broadcast to their
    common shape)."""
    zero = anp.zeros(anp.shape(g))
    return anp.where(c, g, zero) if take else anp.where(c, zero, g)


def _where_vjp(take: bool) -> Callable:
    def vjp_maker(ans: Any, c: Any, x: Any = None, y: Any = None) -> Callable:
        target = x if take else y
        return unbroadcast_f(target, lambda g: _masked(c, g, take))

    return vjp_maker


def _where_jvp(take: bool) -> Callable:
    def jvp(g: Any, ans: Any, c: Any, x: Any = None, y: Any = None) -> Any:
        # Broadcast the tangent against the output, not only against the
        # condition: a scalar ``x`` beside a matrix ``y`` gives a matrix.
        g_full = g + anp.zeros(anp.shape(ans))
        return _masked(c, g_full, take)

    return jvp


# autograd 1.9 wraps the ``where`` primitive in a plain function (to keep
# Python types for Python inputs); the derivative rules belong to the
# primitive inside it.
WHERE_PRIMITIVE = getattr(anp, "_original_where", anp.where)

defvjp(WHERE_PRIMITIVE, None, _where_vjp(True), _where_vjp(False))
defjvp(WHERE_PRIMITIVE, None, _where_jvp(True), _where_jvp(False))
