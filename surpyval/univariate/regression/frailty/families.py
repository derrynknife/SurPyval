"""
The frailty distributions of the shared-frailty models.

Each family gives the integral that a group's likelihood needs,

.. math::
    I(D, H) = \\int_0^\\infty u^D e^{-u H} \\, dG(u),

for a group with ``D`` events and summed cumulative hazard
``H = sum eta H0`` (``eta = exp(beta'Z)``), and the posterior mean frailty
``I(D + 1, H) / I(D, H)``. The same integral with ``D = 0`` is the Laplace
transform of the frailty, which is the marginal survival
``S(t | Z) = I(0, eta H0(t))``, and the marginal hazard is
``eta h0(t) I(1, s) / I(0, s)`` at ``s = eta H0(t)``.

* ``gamma``: mean 1 and variance ``theta``; ``I`` has a closed form
  (``frailty_fitter._group_frailty_ll``).
* ``lognormal``: ``u = exp(w)`` with ``w ~ N(0, theta)``, the
  parameterisation of R's ``frailtypack`` (``RandDist = "LogN"``),
  ``coxme`` and ``survival::frailty(dist = "gaussian")``: ``theta`` is the
  variance of the log-frailty, and the median frailty is 1. ``I`` has no
  closed form; it is computed by adaptive Gauss-Hermite quadrature
  (:func:`lognormal_log_integral`).
"""

from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.tracer import Box, getval
from scipy.integrate import quad
from scipy.special import roots_hermite, wrightomega

FAMILIES = ("gamma", "lognormal")

# Gauss-Hermite nodes of the log-normal integral. The rule is centred on
# the mode of each group's integrand and scaled by its curvature there
# (adaptive quadrature), so the integrand it sees is close to a Gaussian
# however many events the group has. Its error, against scipy's adaptive
# quadrature at a relative tolerance of 2e-14 over 0 to 200 events per
# group and cumulative hazards from 1e-4 to 1000 (the accuracy check in
# ``test_frailty_lognormal.py``), grows with the log-frailty variance
# theta, where a group with few events has an integrand with a sharp
# right edge (``exp(-H e^w)``) that no polynomial follows well. Thirty
# nodes keep the log-integral within 1e-12 for theta <= 0.5, 3e-10 at 1,
# 7e-8 at 2, 1e-5 at 5 and 1e-3 at 20 (plain Gauss-Hermite with 100 nodes
# is off by hundreds where a group has many events); twenty nodes give
# 2e-8 at theta = 1 and 2e-6 at 2.
N_NODES = 30
_NODES: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}

_EPS = float(np.finfo(float).eps)


def check_family(family: Any) -> str:
    """The frailty family's name, or a ``ValueError`` naming the choices."""
    if family not in FAMILIES:
        raise ValueError(
            "family must be one of {}; got {!r}.".format(
                ", ".join(repr(f) for f in FAMILIES), family
            )
        )
    return str(family)


def _hermite(n_nodes: int) -> tuple[npt.NDArray, npt.NDArray]:
    """The nodes and the weights, divided by ``sqrt(pi)`` so that they sum
    to 1, of the ``n_nodes``-point Gauss-Hermite rule (weight
    ``exp(-z^2)``)."""
    if n_nodes not in _NODES:
        z, wt = roots_hermite(n_nodes)
        _NODES[n_nodes] = (z, wt / np.sqrt(np.pi))
    return _NODES[n_nodes]


def lognormal_mode(
    D: npt.NDArray, H: npt.NDArray, theta: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    The mode ``w*`` of ``D w - H exp(w) - w^2 / (2 theta)``, and
    ``omega = H e^{w*} theta``.

    Setting the derivative to zero gives
    ``w* = theta D - W(theta H exp(theta D))`` with ``W`` the Lambert W
    function, which is ``theta D - omega`` with ``omega`` the Wright omega
    function of ``theta D + log(theta H)``: closed form, and without the
    overflow of ``exp(theta D)``. The curvature there is
    ``(1 + omega) / theta``.
    """
    with np.errstate(divide="ignore"):
        omega = wrightomega(theta * D + np.log(theta * H)).real
    return theta * D - omega, omega


def lognormal_log_integral(
    D: npt.ArrayLike,
    H: Any,
    theta: Any,
    ops: Any = None,
    n_nodes: int = N_NODES,
) -> Any:
    """
    ``log I(D, H)`` for the log-normal frailty, elementwise:

    .. math::
        \\log \\int_{-\\infty}^{\\infty}
        e^{D w - H e^w} \\, \\phi(w; 0, \\theta) \\, dw,

    by ``n_nodes``-point Gauss-Hermite quadrature centred on the
    integrand's mode ``r`` and scaled by its curvature there (adaptive
    quadrature, as ``lme4``'s ``nAGQ``): nodes ``w = r + y`` with
    ``y = sqrt(2) s z``, ``s^2 = theta / (1 + omega)``.

    The value is computed in a form free of cancellation, so that it keeps
    its relative accuracy when it is small (a marginal cumulative hazard
    near 0): expanding the exponent about the mode,

    .. math::
        \\log I = D r - \\frac{r^2}{2\\theta} - \\tfrac12\\log(1 + \\omega)
        - \\frac{\\omega}{\\theta}
        + \\log \\sum_k p_k \\exp\\Big(-\\frac{\\omega}{\\theta}
          \\big(e^{y_k} - 1 - y_k - \\tfrac12 y_k^2\\big)\\Big),

    with ``p_k`` the rule's weights (summing to 1). The mode and the scale
    only place the nodes -- the rule is a valid quadrature wherever they
    are -- so they are not differentiated: the derivatives in ``H`` and
    ``theta`` are those of the rule's sum with the nodes held fixed, taken
    through the direct form of the exponent when ``H`` or ``theta`` is
    traced by autograd. ``ops`` is accepted for the signature of the
    gamma family's term and not needed.

    ``D`` and ``H`` broadcast together (any shape); ``theta`` is a scalar.
    As ``theta -> 0`` the value tends to ``-H``, the no-frailty term; a
    ``theta`` too small to change it by more than rounding gives ``-H``.
    """
    D = np.asarray(D, dtype=float)
    Hv = np.asarray(getval(H), dtype=float)
    tv = float(getval(theta))
    D, Hv = np.broadcast_arrays(D, Hv)
    big = max(float(np.max(Hv, initial=0.0)), float(np.max(D, initial=0.0)))
    if tv * max(big, 1.0) ** 2 < _EPS:
        # The corrections to -H are of order theta (D - H)^2 and theta H.
        return -1.0 * H + 0.0 * D
    r, omega = lognormal_mode(D, Hv, tv)
    z, p = _hermite(n_nodes)
    y = np.sqrt(2.0 * tv / (1.0 + omega))[..., None] * z
    with np.errstate(over="ignore", invalid="ignore"):
        # The value, without cancellation (see above). The log of the sum
        # is ``log1p`` of the sum of ``p (e^delta - 1)`` while that is small
        # (the integrand near its Gaussian approximation, and the value is
        # then kept to full relative accuracy), and a log-sum-exp otherwise.
        remainder = np.expm1(y) - y - 0.5 * y**2
        delta = -(omega / tv)[..., None] * remainder
        near = np.sum(p * np.expm1(delta), axis=-1)
        log_p_delta = np.log(p) + delta
        top = np.max(log_p_delta, axis=-1)
        far = top + np.log(np.sum(np.exp(log_p_delta - top[..., None]), -1))
        log_sum = np.where(
            np.abs(near) < 0.5, np.log1p(np.clip(near, -0.5, 0.5)), far
        )
        value = (
            D * r - r**2 / (2.0 * tv) - 0.5 * np.log1p(omega) - omega / tv
        ) + log_sum
    if not (isinstance(H, Box) or isinstance(theta, Box)):
        return value
    # The derivatives: those of the direct form of the rule's log-sum with
    # the nodes fixed, carried by a term whose value is exactly 0.
    xp = anp
    w = r[..., None] + y
    with np.errstate(over="ignore"):
        e_w = np.exp(w)
    a = (
        D[..., None] * w
        - xp.expand_dims(H + 0.0 * D, -1) * e_w
        - w**2 / (2.0 * theta)
        - 0.5 * xp.log(theta)
        + z**2
    )
    a_top = np.max(getval(a), axis=-1)
    direct = xp.log(xp.sum(p * xp.exp(a - a_top[..., None]), axis=-1))
    return value + (direct - getval(direct))


def lognormal_posterior_mean(
    D: npt.ArrayLike, H: npt.ArrayLike, theta: float
) -> npt.NDArray:
    """The posterior mean frailty ``I(D + 1, H) / I(D, H)`` of each group."""
    D = np.asarray(D, dtype=float)
    H = np.asarray(H, dtype=float)
    return np.exp(
        lognormal_log_integral(D + 1.0, H, theta)
        - lognormal_log_integral(D, H, theta)
    )


def frailty_cv2(family: str, theta: float) -> float:
    """``Var(u) / E(u)^2``, the frailty's squared coefficient of variation:
    ``theta`` for the gamma (mean 1), ``exp(theta) - 1`` for the log-normal.
    """
    if family == "lognormal":
        return float(np.expm1(theta))
    return float(theta)


def kendall_tau(family: str, theta: float) -> float:
    """
    Kendall's tau between two event times of one group (no covariates or
    censoring), the dependence measure that compares frailty families
    (Hougaard 2000, section 4.2): ``theta / (theta + 2)`` for the gamma,
    and ``4 int_0^inf s L(s) L''(s) ds - 1`` for the log-normal, with ``L``
    the frailty's Laplace transform, by quadrature.
    """
    if theta <= 0:
        return 0.0
    if family == "gamma":
        return float(theta / (theta + 2.0))

    def integrand(v: float) -> float:
        # s = exp(v): s L(s) L''(s) ds = exp(2 v) I(0, s) I(2, s) dv
        s = np.exp(v)
        logs = lognormal_log_integral(np.array([0.0, 2.0]), s, theta)
        return float(np.exp(2.0 * v + logs.sum()))

    centre = -0.5 * theta  # log of the median of 1 / u, roughly
    total, _ = quad(integrand, centre - 60, centre + 60, limit=400)
    return float(4.0 * total - 1.0)
