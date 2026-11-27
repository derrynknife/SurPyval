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
* ``inverse_gaussian``: mean 1 and variance ``theta`` (``lambda = 1 /
  theta`` in the usual ``IG(mu, lambda)``); ``I`` has a closed form for a
  whole number of events (:func:`inverse_gaussian_log_integral`).
* ``positive_stable``: Laplace transform ``exp(-s^alpha)`` with
  ``alpha = 1 - theta``; ``theta = 0`` is no frailty and ``theta`` is
  Kendall's tau. The frailty has no finite mean (nor variance), and the
  model stays one of proportional hazards when the frailty is integrated
  out, with every coefficient multiplied by ``alpha``. ``I`` has a closed
  form for a whole number of events
  (:func:`positive_stable_log_integral`).
"""

from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.scipy.special import logsumexp as _ad_logsumexp
from autograd.tracer import Box, getval
from scipy.integrate import quad
from scipy.special import exp1, gammaln, roots_hermite, wrightomega

from surpyval.utils.validation import check_option

FAMILIES = ("gamma", "lognormal", "inverse_gaussian", "positive_stable")
#: The families whose integral comes from the derivatives of their
#: Laplace transform, in closed form for a whole number of events.
LAPLACE_FAMILIES = ("inverse_gaussian", "positive_stable")

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
# Above this ``theta D`` the mode ``theta D - omega`` loses more than
# 1e-12 to cancellation, and is taken in its logarithmic form
# (``lognormal_mode``). Ordinary fits (theta of a few, D up to hundreds)
# stay below it.
_CANCELS = 1e4


def check_family(family: Any) -> str:
    """The frailty family's name, or a ``ValueError`` naming the choices."""
    check_option("family", family, FAMILIES)
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

    Where ``theta D`` is large, ``theta D - omega`` cancels (``omega`` is
    ``theta D`` less a logarithm): the mode is then taken as
    ``log(omega) - log(theta H)``, which ``omega + log(omega) = theta D +
    log(theta H)`` makes equal to it, without the cancellation.
    """
    theta_d = theta * D
    with np.errstate(divide="ignore"):
        log_theta_h = np.log(theta * H)
        omega = wrightomega(theta_d + log_theta_h).real
    mode = theta_d - omega
    big = (theta_d > _CANCELS) & (omega > 0) & np.isfinite(log_theta_h)
    if np.any(big):
        mode = np.where(
            big, np.log(np.where(big, omega, 1.0)) - log_theta_h, mode
        )
    return mode, omega


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
    # theta big^2 < eps, without squaring a big above 1e154, which
    # overflows (an OverflowError, not inf, for a Python float; #617)
    if tv <= 0 or max(big, 1.0) < np.sqrt(_EPS / tv):
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
    ``theta`` for the gamma and the inverse Gaussian (mean 1),
    ``exp(theta) - 1`` for the log-normal, and ``inf`` for the positive
    stable with ``theta > 0`` (no finite mean or variance).
    """
    if family == "lognormal":
        return float(np.expm1(theta))
    if family == "positive_stable":
        return float(np.inf) if theta > 0 else 0.0
    return float(theta)


def kendall_tau(family: str, theta: float) -> float:
    """
    Kendall's tau between two event times of one group (no covariates or
    censoring), the dependence measure that compares frailty families
    (Hougaard 2000, section 4.2): ``theta / (theta + 2)`` for the gamma,
    ``4 int_0^inf s L(s) L''(s) ds - 1`` for the log-normal, with ``L``
    the frailty's Laplace transform, by quadrature; ``theta`` itself for
    the positive stable (``1 - alpha``), and
    :func:`inverse_gaussian_kendall_tau` for the inverse Gaussian.
    """
    if theta <= 0:
        return 0.0
    if family == "gamma":
        return float(theta / (theta + 2.0))
    if family == "positive_stable":
        return float(theta)
    if family == "inverse_gaussian":
        return inverse_gaussian_kendall_tau(theta)

    def integrand(v: float) -> float:
        # s = exp(v): s L(s) L''(s) ds = exp(2 v) I(0, s) I(2, s) dv
        s = np.exp(v)
        logs = lognormal_log_integral(np.array([0.0, 2.0]), s, theta)
        return float(np.exp(2.0 * v + logs.sum()))

    centre = -0.5 * theta  # log of the median of 1 / u, roughly
    total, _ = quad(integrand, centre - 60, centre + 60, limit=400)
    return float(4.0 * total - 1.0)


# -- the families with a closed-form Laplace transform ----------------------


#: The most events in one group that the positive stable frailty's
#: integral takes: its coefficient table is a recursion of ``O(D^2)`` time
#: and memory (a million entries at 1000 events, evaluated at every step
#: of a fit).
STABLE_MAX_EVENTS = 1000


def whole_counts(D: npt.ArrayLike, family: str) -> npt.NDArray:
    """
    ``D`` as whole numbers, or a ``ValueError`` saying why they must be.

    The integrals of :data:`LAPLACE_FAMILIES` are the derivatives of the
    frailty's Laplace transform, of order ``D``: a closed form for a whole
    number of events in a group, and none for a fractional count (a row
    with a non-integer ``n``).
    """
    D = np.asarray(D, dtype=float)
    whole = np.round(D)
    if not np.all(np.isfinite(D) & (D >= 0) & (np.abs(D - whole) < 1e-9)):
        raise ValueError(
            "family={!r} needs a whole number of events in each group: its "
            "likelihood is a derivative of the frailty's Laplace transform "
            "of that order, which a fractional count (a non-integer 'n' on "
            "an event row) does not have. Give whole counts, or use "
            "family='gamma' or 'lognormal', which take any count.".format(
                family
            )
        )
    whole = whole.astype(int)
    top = int(np.max(whole, initial=0))
    if family == "positive_stable" and top > STABLE_MAX_EVENTS:
        raise ValueError(
            "family='positive_stable' takes at most {} events in one group "
            "(its likelihood is a recursion whose cost grows with the "
            "square of the count), and a group here has {}. Use "
            "family='inverse_gaussian' or 'gamma', whose integrals cost "
            "no more for many events; where the group is a cluster of "
            "several units, giving each unit its own frailty (no "
            "'cluster=') may bring each under the limit.".format(
                STABLE_MAX_EVENTS, top
            )
        )
    return whole


def inverse_gaussian_log_integral(D: npt.ArrayLike, H: Any, theta: Any) -> Any:
    """
    ``log I(D, H)`` for the inverse Gaussian frailty of mean 1 and variance
    ``theta``, elementwise, in closed form.

    The frailty's density is that of ``IG(1, lambda)`` with ``lambda = 1 /
    theta``, :math:`g(u) = \\sqrt{\\lambda / (2\\pi u^3)}
    \\exp(-\\lambda (u - 1)^2 / (2u))`, and its Laplace transform is

    .. math::
        L(s) = E[e^{-us}] = \\exp\\Big(\\frac{1 - \\sqrt{1 + 2\\theta s}}
        {\\theta}\\Big)

    (Hougaard 2000, *Analysis of Multivariate Survival Data*, section
    7.3). ``I(D, H) = (-1)^D L^{(D)}(H)``: writing the integrand
    :math:`u^D e^{-uH} g(u)` as a generalised inverse Gaussian kernel
    :math:`u^{D - 3/2} \\exp(-(a u + b / u) / 2)` with ``a = lambda + 2H``
    and ``b = lambda``, its integral is
    :math:`2 (b/a)^{\\nu/2} K_\\nu(\\sqrt{ab})` (DLMF 10.32.10) with
    ``nu = D - 1/2``, and the Bessel function of half-integer order is a
    finite sum (DLMF 10.49.12). With ``q = sqrt(1 + 2 theta H)``,

    .. math::
        \\log I(D, H) = \\frac{1 - q}{\\theta} - D \\log q
        + \\log \\sum_{k=0}^{D-1} \\frac{(D - 1 + k)!}{k! \\, (D - 1 - k)!}
        \\Big(\\frac{\\theta}{2q}\\Big)^k

    (the sum is 1 for ``D = 0``). It is computed as ``-2H / (1 + q) - (D /
    2) log1p(2 theta H)`` plus a log-sum-exp of the sum's terms, all
    positive, so it keeps its digits for every ``theta`` and ``H``, and
    tends to ``-H`` (no frailty) as ``theta -> 0``. ``D`` must be whole
    numbers (:func:`whole_counts`); ``H`` and ``theta`` may be traced by
    autograd. ``theta <= 0`` gives ``-H``.
    """
    D = np.asarray(D, dtype=float)
    shape = np.broadcast(D, np.asarray(getval(H))).shape
    D = np.broadcast_to(D, shape)
    H = H + 0.0 * D
    if not float(getval(theta)) > 0:
        return -1.0 * H
    Di = whole_counts(D, "inverse_gaussian")
    two_theta_h = 2.0 * theta * H
    q = anp.sqrt(1.0 + two_theta_h)
    value = -2.0 * H / (1.0 + q) - 0.5 * D * anp.log1p(two_theta_h)
    top = int(np.max(Di, initial=0))
    if top <= 1:
        return value
    k = np.arange(top, dtype=float)
    with np.errstate(invalid="ignore"):
        log_c = (
            gammaln(Di[..., None] + k)
            - gammaln(k + 1.0)
            - gammaln(np.maximum(Di[..., None] - k, 1.0))
        )
    log_c = np.where(k < Di[..., None], log_c, -np.inf)
    log_c[..., 0] = 0.0
    log_r = anp.log(theta) - anp.log(2.0 * q)
    terms = log_c + k * anp.expand_dims(log_r, -1)
    return value + _ad_logsumexp(terms, axis=-1)


def _stable_coefficients(top: int, theta: Any) -> Any:
    """
    The logs of the coefficients ``c_{d,j}`` of the positive stable
    frailty's integral, ``(top + 1, top + 1)`` with ``-inf`` where a
    coefficient is 0: row ``d``, column ``j`` (see
    :func:`positive_stable_log_integral`). The recursion
    ``c_{d+1,j} = alpha c_{d,j-1} + (d - j alpha) c_{d,j}`` has positive
    terms only (``d - j alpha = (d - j) + j theta``, written so, without
    the cancellation of ``d - j alpha`` as ``alpha -> 1``), so it is
    carried in logs.
    """
    log_alpha = anp.log(1.0 - theta)
    neg = -np.inf
    rows = [anp.concatenate([np.zeros(1), np.full(top, neg)])]
    if top == 0:
        return anp.stack(rows)
    prev = anp.reshape(log_alpha, (1,))  # c_{1,1} = alpha
    rows.append(
        anp.concatenate([np.full(1, neg), prev, np.full(top - 1, neg)])
    )
    for d in range(1, top):
        j = np.arange(1, d + 1, dtype=float)
        # d - j alpha, for j = 1 .. d, and (j = d + 1) alpha c_{d,d}
        stay = anp.log((d - j) + j * theta) + prev
        move = log_alpha + prev
        prev = anp.concatenate(
            [stay[:1], anp.logaddexp(stay[1:], move[:-1]), move[-1:]]
        )
        rows.append(
            anp.concatenate([np.full(1, neg), prev, np.full(top - d - 1, neg)])
        )
    return anp.stack(rows)


def positive_stable_log_integral(D: npt.ArrayLike, H: Any, theta: Any) -> Any:
    """
    ``log I(D, H)`` for the positive stable frailty with index
    ``alpha = 1 - theta``, elementwise, in closed form.

    The positive stable distribution of index ``alpha`` in (0, 1] has the
    Laplace transform

    .. math::
        L(s) = E[e^{-us}] = \\exp(-s^{\\alpha})

    (Hougaard 1986, *Biometrika* 73, 387-396; Hougaard 2000, section 7.2):
    ``alpha = 1`` (``theta = 0``) is the frailty fixed at 1, no frailty,
    and ``theta = 1 - alpha`` is Kendall's tau between two event times of
    a group. The frailty has no finite mean, so it is not scaled to mean
    1: its scale is that of the baseline hazard. ``I(D, H) = (-1)^D
    L^{(D)}(H)``; differentiating ``D`` times,

    .. math::
        I(D, H) = e^{-H^\\alpha} \\sum_{j=1}^{D} c_{D,j} \\,
        H^{j\\alpha - D}, \\qquad
        c_{d+1,j} = \\alpha \\, c_{d,j-1} + (d - j\\alpha) \\, c_{d,j},

    from ``c_{0,0} = 1`` (``I(0, H) = L(H)``): each step is ``I(d + 1, H) =
    -dI(d, H)/dH``, and the derivative of ``e^{-H^alpha} H^{j alpha - d}``
    is ``-e^{-H^alpha} (alpha H^{(j+1) alpha - d - 1} + (d - j alpha)
    H^{j alpha - d - 1})``. Every coefficient is positive (``j <= d``), so
    the sum is taken as a log-sum-exp; as ``theta -> 0`` only ``c_{D,D} =
    alpha^D`` stays, and the value tends to ``-H``.

    ``D`` must be whole numbers (:func:`whole_counts`); ``H`` and
    ``theta`` may be traced by autograd. ``theta <= 0`` gives ``-H``, and
    ``theta >= 1`` (``alpha <= 0``, no distribution) ``-inf``. A group
    with events and ``H = 0`` has ``I = E[u^D]``, which is infinite.
    """
    D = np.asarray(D, dtype=float)
    shape = np.broadcast(D, np.asarray(getval(H))).shape
    D = np.broadcast_to(D, shape)
    H = H + 0.0 * D
    tv = float(getval(theta))
    if not tv > 0:
        return -1.0 * H
    if not tv < 1:
        return np.full(shape, -np.inf)
    Di = whole_counts(D, "positive_stable")
    alpha = 1.0 - theta
    Hv = np.asarray(getval(H), dtype=float)
    positive = Hv > 0
    # (log H only where H > 0: a group with H = 0 has I(0, 0) = 1 and,
    # with events, I = inf, set below)
    log_h = anp.log(anp.where(positive, H, 1.0))
    value = -anp.where(positive, anp.exp(alpha * log_h), 0.0)
    top = int(np.max(Di, initial=0))
    if top == 0:
        return value
    table = _stable_coefficients(top, theta)
    j = np.arange(top + 1, dtype=float)
    terms = table[Di] + (j * alpha - anp.expand_dims(D, -1)) * anp.expand_dims(
        log_h, -1
    )
    value = value + _ad_logsumexp(terms, axis=-1)
    if np.any(~positive & (Di > 0)):
        value = anp.where(~positive & (Di > 0), np.inf, value)
    return value


def laplace_log_integral(
    family: str, D: npt.ArrayLike, H: Any, theta: Any
) -> Any:
    """``log I(D, H)`` of a family of :data:`LAPLACE_FAMILIES`."""
    if family == "inverse_gaussian":
        return inverse_gaussian_log_integral(D, H, theta)
    if family == "positive_stable":
        return positive_stable_log_integral(D, H, theta)
    raise ValueError(
        "laplace_log_integral takes one of {}; got {!r}.".format(
            LAPLACE_FAMILIES, family
        )
    )


def posterior_mean(
    family: str, D: npt.ArrayLike, H: npt.ArrayLike, theta: float
) -> npt.NDArray:
    """
    Each group's posterior mean frailty ``I(D + 1, H) / I(D, H)``, for a
    family of :data:`LAPLACE_FAMILIES` (finite for every group with
    ``H > 0``, the positive stable's too, though its prior mean is not).
    """
    D = np.asarray(D, dtype=float)
    H = np.asarray(H, dtype=float)
    return np.exp(
        laplace_log_integral(family, D + 1.0, H, theta)
        - laplace_log_integral(family, D, H, theta)
    )


#: The largest ``theta`` of each family: the positive stable's ``theta =
#: 1 - alpha`` is below 1; the others' are unbounded.
THETA_UPPER = {"positive_stable": 1.0}


def theta_upper(family: str) -> float:
    """The supremum of the family's ``theta``."""
    return THETA_UPPER.get(family, np.inf)


def inverse_gaussian_kendall_tau(theta: float) -> float:
    """
    Kendall's tau of the inverse Gaussian frailty of variance ``theta``,

    .. math::
        \\tau = \\frac12 - \\frac{1}{\\theta}
        + \\frac{2}{\\theta^2} e^{2/\\theta} E_1(2/\\theta),

    with ``E_1`` the exponential integral (Hougaard 2000, section 7.3),
    which rises from 0 at ``theta = 0`` to 1/2 as ``theta -> inf``. Its
    terms cancel as ``theta -> 0``, where the series of ``E_1``'s
    asymptotic expansion, ``tau = theta / 2 - theta^2 / 2 + 3 theta^3 / 4
    - ...``, is used instead.
    """
    if theta <= 0:
        return 0.0
    if theta < 0.02:
        # e^x E1(x) ~ sum_k (-1)^k k! / x^(k+1), x = 2 / theta: the terms
        # through k = 8 leave an error below 1e-14 here.
        x = 2.0 / theta
        series = sum(
            (-1.0) ** k * float(np.prod(np.arange(1, k + 1))) / x ** (k + 1)
            for k in range(2, 12)
        )
        # (the k = 0 and 1 terms cancel the first two terms of tau)
        return float(0.5 * x**2 * series)
    x = 2.0 / theta
    return float(0.5 - 1.0 / theta + x**2 / 2.0 * np.exp(x) * exp1(x))
