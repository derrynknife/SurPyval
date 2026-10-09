import math
from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.tracer import isbox

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter


@singleton_fitter
class CoxLewis(NHPPFitter):
    """
    The Cox-Lewis (log-linear) non-homogeneous Poisson process, with

    .. math::
        \\lambda(t) = e^{\\alpha + \\beta t}, \\qquad
        \\Lambda(t) = \\frac{e^{\\alpha}}{\\beta}
        \\left(e^{\\beta t} - 1\\right).

    ``alpha`` is the log of the intensity at ``t = 0`` and ``beta`` its
    proportional change per unit time: positive is deteriorating, negative
    improving. With a negative ``beta`` the cumulative intensity levels off
    at ``exp(alpha) / -beta``, and ``inv_cif`` returns ``inf`` beyond it;
    such a model's ``count_terminated_simulation`` raises a ``ValueError``
    (a sequence may never reach the count), so simulate it to a time.
    ``CoxLewis`` is an instance of this class; ``fit`` and ``from_params``
    return a ``ParametricRecurrenceModel``.

    Examples
    --------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import CoxLewis
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1).cumsum()
    >>> model = CoxLewis.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Cox-Lewis
    Fitted by           : MLE
    Parameters          :
         alpha: 0.384812737762836
          beta: 0.19396672109211047
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([ 1.62151879,  3.59013322,  5.98014113,  8.88174429, 12.40445268,
           16.6812175 ])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([1.78385983, 2.16570551, 2.62928751, 3.19210196, 3.87539016,
           4.70494021])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([0.63925589, 1.20792021, 1.72004461, 2.18586234, 2.61305756,
           3.00754742])
    """

    def __init__(self) -> None:
        self.name = "Cox-Lewis"
        self.parameter_names = ["alpha", "beta"]
        self.has_scale = True
        # alpha is the *log*-intensity intercept and is legitimately
        # negative whenever the baseline rate is below one event per
        # time unit; the old (0, None) bound silently pinned such fits
        # at alpha = 0 (#286).
        self.bounds = ((None, None), (None, None))
        # The log-linear intensity is defined at any time, so an item
        # observed from a negative ``tl`` may have events at negative times.
        self.support = (-np.inf, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        # The Cox-Lewis intensity is log-linear, so its cumulative intensity
        # is the integral of ``exp(alpha + beta * x)`` from 0 to ``x``,
        # ``exp(alpha) * (exp(beta x) - 1) / beta``. Written with expm1 and
        # its beta -> 0 limit ``exp(alpha) * x`` (an HPP): the direct form
        # is 0/0 at beta = 0 and loses digits for a tiny beta.
        alpha = params[0]
        beta = params[1]
        if not any(map(isbox, (x, alpha, beta))):
            x = onp.asarray(x, dtype=float)
            return onp.exp(alpha) * _expm1_over_value(beta, x)
        # Traced: differentiable by autograd (#760)
        return np.exp(alpha) * _expm1_over(beta, x)

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha + beta * x)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return alpha + beta * x

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        # For an improving system (beta < 0) the cumulative intensity is
        # bounded above by exp(alpha) / -beta, so counts at or beyond that
        # asymptote are never reached: return inf rather than log of a
        # non-positive number. log1p and the beta -> 0 limit
        # ``N exp(-alpha)`` keep a tiny or zero beta exact.
        scaled = np.asarray(N, dtype=float) * np.exp(-alpha)
        arg = scaled * beta
        reached = arg > -1.0
        safe_arg = np.where(reached, arg, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                beta == 0,
                scaled,
                np.log1p(safe_arg) / np.where(beta == 0, 1.0, beta),
            )
        return np.where(reached, ratio, np.inf)

    def _default_start(
        self, data: Any, x_unique: npt.NDArray, mcf_hat: npt.NDArray
    ) -> npt.NDArray:
        # The constant rate through the end of the MCF (beta = 0, an HPP).
        # The all-ones start is a rate that grows e-fold per unit time: by
        # t = 60 its cif is 1e26, where the least-squares search lost its
        # way and stopped at a cif(55) of 113 against an MCF of 4.4 (#419).
        # It was also not unit free: time in hours or in days started at
        # different places.
        span = float(x_unique[-1])
        if span > 0 and mcf_hat[-1] > 0:
            return np.array([np.log(mcf_hat[-1] / span), 0.0])
        return self.parameter_initialiser(data.x)


def _expm1_over_value(beta: Any, x: npt.NDArray) -> npt.NDArray:
    """``(exp(beta * x) - 1) / beta``, with its limit ``x`` at beta = 0,
    in plain numpy."""
    safe_beta = onp.where(beta == 0, 1.0, beta)
    return onp.where(beta == 0, x, onp.expm1(beta * x) / safe_beta)


#: ``_expm1_over_value`` as an autograd primitive (#760), whose
#: derivatives are ``_expm1_over_slope`` in ``beta`` and ``e**(beta x)``
#: in ``x``, each differentiable in turn
_expm1_over = primitive(_expm1_over_value)


#: ``(k - 1) / k!`` for ``k = 2, 3, ...``: the series of the slope of
#: ``expm1(u) / beta`` in ``beta``, over ``x**2``, in powers of ``u =
#: beta x`` (to 1e-20 below ``|u|`` of 1/2)
_SLOPE_SERIES = tuple((k - 1) / math.factorial(k) for k in range(2, 20))


def _expm1_over_slope(beta: Boxable, x: Boxable) -> Boxable:
    """The derivative of ``_expm1_over`` in ``beta``, ``(x e**u -
    expm1(u) / beta) / beta`` with ``u = beta x``, written for autograd.
    That form cancels as ``u -> 0`` (and is 0 / 0 at 0, where the slope is
    ``x**2 / 2``): below ``|u|`` of 1/2 it is the series instead. Each
    branch is evaluated only where it is taken (at a point of the other's
    where not), so neither puts a nan into the other's gradient."""
    u = beta * x
    small = np.abs(u) < 0.5
    u_small = np.where(small, u, 0.0)
    series = 0.0
    for coeff in reversed(_SLOPE_SERIES):
        series = series * u_small + coeff
    x_small = np.where(small, x, 0.0)
    near = x_small * x_small * series
    beta_far = np.where(small, 1.0, beta)
    u_far = np.where(small, 1.0, u)
    with np.errstate(over="ignore"):
        far = (
            np.where(small, 1.0, x) * np.exp(u_far)
            - np.expm1(u_far) / beta_far
        ) / beta_far
    return np.where(small, near, far)


defvjp(
    _expm1_over,
    lambda ans, beta, x: unbroadcast_f(
        beta, lambda g: g * _expm1_over_slope(beta, x)
    ),
    lambda ans, beta, x: unbroadcast_f(x, lambda g: g * np.exp(beta * x)),
)
