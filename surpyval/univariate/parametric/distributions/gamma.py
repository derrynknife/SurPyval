from __future__ import annotations

from typing import Any, Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import gamma as agamma
from autograd.scipy.special import gammaln as agammaln
from autograd.tracer import getval, isbox
from scipy.special import digamma, gammaincinv

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.autograd_gamma_compat import gammainc as agammainc
from surpyval.utils.autograd_gamma_compat import gammainccln as agammainccln
from surpyval.utils.autograd_gamma_compat import gammaincln as agammaincln
from surpyval.utils.surpyval_data import SurpyvalData

from ._stable import on_support, positive_or_one, power_at_zero

#: The standard-Gamma time past which (and past ``a + 2 sqrt(a) + 1``)
#: the hazard is taken from the continued fraction (``Gamma_.hf``), which
#: converges there within 12 terms for a shape up to 10, 30 below 100 and
#: 100 for any shape. Below it ``f / S`` is kept. That loses about ``y
#: eps`` to rounding, and its derivatives in ``x`` and ``alpha``,
#: differences of terms of size 1 that cancel to one of size ``1 / y``, a
#: factor ``y`` more: ``alpha``'s was 2e-7 off mpmath's at ``y = 100``
#: and 4e-5 at 1000, ``x``'s 8e-8 at 1000 (#777).
_FRACTION_FROM = 30.0
#: The fraction's most terms, and its convergence test.
_FRACTION_TERMS = 500
_FRACTION_TOL = 4e-16


@primitive
def _hazard_over_rate(a: Boxable, y: Boxable) -> Boxable:
    r"""
    The standard Gamma's hazard at ``y`` (the Gamma's over its rate, at
    :math:`y = \beta x`), for :math:`y > a + 1`, from Legendre's continued
    fraction of the upper incomplete gamma,

    .. math::
        \Gamma(a, y) = \frac{y^{a} e^{-y}}{y + 1 - a -
        \frac{1 (1 - a)}{y + 3 - a - \frac{2 (2 - a)}{y + 5 - a -
        \cdots}}}.

    The hazard :math:`r = y^{a - 1} e^{-y} / \Gamma(a, y)` is that
    denominator over :math:`y`, which is :math:`1 + (1 - a)(1 - g) / y`
    with :math:`g = 1 / (y + 3 - a - \cdots)`, the fraction from its
    second term (:func:`_fraction_jet`): its excess over 1 is carried as
    itself, so neither the value nor its derivatives is a difference of
    terms of size 1. From :math:`y = a + 1` up, the hazard is within
    1e-15 of mpmath's (5e-15 at a shape of 1e4).

    An autograd primitive (#777), with the fraction differentiated along
    with it (:func:`_fraction_jet`) rather than traced: its derivative in
    ``y`` is :math:`r (a - 1) g / y` (the hazard's own, :math:`r ((a -
    1) / y - 1 + r)`, with the cancelling terms taken out), in ``a`` it
    is :math:`-(1 - g + (1 - a) g_a) / y`, and their derivatives are the
    fraction's second ones, so a Hessian is exact. Traced, the fraction
    from a time of 10 made a Gamma PH fit 1.7 times slower (#760).
    """
    return _plain_float(_ratio(a, y, _cached_jet(a, y, 0)[0]))


def _ratio(a: Any, y: Any, g: Any) -> Any:
    """``r = 1 + (1 - a)(1 - g) / y`` (see :func:`_hazard_over_rate`)."""
    return 1.0 + (1.0 - a) * (1.0 - g) / y


@primitive
def _hazard_over_rate_da(a: Boxable, y: Boxable) -> Boxable:
    """``d r / d a`` (:func:`_hazard_over_rate`)."""
    g, first, _ = _cached_jet(a, y, 1)
    return _plain_float(-((1.0 - g) + (1.0 - a) * first[0]) / y)


@primitive
def _hazard_over_rate_dy(a: Boxable, y: Boxable) -> Boxable:
    """``d r / d y = r (a - 1) g / y`` (:func:`_hazard_over_rate`)."""
    g = _cached_jet(a, y, 0)[0]
    return _plain_float(_ratio(a, y, g) * (a - 1.0) * g / y)


def _hazard_over_rate_second(a: Any, y: Any) -> tuple:
    """``(r_aa, r_ay, r_yy)``, from the fraction's derivatives."""
    g, (g_a, g_y), (g_aa, g_ay, g_yy) = _cached_jet(a, y, 2)
    u, v = 1.0 - a, 1.0 - g
    return (
        (2.0 * g_a - u * g_aa) / y,
        (v + u * g_a) / y**2 + (g_y - u * g_ay) / y,
        u / y * (2.0 * g_y / y + 2.0 * v / y**2 - g_yy),
    )


def _vjp(target_is_a: bool, k: int) -> Callable:
    """The VJP maker, in ``a`` or ``y``, of a first derivative of
    :func:`_hazard_over_rate`: entry ``k`` of
    :func:`_hazard_over_rate_second` (higher orders cut)."""

    def make(ans: Any, a: Any, y: Any) -> Callable:
        second = _hazard_over_rate_second(getval(a), getval(y))[k]
        target = a if target_is_a else y
        return unbroadcast_f(target, lambda g: getval(g) * second)

    return make


defvjp(
    _hazard_over_rate,
    lambda ans, a, y: unbroadcast_f(
        a, lambda g: g * _hazard_over_rate_da(a, y)
    ),
    lambda ans, a, y: unbroadcast_f(
        y, lambda g: g * _hazard_over_rate_dy(a, y)
    ),
)
defvjp(_hazard_over_rate_da, _vjp(True, 0), _vjp(False, 1))
defvjp(_hazard_over_rate_dy, _vjp(True, 1), _vjp(False, 2))


def _fraction_jet(a: Any, y: Any, order: int = 0) -> tuple:
    """``(g, first, second)``: the continued fraction's tail ``g`` (see
    :func:`_hazard_over_rate`) with, at ``order`` 1, its derivative in
    ``a``, ``[g_a]``, and at ``order`` 2 its first derivatives ``[g_a,
    g_y]`` and second ``[g_aa, g_ay, g_yy]`` (``None`` where not taken),
    in plain numpy.

    ``g`` is evaluated forwards by the modified Lentz method
    (:func:`_fraction_value`). Its derivative in ``a``, which a gradient
    takes, is a complex step: the same loop at ``a + i h``, whose
    imaginary part is ``h g_a`` to rounding, with no difference taken
    (the step's error is ``h**2`` relative), in about the time of the
    value alone. The second derivatives, which only a Hessian takes, are
    carried along the loop's steps (:func:`_fraction_second`)."""
    if order == 2:
        return _fraction_second(a, y)
    if order == 1:
        a = onp.asarray(a, dtype=float)
        step = _COMPLEX_STEP * onp.maximum(a, 1.0)
        g = _fraction_value(a + 1j * step, y)
        return g.real, (g.imag / step)[None], None
    return _fraction_value(a, y), None, None


#: The complex step in ``a``, relative to it (at least 1), of
#: ``_fraction_jet``'s derivative.
_COMPLEX_STEP = 1e-20


def _fraction_value(a: Any, y: Any) -> Any:
    """The fraction's tail ``g`` (:func:`_hazard_over_rate`) by the
    modified Lentz method (Numerical Recipes, ``gcf``), until every
    point's has converged. ``a`` may be complex (a complex step,
    :func:`_fraction_jet`): the imaginary parts must have converged
    too."""
    tiny = 1e-300
    b = y + 3.0 - a
    c: Any = 1.0 / tiny
    d = 1.0 / b
    g = d
    stepped = onp.iscomplexobj(g)
    with onp.errstate(all="ignore"):
        for i in range(2, _FRACTION_TERMS):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = onp.where(onp.abs(d) < tiny, tiny, d)
            c = b + an / c
            c = onp.where(onp.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            delta = d * c
            last, g = g, g * delta
            if onp.all(onp.abs(delta - 1.0) <= _FRACTION_TOL) and (
                not stepped or _settled(last.imag, g.imag)
            ):
                break
    return g


def _fraction_second(a: Any, y: Any) -> tuple:
    """``(g, [g_a, g_y], [g_aa, g_ay, g_yy])``: the fraction's tail with
    its first and second derivatives, carried along each step of the
    Lentz loop (:func:`_fraction_value`) until every point's have
    converged."""
    a = onp.asarray(a, dtype=float)
    y = onp.asarray(y, dtype=float)
    shape = onp.broadcast_shapes(a.shape, y.shape)
    a = onp.broadcast_to(a, shape) if a.ndim else float(a)
    tiny = 1e-300
    b = onp.broadcast_to(y, shape) + 3.0 - a
    c = onp.full(b.shape, 1.0 / tiny)
    d = 1.0 / b
    g = d
    # Each quantity's derivatives are stacked: first [q_a, q_y], second
    # [q_aa, q_ay, q_yy]. b moves by -1 with a and +1 with y at every
    # step, and an = -i (i - a) by i with a.
    b1 = onp.array([-1.0, 1.0]).reshape((2,) + (1,) * b.ndim)
    an1 = onp.zeros_like(b1)
    c1 = onp.zeros((2,) + b.shape)
    d1 = -b1 * d * d
    g1 = d1
    c2 = onp.zeros((3,) + b.shape)
    d2 = _pairs(b1, b1) * d**3
    g2 = d2
    with onp.errstate(all="ignore"):
        for i in range(2, _FRACTION_TERMS):
            an = -i * (i - a)
            an1[0] = i
            b = b + 2.0
            big_d = an * d + b
            big_c = b + an / c
            keep_d = onp.abs(big_d) >= tiny
            keep_c = onp.abs(big_c) >= tiny
            big_d = onp.where(keep_d, big_d, tiny)
            big_c = onp.where(keep_c, big_c, tiny)
            new_d = 1.0 / big_d
            delta = new_d * big_c
            w = 1.0 / c
            w1 = -c1 * w * w
            w2 = (-c2 + _pairs(c1, c1) * w) * w * w
            # (a clamped term is a constant)
            D1 = (an1 * d + an * d1 + b1) * keep_d
            C1 = (b1 + an1 * w + an * w1) * keep_c
            D2 = (_pairs(an1, d1) + an * d2) * keep_d
            C2 = (_pairs(an1, w1) + an * w2) * keep_c
            nd1 = -D1 * new_d * new_d
            nd2 = (-D2 + _pairs(D1, D1) * new_d) * new_d * new_d
            e1 = nd1 * big_c + new_d * C1
            e2 = nd2 * big_c + _pairs(nd1, C1) + new_d * C2
            new_g1 = g1 * delta + g * e1
            new_g2 = g2 * delta + _pairs(g1, e1) + g * e2
            done = (
                onp.all(onp.abs(delta - 1.0) <= _FRACTION_TOL)
                and _settled(g1, new_g1)
                and _settled(g2, new_g2)
            )
            g1, c1, d1 = new_g1, C1, nd1
            g2, c2, d2 = new_g2, C2, nd2
            c, d = big_c, new_d
            g = g * delta
            if done:
                break
    return g, g1, g2


def _pairs(u: Any, v: Any) -> Any:
    """``[2 u_a v_a, u_a v_y + u_y v_a, 2 u_y v_y]`` from first
    derivatives ``u = [u_a, u_y]`` and ``v``: the second derivatives of
    a product ``u v`` carried by its factors' first."""
    return u[_FIRST] * v[_SECOND] + u[_SECOND] * v[_FIRST]


#: The rows of first derivatives each of [aa, ay, yy] pairs (``_pairs``).
_FIRST = [0, 0, 1]
_SECOND = [0, 1, 1]


def _settled(old: Any, new: Any) -> bool:
    """Whether every one of the derivatives ``new`` is within the
    fraction's tolerance of ``old``."""
    return bool(onp.all(onp.abs(new - old) <= _FRACTION_TOL * onp.abs(new)))


#: The last jet taken (``_cached_jet``): ``(key, order, jet)``.
_LAST_JET: list = [None]


def _cached_jet(a: Any, y: Any, order: int) -> tuple:
    """``_fraction_jet(a, y, order)``, or the last one taken where it was
    at the same point to at least that order: the hazard, its
    derivatives and their derivatives each come from one jet."""
    a = onp.asarray(a, dtype=float)
    y = onp.asarray(y, dtype=float)
    key = (a.shape, y.shape, a.tobytes(), y.tobytes())
    last = _LAST_JET[0]
    if last is not None and last[0] == key and last[1] >= order:
        return last[2]
    jet = _fraction_jet(a, y, order)
    _LAST_JET[0] = (key, order, jet)
    return jet


def _plain_float(v: Any) -> Any:
    return v if onp.ndim(v) else float(v)


class Gamma_(OptimisedFitMixin, ParametricFitter):
    r"""
    The Gamma distribution: the time to the ``alpha``-th event of a
    Poisson process of rate ``beta`` (for a whole ``alpha``), so a model
    of failures that need several shocks; with ``alpha = 1`` it is the
    Exponential.

    The parameters are the shape ``alpha`` and the rate ``beta``, both
    positive. On the support :math:`(0, \infty)`, with :math:`\gamma` the
    lower incomplete gamma function,

    .. math::
        R(x) = 1 - \frac{\gamma \left ( \alpha, \beta x \right )}
        {\Gamma \left ( \alpha \right )}.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import Gamma
    >>> model = Gamma.from_params([2, 0.1])
    >>> model.sf([10, 20, 40]).round(4)
    array([0.7358, 0.406 , 0.0916])
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="linear",
        )
        # The Gamma has no linearising probability plot, for the same
        # reason as the Beta above it: the CDF is the regularised
        # incomplete gamma function, and the shape sits *inside* that
        # special function rather than outside it as an exponent. The
        # only straight-line y-axis is the inverse incomplete gamma,
        # which needs the shape -- so to draw the axis you need the
        # answer, and to get the answer you need the axis.
        #
        # MPP broke the circle by guessing the shape from moments,
        # drawing the plot on that guess and regressing. When the guess
        # is off the axis is the wrong axis, the points are no longer
        # straight on it, and the regression fits a line through a
        # curve -- returning a confident, wrong estimate rather than an
        # error. An offset makes it worse: the shift distorts the low-x
        # end hardest, which is exactly where the shape information is.
        #
        # Fit by MLE (the default), MSE or MOM instead. ``plot()`` still
        # works, because it transforms with the *fitted* parameters, so
        # the axis is the right one by the time it is drawn.
        self.supports_mpp = False

    def _offset_limit_family(self) -> Any:
        """The ``Normal``: as the offset runs to -inf with the shape -> inf,
        the shifted Gamma tends to a Normal distribution (the central
        limit theorem) (#599; see
        ``OptimisedFitMixin._offset_limit_family``)."""
        from surpyval.univariate.parametric import Normal

        return Normal

    @staticmethod
    def _moment_estimate(x: npt.NDArray) -> tuple[float, float]:
        """Closed-form approximation to the Gamma MLE.

        The shape solves ``log(alpha) - digamma(alpha) = s`` with
        ``s = log(mean x) - mean(log x)``; this is the standard
        approximation to that root (Minka 2002, after Thom 1958), good
        to about 1.5% and used only as a starting point.
        """
        s = np.log(x.sum() / len(x)) - np.log(x).sum() / len(x)
        # s is exactly zero for a tied sample -- the log of the mean and
        # the mean of the logs coincide -- and alpha divides by it, so
        # the seed comes back as (inf, inf). A failed optimiser falls
        # back to its initial guess (#261), so those infinities are
        # returned to the caller as the fitted parameters. Seed the
        # exponential case instead: a tied sample carries no information
        # about the shape.
        if not np.isfinite(s) or s <= np.finfo(float).tiny:
            return 1.0, len(x) / x.sum()
        alpha = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        beta = x.sum() / (len(x) * alpha)
        return alpha, 1.0 / beta

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            # The moments are taken *after* the shift by the starting
            # offset (``_offset_seed``). On offset data
            # ``s = log(mean x) - mean(log x)`` is squashed towards zero
            # by the constant, and since alpha grows like ``1 / 12s`` the
            # estimate explodes: 649 for a true shape of 3, which made
            # MSE and MOM offset fits return silent nonsense.
            return self._offset_seed(data)
        return np.asarray(self._moment_estimate(data.x), dtype=float)

    def sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Survival (or Reliability) function for the Gamma Distribution:

        .. math::
            R(x) = 1 - \frac{\gamma \left ( \alpha, \beta x \right )
            }{\Gamma \left ( \alpha \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution


        Returns
        -------

        sf : scalar or numpy array
            The value(s) for the survival function at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.sf(x, 3, 2)
        array([0.67667642, 0.23810331, 0.0619688 , 0.01375397, 0.0027694 ])
        """
        # the upper incomplete gamma directly, not 1 - P: the difference is
        # 0 past survival ~1e-16
        return np.exp(self.log_sf(x, alpha, beta))

    def ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        CDF (or unreliability or failure) function for the Gamma Distribution:

        .. math::
            F(x) = \frac{\gamma \left ( \alpha, \beta x \right )}
            {\Gamma \left ( \alpha \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution


        Returns
        -------

        ff : scalar or numpy array
            The value(s) for the CDF at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.ff(x, 3, 2)
        array([0.32332358, 0.76189669, 0.9380312 , 0.98624603, 0.9972306 ])
        """
        x = np.array(x)
        return agammainc(alpha, beta * x)

    def df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Density function for the Gamma Distribution:

        .. math::
            f(x) = \frac{\beta^{\alpha }}{\Gamma \left ( \alpha \right )}
            x^{\alpha - 1}e^{-\beta x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution


        Returns
        -------

        df : scalar or numpy array
            The density of the distribution at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.df(x, 3, 2)
        array([0.54134113, 0.29305022, 0.08923508, 0.02146961, 0.00453999])
        """
        # exp(log_df): beta^alpha and x^(alpha - 1) overflow separately
        # at a large shape, to inf * 0 = NaN or an OverflowError (#444,
        # #445)
        return np.exp(self.log_df(x, alpha, beta))

    def hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Gamma Distribution:

        .. math::
            h(x) = \frac{\frac{\beta^{\alpha }}{\Gamma \left ( \alpha \right )
            }x^{\alpha - 1}e^{-\beta x}}{1 - \frac{\gamma \left ( \alpha, \beta
            x \right )}{\Gamma \left ( \alpha \right )}}

        In the tail (:math:`\beta x` past 30 and past
        :math:`\alpha + 2\sqrt{\alpha} + 1`) it is taken from the continued
        fraction of the upper incomplete gamma, in which the density's and
        the survival function's :math:`e^{-\beta x}` cancel exactly: the
        quotient of the two loses :math:`\beta x` times the machine
        precision to rounding (every digit by :math:`10^{15}`), and its
        derivatives in ``x`` and ``alpha`` a factor :math:`\beta x` more.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution


        Returns
        -------

        hf : scalar or numpy array
            The instantaneous hazard rate of the distribution at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.hf(x, 3, 2)
        array([0.8       , 1.23076923, 1.44      , 1.56097561, 1.63934426])
        """
        x = np.asarray(x) if not isbox(x) else x
        y = beta * x
        # f / S is a difference of two logs of size y: it loses y eps to
        # rounding (4e-9 at y = 1e8, every digit past 1e15, #760), and
        # its derivatives a factor y more (#777). In the tail their e**-y
        # is cancelled exactly, by the continued fraction of the upper
        # incomplete gamma (``_hazard_over_rate``)
        # (which points are in the tail, and the stand-ins below, are
        # plain values: traced, they were operations in every gradient)
        y_v, a_v = getval(y), getval(alpha)
        start = a_v + 2.0 * np.sqrt(a_v) + 1.0
        tail = (y_v > _FRACTION_FROM) & (y_v < np.inf) & (y_v > start)
        if not np.any(tail):
            return self._hf_from_logs(x, alpha, beta)
        # each branch at a point it is finite at (y = 1 in the body, and
        # well inside the tail in the tail, where the fraction converges
        # fast), so neither puts a nan into the other's gradient
        body = self._hf_from_logs(
            np.where(tail, 1.0 / getval(beta), x), alpha, beta
        )
        y_tail = np.where(tail, y, 2.0 * (start + _FRACTION_FROM))
        if isbox(alpha):
            # the fraction with its slope in alpha, which the gradient
            # takes, in the one pass (``_cached_jet``)
            _cached_jet(a_v, getval(y_tail), 1)
        ratio = _hazard_over_rate(alpha, y_tail)
        return np.where(tail, beta * ratio, body)

    def _hf_from_logs(
        self, x: Numeric, alpha: Boxable, beta: Boxable
    ) -> Boxable:
        """The hazard as ``exp(log f - log S)``."""
        # in logs, so the ratio stays finite deep in the tail
        log_sf = self.log_sf(x, alpha, beta)
        gone = log_sf == -np.inf
        if not np.any(gone):
            return np.exp(self.log_df(x, alpha, beta) - log_sf)
        # where both logs are -inf (at x = inf, or where beta x
        # overflows), the hazard's limit, the rate beta (#561)
        with np.errstate(invalid="ignore"):
            out = np.exp(self.log_df(x, alpha, beta) - log_sf)
        return np.where(gone, beta + np.zeros_like(out), out)

    def Hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Gamma Distribution:

        .. math::
            H(x) = -\ln(1 - \frac{\gamma \left ( \alpha, \beta x \right )}
            {\Gamma \left ( \alpha \right )})

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution


        Returns
        -------

        Hf : scalar or numpy array
            The cumulative hazard rate of the distribution at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.Hf(x, 3, 2)
        array([0.39056209, 1.43505064, 2.78112418, 4.28642793, 5.88912614])
        """
        return -self.log_sf(x, alpha, beta)

    def qf(self, u: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Quantile function for the Gamma Distribution:

        .. math::
            q(u) = \frac{P^{-1} \left ( \alpha, u \right )}{\beta}

        where :math:`P^{-1}` inverts the regularised lower incomplete gamma
        function :math:`P(\alpha, z) = \gamma(\alpha, z) / \Gamma(\alpha)`.

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Gamma distribution at each value u.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Gamma.qf(u, 3, 4)
        array([0.27551633, 0.38376105, 0.47844395, 0.57126923, 0.66851508])
        """
        return gammaincinv(alpha, u) / beta

    def mean(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Calculates the mean of the Gamma distribution with given parameters.

        .. math::
            E = \frac{\alpha}{\beta}

        Parameters
        ----------

        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Gamma distribution

        Examples
        --------
        >>> from surpyval import Gamma
        >>> Gamma.mean(3, 4)
        0.75
        """
        return alpha / beta

    def moment(self, m: int, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Calculates the m-th moment of the Gamma distribution with
        given parameters.

        .. math::
            E = \frac{\Gamma \left ( m + \alpha \right )}{\beta^{m}\Gamma
            \left ( \alpha \right )}

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution

        Returns
        -------

        mean : scalar or numpy array
            The moment(s) of the Gamma distribution

        Examples
        --------
        >>> from surpyval import Gamma
        >>> Gamma.moment(3, 3, 4)
        np.float64(0.9375)
        """
        return agamma(m + alpha) / (beta**m * agamma(alpha))

    def entropy(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the Gamma distribution.

        .. math::
            S = \alpha - \ln \left ( \beta \right ) + \ln \Gamma \left (
            \alpha \right ) + \left ( 1 - \alpha \right ) \psi \left (
            \alpha \right )

        Where psi is the digamma function

        Parameters
        ----------

        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the Gamma distribution

        Examples
        --------
        >>> from surpyval import Gamma
        >>> Gamma.entropy(3, 4)
        np.float64(0.46128414924312033)
        """
        return (
            alpha
            - np.log(beta)
            + agammaln(alpha)
            + (1 - alpha) * digamma(alpha)
        )

    def log_df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Calculates the log of the density function of the Gamma distribution
        at x.

        .. math::
            \log f(x) = \log \left ( \frac{\beta^{\alpha}}{\Gamma(\alpha)}
            x^{\alpha - 1}e^{-\beta x} \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution

        Returns
        -------

        log_df : scalar or numpy array
            The log of the density function of the Gamma distribution at x

        """
        # x = 0 is the limit of beta^alpha x^(alpha - 1) / Gamma(alpha):
        # the formula is 0 * log 0 = NaN there at alpha = 1 (#444).
        x_pos = positive_or_one(x)
        at_inf = x_pos == np.inf
        if np.any(at_inf):
            x_pos = np.where(at_inf, 1.0, x_pos)
        log_scale = alpha * np.log(beta) - agammaln(alpha)
        inside = log_scale + (alpha - 1) * np.log(x_pos) - beta * x_pos
        if np.any(at_inf):
            inside = np.where(at_inf, -np.inf, inside)
        return on_support(
            x, inside, lambda: power_at_zero(alpha - 1, log_scale)[1]
        )

    def log_ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        return agammaincln(alpha, beta * x)

    def log_sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        return agammainccln(alpha, beta * x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        alpha = params[0]
        return gammaincinv(alpha, y)

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        alpha = params[0]
        return agammainc(alpha, y)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x


Gamma: Gamma_ = Gamma_("Gamma")
