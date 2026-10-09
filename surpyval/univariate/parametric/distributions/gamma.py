from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt
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
#: converges there within 100 terms for any shape (5 for a shape below
#: 100). Below it ``f / S`` loses about ``y eps`` to rounding (at most
#: 1.5e-13 for a shape below 100), and costs a fraction of what the
#: fraction does where autograd traces it (a Gamma PH fit's likelihood).
_FRACTION_FROM = 1000.0
#: The fraction's most terms, and its convergence test.
_FRACTION_TERMS = 500
_FRACTION_TOL = 4e-16


def _hazard_over_rate(a: Boxable, y: Boxable) -> Boxable:
    r"""
    The standard Gamma's hazard at ``y`` (the Gamma's over its rate, at
    :math:`y = \beta x`), for :math:`y > a + 1`, from Legendre's continued
    fraction of the upper incomplete gamma,

    .. math::
        \Gamma(a, y) = \frac{y^{a} e^{-y}}{y + 1 - a -
        \frac{1 (1 - a)}{y + 3 - a - \frac{2 (2 - a)}{y + 5 - a -
        \cdots}}}.

    The hazard :math:`y^{a - 1} e^{-y} / \Gamma(a, y)` is that
    denominator over :math:`y`, which is :math:`1 + (1 - a)(1 - g) / y`
    with :math:`g = 1 / (y + 3 - a - \cdots)`, the fraction from its
    second term: its excess over 1 is carried as itself, so neither the
    value nor its derivative in ``y`` (of size :math:`y^{-2}`) is a
    difference of terms of size 1. ``g`` is evaluated forwards by the
    modified Lentz method (Numerical Recipes, ``gcf``): from :math:`y =
    a + 1` up, the hazard is within 1e-15 of mpmath's (5e-15 at a shape
    of 1e4). Written in ``autograd.numpy``, so it is differentiable; the
    loop stops once every point has converged.
    """
    tiny = 1e-300
    b = y + 3.0 - a
    c: Boxable = 1.0 / tiny
    d = 1.0 / b
    g = d
    for i in range(2, _FRACTION_TERMS):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = b + an / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        g = g * delta
        if np.all(np.abs(getval(delta) - 1.0) <= _FRACTION_TOL):
            break
    return 1.0 + (1.0 - a) * (1.0 - g) / y


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

        Far in the tail (:math:`\beta x` past 1000 and past
        :math:`\alpha + 2\sqrt{\alpha} + 1`) it is taken from the continued
        fraction of the upper incomplete gamma, in which the density's and
        the survival function's :math:`e^{-\beta x}` cancel exactly: the
        quotient of the two loses :math:`\beta x` times the machine
        precision to rounding (every digit by :math:`10^{15}`).

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
        # rounding (4e-9 at y = 1e8, every digit past 1e15, #760). In the
        # tail their e**-y is cancelled exactly, by the continued
        # fraction of the upper incomplete gamma (``_hazard_over_rate``)
        tail = (y > _FRACTION_FROM) & (y < np.inf)
        tail = tail & (y > alpha + 2.0 * np.sqrt(alpha) + 1.0)
        if not np.any(tail):
            return self._hf_from_logs(x, alpha, beta)
        # each branch at a point it is finite at (y = 1 in the body, and
        # well inside the tail in the tail, where the fraction converges
        # fast), so neither puts a nan into the other's gradient
        body = self._hf_from_logs(np.where(tail, 1.0 / beta, x), alpha, beta)
        far = 2.0 * (alpha + 2.0 * np.sqrt(alpha) + 1.0 + _FRACTION_FROM)
        y_tail = np.where(tail, y, far)
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
