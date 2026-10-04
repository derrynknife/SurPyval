from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from numpy import euler_gamma
from scipy.special import gamma as gamma_func

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData

from ._stable import (
    log1mexp,
    log_ratio,
    on_support,
    positive_or_one,
    power_at_zero,
)


class Weibull_(OptimisedFitMixin, ParametricFitter):
    r"""
    The Weibull distribution, the most used model of time to failure: a
    shape ``beta`` below 1 is a falling hazard (early failures), 1 a
    constant one (the Exponential) and above 1 a rising one (wear-out).

    The parameters are the scale ``alpha``, the time by which 63.2% of
    units have failed, and the shape ``beta``, both positive. On the
    support :math:`(0, \infty)`,

    .. math::
        R(x) = e^{-\left ( x / \alpha \right )^\beta}.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> model = Weibull.from_params([100, 2])
    >>> model.sf([50, 100, 150]).round(4)
    array([0.7788, 0.3679, 0.1054])
    >>> x = [12.0, 25.0, 31.0, 40.0, 48.0, 55.0, 63.0, 71.0, 84.0, 102.0]
    >>> Weibull.fit(x).params.round(2)
    array([60.01,  2.14])
    """

    # The scale of the Wald band on sf and ff (Parametric._cb_sf_bound):
    # log(-log sf), on which this family is a straight line in
    # log time (#477).
    _cb_link = "loglog"

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="log",
        )

    def _offset_limit_family(self) -> Any:
        """The ``Gumbel``: as the offset runs to -inf with beta -> inf,
        ``gamma + alpha W^(1/beta)`` tends to
        ``gamma + alpha (1 + log(W) / beta)``, a smallest extreme value
        distribution (#599; see
        ``OptimisedFitMixin._offset_limit_family``)."""
        from surpyval.univariate.parametric import Gumbel

        return Gumbel

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            # The probability plot of the data shifted by the starting
            # offset (``_offset_seed``). The plot's own offset fit, whose
            # offset was then replaced, was not one distribution (#622).
            return self._offset_seed(data)
        mpp_model = self.fit_from_surpyval_data(
            data, how="MPP", heuristic="Nelson-Aalen"
        )
        return np.asarray(mpp_model.params, dtype=float)

    def sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the Weibull Distribution:

        .. math::
            R(x) = e^{-\left ( \frac{x}{\alpha} \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.sf(x, 3, 4)
        array([9.87730216e-01, 8.20754808e-01, 3.67879441e-01, 4.24047953e-02,
               4.45617596e-04])
        """
        return np.exp(-((x / alpha) ** beta))

    def ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Weibull Distribution:

        .. math::
            F(x) = 1 - e^{-\left ( \frac{x}{\alpha} \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.ff(x, 3, 4)
        array([0.01226978, 0.17924519, 0.63212056, 0.9575952 , 0.99955438])
        """
        # np.expm1 is accurate for small values of x while being the
        # same as np.exp for large values
        return -np.expm1(-((x / alpha) ** beta))

    def df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Density function for the Weibull Distribution:

        .. math::
            f(x) = \frac{\beta}{\alpha} \left ( \frac{x}{\alpha}
            \right )^{\beta - 1} e^{-\left ( \frac{x}{\alpha}
            \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.df(x, 3, 4)
        array([0.0487768 , 0.32424881, 0.49050592, 0.13402009, 0.00275073])
        """
        # At x = 0 with beta < 1, 0 ** (beta - 1) is inf: the density
        # really is unbounded there.
        with np.errstate(divide="ignore"):
            power = (x / alpha) ** (beta - 1)
        sf = np.exp(-((x / alpha) ** beta))
        # Far in the tail the power overflows where sf is 0: the density
        # is 0 there, not inf * 0 (#561)
        return (beta / alpha) * np.where(sf == 0, 0.0, power) * sf

    def hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Weibull Distribution:

        .. math::
            h(x) = \frac{\beta}{\alpha} \left ( \frac{x}{\alpha} \right
            )^{\beta - 1}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.hf(x, 3, 4)
        array([0.04938272, 0.39506173, 1.33333333, 3.16049383, 6.17283951])
        """
        # At x = 0 with beta < 1, 0 ** (beta - 1) is inf: the hazard
        # really is unbounded there.
        with np.errstate(divide="ignore"):
            return (beta / alpha) * (x / alpha) ** (beta - 1)

    def Hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Weibull Distribution:

        .. math::
            H(x) = \left ( \frac{x}{\alpha} \right )^{\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.Hf(x, 3, 4)
        array([0.01234568, 0.19753086, 1.        , 3.16049383, 7.71604938])
        """
        return (x / alpha) ** beta

    def qf(self, u: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Quantile function for the Weibull distribution:

        .. math::
            q(u) = \alpha \left ( -\ln \left ( 1 - u \right ) \right )^{1/
            \beta}

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Weibull distribution at each value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Weibull.qf(u, 3, 4)
        array([1.70919151, 2.06189877, 2.31840554, 2.5362346 , 2.73733292])
        """
        return alpha * (-np.log1p(-u)) ** (1 / beta)

    def mean(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Mean of the Weibull distribution

        .. math::
            E = \alpha \Gamma \left ( 1 + \frac{1}{\beta} \right )

        Parameters
        ----------

        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Weibull distribution

        Examples
        --------
        >>> from surpyval import Weibull
        >>> Weibull.mean(3, 4)
        np.float64(2.7192074311664314)
        """
        return alpha * gamma_func(1 + 1.0 / beta)

    def moment(self, m: int, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        m-th moment of the Weibull distribution

        .. math::
            M(m) = \alpha^m \Gamma \left ( 1 + \frac{m}{\beta} \right )

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        mean : scalar or numpy array
            The moment(s) of the Weibull distribution

        Examples
        --------
        >>> from surpyval import Weibull
        >>> Weibull.moment(2, 3, 4)
        np.float64(7.976042329074821)
        """
        return alpha**m * gamma_func(1 + m / beta)

    def entropy(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""
        Differential entropy of the Weibull distribution,

        .. math::
            S = \gamma_{e}\left(1 - \frac{1}{\beta}\right)
                + \ln\frac{\alpha}{\beta} + 1,

        with :math:`\gamma_{e}` the Euler-Mascheroni constant.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> Weibull.entropy(10, 2)
        np.float64(2.898045744884867)
        """
        return euler_gamma * (1 - 1 / beta) + np.log(alpha) - np.log(beta) + 1

    def log_df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # x = 0 is the limit of (beta / alpha) (x / alpha)^(beta - 1):
        # the formula is 0 * log 0 = NaN there at beta = 1 (#444).
        x_pos = positive_or_one(x)
        at_inf = x_pos == np.inf
        if np.any(at_inf):
            x_pos = np.where(at_inf, 1.0, x_pos)
        log_scale = np.log(beta) - np.log(alpha)
        with np.errstate(over="ignore"):
            t = (x_pos / alpha) ** beta
        inside = log_scale + (beta - 1) * log_ratio(x_pos, alpha) - t
        if np.any(at_inf):
            inside = np.where(at_inf, -np.inf, inside)
        return on_support(
            x, inside, lambda: power_at_zero(beta - 1, log_scale)[1]
        )

    def log_sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        return -((x / alpha) ** beta)

    def log_ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # log(1 - e^-t) from t and log t: exact where F rounds to 1 (the
        # generic log(-expm1(-t)) is 0 there, #442) and finite where t
        # underflows (#443).
        x_pos = positive_or_one(x)
        log_t = beta * log_ratio(x_pos, alpha)
        with np.errstate(over="ignore"):
            t = (x_pos / alpha) ** beta
        log_ff, _ = log1mexp(t, log_t)
        return on_support(x, log_ff, -np.inf)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log(1 - y[~mask]))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return 1 - np.exp(-np.exp(y))

    def unpack_rr(
        self, params: npt.NDArray, rr: str
    ) -> tuple[Boxable, Boxable]:
        if rr == "y":
            beta = params[0]
            alpha = np.exp(params[1] / -beta)
        elif rr == "x":
            beta = 1.0 / params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta


# Deliberately not annotated ``: ParametricFitter``. That declaration
# erases the concrete type, and the base class declares none of sf, ff,
# df, hf, Hf, qf or mean -- so with py.typed shipped, the example in
# ``sf``'s own docstring did not type check for a user:
#
#     Weibull.sf(x, 3, 4)
#     error: "ParametricFitter" has no attribute "sf"
#
# Naming ``Weibull_`` exposes the real signatures, which is what makes
# the annotations on them worth having. Anywhere a ``ParametricFitter``
# is wanted this still is one.
#
# The annotation cannot simply be dropped and inferred: the regression
# subpackages and ``fit_best`` import this name, and without an explicit
# type mypy cannot resolve it through that cycle ("Cannot determine type
# of Weibull", in six modules).
Weibull: Weibull_ = Weibull_("Weibull")
