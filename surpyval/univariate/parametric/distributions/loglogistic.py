from __future__ import annotations

import autograd.numpy as np
import numpy.typing as npt
from autograd.scipy.special import expit
from scipy.stats import fisk

from surpyval.univariate.parametric._fit_inputs import _offset_start
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData

from ._stable import (
    log_ratio,
    on_support,
    positive_or_one,
    power_at_zero,
    softplus,
)


class LogLogistic_(OptimisedFitMixin, ParametricFitter):
    # The scale of the Wald band on sf and ff (Parametric._cb_sf_bound):
    # the logit of ff, on which this family is a straight line in
    # log time (#477).
    _cb_link = "logit"

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

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            # The data arrives already validated in xcnt form, so the
            # ``xcnt_handler`` round trip that used to open this branch
            # is gone. It never had anything to re-derive: no caller has
            # ever passed ``t`` down to an initialiser.
            #
            # alpha and beta are seeded as for an unshifted fit, from the
            # probability plot of the data shifted by the fitter's own
            # starting offset (see ``_offset_start``). The seed was the
            # sum of the unshifted values over the number of failures and
            # a shape of 2: for data well clear of zero the scale came
            # out several times too large, and the search's first step
            # to correct it overshot so far below zero, where the scale
            # is searched as a log, that it underflowed -- in data units
            # of thousands the fit ended on Nelder-Mead at the starting
            # offset.
            x, c, n = data.x, data.c, data.n
            gamma_init = _offset_start(x)
            shifted = self.fit_from_surpyval_data(
                SurpyvalData(x - gamma_init, c, n, group_and_sort=False),
                how="MPP",
            )
            return np.array([gamma_init, *shifted.params], dtype=float)
        else:
            return np.asarray(
                self.fit_from_surpyval_data(data, how="MPP").params,
                dtype=float,
            )

    def sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the LogLogistic Distribution:

        .. math::
            R(x) = 1 - \frac{1}{1 + \left ( x / \alpha \right )^{-\beta}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.sf(x, 3, 4)
        array([0.98780488, 0.83505155, 0.5       , 0.24035608, 0.11473088])
        """
        # The logistic function of -z, z = beta ln(x / alpha): exact
        # where (x / alpha)^beta under- or overflows. x = 0 is its limit,
        # 1 (the negative power raised or was NaN there, #280).
        z, x_pos = self._z(x, alpha, beta)
        return on_support(x, expit(-z), 1.0)

    def ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the LogLogistic
        Distribution:

        .. math::
            F(x) = \frac{1}{1 + \left ( x /\alpha \right )^{-\beta}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.ff(x, 3, 4)
        array([0.01219512, 0.16494845, 0.5       , 0.75964392, 0.88526912])
        """
        # The logistic function of z: z^beta / (1 + z^beta) was
        # inf / inf = NaN far right (#444), and the negative power of
        # 1 / (1 + z^-beta) raised or was NaN at x = 0 (#280).
        z, x_pos = self._z(x, alpha, beta)
        return on_support(x, expit(z), 0.0)

    def df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Density function for the LogLogistic Distribution:

        .. math::
            f(x) = \frac{\left ( \beta / \alpha \right ) \left ( x / \alpha
            \right )^{\beta - 1}}{\left ( 1 + \left ( x / \alpha
            \right )^{\beta} \right )^2}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.df(x, 3, 4)
        array([0.0481856 , 0.27548092, 0.33333333, 0.18258504, 0.08125416])
        """
        # exp(log_df): the direct form is inf / inf = NaN far right, and
        # 0 where its denominator overflows first (#444)
        return np.exp(self.log_df(x, alpha, beta))

    def hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the LogLogistic Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.hf(x, 3, 4)
        array([0.04878049, 0.32989691, 0.66666667, 0.75964392, 0.7082153 ])
        """
        # (beta / x) F in logs: the quotient f / R is NaN far right, where
        # both are 0 or the density is NaN (#444)
        z, x_pos = self._z(x, alpha, beta)
        inside = np.exp(np.log(beta) - np.log(x_pos) - softplus(-z))
        return on_support(x, inside, lambda: self._at_zero(alpha, beta)[0])

    def Hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the LogLogistic Distribution:

        .. math::
            H(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.Hf(x, 3, 4)
        array([0.01227009, 0.18026182, 0.69314718, 1.42563378, 2.16516608])
        """
        # log(1 + (x / alpha)^beta): -log(sf) is -0.0 where sf rounds to
        # 1 (#442) and inf where it underflows (#443)
        z, x_pos = self._z(x, alpha, beta)
        return on_support(x, softplus(z), 0.0)

    def qf(self, u: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Quantile function for the LogLogistic distribution:

        .. math::
            q(u) = \alpha \left ( \frac{u}{1 - u} \right )^{\frac{1}{\beta}}

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the LogLogistic distribution at each value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> LogLogistic.qf(u, 3, 4)
        array([1.73205081, 2.12132034, 2.42732013, 2.71080601, 3.        ])
        """
        return alpha * (u / (1 - u)) ** (1.0 / beta)

    def mean(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Mean of the LogLogistic distribution

        .. math::
            E = \frac{\alpha \pi / \beta}{sin \left ( \pi / \beta \right )}

        Parameters
        ----------

        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the LogLogistic distribution

        Examples
        --------
        >>> from surpyval import LogLogistic
        >>> LogLogistic.mean(3, 4)
        np.float64(3.332162203618775)
        """
        if beta > 1:
            return (alpha * np.pi / beta) / (np.sin(np.pi / beta))
        else:
            return np.nan

    @staticmethod
    def _z(x: Numeric, alpha: Boxable, beta: Boxable) -> tuple:
        """``z = beta ln(x / alpha)``, the logistic variable (``F`` is
        its logistic function), and ``x`` with the points at and below
        0 replaced by 1."""
        x_pos = positive_or_one(x)
        return beta * log_ratio(x_pos, alpha), x_pos

    @staticmethod
    def _at_zero(alpha: Boxable, beta: Boxable) -> tuple:
        """The density (and hazard) at x = 0 and its log, where it
        behaves like (beta / alpha) (x / alpha)^(beta - 1)."""
        return power_at_zero(beta - 1.0, np.log(beta) - np.log(alpha))

    def log_df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # ln(beta / x) + ln F + ln R. The limit at x = 0 is exact: the
        # formula was 0 * log 0 = NaN there at beta = 1 (#444).
        z, x_pos = self._z(x, alpha, beta)
        inside = np.log(beta) - np.log(x_pos) - softplus(-z) - softplus(z)
        return on_support(x, inside, lambda: self._at_zero(alpha, beta)[1])

    def log_sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # -log(1 + e^z): neither overflows (log(alpha^beta + x^beta) did
        # for beta ln(x) beyond ~709, #280) nor cancels (the difference
        # of two logs of that size lost the digits of a log near 0, #442)
        z, x_pos = self._z(x, alpha, beta)
        return on_support(x, -softplus(z), 0.0)

    def log_ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        z, x_pos = self._z(x, alpha, beta)
        return on_support(x, -softplus(-z), -np.inf)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = -np.log(1.0 / y[~mask] - 1)
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return 1.0 / (np.exp(-y) + 1)

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

    def moment(self, m: int, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""
        The ``m``-th raw moment :math:`E[X^{m}]` of the LogLogistic
        distribution. It exists only for :math:`\beta > m`; otherwise
        ``nan`` is returned.

        Examples
        --------
        >>> from surpyval import LogLogistic
        >>> LogLogistic.moment(2, 10, 3)
        np.float64(241.83991523122904)
        """
        return fisk.moment(m, beta, scale=alpha)

    def entropy(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the LogLogistic distribution.

        .. math::
            S = \ln \left ( \frac{\alpha}{\beta} \right ) + 2

        Parameters
        ----------

        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the LogLogistic distribution

        Examples
        --------
        >>> from surpyval import LogLogistic
        >>> LogLogistic.entropy(3, 4)
        np.float64(1.7123179275482192)
        """
        return np.log(alpha / beta) + 2


LogLogistic: LogLogistic_ = LogLogistic_("LogLogistic")
