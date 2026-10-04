from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from surpyval.univariate import parametric as para
from surpyval.univariate.parametric.fitters.closed_form import (
    is_uncensored_and_untruncated,
    weighted_mean_and_std,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils import normal as norm
from surpyval.utils.surpyval_data import SurpyvalData

from ._stable import normal_hazard, on_support, positive_or_one


def _log_pos(x: Numeric) -> Boxable:
    """``log(x)`` for x > 0, and 0 (a stand-in the caller replaces) at and
    below 0."""
    return np.log(positive_or_one(x))


class LogNormal_(OptimisedFitMixin, ParametricFitter):
    r"""
    The LogNormal distribution: the logarithm of the time to failure is
    Normal. It suits failures from the product of many small effects
    (fatigue, corrosion, repair times). ``Galton`` is the same
    distribution under another name.

    The parameters are ``mu`` and ``sigma`` (positive), the mean and the
    standard deviation of the log of the time. On the support
    :math:`(0, \infty)`, with :math:`\Phi` the standard normal CDF,

    .. math::
        R(x) = 1 - \Phi \left( \frac{\ln(x) - \mu}{\sigma} \right ).

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import LogNormal
    >>> model = LogNormal.from_params([4.0, 0.5])
    >>> model.sf([30, 55, 100]).round(4)
    array([0.8845, 0.4941, 0.1131])
    """

    # The scale of the Wald band on sf and ff (Parametric._cb_sf_bound):
    # the normal quantile of ff, on which this family is a straight line in
    # log time (#477).
    _cb_link = "probit"

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            # mu is the mean of the *log* data — any real number. A (0, None)
            # bound made every fit with geometric mean < 1 fail (#257).
            bounds=((None, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["mu", "sigma"],
            param_map={"mu": 0, "sigma": 1},
            plot_x_scale="log",
            y_ticks=[
                0.001,
                0.01,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.99,
                0.999,
            ],
        )

    def _offset_limit_family(self) -> Any:
        """The ``Normal``: as the offset runs to -inf with sigma -> 0,
        ``gamma + exp(mu + sigma Z)`` tends to a Normal distribution
        (#599; see ``OptimisedFitMixin._offset_limit_family``)."""
        from surpyval.univariate.parametric import Normal

        return Normal

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            # mu and sigma from the data shifted by the starting offset
            # (``_offset_seed``), where the log transform is defined
            return self._offset_seed(data)
        x, c, n = data.x, data.c, data.n
        norm_mod = para.Normal.fit(np.log(x), c=c, n=n, how="MLE")
        mu, sigma = norm_mod.params
        return np.array([mu, sigma], dtype=float)

    def _closed_form_mle(self, data: SurpyvalData) -> npt.NDArray | None:
        r"""Exact MLE on complete data: the Normal closed form applied to
        :math:`\log x`, since the parameters are those of the underlying
        normal. Censoring or truncation fall back to the optimiser for
        the same reason they do for the Normal.
        """
        if not is_uncensored_and_untruncated(data):
            return None
        x = np.asarray(data.x, dtype=float)
        if x.ndim != 1 or not (x > 0).all():
            return None
        return weighted_mean_and_std(np.log(x), data.n)

    def sf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Survival (or Reliability) function for the LogNormal Distribution:

        .. math::
            R(x) = 1 - \Phi \left( \frac{\ln(x) - \mu}{\sigma} \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogNormal.sf(x, 3, 4)
        array([0.77337265, 0.71793339, 0.68273014, 0.65668272, 0.63594491])
        """
        # norm.sf, not 1 - cdf: the difference is 0 past survival ~1e-16.
        # log(0) = -inf gives sf(0) = 1, which is right.
        with np.errstate(divide="ignore"):
            log_x = np.log(x)
        return norm.sf(log_x, mu, sigma)

    def ff(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the LogNormal Distribution:

        .. math::
            F(x) = \Phi \left( \frac{\ln(x) - \mu}{\sigma} \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogNormal.ff(x, 3, 4)
        array([0.22662735, 0.28206661, 0.31726986, 0.34331728, 0.36405509])
        """
        return on_support(x, norm.cdf(_log_pos(x), mu, sigma), 0.0)

    def df(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Density function for the LogNormal Distribution:

        .. math::
            f(x) = \frac{1}{x \sigma \sqrt{2\pi}}e^{-\frac{1}{2}\left (
                \frac{\ln x - \mu}{\sigma} \right )^{2}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogNormal.df(x, 3, 4)
        array([0.07528436, 0.04222769, 0.02969364, 0.02298522, 0.01877747])
        """
        # 0 at x = 0, where the formula is inf * 0 = NaN (#444)
        return np.exp(self.log_df(x, mu, sigma))

    def hf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the LogNormal Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogNormal.hf(x, 3, 4)
        array([0.09734551, 0.05881839, 0.04349249, 0.03500202, 0.02952687])
        """
        # the normal hazard of log x, which stays finite where the
        # density and survival function underflow; 0 at x = 0 and inf
        finite = (x > 0) & (x < np.inf)
        x_pos = np.where(finite, x, 1.0)
        z = (np.log(x_pos) - mu) / sigma
        scale = sigma * x_pos
        if np.any(scale == 0):
            # sigma x underflows at a subnormal x with a small sigma: 0 / 0
            # where the hazard is 0, divided out one factor at a time (#561)
            scale = np.where(scale == 0, 1.0, scale)
            hz = normal_hazard(z)
            ratio = np.where(
                sigma * x_pos == 0, hz / sigma / x_pos, hz / scale
            )
            return on_support(x, np.where(finite, ratio, 0.0), 0.0)
        inside = np.where(finite, normal_hazard(z) / scale, 0.0)
        return on_support(x, inside, 0.0)

    def Hf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the LogNormal Distribution:

        .. math::
            H(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogNormal.Hf(x, 3, 4)
        array([0.25699427, 0.33137848, 0.3816556 , 0.4205543 , 0.45264333])
        """
        return -self.log_sf(x, mu, sigma)

    def qf(self, u: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Quantile function for the LogNormal Distribution:

        .. math::
            q(u) = e^{\mu + \sigma \Phi^{-1} \left( u \right )}

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the LogNormal distribution at each value u.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> u = np.array([0.1, 0.2, 0.3, 0.4])
        >>> LogNormal.qf(u, 3, 4)
        array([0.11928899, 0.69316658, 2.46550819, 7.29078766])
        """
        return np.exp(norm.ppf(u, mu, sigma))

    def mean(self, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Mean of the LogNormal Distribution:

        .. math::
            E = e^{\mu + \frac{\sigma^2}{2}}

        Parameters
        ----------

        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean of the LogNormal distribution.

        Examples
        --------
        >>> from surpyval import LogNormal
        >>> LogNormal.mean(3, 4)
        np.float64(59874.14171519782)
        """
        return np.exp(mu + (sigma**2) / 2)

    def moment(self, m: int, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        m-th (non central) moment of the LogNormal distribution

        .. math::
            E\left [ X^{m} \right ] = e^{m\mu + \frac{m^{2}\sigma^{2}}{2}}

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the LogNormal distribution

        Examples
        --------
        >>> from surpyval import LogNormal
        >>> LogNormal.moment(2, 3, 4)
        np.float64(3.1855931757113756e+16)
        """
        return np.exp(m * mu + (m**2 * sigma**2) / 2)

    def entropy(self, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the LogNormal distribution.

        .. math::
            S = \mu + \frac{1}{2} \ln \left ( 2\pi e \sigma^{2} \right )

        Parameters
        ----------

        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the LogNormal distribution

        Examples
        --------
        >>> from surpyval import LogNormal
        >>> LogNormal.entropy(3, 4)
        np.float64(5.805232894324563)
        """
        return mu + 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    def log_df(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        # -inf at x = 0, where the formula is inf - inf = NaN (#444)
        log_x = _log_pos(x)
        inside = -log_x + norm.logpdf(log_x, mu, sigma)
        return on_support(x, inside, -np.inf)

    def log_ff(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        return on_support(x, norm.logcdf(_log_pos(x), mu, sigma), -np.inf)

    def log_sf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        return on_support(x, norm.logsf(_log_pos(x), mu, sigma), 0.0)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return para.Normal.qf(y, 0, 1)

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return para.Normal.ff(y, 0, 1)

    def unpack_rr(
        self, params: npt.NDArray, rr: str
    ) -> tuple[Boxable, Boxable]:
        if rr == "y":
            sigma, mu = params
            mu = -mu / sigma
            sigma = 1.0 / sigma
        elif rr == "x":
            sigma, mu = params
        return mu, sigma

    def _mom(self, x: npt.NDArray) -> tuple[float, float]:
        norm_mod = para.Normal.fit(np.log(x), how="MOM")
        mu, sigma = norm_mod.params
        return mu, sigma

    def _mom_offset(self, x: npt.NDArray) -> "npt.NDArray | None":
        r"""The offset LogNormal matching the sample's mean, variance and
        third central moment exactly, as ``(gamma, mu, sigma)``; ``None``
        where no such model exists and the optimiser has to find the
        closest one.

        The skewness depends on sigma alone, through ``w = exp(sigma^2)``:

        .. math::
            g = (w + 2)\sqrt{w - 1},

        so with ``t = w - 1`` it is the root of ``t (t + 3)^2 = g^2``,
        which increases in ``t`` and lies in ``(0, g^2 / 9]``. The
        variance ``e^{2\mu} w (w - 1)`` then gives mu, and the mean the
        offset.

        The optimiser reached the same answer only from some starting
        points. The mismatch has a second, spurious minimum with the
        offset pressed against the first observation (0.19 on a sample
        whose moments are matched exactly at an offset 24 below it), and
        which one a search ends in depends on its route -- on how far it
        steps in mu, a log-location whose starting magnitude moves with
        the data's units. The same sample fitted by moments in five
        different units came back with the offset at the first
        observation in three of them.

        None is returned for a sample with no positive skew (a LogNormal
        is always right skewed) and where the solution puts the offset
        at or above the smallest value.
        """
        x = np.asarray(x, dtype=float)
        m1 = x.mean()
        m2 = np.mean((x - m1) ** 2)
        m3 = np.mean((x - m1) ** 3)
        if not (m2 > 0 and m3 > 0):
            return None
        g2 = m3**2 / m2**3
        t = brentq(
            lambda t: t * (t + 3) ** 2 - g2,
            0.0,
            g2 / 9.0,
            xtol=1e-300,
            rtol=4 * np.finfo(float).eps,
        )
        if not t > 0:
            return None
        sigma = np.sqrt(np.log1p(t))
        mu = 0.5 * np.log(m2 / ((1 + t) * t))
        gamma = m1 - np.sqrt(m2 / t)
        if not (np.isfinite([gamma, mu, sigma]).all() and gamma < x.min()):
            return None
        return np.array([gamma, mu, sigma])


LogNormal: LogNormal_ = LogNormal_("LogNormal")


Galton: LogNormal_ = LogNormal_("Galton")
