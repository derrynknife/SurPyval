import numpy.typing as npt
from autograd.scipy.stats import norm
from scipy.stats import norm as scipy_norm

from surpyval import np
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
from surpyval.utils.surpyval_data import SurpyvalData


class LogNormal_(OptimisedFitMixin, ParametricFitter):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            # mu is the mean of the *log* data — any real number. A (0, None)
            # bound made every fit with geometric mean < 1 fail (#257).
            bounds=((None, None), (0, None)),
            support=(0, np.inf),
            param_names=["mu", "sigma"],
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

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        x, c, n = data.x, data.c, data.n
        if offset:
            # Shift the data so the log transform is defined, then
            # initialise mu and sigma from the shifted data
            gamma_init = np.min(x) - 1.0
            norm_mod = para.Normal.fit(
                np.log(x - gamma_init), c=c, n=n, how="MLE"
            )
            mu, sigma = norm_mod.params
            return np.array([gamma_init, mu, sigma], dtype=float)
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
        return 1 - self.ff(x, mu, sigma)

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
        return norm.cdf(np.log(x), mu, sigma)

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
        return 1.0 / x * norm.pdf(np.log(x), mu, sigma)

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
        return self.df(x, mu, sigma) / self.sf(x, mu, sigma)

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
        return -np.log(self.sf(x, mu, sigma))

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
        return np.exp(scipy_norm.ppf(u, mu, sigma))

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
            E = ... complicated.

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
        return -np.log(x) + norm.logpdf(np.log(x), mu, sigma)

    def log_ff(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        return norm.logcdf(np.log(x), mu, sigma)

    def log_sf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        return norm.logsf(np.log(x), mu, sigma)

    def mpp_x_transform(self, x: Numeric) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Boxable:
        return para.Normal.qf(y, 0, 1)

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Boxable:
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

    def _mom(self, x: npt.NDArray) -> tuple[Boxable, Boxable]:
        norm_mod = para.Normal.fit(np.log(x), how="MOM")
        mu, sigma = norm_mod.params
        return mu, sigma


LogNormal: LogNormal_ = LogNormal_("LogNormal")


Galton: LogNormal_ = LogNormal_("Galton")
