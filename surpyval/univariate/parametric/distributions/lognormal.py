from autograd.scipy.stats import norm
from scipy.stats import norm as scipy_norm

from surpyval import np
from surpyval.univariate import parametric as para
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class LogNormal_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
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

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        if offset:
            # Shift the data so the log transform is defined, then
            # initialise mu and sigma from the shifted data
            gamma_init = np.min(x) - 1.0
            norm_mod = para.Normal.fit(
                np.log(x - gamma_init), c=c, n=n, how="MLE"
            )
            mu, sigma = norm_mod.params
            return gamma_init, mu, sigma
        norm_mod = para.Normal.fit(np.log(x), c=c, n=n, how="MLE")
        mu, sigma = norm_mod.params
        return mu, sigma

    def sf(self, x, mu, sigma):
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

    def cs(self, x, X, mu, sigma):
        r"""

        Conditional survival function for the LogNormal Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The value(s) at which the function will be calculated
        X : numpy array or scalar
            The value(s) at which each value(s) in x was known to have survived
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        cs : scalar or numpy array
            the conditional survival probability at x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogNormal.cs(x, 5, 3, 4)
        array([0.97287811, 0.9496515 , 0.92933892, 0.91129122, 0.89505592])
        """
        return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

    def ff(self, x, mu, sigma):
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

    def df(self, x, mu, sigma):
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

    def hf(self, x, mu, sigma):
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

    def Hf(self, x, mu, sigma):
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

    def qf(self, p, mu, sigma):
        r"""

        Quantile function for the LogNormal Distribution:

        .. math::
            q(p) = e^{\mu + \sigma \Phi^{-1} \left( p \right )}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the LogNormal distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> LogNormal.qf(p, 3, 4)
        array([ 0.11928899,  0.69316658,  2.46550819,  7.29078766,
                20.08553692])
        """
        return np.exp(scipy_norm.ppf(p, mu, sigma))

    def mean(self, mu, sigma):
        r"""

        Quantile function for the LogNormal Distribution:

        .. math::
            E = e^{\mu + \frac{\sigma^2}{2}}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the LogNormal distribution at each value p.

        Examples
        --------
        >>> from surpyval import LogNormal
        >>> LogNormal.mean(3, 4)
        59874.14171519782
        """
        return np.exp(mu + (sigma**2) / 2)

    def moment(self, n, mu, sigma):
        r"""

        n-th (non central) moment of the LogNormal distribution

        .. math::
            E = ... complicated.

        Parameters
        ----------

        n : integer or numpy array of integers
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
        3.1855931757113756e+16
        """
        return np.exp(n * mu + (n**2 * sigma**2) / 2)

    def entropy(self, mu, sigma):
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
        5.805232894324563
        """
        return mu + 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    def log_df(self, x, mu, sigma):
        return -np.log(x) + norm.logpdf(np.log(x), mu, sigma)

    def log_ff(self, x, mu, sigma):
        return norm.logcdf(np.log(x), mu, sigma)

    def log_sf(self, x, mu, sigma):
        return norm.logsf(np.log(x), mu, sigma)

    def mpp_x_transform(self, x, gamma=0):
        return np.log(x - gamma)

    def mpp_y_transform(self, y, *params):
        return para.Normal.qf(y, 0, 1)

    def mpp_inv_y_transform(self, y, *params):
        return para.Normal.ff(y, 0, 1)

    def unpack_rr(self, params, rr):
        if rr == "y":
            sigma, mu = params
            mu = -mu / sigma
            sigma = 1.0 / sigma
        elif rr == "x":
            sigma, mu = params
        return mu, sigma

    def _mom(self, x):
        norm_mod = para.Normal.fit(np.log(x), how="MOM")
        mu, sigma = norm_mod.params
        return mu, sigma


LogNormal: ParametricFitter = LogNormal_("LogNormal")


Galton: ParametricFitter = LogNormal_("Galton")
