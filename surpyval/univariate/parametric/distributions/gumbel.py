from numpy import euler_gamma
from scipy.stats import gumbel_l

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class Gumbel_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((None, None), (0, None)),
            support=(-np.inf, np.inf),
            param_names=["mu", "sigma"],
            param_map={"mu": 0, "sigma": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(self, x, c=None, n=None, offset=True):
        if (2 in c) or (-1 in c):
            heuristic = "Turnbull"
        else:
            heuristic = "Nelson-Aalen"
        return self.fit(x, c, n, how="MPP", heuristic=heuristic).params

    def sf(self, x, mu, sigma):
        r"""

        Surival (or Reliability) function for the Gumbel Distribution:

        .. math::
            R(x) = 1 - e^{e^{-\left ( x - \mu \right ) / \sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        sf : scalar or numpy array
            The scalar value of the survival function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the survival function at each
            corresponding value in the input array.


        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.sf(x, 3, 2)
        array([0.69220063, 0.54523921, 0.36787944, 0.19229565, 0.06598804])
        """
        return np.exp(-np.exp((x - mu) / sigma))

    def cs(self, x, X, mu, sigma):
        return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

    def ff(self, x, mu, sigma):
        r"""

        CDF (or Failure) function for the Gumbel Distribution:

        .. math::
            F(x) = e^{e^{-\left ( x - \mu \right )/\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        ff : scalar or numpy array
            The scalar value of the failure function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the failure function at each
            corresponding value in the input array.


        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.ff(x, 3, 2)
        array([0.30779937, 0.45476079, 0.63212056, 0.80770435, 0.93401196])
        """
        return -np.expm1(-self.Hf(x, mu, sigma))

    def df(self, x, mu, sigma):
        r"""

        Density function (pdf) for the Gumbel Distribution:

        .. math::
            f(x) = \frac{1}{\sigma}e^{\left (\frac{x - \mu}{\sigma} -
            e^{\frac{x-\mu}{\sigma}} \right)}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        df : scalar or numpy array
            The scalar value of the density of the distribution if a scalar was
            passed. If an array like object was passed then a numpy array is
            returned with the value of the density at each corresponding value
            in the input array.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.df(x, 3, 2)
        array([0.12732319, 0.16535215, 0.18393972, 0.15852096, 0.08968704])
        """
        z = (x - mu) / sigma
        return (1 / sigma) * np.exp(z - np.exp(z))

    def hf(self, x, mu, sigma):
        r"""

        Instantaneous hazard rate for the Gumbel Distribution:

        .. math::
            h(x) = \frac{1}{\sigma} e^{\frac{x-\mu}{\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) for the instantaneous hazard rate for the Gumbel
            distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.hf(x, 3, 2)
        array([0.18393972, 0.30326533, 0.5       , 0.82436064, 1.35914091])
        """
        z = (x - mu) / sigma
        return (1 / sigma) * np.exp(z)

    def Hf(self, x, mu, sigma):
        r"""

        Cumulative hazard rate for the Gumbel Distribution:

        .. math::
            H(x) = e^{\frac{x-\mu}{\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) for the cumulative hazard rate for the Gumbel
            distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.Hf(x, 3, 2)
        array([0.36787944, 0.60653066, 1.        , 1.64872127, 2.71828183])
        """
        return np.exp((x - mu) / sigma)

    def qf(self, p, mu, sigma):
        r"""

        Quantile function for the Gumbel Distribution:

        .. math::
            q(p) = \mu + \sigma\ln\left ( -\ln\left ( 1 - p \right ) \right )

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array like or scalar
            The location parameter(s) of the distribution
        sigma : numpy array like or scalar
            The scale parameter(s) of the distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Gumbel distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Gumbel.qf(p, 3, 2)
        array([-1.50073465e+00, 1.20026481e-04, 9.38139134e-01, 1.65654602e+00,
        2.26697416e+00])
        """
        return mu + sigma * (np.log(-np.log1p(-p)))

    def mean(self, mu, sigma):
        r"""

        Calculates the mean of the Gumbel distribution with given parameters.

        .. math::
            E = \mu + \sigma\gamma

        Where gamma is the Euler-Mascheroni constant

        Parameters
        ----------

        mu : numpy array like or scalar
            The location parameter(s) of the distribution
        sigma : numpy array like or scalar
            The scale parameter(s) of the distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Gumbel distribution

        Examples
        --------
        >>> from surpyval import Gumbel
        >>> Gumbel.mean(3, 2)
        4.1544313298030655
        """
        return mu + sigma * euler_gamma

    def log_df(self, x, mu, sigma):
        z = (x - mu) / sigma
        return z - np.exp(z) - np.log(sigma)

    def log_sf(self, x, mu, sigma):
        return -self.Hf(x, mu, sigma)

    def log_ff(self, x, mu, sigma):
        return np.log(-np.expm1(-self.Hf(x, mu, sigma)))

    def moment(self, n, mu, sigma):
        return gumbel_l.moment(n, loc=mu, scale=sigma)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log(1 - y[~mask]))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1 - np.exp(-np.exp(y))

    def unpack_rr(self, params, rr):
        if rr == "y":
            sigma = 1.0 / params[0]
            mu = -sigma * params[1]
        elif rr == "x":
            sigma = params[0]
            mu = params[1]
        return mu, sigma


Gumbel: ParametricFitter = Gumbel_("Gumbel")
GumbelSEV: ParametricFitter = Gumbel_("GumbelSEV")
