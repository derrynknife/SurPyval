from numpy import euler_gamma
from scipy.special import ndtri as z
from scipy.stats import uniform

from surpyval import np
from surpyval.parametric.parametric_fitter import ParametricFitter


class GumbelLEV_(ParametricFitter):
    def __init__(self, name):
        super().__init__(name)
        self.k = 2
        self.bounds = (
            (None, None),
            (0, None),
        )
        self.support = (-np.inf, np.inf)
        self.plot_x_scale = "linear"
        self.y_ticks = [
            0.0001,
            0.0002,
            0.0003,
            0.001,
            0.002,
            0.003,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
            0.9999,
        ]
        self.param_names = ["mu", "sigma"]
        self.param_map = {"mu": 0, "sigma": 1}

    def _parameter_initialiser(self, x, c=None, n=None):
        # if (2 in c) or (-1 in c):
        #     heuristic = "Turnbull"
        # else:
        heuristic = "Fleming-Harrington"
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
        return 1 - self.ff(x, mu, sigma)

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
        z = (x - mu) / sigma
        return np.exp(-np.exp(-z))

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
        return np.exp(-z) * np.exp(-np.exp(-z))

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
        return mu + sigma * (np.log(-np.log(1 - p)))

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

    def random(self, size, mu, sigma):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        mu : numpy array like or scalar
            The location parameter(s) of the distribution
        sigma : numpy array like or scalar
            The scale parameter(s) of the distribution

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> Gumbel.random(10, 3, 2)
        array([1.50706388, 3.3098799 , 4.32358009, 2.9914246 , 4.47216839,
               3.56676358, 4.19781514, 4.49123942, 7.29849677, 6.32996653])
        >>> Gumbel.random((5, 5), 3, 2)
        array([[ 5.97265715, 5.89177067, 2.95883424, 2.46315557, 5.15250379],
               [ 2.33808212, 7.42817091, 0.90560051, 8.05897841, 6.30714544],
               [ 6.13076426, 6.31925048, 4.34031705, 3.01309504,-0.70053049],
               [ 5.84888474, 5.95097491, 6.23960618, 6.24830057, 4.89655192],
               [ 6.29507963, 4.21798292, 4.22835474, 5.23521822, 2.76053242]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, mu, sigma)

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

    def var_z(self, x, mu, sigma, cv_matrix):
        z_hat = (x - mu) / sigma
        var_z = (1.0 / sigma) ** 2 * (
            cv_matrix[0, 0]
            + z_hat**2 * cv_matrix[1, 1]
            + 2 * z_hat * cv_matrix[0, 1]
        )
        return var_z

    def z_cb(self, x, mu, sigma, cv_matrix, alpha_ci=0.05):
        z_hat = (x - mu) / sigma
        var_z = self.var_z(x, mu, sigma, cv_matrix)
        bounds = z_hat + np.array([1.0, -1.0]).reshape(2, 1) * z(
            alpha_ci / 2
        ) * np.sqrt(var_z)
        return bounds

    # def R_cb(self, x, mu, sigma, cv_matrix, alpha_ci=0.05):
    # return self.sf(self.z_cb(x, mu, sigma, cv_matrix, alpha_ci=alpha_ci),
    # 0, 1).T


GumbelLEV: ParametricFitter = GumbelLEV_("GumbelLEV")
