import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

from surpyval import xcn_handler
from surpyval.parametric.parametric_fitter import ParametricFitter

class Logistic_(ParametricFitter):
    def __init__(self, name):
        self.name = name
        self.k = 2
        self.bounds = ((None, None), (0, None),)
        self.support = (-np.inf, np.inf)
        self.plot_x_scale = 'linear'
        self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
            0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
            0.9, 0.95, 0.99, 0.999, 0.9999]
        self.param_names = ['mu', 'sigma']
        self.param_map = {
            'mu'    : 0,
            'sigma' : 1
        }

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        return self.fit(x, c, n, t, how='MPP').params
        x, c, n = xcn_handler(x, c, n)
        flag = (c == 0).astype(int)
        if offset:
            return x.sum() / (n * flag).sum(), 1., 1.
        else:
            return x.sum() / (n * flag).sum(), 1.

    def sf(self, x, mu, sigma):
        r"""

        Survival (or reliability) function for the Logistic Distribution:

        .. math::
            R(x) = 1 - \frac{1}{1 + e^{- \left ( x - \mu \right )/ \sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        sf : scalar or numpy array 
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.sf(x, 3, 4)
        array([0.62245933, 0.5621765 , 0.5       , 0.4378235 , 0.37754067])
        """
        return 1 - self.ff(x, mu, sigma)

    def cs(self, x, X, mu, sigma):
        return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

    def ff(self, x, mu, sigma):
        r"""

        Failure (CDF or unreliability) function for the Logistic Distribution:

        .. math::
            F(x) = \frac{1}{1 + e^{- \left ( x - \mu \right )/ \sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        ff : scalar or numpy array 
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.ff(x, 3, 4)
        array([0.37754067, 0.4378235 , 0.5       , 0.5621765 , 0.62245933])
        """
        z = (x - mu) / sigma
        return 1. / (1 + np.exp(-z))

    def df(self, x, mu, sigma):
        r"""

        Failure (CDF or unreliability) function for the Logistic Distribution:

        .. math::
            f(x) = \frac{e^{-\left ( x - \mu \right ) / \sigma}}{\sigma \left ( 1 + e^{-\left ( x - \mu \right )/ \sigma}\right )^2}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        df : scalar or numpy array 
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.df(x, 3, 4)
        array([0.05875093, 0.06153352, 0.0625    , 0.06153352, 0.05875093])
        """
        z = (x - mu) / sigma
        return np.exp(-z) / (sigma * (1 + np.exp(-z))**2)

    def hf(self, x, mu, sigma):
        r"""

        Instantaneous hazard rate for the Logistic Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        hf : scalar or numpy array 
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.hf(x, 3, 4)
        array([0.09438517, 0.10945587, 0.125     , 0.14054413, 0.15561483])
        """
        return self.df(x, mu, sigma) / self.sf(x, mu, sigma)

    def Hf(self, x, mu, sigma):
        r"""

        Cumulative hazard rate for the Logistic distribution:

        .. math::
            h(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        hf : scalar or numpy array 
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Logistic.Hf(x, 3, 4)
        array([0.47407698, 0.57593942, 0.69314718, 0.82593942, 0.97407698])
        """
        return -np.log(self.sf(x, mu, sigma))

    def qf(self, p, mu, sigma):
        r"""

        Quantile function for the Logistic distribution:

        .. math::
            q(p) = \mu + \sigma \ln \left ( \frac{p}{1 - p} \right)

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        q : scalar or numpy array 
            The quantiles for the Logistic distribution at each value p

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Logistic.qf(p, 3, 4)
        array([-5.78889831, -2.54517744, -0.38919144,  1.37813957,  3.        ])
        """
        return mu + sigma * np.log(p/(1 - p))

    def mean(self, mu, sigma):
        r"""

        Mean of the Logistic distribution

        .. math::
            E = \mu

        Parameters
        ----------

        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        mu : scalar or numpy array 
            The mean(s) of the Logistic distribution

        Examples
        --------
        >>> from surpyval import Logistic
        >>> Logistic.mean(3, 4)
        3
        """
        return mu

    def random(self, size, mu, sigma):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        mu : numpy array or scalar
            The location parameter for the Logistic distribution
        sigma : numpy array or scalar
            The scale parameter for the Logistic distribution

        Returns
        -------

        random : scalar or numpy array 
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Logistic
        >>> Logistic.random(10, 3, 4)
        array([-8.03085073, -1.69001847,  5.25971637,  4.49119392,  3.92027233,
               -0.8320818 , -7.08778338,  5.01180405,  0.82373259,  8.51506487])
        >>> Logistic.random((5, 5), 3, 4)
        array([[ 7.11691946, 14.31662627,  8.5383889 ,  1.26608344,  0.97633704],
               [-7.11229405,  8.56748118,  1.5959416 , -3.89229554, -2.44623464],
               [ 5.58805039, -0.11768336, -0.55000158,  8.5302643 ,  6.92591024],
               [-2.88281091, -9.79724128, -3.80713019,  1.74120972, 15.37924263],
               [-4.42521443, -0.69577732,  3.54658395,  2.82310964,  3.95850831]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, mu, sigma)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        mask = ((y == 0) | (y == 1))
        out = np.zeros_like(y)
        out[~mask] = -np.log(1./y[~mask] - 1)
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1./(np.exp(-y) + 1)

    def unpack_rr(self, params, rr):
        if   rr == 'y':
            sigma = 1./params[0]
            mu    = -sigma * params[1]
        elif rr == 'x':
            sigma  = params[0]
            mu = params[1]
        return mu, sigma

Logistic = Logistic_('Logistic')