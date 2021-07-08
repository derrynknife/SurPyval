import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter
from scipy.special import factorial
from .fitters.mpp import mpp

import warnings

class Exponential_(ParametricFitter):
    def __init__(self, name):
        self.name = name
        self.k = 1
        self.bounds = ((0, None),)
        self.support = (0, np.inf)
        self.plot_x_scale = 'linear'
        self.y_ticks = [0.05, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
        self.param_names = ['lambda']
        self.param_map = {
            'lambda' : 0,
        }

    def _parameter_initialiser(self, x, c=None, n=None, offset=False):
        x, c, n = surpyval.xcn_handler(x, c, n)
        c = (c == 0).astype(np.int64)
        rate = (n * c).sum()/x.sum()
        if offset:
            return np.min(x) - (np.max(x) - np.min(x))/10., rate
        else:
            return np.array([rate])

    def sf(self, x, failure_rate):
        r"""

        Surival (or Reliability) function for the Exponential Distribution:

        .. math::
            R(x) = e^{-\lambda x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        sf : scalar or numpy array 
            The scalar value of the survival function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the survival function at each corresponding value in the input array.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Exponential.sf(x, 3)
        array([4.97870684e-02, 2.47875218e-03, 1.23409804e-04, 6.14421235e-06,
               3.05902321e-07])
        """
        return np.exp(-failure_rate * x)

    def cs(self, x, X, failure_rate):
        # The exponential distribution is memoryless so of course it is the same as the survival function
        return self.sf(x, failure_rate)

    def ff(self, x, failure_rate):
        r"""

        CDF (or unreliability or failure) function for the Exponential Distribution:

        .. math::
            F(x) = 1 - e^{-\lambda x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        ff : scalar or numpy array 
            The value(s) of the CDF for each value of x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Exponential.ff(x, 3)
        array([0.95021293, 0.99752125, 0.99987659, 0.99999386, 0.99999969])
        """
        return 1 - np.exp(-failure_rate * x)

    def df(self, x, failure_rate):
        r"""

        Density function for the Exponential Distribution:

        .. math::
            f(x) = \lambda e^{-\lambda x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        df : scalar or numpy array 
            The density of the Exponential distribution for each value of x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Exponential.df(x, 3)
        array([1.49361205e-01, 7.43625653e-03, 3.70229412e-04, 1.84326371e-05,
               9.17706962e-07])
        """
        return failure_rate * np.exp(-failure_rate * x)

    def hf(self, x, failure_rate):
        r"""

        Instantaneous hazard rate for the Exponential Distribution.

        .. math::
            f(x) = \lambda

        The failure rate for the exponential distribution is constant. So this function only returns the input failure rate in the same shape as x.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        hf : scalar or numpy array 
            The instantaneous hazard rate of the Exponential distribution for each value of x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Exponential.hf(x, 3)
        array([3, 3, 3, 3, 3])
        """
        return np.ones_like(x) * failure_rate

    def Hf(self, x, failure_rate):
        r"""

        Cumulative hazard rate for the Exponential Distribution.

        .. math::
            f(x) = \lambda x

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        hf : scalar or numpy array 
            The cumulative hazard rate of the Exponential distribution for each value of x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Exponential.Hf(x, 3)
        array([ 3,  6,  9, 12, 15])
        """
        return failure_rate * x

    def qf(self, p, failure_rate):
        r"""

        Quantile function for the Exponential Distribution:

        .. math::
            q(p) = \frac{-\ln\left ( p \right )}{\lambda}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated 
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        q : scalar or numpy array 
            The quantiles for the Exponential distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Exponential.qf(p, 3)
        array([0.76752836, 0.5364793 , 0.40132427, 0.30543024, 0.23104906])
        """
        return -np.log(p)/failure_rate

    def mean(self, failure_rate):
        r"""

        Calculates the mean of the Exponential distribution with given parameters.

        .. math::
            E = \frac{1}{\lambda}

        Parameters
        ----------

        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        mean : scalar or numpy array 
            The mean(s) of the Exponential distribution 

        Examples
        --------
        >>> from surpyval import Exponential
        >>> Exponential.mean(3)
        0.3333333333333333
        """
        return 1. / failure_rate

    def moment(self, n, failure_rate):
        r"""

        Calculates the n-th moment of the Exponential distribution.

        .. math::
            E = \frac{n!}{\lambda^{n}}

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        moment : scalar or numpy array 
            The moment(s) of the Exponential distribution

        Examples
        --------
        >>> from surpyval import Exponential
        >>> Exponential.moment(2, 3)
        0.2222222222222222
        """
        return factorial(n) / (failure_rate ** n)

    def entropy(self, failure_rate):
        r"""

        Calculates the entropy of the Exponential distribution.

        .. math::
            S = 1 - \ln \left ( \lambda \right )

        Parameters
        ----------

        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        entropy : scalar or numpy array 
            The entropy(ies) of the Exponential distribution

        Examples
        --------
        >>> from surpyval import Exponential
        >>> Exponential.entropy(3)
        -0.09861228866810978
        """
        return 1 - np.log(failure_rate)

    def random(self, size, failure_rate):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        failure_rate : numpy array or scalar
            The scale parameter for the Exponential distribution

        Returns
        -------

        random : scalar or numpy array 
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Exponential
        >>> Exponential.random(10, 3)
        array([0.32480264, 0.03186663, 0.41807108, 0.74221745, 0.06133774,
               0.2128422 , 0.36299424, 0.12250138, 0.61431089, 0.02266754])
        >>> Exponential.random((5, 5), 3)
        array([[0.25425552, 0.16867629, 0.21692401, 0.07020826, 0.03676643],
               [0.65528908, 0.20774767, 0.00625475, 0.04122388, 0.07089254],
               [1.22844679, 0.36199751, 0.564159  , 1.86811492, 0.08132478],
               [0.33541878, 0.38614518, 0.09285907, 0.33422975, 0.32515494],
               [0.03529228, 0.63134988, 0.45528738, 0.05037512, 0.7338039 ]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, failure_rate)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        mask = ((y == 0) | (y == 1))
        out = np.zeros_like(y)
        out[~mask] = -np.log(1 - y[~mask])
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1 - np.exp(y)

    def mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False, offset=False):
        assert rr in ['x', 'y']
        x_pp, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

        if not on_d_is_0:
            x_pp = x_pp[d > 0]
            F    = F[d > 0]
        
        # Linearise
        y_pp = self.mpp_y_transform(F)

        mask = np.isfinite(y_pp)
        if mask.any():
            warnings.warn("Some Infinite values encountered in plotting points and have been ignored.", stacklevel=2)
            y_pp = y_pp[mask]
            x_pp = x_pp[mask]

        if offset:
            if   rr == 'y':
                params = np.polyfit(x_pp, y_pp, 1)
            elif rr == 'x':
                params = np.polyfit(y_pp, x_pp, 1)
            failure_rate = params[0]
            offset = -params[1] * (1./failure_rate)
            return tuple([offset, failure_rate])
        else:
            if   rr == 'y':
                x_pp = x_pp[:,np.newaxis]
                failure_rate = np.linalg.lstsq(x_pp, y_pp, rcond=None)[0]
            elif rr == 'x':
                y_pp = y_pp[:,np.newaxis]
                mttf = np.linalg.lstsq(y_pp, x_pp, rcond=None)[0]
                failure_rate = 1. / mttf
            return tuple([failure_rate[0]])

    def lambda_cb(self, x, failure_rate, cv_matrix, cb=0.05):
        return failure_rate * np.exp(np.array([-1, 1]).reshape(2, 1) * (z(cb/2) * 
                                    np.sqrt(cv_matrix.item()) / failure_rate))

    def R_cb(self, x, failure_rate, cv_matrix, cb=0.05):
        return np.exp(-self.lambda_cb(x, failure_rate, cv_matrix, cb=0.05) * x).T

Exponential = Exponential_('Exponential')