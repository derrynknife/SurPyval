import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

from .fitters.mpp import mpp

class Weibull_(ParametricFitter):
    def __init__(self, name):
        self.name = name
        # Set 'k', the number of parameters
        self.k = 2
        self.bounds = ((0, None), (0, None),)
        self.support = (0, np.inf)
        self.plot_x_scale = 'log'
        self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
            0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
            0.9, 0.95, 0.99, 0.999, 0.9999]
        self.param_names = ['alpha', 'beta']
        self.param_map = {
            'alpha' : 0,
            'beta' : 1
        }

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        log_x = np.log(x)
        log_x[np.isnan(log_x)] = -np.inf
        if (2 in c) or (-1 in c):
            heuristic = "Turnbull"
        else:
            heuristic = "Nelson-Aalen"
        if offset:
            results = mpp(dist=self, x=x, c=c, n=n, t=t, 
                          heuristic=heuristic, on_d_is_0=True, offset=True)
            return (results['gamma'], *results['params'])
        else:
            gumb = para.Gumbel.fit(log_x, c, n, t, how='MLE')
            if not gumb.res.success:
                gumb = para.Gumbel.fit(log_x, c, n, t, how='MPP', heuristic=heuristic)
            mu, sigma = gumb.params
            alpha, beta = np.exp(mu), 1. / sigma
            if (np.isinf(alpha) | np.isnan(alpha)):
                alpha = np.median(x)
            if (np.isinf(beta) | np.isnan(beta)):
                beta = 1.
            return alpha, beta

    def sf(self, x, alpha, beta):
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
        return np.exp(-(x / alpha)**beta)

    def ff(self, x, alpha, beta):
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

        sf : scalar or numpy array 
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.ff(x, 3, 4)
        array([0.01226978, 0.17924519, 0.63212056, 0.9575952 , 0.99955438])
        """
        return 1 - np.exp(-(x / alpha)**beta)

    def cs(self, x, X, alpha, beta):
        r"""

        Conditional survival function for the Weibull Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

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

        cs : scalar or numpy array 
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.cs(x, 5, 3, 4)
        array([2.21654222e+03, 1.84183662e+03, 8.25549630e+02, 9.51596070e+01,
               1.00000000e+00])
        """
        return self.sf(x, alpha, beta) / self.sf(X, alpha, beta)

    def df(self, x, alpha, beta):
        r"""

        Density function for the Weibull Distribution:

        .. math::
            f(x) = \frac{\beta}{\alpha} \frac{x}{\alpha}^{\beta - 1} e^{-\left ( \frac{x}{\alpha} \right )^\beta}

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
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.df(x, 5, 3, 4)
        array([0.0487768 , 0.32424881, 0.49050592, 0.13402009, 0.00275073])
        """
        return (beta / alpha) * (x / alpha)**(beta-1) * np.exp(-(x / alpha)**beta)

    def hf(self, x, alpha, beta):
        r"""

        Instantaneous hazard rate for the Weibull Distribution:

        .. math::
            h(x) = \frac{\beta}{\alpha} \left ( \frac{x}{\alpha} \right )^{\beta - 1}

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
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.hf(x, 3, 4)
        array([0.04938272, 0.39506173, 1.33333333, 3.16049383, 6.17283951])
        """
        return (beta / alpha) * (x / alpha)**(beta - 1)

    def Hf(self, x, alpha, beta):
        r"""

        Cumulative hazard rate for the Weibull Distribution:

        .. math::
            h(x) = \frac{x}{\alpha}^{\beta}

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
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.Hf(x, 3, 4)
        array([0.01234568, 0.19753086, 1.        , 3.16049383, 7.71604938])
        """
        return (x / alpha)**beta

    def qf(self, p, alpha, beta):
        r"""

        Quantile function for the Weibull distribution:

        .. math::
            q(p) = \alpha \left ( -\ln \left ( 1 - p \right ) \right )^{1/ \beta}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        q : scalar or numpy array 
            The quantiles for the Weibull distribution at each value p

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Weibull.qf(p, 3, 4)
        array([1.70919151, 2.06189877, 2.31840554, 2.5362346 , 2.73733292])
        """
        return alpha * (-np.log(1 - p))**(1/beta)

    def mean(self, alpha, beta):
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
        2.7192074311664314
        """
        return alpha * gamma_func(1 + 1./beta)

    def moment(self, n, alpha, beta):
        r"""

        n-th moment of the Weibull distribution

        .. math::
            M(n) = \alpha^n \Gamma \left ( 1 + \frac{n}{\beta} \right )

        Parameters
        ----------

        n : integer or numpy array of integers
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
        7.976042329074821
        """
        return alpha**n * gamma_func(1 + n/beta)

    def entropy(self, alpha, beta):
        return euler_gamma * (1 - 1/beta) + np.log(alpha / beta) + 1

    def random(self, size, alpha, beta):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        random : scalar or numpy array 
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> Weibull.random(10, 3, 4)
        array([1.79782451, 1.7143211 , 2.84778674, 3.12226231, 2.61000839,
               3.05456332, 3.00280851, 2.61910071, 1.37991527, 4.17488394])
        >>> Weibull.random((5, 5), 3, 4)
        array([[1.64782514, 2.79157632, 1.85500681, 2.91908736, 2.46089933],
               [1.85880127, 0.96787742, 2.29677031, 2.42394129, 2.63889601],
               [2.14351859, 3.90677225, 2.24013855, 2.49467774, 3.43755278],
               [3.24417396, 1.40775181, 2.49584969, 3.07603353, 2.54679499],
               [1.98330076, 2.95002633, 3.35402601, 3.11429283, 3.45706789]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, alpha, beta)

    def mpp_x_transform(self, x):
        return np.log(x)

    def mpp_inv_x_transform(self, x, gamma=0):
        return np.exp(x - gamma)

    def mpp_y_transform(self, y, *params):
        mask = ((y == 0) | (y == 1))
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log((1 - y[~mask])))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1 - np.exp(-np.exp(y))

    def unpack_rr(self, params, rr):
        if rr == 'y':
            beta  = params[0]
            alpha = np.exp(params[1]/-beta)
        elif rr == 'x':
            beta  = 1./params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta

    def u(self, x, alpha, beta):
        return beta * (np.log(x) - np.log(alpha))

    def u_cb(self, x, alpha, beta, cv_matrix, cb=0.05):
        u = self.u(x, alpha, beta)
        var_u = self.var_u(x, alpha, beta, cv_matrix)
        diff = z(cb/2) * np.sqrt(var_u)
        bounds = u + np.array([1., -1.]).reshape(2, 1) * diff
        return bounds

    def du(self, x, alpha, beta):
        du_dbeta = np.log(x) - np.log(alpha)
        du_dalpha  = -beta/alpha
        return du_dalpha, du_dbeta

    def var_u(self, x, alpha, beta, cv_matrix):
        da, db = self.du(x, alpha, beta)
        var_u = (da**2 * cv_matrix[0, 0] + db**2 * cv_matrix[1, 1] + 
            2 * da * db * cv_matrix[0, 1])
        return var_u

    def R_cb(self, x, alpha, beta, cv_matrix, alpha_ci=0.05):
        return np.exp(-np.exp(self.u_cb(x, alpha, beta, cv_matrix, alpha_ci))).T

    def _jacobian(self, x, alpha, beta, c=None, n=None):
        f = c == 0
        l = c == -1
        r = c == 1
        dll_dbeta = (
            1./beta * np.sum(n[f]) +
            np.sum(n[f] * np.log(x[f]/alpha)) - 
            np.sum(n[f] * (x[f]/alpha)**beta * np.log(x[f]/alpha)) - 
            np.sum(n[r] * (x[r]/alpha)**beta * np.log(x[r]/alpha)) +
            np.sum(n[l] * (x[l]/alpha)**beta * np.log(x[l]/alpha) *
                np.exp(-(x[l]/alpha)**beta) / 
                (1 - np.exp(-(x[l]/alpha)**beta)))
        )

        dll_dalpha = ( 0 -
            beta/alpha * np.sum(n[f]) +
            beta/alpha * np.sum(n[f] * (x[f]/alpha)**beta) +
            beta/alpha * np.sum(n[r] * (x[r]/alpha)**beta) -
            beta/alpha * np.sum(n[l] * (x[l]/alpha)**beta * 
                np.exp(-(x[l]/alpha)**beta) /
                (1 - np.exp(-(x[l]/alpha)**beta)))
        )
        return -np.array([dll_dalpha, dll_dbeta])

Weibull = Weibull_('Weibull')