import autograd.numpy as np
from autograd import jacobian
from scipy.stats import uniform
from scipy.optimize import approx_fprime
from scipy.special import gamma as gamma_func
from autograd.scipy.special import gamma as agamma
from scipy.special import gammainc, gammaincinv
from autograd_gamma import gammainc as agammainc
from scipy.special import ndtri as z

from scipy.optimize import minimize
from scipy.stats import pearsonr

import warnings

import surpyval
from surpyval import parametric as para
from surpyval import nonparametric as nonp
from surpyval.parametric.parametric_fitter import ParametricFitter

class Gamma_(ParametricFitter):
    r"""

    Class used to generate the Gamma class.

    .. code:: python

        from surpyval import Gamma

    """
    def __init__(self, name):
        self.name = name
        self.k = 2
        self.bounds = ((0, None), (0, None),)
        self.support = (0, np.inf)
        self.plot_x_scale = 'linear'
        self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
            0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
            0.9, 0.95, 0.99, 0.999, 0.9999]
        self.param_names = ['alpha', 'beta']
        self.param_map = {
            'alpha' : 0,
            'beta'  : 1
        }

    def _parameter_initialiser(self, x, c=None, n=None, offset=False):
        # These equations are truly magical
        if offset:
            s = np.log(x.sum()/len(x)) - np.log(x).sum()/len(x)
            alpha = (3 - s + np.sqrt((s - 3)**2 + 24*s)) / (12*s)
            beta = x.sum()/(len(x)*alpha)
            return alpha, 1./beta, np.min(x) - 1
        else:
            s = np.log(x.sum()/len(x)) - np.log(x).sum()/len(x)
            alpha = (3 - s + np.sqrt((s - 3)**2 + 24*s)) / (12*s)
            beta = x.sum()/(len(x)*alpha)
            return alpha, 1./beta

    def sf(self, x, alpha, beta):
        r"""

        Surival (or Reliability) function for the Gamma Distribution:

        .. math::
            R(x) = 1 - \frac{\gamma \left ( \alpha, \beta x \right )}{\Gamma \left ( \alpha \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution


        Returns
        -------

        sf : scalar or numpy array 
            The value(s) for the survival function at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.sf(x, 3, 2)
        array([0.67667642, 0.23810331, 0.0619688 , 0.01375397, 0.0027694 ])
        """
        return 1 - self.ff(x, alpha, beta)

    def cs(self, x, X, alpha, beta):
        return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

    def ff(self, x, alpha, beta):
        r"""

        CDF (or unreliability or failure) function for the Gamma Distribution:

        .. math::
            F(x) = \frac{\gamma \left ( \alpha, \beta x \right )}{\Gamma \left ( \alpha \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution


        Returns
        -------

        ff : scalar or numpy array 
            The value(s) for the CDF at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.ff(x, 3, 2)
        array([0.32332358, 0.76189669, 0.9380312 , 0.98624603, 0.9972306 ])
        """
        return agammainc(alpha, beta * x)

    def df(self, x, alpha, beta):
        r"""

        Density function for the Gamma Distribution:

        .. math::
            f(x) = \frac{\beta^{\alpha }}{\Gamma \left ( \alpha \right )}x^{\alpha - 1}e^{-\beta x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution


        Returns
        -------

        df : scalar or numpy array 
            The density of the distribution at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.df(x, 3, 2)
        array([0.54134113, 0.29305022, 0.08923508, 0.02146961, 0.00453999])
        """
        return ((beta ** alpha) * x ** (alpha - 1) * np.exp(-(x * beta)) / (agamma(alpha)))

    def hf(self, x, alpha, beta):
        r"""

        Instantaneous hazard rate for the Gamma Distribution:

        .. math::
            h(x) = \frac{\frac{\beta^{\alpha }}{\Gamma \left ( \alpha \right )}x^{\alpha - 1}e^{-\beta x}}{1 - \frac{\gamma \left ( \alpha, \beta x \right )}{\Gamma \left ( \alpha \right )}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution


        Returns
        -------

        Hf : scalar or numpy array 
            The instantaneous hazard rate of the distribution at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.hf(x, 3, 2)
        array([0.8       , 1.23076923, 1.44      , 1.56097561, 1.63934426])
        """
        return self.df(x, alpha, beta) / self.sf(x, alpha, beta)

    def Hf(self, x, alpha, beta):
        r"""

        Cumulative hazard rate for the Gamma Distribution:

        .. math::
            H(x) = -\ln(1 - \frac{\gamma \left ( \alpha, \beta x \right )}{\Gamma \left ( \alpha \right )})

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution


        Returns
        -------

        Hf : scalar or numpy array 
            The cumulative hazard rate of the distribution at each x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.Hf(x, 3, 2)
        array([0.39056209, 1.43505064, 2.78112418, 4.28642793, 5.88912614])
        """
        return -np.log(self.sf(x, alpha, beta))

    def qf(self, p, alpha, beta):
        return gammaincinv(alpha, p) / beta

    def mean(self, alpha, beta):
        return alpha / beta

    def moment(self, n, alpha, beta):
        return agamma(n + alpha) / (beta**n * agamma(alpha))

    def random(self, size, alpha, beta):
        U = uniform.rvs(size=size)
        return self.qf(U, alpha, beta)

    def mpp_y_transform(self, y, *params):
        alpha = params[0]
        return gammaincinv(alpha, y)

    def mpp_inv_y_transform(self, y, *params):
        alpha = params[0]
        return agammainc(alpha, y)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False, offset=False):
        x_pp, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

        if on_d_is_0:
            pass
        else:
            F = F[d > 0]
            x_pp = x_pp[d > 0]

        if (F == 1).any():
            mask = F != 1
            warnings.warn("Some heuristic values for CDF = 1 have been encountered in plotting points and have been ignored.", stacklevel=2)
            F = F[mask]
            x_pp = x_pp[mask]

        init = self.parameter_initialiser(x_pp, c, n)

        mask = np.isfinite(F)
        if mask.any():
            warnings.warn("Some Infinite values encountered in plotting points and have been ignored.", stacklevel=2)
            F = F[mask]
            x_pp = x_pp[mask]

        if offset:
            fun = lambda a : -pearsonr(x_pp, self.mpp_y_transform(F, a, 1.))[0]
            res = minimize(fun, init[0], bounds=((0, None),))
            alpha = res.x[0]

            y_pp = self.mpp_y_transform(F, alpha)

            if   rr == 'y':
                params = np.polyfit(x_pp, y_pp, 1)
                beta = params[0]
                gamma = -params[1]/beta
            elif rr == 'x':
                params = np.polyfit(y_pp, x_pp, 1)
                beta = 1./params[0]
                gamma = -params[1]/beta

            return gamma, alpha, beta
        else:
            if rr == 'y':
                x_pp = x_pp[:,np.newaxis]
                fun = lambda alpha : np.linalg.lstsq(x_pp, self.mpp_y_transform(F, alpha, 1.))[1]
                res = minimize(fun, init[0], bounds=((0, None),))
                alpha = res.x[0]
                beta, residuals, _, _ = np.linalg.lstsq(x_pp, self.mpp_y_transform(F, alpha, 1.))
                beta = beta[0]
            else:
                fun = lambda a : np.linalg.lstsq(self.mpp_y_transform(F, a, 1.)[:, np.newaxis], x)[1]
                res = minimize(fun, init[0], bounds=((0, None),))
                alpha = res.x[0]
                beta = np.linalg.lstsq(self.mpp_y_transform(F, alpha, 1.)[:, np.newaxis], x)[0]
                beta = 1./beta
            return alpha, beta

    def var_R(self, dR, cv_matrix):
        dr_dalpha = dR[:, 0]
        dr_dbeta  = dR[:, 1]
        var_r = (dr_dalpha**2 * cv_matrix[0, 0] + 
                 dr_dbeta**2  * cv_matrix[1, 1] + 
                 2 * dr_dalpha * dr_dbeta * cv_matrix[0, 1])
        return var_r

    def R_cb(self, x, alpha, beta, cv_matrix, cb=0.05):
        R_hat = self.sf(x, alpha, beta)
        dR_f = lambda t : self.sf(*t)
        jac = jacobian(dR_f)
        #jac = lambda t : approx_fprime(t, dR_f, surpyval.EPS)[1::]
        x_ = np.array(x)
        if x_.size == 1:
            dR = jac(np.array((x_, alpha, beta))[1::])
            dR = dR.reshape(1, 2)
        else:
            out = []
            for xx in x_:
                out.append(jac(np.array((xx, alpha, beta)))[1::])
            dR = np.array(out)
        K = z(cb/2)
        exponent = K * np.array([-1, 1]).reshape(2, 1) * np.sqrt(self.var_R(dR, cv_matrix))
        exponent = exponent/(R_hat*(1 - R_hat))
        R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
        return R_cb.T

Gamma = Gamma_('Gamma')