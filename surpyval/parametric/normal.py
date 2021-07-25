import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z
from autograd.scipy.stats import norm
from scipy.stats import norm as scipy_norm

from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Normal_(ParametricFitter):
    r"""

    Class used to generate the Normal (Gauss) class.

    .. code:: python

        from surpyval import Normal

    """
    def __init__(self, name):
        self.name = name
        self.k = 2
        self.bounds = ((None, None), (0, None),)
        self.support = (-np.inf, np.inf)
        self.plot_x_scale = 'linear'
        self.y_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 
                0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
        self.param_names = ['mu', 'sigma']
        self.param_map = {
            'mu'    : 0,
            'sigma' : 1
        }

    def _parameter_initialiser(self, x, c=None, n=None, t=None):
        return para.Normal.fit(x, c, n, t, how='MPP').params

    def log_sf(self, x, mu, sigma):
        return np.log(self.sf(x, mu, sigma))

    def sf(self, x, mu, sigma):
        r"""

        Surival (or Reliability) function for the Normal Distribution:

        .. math::
            R(x) = 1 - \Phi \left( \frac{x - \mu}{\sigma} \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        sf : scalar or numpy array 
            The value(s) of the survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Normal.sf(x, 3, 4)
        array([0.69146246, 0.59870633, 0.5       , 0.40129367, 0.30853754])
        """
        return norm.sf(x, mu, sigma)

    def cs(self, x, X, mu, sigma):
        r"""

        Conditional survival function for the Normal Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The value(s) at which the function will be calculated
        X : numpy array or scalar
            The value(s) at which each value(s) in x was known to have survived
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        cs : scalar or numpy array 
            the conditional survival probability at x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Normal.cs(x, 5, 3, 4)
        array([0.73452116, 0.51421702, 0.34242113, 0.2165286 , 0.1298356 ])
        """
        return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

    def ff(self, x, mu, sigma):
        r"""

        CDF (or unreliability or failure) function for the Normal Distribution:

        .. math::
            F(x) = \Phi \left( \frac{x - \mu}{\sigma} \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        ff : scalar or numpy array 
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Normal.ff(x, 3, 4)
        array([0.30853754, 0.40129367, 0.5       , 0.59870633, 0.69146246])
        """
        return norm.cdf(x, mu, sigma)

    def df(self, x, mu, sigma):
        r"""

        Density function for the Normal Distribution:

        .. math::
            f(x) = \frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}\left ( \frac{x - \mu}{\sigma} \right )^{2}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        df : scalar or numpy array 
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Normal.df(x, 3, 4)
        array([0.08801633, 0.09666703, 0.09973557, 0.09666703, 0.08801633])
        """
        return norm.pdf(x, mu, sigma)

    def hf(self, x, mu, sigma):
        r"""

        Instantaneous hazard rate for the Normal Distribution:

        .. math::
            h(x) = \frac{\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}\left ( \frac{x - \mu}{\sigma} \right )^{2}}}{1 - \Phi \left( \frac{x - \mu}{\sigma} \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        hf : scalar or numpy array 
            The value(s) of the instantaneous hazard rate function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Normal.hf(x, 3, 4)
        array([0.12729011, 0.16145984, 0.19947114, 0.24088849, 0.28526944])
        """
        return norm.pdf(x, mu, sigma) / self.sf(x, mu, sigma)

    def Hf(self, x, mu, sigma):
        r"""

        Cumulative hazard rate for the Normal Distribution:

        .. math::
            H(x) = -\ln \left( 1 - \Phi \left( \frac{x - \mu}{\sigma} \right ) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        ff : scalar or numpy array 
            The value(s) of the cumulative hazard rate function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Normal.Hf(x, 3, 4)
        array([0.36894642, 0.51298408, 0.69314718, 0.91306176, 1.17591176])
        """
        return -np.log(norm.sf(x, mu, sigma))

    def qf(self, p, mu, sigma):
        r"""

        Quantile function for the Normal Distribution:

        .. math::
            q(p) = \Phi^{-1} \left( p \right )

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated 
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        q : scalar or numpy array 
            The quantiles for the Normal distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Normal.qf(p, 3, 4)
        array([-2.12620626, -0.36648493,  0.90239795,  1.98661159,  3.        ])
        """
        return scipy_norm.ppf(p, mu, sigma)

    def mean(self, mu, sigma):
        r"""

        Mean of the Normal distribution

        .. math::
            E = \mu

        Parameters
        ----------

        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        mu : scalar or numpy array 
            The mean(s) of the Normal distribution

        Examples
        --------
        >>> from surpyval import Normal
        >>> Normal.mean(3, 4)
        3
        """
        return mu

    def moment(self, n, mu, sigma):
        r"""

        n-th (non central) moment of the Normal distribution

        .. math::
            E = ... complicated.

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        moment : scalar or numpy array 
            The moment(s) of the Normal distribution

        Examples
        --------
        >>> from surpyval import Normal
        >>> Normal.moment(2, 3, 4)
        25.0
        """
        return scipy_norm.moment(n, mu, sigma)

    def random(self, size, mu, sigma):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        mu : numpy array or scalar
            The location parameter for the Normal distribution
        sigma : numpy array or scalar
            The scale parameter for the Normal distribution

        Returns
        -------

        random : scalar or numpy array 
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Normal
        >>> Normal.random(10, 3, 4)
        array([-1.28484969, -1.68138703,  0.13414348,  6.53416927, -1.95649712,
                3.09951162,  6.90469836,  4.90063467,  1.11075072,  4.97841115])
        >>> Normal.random((5, 5), 3, 4)
        array([[ 1.57569952,  4.98472487,  3.19475597,  5.12581251, -0.98020861],
               [ 6.73877217,  1.08561611,  3.07634125,  3.54656313, 13.32064634],
               [-0.45094731,  2.52588422, -1.61414841,  8.39084564, -1.35261631],
               [ 1.98090151,  8.22151826,  5.59184063, -2.62221656,  0.20879673],
               [-2.0790734 ,  2.67886095,  2.54115153,  5.49853925,  4.57056015]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, mu, sigma)

    def mpp_x_transform(self, x):
        return x

    def mpp_y_transform(self, y, *params):
        return self.qf(y, 0, 1)

    def mpp_inv_y_transform(self, y, *params):
        return self.ff(y, 0, 1)

    def unpack_rr(self, params, rr):
        if rr == 'y':
            sigma, mu = params
            mu = -mu/sigma
            sigma = 1./sigma
        elif rr == 'x':
            sigma, mu = params
        return mu, sigma

    def var_z(self, x, mu, sigma, cv_matrix):
        z_hat = (x - mu)/sigma
        var_z = (1./sigma)**2 * (cv_matrix[0, 0] + z_hat**2 * cv_matrix[1, 1] + 
            2 * z_hat * cv_matrix[0, 1])
        return var_z

    def z_cb(self, x, mu, sigma, cv_matrix, alpha_ci=0.05):
        z_hat = (x - mu)/sigma
        var_z = self.var_z(x, mu, sigma, cv_matrix)
        bounds = z_hat + np.array([1., -1.]).reshape(2, 1) * z(alpha_ci/2) * np.sqrt(var_z)
        return bounds

    def R_cb(self, x, mu, sigma, cv_matrix, alpha_ci=0.05):
        return self.sf(self.z_cb(x, mu, sigma, cv_matrix, alpha_ci=alpha_ci), 0, 1).T

Normal = Normal_('Normal')
Gauss = Normal_('Gauss')
