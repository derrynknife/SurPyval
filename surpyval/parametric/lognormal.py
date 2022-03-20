from surpyval import np
from scipy.stats import uniform
from scipy.special import ndtri as z
from autograd.scipy.stats import norm
from scipy.stats import norm as scipy_norm
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class LogNormal_(ParametricFitter):
    def __init__(self, name):
        self.name = name
        self.k = 2
        self.bounds = ((0, None), (0, None),)
        self.support = (0, np.inf)
        self.plot_x_scale = 'log'
        self.y_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
        self.param_names = ['mu', 'sigma']
        self.param_map = {
            'mu'    : 0,
            'sigma' : 1
        }

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        # Need an offset mpp function here
        norm_mod = para.Normal.fit(np.log(x), c=c, n=n, t=t, how='MLE')
        mu, sigma = norm_mod.params
        return mu, sigma

    def sf(self, x, mu, sigma):
        r"""

        Surival (or Reliability) function for the LogNormal Distribution:

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
            The value(s) of the survival function at x.

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
            f(x) = \frac{1}{x \sigma \sqrt{2\pi}}e^{-\frac{1}{2}\left ( \frac{\ln x - \mu}{\sigma} \right )^{2}}

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
        return 1./x * norm.pdf(np.log(x), mu, sigma)

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
        array([ 0.11928899,  0.69316658,  2.46550819,  7.29078766, 20.08553692])
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
        return np.exp(mu + (sigma**2)/2)

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
        return np.exp(n*mu + (n**2 * sigma**2)/2)

    def random(self, size, mu, sigma):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        mu : numpy array or scalar
            The location parameter for the LogNormal distribution
        sigma : numpy array or scalar
            The scale parameter for the LogNormal distribution

        Returns
        -------

        random : scalar or numpy array 
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogNormal
        >>> LogNormal.random(10, 3, 4)
        array([1.74605298e+00, 1.90729963e+02, 1.90090366e+03, 2.59154042e-02,
               3.71460694e-02, 3.38580771e+03, 7.58826512e+04, 7.23252303e+00,
               1.21226718e+03, 4.15054624e+00])
        >>> LogNormal.random((5, 5), 3, 4)
        array([[4.59689256e+00, 2.91472936e-01, 4.66833783e+02, 9.88539048e+01,
                3.88094471e+01],
               [7.10705735e-01, 5.00788529e-02, 2.49032431e+01, 2.19196376e+01,
                2.05043988e+02],
               [1.32193999e+03, 7.38943238e-01, 5.16503535e-01, 9.09249819e+02,
                2.69407879e+03],
               [7.29473033e+00, 5.68246498e+03, 1.74464896e+00, 1.26043004e+00,
                3.84009666e+03],
               [1.47997384e+00, 2.21809242e+02, 1.32564564e+02, 8.06883052e-02,
                1.05118538e+02]])
        """
        return np.exp(para.Normal.random(size, mu, sigma))

    def mpp_x_transform(self, x, gamma=0):
        return np.log(x - gamma)

    def mpp_y_transform(self, y, *params):
        return para.Normal.qf(y, 0, 1)

    def mpp_inv_y_transform(self, y, *params):
        return para.Normal.ff(y, 0, 1)

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

    # def R_cb(self, x, mu, sigma, cv_matrix, alpha_ci=0.05):
        # t = np.log(x)
        # return para.Normal.sf(self.z_cb(t, mu, sigma, cv_matrix, alpha_ci=alpha_ci), 0, 1).T

    def _mom(self, x):
        norm_mod = para.Normal.fit(np.log(x), how='MOM')
        mu, sigma = norm_mod.params
        return mu, sigma

LogNormal = LogNormal_('LogNormal')
Galton = LogNormal_('Galton')