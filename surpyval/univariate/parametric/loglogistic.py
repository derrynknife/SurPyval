from scipy.stats import fisk, uniform

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter
from surpyval.utils import xcnt_handler


class LogLogistic_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            param_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="log",
            y_ticks=[
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
            ],
        )
        self.k = 2
        self.bounds = (
            (0, None),
            (0, None),
        )
        self.support = (0, np.inf)
        self.plot_x_scale = "log"
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

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        if offset:
            x, c, n, _ = xcnt_handler(x, c, n, t)
            flag = (c == 0).astype(int)
            value_range = np.max(x) - np.min(x)
            gamma_init = np.min(x) - value_range / 10
            return gamma_init, x.sum() / (n * flag).sum(), 2.0, 1.0
        else:
            return self.fit(x, c, n, how="MPP").params

    def sf(self, x, alpha, beta):
        """

        Survival (or reliability) function for the LogLogistic Distribution:

        .. math::
            R(x) = 1 - \frac{1}{1 + \\left ( x /\alpha \right )^{-\beta}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.sf(x, 3, 4)
        array([0.62245933, 0.5621765 , 0.5       , 0.4378235 , 0.37754067])
        """
        exp_term = (x / alpha) ** -beta
        return exp_term / (1 + exp_term)

    def cs(self, x, X, alpha, beta):
        r"""

        Conditional survival function for the LogLogistic Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        X : numpy array or scalar
            The value(s) at which each value(s) in x was known to have survived
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        cs : scalar or numpy array
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.cs(x, 5, 3, 4)
        array([0.51270879, 0.28444803, 0.16902083, 0.10629329, 0.07003273])
        """
        return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

    def ff(self, x, alpha, beta):
        r"""

        Failure (CDF or unreliability) function for the LogLogistic
        Distribution:

        .. math::
            F(x) = \frac{1}{1 + \left ( x /\alpha \right )^{-\beta}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.ff(x, 3, 4)
        array([0.01219512, 0.16494845, 0.5       , 0.75964392, 0.88526912])
        """
        return 1.0 / (1 + (x / alpha) ** -beta)

    def df(self, x, alpha, beta):
        r"""

        Density function for the LogLogistic Distribution:

        .. math::
            f(x) = \frac{\left ( \beta / \alpha \right ) \left ( x / \alpha
            \right )^{\beta - 1}}{\left ( 1 + \left ( x / \alpha
            \right )^{-\beta} \right )^2}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.df(x, 3, 4)
        array([0.0481856 , 0.27548092, 0.33333333, 0.18258504, 0.08125416])
        """
        return ((beta / alpha) * (x / alpha) ** (beta - 1.0)) / (
            (1.0 + (x / alpha) ** beta) ** 2.0
        )

    def hf(self, x, alpha, beta):
        r"""

        Instantaneous hazard rate for the LogLogistic Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.hf(x, 3, 4)
        array([0.04878049, 0.32989691, 0.66666667, 0.75964392, 0.7082153 ])
        """
        return self.df(x, alpha, beta) / self.sf(x, alpha, beta)

    def Hf(self, x, alpha, beta):
        r"""

        Cumulative hazard rate for the LogLogistic Distribution:

        .. math::
            H(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> LogLogistic.Hf(x, 3, 4)
        array([0.01227009, 0.18026182, 0.69314718, 1.42563378, 2.16516608])
        """
        return -np.log(self.sf(x, alpha, beta))

    def qf(self, p, alpha, beta):
        r"""

        Quantile function for the LogLogistic distribution:

        .. math::
            q(p) = \alpha \left ( \frac{p}{1 - p} \right )^{\frac{1}{\beta}}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the LogLogistic distribution at each value p

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> LogLogistic.qf(p, 3, 4)
        array([1.73205081, 2.12132034, 2.42732013, 2.71080601, 3.        ])
        """
        return alpha * (p / (1 - p)) ** (1.0 / beta)

    def mean(self, alpha, beta):
        r"""

        Mean of the LogLogistic distribution

        .. math::
            E = \frac{\alpha \pi / \beta}{sin \left ( \pi / \beta \right )}

        Parameters
        ----------

        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the LogLogistic distribution

        Examples
        --------
        >>> from surpyval import LogLogistic
        >>> LogLogistic.mean(3, 4)
        3
        """
        if beta > 1:
            return (alpha * np.pi / beta) / (np.sin(np.pi / beta))
        else:
            return np.nan

    def random(self, size, alpha, beta):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        alpha : numpy array or scalar
            scale parameter for the LogLogistic distribution
        beta : numpy array or scalar
            shape parameter for the LogLogistic distribution

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import LogLogistic
        >>> LogLogistic.random(10, 3, 4)
        array([4.46072122, 2.1336253 , 2.74159711, 2.90125715, 3.2390347 ,
               5.45223664, 4.28281376, 2.7017541 , 3.023811  , 2.16225601])
        >>> LogLogistic.random((5, 5), 3, 4)
        array([[1.97744499, 4.02823921, 1.95761719, 1.20481591, 3.7166738 ],
               [2.94863864, 3.02609811, 3.30563774, 2.39100075, 3.24937459],
               [3.16102391, 1.77003533, 4.73831093, 0.36936215, 1.41566853],
               [3.88505024, 2.88183095, 2.43977804, 2.62385959, 3.40881857],
               [1.2349273 , 1.83914641, 3.68502568, 6.49834769, 8.62995574]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, alpha, beta)

    def log_df(self, x, alpha, beta):
        return (
            np.log(beta / alpha)
            + (beta - 1) * np.log(x / alpha)
            - 2 * np.log(1 + (x / alpha) ** beta)
        )

    def log_sf(self, x, alpha, beta):
        return beta * np.log(alpha) - np.log(alpha**beta + x**beta)

    def log_ff(self, x, alpha, beta):
        return beta * np.log(x) - np.log(alpha**beta + x**beta)

    def mpp_x_transform(self, x, gamma=0):
        return np.log(x - gamma)

    def mpp_y_transform(self, y, *params):
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = -np.log(1.0 / y[~mask] - 1)
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1.0 / (np.exp(-y) + 1)

    def unpack_rr(self, params, rr):
        if rr == "y":
            beta = params[0]
            alpha = np.exp(params[1] / -beta)
        elif rr == "x":
            beta = 1.0 / params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta

    def moment(self, n, alpha, beta):
        return fisk.moment(n, beta, scale=alpha)


LogLogistic: ParametricFitter = LogLogistic_("LogLogistic")
