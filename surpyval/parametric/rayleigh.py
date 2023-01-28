from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.stats import uniform

from surpyval import nonparametric as nonp
from surpyval import np
from surpyval.parametric.parametric_fitter import ParametricFitter


class Rayleigh_(ParametricFitter):
    def __init__(self, name):
        super().__init__(name)
        # Set 'k', the number of parameters
        self.k = 1
        self.bounds = ((0, None),)
        self.support = (0, np.inf)
        self.plot_x_scale = "linear"
        self.y_ticks = [
            0.0001,
            0.0002,
            0.0003,
            0.001,
            0.002,
            0.003,
            0.005,
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
        self.param_names = ["sigma"]
        self.param_map = {"sigma": 0}

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        if offset:
            return 1.0, np.min(x) - 1
        else:
            return 1.0

    def sf(self, x, sigma):
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
        return np.exp(-(x**2) / (2 * sigma**2))

    def ff(self, x, sigma):
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
        return 1 - np.exp(-(x**2) / (2 * sigma**2))

    def cs(self, x, X, sigma):
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
        return self.sf(x, sigma) / self.sf(X, sigma)

    def df(self, x, sigma):
        r"""

        Density function for the Weibull Distribution:

        .. math::
            f(x) = \frac{\beta}{\alpha} \frac{x}{\alpha}^{\beta - 1} e^{-\left
            ( \frac{x}{\alpha} \right )^\beta}

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
        return (x / (sigma**2)) * self.sf(x, sigma)

    def hf(self, x, sigma):
        r"""

        Instantaneous hazard rate for the Weibull Distribution:

        .. math::
            h(x) = \frac{\beta}{\alpha} \left ( \frac{x}{\alpha} \right
            )^{\beta - 1}

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
        return x / (sigma**2)

    def Hf(self, x, sigma):
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
        return x**2 / (2 * sigma**2)

    def qf(self, p, sigma):
        r"""

        Quantile function for the Weibull distribution:

        .. math::
            q(p) = \alpha \left ( -\ln \left ( 1 - p \right ) \right )^{1/
            \beta}

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
        return sigma * np.sqrt(2 * np.log(1 / (1 - p)))

    def mean(self, sigma):
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
        return sigma * np.sqrt(np.pi / 2)

    def moment(self, n, sigma):
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
        return (sigma**n) * (2 ** (n / 2)) * gamma_func(1 + n / 2)

    def entropy(self, sigma):
        return euler_gamma / 2 + 1 + np.log(sigma / (np.sqrt(2)))

    def random(self, size, sigma):
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
        return self.qf(U, sigma)

    def mpp(
        self,
        x,
        c=None,
        n=None,
        heuristic="Nelson-Aalen",
        rr="y",
        on_d_is_0=False,
        offset=False,
    ):
        assert rr in ["x", "y"]
        x_pp, r, d, F = nonp.plotting_positions(
            x, c=c, n=n, heuristic=heuristic
        )

        if not on_d_is_0:
            x_pp = x_pp[d > 0]
            F = F[d > 0]

        # Linearise
        y_pp = self.mpp_y_transform(F)
        x_pp = self.mpp_x_transform(x_pp)

        # mask = np.isinfinite(y_pp)
        # if mask.any():
        #     warnings.warn("Some Infinite values encountered in plotting
        # points and have been ignored.", stacklevel=2)
        #     y_pp = y_pp[mask]
        #     x_pp = x_pp[mask]

        if offset:
            if rr == "y":
                params = np.polyfit(x_pp, y_pp, 1)
                sigma = np.sqrt(0.5) * (1.0 / params[0])
                gamma = -params[1] / params[0]
                params = [sigma]
            elif rr == "x":
                params = np.polyfit(y_pp, x_pp, 1)
                sigma = np.sqrt(0.5) * (params[0])
                gamma = params[1]
                params = [sigma]

            return {"params": params, "gamma": gamma}

        else:
            if rr == "y":
                x_pp = x_pp[:, np.newaxis]
                gradient = np.linalg.lstsq(x_pp, y_pp, rcond=None)[0]
                sigma = np.sqrt(0.5) * (1 / gradient[0])
            elif rr == "x":
                y_pp = y_pp[:, np.newaxis]
                gradient = np.linalg.lstsq(y_pp, x_pp, rcond=None)[0]
                sigma = np.sqrt(0.5) * (gradient[0])

            params = [sigma]

            return {"params": params}

    def mpp_x_transform(self, x):
        return x

    def mpp_inv_x_transform(self, x, gamma=0):
        return x - gamma

    def mpp_y_transform(self, y, *params):
        mask = y == 0
        out = np.zeros_like(y)
        out[~mask] = np.sqrt(-np.log(1 - y[~mask]))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1 - np.exp(-(y**2))


Rayleigh: ParametricFitter = Rayleigh_("Rayleigh")
