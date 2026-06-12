from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class Rayleigh_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=1,
            bounds=((0, None),),
            support=(0, np.inf),
            param_names=["sigma"],
            param_map={"sigma": 0},
            plot_x_scale="linear",
            y_ticks=[
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
            ],
        )

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        if offset:
            return 1.0, np.min(x) - 1
        else:
            return 1.0

    def sf(self, x, sigma):
        r"""

        Survival (or reliability) function for the Rayleigh Distribution:

        .. math::
            R(x) = e^{-\frac{x^2}{2\sigma^2}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Rayleigh.sf(x, 3)
        array([0.94595947, 0.8007374 , 0.60653066, 0.41111229, 0.24935221])
        """
        return np.exp(-(x**2) / (2 * sigma**2))

    def ff(self, x, sigma):
        r"""

        Failure (CDF or unreliability) function for the Rayleigh
        Distribution:

        .. math::
            F(x) = 1 - e^{-\frac{x^2}{2\sigma^2}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Rayleigh.ff(x, 3)
        array([0.05404053, 0.1992626 , 0.39346934, 0.58888771, 0.75064779])
        """
        return -np.expm1(-(x**2) / (2 * sigma**2))

    def cs(self, x, X, sigma):
        r"""

        Conditional survival function for the Rayleigh Distribution:

        .. math::
            R(x, X) = \frac{R(x)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        cs : scalar or numpy array
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Rayleigh.cs(x, 5, 3)
        array([3.79366789, 3.21127054, 2.43242545, 1.64872127, 1.        ])
        """
        return self.sf(x, sigma) / self.sf(X, sigma)

    def df(self, x, sigma):
        r"""

        Density function for the Rayleigh Distribution:

        .. math::
            f(x) = \frac{x}{\sigma^2} e^{-\frac{x^2}{2\sigma^2}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Rayleigh.df(x, 3)
        array([0.10510661, 0.17794165, 0.20217689, 0.18271657, 0.138529  ])
        """
        return (x / (sigma**2)) * self.sf(x, sigma)

    def hf(self, x, sigma):
        r"""

        Instantaneous hazard rate for the Rayleigh Distribution:

        .. math::
            h(x) = \frac{x}{\sigma^2}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Rayleigh.hf(x, 3)
        array([0.11111111, 0.22222222, 0.33333333, 0.44444444, 0.55555556])
        """
        return x / (sigma**2)

    def Hf(self, x, sigma):
        r"""

        Cumulative hazard rate for the Rayleigh Distribution:

        .. math::
            H(x) = \frac{x^2}{2\sigma^2}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Rayleigh.Hf(x, 3)
        array([0.05555556, 0.22222222, 0.5       , 0.88888889, 1.38888889])
        """
        return x**2 / (2 * sigma**2)

    def qf(self, p, sigma):
        r"""

        Quantile function for the Rayleigh distribution:

        .. math::
            q(p) = \sigma \sqrt{-2 \ln \left ( 1 - p \right )}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Rayleigh distribution at each value p

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Rayleigh.qf(p, 3)
        array([1.37713082, 2.00414169, 2.53380129, 3.03230296, 3.53223007])
        """
        return sigma * np.sqrt(2 * np.log(1 / (1 - p)))

    def mean(self, sigma):
        r"""

        Mean of the Rayleigh distribution

        .. math::
            E = \sigma \sqrt{\frac{\pi}{2}}

        Parameters
        ----------

        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Rayleigh distribution

        Examples
        --------
        >>> from surpyval import Rayleigh
        >>> Rayleigh.mean(3)
        3.7599424119465006
        """
        return sigma * np.sqrt(np.pi / 2)

    def moment(self, n, sigma):
        r"""

        n-th moment of the Rayleigh distribution

        .. math::
            M(n) = \sigma^n 2^{n/2} \Gamma \left ( 1 + \frac{n}{2} \right )

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the Rayleigh distribution

        Examples
        --------
        >>> from surpyval import Rayleigh
        >>> Rayleigh.moment(2, 3)
        18.0
        """
        return (sigma**n) * (2 ** (n / 2)) * gamma_func(1 + n / 2)

    def entropy(self, sigma):
        return euler_gamma / 2 + 1 + np.log(sigma / (np.sqrt(2)))

    def log_df(self, x, sigma):
        return np.log(x) - 2 * np.log(sigma) - 0.5 * (x / sigma) ** 2

    def log_sf(self, x, sigma):
        return -0.5 * (x / sigma) ** 2

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
        x_pp, r, d, F = plotting_positions(x, c=c, n=n, heuristic=heuristic)

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
                params = np.array([sigma])
            elif rr == "x":
                params = np.polyfit(y_pp, x_pp, 1)
                sigma = np.sqrt(0.5) * (params[0])
                gamma = params[1]
                params = np.array([sigma])

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

            params = np.array([sigma])

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
