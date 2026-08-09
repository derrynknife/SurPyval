from numpy import euler_gamma
from scipy.special import gamma as gamma_func

from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions
from surpyval.univariate.parametric.parametric_fitter import (
    OptimisedFitMixin,
    ParametricFitter,
)


class Rayleigh_(OptimisedFitMixin, ParametricFitter):
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

    def _parameter_initialiser(self, data, offset=False):
        x = data.x
        # sqrt(E[x^2] / 2) is the closed-form uncensored MLE for sigma
        if offset:
            gamma_init = np.min(x) - 1.0
            sigma_init = np.sqrt(np.mean((x - gamma_init) ** 2) / 2)
            return np.array([gamma_init, sigma_init], dtype=float)
        # A one-tuple, not the bare scalar this used to return. Rayleigh
        # is the only single-parameter distribution here, and the scalar
        # made `np.array(init)` in _initial_guess 0-dimensional rather
        # than length-1. The lfp and zi paths then concatenate the p and
        # f0 seeds onto it, and a 0-d array cannot be concatenated, so
        # `Rayleigh.fit(x, lfp=True)` and `zi=True` both raised
        # "zero-dimensional arrays cannot be concatenated".
        return np.array(
            [
                np.sqrt(np.mean(x**2) / 2),
            ],
            dtype=float,
        )

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

    def qf(self, u, sigma):
        r"""

        Quantile function for the Rayleigh distribution:

        .. math::
            q(u) = \sigma \sqrt{-2 \ln \left ( 1 - u \right )}

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        sigma : numpy array or scalar
            scale parameter for the Rayleigh distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Rayleigh distribution at each value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Rayleigh
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Rayleigh.qf(u, 3)
        array([1.37713082, 2.00414169, 2.53380129, 3.03230296, 3.53223007])
        """
        return sigma * np.sqrt(2 * np.log(1 / (1 - u)))

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
        np.float64(3.7599424119465006)
        """
        return sigma * np.sqrt(np.pi / 2)

    def moment(self, m, sigma):
        r"""

        m-th moment of the Rayleigh distribution

        .. math::
            M(m) = \sigma^m 2^{m/2} \Gamma \left ( 1 + \frac{m}{2} \right )

        Parameters
        ----------

        m : integer
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
        np.float64(18.0)
        """
        return (sigma**m) * (2 ** (m / 2)) * gamma_func(1 + m / 2)

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
        t=None,
        heuristic="Nelson-Aalen",
        rr="y",
        on_d_is_0=False,
        offset=False,
    ):
        assert rr in ["x", "y"]
        # Forward the truncation windows: the custom Rayleigh path used to
        # drop them silently, making fits with and without tl/tr
        # bit-identical (#280).
        x_pp, r, d, F = plotting_positions(
            x, c=c, n=n, t=t, heuristic=heuristic
        )

        if not on_d_is_0:
            x_pp = x_pp[d > 0]
            F = F[d > 0]

        # Plotting positions of exactly 0 or 1 have no finite transform
        # (sqrt(-log 0) = inf poisoned e.g. the ECDF heuristic, #280).
        valid = (F != 0) & (F != 1)
        x_pp = x_pp[valid]
        F = F[valid]

        # Linearise
        y_pp = self.mpp_y_transform(F)
        x_pp = self.mpp_x_transform(x_pp)

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

    def mpp_y_transform(self, y, *params):
        mask = y == 0
        out = np.zeros_like(y)
        out[~mask] = np.sqrt(-np.log(1 - y[~mask]))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        return 1 - np.exp(-(y**2))


Rayleigh: Rayleigh_ = Rayleigh_("Rayleigh")
