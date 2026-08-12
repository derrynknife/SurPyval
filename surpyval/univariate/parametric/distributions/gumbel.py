import numpy.typing as npt
from numpy import euler_gamma
from scipy.stats import gumbel_l

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData


class Gumbel_(OptimisedFitMixin, ParametricFitter):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((None, None), (0, None)),
            support=(-np.inf, np.inf),
            param_names=["mu", "sigma"],
            param_map={"mu": 0, "sigma": 1},
            plot_x_scale="linear",
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if (2 in data.c) or (-1 in data.c):
            heuristic = "Turnbull"
        else:
            heuristic = "Nelson-Aalen"
        return np.asarray(
            self.fit_from_surpyval_data(
                data, how="MPP", heuristic=heuristic
            ).params,
            dtype=float,
        )

    def sf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Survival (or Reliability) function for the Gumbel Distribution:

        .. math::
            R(x) = 1 - e^{e^{-\left ( x - \mu \right ) / \sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        sf : scalar or numpy array
            The scalar value of the survival function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the survival function at each
            corresponding value in the input array.


        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.sf(x, 3, 2)
        array([0.69220063, 0.54523921, 0.36787944, 0.19229565, 0.06598804])
        """
        return np.exp(-np.exp((x - mu) / sigma))

    def ff(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        CDF (or Failure) function for the Gumbel Distribution:

        .. math::
            F(x) = e^{e^{-\left ( x - \mu \right )/\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        ff : scalar or numpy array
            The scalar value of the failure function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the failure function at each
            corresponding value in the input array.


        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.ff(x, 3, 2)
        array([0.30779937, 0.45476079, 0.63212056, 0.80770435, 0.93401196])
        """
        return -np.expm1(-self.Hf(x, mu, sigma))

    def df(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Density function (pdf) for the Gumbel Distribution:

        .. math::
            f(x) = \frac{1}{\sigma}e^{\left (\frac{x - \mu}{\sigma} -
            e^{\frac{x-\mu}{\sigma}} \right)}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        df : scalar or numpy array
            The scalar value of the density of the distribution if a scalar was
            passed. If an array like object was passed then a numpy array is
            returned with the value of the density at each corresponding value
            in the input array.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.df(x, 3, 2)
        array([0.12732319, 0.16535215, 0.18393972, 0.15852096, 0.08968704])
        """
        z = (x - mu) / sigma
        return (1 / sigma) * np.exp(z - np.exp(z))

    def hf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Gumbel Distribution:

        .. math::
            h(x) = \frac{1}{\sigma} e^{\frac{x-\mu}{\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) for the instantaneous hazard rate for the Gumbel
            distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.hf(x, 3, 2)
        array([0.18393972, 0.30326533, 0.5       , 0.82436064, 1.35914091])
        """
        z = (x - mu) / sigma
        return (1 / sigma) * np.exp(z)

    def Hf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Gumbel Distribution:

        .. math::
            H(x) = e^{\frac{x-\mu}{\sigma}}

        Parameters
        ----------

        x : numpy array or scalar
            The values of the random variables at which the survival function
            will be calculated
        mu : numpy array like or scalar
            The location parameter of the distribution
        sigma : numpy array like or scalar
            The scale parameter of the distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) for the cumulative hazard rate for the Gumbel
            distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gumbel.Hf(x, 3, 2)
        array([0.36787944, 0.60653066, 1.        , 1.64872127, 2.71828183])
        """
        return np.exp((x - mu) / sigma)

    def qf(self, u: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Quantile function for the Gumbel Distribution:

        .. math::
            q(u) = \mu + \sigma\ln\left ( -\ln\left ( 1 - u \right ) \right )

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array like or scalar
            The location parameter(s) of the distribution
        sigma : numpy array like or scalar
            The scale parameter(s) of the distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Gumbel distribution at each value u.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gumbel
        >>> u = np.array([0.1, 0.3, 0.5])
        >>> Gumbel.qf(u, 3, 2)
        array([-1.50073465,  0.93813913,  2.26697416])
        """
        return mu + sigma * (np.log(-np.log1p(-u)))

    def mean(self, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Calculates the mean of the Gumbel distribution with given parameters.

        .. math::
            E = \mu - \sigma\gamma

        Where gamma is the Euler-Mascheroni constant. The Gumbel
        distribution here is the smallest extreme value distribution,
        so the mean sits below the location parameter.

        Parameters
        ----------

        mu : numpy array like or scalar
            The location parameter(s) of the distribution
        sigma : numpy array like or scalar
            The scale parameter(s) of the distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Gumbel distribution

        Examples
        --------
        >>> from surpyval import Gumbel
        >>> Gumbel.mean(3, 2)
        1.8455686701969343
        """
        return mu - sigma * euler_gamma

    def log_df(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        z = (x - mu) / sigma
        return z - np.exp(z) - np.log(sigma)

    def log_sf(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        return -self.Hf(x, mu, sigma)

    def log_ff(self, x: Numeric, mu: Boxable, sigma: Boxable) -> Boxable:
        return np.log(-np.expm1(-self.Hf(x, mu, sigma)))

    def moment(self, m: int, mu: Boxable, sigma: Boxable) -> Boxable:
        return gumbel_l.moment(m, loc=mu, scale=sigma)

    def entropy(self, mu: Boxable, sigma: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the Gumbel distribution.

        .. math::
            S = \ln \left ( \sigma \right ) + \gamma + 1

        Where gamma is the Euler-Mascheroni constant

        Parameters
        ----------

        mu : numpy array or scalar
            The location parameter(s) of the distribution
        sigma : numpy array or scalar
            The scale parameter(s) of the distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the Gumbel distribution

        Examples
        --------
        >>> from surpyval import Gumbel
        >>> Gumbel.entropy(3, 2)
        np.float64(2.270362845461478)
        """
        return np.log(sigma) + euler_gamma + 1

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log(1 - y[~mask]))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return 1 - np.exp(-np.exp(y))

    def unpack_rr(
        self, params: npt.NDArray, rr: str
    ) -> tuple[Boxable, Boxable]:
        if rr == "y":
            sigma = 1.0 / params[0]
            mu = -sigma * params[1]
        elif rr == "x":
            sigma = params[0]
            mu = params[1]
        return mu, sigma


Gumbel: Gumbel_ = Gumbel_("Gumbel")
GumbelSEV: Gumbel_ = Gumbel_("GumbelSEV")
