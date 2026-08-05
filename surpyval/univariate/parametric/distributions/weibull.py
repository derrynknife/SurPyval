import numpy.typing as npt
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    ParametricFitter,
)


class Weibull_(ParametricFitter):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            param_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="log",
        )

    def _parameter_initialiser(
        self,
        x: Numeric,
        c: npt.NDArray | None = None,
        n: npt.NDArray | None = None,
        t: npt.NDArray | None = None,
        offset: bool = False,
    ) -> tuple[float, ...]:
        mpp_model = self.fit(
            x, c, n, offset=offset, how="MPP", heuristic="Nelson-Aalen"
        )
        if offset:
            return (mpp_model.gamma, *mpp_model.params)
        else:
            return tuple(mpp_model.params)

    def sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
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
        return np.exp(-((x / alpha) ** beta))

    def ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
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

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.ff(x, 3, 4)
        array([0.01226978, 0.17924519, 0.63212056, 0.9575952 , 0.99955438])
        """
        # np.expm1 is accurate for small values of x while being the
        # same as np.exp for large values
        return -np.expm1(-((x / alpha) ** beta))

    def cs(
        self, x: Numeric, X: Numeric, alpha: Boxable, beta: Boxable
    ) -> Boxable:
        r"""

        Conditional survival function for the Weibull Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        X : numpy array or scalar
            The values at which the item is known to have survived
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
        array([2.52537548e-04, 3.00394073e-10, 2.45288508e-19, 1.48999440e-32,
               5.42544000e-51])
        """
        return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

    def df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
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
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.df(x, 3, 4)
        array([0.0487768 , 0.32424881, 0.49050592, 0.13402009, 0.00275073])
        """
        return (
            (beta / alpha)
            * (x / alpha) ** (beta - 1)
            * np.exp(-((x / alpha) ** beta))
        )

    def hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
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

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.hf(x, 3, 4)
        array([0.04938272, 0.39506173, 1.33333333, 3.16049383, 6.17283951])
        """
        return (beta / alpha) * (x / alpha) ** (beta - 1)

    def Hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
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

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.Hf(x, 3, 4)
        array([0.01234568, 0.19753086, 1.        , 3.16049383, 7.71604938])
        """
        return (x / alpha) ** beta

    def qf(self, p: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
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
        return alpha * (-np.log1p(-p)) ** (1 / beta)

    def mean(self, alpha: Boxable, beta: Boxable) -> Boxable:
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
        np.float64(2.7192074311664314)
        """
        return alpha * gamma_func(1 + 1.0 / beta)

    def moment(self, n: int, alpha: Boxable, beta: Boxable) -> Boxable:
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
        np.float64(7.976042329074821)
        """
        return alpha**n * gamma_func(1 + n / beta)

    def entropy(self, alpha: Boxable, beta: Boxable) -> Boxable:
        return euler_gamma * (1 - 1 / beta) + np.log(alpha) - np.log(beta) + 1

    def log_df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        scaled = x / alpha
        return (
            np.log(beta)
            - np.log(alpha)
            + (beta - 1) * np.log(scaled)
            - (scaled) ** beta
        )

    def log_sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        return -((x / alpha) ** beta)

    def mpp_x_transform(self, x: Numeric) -> npt.NDArray:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> npt.NDArray:
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log(1 - y[~mask]))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Boxable:
        return 1 - np.exp(-np.exp(y))

    def unpack_rr(
        self, params: npt.NDArray, rr: str
    ) -> tuple[Boxable, Boxable]:
        if rr == "y":
            beta = params[0]
            alpha = np.exp(params[1] / -beta)
        elif rr == "x":
            beta = 1.0 / params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta


# Deliberately not annotated ``: ParametricFitter``. That declaration
# erases the concrete type, and the base class declares none of sf, ff,
# df, hf, Hf, qf or mean -- so with py.typed shipped, the example in
# ``sf``'s own docstring did not type check for a user:
#
#     Weibull.sf(x, 3, 4)
#     error: "ParametricFitter" has no attribute "sf"
#
# Naming ``Weibull_`` exposes the real signatures, which is what makes
# the annotations on them worth having. Anywhere a ``ParametricFitter``
# is wanted this still is one.
#
# The annotation cannot simply be dropped and inferred: the regression
# subpackages and ``fit_best`` import this name, and without an explicit
# type mypy cannot resolve it through that cycle ("Cannot determine type
# of Weibull", in six modules).
Weibull: Weibull_ = Weibull_("Weibull")
