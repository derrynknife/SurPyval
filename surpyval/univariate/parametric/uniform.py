from scipy.optimize import minimize
from scipy.stats import uniform

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class Uniform_(ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((None, None), (None, None)),
            support=(np.nan, np.nan),
            param_names=["a", "b"],
            param_map={"a": 0, "b": 1},
            plot_x_scale="linear",
            y_ticks=np.linspace(0, 1, 21)[1:-1],
        )

    def _parameter_initialiser(self, x, c=None, n=None):
        return np.min(x) - 1.0, np.max(x) + 1.0

    def sf(self, x, a, b):
        r"""

        Surival (or Reliability) function for the Uniform Distribution:

        .. math::
            R(x) = \frac{b - x}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.sf(x, 0, 6)
        array([0.83333333, 0.66666667, 0.5       , 0.33333333, 0.16666667])
        """
        return 1 - self.ff(x, a, b)

    def cs(self, x, X, a, b):
        r"""

        Surival (or Reliability) function for the Uniform Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        cs : scalar or numpy array
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.cs(x, 4, 0, 10)
        array([0.83333333, 0.66666667, 0.5       , 0.33333333, 0.16666667])
        """
        return self.sf(x + X, a, b) / self.sf(X, a, b)

    def ff(self, x, a, b):
        r"""

        Failure (CDF or unreliability) function for the Uniform Distribution:

        .. math::
            F(x) = \frac{x - a}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.sf(x, 0, 6)
        array([0.16666667, 0.33333333, 0.5       , 0.66666667, 0.83333333])
        """
        f = np.zeros_like(x)
        f = np.where(x < a, 0, f)
        f = np.where(x > b, 1, f)
        f = np.where(((x <= b) & (x >= a)), (x - a) / (b - a), f)
        return f

    def df(self, x, a, b):
        r"""

        Failure (CDF or unreliability) function for the Uniform Distribution:

        .. math::
            f(x) = \frac{1}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.df(x, 0, 6)
        array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667])
        """
        d = np.zeros_like(x)
        d = np.where(x < a, 0, d)
        d = np.where(x > b, 0, d)
        d = np.where(((x <= b) & (x >= a)), 1.0 / (b - a), d)
        return d

    def hf(self, x, a, b):
        r"""

        Instantaneous hazard rate for the Uniform Distribution:

        .. math::
            h(x) = \frac{1}{b - x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.hf(x, 0, 6)
        array([0.2       , 0.25      , 0.33333333, 0.5       , 1.        ])
        """
        return self.df(x, a, b) / self.sf(x, a, b)

    def Hf(self, x, a, b):
        r"""

        Instantaneous hazard rate for the Uniform Distribution:

        .. math::
            H(x) = \ln \left ( b - a \right ) - \ln \left ( b - x \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.Hf(x, 0, 6)
        array([0.18232156, 0.40546511, 0.69314718, 1.09861229, 1.79175947])
        """
        return -np.log(self.sf(x, a, b))

    def qf(self, p, a, b):
        r"""

        Quantile function for the Uniform Distribution:

        .. math::
            q(p) = a + p(b - a)

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Uniform distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Uniform.qf(p, 0, 6)
        array([0.6, 1.2, 1.8, 2.4, 3. ])
        """
        return a + p * (b - a)

    def mean(self, a, b):
        r"""

        Mean of the Uniform distribution

        .. math::
            E = \frac{1}{2} \left ( a + b \right )

        Parameters
        ----------

        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Uniform distribution

        Examples
        --------
        >>> from surpyval import Uniform
        >>> Uniform.mean(0, 6)
        3.0
        """
        return 0.5 * (a + b)

    def moment(self, n, a, b):
        r"""

        n-th (non central) moment of the Uniform distribution

        .. math::
            M(n) = \frac{1}{n +1} \sum_{i=0}^{n}a^ib^{n-i}

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the Uniform distribution

        Examples
        --------
        >>> from surpyval import Uniform
        >>> Uniform.moment(2, 0, 6)
        12.0
        """
        if n == 0:
            return 1
        else:
            out = np.zeros(n)
            for i in range(n):
                out[i] = a**i * b ** (n - i)
            return np.sum(out) / (n + 1)

    def p(self, c, n):
        return 1 - 2 * (1 + c) ** (1.0 - n) + (1 + 2 * c) ** (1.0 - n)

    def random(self, size, a, b):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> Uniform.random(10, 0, 6)
        array([3.50214341, 3.7978912 , 5.12238656, 4.27185221, 3.05507685,
               2.71236199, 4.89311322, 1.11373047, 4.90549424, 1.76321338])
        >>> Uniform.random((5, 5), 0, 6)
        array([[4.76809829, 4.42155933, 2.59469997, 4.31525748, 5.53469545],
               [0.06222942, 1.26267164, 1.74188626, 1.05235807, 0.92461476],
               [2.06215303, 0.02184135, 0.97058002, 3.02219656, 3.22137982],
               [2.14951891, 3.18096661, 2.37105309, 0.65710124, 0.68828779],
               [0.58827207, 3.7633596 , 5.62330526, 5.24481753, 4.23162212]])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, a, b)

    def ab_cb(self, x, a, b, N, alpha=0.05):
        # Parameter confidence intervals from here:
        # https://mathoverflow.net/questions/278675/confidence-intervals-for-the-endpoints-of-the-uniform-distribution
        #
        sample_range = np.max(x) - np.min(x)

        def fun(c):
            return self.p(c, N)

        c_hat = minimize(fun, 1.0).x
        return a - c_hat * sample_range, b + c_hat * sample_range

    def mle(self, data):
        if (data.c[data.x == data.x.max()] == 1).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                + " the highest value is right censored"
            )

        if (data.c[data.x == data.x.min()] == -1).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                + " the lowest value is left censored"
            )

        tl = data.t[:, 0]
        tr = data.t[:, 1]

        if np.isfinite(tr[data.x == data.x.max()]).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                + " the highest value is right truncated"
            )

        if np.isfinite(tl[data.x == data.x.min()]).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                + " the lowest value is left truncated"
            )

        params = np.array([np.min(data.x), np.max(data.x)])
        results = {}
        results["params"] = params
        return results

    def mpp_x_transform(self, x):
        return x

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_inv_y_transform(self, y, *params):
        return y

    def unpack_rr(self, params, rr):
        if rr == "y":
            a = -params[1] / params[0]
            b = (1 - params[1]) / params[0]
        if rr == "x":
            a = params[1]
            b = params[0] + params[1]

        return a, b

    def _mom(self, x):
        mu_1 = np.mean(x)
        mu_2 = np.mean(x**2)

        d = np.sqrt(3 * (mu_2 - mu_1**2))
        a = mu_1 - d
        b = mu_1 + d
        return a, b


Uniform: ParametricFitter = Uniform_("Uniform")
