from autograd.scipy.special import gamma as agamma
from autograd.scipy.special import gammaln as agammaln
from scipy.special import digamma, gammaincinv

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.autograd_gamma_compat import gammainc as agammainc
from surpyval.utils.autograd_gamma_compat import gammainccln as agammainccln
from surpyval.utils.autograd_gamma_compat import gammaincln as agammaincln


class Gamma_(OptimisedFitMixin, ParametricFitter):
    r"""

    Class used to generate the Gamma class.

    .. code:: python

        from surpyval import Gamma

    """

    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            param_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="linear",
        )
        # The Gamma has no linearising probability plot, for the same
        # reason as the Beta above it: the CDF is the regularised
        # incomplete gamma function, and the shape sits *inside* that
        # special function rather than outside it as an exponent. The
        # only straight-line y-axis is the inverse incomplete gamma,
        # which needs the shape -- so to draw the axis you need the
        # answer, and to get the answer you need the axis.
        #
        # MPP broke the circle by guessing the shape from moments,
        # drawing the plot on that guess and regressing. When the guess
        # is off the axis is the wrong axis, the points are no longer
        # straight on it, and the regression fits a line through a
        # curve -- returning a confident, wrong estimate rather than an
        # error. An offset makes it worse: the shift distorts the low-x
        # end hardest, which is exactly where the shape information is.
        #
        # Fit by MLE (the default), MSE or MOM instead. ``plot()`` still
        # works, because it transforms with the *fitted* parameters, so
        # the axis is the right one by the time it is drawn.
        self.supports_mpp = False

    @staticmethod
    def _moment_estimate(x):
        """Closed-form approximation to the Gamma MLE.

        The shape solves ``log(alpha) - digamma(alpha) = s`` with
        ``s = log(mean x) - mean(log x)``; this is the standard
        approximation to that root (Minka 2002, after Thom 1958), good
        to about 1.5% and used only as a starting point.
        """
        s = np.log(x.sum() / len(x)) - np.log(x).sum() / len(x)
        # s is exactly zero for a tied sample -- the log of the mean and
        # the mean of the logs coincide -- and alpha divides by it, so
        # the seed comes back as (inf, inf). A failed optimiser falls
        # back to its initial guess (#261), so those infinities are
        # returned to the caller as the fitted parameters. Seed the
        # exponential case instead: a tied sample carries no information
        # about the shape.
        if not np.isfinite(s) or s <= np.finfo(float).tiny:
            return 1.0, len(x) / x.sum()
        alpha = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        beta = x.sum() / (len(x) * alpha)
        return alpha, 1.0 / beta

    def _parameter_initialiser(self, data, offset=False):
        x = data.x
        if offset:
            # ``gamma`` leads the vector, as it does for every other
            # offset-capable distribution. Returning it last put the
            # shape in slot 0, where ``_initial_guess`` overwrites that
            # slot with the offset seed -- so the shape estimate was
            # destroyed and the offset written into the scale as well.
            #
            # The moments must also be taken *after* the shift. On
            # offset data ``s = log(mean x) - mean(log x)`` is squashed
            # towards zero by the constant, and since alpha grows like
            # ``1 / 12s`` the estimate explodes: 649 for a true shape of
            # 3. Together these made MSE and MOM offset fits return
            # silent nonsense.
            gamma_init = np.min(x) - 1.0
            alpha, beta = self._moment_estimate(x - gamma_init)
            return np.array([gamma_init, alpha, beta], dtype=float)
        return np.asarray(self._moment_estimate(x), dtype=float)

    def sf(self, x, alpha, beta):
        r"""

        Survival (or Reliability) function for the Gamma Distribution:

        .. math::
            R(x) = 1 - \frac{\gamma \left ( \alpha, \beta x \right )
            }{\Gamma \left ( \alpha \right )}

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
        r"""

        Conditional survival function for the Gamma Distribution:

        .. math::
            R(x) = e^{-\lambda x}

        Parameters
        ----------

        x : numpy array or scalar
            The value(s) at which the function will be calculated
        X : numpy array or scalar
            The value(s) at which each value(s) in x was known to have survived
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution

        Returns
        -------

        cs : scalar or numpy array
            the conditional survival probability.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Gamma.cs(x, 5, 3, 4)
        array([2.59402488e-02, 6.39048747e-04, 1.51519143e-05, 3.48776510e-07,
               7.79933496e-09])
        """
        return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

    def ff(self, x, alpha, beta):
        r"""

        CDF (or unreliability or failure) function for the Gamma Distribution:

        .. math::
            F(x) = \frac{\gamma \left ( \alpha, \beta x \right )}
            {\Gamma \left ( \alpha \right )}

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
        x = np.array(x)
        return agammainc(alpha, beta * x)

    def df(self, x, alpha, beta):
        r"""

        Density function for the Gamma Distribution:

        .. math::
            f(x) = \frac{\beta^{\alpha }}{\Gamma \left ( \alpha \right )}
            x^{\alpha - 1}e^{-\beta x}

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
        return (
            (beta**alpha)
            * x ** (alpha - 1)
            * np.exp(-(x * beta))
            / (agamma(alpha))
        )

    def hf(self, x, alpha, beta):
        r"""

        Instantaneous hazard rate for the Gamma Distribution:

        .. math::
            h(x) = \frac{\frac{\beta^{\alpha }}{\Gamma \left ( \alpha \right )
            }x^{\alpha - 1}e^{-\beta x}}{1 - \frac{\gamma \left ( \alpha, \beta
            x \right )}{\Gamma \left ( \alpha \right )}}

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

        hf : scalar or numpy array
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
            H(x) = -\ln(1 - \frac{\gamma \left ( \alpha, \beta x \right )}
            {\Gamma \left ( \alpha \right )})

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
        r"""

        Quantile function for the Gamma Distribution:

        .. math::
            q(p) = \frac{-\ln\left ( p \right )}{\lambda}

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Gamma distribution at each value p.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Gamma
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> Gamma.qf(p, 3, 4)
        array([0.27551633, 0.38376105, 0.47844395, 0.57126923, 0.66851508])
        """
        return gammaincinv(alpha, p) / beta

    def mean(self, alpha, beta):
        r"""

        Calculates the mean of the Gamma distribution with given parameters.

        .. math::
            E = \frac{\alpha}{\beta}

        Parameters
        ----------

        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Gamma distribution

        Examples
        --------
        >>> from surpyval import Gamma
        >>> Gamma.mean(3, 4)
        0.75
        """
        return alpha / beta

    def moment(self, n, alpha, beta):
        r"""

        Calculates the n-th moment of the Gamma distribution with
        given parameters.

        .. math::
            E = \frac{\Gamma \left ( n + \alpha \right )}{\beta^{n}\Gamma
            \left ( \alpha \right )}

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The scale parameter for the Gamma distribution

        Returns
        -------

        mean : scalar or numpy array
            The moment(s) of the Gamma distribution

        Examples
        --------
        >>> from surpyval import Gamma
        >>> Gamma.moment(3, 3, 4)
        np.float64(0.9375)
        """
        return agamma(n + alpha) / (beta**n * agamma(alpha))

    def entropy(self, alpha, beta):
        r"""

        Calculates the entropy of the Gamma distribution.

        .. math::
            S = \alpha - \ln \left ( \beta \right ) + \ln \Gamma \left (
            \alpha \right ) + \left ( 1 - \alpha \right ) \psi \left (
            \alpha \right )

        Where psi is the digamma function

        Parameters
        ----------

        alpha : numpy array or scalar
            The shape parameter for the Gamma distribution
        beta : numpy array or scalar
            The rate parameter for the Gamma distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the Gamma distribution

        Examples
        --------
        >>> from surpyval import Gamma
        >>> Gamma.entropy(3, 4)
        np.float64(0.46128414924312033)
        """
        return (
            alpha
            - np.log(beta)
            + agammaln(alpha)
            + (1 - alpha) * digamma(alpha)
        )

    def log_df(self, x, alpha, beta):
        r"""

        Calculates the log of the density function of the Gamma distribution
        at x.

        .. math::
            \log f(x) = \log \left ( \frac{\lambda^{\alpha}}{\Gamma(\alpha)}
            x^{\alpha - 1}e^{-\lambda x} \right )

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

        log_df : scalar or numpy array
            The log of the density function of the Gamma distribution at x

        """
        return (
            alpha * np.log(beta)
            + (alpha - 1) * np.log(x)
            - beta * x
            - agammaln(alpha)
        )

    def log_ff(self, x, alpha, beta):
        return agammaincln(alpha, beta * x)

    def log_sf(self, x, alpha, beta):
        return agammainccln(alpha, beta * x)

    def mpp_y_transform(self, y, *params):
        alpha = params[0]
        return gammaincinv(alpha, y)

    def mpp_inv_y_transform(self, y, *params):
        alpha = params[0]
        return agammainc(alpha, y)

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma


Gamma: Gamma_ = Gamma_("Gamma")
