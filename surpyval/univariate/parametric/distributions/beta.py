from __future__ import annotations

import autograd.numpy as np
import numpy.typing as npt
from autograd.scipy.special import betaln as abetaln
from scipy.special import betaincinv, digamma

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.autograd_gamma_compat import _log_beta_front
from surpyval.utils.autograd_gamma_compat import betainc as abetainc
from surpyval.utils.autograd_gamma_compat import betainccln as abetainccln
from surpyval.utils.autograd_gamma_compat import betaincln as abetaincln
from surpyval.utils.autograd_gamma_compat import betaln_accurate
from surpyval.utils.surpyval_data import SurpyvalData

from .beta4 import _power_log, _power_log1m


def _log_beta(alpha: Boxable, beta: Boxable) -> Boxable:
    """ln B(alpha, beta), exactly -ln(alpha) when beta is 1 (and the
    mirror), where the density is a power of x alone: at alpha = 1/2,
    beta = 1 the log density at x = 1/4 is then exactly 0."""
    return np.where(
        beta == 1.0,
        -np.log(alpha),
        np.where(alpha == 1.0, -np.log(beta), betaln_accurate(alpha, beta)),
    )


class Beta_(OptimisedFitMixin, ParametricFitter):
    r"""
    The Beta distribution on ``(0, 1)``: a model of a proportion or a
    probability (for an arbitrary interval see :class:`Beta4`).

    The parameters are the shapes ``alpha`` and ``beta``, both positive.
    On :math:`(0, 1)`, with :math:`B` the beta function,

    .. math::
        R(x) = 1 - \frac{1}{B \left ( \alpha, \beta \right )}
        \int_{0}^{x}t^{\alpha-1}\left (1 - t \right )^{\beta - 1}dt.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import Beta
    >>> model = Beta.from_params([2, 5])
    >>> model.sf([0.1, 0.3, 0.5]).round(4)
    array([0.8857, 0.4202, 0.1094])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution
        """

    # rescale_time: on a new clock the Beta lives on (0, 1 / factor), so
    # it is a Beta4 (rescale_time(factor) refuses and says so).
    _time_roles_refusal = (
        "its support is fixed at (0, 1), and on a new clock it is "
        "(0, 1 / factor), which the Beta cannot describe: build the "
        "Beta4 with the same alpha and beta and the ends 0 and "
        "1 / factor, Beta4.from_params([alpha, beta, 0, 1 / factor])"
    )

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, 1),
            parameter_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="linear",
        )
        # The Beta distribution has no linearising probability plot: its CDF
        # is the regularised incomplete beta function (no closed-form inverse),
        # and with two interacting shape parameters on bounded support it is
        # not a location-scale family, so there is no parameter-free transform
        # that turns the probability plot into a straight line. Fit by MLE, MSE
        # or MOM instead (MOM is analytic for the Beta).
        self.supports_mpp = False

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if (data.c == 0).all():
            x = np.repeat(data.x, data.n)
            p = self._mom(x)
        else:
            p = 1.0, 1.0
        return np.asarray(p, dtype=float)

    def sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the Beta Distribution:

        .. math::
            R(x) = 1 - \frac{1}{B \left ( \alpha, \beta \right )}
            \int_{0}^{x}t^{\alpha-1}\left (1 - t \right )^{\beta - 1}dt

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.sf(x, 3, 4)
        array([0.98415, 0.90112, 0.74431, 0.54432, 0.34375])
        """
        # 1 - F where F is below 1/2, and the upper tail itself (from its
        # log) where it is small: 1 - F was 0 where R is 1e-30 (#458).
        ff = self.ff(x, alpha, beta)
        return np.where(
            ff < 0.5, 1.0 - ff, np.exp(self.log_sf(x, alpha, beta))
        )

    def ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Beta Distribution:

        .. math::
            F(x) = \frac{1}{B \left ( \alpha, \beta \right )}
            \int_{0}^{x}t^{\alpha-1}\left (1 - t \right )^{\beta -1}dt

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.ff(x, 3, 4)
        array([0.01585, 0.09888, 0.25569, 0.45568, 0.65625])
        """
        return abetainc(alpha, beta, x)

    def df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Density function for the Beta Distribution:

        .. math::
            f(x) = \frac{x^{\alpha-1}\left(1 - x \right )^{\beta-1}}{B \left (
            \alpha , \beta \right )}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.df(x, 3, 4)
        array([0.4374, 1.2288, 1.8522, 2.0736, 1.875 ])
        """
        # From the log density: B(1000, 1000) underflows, and the ratio
        # of the powers to it was 0/0 (#458).
        return np.exp(self.log_df(x, alpha, beta))

    def hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Beta distribution.

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.hf(x, 3, 4)
        array([0.44444444, 1.36363636, 2.48847926, 3.80952381, 5.45454545])
        """
        # On the log scale: df/sf was 0/0 where both underflow, and inf
        # where only sf did (#458). At x = 1 nothing survives: inf.
        x = np.asarray(x, dtype=float)
        log_hf = self.log_df(x, alpha, beta) - self.log_sf(x, alpha, beta)
        return np.where(x >= 1.0, np.inf, np.exp(log_hf))

    def Hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Beta distribution.

        .. math::
            H(x) = -\ln\left( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> x = np.array([.1, .2, .3, .4, .5])
        >>> Beta.Hf(x, 3, 4)
        array([0.01597695, 0.10411684, 0.29529766, 0.60821797, 1.06784063])
        """
        return -self.log_sf(x, alpha, beta)

    def qf(self, u: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Quantile function for the Beta Distribution:

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Beta distribution at each value u.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Beta
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Beta.qf(u, 3, 4)
        array([0.20090888, 0.26864915, 0.32332388, 0.37307973, 0.42140719])
        """
        u = np.asarray(u, dtype=float)
        x = betaincinv(alpha, beta, u)
        # scipy stops at the smallest normal double where the quantile is
        # far below it (1e-29699 at shapes 1e-3). There I_x(a, b) is
        # x^a / (a B(a, b)) to first order, so log x is known (#458).
        # The next term is a relative (b - 1) a x / (a + 1), so the first
        # order is exact to double precision below x (|b| + 1) of 1e-17.
        with np.errstate(divide="ignore"):
            # log(a B(a, b)) to its own precision (log a + betaln lost
            # 2e-13 of it at a = 1e-3, b = 1e3, which / a is 2e-10 of x)
            front = _log_beta_front(
                np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
            )
            log_x = (np.log(u) - front) / alpha
        small = np.exp(log_x)
        exact = small * (np.abs(beta) + 1.0) < 1e-17
        return np.where(exact | ~(x > np.finfo(float).tiny), small, x)

    def isf(self, s: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""The inverse survival function: the time at which the survival
        function is ``s``.

        Below ``s = 1/2`` it is one less the quantile at ``s`` of the Beta
        with the shapes swapped (the law of ``1 - X``), and ``qf(1 - s)``
        above it.

        Examples
        --------
        >>> from surpyval import Beta
        >>> Beta.isf([0.5, 1e-300], 2, 3)
        array([0.38572757, 1.        ])
        """
        s = np.asarray(s, dtype=float)
        upper = s < 0.5
        # 1 - X is Beta(beta, alpha): its lower quantile at s, whose small
        # values qf keeps (scipy's betainccinv gave NaN at s = 1e-300)
        return np.where(
            upper,
            1.0 - self.qf(np.where(upper, s, 0.25), beta, alpha),
            self.qf(np.where(upper, 0.5, 1.0 - s), alpha, beta),
        )

    def mean(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Mean of the Beta distribution

        .. math::
            E = \frac{\alpha}{\alpha + \beta}

        Parameters
        ----------

        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta
        >>> Beta.mean(3, 4)
        0.42857142857142855
        """
        return alpha / (alpha + beta)

    def moment(self, m: int, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        m-th (non central) moment of the Beta distribution

        .. math::
            E = \frac{B \left( m + \alpha, \beta \right )}{B
            \left ( \alpha, \beta \right )}

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta
        >>> Beta.moment(2, 3, 4)
        np.float64(0.2142857142857143)
        """
        return np.exp(abetaln(m + alpha, beta) - abetaln(alpha, beta))

    def entropy(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the Beta distribution.

        .. math::
            S = \ln B \left ( \alpha, \beta \right )
            - \left ( \alpha - 1 \right ) \psi \left ( \alpha \right )
            - \left ( \beta - 1 \right ) \psi \left ( \beta \right )
            + \left ( \alpha + \beta - 2 \right ) \psi \left ( \alpha +
            \beta \right )

        Where B is the beta function and psi is the digamma function

        Parameters
        ----------

        alpha : numpy array or scalar
            The first shape parameter for the Beta distribution
        beta : numpy array or scalar
            The second shape parameter for the Beta distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the Beta distribution

        Examples
        --------
        >>> from surpyval import Beta
        >>> Beta.entropy(3, 4)
        np.float64(-0.3443445622221013)
        """
        return (
            abetaln(alpha, beta)
            - (alpha - 1) * digamma(alpha)
            - (beta - 1) * digamma(beta)
            + (alpha + beta - 2) * digamma(alpha + beta)
        )

    def log_df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # At an edge the limit: -inf, the constant or inf as the shape there
        # is above, at or below 1 (0 * log 0 was NaN, #458).
        x = np.asarray(x, dtype=float)
        return (
            _power_log(alpha - 1.0, x)
            + _power_log1m(beta - 1.0, x)
            - _log_beta(alpha, beta)
        )

    def log_ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        return abetaincln(alpha, beta, x)

    def log_sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # The upper tail's own log: log(1 - F) lost R below 1e-16 and was
        # -inf where R underflowed (#458).
        return abetainccln(alpha, beta, x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return self.qf(y, *params)

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        # The inverse of the quantile transform is the CDF; the point must
        # be the *last* argument of betainc, not the first shape (#257).
        return abetainc(*params, y)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x

    def _mom(self, x: npt.NDArray) -> tuple[float, float]:
        """
        MOM: Method of Moments for the beta distribution has an analytic answer
        """
        mean = x.mean()
        var = x.var()
        # Zero (or near zero) variance divides through to a shape pair of
        # order 1e31 rather than an error, and since a failed optimiser
        # falls back to its initial guess (#261) that pair is what gets
        # returned to the caller. A tied sample says nothing about the
        # shapes, so seed the uniform Beta and let the optimiser move if
        # the data can move it.
        if not np.isfinite(var) or var <= np.finfo(float).tiny:
            return 1.0, 1.0
        term1 = (mean * (1 - mean) / var) - 1
        alpha = term1 * mean
        beta = term1 * (1 - mean)
        if not (np.isfinite(alpha) and np.isfinite(beta)) or (
            alpha <= 0 or beta <= 0
        ):
            return 1.0, 1.0

        return alpha, beta

    def _plot_x_bounds(
        self, x: npt.NDArray, params: npt.NDArray
    ) -> tuple[float, float] | None:
        return 0.0, 1.0


Beta: Beta_ = Beta_("Beta")
