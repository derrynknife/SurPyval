from __future__ import annotations

from math import comb
from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.tracer import getval

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils import normal as norm
from surpyval.utils.surpyval_data import SurpyvalData

from ._quantile import broadcast_values
from ._stable import on_support, positive_or_one

_HALF_LOG_2PI = float(0.5 * onp.log(2.0 * onp.pi))


def _standard_quantile(alpha: Any, z: Any) -> Any:
    r"""The quantile of the Birnbaum-Saunders of scale 1 at the standard
    normal quantile ``z``: :math:`(w + \sqrt{w^2 + 1})^2` with :math:`w =
    \alpha z / 2`, written as :math:`1 / (\sqrt{w^2 + 1} - w)^2` for a
    negative :math:`w`, where the sum cancels (``hypot`` so that
    :math:`w^2` does not overflow)."""
    w = 0.5 * alpha * z
    root = onp.hypot(w, 1.0)
    with onp.errstate(divide="ignore", invalid="ignore"):
        up = (w + root) ** 2
        down = 1.0 / (root - w) ** 2
    out = onp.where(w >= 0, up, down)
    # z = +-inf: the ends of the support
    out = onp.where(onp.isposinf(z), onp.inf, out)
    return onp.where(onp.isneginf(z), 0.0, out)


def _double_factorial(n: int) -> int:
    """:math:`n!! = n (n - 2) (n - 4) \\cdots`, 1 for ``n <= 0``."""
    out = 1
    for i in range(n, 0, -2):
        out *= i
    return out


def _chebyshev_even(m: int) -> list[int]:
    r"""The coefficients :math:`c_j` of :math:`T_{2m}(y) = \sum_j c_j
    y^{2j}`, the Chebyshev polynomial of degree :math:`2m` (even, with
    integer coefficients), from the recurrence :math:`T_{n+1} = 2 y T_n -
    T_{n-1}`."""
    prev, cur = [1], [0, 1]
    for _ in range(2 * m - 1):
        nxt = [0] * (len(cur) + 1)
        for i, v in enumerate(cur):
            nxt[i + 1] += 2 * v
        for i, v in enumerate(prev):
            nxt[i] -= v
        prev, cur = cur, nxt
    out = cur if m > 0 else [1]
    return [out[2 * j] for j in range(len(out) // 2 + 1)]


class BirnbaumSaunders_(OptimisedFitMixin, ParametricFitter):
    r"""
    The Birnbaum-Saunders distribution (also called the fatigue-life
    distribution): a positive, right-skewed lifetime whose logarithm is
    close to normal for a small shape and more spread out for a large
    one. It was derived in 1969 as the time for accumulated damage to
    reach a critical level; as a statistical model it is a two-parameter
    alternative to the lognormal and the inverse Gaussian for skewed
    positive data, in reliability and in medicine alike.

    The parameters are the shape ``alpha`` (:math:`\alpha`) and the scale
    ``beta`` (:math:`\beta`), both positive; ``beta`` is the median. With

    .. math::
        \xi(x) = \frac{1}{\alpha}\left(\sqrt{\frac{x}{\beta}} -
        \sqrt{\frac{\beta}{x}}\right) = \frac{x - \beta}{\alpha
        \sqrt{x \beta}},

    the distribution function is :math:`F(x) = \Phi(\xi(x))`, with
    :math:`\Phi` the standard normal distribution function, so

    .. math::
        R(x) = \Phi(-\xi(x)).

    This is ``scipy.stats.fatiguelife(alpha, scale=beta)`` and R's
    ``VGAM::pbisa(x, scale = beta, shape = alpha)``. The mean is
    :math:`\beta(1 + \alpha^2/2)` and the variance :math:`(\alpha
    \beta)^2 (1 + 5\alpha^2/4)`. As :math:`\alpha` falls to 0 the
    distribution concentrates on :math:`\beta`, and :math:`\ln x` tends
    to a normal of mean :math:`\ln\beta` and standard deviation
    :math:`\alpha`.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import BirnbaumSaunders
    >>> model = BirnbaumSaunders.from_params([0.5, 10])
    >>> model.sf([5, 10, 20]).round(4)
    array([0.9214, 0.5   , 0.0786])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution
        """

    # How the parameters move with the time (rescale_time): alpha has no
    # unit; beta, the median, is a time.
    _time_roles = ("shape", "time")

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["alpha", "beta"],
            param_map={"alpha": 0, "beta": 1},
            plot_x_scale="log",
        )
        # No linearising probability plot: x = beta + alpha sqrt(beta) z
        # sqrt(x) is a line in z sqrt(x), which mixes the time with the
        # probability, so no axis of the time alone and one of the
        # probability alone makes it straight. ``plot()`` draws on the
        # fitted shape.
        self.supports_mpp = False

    def _refuse_mpp_without_plot(self) -> None:
        raise ValueError(
            "BirnbaumSaunders distribution does not work with probability "
            "plot fitting: its straight line mixes the time with the "
            "probability, so no plot axis makes it straight without alpha; "
            "use how='MLE', 'MPS', 'MSE' or 'MOM' instead"
        )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            return self._offset_seed(data)
        # The lognormal fitted to the data: as alpha falls to 0, ln x is
        # normal with mean ln beta and standard deviation alpha.
        from surpyval.univariate.parametric import LogNormal

        with onp.errstate(all="ignore"):
            mu, sigma = (
                float(v)
                for v in LogNormal._fit_from_surpyval_data(
                    data, how="MLE"
                ).params
            )
        if not (onp.isfinite(mu) and onp.isfinite(sigma) and sigma > 0):
            x = onp.asarray(data.x, dtype=float)
            x = x[onp.isfinite(x) & (x > 0)]
            mu = float(onp.median(onp.log(x))) if x.size else 0.0
            sigma = float(onp.std(onp.log(x))) if x.size > 1 else 1.0
            sigma = sigma if sigma > 0 else 1.0
        return onp.array([sigma, onp.exp(mu)])

    @staticmethod
    def _xi(x: Boxable, alpha: Boxable, beta: Boxable) -> Boxable:
        r""":math:`\xi = (x - \beta) / (\alpha \sqrt{x} \sqrt{\beta})`,
        without the cancellation of :math:`\sqrt{x/\beta} -
        \sqrt{\beta/x}` near the median; :math:`\pm\infty` at the ends of
        the support (``x`` positive, possibly infinite)."""
        top = onp.isposinf(onp.asarray(getval(x), dtype=float))
        x_in = np.where(top, 1.0, x) if onp.any(top) else x
        xi = (x_in - beta) / (alpha * np.sqrt(x_in) * np.sqrt(beta))
        if onp.any(top):
            xi = np.where(top, np.inf, xi)
        return xi

    def log_df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        # f = phi(xi) (x + beta) / (2 alpha sqrt(beta) x^{3/2})
        x_pos = positive_or_one(x)
        top = onp.isposinf(onp.asarray(getval(x_pos), dtype=float))
        x_in = np.where(top, 1.0, x_pos) if onp.any(top) else x_pos
        xi = self._xi(x_in, alpha, beta)
        inside = (
            -0.5 * xi * xi
            - _HALF_LOG_2PI
            + np.log(x_in + beta)
            - np.log(2.0 * alpha)
            - 0.5 * np.log(beta)
            - 1.5 * np.log(x_in)
        )
        if onp.any(top):
            inside = np.where(top, -np.inf, inside)
        return on_support(x, inside, -np.inf)

    def log_sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        x_pos = positive_or_one(x)
        return on_support(x, norm.logsf(self._xi(x_pos, alpha, beta)), 0.0)

    def log_ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        x_pos = positive_or_one(x)
        inside = norm.logcdf(self._xi(x_pos, alpha, beta))
        return on_support(x, inside, -np.inf)

    def sf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the BirnbaumSaunders
        distribution:

        .. math::
            R(x) = \Phi\left(-\frac{x - \beta}{\alpha \sqrt{x \beta}}
            \right)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import BirnbaumSaunders
        >>> x = np.array([1, 5, 10, 20])
        >>> BirnbaumSaunders.sf(x, 0.5, 10)
        array([0.99999999, 0.9213504 , 0.5       , 0.0786496 ])
        """
        return np.exp(self.log_sf(x, alpha, beta))

    def ff(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the BirnbaumSaunders
        distribution:

        .. math::
            F(x) = \Phi\left(\frac{x - \beta}{\alpha \sqrt{x \beta}}\right)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import BirnbaumSaunders
        >>> x = np.array([1, 5, 10, 20])
        >>> BirnbaumSaunders.ff(x, 0.5, 10)
        array([6.27432375e-09, 7.86496035e-02, 5.00000000e-01, 9.21350396e-01])
        """
        return np.exp(self.log_ff(x, alpha, beta))

    def df(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Density function for the BirnbaumSaunders distribution:

        .. math::
            f(x) = \frac{x + \beta}{2 \alpha \sqrt{\beta}\, x^{3/2}}
            \phi\left(\frac{x - \beta}{\alpha \sqrt{x \beta}}\right)

        with :math:`\phi` the standard normal density.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import BirnbaumSaunders
        >>> x = np.array([1, 5, 10, 20])
        >>> BirnbaumSaunders.df(x, 0.5, 10)
        array([1.27859248e-07, 6.22661246e-02, 7.97884561e-02, 1.55665312e-02])
        """
        return np.exp(self.log_df(x, alpha, beta))

    def hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the BirnbaumSaunders distribution,
        :math:`h(x) = f(x)/R(x)`. It rises to a peak and then falls to
        :math:`1 / (2 \alpha^2 \beta)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import BirnbaumSaunders
        >>> x = np.array([1, 5, 10, 20])
        >>> BirnbaumSaunders.hf(x, 0.5, 10)
        array([1.27859248e-07, 6.75813728e-02, 1.59576912e-01, 1.97922564e-01])
        """
        x_pos = positive_or_one(x)
        top = onp.isposinf(onp.asarray(getval(x_pos), dtype=float))
        x_in = np.where(top, 1.0, x_pos) if onp.any(top) else x_pos
        # in logs: far above the median the density and the survival
        # function both underflow, their ratio does not
        inside = np.exp(
            self.log_df(x_in, alpha, beta) - self.log_sf(x_in, alpha, beta)
        )
        if onp.any(top):
            inside = np.where(top, 1.0 / (2.0 * alpha**2 * beta), inside)
        return on_support(x, inside, 0.0)

    def Hf(self, x: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the BirnbaumSaunders distribution,
        :math:`H(x) = -\ln R(x)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import BirnbaumSaunders
        >>> x = np.array([1, 5, 10, 20])
        >>> BirnbaumSaunders.Hf(x, 0.5, 10)
        array([6.27432377e-09, 8.19148629e-02, 6.93147181e-01, 2.54275269e+00])
        """
        return -self.log_sf(x, alpha, beta)

    def qf(self, u: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""

        Quantile function for the BirnbaumSaunders distribution, in
        closed form:

        .. math::
            q(u) = \frac{\beta}{4}\left(\alpha z + \sqrt{\alpha^2 z^2 +
            4}\right)^2, \qquad z = \Phi^{-1}(u),

        computed without the cancellation of the sum for :math:`u <
        1/2`.

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            shape parameter for the BirnbaumSaunders distribution
        beta : numpy array or scalar
            scale parameter (the median) for the BirnbaumSaunders
            distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the BirnbaumSaunders distribution at each
            value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import BirnbaumSaunders
        >>> u = np.array([.1, .5, .9])
        >>> BirnbaumSaunders.qf(u, 0.5, 10)
        array([ 5.3243695 , 10.        , 18.78156654])
        """
        u, a, b = broadcast_values(u, alpha, beta)
        with onp.errstate(invalid="ignore"):
            z = norm.ppf(u)
        return b * _standard_quantile(a, z)

    def isf(self, s: Numeric, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""The inverse survival function: the time at which the survival
        function is ``s``, from ``s`` itself (:math:`z = -\Phi^{-1}(s)`)
        so that it holds deep in the upper tail.

        Examples
        --------
        >>> from surpyval import BirnbaumSaunders
        >>> BirnbaumSaunders.isf([0.5, 1e-300], 0.5, 10)
        array([  10.        , 3451.18938502])
        """
        s, a, b = broadcast_values(s, alpha, beta)
        with onp.errstate(invalid="ignore"):
            z = norm.isf(s)
        return b * _standard_quantile(a, z)

    def mean(self, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""The mean of the BirnbaumSaunders distribution,
        :math:`\beta(1 + \alpha^2/2)`.

        Examples
        --------
        >>> from surpyval import BirnbaumSaunders
        >>> BirnbaumSaunders.mean(0.5, 10)
        11.25
        """
        return beta * (1.0 + 0.5 * alpha * alpha)

    def moment(self, m: int, alpha: Boxable, beta: Boxable) -> Boxable:
        r"""
        The ``m``-th raw moment, in closed form. With :math:`w = \alpha
        Z / 2`, :math:`Z` standard normal, :math:`X = \beta e^{2
        \operatorname{asinh} w}`, so :math:`E[X^m] = \beta^m E[\cosh(2m
        \operatorname{asinh} w)]` (the odd part has mean 0), which is the
        Chebyshev polynomial :math:`T_{2m}` of :math:`\sqrt{1 + w^2}`, a
        polynomial in :math:`w^2`, whose moments are :math:`E[w^{2k}] =
        (\alpha/2)^{2k} (2k - 1)!!`. The variance is :math:`(\alpha
        \beta)^2 (1 + 5\alpha^2/4)`.

        Examples
        --------
        >>> from surpyval import BirnbaumSaunders
        >>> float(BirnbaumSaunders.moment(2, 0.5, 10))
        159.375
        """
        if m == 0:
            return 1.0 + 0.0 * beta
        # T_{2m}(y) = sum_j c_j (1 + w^2)^j = sum_k d_k w^{2k}
        c = _chebyshev_even(m)
        d = [0] * len(c)
        for j, cj in enumerate(c):
            for k in range(j + 1):
                d[k] += cj * comb(j, k)
        # E[w^{2k}] = (alpha / 2)^{2k} (2k - 1)!!
        # so E[X^m] / beta^m = sum_k d_k (2k - 1)!! q^k, q = alpha^2 / 4,
        # by Horner's rule (exact integer coefficients)
        q = 0.25 * alpha * alpha
        total: Any = 0.0
        for k in reversed(range(len(d))):
            total = total * q + d[k] * _double_factorial(2 * k - 1)
        return beta**m * total

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        # ln of the quantile of the Birnbaum-Saunders of scale 1 on the
        # fitted shape: a line of slope 1 in ln x, through ln beta
        alpha = float(getval(params[-2]))
        y = onp.asarray(y, dtype=float)
        out = onp.full_like(y, onp.nan)
        inside = (y > 0) & (y < 1)
        out[inside] = onp.log(_standard_quantile(alpha, norm.ppf(y[inside])))
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        alpha = float(getval(params[-2]))
        return self.ff(onp.exp(onp.asarray(y, dtype=float)), alpha, 1.0)


BirnbaumSaunders: BirnbaumSaunders_ = BirnbaumSaunders_("BirnbaumSaunders")
