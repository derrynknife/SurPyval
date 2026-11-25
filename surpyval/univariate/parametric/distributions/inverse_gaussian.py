from __future__ import annotations

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
from surpyval.utils.warnings import ignoring_warnings

from ._stable import _erfcx, log1mexp, on_support, positive_or_one

_LOG_2PI = float(onp.log(2.0 * onp.pi))
_SQRT_HALF_PI = float(onp.sqrt(onp.pi / 2.0))
_SQRT2 = float(onp.sqrt(2.0))


def _mills(x: Boxable) -> Boxable:
    r"""Mills' ratio :math:`M(x) = (1 - \Phi(x)) / \phi(x) = \sqrt{\pi/2}\,
    \operatorname{erfcx}(x / \sqrt 2)`, for :math:`x \geq 0`."""
    return _SQRT_HALF_PI * _erfcx(x / _SQRT2)


#: Gauss-Legendre nodes and weights on [-1, 1] for the gap of Mills'
#: ratio (``_mills_gap``).
_GL_NODES, _GL_WEIGHTS = (
    onp.asarray(v, dtype=float) for v in onp.polynomial.legendre.leggauss(12)
)
#: Above this t, ``1 - t M(t)`` is its asymptotic series (12 terms: the
#: last is 5e-19 of it at t = 20); below it, the direct form, which loses
#: at most t^2 eps.
_SERIES_T = 20.0
_DOUBLE_FACTORIAL = tuple(
    float(onp.prod(onp.arange(1, 2 * n + 2, 2))) for n in range(12)
)


def _one_minus_t_mills(t: Boxable) -> Boxable:
    r""":math:`1 - t M(t) = -M'(t)`, positive, about :math:`1/t^2` for a
    large :math:`t`."""
    big = t > _SERIES_T
    t_s = np.where(big, t, _SERIES_T)
    w = 1.0 / (t_s * t_s)
    series: Any = 0.0
    for k in reversed(range(12)):
        series = series * w + (-1.0) ** k * _DOUBLE_FACTORIAL[k]
    series = series * w
    t_d = np.where(big, 1.0, t)
    return np.where(big, series, 1.0 - t_d * _mills(t_d))


def _mills_gap(a: Boxable, d: Boxable) -> Boxable:
    r""":math:`M(a) - M(b)` for :math:`a > 0` and :math:`b = a + d`.
    Where :math:`b` is close to :math:`a` (far above the mean) the
    difference of the two cancels, to nothing at all by :math:`x/\mu =
    10^{16}`; there it is
    the integral :math:`\int_a^b (1 - t M(t))\,dt` by twelve-point
    Gauss-Legendre, of a smooth integrand on an interval no longer than
    half its start."""
    near = d < 0.5 * a
    if not np.any(near):
        return _mills(a) - _mills(a + d)
    a_n = np.where(near, a, 1.0)
    half = 0.5 * np.where(near, d, 0.2)
    total: Any = 0.0
    for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
        total = total + weight * _one_minus_t_mills(a_n + half * (1.0 + node))
    quad = half * total
    if np.all(near):
        return quad
    a_f = np.where(near, 1.0, a)
    d_f = np.where(near, 2.0, d)
    return np.where(near, quad, _mills(a_f) - _mills(a_f + d_f))


def _ab(x: Boxable, mu: Boxable, lam: Boxable) -> tuple:
    r""":math:`a = \sqrt{\lambda/x}\,(x/\mu - 1)`, :math:`b =
    \sqrt{\lambda/x}\,(x/\mu + 1)` and their gap :math:`d = b - a = 2
    \sqrt{\lambda/x}`, kept on its own: far above the mean it is a tiny
    fraction of either, and their difference would have lost it. Written
    as :math:`\sqrt{\lambda x}/\mu \mp \sqrt{\lambda/x}`, which does
    not overflow where :math:`x/\mu` would; at :math:`x = \infty`, from a
    stand-in."""
    top = onp.isposinf(onp.asarray(getval(x), dtype=float))
    x_in = np.where(top, 1.0, x) if onp.any(top) else x
    centre = np.sqrt(lam * x_in) / mu
    half = np.sqrt(lam / x_in)
    a, b, d = centre - half, centre + half, 2.0 * half
    if onp.any(top):
        a = np.where(top, np.inf, a)
        b = np.where(top, np.inf, b)
        d = np.where(top, 0.0, d)
    return a, b, d


class InverseGaussian_(OptimisedFitMixin, ParametricFitter):
    r"""
    The inverse Gaussian distribution: the time a Brownian motion with
    a steady drift takes to first reach a fixed level. It is the natural
    lifetime of a unit that fails when accumulated, noisy wear reaches a
    threshold, and a common model for durations in medicine and
    economics (a time to recovery, a spell of unemployment) whose hazard
    rises and then falls to a constant.

    The parameters are the mean ``mu`` and the shape ``lam``
    (:math:`\lambda`), both positive; the variance is
    :math:`\mu^3/\lambda`. With :math:`a = \sqrt{\lambda/x}\,(x/\mu -
    1)` and :math:`b = \sqrt{\lambda/x}\,(x/\mu + 1)`,

    .. math::
        R(x) = \Phi(-a) - e^{2\lambda/\mu}\, \Phi(-b),

    with :math:`\Phi` the standard normal distribution function; the
    parameterisation is that of R's ``statmod`` (``mean``, ``shape``) and
    ``SuppDists`` (``nu``, ``lambda``). As :math:`\lambda/\mu` grows the
    distribution tends to a normal with mean :math:`\mu`.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.
    Exact, untruncated data have a closed-form maximum likelihood
    estimate (:math:`\hat\mu = \bar x`, :math:`1/\hat\lambda = \overline{1/x}
    - 1/\bar x`), which ``fit`` uses.

    Examples
    --------
    >>> from surpyval import InverseGaussian
    >>> model = InverseGaussian.from_params([10, 20])
    >>> model.sf([5, 10, 20]).round(4)
    array([0.7676, 0.3723, 0.085 ])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        mu : numpy array or scalar
            mean of the InverseGaussian distribution
        lam : numpy array or scalar
            shape parameter of the InverseGaussian distribution
        """

    # Both are times (rescale_time): the mean, and the shape, which has
    # the units of the mean (the variance is mu^3 / lam).
    _time_roles = ("time", "time")

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((0, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["mu", "lam"],
            param_map={"mu": 0, "lam": 1},
            plot_x_scale="log",
        )
        # No linearising probability plot: the straight-line axis is the
        # quantile of the inverse Gaussian of mean 1 and shape lam / mu,
        # which needs the answer. ``plot()`` draws on the fitted shape.
        self.supports_mpp = False

    def _closed_form_mle(self, data: SurpyvalData) -> npt.NDArray | None:
        """The MLE of exact, untruncated data: ``mu`` the mean, ``1/lam``
        the mean of ``1/x - 1/mu``. ``None`` (search) for anything else,
        or for data whose times are all equal (no finite ``lam``)."""
        c = onp.asarray(data.c)
        if onp.any(c != 0) or onp.any(onp.isfinite(data.t)):
            return None
        x = onp.asarray(data.x, dtype=float)
        n = onp.asarray(data.n, dtype=float)
        if x.ndim != 1 or onp.any(x <= 0):
            return None
        mu = float(onp.sum(n * x) / onp.sum(n))
        inv = float(onp.sum(n * (1.0 / x - 1.0 / mu)) / onp.sum(n))
        if not inv > 0:
            return None
        return onp.array([mu, 1.0 / inv])

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            return self._offset_seed(data)
        # The lognormal fitted to the data, matched in mean and variance
        from surpyval.univariate.parametric import LogNormal

        with onp.errstate(all="ignore"):
            m, s = (
                float(v)
                for v in LogNormal._fit_from_surpyval_data(
                    data, how="MLE"
                ).params
            )
        mean = onp.exp(m + s * s / 2.0)
        var = onp.expm1(s * s) * mean**2
        if not (onp.isfinite(mean) and onp.isfinite(var) and var > 0):
            x = onp.asarray(data.x, dtype=float)
            x = x[onp.isfinite(x) & (x > 0)]
            mean = float(onp.mean(x)) if x.size else 1.0
            var = float(onp.var(x)) if x.size > 1 else mean**2
            var = var if var > 0 else mean**2
        return onp.array([mean, mean**3 / var])

    def log_df(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        x_pos = positive_or_one(x)
        top = onp.isposinf(onp.asarray(getval(x_pos), dtype=float))
        x_in = np.where(top, 1.0, x_pos) if onp.any(top) else x_pos
        # (x - mu)^2 / x as (x - mu) ((x - mu) / x): the square overflows
        # from x = 1e154
        d = x_in - mu
        inside = (
            0.5 * (np.log(lam) - _LOG_2PI)
            - 1.5 * np.log(x_in)
            - lam / (2.0 * mu**2) * d * (d / x_in)
        )
        if onp.any(top):
            inside = np.where(top, -np.inf, inside)
        return on_support(x, inside, -np.inf)

    def log_ff(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        # F = Phi(a) + e^{2 lam / mu} Phi(-b): two positive terms, added
        # in logs. Above the mean (a > 0), where F is near 1, log F =
        # log(1 - S) from the accurate log S instead: the sum keeps F to
        # an ulp, but its log is then about -S, and an ulp of F is all of
        # it once S is below 1e-16 (log_ff was 1.3e-4 off at S = 1e-300).
        x_pos = positive_or_one(x)
        a, b, d = _ab(x_pos, mu, lam)
        above = a > 0
        if np.any(above):
            a_up = np.where(above, a, 1.0)
            d_up = np.where(above, d, 1.0)
            r = -self._log_sf_above(a_up, d_up)
            from_sf = log1mexp(r, np.log(r))[0]
            if np.all(above):
                return on_support(x, from_sf, -np.inf)
            # half the mean: a point below it, for the direct sum
            x_dn = np.where(above, 0.5 * mu + np.zeros_like(a), x_pos)
            a, b, d = _ab(x_dn, mu, lam)
        direct = np.logaddexp(norm.logcdf(a), 2.0 * lam / mu + norm.logcdf(-b))
        inside = np.where(above, from_sf, direct) if np.any(above) else direct
        return on_support(x, inside, -np.inf)

    def log_sf(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        # Above the mean (a > 0): S = phi(a) (M(a) - M(b)), M Mills'
        # ratio, since e^{2 lam / mu} phi(b) = phi(a); no term underflows
        # or overflows (e^{2 lam / mu} does past lam / mu = 355). Below
        # it, S = 1 - F from log F.
        x_pos = positive_or_one(x)
        a, b, d = _ab(x_pos, mu, lam)
        above = a > 0
        out: Any
        if np.all(above):
            out = self._log_sf_above(a, d)
        elif not np.any(above):
            out = self._log_sf_below(x_pos, mu, lam)
        else:
            a_up = np.where(above, a, 1.0)
            d_up = np.where(above, d, 1.0)
            # half the mean: a point below it, where a < 0
            x_dn = np.where(above, 0.5 * mu + np.zeros_like(a), x_pos)
            out = np.where(
                above,
                self._log_sf_above(a_up, d_up),
                self._log_sf_below(x_dn, mu, lam),
            )
        return on_support(x, out, 0.0)

    @staticmethod
    def _log_sf_above(a: Boxable, d: Boxable) -> Boxable:
        return norm.logpdf(a) + np.log(_mills_gap(a, d))

    def _log_sf_below(self, x: Boxable, mu: Boxable, lam: Boxable) -> Boxable:
        a, b, d = _ab(x, mu, lam)
        log_f = np.logaddexp(norm.logcdf(a), 2.0 * lam / mu + norm.logcdf(-b))
        r = -log_f
        return log1mexp(r, np.log(r))[0]

    def sf(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the InverseGaussian
        distribution:

        .. math::
            R(x) = \Phi(-a) - e^{2\lambda/\mu}\, \Phi(-b)

        computed as :math:`\phi(a)\,(M(a) - M(b))` above the mean, with
        :math:`M` Mills' ratio, so that neither term overflows nor the
        difference cancels.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            mean of the InverseGaussian distribution
        lam : numpy array or scalar
            shape parameter of the InverseGaussian distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import InverseGaussian
        >>> x = np.array([1, 5, 10, 20])
        >>> InverseGaussian.sf(x, 10, 20)
        array([0.9999478 , 0.76764281, 0.37230216, 0.08495332])
        """
        return np.exp(self.log_sf(x, mu, lam))

    def ff(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the InverseGaussian
        distribution:

        .. math::
            F(x) = \Phi(a) + e^{2\lambda/\mu}\, \Phi(-b)

        (two positive terms, added in logs).

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            mean of the InverseGaussian distribution
        lam : numpy array or scalar
            shape parameter of the InverseGaussian distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import InverseGaussian
        >>> x = np.array([1, 5, 10, 20])
        >>> InverseGaussian.ff(x, 10, 20)
        array([5.22014676e-05, 2.32357189e-01, 6.27697838e-01, 9.15046681e-01])
        """
        return np.exp(self.log_ff(x, mu, lam))

    def df(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""

        Density function for the InverseGaussian distribution:

        .. math::
            f(x) = \sqrt{\frac{\lambda}{2\pi x^3}} \exp\left(
            -\frac{\lambda (x - \mu)^2}{2\mu^2 x}\right)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            mean of the InverseGaussian distribution
        lam : numpy array or scalar
            shape parameter of the InverseGaussian distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import InverseGaussian
        >>> x = np.array([1, 5, 10, 20])
        >>> InverseGaussian.df(x, 10, 20)
        array([0.00054155, 0.09678829, 0.05641896, 0.01209854])
        """
        return np.exp(self.log_df(x, mu, lam))

    def hf(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the InverseGaussian distribution,
        :math:`h(x) = f(x)/R(x)`. It rises and then falls to
        :math:`\lambda / (2\mu^2)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            mean of the InverseGaussian distribution
        lam : numpy array or scalar
            shape parameter of the InverseGaussian distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import InverseGaussian
        >>> x = np.array([1, 5, 10, 20])
        >>> InverseGaussian.hf(x, 10, 20)
        array([0.00054158, 0.12608506, 0.15154078, 0.14241393])
        """
        # Above the mean the density is phi(a) sqrt(lam / x^3) and the
        # survival function phi(a) (M(a) - M(b)), so the hazard is
        # sqrt(lam / x^3) / (M(a) - M(b)), with phi(a), which underflows
        # far out, cancelled; it tends to lam / (2 mu^2).
        x_pos = positive_or_one(x)
        top = onp.isposinf(onp.asarray(getval(x_pos), dtype=float))
        x_in = np.where(top, 2.0 * mu + np.zeros_like(x_pos), x_pos)
        a, b, d = _ab(x_in, mu, lam)
        above = a > 0
        a_up = np.where(above, a, 1.0)
        d_up = np.where(above, d, 1.0)
        x_up = np.where(above, x_in, 1.0)
        up = (
            0.5 * np.log(lam)
            - 1.5 * np.log(x_up)
            - np.log(_mills_gap(a_up, d_up))
        )
        x_dn = np.where(above, 0.5 * mu + 0.0 * x_in, x_in)
        down = self.log_df(x_dn, mu, lam) - self._log_sf_below(x_dn, mu, lam)
        inside = np.exp(np.where(above, up, down))
        if onp.any(top):
            inside = np.where(top, lam / (2.0 * mu**2), inside)
        return on_support(x, inside, 0.0)

    def Hf(self, x: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the InverseGaussian distribution,
        :math:`H(x) = -\ln R(x)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            mean of the InverseGaussian distribution
        lam : numpy array or scalar
            shape parameter of the InverseGaussian distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import InverseGaussian
        >>> x = np.array([1, 5, 10, 20])
        >>> InverseGaussian.Hf(x, 10, 20)
        array([5.22028301e-05, 2.64430744e-01, 9.88049491e-01, 2.46565337e+00])
        """
        return -self.log_sf(x, mu, lam)

    def _solve(self, p: Any, s: Any, mu: float, lam: float) -> Any:
        """The time with ``F = p`` (``S = s``): scipy's quantile for a
        start, then Newton's method in ``ln x`` on the log of the smaller
        tail."""
        from scipy.stats import invgauss

        p = onp.asarray(p, dtype=float)
        s = onp.asarray(s, dtype=float)
        lower = p <= s
        target = onp.log(onp.where(lower, p, s))
        dist = invgauss(mu / lam, scale=lam)
        with onp.errstate(all="ignore"), ignoring_warnings():
            # Far in a tail, its asymptote: S ~ e^{-lam x / (2 mu^2)}
            # above, F ~ Phi(-sqrt(lam / x)) below; elsewhere scipy's.
            far = onp.where(lower, p, s) < 1e-8
            asymptote = onp.where(
                lower,
                lam / norm.ppf(onp.where(far, p, 0.5)) ** 2,
                -2.0 * mu**2 * onp.log(onp.where(far, s, 0.5)) / lam,
            )
            x = onp.where(
                far,
                asymptote,
                onp.where(
                    lower,
                    dist.ppf(onp.where(far, 0.5, p)),
                    dist.isf(onp.where(far, 0.5, s)),
                ),
            )
            x = onp.where(onp.isfinite(x) & (x > 0), x, mu)
            y = self._newton(onp.log(x), lower, target, mu, lam, 60)
            # Where Newton's method did not reach the target (a start far
            # from it, on a flat stretch of the tail), bisection on ln x,
            # which needs only that the tail is monotone.
            current = onp.where(
                lower,
                self.log_ff(onp.exp(y), mu, lam),
                self.log_sf(onp.exp(y), mu, lam),
            )
            missed = ~(
                onp.abs(current - target)
                <= 1e-12 * onp.maximum(1.0, onp.abs(target))
            )
            if onp.any(missed):
                lo = onp.full(y.shape, onp.log(mu) - 700.0)
                hi = onp.full(y.shape, onp.log(mu) + 700.0)
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    xm = onp.exp(mid)
                    # ln F rises with x, ln S falls
                    above = onp.where(
                        lower,
                        self.log_ff(xm, mu, lam) > target,
                        self.log_sf(xm, mu, lam) < target,
                    )
                    hi = onp.where(above, mid, hi)
                    lo = onp.where(above, lo, mid)
                y = onp.where(missed, 0.5 * (lo + hi), y)
                y = self._newton(y, lower, target, mu, lam, 4)
            out = onp.exp(y)
        out = onp.where(p <= 0, 0.0, onp.where(s <= 0, onp.inf, out))
        return onp.where(onp.isnan(p) | onp.isnan(s), onp.nan, out)

    def _newton(
        self,
        y: Any,
        lower: Any,
        target: Any,
        mu: float,
        lam: float,
        steps: int,
    ) -> Any:
        """Newton's method in ``y = ln x`` on the log of the smaller tail
        (``ln F`` where ``lower``, else ``ln S``), to ``target``."""
        for _ in range(steps):
            x = onp.exp(y)
            current = onp.where(
                lower, self.log_ff(x, mu, lam), self.log_sf(x, mu, lam)
            )
            # d ln F / d ln x = x f / F, d ln S / d ln x = -x f / S
            slope = onp.exp(y + self.log_df(x, mu, lam) - current)
            slope = onp.where(lower, slope, -slope)
            step = (current - target) / slope
            step = onp.clip(onp.where(onp.isfinite(step), step, 0.0), -2, 2)
            y = y - step
            if onp.all(onp.abs(step) <= 4e-16 * onp.maximum(1, onp.abs(y))):
                break
        return y

    def qf(self, u: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""

        Quantile function for the InverseGaussian distribution: it has no
        closed form, so it is solved, by Newton's method in :math:`\ln x`
        on the log of the smaller tail from scipy's quantile (or a tail
        asymptote), with bisection where that does not converge.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import InverseGaussian
        >>> InverseGaussian.qf(np.array([.1, .5, .9]), 10, 20)
        array([ 3.56985117,  8.04339041, 18.85995336])
        """
        u = onp.asarray(u, dtype=float)
        return self._solve(u, 1.0 - u, float(getval(mu)), float(getval(lam)))

    def isf(self, s: Numeric, mu: Boxable, lam: Boxable) -> Boxable:
        r"""The inverse survival function: the time at which the survival
        function is ``s``, from ``s`` itself so that it holds deep in the
        upper tail.

        Examples
        --------
        >>> from surpyval import InverseGaussian
        >>> InverseGaussian.isf([0.5, 1e-300], 10, 20)
        array([   8.04339041, 6824.11058918])
        """
        s = onp.asarray(s, dtype=float)
        return self._solve(1.0 - s, s, float(getval(mu)), float(getval(lam)))

    def mean(self, mu: Boxable, lam: Boxable) -> Boxable:
        r"""The mean of the InverseGaussian distribution, :math:`\mu`.

        Examples
        --------
        >>> from surpyval import InverseGaussian
        >>> InverseGaussian.mean(10, 20)
        10
        """
        return mu

    def moment(self, m: int, mu: Boxable, lam: Boxable) -> Boxable:
        r"""
        The ``m``-th raw moment,

        .. math::
            E[X^m] = \mu^m \sum_{k=0}^{m-1} \frac{(m - 1 + k)!}{k!\,(m - 1
            - k)!} \left(\frac{\mu}{2\lambda}\right)^k.
        """
        from math import factorial

        if m == 0:
            return 1.0 + 0.0 * mu
        total: Any = 0.0
        for k in range(m):
            coef = factorial(m - 1 + k) / (factorial(k) * factorial(m - 1 - k))
            total = total + coef * (mu / (2.0 * lam)) ** k
        return mu**m * total

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        # ln of the quantile of the inverse Gaussian of mean 1 and shape
        # lam / mu, on the fitted shape: a line of slope 1 in ln x
        mu, lam = (float(getval(p)) for p in params[-2:])
        y = onp.asarray(y, dtype=float)
        out = onp.full_like(y, onp.nan)
        inside = (y > 0) & (y < 1)
        out[inside] = onp.log(
            self._solve(y[inside], 1.0 - y[inside], 1.0, lam / mu)
        )
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mu, lam = (float(getval(p)) for p in params[-2:])
        return self.ff(onp.exp(onp.asarray(y, dtype=float)), 1.0, lam / mu)


InverseGaussian: InverseGaussian_ = InverseGaussian_("InverseGaussian")
