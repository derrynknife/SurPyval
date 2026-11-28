from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.scipy.special import gammaln as agammaln
from autograd.tracer import getval
from scipy.special import gammainccinv, gammaincinv, gammaln

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils import normal as norm
from surpyval.utils.autograd_gamma_compat import gammainccln as agammainccln
from surpyval.utils.autograd_gamma_compat import gammaincln as agammaincln
from surpyval.utils.surpyval_data import SurpyvalData

from . import _temme
from ._quantile import broadcast_values
from ._stable import log1mexp, on_support, positive_or_one, power_at_zero

_LOG_SQRT_2PI = float(0.5 * onp.log(2.0 * onp.pi))
#: Below this ln u the gamma variable u underflows (e^-700 ~ 1e-304).
_LOG_U_TINY = -700.0

#: Stirling's series of ``ln Gamma(q) - (q - 1/2) ln q + q - ln sqrt(2 pi)``
#: in ``z = 1 / q``: ``B_2n / (2n (2n - 1)) z^(2n - 1)``, to the term below
#: double precision at ``q = 10`` (the next is 3e-17 there).
_STIRLING = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
)
#: The series is used for ``z = 1/q`` up to this (``q >= 10``).
_STIRLING_Z = 0.1

#: Below this ``|r|`` the series of ``((1 + r) log(1 + r) - r) / r^2``.
_PHI_SMALL = 0.1
_PHI_SERIES = tuple((-1.0) ** n / ((n + 1.0) * (n + 2.0)) for n in range(16))


def _stirlerr(z: Boxable) -> Boxable:
    r"""The remainder of Stirling's formula, :math:`\ln\Gamma(q) - (q -
    \tfrac12)\ln q + q - \ln\sqrt{2\pi}`, at :math:`q = 1/z`; 0 at
    :math:`z = 0`. A series in :math:`z` for :math:`q \ge 10`, where the
    direct form cancels, so it is smooth through :math:`Q^2 = z = 0`."""
    small = z <= _STIRLING_Z
    zs = np.where(small, z, 0.0)
    z2 = zs * zs
    series = _STIRLING[-1]
    for c in reversed(_STIRLING[:-1]):
        series = series * z2 + c
    series = series * zs
    if np.all(small):
        return series
    q = 1.0 / np.where(small, 1.0, z)
    direct = agammaln(q) - (q - 0.5) * np.log(q) + q - _LOG_SQRT_2PI
    return np.where(small, series, direct)


def _phi(r: Boxable) -> Boxable:
    r""":math:`((1 + r)\ln(1 + r) - r) / r^2`, 1/2 at :math:`r = 0`, for
    :math:`r > -1`."""
    small = np.abs(r) < _PHI_SMALL
    rs = np.where(small, r, 0.0)
    series = _PHI_SERIES[-1]
    for c in reversed(_PHI_SERIES[:-1]):
        series = series * rs + c
    if np.all(small):
        return series
    rl = np.where(small, 1.0, r)
    direct = ((1.0 + rl) * np.log1p(rl) - rl) / rl**2
    return np.where(small, series, direct)


def _log_g(w: Boxable, Q: Boxable) -> Boxable:
    r"""The log density of the standardised log time :math:`w`,

    .. math::
        \ln g(w) = \ln|Q| + q\ln q - \ln\Gamma(q) + q(Qw - e^{Qw}),
        \quad q = Q^{-2},

    written as :math:`-\ln\sqrt{2\pi} - \operatorname{stirlerr}(q) - w^2
    h(Qw)`: the terms of size :math:`q \ln q` cancel exactly, so it is
    finite and smooth through :math:`Q = 0`, where it is the standard
    normal's."""
    return -_LOG_SQRT_2PI - _stirlerr(Q * Q) - (w * w) * _temme.h(Q * w)


def _log_tails(w: Boxable, Q: Boxable) -> tuple[Boxable, Boxable]:
    r"""``(ln S(w), ln F(w))`` of the standardised log time.

    Near the lognormal (:math:`|Q| < 0.1`, :math:`|\eta| \le 1`) from
    Temme's expansion (``_temme``); elsewhere from the regularised
    incomplete gamma functions at shape :math:`q = Q^{-2}` and :math:`u =
    q e^{Qw}`: :math:`S = Q(q, u)` for :math:`Q > 0` and :math:`P(q, u)`
    for :math:`Q < 0`. Each branch sees only points it is exact on.
    """
    w = w + np.zeros_like(Q)
    Q = Q + np.zeros_like(w)
    w_v = onp.asarray(getval(w), dtype=float)
    Q_v = onp.asarray(getval(Q), dtype=float)
    with onp.errstate(all="ignore"):
        temme = (onp.abs(Q_v) < _temme.Q_MAX) & (
            onp.abs(_temme.eta_of(w_v, Q_v)) <= _temme.ETA_MAX
        )
    temme = temme | (Q_v == 0)
    out_s: Any = None
    out_f: Any = None
    if onp.any(temme):
        w_t = np.where(temme, w, 0.0) if not onp.all(temme) else w
        Q_t = np.where(temme, Q, 0.0) if not onp.all(temme) else Q
        small, upper = _temme.small_tail(w_t, Q_t)
        r = -small
        big, _ = log1mexp(r, np.log(np.where(r > 0, r, 1.0)) + 0.0)
        big = np.where(r > 0, big, -np.inf)
        log_s = np.where(upper, small, big)
        log_f = np.where(upper, big, small)
        if onp.all(temme):
            return log_s, log_f
        out_s, out_f = log_s, log_f
    # the incomplete gamma functions
    w_c = np.where(temme, 0.0, w) if out_s is not None else w
    Q_c = np.where(temme, 0.5, Q) if out_s is not None else Q
    q = 1.0 / (Q_c * Q_c)
    log_u = np.log(q) + Q_c * w_c
    # Where u underflows the gamma variable's lower tail is its asymptote
    # P(q, u) = u^q / Gamma(q + 1) (1 + O(u)), in logs, and its upper
    # tail is 1: P(q, 0) is 0 and its log -inf, far short of the tail.
    tiny = onp.asarray(getval(log_u), dtype=float) < _LOG_U_TINY
    if onp.any(tiny):
        log_u_in = np.where(tiny, 0.0, log_u)
        log_u_tiny = np.where(tiny, log_u, _LOG_U_TINY)
    else:
        log_u_in = log_u
    with onp.errstate(over="ignore"):
        u = np.exp(log_u_in)
    upper_c, lower_c = agammainccln(q, u), agammaincln(q, u)
    if onp.any(tiny):
        lower_t = q * log_u_tiny - agammaln(q + 1.0)
        lower_c = np.where(tiny, lower_t, lower_c)
        # the upper tail is 1 - P, in logs: about -P, which is not 0 for
        # a small shape (P = 1e-8 at u = 1e-800 for q = 0.01, where 0
        # made sf 1 and log_sf 0 in the lower tail at Q = 10)
        upper_c = np.where(
            tiny, log1mexp(-lower_t, np.log(-lower_t))[0], upper_c
        )
    positive = Q_v > 0
    log_s_c = np.where(positive, upper_c, lower_c)
    log_f_c = np.where(positive, lower_c, upper_c)
    if out_s is None:
        return log_s_c, log_f_c
    return (
        np.where(temme, out_s, log_s_c),
        np.where(temme, out_f, log_f_c),
    )


#: The series of the reverse hazard ``gamma(q, u) e^u / u^q`` (below)
#: is summed to this many terms, for ``u`` below ``_SERIES_U``.
_SERIES_TERMS = 60
_SERIES_U = 0.5


def _log_hazard(w: Boxable, Q: Boxable) -> Boxable:
    r"""The log hazard of the standardised log time, :math:`\ln g(w) -
    \ln S(w)`, which in the far upper tail is a difference of two logs
    that both underflow to -inf. There it is taken from the gamma
    variable :math:`u = q e^{Qw}` directly:

    - :math:`Q > 0`, :math:`u > q + 1`: :math:`h = Q\,u\,r(q, u)`, with
      :math:`r` the standard gamma's hazard (``Gamma``'s continued
      fraction), and :math:`\ln u = \ln q + Qw` even where :math:`u`
      overflows (:math:`r` is 1 there);
    - :math:`Q < 0`, :math:`u < 1/2`: :math:`h = |Q| q / \sum_{n \ge 0}
      u^n / ((q + 1) \cdots (q + n))`, from the series of the lower
      incomplete gamma.

    Elsewhere (and near the lognormal, where Temme's tails do not
    underflow) the difference of logs.
    """
    from .gamma import _hazard_over_rate

    w = w + np.zeros_like(Q)
    Q = Q + np.zeros_like(w)
    w_v = onp.asarray(getval(w), dtype=float)
    Q_v = onp.asarray(getval(Q), dtype=float)
    with onp.errstate(all="ignore"):
        temme = (onp.abs(Q_v) < _temme.Q_MAX) & (
            onp.abs(_temme.eta_of(w_v, Q_v)) <= _temme.ETA_MAX
        )
        q_v = 1.0 / (Q_v * Q_v)
        log_u_v = onp.log(q_v) + Q_v * w_v
        upper = ~temme & (Q_v > 0) & (log_u_v > onp.log(q_v + 1.0))
        lower = ~temme & (Q_v < 0) & (log_u_v < onp.log(_SERIES_U))
    plain = ~(upper | lower)
    out: Any = None
    if onp.any(plain) or plain.size == 0:
        w_p = np.where(plain, w, 0.0)
        Q_p = np.where(plain, Q, 0.5)
        out = _log_g(w_p, Q_p) - _log_tails(w_p, Q_p)[0]
        if onp.all(plain):
            return out
    if onp.any(upper):
        Q_u = np.where(upper, Q, 1.0)
        q_u = 1.0 / (Q_u * Q_u)
        log_u = np.log(q_u) + Q_u * np.where(upper, w, 3.0)
        u = np.exp(np.minimum(log_u, 690.0))
        tail = np.log(Q_u) + log_u + np.log(_hazard_over_rate(q_u, u))
        out = tail if out is None else np.where(upper, tail, out)
    if onp.any(lower):
        Q_l = np.where(lower, Q, -1.0)
        q_l = 1.0 / (Q_l * Q_l)
        u = np.exp(np.log(q_l) + Q_l * np.where(lower, w, 2.0))
        total: Any = 1.0
        term: Any = 1.0
        for k in range(1, _SERIES_TERMS):
            term = term * u / (q_l + k)
            total = total + term
        tail = np.log(-Q_l * q_l) - np.log(total)
        out = tail if out is None else np.where(lower, tail, out)
    return out


def _std_quantile(p: Any, s: Any, Q: Any) -> npt.NDArray:
    r"""The standardised log time :math:`w` at which :math:`F = p` (and
    :math:`S = s = 1 - p`, given separately so that neither rounds):
    the inverse incomplete gamma function for a start, or the normal
    quantile with its first correction in :math:`Q` near the lognormal,
    then Newton's method on the log of the smaller tail, whose accuracy
    is that of :func:`_log_tails`."""
    p, s, Q = (
        onp.array(v)
        for v in onp.broadcast_arrays(
            onp.asarray(p, dtype=float),
            onp.asarray(s, dtype=float),
            onp.asarray(Q, dtype=float),
        )
    )
    lower = p <= s
    target = onp.log(onp.where(lower, p, s))
    with onp.errstate(all="ignore"):
        # Q may vary from point to point (a posterior's quantile at each
        # of its parameter points): both starts, each where it belongs
        near = onp.abs(Q) < _temme.Q_MAX
        z = onp.where(lower, norm.ppf(p), norm.isf(s))
        w_near = z - Q * (z * z + 2.0) / 6.0
        Q_far = onp.where(near, 1.0, Q)
        q = 1.0 / Q_far**2
        small = onp.where(lower, p, s)
        # the gamma variable's lower tail is F for Q > 0, S for Q < 0
        gamma_lower = lower == (Q_far > 0)
        u = onp.where(
            gamma_lower,
            gammaincinv(q, small),
            gammainccinv(q, small),
        )
        # where the gamma variable's lower tail underflows, its
        # asymptote P(q, u) ~ u^q / Gamma(q + 1)
        log_u = onp.where(
            gamma_lower & ~(u > 0),
            (onp.log(small) + gammaln(q + 1.0)) / q,
            onp.log(u),
        )
        w = onp.where(near, w_near, (log_u - onp.log(q)) / Q_far)
        w = onp.where(onp.isfinite(w), w, 0.0)
        for _ in range(60):
            log_s, log_f = _log_tails(w, Q)
            log_s = onp.asarray(log_s, dtype=float)
            log_f = onp.asarray(log_f, dtype=float)
            current = onp.where(lower, log_f, log_s)
            # d(ln F)/dw = g / F, d(ln S)/dw = -g / S
            slope = onp.exp(_log_g(w, Q) - current)
            slope = onp.where(lower, slope, -slope)
            step = (current - target) / slope
            step = onp.where(onp.isfinite(step), step, 0.0)
            # no step longer than 1 in w, or than |w| far out: the log
            # tails are close to linear or quadratic in w
            limit = onp.maximum(1.0, onp.abs(w))
            step = onp.clip(step, -limit, limit)
            w = w - step
            if onp.all(onp.abs(step) <= 4e-16 * onp.maximum(1.0, onp.abs(w))):
                break
    out = onp.where(p <= 0, -onp.inf, onp.where(s <= 0, onp.inf, w))
    return onp.where(onp.isnan(p) | onp.isnan(s), onp.nan, out)


class GeneralizedGamma_(OptimisedFitMixin, ParametricFitter):
    r"""
    The generalised gamma distribution in Prentice's (1974)
    parameterisation, the one R's ``flexsurv`` calls ``gengamma``: one
    family that holds the Weibull, the lognormal and the gamma, so a
    single fit says which of them the data look like.

    The parameters are ``mu``, the location of the log time, ``sigma``,
    the scale of the log time (positive), and the shape ``Q``, any real
    number. With :math:`w = (\ln x - \mu) / \sigma` and :math:`q =
    Q^{-2}`, the survival function is

    .. math::
        R(x) = \begin{cases}
        1 - P\left(q, q e^{Q w}\right) & Q > 0 \\
        1 - \Phi(w) & Q = 0 \\
        P\left(q, q e^{Q w}\right) & Q < 0
        \end{cases}

    where :math:`P` is the regularised lower incomplete gamma function
    and :math:`\Phi` the standard normal distribution function.
    ``Q = 0`` is the lognormal (``LogNormal(mu, sigma)``), ``Q = 1`` the
    Weibull with scale :math:`e^{\mu}` and shape :math:`1 / \sigma`, and
    ``Q = sigma`` the gamma with shape :math:`1 / \sigma^2` and rate
    :math:`\sigma^2 e^{-\mu}`; ``Q = sigma = 1`` is the exponential. The
    family is continuous in ``Q`` through 0: the functions are exact
    there, not a switch.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import GeneralizedGamma
    >>> model = GeneralizedGamma.from_params([2.0, 0.5, 1.0])
    >>> model.sf([2, 5, 10]).round(4)
    array([0.9294, 0.6326, 0.1602])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution
        """

    # How the parameters move with the time (rescale_time): mu is a log
    # time; sigma and Q have no unit.
    _time_roles = ("log_time", "shape", "shape")

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=3,
            bounds=((None, None), (0, None), (None, None)),
            support=(0, np.inf),
            parameter_names=["mu", "sigma", "Q"],
            param_map={"mu": 0, "sigma": 1, "Q": 2},
            plot_x_scale="log",
        )
        # No linearising probability plot: the straight-line axis is the
        # quantile of the standardised log time, which depends on Q, so
        # it needs the answer to draw (as the Gamma's). ``plot()`` draws
        # on the fitted Q, where it is the right axis.
        self.supports_mpp = False

    def _offset_limit_family(self) -> Any:
        """The ``Normal``: as the offset runs to -inf the shifted
        lognormal, a member of the family, tends to a Normal (see
        ``OptimisedFitMixin._offset_limit_family``)."""
        from surpyval.univariate.parametric import Normal

        return Normal

    def _runaway_advice(
        self, runaway: list[str], values: dict, offset: bool = False
    ) -> str:
        """The GeneralizedGamma's limits as Q runs off. The gamma variable
        of shape q = 1/Q^2 -> 0 has G^q tending to a uniform, so x tends
        to c U^(sigma Q): for Q -> inf the power law F = (x/c)^(1/(sigma
        Q)) up to c, which ends at the largest observation, and for Q ->
        -inf its mirror, a support that starts at c."""
        if "Q" in runaway and values["Q"] > 0:
            return (
                "as Q grows the GeneralizedGamma approaches a power law "
                "F(x) = (x/c)**(1/(sigma*Q)) up to x = c, which ends at the "
                "largest observation: the data look bounded above, which no "
                "GeneralizedGamma is; compare a family with an upper limit, "
                "or the Gompertz and Weibull (surpyval.fit_best)"
            )
        if "Q" in runaway and values["Q"] < 0:
            return (
                "as Q falls the GeneralizedGamma approaches a distribution "
                "whose support starts at the smallest observation: the data "
                "look bounded below; compare an offset fit (offset=True)"
            )
        return super()._runaway_advice(runaway, values, offset)

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            return self._offset_seed(data)
        # The lognormal member of the family (Q = 0), fitted to the data:
        # its maximum is closed form or a fast search.
        from surpyval.univariate.parametric import LogNormal

        with onp.errstate(all="ignore"):
            ln = LogNormal._fit_from_surpyval_data(data, how="MLE")
        mu, sigma = (float(v) for v in ln.params)
        if not (onp.isfinite(mu) and onp.isfinite(sigma) and sigma > 0):
            log_x = onp.log(onp.asarray(data.x, dtype=float))
            log_x = log_x[onp.isfinite(log_x)]
            mu = float(onp.mean(log_x)) if log_x.size else 0.0
            sigma = float(onp.std(log_x)) if log_x.size > 1 else 1.0
            sigma = sigma if sigma > 0 else 1.0
        return onp.array([mu, sigma, 0.0])

    def _alternative_base_starts(
        self, data: SurpyvalData, offset: bool
    ) -> list[npt.NDArray]:
        """The Weibull member (Q = 1), fitted to the data, as a second
        start: the likelihood in Q can have a shoulder between the
        lognormal and the Weibull, and a search from the better of the
        two reaches the maximum from its own side."""
        if offset:
            return []
        from surpyval.univariate.parametric import Weibull

        try:
            with onp.errstate(all="ignore"):
                wb = Weibull._fit_from_surpyval_data(data, how="MLE")
            alpha, beta = (float(v) for v in wb.params)
        except Exception:
            return []
        if not (alpha > 0 and beta > 0 and onp.isfinite(alpha * beta)):
            return []
        return [onp.array([onp.log(alpha), 1.0 / beta, 1.0])]

    @staticmethod
    def _w(x: Numeric, mu: Boxable, sigma: Boxable) -> tuple:
        """The standardised log time ``(ln x - mu) / sigma`` and ``x``
        with the points at and below 0, and at infinity, replaced by 1
        (for :meth:`_edges` to overwrite)."""
        x_pos = positive_or_one(x)
        if onp.any(onp.isposinf(getval(x_pos))):
            x_pos = np.where(np.isinf(x_pos), 1.0, x_pos)
        return (np.log(x_pos) - mu) / sigma, x_pos

    @staticmethod
    def _edges(x: Numeric, inside: Boxable, at_zero: Any, at_inf: Any) -> Any:
        """``inside`` in the support, its limits at x = 0 and x = inf."""
        out = on_support(x, inside, at_zero)
        if onp.any(onp.isposinf(onp.asarray(getval(x), dtype=float))):
            if callable(at_inf):
                at_inf = at_inf()
            out = np.where(np.isposinf(x), at_inf + np.zeros_like(out), out)
        return out

    @staticmethod
    def _at_zero(mu: Boxable, sigma: Boxable, Q: Boxable) -> tuple:
        r"""The density (and hazard) at x = 0 and its log. For
        :math:`Q > 0` it behaves like :math:`c\,x^{1/(Q\sigma) - 1}`, with
        :math:`\ln c = q\ln q - \ln(\sigma/Q) - \ln\Gamma(q) - \mu
        /(Q\sigma)`; for :math:`Q \le 0` it is 0."""
        Q_pos = np.where(Q > 0, Q, 1.0)
        q = 1.0 / (Q_pos * Q_pos)
        log_c = (
            q * np.log(q)
            - np.log(sigma / Q_pos)
            - agammaln(q)
            - mu / (Q_pos * sigma)
        )
        value, log_value = power_at_zero(1.0 / (Q_pos * sigma) - 1.0, log_c)
        return (
            np.where(Q > 0, value, 0.0),
            np.where(Q > 0, log_value, -np.inf),
        )

    def log_df(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        # ln g(w) - ln sigma - ln x; g is exact through Q = 0 (_log_g)
        w, x_pos = self._w(x, mu, sigma)
        inside = _log_g(w, Q) - np.log(sigma) - np.log(x_pos)
        return self._edges(
            x, inside, lambda: self._at_zero(mu, sigma, Q)[1], -np.inf
        )

    def log_sf(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        w, _ = self._w(x, mu, sigma)
        return self._edges(x, _log_tails(w, Q)[0], 0.0, -np.inf)

    def log_ff(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        w, _ = self._w(x, mu, sigma)
        return self._edges(x, _log_tails(w, Q)[1], -np.inf, 0.0)

    def sf(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        Survival (or reliability) function for the GeneralizedGamma
        distribution, with :math:`w = (\ln x - \mu)/\sigma` and :math:`q =
        Q^{-2}`:

        .. math::
            R(x) = 1 - P\left(q, q e^{Q w}\right) \; (Q > 0), \quad
            1 - \Phi(w) \; (Q = 0), \quad
            P\left(q, q e^{Q w}\right) \; (Q < 0)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedGamma
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedGamma.sf(x, 1.5, 0.6, 0.5)
        array([0.97072442, 0.84947101, 0.36260394, 0.0482264 ])
        """
        return np.exp(self.log_sf(x, mu, sigma, Q))

    def ff(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the GeneralizedGamma
        distribution, :math:`F(x) = 1 - R(x)`, computed as its own tail so
        that neither rounds.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedGamma
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedGamma.ff(x, 1.5, 0.6, 0.5)
        array([0.02927558, 0.15052899, 0.63739606, 0.9517736 ])
        """
        return np.exp(self.log_ff(x, mu, sigma, Q))

    def df(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        Density function for the GeneralizedGamma distribution:

        .. math::
            f(x) = \frac{|Q| \, q^{q}}{\sigma x \, \Gamma(q)}
            \exp\left\{q \left(Q w - e^{Q w}\right)\right\}

        with :math:`w = (\ln x - \mu)/\sigma`, :math:`q = Q^{-2}`, and the
        lognormal density at :math:`Q = 0`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedGamma
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedGamma.df(x, 1.5, 0.6, 0.5)
        array([0.0761597 , 0.15668244, 0.12802968, 0.02098509])
        """
        return np.exp(self.log_df(x, mu, sigma, Q))

    def hf(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        Instantaneous hazard rate for the GeneralizedGamma distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedGamma
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedGamma.hf(x, 1.5, 0.6, 0.5)
        array([0.07845657, 0.18444707, 0.35308409, 0.43513705])
        """
        # From the hazard of the standardised log time (_log_hazard): the
        # difference ln f - ln S is -inf - -inf once both underflow.
        w, x_pos = self._w(x, mu, sigma)
        inside = np.exp(_log_hazard(w, Q) - np.log(sigma) - np.log(x_pos))
        return self._edges(
            x,
            inside,
            lambda: self._at_zero(mu, sigma, Q)[0],
            lambda: self._hf_at_inf(mu, sigma, Q),
        )

    @staticmethod
    def _hf_at_inf(mu: Boxable, sigma: Boxable, Q: Boxable) -> Boxable:
        r"""The hazard's limit as :math:`x \to \infty`: it behaves like
        :math:`x^{Q/\sigma - 1}` for :math:`Q > 0`, so it is infinite for
        :math:`Q > \sigma`, the gamma's rate :math:`e^{-\mu}/\sigma^2` at
        :math:`Q = \sigma`, and 0 below (and for :math:`Q \le 0`)."""
        return np.where(
            Q > sigma,
            np.inf,
            np.where(Q == sigma, np.exp(-mu) / sigma**2, 0.0),
        )

    def Hf(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        Cumulative hazard rate for the GeneralizedGamma distribution:

        .. math::
            H(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedGamma
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedGamma.Hf(x, 1.5, 0.6, 0.5)
        array([0.02971266, 0.16314147, 1.01444411, 3.03184863])
        """
        return -self.log_sf(x, mu, sigma, Q)

    def qf(
        self, u: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        Quantile function for the GeneralizedGamma distribution:

        .. math::
            q(u) = \exp\left(\mu + \sigma w_u\right)

        where :math:`w_u` is the standardised log time with :math:`F = u`:
        :math:`\ln(P^{-1}(q, u) / q) / Q` for :math:`Q > 0` (with the
        upper tail's inverse for :math:`Q < 0`) and the normal quantile at
        :math:`Q = 0`, refined by Newton's method on the log of the tail.

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedGamma distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedGamma distribution
        Q : numpy array or scalar
            shape parameter for the GeneralizedGamma distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the GeneralizedGamma distribution at each
            value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedGamma
        >>> u = np.array([.1, .5, .9])
        >>> GeneralizedGamma.qf(u, 1.5, 0.6, 0.5)
        array([1.65598129, 4.04446948, 8.29397224])
        """
        u, mu_, sigma_, Q_ = broadcast_values(u, mu, sigma, Q)
        w = _std_quantile(u, 1.0 - u, Q_)
        with onp.errstate(over="ignore"):
            return onp.exp(mu_ + sigma_ * w)

    def isf(
        self, s: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""The inverse survival function: the time at which the survival
        function is ``s``, from ``s`` itself so that it holds deep in the
        upper tail.

        Examples
        --------
        >>> from surpyval import GeneralizedGamma
        >>> GeneralizedGamma.isf([0.5, 1e-300], 1.5, 0.6, 0.5)
        array([   4.04446948, 2236.18962511])
        """
        s, mu_, sigma_, Q_ = broadcast_values(s, mu, sigma, Q)
        w = _std_quantile(1.0 - s, s, Q_)
        with onp.errstate(over="ignore"):
            return onp.exp(mu_ + sigma_ * w)

    def moment(
        self, m: int, mu: Boxable, sigma: Boxable, Q: Boxable
    ) -> Boxable:
        r"""

        The ``m``-th raw moment :math:`E[X^m]` of the GeneralizedGamma
        distribution, with :math:`s = m\sigma/Q` and :math:`q = Q^{-2}`:

        .. math::
            E[X^m] = e^{m\mu} \, q^{-s}\, \frac{\Gamma(q + s)}{\Gamma(q)}

        and :math:`e^{m\mu + m^2\sigma^2/2}` at :math:`Q = 0`. It is
        infinite where :math:`Q < 0` and :math:`m \sigma |Q| \geq 1`: the
        upper tail is then too heavy. Computed in a form that is exact
        through :math:`Q = 0`.

        Examples
        --------
        >>> from surpyval import GeneralizedGamma
        >>> round(float(GeneralizedGamma.moment(2, 1.5, 0.6, 0.5)), 6)
        28.940473
        """
        # ln E = m mu + m^2 sigma^2 phi(r) - ln(1 + r) / 2
        #        + stirlerr(q (1 + r)) - stirlerr(q),  r = m sigma Q
        r = m * sigma * Q
        exists = r > -1.0
        r_ok = np.where(exists, r, 0.0)
        z = Q * Q
        log_e = (
            m * mu
            + (m * sigma) ** 2 * _phi(r_ok)
            - 0.5 * np.log1p(r_ok)
            + _stirlerr(z / (1.0 + r_ok))
            - _stirlerr(z)
        )
        return np.where(exists, np.exp(log_e), np.inf)

    def mean(self, mu: Boxable, sigma: Boxable, Q: Boxable) -> Boxable:
        r"""

        The mean of the GeneralizedGamma distribution, its first moment
        (see :meth:`moment`); infinite for :math:`Q \le -1/\sigma`.

        Examples
        --------
        >>> from surpyval import GeneralizedGamma
        >>> round(float(GeneralizedGamma.mean(1.5, 0.6, 0.5)), 6)
        4.610459
        """
        return self.moment(1, mu, sigma, Q)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        # the standardised log time at the plotting position, on the
        # fitted Q: a straight line in ln x of slope 1 / sigma
        Q = float(getval(params[-1]))
        y = onp.asarray(y, dtype=float)
        out = onp.full_like(y, onp.nan)
        inside = (y > 0) & (y < 1)
        out[inside] = _std_quantile(y[inside], 1.0 - y[inside], Q)
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        Q = float(getval(params[-1]))
        return onp.exp(onp.asarray(_log_tails(onp.asarray(y), Q)[1]))


GeneralizedGamma: GeneralizedGamma_ = GeneralizedGamma_("GeneralizedGamma")
