from __future__ import annotations

from typing import Callable

import autograd.numpy as np
import numpy.typing as npt
from scipy import integrate

from surpyval.univariate import parametric as para
from surpyval.univariate.parametric._fit_inputs import _offset_start
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData

# ln 2: log(1 - e^-r) is taken as log1p(-e^-r) above it and as
# log(-expm1(-r)) below it, each exact on its own side.
_LN2 = float(np.log(2.0))
# Below this log r, log(1 - e^-r) = log r - r / 2 to double precision (the
# next term is r^2 / 24), and it stays finite after r itself underflows.
_LOG_SMALL = -20.0
# Above this t, e^-t < 5e-18 and -log(1 - e^-t) = e^-t (1 + e^-t / 2 ...)
# is e^-t to double precision, so its log is -t, which stays finite after
# e^-t underflows.
_T_LARGE = 40.0


def _log1mexp(r: Boxable, log_r: Boxable) -> tuple[Boxable, Boxable]:
    r"""
    :math:`\log(1 - e^{-r})` for :math:`r \geq 0`, and
    :math:`\log((1 - e^{-r}) / r)`, from :math:`r` and :math:`\log r`.

    Each ``np.where`` branch sees only arguments it is exact and finite
    on, so neither the values nor autograd's gradients of the branch not
    taken can be NaN.
    """
    small = log_r < _LOG_SMALL
    r_mid = np.where(small, 1.0, r)
    mid = np.where(
        r_mid > _LN2,
        np.log1p(-np.exp(-r_mid)),
        np.log(-np.expm1(-r_mid)),
    )
    log_r_mid = np.where(small, 0.0, log_r)
    log_r_small = np.where(small, log_r, _LOG_SMALL)
    half_r = np.exp(log_r_small) / 2.0
    value = np.where(small, log_r_small - half_r, mid)
    ratio = np.where(small, -half_r, mid - log_r_mid)
    return value, ratio


def _log_neg_log1mexp(t: Boxable, log_g: Boxable) -> Boxable:
    r"""
    :math:`\log(-\log(1 - e^{-t}))` given :math:`\log(1 - e^{-t})`: the
    log of the right tail's :math:`-\log F^{1/\mu}`, which is
    :math:`-t` once :math:`e^{-t}` is below double precision.
    """
    large = t > _T_LARGE
    neg_log_g = np.where(large, 1.0, -log_g)
    t_large = np.where(large, t, _T_LARGE)
    return np.where(large, -t_large, np.log(neg_log_g))


def _tanh_sinh_nodes(
    h: float, z_max: float
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""
    The tanh-sinh rule for :math:`\int_0^1 f(p)\, dp`: with
    :math:`y = \pi \sinh z` and :math:`p = 1 / (1 + e^{-y})`, the
    trapezoidal rule of step ``h`` in ``z`` over :math:`[-z_{max},
    z_{max}]`. Returns the log of each node's weight,
    :math:`\ln(h\, dp/dz) = \ln(h \pi \cosh z\, p (1 - p))`, and
    :math:`\ln(-\ln p)`, both from ``y`` so that neither rounds where
    :math:`p` or :math:`1 - p` does: :math:`1 - p` is :math:`10^{-454}`
    at the last node.
    """
    z = h * np.arange(-round(z_max / h), round(z_max / h) + 1)
    y = np.pi * np.sinh(z)
    # ln p = -ln(1 + e^-y) and ln(1 - p) = -ln(1 + e^y)
    neg_log_p = np.logaddexp(0.0, -y)
    log_weight = (
        np.log(h * np.pi * np.cosh(z)) - neg_log_p - np.logaddexp(0.0, y)
    )
    # ln(-ln p) is -y to double precision once e^-y underflows
    far = y > 700.0
    log_neg_log_p = np.where(far, -y, np.log(np.where(far, 1.0, neg_log_p)))
    return log_weight, log_neg_log_p


# The tanh-sinh rule of ``_t_power_expectation``. Against 25-digit
# references (mpmath) over beta from 0.05 to 1e8, mu from 1e-3 to 1e6 and
# moments 1 to 4, a step of 1/16 is within 1.2e-11 and 1/24 within 2e-14
# (rounding); 1/32 keeps that margin. 6.5 reaches 1 - p = 1e-454, past
# where the rule's terms underflow for any moment that is finite.
_TS_LOG_WEIGHT, _TS_LOG_NEG_LOG_P = _tanh_sinh_nodes(1.0 / 32.0, 6.5)


def _t_power_expectation(s: npt.ArrayLike, mu: npt.ArrayLike) -> npt.NDArray:
    r"""
    :math:`E[T^{s}]` for :math:`T = (X/\alpha)^{\beta}`, whose density
    :math:`\mu (1 - e^{-t})^{\mu - 1} e^{-t}` is free of :math:`\alpha`
    and :math:`\beta`, elementwise over ``s`` and ``mu``.

    It is :math:`\int_0^1 Q(p)^{s}\, dp` with
    :math:`Q(p) = -\ln(1 - p^{1/\mu})` the quantile of :math:`T`: over
    the probability the mass is spread evenly whatever :math:`\mu` and
    :math:`s`, and what is left at the ends is a power of :math:`p` at 0
    and of :math:`-\ln(1 - p)` at 1, which a tanh-sinh rule integrates to
    double precision with a few hundred fixed nodes (``_TS_LOG_WEIGHT``),
    for every parameter set at once. ``quad`` over ``t`` took 9 million
    calls of a Python integrand in one method-of-moments fit (#586), was
    off by up to 1.4e-9 at :math:`\mu = 0.01` and overflowed a Python
    float for :math:`s` near 80 (:math:`\beta = 0.05`).

    The terms are taken on the log scale, :math:`\ln Q` from
    :math:`r = -\ln p / \mu` as ``_log_forms`` takes the survival
    function's: :math:`p^{1/\mu}` underflows long before
    :math:`Q(p)^{s}` does (:math:`\mu = 10^{-3}`), and in the right tail
    :math:`Q` is carried by :math:`\ln(-\ln p)` after :math:`1 - p`
    rounds to 0.
    """
    # the nodes along a last axis
    s_n = np.asarray(s, dtype=float)[..., None]
    mu_n = np.asarray(mu, dtype=float)[..., None]
    with np.errstate(all="ignore"):
        log_r = _TS_LOG_NEG_LOG_P - np.log(mu_n)
        r = np.exp(log_r)
        # ln(1 - p^(1/mu)) = ln(1 - e^-r), then ln Q = ln(-that)
        log_q = _log_neg_log1mexp(r, _log1mexp(r, log_r)[0])
        terms = s_n * log_q + _TS_LOG_WEIGHT
        top = np.max(terms, axis=-1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        total = np.exp(top[..., 0]) * np.sum(np.exp(terms - top), axis=-1)
    return total


class ExpoWeibull_(OptimisedFitMixin, ParametricFitter):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=3,
            bounds=(
                (0, None),
                (0, None),
                (0, None),
            ),
            support=(0, np.inf),
            parameter_names=["alpha", "beta", "mu"],
            param_map={"alpha": 0, "beta": 1, "mu": 2},
            plot_x_scale="log",
        )
        self.supports_mpp = False

    def _gumbel_seed(
        self,
        x: npt.NDArray,
        c: npt.NDArray | None,
        n: npt.NDArray | None,
        refine: bool,
    ) -> tuple[float, float]:
        """
        Seed alpha and beta from a Gumbel fit to log(x).

        The ExpoWeibull with mu = 1 is a Weibull, and a Weibull's logs
        are Gumbel distributed with mu = log(alpha) and sigma = 1 / beta,
        so a fit of log(x) gives both shape parameters at once.

        ``refine`` runs the Gumbel MLE rather than reading the
        probability plot alone. Without an offset the plot is already
        good enough: refining cost 15-30% of the fit and changed nothing
        over 54 parameter combinations plus right, left and heavily tied
        data, every fit reaching the same optimum to the optimiser's own
        tolerance. With an offset the plot alone is measurably worse --
        five of 48 offset fits landed on a worse optimum, one of them at
        685.85 against 622.26 -- so the offset path refines.
        """
        log_x = np.log(x)
        log_x[np.isnan(log_x)] = 0
        gumb = para.Gumbel.fit(log_x, c, n, how="MLE" if refine else "MPP")
        # ``res`` is the optimiser result, present only on an MLE
        # fit -- which is the only branch that sets refine.
        if refine and not gumb.res.success:  # type: ignore[attr-defined]
            gumb = para.Gumbel.fit(log_x, c, n, how="MPP")
        mu, sigma = gumb.params
        alpha, beta = np.exp(mu), 1.0 / sigma
        if np.isinf(alpha) | np.isnan(alpha):
            alpha = np.median(x)
        if np.isinf(beta) | np.isnan(beta):
            beta = 1.0
        return alpha, beta

    def _runaway_advice(self, runaway: list[str], values: dict) -> str:
        """The ExpoWeibull's two limits (#584). As ``mu`` grows, ``F =
        g**mu`` is the largest of ``mu`` Weibull lifetimes, which tends to
        a largest-extreme-value law of ``x**beta``, and with ``beta``
        falling too, of ``log(x)``: a Frechet distribution (60 interval
        censored, truncated rows reached log-likelihood -73.45 at mu =
        8e19 against the Frechet's -73.38). As ``beta`` grows with ``mu``
        falling, ``F`` tends to the power law ``(x/alpha)**(beta mu)`` up
        to ``alpha``, which ends at the largest observation (50 Weibull
        draws: -248.4447 at beta = 4e5 against the power law's
        -248.4433)."""
        if "mu" in runaway and values["mu"] > 1:
            return (
                "as mu grows the ExpoWeibull approaches a largest extreme "
                "value law of x**beta, and as beta falls too, of log(x) (a "
                "Frechet distribution): compare surpyval.GumbelLEV fitted "
                "to the logs of the times"
            )
        if "beta" in runaway and values["beta"] > 1 and values["mu"] < 1:
            return (
                "as beta grows and mu falls the ExpoWeibull approaches the "
                "power law F(x) = (x/alpha)**(beta*mu) up to x = alpha, "
                "which ends at the largest observation: the data look "
                "bounded above, which no ExpoWeibull is; compare a family "
                "with an upper limit"
            )
        return super()._runaway_advice(runaway, values)

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        x, c, n = data.x, data.c, data.n
        if offset:
            # Estimate the offset first and seed alpha and beta from the
            # shifted data. Taking logs before removing the shift reads
            # log(x) instead of log(x - gamma), and a large shift
            # compresses those logs into a narrow band: at gamma = 100
            # with a true beta of 2 the seed came back at beta = 23 and
            # the MLE then failed outright, falling back to MPP.
            #
            # ``_offset_start`` because the fitter overwrites the returned
            # offset with exactly that (see ``_initial_guess``); seeding
            # alpha and beta against a different shift than the one
            # actually installed defeats the point of shifting at all.
            gamma = _offset_start(x)
            alpha, beta = self._gumbel_seed(x - gamma, c, n, refine=True)
            return np.array([gamma, alpha, beta, 1.0], dtype=float)
        return np.array(
            [*self._gumbel_seed(x, c, n, refine=False), 1.0],
            dtype=float,
        )

    @staticmethod
    def _log_forms(
        x: Numeric,
        alpha: Boxable,
        beta: Boxable,
        mu: Boxable,
        right: bool = True,
    ) -> dict[str, Boxable]:
        r"""
        The pieces every function is built from, each on the log scale so
        that none of them rounds to 0, 1 or inf before it has to.

        With :math:`t = (x/\alpha)^{\beta}` and
        :math:`g = 1 - e^{-t}` (so :math:`F = g^{\mu}`):

        - ``log_t`` is :math:`\beta(\ln x - \ln \alpha)`, which does not
          overflow where :math:`x/\alpha` does;
        - ``log_g`` is :math:`\ln g`, exact in the lower tail where
          :math:`1 - e^{-t}` is exactly 0 once :math:`t < 10^{-16}`;
        - ``ratio_g`` is :math:`\ln(g / t)`, so that the density's
          :math:`\ln t + (\mu - 1) \ln g` is taken as
          :math:`\mu \ln g - \ln(g / t)`: as written it is the
          difference of two terms of the size of :math:`\beta \ln(x /
          \alpha)`, which at :math:`\beta = 10^{20}` cancel to an error of
          :math:`10^4` (#472);
        - ``log_ff`` is :math:`\mu \ln g`;
        - ``log_nl`` is :math:`\ln(-\ln g)`, which is :math:`-t` in the
          right tail after :math:`e^{-t}` underflows, so that
          ``log_r`` :math:`= \ln \mu + \ln(-\ln g) = \ln(-\ln F)`
          stays finite there;
        - ``log_sf`` is :math:`\ln(1 - F) = \ln(1 - e^{-r})` with
          :math:`r = -\ln F`, and ``ratio_r`` is
          :math:`\ln((1 - e^{-r}) / r)`, which ``hf`` needs to cancel the
          :math:`e^{-t}` of the density against that of the survival
          function exactly rather than as a difference of two large logs.

        ``right=False`` stops at ``log_ff``, all that ``ff``, ``log_ff``
        and the density need. Points at or below 0, and at infinity, are
        evaluated at 1 (the caller replaces them), so that no branch of any
        ``np.where`` sees a log of 0, or the ``inf - inf`` of the log forms
        at ``x = inf`` (#561).
        """
        x_pos = np.where((x > 0) & (x < np.inf), x, 1.0)
        log_x = np.log(x_pos)
        log_t = beta * (log_x - np.log(alpha))
        with np.errstate(over="ignore"):
            t = np.exp(log_t)
        log_g, ratio_g = _log1mexp(t, log_t)
        log_ff = mu * log_g
        out = {
            "log_x": log_x,
            "t": t,
            "log_g": log_g,
            "ratio_g": ratio_g,
            "log_ff": log_ff,
        }
        if not right:
            return out
        log_nl = _log_neg_log1mexp(t, log_g)
        log_sf, ratio_r = _log1mexp(-log_ff, np.log(mu) + log_nl)
        out.update(log_nl=log_nl, log_sf=log_sf, ratio_r=ratio_r)
        return out

    @staticmethod
    def _support(
        x: Numeric, inside: Boxable, at_zero: Boxable, at_inf: Boxable
    ) -> Boxable:
        """``inside`` for 0 < x < inf, ``at_zero`` at 0, ``at_inf`` at
        inf and NaN below 0."""
        # autograd's ``where`` does not unbroadcast its gradient, so a
        # parameter-dependent ``at_zero`` must have the full shape.
        at_zero = at_zero + np.zeros_like(inside)
        at_inf = at_inf + np.zeros_like(inside)
        inside = np.where(x == np.inf, at_inf, inside)
        out = np.where(x > 0, inside, np.where(x == 0, at_zero, np.nan))
        # a scalar in, a scalar out (a 0-d where is an array)
        return out[()]

    def sf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        Survival (or reliability) function for the ExpoWeibull Distribution:

        .. math::
            R(x) = 1 - \left [ 1 - e^{-\left ( \frac{x}{\alpha} \right )^\beta}
             \right ]^{\mu}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.sf(x, 3, 4, 1.2)
        array([9.94911330e-01, 8.72902497e-01, 4.23286791e-01, 5.06674866e-02,
               5.34717283e-04])
        """
        # -expm1(log F) is the cancellation-free form of 1 - F (#257);
        # past t = 1 exp(log_sf), which keeps the survival function from
        # underflowing with e^-t when mu e^-t is still representable
        # (#436).
        p = self._log_forms(x, alpha, beta, mu)
        inside = np.where(
            p["t"] > 1.0, np.exp(p["log_sf"]), -np.expm1(p["log_ff"])
        )
        return self._support(x, inside, 1.0, 0.0)

    def ff(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the ExpoWeibull
        Distribution:

        .. math::
            F(x) = \left [ 1 - e^{-\left ( \frac{x}{\alpha} \right )^\beta}
            \right ]^{\mu}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.ff(x, 3, 4, 1.2)
        array([0.00508867, 0.1270975 , 0.57671321, 0.94933251, 0.99946528])
        """
        p = self._log_forms(x, alpha, beta, mu, right=False)
        return self._support(x, np.exp(p["log_ff"]), 0.0, 1.0)

    def df(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        Density function for the ExpoWeibull Distribution:

        .. math::
            f(x) = \mu \left ( \frac{\beta}{\alpha} \right ) \left ( \frac{x}
            {\alpha} \right )^{\beta - 1} \left [ 1 - e^{-\left ( \frac{x}
            {\alpha} \right )^\beta} \right ]^{\mu - 1} e^{- \left ( \frac{x}
            {\alpha} \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.df(x, 3, 4, 1.2)
        array([0.02427515, 0.27589838, 0.53701385, 0.15943643, 0.00330058])
        """
        return np.exp(self.log_df(x, alpha, beta, mu))

    def hf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        Instantaneous hazard rate for the ExpoWeibull Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.hf(x, 3, 4, 1.2)
        array([0.02439931, 0.3160701 , 1.26867613, 3.14672068, 6.17256436])
        """
        # f / R with the e^-t of both cancelled algebraically: with
        # r = -ln F and q = r / (mu e^-t) = -ln(1 - e^-t) / e^-t,
        # h = (beta / x) t g^(mu - 1) / (q (1 - e^-r) / r). The quotient
        # of the two separately computed functions is 0 / 0 once both
        # underflow, and their log difference loses t * eps (#436).
        p = self._log_forms(x, alpha, beta, mu)
        large = p["t"] > _T_LARGE
        # ln q, e^-t / 2 to double precision (so 0) in the right tail
        log_q = np.where(
            large, 0.0, p["log_nl"] + np.where(large, 0.0, p["t"])
        )
        # ln t + (mu - 1) ln g as mu ln g - ln(g / t), whose terms do not
        # cancel (#472).
        log_hf = (
            np.log(beta)
            - p["log_x"]
            + mu * p["log_g"]
            - p["ratio_g"]
            - p["ratio_r"]
            - log_q
        )
        with np.errstate(over="ignore"):
            inside = np.exp(log_hf)
        return self._support(
            x,
            inside,
            self._df_at_zero(alpha, beta, mu),
            self._hf_at_inf(alpha, beta),
        )

    def Hf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        Instantaneous hazard rate for the ExpoWeibull Distribution:

        .. math::
            H(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.Hf(x, 3, 4, 1.2)
        array([5.10166141e-03, 1.35931416e-01, 8.59705336e-01, 2.98247086e+00,
               7.53377239e+00])
        """
        # 0 - rather than a unary minus, which gives -0.0 at x = 0
        return 0.0 - self.log_sf(x, alpha, beta, mu)

    def qf(
        self, u: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        Quantile function for the ExpoWeibull Distribution:

        .. math::
            q(u) = \alpha \left ( -\ln \left ( 1 - u^{1/\mu} \right )
            \right )^{1/\beta}

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        Q : scalar or numpy array
            The quantiles for the ExpoWeibull distribution at each value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> ExpoWeibull.qf(u, 3, 4, 1.2)
        array([1.89361341, 2.2261045 , 2.46627621, 2.66992747, 2.85807988])
        """
        # t = -ln(1 - v) with v = u^(1/mu), exact on both sides of
        # v = 1/2: at u = 1 - 1e-16 and mu = 500 v rounds to 1 and the
        # direct form is inf (#436). It is carried as log(t), since v
        # underflows long before the quantile does (u = 1e-30, mu = 0.01:
        # v = 1e-3000, but with beta = 1000 the quantile is alpha * 1e-3);
        # for small v, log(t) = log(v) + log(t / v) with t / v -> 1.
        # Outside [0, 1] it is NaN, without a warning.
        with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
            log_v = np.log(u) / mu
            v = np.exp(log_v)
            low = v < 0.5
            v_low = np.where(low & (v > 0), v, 0.25)
            log_v_high = np.where(low, -1.0, log_v)
            log_t = np.where(
                low,
                log_v + np.log(-np.log1p(-v_low) / v_low),
                np.log(-np.log(-np.expm1(log_v_high))),
            )
            log_t = np.where(low & ~(v > 0), log_v, log_t)
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            q = alpha * np.exp(log_t / beta)
            # exp(log t / beta) overflows (or underflows) where the
            # quantile does not: at alpha = 1e-308, beta = 0.0076 and mu =
            # 3e95, qf(0.95) is 45 (#601). There it is taken in logs.
            far = ~np.isfinite(q) | (q == 0)
            return np.where(far, np.exp(np.log(alpha) + log_t / beta), q)

    def log_df(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        p = self._log_forms(x, alpha, beta, mu, right=False)
        # ln f = ln(beta mu / x) + ln t + (mu - 1) ln g - t, with
        # ln t + (mu - 1) ln g taken as mu ln g - ln(g / t): as written,
        # beta ln x - beta ln alpha + (mu - 1) ln g cancelled to an error of
        # about 1e4 at beta = 1e20, and put the likelihood far above its
        # maximum (#472).
        inside = (
            np.log(beta)
            + np.log(mu)
            - p["log_x"]
            + mu * p["log_g"]
            - p["ratio_g"]
            - p["t"]
        )
        bm = beta * mu
        at_zero = np.where(
            bm < 1, np.inf, np.where(bm == 1, -np.log(alpha), -np.inf)
        )
        return self._support(x, inside, at_zero, -np.inf)

    @staticmethod
    def _df_at_zero(alpha: Boxable, beta: Boxable, mu: Boxable) -> Boxable:
        r"""
        The density's limit at 0, where it behaves like
        :math:`x^{\beta\mu - 1} \mu\beta / \alpha^{\beta\mu}`: inf,
        :math:`1/\alpha` or 0 as :math:`\beta\mu` is below, at or above
        1 (the hazard's too, as the survival function there is 1).
        """
        bm = beta * mu
        return np.where(bm < 1, np.inf, np.where(bm == 1, 1.0 / alpha, 0.0))

    @staticmethod
    def _hf_at_inf(alpha: Boxable, beta: Boxable) -> Boxable:
        r"""
        The hazard's limit at infinity, where ``F`` is 1 and it is the
        Weibull's :math:`(\beta / \alpha) (x / \alpha)^{\beta - 1}`: 0,
        :math:`1/\alpha` or inf as :math:`\beta` is below, at or above 1.
        """
        return np.where(
            beta < 1, 0.0, np.where(beta == 1, 1.0 / alpha, np.inf)
        )

    def log_ff(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        p = self._log_forms(x, alpha, beta, mu, right=False)
        return self._support(x, p["log_ff"], -np.inf, 0.0)

    def log_sf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        # log(1 - e^-r) with r = -ln F carried as ln r, which in the right
        # tail is ln(mu) - t: finite after e^-t underflows, where the log
        # of the survival function itself is -inf (#257, #436).
        p = self._log_forms(x, alpha, beta, mu)
        return self._support(x, p["log_sf"], 0.0, -np.inf)

    def moment(
        self, m: int, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        r"""

        m-th (non central) moment of the ExpoWeibull distribution.

        .. math::
            E = \int_{0}^{\infty} x^{m} f(x) dx

        There is a closed form -- an infinite series in
        :math:`\binom{\mu - 1}{i}(-1)^{i}(i + 1)^{-(1 + m/\beta)}` -- but
        it only terminates when :math:`\mu` is a positive integer, and
        for other :math:`\mu` it is alternating and slow to converge,
        losing significance to cancellation as :math:`\mu` grows. So the
        integral is taken by quadrature on the distribution's own scale:
        with :math:`t = (x/\alpha)^{\beta}`,

        .. math::
            E[X^{m}] = \alpha^{m} \int_{0}^{\infty} t^{m/\beta}\,
            \mu (1 - e^{-t})^{\mu - 1} e^{-t}\, dt
            = \alpha^{m} \int_{0}^{1} Q(p)^{m/\beta}\, dp ,

        whose integrand does not depend on :math:`\alpha`, so the result
        is equally accurate at any scale. The second form, over the
        probability :math:`p` with :math:`Q(p) = -\ln(1 - p^{1/\mu})` the
        quantile of :math:`t`, is taken by a fixed tanh-sinh rule, all
        parameter sets at once (see ``_t_power_expectation``).

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the ExpoWeibull distribution

        Examples
        --------
        >>> from surpyval import ExpoWeibull
        >>> ExpoWeibull.moment(2, 3, 4, 1.2)
        8.59842561360511
        """

        a, b, u = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (alpha, beta, mu))
        )
        out = a**m * _t_power_expectation(float(m) / b, u)
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def _t_expectation(
        g: Callable[[float], float], beta: Boxable, mu: Boxable
    ) -> float:
        r"""
        :math:`E[g(T)]` for :math:`T = (X/\alpha)^{\beta}`, whose density
        :math:`\mu (1 - e^{-t})^{\mu - 1} e^{-t}` is free of
        :math:`\alpha` (and of :math:`\beta`), by ``quad``, for
        ``entropy``; the moments, :math:`g(t) = t^{s}`, have the fixed
        rule of ``_t_power_expectation`` (#586).

        Integrating over ``x`` directly, the old way, put the mass wherever
        :math:`\alpha` put it, and ``quad`` over :math:`[0, \infty)`
        missed it away from unit scale: the mean at
        :math:`\alpha = 10^{-4}` came back as exactly 0, and at
        :math:`10^{4}` 0.2% high with an IntegrationWarning. In ``t`` the
        mass is always near 1; the split there keeps the (integrable)
        singularity at 0 for :math:`\mu < 1` apart from the tail.
        """
        mu_f = float(mu)
        log_mu = np.log(mu_f)

        def integrand(t: float) -> float:
            if t <= 0.0:
                return 0.0
            log_p = log_mu + (mu_f - 1.0) * np.log(-np.expm1(-t)) - t
            return float(g(t) * np.exp(log_p))

        with np.errstate(all="ignore"):
            lower = integrate.quad(integrand, 0.0, 1.0, limit=200)[0]
            upper = integrate.quad(integrand, 1.0, np.inf, limit=200)[0]
        return float(lower + upper)

    def mean(self, alpha: Boxable, beta: Boxable, mu: Boxable) -> Boxable:
        r"""
        The mean of the ExpoWeibull distribution, the first moment (see
        :meth:`moment`), found by numerical integration.

        Examples
        --------
        >>> from surpyval import ExpoWeibull
        >>> ExpoWeibull.mean(3, 4, 1.2)
        2.8422622081888917
        """
        return self.moment(1, alpha, beta, mu)

    def entropy(self, alpha: Boxable, beta: Boxable, mu: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the ExpoWeibull distribution.

        The entropy of the ExpoWeibull distribution has no closed form
        and is therefore computed by numerical integration of:

        .. math::
            S = -\int_{0}^{\infty} f(x) \ln f(x) dx

        taken, like :meth:`moment`, over :math:`t = (x/\alpha)^{\beta}`
        (so :math:`S = \ln\alpha - E[\ln f_{1}(T^{1/\beta})]`, with
        :math:`f_{1}` the density at unit scale), which keeps it accurate
        at any scale.

        Parameters
        ----------

        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        entropy : scalar
            The entropy of the ExpoWeibull distribution

        Examples
        --------
        >>> from surpyval import ExpoWeibull
        >>> ExpoWeibull.entropy(3, 1.5, 0.8)
        1.8227536487527594
        """

        b, m = float(beta), float(mu)

        def log_f1(t: float) -> float:
            # log density at unit scale, at x = t ** (1 / beta)
            return float(
                np.log(b)
                + np.log(m)
                + (b - 1.0) / b * np.log(t)
                + (m - 1.0) * np.log(-np.expm1(-t))
                - t
            )

        return float(np.log(float(alpha)) - self._t_expectation(log_f1, b, m))

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mu = params[-1]
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log1p(-y[~mask] ** (1.0 / mu)))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        i = len(params)
        mu = params[i - 1]
        return (1 - np.exp(-np.exp(y))) ** mu

    def unpack_rr(
        self, params: npt.NDArray, rr: str
    ) -> tuple[Boxable, Boxable, float]:
        if rr == "y":
            beta = params[0]
            alpha = np.exp(params[1] / -beta)
        elif rr == "x":
            beta = 1.0 / params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta, 1.0


ExpoWeibull: ExpoWeibull_ = ExpoWeibull_("ExpoWeibull")
