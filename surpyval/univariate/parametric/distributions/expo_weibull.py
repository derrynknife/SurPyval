from __future__ import annotations

from typing import Callable

import numpy.typing as npt
from scipy import integrate

from surpyval import np
from surpyval.univariate import parametric as para
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
    _offset_start,
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
            param_names=["alpha", "beta", "mu"],
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
        and the density need. Points at or below 0 are evaluated at 1 (the
        caller replaces them), so that no branch of any ``np.where`` sees
        a log of 0.
        """
        x_pos = np.where(x > 0, x, 1.0)
        log_x = np.log(x_pos)
        log_t = beta * (log_x - np.log(alpha))
        with np.errstate(over="ignore"):
            t = np.exp(log_t)
        log_g, _ = _log1mexp(t, log_t)
        log_ff = mu * log_g
        out = {"log_x": log_x, "t": t, "log_g": log_g, "log_ff": log_ff}
        if not right:
            return out
        log_nl = _log_neg_log1mexp(t, log_g)
        log_sf, ratio_r = _log1mexp(-log_ff, np.log(mu) + log_nl)
        out.update(log_nl=log_nl, log_sf=log_sf, ratio_r=ratio_r)
        return out

    @staticmethod
    def _support(x: Numeric, inside: Boxable, at_zero: Boxable) -> Boxable:
        """``inside`` for x > 0, ``at_zero`` at 0 and NaN below."""
        # autograd's ``where`` does not unbroadcast its gradient, so a
        # parameter-dependent ``at_zero`` must have the full shape.
        at_zero = at_zero + np.zeros_like(inside)
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
        return self._support(x, inside, 1.0)

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
        return self._support(x, np.exp(p["log_ff"]), 0.0)

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
        log_hf = (
            np.log(beta)
            + (beta - 1) * p["log_x"]
            - beta * np.log(alpha)
            + (mu - 1) * p["log_g"]
            - p["ratio_r"]
            - log_q
        )
        with np.errstate(over="ignore"):
            inside = np.exp(log_hf)
        return self._support(x, inside, self._df_at_zero(alpha, beta, mu))

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
        with np.errstate(over="ignore", under="ignore"):
            return alpha * np.exp(log_t / beta)

    def log_df(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        p = self._log_forms(x, alpha, beta, mu, right=False)
        inside = (
            np.log(beta)
            + np.log(mu)
            + (beta - 1) * p["log_x"]
            - beta * np.log(alpha)
            + (mu - 1) * p["log_g"]
            - p["t"]
        )
        bm = beta * mu
        at_zero = np.where(
            bm < 1, np.inf, np.where(bm == 1, -np.log(alpha), -np.inf)
        )
        return self._support(x, inside, at_zero)

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

    def log_ff(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        p = self._log_forms(x, alpha, beta, mu, right=False)
        return self._support(x, p["log_ff"], -np.inf)

    def log_sf(
        self, x: Numeric, alpha: Boxable, beta: Boxable, mu: Boxable
    ) -> Boxable:
        # log(1 - e^-r) with r = -ln F carried as ln r, which in the right
        # tail is ln(mu) - t: finite after e^-t underflows, where the log
        # of the survival function itself is -inf (#257, #436).
        p = self._log_forms(x, alpha, beta, mu)
        return self._support(x, p["log_sf"], 0.0)

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
        integral is taken by quadrature, as ``entropy`` does for the same
        reason, on the distribution's own scale: with
        :math:`t = (x/\alpha)^{\beta}`,

        .. math::
            E[X^{m}] = \alpha^{m} \int_{0}^{\infty} t^{m/\beta}\,
            \mu (1 - e^{-t})^{\mu - 1} e^{-t}\, dt ,

        whose integrand does not depend on :math:`\alpha`, so the result
        is equally accurate at any scale.

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
        8.598425613605164
        """

        a, b, u = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (alpha, beta, mu))
        )
        out = np.empty(a.shape)
        for i in np.ndindex(*a.shape):
            m_b = float(m) / b[i]
            out[i] = a[i] ** m * self._t_expectation(
                lambda t: t**m_b, b[i], u[i]
            )
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def _t_expectation(
        g: Callable[[float], float], beta: Boxable, mu: Boxable
    ) -> float:
        r"""
        :math:`E[g(T)]` for :math:`T = (X/\alpha)^{\beta}`, whose density
        :math:`\mu (1 - e^{-t})^{\mu - 1} e^{-t}` is free of
        :math:`\alpha` (and of :math:`\beta`).

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
        2.8422622081888997
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
