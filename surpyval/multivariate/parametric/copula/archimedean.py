"""Archimedean copula families: Independence, Clayton, Gumbel, Frank.

All four have a closed-form CDF. Gumbel's is written in ``autograd.numpy``
so the default autograd partial derivatives (``du``, ``dv``, ``pdf``) are
exact; Independence, Clayton and Frank supply closed forms for them, and
Clayton and Frank evaluate everything in log space so that strong
dependence neither overflows nor cancels. Each family converts an empirical
Kendall's tau into a starting parameter for the optimiser.
"""

import math
from typing import Any

import numpy as onp
import numpy.typing as npt
from scipy.optimize import brentq

from surpyval import np
from surpyval.multivariate.parametric.copula.copula import _EPS, Copula


class IndependenceCopula(Copula):
    """The independence copula ``C(u, v) = u v`` (no parameter)."""

    name = "Independence"
    bounds = ()
    param_names = ()

    def cdf(self, u: Any, v: Any, *params: Any) -> Any:
        return u * v

    def du(self, u: Any, v: Any, *params: Any) -> Any:
        return np.asarray(v) * np.ones_like(np.asarray(u))

    def dv(self, u: Any, v: Any, *params: Any) -> Any:
        return np.asarray(u) * np.ones_like(np.asarray(v))

    def pdf(self, u: Any, v: Any, *params: Any) -> Any:
        return np.ones_like(np.asarray(u) * np.asarray(v))

    def kendall_tau(self, *params: float) -> float:
        return 0.0

    def spearman_rho(self, *params: float) -> float:
        return 0.0

    def fit(
        self,
        x: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        t: "npt.ArrayLike | None" = None,
        margins: Any = None,
        how: str = "IFM",
        xl: "npt.ArrayLike | None" = None,
        xr: "npt.ArrayLike | None" = None,
        init: "npt.ArrayLike | None" = None,
    ) -> Any:
        """
        Fit the margins only (the independence copula has no parameter);
        arguments as for :meth:`Copula.fit`, with ``how`` ignored (``init``,
        if given, must be empty).
        """
        # No parameter to estimate; only the margins are fitted.
        return super().fit(
            x,
            c=c,
            n=n,
            t=t,
            margins=margins,
            how="IFM",
            xl=xl,
            xr=xr,
            init=init,
        )

    def _fit_theta(
        self,
        margin_models: list,
        data: Any,
        init: "npt.NDArray | None" = None,
    ) -> npt.NDArray:
        return onp.asarray([], dtype=float)

    def _fit_joint(
        self,
        margins: Any,
        margin_models: list,
        data: Any,
        init: "npt.NDArray | None" = None,
    ) -> tuple:
        return onp.asarray([], dtype=float), margin_models


class ClaytonCopula(Copula):
    """Clayton copula (lower-tail dependence), ``theta > 0``."""

    name = "Clayton"
    bounds = ((0, None),)
    param_names = ("theta",)

    # Everything is computed through ``log(base)``, ``base = u ** -theta +
    # v ** -theta - 1``. For moderate exponents ``base - 1 =
    # expm1(-theta log u) + expm1(-theta log v)``: the direct form rounds
    # to exactly 1 once theta is below ~1e-16, so C became 1 and the
    # density 1 / (u v), a spurious likelihood maximum that negatively
    # dependent data ran the fit into; in log form every expression tends
    # to the independence copula. For large exponents (theta >= 31 at
    # u = 1e-10) expm1 overflowed and C collapsed to 0, so there the
    # largest exponent m is factored out: ``log(base) = m + log(e^(x - m)
    # + e^(y - m) - e^-m)``, a sum of at least 1 - e^-m.
    _LOG_BASE_SPLIT = 30.0

    @classmethod
    def _log_base(cls, u: Any, v: Any, theta: Any) -> Any:
        x = -theta * np.log(u)
        y = -theta * np.log(v)
        m = np.maximum(x, y)
        cap = cls._LOG_BASE_SPLIT
        # Each branch is evaluated everywhere; the exponents are capped in
        # the moderate one so it never overflows where it is not used.
        moderate = np.log1p(
            np.expm1(np.minimum(x, cap)) + np.expm1(np.minimum(y, cap))
        )
        large = m + np.log(np.exp(x - m) + np.exp(y - m) - np.exp(-m))
        return np.where(m > cap, large, moderate)

    # Named single parameter narrows the variadic base contract.
    def cdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return np.exp(-self._log_base(u, v, theta) / theta)

    # Named single parameter narrows the variadic base contract.
    def du(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return np.exp(
            (-theta - 1.0) * np.log(u)
            + (-1.0 / theta - 1.0) * self._log_base(u, v, theta)
        )

    # Named single parameter narrows the variadic base contract.
    def dv(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return self.du(v, u, theta)

    # Named single parameter narrows the variadic base contract.
    def pdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return np.exp(
            np.log1p(theta)
            + (-theta - 1.0) * (np.log(u) + np.log(v))
            + (-1.0 / theta - 2.0) * self._log_base(u, v, theta)
        )

    def kendall_tau(self, theta: float) -> float:  # type: ignore[override]
        return theta / (theta + 2.0)

    def tail_dependence(self, theta: float) -> tuple:  # type: ignore[override]
        return (2.0 ** (-1.0 / theta), 0.0)

    def _init_theta(self, dims: list) -> npt.NDArray:
        tau = onp.clip(self._emp_tau(dims), 1e-3, 0.95)
        return onp.asarray([max(2.0 * tau / (1.0 - tau), 1e-2)])


class GumbelCopula(Copula):
    """Gumbel-Hougaard copula (upper-tail dependence), ``theta >= 1``."""

    name = "Gumbel"
    bounds = ((1, None),)
    param_names = ("theta",)
    closed_bounds = ("theta",)

    # Named single parameter narrows the variadic base contract.
    def cdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        lu = (-np.log(u)) ** theta
        lv = (-np.log(v)) ** theta
        return np.exp(-((lu + lv) ** (1.0 / theta)))

    def kendall_tau(self, theta: float) -> float:  # type: ignore[override]
        return 1.0 - 1.0 / theta

    def tail_dependence(self, theta: float) -> tuple:  # type: ignore[override]
        return (0.0, 2.0 - 2.0 ** (1.0 / theta))

    def _init_theta(self, dims: list) -> npt.NDArray:
        tau = onp.clip(self._emp_tau(dims), 1e-3, 0.95)
        return onp.asarray([max(1.0 / (1.0 - tau), 1.0 + 1e-2)])


class FrankCopula(Copula):
    """Frank copula (symmetric, no tail dependence), ``theta != 0``.

    Every primitive is evaluated in log space from terms of
    :math:`C(u, v) = -\\frac{1}{\\theta} \\log\\left(1 +
    \\frac{(e^{-\\theta u} - 1)(e^{-\\theta v} - 1)}{e^{-\\theta} - 1}
    \\right)` that are sums of non-negative numbers, so they stay accurate
    for any :math:`\\theta`. The textbook form overflowed once
    :math:`e^{-\\theta}` rounded against 1 (:math:`\\theta \\gtrsim 37`,
    Kendall's tau 0.9): the CDF and density became infinite near the upper
    corner, the likelihood ``+inf`` and the sampler's second margin far
    from uniform. ``theta = 0`` is the independence copula.
    """

    name = "Frank"
    bounds = ((None, None),)
    param_names = ("theta",)

    # Named single parameter narrows the variadic base contract.
    def cdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        u, v, theta = _frank_args(u, v, theta)
        if theta == 0.0:
            return u * v
        with onp.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if theta > 0:
                # log(1 + A) with A = g(u) g(v) / g(1) in (-1, 0],
                # g(t) = expm1(-theta t). Near the lower corner A is small
                # and log1p(A) keeps the relative accuracy of the (tiny) C;
                # elsewhere 1 + A = S / (1 - e^-theta) with S the positive
                # sum of ``_frank_log_s``, which neither cancels nor
                # overflows.
                log_a = (
                    _log1mexp(theta * u)
                    + _log1mexp(theta * v)
                    - _log1mexp(theta)
                )
                small = log_a < onp.log(0.5)
                via_a = onp.log1p(-onp.exp(onp.minimum(log_a, 0.0)))
                via_s = _frank_log_s(u, v, theta) - _log1mexp(theta)
                log1p_a = onp.where(small, via_a, via_s)
            else:
                # theta < 0: A = expm1(eta u) expm1(eta v) / expm1(eta) > 0
                # (eta = -theta), so log(1 + A) = softplus(log A).
                eta = -theta
                log_a = _logexpm1(eta * u) + _logexpm1(eta * v)
                log1p_a = onp.logaddexp(0.0, log_a - _logexpm1(eta))
        return -log1p_a / theta

    # Named single parameter narrows the variadic base contract.
    def du(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        u, v, theta = _frank_args(u, v, theta)
        if theta == 0.0:
            return v * onp.ones_like(u)
        with onp.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return onp.exp(_frank_log_du(u, v, theta))

    # Named single parameter narrows the variadic base contract.
    def dv(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return self.du(v, u, theta)

    # Named single parameter narrows the variadic base contract.
    def pdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        u, v, theta = _frank_args(u, v, theta)
        if theta == 0.0:
            return onp.ones_like(u)
        with onp.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if theta > 0:
                # c = theta (1 - e^-theta) e^{-theta (u + v)} / S^2
                log_c = (
                    onp.log(theta)
                    + _log1mexp(theta)
                    - theta * (u + v)
                    - 2.0 * _frank_log_s(u, v, theta)
                )
            else:
                # c = eta expm1(eta) e^{eta (u + v)} / T^2 with
                # T = expm1(eta) + expm1(eta u) expm1(eta v)
                eta = -theta
                log_c = (
                    onp.log(eta)
                    + _logexpm1(eta)
                    + eta * (u + v)
                    - 2.0 * _frank_log_t(u, v, eta)
                )
        return onp.exp(log_c)

    def sample_uv(
        self,
        size: Any,
        params: Any,
        random_state: "int | None" = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Draw ``(u, v)`` pairs, inverting the h-function in closed form:
        given ``u`` and a uniform ``w``,

        .. math::
            v = -\\frac{1}{\\theta} \\log
            \\frac{w e^{-\\theta} + (1 - w) e^{-\\theta u}}
                  {w + (1 - w) e^{-\\theta u}},

        with both sums formed in log space (exact for any ``theta``)."""
        theta = _frank_theta(params[0])
        rng = onp.random.default_rng(random_state)
        u = rng.uniform(_EPS, 1 - _EPS, size=size)
        w = rng.uniform(_EPS, 1 - _EPS, size=size)
        if theta == 0.0:
            return u, w
        log_w, log_1mw = onp.log(w), onp.log1p(-w)
        num = onp.logaddexp(log_w - theta, log_1mw - theta * u)
        den = onp.logaddexp(log_w, log_1mw - theta * u)
        v = onp.clip(-(num - den) / theta, _EPS, 1 - _EPS)
        return u, v

    # Near theta = 0 the closed forms subtract nearly equal numbers (the
    # relative error reached 2% at theta = 1e-6), so their Taylor series
    # are used there; the next terms are below 1e-13 relative.
    _SERIES_THETA = 0.05

    def kendall_tau(self, theta: float) -> float:  # type: ignore[override]
        if abs(theta) < self._SERIES_THETA:
            return theta / 9 - theta**3 / 900 + theta**5 / 52920
        return 1.0 - 4.0 / theta * (1.0 - _debye(1, theta))

    def spearman_rho(self, theta: float) -> float:  # type: ignore[override]
        """Spearman's rho in closed form,
        :math:`1 - \\frac{12}{\\theta}(D_1(\\theta) - D_2(\\theta))` with
        :math:`D_k` the Debye functions."""
        if abs(theta) < self._SERIES_THETA:
            return theta / 6 - theta**3 / 450 + theta**5 / 23520
        return 1.0 - 12.0 / theta * (_debye(1, theta) - _debye(2, theta))

    def _init_theta(self, dims: list) -> npt.NDArray:
        tau = onp.clip(self._emp_tau(dims), -0.95, 0.95)
        if abs(tau) < 1e-3:
            return onp.asarray([1e-2])

        def gap(theta: float) -> float:
            return self.kendall_tau(theta) - tau

        # A Kendall's tau of +-0.95 needs |theta| of about 76; the bracket
        # of +-50 this used missed every tau above 0.92.
        try:
            theta = brentq(gap, -200, 200)
        except ValueError:
            theta = 2.0 if tau > 0 else -2.0
        if abs(theta) < 1e-2:
            theta = 1e-2 if tau >= 0 else -1e-2
        return onp.asarray([theta])


def _frank_theta(theta: Any) -> float:
    """The Frank parameter as a float (one value, like every family's)."""
    arr = onp.asarray(theta, dtype=float)
    if arr.size != 1:
        raise ValueError(f"theta must be a single value, got {arr.tolist()}")
    return float(arr.reshape(()))


def _frank_args(u: Any, v: Any, theta: Any) -> tuple:
    """``u`` and ``v`` as float arrays of their common shape, and theta."""
    u, v = onp.broadcast_arrays(
        onp.asarray(u, dtype=float), onp.asarray(v, dtype=float)
    )
    return u, v, _frank_theta(theta)


def _log1mexp(x: Any) -> Any:
    """``log(1 - exp(-x))`` for ``x > 0``, accurate for small and large x."""
    return onp.log(-onp.expm1(-x))


def _logexpm1(x: Any) -> Any:
    """``log(exp(x) - 1)`` for ``x > 0`` without overflow:
    ``expm1(x) = e^x (1 - e^-x)``."""
    return x + _log1mexp(x)


def _frank_log_s(u: Any, v: Any, theta: float) -> Any:
    """For ``theta > 0``, ``log S`` with ``S = e^{-theta u} + e^{-theta v}
    - e^{-theta (u + v)} - e^{-theta}``, which is ``(1 + A)(1 - e^-theta)``.

    ``S = (e^{-theta u} - e^{-theta}) + e^{-theta v} (1 - e^{-theta u})``:
    two non-negative terms, each formed in log space.
    """
    first = -theta + _logexpm1(theta * (1.0 - u))
    second = -theta * v + _log1mexp(theta * u)
    return onp.logaddexp(first, second)


def _frank_log_t(u: Any, v: Any, eta: float) -> Any:
    """For ``theta = -eta < 0``, ``log T`` with ``T = expm1(eta) +
    expm1(eta u) expm1(eta v)`` (every term positive)."""
    return onp.logaddexp(
        _logexpm1(eta), _logexpm1(eta * u) + _logexpm1(eta * v)
    )


def _frank_log_du(u: Any, v: Any, theta: float) -> Any:
    """``log dC/du`` of the Frank copula for ``theta != 0``."""
    if theta > 0:
        # du = e^{-theta u} (1 - e^{-theta v}) / S
        return -theta * u + _log1mexp(theta * v) - _frank_log_s(u, v, theta)
    # du = e^{eta u} expm1(eta v) / T
    eta = -theta
    return eta * u + _logexpm1(eta * v) - _frank_log_t(u, v, eta)


def _debye(k: int, theta: float) -> float:
    """Debye function ``D_k(t) = (k / t^k) int_0^t s^k / (e^s - 1) ds``
    (either sign of ``t``)."""
    from scipy.integrate import quad

    def integrand(s: float) -> float:
        # s^k / expm1(s), written so neither sign of s overflows.
        if s > 0:
            return s**k * math.exp(-s) / -math.expm1(-s)
        if s < 0:
            return s**k / math.expm1(s)
        return 1.0 if k == 1 else 0.0

    val, _ = quad(integrand, 0, theta, limit=200)
    return k * val / theta**k


Independence = IndependenceCopula()
Clayton = ClaytonCopula()
Gumbel = GumbelCopula()
Frank = FrankCopula()
