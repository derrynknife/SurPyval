"""Archimedean copula families: Independence, Clayton, Gumbel, Frank, Joe
and Ali-Mikhail-Haq (AMH).

All six have a closed-form CDF, and each supplies closed forms for its
partial derivatives (``du``, ``dv``) and density (``pdf``); Clayton, Gumbel,
Frank and Joe evaluate everything in log space so that strong dependence
neither overflows nor cancels. Each family converts an empirical Kendall's
tau into a starting parameter for the optimiser.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as onp
import numpy.typing as npt
from scipy.optimize import brentq

from surpyval import np
from surpyval.multivariate.parametric.copula.copula import _EPS, Copula
from surpyval.utils.rng import as_generator


class IndependenceCopula(Copula):
    """The independence copula ``C(u, v) = u v`` (no parameter).

    Fitting it fits the two margins separately; the joint survival is then
    the product of theirs. It is the baseline a dependent copula is
    compared against.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Independence
    >>> rng = np.random.default_rng(0)
    >>> x1 = 10 * rng.weibull(2, 100)
    >>> x2 = 20 * rng.weibull(3, 100)
    >>> model = Independence.fit([x1, x2], margins=[Weibull, Weibull])
    >>> [m.params.round(3) for m in model.margins]
    [array([10.763,  1.91 ]), array([20.847,  3.463])]
    >>> model.kendall_tau()
    0.0
    >>> model.sf([[5, 15]]).round(4)
    array([0.5762])
    """

    name = "Independence"
    bounds = ()
    parameter_names: list[str] = []

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
    parameter_names = ["theta"]
    dependence_limits = {1: "theta grows without bound"}

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
        return (float(2.0 ** (-1.0 / theta)), 0.0)

    def _init_theta(self, dims: list) -> npt.NDArray:
        tau = onp.clip(self._emp_tau(dims), 1e-3, 0.95)
        return onp.asarray([max(2.0 * tau / (1.0 - tau), 1e-2)])


class GumbelCopula(Copula):
    """Gumbel-Hougaard copula (upper-tail dependence), ``theta >= 1``."""

    name = "Gumbel"
    bounds = ((1, None),)
    parameter_names = ["theta"]
    closed_bounds = ("theta",)
    dependence_limits = {1: "theta grows without bound"}

    # With x = -log u, y = -log v and A = (x^theta + y^theta)^(1/theta),
    # C = exp(-A). Every primitive is formed from log x, log y and log A
    # (a log-sum-exp), never from x^theta itself: near the upper corner
    # x^theta underflowed to 0 (theta >= 20 at u = 1 - 1e-10, theta = 100
    # already at u = 0.98) and the autograd derivatives of the old
    # ``exp(-(x^theta + y^theta)^(1/theta))`` became 0 ** negative, an
    # infinite or NaN density and h-function.
    @staticmethod
    def _logs(u: Any, v: Any, theta: Any) -> tuple:
        log_x = np.log(-np.log(u))
        log_y = np.log(-np.log(v))
        # (a missing u or v gives NaN, without numpy's warning about it)
        with onp.errstate(invalid="ignore"):
            log_a = np.logaddexp(theta * log_x, theta * log_y) / theta
        return log_x, log_y, log_a

    # Named single parameter narrows the variadic base contract.
    def cdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return np.exp(-np.exp(self._logs(u, v, theta)[2]))

    # Named single parameter narrows the variadic base contract.
    def du(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        # dC/du = C A^(1 - theta) x^(theta - 1) / u
        log_x, _, log_a = self._logs(u, v, theta)
        return np.exp(
            -np.exp(log_a)
            + (1.0 - theta) * log_a
            + (theta - 1.0) * log_x
            - np.log(u)
        )

    # Named single parameter narrows the variadic base contract.
    def dv(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return self.du(v, u, theta)

    # Named single parameter narrows the variadic base contract.
    def pdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        # c = C (x y)^(theta - 1) A^(1 - 2 theta) (A + theta - 1) / (u v)
        log_x, log_y, log_a = self._logs(u, v, theta)
        a = np.exp(log_a)
        return np.exp(
            -a
            + (theta - 1.0) * (log_x + log_y)
            + (1.0 - 2.0 * theta) * log_a
            + np.log(a + theta - 1.0)
            - np.log(u)
            - np.log(v)
        )

    def kendall_tau(self, theta: float) -> float:  # type: ignore[override]
        return 1.0 - 1.0 / theta

    def tail_dependence(self, theta: float) -> tuple:  # type: ignore[override]
        return (0.0, float(2.0 - 2.0 ** (1.0 / theta)))

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
    parameter_names = ["theta"]
    dependence_limits = {
        1: "theta grows without bound",
        -1: "theta falls without bound",
    }

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
        rng = as_generator(random_state)
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


class JoeCopula(Copula):
    """Joe copula (upper-tail dependence), ``theta >= 1``.

    .. math::
        C(u, v) = 1 - \\left(\\bar u^\\theta + \\bar v^\\theta - \\bar
        u^\\theta \\bar v^\\theta\\right)^{1/\\theta},
        \\qquad \\bar u = 1 - u,

    the parameterisation of R's ``copula::joeCopula`` and of
    ``VineCopula`` (family 6). ``theta = 1`` is the independence copula,
    and the dependence grows with ``theta`` towards the comonotone copula.
    Like the Gumbel it has upper-tail dependence only,
    :math:`\\lambda_U = 2 - 2^{1/\\theta}`, but for a given Kendall's tau
    a stronger one: at tau = 0.5 the Joe has :math:`\\lambda_U = 0.71`
    (``theta = 2.86``), the Gumbel 0.59.

    With :math:`a = 1 - \\bar u^\\theta` and :math:`b = 1 - \\bar
    v^\\theta`, the bracket is :math:`A = 1 - a b`, and every primitive is
    formed from ``log A``: as ``log1p(-a b)`` near the lower corner (where
    ``a b`` is small and ``C`` is tiny), as a log-sum-exp of :math:`\\bar
    u^\\theta` and :math:`\\bar v^\\theta a` near the upper one.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Joe
    >>> margins = [
    ...     Weibull.from_params([10, 2]),
    ...     Weibull.from_params([20, 3]),
    ... ]
    >>> model = Joe.from_params([2.0], margins)
    >>> round(model.kendall_tau(), 4)
    0.3551
    >>> [round(x, 4) for x in model.tail_dependence()]
    [0.0, 0.5858]
    """

    name = "Joe"
    bounds = ((1, None),)
    parameter_names = ["theta"]
    closed_bounds = ("theta",)
    dependence_limits = {1: "theta grows without bound"}

    @staticmethod
    def _parts(u: Any, v: Any, theta: Any) -> tuple:
        """``(log ubar, log vbar, log b, a b, log A)`` (see the class
        docstring)."""
        log_ubar, log_vbar = onp.log1p(-u), onp.log1p(-v)
        a = -onp.expm1(theta * log_ubar)
        b = -onp.expm1(theta * log_vbar)
        ab = a * b
        # (a missing u or v gives NaN, without numpy's warning about it)
        with onp.errstate(divide="ignore", invalid="ignore"):
            near_upper = onp.logaddexp(
                theta * log_ubar, theta * log_vbar + onp.log(a)
            )
        # (the branch not taken is kept finite: ab may round to 1)
        log_A = onp.where(
            ab < 0.5, onp.log1p(-onp.minimum(ab, 0.5)), near_upper
        )
        return log_ubar, log_vbar, onp.log(b), ab, log_A

    # Named single parameter narrows the variadic base contract.
    def cdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        u, v, theta = _frank_args(u, v, theta)
        return -onp.expm1(self._parts(u, v, theta)[4] / theta)

    # Named single parameter narrows the variadic base contract.
    def du(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        # dC/du = ubar^(theta - 1) b A^(1/theta - 1)
        u, v, theta = _frank_args(u, v, theta)
        log_ubar, _, log_b, _, log_A = self._parts(u, v, theta)
        return onp.exp(
            (theta - 1.0) * log_ubar + log_b + (1.0 / theta - 1.0) * log_A
        )

    # Named single parameter narrows the variadic base contract.
    def dv(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return self.du(v, u, theta)

    # Named single parameter narrows the variadic base contract.
    def pdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        # c = (ubar vbar)^(theta - 1) A^(1/theta - 2) (theta - 1 + A)
        u, v, theta = _frank_args(u, v, theta)
        log_ubar, log_vbar, _, _, log_A = self._parts(u, v, theta)
        return onp.exp(
            (theta - 1.0) * (log_ubar + log_vbar)
            + (1.0 / theta - 2.0) * log_A
            + onp.log(theta - 1.0 + onp.exp(log_A))
        )

    def kendall_tau(self, theta: float) -> float:  # type: ignore[override]
        """Kendall's tau, :math:`1 - \\frac{2}{\\theta}\\,
        \\frac{\\psi(2 + \\delta) - \\psi(2)}{\\delta}` with
        :math:`\\delta = 2/\\theta - 1` and :math:`\\psi` the digamma
        function (the closed form of R's ``copula::tau`` for the Joe
        copula, written so that ``theta = 2`` is not 0/0)."""
        from scipy.special import polygamma, psi

        theta = float(theta)
        delta = 2.0 / theta - 1.0
        if abs(delta) < 1e-4:
            # Taylor series of the difference quotient about delta = 0
            ratio = (
                polygamma(1, 2.0)
                + polygamma(2, 2.0) * delta / 2.0
                + polygamma(3, 2.0) * delta**2 / 6.0
            )
        else:
            ratio = (psi(2.0 + delta) - psi(2.0)) / delta
        return float(1.0 - 2.0 / theta * ratio)

    def tail_dependence(self, theta: float) -> tuple:  # type: ignore[override]
        return (0.0, float(2.0 - 2.0 ** (1.0 / theta)))

    def _init_theta(self, dims: list) -> npt.NDArray:
        return onp.asarray(
            [_invert_tau(self, self._emp_tau(dims), 1.0, 1e4, 1.0 + 1e-2)]
        )


class AMHCopula(Copula):
    """Ali-Mikhail-Haq copula (weak dependence), ``-1 <= theta <= 1``.

    .. math::
        C(u, v) = \\frac{u v}{1 - \\theta (1 - u)(1 - v)},

    the parameterisation of R's ``copula::amhCopula``. ``theta = 0`` is the
    independence copula. The family only reaches weak dependence: Kendall's
    tau lies in :math:`[-0.1817, 1/3]` and Spearman's rho in
    :math:`[-0.2711, 0.4784]`, both bounds attained at ``theta = -1``
    and ``1``. A fit to data more strongly dependent than that runs to the
    bound (a valid copula) and returns it, without a warning, as a Clayton
    fit to negatively dependent data runs to independence; use the
    Clayton, Frank or Gaussian copula there. It has no tail dependence,
    except :math:`\\lambda_L = 1/2` at ``theta = 1`` (where it is the
    Clayton copula with ``theta = 1``).

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import AMH
    >>> margins = [
    ...     Weibull.from_params([10, 2]),
    ...     Weibull.from_params([20, 3]),
    ... ]
    >>> model = AMH.from_params([0.5], margins)
    >>> round(model.kendall_tau(), 4), round(model.spearman_rho(), 4)
    (0.1288, 0.1924)
    """

    name = "AMH"
    bounds = ((-1, 1),)
    parameter_names = ["theta"]
    closed_bounds = ("theta",)

    @staticmethod
    def _d(u: Any, v: Any, theta: float) -> Any:
        """``1 - theta (1 - u)(1 - v)``, written as ``(1 - theta) +
        theta (u + v - u v)`` so that it keeps its relative accuracy where
        it is small (``theta`` near 1 and ``u``, ``v`` near 0)."""
        return (1.0 - theta) + theta * (u + v * (1.0 - u))

    # Named single parameter narrows the variadic base contract.
    def cdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        u, v, theta = _frank_args(u, v, theta)
        return u * v / self._d(u, v, theta)

    # Named single parameter narrows the variadic base contract.
    def du(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        # dC/du = v (1 - theta (1 - v)) / D^2
        u, v, theta = _frank_args(u, v, theta)
        return v * ((1.0 - theta) + theta * v) / self._d(u, v, theta) ** 2

    # Named single parameter narrows the variadic base contract.
    def dv(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        return self.du(v, u, theta)

    # Named single parameter narrows the variadic base contract.
    def pdf(self, u: Any, v: Any, theta: Any) -> Any:  # type: ignore[override]
        # c = (1 + theta((1 + u)(1 + v) - 3) + theta^2 (1 - u)(1 - v)) / D^3,
        # its numerator regrouped as (1 - theta)(1 - theta + theta s) +
        # theta (1 + theta) q, s = u + v, q = u v, which is 2 u v (not a
        # difference of numbers near 1) at theta = 1.
        u, v, theta = _frank_args(u, v, theta)
        num = (1.0 - theta) * (1.0 - theta + theta * (u + v)) + theta * (
            1.0 + theta
        ) * (u * v)
        return num / self._d(u, v, theta) ** 3

    # Below |theta| = 0.1 the closed forms divide nearly cancelling terms
    # by theta^2; their power series (exact, from the series of log and
    # dilog) are used there, to 20 terms (the next is below 1e-21).
    _SERIES_THETA = 0.1

    def kendall_tau(self, theta: float) -> float:  # type: ignore[override]
        """Kendall's tau, :math:`1 - \\frac{2}{3 \\theta^2}(\\theta + (1 -
        \\theta)^2 \\log(1 - \\theta))` (Nelsen 2006, example 5.4), from
        -0.1817 at ``theta = -1`` to 1/3 at ``theta = 1``."""
        from scipy.special import xlogy

        theta = float(theta)
        if abs(theta) < self._SERIES_THETA:
            m = onp.arange(1, 21)
            return float(
                4.0 / 3.0 * onp.sum(theta**m / (m * (m + 1.0) * (m + 2.0)))
            )
        return float(
            1.0
            - 2.0
            * (theta + xlogy((1.0 - theta) ** 2, 1.0 - theta))
            / (3.0 * theta**2)
        )

    def spearman_rho(self, theta: float) -> float:  # type: ignore[override]
        """Spearman's rho, :math:`\\frac{12 (1 + \\theta)}{\\theta^2}
        \\mathrm{Li}_2(\\theta) - \\frac{24 (1 - \\theta)}{\\theta^2}
        \\log(1 - \\theta) - \\frac{3 (\\theta + 12)}{\\theta}` (Nelsen
        2006, exercise 5.10, with the dilogarithm
        :math:`\\mathrm{Li}_2`), from -0.2711 at ``theta = -1`` to 0.4784
        at ``theta = 1``."""
        from scipy.special import spence, xlogy

        theta = float(theta)
        if abs(theta) < self._SERIES_THETA:
            m = onp.arange(1, 21)
            return float(
                12.0 * onp.sum(theta**m / ((m + 1.0) ** 2 * (m + 2.0) ** 2))
            )
        # scipy's spence(1 - x) is the dilogarithm Li2(x)
        return float(
            12.0 * (1.0 + theta) / theta**2 * spence(1.0 - theta)
            - 24.0 * xlogy(1.0 - theta, 1.0 - theta) / theta**2
            - 3.0 * (theta + 12.0) / theta
        )

    def tail_dependence(self, theta: float) -> tuple:  # type: ignore[override]
        return (0.5 if float(theta) == 1.0 else 0.0, 0.0)

    def _init_theta(self, dims: list) -> npt.NDArray:
        # The fit starts strictly inside the bounds; data beyond the
        # family's range of tau start next to the nearer bound.
        return onp.asarray(
            [_invert_tau(self, self._emp_tau(dims), -1.0, 1.0, 0.0)]
        )


def _invert_tau(
    family: Copula, tau: float, low: float, high: float, default: float
) -> float:
    """The parameter in ``(low, high)`` whose Kendall's tau is ``tau``, for
    a family whose tau increases with its one parameter.

    ``tau`` is first moved inside the family's range, just short of the
    tau at each end (``low`` and ``high`` themselves may be valid
    parameters, a limit, or the start of a range that is never reached in
    floating point), so the root is strictly inside; ``default`` is
    returned if no root is bracketed.
    """
    span = high - low
    lo = low + 1e-3 * min(span, 1.0)
    hi = high - 1e-3 * min(span, 1.0)
    tau_lo, tau_hi = family.kendall_tau(lo), family.kendall_tau(hi)
    if not tau_lo < tau < tau_hi:
        tau = float(onp.clip(tau, tau_lo, tau_hi))
        return lo if tau == tau_lo else hi
    try:
        return float(brentq(lambda p: family.kendall_tau(p) - tau, lo, hi))
    except ValueError:
        return default


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
Joe = JoeCopula()
AMH = AMHCopula()
