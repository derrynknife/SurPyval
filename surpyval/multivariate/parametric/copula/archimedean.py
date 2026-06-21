"""Archimedean copula families: Independence, Clayton, Gumbel, Frank.

All four have a closed-form CDF written in ``autograd.numpy`` so the
default autograd partial derivatives (``du``, ``dv``, ``pdf``) are exact.
Clayton additionally supplies closed forms for speed. Each family converts
an empirical Kendall's tau into a starting parameter for the optimiser.
"""

import numpy as onp
from scipy.optimize import brentq

from surpyval import np
from surpyval.multivariate.parametric.copula.copula import Copula


class IndependenceCopula(Copula):
    """The independence copula ``C(u, v) = u v`` (no parameter)."""

    name = "Independence"
    bounds = ()
    param_names = ()

    def cdf(self, u, v, *params):
        return u * v

    def du(self, u, v, *params):
        return np.asarray(v) * np.ones_like(np.asarray(u))

    def dv(self, u, v, *params):
        return np.asarray(u) * np.ones_like(np.asarray(v))

    def pdf(self, u, v, *params):
        return np.ones_like(np.asarray(u) * np.asarray(v))

    def kendall_tau(self, *params):
        return 0.0

    def spearman_rho(self, *params):
        return 0.0

    def fit(
        self,
        x,
        c=None,
        n=None,
        t=None,
        margins=None,
        how="IFM",
        xl=None,
        xr=None,
    ):
        # No parameter to estimate; only the margins are fitted.
        return super().fit(
            x, c=c, n=n, t=t, margins=margins, how="IFM", xl=xl, xr=xr
        )

    def _fit_theta(self, margin_models, data):
        return onp.asarray([], dtype=float)

    def _fit_joint(self, margins, margin_models, data):
        return onp.asarray([], dtype=float), margin_models


class ClaytonCopula(Copula):
    """Clayton copula (lower-tail dependence), ``theta > 0``."""

    name = "Clayton"
    bounds = ((0, None),)
    param_names = ("theta",)

    def cdf(self, u, v, theta):
        return (u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta)

    def du(self, u, v, theta):
        base = u ** (-theta) + v ** (-theta) - 1.0
        return u ** (-theta - 1.0) * base ** (-1.0 / theta - 1.0)

    def dv(self, u, v, theta):
        return self.du(v, u, theta)

    def pdf(self, u, v, theta):
        base = u ** (-theta) + v ** (-theta) - 1.0
        return (
            (1.0 + theta)
            * (u * v) ** (-theta - 1.0)
            * base ** (-1.0 / theta - 2.0)
        )

    def kendall_tau(self, theta):
        return theta / (theta + 2.0)

    def tail_dependence(self, theta):
        return (2.0 ** (-1.0 / theta), 0.0)

    def _init_theta(self, dims):
        tau = onp.clip(self._emp_tau(dims), 1e-3, 0.95)
        return onp.asarray([max(2.0 * tau / (1.0 - tau), 1e-2)])


class GumbelCopula(Copula):
    """Gumbel-Hougaard copula (upper-tail dependence), ``theta >= 1``."""

    name = "Gumbel"
    bounds = ((1, None),)
    param_names = ("theta",)

    def cdf(self, u, v, theta):
        lu = (-np.log(u)) ** theta
        lv = (-np.log(v)) ** theta
        return np.exp(-((lu + lv) ** (1.0 / theta)))

    def kendall_tau(self, theta):
        return 1.0 - 1.0 / theta

    def tail_dependence(self, theta):
        return (0.0, 2.0 - 2.0 ** (1.0 / theta))

    def _init_theta(self, dims):
        tau = onp.clip(self._emp_tau(dims), 1e-3, 0.95)
        return onp.asarray([max(1.0 / (1.0 - tau), 1.0 + 1e-2)])


class FrankCopula(Copula):
    """Frank copula (symmetric, no tail dependence), ``theta != 0``."""

    name = "Frank"
    bounds = ((None, None),)
    param_names = ("theta",)

    def cdf(self, u, v, theta):
        # Guard the removable singularity at theta -> 0 (independence).
        theta = np.where(np.abs(theta) < 1e-8, 1e-8, theta)
        num = (np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0)
        return -1.0 / theta * np.log(1.0 + num / (np.exp(-theta) - 1.0))

    def kendall_tau(self, theta):
        if abs(theta) < 1e-8:
            return 0.0
        return 1.0 - 4.0 / theta * (1.0 - _debye1(theta))

    def _init_theta(self, dims):
        tau = onp.clip(self._emp_tau(dims), -0.95, 0.95)
        if abs(tau) < 1e-3:
            return onp.asarray([1e-2])

        def gap(theta):
            return self.kendall_tau(theta) - tau

        try:
            theta = brentq(gap, -50, 50)
        except ValueError:
            theta = 2.0 if tau > 0 else -2.0
        if abs(theta) < 1e-2:
            theta = 1e-2 if tau >= 0 else -1e-2
        return onp.asarray([theta])


def _debye1(theta):
    """First Debye function ``D_1(t) = (1/t) int_0^t s/(e^s-1) ds``."""
    from scipy.integrate import quad

    val, _ = quad(lambda s: s / onp.expm1(s), 0, theta)
    return val / theta


Independence = IndependenceCopula()
Clayton = ClaytonCopula()
Gumbel = GumbelCopula()
Frank = FrankCopula()
