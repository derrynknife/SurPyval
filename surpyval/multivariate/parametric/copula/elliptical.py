"""Elliptical copulas: the Gaussian copula.

The Gaussian copula's CDF is the bivariate-normal CDF, which has no
``autograd`` path, so every primitive here is supplied in closed form using
``scipy``. The single parameter is the correlation ``rho in (-1, 1)``; it is
optimised on a ``tanh`` reparameterisation so the optimiser stays in range.
"""

import numpy as onp
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

from surpyval.multivariate.parametric.copula.copula import Copula

_RHO_MAX = 0.9999


class GaussianCopula(Copula):
    """Gaussian copula, ``rho in (-1, 1)`` (no tail dependence)."""

    name = "Gaussian"
    bounds = ((-1, 1),)
    param_names = ("rho",)

    @staticmethod
    def _clip_rho(rho):
        return float(onp.clip(rho, -_RHO_MAX, _RHO_MAX))

    def cdf(self, u, v, rho):
        rho = self._clip_rho(rho)
        a = ndtri(onp.clip(onp.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        b = ndtri(onp.clip(onp.asarray(v, dtype=float), 1e-12, 1 - 1e-12))
        pts = onp.stack([onp.ravel(a), onp.ravel(b)], axis=-1)
        cov = [[1.0, rho], [rho, 1.0]]
        out = multivariate_normal.cdf(pts, mean=[0.0, 0.0], cov=cov)
        return onp.asarray(out).reshape(onp.asarray(a).shape)

    def du(self, u, v, rho):
        rho = self._clip_rho(rho)
        a = ndtri(onp.clip(onp.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        b = ndtri(onp.clip(onp.asarray(v, dtype=float), 1e-12, 1 - 1e-12))
        return ndtr((b - rho * a) / onp.sqrt(1.0 - rho**2))

    def dv(self, u, v, rho):
        return self.du(v, u, rho)

    def pdf(self, u, v, rho):
        rho = self._clip_rho(rho)
        a = ndtri(onp.clip(onp.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        b = ndtri(onp.clip(onp.asarray(v, dtype=float), 1e-12, 1 - 1e-12))
        denom = 1.0 - rho**2
        quad = (rho**2 * (a**2 + b**2) - 2.0 * rho * a * b) / (2.0 * denom)
        return onp.exp(-quad) / onp.sqrt(denom)

    def kendall_tau(self, rho):
        return 2.0 / onp.pi * onp.arcsin(self._clip_rho(rho))

    def spearman_rho(self, rho):
        return 6.0 / onp.pi * onp.arcsin(self._clip_rho(rho) / 2.0)

    def sample_uv(self, size, params, random_state=None):
        rho = self._clip_rho(params[0])
        rng = onp.random.default_rng(random_state)
        z1 = rng.standard_normal(size)
        z2 = rng.standard_normal(size)
        z2 = rho * z1 + onp.sqrt(1.0 - rho**2) * z2
        return ndtr(z1), ndtr(z2)

    def _bounds_transforms(self):
        # tanh keeps rho strictly inside (-1, 1) during optimisation.
        def to_unbounded(params):
            return onp.arctanh(onp.clip(params, -_RHO_MAX, _RHO_MAX))

        def to_bounded(phi):
            return onp.tanh(onp.asarray(phi, dtype=float))

        return to_unbounded, to_bounded

    def _init_theta(self, dims):
        return onp.asarray([onp.sin(onp.pi / 2.0 * self._emp_tau(dims))])


Gaussian = GaussianCopula()
