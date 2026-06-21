"""The fitted joint model from ``Copula.fit`` / ``Copula.from_params``."""

import numpy as onp

from surpyval.distribution import MultivariateDistribution

_EPS = 1e-10


class CopulaModel(MultivariateDistribution):
    """A fitted bivariate copula glued to two univariate margins.

    Attributes
    ----------
    copula : Copula
        The copula family.
    params : numpy.ndarray
        The fitted copula parameter(s) (empty for the independence copula).
    margins : list
        The fitted margin models (each exposes ``ff``/``df``/``qf``).
    """

    def __init__(self, copula, params, margins, data=None, how="given"):
        self.copula = copula
        self.params = onp.atleast_1d(onp.asarray(params, dtype=float))
        self.margins = list(margins)
        self.data = data
        self.method = how

    # -- internal ---------------------------------------------------------
    def _uv(self, x):
        x = onp.atleast_2d(onp.asarray(x, dtype=float))
        if x.shape[1] != 2:
            raise ValueError("x must have two columns (one per dimension)")
        u = onp.clip(self.margins[0].ff(x[:, 0]), _EPS, 1 - _EPS)
        v = onp.clip(self.margins[1].ff(x[:, 1]), _EPS, 1 - _EPS)
        return x, u, v

    # -- joint survival interface ----------------------------------------
    def cdf(self, x):
        """Joint CDF ``P(X_1 <= x_1, X_2 <= x_2)``."""
        _, u, v = self._uv(x)
        return onp.asarray(self.copula.cdf(u, v, *self.params))

    def sf(self, x):
        """Joint survival ``P(X_1 > x_1, X_2 > x_2)``."""
        _, u, v = self._uv(x)
        c = onp.asarray(self.copula.cdf(u, v, *self.params))
        return 1.0 - u - v + c

    def pdf(self, x):
        """Joint density ``c(F_1, F_2) f_1 f_2``."""
        x, u, v = self._uv(x)
        c = onp.asarray(self.copula.pdf(u, v, *self.params))
        f1 = onp.asarray(self.margins[0].df(x[:, 0]))
        f2 = onp.asarray(self.margins[1].df(x[:, 1]))
        return c * f1 * f2

    def ff(self, x):
        """Alias of :meth:`cdf` for consistency with surpyval naming."""
        return self.cdf(x)

    def conditional_cdf(self, x, given_dim=0):
        """``P(X_other <= x_other | X_d = x_d)`` -- the copula h-function."""
        x, u, v = self._uv(x)
        if given_dim == 0:
            return onp.asarray(self.copula.du(u, v, *self.params))
        return onp.asarray(self.copula.dv(u, v, *self.params))

    # -- sampling ---------------------------------------------------------
    def random(self, size, random_state=None):
        """Draw correlated samples; returns an array of shape ``(size, 2)``."""
        u, v = self.copula.sample_uv(size, self.params, random_state)
        x1 = onp.asarray(self.margins[0].qf(u))
        x2 = onp.asarray(self.margins[1].qf(v))
        return onp.column_stack([x1, x2])

    # -- dependence summaries --------------------------------------------
    def kendall_tau(self):
        return self.copula.kendall_tau(*self.params)

    def spearman_rho(self):
        return self.copula.spearman_rho(*self.params)

    def tail_dependence(self):
        return self.copula.tail_dependence(*self.params)

    # -- serialisation ----------------------------------------------------
    def to_dict(self):
        margins = []
        for m in self.margins:
            margins.append(m.to_dict() if hasattr(m, "to_dict") else None)
        return {
            "parameterization": "copula",
            "copula": self.copula.name,
            "params": self.params.tolist(),
            "how": self.method,
            "margins": margins,
        }

    def __repr__(self):
        param_str = ", ".join(
            f"{n}={p:.4g}"
            for n, p in zip(self.copula.param_names, self.params)
        )
        margin_names = [
            getattr(getattr(m, "dist", m), "name", "?") for m in self.margins
        ]
        return (
            "Copula SurPyval Model"
            "\n====================="
            f"\nCopula    : {self.copula.name}"
            f"\nParameters: {param_str if param_str else '(none)'}"
            f"\nMargins   : {', '.join(margin_names)}"
            f"\nFitted by : {self.method}"
        )
