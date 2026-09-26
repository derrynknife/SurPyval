"""The fitted joint model from ``Copula.fit`` / ``Copula.from_params``."""

from typing import Any

import numpy as onp
import numpy.typing as npt

from surpyval.distribution import MultivariateDistribution
from surpyval.serialisation import SerialisableMixin, stamp_schema

_EPS = 1e-10


class CopulaModel(SerialisableMixin, MultivariateDistribution):
    """A fitted bivariate copula glued to two univariate margins.

    Attributes
    ----------
    copula : Copula
        The copula family.
    params : numpy.ndarray
        The fitted copula parameter(s) (empty for the independence copula).
    margins : list
        The fitted margin models (each exposes ``ff``/``df``/``qf``).
    k : int or None
        The number of parameters the fit estimated: the copula's plus those
        of every margin the fit estimated (all of them for ``how="MLE"``;
        under ``how="IFM"`` those passed as distributions, not a margin
        passed already fitted). ``None`` for ``from_params``.
    """

    def __init__(
        self,
        copula: Any,
        params: npt.ArrayLike,
        margins: Any,
        data: Any = None,
        how: str = "given",
        k: "int | None" = None,
    ) -> None:
        self.copula = copula
        self.params = onp.atleast_1d(onp.asarray(params, dtype=float))
        self.margins = list(margins)
        self.data = data
        self.method = how
        self.k = k
        # The fitted negative log-likelihood and weighted row count, computed
        # on first use from ``data`` (or restored by ``from_dict``, which has
        # no data).
        self._neg_ll: "float | None" = None
        self._n_obs: "float | None" = None

    # -- internal ---------------------------------------------------------
    def _uv(
        self, x: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x = onp.atleast_2d(onp.asarray(x, dtype=float))
        if x.shape[1] != 2:
            raise ValueError("x must have two columns (one per dimension)")
        u = onp.clip(self.margins[0].ff(x[:, 0]), _EPS, 1 - _EPS)
        v = onp.clip(self.margins[1].ff(x[:, 1]), _EPS, 1 - _EPS)
        return x, u, v

    # -- joint survival interface ----------------------------------------
    def cdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint CDF ``P(X_1 <= x_1, X_2 <= x_2)``."""
        _, u, v = self._uv(x)
        return onp.asarray(self.copula.cdf(u, v, *self.params))

    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint survival ``P(X_1 > x_1, X_2 > x_2)``."""
        _, u, v = self._uv(x)
        c = onp.asarray(self.copula.cdf(u, v, *self.params))
        return 1.0 - u - v + c

    def pdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Joint density ``c(F_1, F_2) f_1 f_2``."""
        x, u, v = self._uv(x)
        c = onp.asarray(self.copula.pdf(u, v, *self.params))
        f1 = onp.asarray(self.margins[0].df(x[:, 0]))
        f2 = onp.asarray(self.margins[1].df(x[:, 1]))
        return c * f1 * f2

    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """Alias of :meth:`cdf` for consistency with surpyval naming."""
        return self.cdf(x)

    def conditional_cdf(
        self, x: npt.ArrayLike, given_dim: int = 0
    ) -> npt.NDArray:
        """``P(X_other <= x_other | X_d = x_d)`` -- the copula h-function.

        ``given_dim=0`` conditions on the first series, giving
        :math:`P(X_2 \\le x_2 \\mid X_1 = x_1)`; ``given_dim=1``
        conditions on the second. Any other value raises ``ValueError``
        (it used to be read silently as ``1``).
        """
        if given_dim not in (0, 1):
            raise ValueError(
                f"given_dim must be 0 or 1 (the series conditioned on), got "
                f"{given_dim!r}."
            )
        x, u, v = self._uv(x)
        if given_dim == 0:
            return onp.asarray(self.copula.du(u, v, *self.params))
        return onp.asarray(self.copula.dv(u, v, *self.params))

    # -- sampling ---------------------------------------------------------
    def random(
        self,
        size: "int | tuple[int, ...]",
        random_state: "int | None" = None,
    ) -> npt.NDArray:
        """Draw correlated samples: an array of shape ``(size, 2)`` for an
        integer ``size``, one row per draw, or ``(*size, 2)`` for a tuple
        (the two series on the last axis)."""
        shape: tuple[int, ...] = (
            (int(size),)
            if isinstance(size, (int, onp.integer))
            else tuple(size)
        )
        count = int(onp.prod(shape))
        # Draw flat, then shape: the margins' quantile functions and
        # ``column_stack`` treat a 2-D draw as extra columns, so a (2, 3)
        # request came back as (2, 6).
        u, v = self.copula.sample_uv(count, self.params, random_state)
        x1 = onp.asarray(self.margins[0].qf(u), dtype=float).ravel()
        x2 = onp.asarray(self.margins[1].qf(v), dtype=float).ravel()
        return onp.column_stack([x1, x2]).reshape(shape + (2,))

    # -- dependence summaries --------------------------------------------
    def kendall_tau(self) -> float:
        """Kendall's rank correlation implied by the fitted copula."""
        return self.copula.kendall_tau(*self.params)

    def spearman_rho(self) -> float:
        """Spearman's rank correlation implied by the fitted copula."""
        return self.copula.spearman_rho(*self.params)

    def tail_dependence(self) -> tuple:
        """The lower and upper tail-dependence coefficients
        ``(lambda_L, lambda_U)`` of the fitted copula."""
        return self.copula.tail_dependence(*self.params)

    # -- likelihood and information criteria ------------------------------
    def _has_likelihood(self) -> bool:
        # Fitted to data, or restored from a dict that stored the fit's
        # likelihood; not built by ``from_params``.
        return self.k is not None and (
            self.data is not None or self._neg_ll is not None
        )

    def _fitted_k(self) -> int:
        """The estimated-parameter count; raises for a model with no
        likelihood."""
        if self.k is None or not self._has_likelihood():
            raise ValueError(
                "The log-likelihood is only available for a model fitted to "
                "data with `fit`, not one built with `from_params`."
            )
        return self.k

    def neg_ll(self) -> float:
        """
        The negative log-likelihood of the fitted model: the full joint
        likelihood of the data it was fitted to, with each row's censoring
        (right, left, interval, per series), its truncation and its count
        ``n``, and the margins' densities for the observed entries. Raises
        ``ValueError`` for a ``from_params`` model.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton
        >>> margins = [
        ...     Weibull.from_params([10, 2]),
        ...     Weibull.from_params([20, 3]),
        ... ]
        >>> X = Clayton.from_params([2.0], margins).random(300, random_state=0)
        >>> model = Clayton.fit(X, margins=[Weibull, Weibull])
        >>> round(model.neg_ll(), 3)
        1729.371
        >>> model.k, round(model.aic(), 3)
        (5, 3468.741)
        """
        return self._fit_stats()[0]

    def _fit_stats(self) -> tuple[float, float]:
        """``(neg_ll, n_obs)``: the negative log-likelihood and the weighted
        row count, computed from the data once (or restored by
        :meth:`from_dict`)."""
        self._fitted_k()
        if self._neg_ll is None or self._n_obs is None:
            dims = [
                self.copula._prepare_dim(
                    self.margins[d], *self.data.dimension(d)
                )
                for d in range(self.data.D)
            ]
            self._neg_ll = self.copula.neg_ll(self.params, dims, self.data.n)
            self._n_obs = float(onp.sum(self.data.n))
        return float(self._neg_ll), float(self._n_obs)

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood, ``-neg_ll()``."""
        return -self.neg_ll()

    def aic(self) -> float:
        """
        Akaike's information criterion, :math:`2k - 2\\ln L`, with ``k`` the
        number of estimated parameters (see the class docstring). Lower is
        better.
        """
        return 2.0 * self._fitted_k() + 2.0 * self.neg_ll()

    def bic(self) -> float:
        """
        The Bayesian information criterion, :math:`k \\ln N - 2\\ln L`, with
        ``N`` the number of joint observations (rows, weighted by ``n``).
        Lower is better.
        """
        neg_ll, n_obs = self._fit_stats()
        return float(self._fitted_k() * onp.log(n_obs) + 2.0 * neg_ll)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        """
        Serialise to a plain dictionary: the copula family, its
        parameter(s), the fit method and each margin's own ``to_dict``.
        The data is not stored, but for a fitted model the negative
        log-likelihood, parameter count and row count are, so the
        restored model still reports ``neg_ll``/``aic``/``bic``. Restore
        with :meth:`from_dict` or ``surpyval.from_dict``.
        """
        margins = []
        for m in self.margins:
            margins.append(m.to_dict() if hasattr(m, "to_dict") else None)
        out: dict = {
            "parameterization": "copula",
            "copula": self.copula.name,
            "params": self.params.tolist(),
            "how": self.method,
            "margins": margins,
        }
        if self._has_likelihood():
            out["neg_ll"], out["n_obs"] = self._fit_stats()
            out["k"] = self._fitted_k()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CopulaModel":
        """Rebuild a copula model from a :meth:`to_dict` dictionary."""
        import surpyval

        from .archimedean import Clayton, Frank, Gumbel, Independence
        from .elliptical import Gaussian

        families = {
            c.name: c for c in (Independence, Clayton, Gumbel, Frank, Gaussian)
        }
        copula_name = model_dict["copula"]
        if copula_name not in families:
            raise ValueError(
                f"Unknown copula family {copula_name!r}; expected one of "
                f"{sorted(families)}. A custom copula cannot be rebuilt "
                "from its name alone."
            )
        margins = []
        for i, m in enumerate(model_dict["margins"]):
            if m is None:
                raise ValueError(
                    f"Margin {i} was not serialisable (it has no `to_dict`), "
                    "so this copula model cannot be rebuilt."
                )
            margins.append(surpyval.from_dict(m))
        model = cls(
            families[copula_name],
            model_dict["params"],
            margins,
            data=None,
            how=model_dict.get("how", "given"),
            k=model_dict.get("k"),
        )
        # Dicts written before the likelihood was stored have none; such a
        # model (like a ``from_params`` one) has no likelihood to report.
        if "neg_ll" in model_dict:
            model._neg_ll = float(model_dict["neg_ll"])
            model._n_obs = float(model_dict["n_obs"])
        return model

    def __repr__(self) -> str:
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
