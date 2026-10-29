"""The posterior of a Bayesian fit: the parameter values the data and
the prior allow, with their probabilities.

Every engine hands back the same thing: a set of parameter points with
weights summing to one (a grid's points weighted by their density, a
conjugate posterior's equal-probability points, or a sampler's draws
with equal weights). Every summary of a Bayesian model is a weighted sum
or a weighted quantile over these points, so it is deterministic for the
grid and conjugate engines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from surpyval.bayes.engines.grid import lattice_quantile, weighted_quantile
from surpyval.bayes.transforms import Transform
from surpyval.univariate.parametric.parametric import draw_state

_CONJUGATE_FAMILIES = {
    "gamma": lambda a, b: stats.gamma(a, scale=1.0 / b),
    "beta": lambda a, b: stats.beta(a, b),
    "normal": lambda m, s: stats.norm(m, s),
}


class Posterior:
    """The posterior distribution of a model's parameters.

    Attributes
    ----------
    parameter_names : list of str
        Every parameter of the model, in the order of the columns of
        :attr:`points`: ``gamma`` (an offset), the distribution's
        parameters, ``lfp_p`` and ``f0``.
    free_names : list of str
        The parameters that are not fixed.
    kind : str
        ``"grid"``, ``"laplace"``, ``"conjugate"`` or ``"draws"``.
    points : numpy.ndarray
        The parameter points, shape ``(N, k)``.
    weights : numpy.ndarray
        Their probabilities, summing to 1.

    Examples
    --------
    >>> import surpyval as sp
    >>> model = sp.Weibull.fit_bayes(
    ...     [2100, 3400, 4050, 5000, 5000], c=[0, 0, 0, 1, 1],
    ...     priors={"beta": sp.prior.around(2, between=(1.5, 3))})
    >>> post = model.posterior
    >>> post.kind, post.parameter_names
    ('grid', ['alpha', 'beta'])
    >>> post.draws(3, random_state=1).shape
    (3, 2)
    """

    def __init__(
        self,
        parameter_names: list[str],
        free_names: list[str],
        points: npt.NDArray,
        weights: npt.NDArray,
        kind: str,
        *,
        transforms: list[Transform] | None = None,
        cell: npt.NDArray | None = None,
        exact: dict | None = None,
        shape: tuple | None = None,
    ) -> None:
        self.parameter_names = list(parameter_names)
        self.free_names = list(free_names)
        self.points = np.asarray(points, dtype=float)
        w = np.asarray(weights, dtype=float)
        total = float(w.sum())
        self.weights = w if abs(total - 1.0) < 1e-12 else w / total
        self.kind = kind
        self.transforms = transforms
        self.cell = None if cell is None else np.asarray(cell, dtype=float)
        self.exact = exact
        #: How a grid's points are laid out, ``(lattices, m, ..., m)``.
        self.shape = None if shape is None else tuple(int(v) for v in shape)
        #: log density of the free parameters (natural scale), set by the
        #: model where it is known exactly (grid and conjugate).
        self.log_density_fn: Callable[[npt.NDArray], npt.NDArray] | None
        self.log_density_fn = None

    def __repr__(self) -> str:
        return (
            f"Posterior({self.kind}, {len(self.weights)} points, "
            f"parameters {', '.join(self.free_names)})"
        )

    # -- helpers ----------------------------------------------------------
    def _col(self, name: str) -> int:
        if name not in self.parameter_names:
            raise ValueError(
                f"No parameter {name!r}; the parameters are "
                f"{', '.join(self.parameter_names)}."
            )
        return self.parameter_names.index(name)

    def _exact(self, name: str) -> Any:
        if self.exact is not None and self.exact.get("name") == name:
            return _CONJUGATE_FAMILIES[self.exact["family"]](
                *self.exact["params"]
            )
        return None

    @property
    def free_idx(self) -> list[int]:
        return [self.parameter_names.index(n) for n in self.free_names]

    def quantile(self, name: str, q: npt.ArrayLike) -> npt.NDArray:
        """The posterior quantiles ``q`` of the parameter ``name``."""
        exact = self._exact(name)
        q = np.asarray(q, dtype=float)
        if exact is not None:
            return np.asarray(exact.ppf(q), dtype=float)
        col = self.points[:, self._col(name)]
        if name not in self.free_names:
            return np.full(q.shape, col[0])
        return self.wquantile(col, q)

    def wquantile(self, values: npt.NDArray, q: npt.ArrayLike) -> Any:
        """Posterior quantiles ``q`` of a function's ``values`` at the
        points (``(N,)`` or ``(N, s)``): on a grid each point's mass is
        spread over its cell (``lattice_quantile``); a non-finite value
        is taken as infinite."""
        v = np.asarray(values, dtype=float)
        v = np.where(np.isnan(v), np.inf, v)
        if self.shape is not None and int(np.prod(self.shape)) == len(v):
            return lattice_quantile(v, self.weights, q, self.shape)
        return weighted_quantile(v, self.weights, q)

    def mean(self, name: str) -> float:
        """The posterior mean of the parameter ``name``."""
        exact = self._exact(name)
        if exact is not None:
            return float(exact.mean())
        return float(self.weights @ self.points[:, self._col(name)])

    def median(self) -> npt.NDArray:
        """Every parameter's posterior median, in :attr:`parameter_names`
        order."""
        return np.array(
            [float(self.quantile(n, 0.5)) for n in self.parameter_names]
        )

    def expectation(self, values: npt.NDArray) -> npt.NDArray:
        """The posterior mean of a function's values at the points
        (``(N, ...)``).

        Each value's sum runs in the same order whatever its place among
        the others (a contiguous row per value, summed alone), so a
        function evaluated at reordered query points gives the same
        numbers reordered, to the last bit. A matrix product
        (``weights @ values``) does not: BLAS blocks the columns, and a
        column's summation order depends on where it falls.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.bayes.posterior import Posterior
        >>> post = Posterior(["a"], ["a"], np.zeros((3, 1)),
        ...                  np.array([0.2, 0.3, 0.5]), "grid")
        >>> post.expectation(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        array([3.6, 4.6])
        """
        v = np.asarray(values, dtype=float)
        w = self.weights
        rows = (v.reshape(len(w), -1) * w[:, None]).T
        out = np.ascontiguousarray(rows).sum(axis=1)
        return out.reshape(v.shape[1:])

    # -- draws --------------------------------------------------------------
    def draws(self, k: int = 4000, random_state: Any = None) -> npt.NDArray:
        """
        ``k`` draws from the posterior, shape ``(k, len(parameter_names))``.

        A grid posterior draws cells by their weight and a point uniformly
        within each cell; a conjugate posterior draws exactly; a posterior
        from draws resamples them.

        Parameters
        ----------
        k : int, optional
            How many draws. Defaults to 4000.
        random_state : int or numpy.random.Generator, optional
            The seed rule of every SurPyval draw: ``None`` uses numpy's
            global stream, an int its own.
        """
        k = int(k)
        if k < 0:
            raise ValueError(f"k must be a non-negative count; got {k}.")
        if isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = draw_state(random_state)
        if rng is None:
            # numpy's global stream, so np.random.seed reproduces it
            rng = np.random.mtrand._rand
        exact = self.exact
        if exact is not None:
            out = np.repeat(self.points[:1], k, axis=0)
            col = self._col(exact["name"])
            out[:, col] = self._exact(exact["name"]).ppf(rng.uniform(size=k))
            return out
        idx = rng.choice(len(self.weights), size=k, p=self.weights)
        out = self.points[idx].copy()
        if self.cell is not None and self.transforms is not None:
            d = len(self.free_names)
            u = rng.uniform(-0.5, 0.5, size=(k, d))
            phi = np.stack(
                [
                    t.to_phi(out[:, i])
                    for t, i in zip(self.transforms, self.free_idx)
                ],
                axis=1,
            )
            cell = self.cell if self.cell.ndim == 2 else self.cell[idx]
            if cell.ndim == 2:
                phi = phi + u @ cell.T
            else:
                phi = phi + np.einsum("kij,kj->ki", cell, u)
            for j, (t, i) in enumerate(zip(self.transforms, self.free_idx)):
                out[:, i] = t.to_theta(phi[:, j])
        return out

    def to_dataframe(self, k: int = 4000, random_state: Any = None) -> Any:
        """``k`` draws (:meth:`draws`) as a ``pandas.DataFrame``, one column
        per parameter."""
        import pandas as pd

        return pd.DataFrame(
            self.draws(k, random_state), columns=self.parameter_names
        )

    def to_arviz(self, k: int = 4000, random_state: Any = None) -> Any:
        """The draws as an ArviZ ``InferenceData``, for its plots and
        diagnostics (needs ArviZ, which SurPyval does not require)."""
        try:
            import arviz as az  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "to_arviz needs ArviZ: pip install arviz."
            ) from None
        d = self.draws(k, random_state)
        return az.from_dict(
            posterior={
                n: d[None, :, i]
                for i, n in enumerate(self.parameter_names)
                if n in self.free_names
            }
        )

    # -- densities ------------------------------------------------------------
    def density(self, points: npt.ArrayLike) -> npt.NDArray:
        """
        The joint posterior density of the free parameters at ``points``
        (shape ``(m, d)``, the free parameters in :attr:`free_names`
        order, on their own scales). Exact for a grid or conjugate
        posterior: the prior times the likelihood, divided by the
        evidence the grid computed.
        """
        if self.log_density_fn is None:
            raise ValueError(
                f"A posterior from {self.kind} has no density; use "
                "marginal(name) for a smoothed estimate."
            )
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if p.shape[-1] != len(self.free_names):
            p = p.reshape(-1, len(self.free_names))
        with np.errstate(all="ignore"):
            return np.exp(self.log_density_fn(p))

    def marginal(
        self, name: str, values: npt.ArrayLike | None = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        The posterior density of one parameter, ``(values, density)``.

        Exact for a one-parameter grid or conjugate posterior; with more
        free parameters, a kernel estimate from the weighted points (on
        the parameter's unconstrained scale, carried back to its own).
        ``values`` defaults to 201 points from its 0.05% to its 99.95%
        point.
        """
        if name not in self.free_names:
            raise ValueError(
                f"{name} is not a free parameter "
                f"({', '.join(self.free_names)})."
            )
        if values is None:
            lo, hi = self.quantile(name, [0.0005, 0.9995])
            values = np.linspace(lo, hi, 201)
        v = np.asarray(values, dtype=float)
        exact = self._exact(name)
        if exact is not None:
            return v, exact.pdf(v)
        if len(self.free_names) == 1 and self.log_density_fn is not None:
            return v, self.density(v[:, None])
        j = self.free_names.index(name)
        if self.transforms is not None:
            t = self.transforms[j]
        else:
            col = self.points[:, self._col(name)]
            t = Transform("log") if np.all(col > 0) else Transform("identity")
        col = self.points[:, self._col(name)]
        phi = t.to_phi(col)
        w = self.weights
        m = float(w @ phi)
        sd = float(np.sqrt(w @ (phi - m) ** 2))
        n_eff = 1.0 / float(np.sum(w**2))
        bw = 1.06 * sd * min(n_eff, 1e4) ** (-0.2)
        if self.cell is not None:
            c = (
                self.cell
                if self.cell.ndim == 2
                else np.einsum("n,nij->ij", w, self.cell)
            )
            bw = max(bw, float(np.max(np.abs(c[j]))))
        with np.errstate(all="ignore"):
            pv = t.to_phi(v)
            dens_phi = np.array(
                [w @ stats.norm.pdf(p, phi, bw) for p in np.atleast_1d(pv)]
            )
            dens = dens_phi * np.exp(-t.log_jacobian(pv))
        return v, np.where(np.isfinite(dens), dens, 0.0)

    # -- saving ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "parameter_names": self.parameter_names,
            "free_names": self.free_names,
            "kind": self.kind,
            "points": self.points.tolist(),
            "weights": self.weights.tolist(),
            "transforms": (
                None
                if self.transforms is None
                else [t.to_dict() for t in self.transforms]
            ),
            "cell": None if self.cell is None else self.cell.tolist(),
            "exact": self.exact,
            "shape": None if self.shape is None else list(self.shape),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Posterior:
        transforms = d.get("transforms")
        return cls(
            d["parameter_names"],
            d["free_names"],
            np.asarray(d["points"], dtype=float),
            np.asarray(d["weights"], dtype=float),
            d["kind"],
            transforms=(
                None
                if transforms is None
                else [Transform.from_dict(t) for t in transforms]
            ),
            cell=None if d.get("cell") is None else np.asarray(d["cell"]),
            exact=d.get("exact"),
            shape=d.get("shape"),
        )
