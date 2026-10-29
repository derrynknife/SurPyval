"""The fitted distributional regression model."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.serialisation import stamp_schema
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.validation import warn_outside_unit_interval

from .._covariate_link import CovariateLink
from .._kinds import DISTRIBUTIONAL
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import (
    check_covariate_width,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd


class DistributionalRegressionModel(ParametricRegressionModel):
    """
    A fitted distributional regression (:func:`Distributional`): every
    parameter of the distribution has its own linear predictor and link,

    .. math::
        g_k\\bigl(\\theta_k(z)\\bigr) = g_k(\\theta_k) + x_k(z)' b_k,

    so each row of covariates has its own distribution, here the
    distribution's functions at that row's parameters.

    ``params`` are the distribution's parameters at ``Z = 0`` (at
    ``center`` for a fit with ``center=True``), then each parameter's
    coefficients on its link scale, named ``<parameter>_<column>``:
    ``beta_load = 0.2`` with a log link multiplies the shape by
    ``e^0.2`` per unit of load. ``covariate_positions`` holds each
    parameter's columns of ``Z`` and ``links`` its link.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import WeibullDistributional
    >>> rng = np.random.default_rng(3)
    >>> load = rng.uniform(0, 1, 300)
    >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
    >>> model = WeibullDistributional.fit(x, load)
    >>> model.parameters_at([[0.0], [1.0]]).round(2)
    array([[9.39, 1.39],
           [5.94, 2.83]])
    """

    #: Each parameter's link (``"log"`` or ``"identity"``), in the
    #: distribution's parameter order.
    links: tuple[str, ...]
    #: Each parameter's columns of ``Z``, one integer array per
    #: parameter.
    covariate_positions: tuple
    #: The numbers of the coefficients the data could not determine.
    _aliased_coefficients: npt.NDArray
    #: The number of columns of ``Z``.
    _width: int

    def __repr__(self) -> str:
        out = super().__repr__()
        if not hasattr(self, "params"):
            return out
        lines = out.split("\n")
        title = "Distributional Regression SurPyval Model"
        lines[0], lines[1] = title, "=" * len(title)
        names = self.feature_names
        terms = []
        for name, link, cols in zip(
            self.distribution.parameter_names,
            self.links,
            self.covariate_positions,
        ):
            shown = [
                names[j] if names is not None and j < len(names) else f"Z{j}"
                for j in np.asarray(cols, dtype=int).tolist()
            ]
            g = name if link == "identity" else f"{link}({name})"
            terms.append("{} ~ {}".format(g, " + ".join(["1", *shown])))
        at = next(
            (i for i, line in enumerate(lines) if line.startswith("Fitted")),
            len(lines) - 1,
        )
        lines.insert(at, "Parameters          : " + "; ".join(terms))
        return "\n".join(
            line for line in lines if not line.startswith("Regression Model")
        )

    # -- the coefficients ------------------------------------------------

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` with a coefficient the data cannot
        determine, on some parameter (constant, or a linear combination of
        that parameter's other columns): the coefficient is ``nan`` in
        ``params``, and is taken as 0 in predictions."""
        coefs = self._aliased_coefficients
        if coefs.size == 0:
            return np.array([], dtype=int)
        columns = np.concatenate(
            [np.asarray(c, dtype=int) for c in self.covariate_positions]
        )
        return np.unique(columns[coefs])

    def _eval_params(self) -> npt.NDArray:
        if not self._aliased_coefficients.size:
            return self.params
        params = np.array(self.params, dtype=float)
        params[self.k_dist + self._aliased_coefficients] = 0.0
        return params

    def _held(self) -> set:
        names = self.parameter_names
        return set(self.fixed) | {
            names[self.k_dist + j] for j in self._aliased_coefficients.tolist()
        }

    def covariance(self) -> npt.NDArray:
        """
        The covariance of the fitted parameters, in the order of
        :attr:`parameter_names`: the inverse of the observed information
        (the exact Hessian the fit computed), with zero rows for fixed
        parameters and ``nan`` for aliased coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> model.covariance().shape
        (4, 4)
        """
        restored = self._restored_covariance
        if restored is not None:
            return restored
        self._check_inference()
        _, _, cov = self._inference_state()
        if self._fit_centring is not None:
            J = self._fit_centring[2]
            cov = J @ cov @ J.T
        if self._aliased_coefficients.size:
            cov = np.array(cov, dtype=float)
            at = self.k_dist + self._aliased_coefficients
            cov[at, :] = np.nan
            cov[:, at] = np.nan
        return cov

    def _n_covariates(self) -> int:
        return int(self._width)

    def _check_covariate_width(self, Z: Any) -> None:
        names = self.feature_names
        n = self._n_covariates()
        if names is None or len(names) != n:
            names = [f"Z[:, {j}]" for j in range(n)]
        check_covariate_width(Z, names, "covariate")

    def _is_linear_predictor(self) -> bool:
        return True

    def _coefficient_names(self) -> list[str]:
        return list(self.parameter_names[self.k_dist :])

    def _coefficient_columns(self) -> dict[str, int]:
        """``{coefficient name: its column of Z}``."""
        names = self.parameter_names[self.k_dist :]
        columns = np.concatenate(
            [np.asarray(c, dtype=int) for c in self.covariate_positions]
        )
        return {name: int(j) for name, j in zip(names, columns)}

    def _exp_meaning(self) -> str | None:
        return None

    def _wald_tested(self) -> list[int]:
        return list(range(self.k_dist, len(self.params)))

    def phi(self, Z: Any) -> npt.NDArray:
        """Not defined: a distributional regression has no single
        covariate multiplier. :meth:`parameters_at` gives each row's
        parameters."""
        raise NotImplementedError(
            "phi() is not defined for a distributional regression: the "
            "covariates move each parameter on its own, not through one "
            "multiplier. Use parameters_at(Z) for each row's parameters."
        )

    # -- each row's distribution -----------------------------------------

    def _rows(self, Z: Any) -> npt.NDArray:
        rows = covariate_rows(self._prepare_Z(Z), self._n_covariates())
        return np.asarray(self._centred(rows), dtype=float)

    def parameters_at(self, Z: npt.ArrayLike | pd.DataFrame) -> Any:
        """
        The distribution's parameters at each covariate row of ``Z``:
        an array with a row per row of ``Z`` and a column per parameter
        (a DataFrame, with the parameters' names, when ``Z`` is one).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> model.parameters_at([[0.5]]).round(2)
        array([[7.47, 1.98]])
        """
        rows = self._rows(Z)
        with np.errstate(all="ignore"):
            values = self.model._dist_params_at(rows, self._eval_params())
        out = np.column_stack(
            [
                np.asarray(v, dtype=float) * np.ones(rows.shape[0])
                for v in values
            ]
        )
        if type(Z).__name__ == "DataFrame":
            import pandas as pd

            return pd.DataFrame(
                out,
                columns=list(self.distribution.parameter_names),
                index=getattr(Z, "index", None),
            )
        return out

    def distribution_at(self, Z: Any) -> Any:
        """
        The distribution of a unit with the covariate row ``Z``, as a
        univariate model (``Weibull.from_params(...)``), with every
        univariate function: ``qf``, ``mean``, ``plot`` and the rest.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> model.distribution_at([1.0]).params.round(2)
        array([5.94, 2.83])
        """
        values = self.parameters_at(np.atleast_2d(np.asarray(Z, float)))
        if values.shape[0] != 1:
            raise ValueError(
                "distribution_at takes one covariate row; got {} rows. Use "
                "parameters_at for several.".format(values.shape[0])
            )
        return self.distribution.from_params(list(values[0]))

    def _link_value(self, name: str, rows: npt.NDArray, params: Any) -> Any:
        """Parameter ``name`` at each row, on its link's scale."""
        from . import _links

        k = list(self.distribution.parameter_names).index(name)
        value = self.model._dist_params_at(rows, params)[k]
        return _links.link(self.links[k], value)

    def parameter_cb(
        self,
        name: str,
        Z: npt.ArrayLike | pd.DataFrame,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Confidence bounds on the distribution's parameter ``name`` at each
        covariate row of ``Z``: a row per row, ``[lower, upper]`` for a
        two-sided bound. Computed on the parameter's link scale, so they
        stay inside its range (a shape's above 0).

        ``method="wald"`` (the default) is the delta method from the
        observed information; ``"lr"`` the extremes of the parameter over
        the likelihood-ratio region, ``2 [l_hat - l] <= chi2_1``, as
        :meth:`cb` with ``method="lr"`` takes them; ``"bootstrap"`` the
        BCa interval from ``n_boot`` parametric bootstrap refits
        (``random_state`` seeds them).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> model.parameter_cb("beta", [[0.0], [1.0]]).round(2)
        array([[1.15, 1.67],
               [2.36, 3.39]])
        """
        from surpyval.utils.validation import check_alpha_ci, check_option

        from . import _links

        check_alpha_ci(alpha_ci)
        check_option("bound", bound, ("two-sided", "lower", "upper"))
        check_option("method", method, ("wald", "lr", "bootstrap"))
        names = list(self.distribution.parameter_names)
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; the {} distribution's parameters "
                "are {}.".format(name, self.distribution.name, names)
            )
        link = self.links[names.index(name)]
        if method == "wald":
            eta = self._parameter_wald(name, Z, alpha_ci, bound)
        elif method == "lr":
            eta = self._parameter_lr(name, Z, alpha_ci, bound)
        else:
            eta = self._parameter_bootstrap(
                name, Z, alpha_ci, bound, n_boot, random_state
            )
        with np.errstate(over="ignore"):
            out = np.asarray(_links.inverse(link, eta), dtype=float)
        return out if bound == "two-sided" else out.reshape(-1)

    def _parameter_wald(
        self, name: str, Z: Any, alpha_ci: float, bound: str
    ) -> npt.NDArray:
        from autograd import jacobian
        from scipy.stats import norm

        self._check_inference()
        rows = self._rows(Z)
        p = np.asarray(self._eval_params(), dtype=float)
        eta = np.asarray(self._link_value(name, rows, p), dtype=float)
        eta = eta * np.ones(rows.shape[0])
        with np.errstate(all="ignore"):
            J = np.asarray(
                jacobian(lambda v: self._link_value(name, rows, v))(p),
                dtype=float,
            ).reshape(rows.shape[0], -1)
        cov = np.nan_to_num(np.asarray(self.covariance(), dtype=float))
        se = np.sqrt(np.einsum("ij,jk,ik->i", J, cov, J))
        if bound == "two-sided":
            z = norm.ppf(1 - alpha_ci / 2)
            return np.column_stack([eta - z * se, eta + z * se])
        z = norm.ppf(1 - alpha_ci)
        return eta - z * se if bound == "lower" else eta + z * se

    def _parameter_lr(
        self, name: str, Z: Any, alpha_ci: float, bound: str
    ) -> npt.NDArray:
        from .._likelihood_ratio import critical_value, lr_search

        search = lr_search(self, reported=False)
        rows = np.asarray(
            self._centred(
                covariate_rows(self._prepare_Z(Z), self._n_covariates()),
                search.center,
            ),
            dtype=float,
        )
        want = (
            bound in ("two-sided", "lower"),
            bound in ("two-sided", "upper"),
        )

        def psi_of(i: int, theta: npt.NDArray) -> float:
            value = self._link_value(name, rows[i : i + 1], theta)
            return float(np.ravel(np.asarray(value, dtype=float))[0])

        big = float(np.finfo(float).max)
        lower, upper, failed, _ = search._side_by_side(
            psi_of,
            np.arange(rows.shape[0], dtype=float),
            lambda i: (name, rows[i].tobytes()),
            "parameter " + name,
            critical_value(alpha_ci, bound),
            (-big, big),
            lambda v: v,
            want,
            falling=False,
        )
        if failed:
            warnings.warn(
                "The likelihood-ratio bound on {} could not be found at {} "
                "of the {} rows (nan there); method='wald' gives a bound in "
                "its place.".format(name, len(failed), rows.shape[0]),
                RuntimeWarning,
                stacklevel=3,
            )
        if bound == "two-sided":
            return np.column_stack([lower, upper])
        return np.asarray(lower if bound == "lower" else upper)

    def _parameter_bootstrap(
        self,
        name: str,
        Z: Any,
        alpha_ci: float,
        bound: str,
        n_boot: int,
        random_state: Any,
    ) -> npt.NDArray:
        from .._bootstrap import function_bounds, refits

        fits = refits(self, n_boot, random_state)
        fits.warn()
        given = covariate_rows(self._prepare_Z(Z), self._n_covariates())

        def value(params: Any, center: Any = None) -> Any:
            rows = self._centred(given, center)
            out = self._link_value(name, np.asarray(rows, float), params)
            return out * np.ones(given.shape[0])

        return function_bounds(self, fits, value, None, alpha_ci, bound)

    def plot_parameter(
        self,
        name: str,
        covariate: str | int,
        ax: Any = None,
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
        values: npt.ArrayLike | None = None,
    ) -> Any:
        """
        Plot the distribution's parameter ``name`` against one covariate
        (a column name, or its number), the other covariates at their
        means, with its Wald confidence band (:meth:`parameter_cb`).
        ``values`` are the covariate's values to draw at, by default its
        fitted range. It needs the fitted data unless ``values`` is given.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> ax = model.plot_parameter("beta", 0)
        """
        from surpyval import plotting
        from surpyval.utils.validation import check_alpha_ci

        check_alpha_ci(alpha_ci)
        names = self.feature_names
        if isinstance(covariate, str):
            if names is None or covariate not in names:
                raise ValueError(
                    "No covariate {!r}: the model's covariates are {}. Give "
                    "a column number for a model fitted to an "
                    "array.".format(
                        covariate,
                        names if names is not None else "unnamed",
                    )
                )
            j = names.index(covariate)
            label = covariate
        else:
            j = int(covariate)
            if not 0 <= j < self._n_covariates():
                raise ValueError(
                    "covariate {} is out of range: the model has {} "
                    "covariate(s).".format(j, self._n_covariates())
                )
            label = names[j] if names is not None else f"Z[:, {j}]"
        data = getattr(self, "data", None)
        if data is not None:
            Zd = np.asarray(data.Z, dtype=float)
            n = np.asarray(data.n, dtype=float)
            mean = n @ Zd / n.sum()
        elif values is None:
            raise ValueError(
                "plot_parameter needs the fitted data for the covariates' "
                "range and means, which a restored model does not keep: "
                "pass values=."
            )
        else:
            mean = np.zeros(self._n_covariates())
        if values is None:
            grid = np.linspace(Zd[:, j].min(), Zd[:, j].max(), 101)
        else:
            grid = np.asarray(values, dtype=float).reshape(-1)
        rows = np.tile(mean, (grid.size, 1))
        rows[:, j] = grid
        k = list(self.distribution.parameter_names).index(name)
        est = self.parameters_at(rows)[:, k]
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()
        if plot_bounds:
            band = self.parameter_cb(name, rows, alpha_ci=alpha_ci)
            ax.fill_between(
                grid,
                band[:, 0],
                band[:, 1],
                alpha=0.2,
                label=f"{(1 - alpha_ci) * 100:g}% confidence band",
            )
        ax.plot(grid, est, label=f"{name}, other covariates at their means")
        ax.set_xlabel(label)
        ax.set_ylabel(name)
        plotting.style_axes(ax)
        plotting.style_legend(ax)
        return ax

    @keeps_query_shape
    def qf(
        self,
        p: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The quantile at probability ``p`` for covariates ``Z``: the
        distribution's quantile function at each row's parameters, in
        closed form. ``grid=True`` gives every ``p`` for every row.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> b10 = model.qf(0.1, [[0.0], [1.0]])
        >>> model.ff(b10, [[0.0], [1.0]]).round(12)
        array([0.1, 0.1])
        """
        u = np.asarray(p, dtype=float)
        rows = covariate_rows(self._prepare_Z(Z), self._n_covariates())
        shape = None
        if grid:
            shape = (rows.shape[0], u.size)
            u = np.tile(u.reshape(-1), shape[0])
            rows = np.repeat(rows, shape[1], axis=0)
        else:
            check_paired_rows(u.size, rows.shape[0])
            size = max(u.size, rows.shape[0])
            u = np.broadcast_to(u.reshape(-1), (size,))
            rows = np.broadcast_to(rows, (size, rows.shape[1]))
        rows = np.asarray(self._centred(rows), dtype=float)
        outside = warn_outside_unit_interval(u)
        u = np.where(outside, np.nan, u)
        with np.errstate(all="ignore"):
            values = self.model._dist_params_at(rows, self._eval_params())
            out = np.asarray(self.distribution.qf(u, *values), dtype=float)
        out = out * np.ones(u.shape)
        if shape is not None:
            out = out.reshape(shape)
        return out

    def mean(self, Z: npt.ArrayLike | pd.DataFrame) -> Any:
        """
        The mean life of a unit with covariates ``Z``: the distribution's
        mean at each row's parameters, in closed form. An infinite mean
        (a log-logistic shape at or below 1) is returned as ``inf``, with
        a warning; a missing covariate gives ``nan``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> model.mean([[0.0], [1.0]]).round(2)
        array([8.57, 5.29])
        """
        rows = self._rows(Z)
        out = np.full(rows.shape[0], np.nan)
        known = np.all(np.isfinite(rows), axis=1)
        dist: Any = self.distribution
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = self.model._dist_params_at(rows, self._eval_params())
            values = [
                np.asarray(v, float) * np.ones(rows.shape[0]) for v in values
            ]
            for j in np.flatnonzero(known):
                out[j] = float(dist.mean(*(float(v[j]) for v in values)))
        out = np.where(np.isnan(out) & known, np.inf, out)
        infinite = int(np.sum(np.isposinf(out)))
        if infinite:
            warnings.warn(
                "The mean life is infinite at {} of the {} covariate rows: "
                "the distribution there has no finite mean (for example a "
                "log-logistic shape at or below 1). Use a quantile (qf) "
                "instead.".format(infinite, rows.shape[0]),
                UserWarning,
                stacklevel=2,
            )
        return out[0] if out.size == 1 else out

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-ready ``dict``: the
        distribution, each parameter's link and columns, the parameters
        and, where it can be computed, their covariance (so the restored
        model gives the same confidence bounds).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.sf(5, [0.5]), model.sf(5, [0.5])))
        True
        """
        out: dict[str, Any] = {
            "model": "DistributionalRegressionModel",
            "kind": self.kind,
            "distribution": self.distribution.name,
            "links": list(self.links),
            "covariate_positions": [
                np.asarray(c, dtype=int).tolist()
                for c in self.covariate_positions
            ],
            "width": int(self._width),
            "phi_param_map": {
                str(k): int(v) for k, v in self.phi_param_map.items()
            },
            "params": np.asarray(self.params, dtype=float).tolist(),
            "aliased_coefficients": np.asarray(
                self._aliased_coefficients, dtype=int
            ).tolist(),
            "k": int(self.k),
            "k_dist": int(self.k_dist),
            "fixed": {str(k): float(v) for k, v in self.fixed.items()},
        }
        if self._has_center():
            out["center"] = np.asarray(self.center, dtype=float).tolist()
        serialise_covariate_meta(self, out)
        if hasattr(self, "data") and getattr(self, "res", None) is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cov = self.covariance()
            except Exception:
                cov = None
        else:
            cov = self._restored_covariance
        if cov is not None:
            free = ~np.isnan(np.diag(cov))
            if np.all(np.isfinite(np.asarray(cov)[np.ix_(free, free)])):
                out["covariance"] = np.asarray(cov, dtype=float).tolist()
                state = self._serialised_inference()
                if state is not None:
                    out["inference_centring"] = state
        if hasattr(self, "_neg_ll"):
            out["_neg_ll"] = float(self._neg_ll)
        out.update(maximum_entry(self.maximum))
        ic_n = self._ic_sample_size_or_none()
        if ic_n is not None:
            out["ic_n"] = ic_n
        if self._data_repr():
            out["data_summary"] = self._data_repr()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> DistributionalRegressionModel:
        """
        Rebuild a model from its :meth:`to_dict` dictionary; it predicts
        as the original did, and gives the same bounds where the
        dictionary has a covariance.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import DistributionalRegressionModel
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> d = WeibullDistributional.fit(x, load).to_dict()
        >>> DistributionalRegressionModel.from_dict(d).links
        ('log', 'log')
        """
        import surpyval
        from surpyval.univariate.parametric.parametric_fitter import (
            ParametricFitter,
        )

        from .fitter import DistributionalFitter, _link_name

        if model_dict.get("model") != "DistributionalRegressionModel":
            raise ValueError(
                "Not a distributional regression model's dict (its 'model' "
                "is {!r}).".format(model_dict.get("model"))
            )
        dist = getattr(surpyval, str(model_dict["distribution"]), None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                "Unknown distribution {!r}".format(model_dict["distribution"])
            )
        links = tuple(str(v) for v in model_dict["links"])
        positions = tuple(
            np.asarray(c, dtype=int) for c in model_dict["covariate_positions"]
        )
        width = int(model_dict["width"])
        fitter = DistributionalFitter(
            dist, links=dict(zip(dist.parameter_names, links))
        )._bound(positions, width, links)
        params = np.array(model_dict["params"], dtype=float)
        k_dist = int(model_dict["k_dist"])
        pmap = {str(k): int(v) for k, v in model_dict["phi_param_map"].items()}
        if len(params) != k_dist + len(pmap) or len(positions) != k_dist:
            raise ValueError(
                "The model dict's parameters do not match its "
                "distribution's {} parameters and {} coefficient(s).".format(
                    k_dist, len(pmap)
                )
            )
        out = cls()
        out.model = fitter
        out.distribution = dist
        out.dist = dist
        out.distribution_param_map = dict(fitter.param_map)
        out.phi_param_map = pmap
        out.reg_model = CovariateLink(_link_name(fitter), pmap)
        out.kind = DISTRIBUTIONAL
        out.params = params
        out.dist_params = params[:k_dist]
        out.phi_params = params[k_dist:]
        out.k_dist = k_dist
        out.fixed = {
            str(k): float(v) for k, v in model_dict.get("fixed", {}).items()
        }
        out.links = links
        out.covariate_positions = fitter._require_positions()
        out._aliased_coefficients = np.array(
            model_dict.get("aliased_coefficients", []), dtype=int
        )
        out._width = width
        out.k = len(params) - len(out.fixed) - out._aliased_coefficients.size
        out._restored = True
        out._data_summary = model_dict.get("data_summary")
        out.center = np.array(
            model_dict.get("center", np.zeros(width)), dtype=float
        )
        if out.center.shape != (width,):
            raise ValueError(
                "The model dict's 'center' has {} value(s) for {} "
                "covariate column(s).".format(out.center.size, width)
            )
        restore_covariate_meta(out, model_dict)
        if "covariance" in model_dict:
            out._restored_covariance = np.array(
                model_dict["covariance"], dtype=float
            )
        state = model_dict.get("inference_centring")
        if isinstance(state, dict) and "covariance" in model_dict:
            restored = (
                np.array(state.get("params", ()), dtype=float),
                np.array(state.get("at", ()), dtype=float),
                np.array(state.get("covariance", ()), dtype=float),
            )
            n_par = len(params)
            if (
                restored[0].shape != (n_par,)
                or restored[1].shape != (width,)
                or restored[2].shape != (n_par, n_par)
            ):
                raise ValueError(
                    "The model dict's 'inference_centring' does not match "
                    "its {} parameter(s).".format(n_par)
                )
            out._restored_inference = restored
        if "_neg_ll" in model_dict:
            out._neg_ll = float(model_dict["_neg_ll"])
        out.maximum = restored_maximum(model_dict)
        out._ic_n = cls._restored_ic_n(model_dict)
        return out
