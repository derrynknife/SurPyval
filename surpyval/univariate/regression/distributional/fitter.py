"""Distributional regression: covariates on every parameter.

``Distributional(Weibull)`` gives each parameter of the distribution its
own linear predictor and link,

.. math::
    g_k\\bigl(\\theta_k(z)\\bigr) = \\eta_k(z) = b_{k0} + s_k\\, x_k(z)' b_k,

so the covariates can move the scale *and* the shape (or the location and
the spread) of the life distribution. The sign ``s_k`` is SurPyval's
convention, that of its AFT models: a positive coefficient makes lives
shorter. It is -1 for a parameter whose increase makes every quantile
later (a scale, a location; the AFT model is the case with covariates on
the scale only, coefficient for coefficient) and +1 for one whose
increase makes every quantile earlier (a rate); a shape or spread that
moves some quantiles each way (a Weibull's shape, a ``sigma``) has no
direction of its own and keeps the link's sign, +1, so a positive
coefficient raises it. The likelihood is the regressions'
own (``regression_neg_ll``): every censoring, truncation and count of
SurPyval's data model, each row with its own parameters.

The fit runs on its own pipeline (no edit to the shared one), built from
the shared helpers with inputs per parameter: each parameter's columns
are checked for aliasing on their own (every parameter has an
intercept), each coefficient is searched and judged in its own column's
units, and the no-finite-maximum check sees every coefficient and every
parameter's intercept.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, cast

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd import jacobian

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.covariates import coefficient_floor
from surpyval.utils.linalg import fixed_order_dot
from surpyval.utils.surpyval_data import SurpyvalData, accepts_surpyval_data
from surpyval.utils.units import takes_unit

from .._aliasing import (
    aliased_columns,
    constant_columns,
    dataframe_covariates,
    fit_columns,
    warn_aliased,
)
from .._covariate_link import CovariateLink
from .._fit_skeleton import (
    LOG_MAX,
    MirroredDistributionAttrs,
    canonical_order,
    centred_copy,
    check_baseline_support,
    check_fixed_and_init,
    covariate_center,
    drop_nonfinite_covariates,
    finite_start,
    judge_search,
    mirror_distribution,
    natural_information,
    optimise_ph,
    require_finite_fit,
    say_verdict,
    uniform_draws,
)
from .._kinds import DISTRIBUTIONAL
from .._likelihood import regression_neg_ll
from ..regression_data import (
    DataFrameRegressionMixin,
    design_matrix_from_df,
    truncation_window,
)
from . import _links

#: The distributions a distributional regression takes: continuous, with
#: a support that does not depend on the parameters, and every parameter
#: on the whole line or positive.
SUPPORTED = (
    "Exponential",
    "Rayleigh",
    "Weibull",
    "Gamma",
    "LogNormal",
    "LogLogistic",
    "Normal",
    "Gumbel",
    "GumbelLEV",
    "Logistic",
    "Beta",
    "ExpoWeibull",
)

#: The advice of a refusal of the parameters at Z = 0.
_ORIGIN_HINT = (
    "Fit with center=True to report the parameters at the covariate means "
    "(model.center) instead, or move the covariates nearer 0."
)

#: The value of each function at (and below) the start of the support.
_AT_SUPPORT_START = {
    "sf": 1.0,
    "ff": 0.0,
    "Hf": 0.0,
    "log_sf": 0.0,
    "log_ff": -np.inf,
}

#: The name of the covariate link a fitted model carries.
LINK_NAME = "Distributional"


def _check_distribution(distribution: Any) -> None:
    """Refuse a distribution the model is not defined for, saying why."""
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )

    name = getattr(distribution, "name", None)
    if not isinstance(distribution, ParametricFitter):
        raise ValueError(
            "Distributional takes a parametric distribution (Weibull, "
            "LogNormal, Gamma, ...); got {!r}. For a fitted model, pass its "
            "distribution (model.dist).".format(distribution)
        )
    if name not in SUPPORTED:
        raise ValueError(
            "Distributional regression is not available for the {} "
            "distribution: it takes a continuous distribution whose support "
            "does not depend on its parameters, one of {}. For covariates "
            "on a single effect, use an AFT or PH model (e.g. "
            "AFT({})).".format(name, ", ".join(SUPPORTED), name)
        )


def _as_list(value: Any) -> list:
    if isinstance(value, (str, int, np.integer)):
        return [value]
    return list(value)


class DistributionalFitter(
    MirroredDistributionAttrs, DataFrameRegressionMixin
):
    """A regression fitter whose covariates move every parameter of its
    distribution, each through its own link; build one with
    :func:`Distributional`.

    The parameters are the distribution's at the reference row (``Z =
    0``, or the covariate means with ``center=True``), on their natural
    scale, then each parameter's coefficients on its link scale, named
    ``<parameter>_<column>``, with SurPyval's sign (:attr:`signs`): a
    positive coefficient on a scale or location means a shorter life,
    as in the AFT models.
    """

    fitter_kind = "distributional regression fitter"
    name_suffix = "Distributional"

    #: Each parameter's columns of ``Z`` (one integer array per
    #: parameter), set on the fitter a fitted model carries; ``None`` on
    #: a fitter not yet fitted.
    _positions: tuple[npt.NDArray, ...] | None = None
    #: The width of ``Z`` the positions are for.
    _width: int | None = None
    #: Each parameter's link.
    links: tuple[str, ...]
    #: The covariates and links the fitter was built with, the defaults
    #: of its ``fit``.
    _covariates: Mapping[str, Any] | None = None
    _links_given: Mapping[str, str] | None = None

    def __init__(
        self,
        distribution: Any,
        covariates: Mapping[str, Any] | None = None,
        links: Mapping[str, str] | None = None,
    ) -> None:
        _check_distribution(distribution)
        mirror_distribution(self, distribution)
        self.name = distribution.name + self.name_suffix
        self._covariates = covariates
        self._links_given = links
        self.links = self._resolve_links(links)

    def __reduce__(self) -> Any:
        return (_rebuild_fitter, (self.dist, self.__getstate__()))

    def __getstate__(self) -> dict:
        return {
            "_positions": self._positions,
            "_width": self._width,
            "links": self.links,
            "_covariates": self._covariates,
            "_links_given": self._links_given,
        }

    # -- the specification ------------------------------------------------

    @property
    def signs(self) -> tuple[float, ...]:
        """The sign ``s_k`` of each parameter's coefficients,
        ``g_k(theta_k(z)) = g_k(theta_k) + s_k x_k' b_k``: -1 for a
        parameter whose increase makes lives longer (a scale, a location),
        so that a positive coefficient always means a shorter life, as in
        SurPyval's AFT models; +1 otherwise (a rate, or a shape or spread
        with no direction of its own). See :func:`._links.sign`."""
        return tuple(
            _links.sign(self.dist.name, name) for name in self.parameter_names
        )

    def _resolve_links(
        self, links: Mapping[str, str] | None
    ) -> tuple[str, ...]:
        links = dict(links or {})
        unknown = [k for k in links if k not in self.param_map]
        if unknown:
            raise ValueError(
                "Unknown parameter(s) {} in `links`; the {} distribution's "
                "parameters are {}.".format(
                    unknown, self.dist.name, list(self.parameter_names)
                )
            )
        out = []
        for name, bounds in zip(self.parameter_names, self.bounds):
            chosen = links.get(name, _links.default_link(bounds))
            allowed = _links.allowed_links(bounds)
            if chosen not in allowed:
                raise ValueError(
                    "links[{!r}] = {!r} is not available: {} has bounds {}, "
                    "and the link's inverse must stay inside them. Use one "
                    "of {}.".format(name, chosen, name, bounds, list(allowed))
                )
            out.append(chosen)
        return tuple(out)

    def _resolve_covariates(
        self,
        covariates: Mapping[str, Any] | None,
        p: int,
        names: list[str] | None,
    ) -> tuple[npt.NDArray, ...]:
        """Each parameter's columns of ``Z`` (``p`` of them, named
        ``names`` where the fit has names): every column on every
        parameter where ``covariates`` is ``None``; else the named
        parameters' columns, and none for the others."""
        if covariates is None:
            return tuple(np.arange(p) for _ in self.parameter_names)
        if not isinstance(covariates, Mapping):
            raise ValueError(
                "`covariates` is a dict from parameter name to its "
                "columns, e.g. {{'beta': [0]}}; got {!r}.".format(covariates)
            )
        unknown = [k for k in covariates if k not in self.param_map]
        if unknown:
            raise ValueError(
                "Unknown parameter(s) {} in `covariates`; the {} "
                "distribution's parameters are {}.".format(
                    unknown, self.dist.name, list(self.parameter_names)
                )
            )
        out = []
        for name in self.parameter_names:
            cols = []
            for col in _as_list(covariates.get(name, [])):
                if isinstance(col, str):
                    if names is None or col not in names:
                        raise ValueError(
                            "covariates[{!r}] names the column {!r}, but "
                            "the covariates have {}. Name columns with "
                            "fit_from_df (Z_cols) or a DataFrame Z; with an "
                            "array Z, give column numbers.".format(
                                name,
                                col,
                                (
                                    f"the columns {names}"
                                    if names is not None
                                    else "no column names"
                                ),
                            )
                        )
                    cols.append(names.index(col))
                elif isinstance(col, (int, np.integer)) and not isinstance(
                    col, bool
                ):
                    if not 0 <= int(col) < p:
                        raise ValueError(
                            "covariates[{!r}] has column {}, but Z has {} "
                            "column(s) (0 to {}).".format(name, col, p, p - 1)
                        )
                    cols.append(int(col))
                else:
                    raise ValueError(
                        "covariates[{!r}] holds {!r}: give column numbers "
                        "or column names.".format(name, col)
                    )
            if len(set(cols)) != len(cols):
                raise ValueError(
                    "covariates[{!r}] names a column more than once: "
                    "{}.".format(name, cols)
                )
            out.append(np.array(sorted(cols), dtype=int))
        return tuple(out)

    def _coefficient_names(self, names: list[str] | None) -> list[str]:
        """``<parameter>_<column>`` for every coefficient, parameter by
        parameter, the column by its name or ``coef_j``."""
        positions = self._require_positions()
        out = []
        for name, cols in zip(self.parameter_names, positions):
            for j in cols.tolist():
                col = (
                    names[j]
                    if names is not None and j < len(names)
                    else f"coef_{j}"
                )
                out.append(f"{name}_{col}")
        return out

    def _require_positions(self) -> tuple[npt.NDArray, ...]:
        if self._positions is None:
            raise ValueError(
                "This fitter has not been fitted: its functions take a "
                "fitted model's parameters. Call fit first and use the "
                "model's functions (model.sf(x, Z), ...)."
            )
        return self._positions

    def _bound(
        self,
        positions: tuple[npt.NDArray, ...],
        width: int,
        links: tuple[str, ...],
    ) -> DistributionalFitter:
        """A copy of this fitter for the model fitted with ``positions``
        (its ``fit`` refits with the same columns, as the bootstrap
        does)."""
        out = DistributionalFitter.__new__(DistributionalFitter)
        mirror_distribution(out, self.dist)
        out.name = self.name
        out.links = links
        out._links_given = dict(zip(self.parameter_names, links))
        out._positions = tuple(np.asarray(p, dtype=int) for p in positions)
        out._width = int(width)
        out._covariates = {
            name: cols.tolist()
            for name, cols in zip(self.parameter_names, out._positions)
        }
        return out

    # -- the model's functions ----------------------------------------------

    def _dist_params_at(self, Z: Numeric, params: Any) -> list:
        """The distribution's parameters at each row of ``Z`` (each an
        array with one value per row) for the model's ``params``."""
        positions = self._require_positions()
        Z_arr: Any = np.atleast_2d(np.asarray(Z, dtype=float))
        k = self.k_dist
        out = []
        start = k
        signs = self.signs
        for i, (cols, name) in enumerate(zip(positions, self.links)):
            base = params[i]
            m = cols.size
            if m == 0:
                out.append(base * np.ones(Z_arr.shape[0]))
                continue
            coefs = np.array(list(params[start : start + m]))
            start += m
            # Each row's sum in a fixed order (``fixed_order_dot``), so
            # that permuting the rows permutes the result exactly: a BLAS
            # product sums a row by where it sits (on Linux x86-64).
            shift = fixed_order_dot(Z_arr[:, cols], coefs)
            out.append(_links.move(name, base, signs[i] * shift))
        return out

    def _call(self, fn: str, x: Numeric, Z: Numeric, params: tuple) -> Any:
        dist_params = self._dist_params_at(Z, params)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        start = _AT_SUPPORT_START.get(fn)
        lower = float(self.support[0])
        if start is not None and np.isfinite(lower):
            at = x <= lower
            if np.any(at):
                # At or below the start of the support (a left truncation
                # at 0, the usual "no truncation"), the value is known
                # exactly; the distribution's formula there (0 ** beta)
                # has nan second derivatives, which took the exact
                # information and the verdict with them.
                inside = np.where(at, lower + 1.0, x)
                value = getattr(self.dist, fn)(inside, *dist_params)
                return np.where(at, start, value)
        return getattr(self.dist, fn)(x, *dist_params)

    def sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Survival function at ``x`` for covariates ``Z``: the
        distribution's at each row's parameters. ``params`` are a fitted
        model's (``model.params``)."""
        return self._call("sf", x, Z, params)

    def ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Failure function; as :meth:`sf`."""
        return self._call("ff", x, Z, params)

    def df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Density; as :meth:`sf`."""
        return self._call("df", x, Z, params)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Hazard rate; as :meth:`sf`."""
        return self._call("hf", x, Z, params)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Cumulative hazard; as :meth:`sf`."""
        return self._call("Hf", x, Z, params)

    def log_sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._call("log_sf", x, Z, params)

    def log_ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._call("log_ff", x, Z, params)

    def log_df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._call("log_df", x, Z, params)

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        """The negative log-likelihood of ``data`` (with covariates) at
        ``params``."""
        return regression_neg_ll(self, data, *params)

    def random(
        self,
        size: int,
        Z: npt.ArrayLike,
        *params: float,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``size`` draws at each covariate row of ``Z``, row by row, and
        the row each was drawn at: the distribution's quantile function at
        uniform draws, with each row's parameters."""
        from surpyval.utils.rng import as_generator

        Z_arr = np.atleast_2d(np.asarray(Z, dtype=float))
        rng = None if random_state is None else as_generator(random_state)
        x, Z_out = [], []
        for row in Z_arr:
            dist_params = [
                float(np.ravel(v)[0])
                for v in self._dist_params_at(row, params)
            ]
            U = uniform_draws(size, rng)
            x.append(np.asarray(self.dist.qf(U, *dist_params), dtype=float))
            Z_out.append(np.tile(row, (size, 1)))
        return np.concatenate(x), np.vstack(Z_out)

    # -- fitting -----------------------------------------------------------

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        covariates: Mapping[str, Any] | None = None,
        links: Mapping[str, str] | None = None,
    ) -> Any:
        """
        Fit by maximum likelihood, with every censoring, truncation and
        count of SurPyval's data model.

        Parameters
        ----------
        x : array_like
            The times (an ``(N, 2)`` array of ``[lower, upper]`` for
            interval censored rows).
        Z : array_like or DataFrame
            The covariates, one row per observation (a 1-D array is one
            covariate). Rows with a missing or infinite covariate are
            dropped, with a warning.
        c, n, t, tl, tr : array_like, optional
            Censoring (0 observed, 1 right, -1 left, 2 interval), counts
            and truncation, as for every fit.
        init : array_like, optional
            Initial values, in the order of the model's ``params``.
        fixed : dict, optional
            Parameters to hold, by name: a coefficient held at 0 leaves
            its column out of that parameter.
        center : bool, optional
            ``False`` (the default) reports the distribution's parameters
            at ``Z = 0``; ``True`` at the covariate means
            (``model.center``).
        covariates : dict, optional
            Each parameter's columns of ``Z``, by number (or by name, for
            a DataFrame ``Z``): ``{"alpha": [0, 1], "beta": [0]}``. A
            parameter not named has none (its intercept only). Default:
            every column on every parameter.
        links : dict, optional
            A link per parameter, by name (``{"mu": "log"}``). The
            defaults are identity for a parameter on the whole line and
            log for a positive one; an identity link on a positive
            parameter is refused.

        Returns
        -------
        DistributionalRegressionModel

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> model = WeibullDistributional.fit(x, load)
        >>> model.parameter_names
        ['alpha', 'beta', 'alpha_coef_0', 'beta_coef_0']
        """
        if covariates is None:
            covariates = self._covariates
        fitter_links = self._links_given if links is None else links
        link_names = self._resolve_links(fitter_links)
        t = truncation_window(x, t, tl, tr)
        data = SurpyvalData(x, c, n, t, group_and_sort=False)
        Z_in = Z if hasattr(Z, "ndim") else np.asarray(Z)
        if getattr(Z_in, "ndim", 2) == 1:
            Z = np.asarray(Z_in).reshape(-1, 1)
        data, Z = drop_nonfinite_covariates(data, Z)
        order = canonical_order(data, Z)
        if np.any(order != np.arange(order.size)):
            data, Z = data[order], Z[order]
        data.add_covariates(Z)
        check_baseline_support(self, data)

        Z_data = np.asarray(data.Z, dtype=float)
        p = Z_data.shape[1]
        names = fit_columns()
        positions = self._resolve_covariates(covariates, p, names)
        bound = self._bound(positions, p, link_names)
        coef_names = bound._coefficient_names(names)
        K = self.k_dist
        all_names = [*self.parameter_names, *coef_names]
        param_map = {name: i for i, name in enumerate(all_names)}
        fixed = {} if fixed is None else dict(fixed)
        check_fixed_and_init(fixed, init, param_map)

        # Each coefficient's column, and its parameter
        column_of = np.concatenate([np.asarray(c, int) for c in positions])
        aliased = _alias(
            Z_data, data.n, positions, coef_names, fixed, column_of
        )
        held = {**fixed, **{coef_names[j]: 0.0 for j in aliased}}

        center_vec = covariate_center(Z_data, data.n) if p else np.zeros(0)
        fixed_intercept = any(
            name in fixed and positions[k].size
            for k, name in enumerate(self.parameter_names)
        )
        if not center and fixed_intercept:
            # A parameter held at Z = 0 is not at the means: fit there.
            center_vec = np.zeros(p)
        if p and not np.all(np.isfinite(center_vec)):
            center_vec = np.zeros(p)
        moves = bool(p) and bool(np.any(center_vec))
        data_c = centred_copy(data, center_vec) if moves else data

        space = _SearchSpace(
            all_names, held, K, link_names, positions, center_vec, bound.signs
        )
        free, positions_free = space.free, space.positions_free

        def fun(tv: Any) -> Any:
            return bound.neg_ll(data_c, *space.to_natural(tv))

        def default_init() -> npt.NDArray:
            base = self.dist.fit_from_surpyval_data(data).params
            return space.start(base)

        user_init = init is not None and len(np.atleast_1d(init)) > 0
        with np.errstate(all="ignore"):
            if user_init:
                nat = space.with_held(np.atleast_1d(init))
                if moves and not center:
                    nat = np.asarray(space.shift(nat, 1.0), dtype=float)
                init_t = space.to_search(nat)
            else:
                init_t = default_init()
        init_t = finite_start(fun, init_t, default_init if user_init else None)

        coef_free = [
            (positions_free[K + j], j)
            for j in range(len(coef_names))
            if K + j in positions_free
        ]
        floor = coefficient_floor(
            len(init_t),
            [(pos, int(column_of[j])) for pos, j in coef_free],
            Z_data,
        )
        baseline = [
            (positions_free[i], all_names[i])
            for i in range(K)
            if i in positions_free
        ]
        n_obs = float(np.sum(data.n))
        with np.errstate(all="ignore"):
            res = optimise_ph(fun, init_t, quiet=True, floor=floor)
            verdict = judge_search(
                fun,
                res,
                coef_free,
                init_t,
                n_obs,
                floor=floor,
                one_sided=(),
                baseline=baseline,
            )
            if verdict.maximum == "verified":
                verdict = _joint_run_off(
                    fun, verdict, floor, coef_free, baseline
                )
        if verdict.runaway:
            # Named, not numbered: a coefficient per column per parameter
            verdict = verdict._replace(
                runaway=cast(
                    "list[int]", [coef_names[j] for j in verdict.runaway]
                )
            )
        res = verdict.res
        require_finite_fit(float(res.fun))
        with np.errstate(all="ignore"):
            params_c = np.asarray(space.to_natural(res.x), dtype=float)

        fit_centring = None
        model_center = np.zeros(p)
        if moves and center:
            params = params_c
            model_center = np.array(center_vec, dtype=float)
        elif moves:
            with np.errstate(all="ignore"):
                params = np.asarray(space.shift(params_c, -1.0), dtype=float)
            _check_origin(params, params_c, self, link_names, center_vec)
            J = np.asarray(
                jacobian(lambda v: space.shift(v, -1.0))(params_c),
                dtype=float,
            )
            fit_centring = (params_c, center_vec, J)
        else:
            params = params_c

        model = _assemble(
            bound,
            params,
            res,
            fixed,
            held,
            coef_names,
            aliased,
            data,
            model_center,
            fit_centring,
        )
        say_verdict(
            verdict,
            dist=self.dist.name,
            values=dict(zip(self.parameter_names, model.params)),
        )
        model.maximum = verdict.maximum

        if not verdict.no_maximum and verdict.derivatives is not None:
            H_p = natural_information(
                verdict.derivatives,
                lambda u: space.to_natural(u)[np.array(free, dtype=int)],
                res.x,
            )
            if H_p is not None:
                at = params_c if (moves and not center) else params
                point_center = center_vec if moves else None
                model._information = (
                    model._covariance_point(at, point_center),
                    H_p,
                )
        return model

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit_penalised(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        penalizer: Any = 0.1,
        l1_ratio: float = 0.0,
        standardise: bool = True,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        tol: float = 1e-10,
        w: npt.ArrayLike | float | None = None,
        covariates: Mapping[str, Any] | None = None,
        links: Mapping[str, str] | None = None,
    ) -> Any:
        """
        Fit with a ridge, lasso or elastic-net penalty on the covariate
        coefficients of every parameter (the intercepts, the
        distribution's parameters at ``Z = 0``, are not penalised).

        The objective is ``WeibullPH.fit_penalised``'s: the negative
        log-likelihood over ``N`` (the sum of ``n``) plus
        :math:`\\sum_j \\lambda_j ((1 - a) b_j^2 / 2 + a |b_j|)` over the
        coefficients of the covariates divided by their standard
        deviations (``standardise=True``). With covariates on the scale
        alone and its log link, a Weibull's fit is ``WeibullAFT``'s
        penalised fit, coefficient for coefficient (the scale's
        coefficients take the AFT sign, ``model.coefficient_signs``). The
        model has no covariance, standard errors, bounds or tests (they
        raise an error that says why).

        Parameters
        ----------
        x, Z, c, n, t, tl, tr, covariates, links
            As for :meth:`fit`.
        penalizer : float, array_like or dict, optional
            One value for every coefficient; one per column of ``Z`` (that
            column's coefficient on every parameter); or one per
            parameter, a dict ``{"beta": 0.5}`` whose values are one value
            or one per column that parameter takes. A parameter the dict
            does not name is not penalised. Default 0.1.
        l1_ratio : float, optional
            From 0 (ridge, the default) to 1 (lasso).
        standardise : bool, optional
            Penalise the coefficients of the standardised covariates
            (default ``True``).
        tol : float, optional
            The optimality tolerance on the gradient of the objective.
        w : array_like, optional
            Not taken yet: weights are refused.

        Returns
        -------
        DistributionalRegressionModel
            The fitted model, with ``penalty`` holding one entry per
            coefficient, in the order of the coefficients in ``params``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> Z = rng.normal(size=(300, 3))
        >>> x = 10 * np.exp(-0.5 * Z[:, 0]) * rng.weibull(1.5, 300)
        >>> model = WeibullDistributional.fit_penalised(
        ...     x, Z, penalizer={"alpha": 0.02, "beta": 0.2}, l1_ratio=1
        ... )
        >>> model.params[2:].round(3)
        array([ 0.496, -0.014,  0.009,  0.   ,  0.   ,  0.   ])
        """
        from .._penalised import (
            PenalisedObjective,
            penalised_fit,
            refuse_weights,
        )

        refuse_weights(w)
        if covariates is None:
            covariates = self._covariates
        fitter_links = self._links_given if links is None else links
        link_names = self._resolve_links(fitter_links)
        t = truncation_window(x, t, tl, tr)
        data = SurpyvalData(x, c, n, t, group_and_sort=False)
        Z_in = Z if hasattr(Z, "ndim") else np.asarray(Z)
        if getattr(Z_in, "ndim", 2) == 1:
            Z = np.asarray(Z_in).reshape(-1, 1)
        data, Z = drop_nonfinite_covariates(data, Z)
        order = canonical_order(data, Z)
        if np.any(order != np.arange(order.size)):
            data, Z = data[order], Z[order]
        data.add_covariates(Z)
        check_baseline_support(self, data)

        Z_data = np.asarray(data.Z, dtype=float)
        p = Z_data.shape[1]
        names = fit_columns()
        positions = self._resolve_covariates(covariates, p, names)
        bound = self._bound(positions, p, link_names)
        coef_names = bound._coefficient_names(names)
        K = self.k_dist
        all_names = [*self.parameter_names, *coef_names]
        column_of = np.concatenate(
            [np.zeros(0, int), *[np.asarray(c, int) for c in positions]]
        )
        if column_of.size == 0:
            raise ValueError(
                "fit_penalised needs at least one coefficient: no "
                "parameter takes a covariate (`covariates`). Fit the "
                "distribution alone, or give a parameter its columns."
            )
        pen = _coefficient_penalizer(
            penalizer, positions, self.parameter_names, p
        )
        # One column per coefficient (its covariate), so each coefficient
        # is standardised by its own column's spread.
        Z_coef = Z_data[:, column_of]
        space = _SearchSpace(
            all_names, {}, K, link_names, positions, np.zeros(p), bound.signs
        )
        w_rows = np.asarray(data.n, dtype=float)

        def build(scale: npt.NDArray, N: float) -> PenalisedObjective:
            col_scale = np.ones(p)
            col_scale[column_of] = scale
            from .._penalised_fit import _intervals_from_the_support

            scaled = _intervals_from_the_support(self, data)
            scaled.add_covariates(Z_data / col_scale)

            def objective(tv: Any) -> Any:
                return bound.neg_ll(scaled, *space.to_natural(tv)) / N

            from autograd import grad as ag_grad
            from autograd import hessian as ag_hessian

            gradient = ag_grad(objective)
            curvature = ag_hessian(objective)
            with np.errstate(all="ignore"):
                base = self.dist.fit_from_surpyval_data(data).params
            return PenalisedObjective(
                lambda tv: float(objective(np.asarray(tv))),
                lambda tv: onp.asarray(gradient(np.asarray(tv)), dtype=float),
                lambda tv: onp.asarray(curvature(np.asarray(tv)), dtype=float),
                onp.asarray(space.start(base), dtype=float),
            )

        fit = penalised_fit(
            build,
            Z_coef,
            w_rows,
            onp.arange(K, K + column_of.size),
            pen,
            l1_ratio,
            standardise,
            tol,
        )
        res = fit.result
        with np.errstate(all="ignore"):
            params = onp.asarray(space.to_natural(res.x), dtype=float)
        params[K:] = fit.coefficients
        with np.errstate(all="ignore"):
            neg_ll = float(bound.neg_ll(data, *params))
        if not onp.isfinite(neg_ll):
            raise ValueError(
                "The penalised fit did not reach a finite likelihood; check "
                "the data, or use a larger `penalizer`."
            )
        from scipy.optimize import OptimizeResult

        model = _assemble(
            bound,
            params,
            OptimizeResult(x=res.x, fun=neg_ll, success=res.verified),
            {},
            {},
            coef_names,
            [],
            data,
            np.zeros(p),
            None,
        )
        model.penalty = fit.penalty
        model.maximum = "verified" if res.verified else "unverified"
        if not res.verified:
            from surpyval.utils.no_maximum import warn_unverified

            warn_unverified("The penalised fit")
        return model

    def fit_from_df(
        self,
        df: Any,
        x_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
        formula: str | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        covariates: Mapping[str, Any] | None = None,
        formulas: Mapping[str, str] | None = None,
        links: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Fit from a pandas DataFrame, with the columns named.

        The covariates are given one of three ways: ``Z_cols`` (with
        ``covariates`` choosing each parameter's columns by name, default
        every column on every parameter); ``formula``, one formula on
        every parameter; or ``formulas``, a formula per parameter
        (``{"alpha": "load + temp", "beta": "load"}``; a parameter not
        named has its intercept only). The other arguments are those of
        :meth:`fit` and of every regression's ``fit_from_df``.

        Returns
        -------
        DistributionalRegressionModel
            The fitted model, which predicts from a DataFrame with the
            same columns.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import WeibullDistributional
        >>> rng = np.random.default_rng(3)
        >>> load = rng.uniform(0, 1, 300)
        >>> x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
        >>> df = pd.DataFrame({"hours": x, "load": load})
        >>> model = WeibullDistributional.fit_from_df(
        ...     df, x_col="hours", formulas={"beta": "load"}
        ... )
        >>> model.parameter_names
        ['alpha', 'beta', 'beta_load']
        """
        if formulas is not None:
            if formula is not None or Z_cols is not None:
                raise ValueError(
                    "Give `formulas` (a formula per parameter) alone, not "
                    "with `formula` or `Z_cols`."
                )
            if covariates is not None:
                raise ValueError(
                    "`formulas` chooses each parameter's covariates; "
                    "`covariates` is for `Z_cols`. Give one of them."
                )
            formula, covariates = self._union_formula(df, formulas)
        elif covariates is not None and Z_cols is None:
            raise ValueError(
                "`covariates` chooses among the `Z_cols`; with a formula, "
                "give a formula per parameter with `formulas`."
            )
        target = self
        if covariates is not None or links is not None:
            target = DistributionalFitter(
                self.dist,
                covariates if covariates is not None else self._covariates,
                links if links is not None else self._links_given,
            )
        return DataFrameRegressionMixin.fit_from_df(
            target,
            df,
            x_col=x_col,
            Z_cols=Z_cols,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            tr_col=tr_col,
            xl_col=xl_col,
            xr_col=xr_col,
            formula=formula,
            init=init,
            fixed=fixed,
            center=center,
            **kwargs,
        )

    def _union_formula(
        self, df: Any, formulas: Mapping[str, str]
    ) -> tuple[str, dict[str, list[str]]]:
        """One formula with every parameter's terms, and each
        parameter's columns of its expansion, by name."""
        if not isinstance(formulas, Mapping):
            raise ValueError(
                "`formulas` is a dict from parameter name to its formula, "
                "e.g. {'beta': 'load'}."
            )
        unknown = [k for k in formulas if k not in self.param_map]
        if unknown:
            raise ValueError(
                "Unknown parameter(s) {} in `formulas`; the {} "
                "distribution's parameters are {}.".format(
                    unknown, self.dist.name, list(self.parameter_names)
                )
            )
        sides: list[str] = []
        covariates: dict[str, list[str]] = {}
        for name, text in formulas.items():
            rhs = str(text).split("~")[-1].strip()
            if rhs in ("", "1"):
                covariates[name] = []
                continue
            sides.append(rhs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, cols, _ = design_matrix_from_df(df, None, rhs)
            covariates[name] = list(cols)
        if not sides:
            raise ValueError(
                "Every formula in `formulas` is intercept-only; fit the "
                "distribution without covariates instead "
                "({}.fit).".format(self.dist.name)
            )
        union = " + ".join(f"({s})" for s in sides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, union_cols, _ = design_matrix_from_df(df, None, union)
        union_cols = list(union_cols)
        for name, cols in covariates.items():
            missing = [col for col in cols if col not in union_cols]
            if missing:
                raise ValueError(
                    "The formula for {!r} expands to the column(s) {}, "
                    "which the formulas together code differently (an "
                    "interaction whose main effect another formula has). "
                    "Write the main effects into {!r}'s formula too.".format(
                        name, missing, name
                    )
                )
        return union, covariates


class _SearchSpace:
    """The search of a fit: the link-scale intercepts at the covariate
    centre and the coefficients that are not held (fixed or aliased),
    and the maps between it and the model's parameters."""

    def __init__(
        self,
        names: list[str],
        held: dict,
        k_dist: int,
        links: tuple[str, ...],
        positions: tuple[npt.NDArray, ...],
        center: npt.NDArray,
        signs: tuple[float, ...],
    ) -> None:
        self.n = len(names)
        self.signs = signs
        self.k_dist = k_dist
        self.links = links
        self.positions = positions
        self.center = center
        self.free = [i for i, nm in enumerate(names) if nm not in held]
        self.positions_free = {i: pos for pos, i in enumerate(self.free)}
        self.values = np.zeros(self.n)
        self.held_index = [names.index(nm) for nm in held]
        for nm, v in held.items():
            self.values[names.index(nm)] = float(v)

    def to_natural(self, tv: Any) -> Any:
        """The model's parameters (at the centre) for the search vector."""
        entries = []
        for i in range(self.n):
            if i in self.positions_free:
                v = tv[self.positions_free[i]]
                if i < self.k_dist:
                    v = _links.inverse(self.links[i], v)
                entries.append(v)
            else:
                entries.append(self.values[i])
        return np.array(entries)

    def to_search(self, nat: npt.NDArray) -> npt.NDArray:
        out = np.array(nat, dtype=float)
        for i in range(self.k_dist):
            out[i] = _links.link(self.links[i], out[i])
        return out[self.free]

    def with_held(self, nat: npt.ArrayLike) -> npt.NDArray:
        out = np.array(nat, dtype=float)
        out[self.held_index] = self.values[self.held_index]
        return out

    def start(self, base: npt.ArrayLike) -> npt.NDArray:
        """The search vector at the distribution's parameters ``base``
        (those not held) and every free coefficient 0."""
        nat = np.array(self.values, dtype=float)
        for i in range(self.k_dist):
            if i in self.positions_free:
                nat[i] = float(np.asarray(base)[i])
        return self.to_search(nat)

    def shift(self, nat: Any, sign: float) -> Any:
        """``nat`` moved from the reference row to the centre
        (``sign=+1``) or back (``-1``)."""
        entries = [nat[i] for i in range(self.n)]
        start = self.k_dist
        for k in range(self.k_dist):
            cols = self.positions[k]
            if cols.size:
                coefs = nat[start : start + cols.size]
                s = self.signs[k] * np.dot(coefs, self.center[cols])
                entries[k] = _links.move(self.links[k], nat[k], sign * s)
                start += cols.size
        return np.array(entries)


def _assemble(
    bound: DistributionalFitter,
    params: npt.NDArray,
    res: Any,
    fixed: dict,
    held: dict,
    coef_names: list[str],
    aliased: list[int],
    data: SurpyvalData,
    center: npt.NDArray,
    fit_centring: tuple | None,
) -> Any:
    """The fitted model."""
    from .model import DistributionalRegressionModel

    K = bound.k_dist
    model = DistributionalRegressionModel()
    model.center = center
    model._fit_centring = fit_centring
    model.distribution_param_map = dict(bound.param_map)
    pmap = {name: j for j, name in enumerate(coef_names)}
    model.phi_param_map = pmap
    model.model = bound
    model.reg_model = CovariateLink(_link_name(bound), pmap)
    model.kind = DISTRIBUTIONAL
    model.distribution = bound.dist
    model.dist = bound.dist
    reported = np.array(params, dtype=float)
    reported[[K + j for j in aliased]] = np.nan
    model.params = reported
    model.dist_params = np.array(reported[:K])
    model.phi_params = np.array(reported[K:])
    model.res = res
    model._neg_ll = float(res.fun)
    model.fixed = dict(fixed)
    model.k_dist = K
    model.k = K + len(coef_names) - len(held)
    model.data = data
    model.links = bound.links
    model._coefficient_signs = bound.signs
    model.covariate_positions = bound._require_positions()
    model._aliased_coefficients = np.array(sorted(aliased), dtype=int)
    model._width = int(np.shape(data.Z)[1])
    return model


def _coefficient_penalizer(
    penalizer: Any,
    positions: tuple[npt.NDArray, ...],
    parameter_names: list[str],
    p: int,
) -> npt.NDArray:
    """The ``penalizer`` of a distributional ``fit_penalised`` as one
    value per coefficient, in the order of the coefficients: a single
    value, one per column of ``Z`` (that column on every parameter), or
    a dict of one value (or one per its columns) per parameter."""
    if isinstance(penalizer, Mapping):
        unknown = sorted(set(penalizer) - set(parameter_names))
        if unknown:
            raise ValueError(
                "`penalizer` names {}, which {} not among the "
                "distribution's parameters {}.".format(
                    unknown,
                    "is" if len(unknown) == 1 else "are",
                    list(parameter_names),
                )
            )
        parts = []
        for name, cols in zip(parameter_names, positions):
            value = onp.asarray(penalizer.get(name, 0.0), dtype=float)
            if value.ndim == 0:
                value = onp.full(cols.size, float(value))
            if value.shape != (cols.size,):
                raise ValueError(
                    "`penalizer[{!r}]` must be one value or one per column "
                    "{} takes ({}); got shape {}.".format(
                        name, name, cols.size, value.shape
                    )
                )
            parts.append(value)
        return onp.concatenate(parts)
    value = onp.asarray(penalizer, dtype=float)
    if value.ndim == 0:
        return onp.full(sum(c.size for c in positions), float(value))
    if value.shape != (p,):
        raise ValueError(
            "`penalizer` must be one value, one per covariate column "
            "({}), or a dict of one per parameter; got shape {}.".format(
                p, value.shape
            )
        )
    return onp.concatenate([value[onp.asarray(c, int)] for c in positions])


def _rebuild_fitter(dist: Any, state: dict) -> DistributionalFitter:
    out = DistributionalFitter.__new__(DistributionalFitter)
    mirror_distribution(out, dist)
    out.name = dist.name + DistributionalFitter.name_suffix
    for key, value in state.items():
        setattr(out, key, value)
    return out


def _link_name(fitter: DistributionalFitter) -> str:
    """``Distributional [log(alpha), log(beta)]``: the links."""
    parts = []
    for name, link in zip(fitter.parameter_names, fitter.links):
        parts.append(name if link == "identity" else f"{link}({name})")
    return "{} [{}]".format(LINK_NAME, ", ".join(parts))


def _alias(
    Z: npt.NDArray,
    n: npt.NDArray,
    positions: tuple[npt.NDArray, ...],
    coef_names: list[str],
    fixed: dict,
    column_of: npt.NDArray,
) -> list[int]:
    """The coefficients (their numbers) the data cannot determine, each
    parameter's columns checked on their own (every parameter has an
    intercept), with one warning naming their columns of ``Z``."""
    if Z.shape[0] == 0:
        return []
    n = np.asarray(n, dtype=float).reshape(-1)
    out: list[int] = []
    start = 0
    for cols in positions:
        numbers = np.arange(start, start + cols.size)
        start += cols.size
        free = np.array(
            [j for j in numbers if coef_names[j] not in fixed], dtype=int
        )
        if free.size == 0:
            continue
        Zf = Z[:, column_of[free]]
        constant = constant_columns(Zf)
        Zc = Zf - covariate_center(Zf, n)
        gram = Zc.T @ (n[:, None] * Zc)
        out += free[aliased_columns(gram, Z.shape[0], constant)].tolist()
    if out:
        warn_aliased(
            sorted({int(column_of[j]) for j in out}),
            "they are constant (each parameter has its own intercept) or a "
            "linear combination of the other columns on the same parameter",
        )
    return sorted(out)


#: A direction of the search is walked when its curvature is below this
#: fraction of the largest (in units of each parameter's size).
_FLAT = 1e-6
#: The steps of the walk along it, in those units.
_STEPS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
#: The steps a walk must keep rising for (to 8 units out).
_MIN_STEPS = 4


def _profile_walk(
    fun: Any, x: npt.NDArray, e: npt.NDArray, size: npt.NDArray
) -> list[float]:
    """The profile of ``fun`` along the direction ``e`` from ``x``: at each
    step, the minimum over the other directions (from the last step's
    answer), so the walk follows a curved valley rather than leaving it."""
    from autograd import grad as _grad
    from scipy.optimize import minimize

    unit = e / np.linalg.norm(e / size)
    # The other directions, in units of each parameter's size
    basis = np.linalg.svd((unit / size).reshape(1, -1))[2][1:].T
    other = basis * size[:, None]
    y = np.zeros(other.shape[1])
    out: list[float] = []
    for s in _STEPS:
        base = x + s * unit

        def g(v: Any, base: Any = base) -> Any:
            return fun(base + np.dot(other, v))

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                g, y, jac=_grad(g), method="BFGS", options={"maxiter": 200}
            )
        if not np.isfinite(res.fun):
            # Off the end of what can be evaluated: the walk so far
            return out
        y = res.x
        out.append(float(res.fun))
    return out


def _joint_run_off(
    fun: Any,
    verdict: Any,
    floor: Any,
    coefs: list[tuple[int, int]],
    baseline: list[tuple[int, str]],
) -> Any:
    """A verified answer that is not a maximum: the likelihood still rises,
    towards a limit, along a direction of several parameters at once.

    A group with no failures can have its scale and its shape (or its
    location and spread) run off together: the group's survival then
    reaches 1 so fast that the search stops on a plateau within 1e-9 nats
    of the supremum, its gradient zero to rounding, and the shared checks,
    which follow one parameter's profile at a time, call it a maximum.
    Each direction of near-zero curvature has its profile walked out both
    ways; where the likelihood along one never falls below the answer's
    and is higher at the end, every parameter with a share of the
    direction is named as running off. (At a maximum the profile falls
    away on both sides, however flat.)"""
    from .._fit_skeleton import SearchVerdict

    if verdict.derivatives is None:
        return verdict
    H, _ = verdict.derivatives
    x = np.asarray(verdict.res.x, dtype=float)
    size = np.maximum(np.abs(x), np.broadcast_to(floor, x.shape))
    with np.errstate(all="ignore"):
        Hu = H * np.outer(size, size)
    if not np.all(np.isfinite(Hu)) or x.size < 2:
        return verdict
    w, V = np.linalg.eigh(0.5 * (Hu + Hu.T))
    top = float(np.max(np.abs(w)))
    f0 = float(verdict.res.fun)
    tol = 64 * np.finfo(float).eps * max(1.0, abs(f0))
    running: set = set()
    flat = [V[:, k] for k in np.flatnonzero(w <= _FLAT * top)]
    # The flat directions, and each parameter with a share of one alone
    # (the valley a run-off follows can curve away from both)
    shares = [np.abs(d) >= 0.1 * np.max(np.abs(d)) for d in flat]
    alone = sorted(set().union(*(np.flatnonzero(m).tolist() for m in shares)))
    candidates = list(zip(flat, shares)) + [
        (np.eye(x.size)[i], np.eye(x.size, dtype=bool)[i]) for i in alone
    ]
    for direction, share in candidates:
        for sign in (1.0, -1.0):
            values = _profile_walk(fun, x, sign * direction * size, size)
            # The walk until it first rises (far out, the inner search can
            # lose the valley): eight units out at least, never above the
            # answer, and higher in likelihood at its end
            kept: list = []
            for v in values:
                if v > f0 + tol:
                    break
                kept.append(v)
            if len(kept) >= _MIN_STEPS and kept[-1] < f0 - tol:
                running |= set(np.flatnonzero(share).tolist())
    if not running:
        return verdict
    numbers = [j for pos, j in coefs if pos in running]
    names = tuple(name for pos, name in baseline if pos in running)
    return SearchVerdict(
        verdict.res, "no finite maximum", verdict.derivatives, numbers, names
    )


def _check_origin(
    params: npt.NDArray,
    params_c: npt.NDArray,
    fitter: DistributionalFitter,
    links: tuple[str, ...],
    center: npt.NDArray,
) -> None:
    """Refuse a model whose parameters at ``Z = 0`` cannot be
    represented (a covariate far from 0, such as a year)."""
    K = fitter.k_dist
    bad = []
    for k in range(K):
        value = params[k]
        if not np.isfinite(value) or (links[k] == "log" and value <= 0):
            bad.append(fitter.parameter_names[k])
        elif links[k] == "log":
            ratio = np.log(value) - np.log(params_c[k])
            if not abs(ratio) < LOG_MAX:
                bad.append(fitter.parameter_names[k])
    if bad:
        raise ValueError(
            "The {} distribution's {} at Z = 0 cannot be represented for "
            "these covariates (their means are {}): the linear predictor "
            "moves {} out of range between the means and 0. {}".format(
                fitter.dist.name,
                ", ".join(bad),
                np.array2string(np.asarray(center, dtype=float), precision=4),
                "it" if len(bad) == 1 else "them",
                _ORIGIN_HINT,
            )
        )


def Distributional(
    distribution: Any,
    covariates: Mapping[str, Any] | None = None,
    links: Mapping[str, str] | None = None,
) -> DistributionalFitter:
    """
    A distributional regression fitter: the covariates move every
    parameter of ``distribution``, each through its own link (GAMLSS's
    model, on SurPyval's survival data).

    Parameters
    ----------
    distribution : ParametricFitter
        The distribution: Weibull, Gamma, LogNormal, LogLogistic,
        Exponential, Rayleigh, Normal, Gumbel, GumbelLEV, Logistic, Beta
        or ExpoWeibull.
    covariates : dict, optional
        The default of :meth:`DistributionalFitter.fit`'s
        ``covariates``: each parameter's columns.
    links : dict, optional
        The default of :meth:`DistributionalFitter.fit`'s ``links``.

    Returns
    -------
    DistributionalFitter

    Examples
    --------
    >>> from surpyval import Distributional, Weibull
    >>> Distributional(Weibull)
    WeibullDistributional: distributional regression fitter (Weibull baseline)
    >>> Distributional(Weibull).links
    ('log', 'log')
    """
    return DistributionalFitter(distribution, covariates, links)
