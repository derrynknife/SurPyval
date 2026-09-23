import warnings
from typing import Callable

import autograd.numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import HazardIdentitiesMixin, make_objective
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from .lifemodel import LifeModel


class ParameterSubstitutionFitter(
    HazardIdentitiesMixin, DataFrameRegressionMixin
):
    def __init__(
        self,
        kind: str,
        name: str,
        distribution: OptimisedFitMixin,
        life_model: LifeModel,
        life_parameter: str,
        baseline: list[str] | str | None = None,
        param_transform: Callable[[Boxable], Boxable] | None = None,
        inverse_param_transform: Callable[[Boxable], Boxable] | None = None,
    ) -> None:
        if baseline is None:
            baseline = []
        elif not isinstance(baseline, list):
            # Baseline used if using a function that deviates from some number,
            # e.g. np.exp(np.dot(Z, beta))
            baseline = [baseline]

        self.name = name
        self.kind = kind
        self.dist = distribution
        self.life_model = life_model
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v: i for i, v in enumerate(self.dist.param_names)}
        self.phi = life_model.phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.baseline = baseline
        self.life_parameter = life_parameter
        self.fixed = {life_parameter: 1.0}

        if param_transform is None:
            self.param_transform = lambda x: x
            self.inverse_param_transform = lambda x: x
        else:
            # Supplied as a pair -- accelerated_life.py passes both or
            # neither -- so the inverse is not None here.
            assert inverse_param_transform is not None
            self.param_transform = param_transform
            self.inverse_param_transform = inverse_param_transform

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.array(x)
        if np.isscalar(Z):
            Z_arr = np.ones_like(x) * Z
        else:
            Z_arr = np.array(Z)
        if Z_arr.ndim == 1:
            # A 1-D stress vector (one stress variable) becomes a single
            # column so the per-stress masking below works (#261).
            Z_arr = Z_arr.reshape(-1, 1)

        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        Hf = np.zeros_like(x)
        stresses = np.unique(Z_arr, axis=0)
        for stress in stresses:
            life_param_mask = (
                np.arange(len(dist_params))
                == self.param_map[self.life_parameter]
            )
            dist_params_i = np.where(
                life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params,
            )
            mask = (Z_arr == stress).all(axis=1)
            Hf = np.where(mask, self.Hf_dist(x, *dist_params_i), Hf)

        return Hf

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.array(x)
        if np.isscalar(Z):
            Z_arr = np.ones_like(x) * Z
        else:
            Z_arr = np.array(Z)
        if Z_arr.ndim == 1:
            # A 1-D stress vector (one stress variable) becomes a single
            # column so the per-stress masking below works (#261).
            Z_arr = Z_arr.reshape(-1, 1)

        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        hf = np.zeros_like(x)
        for stress in np.unique(Z_arr, axis=0):
            life_param_mask = (
                np.arange(len(dist_params))
                == self.param_map[self.life_parameter]
            )
            dist_params_i = np.where(
                life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params,
            )
            mask = (Z_arr == stress).all(axis=1)
            hf = np.where(mask, self.hf_dist(x, *dist_params_i), hf)

        return hf

    # sf/ff/df and the log identities come from HazardIdentitiesMixin;
    # Hf and hf above already do the scalar/1-D stress coercion (#261),
    # so the identities need no preamble of their own.

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def random(
        self, size: int, Z: Numeric | tuple[float, float], *params: Boxable
    ) -> tuple[npt.NDArray, npt.NDArray]:
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        x = []
        Z_out = []
        if isinstance(Z, tuple):
            # A (low, high) pair draws the stresses uniformly.
            Z = np.random.uniform(*Z, size)
        Z_arr = np.asarray(Z)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        for stress in np.unique(Z_arr, axis=0):
            life_param_mask = (
                np.arange(len(dist_params))
                == self.param_map[self.life_parameter]
            )
            dist_params_i = np.where(
                life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params,
            )

            U = np.random.uniform(0, 1, size)
            x.append(self.dist.qf(U, *dist_params_i))
            if np.isscalar(stress):
                cols = 1
            else:
                cols = len(stress)
            Z_out.append(np.ones((size, cols)) * stress)
        return np.array(x).flatten(), np.concatenate(Z_out)

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
    ) -> ParametricRegressionModel:
        x_arr: npt.NDArray = np.asarray(x)
        data = SurpyvalData(x=x, c=c, n=n, t=t, group_and_sort=False)
        # A 1-D stress vector (one stress variable) becomes a single column
        # so the per-stress masking in the initialiser works (#261).
        Z_arr = np.asarray(Z)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        data.add_covariates(Z_arr)
        life_parameter_idx = self.param_map[self.life_parameter]
        if fixed is None:
            fixed = {}
        if init is None or len(init) == 0:  # type: ignore[arg-type]
            stress_data = []
            params_at_Z = []

            # How do I make this work when there is only one failure per
            # stress?
            base_line_dist_init = self.dist.fit_from_surpyval_data(data).params

            for s in np.unique(data.Z, axis=0):
                mask = (data.Z == s).all(axis=1)
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        params_at_s = self.dist.fit_from_surpyval_data(
                            data[mask]
                        ).params
                        params_at_Z.append(params_at_s)
                    except Exception:
                        params_at_s = np.copy(base_line_dist_init)
                        params_at_s[life_parameter_idx] = x_arr[mask].mean()
                        params_at_Z.append(params_at_s)
                    finally:
                        stress_data.append(s)

            params_at_Z = np.array(params_at_Z)
            dist_init = params_at_Z.mean(axis=0)

            stress_data = np.array(stress_data)

            if len(params_at_Z) < 2:
                raise ValueError(
                    "Insufficient data at separate Z values. Try manually \
                    setting initial guess using `init` keyword in `fit`"
                )

            parameter_data = params_at_Z[:, life_parameter_idx]

            parameter_data = self.inverse_param_transform(parameter_data)

            # Every life model's phi_init is (life, Z). There used to be
            # a branch here for a "(Z)"-only signature, chosen by
            # comparing str(inspect.signature(...)) == "(Z)", and another
            # for a non-callable phi_init. Neither could run: all ten
            # life models are callable with the two-argument signature.
            phi_init = self.life_model.phi_init(parameter_data, stress_data)
            init = np.array([*dist_init, *phi_init])
        else:
            init = np.array(init)

        if self.baseline != []:
            baseline_model = self.dist.fit_from_surpyval_data(data)
            baseline_fixed = {
                k: baseline_model.params[baseline_model.param_map[k]]
                for k in self.baseline
            }
            fixed = {**baseline_fixed, **fixed}

        if self.fixed != {}:
            fixed = {**self.fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.life_model.phi_bounds):
            bounds = (*self.bounds, *self.life_model.phi_bounds(data.Z))
        else:
            bounds = (*self.bounds, *self.life_model.phi_bounds)

        if callable(self.life_model.phi_param_map):
            phi_param_map = self.life_model.phi_param_map(data.Z)
        else:
            phi_param_map = self.life_model.phi_param_map

        # Keep the merged map local: assigning it to ``self.param_map``
        # mutated the fitter, so a second ``fit()`` re-merged on top of the
        # already-merged map and produced out-of-range indices (#261).
        param_map = {
            **self.param_map,
            **{k: v + len(self.param_map) for k, v in phi_param_map.items()},
        }

        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            x, bounds, fixed, param_map
        )

        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            fun = make_objective(self, data, inv_trans, const)

            res1 = minimize(
                fun, init, method="Nelder-Mead", options={"maxiter": 1000}
            )
            res2 = minimize(
                fun,
                res1.x,
                method="TNC",
                # tol=1e-20,
                # options={"maxiter": 1000},
            )
            if not res2.success:
                res = res1
            else:
                res = res2

        params = inv_trans(const(res.x))
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        model = ParametricRegressionModel()
        model.model = self
        model.kind = self.kind
        model.distribution = self.dist
        model.reg_model = self.life_model
        model.params = np.array(params)
        model.dist_params = dist_params
        model.phi_params = phi_params
        model.res = res
        model._neg_ll = res.fun
        # Store the full merged fixed dict (baseline-derived + fitter-level +
        # user-supplied), not just the fitter's own — otherwise standard
        # errors are reported for parameters that were held fixed (#261).
        model.fixed = fixed
        model.k_dist = self.k_dist
        model.fun = fun

        model.k = len(bounds)

        model.data = {"x": x, "c": c, "n": n, "t": t}
        model.data = data

        return model
