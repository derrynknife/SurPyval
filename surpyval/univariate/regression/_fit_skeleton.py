"""Shared ``fit()`` plumbing for the parametric regression families.

PH, AFT, PO and (parametric) AH used to carry five copy-pasted versions
of the same skeleton — data prep, the #251 param-map offset merge,
``bounds_convert``, optimisation, and model assembly — which had already
drifted (different optimiser ladders, only some families setting
``dist_params``/``phi_params``). The skeleton now lives here once; each
family supplies only its optimiser strategy and covariate-link object
(#295). The Accelerated Life parameter-substitution fitter remains
separate: its life-model parameter juggling does not fit this shape.
"""

from typing import Any, Callable

import autograd.numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from .parametric_regression_model import ParametricRegressionModel


class LogLinearPhi:
    """The ``exp(beta'Z)`` covariate link shared by PH, AFT and PO —
    previously defined inline in at least seven places (#295).

    ``name`` must match the family's serialisation tag exactly (PH
    historically uses ``e^`` where AFT/PO use ``exp``), so it is a
    constructor argument.
    """

    def __init__(self, name: str, phi_param_map: dict):
        self.name = name
        self.phi_param_map = phi_param_map

    @staticmethod
    def phi(Z, *params):
        return np.exp(np.dot(Z, np.array(params)))

    @staticmethod
    def phi_bounds(Z):
        return ((None, None),) * Z.shape[1]

    @staticmethod
    def make_param_map(Z):
        return {"beta_" + str(i): i for i in range(Z.shape[1])}


class HazardIdentitiesMixin:
    """The standard survival identities in terms of ``Hf``/``hf``.

    Any fitter defining ``Hf(x, Z, *params)`` and ``hf(x, Z, *params)``
    gets ``sf``/``ff``/``df`` and their logs from here instead of
    carrying its own copy (#297). ``ff`` uses ``-expm1(-H)``, which
    stays accurate when ``H`` is tiny (the deep left tail), and
    ``log_df = log(h) - H`` avoids exponentiating and re-logging. For
    additive-hazard models the hazard can be driven non-positive, in
    which case ``log_df`` is nan and the MLE machinery rejects the
    point — the fit fails rather than returning an invalid model.
    """

    def sf(self, x, Z, *params):
        return np.exp(-self.Hf(x, Z, *params))

    def ff(self, x, Z, *params):
        return -np.expm1(-self.Hf(x, Z, *params))

    def df(self, x, Z, *params):
        return self.hf(x, Z, *params) * np.exp(-self.Hf(x, Z, *params))

    def log_sf(self, x, Z, *params):
        return -self.Hf(x, Z, *params)

    def log_ff(self, x, Z, *params):
        return np.log(self.ff(x, Z, *params))

    def log_df(self, x, Z, *params):
        return np.log(self.hf(x, Z, *params)) - self.Hf(x, Z, *params)


def prepare_regression_fit(
    fitter: Any,
    x,
    Z,
    c,
    n,
    t,
    init,
    fixed,
    phi_bounds,
    phi_param_map,
    phi_init=None,
):
    """Common head of every parametric-regression ``fit``.

    Returns ``(data, fun_builder_inputs)`` where the second element is the
    tuple ``(init_t, bounds, pmap, transform, inv_trans, const, not_fixed,
    fixed)`` — everything the family's optimiser step and the final
    assembly need. ``phi_bounds``/``phi_param_map``/``phi_init`` may be
    callables of the covariate array or static values.
    """
    data = SurpyvalData(x, c, n, t, group_and_sort=False)
    data.add_covariates(Z)

    fixed = {} if fixed is None else fixed
    Z_data = np.asarray(data.Z)

    if init is None or len(np.atleast_1d(init)) == 0:
        ps = fitter.dist.fit_from_surpyval_data(data).params
        if callable(phi_init):
            init_phi = phi_init(Z_data)
        else:
            init_phi = np.zeros(Z_data.shape[1])
        init = np.array([*ps, *init_phi])
    else:
        init = np.array(init)

    bounds = (
        *fitter.bounds,
        *(phi_bounds(Z_data) if callable(phi_bounds) else phi_bounds),
    )
    pmap = phi_param_map(Z_data) if callable(phi_param_map) else phi_param_map
    # The covariate coefficients sit after the distribution parameters in
    # the packed parameter vector, so their map indices must be offset by
    # the number of distribution parameters — otherwise
    # ``fixed={"beta_0": v}`` silently pins the first *distribution*
    # parameter instead (#251).
    param_map = {
        **fitter.param_map,
        **{k: v + len(fitter.param_map) for k, v in pmap.items()},
    }

    transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
        data.x, bounds, fixed, param_map
    )
    init_t = transform(init)[not_fixed]
    return data, (init_t, bounds, pmap, transform, inv_trans, const, fixed)


def assemble_regression_model(
    fitter: Any,
    kind: str,
    reg_model: Any,
    data: SurpyvalData,
    res: Any,
    params,
    bounds,
    pmap: dict,
    fixed: dict,
    neg_ll: "float | None" = None,
) -> ParametricRegressionModel:
    """Common tail of every parametric-regression ``fit``."""
    model = ParametricRegressionModel()
    model.distribution_param_map = fitter.param_map
    model.phi_param_map = pmap
    model.model = fitter
    model.reg_model = reg_model
    model.kind = kind
    model.distribution = fitter.dist
    model.params = np.array(params)
    model.dist_params = np.array(params[: fitter.k_dist])
    model.phi_params = np.array(params[fitter.k_dist :])
    model.res = res
    model._neg_ll = float(res.fun) if neg_ll is None else neg_ll
    model.fixed = fixed
    model.k_dist = fitter.k_dist
    model.k = len(bounds)
    model.data = data
    return model


def optimise_ph(fun: Callable, init_t):
    """PH's historical ladder: default method, then TNC unconditionally."""
    res = minimize(fun, init_t)
    return minimize(fun, res.x, method="TNC")


def optimise_nm_tnc(fun: Callable, init_t):
    """AFT/PO's historical ladder: Nelder-Mead, then TNC kept only on
    success."""
    res = minimize(
        fun, init_t, method="Nelder-Mead", options={"maxiter": 1000}
    )
    res2 = minimize(fun, res.x, method="TNC")
    return res2 if res2.success else res
