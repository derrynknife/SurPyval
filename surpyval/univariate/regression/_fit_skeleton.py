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

import warnings
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from autograd import jacobian
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import (
    bounds_convert,
    preconditioned_bfgs,
)
from surpyval.univariate.parametric.parametric_fitter import Boxable, Numeric
from surpyval.utils import (
    _caller_stacklevel,
    check_covariate_rows,
    finite_covariate_mask,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .parametric_regression_model import ParametricRegressionModel


class LogLinearPhi:
    """The ``exp(beta'Z)`` covariate link shared by PH, AFT and PO —
    previously defined inline in at least seven places (#295).

    ``name`` must match the family's serialisation tag exactly (PH
    historically uses ``e^`` where AFT/PO use ``exp``), so it is a
    constructor argument.
    """

    #: The two historical serialisation names for this link: PH models
    #: are serialised with ``e^``, AFT and PO with ``exp``. Both spell
    #: the same function.
    NAME_E = "Log Linear [e^(beta'Z)]"
    NAME_EXP = "Log Linear [exp(beta'Z)]"

    def __init__(self, name: str, phi_param_map: dict) -> None:
        self.name = name
        self.phi_param_map = phi_param_map

    @staticmethod
    def phi(Z: Numeric, *params: Boxable) -> Boxable:
        return np.exp(np.dot(Z, np.array(params)))

    @staticmethod
    def phi_bounds(Z: npt.NDArray) -> tuple:
        return ((None, None),) * Z.shape[1]

    @staticmethod
    def make_param_map(Z: npt.NDArray) -> dict[str, int]:
        return {"beta_" + str(i): i for i in range(Z.shape[1])}


def make_objective(
    fitter: Any, data: SurpyvalData, inv_trans: Callable, const: Callable
) -> Callable:
    """The optimiser objective every regression fitter used to build
    inline: the fitter's negative log-likelihood evaluated in the
    transformed (unconstrained, fixed-parameters-removed) search space.
    """

    def fun(params: npt.NDArray) -> Boxable:
        return fitter.neg_ll(data, *inv_trans(const(params)))

    return fun


class MirroredDistributionAttrs:
    """Class-level declarations for the attributes
    :func:`mirror_distribution` sets, so a fitter that inherits this
    alongside its other mixins has them visible to the type checker."""

    dist: Any
    k_dist: int
    bounds: tuple
    support: tuple
    param_names: list
    param_map: dict


def mirror_distribution(fitter: Any, distribution: Any) -> None:
    """Copy a distribution's metadata onto a regression fitter.

    Every parametric regression fitter starts by mirroring the same six
    attributes of its underlying distribution -- ``dist``, ``k_dist``,
    ``bounds``, ``support``, ``param_names`` and the name-to-index
    ``param_map`` -- and each family's ``__init__`` carried the block
    verbatim. The ``*_dist`` method aliases stay with each family: which
    ones it needs depends on which identities it implements.
    """
    fitter.dist = distribution
    fitter.k_dist = len(distribution.param_names)
    fitter.bounds = distribution.bounds
    fitter.support = distribution.support
    fitter.param_names = distribution.param_names
    fitter.param_map = {v: i for i, v in enumerate(distribution.param_names)}


class HazardIdentitiesMixin:
    """The standard survival identities in terms of ``Hf``/``hf``.

    Any fitter defining ``Hf(x, Z, *params)`` and ``hf(x, Z, *params)``
    gets ``sf``/``ff``/``df`` and their logs from here instead of
    carrying its own copy (#297). ``ff`` uses ``-expm1(-H)``, which
    stays accurate when ``H`` is tiny (the deep left tail), and
    ``log_df = log(h) - H`` avoids exponentiating and re-logging. For
    additive-hazard models the hazard can be driven non-positive, in
    which case ``log_df`` is nan and the optimiser rejects the point, so
    the fit stays where the hazard is positive at every failure.
    """

    def mpp_x_transform(self, x: Numeric, gamma: Boxable = 0) -> Boxable:
        # Probability-plot x axis: a location shift is the only x
        # transform any of these hazard-based fitters uses.
        return x - gamma

    if TYPE_CHECKING:
        # The host class supplies these; declared rather than defined so
        # a fitter that forgets one gets the AttributeError that names
        # it (the same pattern as ParametricFitter's contract block).
        def Hf(self, x: Any, Z: Any, *params: Any) -> Any: ...
        def hf(self, x: Any, Z: Any, *params: Any) -> Any: ...

    def sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Survival function at ``x`` for covariates ``Z``,
        :math:`R(x \\mid Z) = e^{-H(x \\mid Z)}`.

        ``params`` are the distribution parameters followed by the
        covariate coefficients, in the order of a fitted model's
        ``params``. A fitted model's own ``sf(x, Z)`` supplies them.
        """
        return np.exp(-self.Hf(x, Z, *params))

    def ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Failure (CDF) function at ``x`` for covariates ``Z``,
        :math:`F(x \\mid Z) = 1 - e^{-H(x \\mid Z)}`. ``params`` as for
        :meth:`sf`.
        """
        return -np.expm1(-self.Hf(x, Z, *params))

    def df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Density at ``x`` for covariates ``Z``,
        :math:`f(x \\mid Z) = h(x \\mid Z) e^{-H(x \\mid Z)}`. ``params`` as
        for :meth:`sf`.
        """
        return self.hf(x, Z, *params) * np.exp(-self.Hf(x, Z, *params))

    def log_sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return -self.Hf(x, Z, *params)

    def log_ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return np.log(self.ff(x, Z, *params))

    def log_df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return np.log(self.hf(x, Z, *params)) - self.Hf(x, Z, *params)


def prepare_regression_fit(
    fitter: Any,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: "npt.ArrayLike | None",
    n: "npt.ArrayLike | None",
    t: "npt.ArrayLike | None",
    init: "npt.ArrayLike | None",
    fixed: "dict[str, float] | None",
    phi_bounds: "Callable[[npt.NDArray], tuple] | tuple",
    phi_param_map: "Callable[[npt.NDArray], dict] | dict",
    phi_init: "Callable[[npt.NDArray], npt.NDArray] | None" = None,
) -> tuple[SurpyvalData, tuple]:
    """Common head of every parametric-regression ``fit``.

    Returns ``(data, fun_builder_inputs)`` where the second element is the
    tuple ``(init_t, bounds, pmap, transform, inv_trans, const, not_fixed,
    fixed)`` — everything the family's optimiser step and the final
    assembly need. ``phi_bounds``/``phi_param_map``/``phi_init`` may be
    callables of the covariate array or static values.
    """
    data = SurpyvalData(x, c, n, t, group_and_sort=False)
    # A one-dimensional Z is a single covariate (one value per row), as
    # the Cox, accelerated-life and other fitters already read it.
    Z_in = Z if hasattr(Z, "ndim") else np.asarray(Z)
    if getattr(Z_in, "ndim", 2) == 1:
        Z = np.asarray(Z_in).reshape(-1, 1)
    data, Z = drop_nonfinite_covariates(data, Z)
    data.add_covariates(Z)

    fixed = {} if fixed is None else fixed
    Z_data = np.asarray(data.Z)

    def default_init() -> npt.NDArray:
        ps = fitter.dist.fit_from_surpyval_data(data).params
        if callable(phi_init):
            init_phi = phi_init(Z_data)
        else:
            init_phi = np.zeros(Z_data.shape[1])
        return np.array([*ps, *init_phi])

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
    check_fixed_and_init(fixed, init, param_map)

    user_init = init is not None and len(np.atleast_1d(init)) > 0
    init = np.array(init) if user_init else default_init()

    transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
        data.x, bounds, fixed, param_map
    )
    init_t = transform(init)[not_fixed]
    init_t = finite_start(
        make_objective(fitter, data, inv_trans, const),
        init_t,
        (lambda: transform(default_init())[not_fixed]) if user_init else None,
    )
    return data, (init_t, bounds, pmap, transform, inv_trans, const, fixed)


def drop_nonfinite_covariates(
    data: SurpyvalData, Z: npt.ArrayLike
) -> tuple[SurpyvalData, npt.NDArray]:
    """Check ``Z`` against the data and drop the rows with a non-finite
    covariate from both, with a warning saying how many.

    A NaN covariate makes the likelihood nan wherever the coefficients are
    evaluated, so the optimisers never moved and the fit came back at its
    starting values (with ``res.success`` False and no warning); Cox,
    Lin-Ying and Buckley-James already dropped such rows. A ``Z`` with the
    wrong number of rows is refused by name rather than failing as an
    ``IndexError`` inside the observation-type split.
    """
    Z_arr = np.asarray(Z)
    check_covariate_rows(Z_arr, len(data))
    mask = finite_covariate_mask(Z_arr)
    if mask.all():
        return data, Z_arr
    return data[mask], Z_arr[mask]


def check_fixed_and_init(
    fixed: "dict[str, float] | None",
    init: "npt.ArrayLike | None",
    param_map: dict[str, int],
    always_fixed: "dict[str, float] | None" = None,
) -> None:
    """Refuse ``fixed``/``init`` values that cannot describe this model.

    ``param_map`` maps every parameter name (distribution parameters, then
    covariate coefficients) to its position. Each mistake used to surface
    as an unrelated error from deep inside the transforms: a ``KeyError``
    for an unknown name, "zero-size array" when nothing was left to fit,
    and a ``zip()`` length error for an ``init`` of the wrong length.
    ``always_fixed`` holds parameters the fitter pins itself (the
    accelerated-life placeholder), which count towards "nothing to fit".
    """
    names = sorted(param_map, key=param_map.__getitem__)
    fixed = {} if fixed is None else fixed
    unknown = [k for k in fixed if k not in param_map]
    if unknown:
        raise ValueError(
            "Unknown parameter(s) {} in `fixed`; this model's parameters "
            "are {}.".format(unknown, names)
        )
    if len({**(always_fixed or {}), **fixed}) >= len(param_map):
        raise ValueError(
            "Every parameter is fixed, so there is nothing to fit. Build "
            "the model from known parameters instead, or leave at least "
            "one parameter free."
        )
    if init is not None and len(np.atleast_1d(init)) > 0:
        if len(np.atleast_1d(init)) != len(param_map):
            raise ValueError(
                "`init` has {} value(s) but the model has {} parameters "
                "({}), in that order.".format(
                    len(np.atleast_1d(init)), len(param_map), names
                )
            )


def finite_start(
    fun: Callable,
    init_t: npt.ArrayLike,
    default_init_t: "Callable[[], npt.NDArray] | None",
) -> npt.NDArray:
    """A starting point at which the objective ``fun`` is finite.

    Every optimiser rung starts from ``init_t``, and none of them can move
    off a start where the negative log-likelihood is ``inf`` or ``nan``:
    Nelder-Mead sees the same non-finite value at every vertex and stops,
    returning the start unchanged -- which the fitters then reported as
    the fitted parameters. A user-supplied ``init`` that lands there (a
    fitted parameter vector from another family, say, with the
    coefficients of the opposite sign) falls back to the default start,
    with a warning; if that is non-finite too the fit cannot begin, and
    says so instead of returning a model that was never fitted.
    """
    start: npt.NDArray = np.asarray(init_t)
    with np.errstate(all="ignore"):
        if np.isfinite(fun(start)):
            return start
        if default_init_t is not None:
            alt = default_init_t()
            if np.isfinite(fun(alt)):
                warnings.warn(
                    "The log-likelihood is not finite at the supplied "
                    "`init`, so the fit cannot start there; starting from "
                    "the default initial values instead.",
                    stacklevel=4,
                )
                return alt
    raise ValueError(
        "The log-likelihood is not finite at the initial parameter values, "
        "so the fit cannot start. Supply an `init` at which the model gives "
        "every observation a positive likelihood."
    )


def require_finite_fit(neg_ll: float) -> None:
    """Refuse to return a model whose fitted log-likelihood is not finite.

    Reached only if every optimiser rung failed to find a finite point
    (``finite_start`` guarantees the start was one), so the parameters
    are not an estimate of anything.
    """
    if not np.isfinite(neg_ll):
        raise ValueError(
            "The fit did not converge: the log-likelihood is not finite at "
            "the parameters the optimiser returned."
        )


def assemble_regression_model(
    fitter: Any,
    kind: str,
    reg_model: Any,
    data: SurpyvalData,
    res: Any,
    params: npt.ArrayLike,
    bounds: tuple,
    pmap: dict,
    fixed: dict,
    neg_ll: "float | None" = None,
) -> ParametricRegressionModel:
    """Common tail of every parametric-regression ``fit``."""
    require_finite_fit(float(res.fun) if neg_ll is None else neg_ll)
    model = ParametricRegressionModel()
    model.distribution_param_map = fitter.param_map
    model.phi_param_map = pmap
    model.model = fitter
    model.reg_model = reg_model
    model.kind = kind
    model.distribution = fitter.dist
    params_arr = np.array(params)
    model.params = params_arr
    model.dist_params = np.array(params_arr[: fitter.k_dist])
    model.phi_params = np.array(params_arr[fitter.k_dist :])
    model.res = res
    model._neg_ll = float(res.fun) if neg_ll is None else neg_ll
    model.fixed = fixed
    model.k_dist = fitter.k_dist
    # Estimated parameters only: a parameter held at a value by ``fixed``
    # costs the model nothing in AIC/BIC.
    model.k = len(bounds) - len(fixed)
    model.data = data
    return model


def optimise_ph(fun: Callable, init_t: npt.NDArray) -> Any:
    """Preconditioned BFGS on the analytic gradient, TNC as the fallback.

    The historical ladder was ``minimize(fun, init_t)`` followed by TNC
    kept unconditionally. Three things were wrong with it (#328).

    ``fun`` closes over ``regression_neg_ll``, which is written in
    ``autograd.numpy`` and is therefore differentiable -- but no ``jac``
    was passed, so scipy fell back to a two-point finite difference and
    paid ``p + 1`` extra objective evaluations per gradient. That is
    what made the fit slow down as the covariate count rose.

    Nor was the search preconditioned, so PH inherited the scale
    sensitivity ``preconditioned_bfgs`` was written to cure: on a Weibull
    PH at data scale 1e6 the old ladder settled 1.5 nats of
    log-likelihood short of the optimum, which is a different fitted
    model, not a tolerance artefact.

    Finally TNC's answer was returned whether or not it had succeeded --
    a rung of the ladder that could only ever be an improvement was
    allowed to be a regression. ``optimise_nm_tnc``, immediately below,
    already guarded against that.

    The rungs now stop at the first success, matching the univariate MLE
    ladder, and the derivative-free rung remains for the fits where the
    gradient is unusable (a distribution whose autograd derivative goes
    nan on the way, most often).
    """
    jac = jacobian(fun)

    best = None
    for method in ("BFGS", "TNC", "Nelder-Mead"):
        x0 = init_t if best is None else best.x
        if method == "BFGS":
            res = preconditioned_bfgs(
                fun, x0, jac=jac, options={"maxiter": 1000}
            )
        elif method == "TNC":
            res = minimize(
                fun, x0, method="TNC", jac=jac, options={"maxfun": 1000}
            )
        else:
            res = minimize(
                fun, init_t, method="Nelder-Mead", options={"maxiter": 1000}
            )

        if not np.isfinite(res.fun) or np.isnan(res.x).any():
            continue
        if best is None or res.fun < best.fun:
            best = res
        if res.success:
            break

    if best is None:
        # Every rung produced a nan; hand back the last one so the caller
        # sees a failed OptimizeResult rather than a None (and
        # ``require_finite_fit`` refuses it).
        return res
    if not best.success and not _is_stationary(
        _gradient(fun, best.x), best.fun
    ):
        # BFGS routinely stops on "precision loss" at the optimum itself;
        # only a point where the gradient is not zero is a failure.
        warn_if_not_converged(best)
    return best


def warn_if_not_converged(res: Any) -> None:
    """Say so when no optimiser rung converged.

    The best point found is still returned, but not silently: it used to
    come back with ``res.success`` False and nothing said, which is how a
    nan covariate passed off the starting values as a fit.
    """
    if not res.success:
        warnings.warn(
            "The optimiser did not converge ({}); the fitted parameters may "
            "not be the maximum-likelihood estimates. Check the data, or "
            "supply a better `init`.".format(str(res.message).rstrip(".")),
            stacklevel=_caller_stacklevel(),
        )


def optimise_nm_tnc(fun: Callable, init_t: npt.NDArray) -> Any:
    """AFT/PO's historical ladder: Nelder-Mead, then TNC kept only on
    success -- and, when that ladder has not reached a stationary point,
    the gradient-based :func:`optimise_ph` ladder from where it stopped.

    Nelder-Mead runs out of iterations on the harder fits (four
    covariates on 34 tires, say), and TNC can then fail as well, or report
    success where the gradient is plainly not zero; either way the fit
    used to come back tenths of a nat short of the maximum, silently. A
    converged fit is returned exactly as before.
    """
    res = minimize(
        fun, init_t, method="Nelder-Mead", options={"maxiter": 1000}
    )
    res2 = minimize(fun, res.x, method="TNC")
    best = res2 if res2.success else res
    g = _gradient(fun, best.x)
    if g is None:
        # An objective autograd cannot differentiate (the AFT
        # time-varying likelihood): the optimiser's verdict is all there is.
        warn_if_not_converged(best)
        return best
    if best.success and _is_stationary(g, best.fun):
        return best
    polished = optimise_ph(fun, best.x)  # warns if it cannot converge
    if np.isfinite(polished.fun) and polished.fun <= best.fun:
        return polished
    return best


def _gradient(fun: Callable, x: npt.NDArray) -> "npt.NDArray | None":
    """The autograd gradient of ``fun`` at ``x``, or ``None`` where it is
    unavailable (an objective written with in-place numpy) or not finite
    (a distribution whose derivative goes nan)."""
    try:
        with np.errstate(all="ignore"):
            g = np.asarray(jacobian(fun)(x), dtype=float)
    except (TypeError, ValueError):
        return None
    return g if np.all(np.isfinite(g)) else None


def _is_stationary(g: "npt.NDArray | None", f: float) -> bool:
    """Whether the gradient ``g`` is small next to the objective value
    ``f`` -- so a reported success really is an optimum. With no usable
    gradient the optimiser's own verdict stands."""
    if g is None:
        return True
    return bool(np.max(np.abs(g), initial=0.0) <= 1e-2 * max(1.0, abs(f)))
