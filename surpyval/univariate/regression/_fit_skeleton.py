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

import copy
import warnings
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from autograd import (
    elementwise_grad,
    grad,
    hessian,
    jacobian,
    value_and_grad,
)
from autograd.differential_operators import make_hvp, make_vjp
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
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import SurpyvalData

from ._aliasing import aliased_columns, constant_columns, warn_aliased
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
        # A coefficient running off on separated data makes beta'Z large;
        # exp overflows to inf, the right limit (survival 0), and must not
        # leak numpy's raw overflow warning (principle 22).
        with np.errstate(over="ignore"):
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


# -- covariate centring (#463) ------------------------------------------------
#
# A log-linear link evaluates exp(beta'Z), which on a covariate far from 0
# (a year, a date as a day count) overflows at the optimiser's first trial
# steps, and the fit wandered off to a wrong answer silently (a WeibullPH
# beta of 0.0247 instead of 0.907 with 2000 added to the covariate).
#
# ``fit(..., center=True)`` fits on Z - center, center the n-weighted
# covariate means, as CoxPH does (#459), and the model keeps its baseline
# there (``model.center``): exp(beta'(Z - center)) stays near 1 on the data
# whatever the covariates' origin. That works for every family and link.
#
# By default the baseline is reported at Z = 0, as it always was. For the
# family/link pairs below, centring is an exact reparameterisation -- the
# model with its baseline at the centre is the model with its baseline at
# Z = 0, with the baseline parameters moved by the linear predictor at the
# centre, s = beta'center -- so the fit still runs on centred covariates,
# and its parameters are mapped back to Z = 0, where they must be
# representable (else the fit says so, and points to ``center=True``).
# Each entry maps the distribution parameters at the centre to those at
# Z = 0 (the map with -s is its inverse) and names the parameters it moves.
#
# - PH, H(x | Z) = exp(beta'Z) H0(x): H0 at Z = 0 is exp(-s) times H0 at
#   the centre -- a scale change for Weibull, Rayleigh and Exponential, a
#   location shift for Gumbel (H0 = exp((x - mu) / sigma)).
# - AFT, H(x | Z) = H0(exp(beta'Z) x): time at Z = 0 is time at the
#   centre scaled by exp(s), so any scale family maps (a scale parameter
#   times exp(s), a rate times exp(-s), a log-time location plus s, and
#   both location and scale times exp(s) for a distribution on the whole
#   line).
# - PO, S/F(x | Z) = exp(beta'Z) S/F_0(x): the survival odds at Z = 0 are
#   exp(-s) times those at the centre, a scale change for LogLogistic
#   (odds (x / alpha)^-beta) and a location shift for Logistic (odds
#   exp(-(x - mu) / sigma)).
#
# Every other pair (LogNormal, Gamma, Normal or Logistic PH, and PO with
# any baseline but those two) is not closed under the change: the model
# with its baseline at the covariate means is a different model from the
# one with its baseline at 0, with a different maximum likelihood. By
# default those are fitted at Z = 0 as they always were, on the
# covariates as given, as are the additive hazards models, whose term
# beta'Z has no exp to overflow. On covariates far from 0 such a fit may
# fail; ``center=True`` is the way round it.


def _ph_weibull(p: Any, s: Any) -> list:
    return [p[0] * np.exp(s / p[1]), p[1]]


def _rate_down(p: Any, s: Any) -> list:
    return [p[0] * np.exp(-s)]


def _first_scale_up(p: Any, s: Any) -> list:
    return [p[0] * np.exp(s), *p[1:]]


def _both_scale_up(p: Any, s: Any) -> list:
    return [p[0] * np.exp(s), p[1] * np.exp(s)]


ORIGIN_MAPS: "dict[tuple[str, str], tuple[tuple[int, ...], Callable]]" = {
    ("Proportional Hazard", "Weibull"): ((0,), _ph_weibull),
    ("Proportional Hazard", "Exponential"): ((0,), _rate_down),
    ("Proportional Hazard", "Rayleigh"): (
        (0,),
        lambda p, s: [p[0] * np.exp(s / 2.0)],
    ),
    ("Proportional Hazard", "Gumbel"): (
        (0,),
        lambda p, s: [p[0] + p[1] * s, p[1]],
    ),
    ("Accelerated Failure Time", "Weibull"): ((0,), _first_scale_up),
    ("Accelerated Failure Time", "Exponential"): ((0,), _rate_down),
    ("Accelerated Failure Time", "Rayleigh"): ((0,), _first_scale_up),
    ("Accelerated Failure Time", "LogLogistic"): ((0,), _first_scale_up),
    ("Accelerated Failure Time", "ExpoWeibull"): ((0,), _first_scale_up),
    ("Accelerated Failure Time", "Gamma"): (
        (1,),
        lambda p, s: [p[0], p[1] * np.exp(-s)],
    ),
    ("Accelerated Failure Time", "LogNormal"): (
        (0,),
        lambda p, s: [p[0] + s, p[1]],
    ),
    ("Accelerated Failure Time", "Normal"): ((0, 1), _both_scale_up),
    ("Accelerated Failure Time", "Logistic"): ((0, 1), _both_scale_up),
    ("Accelerated Failure Time", "Gumbel"): ((0, 1), _both_scale_up),
    ("Accelerated Failure Time", "GumbelLEV"): ((0, 1), _both_scale_up),
    ("Proportional Odds", "LogLogistic"): (
        (0,),
        lambda p, s: [p[0] * np.exp(-s / p[1]), p[1]],
    ),
    ("Proportional Odds", "Logistic"): (
        (0,),
        lambda p, s: [p[0] - p[1] * s, p[1]],
    ),
}

#: The hint every refusal below ends with.
_CENTER_HINT = (
    "Fit with center=True to report the baseline at the covariate means "
    "(model.center) instead, or move the covariates nearer 0."
)


def covariate_center(Z: npt.ArrayLike, n: npt.ArrayLike) -> npt.NDArray:
    """The ``n``-weighted mean of the covariate rows, where a centred fit
    puts its baseline (#459, #463)."""
    Z_arr = np.asarray(Z, dtype=float)
    n_arr = np.asarray(n, dtype=float).reshape(-1)
    return np.dot(n_arr, Z_arr) / n_arr.sum()


class Centring:
    """A fit on the centred covariates ``Z - center``.

    With ``move`` (a map of :data:`ORIGIN_MAPS`) the fitted parameters are
    moved to ``Z = 0`` afterwards: ``to_origin`` moves a full parameter
    vector (distribution parameters, then coefficients) from the centre to
    ``Z = 0``, and ``from_origin`` back. Without, the model keeps its
    baseline at the centre (``fit(..., center=True)``).
    """

    #: The data as given (uncentred covariates), which the fitted model
    #: keeps; set by :func:`prepare_regression_fit`.
    raw: "SurpyvalData | None" = None

    def __init__(
        self,
        center: npt.NDArray,
        k_dist: int,
        move: "Callable | None" = None,
    ):
        self.center = np.asarray(center, dtype=float)
        self.k_dist = k_dist
        self._move = move

    @classmethod
    def plan(
        cls,
        fitter: Any,
        kind: "str | None",
        Z: npt.NDArray,
        n: npt.NDArray,
        fixed: dict,
        center: bool = False,
    ) -> "Centring | None":
        """The centring of a fit, or ``None`` to fit on ``Z`` as given.

        ``center=True`` always centres, and keeps the baseline there.
        Otherwise the fit is centred only where the baseline maps back to
        ``Z = 0`` exactly (``kind`` names the log-linear family) and it
        matters: not for covariates whose means are already 0, nor where
        ``fixed`` holds a baseline parameter the map moves (a value fixed
        at ``Z = 0`` is not a fixed value at the centre)."""
        if np.size(Z) == 0:
            return None
        if center:
            return cls(covariate_center(Z, n), fitter.k_dist)
        entry = ORIGIN_MAPS.get((kind or "", getattr(fitter.dist, "name", "")))
        if entry is None:
            return None
        moved, move = entry
        if any(fitter.param_names[i] in fixed for i in moved):
            return None
        mean = covariate_center(Z, n)
        if not np.all(np.isfinite(mean)) or not np.any(mean):
            return None
        return cls(mean, fitter.k_dist, move)

    @property
    def maps_back(self) -> bool:
        """Whether the fitted baseline is moved to ``Z = 0``."""
        return self._move is not None

    def _shifted(self, params: Any, sign: float) -> Any:
        dist = params[: self.k_dist]
        beta = params[self.k_dist :]
        s = sign * np.dot(beta, self.center)
        assert self._move is not None
        return np.concatenate([np.array(self._move(dist, s)), beta])

    def to_origin(self, params: Any) -> Any:
        """The parameters with the baseline at the centre moved to Z = 0."""
        return self._shifted(params, 1.0)

    def from_origin(self, params: Any) -> Any:
        """The parameters with the baseline at Z = 0 moved to the centre."""
        return self._shifted(params, -1.0)

    def finish(
        self,
        params_c: npt.NDArray,
        neg_ll_c: float,
        raw_neg_ll: Callable,
        bounds: tuple,
        dist_name: str = "",
    ) -> "tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]":
        """``(params, center, jacobian)`` of the fitted model.

        A baseline kept at the centre is returned as fitted, with
        ``jacobian`` ``None``. Otherwise the parameters of the centred fit
        ``params_c`` are moved to ``Z = 0``, which must be representable:
        the moved values finite and inside their bounds, and the
        log-likelihood of the data as given, ``raw_neg_ll``, reproducing
        the centred fit's ``neg_ll_c`` (it overflows otherwise --
        ``exp(beta'Z)`` on the raw covariates -- or loses the precision the
        fit had); a ``ValueError`` says so if not. The model then has a
        zero centre, and ``jacobian``, the derivative of the move, carries
        the covariance over.
        """
        params_c = np.asarray(params_c, dtype=float)
        if not self.maps_back:
            return params_c, self.center, None
        with np.errstate(all="ignore"):
            params_0 = np.asarray(self.to_origin(params_c), dtype=float)
            ok = bool(np.all(np.isfinite(params_0)))
            for value, (lower, upper) in zip(params_0, bounds):
                if (lower is not None and not value > lower) or (
                    upper is not None and not value < upper
                ):
                    ok = False
            if ok:
                ll_0 = float(raw_neg_ll(*params_0))
                ok = bool(np.isfinite(ll_0)) and abs(
                    ll_0 - neg_ll_c
                ) <= 1e-8 * max(1.0, abs(neg_ll_c))
            if ok:
                J = np.asarray(jacobian(self.to_origin)(params_c), float)
                ok = bool(np.all(np.isfinite(J)))
        if not ok:
            s = float(np.dot(params_c[self.k_dist :], self.center))
            raise ValueError(
                "The baseline at Z = 0 cannot be represented for these "
                "covariates: their means are {} and the linear predictor "
                "there is beta'center = {:.4g}, so the {} baseline at Z = 0 "
                "(parameters {}, moved from {} at the means) over- or "
                "underflows, and exp(beta'Z) on the covariates as given "
                "with it. {}".format(
                    np.array2string(self.center, precision=4),
                    s,
                    dist_name,
                    np.array2string(
                        params_0[: self.k_dist], precision=4, separator=", "
                    ),
                    np.array2string(
                        params_c[: self.k_dist], precision=4, separator=", "
                    ),
                    _CENTER_HINT,
                )
            )
        return params_0, np.zeros_like(self.center), J


def centred_copy(data: SurpyvalData, center: npt.NDArray) -> SurpyvalData:
    """A shallow copy of ``data`` whose covariates are ``Z - center``."""
    out = copy.copy(data)
    out.add_covariates(np.asarray(data.Z, dtype=float) - center)
    return out


def uniform_draws(size: int, random_state: Any = None) -> npt.NDArray:
    """``size`` uniform draws on ``(0, 1)`` for the inverse-transform
    samplers (``random``) of the regression fitters.

    ``random_state=None`` draws from numpy's global generator, as these
    samplers always have, so ``np.random.seed`` reproduces a draw (and
    gives the same draws as before ``random_state`` existed). Anything
    else is a stream of its own (:func:`surpyval.utils.rng.as_generator`)
    that neither depends on nor advances the global one.
    """
    if random_state is None:
        return np.random.uniform(0, 1, size)
    return as_generator(random_state).uniform(0, 1, size)


class _Fixed(dict):
    """The ``fixed`` of a fit, with the aliased coefficients (#476) held
    at 0 among them; ``aliased`` names those, which the fitted model
    reports as ``nan`` rather than as fixed."""

    aliased: "tuple[str, ...]" = ()


def alias_coefficients(
    fitter: Any,
    kind: "str | None",
    Z: npt.NDArray,
    n: npt.NDArray,
    fixed: dict,
    pmap: dict,
) -> dict:
    """``fixed`` with the coefficients the data cannot determine held at
    0 and named, with one warning (#476; see :mod:`._aliasing`).

    Only where each coefficient multiplies one column of ``Z``
    (``beta_j`` for column ``j``). A constant column is aliased where the
    family has an intercept -- where adding a constant to the linear
    predictor moves the baseline parameters and nothing else
    (:data:`ORIGIN_MAPS`, a scale family, as R's ``survreg`` and ``lm``
    treat an intercept) -- and otherwise only a column of zeros is.
    Columns whose coefficient the caller fixed are offsets, left out.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] == 0 or Z.shape[0] == 0:
        return fixed
    p = Z.shape[1]
    if pmap != {"beta_{}".format(j): j for j in range(p)}:
        return fixed
    free = np.array([j for j in range(p) if "beta_{}".format(j) not in fixed])
    if free.size == 0:
        return fixed
    n = np.asarray(n, dtype=float).reshape(-1)
    Zf = Z[:, free]
    intercept = (kind or "", getattr(fitter.dist, "name", "")) in ORIGIN_MAPS
    if intercept:
        Zf = Zf - covariate_center(Zf, n)
        constant = constant_columns(Z[:, free])
    else:
        constant = np.all(Zf == 0, axis=0)
    gram = Zf.T @ (n[:, None] * Zf)
    aliased = free[aliased_columns(gram, Z.shape[0], constant)]
    if aliased.size == 0:
        return fixed
    warn_aliased(
        aliased,
        (
            "they are constant (the baseline distribution's scale is the "
            "model's intercept) or a linear combination of the other "
            "columns"
            if intercept
            else "they are all zero or a linear combination of the other "
            "columns"
        ),
    )
    names = tuple("beta_{}".format(j) for j in aliased.tolist())
    out = _Fixed({**fixed, **{name: 0.0 for name in names}})
    out.aliased = names
    return out


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
    kind: "str | None" = None,
    center: bool = False,
) -> tuple[SurpyvalData, tuple]:
    """Common head of every parametric-regression ``fit``.

    Returns ``(data, fun_builder_inputs)`` where the second element is the
    tuple ``(init_t, bounds, pmap, transform, inv_trans, const, fixed,
    centring)`` — everything the family's optimiser step and the
    final assembly need. ``phi_bounds``/``phi_param_map``/``phi_init`` may
    be callables of the covariate array or static values.

    ``kind`` names a log-linear family (``"Proportional Hazard"``, ...),
    whose fit runs on centred covariates where its baseline maps back to
    ``Z = 0`` exactly, and ``center=True`` centres any family and keeps the
    baseline at the covariate means (:class:`Centring`). ``data`` is then
    the centred copy the objective uses, an ``init`` given at ``Z = 0`` is
    moved to the centre, and ``centring`` (else ``None``) keeps the data as
    given for :func:`assemble_regression_model`.
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
    fixed = alias_coefficients(
        fitter,
        kind,
        Z_data,
        data.n,
        fixed,
        phi_param_map(Z_data) if callable(phi_param_map) else phi_param_map,
    )
    centring = Centring.plan(fitter, kind, Z_data, data.n, fixed, center)
    if centring is not None:
        centring.raw = data
        data = centred_copy(data, centring.center)

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
    if not user_init:
        init = default_init()
    elif centring is None or not centring.maps_back:
        init = np.array(init)
    else:
        # A user's ``init`` has its baseline at Z = 0, as a fitted model
        # reports it; the search runs with the baseline at the centre.
        with np.errstate(all="ignore"):
            init = centring.from_origin(np.asarray(init, dtype=float))

    transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
        data.x, bounds, fixed, param_map
    )
    with np.errstate(all="ignore"):
        init_t = transform(init)[not_fixed]
    init_t = finite_start(
        make_objective(fitter, data, inv_trans, const),
        init_t,
        (lambda: transform(default_init())[not_fixed]) if user_init else None,
    )
    return data, (
        init_t,
        bounds,
        pmap,
        transform,
        inv_trans,
        const,
        fixed,
        centring,
    )


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
    ``IndexError`` inside the observation-type split. ``Z`` is read as
    floats, so a ``None`` in a list is a missing value too: it was kept as
    an object array, and the fit failed after the drop with a TypeError.
    """
    Z_arr = np.asarray(Z, dtype=float)
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
    centring: "Centring | None" = None,
) -> ParametricRegressionModel:
    """Common tail of every parametric-regression ``fit``.

    With a ``centring``, ``data`` and ``params`` are those of the centred
    fit: the model keeps the data as given, and its parameters and
    ``center`` are placed by :meth:`Centring.finish`.
    """
    require_finite_fit(float(res.fun) if neg_ll is None else neg_ll)
    fit_centring = None
    center = np.zeros(np.shape(data.Z)[1])
    if centring is not None and centring.raw is not None:
        raw = centring.raw
        params_c = np.asarray(params, dtype=float)
        params, center, J = centring.finish(
            params_c,
            float(res.fun) if neg_ll is None else neg_ll,
            lambda *p: fitter.neg_ll(raw, *p),
            bounds,
            fitter.dist.name,
        )
        if J is not None:
            fit_centring = (params_c, centring.center, J)
        data = raw
    model = ParametricRegressionModel()
    model.center = center
    model._fit_centring = fit_centring
    model.distribution_param_map = fitter.param_map
    model.phi_param_map = pmap
    model.model = fitter
    model.reg_model = reg_model
    model.kind = kind
    model.distribution = fitter.dist
    params_arr = np.array(params)
    aliased = getattr(fixed, "aliased", ())
    if aliased:
        # Reported as nan, R's NA (#476); the model predicts with 0.
        params_arr = np.array(params_arr, dtype=float)
        params_arr[[fitter.k_dist + pmap[name] for name in aliased]] = np.nan
    model.params = params_arr
    model.dist_params = np.array(params_arr[: fitter.k_dist])
    model.phi_params = np.array(params_arr[fitter.k_dist :])
    model.res = res
    model._neg_ll = float(res.fun) if neg_ll is None else neg_ll
    model.fixed = {k: v for k, v in fixed.items() if k not in aliased}
    model.k_dist = fitter.k_dist
    # Estimated parameters only: a parameter held at a value by ``fixed``
    # costs the model nothing in AIC/BIC.
    model.k = len(bounds) - len(fixed)
    model.data = data
    return model


def optimise_ph(
    fun: Callable, init_t: npt.NDArray, quiet: bool = False
) -> Any:
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

    A result that is not verifiably an optimum is flagged
    ``stopped_short``, and warned of unless ``quiet`` (the caller then
    warns through :func:`finish_search`).
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
        best.stopped_short = True
        if not quiet:
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


# -- no finite maximum (#392) -------------------------------------------------
#
# A covariate that separates the events from the survivors -- one level of it
# with no events, say -- gives a likelihood that keeps increasing as its
# coefficient grows, towards a supremum it never reaches. The optimisers stop
# wherever the rise has become too small for their tolerances (a WeibullPH
# coefficient of -16, a PO one of +33), report success, and the fit used to
# be returned silently. CoxPH detects its own case from the collapse of the
# information (``cox_ph._warn_if_monotone``); its root-finder runs on until
# the collapse is complete, whereas these optimisers stop part-way, at a
# point set by their tolerances, so a fixed collapse ratio cannot tell.
#
# Newton's method can. Along a coefficient's profile the log-likelihood of
# such data approaches its supremum like C - A exp(-s t) (or with a Gaussian
# tail, for a LogNormal AFT), and at every point on the way the Newton step
# is as long as the distance over which the curvature itself falls away:
# the next step is the same length again, and Newton's method never
# converges. Kantorovich's theorem makes that the test. For the negative
# log-likelihood f along the profile, the Newton step from the fit is
# -f'/f'', and Newton's method is guaranteed to converge to a minimum within
# twice that distance if h = |f'''| |f'| / f''^2 <= 1/2 (the relative change
# of the curvature over one step, with |f'''| its local bound). At a fit that
# has reached a maximum the step is at the level of the optimiser's
# tolerance, and so is h (at most 2e-5 on the ordinary fits of the
# conformance registry, and 2e-4 over the 1360 refits of their calibration
# study); on the way to a supremum h is 1 (exactly, for an exponential
# tail) wherever the optimiser stopped, and the curvature falls in the
# direction the likelihood rises (f' f''' > 0). A curvature that is zero or
# negative there is no maximum either: the additive hazards likelihood rises
# linearly as a no-event level's coefficient falls, without bound.
#
# Reading a profile costs third derivatives, several times an ordinary fit's
# own work, so a coefficient's is read only when Newton's method has not
# already shown the fit to be a maximum in it (``_cleared``, which costs one
# Hessian, needed for the profile anyway). The coefficient's part of the
# Newton step -H^{-1} g is at the level of the optimiser's tolerance at a
# maximum. On the way to a supremum it is 1/s, however far the optimiser
# went: write the gradient as H d plus the tail's s A e^{-st} along the flat
# direction u, d the optimiser's leftover displacement of the other
# parameters; along u, H d is the curvature s^2 A e^{-st} times d's small
# component, so the step along u is 1/s plus that component, and the
# leftover error cannot hide the runaway (which it does on the profile line
# until polished, see ``_profile``). The coefficient itself is then about t,
# and s t is the linear predictor the runaway drives, which exp keeps within
# log(largest float) = 709.8 of 0: beyond it the rows it moves underflow and
# the likelihood no longer depends on the coefficient at all. So a runaway's
# step is at least 1/709.8 of its size (measured: 1/100 to 1/5 on every
# runaway in the conformance registry), and a coefficient with a smaller
# step has converged. A larger step (at most 1/5900 of the coefficient on
# the registry's ordinary fits, and on a few of the calibration refits
# more), or a Hessian that is not positive definite, has the profile read,
# which only costs time.


def search_derivatives(
    neg_ll: Callable, x: npt.ArrayLike
) -> "tuple[npt.NDArray, npt.NDArray] | None":
    """``(H, g)``, the Hessian and gradient of ``neg_ll`` at ``x`` by
    autograd, from one trace; ``None`` for an objective autograd cannot
    differentiate. The no-maximum check reads them, and the fitted model
    keeps the Hessian for its covariance (:func:`keep_information`)."""
    at = np.asarray(x, dtype=float)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            hvp, g = make_hvp(neg_ll)(at)
            H = np.array([hvp(e) for e in np.eye(at.size)], dtype=float)
            return H, np.asarray(g, dtype=float)
    except (TypeError, ValueError, ArithmeticError):
        return None


def runaway_coefficients(
    neg_ll: Callable,
    x: npt.ArrayLike,
    coefs: "list[int]",
    start: "npt.ArrayLike | None" = None,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None" = None,
) -> "list[int]":
    """The positions in ``coefs`` of the parameters along which the
    likelihood has no finite maximum near ``x``.

    ``neg_ll`` is the negative log-likelihood the optimiser minimised,
    differentiable by autograd, ``x`` the point it returned (in its search
    space), ``coefs`` the positions in ``x`` of the parameters to check
    (the covariate coefficients) and ``start`` the point the search began
    from. Each is checked along its profile: the coefficient moves by 1 and
    the other parameters by the amounts that keep them at their best values
    for it (to first order), after the other parameters are first brought
    to their best values for the fitted coefficient (the optimiser stops
    them only as close as its tolerance, and on a flat profile that residue
    would swamp its derivatives). Where the profile cannot be formed -- the
    other parameters have no curvature either, in a fit that has run off in
    several directions at once -- the coefficient's own axis is used. It
    runs away when Newton's method cannot converge along that line, the
    Kantorovich test described above.

    There is no verdict (an empty list) for an objective autograd cannot
    differentiate, nor along a line on which the derivatives are not
    finite, or on which the likelihood does not change at all: at ``x``,
    or, with ``start``, at the start either (see
    :func:`_flat_at_start`), as it does not along a combination of
    collinear covariates, whose coefficients are not identified rather than
    infinite. ``derivatives`` are those of :func:`search_derivatives` at
    ``x``, if the caller has them.
    """
    at = np.asarray(x, dtype=float)
    if derivatives is None:
        derivatives = search_derivatives(neg_ll, at)
    if derivatives is None:
        return []
    H, g = derivatives
    cleared = _cleared(at, H, g)
    out = []
    for k, j in enumerate(coefs):
        if cleared[j]:
            continue
        axis = np.zeros(at.size)
        axis[j] = 1.0
        with np.errstate(all="ignore"):
            d = None
            if np.all(np.isfinite(H)):
                point, v = _profile(neg_ll, at, H, j)
                d = _line_derivatives(neg_ll, point, v)
            if d is None:
                point, v = at, axis
                d = _line_derivatives(neg_ll, point, v)
            if d is None or d[0] == 0.0:
                continue
            runaway = _no_convergence(neg_ll, point, v, d, j)
        if runaway:
            if start is None or not _flat_at_start(neg_ll, start, v):
                out.append(k)
    return out


#: The largest linear predictor exp can take, log(largest float): a
#: runaway's Newton step is at least this fraction of its coefficient's
#: size (see above).
_LOG_FLOAT_RANGE = float(np.log(np.finfo(float).max))


def _cleared(x: npt.NDArray, H: npt.NDArray, g: npt.NDArray) -> npt.NDArray:
    """Which parameters Newton's method shows to be at a maximum at ``x``,
    ``H`` and ``g`` the Hessian and gradient there: those whose part of the
    Newton step ``-H^{-1} g`` is no more than ``1 / log(largest float)`` of
    their size, which no parameter running off to a supremum can be (see
    above).

    Parameters the likelihood does not depend on at ``x`` to second order
    (a zero gradient and Hessian row, as a frailty variance held at its
    limit of 0 has) are left out of the step. None is cleared where the
    Hessian of the others is not finite and positive definite: a maximum
    has one, and a runaway's may not (a linear rise has no curvature)."""
    cleared = np.zeros(x.size, dtype=bool)
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
        return cleared
    used = np.flatnonzero(np.any(H != 0, axis=1) | (g != 0))
    H_u = H[np.ix_(used, used)]
    try:
        np.linalg.cholesky(H_u)
        step = np.linalg.solve(H_u, g[used])
    except np.linalg.LinAlgError:
        return cleared
    with np.errstate(all="ignore"):
        cleared[used] = np.abs(step) * _LOG_FLOAT_RANGE <= np.abs(x[used])
    return cleared


def _no_convergence(
    neg_ll: Callable,
    point: npt.NDArray,
    v: npt.NDArray,
    d: "tuple[float, float, float]",
    j: int,
) -> bool:
    """Whether Newton's method cannot be shown to converge along the
    profile of parameter ``j`` through ``point`` (direction ``v``, with
    ``v[j] = 1``), where the objective's first derivatives are ``d``: it
    has no curvature, or Kantorovich's ``h = |f'''| |f'| / f''^2`` is above
    1/2 with the curvature falling the way the likelihood rises (see
    above).

    ``f'''`` is the rate of change of the profile's curvature, the Schur
    complement of the Hessian in ``j``, from the curvature a quarter of a
    Newton step either side. Autograd's third derivative along the line
    was rounding noise at a stopped runaway, where the derivatives are
    near 1e-7: its sign changed with the build, so a WeibullAFT with a
    fixed coefficient warned on one Python and not on another. The
    curvature needs second derivatives only, which are well conditioned
    there. Where it cannot be formed, the line's own third derivative is
    used."""
    d1, d2, d3 = d
    if not d2 > 0.0:
        return True
    half = 0.125 * abs(d1 / d2)
    if half > 0.0:
        ahead = _profile_curvature(neg_ll, point + half * v, j)
        behind = _profile_curvature(neg_ll, point - half * v, j)
        if ahead is not None and behind is not None:
            d3 = (ahead - behind) / (2.0 * half)
    return d1 * d3 > 0.5 * d2**2


def _profile_curvature(
    neg_ll: Callable, point: npt.NDArray, j: int
) -> "float | None":
    """The curvature of the profile of parameter ``j`` at ``point``: the
    Schur complement ``H_jj - H_jo H_oo^+ H_oj`` of its Hessian, over the
    other parameters the likelihood depends on there (a frailty variance
    held at its limit has a zero row). ``None`` where the Hessian is not
    finite or cannot be taken."""
    derivatives = search_derivatives(neg_ll, point)
    if derivatives is None or not np.all(np.isfinite(derivatives[0])):
        return None
    H = derivatives[0]
    others = [i for i in range(H.shape[0]) if i != j and np.any(H[i] != 0.0)]
    if not others:
        return float(H[j, j])
    H_oo = H[np.ix_(others, others)]
    H_oj = H[others, j]
    return float(H[j, j] - H_oj @ np.linalg.pinv(H_oo) @ H_oj)


def _flat_at_start(
    neg_ll: Callable, start: npt.ArrayLike, v: npt.NDArray
) -> bool:
    """Whether ``neg_ll`` has neither slope nor curvature along ``v`` at
    ``start``, to rounding: its derivatives along ``v`` within ``size *
    eps`` of the size of its gradient and Hessian there (the tolerance of
    ``numpy.linalg.matrix_rank``), so that ``v`` is a direction the
    likelihood does not depend on at all. A likelihood running off to a
    supremum does depend on it, most of all near the start; one whose
    covariates are collinear (each level of a factor coded, with no
    intercept) does not, anywhere, and is no concern of this check (CoxPH
    warns of it as collinear)."""
    x0 = np.asarray(start, dtype=float)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            g0 = np.asarray(grad(neg_ll)(x0), dtype=float)
            H0 = np.asarray(hessian(neg_ll)(x0), dtype=float)
    except (TypeError, ValueError, ArithmeticError):
        return False
    if not (np.all(np.isfinite(g0)) and np.all(np.isfinite(H0))):
        return False
    tol = x0.size * float(np.finfo(float).eps)
    size = float(np.dot(v, v))
    slope = abs(float(np.dot(g0, v)))
    curvature = abs(float(v @ H0 @ v))
    return bool(
        slope <= tol * np.linalg.norm(g0) * np.sqrt(size)
        and curvature <= tol * np.linalg.norm(H0, 2) * size
    )


def _profile(
    neg_ll: Callable, x: npt.NDArray, H: npt.NDArray, j: int
) -> "tuple[npt.NDArray, npt.NDArray]":
    """``(point, direction)``: parameter ``j``'s profile line (see
    :func:`runaway_coefficients`), with ``H`` the Hessian at ``x``. The
    direction is not finite where it cannot be found."""
    rest = [i for i in range(x.size) if i != j]
    v = np.zeros(x.size)
    v[j] = 1.0
    point = x.copy()
    if not rest:
        return point, v
    try:
        # The pseudo-inverse: a parameter with no curvature to rounding (a
        # frailty variance at its boundary of 0) is not moved.
        inv_rr = np.linalg.pinv(H[np.ix_(rest, rest)])
        v[rest] = -inv_rr @ H[rest, j]
        # Newton steps on the other parameters, with the coefficient held
        # and their Hessian held at its value at x; a step that does not
        # lower the objective ends it (it has converged to rounding, or the
        # Hessian is no guide there).
        gradient = grad(neg_ll)
        f0 = float(neg_ll(point))
        for _ in range(_POLISH_STEPS):
            step = inv_rr @ gradient(point)[rest]
            trial = point.copy()
            trial[rest] = trial[rest] - step
            f1 = float(neg_ll(trial))
            if not (np.isfinite(f1) and f1 < f0):
                break
            point, f0 = trial, f1
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        v[rest] = np.nan
    return point, v


def _line_derivatives(
    neg_ll: Callable, point: npt.NDArray, v: npt.NDArray
) -> "tuple[float, float, float] | None":
    """The first three derivatives of ``neg_ll`` at ``point`` along ``v``,
    or ``None`` where they are not finite (or cannot be taken)."""
    if not np.all(np.isfinite(v)):
        return None
    moving = v != 0

    def line(t: Any) -> Any:
        return neg_ll(point + t * v)

    def moving_only(t: Any) -> Any:
        # The parameters that do not move held as constants, so that a
        # derivative that is not finite in one of them (a Gamma baseline's
        # shape near 0) cannot reach the line's.
        return neg_ll(
            np.array(
                [p + t * u if m else p for p, u, m in zip(point, v, moving)]
            )
        )

    out = None
    for along in (line,) if moving.all() else (line, moving_only):
        try:
            with warnings.catch_warnings():
                # autograd says so of a derivative that is constant (a
                # likelihood linear along the line); it is 0, not a fault
                warnings.filterwarnings("ignore", "Output seems independent")
                d1 = grad(along)(0.0)
                d2, d3 = value_and_grad(grad(grad(along)))(0.0)
            out = float(d1), float(d2), float(d3)
        except (TypeError, ValueError, ArithmeticError):
            return None
        if np.all(np.isfinite(out)):
            return out
    return None


#: The most Newton steps taken on the other parameters before a profile is
#: read (see ``_profile``); they start within the optimiser's
#: tolerance of their optimum, and two or three reach rounding.
_POLISH_STEPS = 10

#: What the warning says (through :func:`warn_no_maximum`), with the
#: coefficients' numbers.
NO_MAXIMUM_WHAT = (
    "the likelihood keeps increasing as coefficient(s) {} grow without "
    "bound, so the estimate is infinite (a covariate separates the events "
    "from the survivors, as one level with no events does)"
)
NO_MAXIMUM_CONSEQUENCE = (
    "The fit stopped where the increase became too small to follow, and "
    "the reported value, its standard error and its bounds are meaningless"
)
NO_MAXIMUM_ADVICE = (
    "consider removing or coarsening the covariate, or a penalised fit"
)


def free_coefficients(
    fitter: Any, fixed: dict, pmap: dict
) -> "list[tuple[int, int]]":
    """``(position, number)`` of each covariate coefficient that is not
    ``fixed``: its position in the search vector of
    :func:`prepare_regression_fit` (the free parameters, distribution
    parameters first, in ``bounds_convert``'s order) and its number in the
    model's ``phi_params``."""
    k_dist = len(fitter.param_map)
    names = [*fitter.param_map, *sorted(pmap, key=pmap.__getitem__)]
    free = [i for i, name in enumerate(names) if name not in fixed]
    return [(pos, i - k_dist) for pos, i in enumerate(free) if i >= k_dist]


def finish_search(
    fun: Callable,
    res: Any,
    coefs: "list[tuple[int, int]]",
    start: "npt.ArrayLike | None" = None,
) -> "tuple[bool, tuple[npt.NDArray, npt.NDArray] | None]":
    """Warn, once, of anything wrong with the optimiser's answer ``res``
    for the objective ``fun`` from ``start``: a likelihood with no finite
    maximum in a coefficient (``coefs`` as :func:`free_coefficients` gives
    them; see :func:`runaway_coefficients`), or else a search that stopped
    short (as :func:`optimise_ph` and :func:`optimise_nm_tnc` flag it with
    ``quiet=True``). Returns whether the likelihood had no maximum, and the
    Hessian and gradient of ``fun`` at ``res.x`` (``None`` where autograd
    cannot take them), for :func:`keep_information`."""
    derivatives = search_derivatives(fun, res.x)
    positions = [pos for pos, _ in coefs]
    runaway = runaway_coefficients(fun, res.x, positions, start, derivatives)
    if runaway:
        warn_no_maximum(
            NO_MAXIMUM_WHAT.format([coefs[k][1] for k in runaway]),
            NO_MAXIMUM_CONSEQUENCE,
            NO_MAXIMUM_ADVICE,
        )
        return True, derivatives
    if getattr(res, "stopped_short", False):
        warn_if_not_converged(res)
    return False, derivatives


def natural_information(
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    to_natural: Callable,
    t: npt.ArrayLike,
) -> "npt.NDArray | None":
    """The Hessian of the negative log-likelihood in the natural
    parameters, from its Hessian ``H_t`` and gradient ``g_t`` in the
    search space (``derivatives``, at ``t``), where ``to_natural`` maps the
    search vector to the natural free parameters one coordinate at a time,
    ``p = phi(t)``. By the chain rule ``g_t = phi' g_p`` and ``H_t = D H_p D
    + diag(g_p phi'')``, ``D = diag(phi')``, which is inverted exactly (to
    rounding).

    ``None`` where it cannot serve as an observed information, so that the
    covariance is computed as before, from a numerical Hessian: ``H_t`` or
    ``H_p`` not finite or not positive definite -- a parameter the
    likelihood does not depend on (a frailty variance at its limit of 0,
    where ``phi'`` is 0 too), or a flat direction."""
    if derivatives is None:
        return None
    H_t, g_t = derivatives
    t = np.asarray(t, dtype=float)
    if not (np.all(np.isfinite(H_t)) and np.all(np.isfinite(g_t))):
        return None
    try:
        # The search-space Hessian first: where it is not positive
        # definite the covariance stays as it was.
        np.linalg.cholesky(H_t)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            # phi' and, from the same trace, phi'' (each coordinate's map
            # depends on that coordinate alone).
            vjp, d1 = make_vjp(elementwise_grad(to_natural))(t)
            d2 = np.asarray(vjp(np.ones_like(t)), dtype=float)
            d1 = np.asarray(d1, dtype=float)
            g_p = g_t / d1
            H_p = (H_t - np.diag(g_p * d2)) / np.outer(d1, d1)
        if not np.all(np.isfinite(H_p)):
            return None
        H_p = 0.5 * (H_p + H_p.T)
        np.linalg.cholesky(H_p)
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        return None
    return H_p


def keep_information(
    model: Any,
    no_maximum: bool,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    inv_trans: Callable,
    const: Callable,
    t: npt.ArrayLike,
    centring: "Centring | None",
) -> None:
    """Give ``model`` the exact observed information of its fit: the
    natural-space Hessian of the free parameters (:func:`natural_information`)
    at the parameters and covariate centre its covariance is computed at
    (``ParametricRegressionModel._inference_state``), which
    ``_observed_covariance`` then uses instead of a numerical Hessian.
    ``derivatives`` are those of the search objective at ``t``, whose
    natural parameters are ``inv_trans(const(t))``, and ``centring`` that
    of the fit. Not for a fit whose likelihood has no maximum
    (``no_maximum``), whose covariance stays as it was, nor where those
    parameters are not exactly the ones the model computes its covariance
    at."""
    if no_maximum or derivatives is None:
        return
    names = model.parameter_names()
    held = model._held()
    free = np.array([i for i, name in enumerate(names) if name not in held])
    if model._fit_centring is not None:
        p_hat = model._fit_centring[0]
    else:
        p_hat = model._eval_params()
    with np.errstate(all="ignore"):
        at = np.asarray(inv_trans(const(t)), dtype=float)
    if not np.array_equal(at, np.asarray(p_hat, dtype=float)):
        return
    H_p = natural_information(
        derivatives, lambda u: inv_trans(const(u))[free], t
    )
    if H_p is None:
        return
    center = None if centring is None else centring.center
    model._information = (model._covariance_point(p_hat, center), H_p)


def optimise_nm_tnc(
    fun: Callable, init_t: npt.NDArray, quiet: bool = False
) -> Any:
    """AFT/PO's historical ladder: Nelder-Mead, then TNC kept only on
    success -- and, when that ladder has not reached a stationary point,
    the gradient-based :func:`optimise_ph` ladder from where it stopped.

    Nelder-Mead runs out of iterations on the harder fits (four
    covariates on 34 tires, say), and TNC can then fail as well, or report
    success where the gradient is plainly not zero; either way the fit
    used to come back tenths of a nat short of the maximum, silently. A
    converged fit is returned exactly as before. ``quiet`` as for
    :func:`optimise_ph`.
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
        best.stopped_short = not best.success
        if not quiet:
            warn_if_not_converged(best)
        return best
    if best.success and _is_stationary(g, best.fun):
        return best
    # (optimise_ph warns if it cannot converge, unless quiet.)
    polished = optimise_ph(fun, best.x, quiet)
    if np.isfinite(polished.fun) and polished.fun <= best.fun:
        return polished
    best.stopped_short = getattr(polished, "stopped_short", False)
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
