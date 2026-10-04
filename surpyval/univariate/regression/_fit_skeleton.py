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
import functools
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import autograd.numpy as np
import numpy.typing as npt
from autograd import elementwise_grad, jacobian
from autograd.differential_operators import make_vjp
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import (
    Gradient,
    bounds_convert,
    is_local_minimum,
    minimize_with_gradient,
    preconditioned_bfgs,
)
from surpyval.univariate.parametric.fitters.runaway import (  # noqa: F401
    LOG_MAX,
    runaway_coefficients,
    runaways_in_units,
    search_derivatives,
)
from surpyval.univariate.parametric.parametric_fitter import Boxable, Numeric
from surpyval.utils import (
    _caller_stacklevel,
    check_covariate_rows,
    finite_covariate_mask,
)
from surpyval.utils.covariates import (
    coefficient_floor,
    coefficient_names,
)
from surpyval.utils.fitter_repr import FitterRepr, baseline_name
from surpyval.utils.no_maximum import warn_no_maximum, warn_unverified
from surpyval.utils.removed_names import removed_parameter_note
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import SurpyvalData

from ._aliasing import (
    aliased_columns,
    constant_columns,
    fit_columns,
    warn_aliased,
)
from ._covariate_link import CovariateLink
from ._kinds import (
    ACCELERATED_FAILURE_TIME,
    PROPORTIONAL_HAZARD,
    PROPORTIONAL_ODDS,
)
from .parametric_regression_model import ParametricRegressionModel


class LogLinearPhi(CovariateLink):
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

    def __init__(self, name: str, phi_param_map: dict[str, int]) -> None:
        super().__init__(name, phi_param_map)

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
    def make_param_map(
        Z: npt.NDArray, taken: "Iterable[str]" = ()
    ) -> dict[str, int]:
        """One coefficient per column of ``Z``, named by its column where
        the fit has the names (:func:`._aliasing.fit_columns`), else
        ``coef_j``, unique among themselves and ``taken`` (the baseline's
        parameters; :func:`~surpyval.utils.covariates.coefficient_names`,
        #614)."""
        return coefficient_map(Z, taken)


def coefficient_map(
    Z: npt.NDArray, taken: "Iterable[str]" = ()
) -> dict[str, int]:
    """:meth:`LogLinearPhi.make_param_map`: one coefficient per column of
    ``Z``, by name, with its column's number (#614)."""
    names = coefficient_names(Z.shape[1], fit_columns(), taken)
    return {name: j for j, name in enumerate(names)}


def per_column_map(pmap: dict, p: int) -> bool:
    """Whether the covariate parameters ``pmap`` were named by a custom
    ``phi_param_map`` the way coefficients were named before v0.23,
    ``beta_j`` for column ``j`` of ``p``: a custom link that names them so
    is treated, as it was, as one coefficient per column."""
    return pmap == {"beta_{}".format(j): j for j in range(p)}


def split_log_linear(
    fitter: Any, x: Numeric, Z: Numeric, params: tuple
) -> "tuple[Numeric, tuple, Boxable]":
    """``(x, dist_params, phi)`` for a covariate-function evaluation of
    the AFT and PO fitters: ``x`` as a 1-D float array, the distribution
    parameters (the first ``fitter.k_dist`` of ``params``) and the
    multiplier ``fitter._phi(Z, *coefficients)`` for ``Z`` read as rows.
    The parameters stay autograd values, so the likelihood built from
    them can be differentiated."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    k = fitter.k_dist
    return x, params[:k], fitter._phi(Z, *params[k:])


def make_objective(
    fitter: Any, data: SurpyvalData, inv_trans: Callable, const: Callable
) -> Callable:
    """The optimiser objective every regression fitter used to build
    inline: the fitter's negative log-likelihood evaluated in the
    transformed (unconstrained, fixed-parameters-removed) search space.

    A ``functools.partial`` of a module-level function rather than a
    closure, so the accelerated life model, which keeps it as ``fun``,
    pickles (#573).
    """
    return functools.partial(_objective, fitter, data, inv_trans, const)


def _objective(
    fitter: Any,
    data: SurpyvalData,
    inv_trans: Callable,
    const: Callable,
    params: npt.NDArray,
) -> Boxable:
    """``make_objective``'s objective at ``params``."""
    return fitter.neg_ll(data, *inv_trans(const(params)))


class MirroredDistributionAttrs(FitterRepr):
    """Class-level declarations for the attributes
    :func:`mirror_distribution` sets, so a fitter that inherits this
    alongside its other mixins has them visible to the type checker; and
    the fitter's ``repr``, ``WeibullAFT: accelerated failure time fitter
    (Weibull baseline)`` (#614)."""

    dist: Any
    k_dist: int
    bounds: tuple
    support: tuple
    parameter_names: list
    param_map: dict
    #: The end of the public name of a family's fitter after its
    #: distribution's, for a fitter without a ``name`` (``WeibullAFT``).
    name_suffix: str = ""

    def _repr_name(self) -> str:
        name = getattr(self, "name", None)
        if isinstance(name, str) and name:
            return name
        return str(getattr(self.dist, "name", "")) + self.name_suffix

    def _repr_details(self) -> "list[str]":
        return baseline_name(self)


def mirror_distribution(fitter: Any, distribution: Any) -> None:
    """Copy a distribution's metadata onto a regression fitter.

    Every parametric regression fitter starts by mirroring the same six
    attributes of its underlying distribution -- ``dist``, ``k_dist``,
    ``bounds``, ``support``, ``parameter_names`` and the name-to-index
    ``param_map`` -- and each family's ``__init__`` carried the block
    verbatim. The ``*_dist`` method aliases stay with each family: which
    ones it needs depends on which identities it implements.
    """
    fitter.dist = distribution
    fitter.k_dist = len(distribution.parameter_names)
    fitter.bounds = distribution.bounds
    fitter.support = distribution.support
    fitter.parameter_names = distribution.parameter_names
    fitter.param_map = {
        v: i for i, v in enumerate(distribution.parameter_names)
    }


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
    (PROPORTIONAL_HAZARD, "Weibull"): ((0,), _ph_weibull),
    (PROPORTIONAL_HAZARD, "Exponential"): ((0,), _rate_down),
    (PROPORTIONAL_HAZARD, "Rayleigh"): (
        (0,),
        lambda p, s: [p[0] * np.exp(s / 2.0)],
    ),
    (PROPORTIONAL_HAZARD, "Gumbel"): (
        (0,),
        lambda p, s: [p[0] + p[1] * s, p[1]],
    ),
    (ACCELERATED_FAILURE_TIME, "Weibull"): ((0,), _first_scale_up),
    (ACCELERATED_FAILURE_TIME, "Exponential"): ((0,), _rate_down),
    (ACCELERATED_FAILURE_TIME, "Rayleigh"): ((0,), _first_scale_up),
    (ACCELERATED_FAILURE_TIME, "LogLogistic"): ((0,), _first_scale_up),
    (ACCELERATED_FAILURE_TIME, "ExpoWeibull"): ((0,), _first_scale_up),
    (ACCELERATED_FAILURE_TIME, "Gamma"): (
        (1,),
        lambda p, s: [p[0], p[1] * np.exp(-s)],
    ),
    (ACCELERATED_FAILURE_TIME, "LogNormal"): (
        (0,),
        lambda p, s: [p[0] + s, p[1]],
    ),
    (ACCELERATED_FAILURE_TIME, "Normal"): ((0, 1), _both_scale_up),
    (ACCELERATED_FAILURE_TIME, "Logistic"): ((0, 1), _both_scale_up),
    (ACCELERATED_FAILURE_TIME, "Gumbel"): ((0, 1), _both_scale_up),
    (ACCELERATED_FAILURE_TIME, "GumbelLEV"): ((0, 1), _both_scale_up),
    (PROPORTIONAL_ODDS, "LogLogistic"): (
        (0,),
        lambda p, s: [p[0] * np.exp(-s / p[1]), p[1]],
    ),
    (PROPORTIONAL_ODDS, "Logistic"): (
        (0,),
        lambda p, s: [p[0] - p[1] * s, p[1]],
    ),
}

#: The hint every refusal below ends with.
_CENTER_HINT = (
    "Fit with center=True to report the baseline at the covariate means "
    "(model.center) instead, or move the covariates nearer 0."
)


def baseline_at_origin_error(
    what: str,
    center: npt.ArrayLike,
    lp_center: float,
    log_ratio: float,
    why: str = "",
) -> ValueError:
    """The refusal of a semi-parametric baseline fitted at the covariate
    means that cannot be moved to ``Z = 0`` (#463): the ``what`` (``"baseline
    hazard"``, ...) at ``Z = 0`` is ``exp(log_ratio)`` times that at the
    ``center``, where the linear predictor is ``lp_center``, and over- or
    underflows. ``why`` adds a reason in brackets. Cox, the semi-parametric
    proportional odds model and Fine-Gray raise it; the parametric fits'
    :meth:`Centring.finish` has its own, as their baseline moves through
    its parameters."""
    return ValueError(
        "The {} at Z = 0 cannot be represented for these covariates: their "
        "means are {} and the linear predictor there is beta'center = "
        "{:.4g}, so the baseline at Z = 0 is exp({:.4g}) times that at the "
        "means, which over- or underflows{}. {}".format(
            what,
            np.array2string(np.asarray(center), precision=4),
            lp_center,
            log_ratio,
            why,
            _CENTER_HINT,
        )
    )


def covariate_center(Z: npt.ArrayLike, n: npt.ArrayLike) -> npt.NDArray:
    """The ``n``-weighted mean of the covariate rows, where a centred fit
    puts its baseline (#459, #463).

    Every regression fit centres on it where it can: ``exp(beta'Z)`` then
    stays near 1 for the rows of the data instead of overflowing on a
    column far from 0 (a year, a date as a day count). The Cox partial
    likelihood depends on the covariates only through their differences
    within a risk set, so its coefficients are unchanged; with
    ``center=True`` the model keeps its baseline at this point and
    predicts with ``exp(beta'(Z - center))``, as R's ``coxph``, lifelines
    and scikit-survival do (for start-stop data R's mean is over the
    interval rows, as here).
    """
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
        if any(fitter.parameter_names[i] in fixed for i in moved):
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


class FixedWithAliased(dict):
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
    per_column: bool = True,
) -> dict:
    """``fixed`` with the coefficients the data cannot determine held at
    0 and named, with one warning (#476; see :mod:`._aliasing`).

    Only where each coefficient multiplies one column of ``Z``
    (``per_column``: the ``j``-th of ``pmap`` for column ``j``). A
    constant column is aliased where the family has an intercept -- where
    adding a constant to the linear predictor moves the baseline
    parameters and nothing else (:data:`ORIGIN_MAPS`, a scale family, as
    R's ``survreg`` and ``lm`` treat an intercept) -- and otherwise only a
    column of zeros is. Columns whose coefficient the caller fixed are
    offsets, left out.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] == 0 or Z.shape[0] == 0:
        return fixed
    p = Z.shape[1]
    names = sorted(pmap, key=pmap.__getitem__)
    if not per_column or sorted(pmap.values()) != list(range(p)):
        return fixed
    free = np.array([j for j in range(p) if names[j] not in fixed])
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
    held = tuple(names[j] for j in aliased.tolist())
    out = FixedWithAliased({**fixed, **{name: 0.0 for name in held}})
    out.aliased = held
    return out


def check_baseline_support(fitter: Any, data: SurpyvalData) -> None:
    """Refuse times outside the support of the fitter's baseline
    distribution (#565), with the univariate fits' check and wording
    (``OutsideSupportError``, a ``ValueError``), and also a censored time
    below the support's lower end: a Weibull, Gamma or Exponential AFT
    took a negative censored time, and its likelihood's derivatives were
    nan there. A baseline on the whole line (Normal, Gumbel, Logistic)
    refuses nothing."""
    check = getattr(fitter.dist, "_check_inside_support", None)
    if check is not None:
        check(data, every_row=True)


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

    ``kind`` names a log-linear family (``PROPORTIONAL_HAZARD``, ...),
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
    # After the rows with a missing covariate are dropped (principle 3)
    check_baseline_support(fitter, data)

    fixed = {} if fixed is None else fixed
    Z_data = np.asarray(data.Z)
    # One coefficient per column, each named by its column or ``coef_j``
    # (#614), or a custom link's own parameters.
    if phi_param_map is LogLinearPhi.make_param_map:
        pmap = coefficient_map(Z_data, fitter.param_map)
        per_column = True
    else:
        pmap = (
            phi_param_map(Z_data) if callable(phi_param_map) else phi_param_map
        )
        per_column = per_column_map(pmap, Z_data.shape[1])
    fixed = alias_coefficients(
        fitter, kind, Z_data, data.n, fixed, pmap, per_column
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
    # The covariate coefficients sit after the distribution parameters in
    # the packed parameter vector, so their map indices must be offset by
    # the number of distribution parameters — otherwise
    # ``fixed={"coef_0": v}`` silently pins the first *distribution*
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
            "are {}{}.".format(
                unknown,
                names,
                "".join(removed_parameter_note(k, names) for k in unknown),
            )
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
                    stacklevel=_caller_stacklevel(),
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
    raw_neg_ll: "Callable | None" = None,
) -> ParametricRegressionModel:
    """Common tail of every parametric-regression ``fit``.

    With a ``centring``, ``data`` and ``params`` are those of the centred
    fit: the model keeps the data as given, and its parameters and
    ``center`` are placed by :meth:`Centring.finish`, which checks them
    against the likelihood of the data as given, ``raw_neg_ll(*params)``
    (by default ``fitter.neg_ll`` of ``centring.raw``).
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
            (
                (lambda *p: fitter.neg_ll(raw, *p))
                if raw_neg_ll is None
                else raw_neg_ll
            ),
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
    model.dist = fitter.dist
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
    fun: Callable,
    init_t: npt.NDArray,
    quiet: bool = False,
    floor: "float | npt.ArrayLike" = 1.0,
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
    warns through :func:`finish_search`). ``floor`` is BFGS's least unit
    per component (:func:`coefficient_floor`).
    """
    # The value and the gradient from one pass (#593)
    jac = Gradient(fun)

    best = None
    for method in ("BFGS", "TNC", "Nelder-Mead"):
        x0 = init_t if best is None else best.x
        if method == "BFGS":
            res = preconditioned_bfgs(
                fun, x0, jac=jac, options={"maxiter": 1000}, floor=floor
            )
        elif method == "TNC":
            res = minimize_with_gradient(
                fun, x0, (), jac, method="TNC", options={"maxfun": 1000}
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
    """Say so when no optimiser rung converged (``warn_unverified``).

    The best point found is still returned, but not silently: it used to
    come back with ``res.success`` False and nothing said, which is how a
    nan covariate passed off the starting values as a fit.
    """
    if not res.success:
        warn_unverified(
            "The maximum-likelihood search",
            "the optimiser reported: {}".format(str(res.message).rstrip(".")),
        )


# -- no finite maximum (#392) -------------------------------------------------
#
# The check itself, Newton's along each coefficient's profile, is
# ``runaway_coefficients`` in ``surpyval.univariate.parametric.fitters.
# runaway``, shared with the univariate maximum-likelihood fit (#584).


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
    free = free_parameters(fitter, fixed, pmap)
    return [(pos, i - k_dist) for pos, i in enumerate(free) if i >= k_dist]


def free_parameters(fitter: Any, fixed: dict, pmap: dict) -> "list[int]":
    """The index, among all the model's parameters (distribution parameters
    first, then the coefficients in ``pmap``'s order), of each that is not
    ``fixed``, in the order of the search vector."""
    names = [*fitter.param_map, *sorted(pmap, key=pmap.__getitem__)]
    return [i for i, name in enumerate(names) if name not in fixed]


def one_sided_positions(
    bounds: "tuple | list", free: "Iterable[int]"
) -> "tuple[int, ...]":
    """The positions in the search vector of the free parameters (``free``,
    their indices into ``bounds``, as ``bounds_convert``'s ``not_fixed``)
    that have exactly one bound, which ``bounds_convert`` searches as the
    log of their distance from it within a unit and linearly beyond: the
    no-maximum check judges them on the log scale throughout
    (``runaways_in_units``, #628)."""
    return tuple(
        pos
        for pos, i in enumerate(free)
        if (bounds[i][0] is None) != (bounds[i][1] is None)
    )


class SearchVerdict(NamedTuple):
    """What a regression fit's search reached (:func:`judge_search`)."""

    #: The optimiser's answer, polished where it was not verified.
    res: Any
    #: The model's ``maximum``: ``"verified"``, ``"unverified"``, ``"no
    #: finite maximum"`` or, for an objective autograd cannot
    #: differentiate whose optimiser reported success, ``"unknown"``.
    maximum: str
    #: The Hessian and gradient of the objective at ``res.x``
    #: (:func:`search_derivatives`), for :func:`keep_information`.
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None"
    #: The numbers of the coefficients with no finite maximum.
    runaway: "list[int]"

    @property
    def no_maximum(self) -> bool:
        return self.maximum == "no finite maximum"


def is_verified(
    x: npt.ArrayLike,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    n_obs: float,
    held: "tuple[int, ...]" = (),
    floor: "float | npt.ArrayLike" = 1.0,
) -> bool:
    """Whether ``x`` is a verified minimum of the objective whose Hessian
    and gradient there are ``derivatives`` (:func:`search_derivatives`):
    the test of ``is_local_minimum``, per observation (``n_obs``) and in
    units of ``max(|x|, floor)`` (:func:`coefficient_floor`), on the
    components of ``x`` other than ``held`` -- a parameter at a boundary of
    its space, whose own condition the caller has checked -- and not
    differentiating again."""
    if derivatives is None:
        return False
    H, g = derivatives
    at = np.asarray(x, dtype=float)
    keep = [i for i in range(at.size) if i not in held]
    sub = np.ix_(keep, keep)
    floors = np.broadcast_to(np.asarray(floor, dtype=float), at.shape)
    return is_local_minimum(
        lambda _: 0.0,  # (only the derivatives are read)
        lambda _: g[keep],
        lambda _: H[sub],
        at[keep],
        floor=floors[keep],
        obj_scale=n_obs,
    )


def judge_search(
    fun: Callable,
    res: Any,
    coefs: "list[tuple[int, int]]",
    start: "npt.ArrayLike | None" = None,
    n_obs: float = 1.0,
    verified: "bool | None" = None,
    held: "tuple[int, ...]" = (),
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
) -> SearchVerdict:
    """What the optimiser's answer ``res`` for the objective ``fun``, from
    ``start``, is (principles 12 and 13), without a word: a likelihood with
    no finite maximum in a coefficient (``coefs`` as
    :func:`free_coefficients` gives them; see :func:`runaway_coefficients`),
    else a verified maximum or not.

    ``verified`` is the caller's own verdict, where it has checked the
    answer itself (``verify_or_polish``); otherwise the answer is checked
    here, with the Hessian and gradient the no-maximum check and the
    covariance need anyway (:func:`is_verified`, ``held`` left out), and an
    answer that is not verified is polished with BFGS in the units maximum
    likelihood searches in (``preconditioned_bfgs``) and checked again, as
    ``verify_or_polish`` does. A fit that stopped short (as
    :func:`optimise_ph` and :func:`optimise_nm_tnc` flag it with
    ``quiet=True``) is usually rescued that way; an ordinary fit is already
    verified and is not touched. An objective autograd cannot differentiate
    keeps the optimiser's verdict: ``"unverified"`` if it stopped short,
    else ``"unknown"``. ``floor`` is each component's least unit for the
    check and the polish (:func:`coefficient_floor`), and ``one_sided``
    the positions of the parameters with one bound
    (:func:`one_sided_positions`), which the no-maximum check judges on the
    log scale (:func:`runaways_in_units`)."""
    if not (np.isfinite(res.fun) and np.all(np.isfinite(res.x))):
        # No answer to judge (``require_finite_fit`` refuses it)
        return SearchVerdict(res, "unverified", None, [])
    derivatives = search_derivatives(fun, res.x)
    positions = [pos for pos, _ in coefs]
    runaway = runaways_in_units(
        fun, res.x, positions, start, derivatives, floor, one_sided
    )
    if runaway:
        numbers = [coefs[k][1] for k in runaway]
        return SearchVerdict(res, "no finite maximum", derivatives, numbers)
    if verified is None:
        if derivatives is None:
            stopped = getattr(res, "stopped_short", False)
            state = "unverified" if stopped else "unknown"
            return SearchVerdict(res, state, None, [])
        verified = is_verified(res.x, derivatives, n_obs, held, floor)
        if not verified:
            res, derivatives, verified = _polish(
                fun, res, derivatives, n_obs, held, floor
            )
    state = "verified" if verified else "unverified"
    return SearchVerdict(res, state, derivatives, [])


def _polish(
    fun: Callable,
    res: Any,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    n_obs: float,
    held: "tuple[int, ...]",
    floor: "float | npt.ArrayLike" = 1.0,
) -> "tuple[Any, tuple[npt.NDArray, npt.NDArray] | None, bool]":
    """``(res, derivatives, verified)`` after a BFGS polish of ``res``,
    kept where it is no worse (see :func:`judge_search`)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Output seems independent")
        try:
            polish = preconditioned_bfgs(
                fun, res.x, (), Gradient(fun), floor=floor, obj_scale=n_obs
            )
        except (TypeError, ValueError, ArithmeticError):
            polish = None
    if (
        polish is not None
        and np.all(np.isfinite(polish.x))
        and np.isfinite(polish.fun)
        and polish.fun <= res.fun
    ):
        res = polish
        derivatives = search_derivatives(fun, res.x)
    return (
        res,
        derivatives,
        is_verified(res.x, derivatives, n_obs, held, floor),
    )


def say_verdict(
    verdict: SearchVerdict, what: str = "The maximum-likelihood search"
) -> None:
    """The one warning for a :class:`SearchVerdict` that is not a verified
    maximum: "No finite maximum", naming the coefficients, or that
    ``what`` (the search, as the subject of a sentence) did not reach a
    verified maximum, with what the optimiser reported if it stopped short
    of one (:func:`optimise_ph`)."""
    if verdict.no_maximum:
        warn_no_maximum(
            NO_MAXIMUM_WHAT.format(verdict.runaway),
            NO_MAXIMUM_CONSEQUENCE,
            NO_MAXIMUM_ADVICE,
        )
    elif verdict.maximum == "unverified":
        res = verdict.res
        reason = None
        if getattr(res, "stopped_short", False) and not res.success:
            reason = "the optimiser reported: {}".format(
                str(getattr(res, "message", "")).rstrip(".")
            )
        warn_unverified(what, reason)


def finish_search(
    fun: Callable,
    res: Any,
    coefs: "list[tuple[int, int]]",
    start: "npt.ArrayLike | None" = None,
    n_obs: float = 1.0,
    verified: "bool | None" = None,
    what: str = "The maximum-likelihood search",
    held: "tuple[int, ...]" = (),
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
) -> SearchVerdict:
    """:func:`judge_search`, then its one warning (:func:`say_verdict`),
    for a fit whose model does not depend on the polish (or is built after
    it), with ``floor`` and ``one_sided`` as there. Returns the verdict:
    its ``res``, its ``maximum`` for the model, and the Hessian and
    gradient of ``fun`` at ``res.x`` (``None`` where autograd cannot take
    them), for :func:`keep_information`."""
    verdict = judge_search(
        fun, res, coefs, start, n_obs, verified, held, floor, one_sided
    )
    say_verdict(verdict, what)
    return verdict


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
    names = model.parameter_names
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
    fun: Callable,
    init_t: npt.NDArray,
    quiet: bool = False,
    floor: "float | npt.ArrayLike" = 1.0,
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

    When autograd can differentiate the objective, the gradient ladder of
    :func:`optimise_ph` runs first, and its answer is kept when it is a
    verified optimum. Nelder-Mead was the bulk of an AFT or PO fit: 436
    derivative-free evaluations on a 5-covariate Weibull AFT, where the
    gradient ladder needs a few dozen and reaches the same maximum 4-6x
    sooner (#499). A result it cannot verify falls through to the ladder
    below, unchanged. ``floor`` is passed to :func:`optimise_ph`.
    """
    if _gradient(fun, init_t) is not None:
        fast = optimise_ph(fun, init_t, quiet=True, floor=floor)
        stopped_short = getattr(fast, "stopped_short", False)
        if np.isfinite(fast.fun) and not stopped_short:
            return fast
    res = minimize(
        fun, init_t, method="Nelder-Mead", options={"maxiter": 1000}
    )
    res2 = minimize(fun, res.x, method="TNC")
    best = res2 if res2.success else res
    g = _gradient(fun, best.x)
    if g is None:
        # An objective autograd cannot differentiate: the optimiser's
        # verdict is all there is.
        best.stopped_short = not best.success
        if not quiet:
            warn_if_not_converged(best)
        return best
    if best.success and _is_stationary(g, best.fun):
        return best
    # (optimise_ph warns if it cannot converge, unless quiet.)
    polished = optimise_ph(fun, best.x, quiet, floor)
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


def fit_log_linear(
    fitter: Any,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: "npt.ArrayLike | None",
    n: "npt.ArrayLike | None",
    t: "npt.ArrayLike | None",
    init: "npt.ArrayLike | None",
    fixed: "dict[str, float] | None",
    center: bool,
    *,
    kind: str,
    optimiser: Callable,
    reg_model: Callable[[dict], Any],
    phi_bounds: "Callable[[npt.NDArray], tuple] | tuple" = (
        LogLinearPhi.phi_bounds
    ),
    phi_param_map: "Callable[[npt.NDArray], dict] | dict" = (
        LogLinearPhi.make_param_map
    ),
    phi_init: "Callable[[npt.NDArray], npt.NDArray] | None" = None,
    log_linear: bool = True,
) -> ParametricRegressionModel:
    """The whole ``fit`` of the PH, AFT and PO families (#238, #302).

    Prepares the data and the search (:func:`prepare_regression_fit`),
    maximises the likelihood with ``optimiser`` (:func:`optimise_ph` or
    :func:`optimise_nm_tnc`, called with ``quiet=True``), builds the model
    of ``kind`` (:func:`assemble_regression_model`) with the covariate
    link ``reg_model(pmap)``, then warns of anything wrong with the search
    and keeps the exact information (#392). ``log_linear=False`` (a custom
    PH ``phi``) fits without the move of the baseline to ``Z = 0``, which
    only the log-linear link has (#463; :class:`Centring`). Each family
    keeps its own ``fit`` signature and docstring and calls this.
    """
    data, prep = prepare_regression_fit(
        fitter,
        x,
        Z,
        c,
        n,
        t,
        init,
        fixed,
        phi_bounds,
        phi_param_map,
        phi_init,
        kind=kind if log_linear else None,
        center=center,
    )
    (
        init_t,
        bounds,
        pmap,
        transform,
        inv_trans,
        const,
        fixed,
        centring,
    ) = prep

    coefs = free_coefficients(fitter, fixed, pmap)
    # Each coefficient searched and judged in its own covariate's units
    # (#577); a custom ``phi``'s parameters need not be one per column.
    floor = (
        coefficient_floor(len(init_t), coefs, data.Z) if log_linear else 1.0
    )
    with np.errstate(all="ignore"):

        fun = make_objective(fitter, data, inv_trans, const)

        res = optimiser(fun, init_t, quiet=True, floor=floor)

        # What the search reached (#392), its answer polished where it was
        # not a verified maximum; said once the model is built.
        verdict = judge_search(
            fun,
            res,
            coefs,
            init_t,
            float(np.sum(data.n)),
            floor=floor,
            one_sided=one_sided_positions(
                bounds, free_parameters(fitter, fixed, pmap)
            ),
        )
        res = verdict.res

    params = inv_trans(const(res.x))

    model = assemble_regression_model(
        fitter,
        kind,
        reg_model(pmap),
        data,
        res,
        params,
        bounds,
        pmap,
        fixed,
        centring=centring,
    )
    # After the model is built (which may refuse the data), one
    # warning for what the search found (#392).
    say_verdict(verdict)
    model.maximum = verdict.maximum
    # The exact information for the model's covariance (#392).
    keep_information(
        model,
        verdict.no_maximum,
        verdict.derivatives,
        inv_trans,
        const,
        res.x,
        centring,
    )
    return model
