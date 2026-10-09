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
from scipy.optimize import OptimizeResult, minimize

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
from ._baseline_profile import filled_derivatives, to_log, walk_profile
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

        Far in the upper tail, where :math:`H` is so large that
        :math:`e^{-H}` underflows to 0, the density is 0, also where the
        hazard itself has overflowed to ``inf`` (a large Weibull shape, or
        a covariate far from 0): it is not ``inf * 0`` (#714).
        """
        h = self.hf(x, Z, *params)
        sf = np.exp(-self.Hf(x, Z, *params))
        # The families here have h(x) e^{-H(x)} -> 0 as H(x) -> inf (a
        # Weibull's is beta / x * H e^{-H}): an overflowed hazard next to
        # an underflowed survival is a density of 0.
        gone = np.isposinf(h) & (sf == 0)
        if not np.any(gone):
            return h * sf
        with np.errstate(invalid="ignore"):
            return np.where(gone, 0.0, h * sf)

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
#: The parametric refusal's hint where the search found no finite maximum
#: (#634), with what ran off (:func:`_runaway_clause`).
_NO_MAXIMUM_CENTER_HINT = (
    "The data may have no finite maximum: the search found the likelihood "
    "increasing as {}, which takes the baseline at Z = 0 out of range. Fit "
    "with center=True to have the model at the covariate means, with a "
    "warning saying so ('No finite maximum'), or {}."
)


def _runaway_clause(
    runaway: "list[int] | None", baseline: "tuple[str, ...]", dist: str
) -> "tuple[str, str]":
    """``(what, advice)`` for :data:`_NO_MAXIMUM_CENTER_HINT`: the
    coefficients (their numbers) and the baseline's parameters (their
    names, #714) the search found running off."""
    parts, advice = [], []
    if runaway:
        parts.append(
            "coefficient(s) {} grow without bound (a covariate that "
            "separates the events from the survivors)".format(runaway)
        )
        advice.append("remove or coarsen the covariate")
    if baseline:
        parts.append(
            "the {} baseline's {} run{} on, towards a limit of the family "
            "that none of its members reaches".format(
                dist or "distribution",
                ", ".join(baseline),
                "s" if len(baseline) == 1 else "",
            )
        )
        advice.append("compare the fits with other baselines")
    return ", and as ".join(parts), " or ".join(advice)


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
        moved: "tuple[int, ...]" = (),
    ):
        self.center = np.asarray(center, dtype=float)
        self.k_dist = k_dist
        self._move = move
        #: The baseline's parameters (their indices) that ``move`` moves.
        self.moved = moved

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
        return cls(mean, fitter.k_dist, move, moved)

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
        runaway: "list[int] | None" = None,
        baseline: "tuple[str, ...]" = (),
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
        the covariance over. ``runaway`` names the coefficients the search
        found running off, and ``baseline`` the baseline's parameters,
        where it found any: the refusal then says the data may have no
        finite maximum (#634, #714), not to move the covariates.
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
                    (
                        _NO_MAXIMUM_CENTER_HINT.format(
                            *_runaway_clause(runaway, baseline, dist_name)
                        )
                        if runaway or baseline
                        else _CENTER_HINT
                    ),
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
    order = canonical_order(data, Z)
    if np.any(order != np.arange(order.size)):
        data, Z = data[order], Z[order]
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


def canonical_order(data: SurpyvalData, Z: npt.ArrayLike) -> npt.NDArray:
    """The rows of ``data`` (with covariates ``Z``) sorted by every column:
    time, censoring, count, truncation and covariates, in that order.

    A fit runs on its rows in this order, so that it is the same, to the
    last digit, whatever order they are given in (#728). In the order
    given, the sums of the likelihood rounded differently, the search
    stopped elsewhere, and on data with no finite maximum the verdict
    could follow: a level with only censored rows gave "No finite
    maximum" in one order and "unverified" in another, and a refusal of a
    baseline at Z = 0 in a third. Rows equal in every column contribute
    the same terms, so their order among themselves does not matter."""
    rows = len(data)
    x = np.asarray(data.x, dtype=float).reshape(rows, -1)
    t = np.asarray(data.t, dtype=float).reshape(rows, -1)
    keys = [
        *np.asarray(Z, dtype=float).reshape(rows, -1).T[::-1],
        *t.T[::-1],
        np.asarray(data.n, dtype=float),
        np.asarray(data.c, dtype=float),
        *x.T[::-1],
    ]
    return np.lexsort(keys)


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
    runaway: "list[int] | None" = None,
    baseline: "tuple[str, ...]" = (),
) -> ParametricRegressionModel:
    """Common tail of every parametric-regression ``fit``.

    With a ``centring``, ``data`` and ``params`` are those of the centred
    fit: the model keeps the data as given, and its parameters and
    ``center`` are placed by :meth:`Centring.finish`, which checks them
    against the likelihood of the data as given, ``raw_neg_ll(*params)``
    (by default ``fitter.neg_ll`` of ``centring.raw``); ``runaway`` and
    ``baseline``, the coefficients and the baseline's parameters the
    search found running off, for its refusal.
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
            runaway,
            baseline,
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
#: What the warning says where the baseline distribution's own parameters
#: run off (#634), with the family and the parameters (and their values).
NO_MAXIMUM_BASELINE_WHAT = (
    "the {} baseline's {} run{} on, towards a limit of the family that "
    "none of its members reaches"
)
NO_MAXIMUM_BASELINE_ADVICE = (
    "a baseline distribution that contains the limit may describe the "
    "data: compare the fits with other baselines (a WeibullPO whose alpha "
    "runs off tends to LogLogisticPO, say)"
)


def _no_maximum_message(
    verdict: "SearchVerdict", dist: str = "", values: "dict | None" = None
) -> "tuple[str, str]":
    """``(what, advice)`` of the "No finite maximum" warning for
    ``verdict``: its coefficients, its baseline's parameters (with their
    ``values`` by name, where given), or both."""
    what = NO_MAXIMUM_WHAT.format(verdict.runaway)
    if not verdict.baseline:
        return what, NO_MAXIMUM_ADVICE
    values = values or {}
    named = ", ".join(
        f"{name} ({float(values[name]):.4g})" if name in values else name
        for name in verdict.baseline
    )
    one = len(verdict.baseline) == 1
    clause = NO_MAXIMUM_BASELINE_WHAT.format(
        dist or "distribution", named, "s" if one else ""
    )
    if not verdict.runaway:
        return (
            "the likelihood keeps increasing as " + clause,
            NO_MAXIMUM_BASELINE_ADVICE,
        )
    return (
        f"{what}, and as {clause}",
        f"{NO_MAXIMUM_ADVICE}; {NO_MAXIMUM_BASELINE_ADVICE}",
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


def free_baseline(fitter: Any, fixed: dict) -> "list[tuple[int, str]]":
    """``(position, name)`` of each of the baseline distribution's
    parameters that is not ``fixed``, its position in the search vector
    (they lead it), for the no-maximum check (:func:`judge_search`,
    #634)."""
    names = [name for name in fitter.param_map if name not in fixed]
    return list(enumerate(names))


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
    #: The names of the baseline distribution's parameters with no finite
    #: maximum (#634).
    baseline: "tuple[str, ...]" = ()

    @property
    def no_maximum(self) -> bool:
        return self.maximum == "no finite maximum"


#: The most log-likelihood a verified maximum may be short of the maximum
#: of its quadratic model (``newton_gain``): at least this many nats, and
#: this much per observation (#634).
GAIN_TOL = 1e-3
GAIN_TOL_PER_OBS = 1e-7


def newton_gain(
    x: npt.ArrayLike,
    derivatives: "tuple[npt.NDArray, npt.NDArray]",
    keep: "list[int]",
    one_sided: "tuple[int, ...]" = (),
) -> float:
    """How much the log-likelihood rises from ``x`` to the maximum of its
    quadratic model, ``g' H^-1 g / 2`` (the Newton decrement), on the
    components ``keep``, each parameter with one bound (``one_sided``) on
    the log scale of its distance from it, where its run is a straight
    line (as :func:`runaways_in_units` judges it); ``inf`` where the
    Hessian is not positive definite.

    A likelihood very flat in one direction passes the gradient test far
    from its maximum: a WeibullPO was "verified" 0.006 below its maximum
    with alpha 14 times short of it (#634). The gain is in nats whatever
    the units of the parameters, and at an ordinary maximum it is below
    1e-11."""
    step = _newton_step(x, derivatives, keep, one_sided)
    if step is None:
        return float("inf")
    return step[1]


def _newton_step(
    x: npt.ArrayLike,
    derivatives: "tuple[npt.NDArray, npt.NDArray]",
    keep: "list[int]",
    one_sided: "tuple[int, ...]" = (),
) -> "tuple[npt.NDArray, float] | None":
    """The Newton step from ``x`` and its gain (:func:`newton_gain`), as
    the point it reaches: the components ``keep`` moved, each parameter
    with one bound (``one_sided``) on the log scale of its distance from
    it, the others linearly. ``None`` where the Hessian is not positive
    definite."""
    H, g = derivatives
    at = np.asarray(x, dtype=float)
    log = np.zeros(at.size, dtype=bool)
    log[list(one_sided)] = True
    linear = log & (at >= 0.0)
    # The chain rule to u = log1p(x) where the map is linear (x >= 0)
    d1 = np.where(linear, at + 1.0, 1.0)
    d2 = np.where(linear, at + 1.0, 0.0)
    with np.errstate(all="ignore"):
        H_u = np.outer(d1, d1) * H + np.diag(g * d2)
        g_u = d1 * g
        H_u, g_u = H_u[np.ix_(keep, keep)], g_u[keep]
        if not (np.all(np.isfinite(H_u)) and np.all(np.isfinite(g_u))):
            return None
        try:
            L = np.linalg.cholesky(0.5 * (H_u + H_u.T))
        except np.linalg.LinAlgError:
            return None
        w = np.linalg.solve(L, g_u)
        du = -np.linalg.solve(L.T, w)
        # From u back to the search's own scale: x = expm1(u) on the
        # linear side of a one-sided map, u itself on its log side
        u = np.where(linear, np.log1p(np.maximum(at, 0.0)), at)[keep] + du
        moved = np.array(at, dtype=float)
        moved[keep] = np.where(
            log[keep], np.where(u >= 0.0, np.expm1(u), u), u
        )
    return moved, float(0.5 * np.dot(w, w))


def is_verified(
    x: npt.ArrayLike,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    n_obs: float,
    held: "tuple[int, ...]" = (),
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
) -> bool:
    """Whether ``x`` is a verified minimum of the objective whose Hessian
    and gradient there are ``derivatives`` (:func:`search_derivatives`):
    the test of ``is_local_minimum``, per observation (``n_obs``) and in
    units of ``max(|x|, floor)`` (:func:`coefficient_floor`), on the
    components of ``x`` other than ``held`` -- a parameter at a boundary of
    its space, whose own condition the caller has checked -- and not
    differentiating again; and the log-likelihood within ``GAIN_TOL``
    nats (or ``GAIN_TOL_PER_OBS`` per observation, if more) of the
    maximum of its quadratic model (:func:`newton_gain`, with the
    parameters with one bound, ``one_sided``, on their log scale), which
    a likelihood very flat in one direction can be far from while its
    gradient passes (#634)."""
    if derivatives is None:
        return False
    H, g = derivatives
    at = np.asarray(x, dtype=float)
    keep = [i for i in range(at.size) if i not in held]
    sub = np.ix_(keep, keep)
    floors = np.broadcast_to(np.asarray(floor, dtype=float), at.shape)
    if not is_local_minimum(
        lambda _: 0.0,  # (only the derivatives are read)
        lambda _: g[keep],
        lambda _: H[sub],
        at[keep],
        floor=floors[keep],
        obj_scale=n_obs,
    ):
        return False
    tol = max(GAIN_TOL, GAIN_TOL_PER_OBS * n_obs)
    return newton_gain(at, derivatives, keep, one_sided) <= tol


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
    baseline: "list[tuple[int, str]] | tuple" = (),
) -> SearchVerdict:
    """What the optimiser's answer ``res`` for the objective ``fun``, from
    ``start``, is (principles 12 and 13), without a word: a likelihood with
    no finite maximum in a coefficient (``coefs`` as
    :func:`free_coefficients` gives them; see :func:`runaway_coefficients`)
    or in a parameter of the baseline distribution (``baseline``, their
    ``(position, name)`` pairs, :func:`free_baseline`: a WeibullPO's alpha
    running to infinity, towards the log-logistic proportional odds model,
    #634), else a verified maximum or not.

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
    log scale (:func:`runaways_in_units`). An answer still not verified
    has the profiles of the baseline's parameters with one bound walked
    (:func:`_walk_baseline`, #710). An answer verified here is taken the
    rest of the way to the maximum, to 1e-9 nats (1e-11 an observation),
    by Newton's method (:func:`_newton_finish`, #746)."""
    if not (np.isfinite(res.fun) and np.all(np.isfinite(res.x))):
        # No answer to judge (``require_finite_fit`` refuses it)
        return SearchVerdict(res, "unverified", None, [])
    positions = [pos for pos, _ in coefs] + [pos for pos, _ in baseline]

    def running_off(res: Any, derivatives: Any) -> "SearchVerdict | None":
        runaway = runaways_in_units(
            fun, res.x, positions, start, derivatives, floor, one_sided
        )
        if not runaway:
            return None
        numbers = [coefs[k][1] for k in runaway if k < len(coefs)]
        names = tuple(
            baseline[k - len(coefs)][1] for k in runaway if k >= len(coefs)
        )
        return SearchVerdict(
            res, "no finite maximum", derivatives, numbers, names
        )

    derivatives = search_derivatives(fun, res.x)
    off = running_off(res, derivatives)
    if off is not None:
        return off
    finish = verified is None
    if verified is None:
        if derivatives is None:
            stopped = getattr(res, "stopped_short", False)
            state = "unverified" if stopped else "unknown"
            return SearchVerdict(res, state, None, [])
        verified = is_verified(
            res.x, derivatives, n_obs, held, floor, one_sided
        )
        if not verified:
            res, derivatives, verified = _polish(
                fun, res, derivatives, n_obs, held, floor, one_sided
            )
            if not verified and baseline:
                # The polish follows a baseline parameter's run-off
                # further than the search did (a WeibullPO's alpha from
                # 2e9 on), where its profile shows it (#634)
                off = running_off(res, derivatives)
                if off is not None:
                    return off
    if finish and verified and derivatives is not None:
        # The rest of the way to the maximum, where a flat direction
        # left the answer short of it (#746)
        res, derivatives = _newton_finish(
            fun, res, derivatives, n_obs, held, floor, one_sided
        )
    if not verified and derivatives is not None:
        # A baseline shape or scale on its way to a limit of the family,
        # or a fit stopped short of a maximum where the Hessian cannot be
        # had: its profile, walked (#710)
        walked = _walk_baseline(
            fun,
            res,
            derivatives,
            coefs,
            n_obs,
            held,
            floor,
            one_sided,
            baseline,
            finish,
        )
        if walked is not None:
            return walked
    if verified and derivatives is not None:
        far = _far_run_off(
            fun, res, derivatives, coefs, floor, one_sided, baseline
        )
        if far is not None:
            return far
    state = "verified" if verified else "unverified"
    return SearchVerdict(res, state, derivatives, [])


#: A verified answer with a baseline parameter of one bound further than
#: this many e-folds from it has that parameter's profile walked (#728).
FAR_EFOLDS = 50.0


def _far_run_off(
    fun: Callable,
    res: Any,
    at_res: "tuple[npt.NDArray, npt.NDArray]",
    coefs: "list[tuple[int, int]]",
    floor: "float | npt.ArrayLike",
    one_sided: "tuple[int, ...]",
    baseline: "list[tuple[int, str]] | tuple",
) -> "SearchVerdict | None":
    """The "no finite maximum" of :func:`judge_search` for an answer
    ``res`` that its derivatives (``at_res``) verify, where a baseline
    parameter with one bound, more than ``FAR_EFOLDS`` from it, runs to
    it on its profile (``walk_profile``); else ``None``.

    So far out, the derivatives can be rounding: a LogNormalPH whose sigma
    runs to 0 with a coefficient (the hazard rising from 0 at a
    threshold) rises by 1e-9 nats over the 7 e-folds from 1e-80 to
    1e-83, where log_ndtr's derivatives are rounding, and a search that
    stopped at 1e-80 passed the test of a maximum there, though one that
    stopped at 1e-83 did not (its Hessian overflowed) and walked the
    profile to the limit. A maximum that far out (a WeibullPO's alpha at
    1e-141) is found on its walk and stays as it was."""
    for pos, _ in baseline:
        if pos not in one_sided:
            continue
        if abs(to_log(float(res.x[pos]))) <= FAR_EFOLDS:
            continue
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            found = walk_profile(fun, res.x, pos, one_sided, floor)
        if found is not None and found.kind == "run-off":
            return SearchVerdict(
                res,
                "no finite maximum",
                at_res,
                [number for at, number in coefs if at in found.running],
                tuple(name for at, name in baseline if at in found.running),
            )
    return None


def _walk_baseline(
    fun: Callable,
    res: Any,
    at_res: "tuple[npt.NDArray, npt.NDArray]",
    coefs: "list[tuple[int, int]]",
    n_obs: float,
    held: "tuple[int, ...]",
    floor: "float | npt.ArrayLike",
    one_sided: "tuple[int, ...]",
    baseline: "list[tuple[int, str]] | tuple",
    finish: bool = True,
) -> "SearchVerdict | None":
    """The verdict of :func:`judge_search` on an answer ``res`` it has
    not verified (the derivatives there ``at_res``), from the profile of
    each of the baseline's parameters with one bound (``walk_profile``,
    #710): "no finite maximum" where the likelihood rises along one to a
    limit of the family, naming it and the parameters that run off with
    it; "verified" where the profile has a maximum and the search,
    finished from there (as :func:`_polish` does), reaches a verified
    maximum -- unless ``finish`` is false, for a caller whose model is
    built from ``res`` already (the accelerated life fit). ``None`` where
    the profiles say neither (the answer stays unverified)."""
    for pos, _ in baseline:
        if pos not in one_sided:
            continue
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            found = walk_profile(fun, res.x, pos, one_sided, floor)
        if found is None:
            continue
        if found.kind == "run-off":
            return SearchVerdict(
                res,
                "no finite maximum",
                at_res,
                [number for at, number in coefs if at in found.running],
                tuple(name for at, name in baseline if at in found.running),
            )
        if not finish:
            continue
        # The profile's maximum, or the answer itself where that is no
        # better (a maximum whose Hessian autograd cannot give)
        best = found.res if found.res.fun < res.fun else res
        start = OptimizeResult(
            x=np.array(best.x, dtype=float),
            fun=float(best.fun),
            success=True,
            nit=0,
        )
        polished, derivatives, verified = _polish(
            fun,
            start,
            search_derivatives(fun, start.x),
            n_obs,
            held,
            floor,
            one_sided,
        )
        if not verified:
            # Its Hessian by differences where autograd's overflows (the
            # model's covariance is then computed as before)
            verified = is_verified(
                polished.x,
                filled_derivatives(fun, polished.x, floor),
                n_obs,
                held,
                floor,
                one_sided,
            )
        if verified:
            if derivatives is not None:
                polished, derivatives = _newton_finish(
                    fun, polished, derivatives, n_obs, held, floor, one_sided
                )
            return SearchVerdict(polished, "verified", derivatives, [])
    return None


def _polish(
    fun: Callable,
    res: Any,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    n_obs: float,
    held: "tuple[int, ...]",
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
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
        is_verified(res.x, derivatives, n_obs, held, floor, one_sided),
    )


#: A verified answer is taken by Newton's method until the log-likelihood
#: is within this many nats of the maximum of its quadratic model
#: (``newton_gain``), or this many per observation if more, or within the
#: rounding of the log-likelihood (:func:`_newton_finish`, #746). (The
#: searches stop an ordinary fit of 2000 rows 1e-8 nats short, of 20,000
#: rows 1e-7: a Newton step more for each was 15-25% of the fit's time,
#: for a change a thousandth of a standard error.)
FINE_GAIN = 1e-9
FINE_GAIN_PER_OBS = 1e-11
#: The most Newton steps :func:`_newton_finish` takes.
FINISH_STEPS = 10


def _newton_finish(
    fun: Callable,
    res: Any,
    derivatives: "tuple[npt.NDArray, npt.NDArray]",
    n_obs: float,
    held: "tuple[int, ...]" = (),
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
) -> "tuple[Any, tuple[npt.NDArray, npt.NDArray]]":
    """``(res, derivatives)``: the verified answer ``res``, taken the rest
    of the way to its maximum by Newton's method (each parameter with one
    bound on the log scale of its distance from it, as ``newton_gain``
    measures the gain), until the gain left is within ``FINE_GAIN`` nats
    (``FINE_GAIN_PER_OBS`` for each of ``n_obs`` observations, if more)
    or the rounding of the log-likelihood; each step is kept only where
    it lowers ``fun``, halved up to ten times, and the answer only where
    it is still verified.

    A verified answer may be ``GAIN_TOL`` nats (1e-3) short of the
    maximum, which is far where the likelihood is very flat in one
    direction: a WeibullPO with alpha near its limit (#583's draw 207,
    alpha 8.8e30) stopped 3e-5 nats short of its maximum (alpha 1.16e31),
    and where it stopped followed the row order (#746). An ordinary fit
    is usually that close already and is not touched."""
    keep = [i for i in range(np.size(res.x)) if i not in held]
    x = np.asarray(res.x, dtype=float)
    f = float(res.fun)
    at = derivatives
    steps = 0
    eps = float(np.finfo(float).eps)
    tol = max(FINE_GAIN, FINE_GAIN_PER_OBS * n_obs)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Output seems independent")
        for _ in range(FINISH_STEPS):
            step = _newton_step(x, at, keep, one_sided)
            if step is None:
                break
            point, gain = step
            if gain <= max(tol, 64.0 * eps * max(1.0, abs(f))):
                break
            moved = None
            for _ in range(11):
                try:
                    f_point = float(fun(point))
                except (TypeError, ValueError, ArithmeticError):
                    break
                if np.all(np.isfinite(point)) and f_point < f:
                    moved = point, f_point
                    break
                point = 0.5 * (x + point)
            if moved is None:
                break
            derivs = search_derivatives(fun, moved[0])
            if derivs is None:
                break
            (x, f), at = moved, derivs
            steps += 1
    if not steps or not is_verified(x, at, n_obs, held, floor, one_sided):
        return res, derivatives
    out = copy.copy(res)
    out.x, out.fun = x, f
    return out, at


def say_verdict(
    verdict: SearchVerdict,
    what: str = "The maximum-likelihood search",
    dist: str = "",
    values: "dict | None" = None,
) -> None:
    """The one warning for a :class:`SearchVerdict` that is not a verified
    maximum: "No finite maximum", naming the coefficients and the
    parameters of the ``dist`` baseline (with their ``values``, by name,
    where given), or that ``what`` (the search, as the subject of a
    sentence) did not reach a verified maximum, with what the optimiser
    reported if it stopped short of one (:func:`optimise_ph`)."""
    if verdict.no_maximum:
        message, advice = _no_maximum_message(verdict, dist, values)
        warn_no_maximum(message, NO_MAXIMUM_CONSEQUENCE, advice)
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
    baseline: "list[tuple[int, str]] | tuple" = (),
    dist: str = "",
    values: "dict | None" = None,
) -> SearchVerdict:
    """:func:`judge_search`, then its one warning (:func:`say_verdict`),
    for a fit whose model does not depend on the polish (or is built after
    it), with ``floor``, ``one_sided`` and ``baseline`` as there, and
    ``dist`` and ``values`` as for :func:`say_verdict`. Returns the
    verdict: its ``res``, its ``maximum`` for the model, and the Hessian
    and gradient of ``fun`` at ``res.x`` (``None`` where autograd cannot
    take them), for :func:`keep_information`."""
    verdict = judge_search(
        fun,
        res,
        coefs,
        start,
        n_obs,
        verified,
        held,
        floor,
        one_sided,
        baseline,
    )
    say_verdict(verdict, what, dist, values)
    return verdict


#: A baseline parameter moved to Z = 0 that moves, there, less than this
#: fraction of the run-off's largest component along it (in the units of
#: ``runaways_in_units``) stays where it is at Z = 0 (#760): about 1e-5
#: of it where every event is at Z = 0, and as much as the coefficients
#: where the baseline there runs on with them.
ORIGIN_STILL = 1e-2


def _units(
    x: npt.NDArray, floor: "float | npt.ArrayLike", one_sided: tuple
) -> "tuple[npt.NDArray, npt.NDArray, npt.NDArray]":
    """``(u, size, log)``: the search vector ``x`` in the units of
    ``runaways_in_units`` (``u / size``), each parameter with one bound
    (``one_sided``, the mask ``log``) as the log of its distance from
    it."""
    log = np.zeros(x.size, dtype=bool)
    log[list(one_sided)] = True
    u = np.where(log & (x >= 0.0), np.log1p(np.abs(x)), x)
    unit = np.where(
        log, 1.0, np.broadcast_to(np.asarray(floor, dtype=float), x.shape)
    )
    return u, np.maximum(np.abs(u), unit), log


def _still_at_origin(
    verdict: SearchVerdict,
    centring: "Centring",
    names: "list[tuple[int, int, str]]",
    free: "list[int]",
    maps: "tuple[Callable, Callable, Callable]",
    floor: "float | npt.ArrayLike",
    one_sided: "tuple[int, ...]",
) -> "set[str]":
    """The ``names`` (``(index, position, name)`` of baseline parameters
    the move to Z = 0 shifts) that stay where they are at Z = 0 along the
    run-off: the Newton step at the answer, taken in the units of
    ``runaways_in_units``, where its run-off part dominates, moves each
    of them at Z = 0 by less than ``ORIGIN_STILL`` of its largest
    component (see :func:`baseline_at_origin`)."""
    if verdict.derivatives is None:
        return set()
    transform, inv_trans, const = maps
    H, g = verdict.derivatives
    x = np.asarray(verdict.res.x, dtype=float)
    u, size, log = _units(x, floor, one_sided)
    linear = log & (x >= 0.0)
    d1 = size * np.where(linear, x + 1.0, 1.0)
    d2 = size**2 * np.where(linear, x + 1.0, 0.0)
    H_v = np.outer(d1, d1) * H + np.diag(g * d2)
    if not (np.all(np.isfinite(H_v)) and np.all(np.isfinite(g))):
        return set()
    step = -np.linalg.pinv(H_v, hermitian=True) @ (d1 * g)
    largest = float(np.max(np.abs(step)))
    if not (np.isfinite(largest) and largest > 0.0):
        return set()
    h = 1e-6 / largest

    def at_origin(v: npt.NDArray) -> npt.NDArray:
        w = size * v
        point = np.where(log & (w >= 0.0), np.expm1(w), w)
        moved = centring.to_origin(np.asarray(inv_trans(const(point))))
        return np.asarray(transform(moved), dtype=float)[free]

    ahead, behind = at_origin(u / size + h * step), at_origin(
        u / size - h * step
    )
    if not (np.all(np.isfinite(ahead)) and np.all(np.isfinite(behind))):
        return set()
    u_a, size_a, _ = _units(ahead, floor, one_sided)
    u_b, _, _ = _units(behind, floor, one_sided)
    return {
        name
        for _, pos, name in names
        if abs(u_a[pos] - u_b[pos]) / size_a[pos] < ORIGIN_STILL * 2e-6
    }


def baseline_at_origin(
    verdict: SearchVerdict,
    fitter: Any,
    centring: "Centring | None",
    params_c: npt.NDArray,
    start: npt.NDArray,
    coefs: "list[tuple[int, int]]",
    fixed: dict,
    pmap: dict,
    maps: "tuple[Callable, Callable, Callable]",
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
) -> SearchVerdict:
    """``verdict`` naming only the baseline parameters that run off at
    ``Z = 0``, for a fit with no finite maximum whose baseline is moved
    there from the covariate means (``centring``) (#760).

    The search runs on centred covariates, where a parameter that the
    move shifts (``centring.moved``, a location or scale) runs off with
    the coefficients whenever they do and the covariate means are not 0.
    With every event at Z = 0, a LogisticPO's mu at the means runs on with
    the coefficients, while its mu at Z = 0, which the model reports, is
    the finite 6.941. Where there are no events at Z = 0 the baseline
    there runs on too (a WeibullPH's alpha to 0 along a coefficient of
    1/T). Such a parameter is left unnamed where both say it stays where
    it is at Z = 0: the run-off's direction (:func:`_still_at_origin`),
    and the check of :func:`judge_search` made as the model has it, at
    Z = 0 (on the objective of the data as given, from the fit's
    ``start`` and its answer ``params_c`` moved there, in the search's
    units: ``maps`` are the ``(transform, inv_trans, const)`` of
    ``bounds_convert`` for the parameters ``fixed`` and coefficients
    ``pmap``), which must find the same coefficients running off."""
    if not (
        verdict.no_maximum
        and verdict.runaway
        and centring is not None
        and centring.maps_back
        and centring.raw is not None
    ):
        return verdict
    transform, inv_trans, const = maps
    free = free_parameters(fitter, fixed, pmap)
    baseline = free_baseline(fitter, fixed)
    names = [
        (i, pos, name)
        for pos, name in baseline
        for i in centring.moved
        if fitter.parameter_names[i] == name and name in verdict.baseline
    ]
    if not names:
        return verdict
    positions = [pos for pos, _ in coefs] + [pos for pos, _ in baseline]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Output seems independent")
        try:
            still = _still_at_origin(
                verdict, centring, names, free, maps, floor, one_sided
            )
            if not still:
                return verdict
            at = np.asarray(
                transform(centring.to_origin(params_c)), dtype=float
            )[free]
            begin = np.asarray(
                transform(centring.to_origin(inv_trans(const(start)))),
                dtype=float,
            )[free]
            found = runaways_in_units(
                make_objective(fitter, centring.raw, inv_trans, const),
                at,
                positions,
                begin,
                None,
                floor,
                one_sided,
            )
        except (
            TypeError,
            ValueError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ):
            return verdict
    numbers = [coefs[k][1] for k in found if k < len(coefs)]
    if not set(verdict.runaway).issubset(numbers):
        return verdict
    running = {baseline[k - len(coefs)][1] for k in found if k >= len(coefs)}
    return verdict._replace(
        baseline=tuple(
            name
            for name in verdict.baseline
            if name not in still or name in running
        )
    )


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
            baseline=free_baseline(fitter, fixed),
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
        runaway=verdict.runaway,
        baseline=verdict.baseline,
    )
    # The baseline's run-offs as the model has it, at Z = 0 (#760)
    verdict = baseline_at_origin(
        verdict,
        fitter,
        centring,
        params,
        init_t,
        coefs,
        fixed,
        pmap,
        (transform, inv_trans, const),
        floor,
        one_sided_positions(bounds, free_parameters(fitter, fixed, pmap)),
    )
    # After the model is built (which may refuse the data), one
    # warning for what the search found (#392).
    say_verdict(
        verdict,
        dist=getattr(fitter.dist, "name", ""),
        values=dict(zip(fitter.param_map, model.params)),
    )
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
