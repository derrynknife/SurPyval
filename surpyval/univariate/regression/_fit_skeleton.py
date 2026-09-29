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
from surpyval.utils.rng import as_generator
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
# covariates as given (:class:`OriginWatch` refuses the fit where that
# breaks down), as are the additive hazards models, whose term beta'Z has
# no exp to overflow.


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


def far_from_zero(Z: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """Which covariates' means are more than ten of their standard
    deviations from 0 (a constant column if it is not 0): far enough that
    ``exp(beta'Z)`` swings by orders of magnitude over a coefficient
    change the data barely distinguish."""
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.size == 0:
        return np.zeros(np.shape(Z_arr)[-1], dtype=bool)
    mean = covariate_center(Z_arr, n)
    n_arr = np.asarray(n, dtype=float).reshape(-1)
    sd = np.sqrt(np.dot(n_arr, (Z_arr - mean) ** 2) / n_arr.sum())
    return np.abs(mean) > 10.0 * sd


class OriginWatch:
    """Refuses an uncentred log-linear fit that the covariates' distance
    from 0 has broken (#463).

    Where the baseline at ``Z = 0`` has no exact map from the covariate
    means (:data:`ORIGIN_MAPS`), the default fit is at ``Z = 0``, as it
    always was. On covariates far from 0 that can fail: ``exp(beta'Z)``
    over- or underflows on the data at the parameters the optimiser
    returns, the likelihood or its gradient is not finite there, or the
    optimiser does not reach a verified maximum. Each of those used to
    return a model (a few with a warning), a Gamma PO one with a
    log-likelihood of +3914. Or the optimiser stops at a local maximum
    where the coefficient of the far covariate has collapsed towards 0 -- the
    effect exp(beta'Z) can no longer express without overflowing -- as a
    LogNormal PH fit on a covariate 300 from 0 did (0.006 against 0.64),
    silently; :meth:`compare` catches that against the same model fitted
    with its baseline at the means. The fit now raises in each case, and
    points to ``center=True``. A fit the old code managed is unchanged.
    """

    def __init__(self, Z: npt.NDArray, n: npt.NDArray, k_dist: int):
        self.Z = np.asarray(Z, dtype=float)
        self.far_cols = far_from_zero(self.Z, n)
        self.far = bool(np.any(self.far_cols))
        self.center = covariate_center(self.Z, n)
        self.k_dist = k_dist

    def run(self, optimise: Callable, fun: Callable, init_t: Any) -> Any:
        """``optimise(fun, init_t)``, refused if the covariates' origin
        broke it; its warnings are passed on otherwise."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = optimise(fun, init_t)
        self._check(res, fun, caught)
        for w in caught:
            warnings.warn_explicit(
                w.message, w.category, w.filename, w.lineno, source=w.source
            )
        return res

    def check_params(self, params: npt.NDArray) -> None:
        """Refuse parameters at which ``exp(beta'Z)`` over- or underflows
        on the data (covariates far from 0)."""
        if not self.far:
            return
        beta = np.asarray(params, dtype=float)[self.k_dist :]
        with np.errstate(all="ignore"):
            lp = np.dot(self.Z, beta)
        limit = np.log(np.finfo(float).max)
        if not np.all(np.abs(lp) < limit):
            self._refuse(
                "exp(beta'Z) over- or underflows on the data at the fitted "
                "coefficients (beta'Z reaches {:.4g})".format(
                    float(np.max(np.abs(lp)))
                )
            )

    def compare(self, model: Any, fit_centred: Callable) -> None:
        """Refuse a fit on covariates far from 0 whose coefficient of such
        a covariate has collapsed: it differs by more than half from the
        coefficient of the same model fitted with its baseline at the
        covariate means (``fit_centred()``, a ``center=True`` fit), which
        the data determine to within a third of itself (three standard
        errors). The two models differ only in where their baseline is
        anchored, and both estimate the same log hazard (or odds) ratio;
        a coefficient that moves that far is the optimiser failing, not
        the model. Nothing is checked on covariates near 0, where the
        default fit is as it always was."""
        if not self.far:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                try:
                    ref = fit_centred()
                    se = np.asarray(ref.standard_errors(), dtype=float)
                except ValueError:
                    return
        k = self.k_dist
        beta = np.asarray(model.params, dtype=float)[k:]
        beta_c = np.asarray(ref.params, dtype=float)[k:]
        se = se[k:]
        with np.errstate(invalid="ignore"):
            collapsed = (
                self.far_cols
                & (np.abs(beta_c) > 3.0 * se)
                & (np.abs(beta - beta_c) > 0.5 * np.abs(beta_c))
            )
        if np.any(collapsed):
            j = int(np.flatnonzero(collapsed)[0])
            self._refuse(
                "the coefficient of covariate {} is {:.4g}, where the same "
                "model with its baseline at the covariate means has "
                "{:.4g} (standard error {:.2g}); exp(beta'Z) cannot carry "
                "that effect this far from 0, and the optimiser stopped "
                "at a local maximum without it".format(
                    j, beta[j], beta_c[j], se[j]
                )
            )

    def _check(self, res: Any, fun: Callable, caught: list) -> None:
        if not self.far:
            return
        if not np.isfinite(res.fun) or _gradient(fun, res.x) is None:
            self._refuse(
                "the log-likelihood or its gradient is not finite at the "
                "optimiser's answer"
            )
        if any(
            "did not converge" in str(w.message)
            or "verified maximum" in str(w.message)
            for w in caught
        ):
            self._refuse("the optimiser did not reach a verified maximum")

    def _refuse(self, what: str) -> None:
        raise ValueError(
            "The fit with the baseline at Z = 0 failed on these covariates, "
            "whose means are {}: {}. This model's baseline does not map "
            "exactly between Z = 0 and the covariate means, so the fit "
            "cannot be centred and reported at 0. {}".format(
                np.array2string(self.center, precision=4), what, _CENTER_HINT
            )
        )


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
    centring, watch)`` — everything the family's optimiser step and the
    final assembly need. ``phi_bounds``/``phi_param_map``/``phi_init`` may
    be callables of the covariate array or static values.

    ``kind`` names a log-linear family (``"Proportional Hazard"``, ...),
    whose fit runs on centred covariates where its baseline maps back to
    ``Z = 0`` exactly, and ``center=True`` centres any family and keeps the
    baseline at the covariate means (:class:`Centring`). ``data`` is then
    the centred copy the objective uses, an ``init`` given at ``Z = 0`` is
    moved to the centre, and ``centring`` (else ``None``) keeps the data as
    given for :func:`assemble_regression_model`. An uncentred log-linear
    fit gets an :class:`OriginWatch` (else ``None``) for its optimiser.
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
    centring = Centring.plan(fitter, kind, Z_data, data.n, fixed, center)
    watch = None
    if centring is not None:
        centring.raw = data
        data = centred_copy(data, centring.center)
    elif kind is not None:
        watch = OriginWatch(Z_data, data.n, fitter.k_dist)

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
        watch,
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
    watch: "OriginWatch | None" = None,
) -> ParametricRegressionModel:
    """Common tail of every parametric-regression ``fit``.

    With a ``centring``, ``data`` and ``params`` are those of the centred
    fit: the model keeps the data as given, and its parameters and
    ``center`` are placed by :meth:`Centring.finish`. A ``watch`` checks
    the parameters of an uncentred fit.
    """
    if watch is not None:
        watch.check_params(np.asarray(params, dtype=float))
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
