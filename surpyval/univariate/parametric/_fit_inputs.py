"""The checks a parametric fit makes of its inputs, and its starts.

``FitInputsMixin``, which
:class:`~surpyval.univariate.parametric.optimised_fit.OptimisedFitMixin`
inherits, holds what ``fit`` does before it searches: refusing data and
options it cannot fit (``_validate_fit_inputs``, with the identifiability
and point-mass checks), and the initial guesses the searches start from
(``_initial_guess`` and the alternative starts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt

import surpyval
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import check_option

from ..nonparametric import plotting_positions as pp
from .fitters import offset_step
from .parametric import Parametric

# The families whose likelihood grows without bound as they concentrate
# on one point (a location or scale with a free shape; see
# ``ParametricFitter._point_mass_region``), and those that do so only
# with an offset: a scale family whose offset runs up to the point.
_POINT_MASS_FAMILIES = frozenset(
    {
        "Weibull",
        "Gamma",
        "LogNormal",
        "LogLogistic",
        "ExpoWeibull",
        "Normal",
        "Gumbel",
        "GumbelLEV",
        "Logistic",
        "Beta",
        "Beta4",
        "Uniform",
    }
)


_OFFSET_POINT_MASS_FAMILIES = frozenset({"Exponential", "Rayleigh"})

# The families not refused by the point-mass check that move their mass
# later (or earlier) without limit through a parameter in which they are
# ordered by likelihood ratio: a scale (Exponential, Rayleigh, the
# DiscreteWeibull's q) or the Poisson, Geometric and NegativeBinomial
# success parameter. On data that bound no failure from above (or below)
# their likelihood has no finite maximum (``_warn_if_one_sided``).
_ONE_SIDED_FAMILIES = frozenset(
    {
        "Exponential",
        "Rayleigh",
        "Poisson",
        "Geometric",
        "NegativeBinomial",
        "DiscreteWeibull",
    }
)


def _offset_start(x: npt.ArrayLike) -> float:
    """Starting offset: just below the smallest value, by a step on the
    data's own scale.

    The step is the mean spacing of the sorted finite values (see
    ``offset_step``), which scales with the data. A step of one *unit*
    (``min(x) - 1``) would make the start depend on the units the data
    were recorded in: at a scale of 1e-3 it sits a thousand spreads below
    the data, where the likelihood is flat in the offset and the search
    never moves it, and at 1e5 a hair below the smallest value.

    Every offset initialiser seeds its other parameters from the data
    shifted by this same value, since the fitter installs it as the
    starting offset: shape and scale seeds taken against a different
    shift describe a different distribution from the one the search
    starts at.
    """
    finite = np.asarray(x, dtype=float).ravel()
    return float(np.min(finite[np.isfinite(finite)])) - offset_step(x)


def _imputed_data(
    x: npt.NDArray, c: npt.NDArray, n: npt.NDArray
) -> SurpyvalData:
    """Wrap ``_initial_guess``'s working copy as a ``SurpyvalData``.

    ``group_and_sort=False`` because this is not user input. The rows
    have already been validated once, and merging duplicates or
    reordering them would change what the initialisers see for no gain.

    The truncation bounds are deliberately left at their defaults rather
    than carried over from the data being seeded. The imputation moves
    interval- and left-censored points to a midpoint, which can put an
    observation at or before its own left-truncation time -- a
    contradiction ``xcnt_handler`` rejects outright (#260). Seeding is
    not inference, so the untruncated copy is the right one: it is what
    every initialiser has always been given, since no caller ever passed
    ``t`` down.
    """
    return SurpyvalData(x=x, c=c, n=n, group_and_sort=False)


def _rows_as_read(
    surv_data: SurpyvalData,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Each row's lower and upper bounds and censoring code as the
    likelihood reads it.

    A right censored row with a finite right truncation ``tr`` is the
    interval ``[x, tr]``, and a left censored row with a finite left
    truncation ``tl`` the interval ``[tl, x]`` (#310; the masks of
    ``SurpyvalData._split_to_observation_types``), so such rows are
    returned with code 2 and those bounds. Every other row is returned as
    given, an exact or censored value ``x`` as ``(x, x)``. The checks that
    decide whether the data leave a fit read the data this way, as the
    likelihood does (#559).
    """
    x = np.asarray(surv_data.x, dtype=float)
    lo, hi = (x[:, 0], x[:, 1]) if x.ndim == 2 else (x, x)
    c = np.asarray(surv_data.c)
    recast_r = surv_data.mask_i & (c == 1)
    recast_l = surv_data.mask_i & (c == -1)
    lo = np.where(recast_l, np.asarray(surv_data.tl, dtype=float), lo)
    hi = np.where(recast_r, np.asarray(surv_data.tr, dtype=float), hi)
    return lo, hi, np.where(recast_l | recast_r, 2, c)


def _row_sets(
    surv_data: SurpyvalData,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The lower and upper ends of each row's set of failure times, and
    its censoring code as the likelihood reads it (``_rows_as_read``).

    An exact value is ``{x}``, a right censored one ``(x, inf)``, a left
    censored one ``(-inf, x]`` and an interval ``(xl, xr]``; a set that
    reaches an edge of its truncation window extends beyond it (see
    ``FitInputsMixin._point_mass_region``).
    """
    xl, xr, c = _rows_as_read(surv_data)
    tl = np.asarray(surv_data.tl, dtype=float)
    tr = np.asarray(surv_data.tr, dtype=float)
    lower = np.where(c == -1, -np.inf, xl)
    upper = np.where(c == 1, np.inf, np.where(c == 0, xl, xr))
    # Sets that reach an edge of their window extend beyond it
    lower = np.where((c != 0) & (lower == tl), -np.inf, lower)
    upper = np.where(upper == tr, np.inf, upper)
    return lower, upper, c


PARA_METHODS = ["MPP", "MLE", "MPS", "MSE", "MOM"]


def normalise_how(how: Any) -> Any:
    """``how`` in upper case when it names an estimation method: it is
    typed by hand, and ``how="mle"`` raised (#485). Anything else is
    returned as it is, for the caller's own check to refuse."""
    if isinstance(how, str) and how.upper() in PARA_METHODS:
        return how.upper()
    return how


class OutsideSupportError(ValueError):
    """Data outside a distribution's support, so the distribution cannot
    describe it at all. A ``ValueError``; its own class lets
    ``fit_best`` pass over such a candidate quietly."""


def _check_mps_data(surv_data: SurpyvalData) -> None:
    """The data maximum product of spacings (MPS) can take."""
    if (surv_data.c == 2).any():
        # neg_mean_D has no interval-censored term; without this
        # guard 2-D input dies deep in np.hstack with a cryptic
        # dimensions error (#268).
        raise ValueError(
            "MPS does not support interval-censored observations; "
            "use MLE (or MPP with the Turnbull heuristic) for "
            "interval data."
        )

    if (surv_data.tl[0] != surv_data.tl).any():
        raise ValueError(
            "Left truncated value can only be single number when using MPS"
        )

    if (surv_data.tr[0] != surv_data.tr).any():
        raise ValueError(
            "Right truncated value can only be single number when using " "MPS"
        )


class FitInputsMixin:
    """The input checks and initial guesses of
    :class:`~surpyval.univariate.parametric.optimised_fit.OptimisedFitMixin`,
    which inherits this mixin."""

    if TYPE_CHECKING:
        # Supplied by ParametricFitter, which every user of this mixin
        # also inherits. Declared rather than defined so the methods
        # below type check without the mixin pretending to own them.
        name: str
        k: int
        parameter_names: list[str]
        support: tuple[int | float, int | float]
        discrete: bool
        supports_mpp: bool

        def _parameter_initialiser(
            self, data: SurpyvalData, offset: bool = False
        ) -> npt.NDArray: ...

    def _check_identifiable(
        self,
        surv_data: SurpyvalData,
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float] | None,
    ) -> Any:
        """
        Reject data that cannot pin down the free parameters.

        A right censored observation says only "later than this", so it
        constrains a fitted curve without locating a point on it. What
        locates a point is an exact observation, a left censored one, or
        an interval. Fewer *distinct* such values than there are free
        parameters and the likelihood has a flat direction: for a
        Weibull on a tied sample it is unbounded, since a spike of
        arbitrary height can sit on the repeated value, and the reported
        answer is wherever the optimiser happened to stop: three tied
        observations at 10 would return ``beta = 512`` with
        ``success=True`` and no warning.

        The count is of *free* parameters, not of the distribution's
        parameters, so fixing one buys back a degree of freedom: a
        Weibull fit to a single observation with ``beta`` fixed is well
        posed and recovers ``alpha = (sum x^beta / n) ** (1 / beta)``.
        That is why this cannot be a per-distribution constant.
        """
        n_free = (
            self.k
            + int(bool(offset))
            + int(bool(lfp))
            + int(bool(zi))
            - len(fixed or {})
        )
        if n_free <= 0:
            return

        # A right censored row with a finite ``tr`` is an interval, and
        # locates a failure as one (#559)
        lo, hi, c = _rows_as_read(surv_data)
        informative = c != 1
        if not informative.any():
            return
        rows = np.column_stack([lo, hi])[informative]
        distinct = np.unique(rows, axis=0).shape[0]

        if distinct < n_free:
            raise ValueError(
                f"{self.name} has {n_free} free parameter(s) but the data "
                f"contains only {distinct} distinct non-right-censored "
                f"value(s). The likelihood has a flat (or unbounded) "
                f"direction, so no unique fit exists. Provide more "
                f"distinct observations, fix a parameter with "
                f"`fixed=`, or choose a distribution with fewer "
                f"parameters."
            )

    def _check_has_maximum(
        self,
        surv_data: SurpyvalData,
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float] | None,
    ) -> None:
        """Refuse data whose likelihood has no maximum because one failure
        time explains every row (see ``_point_mass_region``)."""
        if fixed or zi:
            return
        region = self._point_mass_region(surv_data, offset, lfp)
        if region is not None:
            raise ValueError(
                f"The {self.name} likelihood has no maximum on this data: "
                f"a single failure time {region} is consistent with every "
                f"observation, so a distribution ever more concentrated "
                f"there explains the data ever better, and the fit would "
                f"run off to a degenerate spike. Provide observations "
                f"that disagree about when the failures happened, or fix "
                f"a parameter with `fixed=`."
            )

    def _point_mass_region(
        self, surv_data: SurpyvalData, offset: bool, lfp: bool
    ) -> str | None:
        """Where a single failure time consistent with every row of the
        data lies, if this family can concentrate its mass there; else
        ``None``.

        Such a time means the likelihood has no maximum (#392): a failure
        at 0.5 and one known only to be before 1 are both explained
        perfectly by a spike at 0.5, so a Weibull's likelihood grows
        without bound with its shape, and the fit would return wherever it
        stopped (``beta = 395.7``, a Normal ``sigma`` of 5e-324) in
        silence. Tied exact values are the special case the distinct-value
        count already refuses; this is the general one, as the Turnbull
        existence check (#327) is for the non-parametric fit.

        Each row's times form a set: an exact value ``{x}``, a right
        censored one ``(x, inf)``, a left censored one ``(-inf, x]`` and an
        interval ``(xl, xr]``. The time exists when their intersection
        meets the support. With a limited failure population the right
        censored rows are explained by the units that never fail, so they
        are left out. Only the families that approach a point mass
        anywhere in their support are checked (``_POINT_MASS_FAMILIES``,
        and any scale family given an offset); a one-parameter family such
        as the Exponential has a maximum on such data.

        A truncated row is conditioned on its window ``(tl, tr]``, and a
        spike outside the window keeps its likelihood where the row's set
        reaches the window's edge: the conditional distribution then piles
        up at that edge. So a set ending at ``tr`` (an exact value at its
        own ``tr``, whose ``f(x) / F(tr)`` grows without bound, or a left
        censored or interval one up to ``tr``) extends to every time
        above ``tr``, and a set starting at ``tl`` (an interval from
        ``tl``) to every time below it. An exact failure at 1 observable
        only up to 1, one at 2 and one known to be before 3 would give a
        Weibull ``beta`` of 455.6 (a Normal ``sigma`` of 0.037), in
        silence.
        """
        if self.discrete or self.name not in _POINT_MASS_FAMILIES | (
            _OFFSET_POINT_MASS_FAMILIES if offset else frozenset()
        ):
            return None
        lower, upper, c = _row_sets(surv_data)
        if lfp:
            kept = c != 1
            lower, upper, c = lower[kept], upper[kept], c[kept]
        if c.size == 0 or np.all(c == 1):
            return None
        if self.name == "Uniform" and not np.any(c == 0):
            # Its own fit refuses these, saying it needs an exact value
            return None
        lo, hi = float(np.max(lower)), float(np.min(upper))
        # Only an exact value's lower end is closed
        lo_closed = bool(np.all(c[lower == lo] == 0))
        support = np.asarray(self.support, dtype=float)
        if offset or np.isnan(support).any():
            support = np.array([-np.inf, np.inf])
        if lo == hi:
            inside = lo_closed and support[0] < lo < support[1]
            return f"({lo:g})" if inside else None
        lo, hi = max(lo, support[0]), min(hi, support[1])
        if not lo < hi:
            return None
        if not np.isfinite(hi):
            return f"(any time after {lo:g})"
        if not np.isfinite(lo):
            return f"(any time up to {hi:g})"
        return f"(any time in ({lo:g}, {hi:g}])"

    def _warn_if_one_sided(
        self, surv_data: SurpyvalData, results: dict, zi: bool, lfp: bool
    ) -> bool:
        """Warn, and return ``True``, when no row of the data bounds a
        failure from above (or none from below), where the likelihood of
        the families in ``_ONE_SIDED_FAMILIES`` has no finite maximum.

        Each row's set of failure times (``_row_sets``) then reaches up to
        infinity: every row is right censored, or an interval reaching the
        top of its truncation window, which is how the likelihood reads a
        right censored row with a finite ``tr`` (#559). Each row's
        likelihood is the probability of its set given its window, and a
        family ordered by likelihood ratio in a parameter raises every
        such probability as that parameter moves its mass later: the
        likelihood keeps rising toward the end of that parameter's range.
        Suspensions at 1, ..., 5 truncated at 2, 6, 4, 8 and 9 ran an
        Exponential to ``failure_rate = 3.6e-7`` and a Rayleigh to
        ``sigma = 313.5``, reported as verified maxima. Data whose every
        row is left censored, or an interval from the bottom of its window
        or of the support, are the mirror image. The other families
        refuse such data in ``_check_has_maximum``.

        The criterion is on the data alone, and an observation bounded on
        both sides (an exact value, an interval inside its window) breaks
        it, so it cannot fire on data with a finite maximum. Only a free
        fit is checked: with ``zi`` or ``lfp`` the extra parameter changes
        the argument, the caller does not check a fit with ``fixed``
        parameters, and an offset fit of these families on such data is
        refused by ``_check_has_maximum`` before it starts.
        """
        if zi or lfp or self.name not in _ONE_SIDED_FAMILIES:
            return False
        lower, upper, _ = _row_sets(surv_data)
        if np.all(np.isposinf(upper)):
            side, kind, edge, way = "above", "right", "top", "later"
        elif np.all(lower <= self.support[0]):
            side, kind, edge, way = "below", "left", "bottom", "earlier"
        else:
            return False
        params = np.atleast_1d(np.asarray(results.get("params", [])))
        reached = ", ".join(
            f"{name} = {value:.4g}"
            for name, value in zip(self.parameter_names, params)
        )
        warn_no_maximum(
            f"no observation bounds a failure from {side} (every row is "
            f"{kind} censored, or an interval reaching the {edge} of its "
            f"truncation window), so the {self.name} likelihood keeps "
            f"rising as the distribution moves its mass {way} than every "
            f"observation",
            f"The reported {reached}, the standard errors and the bounds "
            f"are where the search stopped and are meaningless",
            "a fit needs a failure observed exactly, or known to lie in an "
            "interval inside its truncation window",
        )
        return True

    def _validate_fit_inputs(
        self,
        surv_data: SurpyvalData,
        how: str,
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float] | None,
        heuristic: str,
        turnbull_estimator: str,
    ) -> Any:
        self._check_offset_and_grid(surv_data, offset)
        self._check_method(surv_data, how, offset, lfp, zi, fixed)
        self._check_censoring_for_method(surv_data, how, heuristic)

        if (
            (heuristic == "Turnbull")
            and (not ((-1 in surv_data.c) or (2 in surv_data.c)))
            and ((~np.isfinite(surv_data.tr)).all())
        ):
            # The Turnbull method is extremely memory intensive.
            # So if no left or interval censoring and no right-truncation
            # then this is equivalent.
            heuristic = turnbull_estimator

        if (not offset) and (not zi):
            self._check_inside_support(surv_data)

        if how == "MPS":
            _check_mps_data(surv_data)

        return heuristic

    def _check_offset_and_grid(
        self, surv_data: SurpyvalData, offset: bool
    ) -> None:
        """Whether the family can be offset, and a discrete one's grid."""
        # Offsetting (a free location/threshold ``gamma``) only makes sense
        # for distributions supported on a half-line ``[0, inf)``. A
        # distribution with a finite upper bound (e.g. Beta on ``[0, 1]``)
        # or a data-dependent support cannot be offset: shifting the lower
        # bound while pinning the upper one is not a member of the family.
        # Use the 4-parameter Beta instead if you need a shifted/scaled
        # Beta on an arbitrary ``[a, b]`` interval.
        offsettable = (self.support[0] == 0) and np.isinf(self.support[1])
        if offset and not offsettable:
            detail = f"{self.name} distribution cannot be offset"
            raise ValueError(detail)

        # A discrete distribution's mass sits on the integers, and its
        # likelihood reads the data as integer counts: shifting it by a
        # continuous ``gamma`` is not a member of the family. Its support
        # of ``[0, inf)`` passes the check above, and the fit would then
        # die deep in the optimiser with an unrelated zip() error.
        if offset and self.discrete:
            raise ValueError(
                f"{self.name} is a discrete distribution and cannot be "
                "offset; subtract a known integer shift from the data "
                "instead."
            )

        # A discrete distribution's mass sits on the integers, and between
        # them its sf is interpolated by some formulas (Geometric,
        # NegativeBinomial, ...) and floored by others (Poisson), so a
        # non-integer observation has no consistent meaning (Geometric
        # would fit [1.5, 2.2, 3.7, 1.1] and return p = 0.47).
        if self.discrete:
            values = np.concatenate(
                [np.ravel(surv_data.x), np.ravel(surv_data.t)]
            ).astype(float)
            values = values[np.isfinite(values)]
            off_grid = values[values != np.round(values)]
            if off_grid.size:
                raise ValueError(
                    f"{self.name} is a discrete distribution, so its data "
                    "(and any truncation bounds) must be whole numbers; "
                    f"got {off_grid[0]:g}. Round or bin the data first, or "
                    "use a continuous distribution."
                )

    def _check_method(
        self,
        surv_data: SurpyvalData,
        how: str,
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float] | None,
    ) -> None:
        """Whether ``how`` (and the model options) can fit this data."""
        # Probability plotting is exempt. It is a regression through the
        # plotting positions, not a likelihood maximisation, so it has no
        # unbounded direction to fall into and always returns finite
        # parameters. It is also how several distributions seed
        # themselves, and that internal call does not carry the caller's
        # ``fixed``, so checking it would reject well posed fits.
        if how != "MPP":
            self._check_identifiable(surv_data, offset, lfp, zi, fixed)
        if how == "MLE":
            self._check_has_maximum(surv_data, offset, lfp, zi, fixed)

        if fixed and how == "MPP":
            detail = (
                "Probability plotting (MPP) does not support"
                " fixing parameters"
            )
            raise ValueError(detail)

        check_option("how", how, PARA_METHODS, "Case does not matter.")

        if how == "MPP" and not self.supports_mpp:
            detail = (
                f"{self.name} distribution does not work"
                " with probability plot fitting; use how='MLE', 'MSE' or"
                " 'MOM' instead"
            )
            raise ValueError(detail)

        if how == "MPS" and self.discrete:
            detail = (
                f"{self.name} is a discrete distribution; maximum product"
                " of spacings (MPS) is defined by increments of a"
                " continuous CDF, and repeated integer observations make"
                " the spacings degenerate. Use how='MLE' instead."
            )
            raise ValueError(detail)

        if np.isfinite(surv_data.t).any() and how == "MSE":
            detail = "Mean square error doesn't yet support truncation"
            raise NotImplementedError(detail)

        if np.isfinite(surv_data.t).any() and how == "MOM":
            detail = "Method of moments doesn't support truncation"
            raise ValueError(detail)

        if (lfp or zi) and (how != "MLE"):
            detail = (
                "Limited failure or zero-inflated models"
                " can only be made with MLE"
            )
            raise ValueError(detail)

        if zi and (self.support[0] != 0):
            detail = (
                "zero-inflated models can only work with models starting at 0"
            )
            raise ValueError(detail)

    @staticmethod
    def _check_censoring_for_method(
        surv_data: SurpyvalData, how: str, heuristic: str
    ) -> None:
        """Censoring the data has (or lacks) that leaves no fit.

        Maximum likelihood and probability plotting (whose Turnbull
        heuristic is a likelihood) read a censored row with a finite
        truncation bound on its censored side as an interval (#310), so
        whether any row locates a failure is decided from the rows as they
        read them (#559). Maximum product of spacings refuses data whose
        every row is right (or left) censored as before, whatever their
        truncation: with no exact value it has no spacing to score, and
        its objective is the conditional likelihood of the censored rows,
        which has no maximum on such data (it would run a Weibull to
        ``beta = 23.9`` on suspensions at 1, ..., 5 truncated at 9).
        """
        if how == "MPS":
            only_right = bool((surv_data.c == 1).all())
            only_left = bool((surv_data.c == -1).all())
        else:
            only_right = bool(surv_data.mask_r.all())
            only_left = bool(surv_data.mask_l.all())
        if only_right:
            # No failure: the likelihood keeps rising as the distribution
            # moves out past every suspension, with a shape fixed or not.
            raise ValueError(
                "Cannot have only right censored data: with no failure the "
                "likelihood has no maximum (it keeps rising as the "
                "distribution moves out past every suspension). For a "
                "zero-failure analysis of a Weibull or Exponential with a "
                "known shape, surpyval.weibayes(x, c, n, beta=...) gives the "
                "standard lower bound on the scale"
            )

        if only_left:
            raise ValueError("Cannot have only left censored data")

        if surpyval.utils.check_no_censoring(surv_data.c) and (how == "MOM"):
            raise ValueError("Method of moments doesn't support censoring")

        if (
            (surpyval.utils.no_left_or_int(surv_data.c))
            and (how == "MPP")
            and (not heuristic == "Turnbull")
        ):
            detail = (
                "Probability plotting estimation with left or "
                "interval censoring only works with Turnbull heuristic"
            )
            raise ValueError(detail)

    def _check_inside_support(self, surv_data: SurpyvalData) -> None:
        """Every observation leaves the event some probability."""
        lower, upper = self.support
        # One line that names the bounds as the check applies them: an
        # observation must lie strictly inside, so the bounds are written
        # open ("[0, inf]" would read as though 0 were allowed while 0 is
        # what it rejects).
        detail = (
            f"Some of your data is outside the support of the "
            f"{self.name} distribution: observed values must lie "
            f"strictly between {lower} and {upper}, i.e. in "
            f"({lower}, {upper}), and a censored value must leave the "
            f"event some probability. Are some of your observed values "
            f"{lower}, -inf or inf?"
        )
        x_sd, c_sd = surv_data.x, surv_data.c
        if x_sd.ndim == 2:
            bad = (
                ((x_sd[:, 0] <= lower) & (c_sd == 0))
                | ((x_sd[:, 1] >= upper) & (c_sd == 0))
                # An interval endpoint strictly below the support makes
                # the CDF evaluate outside its domain: NaN likelihood
                # everywhere and a silent initial-guess "fit" (#261).
                | ((x_sd[:, 0] < lower) & (c_sd == 2))
                # Survival past the end of the support, or a window
                # wholly beyond it, has probability zero.
                | ((x_sd[:, 0] >= upper) & ((c_sd == 1) | (c_sd == 2)))
            )
        else:
            bad = (
                ((x_sd <= lower) & (c_sd == 0))
                | ((x_sd >= upper) & (c_sd == 0))
                # A left-censored point at or below the support start
                # is a zero-probability observation: the likelihood is
                # -inf/NaN everywhere and the optimiser silently
                # returns the initial guess (#261).
                | ((x_sd <= lower) & (c_sd == -1))
                # Likewise a right-censored point at or beyond the
                # support's end (a Beta censored at 1.5 would return its
                # start with an infinite likelihood).
                | ((x_sd >= upper) & (c_sd == 1))
            )
        if bad.any():
            # A failure at exactly 0 is a unit dead on arrival, which
            # the zero-inflated model is for; a new user will not know
            # the option exists (#514). It needs a support from 0.
            at_zero = (x_sd if x_sd.ndim == 1 else x_sd.max(axis=1)) == 0
            if lower == 0 and (bad & at_zero & (c_sd == 0)).any():
                detail += (
                    " For units that failed at time 0 (dead on "
                    "arrival), fit with `zi=True`."
                )
            raise OutsideSupportError(detail)

    def _clamp_truncation_to_support(self, t: Any) -> Any:
        """Clamp the truncation bounds to the distribution's support.

        Returns the left and right truncation arrays with any value that
        falls outside a *finite* support edge moved onto that edge. An
        infinite support edge leaves the corresponding bound untouched.
        """
        tl = t[:, 0]
        tr = t[:, 1]

        if np.isfinite(self.support[0]):
            tl = np.where(tl < self.support[0], self.support[0], tl)

        if np.isfinite(self.support[1]):
            tr = np.where(tr > self.support[1], self.support[1], tr)

        return tl, tr

    def _initial_guess(
        self,
        data: SurpyvalData,
        offset: bool,
        zi: bool,
        lfp: bool,
        heuristic: str,
    ) -> npt.NDArray:
        """Derive an initial parameter vector for the iterative fitters.

        Builds a working copy of the data with interval- and
        left-censored points imputed to point observations, asks the
        distribution's ``_parameter_initialiser`` for a seed, and appends
        the limited-failure (``p``) and zero-inflation (``f0``) seeds when
        those models are requested. The returned vector is in the natural
        (untransformed) parameter space.

        The working copy is rewrapped as a ``SurpyvalData`` before it is
        handed on, rather than the caller's own object being forwarded:
        the imputation rewrites ``x`` and ``c``, and the masks below drop
        rows, so the caller's object no longer describes it.
        """
        x, c, n = data.x, data.c, data.n
        if (c == 1).all():
            # No row is a failure, so the rows that locate one are those
            # the likelihood reads as intervals: right censored with a
            # finite ``tr`` (#559). They are seeded as the interval rows
            # are, below; with a failure anywhere else, a suspension is
            # seed enough and the start stays as it was.
            lo, hi, c = _rows_as_read(data)
            x = np.column_stack([lo, np.where(c == 2, hi, lo)])
        if x.ndim == 2:
            # If x has 2 dims, then there is intervally
            # censored data. Simply take the midpoint to
            # get the initial estimate.
            x_init = x.mean(axis=1)
            c_init = np.copy(c)
            c_init[c_init == 2] = 0
            n_init = np.copy(n)
        else:
            x_init = np.copy(x)
            c_init = np.copy(c)
            n_init = np.copy(n)

        # If there is left censoring, assume that the
        # left censored value is the midpoint between
        # the censored value and the lowest x value
        x_init[c_init == -1] = (x_init[c_init == -1] + x.min()) / 2
        c_init[c_init == -1] = 0

        # check if the one support is -inf or inf and the other is
        # finite. If it isn't, then the distribution cannot be offset.
        # i.e if both finite or both infinite, then cannot be offset,
        # zero-inflated, or limited failure.
        if (
            np.all(np.isinf(self.support))
            or np.all(np.isfinite(self.support))
            or np.all(np.isnan(self.support))
        ):
            with np.errstate(all="ignore"):
                init = np.array(
                    self._parameter_initialiser(
                        _imputed_data(x_init, c_init, n_init)
                    )
                )
        else:
            with np.errstate(all="ignore"):
                # Remove x where x is out of support
                # This is if data for a zi or lfp model is present
                if not offset:
                    in_support_mask = (x_init > self.support[0]) & (
                        x_init < self.support[1]
                    )

                    # Reduce x, c, and n to the case where it is in the
                    # support of the distribution
                    x_init = x_init[in_support_mask]
                    c_init = c_init[in_support_mask]
                    n_init = n[in_support_mask]
                elif zi:
                    # Exact zeros belong to the zero-inflation
                    # mass; including them would drag the offset
                    # initial guess below zero
                    nonzero_mask = x_init != 0
                    x_init = x_init[nonzero_mask]
                    c_init = c_init[nonzero_mask]
                    n_init = n_init[nonzero_mask]

                # Create an initial estimate with the new points
                init = self._parameter_initialiser(
                    _imputed_data(x_init, c_init, n_init), offset=offset
                )
                init = np.array(init)

                if offset:
                    x_nonzero = x[x != 0] if zi else x
                    init[0] = _offset_start(x_nonzero)

        if lfp:
            _, _, _, F = pp(x_init, c_init, n_init, heuristic="Nelson-Aalen")

            max_F = np.max(F)
            # Kept off the bounds 0 and 1, which the optimiser's arctanh
            # transform maps to -inf and inf.
            init = np.concatenate(
                [init, [np.clip(min(0.6, max_F), 1e-3, 0.999)]]
            )

        if zi:
            if x.ndim == 2:
                x_0 = x[c == 0, 0]
            else:
                x_0 = x[c == 0]

            n_0 = n[c == 0]
            total_failures_at_zero = n_0[x_0 == 0].sum()

            f_0_init = total_failures_at_zero / n.sum()
            # With no failures at zero the natural seed is 0, the edge of
            # f0's bounds, which the optimiser's arctanh transform maps to
            # -inf: every evaluation then warned (35 RuntimeWarnings for a
            # small lfp + zi fit). Start just inside instead.
            init = np.concatenate([init, [np.clip(f_0_init, 1e-3, 0.999)]])

        return init

    def _alternative_base_starts(
        self, data: SurpyvalData, offset: bool
    ) -> "list[npt.NDArray]":
        """
        Further starting points for the distribution's own parameters
        (leading with the offset when ``offset``), tried in addition to the
        default one when fitting by maximum likelihood; none by default.
        """
        return []

    def _alternative_starts(
        self,
        surv_data: SurpyvalData,
        offset: bool,
        zi: bool,
        lfp: bool,
        heuristic: str,
    ) -> "list[npt.NDArray]":
        """
        Complete alternative starting vectors for a maximum-likelihood fit:
        the distribution's own alternatives, with the default's ``p`` /
        ``f0`` seeds appended, and for a limited failure population a start
        from the failures alone.
        """
        bases = self._alternative_base_starts(surv_data, offset)
        starts: list = []
        if bases:
            with np.errstate(all="ignore"):
                default = np.atleast_1d(
                    self._initial_guess(surv_data, offset, zi, lfp, heuristic)
                )
            tail = default[len(default) - int(lfp) - int(zi) :]
            starts += [np.concatenate([np.asarray(b), tail]) for b in bases]
        if lfp and not zi:
            failures = self._lfp_failures_start(surv_data, offset)
            if failures is not None:
                starts.append(failures)
        return starts

    def _lfp_failures_start(
        self, surv_data: SurpyvalData, offset: bool
    ) -> "npt.NDArray | None":
        """
        A limited-failure-population starting point from the failures
        alone: the distribution's own initialiser on the observed failures
        (treated as a complete sample of the susceptible units), and ``p``
        at the observed failure fraction. ``None`` when there are too few
        distinct failures to seed from.
        """
        x = np.asarray(surv_data.x, dtype=float)
        c = np.asarray(surv_data.c)
        n = np.asarray(surv_data.n, dtype=float)
        if x.ndim != 1:
            return None
        observed = c == 0
        if n[observed].sum() < 2 or np.unique(x[observed]).size < 2:
            return None
        with np.errstate(all="ignore"):
            try:
                base = np.array(
                    self._parameter_initialiser(
                        _imputed_data(
                            x[observed],
                            np.zeros(int(observed.sum()), dtype=int),
                            n[observed],
                        ),
                        offset=offset,
                    ),
                    dtype=float,
                )
            except Exception:
                return None
        if offset:
            base[0] = _offset_start(x)
        if not np.all(np.isfinite(base)):
            return None
        p0 = float(np.clip(n[observed].sum() / n.sum(), 1e-3, 0.999))
        return np.concatenate([base, [p0]])

    def _check_fixed_and_init(
        self, model: Parametric, fixed: Any, init: Any, how: str
    ) -> None:
        """Refuse a ``fixed`` or ``init`` the fit cannot use, with a
        message that says why.

        Without it, an unknown name in ``fixed`` would be a bare KeyError,
        a fixed value outside its parameter's bounds (a negative scale, a
        proportion above one, an offset past the first observation) would
        send the optimiser a nan and end in an "MLE Failed" warning and a
        "non-finite parameters" error, and a wrongly sized or
        out-of-bounds ``init`` would fail in ``zip`` or with an
        IndexError.
        """
        names = sorted(model.param_map, key=model.param_map.__getitem__)

        def outside(name: str, value: Any) -> str | None:
            lo, hi = model.bounds[model.param_map[name]]
            lo_v = -np.inf if lo is None else lo
            hi_v = np.inf if hi is None else hi
            if np.isfinite(value) and lo_v < value < hi_v:
                return None
            return f"{name} = {value} lies outside its bounds ({lo_v}, {hi_v})"

        unknown = [name for name in fixed or {} if name not in model.param_map]
        if unknown:
            hints = {
                "gamma": "an offset needs offset=True",
                "f0": "zero inflation needs zi=True",
                model.lfp_name: (
                    "the limited-failure proportion needs lfp=True"
                ),
            }
            hint = "; ".join(hints[k] for k in unknown if k in hints)
            raise ValueError(
                "Unknown parameter(s) {} in `fixed`{}; this model's "
                "parameters are {}.".format(
                    unknown, " ({})".format(hint) if hint else "", names
                )
            )
        for name, value in (fixed or {}).items():
            problem = outside(name, value)
            if problem is not None:
                raise ValueError(f"Cannot fix {name}: {problem}.")

        if how == "MPP" or init is None or len(np.atleast_1d(init)) == 0:
            return
        init_arr = np.atleast_1d(np.asarray(init, dtype=float))
        n_free = len(names) - len(fixed or {})
        free = [name for name in names if name not in (fixed or {})]
        if fixed and len(init_arr) == n_free:
            checked = list(zip(free, init_arr))
        elif len(init_arr) == len(names):
            checked = list(zip(names, init_arr))
        else:
            expected = (
                f"{n_free} (one per free parameter, {free}) or "
                f"{len(names)}"
                if fixed
                else f"{len(names)}"
            )
            raise ValueError(
                f"`init` has {len(init_arr)} value(s) but this {self.name} "
                f"model needs {expected}: {names}."
            )
        for name, value in checked:
            if fixed and name in fixed:
                continue
            problem = outside(name, value)
            if problem is not None:
                raise ValueError(f"Bad `init`: {problem}.")
