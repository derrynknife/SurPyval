"""Recurrent events seen only as counts at inspections: panel-count data.

Each unit is inspected at its own times, and all that is recorded at an
inspection is the number of events since the one before (failures found
at each inspection of a machine; new tumours found at each clinic visit).
The event times themselves are never seen, so the mean cumulative
function of :class:`~surpyval.recurrent.NonParametricCounting`, which
needs them, cannot be used.

Two models are here:

- :class:`PanelCountMCF`: the non-parametric estimate of the mean
  cumulative function from the counts, by the maximum pseudo-likelihood
  estimator (isotonic regression of the cumulative counts, Sun and
  Kalbfleisch 1995) or the non-parametric maximum likelihood estimator
  under a Poisson working model (Wellner and Zhang 2000);
- :class:`PanelCountRegression`: the proportional means model
  :math:`E[N(t) \\mid Z] = \\Lambda_0(t) e^{\\beta' Z}` with an unspecified
  baseline, fitted by the same two likelihoods (Wellner and Zhang 2007).

Both are robust to the Poisson assumption in their estimates; their
bounds come from resampling units (the bootstrap), which does not assume
it either.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression, minimize

from surpyval.recurrent.inference import check_alpha_ci
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    expand,
    fit_columns,
    warn_aliased,
)
from surpyval.utils.covariates import coefficient_names
from surpyval.utils.linalg import percentile_bounds
from surpyval.utils.recurrent_utils import item_label
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import refuses_surpyval_data_with
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_option
from surpyval.utils.warnings import ignoring_warnings

__all__ = ["PanelCountMCF", "PanelCountRegression"]

#: The estimators: maximum pseudo-likelihood and maximum likelihood.
_HOW = ("pseudo", "likelihood")
#: The interpolations of the mean function between inspection times.
_INTERP = ("step", "linear")

# The EM (self-consistency) iteration of the likelihood estimators.
_EM_TOL = 1e-10
#: The tolerance of a bootstrap resample's refit: its error is far below
#: the spread of the resamples that the bounds read.
_BOOT_TOL = 1e-7
_EM_MAXITER = 20000
#: EM iterations before the projected Newton steps take over, the most of
#: those, and the largest grid they are used on (their Hessian is L x L).
_EM_BEFORE_NEWTON = 30
_NEWTON_MAXITER = 200
#: The joint (baseline and beta) Newton steps tried before the alternation
#: resumes; each step may halve 14 times (to 1e-4 of the Newton step).
_JOINT_MAXITER = 30
_NEWTON_MAX_GRID = 2000
#: The Karush-Kuhn-Tucker gap accepted when the iterative convex minorant
#: algorithm can no longer raise the likelihood in floating point.
_STALL_TOL = 1e-6
#: Iterative convex minorant steps before Newton's.
_ICM_FIRST = 20
#: The alternations of the likelihood regression.
_ALT_MAXITER = 2000


class PanelCountConvergenceWarning(UserWarning):
    """The likelihood estimator's iteration stopped before converging."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class _Panel:
    """Validated panel counts, sorted by unit and then by time.

    ``x`` the inspection times, ``n`` the counts since the previous
    inspection (or since time 0 for a unit's first), ``unit`` the index
    of each row's unit in ``ids``, ``Z`` one covariate row per unit (or
    ``None``). The grid is the distinct inspection times; a row's count
    covers the grid cells ``lo .. hi`` (each cell ``(grid[l - 1],
    grid[l]]``)."""

    x: npt.NDArray
    n: npt.NDArray
    unit: npt.NDArray
    ids: npt.NDArray
    Z: npt.NDArray | None
    ids_given: bool

    def __init__(
        self,
        x: npt.NDArray,
        n: npt.NDArray,
        unit: npt.NDArray,
        ids: npt.NDArray,
        Z: npt.NDArray | None,
        ids_given: bool,
    ) -> None:
        self.x, self.n, self.unit, self.ids = x, n, unit, ids
        self.Z, self.ids_given = Z, ids_given
        m = len(ids)
        first = np.r_[True, unit[1:] != unit[:-1]]
        last_row = np.r_[unit[1:] != unit[:-1], True]
        self.prev = np.where(first, 0.0, np.r_[0.0, x[:-1]])
        # The cumulative count of each unit up to each inspection
        cum = np.cumsum(n)
        start = np.repeat(cum[first] - n[first], np.bincount(unit, None, m))
        self.N = cum - start
        self.grid = np.unique(x)
        self.hi = np.searchsorted(self.grid, x)
        prev_hi = np.r_[-1, self.hi[:-1]]
        self.lo = np.where(first, 0, prev_hi + 1)
        #: Each unit's last inspection (grid index) and total count
        self.last = self.hi[last_row]
        self.total = self.N[last_row]
        # The likelihood reads the mean function only at the ends of the
        # rows with a positive count (and the inspection before each) and
        # at each unit's last inspection: its value at another inspection
        # time is not determined by the likelihood (any value between its
        # neighbours' is as likely). The likelihood fit is made on these
        # times, ``rel``, and carried as a step function between them.
        pos = n > 0
        ends = np.r_[self.hi[pos], self.lo[pos & ~first] - 1]
        ends = np.r_[ends, self.last]
        self.rel = np.unique(ends[ends >= 0])
        at = np.searchsorted
        self.r_L = int(self.rel.size)
        self.r_last = at(self.rel, self.last)
        self.r_parts = (
            n[pos],
            np.where(self.lo[pos] > 0, at(self.rel, self.lo[pos] - 1) + 1, 0),
            at(self.rel, self.hi[pos]),
        )

    def expand(self, lam_rel: npt.NDArray) -> npt.NDArray:
        """A mean function at the times ``rel`` as a step function on the
        whole grid (0 before the first of them)."""
        idx = np.searchsorted(self.rel, np.arange(self.grid.size), "right")
        idx = idx - 1
        return np.where(idx >= 0, np.asarray(lam_rel)[np.maximum(idx, 0)], 0.0)

    @property
    def n_units(self) -> int:
        return len(self.ids)

    def resample(self, rng: np.random.Generator) -> _Panel:
        """The panel of ``n_units`` units drawn with replacement."""
        m = self.n_units
        pick = rng.integers(0, m, m)
        counts = np.bincount(self.unit, minlength=m)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        rows = np.concatenate(
            [np.arange(starts[k], starts[k] + counts[k]) for k in pick]
        )
        unit = np.repeat(np.arange(m), counts[pick])
        Z = None if self.Z is None else self.Z[pick]
        return _Panel(self.x[rows], self.n[rows], unit, np.arange(m), Z, False)


def _panel_data(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None,
    n: npt.ArrayLike,
    Z: npt.ArrayLike | None = None,
) -> _Panel:
    """Check and sort panel counts (see :meth:`PanelCountMCF.fit`)."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(
            "x must be a 1-D array of inspection times with at least one "
            "row; got shape {}.".format(x_arr.shape)
        )
    rows = x_arr.size
    if n is None:
        raise ValueError(
            "n is required: the number of events counted at each "
            "inspection since the previous one (0 when none were found)."
        )
    n_arr = np.asarray(n, dtype=float)
    if n_arr.shape != (rows,):
        raise ValueError(
            "n must have one count per row of x ({} rows); got shape "
            "{}.".format(rows, n_arr.shape)
        )
    if not np.isfinite(x_arr).all():
        raise ValueError(
            "Inspection times 'x' must be finite (no NaN or inf)."
        )
    if np.any(x_arr <= 0):
        raise ValueError(
            "Inspection times 'x' must be positive: each unit's counts "
            "start at time 0, so its first count covers (0, x]. Measure "
            "times from the start of each unit's observation."
        )
    if not np.isfinite(n_arr).all() or np.any(n_arr < 0):
        raise ValueError(
            "Counts 'n' must be finite and not negative (0 for an "
            "inspection that found no events)."
        )
    ids_given = i is not None
    if i is None:
        i_arr = np.zeros(rows, dtype=int)
    else:
        i_arr = np.asarray(i)
        if i_arr.shape != (rows,):
            raise ValueError(
                "i must have one unit id per row of x ({} rows); got "
                "shape {}.".format(rows, i_arr.shape)
            )
        from surpyval.utils import is_missing_event

        if (
            np.issubdtype(i_arr.dtype, np.number)
            and not np.isfinite(i_arr).all()
        ) or (
            i_arr.dtype == object and any(is_missing_event(v) for v in i_arr)
        ):
            raise ValueError(
                "Unit ids 'i' must not be missing (None or NaN) or infinite."
            )
    ids, unit = np.unique(i_arr, return_inverse=True)
    unit = unit.ravel()
    order = np.lexsort((x_arr, unit))
    x_arr, n_arr, unit = x_arr[order], n_arr[order], unit[order]
    same = (unit[1:] == unit[:-1]) & (x_arr[1:] == x_arr[:-1])
    if np.any(same):
        k = int(np.flatnonzero(same)[0])
        raise ValueError(
            "Unit {} is inspected twice at x = {}: give one row per "
            "inspection, with the count since the previous one (add the "
            "two counts).".format(item_label(ids[unit[k]]), x_arr[k])
        )
    Z_units = None
    if Z is not None:
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.ndim != 2 or Z_arr.shape[0] != rows:
            raise ValueError(
                "Z must have one row of covariates per row of x ({} "
                "rows); got shape {}.".format(rows, np.shape(Z))
            )
        if not np.isfinite(Z_arr).all():
            raise ValueError(
                "Covariates 'Z' must be finite (no NaN or inf): leave out "
                "the units with a missing value, or give one."
            )
        Z_arr = Z_arr[order]
        first = np.r_[True, unit[1:] != unit[:-1]]
        Z_units = Z_arr[first]
        if np.any(Z_arr != Z_units[unit]):
            k = int(np.flatnonzero(np.any(Z_arr != Z_units[unit], axis=1))[0])
            raise ValueError(
                "Unit {} has covariates that change between its rows: "
                "the proportional means model takes one covariate value "
                "per unit (time-varying covariates are not "
                "supported).".format(item_label(ids[unit[k]]))
            )
    return _Panel(x_arr, n_arr, unit, ids, Z_units, ids_given)


#: The refusal of a SurpyvalData by the panel-count models' ``fit``.
_REFUSAL = (
    "PanelCountMCF and PanelCountRegression take arrays of visits and "
    "counts, not a SurpyvalData: x the inspection times, n the number of "
    "events found at each inspection since the previous one (0 where none "
    "were found), i the unit. A SurpyvalData's rows are events or "
    "stretches of a unit's history, and its n counts identical units (at "
    "least 1), so an inspection that found nothing has no row in it. Pass "
    "the arrays to fit, or the columns to fit_from_df; for the times of "
    "the events themselves, NonParametricCounting.fit(data) and the other "
    "recurrent models take a SurpyvalData."
)
_refuses_data = refuses_surpyval_data_with(_REFUSAL)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def _pseudo_mean(panel: _Panel, w_row: npt.NDArray) -> npt.NDArray:
    """The maximum pseudo-likelihood mean function at the grid: the
    weighted isotonic regression of the cumulative counts (each row
    weighted by its unit's ``exp(beta' Z)``, ``w_row``)."""
    L = panel.grid.size
    S = np.bincount(panel.hi, panel.N, L)
    W = np.bincount(panel.hi, w_row, L)
    return isotonic_regression(S / W, weights=W, increasing=True).x


def _pseudo_loglik(
    panel: _Panel, lam: npt.NDArray, eta_row: npt.NDArray
) -> float:
    """The log pseudo-likelihood, the Poisson likelihood of each
    cumulative count taken as if independent of the others."""
    at = lam[panel.hi]
    N = panel.N
    with np.errstate(divide="ignore", invalid="ignore"):
        log_at = np.where(N > 0, np.log(at), 0.0)
    return float(np.sum(N * (log_at + eta_row) - at * np.exp(eta_row)))


def _likelihood_parts(
    panel: _Panel,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The rows with a positive count (the only ones whose cells the
    likelihood's log term reads): their counts and cell ranges on the
    likelihood's grid, ``panel.rel``."""
    return panel.r_parts


def _full_loglik(
    panel: _Panel,
    delta: npt.NDArray,
    w_unit: npt.NDArray,
    parts: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> float:
    """The log-likelihood of the counts as independent Poisson counts
    with means ``exp(beta' Z) (Lambda(t_j) - Lambda(t_{j-1}))``, without
    the terms in ``beta`` alone (the ``n log w`` sum)."""
    n, lo, hi = parts
    C = np.r_[0.0, np.cumsum(delta)]
    D = C[hi + 1] - C[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(
            np.sum(n * np.log(D)) - np.sum(w_unit * C[panel.r_last + 1])
        )


def _em_map(
    delta: npt.NDArray,
    R: npt.NDArray,
    parts: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> tuple[npt.NDArray, npt.NDArray]:
    """One self-consistency (EM) step of the likelihood estimator, and the
    ratio ``A / R`` of each cell (1 where the likelihood is stationary in
    that cell's increment; at most 1 where the increment is 0)."""
    n, lo, hi = parts
    L = delta.size
    C = np.r_[0.0, np.cumsum(delta)]
    D = C[hi + 1] - C[lo]
    # A trial point (SQUAREM's extrapolation) can leave a count's interval
    # with no mean: its map is then not finite, and the caller's check of
    # the likelihood turns it down.
    with np.errstate(divide="ignore", invalid="ignore"):
        g = n / D
        # Sum each count's share over the cells it covers: a difference
        # array
        acc = np.bincount(lo, g, L + 1) - np.bincount(hi + 1, g, L + 1)
        ratio = np.cumsum(acc)[:L] / R
        return delta * ratio, ratio


def _kkt(delta: npt.NDArray, ratio: npt.NDArray) -> float:
    """The Karush-Kuhn-Tucker gap of the likelihood in the increments, in
    the ratio ``A / R`` of each cell (``A`` the counts' share of the cell,
    ``R`` the units under observation in it): no cell's increment can be
    raised (``ratio <= 1``), and every positive one is stationary
    (``ratio = 1``). An increment below 1e-12 of the largest counts as 0.
    (Weighting the gap by the increment, as before, passed a small
    increment that was still far from its value.)"""
    top = float(np.max(delta))
    positive = delta > 1e-12 * top
    gap = float(np.max(ratio)) - 1.0
    if np.any(positive):
        gap = max(gap, float(np.max(np.abs(ratio[positive] - 1.0))))
    return gap


def _cell_hessian(
    lo: npt.NDArray, hi: npt.NDArray, wts: npt.NDArray, L: int
) -> npt.NDArray:
    """``sum_r wts_r [lo_r <= k, l <= hi_r]``: minus the Hessian of the
    log term in the increments, from the counts' cell ranges by a 2-D
    cumulative sum over (start, end) pairs, in O(L^2) whatever the number
    of rows."""
    M = np.bincount(lo * L + hi, wts, L * L).reshape(L, L)
    P = np.cumsum(M, axis=0)
    P = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]
    # P[k, l] for k <= l is the entry; the lower triangle mirrors it
    upper = np.triu(P)
    return upper + np.triu(P, 1).T


def _newton_step(
    lo: npt.NDArray,
    hi: npt.NDArray,
    wts: npt.NDArray,
    L: int,
    free: npt.NDArray,
    grad: npt.NDArray,
) -> npt.NDArray:
    """Solve ``H d = grad`` on the free cells, ``H`` minus the Hessian of
    the log term (:func:`_cell_hessian`): directly up to
    ``_NEWTON_MAX_GRID`` cells, beyond by conjugate gradients, whose
    products with ``H`` (``B' W B v``, ``B`` the rows' cell ranges) take
    two cumulative sums and so O(rows + grid)."""
    if L <= _NEWTON_MAX_GRID:
        H = _cell_hessian(lo, hi, wts, L)[np.ix_(free, free)]
        ridge = 1e-12 * max(float(np.max(np.diag(H))), 1e-300)
        return np.linalg.solve(H + ridge * np.eye(H.shape[0]), grad)
    from scipy.sparse.linalg import LinearOperator, cg

    diag = np.cumsum(
        np.bincount(lo, wts, L + 1) - np.bincount(hi + 1, wts, L + 1)
    )
    diag = diag[:L][free]
    ridge = 1e-12 * max(float(np.max(diag)), 1e-300)

    def matvec(v: npt.NDArray) -> npt.NDArray:
        full = np.zeros(L)
        full[free] = np.ravel(v)
        C = np.r_[0.0, np.cumsum(full)]
        u = wts * (C[hi + 1] - C[lo])
        acc = np.bincount(lo, u, L + 1) - np.bincount(hi + 1, u, L + 1)
        return np.cumsum(acc)[:L][free] + ridge * np.ravel(v)

    k = int(free.sum())
    H = LinearOperator((k, k), matvec=matvec, dtype=float)
    M = LinearOperator((k, k), matvec=lambda v: np.ravel(v) / (diag + ridge))
    d, info = cg(H, grad, rtol=1e-10, maxiter=1000, M=M)
    if not np.all(np.isfinite(d)):
        raise np.linalg.LinAlgError("conjugate gradients failed")
    return d


def _epsilon_active(
    delta: npt.NDArray,
    grad: npt.NDArray,
    parts: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> npt.NDArray:
    """The increments held at 0 in a projected Newton step: those within
    epsilon of 0 that would fall, epsilon the length of a projected,
    diagonally scaled gradient step (Bertsekas, 1982)."""
    n, lo, hi = parts
    L = delta.size
    C = np.r_[0.0, np.cumsum(delta)]
    D = C[hi + 1] - C[lo]
    q2 = n / D**2
    diag = np.cumsum(
        np.bincount(lo, q2, L + 1) - np.bincount(hi + 1, q2, L + 1)
    )[:L]
    trial = np.maximum(delta + grad / np.maximum(diag, 1e-300), 0.0)
    top = float(np.max(delta))
    eps = min(float(np.max(np.abs(trial - delta))), 0.1 * top)
    eps = max(eps, 1e-6 * top)
    return (delta <= eps) & (grad <= 0)


def _projected_newton(
    panel: _Panel,
    delta: npt.NDArray,
    w_unit: npt.NDArray,
    R: npt.NDArray,
    parts: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    tol: float,
) -> tuple[npt.NDArray, bool]:
    """Projected Newton steps (Bertsekas, 1982) on the concave likelihood
    in the increments, which are kept non-negative: Newton on the cells
    that are positive or would rise, the others held at 0, with an Armijo
    search along the projected path."""
    n, lo, hi = parts
    L = delta.size
    ll = _full_loglik(panel, delta, w_unit, parts)
    for _ in range(_NEWTON_MAXITER):
        _, ratio = _em_map(delta, R, parts)
        if _kkt(delta, ratio) < tol:
            return delta, True
        grad = (ratio - 1.0) * R
        # Held at 0: the cells at or near 0 that would fall (Bertsekas's
        # epsilon-active set)
        held = _epsilon_active(delta, grad, parts)
        free = ~held
        if np.any(held & (delta > 0)):
            zeroed = np.where(held, 0.0, delta)
            ll_zeroed = _full_loglik(panel, zeroed, w_unit, parts)
            if ll_zeroed >= ll:
                delta, ll = zeroed, ll_zeroed
        C = np.r_[0.0, np.cumsum(delta)]
        D = C[hi + 1] - C[lo]
        try:
            step = np.zeros(L)
            step[free] = _newton_step(lo, hi, n / D**2, L, free, grad[free])
        except np.linalg.LinAlgError:
            return delta, False
        t = 1.0
        for _ in range(60):
            new = np.maximum(delta + t * step, 0.0)
            ll_new = _full_loglik(panel, new, w_unit, parts)
            if np.isfinite(ll_new) and ll_new >= ll + 1e-4 * float(
                grad @ (new - delta)
            ):
                break
            t *= 0.5
        else:
            # No ascent the likelihood can resolve in floating point
            return delta, _kkt(delta, ratio) < _STALL_TOL
        if ll_new <= ll and np.array_equal(new, delta):
            return delta, _kkt(delta, ratio) < _STALL_TOL
        delta, ll = new, ll_new
    return delta, False


def _icm(
    panel: _Panel,
    delta: npt.NDArray,
    w_unit: npt.NDArray,
    R: npt.NDArray,
    parts: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    tol: float,
    maxiter: int = _EM_MAXITER,
) -> tuple[npt.NDArray, bool]:
    """The iterative convex minorant algorithm (Groeneboom and Wellner,
    1992; Jongbloed, 1998, with its line search), as Wellner and Zhang
    (2000) use it: in the mean function's values ``Lambda`` at the grid,
    each step is the weighted isotonic regression of ``Lambda + g / h``
    (``g`` the gradient, ``h`` the diagonal of minus the Hessian), cut at
    0, searched along. Every step costs O(rows + grid), so it serves
    grids too large for Newton's method."""
    n, lo, hi = parts
    L = delta.size
    end_w = np.bincount(panel.r_last, w_unit, L)
    lam = np.cumsum(delta)
    ll = _full_loglik(panel, delta, w_unit, parts)
    for _ in range(maxiter):
        _, ratio = _em_map(delta, R, parts)
        if _kkt(delta, ratio) < tol:
            return delta, True
        C = np.r_[0.0, lam]
        D = C[hi + 1] - C[lo]
        q = n / D
        # Lambda at a row's end gains n / D, at the cell before its start
        # loses it (Lambda_0 = 0 is fixed, so a start at 0 drops out)
        start = lo > 0
        g = np.bincount(hi, q, L) - end_w
        g -= np.bincount(lo[start] - 1, q[start], L)
        q2 = n / D**2
        h = np.bincount(hi, q2, L) + np.bincount(lo[start] - 1, q2[start], L)
        floor = 1e-8 * float(np.mean(h[h > 0])) if np.any(h > 0) else 1.0
        h = np.maximum(h, floor)
        target = isotonic_regression(lam + g / h, weights=h).x
        target = np.maximum(target, 0.0)
        direction = target - lam
        slope = float(g @ direction)
        if slope <= 0:
            return delta, _kkt(delta, ratio) < _STALL_TOL
        t = 1.0
        for _ in range(60):
            new = lam + t * direction
            d_new = np.diff(np.r_[0.0, new])
            d_new = np.maximum(d_new, 0.0)
            ll_new = _full_loglik(panel, d_new, w_unit, parts)
            if np.isfinite(ll_new) and ll_new >= ll + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            # No ascent the likelihood can resolve in floating point
            return delta, _kkt(delta, ratio) < _STALL_TOL
        if ll_new <= ll:
            # Accepted only by rounding: the search has stalled
            return delta, _kkt(delta, ratio) < _STALL_TOL
        lam, delta, ll = np.cumsum(d_new), d_new, ll_new
    return delta, False


def _full_mean(
    panel: _Panel,
    w_unit: npt.NDArray,
    start: npt.NDArray | None = None,
    tol: float = _EM_TOL,
    icm_first: int = _ICM_FIRST,
) -> tuple[npt.NDArray, bool]:
    """The maximum likelihood increments of the mean function on the grid,
    and whether they converged.

    The self-consistency (EM) algorithm, accelerated by SQUAREM (Varadhan
    and Roland, 2008), brings the increments near the maximum, and
    projected Newton steps finish (on grids of up to ``_NEWTON_MAX_GRID``
    times; beyond, EM runs to the end). Convergence is a Karush-Kuhn-Tucker
    check (``_kkt``) to ``tol``."""
    L = panel.r_L
    # Units under observation in each cell, weighted by exp(beta' Z)
    R = np.cumsum(np.bincount(panel.r_last, w_unit, L)[::-1])[::-1]
    parts = _likelihood_parts(panel)
    if parts[0].size == 0:
        return np.zeros(L), True
    cold = start is None or bool(np.any(~np.isfinite(start)))
    if cold:
        # A strictly positive start: every cell can then take a share.
        lam = _pseudo_mean(panel, w_unit[panel.unit])[panel.rel]
        inc = np.diff(np.r_[0.0, lam])
        start = 0.5 * inc + 0.5 * lam[-1] / L + 1e-300
    # A cell no positive count covers has its increment at 0 at the
    # maximum (only the -R delta term reads it); every other starts
    # positive, so each count's interval has a positive mean.
    n, lo, hi = parts
    covered = (
        np.cumsum(
            np.bincount(lo, None, L + 1) - np.bincount(hi + 1, None, L + 1)
        )[:L]
        > 0
    )
    floor = 1e-6 * float(np.sum(n)) / float(np.sum(R)) + 1e-300
    delta = np.where(
        covered, np.maximum(np.asarray(start, dtype=float), floor), 0.0
    )
    ll = _full_loglik(panel, delta, w_unit, parts)
    # A start the caller gave is near the maximum already
    before_newton = _EM_BEFORE_NEWTON if cold else 2
    for k in range(_EM_MAXITER):
        if k == before_newton:
            # A few iterative convex minorant steps find which increments
            # are 0 (their isotonic projection does it in one pass), then
            # Newton finishes; the minorant algorithm to the end if it
            # cannot.
            if icm_first:
                delta, ok = _icm(
                    panel, delta, w_unit, R, parts, tol, icm_first
                )
                if ok:
                    return delta, True
            for finish in (_projected_newton, _icm):
                delta, ok = finish(panel, delta, w_unit, R, parts, tol)
                if ok:
                    return delta, True
            ll = _full_loglik(panel, delta, w_unit, parts)
        d1, ratio = _em_map(delta, R, parts)
        if _kkt(delta, ratio) < tol:
            return delta, True
        d2, _ = _em_map(d1, R, parts)
        r = d1 - delta
        v = d2 - d1 - r
        nr, nv = np.linalg.norm(r), np.linalg.norm(v)
        new = d2
        if nv > 0:
            alpha = min(-nr / nv, -1.0)
            trial = np.maximum(delta - 2 * alpha * r + alpha**2 * v, 0.0)
            trial, _ = _em_map(trial, R, parts)
            if _full_loglik(panel, trial, w_unit, parts) >= _full_loglik(
                panel, d2, w_unit, parts
            ):
                new = trial
        ll_new = _full_loglik(panel, new, w_unit, parts)
        if not np.isfinite(ll_new) or ll_new < ll - 1e-12 * abs(ll):
            new = d2
            ll_new = _full_loglik(panel, new, w_unit, parts)
        delta, ll = new, ll_new
    return delta, False


def _warn_unconverged(what: str) -> None:
    warnings.warn(
        "{} did not converge in {} iterations of the self-consistency "
        "algorithm; the estimate returned is the last iterate. Use "
        "how='pseudo', whose estimate is exact, or check the data for "
        "units with very many inspections.".format(what, _EM_MAXITER),
        PanelCountConvergenceWarning,
        stacklevel=3,
    )


def _nonparametric_mean(
    panel: _Panel,
    how: str,
    near: tuple[npt.NDArray, npt.NDArray] | None = None,
    tol: float = _EM_TOL,
) -> tuple[npt.NDArray, bool]:
    """The estimate at the panel's grid. ``near``, an estimate ``(times,
    values)`` of the same mean function (the full data's, for a
    bootstrap resample), starts the likelihood's search, and ``tol`` is
    its Karush-Kuhn-Tucker tolerance."""
    ones = np.ones(panel.n_units)
    if how == "pseudo":
        return _pseudo_mean(panel, ones[panel.unit]), True
    start = None
    if near is not None:
        lam = _step_lookup(near[0], near[1], panel.grid[panel.rel], "step")
        lam = np.where(np.isfinite(lam), lam, near[1][-1])
        start = np.diff(np.r_[0.0, np.maximum.accumulate(lam)])
    delta, ok = _full_mean(panel, ones, start, tol)
    return panel.expand(np.cumsum(delta)), ok


def _aliased(Z: npt.NDArray) -> npt.NDArray:
    """The covariate columns (one row per unit) the proportional means
    model cannot estimate, with one warning naming them: a constant column
    (the baseline's scale absorbs it) or one that is a linear combination
    of the others; the later of two collinear columns, as in R."""
    Zc = Z - Z.mean(axis=0)
    aliased = aliased_columns(Zc.T @ Zc, Z.shape[0], constant_columns(Z))
    if aliased.size:
        warn_aliased(
            aliased,
            "they are constant over the units (the baseline mean "
            "function's scale is the model's intercept) or a linear "
            "combination of the other columns",
        )
    return aliased


def _regression_fit(
    panel: _Panel,
    how: str,
    init: npt.NDArray | None = None,
    near: tuple[npt.NDArray, npt.NDArray] | None = None,
    tol: float = _EM_TOL,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """The proportional means fit: ``beta`` and the baseline mean function
    at the grid at ``Z = 0``, and whether it converged. ``init`` starts
    ``beta`` and ``near``, a baseline ``(times, values)`` at ``Z = 0`` (the
    full data's, for a bootstrap resample), the likelihood's baseline.

    The pseudo-likelihood's profile over ``beta`` is concave (the
    pseudo-likelihood is jointly concave in ``log Lambda`` and ``beta``)
    and its baseline for a given ``beta`` is exact, so ``beta`` is found by
    BFGS on the profile with its exact gradient. The likelihood is
    maximised by alternating the self-consistency algorithm for the
    baseline with Newton's method for ``beta`` (a Poisson regression of
    each unit's total count with the baseline at its last inspection as
    offset), each step raising the likelihood."""
    assert panel.Z is not None
    Z = panel.Z
    zbar = Z.mean(axis=0)
    scale = (Z - zbar).std(axis=0)
    # Centred and scaled, so the baseline and beta are not traded off in
    # the search and every coordinate has the same unit
    Zs = (Z - zbar) / scale
    b = np.zeros(Z.shape[1]) if init is None else init * scale
    if how == "pseudo":
        b, lam_c, ok = _pseudo_regression(panel, Zs, b)
    else:
        start = None
        if near is not None and init is not None:
            lam = _step_lookup(near[0], near[1], panel.grid[panel.rel], "step")
            lam = np.where(np.isfinite(lam), lam, near[1][-1])
            lam = np.maximum.accumulate(lam) * np.exp(float(zbar @ init))
            start = np.diff(np.r_[0.0, lam])
        b, lam_c, ok = _likelihood_regression(panel, Zs, b, start, tol)
    beta = b / scale
    # The baseline at Z = 0, from the one at the mean covariates
    lam0 = lam_c * np.exp(-float(zbar @ beta))
    return beta, lam0, ok


def _pseudo_regression(
    panel: _Panel, Zs: npt.NDArray, b0: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    def negll(b: npt.NDArray) -> tuple[float, npt.NDArray]:
        eta = Zs @ b
        w_row = np.exp(eta)[panel.unit]
        lam = _pseudo_mean(panel, w_row)
        value = _pseudo_loglik(panel, lam, eta[panel.unit])
        # The envelope theorem: the baseline is at its maximum
        grad = Zs[panel.unit].T @ (panel.N - lam[panel.hi] * w_row)
        return -value, -grad

    res = minimize(negll, b0, jac=True, method="BFGS", options={"gtol": 1e-9})
    b = res.x
    _, grad = negll(b)
    lam = _pseudo_mean(panel, np.exp(Zs @ b)[panel.unit])
    # The gradient, relative to the size of the counts it sums
    size = max(float(np.sum(panel.N)), 1.0)
    return b, lam, bool(np.max(np.abs(grad)) <= 1e-6 * size)


def _joint_newton(
    panel: _Panel,
    Zs: npt.NDArray,
    delta: npt.NDArray,
    b: npt.NDArray,
    totals: npt.NDArray,
    tol: float = _EM_TOL,
) -> tuple[npt.NDArray, npt.NDArray] | None:
    """Projected Newton steps on the likelihood in the baseline's
    increments and ``beta`` together, from a point near the maximum; the
    increments and ``beta`` at a Karush-Kuhn-Tucker point (``tol``),
    or ``None`` where a step is not an ascent (the alternation then goes
    on)."""
    parts = _likelihood_parts(panel)
    n, lo, hi = parts
    L, p = delta.size, b.size

    def loglik(d: npt.NDArray, bb: npt.NDArray) -> float:
        return _full_loglik(panel, d, np.exp(Zs @ bb), parts) + float(
            totals @ bb
        )

    ll = loglik(delta, b)
    for _ in range(_JOINT_MAXITER):
        w = np.exp(Zs @ b)
        R = np.cumsum(np.bincount(panel.r_last, w, L)[::-1])[::-1]
        _, ratio = _em_map(delta, R, parts)
        lam_last = np.cumsum(delta)[panel.r_last]
        g_b = totals - Zs.T @ (w * lam_last)
        size = max(float(np.sum(panel.total)), 1.0)
        if _kkt(delta, ratio) < tol and np.max(np.abs(g_b)) < (1e-8 * size):
            return delta, b
        g_d = (ratio - 1.0) * R
        held = _epsilon_active(delta, g_d, parts)
        free = ~held
        if np.any(held & (delta > 0)):
            zeroed = np.where(held, 0.0, delta)
            ll_zeroed = loglik(zeroed, b)
            if ll_zeroed >= ll:
                delta, ll = zeroed, ll_zeroed
        C = np.r_[0.0, np.cumsum(delta)]
        D = C[hi + 1] - C[lo]
        H_dd = _cell_hessian(lo, hi, n / D**2, L)
        # Units under observation in each cell, weighted by w Z
        H_db = np.cumsum(
            np.stack(
                [np.bincount(panel.r_last, w * Zs[:, k], L) for k in range(p)],
                axis=1,
            )[::-1],
            axis=0,
        )[::-1]
        H_bb = (Zs * (w * lam_last)[:, None]).T @ Zs
        H = np.block([[H_dd, H_db], [H_db.T, H_bb]])
        keep = np.r_[free, np.ones(p, bool)]
        grad = np.r_[g_d, g_b]
        H = H[np.ix_(keep, keep)]
        ridge = 1e-12 * max(float(np.max(np.diag(H))), 1e-300)
        try:
            sub = np.linalg.solve(H + ridge * np.eye(H.shape[0]), grad[keep])
        except np.linalg.LinAlgError:
            return None
        step = np.zeros(L + p)
        step[keep] = sub
        if float(grad @ step) <= 0:
            return None
        t = 1.0
        for _ in range(14):
            d_new = np.maximum(delta + t * step[:L], 0.0)
            b_new = b + t * step[L:]
            ll_new = loglik(d_new, b_new)
            moved = np.r_[d_new - delta, b_new - b]
            if np.isfinite(ll_new) and ll_new >= ll + 1e-4 * float(
                grad @ moved
            ):
                break
            t *= 0.5
        else:
            return None
        delta, b, ll = d_new, b_new, ll_new
    return None


def _likelihood_regression(
    panel: _Panel,
    Zs: npt.NDArray,
    b0: npt.NDArray,
    start: npt.NDArray | None = None,
    final_tol: float = _EM_TOL,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    b = b0.copy()
    totals = Zs.T @ panel.total
    delta = start
    tol = max(1e-4, final_tol)
    converged = False
    tried_joint = False
    # A start near the maximum (a bootstrap resample's, from the full
    # data's fit) goes to the joint Newton steps after one alternation
    warm = start is not None
    for _ in range(_ALT_MAXITER):
        delta, inner_ok = _full_mean(panel, np.exp(Zs @ b), delta, tol)
        at_last = np.cumsum(delta)[panel.r_last]
        before = b.copy()
        for _ in range(100):
            mu = np.exp(Zs @ b) * at_last
            grad = totals - Zs.T @ mu
            hess = (Zs * mu[:, None]).T @ Zs
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                return b, panel.expand(np.cumsum(delta)), False
            b = b + step
            if not np.all(np.isfinite(b)):
                return before, panel.expand(np.cumsum(delta)), False
            if np.max(np.abs(step)) < 1e-12:
                break
        change = float(np.max(np.abs(b - before)))
        if change < 1e-9 and tol <= final_tol and inner_ok:
            converged = True
            break
        if (
            (change < 1e-3 or warm)
            and not tried_joint
            and panel.r_L <= (_NEWTON_MAX_GRID)
        ):
            # Near the maximum: Newton on the baseline and beta together
            # converges quadratically where the alternation zigzags.
            tried_joint = True
            joint = _joint_newton(panel, Zs, delta, b, totals, final_tol)
            if joint is not None:
                return joint[1], panel.expand(np.cumsum(joint[0])), True
        # Solve the baseline more tightly as beta settles
        tol = max(final_tol, min(tol, 1e-3 * change))
    assert delta is not None
    return b, panel.expand(np.cumsum(delta)), converged


def _step_lookup(
    grid: npt.NDArray, values: npt.NDArray, x: npt.NDArray, interp: str
) -> npt.NDArray:
    """The mean function (``values`` at ``grid``, from 0 at time 0) at
    ``x``: a step function or linear between the inspection times; NaN
    before 0, after the last inspection and at NaN."""
    values = np.asarray(values, dtype=float)
    if interp == "step":
        idx = np.searchsorted(grid, x, side="right") - 1
        out = values[..., np.clip(idx, 0, None)]
        out = np.where(idx < 0, 0.0, out)
    else:
        g = np.r_[0.0, grid]
        if values.ndim == 1:
            out = np.interp(x, g, np.r_[0.0, values])
        else:
            out = np.stack([np.interp(x, g, np.r_[0.0, v]) for v in values])
    bad = (x < 0) | (x > grid[-1]) | np.isnan(x)
    return np.where(bad, np.nan, out)


def _boot_bounds(
    draws: npt.NDArray, alpha_ci: float, bound: str
) -> npt.NDArray:
    """Percentile bootstrap bounds of each column of ``draws``, leaving
    out the resamples on which a column is not defined (NaN)."""
    out = np.full(
        draws.shape[1:] + ((2,) if bound == "two-sided" else ()), np.nan
    )
    for k in range(draws.shape[1]):
        col = draws[:, k]
        col = col[np.isfinite(col)]
        if col.size:
            out[k] = percentile_bounds(col, alpha_ci, bound)
    return out


def _check_boot_args(n_boot: Any) -> None:
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 2
    ):
        raise ValueError(
            "n_boot must be an integer of at least 2 (the number of "
            "resamples of the units); got {!r}. 1000 or more is usual "
            "for 95% bounds.".format(n_boot)
        )


def _data_dict(panel: _Panel, x: Any, i: Any, n: Any) -> dict:
    return {
        "x": np.asarray(x, dtype=float).tolist(),
        "i": None if i is None else [to_native(v) for v in np.asarray(i)],
        "n": np.asarray(n, dtype=float).tolist(),
    }


# ---------------------------------------------------------------------------
# Non-parametric
# ---------------------------------------------------------------------------
class PanelCountMCF(SerialisableMixin):
    r"""
    The non-parametric mean cumulative function (MCF) from panel counts:
    the expected number of events per unit by time :math:`t`, estimated
    from counts made at inspections, with no event times.

    Each unit :math:`i` is inspected at its own times :math:`t_{i1} <
    \dots < t_{iJ_i}`, and the count :math:`n_{ij}` of events in
    :math:`(t_{i,j-1}, t_{ij}]` is recorded (:math:`t_{i0} = 0`). The mean
    function :math:`\Lambda(t) = E[N(t)]` is estimated at the distinct
    inspection times, as a non-decreasing step function, by one of two
    estimators:

    - ``how="likelihood"`` (the default), the non-parametric maximum
      likelihood estimator of Wellner and Zhang (2000): the
      :math:`\Lambda` that maximises the likelihood of the counts as if
      each were Poisson with mean :math:`\Lambda(t_{ij}) -
      \Lambda(t_{i,j-1})`,

      .. math::
          \sum_{i,j} n_{ij} \log\big(\Lambda(t_{ij}) -
          \Lambda(t_{i,j-1})\big) - \sum_i \Lambda(t_{iJ_i}).

      It is found by the self-consistency (EM) algorithm, accelerated.
    - ``how="pseudo"``, the maximum pseudo-likelihood estimator (Sun and
      Kalbfleisch, 1995; Wellner and Zhang, 2000): the likelihood of the
      cumulative counts :math:`N_{ij} = n_{i1} + \dots + n_{ij}` taken
      as independent Poisson counts, whose maximiser is the isotonic
      (non-decreasing) regression of the :math:`N_{ij}` on the times. It
      is exact and fast; the likelihood estimator is more efficient when
      the inspection times differ a lot between units.

    The Poisson model is a working model: both estimators are consistent
    for the mean function of any counting process whose inspection times
    do not depend on its counts. When every unit is inspected at the same
    times both reduce to the mean count per unit at each inspection.

    Bounds (:meth:`mcf_cb`) are percentile bootstrap bounds from
    resampling units, which do not assume the Poisson model.

    Call ``PanelCountMCF.fit`` (or ``fit_from_df``); it returns a fitted
    instance.

    References
    ----------
    Sun, J. and Kalbfleisch, J. D. (1995), "Estimation of the mean
    function of point processes based on panel count data", *Statistica
    Sinica*, 5(1), 279-289.

    Wellner, J. A. and Zhang, Y. (2000), "Two estimators of the mean of a
    counting process with panel count data", *Annals of Statistics*,
    28(3), 779-814.
    """

    #: The estimator, ``"likelihood"`` or ``"pseudo"``.
    how: str
    #: The distinct inspection times, where the estimate is made.
    x: npt.NDArray
    #: The estimated mean function at ``x``.
    mcf_hat: npt.NDArray
    #: Whether the likelihood estimator's iteration converged.
    converged: bool
    _panel: _Panel
    _raw: dict
    _boot: dict

    @classmethod
    @takes_unit
    @_refuses_data
    def fit(
        cls,
        x: npt.ArrayLike,
        n: npt.ArrayLike,
        i: npt.ArrayLike | None = None,
        how: str = "likelihood",
    ) -> PanelCountMCF:
        """
        Estimate the mean cumulative function from panel counts.

        Parameters
        ----------
        x : array_like
            The inspection times, one row per inspection of a unit,
            measured from the start of the unit's observation (all
            positive).
        n : array_like
            The number of events found at each inspection: those since
            the unit's previous inspection, or since time 0 at its first.
            0 where none were found (those rows count: they say no events
            happened in the interval).
        i : array_like, optional
            The unit each row belongs to. Defaults to a single unit.
        how : str, optional
            ``"likelihood"`` (the default) for the maximum likelihood
            estimator, or ``"pseudo"`` for the maximum pseudo-likelihood
            (isotonic regression) estimator.

        Returns
        -------
        PanelCountMCF
            The fitted estimate.

        Examples
        --------
        Three units, each inspected at its own times:

        >>> from surpyval.recurrent import PanelCountMCF
        >>> x = [2, 5, 9, 3, 8, 4, 6, 10]
        >>> n = [1, 0, 2, 1, 2, 0, 2, 1]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3]
        >>> model = PanelCountMCF.fit(x, n, i=i, how="pseudo")
        >>> model.mcf([2, 5, 10]).round(4)
        array([0.6667, 1.    , 3.    ])
        """
        check_option("how", how, _HOW)
        panel = _panel_data(x, i, n)
        out = cls._from_panel(panel, how)
        out._raw = {"x": x, "i": i, "n": n}
        if not out.converged:
            _warn_unconverged("The likelihood estimate of the MCF")
        return out

    @classmethod
    def _from_panel(cls, panel: _Panel, how: str) -> PanelCountMCF:
        out = cls()
        out.how = how
        out._panel = panel
        out.x = panel.grid
        out.mcf_hat, out.converged = _nonparametric_mean(panel, how)
        out._boot = {}
        return out

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        n_col: str,
        i_col: str | None = None,
        how: str = "likelihood",
    ) -> PanelCountMCF:
        """
        Fit to panel counts held in the columns of a DataFrame, one row
        per inspection: the columns are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import PanelCountMCF
        >>> df = pd.DataFrame({
        ...     "hours": [500, 1000, 1500, 700, 1400],
        ...     "found": [0, 1, 1, 1, 0],
        ...     "pump": ["A", "A", "A", "B", "B"],
        ... })
        >>> model = PanelCountMCF.fit_from_df(
        ...     df, "hours", "found", i_col="pump"
        ... )
        >>> model.mcf([1000, 1500]).round(4)
        array([1., 2.])
        """
        from surpyval.utils.dataframe import frame_column

        return cls.fit(
            frame_column(df, x_col, "x_col", time=True),
            frame_column(df, n_col, "n_col"),
            i=None if i_col is None else frame_column(df, i_col, "i_col"),
            how=how,
        )

    @keeps_query_shape
    def mcf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        """
        The estimated mean cumulative function at ``x``.

        Parameters
        ----------
        x : array_like
            The times.
        interp : str, optional
            ``"step"`` (the default): the estimate at the last inspection
            time at or before ``x`` (0 before the first), the form of the
            estimator; or ``"linear"``, interpolated between inspection
            times from 0 at time 0.

        Returns
        -------
        numpy array
            The mean function at each ``x``; NaN before 0 and after the
            last inspection, where the data say nothing.
        """
        check_option("interp", interp, _INTERP)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        return _step_lookup(self.x, self.mcf_hat, q, interp)

    def bootstrap(
        self, n_boot: int = 1000, random_state: Any = None
    ) -> npt.NDArray:
        """
        The estimate refitted to ``n_boot`` resamples of the units (drawn
        with replacement), at the inspection times ``x``: one row per
        resample. A time beyond a resample's last inspection is NaN.

        A seed (``random_state``) caches the resamples, so the bounds at
        other times or levels reuse them.
        """
        _check_boot_args(n_boot)
        key = (int(n_boot), _seed_key(random_state))
        if key[1] is not None and key in self._boot:
            return self._boot[key]
        rng = as_generator(random_state)
        draws = np.empty((n_boot, self.x.size))
        for b in range(n_boot):
            panel = self._panel.resample(rng)
            lam, _ = _nonparametric_mean(
                panel, self.how, (self.x, self.mcf_hat), _BOOT_TOL
            )
            draws[b] = _step_lookup(panel.grid, lam, self.x, "step")
        if key[1] is not None:
            self._boot[key] = draws
        return draws

    @keeps_query_shape
    def mcf_cb(
        self,
        x: npt.ArrayLike,
        bound: str = "two-sided",
        interp: str = "step",
        *,
        alpha_ci: float = 0.05,
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap bounds on the mean cumulative function at
        ``x``, from refits to resamples of the units. They do not assume
        the Poisson model. A resample that ends before a time leaves it
        out of that time's bound.

        Parameters
        ----------
        x : array_like
            The times.
        bound : str, optional
            ``"two-sided"`` (the default; columns ``[lower, upper]``),
            ``"lower"`` or ``"upper"``.
        interp : str, optional
            ``"step"`` or ``"linear"``, as for :meth:`mcf`.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Defaults to 0.05.
        n_boot : int, optional
            The number of resamples. Defaults to 1000.
        random_state : optional
            The seed of the resampling.

        Returns
        -------
        numpy array
            The bound(s) at each ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import PanelCountMCF
        >>> x = [2, 5, 9, 3, 8, 4, 6, 10, 3, 7]
        >>> n = [1, 0, 2, 1, 2, 0, 2, 1, 0, 1]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
        >>> model = PanelCountMCF.fit(x, n, i=i)
        >>> model.mcf_cb([5], n_boot=200, random_state=1).shape
        (1, 2)
        """
        check_option("bound", bound, BOUNDS)
        check_option("interp", interp, _INTERP)
        check_alpha_ci(alpha_ci)
        draws = self.bootstrap(n_boot, random_state)
        at_grid = _boot_bounds(draws, alpha_ci, bound)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        # The bounds as functions of time, as the estimate is
        vals = at_grid.T if bound == "two-sided" else at_grid
        out = _step_lookup(self.x, vals, q, interp)
        return out.T if bound == "two-sided" else out

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the estimate to a plain, JSON-serialisable dict. The data
        are stored, so the reloaded model gives the estimate and its
        bootstrap bounds exactly as the original.
        """
        out: dict[str, Any] = {"model": "PanelCountMCF", "how": self.how}
        out.update(_data_dict(self._panel, **self._raw))
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> PanelCountMCF:
        """Rebuild the estimate from :meth:`to_dict`."""
        require_model_tag(model_dict, "PanelCountMCF", "a panel-count MCF")
        return cls.fit(
            model_dict["x"],
            model_dict["n"],
            i=model_dict.get("i"),
            how=model_dict["how"],
        )

    def __repr__(self) -> str:
        panel = self._panel
        what = {
            "likelihood": "maximum likelihood (Wellner-Zhang)",
            "pseudo": "maximum pseudo-likelihood (isotonic regression)",
        }[self.how]
        return (
            "Panel-count mean cumulative function\n"
            f"Estimator: {what}\n"
            f"Data: {panel.n_units} units, {panel.x.size} inspections, "
            f"{int(panel.n.sum())} events counted\n"
            f"MCF at last inspection: {float(self.mcf_hat[-1]):.6g} at "
            f"t = {float(self.x[-1]):.6g}"
        )


def _seed_key(random_state: Any) -> Any:
    """A hashable key for a reproducible seed, ``None`` for anything that
    is not one (``None``, a Generator): those are not cached."""
    if isinstance(random_state, (int, np.integer)) and not isinstance(
        random_state, bool
    ):
        return int(random_state)
    return None


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
class PanelCountRegression(SerialisableMixin):
    r"""
    The proportional means model for panel counts: the expected number of
    events by time :math:`t` of a unit with covariates :math:`Z` is a
    baseline mean function scaled by the covariates,

    .. math::
        E[N(t) \mid Z] = \Lambda_0(t)\, e^{\beta' Z},

    with :math:`\Lambda_0` left unspecified (a non-decreasing step function
    at the inspection times). :math:`e^{\beta_k}` is the ratio of the
    expected counts for a one-unit increase in covariate :math:`k`, at
    every time.

    The data are panel counts as for :class:`PanelCountMCF`, with one
    covariate row per unit. The fit maximises over :math:`\beta` the
    profile of one of two likelihoods (Wellner and Zhang, 2007; Zhang,
    2002), both of which treat the counts as Poisson as a working model
    only:

    - ``how="pseudo"`` (the default), the pseudo-likelihood of the
      cumulative counts, whose baseline for a given :math:`\beta` is a
      weighted isotonic regression (``spef``'s ``"MPL"`` method);
    - ``how="likelihood"``, the likelihood of the counts in each
      inspection interval, whose baseline for a given :math:`\beta` is
      found by the self-consistency algorithm.

    Both give consistent, asymptotically normal estimates of
    :math:`\beta` when the counting process is not Poisson, as long as the
    mean model holds and the inspection times do not depend on the
    counts. Their standard errors and bounds (:meth:`param_cb`,
    :meth:`mcf_cb`) come from refitting to resamples of the units (the
    bootstrap), as ``spef`` gives them for these methods.

    Call ``PanelCountRegression.fit`` (or ``fit_from_df``); it returns a
    fitted instance.

    References
    ----------
    Wellner, J. A. and Zhang, Y. (2007), "Two likelihood-based
    semiparametric estimation methods for panel count data with
    covariates", *Annals of Statistics*, 35(5), 2106-2142.

    Zhang, Y. (2002), "A semiparametric pseudolikelihood estimation method
    for panel count data", *Biometrika*, 89(1), 39-48.
    """

    #: The estimator, ``"pseudo"`` or ``"likelihood"``.
    how: str
    #: The coefficients :math:`\beta`.
    params: npt.NDArray
    #: The distinct inspection times.
    x: npt.NDArray
    #: The baseline mean function :math:`\Lambda_0` (at ``Z = 0``) at ``x``.
    baseline_hat: npt.NDArray
    #: Whether the search reached a stationary point of the likelihood.
    converged: bool
    #: The columns of ``Z`` that cannot be estimated (their coefficient is
    #: ``nan``; see the regression models' aliasing).
    aliased: npt.NDArray
    #: The covariates' column names, where ``fit_from_df`` gave them.
    feature_names: list[str] | None = None
    _kept: npt.NDArray
    _panel: _Panel
    _raw: dict
    _boot: dict

    @classmethod
    @takes_unit
    @_refuses_data
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike,
        i: npt.ArrayLike | None = None,
        how: str = "pseudo",
        init: npt.ArrayLike | None = None,
    ) -> PanelCountRegression:
        """
        Fit the proportional means model to panel counts.

        Parameters
        ----------
        x : array_like
            The inspection times, one row per inspection of a unit (all
            positive, from the start of the unit's observation).
        Z : array_like
            The covariates: one row per row of ``x`` (the same on every
            row of a unit), one column per covariate.
        n : array_like
            The number of events found at each inspection since the
            previous one (or since time 0); 0 where none were found.
        i : array_like, optional
            The unit each row belongs to. Defaults to a single unit (which
            cannot estimate a covariate effect: give ``i``).
        how : str, optional
            ``"pseudo"`` (the default) or ``"likelihood"``.
        init : array_like, optional
            A start for the coefficients. Defaults to 0.

        Returns
        -------
        PanelCountRegression
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import PanelCountRegression
        >>> x = [2, 5, 9, 3, 8, 4, 6, 10, 3, 7]
        >>> n = [1, 0, 2, 1, 2, 0, 2, 1, 0, 1]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
        >>> Z = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
        >>> model = PanelCountRegression.fit(x, Z, n, i=i)
        >>> model.params.round(4)
        array([0.1823])
        """
        check_option("how", how, _HOW)
        if Z is None:
            raise ValueError(
                "Z is required: the covariates of each row. Without "
                "covariates use PanelCountMCF."
            )
        panel = _panel_data(x, i, n, Z)
        assert panel.Z is not None
        p = panel.Z.shape[1]
        init_arr = None
        if init is not None:
            init_arr = np.atleast_1d(np.asarray(init, dtype=float))
            if init_arr.shape != (p,):
                raise ValueError(
                    "init must have one value per covariate ({}); got "
                    "{}.".format(p, init_arr.size)
                )
        if panel.total.sum() == 0:
            raise ValueError(
                "No events were counted at any inspection, so the "
                "covariate effects cannot be estimated."
            )
        aliased = _aliased(panel.Z)
        kept = np.setdiff1d(np.arange(p), aliased)
        if kept.size == 0:
            raise ValueError(
                "No covariate in Z can be estimated (each is constant over "
                "the units or a combination of the others). Without "
                "covariates use PanelCountMCF."
            )
        if aliased.size:
            panel.Z = panel.Z[:, kept]
            if init_arr is not None:
                init_arr = init_arr[kept]
        if init_arr is not None:
            init_arr = np.nan_to_num(init_arr)
        out = cls._from_panel(panel, how, init_arr)
        out.aliased = aliased
        out._kept = kept
        out.params = expand(out.params, kept, p)
        out.feature_names = fit_columns()
        out._raw = {"x": x, "i": i, "n": n, "Z": Z}
        if not out.converged:
            warnings.warn(
                "The proportional means fit did not reach a point where "
                "the gradient of the profile {} is zero; the coefficients "
                "returned are the best point found. Check for a covariate "
                "that separates units with and without events, or try "
                "another init.".format(
                    "pseudo-likelihood" if how == "pseudo" else "likelihood"
                ),
                PanelCountConvergenceWarning,
                stacklevel=2,
            )
        return out

    @classmethod
    def _from_panel(
        cls, panel: _Panel, how: str, init: npt.NDArray | None = None
    ) -> PanelCountRegression:
        out = cls()
        out.how = how
        out._panel = panel
        out.x = panel.grid
        out.params, out.baseline_hat, out.converged = _regression_fit(
            panel, how, init
        )
        out._boot = {}
        return out

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        Z_cols: str | list[str],
        n_col: str,
        i_col: str | None = None,
        how: str = "pseudo",
        init: npt.ArrayLike | None = None,
    ) -> PanelCountRegression:
        """
        Fit to panel counts held in the columns of a DataFrame, one row
        per inspection: the columns are the arguments of :meth:`fit`,
        ``Z_cols`` one column name or a list of them.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import PanelCountRegression
        >>> df = pd.DataFrame({
        ...     "t": [2, 5, 9, 3, 8, 4, 6, 10, 3, 7],
        ...     "n": [1, 0, 2, 1, 2, 0, 2, 1, 0, 1],
        ...     "unit": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        ...     "treated": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        ... })
        >>> model = PanelCountRegression.fit_from_df(
        ...     df, "t", "treated", "n", i_col="unit"
        ... )
        >>> model.params.round(4)
        array([0.1823])
        """
        from surpyval.utils.dataframe import frame_column, frame_columns

        names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        with covariate_columns(names):
            return cls.fit(
                frame_column(df, x_col, "x_col", time=True),
                frame_columns(df, Z_cols, "Z_cols").astype(float),
                frame_column(df, n_col, "n_col"),
                i=None if i_col is None else frame_column(df, i_col, "i_col"),
                how=how,
                init=init,
            )

    @property
    def n_covariates(self) -> int:
        return int(self.params.size)

    def _Z(self, Z: npt.ArrayLike) -> npt.NDArray:
        Z_arr = np.atleast_1d(np.asarray(Z, dtype=float))
        if Z_arr.size != self.n_covariates:
            raise ValueError(
                "Z must have one value per covariate ({}); got {}.".format(
                    self.n_covariates, Z_arr.size
                )
            )
        return Z_arr.ravel()

    @keeps_query_shape
    def baseline_mcf(
        self, x: npt.ArrayLike, interp: str = "step"
    ) -> npt.NDArray:
        """
        The baseline mean function :math:`\\Lambda_0(x)`, the expected
        count by ``x`` of a unit whose covariates are all 0; NaN before 0
        and after the last inspection. ``interp`` as for
        :meth:`PanelCountMCF.mcf`.
        """
        check_option("interp", interp, _INTERP)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        return _step_lookup(self.x, self.baseline_hat, q, interp)

    @keeps_query_shape
    def mcf(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, interp: str = "step"
    ) -> npt.NDArray:
        """
        The expected number of events by ``x`` of a unit with covariates
        ``Z`` (one value per covariate): :math:`\\Lambda_0(x) e^{\\beta'
        Z}`.
        """
        check_option("interp", interp, _INTERP)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        # An aliased coefficient is taken as 0
        ratio = np.exp(float(self._Z(Z) @ np.nan_to_num(self.params)))
        return _step_lookup(self.x, self.baseline_hat, q, interp) * ratio

    def bootstrap(
        self, n_boot: int = 1000, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Refits to ``n_boot`` resamples of the units (drawn with
        replacement): the coefficients (one row per resample) and the
        baseline mean function at the inspection times ``x`` (NaN beyond
        a resample's last inspection). A seed caches them.
        """
        _check_boot_args(n_boot)
        key = (int(n_boot), _seed_key(random_state))
        if key[1] is not None and key in self._boot:
            return self._boot[key]
        rng = as_generator(random_state)
        betas = np.full((n_boot, self.n_covariates), np.nan)
        base = np.empty((n_boot, self.x.size))
        kept = self._kept
        failed = 0
        for b in range(n_boot):
            panel = self._panel.resample(rng)
            assert panel.Z is not None
            centred = panel.Z - panel.Z.mean(axis=0)
            if (
                panel.total.sum() == 0
                or np.any(np.all(centred == 0, axis=0))
                or np.linalg.matrix_rank(centred) < panel.Z.shape[1]
            ):
                betas[b], base[b] = np.nan, np.nan
                failed += 1
                continue
            beta, lam0, _ = _regression_fit(
                panel,
                self.how,
                self.params[kept],
                (self.x, self.baseline_hat),
                _BOOT_TOL,
            )
            betas[b, kept] = beta
            base[b] = _step_lookup(panel.grid, lam0, self.x, "step")
        if failed:
            warnings.warn(
                "{} of {} resamples could not be fitted (a covariate took "
                "one value, or no events were counted, in the resample) and "
                "are left out of the bounds; with so few units the "
                "bootstrap bounds are rough.".format(failed, n_boot),
                UserWarning,
                stacklevel=3,
            )
        out = (betas, base)
        if key[1] is not None:
            self._boot[key] = out
        return out

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``: each covariate's column (from
        ``fit_from_df``), else ``coef_0``, ``coef_1``, ..."""
        return coefficient_names(self.n_covariates, self.feature_names)

    def _parameter_bounds(self) -> list[tuple[None, None]]:
        """Every coefficient is unbounded."""
        return [(None, None)] * self.n_covariates

    def standard_errors(
        self, n_boot: int = 1000, random_state: Any = None
    ) -> npt.NDArray:
        """
        The bootstrap standard errors of the coefficients: the standard
        deviation of the refits to ``n_boot`` resamples of the units
        (``nan`` for an aliased coefficient). There is no model-based
        covariance: the Poisson working model's would be wrong for counts
        that are not Poisson. :meth:`bootstrap` gives the refits, for
        their covariance.
        """
        betas, _ = self.bootstrap(n_boot, random_state)
        out = np.full(self.n_covariates, np.nan)
        for k in self._kept:
            col = betas[:, k]
            col = col[np.isfinite(col)]
            if col.size > 1:
                out[k] = float(np.std(col, ddof=1))
        return out

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "percentile",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Bootstrap confidence bound(s) on a coefficient.

        Parameters
        ----------
        name : str
            The coefficient, one of :attr:`parameter_names`.
        alpha_ci : float, optional
            The total tail probability. Defaults to 0.05.
        bound : str, optional
            ``"two-sided"`` (``[lower, upper]``), ``"lower"`` or
            ``"upper"``.
        method : str, optional
            ``"percentile"`` (the default), the quantiles of the refits to
            resamples of the units; or ``"normal"``, the estimate plus or
            minus the normal quantile times the bootstrap standard error.
        n_boot, random_state : optional
            The number of resamples (1000) and their seed.

        Returns
        -------
        numpy array
            The bound(s): ``nan`` for an aliased coefficient.

        Examples
        --------
        >>> from surpyval.recurrent import PanelCountRegression
        >>> x = [2, 5, 9, 3, 8, 4, 6, 10, 3, 7, 6, 9]
        >>> n = [1, 0, 2, 1, 2, 0, 2, 1, 0, 1, 2, 1]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
        >>> Z = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1]
        >>> model = PanelCountRegression.fit(x, Z, n, i=i)
        >>> model.param_cb("coef_0", n_boot=200, random_state=1).shape
        (2,)
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        check_option("method", method, ("percentile", "normal"))
        check_option("name", name, tuple(self.parameter_names))
        k = self.parameter_names.index(name)
        betas, _ = self.bootstrap(n_boot, random_state)
        col = betas[:, k]
        col = col[np.isfinite(col)]
        if col.size < 2:
            return (
                np.full(2, np.nan)
                if bound == "two-sided"
                else np.asarray(np.nan)
            )
        if method == "percentile":
            return np.asarray(percentile_bounds(col, alpha_ci, bound))
        from scipy.stats import norm

        se = float(np.std(col, ddof=1))
        est = float(self.params[k])
        if bound == "two-sided":
            z = norm.ppf(1 - alpha_ci / 2)
            return np.array([est - z * se, est + z * se])
        z = norm.ppf(1 - alpha_ci)
        return np.asarray(est - z * se if bound == "lower" else est + z * se)

    @keeps_query_shape
    def mcf_cb(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        bound: str = "two-sided",
        interp: str = "step",
        *,
        alpha_ci: float = 0.05,
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap bounds on the expected count by ``x`` of a
        unit with covariates ``Z``, from refits to resamples of the units
        (each refit's baseline and coefficients together). Arguments as
        for :meth:`PanelCountMCF.mcf_cb`.
        """
        check_option("bound", bound, BOUNDS)
        check_option("interp", interp, _INTERP)
        check_alpha_ci(alpha_ci)
        z = self._Z(Z)
        betas, base = self.bootstrap(n_boot, random_state)
        # An aliased coefficient is taken as 0
        draws = base * np.exp(np.nan_to_num(betas) @ z)[:, None]
        at_grid = _boot_bounds(draws, alpha_ci, bound)
        q = np.atleast_1d(np.asarray(x, dtype=float))
        vals = at_grid.T if bound == "two-sided" else at_grid
        out = _step_lookup(self.x, vals, q, interp)
        return out.T if bound == "two-sided" else out

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict. The data
        are stored (with the fitted coefficients), so the reloaded model
        gives every estimate and bootstrap bound as the original.
        """
        raw = self._raw
        out: dict[str, Any] = {
            "model": "PanelCountRegression",
            "how": self.how,
        }
        out.update(_data_dict(self._panel, raw["x"], raw["i"], raw["n"]))
        Z = np.asarray(raw["Z"], dtype=float)
        out["Z"] = Z.reshape(len(out["x"]), -1).tolist()
        # An aliased coefficient (nan) is stored as null
        out["params"] = [
            None if not np.isfinite(b) else float(b) for b in self.params
        ]
        if self.feature_names is not None:
            out["feature_names"] = [str(v) for v in self.feature_names]
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> PanelCountRegression:
        """Rebuild the model from :meth:`to_dict`, refitting from the
        stored coefficients."""
        require_model_tag(
            model_dict, "PanelCountRegression", "a panel-count regression"
        )
        init = model_dict.get("params")
        if init is not None:
            init = [np.nan if b is None else b for b in init]
        names = model_dict.get("feature_names")
        with (
            ignoring_warnings(PanelCountConvergenceWarning),
            covariate_columns(names),
        ):
            out = cls.fit(
                model_dict["x"],
                model_dict["Z"],
                model_dict["n"],
                i=model_dict.get("i"),
                how=model_dict["how"],
                init=init,
            )
        return out

    def __repr__(self) -> str:
        panel = self._panel
        what = {
            "likelihood": "maximum likelihood",
            "pseudo": "maximum pseudo-likelihood",
        }[self.how]
        coefs = ", ".join(f"{b:.6g}" for b in self.params)
        return (
            "Panel-count proportional means model\n"
            f"Estimator: {what} (Wellner-Zhang)\n"
            f"Data: {panel.n_units} units, {panel.x.size} inspections, "
            f"{int(panel.n.sum())} events counted\n"
            f"Coefficients (beta): [{coefs}]"
        )
