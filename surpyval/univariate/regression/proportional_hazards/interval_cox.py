"""
The semi-parametric proportional hazards (Cox) model for interval-censored
data (Finkelstein 1986; Huang 1996; Pan 1999; Anderson-Bergman 2017).

When the event is only known to have happened between two inspections,
the Cox partial likelihood cannot be formed: nobody knows who was at risk
when the event happened. This model keeps Cox's assumption -- the
covariates multiply the hazard, and the baseline is left to the data --
and fits the full likelihood instead:

.. math::
    S(x \\mid Z) = S_0(x)^{\\exp(\\beta' Z)},

where :math:`S_0` is a non-parametric maximum-likelihood estimate (NPMLE)
with mass only on the Turnbull intervals of the pooled data, as the
Turnbull estimator's. Every kind of row is used as it was observed: an
exact time ``x`` (``c = 0``) contributes the drop of the survival at
``x``, an interval ``(l, r]`` (``c = 2``) the probability
:math:`S(l \\mid Z) - S(r \\mid Z)`, a left-censored row (``c = -1``)
:math:`1 - S(r \\mid Z)`, and a right-censored row (``c = 1``)
:math:`S(l \\mid Z)`; with an entry time ``tl``, each is divided by
:math:`S(t_l \\mid Z)`.

Estimation
----------
The coefficients and the baseline are maximised together (the baseline
does not cancel as it does in Cox's partial likelihood), by a projected
Newton method on the coefficients and the baseline's cumulative hazard
increments over the Turnbull intervals, held non-negative (see
``_interval_npmle``). It is the same maximum as R's
``icenReg::ic_sp(model = "ph")``, which alternates iterative convex
minorant steps on the baseline with Newton steps on the coefficients.

The standard errors are the profile likelihood's (Murphy and van der
Vaart 2000): the curvature of :math:`p\\ell(\\beta) = \\max_{S_0}
\\ell(\\beta, S_0)`, the likelihood maximised over the baseline, by
second differences with a step of :math:`1/\\sqrt{N}` standard
deviations of each covariate. With interval-censored data the baseline's
NPMLE converges more slowly than :math:`\\sqrt{N}`, so the observed
information of the baseline's masses is not a valid variance; the profile
likelihood's curvature at a step of this size is (Murphy and van der
Vaart 1999, 2000). The case bootstrap, which ``ic_sp`` uses, is offered
too, as are profile likelihood-ratio intervals.

References
----------
Finkelstein, D. M. (1986), "A proportional hazards model for
interval-censored failure time data", Biometrics 42, 845-854.

Huang, J. (1996), "Efficient estimation for the proportional hazards
model with interval censoring", Annals of Statistics 24, 540-568.

Pan, W. (1999), "Extending the iterative convex minorant algorithm to the
Cox model for interval-censored data", Journal of Computational and
Graphical Statistics 8, 109-120.

Murphy, S. A. and van der Vaart, A. W. (2000), "On profile likelihood",
Journal of the American Statistical Association 95, 449-465.

Anderson-Bergman, C. (2017), "icenReg: regression models for interval
censored data in R", Journal of Statistical Software 81(12).
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.diagnostics.mixin import DiagnoseMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.univariate.nonparametric._turnbull_npmle import (
    DOES_NOT_EXIST,
    NOT_UNIQUE,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils import xcnt_handler
from surpyval.utils.covariates import (
    check_covariate_rows,
    finite_covariate_mask,
)
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import percentile_bounds, wald_bound_on_support
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_bootstrap_refits,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
)

from .._aliasing import aliased_columns, constant_columns, expand, warn_aliased
from .._bootstrap import bound_method, check_n_boot
from .._concordance import ConcordanceMixin
from .._fit_skeleton import LOG_MAX, baseline_at_origin_error, covariate_center
from .._prediction import (
    ConditionalSurvivalMixin,
    paired_probabilities,
    step_quantiles,
    unique_rows,
)
from .._summary import coefficient_names, coefficient_repr, coefficient_table
from ..kinds import PROPORTIONAL_HAZARD
from ..life_at import (
    AtCovariatesMixin,
    checked_params,
    draws_by_quantile,
    rebuilt,
)
from ..regression_data import (
    FrameDesign,
    LinearPredictorMixin,
    check_finite_event_times,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from ._interval_npmle import ICLikelihood, find_atoms, maximise, profile

if TYPE_CHECKING:
    import pandas as pd

#: The profile likelihood's second differences step ``PROFILE_STEP /
#: sqrt(N)`` standard deviations of each covariate.
PROFILE_STEP = 1.0
#: The covariance methods.
COVARIANCE_METHODS = ("profile", "bootstrap")


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
class _Data:
    """The rows as the fit uses them: ``xl``, ``xr`` (``-inf`` for a
    left-censored row's lower end, ``inf`` for a right-censored row's
    upper end, equal for an exact time), ``c``, ``n``, ``tl`` (``-inf``
    for none) and ``Z``, sorted by every column."""

    def __init__(
        self,
        xl: npt.NDArray,
        xr: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
        Z: npt.NDArray,
    ) -> None:
        order = np.lexsort([n, *Z.T[::-1], tl, c, xr, xl])
        self.xl, self.xr, self.c, self.n, self.tl, self.Z = (
            a[order] for a in (xl, xr, c, n, tl, Z)
        )

    def subset(self, counts: npt.NDArray) -> _Data:
        keep = counts > 0
        return _Data(
            self.xl[keep],
            self.xr[keep],
            self.c[keep],
            counts[keep].astype(float),
            self.tl[keep],
            self.Z[keep],
        )

    def x(self) -> npt.NDArray:
        """The rows as SurPyval's two-column ``x``."""
        xl = np.where(self.c == -1, self.xr, self.xl)
        xr = np.where(self.c == 1, self.xl, self.xr)
        return np.column_stack([xl, xr])


_CENSORING = (
    "'c' must be 0 (an exact time), 1 (right censored), -1 (left "
    "censored) or 2 (interval censored, with x given as [left, right] "
    "pairs)."
)


def _validate(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
) -> _Data:
    if tl is not None and np.ndim(tl) == 2:
        raise ValueError(
            "IntervalCoxPH supports left truncation (delayed entry) only, "
            "given as a one-dimensional `tl`; right or interval truncation "
            "is not available. Use a parametric proportional hazards model "
            "(e.g. WeibullPH.fit(x, Z, c=c, t=t)) for such data."
        )
    x_h, c_h, n_h, t_h = xcnt_handler(x, c, n, tl=tl, group_and_sort=False)
    c_arr = np.asarray(c_h, dtype=float)
    x_arr = np.asarray(x_h, dtype=float)
    if x_arr.ndim == 1:
        x_arr = np.column_stack([x_arr, x_arr])
    n_arr = np.asarray(n_h, dtype=float)
    tl_arr = np.asarray(t_h, dtype=float)[:, 0]
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    elif Z_arr.ndim != 2:
        raise ValueError("Covariate matrix must be two dimensional")
    check_covariate_rows(Z_arr, x_arr.shape[0])
    mask = finite_covariate_mask(Z_arr)
    x_arr, c_arr, n_arr, tl_arr, Z_arr = (
        a[mask] for a in (x_arr, c_arr, n_arr, tl_arr, Z_arr)
    )
    check_finite_event_times(x_arr, c_arr)
    keep = n_arr > 0
    x_arr, c_arr, n_arr, tl_arr, Z_arr = (
        a[keep] for a in (x_arr, c_arr, n_arr, tl_arr, Z_arr)
    )
    if not np.any(c_arr != 1):
        raise ValueError(
            "IntervalCoxPH needs at least one row whose event is observed "
            "or bracketed (c = 0, -1 or 2); with every row right censored "
            "there is no baseline to estimate."
        )
    xl = np.where(c_arr == -1, -np.inf, x_arr[:, 0])
    xr = np.where(c_arr == 1, np.inf, x_arr[:, 1])
    xr = np.where(c_arr == 0, xl, xr)
    tl_arr = np.where(np.isfinite(tl_arr), tl_arr, -np.inf)
    return _Data(xl, xr, c_arr, n_arr, tl_arr, Z_arr)


def _aliased(data: _Data) -> tuple[npt.NDArray, npt.NDArray]:
    """The columns the likelihood cannot determine (constant, which the
    baseline absorbs, or a combination of the others), and each column's
    scale for the no-maximum check: the count of informative rows times
    the column's variance."""
    Z, n = data.Z, data.n
    Zc = Z - covariate_center(Z, n)
    gram = (Zc * n[:, None]).T @ Zc
    aliased = aliased_columns(gram, Z.shape[0], constant_columns(Z))
    informative = float(np.sum(n[data.c != 1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = informative * np.diag(gram) / float(np.sum(n))
    return aliased, scale


class _Fit:
    """One maximisation of the likelihood on ``data``: the atoms, the
    likelihood on the centred kept columns, and its maximum."""

    def __init__(self, data: _Data, kept: npt.NDArray, tol: float):
        self.data = data
        self.kept = kept
        self.mean = covariate_center(data.Z, data.n)
        self.atoms = find_atoms(data.xl, data.xr, data.c, data.tl, data.n)
        Zc = (data.Z - self.mean)[:, kept]
        self.lik = ICLikelihood(self.atoms, Zc, data.n)
        zero = np.zeros(kept.size)
        with np.errstate(all="ignore"):
            start = profile(self.lik, zero, self.lik.start())
            self.solution = (
                maximise(self.lik, zero, start.d, tol=tol)
                if kept.size
                else start
            )

    def profile_value(self, beta: npt.NDArray) -> float:
        with np.errstate(all="ignore"):
            return profile(self.lik, beta, self.solution.d).value

    def schur(self) -> npt.NDArray:
        """The information of the coefficients with the baseline's
        positive increments profiled out exactly (the Schur complement),
        the scale of the no-maximum check."""
        der = self.solution.der
        free = self.solution.d > 0
        Hdd = der.H_dd[np.ix_(free, free)]
        Hbd = der.H_bd[:, free]
        try:
            inner = np.linalg.solve(Hdd, Hbd.T) if free.any() else 0.0
        except np.linalg.LinAlgError:
            return np.full(der.H_bb.shape, np.nan)
        return -(der.H_bb - Hbd @ inner)

    def profile_information(self) -> npt.NDArray:
        """The negative Hessian of the profile log-likelihood at the
        estimate, by second differences of ``PROFILE_STEP / sqrt(N)``
        standard deviations of each covariate."""
        data, kept = self.data, self.kept
        k = kept.size
        beta = self.solution.beta
        N = float(np.sum(data.n))
        Zc = (data.Z - self.mean)[:, kept]
        sd = np.sqrt(np.dot(data.n, Zc**2) / N)
        h = PROFILE_STEP / np.sqrt(N) / np.where(sd > 0, sd, 1.0)
        f0 = self.solution.value
        info = np.zeros((k, k))
        eye = np.eye(k)
        for i in range(k):
            up = self.profile_value(beta + h[i] * eye[i])
            down = self.profile_value(beta - h[i] * eye[i])
            info[i, i] = -(up - 2.0 * f0 + down) / h[i] ** 2
            for j in range(i):
                ei, ej = h[i] * eye[i], h[j] * eye[j]
                pp = self.profile_value(beta + ei + ej)
                pm = self.profile_value(beta + ei - ej)
                mp = self.profile_value(beta - ei + ej)
                mm = self.profile_value(beta - ei - ej)
                info[i, j] = info[j, i] = -(pp - pm - mp + mm) / (
                    4.0 * h[i] * h[j]
                )
        return info


def _inverse(info: npt.NDArray) -> npt.NDArray:
    """The covariance from an information matrix, ``nan`` rows and
    columns where a variance is not positive."""
    k = info.shape[0]
    if k == 0:
        return np.zeros((0, 0))
    if not np.all(np.isfinite(info)):
        return np.full((k, k), np.nan)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    bad = ~(np.diag(cov) > 0)
    cov[bad, :] = np.nan
    cov[:, bad] = np.nan
    return cov


def _check_maximum(
    fit: _Fit, scale: npt.NDArray, kept: npt.NDArray, warn: bool = True
) -> str:
    """The model's ``maximum`` and, with ``warn``, one warning when it is
    not ``"verified"``. A coefficient whose information (the baseline
    profiled out) has fallen below ``1e-4`` of ``scale`` is taken to have
    run away, as in ``ProportionalOdds``: the likelihood keeps rising as
    it grows. The threshold is looser than ``ProportionalOdds``'s
    ``1e-8``: with interval-censored rows the likelihood flattens slowly
    as a separating coefficient grows, and the iteration stops (or runs
    out of steps) with the information still between ``1e-7`` and
    ``1e-5`` of ``scale``, where an ordinary fit's is a sizeable fraction
    of it."""
    sol = fit.solution
    if kept.size:
        info = np.diag(np.atleast_2d(fit.schur()))
        d0 = np.asarray(scale, dtype=float)[kept]
        collapsed = np.flatnonzero(
            (d0 > 0) & ~(np.nan_to_num(info, nan=0.0) > 1e-4 * d0)
        )
        if collapsed.size:
            if warn:
                warn_no_maximum(
                    "the likelihood keeps increasing as coefficient(s) {} "
                    "of Z grow without bound, so the estimate is infinite "
                    "(the covariate separates the early events from the "
                    "late ones, as a level with no events does)".format(
                        kept[collapsed].tolist()
                    ),
                    "The reported value, its standard error and its "
                    "p-value are meaningless",
                    "consider removing or coarsening the covariate",
                )
            return "no finite maximum"
    if not sol.converged:
        if warn:
            warn_unverified(
                "The IntervalCoxPH fit",
                "the projected Newton iteration stopped after {} step(s) "
                "without reaching its tolerance".format(sol.n_iter),
                "check the covariates for extreme values, or rescale them",
            )
        return "unverified"
    return "verified"


# ---------------------------------------------------------------------------
# The fitted model
# ---------------------------------------------------------------------------
class IntervalCoxPHModel(
    AtCovariatesMixin,
    ConditionalSurvivalMixin,
    InformationCriteriaMixin,
    LinearPredictorMixin,
    ConcordanceMixin,
    DiagnoseMixin,
    SerialisableMixin,
):
    """
    A fitted semi-parametric proportional hazards model for
    interval-censored data, returned by :meth:`IntervalCoxPH.fit
    <IntervalCoxPH_.fit>` and ``fit_from_df``.

    The covariates multiply the hazard of a baseline left to the data:

    .. math::
        S(x \\mid Z) = S_0(x)^{\\exp(\\beta' (Z - \\text{center}))},

    so ``exp(beta)`` are hazard ratios, as in ``CoxPH``. The baseline is
    a step function that can only drop within the Turnbull intervals
    (``intervals``): ``x`` are the times at which it is known, the right
    ends of those intervals, and ``H0`` its cumulative hazard there
    (``inf`` once all the mass is spent). Within a Turnbull interval the
    data do not say where the drop happens; the survival is held at its
    value at the interval's start until its end, as the Turnbull
    estimator's is. Before the first time the survival is 1, and after the
    last it holds its last value.

    Examples
    --------
    Times to the first sign of an illness, known only to lie between two
    visits (``c = 2``), or to come before the first visit (``c = -1``) or
    after the last (``c = 1``), for two groups:

    >>> import numpy as np
    >>> from surpyval import IntervalCoxPH
    >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [6, 9], [5, 5],
    ...      [7, 7], [1, 4], [2, 6], [5, 9], [8, 12], [9, 9], [10, 10]]
    >>> c = [2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1]
    >>> group = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    >>> model = IntervalCoxPH.fit(x, group, c=c)
    >>> model.beta.round(4)
    array([-1.6463])
    >>> model.sf([3, 6], [[0], [1]]).round(4)
    array([0.6467, 0.7067])
    """

    feature_names: list[str] | None = None
    _CHECKS_WIDTH = True
    formula: str | None = None
    _model_spec: Any = None

    beta: npt.NDArray
    params: npt.NDArray
    _se: npt.NDArray
    _covariance: npt.NDArray | None
    p_values: npt.NDArray
    #: The times at which the baseline is known (the right ends of the
    #: Turnbull intervals) and its cumulative hazard there.
    x: npt.NDArray
    H0: npt.NDArray
    #: The Turnbull intervals ``(left, right]`` the baseline's mass is on
    #: (``left == right`` for an exact time), one row each, and the
    #: baseline's probability mass on each.
    intervals: npt.NDArray
    p0: npt.NDArray
    center: npt.NDArray | None = None
    n_iter: int = 0
    maximum: str = "unknown"
    _fit_data: Any = None
    _data_summary: str | None = None
    _lr_search: Any = None
    _boot: Any = None
    kind: str
    parameterization: str

    def __init__(self) -> None:
        self.kind = PROPORTIONAL_HAZARD
        self.parameterization = "Semi-Parametric"

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: each covariate's
        column (a formula, ``fit_from_df`` or a DataFrame ``Z``), else
        ``coef_0``, ``coef_1``, ..."""
        return coefficient_names(self, len(self.params))

    def phi(self, Z: npt.ArrayLike | pd.DataFrame) -> npt.NDArray:
        """
        The hazard ratio :math:`e^{\\beta' (Z - \\text{center})}` of
        covariates ``Z`` against a unit at ``center``.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.phi([[1], [0]]).round(4)
        array([0.2654, 1.    ])
        """
        with np.errstate(over="ignore"):
            return np.exp(self._log_phi(Z))

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        return self._log_phi(Z_arr.reshape(x.size, -1))

    def _concordance_data(self) -> tuple | None:
        # Concordance needs one time per row: not defined for intervals.
        data = self._fit_data
        if data is None or np.any(np.isin(data.c, (-1, 2))):
            return None
        return data.xl, data.c, data.n, data.Z

    # -- the baseline at the query times -------------------------------

    def _parts(
        self, x: npt.ArrayLike, Z: npt.ArrayLike | pd.DataFrame, grid: bool
    ) -> tuple:
        """``(H, H_prev, log_r)``: the baseline cumulative hazard at the
        last time at or before ``x`` and at the one before, and the log
        hazard ratio; paired, or on the grid."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        H0 = np.asarray(self.H0, dtype=float)
        idx = np.searchsorted(self.x, x, side="right") - 1
        top = self.x.size - 1
        H = np.where(idx < 0, 0.0, H0[np.clip(idx, 0, top)])
        Hp = np.where(idx < 1, 0.0, H0[np.clip(idx - 1, 0, top)])
        missing = np.isnan(x)
        H = np.where(missing, np.nan, H)
        Hp = np.where(missing, np.nan, Hp)
        if grid:
            rows = covariate_rows(
                self._prepare_Z(Z), np.asarray(self.beta).shape[0]
            )
            log_r = self._log_risk(rows)[:, None]
            return H[None, :], Hp[None, :], log_r
        log_r = self._log_phi(Z)
        check_paired_rows(x.size, np.size(log_r))
        return H, Hp, log_r

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Cumulative hazard :math:`H_0(x)\\, e^{\\beta' (Z - \\text{center})}`
        at ``x`` for covariates ``Z`` (one row for every ``x``, or one row
        per ``x``, paired in the order given; a DataFrame for a model
        fitted with ``fit_from_df``). 0 before the first time, held after
        the last (``inf`` once the baseline has spent all its mass). With
        ``grid=True`` every time is evaluated for every row of ``Z``,
        giving shape ``(len(Z),) + x.shape``.
        """
        H, _, log_r = self._parts(x, Z, grid)
        return self._times_risk(H, log_r)

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Survival :math:`S_0(x)^{\\exp(\\beta' (Z - \\text{center}))}` at
        ``x`` for covariates ``Z``; arguments as for :meth:`Hf`.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.sf([2, 5, 8], [[0], [1]], grid=True).round(3)
        array([[0.603, 0.027, 0.   ],
               [0.874, 0.384, 0.   ]])
        """
        H, _, log_r = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):
            return np.exp(-self._times_risk(H, log_r))

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """Failure probability ``1 - sf`` at ``x`` for covariates ``Z``;
        arguments as for :meth:`Hf`."""
        H, _, log_r = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):
            return -np.expm1(-self._times_risk(H, log_r))

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The jump of the cumulative hazard :meth:`Hf` at the last baseline
        time at or before ``x`` (0 before the first time): a step size,
        not a hazard rate, as ``CoxPH``'s ``hf`` is. Arguments as for
        :meth:`Hf`.
        """
        H, Hp, log_r = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):
            return self._times_risk(H, log_r) - self._times_risk(Hp, log_r)

    @keeps_query_shape
    def df(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The probability mass at the last baseline time at or before ``x``,
        ``ff`` there less ``ff`` at the time before: the baseline is a
        step function, so this is a jump, not a density. Arguments as for
        :meth:`Hf`.
        """
        H, Hp, log_r = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):
            return np.exp(-self._times_risk(Hp, log_r)) - np.exp(
                -self._times_risk(H, log_r)
            )

    @keeps_query_shape
    def qf(
        self,
        p: npt.ArrayLike,
        Z: npt.ArrayLike | pd.DataFrame,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The quantile function: the first baseline time at which the
        predicted failure probability ``ff(x, Z)`` reaches ``p``, ``nan``
        where it never does (the data end with survival above ``1 - p``).
        ``Z`` is paired with ``p`` as :meth:`sf` pairs it with ``x``;
        ``grid=True`` gives every ``p`` for every row. A probability
        outside [0, 1] gives ``nan``, with a warning.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.qf(0.5, [[0], [1]])
        array([5., 5.])
        """
        rows = covariate_rows(
            self._prepare_Z(Z), np.asarray(self.beta).shape[0]
        )
        u, rows, shape = paired_probabilities(p, rows, grid)
        uniq, which = unique_rows(rows)
        F = np.atleast_2d(self.ff(self.x, uniq, grid=True))
        out = step_quantiles(F[which], self.x, u)
        return out if shape is None else out.reshape(shape)

    def baseline_sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The baseline survival :math:`S_0(x)` (at ``Z = center``) at
        ``x``.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.baseline_sf([1, 4, 6]).round(4)
        array([1.    , 0.5264, 0.0271])
        """
        x_arr = np.asarray(x, dtype=float)
        H0 = np.asarray(self.H0, dtype=float)
        idx = np.searchsorted(self.x, x_arr, side="right") - 1
        H = np.where(idx < 0, 0.0, H0[np.clip(idx, 0, self.x.size - 1)])
        return np.where(np.isnan(x_arr), np.nan, np.exp(-H))

    # -- model comparison --------------------------------------------------

    def _ic_k(self) -> int:
        return int(np.isfinite(np.asarray(self.params, dtype=float)).sum())

    def _ic_sample_size_from_data(self) -> float:
        if self._fit_data is not None:
            return ic_sample_size(self._fit_data.c, self._fit_data.n)
        return float(np.nan)

    # -- inference ---------------------------------------------------------

    def covariance(
        self,
        method: str = "profile",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        The covariance of the coefficients (``nan`` rows and columns for
        an aliased coefficient).

        Parameters
        ----------
        method : {'profile', 'bootstrap'}, optional
            ``"profile"`` (the default): the inverse of the profile
            likelihood's curvature at the estimate (see the module notes).
            ``"bootstrap"``: the covariance of ``n_boot`` refits to the
            rows resampled with replacement (the case bootstrap), as
            R's ``icenReg::ic_sp`` gives it. It needs the data the model
            was fitted to.
        n_boot : int, optional
            The number of bootstrap refits. Default 200.
        random_state : int or numpy.random.Generator, optional
            Seeds the bootstrap; ``None`` draws from numpy's global
            generator.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.covariance().round(4)
        array([[0.9779]])
        """
        check_option("method", method, COVARIANCE_METHODS)
        if method == "bootstrap":
            draws = self._bootstrap(n_boot, random_state)
            p = draws.shape[1]
            cov = np.full((p, p), np.nan)
            kept = np.flatnonzero(~np.isnan(np.asarray(self.beta, float)))
            sub = draws[:, kept]
            sub = sub[np.all(np.isfinite(sub), axis=1)]
            if sub.shape[0] > 1:
                cov[np.ix_(kept, kept)] = np.atleast_2d(
                    np.cov(sub, rowvar=False)
                )
            return cov
        return self._require_covariance()

    def standard_errors(
        self,
        method: str = "profile",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """The coefficients' standard errors, the square roots of the
        diagonal of :meth:`covariance` (same arguments)."""
        if method == "profile":
            self._require_covariance()
            return self._se
        with np.errstate(invalid="ignore"):
            return np.sqrt(
                np.diag(self.covariance(method, n_boot, random_state))
            )

    def _require_covariance(self) -> npt.NDArray:
        if self._covariance is None:
            raise no_covariance_error(
                "it was built from coefficients with with_params; use the "
                "fitted model's"
            )
        return self._covariance

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows (and columns) of :meth:`covariance`."""
        return list(self.parameter_names)

    def _estimates(self) -> npt.NDArray:
        return np.array(self.params, dtype=float)

    def _need_data(self, what: str) -> _Data:
        if self._fit_data is None:
            raise ValueError(
                "{} needs the data the model was fitted to, which a model "
                "restored from a dict, or built with with_params, does not "
                "keep. Call it on the fitted model, or refit.".format(what)
            )
        return self._fit_data

    def _bootstrap(self, n_boot: Any, random_state: Any) -> npt.NDArray:
        """The coefficients of ``n_boot`` refits to the units resampled
        with replacement (one row per refit, ``nan`` for a refit that
        failed), kept for a repeat call with the same integer seed."""
        n_boot = check_n_boot(n_boot)
        data = self._need_data("The bootstrap")
        key = None
        if isinstance(random_state, (int, np.integer)):
            key = (n_boot, int(random_state))
            if self._boot is not None and self._boot[0] == key:
                return self._boot[1]
        counts = np.asarray(data.n, dtype=float)
        if not np.allclose(counts, np.round(counts)):
            raise ValueError(
                "The bootstrap resamples units, so the counts 'n' must be "
                "whole numbers; use method='profile' for fractional "
                "counts."
            )
        rng = as_generator(random_state)
        units = int(np.round(counts.sum()))
        prob = counts / counts.sum()
        p = np.asarray(self.beta).size
        out = np.full((n_boot, p), np.nan)
        failed = no_max = unverified = 0
        for b in range(n_boot):
            draw = rng.multinomial(units, prob)
            sample = data.subset(draw)
            try:
                with np.errstate(all="ignore"):
                    fit = _Fit(sample, np.arange(p), 1e-10)
                aliased, scale = _aliased(sample)
                if aliased.size:
                    failed += 1
                    continue
                state = _check_maximum(fit, scale, np.arange(p), warn=False)
            except (ValueError, np.linalg.LinAlgError, ArithmeticError):
                failed += 1
                continue
            if state == "no finite maximum":
                no_max += 1
                continue
            if state == "unverified":
                unverified += 1
            out[b] = fit.solution.beta
        warn_bootstrap_refits(
            n_boot,
            no_estimate=no_max,
            unverified=unverified,
            failed=failed,
            advice="treat the bootstrap as rough, or use method='profile'",
        )
        if key is not None:
            self._boot = (key, out)
        return out

    def _profile_baseline(self, beta: npt.NDArray) -> npt.NDArray:
        """The baseline cumulative hazard ``H0`` on ``x`` that maximises
        the likelihood at the coefficients ``beta`` (the profile
        likelihood's baseline), at the model's centre."""
        data = self._need_data("with_params")
        kept = np.flatnonzero(~np.isnan(np.asarray(self.beta, dtype=float)))
        mean = covariate_center(data.Z, data.n)
        atoms = find_atoms(data.xl, data.xr, data.c, data.tl, data.n)
        lik = ICLikelihood(atoms, (data.Z - mean)[:, kept], data.n)
        with np.errstate(all="ignore"):
            sol = profile(lik, beta[kept], lik.start())
        full = np.zeros(np.asarray(self.beta).size)
        full[kept] = beta[kept]
        _, H0, _, _ = _baseline(
            atoms,
            sol.d,
            full,
            mean,
            data.Z,
            at_center=bool(np.any(self.center)),
        )
        return H0

    _FIT_STATE = (
        "_covariance",
        "_se",
        "p_values",
        "_fit_data",
        "_lr_search",
        "_boot",
        "_neg_ll",
        "_aic",
        "_bic",
        "_aic_c",
        "_ic_n",
        "_data_summary",
        "n_iter",
        "maximum",
    )

    def with_params(self, params: npt.ArrayLike) -> IntervalCoxPHModel:
        """
        The same model with other coefficients, and the baseline the
        profile likelihood gives at them (re-estimated from the data the
        model was fitted to; at the fitted coefficients, the fitted
        baseline itself).

        Parameters
        ----------
        params : array like
            The coefficients, in the order of :attr:`covariance_names`.

        Returns
        -------
        IntervalCoxPHModel
            A model without the data, covariance or likelihood.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> same = model.with_params(model._estimates())
        >>> bool(same.sf(4, [1]) == model.sf(4, [1]))
        True
        """
        values = checked_params(params, self.covariance_names)
        fitted = np.asarray(self.beta, dtype=float)
        values[np.isnan(fitted)] = np.nan
        if np.array_equal(values, fitted, equal_nan=True):
            if self._fit_data is None:
                self._need_data("with_params")
            H0 = np.array(self.H0, dtype=float)
        else:
            H0 = self._profile_baseline(np.nan_to_num(values, nan=0.0))
        out = rebuilt(self, self._FIT_STATE)
        out.beta = values
        out.params = values.copy()
        out.H0 = H0
        out._covariance = None
        out._se = np.full(values.shape, np.nan)
        out.p_values = np.full(values.shape, np.nan)
        out.maximum = "not applicable"
        return out

    def random(
        self,
        size: int,
        Z: npt.ArrayLike | pd.DataFrame,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` lifetimes for each covariate row of ``Z``, by
        inverting :meth:`qf` on the step survival curve; a draw beyond
        the curve's end is ``inf``. ``random_state`` seeds the draw
        (``None`` draws from numpy's global generator). Returns the draws
        and a 2-D array of the covariate row each was drawn at.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> draws, rows = model.random(4, [[0]], random_state=1)
        >>> draws
        array([5., 2., 5., 2.])
        """
        rows = covariate_rows(
            self._prepare_Z(Z), np.asarray(self.beta).shape[0]
        )
        return draws_by_quantile(self, size, rows, random_state)

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Confidence bound(s) on a coefficient.

        - ``"wald"`` (the default): from the profile likelihood's
          curvature (:meth:`covariance`).
        - ``"lr"``: the profile-likelihood interval, the values whose
          deviance of :math:`p\\ell(\\beta)`, the other coefficients and
          the baseline re-fitted, stays below the :math:`\\chi^2_1`
          critical value (aliases ``"likelihood"``,
          ``"likelihood-ratio"``, ``"profile"``); asymptotically
          :math:`\\chi^2` here (Murphy and van der Vaart 2000).
        - ``"bootstrap"``: the percentile interval of ``n_boot`` refits
          to the units resampled with replacement, seeded by
          ``random_state``.

        The likelihood-ratio and bootstrap bounds need the data the model
        was fitted to, which a model restored from a dict does not keep.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.param_cb("coef_0").round(4)
        array([-3.2649,  0.6115])
        >>> model.param_cb("coef_0", method="lr").round(4)
        array([-4.1324,  0.4333])
        """
        check_alpha_ci(alpha_ci)
        from .._likelihood_ratio import profile_interval

        names = self.parameter_names
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        check_option("bound", bound, BOUNDS)
        method = bound_method(method)
        if method == "lr":
            return profile_interval(self._lr_region(), name, alpha_ci, bound)
        idx = names.index(name)
        if method == "bootstrap":
            draws = self._bootstrap(n_boot, random_state)[:, idx]
            draws = draws[np.isfinite(draws)]
            if draws.size < 2:
                return np.full(2 if bound == "two-sided" else (), np.nan)
            return np.asarray(percentile_bounds(draws, alpha_ci, bound))
        return wald_bound_on_support(
            float(self.params[idx]),
            float(self.covariance()[idx, idx]),
            None,
            None,
            alpha_ci,
            bound,
            name=name,
        )

    def _lr_region(self) -> Any:
        from .._likelihood_ratio import LikelihoodRegion

        data = self._need_data("Likelihood-ratio bounds")
        params = np.nan_to_num(np.asarray(self.params, dtype=float), nan=0.0)
        point = params.tobytes()
        search = self._lr_search
        if search is not None and search.point == point:
            return search
        aliased = np.flatnonzero(np.isnan(np.asarray(self.beta, dtype=float)))
        kept = np.setdiff1d(np.arange(params.size), aliased)
        mean = covariate_center(data.Z, data.n)
        atoms = find_atoms(data.xl, data.xr, data.c, data.tl, data.n)
        lik = ICLikelihood(atoms, (data.Z - mean)[:, kept], data.n)
        with np.errstate(all="ignore"):
            d_hat = profile(lik, params[kept], lik.start()).d

        def neg_ll(beta: npt.NDArray) -> float:
            with np.errstate(all="ignore"):
                return -float(profile(lik, beta[kept], d_hat).value)

        search = LikelihoodRegion(
            neg_ll,
            params,
            self.parameter_names,
            self._parameter_bounds(),
            set(aliased.tolist()),
            self._covariance,
            point,
        )
        self._lr_search = search
        return search

    def cb(self, *args: Any, **kwargs: Any) -> npt.NDArray:
        """
        Not available: the baseline of an interval-censored NPMLE
        converges more slowly than the coefficients, and no variance of
        the predicted curve is offered. Bound the coefficients with
        :meth:`param_cb`, or fit a parametric proportional hazards model
        (``WeibullPH.fit(x, Z, c=c)``), whose ``cb`` bounds the curve at
        any covariates.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> try:
        ...     model.cb(2, [0])
        ... except ValueError as error:
        ...     print(str(error)[:44])
        IntervalCoxPH has no confidence bounds on th
        """
        raise ValueError(
            "IntervalCoxPH has no confidence bounds on the predicted curves "
            "(the interval-censored baseline converges more slowly than "
            "the coefficients, and no variance of it is offered). Bound "
            "the coefficients with param_cb (method='wald', 'lr' or "
            "'bootstrap'), or fit a parametric proportional hazards model "
            "(e.g. WeibullPH.fit(x, Z, c=c)), whose cb bounds the curves."
        )

    def _parameter_bounds(self) -> list[tuple[None, None]]:
        """The support of each coefficient: unbounded."""
        return [(None, None)] * len(self.params)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_lr_search", None)
        return state

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The coefficient table: one row per covariate, with the
        coefficient, the hazard ratio ``exp(coef)``, the profile
        likelihood's standard error, two-sided ``1 - alpha_ci`` Wald
        intervals for both, the Wald statistic ``z`` and its two-sided
        p-value.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(4)
                     coef  exp(coef)  se(coef)       p
        covariate
        coef_0    -1.3267     0.2654    0.9889  0.1797
        """
        check_alpha_ci(alpha_ci)
        beta = np.asarray(self.beta, dtype=float)
        names = coefficient_names(self, beta.size)
        se = np.asarray(self._se, dtype=float)
        return coefficient_table(names, beta, se, alpha_ci, p=self.p_values)

    def _data_repr(self) -> str:
        data = self._fit_data
        if data is None:
            return self._data_summary or ""
        x = data.x()
        if not np.isfinite(data.tl).any():
            return data_summary(data.c, data.n, x=x)
        return data_summary(data.c, data.n, tl=data.tl, lower=-np.inf, x=x)

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Proportional Hazards"
            + "\nKind                : NPMLE for interval-censored data "
            "(Turnbull intervals)" + "\nParameterization    : Semi-Parametric"
        )
        data_line = self._data_repr()
        if data_line:
            out += "\nData                : " + data_line
        if np.any(self._center()):
            out += (
                "\nBaseline at         : the covariate means, Z = {}".format(
                    np.array2string(self._center(), separator=", ")
                )
            )
        out += (
            "\nCoefficients        : exp(coef) is the hazard ratio; Wald "
            "95% intervals from the profile likelihood\n"
        )
        return out + coefficient_repr(self.summary()) + "\n"

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-serialisable dict: the
        coefficients with their covariance and p-values, the Turnbull
        intervals with the baseline's masses, and the baseline's times
        and survival (from which ``H0`` is rebuilt). Every prediction
        round-trips exactly; the fitted rows are not stored.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH, IntervalCoxPHModel
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> restored = IntervalCoxPHModel.from_dict(model.to_dict())
        >>> bool(restored.sf(4, [1]) == model.sf(4, [1]))
        True
        """
        H0 = np.asarray(self.H0, dtype=float)
        infinite = ~np.isfinite(H0)
        out: dict[str, Any] = {
            "model": "IntervalCoxPHModel",
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "se": np.asarray(self._se, dtype=float).tolist(),
            "covariance": (
                np.full((self.beta.size, self.beta.size), np.nan)
                if self._covariance is None
                else np.asarray(self._covariance, dtype=float)
            ).tolist(),
            "p_values": np.asarray(self.p_values, dtype=float).tolist(),
            "x": np.asarray(self.x, dtype=float).tolist(),
            # The finite cumulative hazards, and where it is infinite
            "H0": np.where(infinite, 0.0, H0).tolist(),
            "H0_infinite": infinite.tolist(),
            "intervals": np.asarray(self.intervals, dtype=float).tolist(),
            "p0": np.asarray(self.p0, dtype=float).tolist(),
            "_neg_ll": float(getattr(self, "_neg_ll", np.nan)),
            "ic_n": float(self._ic_sample_size()),
            "n_iter": int(self.n_iter),
            **maximum_entry(self.maximum),
        }
        if np.any(self._center()):
            out["center"] = self._center().tolist()
        if self._data_repr():
            out["data_summary"] = self._data_repr()
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> IntervalCoxPHModel:
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH, IntervalCoxPHModel
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> IntervalCoxPHModel.from_dict(model.to_dict()).beta.round(4)
        array([-1.3267])
        """
        require_model_tag(
            model_dict,
            "IntervalCoxPHModel",
            "a semi-parametric proportional hazards model for "
            "interval-censored data",
        )
        out = cls()
        for key in ("beta", "params", "p_values", "x", "p0"):
            setattr(out, key, np.array(model_dict[key], dtype=float))
        out._se = np.array(model_dict["se"], dtype=float)
        H0 = np.array(model_dict["H0"], dtype=float)
        infinite = np.array(model_dict.get("H0_infinite", []), dtype=bool)
        if infinite.size:
            H0[infinite] = np.inf
        out.H0 = H0
        out.intervals = np.array(model_dict["intervals"], dtype=float).reshape(
            -1, 2
        )
        out._covariance = np.array(
            model_dict["covariance"], dtype=float
        ).reshape(out.beta.size, out.beta.size)
        out.center = np.array(
            model_dict.get("center", np.zeros(out.beta.size)), dtype=float
        )
        out._neg_ll = float(model_dict.get("_neg_ll", np.nan))
        out._ic_n = cls._restored_ic_n(model_dict)
        out.n_iter = int(model_dict.get("n_iter", 0))
        out.maximum = restored_maximum(model_dict)
        out._data_summary = model_dict.get("data_summary")
        restore_covariate_meta(out, model_dict)
        return out


def _baseline(
    atoms: Any,
    d: npt.NDArray,
    beta: npt.NDArray,
    mean: npt.NDArray,
    Z: npt.NDArray,
    at_center: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """``(x, H0, intervals, p0)``: the baseline's times (the distinct
    finite right ends of the atoms) and cumulative hazard there, the
    atoms and their masses; at the covariate means with ``at_center``,
    else moved to ``Z = 0`` (refused where it over- or underflows)."""
    Hc = np.concatenate([np.cumsum(d), [np.inf]])
    if at_center:
        H = Hc
    else:
        shift = -float(beta @ mean)
        lp = Z @ beta
        if not (abs(shift) < LOG_MAX and np.all(np.abs(lp) < LOG_MAX)):
            raise baseline_at_origin_error(
                "baseline hazard",
                mean,
                -shift,
                shift,
                lp=lp if abs(shift) < LOG_MAX else None,
            )
        with np.errstate(over="ignore", invalid="ignore"):
            H = Hc * np.exp(shift)
        H = np.where(np.isinf(Hc), np.inf, H)
    S = np.exp(-H)
    S_prev = np.concatenate([[1.0], S[:-1]])
    p0 = S_prev - S
    right = np.asarray(atoms.right, dtype=float)
    finite = np.isfinite(right)
    times, last = np.unique(right[finite][::-1], return_index=True)
    # The value at the last atom ending at each time
    idx = np.flatnonzero(finite)[::-1][last]
    intervals = np.column_stack([atoms.left, atoms.right])
    return times, H[idx], intervals, p0


# ---------------------------------------------------------------------------
# The fitter
# ---------------------------------------------------------------------------
class IntervalCoxPH_(FitterRepr):
    """
    The semi-parametric proportional hazards (Cox) model for
    interval-censored data: the covariates multiply the hazard,
    :math:`S(x \\mid Z) = S_0(x)^{\\exp(\\beta' Z)}`, and the baseline
    :math:`S_0` is a non-parametric maximum-likelihood estimate on the
    Turnbull intervals of the pooled data, fitted jointly with the
    coefficients (R's ``icenReg::ic_sp(model = "ph")``). It takes exact,
    right-, left- and interval-censored rows, counts and left truncation.
    ``IntervalCoxPH`` is an instance of this class; its ``fit`` returns an
    :class:`IntervalCoxPHModel`.

    With exact and right-censored data only, ``CoxPH`` is the usual
    choice: its partial likelihood treats the baseline as a continuous
    hazard, where this model treats each event time as a drop of a
    discrete survival curve (the grouped-data likelihood of Prentice and
    Gloeckler 1978), so the two give close but not identical coefficients.
    """

    fitter_kind = (
        "semi-parametric interval-censored proportional hazards fitter"
    )

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        tol: float = 1e-10,
        center: bool = False,
    ) -> IntervalCoxPHModel:
        """
        Fit the semi-parametric proportional hazards model to
        interval-censored data.

        Parameters
        ----------
        x : array_like
            The times: one column for exact, right- and left-censored
            rows, or ``[left, right]`` pairs (two columns), where an
            interval-censored row's event lies in ``(left, right]``.
        Z : array_like
            The covariates, one row per observation. Rows with a missing
            or infinite covariate are dropped, with a warning. A constant
            column (the baseline absorbs it) or a linear combination of
            the others is aliased: its coefficient is ``nan`` and one
            warning names it.
        c : array_like, optional
            The censoring flags: 0 exact, 1 right censored, -1 left
            censored, 2 interval censored. Defaults to all exact.
        n : array_like, optional
            The count of each row. Defaults to 1.
        tl : array_like, optional
            Left-truncation (delayed entry) times: each row's likelihood
            is conditional on surviving to its entry. Right truncation is
            refused; use a parametric PH model (``WeibullPH``) with
            ``t=[tl, tr]``.
        tol : float, optional
            The convergence tolerance on the Newton decrement of the
            log-likelihood. Default 1e-10.
        center : bool, optional
            ``False`` (the default) reports the baseline at ``Z = 0``;
            ``True`` at the covariate means, stored as ``model.center``.
            The coefficients and predictions are the same either way.

        Returns
        -------
        IntervalCoxPHModel
            The fitted model.

        Examples
        --------
        >>> from surpyval import IntervalCoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = IntervalCoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.beta.round(4)
        array([-1.3267])
        >>> model.standard_errors().round(4)
        array([0.9889])
        """
        data = _validate(x, Z, c, n, tl)
        p = data.Z.shape[1]
        aliased, scale = _aliased(data)
        kept = np.setdiff1d(np.arange(p), aliased)
        if aliased.size:
            warn_aliased(
                aliased,
                "the likelihood does not depend on them beyond a "
                "combination of the other columns (a constant column, "
                "which the baseline absorbs, or a linear combination of "
                "the others)",
            )
        fit = _Fit(data, kept, tol)
        verdict = fit.atoms.verdict
        if verdict == DOES_NOT_EXIST:
            raise ValueError(
                "The likelihood of these left-truncated data has no "
                "maximum: no unit at risk is known to survive past some "
                "time before a later entry, so the baseline there is not "
                "determined. Drop the rows that enter after that gap, or "
                "fit a parametric PH model (e.g. WeibullPH) instead."
            )
        if verdict == NOT_UNIQUE:
            warn_unverified(
                "The IntervalCoxPH fit",
                "the truncated data leave part of the baseline undetermined "
                "(no unit is at risk across a gap between entries), so the "
                "baseline there is one of many with the same likelihood",
                "read the curve only where units were at risk",
            )
        maximum = _check_maximum(fit, scale, kept)
        sol = fit.solution
        if kept.size and maximum != "no finite maximum":
            info = fit.profile_information()
        else:
            info = np.full((kept.size, kept.size), np.nan)
        cov_k = _inverse(info)
        with np.errstate(invalid="ignore", divide="ignore"):
            se_k = np.sqrt(np.diag(cov_k))
            p_k = 2.0 * norm.sf(np.abs(sol.beta / se_k))
        beta = expand(sol.beta, kept, p)
        cov = np.full((p, p), np.nan)
        cov[np.ix_(kept, kept)] = cov_k
        full = np.zeros(p)
        full[kept] = sol.beta
        times, H0, intervals, p0 = _baseline(
            fit.atoms, sol.d, full, fit.mean, data.Z, at_center=center
        )
        model = IntervalCoxPHModel()
        model.beta = beta
        model.params = copy(beta)
        model._se = expand(se_k, kept, p)
        model._covariance = cov
        model.p_values = expand(p_k, kept, p)
        model.x = times
        model.H0 = H0
        model.intervals = intervals
        model.p0 = p0
        model.center = fit.mean if center else np.zeros(p)
        model.log_likelihood = float(sol.value)
        model.n_iter = int(sol.n_iter)
        model.maximum = maximum
        model._fit_data = data
        return model

    @takes_unit
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        tl_col: str | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
        tol: float = 1e-10,
        center: bool = False,
    ) -> IntervalCoxPHModel:
        """
        Fit the model from a pandas DataFrame, keeping the covariate
        names for prediction and the coefficient table; see :meth:`fit`.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str, optional
            The column of the times, for data with no interval rows. Give
            this or ``xl_col`` and ``xr_col``.
        Z_cols : str or list of str, optional
            The covariate column(s). Give this or ``formula``.
        c_col, n_col, tl_col : str, optional
            The columns of the censoring flags, counts and entry times.
        formula : str, optional
            A ``formulaic`` formula for the covariates.
        xl_col, xr_col : str, optional
            The columns of each row's interval ``(xl, xr]``: equal for an
            exact time, ``xr`` infinite for a right-censored row and
            ``xl`` 0 or ``-inf`` for a left-censored one, which the
            censoring flags are read from when ``c_col`` is not given.
        tol, center
            As for :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import IntervalCoxPH
        >>> df = pd.DataFrame({
        ...     "visit_before": [0, 1, 2, 4, 3, 5, 7, 1, 2, 5],
        ...     "visit_after": [2, 3, 5, 6, 8, 5, float("inf"), 4, 6, 9],
        ...     "treated": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ... })
        >>> model = IntervalCoxPH.fit_from_df(
        ...     df, xl_col="visit_before", xr_col="visit_after",
        ...     Z_cols="treated",
        ... )
        >>> model.beta.round(4)
        array([-1.3267])
        """
        check_columns(
            df,
            x_col=x_col,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            xl_col=xl_col,
            xr_col=xr_col,
        )
        design = FrameDesign(df, Z_cols, formula)
        if (xl_col is None) != (xr_col is None):
            raise ValueError(
                "Give both 'xl_col' and 'xr_col' (each row's interval), or "
                "'x_col' alone."
            )
        if xl_col is not None:
            if x_col is not None:
                raise ValueError(
                    "Give either 'x_col' or 'xl_col' and 'xr_col', not both."
                )
            xl = np.asarray(design.column(xl_col), dtype=float)
            xr = np.asarray(design.column(xr_col), dtype=float)
            c = design.column(c_col)
            if c is None:
                c = np.where(
                    xl == xr,
                    0,
                    np.where(np.isinf(xr), 1, np.where(xl <= 0, -1, 2)),
                )
            c_arr = np.asarray(c)
            x = np.column_stack(
                [
                    np.where(c_arr == -1, xr, xl),
                    np.where(c_arr == 1, xl, xr),
                ]
            )
        elif x_col is None:
            raise ValueError(
                "Give the times: 'x_col', or 'xl_col' and 'xr_col'."
            )
        else:
            x = design.column(x_col)
            c = design.column(c_col)
        return design.fit(
            self.fit,
            x,
            design.Z,
            c=c,
            n=design.column(n_col),
            tl=design.column(tl_col),
            tol=tol,
            center=center,
        )


IntervalCoxPH = IntervalCoxPH_()
