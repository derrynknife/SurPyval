r"""``CoxPH``'s full-likelihood route: the proportional hazards model with
a non-parametric maximum-likelihood (NPMLE) baseline, for data the partial
likelihood cannot take -- left- or interval-censored rows, right
truncation -- with one cause or several (cause-specific hazards).

``CoxPH.fit`` comes here when a row is left censored (``c = -1``),
interval censored (``c = 2``) or right truncated (finite ``tr``); exact
and right-censored data, with or without left truncation, keep the
partial likelihood. The fitted model is a :class:`CoxNPMLEModel`, and
its ``estimator`` says which route was taken (``"NPMLE"``).

The model
---------
Each cause :math:`k` has its own hazard,
:math:`\lambda_k(t \mid Z) = \lambda_{0k}(t)\, e^{\beta_k' Z}`, and the
probability of having had no event of any cause by :math:`t` is
:math:`S(t \mid Z) = \exp(-\sum_k \Lambda_k(t \mid Z))`. Each baseline is
a step function on the Turnbull intervals of the data, and all are
fitted together with the coefficients in one likelihood (see
``_cs_npmle``). Every row is used as it was seen: a failure from cause
:math:`k` somewhere in :math:`(l, r]` contributes the probability of
failing from cause :math:`k` there, :math:`\int_l^r S(t^- \mid Z)\,
d\Lambda_k(t \mid Z)`; a right-censored row :math:`S(l \mid Z)`; a
failure with no recorded cause (with ``masked="include"``) the sum over
the causes; and a row seen only because its event fell in its window
:math:`(t_l, t_r]` is divided by the probability of that,
:math:`S(t_l \mid Z) - S(t_r \mid Z)`.

The window uses the all-cause survival because delayed entry and right
truncation select units on their first event of *any* cause: a unit
enters at :math:`t_l` only if no event of any cause has happened by
then, and is in a right-truncated sample only if its event (of whichever
cause) happened by :math:`t_r`. This is the competing-risks truncated
likelihood of Hudgens, Satten and Longini (2001); truncation on one
cause's event alone is a different design and is not offered.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.univariate.nonparametric._turnbull_npmle import (
    DOES_NOT_EXIST,
    NOT_UNIQUE,
)
from surpyval.utils import xcnt_handler
from surpyval.utils.covariates import (
    check_covariate_rows,
    finite_covariate_mask,
)
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
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
)
from surpyval.utils.warnings import ignoring_warnings

from .._aliasing import aliased_columns, constant_columns, warn_aliased
from .._bootstrap import bound_method, check_n_boot
from .._fit_skeleton import LOG_MAX, baseline_at_origin_error, covariate_center
from .._prediction import paired_probabilities, step_quantiles, unique_rows
from .._summary import coefficient_names, coefficient_repr, coefficient_table
from ..life_at import draws_by_quantile
from ..regression_data import (
    LinearPredictorMixin,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from . import _cs_npmle as engine

if TYPE_CHECKING:
    import pandas as pd

#: The profile likelihood's second differences step ``PROFILE_STEP /
#: sqrt(N)`` standard deviations of each covariate (as ``IntervalCoxPH``).
PROFILE_STEP = 1.0
#: The Newton decrement a profile-likelihood point is solved to: its value
#: is then within about 1e-9 of the maximum over the baselines, against
#: differences of order 0.1 at the curvature's step (a relative error of
#: about 1e-8 in a standard error).
PROFILE_TOL = 1e-9
COVARIANCE_METHODS = ("profile", "bootstrap")
MASKED = ("refuse", "include")
#: What ``estimator`` says for each of ``CoxPH``'s routes (as
#: ``CompetingRisks`` says "Aalen-Johansen" or "NPMLE").
NPMLE = "NPMLE"
PARTIAL_LIKELIHOOD = "partial likelihood"
_CB_ON = ("sf", "ff", "Hf", "cif", "R", "F")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def two_columns(x: Any) -> Any:
    """``x`` with a mix of single times and ``[left, right]`` pairs (a
    ragged list) as two columns, a single time ``t`` as ``[t, t]``; any
    other ``x`` as given."""
    if isinstance(x, (list, tuple)) and any(
        isinstance(v, (list, tuple, np.ndarray)) for v in x
    ):
        if not all(isinstance(v, (list, tuple, np.ndarray)) for v in x):
            return np.array(
                [
                    (
                        list(v)
                        if isinstance(v, (list, tuple, np.ndarray))
                        else [v, v]
                    )
                    for v in x
                ],
                dtype=float,
            )
    return x


def needs_full_likelihood(
    x: Any, c: Any, tr: Any, masked: str = "refuse"
) -> bool:
    """Whether ``CoxPH.fit`` must take the full-likelihood route: a left-
    or interval-censored row, an interval given as two different times, a
    masked cause to include, or a finite right-truncation time.

    The first filter is ``CompetingRisks``' own (``windowed_input``):
    data it reads as windows may hold such rows. Two-column ``x`` whose
    every pair is one time, with flags 0 and 1 only, is exact and
    right-censored data, and keeps the partial likelihood."""
    from surpyval.univariate.competing_risks.nonparametric._ic_data import (
        windowed_input,
    )

    if tr is not None and np.any(np.isfinite(np.asarray(tr, dtype=float))):
        return True
    x = two_columns(x)
    if not windowed_input(x, c, None, None, masked):
        return False
    if masked == "include":
        return True
    if c is not None:
        c_arr = np.asarray(c, dtype=object)
        if np.isin(c_arr, (-1, 2)).any():
            return True
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return False
    if x_arr.ndim == 2 and x_arr.shape[1] == 2:
        both = np.all(np.isfinite(x_arr), axis=1)
        return bool(np.any(both & (x_arr[:, 0] != x_arr[:, 1])))
    return False


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
class _Rows:
    """The rows as the fit uses them: ``xl``, ``xr`` (``-inf`` for a
    left-censored row's lower end, ``inf`` for a right-censored row's
    upper end, equal for an exact time), ``cause`` (index; -1 for none),
    ``n``, ``tl``, ``tr`` and ``Z``."""

    def __init__(self, xl, xr, cause, n, tl, tr, Z):  # type: ignore
        self.xl = np.asarray(xl, dtype=float)
        self.xr = np.asarray(xr, dtype=float)
        self.cause = np.asarray(cause, dtype=np.int64)
        self.n = np.asarray(n, dtype=float)
        self.tl = np.asarray(tl, dtype=float)
        self.tr = np.asarray(tr, dtype=float)
        self.Z = np.asarray(Z, dtype=float)

    def subset(self, counts: npt.NDArray) -> _Rows:
        keep = counts > 0
        return _Rows(
            self.xl[keep],
            self.xr[keep],
            self.cause[keep],
            counts[keep].astype(float),
            self.tl[keep],
            self.tr[keep],
            self.Z[keep],
        )

    @property
    def c(self) -> npt.NDArray:
        return np.where(
            self.xl == self.xr,
            0,
            np.where(np.isinf(self.xr), 1, np.where(np.isinf(self.xl), -1, 2)),
        )


def _objects(e: Any, N: int) -> npt.NDArray:
    out = np.empty(N, dtype=object)
    vals = list(e) if not isinstance(e, np.ndarray) else list(e.ravel())
    if len(vals) != N:
        raise ValueError(
            f"e must have one cause per row: got {len(vals)} for {N} rows."
        )
    for i, v in enumerate(vals):
        out[i] = v
    return out


def parse(
    x: Any,
    Z: Any,
    c: Any,
    n: Any,
    tl: Any,
    tr: Any,
    e: Any,
    masked: str,
) -> tuple[_Rows, list, bool]:
    """Check and read the data of the full-likelihood route. Returns the
    rows, the causes (one ``None`` cause without ``e``) and whether the
    fit has a single, unnamed cause."""
    from surpyval.univariate.competing_risks.labels import ordered_labels
    from surpyval.utils.data_formats import missing_events

    if masked not in MASKED:
        raise ValueError(
            f"masked must be one of {list(MASKED)}; got {masked!r}. "
            "'refuse' (the default) refuses a failure with no cause; "
            "'include' counts it as a failure from some cause in its "
            "window."
        )
    x = two_columns(x)
    if c is not None and x is not None and np.ndim(x) == 2:
        # A right-censored row may be written [xl, inf] and a left-censored
        # one [-inf, xr]: read as the time alone, as the handler wants it.
        x = np.array(x, dtype=float)
        flags = np.asarray(c)
        if flags.shape == (x.shape[0],):
            x[flags == 1, 1] = x[flags == 1, 0]
            x[flags == -1, 0] = x[flags == -1, 1]
    x_h, c_h, n_h, t_h = xcnt_handler(
        x, c, n, tl=tl, tr=tr, group_and_sort=False
    )
    c_arr = np.asarray(c_h).astype(int)
    x_arr = np.asarray(x_h, dtype=float)
    if x_arr.ndim == 1:
        x_arr = np.column_stack([x_arr, x_arr])
    n_arr = np.asarray(n_h, dtype=float)
    t_arr = np.asarray(t_h, dtype=float)
    N = c_arr.size
    single = e is None
    if single:
        cause_obj = np.empty(c_arr.size, dtype=object)
        cause_obj[c_arr != 1] = "event"
    else:
        cause_obj = _objects(e, N)
        missing = missing_events(cause_obj)
        cause_obj[missing] = None
        if c is None:
            c_arr = np.where(missing & (c_arr == 0), 1, c_arr)
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    elif Z_arr.ndim != 2:
        raise ValueError("Covariate matrix must be two dimensional")
    check_covariate_rows(Z_arr, N)
    mask = finite_covariate_mask(Z_arr) & (n_arr > 0)
    x_arr, c_arr, n_arr, t_arr, Z_arr, cause_obj = (
        a[mask] for a in (x_arr, c_arr, n_arr, t_arr, Z_arr, cause_obj)
    )
    missing = np.array([v is None for v in cause_obj], dtype=bool)
    suspended = (c_arr == 1) & ~missing
    if suspended.any():
        from surpyval.utils.warnings import named, warn_suspension_types

        cause_obj[suspended] = None
        missing |= suspended
        warn_suspension_types(
            int(suspended.sum()),
            "rows " + named(np.flatnonzero(suspended).tolist()),
        )
    unknown = (c_arr != 1) & missing
    if unknown.any() and masked == "refuse":
        k = int(unknown.sum())
        raise ValueError(
            f"{k} failure{'' if k == 1 else 's'} (c = 0, -1 or 2) with no "
            "cause in e (masked causes). Give each failure its cause, or "
            "pass masked='include' to count such a row as a failure from "
            "some cause in its window (this assumes whether the cause was "
            "recorded does not depend on which cause it was). A row that "
            "had not failed is right censored: c = 1."
        )
    if single:
        causes: list = [None]
    else:
        causes = ordered_labels(cause_obj[~missing])
    if (
        not np.any(c_arr != 1)
        or not causes
        or (not single and not np.any(~missing))
    ):
        raise ValueError(
            "CoxPH needs at least one row whose event is observed or "
            "bracketed (c = 0, -1 or 2), with its cause in e for a "
            "competing-risks fit; with every row right censored there is "
            "no baseline to estimate."
        )
    index = {k: i for i, k in enumerate(causes)}
    cause = np.full(c_arr.size, -1, dtype=np.int64)
    for i in np.flatnonzero(~missing):
        cause[i] = 0 if single else index[cause_obj[i]]
    if np.any(np.isnan(x_arr[c_arr != -1, 0])) or np.any(
        np.isnan(x_arr[c_arr != 1, 1])
    ):
        raise ValueError("CoxPH needs a time on every row (no NaN).")
    xl = np.where(c_arr == -1, -np.inf, x_arr[:, 0])
    xr = np.where(c_arr == 1, np.inf, x_arr[:, 1])
    xr = np.where(c_arr == 0, xl, xr)
    tl_arr = np.where(np.isfinite(t_arr[:, 0]), t_arr[:, 0], -np.inf)
    tr_arr = np.where(np.isfinite(t_arr[:, 1]), t_arr[:, 1], np.inf)
    return (
        _Rows(xl, xr, cause, n_arr, tl_arr, tr_arr, Z_arr),
        causes,
        single,
    )


def _aliased(rows: _Rows) -> tuple[npt.NDArray, npt.NDArray]:
    """The columns the likelihood cannot determine, and each column's
    scale for the no-maximum check, per cause: the cause's informative
    rows times the column's variance (``(K, p)``)."""
    Z, n = rows.Z, rows.n
    Zc = Z - covariate_center(Z, n)
    gram = (Zc * n[:, None]).T @ Zc
    aliased = aliased_columns(gram, Z.shape[0], constant_columns(Z))
    K = max(int(rows.cause.max()) + 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.diag(gram) / float(np.sum(n))
    informative = np.array(
        [
            float(np.sum(n[(rows.cause == k) | (rows.cause < 0)]))
            for k in range(K)
        ]
    )
    return aliased, informative[:, None] * var[None, :]


class _Fit:
    """One maximisation of the likelihood on ``rows``."""

    def __init__(self, rows: _Rows, K: int, kept: npt.NDArray, tol: float):
        self.rows = rows
        self.K = K
        self.kept = kept
        self.mean = covariate_center(rows.Z, rows.n)
        self.s = engine.build(
            rows.xl, rows.xr, rows.cause, rows.n, rows.tl, rows.tr, K
        )
        Zc = (rows.Z - self.mean)[:, kept]
        self.lik = engine.Likelihood(self.s, Zc)
        zero = np.zeros((K, kept.size))
        with np.errstate(all="ignore"):
            start = engine.profile(
                self.lik, zero, self.lik.start(), tol=PROFILE_TOL
            )
            self.solution = (
                engine.maximise(self.lik, start.theta, tol=tol)
                if kept.size
                else start
            )

    @property
    def beta(self) -> npt.NDArray:
        return self.lik.split(self.solution.theta)[0]

    @property
    def d(self) -> npt.NDArray:
        return self.lik.split(self.solution.theta)[1]

    def profile_value(self, beta: npt.NDArray) -> float:
        with np.errstate(all="ignore"):
            return engine.profile(
                self.lik, beta, self.solution.theta, tol=PROFILE_TOL
            ).value

    def schur(self) -> npt.NDArray:
        """The coefficients' information with the baselines' positive
        increments profiled out (the scale of the no-maximum check)."""
        nb = self.lik.n_beta
        H = self.solution.der.hess
        theta = self.solution.theta
        free = np.concatenate([np.zeros(nb, bool), theta[nb:] > 0])
        Hbb = H[:nb, :nb]
        Hbd = H[:nb][:, free]
        Hdd = H[np.ix_(free, free)]
        # The baselines need not be unique at the maximum (mass can move
        # between neighbouring atoms at no cost there), so the increments'
        # block can be singular: its pseudo-inverse profiles out the
        # directions the likelihood sees.
        try:
            inner = (
                np.linalg.pinv(Hdd, rcond=1e-10, hermitian=True) @ Hbd.T
                if free.any()
                else 0.0
            )
        except np.linalg.LinAlgError:
            return np.full(Hbb.shape, np.nan)
        return -(Hbb - Hbd @ inner)

    def profile_information(self) -> npt.NDArray:
        """The negative Hessian of the profile log-likelihood over the
        coefficients (flattened by cause), by second differences of
        ``PROFILE_STEP / sqrt(N)`` standard deviations of each
        covariate."""
        rows, kept, K = self.rows, self.kept, self.K
        p = kept.size
        beta = self.beta.ravel()
        k = beta.size
        N = float(np.sum(rows.n))
        Zc = (rows.Z - self.mean)[:, kept]
        sd = np.sqrt(np.dot(rows.n, Zc**2) / N)
        h = np.tile(PROFILE_STEP / np.sqrt(N) / np.where(sd > 0, sd, 1.0), K)
        f0 = self.solution.value
        info = np.zeros((k, k))
        eye = np.eye(k)

        def at(b: npt.NDArray) -> float:
            return self.profile_value(b.reshape(K, p))

        for i in range(k):
            up = at(beta + h[i] * eye[i])
            down = at(beta - h[i] * eye[i])
            info[i, i] = -(up - 2.0 * f0 + down) / h[i] ** 2
            for j in range(i):
                ei, ej = h[i] * eye[i], h[j] * eye[j]
                pp = at(beta + ei + ej)
                pm = at(beta + ei - ej)
                mp = at(beta - ei + ej)
                mm = at(beta - ei - ej)
                info[i, j] = info[j, i] = -(pp - pm - mp + mm) / (
                    4.0 * h[i] * h[j]
                )
        return info


def _inverse(info: npt.NDArray) -> npt.NDArray:
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
    fit: _Fit, scale: npt.NDArray, names: list, warn: bool = True
) -> str:
    """The model's ``maximum`` and, with ``warn``, one warning when it is
    not ``"verified"`` (``IntervalCoxPH``'s rule: a coefficient whose
    information has fallen below ``1e-4`` of its scale has run away)."""
    sol = fit.solution
    kept = fit.kept
    if kept.size:
        info = np.diag(np.atleast_2d(fit.schur()))
        d0 = np.asarray(scale, dtype=float)[:, kept].ravel()
        collapsed = np.flatnonzero(
            (d0 > 0) & ~(np.nan_to_num(info, nan=0.0) > 1e-4 * d0)
        )
        if collapsed.size:
            if warn:
                warn_no_maximum(
                    "the likelihood keeps increasing as coefficient(s) {} "
                    "grow without bound, so the estimate is infinite (the "
                    "covariate separates the early events from the late "
                    "ones, as a level with no events of a cause "
                    "does)".format([names[i] for i in collapsed]),
                    "The reported value, its standard error and its "
                    "p-value are meaningless",
                    "consider removing or coarsening the covariate",
                )
            return "no finite maximum"
    if not sol.converged:
        if warn:
            warn_unverified(
                "The CoxPH full-likelihood fit",
                "the projected Newton iteration stopped after {} step(s) "
                "without reaching its tolerance".format(sol.n_iter),
                "check the covariates for extreme values, or rescale them",
            )
        return "unverified"
    return "verified"


def _names(causes: list, single: bool, cols: list) -> list:
    if single:
        return list(cols)
    return [f"{k}: {c}" for k in causes for c in cols]


def _warn_tau(tau: float) -> None:
    warnings.warn(
        "Every row is right truncated, so no unit could be seen to fail "
        f"after {tau:g} (the largest truncation time). How many units "
        "fail after that cannot be told from these data, so the fit "
        f"takes every unit to fail by {tau:g}, as the Turnbull estimate "
        "does: its curves are those of units known to fail by then, and "
        f"are nan after {tau:g}. Fit a parametric model to extrapolate.",
        UserWarning,
        stacklevel=4,
    )


def fit_npmle(
    rows: _Rows,
    causes: list,
    single: bool,
    tol: float = 1e-10,
    center: bool = False,
    feature_names: list | None = None,
    warn_tau: bool = True,
    inference: bool = True,
) -> CoxNPMLEModel:
    """Fit the full-likelihood model to ``rows`` (see the module notes)."""
    K = len(causes)
    p = rows.Z.shape[1]
    aliased, scale = _aliased(rows)
    kept = np.setdiff1d(np.arange(p), aliased)
    if aliased.size:
        warn_aliased(
            aliased,
            "the likelihood does not depend on them beyond a combination "
            "of the other columns (a constant column, which the baselines "
            "absorb, or a linear combination of the others)",
        )
    fit = _Fit(rows, K, kept, tol)
    if fit.s.verdict == DOES_NOT_EXIST:
        raise ValueError(
            "The likelihood of these truncated data has no maximum: no "
            "unit at risk is known to survive past some time before a "
            "later entry, so the baseline there is not determined. Drop "
            "the rows that enter after that gap, or fit a parametric PH "
            "model (e.g. WeibullPH) instead."
        )
    if fit.s.verdict == NOT_UNIQUE:
        warn_unverified(
            "The CoxPH full-likelihood fit",
            "the truncated data leave part of the baseline undetermined "
            "(no unit is at risk across a gap between entries), so the "
            "baseline there is one of many with the same likelihood",
            "read the curves only where units were at risk",
        )
    if warn_tau and np.isfinite(fit.s.tau):
        _warn_tau(fit.s.tau)
    cols = coefficient_names(_Named(feature_names), p)
    names = _names(causes, single, [cols[i] for i in kept])
    maximum = _check_maximum(fit, scale, names)
    if inference and kept.size and maximum != "no finite maximum":
        info = fit.profile_information()
    else:
        info = np.full((K * kept.size,) * 2, np.nan)
    cov_k = _inverse(info)
    beta_k = fit.beta
    with np.errstate(invalid="ignore", divide="ignore"):
        se_k = np.sqrt(np.diag(cov_k))
        p_k = 2.0 * norm.sf(np.abs(beta_k.ravel() / se_k))
    betas = np.full((K, p), np.nan)
    betas[:, kept] = beta_k
    flat_kept = (np.arange(K)[:, None] * p + kept[None, :]).ravel()
    cov = np.full((K * p, K * p), np.nan)
    cov[np.ix_(flat_kept, flat_kept)] = cov_k
    se = np.full(K * p, np.nan)
    se[flat_kept] = se_k
    pv = np.full(K * p, np.nan)
    pv[flat_kept] = p_k
    full = np.zeros((K, p))
    full[:, kept] = beta_k
    d = _baseline(fit.d, full, fit.mean, rows.Z, at_center=center)
    model = CoxNPMLEModel()
    model.causes = list(causes)
    model.single = bool(single)
    model.betas = betas
    model._se = se
    model._covariance = cov
    model.p_values = pv
    model.d = d
    model.left = np.asarray(fit.s.left, dtype=float)
    model.right = np.asarray(fit.s.right, dtype=float)
    model.allowed = np.asarray(fit.s.allowed, dtype=bool)
    model.report_lower = np.asarray(fit.s.report_lower, dtype=float)
    model.tau = float(fit.s.tau)
    model.center = fit.mean if center else np.zeros(p)
    model.log_likelihood = float(fit.solution.value)
    model.n_iter = int(fit.solution.n_iter)
    model.maximum = maximum
    model._fit_data = rows
    model.tol = float(tol)
    if feature_names is not None:
        model.feature_names = list(feature_names)
    return model


class _Named:
    def __init__(self, names: list | None):
        self.feature_names = names


def _baseline(
    d: npt.NDArray,
    beta: npt.NDArray,
    mean: npt.NDArray,
    Z: npt.NDArray,
    at_center: bool,
) -> npt.NDArray:
    """The increments at the covariate means (``at_center``), else moved
    to ``Z = 0`` (refused where that over- or underflows)."""
    if at_center:
        return np.array(d, dtype=float)
    shift = -(beta @ mean)
    lp = Z @ beta.T
    if not (np.all(np.abs(shift) < LOG_MAX) and np.all(np.abs(lp) < LOG_MAX)):
        raise baseline_at_origin_error(
            "baseline hazard",
            mean,
            float(-shift[np.argmax(np.abs(shift))]),
            float(shift[np.argmax(np.abs(shift))]),
        )
    return d * np.exp(shift)[:, None]


# ---------------------------------------------------------------------------
# The fitted model
# ---------------------------------------------------------------------------
class CoxNPMLEModel(
    InformationCriteriaMixin, LinearPredictorMixin, SerialisableMixin
):
    r"""
    A proportional hazards model fitted by ``CoxPH.fit`` through its full
    likelihood, with a non-parametric (NPMLE) baseline for each cause:
    the route ``CoxPH`` takes for left- or interval-censored rows and for
    right truncation. ``estimator`` is ``"NPMLE"``.

    With causes (``CoxPH.fit(..., e=causes)``) each cause has its own
    coefficients, ``betas[k]``, which multiply its own hazard
    (``exp(betas)`` are cause-specific hazard ratios), and ``cif`` gives
    each cause's cumulative incidence. Without ``e`` there is one cause,
    and the model reads as a plain Cox model: ``beta``, ``sf``, ``qf``.

    The baselines are step functions that can only rise within the
    Turnbull intervals of the data. Within an interval the data do not
    say when the events happened, so the curves are held at their value
    at the interval's start and move at its end, as ``Turnbull``'s does.
    Before the first time the survival is 1, and after the last it holds
    its last value -- except where every row is right truncated: then the
    data say nothing after the largest truncation time ``tau``, and every
    curve is ``nan`` there.

    Examples
    --------
    Inspections find each unit failed between two visits, with the cause
    of failure:

    >>> import numpy as np
    >>> from surpyval import CoxPH
    >>> x = [[0, 2], [0, 2], [2, 4], [2, 4], [4, 6], [4, 6], [6, 8],
    ...      [1, 3], [3, 5], [5, 7], [7, 9], [8, 8], [9, 9], [10, 10]]
    >>> c = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    >>> e = ["wear", "crack", "wear", "wear", "crack", "wear", "crack",
    ...      "wear", "crack", "wear", "crack", None, None, None]
    >>> Z = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
    >>> model = CoxPH.fit(x, Z, c=c, e=e)
    >>> model.estimator
    'NPMLE'
    >>> model.parameter_names
    ['crack: coef_0', 'wear: coef_0']
    >>> model.cif([4, 9], [0], "wear").round(4)
    array([0.3648, 0.5158])
    """

    #: How the model was fitted: ``"NPMLE"``, the full likelihood with
    #: non-parametric baselines (``CoxPH``'s partial-likelihood fits say
    #: ``"partial likelihood"``).
    estimator: str = NPMLE
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None
    causes: list
    single: bool
    betas: npt.NDArray
    _se: npt.NDArray
    _covariance: npt.NDArray | None
    p_values: npt.NDArray
    #: The increments of each cause's baseline cumulative hazard on each
    #: atom ``(causes, atoms)``, at ``center``; on the last atom, the
    #: weights of its split between the causes (its increment is
    #: infinite: every unit still event-free there fails there).
    d: npt.NDArray
    #: The atoms' time intervals ``(left, right]``.
    left: npt.NDArray
    right: npt.NDArray
    allowed: npt.NDArray
    report_lower: npt.NDArray
    tau: float = np.inf
    center: npt.NDArray | None = None
    n_iter: int = 0
    tol: float = 1e-10
    maximum: str = "unknown"
    _fit_data: Any = None
    _lr_search: Any = None
    _boot: Any = None
    kind: str = "Proportional Hazards"
    parameterization: str = "Semi-Parametric"

    def __init__(self) -> None:
        self._boot = {}

    # -- names ----------------------------------------------------------

    @property
    def event_idx_map(self) -> dict:
        """Each cause's label and its row of ``betas``."""
        return {k: i for i, k in enumerate(self.causes)}

    @property
    def n_event_types(self) -> int:
        return len(self.causes)

    @property
    def beta(self) -> npt.NDArray:
        """The coefficients: ``betas[0]`` with one cause, else ``betas``
        flattened cause by cause (as ``params``)."""
        b = np.asarray(self.betas, dtype=float)
        return b[0].copy() if self.single else b.ravel()

    @property
    def params(self) -> npt.NDArray:
        """The coefficients, cause by cause, named by
        ``parameter_names``."""
        return np.asarray(self.betas, dtype=float).ravel()

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``: each covariate's column (else
        ``coef_j``), led by its cause's label with several causes
        (``"<cause>: <covariate>"``)."""
        cols = coefficient_names(self, np.shape(self.betas)[1])
        return _names(self.causes, self.single, cols)

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows (and columns) of :meth:`covariance`."""
        return list(self.parameter_names)

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficients the likelihood cannot
        determine (a constant column, which the baselines absorb, or a
        linear combination of the others): ``nan`` in every cause's
        ``betas``, and taken as 0 by the predictions."""
        b = np.atleast_2d(np.asarray(self.betas, dtype=float))
        return np.flatnonzero(np.isnan(b[0]))

    @property
    def x(self) -> npt.NDArray:
        """The times at which the curves can move: the atoms' finite
        right ends."""
        r = np.asarray(self.right, dtype=float)
        return np.unique(r[np.isfinite(r)])

    def _cause_index(self, event: Any) -> int:
        if self.single and event is None:
            return 0
        from surpyval.utils import validate_event

        if event is None:
            raise ValueError(
                "Name the cause: event one of " f"{list(self.causes)}."
            )
        validate_event(self.event_idx_map, event)
        return self.event_idx_map[event]

    # -- curves ---------------------------------------------------------

    def _Zc(self, Z: Any) -> npt.NDArray:
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        p = np.shape(self.betas)[1]
        if Z_arr.ndim == 0:
            Z_arr = Z_arr.reshape(1, 1)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(1, -1) if Z_arr.size == p else Z_arr[:, None]
        return Z_arr - np.asarray(self._center(), dtype=float)

    def _beta0(self) -> npt.NDArray:
        return np.nan_to_num(np.asarray(self.betas, dtype=float), nan=0.0)

    def _curves(self, Zc: npt.NDArray) -> tuple:
        """At each centred row: each cause's mass on each atom, the
        survival after each atom, and each cause's cumulative hazard
        after each atom."""
        mass, S = engine.atom_masses(self._beta0(), self.d, Zc)
        with np.errstate(over="ignore", invalid="ignore"):
            r = np.exp(Zc @ self._beta0().T)
        u = r[:, :, None] * self.d[None, :, :]
        H = np.cumsum(u, axis=2)
        L = self.d.shape[1] - 1
        H[:, :, L] = np.where(
            u[:, :, L] > 0, np.inf, H[:, :, L - 1] if L else 0.0
        )
        return mass, S, H

    def _evaluate(
        self,
        what: str,
        x: npt.ArrayLike,
        Z: Any,
        event: Any,
        grid: bool,
    ) -> npt.NDArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        p = np.shape(self.betas)[1]
        if grid:
            rows = covariate_rows(self._prepare_Z(Z), p)
        else:
            rows = covariate_rows(self._prepare_Z(Z), p)
            if rows.shape[0] not in (1, x_arr.size):
                check_paired_rows(x_arr.size, rows.shape[0])
        uniq, which = unique_rows(rows)
        Zc = uniq - np.asarray(self._center(), dtype=float)
        mass, S, H = self._curves(Zc)
        right = np.asarray(self.right, dtype=float)
        order = np.argsort(right, kind="stable")
        # The number of atoms ending at or before each time.
        idx = np.searchsorted(right[order], x_arr, side="right")
        if what == "cif" or event is not None:
            k = self._cause_index(event)
        else:
            k = None

        def table() -> npt.NDArray:
            # values after each atom, (rows, m + 1) with a leading start
            m = right.size
            zeros = np.zeros((uniq.shape[0], 1))
            if what == "cif":
                cum = np.cumsum(mass[:, k, order], axis=1)
                return np.concatenate([zeros, cum], axis=1)
            if k is None:
                Sa = S[:, order]
                return np.concatenate([np.ones_like(zeros), Sa], axis=1)
            Hk = H[:, k, order]
            return np.concatenate([zeros, Hk], axis=1)[:, : m + 1]

        tab = table()
        if what == "cif":
            vals = tab
        elif k is None:
            S_tab = tab
            vals = {
                "sf": S_tab,
                "ff": 1.0 - S_tab,
                "Hf": _neglog(S_tab),
            }[what if what in ("sf", "ff", "Hf") else "sf"]
            if what in ("hf", "df"):
                base = S_tab if what == "df" else _neglog(S_tab)
                prev = np.concatenate([base[:, :1], base[:, :-1]], axis=1)
                vals = prev - base if what == "df" else base - prev
        else:
            Hk = tab
            if what in ("sf", "ff", "df"):
                Sk = np.exp(-Hk)
                if what == "df":
                    prev = np.concatenate([Sk[:, :1], Sk[:, :-1]], axis=1)
                    vals = prev - Sk
                else:
                    vals = Sk if what == "sf" else 1.0 - Sk
            elif what == "hf":
                prev = np.concatenate([Hk[:, :1], Hk[:, :-1]], axis=1)
                with np.errstate(invalid="ignore"):
                    vals = np.where(
                        np.isinf(Hk) & np.isinf(prev), 0.0, Hk - prev
                    )
            else:
                vals = Hk
        if grid:
            out = vals[which][:, idx]
            bad = np.isnan(x_arr) | (x_arr > self.tau)
            out[:, bad] = np.nan
            return out
        r_idx = (
            which
            if which.size == x_arr.size
            else np.zeros(x_arr.size, dtype=int) + which[0]
        )
        out = vals[r_idx, idx]
        return np.where(np.isnan(x_arr) | (x_arr > self.tau), np.nan, out)

    def _shaped(self, out: npt.NDArray, x: Any, grid: bool, Z: Any) -> Any:
        return out

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The probability of no event of any cause by ``x`` at covariates
        ``Z`` (one row for every ``x``, or one row per ``x``; a
        DataFrame for a model fitted from one), :math:`\\exp(-\\sum_k
        \\Lambda_k(x \\mid Z))`; with ``event``, that cause's net survival
        :math:`\\exp(-\\Lambda_k(x \\mid Z))`. ``grid=True`` evaluates
        every time for every row.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.sf([2, 5, 8], [[0], [1]], grid=True).round(3)
        array([[0.603, 0.027, 0.   ],
               [0.874, 0.384, 0.   ]])
        """
        return self._evaluate("sf", x, Z, event, grid)

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """``1 - sf``; arguments as for :meth:`sf`."""
        return self._evaluate("ff", x, Z, event, grid)

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """The cumulative hazard: all causes' (``-log sf``) or, with
        ``event``, that cause's; ``inf`` once the baseline has spent all
        its mass. Arguments as for :meth:`sf`."""
        return self._evaluate("Hf", x, Z, event, grid)

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """The jump of :meth:`Hf` at the last time at or before ``x``: a
        step size, not a rate, as ``CoxPH``'s ``hf``. Arguments as for
        :meth:`sf`."""
        return self._evaluate("hf", x, Z, event, grid)

    @keeps_query_shape
    def df(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """The drop of :meth:`sf` at the last time at or before ``x``: a
        probability mass, not a density. Arguments as for :meth:`sf`."""
        return self._evaluate("df", x, Z, event, grid)

    @keeps_query_shape
    def cif(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The cumulative incidence of cause ``event`` at ``x`` and
        covariates ``Z``: the probability of failing from that cause by
        ``x`` while the others act, :math:`\\int_0^x S(t^- \\mid Z)\\,
        d\\Lambda_k(t \\mid Z)`. The causes' incidences sum to ``ff``.
        Arguments as for :meth:`sf`.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [0, 2], [2, 4], [2, 4], [4, 6], [4, 6], [6, 8],
        ...      [8, 8], [9, 9]]
        >>> c = [2, 2, 2, 2, 2, 2, 2, 1, 1]
        >>> e = ["a", "b", "a", "a", "b", "a", "b", None, None]
        >>> model = CoxPH.fit(x, [0, 1, 0, 1, 0, 1, 0, 1, 0], c=c, e=e)
        >>> bool(abs(model.cif(9, [1], "a") + model.cif(9, [1], "b")
        ...          - model.ff(9, [1])) < 1e-12)
        True
        """
        if event is None and not self.single:
            raise ValueError(
                f"Name the cause: cif(x, Z, event) with event one of "
                f"{list(self.causes)}."
            )
        return self._evaluate("cif", x, Z, event, grid)

    @keeps_query_shape
    def qf(
        self, p: npt.ArrayLike, Z: Any, *, grid: bool = False
    ) -> npt.NDArray:
        """
        The first time at which ``ff(x, Z)`` reaches ``p`` (``nan`` where
        it never does). ``Z`` is paired with ``p`` as :meth:`sf` pairs it
        with ``x``. A probability outside [0, 1] gives ``nan``, with a
        warning.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.qf(0.5, [[0], [1]])
        array([5., 5.])
        """
        rows = covariate_rows(self._prepare_Z(Z), np.shape(self.betas)[1])
        u, rows, shape = paired_probabilities(p, rows, grid)
        uniq, which = unique_rows(rows)
        times = self.x
        times = times[times <= self.tau]
        F = np.atleast_2d(self.ff(times, uniq, grid=True))
        out = step_quantiles(F[which], times, u)
        return out if shape is None else out.reshape(shape)

    def random(
        self, size: int, Z: Any, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` times to the first event (of any cause) for each
        covariate row of ``Z`` by inverting :meth:`qf`; a draw beyond the
        curve's end is ``inf``. Returns the draws and the row each was
        drawn at.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> draws, rows = model.random(4, [[0]], random_state=1)
        >>> draws
        array([5., 2., 5., 2.])
        """
        rows = covariate_rows(self._prepare_Z(Z), np.shape(self.betas)[1])
        return draws_by_quantile(self, size, rows, random_state)

    def phi(self, Z: Any, event: Any = None) -> npt.NDArray:
        """
        The hazard ratio of cause ``event`` (the only cause without
        ``e``) at ``Z`` against a unit at ``center``.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.phi([[1], [0]]).round(4)
        array([0.2654, 1.    ])
        """
        k = self._cause_index(event)
        with np.errstate(over="ignore"):
            return np.exp(self._Zc(Z) @ self._beta0()[k])

    def innermost_intervals(self, event: Any = None) -> dict:
        """
        The intervals cause ``event``'s mass can lie on, at the
        covariates ``center``: ``{"lower", "upper", "mass"}``. The data
        say only that each mass is somewhere in ``(lower, upper]``.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.innermost_intervals()["upper"]
        array([2., 3., 4., 5., 6., 8.])
        """
        k = self._cause_index(event)
        sel = np.asarray(self.allowed, dtype=bool)[k]
        mass, _, _ = self._curves(np.zeros((1, np.shape(self.betas)[1])))
        return {
            "lower": np.asarray(self.report_lower, dtype=float)[k, sel],
            "upper": np.asarray(self.right, dtype=float)[sel],
            "mass": mass[0, k, sel],
        }

    # -- model comparison -----------------------------------------------

    def _ic_k(self) -> int:
        return int(np.isfinite(self.params).sum())

    def _ic_sample_size_from_data(self) -> float:
        if self._fit_data is not None:
            return ic_sample_size(self._fit_data.c, self._fit_data.n)
        return float(np.nan)

    # -- inference ------------------------------------------------------

    def _require_covariance(self) -> npt.NDArray:
        if self._covariance is None:
            raise no_covariance_error(
                "it was built without one; use the fitted model's"
            )
        return self._covariance

    def covariance(
        self,
        method: str = "profile",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        The covariance of ``params``. ``"profile"`` (the default): the
        inverse curvature of the profile likelihood, the baselines
        re-maximised at each point (as ``IntervalCoxPH``);
        ``"bootstrap"``: the covariance of ``n_boot`` refits to the rows
        resampled with replacement.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.covariance().round(4)
        array([[0.9779]])
        """
        check_option("method", method, COVARIANCE_METHODS)
        if method == "bootstrap":
            draws = self._bootstrap(n_boot, random_state)[0]
            P = draws.shape[1]
            cov = np.full((P, P), np.nan)
            kept = np.flatnonzero(np.isfinite(self.params))
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
        """The square roots of the diagonal of :meth:`covariance` (same
        arguments)."""
        if method == "profile":
            self._require_covariance()
            return np.asarray(self._se, dtype=float)
        with np.errstate(invalid="ignore"):
            return np.sqrt(
                np.diag(self.covariance(method, n_boot, random_state))
            )

    def _estimates(self) -> npt.NDArray:
        return np.array(self.params, dtype=float)

    def _need_data(self, what: str) -> _Rows:
        if self._fit_data is None:
            raise ValueError(
                "{} needs the data the model was fitted to, which a model "
                "restored from a dict does not keep. Call it on the fitted "
                "model, or refit.".format(what)
            )
        return self._fit_data

    def _bootstrap(self, n_boot: Any, random_state: Any) -> tuple:
        """``n_boot`` case-bootstrap refits: their coefficients (one row
        each, ``nan`` where a refit failed) and models, kept for a repeat
        call with the same integer seed."""
        n_boot = check_n_boot(n_boot)
        rows = self._need_data("The bootstrap")
        key = None
        if isinstance(random_state, (int, np.integer)) and not isinstance(
            random_state, bool
        ):
            key = (n_boot, int(random_state))
            if key in self._boot:
                return self._boot[key]
        counts = np.asarray(rows.n, dtype=float)
        if not np.allclose(counts, np.round(counts)):
            raise ValueError(
                "The bootstrap resamples units, so the counts 'n' must be "
                "whole numbers; use method='profile' for fractional counts."
            )
        rng = as_generator(random_state)
        units = int(np.round(counts.sum()))
        prob = counts / counts.sum()
        P = self.params.size
        out = np.full((n_boot, P), np.nan)
        models: list = []
        failed = no_max = unverified = 0
        for b in range(n_boot):
            sample = rows.subset(rng.multinomial(units, prob))
            try:
                with ignoring_warnings():
                    with np.errstate(all="ignore"):
                        refit = _refit(self, sample)
            except (ValueError, np.linalg.LinAlgError, ArithmeticError):
                failed += 1
                models.append(None)
                continue
            if refit.maximum == "no finite maximum":
                no_max += 1
                models.append(None)
                continue
            if refit.maximum == "unverified":
                unverified += 1
            out[b] = refit.params
            models.append(refit)
        warn_bootstrap_refits(
            n_boot,
            no_estimate=no_max,
            unverified=unverified,
            failed=failed,
            advice="treat the bootstrap as rough, or use method='profile'",
        )
        res = (out, models)
        if key is not None:
            self._boot[key] = res
        return res

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
        Confidence bound(s) on a coefficient: ``"wald"`` (the default,
        from the profile likelihood's curvature), ``"lr"`` (the profile
        likelihood-ratio interval, the baselines and other coefficients
        re-fitted) or ``"bootstrap"`` (the percentile interval of
        ``n_boot`` case-bootstrap refits).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.param_cb("coef_0").round(4)
        array([-3.2649,  0.6115])
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
            draws = self._bootstrap(n_boot, random_state)[0][:, idx]
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

    def _parameter_bounds(self) -> list[tuple[None, None]]:
        """The support of each coefficient: unbounded."""
        return [(None, None)] * self.params.size

    def _lr_region(self) -> Any:
        from .._likelihood_ratio import LikelihoodRegion

        rows = self._need_data("Likelihood-ratio bounds")
        params = np.nan_to_num(self.params, nan=0.0)
        point = params.tobytes()
        search = self._lr_search
        if search is not None and search.point == point:
            return search
        K, p = np.shape(self.betas)
        aliased_cols = np.flatnonzero(np.isnan(np.asarray(self.betas)[0]))
        kept = np.setdiff1d(np.arange(p), aliased_cols)
        fit = _Fit(rows, K, kept, 1e-10)
        theta0 = fit.solution.theta
        lik = fit.lik

        def neg_ll(flat: npt.NDArray) -> float:
            b = np.asarray(flat, dtype=float).reshape(K, p)[:, kept]
            with np.errstate(all="ignore"):
                return -float(
                    engine.profile(lik, b, theta0, tol=PROFILE_TOL).value
                )

        aliased = set(
            (np.arange(K)[:, None] * p + aliased_cols[None, :])
            .ravel()
            .tolist()
        )
        search = LikelihoodRegion(
            neg_ll,
            params,
            self.parameter_names,
            [(None, None)] * params.size,
            aliased,
            self._covariance,
            point,
        )
        self._lr_search = search
        return search

    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Approximate pointwise bootstrap bounds on a curve at covariates
        ``Z``: ``on="sf"`` (default), ``"ff"``, ``"Hf"`` or ``"cif"``
        (cause ``event``). The rows are resampled ``n_boot`` times, the
        model refitted to each sample, and the bounds are the
        percentiles of the refitted curves. They are approximate: the
        NPMLE baseline converges more slowly than root-n, and for
        current-status data the plain bootstrap of it is known not to be
        consistent (Sen, Banerjee and Woodroofe 2010).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.cb(4, [0], n_boot=50, random_state=1).shape
        (2,)
        """
        check_option("on", on, _CB_ON)
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        if method != "bootstrap":
            raise ValueError(
                f"cb of a CoxPH full-likelihood fit has method='bootstrap' "
                f"only (got {method!r}): the NPMLE baselines have no "
                "Wald variance. Bound the coefficients with param_cb "
                "(method='wald', 'lr' or 'bootstrap'), or fit a parametric "
                "model (e.g. WeibullPH) for Wald bounds on the curves."
            )
        on = {"R": "sf", "F": "ff"}.get(on, on)
        if on == "cif":
            if event is None and not self.single:
                raise ValueError(
                    "Name the cause for on='cif': event one of "
                    f"{list(self.causes)}."
                )
        models = self._bootstrap(n_boot, random_state)[1]
        draws = []
        for m in models:
            if m is None:
                continue
            with np.errstate(all="ignore"):
                draws.append(
                    np.asarray(getattr(m, on)(x, Z, event), dtype=float)
                )
        point = np.asarray(getattr(self, on)(x, Z, event), dtype=float)
        if len(draws) < 2:
            shape = point.shape + ((2,) if bound == "two-sided" else ())
            return np.full(shape, np.nan)
        arr = np.array(draws)
        with ignoring_warnings(RuntimeWarning):
            if bound == "two-sided":
                lo = np.nanquantile(arr, alpha_ci / 2, axis=0)
                hi = np.nanquantile(arr, 1 - alpha_ci / 2, axis=0)
                out = np.stack([lo, hi], axis=-1)
            else:
                level = alpha_ci if bound == "lower" else 1 - alpha_ci
                out = np.nanquantile(arr, level, axis=0)
        nan = np.isnan(point)
        if bound == "two-sided":
            nan = nan[..., None]
        return np.where(nan, np.nan, out)

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The coefficient table: one row per cause and covariate, with the
        coefficient, the cause-specific hazard ratio ``exp(coef)``, the
        profile likelihood's standard error, Wald intervals, ``z`` and
        its p-value.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> model.summary()[["coef", "exp(coef)", "se(coef)"]].round(4)
                     coef  exp(coef)  se(coef)
        covariate
        coef_0    -1.3267     0.2654    0.9889
        """
        check_alpha_ci(alpha_ci)
        return coefficient_table(
            self.parameter_names,
            self.params,
            np.asarray(self._se, dtype=float),
            alpha_ci,
            p=self.p_values,
        )

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_lr_search", None)
        return state

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Proportional Hazards"
            + ("" if self.single else " (cause-specific)")
            + "\nEstimator           : NPMLE (full likelihood, baselines "
            "on the Turnbull intervals)"
        )
        if not self.single:
            out += f"\nCauses              : {list(self.causes)}"
        if np.isfinite(self.tau):
            out += (
                "\nObserved up to      : {:g} (every row right truncated; "
                "nan after)".format(self.tau)
            )
        out += (
            "\nCoefficients        : exp(coef) is the hazard ratio; Wald "
            "95% intervals from the profile likelihood\n"
        )
        return out + coefficient_repr(self.summary()) + "\n"

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a JSON-serialisable dict; every prediction
        round-trips exactly (the fitted rows are not stored).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.univariate.regression import CoxNPMLEModel
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> restored = CoxNPMLEModel.from_dict(model.to_dict())
        >>> bool(restored.sf(4, [1]) == model.sf(4, [1]))
        True
        """
        P = self.params.size
        out: dict[str, Any] = {
            "model": "CoxNPMLEModel",
            "estimator": self.estimator,
            "causes": [to_native(k) for k in self.causes],
            "single": bool(self.single),
            "betas": np.asarray(self.betas, dtype=float).tolist(),
            "se": np.asarray(self._se, dtype=float).tolist(),
            "covariance": (
                np.full((P, P), np.nan)
                if self._covariance is None
                else np.asarray(self._covariance, dtype=float)
            ).tolist(),
            "p_values": np.asarray(self.p_values, dtype=float).tolist(),
            "d": np.asarray(self.d, dtype=float).tolist(),
            "left": np.asarray(self.left, dtype=float).tolist(),
            "right": np.asarray(self.right, dtype=float).tolist(),
            "allowed": np.asarray(self.allowed, dtype=bool).tolist(),
            "report_lower": np.asarray(
                self.report_lower, dtype=float
            ).tolist(),
            "tau": float(self.tau),
            "center": np.asarray(self._center(), dtype=float).tolist(),
            "_neg_ll": -float(self.log_likelihood),
            "ic_n": float(self._ic_sample_size()),
            "n_iter": int(self.n_iter),
            "tol": float(self.tol),
            **maximum_entry(self.maximum),
        }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> CoxNPMLEModel:
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.univariate.regression import CoxNPMLEModel
        >>> x = [[0, 2], [1, 3], [2, 5], [4, 6], [3, 8], [5, 5], [7, 7],
        ...      [1, 4], [2, 6], [5, 9]]
        >>> c = [2, 2, 2, 2, 2, 0, 1, 2, 2, 2]
        >>> model = CoxPH.fit(x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], c=c)
        >>> CoxNPMLEModel.from_dict(model.to_dict()).beta.round(4)
        array([-1.3267])
        """
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        require_model_tag(
            model_dict,
            "CoxNPMLEModel",
            "a proportional hazards model with an NPMLE baseline",
        )
        out = cls()
        out.causes = [
            None if k is None else label_from_native(k)
            for k in model_dict["causes"]
        ]
        out.single = bool(model_dict["single"])
        out.betas = np.array(model_dict["betas"], dtype=float)
        K, p = out.betas.shape
        out._se = np.array(model_dict["se"], dtype=float)
        out._covariance = np.array(
            model_dict["covariance"], dtype=float
        ).reshape(K * p, K * p)
        out.p_values = np.array(model_dict["p_values"], dtype=float)
        out.d = np.array(model_dict["d"], dtype=float).reshape(K, -1)
        out.left = np.array(model_dict["left"], dtype=float)
        out.right = np.array(model_dict["right"], dtype=float)
        out.allowed = np.array(model_dict["allowed"], dtype=bool).reshape(
            K, -1
        )
        out.report_lower = np.array(
            model_dict["report_lower"], dtype=float
        ).reshape(K, -1)
        tau = model_dict.get("tau")
        # (inf is written as null in strict JSON; its record restores it)
        out.tau = np.inf if tau is None else float(tau)
        out.center = np.array(model_dict["center"], dtype=float)
        out.log_likelihood = -float(model_dict["_neg_ll"])
        out._neg_ll = float(model_dict["_neg_ll"])
        out._ic_n = cls._restored_ic_n(model_dict)
        out.n_iter = int(model_dict.get("n_iter", 0))
        out.tol = float(model_dict.get("tol", 1e-10))
        out.maximum = restored_maximum(model_dict)
        restore_covariate_meta(out, model_dict)
        return out


def _neglog(S: npt.NDArray) -> npt.NDArray:
    with np.errstate(divide="ignore"):
        return 0.0 - np.log(S)


def _refit(model: CoxNPMLEModel, rows: _Rows) -> CoxNPMLEModel:
    """``model`` refitted to ``rows`` (a bootstrap sample), at the same
    centre convention, with no warnings of its own."""
    K = len(model.causes)
    present = np.unique(rows.cause[rows.cause >= 0])
    if present.size < K:
        raise ValueError("a cause has no failure in the sample")
    refit = fit_npmle(
        rows,
        model.causes,
        model.single,
        tol=model.tol,
        center=bool(np.any(model._center())),
        feature_names=model.feature_names,
        warn_tau=False,
        inference=False,
    )
    refit._covariance = None
    return refit
