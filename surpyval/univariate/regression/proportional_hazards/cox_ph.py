# This code was created for and sponsored by Cartiga (www.cartiga.com).
# Cartiga makes no representations or warranties in connection with the code
# and waives any and all liability in connection therewith. Your use of the
# code constitutes acceptance of these terms.

# Copyright 2022 Cartiga LLC

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt
from numpy.linalg import inv, pinv
from pandas import isna
from scipy.optimize import OptimizeResult, minimize, root
from scipy.stats import norm

if TYPE_CHECKING:
    import pandas as pd

from surpyval.univariate.nonparametric import (
    FlemingHarrington,
    KaplanMeier,
    NelsonAalen,
    Turnbull,
)
from surpyval.utils import (
    _caller_stacklevel,
    check_covariate_rows,
    finite_covariate_mask,
    is_missing_event,
    validate_coxph,
    validate_coxph_df_inputs,
)

from .._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    expand,
    warn_aliased,
)
from .._fit_skeleton import covariate_center
from ..regression_data import check_finite_event_times
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from ..tvc_fit import fit_tvc_df
from .tvc import handle_tvc, handle_tvc_timeline

nonparametric_dists = {
    "Nelson-Aalen": NelsonAalen,
    "Kaplan-Meier": KaplanMeier,
    "Fleming-Harrington": FlemingHarrington,
    "Turnbull": Turnbull,
}


def _baseline_method(tie_method: str) -> str:
    """The baseline-hazard estimator that goes with a tie method: Efron's
    tie correction for an Efron fit, Breslow's otherwise."""
    return "efron" if str(tie_method).lower() == "efron" else "breslow"


class _GroupBy:
    """Pure-NumPy grouped aggregation, replacing numpy_indexed.group_by.

    The multi-dimensional sum used to be ``np.add.at``, which is an
    unbuffered scatter with no fast path: at n=50 000 with ten covariates
    it was 6.3s of a 16.9s Efron fit, called about ten times per
    ``jac_hess`` on arrays of shape ``(n, p, p)`` (#329).

    Sorting once here turns each of those into ``np.add.reduceat``, a
    C-level segmented reduction. Every unique key has at least one member
    by construction, so the group starts strictly increase and
    ``reduceat`` is well defined.

    Two cases skip work entirely. When the keys already arrive grouped --
    the common case for start-stop (time-varying covariate) data -- there
    is nothing to permute. And when every key is distinct there is nothing
    to *add*: the sum is the permutation and no reduction is needed at
    all. That second case is continuous event times, where ``reduceat``
    would otherwise be asked for fifty thousand one-element segments and
    pay the per-segment overhead on every one of them.

    One consequence to know about: when both shortcuts apply at once --
    keys already grouped *and* all distinct -- ``sum`` returns the input
    array itself rather than a copy, because the sum of one-element groups
    in their existing order is the input. Treat the result as read-only.
    Every caller in this module builds its argument as a fresh temporary
    and only ever rebinds the result, never mutates it in place.
    """

    def __init__(self, keys: npt.NDArray) -> None:
        self.unique, self._inv = np.unique(keys, return_inverse=True)
        self._inv = np.asarray(self._inv).ravel()
        self._n = len(self.unique)

        counts = np.bincount(self._inv, minlength=self._n)
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._all_distinct = self._n == len(self._inv)

        already_grouped = bool(np.all(np.diff(self._inv) >= 0))
        self._order = (
            None if already_grouped else np.argsort(self._inv, kind="stable")
        )

    def sum(self, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        # ``asarray`` rather than ``astype``: no copy when the caller
        # already handed over float64, which it almost always does.
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            result = np.bincount(self._inv, weights=values, minlength=self._n)
        else:
            ordered = values if self._order is None else values[self._order]
            result = (
                ordered
                if self._all_distinct
                else np.add.reduceat(ordered, self._starts, axis=0)
            )
        return self.unique, result

    def max(self, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        values = np.asarray(values)
        result = np.full(self._n, -np.inf)
        np.maximum.at(result, self._inv, values)
        return self.unique, result


def _efron_tie_terms(n_d: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """The Efron terms of every event time, stored raggedly: for each of
    the ``int(d)`` tied deaths at time ``i``, the time's index and its
    weight ``c = j / d``, ``j = 0, ..., int(d) - 1``.

    The count ``d`` can be fractional -- ``n`` is a weight, not necessarily
    an integer -- and the loop this replaces ran ``range(int(d))`` while
    dividing by the unrounded ``d``. The number of terms therefore
    truncates and the weights do not; getting that backwards would
    silently change the Efron correction for weighted data.

    There are as many terms as deaths, so a per-time sum over them is a
    ``bincount`` of O(deaths) work. A dense ``(times x largest tie)``
    array padded with a mask, as used before, cost 1.5 million entries for
    one 51-way tie among 30 000 times, and ``(times x largest tie x p)``
    in the score (#515).

    Shared by the log-likelihood denominator, the score and the hessian so
    the three agree on the convention by construction.
    """
    n_d = np.asarray(n_d, dtype=float).reshape(-1)
    counts = np.where(n_d >= 1, n_d, 0).astype(int)
    idx = np.repeat(np.arange(len(n_d)), counts)
    first = np.cumsum(counts) - counts
    j = np.arange(idx.size) - first[idx]
    return idx, j / n_d[idx]


class _EfronTies:
    """Efron's tie terms of a fit (:func:`_efron_tie_terms`), worked out
    once: they depend on the deaths ``n_d`` only, not on ``beta``.

    ``one`` marks the times with a single death term (``c = 0``), where
    every Efron sum is its one term; ``tied`` those with more, and
    ``t_idx`` / ``t_c`` are the tied times' terms alone, ``t_idx``
    indexing ``tied``. The ragged sums then run over the tied deaths only,
    nothing at all on continuous data, and take the same additions in the
    same order as a sum over every death, so the results are unchanged to
    the last bit (#516).
    """

    def __init__(self, n_d: npt.NDArray) -> None:
        idx, c = _efron_tie_terms(n_d)
        counts = np.bincount(idx, minlength=len(n_d))
        self.one = np.flatnonzero(counts == 1)
        self.tied = np.flatnonzero(counts > 1)
        self.active = np.flatnonzero(counts >= 1)
        in_tie = counts[idx] > 1
        self.t_idx = np.searchsorted(self.tied, idx[in_tie])
        self.t_c = c[in_tie]

    def _sum(self, values: npt.NDArray) -> npt.NDArray:
        """Sum the tied deaths' ``values`` to their times."""
        return np.bincount(
            self.t_idx, weights=values, minlength=self.tied.size
        )

    def log_denominator(self, R: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
        """Per event time, ``sum_j log(R - c D)``; see
        :func:`efron_log_denominator`."""
        out = np.zeros(len(R))
        out[self.one] = np.log(R[self.one])
        if self.tied.size:
            Rt, Dt = R[self.tied], D[self.tied]
            out[self.tied] = self._sum(
                np.log(Rt[self.t_idx] - self.t_c * Dt[self.t_idx])
            )
        return out

    def sums(self, R: npt.NDArray, D: npt.NDArray) -> tuple[npt.NDArray, ...]:
        """At the tied times, ``sum u``, ``sum c u``, ``sum u^2``,
        ``sum c u^2`` and ``sum c^2 u^2`` over the tied deaths, with
        ``u = 1 / (R - c D)``: the scalars of the score
        (:meth:`expected`) and the information (:func:`_cox_information`)."""
        Rt, Dt = R[self.tied], D[self.tied]
        c = self.t_c
        u = 1.0 / (Rt[self.t_idx] - c * Dt[self.t_idx])
        u2 = u**2
        return (
            self._sum(u),
            self._sum(c * u),
            self._sum(u2),
            self._sum(c * u2),
            self._sum(c**2 * u2),
        )

    def expected(
        self,
        R: npt.NDArray,
        ZR: npt.NDArray,
        ZD: npt.NDArray,
        sums: tuple[npt.NDArray, ...],
    ) -> npt.NDArray:
        """Per event time, the expected covariate sum of the Efron score;
        see :func:`efron_jac`. ``sums`` is :meth:`sums` of ``R`` and
        ``D``."""
        out = np.zeros(ZR.shape)
        out[self.one] = ZR[self.one] / R[self.one, None]
        if self.tied.size:
            s_u, s_cu = sums[0][:, None], sums[1][:, None]
            out[self.tied] = s_u * ZR[self.tied] - s_cu * ZD[self.tied]
        return out


def efron_log_denominator(
    n_d: npt.NDArray, Ri: npt.NDArray, Di: npt.NDArray
) -> npt.NDArray:
    """Per event time, ``sum_j log(R - (j/d) D)`` over the ``d`` tied deaths.

    Where at most one death occurs, ``j`` only ever takes the value 0, so
    ``c = 0`` and this collapses to ``log(R)`` — Breslow's denominator. On
    continuous data that is every event time, which is why the Efron and
    Breslow fits agree digit for digit there; splitting the two cases out
    means that agreement no longer costs a Python loop (#329).
    """
    m = len(n_d)
    R = np.asarray(Ri).reshape(m)
    D = np.asarray(Di).reshape(m)
    return _EfronTies(n_d).log_denominator(R, D)


def efron_jac(
    n_d: npt.NDArray,
    Ri: npt.NDArray,
    ZRi: npt.NDArray,
    Di: npt.NDArray,
    ZDi: npt.NDArray,
) -> npt.NDArray:
    """Per event time, the expected covariate sum of the Efron score,
    ``sum_j (ZR - c ZD) / (R - c D)`` over the ``d`` tied deaths.

    As in :func:`_cox_information`, only ``c`` depends on ``j``, so with
    ``u = 1 / (R - c D)`` the sum factors into two scalars per time:

        sum_j (ZR - c ZD) u  =  (sum u) ZR - (sum c u) ZD.

    This used to be evaluated on a ``numpy.ma`` masked array of shape
    ``(times x largest tie x p)``: 12.9 s of fitting for 30 000 rows with
    one 51-way tie, against 1.15 s with none (#515). A time with a single
    death (c = 0) is ``ZR / R`` exactly as before, so untied data give the
    same score to the last bit; a time with no deaths contributes 0.
    """
    m = len(n_d)
    R = np.asarray(Ri).reshape(m)
    D = np.asarray(Di).reshape(m)
    ties = _EfronTies(n_d)
    sums = ties.sums(R, D) if ties.tied.size else ()
    return ties.expected(R, np.asarray(ZRi), np.asarray(ZDi), sums)


class _RiskSetRows:
    """Where each row of a fit sits on the event-time axis, for
    :func:`_cox_information`.

    ``x`` are the rows' exit times in increasing order (the generators sort
    them first), ``unique_x`` the distinct ones and ``tl`` the entry times.
    A row is at risk at the event times in ``(tl, x]``
    (:func:`cox_at_risk_mask`): those from index ``entered`` -- the number
    of distinct times at or before ``tl`` -- up to and including ``exit``.
    ``truncated`` is whether any row enters after the first time, i.e.
    whether the not-yet-entered terms are anything but exact zeros; when
    not, the generators skip them (#516).
    """

    def __init__(
        self, x: npt.NDArray, unique_x: npt.NDArray, tl: npt.NDArray
    ) -> None:
        self.exit = np.searchsorted(unique_x, x)
        self.truncated = bool(tl.size) and bool(tl.max() >= unique_x[0])
        self.entered = (
            np.searchsorted(unique_x, tl, side="right")
            if self.truncated
            else None
        )

    def over_risk_set(self, per_time: npt.NDArray) -> npt.NDArray:
        """For each row, the sum of ``per_time`` over the event times at
        which the row is at risk."""
        total = np.cumsum(per_time)
        if self.entered is None:
            return total[self.exit]
        total = np.concatenate([[0.0], total])
        return total[self.exit + 1] - total[self.entered]


def _cox_information(
    Z: npt.NDArray,
    rows: _RiskSetRows,
    risk_w: npt.NDArray,
    s_u: npt.NDArray,
    s_u2: npt.NDArray,
    ZR: npt.NDArray,
    efron: tuple[npt.NDArray, ...] | None = None,
) -> npt.NDArray:
    """The observed information (Hessian of the negative partial
    log-likelihood) of a Breslow or Efron fit, without a ``p x p`` array
    per event time (#516).

    Per event time ``i`` the information is (Efron's form, Breslow's being
    ``c = 0`` with ``d`` copies of the one term)

        sum_j (Z2R - c Z2D) u - a a' u^2,   a = ZR - c ZD,  u = 1/(R - c D),

    over the ``d`` tied deaths, ``c = j / d``. The sum over ``j`` factors
    into the scalars ``s_u = sum u``, ``s_cu``, ``s_u2``, ``s_cu2`` and
    ``s_c2u2`` (#329, #515), so it is

        s_u Z2R - s_cu Z2D - s_u2 ZR ZR'
            + s_cu2 (ZR ZD' + ZD ZR') - s_c2u2 ZD ZD'.

    The ``Z2`` sums are where the cost was: ``Z2R`` sums ``w z z'`` over the
    risk set, which used to be built per event time as an ``(times x p x
    p)`` array (and again for the not-yet-entered rows and ``Z2D``) and
    weighted afterwards. Summed over the event times first,

        sum_i s_u(i) Z2R(i) = sum_k w_k z_k z_k' sum_{i: k at risk} s_u(i),

        sum_i s_cu(i) Z2D(i) = sum_{k dies} w_k z_k z_k' s_cu(i_k),

    which is one ``Z' diag(q) Z`` over the rows, with ``q`` from a
    cumulative sum of ``s_u`` (:meth:`_RiskSetRows.over_risk_set`). The
    outer-product terms are ``(times x p)`` matrix products.

    ``risk_w`` is each row's risk weight ``n exp(beta'Z)``, ``s_u`` is per
    event time, and ``s_u2`` and the ``Z``-weighted risk sums ``ZR`` are
    those of the times with a death only. ``efron`` carries Efron's tie
    terms, ``(death_w, s_cu, s_cu2, s_c2u2, ZR, ZD)``: the rows' death
    weight ``n_d exp(beta'Z)``, ``s_cu`` per event time, and the rest at
    the tied times only (elsewhere ``c = 0`` and they vanish); it is
    ``None`` for Breslow, or when no event time is tied.

    The not-yet-entered rows of left-truncated (start-stop) data need no
    term of their own: each row collects ``s_u`` only over the times at
    which it is at risk.
    """
    q = risk_w * rows.over_risk_set(s_u)
    if efron is not None:
        death_w, s_cu, s_cu2, s_c2u2, ZR_t, ZD_t = efron
        q = q - death_w * s_cu[rows.exit]
    info = (Z.T * q) @ Z - (ZR.T * s_u2) @ ZR
    if efron is not None:
        cross = (ZR_t.T * s_cu2) @ ZD_t
        info = info + cross + cross.T - (ZD_t.T * s_c2u2) @ ZD_t
    # The products above are symmetric only to rounding.
    return (info + info.T) / 2


def _sort_by_event_time(
    x: npt.NDArray,
    Z: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Put the rows in event-time order before building the closures.

    Nothing in the partial likelihood depends on the order of the rows --
    every quantity is aggregated to unique event times first -- but
    ``_GroupBy`` gets to skip its permutation when the keys already arrive
    grouped. One reordering of ``Z`` here replaces a gather of the per-row
    arrays on every ``jac_hess`` call (#329).

    The caller keeps the unsorted arrays: ``fit`` stores those on the model
    for the residual and diagnostic code, and the closures only ever hand
    back beta-shaped or unique-time-shaped results.
    """
    order = np.argsort(x, kind="stable")
    return x[order], Z[order], c[order], n[order], tl[order]


def at_risk_beta_Z(
    arr: npt.NDArray, n: npt.NDArray, gb_x: "_GroupBy"
) -> npt.NDArray:
    R = gb_x.sum(n * arr)[1]
    # Get the reverse cumulative sum
    return R[::-1].cumsum(axis=0)[::-1]


def not_yet_entered(pos: npt.NDArray, mass_by_tl: npt.NDArray) -> npt.NDArray:
    """Per unique event time, the total ``mass_by_tl`` of observations whose
    entry (left-truncation) time is at or after that event time — the amount
    to subtract from the reverse-cumulative at-risk sums so that a subject
    only enters the risk set strictly after its ``tl``.

    ``pos`` is ``searchsorted(unique_tl, unique_x, side="left")``. This is an
    exact suffix-sum gather and is valid for *signed* quantities (the
    Z-weighted score and information sums), unlike the previous scatter +
    ``minimum.accumulate`` forward fill, which is only a forward fill for
    positive non-increasing sequences and silently corrupted the gradient and
    Hessian of every delayed-entry / start-stop fit containing a negative
    covariate value (#250).
    """
    suffix = mass_by_tl[::-1].cumsum(axis=0)[::-1]
    pad = np.zeros((1,) + suffix.shape[1:])
    return np.concatenate([suffix, pad], axis=0)[pos]


def _sub(a: "npt.ArrayLike | None", mask: npt.NDArray) -> "npt.NDArray | None":
    """Index ``a`` by ``mask``, passing ``None`` through unchanged."""
    if a is None:
        return None
    return np.asarray(a)[mask]


def _strata_labels(strata: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """The stratum labels as an array, and a mask of the missing ones.

    A label is missing when it is ``None``, ``NaN`` or pandas ``NA``. A
    list is read element by element: ``np.asarray(["a", np.nan])`` would
    turn the ``NaN`` into the string ``"nan"``, a stratum of its own.
    Missing entries of a list are filled with a present label, so the
    array keeps the dtype the present labels alone would give; they are
    dropped by the mask before the labels are used.
    """
    if isinstance(strata, (list, tuple)):
        values = list(strata)
        missing = np.array([is_missing_event(v) for v in values], dtype=bool)
        present = [v for v, m in zip(values, missing) if not m]
        fill = present[0] if present else 0
        arr = np.asarray([fill if m else v for v, m in zip(values, missing)])
        return arr, missing
    arr = np.asarray(strata)
    missing = np.asarray(isna(arr), dtype=bool).reshape(arr.shape)
    return arr, missing


def _kp_tie_term(
    eta: npt.NDArray, Z: npt.NDArray, d: int, derivs: bool = True
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """``log e_d`` of the risk-set scores ``exp(eta)``, with its gradient and
    Hessian in ``beta`` (``eta = Z @ beta`` over the risk set).

    ``e_d`` is the ``d``-th elementary symmetric polynomial -- the sum, over
    every ``d``-subset of the risk set, of the product of its scores -- which
    is the denominator of the Kalbfleisch-Prentice (discrete) tie term. It is
    evaluated by the Gail, Lubin & Rubinstein (1981) recursion over the risk
    set that R's ``coxph(ties="exact")`` also uses: with ``B_k(j)`` the value
    of ``e_k`` over the first ``j`` members,

        B_k(j) = B_k(j - 1) + r_j B_{k-1}(j - 1),

    and the same recursion differentiated once and twice gives the score and
    information. For fixed ``k`` that is a cumulative sum over ``j``, so the
    Python loop runs ``d`` times over vectorised risk-set arrays, O(d m p^2)
    in all. The same recursion used to run as a scalar autograd trace of
    ``m * d`` Python-level operations, re-traced for every gradient, which
    took minutes on a tie set of a hundred.

    Each row ``k`` is rescaled by its largest entry (the running ``log_scale``
    keeps the value) so ``e_d`` cannot overflow; derivatives are carried in
    the same scale, and only their ratios to ``e_d`` are used.
    """
    m, p = Z.shape
    if d > m - d:
        # e_d(v) = prod(v) * e_{m-d}(1/v): the complement runs fewer
        # iterations, and when every member of the risk set dies (d = m) it
        # runs none at all.
        log_e, g, h = _kp_tie_term(-eta, -Z, m - d, derivs)
        return float(eta.sum()) + log_e, Z.sum(axis=0) + g, h

    shift = float(eta.max()) if m else 0.0
    r = np.exp(eta - shift)
    B = np.ones(m + 1)
    dB = np.zeros((m + 1, p))
    d2B = np.zeros((m + 1, p, p))
    ZZ = Z[:, :, None] * Z[:, None, :] if derivs else None
    log_scale = 0.0
    for _ in range(d):
        Bp = B[:-1]
        B = np.concatenate([[0.0], np.cumsum(r * Bp)])
        if derivs:
            dBp, d2Bp = dB[:-1], d2B[:-1]
            t1 = dBp + Z * Bp[:, None]
            t2 = (
                d2Bp
                + Z[:, :, None] * dBp[:, None, :]
                + dBp[:, :, None] * Z[:, None, :]
                + ZZ * Bp[:, None, None]
            )
            dB = np.concatenate(
                [np.zeros((1, p)), np.cumsum(r[:, None] * t1, axis=0)]
            )
            d2B = np.concatenate(
                [
                    np.zeros((1, p, p)),
                    np.cumsum(r[:, None, None] * t2, axis=0),
                ]
            )
        # The cumulative sums only add non-negative terms, so the last entry
        # is the largest.
        scale = B[-1]
        log_scale += np.log(scale)
        B = B / scale
        if derivs:
            dB = dB / scale
            d2B = d2B / scale

    log_e = log_scale + d * shift
    if not derivs:
        return log_e, np.zeros(p), np.zeros((p, p))
    g = dB[-1] / B[-1]
    return log_e, g, d2B[-1] / B[-1] - np.outer(g, g)


def _weighted_moments(
    eta: npt.NDArray, Z: npt.NDArray
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """``log sum exp(eta)`` with the ``exp(eta)``-weighted mean and
    covariance of the rows of ``Z``."""
    shift = eta.max()
    w = np.exp(eta - shift)
    total = w.sum()
    w = w / total
    mean = w @ Z
    Zc = Z - mean
    cov = (w[:, None] * Zc).T @ Zc
    return float(shift + np.log(total)), mean, cov


# DeLong et al. integrand, integrated over ``w = log t``: points further than
# this many log-units below the mode contribute below 1e-26 relative, and the
# coarse grid used to bracket the mode steps by ``_EXACT_COARSE_STEP``.
_EXACT_DROP = 60.0
_EXACT_COARSE_STEP = 0.1
_EXACT_NODES = 401


def _exact_tie_term(
    eta_d: npt.NDArray,
    Z_d: npt.NDArray,
    eta_w: npt.NDArray,
    Z_w: npt.NDArray,
    derivs: bool = True,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """``log`` of the exact (average-over-orderings) tie contribution, with
    its gradient and Hessian in ``beta``.

    For ``d`` tied deaths with scores ``a_j = exp(eta_j)`` and the rest of the
    risk set scoring ``W = sum exp(eta_w)``, the sum over the ``d!`` orderings
    of the sequential Cox terms equals (DeLong, Guirguis & So 1994)

        L = int_0^inf prod_j (1 - exp(-a_j t / W)) exp(-t) dt,

    the formula SAS uses for ``TIES=EXACT``. This replaces an O(2^d) subset
    recursion that was capped at twelve ties and still took tens of seconds
    to fit. In ``w = log t`` the log-integrand

        g(w) = sum_j log(1 - exp(-c_j e^w)) - e^w + w,   c_j = a_j / W,

    is concave, so the integrand is a single smooth bump. A coarse grid
    brackets the region within ``_EXACT_DROP`` log-units of its peak and the
    trapezoid rule on a fine grid there -- spectrally accurate for a smooth,
    negligible-at-the-ends integrand -- gives ``L`` to machine precision. The
    score and information follow by differentiating under the integral:
    with ``E`` the expectation over the normalised integrand,

        d log L = E[dg],   d2 log L = E[d2g] + Var[dg].
    """
    d, p = Z_d.shape
    if eta_w.size == 0:
        # Everyone left at risk dies: every ordering's product telescopes
        # and the orderings sum to exactly one.
        return 0.0, np.zeros(p), np.zeros((p, p))

    lse_w, mean_w, cov_w = _weighted_moments(eta_w, Z_w)
    log_c = eta_d - lse_w

    def log_integrand(w: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.exp(log_c[None, :] + w[:, None])
        with np.errstate(divide="ignore"):
            # log(1 - e^-x); x can underflow to 0 far left of the mode,
            # where the log is -inf and the node simply carries no weight.
            g = np.log(-np.expm1(-x)).sum(axis=1) - np.exp(w) + w
        return g, x

    # The mode lies in (0, log(d + 1)): g' = sum x/(e^x - 1) - e^w + 1 with
    # every summand in (0, 1). g falls at least as fast as w - e^w to the left
    # and as (d + 1)(w - e^w) to the right, so this range contains every point
    # within _EXACT_DROP of the peak.
    coarse = np.arange(
        -(_EXACT_DROP + 2.0), np.log(d + 1.0) + 4.0, _EXACT_COARSE_STEP
    )
    g_coarse = log_integrand(coarse)[0]
    # g is concave, so the points above the threshold form one run; a coarse
    # point below it bounds the region from outside on each side.
    inside = np.flatnonzero(g_coarse >= g_coarse.max() - _EXACT_DROP)
    lo = coarse[max(inside[0] - 1, 0)]
    hi = coarse[min(inside[-1] + 1, coarse.size - 1)]

    nodes = np.linspace(lo, hi, _EXACT_NODES)
    step = nodes[1] - nodes[0]
    g, x = log_integrand(nodes)
    g_max = g.max()
    weight = np.exp(g - g_max)
    total = weight.sum()
    log_L = float(g_max + np.log(step * total))
    if not derivs:
        return log_L, np.zeros(p), np.zeros((p, p))

    # q = x / (e^x - 1) = d/dlog(x) of log(1 - e^-x), written to stay finite
    # for large x; q -> 1 as x -> 0 (a node that underflowed to x = 0).
    one_minus = -np.expm1(-x)
    positive = x > 0
    safe = np.where(positive, one_minus, 1.0)
    q = np.where(positive, x * np.exp(-x) / safe, 1.0)
    # x q'(x), the second log-derivative, is q (1 - x / (1 - e^-x)).
    xq = np.where(positive, q * (1.0 - x / safe), 0.0)

    Zc = Z_d - mean_w
    dg = q @ Zc
    d2g = (
        np.einsum("nd,dp,dq->npq", xq, Zc, Zc)
        - q.sum(axis=1)[:, None, None] * cov_w
    )
    prob = weight / total
    mean_dg = prob @ dg
    hess = (
        np.einsum("n,npq->pq", prob, d2g)
        + np.einsum("n,np,nq->pq", prob, dg, dg)
        - np.outer(mean_dg, mean_dg)
    )
    return log_L, mean_dg, hess


def _cox_aliased(
    info: npt.NDArray,
    Z: npt.NDArray,
    n: npt.NDArray,
    n_events: float,
    strata: "npt.NDArray | None" = None,
) -> npt.NDArray:
    """The columns whose coefficients the partial likelihood cannot
    determine (#409, #476), to be aliased (see
    :mod:`surpyval.univariate.regression._aliasing`).

    ``info`` is the information at ``beta = 0``: the sum over the event
    times of the risk sets' covariate covariance. A direction with no
    information is one in which the covariates do not vary within any
    risk set at an event time, and then the partial likelihood does not
    depend on it at all: a constant column (a Cox model has no
    intercept), one constant within each stratum, collinear columns
    (every level of a factor coded, with no intercept), or every unit at
    risk failing at once. Fitted anyway, a coefficient ran off to 5.5e14
    and every prediction was nan, with a misleading "monotone likelihood"
    warning. ``Z`` are the covariates with counts ``n`` (a column's
    spread over the data is what its information is judged against), and
    ``strata`` the stratum labels; with no event nothing is checked.
    """
    info = np.atleast_2d(np.asarray(info, dtype=float))
    p = info.shape[0]
    if not n_events > 0 or p == 0:
        return np.array([], dtype=int)
    Z = np.asarray(Z, dtype=float)
    n = np.asarray(n, dtype=float).reshape(-1)
    Zc = Z - covariate_center(Z, n)
    spread = n_events * (n @ Zc**2) / n.sum()
    return aliased_columns(
        info, Z.shape[0], constant_columns(Z, strata), spread
    )


_NEWTON_MAX_ITER = 20
_NEWTON_MAX_HALVINGS = 40


def _newton_raphson(
    neg_ll: Callable,
    jac: Callable,
    beta: npt.NDArray,
    tol: float,
    score: npt.NDArray,
    hess: npt.NDArray,
) -> "OptimizeResult | None":
    """Newton-Raphson with step-halving on the negative partial
    log-likelihood, the standard Cox algorithm (R's ``coxph``, lifelines):
    a handful of information matrices where ``root(hybr)`` built one for
    each of its ~16 evaluations (#516).

    ``score`` and ``hess`` are ``jac(beta)`` at the start. A step that
    does not decrease ``neg_ll`` (beyond rounding) is halved. The iteration
    has converged when a full step is at most ``tol`` in size measured
    by the information, ``sqrt(step' H step) <= tol`` -- ``tol`` standard
    errors, so the scale of the covariates does not matter, and the error
    left after that step is of the order of its square. Rounding stops
    the step shrinking near ``1e-16 sqrt(events)``; a step that has
    stopped shrinking once below ``1e-8`` is taken as converged there.

    Returns ``None`` -- the caller then uses the root-finder -- when an
    information matrix is singular or not finite, when halving finds no
    decrease, or after ``_NEWTON_MAX_ITER`` steps, which is where a
    likelihood with no finite maximum (#392) ends up. Otherwise the
    result carries ``jac`` and ``hess`` at the solution.
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    f = float(neg_ll(beta))
    if not np.isfinite(f):
        return None
    lam_prev = np.inf
    for it in range(1, _NEWTON_MAX_ITER + 1):
        score = np.atleast_1d(score)
        hess = np.atleast_2d(hess)
        if not (np.all(np.isfinite(score)) and np.all(np.isfinite(hess))):
            return None
        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError:
            return None
        lam = float(np.sqrt(np.abs(score @ step)))
        if not (np.all(np.isfinite(step)) and np.isfinite(lam)):
            return None
        slack = 1e3 * np.finfo(float).eps * max(abs(f), 1.0)
        t = 1.0
        for _ in range(_NEWTON_MAX_HALVINGS):
            new = beta - t * step
            f_new = float(neg_ll(new))
            if np.isfinite(f_new) and f_new <= f + slack:
                break
            t /= 2
        else:
            return None
        beta, f = new, f_new
        score, hess = jac(beta)
        full = t == 1.0
        if full and (lam <= tol or (lam <= 1e-8 and lam >= lam_prev / 2)):
            return OptimizeResult(
                x=beta,
                fun=f,
                jac=np.atleast_1d(score),
                hess=np.atleast_2d(hess),
                nit=it,
                success=True,
                status=0,
                message="Newton-Raphson converged",
            )
        lam_prev = lam if full else np.inf
    return None


def _solve_beta_and_p_values(
    neg_ll: Callable,
    jac: Callable,
    beta_init: npt.NDArray,
    tol: float,
    Z: npt.NDArray,
    n: npt.NDArray,
    n_events: float,
    strata: "npt.NDArray | None" = None,
) -> tuple[Any, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Maximise the partial likelihood by Newton-Raphson
    (:func:`_newton_raphson`; the score's root-finder, then BFGS, if that
    fails) and compute Wald p-values from the observed information;
    shared by ``fit`` and ``_fit_stratified`` so the most-patched block
    in this file exists exactly once. The covariates ``Z``, counts ``n``,
    weighted number of events and stratum labels are for the aliasing
    check (:func:`_cox_aliased`).

    Returns ``(res, p_values, se, aliased)``: ``res.x`` has 0 at the
    aliased columns (the coefficients the predictions use), and their
    p-values and standard errors ``se`` are nan."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_at_start, info_at_start = jac(beta_init)
    p = len(np.atleast_1d(beta_init))
    aliased = _cox_aliased(info_at_start, Z, n, n_events, strata)
    kept = np.setdiff1d(np.arange(p), aliased)
    if aliased.size:
        warn_aliased(
            aliased,
            "the partial likelihood does not depend on them, as they do "
            "not vary within the risk sets at the event times beyond a "
            "combination of the other columns (a constant column -- a Cox "
            "model has no intercept --, one constant within each stratum, "
            "or a linear combination of the others, as the columns of "
            "every level of a factor are)",
        )
        full_neg_ll, full_jac = neg_ll, jac

        def embed(b: npt.NDArray) -> npt.NDArray:
            out = np.zeros(p)
            out[kept] = b
            return out

        def neg_ll(b: npt.NDArray) -> float:
            return full_neg_ll(embed(b))

        def jac(b: npt.NDArray) -> tuple:
            score, hess = full_jac(embed(b))
            score = np.atleast_1d(score)
            hess = np.atleast_2d(hess)
            return score[kept], hess[np.ix_(kept, kept)]

        info_at_start = np.atleast_2d(info_at_start)[np.ix_(kept, kept)]
        score_at_start = np.atleast_1d(score_at_start)[kept]
        beta_init = np.asarray(beta_init, dtype=float)[kept]
        if kept.size == 0:
            res = OptimizeResult(x=np.zeros(p), success=True, fun=0.0)
            return res, np.full(p, np.nan), np.full(p, np.nan), aliased
    # Where the likelihood is monotone (below) the coefficients run off
    # towards infinity and the risk-set sums underflow to 0 on the way;
    # the resulting log(0) and 0/0 are that divergence, which is reported
    # by name below, not as a stream of RuntimeWarnings.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = _newton_raphson(
            neg_ll, jac, beta_init, tol, score_at_start, info_at_start
        )
        if res is not None:
            hessian_matrix = res.hess
        else:
            # Newton-Raphson gave up: a singular information (no events),
            # no decrease on halving, or a likelihood with no finite
            # maximum (#392). The score's root-finder (the solver before
            # #516) takes over; ``jac`` returns (score, hessian), hence
            # ``jac=True``.
            res = root(jac, beta_init, jac=True, tol=tol)

            # MINPACK's hybr root-finder can stall on delayed-entry data
            # with staggered risk sets (e.g. the start-stop representation
            # used for time-varying covariates) even though the partial
            # log-likelihood is well behaved there. Fall back to a direct
            # minimisation of the negative partial log-likelihood whenever
            # root-finding fails to converge or lands at a worse point, so
            # such fits still succeed.
            if not res.success:
                fallback = minimize(
                    lambda b: float(neg_ll(b)), beta_init, method="BFGS"
                )
                if float(neg_ll(fallback.x)) < float(neg_ll(res.x)):
                    res = fallback

            hessian_matrix = jac(res.x)[1]
    _warn_if_monotone(hessian_matrix, info_at_start, kept)
    # An exactly singular information matrix raises before the
    # pseudo-inverse fallback can run (#259); route it there.
    try:
        var = np.diag(inv(hessian_matrix))
    except np.linalg.LinAlgError:
        var = np.full(len(np.atleast_1d(res.x)), -1.0)
    # Use the pseudo-inverse if the hessian does not have a diagonal that
    # is all positive.
    if np.any(var <= 0):
        var = np.diag(pinv(hessian_matrix))
    # A near-singular information matrix (e.g. a degenerate start-stop
    # design with duplicated rows) can still leave a non-positive
    # variance; the resulting standard error is simply unavailable (nan),
    # which is the correct signal, so suppress the sqrt-of-negative
    # warning rather than emit it.
    with np.errstate(invalid="ignore"):
        se = np.sqrt(var)
        z_score = res.x / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_score)))
    if aliased.size:
        res.x = embed(res.x)
        p_values = expand(p_values, kept, p)
        se = expand(se, kept, p)
    return res, p_values, se, aliased


def _warn_if_monotone(
    info: npt.NDArray,
    info_at_start: npt.NDArray,
    columns: "npt.NDArray | None" = None,
) -> None:
    """Warn when the partial likelihood has no finite maximum.

    When a covariate separates the events from the survivors (every
    failure at each event time has the largest -- or smallest -- value in
    its risk set), the partial likelihood keeps increasing as that
    coefficient grows, and the fit stops wherever the optimiser gave up
    (``beta`` of 35 with a p-value of 1 on such data). The symptom is that
    the information for that coefficient has collapsed: the risk sets'
    weighted covariate variance goes to 0 as the coefficient grows.
    ``columns`` are the columns of ``Z`` the matrices are for (all of
    them by default; the identified ones after aliasing).
    """
    d = np.diag(np.atleast_2d(info))
    d0 = np.diag(np.atleast_2d(info_at_start))
    diverged = np.flatnonzero(
        (d0 > 0) & ~(np.nan_to_num(d, nan=0.0) > 1e-8 * d0)
    )
    if diverged.size:
        if columns is not None:
            diverged = np.asarray(columns)[diverged]
        warn_monotone(str(diverged.tolist()))


def warn_monotone(which: str) -> None:
    """Warn that the partial likelihood has no finite maximum in the
    coefficients ``which`` names (``"[0]"``, or ``"[0] (cause 'a')"``);
    shared with the Fine-Gray fit, a weighted partial likelihood (#392)."""
    warnings.warn(
        "Monotone partial likelihood: it keeps increasing as coefficient"
        "(s) {} grow without bound, so the estimate is infinite (the "
        "covariate separates the events from the survivors). The "
        "reported value, its standard error and its p-value are "
        "meaningless; consider removing or coarsening the covariate, "
        "or a penalised fit.".format(which),
        stacklevel=_caller_stacklevel(),
    )


def _combine_generators(gens: list) -> tuple[Callable, Callable]:
    """Sum per-stratum ``(log_like, jac_hess)`` generators into one.

    The Cox partial likelihood factorises across strata: with a separate
    baseline hazard per stratum and a *shared* coefficient vector, the total
    log-likelihood (and hence its score and observed information) is the sum
    of the per-stratum contributions. Risk sets never cross a stratum
    boundary because each stratum's score/information is built only from its
    own observations.
    """

    def neg_ll(beta: npt.NDArray) -> float:
        return sum(g[0](beta) for g in gens)

    def jac_hess(beta: npt.NDArray) -> tuple:
        jac_total = None
        hess_total = None
        for g in gens:
            j, h = g[1](beta)
            jac_total = j if jac_total is None else jac_total + j
            hess_total = h if hess_total is None else hess_total + h
        return jac_total, hess_total

    return neg_ll, jac_hess


_LOG_MAX = float(np.log(np.finfo(float).max))
_TINY = float(np.finfo(float).tiny)


def _baseline_at_origin(
    beta: npt.NDArray,
    center: npt.NDArray,
    Z: npt.NDArray,
    r: npt.NDArray,
    h0: npt.NDArray,
    what: str = "baseline hazard",
) -> "tuple[npt.NDArray, npt.NDArray]":
    """The risk weights ``r`` and baseline increments ``h0`` fitted at the
    covariate ``center`` moved to ``Z = 0`` (#463), as R's
    ``basehaz(fit, centered = FALSE)``: ``h0 * exp(-beta'center)`` and
    ``r * exp(beta'center)``, computed on the log scale.

    Refused, with a ``ValueError`` that points to ``center=True``, where
    that is not representable: a positive increment underflows (to 0 or a
    subnormal number) or overflows, a risk weight does, or ``exp(beta'Z)``
    overflows on the fitted rows ``Z`` -- the covariates are too far from
    0 for a baseline there to mean anything in floating point.
    """
    beta = np.asarray(beta, dtype=float)
    shift = float(np.dot(beta, center))
    with np.errstate(all="ignore"):
        lp = np.asarray(Z, dtype=float) @ beta
        h0_0 = np.exp(np.log(h0) - shift)
        r_0 = np.exp(np.log(r) + shift)
    ok = (
        bool(np.all(np.abs(lp) < _LOG_MAX))
        and bool(np.all(np.isfinite(h0_0)) and np.all(np.isfinite(r_0)))
        and bool(np.all(h0_0[h0 > 0] >= _TINY))
        and bool(np.all(r_0[r > 0] >= _TINY))
    )
    if not ok:
        raise ValueError(
            "The {} at Z = 0 cannot be represented for these covariates: "
            "their means are {} and the linear predictor there is "
            "beta'center = {:.4g}, so the baseline at Z = 0 is exp({:.4g}) "
            "times that at the means, which over- or underflows (as it "
            "does when a covariate separates the events, and the "
            "coefficients run off towards infinity). Fit with center=True "
            "to report the baseline at the covariate means (model.center) "
            "instead, or move the covariates nearer 0.".format(
                what,
                np.array2string(np.asarray(center), precision=4),
                shift,
                -shift,
            )
        )
    return r_0, h0_0


def cox_at_risk_mask(
    x: npt.NDArray, tl: npt.NDArray, tau: float
) -> npt.NDArray:
    """The Cox risk-set convention, in one place (#299): a row is at risk
    at event time ``tau`` once it has entered (``tl < tau`` — strict, so a
    start-stop row is not at risk at its own entry time) and until it
    exits (``x >= tau`` — inclusive, so a row is at risk at its own event
    or censoring time)."""
    return (tl < tau) & (x >= tau)


class CoxPH_:
    """
    The Cox proportional hazards model: a baseline hazard left entirely
    to the data, multiplied by :math:`e^{\\beta' Z}`,

    .. math::
        h(x \\mid Z) = h_0(x)\\, e^{\\beta' Z}.

    The coefficients are estimated from the partial likelihood (with a
    choice of tie handling, Efron's by default) and the baseline by the
    Breslow estimator, with Efron's tie correction after an Efron fit.
    As in R's ``coxph``, the fit centres the covariates on their
    (``n``-weighted) means, which leaves the coefficients unchanged and
    keeps a covariate far from 0 (a year, a date) from overflowing
    :math:`e^{\\beta' Z}`. The baseline is then reported at ``Z = 0``
    (R's ``basehaz(fit, centered = FALSE)``), or, with ``center=True``, at
    the means, stored as the model's ``center``. Supports right censoring, left
    truncation (delayed entry), stratification and time-varying
    covariates in start-stop form; left- and interval-censored data are
    refused, as the partial likelihood has no term for them (use a
    parametric regression model).
    ``CoxPH`` is an instance of this class; its fit methods return a
    :class:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel`.
    """

    # Best reference I can find that covers all the
    # possibilities for estimating betas
    # http://www-personal.umich.edu/~yili/lect4notes.pdf

    def baseline(
        self,
        beta: npt.NDArray,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        Z: npt.NDArray,
        tl: "npt.NDArray | None" = None,
        tie_method: str = "breslow",
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        # Baseline hazard increments at each distinct time -- the hazard of
        # a unit whose covariates are 0 in the ``Z`` given; ``fit`` passes
        # the centred covariates (#459), and moves the result to Z = 0
        # unless ``center=True`` (#463) -- returned with the risk weight
        # ``r`` and the deaths ``d``. Breslow's increment is
        # ``d / r``. With Efron ties (``tie_method="efron"``) the ``m`` tied
        # deaths at a time see the risk set step down, ``r - (l / m) * r_D``
        # for ``l = 0 .. m-1`` with ``r_D`` the tied deaths' own weight, and
        # the increment is ``sum_l 1 / (r - (l / m) * r_D)``: the
        # covariate-weighted Fleming-Harrington estimator, and the baseline
        # that matches the Efron partial likelihood (R's ``survfit.coxph``
        # does the same). Without ties the two are identical.
        #
        # The risk set at each event time ``tau_i``
        # follows ``cox_at_risk_mask`` (entered ``tl < tau_i``, not yet
        # exited ``x >= tau_i``), each row weighted by its count ``n`` and
        # hazard multiplier ``exp(Z'beta)``. Respecting ``tl`` is what makes
        # the baseline correct for left-truncated and time-varying-covariate
        # (start-stop) data. Computed by suffix sums — the risk set is
        # everyone with ``x >= tau_i`` minus the not-yet-entered
        # ``tl >= tau_i`` (valid because ``tl < x`` on every row) — the same
        # subtraction the Efron generator uses, replacing the previous
        # O(K·N) Python loop (#299).

        unique_x = np.unique(x)
        if tl is None:
            tl = np.full(x.shape[0], -np.inf)

        w = n * np.exp(Z @ beta)

        event = c == 0
        d = np.zeros_like(unique_x)
        np.add.at(d, np.searchsorted(unique_x, x[event]), n[event])

        r_exit = np.zeros_like(unique_x)
        np.add.at(r_exit, np.searchsorted(unique_x, x), w)
        r_exit = r_exit[::-1].cumsum()[::-1]

        # Bucket each row at the largest event time <= its entry time; the
        # suffix sum then gives, at each tau_i, the weight not yet entered.
        k = np.searchsorted(unique_x, tl, side="right") - 1
        entered_late = k >= 0
        r_pre = np.zeros_like(unique_x)
        np.add.at(r_pre, k[entered_late], w[entered_late])
        r_pre = r_pre[::-1].cumsum()[::-1]
        r = r_exit - r_pre

        with np.errstate(divide="ignore", invalid="ignore"):
            h0 = d / r
        if str(tie_method).lower() == "efron":
            r_tied = np.zeros_like(unique_x)
            np.add.at(r_tied, np.searchsorted(unique_x, x[event]), w[event])
            for t in np.flatnonzero(d > 1):
                m = int(round(float(d[t])))
                steps = r[t] - (np.arange(m) / m) * r_tied[t]
                h0[t] = np.sum(1.0 / steps)
        return unique_x, r, d, h0

    def create_efron_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        # The reference used to compute the jacobian and hessian
        # was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
        # Left-truncation is handled by subtracting the pre-entry risk set
        # (``Ri - TRi``) below, so delayed-entry data is fitted correctly.

        x, Z, c, n, tl = _sort_by_event_time(x, Z, c, n, tl)

        # Groupby object for repeated use
        gb_x = _GroupBy(x)
        gb_tl = _GroupBy(tl)
        n_d_x = np.where(c == 0, n, 0)
        n_d = gb_x.sum(n_d_x)[1]
        death_n = n_d_x
        risk_n = n
        n_d_x = n_d_x.reshape(-1, 1)
        n = n.reshape(-1, 1)

        x_ = gb_x.unique
        x_tl = gb_tl.unique
        # For each unique event time, how many unique entry times precede it:
        # feeds the not-yet-entered suffix-sum gather below.
        pos = np.searchsorted(x_tl, x_, side="left")
        rows = _RiskSetRows(x, x_, tl)

        # Efron's tie terms depend on the deaths only.
        m = len(x_)
        ties = _EfronTies(n_d)
        one, tied = ties.one, ties.tied

        def log_like(beta: npt.NDArray) -> float:
            beta_z = Z @ beta

            S_d = gb_x.sum(n_d_x * beta_z.reshape(-1, 1))[1].reshape(-1, 1)
            e_beta_z = np.exp(beta_z).reshape(-1, 1)

            x_, Ri = gb_x.sum(n * e_beta_z)

            Ri = Ri[::-1].cumsum(axis=0)[::-1]

            # Subtract the not-yet-entered mass from the risk sums.
            if rows.truncated:
                Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])

            Di = gb_x.sum(n_d_x * e_beta_z)[1]

            efron_denom = ties.log_denominator(Ri.reshape(m), Di.reshape(m))

            like = S_d.sum() - efron_denom.sum()
            return -like

        S_d = gb_x.sum(n_d_x * Z)[1]

        def jac_hess(beta: npt.NDArray) -> tuple:
            # This line troubled me for longer than I care
            # to admit. I was using n, but it is only the
            # number of deaths at each point, n_d_x

            # Only call this once.. Yay.
            beta_z = Z @ beta

            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            z_e_beta_z = Z * e_beta_z

            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)
            ZRi = at_risk_beta_Z(z_e_beta_z, n, gb_x)

            # Subtract the not-yet-entered mass from the risk sums. The
            # Z-weighted sums are signed, so this must be the exact gather —
            # see ``not_yet_entered`` (#250).
            if rows.truncated:
                Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])
                ZRi = ZRi - not_yet_entered(pos, gb_tl.sum(n * z_e_beta_z)[1])

            Di = gb_x.sum(n_d_x * e_beta_z)[1]
            ZDi = gb_x.sum(n_d_x * z_e_beta_z)[1]

            # As ``efron_jac``, with the tie sums kept for the information.
            R = Ri.reshape(m)
            D = Di.reshape(m)
            sums = ties.sums(R, D) if tied.size else ()
            expected_S_d = ties.expected(R, ZRi, ZDi, sums)

            diff = S_d - expected_S_d
            jacobian = -diff.sum(axis=0)

            # Observed information (Hessian of the negative log-likelihood),
            # positive definite, so ``inv(hess)`` gives the parameter
            # covariance directly; see ``_cox_information`` for the sums.
            # A time with one death term has u = 1 / R.
            s_u = np.zeros(m)
            s_u2 = np.zeros(m)
            s_u[one] = 1.0 / R[one]
            s_u2[one] = s_u[one] ** 2
            efron = None
            if tied.size:
                s_u[tied], s_cu_t, s_u2[tied], s_cu2, s_c2u2 = sums
                s_cu = np.zeros(m)
                s_cu[tied] = s_cu_t
                efron = (
                    death_n * e_beta_z[:, 0],
                    s_cu,
                    s_cu2,
                    s_c2u2,
                    ZRi[tied],
                    ZDi[tied],
                )
            active = ties.active
            hess_matrix = _cox_information(
                Z,
                rows,
                risk_n * e_beta_z[:, 0],
                s_u,
                s_u2[active],
                ZRi[active],
                efron,
            )

            return jacobian, hess_matrix

        return log_like, jac_hess

    def create_breslow_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        # The reference used to compute the jacobian and hessian
        # was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
        # Left-truncation is handled by subtracting the pre-entry risk set
        # (``Ri - TRi``) below, so delayed-entry data is fitted correctly.

        x, Z, c, n, tl = _sort_by_event_time(x, Z, c, n, tl)

        gb_x = _GroupBy(x)
        gb_tl = _GroupBy(tl)
        n_d_x = np.where(c == 0, n, 0)
        n_d = gb_x.sum(n_d_x)[1]
        risk_n = n
        n_d_x = n_d_x.reshape(-1, 1)
        n = n.reshape(-1, 1)

        x_ = gb_x.unique
        x_tl = gb_tl.unique
        # For each unique event time, how many unique entry times precede it:
        # feeds the not-yet-entered suffix-sum gather below.
        pos = np.searchsorted(x_tl, x_, side="left")
        rows = _RiskSetRows(x, x_, tl)
        # The times with a death, the only ones the information sums over.
        active = n_d > 0
        n_d_active = n_d[active]

        # Create the log_like function for the data
        def log_like(beta: npt.NDArray) -> float:
            beta_z = Z @ beta
            di_beta_z = gb_x.sum(n_d_x * beta_z.reshape(-1, 1))[1].reshape(
                -1, 1
            )
            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)

            # Subtract the not-yet-entered mass from the risk sums.
            if rows.truncated:
                Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])

            Ri = np.log(Ri)
            Ri = n_d.reshape(-1, 1) * Ri

            like = di_beta_z - Ri

            return -like.sum()

        S_d = gb_x.sum(n_d_x.reshape(-1, 1) * Z)[1]

        def jac_hess(beta: npt.NDArray) -> tuple:
            # Only call this once.. Yay.
            beta_z = Z @ beta

            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            z_e_beta_z = Z * e_beta_z

            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)
            ZRi = at_risk_beta_Z(z_e_beta_z, n, gb_x)

            # Subtract the not-yet-entered mass from the risk sums. The
            # Z-weighted sums are signed, so this must be the exact gather —
            # see ``not_yet_entered`` (#250).
            if rows.truncated:
                Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])
                ZRi = ZRi - not_yet_entered(pos, gb_tl.sum(n * z_e_beta_z)[1])

            EZ = ZRi / Ri
            EZ = n_d.reshape(-1, 1) * EZ

            jacobian = -(S_d - EZ).sum(axis=0)

            # Breslow's information is Efron's with every tie weight c = 0:
            # per death time n_d (Z2R / R - ZR ZR' / R^2); see
            # ``_cox_information``.
            R = Ri[:, 0]
            s_u = np.zeros(len(R))
            s_u[active] = n_d_active / R[active]
            hess_matrix = _cox_information(
                Z,
                rows,
                risk_n * e_beta_z[:, 0],
                s_u,
                n_d_active / R[active] ** 2,
                ZRi[active],
            )

            return jacobian, hess_matrix

        return log_like, jac_hess

    def _prepare_exact_tie_data(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple:
        """Expand count-weighted rows and pre-compute, per event time, the
        death rows, the (delayed-entry-aware) risk-set rows, and the death
        covariate sum. Shared by the ``exact`` and ``kalbfleisch-prentice``
        generators.

        Counts ``n`` are expanded into individual rows -- a row with a count of
        two events is genuinely two tied deaths -- so the exact and discrete
        tie formulae see the true multiplicity. Non-integer counts have no such
        interpretation and are rejected.
        """
        n = np.asarray(n, dtype=float)
        n_int = np.round(n).astype(int)
        if np.any(n_int < 1) or np.any(np.abs(n - n_int) > 1e-9):
            raise ValueError(
                "The 'exact' and 'kalbfleisch-prentice' tie methods require "
                "integer counts n (each row is expanded to n identical "
                "observations); use 'efron' or 'breslow' for fractional "
                "weights."
            )
        rep = np.repeat(np.arange(len(x)), n_int)
        xe = np.asarray(x, dtype=float)[rep]
        ce = np.asarray(c, dtype=int)[rep]
        tle = np.asarray(tl, dtype=float)[rep]
        Ze = np.asarray(Z, dtype=float)[rep]

        event_times = np.unique(xe[ce == 0])
        death_idx = []
        risk_idx = []
        death_Z_sum = []
        for tau in event_times:
            d_mask = (xe == tau) & (ce == 0)
            r_mask = cox_at_risk_mask(xe, tle, tau)
            death_idx.append(np.where(d_mask)[0])
            risk_idx.append(np.where(r_mask)[0])
            death_Z_sum.append(Ze[d_mask].sum(axis=0))
        return Ze, event_times, death_idx, risk_idx, np.array(death_Z_sum)

    @staticmethod
    def _tie_term_ll_jac_hess(
        n_events: int,
        term: Callable[[npt.NDArray, int, bool], tuple],
    ) -> tuple[Callable, Callable]:
        """Sum per-event-time ``term(eta, i, derivs) -> (log L_i, grad,
        hess)`` contributions into the ``(neg_ll, jac_hess)`` contract used
        by :meth:`fit` (the gradient and Hessian of the *negative*
        log-likelihood, so the Hessian is the observed information)."""

        def total(beta: npt.NDArray, derivs: bool) -> tuple:
            beta = np.asarray(beta, dtype=float)
            p = beta.shape[0]
            ll, score, hess = 0.0, np.zeros(p), np.zeros((p, p))
            for i in range(n_events):
                ll_i, g_i, h_i = term(beta, i, derivs)
                ll += ll_i
                score = score + g_i
                hess = hess + h_i
            return -ll, -score, -hess

        def neg_ll(beta: npt.NDArray) -> float:
            return float(total(beta, False)[0])

        def jac_hess(beta: npt.NDArray) -> tuple:
            _, score, hess = total(beta, True)
            return score, hess

        return neg_ll, jac_hess

    def create_kalbfleisch_prentice_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        """Kalbfleisch-Prentice discrete (conditional-logistic) tie handling.

        Treats tied event times as genuinely discrete: the contribution of a
        tie set ``D`` (``d`` deaths) with risk set ``R`` is

            exp(b' * sum_{j in D} Z_j) / e_d({exp(Z_k'b) : k in R}),

        where ``e_d`` is the ``d``-th elementary symmetric polynomial of the
        risk-set scores -- i.e. the sum over all ``d``-subsets of ``R`` of the
        product of their scores. This is the exact discrete
        proportional-hazards (Cox 1972 discrete model / Kalbfleisch-Prentice)
        likelihood, R's ``ties="exact"``. ``e_d`` and its derivatives come
        from the polynomial recursion in :func:`_kp_tie_term`.
        """
        Ze, event_times, death_idx, risk_idx, S = self._prepare_exact_tie_data(
            x, Z, c, n, tl
        )
        Z_risk = [Ze[r] for r in risk_idx]
        ds = [len(d) for d in death_idx]

        def term(beta: npt.NDArray, i: int, derivs: bool) -> tuple:
            log_e, g, h = _kp_tie_term(
                Z_risk[i] @ beta, Z_risk[i], ds[i], derivs
            )
            return float(S[i] @ beta) - log_e, S[i] - g, -h

        return self._tie_term_ll_jac_hess(len(event_times), term)

    def create_exact_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        """Exact (average-over-orderings) partial-likelihood tie handling.

        Appropriate when ties arise from coarse rounding of an underlying
        continuous time. Each tie set is treated as having occurred in an
        unknown order and its contribution is the sequential Cox partial
        likelihood summed over all orderings of the tied deaths, evaluated
        as the DeLong et al. integral (SAS's ``TIES=EXACT``; see
        :func:`_exact_tie_term`). Reduces to Breslow/Efron when there are no
        ties.
        """
        Ze, event_times, death_idx, risk_idx, S = self._prepare_exact_tie_data(
            x, Z, c, n, tl
        )
        Z_death = [Ze[d] for d in death_idx]
        # The rest of the risk set: at risk at the time but not dying there.
        survivors = [np.setdiff1d(r, d) for r, d in zip(risk_idx, death_idx)]
        Z_surv = [Ze[s] for s in survivors]
        Z_risk = [Ze[r] for r in risk_idx]

        def term(beta: npt.NDArray, i: int, derivs: bool) -> tuple:
            if len(death_idx[i]) == 1:
                # A single death needs no ordering: a_j / sum over the risk
                # set, the Breslow term, in closed form.
                lse, mean, cov = _weighted_moments(Z_risk[i] @ beta, Z_risk[i])
                return float(S[i] @ beta) - lse, S[i] - mean, -cov
            return _exact_tie_term(
                Z_death[i] @ beta,
                Z_death[i],
                Z_surv[i] @ beta,
                Z_surv[i],
                derivs,
            )

        return self._tie_term_ll_jac_hess(len(event_times), term)

    def _resolve_func_generator(self, tie_method: str) -> Callable[..., Any]:
        """Map a ``tie_method`` name to its likelihood generator."""
        generators: dict[str, Callable[..., Any]] = {
            "efron": self.create_efron_ll_jac_hess,
            "breslow": self.create_breslow_ll_jac_hess,
            "exact": self.create_exact_ll_jac_hess,
            "kalbfleisch-prentice": (
                self.create_kalbfleisch_prentice_ll_jac_hess
            ),
            "kp": self.create_kalbfleisch_prentice_ll_jac_hess,
        }
        if tie_method not in generators:
            raise ValueError(
                "tie_method must be one of {}".format(sorted(generators))
            )
        return generators[tie_method]

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        strata: npt.ArrayLike | None = None,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fits Cox Proportional Hazards model to the provided data.

        Parameters
        ----------

        x: array-like
            The observed times of the events.
        Z: array-like
            The covariates of the model, one row per observation. Rows with
            a missing or infinite covariate are dropped, with a warning. A
            column whose coefficient the partial likelihood cannot
            determine -- a constant column (a Cox model has no
            intercept), one constant within each stratum, or a linear
            combination of the others (every level of a factor, with no
            intercept) -- is aliased, as R's ``coxph`` does: the fit runs
            on the other columns, its coefficient and p-value are ``nan``
            (``model.aliased`` lists it), predictions take it as 0, and
            one warning names it (#476).
        c: array-like, optional
            The censoring indicator. 0 if observed (event),
            1 if right-censored. Defaults to all observed. An exactly
            observed time must be finite. Left-censored
            (-1) and interval-censored (2) rows raise a ``ValueError``: the
            partial likelihood has no term for them, so fit such data with
            a parametric regression model (e.g. ``WeibullPH``) instead.
        n: array-like, optional
            The number of observations at each time point.
        tl: array-like, optional
            The left-truncation times of the observations.
        tie_method: str, optional
            The method to use for tie handling. One of ``'efron'``
            (default), ``'breslow'``, ``'exact'`` (the average-over-orderings
            exact partial likelihood, for ties from coarse rounding of
            continuous time) or ``'kalbfleisch-prentice'`` (alias ``'kp'`` --
            the exact discrete/conditional-logistic likelihood, for genuinely
            discrete time). Without ties they all agree. With ties Efron is
            far closer to the exact likelihood than Breslow, which biases
            coefficients towards zero, at almost no extra cost; the baseline
            hazard then uses the matching Efron (Fleming-Harrington style)
            increments. ``'exact'`` removes the remaining bias under heavy
            ties at several times the cost.
        tol: float, optional
            The convergence tolerance. The coefficients are found by
            Newton-Raphson with step-halving on the partial
            log-likelihood (as R's ``coxph`` and lifelines), which stops
            once a step is at most ``tol`` standard errors long (measured
            by the observed information), leaving an error of the order of
            its square. Should Newton-Raphson fail -- as it does where the
            likelihood has no finite maximum -- the score is root-found
            instead, to a relative change in ``beta`` of ``tol``.
        strata: array-like, optional
            Stratum label for each observation. When supplied the model is
            *stratified*: a separate baseline hazard is estimated per stratum
            while the coefficients ``beta`` are shared. The partial likelihood
            is summed within strata (risk sets never cross a stratum boundary),
            which is the standard remedy when proportional hazards fails for a
            nuisance covariate that you would rather not model. Prediction
            (``hf``/``Hf``/``sf``/``ff``/``df``) then takes a ``stratum``
            argument to select that stratum's baseline. Observations with a
            missing label (``None``, ``NaN`` or pandas ``NA``) are dropped,
            with a warning.
        center: bool, optional
            ``False`` (the default) reports the baseline (``h0``, ``H0``,
            each stratum's) at ``Z = 0``; ``True`` reports it at the
            covariate means, stored as ``model.center`` (as R's ``coxph``
            and ``basehaz(fit)``), and ``phi(Z)`` is then relative to them.
            The fit runs on centred covariates either way, so the
            coefficients and predictions are the same; the default refuses,
            with a ``ValueError``, covariates so far from 0 that the
            baseline there over- or underflows.

        Returns
        -------

        model: SemiParametricRegressionModel
            The fitted model: ``params`` (also ``beta``) are the
            coefficients and ``p_values`` their Wald p-values; the
            baseline (``h0``, ``H0``) is that of a unit at ``center``
            (zeros unless ``center=True``). If a
            covariate separates the events from the survivors the partial
            likelihood has no finite maximum; the fit then warns
            ("monotone partial likelihood") and the coefficient is
            meaningless.

        Examples
        --------
        In the Rossi recidivism data ``arrest`` is 1 for a subject
        arrested during follow-up, so the censoring flag is
        ``1 - arrest``:

        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> Z = df[["fin", "age", "prio"]].values
        >>> model = CoxPH.fit(x, Z, c=c)
        >>> model.params.round(4)
        array([-0.347 , -0.0671,  0.0969])
        >>> model.p_values.round(4)
        array([0.0682, 0.0013, 0.0004])
        >>> model.sf([20, 52], [1, 25, 3]).round(4)
        array([0.9326, 0.7963])
        """
        func_generator = self._resolve_func_generator(tie_method)

        if strata is not None:
            return self._fit_stratified(
                x, Z, c, n, tl, tie_method, tol, strata, func_generator, center
            )

        x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, tie_method)
        check_finite_event_times(x, c)

        # Good initial guess assumes no impact
        beta_init = np.zeros(Z.shape[1])

        # Fitted on centred covariates, so exp(beta'Z) cannot overflow on a
        # column far from 0 (#459); see ``covariate_center``.
        mean = covariate_center(Z, n)
        Zc = Z - mean
        neg_ll, jac = func_generator(x, Zc, c, n, tl)

        res, p_values, se, aliased = _solve_beta_and_p_values(
            neg_ll, jac, beta_init, tol, Z, n, float(n[c == 0].sum())
        )

        model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
        model._neg_log_like = neg_ll(res.x)
        model.p_values = p_values
        model.se = se
        model.neg_ll = neg_ll
        model.jac = jac
        model.tie_method = tie_method
        model.baseline_method = _baseline_method(tie_method)
        model.res = res
        model.beta = copy(res.x)
        model.params = res.x
        if aliased.size:
            # Reported as nan (R's NA); ``res.x`` keeps the 0 the
            # predictions use (#476).
            model.beta = np.array(res.x, dtype=float)
            model.beta[aliased] = np.nan
            model.params = model.beta.copy()

        # Retain the per-observation training data (before ``baseline``
        # reassigns ``x`` to the unique event times) so the model can compute
        # residuals (Schoenfeld, martingale, ...) and the proportional-
        # hazards test.
        model._fit_data = {
            "x": np.asarray(x, dtype=float),
            "c": np.asarray(c, dtype=int),
            "n": np.asarray(n, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "tl": np.asarray(tl, dtype=float),
        }

        # The baseline of a unit at the centre, moved to Z = 0 by default.
        x, r, d, h0 = self.baseline(res.x, x, c, n, Zc, tl, tie_method)
        if center:
            model.center = mean
        else:
            r, h0 = _baseline_at_origin(res.x, mean, Z, r, h0)
            model.center = np.zeros_like(mean)
        # Where the fit centred, for the residuals and diagnostics.
        model._fit_center = mean
        model.x = x
        model.r = r
        model.d = d
        model.tl = tl
        model.h0 = h0
        model.H0 = model.h0.cumsum()

        return model

    def _fit_stratified(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: "npt.ArrayLike | None",
        n: "npt.ArrayLike | None",
        tl: "npt.ArrayLike | None",
        tie_method: str,
        tol: float,
        strata: npt.ArrayLike,
        func_generator: Callable,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """Fit a stratified Cox model (shared ``beta``, per-stratum baseline).

        Each stratum is validated and turned into its own partial-likelihood
        generator; the generators are summed (see :func:`_combine_generators`)
        so the score equations are solved once for the shared coefficients.
        A separate baseline hazard is then estimated within each stratum.
        """
        labels_arr, missing = _strata_labels(strata)
        if len(labels_arr) != len(np.atleast_1d(x)):
            raise ValueError("'strata' must have a label for each observation")
        # The per-observation arrays, subset together as rows are dropped.
        obs: list[Any] = [x, c, n, Z, tl]
        if Z is not None:
            # Checked before the per-stratum split, whose boolean mask
            # would otherwise raise a bare IndexError on a Z of the wrong
            # length.
            check_covariate_rows(np.asarray(Z), len(labels_arr))
            # Rows with a missing covariate are dropped here, once, rather
            # than by each stratum's validation (one warning per stratum,
            # each counting only that stratum's rows).
            keep = finite_covariate_mask(np.asarray(Z, dtype=float))
            if not keep.all():
                obs = [_sub(a, keep) for a in obs]
                labels_arr, missing = labels_arr[keep], missing[keep]
        if missing.any():
            # An observation without a stratum has no baseline to belong
            # to; it is dropped, as a row with a missing covariate is.
            if missing.all():
                raise ValueError(
                    "Every stratum label is missing; there is nothing to fit."
                )
            warnings.warn(
                "Dropped {} of {} rows with a missing stratum label.".format(
                    int(missing.sum()), missing.shape[0]
                ),
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
            obs = [_sub(a, ~missing) for a in obs]
            labels_arr = labels_arr[~missing]
        x_o, c_o, n_o, Z_o, tl_o = obs

        labels = np.unique(labels_arr)
        validated = []
        for s in labels:
            mask = labels_arr == s
            xs, cs, ns_, tls, Zs = validate_coxph(
                _sub(x_o, mask),
                _sub(c_o, mask),
                _sub(n_o, mask),
                _sub(Z_o, mask),
                _sub(tl_o, mask),
                tie_method,
            )
            check_finite_event_times(xs, cs)
            validated.append((s, xs, cs, ns_, tls, Zs))

        if not validated:
            raise ValueError("no observations to fit")
        n_params = validated[0][5].shape[1]
        # One centre for every stratum, the mean over all the rows (as R's
        # coxph), so the strata's baselines stay comparable (#459).
        mean = covariate_center(
            np.vstack([v[5] for v in validated]),
            np.concatenate([v[3] for v in validated]),
        )
        per_stratum = []
        for s, xs, cs, ns_, tls, Zs in validated:
            Zcs = Zs - mean
            gen = func_generator(xs, Zcs, cs, ns_, tls)
            per_stratum.append((s, gen, (xs, cs, ns_, Zcs, tls)))

        gens = [g for _, g, _ in per_stratum]
        neg_ll, jac = _combine_generators(gens)

        beta_init = np.zeros(n_params)
        res, p_values, se, aliased = _solve_beta_and_p_values(
            neg_ll,
            jac,
            beta_init,
            tol,
            # The covariates, counts and strata, for the aliasing check.
            np.vstack([v[5] for v in validated]),
            np.concatenate([v[3] for v in validated]),
            sum(
                float(data[2][data[1] == 0].sum())
                for _, _, data in per_stratum
            ),
            np.concatenate(
                [np.full(len(v[1]), k) for k, v in enumerate(validated)]
            ),
        )

        model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
        model._neg_log_like = neg_ll(res.x)
        model.p_values = p_values
        model.se = se
        model.neg_ll = neg_ll
        model.jac = jac
        model.tie_method = tie_method
        model.baseline_method = _baseline_method(tie_method)
        model.res = res
        model.beta = copy(res.x)
        model.params = res.x
        if aliased.size:
            # Reported as nan (R's NA); ``res.x`` keeps the 0 the
            # predictions use (#476).
            model.beta = np.array(res.x, dtype=float)
            model.beta[aliased] = np.nan
            model.params = model.beta.copy()
        model.center = mean if center else np.zeros_like(mean)
        model.is_stratified = True
        model.strata_labels = list(labels)

        # A separate baseline per stratum, each that of a unit at the
        # centre, moved to Z = 0 by default. Prediction selects the
        # stratum's baseline via the ``stratum`` argument to ``hf``/``Hf``.
        Z_all = np.vstack([v[5] for v in validated])
        baselines: dict[Any, dict[str, npt.NDArray]] = {}
        for s, _, (xs, cs, ns_, Zs, tls) in per_stratum:
            bx, br, bd, bh0 = self.baseline(
                res.x, xs, cs, ns_, Zs, tls, tie_method
            )
            if not center:
                br, bh0 = _baseline_at_origin(res.x, mean, Z_all, br, bh0)
            baselines[s] = {
                "x": bx,
                "r": br,
                "d": bd,
                "h0": bh0,
                "H0": bh0.cumsum(),
            }
        model.strata_baselines = baselines

        # Expose the first stratum's baseline as the default so generic
        # attribute access (e.g. ``model.x``) still works; correct prediction
        # must pass an explicit ``stratum``.
        first = baselines[labels[0]]
        model.x = first["x"]
        model.r = first["r"]
        model.d = first["d"]
        model.h0 = first["h0"]
        model.H0 = first["H0"]
        model.tl = None

        return model

    def fit_from_df(
        self,
        df: "pd.DataFrame",
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        tie_method: str = "efron",
        strata_col: str | None = None,
        tl_col: str | None = None,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fits a Cox PH model using a pandas dataframe as the input.

        Parameters
        ----------

        df: pandas.DataFrame
            The dataframe containing the data.
        x_col: str
            The column name of the observed times.
        Z_cols: str or list of str, optional
            The column name(s) of the covariates. Give this or ``formula``.
        c_col: str, optional
            The column name of the censoring indicator.
        n_col: str, optional
            The column name of the number of observations at each time point.
        formula: str, optional
            A ``formulaic`` formula for the covariates (e.g.
            ``"age + site"``), instead of ``Z_cols``; categorical columns get
            reference-level coding. Rows with a missing covariate (in
            ``Z_cols`` or a formula column) are dropped, with a warning.
        tie_method: str, optional
            The tie-handling method: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (alias ``'kp'``). See
            :meth:`fit`.
        strata_col: str, optional
            The column name of the stratum label. When supplied the model is
            fitted stratified (a separate baseline hazard per stratum, shared
            coefficients); see :meth:`fit`. Rows with a missing label are
            dropped, with a warning.
        tl_col: str, optional
            The column name of the left-truncation (delayed-entry) times,
            passed to :meth:`fit` as ``tl``. A subject enters the risk sets
            only after its entry time.
        center: bool, optional
            Report the baseline at the covariate means (``model.center``)
            instead of at ``Z = 0``; see :meth:`fit`.

        Returns
        -------

        model: SemiParametricRegressionModel
            The fitted model.
        """
        x, c, n, tl, strata, Z, form, feature_names, model_spec = (
            validate_coxph_df_inputs(
                df,
                x_col,
                c_col,
                n_col,
                Z_cols,
                formula,
                tl_col=tl_col,
                strata_col=strata_col,
            )
        )

        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(
                x,
                Z,
                c,
                n,
                tl=tl,
                tie_method=tie_method,
                strata=strata,
                center=center,
            )
        model.formula = form
        model.feature_names = feature_names
        model._model_spec = model_spec

        return model

    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fit a Cox model with time-varying covariates in start-stop format.

        Each row is one observation interval ``(xl, xr]`` of a subject
        (identified by ``i``) on which the covariate row ``Z`` is constant;
        ``c`` is ``0`` (event) only on the interval that ends at the subject's
        event and ``1`` (right-censored) otherwise. The rows are validated (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc`) and fitted as delayed-entry observations -- exact for the
        Cox partial likelihood.

        Parameters
        ----------
        i, xl, xr, c, Z : array_like
            The start-stop interval data: subject id, interval entry time
            ``xl``, exit time ``xr``, censoring flag ``c`` (``0`` event at
            ``xr``, ``1`` right-censored -- surpyval's convention), and the
            per-interval covariates.
        n : array_like, optional
            Count weight per interval row.
        tie_method : str, optional
            Tie-handling method: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (``'kp'``); see
            :meth:`fit`.
        tol : float, optional
            Convergence tolerance; see :meth:`fit`.
        center : bool, optional
            Report the baseline at the covariate means of the interval rows
            (``model.center``) instead of at ``Z = 0``; see :meth:`fit`.

        Returns
        -------
        SemiParametricRegressionModel
            The fitted model, with ``is_tvc`` set. Evaluate it along a
            covariate path with ``sf_tvc`` / ``Hf_tvc`` (or the
            interval-oriented ``predict_tvc``); its cluster-robust standard
            errors cluster the rows by subject.

        Examples
        --------
        Seven subjects; four of them move from ``Z = 0`` to ``Z = 1`` part
        way through follow-up, so they contribute two rows each, and only
        the row ending in an event carries ``c = 0``:

        >>> from surpyval import CoxPH
        >>> from surpyval.univariate.regression import StepSchedule
        >>> i  = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6]
        >>> xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0]
        >>> xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8]
        >>> c  = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
        >>> Z  = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
        >>> model = CoxPH.fit_tvc(i, xl, xr, c, Z)
        >>> model.beta.round(4)
        array([1.6982])

        Survival of a unit that switches to ``Z = 1`` at time 2:

        >>> path = StepSchedule.from_changepoints([0, 2], [[0], [1]])
        >>> model.sf_tvc([1, 3, 5], path).round(4)
        array([1.    , 0.6513, 0.3171])
        """
        x, c, n_arr, tl, Z_arr, ident = handle_tvc(i, xl, xr, c, Z, n)
        model = self.fit(
            x=x,
            Z=Z_arr,
            c=c,
            n=n_arr,
            tl=tl,
            tie_method=tie_method,
            tol=tol,
            center=center,
        )
        model.is_tvc = True
        # Subject ids per *internal* (sorted) row, and the permutation from
        # the caller's row order to the internal order: residuals and
        # cluster-robust SEs align with the internal order, so user-supplied
        # per-row labels must be permuted the same way (#259).
        model.tvc_subject_ids = ident
        model.tvc_row_order = np.lexsort(
            (np.asarray(xl, dtype=float), np.asarray(i))
        )
        return model

    def fit_tvc_from_df(
        self,
        df: "pd.DataFrame",
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str] | None = None,
        n_col: str | None = None,
        tie_method: str = "efron",
        center: bool = False,
        formula: str | None = None,
    ) -> SemiParametricRegressionModel:
        """
        Fit a time-varying-covariate Cox model from a start-stop DataFrame.

        See :meth:`fit_tvc`; ``Z_cols`` names the covariate column(s) and the
        remaining arguments name the id / ``xl`` / ``xr`` / ``c`` columns.
        ``tie_method`` and ``center`` are as for :meth:`fit`. Instead of
        ``Z_cols``, ``formula`` (as in :meth:`fit_from_df`) gives the
        covariates as a ``formulaic`` formula, which codes categorical
        (e.g. ``"yes"`` / ``"no"``) columns.
        """
        return fit_tvc_df(
            self.fit_tvc,
            df,
            {"i": i_col, "xl": xl_col, "xr": xr_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            tie_method=tie_method,
            center=center,
        )

    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fit a time-varying-covariate Cox model from a covariate *timeline*.

        This is the timeline / ``xicnt``-style alternative to
        :meth:`fit_tvc`'s explicit ``(start, stop]`` intervals. Each subject's
        rows give its covariate history: a covariate value ``Z`` takes effect
        at time ``x`` and holds until the subject's next row, with the terminal
        event / censoring marked on the last row's ``c``. The timeline is
        expanded to start-stop intervals (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc_timeline`) and fitted exactly as :meth:`fit_tvc`, so it
        gives an identical fit to the equivalent start-stop data.

        Parameters
        ----------
        i : array_like
            Subject identifier for each timeline row (the ``xicnt`` item id).
        x : array_like
            The time each row's covariate value takes effect. Strictly
            increasing within a subject; the first is the entry
            (delayed-entry) time, the last is the event / censoring time.
        Z : array_like
            The covariate vector effective from this row's ``x``. The value on
            a subject's terminal (last) row is ignored.
        c : array_like
            Censoring status; only each subject's last row is read (``0``
            event, ``1`` right-censored).
        n : array_like, optional
            Per-subject count weight (read from the terminal row).
        tie_method : str, optional
            Tie-handling method: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (``'kp'``); see
            :meth:`fit`.
        tol : float, optional
            Convergence tolerance; see :meth:`fit`.
        center : bool, optional
            Report the baseline at the covariate means (``model.center``)
            instead of at ``Z = 0``; see :meth:`fit`.

        Returns
        -------
        SemiParametricRegressionModel
            The fitted model, with ``is_tvc`` set.
        """
        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_tvc_timeline(i, x, Z, c, n)
        return self.fit_tvc(
            i=i_ss,
            xl=xl,
            xr=xr,
            c=c_ss,
            Z=Z_ss,
            n=n_ss,
            tie_method=tie_method,
            tol=tol,
            center=center,
        )

    def fit_tvc_timeline_from_df(
        self,
        df: "pd.DataFrame",
        i_col: str,
        x_col: str,
        Z_cols: str | list[str] | None,
        c_col: str,
        n_col: str | None = None,
        tie_method: str = "efron",
        center: bool = False,
        formula: str | None = None,
    ) -> SemiParametricRegressionModel:
        """
        Fit a timeline TVC Cox model from a DataFrame.

        See :meth:`fit_tvc_timeline`; ``x_col`` names the change-point time
        column (``x``), ``Z_cols`` the covariate column(s) and ``c_col`` the
        terminal event / censoring column (``0`` event, ``1`` censored).
        ``tie_method`` and ``center`` are as for :meth:`fit`. Instead of
        ``Z_cols`` (pass ``None``), ``formula`` gives the covariates as a
        ``formulaic`` formula, as in :meth:`fit_from_df`.
        """
        return fit_tvc_df(
            self.fit_tvc_timeline,
            df,
            {"i": i_col, "x": x_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            tie_method=tie_method,
            center=center,
        )


CoxPH = CoxPH_()
