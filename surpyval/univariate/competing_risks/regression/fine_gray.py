"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC

Fine-Gray subdistribution-hazard regression for competing risks.

Where cause-specific proportional hazards models the hazard of each cause
after removing subjects who fail from a competing cause, the Fine-Gray model
(Fine & Gray, 1999) keeps those subjects in a modified ("subdistribution")
risk set so that a single coefficient vector acts directly on the cumulative
incidence function (CIF) of the cause of interest,

.. math::
    F_k(t \\mid Z) = 1 - \\exp\\{-\\Lambda_{k0}(t)\\,\\exp(\\beta' Z)\\},

with :math:`\\Lambda_{k0}` a baseline cumulative subdistribution hazard. This
makes :math:`\\beta` interpretable as a (log) subdistribution hazard ratio: a
positive coefficient raises the incidence of cause :math:`k`.

Independent right-censoring is handled by inverse-probability-of-censoring
weighting (IPCW): a subject who has already failed from a competing cause
stays in the subdistribution risk set with a time-varying weight
:math:`G(t-)/G(x_i-)`, where :math:`G` is the Kaplan-Meier estimate of the
censoring-time survival function. Subjects who are censored, or who have
already had the event of interest, leave the risk set. The partial likelihood
is the Breslow form of this weighted risk set.

:math:`G` is evaluated just before each time, as R's ``cmprsk::crr`` does
(its ``uuu`` is the censoring Kaplan-Meier at ``ftime-``): an event and a
censoring at the same instant are ordered event first, so the censorings at
:math:`t` do not yet reduce the weight at :math:`t`, nor count against a
competing failure at :math:`x_i`. :math:`G` itself is ``survfit``'s reverse
Kaplan-Meier, as in ``crr``. With no censoring time equal to an event time
the left limits equal :math:`G(t)` and :math:`G(x_i)`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt
from autograd import hessian
from autograd import numpy as anp
from autograd.tracer import getval
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.univariate.information_criteria import InformationCriteriaMixin
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    expand,
    warn_aliased,
)
from surpyval.univariate.regression._fit_skeleton import (
    LOG_MAX,
    baseline_at_origin_error,
    judge_search,
)
from surpyval.univariate.regression.proportional_hazards.cox_likelihood import (  # noqa: E501
    newton_raphson,
)
from surpyval.univariate.regression.proportional_hazards.cox_ph import (
    warn_monotone,
)
from surpyval.univariate.regression.regression_data import (
    LinearPredictorMixin,
)
from surpyval.utils import validate_fine_gray_inputs
from surpyval.utils.covariates import coefficient_floor
from surpyval.utils.dataframe import (
    call_fit,
    cause_column,
    frame_column,
    frame_columns,
    require_frame,
)
from surpyval.utils.deprecation import RenamedToMethod
from surpyval.utils.ipcw import censoring_survival, step_at, step_left_limit
from surpyval.utils.linalg import safe_inv
from surpyval.utils.no_maximum import (
    combined_maximum,
    maximum_entry,
    restored_maximum,
    warn_unverified,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    missing_cause_error,
    unknown_cause_error,
)

#: Newton-Raphson's convergence tolerance, in standard errors of the step
#: (``newton_raphson``), CoxPH's default.
_NEWTON_TOL = 1e-10


def _weighted_neg_ll(
    n_sorted: npt.NDArray,
    sets: Any,
    denom0: npt.NDArray,
    Zk: npt.NDArray,
    nZk_event: npt.NDArray,
    beta: Any,
) -> Any:
    """The weighted negative partial log-likelihood at ``beta``, less its
    value at 0 (see ``_fit_cause``), differentiable by autograd. The linear
    predictor is shifted by its largest value inside the risk-set sums and
    the shift added back outside the logarithm, so ``exp(beta'Z)`` cannot
    overflow (#606)."""
    eta = anp.dot(Zk, beta)
    shift = float(np.max(getval(eta))) if eta.size else 0.0
    weighted_exp = n_sorted * anp.exp(eta - shift)
    denom = _risk_set_sums(weighted_exp, sets)
    ll = anp.dot(nZk_event, beta) - anp.sum(
        sets.d * (anp.log(denom / denom0) + shift)
    )
    return -ll


def _fit_cause(
    x: npt.NDArray,
    Z: npt.NDArray,
    e: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    cause: Any,
    center: bool = False,
) -> dict:
    """
    Fit the Fine-Gray subdistribution-hazard model for a single ``cause``.

    Returns a dict with the fitted coefficients, their standard errors and
    p-values, the baseline cumulative subdistribution hazard (as sorted event
    times and the cumulative hazard at each), the optimiser result, and the
    coefficients along which the partial likelihood has no finite maximum
    (``"runaway"``, see :func:`_warn_if_monotone`).

    The fit runs on the covariates centred at their ``n``-weighted means
    (#463), as ``CoxPH`` does (#459): the partial likelihood, and so
    ``beta`` and its covariance, are unchanged by it, while
    ``exp(beta'Z)`` on a covariate far from 0 (a year, a date) overflowed
    and the fit failed ("SVD did not converge"). The baseline it gives is
    that of a unit at the means; it is kept there, as ``center``, with
    ``center=True``, and otherwise moved to ``Z = 0`` (``center`` zeros),
    which is refused where that over- or underflows.
    """
    is_cause = label_mask(e, cause)
    is_event = (c == 0) & is_cause
    if not is_event.any():
        raise ValueError(f"No observed events for cause {cause!r}")
    is_competing = (c == 0) & ~is_cause

    # Censoring survival for the IPCW weights, taken just before each time,
    # G(t-), as cmprsk::crr does: at a time shared by events and censorings
    # the events come first, so the censorings there must not yet thin the
    # weights (evaluating G(t) counted them against the events they tie with).
    # G(x_i-) > 0 for every row: row i (a positive count) is at risk and
    # uncensored at every earlier censoring time, so no step before x_i can
    # reach zero and the ratio G(t-)/G(x_i-) needs no guard.
    g_times, g_vals = censoring_survival(x, c == 1, n)
    G_x = step_left_limit(g_times, g_vals, x, before=1.0)
    sets = _risk_sets(x, n, is_event, is_competing, G_x, g_times, g_vals)

    Z_raw = Z
    mean = (n @ Z) / n.sum()
    Z = Z - mean
    n_event = n[is_event]
    # The rows in time order, which the risk-set sums run over, and the
    # events' sum of n * Z, the linear term of the log-likelihood.
    Z_sorted = Z[sets.order]
    n_sorted = n[sets.order]
    nZ_event = n_event @ Z[is_event]
    # The optimiser minimises the negative log-likelihood less its value at
    # beta = 0, sum d log(denom0): near the maximum its steps change the
    # likelihood (5e4 at 1e4 rows) by less than its last digit. BFGS
    # stopped there on "precision loss" in 10 of 450 fits at 3e3 rows
    # before #517, and in 17 with plain cumulative sums; with this and the
    # blocked sums of _cumsum, in 1.
    denom0 = _risk_set_sums(n_sorted, sets)
    offset = float(sets.d @ np.log(denom0))

    def partial_neg_ll(Zk: npt.NDArray, nZk_event: npt.NDArray) -> tuple:
        """The objective, differentiable by autograd (the no-maximum check
        and the information take its derivatives); the objective with
        its gradient for BFGS, ``-(nZ_event - sum_j d_j S1_j / S0_j)``
        (``S1`` the risk-set sums of ``w Z``), by hand: a fit of 1e5 rows
        took 1.2 s with autograd's gradient, 0.7 s with this; and the
        gradient with the information, by hand, for Newton-Raphson.

        Each shifts the linear predictor by its largest value inside the
        risk-set sums and adds the shift back outside the logarithm, as
        ``CoxPH`` does, so ``exp(beta'Z)`` cannot overflow at any
        coefficients: on covariates of order 1e4 it did at the search's
        first step, and the fit failed ("SVD did not converge", #606). The
        objective is a ``functools.partial`` of a module-level function,
        not a closure, so the model, which keeps it, pickles (#573)."""
        neg_ll = functools.partial(
            _weighted_neg_ll, n_sorted, sets, denom0, Zk, nZk_event
        )

        def shifted(beta: npt.NDArray) -> tuple:
            # The weights e^(eta - shift), their risk-set sums and the shift
            eta = Zk @ beta
            shift = float(eta.max()) if eta.size else 0.0
            weighted_exp = n_sorted * np.exp(eta - shift)
            return weighted_exp, _risk_set_sums(weighted_exp, sets), shift

        def value_and_gradient(beta: npt.NDArray) -> tuple:
            weighted_exp, denom, shift = shifted(beta)
            value = -(
                nZk_event @ beta
                - np.sum(sets.d * (np.log(denom / denom0) + shift))
            )
            # sum_j d_j S1_j / S0_j = sum_i w_i Z_i sum_j W_ji d_j / S0_j
            weights = _risk_set_weights(sets.d / denom, sets)
            gradient = -(nZk_event - Zk.T @ (weighted_exp * weights))
            return value, gradient

        def gradient_and_information(beta: npt.NDArray) -> tuple:
            # The information sum_j d_j (S2_j / S0_j - M_j M_j'), M_j =
            # S1_j / S0_j, in O(N p^2) as at beta = 0
            # (_information_at_zero); the shift cancels in every ratio.
            weighted_exp, denom, _ = shifted(beta)
            a = weighted_exp * _risk_set_weights(sets.d / denom, sets)
            M = _risk_set_sums((weighted_exp[:, None] * Zk).T, sets).T
            M = M / denom[:, None]
            gradient = -(nZk_event - Zk.T @ a)
            information = Zk.T @ (a[:, None] * Zk) - M.T @ (
                sets.d[:, None] * M
            )
            return gradient, information

        return neg_ll, value_and_gradient, gradient_and_information

    # Coefficients the weighted partial likelihood does not depend on are
    # aliased, as CoxPH's are (#476): fitted on the other columns, and
    # reported as nan.
    p = Z.shape[1]
    neg_ll, value_and_gradient, newton_derivatives = partial_neg_ll(
        Z_sorted, nZ_event
    )
    aliased = aliased_columns(
        _information_at_zero(Z_sorted, n_sorted, sets),
        Z.shape[0],
        constant_columns(Z_raw),
        float(n_event.sum()) * (n @ Z**2) / n.sum(),
    )
    kept = np.setdiff1d(np.arange(p), aliased)
    if aliased.size:
        warn_aliased(
            aliased,
            "the partial likelihood does not depend on them (a constant "
            "column, or a linear combination of the others within the "
            "risk sets, as the columns of every level of a factor are)",
        )
        neg_ll, value_and_gradient, newton_derivatives = partial_neg_ll(
            Z_sorted[:, kept], nZ_event[kept]
        )

    beta0 = np.zeros(kept.size)
    if kept.size:
        # Newton-Raphson with step-halving, as cmprsk::crr and CoxPH: its
        # steps are those of the covariates' own units, whatever they are,
        # where BFGS's first step is 1 in each coefficient, which on a
        # covariate of order 1e4 moved the linear predictor by 1e4 (#606).
        # It gives up where the likelihood has no finite maximum, and BFGS
        # takes over, as before.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            score0, information0 = newton_derivatives(beta0)
            res = newton_raphson(
                lambda b: value_and_gradient(b)[0],
                newton_derivatives,
                beta0,
                _NEWTON_TOL,
                score0,
                information0,
            )
        if res is None:
            res = minimize(value_and_gradient, beta0, jac=True, method="BFGS")
        # A covariate that separates the events of interest from the rest
        # (a level with none of them) drives its coefficient to infinity;
        # BFGS stops where the rise is below its tolerance and reports
        # success (-12.9 on such data). Newton's method cannot converge
        # from there, which is what the check finds (#392). Otherwise the
        # answer must be a verified maximum, polished if it is not (BFGS's
        # absolute tolerance on the gradient is not scale free), each
        # coefficient in its own covariate's units (#577).
        verdict = judge_search(
            neg_ll,
            res,
            [(k, int(kept[k])) for k in range(kept.size)],
            beta0,
            float(n_event.sum()),
            floor=coefficient_floor(
                kept.size,
                [(k, k) for k in range(kept.size)],
                Z_sorted[:, kept],
            ),
        )
        res, derivatives = verdict.res, verdict.derivatives
        runaway, maximum = verdict.runaway, verdict.maximum
    else:
        # Every coefficient aliased: nothing to fit.
        res = OptimizeResult(
            x=beta0, fun=float(neg_ll(beta0)), success=True, nit=0
        )
        derivatives, runaway, maximum = None, [], "verified"
    # The negative log-likelihood itself.
    res.fun = float(res.fun) + offset
    beta = res.x

    # Standard errors from the inverse observed information, the Hessian
    # the check just took.
    H = hessian(neg_ll)(beta) if derivatives is None else derivatives[0]
    cov = safe_inv(H)
    var = np.diag(cov)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(var > 0, var, np.nan))
        z_score = beta / se
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))
    if aliased.size:
        # 0 for the baseline and predictions; the model reports nan.
        beta = expand(beta, kept, p)
        se = expand(se, kept, p)
        p_values = expand(p_values, kept, p)
        full = np.full((p, p), np.nan)
        full[np.ix_(kept, kept)] = cov
        cov = full
        beta = np.where(np.isnan(beta), 0.0, beta)

    # Breslow baseline cumulative subdistribution hazard: at each event-of-
    # interest time, dLambda0 = (events there) / (weighted risk set there).
    denom = _risk_set_sums(n_sorted * np.exp(Z_sorted @ beta), sets)
    uniq_t = sets.times
    baseline_cumhaz = np.cumsum(sets.d / denom)
    if not center:
        baseline_cumhaz = _cumhaz_at_origin(beta, mean, Z_raw, baseline_cumhaz)
        mean = np.zeros_like(mean)

    if aliased.size:
        beta = np.array(beta, dtype=float)
        beta[aliased] = np.nan
    return {
        "cause": cause,
        "beta": beta,
        "center": mean,
        "se": se,
        "p_values": p_values,
        "cov": cov,
        "baseline_times": uniq_t,
        "baseline_cumhaz": baseline_cumhaz,
        "neg_ll": float(res.fun),
        # BIC's sample size: the events of interest, the terms of the
        # partial likelihood (#604; Kuk and Varadhan's BIC_cr).
        "ic_n": float(n_event.sum()),
        "res": res,
        "runaway": runaway,
        "maximum": maximum,
        "objective": neg_ll,
    }


class _RiskSets(NamedTuple):
    """The subdistribution risk sets of one cause, as
    :func:`_risk_set_sums` reads them (see :func:`_risk_sets`)."""

    #: The permutation that sorts the rows by time.
    order: npt.NDArray
    #: The distinct event times of the cause, and the events (counts) at
    #: each.
    times: npt.NDArray
    d: npt.NDArray
    #: The first row, in time order, with ``x >= t`` for each event time.
    start: npt.NDArray
    #: ``G(t-)`` at each event time, and ``1 / G(x_i-)`` for each row, in
    #: time order, that failed from a competing cause (0 for the others).
    G_t: npt.NDArray
    competing_over_G: npt.NDArray


def _risk_sets(
    x: npt.NDArray,
    n: npt.NDArray,
    is_event: npt.NDArray,
    is_competing: npt.NDArray,
    G_x: npt.NDArray,
    g_times: npt.NDArray,
    g_vals: npt.NDArray,
) -> _RiskSets:
    """
    The subdistribution risk sets of the cause whose events are
    ``is_event``, in the O(N) form :func:`_risk_set_sums` evaluates.

    At an event time ``t`` the risk set is every row with ``x_i >= t``,
    with weight 1, and every row that failed from a competing cause before
    ``t``, with weight ``G(t-)/G(x_i-)``; a censored row, or one that
    already had the event of interest, has left it. The weights as an
    (event times x N) matrix cost O(events x N) time and memory (22 GB at
    1e5 rows and three causes, #517); in time order the first part is a
    suffix of the rows and the second a prefix, so both are cumulative
    sums.
    """
    order = np.argsort(x, kind="mergesort")
    times, inv = np.unique(x[is_event], return_inverse=True)
    d = np.bincount(inv, weights=n[is_event], minlength=times.size)
    start = np.searchsorted(x[order], times, side="left")
    G_t = step_left_limit(g_times, g_vals, times, before=1.0)
    # G(x_i-) > 0 for every row (see _fit_cause), so the division is safe.
    competing_over_G = np.where(is_competing, 1.0 / G_x, 0.0)[order]
    return _RiskSets(order, times, d, start, G_t, competing_over_G)


def _risk_set_sums(v: Any, sets: _RiskSets) -> Any:
    """
    The weighted sum of ``v`` over the subdistribution risk set at each
    event time of ``sets``: ``sum_i W_ti v_i``, ``W`` the IPCW weights of
    :func:`_risk_sets`.

    ``v`` is one value per row in time order (``sets.order``), or several
    such rows (shape ``(k, N)``, giving ``(k, event times)``); the sums
    are in O(N), the suffix sum of ``v`` over ``x_i >= t`` plus ``G(t-)``
    times the prefix sum of ``v_i / G(x_i-)`` over the competing failures
    before ``t``. Differentiable by autograd.
    """
    # suffix[k] = sum of v over the rows k, k + 1, ... in time order
    suffix = _cumsum(v[..., ::-1])[..., ::-1]
    # prefix[k] = sum over the competing rows before row k of v / G(x-)
    prefix = anp.concatenate(
        [
            anp.zeros(anp.shape(v)[:-1] + (1,)),
            _cumsum(sets.competing_over_G * v),
        ],
        axis=-1,
    )
    return suffix[..., sets.start] + sets.G_t * prefix[..., sets.start]


def _risk_set_weights(q: npt.NDArray, sets: _RiskSets) -> npt.NDArray:
    """
    ``sum_t q_t W_ti`` for each row ``i`` in time order, ``q`` one value
    per event time of ``sets``: the transpose of :func:`_risk_set_sums`,
    in O(N). Row ``i`` is in the risk set, with weight 1, at the event
    times at or before ``x_i`` (``start_t <= i``), and, for a competing
    failure, with weight ``G(t-)/G(x_i-)`` at the later ones.
    """
    rows = sets.order.size
    at_or_before = _cumsum(np.bincount(sets.start, q, minlength=rows))
    later = np.bincount(sets.start, q * sets.G_t, minlength=rows + 1)
    after = _cumsum(later[::-1])[::-1][1:]
    return at_or_before + sets.competing_over_G * after


def _cumsum(v: Any) -> Any:
    """
    The cumulative sum of ``v`` along its last axis, in blocks of about
    ``sqrt(N)`` values: the running sum within each block plus the sum of
    the blocks before it. Its rounding error grows as ``sqrt(N)`` rather
    than ``N`` (``np.cumsum``'s), which keeps the risk-set sums as
    accurate as the matrix product they replace (#517): with a plain
    cumulative sum their relative error at 1e4 rows was 1.4e-15 (rms)
    rather than the product's 1.4e-16, and the partial likelihood's
    rounding noise 20 times the product's. Differentiable by autograd.
    """
    lead = tuple(anp.shape(v)[:-1])
    rows = anp.shape(v)[-1]
    size = max(1, int(np.ceil(np.sqrt(rows))))
    blocks = -(-rows // size)
    if blocks * size > rows:
        v = anp.concatenate(
            [v, anp.zeros(lead + (blocks * size - rows,))], axis=-1
        )
    split = anp.reshape(v, lead + (blocks, size))
    before = anp.concatenate(
        [
            anp.zeros(lead + (1,)),
            anp.cumsum(anp.sum(split, axis=-1), axis=-1)[..., :-1],
        ],
        axis=-1,
    )
    out = anp.cumsum(split, axis=-1) + before[..., None]
    return anp.reshape(out, lead + (blocks * size,))[..., :rows]


def _information_at_zero(
    Z: npt.NDArray, n: npt.NDArray, sets: _RiskSets
) -> npt.NDArray:
    """The information of the subdistribution partial likelihood at
    ``beta = 0``: over the events, the covariance of ``Z`` in the risk set
    weighted by ``W * n``, ``sum_j d_j (sum_i w_ji Z_i Z_i' / S0_j - m_j
    m_j')``, ``m_j`` the weighted mean. It is what the aliasing check
    (#476) judges the columns by. ``Z`` and ``n`` are in time order
    (``sets.order``); every sum is O(N) (:func:`_risk_set_sums`)."""
    S0 = _risk_set_sums(n, sets)
    M = _risk_set_sums((n[:, None] * Z).T, sets).T / S0[:, None]
    a = n * _risk_set_weights(sets.d / S0, sets)
    return Z.T @ (a[:, None] * Z) - M.T @ (sets.d[:, None] * M)


def _warn_if_monotone(fits: list) -> str:
    """One warning for the causes, among the per-cause fits ``fits``
    (``_fit_cause``'s dicts), whose partial likelihood has no finite
    maximum; as ``CoxPH`` warns (``cox_ph.warn_monotone``), naming the
    cause where the model has more than one. Then one for the causes whose
    search did not reach a verified maximum. Returns the model's
    ``maximum``, the worst of the causes'."""
    runaway = [(fit["cause"], fit["runaway"]) for fit in fits]
    runaway = [(cause, coefs) for cause, coefs in runaway if coefs]
    if len(fits) == 1 and runaway:
        warn_monotone(str(runaway[0][1]))
    elif runaway:
        warn_monotone(
            " and ".join(
                "{} (cause {!r})".format(coefs, cause)
                for cause, coefs in runaway
            )
        )
    unverified = [f["cause"] for f in fits if f["maximum"] == "unverified"]
    if unverified:
        warn_unverified(
            "The Fine-Gray fit"
            + ("" if len(fits) == 1 else " of cause(s) {}".format(unverified))
        )
    return combined_maximum(f["maximum"] for f in fits)


def _cumhaz_at_origin(
    beta: npt.NDArray,
    center: npt.NDArray,
    Z: npt.NDArray,
    cumhaz: npt.NDArray,
) -> npt.NDArray:
    """The baseline cumulative subdistribution hazard fitted at the
    covariate ``center`` moved to ``Z = 0``, ``cumhaz * exp(-beta'center)``
    on the log scale (#463); refused, pointing to ``center=True``, where
    that over- or underflows or ``exp(beta'Z)`` overflows on the rows
    ``Z``."""
    shift = float(np.dot(beta, center))
    with np.errstate(all="ignore"):
        lp = np.asarray(Z, dtype=float) @ beta
        out = np.exp(np.log(cumhaz) - shift)
    tiny = np.finfo(float).tiny
    if not (
        np.all(np.abs(lp) < LOG_MAX)
        and np.all(np.isfinite(out))
        and np.all(out[cumhaz > 0] >= tiny)
    ):
        raise baseline_at_origin_error(
            "baseline cumulative subdistribution hazard", center, shift, -shift
        )
    return out


def paired_covariate_rows(Z: npt.ArrayLike, n_x: int, p: int) -> npt.NDArray:
    """
    The covariate row for each of ``n_x`` query times, shape ``(n_x, p)``.

    ``Z`` is one covariate vector (a scalar for one covariate, a 1-D array
    of ``p`` values or a single row), used at every time, or one row per
    time, paired with the times in order. Any other shape is refused: a
    ``Z`` of several rows used to be flattened into one long vector (a
    shape error from the matrix product) or broadcast against the times.
    """
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim <= 1:
        Z_arr = Z_arr.reshape(1, -1)
    if Z_arr.ndim != 2 or Z_arr.shape[1] != p:
        raise ValueError(
            "Z must hold {} covariate(s) per row, got shape {}.".format(
                p, np.shape(Z)
            )
        )
    if Z_arr.shape[0] not in (1, n_x):
        raise ValueError(
            "Z has {} rows for {} times: give one covariate row, used at "
            "every time, or one row per time.".format(Z_arr.shape[0], n_x)
        )
    return np.broadcast_to(Z_arr, (n_x, p))


class FineGrayModel(
    InformationCriteriaMixin, LinearPredictorMixin, SerialisableMixin
):
    """
    A fitted Fine-Gray subdistribution-hazard model for one cause of interest.

    The natural prediction is the cumulative incidence function :meth:`cif`;
    ``coefficients``/``se``/``p_values`` describe the (log) subdistribution
    hazard ratios.

    ``log_likelihood`` is the maximised weighted partial log-likelihood
    (``cmprsk::crr``'s ``loglik``), ``neg_ll()`` its negative, and
    :meth:`aic`, :meth:`aic_c` and :meth:`bic` penalise it by the
    estimated coefficients, BIC's sample size being the events of the
    cause of interest (#604). They compare Fine-Gray models of the same
    cause on the same data; the weighted partial likelihood is not the
    likelihood of the data, so they do not compare it with another kind
    of model.
    """

    #: The cause of interest the subdistribution hazard is of.
    cause: Any
    #: The coefficients (``beta``, the log subdistribution hazard
    #: ratios), their standard errors, Wald p-values and covariance.
    coefficients: npt.NDArray
    se: npt.NDArray
    p_values: npt.NDArray
    #: The coefficients' covariance, ``covariance()`` (#605).
    _covariance: npt.NDArray
    #: ``covariance()``'s name before v0.23, for one release.
    cov = RenamedToMethod("covariance", "_covariance")
    #: The baseline subdistribution cumulative hazard: its step times and
    #: values.
    _times: npt.NDArray
    _cumhaz: npt.NDArray
    #: The negative partial log-likelihood at the fit.
    _neg_ll: float
    #: The optimiser's result (``None`` on a restored model).
    res: Any
    #: What the fit reached, one of ``MAXIMUM_STATES``
    #: (``surpyval.utils.no_maximum``), as its warnings say; ``"unknown"``
    #: for a model restored from a dict saved without it.
    maximum: str
    #: The negative weighted partial log-likelihood the fit maximised, of
    #: the coefficients it did not alias, on the centred covariates (less
    #: its value at 0); ``None`` on a restored model (not saved).
    _objective: "Callable | None"

    def __init__(self, fit: dict) -> None:
        self.cause = fit["cause"]
        self.coefficients = fit["beta"]
        self.beta = fit["beta"]
        #: The covariate point the baseline is at, and ``phi`` relative to:
        #: zeros (``Z = 0``) by default, the covariate means for a fit with
        #: ``center=True`` (#463).
        self.center = np.asarray(
            fit.get("center", np.zeros(np.size(fit["beta"]))), dtype=float
        )
        self.se = fit["se"]
        self.p_values = fit["p_values"]
        self._covariance = fit["cov"]
        self._times = fit["baseline_times"]
        self._cumhaz = fit["baseline_cumhaz"]
        self._neg_ll = fit["neg_ll"]
        self._ic_n = fit.get("ic_n")
        self.res = fit["res"]
        self.maximum = fit.get("maximum", "unknown")
        self._objective = fit.get("objective")

    _ALIASED_WHY = (
        "a constant column, which the baseline subdistribution hazard "
        "absorbs, or a linear combination of the others"
    )

    def covariance(self) -> npt.NDArray:
        """The coefficients' covariance: the inverse of the weighted
        partial likelihood's information at the fit (a ``nan`` row and
        column for an aliased coefficient). ``cov``, its name before
        v0.23, still gives it, with a ``DeprecationWarning``, until
        v0.24."""
        return self._covariance

    def standard_errors(self) -> npt.NDArray:
        """The coefficients' standard errors, from :meth:`covariance`."""
        return self.se

    def _ic_k(self) -> int:
        # The estimated coefficients (an aliased one, nan, is not).
        return int(np.isfinite(np.asarray(self.beta, dtype=float)).sum())

    def _ic_sample_size_from_data(self) -> float:
        raise ValueError(
            "A Fine-Gray model saved before v0.23 does not store its number "
            "of events of interest, BIC's sample size; refit it."
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted Fine-Gray model to a plain, JSON-serialisable
        dict.

        Stores the coefficients and their covariance, plus the fitted
        subdistribution baseline cumulative-hazard step arrays (and, for a
        fit with ``center=True``, the covariate ``center`` they are at,
        which makes the dict schema 2: a schema-1 reader would take the
        baseline for that at ``Z = 0``), so the reloaded model reproduces
        ``cif``/``sf`` exactly and can still report the coefficient summary.
        The optimiser objects are not stored.
        """
        out = {
            "model": "FineGrayModel",
            # native type: a numpy scalar label breaks JSON/BSON
            "cause": to_native(self.cause),
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "se": np.asarray(self.se, dtype=float).tolist(),
            "p_values": np.asarray(self.p_values, dtype=float).tolist(),
            # The key every model's dict stores it under (#605).
            "covariance": np.asarray(self._covariance, dtype=float).tolist(),
            "baseline_times": np.asarray(self._times, dtype=float).tolist(),
            "baseline_cumhaz": np.asarray(self._cumhaz, dtype=float).tolist(),
            # The key every model's dict stores it under (#605).
            "_neg_ll": float(self._neg_ll),
            **maximum_entry(self.maximum),
        }
        if self._ic_n is not None:
            out["ic_n"] = float(self._ic_n)
        if np.any(self.center):
            out["center"] = np.asarray(self.center, dtype=float).tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "FineGrayModel":
        """Rebuild a Fine-Gray model from a :meth:`to_dict` dictionary."""
        require_model_tag(model_dict, "FineGrayModel", "a Fine-Gray model")
        beta = np.array(model_dict["beta"], dtype=float)
        # No "center" (a default fit, or one saved before #463): the
        # baseline is at Z = 0.
        center = np.array(
            model_dict.get("center", np.zeros(beta.size)), dtype=float
        )
        if center.shape != beta.shape:
            raise ValueError(
                "The model dict's 'center' has {} value(s) for {} "
                "coefficient(s).".format(center.size, beta.size)
            )
        return cls(
            {
                "cause": label_from_native(model_dict["cause"]),
                "beta": beta,
                "center": center,
                "se": np.array(model_dict["se"], dtype=float),
                "p_values": np.array(model_dict["p_values"], dtype=float),
                # "cov" is the key of a dict written before v0.23.
                "cov": np.array(
                    model_dict.get("covariance", model_dict.get("cov")),
                    dtype=float,
                ),
                "baseline_times": np.array(
                    model_dict["baseline_times"], dtype=float
                ),
                "baseline_cumhaz": np.array(
                    model_dict["baseline_cumhaz"], dtype=float
                ),
                # "neg_ll" is the key of a dict written before v0.23.
                "neg_ll": model_dict.get(
                    "_neg_ll", model_dict.get("neg_ll", np.nan)
                ),
                "ic_n": cls._restored_ic_n(model_dict),
                "res": None,
                "maximum": restored_maximum(model_dict),
            }
        )

    def phi(self, Z: npt.ArrayLike) -> npt.NDArray:
        """The subdistribution hazard multiplier
        :math:`e^{\\beta' (Z - \\text{center})}`, relative to a unit at
        ``center``, where the baseline is (``Z = 0`` unless fitted with
        ``center=True``), one value per row of ``Z`` (a scalar for a single
        covariate vector). It can overflow to ``inf`` on covariates far
        from ``center``; :meth:`cif` does not, as it combines it with the
        baseline on the log scale."""
        with np.errstate(over="ignore"):
            return np.exp(
                (np.asarray(Z, dtype=float) - self.center) @ self._coef()
            )

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """
        Cumulative incidence of the cause of interest at times ``x``:
        ``1 - exp(-Lambda0(x) * exp(beta'(Z - center)))``, the baseline
        ``Lambda0`` that of a unit at ``center``. ``Z`` is one covariate
        vector (a 1-D array or a single row), used at every time, or one
        row per time in ``x`` (row ``i`` with ``x[i]``). The CIF is flat
        before the first event time and after the last (the baseline is a
        step function estimated only on the observed range). A missing
        (``NaN``) time or covariate gives ``nan`` in its place.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        rows = paired_covariate_rows(Z, x.size, np.size(self.beta))
        H0 = step_at(self._times, self._cumhaz, x, before=0.0)
        # step_at reads a nan time as the value after the last jump.
        H0 = np.where(np.isnan(x), np.nan, H0)
        # H0 * exp(beta'(Z - center)) on the log scale: a baseline at Z = 0
        # far from the data is tiny and the multiplier huge (#463).
        with np.errstate(divide="ignore", over="ignore"):
            H = np.exp(np.log(H0) + (rows - self.center) @ self._coef())
        return -np.expm1(-H)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """One minus the cumulative incidence (the cause-of-interest-free
        probability under the subdistribution)."""
        return 1.0 - self.cif(x, Z)

    def __repr__(self) -> str:
        lines = [
            "Fine-Gray Subdistribution Hazard Model",
            "======================================",
            f"Cause of interest   : {self.cause}",
            "Coefficients (beta'Z acts on the subdistribution hazard):",
        ]
        for i, (b, s, p) in enumerate(zip(self.beta, self.se, self.p_values)):
            lines.append(f"   beta_{i}  :  {b: .6f}  (se {s:.6f}, p {p:.4f})")
        return "\n".join(lines)


class FineGray_:
    """
    The Fine-Gray subdistribution-hazards regression for one cause of a
    competing-risks problem: the covariates act proportionally on the
    *subdistribution* hazard of the cause of interest, so a coefficient
    describes its effect on that cause's cumulative incidence directly,

    .. math::
        F_k(t \\mid Z) = 1 - \\exp\\left(-\\Lambda_{k0}(t)\\,
        e^{\\beta' Z}\\right).

    Estimated by inverse-probability-of-censoring weighting (IPCW), with one
    Kaplan-Meier censoring distribution for the whole sample, so censoring
    is assumed not to depend on the covariates. A subject that failed from
    a competing cause at :math:`x_i` keeps the weight
    :math:`\\hat G(t-)/\\hat G(x_i-)` at a later event time :math:`t`: the
    censoring survival is taken just before each time, so a censoring tied
    with an event counts after it, as in R's ``cmprsk::crr``. ``FineGray``
    (from ``surpyval.univariate.competing_risks``) is an instance of this
    class; its ``fit`` returns a
    :class:`~surpyval.univariate.competing_risks.regression.fine_gray.FineGrayModel`.
    """

    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        e_col: str,
        Z_cols: "str | list[str]",
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        **fit_options: Any,
    ) -> FineGrayModel:
        """
        Fit the Fine-Gray model from the columns of a
        :class:`pandas.DataFrame`.

        The column names are passed in place of the arrays :meth:`fit`
        takes, with the names of every competing-risks ``fit_from_df``
        (``CompetingRisksProportionalHazards.fit_from_df`` too); ``event``
        and ``center`` are passed to :meth:`fit` unchanged.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            Column of observed times.
        e_col : str
            Column of event-type (cause) labels. Use ``None`` (or a
            blank/NaN cell) for a censored observation.
        Z_cols : str or list of str
            Covariate column(s), in the order of ``beta``.
        c_col : str, optional
            Column of censoring flags (0 observed, 1 right-censored).
        n_col : str, optional
            Column of counts per row.
        **fit_options
            ``event`` (the cause of interest) and ``center``, as for
            :meth:`fit`.

        Returns
        -------
        FineGrayModel
            The model :meth:`fit` returns for the same arrays.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval.univariate.competing_risks import FineGray
        >>> rng = np.random.default_rng(0)
        >>> z = rng.binomial(1, 0.5, 200).astype(float)
        >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * z)))
        >>> t_b = rng.exponential(1 / 0.05, 200)
        >>> df = pd.DataFrame({
        ...     "time": np.minimum(t_a, t_b).round(3),
        ...     "cause": np.where(t_a < t_b, "a", "b"),
        ...     "z": z,
        ... })
        >>> model = FineGray.fit_from_df(
        ...     df, x_col="time", e_col="cause", Z_cols="z", event="a"
        ... )
        >>> model.beta.round(3)
        array([0.663])
        """
        df = require_frame(df)
        arrays = {
            "x": frame_column(df, x_col, "x_col", time=True),
            "Z": frame_columns(df, Z_cols, "Z_cols").astype(float),
            "e": cause_column(df, e_col),
        }
        if c_col is not None:
            arrays["c"] = frame_column(df, c_col, "c_col")
        if n_col is not None:
            arrays["n"] = frame_column(df, n_col, "n_col")
        names = {"x": "x_col", "Z": "Z_cols", "e": "e_col"}
        names |= {"c": "c_col", "n": "n_col"}
        return call_fit(self, arrays, names, fit_options)

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        event: Any = None,
        center: bool = False,
    ) -> FineGrayModel:
        """
        Fit the Fine-Gray model for a cause of interest.

        Parameters
        ----------
        x : array_like
            Observed times.
        Z : ndarray
            Covariate matrix, one row per observation. Rows with a missing
            (``NaN``) or infinite covariate are dropped, with a warning.
        e : array_like
            Event-type (cause) labels; ``None`` for a censored observation.
        c : array_like, optional
            Censoring flags (0 observed, 1 right-censored). Defaults to
            deriving them from ``e``: a missing event (``None``/``NaN``) is
            right-censored, any other is observed. Left/interval censoring is
            not supported.
        n : array_like, optional
            Counts per observation. Defaults to 1.
        event : optional
            The cause of interest (a label in ``e``). May be omitted only
            when the data contains a single event type. The fitted model
            keeps it as ``cause``.
        center : bool, optional
            ``False`` (the default) reports the baseline cumulative
            subdistribution hazard at ``Z = 0``; ``True`` reports it at the
            covariate means, stored as ``model.center``, and ``phi`` is
            then relative to them. The fit runs on centred covariates either
            way, so the coefficients and every prediction are the same; the
            default refuses, with a ``ValueError``, covariates so far from
            0 that the baseline there over- or underflows.

        Returns
        -------
        FineGrayModel
            The fitted model, with :meth:`~FineGrayModel.cif` prediction.

        Examples
        --------
        >>> from surpyval.univariate.competing_risks import FineGray
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
        >>> t_b = rng.exponential(1 / 0.05, 200)
        >>> t_c = rng.uniform(0, 20, 200)  # censoring times
        >>> x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
        >>> first = np.where(t_a < t_b, "a", "b")
        >>> e = np.where(t_c < np.minimum(t_a, t_b), None, first)
        >>> model = FineGray.fit(x, Z, e, event="a")
        >>> model.beta.round(3)
        array([0.908])
        >>> model.cif([5, 10], [[1]]).round(4)
        array([0.5808, 0.7395])
        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        causes = ordered_labels(e)
        if event is None:
            if len(causes) != 1:
                raise missing_cause_error(
                    f"A Fine-Gray model (the data have causes {causes})"
                )
            event = causes[0]
        elif event not in causes:
            raise unknown_cause_error(event, causes)

        fit = _fit_cause(x, Z, e, c, n, event, center)
        maximum = _warn_if_monotone([fit])
        model = FineGrayModel(fit)
        model.maximum = maximum
        return model


FineGray = FineGray_()
