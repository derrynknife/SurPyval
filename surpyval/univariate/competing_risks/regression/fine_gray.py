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
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
from autograd import grad, hessian
from autograd import numpy as anp
from autograd import value_and_grad
from autograd.tracer import getval
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm

from surpyval.diagnostics.mixin import DiagnoseMixin
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
from surpyval.univariate.parametric.fitters import is_local_minimum
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    dataframe_covariates,
    expand,
    warn_aliased,
)
from surpyval.univariate.regression._fit_skeleton import (
    LOG_MAX,
    baseline_at_origin_error,
    judge_search,
)
from surpyval.univariate.regression._partial_effects import (
    CompetingRisksPartialEffectsMixin,
)
from surpyval.univariate.regression._summary import (
    coefficient_names,
    coefficient_table,
)
from surpyval.univariate.regression.proportional_hazards.cox_likelihood import (  # noqa: E501
    newton_raphson,
)
from surpyval.univariate.regression.proportional_hazards.cox_likelihood import (  # noqa: E501
    strata_labels as _labels_and_missing,
)
from surpyval.univariate.regression.proportional_hazards.cox_ph import (
    PENALISED_ADVICE,
    warn_monotone,
)
from surpyval.univariate.regression.proportional_hazards.cox_separation import (  # noqa: E501
    FAR,
    information_collapsed,
    newton_converged,
    runoff_direction,
)
from surpyval.univariate.regression.regression_data import (
    LinearPredictorMixin,
    named_covariates,
)
from surpyval.utils import validate_fine_gray_inputs
from surpyval.utils.covariates import coefficient_floor
from surpyval.utils.dataframe import (
    call_fit,
    cause_column,
    column_arguments,
    frame_column,
    frame_columns,
    read_columns,
    require_frame,
)
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.ipcw import censoring_survival, step_at, step_left_limit
from surpyval.utils.linalg import safe_inv, standard_errors_of
from surpyval.utils.no_maximum import (
    combined_maximum,
    maximum_entry,
    restored_maximum,
    warn_unverified,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    check_alpha_ci,
    check_option,
    missing_cause_error,
    no_covariance_error,
    refuse_subdistribution_entry,
    unknown_cause_error,
)
from surpyval.utils.warnings import caller_stacklevel
from surpyval.utils.weights import (
    effective_sample_size,
    refuse_weighted,
    sandwich,
    warn_if_few_effective,
)

if TYPE_CHECKING:
    import pandas as pd

#: Which covariance ``covariance()``, ``standard_errors()``, ``summary()``
#: and ``p_values`` use unless asked: ``True`` the robust (sandwich) one of
#: Fine and Gray, ``False`` the model-based inverse information. The one
#: place the default is set.
ROBUST_BY_DEFAULT = True

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
    overflow (#606).

    A risk set whose rows are all far below the largest linear predictor
    then has a sum that underflows: far out along a run-off it was 0, and
    the likelihood ``log(0)``, a negative log-likelihood of ``-inf`` that
    every search takes for the best point there is (#760). Where a sum is
    below ``_LEAST_SUM`` the sums are taken in logs instead, each over its
    own risk set (:func:`_log_risk_set_sums`)."""
    eta = anp.dot(Zk, beta)
    if not np.all(np.isfinite(getval(eta))):
        # (a coefficient or linear predictor that is not finite)
        return np.nan
    shift = float(np.max(getval(eta))) if eta.size else 0.0
    weighted_exp = n_sorted * anp.exp(eta - shift)
    denom = _risk_set_sums(weighted_exp, sets)
    if not np.all(getval(denom) >= _LEAST_SUM):
        log_denom = _log_risk_set_sums(eta, n_sorted, sets)
        ll = anp.dot(nZk_event, beta) - anp.sum(
            sets.d * (log_denom - np.log(denom0))
        )
        return -ll
    ll = anp.dot(nZk_event, beta) - anp.sum(
        sets.d * (anp.log(denom / denom0) + shift)
    )
    return -ll


#: The least risk-set sum, of the weights shifted by the largest linear
#: predictor, that :func:`_weighted_neg_ll` uses as it is: below it the
#: sums are taken in logs (#760). A sum this large has lost nothing to
#: terms below the smallest normal number.
_LEAST_SUM = 1e-200

#: The largest linear predictor whose ``exp`` the baseline sums directly,
#: CoxPH's (#738); beyond it they are taken in logs (#760).
_DIRECT_ETA = 300.0

#: A finite stand-in for ``log(0)`` in :func:`_log_risk_set_sums`, whose
#: derivatives would be nan at ``logaddexp(-inf, -inf)``.
_LOG_NONE = -1e300


def _log_suffix_sums(log_v: Any) -> Any:
    """The log of each suffix sum of ``exp(log_v)``, by doubling: each of
    ``log2(N)`` steps adds to every position the partial sum ``2^k`` on,
    with ``logaddexp``, so that autograd can differentiate it (twice)."""
    rows = anp.shape(log_v)[0]
    step = 1
    while step < rows:
        later = anp.concatenate(
            [log_v[step:], anp.full(min(step, rows), _LOG_NONE)]
        )
        log_v = anp.logaddexp(log_v, later)
        step *= 2
    return log_v


def _log_risk_set_sums(
    eta: Any, n_sorted: npt.NDArray, sets: _RiskSets
) -> Any:
    """The log of :func:`_risk_set_sums` of ``n exp(eta)`` (rows in time
    order) at each event time, summed in logs over each risk set: the
    suffix of the rows still under observation, plus ``G(t-)`` times the
    competing failures before ``t``, each over ``G(x_i-)``. Differentiable
    by autograd; ``O(N log N)``, for a linear predictor whose sums
    underflow (:func:`_weighted_neg_ll`)."""
    with np.errstate(divide="ignore"):
        log_n = np.log(n_sorted)
        log_competing = np.log(sets.competing_over_G * n_sorted)
        log_G = np.log(sets.G_t)
    competing = np.isfinite(log_competing)
    log_n = np.where(np.isfinite(log_n), log_n, _LOG_NONE)
    suffix = _log_suffix_sums(log_n + eta)
    # The competing failures before each row: the suffix sums of the
    # reversed rows, moved on by one
    log_c = anp.where(
        competing, np.where(competing, log_competing, 0.0) + eta, _LOG_NONE
    )
    before = _log_suffix_sums(log_c[::-1])[::-1]
    before = anp.concatenate([anp.full(1, _LOG_NONE), before[:-1]])
    log_G = np.where(np.isfinite(log_G), log_G, _LOG_NONE)
    return anp.logaddexp(suffix[sets.start], log_G + before[sets.start])


class _Part(NamedTuple):
    """One stratum of a Fine-Gray fit (the whole sample when it has no
    strata): its rows, in time order, as positions among all the rows,
    and its subdistribution risk sets, built with its own censoring
    distribution ``G``."""

    rows: npt.NDArray
    sets: Any


def _parts_neg_ll(parts: tuple, beta: Any) -> Any:
    """The negative weighted partial log-likelihood summed over the
    strata, each ``parts`` entry the arguments of :func:`_weighted_neg_ll`
    for one stratum but ``beta``; differentiable by autograd. A
    module-level function, so the objective (a ``functools.partial`` of
    it) pickles (#573)."""
    total = 0.0
    for n_sorted, sets, denom0, Zk, nZk_event in parts:
        total = total + _weighted_neg_ll(
            n_sorted, sets, denom0, Zk, nZk_event, beta
        )
    return total


def _fit_cause(
    x: npt.NDArray,
    Z: npt.NDArray,
    e: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    cause: Any,
    center: bool = False,
    w: npt.NDArray | None = None,
    strata: npt.NDArray | None = None,
    cluster: npt.NDArray | None = None,
    common_censoring: bool = False,
) -> dict:
    """
    Fit the Fine-Gray subdistribution-hazard model for a single ``cause``.

    Returns a dict with the fitted coefficients, their robust covariance
    (``"cov"``: Fine and Gray's sandwich, :func:`_robust_covariance`), the
    p-values on it and the model-based covariance (``"model_cov"``, the
    inverse information), the baseline cumulative subdistribution hazard
    (as sorted event times and the cumulative hazard at each), the
    optimiser result, and the
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

    With weights ``w`` (one per row, beside the counts ``n``) every sum
    takes ``n w`` in place of ``n``: the events, the risk sets and the
    censoring distribution ``G`` (the Kaplan-Meier of the censoring
    weighted by ``n w``), so whole-number weights give the fit of the rows
    counted that many times. The covariance is then Fine and Gray's
    sandwich with the sampling weights (:func:`_robust_covariance`),
    which is the unweighted one when ``w = 1``, and there is no
    model-based one.

    With ``strata`` (an integer code per row, 0, 1, ...) the model is
    stratified as R's ``crrSC::crrs`` (``ctype = 1``; Zhou, Latouche,
    Rocha and Fine, 2011): each stratum has its own baseline
    subdistribution hazard, its own risk sets and its own censoring
    distribution ``G``, and the partial likelihood, its information and
    the robust covariance's meat are sums over the strata. With
    ``cluster`` (an integer code per row) the units of a cluster may be
    dependent: the estimate is the same (an independence working model),
    and the robust covariance sums each cluster's influence before
    squaring it, as R's ``crrSC::crrc`` (Zhou, Fine, Latouche and Labopin,
    2012). With ``common_censoring`` the strata share one censoring
    distribution, of all the rows (``crrs`` with ``ctype = 2``, for many
    small strata), and the robust covariance's censoring term is that of
    the one ``G``.
    """
    counts = n
    if w is not None:
        n = n * w
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
    # reach zero and the ratio G(t-)/G(x_i-) needs no guard. A stratified
    # fit takes G within each stratum, from that stratum's rows.
    parts = _strata_parts(
        x, c, n, is_event, is_competing, strata, common_censoring
    )
    # A stratum without events of the cause adds nothing to the fit.
    fitted = [part for part in parts if part.sets.times.size]

    Z_raw = Z
    mean = (n @ Z) / n.sum()
    Z = Z - mean
    n_event = n[is_event]
    # The rows of every stratum with events, each in time order, which the
    # risk-set sums run over; and per stratum the rows, their counts and
    # the events' sum of n * Z, the linear term of the log-likelihood.
    rows = np.concatenate([part.rows for part in fitted])
    Z_sorted = Z[rows]
    nZ_events = [(n * is_event)[part.rows] @ Z[part.rows] for part in fitted]
    # The optimiser minimises the negative log-likelihood less its value at
    # beta = 0, sum d log(denom0): near the maximum its steps change the
    # likelihood (5e4 at 1e4 rows) by less than its last digit. BFGS
    # stopped there on "precision loss" in 10 of 450 fits at 3e3 rows
    # before #517, and in 17 with plain cumulative sums; with this and the
    # blocked sums of _cumsum, in 1.
    denom0s = [_risk_set_sums(n[part.rows], part.sets) for part in fitted]
    offset = float(
        sum(
            part.sets.d @ np.log(denom0)
            for part, denom0 in zip(fitted, denom0s)
        )
    )

    def partial_neg_ll(cols: npt.NDArray) -> tuple:
        """The objective on the columns ``cols``, differentiable by
        autograd (the no-maximum check and the information take its
        derivatives); the objective with its gradient for BFGS, ``-(nZ_event
        - sum_j d_j S1_j / S0_j)`` (``S1`` the risk-set sums of ``w Z``), by
        hand: a fit of 1e5 rows took 1.2 s with autograd's gradient, 0.7 s
        with this; and the gradient with the information, by hand, for
        Newton-Raphson. Each is a sum over the strata.

        Each shifts the linear predictor by its largest value inside the
        risk-set sums and adds the shift back outside the logarithm, as
        ``CoxPH`` does, so ``exp(beta'Z)`` cannot overflow at any
        coefficients: on covariates of order 1e4 it did at the search's
        first step, and the fit failed ("SVD did not converge", #606). The
        objective is a ``functools.partial`` of a module-level function,
        not a closure, so the model, which keeps it, pickles (#573)."""
        args = tuple(
            (n[part.rows], part.sets, denom0, Z[part.rows][:, cols], nZ[cols])
            for part, denom0, nZ in zip(fitted, denom0s, nZ_events)
        )
        # One stratum (the whole sample) keeps the objective a partial of
        # _weighted_neg_ll itself.
        neg_ll = (
            functools.partial(_weighted_neg_ll, *args[0])
            if len(args) == 1
            else functools.partial(_parts_neg_ll, args)
        )

        def shifted(
            n_sorted: npt.NDArray, sets: _RiskSets, Zk: npt.NDArray, beta: Any
        ) -> tuple:
            # The weights e^(eta - shift), their risk-set sums and the shift
            eta = Zk @ beta
            shift = float(eta.max()) if eta.size else 0.0
            weighted_exp = n_sorted * np.exp(eta - shift)
            return weighted_exp, _risk_set_sums(weighted_exp, sets), shift

        def underflows(denom: npt.NDArray) -> bool:
            # A risk set's sum lost to underflow: the derivatives are then
            # autograd's, of the objective in logs (_weighted_neg_ll, #760)
            # (not a coefficient that is not finite, nan either way)
            finite = np.all(np.isfinite(denom))
            return bool(finite and not np.all(denom >= _LEAST_SUM))

        def value_and_gradient(beta: npt.NDArray) -> tuple:
            value, gradient = 0.0, np.zeros(cols.size)
            for n_sorted, sets, denom0, Zk, nZk_event in args:
                weighted_exp, denom, shift = shifted(n_sorted, sets, Zk, beta)
                if underflows(denom):
                    one = functools.partial(
                        _weighted_neg_ll, n_sorted, sets, denom0, Zk, nZk_event
                    )
                    v, gr = value_and_grad(one)(beta)
                    value += float(v)
                    gradient += np.asarray(gr, dtype=float)
                    continue
                value -= nZk_event @ beta - np.sum(
                    sets.d * (np.log(denom / denom0) + shift)
                )
                # sum_j d_j S1_j / S0_j = sum_i w_i Z_i sum_j W_ji d_j / S0_j
                weights = _risk_set_weights(sets.d / denom, sets)
                gradient -= nZk_event - Zk.T @ (weighted_exp * weights)
            return value, gradient

        def gradient_and_information(beta: npt.NDArray) -> tuple:
            # The information sum_j d_j (S2_j / S0_j - M_j M_j'), M_j =
            # S1_j / S0_j, in O(N p^2) as at beta = 0
            # (_information_at_zero); the shift cancels in every ratio.
            gradient = np.zeros(cols.size)
            information = np.zeros((cols.size, cols.size))
            for n_sorted, sets, denom0, Zk, nZk_event in args:
                weighted_exp, denom, _ = shifted(n_sorted, sets, Zk, beta)
                if underflows(denom):
                    one = functools.partial(
                        _weighted_neg_ll, n_sorted, sets, denom0, Zk, nZk_event
                    )
                    gradient += grad(one)(beta)
                    information += hessian(one)(beta)
                    continue
                a = weighted_exp * _risk_set_weights(sets.d / denom, sets)
                M = _risk_set_sums((weighted_exp[:, None] * Zk).T, sets).T
                M = M / denom[:, None]
                gradient -= nZk_event - Zk.T @ a
                information += Zk.T @ (a[:, None] * Zk) - M.T @ (
                    sets.d[:, None] * M
                )
            return gradient, information

        return neg_ll, value_and_gradient, gradient_and_information

    # Coefficients the weighted partial likelihood does not depend on are
    # aliased, as CoxPH's are (#476): fitted on the other columns, and
    # reported as nan. (Within strata, a column constant in every stratum
    # carries no information, and is aliased too.)
    p = Z.shape[1]
    every = np.arange(p)
    neg_ll, value_and_gradient, newton_derivatives = partial_neg_ll(every)
    aliased = aliased_columns(
        sum(
            _information_at_zero(Z[part.rows], n[part.rows], part.sets)
            for part in fitted
        ),
        Z.shape[0],
        constant_columns(Z_raw),
        float(n_event.sum()) * (n @ Z**2) / n.sum(),
    )
    kept = np.setdiff1d(every, aliased)
    if aliased.size:
        warn_aliased(
            aliased,
            "the partial likelihood does not depend on them (a constant "
            "column"
            + ("" if strata is None else ", or one constant within strata")
            + ", or a linear combination of the others within the "
            "risk sets, as the columns of every level of a factor are)",
        )
        neg_ll, value_and_gradient, newton_derivatives = partial_neg_ll(kept)

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
        converged = res is not None and newton_converged(res)
        if res is None:
            res = minimize(value_and_gradient, beta0, jac=True, method="BFGS")
        # The answer must be a verified maximum, polished if it is not
        # (BFGS's absolute tolerance on the gradient is not scale free),
        # each coefficient in its own covariate's units (#577).
        floor = coefficient_floor(
            kept.size, [(k, k) for k in range(kept.size)], Z_sorted[:, kept]
        )
        verdict = judge_search(
            neg_ll,
            res,
            [(k, int(kept[k])) for k in range(kept.size)],
            beta0,
            float(n_event.sum()),
            floor=floor,
        )
        res, derivatives = verdict.res, verdict.derivatives
        # A covariate, or a combination of them, that separates the events
        # of interest from the rest of their risk sets (a level with none
        # of them) drives the coefficients to infinity; BFGS stops where
        # the rise is below its tolerance and reports success (-12.9 on
        # such data). Where the search gives cause, the data decide, as
        # for CoxPH (#746).
        runoff = _runoff_of_cause(
            res,
            verdict.maximum,
            converged,
            information0,
            derivatives,
            newton_derivatives,
            Z_sorted[:, kept],
            float(n_event.sum()),
            floor,
            (x, Z[:, kept], is_event, is_competing, counts, strata),
        )
        off = runoff[0]
        runaway = [int(kept[k]) for k in off]
        maximum, proportion = runoff[1], runoff[2]
    else:
        # Every coefficient aliased: nothing to fit.
        res = OptimizeResult(
            x=beta0, fun=float(neg_ll(beta0)), success=True, nit=0
        )
        derivatives, runaway, maximum = None, [], "verified"
        off, proportion = np.zeros(0, dtype=int), None
    # The negative log-likelihood itself.
    res.fun = float(res.fun) + offset
    beta = res.x

    # Standard errors from the inverse observed information, the Hessian
    # the check just took.
    if derivatives is not None:
        H = derivatives[0]
    elif kept.size:
        H = hessian(neg_ll)(beta)
    else:
        # (every coefficient aliased: autograd cannot take a Hessian in
        # none, and raised "need at least one array to stack")
        H = np.zeros((0, 0))
    # Where the coefficients ran off far, the Hessian can be nan, and
    # inverting it raised "SVD did not converge" (#746)
    cov = safe_inv(H) if np.all(np.isfinite(H)) else np.full(H.shape, np.nan)
    # The model-based covariance, the inverse information (crr's invinf).
    # It is not a weighted fit's covariance, so none is kept for one.
    model_cov = None if w is not None else cov.copy()
    if kept.size and np.all(np.isfinite(cov)):
        # The robust (sandwich) covariance, the inverse information the
        # bread: Fine and Gray's, with the term for the estimated
        # censoring distribution, the weights' too (_robust_covariance)
        # (far out along a run-off a risk set's sum underflows; the
        # covariance is then nan, as the coefficients running off are)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = _fit_robust_covariance(
                cov,
                Z[:, kept],
                res.x,
                parts,
                fitted,
                (counts, w, is_event, c == 1, x),
                strata is None and cluster is None,
                cluster,
                common_censoring,
            )
    # A coefficient running off has no standard error, as in CoxPH (#648)
    cov[off, :] = np.nan
    cov[:, off] = np.nan
    if model_cov is not None:
        model_cov[off, :] = np.nan
        model_cov[:, off] = np.nan
    var = np.diag(cov)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(var > 0, var, np.nan))
        z_score = beta / se
    p_values = 2.0 * norm.sf(np.abs(z_score))
    if aliased.size:
        # 0 for the baseline and predictions; the model reports nan.
        beta = expand(beta, kept, p)
        se = expand(se, kept, p)
        p_values = expand(p_values, kept, p)
        full = np.full((p, p), np.nan)
        full[np.ix_(kept, kept)] = cov
        cov = full
        if model_cov is not None:
            full = np.full((p, p), np.nan)
            full[np.ix_(kept, kept)] = model_cov
            model_cov = full
        beta = np.where(np.isnan(beta), 0.0, beta)

    # Breslow baseline cumulative subdistribution hazard, in each stratum:
    # at each event-of-interest time, dLambda0 = (events there) / (weighted
    # risk set there). Past a linear predictor of 300 (a coefficient
    # running off) exp overflowed, with numpy's warnings: the sums are then
    # taken in logs (#760).
    baselines = [
        _breslow_baseline(part, Z, n, beta, None if center else mean, Z_raw)
        for part in parts
    ]
    if not center:
        mean = np.zeros_like(mean)

    if aliased.size:
        beta = np.array(beta, dtype=float)
        beta[aliased] = np.nan
    out = {
        "cause": cause,
        "beta": beta,
        "center": mean,
        "p_values": p_values,
        "cov": cov,
        "model_cov": model_cov,
        "baseline_times": baselines[0][0],
        "baseline_cumhaz": baselines[0][1],
        "neg_ll": float(res.fun),
        # BIC's sample size: the events of interest, the terms of the
        # partial likelihood (#604; Kuk and Varadhan's BIC_cr).
        "ic_n": float(n_event.sum()),
        "res": res,
        "runaway": runaway,
        "runaway_proportion": proportion,
        "maximum": maximum,
        "objective": neg_ll,
        # The rows the fit used, which diagnose() checks the model on
        # (#874): the covariates as given (not centred), the events of the
        # cause of interest and the competing failures.
        "fit_data": {
            "x": x,
            "Z": Z_raw,
            "c": c,
            "n": counts,
            "w": w,
            "is_event": is_event,
            "is_competing": is_competing,
            "strata": strata,
            "cluster": cluster,
        },
        "weighted": w is not None,
        "effective_n": (
            None if w is None else effective_sample_size(counts, w)
        ),
    }
    if strata is not None:
        out["strata_baselines"] = baselines
    if cluster is not None:
        out["n_clusters"] = int(cluster.max()) + 1
    return out


def _strata_parts(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    is_event: npt.NDArray,
    is_competing: npt.NDArray,
    strata: npt.NDArray | None,
    common_censoring: bool,
) -> list:
    """Each stratum's rows and subdistribution risk sets (one part, the
    whole sample, without strata), with the censoring distribution ``G``
    of the stratum's rows, or of all the rows (``common_censoring``)."""
    n_strata = 1 if strata is None else int(strata.max()) + 1
    if common_censoring:
        common = censoring_survival(x, c == 1, n)
        G_common = step_left_limit(*common, x, before=1.0)
    parts = []
    for s in range(n_strata):
        g = (
            np.arange(x.size)
            if strata is None
            else np.flatnonzero(strata == s)
        )
        if common_censoring:
            (g_times, g_vals), G_x = common, G_common[g]
        else:
            g_times, g_vals = censoring_survival(x[g], c[g] == 1, n[g])
            G_x = step_left_limit(g_times, g_vals, x[g], before=1.0)
        sets = _risk_sets(
            x[g], n[g], is_event[g], is_competing[g], G_x, g_times, g_vals
        )
        parts.append(_Part(g[sets.order], sets))
    return parts


def _breslow_baseline(
    part: _Part,
    Z: npt.NDArray,
    n: npt.NDArray,
    beta: npt.NDArray,
    mean: npt.NDArray | None,
    Z_raw: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """One stratum's Breslow baseline cumulative subdistribution hazard,
    its event times and values: at each event-of-interest time, dLambda0
    = (events there) / (weighted risk set there), for a unit at the
    covariate means (``Z`` centred), moved to ``Z = 0`` unless ``mean`` is
    ``None``. Past a linear predictor of 300 (a coefficient running off)
    exp overflowed, with numpy's warnings: the sums are then taken in
    logs (#760). A stratum without events has none."""
    if not part.sets.times.size:
        return part.sets.times, np.zeros(0)
    eta = Z[part.rows] @ beta
    n_sorted = n[part.rows]
    if np.all(np.abs(eta) <= _DIRECT_ETA):
        increments = part.sets.d / _risk_set_sums(
            n_sorted * np.exp(eta), part.sets
        )
    else:
        with np.errstate(over="ignore", under="ignore"):
            log_denom = _log_risk_set_sums(eta, n_sorted, part.sets)
            increments = np.exp(np.log(part.sets.d) - log_denom)
    cumhaz = np.cumsum(increments)
    if mean is not None:
        cumhaz = _cumhaz_at_origin(beta, mean, Z_raw, cumhaz)
    return part.sets.times, cumhaz


def _fit_robust_covariance(
    bread: npt.NDArray,
    Z: npt.NDArray,
    beta: npt.NDArray,
    parts: list,
    fitted: list,
    rows: tuple,
    plain: bool,
    cluster: npt.NDArray | None,
    common_censoring: bool,
) -> npt.NDArray:
    """The robust covariance of a fit: :func:`_robust_covariance` for a
    fit without strata or clusters (``plain``), else
    :func:`_stratified_robust_covariance`. ``rows`` are ``(n, w,
    is_event, is_censored, x)`` in the rows' own order, ``Z`` centred."""
    n, w, is_event, is_censored, x = rows
    if plain:
        (part,) = parts
        at = part.rows
        return _robust_covariance(
            bread,
            Z[at],
            beta,
            part.sets,
            n[at],
            None if w is None else w[at],
            is_event[at],
            is_censored[at],
            x[at],
        )
    return _stratified_robust_covariance(
        bread,
        Z,
        beta,
        fitted,
        n,
        w,
        is_event,
        is_censored,
        x,
        cluster,
        common_censoring,
    )


def _stratified_robust_covariance(
    bread: npt.NDArray,
    Z: npt.NDArray,
    beta: npt.NDArray,
    parts: list,
    n: npt.NDArray,
    w: npt.NDArray | None,
    is_event: npt.NDArray,
    is_censored: npt.NDArray,
    x: npt.NDArray,
    cluster: npt.NDArray | None,
    common_censoring: bool = False,
) -> npt.NDArray:
    """
    The robust covariance of a stratified or clustered Fine-Gray fit,
    ``bread M bread``. Each unit's influence on the score,
    :func:`_score_residuals`, is that of :func:`_robust_covariance` taken
    within its stratum (its risk sets, baseline and censoring
    distribution); a unit in a stratum without events of the cause has
    none. Without clusters ``M`` is the sum of their outer products,
    ``sum_i n_i w_i^2 r_i r_i'``, R's ``crrSC::crrs`` with ``ctype = 1``;
    with clusters it is the sum over the clusters of the outer product of
    each cluster's total influence, ``s_k = sum_{i in k} n_i w_i r_i``,
    R's ``crrSC::crrc``. With ``common_censoring`` (one censoring
    distribution for every stratum, ``crrs`` with ``ctype = 2``) a unit's
    censoring term is that of the one ``G``, which every stratum's
    weights use: :func:`_censoring_residuals` with ``q`` summed over the
    strata and the censorings and numbers at risk of all the rows. The
    rows are in their own order (``Z`` centred); each part's rows are
    positions among them.
    """
    resid = np.zeros(Z.shape)
    weights = np.ones(n.size) if w is None else w
    if not common_censoring:
        for part in parts:
            at = part.rows
            resid[at] = _score_residuals(
                Z[at],
                beta,
                part.sets,
                n[at],
                None if w is None else w[at],
                is_event[at],
                is_censored[at],
                x[at],
            )
    else:
        times = np.unique(x[is_censored])
        q = np.zeros((times.size, Z.shape[1]))
        for part in parts:
            at = part.rows
            risk, omega, dH, mean, resid[at] = _event_residuals(
                Z[at],
                beta,
                part.sets,
                n[at],
                None if w is None else w[at],
                is_event[at],
                x[at],
            )
            if times.size:
                q += _censoring_q(
                    Z[at], risk, part.sets, omega, x[at], dH, mean, times
                )
        if times.size:
            order = np.argsort(x, kind="mergesort")
            resid[order] += _censoring_martingale(
                q,
                times,
                (n * weights)[order],
                is_censored[order],
                x[order],
            )
    if cluster is None:
        return sandwich(bread, resid, n, weights)
    totals = np.zeros((int(cluster.max()) + 1, Z.shape[1]))
    np.add.at(totals, cluster, (n * weights)[:, None] * resid)
    out = bread @ (totals.T @ totals) @ bread
    return 0.5 * (out + out.T)


def _robust_covariance(
    bread: npt.NDArray,
    Z: npt.NDArray,
    beta: npt.NDArray,
    sets: _RiskSets,
    n: npt.NDArray,
    w: npt.NDArray | None,
    is_event: npt.NDArray,
    is_censored: npt.NDArray,
    x: npt.NDArray,
) -> npt.NDArray:
    """
    The robust (sandwich) covariance of a Fine-Gray fit, ``bread M
    bread``, ``bread`` the inverse of the information of the weighted
    partial likelihood and ``M`` the sum over the units of the outer
    products of their influence on the score.

    Without weights (``w`` is ``None``) it is Fine and Gray's (1999)
    variance, ``M = sum_i n_i (eta_i + psi_i)(eta_i + psi_i)'``, which R's
    ``cmprsk::crr`` reports as ``var``. ``eta_i`` is unit ``i``'s score
    residual (Lin and Wei, 1989) summed over its time at risk,

        eta_i = delta_i (Z_i - Zbar(x_i))
                - sum_t W_it exp(beta'Z_i) (Z_i - Zbar(t)) dLambda0(t),

    ``W_it`` its weight in the subdistribution risk set at the event time
    ``t`` (1 while under observation, ``G(t-)/G(x_i-)`` after a competing
    failure), ``Zbar(t)`` the risk set's weighted mean covariate and
    ``dLambda0`` the baseline's increments. ``psi_i`` is what the unit
    adds through the estimate of the censoring distribution ``G`` in those
    weights (:func:`_censoring_residuals`).

    With sampling weights ``w`` it is ``M = sum_i n_i w_i^2 (eta_i +
    psi_i)(eta_i + psi_i)'``, every sum in ``eta`` and ``psi`` (the risk
    sets, the baseline's increments, the censoring Kaplan-Meier and its
    at-risk counts) taken with ``n w``: a unit's influence is ``w_i``
    times what it would be unweighted. So ``w = 1`` is the unweighted
    variance, and scaling every weight changes nothing. Without ``psi``
    it would be R's ``coxph(Surv(fgstart, fgstop, fgstatus) ~ Z,
    weights = fgwt * w, cluster = id)`` on the counting-process data of
    the fit, which holds ``G`` fixed.

    The rows (``Z`` centred, the counts ``n``, weights ``w``, the
    events ``is_event``, the censorings ``is_censored`` and the times
    ``x``) are in time order (``sets.order``); every sum is O(N p)
    (:func:`_risk_set_sums`, :func:`_risk_set_weights`).
    """
    resid = _score_residuals(Z, beta, sets, n, w, is_event, is_censored, x)
    return sandwich(bread, resid, n, np.ones(n.size) if w is None else w)


def _score_residuals(
    Z: npt.NDArray,
    beta: npt.NDArray,
    sets: _RiskSets,
    n: npt.NDArray,
    w: npt.NDArray | None,
    is_event: npt.NDArray,
    is_censored: npt.NDArray,
    x: npt.NDArray,
) -> npt.NDArray:
    """Each unit's influence on the score, ``eta_i + psi_i`` of
    :func:`_robust_covariance` (one row per row, one unit of each), the
    rows in time order."""
    risk, omega, dH, mean, resid = _event_residuals(
        Z, beta, sets, n, w, is_event, x
    )
    resid += _censoring_residuals(
        Z, risk, sets, omega, is_censored, x, dH, mean
    )
    return resid


def _event_residuals(
    Z: npt.NDArray,
    beta: npt.NDArray,
    sets: _RiskSets,
    n: npt.NDArray,
    w: npt.NDArray | None,
    is_event: npt.NDArray,
    x: npt.NDArray,
) -> tuple:
    """The score residuals ``eta_i`` of :func:`_robust_covariance` (the
    rows in time order), with what the censoring term is built from:
    ``(risk, omega, dH, mean, eta)``, ``risk`` ``exp(beta'Z)`` (shifted),
    ``omega`` the rows' ``n w``, ``dH`` the baseline's increments and
    ``mean`` the risk sets' mean covariate at each event time."""
    with np.errstate(over="ignore", under="ignore"):
        eta = Z @ beta
        risk = np.exp(eta - (float(eta.max()) if eta.size else 0.0))
    omega = n if w is None else n * w
    S0 = _risk_set_sums(omega * risk, sets)
    S1 = _risk_set_sums((omega * risk)[None, :] * Z.T, sets).T
    mean = S1 / S0[:, None]
    # The hazard increments, on the scale of ``risk`` (the shift cancels
    # in their product)
    dH = sets.d / S0
    exposure = _risk_set_weights(dH, sets)
    exposure_mean = np.column_stack(
        [_risk_set_weights(dH * mean[:, k], sets) for k in range(Z.shape[1])]
    )
    resid = -risk[:, None] * (Z * exposure[:, None] - exposure_mean)
    events = np.flatnonzero(is_event)
    at = np.searchsorted(sets.times, x[events])
    resid[events] += Z[events] - mean[at]
    return risk, omega, dH, mean, resid


def _censoring_residuals(
    Z: npt.NDArray,
    risk: npt.NDArray,
    sets: _RiskSets,
    n: npt.NDArray,
    is_censored: npt.NDArray,
    x: npt.NDArray,
    dH: npt.NDArray,
    mean: npt.NDArray,
) -> npt.NDArray:
    """
    Each unit's contribution to the score through the estimated censoring
    distribution, Fine and Gray's ``psi_i`` (1999, section 3.2), as
    ``cmprsk::crr`` computes it:

        psi_i = c_i q(x_i) / Y(x_i) - sum_{u <= x_i} dc(u) q(u) / Y(u)^2,

    the integral of ``q / Y`` against the unit's censoring martingale.
    The sum is over the censoring times ``u``, with ``dc(u)`` the units
    censored there, ``Y(u)`` those with ``x >= u`` and ``c_i`` 1 for a
    censored unit. ``q(u)`` is how the score moves with the censoring
    hazard at ``u``: over the events at ``t >= u``, the part of the risk
    set's covariate spread carried by the competing failures before
    ``u``, whose weights ``G(t-)/G(x_j-)`` that hazard scales,

        q(u) = sum_{t >= u} dLambda0(t) G(t-)
               sum_{j: x_j < u} exp(beta'Z_j) (Z_j - Zbar(t)) / G(x_j-).

    The inner sum splits into a prefix sum over the competing failures
    (of ``exp(beta'Z)/G`` and of that times ``Z``) and the outer into a
    suffix sum over the event times, so every ``q`` is O(N p).

    The rows are in time order; ``risk`` is ``exp(beta'Z)`` (on any
    scale, with ``dH`` on its inverse), ``dH`` the baseline's increments
    and ``mean`` ``Zbar`` at each event time of ``sets``.
    """
    rows, p = Z.shape
    if not is_censored.any():
        return np.zeros((rows, p))
    times = np.unique(x[is_censored])
    q = _censoring_q(Z, risk, sets, n, x, dH, mean, times)
    return _censoring_martingale(q, times, n, is_censored, x)


def _censoring_martingale(
    q: npt.NDArray,
    times: npt.NDArray,
    n: npt.NDArray,
    is_censored: npt.NDArray,
    x: npt.NDArray,
) -> npt.NDArray:
    """``psi_i = c_i q(x_i) / Y(x_i) - sum_{u <= x_i} dc(u) q(u) / Y(u)^2``
    of :func:`_censoring_residuals` for each row (in time order, counts
    ``n``), from ``q`` at the censoring times ``times`` of the rows."""
    inv = np.searchsorted(times, x[is_censored])
    censored = np.bincount(inv, weights=n[is_censored], minlength=times.size)
    first = np.searchsorted(x, times, side="left")
    at_risk = _cumsum(n[::-1])[::-1][first]
    out = np.zeros((x.size, q.shape[1]))
    # The compensator up to each row's time, and the jump of a censored row
    drift = np.concatenate(
        [
            np.zeros((1, q.shape[1])),
            np.cumsum(q * (censored / at_risk**2)[:, None], axis=0),
        ]
    )
    out -= drift[np.searchsorted(times, x, side="right")]
    out[is_censored] += (q / at_risk[:, None])[inv]
    return out


def _censoring_q(
    Z: npt.NDArray,
    risk: npt.NDArray,
    sets: _RiskSets,
    n: npt.NDArray,
    x: npt.NDArray,
    dH: npt.NDArray,
    mean: npt.NDArray,
    times: npt.NDArray,
) -> npt.NDArray:
    """``q(u)`` of :func:`_censoring_residuals` at each time ``u`` of
    ``times``: over the events at ``t >= u``, the part of the risk set's
    covariate spread carried by the competing failures before ``u``. The
    rows (``Z``, ``risk``, ``n``, ``x``) are in time order, those of the
    risk sets ``sets``."""
    p = Z.shape[1]
    first = np.searchsorted(x, times, side="left")
    # The competing failures before each censoring time
    carried = sets.competing_over_G * n * risk
    before = np.concatenate([[0.0], _cumsum(carried)])[first]
    before_Z = np.concatenate(
        [np.zeros((1, p)), _cumsum((carried[:, None] * Z).T).T]
    )[first]
    # The events at or after each censoring time
    step = dH * sets.G_t
    later = np.searchsorted(sets.times, times, side="left")
    after = np.concatenate([_cumsum(step[::-1])[::-1], [0.0]])[later]
    after_mean = np.concatenate(
        [
            _cumsum((step[:, None] * mean).T[:, ::-1])[:, ::-1].T,
            np.zeros((1, p)),
        ]
    )[later]
    return before_Z * after[:, None] - before[:, None] * after_mean


def _runoff_of_cause(
    res: Any,
    maximum: str,
    converged: bool,
    info_at_start: npt.NDArray,
    derivatives: Any,
    newton_derivatives: Callable,
    Z_sorted: npt.NDArray,
    n_events: float,
    floor: npt.NDArray,
    data: tuple,
) -> tuple[npt.NDArray, str, npt.NDArray | None]:
    """The positions of the coefficients that run off, the fit's
    ``maximum`` and, where they run off together and none alone, their
    proportion: the exact test CoxPH makes (#728, #746), on the
    subdistribution risk sets, in place of the parametric judge's verdict
    on each coefficient (``maximum``, of the answer ``res``).

    The weighted partial likelihood is a Breslow likelihood whose risk set
    at an event time is every row still under observation and every row
    that failed from a competing cause before, the latter weighted by
    ``G(t-) / G(x_i-)``, positive (see :func:`_fit_cause`). The weights
    change the size of each term but not where it rises without bound:
    it is concave, and along a direction ``d`` its term at a time rises
    to a limit exactly when the events there share the largest ``d'z`` of
    the rows in its risk set. So it has no finite maximum exactly when the
    Cox likelihood of the same rows, with each competing failure at risk
    to the end (censored at infinity), has none, which
    :func:`runoff_direction` decides from the data.

    It is asked where the search gives cause, as CoxPH asks it: the
    answer is not verified, or not one Newton-Raphson converged to
    (``converged``), or a linear predictor (of ``Z_sorted``, centred) is
    beyond ``FAR`` of the average unit's, or the information (in
    ``derivatives``, else at ``res.x`` by ``newton_derivatives``) has
    collapsed against ``info_at_start``. A run-off the judge saw that the
    data do not have is a search that stopped short: the answer is then
    verified or not afresh, with the derivatives, ``n_events`` and
    ``floor`` the judge uses. ``data`` are the rows ``(x, Z, is_event,
    is_competing, n, strata)`` (``strata`` an integer code per row, or
    ``None``: the risk sets do not cross strata)."""
    with np.errstate(over="ignore", invalid="ignore"):
        eta = Z_sorted @ np.atleast_1d(res.x)
        info = (
            newton_derivatives(res.x)[1]
            if derivatives is None
            else derivatives[0]
        )
    suspect = (
        maximum != "verified"
        or not converged
        or not np.all(np.abs(eta) <= FAR)
        or information_collapsed(info, info_at_start)
    )
    none = np.zeros(0, dtype=int)
    if not suspect:
        return none, maximum, None
    x, Z, is_event, is_competing, n, strata = data
    found = runoff_direction(
        np.where(is_competing, np.inf, x),
        Z,
        np.where(is_event, 0, 1),
        n,
        np.full(len(x), -np.inf),
        strata,
        "breslow",
    )
    if found is not None:
        direction, alone = found
        off = np.flatnonzero(direction)
        return off, "no finite maximum", None if alone else direction[off]
    if maximum != "no finite maximum":
        return none, maximum, None
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score, info = newton_derivatives(res.x)
    verified = is_local_minimum(
        lambda _: 0.0,  # (only the derivatives are read)
        lambda _: score,
        lambda _: info,
        np.atleast_1d(res.x),
        floor=floor,
        obj_scale=max(n_events, 1.0),
    )
    return none, "verified" if verified else "unverified", None


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
    runaway = [
        (fit["cause"], fit["runaway"], fit.get("runaway_proportion"))
        for fit in fits
        if fit["runaway"]
    ]
    if len(fits) == 1 and runaway:
        warn_monotone(
            str(runaway[0][1]), runaway[0][2], advice=PENALISED_ADVICE
        )
    elif runaway:
        warn_monotone(
            " and ".join(
                "{} (cause {!r}{})".format(
                    coefs,
                    cause,
                    (
                        ""
                        if proportion is None
                        else ", together in the proportion "
                        + " : ".join(f"{v:.3g}" for v in proportion)
                    ),
                )
                for cause, coefs, proportion in runaway
            ),
            advice=PENALISED_ADVICE,
        )
    unverified = [f["cause"] for f in fits if f["maximum"] == "unverified"]
    if unverified:
        warn_unverified(
            "The Fine-Gray fit"
            + ("" if len(fits) == 1 else f" of cause(s) {unverified}")
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
    moved = bool(np.all(np.isfinite(out)) and np.all(out[cumhaz > 0] >= tiny))
    if not (moved and np.all(np.abs(lp) < LOG_MAX)):
        # (the rows' lp, where only they overflow)
        raise baseline_at_origin_error(
            "baseline cumulative subdistribution hazard",
            center,
            shift,
            -shift,
            lp=lp if moved else None,
        )
    return out


def _validated(
    x: Any,
    Z: Any,
    e: Any,
    c: Any,
    n: Any,
    w: Any,
    tl: Any = None,
    labels: tuple = (),
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray | None,
    npt.NDArray | None,
]:
    """``validate_fine_gray_inputs``, with the weights ``w`` of the rows
    kept as a sixth entry and their entry times ``tl`` as a seventh
    (``None`` where not given), and the warning of a small effective
    sample size, for the competing-risks regressions. ``labels`` are
    per-row arrays (stratum or cluster codes) dropped with the rows, and
    returned after the rest."""
    out = validate_fine_gray_inputs(x, Z, e, c, n, w, tl, labels=labels)
    if labels:
        kept_labels = out[-len(labels) :]
        out = out[: -len(labels)]
    else:
        kept_labels = ()
    w_kept = None if w is None else out[5]
    tl_kept = None if tl is None else out[-1]
    if w_kept is not None:
        warn_if_few_effective(out[4], w_kept)
    return (*out[:5], w_kept, tl_kept, *kept_labels)


#: What a subdistribution method refusing delayed entry offers instead.
ENTRY_INSTEAD = (
    "For the covariates' effects with delayed entry, fit the "
    "cause-specific Cox model, CompetingRisksProportionalHazards.fit(x, Z, "
    "e, tl=tl) (model='Cox'); for each cause's cumulative incidence, the "
    "Aalen-Johansen estimate CompetingRisks.fit(x, e, tl=tl)."
)


def _same_label(a: Any, b: Any) -> bool:
    """Whether two stratum labels are the same (``1`` and ``1.0`` are;
    ``"1"`` and ``1`` are not)."""
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _label_codes(
    labels: Any, name: str, rows: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """The integer code of each row's label in ``labels`` (``strata`` or
    ``cluster``), -1 where it is missing (``None``, ``NaN``, pandas
    ``NA``), and the distinct labels in code order."""
    arr, missing = _labels_and_missing(labels)
    if arr.ndim != 1 or arr.shape[0] != rows:
        raise ValueError(
            "'{}' must have one label for each row of x: got {} for {} "
            "rows.".format(name, np.size(arr), rows)
        )
    present = arr[~missing]
    if not present.size:
        raise ValueError(
            f"Every '{name}' label is missing; give each row a label."
        )
    levels, codes = np.unique(present, return_inverse=True)
    out = np.full(rows, -1, dtype=int)
    out[~missing] = codes
    return out, levels


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
    CompetingRisksPartialEffectsMixin,
    InformationCriteriaMixin,
    LinearPredictorMixin,
    DiagnoseMixin,
    SerialisableMixin,
):
    """
    A fitted Fine-Gray subdistribution-hazard model for one cause of interest.

    The natural prediction is the cumulative incidence function :meth:`cif`;
    ``coefficients``, :meth:`standard_errors`, ``p_values`` and
    :meth:`summary` describe the (log) subdistribution hazard ratios. The
    standard errors, and the p-values and intervals from them, are the
    robust ones of Fine and Gray (the sandwich ``cmprsk::crr`` reports);
    ``covariance(robust=False)``, ``standard_errors(robust=False)`` and
    ``summary(robust=False)`` give the model-based ones (see
    :meth:`covariance`).

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
    #: The coefficients' two-sided Wald p-values, on the standard errors
    #: of ``standard_errors()`` (the robust ones); ``nan`` on a model
    #: restored from a dict saved before v0.27, which has only the
    #: model-based ones (``summary(robust=False)``).
    p_values: npt.NDArray
    #: The coefficients' covariance, ``covariance()`` (#605): the robust
    #: one; ``None`` on a model restored from a dict saved before v0.27,
    #: which stored only the model-based one.
    _covariance: npt.NDArray | None
    #: The model-based covariance, ``covariance(robust=False)``: the
    #: inverse information. ``None`` for a fit with weights ``w``.
    _model_covariance: npt.NDArray | None
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
    _objective: Callable | None
    #: The covariates' column names (a DataFrame ``Z`` or
    #: ``fit_from_df``), which name the coefficients (#614); ``None``
    #: for an array ``Z``, whose coefficients are ``coef_0``, ...
    feature_names: list[str] | None = None
    #: The rows the model was fitted to (``x``, ``Z`` as given, ``c``,
    #: ``n`` and the masks ``is_event`` and ``is_competing``), on which
    #: ``diagnose()`` checks it (#874); ``None`` on a model restored with
    #: ``from_dict``, which does not store them.
    _fit_data: dict | None = None
    #: Whether the model was fitted with weights ``w``: it then has no
    #: model-based covariance, and the information criteria refuse.
    weighted: bool = False
    #: The effective sample size of the weights (``None`` unweighted).
    effective_n: float | None = None
    #: The strata of a stratified fit (``strata=``), in order; ``None``
    #: for a model without strata.
    strata_labels: list | None = None
    #: Each stratum's baseline cumulative subdistribution hazard, its step
    #: times and values, by stratum label; ``None`` without strata.
    strata_baselines: dict | None = None
    #: The number of clusters of a fit with ``cluster=`` (its robust
    #: covariance sums each cluster's influence); ``None`` without.
    n_clusters: int | None = None
    #: How a stratified fit estimated the censoring distribution:
    #: ``"within"`` each stratum or ``"common"`` to all; ``None``
    #: without strata.
    strata_censoring: str | None = None

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
        self.p_values = fit["p_values"]
        self._covariance = fit["cov"]
        self._model_covariance = fit.get("model_cov")
        self._times = fit["baseline_times"]
        self._cumhaz = fit["baseline_cumhaz"]
        self._neg_ll = fit["neg_ll"]
        self._ic_n = fit.get("ic_n")
        self.res = fit["res"]
        self.maximum = fit.get("maximum", "unknown")
        self._objective = fit.get("objective")
        self._fit_data = fit.get("fit_data")
        self.weighted = bool(fit.get("weighted", False))
        self.effective_n = fit.get("effective_n")
        labels = fit.get("strata_labels")
        if labels is not None:
            self.strata_labels = list(labels)
            self.strata_baselines = {
                label: (
                    np.asarray(times, dtype=float),
                    np.asarray(cumhaz, dtype=float),
                )
                for label, (times, cumhaz) in zip(
                    labels, fit["strata_baselines"]
                )
            }
        self.strata_censoring = fit.get("strata_censoring")
        clusters = fit.get("n_clusters")
        self.n_clusters = None if clusters is None else int(clusters)

    @property
    def is_stratified(self) -> bool:
        """Whether the model was fitted with ``strata``: each stratum then
        has its own baseline, and :meth:`cif` takes ``stratum``."""
        return self.strata_labels is not None

    def _baseline(self, stratum: Any) -> tuple[npt.NDArray, npt.NDArray]:
        """The baseline cumulative subdistribution hazard's step times and
        values: the stratum's for a stratified model (``stratum``
        required), the model's otherwise (``stratum`` refused)."""
        if not self.is_stratified:
            if stratum is not None:
                raise ValueError(
                    "'stratum' was given but this Fine-Gray model is not "
                    "stratified; leave stratum out."
                )
            return self._times, self._cumhaz
        assert self.strata_baselines is not None
        if stratum is None:
            raise ValueError(
                "This Fine-Gray model is stratified, with a baseline for "
                "each stratum; pass stratum=... to choose one (one of "
                "{}).".format(self.strata_labels)
            )
        for label, baseline in self.strata_baselines.items():
            if _same_label(label, stratum):
                return baseline
        raise ValueError(
            "Unknown stratum {!r}; the strata are {}.".format(
                stratum, self.strata_labels
            )
        )

    _ALIASED_WHY = (
        "a constant column, which the baseline subdistribution hazard "
        "absorbs, or a linear combination of the others"
    )

    def covariance(self, robust: bool = ROBUST_BY_DEFAULT) -> npt.NDArray:
        """The coefficients' covariance (a ``nan`` row and column for an
        aliased coefficient). It was ``cov`` before v0.23.

        By default it is the robust (sandwich) covariance of Fine and
        Gray (1999), which R's ``cmprsk::crr`` reports as ``var``:
        ``A^-1 M A^-1``, ``A`` the information of the weighted partial
        likelihood and ``M`` the sum over the units of the squares of
        their influence on the score, the part through the estimated
        censoring distribution ``G`` included. ``robust=False`` gives
        the model-based covariance ``A^-1`` (``crr``'s ``invinf``), which
        takes the censoring weights as known; it was the only covariance
        before v0.27.

        For a fit with sampling weights ``w`` the robust covariance is
        the same sandwich with each unit's influence (the part through
        ``G``, the Kaplan-Meier of the censoring weighted by ``w``,
        included) multiplied by its weight, so ``w = 1`` gives the
        unweighted covariance; ``robust=False`` is refused. A model
        restored from a dict saved before v0.27 has only the model-based
        covariance: ask for it with ``robust=False``, or refit.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import FineGray
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
        >>> t_b = rng.exponential(1 / 0.05, 200)
        >>> t_c = rng.uniform(0, 20, 200)  # censoring times
        >>> x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
        >>> first = np.where(t_a < t_b, "a", "b")
        >>> e = np.where(t_c < np.minimum(t_a, t_b), None, first)
        >>> model = FineGray.fit(x, Z, e, event="a")
        >>> model.covariance().round(5)
        array([[0.04333]])
        >>> model.covariance(robust=False).round(5)
        array([[0.04419]])
        """
        if not robust:
            return self._model_based_covariance()
        if self._covariance is None:
            raise no_covariance_error(
                "the robust one: it was saved before v0.27, which stored "
                "only the model-based covariance; ask for that with "
                "robust=False, or refit the model"
            )
        return self._covariance

    def _model_based_covariance(self) -> npt.NDArray:
        if self._model_covariance is None:
            raise refuse_weighted(
                "The model-based covariance (robust=False)",
                "the inverse information of a weighted partial likelihood "
                "is not the covariance of its estimate",
                "Use the robust covariance, the default.",
            )
        return self._model_covariance

    def standard_errors(self, robust: bool = ROBUST_BY_DEFAULT) -> npt.NDArray:
        """The coefficients' standard errors, the square roots of the
        diagonal of :meth:`covariance`: the robust ones by default, the
        model-based ones with ``robust=False``. They were the attribute
        ``se`` before v0.23."""
        return standard_errors_of(self.covariance(robust))

    @property
    def params(self) -> npt.NDArray:
        """The coefficients (``beta``), named by ``parameter_names``, as
        on the other regression models."""
        return np.asarray(self.beta, dtype=float)

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: each covariate's
        column (a DataFrame ``Z`` or ``fit_from_df``), else ``coef_0``,
        ``coef_1``, ... for an array ``Z`` (#614), as for ``CoxPH``."""
        return coefficient_names(self, np.size(self.beta))

    def summary(
        self, alpha_ci: float = 0.05, robust: bool = ROBUST_BY_DEFAULT
    ) -> pd.DataFrame:
        """
        The coefficient table, as ``CoxPH.summary``: one row per covariate
        (named by ``parameter_names``), with the coefficient, the
        subdistribution hazard ratio ``exp(coef)``, the standard error, a
        two-sided ``1 - alpha_ci`` Wald interval for both, the Wald
        statistic ``z`` and its two-sided p-value. An aliased coefficient
        is ``nan`` throughout.

        The standard errors, and so the intervals, ``z`` and ``p``, are
        the robust ones of :meth:`covariance` by default (Fine and Gray's
        sandwich, as ``cmprsk::crr`` prints); ``robust=False`` builds the
        whole table from the model-based ones instead.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import FineGray
        >>> rng = np.random.default_rng(0)
        >>> Z = pd.DataFrame({"grp": rng.binomial(1, 0.5, 200)})
        >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z["grp"])))
        >>> t_b = rng.exponential(1 / 0.05, 200)
        >>> x = np.minimum(t_a, t_b).round(3)
        >>> e = np.where(t_a < t_b, "a", "b")
        >>> model = FineGray.fit(x, Z, e, event="a")
        >>> model.summary()[["coef", "exp(coef)", "se(coef)"]].round(4)
                     coef  exp(coef)  se(coef)
        covariate
        grp        0.6626     1.9398    0.1727
        """
        check_alpha_ci(alpha_ci)
        return coefficient_table(
            self.parameter_names,
            self.params,
            self.standard_errors(robust),
            alpha_ci,
        )

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
        subdistribution baseline cumulative-hazard step arrays (each
        stratum's, under ``"strata"``, for a stratified fit; and, for a
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
            "p_values": np.asarray(self.p_values, dtype=float).tolist(),
            # The key every model's dict stores it under (#605).
            "_neg_ll": float(self._neg_ll),
            **maximum_entry(self.maximum),
        }
        if self.strata_baselines is None:
            out["baseline_times"] = np.asarray(
                self._times, dtype=float
            ).tolist()
            out["baseline_cumhaz"] = np.asarray(
                self._cumhaz, dtype=float
            ).tolist()
        else:
            # Each stratum's baseline, and no top-level one: a reader that
            # does not know the strata cannot take one stratum's baseline
            # for the model's.
            out["strata"] = [
                {
                    "stratum": to_native(label),
                    "baseline_times": np.asarray(t, dtype=float).tolist(),
                    "baseline_cumhaz": np.asarray(h, dtype=float).tolist(),
                }
                for label, (t, h) in self.strata_baselines.items()
            ]
            out["strata_censoring"] = self.strata_censoring
        if self.n_clusters is not None:
            out["n_clusters"] = int(self.n_clusters)
        if self._covariance is not None:
            # The key every model's dict stores covariance() under (#605),
            # with its standard errors
            out["covariance"] = np.asarray(
                self._covariance, dtype=float
            ).tolist()
            out["se"] = standard_errors_of(self._covariance).tolist()
        if self._model_covariance is not None:
            out["model_covariance"] = np.asarray(
                self._model_covariance, dtype=float
            ).tolist()
        if self._ic_n is not None:
            out["ic_n"] = float(self._ic_n)
        if np.any(self.center):
            out["center"] = np.asarray(self.center, dtype=float).tolist()
        if self.feature_names is not None:
            out["feature_names"] = list(self.feature_names)
        if self.weighted:
            # Schema 4: a reader without weights would offer the
            # information criteria.
            out["weighted"] = True
            assert self.effective_n is not None
            out["effective_n"] = float(self.effective_n)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> FineGrayModel:
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
        weighted = model_dict.get("weighted") is True
        # "cov" is the key of a dict written before v0.23.
        stored = model_dict.get("covariance", model_dict.get("cov"))
        cov = None if stored is None else np.array(stored, dtype=float)
        model_cov = model_dict.get("model_covariance")
        p_values = np.array(model_dict["p_values"], dtype=float)
        if model_cov is not None:
            model_cov = np.array(model_cov, dtype=float)
        elif not weighted:
            # Saved before v0.27: its one covariance, and its p-values,
            # are the model-based ones, and the robust one cannot be
            # rebuilt without the data.
            cov, model_cov = None, cov
            p_values = np.full(beta.shape, np.nan)
        strata = model_dict.get("strata")
        if strata is not None:
            first = strata[0] if strata else {}
            times = first.get("baseline_times", [])
            cumhaz = first.get("baseline_cumhaz", [])
        else:
            times = model_dict["baseline_times"]
            cumhaz = model_dict["baseline_cumhaz"]
        out = cls(
            {
                "cause": label_from_native(model_dict["cause"]),
                "beta": beta,
                "center": center,
                "p_values": p_values,
                "cov": cov,
                "model_cov": model_cov,
                "baseline_times": np.array(times, dtype=float),
                "baseline_cumhaz": np.array(cumhaz, dtype=float),
                # "neg_ll" is the key of a dict written before v0.23.
                "neg_ll": model_dict.get(
                    "_neg_ll", model_dict.get("neg_ll", np.nan)
                ),
                "ic_n": cls._restored_ic_n(model_dict),
                "res": None,
                "maximum": restored_maximum(model_dict),
                "weighted": weighted,
                "effective_n": model_dict.get("effective_n"),
                "n_clusters": model_dict.get("n_clusters"),
                **(
                    {}
                    if strata is None
                    else {
                        "strata_labels": [
                            label_from_native(b["stratum"]) for b in strata
                        ],
                        "strata_baselines": [
                            (b["baseline_times"], b["baseline_cumhaz"])
                            for b in strata
                        ],
                        "strata_censoring": model_dict.get(
                            "strata_censoring", "within"
                        ),
                    }
                ),
            }
        )
        names = model_dict.get("feature_names")
        out.feature_names = None if names is None else list(names)
        return out

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
                (
                    np.asarray(
                        named_covariates(Z, self.feature_names), dtype=float
                    )
                    - self.center
                )
                @ self._coef()
            )

    @keeps_query_shape
    def cif(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, stratum: Any = None
    ) -> npt.NDArray:
        """
        Cumulative incidence of the cause of interest at times ``x``:
        ``1 - exp(-Lambda0(x) * exp(beta'(Z - center)))``, the baseline
        ``Lambda0`` that of a unit at ``center``. ``Z`` is one covariate
        vector (a 1-D array or a single row), used at every time, or one
        row per time in ``x`` (row ``i`` with ``x[i]``). The CIF is flat
        before the first event time and after the last (the baseline is a
        step function estimated only on the observed range). A missing
        (``NaN``) time or covariate gives ``nan`` in its place.

        A stratified model (fitted with ``strata``) has a baseline for
        each stratum: ``stratum`` names the one to use, and is required;
        a model without strata refuses it.
        """
        times, cumhaz = self._baseline(stratum)
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        Z = named_covariates(Z, self.feature_names)
        rows = paired_covariate_rows(Z, x.size, np.size(self.beta))
        # (a stratum without events of the cause has no steps: 0)
        H0 = (
            step_at(times, cumhaz, x, before=0.0)
            if np.size(times)
            else np.zeros(x.size)
        )
        # step_at reads a nan time as the value after the last jump.
        H0 = np.where(np.isnan(x), np.nan, H0)
        # H0 * exp(beta'(Z - center)) on the log scale: a baseline at Z = 0
        # far from the data is tiny and the multiplier huge (#463).
        with np.errstate(divide="ignore", over="ignore"):
            H = np.exp(np.log(H0) + (rows - self.center) @ self._coef())
        return -np.expm1(-H)

    # -- plot_partial_effects (CompetingRisksPartialEffectsMixin) ----------

    def _partial_effects_width(self) -> int:
        return int(np.size(self.beta))

    def _partial_effects_times(self) -> npt.NDArray | None:
        return np.asarray(self._times, dtype=float)

    def _partial_effects_curve(
        self, on: str, x: npt.NDArray, row: npt.NDArray, event: Any
    ) -> npt.NDArray:
        # One cause's model: event, where given, must be that cause.
        if event is not None and event != self.cause:
            raise unknown_cause_error(event, [self.cause])
        if self.is_stratified:
            raise ValueError(
                "plot_partial_effects is not available for a stratified "
                "Fine-Gray model, whose curves differ by stratum: plot "
                "model.cif(x, Z, stratum=...) for the stratum wanted."
            )
        return np.asarray(self.cif(x, row), dtype=float)

    @keeps_query_shape
    def sf(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, stratum: Any = None
    ) -> npt.NDArray:
        """One minus the cumulative incidence (the cause-of-interest-free
        probability under the subdistribution); ``stratum`` as for
        :meth:`cif`."""
        return 1.0 - self.cif(x, Z, stratum=stratum)

    def __repr__(self) -> str:
        lines = [
            "Fine-Gray Subdistribution Hazard Model",
            "======================================",
            f"Cause of interest   : {self.cause}",
            "Coefficients (beta'Z acts on the subdistribution hazard):",
        ]
        names = coefficient_names(self, np.size(self.beta))
        robust = self._covariance is not None
        table = self.summary(robust=robust)
        for nm, b, s, p in zip(
            names, self.beta, table["se(coef)"], table["p"]
        ):
            lines.append(f"   {nm}  :  {b: .6f}  (se {s:.6f}, p {p:.4f})")
        if not self.weighted:
            lines.append(
                "Standard errors     : "
                + (
                    "robust (Fine and Gray's sandwich)"
                    if robust
                    else "model-based (saved before v0.27; refit for the "
                    "robust ones)"
                )
            )
        if self.weighted:
            lines.append(
                "Weights             : effective sample size "
                f"{self.effective_n:.4g}; robust covariance"
            )
        if self.is_stratified:
            assert self.strata_labels is not None
            labels = self.strata_labels
            shown = (
                str(labels)
                if len(labels) <= 8
                else "{} strata ({}, ..., {})".format(
                    len(labels), labels[0], labels[-1]
                )
            )
            lines.append(
                f"Strata              : {shown}, a baseline for each; "
                + (
                    "a censoring distribution within each"
                    if self.strata_censoring != "common"
                    else "one censoring distribution for all"
                )
            )
        if self.n_clusters is not None:
            lines.append(
                f"Clusters            : {self.n_clusters} (cluster-robust "
                "standard errors)"
            )
        return "\n".join(lines)


class FineGray_(FitterRepr):
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

    #: The ``repr`` (#614)
    fitter_kind = "semi-parametric subdistribution hazards fitter"

    @column_arguments("x", "c", "n")
    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        e_col: str,
        Z_cols: str | list[str],
        c_col: str | None = None,
        n_col: str | None = None,
        w_col: str | None = None,
        tl_col: str | None = None,
        strata_col: str | None = None,
        cluster_col: str | None = None,
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
        w_col : str, optional
            Column of weights per row (``w`` of :meth:`fit`).
        tl_col : str, optional
            Column of entry times (``tl`` of :meth:`fit`), refused, as
            there, where an entry is after the first time, with what to
            use instead.
        strata_col : str, optional
            Column of stratum labels (``strata`` of :meth:`fit`).
        cluster_col : str, optional
            Column of cluster labels (``cluster`` of :meth:`fit`).
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
        arrays |= read_columns(
            df, {"c": c_col, "n": n_col, "w": w_col, "tl": tl_col}
        )
        for key, col, arg in (
            ("strata", strata_col, "strata_col"),
            ("cluster", cluster_col, "cluster_col"),
        ):
            if col is not None:
                arrays[key] = frame_column(df, col, arg)
        names = {"x": "x_col", "Z": "Z_cols", "e": "e_col"}
        names |= {"c": "c_col", "n": "n_col", "w": "w_col", "tl": "tl_col"}
        names |= {"strata": "strata_col", "cluster": "cluster_col"}
        # The columns name the coefficients (#614)
        columns = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        with covariate_columns(columns):
            model = call_fit(self, arrays, names, fit_options)
        model.feature_names = [str(column) for column in columns]
        return model

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        event: Any = None,
        center: bool = False,
        w: npt.ArrayLike | float | None = None,
        tl: npt.ArrayLike | float | None = None,
        strata: npt.ArrayLike | None = None,
        cluster: npt.ArrayLike | None = None,
        strata_censoring: str = "within",
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
        w : array_like or float, optional
            Sampling, propensity or censoring weights, one positive number
            per row (beside its count ``n``), or one for every row. Each
            row then counts ``n * w`` in the events, the subdistribution
            risk sets and the censoring distribution ``G`` (the weighted
            Kaplan-Meier of the censoring times), so whole-number weights
            give the estimates of the rows counted that many times. The
            covariance is Fine and Gray's robust (sandwich) one with the
            weights, the part through ``G`` included, so ``w = 1`` gives
            the unweighted fit's; ``aic``, ``bic`` and
            ``aic_c`` and the diagnostics refuse a weighted fit. See
            :doc:`/Weighted Data`.
        tl : array_like or float, optional
            Delayed entry (left truncation) is refused with a
            ``ValueError`` that names what to use instead (the
            cause-specific Cox model, or the Aalen-Johansen estimate):
            the subdistribution risk set keeps the items that failed from
            another cause, and with late entry it needs truncation weights
            (Geskus 2011), which are not built. Entry times all before the
            first time change nothing and are taken.
        strata : array_like, optional
            A stratum label for each row, as for ``CoxPH``: the model is
            then *stratified* (Zhou, Latouche, Rocha and Fine, 2011, R's
            ``crrSC::crrs`` with ``ctype = 1``). Each stratum has its own
            baseline subdistribution hazard, its own risk sets and its own
            censoring distribution ``G`` (the Kaplan-Meier of the
            censoring within the stratum), while the coefficients are
            shared. Use it for a factor whose effect on the incidence is
            not proportional, or whose censoring differs (a centre, a
            site, a production line). Predictions (:meth:`~FineGrayModel.cif`)
            then take ``stratum``. Rows with a missing label are dropped,
            with a warning.
        strata_censoring : ('within', 'common'), str, optional
            How a stratified fit estimates the censoring distribution:
            within each stratum (``'within'``, the default, ``crrs``'s
            ``ctype = 1``), for a few large strata whose censoring may
            differ; or one for all the strata (``'common'``, ``ctype =
            2``), for many small strata (a stratum per centre, say), where
            censoring does not depend on the stratum. Either way each
            stratum keeps its own baseline. Taken only with ``strata``.
        cluster : array_like, optional
            A cluster label for each row, for units that are not
            independent of each other within a cluster (the two eyes or
            kidneys of a patient, the machines of one site, the
            components of one system). The coefficients are those of the
            fit without clusters (an independence working model), and the
            robust covariance sums the influence of each cluster's units
            before squaring it, so the standard errors allow for the
            dependence (Zhou, Fine, Latouche and Labopin, 2012, R's
            ``crrSC::crrc``). The model-based covariance
            (``robust=False``) ignores the clusters. Rows with a missing
            label are dropped, with a warning. It combines with
            ``strata`` and ``w``.

        Returns
        -------
        FineGrayModel
            The fitted model, with :meth:`~FineGrayModel.cif` prediction.
            Its standard errors, p-values and ``summary()`` use the robust
            (sandwich) variance of Fine and Gray, as R's ``cmprsk::crr``;
            the model-based ones are ``standard_errors(robust=False)``
            and ``summary(robust=False)`` (see
            :meth:`~FineGrayModel.covariance`).
            Where a covariate, or a combination of them, separates the
            events of interest from the rest of their risk sets, the
            weighted partial likelihood has no finite maximum: the fit
            warns ("No finite maximum", naming the coefficients and, for
            a combination, their proportion), ``maximum`` is ``"no finite
            maximum"`` and those coefficients are meaningless (their
            standard errors nan). This is decided from the data, as for
            ``CoxPH``.

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
        refuse_subdistribution_entry(x, tl, "FineGray", ENTRY_INSTEAD)
        check_option(
            "strata_censoring", strata_censoring, ("within", "common")
        )
        if strata is None and strata_censoring != "within":
            raise ValueError(
                "strata_censoring='common' is for a stratified fit: give "
                "strata=..., or leave strata_censoring out (a fit without "
                "strata has one censoring distribution already)."
            )
        rows = int(np.size(np.atleast_1d(np.asarray(x, dtype=object)), 0))
        coded = {}
        for name, labels in (("strata", strata), ("cluster", cluster)):
            if labels is not None:
                coded[name] = _label_codes(labels, name, rows)
        validated = _validated(
            x, Z, e, c, n, w, labels=tuple(v[0] for v in coded.values())
        )
        x, Z, e, c, n, w_kept = validated[:6]
        codes: list = list(validated[7:])
        found = dict(zip(coded, codes))
        missing = np.zeros(x.size, dtype=bool)
        for name, values in found.items():
            missing |= values < 0
        if missing.any():
            warnings.warn(
                "Dropped {} of {} rows with a missing {} label.".format(
                    int(missing.sum()),
                    missing.size,
                    " or ".join(k for k, v in found.items() if (v < 0).any()),
                ),
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
            keep = ~missing
            x, Z, e, c, n = x[keep], Z[keep], e[keep], c[keep], n[keep]
            w_kept = None if w_kept is None else w_kept[keep]
            found = {k: v[keep] for k, v in found.items()}
        # Recode: a level whose rows were all dropped has no code
        strata_levels = None
        for name in found:
            used, recoded = np.unique(found[name], return_inverse=True)
            if name == "strata":
                strata_levels = [
                    to_native(level) for level in coded[name][1][used]
                ]
            found[name] = recoded.astype(int)

        causes = ordered_labels(e)
        if event is None:
            if len(causes) != 1:
                raise missing_cause_error(
                    f"A Fine-Gray model (the data have causes {causes})"
                )
            event = causes[0]
        elif event not in causes:
            raise unknown_cause_error(event, causes)

        fit = _fit_cause(
            x,
            Z,
            e,
            c,
            n,
            event,
            center,
            w_kept,
            strata=found.get("strata"),
            cluster=found.get("cluster"),
            common_censoring=strata_censoring == "common",
        )
        if strata_levels is not None:
            fit["strata_labels"] = strata_levels
            fit["strata_censoring"] = strata_censoring
        maximum = _warn_if_monotone([fit])
        model = FineGrayModel(fit)
        model.maximum = maximum
        return model


FineGray = FineGray_()
