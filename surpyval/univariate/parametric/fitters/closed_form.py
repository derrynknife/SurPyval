"""Closed-form maximum likelihood estimation.

A few distributions have an exact analytic MLE for particular data
shapes -- the Exponential's events-over-exposure ratio, the Normal's
mean and standard deviation. Where one applies it is not merely faster
than the optimiser ladder, it is exact, so there is no reason to seed an
initial guess and iterate.

A distribution opts in by defining ``_closed_form_mle(data)``, which
returns either the parameter vector or ``None`` to mean "not solvable in
closed form for *this* data; use the optimiser". Returning ``None``
rather than raising keeps the applicability condition next to the
computation that relies on it, so the two cannot drift apart, and leaves
exceptions to mean what they have always meant: the fit is impossible,
not merely non-analytic.

The result is completed here rather than by each distribution, so every
closed-form fit carries the same information as an optimised one --
``_neg_ll`` and a parameter covariance. Without them ``aic``, ``bic``,
``aic_c`` and ``cb`` break on the fitted model.
"""

import numpy as onp
from autograd import hessian
from numdifftools import Hessian  # type: ignore
from numpy.linalg import LinAlgError, inv, pinv

from surpyval import np


def _neg_ll_at(dist, data, params):
    """The model's negative log-likelihood at ``params`` (no offset,
    zero-inflation or limited-failure component -- the closed-form gate
    excludes all three)."""
    return dist._neg_ll_func(data, *params, 0.0, 0.0, 1.0)


def parameter_covariance(dist, data, params):
    """Asymptotic parameter covariance, the inverse observed information.

    Computed directly in the natural parameter space: the closed-form
    gate rules out fixed, offset, limited-failure and zero-inflation
    parameters, so there is no transform to undo and no delta step --
    unlike the optimiser path, which must work in the unbounded
    transformed space it searched.
    """
    params = onp.asarray(params, dtype=float)

    def fun(p):
        return _neg_ll_at(dist, data, p)

    with onp.errstate(all="ignore"):
        information = onp.atleast_2d(hessian(fun)(params))
        # autograd propagates nan through infinite truncation bounds even
        # where they cannot affect the derivative (the caveat noted at the
        # top of ``mle.py``), and a corrupted primitive shows up as
        # asymmetry (#270). Either way, recompute numerically rather than
        # invert nonsense.
        asymmetric = onp.max(
            onp.abs(information - information.T)
        ) > 1e-4 * max(onp.max(onp.abs(information)), 1.0)
        if onp.isnan(information).any() or asymmetric:
            information = onp.atleast_2d(Hessian(fun)(params))
        if onp.isnan(information).any():
            return None
        try:
            cov = inv(information)
        except LinAlgError:
            cov = pinv(information)
        if onp.isnan(cov).any():
            try:
                cov = pinv(information)
            except LinAlgError:
                return None
    return cov


def closed_form_results(dist, data, params):
    """Complete a closed-form parameter vector into a full results dict.

    Mirrors what the optimiser path returns, so a closed-form fit
    supports the same model methods.
    """
    params = onp.atleast_1d(onp.asarray(params, dtype=float))
    with onp.errstate(all="ignore"):
        neg_ll = float(_neg_ll_at(dist, data, params))

    try:
        cov = parameter_covariance(dist, data, params)
    except (LinAlgError, ValueError, FloatingPointError):
        cov = None

    # Not every MLE is regular. The Uniform's is an order statistic --
    # the estimate sits *on* the support edge rather than at an interior
    # stationary point -- so the observed information is not positive
    # definite and its inverse carries negative variances. Reporting that
    # would turn into silent nan confidence bounds downstream, so no
    # covariance is offered at all and ``cb`` refuses with its usual
    # message.
    if cov is not None:
        diagonal = onp.diag(onp.atleast_2d(cov))
        if not onp.isfinite(diagonal).all() or (diagonal <= 0).any():
            cov = None

    return {
        "params": params,
        "gamma": 0.0,
        "f0": 0.0,
        "p": 1.0,
        "_neg_ll": neg_ll,
        "log_likelihood": -neg_ll,
        "cov_matrix": cov,
        "hess_inv": cov,
        "res": None,
        "optimizer": "closed-form",
    }


def entry_times(data):
    """Left-truncation times with non-finite entries replaced by 0.

    ``-inf`` marks "not truncated"; for a lifetime distribution supported
    on ``[0, inf)`` that is entry at time zero, which is what the
    exposure calculation needs (``x - tl`` would otherwise be infinite).
    """
    tl = onp.asarray(data.t[:, 0], dtype=float)
    return onp.where(onp.isfinite(tl), tl, 0.0)


def is_uncensored_and_untruncated(data):
    """Every observation exact, with no truncation of either kind."""
    return bool(
        (onp.asarray(data.c) == 0).all()
        and onp.isneginf(onp.asarray(data.t[:, 0])).all()
        and onp.isposinf(onp.asarray(data.t[:, 1])).all()
    )


def weighted_mean_and_std(values, n):
    """MLE mean and standard deviation (dividing by the total weight, not
    by ``total - 1``), or ``None`` if the spread is degenerate."""
    values = onp.asarray(values, dtype=float)
    n = onp.asarray(n, dtype=float)
    total = n.sum()
    if not total > 0:
        return None
    mu = (n * values).sum() / total
    var = (n * (values - mu) ** 2).sum() / total
    if not var > 0:
        # sigma is bounded strictly positive; a degenerate sample has no
        # interior maximum, so leave it to the optimiser to report.
        return None
    return onp.array([mu, onp.sqrt(var)])


__all__ = [
    "closed_form_results",
    "entry_times",
    "is_uncensored_and_untruncated",
    "parameter_covariance",
    "weighted_mean_and_std",
    "np",
]
