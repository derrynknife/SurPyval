"""Shared likelihood machinery for parametric regression fitters.

All of SurPyval's parametric regression models (proportional hazards,
proportional odds, accelerated failure time and accelerated life /
parameter substitution) share the same observation model: each row may be
observed, right/left/interval censored, and additionally left, right or
interval truncated. The only thing that differs between the models is how
the covariates ``Z`` enter the distribution functions.

``regression_neg_ll`` therefore expresses the negative log-likelihood purely
in terms of a ``model`` object that exposes the covariate-aware functions
``log_df``, ``log_sf``, ``log_ff`` and ``ff`` (all with the signature
``(x, Z, *params)``). This keeps the censoring and truncation handling in a
single, well-tested place rather than duplicated — and subtly inconsistent —
across each fitter.
"""

from typing import Any

import autograd.numpy as np

from surpyval.univariate.parametric.parametric_fitter import Boxable
from surpyval.utils.surpyval_data import SurpyvalData

# Smallest positive float; used to floor probability masses so that
# ``log`` stays finite for impossible (zero-width) intervals.
_TINY = np.finfo(float).tiny


def truncation_correction(
    model: Any, data: SurpyvalData, *params: Boxable
) -> Boxable:
    """Log of the probability mass within each observation's truncation
    interval, summed over the truncated rows.

    The likelihood of a truncated observation is divided by
    ``P(tl < X < tr | Z)`` to account for the fact that it could only have
    been observed within that window. In log space this is a subtraction,
    so the caller subtracts the value returned here.

    A window bounded on one side only is evaluated in log space rather
    than as a difference of CDFs, because the difference underflows and
    the floor that catches it is worse than the underflow. Left
    truncation gives ``1 - F(tl)``, which goes to zero as the fitted
    scale shrinks; once it reaches zero, flooring at the smallest
    positive float caps the correction at ``log(tiny) = -708`` instead of
    the unbounded value it should take. Since the caller *subtracts*
    this, every truncated row then appears to contribute +708 to the
    log-likelihood, and a region the data rules out entirely starts to
    look like the best fit available. A WeibullPH fit to left-truncated
    data walked into exactly that, reporting a log-likelihood 38,000
    higher than its parameters actually earn (#326).

    ``log_sf`` and ``log_ff`` compute those two cases directly and stay
    finite where the difference cannot, so there is nothing to floor.
    Only a genuinely two-sided window still needs the difference, and
    there both bounds are finite and the mass is not driven to zero by
    the scale alone.
    """
    if data.x_tl.size == 0:
        return 0.0

    tl, tr, Z = data.x_tl, data.x_tr, data.Z_t
    lo_finite = np.isfinite(tl)
    hi_finite = np.isfinite(tr)

    # Each window shape is evaluated only when some row has it, and with a
    # stand-in taken from the rows that do: autograd evaluates every branch
    # of a ``np.where``, so an infinity (or, in a branch no row uses, a
    # bound at which the function is -inf -- ``log F(0)`` for the usual
    # left truncation at 0) reaching one of them would poison the gradient
    # of the selected branch, and warned "divide by zero" from every
    # ``covariance``/``cb`` call made outside the fit's error suppression.
    def stand_in(values: Any, rows: Any) -> Any:
        return np.where(rows, values, values[rows][0])

    log_mass: Boxable = np.zeros(tl.shape[0])
    both = lo_finite & hi_finite
    if both.any():
        tl_b, tr_b = stand_in(tl, both), stand_in(tr, both)
        window = np.maximum(
            model.ff(tr_b, Z, *params) - model.ff(tl_b, Z, *params), _TINY
        )
        log_mass = np.where(both, np.log(window), log_mass)
    left = lo_finite & ~hi_finite  # (tl, inf)
    if left.any():
        log_left = model.log_sf(stand_in(tl, left), Z, *params)
        log_mass = np.where(left, log_left, log_mass)
    right = ~lo_finite & hi_finite  # (-inf, tr)
    if right.any():
        log_right = model.log_ff(stand_in(tr, right), Z, *params)
        log_mass = np.where(right, log_right, log_mass)
    return (data.n_t * log_mass).sum()


def regression_neg_ll(
    model: Any, data: SurpyvalData, *params: Boxable
) -> Boxable:
    """Negative log-likelihood for a covariate-aware survival model.

    Parameters
    ----------
    model : object
        Must provide ``log_df``, ``log_sf``, ``log_ff`` and ``ff``, each
        taking ``(x, Z, *params)``.
    data : SurpyvalData
        Data split into observation types with covariates attached via
        ``add_covariates``.
    *params : float
        The distribution parameters followed by the regression parameters.
    """
    ll: Boxable = 0.0

    if data.x_o.size > 0:
        ll = ll + (data.n_o * model.log_df(data.x_o, data.Z_o, *params)).sum()

    if data.x_r.size > 0:
        ll = ll + (data.n_r * model.log_sf(data.x_r, data.Z_r, *params)).sum()

    if data.x_l.size > 0:
        ll = ll + (data.n_l * model.log_ff(data.x_l, data.Z_l, *params)).sum()

    if data.x_il.size > 0:
        # Interval censoring: P(xl < X < xr | Z) = F(xr) - F(xl).
        right = model.ff(data.x_ir, data.Z_i, *params)
        left = model.ff(data.x_il, data.Z_i, *params)
        ll = ll + (data.n_i * np.log(np.maximum(right - left, _TINY))).sum()

    ll = ll - truncation_correction(model, data, *params)

    return -ll
