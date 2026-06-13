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

import autograd.numpy as np

# Smallest positive float; used to floor probability masses so that
# ``log`` stays finite for impossible (zero-width) intervals.
_TINY = np.finfo(float).tiny


def _ff_safe(model, x, Z, fill, *params):
    """``model.ff(x, Z, *params)`` with non-finite bounds replaced by their
    limiting probability without ever evaluating ``ff`` at a non-finite
    value.

    ``fill`` is the probability a non-finite entry maps to: ``0.0`` for a
    lower/left truncation bound (``F(-inf) = 0``) and ``1.0`` for an
    upper/right bound (``F(+inf) = 1``). The caller knows which bound it is
    passing, so the fill is supplied explicitly rather than inferred from
    the value of ``x``.
    """
    finite = np.isfinite(x)
    # Substitute a finite placeholder so ``ff`` is never called at +/-inf;
    # the placeholder's result is discarded for those entries below.
    x_safe = np.where(finite, x, 0.0)
    return np.where(finite, model.ff(x_safe, Z, *params), fill)


def truncation_correction(model, data, *params):
    """Log of the probability mass within each observation's truncation
    interval, summed over the truncated rows.

    The likelihood of a truncated observation is divided by
    ``P(tl < X < tr | Z)`` to account for the fact that it could only have
    been observed within that window. In log space this is a subtraction,
    so the caller subtracts the value returned here.
    """
    if data.x_tl.size == 0:
        return 0.0

    # Upper bound: F(+inf) = 1; lower bound: F(-inf) = 0.
    right = _ff_safe(model, data.x_tr, data.Z_t, 1.0, *params)
    left = _ff_safe(model, data.x_tl, data.Z_t, 0.0, *params)
    return (data.n_t * np.log(np.maximum(right - left, _TINY))).sum()


def regression_neg_ll(model, data, *params):
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
    ll = 0.0

    if data.x_o.size > 0:
        ll = ll + (
            data.n_o * model.log_df(data.x_o, data.Z_o, *params)
        ).sum()

    if data.x_r.size > 0:
        ll = ll + (
            data.n_r * model.log_sf(data.x_r, data.Z_r, *params)
        ).sum()

    if data.x_l.size > 0:
        ll = ll + (
            data.n_l * model.log_ff(data.x_l, data.Z_l, *params)
        ).sum()

    if data.x_il.size > 0:
        # Interval censoring: P(xl < X < xr | Z) = F(xr) - F(xl).
        right = model.ff(data.x_ir, data.Z_i, *params)
        left = model.ff(data.x_il, data.Z_i, *params)
        ll = ll + (
            data.n_i * np.log(np.maximum(right - left, _TINY))
        ).sum()

    ll = ll - truncation_correction(model, data, *params)

    return -ll
