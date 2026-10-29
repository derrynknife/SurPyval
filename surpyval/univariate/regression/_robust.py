"""Robust ("sandwich") covariance of a fitted parametric regression.

The model-based covariance, the inverse of the observed information
:math:`A`, is right when the model is right and the rows are independent.
The sandwich

.. math::
    V = A^{-1} B A^{-1}, \\qquad B = \\sum_g s_g s_g^{\\top},

where :math:`s_g` is the score (the gradient of the log-likelihood) of
group :math:`g`, stays consistent when the distribution is wrong in
shape (Huber 1967; White 1982), and, with the groups the clusters of
correlated rows, when rows within a cluster are dependent (Liang and
Zeger 1986). With no clusters each unit is its own group: the HC0 form
of R's ``sandwich::sandwich`` and lifelines' ``robust=True``. No
small-sample factor is applied (as in both).

Counts are frequency weights, as the fit reads them: without clusters a
row with ``n = 3`` is three independent units (three groups), and with
clusters its three units share the row's label. Either way the result is
that of the data written out one row per unit.

The scores are the gradient, by central differences, of each row's own
log-likelihood: the terms of
:func:`~surpyval.univariate.regression._likelihood.regression_neg_ll`
computed row by row, so every kind of censoring and truncation the fit
takes is in them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from surpyval.utils.surpyval_data import SurpyvalData

    from .parametric_regression_model import ParametricRegressionModel

_TINY = np.finfo(float).tiny


def note_source_rows(
    data: SurpyvalData,
    rows_given: int,
    Z_given: npt.ArrayLike,
    order: npt.NDArray,
) -> None:
    """Record on ``data`` (the rows a fit runs on) which of the
    ``rows_given`` rows given to the fit each one is: rows with a missing
    covariate were dropped (``Z_given`` is the covariate array as given)
    and the rest put in ``order``. :func:`robust_covariance` reads it to
    match a cluster label given per row to the row it labels."""
    if len(data) == rows_given:
        kept = np.arange(rows_given)
    else:
        Z_arr = np.asarray(Z_given, dtype=float).reshape(rows_given, -1)
        # The rows drop_nonfinite_covariates kept (without its warning).
        kept = np.flatnonzero(np.isfinite(Z_arr).all(axis=1))
    data._source_rows = kept[np.asarray(order)]
    data._rows_given = rows_given


def row_log_likelihood(
    model: Any, data: SurpyvalData, *params: float
) -> npt.NDArray:
    """Each row's log-likelihood for one unit (not times its count), in
    the order of ``data``: ``n @ row_log_likelihood(...)`` is
    ``-regression_neg_ll(model, data, *params)``."""
    ll = np.zeros(len(data))
    if data.mask_o.any():
        ll[data.mask_o] = model.log_df(data.x_o, data.Z_o, *params)
    if data.mask_r.any():
        ll[data.mask_r] = model.log_sf(data.x_r, data.Z_r, *params)
    if data.mask_l.any():
        ll[data.mask_l] = model.log_ff(data.x_l, data.Z_l, *params)
    if data.mask_i.any():
        right = model.ff(data.x_ir, data.Z_i, *params)
        left = model.ff(data.x_il, data.Z_i, *params)
        ll[data.mask_i] = np.log(np.maximum(right - left, _TINY))
    if data.truncated_mask.any():
        ll[data.truncated_mask] -= _row_log_mass(model, data, *params)
    return ll


def _row_log_mass(
    model: Any, data: SurpyvalData, *params: float
) -> npt.NDArray:
    """Each truncated row's log probability of its truncation window, as
    :func:`~surpyval.univariate.regression._likelihood.truncation_correction`
    computes it (log space for a window bounded on one side)."""
    tl, tr, Z = data.x_tl, data.x_tr, data.Z_t
    lo, hi = np.isfinite(tl), np.isfinite(tr)
    out = np.zeros(tl.shape[0])
    both = lo & hi
    if both.any():
        window = model.ff(tr[both], Z[both], *params) - model.ff(
            tl[both], Z[both], *params
        )
        out[both] = np.log(np.maximum(window, _TINY))
    left = lo & ~hi
    if left.any():
        out[left] = model.log_sf(tl[left], Z[left], *params)
    right = ~lo & hi
    if right.any():
        out[right] = model.log_ff(tr[right], Z[right], *params)
    return out


def row_scores(
    model: ParametricRegressionModel,
    data: SurpyvalData,
    p_hat: npt.NDArray,
    free: list,
    step: npt.NDArray,
) -> npt.NDArray:
    """The ``(rows, len(p_hat))`` matrix of each row's score for one unit
    at ``p_hat``: central differences of :func:`row_log_likelihood` in
    the ``free`` parameters (``step`` theirs), 0 in the others."""
    scores = np.zeros((len(data), len(p_hat)))
    fitter = model.model
    with np.errstate(all="ignore"):
        for j, h in zip(free, step):
            up, down = p_hat.copy(), p_hat.copy()
            up[j] += h
            down[j] -= h
            scores[:, j] = (
                row_log_likelihood(fitter, data, *up)
                - row_log_likelihood(fitter, data, *down)
            ) / (2 * h)
    return scores


def _cluster_labels(
    model: ParametricRegressionModel, cluster: npt.ArrayLike
) -> npt.NDArray:
    """The cluster label of each row of ``model.data``, from ``cluster``,
    one label per row given to the fit."""
    data = model.data
    source = getattr(data, "_source_rows", None)
    rows_given = getattr(data, "_rows_given", len(data))
    labels = np.asarray(cluster)
    if labels.ndim != 1 or labels.shape[0] != rows_given:
        raise ValueError(
            "`cluster` must have one label per row given to the fit "
            "({}); got an array of shape {}. Pass the labels in the order "
            "of the rows given to `fit` (for `fit_from_df`, one per row of "
            "the DataFrame, such as `df['id']`).".format(
                rows_given, labels.shape
            )
        )
    if labels.dtype.kind == "f" and np.isnan(labels).any():
        raise ValueError(
            "`cluster` has {} missing label(s); give every row a cluster "
            "label (a row on its own can have a label of its own).".format(
                int(np.isnan(labels).sum())
            )
        )
    if source is None:
        if rows_given != len(data):
            raise ValueError(
                "`cluster` cannot be matched to the rows of this fit; "
                "refit the model to use clustered standard errors."
            )
        source = np.arange(rows_given)
    return labels[source]


def refuse_robust(model: ParametricRegressionModel) -> None:
    """Refuse a model the sandwich cannot be computed for, saying what to
    use instead."""
    if getattr(model, "penalty", None) is not None:
        raise ValueError(
            "Robust standard errors are not available for a penalised fit: "
            "the sandwich of a penalised estimate is not the covariance of "
            "an unbiased one, so intervals from it would not have their "
            "stated coverage. Refit the unpenalised model (fit) for "
            "inference."
        )
    if getattr(model, "is_tvc", False):
        raise ValueError(
            "Robust standard errors are not available for a fit_tvc model: "
            "its likelihood is over the episodes of each unit's covariate "
            "history, not over independent rows. Use param_cb(name, "
            "method='bootstrap') or method='lr' for bounds that do not "
            "rest on the model-based covariance."
        )
    if getattr(model, "_restored", False) or (
        getattr(model, "data", None) is None
        or getattr(model, "res", None) is None
    ):
        raise ValueError(
            "Robust standard errors need the data the model was fitted "
            "to, and this model has none (it was built from parameters or "
            "restored from a dict, which does not keep the data). Refit "
            "the model to the data."
        )


def robust_covariance(
    model: ParametricRegressionModel,
    cluster: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """The sandwich covariance of ``model.params`` (see the module), in
    the order of :attr:`parameter_names`, with the model-based
    covariance's conventions: a fixed parameter's row and column 0, an
    aliased coefficient's ``nan``."""
    refuse_robust(model)
    # The parameterisation the model-based covariance is computed in (the
    # centred fit's, behind a baseline reported at 0), and that covariance.
    p_hat, center, cov = model._inference_state()
    if not np.all(np.isfinite(cov)):
        return model.covariance()
    data = model.data
    if center is not None and np.any(center):
        from ._fit_skeleton import centred_copy

        data = centred_copy(data, center)
    names = model.parameter_names
    held = model._held()
    free = [i for i, nm in enumerate(names) if nm not in held]
    step = model._hessian_step(p_hat)[free]
    scores = row_scores(model, data, p_hat, free, step)
    n = np.asarray(data.n, dtype=float)
    if cluster is None:
        meat = (n[:, None] * scores).T @ scores
    else:
        labels = _cluster_labels(model, cluster)
        _, group = np.unique(labels, return_inverse=True)
        sums = np.zeros((group.max() + 1, scores.shape[1]))
        np.add.at(sums, group, n[:, None] * scores)
        meat = sums.T @ sums
    out = cov @ meat @ cov
    if model._fit_centring is not None:
        J = model._fit_centring[2]
        out = J @ out @ J.T
    if model.aliased.size:
        out = np.array(out, dtype=float)
        out[model.k_dist + model.aliased, :] = np.nan
        out[:, model.k_dist + model.aliased] = np.nan
    return out
