"""The robust (sandwich) covariance of a parametric fit with weights ``w``.

A weighted maximum likelihood fit maximises the pseudo-likelihood
``sum_i n_i w_i l_i(theta)``, whose inverse Hessian ``H^-1`` is not the
covariance of the estimate: it treats each weight as that many units
seen. The robust covariance is ``H^-1 (sum_i n_i w_i^2 s_i s_i') H^-1``,
``s_i`` the score of one unit of row ``i`` (Huber, 1967; Binder, 1983),
as R's ``survreg(..., weights = w, robust = TRUE)`` computes it. It is
carried to every Wald bound (``cb``, ``param_cb``, ``quantile_cb``, ...)
by the delta method as the classical covariance is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.utils.weights import (
    effective_sample_size,
    refuse_weighted,
    row_scores,
    sandwich,
    warn_if_few_effective,
)


def check_weighted_fit(surv_data: Any, how: str) -> None:
    """Refuse a weighted fit by a method other than maximum likelihood,
    and warn where the weights leave few effective units."""
    if not surv_data.weighted:
        return
    if how != "MLE":
        raise refuse_weighted(
            f"how={how!r}",
            "probability plotting, moments, mean square error and product "
            "spacings have no robust variance for weighted data",
            "Fit with how='MLE' (the default), which takes w.",
        )
    warn_if_few_effective(surv_data.n, surv_data.w)


def make_robust(model: Any) -> None:
    """Give a parametric ``model`` fitted to weighted data its robust
    covariance: ``covariance()`` (and ``hess_inv``, its block of the
    distribution's parameters) become the sandwich, so that every Wald
    bound uses it. The offset ``gamma`` is held at its estimate, as for
    the classical covariance."""
    data = model.surv_data
    model.weighted = True
    model.effective_n = effective_sample_size(data.n, data.w)
    # The fit searched with the weights scaled to sum to the units
    # (``likelihood_weights``); its value is reported with the weights as
    # given, as R's ``survreg`` reports it.
    n = np.asarray(data.n, dtype=float)
    scale = float(np.sum(n * data.w) / n.sum())
    if getattr(model, "_neg_ll", None) is not None:
        model._neg_ll = model._neg_ll * scale
    cov = model._covariance
    if cov is None:
        return
    dist = model.dist
    n_core = len(model.params)
    theta = [*model.params]
    if model.lfp:
        theta.append(model.lfp_p)
    if model.zi:
        theta.append(model.f0)
    lfp, zi, gamma = model.lfp, model.zi, model.gamma
    f0_hat, p_hat = model.f0, model.lfp_p

    def loglik(d: Any, th: Any) -> Any:
        rest = [th[i] for i in range(n_core, len(theta))]
        f0 = rest.pop() if zi else f0_hat
        p = rest.pop() if lfp else p_hat
        core = [th[i] for i in range(n_core)]
        return dist._log_likelihood(d, *core, gamma, f0, p)

    scores = row_scores(loglik, data, np.asarray(theta, dtype=float))
    weights = data.likelihood_weights
    robust = sandwich(np.asarray(cov, dtype=float), scores, data.n, weights)
    model._covariance = robust
    model.hess_inv = robust[:n_core, :n_core].copy()
