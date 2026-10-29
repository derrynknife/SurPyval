"""The likelihood-ratio test of a regression model against the model with
no covariate effect (the null model), for ``likelihood_ratio_test()``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .parametric_regression_model import ParametricRegressionModel
    from .semi_parametric_regression_model import (
        SemiParametricRegressionModel,
    )


def _result(ll: float, ll_null: float, df: int) -> dict:
    from scipy.stats import chi2

    if df < 1:
        raise ValueError(
            "The model has no estimated covariate coefficient (each is "
            "fixed or aliased), so there is nothing to test against the "
            "model without covariates."
        )
    # The fit maximises over a larger set than the null model, so the
    # statistic is not negative; a tiny negative value is round-off.
    statistic = max(2.0 * (ll - ll_null), 0.0)
    return {
        "statistic": float(statistic),
        "df": int(df),
        "p_value": float(chi2.sf(statistic, df)),
        "log_likelihood": float(ll),
        "log_likelihood_null": float(ll_null),
    }


def cox_likelihood_ratio_test(model: SemiParametricRegressionModel) -> dict:
    """The global test of a fitted Cox model: twice the gain in the log
    partial likelihood over all the coefficients at 0, on as many degrees
    of freedom as there are estimated coefficients."""
    neg_ll_of = getattr(model, "neg_ll_of", None)
    if neg_ll_of is None or getattr(model, "_neg_ll", None) is None:
        raise ValueError(
            "likelihood_ratio_test needs the partial likelihood of the data "
            "the model was fitted to, which a model restored with "
            "from_dict / from_json does not carry. Call it on the fitted "
            "model, or refit."
        )
    beta = np.asarray(model.params, dtype=float)
    ll_null = -float(neg_ll_of(np.zeros(beta.shape)))
    return _result(-float(model.neg_ll()), ll_null, model._ic_k())


def parametric_likelihood_ratio_test(
    model: ParametricRegressionModel,
) -> dict:
    """The global test of a fitted parametric regression model: the
    baseline distribution fitted to the same data (every coefficient at 0,
    where each built-in link leaves the baseline unchanged) is the null
    model."""
    from .parametric_regression_model import _SERIALISABLE_REG_NAMES

    if not model._is_linear_predictor() or (
        model.reg_model.name not in _SERIALISABLE_REG_NAMES
    ):
        what = (
            "an accelerated life model, whose life-model parameters are "
            "not coefficients that are 0 without an effect"
            if model._is_accelerated_life()
            else "a model with a custom covariate link, which need not "
            "leave the baseline unchanged at coefficients of 0"
        )
        raise ValueError(
            "likelihood_ratio_test is not available for {}. Fit the "
            "model without covariates yourself (the distribution's own "
            "fit) and compare log_likelihood: twice the difference is "
            "chi-squared on the difference in the number of "
            "parameters.".format(what)
        )
    model._require_data("likelihood_ratio_test")
    dist = model.distribution
    fixed: dict[str, Any] = {
        name: value
        for name, value in (model.fixed or {}).items()
        if name in model.distribution_param_map
    }
    data = model.data
    kwargs: dict[str, Any] = {} if not fixed else {"fixed": fixed}
    null = dist.fit(  # type: ignore[attr-defined]
        x=data.x, c=data.c, n=data.n, t=data.t, **kwargs
    )
    k_null = len(model.distribution_param_map) - len(fixed)
    return _result(
        -float(model.neg_ll()), -float(null.neg_ll()), model.k - k_null
    )
