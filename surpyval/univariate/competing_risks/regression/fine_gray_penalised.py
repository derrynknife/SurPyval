"""The penalised Fine-Gray fit, ``FineGray.fit_penalised`` (see
:mod:`surpyval.univariate.regression._penalised` for the objective, the
solver and why a penalised model has no standard errors).

The objective is the negative weighted partial log-likelihood of the
subdistribution hazard (the one :meth:`FineGray.fit` maximises, with its
inverse-probability-of-censoring weights) over ``N``, the number of units,
plus the penalty: the objective of R's ``fastcmprsk::fastCrrp`` (Kawaguchi
et al. 2020) and ``crrp`` (Fu et al. 2017). The weighted partial
likelihood is a Cox partial likelihood on rows that carry the censoring
weights, so the fit is also glmnet's ``family = "cox"`` on those rows,
which is what the tests check it against.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.univariate.competing_risks.labels import label_mask
from surpyval.univariate.regression._penalised import (
    PenalisedObjective,
    penalised_fit,
)
from surpyval.utils.no_maximum import warn_unverified


class _Cached:
    """``f(b)`` kept for the last ``b``, so the value, gradient and
    Hessian at one point cost one evaluation of each."""

    def __init__(self, f: Any) -> None:
        self.f = f
        self.at: npt.NDArray | None = None
        self.value: Any = None

    def __call__(self, b: npt.NDArray) -> Any:
        if self.at is None or not np.array_equal(b, self.at):
            self.value = self.f(b)
            self.at = np.array(b, dtype=float)
        return self.value


def fit_fine_gray_penalised(
    x: npt.NDArray,
    Z: npt.NDArray,
    e: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    cause: Any,
    penalizer: Any,
    l1_ratio: Any,
    standardise: Any,
    center: bool,
    tol: float,
    strata: npt.NDArray | None,
    common_censoring: bool,
) -> dict:
    """The penalised fit of the Fine-Gray model for ``cause`` on the
    checked rows (``fine_gray._prepared``), as the dict
    ``FineGrayModel`` is built from, with the ``penalty``."""
    from .fine_gray import (
        _breslow_baseline,
        _partial_likelihood,
        _risk_set_sums,
        _strata_parts,
    )

    is_cause = label_mask(e, cause)
    is_event = (c == 0) & is_cause
    if not is_event.any():
        raise ValueError(f"No observed events for cause {cause!r}")
    is_competing = (c == 0) & ~is_cause
    parts = _strata_parts(
        x, c, n, is_event, is_competing, strata, common_censoring
    )
    fitted = [part for part in parts if part.sets.times.size]
    p = Z.shape[1]
    # Centred at the count-weighted means, as the unpenalised fit (which
    # changes nothing in the partial likelihood); the penalty's scale is
    # over every row, whatever its stratum.
    mean = (n @ Z) / n.sum()
    Zc = Z - mean
    denom0s = [_risk_set_sums(n[part.rows], part.sets) for part in fitted]
    offset = float(
        sum(
            part.sets.d @ np.log(denom0)
            for part, denom0 in zip(fitted, denom0s)
        )
    )
    every = np.arange(p)

    def likelihood(Z_s: npt.NDArray) -> tuple:
        nZ_events = [
            (n * is_event)[part.rows] @ Z_s[part.rows] for part in fitted
        ]
        return _partial_likelihood(n, fitted, denom0s, Z_s, nZ_events, every)

    def build(scale: npt.NDArray, N: float) -> PenalisedObjective:
        _, value_and_gradient, derivatives = likelihood(Zc / scale)
        first = _Cached(value_and_gradient)
        second = _Cached(derivatives)
        return PenalisedObjective(
            lambda b: float(first(b)[0]) / N,
            lambda b: np.asarray(first(b)[1], dtype=float) / N,
            lambda b: np.asarray(second(b)[1], dtype=float) / N,
            np.zeros(p),
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        fit = penalised_fit(
            build, Z, n, every, penalizer, l1_ratio, standardise, tol
        )
    res = fit.result
    if not res.verified:
        warn_unverified("The penalised Fine-Gray fit")
    beta = fit.coefficients
    _, value_and_gradient, _ = likelihood(Zc)
    neg_ll = float(value_and_gradient(beta)[0]) + offset
    baselines = [
        _breslow_baseline(part, Zc, n, beta, None if center else mean, Z)
        for part in parts
    ]
    out = {
        "cause": cause,
        "beta": beta,
        "center": mean if center else np.zeros(p),
        "p_values": np.full(p, np.nan),
        "cov": None,
        "model_cov": None,
        "baseline_times": baselines[0][0],
        "baseline_cumhaz": baselines[0][1],
        "neg_ll": neg_ll,
        "ic_n": float(n[is_event].sum()),
        "res": res,
        "maximum": "verified" if res.verified else "unverified",
        "objective": None,
        "fit_data": {
            "x": x,
            "Z": Z,
            "c": c,
            "n": n,
            "w": None,
            "is_event": is_event,
            "is_competing": is_competing,
            "strata": strata,
            "cluster": None,
        },
        "weighted": False,
        "effective_n": None,
        "penalty": fit.penalty,
    }
    if strata is not None:
        out["strata_baselines"] = baselines
    return out
