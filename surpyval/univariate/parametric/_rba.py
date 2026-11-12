"""Abernethy's reduced bias adjustment (RBA) of a maximum-likelihood fit
(``Parametric.rba``).

In small samples the maximum-likelihood estimate of a Weibull shape is
too large (more often above the truth than below it), and that of a
(log)normal ``sigma`` too small. The adjustment multiplies the estimate by
a factor of the number of failures ``r``, through Gurland and Tripathi's
``C4``, ``C4(r) = sqrt(2 / (r - 1)) Gamma(r / 2) / Gamma((r - 1) / 2)``:

- Weibull shape, ``basis="median"`` (Abernethy's RBA): ``C4^3.5``, which
  makes the shape median-unbiased (as often above the truth as below);
- Weibull shape, ``basis="mean"``: ``C4^6``, mean-unbiased;
- LogNormal and Normal ``sigma``: ``sqrt(r / (r - 1)) / C4``, which turns
  the MLE ``sigma`` of a complete sample into ``s / C4``, the
  mean-unbiased estimate.

These are R ``WeibullR``'s ``rba`` and its ``"mle-rba"`` fit (Abernethy,
*The New Weibull Handbook*, 5th ed., chapter 5), which applies them to
maximum-likelihood fits only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from surpyval.utils.validation import check_option

#: The families with an adjustment, and the parameter it multiplies.
RBA_PARAMETER = {"Weibull": "beta", "LogNormal": "sigma", "Normal": "sigma"}

RBA_BASES = ("median", "mean")


def c4(r: int) -> float:
    """``C4(r) = sqrt(2 / (r - 1)) Gamma(r / 2) / Gamma((r - 1) / 2)``,
    the mean of the sample standard deviation of ``r`` standard normals.

    Examples
    --------
    >>> from surpyval.univariate.parametric._rba import c4
    >>> round(c4(10), 6)
    0.972659
    """
    return math.sqrt(2.0 / (r - 1)) * math.exp(
        math.lgamma(r / 2.0) - math.lgamma((r - 1) / 2.0)
    )


def rba_factor(r: int, family: str, basis: str = "median") -> float:
    """The factor the family's shape (Weibull) or ``sigma``
    ((log)normal) is multiplied by, for ``r`` failures.

    Examples
    --------
    >>> from surpyval.univariate.parametric._rba import rba_factor
    >>> round(rba_factor(10, "Weibull"), 6)
    0.907652
    """
    k = c4(r)
    if family == "Weibull":
        return k**3.5 if basis == "median" else k**6
    return math.sqrt(r / (r - 1.0)) / k


def rba(model: Any, basis: str = "median") -> Any:
    """``Parametric.rba``: the model with its shape or ``sigma``
    adjusted, built from parameters."""
    check_option("basis", basis, RBA_BASES)
    family = model.dist.canonical_name
    name = RBA_PARAMETER.get(family)
    if name is None:
        raise ValueError(
            "The reduced bias adjustment is defined for the Weibull shape "
            "and the LogNormal and Normal sigma; the "
            f"{family} has none. Use the fit as it is."
        )
    if model.method != "MLE":
        raise ValueError(
            "The reduced bias adjustment is for maximum-likelihood fits "
            f"(its factors are derived for them); this model was fitted by "
            f"{model.method}. Refit with how='MLE', then call rba()."
        )
    if model.offset or model.lfp or model.zi or model.weighted:
        raise ValueError(
            "The reduced bias adjustment is for a plain two-parameter fit; "
            "this one has an offset, a limited-failure or zero-inflated "
            "part, or weights, for which its factors do not hold. Use the "
            "fit as it is, and its bounds for the uncertainty."
        )
    if name in [
        model.dist.parameter_names[i] for i in model._user_fixed_idx()
    ]:
        raise ValueError(
            f"'{name}' was fixed at fit time, so there is no estimate of it "
            "to adjust. Refit without fixing it."
        )
    if family != "Weibull" and basis != "median":
        raise ValueError(
            f"'basis' is for the Weibull shape; the {family} sigma has one "
            "adjustment (to the mean-unbiased s / C4). Call rba() without "
            "'basis'."
        )
    data = getattr(model, "data", None)
    if data is None:
        raise ValueError(
            "The reduced bias adjustment needs the number of failures, from "
            "the data the model was fitted to; this model was restored "
            "without them. Save it with to_dict(with_data=True)."
        )
    t = data.get("t")
    if t is not None and np.isfinite(np.asarray(t, dtype=float)).any():
        raise ValueError(
            "The reduced bias adjustment is for samples without "
            "truncation, for which its factors are derived. Use the fit as "
            "it is."
        )
    c = np.asarray(data["c"])
    n = np.asarray(data["n"], dtype=float)
    r = int(round(float(np.sum(n[c != 1]))))
    if r < 3:
        raise ValueError(
            f"The reduced bias adjustment needs at least 3 failures; the "
            f"data have {r}. Use the fit as it is."
        )
    params = np.array([float(v) for v in np.ravel(model.params)])
    params[model.dist.param_map[name]] *= rba_factor(r, family, basis)
    out = model.with_params(params)
    out.bias_adjustment = f"rba ({basis}, {r} failures)"
    return out
