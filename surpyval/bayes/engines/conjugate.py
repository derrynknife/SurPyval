"""The textbook cases with a closed-form posterior.

- Exponential, a Gamma prior on ``failure_rate``, data that are exact or
  right censored, with any left truncation (the exponential forgets its
  age): the posterior is Gamma(a + r, b + T), with r the failures and T
  the total time on test from each unit's entry.
- Normal with ``sigma`` fixed, a Normal prior on ``mu``, exact data: the
  posterior is Normal.
- Bernoulli, a Beta prior on ``p``, outcomes 0 and 1: the posterior is
  Beta(a + ones, b + zeros).

Anything else (interval or left censoring, right truncation, another
prior) is not conjugate, and the grid is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import betaln, gammaln

#: Equal-probability points a conjugate posterior is summarised on for
#: the functions that have no closed form.
NODES = 4000


@dataclass
class ConjugateResult:
    name: str
    family: str
    params: tuple[float, float]
    log_evidence: float
    label: str


def _prior_family(prior: Any) -> tuple[str, tuple] | None:
    """(family, params) of a prior that is a plain Gamma, Beta or Normal
    SurPyval model, else None."""
    from surpyval.bayes.prior import DistributionPrior, NormalOnScale

    if isinstance(prior, DistributionPrior) and not prior.truncate:
        m = prior.model
        plain = (
            getattr(m, "gamma", 0) == 0
            and getattr(m, "lfp_p", 1) == 1
            and getattr(m, "f0", 0) == 0
        )
        if not plain:
            return None
        name = m.dist.name
        p = tuple(float(v) for v in np.atleast_1d(m.params))
        if name == "Gamma":
            return "gamma", p
        if name == "Beta":
            return "beta", p
        if name == "Normal":
            return "normal", p
    if isinstance(prior, NormalOnScale) and prior.kind == "identity":
        assert prior.loc is not None and prior.scale is not None
        return "normal", (prior.loc, prior.scale)
    return None


def applies(
    dist: Any, bound: Any, data: Any, offset: bool, lfp: bool, zi: bool
) -> str | None:
    """The conjugate case that applies, or None."""
    if offset or lfp or zi or bound.spec.joint:
        return None
    free = bound.free_names
    if len(free) != 1:
        return None
    prior = bound.spec.single.get(free[0])
    fam = None if prior is None else _prior_family(prior)
    if fam is None:
        return None
    c = np.asarray(data.c)
    t = np.asarray(data.t)
    if dist.name == "Exponential" and fam[0] == "gamma":
        if np.all(np.isin(c, (0, 1))) and not np.isfinite(t[:, 1]).any():
            return "exponential-gamma"
    if (
        dist.name == "Normal"
        and free == ["mu"]
        and fam[0] == "normal"
        and np.all(c == 0)
        and not np.isfinite(t).any()
    ):
        return "normal-normal"
    if dist.name == "Bernoulli" and fam[0] == "beta":
        return "bernoulli-beta"
    return None


def solve(case: str, dist: Any, bound: Any, data: Any) -> ConjugateResult:
    """The closed-form posterior of ``case`` (from :func:`applies`)."""
    free = bound.free_names[0]
    fam = _prior_family(bound.spec.single[free])
    assert fam is not None
    a, b = fam[1]
    n = np.asarray(data.n, dtype=float)
    c = np.asarray(data.c)
    x = np.asarray(data.x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    if case == "exponential-gamma":
        tl = np.asarray(data.t, dtype=float)[:, 0]
        entry = np.where(np.isfinite(tl), np.maximum(tl, 0.0), 0.0)
        r = float(n[c == 0].sum())
        T = float(np.sum(n * (x - entry)))
        log_z = (
            a * np.log(b)
            - gammaln(a)
            + gammaln(a + r)
            - (a + r) * np.log(b + T)
        )
        return ConjugateResult(
            free,
            "gamma",
            (a + r, b + T),
            float(log_z),
            f"Gamma prior on failure_rate: {r:g} failures in {T:g} time "
            "on test",
        )
    if case == "normal-normal":
        sigma = bound.fixed["sigma"]
        N = float(n.sum())
        ybar = float(np.sum(n * x) / N)
        S = float(np.sum(n * (x - ybar) ** 2))
        prec = 1.0 / b**2 + N / sigma**2
        mean = (a / b**2 + N * ybar / sigma**2) / prec
        s2 = sigma**2
        log_z = (
            -0.5 * N * np.log(2 * np.pi * s2)
            + 0.5 * np.log(s2 / (s2 + N * b**2))
            - S / (2 * s2)
            - N * (ybar - a) ** 2 / (2 * (s2 + N * b**2))
        )
        return ConjugateResult(
            free,
            "normal",
            (float(mean), float(np.sqrt(1.0 / prec))),
            float(log_z),
            "Normal prior on mu with sigma known",
        )
    if case == "bernoulli-beta":
        ones = float(n[x == 1].sum())
        zeros = float(n[x == 0].sum())
        log_z = betaln(a + ones, b + zeros) - betaln(a, b)
        return ConjugateResult(
            free,
            "beta",
            (a + ones, b + zeros),
            float(log_z),
            f"Beta prior on p: {ones:g} ones and {zeros:g} zeros",
        )
    raise ValueError(case)
