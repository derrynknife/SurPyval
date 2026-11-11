"""
The cause-by-cause assembly shared by the competing-risks models.

A competing-risks model with one model per cause builds its all-cause
and per-cause quantities the same way whatever the per-cause models are
(distributions, or regressions evaluated at covariate rows):

- the cumulative incidence of a cause, as an integral over that cause's
  own probability scale (:func:`incidence_on_probability_scale`);
- the cumulative incidence from hazard increments on a time grid, the
  Aalen-Johansen (multi-state) step (:func:`aalen_johansen_steps`);
- random draws, the earliest of the causes' latent times
  (:func:`first_failures`);
- the covariance of the causes' estimates, block-diagonal where the
  likelihood factorises over causes (:func:`block_diagonal`);
- the information-criterion terms summed over causes
  (:func:`summed_ic_terms`);
- Wald bounds on the incidence (logit scale) and on the all-cause
  functions (log cumulative hazard), from a delta-method variance
  (:func:`wald_cif_bound`, :func:`wald_all_cause_bound`), whose gradient
  :func:`central_differences` takes where it has no closed form.

``ParametricCompetingRisks``, ``ParametricCompetingRisksRegression`` and
the cause-specific Cox model (``CompetingRisksProportionalHazards``) use
these, so each computation exists once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.special import ndtri

from surpyval.utils.linalg import bound_signs, warn_wald_undefined
from surpyval.utils.warnings import caller_stacklevel


def incidence_on_probability_scale(
    upper: npt.NDArray,
    quantile: Callable[[npt.NDArray], npt.ArrayLike],
    others: Sequence[Callable[[npt.NDArray], npt.ArrayLike]],
    ends: Sequence[float] | None = None,
) -> npt.NDArray:
    r"""
    The cumulative incidence of one cause at several points at once,

    .. math::
        \mathrm{CIF}_k = \int_0^{F_k} \prod_{j \neq k}
        S_j\left(F_k^{-1}(p)\right) dp
        = F_k \int_0^1 \prod_{j \neq k} S_j\left(F_k^{-1}(u F_k)\right) du,

    the integral of :math:`h_k S` over time after the substitution
    :math:`p = F_k(t)`, by one adaptive quadrature over every point (to a
    relative accuracy of about :math:`10^{-10}`). The integrand is
    bounded by 1 and monotone, so the result is as accurate where the
    cause's density is infinite (a Weibull shape below 1) or heavy-tailed
    as anywhere else.

    Parameters
    ----------
    upper : numpy array
        :math:`F_k` at each point (in ``(0, 1]``): the cause's own
        failure probability by the point's time.
    quantile : callable
        ``quantile(p)``: the cause's quantile at the probabilities ``p``,
        one per point (the cause's ``qf`` at each point's covariates for a
        regression).
    others : sequence of callables
        ``survival(t)`` of each other cause at the times ``t``, one per
        point.
    ends : sequence of float, optional
        Each other cause's survival at the far end of its support, used
        where the cause's quantile is infinite (a cure fraction). Default
        0 for every cause.

    Returns
    -------
    numpy array
        The cumulative incidence at each point.

    Examples
    --------
    Two exponential causes with rates 1 and 2: the first's incidence by
    ``t`` is ``(1 - exp(-3 t)) / 3``.

    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks._assembly import (
    ...     incidence_on_probability_scale,
    ... )
    >>> t = np.array([0.5, 2.0])
    >>> cif = incidence_on_probability_scale(
    ...     -np.expm1(-t),
    ...     lambda p: -np.log1p(-p),
    ...     [lambda s: np.exp(-2 * s)],
    ... )
    >>> np.allclose(cif, -np.expm1(-3 * t) / 3, rtol=1e-12)
    True
    """
    upper = np.asarray(upper, dtype=float)
    s_end = [0.0] * len(others) if ends is None else list(ends)

    def integrand(u: float) -> npt.NDArray:
        with np.errstate(all="ignore"):
            s = np.asarray(quantile(u * upper), dtype=float).ravel()
            val = np.ones_like(s)
            far = np.isposinf(s)
            for survival, end in zip(others, s_end):
                sj = np.asarray(survival(s), dtype=float).ravel()
                val = val * np.where(far, end, sj)
        # Below a model's support its survival formula can give NaN; it
        # is 1 there.
        return np.where(np.isfinite(val), val, 1.0)

    integral, _ = quad_vec(
        integrand,
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-11,
        norm="max",
        limit=2000,
    )
    return upper * np.asarray(integral, dtype=float)


def aalen_johansen_steps(increments: npt.ArrayLike) -> npt.NDArray:
    r"""
    Each cause's step in cumulative incidence over consecutive intervals
    of time, from the causes' cumulative-hazard increments over them: the
    Aalen-Johansen estimate of the competing-risks multi-state model
    (one "event-free" state, one absorbing state per cause), with each
    step's transition probabilities the matrix exponential of the step's
    hazards, as R's ``survfit`` takes them for a multi-state ``coxph``.

    Over a step the causes' hazards add :math:`dH_k` each and :math:`dH`
    in all. A unit event-free before it, with probability
    :math:`S(t-) = \exp\{-H(t-)\}`, fails from cause :math:`k` over it
    with probability :math:`S(t-)\,(dH_k / dH)\,(1 - e^{-dH})`. The
    steps of the causes sum to the step in :math:`1 - \exp(-H)`, and a
    step stays a probability however large the increment. Exact where
    the causes' hazards are proportional to each other within each step
    (piecewise-constant hazards, or the jumps of a step estimate); on a
    fine grid of a smooth model, a numerical route to its incidence.

    Parameters
    ----------
    increments : array_like
        ``(n_causes, n_steps)``: the increments of each cause's
        cumulative hazard over each step.

    Returns
    -------
    numpy array
        ``(n_causes, n_steps)``: each cause's step in cumulative
        incidence; the cumulative sum along the steps is the incidence.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks._assembly import (
    ...     aalen_johansen_steps,
    ... )
    >>> steps = aalen_johansen_steps([[0.1, 0.2], [0.3, 0.0]])
    >>> steps.round(6)
    array([[0.08242 , 0.12192 ],
           [0.247261, 0.      ]])
    >>> bool(np.isclose(steps.sum(), 1 - np.exp(-0.6)))
    True
    """
    increments = np.asarray(increments, dtype=float)
    total = increments.sum(axis=0)
    # exp(-H(t-)): the survival just before each step.
    before = np.exp(-np.concatenate([[0.0], np.cumsum(total)[:-1]]))
    positive = total > 0
    share = np.where(
        positive, increments / np.where(positive, total, 1.0), 0.0
    )
    return before * -np.expm1(-total) * share


def first_failures(latent: npt.NDArray, causes: Sequence[Any]) -> npt.NDArray:
    """
    Competing-risks draws from latent failure times: each unit fails at
    the earliest of its causes' latent times (one column per cause, in
    the order of ``causes``), from that cause. A unit whose every latent
    time is infinite (a cure fraction) never fails: ``x = inf`` and cause
    ``None``.

    Returns a structured array with fields ``x`` (the time) and ``e``
    (the cause).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks._assembly import (
    ...     first_failures,
    ... )
    >>> out = first_failures(np.array([[3.0, 1.0], [2.0, 5.0]]), ["a", "b"])
    >>> out["x"], list(out["e"])
    (array([1., 2.]), ['b', 'a'])
    """
    latent = np.asarray(latent, dtype=float)
    size = latent.shape[0]
    idx = np.argmin(latent, axis=1)
    x = latent[np.arange(size), idx]
    e = np.array([causes[i] for i in idx], dtype=object)
    never = ~np.isfinite(x)
    if never.any():
        e[never] = None
    out = np.empty(size, dtype=[("x", float), ("e", object)])
    out["x"] = x
    out["e"] = e
    return out


def block_diagonal(blocks: Sequence[npt.ArrayLike]) -> npt.NDArray:
    """The block-diagonal matrix of the causes' covariances, zeros
    between causes: exact where the likelihood factorises over the
    causes, since their estimates then share no information.

    Examples
    --------
    >>> from surpyval.univariate.competing_risks._assembly import (
    ...     block_diagonal,
    ... )
    >>> block_diagonal([[[1.0]], [[2.0, 0.5], [0.5, 3.0]]])
    array([[1. , 0. , 0. ],
           [0. , 2. , 0.5],
           [0. , 0.5, 3. ]])
    """
    mats = [np.atleast_2d(np.asarray(b, dtype=float)) for b in blocks]
    size = sum(b.shape[0] for b in mats)
    out = np.zeros((size, size))
    start = 0
    for b in mats:
        stop = start + b.shape[0]
        out[start:stop, start:stop] = b
        start = stop
    return out


def cause_ic_terms(model: Any) -> tuple[int, float]:
    """``(k, n)`` of one cause's model: the parameters it estimated and
    the sample size of its BIC (its observed failures). A parametric
    model counts only the parameters it estimated (``_ic_k``); a
    Royston-Parmar model carries both as attributes."""
    if hasattr(model, "_ic_k"):
        return int(model._ic_k()), float(model._ic_sample_size())
    return int(model.k), float(model._ic_n)


def summed_ic_terms(models: Sequence[Any]) -> tuple[int, float]:
    """``(K, n)`` of a model of several causes fitted cause by cause: the
    parameters estimated over every cause, and the failures of every
    cause (each cause's fit counts its own failures)."""
    k_total, n_total = 0, 0.0
    for model in models:
        k_cause, n_cause = cause_ic_terms(model)
        k_total += k_cause
        n_total += n_cause
    return k_total, n_total


def central_differences(
    func: Any, est: npt.NDArray, cov: npt.NDArray, size: int
) -> npt.NDArray:
    """The gradient of ``func`` (a vector of ``size`` values) at ``est``
    by central differences, one row per parameter: each step ``1e-5`` of
    the parameter's value or standard error, whichever is larger. A
    parameter with no variance (fixed) gets a zero row; where a step
    leaves the parameter's range (``from_params`` refuses it), the
    one-sided difference on the other side is used."""
    grad = np.zeros((est.size, size))
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    centre: list = []

    def at(values: npt.NDArray) -> Any:
        try:
            return np.ravel(np.asarray(func(values), dtype=float))
        except ValueError:
            return None

    def mid() -> npt.NDArray:
        if not centre:
            centre.append(np.ravel(np.asarray(func(est), dtype=float)))
        return centre[0]

    for j in range(est.size):
        if not sd[j] > 0:
            continue
        h = 1e-5 * max(abs(float(est[j])), float(sd[j]))
        up, down = est.copy(), est.copy()
        up[j] += h
        down[j] -= h
        f_up, f_down = at(up), at(down)
        if f_up is not None and f_down is not None:
            grad[j] = (f_up - f_down) / (2 * h)
        elif f_up is not None:
            grad[j] = (f_up - mid()) / h
        elif f_down is not None:
            grad[j] = (mid() - f_down) / h
        else:
            grad[j] = np.nan
    return grad


def checked_sd(
    var: npt.NDArray, t: npt.NDArray, on: str, where: npt.NDArray
) -> npt.NDArray:
    """``sqrt(var)``, ``nan`` with one warning where (among ``where``) the
    delta-method variance is negative or not finite. A variance a hair
    below 0 by rounding (a flat direction) is 0."""
    var = np.where((var < 0) & (var > -1e-14), 0.0, var)
    bad = where & ~(var >= 0) & ~np.isnan(t)
    if bad.any():
        warn_wald_undefined(
            f"{on} at x = {t[bad].tolist()}",
            "its delta-method variance is negative or not finite (the "
            "parameter covariance is not positive definite, or the "
            "function's derivatives overflow there)",
            stacklevel=caller_stacklevel(),
        )
    return np.sqrt(np.where(bad | ~(var >= 0), np.nan, var))


def wald_all_cause_bound(
    H: npt.NDArray,
    var: npt.NDArray,
    t: npt.NDArray,
    on: str,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """The Wald bound on the all-cause ``sf``, ``ff`` or ``Hf`` (``on``)
    from the all-cause cumulative hazard ``H`` and its delta-method
    variance ``var`` at the times ``t``: one bound on ``log H``, carried
    to the function asked for."""
    ok = np.isfinite(H) & (H > 0)
    sd = np.where(ok, checked_sd(var, t, "Hf", ok), 0.0) / np.where(ok, H, 1.0)
    alpha, signs = bound_signs(alpha_ci, bound)
    if on == "sf":
        # sf falls as H rises: its lower bound is at H's upper one.
        signs = -signs
    H_b = H[:, None] * np.exp(signs[None, :] * ndtri(1 - alpha) * sd[:, None])
    if on == "Hf":
        out = H_b
    elif on == "sf":
        out = np.exp(-H_b)
    else:
        out = -np.expm1(-H_b)
    return out if bound == "two-sided" else out[:, 0]


def wald_cif_bound(
    F: npt.NDArray,
    var: npt.NDArray,
    t: npt.NDArray,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """The Wald bound on a cumulative incidence ``F`` from its
    delta-method variance ``var`` at the times ``t``, on the logit scale
    (so it stays in ``(0, 1)``); ``F`` itself where it is 0 or 1."""
    ok = (F > 0) & (F < 1) & np.isfinite(F)
    sd = np.where(ok, checked_sd(var, t, "cif", ok), 0.0)
    sd_logit = sd / np.where(ok, F * (1 - F), 1.0)
    alpha, signs = bound_signs(alpha_ci, bound)
    centre = np.where(ok, F, 0.5)
    logit_F = np.log(centre) - np.log1p(-centre)
    moved = logit_F[:, None] + signs[None, :] * ndtri(1 - alpha) * (
        sd_logit[:, None]
    )
    out = np.where(ok[:, None], 1.0 / (1.0 + np.exp(-moved)), F[:, None])
    return out if bound == "two-sided" else out[:, 0]
