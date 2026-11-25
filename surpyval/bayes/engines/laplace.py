"""The Laplace approximation: the posterior as a normal at its peak.

In words: find the most probable parameter values (the posterior mode),
measure how sharply the posterior falls away from them in every
direction (its curvature there), and take the posterior to be the normal
distribution with that centre and that spread. It is done on the
parameters' unconstrained scale (logs of positive parameters, logits of
proportions; :mod:`surpyval.bayes.transforms`), where posteriors are
closest to normal, and every result is carried back to the parameters'
own scales.

It works for any number of free parameters, and is good where the
posterior is close to normal on that scale: moderate or large samples,
no parameter pressed against a bound. It is poor with very few failures
or a parameter near a bound, where the posterior is skewed. Every fit
therefore checks itself:

1. Draw ``DRAWS`` points from the normal, by a scrambled Sobol sequence
   with a fixed seed (quasi-random, so the same data give the same
   numbers on every run, and evenly spread).
2. At each, compare the exact log posterior with the normal's log
   density. The ratios are importance weights of the exact posterior
   against the normal. Pareto-smoothed importance sampling
   (:func:`surpyval.bayes.criteria.psis`) fits a generalised Pareto
   distribution to their largest values; its shape ``k`` measures how
   far the normal misses the posterior's tails (Yao et al. 2018). Below
   0.5 the approximation is good; up to 0.7 it is usable; above, it is
   not to be trusted.
3. Optionally (``importance=True``) the draws are reweighted by the
   smoothed weights, which moves every summary towards the exact
   posterior. It costs nothing more than the check. How far the
   reweighting moves each parameter's 2.5%, 50% and 97.5% points (as a
   share of the 95% interval's width) is the second check: past
   ``SHIFT`` the plain normal is off, even where ``k`` is moderate (a
   skewed posterior whose ratios still have a light tail).

The log evidence is Laplace's ``log p(mode) + d/2 log(2 pi) + 1/2 log
|cov|``, or, reweighted, the importance-sampling estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from surpyval.bayes.engines.grid import GridError, hessian

LogPost = Callable[[npt.NDArray], npt.NDArray]

#: The normal's draws: a power of two, which a Sobol sequence needs to
#: stay balanced.
DRAWS = 4096
#: The Sobol scrambling seed, fixed so that a fit is reproducible.
SEED = 20261125
#: Pareto k above which the approximation is not to be trusted
#: (``criteria.good_k`` at 4096 draws).
K_BAD = 0.7
#: Pareto k below which the approximation is good.
K_GOOD = 0.5
#: The most the reweighting may move a free parameter's 2.5%, 50% or
#: 97.5% point (as a share of its 95% interval's width) before the plain
#: approximation is called poor.
SHIFT = 0.1


@dataclass
class LaplaceResult:
    """The normal approximation and its check.

    ``phi`` are the draws of the free parameters (unconstrained scale);
    ``log_w`` their normalised log weights (equal unless reweighted).
    """

    phi: npt.NDArray
    log_w: npt.NDArray
    mode: npt.NDArray
    cov: npt.NDArray
    log_z: float
    log_z_laplace: float
    pareto_k: float
    shift: float
    shift_at: str
    ess: float
    importance: bool
    draws: int


def normal_draws(
    mode: npt.NDArray, cov: npt.NDArray, size: int = DRAWS, seed: int = SEED
) -> npt.NDArray:
    """``size`` quasi-random draws of the normal ``N(mode, cov)``: a
    scrambled Sobol sequence (fixed ``seed``) through the normal's
    quantile function and the Cholesky factor of ``cov``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.engines.laplace import normal_draws
    >>> z = normal_draws(np.zeros(2), np.eye(2), 1024)
    >>> z.shape, bool(np.all(np.abs(z.mean(axis=0)) < 0.01))
    ((1024, 2), True)
    """
    d = int(np.size(mode))
    m = int(np.ceil(np.log2(max(int(size), 2))))
    u = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[: int(size)]
    u = np.clip(u, 0.5 / 2**m, 1 - 0.5 / 2**m)
    L = np.linalg.cholesky(cov)
    out: npt.NDArray = np.asarray(mode, dtype=float) + ndtri(u) @ L.T
    return out


def normal_logpdf(
    phi: npt.NDArray, mode: npt.NDArray, cov: npt.NDArray
) -> npt.NDArray:
    """The log density of ``N(mode, cov)`` at the rows of ``phi``."""
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (np.atleast_2d(phi) - mode).T).T
    d = L.shape[0]
    out: npt.NDArray = (
        -0.5 * np.sum(z**2, axis=1)
        - float(np.sum(np.log(np.diag(L))))
        - 0.5 * d * np.log(2 * np.pi)
    )
    return out


def curvature(
    log_post: LogPost, mode: npt.NDArray, cov: npt.NDArray
) -> npt.NDArray:
    """The covariance of the normal at ``mode``: the inverse of minus the
    Hessian of ``log_post`` there, by central differences scaled to the
    spread in ``cov``. Refuses where the posterior does not curve down in
    every direction (no peak, or a ridge the data do not resolve)."""
    d = mode.size
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    h = np.clip(0.1 * sd, 1e-6, 0.5)
    with np.errstate(all="ignore"):
        H = -hessian(log_post, mode, h)
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        raise GridError(
            "The posterior's curvature at its peak is not finite (the peak "
            "is at the edge of where the density is defined), so no normal "
            "approximation can be made. Use method='grid' (up to three free "
            "parameters) or another sampler's draws (from_draws)."
        )
    vals, vecs = np.linalg.eigh(H)
    scale = max(float(np.max(np.abs(vals))), 1e-300)
    if np.any(vals <= 1e-10 * scale):
        worst = vecs[:, int(np.argmin(vals))]
        return_dir = int(np.argmax(np.abs(worst)))
        raise GridError(
            "The posterior does not curve down in every direction at its "
            f"peak (along parameter {return_dir + 1} of {d} most), so it "
            "has no normal approximation: the data and the priors do not "
            "determine every parameter. Give the parameters priors, fix "
            "some, or use method='grid' (up to three free parameters)."
        )
    out: npt.NDArray = (vecs / vals) @ vecs.T
    return 0.5 * (out + out.T)


def run_laplace(
    log_post: LogPost,
    mode: npt.NDArray,
    cov: npt.NDArray,
    *,
    importance: bool = False,
    draws: int = DRAWS,
    seed: int = SEED,
) -> LaplaceResult:
    """The Laplace approximation at ``mode`` (found by
    :func:`surpyval.bayes.engines.grid.find_mode`, whose ``cov`` sets the
    finite-difference steps), with its Pareto-k check and, where
    ``importance``, the draws reweighted towards the exact posterior."""
    from surpyval.bayes.criteria import psis

    mode = np.asarray(mode, dtype=float)
    cov = curvature(log_post, mode, cov)
    d = mode.size
    with np.errstate(all="ignore"):
        lp_mode = float(np.asarray(log_post(mode[None, :]))[0])
    sign, logdet = np.linalg.slogdet(cov)
    log_z_laplace = float(lp_mode + 0.5 * d * np.log(2 * np.pi) + 0.5 * logdet)
    phi = normal_draws(mode, cov, draws, seed)
    with np.errstate(all="ignore"):
        lp = np.asarray(log_post(phi), dtype=float)
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    log_r = lp - normal_logpdf(phi, mode, cov)
    finite = np.isfinite(log_r)
    S = len(log_r)
    if finite.sum() < 0.5 * S:
        k = np.inf
        smoothed = np.full(S, -np.log(S))
    else:
        safe = np.where(finite, log_r, np.min(log_r[finite]) - 745.0)
        smoothed, k = psis(safe)
        smoothed = np.where(finite, smoothed, -np.inf)
        smoothed = smoothed - logsumexp(smoothed)
    w = np.exp(smoothed)
    ess = float(1.0 / np.sum(w**2))
    log_z_is = float(logsumexp(np.where(finite, log_r, -np.inf)) - np.log(S))
    shift, shift_at = _shift(phi, w, mode, cov)
    use = bool(importance) and np.isfinite(k) and k <= K_BAD
    log_w = smoothed if use else np.full(S, -np.log(S))
    return LaplaceResult(
        phi=phi,
        log_w=log_w,
        mode=mode,
        cov=cov,
        log_z=log_z_is if use else log_z_laplace,
        log_z_laplace=log_z_laplace,
        pareto_k=float(k),
        shift=shift,
        shift_at=shift_at,
        ess=ess,
        importance=use,
        draws=S,
    )


def _shift(
    phi: npt.NDArray, w: npt.NDArray, mode: npt.NDArray, cov: npt.NDArray
) -> tuple[float, str]:
    """The largest move of a free parameter's 2.5%, 50% or 97.5% point
    from the normal's to the reweighted draws', as a share of the
    normal's 95% interval's width (on the unconstrained scale), and
    which: ``(share, "parameter j's p% point")``."""
    from surpyval.bayes.engines.grid import weighted_quantile

    q = np.array([0.025, 0.5, 0.975])
    z = ndtri(q)
    worst, where = 0.0, ""
    for j in range(phi.shape[1]):
        sd = float(np.sqrt(cov[j, j]))
        exact = mode[j] + sd * z
        got = weighted_quantile(phi[:, j], w, q)
        rel = np.abs(got - exact) / (2 * z[-1] * sd)
        i = int(np.argmax(rel))
        if rel[i] > worst or not where:
            worst = float(rel[i])
            where = f"{j}:{100 * q[i]:g}"
    return worst, where
