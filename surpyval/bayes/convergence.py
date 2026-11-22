"""Whether a sampler's draws have converged: R-hat, effective sample
size, divergences and tree depth.

The formulas are those of Vehtari, Gelman, Simpson, Carpenter and
Burkner (2021), "Rank-normalization, folding, and localization: an
improved R-hat for assessing convergence of MCMC", as Stan and ArviZ
compute them:

- :func:`rhat`: each chain is split in half, the draws are replaced by
  the normal scores of their ranks over all the chains, and the usual
  ratio (the variance between and within the half chains, over the
  variance within) is taken; again for the draws folded about their
  median, which sees chains that differ in spread; the larger is the
  answer. Chains that have mixed give 1.
- :func:`ess_bulk`: the effective number of independent draws for the
  centre of the distribution, from the rank-normalised split chains'
  autocorrelations (summed in pairs while the pairs stay positive and
  fall, Geyer's initial monotone sequence).
- :func:`ess_tail`: the smaller of the effective sizes for the 5% and
  the 95% quantile (of the indicators of a draw being below each).

The effective sample sizes do not change when a parameter is put on
another scale by an increasing function (they are the same for ``alpha``
and ``log alpha``); nor does R-hat's first part, but its folded part
depends on the scale a little. A fit reports them for the parameters on
their own scale.

:func:`check` applies the rules a fit by ``method="nuts"`` must pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri
from scipy.stats import rankdata

__all__ = ["rhat", "ess", "ess_bulk", "ess_tail", "summarise", "check"]

#: R-hat above this fails (Vehtari et al., 2021).
RHAT_MAX = 1.01
#: The effective sample size each chain must give, bulk and tail (400
#: over four chains, the same paper's rule).
ESS_PER_CHAIN = 100


def _chains(x: npt.ArrayLike) -> npt.NDArray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] < 4:
        raise ValueError(
            "The draws must be one row per chain, (chains, draws), with at "
            f"least 4 draws a chain; got shape {x.shape}."
        )
    return x


def _split(x: npt.NDArray) -> npt.NDArray:
    """Each chain's two halves as chains (the middle draw of an odd
    length dropped)."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half :]], axis=0)


def _z_scale(x: npt.NDArray) -> npt.NDArray:
    """The normal scores of the draws' ranks over every chain (ties get
    their mean rank; Blom's offsets)."""
    r = rankdata(x, method="average").reshape(x.shape)
    return ndtri((r - 0.375) / (x.size + 0.25))


def _plain_rhat(x: npt.NDArray) -> float:
    n = x.shape[1]
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    between = float(np.var(np.mean(x, axis=1), ddof=1))
    if not within > 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * within + between) / within))


def rhat(x: npt.ArrayLike) -> float:
    """
    The rank-normalised split R-hat of one quantity's draws.

    Parameters
    ----------
    x : array like, ``(chains, draws)``
        The draws, one row per chain (a single row is one chain).

    Returns
    -------
    float
        1 where the chains agree; ``nan`` where the draws are constant.

    Examples
    --------
    Two chains that sit in different places have not converged:

    >>> import numpy as np
    >>> from surpyval.bayes.convergence import rhat
    >>> rng = np.random.default_rng(0)
    >>> same = rng.normal(size=(4, 500))
    >>> bool(rhat(same) < 1.01)
    True
    >>> apart = same + np.array([[0.0], [0.0], [3.0], [3.0]])
    >>> bool(rhat(apart) > 1.5)
    True
    """
    x = _chains(x)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return float("nan")
    s = _split(x)
    bulk = _plain_rhat(_z_scale(s))
    folded = _plain_rhat(_z_scale(np.abs(s - np.median(s))))
    return float(np.nanmax([bulk, folded]))


def _autocovariance(x: npt.NDArray) -> npt.NDArray:
    """Each chain's autocovariance at every lag (divisor: the chain's
    length), by the fast Fourier transform."""
    n = x.shape[1]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    centred = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(centred, size, axis=1)
    return np.fft.irfft(f * np.conj(f), size, axis=1)[:, :n] / n


def ess(x: npt.ArrayLike) -> float:
    """
    The effective sample size of the draws as they are (no splitting and
    no ranks): the number of chains times draws over the integrated
    autocorrelation time, which sums the autocorrelations in pairs while
    the pairs stay positive and fall (Geyer's initial monotone sequence).
    :func:`ess_bulk` and :func:`ess_tail` are this of transformed draws.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.convergence import ess
    >>> x = np.random.default_rng(0).normal(size=(4, 1000))
    >>> bool(3000 < ess(x) < 5000)
    True
    """
    x = _chains(x)
    m, n = x.shape
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return float("nan")
    acov = _autocovariance(x)
    mean_var = float(np.mean(acov[:, 0])) * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus += float(np.var(x.mean(axis=1), ddof=1))
    if not var_plus > 0:
        return float("nan")
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    kept = np.zeros(n)
    even, odd = 1.0, float(rho[1])
    kept[0], kept[1] = even, odd
    t = 1
    while t < n - 3 and even + odd > 0:
        even, odd = float(rho[t + 1]), float(rho[t + 2])
        if even + odd >= 0:
            kept[t + 1], kept[t + 2] = even, odd
        t += 2
    max_t = t - 2
    if even > 0:
        kept[max_t + 1] = even
    t = 1
    while t <= max_t - 2:
        if kept[t + 1] + kept[t + 2] > kept[t - 1] + kept[t]:
            kept[t + 1] = (kept[t - 1] + kept[t]) / 2.0
            kept[t + 2] = kept[t + 1]
        t += 2
    total = m * n
    tau = -1.0 + 2.0 * float(np.sum(kept[: max_t + 1])) + kept[max_t + 1]
    tau = max(tau, 1.0 / np.log10(total))
    return float(total / tau)


def ess_bulk(x: npt.ArrayLike) -> float:
    """
    The bulk effective sample size: :func:`ess` of the rank-normalised
    split chains (how well the centre of the distribution is known).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.convergence import ess_bulk
    >>> x = np.random.default_rng(0).normal(size=(4, 1000))
    >>> bool(ess_bulk(x) > 400)
    True
    """
    x = _chains(x)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return float("nan")
    return ess(_z_scale(_split(x)))


def ess_tail(x: npt.ArrayLike) -> float:
    """
    The tail effective sample size: the smaller of :func:`ess` for the
    5% and the 95% quantile (how well a 90% interval's ends are known).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.convergence import ess_tail
    >>> x = np.random.default_rng(0).normal(size=(4, 1000))
    >>> bool(ess_tail(x) > 400)
    True
    """
    x = _chains(x)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return float("nan")
    out = []
    for p in (0.05, 0.95):
        below = (x <= np.quantile(x, p)).astype(float)
        out.append(ess(_split(below)))
    return float(np.nanmin(out)) if not np.all(np.isnan(out)) else float("nan")


def summarise(
    draws: npt.ArrayLike,
    names: list[str],
    divergent: npt.ArrayLike | None = None,
    depth: npt.ArrayLike | None = None,
    max_depth: int | None = None,
) -> dict:
    """The convergence numbers of a sampler's draws
    (``(chains, draws, parameters)``): ``rhat``, ``ess_bulk`` and
    ``ess_tail`` by parameter name, ``divergences`` (the kept
    iterations whose trajectory diverged) and ``tree_depth_saturated``
    (those that reached ``max_depth`` doublings without turning)."""
    D = np.asarray(draws, dtype=float)
    out: dict[str, Any] = {
        "rhat": {n: rhat(D[:, :, j]) for j, n in enumerate(names)},
        "ess_bulk": {n: ess_bulk(D[:, :, j]) for j, n in enumerate(names)},
        "ess_tail": {n: ess_tail(D[:, :, j]) for j, n in enumerate(names)},
    }
    if divergent is not None:
        out["divergences"] = int(np.sum(divergent))
    if depth is not None and max_depth is not None:
        out["tree_depth_saturated"] = int(
            np.sum(np.asarray(depth) >= max_depth)
        )
        out["max_tree_depth"] = int(max_depth)
    return out


def check(summary: dict, chains: int, draws: int) -> list[str]:
    """What fails in a :func:`summarise` result, each as a sentence with
    the parameter and the number: R-hat above :data:`RHAT_MAX`, a bulk
    or tail effective sample size below :data:`ESS_PER_CHAIN` a chain,
    any divergence, and any iteration at the largest tree depth. Empty
    where everything passes."""
    need = ESS_PER_CHAIN * chains
    total = chains * draws
    problems = []
    bad = {
        n: v
        for n, v in summary["rhat"].items()
        if not v <= RHAT_MAX  # nan fails
    }
    if bad:
        problems.append(
            "the chains disagree (R-hat above {:g}: {})".format(
                RHAT_MAX, ", ".join(f"{n} {v:.3f}" for n, v in bad.items())
            )
        )
    for key, label in (("ess_bulk", "bulk"), ("ess_tail", "tail")):
        low = {n: v for n, v in summary[key].items() if not v >= need}
        if low:
            problems.append(
                "too few effective draws ({} effective sample size below "
                "{:d}: {})".format(
                    label,
                    need,
                    ", ".join(f"{n} {v:.0f}" for n, v in low.items()),
                )
            )
    k = summary.get("divergences", 0)
    if k:
        problems.append(
            f"{k} of the {total} kept iterations diverged (the step was "
            "too long for the posterior's curvature there)"
        )
    k = summary.get("tree_depth_saturated", 0)
    if k:
        problems.append(
            f"{k} of the {total} kept iterations reached the largest tree "
            f"depth ({summary.get('max_tree_depth')}) without turning"
        )
    return problems
