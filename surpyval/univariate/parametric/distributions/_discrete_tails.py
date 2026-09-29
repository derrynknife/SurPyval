"""Helpers for the discrete distributions' tails (#449, #458): the log of a
ratio of gamma functions without cancellation, and an exact quantile from
accurate log tails."""

from typing import Any, Callable

import numpy.typing as npt
from scipy.special import gammaln as _sc_gammaln

from surpyval import np
from surpyval.utils.autograd_gamma_compat import log_gamma_ratio

__all__ = [
    "log_gamma_ratio",
    "poisson_log_tails",
    "reached",
    "refine_quantile",
]

# The relative slack of the quantile's comparison: a probability u = F(k)
# that came out of ``ff`` can recover a threshold an ulp on the wrong side
# of the step, but the tail references leave out any u within 1e-9 of a
# step, so a slack well below that keeps the quantile exact there.
_SLACK = 1e-12


def reached(
    k: Any,
    u: Any,
    log_sf: Callable[[Any], Any],
    log_ff: Callable[[Any], Any],
) -> npt.NDArray:
    """Whether F(k) >= u, compared on the smaller side and the log scale
    (log F against log u below 1/2, log R against log(1 - u) above), so
    neither loses a small probability."""
    lower = u <= 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.where(lower, np.log(u), np.log1p(-np.where(lower, 0, u)))
    slack = np.where(np.isfinite(target), _SLACK * np.abs(target), 0.0)
    return np.where(
        lower,
        log_ff(k) >= target - slack,
        log_sf(k) <= target + slack,
    )


def refine_quantile(
    k: Any,
    u: Any,
    log_sf: Callable[[Any], Any],
    log_ff: Callable[[Any], Any],
    first: float,
) -> npt.NDArray:
    """The smallest integer k >= ``first`` with F(k) >= u, from a start
    ``k`` (scipy's ``ppf``, which compares a CDF that has lost the digits
    of a small upper tail: near u = 1 it was one short, or 2.5% short for
    a Negative Binomial with p = 1e-6). A bracket is grown from the start
    by doubling steps, then bisected over the integers."""
    k = np.array(k, dtype=float, ndmin=1)
    u = np.broadcast_to(np.asarray(u, dtype=float), k.shape)
    todo = np.isfinite(k) & (u > 0) & (u < 1)
    k = np.where(todo, np.maximum(k, first), k)
    ok = np.where(todo, reached(k, u, log_sf, log_ff), True)
    # the bracket (lo, hi]: F(lo) < u <= F(hi), lo = first - 1 at most
    hi = np.where(ok, k, np.nan)
    lo = np.where(ok, np.nan, k)
    step = np.ones_like(k)
    for _ in range(1100):
        grow_down = todo & np.isnan(lo)
        grow_up = todo & np.isnan(hi)
        if not (grow_down.any() or grow_up.any()):
            break
        down = np.maximum(hi - step, first - 1.0)
        at_first = grow_down & (down < first)
        test_down = grow_down & ~at_first
        r_down = np.where(
            test_down,
            reached(np.where(test_down, down, first), u, log_sf, log_ff),
            True,
        )
        lo = np.where(at_first | (test_down & ~r_down), down, lo)
        hi = np.where(test_down & r_down, down, hi)
        up = lo + step
        r_up = np.where(
            grow_up,
            reached(np.where(grow_up, up, first), u, log_sf, log_ff),
            False,
        )
        hi = np.where(grow_up & r_up, up, hi)
        lo = np.where(grow_up & ~r_up, up, lo)
        step = step * 2.0
    for _ in range(1100):
        mid = np.floor(0.5 * (lo + hi))
        live = todo & (hi - lo > 1) & (mid > lo) & (mid < hi)
        if not live.any():
            break
        r_mid = reached(np.where(live, mid, hi), u, log_sf, log_ff)
        hi = np.where(live & r_mid, mid, hi)
        lo = np.where(live & ~r_mid, mid, lo)
    return np.where(todo, hi, k)


def poisson_log_tails(
    k: npt.NDArray, mu: npt.ArrayLike, log_sf: npt.NDArray, log_ff: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """The Poisson's log R(k) = log P(T > k) and log F(k), with the small
    tail summed from the mass (#442, #443).

    scipy's incomplete gamma, from which ``log_sf`` and ``log_ff`` come,
    is off by up to 1e-6 in a 1e-8 tail at mu = 1e6, and its log from the
    asymptotic series by 3e-4 where it underflows. Where a tail is below
    1e-3 it is instead the mass times a series whose terms shrink by the
    ratio of k to mu:

        F(k) = P(T = k) [1 + k / mu + k (k - 1) / mu^2 + ...]      (k < mu)
        R(k) = P(T = k + 1) [1 + mu / (k + 2) + ...]               (k > mu)

    and the other tail is log1p of minus it."""
    k, mu_arr, log_sf, log_ff = (
        np.array(v, dtype=float)
        for v in np.broadcast_arrays(k, mu, log_sf, log_ff)
    )
    with np.errstate(over="ignore"):
        tail = (k >= 0) & (np.minimum(log_sf, log_ff) < np.log(1e-3))
    if not np.any(tail):
        return log_sf, log_ff
    kt, mu = k[tail], mu_arr[tail]
    low = kt < mu
    # the mass the series starts from: at k (lower) or k + 1 (upper)
    j = np.where(low, kt, kt + 1.0)
    log_mass = j * np.log(mu) - mu - _sc_gammaln(j + 1.0)
    total = np.ones_like(kt)
    term = np.ones_like(kt)
    for n in range(1, 100000):
        ratio = np.where(low, (kt - n + 1.0) / mu, mu / (kt + 1.0 + n))
        term = term * np.maximum(ratio, 0.0)
        total = total + term
        if np.all(term <= 1e-17 * total):
            break
    small = log_mass + np.log(total)
    large = np.where(
        small > -np.log(2.0),
        np.log(-np.expm1(small)),
        np.log1p(-np.exp(small)),
    )
    log_ff[tail] = np.where(low, small, large)
    log_sf[tail] = np.where(low, large, small)
    return log_sf, log_ff
