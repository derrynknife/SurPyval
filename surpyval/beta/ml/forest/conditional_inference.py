r"""Conditional-inference selection of a survival tree's splits
(``selection="ctree"``).

The default, greedy search takes the best cut over every feature and every
candidate value. A feature with many distinct values offers many more
cuts, so on data with no effect at all its best cut is better by chance
than that of a feature with few values: greedy search prefers
high-cardinality features whether or not they matter, and it always finds
a cut to make. Conditional inference (Hothorn, Hornik and Zeileis, 2006)
separates the two questions. At each node:

1. every row gets a score :math:`h_i` from the node's own split
   statistic, computed once from the pooled data of the node;
2. for each feature, the best cut is judged by a p-value that allows for
   the number of cuts it had to choose from, so every feature is on the
   same scale whatever its number of values;
3. the feature with the smallest p-value is chosen, and the node is split
   only if that p-value, Bonferroni-adjusted for the number of features
   tested, is below ``alpha_split``;
4. the chosen feature's cut is found by the tree's usual criterion for its
   ``kind``.

**Scores.** The non-parametric kind uses the log-rank scores of the
Turnbull-score split (the Savage scores :math:`\delta_i - \hat H(x_i)` on
right-censored data); with delayed entry, the martingale residuals
:math:`\delta_i - [\hat H(x_i) - \hat H(t_{l,i})]` of the Nelson-Aalen
estimate of the delayed-entry risk sets (Fu and Simonoff, 2017), whose sum
over a child is again the log-rank numerator :math:`O - E`. The
parametric kinds use the contribution of each row to the score (gradient
of the log-likelihood) of the working model at the node's pooled maximum
likelihood estimate, the construction of model-based conditional
inference trees (Zeileis, Hothorn and Hornik, 2008; Hothorn and Zeileis,
2021): one score, for :math:`\log \lambda`, for the exponential kind (for
right-censored data :math:`\delta_i - \hat\lambda x_i`, the parametric
analogue of the log-rank score), and two, for
:math:`(\log \alpha, \log \beta)`, for the Weibull kind, so it has power
against differences of shape as well as scale, as its deviance split has.
Every censoring type and truncation enters through the full likelihood.

**Statistic.** For a cut leaving the rows with the feature at most
:math:`v` on the left, with :math:`m` units on the left out of :math:`N`
(a row with count :math:`n_i` counts :math:`n_i` times), the linear
statistic :math:`T_m = \sum_{i \in L} n_i (h_i - \bar h)` has permutation
mean zero and covariance
:math:`\frac{m (N - m)}{N (N - 1)} \sum_i n_i (h_i - \bar h)(h_i - \bar
h)^\top` (Hothorn et al., 2006, Theorem 1). The standardised quadratic
form :math:`Q_m = T_m^\top \Sigma_m^{+} T_m` (the square of the
Turnbull-score split's statistic for one score) is maximised over the
feature's admissible cuts, those that leave both children
``min_leaf_samples`` rows and ``min_leaf_failures`` failures (thinned to
64 evenly spaced ones if there are more, as the deviance split thins its
candidates): the maximally selected statistic (Lausen and Schumacher,
1992). It is used
rather than ctree's default linear statistic in the raw feature value
because a tree splits at a cut: the maximum has power against a threshold
anywhere, and it depends on the feature only through the order of its
values, so, like the tree itself, the selection is unchanged by any
increasing transformation of a feature.

**p-value.** Under the permutation null the standardised statistics at
successive cuts are asymptotically those of a Brownian bridge, a
Gauss-Markov chain: each :math:`Q_k = |W_k|^2 \sim \chi^2_q`, :math:`q` the
rank of the scores' covariance, with
:math:`W_k = \rho_k W_{k-1} + \sqrt{1 - \rho_k^2}\, E_k` and
:math:`\rho_k = \sqrt{m_{k-1} (N - m_k) / (m_k (N - m_{k-1}))}`. The
p-value of the maximum :math:`b`,

.. math::

    P\left(\max_k Q_k \geq b\right) = P(Q_1 \geq b)
    + \sum_{k \geq 2} P(Q_1, \dots, Q_{k-1} < b \leq Q_k),

is computed exactly for that chain (Hothorn and Zeileis, 2008, compute
the same multivariate normal probability by Monte Carlo integration):
the density of :math:`|W_k|` over the paths that have stayed below
:math:`\sqrt b` is carried from cut to cut by Gauss-Legendre quadrature,
through the non-central :math:`\chi` law of :math:`|W_k|` given
:math:`|W_{k-1}|`. One cut gives the plain :math:`\chi^2_q` p-value, and
many closely spaced cuts little more than a few independent ones, as they
should. It is deterministic and costs well under a second per feature.
Two alternatives were rejected. The Hunter-Worsley (improved Bonferroni)
bound of the ``maxstat`` package (Worsley, 1982; Lausen, Sauerbrei and
Schumacher, 1994; Hothorn and Lausen, 2003) counts every up-crossing of
the chain, and with the many close cuts of a continuous feature it was
about twice the true p-value at 0.05 (for 200 rows), which would have
turned the bias against continuous features. A permutation p-value costs
a thousand passes per feature per node, is coarse when Bonferroni
multiplies it by the number of features, and, being Monte Carlo, would
make a tree depend on the order of the rows and differ between ``n=2``
and two identical rows (Design Principles 4 and 5).

References
----------
Hothorn, T., Hornik, K. and Zeileis, A., 2006. Unbiased recursive
partitioning: a conditional inference framework. *Journal of
Computational and Graphical Statistics*, 15(3), pp.651-674.

Lausen, B. and Schumacher, M., 1992. Maximally selected rank statistics.
*Biometrics*, 48(1), pp.73-85.

Lausen, B., Sauerbrei, W. and Schumacher, M., 1994. Classification and
regression trees (CART) used for the exploration of prognostic factors
measured on different scales. In *Computational Statistics*, pp.483-496.
Physica, Heidelberg.

Hothorn, T. and Zeileis, A., 2008. Generalized maximally selected
statistics. *Biometrics*, 64(4), pp.1263-1269.

Worsley, K.J., 1982. An improved Bonferroni inequality and applications.
*Biometrika*, 69(2), pp.297-302.

Hothorn, T. and Lausen, B., 2003. On the exact distribution of maximally
selected rank statistics. *Computational Statistics & Data Analysis*,
43(2), pp.121-137.

Zeileis, A., Hothorn, T. and Hornik, K., 2008. Model-based recursive
partitioning. *Journal of Computational and Graphical Statistics*, 17(2),
pp.492-514.

Hothorn, T. and Zeileis, A., 2021. Predictive distribution modeling using
transformation forests. *Journal of Computational and Graphical
Statistics*, 30(4), pp.1181-1196.

Fu, W. and Simonoff, J.S., 2017. Survival trees for left-truncated and
right-censored data, with application to time-varying covariate data.
*Biostatistics*, 18(2), pp.352-369.
"""

from collections.abc import Callable, Iterable
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import i0e, ive, roots_legendre
from scipy.stats import chi, chi2, ncx2, norm

from surpyval.beta.ml.forest.deviance_split import (
    _LOG_BETA_BOUNDS,
    _exp_neg_ll,
    _exp_neg_ll_parts,
    _exp_theta0,
    _wei_max_ll,
)
from surpyval.beta.ml.forest.turnbull_score_split import log_rank_scores
from surpyval.utils.surpyval_data import SurpyvalData

SELECTIONS = ("greedy", "ctree")

# The most cuts a feature's statistic is maximised over: above this, the
# admissible cuts are thinned to evenly spaced ones (in the order of the
# feature's values), as the deviance split thins its candidates.
_MAX_TEST_CUTS = 64


def parse_selection(selection: str, alpha_split: float) -> str:
    """Validate ``selection`` and ``alpha_split``; the selection name."""
    if selection not in SELECTIONS:
        raise ValueError(
            f"selection={selection!r} is invalid. Must be 'greedy' (the "
            "best cut over every feature) or 'ctree' (conditional "
            "inference: the feature by a p-value, then its cut)."
        )
    if (
        isinstance(alpha_split, bool)
        or not isinstance(alpha_split, (int, float, np.integer, np.floating))
        or not 0 < alpha_split <= 1
    ):
        raise ValueError(
            f"alpha_split must be a number in (0, 1], the size of the "
            f"test that decides whether a node splits; got {alpha_split!r}."
        )
    return selection


# ---------------------------------------------------------------------------
# Scores


def _bounds(data: SurpyvalData) -> tuple[NDArray, NDArray]:
    # Each row's event interval (L, R]: 0 < x for a left-censored row and
    # (x, inf) for a right-censored one; L = R for an observed row.
    x = np.asarray(data.x, dtype=float)
    c = np.asarray(data.c)
    lo = x if x.ndim == 1 else x[:, 0]
    hi = x if x.ndim == 1 else x[:, -1]
    return np.where(c == -1, -np.inf, lo), np.where(c == 1, np.inf, hi)


def nonparametric_scores(data: SurpyvalData) -> NDArray:
    r"""The log-rank score of each row of ``data``, as an ``(N, 1)`` array.

    Untruncated data: :func:`log_rank_scores` (the Turnbull scores, and
    the Savage scores on right-censored data). Left-truncated
    right-censored data: the martingale residuals
    :math:`\delta_i - [\hat H(x_i) - \hat H(t_{l,i})]` of the
    delayed-entry Nelson-Aalen estimate, the hazard accrued over the
    window :math:`(t_l, x]` the row was at risk in.
    """
    if not np.isfinite(data.t[:, 0]).any():
        return log_rank_scores(data)[:, None]
    # The tree refuses truncation with left or interval censoring, so
    # this data is observed and right censored.
    times, r, d = data.to_xrd()
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.cumsum(np.where(r > 0, d / r, 0.0))

    def H_at(q: NDArray) -> NDArray:
        # H just after q: 0 before the first time.
        idx = np.searchsorted(times, q, side="right") - 1
        return np.where(idx >= 0, H[np.maximum(idx, 0)], 0.0)

    x = np.asarray(data.x, dtype=float)
    x = x if x.ndim == 1 else x[:, 0]
    event = (np.asarray(data.c) == 0).astype(float)
    return (event - (H_at(x) - H_at(data.t[:, 0])))[:, None]


def _exp_hazard(theta: NDArray, q: NDArray) -> tuple[NDArray, NDArray]:
    # Exponential, theta = [log lambda]: H(q) = lambda q from 0, and its
    # gradient dH/dlog(lambda) = H.
    H = np.exp(theta[0]) * np.clip(q, 0.0, None)
    return H, H[:, None]


def _exp_dlog_h(theta: NDArray, x: NDArray) -> NDArray:
    return np.ones((x.size, 1))


def _wei_hazard(theta: NDArray, q: NDArray) -> tuple[NDArray, NDArray]:
    # Weibull, theta = [log alpha, log beta]: H(q) = (q / alpha)^beta, and
    # its gradient (-beta H, H log H), which is 0 at q <= 0.
    beta = np.exp(theta[1])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        log_H = np.where(
            q > 0, beta * (np.log(np.where(q > 0, q, 1.0)) - theta[0]), -np.inf
        )
        H = np.exp(log_H)
        H_log_H = np.where(q > 0, H * log_H, 0.0)
    return H, np.column_stack([-beta * H, H_log_H])


def _wei_dlog_h(theta: NDArray, x: NDArray) -> NDArray:
    # log h = log beta - log alpha + (beta - 1)(log x - log alpha)
    beta = np.exp(theta[1])
    with np.errstate(divide="ignore"):
        log_H = beta * (np.log(x) - theta[0])
    return np.column_stack([np.full(x.size, -beta), 1.0 + log_H])


Hazard = Callable[[NDArray, NDArray], tuple[NDArray, NDArray]]


def _interval_score(
    hazard: Hazard, theta: NDArray, a: NDArray, b: NDArray
) -> NDArray:
    # d/dtheta log(S(a) - S(b)) for a < b (b may be +inf), per row, in
    # the form stable when S(a) is tiny:
    # [-G(a) + exp(-(H(b) - H(a))) G(b)] / [1 - exp(-(H(b) - H(a)))].
    H_a, G_a = hazard(theta, a)
    finite_b = np.isfinite(b)
    H_b, G_b = hazard(theta, np.where(finite_b, b, 0.0))
    with np.errstate(invalid="ignore", over="ignore"):
        gap = np.where(finite_b, H_b - H_a, np.inf)
        survive = np.exp(-gap)
        tail = np.where(finite_b[:, None], survive[:, None] * G_b, 0.0)
        return (-G_a + tail) / -np.expm1(-gap)[:, None]


def _pooled_mle(data: SurpyvalData, kind: str) -> NDArray | None:
    # The node's pooled maximum likelihood estimate, found as the
    # deviance split finds its parent's; None without event information.
    theta0 = _exp_theta0(data)
    if theta0 is None:
        return None
    if kind == "weibull":
        log_alpha0 = -theta0
        box = ((log_alpha0 - 15.0, log_alpha0 + 15.0), _LOG_BETA_BOUNDS)
        return _wei_max_ll(data, box, np.array([log_alpha0, 0.0]))[1]
    parts = _exp_neg_ll_parts(data)
    bounds = (theta0 - 15.0, theta0 + 15.0)
    res = minimize_scalar(
        _exp_neg_ll,
        args=(parts,),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    candidates = [float(res.x), bounds[0], bounds[1]]
    return np.array(
        [min(candidates, key=lambda theta: _exp_neg_ll(theta, parts))]
    )


def parametric_scores(data: SurpyvalData, kind: str) -> NDArray | None:
    r"""Each row's contribution to the score of the working model of
    ``kind`` (``"exponential"`` or ``"weibull"``) at the node's pooled
    maximum likelihood estimate: an ``(N, 1)`` array for
    :math:`\log\lambda`, or ``(N, 2)`` for :math:`(\log\alpha,
    \log\beta)`. ``None`` if the data carries no event information.

    A row's log-likelihood is :math:`\log f(x)` if observed, and
    :math:`\log[S(L) - S(R)]` for an event in :math:`(L, R]` (right
    censored: :math:`R = \infty`; left censored: :math:`L = 0`), less
    :math:`\log[S(t_l) - S(t_r)]` if truncated to :math:`(t_l, t_r]`.
    """
    theta = _pooled_mle(data, kind)
    if theta is None:
        return None
    hazard, dlog_h = (
        (_wei_hazard, _wei_dlog_h)
        if kind == "weibull"
        else (_exp_hazard, _exp_dlog_h)
    )
    L, R = _bounds(data)
    observed = (np.asarray(data.c) == 0) | (L == R)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        scores = np.where(
            observed[:, None],
            dlog_h(theta, np.where(observed, R, 1.0))
            - hazard(theta, np.where(observed, R, 0.0))[1],
            _interval_score(hazard, theta, L, np.where(observed, np.inf, R)),
        )
        tl, tr = data.t[:, 0], data.t[:, 1]
        truncated = np.isfinite(tl) | np.isfinite(tr)
        if truncated.any():
            scores = scores - np.where(
                truncated[:, None],
                _interval_score(hazard, theta, tl, tr),
                0.0,
            )
    return scores


def node_scores(data: SurpyvalData, kind: str) -> NDArray | None:
    """The ``(N, q)`` scores the node of a ``kind`` tree tests its
    features with, or ``None`` if they carry no information (no event
    information, or a score that is not finite)."""
    scores: NDArray | None
    if kind == "non-parametric":
        scores = nonparametric_scores(data)
    else:
        scores = parametric_scores(data, kind)
    if scores is None or not np.isfinite(scores).all():
        return None
    return scores


# ---------------------------------------------------------------------------
# The maximally selected statistic and its p-value


@lru_cache(maxsize=64)
def _gauss_legendre(n_nodes: int) -> tuple[NDArray, NDArray]:
    # scipy's rule is O(n); numpy's leggauss solves an n x n eigenproblem.
    nodes, weights = roots_legendre(n_nodes)
    return nodes, weights


def _chi_transition(
    r_from: NDArray, r_to: NDArray, rho: float, sigma: float, q: int
) -> NDArray:
    # The density of |rho w + sigma E| at r_to, for |w| = r_from and E
    # standard normal in q dimensions (a non-central chi law): the
    # transition density of the chain of norms, as a (from, to) matrix.
    a = rho * r_from[:, None]
    t = r_to[None, :]
    near = np.exp(-((t - a) ** 2) / (2.0 * sigma**2))
    if q == 1:
        far = np.exp(-((t + a) ** 2) / (2.0 * sigma**2))
        return (near + far) / (sigma * np.sqrt(2.0 * np.pi))
    x = a * t / sigma**2
    if q == 2:
        # i0e is the same function as ive(0, .), several times faster
        return (t / sigma**2) * near * i0e(x)
    nu = q / 2.0 - 1.0
    return (t / sigma**2) * (t / a) ** nu * near * ive(nu, x)


def _chi_tail(
    h: float, r: NDArray, rho: float, sigma: float, q: int
) -> NDArray:
    # P(|rho w + sigma E| >= h) for |w| = r.
    if q == 1:
        return norm.sf((h - rho * r) / sigma) + norm.sf((h + rho * r) / sigma)
    return ncx2.sf(h**2 / sigma**2, q, (rho * r / sigma) ** 2)


def max_chain_sf(b: float, rho: NDArray, q: int) -> float:
    r"""The probability that the largest of a chain of standardised
    statistics :math:`Q_1, \dots, Q_K` reaches ``b``.

    Each :math:`Q_k = |W_k|^2 \sim \chi^2_q`, with :math:`W_k` standard
    normal in :math:`q` dimensions and a Gauss-Markov chain:
    :math:`W_k = \rho_k W_{k-1} + \sqrt{1 - \rho_k^2} E_k` (``rho`` has
    the :math:`K - 1` correlations between neighbours). The probability
    is the sum over :math:`k` of the probability that the chain first
    reaches ``b`` at :math:`k`, computed by carrying the density of
    :math:`|W_k|` on the paths that have not yet reached it forward from
    cut to cut, by Gauss-Legendre quadrature on :math:`[0, \sqrt b]`: a
    sum of positive terms, so small p-values keep their precision.
    """
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0 - 1e-12)
    p = float(chi2.sf(b, q))
    h = float(np.sqrt(b))
    if rho.size == 0 or not h > 0:
        return min(p, 1.0)
    sigma = np.sqrt(1.0 - rho**2)
    # Enough nodes to resolve the narrowest step's kernel.
    n_nodes = int(np.clip(np.ceil(6.0 * h / sigma.min()), 48, 800))
    nodes, weights = _gauss_legendre(n_nodes)
    r = 0.5 * h * (nodes + 1.0)
    w = 0.5 * h * weights
    # The density of |W_1| below h
    density = chi.pdf(r, q)
    for rho_k, sigma_k in zip(rho, sigma):
        mass = w * density
        p += float(mass @ _chi_tail(h, r, rho_k, sigma_k, q))
        density = mass @ _chi_transition(r, r, rho_k, sigma_k, q)
    return min(p, 1.0)


def max_statistic_p_value(b: float, m: NDArray, N: float, q: int) -> float:
    r"""The p-value of a maximally selected statistic.

    ``b`` is the largest of the standardised statistics
    :math:`Q_k \sim \chi^2_q` at the cuts leaving ``m`` (increasing) units
    of ``N`` on the left. Under the permutation null they are
    asymptotically a Gauss-Markov chain, with correlation
    :math:`\sqrt{m_{k-1} (N - m_k) / (m_k (N - m_{k-1}))}` between
    neighbours, and the p-value is :func:`max_chain_sf`.
    """
    m = np.asarray(m, dtype=float)
    if m.size == 0:
        return 1.0
    m1, m2 = m[:-1], m[1:]
    rho = np.sqrt(m1 * (N - m2) / (m2 * (N - m1)))
    return max_chain_sf(b, rho, q)


def max_statistic(
    scores: NDArray,
    n: NDArray,
    z: NDArray,
    failures: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
) -> tuple[float, float, int]:
    """The maximally selected standardised statistic of the scores over
    the admissible cuts of one feature ``z``, and its p-value.

    Returns ``(statistic, p_value, n_cuts)``; ``(0, 1, 0)`` when no cut
    is admissible or the scores are all equal.
    """
    n = np.asarray(n, dtype=float)
    N = n.sum()
    if N < 2:
        return 0.0, 1.0, 0
    centred = scores - (n @ scores) / N
    spread = centred.T @ (n[:, None] * centred)
    # The rank of the scores' covariance is the degrees of freedom; a
    # pseudo-inverse drops directions the scores do not vary in.
    eigenvalues, eigenvectors = np.linalg.eigh(spread)
    keep = eigenvalues > 1e-10 * max(float(eigenvalues.max()), 1e-300)
    q = int(keep.sum())
    if q == 0:
        return 0.0, 1.0, 0
    # Whitened scores: their covariance is the identity on its rank.
    whitened = centred @ (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))

    order = np.argsort(z, kind="stable")
    z_sorted = z[order]
    # Cut after each last occurrence of a value: the left child is every
    # row with the feature at most that value.
    cut = np.flatnonzero(z_sorted[:-1] != z_sorted[1:])
    if cut.size == 0:
        return 0.0, 1.0, 0
    n_rows = z.size
    rows_left = cut + 1.0
    failures_left = np.cumsum(failures[order])[cut]
    total_failures = failures.sum()
    ok = (
        (rows_left >= min_leaf_samples)
        & (n_rows - rows_left >= min_leaf_samples)
        & (failures_left >= min_leaf_failures)
        & (total_failures - failures_left >= min_leaf_failures)
    )
    admissible = np.flatnonzero(ok)
    if admissible.size == 0:
        return 0.0, 1.0, 0
    if admissible.size > _MAX_TEST_CUTS:
        spaced = np.linspace(0, admissible.size - 1, _MAX_TEST_CUTS)
        admissible = admissible[np.unique(np.round(spaced).astype(int))]
    cut = cut[admissible]
    m = np.cumsum(n[order])[cut]
    T = np.cumsum(n[order, None] * whitened[order], axis=0)[cut]
    Q = N * (N - 1.0) / (m * (N - m)) * np.sum(T**2, axis=1)
    b = float(Q.max())
    return b, max_statistic_p_value(b, m, N, q), int(m.size)


def ctree_select(
    data: SurpyvalData,
    Z: NDArray,
    kind: str,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
) -> tuple[int, float]:
    """The feature a conditional-inference node splits on, and its
    Bonferroni-adjusted p-value.

    Each feature in ``feature_indices_in`` gets the p-value of its
    maximally selected score statistic; the one with the smallest (the
    larger statistic breaking a tie) is returned with that p-value times
    the number of features tested, capped at 1. ``(-1, 1.0)`` if no
    feature has an admissible cut or the scores carry no information.
    """
    features = [int(u) for u in feature_indices_in]
    scores = node_scores(data, kind)
    if scores is None or not features:
        return -1, 1.0
    n = np.asarray(data.n, dtype=float)
    # The failures each child must keep, n-weighted as every split
    # counts them (#193).
    failures = (np.asarray(data.c) != 1) * n
    best_u, best_key = -1, (np.inf, np.inf)
    for u in features:
        statistic, p_value, n_cuts = max_statistic(
            scores, n, Z[:, u], failures, min_leaf_samples, min_leaf_failures
        )
        if n_cuts and (p_value, -statistic) < best_key:
            best_u, best_key = u, (p_value, -statistic)
    if best_u == -1:
        return -1, 1.0
    return best_u, min(1.0, best_key[0] * len(features))
