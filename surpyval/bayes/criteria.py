"""How well a Bayesian fit predicts data it has not seen: WAIC and
leave-one-out cross-validation (LOO), and a comparison of fits by them.

Both estimate the expected log predictive density of a new observation,
``elpd``, summed over the data's units (a row with count ``n`` is ``n``
units, each left out in turn). Higher is better. ``-2 elpd`` (``ic``) is
on the deviance scale of AIC. Unlike AIC they average over the posterior,
so they take the prior into account, and every row enters through its
own likelihood term: a censored row's survival or failure probability, an
interval's probability, a truncated row's term divided by the probability
of its window.

How the leave-one-out density ``p(y_i | y_-i) = 1 / E[1 / p(y_i | theta)]``
(the expectation over the full posterior) is computed depends on how the
posterior was:

- **Conjugate** fits: in closed form, exactly (the posterior without the
  row is the same family with the row's contribution taken out).
- **Grid** and **Laplace** fits: by the grid's own quadrature of that
  expectation. The points are not random, so there is nothing to smooth;
  what can go wrong is that the leave-one-out posterior reaches past the
  grid, which the grid was laid to hold for the full posterior only. Each
  row's ``edge``, the share of its leave-one-out posterior in the grid's
  outermost cells, says whether it does.
- **Draws** (``from_draws``): Pareto-smoothed importance sampling
  (Vehtari, Gelman and Gabry 2017; Vehtari, Simpson, Gelman, Yao and
  Gabry 2024), with each row's Pareto ``k``.

WAIC (Watanabe 2010) is ``lppd - p_waic``, with ``p_waic`` the posterior
variance of each row's log-likelihood (the variance form).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp, polygamma

from surpyval.bayes.likelihood import LogLikelihood
from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

#: The (point, row) evaluations of the log-likelihood held at once.
_CHUNK = 2_000_000
#: A row whose ``p_waic`` term is above this makes WAIC unreliable
#: (Vehtari, Gelman and Gabry 2017, section 2.2).
P_WAIC_LIMIT = 0.4
#: A row's leave-one-out posterior with more than this share of its mass
#: in the grid's outermost cells may reach past the grid.
EDGE_SHARE = 1e-3
#: The fewest draws Pareto smoothing takes (a tail of 5).
MIN_DRAWS = 25


def good_k(draws: int) -> float:
    """The largest Pareto ``k`` at which ``draws`` importance-sampling
    draws give a reliable estimate: ``min(1 - 1 / log10(S), 0.7)``
    (Vehtari et al. 2024, section 3.2.4).

    Examples
    --------
    >>> from surpyval.bayes.criteria import good_k
    >>> round(good_k(1000), 3), good_k(4000)
    (0.667, 0.7)
    """
    return float(min(1.0 - 1.0 / np.log10(draws), 0.7))


def gpd_fit(x: npt.NDArray) -> tuple[float, float]:
    """The generalised Pareto distribution's shape ``k`` and scale
    ``sigma`` for the exceedances ``x`` (sorted, ascending), by Zhang and
    Stephens' (2009) empirical Bayes estimate, with ``k`` shrunk towards
    0.5 by the weak prior of Vehtari et al. (2024, appendix C): ``(n k +
    10 * 0.5) / (n + 10)``. As R's ``loo`` and ArviZ have it.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.criteria import gpd_fit
    >>> u = (np.arange(1000) + 0.5) / 1000
    >>> k, sigma = gpd_fit(np.expm1(-0.5 * np.log1p(-u)) / 0.5)
    >>> round(k, 2), round(sigma, 2)
    (0.5, 1.0)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = 30 + int(np.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3.0 * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x[None, :]).mean(axis=1)
    profile = n * (np.log(-b / k) - k - 1.0)
    w = np.exp(profile - logsumexp(profile))
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep], b[keep]
    b_hat = float(np.sum(b * w) / w.sum())
    k_hat = float(np.log1p(-b_hat * x).mean())
    sigma = -k_hat / b_hat
    return (n * k_hat + 10 * 0.5) / (n + 10), sigma


def _gpd_quantile(p: npt.NDArray, k: float, sigma: float) -> npt.NDArray:
    if abs(k) < np.finfo(float).eps:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(log_ratios: npt.ArrayLike) -> tuple[npt.NDArray, float]:
    """
    Pareto-smoothed importance sampling of one set of log importance
    ratios: the normalised log weights and the Pareto ``k`` of the
    ratios' tail.

    The ``M = ceil(min(S / 5, 3 sqrt(S)))`` largest of the ``S`` ratios
    are replaced by the expected order statistics of a generalised Pareto
    distribution fitted to their excess over the next largest
    (:func:`gpd_fit`), and no weight is left above the largest raw one
    (Vehtari et al. 2024, with the draws taken as independent). With a
    tail of fewer than 5 ratios nothing is smoothed and ``k`` is
    infinite.

    Parameters
    ----------
    log_ratios : 1-D array like
        ``log`` of the importance ratios, one per draw.

    Returns
    -------
    (numpy.ndarray, float)
        The log weights (their exponentials sum to 1) and ``k``.

    Examples
    --------
    Ratios with a light tail have a small ``k``:

    >>> import numpy as np
    >>> from surpyval.bayes.criteria import psis
    >>> z = np.random.default_rng(1).normal(size=4000)
    >>> log_w, k = psis(0.3 * z)
    >>> bool(k < 0.5), bool(np.isclose(np.exp(log_w).sum(), 1.0))
    (True, True)
    """
    x = np.asarray(log_ratios, dtype=float).copy()
    S = len(x)
    x -= np.max(x)
    order = np.argsort(x, kind="stable")
    tail_len = int(np.ceil(min(S / 5.0, 3.0 * np.sqrt(S))))
    k = np.inf
    if S > tail_len:
        cutoff = max(x[order[-tail_len - 1]], np.log(np.finfo(float).tiny))
        tail = np.flatnonzero(x > cutoff)
        if len(tail) > 4:
            tail = tail[np.argsort(x[tail], kind="stable")]
            excess = np.exp(x[tail]) - np.exp(cutoff)
            k, sigma = gpd_fit(excess)
            if np.isfinite(k) and sigma > 0:
                p = (np.arange(len(tail)) + 0.5) / len(tail)
                with np.errstate(divide="ignore"):
                    x[tail] = np.log(
                        _gpd_quantile(p, k, sigma) + np.exp(cutoff)
                    )
                x[x > 0] = 0.0
            else:
                k = np.inf
    return x - logsumexp(x), float(k)


class PredictiveCriterion:
    """
    WAIC or leave-one-out cross-validation of a Bayesian fit: the
    expected log predictive density of a new observation, summed over the
    data's units, with its standard error and the effective number of
    parameters.

    Made by ``BayesianParametric.waic()`` and ``loo()``.

    Attributes
    ----------
    criterion : str
        ``"waic"`` or ``"loo"``.
    elpd : float
        The estimate, ``sum(n * pointwise)``; higher is better.
    se : float
        Its standard error, ``sqrt(N) * sd(pointwise)`` over the ``N``
        units (``sd`` with divisor ``N - 1``, as R's ``loo``).
    p : float
        The effective number of parameters: ``p_waic``, or ``p_loo``
        (``lppd - elpd``).
    lppd : float
        The log pointwise predictive density of the data themselves.
    pointwise : numpy.ndarray
        Each row's ``elpd`` for one of its units, in the order of the
        data's rows.
    n : numpy.ndarray
        Each row's count.
    method : str
        ``"closed form"`` (a conjugate fit), ``"quadrature"`` (a grid
        fit), ``"psis"`` (LOO from draws, including a Laplace fit's) or
        ``"draws"`` (WAIC from draws).
    pareto_k : numpy.ndarray or None
        LOO from draws: each row's Pareto ``k``; above ``good_k`` the row's
        estimate is unreliable.
    good_k : float or None
        The threshold for ``pareto_k`` at this number of draws.
    edge : numpy.ndarray or None
        LOO on a grid: the share of each row's leave-one-out posterior in
        the grid's outermost cells; above ``EDGE_SHARE`` (0.001) it may
        reach past the grid.
    p_pointwise : numpy.ndarray
        Each row's contribution to ``p`` for one unit (for WAIC, above 0.4
        the estimate is unreliable).
    flagged : numpy.ndarray
        The indices of the rows whose estimate is unreliable.

    Examples
    --------
    >>> import surpyval as sp
    >>> model = sp.Exponential.fit_bayes(
    ...     [120, 340, 560, 800, 1500], c=[0, 0, 0, 0, 1],
    ...     priors={"failure_rate": sp.Gamma.from_params([2.0, 500.0])})
    >>> res = model.loo()
    >>> res.method, round(res.elpd, 2), round(res.p, 2)
    ('closed form', -32.14, 1.23)
    """

    def __init__(
        self,
        criterion: str,
        pointwise: npt.NDArray,
        lppd: npt.NDArray,
        n: npt.NDArray,
        method: str,
        *,
        pareto_k: npt.NDArray | None = None,
        good_k: float | None = None,
        edge: npt.NDArray | None = None,
        p_pointwise: npt.NDArray | None = None,
    ) -> None:
        self.criterion = criterion
        self.pointwise = np.asarray(pointwise, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.method = method
        self.pareto_k = pareto_k
        self.good_k = good_k
        self.edge = edge
        lppd = np.asarray(lppd, dtype=float)
        self.lppd = float(self.n @ lppd)
        self.p_pointwise = (
            lppd - self.pointwise if p_pointwise is None else p_pointwise
        )
        self.elpd = float(self.n @ self.pointwise)
        self.p = float(self.n @ self.p_pointwise)
        self.se = _se(self.pointwise, self.n)
        if pareto_k is not None:
            assert good_k is not None
            bad = pareto_k > good_k
        elif edge is not None:
            bad = edge > EDGE_SHARE
        elif criterion == "waic":
            bad = self.p_pointwise > P_WAIC_LIMIT
        else:
            bad = np.zeros(len(self.n), dtype=bool)
        self.flagged = np.flatnonzero(bad)

    @property
    def ic(self) -> float:
        """``-2 * elpd``, on the scale of AIC (lower is better)."""
        return -2.0 * self.elpd

    @property
    def units(self) -> float:
        """The number of units, ``sum(n)``."""
        return float(self.n.sum())

    def __repr__(self) -> str:
        name = {"waic": "WAIC", "loo": "LOO"}[self.criterion]
        lines = [
            f"{name} ({self.method}), {self.units:g} units",
            f"  elpd_{self.criterion}: {self.elpd:.2f} (se {self.se:.2f})",
            f"  p_{self.criterion}: {self.p:.2f}",
            f"  {'waic' if self.criterion == 'waic' else 'looic'}: "
            f"{self.ic:.2f}",
        ]
        if self.pareto_k is not None:
            bad = len(self.flagged)
            lines.append(
                f"  Pareto k above {self.good_k:.2f}: {bad} of "
                f"{len(self.n)} rows"
            )
        elif self.edge is not None:
            lines.append(
                f"  leave-one-out posterior at the grid's edge: "
                f"{len(self.flagged)} of {len(self.n)} rows"
            )
        return "\n".join(lines)


def _se(values: npt.NDArray, n: npt.NDArray) -> float:
    """``sqrt(N) * sd`` of ``values`` over the ``N = sum(n)`` units, with
    divisor ``N - 1``."""
    N = float(n.sum())
    if N < 2:
        return float("nan")
    mean = float(n @ values) / N
    var = float(n @ (values - mean) ** 2) / (N - 1)
    return float(np.sqrt(N * var))


# ----------------------------------------------------------------------
# The pointwise log-likelihood, a few rows at a time
# ----------------------------------------------------------------------
def _require_data(model: Any, what: str) -> None:
    if model.data is None:
        raise ValueError(
            f"{what} needs the data the posterior came from, and this "
            "model has none: give data= to from_draws (or fit with "
            "fit_bayes)."
        )


def _points(model: Any) -> tuple[npt.NDArray, npt.NDArray]:
    """The posterior's points with positive weight, and their log
    weights."""
    post = model.posterior
    keep = post.weights > 0
    return post.points[keep], np.log(post.weights[keep])


def _row_blocks(model: Any, points: npt.NDArray) -> Any:
    """``(rows, ll)`` for blocks of the data's rows: their indices and
    the log-likelihood of each at every point, ``(points, rows)``."""
    ll = LogLikelihood(
        model.dist, model.data, model.offset, model.lfp, model.zi
    )
    R = len(model.data.c)
    step = max(1, _CHUNK // max(len(points), 1))
    for start in range(0, R, step):
        rows = np.zeros(R, dtype=bool)
        rows[start : start + step] = True
        yield np.flatnonzero(rows), ll._pointwise(points, rows)


def _weighted_moments(
    ll: npt.NDArray, w: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    with np.errstate(invalid="ignore"):
        mean = w @ ll
        var = w @ (ll - mean[None, :]) ** 2
    return mean, var


# ----------------------------------------------------------------------
# Closed forms for the conjugate fits
# ----------------------------------------------------------------------
def _conjugate(model: Any) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``(lppd, p_waic, elpd_loo)`` of each row, for one unit, in closed
    form."""
    exact = model.posterior.exact
    A, B = (float(v) for v in exact["params"])
    d = model.data
    x = np.asarray(d.x, dtype=float)
    x = x if x.ndim == 1 else x[:, 0]
    family = exact["family"]
    if family == "gamma":
        tl = np.asarray(d.t, dtype=float)[:, 0]
        y = x - np.where(np.isfinite(tl), np.maximum(tl, 0.0), 0.0)
        failed = np.asarray(d.c) == 0
        lppd = np.where(
            failed,
            np.log(A) + A * np.log(B) - (A + 1) * np.log(B + y),
            A * (np.log(B) - np.log(B + y)),
        )
        var = A * y**2 / B**2 + np.where(
            failed, polygamma(1, A) - 2 * y / B, 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            loo = np.where(
                failed,
                np.log(A - 1) + (A - 1) * np.log(B - y) - A * np.log(B),
                A * (np.log(B - y) - np.log(B)),
            )
        return lppd, var, loo
    if family == "normal":
        sigma = float(model.fixed["sigma"])
        s2, v = B**2, sigma**2
        lppd = _log_normal(x, A, v + s2)
        var = (s2**2 + 2 * (x - A) ** 2 * s2) / (2 * v**2)
        prec = 1.0 / s2 - 1.0 / v
        mean = (A / s2 - x / v) / prec
        return lppd, var, _log_normal(x, mean, v + 1.0 / prec)
    # beta: a Bernoulli's p
    one = x == 1
    lppd = np.where(one, np.log(A), np.log(B)) - np.log(A + B)
    var = np.where(one, polygamma(1, A), polygamma(1, B)) - polygamma(1, A + B)
    with np.errstate(divide="ignore"):
        loo = np.where(one, np.log(A - 1), np.log(B - 1)) - np.log(A + B - 1)
    return lppd, var, loo


def _log_normal(x: npt.NDArray, mean: Any, var: Any) -> npt.NDArray:
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


# ----------------------------------------------------------------------
# WAIC and LOO
# ----------------------------------------------------------------------
def waic(model: Any) -> PredictiveCriterion:
    """WAIC of a Bayesian fit; see ``BayesianParametric.waic``."""
    _require_data(model, "waic()")
    n = np.asarray(model.data.n, dtype=float)
    if model.method == "conjugate" and model.posterior.exact is not None:
        lppd, var, _ = _conjugate(model)
        method = "closed form"
    else:
        points, log_w = _points(model)
        w = np.exp(log_w)
        lppd = np.empty(len(n))
        var = np.empty(len(n))
        for rows, ll in _row_blocks(model, points):
            lppd[rows] = logsumexp(log_w[:, None] + ll, axis=0)
            var[rows] = _weighted_moments(ll, w)[1]
        method = (
            "draws"
            if model.posterior.kind in ("draws", "laplace")
            else "quadrature"
        )
    res = PredictiveCriterion(
        "waic", lppd - var, lppd, n, method, p_pointwise=var
    )
    if res.flagged.size:
        warnings.warn(
            f"WAIC may be unreliable: {_count(res)} a p_waic term "
            f"above {P_WAIC_LIMIT} (the posterior variance of their "
            "log-likelihood). Use loo(), whose diagnostics say which "
            "estimates to trust.",
            stacklevel=caller_stacklevel(),
        )
    return res


def loo(model: Any) -> PredictiveCriterion:
    """Leave-one-out cross-validation of a Bayesian fit; see
    ``BayesianParametric.loo``."""
    _require_data(model, "loo()")
    n = np.asarray(model.data.n, dtype=float)
    post = model.posterior
    if model.method == "conjugate" and post.exact is not None:
        lppd, _, elpd = _conjugate(model)
        return PredictiveCriterion("loo", elpd, lppd, n, "closed form")
    points, log_w = _points(model)
    lppd = np.empty(len(n))
    elpd = np.empty(len(n))
    if post.kind not in ("draws", "laplace"):
        faces = _faces(post)
        edge = None if faces is None else np.empty(len(n))
        for rows, ll in _row_blocks(model, points):
            lppd[rows] = logsumexp(log_w[:, None] + ll, axis=0)
            with np.errstate(invalid="ignore"):
                tilted = log_w[:, None] - ll
                total = logsumexp(tilted, axis=0)
            elpd[rows] = -total
            if edge is not None and faces is not None:
                edge[rows] = np.exp(logsumexp(tilted[faces], axis=0) - total)
        res = PredictiveCriterion(
            "loo", elpd, lppd, n, "quadrature", edge=edge
        )
        if res.flagged.size:
            warnings.warn(
                f"{_count(res)} a leave-one-out posterior that reaches the "
                f"edge of the grid (more than {EDGE_SHARE:g} of its mass in "
                "the outermost cells; see edge), so their leave-one-out "
                "values may be too high: without them the data say much "
                "less about the parameters. Check those rows, give the "
                "parameters a prior that holds them, or compare by draws "
                "from a sampler (from_draws), whose Pareto k says how far "
                "to trust each row.",
                stacklevel=caller_stacklevel(),
            )
        return res
    S = len(points)
    if S < MIN_DRAWS:
        raise ValueError(
            f"loo() smooths the tail of the importance ratios and needs at "
            f"least {MIN_DRAWS} draws; the posterior has {S}. Give "
            "from_draws more draws (a few thousand is usual)."
        )
    k = np.empty(len(n))
    for rows, ll in _row_blocks(model, points):
        lppd[rows] = logsumexp(log_w[:, None] + ll, axis=0)
        for j, row in enumerate(rows):
            with np.errstate(invalid="ignore"):
                lw, k[row] = psis(log_w - ll[:, j])
                elpd[row] = logsumexp(lw + ll[:, j])
    threshold = good_k(S)
    res = PredictiveCriterion(
        "loo", elpd, lppd, n, "psis", pareto_k=k, good_k=threshold
    )
    if res.flagged.size:
        warnings.warn(
            f"{_count(res)} a Pareto k above {threshold:.2f} (of "
            f"{S} draws; see pareto_k), so their leave-one-out values are "
            "unreliable: those rows change the posterior a lot (often "
            "outliers). Check them, refit without each to find its value "
            "exactly, or, for k up to 0.7, give more draws.",
            stacklevel=caller_stacklevel(),
        )
    return res


def _count(res: PredictiveCriterion) -> str:
    """The flagged units out of all, with the distinct rows where they
    differ, and the verb: "2 of the 21 observations (1 distinct row)
    have"."""
    rows = len(res.flagged)
    units = float(res.n[res.flagged].sum())
    of = f"{units:g} of the {res.units:g} observations"
    if units != rows:
        of += f" ({rows} distinct row{'' if rows == 1 else 's'})"
    return of + (" has" if units == 1 else " have")


def _faces(post: Any) -> npt.NDArray | None:
    """A mask of a grid's points in its outermost cells (``None`` where
    the layout is not known)."""
    shape = post.shape
    keep = post.weights > 0
    if shape is None or int(np.prod(shape)) != len(post.weights):
        return None
    idx = np.unravel_index(np.arange(int(np.prod(shape))), shape)
    out = np.zeros(len(post.weights), dtype=bool)
    for j, m in enumerate(shape[1:], start=1):
        out |= (idx[j] == 0) | (idx[j] == m - 1)
    return out[keep]


# ----------------------------------------------------------------------
# Comparing fits
# ----------------------------------------------------------------------
class PredictiveComparison:
    """
    Bayesian fits ranked by their expected log predictive density (WAIC
    or LOO), each with its difference from the best and the standard
    error of that difference.

    Made by :func:`surpyval.bayes.compare`.

    Attributes
    ----------
    criterion : str
        ``"loo"`` or ``"waic"``.
    table : dict
        By name, best first: ``elpd``, ``se``, ``p``, ``elpd_diff`` (this
        fit's ``elpd`` less the best's, 0 for the best), ``se_diff`` (the
        standard error of that difference, from the rows' paired
        differences; ``nan`` for the best) and ``flagged`` (the number of
        rows whose estimate is unreliable).
    results : dict
        By name, each fit's :class:`PredictiveCriterion`.
    best : str
        The name of the fit with the highest ``elpd``.

    Examples
    --------
    >>> import surpyval as sp
    >>> x = sp.Weibull.random(40, 1000, 2.5, random_state=1).round()
    >>> fits = {
    ...     "Weibull": sp.Weibull.fit_bayes(
    ...         x, priors={"beta": sp.prior.around(2, between=(1, 4))}),
    ...     "Exponential": sp.Exponential.fit_bayes(
    ...         x, priors={"failure_rate": sp.prior.between(1e-4, 1e-2)}),
    ... }
    >>> comparison = sp.bayes.compare(fits)
    >>> comparison.best
    'Weibull'
    >>> row = comparison.table["Exponential"]
    >>> round(row["elpd_diff"], 1), round(row["se_diff"], 1)
    (-17.6, 3.3)
    """

    def __init__(
        self, criterion: str, results: dict, table: dict, best: str
    ) -> None:
        self.criterion = criterion
        self.results = results
        self.table = table
        self.best = best

    def __repr__(self) -> str:
        width = max(len(str(k)) for k in self.table)
        head = (
            f"{'':{width}}  {'elpd':>10} {'se':>8} {'p':>7} "
            f"{'elpd_diff':>10} {'se_diff':>8}"
        )
        lines = [f"Comparison by {self.criterion.upper()}", head]
        for name, r in self.table.items():
            lines.append(
                f"{str(name):{width}}  {r['elpd']:10.2f} {r['se']:8.2f} "
                f"{r['p']:7.2f} {r['elpd_diff']:10.2f} {r['se_diff']:8.2f}"
                + (
                    f"  ({r['flagged']} rows unreliable)"
                    if r["flagged"]
                    else ""
                )
            )
        return "\n".join(lines)


def _same_data(a: Any, b: Any) -> bool:
    for name in ("x", "c", "n", "t"):
        u = np.asarray(getattr(a, name))
        v = np.asarray(getattr(b, name))
        if u.shape != v.shape or not np.array_equal(u, v):
            return False
    return True


def compare(models: Any, criterion: str = "loo") -> PredictiveComparison:
    """
    Bayesian fits to the same data, ranked by how well each predicts an
    observation it has not seen: the expected log predictive density by
    leave-one-out cross-validation (``"loo"``, the default) or WAIC.

    The difference between two fits is summed over the units, and its
    standard error comes from the units' paired differences, which is
    far smaller than the two fits' own standard errors combined when the
    fits are alike. A difference within about two of its standard errors
    does not tell the fits apart.

    Parameters
    ----------
    models : dict or list
        The fits (``BayesianParametric``), by name; a list is named by
        each fit's distribution (numbered where two share one).
    criterion : str, optional
        ``"loo"`` (the default) or ``"waic"``.

    Returns
    -------
    PredictiveComparison
        ``table`` (best first), ``best`` and each fit's ``results``.

    Examples
    --------
    >>> import surpyval as sp
    >>> x = sp.Weibull.random(40, 1000, 2.5, random_state=1).round()
    >>> weibull = sp.Weibull.fit_bayes(
    ...     x, priors={"beta": sp.prior.around(2, between=(1, 4))})
    >>> exponential = sp.Exponential.fit_bayes(
    ...     x, priors={"failure_rate": sp.prior.between(1e-4, 1e-2)})
    >>> sp.bayes.compare([weibull, exponential])
    Comparison by LOO
                       elpd       se       p  elpd_diff  se_diff
    Weibull         -295.03     3.68    1.56       0.00      nan
    Exponential     -312.58     2.72    0.19     -17.55     3.25
    """
    from surpyval.bayes.model import BayesianParametric

    if criterion not in ("loo", "waic"):
        raise ValueError(
            f"criterion must be 'loo' or 'waic'; got {criterion!r}."
        )
    if isinstance(models, dict):
        named = dict(models)
    else:
        named = {}
        fits = list(models)
        names = [getattr(m.dist, "name", type(m).__name__) for m in fits]
        for i, (name, m) in enumerate(zip(names, fits)):
            if names.count(name) > 1:
                name = f"{name} {names[: i + 1].count(name)}"
            named[name] = m
    if len(named) < 2:
        raise ValueError(
            "compare needs at least two fits; for one fit's value use its "
            f"{criterion}()."
        )
    for name, m in named.items():
        if not isinstance(m, BayesianParametric):
            raise ValueError(
                f"{name!r} is a {type(m).__name__}, not a Bayesian fit: "
                "WAIC and LOO average over a posterior. Compare "
                "maximum-likelihood fits by aic()."
            )
        _require_data(m, "compare")
    first_name, first = next(iter(named.items()))
    for name, m in named.items():
        if not _same_data(first.data, m.data):
            raise ValueError(
                f"{name!r} and {first_name!r} were fitted to different "
                "data; their predictive densities are of different "
                "observations, so they do not compare. Fit every model "
                "to the same data."
            )
    results = {}
    for name, m in named.items():
        with collecting_warnings() as caught:
            results[name] = m.loo() if criterion == "loo" else m.waic()
        for w in caught:
            warnings.warn(
                f"{name!r}: {w.message}",
                w.category,
                stacklevel=caller_stacklevel(),
            )
    ranked = sorted(results, key=lambda k: -results[k].elpd)
    best = ranked[0]
    top = results[best]
    table = {}
    for name in ranked:
        r = results[name]
        diff = r.pointwise - top.pointwise
        table[name] = {
            "elpd": r.elpd,
            "se": r.se,
            "p": r.p,
            "elpd_diff": float(r.n @ diff),
            "se_diff": float("nan") if name == best else _se(diff, r.n),
            "flagged": int(len(r.flagged)),
        }
    return PredictiveComparison(criterion, results, table, best)


__all__ = [
    "PredictiveComparison",
    "PredictiveCriterion",
    "compare",
    "good_k",
    "gpd_fit",
    "loo",
    "psis",
    "waic",
]
