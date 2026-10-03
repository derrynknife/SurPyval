r"""Out-of-bag log-likelihood and permutation importance for the forest.

Each tree of a bootstrapped forest is grown without about a third of the
rows (``(1 - 1/N)^N`` is about ``e^{-1}``). For every row, the trees that
left it out form an ensemble that never saw it, and the row's likelihood
under that ensemble is an honest measure of how well the forest predicts
new data. Because it is a likelihood it works for the whole data model:

- observed at :math:`x`: the density :math:`f(x)`;
- right censored at :math:`x`: :math:`S(x)`;
- left censored at :math:`x`: :math:`1 - S(x)`;
- interval censored in :math:`(x_l, x_r]`: :math:`S(x_l) - S(x_r)`;

each divided by :math:`S(t_l) - S(t_r)` for a row truncated to
:math:`(t_l, t_r]`. A censored row's interval is first cut to its
truncation window, as the fitters' likelihoods do (a right-censored row
truncated at :math:`t_r` contributes :math:`S(x) - S(t_r)`). Here
:math:`S` and :math:`f` are the averages over the out-of-bag trees of the
leaf models' survival functions and densities.

A parametric leaf (``kind="weibull"`` or ``"exponential"``) has a density.
A non-parametric leaf is a step function whose drops are at the event
times of the tree's own bootstrap sample, so the probability it puts
exactly at an out-of-bag event time is almost always zero. For the
likelihood such a leaf is read as a continuous distribution instead: its
survival curve is joined linearly between the points where it drops,
starting from 1 at the time origin, so each step's probability is spread
evenly over the gap since the previous event time, and it is continued
past its last drop with the constant hazard it averaged up to there (the
exponential tail completion of Brown, Hollander and Korwar, 1974). The
density is then positive wherever the leaf still has mass, and it is a
density per unit of time, on the same scale as a parametric leaf's.

References
----------
Breiman, L., 2001. Random forests. *Machine Learning*, 45(1), pp.5-32.

Brown, B.W., Hollander, M. and Korwar, R.M., 1974. Nonparametric tests of
independence for censored data, with applications to heart transplant
studies. In *Reliability and Biometry*, pp.327-354. SIAM.

Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer, M.S., 2008.
Random survival forests. *Annals of Applied Statistics*, 2(3),
pp.841-860.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from surpyval.beta.ml.forest.node import Node, route_to_leaves
from surpyval.univariate.nonparametric.nonparametric import NonParametric
from surpyval.utils.surpyval_data import SurpyvalData


class _LeafCurve:
    """A leaf model's survival function and density, as the likelihood
    reads them: the model's own for a parametric leaf, and the continuous
    reading described in the module docstring for a step-function leaf.

    ``sf`` is 1 at minus infinity and 0 at plus infinity whatever the
    model (a leaf with no failures has ``sf`` 1 everywhere, and the
    probability it leaves at infinity counts as "after" every finite
    time, so ``sf(inf)`` is 0 for it too).
    """

    def __init__(self, model: Any, origin: float) -> None:
        self.model = model
        self.step = isinstance(model, NonParametric)
        if not self.step:
            return
        R = np.asarray(model.R, dtype=float)
        times = np.asarray(model.x, dtype=float)
        # The points where the step function drops: its event times (a
        # Turnbull estimate lists the start and end of each step; the end
        # is where it has dropped).
        previous = np.concatenate([[1.0], R[:-1]])
        drops = (R < previous) & (times > origin)
        self.knots_x = np.concatenate([[origin], times[drops]])
        self.knots_S = np.concatenate([[1.0], R[drops]])
        # A drop at (or before) the origin is a point mass there: the
        # curve starts from its value rather than from 1.
        if ((R < previous) & (times <= origin)).any():
            first = np.flatnonzero((R < previous) & (times <= origin))[-1]
            self.knots_S[0] = R[first]
        self.origin = origin
        last_x, last_S = self.knots_x[-1], self.knots_S[-1]
        # The tail's constant hazard: the average hazard up to the last
        # drop, -log S / (time since the origin). None where there is no
        # tail to continue (no drop, or the curve has reached zero).
        self.tail_rate: float | None = None
        if last_x > origin and 0.0 < last_S < 1.0:
            self.tail_rate = -np.log(last_S) / (last_x - origin)

    def sf(self, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        if not self.step:
            values = np.asarray(self.model.sf(x), dtype=float)
            values = np.broadcast_to(values, x.shape).copy()
        else:
            values = np.interp(x, self.knots_x, self.knots_S, left=1.0)
            beyond = x > self.knots_x[-1]
            if self.tail_rate is not None and beyond.any():
                values[beyond] = self.knots_S[-1] * np.exp(
                    -self.tail_rate * (x[beyond] - self.knots_x[-1])
                )
        values[np.isneginf(x)] = 1.0
        values[np.isposinf(x)] = 0.0
        return values

    def df(self, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        if not self.step:
            values = np.asarray(self.model.df(x), dtype=float)
            return np.broadcast_to(values, x.shape).copy()
        kx, kS = self.knots_x, self.knots_S
        values = np.zeros(x.shape)
        # A time in (kx[j - 1], kx[j]] takes the slope of that segment:
        # the probability of the step at kx[j] spread over the gap.
        j = np.searchsorted(kx, x, side="left")
        inside = (j > 0) & (j < kx.size)
        if inside.any():
            jj = j[inside]
            values[inside] = (kS[jj - 1] - kS[jj]) / (kx[jj] - kx[jj - 1])
        beyond = x > kx[-1]
        if self.tail_rate is not None and beyond.any():
            values[beyond] = self.tail_rate * self.sf(x[beyond])
        return values


def time_origin(data: SurpyvalData) -> float:
    """The time a step-function leaf's continuous reading starts from: 0,
    or the smallest finite time in the data if that is negative."""
    values = np.concatenate([np.ravel(data.x), np.ravel(data.t)])
    values = values[np.isfinite(values)]
    return float(min(0.0, values.min())) if values.size else 0.0


class RowTerms:
    """What the likelihood needs of each row, computed once: the region
    its event is known to lie in (cut to its truncation window), whether
    it was observed exactly, its truncation window and its count."""

    def __init__(self, data: SurpyvalData) -> None:
        x = np.asarray(data.x, dtype=float)
        c = np.asarray(data.c)
        tl = np.asarray(data.t[:, 0], dtype=float)
        tr = np.asarray(data.t[:, 1], dtype=float)
        left_x = x if x.ndim == 1 else x[:, 0]
        right_x = x if x.ndim == 1 else x[:, -1]
        self.exact = c == 0
        self.x = left_x
        lo = np.where(c == -1, -np.inf, left_x)
        hi = np.where(c == 1, np.inf, right_x)
        self.lo = np.maximum(lo, tl)
        self.hi = np.minimum(hi, tr)
        self.tl = tl
        self.tr = tr
        self.n = np.asarray(data.n, dtype=float)

    def __len__(self) -> int:
        return self.n.size


def add_tree_terms(
    root: Node,
    Z: NDArray,
    rows: NDArray,
    terms: RowTerms,
    curves: dict[int, _LeafCurve],
    origin: float,
    numerator: NDArray,
    denominator: NDArray,
) -> None:
    """Add one tree's likelihood numerator and truncation denominator for
    each of ``rows`` (whose covariates are ``Z``, one row each) to the
    running sums. ``curves`` caches each leaf's curve across calls."""
    for leaf, idx in route_to_leaves(root, Z):
        key = id(leaf)
        if key not in curves:
            curves[key] = _LeafCurve(leaf.model, origin)
        curve = curves[key]
        r = rows[idx]
        exact = terms.exact[r]
        num = np.empty(r.size)
        if exact.any():
            num[exact] = curve.df(terms.x[r][exact])
        if not exact.all():
            censored = r[~exact]
            num[~exact] = curve.sf(terms.lo[censored]) - curve.sf(
                terms.hi[censored]
            )
        numerator[r] += num
        denominator[r] += curve.sf(terms.tl[r]) - curve.sf(terms.tr[r])


def row_log_likelihood(
    numerator: NDArray, denominator: NDArray, n_oob: NDArray
) -> NDArray:
    """Each row's log-likelihood from its summed terms: the averages over
    the out-of-bag trees share the divisor ``n_oob``, which cancels. NaN
    where no tree left the row out."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(np.maximum(numerator, 0.0)) - np.log(denominator)
    ll[n_oob == 0] = np.nan
    return ll


def weighted_mean(ll: NDArray, n: NDArray) -> float:
    """The count-weighted mean of the rows' log-likelihoods, over the rows
    that have one; NaN if none does."""
    has = ~np.isnan(ll)
    if not has.any():
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.sum(n[has] * ll[has]) / np.sum(n[has]))
