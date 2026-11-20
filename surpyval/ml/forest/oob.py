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

With the cause of each failure (a forest grown with ``e``) the likelihood
is the competing-risks one: a failure from cause :math:`k` at :math:`x`
contributes the cause's sub-density :math:`f_k(x) = d\,F_k(x) / dx`, the
rate at which the cumulative incidence :math:`F_k` of cause :math:`k`
rises there, and a censored row the all-cause survival :math:`S(x)` (with
delayed entry, each divided by :math:`S(t_l)`). An Aalen-Johansen leaf is
read continuously as above: each cause's incidence is joined linearly
between the all-cause drops (the joins add up to the all-cause curve), and
past the last drop the exponential tail is shared between the causes in
the proportions of their incidences there.

Restricted to one cause :math:`k`, the likelihood is the cause-specific
one: a failure from :math:`k` at :math:`x` contributes the cause's net
density :math:`f^{net}_k(x) = h_k(x) S_k(x)`, and every other row (a
censoring, or a failure from another cause) the cause's net survival
:math:`S_k(x) = \exp(-H_k(x))`, as if that row were censored for cause
:math:`k`; delayed entry divides by :math:`S_k(t_l)`. It judges the
predictions of the cause's own hazard only. For one parametric leaf the
causes' restricted likelihoods multiply to the full one
(:math:`\prod_k h_k^{\delta_k} S_k = h_{cause} S`). An Aalen-Johansen
leaf's net survival of a cause (its Kaplan-Meier curve with the other
causes censored) is read as a plain step leaf is.

References
----------
Breiman, L., 2001. Random forests. *Machine Learning*, 45(1), pp.5-32.

Brown, B.W., Hollander, M. and Korwar, R.M., 1974. Nonparametric tests of
independence for censored data, with applications to heart transplant
studies. In *Reliability and Biometry*, pp.327-354. SIAM.

Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer, M.S., 2008.
Random survival forests. *Annals of Applied Statistics*, 2(3),
pp.841-860.

Ishwaran, H., Gerds, T.A., Kogalur, U.B., Moore, R.D., Gange, S.J. and
Lau, B.M., 2014. Random survival forests for competing risks.
*Biostatistics*, 15(4), pp.757-773.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from surpyval.ml.forest.cause_specific_split import cause_codes
from surpyval.ml.forest.node import Node, route_to_leaves
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

    def __init__(
        self,
        model: Any,
        origin: float,
        steps: tuple[NDArray, NDArray] | None = None,
    ) -> None:
        # ``steps``, given, is a step function's ``(times, R)`` read in
        # place of a model's (a cause's net survival in an Aalen-Johansen
        # leaf).
        self.model = model
        self.step = steps is not None or isinstance(model, NonParametric)
        if not self.step:
            return
        if steps is None:
            steps = (model.x, model.R)
        R = np.asarray(steps[1], dtype=float)
        times = np.asarray(steps[0], dtype=float)
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


class _CauseLeafCurve:
    """A competing-risks leaf as the likelihood reads it: the all-cause
    survival, each cause's sub-density and each cause's net survival and
    density (a :class:`_LeafCurve` of the cause's own model).

    For a parametric leaf these are the model's own (``iif`` is the
    sub-density). For an Aalen-Johansen leaf they are the continuous
    reading of the step curves (the module docstring): each cause's
    incidence joined linearly between the all-cause drops, the exponential
    tail shared in the proportions of the incidences at the last drop, and
    each cause's net Kaplan-Meier curve read as a plain step leaf is. A
    cause absent from the leaf has zero incidence, a net survival of 1 and
    no density; a leaf with no failures has ``sf`` 1 (0 at plus infinity,
    as for :class:`_LeafCurve`).
    """

    def __init__(self, leaf: Any, origin: float) -> None:
        self.model = leaf.model
        self.origin = origin
        self.present = list(leaf.present) if self.model is not None else []
        self.step = self.model is not None and not hasattr(
            self.model, "models"
        )
        self._net: dict = {}
        if not self.step:
            return
        model = self.model
        times = np.asarray(model.x, dtype=float)
        S = np.asarray(model.sf(times), dtype=float)
        self._times = times
        previous = np.concatenate([[1.0], S[:-1]])
        dropped = S < previous
        drops = dropped & (times > origin)
        self.knots_x = np.concatenate([[origin], times[drops]])
        # Each cause's incidence at the knots; at the origin, what has
        # happened by then (a drop at or before the origin is a point mass
        # there, as for _LeafCurve).
        early = np.flatnonzero(dropped & (times <= origin))
        self.knots_F = {}
        for k in self.present:
            at = np.asarray(model.cif(times[drops], k), dtype=float)
            first = (
                float(model.cif(times[early[-1]], k)) if early.size else 0.0
            )
            self.knots_F[k] = np.concatenate([[first], at])
        self.knots_S: NDArray = 1.0 - np.sum(
            [self.knots_F[k] for k in self.present], axis=0
        )
        last_x, last_S = self.knots_x[-1], self.knots_S[-1]
        self.tail_rate: float | None = None
        self.share: dict = {}
        if last_x > origin and 0.0 < last_S < 1.0:
            self.tail_rate = -np.log(last_S) / (last_x - origin)
            failed = 1.0 - last_S
            self.share = {
                k: self.knots_F[k][-1] / failed for k in self.present
            }

    def sf(self, x: NDArray) -> NDArray:
        """The all-cause survival at ``x``."""
        x = np.asarray(x, dtype=float)
        if self.model is None:
            values = np.ones(x.shape)
        elif not self.step:
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

    def sub_df(self, x: NDArray, k: Any) -> NDArray:
        """Cause ``k``'s sub-density, the slope of its cumulative
        incidence, at ``x``."""
        x = np.asarray(x, dtype=float)
        if k not in self.present:
            return np.zeros(x.shape)
        if not self.step:
            values = np.asarray(self.model.iif(x, k), dtype=float)
            return np.broadcast_to(values, x.shape).copy()
        kx, F = self.knots_x, self.knots_F[k]
        values = np.zeros(x.shape)
        j = np.searchsorted(kx, x, side="left")
        inside = (j > 0) & (j < kx.size)
        if inside.any():
            jj = j[inside]
            values[inside] = (F[jj] - F[jj - 1]) / (kx[jj] - kx[jj - 1])
        beyond = x > kx[-1]
        if self.tail_rate is not None and beyond.any():
            values[beyond] = (
                self.share[k] * self.tail_rate * self.sf(x[beyond])
            )
        return values

    def net(self, k: Any) -> "_LeafCurve | None":
        """Cause ``k``'s net survival and density, as a
        :class:`_LeafCurve`; ``None`` for a cause absent from the leaf
        (survival 1, no density)."""
        if k not in self.present:
            return None
        if k not in self._net:
            if self.step:
                R = np.asarray(self.model.sf(self._times, event=k), float)
                self._net[k] = _LeafCurve(
                    None, self.origin, steps=(self._times, R)
                )
            else:
                self._net[k] = _LeafCurve(self.model.models[k], self.origin)
        return self._net[k]


def time_origin(data: SurpyvalData) -> float:
    """The time a step-function leaf's continuous reading starts from: 0,
    or the smallest finite time in the data if that is negative."""
    values = np.concatenate([np.ravel(data.x), np.ravel(data.t)])
    values = values[np.isfinite(values)]
    return float(min(0.0, values.min())) if values.size else 0.0


class RowTerms:
    """What the likelihood needs of each row, computed once: the region
    its event is known to lie in (cut to its truncation window), whether
    it was observed exactly, its truncation window and its count; with
    ``causes`` (a forest grown with ``e``), each row's cause as its
    position in ``causes`` (-1 for a censored row) in ``cause``."""

    def __init__(self, data: SurpyvalData, causes: list | None = None) -> None:
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
        self.causes = causes
        self.cause: NDArray | None = None
        if causes is not None:
            codes = cause_codes(data.e, causes)
            self.cause = np.where(self.exact, codes, -1)

    def __len__(self) -> int:
        return self.n.size


def add_tree_terms(
    root: Node,
    Z: NDArray,
    rows: NDArray,
    terms: RowTerms,
    curves: dict[int, Any],
    origin: float,
    numerator: NDArray,
    denominator: NDArray,
    event: int | None = None,
) -> None:
    """Add one tree's likelihood numerator and truncation denominator for
    each of ``rows`` (whose covariates are ``Z``, one row each) to the
    running sums. ``curves`` caches each leaf's curve across calls. With
    causes (``terms.cause``), the competing-risks likelihood, restricted
    to the cause at position ``event`` of ``terms.causes`` if given."""
    for leaf, idx in route_to_leaves(root, Z):
        key = id(leaf)
        if key not in curves:
            if terms.cause is not None:
                curves[key] = _CauseLeafCurve(leaf, origin)
            else:
                curves[key] = _LeafCurve(leaf.model, origin)
        curve = curves[key]
        r = rows[idx]
        if terms.cause is not None:
            _add_cause_terms(curve, r, terms, event, numerator, denominator)
            continue
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


def _add_cause_terms(
    curve: _CauseLeafCurve,
    r: NDArray,
    terms: RowTerms,
    event: int | None,
    numerator: NDArray,
    denominator: NDArray,
) -> None:
    """One leaf's competing-risks likelihood terms for the rows ``r``
    (exact or right-censored, with delayed entry; see the module
    docstring)."""
    assert terms.cause is not None and terms.causes is not None
    causes = terms.causes
    cause = terms.cause[r]
    x = terms.x[r]
    tl = terms.tl[r]
    num = np.empty(r.size)
    if event is None:
        S_tl = curve.sf(tl)
        censored = cause < 0
        if censored.any():
            num[censored] = curve.sf(terms.lo[r][censored])
        for k in np.unique(cause[~censored]):
            mine = cause == k
            num[mine] = curve.sub_df(x[mine], causes[k])
        numerator[r] += num
        denominator[r] += S_tl
        return
    # Restricted to one cause: its net (cause-specific) likelihood.
    net = curve.net(causes[event])
    if net is None:
        # The cause never happens here: survival 1, no density
        numerator[r] += np.where(cause == event, 0.0, 1.0)
        denominator[r] += 1.0
        return
    mine = cause == event
    if mine.any():
        num[mine] = net.df(x[mine])
    if not mine.all():
        num[~mine] = net.sf(terms.lo[r][~mine])
    numerator[r] += num
    denominator[r] += net.sf(tl)


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
