"""Vine (pair-copula) constructions: a copula of ``D`` series built from
bivariate copulas.

A vine writes a ``D``-dimensional copula density as a product of
``D (D - 1) / 2`` bivariate copula densities arranged in ``D - 1`` trees
(Bedford and Cooke 2002; Aas, Czado, Frigessi and Bakken 2009). The first
tree's pair-copulas join pairs of the series themselves; each later
tree's join two series given the ones they already share, through the
conditional distributions of the tree before (the h-functions,
``Copula.du`` and ``Copula.dv``). So each pair can have its own kind of
dependence: one pair a Clayton copula's lower-tail dependence, another
none. Two regular shapes are here, on the series in a given ``order``
``v_1, ..., v_D``:

- a **D-vine** (``structure="d"``), a path: tree 1 joins neighbours
  ``(v_i, v_{i+1})``, tree ``j`` joins ``(v_i, v_{i+j})`` given the
  series between them;
- a **C-vine** (``structure="c"``), stars: tree ``j`` joins the root
  ``v_j`` to each later series given ``v_1, ..., v_{j-1}``.

With ``x`` and ``y`` the two arguments of a pair-copula ``C`` (the
conditional probabilities of its two series given the conditioning
ones), the arguments of the next tree are ``dC/dy`` (the first given the
second) and ``dC/dx``; the density multiplies ``c(x, y)`` over the
edges, and a draw inverts the same recursion one series at a time (the
inverse Rosenblatt transform).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as onp
import numpy.typing as npt
from scipy.optimize import minimize

from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.rng import as_generator
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option

from .copula import _U_CLIP, Copula, _check_margins, _perfect_dependence

#: The shapes of vine.
STRUCTURES = ("d", "c")
# Above this many series the best path of a D-vine is found greedily
_EXHAUSTIVE_ORDER = 8


def _families() -> dict:
    from .archimedean import AMH, Clayton, Frank, Gumbel, Independence, Joe
    from .elliptical import Gaussian, StudentT

    return {
        c.name: c
        for c in (
            Independence,
            Gaussian,
            StudentT,
            Frank,
            Clayton,
            Gumbel,
            Joe,
            AMH,
        )
    }


def default_candidates() -> list:
    """The pair-copulas a fit chooses among by AIC: the independence
    copula, then every built-in family, the Clayton, Gumbel and Joe in
    each of their four rotations.

    Examples
    --------
    >>> from surpyval.multivariate.parametric.copula.vine import (
    ...     default_candidates,
    ... )
    >>> len(default_candidates())
    17
    >>> default_candidates()[:3]
    [Independence copula, Gaussian copula, StudentT copula]
    """
    out = []
    for family in _families().values():
        out.append(family)
        if family.rotatable:
            out.extend(family.rotated(r) for r in (90, 180, 270))
    return out


def _is_independence(family: Any) -> bool:
    return not family.parameter_names


def _clip(u: Any) -> Any:
    return onp.clip(onp.asarray(u, dtype=float), _U_CLIP, 1 - _U_CLIP)


def _as_dim(u: Any) -> dict:
    """Pseudo-observations as a prepared dimension of fully observed
    values with no margin density (see ``Copula._prepare_dim``)."""
    u = _clip(u)
    return {
        "c": onp.zeros(u.shape, dtype=int),
        "u": u,
        "ulo": u,
        "uhi": u,
        "logf": onp.zeros(u.shape),
        "ul": onp.zeros(u.shape),
        "ur": onp.ones(u.shape),
        "has_trunc": False,
    }


def _fit_pair(family: Any, x: Any, y: Any, weights: Any) -> tuple | None:
    """``(theta, neg_ll)`` of the pair-copula ``family`` fitted to the
    pseudo-observations ``(x, y)``, or ``None`` where it has no finite
    likelihood there."""
    if _is_independence(family):
        return onp.zeros(0), 0.0
    dims = [_as_dim(x), _as_dim(y)]
    to_unbounded, to_bounded = family._bounds_transforms()
    with onp.errstate(all="ignore"):
        start = onp.asarray(
            to_unbounded(family._just_inside(family._init_theta(dims))),
            dtype=float,
        )
        if not onp.all(onp.isfinite(start)):
            return None
        res = minimize(
            lambda phi: family.neg_ll(to_bounded(phi), dims, weights),
            start,
            method="Nelder-Mead",
        )
    if not onp.isfinite(res.fun):
        return None
    return onp.asarray(to_bounded(res.x), dtype=float), float(res.fun)


class _Edge:
    """One pair-copula of a vine: its tree, the two series it joins and
    the series it conditions on (as the data's axes), its family and its
    place in the parameter vector."""

    def __init__(
        self, tree: int, pair: tuple, given: tuple, family: Any, start: int
    ) -> None:
        self.tree = tree
        self.pair = pair
        self.given = given
        self.family = family
        self.slice = slice(start, start + len(family.parameter_names))
        self.independent = _is_independence(family)

    @property
    def label(self) -> str:
        out = f"{self.pair[0]}_{self.pair[1]}"
        if self.given:
            out += "|" + "_".join(str(g) for g in self.given)
        return out

    def pdf(self, x: Any, y: Any, params: Any) -> Any:
        if self.independent:
            return onp.ones(onp.shape(x))
        return self.family.pdf(x, y, *params[self.slice])

    def first_given_second(self, x: Any, y: Any, params: Any) -> Any:
        if self.independent:
            return x
        return _clip(self.family.dv(x, y, *params[self.slice]))

    def second_given_first(self, x: Any, y: Any, params: Any) -> Any:
        if self.independent:
            return y
        return _clip(self.family.du(x, y, *params[self.slice]))

    def second_from(self, x: Any, w: Any, params: Any) -> Any:
        """The second argument ``y`` at which ``P(Y <= y | X = x)`` is
        ``w``."""
        if self.independent:
            return w
        return _clip(self.family._invert_du(x, w, params[self.slice]))


def _edge_pairs(kind: str, D: int) -> list:
    """Every edge as ``(tree, first, second)`` in positions of the order,
    tree by tree."""
    out = []
    for tree in range(1, D):
        if kind == "d":
            out.extend((tree, i, i + tree) for i in range(D - tree))
        else:
            out.extend((tree, tree - 1, k) for k in range(tree, D))
    return out


def _given(kind: str, first: int, second: int) -> range:
    """The positions an edge conditions on."""
    return range(first + 1, second) if kind == "d" else range(first)


class VineCopula(Copula):
    """A C- or D-vine of given pair-copula families: the copula of a
    model that ``Vine.fit`` and ``Vine.from_params`` return.

    Its parameters are the pair-copulas' in the order of the edges (tree
    by tree), each named by its family's parameter and its edge:
    ``theta_0_1`` for the pair of series 0 and 1, ``rho_0_2|1`` for the
    pair 0 and 2 given series 1.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Clayton, Frank, Gaussian, Vine
    >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
    >>> model = Vine.from_params(
    ...     [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    ... )
    >>> model.copula
    D-vine copula of 3 series (order 0, 1, 2)
    >>> model.parameter_names
    ['theta_0_1', 'rho_1_2', 'theta_0_2|1']
    """

    name = "Vine"

    def __init__(self, kind: str, order: Any, families: list) -> None:
        self.kind = kind
        self.order = [int(d) for d in order]
        self.n_axes = D = len(self.order)
        self.edges: list[_Edge] = []
        self._at: dict = {}
        names: list = []
        bounds: list = []
        closed: list = []
        flat = [f for tree in families for f in tree]
        for (tree, first, second), family in zip(_edge_pairs(kind, D), flat):
            edge = _Edge(
                tree,
                (self.order[first], self.order[second]),
                tuple(self.order[g] for g in _given(kind, first, second)),
                family,
                len(names),
            )
            self.edges.append(edge)
            self._at[(first, second)] = edge
            for name in family.parameter_names:
                names.append(f"{name}_{edge.label}")
                if name in family.closed_bounds:
                    closed.append(names[-1])
            bounds.extend(family.bounds)
        self.parameter_names = names
        self.bounds = tuple(bounds)
        self.closed_bounds = tuple(closed)
        self.dependence_limits = {}
        self.independence_at = {}

    def __repr__(self) -> str:
        return (
            f"{self.kind.upper()}-vine copula of {self.n_axes} series "
            f"(order {', '.join(str(d) for d in self.order)})"
        )

    def _model_cls(self) -> Any:
        from ._nd_model import MultivariateCopulaModel

        return MultivariateCopulaModel

    def _saved(self) -> dict:
        return {
            "copula": "Vine",
            "axes": self.n_axes,
            "vine": {
                "structure": self.kind,
                "order": list(self.order),
                "families": [
                    [e.family.name, int(e.family.rotation)] for e in self.edges
                ],
            },
        }

    def describe(self, params: Any) -> list:
        """Each edge as a dictionary: its ``tree``, the ``pair`` of series
        it joins, the series it is ``given``, its ``family`` (and
        ``rotation``), its ``params`` and the ``kendall_tau`` of its
        pair-copula (of the pair itself in tree 1; given the conditioning
        series, for the vine's pair-copulas, in the later trees)."""
        params = onp.asarray(params, dtype=float)
        out = []
        for edge in self.edges:
            theta = params[edge.slice]
            out.append(
                {
                    "tree": edge.tree,
                    "pair": edge.pair,
                    "given": edge.given,
                    "family": edge.family.name,
                    "rotation": int(edge.family.rotation),
                    "params": theta.copy(),
                    "kendall_tau": (
                        0.0
                        if edge.independent
                        else float(edge.family.kendall_tau(*theta))
                    ),
                }
            )
        return out

    # -- the density -------------------------------------------------------
    def _log_density(self, U: Any, params: Any) -> Any:
        params = onp.asarray(params, dtype=float)
        V = _clip(U)[:, self.order]
        D = self.n_axes
        total = onp.zeros(V.shape[0])

        def add(edge: _Edge, x: Any, y: Any) -> None:
            nonlocal total
            if not edge.independent:
                with onp.errstate(divide="ignore"):
                    total = total + onp.log(
                        onp.asarray(edge.pdf(x, y, params), dtype=float)
                    )

        if self.kind == "d":
            # low[i]: v_i given the tree's window above it; high[i]: the
            # window's last series given the ones below it
            low = [V[:, i] for i in range(D)]
            high = list(low)
            for tree in range(1, D):
                new_low, new_high = [], []
                for i in range(D - tree):
                    edge = self._at[(i, i + tree)]
                    x, y = low[i], high[i + 1]
                    add(edge, x, y)
                    if tree < D - 1:
                        new_low.append(edge.first_given_second(x, y, params))
                        new_high.append(edge.second_given_first(x, y, params))
                low, high = new_low, new_high
        else:
            # w[k]: v_k given the roots so far
            w = [V[:, k] for k in range(D)]
            for root in range(D - 1):
                for k in range(root + 1, D):
                    edge = self._at[(root, k)]
                    y = w[k]
                    add(edge, w[root], y)
                    if root < D - 2:
                        w[k] = edge.second_given_first(w[root], y, params)
        return total

    def pdf(self, U: Any, *params: Any) -> Any:  # type: ignore[override]
        """The copula's density at the points ``U`` (one row per point, a
        column per series)."""
        return onp.exp(self._log_density(onp.atleast_2d(U), params))

    def _no_joint_probability(self, what: str) -> ValueError:
        return ValueError(
            f"A vine copula has no closed form for {what}: it is the "
            "integral of the density over all the series, which SurPyval "
            "does not compute. Estimate it from draws (the proportion of "
            "`model.random(size, random_state)` in the region), or fit a "
            "Gaussian or Student-t copula, which has it."
        )

    def cdf(self, U: Any, *params: Any) -> Any:  # type: ignore[override]
        """Not computed for a vine; raises a ``ValueError`` saying what to
        use."""
        raise self._no_joint_probability("its joint CDF")

    def _survival(self, U: Any, *params: Any) -> Any:  # type: ignore[override]
        raise self._no_joint_probability("its joint survival function")

    def conditional_cdf(
        self, U: Any, of: int, given: Any, params: Any
    ) -> npt.NDArray:
        """The conditional distributions the vine has in closed form:
        those of the last series of the order given all the others."""
        if of != self.order[-1] or set(given) != set(self.order[:-1]):
            raise ValueError(
                "A vine copula has a closed form for the conditional "
                "distribution of the last series of its order given all "
                f"the others only (of={self.order[-1]}); for another, fit "
                "the vine with that series last in `order`, or use a "
                "Gaussian or Student-t copula."
            )
        params = onp.asarray(params, dtype=float)
        V = _clip(U)[:, self.order]
        D = self.n_axes
        if self.kind == "d":
            low = [V[:, i] for i in range(D)]
            high = list(low)
            for tree in range(1, D):
                new_low, new_high = [], []
                for i in range(D - tree):
                    edge = self._at[(i, i + tree)]
                    x, y = low[i], high[i + 1]
                    new_low.append(edge.first_given_second(x, y, params))
                    new_high.append(edge.second_given_first(x, y, params))
                low, high = new_low, new_high
            return high[0]
        w = [V[:, k] for k in range(D)]
        for root in range(D - 1):
            for k in range(root + 1, D):
                w[k] = self._at[(root, k)].second_given_first(
                    w[root], w[k], params
                )
        return w[D - 1]

    def sample(
        self, size: int, params: Any, random_state: Any = None
    ) -> npt.NDArray:
        """``size`` draws from the copula (one row per draw), by the
        inverse Rosenblatt transform: each series in the order from a
        uniform and the series before it."""
        params = onp.asarray(params, dtype=float)
        rng = as_generator(random_state)
        D = self.n_axes
        P = rng.uniform(_U_CLIP, 1 - _U_CLIP, size=(size, D))
        V = onp.empty((size, D))
        V[:, 0] = P[:, 0]
        if self.kind == "d":
            low = {(0, 0): P[:, 0]}  # (i, k): v_i given v_{i+1..k}
            for k in range(1, D):
                value = P[:, k]  # v_k given v_{i..k-1}, for i = 0, 1, ...
                high = {}
                for i in range(k):
                    high[i] = value
                    value = self._at[(i, k)].second_from(
                        low[(i, k - 1)], value, params
                    )
                V[:, k] = value
                low[(k, k)] = high[k] = value
                for i in range(k - 1, -1, -1):
                    low[(i, k)] = self._at[(i, k)].first_given_second(
                        low[(i, k - 1)], high[i + 1], params
                    )
        else:
            for k in range(1, D):
                value = P[:, k]
                for root in range(k - 1, -1, -1):
                    value = self._at[(root, k)].second_from(
                        P[:, root], value, params
                    )
                V[:, k] = value
        U = onp.empty((size, D))
        U[:, self.order] = V
        return _clip(U)

    def kendall_tau(self, *params: float) -> Any:
        """Kendall's tau of each pair of series joined in the first tree
        (a matrix); ``nan`` for the other pairs, whose dependence is
        spread over several pair-copulas (``describe`` gives each
        pair-copula's own tau; estimate a pair's from draws)."""
        out = onp.full((self.n_axes, self.n_axes), onp.nan)
        onp.fill_diagonal(out, 1.0)
        for entry in self.describe(params):
            if entry["tree"] == 1:
                i, j = entry["pair"]
                out[i, j] = out[j, i] = entry["kendall_tau"]
        return out

    def spearman_rho(self, *params: float) -> Any:
        """Spearman's rho of each pair of series joined in the first tree
        (a matrix); ``nan`` for the other pairs, as :meth:`kendall_tau`."""
        out = onp.full((self.n_axes, self.n_axes), onp.nan)
        onp.fill_diagonal(out, 1.0)
        for edge in self.edges:
            if edge.tree == 1:
                theta = onp.asarray(params, dtype=float)[edge.slice]
                i, j = edge.pair
                out[i, j] = out[j, i] = (
                    0.0
                    if edge.independent
                    else float(edge.family.spearman_rho(*theta))
                )
        return out

    def tail_dependence(self, *params: float) -> tuple:
        """``(lambda_L, lambda_U)``, each a matrix with the coefficients
        of the pairs joined in the first tree; ``nan`` for the other
        pairs."""
        lower = onp.full((self.n_axes, self.n_axes), onp.nan)
        onp.fill_diagonal(lower, 1.0)
        upper = lower.copy()
        for edge in self.edges:
            if edge.tree == 1:
                theta = onp.asarray(params, dtype=float)[edge.slice]
                lam = (
                    (0.0, 0.0)
                    if edge.independent
                    else edge.family.tail_dependence(*theta)
                )
                i, j = edge.pair
                lower[i, j] = lower[j, i] = float(lam[0])
                upper[i, j] = upper[j, i] = float(lam[1])
        return lower, upper

    # -- the likelihood ----------------------------------------------------
    def _rows_loglik(self, params: Any, dims: list) -> Any:
        U = onp.column_stack([d["u"] for d in dims])
        logf = onp.column_stack([d["logf"] for d in dims])
        return self._log_density(U, params) + onp.sum(logf, axis=1)

    def neg_ll(self, params: Any, dims: list, weights: npt.NDArray) -> float:
        params = onp.asarray(params, dtype=float)
        for value, (low, high) in zip(params, self.bounds):
            if (
                not onp.isfinite(value)
                or (low is not None and value < low)
                or (high is not None and value > high)
            ):
                return onp.inf
        with onp.errstate(all="ignore"):
            ll = self._rows_loglik(params, dims)
        total = -float(onp.sum(weights * ll))
        return total if onp.isfinite(total) else onp.inf

    def _bounds_transforms(self) -> tuple:
        parts = [
            (edge.slice, edge.family._bounds_transforms())
            for edge in self.edges
            if not edge.independent
        ]

        def convert(which: int) -> Any:
            def f(values: npt.NDArray) -> npt.NDArray:
                values = onp.asarray(values, dtype=float)
                out = onp.empty(values.shape)
                for sl, transforms in parts:
                    out[sl] = transforms[which](values[sl])
                return out

            return f

        return convert(0), convert(1)

    def _just_inside(self, theta: npt.NDArray) -> npt.NDArray:
        out = onp.array(theta, dtype=float)
        for edge in self.edges:
            if not edge.independent:
                out[edge.slice] = edge.family._just_inside(out[edge.slice])
        return out

    def _init_theta(self, dims: list) -> npt.NDArray:
        U = onp.column_stack([d["u"] for d in dims])
        weights = onp.ones(U.shape[0])
        return _sequential(
            self.kind, self.order, U, weights, [[e.family] for e in self.edges]
        )[1]

    def _warn_if_perfectly_dependent(
        self, data: Any, theta: npt.NDArray
    ) -> bool:
        from .elliptical_nd import _pair_view

        for i, j in itertools.combinations(range(self.n_axes), 2):
            sign, rows = _perfect_dependence(_pair_view(data, i, j))
            if not sign:
                continue
            word, relation = (
                ("concordant", "increasing")
                if sign > 0
                else ("discordant", "decreasing")
            )
            warn_no_maximum(
                f"the {rows} rows of series {i} and {j} are perfectly "
                f"{word} (Kendall's tau = {sign}), a dependence no "
                "pair-copula with a finite parameter has; the likelihood "
                "keeps increasing towards it, or peaks only where the "
                "fitted margins stop mapping one series exactly onto the "
                "other",
                "The reported parameters and the dependence measures "
                "derived from them are meaningless",
                f"series {j} is an {relation} function of series {i}: "
                "model that relationship directly, and fit the vine to "
                "the other series",
            )
            return True
        return False

    @takes_unit
    def fit(self, x: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Fit this vine (its order and families as they are) and its
        margins; ``Vine.fit`` chooses the order and the families and
        then calls it.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton, Frank, Gaussian, Vine
        >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
        >>> truth = Vine.from_params(
        ...     [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
        ... )
        >>> X = truth.random(300, random_state=0)
        >>> model = truth.copula.fit(X, margins=[Weibull] * 3)
        >>> model.parameter_names
        ['theta_0_1', 'rho_1_2', 'theta_0_2|1']
        """
        from surpyval.multivariate.parametric.data import (
            MultivariateSurpyvalData,
        )

        data = x
        if not isinstance(x, MultivariateSurpyvalData):
            names = ("c", "n", "t", "margins", "how", "xl", "xr", "init")
            given = dict(zip(names, args))
            given.update(kwargs)
            data = MultivariateSurpyvalData(
                x, **{k: given.pop(k, None) for k in ("c", "n", "t", "xl", "xr")}
            )
            args, kwargs = (), given
        _check_complete(data)
        return Copula.fit(self, data, *args, **kwargs)


def _check_complete(data: Any) -> None:
    """A vine is fitted to fully observed rows without truncation."""
    censored = onp.any(data.c != 0, axis=1)
    if censored.any():
        raise ValueError(
            f"A vine copula is fitted to fully observed rows; "
            f"{int(censored.sum())} row(s) have a censored value (c). A "
            "censored row's likelihood is an integral of the vine's "
            "density over the censored series, which is not computed. "
            "The Gaussian and Student-t copulas take censored rows of "
            "three or more series (Gaussian.fit, StudentT.fit)."
        )
    if onp.isfinite(data.t).any():
        raise ValueError(
            "A vine copula does not take truncated data (t): its "
            "likelihood divides by the vine's mass over the window, which "
            "is not computed. A copula of two series takes truncation: "
            "fit the series in pairs."
        )


def _sequential(
    kind: str, order: list, U: Any, weights: Any, choices: list
) -> tuple:
    """The families and parameters of a vine fitted tree by tree to the
    pseudo-observations ``U``: each edge, in turn, the candidate of its
    ``choices`` (one list of families per edge) with the lowest AIC on
    the arguments the trees before it give. ``(families, theta)``."""
    V = _clip(U)[:, order]
    D = V.shape[1]
    chosen: list = []
    thetas: list = []
    queue = iter(choices)

    def select(x: Any, y: Any) -> tuple:
        best: tuple | None = None
        for family in next(queue):
            fitted = _fit_pair(family, x, y, weights)
            if fitted is None:
                continue
            aic = 2.0 * len(fitted[0]) + 2.0 * fitted[1]
            if best is None or aic < best[0]:
                best = (aic, family, fitted[0])
        if best is None:
            raise ValueError(
                "No pair-copula candidate has a finite likelihood on one "
                "of the vine's pairs; add the Independence copula to "
                "`candidates`."
            )
        chosen.append(best[1])
        thetas.append(best[2])
        edge = _Edge(0, (0, 1), (), best[1], 0)
        return edge, best[2]

    if kind == "d":
        low = [V[:, i] for i in range(D)]
        high = list(low)
        for tree in range(1, D):
            new_low, new_high = [], []
            for i in range(D - tree):
                x, y = low[i], high[i + 1]
                edge, theta = select(x, y)
                new_low.append(edge.first_given_second(x, y, theta))
                new_high.append(edge.second_given_first(x, y, theta))
            low, high = new_low, new_high
    else:
        w = [V[:, k] for k in range(D)]
        for root in range(D - 1):
            for k in range(root + 1, D):
                edge, theta = select(w[root], w[k])
                w[k] = edge.second_given_first(w[root], w[k], theta)
    theta = onp.concatenate(thetas) if thetas else onp.zeros(0)
    return chosen, theta


def _default_order(kind: str, U: Any) -> list:
    """The order by strongest absolute Kendall's tau: for a C-vine, each
    root in turn the series with the largest sum of ``|tau|`` with the
    ones left; for a D-vine, the path with the largest sum of ``|tau|``
    between neighbours (every path up to eight series, then from the
    strongest pair by nearest neighbour). Ties go to the lower axis."""
    from scipy.stats import kendalltau

    D = U.shape[1]
    tau = onp.zeros((D, D))
    for i, j in itertools.combinations(range(D), 2):
        value = kendalltau(U[:, i], U[:, j]).statistic
        tau[i, j] = tau[j, i] = abs(value) if onp.isfinite(value) else 0.0
    if kind == "c":
        left = list(range(D))
        order = []
        while left:
            sums = [sum(tau[d, e] for e in left if e != d) for d in left]
            order.append(left.pop(int(onp.argmax(sums))))
        return order
    if D <= _EXHAUSTIVE_ORDER:
        best: tuple | None = None
        for path in itertools.permutations(range(D)):
            if path[0] > path[-1]:
                continue  # (a path and its reverse are one vine)
            total = sum(tau[a, b] for a, b in zip(path, path[1:]))
            if best is None or total > best[0] + 1e-12:
                best = (total, path)
        assert best is not None
        return list(best[1])
    i, j = onp.unravel_index(int(onp.argmax(tau)), tau.shape)
    path = [int(min(i, j)), int(max(i, j))]
    left = [d for d in range(D) if d not in path]
    while left:
        ends = [(tau[path[0], d], 0, d) for d in left]
        ends += [(tau[path[-1], d], 1, d) for d in left]
        _, end, d = max(ends, key=lambda e: (e[0], -e[2]))
        path = [d] + path if end == 0 else path + [d]
        left.remove(d)
    return path


def _check_order(order: Any, D: int) -> list:
    try:
        out = [int(d) for d in order]
    except (TypeError, ValueError):
        out = []
    if sorted(out) != list(range(D)):
        raise ValueError(
            f"order must be the {D} axes 0 to {D - 1}, each once, in the "
            f"vine's order; got {order!r}."
        )
    return out


def _is_family(value: Any) -> bool:
    return isinstance(value, Copula) and not isinstance(value, VineCopula)


def _edge_choices(
    families: Any, candidates: Any, truncate: Any, kind: str, D: int
) -> list:
    """One list of candidate families per edge, tree by tree."""
    from .archimedean import Independence

    edges = _edge_pairs(kind, D)
    if truncate is not None and (
        isinstance(truncate, bool)
        or not isinstance(truncate, (int, onp.integer))
        or not 1 <= truncate <= D - 1
    ):
        raise ValueError(
            f"truncate must be a number of trees, 1 to {D - 1}, got "
            f"{truncate!r}."
        )
    if candidates is not None:
        candidates = list(candidates)
        if not candidates or not all(_is_family(c) for c in candidates):
            raise ValueError(
                "candidates must be a list of bivariate copula families, "
                "e.g. [Independence, Gaussian, Clayton, "
                "Clayton.rotated(180)]."
            )
    if families is None:
        pool = default_candidates() if candidates is None else candidates
        out = [list(pool) for _ in edges]
    else:
        if candidates is not None:
            raise ValueError(
                "Give families (each edge's pair-copula) or candidates "
                "(the families to choose among by AIC), not both."
            )
        if _is_family(families):
            out = [[families] for _ in edges]
        else:
            trees = [list(t) if not _is_family(t) else [t] for t in families]
            sizes = [D - tree for tree in range(1, D)]
            if truncate is not None:
                trees = trees[:truncate] + [
                    [Independence] * size for size in sizes[truncate:]
                ][len(trees[truncate:]) * 0 :]
                trees = trees[: D - 1]
            if [len(t) for t in trees] != sizes or not all(
                _is_family(f) for t in trees for f in t
            ):
                raise ValueError(
                    f"families must be one bivariate copula family, or a "
                    f"list per tree of the {D - 1} trees with a family "
                    f"per edge ({sizes} edges, e.g. [[Clayton, Gaussian], "
                    f"[Frank]] for three series); got {families!r}."
                )
            out = [[f] for t in trees for f in t]
    if truncate is not None:
        out = [
            choice if tree <= truncate else [Independence]
            for (tree, _, _), choice in zip(edges, out)
        ]
    return out


def _as_trees(flat: list, D: int) -> list:
    """A flat list of families (edge by edge) as a list per tree."""
    out, start = [], 0
    for tree in range(1, D):
        out.append(flat[start : start + D - tree])
        start += D - tree
    return out


class VineFitter:
    """Vine (pair-copula) copulas of two or more series: ``Vine``.

    A vine builds the dependence among ``D`` series from bivariate
    copulas, one per pair of series, so that each pair has its own kind
    of dependence (see the module's description, and "Vines" in the
    multivariate guide). ``Vine.fit`` chooses the order of the series and
    each pair-copula's family, unless they are given, and fits the
    pair-copulas' parameters and the margins.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.multivariate import Clayton, Frank, Gaussian, Vine
    >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
    >>> truth = Vine.from_params(
    ...     [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    ... )
    >>> X = truth.random(300, random_state=0)
    >>> model = Vine.fit(X, margins=[Weibull] * 3, order=[0, 1, 2])
    >>> [edge["family"] for edge in model.edges]
    ['Clayton', 'Gaussian', 'Frank']
    """

    name = "Vine"

    def __repr__(self) -> str:
        return "Vine copula"

    @takes_unit
    def fit(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        margins: Any = None,
        how: str = "IFM",
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        structure: str = "d",
        families: Any = None,
        order: Any = None,
        truncate: int | None = None,
        candidates: Any = None,
    ) -> Any:
        """Fit a vine copula and its margins to fully observed
        multivariate data.

        The margins are fitted first. Then, on their probabilities, the
        vine is fitted tree by tree: each pair-copula in turn, on the
        arguments the trees before it give, choosing its family by AIC
        where it is not given. Those estimates start the maximum
        likelihood search over all the pair-copulas' parameters together
        (``how="IFM"``), or over them and the margins' (``how="MLE"``).

        Parameters
        ----------
        x, c, n, t, xl, xr
            Multivariate survival data, or a ``MultivariateSurpyvalData``
            as ``x``, with ``D >= 2`` axes. Every value must be observed:
            censored rows and truncation are refused (the Gaussian and
            Student-t copulas take censored rows of three or more
            series).
        margins : sequence of length D
            A distribution to fit, or a fitted model, per series.
        how : {"IFM", "MLE"}
            The margins first and then the vine (the default), or all the
            parameters jointly.
        structure : {"d", "c"}, optional
            A D-vine (a path through the series; the default) or a C-vine
            (a root series per tree). With three series the two are the
            same vines.
        families : Copula or list of lists, optional
            The pair-copulas: one family for every edge, or a list per
            tree with a family per edge, in the order of the edges
            (``[[Clayton, Gaussian], [Frank]]`` for three series; a
            rotated family as ``Clayton.rotated(180)``). By default each
            edge's family is chosen by AIC among ``candidates``.
        order : sequence of int, optional
            The order of the series in the vine (a D-vine's path; a
            C-vine's roots, tree by tree). By default the order of
            strongest absolute Kendall's tau: the D-vine path with the
            largest sum between neighbours, or C-vine roots each with
            the largest sum with the series left.
        truncate : int, optional
            Keep the pair-copulas of the first ``truncate`` trees and
            make the later ones independence copulas (the series are
            then conditionally independent given enough of the others).
        candidates : list of Copula, optional
            The families chosen among by AIC. By default the
            independence copula and every built-in family, the Clayton,
            Gumbel and Joe in each rotation.

        Returns
        -------
        CopulaModel
            A model whose points are ``(x_1, ..., x_D)`` rows, with
            ``pdf``, ``random``, ``edges`` (each pair-copula's family,
            parameters and Kendall's tau), ``aic``, ``covariance``,
            ``standard_errors`` and ``param_cb``. A vine has no closed
            form for the joint ``sf`` and ``cdf``: they refuse, saying
            to estimate them from ``random``.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton, Frank, Gaussian, Vine
        >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
        >>> truth = Vine.from_params(
        ...     [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
        ... )
        >>> X = truth.random(300, random_state=0)
        >>> model = Vine.fit(
        ...     X,
        ...     margins=[Weibull] * 3,
        ...     order=[0, 1, 2],
        ...     families=[[Clayton, Gaussian], [Frank]],
        ... )
        >>> model.parameter_names
        ['theta_0_1', 'rho_1_2', 'theta_0_2|1']
        >>> model.params.round(1)
        array([2.2, 0.5, 3.4])
        """
        from surpyval.multivariate.parametric.data import (
            MultivariateSurpyvalData,
        )

        check_option("structure", structure, STRUCTURES)
        check_option("how", how, ("IFM", "MLE"))
        if isinstance(x, MultivariateSurpyvalData):
            given = [
                k
                for k, v in (("c", c), ("n", n), ("t", t), ("xl", xl), ("xr", xr))
                if v is not None
            ]
            if given:
                raise ValueError(
                    "x is a MultivariateSurpyvalData, which holds the "
                    f"data: leave out {', '.join(given)} (they go in the "
                    "data)."
                )
            data = x
        else:
            data = MultivariateSurpyvalData(x, c=c, n=n, t=t, xl=xl, xr=xr)
        _check_complete(data)
        D = data.D
        margins = _check_margins(margins, D, fitted_only=False)
        choices = _edge_choices(families, candidates, truncate, structure, D)
        probe = VineCopula(structure, range(D), _as_trees([c[0] for c in choices], D))
        models = probe._fit_margins(margins, data)
        U = onp.column_stack(
            [
                probe._prepare_dim(models[d], *data.dimension(d))["u"]
                for d in range(D)
            ]
        )
        if order is None:
            order = _default_order(structure, U)
        order = _check_order(order, D)
        chosen, theta = _sequential(
            structure, order, U, onp.asarray(data.n, dtype=float), choices
        )
        vine = VineCopula(structure, order, _as_trees(chosen, D))
        init = vine._just_inside(theta) if theta.size else None
        return Copula.fit(vine, data, margins=margins, how=how, init=init)

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_cols: list[str],
        c_cols: list[str] | None = None,
        n_col: str | None = None,
        xl_cols: list[str] | None = None,
        xr_cols: list[str] | None = None,
        tl_cols: list[str] | None = None,
        tr_cols: list[str] | None = None,
        **fit_options: Any,
    ) -> Any:
        """Fit a vine copula and its margins to the columns of a
        :class:`pandas.DataFrame`: the arguments of a copula's
        ``fit_from_df`` (``x_cols``, a column per series, and
        ``n_col``), then the options of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton, Vine
        >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
        >>> X = Vine.from_params(
        ...     [2.0, 1.0, 0.5], margins, families=Clayton
        ... ).random(200, random_state=0)
        >>> df = pd.DataFrame(X, columns=["pump", "motor", "seal"])
        >>> model = Vine.fit_from_df(
        ...     df,
        ...     x_cols=["pump", "motor", "seal"],
        ...     margins=[Weibull] * 3,
        ...     families=Clayton,
        ...     order=[0, 1, 2],
        ... )
        >>> model.parameter_names
        ['theta_0_1', 'theta_1_2', 'theta_0_2|1']
        """
        return Copula.fit_from_df(
            self,  # type: ignore[arg-type]
            df,
            x_cols,
            c_cols=c_cols,
            n_col=n_col,
            xl_cols=xl_cols,
            xr_cols=xr_cols,
            tl_cols=tl_cols,
            tr_cols=tr_cols,
            **fit_options,
        )

    @takes_unit
    def from_params(
        self,
        params: Any,
        margins: Any,
        structure: str = "d",
        families: Any = None,
        order: Any = None,
    ) -> Any:
        """The vine model of known parameters and margins, without
        fitting.

        Parameters
        ----------
        params : array like
            The pair-copulas' parameters, edge by edge (tree by tree),
            each family's own (none for an independence copula, ``rho``
            and ``nu`` for a Student-t).
        margins : sequence of fitted models
            One per series, each with ``ff`` and ``df``.
        structure : {"d", "c"}, optional
            A D-vine (the default) or a C-vine.
        families : Copula or list of lists
            The pair-copulas, as for :meth:`fit`. Required.
        order : sequence of int, optional
            The order of the series in the vine; by default the axes'
            own.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.multivariate import Clayton, Frank, Gaussian, Vine
        >>> margins = [Weibull.from_params([10 * (d + 1), 2]) for d in range(3)]
        >>> model = Vine.from_params(
        ...     [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
        ... )
        >>> model.pdf([[8.0, 15.0, 25.0]]).round(8)
        array([2.576e-05])
        >>> [(e["pair"], e["given"], e["family"]) for e in model.edges]
        [((0, 1), (), 'Clayton'), ((1, 2), (), 'Gaussian'), ((0, 2), (1,), 'Frank')]
        """
        check_option("structure", structure, STRUCTURES)
        if not isinstance(margins, (list, tuple)):
            margins = _check_margins(margins, 2, fitted_only=True)
        D = len(margins)
        margins = _check_margins(margins, D, fitted_only=True)
        if families is None:
            raise ValueError(
                "families must be given: one bivariate copula family for "
                "every edge, or a list per tree with a family per edge, "
                "e.g. [[Clayton, Gaussian], [Frank]] for three series."
            )
        choices = _edge_choices(families, None, None, structure, D)
        order = _check_order(range(D) if order is None else order, D)
        vine = VineCopula(structure, order, _as_trees([c[0] for c in choices], D))
        params = vine._check_params(params)
        return vine._model_cls()(vine, params, margins, data=None, how="given")


Vine = VineFitter()


def restored(model_dict: dict) -> VineCopula:
    """The vine of a saved model."""
    saved = model_dict["vine"]
    families = _families()
    flat = []
    for name, rotation in saved["families"]:
        if name not in families:
            raise ValueError(
                f"Unknown copula family {name!r} in a saved vine; expected "
                f"one of {sorted(families)}."
            )
        flat.append(families[name].rotated(int(rotation)))
    D = int(model_dict["axes"])
    return VineCopula(saved["structure"], saved["order"], _as_trees(flat, D))
