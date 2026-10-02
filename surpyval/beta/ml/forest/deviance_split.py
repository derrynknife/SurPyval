r"""Full-likelihood (exponential deviance) split for survival trees.

The risk-set log-rank split (see ``log_rank_split``) is defined only for
data that can be expressed in the xrd (risk set / death count) format:
observed and right-censored observations, optionally with delayed entry
(left truncation). Left-censored, interval-censored and right-truncated
observations carry their event information as *probabilities of intervals*
rather than as points entering and leaving a risk set, so no risk-set
statistic exists for them.

The likelihood does not have that limitation. This module implements the
classic parametric alternative -- the exponential log-likelihood
(deviance) split of Davis & Anderson (1989), scored with SurPyval's full
likelihood so that every observation type contributes exactly:

- observed ``x``:            :math:`\log \lambda - \lambda x`
- right censored at ``x``:   :math:`-\lambda x`
- left censored at ``x``:    :math:`\log(1 - e^{-\lambda x})`
- interval ``(x_l, x_r]``:   :math:`\log(e^{-\lambda x_l} - e^{-\lambda x_r})`
- truncated to ``(t_l, t_r]``: minus
  :math:`\log(S(t_l) - S(t_r))`

A candidate split is scored by the joint maximised log-likelihood of its
two children; since the parent's log-likelihood is constant within a node,
maximising :math:`\ell_L^* + \ell_R^*` maximises the deviance gain
:math:`2(\ell_L^* + \ell_R^* - \ell_{parent}^*)`. For purely observed /
right-censored data this rule and the log-rank rule pick very similar
splits; the deviance rule simply remains defined for everything else.

References
----------
Davis, R.B. and Anderson, J.R., 1989. Exponential survival trees.
*Statistics in Medicine*, 8(8), pp.947-961.
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from surpyval.utils.surpyval_data import SurpyvalData

# Cap on candidate split values per feature: above this, candidates are
# thinned to quantiles so a node's split search stays O(cap * n).
_MAX_SPLIT_CANDIDATES = 64

# Value used for non-finite likelihoods so the bounded optimiser can keep
# working (it cannot handle inf/nan).
_HUGE = 1e300


def needs_full_likelihood_split(data: SurpyvalData) -> bool:
    """
    True when ``data`` cannot be expressed in the xrd format -- i.e. it
    contains left censoring, interval censoring, or right truncation --
    so the risk-set log-rank split is undefined and the full-likelihood
    (deviance) split must be used.
    """
    return bool(
        np.isfinite(data.t[:, 1]).any()
        or (data.c == 2).any()
        or (data.c == -1).any()
    )


def _exp_neg_ll_parts(data: SurpyvalData) -> tuple:
    """
    Pre-extract the arrays the exponential negative log-likelihood needs,
    so the optimiser's objective is pure vectorised arithmetic.
    """
    a_trunc = np.maximum(data.x_tl, 0.0)  # exponential support starts at 0
    return (
        data.x_o,
        data.n_o,
        data.x_r,
        data.n_r,
        data.x_l,
        data.n_l,
        data.x_il,
        data.x_ir,
        data.n_i,
        a_trunc,
        data.x_tr,
        data.n_t,
    )


def _exp_neg_ll(theta: float, parts: tuple) -> float:
    """
    Negative log-likelihood of an exponential distribution with rate
    ``lambda = exp(theta)`` under the full data model (all censoring
    types plus truncation). Returns a large finite value where the
    likelihood is degenerate so bounded optimisation stays stable.
    """
    (
        x_o,
        n_o,
        x_r,
        n_r,
        x_l,
        n_l,
        x_il,
        x_ir,
        n_i,
        t_a,
        t_r,
        n_t,
    ) = parts
    lam = np.exp(theta)
    ll = 0.0
    with np.errstate(all="ignore"):
        if x_o.size:
            ll += np.sum(n_o * (theta - lam * x_o))
        if x_r.size:
            ll += -lam * np.sum(n_r * x_r)
        if x_l.size:
            # F(x) = 1 - exp(-lam x)
            ll += np.sum(n_l * np.log(-np.expm1(-lam * x_l)))
        if x_il.size:
            # S(xl) - S(xr) = exp(-lam xl)(1 - exp(-lam (xr - xl)))
            ll += np.sum(
                n_i * (-lam * x_il + np.log(-np.expm1(-lam * (x_ir - x_il))))
            )
        if t_a.size:
            # subtract log(S(tl) - S(tr)) per truncated observation
            finite_tr = np.isfinite(t_r)
            log_denom = -lam * t_a
            if finite_tr.any():
                width = np.where(finite_tr, t_r - t_a, np.inf)
                log_denom = log_denom + np.where(
                    finite_tr, np.log(-np.expm1(-lam * width)), 0.0
                )
            ll -= np.sum(n_t * log_denom)
    if not np.isfinite(ll):
        return _HUGE
    return -float(ll)


def _exp_theta0(data: SurpyvalData) -> "float | None":
    """
    Crude ``log(lambda)`` scale for centring the search window:
    event-weight over exposure, or ``None`` when the data carries no
    event information.
    """
    # Read the censoring codes directly rather than the likelihood
    # buckets. Those buckets hand censored rows with a finite truncation
    # bound to the likelihood as intervals (#310), which is right for a
    # likelihood but wrong here: a right-censored row would arrive in
    # the interval branch and be counted as an event.
    c, x, n = data.c, data.x, data.n
    lead = x if x.ndim == 1 else x[:, 0]
    trail = x if x.ndim == 1 else x[:, -1]

    observed = c == 0
    right = c == 1
    left = c == -1
    interval = c == 2

    exposure = (
        float(np.sum(n[observed] * lead[observed]))
        + float(np.sum(n[right] * lead[right]))
        + float(np.sum(n[left] * lead[left])) / 2.0
        + float(np.sum(n[interval] * (lead[interval] + trail[interval]))) / 2.0
    )
    events = (
        float(np.sum(n[observed]))
        + float(np.sum(n[left]))
        + float(np.sum(n[interval]))
    )
    if exposure <= 0 or events <= 0:
        return None
    return float(np.log(events / exposure))


def _exp_max_ll(
    data: SurpyvalData, bounds: "tuple[float, float] | None" = None
) -> float:
    """
    Maximised exponential log-likelihood of ``data`` under the full data
    model, via a bounded scalar search over ``log(lambda)``.

    ``bounds`` fixes the search window. Within a node, every candidate
    child must be scored over the SAME window as the parent: for
    degenerate data directions (e.g. a likelihood whose supremum sits at
    ``lambda -> 0``) the attained value depends on where the window
    ends, so per-subset windows would make the child and parent scores
    incomparable. A common window guarantees, by additivity of the
    log-likelihood over a partition, that a split never scores below
    its parent.
    """
    if bounds is None:
        theta0 = _exp_theta0(data)
        if theta0 is None:
            return -_HUGE
        bounds = (theta0 - 15.0, theta0 + 15.0)
    return _exp_max_ll_parts(_exp_neg_ll_parts(data), bounds)


def _exp_max_ll_parts(parts: tuple, bounds: "tuple[float, float]") -> float:
    """:func:`_exp_max_ll` of the data whose likelihood ``parts`` are
    given (see :func:`_exp_neg_ll_parts`)."""
    res = minimize_scalar(
        _exp_neg_ll,
        args=(parts,),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    # The bounded method can converge slightly inside a boundary
    # supremum; take the better of the optimiser's answer and the
    # window edges.
    best = min(
        float(res.fun),
        _exp_neg_ll(bounds[0], parts),
        _exp_neg_ll(bounds[1], parts),
    )
    return -best


# Fixed search window for the Weibull shape on the log scale:
# beta in [exp(-3), exp(3)] ~ [0.05, 20] covers every physically
# plausible failure mechanism.
_LOG_BETA_BOUNDS = (-3.0, 3.0)


def _wei_neg_ll(theta: NDArray, parts: tuple) -> float:
    """
    Negative log-likelihood of a Weibull distribution with scale
    ``alpha = exp(theta[0])`` and shape ``beta = exp(theta[1])`` under
    the full data model (all censoring types plus truncation), with
    ``z(x) = (x / alpha) ** beta`` the cumulative hazard. Returns a
    large finite value where the likelihood is degenerate so bounded
    optimisation stays stable.
    """
    (
        x_o,
        n_o,
        x_r,
        n_r,
        x_l,
        n_l,
        x_il,
        x_ir,
        n_i,
        t_a,
        t_r,
        n_t,
    ) = parts
    log_alpha, log_beta = float(theta[0]), float(theta[1])
    beta = np.exp(log_beta)

    def z(x: NDArray) -> NDArray:
        # (x / alpha) ** beta on the log scale; z(0) = 0
        with np.errstate(all="ignore"):
            return np.where(x > 0, np.exp(beta * (np.log(x) - log_alpha)), 0.0)

    ll = 0.0
    with np.errstate(all="ignore"):
        if x_o.size:
            # log f = log(beta) - log(alpha)
            #         + (beta - 1)(log x - log alpha) - z
            ll += np.sum(
                n_o
                * (
                    log_beta
                    - log_alpha
                    + (beta - 1.0) * (np.log(x_o) - log_alpha)
                    - z(x_o)
                )
            )
        if x_r.size:
            ll += -np.sum(n_r * z(x_r))
        if x_l.size:
            # F(x) = 1 - exp(-z)
            ll += np.sum(n_l * np.log(-np.expm1(-z(x_l))))
        if x_il.size:
            # S(xl) - S(xr) = exp(-z_l)(1 - exp(-(z_r - z_l)))
            z_l = z(x_il)
            z_r = z(x_ir)
            ll += np.sum(n_i * (-z_l + np.log(-np.expm1(-(z_r - z_l)))))
        if t_a.size:
            # subtract log(S(tl) - S(tr)) per truncated observation
            z_a = z(t_a)
            finite_tr = np.isfinite(t_r)
            log_denom = -z_a
            if finite_tr.any():
                z_tr = z(np.where(finite_tr, t_r, 1.0))
                width = np.where(finite_tr, z_tr - z_a, np.inf)
                log_denom = log_denom + np.where(
                    finite_tr, np.log(-np.expm1(-width)), 0.0
                )
            ll -= np.sum(n_t * log_denom)
    if not np.isfinite(ll):
        return _HUGE
    return -float(ll)


def _wei_max_ll(
    data: SurpyvalData,
    box: "tuple[tuple[float, float], tuple[float, float]]",
    start: NDArray,
) -> "tuple[float, NDArray]":
    """
    Maximised Weibull log-likelihood of ``data`` over the bounded
    ``(log alpha, log beta)`` ``box``, warm-started from ``start``.

    Returns ``(max log-likelihood, argmax theta)``. The value at
    ``start`` is always a lower bound on the result, so warm-starting
    each child from the parent's optimum guarantees a non-negative
    split gain by likelihood additivity (the #185 monotonicity
    property, in 2-D).
    """
    return _wei_max_ll_parts(_exp_neg_ll_parts(data), box, start)


def _wei_max_ll_parts(
    parts: tuple,
    box: "tuple[tuple[float, float], tuple[float, float]]",
    start: NDArray,
) -> "tuple[float, NDArray]":
    """:func:`_wei_max_ll` of the data whose likelihood ``parts`` are
    given (see :func:`_exp_neg_ll_parts`)."""
    lower = np.array([box[0][0], box[1][0]])
    upper = np.array([box[0][1], box[1][1]])
    start_arr = np.clip(np.asarray(start, dtype=float), lower, upper)
    at_start = _wei_neg_ll(start_arr, parts)

    # Nelder-Mead's default initial simplex steps are *relative* to the
    # coordinate values, so a start near zero (log beta ~ 0, i.e. beta
    # ~ 1) gets a microscopic simplex and the search stalls. Supply an
    # absolute-step simplex instead.
    simplex = np.array(
        [
            start_arr,
            np.clip(start_arr + np.array([0.5, 0.0]), lower, upper),
            np.clip(start_arr + np.array([0.0, 0.5]), lower, upper),
        ]
    )
    res = minimize(
        _wei_neg_ll,
        start_arr,
        args=(parts,),
        method="Nelder-Mead",
        bounds=box,
        options={
            "xatol": 1e-5,
            "fatol": 1e-8,
            "maxfev": 400,
            "initial_simplex": simplex,
        },
    )
    if float(res.fun) < at_start:
        return -float(res.fun), np.asarray(res.x, dtype=float)
    return -at_start, start_arr


def _candidate_values(Z_u: NDArray) -> NDArray:
    """Unique candidate split values, quantile-thinned for large nodes."""
    values = np.unique(Z_u)
    if values.size > _MAX_SPLIT_CANDIDATES:
        quantiles = np.linspace(0, 1, _MAX_SPLIT_CANDIDATES)
        values = np.unique(np.quantile(Z_u, quantiles))
    return values


def _closed_form(data: SurpyvalData) -> bool:
    """True when ``data`` is observed and right-censored only, untruncated
    and with positive times: the exponential rate and the Weibull scale
    then have closed-form maximum likelihood estimates (#518)."""
    x = np.asarray(data.x, dtype=float)
    return bool(
        x.ndim == 1
        and np.isin(data.c, (0, 1)).all()
        and not np.isfinite(data.t).any()
        and (x > 0).all()
    )


# Scored at once: candidates x rows of the node, at most this many.
_BLOCK = 1 << 20


class _ChildLikelihoods:
    r"""The maximised log-likelihood of the working model in the children
    of a node's candidate splits.

    For observed and right-censored data without truncation the maximum
    is found directly, for every child at once (#518): the exponential's
    in closed form, :math:`\hat\lambda = r / \sum n x`; the Weibull's by
    profiling out the scale, :math:`\hat\alpha^\beta = \sum n x^\beta /
    r`, and solving the one-dimensional score equation of the concave
    profile log-likelihood in :math:`\log\beta` by safeguarded Newton
    steps, all children at once. Both are restricted to the same
    parameter window the bounded optimisers search, and give their
    optimum to machine precision, where the optimisers stopped at their
    tolerances, so the deviances move in the last digits only. A child
    with no observed failure (whose supremum is on the window's edge),
    or whose optimum is outside the window, and every other kind of data
    (left or interval censoring, truncation), takes the bounded
    optimiser of :func:`_exp_max_ll` / :func:`_wei_max_ll`, on the
    likelihood terms of the node sliced to the child rather than a new
    :class:`SurpyvalData` per candidate -- the same arrays, so the same
    numbers.
    """

    def __init__(
        self,
        data: SurpyvalData,
        model: str,
        theta0: float,
        fit_parent: bool = True,
    ):
        # fit_parent=False skips the node's own maximum (parent_ll and
        # the Weibull's warm start), for a caller that only wants the
        # closed forms (leaf_mle).
        self.model = model
        self.parts = _exp_neg_ll_parts(data)
        # Each likelihood term's row: a child's terms are the node's at
        # its rows (as ``data[rows]`` would rebuild them, in row order).
        self.term_rows = (
            data.mask_o,
            data.mask_r,
            data.mask_l,
            data.mask_i,
            data.truncated_mask,
        )
        self.closed = _closed_form(data)
        if self.closed:
            x = np.asarray(data.x, dtype=float)
            self.n = np.asarray(data.n, dtype=float)
            self.observed = (np.asarray(data.c) == 0).astype(float)
            self.x = x
            self.log_x = np.log(x)
        if model == "weibull":
            log_alpha0 = -theta0  # exponential mean as the scale's centre
            self.box = (
                (log_alpha0 - 15.0, log_alpha0 + 15.0),
                _LOG_BETA_BOUNDS,
            )
            start = np.array([log_alpha0, 0.0])
            if not fit_parent:
                return
            if self.closed:
                ll, theta = self._weibull(np.ones((1, data.x.size), bool))
                if np.isfinite(ll[0]):
                    self.parent_ll = float(ll[0])
                    self.start = theta[0]
                    return
            self.parent_ll, self.start = _wei_max_ll_parts(
                self.parts, self.box, start
            )
        else:
            self.bounds = (theta0 - 15.0, theta0 + 15.0)
            if not fit_parent:
                return
            if self.closed:
                ll = self._exponential(np.ones((1, data.x.size), bool))
                if np.isfinite(ll[0]):
                    self.parent_ll = float(ll[0])
                    return
            self.parent_ll = _exp_max_ll_parts(self.parts, self.bounds)

    def split_scores(self, left: NDArray) -> NDArray:
        """``ll(left child) + ll(right child)`` for each row of the
        boolean matrix ``left`` (candidates x node rows)."""
        out = np.empty(left.shape[0])
        step = max(1, _BLOCK // (2 * left.shape[1]))
        for i in range(0, left.shape[0], step):
            block = left[i : i + step]
            ll = self.child_lls(np.concatenate([block, ~block]))
            out[i : i + step] = ll[: block.shape[0]] + ll[block.shape[0] :]
        return out

    def child_lls(self, rows: NDArray) -> NDArray:
        """The maximised log-likelihood of each child, a row of the
        boolean matrix ``rows`` (children x node rows)."""
        if self.closed:
            if self.model == "weibull":
                ll = self._weibull(rows)[0]
            else:
                ll = self._exponential(rows)
        else:
            ll = np.full(rows.shape[0], np.nan)
        # The children the closed forms do not cover: the optimiser.
        for j in np.flatnonzero(~np.isfinite(ll)):
            parts = self._child_parts(rows[j])
            if self.model == "weibull":
                ll[j] = _wei_max_ll_parts(parts, self.box, self.start)[0]
            else:
                ll[j] = _exp_max_ll_parts(parts, self.bounds)
        return ll

    def _child_parts(self, rows: NDArray) -> tuple:
        # The node's likelihood terms at the child's rows.
        o, r, lc, i, t = (rows[m] for m in self.term_rows)
        x_o, n_o, x_r, n_r, x_l, n_l, x_il, x_ir, n_i, t_a, t_r, n_t = (
            self.parts
        )
        return (
            x_o[o],
            n_o[o],
            x_r[r],
            n_r[r],
            x_l[lc],
            n_l[lc],
            x_il[i],
            x_ir[i],
            n_i[i],
            t_a[t],
            t_r[t],
            n_t[t],
        )

    def _exponential(self, rows: NDArray) -> NDArray:
        # ll(theta) = r theta - exp(theta) E, maximised at log(r / E),
        # within the window; NaN where a child has no failure.
        weight = rows * self.n
        r = weight @ self.observed
        exposure = weight @ self.x
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.clip(np.log(r / exposure), *self.bounds)
            ll = r * theta - np.exp(theta) * exposure
        return np.where(r > 0, ll, np.nan)

    def _weibull(
        self,
        rows: NDArray,
        log_beta_bounds: "tuple[float, float]" = _LOG_BETA_BOUNDS,
    ) -> tuple[NDArray, NDArray]:
        # The profile log-likelihood in b = log(beta), with the scale at
        # its maximum A = alpha^beta = S(beta) / r, S = sum n x^beta:
        #   l(b) = r b - r log(S / r) + (beta - 1) L - r,
        # L = sum of n log x over the failures. It is concave in beta, so
        # its derivative g(b) = beta (r / beta - r S'/S + L) has one
        # root, bracketed by its sign and found by Newton steps (a
        # bisection whenever a step leaves the bracket). Sums of x^beta
        # are taken relative to the largest x, so they cannot overflow.
        weight = rows * self.n
        r = weight @ self.observed
        L = weight @ (self.observed * self.log_x)
        shift = self.log_x.max()
        centred = self.log_x - shift

        def moments(b: NDArray, idx: NDArray) -> tuple[NDArray, ...]:
            beta = np.exp(b)
            w = weight[idx] * np.exp(beta[:, None] * centred[None, :])
            s0 = w.sum(axis=1)
            s1 = (w * centred).sum(axis=1)
            s2 = (w * centred**2).sum(axis=1)
            return beta, s0, s1, s2

        def slope(b: NDArray, idx: NDArray) -> tuple[NDArray, NDArray]:
            # g(b) and its derivative, for the children idx
            beta, s0, s1, s2 = moments(b, idx)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = s1 / s0 + shift
                var = s2 / s0 - (s1 / s0) ** 2
                g = beta * (r[idx] / beta - r[idx] * mean + L[idx])
                dg = g - r[idx] - r[idx] * beta**2 * var
            return g, dg

        everyone = np.arange(r.size)
        lo = np.full(r.size, log_beta_bounds[0])
        hi = np.full(r.size, log_beta_bounds[1])
        # The window's edges: a root outside it puts the optimum there.
        at_lo = ~(slope(lo, everyone)[0] > 0)
        at_hi = slope(hi, everyone)[0] >= 0
        b = np.where(
            at_lo,
            lo,
            np.where(at_hi, hi, np.clip(self._start_log_beta(), lo, hi)),
        )
        active = np.flatnonzero(~(at_lo | at_hi))
        for _ in range(200):
            if active.size == 0:
                break
            g, dg = slope(b[active], active)
            # Shrink the bracket by the sign of the slope
            lo[active] = np.where(g > 0, b[active], lo[active])
            hi[active] = np.where(g > 0, hi[active], b[active])
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = b[active] - g / dg
            inside = (newton >= lo[active]) & (newton <= hi[active])
            step = np.where(inside, newton, 0.5 * (lo[active] + hi[active]))
            # At an exact root, stay there
            step = np.where(g == 0, b[active], step)
            done = (g == 0) | (
                np.abs(step - b[active]) <= 1e-14 * (1.0 + np.abs(b[active]))
            )
            b[active] = step
            active = active[~done]
        beta, s0, _, _ = moments(b, everyone)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_S = beta * shift + np.log(s0)
            log_r = np.log(r)
            ll = r * b - r * (log_S - log_r) + (beta - 1.0) * L - r
            log_alpha = (log_S - log_r) / beta
        good = (
            (r > 0)
            & np.isfinite(ll)
            & (log_alpha >= self.box[0][0])
            & (log_alpha <= self.box[0][1])
        )
        return np.where(good, ll, np.nan), np.column_stack([log_alpha, b])

    def _start_log_beta(self) -> float:
        # Newton starts from the node's optimum (the parent's shape)
        start = getattr(self, "start", None)
        return 0.0 if start is None else float(start[1])


# A leaf's Weibull shape window (see leaf_mle): far wider than the split
# search's, since a leaf is not compared with its siblings.
_LEAF_LOG_BETA_BOUNDS = (-10.0, 10.0)


def leaf_mle(data: SurpyvalData, model: str) -> "NDArray | None":
    r"""The maximum likelihood parameters of the working model fitted to
    a leaf's ``data``, found as :class:`_ChildLikelihoods` finds a child's:
    the exponential rate :math:`r / \sum n x` in closed form, the Weibull
    ``[alpha, beta]`` from the profile likelihood. The same maximum as
    ``Weibull.fit`` / ``Exponential.fit`` (to machine precision rather than
    an optimiser's tolerance), for the cost of a few array operations.

    The split search keeps the Weibull shape within ``beta`` 0.05 to 20,
    so that every child is scored over one window; a leaf is a model in
    its own right, and its shape is searched up to :math:`e^{\pm 10}`, as
    widely as ``Weibull.fit`` finds it (failures bunched together in a
    bootstrap sample give shapes in the hundreds).

    ``None`` where that does not apply, for the caller to fit the leaf in
    full: data other than observed and right-censored, a leaf without a
    failure, a Weibull whose shape reaches that wider window, where the
    likelihood may have no finite maximum, or a Weibull with fewer than
    two distinct failure times, which ``Weibull.fit`` refuses (#462).
    """
    if not _closed_form(data):
        return None
    theta0 = _exp_theta0(data)
    if theta0 is None:
        return None
    lik = _ChildLikelihoods(data, model, theta0, fit_parent=False)
    rows = np.ones((1, np.size(data.x)), bool)
    if model == "weibull":
        if np.unique(lik.x[lik.observed == 1]).size < 2:
            return None
        ll, theta = lik._weibull(rows, _LEAF_LOG_BETA_BOUNDS)
        log_alpha, log_beta = theta[0]
        if not (
            np.isfinite(ll[0])
            and _LEAF_LOG_BETA_BOUNDS[0] < log_beta < _LEAF_LOG_BETA_BOUNDS[1]
        ):
            return None
        return np.exp([log_alpha, log_beta])
    r = float(np.sum(lik.n * lik.observed))
    if r <= 0:
        return None
    return np.array([r / float(np.sum(lik.n * lik.x))])


# The degrees of freedom a split adds: the working model's parameters.
_SPLIT_DOF = {"exponential": 1, "weibull": 2}

# The smallest gain that counts as one: below it, the "gain" is only the
# optimiser's noise on data every partition of which scores the parent's
# log-likelihood (fully degenerate data, #185).
_GAIN_FLOOR = 1e-6


def parse_min_split_gain(
    min_split_gain: float | str, kind: str
) -> float | str:
    """Validate ``min_split_gain`` for a tree of ``kind``: a number at
    least 0, ``"aic"`` or ``"bic"``. Only the likelihood kinds use it, so a
    non-parametric tree accepts only the default 0."""
    if isinstance(min_split_gain, str):
        resolved = min_split_gain.lower()
        if resolved not in ("aic", "bic"):
            raise ValueError(
                f"min_split_gain={min_split_gain!r} is invalid. Must be a "
                "log-likelihood gain (a number at least 0), 'aic' or 'bic'."
            )
    elif (
        isinstance(min_split_gain, bool)
        or not isinstance(
            min_split_gain, (int, float, np.integer, np.floating)
        )
        or not (np.isfinite(min_split_gain) and min_split_gain >= 0)
    ):
        raise ValueError(
            f"min_split_gain must be a log-likelihood gain (a finite number "
            f"at least 0), 'aic' or 'bic'; got {min_split_gain!r}."
        )
    else:
        resolved = float(min_split_gain)  # type: ignore[assignment]
    if kind == "non-parametric" and resolved != 0:
        raise ValueError(
            "min_split_gain applies to the likelihood splits of "
            "kind='weibull' and kind='exponential'; a non-parametric "
            "tree's log-rank split is not a likelihood. Stop a "
            "non-parametric tree with selection='ctree' (and alpha_split) "
            "instead."
        )
    return resolved


def split_gain_threshold(
    min_split_gain: float | str, data: SurpyvalData, model: str
) -> float:
    """The log-likelihood gain a split of the node ``data`` must exceed:
    ``min_split_gain`` itself, or for ``"aic"`` the working model's
    degrees of freedom ``k``, for ``"bic"`` ``k log(d) / 2`` with ``d``
    the node's n-weighted failures (its units if it has none, as
    :meth:`~surpyval.univariate.information_criteria.InformationCriteriaMixin.bic`
    counts them). Never below the floor that tells a gain from the
    optimiser's noise."""
    if min_split_gain == "aic":
        threshold = float(_SPLIT_DOF[model])
    elif min_split_gain == "bic":
        d = float(np.sum(data.n * (data.c != 1)))
        if d <= 0:
            d = float(np.sum(data.n))
        threshold = _SPLIT_DOF[model] * np.log(d) / 2.0
    else:
        threshold = float(min_split_gain)
    return max(threshold, _GAIN_FLOOR)


def deviance_split(
    data: SurpyvalData,
    Z: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
    model: str = "exponential",
    min_split_gain: float | str = 0.0,
) -> tuple[int, float]:
    """
    Best ``(feature index, value)`` split by the deviance criterion
    under the full likelihood, with ``model`` selecting the working
    model: ``"exponential"`` (1 parameter; splits on rate) or
    ``"weibull"`` (2 parameters; splits on scale *and* shape).

    Mirrors ``log_rank_split``'s contract: candidates leaving either
    child with fewer than ``min_leaf_samples`` observations or fewer
    than ``min_leaf_failures`` event-informative observations (any
    observation that is not purely right censored, weighted by ``n``)
    are discarded, and ``(-1, -inf)`` is returned when no candidate
    survives.

    Each child's maximised log-likelihood is found as
    :class:`_ChildLikelihoods` describes: in closed form (exponential) or
    from the profile likelihood (Weibull) on observed and right-censored
    data, for every candidate of a feature at once; by the bounded
    optimiser otherwise.

    Parameters
    ----------
    data : SurpyvalData
        Survival data in the full xcnt data model: observed, left, right
        and interval censoring, with optional left and/or right
        truncation.
    Z : NDArray
        Covariate matrix, shape ``(n_samples, n_features)``.
    min_leaf_samples : int
        Minimum number of samples each child must keep.
    min_leaf_failures : int
        Minimum ``n``-weighted count of event-informative observations
        (``c != 1``) each child must keep.
    feature_indices_in : Iterable[int]
        Indices of the features considered for the split.
    model : {"exponential", "weibull"}, optional
        The working model whose maximised log-likelihood scores each
        candidate's children.
    min_split_gain : float, "aic" or "bic", optional
        The best cut is made only if it raises the maximised
        log-likelihood by more than this (see
        :func:`split_gain_threshold`); 0 (the default) asks only for a
        gain beyond the optimiser's noise.

    Returns
    -------
    tuple[int, float]
        The best feature index and split value, or ``(-1, -inf)`` if no
        valid split exists.
    """
    best_score = -np.inf
    best_u = -1
    best_v = -float("inf")

    event_weight = data.n * (data.c != 1)
    total_events = event_weight.sum()

    # All candidates -- and both children of each -- are scored over the
    # parent's search window (and, for the Weibull model, warm-started
    # from the parent's optimum), so their maximised log-likelihoods are
    # directly comparable and the split gain is non-negative by
    # likelihood additivity.
    theta0 = _exp_theta0(data)
    if theta0 is None:
        return best_u, best_v
    search = _ChildLikelihoods(data, model, theta0)
    parent_ll = search.parent_ll

    for u in feature_indices_in:
        Z_u = Z[:, u]
        values = _candidate_values(Z_u)
        # One row per candidate: the rows of its left child
        left = Z_u[None, :] <= values[:, None]
        n_left = left.sum(axis=1)
        events_left = left @ event_weight
        ok = (
            (n_left >= min_leaf_samples)
            & (left.shape[1] - n_left >= min_leaf_samples)
            & (events_left >= min_leaf_failures)
            & (total_events - events_left >= min_leaf_failures)
        )
        if not ok.any():
            continue
        score = np.full(values.size, -np.inf)
        score[ok] = search.split_scores(left[ok])
        k = int(np.argmax(score))
        if score[k] > best_score:
            best_score = float(score[k])
            best_u = int(u)
            best_v = float(values[k])

    # A split that does not improve on the parent's log-likelihood by a
    # meaningful margin carries no information (for fully degenerate
    # data every partition scores exactly the parent value), nor one
    # that gains less than min_split_gain asks (#189); stop rather than
    # split.
    threshold = split_gain_threshold(min_split_gain, data, model)
    if best_u != -1 and best_score <= parent_ll + threshold:
        return -1, -float("inf")

    return best_u, best_v
