"""
The joint Bayesian target of a parametric competing-risks model whose data
hold failures found in a window (left censored, ``c = -1``, or interval
censored, ``c = 2``).

A unit known to have failed from cause :math:`k` somewhere in
:math:`(a, b]` contributes the probability of exactly that: that the
first failure of any cause fell in the window *and* was cause :math:`k`'s,

.. math::
    \\int_a^b h_k(u)\\, S(u)\\, du
    = \\int_a^b f_k(u) \\prod_{j \\neq k} S_j(u)\\, du
    = \\mathrm{CIF}_k(b) - \\mathrm{CIF}_k(a),

where :math:`S = \\prod_j S_j` is the all-cause survival. It is the
cause's *sub-density* integrated over the window, not the difference of
cause :math:`k`'s own (net) distribution, :math:`F_k(b) - F_k(a)`, which
would ignore the other causes acting inside the window. Every cause's
parameters are in that integral, so the posterior no longer splits into a
fit a cause; this target holds them all at once, for the shared engines
(the grid for up to three free parameters, Laplace beyond).

The rest of the likelihood is the maximum-likelihood fit's
(``_interval_likelihood``): an exact failure from cause :math:`k` at
:math:`x` gives :math:`h_k(x) S(x)`, a unit still working at :math:`x`
gives :math:`S(x)`, and every unit seen only from its entry time
:math:`t_l` is divided by :math:`S(t_l)`; a left-censored row's window
runs from its entry (or from the start). The rows are prepared by that
module, so the two fits read the data in the same way.

Here the likelihood is evaluated at many parameter points at once (the
grid's lattice), so the window's integral is taken afresh at each point,
on the cause's own probability scale :math:`p = F_k(u)` at that point's
parameters, where the integrand is the other causes' survival
(bounded by 1): a 48-point Gauss-Legendre rule with its nodes gathered at
both ends, as the Bayesian incidence's draws are taken. A window too far
in the cause's tail for that scale to resolve is integrated in time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.target import Target
from surpyval.univariate.competing_risks.parametric import (
    _interval_likelihood as joint,
)
from surpyval.utils.warnings import ignoring_warnings

#: The rule on a window's probability scale (nodes gathered at both
#: ends by ``p = 3 v^2 - 2 v^3``), and on its time scale.
_V, _W = np.polynomial.legendre.leggauss(48)
_V = 0.5 * (_V + 1.0)
_P_NODES = 3 * _V**2 - 2 * _V**3
_P_WEIGHTS = 0.5 * _W * 6 * _V * (1 - _V)
_T_NODES = _V
_T_WEIGHTS = 0.5 * _W
#: Parameter points a block of the vectorised likelihood.
_BLOCK = 2048


def _H(dist: Any, lo: float, x: npt.NDArray, p: list) -> npt.NDArray:
    """The cumulative hazard at ``x`` (broadcast against the parameters
    ``p``), 0 at or below the support's lower end ``lo``."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x) & (x > lo)
    inside = (lo + 1.0) if np.isfinite(lo) else 0.0
    safe = np.where(mask, x, inside)
    with np.errstate(all="ignore"):
        out = np.asarray(dist.Hf(safe, *p), dtype=float)
    return np.where(mask, out, 0.0)


def window_log_probability(
    i: int,
    a: npt.NDArray,
    b: npt.NDArray,
    dists: list,
    los: list,
    params: list,
) -> npt.NDArray:
    """
    The log probability that a unit fails first from cause ``i`` in each
    window ``(a, b]``, :math:`\\log \\int_a^b f_i(u) \\prod_{j \\neq i}
    S_j(u)\\,du`, at ``N`` parameter points at once: ``params[j]`` is
    cause ``j``'s parameters, each an ``(N, 1)`` column. Returns
    ``(N, len(a))``.
    """
    dist = dists[i]
    p_i = [q[:, :, None] for q in params[i]]  # (N, 1, 1)
    a = np.asarray(a, dtype=float)[None, :]
    b = np.asarray(b, dtype=float)[None, :]
    lo_i = los[i]
    from_start = ~np.isfinite(a) | (a <= lo_i)
    with np.errstate(all="ignore"):
        Pa = np.where(
            from_start,
            0.0,
            np.asarray(
                dist.ff(np.where(from_start, b, a), *params[i]), dtype=float
            ),
        )
        Pb = np.asarray(dist.ff(b, *params[i]), dtype=float)
    width = Pb - Pa  # (N, R)

    def others(u: npt.NDArray) -> npt.NDArray:
        H = np.zeros(u.shape)
        for j, d in enumerate(dists):
            if j != i:
                H = H + _H(d, los[j], u, [q[:, :, None] for q in params[j]])
        return np.exp(-H)

    # On the cause's probability scale: int_{Pa}^{Pb} S_others(F^-1(p)) dp.
    P = Pa[..., None] + width[..., None] * _P_NODES  # (N, R, 48)
    with np.errstate(all="ignore"):
        U = np.asarray(dist.qf(np.clip(P, 0.0, 1.0), *p_i), dtype=float)
    g = others(U)
    g = np.where(np.isfinite(g), g, 0.0)
    on_p = width * np.sum(g * _P_WEIGHTS, axis=-1)
    # A window the probability scale cannot resolve: in time,
    # int_a^b f_i(u) S_others(u) du.
    coarse = ~(width > 1e-13 * np.maximum(Pb, 1e-300))
    out = np.where(coarse, 0.0, on_p)
    if coarse.any():
        start = np.where(from_start, np.maximum(lo_i, b - 1.0), a)
        start = np.broadcast_to(start, coarse.shape)
        bb = np.broadcast_to(b, coarse.shape)
        T = start[..., None] + (bb - start)[..., None] * _T_NODES
        with np.errstate(all="ignore"):
            f = np.asarray(dist.df(T, *p_i), dtype=float)
        f = np.where(np.isfinite(f), f, 0.0)
        on_t = (bb - start) * np.sum(f * others(T) * _T_WEIGHTS, axis=-1)
        out = np.where(coarse, on_t, out)
    with np.errstate(all="ignore"):
        return np.where(out > 0, np.log(np.where(out > 0, out, 1.0)), -np.inf)


class JointCompetingRisksTarget(Target):
    """The likelihood of every cause's parameters at once (see the module
    description), for the shared Bayesian engines.

    Parameters
    ----------
    causes : list
        The causes, in order.
    dists : list
        Each cause's distribution.
    x, c, n, e, t
        The data as ``ParametricCompetingRisks._validate`` returns them.
    mle : ParametricCompetingRisks or None
        The joint maximum-likelihood fit to the same data.
    """

    tag = "competing_risks_joint"
    family = "competing-risks model"
    unit_name = "row"

    def __init__(
        self,
        causes: list,
        dists: list,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        e: npt.NDArray,
        t: npt.NDArray,
        mle: Any = None,
    ) -> None:
        self.causes = list(causes)
        self.dists = list(dists)
        self.prepared = joint.prepare(self.causes, self.dists, x, c, n, e, t)
        self.mle = mle
        self.data = (x, c, n, e, t)
        self.fixed = {}
        self.dist = None
        labels = [str(k) for k in self.causes]
        if len(set(labels)) < len(labels):
            labels = [repr(k) for k in self.causes]
        self.labels = labels
        self.names = []
        self.slices = []
        start = 0
        for label, d in zip(labels, self.dists):
            self.names += [f"{label}.{p}" for p in d.parameter_names]
            self.slices.append(slice(start, start + d.k))
            start += d.k
        self.bounds = []
        for d in self.dists:
            self.bounds += list(d.bounds)

    def name_of(self, cause: Any, parameter: str) -> str:
        """The target's name of a cause's parameter."""
        return f"{self.labels[self.causes.index(cause)]}.{parameter}"

    # -- the likelihood ------------------------------------------------------

    def _row_log_lik(self, theta: npt.NDArray) -> list:
        """Each group of rows' log-likelihood a row (not yet times its
        count) at ``(N, k)`` parameter points: ``[(values, counts)]``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        data = self.prepared
        dists, los = self.dists, data.los
        params = [
            [theta[:, s][:, j : j + 1] for j in range(d.k)]
            for s, d in zip(self.slices, dists)
        ]
        K = len(dists)

        def H_all(view_points: list) -> npt.NDArray:
            # sum_j H_j at the points (every cause's view of them)
            total: Any = 0.0
            for j in range(K):
                pts = view_points[j]
                x = np.where(pts.mask, pts.safe, -np.inf)
                total = total + _H(dists[j], los[j], x[None, :], params[j])
            return np.asarray(total) * np.ones((theta.shape[0], 1))

        out = []
        with np.errstate(all="ignore"):
            if data.n_censored.size:
                out.append((-H_all(data.censored_at), data.n_censored))
            if data.n_entry.size:
                out.append((H_all(data.entry_at), data.n_entry))
            for i, rows in enumerate(data.rows):
                if rows.n_exact.size:
                    pts = rows.exact_at[i]
                    log_f = np.asarray(
                        dists[i].log_df(pts.safe[None, :], *params[i]),
                        dtype=float,
                    )
                    log_f = np.where(pts.mask[None, :], log_f, -np.inf)
                    # f_i S_others = h_i S: log f_i - sum_{j != i} H_j
                    others = H_all(rows.exact_at) - _H(
                        dists[i],
                        los[i],
                        np.where(pts.mask, pts.safe, -np.inf)[None, :],
                        params[i],
                    )
                    out.append((log_f - others, rows.n_exact))
                if rows.n_window.size:
                    out.append(
                        (
                            window_log_probability(
                                i, rows.a, rows.b, dists, los, params
                            ),
                            rows.n_window,
                        )
                    )
        return out

    def log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        total = np.empty(theta.shape[0])
        for s in range(0, theta.shape[0], _BLOCK):
            block = theta[s : s + _BLOCK]
            ll = np.zeros(block.shape[0])
            with np.errstate(all="ignore"):
                for values, counts in self._row_log_lik(block):
                    terms = np.where(counts[None, :] > 0, values, 0.0)
                    ll = ll + terms @ counts
            total[s : s + _BLOCK] = ll
        return np.where(np.isfinite(total), total, -np.inf)

    def neg_ll(self, theta: npt.NDArray) -> float:
        return -float(self.log_lik(np.asarray(theta, dtype=float))[0])

    def _units_check(self) -> None:
        if self.prepared.n_entry.size:
            raise ValueError(
                "WAIC and leave-one-out need each unit's log-likelihood; "
                "with left truncation (tl) a joint competing-risks fit "
                "does not split its entry terms by unit. Compare fits "
                "without tl, or by their evidence."
            )

    def unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        """Each group of identical rows' log-likelihood a row (right
        censored, then each cause's exact and window rows)."""
        self._units_check()
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.concatenate([v for v, _ in self._row_log_lik(theta)], axis=1)

    def unit_counts(self) -> npt.NDArray:
        self._units_check()
        data = self.prepared
        parts = [data.n_censored] if data.n_censored.size else []
        for rows in data.rows:
            if rows.n_exact.size:
                parts.append(rows.n_exact)
            if rows.n_window.size:
                parts.append(rows.n_window)
        return np.concatenate(parts)

    # -- the maximum-likelihood fit -----------------------------------------

    def estimates(self) -> npt.NDArray:
        return np.concatenate(
            [
                np.asarray(self.mle.models[k].params, dtype=float).ravel()
                for k in self.causes
            ]
        )

    def covariance(self) -> npt.NDArray:
        return np.asarray(self.mle._joint["covariance"], dtype=float)

    def maximum(self) -> str:
        if self.mle is None:
            return "unknown"
        return str(getattr(self.mle.models[self.causes[0]], "maximum", ""))

    def mle_interval(self, name: str, alpha_ci: float) -> tuple | None:
        if self.mle is None:
            return None
        i = self.names.index(name)
        for k, s, d in zip(self.causes, self.slices, self.dists):
            if s.start <= i < s.stop:
                try:
                    with ignoring_warnings(), np.errstate(all="ignore"):
                        ci = np.asarray(
                            self.mle.models[k].param_cb(
                                d.parameter_names[i - s.start],
                                alpha_ci=alpha_ci,
                            ),
                            dtype=float,
                        ).reshape(-1)
                except (ValueError, TypeError, KeyError, AttributeError):
                    return None
                if ci.size == 2 and np.all(np.isfinite(ci)):
                    return float(ci[0]), float(ci[1])
        return None

    def with_params(self, theta: npt.NDArray) -> Any:
        from surpyval.univariate.competing_risks.parametric import (
            parametric_competing_risks as pcr,
        )

        theta = np.asarray(theta, dtype=float).ravel()
        return pcr.ParametricCompetingRisks.from_fitted(
            {
                k: d.from_params(theta[s])
                for k, s, d in zip(self.causes, self.slices, self.dists)
            }
        )

    def check(self, bound: Any) -> None:
        for j in bound.spec.joint:
            raise ValueError(
                "With failures found in a window (c = -1 or 2) the causes "
                "are fitted together, and their priors are taken a "
                "parameter at a time ({cause: {'beta': ...}} or one set "
                "for every cause); a joint prior "
                f"({type(j).__name__}) is not taken there. State it a "
                "parameter at a time."
            )

    def describe_data(self) -> str:
        n = float(np.sum(np.asarray(self.data[2], dtype=float)))
        return f"{n:g} units, {len(self.causes)} causes fitted together"

    def to_dict(self) -> dict:
        raise ValueError(
            "The joint competing-risks target is saved with its "
            "BayesianCompetingRisks (to_dict), not alone."
        )


def to_target_priors(
    priors_by_cause: dict, target: JointCompetingRisksTarget
) -> dict:
    """Each cause's per-parameter priors under the target's names."""
    from collections.abc import Mapping

    from surpyval.bayes.prior import JointPrior

    out: dict = {}
    for k in target.causes:
        p = priors_by_cause.get(k)
        if p is None:
            continue
        if isinstance(p, JointPrior) or not isinstance(p, Mapping):
            raise ValueError(
                f"Cause {k!r}: with failures found in a window (c = -1 or "
                "2) the causes are fitted together, and their priors are "
                "taken a parameter at a time ({'beta': ...}); a joint "
                f"prior ({type(p).__name__}) is not taken there. State "
                "it a parameter at a time."
            )
        names = list(target.dists[target.causes.index(k)].parameter_names)
        for name, value in p.items():
            if name not in names:
                raise ValueError(
                    f"Cause {k!r} has no parameter {name!r}; its "
                    f"parameters are {', '.join(names)}."
                )
            out[target.name_of(k, name)] = value
    return out


__all__ = [
    "JointCompetingRisksTarget",
    "to_target_priors",
    "window_log_probability",
]
