"""
Bayesian parametric competing risks, by composition.

With a distribution for each cause's hazard, the likelihood of
competing-risks data factorises: a unit that failed from cause ``k``
contributes cause ``k``'s density and every other cause's survival, and a
unit still working contributes every cause's survival. So, with priors
stated a cause at a time, the causes' posteriors are independent, and
each is the ordinary Bayesian fit of that cause's distribution with the
other causes' failures right censored (``dist.fit_bayes``). Nothing new
is computed for the posterior; this module combines the causes' fits.

What a combination needs:

- **The predictions for a new unit** (``sf``, ``ff``, ``cif``, ``qf``):
  the posterior predictive. Because the causes' posteriors are
  independent, the expected all-cause survival is the product of the
  causes' expected survivals, and a cause's expected cumulative incidence
  is the integral of its expected density against the other causes'
  expected survivals, so these are exact, from each cause's own
  predictive functions.
- **Credible intervals** on a cause's cumulative incidence, the
  all-cause survival or an all-cause B-life: the function at
  ``DRAWS`` joint draws, one from each cause's posterior (independent
  draws are joint draws), with a fixed seed so that the numbers repeat.

Failures found in a window (``c`` of -1 or 2) break the factorisation:
such a unit's likelihood holds every cause's parameters. Those data are
fitted on the joint target of :mod:`._joint_bayes` instead
(:func:`fit_joint`), and the predictions average over its points.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.prior import quantity_at
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    check_alpha_ci,
    check_option,
    unknown_cause_error,
    warn_outside_unit_interval,
)
from surpyval.utils.warnings import ignoring_warnings

#: The joint draws behind every credible interval of a combination.
DRAWS = 4000
#: Their seed: cause ``j`` (in ``causes`` order) draws with ``SEED + j``.
SEED = 20261126
#: Gauss-Legendre nodes of a draw's cumulative incidence, on the cause's
#: probability scale.
#: The nodes are mapped by ``p = 3 v^2 - 2 v^3``, which gathers them at
#: both ends, where the integrand can have an infinite slope (a cause's
#: hazard against another's of a different shape).
_V, _W = np.polynomial.legendre.leggauss(48)
_V = 0.5 * (_V + 1.0)
GAUSS_NODES = 3 * _V**2 - 2 * _V**3
GAUSS_WEIGHTS = 0.5 * _W * 6 * _V * (1 - _V)
#: Points of the predictive cumulative incidence's grid, a cause.
_GRID = 3000

_CRI_ON = ("cif", "sf", "ff")


def _plain(fit: Any) -> dict[str, npt.NDArray]:
    """A cause's posterior as columns ``{name: (N,)}``."""
    P = fit.posterior.points
    return {n: P[:, i] for i, n in enumerate(fit.posterior.parameter_names)}


def _levels(alpha_ci: float, bound: str) -> list[float]:
    check_alpha_ci(alpha_ci)
    check_option("bound", bound, ("two-sided", "lower", "upper"))
    if bound == "two-sided":
        return [alpha_ci / 2, 1 - alpha_ci / 2]
    return [alpha_ci] if bound == "lower" else [1 - alpha_ci]


def _shape(values: npt.NDArray, levels: list[float], shape: tuple) -> Any:
    """Quantiles of ``values`` (draws on the first axis) at ``levels``,
    in ``shape`` with a last ``[lower, upper]`` axis for two."""
    out = [np.reshape(np.quantile(values, q, axis=0), shape) for q in levels]
    return np.stack(out, axis=-1) if len(out) == 2 else out[0]


class BayesianCompetingRisks(SerialisableMixin):
    """
    A Bayesian parametric competing-risks model
    (``ParametricCompetingRisks.fit_bayes``): a distribution for each
    cause's hazard, each with its own posterior.

    The likelihood factorises over the causes, so with priors stated a
    cause at a time their posteriors are independent, and each is the
    :class:`~surpyval.bayes.BayesianParametric` of its cause, fitted with
    the other causes' failures right censored, in ``models``.

    The predictions for a new unit (``sf``, ``ff``, ``hf``, ``Hf``,
    ``df``, ``cif``, ``qf``) are posterior predictive, exact from the
    causes' own predictive functions. Credible intervals (``cri``,
    ``quantile_cri``) come from ``DRAWS`` joint draws, one from each
    cause's posterior, with a fixed seed.

    With failures found in a window (``c`` of -1 or 2) the causes are
    fitted together (``joint`` is True): ``posterior`` is the joint
    posterior of every cause's parameters (named ``cause.parameter``),
    each of ``models`` is its cause's marginal, the predictions are the
    posterior means over the joint posterior's points (a cause's
    incidence at each point by the 48-point rule of :meth:`cif_draws`,
    to about :math:`10^{-8}`), and the joint draws are drawn from the
    joint posterior.

    Attributes
    ----------
    models : dict
        Each cause's Bayesian fit, by cause.
    causes : list
        The causes, in order.
    mle : ParametricCompetingRisks or None
        The maximum-likelihood fit to the same data.

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.univariate.competing_risks import (
    ...     ParametricCompetingRisks,
    ... )
    >>> x = [12, 25, 31, 40, 44, 52, 60, 71, 80, 80]
    >>> e = ["wear", "crack", "wear", "wear", "crack", "wear", "crack",
    ...      "wear", None, None]
    >>> model = ParametricCompetingRisks.fit_bayes(
    ...     x, e, dist=sp.Exponential)
    >>> sorted(model.models)
    ['crack', 'wear']
    >>> lo, hi = model.cri(50.0, "wear")
    >>> bool(lo < model.cif(50.0, "wear") < hi)
    True
    """

    model_tag = "BayesianCompetingRisks"

    def __init__(
        self,
        models: dict,
        mle: Any = None,
        *,
        posterior: Any = None,
        columns: dict | None = None,
        result: Any = None,
    ) -> None:
        if len(models) == 0:
            raise ValueError("At least one cause's fit is required.")
        self.models = {
            (k.item() if isinstance(k, np.generic) else k): m
            for k, m in models.items()
        }
        self.causes = ordered_labels(self.models)
        self.mle = mle
        #: The joint posterior of every cause's parameters, where the
        #: causes were fitted together (failures found in a window); else
        #: ``None`` (the causes' posteriors are independent).
        self.posterior = posterior
        #: Each cause's columns of the joint posterior, ``{cause: [i,
        #: ...]}`` in its distribution's parameter order.
        self.columns = None if columns is None else dict(columns)
        #: The joint fit (a :class:`~surpyval.bayes.BayesianResult`: its
        #: ``diagnostics``, ``waic``, ``loo``), where it was fitted here.
        self.result = result
        if posterior is not None:
            kind = result.method if result is not None else posterior.kind
            self.method = {k: kind for k in self.causes}
        else:
            self.method = {k: self.models[k].method for k in self.causes}
        self._draws: dict | None = None
        self._points: dict | None = None

    @property
    def joint(self) -> bool:
        """Whether the causes were fitted together, so that their
        posteriors are dependent."""
        return self.posterior is not None

    # -- causes -------------------------------------------------------------

    def _check_event(self, event: Any) -> None:
        if event not in self.models:
            raise unknown_cause_error(event, self.causes)

    def _fits(self) -> list:
        return [self.models[k] for k in self.causes]

    # -- the posterior predictive ---------------------------------------------

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        The probability that a new unit has failed from no cause by
        ``x`` (``event=None``): the product of the causes' predictive
        survivals, which is exact since their posteriors are
        independent. ``event=k`` is cause ``k``'s net survival (its own
        predictive ``sf``, as if the other causes were removed).
        """
        x = np.asarray(x, dtype=float)
        if event is not None:
            self._check_event(event)
            return np.asarray(self.models[event].sf(x), dtype=float)
        if self.joint:
            return self._expect(lambda th: self._sf_at(th, x), x.shape)
        out = np.ones(x.shape)
        for fit in self._fits():
            out = out * np.asarray(fit.sf(x), dtype=float)
        return out

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The predictive cumulative hazard, ``-log sf``: the causes'
        predictive cumulative hazards summed (``event=None``), or one
        cause's."""
        x = np.asarray(x, dtype=float)
        if event is not None:
            self._check_event(event)
            return np.asarray(self.models[event].Hf(x), dtype=float)
        if self.joint:
            with np.errstate(all="ignore"):
                return -np.log(np.asarray(self.sf(x), dtype=float))
        return sum(  # type: ignore[return-value]
            np.asarray(fit.Hf(x), dtype=float) for fit in self._fits()
        )

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The predictive probability of having failed by ``x``: from any
        cause (``event=None``), ``-expm1(-Hf)``; ``event=k`` is cause
        ``k``'s net failure probability. For the probability of failing
        *from* cause ``k`` while the others act, use :meth:`cif`."""
        x = np.asarray(x, dtype=float)
        if event is not None:
            self._check_event(event)
            return np.asarray(self.models[event].ff(x), dtype=float)
        with np.errstate(all="ignore"):
            return -np.expm1(-np.asarray(self.Hf(x), dtype=float))

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The predictive hazard: the causes' predictive hazards summed
        (``event=None``; exact, as ``-d log sf / dx``), or one cause's."""
        x = np.asarray(x, dtype=float)
        if event is not None:
            self._check_event(event)
            return np.asarray(self.models[event].hf(x), dtype=float)
        if self.joint:
            # -d log E[S] / dx = E[S h] / E[S]
            with np.errstate(all="ignore"):
                return np.asarray(self.df(x), dtype=float) / np.asarray(
                    self.sf(x), dtype=float
                )
        return sum(  # type: ignore[return-value]
            np.asarray(fit.hf(x), dtype=float) for fit in self._fits()
        )

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The predictive density of the time to the first failure from
        any cause (``event=None``), ``sf * hf``; or cause ``k``'s net
        density."""
        x = np.asarray(x, dtype=float)
        if event is not None:
            self._check_event(event)
            return np.asarray(self.models[event].df(x), dtype=float)
        if self.joint:
            return self._expect(lambda th: self._df_at(th, x), x.shape)
        with np.errstate(all="ignore"):
            s = np.asarray(self.sf(x), dtype=float)
            return np.where(s == 0, 0.0, s * np.asarray(self.hf(x)))

    def _grid(self, x: npt.NDArray) -> npt.NDArray:
        """The times of the predictive incidence's integral: each cause's
        plug-in quantiles (at the posterior medians) on a fine
        logit-spaced set of probabilities, which follows every cause's
        support and scale (and the times ``x``, where given)."""
        p = 1.0 / (1.0 + np.exp(-np.linspace(-28.0, 28.0, _GRID)))
        parts = [x[np.isfinite(x)]]
        for fit in self._fits():
            with ignoring_warnings(), np.errstate(all="ignore"):
                q = np.asarray(fit.at_params().qf(p), dtype=float)
            parts.append(q[np.isfinite(q)])
        return np.unique(np.concatenate(parts))

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        The predictive cumulative incidence of cause ``event``: the
        probability that a new unit has failed from that cause by ``x``,
        while the others act. ``event=None`` is the all-cause
        :meth:`ff`.

        It is the integral of cause ``k``'s predictive failure
        probability against the other causes' predictive survival,
        :math:`\\int_0^x \\prod_{j \\neq k} \\bar S_j(u)\\, d\\bar F_k(u)`,
        exact for the posterior predictive since the causes' posteriors
        are independent; taken by the trapezoidal rule on the scale of
        :math:`\\bar F_k` over a grid of 3,000 times a cause (the causes'
        quantiles from :math:`10^{-12}` to :math:`1 - 10^{-12}`, and
        ``x``), to about :math:`10^{-6}`. The causes' incidences sum to
        :meth:`ff` to that accuracy. ``x = inf`` gives the probability
        that the cause is the one a unit fails from.
        """
        x = np.asarray(x, dtype=float)
        if event is None:
            return np.asarray(self.ff(x), dtype=float)
        self._check_event(event)
        if self.joint:
            return self._expect(
                lambda th: self._cif_at(th, x.ravel(), event), x.shape
            )
        flat = x.ravel()
        out = np.full(flat.shape, np.nan)
        ok = ~np.isnan(flat)
        if not ok.any():
            return out.reshape(x.shape)
        grid, F, g, cum, tail = self._incidence_table(event)
        at = flat[ok]
        fin = np.isfinite(at)
        vals = np.where(np.isposinf(at), tail, 0.0)
        if fin.any():
            t = at[fin]
            with ignoring_warnings(), np.errstate(all="ignore"):
                F_t = np.asarray(self.models[event].ff(t), dtype=float)
                g_t = np.ones(t.shape)
                for k in self.causes:
                    if k != event:
                        g_t = g_t * np.asarray(
                            self.models[k].sf(t), dtype=float
                        )
            F_t = np.where(np.isfinite(F_t), F_t, 0.0)
            g_t = np.where(np.isfinite(g_t), g_t, 0.0)
            # The table's integral to the grid time at or before t, and
            # the trapezoid from there to t; before the grid, F(t) g(t).
            j = np.searchsorted(grid, t, side="right") - 1
            jj = np.clip(j, 0, grid.size - 1)
            part = cum[jj] + 0.5 * (g[jj] + g_t) * (F_t - F[jj])
            vals[fin] = np.where(j < 0, F_t * g_t, part)
        out[ok] = vals
        return out.reshape(x.shape)

    def _incidence_table(self, event: Any) -> tuple:
        """Cause ``event``'s predictive incidence on a fixed grid of
        times (:meth:`_grid`, without the query times, so a query's value
        does not depend on the others asked with it): the grid, the
        cause's predictive failure probability and the other causes'
        predictive survival on it, the integral to each grid time, and
        the integral to infinity. Computed once."""
        cache = self.__dict__.setdefault("_tables", {})
        if event in cache:
            return cache[event]
        grid = self._grid(np.zeros(0))
        fit = self.models[event]
        with ignoring_warnings(), np.errstate(all="ignore"):
            F = np.asarray(fit.ff(grid), dtype=float)
            g = np.ones(grid.shape)
            g_end = 1.0
            for k in self.causes:
                if k != event:
                    g = g * np.asarray(self.models[k].sf(grid), dtype=float)
                    g_end *= float(
                        np.asarray(self.models[k].sf(np.inf), dtype=float)
                    )
            F_end = float(np.asarray(fit.ff(np.inf), dtype=float))
        F = np.where(np.isfinite(F), F, 0.0)
        g = np.where(np.isfinite(g), g, 0.0)
        # The mass before the grid's first time, where the others'
        # survival is (very nearly) 1.
        cum = F[0] * g[0] + np.concatenate(
            [[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(F))]
        )
        if not np.isfinite(g_end):
            g_end = 0.0
        if not np.isfinite(F_end):
            F_end = 1.0
        tail = cum[-1] + 0.5 * (g[-1] + g_end) * max(F_end - F[-1], 0.0)
        cache[event] = (grid, F, g, cum, tail)
        return cache[event]

    def probability_of_cause(self, event: Any) -> float:
        """The predictive probability that a new unit fails from
        ``event``: :meth:`cif` at ``inf``."""
        return float(self.cif(np.inf, event))

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        The predictive quantile: the time by which a fraction ``p`` of
        new units will have failed from any cause (``event=None``), all
        uncertainty considered, the root of :meth:`Hf` ``= -log(1 - p)``;
        or cause ``k``'s net predictive quantile. This is not the
        posterior median of the B-life, which :meth:`quantile_cri` gives
        (``point=True``) with its interval.
        """
        u = np.asarray(p, dtype=float)
        if event is not None:
            self._check_event(event)
            return np.asarray(self.models[event].qf(u), dtype=float)
        flat = np.atleast_1d(u).ravel()
        outside = warn_outside_unit_interval(flat)
        out = np.full(flat.shape, np.nan)
        K = len(self.causes)
        use = ~outside & ~np.isnan(flat)
        if not use.any():
            return out.reshape(u.shape)

        def net_q(q: npt.NDArray) -> npt.NDArray:
            with ignoring_warnings(), np.errstate(all="ignore"):
                return np.min(
                    np.stack(
                        [
                            np.asarray(f.qf(q), dtype=float).ravel()
                            for f in self._fits()
                        ]
                    ),
                    axis=0,
                )

        p_u = flat[use]
        ends = (p_u == 0) | (p_u == 1)
        res = np.full(p_u.shape, np.nan)
        if ends.any():
            res[ends] = net_q(p_u[ends])
        inner = ~ends
        if inner.any():
            q = p_u[inner]
            # The root lies between the causes' net quantiles: no later
            # than the earliest at p, no earlier than the earliest at
            # 1 - (1 - p)^(1/K). (With the causes fitted together their
            # predictive survivals are not independent, and the bound is
            # the union bound instead: ff <= the sum of the causes' net
            # ff, at most p where each is at most p / K.)
            hi = net_q(q)
            lo_p = q / K if self.joint else -np.expm1(np.log1p(-q) / K)
            lo = net_q(lo_p)
            res[inner] = self._root_Hf(lo, hi, -np.log1p(-q))
        out[use] = res
        return out.reshape(u.shape)

    def _root_Hf(
        self, lo: npt.NDArray, hi: npt.NDArray, target: npt.NDArray
    ) -> npt.NDArray:
        """The times where the predictive all-cause cumulative hazard
        reaches ``target``, each between its ``lo`` and ``hi``: the
        Illinois form of regula falsi, every root at once but each by its
        own steps (so a root does not depend on the others asked with
        it), to 1e-12 of the time."""
        out = np.where(hi <= lo, hi, np.nan)
        out = np.where(
            ~np.isfinite(hi), np.where(np.isfinite(lo), np.nan, np.inf), out
        )
        go = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if not go.any():
            return out
        a, b, tg = lo[go], hi[go], target[go]

        def gap(t: npt.NDArray) -> npt.NDArray:
            with ignoring_warnings(), np.errstate(all="ignore"):
                return np.asarray(self.Hf(t), dtype=float).ravel() - tg_now

        tg_now = tg
        fa, fb = gap(a), gap(b)
        side = np.zeros(a.shape, dtype=int)
        done = (fa >= 0) | (fb <= 0)
        root = np.where(fa >= 0, a, np.where(fb <= 0, b, np.nan))
        for _ in range(200):
            act = ~done
            if not act.any():
                break
            idx = np.flatnonzero(act)
            A, B, FA, FB = a[idx], b[idx], fa[idx], fb[idx]
            with np.errstate(all="ignore"):
                c = B - FB * (B - A) / (FB - FA)
            bad = ~np.isfinite(c) | (c <= A) | (c >= B)
            c = np.where(bad, 0.5 * (A + B), c)
            tg_now = tg[idx]
            fc = gap(c)
            left = fc > 0  # the root is in (A, c]
            # update the bracket, halving the stale end's value (Illinois)
            b[idx] = np.where(left, c, B)
            fb[idx] = np.where(left, fc, FB)
            a[idx] = np.where(left, A, c)
            fa[idx] = np.where(left, FA, fc)
            s_new = np.where(left, 1, -1)
            fa[idx] = np.where(left & (side[idx] == 1), fa[idx] / 2, fa[idx])
            fb[idx] = np.where(~left & (side[idx] == -1), fb[idx] / 2, fb[idx])
            side[idx] = s_new
            width = b[idx] - a[idx]
            fin = (width <= 1e-12 * np.maximum(np.abs(b[idx]), 1.0)) | (
                fc == 0
            )
            root[idx] = np.where(fin, np.where(fc == 0, c, b[idx]), root[idx])
            done[idx] = fin
        root = np.where(done, root, 0.5 * (a + b))
        out[go] = root
        return out

    # -- credible intervals, from joint draws -------------------------------

    def _joint(self) -> dict:
        """``DRAWS`` draws of each cause's parameters (independent across
        causes, so jointly drawn), as ``{cause: {name: (DRAWS,)}}``."""
        if self._draws is None and self.joint:
            D = self.posterior.draws(DRAWS, random_state=SEED)
            self._draws = self._split(D)
        if self._draws is None:
            out = {}
            for j, k in enumerate(self.causes):
                fit = self.models[k]
                D = fit.posterior.draws(DRAWS, random_state=SEED + j)
                out[k] = {
                    n: D[:, i]
                    for i, n in enumerate(fit.posterior.parameter_names)
                }
            self._draws = out
        return self._draws

    def _split(self, P: npt.NDArray) -> dict:
        """Rows of the joint posterior's parameters as ``{cause: {name:
        (N,)}}``."""
        assert self.columns is not None
        out = {}
        for k in self.causes:
            names = self.models[k].dist.parameter_names
            out[k] = {n: P[:, i] for n, i in zip(names, self.columns[k])}
        return out

    def _support(self) -> tuple[dict, npt.NDArray]:
        """The joint posterior's points of positive weight, by cause, and
        their weights."""
        if self._points is None:
            w = self.posterior.weights
            keep = w > 0
            self._points = {
                "theta": self._split(self.posterior.points[keep]),
                "w": w[keep] / w[keep].sum(),
            }
        return self._points["theta"], self._points["w"]

    def _expect(self, values_at: Any, shape: tuple) -> npt.NDArray:
        """The posterior mean of a function of every cause's parameters
        (``values_at(theta) -> (N, M)``) over the joint posterior's
        points, a block of points at a time."""
        theta, w = self._support()
        total = None
        step = 2048
        for s in range(0, w.size, step):
            part = {
                k: {n: v[s : s + step] for n, v in d.items()}
                for k, d in theta.items()
            }
            with ignoring_warnings(), np.errstate(all="ignore"):
                vals = np.asarray(values_at(part), dtype=float)
            ws = w[s : s + step]
            vals = np.where(ws[:, None] > 0, vals, 0.0)
            here = np.ascontiguousarray((vals * ws[:, None]).T).sum(axis=1)
            total = here if total is None else total + here
        assert total is not None
        return total.reshape(shape)

    def _sf_at(self, theta: dict, x: npt.NDArray) -> npt.NDArray:
        flat = np.ravel(x)[None, :]
        out = np.ones((len(next(iter(theta[self.causes[0]].values()))), 1))
        for k in self.causes:
            out = out * self._at(k, "R", flat, 1, theta)
        return np.where(np.isnan(flat), np.nan, out)

    def _df_at(self, theta: dict, x: npt.NDArray) -> npt.NDArray:
        # f(x) = S(x) sum_k h_k(x) = sum_k f_k(x) prod_{j != k} S_j(x)
        flat = np.ravel(x)[None, :]
        R = {k: self._at(k, "R", flat, 1, theta) for k in self.causes}
        out: Any = 0.0
        for k in self.causes:
            fit = self.models[k]
            names = fit.dist.parameter_names
            p = [theta[k][n][:, None] for n in names]
            f = np.asarray(fit.dist.df(flat, *p), dtype=float)
            term = np.where(np.isfinite(f), f, 0.0)
            for j in self.causes:
                if j != k:
                    term = term * R[j]
            out = out + term
        return np.asarray(out)

    def _at(
        self,
        cause: Any,
        kind: str,
        value: Any,
        extra: int,
        theta: dict | None = None,
    ) -> Any:
        """``quantity_at`` of a cause at every joint draw (or at the
        points ``theta``), the draws on the first axis and ``extra`` axes
        after it."""
        source = self._joint() if theta is None else theta
        theta = {
            n: v.reshape((-1,) + (1,) * extra)
            for n, v in source[cause].items()
        }
        with ignoring_warnings(), np.errstate(all="ignore"):
            return np.asarray(
                quantity_at(self.models[cause].dist, (kind, value), theta),
                dtype=float,
            )

    def cif_draws(self, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        """
        Cause ``event``'s cumulative incidence at ``x`` under each joint
        draw, ``(DRAWS, len(x))``: the incidence on the cause's
        probability scale, :math:`\\int_0^{F_k(x)} \\prod_{j \\neq k}
        S_j(F_k^{-1}(p))\\, dp`, by 48-point Gauss-Legendre quadrature (a
        bounded, monotone integrand; the nodes gathered at both ends),
        for every draw at once, to about :math:`10^{-8}`.
        """
        self._check_event(event)
        return self._cif_at(None, x, event)

    def _cif_at(self, theta: dict | None, x: Any, event: Any) -> npt.NDArray:
        """Cause ``event``'s incidence at ``x`` under the joint draws
        (``theta=None``) or at the points ``theta``, ``(N, len(x))``."""
        source = self._joint() if theta is None else theta
        N = len(next(iter(source[event].values())))
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        out = np.full((N, x.size), np.nan)
        ok = ~np.isnan(x)
        step = 16
        idx = np.flatnonzero(ok)
        for s in range(0, idx.size, step):
            cols = idx[s : s + step]
            xs = x[cols]
            top = self._at(
                event, "F", np.where(np.isinf(xs), 0.0, xs), 1, source
            )
            top = np.where(np.isposinf(xs)[None, :], 1.0, top)
            top = np.where(np.isneginf(xs)[None, :], 0.0, top)
            top = np.clip(np.where(np.isfinite(top), top, 0.0), 0.0, 1.0)
            p = top[:, :, None] * GAUSS_NODES[None, None, :]
            t = self._at(event, "q", p, 2, source)
            g = np.ones(p.shape)
            for k in self.causes:
                if k != event:
                    g = g * self._at(k, "R", t, 2, source)
            g = np.where(np.isfinite(g), g, 0.0)
            out[:, cols] = top * np.einsum("dtq,q->dt", g, GAUSS_WEIGHTS)
        return out

    def sf_draws(self, x: npt.ArrayLike) -> npt.NDArray:
        """The all-cause survival at ``x`` under each joint draw,
        ``(DRAWS, len(x))``."""
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        out = np.ones((DRAWS, x.size))
        for k in self.causes:
            out = out * self._at(k, "R", x[None, :], 1)
        return out

    def quantile_draws(self, p: float) -> npt.NDArray:
        """The all-cause ``p`` quantile (the B-life at ``100 p`` %) under
        each joint draw, ``(DRAWS,)``: the root of the summed cumulative
        hazards, by bisection between the causes' own quantiles, for
        every draw at once."""
        p = float(p)
        K = len(self.causes)
        if p <= 0 or p >= 1:
            ends = np.stack(
                [self._at(k, "q", np.array([p]), 1)[:, 0] for k in self.causes]
            )
            return np.min(ends, axis=0)
        hi = np.min(
            np.stack(
                [self._at(k, "q", np.array([p]), 1)[:, 0] for k in self.causes]
            ),
            axis=0,
        )
        lo_p = -np.expm1(np.log1p(-p) / K)
        lo = np.min(
            np.stack(
                [
                    self._at(k, "q", np.array([lo_p]), 1)[:, 0]
                    for k in self.causes
                ]
            ),
            axis=0,
        )
        target = -np.log1p(-p)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            H = np.zeros(DRAWS)
            for k in self.causes:
                R = self._at(k, "R", mid[:, None], 1)[:, 0]
                H = H - np.log(R)
            above = H >= target
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        return 0.5 * (lo + hi)

    @keeps_query_shape
    def cri(
        self,
        x: npt.ArrayLike,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval at times ``x``: the posterior's
        ``alpha_ci / 2`` and ``1 - alpha_ci / 2`` points over the joint
        draws.

        Parameters
        ----------
        x : array like or scalar
            The times.
        event : optional
            The cause, for ``on="cif"`` (its cumulative incidence) or for
            a cause's net ``sf`` and ``ff`` (its own fit's ``cri``);
            ``None`` (the default) is all causes together.
        on : {'cif', 'sf', 'ff'}, optional
            Default ``'cif'``; with ``event=None`` the all-cause
            incidence, ``ff``. (``'R'`` and ``'F'`` are ``'sf'`` and
            ``'ff'``, as for every model.)
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.
        """
        on = {"R": "sf", "F": "ff"}.get(on, on)
        check_option("on", on, _CRI_ON)
        levels = _levels(alpha_ci, bound)
        x = np.asarray(x, dtype=float)
        if event is not None:
            self._check_event(event)
            if on != "cif":
                return np.asarray(
                    self.models[event].cri(
                        x, on=on, alpha_ci=alpha_ci, bound=bound
                    )
                )
            values = self.cif_draws(x, event)
        else:
            s = self.sf_draws(x)
            values = s if on == "sf" else 1.0 - s
        return _shape(values, levels, x.shape)

    @keeps_query_shape
    def quantile_cri(
        self,
        p: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        point: bool = False,
    ) -> npt.NDArray:
        """
        A credible interval on the all-cause B-life (``p=0.1`` for B10):
        the time by which a fraction ``p`` of the population fails from
        any cause, over the joint draws. ``point=True`` gives its
        posterior median, the best single guess at the B-life.
        """
        u = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
        levels = [0.5] if point else _levels(alpha_ci, bound)
        outside = warn_outside_unit_interval(u)
        rows = []
        for i, ui in enumerate(u):
            if outside[i] or np.isnan(ui):
                rows.append(np.full((1, len(levels)), np.nan))
                continue
            q = self.quantile_draws(float(ui))
            rows.append(
                np.asarray(np.quantile(q, levels), dtype=float)[None, :]
            )
        out = np.concatenate(rows, axis=0)
        if point or len(levels) == 1:
            return out[:, 0]
        return out

    def param_cri(
        self,
        event: Any,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval for a parameter of one cause's
        distribution."""
        self._check_event(event)
        return np.asarray(
            self.models[event].param_cri(name, alpha_ci, bound), dtype=float
        )

    # -- words and saving -----------------------------------------------------

    def summary(self, alpha_ci: float = 0.05) -> str:
        """Each cause's fit in words, one after another (with the causes
        fitted together, each cause's marginal posterior)."""
        if self.joint:
            return self._joint_summary(alpha_ci)
        return "\n\n".join(
            f"Cause {k!r}\n" + self.models[k].summary(alpha_ci)
            for k in self.causes
        )

    def _joint_summary(self, alpha_ci: float) -> str:
        """The causes fitted together: each parameter's marginal median
        and credible interval, a cause at a time."""
        check_alpha_ci(alpha_ci)
        kind = self.method[self.causes[0]]
        how = {
            "grid": "computed exactly on a grid",
            "laplace": "approximated by a normal at the peak (Laplace)",
        }.get(kind, kind)
        lines = [
            "Bayesian competing-risks model, the causes fitted together "
            "(failures found in a window hold every cause's parameters, "
            f"so their posteriors are dependent); {how}.",
            f"Posterior       median   {1 - alpha_ci:.0%} credible interval",
        ]
        q = [0.5, alpha_ci / 2, 1 - alpha_ci / 2]
        for k in self.causes:
            fit = self.models[k]
            lines.append(f"Cause {k!r} ({fit.dist.name})")
            for name in fit.posterior.parameter_names:
                med, lo, hi = np.asarray(
                    fit.posterior.quantile(name, q), dtype=float
                )
                lines.append(f"  {name:<14}{med:>8.4g}   {lo:.4g} - {hi:.4g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        """A strict-JSON dictionary of the fit: each cause's, with the
        causes in order."""
        import surpyval
        from surpyval.serialisation import to_native

        out: dict[str, Any] = {
            "model": self.model_tag,
            "causes": [to_native(k) for k in self.causes],
            "surpyval_version": surpyval.__version__,
        }
        if self.joint:
            # (each cause's marginal is rebuilt from the joint points)
            assert self.columns is not None
            out["joint"] = {
                "posterior": self.posterior.to_dict(),
                "columns": [
                    [int(i) for i in self.columns[k]] for k in self.causes
                ],
                "distributions": [
                    self.models[k].dist.name for k in self.causes
                ],
                "priors": [
                    self.models[k].priors.to_dict() for k in self.causes
                ],
                "method": self.method[self.causes[0]],
            }
        else:
            out["models"] = [self.models[k].to_dict() for k in self.causes]
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianCompetingRisks:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.bayes import BayesianParametric
        from surpyval.bayes.posterior import Posterior

        require_model_tag(
            model_dict, cls.model_tag, "a Bayesian competing-risks fit"
        )
        causes = [label_from_native(k) for k in model_dict["causes"]]
        saved = model_dict.get("joint")
        if saved is None:
            return cls(
                {
                    k: BayesianParametric.from_dict(d)
                    for k, d in zip(causes, model_dict["models"])
                }
            )
        import surpyval
        from surpyval.bayes.prior import PriorSpec

        post = Posterior.from_dict(saved["posterior"])
        method = saved.get("method", post.kind)
        models = {}
        for k, cols, name, pri in zip(
            causes, saved["columns"], saved["distributions"], saved["priors"]
        ):
            models[k] = _marginal(
                getattr(surpyval, name), post, list(cols), None, method
            )
            models[k].priors = PriorSpec.from_dict(pri)
        out = cls(
            models,
            posterior=post,
            columns=dict(zip(causes, saved["columns"])),
        )
        out.method = {k: method for k in out.causes}
        return out


def labelled(label: str, fit: Any) -> Any:
    """``fit()`` with each warning it raises said again with ``label``
    in front ("Cause 'a': ..."), so that the same warning from two causes
    (transitions) reads as two warnings, each saying which it is about."""
    import warnings

    from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

    with collecting_warnings() as caught:
        out = fit()
    for w in caught:
        warnings.warn(
            f"{label}: {w.message}",
            w.category,
            stacklevel=caller_stacklevel(),
        )
    return out


def split_priors(priors: Any, causes: list) -> dict:
    """Each cause's priors: one set for every cause, or ``{cause:
    priors}`` (a cause left out has none)."""
    by_cause = (
        isinstance(priors, Mapping)
        and len(priors) > 0
        and all(k in causes for k in priors)
    )
    if isinstance(priors, Mapping) and not by_cause:
        if any(k in causes for k in priors):
            raise ValueError(
                "priors mixes causes and parameter names. Give either one "
                "set for every cause (priors={'beta': ...}) or a set a "
                "cause (priors={cause: {'beta': ...}}); the causes are "
                f"{', '.join(repr(k) for k in causes)}."
            )
    return {k: (priors.get(k) if by_cause else priors) for k in causes}


def fit(
    cls: Any,
    x: Any,
    e: Any,
    c: Any = None,
    n: Any = None,
    dist: Any = None,
    tl: Any = None,
    *,
    priors: Any = None,
    method: str = "auto",
    points: int | None = None,
    importance: bool = False,
) -> BayesianCompetingRisks:
    """The Bayesian fit of a parametric competing-risks model; see
    ``ParametricCompetingRisks.fit_bayes``."""
    from surpyval.univariate.competing_risks.parametric import (
        parametric_competing_risks as pcr,
    )
    from surpyval.univariate.parametric import Weibull
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    dist = Weibull if dist is None else dist
    x, c, n, e, t = pcr._validate(x, c, n, e, tl)
    if ((c == -1) | (c == 2)).any():
        return fit_joint(
            cls,
            (x, c, n, e, t),
            dist,
            tl,
            priors=priors,
            method=method,
            points=points,
            importance=importance,
        )
    causes = ordered_labels(e[c == 0])
    if not causes:
        raise ValueError("No observed events to fit a cause to.")
    if isinstance(dist, dict):
        missing = [k for k in causes if k not in dist]
        if missing:
            raise ValueError(
                "`dist` has no distribution for the cause(s) {}; give "
                "one for every cause {}.".format(missing, causes)
            )
    own = split_priors(priors, causes)
    mle = None
    try:
        with quiet_maximum_warnings(), ignoring_warnings():
            mle = cls.fit(x, e, c=c, n=n, dist=dist, tl=tl)
    except ValueError:
        mle = None
    truncated = tl is not None
    models = {}
    for k in causes:
        c_k = np.where(label_mask(e, k) & (c == 0), 0, 1).astype(int)
        d = dist[k] if isinstance(dist, dict) else dist
        extra = {"t": t} if truncated else {}
        try:
            models[k] = labelled(
                f"Cause {k!r}",
                lambda d=d, c_k=c_k, extra=extra, k=k: d.fit_bayes(
                    x,
                    c_k,
                    n,
                    priors=own[k],
                    method=method,
                    points=points,
                    importance=importance,
                    **extra,
                ),
            )
        except ValueError as err:
            raise ValueError(f"Cause {k!r}: {err}") from None
    return BayesianCompetingRisks(models, mle)


def _marginal(
    dist: Any, post: Any, cols: list, priors: Any, method: str
) -> Any:
    """A cause's marginal of the joint posterior, as a
    :class:`~surpyval.bayes.BayesianParametric` over the same points and
    weights (its quantiles spread over the grid's cells, or the Laplace
    normal's block), for the cause's net functions and parameters."""
    from surpyval.bayes import BayesianParametric
    from surpyval.bayes.posterior import Posterior
    from surpyval.bayes.prior import PriorSpec

    names = list(dist.parameter_names)
    joint_names = [post.parameter_names[i] for i in cols]
    free = [n for n, j in zip(names, joint_names) if j in post.free_names]
    free_at = [
        post.free_names.index(j) for j in joint_names if j in post.free_names
    ]
    transforms = (
        None
        if post.transforms is None
        else [post.transforms[i] for i in free_at]
    )
    normal = None
    if post.normal is not None and not post.normal.get("reweighted"):
        mean = np.asarray(post.normal["mean"], dtype=float)[free_at]
        cov = np.asarray(post.normal["cov"], dtype=float)[
            np.ix_(free_at, free_at)
        ]
        normal = {
            "mean": [float(v) for v in mean],
            "cov": [[float(v) for v in row] for row in cov],
            "reweighted": False,
        }
    marginal = Posterior(
        names,
        free,
        post.points[:, cols],
        post.weights,
        "draws",
        transforms=transforms,
        shape=post.shape,
        normal=normal,
    )
    fixed = {
        n: float(post.points[0, i])
        for n, i, j in zip(names, cols, joint_names)
        if j not in post.free_names
    }
    return BayesianParametric(
        dist,
        marginal,
        priors=PriorSpec.parse(priors),
        fixed=fixed,
        method=method,
        diagnostics={
            "method": method,
            "ok": True,
            "joint": "this cause's marginal of the causes' joint posterior",
        },
    )


def fit_joint(
    cls: Any,
    rows: tuple,
    dist: Any,
    tl: Any,
    *,
    priors: Any = None,
    method: str = "auto",
    points: int | None = None,
    importance: bool = False,
) -> BayesianCompetingRisks:
    """The Bayesian fit of every cause's parameters at once: the joint
    target of :mod:`._joint_bayes` on the shared engines (the grid for
    up to three free parameters in all, Laplace beyond). ``rows`` is the
    validated ``(x, c, n, e, t)``. :func:`fit` takes this path for data
    with failures found in a window; with exact and right-censored data
    alone it gives the factorised posterior (the tests check it)."""
    from surpyval.bayes.result import fit_target
    from surpyval.univariate.competing_risks.parametric import (
        parametric_competing_risks as pcr,
    )
    from surpyval.univariate.competing_risks.parametric._joint_bayes import (
        JointCompetingRisksTarget,
        to_target_priors,
    )
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    x, c, n, e, t = rows
    causes = ordered_labels(e[c != 1])
    pcr.check_some_cause(causes, c)
    if isinstance(dist, dict):
        missing = [k for k in causes if k not in dist]
        if missing:
            raise ValueError(
                "`dist` has no distribution for the cause(s) {}; give "
                "one for every cause {}.".format(missing, causes)
            )
    dists = [dist[k] if isinstance(dist, dict) else dist for k in causes]
    own = split_priors(priors, causes)
    mle = None
    try:
        with quiet_maximum_warnings(), ignoring_warnings():
            mle = cls.fit(x, e, c=c, n=n, dist=dist, tl=tl)
    except ValueError:
        mle = None
    target = JointCompetingRisksTarget(causes, dists, x, c, n, e, t, mle)
    result = fit_target(
        target,
        to_target_priors(own, target),
        method=method,
        points=points,
        importance=importance,
    )
    post = result.posterior
    columns = {
        k: list(range(s.start, s.stop)) for k, s in zip(causes, target.slices)
    }
    models = {
        k: _marginal(d, post, columns[k], own[k], result.method)
        for k, d in zip(causes, dists)
    }
    return BayesianCompetingRisks(
        models, mle, posterior=post, columns=columns, result=result
    )


__all__ = ["BayesianCompetingRisks", "DRAWS", "SEED", "fit", "fit_joint"]

# Temporary aliases for the names these helpers had before they became
# public. Remove them once the open branches that import the old names have
# landed.
_NODES = GAUSS_NODES
_WEIGHTS = GAUSS_WEIGHTS
