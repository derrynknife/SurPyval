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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

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
_NODES = 3 * _V**2 - 2 * _V**3
_WEIGHTS = 0.5 * _W * 6 * _V * (1 - _V)
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

    def __init__(self, models: dict, mle: Any = None) -> None:
        if len(models) == 0:
            raise ValueError("At least one cause's fit is required.")
        self.models = {
            (k.item() if isinstance(k, np.generic) else k): m
            for k, m in models.items()
        }
        self.causes = ordered_labels(self.models)
        self.mle = mle
        self.method = {k: self.models[k].method for k in self.causes}
        self._draws: dict | None = None

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
        with np.errstate(all="ignore"):
            s = np.asarray(self.sf(x), dtype=float)
            return np.where(s == 0, 0.0, s * np.asarray(self.hf(x)))

    def _grid(self, x: npt.NDArray) -> npt.NDArray:
        """The times of the predictive incidence's integral: each cause's
        plug-in quantiles (at the posterior medians) on a fine
        logit-spaced set of probabilities, which follows every cause's
        support and scale, with the query times among them."""
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
        flat = x.ravel()
        out = np.full(flat.shape, np.nan)
        ok = ~np.isnan(flat)
        if not ok.any():
            return out
        grid = self._grid(flat[ok])
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
        at = flat[ok]
        vals = np.interp(at, grid, cum, left=0.0, right=cum[-1])
        if not np.isfinite(g_end):
            g_end = 0.0
        if not np.isfinite(F_end):
            F_end = 1.0
        tail = cum[-1] + 0.5 * (g[-1] + g_end) * max(F_end - F[-1], 0.0)
        vals = np.where(np.isposinf(at), tail, vals)
        vals = np.where(np.isneginf(at), 0.0, vals)
        out[ok] = vals
        return out.reshape(x.shape)

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
        for i, ui in enumerate(flat):
            if outside[i] or np.isnan(ui):
                continue
            if ui == 0 or ui == 1:
                bounds = [
                    float(np.asarray(f.qf(np.array([ui]))).ravel()[0])
                    for f in self._fits()
                ]
                out[i] = min(bounds)
                continue
            # The root lies between the causes' net quantiles: no later
            # than the earliest at p, no earlier than the earliest at
            # 1 - (1 - p)^(1/K).
            hi = min(
                float(np.asarray(f.qf(np.array([ui]))).ravel()[0])
                for f in self._fits()
            )
            lo_p = -np.expm1(np.log1p(-ui) / K)
            lo = min(
                float(np.asarray(f.qf(np.array([lo_p]))).ravel()[0])
                for f in self._fits()
            )
            target = -np.log1p(-ui)

            def gap(t: float) -> float:
                return float(np.asarray(self.Hf(t), dtype=float)) - target

            if not np.isfinite(hi):
                out[i] = np.inf if not np.isfinite(lo) else np.nan
                continue
            if hi <= lo:
                out[i] = hi
                continue
            with ignoring_warnings(), np.errstate(all="ignore"):
                out[i] = brentq(gap, lo, hi, xtol=1e-12 * max(hi, 1.0))
        return out.reshape(u.shape)

    # -- credible intervals, from joint draws -------------------------------

    def _joint(self) -> dict:
        """``DRAWS`` draws of each cause's parameters (independent across
        causes, so jointly drawn), as ``{cause: {name: (DRAWS,)}}``."""
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

    def _at(self, cause: Any, kind: str, value: Any, extra: int) -> Any:
        """``quantity_at`` of a cause at every joint draw, the draws on
        the first axis and ``extra`` axes after it."""
        theta = {
            n: v.reshape((-1,) + (1,) * extra)
            for n, v in self._joint()[cause].items()
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
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        out = np.full((DRAWS, x.size), np.nan)
        ok = ~np.isnan(x)
        step = 16
        idx = np.flatnonzero(ok)
        for s in range(0, idx.size, step):
            cols = idx[s : s + step]
            xs = x[cols]
            top = self._at(event, "F", np.where(np.isinf(xs), 0.0, xs), 1)
            top = np.where(np.isposinf(xs)[None, :], 1.0, top)
            top = np.where(np.isneginf(xs)[None, :], 0.0, top)
            top = np.clip(np.where(np.isfinite(top), top, 0.0), 0.0, 1.0)
            p = top[:, :, None] * _NODES[None, None, :]
            t = self._at(event, "q", p, 2)
            g = np.ones(p.shape)
            for k in self.causes:
                if k != event:
                    g = g * self._at(k, "R", t, 2)
            g = np.where(np.isfinite(g), g, 0.0)
            out[:, cols] = top * np.einsum("dtq,q->dt", g, _WEIGHTS)
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
            incidence, ``ff``.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.
        """
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
        """Each cause's fit in words, one after another."""
        return "\n\n".join(
            f"Cause {k!r}\n" + self.models[k].summary(alpha_ci)
            for k in self.causes
        )

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        """A strict-JSON dictionary of the fit: each cause's, with the
        causes in order."""
        import surpyval
        from surpyval.serialisation import to_native

        return stamp_schema(
            {
                "model": self.model_tag,
                "causes": [to_native(k) for k in self.causes],
                "models": [self.models[k].to_dict() for k in self.causes],
                "surpyval_version": surpyval.__version__,
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianCompetingRisks:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.bayes import BayesianParametric

        require_model_tag(
            model_dict, cls.model_tag, "a Bayesian competing-risks fit"
        )
        return cls(
            {
                label_from_native(k): BayesianParametric.from_dict(d)
                for k, d in zip(model_dict["causes"], model_dict["models"])
            }
        )


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
        raise ValueError(
            "fit_bayes takes exact failures and right censoring (with "
            "left truncation): a unit found failed from a cause in a "
            "window (c = -1 or 2) contributes that cause's incidence over "
            "the window, which holds every cause's parameters at once, so "
            "the causes' posteriors are no longer separate fits. Use fit "
            "(maximum likelihood over every cause at once) for such data, "
            "or treat each window's failure as exact at its midpoint "
            "knowingly."
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
            models[k] = d.fit_bayes(
                x,
                c_k,
                n,
                priors=own[k],
                method=method,
                points=points,
                importance=importance,
                **extra,
            )
        except ValueError as err:
            raise ValueError(f"Cause {k!r}: {err}") from None
    return BayesianCompetingRisks(models, mle)


__all__ = ["BayesianCompetingRisks", "DRAWS", "SEED", "fit"]
