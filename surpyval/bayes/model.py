"""``BayesianParametric``: the result of ``fit_bayes``.

It is used like any SurPyval model. Its ``sf``, ``ff``, ``df``, ``hf``,
``Hf``, ``qf``, ``mean`` and ``random`` are those of the **posterior
predictive** distribution: the life of the next unit, with the
uncertainty in the parameters averaged in. Its credible intervals say how
sure we are about the curve itself: ``cri`` (on ``sf``, ``ff``, ``Hf``,
``hf`` or ``df``), ``quantile_cri`` (a B-life), ``mean_cri`` and
``param_cri``.
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import integrate, optimize, stats

from surpyval.bayes.likelihood import LogLikelihood, model_parameter_names
from surpyval.bayes.posterior import Posterior
from surpyval.bayes.prior import (
    DistributionPrior,
    Flat,
    MultivariateNormal,
    NormalOnScale,
    PowerPrior,
    PriorSpec,
    _fmt,
    quantity_at,
)
from surpyval.bayes.transforms import Transform
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import (
    check_alpha_ci,
    warn_outside_unit_interval,
)

_SIDES = ("two-sided", "lower", "upper")
_ON = {
    "sf": "sf",
    "R": "sf",
    "ff": "ff",
    "F": "ff",
    "Hf": "Hf",
    "hf": "hf",
    "df": "df",
}


def _levels(alpha_ci: float, bound: str) -> list[float]:
    check_alpha_ci(alpha_ci)
    if bound == "two-sided":
        return [alpha_ci / 2, 1 - alpha_ci / 2]
    if bound == "lower":
        return [alpha_ci]
    if bound == "upper":
        return [1 - alpha_ci]
    raise ValueError(
        f"bound must be one of {', '.join(_SIDES)}; got {bound!r}."
    )


def _shape_bound(values: npt.NDArray, bound: str) -> npt.NDArray:
    """(m, n_levels) -> (m, 2) two-sided or (m,) one-sided."""
    if bound == "two-sided":
        return values
    return values[..., 0]


class BayesianParametric(SerialisableMixin):
    """
    A Bayesian fit of a parametric distribution: the posterior of its
    parameters, and the posterior predictive distribution of a new unit's
    life.

    Built by ``Distribution.fit_bayes`` (and ``fit_bayes_from_df``), or
    from another sampler's draws by :meth:`from_draws`.

    Attributes
    ----------
    dist : ParametricFitter
        The distribution.
    params : numpy.ndarray
        The posterior medians of the distribution's parameters (named by
        ``parameter_names``); ``gamma``, ``lfp_p`` and ``f0`` are those of
        the offset, limited-failure proportion and zero inflation.
    posterior : Posterior
        The posterior itself: ``draws``, ``to_dataframe``, ``density``,
        ``marginal``.
    priors : PriorSpec
        The priors as given.
    method : str
        ``"grid"``, ``"conjugate"``, ``"laplace"`` or ``"draws"``.
    log_evidence : float
        The log marginal likelihood (for Bayes factors), where every prior
        is proper; ``nan`` otherwise.
    diagnostics : dict
        How the posterior was computed and how accurately.

    Examples
    --------
    Three of 40 pump seals have failed, and the shape is believed to be
    about 2, between 1.5 and 3:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> x = np.r_[2100, 3400, 4050, np.linspace(1500, 6000, 37)]
    >>> c = np.r_[0, 0, 0, np.ones(37)]
    >>> model = sp.Weibull.fit_bayes(
    ...     x, c, priors={"beta": sp.prior.around(2, between=(1.5, 3))})
    >>> model.method
    'grid'
    >>> bool(model.sf(5000) > 0.5)
    True
    >>> model.param_cri("beta").round(1)
    array([1.4, 3.1])
    """

    model_tag = "BayesianParametric"

    def __init__(
        self,
        dist: Any,
        posterior: Posterior,
        *,
        data: SurpyvalData | None = None,
        priors: PriorSpec | None = None,
        offset: bool = False,
        lfp: bool = False,
        zi: bool = False,
        fixed: dict[str, float] | None = None,
        method: str = "draws",
        log_evidence: float = float("nan"),
        diagnostics: dict | None = None,
        exact: dict | None = None,
    ) -> None:
        self.dist = dist
        self.posterior = posterior
        self.data = data
        self.priors = priors if priors is not None else PriorSpec()
        self.offset, self.lfp, self.zi = bool(offset), bool(lfp), bool(zi)
        self.fixed = dict(fixed or {})
        self.method = method
        self.log_evidence = float(log_evidence)
        self.diagnostics = dict(diagnostics or {})
        self._exact = exact
        self._mle_fit: Any = None
        self._mle_known = False
        self._log_post: Any = None
        self._log_z: float | None = None
        self.unit: str | None = None
        self._set_medians()
        self._attach_density()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def _set_medians(self) -> None:
        med = dict(
            zip(self.posterior.parameter_names, self.posterior.median())
        )
        self.params = np.array(
            [med[n] for n in self.dist.parameter_names], dtype=float
        )
        self.gamma = float(med.get("gamma", 0.0))
        self.lfp_p = float(med.get("lfp_p", 1.0))
        self.f0 = float(med.get("f0", 0.0))
        for n in self.dist.parameter_names:
            if n not in ("gamma", "p", "lfp_p", "f0"):
                setattr(self, n, med[n])

    @classmethod
    def _from_grid(
        cls,
        dist: Any,
        data: SurpyvalData | None,
        bound: Any,
        result: Any,
        method: str,
        log_post: Any,
        mle: Any,
    ) -> BayesianParametric:
        theta = bound.full_theta(result.phi)
        free_t = [bound.transforms[i] for i in bound.free_idx]
        post = Posterior(
            bound.names,
            bound.free_names,
            theta,
            np.exp(result.log_w),
            "laplace" if method == "laplace" else "grid",
            transforms=free_t,
            cell=result.cell,
            shape=result.shape,
        )
        proper = bound.proper and data is not None
        if data is None and bound.proper:
            proper = True
        diag = {
            "method": method,
            "free_parameters": list(bound.free_names),
            "points_per_dim": result.points_per_dim,
            "grid_points": int(len(result.log_w)),
            "refinements": result.refinements,
            "integration_error": result.error,
            "half_width_sd": float(np.max(result.half_widths)),
            "widened": result.checks.get("widened", 0),
            "mode": [float(v) for v in bound.full_theta(result.mode)[0]],
            "ok": True,
            "approximate": method == "laplace",
        }
        model = cls(
            dist,
            post,
            data=data,
            priors=bound.spec,
            offset=bound.offset,
            lfp=bound.lfp,
            zi=bound.zi,
            fixed=bound.fixed,
            method=method,
            log_evidence=result.log_z if proper else float("nan"),
            diagnostics=diag,
        )
        model._mle_fit = mle
        model._mle_known = data is not None
        model._log_post = log_post
        model._log_z = result.log_z
        model._attach_density()
        return model

    @classmethod
    def _from_conjugate(
        cls, dist: Any, data: SurpyvalData, bound: Any, res: Any
    ) -> BayesianParametric:
        from surpyval.bayes.engines.conjugate import NODES
        from surpyval.bayes.posterior import _CONJUGATE_FAMILIES

        frozen = _CONJUGATE_FAMILIES[res.family](*res.params)
        u = (np.arange(NODES) + 0.5) / NODES
        theta = np.repeat(
            bound.full_theta(np.zeros((1, bound.d)))[:1], NODES, axis=0
        )
        col = bound.names.index(res.name)
        theta[:, col] = frozen.ppf(u)
        exact = {
            "name": res.name,
            "family": res.family,
            "params": list(res.params),
        }
        post = Posterior(
            bound.names,
            bound.free_names,
            theta,
            np.full(NODES, 1.0 / NODES),
            "conjugate",
            transforms=[bound.transforms[i] for i in bound.free_idx],
            exact=exact,
        )
        diag = {
            "method": "conjugate",
            "free_parameters": list(bound.free_names),
            "case": res.label,
            "ok": True,
            "approximate": False,
        }
        return cls(
            dist,
            post,
            data=data,
            priors=bound.spec,
            offset=False,
            fixed=bound.fixed,
            method="conjugate",
            log_evidence=res.log_evidence,
            diagnostics=diag,
            exact=exact,
        )

    @classmethod
    def from_draws(
        cls,
        dist: Any,
        draws: npt.ArrayLike,
        parameter_names: list[str] | None = None,
        data: Any = None,
        priors: Any = None,
        *,
        weights: npt.ArrayLike | None = None,
        diagnostics: dict | None = None,
    ) -> BayesianParametric:
        """
        A Bayesian model from posterior draws made elsewhere (another
        sampler, such as PyMC or Stan).

        Every function, credible interval and plot then works as for a
        fit by ``fit_bayes``; the summaries are computed from the draws.

        Parameters
        ----------
        dist : ParametricFitter
            The distribution the draws are for.
        draws : 2-D array like
            One row per draw, one column per parameter.
        parameter_names : list of str, optional
            The columns' names: the distribution's parameters, and
            ``gamma``, ``lfp_p`` or ``f0`` for an offset, limited-failure
            or zero-inflated model. Defaults to the distribution's
            ``parameter_names``.
        data : SurpyvalData or (x, c, n, t), optional
            The data the draws came from, for ``plot``, ``compare`` and
            ``update``.
        priors : optional
            The priors used, as for ``fit_bayes``, kept for the summary and
            for ``update`` (which then refits with SurPyval's own engine).
        weights : array like, optional
            A weight per draw (importance weights); equal by default.
        diagnostics : dict, optional
            The sampler's diagnostics, kept as given (R-hat, ESS, ...).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> rng = np.random.default_rng(1)
        >>> draws = np.c_[rng.lognormal(np.log(100), 0.1, 2000),
        ...               rng.lognormal(np.log(2), 0.1, 2000)]
        >>> model = sp.bayes.BayesianParametric.from_draws(
        ...     sp.Weibull, draws, ["alpha", "beta"])
        >>> model.params.round(0)
        array([100.,   2.])
        """
        D = np.atleast_2d(np.asarray(draws, dtype=float))
        names = (
            list(dist.parameter_names)
            if parameter_names is None
            else list(parameter_names)
        )
        if D.shape[1] != len(names):
            raise ValueError(
                f"draws has {D.shape[1]} columns, but {len(names)} names "
                f"were given ({', '.join(names)})."
            )
        missing = [n for n in dist.parameter_names if n not in names]
        if missing:
            raise ValueError(
                f"draws must have a column for every parameter of "
                f"{dist.name}; missing {', '.join(missing)}."
            )
        extra = [
            n
            for n in names
            if n not in dist.parameter_names
            and n not in ("gamma", "lfp_p", "f0")
        ]
        if extra:
            raise ValueError(
                f"Unknown parameter columns {extra}; {dist.name} has "
                f"{', '.join(dist.parameter_names)} (and gamma, lfp_p, f0 "
                "for the offset, limited-failure and zero-inflated forms)."
            )
        if not np.all(np.isfinite(D)):
            raise ValueError("draws must be finite numbers.")
        offset, lfp, zi = (
            ("gamma" in names),
            ("lfp_p" in names),
            ("f0" in names),
        )
        order = model_parameter_names(dist, offset, lfp, zi)
        D = D[:, [names.index(n) for n in order]]
        w = (
            np.full(D.shape[0], 1.0 / D.shape[0])
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if w.shape != (D.shape[0],) or np.any(w < 0) or not w.sum() > 0:
            raise ValueError(
                "weights must be one non-negative number per draw."
            )
        free = [n for j, n in enumerate(order) if np.ptp(D[:, j]) > 0]
        fixed = {
            n: float(D[0, j]) for j, n in enumerate(order) if n not in free
        }
        bounds_all = list(dist.bounds)
        if offset:
            bounds_all = [(None, None)] + bounds_all
        bounds_all += [(0, 1)] * (int(lfp) + int(zi))
        ts = [
            Transform.for_bounds(b)
            for n, b in zip(order, bounds_all)
            if n in free
        ]
        post = Posterior(order, free, D, w, "draws", transforms=ts)
        if data is not None and not isinstance(data, SurpyvalData):
            data = (
                SurpyvalData(*data)
                if isinstance(data, tuple)
                else (SurpyvalData(x=data))
            )
        from surpyval.utils.weights import (
            BAYES_INSTEAD,
            BAYES_WHY,
            refuse_weighted_data,
        )

        refuse_weighted_data(data, "from_draws", BAYES_WHY, BAYES_INSTEAD)
        return cls(
            dist,
            post,
            data=data,
            priors=PriorSpec.parse(priors),
            offset=offset,
            lfp=lfp,
            zi=zi,
            fixed=fixed,
            method="draws",
            diagnostics={
                "method": "draws",
                "ok": True,
                "draws": int(D.shape[0]),
                **(diagnostics or {}),
            },
        )

    # ------------------------------------------------------------------
    # The bound prior and likelihood, rebuilt where needed
    # ------------------------------------------------------------------
    def _bound(self) -> Any:
        from surpyval.bayes.fitting import build_bound

        data = (
            self.data
            if self.data is not None
            else SurpyvalData(x=[1.0], c=[1])
        )
        return build_bound(
            self.dist,
            data,
            self.priors,
            self.offset,
            self.lfp,
            self.zi,
            self.fixed,
        )

    def _attach_density(self) -> None:
        post = self.posterior
        if post.kind == "conjugate":
            fam = post._exact(post.exact["name"]) if post.exact else None
            if fam is not None:
                post.log_density_fn = lambda p, f=fam: f.logpdf(p[:, 0])
            return
        if post.kind != "grid":
            return

        model = self

        def log_density(points: npt.NDArray) -> npt.NDArray:
            if model._log_post is None:
                model._rebuild_log_post()
            ts = post.transforms or []
            phi = np.stack(
                [t.to_phi(points[:, j]) for j, t in enumerate(ts)], axis=1
            )
            jac = sum(t.log_jacobian(phi[:, j]) for j, t in enumerate(ts))
            assert model._log_z is not None
            return model._log_post(phi) - model._log_z - jac

        post.log_density_fn = log_density

    def _rebuild_log_post(self) -> None:
        bound = self._bound()
        ll = (
            None
            if self.data is None
            else LogLikelihood(
                self.dist, self.data, self.offset, self.lfp, self.zi
            )
        )

        def log_post(phi: npt.NDArray) -> npt.NDArray:
            phi = np.atleast_2d(phi)
            out = bound.log_prior(phi)
            if ll is not None:
                with np.errstate(all="ignore"):
                    out = out + ll(bound.full_theta(phi))
            return np.where(np.isnan(out), -np.inf, out)

        self._log_post = log_post
        if self._log_z is None:
            self._log_z = float(
                self.diagnostics.get("log_z", self.log_evidence)
            )

    # ------------------------------------------------------------------
    # Names and values
    # ------------------------------------------------------------------
    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``: the distribution's parameters."""
        return list(self.dist.parameter_names)

    @property
    def extras(self) -> dict[str, float]:
        """The posterior medians of ``gamma``, ``lfp_p`` and ``f0``, where
        the model has them."""
        out = {}
        if self.offset:
            out["gamma"] = self.gamma
        if self.lfp:
            out["lfp_p"] = self.lfp_p
        if self.zi:
            out["f0"] = self.f0
        return out

    @property
    def ok(self) -> bool:
        """Whether the posterior was computed to its stated accuracy."""
        return bool(self.diagnostics.get("ok", True))

    def raise_if_failed(self) -> None:
        """Raise ``ValueError`` where :attr:`ok` is False."""
        if not self.ok:
            raise ValueError(
                f"The Bayesian fit did not pass its checks: {self.diagnostics}"
            )

    def _parameter_bounds(self) -> list[tuple]:
        return list(self.dist.bounds)

    def at_params(self) -> Any:
        """The ordinary (plug-in) model at the posterior medians, for one
        curve instead of the predictive distribution."""
        return self.dist.from_params(list(self.params), **self.extras)

    def draw_models(self, k: int = 100, random_state: Any = None) -> list:
        """
        ``k`` ordinary models, one per posterior draw, for a Monte Carlo
        that needs whole models (a fleet simulation, an RBD).

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> [type(m).__name__ for m in model.draw_models(2, random_state=1)]
        ['Parametric', 'Parametric']
        """
        D = self.posterior.draws(k, random_state)
        names = self.posterior.parameter_names
        out = []
        for row in D:
            v = dict(zip(names, row))
            kw = {}
            if self.offset:
                kw["gamma"] = v["gamma"]
            if self.lfp:
                kw["lfp_p"] = v["lfp_p"]
            if self.zi:
                kw["f0"] = v["f0"]
            out.append(
                self.dist.from_params(
                    [v[n] for n in self.dist.parameter_names], **kw
                )
            )
        return out

    # ------------------------------------------------------------------
    # The functions at every posterior point
    # ------------------------------------------------------------------
    def _theta(self) -> dict[str, npt.NDArray]:
        P = self.posterior.points
        return {
            n: P[:, i] for i, n in enumerate(self.posterior.parameter_names)
        }

    def _components(self, fname: str, x: npt.NDArray) -> npt.NDArray:
        """``fname`` of every posterior point at the 1-D ``x``, shape
        ``(N, len(x))``."""
        th = self._theta()
        N = len(self.posterior.weights)
        dp = [th[n][:, None] for n in self.dist.parameter_names]
        gamma = th.get("gamma", np.zeros(N))[:, None]
        p = th.get("lfp_p", np.ones(N))[:, None]
        f0 = th.get("f0", np.zeros(N))[:, None]
        xg = x[None, :] - gamma
        d = self.dist
        plain = not (self.offset or self.lfp or self.zi)
        with np.errstate(all="ignore"):
            if fname in ("sf", "ff", "Hf"):
                s = d.sf(xg, *dp)
                if not plain:
                    s = 1 - p + (p - f0) * s
                    if self.zi:
                        s = np.where(x[None, :] < 0, 1.0, s)
                if fname == "sf":
                    return s
                if fname == "ff":
                    return 1 - s
                return -np.log(s)
            if fname in ("df", "hf"):
                f = d.df(xg, *dp)
                if not plain:
                    f = (p - f0) * f
                if fname == "df":
                    return f
                s = d.sf(xg, *dp)
                if not plain:
                    s = 1 - p + (p - f0) * s
                return f / s
        raise ValueError(fname)

    def _chunks(self, fname: str, x: npt.NDArray, fn: Any) -> npt.NDArray:
        """``fn(components)`` over ``x`` in chunks, concatenated on the
        last axis."""
        N = max(len(self.posterior.weights), 1)
        step = max(1, 4_000_000 // N)
        parts = [
            fn(self._components(fname, x[i : i + step]))
            for i in range(0, len(x), step)
        ]
        if not parts:
            return fn(self._components(fname, x))
        return np.concatenate(parts, axis=-1)

    def _closed_predictive(self) -> Any:
        """The predictive distribution as an ordinary model where it is
        one exactly (Normal with sigma known; Bernoulli)."""
        ex = self._exact
        if ex is None:
            return None
        if ex["family"] == "normal":
            m, s = ex["params"]
            sigma = self.fixed["sigma"]
            return self.dist.from_params([m, float(np.hypot(sigma, s))])
        if ex["family"] == "beta":
            a, b = ex["params"]
            return self.dist.from_params([a / (a + b)])
        return None

    def _lomax(self) -> tuple[float, float] | None:
        ex = self._exact
        if ex is not None and ex["family"] == "gamma":
            a, b = ex["params"]
            return float(a), float(b)
        return None

    def _expect(self, fname: str, x: npt.NDArray) -> npt.NDArray:
        w = self.posterior.weights
        return self._chunks(fname, x, lambda v: w @ np.nan_to_num(v, nan=0))

    def _predict(self, fname: str, x: Any) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        closed = self._closed_predictive()
        if closed is not None:
            return np.asarray(getattr(closed, fname)(x), dtype=float)
        nan = np.isnan(x)
        xs = np.where(nan, 0.0, x)
        lomax = self._lomax()
        with np.errstate(all="ignore"):
            if lomax is not None:
                a, b = lomax
                xp = np.maximum(xs, 0.0)
                s = (b / (b + xp)) ** a
                f = np.where(xs >= 0, a * b**a / (b + xp) ** (a + 1), 0.0)
            else:
                need_s = fname in ("sf", "ff", "Hf", "hf")
                need_f = fname in ("df", "hf")
                s = self._expect("sf", xs) if need_s else np.ones_like(xs)
                f = self._expect("df", xs) if need_f else np.zeros_like(xs)
            out = {
                "sf": lambda: s,
                "ff": lambda: 1 - s,
                "Hf": lambda: -np.log(s),
                "df": lambda: f,
                "hf": lambda: f / s,
            }[fname]()
        out = np.asarray(out, dtype=float)
        return np.where(nan, np.nan, out)

    # ------------------------------------------------------------------
    # The predictive distribution
    # ------------------------------------------------------------------
    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The survival function of a new unit: the posterior mean of the
        survival function, :math:`S(x) = E[S(x \\mid \\theta)]`.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.sf(100).round(4))
        0.6806
        """
        return self._predict("sf", x)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """The failure function of a new unit, ``1 - sf``.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.ff(100).round(4))
        0.3194
        """
        return self._predict("ff", x)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike) -> npt.NDArray:
        """The density of a new unit's life, the posterior mean of the
        density.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.df(100).round(6))
        0.002521
        """
        return self._predict("df", x)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The hazard of a new unit, ``df / sf`` of the predictive
        distribution.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.hf(100).round(6))
        0.003704
        """
        return self._predict("hf", x)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The cumulative hazard of a new unit, ``-log sf``.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.Hf(100).round(4))
        0.3848
        """
        return self._predict("Hf", x)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        """
        The quantile function of a new unit's life: the time by which a
        fraction ``p`` of new units will have failed, all things
        considered. (This is not the posterior median of the B-life,
        which :meth:`quantile_cri` gives with its interval.)

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.qf(0.5).round(1))
        185.8
        """
        u = np.asarray(p, dtype=float)
        closed = self._closed_predictive()
        if closed is not None:
            return np.asarray(closed.qf(u), dtype=float)
        outside = warn_outside_unit_interval(u)
        out = np.full(u.shape, np.nan)
        lomax = self._lomax()
        for i, ui in enumerate(u):
            if outside[i] or np.isnan(ui):
                continue
            if lomax is not None:
                a, b = lomax
                with np.errstate(divide="ignore"):
                    out[i] = b * ((1 - ui) ** (-1 / a) - 1)
                continue
            out[i] = self._qf_one(float(ui))
        return out

    def _qf_one(self, u: float) -> float:
        top = float(self._predict("ff", np.array([np.inf]))[0])
        if u >= top - 1e-15 and top < 1:
            return float("inf")
        lo_edge = float(self.dist.support[0])
        if u == 0:
            return lo_edge + (self.gamma if self.offset else 0.0)
        if u >= 1:
            return float(self.dist.support[1])
        th = self._theta()
        with np.errstate(all="ignore"):
            comp = np.asarray(
                quantity_at(self.dist, ("q", u), th), dtype=float
            )
        comp = comp[np.isfinite(comp)]
        if comp.size == 0:
            return float("nan")
        a, b = float(comp.min()), float(comp.max())
        f = lambda t: float(self._predict("ff", np.array([t]))[0]) - u  # noqa
        span = max(b - a, abs(b), 1e-12)
        while f(a) > 0:
            a -= span
            span *= 2
        span = max(b - a, abs(b), 1e-12)
        while f(b) < 0:
            b += span
            span *= 2
        if a == b:
            return a
        return float(optimize.brentq(f, a, b, xtol=1e-14, rtol=1e-13))

    def cs(self, x: npt.ArrayLike, given: npt.ArrayLike) -> npt.NDArray:
        """The chance a new unit that has survived to ``given`` survives a
        further ``x``: ``sf(x + given) / sf(given)``.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> float(model.cs(100, 100).round(4))
        0.6996
        """
        x = np.asarray(x, dtype=float)
        g = np.asarray(given, dtype=float)
        with np.errstate(all="ignore"):
            return self.sf(x + g) / self.sf(g)

    def random(self, size: Any, random_state: Any = None) -> npt.NDArray:
        """
        Draws of a new unit's life from the predictive distribution: a
        parameter point from the posterior, then a life from it, for each
        draw.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> model.random(3, random_state=1).shape
        (3,)
        """
        from surpyval.univariate.parametric.parametric import draw_state

        rng = draw_state(random_state)
        if rng is None:
            # numpy's global stream, so np.random.seed reproduces it
            rng = np.random.mtrand._rand
        shape = (size,) if np.isscalar(size) else tuple(size)
        total = int(np.prod(shape))
        D = self.posterior.draws(total, rng)
        names = self.posterior.parameter_names
        v = {n: D[:, i] for i, n in enumerate(names)}
        u = rng.uniform(size=total)
        if self.dist.discrete:
            return self.dist.qf(u, v["p"]).reshape(shape)
        gamma = v.get("gamma", 0.0)
        p = v.get("lfp_p", 1.0)
        f0 = v.get("f0", 0.0)
        with np.errstate(all="ignore"):
            base = np.clip((u - f0) / (p - f0), 0, 1)
            x = gamma + self.dist.qf(
                base, *[v[n] for n in self.dist.parameter_names]
            )
            x = np.where(u < f0, 0.0, x)
            x = np.where(u > p, np.inf, x)
        return np.asarray(x, dtype=float).reshape(shape)

    def _component_means(self) -> npt.NDArray:
        th = self._theta()
        with np.errstate(all="ignore"):
            m = np.asarray(quantity_at(self.dist, ("mean",), th), dtype=float)
        if self.zi:
            m = (1 - th["f0"]) * m
        if self.lfp:
            m = np.where(th["lfp_p"] < 1, np.inf, m)
        return m

    def mean(self) -> float:
        """
        The mean life of a new unit: the posterior mean of the mean.
        Infinite where any posterior point's mean is (a limited failure
        population, a LogLogistic's beta at or below 1), with a warning
        that says how probable that is.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> round(model.mean(), 1)
        312.5
        """
        lomax = self._lomax()
        if lomax is not None:
            a, b = lomax
            return float(b / (a - 1)) if a > 1 else float("inf")
        closed = self._closed_predictive()
        if closed is not None:
            return float(closed.mean())
        m = self._component_means()
        w = self.posterior.weights
        bad = ~np.isfinite(m)
        # (parameter values with less than 1e-12 of the posterior between
        # them are below what the grid resolves)
        if float(w[bad].sum()) > 1e-12:
            mass = float(w[bad].sum())
            warnings.warn(
                f"The mean life of a new unit is infinite: the posterior "
                f"gives probability {mass:.3g} to parameters whose mean is "
                f"infinite. Quote a B-life (qf, quantile_cri) instead.",
                stacklevel=2,
            )
            return float("inf")
        return float(w[~bad] @ m[~bad] / w[~bad].sum())

    def _component_moment(self, k: int) -> npt.NDArray:
        th = self._theta()
        dp = [th[n] for n in self.dist.parameter_names]
        try:
            with np.errstate(all="ignore"):
                m = np.asarray(self.dist.moment(k, *dp), dtype=float)
            if m.shape != dp[0].shape:
                raise ValueError
        except (ValueError, TypeError):
            m = np.array(
                [
                    float(self.dist.moment(k, *[a[i] for a in dp]))
                    for i in range(len(dp[0]))
                ]
            )
        return m

    def moment(self, n: int) -> float:
        """The ``n``-th raw moment of a new unit's life (no offset,
        limited-failure or zero-inflated form)."""
        if self.offset or self.lfp or self.zi:
            raise ValueError(
                "moment() takes the plain distribution; this model has an "
                "offset, limited failure population or zero inflation."
            )
        return float(self.posterior.weights @ self._component_moment(n))

    def var(self) -> float:
        """The variance of a new unit's life (the predictive variance:
        the posterior mean of each variance plus the variance of the
        means)."""
        m1 = self.mean()
        if not np.isfinite(m1):
            return float("inf")
        return float(self.moment(2) - m1**2)

    def mrl(self, x: npt.ArrayLike) -> npt.NDArray:
        """The mean residual life of a new unit at ``x``: the expected
        further life of one that has survived to ``x``, from the
        predictive survival function."""
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty(xs.shape)
        for i, xi in enumerate(xs):
            s = float(self.sf(xi))
            if s <= 0 or np.isnan(s):
                out[i] = np.nan
                continue
            val, _ = integrate.quad(
                lambda t: float(self.sf(t)), xi, np.inf, limit=200
            )
            out[i] = val / s
        return out[0] if np.ndim(x) == 0 else out.reshape(np.shape(x))

    # ------------------------------------------------------------------
    # Credible intervals
    # ------------------------------------------------------------------
    def _monotone(self, g: Any, levels: list[float]) -> npt.NDArray | None:
        """For a conjugate (one-parameter) posterior and ``g`` monotone in
        that parameter: the exact quantiles of ``g``, ``(m, levels)``."""
        ex = self._exact
        if ex is None:
            return None
        post = self.posterior
        col = post.parameter_names.index(ex["name"])
        rows = []
        for q in levels:
            pts = post.points[:2].copy()
            pts[0, col] = post.quantile(ex["name"], q)
            pts[1, col] = post.quantile(ex["name"], 1 - q)
            rows.append(g(pts))
        vals = np.stack(rows, axis=-1)  # (2, m, levels)
        lo_hi = np.where(
            np.asarray(levels) < 0.5,
            np.minimum(vals[0], vals[1]),
            np.maximum(vals[0], vals[1]),
        )
        at_half = np.isclose(levels, 0.5)
        if np.any(at_half):
            lo_hi[..., at_half] = vals[0][..., at_half]
        return lo_hi

    def _sub(self, pts: npt.NDArray) -> BayesianParametric:
        m = copy(self)
        m.posterior = Posterior(
            self.posterior.parameter_names,
            self.posterior.free_names,
            pts,
            np.full(len(pts), 1.0 / len(pts)),
            "draws",
        )
        return m

    @keeps_query_shape
    def cri(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        point: bool = False,
    ) -> npt.NDArray:
        """
        The credible interval on a function of time: the range in which
        the curve lies with posterior probability ``1 - alpha_ci``, given
        the prior and the data (equal-tailed).

        Parameters
        ----------
        x : array like
            The times.
        on : {'sf', 'ff', 'Hf', 'hf', 'df'}, optional
            The function (``'R'`` and ``'F'`` are ``'sf'`` and ``'ff'``).
            Defaults to ``'sf'``.
        alpha_ci : float, optional
            The total tail probability: 0.05 gives a 95% credible
            interval. Defaults to 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            A two-sided interval (``[lower, upper]`` on a last axis) or one
            end of it at level ``alpha_ci``.
        point : bool, optional
            Return the posterior median of the function instead.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> model.cri(100).round(3)
        array([0.441, 0.878])
        """
        if on not in _ON:
            raise ValueError(
                f"on must be one of {', '.join(_ON)}; got {on!r}."
            )
        fname = _ON[on]
        levels = [0.5] if point else _levels(alpha_ci, bound)
        x = np.asarray(x, dtype=float)
        if fname in ("ff", "Hf"):
            # from the bounds on sf, so the three always agree
            flipped = [1 - q for q in levels]
            s = self._function_quantiles("sf", x, flipped)
            out = 1 - s if fname == "ff" else -np.log(s)
        else:
            out = self._function_quantiles(fname, x, levels)
        if point:
            return out[..., 0]
        return _shape_bound(out, bound)

    def _function_quantiles(
        self, fname: str, x: npt.NDArray, levels: list[float]
    ) -> npt.NDArray:
        nan = np.isnan(x)
        xs = np.where(nan, 0.0, x)
        with np.errstate(all="ignore"):
            exact = None
            if fname != "df":
                exact = self._monotone(
                    lambda pts: self._sub(pts)._components(fname, xs), levels
                )
            if exact is not None:
                out = exact
            else:
                out = self._chunks(
                    fname,
                    xs,
                    lambda v: np.asarray(
                        self.posterior.wquantile(v, levels)
                    ).T,
                ).T
        out = np.asarray(out, dtype=float).reshape(len(x), len(levels))
        out[nan] = np.nan
        return out

    @keeps_query_shape
    def quantile_cri(
        self,
        p: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        point: bool = False,
    ) -> npt.NDArray:
        """
        The credible interval on a quantile (a B-life: ``p=0.1`` for B10):
        the range in which the population's B-life lies with posterior
        probability ``1 - alpha_ci``. ``point=True`` gives its posterior
        median, the best single guess at the B-life.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> model.quantile_cri(0.1).round(1)
        array([12.9, 81.1])
        """
        u = np.asarray(p, dtype=float)
        levels = [0.5] if point else _levels(alpha_ci, bound)
        outside = warn_outside_unit_interval(u)
        out = np.full((len(u), len(levels)), np.nan)
        for i, ui in enumerate(u):
            if outside[i] or np.isnan(ui):
                continue

            def g(pts: npt.NDArray, ui: float = ui) -> npt.NDArray:
                th = {
                    n: pts[:, j]
                    for j, n in enumerate(self.posterior.parameter_names)
                }
                with np.errstate(all="ignore"):
                    return np.asarray(
                        quantity_at(self.dist, ("q", ui), th), dtype=float
                    )[:, None]

            exact = self._monotone(g, levels)
            if exact is not None:
                out[i] = exact[0]
                continue
            vals = g(self.posterior.points)[:, 0]
            out[i] = self.posterior.wquantile(vals, levels)
        if point:
            return out[..., 0]
        return _shape_bound(out, bound)

    def mean_cri(
        self, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        """
        The credible interval on the population's mean life. An end is
        infinite where the posterior gives that much probability to
        parameters with an infinite mean.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> model.mean_cri().round(0)
        array([122., 770.])
        """
        levels = _levels(alpha_ci, bound)
        exact = self._monotone(
            lambda pts: self._sub(pts)._component_means()[:, None], levels
        )
        if exact is not None:
            out = exact[0]
        else:
            m = self._component_means()
            out = self.posterior.wquantile(m, levels)
        out = np.asarray(out, dtype=float)
        return out if bound == "two-sided" else out[0]

    def param_cri(
        self,
        name: str | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        The credible interval on a parameter (by name; every parameter of
        :attr:`parameter_names` and the extras, one row each, by default).

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> model.param_cri("failure_rate").round(5)
        array([0.0013 , 0.00819])
        """
        levels = _levels(alpha_ci, bound)
        if name is None:
            names = self.parameter_names + list(self.extras)
            rows = [self.param_cri(n, alpha_ci, bound) for n in names]
            return np.vstack(rows) if bound == "two-sided" else np.hstack(rows)
        out = np.asarray(self.posterior.quantile(name, levels), dtype=float)
        return out if bound == "two-sided" else out[0]

    def cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: a Bayesian fit has credible intervals, not confidence
        bounds; use :meth:`cri`."""
        raise ValueError(
            "A Bayesian fit has credible intervals, not confidence bounds: "
            "use cri(x, on=...) (and quantile_cri, mean_cri, param_cri). "
            f"For confidence bounds fit with {self.dist.name}.fit(...)."
        )

    def quantile_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses, pointing to :meth:`quantile_cri`."""
        raise ValueError(
            "A Bayesian fit has credible intervals: use quantile_cri(p)."
        )

    def mean_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses, pointing to :meth:`mean_cri`."""
        raise ValueError(
            "A Bayesian fit has credible intervals: use mean_cri()."
        )

    def param_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses, pointing to :meth:`param_cri`."""
        raise ValueError(
            "A Bayesian fit has credible intervals: use param_cri(name)."
        )

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------
    def update(
        self,
        x: Any = None,
        c: Any = None,
        n: Any = None,
        t: Any = None,
        *,
        tl: Any = None,
        tr: Any = None,
        xl: Any = None,
        xr: Any = None,
        replace: bool = False,
        method: str | None = None,
    ) -> BayesianParametric:
        """
        The posterior with more data: exact, the original prior times the
        likelihood of all the data, so updating in steps gives the same
        posterior as fitting everything at once.

        Parameters
        ----------
        x, c, n, t, tl, tr, xl, xr
            The new data, as for ``fit`` (``x`` may be a ``SurpyvalData``).
        replace : bool, optional
            ``False`` (the default) adds the new rows to the old.
            ``True`` takes them as the whole current record in place of
            the old data: use it when units that were running are still
            running (or have since failed), so that no unit counts twice.
        method : str, optional
            The engine; by default the one used before.

        Examples
        --------
        >>> import surpyval as sp
        >>> prior = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}
        >>> q1 = sp.Exponential.fit_bayes([100, 250], priors=prior)
        >>> both = q1.update([400])
        >>> once = sp.Exponential.fit_bayes([100, 250, 400], priors=prior)
        >>> bool(abs(both.mean() - once.mean()) < 1e-9)
        True
        """
        from surpyval.bayes.fitting import fit as fit_bayes

        if isinstance(x, SurpyvalData):
            new = x
        else:
            new = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)
        from surpyval.utils.weights import (
            BAYES_INSTEAD,
            BAYES_WHY,
            refuse_weighted_data,
        )

        refuse_weighted_data(new, "update", BAYES_WHY, BAYES_INSTEAD)
        if self.data is None and not replace:
            raise ValueError(
                "This model has no data to add to (it was built from "
                "draws without data); pass replace=True to fit the new data "
                "with its priors."
            )
        if replace or self.data is None:
            data = new
        else:
            data = _combine(self.data, new)
        free = self.diagnostics.get("free_parameters")
        fixed = {k: v for k, v in self.fixed.items()}
        if free is None:
            fixed = {}
        m = method or ("auto" if self.method == "conjugate" else self.method)
        if m == "draws":
            m = "auto"
        model = fit_bayes(
            self.dist,
            data,
            priors=self.priors,
            offset=self.offset,
            lfp=self.lfp,
            zi=self.zi,
            fixed={
                k: v for k, v in fixed.items() if k not in self.priors.fixed()
            },
            method=m,
        )
        model.unit = self.unit
        return model

    def as_prior(
        self,
        weight: float = 1.0,
        param: Any = None,
        widen: float = 1.0,
    ) -> Any:
        """
        This posterior as the prior of another fit.

        With the data at hand, and the whole posterior wanted (no
        ``param``, no ``widen``), it is exact: this fit's priors times its
        data's likelihood raised to ``weight``. Otherwise it is a normal on
        the parameters' unconstrained scale fitted to the posterior (its
        covariance divided by ``weight`` and widened by ``widen``), and
        its description says so.

        Examples
        --------
        >>> import surpyval as sp
        >>> old = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> type(old.as_prior()).__name__
        'PriorSpec'
        """
        weight = float(weight)
        if not 0 < weight <= 1:
            raise ValueError(f"weight must be in (0, 1]; got {weight}.")
        if self.data is not None and param is None and widen == 1:
            spec = PriorSpec(self.priors.single, self.priors.joint)
            spec.joint = spec.joint + [
                PowerPrior(
                    self.data,
                    weight,
                    dist=self.dist,
                    said=f"the earlier fit's data at weight {weight:g}",
                )
            ]
            return spec
        post = self.posterior
        ts = post.transforms or [
            Transform.for_bounds(b) for b in self._parameter_bounds()
        ]
        names = post.free_names
        phi = np.stack(
            [
                t.to_phi(post.points[:, post.parameter_names.index(n)])
                for t, n in zip(ts, names)
            ],
            axis=1,
        )
        w = post.weights
        mean = w @ phi
        cov = (phi - mean).T @ ((phi - mean) * w[:, None])
        cov = np.atleast_2d(cov) * widen**2 / weight
        said = (
            f"a normal summary of an earlier Bayesian {self.dist.name} fit,"
            f" weight {weight:g}"
            + (f", widened x{widen:g}" if widen != 1 else "")
        )
        if param is None:
            return MultivariateNormal(names, mean, cov, said)
        wanted = [param] if isinstance(param, str) else list(param)
        idx = [names.index(p) for p in wanted]
        if len(idx) == 1:
            i = idx[0]
            return NormalOnScale(
                loc=float(mean[i]),
                scale=float(np.sqrt(cov[i, i])),
                kind=ts[i].kind,
                said=f"{wanted[0]} from {said}",
            )
        return MultivariateNormal(
            wanted, mean[idx], cov[np.ix_(idx, idx)], said
        )

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------
    @property
    def mle(self) -> Any:
        """The maximum-likelihood fit to the same data (None where it
        fails or there are no data)."""
        if not self._mle_known:
            from surpyval.bayes.fitting import _mle_start

            self._mle_fit = (
                None
                if self.data is None or self.dist.discrete
                else _mle_start(
                    self.dist,
                    self.data,
                    self.offset,
                    self.lfp,
                    self.zi,
                    self.fixed,
                )
            )
            self._mle_known = True
        return self._mle_fit

    def compare(self, alpha_ci: float = 0.05) -> Comparison:
        """
        The prior, the maximum-likelihood fit (with its confidence
        interval) and the posterior, side by side for each parameter: what
        the prior did, and how much came from the data.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> x = np.r_[2100, 3400, 4050, np.linspace(1500, 6000, 37)]
        >>> c = np.r_[0, 0, 0, np.ones(37)]
        >>> model = sp.Weibull.fit_bayes(
        ...     x, c, priors={"beta": sp.prior.around(2, between=(1.5, 3))})
        >>> table = model.compare().table
        >>> sorted(table)
        ['alpha', 'beta']
        """
        check_alpha_ci(alpha_ci)
        bound = self._bound()
        mle = self.mle
        rows = {}
        for name in bound.free_names:
            t = bound.transforms[bound.names.index(name)]
            prior_iv = _prior_interval(bound, name, t, alpha_ci)
            mle_val = mle_iv = None
            if mle is not None:
                try:
                    if name in self.dist.parameter_names:
                        mle_val = float(
                            mle.params[self.dist.parameter_names.index(name)]
                        )
                    else:
                        mle_val = float(getattr(mle, name))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        iv = np.asarray(
                            mle.param_cb(name, alpha_ci=alpha_ci), float
                        )
                    mle_iv = (float(iv[0]), float(iv[1]))
                except (ValueError, AttributeError, KeyError):
                    pass
            post_iv = tuple(
                float(v)
                for v in self.posterior.quantile(
                    name, [alpha_ci / 2, 1 - alpha_ci / 2]
                )
            )
            rows[name] = {
                "prior": bound.descriptions.get(name, Flat().describe(t)),
                "prior_interval": prior_iv,
                "mle": mle_val,
                "mle_interval": mle_iv,
                "posterior_median": float(self.posterior.quantile(name, 0.5)),
                "posterior_interval": post_iv,
            }
        return Comparison(rows, alpha_ci)

    def _quantity_draws(
        self, on: str, p: float | None, x: float | None
    ) -> tuple[npt.NDArray, npt.NDArray, str]:
        th = self._theta()
        if on == "qf":
            assert p is not None
            v = quantity_at(self.dist, ("q", p), th)
            label = f"{p:.0%} quantile (B{100 * p:g} life)"
        elif on in ("sf", "ff"):
            assert x is not None
            v = quantity_at(self.dist, ("R", x), th)
            if on == "ff":
                v = 1 - v
            label = f"{'reliability' if on == 'sf' else 'failure probability'}"
            label += f" at {_fmt(x)}"
        elif on == "mean":
            v = self._component_means()
            label = "mean life"
        elif on in self.posterior.parameter_names:
            v = th[on]
            label = on
        else:
            raise ValueError(
                "on must be 'qf' (with p=), 'sf' or 'ff' (with x=), "
                f"'mean', or a parameter name; got {on!r}."
            )
        return (
            np.asarray(v, dtype=float),
            self.posterior.weights,
            label,
        )

    def compare_to(
        self,
        other: Any,
        on: str = "sf",
        p: float = 0.10,
        x: float | None = None,
    ) -> Comparison2:
        """
        The posterior probability that this model's quantity is greater
        than the other model's: for example that this design's B10 life
        is longer than the old one's.

        Parameters
        ----------
        other : BayesianParametric or Parametric
            Another Bayesian fit (its posterior is taken as independent of
            this one's), or a maximum-likelihood fit with a covariance
            (its estimate of the quantity is taken as normal on the log
            scale, or the quantity's own scale where it can be negative,
            with its delta-method standard error: the large-sample
            posterior under a flat prior).
        on : str, optional
            ``"sf"`` (the default) or ``"ff"`` at ``x``; ``"qf"``, a
            quantile, with ``p`` (``on="qf", p=0.10`` for the B10 life);
            ``"mean"``; or a parameter name.
        p : float, optional
            The quantile's probability. Defaults to 0.10.
        x : float, optional
            The time for ``on="sf"`` or ``"ff"``.

        Returns
        -------
        Comparison2
            ``probability``, the assumption used and a plain summary.

        Examples
        --------
        >>> import surpyval as sp
        >>> prior = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}
        >>> new = sp.Exponential.fit_bayes([300, 450, 800], priors=prior)
        >>> old = sp.Exponential.fit_bayes([100, 250, 400], priors=prior)
        >>> round(new.compare_to(old, on="qf", p=0.10).probability, 2)
        0.78
        """
        if on in ("sf", "ff") and x is None:
            raise ValueError(
                f"on={on!r} needs the time x=; for a B-life use on='qf' "
                "with p= (p=0.10 for B10)."
            )
        a, wa, label = self._quantity_draws(on, p, x)
        a = np.where(np.isnan(a), np.inf, a)
        if isinstance(other, BayesianParametric):
            b, wb, _ = other._quantity_draws(on, p, x)
            b = np.where(np.isnan(b), np.inf, b)
            order = np.argsort(b, kind="stable")
            bs, cw = b[order], np.cumsum(wb[order])
            # P(B < a) + half of P(B == a), for each a
            lo = np.searchsorted(bs, a, side="left")
            hi = np.searchsorted(bs, a, side="right")
            below = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
            upto = np.where(hi > 0, cw[np.maximum(hi - 1, 0)], 0.0)
            prob = float(wa @ (0.5 * (below + upto)))
            assumption = (
                "the two posteriors taken as independent (separate data)"
            )
        else:
            est, se, log_scale = _mle_quantity(other, on, p, x)
            with np.errstate(all="ignore"):
                if log_scale:
                    z = (np.log(a) - np.log(est)) / se
                else:
                    z = (a - est) / se
            z = np.where(np.isnan(z), np.inf, z)
            prob = float(wa @ stats.norm.cdf(z))
            assumption = (
                "the other model's estimate taken as normal on the "
                + ("log" if log_scale else "natural")
                + " scale with its delta-method standard error (its "
                "large-sample posterior under a flat prior), independent "
                "of this posterior"
            )
        return Comparison2(prob, label, assumption)

    # ------------------------------------------------------------------
    # Checks and summaries
    # ------------------------------------------------------------------
    def prior_check(self) -> PriorCheck:
        """What this model's prior says before any data (see
        ``prior.check``)."""
        if self.data is None and self.method != "draws":
            return PriorCheck(self)
        from surpyval.bayes.fitting import prior_model

        return prior_model(
            self.dist,
            self.priors,
            lfp=self.lfp,
            zi=self.zi,
            fixed={
                k: v
                for k, v in self.fixed.items()
                if k not in self.priors.fixed()
            }
            or None,
        ).prior_check()

    def _warn_conflicts(self) -> None:
        bound = self._bound()
        msgs = []
        for name, p in self.priors.single.items():
            if p.is_fixed or not p.proper:
                continue
            t = bound.transforms[bound.names.index(name)]
            iv = _prior_interval(bound, name, t, 0.01)
            if iv is None:
                continue
            med = float(self.posterior.quantile(name, 0.5))
            if not iv[0] <= med <= iv[1]:
                iv90 = _prior_interval(bound, name, t, 0.1)
                assert iv90 is not None
                msgs.append(
                    f"{name} (prior 90% interval {_fmt(iv90[0])}-"
                    f"{_fmt(iv90[1])}; the posterior median is {_fmt(med)})"
                )
        if msgs:
            warnings.warn(
                "The data disagree with the prior on "
                + "; ".join(msgs)
                + ". Neither is taken as right: check the prior "
                "(prior.check) and the data, and compare() the two.",
                stacklevel=3,
            )

    def summary(self, alpha_ci: float = 0.05) -> str:
        """The fit in plain words: the data, the priors, the posterior's
        medians and credible intervals, a B-life and the mean life, what
        the prior did against the maximum-likelihood fit, and how the
        posterior was computed."""
        return _summary(self, alpha_ci)

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot(
        self,
        heuristic: str = "Nelson-Aalen",
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
        ax: Any = None,
        plot_mle: bool = True,
        show_censored: bool = False,
        color: Any = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        A probability plot: the data at their plotting positions, the
        predictive curve, its credible band (``plot_bounds``) and, dashed,
        the maximum-likelihood fit (``plot_mle``), so what the prior did is
        visible at a glance.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import surpyval as sp
        >>> model = sp.Weibull.fit_bayes(
        ...     [2100, 3400, 4050, 5000, 5000], c=[0, 0, 0, 1, 1],
        ...     priors={"beta": sp.prior.around(2, between=(1.5, 3))})
        >>> ax = model.plot()
        >>> plt.close("all")
        """
        from surpyval.bayes.plotting import probability_plot

        return probability_plot(
            self,
            heuristic,
            plot_bounds,
            alpha_ci,
            ax,
            plot_mle,
            show_censored,
            color,
            label,
            **kwargs,
        )

    def plot_posterior(self, fig: Any = None) -> Any:
        """
        Each parameter's prior (dashed, where it has a density of its own)
        and posterior (filled), and with two free parameters their joint
        posterior's contours with the maximum-likelihood fit marked.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import surpyval as sp
        >>> model = sp.Weibull.fit_bayes(
        ...     [2100, 3400, 4050, 5000, 5000], c=[0, 0, 0, 1, 1],
        ...     priors={"beta": sp.prior.around(2, between=(1.5, 3))})
        >>> fig = model.plot_posterior()
        >>> plt.close("all")
        """
        from surpyval.bayes.plotting import posterior_plot

        return posterior_plot(self, fig)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_log_post"] = None
        post = copy(state["posterior"])
        post.log_density_fn = None
        state["posterior"] = post
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._attach_density()

    def to_dict(self) -> dict:
        """
        A strict-JSON dictionary of the model: the distribution, the
        priors, the data, the method and its settings, and the posterior
        itself, so a restored model predicts exactly as this one without
        recomputing. ``surpyval.from_dict`` reads it.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Exponential.fit_bayes(
        ...     [100, 250, 400], priors={"failure_rate":
        ...         sp.Gamma.from_params([2.0, 500.0])})
        >>> again = sp.from_dict(model.to_dict())
        >>> bool(again.sf(100) == model.sf(100))
        True
        """
        import surpyval

        diag = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in self.diagnostics.items()
        }
        if self._log_z is not None:
            diag["log_z"] = float(self._log_z)
        out = {
            "model": self.model_tag,
            "distribution": self.dist.name,
            "offset": self.offset,
            "lfp": self.lfp,
            "zi": self.zi,
            "fixed": self.fixed,
            "method": self.method,
            "priors": self.priors.to_dict(),
            "data": None if self.data is None else self.data.to_dict(),
            "posterior": self.posterior.to_dict(),
            "log_evidence": (
                None
                if not np.isfinite(self.log_evidence)
                else self.log_evidence
            ),
            "diagnostics": _jsonable(diag),
            "exact": self._exact,
            "surpyval_version": surpyval.__version__,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianParametric:
        """Rebuild a model from :meth:`to_dict`."""
        import surpyval

        require_model_tag(model_dict, cls.model_tag, "a Bayesian fit")
        dist = getattr(surpyval, model_dict["distribution"])
        data = (
            None
            if model_dict.get("data") is None
            else SurpyvalData.from_dict(model_dict["data"])
        )
        ev = model_dict.get("log_evidence")
        model = cls(
            dist,
            Posterior.from_dict(model_dict["posterior"]),
            data=data,
            priors=PriorSpec.from_dict(model_dict["priors"]),
            offset=model_dict["offset"],
            lfp=model_dict["lfp"],
            zi=model_dict["zi"],
            fixed=model_dict.get("fixed"),
            method=model_dict["method"],
            log_evidence=float("nan") if ev is None else float(ev),
            diagnostics=model_dict.get("diagnostics"),
            exact=model_dict.get("exact"),
        )
        lz = model.diagnostics.get("log_z")
        model._log_z = None if lz is None else float(lz)
        model.unit = model_dict.get("unit")
        return model


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _combine(a: SurpyvalData, b: SurpyvalData) -> SurpyvalData:
    """The rows of two datasets together."""

    def two(d: SurpyvalData) -> npt.NDArray:
        x = np.asarray(d.x, dtype=float)
        return np.c_[x, x] if x.ndim == 1 else x

    x = np.vstack([two(a), two(b)])
    c = np.r_[a.c, b.c]
    n = np.r_[a.n, b.n]
    t = np.vstack([a.t, b.t])
    if np.all(x[:, 0] == x[:, 1]):
        x = x[:, 0]
    return SurpyvalData(x=x, c=c, n=n, t=t)


def _prior_interval(
    bound: Any, name: str, t: Transform, alpha: float
) -> tuple[float, float] | None:
    """The equal-tailed prior interval of one parameter where its prior is
    given on it alone."""
    p = bound.spec.single.get(name)
    if isinstance(p, NormalOnScale):
        m, s = p._resolved(t)
        z = stats.norm.ppf(1 - alpha / 2)
        return (float(t.to_theta(m - z * s)), float(t.to_theta(m + z * s)))
    if isinstance(p, DistributionPrior):
        q = np.asarray(p.model.qf([alpha / 2, 1 - alpha / 2]), dtype=float)
        return (float(q[0]), float(q[1]))
    for j in bound.spec.joint:
        if isinstance(j, MultivariateNormal) and name in (j.names or []):
            i = j.names.index(name)  # type: ignore[union-attr]
            m, s = j.mean[i], np.sqrt(j.cov[i, i])
            z = stats.norm.ppf(1 - alpha / 2)
            return (
                float(t.to_theta(m - z * s)),
                float(t.to_theta(m + z * s)),
            )
    return None


def _mle_quantity(
    other: Any, on: str, p: float | None, x: float | None
) -> tuple[float, float, bool]:
    """(estimate, standard error, on the log scale) of a quantity of a
    maximum-likelihood model, by the delta method on its covariance."""
    if not hasattr(other, "covariance") or not hasattr(other, "dist"):
        raise ValueError(
            "compare_to takes another Bayesian fit or a maximum-likelihood "
            f"fit with a covariance; got {type(other).__name__}."
        )
    if getattr(other, "offset", False):
        raise ValueError(
            "An offset model's covariance leaves out its offset, so its "
            "quantities have no standard error to compare with; fit it with "
            "fit_bayes and compare the two Bayesian fits."
        )
    try:
        cov = np.asarray(other.covariance(), dtype=float)
    except ValueError as e:
        raise ValueError(
            f"The other model has no covariance ({e}); fit it with "
            "fit_bayes and compare the two Bayesian fits."
        ) from None
    names, _ = _names_of(other)
    vals = np.array([_value_of(other, n) for n in names], dtype=float)

    def q(v: npt.NDArray) -> float:
        th = {n: np.atleast_1d(v[i]) for i, n in enumerate(names)}
        if on == "qf":
            return float(np.ravel(quantity_at(other.dist, ("q", p), th))[0])
        if on in ("sf", "ff"):
            r = float(np.ravel(quantity_at(other.dist, ("R", x), th))[0])
            return r if on == "sf" else 1 - r
        if on == "mean":
            return float(np.ravel(quantity_at(other.dist, ("mean",), th))[0])
        if on in names:
            return float(v[names.index(on)])
        raise ValueError(f"The other model has no {on!r}.")

    est = q(vals)
    grad = np.zeros(len(vals))
    for i in range(len(vals)):
        h = 1e-6 * max(abs(vals[i]), 1e-8)
        up, dn = vals.copy(), vals.copy()
        up[i] += h
        dn[i] -= h
        grad[i] = (q(up) - q(dn)) / (2 * h)
    var = float(grad @ cov[: len(vals), : len(vals)] @ grad)
    log_scale = est > 0 and on not in ("sf", "ff")
    if log_scale:
        se = np.sqrt(var) / est
    else:
        se = np.sqrt(var)
    if not se > 0:
        raise ValueError(
            "The other model's quantity has no standard error (zero "
            "variance); it cannot be compared as uncertain."
        )
    return est, float(se), log_scale


def _names_of(model: Any) -> tuple[list[str], Any]:
    names = list(model.dist.parameter_names)
    if getattr(model, "lfp", False):
        names.append("lfp_p")
    if getattr(model, "zi", False):
        names.append("f0")
    return names, None


def _value_of(model: Any, name: str) -> float:
    if name in model.dist.parameter_names:
        return float(model.params[model.dist.parameter_names.index(name)])
    return float(getattr(model, name))


class Comparison:
    """Prior, maximum likelihood and posterior side by side (from
    :meth:`BayesianParametric.compare`); ``table`` holds the numbers."""

    def __init__(self, table: dict, alpha_ci: float) -> None:
        self.table = table
        self.alpha_ci = alpha_ci

    def __repr__(self) -> str:
        lvl = f"{1 - self.alpha_ci:.0%}"
        lines = [f"Prior, maximum likelihood and posterior ({lvl} intervals)"]
        for name, r in self.table.items():
            lines.append(f"  {name}")
            pi = r["prior_interval"]
            lines.append(
                f"    prior       {r['prior']}"
                + (
                    f"; {lvl} {_fmt(pi[0])} - {_fmt(pi[1])}"
                    if pi is not None
                    else ""
                )
            )
            if r["mle"] is not None:
                mi = r["mle_interval"]
                ci = (
                    f" ({lvl} confidence interval {_fmt(mi[0])} - "
                    f"{_fmt(mi[1])})"
                    if mi is not None
                    else ""
                )
                lines.append(f"    max. lik.   {_fmt(r['mle'])}{ci}")
            else:
                lines.append("    max. lik.   none (the fit fails)")
            po = r["posterior_interval"]
            lines.append(
                f"    posterior   {_fmt(r['posterior_median'])} "
                f"({lvl} credible interval {_fmt(po[0])} - {_fmt(po[1])})"
            )
        return "\n".join(lines)


class Comparison2:
    """The posterior probability that one model's quantity exceeds
    another's (from :meth:`BayesianParametric.compare_to`)."""

    def __init__(self, probability: float, quantity: str, assumption: str):
        self.probability = float(probability)
        self.quantity = quantity
        self.assumption = assumption

    def __float__(self) -> float:
        return self.probability

    def __repr__(self) -> str:
        return (
            f"There is a {self.probability:.1%} chance that this model's "
            f"{self.quantity} is greater than the other's.\n"
            f"Assumption: {self.assumption}."
        )


class PriorCheck:
    """What a prior implies before any data (from ``prior.check``):
    ``table`` holds ``{quantity: (median, lower, upper)}`` at 90%, and
    ``model`` is the prior as a model."""

    def __init__(self, model: BayesianParametric, alpha_ci: float = 0.1):
        self.model = model
        self.alpha_ci = alpha_ci
        table: dict[str, tuple[float, float, float]] = {}
        for p, lab in ((0.01, "B1"), (0.1, "B10"), (0.5, "B50")):
            med = float(model.quantile_cri(p, point=True))
            lo, hi = model.quantile_cri(p, alpha_ci=alpha_ci)
            table[lab] = (med, float(lo), float(hi))
        if not model.dist.discrete:
            lo, med, hi = model.posterior.wquantile(
                model._component_means(),
                [alpha_ci / 2, 0.5, 1 - alpha_ci / 2],
            )
            table["mean"] = (float(med), float(lo), float(hi))
        for name in model.posterior.free_names:
            lo, med, hi = model.posterior.quantile(
                name, [alpha_ci / 2, 0.5, 1 - alpha_ci / 2]
            )
            table[name] = (float(med), float(lo), float(hi))
        self.table = table

    def plot(self, ax: Any = None, k: int = 50, random_state: Any = 0) -> Any:
        """``k`` curves drawn from the prior on a probability plot."""
        from surpyval.bayes.plotting import prior_curves

        return prior_curves(self.model, ax, k, random_state)

    def __repr__(self) -> str:
        lvl = f"{1 - self.alpha_ci:.0%}"
        lines = [
            f"What the prior says before any data ({self.model.dist.name})",
            f"  {'':10}{'median':>12}   {lvl} prior interval",
        ]
        for k, (med, lo, hi) in self.table.items():
            lines.append(f"  {k:10}{_fmt(med):>12}   {_fmt(lo)} - {_fmt(hi)}")
        return "\n".join(lines)


def _summary(model: BayesianParametric, alpha_ci: float) -> str:
    lvl = f"{1 - alpha_ci:.0%}"
    unit = f" {model.unit}" if model.unit else ""
    lines = []
    head = f"{model.dist.name}: Bayesian fit"
    if model.data is not None:
        n = np.asarray(model.data.n)
        c = np.asarray(model.data.c)
        total = int(n.sum())
        failed = int(n[c == 0].sum())
        running = int(n[c == 1].sum())
        other = total - failed - running
        words = f"{total} items ({failed} failed, {running} running"
        if other:
            words += f", {other} left or interval censored"
        head += f", {words})"
        if np.isfinite(np.asarray(model.data.t)).any():
            head += ", truncated"
    else:
        head += " of the prior alone (no data)"
    lines.append(head)
    try:
        bound = model._bound()
        descr = bound.descriptions
        flat = bound.flat_names
        tmap = dict(zip(bound.names, bound.transforms))
    except ValueError:
        descr, flat, tmap = {}, [], {}
    first = True
    shown_joint = set()
    for name in model.posterior.free_names:
        if name in descr:
            d = descr[name]
            if d in shown_joint:
                continue
            if not model.priors.single.get(name):
                shown_joint.add(d)
                text = d
            else:
                text = f"{name} {d}"
        elif name in flat:
            t = tmap.get(name)
            scale = t.scale_name if t is not None else "its own scale"
            text = (
                f"{name}: no prior given; flat on {scale} (the data alone "
                "identify it)"
            )
        else:
            text = f"{name}: from the draws given"
        lines.append(("Prior:      " if first else "            ") + text)
        first = False
    for j in model.priors.joint:
        if j.is_likelihood:
            lines.append(
                ("Prior:      " if first else "            ") + j.describe()
            )
            first = False
    for k, v in model.fixed.items():
        lines.append(f"            {k}: fixed at {_fmt(v)}")
    lines.append(f"Posterior   {'median':>10}   {lvl} credible interval")
    for name in model.posterior.free_names:
        lo, med, hi = model.posterior.quantile(
            name, [alpha_ci / 2, 0.5, 1 - alpha_ci / 2]
        )
        lines.append(f"  {name:<10}{_fmt(med):>10}   {_fmt(lo)} - {_fmt(hi)}")
    life = float(model.dist.support[0]) == 0
    lab = "B10 life" if life else "10% point"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            b10 = float(model.quantile_cri(0.1, point=True))
            b_lo, b_hi = model.quantile_cri(0.1, alpha_ci=alpha_ci)
            lines.append(
                f"{lab:<12}{_fmt(b10):>10}{unit}   {_fmt(b_lo)} - "
                f"{_fmt(b_hi)}{unit}"
            )
        except ValueError:
            pass
        if not model.dist.discrete:
            lo, med, hi = model.posterior.wquantile(
                model._component_means(),
                [alpha_ci / 2, 0.5, 1 - alpha_ci / 2],
            )
            lab = "Mean life" if life else "Mean"
            lines.append(
                f"{lab:<12}{_fmt(med):>10}{unit}   {_fmt(lo)} - "
                f"{_fmt(hi)}{unit}"
            )
    lines += _what_the_prior_did(model, alpha_ci)
    lines.append(_how_computed(model))
    return "\n".join(lines)


def _what_the_prior_did(model: BayesianParametric, alpha_ci: float) -> list:
    if model.data is None or model.method == "draws":
        return []
    mle = model.mle
    if mle is None:
        return [
            "The maximum-likelihood fit fails on these data, so the "
            "answer rests on the prior."
        ]
    lvl = f"{1 - alpha_ci:.0%}"
    parts = []
    notes = []
    try:
        bound = model._bound()
        flat = set(bound.flat_names)
    except ValueError:
        flat = set()
    for name in model.posterior.free_names:
        try:
            if name in model.dist.parameter_names:
                v = float(mle.params[model.dist.parameter_names.index(name)])
            else:
                v = float(getattr(mle, name))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                iv = np.asarray(mle.param_cb(name, alpha_ci=alpha_ci), float)
        except (ValueError, AttributeError, KeyError):
            continue
        parts.append(
            f"{name} {_fmt(v)} ({lvl} CI {_fmt(iv[0])}-{_fmt(iv[1])})"
        )
        lo, hi = model.posterior.quantile(
            name, [alpha_ci / 2, 1 - alpha_ci / 2]
        )
        width = float(iv[1] - iv[0])
        if name in flat or not np.isfinite(width) or width <= 0:
            if name in flat:
                notes.append(f"{name} came from the data alone")
            continue
        change = 1 - (hi - lo) / width
        med = float(model.posterior.quantile(name, 0.5))
        if not iv[0] <= med <= iv[1]:
            notes.append(
                f"the prior moved {name} outside the maximum-likelihood "
                "interval"
            )
        elif change > 0.2:
            notes.append(
                f"the prior narrowed {name}'s interval by about "
                f"{100 * change:.0f}%"
            )
        else:
            notes.append(f"{name} came mostly from the data")
    if not parts:
        return []
    out = [
        "Compared with the maximum-likelihood fit: " + "; ".join(parts) + ";"
    ]
    if notes:
        out.append(", ".join(notes) + ".")
    return out


def _how_computed(model: BayesianParametric) -> str:
    d = model.diagnostics
    if model.method == "conjugate":
        return f"Computed exactly in closed form ({d.get('case', '')})."
    if model.method == "grid":
        m = d.get("points_per_dim") or []
        grid = " x ".join(str(v) for v in m)
        err = d.get("integration_error", float("nan"))
        return (
            f"Computed exactly on two offset {grid} grids (integration "
            f"error < {max(float(err), 1e-4):.2%})."
        )
    if model.method == "laplace":
        return (
            "Approximated by a normal at the posterior's peak (Laplace); "
            "with few failures this can be poor: check it with the grid."
        )
    return f"From {d.get('draws', len(model.posterior.weights))} draws."
