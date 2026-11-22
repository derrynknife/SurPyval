"""Bayesian fits of the parametric regressions: priors on the coefficients
and on the baseline's parameters.

``WeibullPH.fit_bayes(x, Z, c, priors=...)``, ``AFT(dist).fit_bayes`` and
``AcceleratedLife(dist, life_model).fit_bayes`` come here. The posterior
is the fitter's own likelihood (every censoring and truncation ``fit``
takes) times the priors, sampled by SurPyval's No-U-Turn sampler
(:mod:`surpyval.bayes.engines.nuts`) and checked for convergence
(:mod:`surpyval.bayes.convergence`).

How it is computed
------------------
- The maximum-likelihood fit is made first, as a user would call it. It
  gives the data as the fitter reads them, the parameters' names and
  ranges, the comparison in the summary, and the sampler's start.
- The parameters are sampled on their unconstrained scale (the log of a
  positive parameter), and, within that, in the coordinates of the
  posterior's curvature at its peak (:class:`~.engines.nuts.Affine`):
  a baseline's scale and a coefficient are strongly correlated whenever
  a covariate's mean is far from 0 (a temperature in kelvin), which a
  diagonal mass matrix cannot follow. The covariates are used as given;
  nothing is standardised in the model.
- A coefficient of ``exp(beta'Z)`` with no prior gets
  ``Normal(0, 2.5 / sd)``, ``sd`` its covariate's standard deviation in
  the data (:data:`COEFFICIENT_SCALE`): weakly informative, and the same
  whatever the covariate's unit. Every other parameter with no prior is
  flat on its unconstrained scale, as for ``Weibull.fit_bayes``. The
  defaults used are listed on the result (``default_priors``) and in its
  summary.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from autograd import value_and_grad

from surpyval.bayes._lean import LogPosterior, Term
from surpyval.bayes.posterior import Posterior
from surpyval.bayes.prior import (
    BoundPriors,
    MultivariateNormal,
    NormalOnScale,
    PriorSpec,
)
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.shapes import covariate_rows, keeps_query_shape
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.warnings import ignoring_warnings

#: The default prior on a coefficient of ``exp(beta'Z)`` is
#: ``Normal(0, COEFFICIENT_SCALE / sd)``, ``sd`` the covariate's standard
#: deviation: a change of 2.5 in the log hazard (or log life) for one
#: standard deviation of the covariate is one prior standard deviation.
COEFFICIENT_SCALE = 2.5
#: The families ``fit_bayes`` covers, by the fitted model's ``kind``.
COVERED = (
    "Proportional Hazard",
    "Accelerated Failure Time",
    "Accelerated Life",
)
#: The most draws the predictive functions average over (evenly spaced
#: through the posterior's draws).
PREDICTIVE_DRAWS = 1000
_ON = ("sf", "ff", "Hf", "hf", "df")

_NOT_HERE = {
    "w": "weights (the posterior of a weighted pseudo-likelihood is not a "
    "posterior; use fit(..., w=) for the weighted estimate, or counts n "
    "for identical units)",
    "penalizer": "a penalty (a prior on the coefficients is the Bayesian "
    "penalty: priors={'coef_0': surpyval.Normal.from_params([0, s])}; "
    "fit_penalised gives the penalised maximum-likelihood fit)",
    "l1_ratio": "a penalty (use fit_penalised, or a prior on the "
    "coefficients)",
    "init": "a starting point (the sampler starts from the "
    "maximum-likelihood fit)",
    "center": "centring (the sampler works in the posterior's own "
    "coordinates, so the covariates are used as given)",
}


def refuse_options(what: str, options: dict) -> None:
    """Refuse the arguments of ``fit`` that a Bayesian regression does
    not take, each with what to do instead."""
    for k in options:
        if k in _NOT_HERE:
            raise ValueError(f"{what} does not take {k}: {_NOT_HERE[k]}.")
    if options:
        raise ValueError(
            f"{what} does not take {', '.join(sorted(options))}. It takes "
            "the data (x, Z, c, n, t, tl, tr), priors, fixed, and the "
            "sampler's chains, draws, warmup, random_state and n_jobs."
        )


def refuse_time_varying(Z: Any) -> None:
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if isinstance(Z, (CovariatePath, StepSchedule, dict)):
        raise ValueError(
            "fit_bayes does not take time-varying covariates in this "
            "version: it covers covariates fixed for each unit, with the "
            "proportional hazards, accelerated failure time and accelerated "
            "life fitters. Use fit_tvc for the maximum-likelihood fit."
        )


class RegressionLogPosterior(LogPosterior):
    """The log-posterior of a regression and its gradient at one point of
    the free parameters' unconstrained scale: the fitter's likelihood
    (by automatic differentiation) plus the priors."""

    def __init__(self, fitter: Any, data: SurpyvalData, bound: Any) -> None:
        self._fitter = fitter
        super().__init__(fitter.dist, data, bound)

    def _likelihood_term(
        self, data: SurpyvalData, weight: float
    ) -> tuple[Term, bool]:
        fitter = self._fitter

        def log_likelihood(z: Any) -> Any:
            return -weight * fitter.neg_ll(data, *self.theta(z))

        return value_and_grad(log_likelihood), False


def _coefficient_names(mle: Any) -> list[str]:
    """The parameters that are coefficients of ``exp(beta'Z)``: those of a
    proportional hazards or accelerated failure time model with the
    log-linear link. (An accelerated life model's are its life model's
    parameters, and a custom ``phi``'s need not be one per covariate.)"""
    from surpyval.univariate.regression._fit_skeleton import LogLinearPhi

    if mle.kind == "Accelerated Life":
        return []
    phi = getattr(mle.model, "phi", LogLinearPhi.phi)
    if phi is not LogLinearPhi.phi:
        return []
    names = list(mle.parameter_names)[mle.k_dist :]
    return names if len(names) == np.shape(mle.data.Z)[1] else []


def _default_priors(mle: Any, spec: PriorSpec, fixed: dict) -> dict:
    """``Normal(0, 2.5 / sd)`` for each coefficient with no prior."""
    covered = set(spec.single) | set(fixed)
    for j in spec.joint:
        covered |= set(j.names or [])
    Z = np.asarray(mle.data.Z, dtype=float)
    n = np.asarray(mle.data.n, dtype=float)
    out = {}
    for k, name in enumerate(_coefficient_names(mle)):
        if name in covered:
            continue
        mean = float(np.sum(n * Z[:, k]) / n.sum())
        sd = float(np.sqrt(np.sum(n * (Z[:, k] - mean) ** 2) / n.sum()))
        if not sd > 0:
            raise ValueError(
                f"The covariate of {name} is the same for every row, so "
                "the data say nothing about its coefficient and it has no "
                "scale for a default prior. Leave the column out, or give "
                f"the coefficient a prior (priors={{{name!r}: ...}})."
            )
        scale = COEFFICIENT_SCALE / sd
        out[name] = NormalOnScale(
            loc=0.0,
            scale=scale,
            kind="identity",
            said=f"default: Normal(0, {scale:.4g}), which is "
            f"{COEFFICIENT_SCALE:g} over the covariate's standard "
            f"deviation ({sd:.4g})",
        )
    return out


def _mle(fitter: Any, how: Any) -> Any:
    """The maximum-likelihood fit, quietly but for the rows it drops (a
    missing covariate value), which this fit drops too and says so; its
    refusal is this fit's, with the reason."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

    try:
        with collecting_warnings() as caught:
            with quiet_maximum_warnings(), np.errstate(all="ignore"):
                model = how()
    except ValueError as e:
        raise ValueError(
            "fit_bayes starts from the maximum-likelihood fit, which "
            f"refuses these data: {e}"
        ) from None
    for w in caught:
        if str(w.message).startswith("Dropped "):
            warnings.warn(
                str(w.message), w.category, stacklevel=caller_stacklevel()
            )
    return model


def fit(
    fitter: Any,
    x: Any,
    Z: Any,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    *,
    priors: Any = None,
    fixed: dict[str, float] | None = None,
    tl: Any = None,
    tr: Any = None,
    chains: int = 4,
    draws: int = 1000,
    warmup: int = 1000,
    random_state: Any = None,
    n_jobs: int | None = None,
    **options: Any,
) -> BayesianRegression:
    """The Bayesian fit of a parametric regression; see the fitters'
    ``fit_bayes``."""
    refuse_options("fit_bayes", options)
    refuse_time_varying(Z)
    mle = _mle(
        fitter,
        lambda: fitter.fit(x, Z, c, n, t, fixed=fixed, tl=tl, tr=tr),
    )
    sampler = dict(
        chains=chains,
        draws=draws,
        warmup=warmup,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return from_mle(mle, priors, sampler)


def fit_from_df(
    fitter: Any,
    df: Any,
    *,
    priors: Any = None,
    chains: int = 4,
    draws: int = 1000,
    warmup: int = 1000,
    random_state: Any = None,
    n_jobs: int | None = None,
    **columns: Any,
) -> BayesianRegression:
    """``fit_bayes`` on the columns of a DataFrame, named as
    ``fit_from_df`` names them."""
    for k in ("w_col", "penalizer", "l1_ratio", "init", "center"):
        if columns.get(k) is not None:
            refuse_options("fit_bayes_from_df", {k.replace("_col", ""): 1})
        columns.pop(k, None)
    mle = _mle(fitter, lambda: fitter.fit_from_df(df, **columns))
    sampler = dict(
        chains=chains,
        draws=draws,
        warmup=warmup,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return from_mle(mle, priors, sampler)


def _bind(mle: Any, priors: Any) -> tuple[BoundPriors, dict[str, str]]:
    """The priors (given and default) bound to the regression's
    parameters, and the defaults' descriptions by name."""
    if mle.kind not in COVERED:
        raise ValueError(
            f"fit_bayes covers the proportional hazards, accelerated "
            f"failure time and accelerated life fitters; this is a "
            f"{mle.kind} model. Use fit for its maximum-likelihood fit."
        )
    aliased = np.atleast_1d(getattr(mle, "aliased", ())).astype(int)
    if aliased.size:
        names = list(mle.parameter_names)
        which = ", ".join(names[mle.k_dist + j] for j in aliased.tolist())
        raise ValueError(
            f"The data do not determine {which} (the coefficient of a "
            "column that is constant or a combination of the others). "
            "Leave the column out."
        )
    spec = PriorSpec.parse(priors)
    for j in spec.joint:
        if not isinstance(j, MultivariateNormal):
            raise ValueError(
                "A regression's priors are given on its parameters by name "
                "(priors={'coef_0': ..., 'beta': ...}) or jointly as a "
                "MultivariateNormal; priors on quantities, from data and "
                "from a model are for the distributions' fit_bayes."
            )
    names = list(mle.parameter_names)
    fixed = {k: float(v) for k, v in dict(mle.fixed).items()}
    defaults = _default_priors(mle, spec, fixed)
    full = PriorSpec.parse({**spec.single, **defaults})
    full.joint = list(spec.joint)
    bound = BoundPriors(
        full,
        mle.dist,
        names,
        list(mle.parameter_bounds),
        False,
        False,
        False,
        fixed,
    )
    said = {
        k: p.describe(bound.transforms[names.index(k)])
        for k, p in defaults.items()
    }
    return bound, said


def _check_flat(mle: Any, bound: BoundPriors) -> None:
    """A flat prior is allowed only where the data determine the
    parameter: refused with no failure information, and, where every
    free parameter is flat, unless the maximum-likelihood fit's maximum
    is verified."""
    improper = bound.improper_names
    if not improper:
        return
    if not np.any(np.asarray(mle.data.c) != 1):
        raise ValueError(
            f"With no failures, the data do not determine "
            f"{', '.join(improper)}; give each a prior (for example "
            "prior.around(value, between=(low, high)))."
        )
    if set(improper) == set(bound.free_names) and mle.maximum != "verified":
        raise ValueError(
            "Every parameter has no prior (flat), and the data alone do "
            "not determine them: the maximum-likelihood fit's maximum is "
            f"{mle.maximum!r}. Give a prior for {', '.join(improper)}."
        )


def from_mle(mle: Any, priors: Any, sampler: dict) -> BayesianRegression:
    """The posterior of the regression ``mle`` was fitted for, with
    ``priors``, by the sampler."""
    from surpyval.bayes import convergence
    from surpyval.bayes.engines import nuts
    from surpyval.bayes.fitting import (
        peak,
        scattered,
        warn_not_converged,
    )
    from surpyval.utils.rng import as_generator

    started = time.perf_counter()
    chains, n_draws, warmup = nuts.check_settings(
        sampler["chains"], sampler["draws"], sampler["warmup"]
    )
    bound, defaults = _bind(mle, priors)
    if bound.d == 0:
        raise ValueError(
            "Every parameter is fixed, so there is nothing to fit."
        )
    _check_flat(mle, bound)
    f = RegressionLogPosterior(mle.model, mle.data, bound)
    theta = np.asarray(mle.params, dtype=float)
    with np.errstate(all="ignore"):
        start = bound.to_phi(theta[None, :])[0]
    starts = [s for s in (start, bound.centre()) if s is not None]
    starts = [s for s in starts if np.isfinite(f(s)[0])]
    if not starts:
        raise ValueError(
            "No starting point: the priors give no density to the "
            "maximum-likelihood fit's parameters "
            f"({', '.join(bound.free_names)}). Check the priors' ranges."
        )
    mode, spread = peak(f, starts)
    g = nuts.Affine(f, mode, nuts.curvature_factor(f, mode, spread))
    rng, chain_seed = as_generator(sampler["random_state"]).spawn(2)
    u0 = scattered(g, np.zeros(bound.d), np.ones(bound.d), chains, rng)
    try:
        out = nuts.sample(
            g,
            u0,
            draws=n_draws,
            warmup=warmup,
            random_state=chain_seed,
            n_jobs=sampler["n_jobs"],
        )
    except nuts.SamplerError as e:
        raise ValueError(str(e)) from None
    z = g.back(out["draws"])
    natural = bound.full_theta(z.reshape(-1, bound.d))
    summary = convergence.summarise(
        natural[:, bound.free_idx].reshape(z.shape),
        bound.free_names,
        out["divergent"],
        out["depth"],
        nuts.MAX_TREE_DEPTH,
    )
    problems = convergence.check(summary, chains, n_draws)
    post = Posterior(
        bound.names,
        bound.free_names,
        natural,
        np.full(len(natural), 1.0 / len(natural)),
        "draws",
        transforms=[bound.transforms[i] for i in bound.free_idx],
    )
    diag = {
        "method": "nuts",
        "free_parameters": list(bound.free_names),
        "chains": chains,
        "draws_per_chain": n_draws,
        "draws": chains * n_draws,
        "warmup": warmup,
        **summary,
        "warmup_divergences": int(out["warmup_divergences"].sum()),
        "step_size": [float(v) for v in out["step_size"]],
        "accept": float(out["accept"].mean()),
        "leapfrog_steps": int(out["steps"].sum()),
        "problems": list(problems),
        "ok": not problems,
    }
    model = BayesianRegression(mle, post, bound.spec, defaults, diag)
    model.diagnostics["seconds"] = time.perf_counter() - started
    warn_not_converged(problems, bound.free_names)
    return model


def _levels(alpha_ci: float, bound: str) -> list[float]:
    if not 0 < alpha_ci < 1:
        raise ValueError(
            f"alpha_ci must be between 0 and 1 (0.05 for a 95% interval); "
            f"got {alpha_ci!r}."
        )
    if bound == "two-sided":
        return [alpha_ci / 2, 1 - alpha_ci / 2]
    if bound == "lower":
        return [alpha_ci]
    if bound == "upper":
        return [1 - alpha_ci]
    raise ValueError(
        f"bound must be 'two-sided', 'lower' or 'upper'; got {bound!r}."
    )


class BayesianRegression(SerialisableMixin):
    """
    A Bayesian fit of a parametric regression (proportional hazards,
    accelerated failure time or accelerated life): the posterior of the
    baseline's parameters and the coefficients, and the posterior
    predictive distribution of a new unit's life at given covariates.

    Built by a regression fitter's ``fit_bayes`` (or
    ``fit_bayes_from_df``).

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians of every parameter, named by
        ``parameter_names`` (the baseline's, then the coefficients, as
        the maximum-likelihood fit names them).
    posterior : Posterior
        The posterior's draws: ``draws``, ``to_dataframe``, ``quantile``.
    priors : PriorSpec
        The priors used, the defaults among them.
    default_priors : dict
        The priors SurPyval chose (none was given), by parameter name,
        in words.
    mle : ParametricRegressionModel
        The maximum-likelihood fit to the same data.
    diagnostics : dict
        The sampler's settings and convergence checks (``rhat``,
        ``ess_bulk``, ``ess_tail`` by parameter, ``divergences``,
        ``tree_depth_saturated``, ``problems``).
    kind : str
        The family, as the maximum-likelihood model names it.

    Examples
    --------
    A proportional hazards fit with a prior belief that the covariate
    raises the hazard (a log hazard ratio near 0.5):

    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(1)
    >>> Z = rng.normal(size=(40, 1))
    >>> x = sp.Weibull.random(40, 50.0, 2.0, random_state=2)
    >>> x = x * np.exp(-0.25 * Z[:, 0])
    >>> model = sp.WeibullPH.fit_bayes(
    ...     x, Z, priors={"coef_0": sp.Normal.from_params([0.5, 0.2])},
    ...     chains=2, draws=300, warmup=300, random_state=1)
    >>> model.parameter_names
    ['alpha', 'beta', 'coef_0']
    >>> bool(0.2 < model.params[2] < 0.8)
    True
    >>> lo, hi = model.param_cri("coef_0")
    >>> bool(lo < model.params[2] < hi)
    True
    >>> bool(0 < model.sf(40.0, [[1.0]]) < model.sf(40.0, [[0.0]]) < 1)
    True
    """

    model_tag = "BayesianRegression"
    method = "nuts"

    def __init__(
        self,
        mle: Any,
        posterior: Posterior,
        priors: PriorSpec,
        default_priors: dict[str, str],
        diagnostics: dict,
        data: SurpyvalData | None = None,
    ) -> None:
        self.mle = mle
        self.posterior = posterior
        self.priors = priors
        self.default_priors = dict(default_priors)
        self.diagnostics = dict(diagnostics)
        self.model = mle.model
        self.dist = mle.dist
        self.kind = mle.kind
        # (a loaded maximum-likelihood fit holds no data: it comes apart)
        self.data = mle.data if data is None else data
        self.fixed = dict(mle.fixed)
        self.k_dist = mle.k_dist
        self.k = mle.k
        self.unit = getattr(mle, "unit", None)
        self.params = np.asarray(posterior.median(), dtype=float)
        self._point = mle.with_params(self.params)
        self._subset: npt.NDArray | None = None

    # ------------------------------------------------------------------
    # Names and values
    # ------------------------------------------------------------------
    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``."""
        return list(self.posterior.parameter_names)

    @property
    def ok(self) -> bool:
        """Whether the sampler's draws passed their convergence checks."""
        return bool(self.diagnostics.get("ok", True))

    def raise_if_failed(self) -> None:
        """Raise ``ValueError`` where :attr:`ok` is False."""
        if not self.ok:
            raise ValueError(
                "The Bayesian fit did not pass its checks: "
                + "; ".join(self.diagnostics.get("problems", []))
            )

    def at_params(self) -> Any:
        """The plug-in regression model at the posterior medians (a
        ``ParametricRegressionModel`` built from parameters)."""
        return self._point

    def draw_models(self, k: int = 100, random_state: Any = None) -> list:
        """``k`` ordinary regression models, one per posterior draw."""
        draws = self.posterior.draws(k, random_state=random_state)
        return [self.mle.with_params(d) for d in draws]

    # ------------------------------------------------------------------
    # The posterior predictive at covariates
    # ------------------------------------------------------------------
    def _draws(self) -> npt.NDArray:
        """The draws the predictive functions average over: all of them
        up to :data:`PREDICTIVE_DRAWS`, else that many, evenly spaced."""
        if self._subset is None:
            pts = self.posterior.points
            if len(pts) > PREDICTIVE_DRAWS:
                idx = np.linspace(0, len(pts) - 1, PREDICTIVE_DRAWS)
                pts = pts[np.round(idx).astype(int)]
            self._subset = pts
        return self._subset

    def _per_draw(self, fname: str) -> Any:
        """The family's function ``fname`` at every predictive draw, as a
        function of prepared ``(x, Z)``: one row per draw."""
        fn = getattr(self.model, fname)
        draws = self._draws()

        def values(x: Any, Z: Any, *_: Any) -> npt.NDArray:
            with np.errstate(all="ignore"):
                return np.stack(
                    [np.asarray(fn(x, Z, *d), dtype=float) for d in draws]
                )

        return values

    def _predict(self, fname: str, x: Any, Z: Any, grid: bool) -> Any:
        per_draw = self._per_draw("sf" if fname in ("Hf", "hf") else fname)
        below = 1.0 if fname == "sf" else 0.0

        def mean(x: Any, Z: Any, *p: Any) -> npt.NDArray:
            return per_draw(x, Z).mean(axis=0)

        if fname in ("sf", "ff", "df"):
            return self._point._eval(mean, x, Z, below, grid)
        sf = self._point._eval(mean, x, Z, 1.0, grid)
        if fname == "Hf":
            with np.errstate(divide="ignore"):
                return -np.log(sf)
        density = self._per_draw("df")

        def mean_df(x: Any, Z: Any, *p: Any) -> npt.NDArray:
            return density(x, Z).mean(axis=0)

        df = self._point._eval(mean_df, x, Z, 0.0, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            return df / sf

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """
        The posterior predictive survival function at covariates ``Z``:
        the probability that a new unit with those covariates survives
        past ``x``, the uncertainty in the parameters averaged in.

        Parameters
        ----------
        x : array like or scalar
            The times.
        Z : array like or DataFrame
            The covariates: one row per time, or one row for every time,
            as for the maximum-likelihood model's ``sf``.
        grid : bool, optional
            ``True`` gives every time for every row,
            ``(len(Z),) + x.shape``.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> model.sf([20.0, 40.0], [[0.0]]).shape
        (2,)
        """
        return self._predict("sf", x, Z, grid)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The posterior predictive CDF at covariates ``Z`` (see
        :meth:`sf`).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> bool(abs(model.ff(40.0, [[0.0]])
        ...          + model.sf(40.0, [[0.0]]) - 1) < 1e-12)
        True
        """
        return self._predict("ff", x, Z, grid)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The posterior predictive density at covariates ``Z`` (see
        :meth:`sf`).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> bool(model.df(40.0, [[0.0]]) > 0)
        True
        """
        return self._predict("df", x, Z, grid)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The hazard of the posterior predictive distribution at
        covariates ``Z``, ``df / sf`` (not the posterior mean of the
        hazard: the units that survive are those with low hazards).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> bool(model.hf(40.0, [[0.0]]) > 0)
        True
        """
        return self._predict("hf", x, Z, grid)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The cumulative hazard of the posterior predictive distribution
        at covariates ``Z``, ``-log sf``.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> bool(model.Hf(40.0, [[0.0]]) > 0)
        True
        """
        return self._predict("Hf", x, Z, grid)

    def _draw_quantiles(self, p: npt.NDArray, Z: Any) -> npt.NDArray:
        """Each predictive draw's own quantile function at ``(p, Z)``:
        one row per draw."""
        with ignoring_warnings(), np.errstate(all="ignore"):
            return np.stack(
                [
                    np.atleast_1d(
                        np.asarray(
                            self.mle.with_params(d).qf(p, Z), dtype=float
                        )
                    )
                    for d in self._draws()
                ]
            )

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> Any:
        """
        The quantile function of the posterior predictive distribution
        at covariates ``Z``: the time by which a new unit with those
        covariates has failed with probability ``p``.

        It inverts :meth:`ff` numerically, between the smallest and the
        largest of the draws' own quantiles.

        Parameters
        ----------
        p : array like or scalar
            Probabilities; outside [0, 1] gives ``nan``.
        Z : array like or DataFrame
            One row per probability, or one row for all of them.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> b10 = model.qf(0.1, [[0.0]])
        >>> bool(abs(model.ff(b10, [[0.0]]) - 0.1) < 1e-8)
        True
        """
        from scipy.optimize import brentq

        from surpyval.utils.validation import warn_outside_unit_interval

        p = np.atleast_1d(np.asarray(p, dtype=float))
        warn_outside_unit_interval(p)
        per_draw = self._draw_quantiles(p, Z)
        rows = covariate_rows(self._point._prepare_Z(Z), self._n_covariates())
        rows = np.broadcast_to(rows, (len(p), rows.shape[1]))
        out = np.full(len(p), np.nan)
        for i, pi in enumerate(p):
            q = per_draw[:, i]
            if not np.isfinite(pi) or not 0 <= pi <= 1:
                continue
            if pi in (0.0, 1.0) or not np.all(np.isfinite(q)):
                out[i] = np.nanmin(q) if pi == 0 else np.nanmax(q)
                continue
            lo, hi = float(q.min()), float(q.max())
            if lo == hi:
                out[i] = lo
                continue
            z = rows[i : i + 1]

            def gap(t: float, pi: float = pi, z: Any = z) -> float:
                return float(self.ff(t, z)) - pi

            out[i] = brentq(gap, lo, hi, xtol=1e-300, rtol=1e-13)
        return out

    def random(
        self, size: int, Z: Any, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Lives drawn from the posterior predictive distribution at
        covariates ``Z``: for each, a posterior draw (of those
        :meth:`sf` averages over) at random, and a life from it, so the
        draws follow :meth:`sf`.

        Parameters
        ----------
        size : int
            The number of lives at each covariate row.
        Z : array like or DataFrame
            The covariate rows.
        random_state : None, int or numpy.random.Generator, optional
            The seed, by the one seed rule (``None``: numpy's global
            stream).

        Returns
        -------
        x : numpy.ndarray
            The ``size`` lives of each row, row after row.
        Z : numpy.ndarray
            The covariate row of each life.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> lives, rows = model.random(5, [[0.0], [1.0]], random_state=3)
        >>> lives.shape, rows.shape
        ((10,), (10, 1))
        """
        from surpyval.utils.rng import as_generator

        size = int(size)
        rows = covariate_rows(self._point._prepare_Z(Z), self._n_covariates())
        rows = np.repeat(rows, size, axis=0)
        rng = as_generator(random_state)
        draws = self._draws()
        which = rng.integers(len(draws), size=len(rows))
        u = rng.random(len(rows))
        out = np.empty(len(rows))
        with ignoring_warnings(), np.errstate(all="ignore"):
            for k in np.unique(which):
                at = which == k
                model = self.mle.with_params(draws[k])
                out[at] = np.asarray(model.qf(u[at], rows[at]), dtype=float)
        return out, rows

    def _n_covariates(self) -> int:
        return int(np.asarray(self.data.Z).shape[1])

    # ------------------------------------------------------------------
    # Credible intervals
    # ------------------------------------------------------------------
    def cri(
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for a function of the regression at
        covariates ``Z``: the posterior's ``alpha_ci / 2`` and
        ``1 - alpha_ci / 2`` points of the function's value over the
        parameters' draws.

        Parameters
        ----------
        x : array like or scalar
            The times.
        Z : array like or DataFrame
            The covariates: one row per time, or one row for all.
        on : {'sf', 'ff', 'Hf', 'hf', 'df'}, optional
            The function (``'R'`` and ``'F'`` are ``'sf'`` and ``'ff'``).
            Default ``'sf'``.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> lo, hi = model.cri(40.0, [[0.0]])
        >>> bool(lo < model.sf(40.0, [[0.0]]) < hi)
        True
        """
        # ``'R'`` and ``'F'`` are ``'sf'`` and ``'ff'``, as for every model
        on = {"R": "sf", "F": "ff"}.get(on, on)
        if on not in _ON:
            raise ValueError(
                f"on must be one of {', '.join(_ON)}; got {on!r}. "
                "(quantile_cri gives the interval of a quantile, param_cri "
                "of a parameter.)"
            )
        levels = _levels(alpha_ci, bound)
        shape = np.shape(x)
        # ff and Hf are falling functions of sf: their interval is sf's,
        # transformed, so the three agree exactly (as the predictive's do)
        # rather than by the interpolation between draws.
        via_sf = on in ("ff", "Hf")
        per_draw = self._per_draw("sf" if via_sf else on)
        below = 1.0 if on == "sf" or via_sf else 0.0
        out = []
        for level in levels:

            # (sf's point at 1 - level, where the function falls in sf)
            at = 1.0 - level if via_sf else level

            def at_level(
                x: Any, Z: Any, *p: Any, at: float = at
            ) -> npt.NDArray:
                return np.quantile(per_draw(x, Z), at, axis=0)

            flat = np.reshape(np.asarray(x, dtype=float), -1)
            if via_sf:
                sf = self._point._eval(at_level, flat, Z, 1.0)
                with np.errstate(divide="ignore"):
                    value = 1.0 - sf if on == "ff" else -np.log(sf)
            else:
                value = self._point._eval(at_level, flat, Z, below)
            out.append(np.reshape(value, shape))
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def quantile_cri(
        self,
        p: npt.ArrayLike,
        Z: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for the ``p`` quantile of life at covariates
        ``Z`` (a B10 life at a use condition): the posterior's points of
        each draw's own quantile.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> lo, hi = model.quantile_cri(0.1, [[0.0]])
        >>> bool(lo < hi)
        True
        """
        from surpyval.utils.validation import warn_outside_unit_interval

        levels = _levels(alpha_ci, bound)
        shape = np.shape(p)
        flat = np.reshape(np.asarray(p, dtype=float), -1)
        warn_outside_unit_interval(flat, "quantile_cri")
        per_draw = self._draw_quantiles(flat, Z)
        out = [
            np.reshape(np.quantile(per_draw, level, axis=0), shape)
            for level in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def param_cri(
        self, name: str, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        """
        A credible interval for a parameter (a coefficient by its name,
        ``coef_0`` or its column's).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> model.param_cri("coef_0").shape
        (2,)
        """
        levels = _levels(alpha_ci, bound)
        out = np.asarray(self.posterior.quantile(name, levels), dtype=float)
        return out if len(levels) == 2 else out[0]

    def _refuse_cb(self, what: str, instead: str) -> Any:
        raise ValueError(
            f"A Bayesian fit has credible intervals, not confidence "
            f"bounds: use {instead} in place of {what}."
        )

    def cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: a Bayesian fit's interval is :meth:`cri`."""
        return self._refuse_cb("cb", "cri(x, Z, on=...)")

    def quantile_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`quantile_cri`."""
        return self._refuse_cb("quantile_cb", "quantile_cri(p, Z)")

    def param_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`param_cri`."""
        return self._refuse_cb("param_cb", "param_cri(name)")

    def diagnose(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses, saying where each check is: the sampler's convergence
        in ``diagnostics`` (and the warning the fit gives), and the
        family's residual checks on the maximum-likelihood fit,
        ``model.mle.diagnose()``."""
        raise ValueError(
            "diagnose() has no report for a Bayesian regression yet. The "
            "sampler's convergence checks are in model.diagnostics (rhat, "
            "ess_bulk, ess_tail, divergences; model.ok), and the "
            "family's checks of the data against the model are those of "
            "the maximum-likelihood fit: model.mle.diagnose()."
        )

    # ------------------------------------------------------------------
    # Comparison and summary
    # ------------------------------------------------------------------
    def _bound(self) -> BoundPriors:
        names = self.parameter_names
        return BoundPriors(
            self.priors,
            self.dist,
            names,
            list(self.mle.parameter_bounds),
            False,
            False,
            False,
            {k: float(v) for k, v in self.fixed.items()},
        )

    def compare(self, alpha_ci: float = 0.05) -> Any:
        """
        The prior, the maximum-likelihood estimate (with its Wald
        confidence interval) and the posterior of each free parameter,
        side by side.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> sorted(model.compare().table)
        ['alpha', 'beta', 'coef_0']
        """
        from surpyval.bayes.model import Comparison, _prior_interval

        _levels(alpha_ci, "two-sided")
        bound = self._bound()
        names = self.parameter_names
        estimates = np.asarray(self.mle.params, dtype=float)
        table = {}
        for name in bound.free_names:
            i = names.index(name)
            t = bound.transforms[i]
            described = bound.descriptions.get(name)
            interval = None
            if name in self.priors.single:
                interval = _prior_interval(bound, name, t, alpha_ci)
            mle_interval = None
            try:
                with ignoring_warnings(), np.errstate(all="ignore"):
                    ci = np.asarray(
                        self.mle.param_cb(name, alpha_ci=alpha_ci),
                        dtype=float,
                    ).reshape(-1)
                if ci.size == 2 and np.all(np.isfinite(ci)):
                    mle_interval = (float(ci[0]), float(ci[1]))
            except (ValueError, TypeError, KeyError):
                mle_interval = None
            table[name] = {
                "prior": described
                or f"no prior given; flat on {t.scale_name}",
                "prior_interval": interval,
                "mle": float(estimates[i]),
                "mle_interval": mle_interval,
                "posterior_median": float(self.params[i]),
                "posterior_interval": tuple(
                    float(v) for v in self.param_cri(name, alpha_ci)
                ),
            }
        return Comparison(table, alpha_ci)

    def summary(self, alpha_ci: float = 0.05) -> str:
        """The fit in words: the data, the priors (the defaults marked),
        the posterior of each parameter and how it was computed.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> "coef_0: default: Normal(0," in model.summary()
        True
        """
        from surpyval.bayes.model import _fmt, _how_computed

        lvl = f"{1 - alpha_ci:.0%}"
        n = np.asarray(self.data.n)
        c = np.asarray(self.data.c)
        total, failed = int(n.sum()), int(n[c == 0].sum())
        lines = [
            f"{self.dist.name} {self.kind}: Bayesian fit, {total} items "
            f"({failed} failed), {self._n_covariates()} "
            f"covariate{'s' if self._n_covariates() != 1 else ''}"
        ]
        bound = self._bound()
        names = self.parameter_names
        lines.append("Priors:")
        for name in bound.free_names:
            t = bound.transforms[names.index(name)]
            said = bound.descriptions.get(
                name, f"no prior given; flat on {t.scale_name}"
            )
            lines.append(f"  {name}: {said}")
        for name, v in self.fixed.items():
            lines.append(f"  {name}: fixed at {_fmt(float(v))}")
        width = max(len(s) for s in names)
        lines.append(
            f"Posterior{' ' * max(width - 7, 1)}  median   {lvl} credible "
            "interval      max. lik."
        )
        estimates = np.asarray(self.mle.params, dtype=float)
        for name in bound.free_names:
            i = names.index(name)
            lo, hi = self.param_cri(name, alpha_ci)
            lines.append(
                f"  {name:<{width}}  {_fmt(float(self.params[i])):>8}   "
                f"{_fmt(float(lo))} - {_fmt(float(hi))}"
                f"      {_fmt(float(estimates[i]))}"
            )
        lines.append(_how_computed(self))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_subset"] = None
        return state

    def to_dict(self) -> dict:
        """
        A strict-JSON dictionary of the model: the maximum-likelihood fit
        (which holds the family and the covariates' names), the data, the
        priors, the posterior's draws and the diagnostics.
        ``surpyval.from_dict`` reads it.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z, chains=2, draws=200, warmup=200, random_state=1)
        >>> again = sp.from_dict(model.to_dict())
        >>> bool(again.sf(40.0, [[0.0]]) == model.sf(40.0, [[0.0]]))
        True
        """
        import surpyval
        from surpyval.bayes.model import _jsonable

        diag = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in self.diagnostics.items()
        }
        out = {
            "model": self.model_tag,
            "kind": self.kind,
            "mle": self.mle.to_dict(),
            "priors": self.priors.to_dict(),
            "default_priors": dict(self.default_priors),
            "posterior": self.posterior.to_dict(),
            "data": self.data.to_dict(),
            "diagnostics": _jsonable(diag),
            "surpyval_version": surpyval.__version__,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianRegression:
        """Rebuild a model from :meth:`to_dict`."""
        import surpyval

        require_model_tag(model_dict, cls.model_tag, "a Bayesian regression")
        with ignoring_warnings():
            mle = surpyval.from_dict(model_dict["mle"])
        model = cls(
            mle,
            Posterior.from_dict(model_dict["posterior"]),
            PriorSpec.from_dict(model_dict["priors"]),
            model_dict.get("default_priors", {}),
            model_dict.get("diagnostics", {}),
            data=SurpyvalData.from_dict(model_dict["data"]),
        )
        model.unit = model_dict.get("unit", model.unit)
        return model
