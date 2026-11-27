"""The shared Bayesian fit: one engine rule and one result for every
family that gives a :class:`~surpyval.bayes.target.Target`.

``fit_target`` binds the priors to the target's parameters, chooses the
engine and returns a :class:`BayesianResult` (or the family's subclass of
it). The rule, ``method="auto"``:

1. **closed form** where the priors and the data are conjugate (the
   target says so: a gamma prior on a constant rate with Poisson counts);
2. **the grid** for one to three free parameters: the exact posterior,
   to a stated integration error;
3. **the Laplace approximation** beyond: a normal at the posterior's peak
   on the parameters' unconstrained scale, checked against the exact
   posterior (Pareto k and the shift of the interval's ends), with
   optional importance reweighting.

There is no sampler. An exact posterior of more than three free
parameters goes through PyMC (:mod:`surpyval.pymc`).

Every result says how it was computed (the engine's label and, for
Laplace, its checks), which priors SurPyval chose (``default_priors``)
and where the data disagree with a prior (``diagnostics
["prior_conflict"]``).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.engines.grid import lattice_quantile, weighted_quantile
from surpyval.bayes.posterior import Posterior
from surpyval.bayes.prior import (
    BoundPriors,
    FitPowerPrior,
    Fixed,
    MultivariateNormal,
    NormalOnScale,
    PriorSpec,
)
from surpyval.bayes.transforms import Transform
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.warnings import caller_stacklevel, ignoring_warnings

METHODS = ("auto", "grid", "conjugate", "laplace")
#: The grid takes this many free parameters at most.
GRID_MAX = 3
#: A prior's central interval the maximum-likelihood estimate must fall
#: outside of for the two to be said to disagree.
CONFLICT_LEVEL = 0.99
#: The replicates of a posterior predictive check.
REPLICATES = 200
#: A posterior predictive p-value outside ``(P_EXTREME, 1 - P_EXTREME)``
#: is flagged.
P_EXTREME = 0.025


def interval_levels(alpha_ci: float, bound: str) -> list[float]:
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


def bind(target: Any, priors: Any) -> tuple[BoundPriors, dict[str, str]]:
    """The priors (given and the family's defaults) bound to the
    target's parameters, and the defaults' descriptions by name."""
    spec = PriorSpec.parse(priors)
    names = list(target.names)
    fixed = {k: float(v) for k, v in dict(target.fixed).items()}
    defaults = target.default_priors(spec, fixed)
    full = PriorSpec.parse({**spec.single, **defaults})
    full.joint = list(spec.joint)
    bound = BoundPriors(
        full,
        target.dist,
        names,
        list(target.bounds),
        False,
        False,
        False,
        fixed,
        target=target,
    )
    said = {
        k: p.describe(bound.transforms[names.index(k)])
        for k, p in defaults.items()
    }
    return bound, said


def log_posterior(target: Any, bound: BoundPriors) -> Any:
    """The log posterior at rows of the free parameters' unconstrained
    values: the priors (with the change of scale) plus the family's own
    log-likelihood."""

    def log_post(phi: npt.NDArray) -> npt.NDArray:
        phi = np.atleast_2d(phi)
        out = np.asarray(bound.log_prior(phi), dtype=float).copy()
        finite = np.isfinite(out)
        if finite.any():
            theta = bound.full_theta(phi[finite])
            with np.errstate(all="ignore"):
                out[finite] = out[finite] + target.log_lik(theta)
        return np.where(np.isnan(out), -np.inf, out)

    return log_post


def _grid_summary(target: Any, bound: BoundPriors) -> Any:
    """The numbers the grid's self-check holds steady: the 2.5%, 50% and
    97.5% points of each free parameter and of the family's own check
    quantities."""

    # which of the family's quantities are checked is settled on the
    # first lattice (those finite all over it), so that every lattice's
    # summary has the same columns
    use: list[int] | None = None

    def summary(phi: npt.NDArray, w: npt.NDArray, shape: tuple) -> npt.NDArray:
        nonlocal use
        theta = bound.full_theta(phi)
        cols = [theta[:, i] for i in bound.free_idx]
        with np.errstate(all="ignore"):
            extras = [
                np.asarray(e, dtype=float)
                for e in target.grid_checks(bound.theta_dict(phi))
            ]
        if use is None:
            use = [k for k, e in enumerate(extras) if np.all(np.isfinite(e))]
        cols += [extras[k] for k in use]
        return lattice_quantile(
            np.stack(cols, axis=1), w, [0.025, 0.5, 0.975], shape
        ).T

    return summary


def fit_target(
    target: Any,
    priors: Any = None,
    *,
    method: str = "auto",
    importance: bool = False,
    points: int | None = None,
    result: Any = None,
) -> Any:
    """
    The posterior of ``target``'s parameters under ``priors``.

    Parameters
    ----------
    target : Target
        The family's side of the fit.
    priors
        As for every ``fit_bayes``: a dict by parameter name, a joint
        prior, or a list of these.
    method : {'auto', 'conjugate', 'grid', 'laplace'}, optional
        The engine; ``'auto'`` (the default) is the rule in this
        module's description.
    importance : bool, optional
        Reweight the Laplace engine's draws towards the exact posterior.
    points : int, optional
        The grid's starting points per parameter.
    result : class, optional
        The result's class (a subclass of :class:`BayesianResult`).

    Returns
    -------
    BayesianResult

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.bayes.regression import RegressionTarget
    >>> Z = np.random.default_rng(1).normal(size=(30, 1))
    >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
    >>> target = RegressionTarget(sp.WeibullPH.fit(x, Z))
    >>> model = sp.bayes.fit_target(target)
    >>> type(model).__name__, model.method
    ('BayesianResult', 'laplace')
    """
    from surpyval.bayes.engines.grid import GridError, find_mode, run_grid
    from surpyval.bayes.engines.laplace import run_laplace
    from surpyval.bayes.fitting import _check_points
    from surpyval.bayes.model import laplace_diagnostics, warn_laplace

    started = time.perf_counter()
    cls = BayesianResult if result is None else result
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {', '.join(METHODS)}; got {method!r}."
        )
    points = _check_points(points)
    bound, defaults = bind(target, priors)
    if bound.d == 0:
        raise ValueError(
            "Every parameter is fixed, so there is nothing to fit."
        )
    target.check(bound)
    if method == "auto":
        method = target.default_method
    case = None
    if method in ("auto", "conjugate"):
        case = target.conjugate(bound)
    if method == "conjugate" and case is None:
        raise ValueError(
            f"No closed form applies to this {target.family} fit with "
            "these priors and data. Use method='auto'."
        )
    if case is None:
        if method == "grid" and bound.d > GRID_MAX:
            raise ValueError(
                f"This fit has {bound.d} free parameters "
                f"({', '.join(bound.free_names)}); the grid takes 1 to "
                f"{GRID_MAX}. Use method='laplace' (a normal approximation, "
                "any number), or fix some with fixed= or prior.fixed()."
            )
        if method in ("auto", "conjugate"):
            method = "grid" if bound.d <= GRID_MAX else "laplace"
    else:
        method = "conjugate"
    if importance and method != "laplace":
        raise ValueError(
            "importance=True reweights the Laplace approximation's draws; "
            f"this fit is computed by the {method} engine, which needs "
            "none. Leave importance out, or use method='laplace'."
        )
    if case is not None:
        model = cls._from_target(
            target,
            _conjugate_posterior(bound, case),
            bound.spec,
            defaults,
            {
                "method": "conjugate",
                "free_parameters": list(bound.free_names),
                "case": case.label,
                "log_evidence": float(case.log_evidence),
                "ok": True,
                "approximate": False,
            },
        )
        return _finish(model, bound, started)

    target.check_flat(bound)
    log_post = log_posterior(target, bound)
    starts = []
    theta = target.start()
    if theta is not None:
        with np.errstate(all="ignore"):
            starts.append(bound.to_phi(np.asarray(theta, float)[None, :])[0])
    starts.append(bound.centre())
    if theta is not None:
        # the maximum-likelihood start with every parameter that has a
        # prior moved to its prior's centre: a start where a strong
        # prior far from the estimate still gives a density
        centres = bound.centres()
        if np.isfinite(centres).any():
            starts.append(np.where(np.isnan(centres), starts[0], centres))
    starts = [s for s in starts if s is not None]
    starts = [
        s
        for s in starts
        if np.all(np.isfinite(s)) and np.isfinite(log_post(s)[0])
    ]
    if not starts:
        raise ValueError(
            "No starting point: the priors give no density to the "
            "maximum-likelihood fit's parameters "
            f"({', '.join(bound.free_names)}). Check the priors' ranges."
        )
    free_t = [bound.transforms[i] for i in bound.free_idx]
    # The engines work in the target's working coordinates u where it
    # has them (phi = inverse(u), with the change of variables in the
    # density); the points come back to phi before anything is read.
    rp = target.reparam(bound)
    plain_summary = _grid_summary(target, bound)
    work: Any = log_post
    summary: Any = plain_summary
    where: Any = bound
    if rp is not None:

        def in_u(u: npt.NDArray) -> npt.NDArray:
            u = np.atleast_2d(u)
            with np.errstate(all="ignore"):
                out = log_post(rp.inverse(u)) + rp.log_det(u)
            return np.where(np.isnan(out), -np.inf, out)

        def summary_in_u(
            u: npt.NDArray, w: npt.NDArray, shape: tuple
        ) -> npt.NDArray:
            out: npt.NDArray = plain_summary(rp.inverse(u), w, shape)
            return out

        work, summary = in_u, summary_in_u
        with np.errstate(all="ignore"):
            starts = [rp.forward(s[None, :])[0] for s in starts]
        starts = [s for s in starts if np.all(np.isfinite(s))]
        where = _WorkingNames(bound, rp)
    scales, phi_scales = _search_setup(
        target, bound, log_post, work, starts[0], rp, method
    )
    try:
        mode, cov, _ = find_mode(work, starts, scales)
        if method == "laplace":
            approx = run_laplace(
                work,
                mode,
                cov,
                importance=importance,
                draws=target.laplace_draws,
            )
        else:
            grid = run_grid(
                work,
                mode,
                cov,
                summary,
                points=points,
                flat_names=bound.improper_names,
            )
    except GridError as e:
        raise ValueError(str(e)) from None

    def to_phi(u: npt.NDArray) -> npt.NDArray:
        out: npt.NDArray = u if rp is None else rp.inverse(np.atleast_2d(u))
        return out

    if method == "laplace":
        post = Posterior(
            bound.names,
            bound.free_names,
            _finite_draws(bound.full_theta(to_phi(approx.phi)), approx.log_w),
            np.exp(approx.log_w),
            "laplace",
            transforms=free_t,
            # (in working coordinates the normal is not one on phi: the
            # parameters' quantiles are then the draws')
            normal=_phi_normal(approx, rp),
        )
        diag = dict(laplace_diagnostics(approx, where, importance))
        diag["free_parameters"] = list(bound.free_names)
        diag.update(_working_normal(approx, rp))
        diag["mode"] = _phi_mode(
            bound, log_post, to_phi(approx.mode), rp, phi_scales
        )
        if target.laplace_only:
            diag["regression"] = True
    else:
        post = Posterior(
            bound.names,
            bound.free_names,
            _finite_draws(bound.full_theta(to_phi(grid.phi)), grid.log_w),
            np.exp(grid.log_w),
            "grid",
            transforms=free_t,
            # (a cell is a box in the lattice's coordinates: kept only
            # where those are phi, for draws inside a cell)
            cell=grid.cell if rp is None else None,
            shape=grid.shape,
        )
        diag = {
            "method": "grid",
            "free_parameters": list(bound.free_names),
            "points_per_dim": grid.points_per_dim,
            "grid_points": int(len(grid.log_w)),
            "refinements": grid.refinements,
            "integration_error": grid.error,
            "half_width_sd": float(np.max(grid.half_widths)),
            "widened": grid.checks.get("widened", 0),
            "mode": _phi_mode(
                bound, log_post, to_phi(grid.mode), rp, phi_scales
            ),
            "log_evidence": (
                float(grid.log_z) if bound.proper else float("nan")
            ),
            "ok": True,
            "approximate": False,
        }
    model = cls._from_target(target, post, bound.spec, defaults, diag)
    _finish(model, bound, started)
    if method == "laplace":
        warn_laplace(model.diagnostics)
    return model


def _search_scales(
    target: Any, bound: BoundPriors, log_post: Any, start: npt.NDArray
) -> npt.NDArray:
    """The peak search's scale for each free coordinate: the target's
    (:meth:`~surpyval.bayes.target.Target.coordinate_scales`: the
    maximum-likelihood standard errors on the unconstrained scale, and
    a regression coefficient's one over its covariate's spread), and
    where it gives none, the posterior's curvature at the first start
    (:func:`~surpyval.bayes.engines.grid.start_scales`)."""
    from surpyval.bayes.engines.grid import start_scales

    given = target.coordinate_scales(bound)
    out = np.full(bound.d, np.nan) if given is None else np.array(given)
    out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
    if np.isnan(out).any():
        out = np.where(np.isnan(out), start_scales(log_post, start), out)
    return out


#: The smallest eigenvalue of the correlation matrix of the
#: maximum-likelihood estimates (unconstrained scale) below which a fit
#: with no centring is refused: the parameters are then collinear to
#: about one part in 1e5, which a covariate about 200 of its standard
#: deviations from 0 makes (1e-4 at 50, 3e-7 at 1000, in six seeded
#: Weibull PH designs), and the normal approximation is not to be had.
_COLLINEAR = 1e-5


def _refuse_collinear(target: Any, bound: BoundPriors) -> None:
    """Refuse a fit the engines would get wrong without a centring
    (:meth:`~surpyval.bayes.target.Target.reparam`): where the
    maximum-likelihood correlation matrix's smallest eigenvalue is below
    ``_COLLINEAR``."""
    worst = target.collinearity(bound)
    if worst is None or worst >= _COLLINEAR:
        return
    raise ValueError(
        "The parameters are nearly collinear (the smallest eigenvalue of "
        f"their correlation matrix at the maximum-likelihood fit is "
        f"{worst:.1e}, below {_COLLINEAR:g}), so no approximation of the "
        "posterior can be relied on. In a regression this is a covariate "
        "far from 0 (the baseline and its coefficient then move "
        "together): centre it (subtract its mean, or a value near it) "
        "and refit; the coefficient's posterior is the same, and the "
        "baseline is then the one at that value."
    )


def _finite_draws(theta: npt.NDArray, log_w: npt.NDArray) -> npt.NDArray:
    """The posterior's points on the parameters' own scale, refused where
    some with weight are not finite: a baseline at a covariate's origin
    far from the data (``e^{beta' zbar}`` past the largest float, at a
    covariate near 1000) cannot be held on the user's scale, however
    well the engines found it at the covariates' means."""
    bad = ~np.all(np.isfinite(theta), axis=1) & np.isfinite(log_w)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} of the posterior's {len(theta)} points have "
            "parameters too large or small for double precision on the "
            "scale they are reported on: the baseline is at covariates of "
            "0, far from the data's. Centre the covariates (subtract their "
            "means, or values near them) and refit; the coefficients' "
            "posterior is the same, and the baseline is then the one at "
            "those values."
        )
    return theta


def _search_setup(
    target: Any,
    bound: BoundPriors,
    log_post: Any,
    work: Any,
    start: npt.NDArray,
    rp: Any,
    method: str,
) -> tuple[npt.NDArray | None, npt.NDArray | None]:
    """The peak search's scales in the engines' coordinates
    (:func:`_engine_scales`) and, where the engines work in a
    reparameterisation that keeps some of phi's coordinates, phi's (for
    the peak on phi, :func:`_phi_mode`); and, with no
    reparameterisation, the Laplace engine's refusal of a fit too
    collinear for it (:func:`_refuse_collinear`)."""
    if rp is None and method == "laplace":
        _refuse_collinear(target, bound)
    scales = _engine_scales(target, bound, log_post, work, start, rp)
    phi_scales = (
        _search_scales(target, bound, log_post, rp.inverse(start[None])[0])
        if getattr(rp, "same", None) is not None
        else None
    )
    return scales, phi_scales


def _engine_scales(
    target: Any,
    bound: BoundPriors,
    log_post: Any,
    work: Any,
    start: npt.NDArray,
    rp: Any,
) -> npt.NDArray | None:
    """The peak search's scales in the engines' coordinates: phi's
    (:func:`_search_scales`) where there is no reparameterisation;
    where there is one that names the coordinates it keeps (``same``),
    the target's scale for those and the curvature of the posterior at
    the start for the rest; otherwise none (the target's own choice of
    scale)."""
    from surpyval.bayes.engines.grid import start_scales

    if rp is None:
        return _search_scales(target, bound, log_post, start)
    same = getattr(rp, "same", None)
    if same is None:
        return None
    given = target.coordinate_scales(bound)
    out = np.full(len(start), np.nan)
    if given is not None:
        given = np.asarray(given, dtype=float)
        for j in same:
            if np.isfinite(given[j]) and given[j] > 0:
                out[j] = given[j]
    if np.isnan(out).any():
        out = np.where(np.isnan(out), start_scales(work, start), out)
    return out


def _working_normal(approx: Any, rp: Any) -> dict:
    """Where the engines worked in a target's working coordinates, the
    normal they laid there (``working_normal``: its coordinates' names,
    mean and covariance), for checks of its curvature; nothing
    otherwise."""
    if rp is None:
        return {}
    return {
        "working_normal": {
            "names": list(rp.names),
            "mean": [float(v) for v in approx.mode],
            "cov": [[float(v) for v in row] for row in approx.cov],
        }
    }


def _phi_normal(approx: Any, rp: Any) -> dict | None:
    """The Laplace normal on phi, for the posterior's closed-form
    summaries: the engine's own where it worked in phi, carried through
    an affine change of coordinates (``rp.affine``: ``phi = A u + b``,
    under which a normal stays one; a centred accelerated failure time
    scale family's), and none otherwise (the summaries are then the
    draws')."""
    mean, cov = np.asarray(approx.mode), np.asarray(approx.cov)
    if rp is not None:
        if not getattr(rp, "affine", False):
            return None
        d = mean.size
        b = rp.inverse(np.zeros((1, d)))[0]
        A = np.column_stack(
            [rp.inverse(np.eye(d)[j][None, :])[0] - b for j in range(d)]
        )
        mean, cov = A @ mean + b, A @ cov @ A.T
    return {
        "mean": [float(v) for v in mean],
        "cov": [[float(v) for v in row] for row in cov],
        "reweighted": bool(approx.importance),
    }


def _phi_mode(
    bound: BoundPriors,
    log_post: Any,
    at: npt.NDArray,
    rp: Any,
    scale: npt.NDArray | None = None,
) -> list[float]:
    """The posterior's peak on the parameters' unconstrained scale, as
    every parameter's value: where the engine worked in other
    coordinates its peak there is not this one (the change of variables
    moves it), so it is searched from there."""
    from surpyval.bayes.engines.grid import GridError, find_mode

    at = np.atleast_2d(at)[0]
    if rp is not None:
        try:
            at, _, _ = find_mode(log_post, [at], scale)
        except GridError:
            pass
    return [float(v) for v in bound.full_theta(at)[0]]


class _WorkingNames:
    """The bound priors as the Laplace diagnostics read them, where the
    engine worked in a target's working coordinates."""

    def __init__(self, bound: BoundPriors, rp: Any) -> None:
        self.free_names = list(rp.names)
        self._bound, self._rp = bound, rp

    def full_theta(self, u: npt.NDArray) -> npt.NDArray:
        return self._bound.full_theta(self._rp.inverse(np.atleast_2d(u)))


def _conjugate_posterior(bound: BoundPriors, case: Any) -> Posterior:
    """A closed-form posterior of one parameter as equal-probability
    points (its quantiles and draws stay exact)."""
    from surpyval.bayes.engines.conjugate import NODES
    from surpyval.bayes.posterior import _CONJUGATE_FAMILIES

    frozen = _CONJUGATE_FAMILIES[case.family](*case.params)
    u = (np.arange(NODES) + 0.5) / NODES
    theta = np.repeat(
        bound.full_theta(np.zeros((1, bound.d)))[:1], NODES, axis=0
    )
    theta[:, bound.names.index(case.name)] = frozen.ppf(u)
    return Posterior(
        bound.names,
        bound.free_names,
        theta,
        np.full(NODES, 1.0 / NODES),
        "conjugate",
        transforms=[bound.transforms[i] for i in bound.free_idx],
        exact={
            "name": case.name,
            "family": case.family,
            "params": [float(v) for v in case.params],
        },
    )


def _finish(model: Any, bound: BoundPriors, started: float) -> Any:
    """Record the prior-against-data conflicts and the time taken, and
    warn of the conflicts."""
    notes = prior_conflicts(model, bound)
    model.diagnostics["prior_conflict"] = notes
    model.diagnostics["seconds"] = time.perf_counter() - started
    if notes:
        warnings.warn(
            "The data disagree with the prior: "
            + "; ".join(notes)
            + ". Neither is taken as right: check the prior and the data, "
            "and compare() the two.",
            stacklevel=caller_stacklevel(),
        )
    return model


def prior_conflicts(model: Any, bound: BoundPriors) -> list[str]:
    """Where the data disagree with a prior: each parameter given a
    proper prior (not a default) whose maximum-likelihood estimate, the
    data's own answer, lies outside the prior's central 99% interval."""
    from surpyval.bayes.model import _fmt, _prior_interval

    target = model.target
    if target.mle is None:
        return []
    try:
        estimates = np.asarray(target.estimates(), dtype=float)
    except (ValueError, AttributeError):
        return []
    out = []
    for name, p in model.priors.single.items():
        if p.is_fixed or not p.proper or name in model.default_priors:
            continue
        i = bound.names.index(name)
        t = bound.transforms[i]
        iv = _prior_interval(bound, name, t, 1 - CONFLICT_LEVEL)
        if iv is None or not np.isfinite(estimates[i]):
            continue
        if not iv[0] <= estimates[i] <= iv[1]:
            iv90 = _prior_interval(bound, name, t, 0.1)
            assert iv90 is not None
            out.append(
                f"{name} (prior 90% interval {_fmt(iv90[0])}-"
                f"{_fmt(iv90[1])}; the maximum-likelihood estimate is "
                f"{_fmt(float(estimates[i]))})"
            )
    return out


class PredictiveCheck:
    """
    A posterior predictive check: does data simulated from the fitted
    model look like the data observed?

    For each of ``replicates`` posterior draws a data set is simulated
    under the observed design (the same units, covariates and observation
    windows), and one statistic of it is compared with the same
    statistic of the data. Where the model describes the data, the
    observed value sits among the replicates'.

    Attributes
    ----------
    statistic : str
        The statistic compared.
    observed : float
        Its value in the data.
    replicated : numpy.ndarray
        Its value in each replicate.
    p_value : float
        The share of replicates at or above the observed value (the
        posterior predictive p-value): near 0 or 1, the model does not
        reproduce this feature of the data. It is not a frequentist
        p-value (it is conservative: the data were used twice).
    interval : tuple
        The replicates' central 90% range.
    ok : bool
        Whether ``p_value`` is inside ``(0.025, 0.975)``.

    Examples
    --------
    Twelve failures were seen, and eight simulated data sets had these
    many:

    >>> from surpyval.bayes import PredictiveCheck
    >>> check = PredictiveCheck(
    ...     "number of failures", 12, [9, 11, 12, 13, 15, 10, 14, 12])
    >>> check.p_value, check.ok
    (0.5, True)
    """

    def __init__(
        self, statistic: str, observed: float, replicated: npt.NDArray
    ) -> None:
        self.statistic = statistic
        self.observed = float(observed)
        rep = np.asarray(replicated, dtype=float)
        self.replicated = rep
        good = rep[np.isfinite(rep)]
        self.failed = int(len(rep) - len(good))
        if good.size:
            # ties count half, so a discrete statistic is not pushed to
            # an end
            above = np.sum(good > self.observed)
            same = np.sum(good == self.observed)
            self.p_value = float((above + 0.5 * same) / len(good))
            lo, hi = np.quantile(good, [0.05, 0.95])
            self.interval = (float(lo), float(hi))
        else:
            self.p_value = float("nan")
            self.interval = (float("nan"), float("nan"))
        self.ok = bool(P_EXTREME < self.p_value < 1 - P_EXTREME)

    def __repr__(self) -> str:
        from surpyval.bayes.model import _fmt

        lo, hi = self.interval
        verdict = (
            "the model reproduces it"
            if self.ok
            else "the model does not reproduce it"
        )
        lines = [
            f"Posterior predictive check on {self.statistic}",
            f"  observed            {_fmt(self.observed)}",
            f"  replicates (90%)    {_fmt(lo)} - {_fmt(hi)} "
            f"({len(self.replicated) - self.failed} data sets)",
            f"  share at or above   {self.p_value:.3f}: {verdict}",
        ]
        if self.failed:
            lines.append(
                f"  {self.failed} replicates gave no value and are left out"
            )
        return "\n".join(lines)


class BayesianDiagnosis:
    """The checks of a Bayesian fit, from ``diagnose()``: how the
    posterior was computed and whether that can be trusted, where the
    data disagree with the priors, and whether the fitted model
    reproduces the data (a posterior predictive check).

    Attributes
    ----------
    checks : list of tuple
        ``(name, ok, detail)`` for each check.
    predictive : PredictiveCheck
        The predictive check.
    ok : bool
        Whether every check passed.

    Examples
    --------
    >>> from surpyval.bayes import BayesianDiagnosis
    >>> report = BayesianDiagnosis(
    ...     [("posterior", True, "computed exactly in closed form"),
    ...      ("priors against the data", True, "no disagreement found")],
    ...     {})
    >>> report
    Bayesian fit: checks
      [ok] posterior: computed exactly in closed form
      [ok] priors against the data: no disagreement found
    >>> report.ok
    True
    """

    def __init__(self, checks: list[tuple], predictive: Any) -> None:
        self.checks = checks
        self.predictive = predictive
        self.ok = all(ok for _, ok, _ in checks)

    def __repr__(self) -> str:
        lines = ["Bayesian fit: checks"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'CHECK'}] {name}: {detail}")
        return "\n".join(lines)


class BayesianResult(SerialisableMixin):
    """
    A Bayesian fit of any family with a
    :class:`~surpyval.bayes.target.Target`: the posterior of its
    parameters, and everything derived from it.

    Built by a fitter's ``fit_bayes``; the families' own results
    (:class:`~surpyval.bayes.BayesianRegression`, the recurrent-event
    processes') add their predictions to it.

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians of every parameter, named by
        ``parameter_names``.
    posterior : Posterior
        The posterior's points: ``draws``, ``to_dataframe``,
        ``quantile``.
    priors : PriorSpec
        The priors used, the defaults among them.
    default_priors : dict
        The priors SurPyval chose (none was given), by parameter name,
        in words.
    mle : object
        The maximum-likelihood fit to the same data.
    method : str
        The engine: ``"conjugate"`` (closed form), ``"grid"`` (exact, to
        the stated integration error) or ``"laplace"`` (a checked normal
        approximation).
    diagnostics : dict
        How the posterior was computed; for Laplace its checks against
        the exact posterior (``pareto_k``, ``shift``,
        ``approximation``); ``prior_conflict``, the parameters whose
        prior the data disagree with.

    Examples
    --------
    A regression's fit is one; the credible interval of any function of
    the model (here the B10 life at ``Z = 0``) and the probability of a
    condition come from it:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> Z = np.random.default_rng(1).normal(size=(30, 1))
    >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
    >>> model = sp.WeibullPH.fit_bayes(x, Z)
    >>> isinstance(model, sp.bayes.BayesianResult)
    True
    >>> lo, hi = model.function_cri(lambda m: m.qf(0.1, [[0.0]]))
    >>> bool(lo < hi)
    True
    >>> p = model.probability(lambda m: m.qf(0.1, [[0.0]]) > 10.0)
    >>> bool(0 <= p <= 1)
    True
    """

    model_tag = "BayesianResult"

    def __init__(
        self,
        target: Any,
        posterior: Posterior,
        priors: PriorSpec,
        default_priors: dict[str, str],
        diagnostics: dict,
    ) -> None:
        self.target = target
        self.posterior = posterior
        self.priors = priors
        self.default_priors = dict(default_priors)
        self.diagnostics = dict(diagnostics)
        self.method = str(self.diagnostics.get("method", posterior.kind))
        self.mle = target.mle
        self.data = target.data
        self.dist = target.dist
        self.fixed = dict(target.fixed)
        self.params = np.asarray(posterior.median(), dtype=float)

    @classmethod
    def _from_target(
        cls,
        target: Any,
        posterior: Posterior,
        priors: PriorSpec,
        default_priors: dict[str, str],
        diagnostics: dict,
    ) -> Any:
        return cls(target, posterior, priors, default_priors, diagnostics)

    # ------------------------------------------------------------------
    # Names and values
    # ------------------------------------------------------------------
    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``."""
        return list(self.posterior.parameter_names)

    @property
    def ok(self) -> bool:
        """Whether the posterior was computed (a poor approximation is
        warned of and stated in ``diagnostics["approximation"]``)."""
        return bool(self.diagnostics.get("ok", True))

    @property
    def log_evidence(self) -> float:
        """The log of the evidence (the marginal likelihood) where the
        engine computes it and every prior is proper; else ``nan``."""
        return float(self.diagnostics.get("log_evidence", float("nan")))

    def raise_if_failed(self) -> None:
        """Raise ``ValueError`` where :attr:`ok` is False."""
        if not self.ok:
            raise ValueError(
                "The Bayesian fit did not pass its checks: "
                + "; ".join(self.diagnostics.get("problems", []))
            )

    def at_params(self) -> Any:
        """The family's ordinary model at the posterior medians."""
        return self.target.with_params(self.params)

    def draw_models(self, k: int = 100, random_state: Any = None) -> list:
        """``k`` ordinary models of the family, one per posterior draw."""
        draws = self.posterior.draws(k, random_state=random_state)
        return [self.target.with_params(d) for d in draws]

    # ------------------------------------------------------------------
    # Any function of the model
    # ------------------------------------------------------------------
    def _points(self) -> tuple[npt.NDArray, npt.NDArray]:
        """The posterior's points that carry its mass, and their weights
        (a grid's far corners, with no weight to speak of, are left
        out)."""
        post = self.posterior
        w = post.weights
        if post.kind != "grid" or len(w) <= 4096:
            return post.points, w
        order = np.argsort(w)
        drop = order[np.cumsum(w[order]) < 1e-10]
        keep = np.ones(len(w), dtype=bool)
        keep[drop] = False
        return post.points[keep], w[keep] / w[keep].sum()

    def function_draws(
        self, fn: Callable[[Any], Any]
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        A function of the model at every posterior point: ``(values,
        weights)``, for a histogram or a summary of one's own.

        Parameters
        ----------
        fn : callable
            ``fn(model)`` of the family's ordinary model, returning a
            number or an array.
        """
        points, w = self._points()
        values = []
        with ignoring_warnings(), np.errstate(all="ignore"):
            for p in points:
                values.append(
                    np.asarray(fn(self.target.with_params(p)), dtype=float)
                )
        return np.stack(values), w

    def function_cri(
        self,
        fn: Callable[[Any], Any],
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for any function of the model: the
        posterior's ``alpha_ci / 2`` and ``1 - alpha_ci / 2`` points of
        ``fn(model)`` over the parameters' draws.

        Parameters
        ----------
        fn : callable
            ``fn(model)`` of the family's ordinary model (what ``fit``
            returns), giving a number or an array.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.
        """
        levels = interval_levels(alpha_ci, bound)
        values, w = self.function_draws(fn)
        shape = values.shape[1:]
        flat = values.reshape(len(values), -1)
        out = [
            np.reshape(weighted_quantile(flat, w, level), shape)
            for level in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def probability(self, condition: Callable[[Any], Any]) -> float:
        """
        The posterior probability that ``condition(model)`` holds: the
        chance, given the data and the priors, that a requirement is met
        (``lambda model: model.mtbf(500) >= 100``).
        """
        values, w = self.function_draws(
            lambda m: float(bool(np.all(condition(m))))
        )
        return float(w @ values.reshape(len(w)))

    def param_cri(
        self, name: str, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        """A credible interval for a parameter, by its name."""
        levels = interval_levels(alpha_ci, bound)
        out = np.asarray(self.posterior.quantile(name, levels), dtype=float)
        return out if len(levels) == 2 else out[0]

    # ------------------------------------------------------------------
    # Comparison and summary
    # ------------------------------------------------------------------
    def _bound(self) -> BoundPriors:
        return BoundPriors(
            self.priors,
            self.target.dist,
            self.parameter_names,
            list(self.target.bounds),
            False,
            False,
            False,
            {k: float(v) for k, v in self.fixed.items()},
            target=self.target,
        )

    def compare(self, alpha_ci: float = 0.05) -> Any:
        """
        The prior, the maximum-likelihood estimate (with its confidence
        interval) and the posterior of each free parameter, side by
        side.
        """
        from surpyval.bayes.model import Comparison, _prior_interval

        interval_levels(alpha_ci, "two-sided")
        bound = self._bound()
        names = self.parameter_names
        target = self.target
        estimates = (
            None
            if target.mle is None
            else np.asarray(target.estimates(), dtype=float)
        )
        table = {}
        for name in bound.free_names:
            i = names.index(name)
            t = bound.transforms[i]
            described = bound.descriptions.get(name)
            interval = None
            if name in self.priors.single:
                interval = _prior_interval(bound, name, t, alpha_ci)
            table[name] = {
                "prior": described
                or f"no prior given; flat on {t.scale_name}",
                "prior_interval": interval,
                "mle": None if estimates is None else float(estimates[i]),
                "mle_interval": (
                    None
                    if estimates is None
                    else target.mle_interval(name, alpha_ci)
                ),
                "posterior_median": float(self.params[i]),
                "posterior_interval": tuple(
                    float(v) for v in self.param_cri(name, alpha_ci)
                ),
            }
        return Comparison(table, alpha_ci)

    def _headline(self) -> str:
        return f"{self.target.family}: Bayesian fit, " + (
            self.target.describe_data()
        )

    def summary(self, alpha_ci: float = 0.05) -> str:
        """The fit in words: the data, the priors (the defaults marked),
        the posterior of each parameter, how it was computed and where
        the data disagree with a prior."""
        from surpyval.bayes.model import _fmt, _how_computed

        lvl = f"{1 - alpha_ci:.0%}"
        lines = [self._headline()]
        bound = self._bound()
        names = self.parameter_names
        lines.append("Priors:")
        for name in bound.free_names:
            t = bound.transforms[names.index(name)]
            said = bound.descriptions.get(
                name, f"no prior given; flat on {t.scale_name}"
            )
            lines.append(f"  {name}: {said}")
        for j in self.priors.joint:
            if j.is_likelihood:
                lines.append(f"  and {j.describe()}")
        for name, v in self.fixed.items():
            lines.append(f"  {name}: fixed at {_fmt(float(v))}")
        width = max(len(s) for s in names)
        lines.append(
            f"Posterior{' ' * max(width - 7, 1)}  median   {lvl} credible "
            "interval      max. lik."
        )
        target = self.target
        estimates = (
            None
            if target.mle is None
            else np.asarray(target.estimates(), dtype=float)
        )
        for name in bound.free_names:
            i = names.index(name)
            lo, hi = self.param_cri(name, alpha_ci)
            lines.append(
                f"  {name:<{width}}  {_fmt(float(self.params[i])):>8}   "
                f"{_fmt(float(lo))} - {_fmt(float(hi))}"
                + (
                    ""
                    if estimates is None
                    else f"      {_fmt(float(estimates[i]))}"
                )
            )
        lines.append(_how_computed(self))
        for note in self.diagnostics.get("prior_conflict", []):
            lines.append(f"The data disagree with the prior on {note}.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------
    # Predictive criteria and checks
    # ------------------------------------------------------------------
    def _unit_blocks(self, points: npt.NDArray) -> Any:
        """The log-likelihood of every unit at ``points``, as the
        criteria read it: ``(unit indices, (points, units))``."""
        try:
            ll = np.asarray(self.target.unit_log_lik(points), dtype=float)
        except NotImplementedError:
            raise ValueError(
                f"WAIC and LOO are not available for {self.target.family} "
                "in this version: it has no log-likelihood per unit."
            ) from None
        yield np.arange(ll.shape[1]), ll

    def _unit_counts(self) -> npt.NDArray:
        return np.asarray(self.target.unit_counts(), dtype=float)

    def _same_data(self, other: Any) -> bool:
        return bool(
            isinstance(other, BayesianResult)
            and type(other.target) is type(self.target)
            and self.target.same_data(other.target)
        )

    def waic(self) -> Any:
        """
        WAIC: the expected log predictive density of a new unit, from
        the posterior variance of each unit's log-likelihood. A unit is
        what the family's data are made of (a row of lives; one system's
        whole history for a recurrent-event process). See
        :class:`~surpyval.bayes.criteria.PredictiveCriterion`.
        """
        from surpyval.bayes.criteria import waic

        return waic(self)

    def loo(self) -> Any:
        """
        Leave-one-out cross-validation: the expected log predictive
        density of each unit from the posterior of the others, by
        Pareto-smoothed importance sampling (a grid's by quadrature). A
        unit is what the family's data are made of (a row of lives; one
        system's whole history for a recurrent-event process). See
        :class:`~surpyval.bayes.criteria.PredictiveCriterion`.
        """
        from surpyval.bayes.criteria import loo

        return loo(self)

    def ppc(
        self,
        statistic: str | Callable[[Any], float] | None = None,
        replicates: int = REPLICATES,
        random_state: Any = None,
    ) -> PredictiveCheck:
        """
        A posterior predictive check: simulate ``replicates`` data sets
        from posterior draws, under the observed design, and compare a
        statistic of each with the data's.

        Parameters
        ----------
        statistic : str or callable, optional
            One of the family's statistics by name
            (``statistics``), or a function of a data set as the
            family simulates it. By default the family's first.
        replicates : int, optional
            How many data sets. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed, by the one seed rule (``None``: numpy's global
            stream).

        Returns
        -------
        PredictiveCheck
        """
        from surpyval.utils.rng import as_generator

        known = self.target.statistics()
        if not known and not callable(statistic):
            raise ValueError(
                f"{self.target.family} has no posterior predictive check "
                "in this version."
            )
        if statistic is None:
            name, fn = next(iter(known.items()))
        elif callable(statistic):
            name = getattr(statistic, "__name__", "the given statistic")
            fn = statistic
        elif statistic in known:
            name, fn = statistic, known[statistic]
        else:
            raise ValueError(
                f"No statistic {statistic!r}; this family's are "
                f"{', '.join(known)}, or give a function of a data set."
            )
        replicates = int(replicates)
        if replicates < 20:
            raise ValueError(
                "replicates must be at least 20 for the replicates' range "
                f"to mean anything; got {replicates}."
            )
        rng = as_generator(random_state)
        seed = int(rng.integers(0, 2**31 - 1))
        draws = self.posterior.draws(replicates, random_state=seed)
        values = np.full(replicates, np.nan)
        with ignoring_warnings(), np.errstate(all="ignore"):
            for r, theta in enumerate(draws):
                try:
                    values[r] = float(fn(self.target.simulate(theta, rng)))
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    continue
            observed = float(fn(self.target.observed()))
        return PredictiveCheck(name, observed, values)

    @property
    def statistics(self) -> list[str]:
        """The names of the statistics :meth:`ppc` can compare."""
        return list(self.target.statistics())

    def diagnose(
        self, replicates: int = REPLICATES, random_state: Any = 0
    ) -> BayesianDiagnosis:
        """
        The fit's checks in one report: the engine and (for Laplace) its
        checks against the exact posterior, where the data disagree with
        the priors, and a posterior predictive check on each of the
        family's statistics.

        Parameters
        ----------
        replicates : int, optional
            The data sets simulated for each predictive check.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the predictive checks. Default 0, so the report
            is the same on every call.

        Returns
        -------
        BayesianDiagnosis
            Printable; ``checks`` holds ``(name, ok, detail)`` rows,
            ``predictive`` the predictive checks by statistic and ``ok``
            whether all passed.
        """
        from surpyval.bayes.model import _how_computed
        from surpyval.utils.rng import as_generator

        d = self.diagnostics
        checks = [
            (
                "posterior",
                d.get("approximation", "good") != "poor",
                _how_computed(self),
            )
        ]
        notes = d.get("prior_conflict", [])
        checks.append(
            (
                "priors against the data",
                not notes,
                "; ".join(notes) if notes else "no disagreement found",
            )
        )
        rng = as_generator(random_state)
        predictive = {}
        for name in self.target.statistics():
            check = self.ppc(name, replicates, rng)
            predictive[name] = check
            lo, hi = check.interval
            checks.append(
                (
                    f"predictive check, {name}",
                    check.ok,
                    f"observed {check.observed:.4g}, replicates' 90% range "
                    f"{lo:.4g} to {hi:.4g} (share at or above "
                    f"{check.p_value:.3f})",
                )
            )
        return BayesianDiagnosis(checks, predictive)

    # ------------------------------------------------------------------
    # As the prior of another fit
    # ------------------------------------------------------------------
    def as_prior(
        self,
        weight: float = 1.0,
        param: Any = None,
        widen: float = 1.0,
    ) -> Any:
        """
        This posterior as the prior of another fit of the same family
        (a later test phase, the next batch).

        With the data at hand, and the whole posterior wanted (no
        ``param``, no ``widen``), it is exact: this fit's priors times
        its data's likelihood raised to ``weight``. Otherwise it is a
        normal on the parameters' unconstrained scale fitted to the
        posterior (its covariance divided by ``weight`` and widened by
        ``widen``), and its description says so.
        """
        weight = float(weight)
        if not 0 < weight <= 1:
            raise ValueError(f"weight must be in (0, 1]; got {weight}.")
        if not float(widen) >= 1:
            raise ValueError(f"widen must be at least 1; got {widen}.")
        if self.data is not None and param is None and widen == 1:
            spec = PriorSpec(self.priors.single, self.priors.joint)
            for k, v in self.fixed.items():
                if k not in spec.single:
                    spec.single[k] = Fixed(v)
            spec.joint = spec.joint + [
                FitPowerPrior(
                    self.target,
                    weight,
                    said=f"the earlier fit's data at weight {weight:g}",
                )
            ]
            return spec
        post = self.posterior
        ts = post.transforms or [
            Transform.for_bounds(self.target.bounds[i]) for i in post.free_idx
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
            f"a normal summary of an earlier Bayesian {self.target.family} "
            f"fit, weight {weight:g}"
            + (f", widened x{widen:g}" if widen != 1 else "")
        )
        if param is None:
            return MultivariateNormal(names, mean, cov, said)
        wanted = [param] if isinstance(param, str) else list(param)
        for p in wanted:
            if p not in names:
                raise ValueError(
                    f"param={p!r} is not a free parameter of the earlier "
                    f"fit ({', '.join(names)})."
                )
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
    # Saving and loading
    # ------------------------------------------------------------------
    def _diagnostics_dict(self) -> dict:
        from surpyval.bayes.model import _jsonable

        diag = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in self.diagnostics.items()
        }
        out: dict = _jsonable(diag)
        return out

    def to_dict(self) -> dict:
        """
        A strict-JSON dictionary of the fit: the family's target (its
        maximum-likelihood fit and data), the priors, the posterior's
        points and the diagnostics. ``surpyval.from_dict`` reads it. (A
        prior on a function of the model, ``prior.on_function``, is
        saved by its description, without the function.)
        """
        import surpyval

        out = {
            "model": self.model_tag,
            "target": self.target.to_dict(),
            "priors": self.priors.to_dict(),
            "default_priors": dict(self.default_priors),
            "posterior": self.posterior.to_dict(),
            "diagnostics": self._diagnostics_dict(),
            "surpyval_version": surpyval.__version__,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> Any:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.bayes.target import target_from_dict

        require_model_tag(model_dict, cls.model_tag, "a Bayesian fit")
        with ignoring_warnings():
            target = target_from_dict(model_dict["target"])
        return cls._from_target(
            target,
            Posterior.from_dict(model_dict["posterior"]),
            PriorSpec.from_dict(model_dict["priors"]),
            model_dict.get("default_priors", {}),
            model_dict.get("diagnostics", {}),
        )


__all__ = [
    "BayesianDiagnosis",
    "BayesianResult",
    "PredictiveCheck",
    "bind",
    "fit_target",
    "log_posterior",
    "prior_conflicts",
]
