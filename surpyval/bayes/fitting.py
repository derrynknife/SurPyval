"""``fit_bayes``: prior times likelihood, by the engine that fits.

The thin ``Weibull.fit_bayes`` / ``fit_bayes_from_df`` methods on every
distribution delegate here. Nothing here changes how ``fit`` works: the
maximum-likelihood fit is called only as a user would call it, for a
starting point and for the comparison in the summary.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.engines import conjugate
from surpyval.bayes.engines.grid import (
    GridError,
    find_mode,
    laplace_grid,
    lattice_quantile,
    run_grid,
)
from surpyval.bayes.likelihood import LogLikelihood, model_parameter_names
from surpyval.bayes.prior import BoundPriors, PriorSpec, quantity_at
from surpyval.utils.surpyval_data import SurpyvalData

METHODS = ("auto", "grid", "conjugate", "laplace")
#: Families whose support moves with their parameters, which the grid
#: does not take in this version.
_MOVING_SUPPORT = ("Uniform", "Beta4")


def _parameter_bounds(
    dist: Any, data: SurpyvalData, offset: bool, lfp: bool, zi: bool
) -> list[tuple]:
    from surpyval.univariate.parametric.parametric import _offset_upper

    bounds = list(dist.bounds)
    if offset:
        upper = _offset_upper({"x": data.x, "c": data.c}, zi)
        bounds = [(None, upper)] + bounds
    if lfp:
        bounds.append((0, 1))
    if zi:
        bounds.append((0, 1))
    return bounds


def _check_dist(dist: Any, method: str) -> None:
    if getattr(dist, "discrete", False) and dist.name != "Bernoulli":
        raise ValueError(
            f"fit_bayes covers the continuous distributions (and the "
            f"Bernoulli with a Beta prior) in this version; {dist.name} is "
            "discrete. log_likelihood_fn and posterior draws from another "
            "sampler are the way to a Bayesian fit of it for now."
        )
    if dist.name in ("ExactEventTime", "Hypoexponential"):
        raise ValueError(
            f"fit_bayes does not take {dist.name}: it is given its "
            "parameters (a point mass, or a chain of given rates), not "
            "fitted."
        )
    if dist.name in _MOVING_SUPPORT:
        raise ValueError(
            f"fit_bayes does not take {dist.name} in this version: its "
            "support moves with its parameters, which the grid does not "
            "hold. Use a distribution with a fixed support."
        )
    if method not in METHODS:
        raise ValueError(
            f"method must be one of {', '.join(METHODS)}; got {method!r}. "
            "(A sampler for more than three free parameters is planned.)"
        )


def _validate_data(
    dist: Any, data: SurpyvalData, offset: bool, zi: bool
) -> None:
    """The data checks ``fit`` makes that are about the data, not about
    whether maximum likelihood has an answer (a prior can supply what
    the data lack, so those are left to the posterior's own checks)."""
    dist._check_offset_and_grid(data, offset)
    if zi and dist.support[0] != 0:
        raise ValueError(
            "zero-inflated models can only work with models starting at 0"
        )
    if offset:
        dist._check_inside_support(data, offset=True)
    else:
        dist._check_inside_support(data, every_row=True, zero_inflated=zi)


def _fallback_start(
    dist: Any, data: SurpyvalData, bound: BoundPriors
) -> npt.NDArray | None:
    """A start where the maximum-likelihood fit fails: the plain fit (or
    the distribution's own initial guess) for its parameters, middling
    values for the extras, and each prior's centre where it has one."""
    plain = _mle_start(
        dist,
        data,
        False,
        False,
        False,
        {k: v for k, v in bound.fixed.items() if k in dist.parameter_names},
    )
    if plain is not None:
        dp = list(np.atleast_1d(plain.params))
    else:
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                dp = list(np.atleast_1d(dist._parameter_initialiser(data)))
        except (ValueError, TypeError, RuntimeError, IndexError):
            return None
    vals = dict(zip(dist.parameter_names, dp))
    t_up = bound.transforms[0].upper if bound.offset else 0.0
    vals["gamma"] = t_up - max(abs(t_up), 1.0) * 0.5
    vals["lfp_p"] = 0.9
    vals["f0"] = 0.05
    vals.update(bound.fixed)
    theta = np.array([[float(vals[n]) for n in bound.names]])
    with np.errstate(all="ignore"):
        phi = bound.to_phi(theta)[0]
    for k, p in bound.spec.single.items():
        if k in bound.free_names and not p.is_fixed:
            t = bound.transforms[bound.names.index(k)]
            c = p.centre_phi(t)
            if c is not None:
                phi[bound.free_names.index(k)] = c
    return phi if np.all(np.isfinite(phi)) else None


def _has_failure_information(data: SurpyvalData) -> bool:
    return bool(np.any(np.asarray(data.c) != 1))


def _mle_start(
    dist: Any,
    data: SurpyvalData,
    offset: bool,
    lfp: bool,
    zi: bool,
    fixed: dict[str, float],
) -> Any:
    """The maximum-likelihood fit, or None where it fails (quietly: it is
    only a starting point and a comparison)."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    if not _has_failure_information(data) or dist.discrete:
        return None
    try:
        with warnings.catch_warnings(), quiet_maximum_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                return dist.fit_from_surpyval_data(
                    data,
                    offset=offset,
                    lfp=lfp,
                    zi=zi,
                    fixed=fixed or None,
                )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def _theta_of(model: Any, names: list[str]) -> npt.NDArray:
    vals = dict(zip(model.dist.parameter_names, np.atleast_1d(model.params)))
    vals["gamma"] = getattr(model, "gamma", 0.0)
    vals["lfp_p"] = getattr(model, "lfp_p", 1.0)
    vals["f0"] = getattr(model, "f0", 0.0)
    return np.array([float(vals[n]) for n in names])


def check_summary(dist: Any, bound: BoundPriors) -> Any:
    """The numbers the grid's self-check compares: the 2.5%, 50% and
    97.5% points of each free parameter and of the 10% quantile."""

    def summary(phi: npt.NDArray, w: npt.NDArray, shape: tuple) -> npt.NDArray:
        theta = bound.full_theta(phi)
        cols = [theta[:, i] for i in bound.free_idx]
        with np.errstate(all="ignore"):
            q10 = np.asarray(
                quantity_at(dist, ("q", 0.1), bound.theta_dict(phi)),
                dtype=float,
            )
        if np.all(np.isfinite(q10)):
            cols.append(q10)
        return lattice_quantile(
            np.stack(cols, axis=1), w, [0.025, 0.5, 0.975], shape
        ).T

    return summary


def build_bound(
    dist: Any,
    data: SurpyvalData,
    priors: Any,
    offset: bool,
    lfp: bool,
    zi: bool,
    fixed: dict[str, float] | None,
) -> BoundPriors:
    spec = PriorSpec.parse(priors)
    names = model_parameter_names(dist, offset, lfp, zi)
    bounds = _parameter_bounds(dist, data, offset, lfp, zi)
    fixed = dict(fixed or {})
    for k in fixed:
        if k not in names:
            raise ValueError(
                f"fixed names {k!r}, which is not a parameter of this fit "
                f"({', '.join(names)})."
            )
    return BoundPriors(spec, dist, names, bounds, offset, lfp, zi, fixed)


def fit(
    dist: Any,
    x: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    *,
    priors: Any = None,
    offset: bool = False,
    zi: bool = False,
    lfp: bool = False,
    tl: Any = None,
    tr: Any = None,
    xl: Any = None,
    xr: Any = None,
    fixed: dict[str, float] | None = None,
    method: str = "auto",
    points: int | None = None,
    unit: Any = None,
) -> Any:
    """The Bayesian fit; see ``ParametricFitter.fit_bayes``."""
    from surpyval.utils.units import normalise_unit

    _check_dist(dist, method)
    if isinstance(x, SurpyvalData):
        others = dict(c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)
        given = [k for k, v in others.items() if v is not None]
        if given:
            raise ValueError(
                f"With a SurpyvalData as the data, leave out "
                f"{', '.join(given)}: the data object holds them."
            )
        data = x
    else:
        data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)
    canonical = normalise_unit(unit)
    held = getattr(data, "unit", None)
    if canonical is not None and held is not None and held != canonical:
        raise ValueError(
            f"The data's ages are in {held!r}, but unit={unit!r} was given "
            "to the fit: leave unit= out, or convert the data."
        )
    spec = PriorSpec.parse(priors)
    started = time.perf_counter()
    if not dist.discrete:
        _validate_data(dist, data, offset, zi)
    bound = build_bound(dist, data, spec, offset, lfp, zi, fixed)
    if bound.d == 0:
        raise ValueError(
            "Every parameter is fixed, so there is nothing to fit: use "
            f"{dist.name}.from_params(...)."
        )
    model = _posterior(dist, data, bound, method, points)
    model.unit = canonical if canonical is not None else held
    model.diagnostics["seconds"] = time.perf_counter() - started
    model._warn_conflicts()
    return model


def _posterior(
    dist: Any,
    data: SurpyvalData | None,
    bound: BoundPriors,
    method: str,
    points: int | None,
) -> Any:
    from surpyval.bayes.model import BayesianParametric

    offset, lfp, zi = bound.offset, bound.lfp, bound.zi
    case = None
    if data is not None and method in ("auto", "conjugate"):
        case = conjugate.applies(dist, bound, data, offset, lfp, zi)
    if method == "conjugate" and case is None:
        raise ValueError(
            "No closed form applies: the conjugate cases are an Exponential "
            "with a Gamma prior on failure_rate (exact and right-censored "
            "data, left truncation allowed), a Normal with sigma fixed and "
            "a Normal prior on mu (exact data), and a Bernoulli with a Beta "
            "prior on p. Use method='auto' for the grid."
        )
    if dist.discrete and case is None:
        raise ValueError(
            "A Bernoulli is fitted with a Beta prior on p (the conjugate "
            "case): give priors={'p': surpyval.Beta.from_params([a, b])}."
        )
    if case is not None:
        assert data is not None
        res = conjugate.solve(case, dist, bound, data)
        return BayesianParametric._from_conjugate(dist, data, bound, res)

    if bound.d > 3:
        raise ValueError(
            f"This fit has {bound.d} free parameters "
            f"({', '.join(bound.free_names)}); the grid takes 1 to 3. Fix "
            "some with fixed= or prior.fixed(), or use posterior draws "
            "from another sampler (log_likelihood_fn gives SurPyval's "
            "likelihood)."
        )

    if data is not None and bound.has_flat:
        _check_flat(dist, data, bound)
    ll = None if data is None else LogLikelihood(dist, data, offset, lfp, zi)

    def log_post(phi: npt.NDArray) -> npt.NDArray:
        phi = np.atleast_2d(phi)
        out = bound.log_prior(phi)
        if ll is not None:
            with np.errstate(all="ignore"):
                out = out + ll(bound.full_theta(phi))
        return np.where(np.isnan(out), -np.inf, out)

    starts = []
    mle = None
    if data is not None:
        mle = _mle_start(dist, data, offset, lfp, zi, bound.fixed)
        if mle is not None:
            theta = _theta_of(mle, bound.names)
            theta = np.atleast_2d(theta)
            phi0 = bound.to_phi(theta)[0]
            if np.all(np.isfinite(phi0)):
                starts.append(phi0)
    centre = bound.centre()
    if centre is not None:
        starts.append(centre)
    if data is not None and (mle is None or not starts):
        fallback = _fallback_start(dist, data, bound)
        if fallback is not None:
            starts.append(fallback)
    if not starts:
        raise ValueError(
            "No starting point: the maximum-likelihood fit failed and the "
            "priors do not say where the parameters are. Give priors for "
            f"{', '.join(bound.free_names)}."
        )
    try:
        mode, cov, _ = find_mode(log_post, starts)
        chosen = "laplace" if method == "laplace" else "grid"
        if chosen == "grid":
            result = run_grid(
                log_post,
                mode,
                cov,
                check_summary(dist, bound),
                points=points,
                flat_names=bound.flat_names,
            )
        else:
            result = laplace_grid(mode, cov, points)
    except GridError as e:
        raise ValueError(str(e)) from None
    return BayesianParametric._from_grid(
        dist, data, bound, result, chosen, log_post, mle
    )


def _check_flat(dist: Any, data: SurpyvalData, bound: BoundPriors) -> None:
    """A flat prior is allowed only where the data alone determine the
    parameter (principle 12): refuse where there is no failure
    information at all, and, where every parameter is flat, where the
    maximum-likelihood fit is not a verified maximum."""
    flat = ", ".join(bound.flat_names) or "the parameters with flat priors"
    if not _has_failure_information(data):
        raise ValueError(
            f"With no failures, the data do not determine {flat}; give it "
            "a prior (for example prior.around(value, between=(low, "
            "high)))."
        )
    if bound.flat_names and set(bound.flat_names) == set(bound.free_names):
        mle = _mle_start(
            dist, data, bound.offset, bound.lfp, bound.zi, bound.fixed
        )
        if mle is None or getattr(mle, "maximum", None) != "verified":
            state = "failed" if mle is None else repr(mle.maximum)
            raise ValueError(
                f"Every parameter has no prior (flat), and the data alone "
                f"do not determine them: the maximum-likelihood fit's "
                f"maximum is {state}. Give a prior for {flat}."
            )


def prior_model(dist: Any, priors: Any = None, **options: Any) -> Any:
    """The prior alone as a Bayesian model (no data), for
    ``prior.check``."""
    offset = options.pop("offset", False)
    lfp = options.pop("lfp", False)
    zi = options.pop("zi", False)
    fixed = options.pop("fixed", None)
    points = options.pop("points", None)
    if options:
        raise ValueError(
            f"Unknown options {sorted(options)}; prior.check takes offset, "
            "lfp, zi, fixed and points."
        )
    _check_dist(dist, "grid")
    if offset:
        raise ValueError(
            "An offset's range depends on the data, so its prior is checked "
            "with the data: fit, then call model.prior_check()."
        )
    empty = SurpyvalData(x=[1.0], c=[1])
    bound = build_bound(dist, empty, priors, offset, lfp, zi, fixed)
    if not bound.proper:
        flat = ", ".join(bound.flat_names) or "a parameter"
        raise ValueError(
            f"A prior check needs a proper prior on every free parameter; "
            f"{flat} has none (flat), which says nothing that can be "
            "checked. Give it a prior."
        )
    return _posterior(dist, None, bound, "grid", points)


def fit_from_df(
    dist: Any,
    df: Any,
    x_col: str | None = None,
    c_col: str | None = None,
    n_col: str | None = None,
    xl_col: str | None = None,
    xr_col: str | None = None,
    tl_col: Any = None,
    tr_col: Any = None,
    **options: Any,
) -> Any:
    """``fit_bayes`` on the columns of a DataFrame, named as
    ``fit_from_df`` names them."""
    from surpyval.utils.dataframe import (
        _truncation,
        frame_column,
        refuse_column_names,
        require_frame,
    )

    df = require_frame(df)
    refuse_column_names(options, "x", "c", "n", "xl", "xr", "tl", "tr")
    if x_col is not None and (xl_col is not None or xr_col is not None):
        raise ValueError(
            "Cannot use `x_col` and (`xl_col` and `xr_col`) together"
        )
    arrays: dict[str, Any] = {}
    if x_col is not None:
        arrays["x"] = frame_column(df, x_col, "x_col", time=True)
    elif xl_col is not None and xr_col is not None:
        arrays["x"] = np.column_stack(
            [
                frame_column(df, xl_col, "xl_col", time=True),
                frame_column(df, xr_col, "xr_col", time=True),
            ]
        )
    else:
        raise ValueError(
            "Name the column of times with `x_col`, or the interval ends "
            "with both `xl_col` and `xr_col`"
        )
    arrays["x"] = np.asarray(arrays["x"]).astype(float)
    if c_col is not None:
        arrays["c"] = frame_column(df, c_col, "c_col")
    if n_col is not None:
        arrays["n"] = frame_column(df, n_col, "n_col")
    if tl_col is not None or tr_col is not None:
        rows = len(df)
        arrays["t"] = np.column_stack(
            [
                (
                    np.full(rows, -np.inf)
                    if tl_col is None
                    else _truncation(df, tl_col, "tl_col")
                ),
                (
                    np.full(rows, np.inf)
                    if tr_col is None
                    else _truncation(df, tr_col, "tr_col")
                ),
            ]
        )
    return fit(dist, **arrays, **options)
