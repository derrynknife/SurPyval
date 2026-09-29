"""Step-stress general-path degradation: estimating the accelerated clock.

With ``acceleration="clock"`` in :meth:`DegradationAnalysis.fit` stress
speeds up the clock of every unit's degradation path, so unit ``i``'s
measurements are

    y_ij = g(tau_i(t_ij); theta_i) + eps_ij,   theta_i ~ MVN(mu, Sigma),

with ``tau_i(t) = sum_k AF(z_ik) dt_ik`` the reference-stress time the unit
has aged by ``t`` and ``AF(z) = exp(gamma' (z - z_ref))``. ``Z`` has one row
per measurement: the stress applied over the interval that ends at that
measurement (the first interval starts at time zero).

Once ``gamma`` is known the model is the ordinary general-path model on
``tau``, so everything downstream reuses the existing pipeline. This module
estimates ``gamma``:

* :func:`profile_least_squares` -- the two-stage route. For a trial
  ``gamma`` every unit's path is refitted on its ``tau`` and the residual
  sums of squares are pooled; ``gamma`` minimises the total. A unit held at
  one stress can absorb any acceleration into its own path parameters (every
  built-in path family is closed under rescaling time), so only units whose
  stress changes during the test carry information here.
* :func:`mixed_model_estimate` -- the mixed-model route. The units share one
  population of path parameters, so units at different constant stresses
  identify ``gamma`` too. A Lindstrom-Bates FOCE iteration with ``gamma``
  among the fixed effects (the path linearised in ``theta_i`` *and*
  ``gamma``) finds the neighbourhood, and the FOCE-approximate profile
  likelihood of ``gamma`` is then maximised directly.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar

from .population import (
    _conditional_mode,
    _prior_precision,
    reml_estimate_woodbury,
)


@dataclass
class ClockUnit:
    """One unit's time-ordered data with the stress over each interval."""

    x: npt.NDArray
    y: npt.NDArray
    dt: npt.NDArray
    s: npt.NDArray  # scaled stress over each interval, (n, q)

    def tau(self, g: npt.NDArray) -> npt.NDArray:
        """Reference-stress time at each measurement."""
        return np.cumsum(self.dt * np.exp(self.s @ g))

    def tau_gradient(self, g: npt.NDArray) -> npt.NDArray:
        """``d tau / d g`` at each measurement, ``(n, q)``."""
        exposure = self.dt * np.exp(self.s @ g)
        return np.cumsum(exposure[:, None] * self.s, axis=0)


def clock_units(
    x: npt.NDArray,
    y: npt.NDArray,
    i: npt.NDArray,
    units: npt.NDArray,
    Z: npt.NDArray,
    z_ref: npt.NDArray,
    scale: npt.NDArray,
) -> list[ClockUnit]:
    """Split the data by unit, time-ordered, with scaled interval stresses."""
    out = []
    for unit in units:
        mask = i == unit
        order = np.argsort(x[mask], kind="stable")
        xu = x[mask][order]
        dt = np.diff(np.concatenate([[0.0], xu]))
        s = (Z[mask][order] - z_ref) / scale
        out.append(ClockUnit(xu, y[mask][order], dt, s))
    return out


def path_time_derivative(
    path_model: Any, tau: npt.NDArray, theta: npt.NDArray
) -> npt.NDArray:
    """``d g(tau; theta) / d tau`` by finite differences (one-sided at 0)."""
    h = 1e-6 * np.maximum(np.abs(tau), 1e-6 * max(float(tau.max()), 1.0))
    central = tau > h
    lower = np.where(central, tau - h, tau)
    upper = tau + h
    with np.errstate(all="ignore"):
        rise = np.asarray(path_model.path(upper, *theta), dtype=float)
        rise = rise - np.asarray(path_model.path(lower, *theta), dtype=float)
    return rise / (upper - lower)


def _profile_rss(
    g: npt.NDArray, units: list[ClockUnit], path_model: Any
) -> float:
    total = 0.0
    for unit in units:
        tau = unit.tau(g)
        try:
            with np.errstate(all="ignore"):
                theta = path_model.fit(tau, unit.y)
                resid = unit.y - path_model.path(tau, *theta)
        except Exception:
            return np.inf
        rss = float(resid @ resid)
        if not np.isfinite(rss):
            return np.inf
        total += rss
    return total


def _span(units: list[ClockUnit]) -> float:
    """The largest scaled stress magnitude, which sets a sensible range for
    the scaled coefficients (``|g's|`` up to about 8, a factor of ~3000)."""
    return max(float(np.abs(u.s).max()) for u in units) or 1.0


def profile_least_squares(
    units: list[ClockUnit], path_model: Any, q: int
) -> npt.NDArray:
    """
    Two-stage estimate of the scaled stress coefficients: minimise the
    pooled per-unit residual sum of squares of the paths refitted on
    ``tau(g)``. For one covariate a grid locates the basin before a bounded
    scalar search; otherwise a local search starts from ``g = 0``.
    """
    bound = 8.0 / _span(units)

    if q == 1:
        grid = np.linspace(-bound, bound, 33)
        rss = np.array(
            [_profile_rss(np.array([v]), units, path_model) for v in grid]
        )
        if not np.isfinite(rss).any():
            raise ValueError(
                "the degradation paths could not be refitted on the "
                "accelerated clock for any trial stress coefficient"
            )
        k = int(np.nanargmin(np.where(np.isfinite(rss), rss, np.nan)))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        res = minimize_scalar(
            lambda v: _profile_rss(np.array([v]), units, path_model),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-10 * bound},
        )
        return np.array([res.x if res.fun <= rss[k] else grid[k]])

    def objective(g: npt.NDArray) -> float:
        value = _profile_rss(g, units, path_model)
        return value if np.isfinite(value) else 1e300

    best = minimize(
        objective,
        np.zeros(q),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20_000},
    )
    return np.asarray(best.x, dtype=float)


def _mode_start(
    path_model: Any,
    tau: npt.NDArray,
    y: npt.NDArray,
    mean: npt.NDArray,
    precision: npt.NDArray,
    sigma2: float,
    previous: npt.NDArray,
) -> npt.NDArray:
    """
    Where to start a unit's conditional-mode search: its previous mode, or,
    when that is unusable on the current clock, its least-squares fit there.
    When ``g`` moves, the clock of a unit held at a high stress can stretch
    so far that its previous mode overflows (an exponential path at the new
    ``tau``) and the damped search would stay put; the fresh fit is on the
    right scale.
    """
    with np.errstate(all="ignore"):
        resid = y - np.asarray(path_model.path(tau, *previous), dtype=float)
    delta = previous - mean
    value = float(resid @ resid / sigma2 + delta @ precision @ delta)
    if np.isfinite(value):
        return previous
    try:
        with np.errstate(all="ignore"):
            return np.asarray(path_model.fit(tau, y), dtype=float)
    except Exception:
        return previous


@dataclass
class _Population:
    """The FOCE state: per-unit modes and the population around them."""

    theta: npt.NDArray
    mean: npt.NDArray
    cov: npt.NDArray
    sigma2: float

    def copy(self) -> "_Population":
        return _Population(
            self.theta.copy(), self.mean.copy(), self.cov.copy(), self.sigma2
        )


def _foce(
    units: list[ClockUnit],
    path_model: Any,
    g: npt.NDArray,
    state: _Population,
    free_clock: bool,
    max_outer: int,
    tol: float,
    reml: "bool | None" = None,
) -> tuple[npt.NDArray, _Population, float, bool]:
    """
    The Lindstrom-Bates FOCE alternation on the accelerated clock.

    Each pass finds every unit's conditional mode of ``theta_i`` on its
    clock ``tau_i(g)`` given the current population, linearises the path
    about the modes, and takes a linear mixed-model step on the
    pseudo-response. With ``free_clock`` the linearisation is in ``g`` too
    (``G_i = g'(tau_i) d tau_i / d g``) and the REML step estimates
    ``(mu, g)`` by GLS: a fast joint iteration. Without it ``g`` is held
    fixed and the step maximises the plain likelihood (unless ``reml``),
    whose value at convergence is the FOCE-approximate marginal
    log-likelihood -- the profile objective for ``g``.

    Returns ``(g, state, objective, converged)`` with ``objective`` the last
    step's negative log-likelihood.
    """
    p = state.theta.shape[1]
    g = np.array(g, dtype=float)
    state = state.copy()
    if reml is None:
        reml = free_clock
    step_cap = 2.0 / _span(units)
    objective = np.inf
    for _ in range(max_outer):
        precision = _prior_precision(state.cov, state.sigma2)
        w_list, jac_list, a_list = [], [], []
        for k, unit in enumerate(units):
            tau = unit.tau(g)
            start = _mode_start(
                path_model,
                tau,
                unit.y,
                state.mean,
                precision,
                state.sigma2,
                state.theta[k],
            )
            theta = _conditional_mode(
                path_model,
                tau,
                unit.y,
                state.mean,
                precision,
                state.sigma2,
                start,
            )
            state.theta[k] = theta
            jac = np.asarray(path_model.jacobian(tau, *theta), dtype=float)
            fitted = np.asarray(path_model.path(tau, *theta), dtype=float)
            w = unit.y - fitted + jac @ theta
            if free_clock:
                clock_jac = path_time_derivative(path_model, tau, theta)[
                    :, None
                ] * unit.tau_gradient(g)
                w = w + clock_jac @ g
                a_list.append(np.hstack([jac, clock_jac]))
            else:
                a_list.append(jac)
            w_list.append(w)
            jac_list.append(jac)

        fixed, cov_new, sigma2_new, inner_ok, objective = (
            reml_estimate_woodbury(
                w_list,
                jac_list,
                state.cov,
                state.sigma2,
                a_mat_list=a_list,
                reml=reml,
            )
        )
        mean_new = fixed[:p]
        g_new = g
        if free_clock:
            g_new = fixed[p:]
            # a Gauss-Newton step in g can overshoot far from the solution,
            # where the clock is strongly nonlinear in g; cap it
            largest = float(np.max(np.abs(g_new - g)))
            if largest > step_cap:
                g_new = g + (g_new - g) * (step_cap / largest)
            if not np.isfinite(g_new).all():
                raise ValueError(
                    "the mixed-model estimate of the stress coefficients "
                    "diverged"
                )

        # g is on the scaled-stress axis (order one), so an absolute change;
        # the covariance relative to its diagonal, so near-zero correlations
        # do not stall the test
        diag = np.diag(state.cov)
        sd = np.sqrt(np.maximum(np.outer(diag, diag), 1e-300))
        changes = np.concatenate(
            [
                np.abs(g_new - g),
                np.abs(mean_new - state.mean) / (np.abs(state.mean) + 1e-12),
                (np.abs(cov_new - state.cov) / sd).ravel(),
                [abs(sigma2_new - state.sigma2) / state.sigma2],
            ]
        )
        g = g_new
        state.mean, state.cov, state.sigma2 = mean_new, cov_new, sigma2_new
        if float(np.max(changes)) < tol:
            return g, state, objective, bool(inner_ok)
    return g, state, objective, False


def mixed_model_estimate(
    units: list[ClockUnit],
    path_model: Any,
    g_init: npt.NDArray,
    theta_init: npt.NDArray,
    mean_init: npt.NDArray,
    cov_init: npt.NDArray,
    sigma2_init: float,
) -> tuple[npt.NDArray, bool, tuple]:
    """
    Mixed-model estimate of the scaled stress coefficients.

    The joint FOCE iteration (``g`` among the fixed effects) moves quickly
    to the neighbourhood of the estimate, but crawls along the ridge a unit
    held at one stress creates -- its rate and ``g`` nearly trade off, and
    the linearisation follows that curved ridge in small steps. So it is
    followed by a direct maximisation of the profile likelihood of ``g``:
    for each trial ``g`` the population is refitted by FOCE with ``g`` held
    fixed, and the approximate marginal likelihood compared. ``g`` changes
    the fixed-effects design, so the profile uses the plain likelihood
    rather than REML, whose adjustment term is not comparable across
    designs.

    Returns ``(g, converged, population)`` with ``population`` the REML
    estimate ``(mu, Sigma, sigma2, converged)`` of the reference-stress
    population of path parameters at that ``g``.
    """
    state = _Population(
        np.array(theta_init, dtype=float),
        np.array(mean_init, dtype=float),
        np.array(cov_init, dtype=float),
        float(sigma2_init),
    )
    g, state, _, _ = _foce(
        units, path_model, g_init, state, True, max_outer=30, tol=1e-4
    )

    warm = {"state": state}

    def profile(g_trial: npt.NDArray) -> float:
        try:
            _, fitted, objective, _ = _foce(
                units,
                path_model,
                g_trial,
                warm["state"],
                False,
                max_outer=100,
                tol=1e-5,
            )
        except (ValueError, np.linalg.LinAlgError):
            return np.inf
        if not np.isfinite(objective):
            return np.inf
        warm["state"] = fitted
        return objective

    span = _span(units)
    converged = False
    if g.size == 1:
        h = 0.05 / span
        try:
            res = minimize_scalar(
                lambda v: profile(np.array([v])),
                bracket=(g[0] - h, g[0] + h),
                method="brent",
                # far below the coefficient's sampling error
                options={"xtol": 1e-5},
            )
            if np.isfinite(res.fun):
                g, converged = np.array([res.x]), bool(res.success)
        except ValueError:
            pass
    else:
        simplex = np.vstack([g, g + np.eye(g.size) * (0.05 / span)])
        res = minimize(
            lambda v: min(profile(v), 1e300),
            g,
            method="Nelder-Mead",
            options={
                "initial_simplex": simplex,
                "xatol": 1e-5 / span,
                "fatol": 1e-7,
                "maxiter": 2_000,
            },
        )
        g, converged = np.asarray(res.x, dtype=float), bool(res.success)

    # the population at the estimate, by REML as for any other fit
    _, final, _, pop_ok = _foce(
        units,
        path_model,
        g,
        warm["state"],
        False,
        max_outer=100,
        tol=1e-6,
        reml=True,
    )
    return g, converged, (final.mean, final.cov, final.sigma2, pop_ok)
