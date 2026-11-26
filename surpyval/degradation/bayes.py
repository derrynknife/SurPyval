"""
Bayesian fits of the Wiener and gamma degradation processes.

``WienerProcess.fit_bayes`` and ``GammaProcess.fit_bayes`` put priors on
a process's parameters (and its stress coefficients) and combine them
with the likelihood ``fit`` maximises: the independent increments
between each unit's readings. The posterior is the shared layer's
(:func:`surpyval.bayes.fit_target`): on a grid for up to three free
parameters, exact to the grid's stated error, and the checked Laplace
approximation beyond.

What follows from it is the life of a new unit (the first passage of
its degradation from the starting level ``y0`` to the threshold) and the
remaining life of a unit at its current level, each two ways:

- **for a new unit, all uncertainty included** (``sf``, ``ff``, ``qf``,
  and the remaining life's median and interval): the posterior
  predictive, the first-passage probability averaged over the
  posterior;
- **for the population** (``cri``, ``quantile_cri``, ``mean_cri``): the
  credible interval of the first-passage probability, of a B-life and of
  the mean life, the posterior's quantiles of each.

Every one is computed at every point of the posterior at once, from the
first-passage distribution in closed form (the Inverse-Gaussian for the
Wiener process, the regularised incomplete gamma function for the gamma
process), so nothing is simulated.

The starting level ``y0`` is the maximum-likelihood fit's (given, or
estimated from the readings at the fitted rate) and held: it is not a
parameter of the likelihood of the increments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc, gammaln, log_ndtr, ndtr

from surpyval.bayes.result import BayesianResult, fit_target, interval_levels
from surpyval.bayes.target import Target, register_adapter, register_target
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import check_alpha_ci, check_option
from surpyval.utils.warnings import ignoring_warnings

#: The most elements of a points-by-increments block of the likelihood.
_BLOCK = 2_000_000
_ON = ("sf", "ff")


# ---------------------------------------------------------------------------
# The first passage, at every posterior point at once
# ---------------------------------------------------------------------------
def _wiener_ff(
    tau: npt.NDArray, mu: npt.NDArray, sigma: npt.NDArray, D: float
) -> npt.NDArray:
    """The Wiener process's first-passage probability by clock time
    ``tau`` over the distance ``D``, broadcasting ``tau`` against the
    parameters (the Inverse-Gaussian CDF, with its second term in log
    space, as ``WienerProcessModel`` takes it)."""
    tau, mu, sigma = np.broadcast_arrays(tau, mu, sigma)
    out = np.zeros(tau.shape)
    pos = (tau > 0) & np.isfinite(tau)
    t, m, s = tau[pos], mu[pos], sigma[pos]
    root = s * np.sqrt(t)
    a = (m * t - D) / root
    b = (m * t + D) / root
    out[pos] = np.minimum(ndtr(a) + np.exp(2 * m * D / s**2 + log_ndtr(-b)), 1)
    inf = np.isposinf(tau)
    ever = np.where(mu > 0, 1.0, np.exp(np.minimum(2 * mu * D / sigma**2, 0)))
    out[inf] = ever[inf]
    return out


def _wiener_sf(
    tau: npt.NDArray, mu: npt.NDArray, sigma: npt.NDArray, D: float
) -> npt.NDArray:
    """``1 - _wiener_ff``, from its own terms (exact in the upper tail)."""
    tau, mu, sigma = np.broadcast_arrays(tau, mu, sigma)
    out = np.ones(tau.shape)
    pos = (tau > 0) & np.isfinite(tau)
    t, m, s = tau[pos], mu[pos], sigma[pos]
    root = s * np.sqrt(t)
    a = (m * t - D) / root
    b = (m * t + D) / root
    log_upper = log_ndtr(-a)
    ratio = np.minimum(2 * m * D / s**2 + log_ndtr(-b) - log_upper, 0.0)
    out[pos] = np.exp(log_upper + np.log(-np.expm1(ratio)))
    inf = np.isposinf(tau)
    ever = np.where(mu > 0, 1.0, np.exp(np.minimum(2 * mu * D / sigma**2, 0)))
    out[inf] = 1.0 - ever[inf]
    return out


def _gamma_ff(
    tau: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray, D: float
) -> npt.NDArray:
    """The gamma process's first-passage probability by clock time
    ``tau``: ``P(W(tau) >= D)``, ``W(tau) ~ Gamma(alpha tau, beta)``."""
    tau, alpha, beta = np.broadcast_arrays(tau, alpha, beta)
    out = np.zeros(tau.shape)
    pos = (tau > 0) & np.isfinite(tau)
    out[pos] = gammaincc(alpha[pos] * tau[pos], beta[pos] * D)
    out[np.isposinf(tau)] = 1.0
    return out


def _gamma_sf(
    tau: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray, D: float
) -> npt.NDArray:
    tau, alpha, beta = np.broadcast_arrays(tau, alpha, beta)
    out = np.ones(tau.shape)
    pos = (tau > 0) & np.isfinite(tau)
    out[pos] = gammainc(alpha[pos] * tau[pos], beta[pos] * D)
    out[np.isposinf(tau)] = 0.0
    return out


def _gamma_passage_integral(c: npt.NDArray) -> npt.NDArray:
    """``g(c) = int_0^inf P(Gamma(s, 1) < c) ds``: a gamma process's mean
    first-passage time is ``g(beta D) / alpha`` (substitute ``s = alpha
    t`` in the integral of its survival). Computed by quadrature where
    the integrand falls, at each distinct ``c`` or, for many, on 400
    nodes in ``log c`` and interpolated (to about 1e-9 relative)."""
    c = np.asarray(c, dtype=float)

    def one(v: float) -> float:
        if not (np.isfinite(v) and v > 0):
            return float("nan")
        spread = 12.0 * np.sqrt(v) + 30.0
        lo = max(v - spread, 0.0)
        hi = v + 3.0 * spread
        with ignoring_warnings():
            rest, _ = quad(lambda s: gammainc(s, v), lo, hi, limit=200)
        return lo + rest

    flat = c.ravel()
    ok = np.isfinite(flat) & (flat > 0)
    out = np.full(flat.shape, np.nan)
    if not ok.any():
        return out.reshape(c.shape)
    uniq = np.unique(flat[ok])
    if uniq.size <= 64:
        vals = np.array([one(v) for v in uniq])
        out[ok] = vals[np.searchsorted(uniq, flat[ok])]
        return out.reshape(c.shape)
    nodes = np.geomspace(uniq[0], uniq[-1], 400)
    # g(c) - c is smooth and slowly varying (about 1/2 for large c)
    gap = np.array([one(v) for v in nodes]) - nodes
    out[ok] = flat[ok] + np.interp(np.log(flat[ok]), np.log(nodes), gap)
    return out.reshape(c.shape)


def _solve_quantile(
    ff: Any, p: float, scale: npt.NDArray, ever: npt.NDArray
) -> npt.NDArray:
    """The clock times ``tau`` (one a row) at which ``ff(tau)`` (rows of
    one value) reaches ``p``: bracketed by factors of ten about
    ``scale``, then bisected on ``log tau`` to the last bits; ``inf``
    where the passage probability never reaches ``p`` (``ever`` below
    it)."""
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    out = np.full(scale.shape, np.inf)
    live = ever >= p
    if not live.any():
        return out
    lo = np.log(scale[live]) - np.log(10.0)
    hi = np.log(scale[live]) + np.log(10.0)
    rows = np.flatnonzero(live)
    for _ in range(60):
        below = ff(np.exp(lo), rows) > p
        if not below.any():
            break
        lo = np.where(below, lo - np.log(10.0), lo)
    for _ in range(60):
        short = ff(np.exp(hi), rows) < p
        if not short.any():
            break
        hi = np.where(short, hi + np.log(10.0), hi)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        over = ff(np.exp(mid), rows) >= p
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    out[rows] = np.exp(0.5 * (lo + hi))
    return out


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------
@register_target
class ProcessTarget(Target):
    """A Wiener or gamma process's side of a Bayesian fit
    (:mod:`surpyval.bayes.target`): the likelihood of the increments
    ``fit`` maximises, one unit (a unit's whole path of readings) as the
    unit of the criteria, and replicates of the increments over the same
    time steps and stresses.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.degradation import WienerProcess
    >>> from surpyval.degradation.bayes import ProcessTarget
    >>> t = np.tile(np.arange(0, 60, 10.0), 3)
    >>> i = np.repeat(np.arange(3), 6)
    >>> y = 0.5 * t + np.random.default_rng(2).normal(0, 1, t.size)
    >>> target = ProcessTarget.from_readings(
    ...     WienerProcess, t, y, i, threshold=60)
    >>> target.names
    ['mu', 'sigma']
    >>> bool(np.isclose(target.neg_ll(target.start()), target.mle.neg_ll()))
    True
    """

    tag = "degradation-process"
    unit_name = "unit"

    def __init__(
        self,
        kind: str,
        mle: Any,
        readings: dict,
        resolution: float | None = None,
        gauge: dict | None = None,
    ) -> None:
        from surpyval.degradation.process_models import (
            GammaProcess,
            WienerProcess,
            _increments_and_stress,
            _quantised_unit_log_likelihood,
            _unit_steps,
        )

        check_option("kind", kind, ("wiener", "gamma"))
        self.kind = kind
        self.mle = mle
        self.data = readings
        self.dist = WienerProcess if kind == "wiener" else GammaProcess
        self.family = "Wiener process" if kind == "wiener" else "gamma process"
        self.names = list(mle.covariance_names)
        self.bounds = list(mle._declared_bounds())
        self.fixed = {}
        x, y, i, Z = (readings[k] for k in ("x", "y", "i", "Z"))
        dt, dy, z = _increments_and_stress(x, y, i, Z)
        self.dt, self.dy = dt, dy
        self.unit_index, _ = _unit_steps(i)
        self.n_units = int(self.unit_index.max()) + 1
        self.k = len(mle.parameter_names)
        self.z = None
        if mle.gamma is not None:
            assert z is not None
            self.z = z - np.asarray(mle.stress_ref, dtype=float)[None, :]
        self.resolution = None
        self.zero = np.zeros(dy.shape, dtype=bool)
        #: the gauge's settings (``step``, ``rounding``, ``exact_start``,
        #: ``method``), or ``None`` for readings taken as exact
        self.gauge = None if gauge is None else _gauge_settings(**gauge)
        self._gauge_ll: Any = None
        if self.gauge is not None:
            step = self.gauge["step"]
            self.dy = GammaProcess._gauge_increments(dy, step)
            cells, self._gauge_start = GammaProcess._gauge_design(
                step,
                self.gauge["rounding"],
                self.gauge["exact_start"],
                self.gauge["method"],
            )
            self._gauge_cells = cells
            self._gauge_ll = _quantised_unit_log_likelihood(
                self.dy, readings["i"], step, cells, self._gauge_start
            )
        elif kind == "gamma":
            self.zero, delta = GammaProcess._zero_increments(dy, resolution)
            self.resolution = float(delta) if self.zero.any() else None
            self._log_dy = np.log(np.where(self.zero, 1.0, dy))
        self._given_resolution = resolution

    @classmethod
    def from_readings(
        cls,
        fitter: Any,
        x: Any,
        y: Any,
        i: Any,
        threshold: float,
        Z: Any = None,
        stress_ref: Any = None,
        y0: float | None = None,
        resolution: float | None = None,
        gauge: dict | None = None,
    ) -> ProcessTarget:
        """The target of ``fitter``'s maximum-likelihood fit to the
        readings (``gauge``: the gamma process's gauge settings, as
        :func:`_gauge_settings` takes them)."""
        from surpyval.degradation.process_models import GammaProcess

        kind = "gamma" if fitter is GammaProcess else "wiener"
        extra: dict = {} if kind == "wiener" else {"resolution": resolution}
        if gauge is not None:
            g = _gauge_settings(**gauge)
            extra.update(
                gauge=g["step"],
                rounding=g["rounding"],
                exact_start=g["exact_start"],
                gauge_method=g["method"],
            )
        mle = fitter.fit(
            x,
            y,
            i,
            threshold,
            Z=Z,
            stress_ref=stress_ref,
            y0=y0,
            **extra,
        )
        readings = {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "i": np.asarray(i),
            "Z": None if Z is None else np.asarray(Z, dtype=float),
        }
        return cls(kind, mle, readings, resolution, gauge)

    def __reduce__(self) -> tuple:
        return (
            type(self),
            (
                self.kind,
                self.mle,
                self.data,
                self._given_resolution,
                self.gauge,
            ),
        )

    # -- the likelihood ------------------------------------------------------
    def _increment_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        """The log-likelihood of each increment at rows of parameters,
        ``(N, n)``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        dt, dy = self.dt[None, :], self.dy[None, :]
        if self.z is None:
            dtau = np.broadcast_to(dt, (len(theta), dt.shape[1]))
        else:
            dtau = dt * np.exp(theta[:, self.k :] @ self.z.T)
        a, b = theta[:, :1], theta[:, 1:2]
        with np.errstate(all="ignore"):
            if self.kind == "wiener":
                var = b**2 * dtau
                return -0.5 * (
                    np.log(2 * np.pi * var) + (dy - a * dtau) ** 2 / var
                )
            k = a * dtau
            out = (
                k * np.log(b) + (k - 1.0) * self._log_dy - b * dy - gammaln(k)
            )
            if self.zero.any():
                assert self.resolution is not None
                below = gammainc(k[:, self.zero], b * self.resolution)
                out[:, self.zero] = np.log(
                    np.maximum(below, np.finfo(float).tiny)
                )
            return out

    def _blocks(self, theta: npt.NDArray) -> Any:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        step = max(1, _BLOCK // max(self.dy.size, 1))
        for s in range(0, len(theta), step):
            yield s, self._increment_log_lik(theta[s : s + step])

    def _dtau(self, theta: npt.NDArray) -> npt.NDArray:
        """The clock steps ``(N, n)`` at rows of parameters."""
        dt = self.dt[None, :]
        if self.z is None:
            return np.broadcast_to(dt, (len(theta), dt.shape[1]))
        return dt * np.exp(theta[:, self.k :] @ self.z.T)

    def _gauge_unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        """Each unit's log-likelihood of gauge readings (the quantised
        likelihood ``fit(..., gauge=)`` maximises), ``(N, units)``, a
        block of rows at a time (the recursion holds ``cells**2`` numbers
        an increment and a row)."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        out = np.empty((len(theta), self.n_units))
        per_row = self.dy.size * (self._gauge_cells**2 + 2 * self._gauge_cells)
        step = max(1, _BLOCK // max(per_row, 1))
        with np.errstate(all="ignore"):
            for s in range(0, len(theta), step):
                rows = theta[s : s + step]
                k = rows[:, :1] * self._dtau(rows)
                out[s : s + len(rows)] = self._gauge_ll(k, rows[:, 1])
        # a point outside the parameters' range (alpha or beta not
        # positive) has no likelihood
        bad = ~np.all(np.isfinite(theta[:, :2]) & (theta[:, :2] > 0), axis=1)
        out[bad | np.isnan(out).any(axis=1)] = -np.inf
        return out

    def log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if self.gauge is not None:
            return self._gauge_unit_log_lik(theta).sum(axis=1)
        out = np.empty(len(theta))
        for s, block in self._blocks(theta):
            out[s : s + len(block)] = block.sum(axis=1)
        return np.where(np.isnan(out), -np.inf, out)

    def neg_ll(self, theta: npt.NDArray) -> float:
        return float(-self.log_lik(np.asarray(theta, dtype=float))[0])

    def estimates(self) -> npt.NDArray:
        return np.asarray(self.mle._estimates(), dtype=float)

    def unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if self.gauge is not None:
            return self._gauge_unit_log_lik(theta)
        member = np.zeros((self.dy.size, self.n_units))
        member[np.arange(self.dy.size), self.unit_index] = 1.0
        out = np.empty((len(theta), self.n_units))
        for s, block in self._blocks(theta):
            out[s : s + len(block)] = block @ member
        return out

    def unit_counts(self) -> npt.NDArray:
        return np.ones(self.n_units)

    # -- models and replicates ----------------------------------------------
    def with_params(self, theta: npt.NDArray) -> Any:
        return self.mle.with_params(np.asarray(theta, dtype=float))

    def simulate(self, theta: npt.NDArray, rng: Any) -> npt.NDArray:
        """The increments again, over the same time steps and stresses,
        from the process at ``theta``."""
        theta = np.asarray(theta, dtype=float)
        dtau = self.dt
        if self.z is not None:
            dtau = self.dt * np.exp(self.z @ theta[self.k :])
        a, b = theta[0], theta[1]
        if self.kind == "wiener":
            return rng.normal(a * dtau, b * np.sqrt(dtau))
        dy = rng.gamma(a * dtau, 1.0 / b)
        if self.gauge is not None:
            return self._read_off_gauge(dy, rng)
        if self.resolution is not None:
            dy = np.where(dy < self.resolution, 0.0, dy)
        return dy

    def _read_off_gauge(self, dy: npt.NDArray, rng: Any) -> npt.NDArray:
        """The recorded increments of true increments ``dy`` read off the
        gauge: each unit's level starts in its first bin (at the exact
        start's point, or anywhere in it, uniformly, as the likelihood
        has it), and each reading records the bin the level is in."""
        assert self.gauge is not None
        step = self.gauge["step"]
        unit = self.unit_index
        if self._gauge_start is None:
            offset = rng.uniform(0.0, step, self.n_units)
        else:
            offset = np.full(self.n_units, self._gauge_start)
        # each unit's increments are contiguous, in path order
        total = np.cumsum(dy)
        first = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1]])
        before = np.r_[0.0, total][first]
        level = offset[unit] + total - before[unit]
        bins = np.floor(level / step)
        previous = np.r_[0.0, bins[:-1]]
        previous[first] = 0.0
        return step * (bins - previous)

    def observed(self) -> npt.NDArray:
        return self.dy

    def statistics(self) -> dict:
        unit, dt = self.unit_index, self.dt
        span = np.bincount(unit, weights=dt, minlength=self.n_units)

        def spread(dy: npt.NDArray) -> float:
            rise = np.bincount(unit, weights=dy, minlength=self.n_units)
            return float(np.std(rise / span))

        def fastest(dy: npt.NDArray) -> float:
            return float(np.max(dy / dt))

        out = {}
        if self.n_units > 1:
            out["spread of the units' mean rates"] = spread
        out["largest increment a unit of time"] = fastest
        return out

    # -- priors --------------------------------------------------------------
    def check(self, bound: Any) -> None:
        from surpyval.bayes.prior import PowerPrior

        for j in bound.spec.joint:
            if isinstance(j, PowerPrior):
                raise ValueError(
                    "prior.from_data holds the lives of a life "
                    "distribution. For an earlier fleet's readings use "
                    "prior.from_fit(earlier_model, weight=...) (their "
                    "likelihood as a power prior), or "
                    "prior.from_model(earlier_model, param='mu')."
                )

    def check_flat(self, bound: Any) -> None:
        improper = bound.improper_names
        if not improper:
            return
        if any(j.is_likelihood for j in bound.spec.joint):
            return
        if self.dy.size <= len(improper):
            raise ValueError(
                f"{', '.join(improper)} "
                f"{'has' if len(improper) == 1 else 'have'} no prior "
                f"(flat), and {self.dy.size} increment"
                f"{'' if self.dy.size == 1 else 's'} do not determine "
                f"{'it' if len(improper) == 1 else 'them'}. Give "
                f"{'it a prior' if len(improper) == 1 else 'each a prior'} "
                "(for example prior.around(value, between=(low, high)))."
            )

    def life_quantity(self, spec: tuple) -> Any:
        """A life quantity of ``on_quantities`` (``("R", t)``, ``("q",
        p)``, ``("mean",)``) as a function of the parameters' dict: the
        first passage of a new unit from ``y0``, at the reference
        stress."""

        def value(theta: dict) -> npt.NDArray:
            p = [np.asarray(theta[n], dtype=float) for n in self.names]
            shape = np.broadcast(*p).shape
            rows = np.stack([np.broadcast_to(v, shape).ravel() for v in p], 1)
            life = ProcessLife(self, rows, None, None)
            if spec[0] == "R":
                out = life.sf(np.array([float(spec[1])]))[:, 0]
            elif spec[0] == "q":
                out = life.quantile(float(spec[1]))
            else:
                out = life.mean()
            return out.reshape(shape)

        return value

    # -- words and saving ----------------------------------------------------
    def describe_data(self) -> str:
        out = (
            f"{self.n_units} unit{'s' if self.n_units != 1 else ''}, "
            f"{self.dy.size} increment{'s' if self.dy.size != 1 else ''}"
        )
        if self.gauge is not None:
            out += f", read off a gauge of step {self.gauge['step']:g}"
        return out

    def same_data(self, other: Any) -> bool:
        return bool(
            self.kind == getattr(other, "kind", None)
            and getattr(self, "gauge", None) == getattr(other, "gauge", None)
            and np.array_equal(self.dt, other.dt)
            and np.array_equal(self.dy, other.dy)
            and (
                (self.z is None and other.z is None)
                or (
                    self.z is not None
                    and other.z is not None
                    and np.array_equal(self.z, other.z)
                )
            )
        )

    def to_dict(self) -> dict:
        from surpyval.serialisation import to_native

        r = self.data
        return {
            "target": self.tag,
            "kind": self.kind,
            "mle": self.mle.to_dict(),
            "x": r["x"].tolist(),
            "y": r["y"].tolist(),
            "i": [to_native(v) for v in r["i"]],
            "Z": None if r["Z"] is None else r["Z"].tolist(),
            "resolution": self._given_resolution,
            "gauge": self.gauge,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProcessTarget:
        import surpyval

        with ignoring_warnings():
            mle = surpyval.from_dict(d["mle"])
        readings = {
            "x": np.asarray(d["x"], dtype=float),
            "y": np.asarray(d["y"], dtype=float),
            "i": np.asarray(d["i"]),
            "Z": None if d["Z"] is None else np.asarray(d["Z"], dtype=float),
        }
        return cls(
            d["kind"], mle, readings, d.get("resolution"), d.get("gauge")
        )


def _gauge_settings(
    step: float,
    rounding: str = "nearest",
    exact_start: bool = False,
    method: str = "exact",
) -> dict:
    """A gauge's settings, checked, as ``GammaProcess.fit`` takes them
    (``gauge``, ``rounding``, ``exact_start``, ``gauge_method``)."""
    step = float(step)
    if not (np.isfinite(step) and step > 0):
        raise ValueError(
            f"gauge must be a positive, finite number, got {step!r}"
        )
    check_option("rounding", rounding, ("nearest", "floor"))
    check_option("gauge_method", method, ("exact", "independent"))
    return {
        "step": step,
        "rounding": str(rounding),
        "exact_start": bool(exact_start),
        "method": str(method),
    }


def _is_process_fit(model: Any) -> bool:
    from surpyval.degradation.process_models import FirstPassageProcessModel

    return isinstance(model, FirstPassageProcessModel)


def _target_of_fit(model: Any) -> Any:
    """A fitted process's target for ``prior.from_model``: its names,
    ranges, estimates and covariance (it holds no readings, so
    ``prior.from_fit`` takes a Bayesian fit's)."""
    from surpyval.degradation.process_models import WienerProcessModel

    out = ProcessTarget.__new__(ProcessTarget)
    out.kind = "wiener" if isinstance(model, WienerProcessModel) else "gamma"
    out.mle = model
    out.data = None
    out.family = "Wiener process" if out.kind == "wiener" else "gamma process"
    out.names = list(model.covariance_names)
    out.bounds = list(model._declared_bounds())
    out.fixed = {}
    return out


register_adapter(_is_process_fit, _target_of_fit)


# ---------------------------------------------------------------------------
# The life at given parameters
# ---------------------------------------------------------------------------
class PointClocks:
    """
    The process clock of every posterior point at once, under one stress
    row or a stress profile (a
    :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`).

    Each point runs its own clock, ``tau(t) = int_0^t AF(z(s)) ds`` with
    ``AF(z) = exp(gamma' (z - stress_ref))`` at that point's stress
    coefficients ``gamma``: at a constant stress ``AF t``; under a profile
    piecewise linear in ``t``, with the slope of the stress in force (what
    :class:`~surpyval.degradation._clock.StressClock` is for one model).
    The profile's segments are shared by the points, so only the slopes
    differ from one point to the next, and a block of points is computed
    with one matrix product.

    Parameters
    ----------
    gammas : array, ``(N, q)``, or None
        The points' stress coefficients (``q = 0`` or ``None``: a fit
        without stress, whose clock is the time itself).
    stress_ref : array, ``(q,)``, or None
    Z : array like, StepSchedule or None
        One stress row, a stress profile, or ``None`` (the reference
        stress).
    """

    def __init__(self, gammas: Any, stress_ref: Any, Z: Any) -> None:
        from surpyval.univariate.regression.tvc_schedule import (
            StepSchedule,
            check_from_origin,
        )

        self.schedule: Any = None
        self.n = 1 if gammas is None else len(gammas)
        self.af = np.ones(self.n)
        if gammas is None or Z is None or np.shape(gammas)[1] == 0:
            return
        self.gammas = np.asarray(gammas, dtype=float)
        self.ref = np.asarray(stress_ref, dtype=float)
        q = self.gammas.shape[1]
        if isinstance(Z, StepSchedule):
            if Z.p != q:
                raise ValueError(
                    "the StepSchedule has {} covariate(s) but the model was "
                    "fitted with {}".format(Z.p, q)
                )
            self.schedule = check_from_origin(Z)
            return
        from surpyval.degradation._clock import stress_row

        z = stress_row(Z, q) - self.ref
        self.af = np.exp(self.gammas @ z)

    def _knots(self, horizon: float) -> tuple[npt.NDArray, npt.NDArray]:
        """The profile's segments from 0 to past ``horizon`` (and past its
        last step): the calendar knots ``(K + 1,)`` and each segment's
        stress less the reference, ``(K, q)``."""
        from surpyval.univariate.regression.tvc_schedule import (
            segments_from_origin,
        )

        edges = self.schedule.edges
        finite = edges[np.isfinite(edges)]
        horizon = max(horizon, float(finite.max(initial=0.0)) + 1.0, 1.0)
        starts, ends, Zs = segments_from_origin(self.schedule, horizon)
        knots_t = np.concatenate([[starts[0]], ends])
        return knots_t, np.asarray(Zs, dtype=float) - self.ref

    def _block(self, K: int) -> int:
        return max(1, _BLOCK // max(K, 1))

    def _slopes(
        self, rows: Any, knots_t: npt.NDArray, dz: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The block of points' slopes ``(n, K)`` and their clocks at the
        knots ``(n, K + 1)``."""
        af = np.exp(self.gammas[rows] @ dz.T)
        steps = af * np.diff(knots_t)[None, :]
        knots_tau = np.concatenate(
            [np.zeros((len(af), 1)), np.cumsum(steps, axis=1)], axis=1
        )
        return af, knots_tau

    def tau(self, t: npt.NDArray) -> npt.NDArray:
        """``(N, m)``: each point's clock at the calendar times ``t``
        (``(m,)``)."""
        t = np.asarray(t, dtype=float)
        if self.schedule is None:
            return self.af[:, None] * t[None, :]
        finite = t[np.isfinite(t)]
        knots_t, dz = self._knots(float(finite.max(initial=0.0)))
        K = len(dz)
        j = np.searchsorted(knots_t, np.where(np.isfinite(t), t, 0.0), "right")
        j = np.clip(j - 1, 0, K - 1)
        out = np.empty((self.n, t.size))
        step = self._block(K)
        for s in range(0, self.n, step):
            rows = slice(s, s + step)
            af, knots_tau = self._slopes(rows, knots_t, dz)
            # before the origin the clock has not started
            since = np.maximum(t - knots_t[j], 0.0)
            out[rows] = knots_tau[:, j] + af[:, j] * since[None, :]
        out[:, np.isposinf(t)] = np.inf
        out[:, np.isnan(t)] = np.nan
        out[:, t < 0] = 0.0
        return out

    def inverse(self, tau: npt.NDArray) -> npt.NDArray:
        """``(N,)``: the calendar time at which each point's clock reads
        its own ``tau`` (``(N,)``)."""
        tau = np.asarray(tau, dtype=float)
        if self.schedule is None:
            return tau / self.af
        out = np.full(self.n, np.nan)
        out[np.isposinf(tau)] = np.inf
        ok = np.isfinite(tau)
        horizon = 1.0
        knots_t, dz = self._knots(horizon)
        step = self._block(len(dz))
        for s in range(0, self.n, step):
            rows = np.arange(s, min(s + step, self.n))
            rows = rows[ok[rows]]
            if not rows.size:
                continue
            want = tau[rows]
            af, knots_tau = self._slopes(rows, knots_t, dz)
            # lengthen the profile until every point's clock passes its
            # tau (a cyclic profile repeats; any other holds its last
            # stress)
            for _ in range(200):
                if np.all(knots_tau[:, -1] >= want):
                    break
                horizon = 2.0 * float(knots_t[-1])
                knots_t, dz = self._knots(horizon)
                step = self._block(len(dz))
                af, knots_tau = self._slopes(rows, knots_t, dz)
            K = len(dz)
            j = np.sum(knots_tau <= want[:, None], axis=1) - 1
            j = np.clip(j, 0, K - 1)
            at = np.arange(len(rows))
            out[rows] = knots_t[j] + (want - knots_tau[at, j]) / af[at, j]
        return out


class ProcessLife:
    """The first passage from a level to the threshold, at rows of
    parameters (every posterior point at once), at one stress.

    Parameters
    ----------
    target : ProcessTarget
    theta : array, ``(N, k)``
        The parameter rows, in ``target.names`` order.
    Z : array like, StepSchedule or None
        One stress row or a stress profile (a model fitted with stress),
        else ``None``: the reference stress.
    start : float or None
        The level the passage starts from; the fit's ``y0`` by default.
    """

    def __init__(
        self, target: Any, theta: npt.NDArray, Z: Any, start: float | None
    ) -> None:
        mle = target.mle
        self.kind = target.kind
        self.theta = np.atleast_2d(np.asarray(theta, dtype=float))
        level = mle.y0 if start is None else float(start)
        self.distance = float(mle.threshold) - level
        if not self.distance > 0:
            raise ValueError(
                f"The starting level {level:g} is at or above the "
                f"threshold {mle.threshold:g}: there is no passage left."
            )
        k = len(mle.parameter_names)
        stress = None if mle.gamma is None else mle._covariates(Z)
        self.clock = PointClocks(self.theta[:, k:], mle.stress_ref, stress)
        self.a = self.theta[:, 0]
        self.b = self.theta[:, 1]

    @property
    def is_profile(self) -> bool:
        """Whether the stress is a profile (a StepSchedule)."""
        return self.clock.schedule is not None

    def _tau(self, t: npt.NDArray) -> npt.NDArray:
        return self.clock.tau(np.asarray(t, dtype=float))

    def sf(self, t: npt.NDArray) -> npt.NDArray:
        """``(N, m)``: the probability of no passage by each time."""
        f = _wiener_sf if self.kind == "wiener" else _gamma_sf
        with np.errstate(all="ignore"):
            return f(
                self._tau(t), self.a[:, None], self.b[:, None], self.distance
            )

    def ff(self, t: npt.NDArray) -> npt.NDArray:
        f = _wiener_ff if self.kind == "wiener" else _gamma_ff
        with np.errstate(all="ignore"):
            return f(
                self._tau(t), self.a[:, None], self.b[:, None], self.distance
            )

    def ever(self) -> npt.NDArray:
        """The probability that the passage ever happens (below 1 for a
        Wiener process that drifts down)."""
        if self.kind == "gamma":
            return np.ones(len(self.a))
        with np.errstate(all="ignore"):
            return np.where(
                self.a > 0,
                1.0,
                np.exp(np.minimum(2 * self.a * self.distance / self.b**2, 0)),
            )

    def quantile(self, p: float) -> npt.NDArray:
        """``(N,)``: the time by which a fraction ``p`` have passed."""
        D = self.distance
        f: Any
        if self.kind == "wiener":
            scale = np.where(self.a > 0, D / self.a, D**2 / self.b**2)
            f = _wiener_ff
        else:
            scale = self.b * D / self.a
            f = _gamma_ff

        def ff(tau: npt.NDArray, rows: npt.NDArray) -> npt.NDArray:
            with np.errstate(all="ignore"):
                return f(tau, self.a[rows], self.b[rows], D)

        if p <= 0:
            return np.zeros(len(self.a))
        tau = _solve_quantile(ff, p, scale, self.ever())
        return self.clock.inverse(tau)

    def mean(self) -> npt.NDArray:
        """``(N,)``: the mean time to the passage (infinite where it may
        never happen), at a constant stress."""
        if self.is_profile:
            raise ValueError(
                "The mean life under a stress profile is not computed at "
                "every posterior point at once yet: use "
                "function_cri(lambda m: m.mean(schedule)) for its interval "
                "(and its posterior median as function_cri(lambda m: "
                "m.mean(schedule), alpha_ci=1.0))."
            )
        D = self.distance
        with np.errstate(all="ignore"):
            if self.kind == "wiener":
                tau = np.where(self.a > 0, D / self.a, np.inf)
            else:
                tau = _gamma_passage_integral(self.b * D) / self.a
        return tau / self.clock.af


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
class BayesianProcessRUL:
    """
    The remaining life of a unit at its current degradation level, from
    a Bayesian process fit (``BayesianProcessModel.predict_rul``).

    Attributes
    ----------
    median : float
        The predictive median remaining life: for this unit, all
        uncertainty included, an even chance of passing the threshold
        before it.
    interval : tuple
        The predictive equal-tailed interval: where this unit's remaining
        life lies with probability ``1 - alpha_ci``.
    median_cri : tuple
        The credible interval of the population's median remaining life
        (of units at this level): the uncertainty in the parameters
        alone.
    alpha_ci : float
        1 minus the intervals' level.
    """

    def __init__(
        self,
        model: Any,
        current: float,
        Z: Any,
        alpha_ci: float,
    ) -> None:
        self._model = model
        self._Z = Z
        self.current = float(current)
        self.alpha_ci = float(alpha_ci)
        lo, hi = alpha_ci / 2, 1 - alpha_ci / 2
        self.median = float(model.qf(0.5, Z, y0=current))
        self.interval = (
            float(model.qf(lo, Z, y0=current)),
            float(model.qf(hi, Z, y0=current)),
        )
        cri = model.quantile_cri(0.5, Z, alpha_ci=alpha_ci, y0=current)
        self.median_cri = (float(cri[0]), float(cri[1]))

    def sf(self, t: npt.ArrayLike) -> npt.NDArray:
        """The predictive probability that the unit lasts ``t`` more."""
        return np.asarray(self._model.sf(t, self._Z, y0=self.current))

    def cri(
        self,
        t: npt.ArrayLike,
        alpha_ci: float | None = None,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible band on the probability of lasting ``t`` more."""
        a = self.alpha_ci if alpha_ci is None else alpha_ci
        return np.asarray(
            self._model.cri(
                t, self._Z, on="sf", alpha_ci=a, bound=bound, y0=self.current
            )
        )

    def __repr__(self) -> str:
        level = 100 * (1 - self.alpha_ci)
        return (
            "Remaining useful life (Bayesian)\n"
            f"Current level           : {self.current:.6g}\n"
            f"Median (predictive)     : {self.median:.6g}\n"
            f"{level:g}% predictive interval : "
            f"{self.interval[0]:.6g} to {self.interval[1]:.6g}\n"
            f"{level:g}% credible interval of the median : "
            f"{self.median_cri[0]:.6g} to {self.median_cri[1]:.6g}"
        )


class BayesianProcessModel(BayesianResult):
    """
    A Bayesian fit of a Wiener or gamma degradation process
    (``WienerProcess.fit_bayes``, ``GammaProcess.fit_bayes``): the
    posterior of the process's parameters, and the life and remaining
    life that follow.

    ``sf``, ``ff`` and ``qf`` are the posterior predictive life of a new
    unit (its first passage from ``y0`` to the threshold, with the
    uncertainty in the parameters averaged in); ``cri`` and
    ``quantile_cri`` are credible intervals on the population's
    first-passage probability and B-lives; ``mean`` is the posterior
    median of the mean life (``mean_cri`` its interval); and
    ``predict_rul`` gives a unit's remaining life at its current level.
    Each is computed at every point of the posterior from the
    first-passage distribution in closed form.

    For a fit with stress, ``Z`` is one stress row (the stress the unit
    runs at) or a stress profile, a
    :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`:
    each posterior point's clock runs through the profile at that point's
    stress coefficients, so ``sf``, ``ff``, ``qf``, ``cri``,
    ``quantile_cri`` and ``predict_rul`` take one as the fitted model's
    methods do. The mean life under a profile is reached by
    :meth:`function_cri`, ``lambda m: m.mean(schedule)``.

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians, named by ``parameter_names``.
    threshold, y0 : float
        The failure level, and the level a new unit starts at (the
        maximum-likelihood fit's).
    mle : WienerProcessModel or GammaProcessModel
        The maximum-likelihood fit to the same readings.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.degradation import WienerProcess
    >>> rng = np.random.default_rng(1)
    >>> t = np.tile(np.arange(0, 110, 10.0), 5)
    >>> i = np.repeat(np.arange(5), 11)
    >>> steps = rng.normal(0.5 * 10, 1.0 * np.sqrt(10), size=(5, 10))
    >>> y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
    >>> model = WienerProcess.fit_bayes(t, y, i, threshold=100)
    >>> model.method
    'grid'
    >>> lo, hi = model.quantile_cri(0.1)    # the B10 life
    >>> bool(lo < model.qf(0.1) < hi)
    True
    """

    model_tag = "BayesianProcessModel"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = float(self.mle.threshold)
        self.y0 = float(self.mle.y0)
        self.kind = self.target.kind
        self.unit = getattr(self.mle, "unit", None)

    @property
    def is_accelerated(self) -> bool:
        """Whether the fit has stress coefficients."""
        return bool(self.mle.is_accelerated)

    def _life(self, Z: Any, y0: float | None) -> ProcessLife:
        if self.is_accelerated:
            if Z is None:
                raise ValueError(
                    "This process was fitted with stress; pass Z: one "
                    "stress row for a constant stress, or a StepSchedule "
                    "for a stress profile."
                )
        elif Z is not None:
            raise ValueError(
                "This process was fitted without stress; do not pass Z."
            )
        return ProcessLife(self.target, self.posterior.points, Z, y0)

    # -- the predictive life of a new unit ----------------------------------
    @keeps_query_shape
    def sf(
        self, x: npt.ArrayLike, Z: Any = None, *, y0: float | None = None
    ) -> npt.NDArray:
        """
        The probability that a new unit has not reached the threshold by
        ``x``, all uncertainty included: the first-passage survival
        averaged over the posterior. ``y0`` is the level the unit starts
        at (by default the fit's).
        """
        x = np.asarray(x, dtype=float)
        life = self._life(Z, y0)
        out = self.posterior.expectation(life.sf(x.ravel()))
        out = np.where(np.isnan(x.ravel()), np.nan, out)
        return out.reshape(x.shape)

    @keeps_query_shape
    def ff(
        self, x: npt.ArrayLike, Z: Any = None, *, y0: float | None = None
    ) -> npt.NDArray:
        """The predictive probability of having reached the threshold by
        ``x`` (averaged from the points' own failure probabilities, so
        it keeps its digits where it is small)."""
        x = np.asarray(x, dtype=float)
        life = self._life(Z, y0)
        out = self.posterior.expectation(life.ff(x.ravel()))
        out = np.where(np.isnan(x.ravel()), np.nan, out)
        return out.reshape(x.shape)

    @keeps_query_shape
    def qf(
        self, p: npt.ArrayLike, Z: Any = None, *, y0: float | None = None
    ) -> npt.NDArray:
        """
        The predictive quantile: the time by which a fraction ``p`` of
        new units will have reached the threshold, all uncertainty
        considered (the root of :meth:`ff` ``= p``). It is not the
        posterior median of the B-life, which :meth:`quantile_cri` gives
        (``point=True``).
        """
        u = np.asarray(p, dtype=float)
        flat = np.atleast_1d(u).ravel()
        if np.any((flat < 0) | (flat > 1)):
            raise ValueError("p must be between 0 and 1.")
        life = self._life(Z, y0)
        w = self.posterior.weights
        keep = w > 1e-14 * w.max()
        out = np.full(flat.shape, np.nan)
        for j, pj in enumerate(flat):
            if np.isnan(pj):
                continue
            if pj == 0:
                out[j] = 0.0
                continue
            q = life.quantile(float(pj))[keep]
            lo, hi = float(np.min(q)), float(np.max(q))

            def gap(log_t: float) -> float:
                F = life.ff(np.array([np.exp(log_t)]))[:, 0]
                return float(self.posterior.expectation(F)) - pj

            if not np.isfinite(hi):
                hi_t = max(lo, 1.0) * 2.0
                for _ in range(200):
                    if gap(np.log(hi_t)) >= 0:
                        break
                    hi_t *= 4.0
                else:
                    out[j] = np.inf
                    continue
                hi = hi_t
            if hi <= lo * (1 + 1e-14):
                out[j] = lo
                continue
            out[j] = np.exp(brentq(gap, np.log(lo), np.log(hi), xtol=1e-13))
        return out.reshape(u.shape)

    # -- the population: credible intervals ----------------------------------
    @keeps_query_shape
    def cri(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        *,
        y0: float | None = None,
    ) -> npt.NDArray:
        """
        A credible interval on the population's first-passage
        probability at times ``x``: the range in which the fraction of
        units still below the threshold (``on="sf"``) or past it
        (``"ff"``) lies with posterior probability ``1 - alpha_ci``.
        Two-sided adds a last ``[lower, upper]`` axis.
        """
        check_option("on", on, _ON)
        levels = interval_levels(alpha_ci, bound)
        x = np.asarray(x, dtype=float)
        life = self._life(Z, y0)
        values = life.sf(x.ravel()) if on == "sf" else life.ff(x.ravel())
        out = [
            np.reshape(self.posterior.wquantile(values, q), x.shape)
            for q in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    @keeps_query_shape
    def quantile_cri(
        self,
        p: npt.ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        point: bool = False,
        *,
        y0: float | None = None,
    ) -> npt.NDArray:
        """
        A credible interval on the population's B-life (``p=0.1`` for
        B10): the time by which a fraction ``p`` of units reach the
        threshold. ``point=True`` gives its posterior median, the best
        single guess at the B-life.
        """
        u = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
        levels = [0.5] if point else interval_levels(alpha_ci, bound)
        life = self._life(Z, y0)
        rows = []
        for ui in u:
            if not 0 < ui < 1:
                rows.append(np.full(len(levels), np.nan))
                continue
            q = life.quantile(float(ui))
            rows.append(
                np.atleast_1d(
                    np.asarray(self.posterior.wquantile(q, levels), float)
                )
            )
        out = np.stack(rows)
        if point or len(levels) == 1:
            return out[:, 0]
        return out

    def mean(self, Z: Any = None, *, y0: float | None = None) -> float:
        """The posterior median of the mean life of a new unit (its
        posterior mean can be very large, or infinite, where a slow
        process is still credible)."""
        life = self._life(Z, y0)
        return float(self.posterior.wquantile(life.mean(), 0.5))

    def mean_cri(
        self,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        *,
        y0: float | None = None,
    ) -> npt.NDArray:
        """A credible interval on the mean life."""
        levels = interval_levels(alpha_ci, bound)
        values = self._life(Z, y0).mean()
        out = [float(self.posterior.wquantile(values, q)) for q in levels]
        return np.asarray(out if len(out) == 2 else out[0])

    def predict_rul(
        self,
        current_degradation: float,
        *,
        Z: Any = None,
        alpha_ci: float = 0.05,
    ) -> BayesianProcessRUL:
        """
        The remaining life of a unit at its current degradation level:
        the predictive median and interval for this unit, and the
        credible interval of the population's median.

        Increments are independent, so the remaining passage over
        ``threshold - current_degradation`` follows the law of a fresh
        passage from that level. For a fit with stress, ``Z`` is the
        stress the unit runs at from now on: one row, or a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
        whose time zero is now.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.degradation import WienerProcess
        >>> rng = np.random.default_rng(1)
        >>> t = np.tile(np.arange(0, 110, 10.0), 5)
        >>> i = np.repeat(np.arange(5), 11)
        >>> steps = rng.normal(0.5 * 10, 1.0 * np.sqrt(10), size=(5, 10))
        >>> y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
        >>> model = WienerProcess.fit_bayes(t, y, i, threshold=100)
        >>> rul = model.predict_rul(60.0)
        >>> bool(rul.interval[0] < rul.median_cri[0] < rul.median
        ...      < rul.median_cri[1] < rul.interval[1])
        True
        """
        check_alpha_ci(alpha_ci)
        current = float(current_degradation)
        if np.isnan(current):
            raise ValueError(
                "current_degradation must be a number, got nan: the "
                "remaining life of a unit needs its current degradation"
            )
        if current >= self.threshold:
            raise ValueError(
                f"The unit is at {current:g}, at or past the threshold "
                f"{self.threshold:g}: it has no remaining life."
            )
        return BayesianProcessRUL(self, current, Z, alpha_ci)


def fit(
    fitter: Any,
    x: Any,
    y: Any,
    i: Any,
    threshold: float,
    Z: Any = None,
    stress_ref: Any = None,
    y0: float | None = None,
    resolution: float | None = None,
    *,
    gauge: dict | None = None,
    priors: Any = None,
    method: str = "auto",
    points: int | None = None,
    importance: bool = False,
) -> BayesianProcessModel:
    """The Bayesian fit of a Wiener or gamma process; see the fitters'
    ``fit_bayes`` (``gauge``: the gamma process's gauge settings, as
    :func:`_gauge_settings` takes them)."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    try:
        with quiet_maximum_warnings(), np.errstate(all="ignore"):
            target = ProcessTarget.from_readings(
                fitter,
                x,
                y,
                i,
                threshold,
                Z,
                stress_ref,
                y0,
                resolution,
                gauge,
            )
    except ValueError as e:
        raise ValueError(
            "fit_bayes starts from the maximum-likelihood fit, which "
            f"refuses these readings: {e}"
        ) from None
    if getattr(target.mle, "maximum", "") == "no finite maximum":
        raise ValueError(
            "Every increment is proportional to its time step (noise-free "
            "readings), so the likelihood has no finite maximum and the "
            "gamma process's alpha is not determined. Check the readings."
        )
    model: BayesianProcessModel = fit_target(
        target,
        priors,
        method=method,
        importance=importance,
        points=points,
        result=BayesianProcessModel,
    )
    return model


__all__ = [
    "BayesianProcessModel",
    "BayesianProcessRUL",
    "PointClocks",
    "ProcessLife",
    "ProcessTarget",
    "fit",
]
