"""Fitting every transition's parametric model at once.

A parametric multi-state model whose likelihood is a product of one
factor per transition is fitted a transition at a time (each with its own
univariate fitter). Where it is not -- a move known only to lie between
two times, states seen only at visits -- the transitions share factors,
and the models are fitted together here: the parameters of every
transition in one vector, on an unconstrained scale (the log of a
positive parameter, the logit of one between two limits), the negative
log-likelihood minimised by BFGS from the transitions' own fits, and the
covariance from the inverse of its Hessian there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# A negative log-likelihood the optimiser is given where the model is not
# defined (a parameter overflowed on its way back from the search scale).
_BAD = 1e300


def _bounds(model: Any) -> list:
    bounds = getattr(model, "parameter_bounds", None)
    if bounds is None or callable(bounds):
        bounds = model._parameter_bounds()
    return [(lo, hi) for lo, hi in bounds]


@dataclass
class Packer:
    """Each transition's parameters (in the order of its
    ``covariance_names``) in one search vector of the free ones, on an
    unconstrained scale."""

    models: list
    sizes: list
    est: npt.NDArray  # every parameter, on its own scale
    free: npt.NDArray  # bool: estimated (finite in the starting fit)
    lower: npt.NDArray
    upper: npt.NDArray

    @classmethod
    def of(cls, models: list) -> Packer:
        est, lower, upper, sizes = [], [], [], []
        for m in models:
            values = np.ravel(np.asarray(m._estimates(), dtype=float))
            bounds = _bounds(m)
            if len(bounds) < values.size:
                bounds = bounds + [(None, None)] * (values.size - len(bounds))
            est.append(values)
            sizes.append(values.size)
            for lo, hi in bounds[: values.size]:
                lower.append(-np.inf if lo is None else float(lo))
                upper.append(np.inf if hi is None else float(hi))
        e = np.concatenate(est)
        lo_arr, hi_arr = np.array(lower), np.array(upper)
        # (a parameter held where the model's fit left it: an aliased
        # coefficient, an accelerated life model's placeholder)
        free = np.isfinite(e) & (e > lo_arr) & (e < hi_arr)
        return cls(list(models), sizes, e, free, lo_arr, hi_arr)

    @property
    def k(self) -> int:
        return int(self.free.sum())

    def to_search(self, values: npt.NDArray) -> npt.NDArray:
        v = values[self.free]
        lo, hi = self.lower[self.free], self.upper[self.free]
        out = v.copy()
        both = np.isfinite(lo) & np.isfinite(hi)
        low = np.isfinite(lo) & ~np.isfinite(hi)
        high = ~np.isfinite(lo) & np.isfinite(hi)
        with np.errstate(all="ignore"):
            out[low] = np.log(v[low] - lo[low])
            out[high] = np.log(hi[high] - v[high])
            f = (v[both] - lo[both]) / (hi[both] - lo[both])
            out[both] = np.log(f) - np.log1p(-f)
        return out

    def from_search(self, theta: npt.NDArray) -> npt.NDArray:
        out = self.est.copy()
        lo, hi = self.lower[self.free], self.upper[self.free]
        v = np.asarray(theta, dtype=float).copy()
        both = np.isfinite(lo) & np.isfinite(hi)
        low = np.isfinite(lo) & ~np.isfinite(hi)
        high = ~np.isfinite(lo) & np.isfinite(hi)
        with np.errstate(all="ignore"):
            v[low] = lo[low] + np.exp(theta[low])
            v[high] = hi[high] - np.exp(theta[high])
            v[both] = lo[both] + (hi[both] - lo[both]) / (
                1.0 + np.exp(-theta[both])
            )
        out[self.free] = v
        return out

    def jacobian(self, theta: npt.NDArray) -> npt.NDArray:
        """d(parameter) / d(search value), for the free parameters."""
        lo, hi = self.lower[self.free], self.upper[self.free]
        d = np.ones(theta.size)
        both = np.isfinite(lo) & np.isfinite(hi)
        low = np.isfinite(lo) & ~np.isfinite(hi)
        high = ~np.isfinite(lo) & np.isfinite(hi)
        d[low] = np.exp(theta[low])
        d[high] = -np.exp(theta[high])
        p = 1.0 / (1.0 + np.exp(-theta[both]))
        d[both] = (hi[both] - lo[both]) * p * (1.0 - p)
        return d

    def rebuild(self, values: npt.NDArray) -> list:
        out = []
        start = 0
        for m, size in zip(self.models, self.sizes):
            out.append(m.with_params(values[start : start + size]))
            start += size
        return out


@dataclass
class JointFit:
    """What a joint fit reached."""

    models: list
    est: npt.NDArray  # every parameter, by the models' covariance_names
    cov: npt.NDArray
    neg_ll: float
    k: int
    maximum: str

    def to_dict(self) -> dict:
        return {
            "est": [float(v) for v in self.est],
            "cov": np.asarray(self.cov, dtype=float).tolist(),
            "neg_ll": float(self.neg_ll),
            "k": int(self.k),
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, d: dict, models: list) -> JointFit:
        return cls(
            models=models,
            est=np.array(d["est"], dtype=float),
            cov=np.array(d["cov"], dtype=float),
            neg_ll=float(d["neg_ll"]),
            k=int(d["k"]),
            maximum=str(d["maximum"]),
        )


def fit(
    models: list,
    neg_ll: Callable[[list], float],
    what: str,
) -> JointFit:
    """Maximise the likelihood over every transition's parameters, from
    ``models`` (the transitions' starting fits); ``neg_ll(models)`` is the
    negative log-likelihood of a list of rebuilt models. ``what`` names
    the fit in a warning."""
    from scipy.optimize import minimize

    from surpyval.utils.linalg import numerical_hessian
    from surpyval.utils.no_maximum import warn_unverified
    from surpyval.utils.warnings import ignoring_warnings

    pack = Packer.of(models)

    def f(theta: npt.NDArray) -> float:
        try:
            with np.errstate(all="ignore"), ignoring_warnings():
                value = float(neg_ll(pack.rebuild(pack.from_search(theta))))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return _BAD
        return value if np.isfinite(value) else _BAD

    theta0 = pack.to_search(pack.est)
    res = minimize(f, theta0, method="BFGS", options={"gtol": 1e-6})
    theta = res.x
    # (BFGS's own stop can come a little early on a flat valley: once more
    # from where it stopped, which is cheap where it was already there)
    res2 = minimize(f, theta, method="BFGS", options={"gtol": 1e-7})
    if res2.fun <= res.fun:
        res, theta = res2, res2.x
    value = float(res.fun)
    # (a step of the search scale's own size: a log shape near 0, a shape
    # near 1, would otherwise get a step too small for the rounding of
    # the likelihood)
    step = np.finfo(float).eps ** 0.25 * np.maximum(np.abs(theta), 1.0)
    H = numerical_hessian(f, theta, step)
    maximum = "verified"
    cov_t = None
    try:
        eig = np.linalg.eigvalsh(0.5 * (H + H.T))
        if np.all(np.isfinite(H)) and eig.min() > 0:
            cov_t = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_t = None
    if cov_t is None or value >= _BAD:
        maximum = "unverified"
        warn_unverified(
            what,
            "the Hessian of the negative log-likelihood is not positive "
            "definite there",
            "check that every transition has enough moves to estimate its "
            "distribution, or use a simpler distribution (Exponential) for "
            "the rare ones",
        )
    values = pack.from_search(theta)
    n_all = values.size
    cov = np.zeros((n_all, n_all))
    if cov_t is not None:
        J = pack.jacobian(theta)
        idx = np.flatnonzero(pack.free)
        cov[np.ix_(idx, idx)] = J[:, None] * cov_t * J[None, :]
    else:
        cov[:] = np.nan
    return JointFit(
        models=pack.rebuild(values),
        est=values,
        cov=cov,
        neg_ll=value,
        k=pack.k,
        maximum=maximum,
    )
