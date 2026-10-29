"""The unconstrained scale of each parameter.

A Bayesian fit works on the scale where every parameter can take any
real value: the log of a positive parameter, the logit of a probability,
the parameter itself on the real line. A flat prior (``prior.flat()``) is
flat on this scale, a joint normal prior (``prior.from_model``) is normal
on it, and the grid is laid on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Transform:
    """The map between a parameter ``theta`` in ``(lower, upper)`` and its
    unconstrained value ``phi``.

    ``kind`` is ``"identity"`` (the real line), ``"log"`` (``theta >
    lower``), ``"neglog"`` (``theta < upper``) or ``"logit"`` (both ends
    finite).

    Examples
    --------
    >>> from surpyval.bayes.transforms import Transform
    >>> t = Transform.for_bounds((0, None))
    >>> t.kind, float(t.to_phi(1.0)), float(t.to_theta(0.0))
    ('log', 0.0, 1.0)
    """

    kind: str
    lower: float = -np.inf
    upper: float = np.inf

    @classmethod
    def for_bounds(cls, bounds: tuple) -> Transform:
        lo = -np.inf if bounds[0] is None else float(bounds[0])
        hi = np.inf if bounds[1] is None else float(bounds[1])
        if np.isfinite(lo) and np.isfinite(hi):
            return cls("logit", lo, hi)
        if np.isfinite(lo):
            return cls("log", lo, hi)
        if np.isfinite(hi):
            return cls("neglog", lo, hi)
        return cls("identity", lo, hi)

    @property
    def scale_name(self) -> str:
        """The scale in words, for summaries."""
        return {
            "identity": "the parameter's own scale",
            "log": "the log scale",
            "neglog": "the log scale (of the distance below its upper end)",
            "logit": "the logit scale",
        }[self.kind]

    def to_phi(self, theta: npt.ArrayLike) -> npt.NDArray:
        theta = np.asarray(theta, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kind == "identity":
                return theta
            if self.kind == "log":
                return np.log(theta - self.lower)
            if self.kind == "neglog":
                return np.log(self.upper - theta)
            u = (theta - self.lower) / (self.upper - self.lower)
            return np.log(u) - np.log1p(-u)

    def to_theta(self, phi: npt.ArrayLike) -> npt.NDArray:
        phi = np.asarray(phi, dtype=float)
        with np.errstate(over="ignore"):
            if self.kind == "identity":
                return phi
            if self.kind == "log":
                return self.lower + np.exp(phi)
            if self.kind == "neglog":
                return self.upper - np.exp(phi)
            u = 0.5 * (1.0 + np.tanh(0.5 * phi))
            return self.lower + (self.upper - self.lower) * u

    def log_jacobian(self, phi: npt.ArrayLike) -> npt.NDArray:
        """``log |d theta / d phi|`` at ``phi``."""
        phi = np.asarray(phi, dtype=float)
        if self.kind == "identity":
            return np.zeros_like(phi)
        if self.kind in ("log", "neglog"):
            return phi
        # d/dphi of logistic(phi) = u (1 - u), in a stable form
        log_u = -np.logaddexp(0.0, -phi)
        log_1mu = -np.logaddexp(0.0, phi)
        return np.log(self.upper - self.lower) + log_u + log_1mu

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "lower": None if not np.isfinite(self.lower) else self.lower,
            "upper": None if not np.isfinite(self.upper) else self.upper,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transform:
        lo = -np.inf if d.get("lower") is None else float(d["lower"])
        hi = np.inf if d.get("upper") is None else float(d["upper"])
        return cls(str(d["kind"]), lo, hi)
