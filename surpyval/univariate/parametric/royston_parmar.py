"""
Royston-Parmar flexible parametric survival models.

A Royston-Parmar model replaces the *straight line* that a Weibull draws for
its log-cumulative-hazard against log time with a **restricted cubic spline**,
giving a fully parametric model with an arbitrarily flexible baseline shape --
smooth, differentiable, and extrapolable, unlike the step baseline of a Cox
fit, yet free of the rigid single shape a Weibull or log-normal imposes.

The model is defined on one of three link scales through

.. math::
    g\\bigl(S(t)\\bigr) = s(\\log t ; \\gamma),

with ``s`` a restricted cubic spline. The link ``g`` chooses the family:

* ``scale="hazard"`` -- :math:`g = \\log(-\\log S) = \\log H`, a
  proportional-hazards flexible model (``s`` with no internal knots is exactly
  a Weibull);
* ``scale="odds"`` -- :math:`g = \\operatorname{logit}(1 - S)`, a
  proportional-odds flexible model;
* ``scale="normal"`` -- :math:`g = \\Phi^{-1}(1 - S)`, a probit flexible model
  (no internal knots is exactly a log-normal).

Knots are placed at quantiles of the uncensored (event) log-times: the boundary
knots at their extremes, the internal knots at equally-spaced centiles. Beyond
the boundary knots the spline is linear (the "restricted" part), so the model
extrapolates with a Weibull-like tail. The number of knots -- set through
``df`` -- is the modelling choice; compare a few by AIC / BIC.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import ndtri as _ndtri
from scipy.stats import norm

from surpyval.serialisation import stamp_schema, to_native

_SCALES = ("hazard", "odds", "normal")


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted-cubic-spline design matrix (Durrleman-Simon) at ``x``.

    Columns are ``[1, x, v_1(x), ..., v_m(x)]`` for the ``m`` internal knots.
    """
    x = np.asarray(x, dtype=float)
    kmin, kmax = knots[0], knots[-1]
    cols = [np.ones_like(x), x]
    span = kmax - kmin
    for kj in knots[1:-1]:
        lam = (kmax - kj) / span
        cols.append(
            np.maximum(x - kj, 0.0) ** 3
            - lam * np.maximum(x - kmin, 0.0) ** 3
            - (1.0 - lam) * np.maximum(x - kmax, 0.0) ** 3
        )
    return np.column_stack(cols)


def _rcs_deriv(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Derivative of the RCS basis with respect to ``x``."""
    x = np.asarray(x, dtype=float)
    kmin, kmax = knots[0], knots[-1]
    cols = [np.zeros_like(x), np.ones_like(x)]
    span = kmax - kmin
    for kj in knots[1:-1]:
        lam = (kmax - kj) / span
        cols.append(
            3.0 * np.maximum(x - kj, 0.0) ** 2
            - 3.0 * lam * np.maximum(x - kmin, 0.0) ** 2
            - 3.0 * (1.0 - lam) * np.maximum(x - kmax, 0.0) ** 2
        )
    return np.column_stack(cols)


def _place_knots(x_events: np.ndarray, n_internal: int) -> np.ndarray:
    """Boundary + internal knots at quantiles of the event log-times."""
    lx = np.log(np.asarray(x_events, dtype=float))
    qs = np.linspace(0.0, 1.0, n_internal + 2)
    return np.quantile(lx, qs)


def _scale_terms(eta: np.ndarray, scale: str):
    """``(log S, log(-dS/deta))`` at linear predictor ``eta`` for a scale."""
    if scale == "hazard":
        log_S = -np.exp(eta)
        return log_S, eta + log_S
    if scale == "odds":
        log_S = -np.logaddexp(0.0, eta)  # log(1 / (1 + e^eta))
        log_1mS = eta + log_S  # log(e^eta / (1 + e^eta))
        return log_S, log_S + log_1mS
    if scale == "normal":
        return norm.logsf(eta), norm.logpdf(eta)
    raise ValueError(f"scale must be one of {_SCALES}; got {scale!r}")


def _sf_from_eta(eta: np.ndarray, scale: str) -> np.ndarray:
    if scale == "hazard":
        return np.exp(-np.exp(eta))
    if scale == "odds":
        return 1.0 / (1.0 + np.exp(eta))
    return norm.sf(eta)


class RoystonParmarModel:
    """A fitted Royston-Parmar flexible parametric model.

    Carries the spline ``knots``, the coefficients ``params`` (``gamma``), the
    link ``scale``, and (for a maximum-likelihood fit) the coefficient
    covariance for confidence bounds. Exposes the usual distribution surface:
    :meth:`sf`, :meth:`ff`, :meth:`hf`, :meth:`Hf`, :meth:`df`, :meth:`qf`,
    :meth:`random`, :meth:`mean`, and :meth:`cb`.
    """

    def __init__(self) -> None:
        self.scale = "hazard"
        self.knots = np.array([])
        self.params = np.array([])
        self.covariance: "np.ndarray | None" = None
        self.support = (0.0, np.inf)
        self.n = 0
        self.n_events = 0
        self._neg_ll = 0.0

    # -- linear predictor --------------------------------------------------

    def _eta(self, t: np.ndarray) -> np.ndarray:
        return _rcs_basis(np.log(np.asarray(t, dtype=float)), self.knots) @ (
            self.params
        )

    def _eta_deriv(self, t: np.ndarray) -> np.ndarray:
        return _rcs_deriv(np.log(np.asarray(t, dtype=float)), self.knots) @ (
            self.params
        )

    # -- distribution functions -------------------------------------------

    def sf(self, t: Any) -> np.ndarray:
        with np.errstate(all="ignore"):
            return _sf_from_eta(self._eta(t), self.scale)

    def ff(self, t: Any) -> np.ndarray:
        return 1.0 - self.sf(t)

    def Hf(self, t: Any) -> np.ndarray:
        return -np.log(self.sf(t))

    def hf(self, t: Any) -> np.ndarray:
        return self.df(t) / self.sf(t)

    def df(self, t: Any) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        with np.errstate(all="ignore"):
            eta = self._eta(t)
            sp = self._eta_deriv(t)
            _, log_negdS = _scale_terms(eta, self.scale)
            return np.exp(log_negdS + np.log(sp) - np.log(t))

    def qf(self, q: Any) -> np.ndarray:
        """Quantile function: the time at which ``ff(t) = q``."""
        scalar_in = np.ndim(q) == 0
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.empty_like(q)
        for i, qi in enumerate(q):
            target = 1.0 - qi  # sf(t) = 1 - q
            lo = self.knots[0] - 20.0
            hi = self.knots[-1] + 20.0
            out[i] = np.exp(
                brentq(
                    lambda lx: float(np.ravel(self.sf(np.exp(lx)))[0])
                    - target,
                    lo,
                    hi,
                )
            )
        return out[0] if scalar_in else out

    def random(self, size: int) -> np.ndarray:
        return self.qf(np.random.uniform(0, 1, size))

    def mean(self) -> float:
        """Mean life, by integrating the survival function."""
        from scipy.integrate import quad

        val, _ = quad(
            lambda t: float(np.ravel(self.sf(t))[0]), 0, np.inf, limit=200
        )
        return val

    # -- confidence bounds -------------------------------------------------

    def cb(
        self,
        t: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """Confidence bound on a function, via the linear predictor.

        The bound is formed on the (unbounded) linear predictor ``eta`` --
        whose variance is ``B Sigma B'`` from the covariance -- and then
        pushed through the link, so ``sf`` / ``ff`` bounds stay in ``(0, 1)``.
        ``S`` is monotone decreasing in ``eta`` on every scale.
        """
        if self.covariance is None:
            raise ValueError("Confidence bounds need a covariance (MLE fit).")
        t = np.atleast_1d(np.asarray(t, dtype=float))
        B = _rcs_basis(np.log(t), self.knots)
        eta = B @ self.params
        var = np.einsum("ij,jk,ik->i", B, self.covariance, B)
        se = np.sqrt(np.maximum(var, 0.0))

        if bound == "two-sided":
            z = _ndtri(1.0 - alpha_ci / 2.0)
            eta_hi = eta + z * se
            eta_lo = eta - z * se
            sf_lo = _sf_from_eta(eta_hi, self.scale)  # S decreasing in eta
            sf_hi = _sf_from_eta(eta_lo, self.scale)
            band = np.column_stack([sf_lo, sf_hi])
        else:
            z = _ndtri(1.0 - alpha_ci)
            signed = eta + (z if bound == "lower" else -z) * se
            band = _sf_from_eta(signed, self.scale)

        if on in ("sf", "R"):
            return band
        if on in ("ff", "F"):
            return 1.0 - (band[:, ::-1] if band.ndim == 2 else band)
        if on == "Hf":
            return -np.log(band[:, ::-1] if band.ndim == 2 else band)
        raise ValueError("cb 'on' supports 'sf', 'ff' and 'Hf'.")

    # -- information criteria ---------------------------------------------

    @property
    def k(self) -> int:
        return len(self.params)

    def neg_ll(self) -> float:
        return self._neg_ll

    def aic(self) -> float:
        return 2 * self.k + 2 * self._neg_ll

    def bic(self) -> float:
        return self.k * np.log(self.n) + 2 * self._neg_ll

    def summary(self) -> str:
        lines = [
            "Royston-Parmar Flexible Parametric Model",
            "========================================",
            f"Scale               : {self.scale}",
            f"Knots (log-time)    : {np.round(self.knots, 4).tolist()}",
            f"Internal knots      : {len(self.knots) - 2}",
            f"Observations        : {self.n}  (events {self.n_events})",
            f"log-likelihood      : {-self._neg_ll:.4f}",
            f"AIC / BIC           : {self.aic():.2f} / {self.bic():.2f}",
            "Coefficients        :",
        ]
        for i, g in enumerate(self.params):
            lines.append(f"    gamma_{i:<3}: {g:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "model": "RoystonParmarModel",
            "scale": self.scale,
            "knots": np.asarray(self.knots, float).tolist(),
            "params": np.asarray(self.params, float).tolist(),
            "n": int(self.n),
            "n_events": int(self.n_events),
            "_neg_ll": to_native(self._neg_ll),
        }
        if self.covariance is not None:
            out["covariance"] = np.asarray(self.covariance, float).tolist()
        return stamp_schema(out)

    def to_json(self, fp: "str | Path") -> None:
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "RoystonParmarModel":
        if model_dict.get("model") != "RoystonParmarModel":
            raise ValueError("Must create a RoystonParmarModel from its dict")
        out = cls()
        out.scale = model_dict["scale"]
        out.knots = np.array(model_dict["knots"], dtype=float)
        out.params = np.array(model_dict["params"], dtype=float)
        out.n = int(model_dict.get("n", 0))
        out.n_events = int(model_dict.get("n_events", 0))
        out._neg_ll = float(model_dict.get("_neg_ll", 0.0))
        if "covariance" in model_dict:
            out.covariance = np.array(model_dict["covariance"], dtype=float)
        return out

    @classmethod
    def from_json(cls, fp: "str | Path") -> "RoystonParmarModel":
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))


def _numerical_hessian(f, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    steps = np.maximum(np.abs(x), 1.0) * eps
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy()
            xpp[i] += steps[i]
            xpp[j] += steps[j]
            xpm = x.copy()
            xpm[i] += steps[i]
            xpm[j] -= steps[j]
            xmp = x.copy()
            xmp[i] -= steps[i]
            xmp[j] += steps[j]
            xmm = x.copy()
            xmm[i] -= steps[i]
            xmm[j] -= steps[j]
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (
                4 * steps[i] * steps[j]
            )
    return H


class RoystonParmar_:
    """Fitter for :class:`RoystonParmarModel`. Use the singleton
    :data:`RoystonParmar`.
    """

    def fit(
        self,
        x: Any,
        c: Any = None,
        n: Any = None,
        tl: Any = None,
        df: int = 3,
        scale: str = "hazard",
        knots: Any = None,
    ) -> RoystonParmarModel:
        """Fit a Royston-Parmar model by maximum likelihood.

        Parameters
        ----------
        x : array_like
            Observed times (all strictly positive).
        c : array_like, optional
            Censoring flags: ``0`` event, ``1`` right-censored. Default all
            events.
        n : array_like, optional
            Observation weights / counts. Default 1.
        tl : array_like or scalar, optional
            Left-truncation (delayed-entry) time per observation.
        df : int, optional
            Degrees of freedom = number of spline terms beyond the intercept;
            ``df - 1`` internal knots. ``df = 1`` is a Weibull (``scale`` =
            hazard) or log-normal (``scale`` = normal). Default 3.
        scale : {"hazard", "odds", "normal"}, optional
            The link scale (proportional hazards, proportional odds, probit).
        knots : array_like, optional
            Explicit knot locations *on the log-time scale* (including the two
            boundary knots), overriding the quantile-based default.
        """
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}; got {scale!r}")
        x = np.asarray(x, dtype=float).ravel()
        if np.any(x <= 0):
            raise ValueError(
                "Royston-Parmar requires strictly positive times."
            )
        c = (
            np.zeros(x.shape[0], dtype=int)
            if c is None
            else np.asarray(c, dtype=int).ravel()
        )
        if not np.all(np.isin(c, (0, 1))):
            raise ValueError(
                "Royston-Parmar supports observed (c=0) and right-censored "
                "(c=1) data only."
            )
        w = (
            np.ones(x.shape[0])
            if n is None
            else np.asarray(n, dtype=float).ravel()
        )
        if tl is not None:
            tl = np.broadcast_to(np.asarray(tl, dtype=float), x.shape).copy()

        event = c == 0
        if int(event.sum()) == 0:
            raise ValueError("At least one event (c=0) is required.")

        if knots is None:
            n_internal = max(int(df) - 1, 0)
            knots = _place_knots(x[event], n_internal)
        else:
            knots = np.asarray(knots, dtype=float)
        n_params = len(knots)  # [1, x] + (len(knots) - 2) internal terms

        lx = np.log(x)
        B = _rcs_basis(lx, knots)
        Bd = _rcs_deriv(lx, knots)
        B_tl = None if tl is None else _rcs_basis(np.log(tl), knots)

        def neg_ll(g):
            eta = B @ g
            sp = Bd @ g
            log_S, log_negdS = _scale_terms(eta, scale)
            ll = np.sum(w * log_S)  # log S for everyone
            ll += np.sum(
                w[event]
                * (
                    log_negdS[event]
                    + np.log(sp[event])
                    - lx[event]
                    - log_S[event]
                )
            )  # events: swap log S -> log f = log(-dS) + log s' - log t
            if B_tl is not None:
                log_S_tl, _ = _scale_terms(B_tl @ g, scale)
                ll -= np.sum(w * log_S_tl)  # condition on survival to tl
            return -ll

        # Initialise from the Weibull/log-normal that the no-knot model is.
        init = np.zeros(n_params)
        init[1] = 1.0
        with np.errstate(all="ignore"):
            res = minimize(
                neg_ll,
                init,
                method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9},
            )
            res = minimize(neg_ll, res.x, method="BFGS")
        gamma = res.x

        covariance = None
        with np.errstate(all="ignore"):
            try:
                cov = np.linalg.inv(_numerical_hessian(neg_ll, gamma))
                if np.all(np.isfinite(cov)):
                    covariance = cov
            except np.linalg.LinAlgError:
                covariance = None

        model = RoystonParmarModel()
        model.scale = scale
        model.knots = knots
        model.params = gamma
        model.covariance = covariance
        model.n = int(x.shape[0])
        model.n_events = int(event.sum())
        model._neg_ll = float(res.fun)
        return model


RoystonParmar = RoystonParmar_()
