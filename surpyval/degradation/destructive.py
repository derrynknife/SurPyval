r"""
Destructive degradation modelling.

In ordinary (repeated-measures) degradation testing each unit is measured many
times, tracing a path that is extrapolated to a failure threshold (see
:class:`~surpyval.degradation.DegradationAnalysis`). In a **destructive** test
the measurement *destroys* the specimen, so each unit yields exactly one
``(time, degradation)`` observation -- material/adhesive strength that can only
be read by breaking the specimen, insulation breakdown voltage, and so on. With
one point per unit there are no paths to fit; instead the *distribution of the
degradation as a function of time* is modelled directly and the failure-time
distribution induced from it.

Model
-----
The destructive measurement at time ``t`` follows a location-scale distribution
whose location moves with a transform of time,

.. math::
    Y \mid t \sim \mathrm{dist}\bigl(
    \text{loc} = \beta_0 + \beta_1\,\varphi(t),\ \text{scale} = \sigma\bigr),

with ``dist`` a surpyval location-scale distribution (``Normal`` for a
real-valued response, ``LogNormal`` for a positive one) and :math:`\varphi` a
time transform (``linear`` / ``log`` / ``sqrt`` / ``reciprocal``). Because the
fit is expressed through the distribution's own ``log_df`` / ``log_sf`` /
``log_ff``, censored measurements -- a strength below the test floor
(left-censored), a specimen that did not break at the maximum load
(right-censored) -- are handled by the ordinary ``c`` convention.

A unit fails when its degradation crosses the threshold ``D_f``. Reading that
off the fitted degradation distribution gives the induced lifetime
distribution:

* **increasing** degradation (wear/crack growth) -- ``F_T(t) = P(Y(t) > D_f)``;
* **decreasing** degradation (strength loss) -- ``F_T(t) = P(Y(t) < D_f)``.

This assumes the population ordering is preserved over time (only the location
moves), the standard destructive-degradation / degradation-distribution model
(Meeker & Escobar).
"""

import numpy as np
from scipy.optimize import minimize

from surpyval.serialisation import stamp_schema
from surpyval.univariate.parametric import LogNormal, Normal

# Time-transform bases phi(t): (callable, display name). The linear predictor
# is loc(t) = beta0 + beta1 * phi(t); the free parameters are the regression
# coefficients, so phi carries no parameters of its own.
_TRANSFORMS = {
    "linear": (lambda t: t, "t"),
    "log": (lambda t: np.log(t), "log(t)"),
    "sqrt": (lambda t: np.sqrt(t), "sqrt(t)"),
    "reciprocal": (lambda t: 1.0 / t, "1/t"),
}

# Distributions whose response is positive; their location parameter acts on
# the log scale, so ordinary-least-squares initial values use log(y).
_LOG_RESPONSE = {"LogNormal", "LogLogistic"}

_DIST_BY_NAME = {"Normal": Normal, "LogNormal": LogNormal}


def _resolve_distribution(distribution):
    if isinstance(distribution, str):
        if distribution not in _DIST_BY_NAME:
            raise ValueError(
                "distribution {!r} is not a known destructive-degradation "
                "response; use one of {} or pass the distribution object "
                "directly".format(distribution, sorted(_DIST_BY_NAME))
            )
        return _DIST_BY_NAME[distribution]
    return distribution


class DestructiveDegradationModel:
    """
    Result of :meth:`DestructiveDegradation.fit`.

    Exposes the induced *lifetime* distribution at the failure threshold
    (``sf`` / ``ff`` / ``Hf`` / ``df``) plus the fitted *degradation*
    distribution over time (``degradation_quantile``). The fitted parameters
    are the location intercept and slope ``beta`` and the scale ``sigma``.
    """

    def __init__(
        self,
        distribution,
        transform,
        direction,
        beta,
        sigma,
        threshold,
        data,
        neg_ll,
        transform_scores=None,
    ):
        self.distribution = distribution
        self.transform = transform  # name
        self._phi = _TRANSFORMS[transform][0]
        self.direction = direction  # "increasing" | "decreasing"
        self.beta = np.asarray(beta, dtype=float)  # [beta0, beta1]
        self.sigma = float(sigma)
        self.threshold = float(threshold)
        self.data = data  # dict of x, y, c (kept for bootstrap)
        self._neg_ll = float(neg_ll)
        self.k = self.beta.shape[0] + 1  # + sigma
        self.transform_scores = transform_scores

    # -- degradation distribution over time -------------------------------

    def _loc(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.beta[0] + self.beta[1] * self._phi(t)

    def degradation_quantile(self, q, t):
        """
        The ``q``-quantile of the destructive measurement at time ``t`` (the
        fitted degradation distribution ``dist(loc(t), sigma)``).
        """
        loc = self._loc(t)
        out = np.asarray(self.distribution.qf(q, loc, self.sigma), dtype=float)
        return out[0] if np.ndim(t) == 0 else out

    def median_degradation(self, t):
        """Median destructive measurement at time ``t``."""
        return self.degradation_quantile(0.5, t)

    # -- induced lifetime distribution at the threshold -------------------

    def ff(self, t):
        """Failure (CDF) of the lifetime induced by crossing the threshold."""
        loc = self._loc(t)
        thr = self.threshold
        if self.direction == "increasing":
            # failed once degradation exceeds the threshold
            out = np.asarray(
                self.distribution.sf(thr, loc, self.sigma), dtype=float
            )
        else:
            out = np.asarray(
                self.distribution.ff(thr, loc, self.sigma), dtype=float
            )
        return out[0] if np.ndim(t) == 0 else out

    def sf(self, t):
        """Reliability of the induced lifetime distribution."""
        return 1.0 - self.ff(t)

    def Hf(self, t):
        """Cumulative hazard of the induced lifetime distribution."""
        return -np.log(np.maximum(self.sf(t), np.finfo(float).tiny))

    def df(self, t):
        """
        Density of the induced lifetime distribution (finite-difference of the
        CDF; the closed form depends on the time transform).
        """
        scalar = np.ndim(t) == 0
        ta = np.atleast_1d(np.asarray(t, dtype=float))
        h = np.maximum(np.abs(ta), 1.0) * 1e-6
        out = (self.ff(ta + h) - self.ff(ta - h)) / (2.0 * h)
        return out[0] if scalar else out

    # -- confidence bounds (bootstrap) ------------------------------------

    def cb(
        self,
        t,
        on="sf",
        alpha_ci=0.05,
        bound="two-sided",
        n_boot=200,
        seed=None,
    ):
        """
        Bootstrap confidence bounds on the induced lifetime function ``on``.

        Units are resampled with replacement (each carrying its own
        ``(x, y, c)``) and the whole fit is rerun, folding the estimation
        uncertainty into the band. Percentile bounds are returned.

        Parameters
        ----------
        t : array_like
            Times at which to evaluate the bound(s).
        on : {'sf', 'ff', 'Hf'}, optional
            The lifetime function to bound. Default ``'sf'``.
        alpha_ci : float, optional
            Total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
        n_boot : int, optional
            Number of bootstrap resamples. Default 200.
        seed : optional
            Seed for the resampling.
        """
        if on not in ("sf", "ff", "Hf"):
            raise ValueError("`on` must be one of 'sf', 'ff', 'Hf'")
        if bound not in ("two-sided", "lower", "upper"):
            raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")
        t = np.atleast_1d(np.asarray(t, dtype=float))
        rng = np.random.default_rng(seed)
        x, y, c = self.data["x"], self.data["y"], self.data["c"]
        n = x.shape[0]

        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                m = DestructiveDegradation.fit(
                    x[idx],
                    y[idx],
                    threshold=self.threshold,
                    c=c[idx],
                    distribution=self.distribution,
                    transform=self.transform,
                    direction=self.direction,
                )
            except Exception:
                continue
            draws.append(getattr(m, on)(t))
        if not draws:
            raise RuntimeError("every bootstrap resample failed to fit")
        draws = np.vstack(draws)

        if bound == "lower":
            return np.quantile(draws, alpha_ci, axis=0)
        if bound == "upper":
            return np.quantile(draws, 1.0 - alpha_ci, axis=0)
        lo = np.quantile(draws, alpha_ci / 2.0, axis=0)
        hi = np.quantile(draws, 1.0 - alpha_ci / 2.0, axis=0)
        return np.stack([lo, hi], axis=-1)

    # -- serialisation ----------------------------------------------------

    def to_dict(self):
        """Serialise to a plain JSON-safe dict."""
        return stamp_schema(
            {
                "model": "DestructiveDegradationModel",
                "distribution": self.distribution.name,
                "transform": self.transform,
                "direction": self.direction,
                "beta": self.beta.tolist(),
                "sigma": float(self.sigma),
                "threshold": float(self.threshold),
            }
        )

    @classmethod
    def from_dict(cls, d):
        """Rebuild a model from :meth:`to_dict`."""
        if d.get("model") != "DestructiveDegradationModel":
            got = d.get("model")
            raise ValueError(
                "dict is not a DestructiveDegradationModel "
                "(model={!r})".format(got)
            )
        dist = _resolve_distribution(d["distribution"])
        return cls(
            distribution=dist,
            transform=d["transform"],
            direction=d["direction"],
            beta=np.asarray(d["beta"], dtype=float),
            sigma=float(d["sigma"]),
            threshold=float(d["threshold"]),
            data=None,
            neg_ll=np.nan,
        )

    def __repr__(self):
        return (
            "Destructive Degradation Model"
            "\n============================="
            "\nResponse distribution : {}"
            "\nTime transform        : {}"
            "\nDirection             : {}"
            "\nThreshold             : {:.6g}"
            "\nLocation              : {:.6g} + {:.6g}*{}"
            "\nScale (sigma)         : {:.6g}"
        ).format(
            self.distribution.name,
            _TRANSFORMS[self.transform][1],
            self.direction,
            self.threshold,
            self.beta[0],
            self.beta[1],
            _TRANSFORMS[self.transform][1],
            self.sigma,
        )


class DestructiveDegradation_:
    """
    Fitter for destructive degradation data (one destructive measurement per
    unit). Use the module-level singleton :data:`DestructiveDegradation`.
    """

    def _neg_ll(self, dist, phi_t, y, c, params):
        beta0, beta1, log_sigma = params
        sigma = np.exp(log_sigma)
        loc = beta0 + beta1 * phi_t
        ll = 0.0
        obs = c == 0
        if obs.any():
            ll = ll + dist.log_df(y[obs], loc[obs], sigma).sum()
        rc = c == 1
        if rc.any():
            ll = ll + dist.log_sf(y[rc], loc[rc], sigma).sum()
        lc = c == -1
        if lc.any():
            ll = ll + dist.log_ff(y[lc], loc[lc], sigma).sum()
        return -ll

    def _fit_one(self, dist, transform, x, y, c):
        phi = _TRANSFORMS[transform][0]
        phi_t = phi(x)
        # OLS initial values (on the log response for positive-support dists).
        resp = (
            np.log(y)
            if (dist.name in _LOG_RESPONSE and np.all(y > 0))
            else y.astype(float)
        )
        A = np.column_stack([np.ones_like(phi_t), phi_t])
        coef, *_ = np.linalg.lstsq(A, resp, rcond=None)
        resid = resp - A @ coef
        sigma0 = max(float(np.std(resid)), 1e-3)
        init = np.array([coef[0], coef[1], np.log(sigma0)])

        with np.errstate(all="ignore"):

            def fun(p):
                return self._neg_ll(dist, phi_t, y, c, p)

            res = minimize(fun, init, method="Nelder-Mead")
            res2 = minimize(fun, res.x, method="BFGS")
            res = res2 if res2.success else res

        beta = res.x[:2]
        sigma = float(np.exp(res.x[2]))
        return beta, sigma, float(res.fun)

    def fit(
        self,
        x,
        y,
        threshold,
        c=None,
        distribution=LogNormal,
        transform="linear",
        direction="auto",
    ):
        r"""
        Fit a destructive degradation model.

        Parameters
        ----------
        x : array_like
            Measurement time of each unit (one value per unit).
        y : array_like
            The destructive degradation measurement of each unit.
        threshold : float
            The degradation level ``D_f`` at which a unit is deemed failed.
        c : array_like, optional
            Censoring of each *measurement* (not the time): ``0`` observed,
            ``1`` right-censored (e.g. did not break at the maximum load),
            ``-1`` left-censored (below the test floor). Default all observed.
        distribution : Parametric or str, optional
            Location-scale response distribution -- ``LogNormal`` (default,
            positive response) or ``Normal``.
        transform : str, optional
            Time transform :math:`\varphi(t)` for the location: ``"linear"``,
            ``"log"``, ``"sqrt"``, ``"reciprocal"``, or ``"best"`` to pick the
            transform with the lowest AICc.
        direction : {'auto', 'increasing', 'decreasing'}, optional
            Whether degradation moves *up* toward the threshold (wear) or
            *down* toward it (strength loss). ``"auto"`` infers it from the
            sign of the time-degradation trend.

        Returns
        -------
        DestructiveDegradationModel
        """
        dist = _resolve_distribution(distribution)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        c = (
            np.zeros(x.shape[0], dtype=int)
            if c is None
            else np.atleast_1d(np.asarray(c, dtype=int))
        )
        if not (x.shape[0] == y.shape[0] == c.shape[0]):
            raise ValueError("x, y and c must have the same length")
        if x.shape[0] < 3:
            raise ValueError(
                "destructive degradation needs at least 3 units to identify "
                "the trend and scale"
            )
        if not np.isin(c, (-1, 0, 1)).all():
            raise ValueError("c must be 0 (observed), 1 (right) or -1 (left)")

        if direction == "auto":
            # Direction from the sign of the (raw-time) trend in the data.
            obs = c == 0
            xt, yt = (x[obs], y[obs]) if obs.sum() >= 3 else (x, y)
            slope = np.polyfit(xt, yt, 1)[0]
            direction = "increasing" if slope >= 0 else "decreasing"
        elif direction not in ("increasing", "decreasing"):
            raise ValueError(
                "direction must be 'auto', 'increasing' or 'decreasing'"
            )

        if transform == "best":
            scores = {}
            fits = {}
            n = x.shape[0]
            for name in _TRANSFORMS:
                try:
                    beta, sigma, nll = self._fit_one(dist, name, x, y, c)
                except Exception:
                    continue
                k = 3
                aic = 2 * k + 2 * nll
                aicc = (
                    aic + (2 * k**2 + 2 * k) / (n - k - 1)
                    if n - k - 1 > 0
                    else aic
                )
                scores[name] = aicc
                fits[name] = (beta, sigma, nll)
            if not fits:
                raise RuntimeError("no time transform could be fit")
            best = min(scores, key=scores.get)
            beta, sigma, nll = fits[best]
            transform = best
            transform_scores = scores
        else:
            if transform not in _TRANSFORMS:
                raise ValueError(
                    "transform must be one of {} or 'best'".format(
                        sorted(_TRANSFORMS)
                    )
                )
            beta, sigma, nll = self._fit_one(dist, transform, x, y, c)
            transform_scores = None

        return DestructiveDegradationModel(
            distribution=dist,
            transform=transform,
            direction=direction,
            beta=beta,
            sigma=sigma,
            threshold=threshold,
            data={"x": x, "y": y, "c": c},
            neg_ll=nll,
            transform_scores=transform_scores,
        )


#: Singleton fitter -- call ``DestructiveDegradation.fit(...)``.
DestructiveDegradation = DestructiveDegradation_()
