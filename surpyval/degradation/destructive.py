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

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.parametric import LogNormal
from surpyval.univariate.parametric.parametric import resolve_distribution
from surpyval.utils.dataframe import call_fit, frame_column, require_frame
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape

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


def _resolve_distribution(distribution: Any) -> Any:
    """
    The response distribution, given as the fitter or its name.

    A name resolves through the package's distribution registry (the same
    lookup the parametric models' ``from_dict`` uses), so every
    distribution ``fit`` accepts -- and hence every name ``to_dict`` can
    write -- reads back; only ``Normal`` and ``LogNormal`` used to, so a
    model fitted with, say, ``Logistic`` could be saved but not loaded.
    """
    if isinstance(distribution, str):
        return resolve_distribution(distribution)
    return distribution


def _transform_ok(transform: str, x: npt.NDArray) -> bool:
    """Whether the time transform is finite at every time in ``x``."""
    with np.errstate(all="ignore"):
        return bool(np.isfinite(_TRANSFORMS[transform][0](x)).all())


def _warn_if_noise_free(model: Any, x: npt.NDArray) -> None:
    """Warn when the fitted spread has collapsed onto the path (#392).

    Measurements that lie exactly on a path ``loc(t)`` (noise-free
    readings, and censored ones on the right side of it) leave the
    likelihood without a finite maximum: it keeps increasing as ``sigma``
    shrinks, and the fit stopped where rounding error in the residuals
    finally stood in for noise (``sigma = 9.9e-16`` on ``y = exp(4 - 0.02
    x)``), in silence.

    The criterion: at every measurement time the fitted distribution's
    interquartile range is below ``sqrt(eps)`` (1.5e-8) of its median's
    size. A maximum of the likelihood fixes a parameter to only half the
    digits of a double (the log-likelihood is quadratic there, so a
    relative change ``d`` moves it by about ``d**2``), so a spread that
    small is 0 to the precision of the fit: the readings are on the path.
    Any real measurement noise is orders of magnitude larger.
    """
    times = np.unique(x)
    with np.errstate(all="ignore"):
        low = model.degradation_quantile(0.25, times)
        high = model.degradation_quantile(0.75, times)
        mid = np.abs(model.degradation_quantile(0.5, times))
        tight = np.abs(high - low) <= np.sqrt(np.finfo(float).eps) * mid
    if not np.all(tight):
        return
    b0, b1 = model.beta
    path = f"{b0:.6g} + {b1:.6g}*{_TRANSFORMS[model.transform][1]}"
    warn_no_maximum(
        f"every measurement lies on the fitted path, location {path} "
        "(noise-free readings), so the likelihood keeps increasing as the "
        "scale sigma shrinks",
        f"The reported sigma = {model.sigma:.4g} (where the search "
        "stopped), its standard error and the bounds are meaningless",
        "the degradation is deterministic, every unit crossing the "
        "threshold at the same time; model it as such rather than with a "
        "response distribution",
    )


class DestructiveDegradationModel(SerialisableMixin):
    """
    Result of :meth:`DestructiveDegradation.fit`.

    Exposes the induced *lifetime* distribution at the failure threshold
    (``sf`` / ``ff`` / ``Hf`` / ``df``) plus the fitted *degradation*
    distribution over time (``degradation_quantile``). The fitted parameters
    are the location intercept and slope ``beta`` and the scale ``sigma``.

    Examples
    --------
    Six units destroyed in a strength test at each of four ages; a unit
    has failed once its strength is below 20:

    >>> import numpy as np
    >>> from surpyval.degradation import DestructiveDegradation
    >>> rng = np.random.default_rng(1)
    >>> x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    >>> y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    >>> model = DestructiveDegradation.fit(x, y, threshold=20)

    The median strength at ages 10 and 50, and the probability a unit is
    still above the threshold at 50 and 80:

    >>> model.degradation_quantile(0.5, [10, 50]).round(3)
    array([45.674, 19.987])
    >>> model.sf([50, 80]).round(4)
    array([0.4956, 0.    ])
    """

    def __init__(
        self,
        distribution: Any,
        transform: str,
        direction: str,
        beta: npt.NDArray,
        sigma: float,
        threshold: float,
        data: "dict | None",
        neg_ll: float,
        transform_scores: "dict | None" = None,
    ) -> None:
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

    def _loc(self, t: npt.ArrayLike) -> npt.NDArray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.beta[0] + self.beta[1] * self._phi(t)

    def degradation_quantile(
        self, p: npt.ArrayLike, x: npt.ArrayLike
    ) -> npt.NDArray:
        """
        The ``p``-quantile of the destructive measurement at time ``x`` (the
        fitted degradation distribution ``dist(loc(x), sigma)``).

        Parameters
        ----------
        p : float or array_like
            Probability (or probabilities) in ``(0, 1)``.
        x : float or array_like
            Time(s) at which to read the degradation distribution; a
            scalar ``x`` gives a scalar result.
        """
        loc = self._loc(x)
        out = np.asarray(self.distribution.qf(p, loc, self.sigma), dtype=float)
        return out[0] if np.ndim(x) == 0 else out

    def median_degradation(self, x: npt.ArrayLike) -> npt.NDArray:
        """Median destructive measurement at time ``x``."""
        return self.degradation_quantile(0.5, x)

    # -- induced lifetime distribution at the threshold -------------------

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """Failure (CDF) of the lifetime induced by crossing the threshold."""
        loc = self._loc(x)
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
        return out

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Reliability of the induced lifetime distribution."""
        return 1.0 - self.ff(x)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Cumulative hazard of the induced lifetime distribution."""
        return -np.log(np.maximum(self.sf(x), np.finfo(float).tiny))

    @keeps_query_shape
    def df(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        Density of the induced lifetime distribution (finite-difference of the
        CDF; the closed form depends on the time transform).
        """
        x = np.asarray(x, dtype=float)
        h = np.maximum(np.abs(x), 1.0) * 1e-6
        return (self.ff(x + h) - self.ff(x - h)) / (2.0 * h)

    # -- confidence bounds (bootstrap) ------------------------------------

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        n_boot: int = 200,
        random_state: "int | None" = None,
    ) -> npt.NDArray:
        """
        Bootstrap confidence bounds on the induced lifetime function ``on``.

        Units are resampled with replacement (each carrying its own
        ``(x, y, c)``) and the whole fit is rerun, folding the estimation
        uncertainty into the band. Percentile bounds are returned.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the bound(s).
        on : {'sf', 'ff', 'Hf'}, optional
            The lifetime function to bound (``'R'`` and ``'F'`` are
            accepted for ``'sf'`` and ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            Total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis, with
            ``alpha_ci / 2`` in each tail. Default ``'two-sided'``.
        n_boot : int, optional
            Number of bootstrap resamples. Default 200.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for the resampling. ``None`` (the default) seeds
            from numpy's global RNG, so ``np.random.seed`` controls it.
        """
        # 'R' and 'F' are the aliases every other ``cb`` takes; they
        # were refused here (#416).
        valid = ("sf", "R", "ff", "F", "Hf")
        if on not in valid:
            raise ValueError(
                "'on' must be one of {}; got {!r}".format(valid, on)
            )
        on = {"R": "sf", "F": "ff"}.get(on, on)
        bounds = ("two-sided", "lower", "upper")
        if bound not in bounds:
            raise ValueError(
                "'bound' must be one of {}; got {!r}".format(bounds, bound)
            )
        x = np.atleast_1d(np.asarray(x, dtype=float))
        rng = as_generator(random_state)
        if self.data is None:
            raise ValueError(
                "Bootstrap bounds need the fit data, which this model "
                "does not carry (it was restored from a dictionary "
                "written before the data was stored); refit it to get "
                "bounds."
            )
        xd, yd, cd = self.data["x"], self.data["y"], self.data["c"]
        n = xd.shape[0]

        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                m = DestructiveDegradation.fit(
                    xd[idx],
                    yd[idx],
                    threshold=self.threshold,
                    c=cd[idx],
                    distribution=self.distribution,
                    transform=self.transform,
                    direction=self.direction,
                )
            except Exception:
                continue
            draws.append(getattr(m, on)(x))
        if not draws:
            raise RuntimeError("every bootstrap resample failed to fit")
        draws_arr = np.vstack(draws)

        if bound == "lower":
            return np.quantile(draws_arr, alpha_ci, axis=0)
        if bound == "upper":
            return np.quantile(draws_arr, 1.0 - alpha_ci, axis=0)
        lo = np.quantile(draws_arr, alpha_ci / 2.0, axis=0)
        hi = np.quantile(draws_arr, 1.0 - alpha_ci / 2.0, axis=0)
        return np.stack([lo, hi], axis=-1)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted model to a plain, JSON-serialisable dict.

        The fit data ``(x, y, c)`` is stored along with the fitted
        parameters -- as ``DegradationModel`` stores its raw data -- so
        the restored model reproduces the original's predictions *and*
        its bootstrap :meth:`cb` (with the same ``random_state``, exactly).

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict = {
            "model": "DestructiveDegradationModel",
            "distribution": self.distribution.name,
            "transform": self.transform,
            "direction": self.direction,
            "beta": self.beta.tolist(),
            "sigma": float(self.sigma),
            "threshold": float(self.threshold),
            "neg_ll": float(self._neg_ll),
            "transform_scores": (
                None
                if self.transform_scores is None
                else {
                    str(k): float(v) for k, v in self.transform_scores.items()
                }
            ),
            "data": (
                None
                if self.data is None
                else {
                    "x": np.asarray(self.data["x"], dtype=float).tolist(),
                    "y": np.asarray(self.data["y"], dtype=float).tolist(),
                    "c": np.asarray(self.data["c"], dtype=int).tolist(),
                }
            ),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, d: dict) -> "DestructiveDegradationModel":
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Dictionaries written before the fit data was stored still load;
        the model they give predicts, but its :meth:`cb` raises because
        there is no data to resample.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            d, "DestructiveDegradationModel", "a destructive degradation model"
        )
        dist = _resolve_distribution(d["distribution"])
        data = d.get("data")
        return cls(
            distribution=dist,
            transform=d["transform"],
            direction=d["direction"],
            beta=np.asarray(d["beta"], dtype=float),
            sigma=float(d["sigma"]),
            threshold=float(d["threshold"]),
            data=(
                None
                if data is None
                else {
                    "x": np.asarray(data["x"], dtype=float),
                    "y": np.asarray(data["y"], dtype=float),
                    "c": np.asarray(data["c"], dtype=int),
                }
            ),
            neg_ll=float(d.get("neg_ll", np.nan)),
            transform_scores=d.get("transform_scores"),
        )

    def __repr__(self) -> str:
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

    def _neg_ll(
        self,
        dist: Any,
        phi_t: npt.NDArray,
        y: npt.NDArray,
        c: npt.NDArray,
        params: npt.NDArray,
    ) -> float:
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

    def _fit_one(
        self,
        dist: Any,
        transform: str,
        x: npt.NDArray,
        y: npt.NDArray,
        c: npt.NDArray,
    ) -> tuple:
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

            def fun(p: npt.NDArray) -> float:
                return self._neg_ll(dist, phi_t, y, c, p)

            res = minimize(fun, init, method="Nelder-Mead")
            res2 = minimize(fun, res.x, method="BFGS")
            res = res2 if res2.success else res

        beta = res.x[:2]
        sigma = float(np.exp(res.x[2]))
        return beta, sigma, float(res.fun)

    def fit_from_df(
        self,
        df: Any,
        x_col: str = "x",
        y_col: str = "y",
        c_col: "str | None" = None,
        **fit_kwargs: Any,
    ) -> "DestructiveDegradationModel":
        """
        Fit a destructive degradation model from the columns of a
        :class:`pandas.DataFrame`, with the argument names of
        ``DegradationAnalysis.fit_from_df``.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per unit tested.
        x_col : str, optional
            Column of the measurement times. Defaults to ``"x"``.
        y_col : str, optional
            Column of the measurements. Defaults to ``"y"``.
        c_col : str, optional
            Column of the measurements' censoring flags. Default all
            observed.
        **fit_kwargs
            Remaining arguments passed to :meth:`fit`: ``threshold``
            (required), and optionally ``distribution``, ``transform`` and
            ``direction``.

        Returns
        -------
        DestructiveDegradationModel
            The model :meth:`fit` returns for the same arrays.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval.degradation import DestructiveDegradation
        >>> rng = np.random.default_rng(1)
        >>> age = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
        >>> df = pd.DataFrame({
        ...     "age": age,
        ...     "strength": np.exp(4.0 - 0.02 * age + rng.normal(0, 0.1, 24)),
        ... })
        >>> model = DestructiveDegradation.fit_from_df(
        ...     df, x_col="age", y_col="strength", threshold=20
        ... )
        >>> model.sf([50, 80]).round(4)
        array([0.4956, 0.    ])
        """
        df = require_frame(df)
        arrays = {
            "x": frame_column(df, x_col, "x_col", time=True),
            "y": frame_column(df, y_col, "y_col"),
        }
        if c_col is not None:
            arrays["c"] = frame_column(df, c_col, "c_col")
        names = {"x": "x_col", "y": "y_col", "c": "c_col"}
        return call_fit(self, arrays, names, fit_kwargs)

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        threshold: float,
        c: "npt.ArrayLike | None" = None,
        distribution: Any = LogNormal,
        transform: str = "linear",
        direction: str = "auto",
    ) -> "DestructiveDegradationModel":
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
            positive response), ``Normal``, or another such as
            ``Logistic`` or ``LogLogistic`` -- as the object or its name.
            A distribution with positive support needs every measurement
            positive.
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
            The fitted model, whose life-distribution methods (``sf``,
            ``ff``, ...) give the probability of having crossed
            ``threshold`` by each time.

        Warns
        -----
        UserWarning
            "No finite maximum" when every measurement lies on the fitted
            path (noise-free readings): the fitted spread is then 0 to the
            precision of the fit, and ``sigma`` is meaningless.

        Examples
        --------
        Six units broken at each of four ages; strength falls
        log-linearly with age, and a unit has failed once its strength
        is below 20:

        >>> import numpy as np
        >>> from surpyval.degradation import DestructiveDegradation
        >>> rng = np.random.default_rng(1)
        >>> x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
        >>> y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
        >>> model = DestructiveDegradation.fit(x, y, threshold=20)
        >>> model
        Destructive Degradation Model
        =============================
        Response distribution : LogNormal
        Time transform        : t
        Direction             : decreasing
        Threshold             : 20
        Location              : 4.02814 + -0.0206616*t
        Scale (sigma)         : 0.0612395
        >>> model.sf([50, 80]).round(4)
        array([0.4956, 0.    ])
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
        # Bad input used to fit silently to nonsense or fail deep inside
        # the least-squares start (``LinAlgError: SVD did not converge``,
        # with LAPACK noise on stderr); refuse it up front instead.
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("x and y must contain only finite values")
        if isinstance(threshold, np.ndarray) and threshold.ndim == 0:
            threshold = threshold.item()
        if not isinstance(threshold, Number) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite number")
        if dist.support[0] >= 0 and np.any(y <= 0):
            raise ValueError(
                "the {} response distribution has positive support, but "
                "some measurements are zero or negative; use a "
                "distribution on the real line (e.g. Normal) for this "
                "response".format(dist.name)
            )

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
                if not _transform_ok(name, x):
                    continue  # e.g. log(t) or 1/t with a time of zero
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
            best = min(scores, key=lambda k: scores[k])
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
            if not _transform_ok(transform, x):
                raise ValueError(
                    "the {!r} time transform, {}, is not finite at every "
                    "measurement time (it needs positive times); use "
                    "another transform or drop the non-positive "
                    "times".format(transform, _TRANSFORMS[transform][1])
                )
            beta, sigma, nll = self._fit_one(dist, transform, x, y, c)
            transform_scores = None

        model = DestructiveDegradationModel(
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
        _warn_if_noise_free(model, x)
        return model


#: Singleton fitter -- call ``DestructiveDegradation.fit(...)``.
DestructiveDegradation = DestructiveDegradation_()
