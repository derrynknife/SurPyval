"""The dynamic prediction of a joint model: one item's failure age given
its readings so far and that it is still running (#891)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    warn_outside_unit_interval,
)

if TYPE_CHECKING:  # pragma: no cover
    from .joint_model import JointModel

#: The ages ``qf`` first looks at: the age now plus these multiples of
#: the model's ``alpha``, then refined by safeguarded Newton steps.
_GRID = np.logspace(-6.0, 3.0, 91)
#: Newton steps of ``qf``, and the log-survival they stop within
#: (relative, with a floor at the survival's own rounding).
_STEPS = 30
_QF_TOL = 1e-10
_QF_FLOOR = 1e-14


class DynamicPrediction:
    """
    One item's failure-age distribution given its covariates, its marker
    readings so far and that it was still running at age ``age``: the
    dynamic prediction of a joint model, which :meth:`JointModel.predict
    <surpyval.univariate.joint.joint_model.JointModel.predict>` returns.

    The item's path parameters are averaged over their distribution given
    its readings and its survival to ``age`` (adaptive Gauss-Hermite
    quadrature at the mode), so ``sf(x)`` is ``P(T > x | T > age,
    readings)``: 1 up to ``age``, then falling. The functions take ages
    (since new, as the model's), not times from now; the remaining life
    is :meth:`remaining_life`. The estimates are at the fitted parameters;
    :meth:`cb` adds their uncertainty.

    Attributes
    ----------
    model : JointModel
        The model it was made from.
    age : float
        The age the item was last seen running.
    markers : dict
        The readings it is conditioned on, ``{age: value}``.
    covariates : numpy.ndarray
        The item's covariates, in the order of the model's
        ``feature_names``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
    >>> rng = np.random.default_rng(3)
    >>> items = []
    >>> for k in range(60):
    ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
    ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
    ...     end = min(life, 120.0)
    ...     ages = np.arange(0.0, end, 10.0)
    ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
    ...     outcome = "observed" if life < 120 else "right_censored"
    ...     items.append(
    ...         ItemLifetime(k, **{outcome: end},
    ...                      markers=dict(zip(ages, y)))
    ...     )
    >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
    >>> pred = model.predict(markers={0: 2.1, 20: 2.5, 40: 3.0}, given=40)
    >>> float(pred.sf(40))
    1.0
    >>> life = pred.remaining_life([0.1, 0.5, 0.9])
    >>> bool(0 < life[0] < life[1] < life[2])
    True
    """

    def __init__(
        self, model: JointModel, Z: Any, markers: Any, given: Any
    ) -> None:
        z, ages, values, u = model._question(Z, markers, given)
        self.model = model
        self._args = (Z, markers, given)
        self.covariates = z
        self.age = float(u)
        self.markers = dict(zip(ages.tolist(), values.tolist()))
        self._ages, self._values = ages, values
        self._missing = not bool(np.all(np.isfinite(z)))

    # -- the failure age ---------------------------------------------------

    def _log_sf_hf(self, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """``log sf`` and ``log hf`` at the flat ages ``x``: 0 and
        ``-inf`` up to ``age``, ``nan`` for ``nan`` (and every age where
        a covariate is missing)."""
        x = np.asarray(x, dtype=float)
        log_sf = np.where(np.isnan(x), np.nan, 0.0)
        log_hf = np.where(np.isnan(x), np.nan, -np.inf)
        if self._missing:
            return np.full(x.shape, np.nan), np.full(x.shape, np.nan)
        after = np.isfinite(x) & (x > self.age)
        # an infinite age: survival 0 there
        log_sf = np.where(np.isposinf(x), -np.inf, log_sf)
        if after.any():
            s, h = self.model._conditional(
                x[after], self.covariates, self._ages, self._values, self.age
            )
            log_sf[after], log_hf[after] = s, h
        return log_sf, log_hf

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """``P(T > x | T > age, readings)``: the probability the item is
        still running at age ``x`` (1 up to ``age``)."""
        return np.exp(self._log_sf_hf(np.asarray(x, dtype=float))[0])

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """The probability the item has failed by age ``x``, given it was
        running at ``age``: ``1 - sf(x)`` (0 up to ``age``)."""
        return -np.expm1(self._log_sf_hf(np.asarray(x, dtype=float))[0])

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The cumulative hazard from ``age`` to ``x``, ``-log sf(x)``."""
        return 0.0 - self._log_sf_hf(np.asarray(x, dtype=float))[0]

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The item's hazard at age ``x`` given its readings and that it
        reached ``x``: the hazard at ``x`` averaged over its path
        parameters' distribution given the readings and survival to
        ``x`` (0 up to ``age``, where the item is known to be running)."""
        return np.exp(self._log_sf_hf(np.asarray(x, dtype=float))[1])

    @keeps_query_shape
    def df(self, x: npt.ArrayLike) -> npt.NDArray:
        """The density of the failure age at ``x`` given the readings and
        survival to ``age``: ``hf(x) sf(x)`` (0 up to ``age``)."""
        log_sf, log_hf = self._log_sf_hf(np.asarray(x, dtype=float))
        return np.exp(log_sf + log_hf)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        """The age by which the item has failed with probability ``p``,
        given it was running at ``age``: ``age`` for ``p = 0``, ``inf``
        for 1 and where the survival never falls to ``1 - p`` (a path
        that falls away can leave a chance of never failing), ``nan``
        with a warning outside [0, 1]."""
        p = np.asarray(p, dtype=float)
        p = np.where(warn_outside_unit_interval(p), np.nan, p)
        out = np.full(p.shape, np.nan)
        if self._missing:
            return out
        ok = np.isfinite(p)
        out[ok & (p <= 0)] = self.age
        out[ok & (p >= 1)] = np.inf
        inner = np.flatnonzero(ok & (p > 0) & (p < 1))
        if inner.size:
            out[inner] = self._solve(np.log1p(-p[inner]))
        return out

    def _solve(self, target: npt.NDArray) -> npt.NDArray:
        """The ages at which ``log sf`` falls to ``target`` (each below
        0): bracketed on a grid of ages, then by Newton steps on ``log
        sf``, whose derivative is ``-hf``, kept inside the bracket
        (bisection where a step leaves it)."""
        u = self.age
        alpha = float(
            self.model.params[self.model.parameter_names.index("alpha")]
        )
        grid = u + alpha * _GRID
        log_sf, _ = self._log_sf_hf(grid)
        # the first grid age at or below each target
        below = log_sf[None, :] <= target[:, None]
        never = ~below.any(axis=1)
        k = np.argmax(below, axis=1)
        hi = grid[k]
        lo = np.where(k > 0, grid[np.maximum(k - 1, 0)], u)
        f_lo = np.where(k > 0, log_sf[np.maximum(k - 1, 0)], 0.0)
        f_hi = log_sf[k]
        # a start between them, by linear interpolation of log sf
        with np.errstate(all="ignore"):
            frac = (f_lo - target) / (f_lo - f_hi)
        frac = np.where(np.isfinite(frac), np.clip(frac, 0.0, 1.0), 0.5)
        t = lo + frac * (hi - lo)
        active = ~never
        for _ in range(_STEPS):
            if not active.any():
                break
            idx = np.flatnonzero(active)
            f, log_h = self._log_sf_hf(t[idx])
            gap = f - target[idx]
            done = np.abs(gap) <= np.maximum(
                _QF_TOL * np.abs(target[idx]), _QF_FLOOR
            )
            # narrow the bracket: log sf falls with age
            lo[idx] = np.where(gap > 0, t[idx], lo[idx])
            hi[idx] = np.where(gap > 0, hi[idx], t[idx])
            with np.errstate(all="ignore"):
                step = t[idx] + gap / np.exp(log_h)
            inside = np.isfinite(step) & (step > lo[idx]) & (step < hi[idx])
            t[idx] = np.where(
                done, t[idx], np.where(inside, step, 0.5 * (lo[idx] + hi[idx]))
            )
            active[idx[done]] = False
        return np.where(never, np.inf, t)

    def remaining_life(self, p: npt.ArrayLike = 0.5) -> Any:
        """
        The remaining life from ``age`` that the item outlives with
        probability ``1 - p``: ``qf(p) - age``. The default is the median
        remaining life; ``remaining_life([0.05, 0.5, 0.95])`` gives it
        with a 90% prediction interval (of the item's life, at the fitted
        parameters: :meth:`cb` bounds the survival for the parameters'
        uncertainty).
        """
        return self.qf(p) - self.age

    # -- uncertainty -------------------------------------------------------

    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Wald confidence bounds on the item's conditional ``sf``, ``ff`` or
        ``Hf`` at ages ``x``, from the parameters' covariance: the delta
        method on ``log(-log sf)``, as :meth:`JointModel.cb
        <surpyval.univariate.joint.joint_model.JointModel.cb>`, which it
        is (``Hf`` is ``-log`` of the bounds on ``sf``). Up to ``age``
        the bounds are the estimate.

        Parameters
        ----------
        x : array_like
            The ages.
        on : {'sf', 'ff', 'Hf'}, optional
            The function bounded (``'R'`` and ``'F'`` are ``'sf'`` and
            ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Default ``'two-sided'``, ``[lower, upper]`` on a last axis.
        """
        check_option("on", on, ("sf", "R", "ff", "F", "Hf"))
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        Z, markers, given = self._args
        if on != "Hf":
            return self.model.cb(
                x,
                Z,
                markers,
                given,
                on=on,
                alpha_ci=alpha_ci,
                bound=bound,
            )
        # an upper bound on Hf is a lower one on sf
        flipped = {
            "two-sided": "two-sided",
            "lower": "upper",
            "upper": "lower",
        }
        sf = self.model.cb(
            x,
            Z,
            markers,
            given,
            on="sf",
            alpha_ci=alpha_ci,
            bound=flipped[bound],
        )
        with np.errstate(divide="ignore"):
            out = 0.0 - np.log(sf)
        return out[..., ::-1] if bound == "two-sided" else out

    # -- the marker --------------------------------------------------------

    def marker(self, x: npt.ArrayLike) -> Any:
        """The mean of the item's true (noise-free) marker at ages ``x``,
        given its readings and survival to ``age``: its smoothed path up
        to ``age``, and beyond it the path projected (not conditioned on
        surviving to ``x``)."""
        x_arr = np.asarray(x, dtype=float)
        mean, _ = self._moments(x_arr)
        return mean.reshape(x_arr.shape)[()]

    def marker_interval(
        self, x: npt.ArrayLike, alpha_ci: float = 0.05
    ) -> npt.NDArray:
        """An equal-tailed ``1 - alpha_ci`` interval for the item's true
        marker at ages ``x``, mean plus or minus a normal quantile of its
        standard deviation, at the fitted parameters: ``[lower, upper]``
        on a last axis. It is the spread of the item's own path, not of a
        new reading (which adds the noise, ``sd_noise``), and for a path
        not linear in its parameters a normal approximation."""
        check_alpha_ci(alpha_ci)
        x_arr = np.asarray(x, dtype=float)
        mean, sd = self._moments(x_arr)
        q = norm.ppf(1 - alpha_ci / 2)
        out = np.stack([mean - q * sd, mean + q * sd], axis=-1)
        return out.reshape(x_arr.shape + (2,))

    def _moments(self, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        flat = np.atleast_1d(x).ravel()
        if self._missing:
            return np.full(flat.shape, np.nan), np.full(flat.shape, np.nan)
        safe = np.where(np.isfinite(flat), flat, 0.0)
        mean, sd = self.model._marker_moments(
            safe, self.covariates, self._ages, self._values, self.age
        )
        bad = ~np.isfinite(flat)
        return np.where(bad, np.nan, mean), np.where(bad, np.nan, sd)

    # -- draws -------------------------------------------------------------

    def random(
        self,
        size: int,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray:
        """
        Failure ages drawn from the prediction (each after ``age``; ``inf``
        for a draw beyond where the survival stops falling), by the
        quantile function of uniform draws.

        Parameters
        ----------
        size : int
            The number of draws.
        random_state : int or numpy.random.Generator, optional
            ``None`` (the default) draws from numpy's global generator,
            so ``np.random.seed`` reproduces it; an int or a generator
            gives a stream of its own.
        """
        rng = as_generator(random_state)
        return np.asarray(self.qf(rng.uniform(size=int(size))), dtype=float)

    def __repr__(self) -> str:
        median = float(self.remaining_life(0.5))
        return (
            "Dynamic prediction of a joint model\n"
            "===================================\n"
            f"Age now             : {self.age:.6g}\n"
            f"Readings            : {len(self.markers)}\n"
            f"Median life left    : {median:.6g}"
        )
