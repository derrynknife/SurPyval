"""Stochastic-process degradation models: Wiener and Gamma processes.

Unlike the general-path (pseudo-failure-time) approach in
:mod:`surpyval.degradation.degradation_analysis`, these model the degradation
*increments* directly as a stochastic process. Each has a first-passage
failure-time distribution derived analytically from the process, so the life
distribution comes straight from the fitted process rather than via noisy
pseudo failure times, and irregular measurement spacing is handled naturally
(every increment carries its own ``dt``).

Two complementary processes are provided, chosen by the physics of the
degradation:

* :class:`WienerProcess` -- Brownian motion with drift, ``W(t) = mu*t +
  sigma*B(t)``. Increments are Gaussian and may go up or down, so it suits
  *non-monotone* / noisy degradation signals (sensor drift, measurement noise).
  Its first passage to a threshold is a closed-form **Inverse Gaussian** law.
* :class:`GammaProcess` -- a sum of independent non-negative increments, so the
  path is *strictly monotone increasing*. It suits genuinely irreversible
  damage (wear, corrosion, crack growth, fatigue). Its first-passage
  distribution comes from the (regularised) incomplete gamma function.

Accelerated and step-stress tests
---------------------------------
Both fitters take an optional stress ``Z``, one row per measurement, which may
change *during* a unit's test (a step-stress profile) as well as between units.
Stress acts by speeding up the process clock -- the cumulative-exposure
principle: with the acceleration factor ``AF(z) = exp(gamma' (z - z_ref))``,
the process runs on the effective time ``tau(t) = integral of AF(z(s)) ds``, so
an increment over ``dt`` at stress ``z`` is an increment over ``AF(z) dt`` at
the reference stress ``z_ref``. For the Wiener process this is the time-scale
transformation of Whitmore and Schenkelberg (1997) -- drift and variance both
scale with ``AF`` -- and for the gamma process the shape accrues at
``alpha * AF(z)``. Because only the clock changes, the life under *any*
piecewise-constant stress profile is the reference-stress first-passage law
evaluated at ``tau(t)``: closed form for both processes. With ``z = 1/T`` the
acceleration factor is the Arrhenius relationship.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import gammaincc, gammaln
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)

from ._clock import StressClock, stress_row

__all__ = [
    "WienerProcess",
    "WienerProcessModel",
    "GammaProcess",
    "GammaProcessModel",
    "ProcessRUL",
]


def _increments(
    x: npt.ArrayLike, y: npt.ArrayLike, i: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Validate degradation data and return the pooled per-unit increments.

    Returns ``(dt, dy)`` arrays of the time and degradation increments between
    consecutive (time-ordered) measurements within each unit, pooled across
    units. Requires strictly increasing times within a unit.
    """
    dt, dy, _ = _increments_and_stress(x, y, i, None)
    return dt, dy


def _increments_and_stress(
    x: npt.ArrayLike, y: npt.ArrayLike, i: npt.ArrayLike, Z: Any
) -> tuple[npt.NDArray, npt.NDArray, "npt.NDArray | None"]:
    """
    :func:`_increments`, plus the stress in force over each increment.

    ``Z`` has one row per measurement (or is ``None``). The stress over the
    interval ``(x[j-1], x[j]]`` is the row of the measurement that ends it --
    the stress applied since the previous measurement -- so a step change
    taken just after a measurement is exact. Returns ``(dt, dy, z)`` with
    ``z`` one row per increment, or ``None`` without ``Z``.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    i = np.atleast_1d(np.asarray(i))
    if x.ndim != 1 or y.ndim != 1 or i.ndim != 1:
        raise ValueError("x, y, and i must be one dimensional")
    if not (len(x) == len(y) == len(i)):
        raise ValueError("x, y, and i must have the same length")
    if len(x) == 0:
        raise ValueError("x, y, and i must not be empty")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must contain only finite values")
    Z_arr = None
    if Z is not None:
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.ndim != 2 or len(Z_arr) != len(x):
            raise ValueError(
                "Z must have one row per measurement (same length as x, y "
                "and i); got shape {} for {} measurements".format(
                    np.shape(Z), len(x)
                )
            )
        if Z_arr.shape[1] == 0 or not np.isfinite(Z_arr).all():
            raise ValueError(
                "Z must have at least one column and only finite values"
            )

    dts, dys, zs = [], [], []
    for unit in np.unique(i):
        mask = i == unit
        xu, yu = x[mask], y[mask]
        order = np.argsort(xu)
        xu, yu = xu[order], yu[order]
        if len(xu) < 2:
            continue
        if Z_arr is not None:
            zs.append(Z_arr[mask][order][1:])
        dt = np.diff(xu)
        dy = np.diff(yu)
        if np.any(dt <= 0):
            raise ValueError(
                "unit {!r} has repeated or non-increasing measurement "
                "times; times must be strictly increasing within a "
                "unit".format(unit)
            )
        dts.append(dt)
        dys.append(dy)

    if not dts:
        raise ValueError(
            "no unit has two or more measurements; at least one increment "
            "is required to fit a process model"
        )
    z_int = None if Z_arr is None else np.concatenate(zs)
    return np.concatenate(dts), np.concatenate(dys), z_int


def _stress_design(
    z_int: npt.NDArray, stress_ref: Any
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Centre and scale the per-increment stresses for the optimiser.

    Returns ``(s, z_ref, scale)`` with ``s = (z - z_ref) / scale``, so the
    fitted coefficient on the original scale is ``g / scale``. The reference
    stress defaults to the mean stress over the increments. Raises when the
    coefficients are not identifiable: with a single stress level (or a
    covariate that is constant, or a combination of the others) the stress
    effect is confounded with the reference-stress parameters.
    """
    q = z_int.shape[1]
    design = np.column_stack([np.ones(len(z_int)), z_int])
    if np.linalg.matrix_rank(design) < q + 1:
        raise ValueError(
            "the stress coefficients cannot be estimated: Z needs at least "
            "two distinct stress levels across the measurement intervals, "
            "and no covariate may be constant or a combination of the others"
        )
    if stress_ref is None:
        z_ref = z_int.mean(axis=0)
    else:
        z_ref = stress_row(stress_ref, q)
    scale = z_int.std(axis=0)
    return (z_int - z_ref) / scale, z_ref, scale


def _minimise(fun: Callable, x0: npt.NDArray) -> npt.NDArray:
    """BFGS, falling back to (and polishing with) Nelder-Mead."""
    best = minimize(fun, x0, method="BFGS")
    if not (best.success and np.isfinite(best.fun)):
        nm = minimize(
            fun,
            best.x if np.isfinite(best.fun) else x0,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
        )
        if np.isfinite(nm.fun) and (
            not np.isfinite(best.fun) or nm.fun <= best.fun
        ):
            best = nm
    if not np.isfinite(best.fun):
        raise ValueError(
            "the stress-dependent process fit did not converge to a finite "
            "likelihood"
        )
    return np.asarray(best.x, dtype=float)


class ProcessRUL:
    """Remaining-useful-life summary from a fitted process model.

    Attributes
    ----------
    rul : float
        Median remaining useful life from the current state.
    rul_interval : tuple of float
        Equal-tailed ``1 - alpha_ci`` interval for the remaining life.
    prob_already_failed : float
        ``1.0`` when the current degradation is at or beyond the threshold
        (the remaining life and its interval are then ``0``), else ``0.0``.
    alpha_ci : float
        The tail probability of ``rul_interval``.
    """

    def __init__(
        self,
        rul: float,
        rul_interval: tuple,
        prob_already_failed: float,
        alpha_ci: float,
    ) -> None:
        self.rul = rul
        self.rul_interval = rul_interval
        self.prob_already_failed = prob_already_failed
        self.alpha_ci = alpha_ci

    def __repr__(self) -> str:
        lo, hi = self.rul_interval
        return (
            "ProcessRUL(rul={:.4g}, interval=({:.4g}, {:.4g}), "
            "prob_already_failed={:.4g})".format(
                self.rul, lo, hi, self.prob_already_failed
            )
        )


# --------------------------------------------------------------------------
# Shared first-passage machinery
# --------------------------------------------------------------------------


class FirstPassageProcessModel(SerialisableMixin):
    """
    The machinery shared by the fitted process models.

    Both models reduce their failure-time distribution to one hook,
    ``_ff_distance(t, distance)`` -- the probability the process has
    crossed ``distance`` by time ``t``. Everything expressible in terms
    of that CDF lives here once: ``ff``/``sf``, the hazard identities,
    the bracket-and-``brentq`` quantile (each subclass supplies only its
    starting bracket via ``_quantile_hi0``), ``predict_rul`` (independent
    increments make the remaining passage over the residual distance a
    fresh copy of the same law), and the ``to_dict``/``from_dict`` pair,
    driven by ``param_names``. The density, the mean, sampling and the
    repr stay on the subclasses: those genuinely differ (closed form
    against numeric derivative, closed form against quadrature, Wald
    sampling against inverse-CDF).

    ``WienerProcessModel`` and ``GammaProcessModel`` carried all of this
    verbatim twice -- ``predict_rul``, ``qf`` and the wrappers were the
    largest within-file duplicates in the package.
    """

    # Subclasses set these classattrs for serialisation and messages.
    _model_tag: str
    _human_name: str

    param_names: list
    threshold: float
    #: Stress coefficients and the reference stress, for a model fitted
    #: with ``Z``; both ``None`` otherwise.
    gamma: "npt.NDArray | None"
    stress_ref: "npt.NDArray | None"

    def __init__(
        self,
        *params_then_threshold: float,
        gamma: Any = None,
        stress_ref: Any = None,
    ) -> None:
        # Subclasses define their own named-parameter __init__; this
        # signature exists so ``from_dict`` type-checks against the base.
        raise NotImplementedError

    def _init_stress(self, gamma: Any, stress_ref: Any) -> None:
        if (gamma is None) != (stress_ref is None):
            raise ValueError("gamma and stress_ref must be given together")
        if gamma is None:
            self.gamma = None
            self.stress_ref = None
            return
        self.gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
        self.stress_ref = np.atleast_1d(np.asarray(stress_ref, dtype=float))
        if self.gamma.shape != self.stress_ref.shape or self.gamma.ndim != 1:
            raise ValueError(
                "gamma and stress_ref must be one-dimensional and the same "
                "length"
            )

    def _ff_distance(self, t: npt.ArrayLike, distance: float) -> npt.NDArray:
        raise NotImplementedError

    def _quantile_hi0(self, distance: float) -> float:
        """Starting upper bracket for the quantile search."""
        raise NotImplementedError

    def _df0(self, t: npt.NDArray) -> npt.NDArray:
        """First-passage density at the reference stress (clock time)."""
        raise NotImplementedError

    def _mean0(self) -> float:
        """Mean first-passage time at the reference stress."""
        raise NotImplementedError

    def _random0(self, size: int, rng: np.random.Generator) -> npt.NDArray:
        """First-passage draws at the reference stress (clock time)."""
        raise NotImplementedError

    # -- stress -------------------------------------------------------------

    @property
    def is_accelerated(self) -> bool:
        """True for a model fitted with a stress ``Z``."""
        return self.gamma is not None

    def acceleration_factor(self, Z: Any) -> float:
        """
        How much faster the process runs at stress ``Z`` than at the
        reference stress: ``exp(gamma' (z - stress_ref))``.

        A life at the reference stress divides by this to give the life
        at ``Z``, and a unit at ``Z`` accumulates degradation this many
        times faster.

        Parameters
        ----------
        Z : array like
            One stress row.
        """
        if self.gamma is None or self.stress_ref is None:
            raise ValueError(
                "This process model was fitted without stress, so it has no "
                "acceleration factor"
            )
        z = stress_row(Z, self.gamma.size)
        return float(np.exp(self.gamma @ (z - self.stress_ref)))

    def _clock(self, Z: Any) -> "StressClock | None":
        """The clock for stress ``Z``, validating the argument."""
        if not self.is_accelerated:
            if Z is not None:
                raise ValueError(
                    "This process model was fitted without stress; do not "
                    "pass Z."
                )
            return None
        if Z is None:
            raise ValueError(
                "This process model depends on stress; pass Z -- one stress "
                "row for a constant stress, or a StepSchedule for a stress "
                "profile."
            )
        assert self.gamma is not None
        return StressClock(self.acceleration_factor, self.gamma.size, Z)

    def _stress_repr(self) -> str:
        if self.gamma is None or self.stress_ref is None:
            return ""
        return (
            "Stress coefficients : {}\n"
            "Reference stress    : {}\n".format(
                np.array2string(self.gamma, precision=6),
                np.array2string(self.stress_ref, precision=6),
            )
        )

    def _mean_label(self) -> str:
        if self.is_accelerated:
            return "Mean life (ref.)    : {:.6g}".format(self._mean0())
        return "Mean time to failure: {:.6g}".format(self._mean0())

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise this fitted process model to a plain dict."""
        out: dict = {"model": self._model_tag}
        for name in self.param_names:
            out[name] = getattr(self, name)
        out["threshold"] = self.threshold
        if self.gamma is not None and self.stress_ref is not None:
            out["gamma"] = self.gamma.tolist()
            out["stress_ref"] = self.stress_ref.tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "FirstPassageProcessModel":
        """Rebuild a fitted process model from a :meth:`to_dict` dict."""
        require_model_tag(
            model_dict,
            cls._model_tag,
            "a {} model".format(cls._human_name),
        )
        return cls(
            *(model_dict[name] for name in cls.param_names),
            model_dict["threshold"],
            gamma=model_dict.get("gamma"),
            stress_ref=model_dict.get("stress_ref"),
        )

    # -- the failure-time distribution --------------------------------------

    def ff(self, t: npt.ArrayLike, Z: Any = None) -> "npt.NDArray | float":
        """
        Failure (CDF) of the first-passage time to the threshold.

        For a model fitted with stress, ``Z`` is required: one stress row for a
        constant stress, or a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` for
        a stress profile. The same applies to every method below.
        """
        clock = self._clock(Z)
        scalar = np.isscalar(t)
        tt = np.atleast_1d(t)
        if clock is not None:
            tt = clock.tau(np.asarray(tt, dtype=float))
        res = self._ff_distance(tt, self.threshold)
        return float(res[0]) if scalar else res

    def sf(self, t: npt.ArrayLike, Z: Any = None) -> "npt.NDArray | float":
        """Survival function of the first-passage time."""
        clock = self._clock(Z)
        scalar = np.isscalar(t)
        tt = np.atleast_1d(t)
        if clock is not None:
            tt = clock.tau(np.asarray(tt, dtype=float))
        res = 1.0 - self._ff_distance(tt, self.threshold)
        return float(res[0]) if scalar else res

    def df(self, t: npt.ArrayLike, Z: Any = None) -> "npt.NDArray | float":
        """
        Density of the first-passage time. Under a stress path it is the
        reference-stress density at the clock time ``tau(t)`` times the
        clock's rate, ``AF`` of the stress in force at ``t``.
        """
        clock = self._clock(Z)
        scalar = np.isscalar(t)
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        if clock is None:
            res = self._df0(tt)
        else:
            res = self._df0(clock.tau(tt)) * clock.rate_at(tt)
        return float(res[0]) if scalar else res

    def hf(self, t: npt.ArrayLike, Z: Any = None) -> "npt.NDArray | float":
        """Hazard function of the first-passage time."""
        return self.df(t, Z) / self.sf(t, Z)

    def Hf(self, t: npt.ArrayLike, Z: Any = None) -> "npt.NDArray | float":
        """Cumulative hazard of the first-passage time."""
        return -np.log(self.sf(t, Z))

    def qf(self, p: npt.ArrayLike, Z: Any = None) -> "npt.NDArray | float":
        """Quantile (inverse CDF) of the first-passage time."""
        clock = self._clock(Z)
        p = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.array([self._quantile(pi, self.threshold) for pi in p])
        if clock is not None:
            out = clock.inverse(out)
        return float(out[0]) if out.shape == (1,) else out

    def mean(self, Z: Any = None) -> float:
        """
        Mean time to failure. At a constant stress it is the
        reference-stress mean divided by the acceleration factor; under a
        stress profile it is the integral of the survival function.
        """
        clock = self._clock(Z)
        if clock is None:
            return self._mean0()
        if clock.rate is not None:
            return self._mean0() / clock.rate
        val, _ = quad(
            lambda t: float(np.ravel(self.sf(t, Z))[0]),
            0.0,
            np.inf,
            limit=200,
        )
        return val

    def random(
        self,
        size: int,
        random_state: "int | None" = None,
        Z: Any = None,
    ) -> npt.NDArray:
        """
        Draw first-passage (failure) times from the fitted model.

        Parameters
        ----------
        size : int
            Number of draws.
        random_state : int, optional
            Seed for reproducible draws.
        Z : array like or StepSchedule, optional
            The stress, for a model fitted with ``Z`` (required then);
            each reference-stress draw is carried to calendar time along
            its clock.
        """
        clock = self._clock(Z)
        rng = np.random.default_rng(random_state)
        draws = self._random0(size, rng)
        return draws if clock is None else clock.inverse(draws)

    def _quantile(self, p: float, distance: float) -> float:
        if not (0.0 < p < 1.0):
            return 0.0 if p <= 0.0 else np.inf
        # bracket from the subclass's starting scale and expand until it
        # contains the quantile
        hi = self._quantile_hi0(distance)
        while self._ff_distance(np.array([hi]), distance)[0] < p:
            hi *= 2.0
            if hi > 1e12:
                return np.inf
        return brentq(
            lambda t: self._ff_distance(np.array([t]), distance)[0] - p,
            1e-12,
            hi,
        )

    def predict_rul(
        self,
        current_degradation: float,
        alpha_ci: float = 0.05,
        Z: Any = None,
    ) -> ProcessRUL:
        """
        Remaining useful life given the current degradation level.

        Increments are independent for both processes, so the remaining
        first passage over the residual distance ``threshold -
        current_degradation`` follows the same law as a fresh process;
        its median and equal-tailed interval are returned.

        Parameters
        ----------
        current_degradation : float
            The unit's current degradation level.
        alpha_ci : float, optional
            Tail probability of the returned interval. Default ``0.05``.
        Z : array like or StepSchedule, optional
            For a model fitted with stress: the stress the unit will run at
            from now on -- one row for a constant stress, or a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
            whose time zero is *now*.

        Returns
        -------
        ProcessRUL
            The median remaining life and its equal-tailed interval.
        """
        clock = self._clock(Z)
        distance = self.threshold - float(current_degradation)
        if distance <= 0:
            return ProcessRUL(0.0, (0.0, 0.0), 1.0, alpha_ci)
        med = self._quantile(0.5, distance)
        lo = self._quantile(alpha_ci / 2.0, distance)
        hi = self._quantile(1.0 - alpha_ci / 2.0, distance)
        if clock is not None:
            med, lo, hi = (
                float(v) for v in clock.inverse(np.array([med, lo, hi]))
            )
        return ProcessRUL(med, (lo, hi), 0.0, alpha_ci)


# --------------------------------------------------------------------------
# Wiener process
# --------------------------------------------------------------------------


class WienerProcessModel(FirstPassageProcessModel):
    """
    A fitted Wiener-process degradation model, ``W(t) = mu*t + sigma*B(t)``.

    The first passage of the process to the failure ``threshold`` (from an
    assumed degradation of zero at ``t = 0``) is Inverse-Gaussian distributed
    with mean ``threshold / mu`` and shape ``threshold**2 / sigma**2``, and the
    failure-time methods below evaluate that distribution. A positive drift
    ``mu`` is required for a proper (non-defective) life distribution.

    Parameters
    ----------
    mu : float
        Fitted drift (mean degradation rate).
    sigma : float
        Fitted diffusion (volatility) coefficient.
    threshold : float
        The degradation level defining failure.
    gamma, stress_ref : array like, optional
        For a model fitted with stress ``Z``: the stress coefficients and
        the reference stress at which ``mu`` and ``sigma`` apply. At stress
        ``z`` the process clock runs ``exp(gamma' (z - stress_ref))`` times
        faster, scaling both the drift and the variance per unit time.
    """

    _model_tag = "WienerProcessModel"
    _human_name = "Wiener-process"
    param_names = ["mu", "sigma"]

    def __init__(
        self,
        mu: float,
        sigma: float,
        threshold: float,
        gamma: Any = None,
        stress_ref: Any = None,
    ) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.threshold = float(threshold)
        self.params = np.array([self.mu, self.sigma])
        self._init_stress(gamma, stress_ref)

    def _ig(self, distance: float) -> tuple[float, float]:
        # Inverse-Gaussian (mean nu, shape lam) parameters for first passage
        # over ``distance`` at drift mu / diffusion sigma.
        nu = distance / self.mu
        lam = distance**2 / self.sigma**2
        return nu, lam

    def _ff_distance(self, t: npt.ArrayLike, distance: float) -> npt.NDArray:
        # Inverse-Gaussian first-passage CDF over ``distance``.
        t = np.asarray(t, dtype=float)
        nu, lam = self._ig(distance)
        out = np.zeros_like(t, dtype=float)
        pos = t > 0
        tp = t[pos]
        root = np.sqrt(lam / tp)
        cdf = norm.cdf(root * (tp / nu - 1.0)) + np.exp(
            2.0 * lam / nu
        ) * norm.cdf(-root * (tp / nu + 1.0))
        out[pos] = cdf
        return out

    def _df0(self, t: npt.NDArray) -> npt.NDArray:
        # Density of the first-passage (Inverse-Gaussian) time.
        nu, lam = self._ig(self.threshold)
        out = np.zeros_like(t)
        pos = t > 0
        tp = t[pos]
        out[pos] = np.sqrt(lam / (2.0 * np.pi * tp**3)) * np.exp(
            -lam * (tp - nu) ** 2 / (2.0 * nu**2 * tp)
        )
        return out

    def _mean0(self) -> float:
        # Mean time to failure, ``threshold / mu``.
        return self.threshold / self.mu

    def _quantile_hi0(self, distance: float) -> float:
        # bracket around the first-passage mean
        return distance / self.mu

    def _random0(self, size: int, rng: np.random.Generator) -> npt.NDArray:
        nu, lam = self._ig(self.threshold)
        return rng.wald(nu, lam, size=size)

    def __repr__(self) -> str:
        return (
            "Wiener Process Degradation Model\n"
            "================================\n"
            "Drift (mu)          : {:.6g}\n"
            "Diffusion (sigma)   : {:.6g}\n"
            "Threshold           : {:.6g}\n"
            "{}{}".format(
                self.mu,
                self.sigma,
                self.threshold,
                self._stress_repr(),
                self._mean_label(),
            )
        )


class WienerProcess:
    """Fitter for the Wiener-process degradation model (see
    :class:`WienerProcessModel`)."""

    @classmethod
    def fit(
        cls,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        i: npt.ArrayLike,
        threshold: float,
        Z: npt.ArrayLike | None = None,
        stress_ref: npt.ArrayLike | None = None,
    ) -> "WienerProcessModel":
        """
        Fit a Wiener-process degradation model by maximum likelihood.

        Parameters
        ----------
        x : array_like
            Measurement times.
        y : array_like
            Degradation measurements.
        i : array_like
            Unit identifier for each measurement.
        threshold : float
            The degradation level defining failure.
        Z : array_like, optional
            Stress covariates, one row per measurement, for an accelerated
            or step-stress test. The stress may differ between units and
            change during a unit's test: the stress on a measurement is the
            stress applied since the previous one. Stress speeds up the
            process clock by ``exp(gamma' (z - stress_ref))``, scaling the
            drift and the variance per unit time together (the time-scale
            transformation of Whitmore and Schenkelberg, 1997); ``gamma``
            is estimated with ``mu`` and ``sigma`` by maximum likelihood.
            At least two distinct stress levels are needed.
        stress_ref : array_like, optional
            The reference stress at which the fitted ``mu`` and ``sigma``
            apply. Defaults to the mean stress over the measurement
            intervals; pass the use conditions to read the model at them.

        Returns
        -------
        WienerProcessModel
            The fitted model, whose life-distribution methods (``sf``,
            ``ff``, ``mean``, ...) give the first-passage time to
            ``threshold``.

        Examples
        --------
        Five units whose degradation drifts upwards at 0.5 per unit time
        with Brownian noise:

        >>> import numpy as np
        >>> from surpyval.degradation import WienerProcess
        >>> rng = np.random.default_rng(1)
        >>> t = np.tile(np.arange(0, 110, 10.0), 5)  # 5 units, 11 readings
        >>> i = np.repeat(np.arange(5), 11)
        >>> steps = rng.normal(0.5 * 10, 1.0 * np.sqrt(10), size=(5, 10))
        >>> y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
        >>> model = WienerProcess.fit(t, y, i, threshold=100)
        >>> model
        Wiener Process Degradation Model
        ================================
        Drift (mu)          : 0.488591
        Diffusion (sigma)   : 0.88113
        Threshold           : 100
        Mean time to failure: 204.67
        >>> model.sf([150, 200]).round(4)
        array([0.9922, 0.548 ])
        """
        if Z is None:
            if stress_ref is not None:
                raise ValueError("stress_ref is only meaningful with Z")
            dt, dy = _increments(x, y, i)
            # Increments dy | dt ~ Normal(mu*dt, sigma**2 * dt), independent.
            # Closed-form MLE:
            mu = dy.sum() / dt.sum()
            sigma2 = np.mean((dy - mu * dt) ** 2 / dt)
            sigma = np.sqrt(sigma2)
            cls._check_drift(mu)
            return WienerProcessModel(mu, sigma, threshold)

        dt, dy, z_int = _increments_and_stress(x, y, i, Z)
        assert z_int is not None
        s, z_ref, scale = _stress_design(z_int, stress_ref)

        def profile(g: npt.NDArray) -> tuple[float, float, float]:
            # dy ~ Normal(mu * dtau, sigma**2 * dtau) with dtau = AF * dt;
            # mu and sigma**2 have closed forms given the clock.
            dtau = dt * np.exp(s @ g)
            mu = dy.sum() / dtau.sum()
            sigma2 = np.mean((dy - mu * dtau) ** 2 / dtau)
            neg = 0.5 * (np.sum(np.log(dtau)) + len(dy) * np.log(sigma2))
            return float(neg), float(mu), float(sigma2)

        g = _minimise(lambda v: profile(v)[0], np.zeros(z_int.shape[1]))
        _, mu, sigma2 = profile(g)
        cls._check_drift(mu)
        return WienerProcessModel(
            mu,
            np.sqrt(sigma2),
            threshold,
            gamma=g / scale,
            stress_ref=z_ref,
        )

    @staticmethod
    def _check_drift(mu: float) -> None:
        if mu <= 0:
            raise ValueError(
                "fitted drift mu = {:.4g} is not positive, so the process "
                "does not reliably reach the threshold and the first-passage "
                "life distribution is defective. Check the sign of the "
                "degradation / threshold, or use a monotone model.".format(mu)
            )


# --------------------------------------------------------------------------
# Gamma process
# --------------------------------------------------------------------------


class GammaProcessModel(FirstPassageProcessModel):
    """
    A fitted Gamma-process degradation model with stationary independent
    increments: over an interval ``dt`` the degradation increment is
    ``Gamma(shape = alpha * dt, rate = beta)``. The path is monotone
    increasing.

    Because the path is monotone, the first passage to the failure
    ``threshold`` has failure CDF ``P(W(t) >= threshold)``, evaluated with the
    regularised upper incomplete gamma function.

    Parameters
    ----------
    alpha : float
        Fitted shape rate (shape accrues as ``alpha * t``).
    beta : float
        Fitted rate parameter of the increments.
    threshold : float
        The degradation level defining failure.
    gamma, stress_ref : array like, optional
        For a model fitted with stress ``Z``: the stress coefficients and
        the reference stress at which ``alpha`` applies. At stress ``z``
        the shape accrues at ``alpha * exp(gamma' (z - stress_ref))``.
    """

    _model_tag = "GammaProcessModel"
    _human_name = "gamma-process"
    param_names = ["alpha", "beta"]

    def __init__(
        self,
        alpha: float,
        beta: float,
        threshold: float,
        gamma: Any = None,
        stress_ref: Any = None,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.params = np.array([self.alpha, self.beta])
        self._init_stress(gamma, stress_ref)

    def _ff_distance(self, t: npt.ArrayLike, distance: float) -> npt.NDArray:
        # P(T <= t) = P(W(t) >= distance) with W(t) ~ Gamma(alpha t, beta).
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t, dtype=float)
        pos = t > 0
        out[pos] = gammaincc(self.alpha * t[pos], self.beta * distance)
        return out

    def _df0(self, t: npt.NDArray) -> npt.NDArray:
        # Density of the first-passage time (numeric derivative of ``ff``).
        h = 1e-6
        out = np.zeros_like(t)
        pos = t > 0
        tp = t[pos]
        step = np.maximum(h, tp * h)
        f_hi = self._ff_distance(tp + step, self.threshold)
        f_lo = self._ff_distance(np.maximum(tp - step, 1e-12), self.threshold)
        out[pos] = (f_hi - f_lo) / ((tp + step) - np.maximum(tp - step, 1e-12))
        out = np.clip(out, 0.0, None)
        return out

    def _mean0(self) -> float:
        # Mean time to failure, the integral of the survival function.
        val, _ = quad(
            lambda t: self._sf_distance_scalar(t, self.threshold),
            0.0,
            np.inf,
            limit=200,
        )
        return val

    def _sf_distance_scalar(self, t: float, distance: float) -> float:
        if t <= 0:
            return 1.0
        return 1.0 - gammaincc(self.alpha * t, self.beta * distance)

    def _quantile_hi0(self, distance: float) -> float:
        # rough starting scale from the mean increment rate
        rate = self.alpha / self.beta  # mean degradation per unit time
        return max(distance / rate, 1.0)

    def _random0(self, size: int, rng: np.random.Generator) -> npt.NDArray:
        # Inverse-CDF sampling.
        u = rng.uniform(size=size)
        return np.array([self._quantile(ui, self.threshold) for ui in u])

    def __repr__(self) -> str:
        return (
            "Gamma Process Degradation Model\n"
            "===============================\n"
            "Shape rate (alpha)  : {:.6g}\n"
            "Rate (beta)         : {:.6g}\n"
            "Threshold           : {:.6g}\n"
            "{}{}".format(
                self.alpha,
                self.beta,
                self.threshold,
                self._stress_repr(),
                self._mean_label(),
            )
        )


class GammaProcess:
    """Fitter for the Gamma-process degradation model (see
    :class:`GammaProcessModel`)."""

    @classmethod
    def fit(
        cls,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        i: npt.ArrayLike,
        threshold: float,
        Z: npt.ArrayLike | None = None,
        stress_ref: npt.ArrayLike | None = None,
    ) -> "GammaProcessModel":
        """
        Fit a Gamma-process degradation model by maximum likelihood.

        The degradation must be monotone increasing (all increments
        non-negative); an increment that decreases raises an error pointing to
        the Wiener model for non-monotone signals.

        Parameters
        ----------
        x : array_like
            Measurement times.
        y : array_like
            Degradation measurements.
        i : array_like
            Unit identifier for each measurement.
        threshold : float
            The degradation level defining failure.
        Z : array_like, optional
            Stress covariates, one row per measurement, for an accelerated
            or step-stress test. The stress may differ between units and
            change during a unit's test: the stress on a measurement is the
            stress applied since the previous one. Stress speeds up the
            process clock, so the shape accrues at ``alpha * exp(gamma'
            (z - stress_ref))`` per unit time with ``beta`` unchanged;
            ``gamma`` is estimated with ``alpha`` and ``beta`` by maximum
            likelihood. At least two distinct stress levels are needed.
        stress_ref : array_like, optional
            The reference stress at which the fitted ``alpha`` applies.
            Defaults to the mean stress over the measurement intervals.

        Returns
        -------
        GammaProcessModel
            The fitted model, whose life-distribution methods (``sf``,
            ``ff``, ``mean``, ...) give the first-passage time to
            ``threshold``.

        Examples
        --------
        Five units whose wear accumulates in non-negative gamma-distributed
        increments (mean 0.5 per unit time):

        >>> import numpy as np
        >>> from surpyval.degradation import GammaProcess
        >>> rng = np.random.default_rng(1)
        >>> t = np.tile(np.arange(0, 110, 10.0), 5)  # 5 units, 11 readings
        >>> i = np.repeat(np.arange(5), 11)
        >>> steps = rng.gamma(shape=2.0 * 10, scale=0.25, size=(5, 10))
        >>> y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
        >>> model = GammaProcess.fit(t, y, i, threshold=100)
        >>> model
        Gamma Process Degradation Model
        ===============================
        Shape rate (alpha)  : 2.61045
        Rate (beta)         : 5.45561
        Threshold           : 100
        Mean time to failure: 209.183
        >>> model.sf([150, 200]).round(4)
        array([1.    , 0.8477])
        """
        if Z is None:
            if stress_ref is not None:
                raise ValueError("stress_ref is only meaningful with Z")
            dt, dy = _increments(x, y, i)
            cls._check_monotone(dy)
            alpha, beta = cls._profile_fit(dt, dy)
            return GammaProcessModel(alpha, beta, threshold)

        dt, dy, z_int = _increments_and_stress(x, y, i, Z)
        assert z_int is not None
        cls._check_monotone(dy)
        s, z_ref, scale = _stress_design(z_int, stress_ref)
        dy = np.where(dy <= 0, 1e-12, dy)
        sum_dy = dy.sum()
        log_dy = np.log(dy)

        def neg_ll(v: npt.NDArray) -> float:
            # v = [log alpha, g]; beta profiled out given the clock
            alpha = np.exp(v[0])
            dtau = dt * np.exp(s @ v[1:])
            beta = alpha * dtau.sum() / sum_dy
            k = alpha * dtau
            ll = np.sum(
                k * np.log(beta) + (k - 1.0) * log_dy - beta * dy - gammaln(k)
            )
            return float(-ll)

        # the stress-free fit is the starting point (g = 0)
        alpha0, _ = cls._profile_fit(dt, dy)
        v0 = np.concatenate([[np.log(alpha0)], np.zeros(z_int.shape[1])])
        v = _minimise(neg_ll, v0)
        alpha = float(np.exp(v[0]))
        g = v[1:]
        beta = alpha * float((dt * np.exp(s @ g)).sum()) / sum_dy
        return GammaProcessModel(
            alpha, beta, threshold, gamma=g / scale, stress_ref=z_ref
        )

    @staticmethod
    def _check_monotone(dy: npt.NDArray) -> None:
        if np.any(dy < 0):
            raise ValueError(
                "the degradation decreases over at least one interval, but a "
                "Gamma process is monotone increasing. Use WienerProcess for "
                "non-monotone / noisy signals."
            )

    @staticmethod
    def _profile_fit(dt: npt.NDArray, dy: npt.NDArray) -> tuple[float, float]:
        """The stationary (stress-free) fit: ``(alpha, beta)``."""
        # Guard against exact-zero increments (log 0) by nudging.
        dy = np.where(dy <= 0, 1e-12, dy)

        sum_dt = dt.sum()
        sum_dy = dy.sum()
        log_dy = np.log(dy)

        def neg_ll(alpha: float) -> float:
            # Profile out beta: d/dbeta -> beta = alpha * sum_dt / sum_dy.
            beta = alpha * sum_dt / sum_dy
            k = alpha * dt
            ll = np.sum(
                k * np.log(beta) + (k - 1.0) * log_dy - beta * dy - gammaln(k)
            )
            return -ll

        res = minimize_scalar(neg_ll, bounds=(1e-6, 1e6), method="bounded")
        alpha = float(res.x)
        beta = alpha * sum_dt / sum_dy
        return alpha, beta
