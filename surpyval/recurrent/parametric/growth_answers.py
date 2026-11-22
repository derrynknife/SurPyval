"""The questions asked of a fitted reliability growth model: how many
faults are left, how reliable the system is now, and how much more
testing reaches a target. They are methods of every fitted Poisson
process (``ParametricRecurrenceModel``), written here once."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from surpyval.recurrent.inference import check_alpha_ci
from surpyval.utils.linalg import (
    bound_signs,
    delta_method_se,
    log_transformed_cb,
)
from surpyval.utils.validation import BOUNDS, check_option

# The bound on a falling function of f that a bound on f gives.
_OPPOSITE = {"two-sided": "two-sided", "lower": "upper", "upper": "lower"}

#: How far past a million times the model's own time scale a search for
#: a target goes before saying it is never reached: steps of 4, so to a
#: million million times it. (Further out, a difference of expected
#: counts is lost to rounding.)
_FAR_STEPS = 10


class NeverReached(ValueError):
    """A target the fitted curve does not come down to."""


def _rises(before: float, after: float) -> bool:
    """Whether a value has risen over a step of 4 in time by more than
    the rounding of a difference of counts far out (a part in a
    thousand)."""
    return after > before * 1.001 and after > 0


def last_time_above(func: Callable, level: float, scale: float) -> float:
    """The last time at which ``func`` (an intensity, or the expected
    failures in a mission starting then) is above ``level``: the time
    from which it stays at or below it. 0 where it never is above.

    ``func`` is looked at on a grid of times around ``scale`` (from a
    millionth of it to a million times it, 20 a decade) and then further
    out in steps of 4 until it is at or below the level; the crossing is
    solved inside the last grid step above the level (a value that is
    ``nan`` counts as above it). Raises
    ``NeverReached`` where the function is still above the level, or
    still rising, at the end of the search.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent.parametric.growth_answers import (
    ...     last_time_above,
    ... )
    >>> round(last_time_above(lambda t: 8.0 * np.exp(-0.5 * t), 1.0, 1.0), 6)
    4.158883
    """

    def at(t: np.ndarray) -> np.ndarray:
        # (a value that cannot be worked out is taken as above the level)
        with np.errstate(all="ignore"):
            values = np.asarray(func(t), dtype=float)
        return np.where(np.isnan(values), np.inf, values)

    grid = scale * np.geomspace(1e-6, 1e6, 241)
    times, seen = list(grid), list(at(grid))
    steps = 0
    while not seen[-1] <= level:
        if steps == _FAR_STEPS:
            raise NeverReached(
                "rising" if _rises(seen[-2], seen[-1]) else "still above"
            )
        steps += 1
        times.append(times[-1] * 4.0)
        seen.append(float(at(np.array([times[-1]]))[0]))
    beyond = float(at(np.array([times[-1] * 4.0]))[0])
    if _rises(seen[-1], beyond):
        raise NeverReached("rising")
    above = np.flatnonzero(np.asarray(seen) > level)
    if above.size == 0:
        return 0.0
    k = int(above[-1])

    def gap(t: float) -> float:
        return float(at(np.array([t]))[0]) - level

    return float(brentq(gap, times[k], times[k + 1], xtol=1e-300, rtol=1e-13))


class GrowthAnswersMixin:
    """Remaining faults, mission reliability and the testing needed to
    reach a target, for a fitted Poisson process."""

    # What the fitted model gives this mixin (declared for the type
    # checker).
    dist: Any
    params: np.ndarray
    _mle: np.ndarray
    cif: Callable
    iif: Callable
    iif_cb: Callable
    inv_cif: Callable
    covariance: Callable
    _check_inference: Callable
    _ages: Callable
    _ahead: Callable

    # -- remaining faults -------------------------------------------------

    def _total(self, params: Any = None) -> float:
        """The expected number of failures over all time."""
        values = self.params if params is None else params
        with np.errstate(all="ignore"):
            return float(self.dist.cif(np.inf, *values))

    def _require_finite_total(self, what: str) -> None:
        total = self._total()
        if not np.isfinite(total):
            raise ValueError(
                "{what} needs a model with a finite number of faults, "
                "and the fitted {name} model expects failures without "
                "limit (its cumulative intensity keeps growing), so it "
                "has no number of faults left. Fit a finite-fault model "
                "(GoelOkumoto, YamadaDelayedS or InflectionS), or ask "
                "this one for the failures expected in a stretch of "
                "time: cif(t2) - cif(t1).".format(
                    what=what, name=self.dist.name
                )
            )

    def remaining_faults(self, x: ArrayLike) -> np.ndarray:
        """
        The faults expected to be left after testing to ``x``: the
        expected number in all less those expected by ``x``,
        :math:`\\Lambda(\\infty) - \\Lambda(x)` (``a - cif(x)`` for the
        software reliability growth models with a total ``a``). With
        several systems it is the faults left in one.

        Only a process with a finite number of faults has an answer
        (``GoelOkumoto``, ``YamadaDelayedS``, ``InflectionS``, and a
        ``CoxLewis`` with a falling intensity); for one whose failures
        go on without limit (``MusaOkumoto``, ``CrowAMSAA``, ``Duane``,
        ``HPP``) it raises a ``ValueError`` saying so.

        Parameters
        ----------
        x : array_like
            Times tested to (usually the end of the test).

        Returns
        -------
        numpy.ndarray
            The expected faults left, of the shape of ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import GoelOkumoto
        >>> model = GoelOkumoto.from_params([100, 0.05])
        >>> model.remaining_faults([0, 20, 60]).round(3)
        array([100.   ,  36.788,   4.979])
        """
        self._require_finite_total("remaining_faults")
        x = np.asarray(x, dtype=float)
        return self._total() - np.asarray(self.cif(x), dtype=float)

    def remaining_faults_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on :meth:`remaining_faults` at ``x``, by the
        delta method on the log of the faults left (so they cannot go
        below 0), from the parameter covariance: the construction of
        :meth:`cif_cb`.

        The bounds are on the *expected* number left, as the model's
        other bounds are on expected counts. They are Wald bounds, and
        with few failures, or little of the curve's levelling off seen,
        the upper one is too low: the likelihood of a growth model falls
        slowly as the number of faults grows. See "When not to trust
        the answers" in the Software Reliability Growth guide.

        Parameters
        ----------
        x : array_like
            Times tested to.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are an ``(len(x), 2)`` array of ``[lower,
            upper]``; a one-sided bound has the shape of ``x``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GoelOkumoto
        >>> weeks = np.arange(8.0)
        >>> x = np.column_stack([weeks, weeks + 1])
        >>> model = GoelOkumoto.fit(
        ...     x, c=[2] * 8, n=[12, 9, 8, 5, 4, 4, 2, 1]
        ... )
        >>> model.remaining_faults(8.0).round(2)
        np.float64(5.02)
        >>> model.remaining_faults_cb(8.0).round(2)
        array([ 1.34, 18.74])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._require_finite_total("remaining_faults_cb")
        self._check_inference()
        shape = np.shape(x)
        flat = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)

        def left(params: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                return self.dist.cif(np.inf, *params) - self.dist.cif(
                    flat, *params
                )

        se = delta_method_se(left, self._mle, self.covariance(), True)
        cb = log_transformed_cb(left(self._mle), se, alpha_ci, bound)
        return cb.reshape(shape + cb.shape[1:])

    # -- the events of a stretch --------------------------------------------

    def expected_events(self, start: ArrayLike, end: ArrayLike) -> np.ndarray:
        """
        The events expected between the times ``start`` and ``end``,
        ``cif(end) - cif(start)``: the failures of a further stretch of
        testing or use, or the defects of a later phase.

        Parameters
        ----------
        start, end : array_like
            The stretch's ends (each ``end`` at or after its ``start``),
            of shapes that broadcast.

        Examples
        --------
        >>> from surpyval.recurrent import GoelOkumoto
        >>> model = GoelOkumoto.from_params([100, 0.05])
        >>> model.expected_events([0, 20], [20, 40]).round(2)
        array([63.21, 23.25])
        """
        lo, hi = self._stretch(start, end)
        return np.asarray(self.cif(hi), dtype=float) - np.asarray(
            self.cif(lo), dtype=float
        )

    @staticmethod
    def _stretch(start: ArrayLike, end: ArrayLike) -> tuple:
        lo = np.asarray(start, dtype=float)
        hi = np.asarray(end, dtype=float)
        lo, hi = np.broadcast_arrays(lo, hi)
        if np.any(hi < lo):
            raise ValueError(
                "end must be at or after start for each stretch; got "
                "start {} with end {}".format(lo[hi < lo][:3], hi[hi < lo][:3])
            )
        return lo, hi

    def expected_events_cb(
        self,
        start: ArrayLike,
        end: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on :meth:`expected_events`, by the delta
        method on its log (so they cannot go below 0), from the
        parameter covariance. They are bounds on the *expected* number,
        not a prediction interval for the count that will be seen, which
        also has its own Poisson scatter.

        Parameters
        ----------
        start, end : array_like
            The stretch's ends.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds have a last axis of ``[lower, upper]``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GoelOkumoto
        >>> weeks = np.arange(8.0)
        >>> x = np.column_stack([weeks, weeks + 1])
        >>> model = GoelOkumoto.fit(
        ...     x, c=[2] * 8, n=[12, 9, 8, 5, 4, 4, 2, 1]
        ... )
        >>> model.expected_events(8, 12).round(2)
        np.float64(3.43)
        >>> model.expected_events_cb(8, 12).round(2)
        array([1.19, 9.88])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_inference()
        lo, hi = self._stretch(start, end)
        cb = self._gained_cb(
            lo.reshape(-1), (hi - lo).reshape(-1), alpha_ci, bound
        )
        return cb.reshape(lo.shape + cb.shape[1:])

    # -- reliability over a mission ----------------------------------------

    def _log_gained(self, start: np.ndarray, x: np.ndarray) -> Callable:
        """The log of the failures expected between ``start`` and
        ``start + x`` (flat arrays of one length), as a function of the
        parameters.

        It is the log of a difference of two counts where that is at
        least half the later count (so it keeps its digits). Further out,
        where a curve that levels off makes the two counts nearly equal,
        it is the log of the intensity's integral over the mission
        instead, summed in logs: an 8-point
        Gauss-Legendre rule on each of 41 pieces that halve towards the
        mission's start, where a falling intensity has its weight."""
        with np.errstate(all="ignore"):
            later = np.asarray(
                self.dist.cif(start + x, *self._mle), dtype=float
            )
            kept = later - np.asarray(
                self.dist.cif(start, *self._mle), dtype=float
            )
        # (a difference of two counts loses as many digits as the counts
        # are larger than it: 1e-6 of them lost six, which the delta
        # method's differences then magnified to 1e-8 of a bound)
        direct = kept > 0.5 * later
        far = ~direct
        edges = np.concatenate([[0.0], 0.5 ** np.arange(40, -1, -1.0)])
        node, weight = np.polynomial.legendre.leggauss(8)
        lo, hi = edges[:-1, None], edges[1:, None]
        share = (lo + (hi - lo) * (node + 1.0) / 2.0).reshape(-1)
        log_w = np.log(((hi - lo) * weight / 2.0).reshape(-1))
        times = start[far, None] + x[far, None] * share
        log_x = np.log(x[far])

        def log_gained(params: np.ndarray) -> np.ndarray:
            out = np.empty(start.size)
            with np.errstate(all="ignore"):
                out[direct] = np.log(
                    self.dist.cif((start + x)[direct], *params)
                    - self.dist.cif(start[direct], *params)
                )
                if far.any():
                    terms = (
                        np.asarray(
                            self.dist.log_iif(times, *params), dtype=float
                        )
                        + log_w
                    )
                    out[far] = logsumexp(terms, axis=1) + log_x
            return out

        return log_gained

    def _gained_cb(
        self, start: np.ndarray, x: np.ndarray, alpha_ci: float, bound: str
    ) -> np.ndarray:
        """Bounds on the failures expected between ``start`` and ``start
        + x`` (flat arrays of one length): the delta method on their
        log."""
        log_gained = self._log_gained(start, x)
        with np.errstate(all="ignore"):
            se = delta_method_se(
                log_gained, self._mle, self.covariance(), True
            )
            alpha, signs = bound_signs(alpha_ci, bound)
            z = norm.ppf(1.0 - alpha)
            cb = np.exp(
                log_gained(self._mle)[:, None] + signs * z * se[:, None]
            )
        return cb if bound == "two-sided" else cb[:, 0]

    def next_failure_sf_cb(
        self,
        x: ArrayLike,
        age: ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on :meth:`next_failure_sf`, the chance of
        running a further ``x`` from ``age`` without a failure (the
        reliability over a mission of length ``x``): the delta method on
        the log of the failures expected in the mission,
        :math:`\\Lambda(t + x) - \\Lambda(t)`, turned into bounds on
        :math:`\\exp(-\\cdot)`, so they stay inside [0, 1].

        Parameters
        ----------
        x : array_like
            Times ahead of now (at least 0): the mission's length.
        age : array_like, optional
            Each unit's time now. By default the fitted data's items,
            each at the end of its observation (the end of the test).
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            A lower bound is a lower bound on the reliability.

        Returns
        -------
        numpy.ndarray
            The shape of :meth:`next_failure_sf`'s result, with a last
            axis of ``[lower, upper]`` for two-sided bounds.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GoelOkumoto
        >>> weeks = np.arange(8.0)
        >>> x = np.column_stack([weeks, weeks + 1])
        >>> model = GoelOkumoto.fit(
        ...     x, c=[2] * 8, n=[12, 9, 8, 5, 4, 4, 2, 1]
        ... )
        >>> model.next_failure_sf(0.5, age=[8.0]).round(3)
        array([0.511])
        >>> model.next_failure_sf_cb(0.5, age=[8.0]).round(3)
        array([[0.202, 0.754]])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_inference()
        values = self._ahead(x)
        now = self._ages(age)
        flat = np.atleast_1d(values).reshape(-1)
        start = np.repeat(now, flat.size)
        ahead = np.tile(flat, now.size)
        cb = self._gained_cb(start, ahead, alpha_ci, _OPPOSITE[bound])
        out = np.exp(-cb)
        if bound == "two-sided":
            out = out[:, ::-1].reshape(now.size, flat.size, 2)
            return out[:, 0, :] if values.ndim == 0 else out
        out = out.reshape(now.size, flat.size)
        return out[:, 0] if values.ndim == 0 else out

    # -- testing to a target ------------------------------------------------

    def _time_scale(self) -> float:
        """A time of the model's own size, whatever its unit: when it
        expects its first failure (or half of all it expects, if that is
        fewer than two)."""
        total = self._total()
        count = 1.0 if not total < 2.0 else 0.5 * total
        with np.errstate(all="ignore"):
            scale = float(np.asarray(self.inv_cif(count), dtype=float))
        if not (np.isfinite(scale) and scale > 0):
            raise ValueError(
                "The fitted {} model has no time by which it expects "
                "a failure, so there is no testing time to "
                "find.".format(self.dist.name)
            )
        return scale

    def _reach(
        self, func: Callable, level: float, what: str, target: float
    ) -> float:
        try:
            return last_time_above(func, level, self._time_scale())
        except NeverReached as never:
            scale = self._time_scale()
            with np.errstate(all="ignore"):
                now = float(self.iif(scale))
                far = float(self.iif(1e6 * scale))
            how = (
                "is rising"
                if str(never) == "rising"
                else "does not come down that far"
            )
            raise ValueError(
                "The fitted {name} model never reaches {what} = "
                "{target:g}: its intensity {how}, however long the test "
                "(iif is {now:g} at the model's first expected failure "
                "and {far:g} a million times later; the search goes to "
                "a million million times it). Testing longer will not "
                "get there on this model; check the trend "
                "(trend_test()), or set a target the model can "
                "reach.".format(
                    name=self.dist.name,
                    what=what,
                    target=target,
                    how=how,
                    now=now,
                    far=far,
                )
            ) from None

    @staticmethod
    def _positive(value: Any, name: str, below: float = np.inf) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = np.nan
        if not (0 < number < below):
            limit = "" if below == np.inf else f" and below {below:g}"
            raise ValueError(
                f"{name} must be a number above 0{limit}; got {value!r}"
            )
        return number

    def time_to_intensity(self, target: float) -> float:
        """
        The testing time at which the intensity has fallen to
        ``target``: the time from which :meth:`iif` stays at or below
        it. (A target MTBF ``m`` is the intensity ``1 / m``.) The extra
        testing needed is this less the time tested so far; a result
        before the end of the test says the target is already met.

        Where the intensity rises before it falls (the S-shaped models)
        it is the time on the way down. 0 where the intensity is never
        above the target. A model whose intensity does not fall to the
        target (a constant or rising one) raises a ``ValueError`` saying
        so.

        Parameters
        ----------
        target : float
            The intensity to reach (failures per unit time), above 0.

        Returns
        -------
        float
            The time, on the model's clock (from the start of testing).

        Examples
        --------
        A Goel-Okumoto intensity is ``a b exp(-b t)``, which is 0.5 at
        ``log(a b / 0.5) / b``:

        >>> from surpyval.recurrent import GoelOkumoto
        >>> model = GoelOkumoto.from_params([100, 0.05])
        >>> round(model.time_to_intensity(0.5), 4)
        46.0517
        """
        level = self._positive(target, "target")
        return self._reach(self.iif, level, "an intensity", level)

    def time_to_intensity_cb(
        self,
        target: float,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> Any:
        """
        Confidence bounds on :meth:`time_to_intensity`, by inverting the
        bounds on the intensity: the upper bound is the time at which
        the *upper* confidence bound of :meth:`iif_cb` (the delta
        method) has fallen to the target, and the lower bound the time
        at which its lower bound has. The upper bound is ``inf`` where
        the intensity's upper bound never comes down to the target: the
        data do not rule out that the target is never reached.

        Parameters
        ----------
        target : float
            The intensity to reach, above 0.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are an array ``[lower, upper]``; a
            one-sided bound a number.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GoelOkumoto
        >>> weeks = np.arange(8.0)
        >>> x = np.column_stack([weeks, weeks + 1])
        >>> model = GoelOkumoto.fit(
        ...     x, c=[2] * 8, n=[12, 9, 8, 5, 4, 4, 2, 1]
        ... )
        >>> round(model.time_to_intensity(0.5), 2)
        11.69
        >>> model.time_to_intensity_cb(0.5).round(2)
        array([ 8.53, 20.99])
        """
        level = self._positive(target, "target")
        self.time_to_intensity(level)
        return self._inverted(self._intensity_cb, level, alpha_ci, bound)

    def _intensity_cb(
        self, t: np.ndarray, tail: float, side: str
    ) -> np.ndarray:
        """The one-sided bound of :meth:`iif_cb` at ``t``, worked out
        from the log of the intensity: the same bound, and one that
        keeps its digits where the intensity is too small to square (an
        intensity of 1e-170 has a standard error whose square is 0)."""
        flat = np.atleast_1d(np.asarray(t, dtype=float)).reshape(-1)

        def log_iif(params: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                return np.asarray(
                    self.dist.log_iif(flat, *params), dtype=float
                )

        with np.errstate(all="ignore"):
            se = delta_method_se(log_iif, self._mle, self.covariance(), True)
            z = norm.ppf(1.0 - tail) * (1.0 if side == "upper" else -1.0)
            return np.exp(log_iif(self._mle) + z * se)

    def _inverted(
        self, curve_cb: Callable, level: float, alpha_ci: float, bound: str
    ) -> Any:
        """The times at which the lower and the upper bound of a falling
        curve come down to ``level``."""
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_inference()
        tail = alpha_ci / 2.0 if bound == "two-sided" else alpha_ci
        scale = self._time_scale()

        def crossing(side: str) -> float:
            def curve(t: np.ndarray) -> np.ndarray:
                # one-sided at this tail: 2 * tail in total, one side
                return np.asarray(
                    curve_cb(np.asarray(t, dtype=float), tail, side),
                    dtype=float,
                )

            try:
                return last_time_above(curve, level, scale)
            except NeverReached:
                return np.inf

        if bound == "two-sided":
            return np.array([crossing("lower"), crossing("upper")])
        return crossing(bound)

    def time_to_reliability(self, target: float, x: float) -> float:
        """
        The testing time from which a mission of length ``x`` is run
        without a failure with probability at least ``target``: the
        time ``t`` at which :meth:`next_failure_sf` ``(x, age=t)``
        reaches the target, that is, at which the failures expected in
        the mission, :math:`\\Lambda(t + x) - \\Lambda(t)`, have fallen
        to :math:`-\\log(\\text{target})`.

        0 where the target is met from the start; a ``ValueError``
        where the model never reaches it (see :meth:`time_to_intensity`).

        Parameters
        ----------
        target : float
            The reliability to reach, between 0 and 1.
        x : float
            The mission's length, above 0.

        Returns
        -------
        float
            The time, on the model's clock (from the start of testing).

        Examples
        --------
        >>> from surpyval.recurrent import GoelOkumoto
        >>> model = GoelOkumoto.from_params([100, 0.05])
        >>> t = model.time_to_reliability(0.9, 10.0)
        >>> round(t, 4)
        118.4557
        >>> model.next_failure_sf(10.0, age=[t]).round(6)
        array([0.9])
        """
        level, mission = self._mission(target, x)
        return self._reach(
            lambda t: np.asarray(self.cif(np.asarray(t) + mission))
            - np.asarray(self.cif(t)),
            level,
            f"a reliability over a mission of {mission:g}",
            float(target),
        )

    def _mission(self, target: float, x: float) -> tuple[float, float]:
        reliability = self._positive(target, "target", below=1.0)
        mission = self._positive(x, "x (the mission's length)")
        return -float(np.log(reliability)), mission

    def time_to_reliability_cb(
        self,
        target: float,
        x: float,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> Any:
        """
        Confidence bounds on :meth:`time_to_reliability`, by inverting
        the bounds on the mission's reliability (those of
        :meth:`next_failure_sf_cb`, the delta method): the upper bound
        is the time at which the *lower* confidence bound of the
        reliability has risen to the target, and the lower bound the
        time at which its upper bound has. The upper bound is ``inf``
        where the reliability's lower bound never gets there.

        See :meth:`time_to_intensity_cb` for the arguments.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GoelOkumoto
        >>> weeks = np.arange(8.0)
        >>> x = np.column_stack([weeks, weeks + 1])
        >>> model = GoelOkumoto.fit(
        ...     x, c=[2] * 8, n=[12, 9, 8, 5, 4, 4, 2, 1]
        ... )
        >>> round(model.time_to_reliability(0.9, 0.5), 2)
        14.45
        >>> model.time_to_reliability_cb(0.9, 0.5).round(2)
        array([10.31, 26.76])
        """
        level, mission = self._mission(target, x)
        self.time_to_reliability(target, x)

        def gained_cb(t: np.ndarray, tail: float, side: str) -> np.ndarray:
            flat = np.atleast_1d(t).reshape(-1)
            return self._gained_cb(
                flat, np.full(flat.size, mission), tail, side
            )

        return self._inverted(gained_cb, level, alpha_ci, bound)
