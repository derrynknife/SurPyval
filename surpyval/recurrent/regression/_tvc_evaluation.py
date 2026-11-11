"""A fitted proportional-intensity model along a time-varying covariate.

``TVCEvaluationMixin`` gives
:class:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel`
``cif_tvc``, ``iif_tvc`` and ``cb_tvc``: the expected number of events,
the intensity and their confidence bounds along a step-valued covariate
schedule, with the names and arguments of the life regressions' path
methods (``Hf_tvc``, ``sf_tvc``, ``cb_tvc`` in
:mod:`surpyval.univariate.regression._tvc_evaluation`).

The intensity at a time depends only on the time and the covariates in
force then, so the expected count along a schedule is the sum over its
segments ``(a, b]`` of the baseline's expected count over the segment
times the segment's covariate factor,

.. math::
    \\Lambda\\bigl(x \\mid Z(\\cdot)\\bigr) = \\sum_{\\text{seg } (a, b]}
    e^{\\beta' z}\\,\\bigl[\\Lambda_0(\\min(x, b)) - \\Lambda_0(a)\\bigr] .
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.recurrent.inference import check_alpha_ci
from surpyval.utils.linalg import delta_method_se, log_transformed_cb
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import BOUNDS, check_option


def refuse_schedule(Z: Any, method: str) -> None:
    """Refuse a covariate schedule or path given where one item's fixed
    covariate vector is read (the simulation methods and ``mcf``)."""
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if isinstance(Z, (StepSchedule, CovariatePath)):
        raise ValueError(
            f"Z is a {type(Z).__name__}, but {method} simulates an item "
            "at one fixed covariate vector: simulation along a covariate "
            "schedule is not available yet. Pass one item's covariate "
            "values, or use cif_tvc (with cb_tvc) for the expected number "
            "of events along the schedule."
        )


class TVCEvaluationMixin:
    """The time-varying-covariate evaluation of a
    :class:`ProportionalIntensityModel`, which inherits this mixin and
    supplies the attributes declared below."""

    if TYPE_CHECKING:
        dist: Any
        params: npt.NDArray
        coeffs: npt.NDArray
        feature_names: list[str] | None

        def _coef(self) -> npt.NDArray: ...
        def _check_inference(self) -> None: ...
        def _estimated(self) -> npt.NDArray: ...
        def _mle_values(self) -> npt.NDArray: ...
        def covariance(self) -> npt.NDArray: ...

    def _tvc_schedule(
        self, Z: Any, xl: npt.ArrayLike | None, lo: float, given: bool
    ) -> Any:
        """``Z`` (with ``xl`` for the array form) as a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
        of this model's covariates that says what they are from ``lo`` on
        (the time origin, or the age ``given``)."""
        from surpyval.univariate.regression.tvc_path import CovariatePath
        from surpyval.univariate.regression.tvc_schedule import (
            as_step_schedule,
            check_from_origin,
            is_item,
            item_covariate_path,
        )

        if is_item(Z):
            if xl is not None:
                raise ValueError(
                    "xl must not be given when Z is an item: the item "
                    "gives its own covariates over time"
                )
            schedule = item_covariate_path(Z, self.feature_names)
        elif isinstance(Z, CovariatePath):
            schedule = Z
        else:
            schedule = as_step_schedule(Z, xl)
        if isinstance(schedule, CovariatePath):
            raise NotImplementedError(
                "Z is a CovariatePath (a covariate that changes "
                "continuously): the proportional-intensity models follow "
                "step schedules only. Approximate the path by a "
                "StepSchedule, e.g. StepSchedule.from_changepoints(times, "
                "path(times)) on a grid of times fine enough for the "
                "covariate's changes."
            )
        n_cov = int(np.size(self.coeffs))
        if schedule.p != n_cov:
            raise ValueError(
                f"the schedule has {schedule.p} covariate(s) but the model "
                f"was fit with {n_cov}"
            )
        if not given:
            return check_from_origin(schedule)
        if schedule.start > lo:
            raise ValueError(
                f"The covariates start at {schedule.start:g}, after the "
                f"age given ({lo:g}): the events from that age on depend "
                f"on the covariates from then on. Give their values from "
                f"{lo:g} or earlier."
            )
        return schedule

    @staticmethod
    def _tvc_segments(schedule: Any, lo: float, x: npt.NDArray) -> tuple:
        """The schedule's segments from ``lo`` to the largest finite
        query time: ``(starts, ends, Z)``."""
        from surpyval.univariate.regression.tvc_schedule import segments_from

        if np.isposinf(x).any() and schedule.period is not None:
            raise ValueError(
                "x has an infinite time, and the schedule repeats: it has "
                "no last value to hold to infinity. Give finite times."
            )
        finite = x[np.isfinite(x)]
        t_max = float(finite.max()) if finite.size else lo
        return segments_from(schedule, lo, t_max)

    def _cif_tvc(
        self,
        x: npt.NDArray,
        segments: tuple,
        lo: float,
        given: bool,
        theta: npt.NDArray,
        beta: npt.NDArray,
    ) -> npt.NDArray:
        """The expected count from ``lo`` to each ``x`` along
        ``segments``, at baseline parameters ``theta`` and coefficients
        ``beta``."""
        starts, ends, Zs = segments
        factor = np.exp(Zs @ beta)

        def base(t: npt.NDArray) -> npt.NDArray:
            t = np.asarray(t, dtype=float)
            return np.asarray(
                self.dist.cif(t.ravel(), *theta), dtype=float
            ).reshape(t.shape)

        with np.errstate(all="ignore"):
            upper = np.minimum(np.maximum(x[:, None], starts), ends)
            out = ((base(upper) - base(starts)) * factor).sum(axis=1)
            # Past the last segment's end its value is held.
            beyond = np.isposinf(x)
            if beyond.any():
                tail = base(np.array([np.inf])) - base(ends[-1:])
                out = np.where(beyond, out + tail[0] * factor[-1], out)
            if given:
                # Nothing is counted at or before the age given.
                out = np.where(x <= lo, 0.0, out)
            else:
                # Before the origin, the baseline's own value there with
                # the covariates in force at the origin, as cif(x, Z).
                out = np.where(x < lo, base(x) * factor[0], out)
        return np.where(np.isnan(x), np.nan, out)

    def _iif_tvc(
        self,
        x: npt.NDArray,
        segments: tuple,
        theta: npt.NDArray,
        beta: npt.NDArray,
    ) -> npt.NDArray:
        """The intensity at each ``x`` with the covariates in force
        there along ``segments``."""
        starts, _, Zs = segments
        factor = np.exp(Zs @ beta)
        at = np.clip(
            np.searchsorted(starts, x, side="right") - 1, 0, len(starts) - 1
        )
        with np.errstate(all="ignore"):
            base = np.asarray(self.dist.iif(x, *theta), dtype=float)
            return base * factor[at]

    @staticmethod
    def _given(given: float | None) -> tuple[float, bool]:
        """``(lo, counted_from_given)``: where the count starts."""
        if given is None:
            return 0.0, False
        try:
            lo = float(given)
        except (TypeError, ValueError):
            raise ValueError(
                f"given must be one age (a number); got {given!r}"
            ) from None
        return lo, True

    @keeps_query_shape
    def cif_tvc(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | Any,
        xl: npt.ArrayLike | None = None,
        given: float | None = None,
    ) -> npt.NDArray:
        r"""
        The expected number of events by ``x`` for an item whose
        covariates follow a step schedule ``Z(t)``.

        The intensity at a time is the baseline's then times
        :math:`e^{\beta' Z}` for the covariates in force then, so the
        expected count is summed over the schedule's segments
        :math:`(a, b]`:

        .. math::
            \Lambda\bigl(x \mid Z(\cdot)\bigr) = \sum_{\text{seg}}
            e^{\beta' z}\,\bigl[\Lambda_0(\min(x, b)) - \Lambda_0(a)\bigr].

        A schedule that never changes gives ``cif(x, Z)``. The count is
        from the time origin, 0, so the schedule must start at or before
        0 (one starting later is refused: nothing says what the
        covariates were before it), unless ``given`` is set.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the expected count.
        Z : StepSchedule or array_like
            The covariate schedule: a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            or an array of per-segment covariate rows with ``xl`` giving
            the segment start times, as for the life regressions'
            ``sf_tvc``. A
            :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
            (a covariate that changes continuously) is refused.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            Count from this age instead of from 0: the expected number
            of events in ``(given, x]``, and 0 for ``x <= given``. The
            schedule then need only start at or before ``given``. A
            ``nan`` ``given`` gives ``nan``.

        Returns
        -------
        ndarray
            The expected number of events at each ``x``.

        Examples
        --------
        A pump whose load steps up at month 12:

        >>> import numpy as np
        >>> from surpyval import StepSchedule
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> xl = [0, 6, 10, 13, 17, 0, 3, 7, 10, 16]
        >>> xr = [6, 10, 13, 17, 20, 3, 7, 10, 16, 20]
        >>> c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
        >>> high_load = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        >>> model = ProportionalIntensityHPP.fit_tvc(i, xl, xr, c, high_load)
        >>> plan = StepSchedule.from_changepoints([0, 12], [0.0, 1.0])
        >>> model.cif_tvc([6, 12, 18, 24], plan).round(4)
        array([0.6, 1.2, 2.4, 3.6])

        From month 18 on, for a pump whose covariates are only planned
        from there:

        >>> model.cif_tvc([24], [[1.0]], xl=[18], given=18).round(4)
        array([1.2])
        """
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        lo, from_given = self._given(given)
        schedule = self._tvc_schedule(Z, xl, lo, from_given)
        if np.isnan(lo) or xq.size == 0:
            return np.full(xq.shape, np.nan)
        segments = self._tvc_segments(schedule, lo, xq)
        return self._cif_tvc(
            xq,
            segments,
            lo,
            from_given,
            np.asarray(self.params, dtype=float),
            self._coef(),
        )

    @keeps_query_shape
    def iif_tvc(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | Any,
        xl: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        r"""
        The intensity (events per unit of time) at ``x`` for an item
        whose covariates follow a step schedule ``Z(t)``: the baseline's
        intensity at ``x`` times :math:`e^{\beta' Z}` for the covariates
        in force at ``x``. At a time where the schedule steps, it is the
        intensity just after the step. A schedule that never changes
        gives ``iif(x, Z)``.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the intensity.
        Z : StepSchedule or array_like
            The covariate schedule, as for :meth:`cif_tvc`. It must
            start at or before the time origin, 0.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.

        Returns
        -------
        ndarray
            The intensity at each ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> xl = [0, 6, 10, 13, 17, 0, 3, 7, 10, 16]
        >>> xr = [6, 10, 13, 17, 20, 3, 7, 10, 16, 20]
        >>> c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
        >>> high_load = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        >>> model = ProportionalIntensityHPP.fit_tvc(i, xl, xr, c, high_load)
        >>> model.iif_tvc([6, 12, 18], [[0.0], [1.0]], xl=[0, 12]).round(4)
        array([0.1, 0.2, 0.2])
        """
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        schedule = self._tvc_schedule(Z, xl, 0.0, False)
        if xq.size == 0:
            return np.full(xq.shape, np.nan)
        segments = self._tvc_segments(schedule, 0.0, xq)
        return self._iif_tvc(
            xq, segments, np.asarray(self.params, dtype=float), self._coef()
        )

    @keeps_query_shape
    def cb_tvc(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike | Any,
        xl: npt.ArrayLike | None = None,
        given: float | None = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Confidence bounds on the expected number of events
        (:meth:`cif_tvc`) or the intensity (:meth:`iif_tvc`) along a
        step schedule ``Z(t)``.

        The bounds are those of ``cif_cb`` and ``iif_cb`` carried along
        the schedule: the delta method on the log scale, from the joint
        covariance of the baseline parameters and the coefficients. A
        schedule that never changes gives ``cif_cb(x, Z)`` (or
        ``iif_cb``) exactly.

        Parameters
        ----------
        x : array_like
            Times at which to bound the function.
        Z : StepSchedule or array_like
            The covariate schedule, as for :meth:`cif_tvc`.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            Count from this age, as for :meth:`cif_tvc` (``on="cif"``
            only): the bounds are on the expected number of events in
            ``(given, x]``, and 0, with no width, at and before
            ``given``.
        on : {'cif', 'iif'}, optional
            The function to bound. Default ``'cif'``.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.

        Returns
        -------
        numpy array
            The confidence bound(s) at each ``x``: the query's shape,
            with ``[lower, upper]`` on a last axis for two-sided bounds.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import StepSchedule
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> xl = [0, 6, 10, 13, 17, 0, 3, 7, 10, 16]
        >>> xr = [6, 10, 13, 17, 20, 3, 7, 10, 16, 20]
        >>> c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
        >>> high_load = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        >>> model = ProportionalIntensityHPP.fit_tvc(i, xl, xr, c, high_load)
        >>> plan = StepSchedule.from_changepoints([0, 12], [0.0, 1.0])
        >>> cb = model.cb_tvc(24, plan)
        >>> bool(cb[0] < model.cif_tvc(24, plan) < cb[1])
        True

        A schedule that never changes gives the fixed-covariate bounds:

        >>> flat = StepSchedule.constant([1.0])
        >>> bool(np.allclose(model.cb_tvc([10, 20], flat),
        ...                  model.cif_cb([10, 20], [1.0])))
        True
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, ("cif", "iif"))
        check_option("bound", bound, BOUNDS)
        if on == "iif" and given is not None:
            raise ValueError(
                "given applies to on='cif' (the events counted from an "
                "age): the intensity at a time depends only on the "
                "covariates then. Leave given out, with a schedule from "
                "the time origin."
            )
        self._check_inference()
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        shape = xq.shape + ((2,) if bound == "two-sided" else ())
        lo, from_given = self._given(given)
        schedule = self._tvc_schedule(Z, xl, lo, from_given)
        if np.isnan(lo) or xq.size == 0:
            return np.full(shape, np.nan)
        segments = self._tvc_segments(schedule, lo, xq)
        k = len(self.params)

        def function_at(theta: npt.NDArray) -> npt.NDArray:
            if on == "iif":
                return self._iif_tvc(xq, segments, theta[:k], theta[k:])
            return self._cif_tvc(
                xq, segments, lo, from_given, theta[:k], theta[k:]
            )

        # An aliased coefficient is held at 0 and has no variance.
        held = ~self._estimated()
        cov = self.covariance()
        cov[held, :] = 0.0
        cov[:, held] = 0.0
        mle = self._mle_values()
        se = delta_method_se(function_at, mle, cov)
        return log_transformed_cb(function_at(mle), se, alpha_ci, bound)
