"""A fitted parametric regression model along a time-varying covariate.

``TVCEvaluationMixin`` holds the time-varying-covariate evaluation for
:class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`,
which inherits it: ``Hf_tvc``, ``sf_tvc``, ``mean_tvc`` and ``cb_tvc``
along a step-valued covariate schedule or a continuous covariate path, for
the proportional hazards, additive hazards, proportional odds,
accelerated failure time and accelerated life families (#172).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

from ._kinds import (
    ACCELERATED_FAILURE_TIME,
    ACCELERATED_LIFE,
    ADDITIVE_HAZARD,
    PROPORTIONAL_HAZARD,
    PROPORTIONAL_ODDS,
)

if TYPE_CHECKING:
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )


class TVCEvaluationMixin:
    """The time-varying-covariate evaluation of a
    :class:`ParametricRegressionModel`, which inherits this mixin.

    Separated from ``ParametricRegressionModel`` to keep the evaluation
    along a covariate path in one place; every method here reads the
    fitted model through the attributes and helpers
    ``ParametricRegressionModel`` (and its ``InferenceMixin``) defines.
    """

    if TYPE_CHECKING:
        # Supplied by ParametricRegressionModel, the one class that
        # inherits this mixin. Declared rather than defined so the methods
        # below type check without the mixin pretending to own them.
        k_dist: int
        kind: str
        distribution: ParametricFitter
        model: Any
        center: "npt.NDArray | None"

        @property
        def life_parameter(self) -> "str | None": ...
        def _eval_params(self) -> npt.NDArray: ...
        def _n_covariates(self) -> int: ...
        def _is_accelerated_life(self) -> bool: ...
        def _is_additive(self) -> bool: ...

        def _centred(
            self, Z: npt.ArrayLike, center: "npt.NDArray | None" = None
        ) -> Any: ...

        def _warn_negative_hazard(
            self,
            count: int,
            size: int,
            max_sf: "float | None",
            stacklevel: int,
        ) -> None: ...

        def _check_inference(self) -> None: ...

        def _inference_state(
            self,
        ) -> "tuple[npt.NDArray, npt.NDArray | None, npt.NDArray]": ...

        def _sf_bounds(
            self,
            H_of: Any,
            sf_of: Any,
            params: npt.NDArray,
            cov: npt.NDArray,
            shape: tuple,
            on: str,
            alpha_ci: float,
            bound: str,
        ) -> npt.NDArray: ...

    # Families whose survival along a step-valued covariate path has an exact
    # closed form. Proportional hazards, additive hazards and proportional
    # odds have a hazard that depends only on the time and the *current*
    # covariate, so the cumulative hazard is a sum of per-segment increments
    # of the constant-covariate ``Hf``; accelerated failure time instead
    # accumulates an *accelerated age* over the segments and then evaluates the
    # baseline once. Accelerated life does the same (cumulative exposure,
    # with the rate 1 / L(Z)) where its life parameter scales time; a
    # location life parameter is refused (see ``_check_tvc_evaluable``).
    _TVC_ADDITIVE_KINDS = (
        PROPORTIONAL_HAZARD,
        ADDITIVE_HAZARD,
        PROPORTIONAL_ODDS,
    )
    _TVC_EVALUABLE_KINDS = (
        PROPORTIONAL_HAZARD,
        ADDITIVE_HAZARD,
        PROPORTIONAL_ODDS,
        ACCELERATED_FAILURE_TIME,
        ACCELERATED_LIFE,
    )
    #: The accelerated-life distributions whose life parameter is a scale
    #: of time (Weibull ``alpha``, the Exponential and Gamma rates
    #: ``1 / L``, LogNormal's ``exp(mu)``): S(t | V) = S_1(t / L(V)), with
    #: S_1 the distribution at unit life, so a changing stress accumulates
    #: an age ``int du / L(V(u))`` (#172).
    _TVC_SCALE_LIFE = ("Weibull", "Exponential", "Gamma", "LogNormal")

    def _check_tvc_evaluable(self) -> None:
        """Refuse a family with no form along a covariate path."""
        if self.kind not in self._TVC_EVALUABLE_KINDS:
            raise NotImplementedError(
                "time-varying-covariate evaluation is defined for the "
                "proportional-hazards, additive-hazards, proportional-odds, "
                "accelerated-failure-time and accelerated-life families "
                "(this model is '{}').".format(self.kind)
            )
        name = self.distribution.name
        if self._is_accelerated_life() and name not in (self._TVC_SCALE_LIFE):
            raise NotImplementedError(
                "An Accelerated Life model is evaluated along a changing "
                "stress by cumulative exposure, S(t) = S_1(int_0^t du / "
                "L(V(u))), which needs a life parameter that scales time "
                "({} only). The {} life parameter '{}' is a location: a "
                "change of stress shifts the distribution rather than "
                "rescaling time, so there is no accumulated age to carry "
                "from one stress to the next. Fit a scale-life "
                "distribution (e.g. AcceleratedLife(Weibull, ...)) or an "
                "accelerated failure time model for this.".format(
                    ", ".join(self._TVC_SCALE_LIFE),
                    name,
                    self.life_parameter,
                )
            )

    def _tvc_scales_time(self) -> bool:
        """Whether the covariate rescales time along a path (AFT, and
        accelerated life), rather than setting the current hazard."""
        return self.kind in (ACCELERATED_FAILURE_TIME, ACCELERATED_LIFE)

    def _tvc_theta(
        self, theta: "tuple | None"
    ) -> "tuple[npt.NDArray, npt.NDArray | None]":
        """``(params, center)`` to evaluate a path at: ``theta``, or the
        model's own (``cb_tvc`` passes those of ``_inference_state``)."""
        if theta is None:
            return self._eval_params(), self.center
        return theta

    def _tvc_rate(self, Zc: npt.NDArray, params: npt.NDArray) -> npt.NDArray:
        """The rate at which a covariate row (already centred) ages a unit:
        AFT's ``phi = exp(beta'z)``, and an accelerated life model's
        ``1 / L(z)``. One value per row."""
        Zc = np.atleast_2d(np.asarray(Zc, dtype=float))
        phi_params = params[self.k_dist :]
        with np.errstate(all="ignore"):
            if self._is_accelerated_life():
                rate = 1.0 / np.asarray(
                    self.model.phi(Zc, *phi_params), dtype=float
                )
            else:
                rate = np.asarray(
                    self.model._phi(Zc, *phi_params), dtype=float
                )
        rate = rate.ravel()
        if rate.size == 1 and Zc.shape[0] != 1:
            rate = np.full(Zc.shape[0], float(rate[0]))
        return rate

    def _tvc_segments(
        self, schedule: Any, t_max: float
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Materialise ``schedule`` to ``t_max`` with the first segment held back
        to the time origin (survival measured from ``0``).
        """
        from .tvc_schedule import segments_from_origin

        return segments_from_origin(schedule, t_max)

    def _to_schedule(self, Z: Any, xl: "npt.ArrayLike | None") -> Any:
        """
        Coerce the ``sf_tvc`` covariate argument into a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` and
        check its covariate count against the fitted model.
        """
        from .tvc_schedule import as_covariate_path

        schedule = as_covariate_path(Z, xl)
        # Columns of Z (an accelerated life model's life-model parameters
        # are not one per column).
        n_cov = self._n_covariates()
        if schedule.p != n_cov:
            raise ValueError(
                "the {} has {} covariate(s) but the model was fit with "
                "{}".format(
                    "schedule" if hasattr(schedule, "segments") else "path",
                    schedule.p,
                    n_cov,
                )
            )
        return schedule

    @keeps_query_shape
    def Hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard for a covariate following a path ``Z(t)``: a step
        schedule, or a continuously varying path.

        For the proportional-hazards, additive-hazards and proportional-odds
        families the hazard at time :math:`t` depends only on :math:`t` and
        the covariate value *at* :math:`t`, so along a piecewise constant path
        the cumulative hazard is exactly the sum of the per-segment increments
        of the constant-covariate cumulative hazard

        .. math::
            H\bigl(x \mid Z(\cdot)\bigr)
            = \sum_{\text{seg } (a, b]} \bigl[\,H(b, z) - H(a, z)\,\bigr] .

        For proportional odds, with :math:`\phi = e^{\beta' z}` multiplying the
        survival odds, the hazard is
        :math:`h(t \mid z) = h_0(t) / (F_0(t) + \phi S_0(t))` and its integral
        at constant :math:`z` is
        :math:`H(t, z) = H_0(t) - \ln\phi + \ln(F_0(t) + \phi S_0(t))
        = -\ln S(t \mid z)`, so each segment contributes
        :math:`\ln[S(a \mid z) / S(b \mid z)]`. On entering a segment the
        hazard switches to the new covariate's PO hazard; the survival does
        not jump to the new covariate's PO curve. The first segment is held
        back to the bottom of the baseline's support (for a baseline defined
        below zero, such as ``Logistic``, the value in force at time zero is
        taken to apply before it too), so the result is the unconditional
        survival.

        For accelerated failure time the covariate rescales time, so the path
        accumulates an *accelerated age*
        :math:`\psi(x) = \sum_{\text{seg}} e^{\beta' z}\,(b - a)` and the
        cumulative hazard is the baseline evaluated there,
        :math:`H(x \mid Z(\cdot)) = H_0(\psi(x))`. An accelerated life model
        whose life parameter scales time (Weibull ``alpha``, the Exponential
        and Gamma rates, LogNormal's ``exp(mu)``) does the same with the rate
        :math:`1 / L(z)` and the distribution at unit life, Nelson's
        cumulative exposure; one whose life parameter is a location (Normal,
        Gumbel, Logistic) raises ``NotImplementedError``. Either way a
        single constant segment reduces exactly to ``Hf(x, Z)``.

        Along a
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`, a
        covariate that changes continuously, the same hazards are
        integrated: :math:`H(x) = \int_0^x h(u \mid Z(u))\, du`, and for
        accelerated failure time
        :math:`\psi(x) = \int_0^x e^{\beta' Z(u)}\, du` (Nelson's cumulative
        exposure). For proportional odds the hazard is that of the current
        covariate, :math:`h_0(t) / (F_0(t) + \phi(Z(t)) S_0(t))`: the limit
        of the step sum above, and the model ``fit_tvc`` fits. The integral
        is by adaptive Gauss-Kronrod quadrature to a relative error of about
        ``1e-10``; where that is not reached, one ``RuntimeWarning`` says at
        how many query times. The type of ``Z`` picks the method: a
        ``StepSchedule`` is always summed exactly.

        The path is measured from time zero: a schedule starting after zero
        has its first value held back to zero, and the part of a schedule
        before zero is ignored (the value in force at zero applies from
        there). Any time is a valid query, zero and below included: a
        constant path gives ``Hf(x, Z)`` there too.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the cumulative hazard.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path -- a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            a :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`,
            or an array of per-segment covariate rows (with ``xl`` giving the
            segment start times).
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.

        Returns
        -------
        ndarray
            The cumulative hazard at each ``x``.

        Examples
        --------
        An exponential proportional hazards model along a stress ramp
        ``Z(t) = 0.1 t``, whose cumulative hazard is
        :math:`\lambda (e^{0.1 \beta t} - 1) / (0.1 \beta)`:

        >>> import numpy as np
        >>> from surpyval import CovariatePath, ExponentialPH
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 2, (300, 1))
        >>> x = rng.exponential(10 * np.exp(-0.7 * Z[:, 0]))
        >>> model = ExponentialPH.fit(x, Z)
        >>> lam, beta = model.params
        >>> ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
        >>> t = np.array([2.0, 5.0, 10.0])
        >>> H = model.Hf_tvc(t, ramp)
        >>> exact = lam * np.expm1(0.1 * beta * t) / (0.1 * beta)
        >>> bool(np.allclose(H, exact, rtol=1e-12, atol=0))
        True
        """
        H, falls, accuracy = self._hf_tvc(x, Z, xl)
        self._warn_tvc(H, falls, accuracy, stacklevel=5)
        return H

    def _warn_tvc(
        self,
        H: npt.NDArray,
        falls: int,
        accuracy: "tuple[int, int, float, str] | None",
        stacklevel: int,
    ) -> None:
        """The warnings of ``sf_tvc`` / ``Hf_tvc``: a falling additive
        hazard (#376), and a quadrature that missed its target (#172);
        ``stacklevel`` counts from here to the caller of the public
        method."""
        if falls:
            self._warn_negative_hazard(
                falls, H.size, self._max_sf(H), stacklevel=stacklevel
            )
        if accuracy is not None:
            from .tvc_path import warn_missed_target

            missed, total, worst, limit = accuracy
            warn_missed_target(
                missed, total, worst, limit, self._tvc_rtol, stacklevel
            )

    @staticmethod
    def _max_sf(H: npt.NDArray) -> "float | None":
        finite = H[np.isfinite(H)]
        if finite.size and finite.min() < 0:
            return float(np.exp(-finite.min()))
        return None

    def _hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None",
        given: "float | None" = None,
        theta: "tuple | None" = None,
        frozen: "dict | None" = None,
    ) -> "tuple[npt.NDArray, int, tuple | None]":
        """The cumulative hazard along the path, the number of query
        times at which an additive hazard fell (a negative ``H`` or a
        negative segment increment, #376) -- 0 for the other families --
        and, for a ``CovariatePath`` whose quadrature missed its target,
        ``(missed, total, worst)`` (else ``None``). ``given`` is used only
        for a ``CovariatePath``: ``H`` is then integrated from ``given``,
        ``H(x) - H(given)``.

        ``theta`` is ``(params, center)`` to evaluate at instead of the
        model's own. ``frozen`` is a dict that keeps a path's quadrature
        mesh: an empty one gets the mesh of this call (under
        ``"edges"``), and one holding a mesh is integrated on it with no
        refinement, so that a function of the parameters is smooth in
        them (``cb_tvc``'s delta method, #172)."""
        from .tvc_path import CovariatePath

        self._check_tvc_evaluable()
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        schedule = self._to_schedule(Z, xl)
        # A missing query time has no value (NaN); the others are
        # evaluated as usual.
        missing = np.isnan(xq)
        if missing.all():
            return np.full(xq.shape, np.nan), 0, None
        if isinstance(schedule, CovariatePath):
            # Integrated, not summed.
            return self._tvc_hf_path(xq, schedule, given, theta, frozen)
        # A horizon at or below 0 materialises the one segment in force at
        # 0: H is then 0, or the baseline's value for a time below 0.
        t_max = float(np.max(xq[~missing]))
        starts, ends, Zseg = self._tvc_segments(schedule, t_max)
        xq_eval = np.where(missing, t_max, xq)

        falls = np.zeros(xq.shape[0], dtype=bool)
        if self.kind in self._TVC_ADDITIVE_KINDS:
            H = self._tvc_hf_additive(
                xq_eval, starts, ends, Zseg, falls, theta
            )
        else:
            H = self._tvc_hf_aft(xq_eval, starts, ends, Zseg, theta)
        if self._is_additive():
            falls |= H < 0
        falls &= ~missing
        return np.where(missing, np.nan, H), int(falls.sum()), None

    #: The relative accuracy the quadrature along a ``CovariatePath``
    #: aims for on the cumulative hazard (private: tests change it).
    _tvc_rtol: float = 1e-10

    def _tvc_hf_path(
        self,
        xq: npt.NDArray,
        path: Any,
        given: "float | None",
        theta: "tuple | None" = None,
        frozen: "dict | None" = None,
    ) -> "tuple[npt.NDArray, int, tuple | None]":
        r"""
        The cumulative hazard along a continuously varying ``path`` (#172),
        less its value at ``given`` when that is supplied; the values and
        counts, ``theta`` and ``frozen`` are as for :meth:`_hf_tvc`.

        At and before time 0 the value in force at 0 applies, exactly as
        for a step schedule, so there the one-segment step sum gives ``H``.
        After 0 the hazard (for AFT and accelerated life, the rate at which
        the unit ages) is integrated over panels by
        :func:`~.tvc_path.integrate_panels`, and summed outward from
        ``given`` (or 0): nothing is subtracted for a baseline that starts
        at 0.

        Along a periodic path the age a time-scaling family accumulates
        over a whole period is the same every period, so only one period
        is integrated: :math:`\psi(t) = k\,\Psi_P + \psi(t - kP)` with
        :math:`k = \lfloor t / P \rfloor` (#172, the periodic shortcut). A
        hazard family has no such shortcut: its baseline ages.
        """
        from .tvc_path import (
            integrate_panels,
            missed_target,
            path_mesh,
            sum_between,
        )

        aft = self._tvc_scales_time()
        missing = np.isnan(xq)
        xe = np.where(missing, 0.0, xq)
        n = xq.shape[0]
        g_pos = given is not None and given > 0

        # H at min(x, 0), at 0 and at min(given, 0): the one segment in
        # force at 0, [0, 0], of the step sum.
        z0 = np.asarray(path._values(np.zeros(1), left=False), dtype=float)
        g_low = min(given, 0.0) if given is not None else 0.0
        low_t = np.concatenate([np.minimum(xe, 0.0), [0.0, g_low]])
        falls_low = np.zeros(low_t.shape, dtype=bool)
        seg = (np.zeros(1), np.zeros(1), z0.reshape(1, -1))
        if aft:
            H_low = self._tvc_hf_aft(low_t, *seg, theta)
        else:
            H_low = self._tvc_hf_additive(low_t, *seg, falls_low, theta)
        H_x_low, H_at0, H_g_low = H_low[:n], H_low[n], H_low[n + 1]
        falls = falls_low[:n].copy()

        points = xe[xe > 0]
        if g_pos:
            points = np.append(points, given)
        if points.size == 0:
            # Every time is at or before 0.
            H_full = H_x_low
            H = H_full if given is None else H_full - H_g_low
            accuracy = None
        else:
            ex = np.maximum(xe, 0.0)
            period = path.period
            # The periodic shortcut: integrate one period only.
            whole = aft and period is not None and np.max(points) > period

            def split(t: npt.NDArray) -> tuple:
                # Whole periods, and the time into the last one.
                k = np.floor(t / period)
                return k, np.clip(t - k * period, 0.0, period)

            if whole:
                k_x, r_x = split(ex)
                inner = np.append(r_x[r_x > 0], period)
                if g_pos:
                    inner = np.append(inner, split(np.array([given]))[1])
                inner = inner[inner > 0]
            else:
                inner = points
            if frozen is not None and "edges" in frozen:
                mesh, rounds = frozen["edges"], 0
            else:
                # The model's own time scale, where the hazard can sit
                # however far out the query is.
                params, center = self._tvc_theta(theta)
                scale = self._tvc_time_scale(
                    0.0, self._centred(z0, center), params
                )
                mesh, rounds = path_mesh(path, np.unique(inner), scale), None
            res = integrate_panels(
                self._path_panel_terms(path, theta),
                mesh,
                self._tvc_rtol,
                max_rounds=rounds,
            )
            if frozen is not None and "edges" not in frozen:
                frozen["edges"] = res["edges"]
            edges, value = res["edges"], res["value"]

            def age(t: npt.NDArray) -> npt.NDArray:
                # The integral from 0 to each t (the accelerated age, for
                # a family that scales time).
                if not whole:
                    return sum_between(edges, value, 0.0, t)
                k, r = split(t)
                cycle = sum_between(edges, value, 0.0, np.array([period]))
                return k * cycle[0] + sum_between(edges, value, 0.0, r)

            from_0 = age(ex)
            if aft:
                # The accelerated age, through the baseline once.
                H_full = np.where(xe > 0, self._aft_H0(from_0, theta), H_x_low)
                H = H_full
                if given is not None:
                    if g_pos:
                        psi_g = age(np.array([float(given)]))
                        H = H_full - self._aft_H0(psi_g, theta)[0]
                    else:
                        H = H_full - H_g_low
                origin = 0.0
                if whole:
                    # Each value carries a whole period's integral (but
                    # for those in the first period, where this is
                    # conservative).
                    reach = np.full(ex.shape, float(period))
                else:
                    reach = np.maximum(ex, given) if g_pos else ex
            else:
                # H(x) = A(x) + int_0^max(x, 0) h, with A(x) the step
                # value at min(x, 0) (0 for a baseline that starts at 0).
                A_x = np.where(xe > 0, H_at0, H_x_low)
                H_full = A_x + from_0
                origin = float(given) if given is not None and g_pos else 0.0
                H = H_full
                if given is not None:
                    A_g = H_at0 if g_pos else H_g_low
                    H = (A_x - A_g) + sum_between(edges, value, origin, ex)
                reach = ex
                if self._is_additive():
                    # A negative hazard at a node before x (#376).
                    fell = sum_between(
                        edges, res["flag"], 0.0, ex, signed=False
                    )
                    falls |= fell > 0
            accuracy = missed_target(
                res, origin, reach, self._tvc_rtol, missing
            )
        if self._is_additive():
            falls |= H_full < 0
        falls &= ~missing
        return np.where(missing, np.nan, H), int(falls.sum()), accuracy

    def _aft_H0(
        self, psi: npt.NDArray, theta: "tuple | None" = None
    ) -> npt.NDArray:
        """The baseline cumulative hazard at accelerated ages ``psi``: the
        AFT baseline, or an accelerated life model's distribution at unit
        life (its life parameter set to ``L = 1``). (An age of 0 makes a
        log-time baseline evaluate log(0) = -inf on its way to the correct
        H = 0.)"""
        params = self._tvc_theta(theta)[0]
        dist = np.array(params[: self.k_dist], dtype=float)
        if self._is_accelerated_life():
            slot = self.model.param_map[self.model.life_parameter]
            dist[slot] = self.model.param_transform(1.0)
        with np.errstate(divide="ignore"):
            return np.asarray(
                self.model.Hf_dist(np.asarray(psi, dtype=float), *dist),
                dtype=float,
            ).ravel()

    def _path_panel_terms(
        self, path: Any, theta: "tuple | None" = None
    ) -> Any:
        """
        The family's ``panel_terms(a, b)`` for
        :func:`~.tvc_path.integrate_panels`: on each panel ``[a, b]`` the
        exact increment with the covariate frozen at the panel's midpoint
        value ``zbar``, and at the 15 Kronrod nodes the correction
        integrand -- the hazard along the path less the hazard at ``zbar``
        (for AFT and accelerated life, the ageing rate ``phi(Z(u)) -
        phi(zbar)``) -- its size, for the rounding floor, and (for AH)
        whether the hazard is negative at a node. The correction is 0
        where the path is flat.
        """
        from .tvc_path import _NODES

        M = self.model
        params, center = self._tvc_theta(theta)
        aft = self._tvc_scales_time()
        additive = self._is_additive()
        starts_at_0 = float(self.distribution.support[0]) >= 0

        def flat(values: Any, n: int) -> npt.NDArray:
            arr = np.asarray(values, dtype=float)
            if arr.size == 1 and n != 1:
                return np.full(n, float(arr.ravel()[0]))
            return arr.reshape(n)

        def terms(a: npt.NDArray, b: npt.NDArray) -> tuple:
            m = a.shape[0]
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            u = (mid[:, None] + half[:, None] * _NODES[None, :]).ravel()
            n = u.shape[0]
            zu = self._centred(path._values(u), center)
            zbar = self._centred(path._values(mid), center)
            zrep = np.repeat(zbar, _NODES.shape[0], axis=0)
            with np.errstate(all="ignore"):
                if aft:
                    along = self._tvc_rate(zu, params)
                    frozen = self._tvc_rate(zrep, params)
                    exact = self._tvc_rate(zbar, params) * (b - a)
                else:
                    along = flat(M.hf(u, zu, *params), n)
                    frozen = flat(M.hf(u, zrep, *params), n)
                    # The first panel of a baseline that starts at 0 has
                    # H(0) = 0 (a log-time baseline would say log(0)).
                    first = (a == 0) & starts_at_0
                    hi = flat(M.Hf(b, zbar, *params), m)
                    lo = flat(M.Hf(np.where(first, b, a), zbar, *params), m)
                    exact = hi - np.where(first, 0.0, lo)
                g = (along - frozen).reshape(m, -1)
            scale = np.abs(along).reshape(m, -1)
            if additive:
                flag = (along < 0).reshape(m, -1).any(axis=1)
            else:
                flag = np.zeros(m, dtype=bool)
            return exact, g, scale, flag

        return terms

    def _tvc_hf_additive(
        self,
        xq: npt.NDArray,
        starts: npt.NDArray,
        ends: npt.NDArray,
        Zseg: npt.NDArray,
        falls: "npt.NDArray | None" = None,
        theta: "tuple | None" = None,
    ) -> npt.NDArray:
        """
        Cumulative hazard along a step path for the families whose hazard
        depends only on the time and the current covariate (PH, AH, PO):
        telescoping sum of the model's ``Hf`` increment on each segment, the
        last clipped at the query time.
        """
        params, center = self._tvc_theta(theta)
        H = np.zeros(xq.shape[0], dtype=float)
        support_lo = float(self.distribution.support[0])
        for i, (a, b, z) in enumerate(zip(starts, ends, Zseg)):
            zrow = self._centred(
                np.asarray(z, dtype=float).reshape(1, -1), center
            )
            # Query times before 0 fall in the first segment when the
            # baseline is defined there.
            upper = np.clip(xq, min(a, support_lo) if i == 0 else a, b)
            # A query time of 0 makes a log-time baseline (LogNormal,
            # LogLogistic) evaluate log(0) = -inf on its way to the correct
            # H = 0; that is not worth a warning.
            with np.errstate(divide="ignore"):
                hi = np.asarray(
                    self.model.Hf(upper, zrow, *params), dtype=float
                ).ravel()
                # The first segment runs from the bottom of the support,
                # where H = 0. Subtracting H(0, z) instead would, for a
                # baseline defined below zero (Normal, Gumbel, Logistic),
                # give the survival conditional on reaching 0, not sf(x, Z).
                if i == 0:
                    lo = np.zeros(1)
                else:
                    lo = np.asarray(
                        self.model.Hf(np.array([a]), zrow, *params),
                        dtype=float,
                    ).ravel()
            if falls is not None and self._is_additive():
                # A negative increment: the additive hazard fell in this
                # segment before the query time (#376).
                falls |= (hi - lo) < 0
            H = H + (hi - lo)
        return H

    def _tvc_hf_aft(
        self,
        xq: npt.NDArray,
        starts: npt.NDArray,
        ends: npt.NDArray,
        Zseg: npt.NDArray,
        theta: "tuple | None" = None,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard along a step path for accelerated failure time
        and accelerated life.

        The covariate rescales time by ``phi(z) = exp(beta'z)`` (for
        accelerated life ``1 / L(z)``), so each segment contributes
        ``phi(z) * (width)`` of *accelerated age*. The accumulated age
        ``psi(x)`` is then fed once through the baseline cumulative hazard
        ``H0`` (for accelerated life, the distribution at unit life). This
        is exact for a step covariate and reduces to ``Hf(x, Z)`` for a
        single constant segment.
        """
        params, center = self._tvc_theta(theta)
        psi = np.zeros(xq.shape[0], dtype=float)
        support_lo = float(self.distribution.support[0])
        for i, (a, b, z) in enumerate(zip(starts, ends, Zseg)):
            zrow = self._centred(
                np.asarray(z, dtype=float).reshape(1, -1), center
            )
            rate = float(self._tvc_rate(zrow, params)[0])
            # Query times before 0 fall in the first segment when the
            # baseline is defined there (a negative age, as sf(x, Z)).
            width = np.clip(xq, min(a, support_lo) if i == 0 else a, b) - a
            psi = psi + rate * width
        return self._aft_H0(psi, theta)

    @keeps_query_shape
    def sf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
    ) -> npt.NDArray:
        r"""
        Survival for a covariate that follows a path ``Z(t)``: a step
        (piecewise-constant) schedule, or a continuously varying path.

        With a time-varying covariate the survival depends on the whole
        covariate path, not one fixed vector. This is exact along a step path
        for the proportional-hazards, additive-hazards, proportional-odds and
        accelerated-failure-time families: ``S(x) = exp(-H(x))`` with ``H`` the
        per-segment accumulation in :meth:`Hf_tvc` (a cumulative-hazard sum for
        PH/AH/PO, an accelerated-age sum fed through the baseline for AFT).
        Along a
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath` the
        same quantities are integrated by quadrature, to a relative error of
        about ``1e-10`` on ``H`` (see :meth:`Hf_tvc`). A constant path gives
        ``sf(x, Z)``. An accelerated life model follows cumulative exposure
        where its life parameter scales time, and raises
        ``NotImplementedError`` where it is a location (see :meth:`Hf_tvc`).
        :meth:`cb_tvc` bounds this survival and :meth:`mean_tvc` integrates
        it.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate survival.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path. A
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
            (built from change-points, intervals, a cyclic pattern, or a
            step-valued expression), a
            :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
            (a covariate that changes continuously), or an array of
            per-segment covariate rows with ``xl`` giving the segment start
            times.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            If supplied, return the *conditional* survival given the item has
            survived to age ``given``:
            ``S(x | given) = exp(-(H(x) - H(given)))`` for ``x > given``,
            and 1 for ``x <= given`` (survival to those times is certain).
            Along a ``CovariatePath`` the hazard is integrated from
            ``given`` on, so nothing is subtracted. A ``nan`` ``given``
            gives ``nan``.

        Returns
        -------
        ndarray
            Survival at each ``x`` (conditional on ``given`` when supplied).

        Examples
        --------
        A proportional-odds model whose covariate switches from 0 to 1 at
        ``t = 6``: before the switch the survival is that of ``Z = 0``, after
        it the hazard is that of ``Z = 1``.

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPO
        >>> from surpyval.univariate.regression import StepSchedule
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(0.5 * Z[:, 0])
        >>> model = WeibullPO.fit(x, Z)
        >>> sched = StepSchedule.from_changepoints([0, 6], [[0.0], [1.0]])
        >>> model.sf_tvc([4, 8, 12], sched).round(4)
        array([0.7721, 0.5698, 0.4292])
        >>> model.sf([4, 8, 12], [[0]]).round(4)
        array([0.7721, 0.4937, 0.2809])

        Along a covariate ramped from 0 to 1 over the first 10 time units,
        and conditional on survival to 4:

        >>> from surpyval import CovariatePath
        >>> ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
        >>> model.sf_tvc([4, 8, 12], ramp).round(4)
        array([0.8195, 0.6331, 0.471 ])
        >>> model.sf_tvc([2, 4, 8, 12], ramp, given=4).round(4)
        array([1.    , 1.    , 0.7725, 0.5747])
        """
        from .tvc_path import CovariatePath

        g = None if given is None else float(given)
        # Along a path the hazard is integrated from given on; a step
        # schedule subtracts H(given).
        from_given = (
            isinstance(Z, CovariatePath) and g is not None and not np.isnan(g)
        )
        H, falls, accuracy = self._hf_tvc(x, Z, xl, g if from_given else None)
        self._warn_tvc(H, falls, accuracy, stacklevel=5)
        if g is not None and not from_given:
            if np.isnan(g):
                # A missing conditioning age: nothing is known (as Cox).
                H = np.full(np.shape(H), np.nan)
            else:
                # H(given) is 0 at or below 0, unless the baseline has
                # mass below 0 (then it is -log of the survival to given).
                H = H - self._hf_tvc(g, Z, xl)[0]
        if g is not None and not np.isnan(g):
            # Given survival to g, survival to any x <= g is certain: the
            # difference H(x) - H(g) is not a cumulative hazard there, and
            # gave a "survival" above 1 (#523).
            xq = np.atleast_1d(np.asarray(x, dtype=float))
            H = np.where(xq <= g, 0.0, H)
        return np.exp(-H)

    def mean_tvc(
        self,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
    ) -> float:
        r"""
        The mean life along a covariate path ``Z(t)`` (a step schedule or
        a continuously varying path, as for :meth:`sf_tvc`), or, with
        ``given``, the mean *residual* life of a unit that has survived to
        that age along it.

        .. math::
            \text{mean} = \int_0^\infty S(t)\, dt, \qquad
            \text{mrl}(g) = \int_g^\infty S(t \mid g)\, dt .

        The outer integral is adaptive Gauss-Kronrod on panels graded
        geometrically from the start, and its nodes are simply more query
        times of :meth:`sf_tvc`: each round of refinement is one pass along
        the path, not an integral per node. A step schedule is integrated
        as the matching piecewise-constant path, whose survival is its step
        sum to rounding. For a baseline defined below zero (``Normal``,
        ``Gumbel``, ``Logistic``) the mean without ``given`` also takes off
        :math:`\int_{-\infty}^0 F(t)\, dt`, the covariate held at its value
        at 0 before it, as :meth:`sf_tvc` does.

        A path can stop units from failing: a hazard that dies away (a
        stress driven to a level with no failures, an additive hazard
        driven to 0) leaves the survival levelling off above 0. A fraction
        of units then never fails and the mean is infinite: ``inf`` is
        returned, with a warning that gives the survival where the
        integration stopped, as a univariate model's ``mean()`` returns
        ``inf`` for a limited-failure population. An additive hazard can
        also turn negative, and survival rise above 1 (#376): the mean is
        then infinite as well, or undefined (``nan``, with a warning)
        where the failure probability before time 0 grows without
        limit.

        Parameters
        ----------
        Z : StepSchedule, CovariatePath or array_like
            The covariate path, as for :meth:`sf_tvc`.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            An age survived to: the mean remaining life from it. A ``nan``
            ``given`` gives ``nan``.

        Returns
        -------
        float
            The mean (residual) life along the path.

        Examples
        --------
        At a constant covariate the mean of a Weibull proportional hazards
        model is that of a Weibull with scale
        :math:`\alpha e^{-\beta z / \gamma}` (shape :math:`\gamma`):

        >>> import numpy as np
        >>> from scipy.special import gamma
        >>> from surpyval import CovariatePath, StepSchedule, WeibullPH
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 1))
        >>> x = 100 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> alpha, shape, beta = model.params
        >>> exact = alpha * np.exp(-beta * 0.5 / shape) * gamma(1 + 1 / shape)
        >>> mean = model.mean_tvc(StepSchedule.constant([0.5]))
        >>> bool(np.isclose(mean, exact, rtol=1e-9))
        True

        Along a stress ramped from 0 to 1 over 50 hours the mean lies
        between those at the two ends, and a unit that has survived the
        ramp has less left:

        >>> ramp = CovariatePath.from_points([0, 50], [0.0, 1.0])
        >>> round(model.mean_tvc(ramp), 2)
        58.92
        >>> round(model.mean_tvc(StepSchedule.constant([0.0])), 2)
        101.32
        >>> round(model.mean_tvc(StepSchedule.constant([1.0])), 2)
        51.42
        >>> round(model.mean_tvc(ramp, given=50), 2)
        24.07
        """
        from .tvc_path import (
            _TAIL_KNOTS,
            CovariatePath,
            integrate_to_infinity,
            step_path,
        )

        self._check_tvc_evaluable()
        path = self._to_schedule(Z, xl)
        if not isinstance(path, CovariatePath):
            path = step_path(path)
        g = None if given is None else float(given)
        if g is not None and np.isnan(g):
            return np.nan
        origin = 0.0 if g is None else g
        # What the passes along the path warn of, gathered into one
        # warning each.
        seen = {"falls": 0, "points": 0, "missed": 0, "total": 0}
        worst: list = []

        def sf_at(t: npt.NDArray) -> npt.NDArray:
            H, falls, accuracy = self._tvc_hf_path(
                np.asarray(t, dtype=float), path, g
            )
            seen["falls"] += falls
            seen["points"] += H.size
            if accuracy is not None:
                seen["missed"] += accuracy[0]
                seen["total"] += accuracy[1]
                worst.append(accuracy[2:])
            with np.errstate(over="ignore"):
                # A falling additive hazard can send sf above 1 without
                # limit: the mean is then infinite (below).
                return np.exp(-H)

        params, center = self._tvc_theta(None)
        zc = self._centred(
            path._values(np.array([max(origin, 0.0)]), left=False), center
        )
        scale = self._tvc_time_scale(origin, zc, params)

        def knots(t_max: float) -> npt.NDArray:
            # The path's kinks and jumps as outer panel edges, unless it
            # repeats too often for that to help.
            if path._n_breakpoints(t_max) > _TAIL_KNOTS:
                return np.empty(0)
            return path.breakpoints(t_max)

        value, tail = integrate_to_infinity(
            sf_at, origin, scale, self._tvc_rtol, knots
        )
        if g is None and float(self.distribution.support[0]) < 0:
            # Less the area under F before 0, where the value at 0 holds.
            def ff_below(t: npt.NDArray) -> npt.NDArray:
                with np.errstate(all="ignore"):
                    return np.asarray(
                        self.model.ff(-t, zc, *params), dtype=float
                    ).ravel()

            below, below_tail = integrate_to_infinity(
                ff_below, 0.0, scale, self._tvc_rtol
            )
            value -= below
            if below_tail is not None:
                # F does not fall away before 0: an additive hazard that
                # is negative there (#376) makes it grow without limit.
                value = np.nan
        if seen["falls"]:
            self._warn_negative_hazard(
                seen["falls"], seen["points"], None, stacklevel=3
            )
        if seen["missed"]:
            from .tvc_path import warn_missed_target

            rel = max(w[0] for w in worst)
            limit = (
                "panels"
                if any(w[1] == "panels" for w in worst)
                else ("rounds")
            )
            warn_missed_target(
                seen["missed"],
                seen["total"],
                rel,
                limit,
                self._tvc_rtol,
                stacklevel=3,
            )
        if tail is not None:
            at, sf_end = tail
            warnings.warn(
                "The survival along this path has not fallen to 0: it is "
                "still {:.4g} at t = {:.4g}, so a fraction of units never "
                "fails along it (the hazard dies away, or an additive "
                "hazard turns negative) and the mean {}life is infinite; "
                "inf is returned, as a univariate model's mean() does for "
                "a limited-failure population. sf_tvc gives the survival "
                "along the path.".format(
                    sf_end, at, "residual " if g is not None else ""
                ),
                stacklevel=2,
            )
            return np.inf
        if np.isnan(value):
            warnings.warn(
                "The mean is undefined: before time 0, where the "
                "covariate's value at 0 holds, the failure probability of "
                "this additive hazards model grows without limit (its "
                "hazard is negative there), so the area it takes off the "
                "mean does not converge; nan is returned. The mean "
                "residual life (given=) avoids the times before 0.",
                stacklevel=2,
            )
        return float(value)

    def _tvc_time_scale(
        self, origin: float, zc: npt.NDArray, params: npt.NDArray
    ) -> float:
        """A time scale for the integral to infinity from ``origin``: how
        long the cumulative hazard takes to grow by 1 with the covariate
        held at ``zc`` (a power of 2; 1 where it never does)."""
        u = 2.0 ** np.arange(-60, 61)
        with np.errstate(all="ignore"):
            t = np.append(origin + u, origin)
            H = np.asarray(self.model.Hf(t, zc, *params), dtype=float)
            H = np.broadcast_to(H.ravel(), t.shape)
            grown = H[:-1] - H[-1]
        ok = np.isfinite(grown) & (grown >= 1.0)
        return float(u[np.argmax(ok)]) if ok.any() else 1.0

    @keeps_query_shape
    def cb_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the survival, failure probability or
        cumulative hazard along a covariate path ``Z(t)``: a step schedule
        or a continuously varying path, as for :meth:`sf_tvc`.

        The bounds are those of :meth:`cb`, carried along the path. With
        ``method="wald"`` (the default), a Wald bound on the baseline
        family's probability-plot scale, formed from the cumulative hazard
        of :meth:`Hf_tvc`, its standard error propagated from the fitted
        parameter covariance by the delta method. With ``method="lr"``,
        the likelihood-ratio bound of :meth:`cb`: at each ``x`` the extreme
        of the function along the path over the likelihood region of all
        the parameters (#617), about a second a time where the Wald bound
        takes milliseconds; it needs the data the model was fitted to.
        With ``method="bootstrap"``, the percentile interval over the
        parametric bootstrap refits of :meth:`cb` (with the same
        ``n_boot`` and integer ``random_state`` it reuses them); not for
        a model fitted to time-varying covariates, whose resamples would
        need each subject's covariate path.
        Either way ``ff`` and ``Hf`` follow from the same bound, so the
        three agree with each other, and a constant path gives :meth:`cb`
        with the same ``method``.

        Along a
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
        the integral is taken on a quadrature mesh adapted at the fitted
        parameters and then held fixed, so the function differentiated (or
        searched) is smooth in the parameters. The Wald bound's cost is
        ``2k + 1`` evaluations along the path for ``k`` parameters.

        With ``given`` the bounds are on the conditional survival
        :math:`S(x \mid \text{survived to } g)` of :meth:`sf_tvc`: 1, with
        no width, at and before ``given``.

        Parameters
        ----------
        x : array_like
            Times at which to bound the function.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path, as for :meth:`sf_tvc`.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            Condition on survival to this age, as for :meth:`sf_tvc`. A
            ``nan`` ``given`` gives ``nan``.
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound (``'R'`` and ``'F'`` are accepted for
            ``'sf'`` and ``'ff'``). Default ``'sf'``. The hazard and the
            density along a path are not bounded.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.
        method : {'wald', 'lr', 'bootstrap'}, optional
            ``'wald'`` (the default), ``'lr'`` or ``'bootstrap'``, as
            above and as for :meth:`cb` (``'lr'`` also as
            ``'likelihood'``, ``'likelihood-ratio'`` or ``'profile'``).
        n_boot : int, optional
            The number of bootstrap refits (``method='bootstrap'`` only).
            Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the bootstrap (``method='bootstrap'`` only), as for
            :meth:`cb`.

        Returns
        -------
        numpy array
            The confidence bound(s) on ``on`` at each ``x``: the query's
            shape, with ``[lower, upper]`` on a last axis for two-sided
            bounds.

        Examples
        --------
        A proportional hazards model along a stress ramped from 0 to 1 over
        50 hours:

        >>> import numpy as np
        >>> from surpyval import CovariatePath, WeibullPH
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 1))
        >>> x = 100 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> ramp = CovariatePath.from_points([0, 50], [0.0, 1.0])
        >>> model.sf_tvc([40, 80], ramp).round(4)
        array([0.771 , 0.1911])
        >>> model.cb_tvc([40, 80], ramp).round(4)
        array([[0.7243, 0.8109],
               [0.1256, 0.267 ]])

        A constant path gives the ordinary bounds:

        >>> flat = CovariatePath.from_points([0], [0.5])
        >>> bool(np.allclose(model.cb_tvc([40, 80], flat),
        ...                  model.cb(np.array([40, 80]), [0.5])))
        True
        """
        check_alpha_ci(alpha_ci)
        from ._bootstrap import bound_method, function_bounds, tvc_refits
        from ._likelihood_ratio import cb_tvc_lr, lr_search
        from .tvc_path import CovariatePath

        method = bound_method(method)
        lr = method == "lr"
        self._check_inference()
        check_option(
            "on",
            on,
            ("sf", "R", "ff", "F", "Hf"),
            "cb_tvc bounds the survival, failure probability and cumulative "
            "hazard along a path, not the hazard or the density.",
        )
        check_option("bound", bound, BOUNDS)
        self._check_tvc_evaluable()
        xq: npt.NDArray = np.atleast_1d(np.asarray(x, dtype=float))
        shape = xq.shape + ((2,) if bound == "two-sided" else ())
        g = None if given is None else float(given)
        if (g is not None and np.isnan(g)) or xq.size == 0:
            # A missing conditioning age: nothing is known (as sf_tvc).
            self._to_schedule(Z, xl)
            return np.full(shape, np.nan)
        on_path = isinstance(Z, CovariatePath)
        # In the parameterisation of the centred fit when there is one, as
        # for cb (#463).
        if lr:
            search = lr_search(self, reported=False)
            params, center = search.params, search.center
        elif method == "bootstrap":
            # The refits are of the model's own parameters, each with the
            # covariate point of its baseline.
            fits = tvc_refits(self, n_boot, random_state)
            params, center = self._eval_params(), self.center
        else:
            params, center, cov = self._inference_state()
        # The path's mesh, adapted at the fitted parameters and then held.
        frozen: dict = {}

        def H_of(p: npt.NDArray, at: Any = None) -> npt.NDArray:
            theta = (p, center if at is None else at)
            if on_path or g is None:
                H = self._hf_tvc(xq, Z, xl, g, theta, frozen)[0]
            else:
                H = (
                    self._hf_tvc(xq, Z, xl, None, theta)[0]
                    - self._hf_tvc(g, Z, xl, None, theta)[0]
                )
            if g is not None:
                # Survival to x <= g is certain (as sf_tvc, #523).
                H = np.where(xq <= g, 0.0, H)
            return H

        # The estimate first: it adapts the mesh, and gives sf_tvc's
        # warnings (a falling additive hazard, a missed target).
        H, falls, accuracy = self._hf_tvc(
            xq, Z, xl, g if on_path else None, (params, center), frozen
        )
        self._warn_tvc(H, falls, accuracy, stacklevel=5)
        if lr:
            return cb_tvc_lr(
                search, xq, H_of, H_of(params), on, alpha_ci, bound
            )
        if method == "bootstrap":
            on = {"R": "sf", "F": "ff"}.get(on, on)
            return function_bounds(self, fits, H_of, on, alpha_ci, bound)
        return self._sf_bounds(
            H_of,
            lambda p: np.exp(-H_of(p)),
            params,
            cov,
            xq.shape,
            on,
            alpha_ci,
            bound,
        )
