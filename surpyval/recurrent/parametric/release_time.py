"""The cheapest time to stop testing and release (Okumoto and Goel 1980),
for a fitted Poisson process.

With ``c1`` the cost of fixing a fault found in testing, ``c2`` that of
one found in the field and ``c3`` the cost of a unit of testing time,
releasing at ``T`` costs, in expectation,

.. math::
    C(T) = c_1 \\Lambda(T) + c_2 \\left(\\Lambda(T_{life}) -
    \\Lambda(T)\\right) + c_3 T,

where the software is in use until ``T_life`` on the test's clock. Its
slope is ``c3 - (c2 - c1) lambda(T)``: testing pays while the intensity
is above ``c3 / (c2 - c1)`` and costs more than it saves afterwards, so
the cheapest release is when the intensity has fallen to that level.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import brentq

from surpyval.recurrent.inference import check_alpha_ci
from surpyval.utils.linalg import delta_method_se, log_transformed_cb
from surpyval.utils.rng import as_generator
from surpyval.utils.validation import BOUNDS, check_option
from surpyval.utils.warnings import ignoring_warnings

from .growth_answers import NeverReached, last_time_above


class _Plan:
    """A release question (costs, life, reliability target) asked of a
    process ``dist`` at parameters ``params``."""

    def __init__(
        self,
        dist: Any,
        fix_cost: float,
        field_cost: float,
        test_cost: float,
        life: float,
        gained_level: float | None,
        mission: float | None,
    ) -> None:
        self.dist = dist
        self.c1, self.c2, self.c3 = fix_cost, field_cost, test_cost
        self.life = life
        self.gained_level, self.mission = gained_level, mission

    def cif(self, t: Any, params: Any) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(self.dist.cif(t, *params), dtype=float)

    def iif(self, t: Any, params: Any) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(self.dist.iif(t, *params), dtype=float)

    def scale(self, params: Any) -> float:
        """A time of the model's own size (as ``_time_scale``)."""
        total = float(self.cif(np.inf, params))
        count = 1.0 if not total < 2.0 else 0.5 * total
        with np.errstate(all="ignore"):
            return float(np.asarray(self.dist.inv_cif(count, *params)))

    def cost(self, t: Any, params: Any) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        found = self.cif(t, params)
        return (
            self.c1 * found
            + self.c2 * (self.cif(self.life, params) - found)
            + self.c3 * t
        )

    def cost_time(self, params: Any) -> float:
        """The release time of least expected cost: 0, or where the
        intensity comes down to ``c3 / (c2 - c1)`` (the later, on an
        S-shaped curve), whichever costs less; no later than the life.
        Raises ``NeverReached`` where the intensity does not come down."""
        level = self.c3 / (self.c2 - self.c1)
        t = last_time_above(
            lambda s: self.iif(s, params), level, self.scale(params)
        )
        t = min(t, self.life)
        if t > 0 and float(self.cost(0.0, params)) <= float(
            self.cost(t, params)
        ):
            return 0.0
        return t

    def reliability_time(self, params: Any) -> float:
        """The time from which the mission meets the reliability."""
        assert self.gained_level is not None and self.mission is not None
        mission = self.mission
        return last_time_above(
            lambda s: self.cif(np.asarray(s) + mission, params)
            - self.cif(s, params),
            self.gained_level,
            self.scale(params),
        )

    def time(self, params: Any) -> float:
        """The release time: the later of the cheapest and the time the
        reliability target is met."""
        t = self.cost_time(params)
        if self.gained_level is not None:
            t = max(t, self.reliability_time(params))
        return t

    def band(self, params: Any, within: float) -> tuple[float, float]:
        """The release times whose expected cost is within the share
        ``within`` of the least."""
        best = self.cost_time(params)
        least = float(self.cost(best, params))
        ceiling = least + within * abs(least)

        def over(t: float) -> float:
            return float(self.cost(t, params)) - ceiling

        lower = 0.0
        if best > 0 and over(0.0) > 0:
            # Bisected from the optimum down: on an S-shaped curve the
            # cost can dip under the ceiling again near 0, which is not
            # part of the stretch around the optimum.
            lo, hi = 0.0, best
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if over(mid) > 0 else (lo, mid)
            lower = hi
        step = best if best > 0 else self.scale(params)
        hi = best + step
        while over(hi) <= 0 and hi < self.life and hi < 1e300:
            step *= 2.0
            hi = best + step
        hi = min(hi, self.life)
        if over(hi) <= 0:
            return lower, hi
        return lower, float(brentq(over, best, hi, xtol=1e-300, rtol=1e-12))


class ReleaseTimeMixin:
    """The best time to release, for a fitted Poisson process."""

    dist: Any
    params: np.ndarray
    data: Any
    _mle: np.ndarray
    covariance: Callable
    _check_inference: Callable
    _check_has_data: Callable
    _check_fitted: Callable
    _positive: Callable
    _total: Callable

    def _release_plan(
        self,
        test_cost: float,
        fix_cost: float,
        field_cost: float,
        life: float | None,
        reliability: float | None,
        mission: float | None,
    ) -> _Plan:
        c3 = self._positive(test_cost, "test_cost")
        costs = {"fix_cost": fix_cost, "field_cost": field_cost}
        for name, value in costs.items():
            try:
                ok = float(value) >= 0 and np.isfinite(float(value))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(
                    f"{name} must be a cost of at least 0; got {value!r}"
                )
        c1, c2 = float(fix_cost), float(field_cost)
        if not c2 > c1:
            raise ValueError(
                "field_cost ({:g}) is not above fix_cost ({:g}): a fault "
                "costs no more to fix after release than in testing, so "
                "no amount of testing pays for itself and there is no "
                "best release time. Check the costs: field_cost is the "
                "whole cost of a fault that reaches use (its failures, "
                "the fix and its delivery).".format(c2, c1)
            )
        if life is None:
            if not np.isfinite(self._total()):
                raise ValueError(
                    "The fitted {} model expects failures without limit, "
                    "so the cost of the faults left to the field is "
                    "unbounded without an end of use: give life=, the "
                    "time on the test's clock (testing plus use) at which "
                    "the software goes out of service.".format(self.dist.name)
                )
            horizon = np.inf
        else:
            horizon = self._positive(life, "life")
        if (reliability is None) != (mission is None):
            raise ValueError(
                "reliability and mission go together: the chance "
                "(reliability) of running a stretch of use (mission) "
                "without a failure. Give both, or neither."
            )
        level = None
        length = None
        if reliability is not None:
            level = -float(
                np.log(self._positive(reliability, "reliability", below=1.0))
            )
            length = self._positive(mission, "mission")
        return _Plan(self.dist, c1, c2, c3, horizon, level, length)

    def _never_pays(self, plan: _Plan, never: NeverReached) -> ValueError:
        how = "is rising" if str(never) == "rising" else "does not fall"
        return ValueError(
            "The fitted {name} model has no best release time: its "
            "intensity {how} to test_cost / (field_cost - fix_cost) = "
            "{level:g}, the failure rate below which a unit of testing "
            "costs more than the faults it finds would cost in the field"
            "{target}. On this model more testing always pays, which "
            "says the model (or the data) shows no reliability growth: "
            "check trend_test(), or fit a growth model.".format(
                name=self.dist.name,
                how=how,
                level=plan.c3 / (plan.c2 - plan.c1),
                target=(
                    ""
                    if plan.gained_level is None
                    else " (or the reliability target is never met)"
                ),
            )
        )

    def release_time(
        self,
        test_cost: float,
        fix_cost: float,
        field_cost: float,
        *,
        life: float | None = None,
        reliability: float | None = None,
        mission: float | None = None,
        within: float = 0.01,
    ) -> dict:
        """
        The testing time at which to release for the least expected
        total cost (Okumoto and Goel 1980), optionally no earlier than
        a reliability target is met.

        Releasing after testing to ``T`` is expected to cost
        ``fix_cost * cif(T) + field_cost * (cif(life) - cif(T)) +
        test_cost * T``: the faults found in testing, those left to be
        found in use, and the testing itself. Testing pays while the
        intensity is above ``test_cost / (field_cost - fix_cost)``, so
        the cheapest release is when it has fallen to that level (0
        where it never is above it: release without testing).

        Parameters
        ----------
        test_cost : float
            The cost of one unit of testing time (the model's unit),
            above 0. Any currency, the same for the three costs.
        fix_cost : float
            The cost of a fault found and fixed in testing.
        field_cost : float
            The cost of a fault found in use, above ``fix_cost``.
        life : float, optional
            The time on the test's clock (testing plus use) at which the
            software goes out of service. By default it is used for
            ever, which needs a model with a finite number of faults; a
            model without one (``MusaOkumoto``, ``CrowAMSAA``) raises a
            ``ValueError`` asking for it. Keyword only, as those below.
        reliability, mission : float, optional
            A target the release must meet as well: the chance
            ``reliability`` of running a ``mission`` of use without a
            failure (as :meth:`time_to_reliability`). The release is
            then the later of the cheapest time and the time the target
            is met.
        within : float, optional
            The share of the least cost that counts as near it, for the
            band of near-best times. Default 0.01 (1%).

        Returns
        -------
        dict
            ``"time"``: the release time, on the model's clock (subtract
            the testing done to get what is left to do); ``"cost"``: the
            expected total cost of releasing then; ``"binding"``:
            ``"cost"``, ``"reliability"`` (the target sets the time) or
            ``"life"`` (testing would pay to the end of the life);
            ``"cost_time"`` and ``"least_cost"``: the cheapest time and
            its cost, whatever the target; ``"reliability_time"``: the
            time the target is met (``None`` without one); ``"band"``:
            ``(earliest, latest)``, the release times whose expected
            cost is within ``within`` of the least. A wide band says the
            cost hardly depends on the day chosen.

        Raises
        ------
        ValueError
            Where ``field_cost`` is not above ``fix_cost``, or the
            fitted intensity never falls to the level at which testing
            stops paying (a constant or rising intensity), saying which.

        References
        ----------
        Okumoto, K. and Goel, A. L. (1980), "Optimum release time for
        software systems based on reliability and cost criteria", Journal
        of Systems and Software 1, 315-318.

        Examples
        --------
        A Goel-Okumoto model with 100 faults: a day of testing costs
        50, a fault 10 to fix in testing and 500 in the field. The
        cheapest release is at ``log(a b (c2 - c1) / c3) / b``:

        >>> from surpyval.recurrent import GoelOkumoto
        >>> model = GoelOkumoto.from_params([100, 0.05])
        >>> plan = model.release_time(50, 10, 500)
        >>> round(plan["time"], 2), round(plan["cost"])
        (77.84, 5892)
        >>> [round(t, 1) for t in plan["band"]]
        [71.3, 85.1]

        With a 95% chance of 10 days without a failure required as
        well, the release waits for the target:

        >>> plan = model.release_time(50, 10, 500, reliability=0.95,
        ...                           mission=10.0)
        >>> round(plan["time"], 2), plan["binding"]
        (132.85, 'reliability')
        """
        plan = self._release_plan(
            test_cost, fix_cost, field_cost, life, reliability, mission
        )
        share = self._positive(within, "within")
        params = np.asarray(self.params, dtype=float)
        try:
            cheapest = plan.cost_time(params)
            met = (
                None
                if plan.gained_level is None
                else plan.reliability_time(params)
            )
        except NeverReached as never:
            raise self._never_pays(plan, never) from None
        time = cheapest if met is None else max(cheapest, met)
        if met is not None and met > cheapest:
            binding = "reliability"
        elif cheapest >= plan.life:
            binding = "life"
        else:
            binding = "cost"
        return {
            "time": float(time),
            "cost": float(plan.cost(time, params)),
            "binding": binding,
            "cost_time": float(cheapest),
            "least_cost": float(plan.cost(cheapest, params)),
            "reliability_time": None if met is None else float(met),
            "band": plan.band(params, share),
        }

    def release_time_cb(
        self,
        test_cost: float,
        fix_cost: float,
        field_cost: float,
        *,
        life: float | None = None,
        reliability: float | None = None,
        mission: float | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> Any:
        """
        Confidence bounds on the release time of :meth:`release_time`
        (its ``"time"``), which is uncertain because the fitted model
        is.

        Parameters
        ----------
        test_cost, fix_cost, field_cost, life, reliability, mission :
            As for :meth:`release_time`.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are an array ``[lower, upper]``; a
            one-sided bound a number.
        method : {'wald', 'bootstrap'}, optional
            ``"wald"`` (default): the delta method on the log of the
            release time, from the parameter covariance.

            ``"bootstrap"``: a parametric bootstrap. Data are simulated
            from the fitted model over each system's observation, the
            model is fitted again and the release time worked out again,
            ``n_boot`` times; the bounds are the percentiles of those
            times. A simulated data set with no best release time (one
            that shows no growth, or whose intensity never falls far
            enough) counts as a release time of ``inf``, so an upper
            bound of ``inf`` says that such data sets are not rare. It
            needs the failure times (not counts in periods).
        n_boot : int, optional
            The bootstrap's data sets. Default 200.
        random_state : int or numpy.random.Generator, optional
            The bootstrap's seed, for a reproducible bound.

        Examples
        --------
        >>> from surpyval.recurrent import GoelOkumoto
        >>> truth = GoelOkumoto.from_params([100, 0.05])
        >>> d = truth.time_terminated_simulation_data(60.0, random_state=1)
        >>> model = GoelOkumoto.fit(d.x, c=d.c)
        >>> round(model.release_time(50, 10, 500)["time"], 1)
        67.4
        >>> model.release_time_cb(50, 10, 500).round(1)
        array([55.1, 82.4])
        >>> model.release_time_cb(50, 10, 500, method="bootstrap",
        ...                       n_boot=50, random_state=1).round(1)
        array([54.9, 79.9])
        """
        check_option("method", method, ("wald", "bootstrap"))
        check_option("bound", bound, BOUNDS)
        check_alpha_ci(alpha_ci)
        plan = self._release_plan(
            test_cost, fix_cost, field_cost, life, reliability, mission
        )
        try:
            estimate = plan.time(np.asarray(self.params, dtype=float))
        except NeverReached as never:
            raise self._never_pays(plan, never) from None
        if method == "bootstrap":
            return self._release_bootstrap(
                plan, alpha_ci, bound, n_boot, random_state
            )
        self._check_inference()

        def time(params: np.ndarray) -> np.ndarray:
            try:
                return np.array([plan.time(params)])
            except (NeverReached, ValueError):
                return np.array([np.nan])

        se = delta_method_se(time, self._mle, self.covariance(), True)
        cb = log_transformed_cb(np.array([estimate]), se, alpha_ci, bound)
        return cb[0]

    def _release_bootstrap(
        self,
        plan: _Plan,
        alpha_ci: float,
        bound: str,
        n_boot: int,
        random_state: Any,
    ) -> Any:
        from surpyval.recurrent import diagnostics

        self._check_has_data("A bootstrap bound on the release time")
        self._check_fitted()
        diagnostics.validate_diagnostic_data(
            self.data, "A bootstrap bound on the release time"
        )
        if not (isinstance(n_boot, (int, np.integer)) and n_boot >= 20):
            raise ValueError(
                "n_boot is the number of simulated data sets, a whole "
                f"number of at least 20; got {n_boot!r}"
            )
        rng = as_generator(random_state)
        windows = diagnostics._per_item_windows(self.data)
        params = np.asarray(self.params, dtype=float)

        def mass(item: Any, entry: float, close: float) -> tuple:
            start = float(plan.cif(entry, params))
            return start, float(plan.cif(close, params)) - start

        def inv_cif(item: Any) -> Callable:
            return lambda count: self.dist.inv_cif(count, *params)

        times = np.full(n_boot, np.inf)
        for k in range(n_boot):
            try:
                with ignoring_warnings():
                    sim = diagnostics._simulate_windows(
                        rng, windows, mass, inv_cif
                    )
                    refit = self.dist._fit_from_recurrent_data(sim)
                times[k] = plan.time(np.asarray(refit.params, dtype=float))
            except (NeverReached, ValueError):
                continue  # no best release time: inf
        tail = alpha_ci / 2.0 if bound == "two-sided" else alpha_ci
        lower = float(np.quantile(times, tail, method="lower"))
        upper = float(np.quantile(times, 1.0 - tail, method="higher"))
        if bound == "two-sided":
            return np.array([lower, upper])
        return lower if bound == "lower" else upper
