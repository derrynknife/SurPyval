"""Planning an accelerated life test (#864): which stress levels to test
at, and how many units at each, to estimate a quantile of the life at the
use stress as precisely as the test allows.

The model is the one :class:`surpyval.AcceleratedLife` fits: a Weibull or
lognormal lifetime whose life parameter is a life-stress model ``L(V)``
(Arrhenius, power, Eyring, ...), with a shape that does not change with
stress. Every unit is tested until ``censor_time``. For planning values
of the model's parameters, the large-sample variance of the log of the
estimated quantile at the use stress follows from the test's expected
Fisher information (Meeker and Escobar, *Statistical Methods for
Reliability Data*, ch. 20; Nelson, *Accelerated Testing*, ch. 6):

- :func:`accelerated_test_precision` gives the precision of a test you
  propose;
- :func:`accelerated_test_plan` finds the best test with the highest
  stress fixed: the statistically optimum plan of two levels, or a
  compromise plan of three or more equally spaced levels (Meeker and
  Hahn, 1985), which tests a share of the units in between and so is
  less hurt when the life-stress model is wrong.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import autograd
import numpy as np
from autograd import numpy as anp
from scipy import optimize, stats

from surpyval.test_planning import _information as info
from surpyval.test_planning._checks import (
    check_count,
    check_positive,
    check_probability,
)
from surpyval.test_planning._serialise import dataclass_dict
from surpyval.test_planning.estimation import (
    SimulatedPrecision,
    _simulated,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.validation import check_alpha_ci


@dataclass(frozen=True, repr=False)
class AcceleratedTestPlan:
    """An accelerated life test and the precision with which it estimates
    a quantile at the use stress, from :func:`accelerated_test_plan` and
    :func:`accelerated_test_precision`.

    Attributes
    ----------
    distribution : str
        The lifetime distribution, ``"Weibull"`` or ``"LogNormal"``.
    life_model : str
        The life-stress model's name.
    life_params : dict
        Its planning values, by name.
    shape : float
        The planning value of the shape: the Weibull's ``beta`` or the
        lognormal's ``sigma``.
    use_stress : float
        The stress the quantile is estimated at.
    censor_time : float
        When the test stops.
    stresses : tuple of float
        The test stresses, lowest first.
    units : tuple of int
        The units to test at each.
    fractions : tuple of float
        The shares of the units at each, before rounding to whole units.
    expected_failures : tuple of float
        The failures each stress expects by ``censor_time``.
    p : float
        The probability of the quantile estimated.
    estimate : float
        Its planned value at the use stress.
    standard_error : float
        The large-sample standard error of the log of its estimate.
    precision : float
        The precision factor ``R``: the ``1 - alpha_ci`` interval is
        ``[estimate / R, estimate * R]``.
    alpha_ci : float
        The significance level of that interval.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval import life_models
    >>> from surpyval.test_planning import accelerated_test_precision
    >>> plan = accelerated_test_precision(
    ...     surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0}, 2.0,
    ...     use_stress=10.0, stresses=(30.0, 45.0, 60.0),
    ...     units=(20, 10, 10), censor_time=1000.0)
    >>> plan.units, [round(f, 1) for f in plan.expected_failures]
    ((20, 10, 10), [11.1, 9.8, 10.0])
    """

    distribution: str
    life_model: str
    life_params: dict
    shape: float
    use_stress: float
    censor_time: float
    stresses: tuple[float, ...]
    units: tuple[int, ...]
    fractions: tuple[float, ...]
    expected_failures: tuple[float, ...]
    p: float
    estimate: float
    standard_error: float
    precision: float
    alpha_ci: float

    def to_dict(self) -> dict:
        """The plan as a dict of native Python values, strict JSON: every
        attribute (the tuples as lists, ``life_params`` as a dict), under
        ``"result": "AcceleratedTestPlan"``. An infinite or missing number
        is ``None``, recorded under ``"non_finite"``.

        Examples
        --------
        >>> import json
        >>> import surpyval
        >>> from surpyval import life_models
        >>> from surpyval.test_planning import accelerated_test_precision
        >>> plan = accelerated_test_precision(
        ...     surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0},
        ...     2.0, use_stress=10.0, stresses=(40.0, 60.0),
        ...     units=(30, 20), censor_time=1000.0)
        >>> d = plan.to_dict()
        >>> d["stresses"], d["units"], d["life_params"]
        ([40.0, 60.0], [30, 20], {'a': 1000000.0, 'n': -2.0})
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        """
        return dataclass_dict(self)

    def simulate(
        self, n_sim: int = 200, *, random_state: Any = None
    ) -> SimulatedPrecision:
        """Simulate the test ``n_sim`` times, fit the accelerated life
        model to each simulated data set with
        :class:`surpyval.AcceleratedLife`, and report the precision its
        estimates of the quantile at the use stress had.

        Parameters
        ----------
        n_sim : int, optional
            The number of simulated tests, by default 200.
        random_state : optional
            Keyword only: the seed, as every SurPyval draw takes it.

        Returns
        -------
        SimulatedPrecision

        Examples
        --------
        >>> import surpyval
        >>> from surpyval import life_models
        >>> from surpyval.test_planning import accelerated_test_precision
        >>> plan = accelerated_test_precision(
        ...     surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0},
        ...     2.0, use_stress=10.0, stresses=(40.0, 60.0),
        ...     units=(30, 20), censor_time=1000.0)
        >>> plan.simulate(50, random_state=3).within > 0.8
        True
        """
        import surpyval
        from surpyval import life_models

        n_sim = check_count(n_sim, "n_sim")
        rng = as_generator(random_state)
        fitter = getattr(surpyval, self.distribution)
        model = getattr(life_models, self.life_model)
        values = list(self.life_params.values())
        Z = np.repeat(self.stresses, self.units)
        lives = np.asarray(model.phi(Z, *values), dtype=float)
        logs, unusable = [], 0
        for _ in range(n_sim):
            u = rng.uniform(size=Z.size)
            if self.distribution == "Weibull":
                x = lives * (-np.log1p(-u)) ** (1.0 / self.shape)
            else:
                x = lives * np.exp(self.shape * stats.norm.ppf(u))
            c = (x > self.censor_time).astype(int)
            x = np.minimum(x, self.censor_time)
            if np.sum(c == 0) < 3:
                unusable += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = surpyval.AcceleratedLife(fitter, model).fit(x, Z, c=c)
            value = fitted.qf(self.p, self.use_stress)
            logs.append(float(np.log(np.asarray(value).item())))
        return _simulated(
            np.array(logs),
            np.log(self.estimate),
            self.precision,
            self.alpha_ci,
            unusable,
        )

    def __repr__(self) -> str:
        header = "Accelerated Test Plan"
        params = ", ".join(f"{k}={v:.6g}" for k, v in self.life_params.items())
        shape = "beta" if self.distribution == "Weibull" else "sigma"
        level = f"{100 * (1 - self.alpha_ci):.6g}%"
        lines = [
            header,
            "=" * len(header),
            f"Model            : {self.distribution}, "
            f"{self.life_model}({params}), {shape}={self.shape:.6g}",
            f"Censor time      : {self.censor_time:.6g}",
            "Stress    Units  Expected failures",
        ]
        for stress, units, failures in zip(
            self.stresses, self.units, self.expected_failures
        ):
            lines.append(f"{stress:<9.6g} {units:<6d} {failures:.3g}")
        lines += [
            f"Estimating       : B{100 * self.p:.6g} at {self.use_stress:.6g}"
            f" = {self.estimate:.6g}",
            f"Precision factor : {self.precision:.6g} ({level} interval)",
        ]
        return "\n".join(lines)


class _Model:
    """The planning model: the log life at each stress, its gradient in
    the life-stress parameters, and the information of a test."""

    def __init__(
        self,
        distribution: Any,
        life_model: Any,
        life_params: Any,
        shape: float,
        censor_time: float,
    ) -> None:
        self.family = info.family_of(distribution)
        self.distribution = distribution.name
        if getattr(life_model, "n_stresses", None) != 1 or not hasattr(
            life_model, "phi_param_map"
        ):
            raise ValueError(
                "life_model must be a single-stress life model from "
                "surpyval.life_models (Power, Arrhenius, Eyring, ...), got "
                f"{life_model!r}."
            )
        self.life_model = life_model
        self.names = list(life_model.phi_param_map)
        self.values = _life_values(life_params, self.names)
        shape = check_positive(shape, "shape")
        self.shape = shape
        self.sigma = 1.0 / shape if self.family == "sev" else shape
        self.censor_time = check_positive(censor_time, "censor_time")
        self._gradient = autograd.grad(self._log_life_of)
        #: The stress the quantile is estimated at, set once checked.
        self.use_stress = float("nan")

    def _log_life_of(self, values: Any, stress: float) -> Any:
        Z = anp.array([stress])
        return anp.sum(anp.log(self.life_model.phi(Z, *values)))

    def log_life(self, stress: float) -> float:
        value = self._log_life_of(self.values, stress)
        if not np.isfinite(value):
            raise ValueError(
                f"The life model gives no positive life at stress "
                f"{stress!r} with these planning values."
            )
        return float(value)

    def information(self, stress: float) -> np.ndarray:
        """One unit's information about (life parameters, sigma)."""
        zeta = (np.log(self.censor_time) - self.log_life(stress)) / self.sigma
        unit = info.unit_information(self.family, zeta) / self.sigma**2
        k = len(self.values)
        jacobian = np.zeros((2, k + 1))
        jacobian[0, :k] = self._gradient(self.values, stress)
        jacobian[1, k] = 1.0
        return jacobian.T @ unit @ jacobian

    def failure_chance(self, stress: float) -> float:
        zeta = (np.log(self.censor_time) - self.log_life(stress)) / self.sigma
        return float(info.cdf(self.family, zeta))

    def variance(
        self, stresses: Sequence[float], weights: Sequence[float], p: float
    ) -> float:
        """The variance of the log quantile at the use stress, per unit
        tested, for the shares ``weights`` at ``stresses``."""
        total = sum(w * self.information(s) for s, w in zip(stresses, weights))
        gradient = np.append(
            self._gradient(self.values, self.use_stress),
            float(info.ppf(self.family, p)),
        )
        try:
            solved = np.linalg.solve(total, gradient)
        except np.linalg.LinAlgError:
            return float(np.inf)
        value = float(gradient @ solved)
        return value if value > 0 else float(np.inf)

    def stress_at(self, position: float, high: float) -> float:
        """The stress at ``position`` of the way from the use stress
        (0) to ``high`` (1), on the scale of the planned log life."""
        if position >= 1.0:
            return high
        use, top = self.log_life(self.use_stress), self.log_life(high)
        target = use + position * (top - use)
        lower, upper = sorted((self.use_stress, high))
        return float(
            optimize.brentq(
                lambda s: self.log_life(s) - target,
                lower,
                upper,
                xtol=1e-12 * max(abs(high), 1.0),
            )
        )


def _life_values(life_params: Any, names: list[str]) -> list[float]:
    """The life model's planning values in its parameter order."""
    if isinstance(life_params, Mapping):
        if set(life_params) != set(names):
            raise ValueError(
                f"life_params must name the life model's parameters "
                f"({', '.join(names)}), got {sorted(life_params)}."
            )
        values = [life_params[name] for name in names]
    else:
        values = list(np.ravel(life_params))
        if len(values) != len(names):
            raise ValueError(
                f"life_params must be the life model's {len(names)} "
                f"parameters ({', '.join(names)}), got {life_params!r}."
            )
    floats = [float(v) for v in values]
    if not np.all(np.isfinite(floats)):
        raise ValueError(f"life_params must be finite, got {life_params!r}.")
    return floats


def _whole_units(n: int, fractions: Sequence[float]) -> tuple[int, ...]:
    """``n`` units shared out as ``fractions``, by largest remainder, with
    at least one at each stress (a planned level is tested)."""
    exact = n * np.asarray(fractions, dtype=float)
    units = np.floor(exact).astype(int)
    order = np.argsort(-(exact - units), kind="stable")
    units[order[: n - units.sum()]] += 1
    while units.min() == 0:
        units[units.argmax()] -= 1
        units[units.argmin()] += 1
    return tuple(int(u) for u in units)


def _finish(
    model: _Model,
    stresses: Sequence[float],
    units: Sequence[int],
    fractions: Sequence[float],
    p: float,
    alpha_ci: float,
) -> AcceleratedTestPlan:
    n = sum(units)
    if sum(u > 0 for u in units) < len(model.values):
        raise ValueError(
            f"The {model.life_model.name} model has {len(model.values)} "
            "parameters: test units at as many stresses at least."
        )
    variance = model.variance(stresses, [u / n for u in units], p)
    if not np.isfinite(variance):
        raise ValueError(
            "This test cannot estimate the model: its information is "
            "singular (too few failures, or too few stresses)."
        )
    error = float(np.sqrt(variance / n))
    z = float(stats.norm.isf(alpha_ci / 2.0))
    log_quantile = model.log_life(model.use_stress) + model.sigma * float(
        info.ppf(model.family, p)
    )
    return AcceleratedTestPlan(
        distribution=model.distribution,
        life_model=model.life_model.name,
        life_params=dict(zip(model.names, model.values)),
        shape=model.shape,
        use_stress=model.use_stress,
        censor_time=model.censor_time,
        stresses=tuple(float(s) for s in stresses),
        units=tuple(int(u) for u in units),
        fractions=tuple(float(f) for f in fractions),
        expected_failures=tuple(
            u * model.failure_chance(s) for s, u in zip(stresses, units)
        ),
        p=p,
        estimate=float(np.exp(log_quantile)),
        standard_error=error,
        precision=float(np.exp(z * error)),
        alpha_ci=float(alpha_ci),
    )


def accelerated_test_precision(
    distribution: Any,
    life_model: Any,
    life_params: Any,
    shape: float,
    *,
    use_stress: float,
    stresses: Sequence[float],
    units: Sequence[int],
    censor_time: float,
    p: float = 0.1,
    alpha_ci: float = 0.05,
) -> AcceleratedTestPlan:
    """How precisely a proposed accelerated life test will estimate a
    quantile of the life at the use stress.

    ``units[i]`` units are tested at ``stresses[i]``, each until
    ``censor_time``, and the accelerated life model fitted to the
    results. For planning values of the model, the large-sample
    precision factor ``R`` of the estimated quantile at ``use_stress``:
    its ``1 - alpha_ci`` confidence interval will be about ``[estimate /
    R, estimate * R]``.

    Parameters
    ----------
    distribution : surpyval.Weibull or surpyval.LogNormal
        The lifetime distribution.
    life_model : life model
        A single-stress life model from :mod:`surpyval.life_models`:
        ``Power``, ``InversePower``, ``Arrhenius``, ``Eyring``, ...
    life_params : dict or sequence of float
        The life model's planning values, by name or in its order
        (``life_model.phi_param_map``).
    shape : float
        The planning value of the shape, the same at every stress: the
        Weibull's ``beta`` or the lognormal's ``sigma``.
    use_stress : float
        Keyword only: the stress to estimate the quantile at.
    stresses : sequence of float
        Keyword only: the test stresses.
    units : sequence of int
        Keyword only: the units to test at each.
    censor_time : float
        Keyword only: when the test stops.
    p : float, optional
        Keyword only: the probability of the quantile, by default 0.1
        (the B10 life).
    alpha_ci : float, optional
        Keyword only: the significance level of the interval, by default
        0.05.

    Returns
    -------
    AcceleratedTestPlan
        The test, its expected failures at each stress, and the precision
        of its estimate.

    Raises
    ------
    ValueError
        If an argument is out of range, or the test cannot estimate the
        model (too few stresses with units).

    Examples
    --------
    A power-law voltage test (life ``1e6 V^-2`` hours, Weibull shape 2):
    40 units, 1000 hours, at 30, 45 and 60 V, to estimate the B10 life at
    10 V, about 3,250 hours, to within a factor of about 2.5:

    >>> import surpyval
    >>> from surpyval import life_models
    >>> from surpyval.test_planning import accelerated_test_precision
    >>> plan = accelerated_test_precision(
    ...     surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0}, 2.0,
    ...     use_stress=10.0, stresses=(30.0, 45.0, 60.0),
    ...     units=(20, 10, 10), censor_time=1000.0)
    >>> round(plan.estimate), round(plan.precision, 2)
    (3246, 2.48)
    """
    check_alpha_ci(alpha_ci)
    p = check_probability(p, "p")
    model = _Model(distribution, life_model, life_params, shape, censor_time)
    model.use_stress = _check_stress(use_stress, "use_stress", model)
    levels = [_check_stress(s, "stresses", model) for s in np.ravel(stresses)]
    counts = [check_count(u, "units", minimum=0) for u in np.ravel(units)]
    if len(levels) != len(counts) or not levels:
        raise ValueError(
            "stresses and units must be the same length, one count per "
            f"stress, got {len(levels)} and {len(counts)}."
        )
    order = np.argsort(levels, kind="stable")
    levels = [levels[i] for i in order]
    counts = [counts[i] for i in order]
    total = sum(counts)
    if total == 0:
        raise ValueError("units must test at least one unit.")
    fractions = [c / total for c in counts]
    return _finish(model, levels, counts, fractions, p, alpha_ci)


def accelerated_test_plan(
    distribution: Any,
    life_model: Any,
    life_params: Any,
    shape: float,
    *,
    use_stress: float,
    high_stress: float,
    n: int,
    censor_time: float,
    levels: int = 2,
    allocation: Sequence[float] | None = None,
    middle_fraction: float = 0.2,
    p: float = 0.1,
    alpha_ci: float = 0.05,
) -> AcceleratedTestPlan:
    """The accelerated life test that estimates a quantile at the use
    stress most precisely, with the highest stress fixed.

    The test runs ``n`` units until ``censor_time`` at ``levels``
    stresses, the highest ``high_stress`` and the others equally spaced
    between it and a lowest stress, on the scale of the planned log life
    (for the log-linear life models, equally spaced in ``log V`` for the
    power law and in ``1 / V`` for Arrhenius). The lowest stress, and the
    share of units at it, are chosen to minimise the large-sample
    variance of the estimated quantile at ``use_stress``:

    - with two levels, the statistically optimum plan (Nelson and
      Kielpinski, 1976; Meeker and Escobar, section 20.3);
    - with three or more, a compromise plan (Meeker and Hahn, 1985):
      ``middle_fraction`` of the units are shared equally among the
      middle stresses, which make the plan robust to a curved life-stress
      relation, at a small cost in precision;
    - with ``allocation``, the shares at each stress, lowest first, are
      fixed (``(4/7, 2/7, 1/7)``, the 4:2:1 plan), and only the lowest
      stress is chosen.

    Parameters
    ----------
    distribution : surpyval.Weibull or surpyval.LogNormal
        The lifetime distribution.
    life_model : life model
        A single-stress life model from :mod:`surpyval.life_models`.
    life_params : dict or sequence of float
        The life model's planning values, by name or in its order.
    shape : float
        The planning value of the shape: the Weibull's ``beta`` or the
        lognormal's ``sigma``.
    use_stress : float
        Keyword only: the stress to estimate the quantile at.
    high_stress : float
        Keyword only: the highest stress the test may use, beyond which
        the failure mechanism could change.
    n : int
        Keyword only: the number of units.
    censor_time : float
        Keyword only: when the test stops.
    levels : int, optional
        Keyword only: the number of stresses, at least 2, by default 2.
    allocation : sequence of float, optional
        Keyword only: fixed shares of the units at each stress, lowest
        first, summing to 1.
    middle_fraction : float, optional
        Keyword only: with three or more levels and no ``allocation``,
        the share of the units at the middle stresses, by default 0.2.
    p : float, optional
        Keyword only: the probability of the quantile, by default 0.1
        (the B10 life).
    alpha_ci : float, optional
        Keyword only: the significance level of the interval, by default
        0.05.

    Returns
    -------
    AcceleratedTestPlan
        The test, with its stresses and units at each, and the precision
        it reaches.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    The voltage test above, planned: the optimum two-level plan tests
    about 78% of the units at a low stress near 25 V, and the compromise
    plan of three levels gives up a little precision for a middle level:

    >>> import surpyval
    >>> from surpyval import life_models
    >>> from surpyval.test_planning import accelerated_test_plan
    >>> optimum = accelerated_test_plan(
    ...     surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0}, 2.0,
    ...     use_stress=10.0, high_stress=60.0, n=40, censor_time=1000.0)
    >>> [round(s, 1) for s in optimum.stresses], optimum.units
    ([24.9, 60.0], (31, 9))
    >>> compromise = accelerated_test_plan(
    ...     surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0}, 2.0,
    ...     use_stress=10.0, high_stress=60.0, n=40, censor_time=1000.0,
    ...     levels=3)
    >>> compromise.units
    (24, 8, 8)
    >>> round(optimum.precision, 2), round(compromise.precision, 2)
    (2.06, 2.2)
    """
    check_alpha_ci(alpha_ci)
    p = check_probability(p, "p")
    n = check_count(n, minimum=2)
    model = _Model(distribution, life_model, life_params, shape, censor_time)
    model.use_stress = _check_stress(use_stress, "use_stress", model)
    high = _check_stress(high_stress, "high_stress", model)
    if not model.log_life(high) < model.log_life(model.use_stress):
        raise ValueError(
            "At high_stress the planned life must be shorter than at "
            f"use_stress: with these planning values it is not "
            f"({high!r} against {use_stress!r})."
        )
    k = check_count(levels, "levels", minimum=2)
    if k < len(model.values):
        raise ValueError(
            f"The {life_model.name} model has {len(model.values)} "
            f"parameters: levels must be at least {len(model.values)}."
        )
    fixed = _allocation(allocation, k, middle_fraction)

    def shares(free: float | None) -> list[float]:
        if fixed is not None:
            return fixed
        middle = [middle_fraction / (k - 2)] * (k - 2) if k > 2 else []
        rest = 1.0 - (middle_fraction if k > 2 else 0.0)
        return [free * rest, *middle, (1.0 - free) * rest]  # type: ignore

    def positions(lowest: float) -> np.ndarray:
        return lowest + (1.0 - lowest) * np.arange(k) / (k - 1)

    def objective(point: np.ndarray) -> float:
        lowest = float(1.0 / (1.0 + np.exp(-point[0])))
        free = None if fixed is not None else 1.0 / (1.0 + np.exp(-point[1]))
        stresses = [model.stress_at(x, high) for x in positions(lowest)]
        value = model.variance(stresses, shares(free), p)
        return float(np.log(value)) if np.isfinite(value) else 1e300

    start = _grid_start(objective, fixed is None)
    best = optimize.minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000},
    )
    lowest = float(1.0 / (1.0 + np.exp(-best.x[0])))
    free = None if fixed is not None else 1.0 / (1.0 + np.exp(-best.x[1]))
    stresses = [model.stress_at(x, high) for x in positions(lowest)]
    fractions = shares(free)
    order = np.argsort(stresses, kind="stable")
    stresses = [stresses[i] for i in order]
    fractions = [fractions[i] for i in order]
    units = _whole_units(n, fractions)
    return _finish(model, stresses, units, fractions, p, alpha_ci)


def _grid_start(objective: Any, with_share: bool) -> np.ndarray:
    """The best point of a coarse grid, to start the search from."""
    lows = np.linspace(-4.0, 4.0, 33)
    if not with_share:
        values = [objective(np.array([a])) for a in lows]
        return np.array([lows[int(np.argmin(values))]])
    shares = np.linspace(-3.0, 3.0, 25)
    grid = [(a, b) for a in lows for b in shares]
    values = [objective(np.array(point)) for point in grid]
    return np.array(grid[int(np.argmin(values))])


def _allocation(
    allocation: Sequence[float] | None, k: int, middle_fraction: float
) -> list[float] | None:
    """The fixed shares, checked, or None to optimise them."""
    if allocation is None:
        if k > 2:
            middle = float(middle_fraction)
            if not 0.0 <= middle < 1.0:
                raise ValueError(
                    "middle_fraction must be in [0, 1), got "
                    f"{middle_fraction!r}."
                )
        return None
    shares = [float(a) for a in np.ravel(allocation)]
    if len(shares) != k:
        raise ValueError(
            f"allocation must give a share for each of the {k} levels, "
            f"got {len(shares)}."
        )
    if min(shares) <= 0 or abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(
            f"allocation must be positive shares summing to 1, got "
            f"{allocation!r}."
        )
    return shares


def _check_stress(stress: Any, name: str, model: _Model) -> float:
    try:
        value = float(stress)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {stress!r}.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {stress!r}.")
    if value <= 0 and model.life_model.positive_stress_columns:
        raise ValueError(
            f"The {model.life_model.name} model needs positive stresses: "
            f"{name} got {stress!r}."
        )
    model.log_life(value)
    return value
