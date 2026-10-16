"""Planning a life test for estimation precision (#863): how many units,
and how long a test, to estimate a quantile (a B-life) or the mean of a
Weibull or lognormal lifetime to a given precision.

Precision is the factor ``R`` of the confidence interval ``[estimate / R,
estimate * R]`` on the quantity, the interval a fit's Wald bound on the
log scale gives: ``R = exp(z * se)``, with ``se`` the standard error of
the log of the estimate and ``z`` the normal quantile for ``alpha_ci``.
"Within 20%" is ``R = 1.2``. The standard error is the large-sample one,
from the expected Fisher information of the test (Meeker and Escobar,
*Statistical Methods for Reliability Data*, ch. 10), for planning values
of the parameters; :meth:`LifeTestPlan.simulate` checks it by simulating
the test and fitting each simulated data set. The approximation is good
with a few dozen failures, and a little optimistic when they are few and
the quantity lies beyond the end of the test: simulate to check such a
plan.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, stats

from surpyval.test_planning import _information as info
from surpyval.test_planning._checks import (
    check_count,
    check_flag,
    check_positive,
    check_probability,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.validation import check_alpha_ci, check_option

QUANTITIES = ("quantile", "mean")


@dataclass(frozen=True, repr=False)
class SimulatedPrecision:
    """The precision of a planned test found by simulating it, from
    :meth:`LifeTestPlan.simulate` and
    :meth:`AcceleratedTestPlan.simulate`.

    Attributes
    ----------
    precision : float
        The precision factor the simulated estimates' spread gives,
        ``exp(z * sd)`` with ``sd`` the standard deviation of their logs:
        the large-sample plan's ``precision`` if its approximation holds.
    within : float
        The share of the simulated estimates within the plan's precision
        factor of the true value: about ``1 - alpha_ci`` if the plan
        holds.
    n_sim : int
        The number of simulated tests.
    unusable : int
        The simulated tests with too few failures to fit (fewer than
        two), left out of the two figures above.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.test_planning import life_test_precision
    >>> plan = life_test_precision(surpyval.Weibull, (1000.0, 2.0), 30)
    >>> check = plan.simulate(200, random_state=1)
    >>> round(check.precision, 2), round(plan.precision, 2)
    (1.49, 1.51)
    """

    precision: float
    within: float
    n_sim: int
    unusable: int

    def __repr__(self) -> str:
        header = "Simulated Precision"
        return "\n".join(
            [
                header,
                "=" * len(header),
                f"Precision factor : {self.precision:.6g}",
                f"Within planned   : {self.within:.6g}",
                f"Simulated tests  : {self.n_sim}",
                f"Unusable         : {self.unusable}",
            ]
        )


@dataclass(frozen=True, repr=False)
class LifeTestPlan:
    """A life test and the precision with which it estimates a quantity,
    from :func:`life_test_precision`, :func:`life_test_sample_size` and
    :func:`life_test_length`.

    Attributes
    ----------
    distribution : str
        The planned lifetime distribution, ``"Weibull"`` or
        ``"LogNormal"``.
    params : tuple of float
        Its planning values.
    n : int
        The number of units on test.
    censor_time : float or None
        When a time-censored (Type I) test stops; None for a test stopped
        at a failure, or run until every unit fails.
    failures : int or None
        The failure a failure-censored (Type II) test stops at; None for
        a time-censored or complete test.
    expected_failures : float
        The failures the test expects: ``n`` times the chance of failing
        by ``censor_time``, ``failures`` or ``n``.
    p : float or None
        The probability of the quantile estimated; None for the mean.
    estimate : float
        The planned value of the quantity, at the planning values.
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
    >>> from surpyval.test_planning import life_test_precision
    >>> life_test_precision(surpyval.Weibull, (1000.0, 2.0), 30,
    ...                     censor_time=500.0)
    Life Test Plan
    ==============
    Distribution     : Weibull(1000, 2)
    Units            : 30
    Test             : time-censored at 500
    Expected failures: 6.63598
    Estimating       : B10 = 324.593
    Precision factor : 1.61898 (95% interval)
    """

    distribution: str
    params: tuple[float, ...]
    n: int
    censor_time: float | None
    failures: int | None
    expected_failures: float
    p: float | None
    estimate: float
    standard_error: float
    precision: float
    alpha_ci: float

    def simulate(
        self, n_sim: int = 500, *, random_state: Any = None
    ) -> SimulatedPrecision:
        """Simulate the test ``n_sim`` times, fit the distribution to each
        simulated data set by maximum likelihood, and report the
        precision the estimates had.

        Parameters
        ----------
        n_sim : int, optional
            The number of simulated tests, by default 500.
        random_state : optional
            Keyword only: the seed, as every SurPyval draw takes it.

        Returns
        -------
        SimulatedPrecision

        Examples
        --------
        >>> import surpyval
        >>> from surpyval.test_planning import life_test_precision
        >>> plan = life_test_precision(surpyval.LogNormal, (6.0, 0.5), 20,
        ...                            quantity="mean")
        >>> plan.simulate(200, random_state=2).within > 0.9
        True
        """
        import surpyval

        n_sim = check_count(n_sim, "n_sim")
        rng = as_generator(random_state)
        fitter = getattr(surpyval, self.distribution)
        truth = np.log(self.estimate)
        logs, unusable = [], 0
        for _ in range(n_sim):
            x, c = self._draw(fitter, rng)
            if np.sum(c == 0) < 2:
                unusable += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = fitter.fit(x, c=c)
            value = model.mean() if self.p is None else model.qf(self.p)
            logs.append(float(np.log(np.asarray(value).item())))
        return _simulated(
            np.array(logs), truth, self.precision, self.alpha_ci, unusable
        )

    def _draw(self, fitter: Any, rng: np.random.Generator) -> tuple:
        """One simulated test's times and censoring flags."""
        x = fitter.random(self.n, *self.params, random_state=rng)
        stop = self.censor_time
        if self.failures is not None:
            stop = np.sort(x)[self.failures - 1]
        if stop is None:
            return x, np.zeros(self.n, dtype=int)
        c = (x > stop).astype(int)
        return np.minimum(x, stop), c

    def __repr__(self) -> str:
        header = "Life Test Plan"
        params = ", ".join(f"{v:.6g}" for v in self.params)
        if self.censor_time is not None:
            test = f"time-censored at {self.censor_time:.6g}"
        elif self.failures is not None:
            test = f"failure-censored at failure {self.failures}"
        else:
            test = "every unit run to failure"
        quantity = "mean" if self.p is None else f"B{100 * self.p:.6g}"
        level = f"{100 * (1 - self.alpha_ci):.6g}%"
        return "\n".join(
            [
                header,
                "=" * len(header),
                f"Distribution     : {self.distribution}({params})",
                f"Units            : {self.n}",
                f"Test             : {test}",
                f"Expected failures: {self.expected_failures:.6g}",
                f"Estimating       : {quantity} = {self.estimate:.6g}",
                f"Precision factor : {self.precision:.6g} ({level} "
                "interval)",
            ]
        )


def _simulated(
    logs: np.ndarray,
    truth: float,
    planned: float,
    alpha_ci: float,
    unusable: int,
) -> SimulatedPrecision:
    """The precision of simulated log estimates ``logs`` of ``truth``."""
    z = float(stats.norm.isf(alpha_ci / 2.0))
    if logs.size < 2:
        return SimulatedPrecision(
            np.nan, np.nan, logs.size + unusable, unusable
        )
    spread = float(np.exp(z * np.std(logs, ddof=1)))
    within = float(np.mean(np.abs(logs - truth) <= np.log(planned)))
    return SimulatedPrecision(spread, within, logs.size + unusable, unusable)


def _quantity(p: float | None, quantity: str) -> float | None:
    """The ``p`` of the quantity estimated, None for the mean."""
    check_option("quantity", quantity, QUANTITIES)
    if quantity == "mean":
        return None
    if p is None:
        raise ValueError(
            "A quantile needs its probability: give p, or quantity='mean'."
        )
    return check_probability(p, "p")


def _variance_per_unit(
    family: str, sigma: float, zeta: float, p: float | None
) -> float:
    """``n`` times the large-sample variance of the log estimate, for
    units censored at the standardised time ``zeta``."""
    gradient, _ = info.log_quantity_gradient(family, sigma, p)
    information = info.unit_information(family, zeta) / sigma**2
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return float(np.inf)
    return float(gradient @ covariance @ gradient)


def _censoring(
    family: str,
    mu: float,
    sigma: float,
    n: int,
    censor_time: float | None,
    failures: int | None,
) -> tuple[float, float]:
    """The standardised censoring time and the expected failures."""
    if censor_time is not None and failures is not None:
        raise ValueError(
            "Give censor_time (a time-censored test) or failures (a "
            "failure-censored one), not both."
        )
    if censor_time is not None:
        censor_time = check_positive(censor_time, "censor_time")
        zeta = (np.log(censor_time) - mu) / sigma
        return float(zeta), float(n * info.cdf(family, zeta))
    if failures is not None:
        failures = check_count(failures, "failures")
        if failures > n:
            raise ValueError(
                f"failures must be at most n ({n}), got {failures}."
            )
        if failures == n:
            return float(np.inf), float(n)
        # A Type II test is, in large samples, the Type I test censored
        # where the failed fraction is failures / n.
        return float(info.ppf(family, failures / n)), float(failures)
    return float(np.inf), float(n)


def _plan(
    distribution: Any,
    params: Any,
    n: int,
    censor_time: float | None,
    failures: int | None,
    p: float | None,
    alpha_ci: float,
) -> LifeTestPlan:
    family, mu, sigma = info.location_scale(distribution, params)
    zeta, expected = _censoring(family, mu, sigma, n, censor_time, failures)
    variance = _variance_per_unit(family, sigma, zeta, p)
    error = float(np.sqrt(variance / n))
    z = float(stats.norm.isf(alpha_ci / 2.0))
    _, offset = info.log_quantity_gradient(family, sigma, p)
    return LifeTestPlan(
        distribution=distribution.name,
        params=tuple(float(v) for v in np.ravel(params)),
        n=n,
        censor_time=None if censor_time is None else float(censor_time),
        failures=failures,
        expected_failures=expected,
        p=p,
        estimate=float(np.exp(mu + offset)),
        standard_error=error,
        precision=float(np.exp(z * error)),
        alpha_ci=float(alpha_ci),
    )


def life_test_precision(
    distribution: Any,
    params: Any,
    n: int,
    *,
    censor_time: float | None = None,
    failures: int | None = None,
    p: float | None = 0.1,
    quantity: str = "quantile",
    alpha_ci: float = 0.05,
) -> LifeTestPlan:
    """How precisely a life test of ``n`` units will estimate a quantile
    or the mean.

    The large-sample precision factor ``R`` of the estimate, for the
    planning values ``params``: the ``1 - alpha_ci`` confidence interval
    on the quantity will be about ``[estimate / R, estimate * R]``. The
    test stops at ``censor_time`` (time-censored, Type I), at its
    ``failures``-th failure (failure-censored, Type II), or, given
    neither, when every unit has failed.

    Parameters
    ----------
    distribution : surpyval.Weibull or surpyval.LogNormal
        The lifetime distribution the test's data will be fitted with.
    params : sequence of float
        Its planning values: ``(alpha, beta)`` for the Weibull, ``(mu,
        sigma)`` for the lognormal. The shape matters most; prior tests
        or engineering knowledge give it.
    n : int
        The number of units on test.
    censor_time : float, optional
        Keyword only: when a time-censored test stops.
    failures : int, optional
        Keyword only: the failure a failure-censored test stops at.
    p : float, optional
        Keyword only: the probability of the quantile to estimate, by
        default 0.1 (the B10 life).
    quantity : str, optional
        Keyword only: ``"quantile"`` (the default) or ``"mean"``.
    alpha_ci : float, optional
        Keyword only: the significance level of the interval, by default
        0.05.

    Returns
    -------
    LifeTestPlan
        The test and the precision of its estimate.

    Raises
    ------
    ValueError
        If an argument is out of range, or both ``censor_time`` and
        ``failures`` are given.

    Examples
    --------
    Thirty units of a wear-out part (Weibull shape 2, scale 1000 hours)
    on a 500-hour test, about 7 failures, estimate its B10 life of 325
    hours to within a factor of 1.62 at 95% confidence; run until all
    thirty fail, 1.51:

    >>> import surpyval
    >>> from surpyval.test_planning import life_test_precision
    >>> plan = life_test_precision(surpyval.Weibull, (1000.0, 2.0), 30,
    ...                            censor_time=500.0)
    >>> round(plan.estimate, 1), round(plan.precision, 3)
    (324.6, 1.619)
    >>> plan = life_test_precision(surpyval.Weibull, (1000.0, 2.0), 30)
    >>> round(plan.precision, 2)
    1.51
    """
    check_alpha_ci(alpha_ci)
    n = check_count(n)
    p = _quantity(p, quantity)
    return _plan(distribution, params, n, censor_time, failures, p, alpha_ci)


def life_test_sample_size(
    distribution: Any,
    params: Any,
    precision: float,
    *,
    censor_time: float | None = None,
    failure_fraction: float | None = None,
    p: float | None = 0.1,
    quantity: str = "quantile",
    alpha_ci: float = 0.05,
) -> LifeTestPlan:
    """The fewest units whose life test estimates a quantile or the mean
    to a precision.

    The smallest ``n`` whose test, stopped at ``censor_time``
    (time-censored), when ``failure_fraction`` of the units have failed
    (failure-censored), or when all have, estimates the quantity with a
    precision factor of at most ``precision``: the large-sample variance
    of the log estimate falls as ``1 / n``.

    Parameters
    ----------
    distribution : surpyval.Weibull or surpyval.LogNormal
        The lifetime distribution the test's data will be fitted with.
    params : sequence of float
        Its planning values: ``(alpha, beta)`` for the Weibull, ``(mu,
        sigma)`` for the lognormal.
    precision : float
        The precision factor to reach, more than 1: 1.2 for "within
        20%".
    censor_time : float, optional
        Keyword only: when a time-censored test stops.
    failure_fraction : float, optional
        Keyword only: the share of the units whose failure stops a
        failure-censored test, in (0, 1].
    p : float, optional
        Keyword only: the probability of the quantile to estimate, by
        default 0.1 (the B10 life).
    quantity : str, optional
        Keyword only: ``"quantile"`` (the default) or ``"mean"``.
    alpha_ci : float, optional
        Keyword only: the significance level of the interval, by default
        0.05.

    Returns
    -------
    LifeTestPlan
        The test, with its ``n``, and the precision it reaches.

    Raises
    ------
    ValueError
        If an argument is out of range, or both ``censor_time`` and
        ``failure_fraction`` are given.

    Examples
    --------
    To estimate the B10 life within 30% at 90% confidence, a 600-hour
    test of that part needs 72 units:

    >>> import surpyval
    >>> from surpyval.test_planning import life_test_sample_size
    >>> plan = life_test_sample_size(surpyval.Weibull, (1000.0, 2.0), 1.3,
    ...                              censor_time=600.0, alpha_ci=0.1)
    >>> plan.n, round(plan.precision, 3)
    (72, 1.298)
    """
    check_alpha_ci(alpha_ci)
    precision = _check_precision(precision)
    p = _quantity(p, quantity)
    family, mu, sigma = info.location_scale(distribution, params)
    if censor_time is not None and failure_fraction is not None:
        raise ValueError(
            "Give censor_time (a time-censored test) or failure_fraction "
            "(a failure-censored one), not both."
        )
    if failure_fraction is not None:
        fraction = float(failure_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "failure_fraction must be in (0, 1], got "
                f"{failure_fraction!r}."
            )
        zeta = np.inf if fraction == 1.0 else float(info.ppf(family, fraction))
    elif censor_time is not None:
        censor_time = check_positive(censor_time, "censor_time")
        zeta = (np.log(censor_time) - mu) / sigma
    else:
        zeta = np.inf
    variance = _variance_per_unit(family, sigma, zeta, p)
    z = float(stats.norm.isf(alpha_ci / 2.0))
    n = max(int(np.ceil(z**2 * variance / np.log(precision) ** 2 - 1e-9)), 2)
    failures = None
    if failure_fraction is not None:
        failures = max(int(np.ceil(fraction * n - 1e-9)), 1)
    return _plan(distribution, params, n, censor_time, failures, p, alpha_ci)


def life_test_length(
    distribution: Any,
    params: Any,
    n: int,
    precision: float,
    *,
    failure_censored: bool = False,
    p: float | None = 0.1,
    quantity: str = "quantile",
    alpha_ci: float = 0.05,
) -> LifeTestPlan:
    """How long a life test of ``n`` units must run to estimate a
    quantile or the mean to a precision.

    The shortest time-censored test (or, with ``failure_censored``, the
    earliest failure to stop at) whose estimate of the quantity has a
    precision factor of at most ``precision``. Running longer always
    helps, so the answer is the time at which the precision is reached.

    Parameters
    ----------
    distribution : surpyval.Weibull or surpyval.LogNormal
        The lifetime distribution the test's data will be fitted with.
    params : sequence of float
        Its planning values: ``(alpha, beta)`` for the Weibull, ``(mu,
        sigma)`` for the lognormal.
    n : int
        The number of units on test.
    precision : float
        The precision factor to reach, more than 1.
    failure_censored : bool, optional
        Keyword only: whether to stop at a failure rather than at a
        time, by default False.
    p : float, optional
        Keyword only: the probability of the quantile to estimate, by
        default 0.1 (the B10 life).
    quantity : str, optional
        Keyword only: ``"quantile"`` (the default) or ``"mean"``.
    alpha_ci : float, optional
        Keyword only: the significance level of the interval, by default
        0.05.

    Returns
    -------
    LifeTestPlan
        The test, with its ``censor_time`` (or ``failures``), and the
        precision it reaches.

    Raises
    ------
    ValueError
        If an argument is out of range, or not even running every unit
        to failure reaches ``precision``.

    Examples
    --------
    Forty units estimate the B10 life within a factor of 1.5 at 90%
    confidence if the test runs about 330 hours, or to the 5th failure:

    >>> import surpyval
    >>> from surpyval.test_planning import life_test_length
    >>> plan = life_test_length(surpyval.Weibull, (1000.0, 2.0), 40, 1.5,
    ...                         alpha_ci=0.1)
    >>> round(plan.censor_time), round(plan.precision, 3)
    (329, 1.5)
    >>> life_test_length(surpyval.Weibull, (1000.0, 2.0), 40, 1.5,
    ...                  failure_censored=True, alpha_ci=0.1).failures
    5
    """
    check_alpha_ci(alpha_ci)
    n = check_count(n)
    precision = _check_precision(precision)
    p = _quantity(p, quantity)
    family, mu, sigma = info.location_scale(distribution, params)
    z = float(stats.norm.isf(alpha_ci / 2.0))
    target = n * (np.log(precision) / z) ** 2  # the per-unit variance

    def excess(zeta: float) -> float:
        return _variance_per_unit(family, sigma, zeta, p) - target

    complete = _variance_per_unit(family, sigma, np.inf, p)
    if complete > target:
        best = np.exp(z * np.sqrt(complete / n))
        raise ValueError(
            f"Not even running all {n} units to failure reaches a precision "
            f"factor of {precision!r} (it gives {best:.4g}): test more "
            "units."
        )
    if check_flag(failure_censored, "failure_censored"):
        for failures in range(1, n + 1):
            plan = _plan(distribution, params, n, None, failures, p, alpha_ci)
            if plan.precision <= precision * (1.0 + 1e-12):
                return plan
    edge = info._COMPLETE[family]
    lower = edge - 1.0
    while excess(lower) <= 0.0:
        lower -= 1.0
    if excess(edge) > 0.0:  # reached only by running every unit out
        return _plan(distribution, params, n, None, None, p, alpha_ci)
    zeta = optimize.brentq(excess, lower, edge, xtol=1e-12)
    censor_time = float(np.exp(mu + sigma * zeta))
    return _plan(distribution, params, n, censor_time, None, p, alpha_ci)


def _check_precision(precision: float) -> float:
    precision = check_positive(precision, "precision")
    if precision <= 1.0:
        raise ValueError(
            "precision must be more than 1 (1.2 for within 20%), got "
            f"{precision!r}."
        )
    return precision
