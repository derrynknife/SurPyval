"""Sequential demonstration tests (#865): Wald's sequential probability
ratio test (SPRT), for an MTBF at a constant failure rate and for a
pass/fail reliability.

A sequential test decides as the evidence comes in: after each failure,
and as test time (or units) accumulate, it accepts the design, rejects it
or keeps testing. With the same risks it usually decides sooner than the
fixed-length test of :func:`mtbf_demonstration_plan` or
:func:`demonstration_plan`.

The boundaries are Wald's: with the consumer's risk ``alpha_ci`` (that a
design at the target passes) and the producer's risk (that a good design
fails), the test accepts when the likelihood ratio of the target to the
good design falls to ``alpha_ci / (1 - producer_risk)`` and rejects when
it reaches ``(1 - alpha_ci) / producer_risk``. Truncated, as practice
requires (MIL-HDBK-781), it stops at the fixed-length test with the same
risks: it rejects at that test's first failure too many and accepts when
that test's time (or units) is reached. The risks of the test as run are
computed exactly, not taken from Wald's approximations.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import special, stats

from surpyval.test_planning._checks import (
    check_count,
    check_failures,
    check_flag,
    check_positive,
    check_probability,
)
from surpyval.test_planning.demonstration import (
    demonstration_plan,
    mtbf_demonstration_plan,
)
from surpyval.utils.validation import check_alpha_ci

#: The probability left undecided at which an untruncated test's
#: operating characteristic stops being followed.
_UNDECIDED = 1e-12


def _wald_limits(alpha_ci: float, producer_risk: float) -> tuple[float, float]:
    """The logs of Wald's accept and reject limits on the likelihood ratio
    of the target to the good design."""
    accept = np.log(alpha_ci / (1.0 - producer_risk))
    reject = np.log((1.0 - alpha_ci) / producer_risk)
    return float(accept), float(reject)


def _check_risks(alpha_ci: float, producer_risk: float) -> None:
    check_alpha_ci(alpha_ci)
    check_probability(producer_risk, "producer_risk")
    if alpha_ci + producer_risk >= 1.0:
        raise ValueError(
            "alpha_ci and producer_risk must add to less than 1, got "
            f"{alpha_ci!r} and {producer_risk!r}: a test that a coin toss "
            "could pass needs no testing."
        )


def _check_mtbf(value: float) -> float:
    return check_positive(value, "mtbf")


def _check_reliability(value: float) -> float:
    return check_probability(value, "reliability")


def _risks_lines(consumer: float, producer: float) -> list[str]:
    return [
        f"Consumer's risk  : {consumer:.6g}",
        f"Producer's risk  : {producer:.6g}",
    ]


class SequentialMTBFTest:
    """A sequential test of an MTBF at a constant failure rate, from
    :func:`sequential_mtbf_test`.

    Failures are counted, and test time totalled over the units (failed
    ones repaired or replaced), as the test runs. With ``r`` failures in
    total time ``T`` the log of the likelihood ratio of the target MTBF
    to the good one is ``r ln(good_mtbf / mtbf) - T (1 / mtbf -
    1 / good_mtbf)``: the test accepts the design once ``T`` reaches
    :meth:`accept_time` of ``r``, and rejects it if its ``r``-th failure
    comes by :meth:`reject_time` of ``r`` (or, truncated, is its
    ``max_failures``-th).

    Attributes
    ----------
    mtbf : float
        The MTBF to demonstrate.
    good_mtbf : float
        The MTBF of a design that should pass.
    boundary_risks : (float, float)
        The consumer's and producer's risks the boundaries were set for:
        the ``alpha_ci`` and ``producer_risk`` asked for.
    max_time : float or None
        The total test time at which a truncated test accepts; None if it
        is not truncated.
    max_failures : int or None
        The failure at which a truncated test rejects; None if it is not
        truncated.
    consumer_risk : float
        The chance, exactly, that a design at ``mtbf`` passes the test as
        run.
    producer_risk : float
        The chance, exactly, that a design at ``good_mtbf`` fails it.

    Examples
    --------
    >>> from surpyval.test_planning import sequential_mtbf_test
    >>> test = sequential_mtbf_test(100.0, 200.0, alpha_ci=0.1,
    ...                             producer_risk=0.1)
    >>> round(test.accept_time(0), 1), round(test.accept_time(3), 1)
    (439.4, 855.3)
    >>> round(test.reject_time(4), 1)
    115.1
    >>> test.decide(failures=2, test_time=800.0)
    'accept'
    """

    def __init__(
        self,
        mtbf: float,
        good_mtbf: float,
        alpha_ci: float,
        producer_risk: float,
        max_time: float | None,
        max_failures: int | None,
    ) -> None:
        self.mtbf = mtbf
        self.good_mtbf = good_mtbf
        self.boundary_risks = (alpha_ci, producer_risk)
        self.max_time = max_time
        self.max_failures = max_failures
        self._step = float(np.log(good_mtbf / mtbf))
        self._slope = 1.0 / mtbf - 1.0 / good_mtbf
        self._log_accept, self._log_reject = _wald_limits(
            alpha_ci, producer_risk
        )
        target = self._outcome(mtbf)
        good = self._outcome(good_mtbf)
        self.consumer_risk = target[0]
        self.producer_risk = 1.0 - good[0]

    def accept_time(self, failures: int) -> float:
        """The total test time at which the test accepts the design, if
        it has had ``failures`` failures by then.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_mtbf_test
        >>> test = sequential_mtbf_test(100.0, 200.0)
        >>> round(test.accept_time(0), 1)
        554.5
        """
        failures = check_failures(failures)
        time = (failures * self._step - self._log_accept) / self._slope
        if self.max_time is not None:
            time = min(time, self.max_time)
        return float(time)

    def reject_time(self, failures: int) -> float:
        """The total test time by which the ``failures``-th failure
        rejects the design: it rejects if that failure comes at or before
        this time. Zero when that failure cannot reject.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_mtbf_test
        >>> test = sequential_mtbf_test(100.0, 200.0)
        >>> test.reject_time(1)
        0.0
        >>> round(test.reject_time(6), 1)
        520.1
        """
        failures = check_failures(failures)
        if self.max_failures is not None and failures >= self.max_failures:
            return float(np.inf)
        time = (failures * self._step - self._log_reject) / self._slope
        return float(max(time, 0.0))

    def decide(self, failures: int, test_time: float) -> str:
        """What the test says with ``failures`` failures in total test
        time ``test_time``: ``"accept"``, ``"reject"`` or
        ``"continue"``.

        Ask at each failure, with the total test time it came at, and as
        test time accumulates between failures.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_mtbf_test
        >>> test = sequential_mtbf_test(100.0, 200.0)
        >>> test.decide(0, 300.0), test.decide(7, 250.0)
        ('continue', 'reject')
        """
        failures = check_failures(failures)
        test_time = float(test_time)
        if not test_time >= 0.0:
            raise ValueError(
                f"test_time must be 0 or more, got {test_time!r}."
            )
        reject = self.reject_time(failures)
        if reject > 0.0 and test_time <= reject:
            return "reject"
        if test_time >= self.accept_time(failures):
            return "accept"
        return "continue"

    def pass_probability(self, mtbf: npt.ArrayLike) -> npt.NDArray | float:
        """The chance that a design of a given true MTBF passes the test
        (its operating characteristic), computed exactly.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_mtbf_test
        >>> test = sequential_mtbf_test(100.0, 200.0)
        >>> round(test.pass_probability(100.0), 3)
        0.061
        """
        return _per_value(lambda m: self._outcome(m)[0], mtbf, _check_mtbf)

    def expected_test_time(self, mtbf: npt.ArrayLike) -> npt.NDArray | float:
        """The mean total test time to a decision for a design of a given
        true MTBF.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_mtbf_test
        >>> test = sequential_mtbf_test(100.0, 200.0)
        >>> round(test.expected_test_time(200.0))
        1178
        """
        return _per_value(lambda m: self._outcome(m)[1], mtbf, _check_mtbf)

    def _outcome(self, mtbf: float) -> tuple[float, float]:
        """The chance of accepting and the mean test time for a design of
        MTBF ``mtbf``: the failures a Poisson process of rate ``1 /
        mtbf``, followed exactly between the boundaries' corners."""
        rate = 1.0 / mtbf
        most = 64 if self.max_failures is None else self.max_failures
        while True:
            outcome = self._follow(rate, most)
            if outcome is not None:
                return outcome
            most *= 2

    def _follow(self, rate: float, most: int) -> tuple[float, float] | None:
        """Follow the test up to ``most`` failures; None if it may run on
        past them undecided."""
        accept_at = [self.accept_time(r) for r in range(most)]
        reject_by = [self.reject_time(r) for r in range(most + 1)]
        corners = sorted(
            {0.0, *accept_at, *(t for t in reject_by if np.isfinite(t))}
        )
        if self.max_time is not None:
            corners = [t for t in corners if t <= self.max_time]
        mass = np.zeros(most)
        mass[0] = 1.0
        accepted, mean_time = 0.0, 0.0
        for start, end in zip(corners, corners[1:] + [np.inf]):
            for r in range(most):
                if accept_at[r] <= start:
                    accepted += mass[r]
                    mass[r] = 0.0
            if mass.sum() < _UNDECIDED or not np.isfinite(end):
                break
            # Arriving at a count whose reject time is still ahead
            # rejects; the first such count bounds the counts kept.
            stop = next(
                (r for r in range(most + 1) if reject_by[r] >= end), most
            )
            mean_time += _time_below(mass, rate, end - start, stop)
            mass, leaked = _advance(mass, rate * (end - start), stop)
            if stop == most and self.max_failures is None:
                if leaked > _UNDECIDED:
                    return None  # past the failures followed: follow more
        if self.max_failures is None and mass.sum() >= _UNDECIDED:
            return None
        accepted += mass.sum()  # truncated: accepted at max_time
        return float(accepted), float(mean_time)

    def __repr__(self) -> str:
        header = "Sequential MTBF Test"
        lines = [
            header,
            "=" * len(header),
            f"MTBF to show     : {self.mtbf:.6g}",
            f"Good MTBF        : {self.good_mtbf:.6g}",
            "Accept at time   : "
            f"{self.accept_time(0):.6g} + {self._step / self._slope:.6g} "
            "per failure",
        ]
        if self.max_time is not None:
            lines += [
                f"Truncated at     : {self.max_time:.6g} test time, "
                f"failure {self.max_failures}",
            ]
        return "\n".join(
            lines + _risks_lines(self.consumer_risk, self.producer_risk)
        )


def _time_below(
    mass: npt.NDArray, rate: float, length: float, stop: int
) -> float:
    """The test time spent, over an interval of ``length``, by the
    probability ``mass`` on the failure counts, before any count reaches
    ``stop``: from count ``i`` the mean time until the ``stop - i``-th
    failure or the interval's end."""
    total = 0.0
    for count in np.flatnonzero(mass[:stop]):
        shapes = np.arange(1, stop - count + 1)
        total += mass[count] * special.gammainc(shapes, rate * length).sum()
    return float(total / rate)


def _advance(
    mass: npt.NDArray, expected: float, stop: int
) -> tuple[npt.NDArray, float]:
    """The probability on each failure count after a stretch with
    ``expected`` failures, and the probability that reached ``stop`` (it
    rejected, and is removed)."""
    after = np.zeros_like(mass)
    jumps = stats.poisson.pmf(np.arange(stop), expected)
    for count in np.flatnonzero(mass[:stop]):
        after[count:stop] += mass[count] * jumps[: stop - count]
    return after, float(mass.sum() - after.sum())


class SequentialDemonstrationTest:
    """A sequential pass/fail test of a reliability, from
    :func:`sequential_demonstration_test`.

    Units are tested one at a time (or in any order, counting as each
    finishes), each for one mission. With ``f`` failures among ``n``
    units the log of the likelihood ratio of the target reliability to
    the good one is ``f ln(q1 / q0) + (n - f) ln(R1 / R0)``, with ``R1``
    the target, ``R0`` the good reliability and ``q = 1 - R``: the test
    accepts the design while the failures are at most
    :meth:`accept_failures` of ``n``, and rejects it when they reach
    :meth:`reject_failures` of ``n``.

    Attributes
    ----------
    reliability : float
        The reliability to demonstrate.
    good_reliability : float
        The reliability of a design that should pass.
    boundary_risks : (float, float)
        The consumer's and producer's risks the boundaries were set for:
        the ``alpha_ci`` and ``producer_risk`` asked for.
    max_units : int or None
        The number of units after which a truncated test accepts; None if
        it is not truncated.
    max_failures : int or None
        The failure at which a truncated test rejects; None if it is not
        truncated.
    consumer_risk : float
        The chance, exactly, that a design at ``reliability`` passes the
        test as run.
    producer_risk : float
        The chance, exactly, that a design at ``good_reliability`` fails
        it.

    Examples
    --------
    >>> from surpyval.test_planning import sequential_demonstration_test
    >>> test = sequential_demonstration_test(0.9, 0.97, alpha_ci=0.1,
    ...                                      producer_risk=0.1)
    >>> test.accept_failures(30), test.reject_failures(30)
    (0, 4)
    >>> test.decide(n=40, failures=1), test.decide(n=60, failures=1)
    ('continue', 'accept')
    """

    def __init__(
        self,
        reliability: float,
        good_reliability: float,
        alpha_ci: float,
        producer_risk: float,
        max_units: int | None,
        max_failures: int | None,
    ) -> None:
        self.reliability = reliability
        self.good_reliability = good_reliability
        self.boundary_risks = (alpha_ci, producer_risk)
        self.max_units = max_units
        self.max_failures = max_failures
        failed = np.log((1.0 - reliability) / (1.0 - good_reliability))
        passed = np.log(good_reliability / reliability)
        self._per_unit = float(passed / (failed + passed))
        self._scale = float(failed + passed)
        self._log_accept, self._log_reject = _wald_limits(
            alpha_ci, producer_risk
        )
        target = self._outcome(reliability)
        good = self._outcome(good_reliability)
        self.consumer_risk = target[0]
        self.producer_risk = 1.0 - good[0]

    def accept_failures(self, n: int) -> int:
        """The most failures among ``n`` units with which the test
        accepts the design; -1 when it cannot accept yet.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_demonstration_test
        >>> test = sequential_demonstration_test(0.9, 0.97)
        >>> test.accept_failures(10), test.accept_failures(60)
        (-1, 1)
        """
        n = check_count(n, minimum=0)
        if self.max_units is not None and n >= self.max_units:
            return int(self.max_failures - 1)  # type: ignore[operator]
        limit = n * self._per_unit + self._log_accept / self._scale
        return int(np.floor(limit + 1e-12)) if limit >= 0 else -1

    def reject_failures(self, n: int) -> int:
        """The fewest failures among ``n`` units with which the test
        rejects the design.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_demonstration_test
        >>> test = sequential_demonstration_test(0.9, 0.97)
        >>> test.reject_failures(10)
        2
        """
        n = check_count(n, minimum=0)
        limit = n * self._per_unit + self._log_reject / self._scale
        failures = max(int(np.ceil(limit - 1e-12)), 1)
        if self.max_failures is not None:
            failures = min(failures, self.max_failures)
        return failures

    def decide(self, n: int, failures: int) -> str:
        """What the test says with ``failures`` failures among the first
        ``n`` units: ``"accept"``, ``"reject"`` or ``"continue"``.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_demonstration_test
        >>> test = sequential_demonstration_test(0.9, 0.97)
        >>> test.decide(20, 0), test.decide(20, 5)
        ('continue', 'reject')
        """
        n = check_count(n, minimum=0)
        failures = check_failures(failures)
        if failures > n:
            raise ValueError(
                f"failures must be at most n ({n}), got {failures}."
            )
        if failures >= self.reject_failures(n):
            return "reject"
        if failures <= self.accept_failures(n):
            return "accept"
        return "continue"

    def pass_probability(
        self, reliability: npt.ArrayLike
    ) -> npt.NDArray | float:
        """The chance that a design of a given true reliability passes
        the test (its operating characteristic), computed exactly.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_demonstration_test
        >>> test = sequential_demonstration_test(0.9, 0.97)
        >>> round(test.pass_probability(0.9), 3)
        0.058
        """
        return _per_value(
            lambda r: self._outcome(r)[0], reliability, _check_reliability
        )

    def expected_units(
        self, reliability: npt.ArrayLike
    ) -> npt.NDArray | float:
        """The mean number of units tested to a decision for a design of
        a given true reliability.

        Examples
        --------
        >>> from surpyval.test_planning import sequential_demonstration_test
        >>> test = sequential_demonstration_test(0.9, 0.97)
        >>> round(test.expected_units(0.97), 1)
        51.5
        """
        return _per_value(
            lambda r: self._outcome(r)[1], reliability, _check_reliability
        )

    def _outcome(self, reliability: float) -> tuple[float, float]:
        """The chance of accepting and the mean number of units for a
        design of ``reliability``: the failure count followed unit by
        unit."""
        mass = np.array([1.0])
        accepted, units, n = 0.0, 0.0, 0
        while mass.sum() >= _UNDECIDED:
            if self.max_units is not None and n >= self.max_units:
                accepted += mass.sum()
                break
            units += mass.sum()
            n += 1
            tested = np.zeros(mass.size + 1)
            tested[:-1] += mass * reliability
            tested[1:] += mass * (1.0 - reliability)
            mass = tested
            reject = self.reject_failures(n)
            mass[reject:] = 0.0
            accept = self.accept_failures(n)
            if accept >= 0:
                accepted += mass[: accept + 1].sum()
                mass[: accept + 1] = 0.0
        return float(accepted), float(units)

    def __repr__(self) -> str:
        header = "Sequential Demonstration Test"
        lines = [
            header,
            "=" * len(header),
            f"Reliability      : {self.reliability:.6g}",
            f"Good reliability : {self.good_reliability:.6g}",
            "Accept failures  : "
            f"{self._log_accept / self._scale:.6g} + "
            f"{self._per_unit:.6g} per unit",
        ]
        if self.max_units is not None:
            lines.append(
                f"Truncated at     : {self.max_units} units, "
                f"failure {self.max_failures}"
            )
        return "\n".join(
            lines + _risks_lines(self.consumer_risk, self.producer_risk)
        )


def _per_value(
    function: Callable[[float], float],
    values: npt.ArrayLike,
    check: Callable[[float], float],
) -> npt.NDArray | float:
    """``function`` of each of ``values``, each checked by ``check``."""
    array = np.asarray(values, dtype=float)
    for value in np.unique(array):
        check(float(value))
    answer = np.vectorize(function, otypes=[float])(array)
    return float(answer) if answer.ndim == 0 else answer


def sequential_mtbf_test(
    mtbf: float,
    good_mtbf: float,
    *,
    alpha_ci: float = 0.05,
    producer_risk: float = 0.2,
    truncate: bool = True,
) -> SequentialMTBFTest:
    """A sequential test of an MTBF at a constant failure rate (Wald's
    sequential probability ratio test).

    The test accepts the design once enough test time has accumulated
    for the failures seen, and rejects it if failures come too soon: a
    design at the target ``mtbf`` should pass with a chance of about
    ``alpha_ci`` (the consumer's risk), and one at ``good_mtbf`` fail with
    a chance of about ``producer_risk``. ``truncate`` (the default, as
    MIL-HDBK-781 requires) stops it at the fixed-length test with the
    same risks (:func:`mtbf_demonstration_plan`): it rejects at that
    test's first failure too many, and accepts at its total test time.
    The test's risks as run are computed exactly, as its
    ``consumer_risk`` and ``producer_risk``.

    Parameters
    ----------
    mtbf : float
        The MTBF to demonstrate, positive (the lower test MTBF).
    good_mtbf : float
        The MTBF of a design that should pass, more than ``mtbf`` (the
        upper test MTBF).
    alpha_ci : float, optional
        Keyword only: the consumer's risk, by default 0.05.
    producer_risk : float, optional
        Keyword only: the producer's risk, by default 0.2.
    truncate : bool, optional
        Keyword only: whether to stop at the fixed-length test with the
        same risks, by default True.

    Returns
    -------
    SequentialMTBFTest
        The test: its boundaries, a ``decide`` method, and its operating
        characteristic and mean test time.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``good_mtbf`` is no more than
        ``mtbf``.

    Examples
    --------
    Demonstrating an MTBF of 100 hours, a design of 200 hours to pass,
    both risks 10%: the fixed-length test runs 2,013 hours, and the
    sequential one about 1,080 on average for the good design and 910
    for one at the target. Truncation and the boundaries' overshoot move
    its risks from the nominal 10%; it gives them exactly:

    >>> from surpyval.test_planning import (
    ...     mtbf_demonstration_plan,
    ...     sequential_mtbf_test,
    ... )
    >>> fixed = mtbf_demonstration_plan(100.0, 200.0, alpha_ci=0.1,
    ...                                 producer_risk=0.1)
    >>> round(fixed.test_time)
    2013
    >>> test = sequential_mtbf_test(100.0, 200.0, alpha_ci=0.1,
    ...                             producer_risk=0.1)
    >>> round(test.expected_test_time(200.0)), round(
    ...     test.expected_test_time(100.0))
    (1082, 914)
    >>> round(test.consumer_risk, 3), round(test.producer_risk, 3)
    (0.128, 0.103)
    """
    check_positive(mtbf, "mtbf")
    check_positive(good_mtbf, "good_mtbf")
    _check_risks(alpha_ci, producer_risk)
    if good_mtbf <= mtbf:
        raise ValueError(
            f"good_mtbf must be more than the MTBF to demonstrate "
            f"({mtbf!r}), got {good_mtbf!r}."
        )
    max_time, max_failures = None, None
    if check_flag(truncate, "truncate"):
        fixed = mtbf_demonstration_plan(
            mtbf, good_mtbf, alpha_ci=alpha_ci, producer_risk=producer_risk
        )
        max_time = fixed.test_time
        max_failures = fixed.failures + 1
    return SequentialMTBFTest(
        float(mtbf),
        float(good_mtbf),
        float(alpha_ci),
        float(producer_risk),
        max_time,
        max_failures,
    )


def sequential_demonstration_test(
    reliability: float,
    good_reliability: float,
    *,
    alpha_ci: float = 0.05,
    producer_risk: float = 0.2,
    truncate: bool = True,
) -> SequentialDemonstrationTest:
    """A sequential pass/fail test of a reliability (Wald's sequential
    probability ratio test for a binomial proportion).

    Units are tested one mission each, and after each the test accepts
    the design if the failures so far are few enough for the units
    tested, rejects it if they are too many, or tests another: a design
    at the target ``reliability`` should pass with a chance of about
    ``alpha_ci`` (the consumer's risk), and one at ``good_reliability``
    fail with a chance of about ``producer_risk``. ``truncate`` (the
    default) stops it at the fixed-length test with the same risks
    (:func:`demonstration_plan`): it rejects at that test's first
    failure too many, and accepts after its number of units. The test's
    risks as run are computed exactly, as its ``consumer_risk`` and
    ``producer_risk``.

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    good_reliability : float
        The reliability of a design that should pass, more than
        ``reliability`` and less than 1.
    alpha_ci : float, optional
        Keyword only: the consumer's risk, by default 0.05.
    producer_risk : float, optional
        Keyword only: the producer's risk, by default 0.2.
    truncate : bool, optional
        Keyword only: whether to stop at the fixed-length test with the
        same risks, by default True.

    Returns
    -------
    SequentialDemonstrationTest
        The test: its boundaries, a ``decide`` method, and its operating
        characteristic and mean number of units.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``good_reliability`` is no
        more than ``reliability``.

    Examples
    --------
    Demonstrating 90% reliability, a design of 97% to pass, both risks
    10%: the fixed-length test takes 78 units, and the sequential one
    about 47 for the good design:

    >>> from surpyval.test_planning import (
    ...     demonstration_plan,
    ...     sequential_demonstration_test,
    ... )
    >>> demonstration_plan(0.9, 0.97, alpha_ci=0.1, producer_risk=0.1).n
    78
    >>> test = sequential_demonstration_test(0.9, 0.97, alpha_ci=0.1,
    ...                                      producer_risk=0.1)
    >>> round(test.expected_units(0.97), 1)
    47.0
    """
    check_probability(reliability, "reliability")
    check_probability(good_reliability, "good_reliability")
    _check_risks(alpha_ci, producer_risk)
    if good_reliability <= reliability:
        raise ValueError(
            "good_reliability must be more than the reliability to "
            f"demonstrate ({reliability!r}), got {good_reliability!r}."
        )
    max_units, max_failures = None, None
    if check_flag(truncate, "truncate"):
        fixed = demonstration_plan(
            reliability,
            good_reliability,
            alpha_ci=alpha_ci,
            producer_risk=producer_risk,
        )
        max_units = fixed.n
        max_failures = fixed.failures + 1
    return SequentialDemonstrationTest(
        float(reliability),
        float(good_reliability),
        float(alpha_ci),
        float(producer_risk),
        max_units,
        max_failures,
    )
