"""Reliability growth planning: the curve a growth test is planned
against, before any failure is seen (MIL-HDBK-189C, section 5).

Three planning models:

- **The idealised growth curve** (the AMSAA-Crow planning model, sections
  5.2 and 5.3): the MTBF is ``M_I`` over a first test phase of length
  ``t_1``, and from then on grows as a power of the test time,
  :math:`M(t) = M_I (t / t_1)^\\alpha / (1 - \\alpha)`, with growth rate
  :math:`\\alpha` (Duane's slope). Give any three of the initial MTBF, the
  growth rate, the final MTBF and the test time and the fourth is solved
  for. Because the curve is a Crow-AMSAA (power-law) process with shape
  :math:`1 - \\alpha`, the chance that the growth test's own data
  demonstrate a requirement at a confidence level is exact (Broemm,
  Ellner and Woodworth 2000; section 5.3.6).
- **PM2-Continuous** (section 5.5): planned from the quantities
  management sets, the management strategy (the share of the initial
  failure intensity that corrective action will address) and the average
  fix effectiveness, with no growth rate. The expected MTBF once every
  mode seen by ``t`` is fixed is
  :math:`1 / [\\lambda_A + (1 - \\mu_d) (\\lambda_B - h(t)) + h(t)]`,
  :math:`h(t) = \\lambda_B / (1 + \\beta t)` the rate of new correctable
  modes.
- **PM2-Discrete** (section 5.6): the same for one-shot items, in
  trials: :math:`R(t) = R_A R_B^{1 - \\mu_d (t - 1) / (n + t - 1)}`.

The planned steps of each come from the schedule of corrective action
periods: fixes go in at the end of each test phase, for the modes seen
up to a lag before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special, stats

from surpyval.test_planning._checks import (
    check_positive,
    check_probability,
)
from surpyval.test_planning._serialise import dataclass_dict
from surpyval.utils.validation import check_alpha_ci

__all__ = [
    "IdealisedGrowthCurve",
    "PM2ContinuousPlan",
    "PM2DiscretePlan",
    "growth_demonstration_probability",
    "growth_test_time",
    "idealised_growth_curve",
    "pm2_continuous",
    "pm2_discrete",
]


# -- shared -----------------------------------------------------------------


def _times(t: npt.ArrayLike, name: str = "t") -> np.ndarray:
    array = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and not negative, got {t!r}.")
    return array


def _scalar(value: np.ndarray) -> Any:
    return value.item() if value.ndim == 0 else value


def _phases(
    ends: npt.ArrayLike, lags: npt.ArrayLike, total: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """The ends of the test phases (each a corrective action period after
    it), increasing and positive, and the lag of each, broadcast."""
    ends_ = np.atleast_1d(np.asarray(ends, dtype=float))
    if (
        ends_.ndim != 1
        or ends_.size == 0
        or np.any(~np.isfinite(ends_))
        or np.any(ends_ <= 0)
        or np.any(np.diff(ends_) <= 0)
    ):
        raise ValueError(
            "phase_ends must be the cumulative test times at which each "
            f"test phase ends, positive and increasing, got {ends!r}."
        )
    if total is not None and ends_[-1] > total * (1 + 1e-12):
        raise ValueError(
            f"phase_ends runs past the planned test time ({total:.6g}): "
            f"the last end is {ends_[-1]:.6g}."
        )
    lags_ = np.broadcast_to(np.asarray(lags, dtype=float), ends_.shape)
    if np.any(~np.isfinite(lags_)) or np.any(lags_ < 0):
        raise ValueError(
            f"lags must be finite and not negative, got {lags!r}."
        )
    return ends_, np.array(lags_)


def _styled_axes(ax: Any) -> Any:
    import matplotlib.pyplot as plt

    from surpyval import plotting

    if ax is None:
        _, ax = plt.subplots()
    plotting.style_axes(ax)
    return ax


def _plot_model_mtbf(ax: Any, model: Any, end: float) -> None:
    """Overlay a fitted growth model's instantaneous MTBF,
    ``1 / iif(t)``, from just after 0 to the end of its data."""
    from surpyval import plotting

    if not hasattr(model, "iif"):
        raise ValueError(
            "model must be a fitted recurrence model with an iif (for "
            "example surpyval.CrowAMSAA.fit(...)), got "
            f"{type(model).__name__}."
        )
    data = getattr(model, "data", None)
    last = end
    if data is not None and hasattr(data, "x"):
        x = np.asarray(data.x, dtype=float)
        if x.size:
            last = float(np.nanmax(x))
    t = np.linspace(last / 200.0, last, 200)
    with np.errstate(divide="ignore"):
        mtbf = 1.0 / np.asarray(model.iif(t), dtype=float)
    color = plotting.next_color(ax)
    ax.plot(t, mtbf, color=color, label="Fitted growth (1 / iif)")


# -- the idealised growth curve (AMSAA-Crow planning model) -------------------


@dataclass(frozen=True)
class IdealisedGrowthCurve:
    """The idealised reliability growth curve of MIL-HDBK-189C (the
    AMSAA-Crow planning model): returned by
    :func:`idealised_growth_curve`.

    The MTBF is ``initial_mtbf`` over the first test phase, which ends at
    ``initial_time``; the fixes at its end start the growth, and from then
    on the MTBF the system would show with every fix so far in is

    .. math::
        M(t) = \\frac{M_I}{1 - \\alpha}
               \\left(\\frac{t}{t_1}\\right)^{\\alpha}, \\quad t > t_1,

    reaching ``final_mtbf`` at the end of the test, ``test_time``. The
    expected number of failures by ``t > t_1`` is
    :math:`(t_1 / M_I) (t / t_1)^{1 - \\alpha}`: a Crow-AMSAA process with
    shape :math:`\\beta = 1 - \\alpha`, whose instantaneous MTBF is
    :math:`M(t)`.

    Attributes
    ----------
    initial_mtbf : float
        ``M_I``, the average MTBF over the first test phase.
    initial_time : float
        ``t_1``, the length of the first test phase.
    growth_rate : float
        ``alpha``, in (0, 1): the slope of the cumulative MTBF against
        test time on log-log axes.
    final_mtbf : float
        ``M_F = M(T)``, the MTBF at the end of the test.
    test_time : float
        ``T``, the total test time.

    Examples
    --------
    MIL-HDBK-189C's SPLAN example 1: 68 hours over the first 500, growing
    at 0.23 for 2800 hours.

    >>> from surpyval.test_planning import idealised_growth_curve
    >>> curve = idealised_growth_curve(
    ...     68.0, 500.0, growth_rate=0.23, test_time=2800.0
    ... )
    >>> round(curve.final_mtbf, 1), round(curve.mtbf(1000.0), 1)
    (131.3, 103.6)
    """

    initial_mtbf: float
    initial_time: float
    growth_rate: float
    final_mtbf: float
    test_time: float

    def mtbf(self, t: npt.ArrayLike) -> Any:
        """The idealised MTBF at test time ``t``: ``M_I`` up to
        ``initial_time``, :math:`M_I (t / t_1)^\\alpha / (1 - \\alpha)`
        after.

        Examples
        --------
        >>> from surpyval.test_planning import idealised_growth_curve
        >>> curve = idealised_growth_curve(
        ...     68.0, 500.0, growth_rate=0.23, test_time=2800.0
        ... )
        >>> round(curve.mtbf(2800.0), 1)
        131.3
        """
        t_ = _times(t)
        a, t1 = self.growth_rate, self.initial_time
        with np.errstate(divide="ignore"):
            grown = self.initial_mtbf / (1 - a) * (t_ / t1) ** a
        return _scalar(np.where(t_ <= t1, self.initial_mtbf, grown))

    def expected_failures(self, t: npt.ArrayLike) -> Any:
        """The expected number of failures by test time ``t``:
        ``t / M_I`` up to ``initial_time``, then
        :math:`(t_1 / M_I) (t / t_1)^{1 - \\alpha}`."""
        t_ = _times(t)
        a, t1, mi = self.growth_rate, self.initial_time, self.initial_mtbf
        grown = t1 / mi * (t_ / t1) ** (1 - a)
        return _scalar(np.where(t_ <= t1, t_ / mi, grown))

    def cumulative_mtbf(self, t: npt.ArrayLike) -> Any:
        """The cumulative MTBF by test time ``t > 0``, the test time over
        the expected failures: ``M_I`` up to ``initial_time``, then
        :math:`M_I (t / t_1)^\\alpha` (Duane's line)."""
        t_ = _times(t)
        if np.any(t_ <= 0):
            raise ValueError(f"t must be positive, got {t!r}.")
        return _scalar(t_ / np.asarray(self.expected_failures(t_)))

    def phase_mtbf(self, phase_ends: npt.ArrayLike) -> pd.DataFrame:
        """The planned MTBF of each test phase: the test time of the phase
        over its expected number of failures (MIL-HDBK-189C, section
        5.2.12), the average of the idealised curve over the phase.

        Parameters
        ----------
        phase_ends : array_like
            The cumulative test time at the end of each phase, increasing;
            the first phase starts at 0. Usually the first is
            ``initial_time`` and the last ``test_time``.

        Returns
        -------
        pandas.DataFrame
            One row per phase: ``start``, ``end``, ``expected_failures``
            in the phase and its average ``mtbf``.
        """
        ends, _ = _phases(phase_ends, 0.0, None)
        starts = np.concatenate([[0.0], ends[:-1]])
        failures = np.asarray(self.expected_failures(ends)) - np.asarray(
            self.expected_failures(starts)
        )
        return pd.DataFrame(
            {
                "start": starts,
                "end": ends,
                "expected_failures": failures,
                "mtbf": (ends - starts) / failures,
            },
            index=pd.RangeIndex(1, len(ends) + 1, name="phase"),
        )

    def demonstration_probability(
        self, requirement: float, *, alpha_ci: float = 0.05
    ) -> float:
        """The chance that the growth test's own data demonstrate
        ``requirement`` at confidence ``1 - alpha_ci``, if the system
        grows along this curve.

        The test passes when the lower ``1 - alpha_ci`` confidence bound
        on the MTBF at the end of the test, from a Crow-AMSAA fit to its
        failure times (Crow's exact time-terminated bound), is at least
        the requirement. See :func:`growth_demonstration_probability`;
        this is it with this curve's ``final_mtbf`` and expected number of
        failures. One minus it is the producer's risk of the plan.
        """
        return growth_demonstration_probability(
            self.final_mtbf / check_positive(requirement, "requirement"),
            float(self.expected_failures(self.test_time)),
            alpha_ci=alpha_ci,
        )

    def oc_curve(
        self,
        requirement: float,
        growth_rates: npt.ArrayLike,
        *,
        alpha_ci: float = 0.05,
    ) -> pd.DataFrame:
        """The operating characteristic of the growth test: the chance of
        demonstrating ``requirement`` had the system grown at each of
        ``growth_rates`` instead, from the same start (``initial_mtbf``
        over ``initial_time``) for the same ``test_time`` (MIL-HDBK-189C,
        section 5.3.7).

        Returns
        -------
        pandas.DataFrame
            One row per growth rate: ``growth_rate``, ``final_mtbf``,
            ``expected_failures`` and ``probability``.
        """
        rates = np.atleast_1d(np.asarray(growth_rates, dtype=float))
        rows = []
        for rate in rates:
            curve = idealised_growth_curve(
                self.initial_mtbf,
                self.initial_time,
                growth_rate=float(rate),
                test_time=self.test_time,
            )
            rows.append(
                {
                    "growth_rate": float(rate),
                    "final_mtbf": curve.final_mtbf,
                    "expected_failures": float(
                        curve.expected_failures(self.test_time)
                    ),
                    "probability": curve.demonstration_probability(
                        requirement, alpha_ci=alpha_ci
                    ),
                }
            )
        return pd.DataFrame(rows)

    def plot(
        self,
        ax: Any = None,
        *,
        phase_ends: npt.ArrayLike | None = None,
        model: Any = None,
        requirement: float | None = None,
    ) -> Any:
        """Plot the idealised curve against test time, with the planned
        MTBF of each phase as steps (``phase_ends``), a fitted growth
        model's instantaneous MTBF over the observed data (``model``,
        for example a :class:`surpyval.CrowAMSAA` fit to the test's
        failure times, on the same time scale) and the requirement as a
        line. Returns the axes."""
        from surpyval import plotting

        ax = _styled_axes(ax)
        t = np.linspace(0.0, self.test_time, 400)
        ax.plot(
            t,
            self.mtbf(t),
            color=plotting.next_color(ax),
            label="Idealised curve",
        )
        if phase_ends is not None:
            table = self.phase_mtbf(phase_ends)
            ax.hlines(
                table["mtbf"],
                table["start"],
                table["end"],
                color=plotting.next_color(ax),
                label="Planned phase MTBF",
            )
        if model is not None:
            _plot_model_mtbf(ax, model, self.test_time)
        if requirement is not None:
            ax.axhline(requirement, color="grey", ls="--", label="Requirement")
        ax.set_xlabel("Cumulative test time")
        ax.set_ylabel("MTBF")
        plotting.style_legend(ax)
        return ax

    def to_dict(self) -> dict:
        """The curve as a dict of native Python values, strict JSON."""
        return dataclass_dict(self)

    def __repr__(self) -> str:
        header = "Idealised Growth Curve"
        return "\n".join(
            [
                header,
                "=" * len(header),
                f"Initial MTBF       : {self.initial_mtbf:.6g}",
                f"Initial phase time : {self.initial_time:.6g}",
                f"Growth rate        : {self.growth_rate:.6g}",
                f"Final MTBF         : {self.final_mtbf:.6g}",
                f"Test time          : {self.test_time:.6g}",
            ]
        )


def idealised_growth_curve(
    initial_mtbf: float,
    initial_time: float,
    *,
    growth_rate: float | None = None,
    final_mtbf: float | None = None,
    test_time: float | None = None,
) -> IdealisedGrowthCurve:
    """The idealised reliability growth curve a test programme is planned
    against (MIL-HDBK-189C, section 5.2).

    In plain words: the system starts at ``initial_mtbf``, averaged over a
    first test phase of length ``initial_time``. Fixes made from then on
    raise its MTBF as a power of the test time, at ``growth_rate`` (0.3 to
    0.4 is a vigorous programme; under 0.2 a slow one). Give two of
    ``growth_rate``, ``final_mtbf`` and ``test_time`` and the third is
    solved for: how long to test, how fast reliability must grow, or what
    MTBF the test ends at.

    The curve is
    :math:`M(t) = M_I (t / t_1)^\\alpha / (1 - \\alpha)` for
    :math:`t > t_1`, and :math:`M_F = M(T)`. So the test time is
    :math:`T = t_1 [M_F (1 - \\alpha) / M_I]^{1 / \\alpha}`, and the
    growth rate is the root in (0, 1) of
    :math:`(T / t_1)^\\alpha / (1 - \\alpha) = M_F / M_I` (exact; the
    handbook's closed form is an approximation to it).

    Parameters
    ----------
    initial_mtbf : float
        ``M_I``, the average MTBF over the first test phase, positive.
    initial_time : float
        ``t_1``, the length of the first test phase, positive.
    growth_rate : float, optional
        ``alpha``, in (0, 1).
    final_mtbf : float, optional
        ``M_F``, the MTBF to reach at the end of the test.
    test_time : float, optional
        ``T``, the total test time, longer than ``initial_time``.

    Returns
    -------
    IdealisedGrowthCurve

    Raises
    ------
    ValueError
        Unless exactly two of ``growth_rate``, ``final_mtbf`` and
        ``test_time`` are given; for a ``final_mtbf`` not above
        ``initial_mtbf`` when the growth rate is solved for (no growth is
        needed); when the test time is solved for and the jump at the end
        of the first phase already reaches ``final_mtbf``
        (:math:`M_I / (1 - \\alpha) \\ge M_F`).

    Examples
    --------
    MIL-HDBK-189C's SPLAN example 1: 68 hours over the first 500, growing
    at 0.23 for 2800 hours.

    >>> from surpyval.test_planning import idealised_growth_curve
    >>> curve = idealised_growth_curve(
    ...     68.0, 500.0, growth_rate=0.23, test_time=2800.0
    ... )
    >>> round(curve.final_mtbf, 1)
    131.3
    >>> round(float(curve.expected_failures(2800.0)), 1)
    27.7

    How long to test to reach 130 hours, and what growth rate would reach
    it in 2800 hours:

    >>> round(idealised_growth_curve(
    ...     68.0, 500.0, growth_rate=0.23, final_mtbf=130.0
    ... ).test_time)
    2686
    >>> round(idealised_growth_curve(
    ...     68.0, 500.0, final_mtbf=130.0, test_time=2800.0
    ... ).growth_rate, 4)
    0.2268
    """
    mi = check_positive(initial_mtbf, "initial_mtbf")
    t1 = check_positive(initial_time, "initial_time")
    given = {
        "growth_rate": growth_rate,
        "final_mtbf": final_mtbf,
        "test_time": test_time,
    }
    missing = [k for k, v in given.items() if v is None]
    if len(missing) != 1:
        raise ValueError(
            "Give exactly two of growth_rate, final_mtbf and test_time; "
            "the one left None is solved for. Missing: "
            f"{missing or 'none'}."
        )
    if growth_rate is not None:
        a = check_probability(growth_rate, "growth_rate")
    if test_time is not None:
        T = check_positive(test_time, "test_time")
        if T <= t1:
            raise ValueError(
                f"test_time ({T:.6g}) must be longer than initial_time "
                f"({t1:.6g}): growth starts at the end of the first phase."
            )
    if final_mtbf is not None:
        mf = check_positive(final_mtbf, "final_mtbf")

    if final_mtbf is None:
        mf = mi / (1 - a) * (T / t1) ** a
    elif test_time is None:
        ratio = mf * (1 - a) / mi
        if ratio <= 1:
            raise ValueError(
                f"final_mtbf ({mf:.6g}) is reached by the jump at the end of "
                f"the first phase (initial_mtbf / (1 - growth_rate) = "
                f"{mi / (1 - a):.6g}): the test needs no time past "
                "initial_time. Give a higher final_mtbf or a lower "
                "growth_rate."
            )
        T = t1 * ratio ** (1 / a)
    else:
        if mf <= mi:
            raise ValueError(
                f"final_mtbf ({mf:.6g}) must be above initial_mtbf "
                f"({mi:.6g}) for there to be a growth rate to solve for."
            )
        log_ratio, log_time = np.log(mf / mi), np.log(T / t1)

        def gap(rate: float) -> float:
            return rate * log_time - np.log1p(-rate) - log_ratio

        a = optimize.brentq(gap, 1e-15, 1 - 1e-15, xtol=1e-15, rtol=1e-15)
    return IdealisedGrowthCurve(
        initial_mtbf=mi,
        initial_time=t1,
        growth_rate=float(a),
        final_mtbf=float(mf),
        test_time=float(T),
    )


def growth_demonstration_probability(
    mtbf_ratio: float,
    expected_failures: float,
    *,
    alpha_ci: float = 0.05,
) -> float:
    """The chance that a reliability growth test's own failure data
    demonstrate a requirement, when the system grows as a Crow-AMSAA
    (power-law) process (MIL-HDBK-189C, section 5.3.6).

    In plain words: at the end of the test a Crow-AMSAA model is fitted
    to the failure times, and the test passes if the lower confidence
    bound on the MTBF it ends at is at least the requirement. Two numbers
    settle the chance of that: how far the true final MTBF is above the
    requirement (``mtbf_ratio``, the growth test's discrimination ratio)
    and how many failures the test is expected to see. Neither the growth
    rate nor the test time matters beyond them.

    With :math:`\\mu` the expected number of failures and
    :math:`d = M(T) / M_R`, given ``n`` failures the shape statistic
    :math:`2 \\beta \\sum \\ln(T / t_i)` is chi-squared with ``2n``
    degrees of freedom, and the test passes when it is at least
    :math:`2 z_n / (\\mu d)`, where Crow's (1982) exact lower bound on the
    MTBF is :math:`n^2 \\hat M(T) / z_n`. So

    .. math::
        P = \\frac{1}{1 - e^{-\\mu}} \\sum_{n \\ge 1}
            \\frac{e^{-\\mu} \\mu^n}{n!}
            P\\left(\\chi^2_{2n} \\ge \\frac{2 z_n}{\\mu d}\\right),

    conditional on at least one failure (Broemm, Ellner and Woodworth
    2000). At ``mtbf_ratio = 1`` it is the consumer's risk, at most
    ``alpha_ci``.

    Parameters
    ----------
    mtbf_ratio : float
        The true MTBF at the end of the test over the requirement,
        positive.
    expected_failures : float
        The expected number of failures in the test, positive: for the
        idealised curve, ``T / ((1 - alpha) M(T))``.
    alpha_ci : float, optional
        Keyword only: one minus the confidence level of the
        demonstration, by default 0.05.

    Returns
    -------
    float

    Examples
    --------
    >>> from surpyval.test_planning import growth_demonstration_probability
    >>> round(growth_demonstration_probability(1.3, 28.0, alpha_ci=0.2), 3)
    0.51
    """
    from surpyval.recurrent.parametric.crow_amsaa import (
        crow_time_terminated_coefficients,
    )

    d = check_positive(mtbf_ratio, "mtbf_ratio")
    mu = check_positive(expected_failures, "expected_failures")
    check_alpha_ci(alpha_ci)
    level = float(alpha_ci)
    top = int(np.ceil(mu + 12.0 * np.sqrt(mu) + 30.0))
    n = np.arange(1, top + 1)
    z = np.array(
        [k**2 / crow_time_terminated_coefficients(int(k), level)[0] for k in n]
    )
    weights = stats.poisson.pmf(n, mu)
    passing = stats.chi2.sf(2.0 * z / (mu * d), 2 * n)
    return float(np.sum(weights * passing) / -np.expm1(-mu))


def growth_test_time(
    initial_mtbf: float,
    initial_time: float,
    growth_rate: float,
    requirement: float,
    *,
    probability: float = 0.8,
    alpha_ci: float = 0.05,
) -> IdealisedGrowthCurve:
    """The idealised growth curve whose test is just long enough to
    demonstrate ``requirement`` from its own data with chance
    ``probability`` (MIL-HDBK-189C, section 5.3.8): one minus
    ``probability`` is the producer's risk.

    The longer the test, the higher the MTBF it ends at and the more
    failures it sees, so the chance of demonstrating rises with the test
    time; the test time solved for is the shortest with that chance. The
    chance moves in small steps (Crow's bound changes with each failure),
    so it is reached within about 0.001.

    Parameters
    ----------
    initial_mtbf, initial_time, growth_rate : float
        As in :func:`idealised_growth_curve`.
    requirement : float
        The MTBF to demonstrate, positive.
    probability : float, optional
        Keyword only: the chance of demonstrating it, by default 0.8.
    alpha_ci : float, optional
        Keyword only: one minus the confidence level of the
        demonstration, by default 0.05.

    Returns
    -------
    IdealisedGrowthCurve

    Examples
    --------
    MIL-HDBK-189C's SPLAN example 2: 48 hours over the first 500, growing
    at 0.3, to demonstrate 100 hours at 80% confidence with chance 0.8.

    >>> from surpyval.test_planning import growth_test_time
    >>> curve = growth_test_time(
    ...     48.0, 500.0, 0.3, 100.0, probability=0.8, alpha_ci=0.2
    ... )
    >>> round(curve.test_time, -1)
    5270.0
    """
    mi = check_positive(initial_mtbf, "initial_mtbf")
    t1 = check_positive(initial_time, "initial_time")
    a = check_probability(growth_rate, "growth_rate")
    mr = check_positive(requirement, "requirement")
    target = check_probability(probability, "probability")
    check_alpha_ci(alpha_ci)

    def chance(log_t: float) -> float:
        curve = idealised_growth_curve(
            mi, t1, growth_rate=a, test_time=t1 * (1.0 + np.exp(log_t))
        )
        return curve.demonstration_probability(mr, alpha_ci=alpha_ci)

    lo, hi = np.log(1e-6), np.log(10.0)
    while chance(hi) < target:
        hi += 1.0
        if hi > 60.0:
            raise ValueError(
                "No test time demonstrates the requirement with chance "
                f"{target}: check requirement and growth_rate."
            )
    if chance(lo) >= target:
        log_t = lo
    else:
        log_t = optimize.brentq(
            lambda u: chance(u) - target, lo, hi, xtol=1e-10
        )
    return idealised_growth_curve(
        mi, t1, growth_rate=a, test_time=t1 * (1.0 + np.exp(log_t))
    )


# -- PM2-Continuous -----------------------------------------------------------


@dataclass(frozen=True)
class PM2ContinuousPlan:
    """A PM2-Continuous reliability growth plan (MIL-HDBK-189C, section
    5.5): returned by :func:`pm2_continuous`.

    The initial failure intensity :math:`\\lambda_I = 1 / M_I` splits into
    the part corrective action will not address,
    :math:`\\lambda_A = (1 - MS) \\lambda_I`, and the part it will,
    :math:`\\lambda_B = MS \\lambda_I`. Correctable (B) modes are found at
    the rate :math:`h(t) = \\lambda_B / (1 + \\beta t)`, and once every B
    mode seen by ``t`` is fixed with average effectiveness
    :math:`\\mu_d`, the failure intensity is

    .. math::
        \\rho(t) = \\lambda_A + (1 - \\mu_d) (\\lambda_B - h(t)) + h(t).

    :math:`\\beta` is set so that :math:`1 / \\rho(T)` is the goal.

    Attributes
    ----------
    initial_mtbf : float
        ``M_I``.
    goal_mtbf : float
        ``M_G``, the MTBF at the end of the test with every fix in.
    test_time : float
        ``T``.
    management_strategy : float
        ``MS``, the share of the initial failure intensity corrective
        action addresses.
    fef : float
        :math:`\\mu_d`, the average fix effectiveness factor.
    beta : float
        The scale of the rate of new B modes, per unit test time.
    growth_potential : float
        :math:`M_{GP} = M_I / (1 - MS \\mu_d)`, the MTBF were every B mode
        found and fixed.

    Examples
    --------
    MIL-HDBK-189C's PM2 example (figure 27): 25 hours initially, a goal of
    90 hours after 30,000 hours of test.

    >>> from surpyval.test_planning import pm2_continuous
    >>> plan = pm2_continuous(
    ...     25.0, 90.0, 30000.0, management_strategy=0.95, fef=0.8
    ... )
    >>> round(float(plan.mtbf(0.0)), 6), round(float(plan.mtbf(30000)), 6)
    (25.0, 90.0)
    >>> round(plan.growth_potential, 2)
    104.17
    """

    initial_mtbf: float
    goal_mtbf: float
    test_time: float
    management_strategy: float
    fef: float
    beta: float
    growth_potential: float

    @property
    def goal_to_potential(self) -> float:
        """``M_G / M_GP``. MIL-HDBK-189C's rule of thumb is 0.6 to 0.8 for
        a well-managed programme that is not overly aggressive."""
        return self.goal_mtbf / self.growth_potential

    @property
    def lambda_a(self) -> float:
        """The failure intensity of the modes that will not be fixed."""
        return (1 - self.management_strategy) / self.initial_mtbf

    @property
    def lambda_b(self) -> float:
        """The initial failure intensity of the modes that will be."""
        return self.management_strategy / self.initial_mtbf

    def new_mode_rate(self, t: npt.ArrayLike) -> Any:
        """:math:`h(t) = \\lambda_B / (1 + \\beta t)`, the rate at which
        new B modes are found at test time ``t``: the failure intensity
        of the B modes not yet seen."""
        t_ = _times(t)
        return _scalar(self.lambda_b / (1 + self.beta * t_))

    def expected_modes(self, t: npt.ArrayLike) -> Any:
        """The expected number of distinct B modes seen by test time
        ``t``, :math:`(\\lambda_B / \\beta) \\ln(1 + \\beta t)`: the
        redesigns the programme should expect."""
        t_ = _times(t)
        return _scalar(self.lambda_b / self.beta * np.log1p(self.beta * t_))

    def fraction_surfaced(self, t: npt.ArrayLike) -> Any:
        """:math:`\\theta(t) = \\beta t / (1 + \\beta t)`, the share of
        the initial B-mode failure intensity in the modes seen by ``t``:
        the most that fixing them could remove."""
        t_ = _times(t)
        return _scalar(self.beta * t_ / (1 + self.beta * t_))

    def intensity(self, t: npt.ArrayLike) -> Any:
        """:math:`\\rho(t)`, the failure intensity once every B mode seen
        by ``t`` is fixed."""
        t_ = _times(t)
        h = self.lambda_b / (1 + self.beta * t_)
        rho = self.lambda_a + (1 - self.fef) * (self.lambda_b - h) + h
        return _scalar(rho)

    def mtbf(self, t: npt.ArrayLike) -> Any:
        """The idealised MTBF :math:`1 / \\rho(t)`: what the system would
        show at test time ``t`` were every B mode seen by then fixed.

        Examples
        --------
        >>> from surpyval.test_planning import pm2_continuous
        >>> plan = pm2_continuous(
        ...     25.0, 90.0, 30000.0, management_strategy=0.95, fef=0.8
        ... )
        >>> round(float(plan.mtbf(0.0)), 6), round(float(plan.mtbf(30000)), 6)
        (25.0, 90.0)
        """
        return _scalar(1.0 / np.asarray(self.intensity(t)))

    def targets(
        self, phase_ends: npt.ArrayLike, lags: npt.ArrayLike = 0.0
    ) -> pd.DataFrame:
        """The planned MTBF of each test phase, the steps of the plan
        (MIL-HDBK-189C, section 5.5.8).

        A test phase runs at a fixed configuration; the fixes go in at the
        corrective action period after it, for the B modes seen up to
        ``lag`` before its end. So the first phase is at ``M_I``, and phase
        ``i + 1`` at :math:`1 / \\rho(t_i - L_i)`. The step after the last
        phase is the MTBF the test leaves the system at.

        Parameters
        ----------
        phase_ends : array_like
            The cumulative test time at the end of each phase (each
            followed by a corrective action period), increasing.
        lags : array_like, optional
            The corrective action lag before each period, in test time: a
            mode must be seen this long before the period for its fix to
            go in. One number for every period, or one each. By default 0.

        Returns
        -------
        pandas.DataFrame
            One row per phase and one for after the test: ``start``,
            ``end`` (``inf`` for the last row), ``fixes_through`` (the
            test time whose modes are fixed: ``nan`` for the first phase),
            ``mtbf``.
        """
        ends, lag = _phases(phase_ends, lags, self.test_time)
        through = np.clip(ends - lag, 0.0, None)
        starts = np.concatenate([[0.0], ends])
        stops = np.concatenate([ends, [np.inf]])
        mtbf = np.concatenate(
            [[self.initial_mtbf], np.atleast_1d(self.mtbf(through))]
        )
        return pd.DataFrame(
            {
                "start": starts,
                "end": stops,
                "fixes_through": np.concatenate([[np.nan], through]),
                "mtbf": mtbf,
            },
            index=pd.RangeIndex(1, len(starts) + 1, name="phase"),
        )

    def demonstration_probability(
        self,
        requirement: float,
        demonstration_time: float,
        *,
        alpha_ci: float = 0.05,
        degradation: float = 0.0,
    ) -> float:
        """The chance that a fixed-configuration demonstration test after
        the growth test (an operational test) demonstrates
        ``requirement`` at confidence ``1 - alpha_ci``, as MIL-HDBK-189C's
        PM2 plans judge a goal (section 5.5.8).

        The system enters the demonstration at the goal MTBF less
        ``degradation`` (the share lost moving from the development to the
        operational environment), so at a constant failure rate
        ``1 / (goal_mtbf (1 - degradation))``. The demonstration passes
        with the most failures ``c`` for which
        :func:`~surpyval.test_planning.demonstrated_mtbf` of
        ``demonstration_time`` with ``c`` failures is at least the
        requirement; the chance is that of at most ``c`` failures.

        Parameters
        ----------
        requirement : float
            The MTBF to demonstrate, positive.
        demonstration_time : float
            The length of the demonstration test, in the same unit.
        alpha_ci : float, optional
            Keyword only: one minus the demonstration's confidence level,
            by default 0.05.
        degradation : float, optional
            Keyword only: the share of the MTBF lost entering the
            demonstration, in [0, 1), by default 0.

        Returns
        -------
        float
            0 when no number of failures, not even none, demonstrates the
            requirement in that time.
        """
        mr = check_positive(requirement, "requirement")
        length = check_positive(demonstration_time, "demonstration_time")
        check_alpha_ci(alpha_ci)
        level = float(alpha_ci)
        if not (0.0 <= degradation < 1.0):
            raise ValueError(
                f"degradation must be in [0, 1), got {degradation!r}."
            )
        entering = self.goal_mtbf * (1.0 - degradation)
        return _demonstration_chance(mr, length, level, entering)

    def plot(
        self,
        ax: Any = None,
        *,
        phase_ends: npt.ArrayLike | None = None,
        lags: npt.ArrayLike = 0.0,
        model: Any = None,
        requirement: float | None = None,
    ) -> Any:
        """Plot the idealised curve against test time, with the planned
        steps (``phase_ends`` and ``lags``, as :meth:`targets` takes them),
        a fitted growth model's instantaneous MTBF over the observed data
        (``model``, for example a :class:`surpyval.CrowAMSAA` fit) and the
        requirement. Returns the axes."""
        from surpyval import plotting

        ax = _styled_axes(ax)
        t = np.linspace(0.0, self.test_time, 400)
        ax.plot(
            t,
            self.mtbf(t),
            color=plotting.next_color(ax),
            label="Idealised curve",
        )
        if phase_ends is not None:
            table = self.targets(phase_ends, lags)
            ends = table["end"].to_numpy().copy()
            ends[-1] = self.test_time
            keep = table["start"].to_numpy() < self.test_time
            ax.hlines(
                table["mtbf"][keep],
                table["start"][keep],
                ends[keep],
                color=plotting.next_color(ax),
                label="Planned phase MTBF",
            )
        if model is not None:
            _plot_model_mtbf(ax, model, self.test_time)
        if requirement is not None:
            ax.axhline(requirement, color="grey", ls="--", label="Requirement")
        ax.set_xlabel("Cumulative test time")
        ax.set_ylabel("MTBF")
        plotting.style_legend(ax)
        return ax

    def to_dict(self) -> dict:
        """The plan as a dict of native Python values, strict JSON."""
        return dataclass_dict(self)

    def __repr__(self) -> str:
        header = "PM2-Continuous Plan"
        return "\n".join(
            [
                header,
                "=" * len(header),
                f"Initial MTBF        : {self.initial_mtbf:.6g}",
                f"Goal MTBF           : {self.goal_mtbf:.6g}",
                f"Test time           : {self.test_time:.6g}",
                f"Management strategy : {self.management_strategy:.6g}",
                f"Average FEF         : {self.fef:.6g}",
                f"Growth potential    : {self.growth_potential:.6g}",
                f"Goal / potential    : {self.goal_to_potential:.6g}",
                f"beta                : {self.beta:.6g}",
            ]
        )


def _demonstration_chance(
    requirement: float, length: float, level: float, mtbf: float
) -> float:
    """The chance a constant-rate demonstration test of ``length`` passes
    at the system's ``mtbf``: at most ``c`` failures, ``c`` the most whose
    chi-squared lower bound ``2 length / chi2(1 - level, 2c + 2)`` is at
    least the requirement."""
    # The bound falls with c: the largest c with chi2 quantile at most
    # 2 length / requirement.
    limit = 2.0 * length / requirement
    if stats.chi2.ppf(1.0 - level, 2) > limit:
        return 0.0
    c = 0
    while stats.chi2.ppf(1.0 - level, 2 * (c + 1) + 2) <= limit:
        c += 1
    return float(stats.poisson.cdf(c, length / mtbf))


def pm2_continuous(
    initial_mtbf: float,
    goal_mtbf: float,
    test_time: float,
    *,
    management_strategy: float,
    fef: float,
) -> PM2ContinuousPlan:
    """A PM2-Continuous reliability growth plan (MIL-HDBK-189C, section
    5.5).

    In plain words: the plan is set by what management controls. Of the
    system's initial failure intensity, a share ``management_strategy``
    is in failure modes corrective action will address; each fix removes,
    on average, a share ``fef`` of its mode's intensity. Testing finds
    those modes, quickly at first and then more slowly, and the plan
    finds the pace of discovery that takes the system from
    ``initial_mtbf`` to ``goal_mtbf`` in ``test_time``. It gives the MTBF
    the system would have at any test time were every mode seen so far
    fixed (:meth:`~PM2ContinuousPlan.mtbf`), the steps after each
    corrective action period (:meth:`~PM2ContinuousPlan.targets`), the
    expected number of modes to fix, and the share of the correctable
    intensity seen. No growth rate is assumed.

    The pace :math:`\\beta` solves :math:`\\rho(T) = 1 / M_G`:

    .. math::
        \\beta = \\frac{1}{T}
                 \\frac{1 - M_I / M_G}{M_I / M_G - M_I / M_{GP}},
        \\qquad M_{GP} = \\frac{M_I}{1 - MS\\, \\mu_d}.

    Parameters
    ----------
    initial_mtbf : float
        ``M_I``, the system's MTBF at the start of the test, positive.
    goal_mtbf : float
        ``M_G``, the MTBF to reach by the end of the test with every fix
        in, between ``initial_mtbf`` and the growth potential.
    test_time : float
        ``T``, the test time, positive.
    management_strategy : float
        Keyword only: ``MS``, in (0, 1].
    fef : float
        Keyword only: :math:`\\mu_d`, the average fix effectiveness, in
        (0, 1].

    Returns
    -------
    PM2ContinuousPlan

    Raises
    ------
    ValueError
        For a goal at or above the growth potential, which no amount of
        testing reaches (raise ``management_strategy`` or ``fef``, or
        lower the goal), or at or below ``initial_mtbf``.

    Examples
    --------
    MIL-HDBK-189C's PM2 example (figure 27): 25 hours initially, a goal of
    90, management strategy 0.95 and average FEF 0.8.

    >>> from surpyval.test_planning import pm2_continuous
    >>> plan = pm2_continuous(
    ...     25.0, 90.0, 30000.0, management_strategy=0.95, fef=0.8
    ... )
    >>> round(plan.growth_potential, 2), round(plan.goal_to_potential, 3)
    (104.17, 0.864)
    """
    mi = check_positive(initial_mtbf, "initial_mtbf")
    mg = check_positive(goal_mtbf, "goal_mtbf")
    T = check_positive(test_time, "test_time")
    ms = _check_share(management_strategy, "management_strategy")
    d = _check_share(fef, "fef")
    mgp = mi / (1 - ms * d) if ms * d < 1 else np.inf
    if not mi < mg < mgp:
        raise ValueError(
            f"goal_mtbf ({mg:.6g}) must be above initial_mtbf ({mi:.6g}) "
            f"and below the growth potential ({mgp:.6g}, initial_mtbf / "
            "(1 - management_strategy * fef)), the most that fixing every "
            "correctable mode could reach: raise management_strategy or "
            "fef, or lower the goal."
        )
    beta = (1 - mi / mg) / (mi / mg - mi / mgp) / T
    return PM2ContinuousPlan(
        initial_mtbf=mi,
        goal_mtbf=mg,
        test_time=T,
        management_strategy=ms,
        fef=d,
        beta=float(beta),
        growth_potential=float(mgp),
    )


def _check_share(value: Any, name: str) -> float:
    try:
        inside = bool(0.0 < value <= 1.0)
    except (TypeError, ValueError):
        inside = False
    if not inside:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}.")
    return float(value)


# -- PM2-Discrete -------------------------------------------------------------


@dataclass(frozen=True)
class PM2DiscretePlan:
    """A PM2-Discrete reliability growth plan for one-shot items
    (MIL-HDBK-189C, section 5.6): returned by :func:`pm2_discrete`.

    The initial reliability on a trial splits as :math:`R_I = R_A R_B`,
    :math:`R_A = R_I^{1 - MS}` the chance no unaddressed (A) mode occurs
    and :math:`R_B = R_I^{MS}` the chance no correctable (B) mode does.
    The B modes' chances of occurring on a trial are a beta sample with
    shape ``n``; once every B mode seen before trial ``t`` is fixed with
    average effectiveness :math:`\\mu_d`, the reliability on trial ``t``
    is

    .. math::
        R(t) = R_A\\, R_B^{1 - \\mu_d (t - 1) / (n + t - 1)}.

    ``n`` is set so that :math:`R(T_L)` is the goal.

    Attributes
    ----------
    initial_reliability : float
        ``R_I``, the reliability on the first trial.
    goal_reliability : float
        ``R_G``.
    trials : float
        ``T_L``, the trial at which the goal is reached: the last trial
        whose modes are fixed in the last corrective action period.
    management_strategy : float
        ``MS``, the share of :math:`-\\ln R_I` in correctable modes.
    fef : float
        :math:`\\mu_d`.
    n : float
        The beta shape.
    growth_potential : float
        :math:`R_{GP} = R_I^{1 - MS \\mu_d}`.

    Examples
    --------
    >>> from surpyval.test_planning import pm2_discrete
    >>> plan = pm2_discrete(
    ...     0.8987, 0.9639, 400, management_strategy=0.9, fef=0.8
    ... )
    >>> round(float(plan.reliability(1)), 4)
    0.8987
    >>> round(float(plan.reliability(400)), 4)
    0.9639
    >>> round(plan.growth_potential, 4)
    0.9705
    """

    initial_reliability: float
    goal_reliability: float
    trials: float
    management_strategy: float
    fef: float
    n: float
    growth_potential: float

    @property
    def r_a(self) -> float:
        """The chance no unaddressed (A) mode occurs on a trial."""
        return self.initial_reliability ** (1 - self.management_strategy)

    @property
    def r_b(self) -> float:
        """The initial chance no correctable (B) mode occurs on a trial."""
        return self.initial_reliability**self.management_strategy

    def _trials(self, t: npt.ArrayLike) -> np.ndarray:
        t_ = np.asarray(t, dtype=float)
        if np.any(~np.isfinite(t_)) or np.any(t_ < 1):
            raise ValueError(f"t must be a trial number, 1 or more: {t!r}.")
        return t_

    def reliability(self, t: npt.ArrayLike) -> Any:
        """The idealised reliability on trial ``t`` once every B mode seen
        before it is fixed.

        Examples
        --------
        >>> from surpyval.test_planning import pm2_discrete
        >>> plan = pm2_discrete(
        ...     0.8987, 0.9639, 400, management_strategy=0.9, fef=0.8
        ... )
        >>> round(float(plan.reliability(1)), 4)
        0.8987
        >>> round(float(plan.reliability(400)), 4)
        0.9639
        """
        t_ = self._trials(t)
        share = self.fef * (t_ - 1) / (self.n + t_ - 1)
        return _scalar(self.r_a * self.r_b ** (1 - share))

    def expected_modes(self, t: npt.ArrayLike) -> Any:
        """The expected number of distinct B modes seen in the first
        ``t`` trials,
        :math:`-n \\ln R_B [\\psi(n + t) - \\psi(n)]`."""
        t_ = self._trials(t)
        dig = special.digamma(self.n + t_) - special.digamma(self.n)
        return _scalar(-self.n * np.log(self.r_b) * dig)

    def fraction_surfaced(self, t: npt.ArrayLike) -> Any:
        """The expected share of the initial B-mode probability of
        failure in the modes seen before trial ``t``,
        :math:`(1 - R_B^{(t - 1) / (n + t - 1)}) / (1 - R_B)`: 0 on the
        first trial, rising to 1."""
        t_ = self._trials(t)
        seen = 1 - self.r_b ** ((t_ - 1) / (self.n + t_ - 1))
        return _scalar(seen / (1 - self.r_b))

    def new_mode_probability(self, t: npt.ArrayLike) -> Any:
        """The chance that trial ``t`` fails from a B mode not seen
        before, :math:`1 - R_B^{n / (n + t - 1)}`."""
        t_ = self._trials(t)
        return _scalar(1 - self.r_b ** (self.n / (self.n + t_ - 1)))

    def targets(
        self, phase_ends: npt.ArrayLike, lags: npt.ArrayLike = 0
    ) -> pd.DataFrame:
        """The planned reliability of each test phase, in trials: the
        first at ``R_I``, and phase ``i + 1`` at :math:`R(t_i - L_i + 1)`,
        every B mode seen in the trials up to ``lags`` before the end of
        phase ``i`` fixed (MIL-HDBK-189C, section 5.6.6).

        Parameters
        ----------
        phase_ends : array_like
            The cumulative number of trials at the end of each phase
            (each followed by a corrective action period), increasing.
        lags : array_like, optional
            The corrective action lag before each period, in trials. By
            default 0.

        Returns
        -------
        pandas.DataFrame
            One row per phase and one for after the test: ``start`` (the
            trials before it), ``end`` (``inf`` for the last row),
            ``fixes_through`` (the last trial whose modes are fixed:
            ``nan`` for the first phase), ``reliability``.
        """
        ends, lag = _phases(phase_ends, lags, None)
        through = np.clip(ends - lag, 0.0, None)
        starts = np.concatenate([[0.0], ends])
        stops = np.concatenate([ends, [np.inf]])
        reliability = np.concatenate(
            [
                [self.initial_reliability],
                np.atleast_1d(self.reliability(through + 1)),
            ]
        )
        return pd.DataFrame(
            {
                "start": starts,
                "end": stops,
                "fixes_through": np.concatenate([[np.nan], through]),
                "reliability": reliability,
            },
            index=pd.RangeIndex(1, len(starts) + 1, name="phase"),
        )

    def plot(
        self,
        ax: Any = None,
        *,
        phase_ends: npt.ArrayLike | None = None,
        lags: npt.ArrayLike = 0,
        requirement: float | None = None,
    ) -> Any:
        """Plot the idealised reliability against trials, with the
        planned steps (``phase_ends`` and ``lags``, as :meth:`targets`
        takes them) and the requirement. Returns the axes."""
        from surpyval import plotting

        ax = _styled_axes(ax)
        t = np.linspace(1.0, self.trials, 400)
        ax.plot(
            t,
            self.reliability(t),
            color=plotting.next_color(ax),
            label="Idealised curve",
        )
        if phase_ends is not None:
            table = self.targets(phase_ends, lags)
            keep = table["start"].to_numpy() < self.trials
            ends = np.minimum(table["end"].to_numpy(), self.trials)
            ax.hlines(
                table["reliability"][keep],
                table["start"][keep],
                ends[keep],
                color=plotting.next_color(ax),
                label="Planned phase reliability",
            )
        if requirement is not None:
            ax.axhline(requirement, color="grey", ls="--", label="Requirement")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Reliability")
        plotting.style_legend(ax)
        return ax

    def to_dict(self) -> dict:
        """The plan as a dict of native Python values, strict JSON."""
        return dataclass_dict(self)

    def __repr__(self) -> str:
        header = "PM2-Discrete Plan"
        return "\n".join(
            [
                header,
                "=" * len(header),
                f"Initial reliability : {self.initial_reliability:.6g}",
                f"Goal reliability    : {self.goal_reliability:.6g}",
                f"Trials              : {self.trials:.6g}",
                f"Management strategy : {self.management_strategy:.6g}",
                f"Average FEF         : {self.fef:.6g}",
                f"Growth potential    : {self.growth_potential:.6g}",
                f"n                   : {self.n:.6g}",
            ]
        )


def pm2_discrete(
    initial_reliability: float,
    goal_reliability: float,
    trials: float,
    *,
    management_strategy: float,
    fef: float,
) -> PM2DiscretePlan:
    """A PM2-Discrete reliability growth plan for one-shot items, tested
    in trials (MIL-HDBK-189C, section 5.6).

    In plain words: as :func:`pm2_continuous`, but the system succeeds
    or fails on each trial (a launch, a shot, a start). A share
    ``management_strategy`` of its initial unreliability, on the log
    scale, is in modes that corrective action will address, each fix
    removing a share ``fef`` of its mode's chance; the plan finds the
    pace of discovery that reaches ``goal_reliability`` at trial
    ``trials``.

    The beta shape solves :math:`R(T_L) = R_G`:

    .. math::
        n = (T_L - 1) \\frac{\\ln(R_{GP} / R_G)}{\\ln(R_G / R_I)},
        \\qquad R_{GP} = R_I^{1 - MS\\, \\mu_d}.

    Parameters
    ----------
    initial_reliability : float
        ``R_I``, the reliability on the first trial, in (0, 1).
    goal_reliability : float
        ``R_G``, between ``initial_reliability`` and the growth potential.
    trials : float
        ``T_L``, the trial at which the goal is to be reached (the last
        trial whose modes the last corrective action period fixes),
        more than 1.
    management_strategy : float
        Keyword only: ``MS``, in (0, 1].
    fef : float
        Keyword only: :math:`\\mu_d`, in (0, 1].

    Returns
    -------
    PM2DiscretePlan

    Raises
    ------
    ValueError
        For a goal at or above the growth potential (raise
        ``management_strategy`` or ``fef``, or lower the goal), or at or
        below ``initial_reliability``.

    Examples
    --------
    >>> from surpyval.test_planning import pm2_discrete
    >>> plan = pm2_discrete(
    ...     0.8987, 0.9639, 400, management_strategy=0.9, fef=0.8
    ... )
    >>> round(plan.growth_potential, 4), round(plan.n, 2)
    (0.9705, 39.09)
    """
    ri = check_probability(initial_reliability, "initial_reliability")
    rg = check_probability(goal_reliability, "goal_reliability")
    tl = check_positive(trials, "trials")
    if tl <= 1:
        raise ValueError(f"trials must be more than 1, got {trials!r}.")
    ms = _check_share(management_strategy, "management_strategy")
    d = _check_share(fef, "fef")
    rgp = ri ** (1 - ms * d)
    if not ri < rg < rgp:
        raise ValueError(
            f"goal_reliability ({rg:.6g}) must be above "
            f"initial_reliability ({ri:.6g}) and below the growth "
            f"potential ({rgp:.6g}, initial_reliability ** (1 - "
            "management_strategy * fef)): raise management_strategy or "
            "fef, or lower the goal."
        )
    n = (tl - 1) * np.log(rgp / rg) / np.log(rg / ri)
    return PM2DiscretePlan(
        initial_reliability=ri,
        goal_reliability=rg,
        trials=tl,
        management_strategy=ms,
        fef=d,
        n=float(n),
        growth_potential=float(rgp),
    )
