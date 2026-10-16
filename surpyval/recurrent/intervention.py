r"""
Did an intervention change the rate of failures, and is the evidence
enough to say?

An *intervention* -- a design change, a new maintenance procedure, a
corrective action -- is made on one or more repairable systems at a known
time. :func:`intervention_test` compares the rate of failures after it
with the rate before it, on the systems' failure records, and says
whether the rate changed, by how much (the ratio of the rates, with a
confidence interval), and whether the exposure seen so far was enough to
detect the change hoped for (the power, the smallest change it could
detect, and how much more exposure would be needed).

Two models of the rate are offered:

``model="hpp"`` (the default)
    The rate is constant before the intervention and constant after it (a
    homogeneous Poisson process, HPP, on each side). The test is the
    exact conditional test of the ratio of two Poisson rates
    (Przyborowski and Wilenski 1940; Cox and Lewis 1966, section 9.4):
    given the failures in all, those after the intervention are binomial,
    whatever the rates, so the test and its confidence interval for the
    ratio (the Clopper-Pearson interval, transformed) are exact. A mid-p
    version (Lancaster 1961) is the option ``method="mid-p"``. With
    ``baseline="per-system"`` (the default) each system keeps its own
    rate and only the ratio is shared: the test is then the exact
    stratified test (the sum of the systems' binomial counts), which is
    not confounded when systems with different rates have different
    shares of their exposure after the intervention.

``model="power-law"``
    The rate changes with age as a power law (the Crow-AMSAA process),
    :math:`u(t) = \lambda \beta t^{\beta - 1}`, and the intervention
    multiplies it by :math:`\rho`, keeping its shape:
    :math:`u(t) = \rho\,\lambda \beta t^{\beta - 1}` after it. The ratio
    is tested by the likelihood ratio, and its interval is the profile-
    likelihood interval (Rigdon and Basu 2000, chapter 5; Cook and Lawless
    2007, section 2.5). Use it when the rate trends within the periods
    (wear-out, or reliability growth), which an HPP comparison would
    mistake for an effect of the intervention.

References
----------
Przyborowski, J. and Wilenski, H. (1940), "Homogeneity of results in
testing samples from Poisson series", Biometrika 31, 313-323.

Cox, D. R. and Lewis, P. A. W. (1966), "The Statistical Analysis of
Series of Events", Methuen.

Lancaster, H. O. (1961), "Significance tests in discrete distributions",
Journal of the American Statistical Association 56, 223-234.

Breslow, N. E. and Day, N. E. (1987), "Statistical Methods in Cancer
Research, Volume II: The Design and Analysis of Cohort Studies", IARC,
chapter 3 (the stratified comparison of rates).

Rigdon, S. E. and Basu, A. P. (2000), "Statistical Methods for the
Reliability of Repairable Systems", Wiley.

Cook, R. J. and Lawless, J. F. (2007), "The Statistical Analysis of
Recurrent Events", Springer.

Gu, K., Ng, H. K. T., Tang, M. L. and Schucany, W. R. (2008), "Testing
the ratio of two Poisson rates", Biometrical Journal 50, 283-298.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from surpyval.test_planning.rate_ratio import (
    detectable_ratio,
    expected_power,
    exposure_multiple,
)
from surpyval.utils.data_formats import refuse_time_values
from surpyval.utils.recurrent_utils import item_label, number_text
from surpyval.utils.validation import check_alpha_ci, check_option

ALTERNATIVES = ("two-sided", "decrease", "increase")
MODELS = ("hpp", "power-law")
BASELINES = ("per-system", "common")
HPP_METHODS = ("exact", "mid-p")
POWER_LAW_METHODS = ("likelihood-ratio",)
VERDICTS = (
    "decrease",
    "increase",
    "no change detected",
    "inconclusive",
    "untestable",
)

# The search range of log(shape) for the power-law likelihood.
_LOG_SHAPE_RANGE = (-5.0, 5.0)
_LOG_SHAPE_GRID = 41
# The search range of log(ratio) for the bounds.
_LOG_RATIO_LIMIT = 60.0


@dataclass
class _System:
    """One system's events and observation windows, and its intervention
    time."""

    label: Any
    events: npt.NDArray
    windows: list
    intervention: float


@dataclass
class _Split:
    """The data on each side of the intervention, system by system: the
    failures (``before``, ``after``), the exposure (``exposure_before``,
    ``exposure_after``) and the pieces of the windows on each side, as
    ``(system index, start, end)`` rows."""

    labels: list
    before: npt.NDArray
    after: npt.NDArray
    exposure_before: npt.NDArray
    exposure_after: npt.NDArray
    pieces_before: npt.NDArray
    pieces_after: npt.NDArray
    events: npt.NDArray
    event_after: npt.NDArray
    event_system: npt.NDArray


@dataclass
class InterventionTestResult:
    """The result of :func:`intervention_test`: did the rate of failures
    change at the intervention, and was there enough evidence to tell?

    The ratio is the rate after the intervention over the rate before:
    below 1 is an improvement, and ``1 - ratio`` is the reduction.

    Attributes
    ----------
    model : str
        ``"hpp"`` or ``"power-law"``.
    baseline : str
        ``"per-system"`` (each system its own rate, the ratio shared) or
        ``"common"`` (one rate for every system).
    method : str
        The test: ``"exact"`` or ``"mid-p"`` (the HPP), or
        ``"likelihood-ratio"`` (the power law).
    alternative : str
        ``"two-sided"``, ``"decrease"`` or ``"increase"``.
    alpha_ci : float
        The significance level of the test, and one minus the confidence
        level of ``ratio_cb``.
    events_before, events_after : int
        The failures before and after the intervention, of the systems
        compared.
    exposure_before, exposure_after : float
        The observed time before and after the intervention, of the
        systems compared.
    expected_after : float
        The failures expected after the intervention had the rate not
        changed, given the failures seen.
    ratio : float
        The estimated ratio of the rates (``nan`` when untestable; 0 with
        no failure after the intervention).
    ratio_cb : numpy.ndarray
        Its confidence interval, ``[lower, upper]``: two-sided for a
        two-sided test, ``[0, upper]`` for ``alternative="decrease"`` and
        ``[lower, inf]`` for ``"increase"``.
    p_value : float
        The p-value of the test of no change.
    statistic : float
        The test statistic: the failures after the intervention (the HPP
        tests), or the likelihood-ratio statistic (the power law).
    change : str
        The test's conclusion at ``alpha_ci``: ``"decrease"`` or
        ``"increase"`` when it finds a change, ``"none"`` otherwise.
    verdict : str
        The conclusion, evidence included: ``"decrease"`` or
        ``"increase"`` (the rate changed); ``"no change detected"`` (it
        did not, and the test had at least ``power_required`` power to
        detect the change ``target_ratio``); ``"inconclusive"`` (it did
        not, but the evidence was too little to tell); ``"untestable"``
        (the data cannot answer: see ``notes``).
    notes : list of str
        What the result means, and what to do when the evidence is short.
    n_systems : int
        The systems in the data.
    n_systems_compared : int
        The systems that inform the comparison: with ``"per-system"``,
        those observed both before and after the intervention.
    target_ratio : float
        The change the evidence is judged against (0.5, a halving of the
        rate, unless given).
    power : float
        The chance that a test on this exposure detects ``target_ratio``
        (the exact power of the HPP test, from the rate before the
        intervention).
    power_required : float
        The power the evidence is judged against (0.8 unless given).
    detectable_ratio : float
        The ratio nearest 1 detectable with ``power_required`` on this
        exposure; ``nan`` when none is (too few failures before).
    exposure_needed : float
        The exposure after the intervention, in all, that would detect
        ``target_ratio`` with ``power_required``; ``inf`` when none would
        (more failures before are needed); ``nan`` for the power law,
        whose exposure is not in time units (see ``events_needed``).
    events_needed : float
        The failures after the intervention to expect, at the rate
        ``target_ratio`` times the rate before, by the time the exposure
        is enough.
    zero_failure_exposure : float
        The exposure after the intervention, in all, at which the test
        would find a decrease were there no further failure (the HPP);
        ``nan`` for the power law and when no exposure would.
    shape : float
        The power law's shape :math:`\\beta` (``nan`` for the HPP).
    trend_p_value : float
        A check of the model: for the HPP, the Laplace test of a trend
        within the periods before and after the intervention; for the
        power law, the likelihood-ratio test of :math:`\\beta = 1` (no
        trend). A small value says the rate trends with age.

    Examples
    --------
    >>> from surpyval.recurrent import intervention_test
    >>> x = [3, 8, 15, 19, 26, 30, 37, 44, 52, 77, 95]
    >>> result = intervention_test(x, T=100, intervention=50)
    >>> result.events_before, result.events_after
    (8, 3)
    >>> round(result.ratio, 3), round(result.p_value, 3)
    (0.375, 0.227)
    >>> result.verdict
    'inconclusive'
    """

    model: str
    baseline: str
    method: str
    alternative: str
    alpha_ci: float
    events_before: int
    events_after: int
    exposure_before: float
    exposure_after: float
    expected_after: float
    ratio: float
    ratio_cb: npt.NDArray
    p_value: float
    statistic: float
    change: str
    verdict: str
    notes: list = field(default_factory=list)
    n_systems: int = 1
    n_systems_compared: int = 1
    target_ratio: float = 0.5
    power: float = float("nan")
    power_required: float = 0.8
    detectable_ratio: float = float("nan")
    exposure_needed: float = float("nan")
    events_needed: float = float("nan")
    zero_failure_exposure: float = float("nan")
    shape: float = float("nan")
    trend_p_value: float = float("nan")

    @property
    def reduction(self) -> float:
        """The estimated reduction in the rate, ``1 - ratio`` (negative
        for an increase)."""
        return 1.0 - self.ratio

    @property
    def reduction_cb(self) -> npt.NDArray:
        """The confidence interval of the reduction, ``1 - ratio_cb``,
        as ``[lower, upper]``."""
        return 1.0 - np.asarray(self.ratio_cb, dtype=float)[::-1]

    @property
    def rate_before(self) -> float:
        """The average rate before the intervention,
        ``events_before / exposure_before``."""
        return _divide(self.events_before, self.exposure_before)

    @property
    def rate_after(self) -> float:
        """The average rate after the intervention,
        ``events_after / exposure_after``."""
        return _divide(self.events_after, self.exposure_after)

    @property
    def sufficient(self) -> bool:
        """Whether the exposure gave at least ``power_required`` power to
        detect ``target_ratio``."""
        return bool(self.power >= self.power_required)

    @property
    def additional_exposure(self) -> float:
        """The exposure still needed after the intervention,
        ``exposure_needed - exposure_after`` (0 when it is already
        enough)."""
        return max(0.0, self.exposure_needed - self.exposure_after)

    def to_dict(self) -> dict:
        """The result as a dict of native Python values, strict JSON: an
        infinite or missing number is written as ``None``, and recorded
        under ``"non_finite"`` (as every SurPyval ``to_dict`` does) so
        that :meth:`from_dict` restores it.

        Examples
        --------
        >>> import json
        >>> from surpyval.recurrent import intervention_test
        >>> result = intervention_test([5, 12, 20, 33], T=60, intervention=40)
        >>> d = result.to_dict()
        >>> d["exposure_needed"], d["non_finite"]["inf"]
        (None, ['/exposure_needed', '/events_needed'])
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        """
        from surpyval.serialisation import encode_non_finite, to_native

        out: dict = {"result": type(self).__name__}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = to_native(value)
            elif isinstance(value, list):
                value = list(value)
            elif isinstance(value, np.generic):
                value = to_native(value)
            out[f.name] = value
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> "InterventionTestResult":
        """The result :meth:`to_dict` wrote.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     InterventionTestResult,
        ...     intervention_test,
        ... )
        >>> result = intervention_test([5, 12, 20, 33], T=60, intervention=40)
        >>> again = InterventionTestResult.from_dict(result.to_dict())
        >>> again.exposure_needed, again.verdict == result.verdict
        (inf, True)
        """
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                "Must create an InterventionTestResult from a dict its "
                "to_dict wrote (with 'result': 'InterventionTestResult')"
            )
        values = decode_non_finite(dict(result_dict))
        names = {f.name for f in fields(cls)}
        missing = sorted(names - set(values))
        if missing:
            raise ValueError(
                "the dict is missing {}".format(", ".join(missing))
            )
        kwargs = {name: values[name] for name in names}
        kwargs["ratio_cb"] = np.asarray(kwargs["ratio_cb"], dtype=float)
        kwargs["notes"] = list(kwargs["notes"])
        return cls(**kwargs)

    def __repr__(self) -> str:
        title = "Intervention test ({}, {})".format(
            "HPP" if self.model == "hpp" else "power law", self.baseline
        )
        level = 100 * (1 - self.alpha_ci)
        lines = [
            title,
            "=" * len(title),
            "Systems          : {} ({} compared)".format(
                self.n_systems, self.n_systems_compared
            ),
            "Before           : {} failures in {:.6g}".format(
                self.events_before, self.exposure_before
            ),
            "After            : {} failures in {:.6g} ({:.4g} expected "
            "with no change)".format(
                self.events_after, self.exposure_after, self.expected_after
            ),
            "Rate ratio       : {:.4g}, {:g}% CI [{:.4g}, {:.4g}]".format(
                self.ratio, level, self.ratio_cb[0], self.ratio_cb[1]
            ),
            "Test             : {} ({}), p-value {:.4g}".format(
                self.method, self.alternative, self.p_value
            ),
        ]
        if self.model == "power-law":
            lines.append(
                "Shape            : {:.4g} (no-trend p-value {:.4g})".format(
                    self.shape, self.trend_p_value
                )
            )
        lines.append(
            "Power            : {:.3g} to detect a ratio of {:g} "
            "({:g} required)".format(
                self.power, self.target_ratio, self.power_required
            )
        )
        lines.append("Detectable ratio : {:.4g}".format(self.detectable_ratio))
        lines.append("Verdict          : {}".format(self.verdict))
        for note in self.notes:
            lines.append("  - " + note)
        return "\n".join(lines)


def _divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


# --- data -------------------------------------------------------------------


def _per_system(
    value: Any, labels: list, name: str, allow_infinite: bool = False
) -> dict:
    """``value`` (a scalar, a dict keyed by system, or an array with one
    entry per sorted system) as a ``{system: float}`` dict."""
    refuse_time_values(value, name)
    if isinstance(value, dict):
        missing = [item_label(q) for q in labels if q not in value]
        if missing:
            raise ValueError(
                "`{}` is missing a value for system(s) {}".format(
                    name, ", ".join(missing)
                )
            )
        resolved = {q: float(value[q]) for q in labels}
    elif np.ndim(value) == 0:
        resolved = {q: float(value) for q in labels}
    else:
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.size != len(labels):
            raise ValueError(
                "array `{}` must have one entry per system, in the sorted "
                "order of the system ids ({} systems, {} entries); a dict "
                "keyed by system id is clearer".format(
                    name, len(labels), array.size
                )
            )
        resolved = {q: float(v) for q, v in zip(labels, array)}
    for q, v in resolved.items():
        if math.isnan(v) or (not allow_infinite and math.isinf(v)):
            raise ValueError(
                "`{}` of system {} must be {}; got {}".format(
                    name,
                    item_label(q),
                    "a number (inf for never)" if allow_infinite else "finite",
                    v,
                )
            )
    return resolved


def _windows_from_dict(windows: Any, labels: list) -> dict:
    if not isinstance(windows, dict):
        raise ValueError(
            "`windows` must be a dict {system: [(start, end), ...]}"
        )
    missing = [item_label(q) for q in labels if q not in windows]
    if missing:
        raise ValueError(
            "`windows` is missing system(s) {}".format(", ".join(missing))
        )
    resolved = {}
    for q, spans in windows.items():
        refuse_time_values(spans, "windows")
        array = np.asarray(spans, dtype=float).reshape(-1, 2)
        order = np.argsort(array[:, 0])
        array = array[order]
        if not np.all(np.isfinite(array)) or np.any(
            array[:, 1] <= array[:, 0]
        ):
            raise ValueError(
                "each window of system {} must be a finite (start, end) "
                "with start < end".format(item_label(q))
            )
        if np.any(array[1:, 0] < array[:-1, 1]):
            raise ValueError(
                "the windows of system {} overlap".format(item_label(q))
            )
        resolved[q] = [(float(s), float(e)) for s, e in array]
    return resolved


def _ends_from_c(
    x: npt.NDArray, i: npt.NDArray, c: Any
) -> tuple[npt.NDArray, npt.NDArray, dict]:
    """The events and each system's observation end from the fitters'
    ``c``: 0 an event, 1 the end of the system's observation."""
    c = np.asarray(c)
    if c.shape != x.shape:
        raise ValueError("`x` and `c` must have the same length")
    if not np.isin(c, [0, 1]).all():
        raise ValueError(
            "`c` must be 0 (a failure) or 1 (the end of a system's "
            "observation)"
        )
    ends = {}
    unended = []
    for q in np.unique(i):
        mask = i == q
        end = x[mask & (c == 1)]
        if end.size != 1:
            unended.append(item_label(q))
            continue
        ends[q] = float(end[0])
    if unended:
        raise ValueError(
            "every system needs one c=1 row, the end of its observation; "
            "system(s) {} have {}. The time since a system's last failure "
            "is evidence about the intervention, so it cannot be left "
            "out".format(", ".join(unended), "none or several")
        )
    keep = c == 0
    return x[keep], i[keep], ends


def _systems(
    x: Any,
    i: Any,
    T: Any,
    c: Any,
    tl: Any,
    windows: Any,
    intervention: Any,
) -> list[_System]:
    """Resolve the data into one :class:`_System` per system."""
    for value, name in ((x, "x"), (T, "T"), (tl, "tl")):
        if value is not None and not isinstance(value, dict):
            refuse_time_values(value, name)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1D array of failure times")
    if not np.all(np.isfinite(x)):
        raise ValueError("failure times `x` must be finite")
    if i is not None:
        i = np.asarray(i)
        if i.shape != x.shape:
            raise ValueError("`x` and `i` must have the same length")
    given = [
        name
        for name, v in (("T", T), ("c", c), ("windows", windows))
        if v is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "give each system's observation in one way: its end `T`, a c=1 "
            "row in `c`, or its `windows`{}. The time since a system's last "
            "failure is evidence about the intervention, so it cannot be "
            "left out".format(
                "" if not given else "; got " + " and ".join(given)
            )
        )
    if windows is not None and tl is not None:
        raise ValueError("give `tl` or `windows`, not both")
    # One system by default, even with no failure.
    labels = [1.0] if i is None else list(np.unique(i))
    if i is None:
        i = np.ones(x.shape[0])
    if windows is not None:
        for q in windows:
            if q not in labels:
                labels.append(q)
        spans = _windows_from_dict(windows, labels)
    else:
        if c is not None:
            x, i, ends = _ends_from_c(x, i, c)
        else:
            if isinstance(T, dict):
                labels += [q for q in T if q not in labels]
            ends = _per_system(T, labels, "T")
        if tl is None:
            starts = {q: 0.0 for q in labels}
        elif (
            c is not None
            and not isinstance(tl, dict)
            and np.ndim(tl) == 1
            and np.size(tl) != len(labels)
        ):
            raise ValueError(
                "with `c`, give `tl` as a scalar, a dict keyed by system, "
                "or one value per system"
            )
        else:
            starts = _per_system(tl, labels, "tl")
        spans = {}
        for q in labels:
            if not ends[q] > starts[q]:
                raise ValueError(
                    "the observation of system {} must end after it starts "
                    "(tl={}, end={})".format(
                        item_label(q),
                        number_text(starts[q]),
                        number_text(ends[q]),
                    )
                )
            spans[q] = [(starts[q], ends[q])]
    if intervention is None:
        raise ValueError(
            "give `intervention`, the time of the intervention on each "
            "system (inf for a system never changed)"
        )
    tau = _per_system(intervention, labels, "intervention", True)
    systems = []
    for q in sorted(labels, key=item_label):
        events = np.sort(x[i == q])
        inside = np.zeros(events.shape, dtype=bool)
        for s, e in spans[q]:
            inside |= (events > s) & (events <= e)
        if not inside.all():
            outside = events[~inside][:5]
            raise ValueError(
                "system {} has failure(s) at {} outside its observation "
                "window(s) {}; a window (start, end] holds the failures "
                "after its start up to its end".format(
                    item_label(q),
                    ", ".join(number_text(t) for t in outside),
                    ", ".join(
                        "({}, {}]".format(number_text(s), number_text(e))
                        for s, e in spans[q]
                    ),
                )
            )
        systems.append(_System(q, events, spans[q], tau[q]))
    return systems


def _split(systems: list[_System]) -> _Split:
    """Each system's failures and exposure before and after its
    intervention. A failure at the intervention time is before it: the
    window before is (start, intervention]."""
    k = len(systems)
    before = np.zeros(k, dtype=int)
    after = np.zeros(k, dtype=int)
    exp_b = np.zeros(k)
    exp_a = np.zeros(k)
    pieces_b: list = []
    pieces_a: list = []
    events: list = []
    event_after: list = []
    event_system: list = []
    for q, system in enumerate(systems):
        tau = system.intervention
        for s, e in system.windows:
            if min(e, tau) > s:
                pieces_b.append((q, s, min(e, tau)))
                exp_b[q] += min(e, tau) - s
            if e > max(s, tau):
                pieces_a.append((q, max(s, tau), e))
                exp_a[q] += e - max(s, tau)
        is_after = system.events > tau
        after[q] = int(is_after.sum())
        before[q] = system.events.size - after[q]
        events.extend(system.events)
        event_after.extend(is_after)
        event_system.extend([q] * system.events.size)
    return _Split(
        labels=[s.label for s in systems],
        before=before,
        after=after,
        exposure_before=exp_b,
        exposure_after=exp_a,
        pieces_before=np.asarray(pieces_b, dtype=float).reshape(-1, 3),
        pieces_after=np.asarray(pieces_a, dtype=float).reshape(-1, 3),
        events=np.asarray(events, dtype=float),
        event_after=np.asarray(event_after, dtype=bool),
        event_system=np.asarray(event_system, dtype=int),
    )


def _laplace_within(split: _Split) -> float:
    """The two-sided p-value of the Laplace test of a trend within the
    periods: each piece of a window, before or after the intervention, is
    a period of its own, so a step at the intervention is no trend."""
    total = expected = variance = 0.0
    for pieces in (split.pieces_before, split.pieces_after):
        for q, s, e in pieces:
            mask = (
                (split.event_system == int(q))
                & (split.events > s)
                & (split.events <= e)
            )
            n = int(mask.sum())
            total += float(split.events[mask].sum())
            expected += n * (s + e) / 2.0
            variance += n * (e - s) ** 2 / 12.0
    if variance <= 0 or split.events.size < 2:
        return float("nan")
    u = (total - expected) / math.sqrt(variance)
    return float(2.0 * stats.norm.sf(abs(u)))


# --- the HPP: exact conditional tests ----------------------------------------


class _Strata:
    """The conditional distribution of the failures after the
    intervention, given each stratum's failures in all: a sum of
    independent binomials, one per stratum, with probability
    :math:`\\rho w_q / (1 + \\rho w_q)`, :math:`w_q` the stratum's
    exposure after over its exposure before."""

    def __init__(
        self, totals: npt.NDArray, weights: npt.NDArray, midp: bool
    ) -> None:
        self.totals = totals.astype(int)
        self.weights = weights
        self.midp = midp
        self.size = int(self.totals.sum())

    def _probabilities(self, rho: float) -> npt.NDArray:
        if math.isinf(rho):
            return np.ones_like(self.weights)
        return rho * self.weights / (1.0 + rho * self.weights)

    def tails(self, s: int, rho: float) -> tuple[float, float]:
        """:math:`\\Pr(S \\le s)` and :math:`\\Pr(S \\ge s)`, each less
        half the point mass for the mid-p."""
        p = self._probabilities(rho)
        if self.totals.size == 1:
            n = self.totals[0]
            lower = float(stats.binom.cdf(s, n, p[0]))
            upper = float(stats.binom.sf(s - 1, n, p[0]))
            point = float(stats.binom.pmf(s, n, p[0]))
        else:
            pmf = np.ones(1)
            for n, pq in zip(self.totals, p):
                pmf = np.convolve(
                    pmf, stats.binom.pmf(np.arange(n + 1), n, pq)
                )
            lower = float(pmf[: s + 1].sum())
            upper = float(pmf[s:].sum())
            point = float(pmf[s])
        if self.midp:
            lower -= 0.5 * point
            upper -= 0.5 * point
        return min(1.0, max(0.0, lower)), min(1.0, max(0.0, upper))

    def estimate(self, s: int) -> float:
        """The conditional maximum-likelihood estimate of the ratio: the
        :math:`\\rho` at which the expected count is ``s``."""
        if s == 0:
            return 0.0
        if s == self.size:
            return float("inf")

        def excess(u: float) -> float:
            p = self._probabilities(math.exp(u))
            return float(np.sum(self.totals * p)) - s

        lo, hi = _bracket(excess, 0.0)
        return math.exp(optimize.brentq(excess, lo, hi, xtol=1e-12))


def _bracket(f: Callable[[float], float], start: float) -> tuple[float, float]:
    """A bracket ``(lo, hi)`` around the root of the increasing ``f``."""
    lo = hi = start
    step = 1.0
    while f(lo) > 0:
        lo -= step
        step *= 2.0
        if lo < -_LOG_RATIO_LIMIT:
            raise _NoRoot()
    step = 1.0
    while f(hi) < 0:
        hi += step
        step *= 2.0
        if hi > _LOG_RATIO_LIMIT:
            raise _NoRoot()
    return lo, hi


class _NoRoot(Exception):
    pass


def _exact_bound(
    strata: _Strata, s: int, tail: float, side: str, start: float
) -> float:
    """The confidence bound on the ratio from inverting the conditional
    test: the upper bound solves :math:`\\Pr(S \\le s; \\rho) =` ``tail``,
    the lower :math:`\\Pr(S \\ge s; \\rho) =` ``tail``."""
    if side == "upper":
        if s == strata.size:
            return float("inf")

        def f(u: float) -> float:
            return tail - strata.tails(s, math.exp(u))[0]

    else:
        if s == 0:
            return 0.0

        def f(u: float) -> float:
            return strata.tails(s, math.exp(u))[1] - tail

    try:
        lo, hi = _bracket(f, start)
    except _NoRoot:
        return 0.0 if side == "lower" else float("inf")
    return math.exp(optimize.brentq(f, lo, hi, xtol=1e-12))


def _hpp_test(
    split: _Split,
    baseline: str,
    midp: bool,
    alternative: str,
    alpha_ci: float,
) -> dict:
    """The exact (or mid-p) conditional test of the ratio, and its
    interval."""
    if baseline == "common":
        compared = (split.exposure_before > 0) | (split.exposure_after > 0)
        totals = np.array([split.before.sum() + split.after.sum()])
        e_b, e_a = split.exposure_before.sum(), split.exposure_after.sum()
        weights = np.array([e_a / e_b]) if e_b > 0 and e_a > 0 else None
        s = int(split.after.sum())
    else:
        compared = (split.exposure_before > 0) & (split.exposure_after > 0)
        totals = (split.before + split.after)[compared]
        weights = (
            split.exposure_after[compared] / split.exposure_before[compared]
        )
        s = int(split.after[compared].sum())
    out: dict = {"compared": compared}
    if weights is None or weights.size == 0 or totals.sum() == 0:
        return out
    keep = totals > 0
    strata = _Strata(totals[keep], weights[keep], midp)
    probabilities = strata._probabilities(1.0)
    lower, upper = strata.tails(s, 1.0)
    if alternative == "decrease":
        p_value = lower
    elif alternative == "increase":
        p_value = upper
    else:
        p_value = min(1.0, 2.0 * min(lower, upper))
    ratio = strata.estimate(s)
    start = math.log(ratio) if 0 < ratio < math.inf else 0.0
    tail = alpha_ci / 2.0 if alternative == "two-sided" else alpha_ci
    lo = (
        0.0
        if alternative == "decrease"
        else _exact_bound(strata, s, tail, "lower", start)
    )
    hi = (
        float("inf")
        if alternative == "increase"
        else _exact_bound(strata, s, tail, "upper", start)
    )
    out.update(
        p_value=float(p_value),
        ratio=float(ratio),
        ratio_cb=np.array([lo, hi]),
        statistic=float(s),
        expected_after=float(np.sum(strata.totals * probabilities)),
    )
    return out


# --- the power law: likelihood ratio -----------------------------------------


class _PowerLaw:
    """The power-law likelihood with a step of :math:`\\rho` at the
    intervention, with each group's scale (one group, or one per system)
    profiled out in closed form:

    .. math::
        \\ell(\\beta, \\rho) = n \\log\\beta + (\\beta - 1) \\sum \\log t
        + n_a \\log\\rho + \\sum_g n_g \\log\\frac{n_g}{A_g(\\beta) +
        \\rho B_g(\\beta)} - n,

    :math:`A_g` and :math:`B_g` the sums of :math:`e^\\beta - s^\\beta`
    over the group's pieces of window before and after. Times are scaled
    by the latest end, which changes the likelihood by a constant."""

    def __init__(self, split: _Split, groups: npt.NDArray) -> None:
        scale = float(
            max(
                split.pieces_before[:, 2].max(initial=0.0),
                split.pieces_after[:, 2].max(initial=0.0),
            )
        )
        self.groups = groups
        n_groups = int(groups.max()) + 1 if groups.size else 0
        self.n_groups = n_groups
        self.pieces = [
            (
                groups[split.pieces_before[:, 0].astype(int)],
                split.pieces_before[:, 1] / scale,
                split.pieces_before[:, 2] / scale,
            ),
            (
                groups[split.pieces_after[:, 0].astype(int)],
                split.pieces_after[:, 1] / scale,
                split.pieces_after[:, 2] / scale,
            ),
        ]
        event_group = groups[split.event_system]
        self.n = int(split.events.size)
        self.n_after = int(split.event_after.sum())
        self.n_group = np.bincount(event_group, minlength=n_groups)
        self.log_t = float(np.sum(np.log(split.events / scale)))

    def exposures(self, beta: float) -> tuple[npt.NDArray, npt.NDArray]:
        out = []
        for g, s, e in self.pieces:
            amount = e**beta - s**beta
            out.append(np.bincount(g, amount, minlength=self.n_groups))
        return out[0], out[1]

    def loglik(self, log_beta: float, rho: float) -> float:
        beta = math.exp(log_beta)
        a, b = self.exposures(beta)
        m = self.n_group > 0
        denominator = (a + rho * b)[m]
        if (self.n_after > 0 and not rho > 0) or not np.all(
            (denominator > 0) & np.isfinite(denominator)
        ):
            # A ratio of 0 with failures after, or exposures that under-
            # or overflow at an extreme shape: no likelihood there.
            return -math.inf
        ll = self.n * log_beta + (beta - 1.0) * self.log_t - self.n
        if self.n_after > 0:
            ll += self.n_after * math.log(rho)
        ll += float(
            np.sum(self.n_group[m] * np.log(self.n_group[m] / denominator))
        )
        return ll

    def best_ratio(self, log_beta: float) -> float:
        """The ratio maximising the likelihood at the shape."""
        if self.n_after == 0:
            return 0.0
        beta = math.exp(log_beta)
        a, b = self.exposures(beta)
        if self.n_groups == 1:
            return (self.n_after / b[0]) / ((self.n - self.n_after) / a[0])

        def excess(u: float) -> float:
            rho = math.exp(u)
            m = self.n_group > 0
            share = rho * b[m] / (a[m] + rho * b[m])
            return float(np.sum(self.n_group[m] * share)) - self.n_after

        try:
            lo, hi = _bracket(excess, 0.0)
        except _NoRoot:
            return 0.0 if excess(0.0) > 0 else float("inf")
        return math.exp(optimize.brentq(excess, lo, hi, xtol=1e-12))

    def maximise(self, objective: Callable[[float], float]) -> float:
        """The log-shape maximising ``objective``: a grid, then Brent's
        method around its best point."""
        grid = np.linspace(*_LOG_SHAPE_RANGE, _LOG_SHAPE_GRID)
        values = np.array([objective(u) for u in grid])
        k = int(np.nanargmax(values))
        if k in (0, grid.size - 1):
            raise _NoMaximum()
        res = optimize.minimize_scalar(
            lambda u: -objective(u),
            bounds=(grid[k - 1], grid[k + 1]),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(res.x) if -res.fun >= values[k] else float(grid[k])

    def profile(self, rho: float, near: float | None = None) -> float:
        """The likelihood at the ratio, maximised over the shape.

        At a fixed ratio the likelihood is concave in the shape (the log
        of each group's exposure, :math:`\\log \\int t^{\\beta - 1}`, is
        convex in it), so a search of the shapes ``near`` a known one
        finds the maximum unless it stops at the edge of its range."""
        if near is not None:
            res = optimize.minimize_scalar(
                lambda v: -self.loglik(v, rho),
                bounds=(near - 1.0, near + 1.0),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if abs(abs(float(res.x) - near) - 1.0) > 1e-4:
                return -float(res.fun)
        try:
            u = self.maximise(lambda v: self.loglik(v, rho))
        except _NoMaximum:
            # The likelihood rises to the edge of the shapes searched: its
            # supremum, approached there, is what the profile is.
            grid = np.linspace(*_LOG_SHAPE_RANGE, _LOG_SHAPE_GRID)
            return max(self.loglik(v, rho) for v in grid[[0, -1]])
        return self.loglik(u, rho)


class _NoMaximum(Exception):
    pass


def _profile_bound(
    model: _PowerLaw,
    top: float,
    critical: float,
    log_ratio: float,
    side: str,
    log_shape: float,
) -> float:
    """Where the profile likelihood falls ``critical / 2`` below its
    maximum ``top``, on ``side`` of the estimate ``log_ratio`` (found at
    the shape ``log_shape``)."""

    def f(u: float) -> float:
        drop = top - model.profile(math.exp(u), log_shape)
        # A likelihood lost to overflow at an extreme ratio is far below.
        return drop - critical / 2.0 if math.isfinite(drop) else critical

    sign = 1.0 if side == "upper" else -1.0
    start = log_ratio if math.isfinite(log_ratio) else -_LOG_RATIO_LIMIT / 2
    inside = start
    step = 0.5
    outside = start + sign * step
    try:
        while f(outside) < 0:
            inside = outside
            step *= 2.0
            outside = start + sign * step
            if abs(outside) > _LOG_RATIO_LIMIT:
                return float("inf") if side == "upper" else 0.0
        lo, hi = sorted((inside, outside))
        return math.exp(optimize.brentq(f, lo, hi, xtol=1e-10))
    except _NoMaximum:
        return float("inf") if side == "upper" else 0.0


def _power_law_test(
    split: _Split, baseline: str, alternative: str, alpha_ci: float
) -> dict:
    if baseline == "common":
        compared = (split.exposure_before > 0) | (split.exposure_after > 0)
        groups = np.zeros(len(split.labels), dtype=int)
    else:
        # Each system its own scale: those seen on one side only still
        # inform the shape.
        compared = (split.exposure_before > 0) & (split.exposure_after > 0)
        groups = np.arange(len(split.labels))
    out: dict = {"compared": compared}
    n_before = int(split.before.sum())
    n_after = int(split.after.sum())
    if (
        split.exposure_before.sum() <= 0
        or split.exposure_after.sum() <= 0
        or not compared.any()
        or n_before + n_after < 2
    ):
        return out
    swapped = n_before == 0
    if swapped:
        # No failure before: the ratio is infinite. Fit its reciprocal,
        # with the periods swapped, which is the same model.
        split = _Split(
            labels=split.labels,
            before=split.after,
            after=split.before,
            exposure_before=split.exposure_after,
            exposure_after=split.exposure_before,
            pieces_before=split.pieces_after,
            pieces_after=split.pieces_before,
            events=split.events,
            event_after=~split.event_after,
            event_system=split.event_system,
        )
        alternative = {
            "decrease": "increase",
            "increase": "decrease",
        }.get(alternative, alternative)
    model = _PowerLaw(split, groups)
    try:
        log_beta = model.maximise(
            lambda u: model.loglik(u, model.best_ratio(u))
        )
        rho = model.best_ratio(log_beta)
        top = model.loglik(log_beta, rho)
        null = model.profile(1.0, log_beta)
    except _NoMaximum:
        out["no_maximum"] = True
        return out
    lr = max(0.0, 2.0 * (top - null))
    log_rho = math.log(rho) if rho > 0 else -math.inf
    signed = math.copysign(math.sqrt(lr), log_rho) if lr > 0 else 0.0
    if alternative == "decrease":
        p_value = float(stats.norm.cdf(signed))
    elif alternative == "increase":
        p_value = float(stats.norm.sf(signed))
    else:
        p_value = float(stats.chi2.sf(lr, 1))
    if alternative == "two-sided":
        critical = float(stats.chi2.isf(alpha_ci, 1))
    else:
        critical = float(stats.norm.isf(alpha_ci)) ** 2
    lo = (
        0.0
        if alternative == "decrease" or rho == 0
        else _profile_bound(model, top, critical, log_rho, "lower", log_beta)
    )
    hi = (
        float("inf")
        if alternative == "increase"
        else _profile_bound(model, top, critical, log_rho, "upper", log_beta)
    )
    # The check of the shape: beta = 1 (no trend), the ratio free.
    flat = model.loglik(0.0, model.best_ratio(0.0))
    trend_p = float(stats.chi2.sf(max(0.0, 2.0 * (top - flat)), 1))
    beta_hat = math.exp(log_beta)
    m = model.n_group > 0
    # The failures after expected under no change, from the null fit.
    null_beta = math.exp(model.maximise(lambda u: model.loglik(u, 1.0)))
    a0, b0 = model.exposures(null_beta)
    expected_after = float(np.sum(model.n_group[m] * b0[m] / (a0[m] + b0[m])))
    # The exposures of the systems compared in the cumulative-intensity
    # scale, where the process is an HPP: the evidence is judged there.
    a_hat, b_hat = model.exposures(beta_hat)
    used = slice(None) if baseline == "common" else compared
    out.update(
        p_value=p_value,
        ratio=rho,
        ratio_cb=np.array([lo, hi]),
        statistic=lr,
        shape=beta_hat,
        trend_p_value=trend_p,
        expected_after=expected_after,
        scaled_before=float(np.sum(a_hat[used])),
        scaled_after=float(np.sum(b_hat[used])),
    )
    if swapped:
        lo, hi = out["ratio_cb"]
        out["ratio"] = 1.0 / rho if rho > 0 else float("inf")
        out["ratio_cb"] = np.array([_reciprocal(hi), _reciprocal(lo)])
        out["expected_after"] = float(model.n - expected_after)
        out["scaled_before"], out["scaled_after"] = (
            out["scaled_after"],
            out["scaled_before"],
        )
    return out


def _reciprocal(value: float) -> float:
    if value == 0:
        return float("inf")
    if math.isinf(value):
        return 0.0
    return 1.0 / value


# --- the evidence ------------------------------------------------------------


def _evidence(
    n_before: int,
    before: float,
    after: float,
    target: float,
    power_required: float,
    tail: float,
    midp: bool,
) -> dict:
    """The power of the exposure to detect ``target``, the change it can
    detect, and the exposure it would need, in the scale in which the
    rate is constant (time for the HPP; the cumulative intensity for the
    power law), with the rate before the intervention taken as the
    rate of the failures before it."""
    if n_before == 0 or before <= 0 or after <= 0:
        return {}
    rate = n_before / before
    expected_after = rate * after
    direction = "decrease" if target < 1 else "increase"
    multiple = exposure_multiple(n_before, target, power_required, tail, midp)
    needed = multiple * before
    out = {
        "power": expected_power(n_before, expected_after, target, tail, midp),
        "detectable_ratio": detectable_ratio(
            n_before, expected_after, power_required, tail, midp, direction
        ),
        "exposure_needed": needed,
        "events_needed": target * rate * needed,
    }
    if direction == "decrease":
        # With no failure after it, the lower tail is (1 - pi0)^n_before,
        # half that for the mid-p: below `tail` once the exposure after
        # reaches this.
        level = 2.0 * tail if midp else tail
        out["zero_failure_exposure"] = before * (
            level ** (-1.0 / n_before) - 1.0
        )
    return out


def _share(probability: float) -> str:
    if probability < 0.01:
        return "less than 1%"
    return "{:.0%}".format(probability)


def _amount(value: float) -> str:
    """An exposure for a note: whole numbers from 100 up, else four
    significant figures."""
    if abs(value) >= 100:
        return "{:,.0f}".format(value)
    return "{:.4g}".format(value)


def _percent(ratio: float) -> str:
    return "{:.0f}%".format(100 * abs(1 - ratio))


def _notes(result: InterventionTestResult, dropped: int, hpp: bool) -> list:
    """What the result means, and what to do next."""
    notes = []
    level = "{:g}%".format(100 * (1 - result.alpha_ci))
    lo, hi = result.ratio_cb
    if result.verdict == "decrease":
        notes.append(
            "The failure rate fell to {:.3g} times its earlier rate, a "
            "reduction of {} ({} confidence interval of the ratio {:.3g} "
            "to {:.3g}).".format(
                result.ratio, _percent(result.ratio), level, lo, hi
            )
        )
    elif result.verdict == "increase":
        notes.append(
            "The failure rate rose to {:.3g} times its earlier rate ({} "
            "confidence interval of the ratio {:.3g} to {:.3g}).".format(
                result.ratio, level, lo, hi
            )
        )
    elif result.verdict == "no change detected":
        notes.append(
            "No change was detected, though the test had {} power to "
            "detect a ratio of {:g}: the rate most likely did not change "
            "that much. The interval of the ratio, {:.3g} to {:.3g}, is "
            "what the data support.".format(
                _share(result.power), result.target_ratio, lo, hi
            )
        )
    elif result.verdict == "inconclusive":
        notes.append(
            "Not enough evidence yet: the test had only {} power to "
            "detect a ratio of {:g} with this exposure.".format(
                _share(result.power), result.target_ratio
            )
        )
    if result.verdict == "no change detected":
        notes.append(
            "The smallest change this exposure detects with {:.0%} power "
            "is a ratio of {:.3g}.".format(
                result.power_required, result.detectable_ratio
            )
        )
    if result.verdict == "inconclusive":
        if math.isfinite(result.exposure_needed) and hpp:
            notes.append(
                "{} exposure after the intervention in all ({} more), "
                "with about {:.3g} failures expected at that rate, would "
                "give {:.0%} power.".format(
                    _amount(result.exposure_needed),
                    _amount(result.additional_exposure),
                    result.events_needed,
                    result.power_required,
                )
            )
        elif math.isfinite(result.events_needed):
            notes.append(
                "About {:.3g} failures expected after the intervention, "
                "at that rate, would give {:.0%} power.".format(
                    result.events_needed, result.power_required
                )
            )
        elif result.power == result.power:
            notes.append(
                "No exposure after the intervention can give {:.0%} power: "
                "the {} failures before it are too few to pin down the "
                "earlier rate. Only more history before the intervention, "
                "or a larger change, would.".format(
                    result.power_required, result.events_before
                )
            )
    if (
        result.verdict == "inconclusive"
        and math.isfinite(result.zero_failure_exposure)
        and result.zero_failure_exposure > result.exposure_after
    ):
        notes.append(
            "If no further failure occurs, a decrease will be shown once "
            "the exposure after the intervention reaches {}.".format(
                _amount(result.zero_failure_exposure)
            )
        )
    if result.events_after == 0 and result.verdict != "untestable":
        notes.append(
            "No failure since the intervention: the ratio's estimate is 0, "
            "and its upper bound, {:.3g}, is what the exposure so far can "
            "vouch for.".format(hi)
        )
    if hpp and result.trend_p_value < result.alpha_ci:
        notes.append(
            "The rate trends within the periods (Laplace p-value {:.3g}): "
            "a comparison of constant rates can take a trend for an effect "
            "of the intervention. Use model='power-law'.".format(
                result.trend_p_value
            )
        )
    if dropped:
        notes.append(
            "{} system{} observed on one side of the intervention only "
            "{} not compared: each system keeps its own rate, so only "
            "systems seen before and after it inform the ratio. "
            "baseline='common' uses them, if the systems share one "
            "rate.".format(
                dropped,
                "" if dropped == 1 else "s",
                "is" if dropped == 1 else "are",
            )
        )
    return notes


def _untestable_reason(
    split: _Split, compared: npt.NDArray, baseline: str, model: str
) -> str:
    if split.exposure_before.sum() <= 0:
        return (
            "No system was observed before the intervention, so there is "
            "no earlier rate to compare with. Give the observation before "
            "it (`tl` earlier, or windows that include it)."
        )
    if split.exposure_after.sum() <= 0:
        return (
            "No system was observed after the intervention yet: there is "
            "nothing to compare. Run the test once there is exposure "
            "after it."
        )
    if not compared.any():
        return (
            "No system was observed both before and after the "
            "intervention, which a per-system comparison needs. Use "
            "baseline='common' if the systems share one failure rate."
        )
    if model == "power-law":
        return (
            "The power law needs at least two failures, and a likelihood "
            "with a maximum; with fewer failures use model='hpp'."
        )
    return (
        "There is no failure in the periods compared, so there is no rate "
        "to compare: exposure without failures cannot show a change in a "
        "rate that was never seen."
    )


def intervention_test(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    T: npt.ArrayLike | dict | None = None,
    *,
    intervention: npt.ArrayLike | dict,
    c: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | dict | None = None,
    windows: dict | None = None,
    model: str = "hpp",
    baseline: str = "per-system",
    alternative: str = "two-sided",
    method: str | None = None,
    alpha_ci: float = 0.05,
    target_ratio: float | None = None,
    power: float = 0.8,
) -> InterventionTestResult:
    r"""
    Test whether an intervention changed the rate of failures, and judge
    whether the evidence is enough.

    The failure records of one or more repairable systems are split at the
    time of the intervention on each (a design change, a new procedure, a
    corrective action), and the rate after it is compared with the rate
    before it. The answer is the ratio of the rates, after over before,
    with a confidence interval and the p-value of no change; the power the
    exposure had to detect the change hoped for (``target_ratio``); the
    smallest change it could detect; and the exposure that would be
    needed. A failure at the intervention time counts before it.

    ``model="hpp"`` (the default) takes the rate as constant on each side,
    and runs the exact conditional test: given the failures in all, the
    number after the intervention is binomial with probability
    :math:`\pi = \rho E_a / (E_b + \rho E_a)` (exposures :math:`E_b` and
    :math:`E_a`), whatever the rates, so the p-value and the interval
    (the Clopper-Pearson interval for :math:`\pi`, transformed to
    :math:`\rho`) are exact; ``method="mid-p"`` gives their mid-p versions,
    closer to the nominal level on average but no longer guaranteed. With
    ``baseline="per-system"`` (the default) each system has its own rate
    and only the ratio is shared: the statistic is the failures after the
    intervention summed over the systems, whose exact distribution is the
    sum of the systems' binomials (Breslow and Day 1987, chapter 3), so
    systems with different rates and different shares of exposure after
    the intervention do not confound the comparison. With
    ``baseline="common"`` the exposures and failures of every system are
    pooled (more powerful, if the systems do share one rate).

    ``model="power-law"`` lets the rate trend with age as the Crow-AMSAA
    power law, :math:`u(t) = \lambda \beta t^{\beta - 1}`, multiplied by
    :math:`\rho` after the intervention, the shape kept. The test is the
    likelihood ratio, with the scale (each system's, for
    ``"per-system"``) and the shape profiled out, and the interval is the
    profile-likelihood interval. Use it when the rate trends with age
    (wear-out, or growth from earlier fixes), which an HPP comparison
    mistakes for an effect of the intervention; the HPP result's
    ``trend_p_value`` flags such data. Here ``x`` is each system's age,
    and observation must start at an age of 0 or more.

    The evidence is judged by the exact power of the HPP test, with the
    rate before the intervention taken as the rate of its failures (for
    the power law, in the scale of the cumulative intensity, where the
    process is an HPP; this ignores the uncertainty in the shape, so it is
    optimistic). For ``"per-system"`` the power is that of the pooled
    test, an approximation. The power cannot be raised without limit by
    exposure after the intervention: the failures before it fix how well
    the earlier rate is known, and when they are too few no exposure
    after it is enough (``exposure_needed`` is then ``inf``).

    The comparison assumes the intervention is the only change between
    the periods. An intervention made *because* of a run of failures
    tends to be followed by fewer, by chance alone (regression to the
    mean); count the period that prompted it as before, and judge the
    after period on its own exposure, not on the first quiet weeks.

    Parameters
    ----------
    x : array_like
        Failure times, measured from each system's start of life for the
        power law. For several systems, the times of every system are
        concatenated and identified by ``i``. With ``c``, the rows in the
        fitters' form (failures, and each system's end of observation).
    i : array_like, optional
        The system each row belongs to. Defaults to one system.
    T : scalar, array_like or dict, optional
        The end of each system's observation: a scalar for every system,
        an array with one value per system (in the sorted order of the
        system ids), or a dict keyed by system (which may name systems
        with no failure). Give the observation as one of ``T``, ``c`` or
        ``windows``: the time since a system's last failure is evidence
        about the intervention, so it is required.
    intervention : scalar, array_like or dict
        Keyword only, required: the time of the intervention on each
        system, in the forms ``T`` takes. ``inf`` marks a system never
        changed, and a time at or before a system's start one changed
        before it was observed.
    c : array_like, optional
        Keyword only: censoring flags in the fitters' form, in place of
        ``T``: 0 a failure, 1 the end of the system's observation (one
        per system, so a system with no failure is a single ``c = 1``
        row).
    tl : scalar, array_like or dict, optional
        Keyword only: the start of each system's observation (delayed
        entry, left truncation), in the forms ``T`` takes. Default 0.
    windows : dict, optional
        Keyword only: each system's observation windows, ``{system:
        [(start, end), ...]}``, for systems observed over separate
        periods; in place of ``T`` (or ``c``) and ``tl``. Every row of
        ``x`` is then a failure.
    model : str, optional
        Keyword only: ``"hpp"`` (the default) or ``"power-law"``.
    baseline : str, optional
        Keyword only: ``"per-system"`` (the default; each system its own
        rate) or ``"common"`` (one rate for every system).
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default), ``"decrease"`` (the
        one-sided test that the rate fell: the question "did the action
        work?") or ``"increase"``.
    method : str, optional
        Keyword only: for the HPP, ``"exact"`` (the default) or
        ``"mid-p"``; for the power law, ``"likelihood-ratio"`` (its only
        test).
    alpha_ci : float, optional
        Keyword only: the significance level, and one minus the
        confidence level of the interval (default 0.05).
    target_ratio : float, optional
        Keyword only: the change the evidence is judged against, as a
        ratio of the rates: 0.5 (a halving) by default, 2 for
        ``alternative="increase"``.
    power : float, optional
        Keyword only: the power the evidence should have to detect
        ``target_ratio``, by default 0.8.

    Returns
    -------
    InterventionTestResult
        The ratio and its interval, the p-value, the verdict, the
        evidence, and notes saying what to do when it is short.

    Raises
    ------
    ValueError
        If the data or an option is invalid. Data that are valid but
        cannot answer the question (no exposure after the intervention,
        no failure at all) give a result with the verdict
        ``"untestable"`` and a note saying why, rather than an error.

    Examples
    --------
    A pump failed eight times in its first 50 weeks, and three times in
    the 50 weeks after its seals were changed:

    >>> from surpyval.recurrent import intervention_test
    >>> x = [3, 8, 15, 19, 26, 30, 37, 44, 52, 77, 95]
    >>> result = intervention_test(x, T=100, intervention=50)
    >>> round(result.ratio, 3), round(result.p_value, 3)
    (0.375, 0.227)
    >>> result.ratio_cb.round(3)
    array([0.064, 1.562])
    >>> result.verdict, round(result.power, 3)
    ('inconclusive', 0.134)

    The one-sided test asks only whether the rate fell:

    >>> one_sided = intervention_test(
    ...     x, T=100, intervention=50, alternative="decrease"
    ... )
    >>> round(one_sided.p_value, 3), one_sided.ratio_cb.round(3)
    (0.113, array([0.   , 1.296]))

    A fleet of three pumps, changed at different times and observed for
    different lengths, each with its own failure rate:

    >>> x = [5, 11, 18, 26, 41, 4, 9, 13, 17, 22, 30, 52, 7, 33]
    >>> i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
    >>> result = intervention_test(
    ...     x, i, T={1: 60, 2: 70, 3: 50}, intervention={1: 30, 2: 25, 3: 20}
    ... )
    >>> result.events_before, result.events_after
    (10, 4)
    >>> round(result.ratio, 3), round(result.p_value, 3)
    (0.277, 0.044)
    >>> result.verdict
    'decrease'
    """
    check_option("model", model, MODELS)
    check_option("baseline", baseline, BASELINES)
    check_option("alternative", alternative, ALTERNATIVES)
    check_alpha_ci(alpha_ci, "It is the test's significance level.")
    hpp = model == "hpp"
    if method is None:
        method = "exact" if hpp else "likelihood-ratio"
    check_option(
        "method",
        method,
        HPP_METHODS if hpp else POWER_LAW_METHODS,
        None if hpp else "The power law is tested by the likelihood ratio.",
    )
    if target_ratio is None:
        target_ratio = 2.0 if alternative == "increase" else 0.5
    target_ratio = _check_target(target_ratio, alternative)
    if not (
        isinstance(power, (int, float, np.integer, np.floating))
        and 0 < power < 1
    ):
        raise ValueError(
            "'power' must be strictly between 0 and 1; got {!r}".format(power)
        )
    systems = _systems(x, i, T, c, tl, windows, intervention)
    if not hpp:
        starts = [s for system in systems for s, _ in system.windows]
        if min(starts) < 0:
            raise ValueError(
                "the power law is a function of age: observation must start "
                "at an age of 0 or more, but a window starts at {}".format(
                    number_text(min(starts))
                )
            )
    split = _split(systems)
    midp = method == "mid-p"
    if hpp:
        fit = _hpp_test(split, baseline, midp, alternative, alpha_ci)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = _power_law_test(split, baseline, alternative, alpha_ci)
    compared = fit["compared"]
    rows = compared if baseline == "per-system" else slice(None)
    n_before = int(split.before[rows].sum())
    n_after = int(split.after[rows].sum())
    exposure_before = float(split.exposure_before[rows].sum())
    exposure_after = float(split.exposure_after[rows].sum())
    tail = alpha_ci / 2.0 if alternative == "two-sided" else alpha_ci
    nan = float("nan")
    if "p_value" in fit:
        if hpp:
            scale_before, scale_after = exposure_before, exposure_after
        else:
            scale_before = fit["scaled_before"]
            scale_after = fit["scaled_after"]
        evidence = _evidence(
            n_before,
            scale_before,
            scale_after,
            target_ratio,
            power,
            tail,
            midp,
        )
        if not hpp:
            evidence.pop("exposure_needed", None)
            evidence.pop("zero_failure_exposure", None)
        p_value = fit["p_value"]
        ratio = fit["ratio"]
        change = "none"
        if p_value < alpha_ci:
            change = "decrease" if ratio < 1 else "increase"
        if change != "none":
            verdict = change
        elif n_before == 0 and alternative != "increase":
            verdict = "untestable"
        elif evidence.get("power", 0.0) >= power:
            verdict = "no change detected"
        else:
            verdict = "inconclusive"
        result = InterventionTestResult(
            model=model,
            baseline=baseline,
            method=method,
            alternative=alternative,
            alpha_ci=alpha_ci,
            events_before=n_before,
            events_after=n_after,
            exposure_before=exposure_before,
            exposure_after=exposure_after,
            expected_after=fit["expected_after"],
            ratio=ratio,
            ratio_cb=fit["ratio_cb"],
            p_value=p_value,
            statistic=fit["statistic"],
            change=change,
            verdict=verdict,
            n_systems=len(systems),
            n_systems_compared=int(np.sum(compared)),
            target_ratio=target_ratio,
            power_required=power,
            shape=fit.get("shape", nan),
            trend_p_value=(
                _laplace_within(split) if hpp else fit["trend_p_value"]
            ),
            **evidence,
        )
        dropped = (
            int(np.sum(~compared)) if baseline == "per-system" and hpp else 0
        )
        result.notes = _notes(result, dropped, hpp)
        if verdict == "untestable":
            result.notes.insert(
                0,
                "No failure before the intervention: there is no failure "
                "rate to show a reduction of. To show that the rate after "
                "it is low enough, use a demonstration test on the "
                "exposure after it (surpyval.test_planning."
                "demonstrated_mtbf).",
            )
        return result
    if fit.get("no_maximum"):
        warnings.warn(
            "The power-law likelihood has no finite maximum in the shape on "
            "these data (all the failures close together, or too few): no "
            "comparison was made. Use model='hpp'.",
            stacklevel=2,
        )
    return InterventionTestResult(
        model=model,
        baseline=baseline,
        method=method,
        alternative=alternative,
        alpha_ci=alpha_ci,
        events_before=n_before,
        events_after=n_after,
        exposure_before=exposure_before,
        exposure_after=exposure_after,
        expected_after=nan,
        ratio=nan,
        ratio_cb=np.array([nan, nan]),
        p_value=nan,
        statistic=nan,
        change="none",
        verdict="untestable",
        notes=[_untestable_reason(split, compared, baseline, model)],
        n_systems=len(systems),
        n_systems_compared=int(np.sum(compared)),
        target_ratio=target_ratio,
        power_required=power,
    )


def _check_target(target_ratio: Any, alternative: str) -> float:
    try:
        value = float(target_ratio)
    except (TypeError, ValueError):
        value = float("nan")
    if not (math.isfinite(value) and value > 0 and value != 1):
        raise ValueError(
            "'target_ratio' must be a positive ratio other than 1 (0.5 for "
            "a halving of the rate); got {!r}".format(target_ratio)
        )
    if alternative == "decrease" and value > 1:
        raise ValueError(
            "'target_ratio' must be below 1 for alternative='decrease'; got "
            "{!r}".format(target_ratio)
        )
    if alternative == "increase" and value < 1:
        raise ValueError(
            "'target_ratio' must be above 1 for alternative='increase'; got "
            "{!r}".format(target_ratio)
        )
    return value
