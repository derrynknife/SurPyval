r"""One function per comparison design: give every quantity but one, and
the one left ``None`` is solved for.

Each design is one equation between its power, its size (the events, or
the units in each group) and the effect it detects. :func:`logrank_design`,
:func:`cox_design`, :func:`noninferiority_design`,
:func:`equivalence_design`, :func:`rmst_design`, :func:`cure_logrank_design`
and :func:`rate_ratio_design` take all of them as
arguments: those given are held, and the one left ``None`` is found. The
answer is a :class:`PowerCalculation`, which says what was solved for
(``solved_for``) and holds it (``value``) beside the power, the events and
the units.

The formulas, references and checks against other software are those of
:mod:`surpyval.test_planning.comparison`,
:mod:`surpyval.test_planning.sample_size` and
:mod:`surpyval.test_planning.rate_ratio`.

Examples
--------
>>> from surpyval.test_planning import logrank_design
>>> logrank_design(0.5, power=0.8).events
66
>>> plan = logrank_design(0.7, power=0.8, event_probability=(0.25, 0.35))
>>> plan.solved_for, plan.n
('n', (412, 412))
>>> plan = logrank_design(0.7, n=(200, 200), event_probability=0.3)
>>> plan.solved_for, round(plan.power, 3)
('power', 0.497)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, stats

from surpyval.test_planning import comparison as _comparison_module
from surpyval.test_planning import rate_ratio as _rates_module
from surpyval.test_planning import sample_size as _sizes_module
from surpyval.test_planning._checks import (
    check_count,
    check_positive,
    check_probability,
    elementwise,
)
from surpyval.test_planning._serialise import result_dict
from surpyval.utils.validation import check_option

# The functions of one unknown that the designs call, checked where they
# are called (an argument left None here is never one of theirs).
_comparison: Any = _comparison_module
_rates: Any = _rates_module
_sizes: Any = _sizes_module


@dataclass(frozen=True, repr=False)
class PowerCalculation:
    """The answer of a design function: which quantity was solved for, and
    every quantity of the design.

    Attributes
    ----------
    design : str
        The design's function: ``"logrank"``, ``"cox"``,
        ``"noninferiority"``, ``"equivalence"``, ``"rmst"``,
        ``"cure_logrank"``, ``"rate_ratio"`` or ``"mean_ratio"``.
    solved_for : str
        The argument that was left ``None`` and found: ``"power"``,
        ``"events"``, ``"n"``, ``"hazard_ratio"``, ``"ratio"``,
        ``"exposure_after"`` or ``"mean_ratio"``.
    value : int, float, tuple or ndarray
        The quantity solved for (an array where the design's arguments
        were arrays).
    power : float or ndarray
        The power: the one given, or the one found.
    n : int, (int, int), ndarray or None
        The units (in each group, for a two-group design), given or found;
        None where neither was asked for (the events alone, or a rate
        ratio's exposure).
    events : int, ndarray or None
        The events, given or found; None where the design was given the
        units, or counts no events (:func:`rmst_design`,
        :func:`rate_ratio_design`).
    inputs : dict
        The design's other arguments, as given.
    """

    design: str
    solved_for: str
    value: Any
    power: Any
    n: Any
    events: Any
    inputs: dict

    def to_dict(self) -> dict:
        """The answer as a dict of native Python values, strict JSON, under
        ``"result": "PowerCalculation"``. A distribution among the inputs
        is written as its own ``to_dict()``, and a function by its
        ``repr``.

        Examples
        --------
        >>> from surpyval.test_planning import logrank_design
        >>> d = logrank_design(0.5, power=0.8).to_dict()
        >>> d["solved_for"], d["value"], d["inputs"]["hazard_ratio"]
        ('events', 66, 0.5)
        """
        values = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self)
        }
        values["inputs"] = _plain_inputs(self.inputs)
        return result_dict(self, values)

    def __repr__(self) -> str:
        header = f"Power Calculation: {self.design}"
        rows = [("Solved for", f"{self.solved_for} = {_show(self.value)}")]
        for name in ("power", "events", "n"):
            value = getattr(self, name)
            if value is not None and name != self.solved_for:
                rows.append((name, _show(value)))
        rows += [
            (name, _show(value))
            for name, value in self.inputs.items()
            if value is not None
        ]
        width = max(len(name) for name, _ in rows)
        lines = [header, "=" * len(header)]
        lines += [f"{name:<{width}} : {text}" for name, text in rows]
        return "\n".join(lines)


def _show(value: Any) -> str:
    """A short text for one value: a number to six figures, a model or
    function by its class's name."""
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_show(v) for v in value) + ")"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if callable(value) or hasattr(value, "sf"):
        name = getattr(getattr(value, "dist", None), "name", None)
        return name if isinstance(name, str) else type(value).__name__
    return repr(value)


def _plain_inputs(inputs: dict) -> dict:
    """The inputs, strict JSON: a model by its ``to_dict()``, a function
    by its ``repr``."""
    from surpyval.test_planning._serialise import _plain

    def plain(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if callable(value):
            return repr(value)
        return _plain(value)

    return {name: plain(value) for name, value in inputs.items()}


def _unknown(design: str, given: dict[str, Any], example: str) -> str:
    """The one name in ``given`` whose value is None, or a refusal naming
    them."""
    missing = [name for name, value in given.items() if value is None]
    if len(missing) == 1:
        return missing[0]
    names = ", ".join(given)
    if not missing:
        raise ValueError(
            f"{design}_design: {names} were all given, so nothing is left "
            "to solve for; leave the one to find as None (for example "
            f"{example})."
        )
    raise ValueError(
        f"{design}_design: {', '.join(missing)} were all left as None; "
        f"give all but one of {names}, and the one left as None is solved "
        f"for (for example {example})."
    )


def _size(design: str, n: Any, events: Any) -> Any:
    """The size given: ``n`` or ``events``, not both."""
    if n is not None and events is not None:
        raise ValueError(
            f"{design}_design: give n or events, not both: the events "
            "follow from the units and event_probability. Leave one of "
            "them as None."
        )
    return n if n is not None else events


def _need_probability(design: str, event_probability: Any) -> None:
    if event_probability is None:
        raise ValueError(
            f"{design}_design: n needs event_probability, the chance that "
            "a unit has an event during the study, to give the expected "
            "events; give it, or give events instead of n."
        )


def _allocation(design: str, ratio: Any, n: Any) -> Any:
    """``ratio``, which the units ``n`` fix when they are given."""
    if n is not None and ratio is not None:
        raise ValueError(
            f"{design}_design: ratio is read from n (the units in each "
            "group) when n is given; leave ratio as None."
        )
    return 1.0 if ratio is None else ratio


def _no_direction(design: str, direction: Any, unknown: str, name: str) -> Any:
    if direction is not None and unknown != name:
        raise ValueError(
            f"{design}_design: direction says which way the smallest "
            f"detectable {name} lies, so it applies only when {name} is "
            f"left as None to be solved for; {unknown} is being solved for "
            "here. Leave direction as None."
        )
    return direction


# --- Power from the events, the designs whose information is events -------


@elementwise
def _logrank_power_from_events(
    hazard_ratio: Any,
    events: Any,
    *,
    alpha_ci: Any,
    ratio: Any,
    method: Any,
    alternative: Any,
) -> float:
    hazard_ratio = _comparison._check_hazard_ratio(hazard_ratio)
    events = check_count(events, "events")
    ratio = _comparison._check_ratio(ratio)
    one = _comparison._events(hazard_ratio, 1.0, ratio, method)
    z = np.sqrt(events / one) - _comparison._z_alpha(alpha_ci, alternative)
    return float(stats.norm.cdf(z))


@elementwise
def _cox_power_from_events(
    hazard_ratio: Any,
    events: Any,
    *,
    sd: Any,
    proportion: Any,
    r_squared: Any,
    alpha_ci: Any,
    alternative: Any,
) -> float:
    hazard_ratio = _comparison._check_hazard_ratio(hazard_ratio)
    events = check_count(events, "events")
    variance = _sizes._covariate_variance(sd, proportion, r_squared)
    z = np.sqrt(events * variance) * abs(np.log(hazard_ratio))
    return float(
        stats.norm.cdf(z - _comparison._z_alpha(alpha_ci, alternative))
    )


@elementwise
def _cox_detectable_from_events(
    events: Any,
    power: Any,
    *,
    sd: Any,
    proportion: Any,
    r_squared: Any,
    alpha_ci: Any,
    alternative: Any,
    direction: Any,
) -> float:
    events = check_count(events, "events")
    power = check_probability(power, "power")
    _sizes.check_option("direction", direction, _sizes.DIRECTIONS)
    variance = _sizes._covariate_variance(sd, proportion, r_squared)
    z = _comparison._z_alpha(alpha_ci, alternative) + float(
        stats.norm.ppf(power)
    )
    log_ratio = z / np.sqrt(events * variance)
    return float(np.exp(log_ratio if direction == "increase" else -log_ratio))


@elementwise
def _noninferiority_power_from_events(
    margin: Any,
    events: Any,
    *,
    hazard_ratio: Any,
    alpha_ci: Any,
    ratio: Any,
) -> float:
    margin, hazard_ratio = _sizes._check_margin_above(margin, hazard_ratio)
    events = check_count(events, "events")
    share = _sizes._share(ratio)
    distance = np.log(margin) - np.log(hazard_ratio)
    z = np.sqrt(events * share * (1.0 - share)) * distance
    return float(
        stats.norm.cdf(z - _comparison._z_alpha(alpha_ci, "two-sided"))
    )


def _equivalence_power_from_events(
    margin: Any,
    events: Any,
    *,
    hazard_ratio: Any,
    alpha_ci: Any,
    ratio: Any,
) -> float:
    lower, upper, theta = _sizes._equivalence_margins(margin, hazard_ratio)
    events = check_count(events, "events")
    share = _sizes._share(ratio)
    z = _comparison._z_alpha(alpha_ci, "two-sided")
    return _sizes._equivalence_power_at(
        events * share * (1.0 - share), lower, upper, theta, z
    )


# --- The designs ---------------------------------------------------------


def logrank_design(
    hazard_ratio: float,
    *,
    power: float | None = None,
    events: int | None = None,
    n: tuple[int, int] | None = None,
    event_probability: float | tuple[float, float] | None = None,
    alpha_ci: float = 0.05,
    ratio: float | None = None,
    method: str = "schoenfeld",
    alternative: str = "two-sided",
) -> PowerCalculation:
    r"""The power, events or units of a log-rank (or Cox) test of a hazard
    ratio between two groups, under proportional hazards.

    Give the power to find the size; give the size (``events``, or ``n``
    with ``event_probability``) to find the power. Exactly one of
    ``power`` and the size is left None.

    The events needed are Schoenfeld's (1983),
    :math:`d = (z_{1 - \alpha / 2} + z_{\text{power}})^2 /
    (\pi (1 - \pi) (\ln \text{HR})^2)`, with :math:`\pi = r / (1 + r)` the
    share of the units in the first group, or Freedman's (1982),
    :math:`d = (z_{1 - \alpha / 2} + z_{\text{power}})^2 (r \text{HR} +
    1)^2 / (r (\text{HR} - 1)^2)` with ``method="freedman"`` (lifelines'
    ``sample_size_necessary_under_cph`` and ``power_under_cph``). The
    units are the events, unrounded, shared out in the ratio ``ratio`` and
    divided by each group's chance of an event during the study; the
    power at given units uses their expected events,
    ``n1 * p1 + n2 * p2``.

    A difference planned as a ratio of mean lives maps to a hazard ratio
    only through the shape: for exponential lives a mean life ``r`` times
    as long is a hazard ratio of ``1 / r``, and for Weibull lives with a
    common shape ``beta``, ``r ** -beta`` (a 30% longer mean life is
    ``1 / 1.3`` for exponential lives, ``1.3 ** -2`` for a shape of 2).
    Where the shape is known, :func:`mean_ratio_design` plans the exact
    test of the means themselves.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, of the first group to the second:
        positive, and not 1.
    power : float, optional
        Keyword only: the chance of detecting it; None to find it.
    events : int, optional
        Keyword only: the events, in both groups together. Left None with
        ``n`` to find the size.
    n : (int, int), optional
        Keyword only: the units in the first group and in the second
        (with ``event_probability``). Left None with ``events`` to find
        the size.
    event_probability : float or (float, float), optional
        Keyword only: the chance that a unit has an event during the
        study, one for both groups or a pair. Needed with ``n``; given
        when solving for the size, the units are found too.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the second,
        by default 1. Read from ``n`` when ``n`` is given.
    method : str, optional
        Keyword only: ``"schoenfeld"`` (the default) or ``"freedman"``.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"power"``, ``"events"`` (no
        ``event_probability``) or ``"n"`` (the events found too). The
        events and units are rounded up. ``hazard_ratio``, ``power`` and
        ``events`` may be arrays, answered element by element, when the
        events alone are found.

    Raises
    ------
    ValueError
        If none or both of ``power`` and the size are given, both ``n``
        and ``events`` are, ``n`` comes without ``event_probability`` or
        with ``ratio``, or an argument is out of range.

    Examples
    --------
    Detecting a halving of the hazard with 80% power at the 5% level
    takes 66 events (88 for 90% power), with equal groups:

    >>> from surpyval.test_planning import logrank_design
    >>> logrank_design(0.5, power=0.8).events
    66
    >>> logrank_design(0.5, power=0.9).events
    88
    >>> logrank_design(0.5, power=0.8, method="freedman").events
    71

    A design expected to cut the failure rate by 30%, where 25% of the
    new units and 35% of the old would fail during the study, and the
    power with 200 units in each group:

    >>> logrank_design(0.7, power=0.8, event_probability=(0.25, 0.35)).n
    (412, 412)
    >>> plan = logrank_design(0.7, n=(200, 200),
    ...                       event_probability=(0.25, 0.35))
    >>> plan.solved_for, round(plan.value, 3)
    ('power', 0.497)
    """
    unknown = _unknown(
        "logrank",
        {"power": power, "n or events": _size("logrank", n, events)},
        "logrank_design(0.5, power=0.8)",
    )
    inputs = {
        "hazard_ratio": hazard_ratio,
        "event_probability": event_probability,
        "alpha_ci": alpha_ci,
        "ratio": ratio,
        "method": method,
        "alternative": alternative,
    }
    common: dict[str, Any] = {"alpha_ci": alpha_ci, "method": method}
    common["alternative"] = alternative
    if unknown == "power":
        if n is not None:
            _need_probability("logrank", event_probability)
            _allocation("logrank", ratio, n)
            found = _comparison._logrank_power(
                hazard_ratio, n, event_probability, **common
            )
        else:
            found = _logrank_power_from_events(
                hazard_ratio,
                events,
                ratio=_allocation("logrank", ratio, n),
                **common,
            )
        return PowerCalculation(
            "logrank", "power", found, found, n, events, inputs
        )
    ratio_ = _allocation("logrank", ratio, None)
    events = _comparison._logrank_events(
        hazard_ratio, power, ratio=ratio_, **common
    )
    if event_probability is None:
        return PowerCalculation(
            "logrank", "events", events, power, None, events, inputs
        )
    n = _comparison._logrank_sample_size(
        hazard_ratio, event_probability, power, ratio=ratio_, **common
    )
    return PowerCalculation("logrank", "n", n, power, n, events, inputs)


def cox_design(
    hazard_ratio: float | None = None,
    *,
    power: float | None = None,
    events: int | None = None,
    n: int | None = None,
    event_probability: float | None = None,
    sd: float | None = None,
    proportion: float | None = None,
    r_squared: float = 0.0,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
) -> PowerCalculation:
    r"""The power, events, units or detectable hazard ratio of a Cox
    model's coefficient for one covariate (Hsieh and Lavori 2000).

    Of ``hazard_ratio``, ``power`` and the size (``events``, or ``n``
    with ``event_probability``), exactly one is left None and found.

    The events needed to detect a hazard ratio per unit of a covariate
    with variance :math:`\sigma^2` (``sd`` squared, or :math:`\pi (1 -
    \pi)` for a covariate of two levels with ``proportion`` :math:`\pi` of
    the units at the first) and squared multiple correlation :math:`R^2`
    with the model's other covariates (``r_squared``) are
    :math:`d = (z_{1 - \alpha / 2} + z_{\text{power}})^2 /
    (\sigma^2 (1 - R^2) (\ln \text{HR})^2)`, with ``r_squared=0`` and
    a two-level covariate Schoenfeld's; the units are the events over the
    chance of an event.

    Parameters
    ----------
    hazard_ratio : float, optional
        The hazard ratio per unit of the covariate: positive, and not 1.
        None to find the smallest detectable one.
    power : float, optional
        Keyword only: the chance of detecting it; None to find it.
    events : int, optional
        Keyword only: the events.
    n : int, optional
        Keyword only: the units (with ``event_probability``).
    event_probability : float, optional
        Keyword only: the chance that a unit has an event during the
        study. Needed with ``n``; given when solving for the size, the
        units are found too.
    sd, proportion : float, optional
        Keyword only: the covariate's standard deviation, or, for one of
        two levels, the proportion of the units at the first. Give one.
    r_squared : float, optional
        Keyword only: its squared multiple correlation with the other
        covariates, at least 0 and below 1; by default 0.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.
    direction : str, optional
        Keyword only, when finding the hazard ratio: ``"increase"`` (the
        default, a ratio above 1) or ``"decrease"`` (below 1).

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"hazard_ratio"``, ``"power"``, ``"events"``
        (no ``event_probability``) or ``"n"`` (the events found too).
        Arguments may be arrays, answered element by element.

    Raises
    ------
    ValueError
        If not exactly one of them is left None, both ``n`` and
        ``events`` are given, ``n`` comes without ``event_probability``,
        ``direction`` is given when the hazard ratio is not being found,
        or an argument is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.test_planning import cox_design
    >>> cox_design(1.5, power=0.8, sd=1.0).events
    48
    >>> cox_design(1.5, power=0.8, sd=1.0, r_squared=0.3).events
    69
    >>> cox_design(0.5, power=0.8, proportion=0.5).events  # Schoenfeld's 66
    66

    Hsieh and Lavori's example, a one-sided test at 5%:

    >>> study = dict(event_probability=0.738, sd=0.3126, r_squared=0.1837,
    ...              alternative="one-sided")
    >>> cox_design(np.exp(1.0), power=0.806, **study).n
    107
    >>> round(cox_design(np.exp(1.0), n=107, **study).power, 3)
    0.806

    200 units of which 30% have an event detect a hazard ratio of about
    1.44 per standard deviation:

    >>> smallest = cox_design(power=0.8, n=200, event_probability=0.3, sd=1.0)
    >>> smallest.solved_for, round(smallest.value, 3)
    ('hazard_ratio', 1.436)
    """
    unknown = _unknown(
        "cox",
        {
            "hazard_ratio": hazard_ratio,
            "power": power,
            "n or events": _size("cox", n, events),
        },
        "cox_design(1.5, power=0.8, sd=1.0)",
    )
    _no_direction("cox", direction, unknown, "hazard_ratio")
    inputs = {
        "event_probability": event_probability,
        "sd": sd,
        "proportion": proportion,
        "r_squared": r_squared,
        "alpha_ci": alpha_ci,
        "alternative": alternative,
        "direction": direction,
    }
    if unknown != "hazard_ratio":
        inputs = {"hazard_ratio": hazard_ratio, **inputs}
    common: dict[str, Any] = {
        "sd": sd,
        "proportion": proportion,
        "r_squared": r_squared,
        "alpha_ci": alpha_ci,
        "alternative": alternative,
    }
    if unknown in ("hazard_ratio", "power"):
        if n is not None:
            _need_probability("cox", event_probability)
        if unknown == "hazard_ratio":
            way = "increase" if direction is None else direction
            if n is not None:
                found = _sizes._cox_detectable(
                    n, event_probability, power, direction=way, **common
                )
            else:
                found = _cox_detectable_from_events(
                    events, power, direction=way, **common
                )
            return PowerCalculation(
                "cox", "hazard_ratio", found, power, n, events, inputs
            )
        if n is not None:
            found = _sizes._cox_power(
                hazard_ratio, n, event_probability, **common
            )
        else:
            found = _cox_power_from_events(hazard_ratio, events, **common)
        return PowerCalculation(
            "cox", "power", found, found, n, events, inputs
        )
    events = _sizes._cox_events(hazard_ratio, power, **common)
    if event_probability is None:
        return PowerCalculation(
            "cox", "events", events, power, None, events, inputs
        )
    n = _sizes._cox_sample_size(
        hazard_ratio, event_probability, power, **common
    )
    return PowerCalculation("cox", "n", n, power, n, events, inputs)


def _margin_design(
    design: str,
    margin: Any,
    power: Any,
    events: Any,
    n: Any,
    event_probability: Any,
    hazard_ratio: float,
    alpha_ci: float,
    ratio: Any,
) -> PowerCalculation:
    """:func:`noninferiority_design` and :func:`equivalence_design`, which
    differ only in their equation."""
    example = f"{design}_design(1.3, power=0.8)"
    unknown = _unknown(
        design,
        {"power": power, "n or events": _size(design, n, events)},
        example,
    )
    inputs = {
        "margin": margin,
        "event_probability": event_probability,
        "hazard_ratio": hazard_ratio,
        "alpha_ci": alpha_ci,
        "ratio": ratio,
    }
    common: dict[str, Any] = {
        "hazard_ratio": hazard_ratio,
        "alpha_ci": alpha_ci,
    }
    private = {
        "noninferiority": (
            _sizes._noninferiority_power,
            _noninferiority_power_from_events,
            _sizes._noninferiority_events,
            _sizes._noninferiority_sample_size,
        ),
        "equivalence": (
            _sizes._equivalence_power,
            _equivalence_power_from_events,
            _sizes._equivalence_events,
            _sizes._equivalence_sample_size,
        ),
    }
    of_n, of_events, events_for, units_for = private[design]
    if unknown == "power":
        if n is not None:
            _need_probability(design, event_probability)
            _allocation(design, ratio, n)
            found = of_n(margin, n, event_probability, **common)
        else:
            found = of_events(
                margin, events, ratio=_allocation(design, ratio, n), **common
            )
        return PowerCalculation(
            design, "power", found, found, n, events, inputs
        )
    ratio_ = _allocation(design, ratio, None)
    events = events_for(margin, power, ratio=ratio_, **common)
    if event_probability is None:
        return PowerCalculation(
            design, "events", events, power, None, events, inputs
        )
    n = units_for(margin, event_probability, power, ratio=ratio_, **common)
    return PowerCalculation(design, "n", n, power, n, events, inputs)


def noninferiority_design(
    margin: float,
    *,
    power: float | None = None,
    events: int | None = None,
    n: tuple[int, int] | None = None,
    event_probability: float | tuple[float, float] | None = None,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
    ratio: float | None = None,
) -> PowerCalculation:
    r"""The power, events or units to show that a hazard ratio is below a
    margin (non-inferiority).

    Give the power to find the size; give the size (``events``, or ``n``
    with ``event_probability``) to find the power.

    The one-sided test at level :math:`\alpha / 2` (a two-sided
    ``1 - alpha_ci`` interval's upper end below the margin) needs
    :math:`d = (z_{1 - \alpha / 2} + z_{\text{power}})^2 / (\pi (1 - \pi)
    (\ln \text{margin} - \ln \text{HR})^2)` events (Jung and others 2005;
    Chow, Shao and Wang 2008), with :math:`\text{HR}` the true hazard
    ratio (``hazard_ratio``) and :math:`\pi` the share of the units in
    the first group.

    Parameters
    ----------
    margin : float
        The non-inferiority margin of the hazard ratio, first group to
        second, above ``hazard_ratio``.
    power : float, optional
        Keyword only: the chance of showing it; None to find it.
    events : int, optional
        Keyword only: the events, in both groups together.
    n : (int, int), optional
        Keyword only: the units in each group (with
        ``event_probability``).
    event_probability : float or (float, float), optional
        Keyword only: the chance of an event during the study, one for
        both groups or a pair. Needed with ``n``; given when solving for
        the size, the units are found too.
    hazard_ratio : float, optional
        Keyword only: the true hazard ratio, below ``margin``; by default
        1.
    alpha_ci : float, optional
        Keyword only: one minus the confidence level of the interval
        compared with the margin, by default 0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the second,
        by default 1. Read from ``n`` when ``n`` is given.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"power"``, ``"events"`` or ``"n"``.

    Raises
    ------
    ValueError
        If none or both of ``power`` and the size are given, or an
        argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import noninferiority_design
    >>> noninferiority_design(1.3, power=0.8).events
    457
    >>> noninferiority_design(1.3, power=0.8, hazard_ratio=0.9).events
    233
    >>> noninferiority_design(1.3, power=0.8, event_probability=0.4).n
    (571, 571)
    >>> plan = noninferiority_design(1.3, n=(571, 571), event_probability=0.4)
    >>> round(plan.power, 3)
    0.801
    """
    return _margin_design(
        "noninferiority",
        margin,
        power,
        events,
        n,
        event_probability,
        hazard_ratio,
        alpha_ci,
        ratio,
    )


def equivalence_design(
    margin: float | tuple[float, float],
    *,
    power: float | None = None,
    events: int | None = None,
    n: tuple[int, int] | None = None,
    event_probability: float | tuple[float, float] | None = None,
    hazard_ratio: float = 1.0,
    alpha_ci: float = 0.05,
    ratio: float | None = None,
) -> PowerCalculation:
    r"""The power, events or units to show that a hazard ratio lies between
    two margins (equivalence).

    Give the power to find the size; give the size (``events``, or ``n``
    with ``event_probability``) to find the power.

    Two one-sided tests: equivalence is shown when the ``1 - alpha_ci``
    confidence interval of the hazard ratio lies inside the margins. Its
    power, with :math:`\theta` the true log hazard ratio, :math:`L` and
    :math:`U` the log margins and :math:`\text{se} = 1 / \sqrt{d \pi (1 -
    \pi)}`, is

    .. math::
        \Phi\!\left(\frac{U - \theta}{\text{se}} - z_{1 - \alpha/2}\right)
        - \Phi\!\left(\frac{L - \theta}{\text{se}} + z_{1 - \alpha/2}\right),

    solved for the events :math:`d` when the power is given. The classical
    two one-sided tests at 5% each compare a 90% interval:
    ``alpha_ci=0.1``.

    Parameters
    ----------
    margin : float or (float, float)
        The equivalence margins of the hazard ratio: one above 1, for
        ``(1 / margin, margin)``, or a pair ``(lower, upper)``.
    power : float, optional
        Keyword only: the chance of showing it; None to find it.
    events : int, optional
        Keyword only: the events, in both groups together.
    n : (int, int), optional
        Keyword only: the units in each group (with
        ``event_probability``).
    event_probability : float or (float, float), optional
        Keyword only: the chance of an event during the study. Needed
        with ``n``; given when solving for the size, the units are found
        too.
    hazard_ratio : float, optional
        Keyword only: the true hazard ratio, strictly between the
        margins; by default 1.
    alpha_ci : float, optional
        Keyword only: one minus the confidence level of the interval
        compared with the margins, by default 0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the second,
        by default 1. Read from ``n`` when ``n`` is given.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"power"``, ``"events"`` or ``"n"``.

    Raises
    ------
    ValueError
        If none or both of ``power`` and the size are given, or an
        argument is out of range.

    Examples
    --------
    >>> from surpyval.test_planning import equivalence_design
    >>> equivalence_design(1.25, power=0.8).events
    845
    >>> equivalence_design(1.25, power=0.8, alpha_ci=0.1).events
    688
    >>> equivalence_design(1.25, power=0.8, event_probability=0.5,
    ...                    alpha_ci=0.1).n
    (688, 688)
    >>> round(equivalence_design(1.25, n=(688, 688), event_probability=0.5,
    ...                   alpha_ci=0.1).power, 3)
    0.8
    """
    return _margin_design(
        "equivalence",
        margin,
        power,
        events,
        n,
        event_probability,
        hazard_ratio,
        alpha_ci,
        ratio,
    )


def rmst_design(
    survival: tuple[Any, Any],
    tau: float,
    *,
    power: float | None = None,
    n: tuple[int, int] | None = None,
    accrual_time: float = 0.0,
    follow_up: float | None = None,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    ratio: float | None = None,
    alternative: str = "two-sided",
) -> PowerCalculation:
    r"""The power or units to detect a difference in restricted mean
    survival time up to ``tau`` between two groups whose survival curves
    are given.

    Give ``power`` to find ``n``, or ``n`` to find the power. The test
    statistic is the difference of the groups' Kaplan-Meier RMSTs
    (:func:`surpyval.rmst_diff`) over its standard error, each group's
    variance computed from its curve and the study's censoring (uniform
    entry over ``accrual_time``, ``follow_up`` after it, exponential
    dropout at ``dropout_rate``), as in Royston and Parmar (2013) and R
    ``npsurvSS``.

    Parameters
    ----------
    survival : (distribution, distribution)
        The two groups' survival curves: distributions with an ``sf``, or
        functions of time giving the survival probability.
    tau : float
        The horizon of the restricted mean.
    power : float, optional
        Keyword only: the chance of detecting the difference; None to find
        it.
    n : (int, int), optional
        Keyword only: the units in each group; None to find them.
    accrual_time : float, optional
        Keyword only: the time over which units enter, by default 0.
    follow_up : float, optional
        Keyword only: the follow-up after the last entry, by default
        ``tau`` (the study ends once every unit has been seen to ``tau``).
    dropout_rate : float, optional
        Keyword only: the rate of exponential loss to follow-up, by
        default 0.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the second,
        by default 1, when finding ``n``.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"power"`` or ``"n"``; ``events`` is None.

    Raises
    ------
    ValueError
        If none or both of ``power`` and ``n`` are given, ``ratio`` is
        given with ``n``, or an argument is out of range.

    Examples
    --------
    Exponential lives with means 10 and 6 years, compared over the first
    5 years, units entering over 2 years and followed for 4 more:

    >>> import surpyval
    >>> from surpyval.test_planning import rmst_design
    >>> curves = (surpyval.Exponential.from_params([1 / 10]),
    ...           surpyval.Exponential.from_params([1 / 6]))
    >>> rmst_design(curves, 5.0, power=0.8, accrual_time=2.0, follow_up=4.0).n
    (152, 152)
    >>> round(rmst_design(curves, 5.0, n=(152, 152), accrual_time=2.0,
    ...            follow_up=4.0).power, 3)
    0.8
    """
    unknown = _unknown(
        "rmst", {"power": power, "n": n}, "rmst_design(curves, 5.0, power=0.8)"
    )
    inputs = {
        "survival": survival,
        "tau": tau,
        "accrual_time": accrual_time,
        "follow_up": follow_up,
        "dropout_rate": dropout_rate,
        "alpha_ci": alpha_ci,
        "ratio": ratio,
        "alternative": alternative,
    }
    common: dict[str, Any] = {
        "accrual_time": accrual_time,
        "follow_up": follow_up,
        "dropout_rate": dropout_rate,
        "alpha_ci": alpha_ci,
        "alternative": alternative,
    }
    if unknown == "power":
        _allocation("rmst", ratio, n)
        found = _sizes._rmst_power(survival, tau, n, **common)
        return PowerCalculation("rmst", "power", found, found, n, None, inputs)
    n = _sizes._rmst_sample_size(
        survival, tau, power, ratio=_allocation("rmst", ratio, None), **common
    )
    return PowerCalculation("rmst", "n", n, power, n, None, inputs)


def cure_logrank_design(
    cured: float | tuple[float, float],
    latency: Any,
    *,
    follow_up: float,
    power: float | None = None,
    n: tuple[int, int] | None = None,
    latency_hazard_ratio: float = 1.0,
    accrual_time: float = 0.0,
    dropout_rate: float = 0.0,
    alpha_ci: float = 0.05,
    ratio: float | None = None,
    alternative: str = "two-sided",
) -> PowerCalculation:
    r"""The power, or the units and events, of a log-rank test when some
    units are cured (never have the event).

    Give ``power`` to find ``n`` (and the events expected with it), or
    ``n`` to find the power. Each group's survival is
    :math:`S_j(t) = \pi_j + (1 - \pi_j) S_{u,j}(t)`, with :math:`\pi_j` its
    cured fraction and :math:`S_{u,j}` the latency distribution, so the
    hazards are not proportional; the mean and variance of the log-rank
    statistic under that alternative (Schoenfeld 1981) are integrated
    over the study's censoring (Wang, Zhang and Lu 2012).

    Parameters
    ----------
    cured : float or (float, float)
        The cured fraction, one for both groups or a pair, each at least 0
        and below 1.
    latency : distribution or (distribution, distribution)
        The latency distribution of the units that are not cured: one for
        both groups (with ``latency_hazard_ratio``) or a pair.
    follow_up : float
        Keyword only: the follow-up after the last entry.
    power : float, optional
        Keyword only: the chance of detecting the difference; None to find
        it.
    n : (int, int), optional
        Keyword only: the units in each group; None to find them.
    latency_hazard_ratio : float, optional
        Keyword only: the hazard ratio of the first group's latency to the
        second's, with one latency distribution; by default 1.
    accrual_time, dropout_rate : float, optional
        Keyword only: the time over which units enter and the rate of
        exponential loss to follow-up, by default 0.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the second,
        by default 1, when finding ``n``.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"power"`` or ``"n"``; finding ``n`` gives the
        expected ``events`` too, each rounded up.

    Raises
    ------
    ValueError
        If none or both of ``power`` and ``n`` are given, ``ratio`` is
        given with ``n``, or an argument is out of range.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.test_planning import cure_logrank_design
    >>> latency = surpyval.Weibull.from_params([2.0, 1.2])
    >>> plan = cure_logrank_design((0.45, 0.30), latency, power=0.8,
    ...                     accrual_time=1.0, follow_up=3.0)
    >>> plan.n, plan.events
    ((247, 247), 265)
    >>> round(cure_logrank_design((0.45, 0.30), latency, n=(247, 247),
    ...                    accrual_time=1.0, follow_up=3.0).power, 3)
    0.8
    """
    unknown = _unknown(
        "cure_logrank",
        {"power": power, "n": n},
        "cure_logrank_design(0.3, latency, power=0.8, follow_up=3.0)",
    )
    inputs = {
        "cured": cured,
        "latency": latency,
        "follow_up": follow_up,
        "latency_hazard_ratio": latency_hazard_ratio,
        "accrual_time": accrual_time,
        "dropout_rate": dropout_rate,
        "alpha_ci": alpha_ci,
        "ratio": ratio,
        "alternative": alternative,
    }
    common: dict[str, Any] = {
        "latency_hazard_ratio": latency_hazard_ratio,
        "follow_up": follow_up,
        "accrual_time": accrual_time,
        "dropout_rate": dropout_rate,
        "alpha_ci": alpha_ci,
        "alternative": alternative,
    }
    if unknown == "power":
        _allocation("cure_logrank", ratio, n)
        found = _sizes._cure_logrank_power(cured, latency, n, **common)
        return PowerCalculation(
            "cure_logrank", "power", found, found, n, None, inputs
        )
    ratio_ = _allocation("cure_logrank", ratio, None)
    n = _sizes._cure_logrank_sample_size(
        cured, latency, power, ratio=ratio_, **common
    )
    events = _sizes._cure_logrank_events(
        cured, latency, power, ratio=ratio_, **common
    )
    return PowerCalculation("cure_logrank", "n", n, power, n, events, inputs)


def rate_ratio_design(
    ratio: float | None = None,
    *,
    rate: float,
    exposure_before: float,
    exposure_after: float | None = None,
    power: float | None = None,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    method: str = "exact",
    direction: str | None = None,
) -> PowerCalculation:
    r"""The power, the exposure after, or the smallest detectable change of
    the exact conditional test of a change in a failure rate at an
    intervention (:func:`surpyval.recurrent.intervention_test`).

    Of ``ratio``, ``exposure_after`` and ``power``, exactly one is left
    None and found. The power is the exact sum over the Poisson count of
    failures, under a homogeneous Poisson process with rate ``rate``
    before the intervention and ``ratio`` times it after; see
    :mod:`surpyval.test_planning.rate_ratio`.

    Parameters
    ----------
    ratio : float, optional
        The rate after the intervention over the rate before; None to find
        the smallest change detected with ``power``.
    rate : float
        Keyword only: the failure rate before the intervention.
    exposure_before : float
        Keyword only: the exposure (unit-time) before it.
    exposure_after : float, optional
        Keyword only: the exposure after it (in all); None to find it.
    power : float, optional
        Keyword only: the chance of detecting the change; None to find it.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``.
    method : str, optional
        Keyword only: ``"exact"`` (the default) or ``"mid-p"``.
    direction : str, optional
        Keyword only, when finding ``ratio``: ``"decrease"`` (the
        default) or ``"increase"``.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"ratio"``, ``"exposure_after"`` or
        ``"power"``; ``n`` and ``events`` are None. Arguments may be
        arrays, answered element by element.

    Raises
    ------
    ValueError
        If not exactly one of them is left None, ``direction`` is given
        when the ratio is not being found, or an argument is out of range.

    Examples
    --------
    30 failures in 10,000 hours before a change:

    >>> from surpyval.test_planning import rate_ratio_design
    >>> round(rate_ratio_design(0.5, rate=30 / 10000, exposure_before=10000,
    ...                         exposure_after=5000).power, 3)
    0.356
    >>> round(rate_ratio_design(rate=30 / 10000, exposure_before=10000,
    ...                         exposure_after=5000, power=0.8).value, 3)
    0.263
    >>> round(rate_ratio_design(0.5, rate=30 / 10000, exposure_before=10000,
    ...                         power=0.8).value, -2)
    41200.0
    """
    unknown = _unknown(
        "rate_ratio",
        {"ratio": ratio, "exposure_after": exposure_after, "power": power},
        "rate_ratio_design(0.5, rate=0.003, exposure_before=10000, power=0.8)",
    )
    _no_direction("rate_ratio", direction, unknown, "ratio")
    common: dict[str, Any] = {"alpha_ci": alpha_ci, "alternative": alternative}
    common["method"] = method
    inputs = {
        "rate": rate,
        "exposure_before": exposure_before,
        **common,
        "direction": direction,
    }
    if unknown == "power":
        found = _rates._rate_ratio_power(
            ratio, rate, exposure_before, exposure_after, **common
        )
    elif unknown == "ratio":
        way = "decrease" if direction is None else direction
        found = _rates._rate_ratio_detectable(
            rate,
            exposure_before,
            exposure_after,
            power,
            direction=way,
            **common,
        )
    else:
        found = _rates._rate_ratio_exposure(
            ratio, rate, exposure_before, power, **common
        )
    values = {"ratio": ratio, "exposure_after": exposure_after}
    if unknown != "power":
        values[unknown] = found
    result = PowerCalculation(
        "rate_ratio",
        unknown,
        found,
        found if unknown == "power" else power,
        None,
        None,
        {**values, **inputs},
    )
    return result


# --- Two means (or scales at a common shape): the exact F test -----------


def _check_mean_ratio(mean_ratio: Any) -> float:
    mean_ratio = float(check_positive(mean_ratio, "mean_ratio"))
    if mean_ratio == 1.0:
        raise ValueError(
            "mean_ratio must differ from 1: no test can detect a "
            "difference of none."
        )
    return mean_ratio


def _f_power(
    mean_ratio: float,
    first: float,
    second: float,
    shape: float,
    alpha_ci: float,
    alternative: str,
) -> float:
    """The power of the F test of two scales with ``first`` and
    ``second`` events: the ratio of the estimated scales to the power
    ``shape`` is ``mean_ratio ** shape`` times an F(2 first, 2 second)
    variate."""
    check_option("alternative", alternative, _comparison.ALTERNATIVES)
    law = stats.f(2.0 * first, 2.0 * second)
    q = mean_ratio**shape
    if alternative == "two-sided":
        upper = law.isf(alpha_ci / 2.0)
        lower = law.ppf(alpha_ci / 2.0)
        return float(law.sf(upper / q) + law.cdf(lower / q))
    if q > 1.0:
        return float(law.sf(law.isf(alpha_ci) / q))
    return float(law.cdf(law.ppf(alpha_ci) / q))


def _check_shape_value(shape: Any) -> float:
    return float(check_positive(shape, "shape"))


@elementwise
def _mean_ratio_power_from_events(
    mean_ratio: Any,
    events: Any,
    *,
    shape: Any,
    alpha_ci: Any,
    alternative: Any,
) -> float:
    mean_ratio = _check_mean_ratio(mean_ratio)
    events = check_count(events, "events", minimum=2)
    shape = _check_shape_value(shape)
    half = events / 2.0
    return _f_power(mean_ratio, half, half, shape, alpha_ci, alternative)


@elementwise
def _mean_ratio_events(
    mean_ratio: Any,
    power: Any,
    *,
    shape: Any,
    alpha_ci: Any,
    alternative: Any,
) -> int:
    """The fewest events, the same number in each group, whose F test
    has ``power``: twice the events per group."""
    mean_ratio = _check_mean_ratio(mean_ratio)
    power = check_probability(power, "power")
    shape = _check_shape_value(shape)

    def short(d: int) -> bool:
        found = _f_power(mean_ratio, d, d, shape, alpha_ci, alternative)
        return found < power

    low, high = 1, 1
    while short(high):
        if high > 10**9:
            raise ValueError(
                "mean_ratio_design: no number of events up to 1e9 in each "
                f"group reaches a power of {power!r} at a mean_ratio of "
                f"{mean_ratio!r}; plan a larger difference."
            )
        low, high = high + 1, 2 * high
    while low < high:
        middle = (low + high) // 2
        if short(middle):
            low = middle + 1
        else:
            high = middle
    return 2 * int(low)


@elementwise
def _mean_ratio_detectable(
    events: Any,
    power: Any,
    *,
    shape: Any,
    alpha_ci: Any,
    alternative: Any,
    direction: Any,
) -> float:
    events = check_count(events, "events", minimum=2)
    power = check_probability(power, "power")
    shape = _check_shape_value(shape)
    _sizes.check_option("direction", direction, _sizes.DIRECTIONS)
    sign = 1.0 if direction == "increase" else -1.0
    half = events / 2.0

    def gap(log_ratio: float) -> float:
        ratio = float(np.exp(sign * log_ratio))
        return _f_power(ratio, half, half, shape, alpha_ci, alternative) - (
            power
        )

    high = 1.0
    while gap(high) < 0.0:
        high *= 2.0
    log_ratio = optimize.brentq(gap, 1e-12, high, xtol=1e-14, rtol=1e-12)
    return float(np.exp(sign * log_ratio))


def mean_ratio_design(
    mean_ratio: float | None = None,
    *,
    shape: float = 1.0,
    power: float | None = None,
    events: int | None = None,
    n: tuple[int, int] | None = None,
    event_probability: float | tuple[float, float] | None = None,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
) -> PowerCalculation:
    r"""The power, events, units or detectable ratio of a comparison of
    two groups' mean lives, exponential or Weibull with a known common
    shape.

    Of ``mean_ratio``, ``power`` and the size (``events``, or ``n`` with
    ``event_probability``), exactly one is left None and found.

    With a common Weibull shape :math:`\beta` the ratio of the mean lives
    is the ratio of the scales, :math:`\rho = \eta_1 / \eta_2`, and
    :math:`t^\beta` is exponential with mean :math:`\eta^\beta`. With
    :math:`d_1` and :math:`d_2` failures, each group's estimate
    :math:`\hat\theta_i = \sum t^\beta / d_i` (total time on test, every
    unit's running time to the power :math:`\beta`, over the failures)
    has :math:`2 d_i \hat\theta_i / \theta_i \sim \chi^2_{2 d_i}`, so
    :math:`\hat\theta_1 / \hat\theta_2 \sim \rho^\beta F(2 d_1, 2 d_2)`
    (Epstein and Sobel 1953; Lawless, *Statistical Models and Methods for
    Lifetime Data*, 2003, section 3.5): the test rejects equal means when
    the ratio falls outside the F distribution's ``alpha_ci`` tails, and
    its power is exact for tests stopped at a number of failures (Type
    II), close for tests stopped at a time. The events are found the same
    in each group (twice that in all), the units from each group's chance
    of a failure during the test.

    The log-rank test of the same difference (:func:`logrank_design`)
    needs nothing assumed about the shape: under the common shape the
    hazard ratio of the first group to the second is
    :math:`\rho^{-\beta}` (``1 / mean_ratio`` for exponential lives), and
    in large samples it needs the same events as this test. Knowing the
    shape is what this design assumes; one estimated from the data costs
    events this design does not count.

    Parameters
    ----------
    mean_ratio : float, optional
        The mean life of the first group over the second's (their scale
        ratio at the common shape): positive, and not 1. None to find the
        smallest detectable one.
    shape : float, optional
        Keyword only: the common Weibull shape, known; by default 1
        (exponential lives).
    power : float, optional
        Keyword only: the chance of detecting the difference; None to
        find it.
    events : int, optional
        Keyword only: the failures, in both groups together, half in
        each.
    n : (int, int), optional
        Keyword only: the units in the first group and in the second
        (with ``event_probability``); their expected failures,
        ``n * event_probability``, are each group's.
    event_probability : float or (float, float), optional
        Keyword only: the chance that a unit fails during the test, one
        for both groups or a pair. Needed with ``n``; given when solving
        for the size, the units are found too.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05.
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default) or ``"one-sided"``
        (in the direction of ``mean_ratio``).
    direction : str, optional
        Keyword only, when finding ``mean_ratio``: ``"increase"`` (the
        default, a ratio above 1) or ``"decrease"``.

    Returns
    -------
    PowerCalculation
        ``solved_for`` is ``"mean_ratio"``, ``"power"``, ``"events"`` (no
        ``event_probability``) or ``"n"`` (the events found too).
        ``mean_ratio``, ``power`` and ``events`` may be arrays, answered
        element by element, unless ``n`` is given.

    Raises
    ------
    ValueError
        If not exactly one of them is left None, both ``n`` and
        ``events`` are given, ``n`` comes without ``event_probability``,
        ``direction`` is given when the ratio is not being found, or an
        argument is out of range.

    Examples
    --------
    Detecting a 30% longer mean life with 80% power at the 5% level, for
    exponential lives, takes 458 failures (229 in each group); the
    log-rank test of the hazard ratio ``1 / 1.3`` plans 457 by
    Schoenfeld's approximation. For a wear-out shape of 2, about a
    quarter of them:

    >>> from surpyval.test_planning import logrank_design, mean_ratio_design
    >>> mean_ratio_design(1.3, power=0.8).events
    458
    >>> logrank_design(1 / 1.3, power=0.8).events
    457
    >>> mean_ratio_design(1.3, shape=2.0, power=0.8).events
    116
    >>> mean_ratio_design(1.3, shape=2.0, power=0.8,
    ...                   event_probability=0.5).n
    (116, 116)
    >>> round(mean_ratio_design(1.3, shape=2.0, n=(100, 100),
    ...                         event_probability=0.5).power, 3)
    0.742

    The smallest ratio 116 failures detect with that power:

    >>> round(mean_ratio_design(shape=2.0, events=116, power=0.8).value, 3)
    1.298
    """
    size = _size("mean_ratio", n, events)
    unknown = _unknown(
        "mean_ratio",
        {"mean_ratio": mean_ratio, "power": power, "n or events": size},
        "mean_ratio_design(1.3, power=0.8)",
    )
    _no_direction("mean_ratio", direction, unknown, "mean_ratio")
    common: dict[str, Any] = {"shape": shape, "alpha_ci": alpha_ci}
    common["alternative"] = alternative
    inputs = {
        "mean_ratio": mean_ratio,
        **common,
        "event_probability": event_probability,
        "direction": direction,
    }
    if n is not None:
        _need_probability("mean_ratio", event_probability)
        first, second = _expected_failures(n, event_probability)
    if unknown == "power":
        if n is None:
            found = _mean_ratio_power_from_events(mean_ratio, events, **common)
        else:
            _check_shape_value(shape)
            found = _f_power(
                _check_mean_ratio(mean_ratio),
                first,
                second,
                float(shape),
                alpha_ci,
                alternative,
            )
        return PowerCalculation(
            "mean_ratio", "power", found, found, n, events, inputs
        )
    if unknown == "mean_ratio":
        way = "increase" if direction is None else direction
        if n is not None:
            raise ValueError(
                "mean_ratio_design: to find the detectable mean_ratio, give "
                "the events (in both groups together) rather than n."
            )
        found = _mean_ratio_detectable(events, power, direction=way, **common)
        return PowerCalculation(
            "mean_ratio",
            "mean_ratio",
            found,
            power,
            None,
            events,
            {**inputs, "mean_ratio": found},
        )
    events = _mean_ratio_events(mean_ratio, power, **common)
    if event_probability is None:
        return PowerCalculation(
            "mean_ratio", "events", events, power, None, events, inputs
        )
    p1, p2 = _comparison._event_probabilities(event_probability)
    per_group = int(events) // 2
    units = (
        int(np.ceil(per_group / p1 - 1e-9)),
        int(np.ceil(per_group / p2 - 1e-9)),
    )
    return PowerCalculation(
        "mean_ratio", "n", units, power, units, events, inputs
    )


def _expected_failures(n: Any, event_probability: Any) -> tuple[float, float]:
    """Each group's expected failures, ``n * event_probability``."""
    pair = tuple(np.ravel(n))
    if len(pair) != 2:
        raise ValueError(
            "mean_ratio_design: n must be the units in the first group and "
            f"in the second, a pair, got {n!r}."
        )
    n1, n2 = (check_count(v, "n") for v in pair)
    p1, p2 = _comparison._event_probabilities(event_probability)
    return n1 * p1, n2 * p2
