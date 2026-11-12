"""Group-sequential designs: a comparison looked at a few times as the
data come in, with boundaries that stop it early for a clear difference
(efficacy) or for a clear lack of one (futility), while keeping the
chance of a false finding at the level asked for.

The core is information-based. At each look ``k`` a normal test
statistic ``Z_k`` is computed from the information ``I_k`` gathered so
far (for a log-rank test the information is proportional to the number
of events). With independent increments, ``Z_k sqrt(I_k)`` is a Brownian
motion with drift ``theta`` in the information: ``Z_k`` has mean
``theta sqrt(I_k)`` and ``corr(Z_j, Z_k) = sqrt(I_j / I_k)`` for
``j <= k``. Every probability of the design (of crossing a boundary at
a look, before crossing another) is a multivariate normal integral,
computed by the recursive numerical integration of Armitage, McPherson
and Rowe (1969), in the form of Jennison and Turnbull (2000, chapter 19):
Simpson's rule on a grid of ``12 r - 3`` points per look, exact to about
``1e-8`` with the default ``r = 32``. Nothing is simulated.

:func:`group_sequential_design` builds a design of this general kind;
:func:`logrank_group_sequential` is the front door for comparing two
groups' survival with a log-rank test: the events at each look and the
units, from :func:`~surpyval.test_planning.logrank_events` and
:func:`~surpyval.test_planning.logrank_sample_size`.

Wald's sequential tests (:func:`~surpyval.test_planning.sequential_mtbf_test`
and :func:`~surpyval.test_planning.sequential_demonstration_test`) look
after every failure or unit and accept or reject one target against
another; a group-sequential design looks a few planned times, at a
normal test statistic, which suits a comparison analysed in batches
(a log-rank or Cox test at interim analyses).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize, special, stats

from surpyval.test_planning._checks import (
    check_count,
    check_flag,
    check_positive,
    check_probability,
)
from surpyval.test_planning._serialise import result_dict
from surpyval.test_planning.comparison import (
    _check_hazard_ratio,
    _check_ratio,
    _event_probabilities,
    _events,
)
from surpyval.utils.validation import check_alpha_ci, check_option

#: The boundaries of a fixed shape, with a constant chosen to give the
#: level: ``b_k = C t_k ** (delta - 1/2)``, and Haybittle-Peto's.
CLASSICAL = ("obrien-fleming", "pocock", "wang-tsiatis", "haybittle-peto")
#: The spending functions (Lan and DeMets 1983), for the efficacy
#: boundary (alpha spending) and the futility boundary (beta spending).
SPENDING = (
    "lan-demets-obrien-fleming",
    "lan-demets-pocock",
    "hwang-shih-decani",
    "kim-demets",
)
EFFICACY = CLASSICAL + SPENDING
ALTERNATIVES = ("two-sided", "one-sided")

#: The parameter a family takes, and its default (None: required).
_PARAMETERS: dict[str, tuple[str, float | None]] = {
    "wang-tsiatis": ("delta", None),
    "haybittle-peto": ("interim z", 3.0),
    "hwang-shih-decani": ("gamma", -4.0),
    "kim-demets": ("rho", 3.0),
}

_NAMES = {
    "obrien-fleming": "O'Brien-Fleming",
    "pocock": "Pocock",
    "wang-tsiatis": "Wang-Tsiatis",
    "haybittle-peto": "Haybittle-Peto",
    "lan-demets-obrien-fleming": "O'Brien-Fleming-type spending",
    "lan-demets-pocock": "Pocock-type spending",
    "hwang-shih-decani": "Hwang-Shih-DeCani spending",
    "kim-demets": "Kim-DeMets (power) spending",
}

#: The grid size of the numerical integration (Jennison and Turnbull's
#: ``r``): ``12 r - 3`` points per look.
GRID = 32
#: The largest boundary searched for; a look that spends less than the
#: chance of a standard normal beyond it gets an infinite boundary.
_Z_MAX = 37.0
_TOL = 1e-13


# --- the numerical integration -------------------------------------------


def _grid(
    mean: float, lower: float, upper: float, r: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """Jennison and Turnbull's (2000, section 19.2) grid on the
    continuation region ``(lower, upper)`` of a look whose statistic has
    mean ``mean``, and its Simpson weights."""
    i = np.arange(1, 6 * r, dtype=float)
    x = np.where(
        i < r,
        mean - 3.0 - 4.0 * np.log(r / i),
        np.where(
            i <= 5 * r,
            mean - 3.0 + 3.0 * (i - r) / (2.0 * r),
            mean + 3.0 + 4.0 * np.log(r / np.maximum(6 * r - i, 1.0)),
        ),
    )
    inside = x[(x > lower) & (x < upper)]
    nodes = [inside]
    if lower > x[0]:
        nodes.insert(0, np.array([lower]))
    if upper < x[-1]:
        nodes.append(np.array([upper]))
    x = np.concatenate(nodes)
    if x.size < 2 or not lower < upper:
        return np.empty(0), np.empty(0)
    z = np.empty(2 * x.size - 1)
    z[0::2] = x
    z[1::2] = 0.5 * (x[:-1] + x[1:])
    d = np.diff(x) / 6.0
    w = np.zeros_like(z)
    w[1::2] = 4.0 * d
    w[0:-1:2] += d
    w[2::2] += d
    return z, w


class _State:
    """The (sub-)density of the statistic on the continuation region of
    the last look, times the Simpson weights: ``None`` before the first
    look."""

    def __init__(self, drift: float, r: int = GRID) -> None:
        self.drift = drift
        self.r = r
        self.t = 0.0
        self.z: npt.NDArray | None = None
        self.wh: npt.NDArray | None = None

    def _args(self, t: float, bound: float) -> npt.NDArray | float:
        """The standardised bound of the increment from each grid point
        (or from 0 before the first look)."""
        if self.z is None:
            return bound - self.drift * np.sqrt(t)
        delta = t - self.t
        return (
            bound * np.sqrt(t) - self.z * np.sqrt(self.t) - self.drift * delta
        ) / np.sqrt(delta)

    def above(self, t: float, bound: float) -> float:
        """The chance of reaching the look at ``t`` and being at or above
        ``bound``."""
        if bound == np.inf:
            return 0.0
        p = special.ndtr(-np.asarray(self._args(t, bound)))
        return float(p if self.wh is None else self.wh @ p)

    def below(self, t: float, bound: float) -> float:
        """The chance of reaching the look at ``t`` and being at or below
        ``bound``."""
        if bound == -np.inf:
            return 0.0
        p = special.ndtr(np.asarray(self._args(t, bound)))
        return float(p if self.wh is None else self.wh @ p)

    def advance(self, t: float, lower: float, upper: float) -> None:
        """Move to the look at ``t``, keeping the paths that continue past
        it (between ``lower`` and ``upper``)."""
        z, w = _grid(self.drift * np.sqrt(t), lower, upper, self.r)
        if self.z is None:
            h = stats.norm.pdf(z - self.drift * np.sqrt(t))
        elif z.size == 0 or self.z.size == 0:
            z, w, h = np.empty(0), np.empty(0), np.empty(0)
        else:
            delta = t - self.t
            u = (
                z[:, None] * np.sqrt(t)
                - self.z[None, :] * np.sqrt(self.t)
                - self.drift * delta
            ) / np.sqrt(delta)
            h = np.sqrt(t / delta) * (stats.norm.pdf(u) @ self.wh)
        self.z, self.wh, self.t = z, w * h, t


def boundary_crossing(
    timing: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    drift: float = 0.0,
    *,
    r: int = GRID,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""The chances of first crossing each look's boundaries.

    For looks at information fractions ``timing`` (``t_k = I_k /
    I_max``), a statistic ``Z_k`` with mean ``drift * sqrt(t_k)``, unit
    variance and ``corr(Z_j, Z_k) = sqrt(t_j / t_k)``: the chance of
    stopping at look ``k`` above ``upper[k]`` (or below ``lower[k]``),
    having stayed between the boundaries at every look before. Computed
    by recursive numerical integration (Jennison and Turnbull 2000,
    chapter 19), not by simulation.

    Parameters
    ----------
    timing : array-like
        The information fractions of the looks, increasing, in (0, 1]
        (the last may differ from 1 when used to evaluate a design whose
        information ran over or short).
    lower, upper : array-like
        The boundaries at each look; ``-inf`` or ``inf`` for none.
    drift : float, optional
        ``theta sqrt(I_max)``, the mean of the statistic at full
        information; 0 (the default) is the null hypothesis.
    r : int, optional
        Keyword only: the grid size, ``12 r - 3`` points per look, by
        default 32.

    Returns
    -------
    (upper, lower) : tuple of arrays
        The chance of first crossing the upper boundary, and the lower,
        at each look.

    Examples
    --------
    Pocock's boundary for two looks at the two-sided 5% level is 2.178
    (Jennison and Turnbull 2000, table 2.1):

    >>> import numpy as np
    >>> from surpyval.test_planning import boundary_crossing
    >>> up, low = boundary_crossing([0.5, 1.0], [-2.178] * 2, [2.178] * 2)
    >>> round(float(up.sum() + low.sum()), 4)
    0.05
    """
    timing = _check_timing(timing)
    lower = np.broadcast_to(np.asarray(lower, dtype=float), timing.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), timing.shape)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("lower and upper must not be NaN.")
    r = check_count(r, "r", minimum=4)
    state = _State(float(drift), r)
    up = np.zeros(timing.size)
    low = np.zeros(timing.size)
    for k, t in enumerate(timing):
        up[k] = state.above(t, upper[k])
        low[k] = state.below(t, lower[k])
        if k < timing.size - 1:
            state.advance(t, lower[k], upper[k])
    return up, low


# --- spending functions and boundaries -----------------------------------


def _spent(name: str, total: float, t: float, param: float | None) -> float:
    """The cumulative probability ``name`` spends by information fraction
    ``t`` (all of ``total`` from ``t = 1`` on)."""
    if t >= 1.0:
        return total
    if t <= 0.0:
        return 0.0
    if name == "lan-demets-obrien-fleming":
        z = stats.norm.isf(total / 2.0)
        return float(2.0 * stats.norm.sf(z / np.sqrt(t)))
    if name == "lan-demets-pocock":
        return float(total * np.log1p((np.e - 1.0) * t))
    assert param is not None
    if name == "hwang-shih-decani":
        if param == 0.0:
            return total * t
        return float(total * np.expm1(-param * t) / np.expm1(-param))
    return float(total * t**param)  # kim-demets


def _check_timing(timing: Any) -> npt.NDArray:
    t = np.asarray(timing, dtype=float).ravel()
    if (
        t.size == 0
        or not np.all(np.isfinite(t))
        or np.any(t <= 0.0)
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError(
            "the information fractions must be positive and increasing, "
            f"got {timing!r}."
        )
    return t


def _check_looks(looks: Any) -> npt.NDArray:
    """The information fractions of the looks: ``looks`` equally spaced
    looks, or the fractions given, ending at 1."""
    if np.ndim(looks) == 0:
        k = check_count(looks, "looks")
        return np.arange(1, k + 1) / k
    t = _check_timing(looks)
    if np.any(t > 1.0) or not np.isclose(t[-1], 1.0, rtol=0, atol=1e-12):
        raise ValueError(
            "looks must be the number of equally spaced looks, or the "
            "information fractions of the looks, increasing and ending at "
            f"1, got {looks!r}."
        )
    t[-1] = 1.0
    return t


def _check_family(
    name: Any, param: Any, argument: str, accepted: tuple[str, ...]
) -> float | None:
    """The family's parameter, checked, or its default."""
    note = None
    if argument == "futility" and name in CLASSICAL:
        note = (
            "The futility boundary is set by beta spending: use one of "
            "the spending functions."
        )
    check_option(argument, name, accepted, note)
    what, default = _PARAMETERS.get(name, ("", None))
    if not what:
        if param is not None:
            raise ValueError(
                f"{argument}={name!r} takes no parameter, got "
                f"{argument}_parameter={param!r}."
            )
        return None
    if param is None:
        if default is None:
            raise ValueError(
                f"{argument}={name!r} needs {argument}_parameter, its "
                f"{what}: 0 gives O'Brien-Fleming's shape, 0.5 Pocock's."
            )
        return default
    try:
        value = float(param)
    except (TypeError, ValueError):
        value = np.nan
    if not np.isfinite(value):
        raise ValueError(
            f"{argument}_parameter must be a finite number, got {param!r}."
        )
    if name == "kim-demets" and value <= 0.0:
        raise ValueError(
            f"{argument}_parameter, Kim-DeMets' rho, must be positive, got "
            f"{param!r}."
        )
    if name == "haybittle-peto" and value <= 0.0:
        raise ValueError(
            f"{argument}_parameter, Haybittle-Peto's interim z, must be "
            f"positive, got {param!r}."
        )
    return value


def _solve_bound(
    spend: float, crossing: Any, lowest: float, what: str
) -> float:
    """The bound at which ``crossing(bound)`` (falling in the bound) is
    ``spend``; ``inf`` when ``spend`` is too small to place one."""
    if spend <= stats.norm.sf(_Z_MAX):
        return np.inf
    if crossing(lowest) < spend:
        raise ValueError(
            f"The {what} boundary cannot spend {spend:.3g} at this look: "
            "less probability is left. Space the looks differently, or "
            "spend less early (a more conservative spending function)."
        )
    return float(
        optimize.brentq(
            lambda b: crossing(b) - spend,
            lowest,
            _Z_MAX,
            xtol=_TOL,
            rtol=4 * np.finfo(float).eps,
        )
    )


class _Spec:
    """What a design is asked to be: its settings, checked."""

    def __init__(
        self,
        timing: npt.NDArray,
        alpha: float,
        beta: float,
        sided: int,
        efficacy: str,
        efficacy_parameter: float | None,
        futility: str | None,
        futility_parameter: float | None,
        binding: bool,
    ) -> None:
        self.timing = timing
        self.alpha = alpha
        self.beta = beta
        self.sided = sided
        self.efficacy = efficacy
        self.efficacy_parameter = efficacy_parameter
        self.futility = futility
        self.futility_parameter = futility_parameter
        self.binding = binding
        #: The information fraction from which all is spent: the last
        #: look's (1 as planned; another after an update).
        self.last = 1.0

    def alpha_spent(self, t: float) -> float:
        """The level spent by ``t``: a two-sided design spends half the
        level on each side with the one-sided function, as gsDesign and
        rpact do (only O'Brien-Fleming-type spending tells the two
        apart)."""
        if t >= self.last:
            return self.alpha
        return self.sided * _spent(
            self.efficacy,
            self.alpha / self.sided,
            t,
            self.efficacy_parameter,
        )

    def beta_spent(self, t: float) -> float:
        assert self.futility is not None
        if t >= self.last:
            return self.beta
        return _spent(self.futility, self.beta, t, self.futility_parameter)


def _classical_upper(spec: _Spec) -> npt.NDArray:
    """The efficacy boundary of a classical design: the constant that
    gives the level, by root finding on the whole design."""
    t = spec.timing

    def level(upper: npt.NDArray) -> float:
        lower = -upper if spec.sided == 2 else np.full(t.size, -np.inf)
        up, low = boundary_crossing(t, lower, upper)
        return float(up.sum() + (low.sum() if spec.sided == 2 else 0.0))

    if spec.efficacy == "haybittle-peto":
        assert spec.efficacy_parameter is not None
        interim = np.full(t.size, spec.efficacy_parameter)

        def final(c: float) -> float:
            interim[-1] = c
            return level(interim) - spec.alpha

        if t.size > 1:
            interim[-1] = np.inf
            if level(interim) >= spec.alpha:
                raise ValueError(
                    "Haybittle-Peto's interim boundary spends the whole "
                    "level before the last look: raise efficacy_parameter "
                    f"(now {spec.efficacy_parameter:g})."
                )
        interim[-1] = optimize.brentq(final, 0.0, _Z_MAX, xtol=_TOL)
        return interim
    if spec.efficacy == "obrien-fleming":
        shape = 1.0 / np.sqrt(t)
    elif spec.efficacy == "pocock":
        shape = np.ones(t.size)
    else:
        assert spec.efficacy_parameter is not None
        shape = t ** (spec.efficacy_parameter - 0.5)
    c = optimize.brentq(
        lambda c: level(c * shape) - spec.alpha,
        1e-3 / shape.max(),
        _Z_MAX / shape.max(),
        xtol=_TOL,
    )
    return c * shape


def _build(
    spec: _Spec, drift: float, classical: npt.NDArray | None
) -> tuple[npt.NDArray, npt.NDArray | None, float]:
    """The boundaries, look by look: the efficacy boundary spending
    alpha under the null (or the classical one given), the futility
    boundary spending beta at ``drift``. Returns them and, for a design
    with futility, the beta the last look would spend beyond what is
    left, its futility boundary at its efficacy one (0 at the design's
    drift)."""
    t = spec.timing
    k_max = t.size
    upper = np.empty(k_max)
    lower = np.full(k_max, -np.inf) if spec.futility else None
    null = _State(0.0)
    alt = _State(drift) if spec.futility else None
    gap = 0.0
    for k in range(k_max):
        if classical is not None:
            upper[k] = classical[k]
        else:
            spend = spec.alpha_spent(t[k]) - spec.alpha_spent(
                t[k - 1] if k else 0.0
            )
            if spec.sided == 2:

                def crossing(b: float, k: int = k) -> float:
                    return null.above(t[k], b) + null.below(t[k], -b)

                lowest = 0.0
            else:

                def crossing(b: float, k: int = k) -> float:
                    return null.above(t[k], b)

                lowest = -_Z_MAX
            upper[k] = _solve_bound(spend, crossing, lowest, "efficacy")
        low_k = -upper[k] if spec.sided == 2 else -np.inf
        if alt is not None and lower is not None:
            spend = spec.beta_spent(t[k]) - spec.beta_spent(
                t[k - 1] if k else 0.0
            )
            if k == k_max - 1:
                # How much more than the beta left the last look would
                # spend with its futility boundary at its efficacy one:
                # 0 at the design's drift, falling as the drift grows.
                gap = alt.below(t[k], upper[k]) - spend
                low_k = upper[k]
            elif spend <= stats.norm.sf(_Z_MAX):
                low_k = -np.inf
            else:
                highest = upper[k] if np.isfinite(upper[k]) else _Z_MAX
                if alt.below(t[k], highest) <= spend:
                    low_k = highest
                else:
                    low_k = float(
                        optimize.brentq(
                            lambda a, k=k: alt.below(float(t[k]), a) - spend,
                            -_Z_MAX,
                            highest,
                            xtol=_TOL,
                        )
                    )
            lower[k] = low_k
        if k < k_max - 1:
            if spec.binding or spec.sided == 2:
                null.advance(t[k], low_k, upper[k])
            else:
                null.advance(t[k], -np.inf, upper[k])
            if alt is not None:
                alt.advance(t[k], low_k, upper[k])
    return upper, lower, gap


def _fixed_drift(spec: _Spec) -> float:
    """The drift a single look needs: ``z_alpha + z_beta``."""
    return float(
        stats.norm.isf(spec.alpha / spec.sided) + stats.norm.isf(spec.beta)
    )


def _design(
    spec: _Spec,
) -> tuple[npt.NDArray, npt.NDArray | None, float]:
    """The boundaries and the drift at which the design has its power."""
    classical = _classical_upper(spec) if spec.efficacy in CLASSICAL else None
    fixed = _fixed_drift(spec)
    if spec.futility is None:
        upper, _, _ = _build(spec, 0.0, classical)
        sides = -upper if spec.sided == 2 else np.full(upper.size, -np.inf)

        def shortfall(drift: float) -> float:
            up, _ = boundary_crossing(spec.timing, sides, upper, drift)
            return float(up.sum()) - (1.0 - spec.beta)

        drift = optimize.brentq(shortfall, 0.5 * fixed, 3.0 * fixed + 3.0)
        return upper, None, float(drift)

    def gap(drift: float) -> float:
        try:
            return _build(spec, drift, classical)[2]
        except ValueError:
            # The futility boundary (higher, the larger the drift) left
            # too little for the efficacy boundary: the drift is too
            # large.
            return -1.0

    drift = optimize.brentq(
        gap,
        fixed * (1.0 - 1e-6),
        3.0 * fixed + 3.0,
        xtol=1e-12,
    )
    upper, lower, _ = _build(spec, drift, classical)
    return upper, lower, float(drift)


# --- the design ----------------------------------------------------------


class GroupSequentialDesign:
    """A group-sequential design, from :func:`group_sequential_design`.

    The test statistic is a standard normal ``Z`` under the null
    hypothesis, oriented so that large values favour the difference
    sought. At look ``k`` (information fraction ``timing[k]``) the
    design stops for efficacy if ``Z >= efficacy_z[k]`` (or, two-sided,
    ``|Z| >= efficacy_z[k]``), for futility if ``Z <= futility_z[k]``,
    and otherwise goes on. :meth:`decide` says which.

    Attributes
    ----------
    timing : ndarray
        The information fractions of the looks, ``I_k / I_max``.
    efficacy_z : ndarray
        The efficacy boundary at each look (``inf`` where a look cannot
        stop for efficacy).
    efficacy_p : ndarray
        The nominal p-value of the efficacy boundary: the significance
        level a look's test is held to (two-sided for a two-sided
        design).
    futility_z : ndarray or None
        The futility boundary at each look, or None without one.
    alpha_spent : ndarray
        The cumulative chance of stopping for efficacy under the null
        hypothesis, by each look (without regard to a non-binding
        futility boundary): the level spent.
    beta_spent : ndarray or None
        The cumulative chance of stopping for futility when the
        difference is there, by each look: the type II error spent.
    drift : float
        The mean of ``Z`` at full information for which the design has
        its ``power``: ``theta sqrt(I_max)``.
    inflation_factor : float
        The maximum information as a multiple of a single analysis's
        with the same level and power.
    power : float
        The chance of stopping for efficacy (in the direction sought)
        at ``drift``.
    stop_null, stop_alternative : dict
        The chance of stopping at each look under the null hypothesis
        and at ``drift``: ``{"efficacy": [...], "futility": [...]}``
        (futility includes the last look's not stopping for efficacy).
    expected_information : dict
        The expected information at the stop, as a fraction of the
        maximum: ``{"null": ..., "alternative": ...}``.

    Examples
    --------
    >>> from surpyval.test_planning import group_sequential_design
    >>> design = group_sequential_design(3, alternative="one-sided",
    ...                                  alpha_ci=0.025, power=0.9)
    >>> design.efficacy_z.round(3).tolist()
    [3.71, 2.511, 1.993]
    >>> round(design.inflation_factor, 4)
    1.0119
    >>> design.decide(2, 2.7)
    'efficacy'
    """

    def __init__(
        self,
        spec: _Spec,
        efficacy_z: npt.NDArray,
        futility_z: npt.NDArray | None,
        drift: float,
    ) -> None:
        self._spec = spec
        self.timing = spec.timing
        self.alpha_ci = spec.alpha
        self.alternative = "two-sided" if spec.sided == 2 else "one-sided"
        self.efficacy = spec.efficacy
        self.efficacy_parameter = spec.efficacy_parameter
        self.futility = spec.futility
        self.futility_parameter = spec.futility_parameter
        self.binding = spec.binding
        self.efficacy_z = efficacy_z
        self.futility_z = futility_z
        self.drift = drift
        self._characterise()

    # -- characteristics --------------------------------------------------
    def _bounds(self) -> tuple[npt.NDArray, npt.NDArray]:
        """The lower and upper boundary as the design is run."""
        if self._spec.sided == 2:
            return -self.efficacy_z, self.efficacy_z
        if self.futility_z is None:
            return np.full(self.timing.size, -np.inf), self.efficacy_z
        return self.futility_z, self.efficacy_z

    def _characterise(self) -> None:
        spec = self._spec
        t = self.timing
        self.efficacy_p = np.minimum(
            spec.sided * stats.norm.sf(self.efficacy_z), 1.0
        )
        lower, upper = self._bounds()
        if spec.sided == 2:
            up0, low0 = boundary_crossing(t, lower, upper)
            self.alpha_spent = np.cumsum(up0 + low0)
        elif spec.futility is not None and not spec.binding:
            up0, _ = boundary_crossing(t, np.full(t.size, -np.inf), upper)
            self.alpha_spent = np.cumsum(up0)
        else:
            up0, _ = boundary_crossing(t, lower, upper)
            self.alpha_spent = np.cumsum(up0)
        fixed = _fixed_drift(spec)
        self.inflation_factor = float((self.drift / fixed) ** 2)
        self.stop_null = self._stops(0.0)
        self.stop_alternative = self._stops(self.drift)
        up1, low1 = boundary_crossing(t, lower, upper, self.drift)
        self.power = float(up1.sum())
        self.beta_spent = None if spec.futility is None else np.cumsum(low1)
        self.expected_information = {
            "null": self._expected(self.stop_null),
            "alternative": self._expected(self.stop_alternative),
        }

    def _stops(self, drift: float) -> dict[str, npt.NDArray]:
        lower, upper = self._bounds()
        up, low = boundary_crossing(self.timing, lower, upper, drift)
        if self._spec.sided == 2:
            efficacy = up + low
            futility = np.zeros(up.size)
        else:
            efficacy, futility = up, low
        futility[-1] = 1.0 - efficacy.sum() - futility[:-1].sum()
        return {"efficacy": efficacy, "futility": futility}

    def _expected(self, stops: dict[str, npt.NDArray]) -> float:
        return float(
            np.sum(self.timing * (stops["efficacy"] + stops["futility"]))
        )

    @property
    def looks(self) -> int:
        """The number of looks."""
        return int(self.timing.size)

    def crossing_probabilities(self, drift: float) -> dict[str, npt.NDArray]:
        """The chance of stopping at each look, for efficacy and for
        futility, when the statistic's mean at full information is
        ``drift`` (``theta sqrt(I_max)``; 0 is the null hypothesis,
        :attr:`drift` the alternative the design is powered for).

        For a two-sided design ``"efficacy"`` counts either side and
        ``"futility"`` is the last look's not crossing.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(2)
        >>> stops = design.crossing_probabilities(design.drift)
        >>> round(float(stops["efficacy"].sum()), 3)
        0.8
        """
        drift = float(drift)
        if not np.isfinite(drift):
            raise ValueError(f"drift must be finite, got {drift!r}.")
        return self._stops(drift)

    def max_information(self, effect: float) -> float:
        """The information the last look needs to detect a difference
        ``effect`` on the scale of the statistic (``Z = estimate *
        sqrt(information)``): ``(drift / effect) ** 2``.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(3)
        >>> round(design.max_information(0.5), 2)
        31.8
        """
        effect = float(effect)
        if not (np.isfinite(effect) and effect != 0.0):
            raise ValueError(
                f"effect must be a finite number other than 0, got {effect!r}."
            )
        return float((self.drift / effect) ** 2)

    # -- using it ---------------------------------------------------------
    def decide(self, look: int, z: float) -> str:
        """What the design says at look ``look`` (counted from 1) with
        statistic ``z``: ``"efficacy"`` (stop: the difference is shown),
        ``"futility"`` (stop: it will not be; at the last look, it was
        not) or ``"continue"``.

        A non-binding futility boundary may be ignored; the level holds
        either way.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(3)
        >>> design.decide(1, 2.5), design.decide(3, -2.1)
        ('continue', 'efficacy')
        """
        look = check_count(look, "look")
        if look > self.looks:
            raise ValueError(
                f"look must be at most {self.looks}, the design's looks, "
                f"got {look!r}."
            )
        z = float(z)
        if np.isnan(z):
            raise ValueError("z must be a number, got nan.")
        k = look - 1
        bound = self.efficacy_z[k]
        statistic = abs(z) if self._spec.sided == 2 else z
        if statistic >= bound:
            return "efficacy"
        if k == self.looks - 1:
            return "futility"
        if self.futility_z is not None and z <= self.futility_z[k]:
            return "futility"
        return "continue"

    def _look_and_z(self, look: Any, z: Any) -> tuple[int, float]:
        look = check_count(look, "look")
        if look > self.looks:
            raise ValueError(
                f"look must be at most {self.looks}, the design's looks, "
                f"got {look!r}."
            )
        z = float(z)
        if not np.isfinite(z):
            raise ValueError(f"z must be a finite number, got {z!r}.")
        return look - 1, z

    def _exceedance(self, k: int, z: float, drift: float) -> float:
        """The chance, at ``drift``, of a result at least as extreme as
        stopping at look ``k`` with ``z`` in the stage-wise ordering
        (an earlier stop for efficacy is more extreme; at the same look a
        larger ``z``): the upper side, with the lower boundary that binds
        (the opposite efficacy boundary of a two-sided design, a binding
        futility boundary) kept and a non-binding one ignored."""
        t = self.timing[: k + 1]
        upper = self.efficacy_z[: k + 1].copy()
        if self._spec.sided == 2:
            lower = -self.efficacy_z[: k + 1].copy()
        elif self.futility_z is not None and self.binding:
            lower = self.futility_z[: k + 1].copy()
        else:
            lower = np.full(k + 1, -np.inf)
        upper[k] = z
        lower[k] = -np.inf
        up, _ = boundary_crossing(t, lower, upper, drift)
        return float(up.sum())

    def _check_stopped(self, k: int, z: float) -> float:
        """``z`` oriented to the upper side, refusing a look the study
        could not have reached (an earlier look's boundary is not
        crossed by it, but stopping at an interim look needs the
        boundary crossed)."""
        sign = 1.0
        if self._spec.sided == 2 and z < 0.0:
            sign = -1.0
        if k < self.looks - 1 and sign * z < self.efficacy_z[k]:
            raise ValueError(
                f"At look {k + 1} the study stops only for efficacy, at "
                f"|z| >= {self.efficacy_z[k]:.4g} (or for futility, which "
                f"gives no estimate here); got z = {z:g}. The ordering "
                "needs the look the study stopped at: use "
                "repeated_interval at a look it went on from."
            )
        return sign

    def p_value(self, look: int, z: float) -> float:
        """The p-value of the study, stopped at look ``look`` with
        statistic ``z``, in the stage-wise ordering (Armitage, Fairbanks
        and Jennison-Turnbull): the chance under the null hypothesis of
        stopping for efficacy at an earlier look, or at this look with a
        statistic at least as large.

        It agrees with the boundaries: it is at most ``alpha_ci`` exactly
        when the study stopped for efficacy. One-sided for a one-sided
        design (a non-binding futility boundary ignored, as it may be),
        two-sided (both boundaries) for a two-sided design. Use it at
        the look the study stopped at, or the last look.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(3, alternative="one-sided",
        ...                                  alpha_ci=0.025)
        >>> round(design.p_value(2, 2.6), 5)
        0.0047
        """
        k, z = self._look_and_z(look, z)
        if k < self.looks - 1:
            self._check_stopped(k, z)
        if self._spec.sided == 2:
            return min(1.0, 2.0 * self._exceedance(k, abs(z), 0.0))
        return self._exceedance(k, z, 0.0)

    def _solve_drift(self, k: int, z: float, target: float) -> float:
        """The drift at which the stage-wise exceedance is ``target``."""

        def gap(drift: float) -> float:
            return self._exceedance(k, z, drift) - target

        centre = z / np.sqrt(self.timing[k])
        lo, hi = centre - 10.0, centre + 10.0
        return float(optimize.brentq(gap, lo, hi, xtol=1e-10))

    def estimate(
        self, look: int, z: float, alpha_ci: float | None = None
    ) -> dict[str, float]:
        """The median-unbiased estimate of the drift and its confidence
        interval, for a study stopped at look ``look`` with statistic
        ``z``, in the stage-wise ordering.

        Stopping early because the difference looked large makes the
        naive estimate (``z / sqrt(t)``) too large on average; the
        median-unbiased estimate is the drift at which the result seen
        is the median result, and the interval the drifts at which it is
        the ``alpha_ci / 2`` and ``1 - alpha_ci / 2`` quantiles. On the
        drift scale (``theta sqrt(I_max)``, the mean of the statistic at
        full information): divide by ``sqrt(I_max)`` for ``theta``;
        :meth:`LogrankGroupSequentialDesign.hazard_ratio_estimate` gives
        the hazard ratio.

        Parameters
        ----------
        look : int
            The look the study stopped at (an interim look only if its
            efficacy boundary was crossed), or the last.
        z : float
            The statistic there.
        alpha_ci : float, optional
            The interval's significance level (two-sided): by default the
            design's for a two-sided design, twice it for a one-sided
            one (a 95% interval for a one-sided 2.5% design).

        Returns
        -------
        dict
            ``{"estimate": ..., "lower": ..., "upper": ...}``.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(3, alternative="one-sided",
        ...                                  alpha_ci=0.025)
        >>> {k: round(v, 3) for k, v in design.estimate(2, 2.6).items()}
        {'estimate': 3.183, 'lower': 0.781, 'upper': 5.584}
        """
        k, z = self._look_and_z(look, z)
        sign = self._check_stopped(k, z) if k < self.looks - 1 else 1.0
        if self._spec.sided == 2 and z < 0.0:
            sign = -1.0
        if alpha_ci is None:
            alpha_ci = self.alpha_ci * (2.0 if self._spec.sided == 1 else 1)
        check_alpha_ci(alpha_ci)
        z = sign * z
        values = {
            "estimate": self._solve_drift(k, z, 0.5),
            "lower": self._solve_drift(k, z, alpha_ci / 2.0),
            "upper": self._solve_drift(k, z, 1.0 - alpha_ci / 2.0),
        }
        if sign < 0.0:
            values = {
                "estimate": -values["estimate"],
                "lower": -values["upper"],
                "upper": -values["lower"],
            }
        return values

    def repeated_interval(self, look: int, z: float) -> tuple[float, float]:
        """The repeated confidence interval for the drift at look
        ``look`` with statistic ``z`` (Jennison and Turnbull 1989):
        ``(z - b_k) / sqrt(t_k)`` to ``(z + b_k) / sqrt(t_k)``, with
        ``b_k`` the efficacy boundary.

        Unlike :meth:`estimate` it needs no stop: the intervals at every
        look together cover the drift with the design's confidence
        (``1 - alpha_ci`` two-sided, ``1 - 2 alpha_ci`` for a one-sided
        design), so it can be read at any look, whatever the study does
        next. It is wider than the stage-wise interval. On the drift
        scale, as :meth:`estimate`.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(3)
        >>> [round(x, 3) for x in design.repeated_interval(1, 2.0)]
        [-2.962, 9.891]
        """
        k, z = self._look_and_z(look, z)
        half = self.efficacy_z[k]
        root = np.sqrt(self.timing[k])
        return float((z - half) / root), float((z + half) / root)

    def update(self, timing: npt.ArrayLike) -> GroupSequentialDesign:
        """The design re-computed for the information fractions the looks
        actually came at.

        Information rarely arrives exactly as planned. With a spending
        function the boundaries can be re-computed at the fractions
        observed: give every look, those done and those still planned,
        as fractions of the *planned* maximum information. The last is
        the final look and spends what is left of the level, wherever it
        falls (above 1 if the study ran over). The levels spent at the
        looks already done are those of their fractions, so the earlier
        boundaries stay as they were used if their fractions are kept.
        A futility boundary is re-computed at the planned drift, and the
        last look's futility boundary is its efficacy boundary; the power
        changes, and is reported.

        Classical boundaries (O'Brien-Fleming, Pocock, Wang-Tsiatis,
        Haybittle-Peto) hold their level only at the planned looks, so
        they are not re-computed: plan with a spending function to keep
        this freedom.

        Examples
        --------
        >>> from surpyval.test_planning import group_sequential_design
        >>> design = group_sequential_design(3, alternative="one-sided",
        ...                                  alpha_ci=0.025)
        >>> actual = design.update([0.4, 0.7, 1.0])
        >>> actual.efficacy_z.round(3).tolist()
        [3.357, 2.445, 2.0]
        """
        if self.efficacy in CLASSICAL:
            raise ValueError(
                f"A {_NAMES[self.efficacy]} boundary holds its level only "
                "at the looks it was planned for; it cannot be re-computed "
                "at others. Plan with a spending function (efficacy="
                "'lan-demets-obrien-fleming', say) to change the looks."
            )
        t = _check_timing(timing)
        spec = _Spec(
            t,
            self._spec.alpha,
            self._spec.beta,
            self._spec.sided,
            self.efficacy,
            self.efficacy_parameter,
            self.futility,
            self.futility_parameter,
            self.binding,
        )
        # The last look spends all that is left, wherever it falls.
        spec.last = float(t[-1])
        upper, lower, _ = _build(spec, self.drift, None)
        return type(self)._updated(self, spec, upper, lower)

    @classmethod
    def _updated(
        cls,
        planned: GroupSequentialDesign,
        spec: _Spec,
        upper: npt.NDArray,
        lower: npt.NDArray | None,
    ) -> GroupSequentialDesign:
        out = cls.__new__(cls)
        GroupSequentialDesign.__init__(out, spec, upper, lower, planned.drift)
        return out

    # -- serialisation ----------------------------------------------------
    def _values(self) -> dict[str, Any]:
        return {
            "timing": self.timing,
            "alpha_ci": self.alpha_ci,
            "beta": self._spec.beta,
            "alternative": self.alternative,
            "efficacy": self.efficacy,
            "efficacy_parameter": self.efficacy_parameter,
            "futility": self.futility,
            "futility_parameter": self.futility_parameter,
            "binding": self.binding,
            "efficacy_z": self.efficacy_z,
            "efficacy_p": self.efficacy_p,
            "futility_z": self.futility_z,
            "alpha_spent": self.alpha_spent,
            "beta_spent": self.beta_spent,
            "drift": self.drift,
            "inflation_factor": self.inflation_factor,
            "power": self.power,
            "stop_null": self.stop_null,
            "stop_alternative": self.stop_alternative,
            "expected_information": self.expected_information,
        }

    def to_dict(self) -> dict:
        """
        The design as a dict of native Python values, strict JSON.

        Returns
        -------
        dict
            The settings (``timing``, ``alpha_ci``, ``beta``,
            ``alternative``, ``efficacy``, ``efficacy_parameter``,
            ``futility``, ``futility_parameter``, ``binding``) and every
            attribute, under ``"result": "GroupSequentialDesign"``. An
            infinite boundary is ``None``, recorded under
            ``"non_finite"``.

        Examples
        --------
        >>> import json
        >>> from surpyval.test_planning import (
        ...     GroupSequentialDesign, group_sequential_design)
        >>> design = group_sequential_design(3)
        >>> d = design.to_dict()
        >>> d["result"], len(d["efficacy_z"])
        ('GroupSequentialDesign', 3)
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        >>> GroupSequentialDesign.from_dict(d).decide(3, 2.0)
        'efficacy'
        """
        return result_dict(self, self._values())

    @classmethod
    def from_dict(cls, d: dict) -> GroupSequentialDesign:
        """Rebuild a design from :meth:`to_dict`'s dict, without
        re-computing it."""
        from surpyval.serialisation import decode_non_finite

        if not isinstance(d, dict) or d.get("result") != cls.__name__:
            raise ValueError(
                f"Not a {cls.__name__} dict: its 'result' is "
                f"{d.get('result') if isinstance(d, dict) else None!r}. "
                f"Pass the dict {cls.__name__}.to_dict() wrote."
            )
        d = decode_non_finite(d)
        out = cls.__new__(cls)
        out._restore(d)
        return out

    def _restore(self, d: dict) -> None:
        def array(value: Any) -> npt.NDArray | None:
            return None if value is None else np.asarray(value, dtype=float)

        def required(value: Any) -> npt.NDArray:
            return np.asarray(value, dtype=float)

        sided = 2 if d["alternative"] == "two-sided" else 1
        timing = np.asarray(d["timing"], dtype=float)
        self._spec = _Spec(
            timing,
            float(d["alpha_ci"]),
            float(d["beta"]),
            sided,
            d["efficacy"],
            d["efficacy_parameter"],
            d["futility"],
            d["futility_parameter"],
            bool(d["binding"]),
        )
        self._spec.last = float(timing[-1])
        self.timing = timing
        self.alpha_ci = float(d["alpha_ci"])
        self.alternative = d["alternative"]
        self.efficacy = d["efficacy"]
        self.efficacy_parameter = d["efficacy_parameter"]
        self.futility = d["futility"]
        self.futility_parameter = d["futility_parameter"]
        self.binding = bool(d["binding"])
        self.efficacy_z = required(d["efficacy_z"])
        self.efficacy_p = required(d["efficacy_p"])
        self.futility_z = array(d["futility_z"])
        self.alpha_spent = required(d["alpha_spent"])
        self.beta_spent = array(d["beta_spent"])
        self.drift = float(d["drift"])
        self.inflation_factor = float(d["inflation_factor"])
        self.power = float(d["power"])
        self.stop_null = {k: required(v) for k, v in d["stop_null"].items()}
        self.stop_alternative = {
            k: required(v) for k, v in d["stop_alternative"].items()
        }
        self.expected_information = {
            k: float(v) for k, v in d["expected_information"].items()
        }

    # -- reading it -------------------------------------------------------
    def _boundary_text(self) -> str:
        name = _NAMES[self.efficacy]
        if self.efficacy_parameter is not None:
            name += f" ({_PARAMETERS[self.efficacy][0]} "
            name += f"{self.efficacy_parameter:g})"
        return name

    def _header(self) -> list[str]:
        level = f"{100 * self.alpha_ci:.4g}%"
        lines = [
            f"Looks            : {self.looks}, at "
            + ", ".join(f"{100 * t:.4g}%" for t in self.timing)
            + " of the information",
            f"Efficacy         : {self._boundary_text()}, "
            f"{self.alternative}, {level} level",
        ]
        if self.futility is None:
            lines.append("Futility         : none")
        else:
            name = _NAMES[self.futility]
            if self.futility_parameter is not None:
                name += f" ({_PARAMETERS[self.futility][0]} "
                name += f"{self.futility_parameter:g})"
            kind = "binding" if self.binding else "non-binding"
            lines.append(f"Futility         : {name}, {kind}")
        extra = 100 * (self.inflation_factor - 1.0)
        lines += [
            f"Power            : {100 * self.power:.4g}%",
            f"Inflation factor : {self.inflation_factor:.4f} (the most "
            f"information is {extra:.3g}% more than one analysis needs)",
            "Expected stop    : at "
            f"{100 * self.expected_information['null']:.3g}% of the "
            "information with no difference, "
            f"{100 * self.expected_information['alternative']:.3g}% with "
            "the difference",
        ]
        return lines

    def _table(self, first: str, column: list[str]) -> list[str]:
        head = (
            f"{'Look':>4}  {first:>11}  {'Efficacy z':>10}  {'Nominal p':>9}"
        )
        if self.futility_z is not None:
            head += f"  {'Futility z':>10}"
        head += f"  {'Alpha spent':>11}"
        rows = [head]
        for k in range(self.looks):
            row = (
                f"{k + 1:>4}  {column[k]:>11}  {self.efficacy_z[k]:>10.4g}"
                f"  {self.efficacy_p[k]:>9.4g}"
            )
            if self.futility_z is not None:
                row += f"  {self.futility_z[k]:>10.4g}"
            row += f"  {self.alpha_spent[k]:>11.4g}"
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        header = "Group Sequential Design"
        column = [f"{100 * t:.4g}%" for t in self.timing]
        return "\n".join(
            [header, "=" * len(header)]
            + self._header()
            + [""]
            + self._table("Information", column)
        )


def _settings(
    looks: Any,
    power: float,
    alpha_ci: float,
    alternative: str,
    efficacy: str,
    efficacy_parameter: float | None,
    futility: str | None,
    futility_parameter: float | None,
    binding: bool,
) -> _Spec:
    check_alpha_ci(alpha_ci)
    power = check_probability(power, "power")
    check_option("alternative", alternative, ALTERNATIVES)
    sided = 2 if alternative == "two-sided" else 1
    if power <= alpha_ci / sided:
        raise ValueError(
            f"power must be more than the level, got {power!r}: a test "
            "that rejects at random has that power."
        )
    timing = _check_looks(looks)
    e_param = _check_family(efficacy, efficacy_parameter, "efficacy", EFFICACY)
    binding = check_flag(binding, "binding")
    f_param = None
    if futility is not None:
        f_param = _check_family(
            futility, futility_parameter, "futility", SPENDING
        )
        if sided == 2:
            raise ValueError(
                "A futility boundary is built for a one-sided design: use "
                "alternative='one-sided' (with alpha_ci halved to keep "
                "the same level on the side sought), or futility=None."
            )
        if binding and efficacy in CLASSICAL:
            raise ValueError(
                "A binding futility boundary needs an alpha-spending "
                "efficacy boundary (efficacy='lan-demets-obrien-fleming', "
                "say), or binding=False."
            )
    elif futility_parameter is not None:
        raise ValueError(
            "futility_parameter is given without a futility boundary: "
            "give futility too, or drop futility_parameter."
        )
    elif binding:
        raise ValueError(
            "binding=True needs a futility boundary: give futility too."
        )
    return _Spec(
        timing,
        float(alpha_ci),
        1.0 - power,
        sided,
        efficacy,
        e_param,
        futility,
        f_param,
        binding,
    )


def group_sequential_design(
    looks: int | Sequence[float] = 3,
    power: float = 0.8,
    *,
    alpha_ci: float = 0.05,
    alternative: str = "two-sided",
    efficacy: str = "lan-demets-obrien-fleming",
    efficacy_parameter: float | None = None,
    futility: str | None = None,
    futility_parameter: float | None = None,
    binding: bool = False,
) -> GroupSequentialDesign:
    r"""A group-sequential design for a normal test statistic.

    A study looked at ``looks`` times can stop at a look as soon as the
    difference sought is clear (the efficacy boundary), or as soon as it
    is clearly not coming (the futility boundary). Looking more than
    once gives more chances of a false finding, so each look is held to
    a stricter level than the overall ``alpha_ci``; the design chooses
    those levels so that the overall chance stays ``alpha_ci``, and the
    maximum information so that the power stays ``power``. The price is
    a little more information at most (the inflation factor); the gain
    is a smaller expected amount, because the study often stops early.

    The design is on the scale of information fractions, for any test
    whose statistic is approximately normal with independent increments
    (a log-rank or Cox score test, a difference of means or of
    proportions): :func:`logrank_group_sequential` gives the events of a
    log-rank test, and :meth:`GroupSequentialDesign.max_information`
    the information for a difference on the statistic's scale.

    The efficacy boundary is one of

    * the classical shapes: ``"obrien-fleming"`` (:math:`b_k = C /
      \sqrt{t_k}`, strict early, close to the fixed test's level at the
      end), ``"pocock"`` (the same :math:`C` at every look),
      ``"wang-tsiatis"`` (:math:`b_k = C t_k^{\Delta - 1/2}`, with
      :math:`\Delta` the ``efficacy_parameter``: 0 is O'Brien-Fleming's,
      0.5 Pocock's), ``"haybittle-peto"`` (``efficacy_parameter``, 3 by
      default, at the interim looks, and the last look's set to give
      exactly the level);
    * alpha-spending functions (Lan and DeMets 1983), which spend the
      level :math:`\alpha(t)` by information fraction :math:`t` and so
      allow looks at other times than planned (:meth:`update
      <GroupSequentialDesign.update>`): ``"lan-demets-obrien-fleming"``
      (:math:`2 - 2 \Phi(z_{1 - \alpha / 2} / \sqrt{t})`, the default),
      ``"lan-demets-pocock"`` (:math:`\alpha \ln(1 + (e - 1) t)`),
      ``"hwang-shih-decani"`` (:math:`\alpha (1 - e^{-\gamma t}) / (1 -
      e^{-\gamma})`, :math:`\gamma` the parameter, -4 by default) and
      ``"kim-demets"`` (:math:`\alpha t^\rho`, :math:`\rho` the
      parameter, 3 by default). A two-sided design spends
      :math:`\alpha / 2` on each side with these, as R's ``gsDesign``
      and ``rpact`` do.

    A futility boundary spends the type II error ``1 - power`` with one
    of the spending functions, under the alternative (beta spending);
    the maximum information is then the one at which the two boundaries
    meet at the last look. Binding (the study must stop when it
    crosses) lets the efficacy boundary be a little lower; non-binding
    (the default) keeps the level if the study goes on regardless, and
    is the usual choice. Futility boundaries are for one-sided
    designs.

    Parameters
    ----------
    looks : int or sequence of float, optional
        The number of equally spaced looks, by default 3, or the
        information fraction of each, increasing and ending at 1.
    power : float, optional
        The chance of stopping for efficacy when the difference is
        there, by default 0.8.
    alpha_ci : float, optional
        Keyword only: the significance level, by default 0.05 (for a
        two-sided design, both sides together).
    alternative : str, optional
        Keyword only: ``"two-sided"`` (the default; symmetric
        boundaries) or ``"one-sided"``.
    efficacy : str, optional
        Keyword only: the efficacy boundary, by default
        ``"lan-demets-obrien-fleming"``.
    efficacy_parameter : float, optional
        Keyword only: Wang-Tsiatis' :math:`\Delta` (required),
        Haybittle-Peto's interim z, Hwang-Shih-DeCani's :math:`\gamma`,
        Kim-DeMets' :math:`\rho`.
    futility : str, optional
        Keyword only: a spending function for a futility boundary, or
        None (the default) for none.
    futility_parameter : float, optional
        Keyword only: the futility spending function's parameter.
    binding : bool, optional
        Keyword only: whether the futility boundary binds, by default
        False.

    Returns
    -------
    GroupSequentialDesign

    Raises
    ------
    ValueError
        If an argument is out of range, or a combination is not built
        (futility on a two-sided design; binding futility with a
        classical efficacy boundary).

    Examples
    --------
    Three equally spaced looks at the two-sided 5% level, with
    O'Brien-Fleming-type spending:

    >>> from surpyval.test_planning import group_sequential_design
    >>> design = group_sequential_design(3)
    >>> design.efficacy_z.round(3).tolist()
    [3.712, 2.511, 1.993]

    Pocock's classical boundary for five looks (Jennison and Turnbull
    2000, table 2.1, 2.413):

    >>> pocock = group_sequential_design(5, efficacy="pocock")
    >>> round(float(pocock.efficacy_z[0]), 3)
    2.413
    """
    spec = _settings(
        looks,
        power,
        alpha_ci,
        alternative,
        efficacy,
        efficacy_parameter,
        futility,
        futility_parameter,
        binding,
    )
    upper, lower, drift = _design(spec)
    return GroupSequentialDesign(spec, upper, lower, drift)


# --- the log-rank front door ---------------------------------------------


class LogrankGroupSequentialDesign(GroupSequentialDesign):
    """A group-sequential log-rank comparison, from
    :func:`logrank_group_sequential`: a :class:`GroupSequentialDesign`
    with the events (and, given the chance of an event, the units) it
    needs.

    The statistic is the log-rank (or Cox score) ``Z``, signed so that
    large values favour the hazard ratio sought: ``Z = -log(HR_hat) /
    se`` for a hazard ratio below 1.

    Attributes
    ----------
    hazard_ratio : float
        The hazard ratio to detect, of the first group to the second.
    ratio : float
        The units in the first group per unit in the second.
    method : str
        ``"schoenfeld"`` or ``"freedman"``, as in
        :func:`~surpyval.test_planning.logrank_events`.
    fixed_events : float
        The events one analysis needs, unrounded.
    max_events : int
        The events at the last look: the inflation factor times
        ``fixed_events``, rounded up.
    events : ndarray of int
        The events at each look, ``timing * max_events``, rounded up.
    expected_events : dict
        The expected events at the stop, ``{"null": ...,
        "alternative": ...}``.
    hazard_ratio_bounds : dict
        The hazard ratio observed at each look that is on the efficacy
        (and futility) boundary, approximately (Schoenfeld's variance,
        ``4 / d`` for equal groups): ``{"efficacy": [...], "futility":
        [...]}``.
    n : (int, int) or None
        The units in the first and the second group, rounded up, if the
        chance of an event was given.

    Examples
    --------
    >>> from surpyval.test_planning import logrank_group_sequential
    >>> design = logrank_group_sequential(0.7, 3, 0.9,
    ...                                   alternative="one-sided",
    ...                                   alpha_ci=0.025)
    >>> design.max_events, design.events.tolist()
    (335, [112, 224, 335])
    """

    def _set_events(
        self,
        hazard_ratio: float,
        ratio: float,
        method: str,
        event_probability: tuple[float, float] | None,
        max_events: int | None = None,
        events: npt.NDArray | None = None,
    ) -> None:
        self.hazard_ratio = hazard_ratio
        self.ratio = ratio
        self.method = method
        self.event_probability = event_probability
        fixed = _fixed_drift(self._spec)
        self.fixed_events = float(_events(hazard_ratio, fixed, ratio, method))
        if max_events is None:
            exact = float(_events(hazard_ratio, self.drift, ratio, method))
            max_events = int(np.ceil(exact - 1e-9))
        self.max_events = max_events
        if events is None:
            events = np.ceil(self.timing * max_events - 1e-9).astype(int)
        self.events = events
        self.expected_events = {
            k: v * max_events for k, v in self.expected_information.items()
        }
        share = ratio / (1.0 + ratio)
        sign = -1.0 if hazard_ratio < 1.0 else 1.0
        scale = np.sqrt(share * (1.0 - share) * self.events)
        bounds = {"efficacy": np.exp(sign * self.efficacy_z / scale)}
        if self.futility_z is not None:
            bounds["futility"] = np.exp(sign * self.futility_z / scale)
        self.hazard_ratio_bounds = bounds
        if event_probability is None:
            self.n = None
        else:
            first, second = event_probability
            exact = float(_events(hazard_ratio, self.drift, ratio, method))
            per_second = exact / (ratio * first + second)
            self.n = (
                int(np.ceil(ratio * per_second - 1e-9)),
                int(np.ceil(per_second - 1e-9)),
            )

    def _to_hazard_ratio(self, drift: float) -> float:
        share = self.ratio / (1.0 + self.ratio)
        sign = -1.0 if self.hazard_ratio < 1.0 else 1.0
        scale = np.sqrt(share * (1.0 - share) * self.max_events)
        return float(np.exp(sign * drift / scale))

    def hazard_ratio_estimate(
        self, look: int, z: float, alpha_ci: float | None = None
    ) -> dict[str, Any]:
        """The hazard ratio estimated at look ``look`` with log-rank
        statistic ``z`` (signed so that large values favour the hazard
        ratio sought), allowing for the looks.

        ``"estimate"``, ``"lower"`` and ``"upper"`` are the
        median-unbiased estimate and its confidence interval in the
        stage-wise ordering (:meth:`GroupSequentialDesign.estimate`),
        for the look the study stopped at; ``"repeated"`` is the
        repeated confidence interval
        (:meth:`GroupSequentialDesign.repeated_interval`), valid at any
        look. The information is Schoenfeld's, ``share (1 - share)``
        times the events, with the looks at the design's events (use
        :meth:`update` first if they came elsewhere).

        Examples
        --------
        >>> from surpyval.test_planning import logrank_group_sequential
        >>> design = logrank_group_sequential(0.7, 3, 0.9,
        ...                                   alternative="one-sided",
        ...                                   alpha_ci=0.025)
        >>> est = design.hazard_ratio_estimate(2, 2.6)
        >>> round(est["estimate"], 3), round(est["upper"], 3)
        (0.706, 0.918)
        """
        k, value = self._look_and_z(look, z)
        repeated = [
            self._to_hazard_ratio(x)
            for x in GroupSequentialDesign.repeated_interval(self, look, z)
        ]
        out: dict[str, Any] = {"repeated": tuple(sorted(repeated))}
        if k < self.looks - 1:
            sign = -1.0 if self._spec.sided == 2 and value < 0 else 1.0
            if sign * value < self.efficacy_z[k]:
                return {
                    "estimate": None,
                    "lower": None,
                    "upper": None,
                    **out,
                }
        drifts = self.estimate(look, z, alpha_ci)
        ratios = sorted(
            self._to_hazard_ratio(drifts[key]) for key in ("lower", "upper")
        )
        return {
            "estimate": self._to_hazard_ratio(drifts["estimate"]),
            "lower": ratios[0],
            "upper": ratios[1],
            **out,
        }

    def update(self, events: npt.ArrayLike) -> LogrankGroupSequentialDesign:
        """The design re-computed for the events each look actually came
        at: every look, those done and those still planned, the last the
        final look (see :meth:`GroupSequentialDesign.update`).

        Examples
        --------
        >>> from surpyval.test_planning import logrank_group_sequential
        >>> design = logrank_group_sequential(0.7, 3, 0.9,
        ...                                   alternative="one-sided",
        ...                                   alpha_ci=0.025)
        >>> actual = design.update([130, 240, 337])
        >>> actual.efficacy_z.round(3).tolist()
        [3.414, 2.41, 2.005]
        """
        counts = np.asarray(events, dtype=float).ravel()
        if counts.size == 0 or not np.all(np.isfinite(counts)):
            raise ValueError(
                f"events must be the events at each look, got {events!r}."
            )
        for count in counts:
            check_positive(count, "events")
        out = GroupSequentialDesign.update(self, counts / self.max_events)
        assert isinstance(out, LogrankGroupSequentialDesign)
        out._set_events(
            self.hazard_ratio,
            self.ratio,
            self.method,
            self.event_probability,
            self.max_events,
            (
                counts.astype(int)
                if np.all(counts == np.round(counts))
                else counts
            ),
        )
        return out

    def _values(self) -> dict[str, Any]:
        return {
            **super()._values(),
            "hazard_ratio": self.hazard_ratio,
            "ratio": self.ratio,
            "method": self.method,
            "event_probability": self.event_probability,
            "fixed_events": self.fixed_events,
            "max_events": self.max_events,
            "events": self.events,
            "expected_events": self.expected_events,
            "hazard_ratio_bounds": self.hazard_ratio_bounds,
            "n": self.n,
        }

    def to_dict(self) -> dict:
        """
        The design as a dict of native Python values, strict JSON: the
        settings and attributes of
        :meth:`GroupSequentialDesign.to_dict`, with ``hazard_ratio``,
        ``ratio``, ``method``, ``event_probability``, ``fixed_events``,
        ``max_events``, ``events``, ``expected_events``,
        ``hazard_ratio_bounds`` and ``n``, under ``"result":
        "LogrankGroupSequentialDesign"``.

        Examples
        --------
        >>> from surpyval.test_planning import (
        ...     LogrankGroupSequentialDesign, logrank_group_sequential)
        >>> design = logrank_group_sequential(0.7)
        >>> d = design.to_dict()
        >>> LogrankGroupSequentialDesign.from_dict(d).max_events
        250
        """
        return result_dict(self, self._values())

    def _restore(self, d: dict) -> None:
        super()._restore(d)
        self.hazard_ratio = float(d["hazard_ratio"])
        self.ratio = float(d["ratio"])
        self.method = d["method"]
        p = d["event_probability"]
        self.event_probability = (
            None if p is None else (float(p[0]), float(p[1]))
        )
        self.fixed_events = float(d["fixed_events"])
        self.max_events = int(d["max_events"])
        self.events = np.asarray(d["events"])
        self.expected_events = {
            k: float(v) for k, v in d["expected_events"].items()
        }
        self.hazard_ratio_bounds = {
            k: np.asarray(v, dtype=float)
            for k, v in d["hazard_ratio_bounds"].items()
        }
        n = d["n"]
        self.n = None if n is None else (int(n[0]), int(n[1]))

    def __repr__(self) -> str:
        header = "Group Sequential Log-rank Design"
        lines = [header, "=" * len(header)]
        lines += [
            f"Hazard ratio     : {self.hazard_ratio:.4g} "
            f"(first group to second, {self.ratio:g} : 1)",
        ]
        lines += self._header()
        lines += [
            f"Events           : {self.max_events} at most "
            f"({self.fixed_events:.4g} for one analysis); expected "
            f"{self.expected_events['null']:.4g} with no difference, "
            f"{self.expected_events['alternative']:.4g} with it",
        ]
        if self.n is not None:
            lines.append(f"Units            : {self.n[0]} and {self.n[1]}")
        column = [str(e) for e in self.events]
        table = self._table("Events", column)
        table[0] += f"  {'HR at bound':>11}"
        for k in range(self.looks):
            table[
                k + 1
            ] += f"  {self.hazard_ratio_bounds['efficacy'][k]:>11.4g}"
        return "\n".join(lines + [""] + table)


def logrank_group_sequential(
    hazard_ratio: float,
    looks: int | Sequence[float] = 3,
    power: float = 0.8,
    *,
    alpha_ci: float = 0.05,
    ratio: float = 1.0,
    method: str = "schoenfeld",
    alternative: str = "two-sided",
    efficacy: str = "lan-demets-obrien-fleming",
    efficacy_parameter: float | None = None,
    futility: str | None = None,
    futility_parameter: float | None = None,
    binding: bool = False,
    event_probability: float | tuple[float, float] | None = None,
) -> LogrankGroupSequentialDesign:
    """A group-sequential log-rank comparison of two groups: the
    boundaries at each look and the events (and units) it needs.

    The group-sequential version of
    :func:`~surpyval.test_planning.logrank_events`: a log-rank test of a
    hazard ratio ``hazard_ratio``, looked at ``looks`` times as events
    accrue, stopping early when the difference is clear (or, with a
    futility boundary, when it is clearly not coming). The maximum
    events are those of a single analysis (Schoenfeld's or Freedman's)
    times the design's inflation factor; the events at each look are
    the information fractions of them. With ``event_probability`` (the
    chance a unit has an event during the study, as in
    :func:`~surpyval.test_planning.logrank_sample_size`) it gives the
    units too.

    The boundaries are those of :func:`group_sequential_design`, which
    explains the options; see it for every argument not listed here.

    Parameters
    ----------
    hazard_ratio : float
        The hazard ratio to detect, of the first group to the second:
        positive, and not 1.
    looks : int or sequence of float, optional
        The number of equally spaced looks, by default 3, or the
        fraction of the events at each, ending at 1.
    power : float, optional
        The chance of detecting it, by default 0.8.
    ratio : float, optional
        Keyword only: the units in the first group per unit in the
        second, by default 1.0.
    method : str, optional
        Keyword only: ``"schoenfeld"`` (the default) or ``"freedman"``.
    event_probability : float or (float, float), optional
        Keyword only: the chance that a unit has an event during the
        study, for both groups or each; gives the units.

    Returns
    -------
    LogrankGroupSequentialDesign

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    Detecting a 30% lower hazard with 90% power at the one-sided 2.5%
    level, with three looks and O'Brien-Fleming-type spending, takes at
    most 335 events (331 for one analysis), and on average about 269 if
    the difference is there:

    >>> from surpyval.test_planning import logrank_group_sequential
    >>> design = logrank_group_sequential(0.7, 3, 0.9,
    ...                                   alternative="one-sided",
    ...                                   alpha_ci=0.025)
    >>> design.max_events, round(design.expected_events["alternative"])
    (335, 269)
    """
    hazard_ratio = _check_hazard_ratio(hazard_ratio)
    ratio = _check_ratio(ratio)
    check_option("method", method, ("schoenfeld", "freedman"))
    probabilities = (
        None
        if event_probability is None
        else _event_probabilities(event_probability)
    )
    spec = _settings(
        looks,
        power,
        alpha_ci,
        alternative,
        efficacy,
        efficacy_parameter,
        futility,
        futility_parameter,
        binding,
    )
    upper, lower, drift = _design(spec)
    design = LogrankGroupSequentialDesign(spec, upper, lower, drift)
    design._set_events(hazard_ratio, ratio, method, probabilities)
    return design
