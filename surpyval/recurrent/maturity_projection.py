"""Maturity projection models: the reliability once the fixes are in,
allowing for the failure modes not seen yet (MIL-HDBK-189C sections
6.3.4 to 6.3.6; MIL-HDBK-00189A sections 7.6 to 7.8).

A development test surfaces failure modes; fixes then remove part of each
seen mode's rate (its fix-effectiveness factor, FEF, ``d_i``). What the
test cannot show directly is how much failure rate sits in the modes it
has not surfaced yet. These models answer that by treating the modes'
initial rates as a sample from a population (gamma for rates, beta for a
one-shot item's chances), in the limit of very many modes, as in a
complex system:

- :func:`ampm_projection`, the AMSAA maturity projection model (AMPM;
  Ellner and Wald 1995): fitted to the first occurrence of each B mode
  (the modes that will be fixed, whenever the fix goes in). The number of
  distinct B modes seen by ``t`` is a Poisson process with mean
  :math:`(\\lambda_B / \\beta) \\ln(1 + \\beta t)`, so new modes turn up at
  the rate :math:`h(t) = \\lambda_B / (1 + \\beta t)`, and the intensity
  once the modes seen by ``t`` are fixed is
  :math:`\\rho(t) = \\lambda_A + (1 - \\mu_d) \\lambda_B + \\mu_d h(t)`.
- :func:`ampm_stein_projection`, AMPM-Stein (Ellner and Hall 2006): from
  the failure count of each mode seen. Each mode's rate is estimated by
  Stein's shrinkage of its own count towards the average,
  :math:`\\tilde\\lambda_i = \\theta N_i / T + (1 - \\theta) \\bar\\lambda`,
  which over many modes gives
  :math:`\\rho_S = \\sum_{\\mathrm{seen}} (1 - d_i) \\theta N_i / T
  + (1 - \\theta) N / T`.
- :func:`adpm_stein_projection`, the discrete analogue for one-shot
  items (Hall 2008): the chance of failing a trial once the fixes are in,
  :math:`r = \\prod_{\\mathrm{seen}} [1 - (1 - d_i) \\theta N_i / T]
  \\cdot \\exp[-(1 - \\theta) N / T]` over ``T`` trials.

In every one the shrinkage :math:`\\theta` (or the AMPM's :math:`\\beta`)
comes from how often modes repeat: when most modes have been seen several
times, the test has found most of what there is to find and
:math:`\\theta` is near 1; when most were seen once, much is still hidden.
Each refuses data with no repeated mode, where that cannot be judged.

These are the many-modes (``k`` to infinity) forms, the ones the handbook
recommends for complex systems; the finite-``k`` forms and the
method-of-moments estimates are not built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special

__all__ = [
    "AMPMProjection",
    "SteinProjection",
    "DiscreteSteinProjection",
    "ampm_projection",
    "ampm_stein_projection",
    "adpm_stein_projection",
]


def _reciprocal(value: float) -> float:
    return float(np.inf) if value <= 0 else float(1.0 / value)


def _missing(label: Any) -> bool:
    return label is None or (isinstance(label, float) and np.isnan(label))


def _factors(fef: Any, seen: set) -> dict:
    if not isinstance(fef, dict):
        raise ValueError(
            "fef must be a dict of mode label -> fix-effectiveness factor; "
            f"got {type(fef).__name__}."
        )
    unknown = set(fef) - seen
    if unknown:
        raise ValueError(
            "These modes have no failures in the data: "
            f"{sorted(map(str, unknown))}. Only modes seen in the test can "
            "have a fix-effectiveness factor."
        )
    out = {}
    for label, value in fef.items():
        d = float(value)
        if not 0.0 <= d <= 1.0:
            raise ValueError(
                "A fix-effectiveness factor is a fraction in [0, 1]; mode "
                f"{label!r} has {value!r}."
            )
        out[label] = d
    return out


def _events(
    x: npt.ArrayLike,
    modes: npt.ArrayLike,
    c: npt.ArrayLike | None,
    i: npt.ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """The failure times and their modes, the end of the test ``T`` (the
    same for every system) and the number of systems."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    labels = list(np.atleast_1d(np.asarray(modes, dtype=object)))
    if len(labels) != x_arr.size:
        raise ValueError(
            f"modes must give one label per row of x ({x_arr.size} rows); "
            f"got {len(labels)}."
        )
    if c is None:
        raise ValueError(
            "Give c: 0 for a failure and 1 for the end of each system's "
            "test (a row at the time the test stopped), so that the test "
            "time is known."
        )
    c_arr = np.atleast_1d(np.asarray(c))
    i_arr = (
        np.ones(x_arr.size, dtype=int)
        if i is None
        else np.atleast_1d(np.asarray(i))
    )
    if c_arr.shape != x_arr.shape or i_arr.shape != x_arr.shape:
        raise ValueError("x, i, c and modes must have the same length.")
    if not np.all(np.isin(c_arr, (0, 1))):
        raise ValueError(
            "A maturity projection takes exact failures (c=0) and the end "
            "of each system's test (c=1); got c values "
            f"{sorted(set(c_arr.tolist()) - {0, 1})}."
        )
    if np.any(~np.isfinite(x_arr)) or np.any(x_arr < 0):
        raise ValueError("x must be finite times from 0.")
    events = c_arr == 0
    mode_arr = np.empty(x_arr.size, dtype=object)
    mode_arr[:] = labels
    unlabelled = [
        int(k) for k in np.flatnonzero(events) if _missing(mode_arr[k])
    ]
    if unlabelled:
        raise ValueError(
            f"Every failure (c=0) needs a mode label; {len(unlabelled)} have "
            f"none (rows {unlabelled[:5]})."
        )
    systems = list(dict.fromkeys(i_arr.tolist()))
    ends = []
    for unit in systems:
        rows = i_arr == unit
        stops = x_arr[rows & ~events]
        if stops.size != 1 or np.any(x_arr[rows & events] > stops[0]):
            raise ValueError(
                "A maturity projection needs each system's end of test: "
                "one c=1 row per system, at or after its failures. "
                f"System {unit!r} has {stops.size}."
            )
        ends.append(float(stops[0]))
    if len(set(ends)) != 1:
        raise ValueError(
            "Every system must be tested to the same end, T; got "
            f"{sorted(set(ends))}."
        )
    if not events.any():
        raise ValueError("There are no failures (c=0) to project from.")
    T = ends[0]
    if T <= 0:
        raise ValueError(f"The end of the test must be after 0; got {T}.")
    return x_arr[events], mode_arr[events], T, len(systems)


# -- AMPM ---------------------------------------------------------------------


@dataclass(frozen=True)
class AMPMProjection:
    """The AMSAA maturity projection (AMPM, many-modes limit): returned by
    :func:`ampm_projection`.

    Attributes
    ----------
    T : float
        The end of the test.
    failures : dict
        The number of failures of each kind, keys ``"A"`` and ``"B"``.
    modes : pandas.DataFrame
        One row per B mode seen, indexed by its label: its failures, its
        first occurrence (``first``) and its fix-effectiveness factor
        (``fef``).
    beta : float
        The scale of the gamma population of B-mode rates in the limit:
        new modes turn up at :math:`h(t) = \\lambda_B / (1 + \\beta t)`.
    lambda_A : float
        The A modes' intensity, ``N_A / T``.
    lambda_B : float
        The B modes' initial intensity, :math:`m \\beta / \\ln(1 +
        \\beta T)` (``m`` B modes seen).
    mean_fef : float
        The mean fix-effectiveness factor of the B modes seen,
        :math:`\\mu_d`.
    """

    T: float
    failures: dict
    modes: pd.DataFrame
    beta: float
    lambda_A: float
    lambda_B: float
    mean_fef: float

    def expected_modes(self, t: npt.ArrayLike) -> Any:
        """The expected number of distinct B modes seen by ``t``,
        :math:`(\\lambda_B / \\beta) \\ln(1 + \\beta t)`; at ``T`` it is the
        number seen."""
        t = np.asarray(t, dtype=float)
        return self.lambda_B / self.beta * np.log1p(self.beta * t)

    def new_mode_intensity(self, t: npt.ArrayLike) -> Any:
        """The rate at which new B modes turn up at ``t``,
        :math:`h(t) = \\lambda_B / (1 + \\beta t)`: also the intensity of
        the B modes not yet seen by ``t``."""
        t = np.asarray(t, dtype=float)
        return self.lambda_B / (1 + self.beta * t)

    def maturity(self, t: npt.ArrayLike) -> Any:
        """The expected fraction of the B modes' initial intensity in
        modes seen by ``t``, :math:`\\beta t / (1 + \\beta t)`: the most
        that fixing what has been seen can remove, however good the
        fixes."""
        t = np.asarray(t, dtype=float)
        return self.beta * t / (1 + self.beta * t)

    def intensity(self, t: npt.ArrayLike) -> Any:
        """The projected intensity once every B mode seen by ``t`` is
        fixed: :math:`\\lambda_A + (1 - \\mu_d) \\lambda_B + \\mu_d h(t)`,
        for ``t`` at or after the end of the test (testing on to ``t`` with
        the same fixes)."""
        return self.growth_potential_intensity + self.mean_fef * (
            self.new_mode_intensity(t)
        )

    def mtbf(self, t: npt.ArrayLike) -> Any:
        """``1 / intensity(t)``."""
        return 1.0 / np.asarray(self.intensity(t))

    @property
    def projected_intensity(self) -> float:
        """The intensity once the B modes seen in the test are fixed,
        ``intensity(T)``."""
        return float(self.intensity(self.T))

    @property
    def projected_mtbf(self) -> float:
        return _reciprocal(self.projected_intensity)

    @property
    def growth_potential_intensity(self) -> float:
        """The intensity once every B mode is found and fixed with the mean
        factor, :math:`\\lambda_A + (1 - \\mu_d) \\lambda_B`."""
        return float(self.lambda_A + (1 - self.mean_fef) * self.lambda_B)

    @property
    def growth_potential_mtbf(self) -> float:
        return _reciprocal(self.growth_potential_intensity)

    def __repr__(self) -> str:
        title = "Maturity projection (AMPM)"
        lines = [
            title,
            "=" * len(title),
            f"Test             : to T = {self.T:g}",
            "Failures         : {} A, {} B ({} B modes)".format(
                self.failures["A"], self.failures["B"], len(self.modes)
            ),
            f"Mean FEF         : {self.mean_fef:.4g}",
            "B modes          : lambda_B = {:.4g}, beta = {:.4g}, "
            "seen {:.1%} of their intensity".format(
                self.lambda_B, self.beta, float(self.maturity(self.T))
            ),
            "                   {:>12} {:>12}".format("intensity", "MTBF"),
        ]
        for label, rate in (
            ("Projected", self.projected_intensity),
            ("Growth potential", self.growth_potential_intensity),
        ):
            lines.append(
                f"{label:<17}: {rate:>12.4g} {_reciprocal(rate):>12.4g}"
            )
        return "\n".join(lines)


def _ampm_beta(first: np.ndarray, T: float) -> float:
    """The maximum likelihood ``beta`` of the first occurrences, a Poisson
    process with mean :math:`(\\lambda_B/\\beta)\\ln(1 + \\beta t)` on
    ``[0, T]``."""
    m = first.size

    def score(log_b: float) -> float:
        b = np.exp(log_b)
        return float(
            m / b
            - m * T / ((1 + b * T) * np.log1p(b * T))
            - np.sum(first / (1 + b * first))
        )

    # The score's sign near beta = 0 is that of m T / 2 - sum(t_i).
    if np.sum(first) >= m * T / 2:
        raise ValueError(
            "The B modes' first occurrences do not thin out: their mean "
            f"({np.mean(first):.4g}) is not below half the test "
            f"({T / 2:.4g}), so new modes were turning up at least as fast "
            "at the end as at the start, and the AMPM has no estimate (its "
            "beta is 0). Test longer, or use CrowAMSAA.projection."
        )
    lo, hi = np.log(1e-12 / T), np.log(1e12 / T)
    if score(hi) > 0:
        raise ValueError(
            "The AMPM's beta has no finite estimate for these first "
            "occurrences (they bunch at the start of the test)."
        )
    while score(lo) <= 0:
        lo -= 5.0
    return float(np.exp(optimize.brentq(score, lo, hi, xtol=1e-14)))


def ampm_projection(
    x: npt.ArrayLike,
    modes: npt.ArrayLike,
    fef: dict,
    c: npt.ArrayLike | None = None,
    i: npt.ArrayLike | None = None,
) -> AMPMProjection:
    """
    The AMSAA maturity projection model (AMPM), its many-modes form
    (MIL-HDBK-189C section 6.3.4; MIL-HDBK-00189A section 7.6).

    In plain words: the B modes' first occurrences show how quickly the
    test is running out of new modes. Fitting the rate at which new modes
    turn up, :math:`h(t) = \\lambda_B / (1 + \\beta t)`, gives both the B
    modes' total initial intensity :math:`\\lambda_B` and the part of it
    still in modes not seen. The projected intensity once the seen modes
    are fixed is

    .. math::
        \\rho(t) = \\lambda_A + (1 - \\mu_d) \\lambda_B + \\mu_d h(t),

    :math:`\\lambda_A = N_A / T`, :math:`\\mu_d` the mean FEF of the B modes
    seen. Unlike the AMSAA-Crow projection (:meth:`CrowAMSAA.projection
    <surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.projection>`) it
    does not matter when the fixes went in, during the test or after it:
    only the first occurrences are used.

    :math:`\\beta` is the maximum likelihood estimate from the first
    occurrences ``t_i`` of the ``m`` B modes, the root of

    .. math::
        \\frac{m}{\\beta} - \\frac{m T}{(1 + \\beta T) \\ln(1 + \\beta T)}
        - \\sum_{i=1}^m \\frac{t_i}{1 + \\beta t_i} = 0,

    and :math:`\\lambda_B = m \\beta / \\ln(1 + \\beta T)`. (The first
    occurrences are a Musa-Okumoto logarithmic Poisson process.) A root
    exists only when the first occurrences thin out, their mean below
    ``T / 2``; otherwise the call refuses.

    Parameters
    ----------
    x : array_like
        Each failure's time, and the end of the test (one row with
        ``c=1``).
    modes : array_like
        Each failure's mode label (anything for the end-of-test row).
    fef : dict
        The B modes (to be fixed, during the test or after it), each with
        its fix-effectiveness factor in [0, 1]. Every other mode is an A
        mode, not to be fixed.
    c : array_like
        0 for a failure, 1 for the end of the test.
    i : array_like, optional
        The system of each row: one system only (the first occurrences
        are on one test clock; for several systems, give the cumulative
        test time of the fleet as ``x``).

    Returns
    -------
    AMPMProjection

    Examples
    --------
    >>> from surpyval.recurrent import ampm_projection
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
    ...          "a1", "b5", None]
    >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
    >>> result = ampm_projection(x, modes, fef, c=[0] * 11 + [1])
    >>> round(result.projected_mtbf, 2), round(result.growth_potential_mtbf, 2)
    (50.12, 66.58)
    """
    t_ev, m_ev, T, systems = _events(x, modes, c, i)
    if systems != 1:
        raise ValueError(
            "The AMPM reads the B modes' first occurrences on one test "
            f"clock; got {systems} systems. Give the fleet's cumulative "
            "test time as x (and one end of test), or use "
            "ampm_stein_projection, which pools systems."
        )
    factors = _factors(fef, set(m_ev.tolist()))
    labels = [m for m in dict.fromkeys(m_ev.tolist()) if m in factors]
    if len(labels) < 2:
        raise ValueError(
            "The AMPM needs at least two B modes (keys of fef with "
            f"failures); got {len(labels)}."
        )
    n_i = np.array([np.sum(m_ev == m) for m in labels], dtype=float)
    first = np.array([t_ev[m_ev == m].min() for m in labels], dtype=float)
    d_i = np.array([factors[m] for m in labels], dtype=float)
    beta = _ampm_beta(first, T)
    m = len(labels)
    n_b = int(n_i.sum())
    table = pd.DataFrame(
        {"failures": n_i.astype(int), "first": first, "fef": d_i},
        index=pd.Index(labels, name="mode"),
    )
    return AMPMProjection(
        T=T,
        failures={"A": int(t_ev.size - n_b), "B": n_b},
        modes=table,
        beta=beta,
        lambda_A=float((t_ev.size - n_b) / T),
        lambda_B=float(m * beta / np.log1p(beta * T)),
        mean_fef=float(d_i.mean()),
    )


# -- AMPM-Stein ---------------------------------------------------------------


def _stein_y(N: float, m: int) -> float:
    """The root ``y > 0`` of ``N ln(1 + y) = m y``: ``beta T`` in the
    many-modes limit of the gamma-Poisson marginal likelihood of the mode
    counts."""

    def f(y: float) -> float:
        return float(N * np.log1p(y) - m * y)

    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
    return float(optimize.brentq(f, 1e-300, hi, xtol=1e-14, rtol=1e-15))


@dataclass(frozen=True)
class SteinProjection:
    """The AMPM-Stein projection (many-modes limit): returned by
    :func:`ampm_stein_projection`.

    Attributes
    ----------
    T : float
        The total test time (systems times the end of the test).
    systems : int
    modes : pandas.DataFrame
        One row per mode seen: its failures, fix-effectiveness factor,
        observed intensity ``N_i / T``, Stein estimate ``theta N_i / T``
        and projected intensity ``(1 - d_i) theta N_i / T``; and whether it
        is an A mode (``kind``).
    theta : float
        The Stein shrinkage factor, :math:`\\beta T / (1 + \\beta T)`.
    beta : float
        The gamma population's scale in the limit, the root of
        :math:`N \\ln(1 + \\beta T) = m \\beta T` over the modes it is
        estimated from.
    lambda_A : float
        The intensity of the A modes (``a_modes``), ``N_A / T``; 0 with
        one classification.
    unseen_intensity : float
        The intensity of the modes not seen, :math:`(1 - \\theta) N / T`.
    projected_intensity : float
    initial_intensity : float
        The intensity before any fix, ``N / T`` over every mode.
    """

    T: float
    systems: int
    modes: pd.DataFrame
    theta: float
    beta: float
    lambda_A: float
    unseen_intensity: float
    projected_intensity: float
    initial_intensity: float

    @property
    def projected_mtbf(self) -> float:
        return _reciprocal(self.projected_intensity)

    @property
    def initial_mtbf(self) -> float:
        return _reciprocal(self.initial_intensity)

    def __repr__(self) -> str:
        title = "Maturity projection (AMPM-Stein)"
        lines = [
            title,
            "=" * len(title),
            "Test             : {} system{}, total time {:g}".format(
                self.systems, "" if self.systems == 1 else "s", self.T
            ),
            "Modes seen       : {} ({} failures)".format(
                len(self.modes), int(self.modes["failures"].sum())
            ),
            f"Shrinkage        : theta = {self.theta:.4g}",
            f"Unseen modes     : {self.unseen_intensity:.4g}",
            "                   {:>12} {:>12}".format("intensity", "MTBF"),
            "{:<17}: {:>12.4g} {:>12.4g}".format(
                "Initial", self.initial_intensity, self.initial_mtbf
            ),
            "{:<17}: {:>12.4g} {:>12.4g}".format(
                "Projected", self.projected_intensity, self.projected_mtbf
            ),
        ]
        return "\n".join(lines)


def ampm_stein_projection(
    x: npt.ArrayLike,
    modes: npt.ArrayLike,
    fef: dict,
    c: npt.ArrayLike | None = None,
    i: npt.ArrayLike | None = None,
    a_modes: Any = None,
) -> SteinProjection:
    """
    The AMPM-Stein projection, its many-modes form (MIL-HDBK-189C section
    6.3.5; MIL-HDBK-00189A section 7.7.5).

    In plain words: a mode seen ``N_i`` times in ``T`` hours has the
    estimate ``N_i / T``, which is too high for the modes that happened to
    show up (and 0, too low, for those that did not). Stein's estimate
    pulls every mode's rate towards the average by a factor
    :math:`\\theta`, chosen to make the total squared error smallest, and
    the projected intensity once the seen modes are fixed is

    .. math::
        \\rho_S = \\sum_{\\mathrm{seen}} (1 - d_i)\\, \\theta \\frac{N_i}{T}
                  + (1 - \\theta) \\frac{N}{T},

    the last term the modes not yet seen. Over many modes whose rates are a
    gamma sample, :math:`\\theta = \\beta T / (1 + \\beta T)`, with
    :math:`\\beta T` the root of :math:`N \\ln(1 + \\beta T) = m \\beta T`
    (``m`` modes seen, ``N`` failures): the maximum likelihood estimate
    from the counts alone (Ellner and Hall 2006). It exists when some
    mode failed more than once.

    No mode needs to be called A or B: a mode with no factor in ``fef``
    has ``d_i = 0``. With ``a_modes``, the modes listed are A modes,
    classified before the test, and the projection is
    :math:`N_A / T + \\sum_{\\mathrm{seen\\ B}} (1 - d_i) \\theta_B N_i / T
    + (1 - \\theta_B) N_B / T`, :math:`\\theta_B` from the B modes alone.

    Parameters
    ----------
    x, modes, c, i : array_like
        As for :func:`ampm_projection`; several systems may be tested side
        by side to the same end, ``T``, pooled as ``kT`` of test.
    fef : dict
        The fix-effectiveness factor, in [0, 1], of each mode to be fixed.
    a_modes : iterable, optional
        Modes classified as A modes (not to be fixed) before the test.

    Returns
    -------
    SteinProjection

    Examples
    --------
    >>> from surpyval.recurrent import ampm_stein_projection
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
    ...          "a1", "b5", None]
    >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
    >>> result = ampm_stein_projection(x, modes, fef, c=[0] * 11 + [1])
    >>> round(result.theta, 4), round(result.projected_mtbf, 2)
    (0.5704, 52.2)
    """
    t_ev, m_ev, T_end, systems = _events(x, modes, c, i)
    T = T_end * systems
    seen = list(dict.fromkeys(m_ev.tolist()))
    factors = _factors(fef, set(seen))
    if a_modes is None:
        a_set: set = set()
    else:
        if isinstance(a_modes, (str, bytes)):
            raise ValueError(
                "a_modes must be a collection of mode labels, not a single "
                f"string; got {a_modes!r}."
            )
        a_set = set(a_modes)
        both = a_set & set(factors)
        if both:
            raise ValueError(
                "An A mode is not fixed, so it has no fix-effectiveness "
                f"factor: {sorted(map(str, both))}."
            )
    n_i = np.array([np.sum(m_ev == m) for m in seen], dtype=float)
    d_i = np.array([factors.get(m, 0.0) for m in seen], dtype=float)
    is_a = np.array([m in a_set for m in seen])
    b = ~is_a
    N_b, m_b = float(n_i[b].sum()), int(b.sum())
    if m_b == 0 or N_b <= m_b:
        what = "B mode" if a_set else "mode"
        raise ValueError(
            f"The Stein shrinkage needs some {what} seen more than once: "
            f"{m_b} {what}s with {int(N_b)} failures leave no way to judge "
            "how much is unseen (its estimate is infinite). Test longer, "
            "or use ampm_projection or CrowAMSAA.projection."
        )
    y = _stein_y(N_b, m_b)
    theta = y / (1 + y)
    shrunk = np.where(b, theta * n_i / T, n_i / T)
    projected_i = (1 - d_i) * shrunk
    lambda_a = float(n_i[is_a].sum() / T)
    unseen = float((1 - theta) * N_b / T)
    table = pd.DataFrame(
        {
            "kind": np.where(is_a, "A", "B"),
            "failures": n_i.astype(int),
            "fef": d_i,
            "intensity": n_i / T,
            "stein": shrunk,
            "projected": projected_i,
        },
        index=pd.Index(seen, name="mode"),
    )
    return SteinProjection(
        T=float(T),
        systems=systems,
        modes=table,
        theta=float(theta),
        beta=float(y / T),
        lambda_A=lambda_a,
        unseen_intensity=unseen,
        projected_intensity=float(projected_i[b].sum() + lambda_a + unseen),
        initial_intensity=float(n_i.sum() / T),
    )


# -- the discrete (one-shot) Stein projection ---------------------------------


@dataclass(frozen=True)
class DiscreteSteinProjection:
    """The discrete AMPM-Stein projection for one-shot items (many-modes
    limit): returned by :func:`adpm_stein_projection`.

    Attributes
    ----------
    trials : int
        The trials of the test, ``T``.
    modes : pandas.DataFrame
        One row per mode seen: its failures, fix-effectiveness factor,
        observed chance ``N_i / T``, Stein estimate ``theta N_i / T`` and
        projected chance ``(1 - d_i) theta N_i / T``.
    pseudo_trials : float
        The beta population's ``n`` (pseudo trials) in the limit, the
        maximum likelihood estimate from the counts.
    theta : float
        The shrinkage factor, :math:`T / (n + T)`.
    initial_reliability : float
        The expected chance that a trial succeeds before any fix,
        :math:`\\exp[-m / \\sum_{j=0}^{T-1} n / (n + j)]`: the population
        estimate, an alternative to the share of trials that succeeded.
    unseen_reliability : float
        The chance of no failure from a mode not seen,
        :math:`\\exp[-(1 - \\theta) N / T]`.
    projected_reliability : float
        The chance a trial succeeds once the seen modes are fixed.
    """

    trials: int
    modes: pd.DataFrame
    pseudo_trials: float
    theta: float
    initial_reliability: float
    unseen_reliability: float
    projected_reliability: float

    def __repr__(self) -> str:
        title = "Maturity projection (discrete AMPM-Stein)"
        lines = [
            title,
            "=" * len(title),
            f"Trials           : {self.trials}",
            "Modes seen       : {} ({} failures)".format(
                len(self.modes), int(self.modes["failures"].sum())
            ),
            "Shrinkage        : theta = {:.4g} (n = {:.4g})".format(
                self.theta, self.pseudo_trials
            ),
            f"Initial          : {self.initial_reliability:.4g}",
            f"Projected        : {self.projected_reliability:.4g}",
        ]
        return "\n".join(lines)


def _adpm_profile(n: float, counts: np.ndarray, T: int) -> float:
    """The many-modes beta-binomial marginal log-likelihood of the mode
    counts at pseudo trials ``n``, with the modes' total rate at its
    maximum (constants dropped)."""
    m = counts.size
    harmonic = float(np.sum(1.0 / (n + np.arange(T))))
    beta_terms = np.sum(special.betaln(counts, n + T - counts))
    return float(-m * np.log(harmonic) + beta_terms)


def adpm_stein_projection(
    failures: dict,
    trials: int,
    fef: dict | None = None,
) -> DiscreteSteinProjection:
    """
    The discrete AMPM-Stein projection for one-shot items, its many-modes
    form (MIL-HDBK-189C section 6.3.6; MIL-HDBK-00189A section 7.8).

    In plain words: ``T`` trials, each a success or a failure; a failure
    has one or more modes, and mode ``i`` was behind ``N_i`` of them. Each
    mode's chance of causing a failure is estimated by shrinking ``N_i /
    T`` towards the average (Stein's estimate), which also gives the modes
    not seen a small chance each. Once the seen modes are fixed, the chance
    a trial succeeds is

    .. math::
        r = \\prod_{\\mathrm{seen}} \\left[1 - (1 - d_i)\\,\\theta
            \\frac{N_i}{T}\\right] \\exp\\left[-(1 - \\theta)\\frac{N}{T}
            \\right],

    the exponential the modes not seen. Over many modes whose chances are
    a beta sample with ``n`` pseudo trials, :math:`\\theta = T / (n + T)`.
    ``n`` is the maximum likelihood estimate from the counts: it
    maximises

    .. math::
        -m \\ln \\sum_{j=0}^{T-1} \\frac{1}{n + j}
        + \\sum_{\\mathrm{seen}} \\ln B(N_i, n + T - N_i),

    the beta-binomial marginal likelihood in the limit of many modes
    with the modes' total chance at its maximum. It needs a mode seen
    more than once.

    Parameters
    ----------
    failures : dict
        Each mode seen and the number of trials it failed (whole numbers
        from 1 to ``trials``). A trial may fail by several modes.
    trials : int
        The trials of the test, ``T``.
    fef : dict, optional
        The fix-effectiveness factor, in [0, 1], of each mode to be fixed;
        the others are not fixed (``d_i = 0``).

    Returns
    -------
    DiscreteSteinProjection

    Examples
    --------
    >>> from surpyval.recurrent import adpm_stein_projection
    >>> result = adpm_stein_projection(
    ...     {"seal": 4, "igniter": 2, "fin": 1, "fuze": 1}, trials=40,
    ...     fef={"seal": 0.8, "igniter": 0.7, "fin": 0.6},
    ... )
    >>> round(result.theta, 3), round(result.projected_reliability, 3)
    (0.721, 0.899)
    """
    if not isinstance(failures, dict) or not failures:
        raise ValueError(
            "failures must be a dict of mode label -> number of trials it "
            f"failed; got {failures!r}."
        )
    if (
        isinstance(trials, bool)
        or not float(trials).is_integer()
        or trials < 1
    ):
        raise ValueError(
            f"trials must be a whole number, 1 or more; got {trials!r}."
        )
    T = int(trials)
    labels = list(failures)
    counts = np.array([float(failures[m]) for m in labels])
    if np.any(counts != np.round(counts)) or np.any(counts < 1):
        raise ValueError(
            "Each mode's failures must be a whole number, 1 or more (list "
            "only the modes seen)."
        )
    if np.any(counts > T):
        raise ValueError(
            f"A mode cannot fail more trials than were run ({T})."
        )
    factors = _factors(fef or {}, set(labels))
    d_i = np.array([factors.get(m, 0.0) for m in labels])
    N, m = float(counts.sum()), counts.size
    if N <= m:
        raise ValueError(
            "The Stein shrinkage needs some mode seen more than once: with "
            "every mode seen once there is no way to judge how much is "
            "unseen. Test further, or fit a discrete growth model "
            "(CrowAMSAADiscrete)."
        )

    def nll(log_n: float) -> float:
        return -_adpm_profile(float(np.exp(log_n)), counts, T)

    # The profile has one maximum in n > 0 when some mode repeats; search
    # its log over a wide bracket.
    grid = np.linspace(np.log(1e-6), np.log(1e9 * T), 200)
    values = [nll(g) for g in grid]
    k = int(np.argmin(values))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = optimize.minimize_scalar(
        nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}
    )
    n = float(np.exp(res.x))
    if k == grid.size - 1:
        raise ValueError(
            "The modes' counts give no finite pseudo trials (the shrinkage "
            "is 0): they look like a test that has seen almost nothing of "
            "what is there."
        )
    theta = T / (n + T)
    shrunk = theta * counts / T
    projected_i = (1 - d_i) * shrunk
    unseen = float(np.exp(-(1 - theta) * N / T))
    initial = float(np.exp(-m / np.sum(n / (n + np.arange(T)))))
    table = pd.DataFrame(
        {
            "failures": counts.astype(int),
            "fef": d_i,
            "chance": counts / T,
            "stein": shrunk,
            "projected": projected_i,
        },
        index=pd.Index(labels, name="mode"),
    )
    return DiscreteSteinProjection(
        trials=T,
        modes=table,
        pseudo_trials=n,
        theta=float(theta),
        initial_reliability=initial,
        unseen_reliability=unseen,
        projected_reliability=float(np.prod(1 - projected_i) * unseen),
    )
