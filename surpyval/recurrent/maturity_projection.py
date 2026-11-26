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

Each record's ``*_cb`` methods give parametric bootstrap bounds: the whole
test is simulated again from the fitted mode population (which modes
surface, how often and, for the AMPM, when), each surfaced mode is given a
fix-effectiveness factor drawn from the assessed ones, and the projection
is estimated again from each simulated test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special

from surpyval.utils.rng import as_generator
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

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

    Examples
    --------
    >>> from surpyval.recurrent import ampm_projection
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
    ...          "a1", "b5", None]
    >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
    >>> result = ampm_projection(x, modes, fef, c=[0] * 11 + [1])
    >>> type(result).__name__, round(result.projected_mtbf, 2)
    ('AMPMProjection', 50.12)
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

    # -- bounds ------------------------------------------------------------

    def intensity_cb(
        self,
        t: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Parametric bootstrap bounds on :meth:`intensity` at ``t``
        (``None``, the default, is the end of the test, ``T``; ``inf`` is
        the growth potential).

        In plain words: each of ``n_boot`` replicates runs the test again
        on a system drawn from the fitted model, estimates the projection
        from it the same way, and compares. The model is the many-modes
        gamma population at the fitted ``lambda_B`` and ``beta``: the B
        modes a test shows are a Poisson number (mean the number seen),
        each with a logarithmic-series count of failures
        (:math:`p = \\beta T / (1 + \\beta T)`) spread uniformly over the
        test (so its first occurrence is ``T`` times the least of that many
        uniforms), and a fix-effectiveness factor drawn from those assessed
        for the B modes seen, independently of its rate (so the mean factor
        varies with which modes surface, as it would); the A failures are
        a Poisson count of ``lambda_A T``.

        ``target="realised"`` (the default) bounds the intensity this
        system will have once the B modes seen by ``t`` are fixed: a
        prediction interval, since that intensity is itself uncertain
        (how large the modes not seen are, and the seen ones' true rates).
        Each replicate also draws its system's realised value (a seen
        mode's rate, given ``N`` failures, is gamma with shape ``N`` and
        scale :math:`\\beta / (1 + \\beta T)`; the modes not seen total a
        gamma of shape :math:`\\lambda_B / \\beta` and the same scale), and
        the bounds are the estimate less the percentiles of the replicates'
        errors, on the log of the intensity. ``target="expected"`` bounds
        the model's projected intensity, its expected value over tests like
        this one: the percentiles of the replicates' estimates. The AMPM
        reads only first occurrences, so neither interval closes up as the
        test lengthens (the total B-mode intensity is read from how fast
        new modes appear, not from the repeats); AMPM-Stein's realised
        interval does.

        Held fixed: the assessed factors themselves (an engineering
        judgement, not a result of the test, so they are not perturbed),
        which modes are B modes, and ``T``. A replicate whose first
        occurrences do not thin out is estimated at its limit (no growth,
        :math:`\\beta \\to 0`) and one with no B mode at ``lambda_A``,
        rather than dropped. The percentiles are ``alpha_ci / 2`` and
        ``1 - alpha_ci / 2`` two-sided, ``alpha_ci`` one-sided.

        Parameters
        ----------
        t : array_like, optional
            As :meth:`intensity` takes it; ``T`` by default.
        alpha_ci : float, optional
            One minus the confidence level, by default 0.05.
        bound : str, optional
            ``"two-sided"`` (rows of lower and upper), ``"lower"`` or
            ``"upper"``.
        method : str, optional
            ``"bootstrap"``, the only bound built: the projection combines
            a fitted population with assessed factors, and its estimates
            sit on the edge of their range for tests a little smaller than
            the one seen, where Wald and likelihood-ratio approximations
            fail.
        target : str, optional
            Keyword only: ``"realised"`` (this system's intensity once
            fixed, the default) or ``"expected"`` (the model's expected
            projected intensity).
        n_boot : int, optional
            Keyword only: the bootstrap's replicates, by default 1000.
        random_state : None, int or numpy.random.Generator, optional
            Keyword only: the bootstrap's seed (see the conventions on
            random draws); ``None`` draws from numpy's global generator.

        Returns
        -------
        numpy.ndarray or float
            ``(..., 2)`` for two-sided bounds, else the shape of ``t``.

        Examples
        --------
        >>> from surpyval.recurrent import ampm_projection
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
        ...          "a1", "b5", None]
        >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
        >>> result = ampm_projection(x, modes, fef, c=[0] * 11 + [1])
        >>> result.projected_mtbf_cb(bound="lower", random_state=1).round(1)
        np.float64(27.6)
        """
        tail = _boot_tail(alpha_ci, bound, method, target, n_boot)
        times = np.asarray(self.T if t is None else t, dtype=float)
        if np.any(np.isnan(times)) or np.any(times < 0):
            raise ValueError("t must be times from 0 (inf for the limit).")
        flat = times.reshape(-1)
        est, real = _ampm_draws(
            self,
            flat,
            int(n_boot),
            as_generator(random_state),
            target == "realised",
        )
        point = np.asarray(self.intensity(flat), dtype=float)
        return _boot_bounds(point, est, real, tail, bound, times.shape, "log")

    def mtbf_cb(
        self,
        t: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Bounds on :meth:`mtbf`: the reciprocals of
        :meth:`intensity_cb`'s (the same replicates for the same seed), so
        a lower bound here is an upper bound on the intensity. See
        :meth:`intensity_cb` for the arguments."""
        return _reciprocal_bounds(
            lambda b: self.intensity_cb(
                t,
                alpha_ci,
                b,
                method,
                target=target,
                n_boot=n_boot,
                random_state=random_state,
            ),
            bound,
        )

    def projected_intensity_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Bounds on :attr:`projected_intensity`, ``intensity_cb(T)``."""
        return self.intensity_cb(
            None,
            alpha_ci,
            bound,
            method,
            target=target,
            n_boot=n_boot,
            random_state=random_state,
        )

    def projected_mtbf_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Bounds on :attr:`projected_mtbf`, ``mtbf_cb(T)``."""
        return self.mtbf_cb(
            None,
            alpha_ci,
            bound,
            method,
            target=target,
            n_boot=n_boot,
            random_state=random_state,
        )

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

    Examples
    --------
    >>> from surpyval.recurrent import ampm_stein_projection
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
    ...          "a1", "b5", None]
    >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
    >>> result = ampm_stein_projection(x, modes, fef, c=[0] * 11 + [1])
    >>> type(result).__name__, round(result.projected_mtbf, 2)
    ('SteinProjection', 52.2)
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

    # -- bounds ------------------------------------------------------------

    def projected_intensity_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Parametric bootstrap bounds on :attr:`projected_intensity`.

        In plain words: each of ``n_boot`` replicates runs the test again
        on a system drawn from the fitted model, estimates the projection
        from it the same way, and compares. The model is the many-modes
        gamma population at the fitted ``beta`` and :math:`\\lambda_B =
        N_B / T`: the B modes a test shows are a Poisson number (mean the
        number seen), each with a logarithmic-series count of failures
        (:math:`p = \\theta`) and a fix-effectiveness factor drawn from
        those of the B modes seen (0 for a mode not to be fixed),
        independently of its rate; with ``a_modes``, the A failures are a
        Poisson count of ``lambda_A T``.

        ``target="realised"`` (the default) bounds the intensity this
        system will have once its seen modes are fixed, the quantity the
        Stein projection estimates: a prediction interval. Each replicate
        also draws its system's realised value (a seen mode's rate, given
        ``N`` failures, is gamma with shape ``N`` and scale
        :math:`\\beta / (1 + \\beta T)`; the modes not seen total a gamma of
        shape :math:`\\lambda_B / \\beta` and the same scale), and the bounds
        are the estimate less the percentiles of the replicates' errors, on
        the log of the intensity. It closes up as the test lengthens.
        ``target="expected"`` bounds the model's expected projected
        intensity over tests like this one (the percentiles of the
        replicates' estimates), which stays as wide as the spread of the
        mode population's total.

        Held fixed: the assessed factors themselves (not perturbed: they
        are a judgement, not a result of the test), the A-mode list and the
        total test time. A replicate with no mode seen more than once is
        estimated at its limit, :math:`\\theta = 0` (no credit for the
        fixes), rather than dropped.

        See :meth:`AMPMProjection.intensity_cb` for the arguments.

        Examples
        --------
        >>> from surpyval.recurrent import ampm_stein_projection
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
        ...          "a1", "b5", None]
        >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}
        >>> result = ampm_stein_projection(x, modes, fef, c=[0] * 11 + [1])
        >>> result.projected_mtbf_cb(bound="lower", random_state=1).round(1)
        np.float64(24.4)
        """
        tail = _boot_tail(alpha_ci, bound, method, target, n_boot)
        est, real = _stein_draws(
            self, int(n_boot), as_generator(random_state), target == "realised"
        )
        point = np.array([self.projected_intensity])
        return _boot_bounds(point, est, real, tail, bound, (), "log")

    def projected_mtbf_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Bounds on :attr:`projected_mtbf`: the reciprocals of
        :meth:`projected_intensity_cb`'s (the same replicates for the same
        seed)."""
        return _reciprocal_bounds(
            lambda b: self.projected_intensity_cb(
                alpha_ci,
                b,
                method,
                target=target,
                n_boot=n_boot,
                random_state=random_state,
            ),
            bound,
        )

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

    Examples
    --------
    >>> from surpyval.recurrent import adpm_stein_projection
    >>> result = adpm_stein_projection(
    ...     {"seal": 4, "igniter": 2, "fin": 1, "fuze": 1}, trials=40,
    ...     fef={"seal": 0.8, "igniter": 0.7, "fin": 0.6},
    ... )
    >>> type(result).__name__, round(result.projected_reliability, 3)
    ('DiscreteSteinProjection', 0.899)
    """

    trials: int
    modes: pd.DataFrame
    pseudo_trials: float
    theta: float
    initial_reliability: float
    unseen_reliability: float
    projected_reliability: float

    # -- bounds ------------------------------------------------------------

    def projected_reliability_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "bootstrap",
        *,
        target: str = "realised",
        n_boot: int = 1000,
        random_state: Any = None,
    ) -> Any:
        """Parametric bootstrap bounds on :attr:`projected_reliability`.

        In plain words: each of ``n_boot`` replicates runs the trials again
        on a design drawn from the fitted model, estimates the projection
        from it the same way, and compares. The model is the many-modes
        beta population at the fitted ``pseudo_trials`` :math:`n` and total
        chance :math:`c = m / \\sum_{j=0}^{T-1} n / (n + j)`: the number of
        modes that fail exactly ``j`` of the ``T`` trials is a Poisson
        count of mean :math:`c\\, n \\binom{T}{j} B(j, n + T - j)`, for
        ``j`` from 1 to ``T``, and each seen mode gets a fix-effectiveness
        factor drawn from those assessed (0 for a mode not to be fixed).

        ``target="realised"`` (the default) bounds the chance that this
        design's trial succeeds once its seen modes are fixed: a prediction
        interval. Each replicate also draws its design's realised value (a
        mode seen ``j`` times has the chance :math:`\\mathrm{Beta}(j, n + T
        - j)`; the modes not seen are drawn exactly as a Poisson process),
        and the bounds are the estimate less the percentiles of the
        replicates' errors, on :math:`\\ln(-\\ln r)`. It closes up as the
        trials grow. ``target="expected"`` bounds the model's expected
        projected reliability over tests like this one (the percentiles of
        the replicates' estimates).

        Held fixed: the assessed factors themselves and the number of
        trials. A replicate with no mode seen more than once is estimated
        at its limit, :math:`\\theta = 0` (no credit for the fixes), rather
        than dropped.

        See :meth:`AMPMProjection.intensity_cb` for the arguments.

        Examples
        --------
        >>> from surpyval.recurrent import adpm_stein_projection
        >>> result = adpm_stein_projection(
        ...     {"seal": 4, "igniter": 2, "fin": 1, "fuze": 1}, trials=40,
        ...     fef={"seal": 0.8, "igniter": 0.7, "fin": 0.6},
        ... )
        >>> result.projected_reliability_cb(
        ...     bound="lower", random_state=1
        ... ).round(3)
        np.float64(0.736)
        """
        tail = _boot_tail(alpha_ci, bound, method, target, n_boot)
        est, real = _adpm_draws(
            self, int(n_boot), as_generator(random_state), target == "realised"
        )
        point = np.array([self.projected_reliability])
        return _boot_bounds(point, est, real, tail, bound, (), "cloglog")

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


def _adpm_n(counts: np.ndarray, T: int) -> float:
    """The maximum likelihood pseudo trials ``n`` of the many-modes
    beta-binomial profile, or ``inf`` where the profile rises to the edge
    of the search (no mode repeats, or so few that the shrinkage is 0)."""
    if counts.size == 0 or counts.sum() <= counts.size:
        return float(np.inf)

    def nll(log_n: float) -> float:
        return -_adpm_profile(float(np.exp(log_n)), counts, T)

    # The profile has one maximum in n > 0 when some mode repeats; search
    # its log over a wide bracket. The grid only brackets the maximum, so
    # it uses the digamma form of the harmonic sum, all points at once.
    grid = np.linspace(np.log(1e-6), np.log(1e9 * T), 200)
    n_grid = np.exp(grid)
    harmonic = special.digamma(n_grid + T) - special.digamma(n_grid)
    values = counts.size * np.log(harmonic) - np.sum(
        special.betaln(counts[None, :], n_grid[:, None] + T - counts),
        axis=1,
    )
    k = int(np.argmin(values))
    if k == grid.size - 1:
        return float(np.inf)
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = optimize.minimize_scalar(
        nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}
    )
    return float(np.exp(res.x))


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

    n = _adpm_n(counts, T)
    if not np.isfinite(n):
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


# -- the bootstrap bounds -----------------------------------------------------


_NO_OTHER_BOUND = (
    "A maturity projection has only bootstrap bounds: it combines a fitted "
    "mode population with assessed fix-effectiveness factors, and its "
    "estimates sit on the edge of their range (no shrinkage, or no growth) "
    "for tests a little smaller than this one, where Wald and "
    "likelihood-ratio approximations fail. Use method='bootstrap'."
)
_TARGETS = ("realised", "expected")


def _boot_tail(
    alpha_ci: float, bound: str, method: str, target: str, n_boot: Any
) -> float:
    """Check a ``*_cb`` call's arguments; the tail of each percentile."""
    check_alpha_ci(alpha_ci)
    check_option("bound", bound, BOUNDS)
    check_option("method", method, ("bootstrap",), _NO_OTHER_BOUND)
    check_option(
        "target",
        target,
        _TARGETS,
        "'realised' bounds this system's value once the fixes are in; "
        "'expected' bounds the model's expected value over tests like it.",
    )
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 2
    ):
        raise ValueError(
            f"n_boot must be a whole number, 2 or more; got {n_boot!r}."
        )
    return alpha_ci / 2 if bound == "two-sided" else alpha_ci


def _defined_rows(values: np.ndarray) -> np.ndarray:
    """``values`` without rows holding ``nan`` (warned of, with counts)."""
    keep = ~np.any(np.isnan(values), axis=1)
    if not np.all(keep):
        import warnings

        warnings.warn(
            f"{int(np.sum(~keep))} of {len(values)} bootstrap replicates "
            "gave no estimate and were left out of the percentiles. Use "
            "more replicates.",
            stacklevel=4,
        )
    return values[keep]


def _boot_bounds(
    point: np.ndarray,
    est: np.ndarray,
    real: np.ndarray | None,
    tail: float,
    bound: str,
    shape: tuple,
    scale: str,
) -> Any:
    """Bounds from the replicates' estimates ``est`` (one row a replicate,
    one column a quantity).

    ``real is None`` (``target="expected"``): the percentiles of ``est``.
    Otherwise (``target="realised"``) the bootstrap of the prediction
    error: on ``z``, the log of an intensity (``scale="log"``) or the
    log of minus the log of a reliability (``scale="cloglog"``), the
    replicates' errors ``z(est) - z(real)`` give ``z(point) - error``
    quantiles, mapped back.
    """
    if real is None:
        draws = _defined_rows(est)
        lo = np.quantile(draws, tail, axis=0)
        hi = np.quantile(draws, 1 - tail, axis=0)
    else:

        def z(v: Any) -> Any:
            v = np.asarray(v, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(v) if scale == "log" else np.log(-np.log(v))

        # A replicate test with no failures estimates 0 (an intensity) or
        # 1 (a reliability): an error of -inf on z, kept, since it is a
        # real outcome of such a test.
        error = _defined_rows(z(est) - z(real))
        z_low = z(point) - np.quantile(error, 1 - tail, axis=0)
        z_high = z(point) - np.quantile(error, tail, axis=0)
        if scale == "log":
            lo, hi = np.exp(z_low), np.exp(z_high)
        else:  # the reliability falls as z rises
            lo, hi = np.exp(-np.exp(z_high)), np.exp(-np.exp(z_low))
    if bound == "two-sided":
        return np.column_stack([lo, hi]).reshape(shape + (2,))
    out = np.asarray(lo if bound == "lower" else hi).reshape(shape)
    return out[()] if out.ndim == 0 else out


def _reciprocal_bounds(rate_cb: Any, bound: str) -> Any:
    """Bounds on ``1 / rate`` from ``rate_cb(bound)``, the rate's bounds:
    the opposite one-sided bound, or the two-sided pair reversed."""
    check_option("bound", bound, BOUNDS)
    opposite = {"two-sided": "two-sided", "lower": "upper", "upper": "lower"}
    rate = np.asarray(rate_cb(opposite[bound]), dtype=float)
    with np.errstate(divide="ignore"):
        out = 1.0 / rate
    if bound == "two-sided":
        return out[..., ::-1]
    return out[()] if out.ndim == 0 else out


def _choice(rng: np.random.Generator, pool: np.ndarray, size: int) -> Any:
    return rng.choice(pool, size) if size else pool[:0]


def _seen_counts(
    rng: np.random.Generator, modes_mean: float, p: float
) -> np.ndarray:
    """The failure counts of the modes a many-modes gamma population
    shows in a test: a Poisson number of modes, each count a
    logarithmic-series variate with ``p = beta T / (1 + beta T)``."""
    m = int(rng.poisson(modes_mean))
    return rng.logseries(p, m).astype(float) if m else np.empty(0)


def _ampm_estimates(first: np.ndarray, T: float) -> tuple[float, float]:
    """The AMPM's ``beta`` and ``lambda_B`` from first occurrences, with
    the limits where the maximum is on the edge: no B mode (``lambda_B =
    0``), or first occurrences that do not thin out (``beta -> 0``, new
    modes at the constant rate ``m / T``)."""
    m = first.size
    if m == 0:
        return 0.0, 0.0
    if np.sum(first) >= m * T / 2:
        return 0.0, m / T
    try:
        beta = _ampm_beta(first, T)
    except ValueError:  # bunched at the start: beta at its search limit
        beta = 1e12 / T
    return beta, float(m * beta / np.log1p(beta * T))


def _ampm_rho(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The AMPM intensity for ``params`` rows ``(beta, lambda_A,
    lambda_B, mu_d)`` (rows) at each ``t`` (columns); ``h(inf) = 0``."""
    beta, lam_a, lam_b, mu = (params[:, k, None] for k in range(4))
    with np.errstate(invalid="ignore"):
        h = np.where(np.isinf(t), 0.0, lam_b / (1 + beta * t))
    return lam_a + (1 - mu) * lam_b + mu * h


def _gamma_realised(
    rng: np.random.Generator,
    a: float,
    s: float,
    pool: np.ndarray,
    u: float,
) -> float:
    """For the B modes a gamma population has not shown by ``T`` (a gamma
    process of shape ``a`` and scale ``s``): their intensity once those
    surfacing in the next ``u`` of test are fixed, each with a factor
    drawn from ``pool``. ``u = 0``: their whole intensity; ``u = inf``:
    every one fixed."""
    if u == 0:
        return float(rng.gamma(a, s))
    if np.isinf(u):
        shares = rng.gamma(a / pool.size, s, pool.size)
        return float(np.sum((1 - pool) * shares))
    s_u = s / (1 + s * u)
    counts = _seen_counts(rng, a * np.log1p(s * u), s * u / (1 + s * u))
    rates = rng.gamma(counts, s_u) if counts.size else counts
    d = _choice(rng, pool, counts.size)
    return float(np.sum((1 - d) * rates) + rng.gamma(a, s_u))


def _ampm_draws(
    fit: AMPMProjection,
    t: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    realised: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Each replicate test's AMPM intensity at ``t`` (rows replicates),
    and, with ``realised``, that replicate system's realised intensity
    once the B modes seen by each ``t`` are fixed."""
    T, beta = fit.T, fit.beta
    p, s = beta * T / (1 + beta * T), beta / (1 + beta * T)
    a = fit.lambda_B / beta  # the population's shape in the limit
    pool = fit.modes["fef"].to_numpy(dtype=float)
    est = np.empty((n_boot, t.size))
    real = np.empty((n_boot, t.size)) if realised else None
    for b in range(n_boot):
        counts = _seen_counts(rng, a * np.log1p(beta * T), p)
        # a mode's failures are uniform over the test, so its first
        # occurrence is T times the least of N uniforms
        first = T * rng.beta(1.0, counts) if counts.size else counts
        d = _choice(rng, pool, counts.size)
        n_a = rng.poisson(fit.lambda_A * T)
        b_hat, lam_b = _ampm_estimates(first, T)
        mu = float(d.mean()) if counts.size else 0.0
        est[b] = _ampm_rho(np.array([[b_hat, n_a / T, lam_b, mu]]), t)[0]
        if real is None:
            continue
        rates = rng.gamma(counts, s) if counts.size else counts
        for k, tk in enumerate(t):
            if tk <= T:  # the modes first seen after tk are not fixed
                fixed = first <= tk
                seen = np.sum(np.where(fixed, 1 - d, 1.0) * rates)
                rest = _gamma_realised(rng, a, s, pool, 0.0)
            else:
                seen = np.sum((1 - d) * rates)
                rest = _gamma_realised(rng, a, s, pool, tk - T)
            real[b, k] = fit.lambda_A + seen + rest
    return est, real


def _stein_rho(
    counts: np.ndarray, d: np.ndarray, n_a: float, T: float
) -> float:
    """The AMPM-Stein projection from the B modes' counts and factors and
    the A failures, with ``theta = 0`` where no B mode repeats."""
    N, m = float(counts.sum()), counts.size
    theta = 0.0
    if m and N > m:
        y = _stein_y(N, m)
        theta = y / (1 + y)
    seen = float(np.sum((1 - d) * theta * counts / T))
    return n_a / T + seen + (1 - theta) * N / T


def _stein_draws(
    fit: SteinProjection,
    n_boot: int,
    rng: np.random.Generator,
    realised: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Each replicate test's AMPM-Stein projection, and, with
    ``realised``, that replicate system's realised intensity once its
    seen B modes are fixed."""
    T, theta = fit.T, fit.theta
    b = (fit.modes["kind"] == "B").to_numpy()
    pool = fit.modes["fef"].to_numpy(dtype=float)[b]
    N_b = float(fit.modes["failures"].to_numpy()[b].sum())
    y = theta / (1 - theta)  # beta T
    s = y / T / (1 + y)
    a = N_b / y  # lambda_B / beta, lambda_B = N_B / T
    est = np.empty((n_boot, 1))
    real = np.empty((n_boot, 1)) if realised else None
    for k in range(n_boot):
        # (lambda_B / beta) ln(1 + beta T) = m_B, the modes seen
        counts = _seen_counts(rng, float(b.sum()), theta)
        d = _choice(rng, pool, counts.size)
        n_a = rng.poisson(fit.lambda_A * T)
        est[k] = _stein_rho(counts, d, n_a, T)
        if real is not None:
            rates = rng.gamma(counts, s) if counts.size else counts
            real[k] = (
                fit.lambda_A
                + np.sum((1 - d) * rates)
                + _gamma_realised(rng, a, s, pool, 0.0)
            )
    return est, real


def _adpm_reliability(counts: np.ndarray, d: np.ndarray, T: int) -> float:
    """The discrete AMPM-Stein projection, with ``theta = 0`` where the
    pseudo trials are infinite (no mode repeats)."""
    n = _adpm_n(counts, T)
    theta = T / (n + T) if np.isfinite(n) else 0.0
    N = float(counts.sum())
    return float(
        np.prod(1 - (1 - d) * theta * counts / T)
        * np.exp(-(1 - theta) * N / T)
    )


def _beta_unseen(rng: np.random.Generator, mass: float, rate: float) -> float:
    """``-sum ln(1 - p)`` over the modes a many-modes beta population has
    not shown in ``T`` trials. On ``q = -ln(1 - p)`` those modes are a
    Poisson process of intensity ``mass e^{-rate q} / (1 - e^{-q})``
    (``mass = c n``, ``rate = n + T``): a gamma process (the ``1 / q``
    part, a gamma of shape ``mass`` and rate ``rate``) plus a compound
    Poisson rest, drawn exactly by thinning ``Exp(rate)`` jumps with the
    chance ``1 / (1 - e^{-q}) - 1 / q``."""
    jumps = rng.exponential(1 / rate, rng.poisson(mass / rate))
    keep = rng.random(jumps.size) < -1 / np.expm1(-jumps) - 1 / jumps
    return float(rng.gamma(mass, 1 / rate) + np.sum(jumps[keep]))


def _adpm_draws(
    fit: DiscreteSteinProjection,
    n_boot: int,
    rng: np.random.Generator,
    realised: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Each replicate test's discrete projection, and, with ``realised``,
    that replicate design's realised reliability once its seen modes are
    fixed."""
    T, n = fit.trials, fit.pseudo_trials
    m = len(fit.modes)
    j = np.arange(1, T + 1, dtype=float)
    harmonic = float(np.sum(1.0 / (n + np.arange(T))))
    total = m / (n * harmonic)  # the population's total chance, c
    # the expected number of modes failing exactly j of the T trials
    mean_j = np.exp(
        np.log(total * n)
        + special.gammaln(T + 1)
        - special.gammaln(j + 1)
        - special.gammaln(T - j + 1)
        + special.betaln(j, n + T - j)
    )
    pool = fit.modes["fef"].to_numpy(dtype=float)
    est = np.empty((n_boot, 1))
    real = np.empty((n_boot, 1)) if realised else None
    for k in range(n_boot):
        counts = np.repeat(j, rng.poisson(mean_j))
        d = _choice(rng, pool, counts.size)
        est[k] = _adpm_reliability(counts, d, T)
        if real is not None:
            # a mode seen j times has the chance Beta(j, n + T - j)
            p = rng.beta(counts, n + T - counts) if counts.size else counts
            real[k] = np.prod(1 - (1 - d) * p) * np.exp(
                -_beta_unseen(rng, total * n, n + T)
            )
    return est, real
