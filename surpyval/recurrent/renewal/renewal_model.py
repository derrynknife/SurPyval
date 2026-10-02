import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._summary import format_table
from surpyval.utils.linalg import (
    bound_signs,
    numerical_hessian,
    wald_bound_on_support,
)
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.validation import alpha_ci_error, option_error
from surpyval.utils.warnings import warn_no_covariance

#: The values of the restoration parameter at which each family is a
#: perfect and a minimal repair process (#513): the Kijima ``q`` of the
#: generalized renewal process is 0 as good as new and 1 as bad as old;
#: the ARA ``rho`` is 1 as good as new and 0 as bad as old; the ARI
#: ``rho = 1`` removes all the intensity at each failure (the most an ARI
#: repair can; it is not a renewal process) and ``rho = 0`` is minimal
#: repair; the G1 ``q = 0`` is a renewal process and G1 has no minimal
#: repair (``None``). ``label`` names the perfect end in the conclusions.
_REPAIR_VALUES: dict[str, dict[str, Any]] = {
    "Generalized Renewal": {
        "perfect": 0.0,
        "minimal": 1.0,
        "label": "perfect",
    },
    "G1 Renewal": {"perfect": 0.0, "minimal": None, "label": "perfect"},
    "ARA Renewal": {"perfect": 1.0, "minimal": 0.0, "label": "perfect"},
    "ARI Recurrence": {
        "perfect": 1.0,
        "minimal": 0.0,
        "label": "maximal",
    },
}


@dataclass(frozen=True)
class RestrictedRepairFit:
    """One likelihood-ratio test of :meth:`RenewalModel.repair_test`: the
    model refitted with its restoration parameter held at one kind of
    repair, and the test of the full fit against it.

    Attributes
    ----------
    hypothesis : str
        ``"perfect repair"`` (``"maximal repair"`` for ARI) or ``"minimal
        repair"``.
    parameter : str
        The restoration parameter's name, ``"q"`` or ``"rho"``.
    value : float
        The value it is held at.
    params : numpy.ndarray
        The restricted maximum-likelihood estimates of every parameter, in
        the order of the model's ``parameter_names`` (the first is
        ``value``); ``nan`` if the refit failed.
    log_likelihood : float
        The restricted maximum of the log-likelihood.
    statistic : float
        The likelihood-ratio statistic, ``2 * (full - restricted)``.
    df : int
        Its degrees of freedom, 1.
    p_value : float
        Its p-value: the chi-squared(1) tail, halved where ``value`` is on
        the edge of the parameter's range (``boundary``).
    boundary : bool
        Whether ``value`` is on the edge of the parameter's range; the
        statistic is then a 50:50 mixture of chi-squared(0) and
        chi-squared(1) under the hypothesis (Self and Liang, 1987).
    message : str
        Why the test is unavailable (the refit failed); empty otherwise.
    """

    hypothesis: str
    parameter: str
    value: float
    params: np.ndarray
    log_likelihood: float
    statistic: float
    df: int
    p_value: float
    boundary: bool
    message: str = ""

    @property
    def available(self) -> bool:
        """Whether the restricted refit succeeded and the test ran."""
        return not self.message

    def rejected(self, alpha_ci: float = 0.05) -> bool:
        """Whether the hypothesis is rejected at level ``alpha_ci``."""
        return self.available and self.p_value < alpha_ci


@dataclass(frozen=True)
class RepairTestResult:
    """The likelihood-ratio tests of an imperfect-repair model against
    perfect and minimal repair (:meth:`RenewalModel.repair_test`).

    Attributes
    ----------
    kind : str
        The model, e.g. ``"Generalized Renewal"``.
    parameter : str
        The restoration parameter's name, ``"q"`` or ``"rho"``.
    estimate : float
        Its fitted value.
    log_likelihood : float
        The fitted model's log-likelihood.
    perfect : RestrictedRepairFit
        The test against perfect repair (``q = 0``; ``rho = 1``; for ARI
        the maximal repair ``rho = 1``).
    minimal : RestrictedRepairFit or None
        The test against minimal repair (``q = 1``; ``rho = 0``); ``None``
        for the G1 renewal process, which has no minimal repair.
    alpha_ci : float
        The level at which ``conclusion`` rejects a hypothesis.
    conclusion : str
        What the tests say together, in one line.
    """

    kind: str
    parameter: str
    estimate: float
    log_likelihood: float
    perfect: RestrictedRepairFit
    minimal: "RestrictedRepairFit | None"
    alpha_ci: float
    conclusion: str

    def __repr__(self) -> str:
        title = "Repair test (likelihood ratio)"
        lines = [
            title,
            "=" * len(title),
            "Model               : {}".format(self.kind),
            "Estimate            : {} = {:.4g}".format(
                self.parameter, self.estimate
            ),
            "Log-likelihood      : {:.6g}".format(self.log_likelihood),
            "Tests               :",
        ]
        rows = []
        tests = [t for t in (self.perfect, self.minimal) if t is not None]
        for test in tests:
            value = "{} = {:g}".format(test.parameter, test.value)
            if not test.available:
                rows.append([test.hypothesis, value, "nan", "nan", "nan", ""])
                continue
            rows.append(
                [
                    test.hypothesis,
                    value,
                    "{:.6g}".format(test.log_likelihood),
                    "{:.4g}".format(test.statistic),
                    "{:.4g}".format(test.p_value),
                    "yes" if test.boundary else "no",
                ]
            )
        table = pd.DataFrame(
            rows,
            columns=[
                "hypothesis",
                "value",
                "log-lik",
                "LR (df 1)",
                "p-value",
                "on edge",
            ],
        )
        lines += [
            "    " + line for line in table.to_string(index=False).split("\n")
        ]
        notes = ["Note: " + t.message for t in tests if not t.available]
        notes.append(
            "Conclusion ({:g}% level): {}".format(
                100 * self.alpha_ci, self.conclusion
            )
        )
        lines += [
            textwrap.fill(note, width=70, subsequent_indent="      ")
            for note in notes
        ]
        return "\n".join(lines)


def _repair_conclusion(
    perfect: RestrictedRepairFit,
    minimal: "RestrictedRepairFit | None",
    estimate: float,
    label: str,
    alpha_ci: float,
) -> str:
    """The one-line conclusion of the repair tests at level ``alpha_ci``
    (see :meth:`RenewalModel.repair_test`)."""
    name = perfect.parameter
    if minimal is None:
        # G1: a renewal process (q = 0) or times between failures that
        # change by the factor 1 + q from one failure to the next.
        if not perfect.available:
            return "not available: " + perfect.message
        if not perfect.rejected(alpha_ci):
            return (
                "consistent with perfect repair (a renewal process); the "
                "G1 process has no minimal repair"
            )
        trend = "deterioration" if estimate < 0 else "improvement"
        return (
            "perfect repair (a renewal process) rejected: each time "
            "between failures is {:.4g} times the one before "
            "({})".format(1.0 + estimate, trend)
        )
    unavailable = [t for t in (perfect, minimal) if not t.available]
    if len(unavailable) == 2:
        return "not available: " + perfect.message
    if unavailable:
        # One test only: say what it says, and that the other is missing.
        (done,) = [t for t in (perfect, minimal) if t.available]
        verdict = "rejected" if done.rejected(alpha_ci) else "not rejected"
        return "{} {} ({} test not available)".format(
            done.hypothesis, verdict, unavailable[0].hypothesis
        )
    perfect_out = perfect.rejected(alpha_ci)
    minimal_out = minimal.rejected(alpha_ci)
    if not perfect_out and not minimal_out:
        return (
            "not determined: the data are consistent with both {} and "
            "minimal repair".format(label)
        )
    if perfect_out and not minimal_out:
        return "consistent with minimal repair; {} repair rejected".format(
            label
        )
    if minimal_out and not perfect_out:
        return "consistent with {} repair; minimal repair rejected".format(
            label
        )
    low, high = sorted((perfect.value, minimal.value))
    if low < estimate < high:
        where = "between {} and minimal repair".format(label)
    elif (estimate - minimal.value) * (minimal.value - perfect.value) > 0:
        where = "worse than minimal repair"
    else:
        where = "better than new"
    return "both {} and minimal repair rejected: {} = {:.4g} is {}".format(
        label, name, estimate, where
    )


#: Below this cumulative hazard the quantile function is accurate enough to
#: invert it: ``1 - p = exp(-H)`` then carries a relative error of about
#: ``eps * exp(H)``, which is 5e-8 at 20 (an error of 3e-9 in ``H``).
_QF_HAZARD_LIMIT = 20.0

_MACHINE_EPS = float(np.finfo(float).eps)


def solve_bracketed(
    g: Callable,
    lo: np.ndarray,
    hi: np.ndarray,
    g_lo: np.ndarray,
    g_hi: np.ndarray,
    xtol: float = 0.0,
    rtol: float = 4 * _MACHINE_EPS,
    maxiter: int = 400,
) -> np.ndarray:
    """
    Solve many bracketed root problems ``g_k(x) = 0`` at once, each with
    ``g_k(lo_k) < 0 < g_k(hi_k)``. ``g(x, sel)`` evaluates the problems
    whose indices are ``sel`` at the points ``x``.

    Each step takes a regula falsi point with the Illinois modification
    (halving the value kept at an end that has not moved for two steps),
    and bisects instead whenever the last two steps together did not halve
    the bracket, so every problem converges at least half as fast as
    bisection, and superlinearly once regula falsi takes hold. No point is
    taken within the tolerance of an end, so the last step closes the
    bracket instead of creeping up on the root. A problem is
    done when its bracket is narrower than ``xtol + rtol * |x|``; the
    midpoint of the final bracket is returned.
    """
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    g_lo = np.array(g_lo, dtype=float)
    g_hi = np.array(g_hi, dtype=float)
    root = lo + 0.5 * (hi - lo)
    moved = np.zeros(lo.size, dtype=int)  # -1 lo moved last, +1 hi did
    bisect = np.zeros(lo.size, dtype=bool)
    # The bracket's width two steps back.
    earlier = hi - lo
    active = np.flatnonzero(
        hi - lo > xtol + rtol * np.maximum(np.abs(lo), np.abs(hi))
    )
    for _ in range(maxiter):
        if not active.size:
            break
        a = active
        lo_a, hi_a, gl, gh = lo[a], hi[a], g_lo[a], g_hi[a]
        width = hi_a - lo_a
        mid = lo_a + 0.5 * width
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            falsi = hi_a - gh * width / (gh - gl)
            x = np.where(bisect[a] | ~np.isfinite(falsi), mid, falsi)
        # Never evaluate within the tolerance of an end (Brent's tolerance
        # step). Once one end has converged, regula falsi lands on it (or
        # past it, by rounding); a point just inside it lets the next sign
        # change close the bracket.
        tol = 0.5 * (xtol + rtol * np.maximum(np.abs(lo_a), np.abs(hi_a)))
        x = np.clip(x, lo_a + tol, hi_a - tol)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gx = np.asarray(g(x, a), dtype=float)
        exact = gx == 0
        below = gx < 0
        # A NaN counts as past the root, so the bracket still shrinks.
        above = ~below & ~exact
        new_lo = np.where(below, x, lo_a)
        new_hi = np.where(above, x, hi_a)
        new_gl = np.where(below, gx, gl)
        new_gh = np.where(above, gx, gh)
        new_gh = np.where(below & (moved[a] == -1), 0.5 * new_gh, new_gh)
        new_gl = np.where(above & (moved[a] == 1), 0.5 * new_gl, new_gl)
        moved[a] = np.where(below, -1, np.where(above, 1, 0))
        lo[a], hi[a], g_lo[a], g_hi[a] = new_lo, new_hi, new_gl, new_gh
        new_width = new_hi - new_lo
        bisect[a] = new_width > 0.5 * earlier[a]
        earlier[a] = width
        root[a] = np.where(exact, x, new_lo + 0.5 * new_width)
        done = exact | (
            new_width
            <= xtol + rtol * np.maximum(np.abs(new_lo), np.abs(new_hi))
        )
        active = a[~done]
    return root


def conditional_gaps(
    lifetime: Any, v: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Draw the time to the next failure of items whose virtual ages are
    ``v``, from the uniforms ``u``, for a virtual-age renewal model with
    the lifetime distribution ``lifetime``.

    The residual life ``X`` from age ``v`` has ``P(X > x) = S(v + x) /
    S(v)``, so ``H(v + X) = H(v) - log(u)``: the draw adds an Exp(1) amount
    to the cumulative hazard ``H`` and inverts it. Working with ``H``
    rather than the survival function keeps this exact at long horizons.
    The old draw, ``qf(1 - u * sf(v)) - v``, lost all precision once
    ``sf(v)`` fell below about 1e-16 (an expected count of about 37), so
    the next age came back as ``v`` itself and the simulated MCF went flat.

    ``H`` is inverted with the quantile function while that is accurate
    (see ``_QF_HAZARD_LIMIT``) and by root finding beyond it.
    """
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    hazard, quantile = _lifetime_functions(lifetime)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        h_v = np.broadcast_to(hazard(v), v.shape)
        target = h_v - np.log(u)
    # No survival past age v (the end of a bounded support): the item
    # fails at once.
    at_once = np.isinf(h_v) & (h_v > 0)
    # u == 0 (or an undefined H): an infinitely late event.
    never = ~at_once & ~np.isfinite(target)
    easy = ~at_once & ~never & (target <= _QF_HAZARD_LIMIT)
    hard = ~at_once & ~never & ~easy
    x = np.array(v, dtype=float)
    if easy.any():
        x[easy] = quantile(-np.expm1(-target[easy]))
    if hard.any():
        x[hard] = _invert_cumulative_hazard(hazard, v[hard], target[hard])
    gap = np.maximum(x - v, 0.0)
    gap[at_once] = 0.0
    gap[never] = np.inf
    return gap


def _lifetime_functions(lifetime: Any) -> "tuple[Callable, Callable]":
    """``H`` and the quantile function of ``lifetime``, on arrays. For a
    plain model (no offset, cure fraction or zero-inflation -- what the
    renewal fitters build) the distribution's own functions are called
    directly, skipping the model methods' argument handling."""
    if (
        getattr(lifetime, "p", None) == 1
        and getattr(lifetime, "f0", None) == 0
        and not getattr(lifetime, "gamma", 0)
    ):
        dist = lifetime.dist
        params = [float(p) for p in lifetime.params]
        lower = dist.support[0]

        def hazard(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            below = x < lower
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                h = dist.Hf(np.where(below, lower, x), *params)
            return np.where(below, 0.0, np.asarray(h, dtype=float))

        def quantile(p: np.ndarray) -> np.ndarray:
            return np.asarray(dist.qf(p, *params), dtype=float)

        return hazard, quantile

    def model_hazard(x: np.ndarray) -> np.ndarray:
        return np.asarray(lifetime.Hf(x), dtype=float)

    def model_quantile(p: np.ndarray) -> np.ndarray:
        return np.asarray(lifetime.qf(p), dtype=float)

    return model_hazard, model_quantile


def _invert_cumulative_hazard(
    hazard: Callable, v: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Solve ``H(x_k) = target_k`` for ``x_k > v_k`` (where ``H(v_k) <
    target_k``), for every ``k`` at once."""

    def g(x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return hazard(x) - target[sel]

    n = v.size
    everything = np.arange(n)
    lo = np.array(v, dtype=float)
    step = np.maximum(np.abs(lo), 1.0)
    hi = lo + step
    growing = everything
    for _ in range(2000):
        if not growing.size:
            break
        growing = growing[g(hi[growing], growing) < 0]
        lo[growing] = hi[growing]
        step[growing] *= 2.0
        hi[growing] = lo[growing] + step[growing]
    # Past a bounded support H can be inf or nan; bisect until hi is a
    # finite point at or above the target so the root is bracketed.
    g_hi = g(hi, everything)
    for _ in range(200):
        bad = np.flatnonzero(~np.isfinite(g_hi))
        if not bad.size:
            break
        mid = 0.5 * (lo[bad] + hi[bad])
        up = g(mid, bad) < 0
        lo[bad[up]] = mid[up]
        hi[bad[~up]] = mid[~up]
        g_hi[bad] = g(hi[bad], bad)
    # H jumps to infinity at the end of the support: fail there.
    out = np.where(np.isfinite(g_hi), hi, lo)
    rest = np.flatnonzero(np.isfinite(g_hi) & (g_hi != 0))
    if rest.size:
        out[rest] = solve_bracketed(
            lambda x, sel: g(x, rest[sel]),
            lo[rest],
            hi[rest],
            g(lo[rest], rest),
            g_hi[rest],
            xtol=4 * _MACHINE_EPS * 1e-300,
        )
    return out


def event_positions(item: np.ndarray) -> np.ndarray:
    """
    The position of each row within its own item: 0 for an item's first row,
    1 for its second, and so on, in the order the rows appear.

    The imperfect-repair likelihoods need it to evaluate their per-item
    recursions for all the items at once, one event position at a time,
    rather than one item and one event at a time.
    """
    item = np.asarray(item)
    if item.size == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(item, kind="stable")
    ordered = item[order]
    new_item = np.empty(item.size, dtype=bool)
    new_item[0] = True
    new_item[1:] = ordered[1:] != ordered[:-1]
    first = np.flatnonzero(new_item)
    counts = np.diff(np.append(first, item.size))
    position = np.empty(item.size, dtype=int)
    position[order] = np.arange(item.size) - np.repeat(first, counts)
    return position


def rows_by_position(position: np.ndarray) -> "list[np.ndarray]":
    """
    The row indices at each event position (see ``event_positions``):
    element ``k`` holds, in row order, the rows that are the ``k``-th of
    their item. With the rows grouped by item, ``rows - 1`` are then the
    same items' rows at position ``k - 1``.
    """
    position = np.asarray(position, dtype=int)
    if position.size == 0:
        return []
    order = np.argsort(position, kind="stable")
    counts = np.bincount(position)
    return np.split(order, np.cumsum(counts)[:-1])


class DiscountedMemory:
    """
    For sequences simulated together, the discounted sum
    ``sum_{j < min(m, k)} (1 - rho)^j h_{k - j}`` of the values ``h_1 ..
    h_k`` recorded so far, newest first: the memory term of the ARA (with
    the arrival times) and ARI (with the intensities at the failures)
    models.

    ``record`` is called once per simulation round with the sequences still
    running, so every sequence it covers has the same number of values. With
    infinite memory the sum is carried forward, ``S_k = h_k + (1 - rho)
    S_{k-1}``; with memory ``m`` the last ``m`` rounds are kept and summed.
    """

    def __init__(self, n: int, rho: float, m: "int | float") -> None:
        self.n = n
        self.decay = 1.0 - rho
        self.infinite = bool(np.isinf(m))
        self.m = 0 if self.infinite else int(m)
        self.rounds = 0
        self.total = np.zeros(n)
        self.recent: list = []

    def value(self, idx: np.ndarray) -> np.ndarray:
        if self.infinite:
            return self.total[idx]
        out = np.zeros(idx.size)
        for j, row in enumerate(reversed(self.recent)):
            out += self.decay**j * row[idx]
        return out

    def record(self, idx: np.ndarray, values: np.ndarray) -> None:
        self.rounds += 1
        if self.infinite:
            self.total[idx] = values + self.decay * self.total[idx]
            return
        row = np.full(self.n, np.nan)
        row[idx] = values
        self.recent.append(row)
        if len(self.recent) > self.m:
            self.recent.pop(0)


class RenewalModel(
    SerialisableMixin, RecurrenceSimulationMixin, LikelihoodInferenceMixin
):
    """
    A fitted renewal / imperfect-repair recurrence model.

    This is the model object returned by the renewal-family fitters
    (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``, ``ARI``), in
    the same way that the intensity fitters (``CrowAMSAA``, ``Duane``, ...)
    return a ``ParametricRecurrenceModel``. It holds the fitted underlying
    lifetime distribution and the restoration parameter, and provides the
    simulation (``mcf``, ``plot``, ``count_terminated_simulation``,
    ``time_terminated_simulation``) and likelihood-inference
    (``log_likelihood``, ``aic``, ``bic``, ``standard_errors``) behaviour via
    the shared mixins.

    These processes have no closed-form intensity, so the mean cumulative
    function is obtained by simulation.

    Parameters
    ----------
    model : Parametric
        The fitted underlying lifetime distribution.
    restoration : float
        The fitted restoration / repair parameter (``q`` for the generalized
        and G1 renewal processes, ``rho`` for ARA and ARI).
    restoration_name : str
        The attribute/label name of the restoration parameter (e.g. ``"q"`` or
        ``"rho"``); it is also exposed as an attribute of that name.
    restoration_label : str
        Human-readable label used in ``__repr__`` (e.g. ``"Restoration
        Factor"`` or ``"Repair Efficiency"``).
    kind : str
        Display name of the process (e.g. ``"Generalized Renewal"``).
    sampler_factory : callable
        ``sampler_factory(model, n) -> step`` returning the simulation
        sampler for ``n`` sequences simulated together: ``step(idx, u)``
        draws the next interarrival times of the sequences ``idx`` from the
        uniforms ``u`` (see
        :func:`surpyval.recurrent.simulation.simulate_sequences`).
    restoration_bounds : tuple, optional
        Natural-space ``(lower, upper)`` bounds of the restoration parameter
        (e.g. ``(0, 1)`` for ARA/ARI's ``rho``), used by ``param_cb`` to pick
        a transform that keeps its confidence bounds inside the support.

    Notes
    -----
    ``params`` is every parameter of the model in one vector, in the order
    of ``parameter_names``: the restoration parameter (``q`` or ``rho``)
    first, then the lifetime distribution's parameters (for ARI, the
    baseline intensity's) -- the order of :meth:`covariance`,
    :meth:`standard_errors` and ``param_cb`` too. The restoration parameter is
    also ``restoration`` (and ``q`` or ``rho``), and the distribution's
    parameters ``model.params``.

    Examples
    --------
    ``ARA.fit`` returns one. Two systems, repaired at each failure and
    observed to time 60:

    >>> import numpy as np
    >>> from surpyval.recurrent import ARA
    >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
    >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    >>> model = ARA.fit(x, i, c=c, m=2)

    The repair efficiency ``rho`` is 1 (as good as new), and the Weibull
    is that of the times between failures:

    >>> round(float(model.rho), 3)
    1.0
    >>> model.model.params.round(3)
    array([13.779,  1.917])
    >>> model.parameter_names
    ['rho', 'alpha', 'beta']
    >>> model.params.round(3)
    array([ 1.   , 13.779,  1.917])

    The expected number of failures per system by times 20 and 60, by
    simulation:

    >>> model.mcf(np.array([20.0, 60.0]), items=1000, random_state=0).round(3)
    array([1.296, 4.56 ])
    """

    # Set by the GeneralizedRenewal fitter for its Kijima sampler.
    _virtual_age_function: Any

    #: Set by the fitter for the generalized-renewal family (``"i"``/``"ii"``);
    #: absent otherwise.
    kijima_type: Any
    #: Set by the fitter for the ARA/ARI families (memory); absent otherwise.
    m: Any
    #: How the parameters were obtained: ``"MLE"`` for a fit, otherwise
    #: given (``fit_from_parameters``). Kept through serialisation.
    how: str = "from_params"

    def __init__(
        self,
        model: Any,
        restoration: float,
        restoration_name: str,
        restoration_label: str,
        kind: str,
        sampler_factory: Callable,
        dist_label: str = "Distribution",
        restoration_bounds: tuple = (None, None),
    ) -> None:
        self.model = model
        self.restoration = restoration
        self._restoration_param_name = restoration_name
        self._restoration_label = restoration_label
        self._restoration_bounds = restoration_bounds
        self._dist_label = dist_label
        self.kind = kind
        self._sampler_factory = sampler_factory
        # Expose the restoration parameter under its conventional name
        # (``q``/``rho``) so existing usage keeps working.
        setattr(self, restoration_name, restoration)
        # What a maximum-likelihood fit reached, one of ``MAXIMUM_STATES``
        # (``surpyval.utils.no_maximum``), as its warnings say: set by the
        # fit; "not applicable" for a model built from its parameters,
        # "unknown" for one restored from a dict saved without it.
        self.maximum = "not applicable"

    # -- serialisation -----------------------------------------------------

    def _family(self) -> str:
        """Identify the renewal family from the fitted attributes."""
        if getattr(self, "kijima_type", None) is not None:
            return "GeneralizedRenewal"
        if getattr(self, "m", None) is not None:
            if self._dist_label == "Baseline Intensity":
                return "ARI"
            return "ARA"
        return "GeneralizedOneRenewal"

    def to_dict(self) -> dict:
        """
        Serialise this fitted renewal / imperfect-repair model to a plain,
        JSON-serialisable dict.

        These processes have no closed-form intensity -- their simulation is
        driven by a sampler closure built from the underlying distribution and
        the restoration parameter -- so what is stored is the family, the
        underlying distribution (by name) and its parameters, the restoration
        parameter, and the family's discrete option (``kijima_type`` for the
        generalized renewal, memory ``m`` for ARA/ARI). On load the family's
        fitter rebuilds the sampler from those, so ``mcf`` and the simulation
        methods reproduce exactly. The likelihood/data state is not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict = {
            "model": "RenewalModel",
            "family": self._family(),
            "dist": self.model.dist.name,
            "params": np.asarray(self.model.params, dtype=float).tolist(),
            "restoration": float(self.restoration),
            "how": self.how,
            **maximum_entry(self.maximum),
        }
        if getattr(self, "kijima_type", None) is not None:
            out["kijima_type"] = self.kijima_type
        if getattr(self, "m", None) is not None:
            out["m"] = self.m
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "RenewalModel":
        """
        Rebuild a renewal model from a :meth:`to_dict` dictionary.

        The family's fitter is used to reconstruct the model from parameters
        (which regenerates the simulation sampler), so the result predicts
        identically to the original.

        See Also
        --------
        to_dict, to_json, from_json
        """
        import surpyval.recurrent as recurrent

        require_model_tag(model_dict, "RenewalModel", "a renewal model")
        family = model_dict["family"]
        fitters: "dict[str, Any]" = {
            "GeneralizedRenewal": recurrent.GeneralizedRenewal,
            "GeneralizedOneRenewal": recurrent.GeneralizedOneRenewal,
            "ARA": recurrent.ARA,
            "ARI": recurrent.ARI,
        }
        if family not in fitters:
            raise ValueError("Unknown renewal family {!r}".format(family))
        out = cls._rebuild(fitters[family], family, model_dict)
        # A reloaded fit still says it was fitted by MLE (it carries no
        # likelihood, as a reloaded parametric model does not).
        out.how = model_dict.get("how", "from_params")
        out.maximum = restored_maximum(model_dict)
        return out

    @staticmethod
    def _rebuild(fitter: Any, family: str, model_dict: dict) -> "RenewalModel":
        from surpyval.recurrent.serialisation import intensity_dist_by_name

        params = model_dict["params"]
        restoration = model_dict["restoration"]

        if family == "ARI":
            # The ARI baseline is a recurrence intensity model, stored
            # under "dist" as every family's underlying model is (the file
            # layout did not change when ARI's argument became
            # ``baseline``, #507).
            baseline = intensity_dist_by_name(model_dict["dist"])
            return fitter.fit_from_parameters(
                params, restoration, m=model_dict["m"], baseline=baseline
            )

        import surpyval
        from surpyval.univariate.parametric.parametric_fitter import (
            ParametricFitter,
        )

        # Restrict the lookup to known distribution fitters so an
        # untrusted model dict cannot resolve arbitrary surpyval
        # attributes (matches the guard in Parametric.from_dict).
        dist = getattr(surpyval, model_dict["dist"], None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                "Unknown distribution {!r}".format(model_dict["dist"])
            )
        if family == "GeneralizedRenewal":
            return fitter.fit_from_parameters(
                params,
                restoration,
                kijima=model_dict["kijima_type"],
                dist=dist,
            )
        if family == "ARA":
            return fitter.fit_from_parameters(
                params, restoration, m=model_dict["m"], dist=dist
            )
        # GeneralizedOneRenewal
        return fitter.fit_from_parameters(params, restoration, dist=dist)

    def _new_batch_sampler(self, n: int) -> Callable:
        return self._sampler_factory(self, n)

    @property
    def params(self) -> np.ndarray:
        """Every parameter of the model, in the order of ``parameter_names``:
        the restoration parameter, then the distribution's parameters."""
        return np.concatenate(
            [[self.restoration], np.asarray(self.model.params, dtype=float)]
        ).astype(float)

    def _parameter_names(self) -> list:
        # The restoration parameter (``q``/``rho``) leads ``_mle``, followed by
        # the underlying lifetime/intensity model's parameters.
        return [self._restoration_param_name, *self.model.dist.parameter_names]

    def _parameter_bounds(self) -> list:
        return [self._restoration_bounds, *self.model.dist.bounds]

    def residuals(self, kind: str = "cumulative_hazard") -> np.ndarray:
        """
        Residual diagnostics for the fitted imperfect-repair model, from the
        time-rescaling theorem applied to the process's *conditional*
        intensity (each interarrival is rescaled by the cumulative hazard
        accumulated over it given the model's virtual age / intensity
        reduction), so they extend the counting-process residuals to the
        renewal / virtual-age families.

        Parameters
        ----------

        kind: {'cumulative_hazard', 'pit', 'martingale'}, optional
            ``'cumulative_hazard'`` returns the rescaled interarrival
            increments of every observed event (pooled across items); see
            below for how far they are iid Exp(1). ``'pit'`` applies the
            probability integral transform ``1 - exp(-e)`` to those residuals
            (U(0, 1) under the same conditions). ``'martingale'`` returns
            one residual per item: its observed event count minus the
            compensator (the sum of the rescaled increments) accumulated
            over its observation.

            Only complete gaps (event to event) are returned. When an
            item's observation ends at a window close rather than at an
            event, its final gap is censored and left out, and that
            selection makes the returned residuals smaller than Exp(1) on
            average -- noticeably so with few events per item (a mean
            near 0.66 with about three events per item). So they are
            exactly iid Exp(1) only for failure-truncated items; otherwise
            read a Q-Q plot against Exp(1) with this downward bias in
            mind, or use ``cramer_von_mises``, which conditions on each
            item's window correctly.

        Returns
        -------

        numpy array
            The residuals.
        """
        self._check_has_data("residuals")
        from surpyval.recurrent import diagnostics

        diagnostics._validate_diagnostic_data(self.data, "Residuals")
        increments = np.asarray(
            self._fitter._rescaled_increments(self, self.data), dtype=float
        )
        c = np.asarray(self.data.c)

        if kind in ("cumulative_hazard", "pit"):
            e = increments[c == 0]
            if kind == "pit":
                return 1.0 - np.exp(-e)
            return e
        elif kind == "martingale":
            residuals = []
            for item in np.unique(self.data.i):
                mask = self.data.i == item
                observed = int((c[mask] == 0).sum())
                residuals.append(observed - float(increments[mask].sum()))
            return np.array(residuals)
        raise option_error(
            "kind", kind, ("cumulative_hazard", "pit", "martingale")
        )

    def trend_test(
        self,
        test: str = "laplace",
        alternative: str = "two-sided",
        *,
        alpha_ci: float = 0.05,
    ) -> Any:
        """
        Run a trend test on the data this model was fitted to. The null
        hypothesis is a *homogeneous* Poisson process (no trend); the statistic
        uses only the event times and windows, not the fitted model, so it
        checks whether an imperfect-repair model was warranted at all.

        Parameters
        ----------

        test: {'laplace', 'mil_hdbk_189c'}, optional
            The trend test to run. Default is 'laplace'.
        alternative: {'two-sided', 'increasing', 'decreasing'}, optional
            The alternative hypothesis. Default is 'two-sided'.
        alpha_ci: float, optional
            The significance level at which the result's ``trend`` is
            judged (default 0.05, keyword only): a trend is named only when
            ``p_value < alpha_ci``.

        Returns
        -------

        TrendTestResult
            The test result, carrying the statistic, p-value, the
            direction of the statistic and the trend concluded at
            ``alpha_ci``.
        """
        self._check_has_data("trend_test")
        from surpyval.recurrent import diagnostics

        return diagnostics.trend_test(
            self.data, test=test, alternative=alternative, alpha_ci=alpha_ci
        )

    def cramer_von_mises(
        self, n_boot: int = 200, random_state: "int | None" = None
    ) -> Any:
        """
        Cramer-von Mises goodness-of-fit test of the fitted imperfect-repair
        model.

        These processes have no marginal cumulative intensity, so the
        conditionally-uniform transforms use the compensator built from each
        interval's rescaled increment (the conditional-intensity residual):
        conditional on an item's number of events, ``Lambda(t_k) /
        Lambda(close)`` are iid U(0, 1) when the fitted model is the true one,
        and the statistic measures their departure from uniformity. Because the
        restoration and lifetime / intensity parameters were estimated from the
        same data, the p-value is a parametric bootstrap: each item is
        resimulated from the fitted model the way it was observed -- an
        item whose last row is an end-of-observation (``c=1``) row over the
        same window, with however many events the model gives it there, and
        a failure-truncated item with its observed number of events -- the
        full model is refitted, and the statistic recomputed. Each replicate
        is a multi-start optimisation, so this is much slower than the
        residual diagnostics.

        Parameters
        ----------

        n_boot: int, optional
            Number of bootstrap replicates for the p-value. Default is 200.
        random_state: int or numpy.random.Generator, optional
            Seed for a reproducible p-value.

        Returns
        -------

        GoodnessOfFitResult
            The observed statistic and its bootstrap p-value.
        """
        self._check_has_data("cramer_von_mises")
        from surpyval.recurrent import diagnostics

        return diagnostics.cramer_von_mises_renewal(
            self, n_boot=n_boot, random_state=random_state
        )

    def __repr__(self) -> str:
        title = f"{self.kind} SurPyval Model"
        lines = [
            title,
            "=" * len(title),
            "{:<20}: {}".format(self._dist_label, self.model.dist.name),
            "Fitted by           : "
            + (
                "MLE"
                if hasattr(self, "_neg_ll") or self.how == "MLE"
                else "given parameters (not fitted)"
            ),
        ]
        if getattr(self, "kijima_type", None) is not None:
            lines.append(f"Kijima Type         : {self.kijima_type}")
        if getattr(self, "m", None) is not None:
            lines.append(f"Memory (m)          : {self.m}")
        lines.append(
            "{:<20}: {}".format(self._restoration_label, self.restoration)
        )
        if not hasattr(self, "_neg_ll"):
            # Built from parameters (or restored): no standard errors.
            param_string = "\n".join(
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(
                    self.model.params, self.model.dist.parameter_names
                )
            )
            return (
                "\n".join(lines)
                + "\nParameters          :\n"
                + param_string
                + "\nRepair test         : not available (no data)"
            )
        table = self.summary()
        lines.append("Parameters          : Wald 95% intervals")
        lines.append(format_table(table, list(table.columns)))
        note = self._edge_note(table)
        if note:
            lines.append(note)
        lines.append(self._repair_line())
        return "\n".join(lines)

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The fitted parameters with their standard errors and Wald
        intervals, one row per entry of ``parameter_names``.

        The intervals are those of :meth:`param_cb`: on the log scale for
        a parameter bounded below (the Kijima ``q``, a positive scale), on
        the logit scale for one bounded on both sides (``rho`` of ARA and
        ARI), so they stay inside the parameter's range. A restoration
        parameter at the edge of its range (a ``q`` driven to 0) has no
        standard error (``nan``); its interval is the profile-likelihood
        one, which starts at the edge, and the other parameters' are
        those of the model with it held there (#461).

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            Columns ``estimate``, ``se``, ``lower <level>`` and
            ``upper <level>``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>> model = GeneralizedRenewal.fit(x, i, c)
        >>> model.summary().round(3)
               estimate     se  lower 95%  upper 95%
        q         0.000    NaN      0.000      0.092
        alpha     2.399  0.287      1.898      3.033
        beta      2.754  0.652      1.732      4.379
        """
        self._check_fitted()
        with warnings.catch_warnings():
            # The boundary is reported in the table (nan), not warned.
            warnings.simplefilter("ignore")
            cov = self.covariance()
        level = "{:g}%".format(100 * (1 - alpha_ci))
        rows = []
        for k, (value, (lo, hi)) in enumerate(
            zip(self._mle, self._parameter_bounds())
        ):
            var = float(cov[k, k])
            if k == 0 and self._edge_value() is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cb = self.param_cb(self._restoration_param_name, alpha_ci)
                rows.append([value, np.nan, cb[0], cb[1]])
                continue
            if not var > 0:
                rows.append([value, np.nan, np.nan, np.nan])
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cb = wald_bound_on_support(
                    float(value), var, lo, hi, alpha_ci, "two-sided"
                )
            rows.append([value, np.sqrt(var), cb[0], cb[1]])
        return pd.DataFrame(
            rows,
            index=self.parameter_names,
            columns=["estimate", "se", f"lower {level}", f"upper {level}"],
            dtype=float,
        )

    def _restoration_at_edge(self) -> bool:
        """Whether the restoration parameter sits on a bound of its range
        (a ``q`` driven to 0), where a Wald interval does not hold."""
        return self._edge_value() is not None

    def _edge_value(self) -> "float | None":
        """The bound of its range the restoration parameter sits on
        (within 1e-6), or ``None``."""
        for edge in self._restoration_bounds:
            if edge is not None and abs(self.restoration - edge) < 1e-6:
                return float(edge)
        return None

    def covariance(self) -> np.ndarray:
        """
        Approximate parameter covariance matrix, ordered to match
        :attr:`parameter_names`: the inverse of the numerical Hessian of
        the negative log-likelihood at the MLE.

        Where the restoration parameter sits on the edge of its range (a
        ``q`` driven to 0, an ARA or ARI ``rho`` at 1), it has no Wald
        variance: its row and column are ``nan``, and :meth:`param_cb`
        gives its profile-likelihood interval. The other parameters'
        covariance is then the inverse information with it held on the
        edge, the model the data reached. The Hessian over every
        parameter took steps across the edge, where the likelihood is not
        the model's, and gave negative variances to the others too (-10.8
        for a Weibull ``alpha``, #461).
        """
        self._check_fitted()
        edge = self._edge_value()
        if edge is None:
            return super().covariance()
        mle = self._mle_values()

        def neg_ll_rest(params: np.ndarray) -> float:
            return self._neg_ll(np.r_[mle[0], params])

        H = numerical_hessian(neg_ll_rest, mle[1:])
        n = mle.size
        out = np.full((n, n), np.nan)
        if not np.all(np.isfinite(H)):
            warn_no_covariance()
            return out
        try:
            out[1:, 1:] = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            warn_no_covariance()
        return out

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bound(s) on a fitted parameter.

        Wald bounds from the observed information, on a scale chosen from
        the parameter's range so they stay inside it: log for one bounded
        below (the Kijima ``q``, a positive scale), logit for one bounded
        on both sides (``rho`` of ARA and ARI), natural otherwise.

        A restoration parameter on the edge of its range (a ``q`` driven
        to 0; an ARA or ARI ``rho`` at 1 or 0) has no Wald interval: the
        likelihood is not regular there and its variance is not defined.
        Its interval is then the profile-likelihood one (#461): the values
        the likelihood-ratio test does not reject, the restricted model
        refitted at each, as :meth:`repair_test` refits it at the perfect
        and minimal values. It is one-sided, from the edge to where twice
        the drop in the profile log-likelihood reaches the chi-squared(1)
        quantile for the level (the whole range if it never does), so a
        one-sided bound towards the edge is the edge itself. The other
        parameters' Wald bounds are those of the model with the
        restoration parameter held on the edge (see :meth:`covariance`).

        Parameters
        ----------

        name : str
            The parameter to bound; one of :attr:`parameter_names`.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.

        Returns
        -------

        numpy array
            The confidence bound(s) on the parameter.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>> model = GeneralizedRenewal.fit(x, i, c)
        >>> model.param_cb("q").round(3)
        array([0.   , 0.092])
        >>> model.param_cb("q", bound="lower").round(3)
        array([0.])
        """
        edge = self._edge_value()
        if name != self._restoration_param_name or edge is None:
            return super().param_cb(name, alpha_ci, bound)
        self._check_fitted()
        if not 0 < alpha_ci < 1:
            raise alpha_ci_error(alpha_ci)
        alpha, signs = bound_signs(alpha_ci, bound)
        crit = float(chi2.ppf(1.0 - 2.0 * alpha, 1)) if alpha < 0.5 else 0.0
        lower, upper = self._restoration_bounds
        # Away from the edge: up from a lower edge, down from an upper one.
        away = 1.0 if edge == lower else -1.0
        out = np.full(signs.shape, edge)
        far = signs == away
        if far.any():
            out[far] = self._profile_end(edge, crit, away)
        return out

    def _profile_ll(self, value: float) -> float:
        """The profile log-likelihood at restoration parameter ``value``:
        the likelihood maximised over the other parameters with it held
        there (``-inf`` if that fails). Kept on the model by value."""
        cache = self.__dict__.setdefault("_profile_cache", {})
        if value in cache:
            return cache[value][0]
        starts = [np.asarray(self._mle[1:], dtype=float)]
        if cache:
            # The nearest value's maximum is the best start.
            near = min(cache, key=lambda v: abs(v - value))
            if np.all(np.isfinite(cache[near][1])):
                starts.insert(0, cache[near][1])
        fun, params = self._restricted_maximum(value, starts)
        cache[value] = (-fun, params)
        return -fun

    def _profile_end(self, edge: float, crit: float, away: float) -> float:
        """The end, away from ``edge``, of the profile-likelihood interval:
        where twice the drop of the profile log-likelihood from its
        maximum reaches ``crit``; the far end of the range if it never
        does."""
        ll_hat = self.log_likelihood

        def excess(value: float) -> float:
            drop = 2.0 * (ll_hat - self._profile_ll(value))
            # A failed profile fit is far outside the interval.
            return drop - crit if np.isfinite(drop) else np.inf

        lower, upper = self._restoration_bounds
        far = upper if away > 0 else lower
        if far is not None:
            if excess(far) <= 0:
                return float(far)
            return float(brentq(excess, edge, far, xtol=1e-8, rtol=1e-8))
        # Unbounded away from the edge (the Kijima q): bracket by doubling.
        inner, step = edge, 0.01
        for _ in range(60):
            outer = edge + away * step
            if excess(outer) > 0:
                return float(
                    brentq(excess, inner, outer, xtol=1e-8, rtol=1e-8)
                )
            inner, step = outer, 2.0 * step
        return float(away * np.inf)

    def _edge_note(self, table: pd.DataFrame) -> str:
        """A note when the restoration parameter has no standard error:
        at the edge of its range (a ``q`` driven to 0), or where the
        likelihood has lost its curvature."""
        name = self._restoration_param_name
        est, se = table.loc[name, ["estimate", "se"]].to_numpy()
        if np.isfinite(se):
            return ""
        if self._restoration_at_edge():
            note = (
                f"Note: {name} = {est:.4g} is at the edge of its range, "
                "so it has no standard error. Its interval is the "
                "profile-likelihood one, from the edge; the others' are "
                f"Wald intervals with {name} held there."
            )
        else:
            note = (
                f"Note: {name} = {est:.4g} is where the likelihood's "
                "curvature is lost, so it has no standard error or "
                "interval."
            )
        return textwrap.fill(note, width=70, subsequent_indent="      ")

    def _repair_line(self) -> str:
        """The repair tests' conclusion as the printed model shows it,
        with the two p-values. Never raises: a test that cannot be run
        is reported as not available."""
        try:
            result = self.repair_test()
        except Exception as error:  # pragma: no cover - defensive
            text = f"Repair test: not available ({error})"
        else:
            p_values = "; ".join(
                f"{t.parameter} = {t.value:g}: p = {t.p_value:.3g}"
                for t in (result.perfect, result.minimal)
                if t is not None and t.available
            )
            text = "Repair test: " + result.conclusion
            if p_values:
                text += f" (LR tests, {p_values})"
        return textwrap.fill(text, width=70, subsequent_indent="      ")

    def repair_test(self, alpha_ci: float = 0.05) -> RepairTestResult:
        """
        The likelihood-ratio tests of the fitted repair quality against
        perfect repair ("as good as new") and minimal repair ("as bad as
        old").

        Each test refits the model with the restoration parameter held at
        that kind of repair and refers twice the difference in
        log-likelihood to a chi-squared distribution with one degree of
        freedom:

        - perfect repair: ``q = 0`` for the generalized (Kijima) and G1
          renewal processes, ``rho = 1`` for ARA -- an ordinary renewal
          process, the distribution fitted to the times between failures.
          ARI has no as-good-as-new repair; its ``rho = 1``, which removes
          all the intensity at each failure (the most an ARI repair can),
          is tested in its place and called maximal repair.
        - minimal repair: ``q = 1``, ``rho = 0`` -- the non-homogeneous
          Poisson process whose cumulative intensity is the lifetime
          distribution's cumulative hazard (the Crow-AMSAA power law for a
          Weibull), or ARI's baseline intensity. The G1 process has no
          minimal repair (its ``q`` scales the times between failures
          geometrically), so it has the perfect-repair test only.

        Where the value tested is on the edge of the parameter's range --
        the Kijima ``q = 0`` (``q >= 0``), and ``rho = 0`` and ``rho = 1``
        (``0 <= rho <= 1``) -- the statistic is a 50:50 mixture of
        chi-squared(0) and chi-squared(1) under the hypothesis, so the
        p-value is half the chi-squared(1) tail (Self and Liang, 1987),
        and 1 when the statistic is 0. The G1 ``q = 0`` and the Kijima
        ``q = 1`` are inside the range: the full chi-squared(1) tail.

        At level ``alpha_ci`` the conclusion is "not determined" when
        neither hypothesis is rejected (the data are consistent with
        both), "consistent with" the one not rejected when only the other
        is, and where the estimate lies (between perfect and minimal
        repair, or worse than minimal repair) when both are.

        The refits are made the first time the model is printed or this
        is called, and kept on the model. If a refit fails, its test is
        reported as not available (one warning) rather than raising.

        Parameters
        ----------
        alpha_ci : float, optional
            The significance level of the conclusion. Default 0.05.

        Returns
        -------
        RepairTestResult
            The two tests (``perfect`` and ``minimal``, each a
            :class:`RestrictedRepairFit` with the restricted fit, the
            statistic, its degrees of freedom and p-value) and the
            ``conclusion``.

        Raises
        ------
        ValueError
            For a model with no likelihood (built from parameters or
            restored from a dict).

        Examples
        --------
        Eight systems simulated under minimal repair: the fitted ``q`` is
        far from 1, but the test finds no evidence against minimal
        repair, and rejects perfect repair:

        >>> import numpy as np
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> rng = np.random.default_rng(8)
        >>> rows = []
        >>> for k in range(8):
        ...     T = rng.uniform(6000, 12000)
        ...     N = rng.poisson(2e-4 * T**1.35)
        ...     ts = np.sort(T * rng.random(N) ** (1 / 1.35))
        ...     rows += [(h, k, 0) for h in ts] + [(T, k, 1)]
        >>> x, i, c = map(np.array, zip(*rows))
        >>> model = GeneralizedRenewal.fit(x, i, c)
        >>> round(float(model.q), 2)
        2.63
        >>> test = model.repair_test()
        >>> round(test.minimal.statistic, 3), round(test.minimal.p_value, 3)
        (0.398, 0.528)
        >>> test.perfect.p_value < 1e-8
        True
        >>> test.conclusion
        'consistent with minimal repair; perfect repair rejected'
        """
        self._check_fitted()
        values = _REPAIR_VALUES.get(self.kind)
        if values is None:
            raise ValueError(
                f"The {self.kind} model has no repair values to test."
            )
        perfect, minimal = self._repair_fits()
        estimate = float(self.restoration)
        return RepairTestResult(
            kind=self.kind,
            parameter=self._restoration_param_name,
            estimate=estimate,
            log_likelihood=self.log_likelihood,
            perfect=perfect,
            minimal=minimal,
            alpha_ci=alpha_ci,
            conclusion=_repair_conclusion(
                perfect,
                minimal,
                estimate,
                values["label"],
                alpha_ci,
            ),
        )

    def _repair_fits(
        self,
    ) -> "tuple[RestrictedRepairFit, RestrictedRepairFit | None]":
        """The restricted refits of :meth:`repair_test`, made once and kept
        on the model (they do not depend on ``alpha_ci``). One warning if
        any failed."""
        cached = getattr(self, "_repair_fit_cache", None)
        if cached is not None:
            return cached
        values = _REPAIR_VALUES[self.kind]
        perfect = self._restricted_fit(
            "perfect", values["perfect"], values["label"] + " repair"
        )
        minimal = (
            None
            if values["minimal"] is None
            else self._restricted_fit(
                "minimal", values["minimal"], "minimal repair"
            )
        )
        failed = [f for f in (perfect, minimal) if f is not None and f.message]
        if failed:
            from surpyval.utils.warnings import caller_stacklevel

            warnings.warn(
                "The repair test is not available: {}. The other "
                "results of the fit are unaffected.".format(
                    "; ".join(f.message for f in failed)
                ),
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        self._repair_fit_cache = (perfect, minimal)
        return perfect, minimal

    def _restricted_starts(self, which: str) -> "list[np.ndarray]":
        """Starting values of the distribution's parameters for the fit
        with the restoration parameter held at ``which`` repair: the full
        fit's, and that restricted model's own fit where the family has
        one (the distribution fitted to the times between failures for a
        renewal process, the plain NHPP fit of an ARI baseline), or the
        fit to the times to first failure."""
        starts = [np.asarray(self._mle[1:], dtype=float)]
        fitter = getattr(self, "_fitter", None)
        data = getattr(self, "data", None)
        dist = self.model.dist
        if fitter is None or data is None:
            return starts
        ari = self._dist_label == "Baseline Intensity"
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                if ari and which == "minimal":
                    extra = fitter._initial_baseline_params(data, dist)
                elif ari:
                    extra = None
                elif which == "perfect":
                    extra = fitter._renewal_dist_params(data, dist)
                else:
                    extra = fitter._initial_dist_params(data, dist)
        except Exception:
            extra = None
        if extra is not None:
            extra = np.asarray(extra, dtype=float)
            if extra.shape == starts[0].shape and np.all(np.isfinite(extra)):
                starts.append(extra)
        return starts

    def _restricted_maximum(
        self, value: float, starts: "list[np.ndarray]"
    ) -> "tuple[float, np.ndarray]":
        """The minimum of the negative log-likelihood with the restoration
        parameter held at ``value``, over the other parameters searched
        from each of ``starts``, and where it is; ``(inf, nan)`` if no
        search reached a finite value."""
        bounds = self._parameter_bounds()[1:]

        def to_free(p: np.ndarray) -> np.ndarray:
            out = np.array(p, dtype=float)
            for k, (lo, hi) in enumerate(bounds):
                if lo is not None and hi is not None:
                    u = (out[k] - lo) / (hi - lo)
                    out[k] = np.log(u / (1 - u))
                elif lo is not None:
                    out[k] = np.log(out[k] - lo)
                elif hi is not None:
                    out[k] = np.log(hi - out[k])
            return out

        def from_free(z: np.ndarray) -> np.ndarray:
            out = np.array(z, dtype=float)
            for k, (lo, hi) in enumerate(bounds):
                if lo is not None and hi is not None:
                    out[k] = lo + (hi - lo) / (1 + np.exp(-out[k]))
                elif lo is not None:
                    out[k] = lo + np.exp(out[k])
                elif hi is not None:
                    out[k] = hi - np.exp(out[k])
            return out

        def restricted(z: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                try:
                    v = float(self._neg_ll(np.r_[value, from_free(z)]))
                except (ValueError, ArithmeticError):
                    return np.inf
            return v if np.isfinite(v) else np.inf

        best = None
        # The searches' own warnings (an overflowing trial point, a line
        # search that stops short) are not the user's.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for start in starts:
                z0 = to_free(start)
                if not np.isfinite(restricted(z0)):
                    continue
                res = minimize(restricted, z0, method="Nelder-Mead")
                polish = minimize(restricted, res.x, method="BFGS")
                if polish.fun < res.fun:
                    res = polish
                if np.isfinite(res.fun) and (
                    best is None or res.fun < best.fun
                ):
                    best = res
        if best is None:
            return np.inf, np.full(len(bounds), np.nan)
        return float(best.fun), from_free(best.x)

    def _restricted_fit(
        self, which: str, value: float, hypothesis: str
    ) -> RestrictedRepairFit:
        """Maximise the likelihood with the restoration parameter held at
        ``value`` and test the full fit against it (see
        :meth:`repair_test`)."""
        name = self._restoration_param_name
        mle = np.asarray(self._mle, dtype=float)
        lower, upper = self._restoration_bounds
        boundary = value == lower or value == upper
        fun, best_params = self._restricted_maximum(
            value, self._restricted_starts(which)
        )
        ll_full = -float(self._neg_ll(mle))
        if not np.isfinite(fun):
            nan_params = np.full(mle.size, np.nan)
            nan_params[0] = value
            return RestrictedRepairFit(
                hypothesis,
                name,
                value,
                nan_params,
                np.nan,
                np.nan,
                1,
                np.nan,
                boundary,
                message=(
                    f"the likelihood with {name} = {value:g} "
                    f"({hypothesis}) could not be maximised"
                ),
            )
        ll_restricted = -float(fun)
        # The restricted model is nested: its maximum cannot exceed the
        # full one's except by the optimisers' tolerance.
        stat = max(2.0 * (ll_full - ll_restricted), 0.0)
        p = float(chi2.sf(stat, 1))
        if boundary:
            # Self and Liang (1987): a 50:50 mixture of chi2(0) and chi2(1).
            p = 0.5 * p if stat > 0 else 1.0
        return RestrictedRepairFit(
            hypothesis,
            name,
            value,
            np.r_[value, best_params],
            ll_restricted,
            stat,
            1,
            p,
            boundary,
        )
