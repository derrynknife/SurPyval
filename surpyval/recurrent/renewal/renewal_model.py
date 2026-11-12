import math
import operator
import textwrap
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

from surpyval.diagnostics.mixin import DiagnoseMixin
from surpyval.recurrent.inference import (
    LikelihoodInferenceMixin,
    check_alpha_ci,
)
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._summary import format_table
from surpyval.utils.linalg import (
    bound_signs,
    scale_free_hessian,
    wald_bound_on_support,
)
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.numeric import solve_bracketed
from surpyval.utils.validation import (
    option_error,
    warn_outside_unit_interval,
)
from surpyval.utils.warnings import ignoring_warnings, warn_no_covariance

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
#: The public name of the repair values (``diagnose()``'s repair checks).
REPAIR_VALUES = _REPAIR_VALUES


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
            f"Model               : {self.kind}",
            "Estimate            : {} = {:.4g}".format(
                self.parameter, self.estimate
            ),
            f"Log-likelihood      : {self.log_likelihood:.6g}",
            "Tests               :",
        ]
        rows = []
        tests = [t for t in (self.perfect, self.minimal) if t is not None]
        for test in tests:
            value = f"{test.parameter} = {test.value:g}"
            if not test.available:
                rows.append([test.hypothesis, value, "nan", "nan", "nan", ""])
                continue
            rows.append(
                [
                    test.hypothesis,
                    value,
                    f"{test.log_likelihood:.6g}",
                    f"{test.statistic:.4g}",
                    f"{test.p_value:.4g}",
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
        (done,) = (t for t in (perfect, minimal) if t.available)
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
        where = f"between {label} and minimal repair"
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

#: A Newton step that moves the point by less than this fraction of itself
#: ends the search: the error left after it is about its square.
_NEWTON_RTOL = 1e-9

#: A Newton step below this fraction may also end the search when the
#: steps so far predict that the error left after it is below the double's
#: rounding (see ``_invert_cumulative_hazard``).
_NEWTON_ASYMPTOTIC = 1e-4

#: Newton steps before a draw is handed to the bracketing solver instead.
_NEWTON_MAXITER = 30


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
    (see ``_QF_HAZARD_LIMIT``) and by Newton's method on ``H`` beyond it
    (see ``_invert_cumulative_hazard``).
    """
    with np.errstate(divide="ignore"):
        energy = -np.log(np.asarray(u, dtype=float))
    return conditional_gaps_from_energy(lifetime, v, energy)


def conditional_gaps_from_energy(
    lifetime: Any, v: np.ndarray, energy: np.ndarray
) -> np.ndarray:
    """
    :func:`conditional_gaps` with the Exp(1) amount ``energy`` (``-log
    u``) given in place of ``u``: the residual life ``x`` from age ``v``
    with ``H(v + x) - H(v) = energy``. A quantile ``p`` of the residual
    life is ``energy = -log1p(-p)``, exact for a small ``p`` where
    ``-log(1 - p)`` is not.
    """
    v = np.asarray(v, dtype=float)
    energy = np.asarray(energy, dtype=float)
    hazard, quantile, both = _lifetime_functions(lifetime)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        H_v = np.broadcast_to(hazard(v), v.shape)
        target = H_v + energy
    # No survival past age v (the end of a bounded support): the item
    # fails at once.
    at_once = np.isinf(H_v) & (H_v > 0)
    # u == 0 (or an undefined H): an infinitely late event.
    never = ~at_once & ~np.isfinite(target)
    easy = ~at_once & ~never & (target <= _QF_HAZARD_LIMIT)
    hard = ~at_once & ~never & ~easy
    x = np.array(v, dtype=float)
    if easy.any():
        x[easy] = quantile(-np.expm1(-target[easy]))
    if hard.any():
        x[hard] = _invert_cumulative_hazard(
            hazard, v[hard], target[hard], both=both, H_v=H_v[hard]
        )
    gap = np.maximum(x - v, 0.0)
    gap[at_once] = 0.0
    gap[never] = np.inf
    return gap


def _lifetime_functions(
    lifetime: Any,
) -> "tuple[Callable, Callable, Callable | None]":
    """``H``, the quantile function and ``both``, on arrays, for
    ``lifetime``: ``both(x)`` gives ``(H(x), h(x))``, the cumulative
    hazard and the hazard rate, for Newton's method (``both(x, H)`` skips
    finding an ``H`` already known, where that saves a call). The rate is
    the model's (or distribution's) ``hf``; ``exp(log f + H)`` from the
    density where there is no ``hf`` (from ``log_df`` where there is one);
    and, for a mixture, both come from one pass over the components.
    ``both`` is ``None`` where there is no hazard rate to use (a discrete
    or a non-parametric lifetime: the latter's ``hf`` is the jumps of a
    step function), and ``H`` is then inverted by bracketing.

    For a plain model (no offset, cure fraction or zero-inflation -- what
    the renewal fitters build) the distribution's own functions are called
    directly, skipping the model methods' argument handling."""
    if (
        getattr(lifetime, "lfp_p", None) == 1
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

        def dist_rate(x: np.ndarray, H: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            below = x < lower
            inside = np.where(below, lower, x)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                if callable(getattr(dist, "hf", None)):
                    h = np.asarray(dist.hf(inside, *params), dtype=float)
                else:
                    h = np.exp(_log_density(dist, inside, *params) + H)
            return np.where(below, 0.0, h)

        def dist_both(
            x: np.ndarray, H: "np.ndarray | None" = None
        ) -> "tuple[np.ndarray, np.ndarray]":
            if H is None:
                H = hazard(x)
            return H, dist_rate(x, H)

        has_rate = callable(getattr(dist, "hf", None)) or callable(
            getattr(dist, "df", None)
        )
        if getattr(dist, "discrete", False) or not has_rate:
            return hazard, quantile, None
        return hazard, quantile, dist_both

    def model_hazard(x: np.ndarray) -> np.ndarray:
        return np.asarray(lifetime.Hf(x), dtype=float)

    def model_quantile(p: np.ndarray) -> np.ndarray:
        return np.asarray(lifetime.qf(p), dtype=float)

    dist = getattr(lifetime, "dist", None)
    if dist is None or getattr(dist, "discrete", False):
        return model_hazard, model_quantile, None
    # A mixture gives both from one pass over its components (``None``
    # where it cannot).
    joint = getattr(lifetime, "_hazard_and_rate", None)
    if callable(joint):

        def mixture_both(
            x: np.ndarray, H: "np.ndarray | None" = None
        ) -> "tuple[np.ndarray, np.ndarray]":
            # ``H`` is not needed: the pass finds it anyway.
            x = np.asarray(x, dtype=float)
            out = joint(x)
            if out is None:
                H = model_hazard(x)
                return H, np.asarray(lifetime.hf(x), dtype=float)
            return out

        return model_hazard, model_quantile, mixture_both
    if callable(getattr(lifetime, "hf", None)):

        def model_both(
            x: np.ndarray, H: "np.ndarray | None" = None
        ) -> "tuple[np.ndarray, np.ndarray]":
            if H is None:
                H = model_hazard(x)
            return H, np.asarray(lifetime.hf(x), dtype=float)

        return model_hazard, model_quantile, model_both
    if callable(getattr(lifetime, "df", None)):

        def density_both(
            x: np.ndarray, H: "np.ndarray | None" = None
        ) -> "tuple[np.ndarray, np.ndarray]":
            if H is None:
                H = model_hazard(x)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                return H, np.exp(_log_density(lifetime, x) + H)

        return model_hazard, model_quantile, density_both
    return model_hazard, model_quantile, None


def _log_density(source: Any, x: np.ndarray, *params: float) -> np.ndarray:
    """``log f(x)``: ``source.log_df`` where it has one (finite where the
    density underflows), else the log of ``source.df``."""
    if callable(getattr(source, "log_df", None)):
        return np.asarray(source.log_df(x, *params), dtype=float)
    return np.log(np.asarray(source.df(x, *params), dtype=float))


def _newton_point(
    x: np.ndarray, H: np.ndarray, h: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """One Newton step towards ``H(x) = target`` from ``x``, where ``H``
    and its derivative ``h`` are known. Where ``x``, ``H`` and ``h`` are
    all positive the step is taken on ``log H`` against ``log x``, which is
    a straight line for a Weibull (one step is exact) and nearly so in the
    upper tail of most lifetimes; elsewhere on ``H`` against ``x``. NaN
    where neither step is defined (a zero hazard rate)."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        logs = (x > 0) & (H > 0) & (h > 0)
        log_step = np.log(target / H) * H / (x * h)
        out = np.where(logs, x * np.exp(log_step), x + (target - H) / h)
    return np.where(np.isfinite(out), out, np.nan)


def _invert_cumulative_hazard(
    hazard: Callable,
    v: np.ndarray,
    target: np.ndarray,
    both: "Callable | None" = None,
    H_v: "np.ndarray | None" = None,
) -> np.ndarray:
    """Solve ``H(x_k) = target_k`` for ``x_k > v_k`` (where ``H(v_k) <
    target_k``), for every ``k`` at once.

    With ``both`` (``x -> (H(x), h(x))``, see ``_lifetime_functions``) this
    takes Newton steps from ``v`` (see ``_newton_point``), each kept
    inside a bracket ``lo < x < hi`` that every evaluation narrows: a step
    that would leave it bisects it instead (or, while no point above the
    root has been found, doubles the distance from ``v``). A draw is done

    - when a step moves it by less than ``_NEWTON_RTOL`` of itself (the
      error left is about that squared);
    - when a step below ``_NEWTON_ASYMPTOTIC`` follows another Newton step
      and the two predict an error after it below the double's rounding:
      Newton's error squares at each step, ``e' = K e^2``, and ``K`` is
      estimated as ``step / previous step^2``, so the error after a step
      ``s`` is about ``s^3 / previous^2``;
    - or when its bracket has closed to rounding.

    Two or three evaluations of ``both`` after the one at ``v`` usually
    do (``H_v``, the cumulative hazard at ``v`` if known, saves finding it
    again). A draw that meets an undefined or infinite ``H`` (past a
    bounded support), or has not converged in ``_NEWTON_MAXITER`` steps,
    is solved by bracketing instead (``_bracket_and_solve``), as is every
    draw where ``both`` is ``None``. One draw at a time (what the
    simulations ask for) runs the same steps on Python floats
    (``_newton_one``), without the array bookkeeping.
    """
    v = np.asarray(v, dtype=float)
    target = np.asarray(target, dtype=float)
    if both is None:
        return _bracket_and_solve(hazard, v, target)
    if H_v is not None:
        H_v = np.broadcast_to(np.asarray(H_v, dtype=float), v.shape)
    if v.size == 1:
        root = _newton_one(
            both,
            float(v.reshape(-1)[0]),
            float(target.reshape(-1)[0]),
            None if H_v is None else float(H_v.reshape(-1)[0]),
        )
        if root is not None:
            return np.full(v.shape, root)
        return _bracket_and_solve(hazard, v, target)
    n = v.size
    out = np.full(n, np.nan)
    lo = np.array(v, dtype=float)
    hi = np.full(n, np.inf)
    first = np.maximum(np.abs(v), 1.0)
    # The relative size of the last Newton step (inf after any other).
    previous = np.full(n, np.inf)
    x = np.array(v, dtype=float)
    active = np.arange(n)
    give_up = np.zeros(n, dtype=bool)
    for k in range(_NEWTON_MAXITER + 1):
        if not active.size:
            break
        a = active
        xa = x[a]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            H, h = both(xa, None if k or H_v is None else H_v[a])
            H = np.broadcast_to(np.asarray(H, dtype=float), xa.shape)
            h = np.broadcast_to(np.asarray(h, dtype=float), xa.shape)
        g = H - target[a]
        # At v itself H is below the target by construction (rounding
        # aside); past a bounded support H is inf or nan.
        bad = ~np.isfinite(H)
        exact = (g == 0) & (k > 0)
        below = (g < 0) | (k == 0)
        lo[a] = np.where(below & ~bad, xa, lo[a])
        hi[a] = np.where(~below & ~bad, xa, hi[a])
        lo_a, hi_a = lo[a], hi[a]
        step = _newton_point(xa, H, h, target[a])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            size = np.abs(step - xa) / np.abs(xa)
            # No estimate of K without an earlier Newton step.
            predicted = np.where(
                np.isfinite(previous[a]), size**3 / previous[a] ** 2, np.inf
            )
        inside = np.isfinite(step) & (step > lo_a) & (step < hi_a)
        # Outside the bracket: bisect it, or reach further out while
        # nothing above the root has been found.
        fallback = np.where(
            np.isfinite(hi_a),
            lo_a + 0.5 * (hi_a - lo_a),
            v[a] + 2.0 * np.maximum(lo_a - v[a], first[a]),
        )
        tiny = np.isfinite(step) & (size <= _NEWTON_RTOL)
        settled = (
            inside & (size <= _NEWTON_ASYMPTOTIC) & (predicted <= _MACHINE_EPS)
        )
        closed = np.isfinite(hi_a) & (
            hi_a - lo_a
            <= 4 * _MACHINE_EPS * np.maximum(np.abs(lo_a), np.abs(hi_a))
        )
        # A step this small is done even if rounding puts it just outside
        # the bracket: the point itself is then the answer.
        out[a] = np.where(
            exact | (tiny & ~inside),
            xa,
            np.where(tiny | settled, step, lo_a + 0.5 * (hi_a - lo_a)),
        )
        previous[a] = np.where(inside, size, np.inf)
        give_up[a] = bad
        x[a] = np.where(inside, step, fallback)
        active = a[~(exact | tiny | settled | closed | bad)]
    give_up[active] = True
    if give_up.any():
        out[give_up] = _bracket_and_solve(hazard, v[give_up], target[give_up])
    return out


def _first(values: Any) -> float:
    """The one value of a one-element result, as a float."""
    return float(np.asarray(values, dtype=float).reshape(-1)[0])


def _newton_point_one(x: float, H: float, h: float, target: float) -> float:
    """``_newton_point`` for one draw, on floats (NaN where no step)."""
    try:
        if x > 0 and H > 0 and h > 0:
            out = x * math.exp(math.log(target / H) * H / (x * h))
        else:
            out = x + (target - H) / h
    except (ArithmeticError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _newton_one(
    both: Callable, v: float, target: float, H_v: "float | None"
) -> "float | None":
    """``_invert_cumulative_hazard``'s Newton steps for one draw, on
    floats, with the same rules; ``None`` where it hands the draw to
    bracketing (an undefined or infinite ``H``, or no convergence)."""
    lo, hi = v, math.inf
    first = max(abs(v), 1.0)
    previous = math.inf
    x = v
    for k in range(_NEWTON_MAXITER + 1):
        point = np.array([x])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if k == 0 and H_v is not None:
                H_arr, h_arr = both(point, np.array([H_v]))
            else:
                H_arr, h_arr = both(point)
        H, h = _first(H_arr), _first(h_arr)
        if not math.isfinite(H):
            return None
        g = H - target
        if g == 0 and k > 0:
            return x
        if g < 0 or k == 0:
            lo = x
        else:
            hi = x
        step = _newton_point_one(x, H, h, target)
        size = abs(step - x) / abs(x) if x != 0 else math.inf
        inside = lo < step < hi
        if size <= _NEWTON_RTOL:
            return step if inside else x
        if (
            inside
            and size <= _NEWTON_ASYMPTOTIC
            and math.isfinite(previous)
            and size**3 / previous**2 <= _MACHINE_EPS
        ):
            return step
        if math.isfinite(hi) and hi - lo <= 4 * _MACHINE_EPS * max(
            abs(lo), abs(hi)
        ):
            return lo + 0.5 * (hi - lo)
        previous = size if inside else math.inf
        if inside:
            x = step
        elif math.isfinite(hi):
            x = lo + 0.5 * (hi - lo)
        else:
            x = v + 2.0 * max(lo - v, first)
    return None


def _bracket_and_solve(
    hazard: Callable, v: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """``_invert_cumulative_hazard`` without a hazard rate: bracket each
    root by doubling a step out from ``v``, then ``solve_bracketed``."""

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


def discount_weight_derivatives(
    rho: float, count: int, order: int
) -> np.ndarray:
    """The ``order``-th derivative in ``rho`` of the weights ``rho * (1 -
    rho)**j``, ``j = 0 .. count - 1``, that the ARA and ARI models put on
    the ``j``-th most recent term of their memory (#710). By Leibniz's
    rule it is ``rho D^n + n D^(n-1)``, with ``D^k`` the ``k``-th
    derivative of ``(1 - rho)**j``: ``(-1)**k j! / (j - k)! (1 -
    rho)**(j - k)``, and 0 for ``k > j``."""
    j = np.arange(count, dtype=float)

    def power_derivative(k: int) -> np.ndarray:
        if k < 0:
            return np.zeros(count)
        falling = np.ones(count)
        for t in range(k):
            falling = falling * (j - t)
        with np.errstate(divide="ignore", invalid="ignore"):
            power = (1.0 - rho) ** np.maximum(j - k, 0.0)
        return np.where(j >= k, (-1.0) ** k * falling * power, 0.0)

    return rho * power_derivative(order) + order * power_derivative(order - 1)


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

    @classmethod
    def from_history(
        cls, histories: "list[np.ndarray]", rho: float, m: "int | float"
    ) -> "DiscountedMemory":
        """The memory of sequences that have each recorded their own
        values already, ``histories[k]`` oldest first (of any lengths):
        for sequences continued from units' current states."""
        memory = cls(len(histories), rho, m)
        if memory.infinite:
            for k, values in enumerate(histories):
                total = 0.0
                for value in np.asarray(values, dtype=float):
                    total = value + memory.decay * total
                memory.total[k] = total
            return memory
        depth = min(memory.m, max((len(h) for h in histories), default=0))
        # The last ``depth`` values of each, aligned at the newest; a
        # sequence with fewer has zeros (no term) before them.
        rows = np.zeros((depth, memory.n))
        for k, values in enumerate(histories):
            start = max(len(values) - depth, 0)
            recent = np.asarray(values, dtype=float)[start:]
            if depth and recent.size:
                rows[depth - recent.size :, k] = recent
        memory.recent = list(rows)
        return memory

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


@dataclass
class UnitStates:
    """Units' current states under a fitted renewal model: what the
    prediction of each unit's next failures starts from
    (:meth:`RenewalModel.unit_states` gives them as a table).

    Every array has one entry per unit, and times are on the model's clock
    (from new, or from entry for an item with delayed entry).
    """

    #: The units' labels (the fitted data's items), or ``None`` for units
    #: given by their ages.
    units: "list | None"
    #: Each unit's failure times so far, oldest first.
    failures: "list[np.ndarray]"
    #: The time now: the end of each unit's history.
    now: np.ndarray
    #: The time since the unit's last failure (or since new / entry).
    since_failure: np.ndarray
    #: The unit's virtual age just after its last repair (0 for G1 and
    #: ARI, which have none).
    after_repair: np.ndarray
    #: The age the lifetime distribution sees now: the virtual age now
    #: (GRP, ARA), the time since the last failure divided by
    #: ``(1 + q) ** j`` (G1), ``nan`` for ARI.
    age: np.ndarray
    #: The factor the next gap is scaled by: ``(1 + q) ** j`` for G1, 1
    #: otherwise.
    scale: np.ndarray
    #: The intensity reduction in force (ARI), 0 otherwise.
    reduction: np.ndarray

    @property
    def failure_count(self) -> np.ndarray:
        """The number of failures of each unit so far."""
        return np.array([f.size for f in self.failures], dtype=int)

    def take(self, rows: np.ndarray) -> "UnitStates":
        """The states of the units ``rows`` (repeats allowed)."""
        rows = np.asarray(rows, dtype=int)
        return UnitStates(
            units=None,
            failures=[self.failures[r] for r in rows],
            now=self.now[rows],
            since_failure=self.since_failure[rows],
            after_repair=self.after_repair[rows],
            age=self.age[rows],
            scale=self.scale[rows],
            reduction=self.reduction[rows],
        )


class RenewalModel(
    DiagnoseMixin,
    SerialisableMixin,
    RecurrenceSimulationMixin,
    LikelihoodInferenceMixin,
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
        methods reproduce exactly. A maximum-likelihood fit's parameter
        covariance (that of :meth:`covariance`) is stored too
        (``"covariance"``), so the reloaded model gives the same
        ``covariance()``, ``standard_errors()``, ``summary()`` and Wald
        ``param_cb``. The data and the likelihood are not stored: the
        likelihood, the information criteria, the repair test and the
        profile-likelihood intervals (``method="lr"``, and the restoration
        parameter's interval on the edge of its range) need a refit, and
        say so.

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
            # What the Wald bounds need (null for a model with no
            # likelihood); optional on reading.
            "covariance": self._covariance_entry(),
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
            raise ValueError(f"Unknown renewal family {family!r}")
        out = cls._rebuild(fitters[family], family, model_dict)
        # A reloaded fit still says it was fitted by MLE (it carries no
        # likelihood, as a reloaded parametric model does not).
        out.how = model_dict.get("how", "from_params")
        out.maximum = restored_maximum(model_dict)
        out._restore_covariance(
            model_dict,
            np.r_[float(out.restoration), np.asarray(out.model.params)],
        )
        return out

    def _covariance_entry(self) -> "list | None":
        """The covariance as ``to_dict`` saves it: the fit's
        :meth:`covariance` (with the restoration parameter's row and
        column ``nan`` on the edge of its range), a restored model's
        saved one, or ``None`` for a model with no likelihood."""
        if not hasattr(self, "_neg_ll"):
            if self._saved_covariance is None:
                return None
            return np.asarray(self._saved_covariance, dtype=float).tolist()
        with ignoring_warnings():
            return np.asarray(self.covariance(), dtype=float).tolist()

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

    @property
    def covariance_names(self) -> list[str]:
        """
        The names of the rows (and columns) of :meth:`covariance`, in its
        order, which :meth:`with_params` takes: the restoration parameter
        (``q`` or ``rho``), then the lifetime distribution's (for ARI the
        baseline intensity's) parameters -- the ``parameter_names``.

        Examples
        --------
        >>> from surpyval.recurrent import ARA
        >>> ARA.fit_from_parameters([10.0, 2.0], 0.4, m=2).covariance_names
        ['rho', 'alpha', 'beta']
        """
        return list(self.parameter_names)

    def _estimates(self) -> np.ndarray:
        """The parameters in the order of :attr:`covariance_names`."""
        return self.params

    def with_params(self, params: ArrayLike) -> "RenewalModel":
        """
        The same model -- family, lifetime distribution (or ARI baseline),
        Kijima type, memory ``m`` and unit -- at other parameters: for
        drawing a fit's parameters from its covariance and carrying them
        into a prediction or simulation, say.

        Parameters
        ----------
        params : array_like
            A flat vector in the order of :attr:`covariance_names`, the
            restoration parameter first, as a draw from :meth:`covariance`
            comes. The values are checked as ``fit_from_parameters``
            checks them (a ``q`` below 0 for the generalized renewal
            process, a ``rho`` outside [0, 1], is refused).

        Returns
        -------
        RenewalModel
            A model built from parameters, as ``fit_from_parameters``
            builds it: it has no data, covariance or fitted likelihood,
            since those belong to the fit of the original parameters, so
            data-based methods (bounds, ``repair_test``, prediction for
            the fitted items) are not available on it. A model fitted
            with weights becomes the plain model at these parameters.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters(
        ...     [10.0, 2.0], 0.5, kijima="ii")
        >>> other = model.with_params([0.2, 12.0, 2.5])
        >>> other.q, other.kijima_type, other.model.params
        (0.2, 'ii', array([12. ,  2.5]))
        >>> bool(np.allclose(model.with_params(model._estimates()).params,
        ...                  model.params))
        True
        """
        names = self.covariance_names
        values = np.asarray(params, dtype=float)
        if values.shape != (len(names),):
            raise ValueError(
                "params must be a flat vector of {} values in the order "
                "of covariance_names {}; got shape {}".format(
                    len(names), names, np.shape(params)
                )
            )
        import surpyval.recurrent as recurrent

        family = self._family()
        fitter = getattr(recurrent, family)
        restoration = float(values[0])
        dist_params = values[1:].tolist()
        out: RenewalModel
        if family == "ARI":
            out = fitter.fit_from_parameters(
                dist_params, restoration, m=self.m, baseline=self.model.dist
            )
        elif family == "GeneralizedRenewal":
            out = fitter.fit_from_parameters(
                dist_params,
                restoration,
                kijima=self.kijima_type,
                dist=self.model.dist,
            )
        elif family == "ARA":
            out = fitter.fit_from_parameters(
                dist_params, restoration, m=self.m, dist=self.model.dist
            )
        else:
            out = fitter.fit_from_parameters(
                dist_params, restoration, dist=self.model.dist
            )
        unit = getattr(self, "unit", None)
        if unit is not None:
            out.unit = unit
        return out

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

        def martingale() -> np.ndarray:
            residuals = []
            for item in np.unique(self.data.i):
                mask = self.data.i == item
                observed = int((c[mask] == 0).sum())
                residuals.append(observed - float(increments[mask].sum()))
            return np.array(residuals)

        return diagnostics.residuals_of(
            kind, lambda: increments[c == 0], martingale
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

    # -- prediction from each unit's current state (#615) ------------------

    def _states(self, age: "ArrayLike | None" = None) -> UnitStates:
        """The units' current states: the fitted data's items at the end
        of their histories (``age=None``), or units that have run ``age``
        since new without a failure."""
        if age is None:
            self._check_has_data("Prediction from each unit's history")
            data = self.data
            item = np.asarray(data.i)
            order = np.argsort(item, kind="stable")
            x = np.asarray(data.x, dtype=float)[order]
            failed = np.asarray(data.c)[order] == 0
            _, first = np.unique(item[order], return_index=True)
            units = list(data.ids)
            times = np.split(x, first[1:])
            events = np.split(failed, first[1:])
            failures = [np.sort(t[e]) for t, e in zip(times, events)]
            now = np.array([t.max() for t in times], dtype=float)
        else:
            now = np.atleast_1d(np.asarray(age, dtype=float)).reshape(-1)
            if not np.all(np.isfinite(now)) or np.any(now < 0):
                raise ValueError(
                    "age must be finite times of at least 0 (each unit's "
                    "time since new, with no failure yet); got "
                    "{}".format(now[~(np.isfinite(now) & (now >= 0))][:5])
                )
            units = None
            failures = [np.zeros(0) for _ in now]
        return self._state_of(units, failures, now)

    def _state_of(
        self, units: "list | None", failures: list, now: np.ndarray
    ) -> UnitStates:
        """The states of units with the failure times ``failures`` (each
        oldest first) at the times ``now``, under this model's family."""
        k = len(failures)
        last = np.array([f[-1] if f.size else 0.0 for f in failures])
        since = now - last
        after = np.zeros(k)
        scale = np.ones(k)
        reduction = np.zeros(k)
        family = self._family()
        rate = float(self.restoration)
        if family == "GeneralizedRenewal":
            from .generalized_renewal import kijima_ii_from_prev_interarrival

            for r, f in enumerate(failures):
                if not f.size:
                    continue
                if self.kijima_type == "i":
                    # v_n = v_{n-1} + q x_n, so q times the last failure
                    after[r] = rate * f[-1]
                else:
                    gaps = np.diff(f, prepend=0.0)
                    after[r] = kijima_ii_from_prev_interarrival(gaps, rate)[-1]
            age = after + since
        elif family == "ARA":
            from .ara import ara_virtual_ages

            for r, f in enumerate(failures):
                if f.size:
                    # The age at the start of the gap after the last failure
                    after[r] = ara_virtual_ages(
                        np.append(f, now[r]), rate, self.m
                    )[-1]
            age = after + since
        elif family == "GeneralizedOneRenewal":
            scale = (1.0 + rate) ** np.array([f.size for f in failures])
            age = since / scale
        else:
            from .ari import ari_reduction

            for r, f in enumerate(failures):
                if f.size:
                    lam = np.asarray(self.model.iif(f), dtype=float)
                    reduction[r] = ari_reduction(lam, rate, self.m)
            age = np.full(k, np.nan)
        return UnitStates(
            units=units,
            failures=failures,
            now=now,
            since_failure=since,
            after_repair=after,
            age=age,
            scale=scale,
            reduction=reduction,
        )

    def unit_states(self, age: "ArrayLike | None" = None) -> pd.DataFrame:
        """
        Each unit's current state: where the prediction of its next
        failures (:meth:`next_failure_sf`, :meth:`next_failure_hf`,
        :func:`surpyval.forecast`) starts from.

        Parameters
        ----------
        age : array_like, optional
            By default the units are the fitted data's items, each at the
            end of its history (its last row: its end of observation, or
            its last failure). Give ``age`` for units that have instead
            run that long since new (or since entry) without a failure.

        Returns
        -------
        pandas.DataFrame
            One row per unit (indexed by the item label), with ``time``,
            the time now on the model's clock (from new, or from entry
            for an item with delayed entry); ``failures``, its number of
            failures so far; ``since_failure``, the time since its last
            failure (or since new); and the family's state: the
            ``virtual_age`` now (GRP and ARA; for G1 the time since the
            last failure on the base lifetime's axis, with the ``scale``
            ``(1 + q) ** failures`` of its next gap) or the intensity
            ``reduction`` in force (ARI).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> model.unit_states(age=[0.0, 4.0])
           time  failures  since_failure  virtual_age
        0   0.0         0            0.0          0.0
        1   4.0         0            4.0          4.0
        """
        states = self._states(age)
        table = pd.DataFrame(
            {
                "time": states.now,
                "failures": states.failure_count,
                "since_failure": states.since_failure,
            },
            index=states.units,
        )
        family = self._family()
        if family == "ARI":
            table["reduction"] = states.reduction
        else:
            table["virtual_age"] = states.age
        if family == "GeneralizedOneRenewal":
            table["scale"] = states.scale
        if states.units is not None:
            table.index.name = "item"
        return table

    def states(
        self,
        age: "ArrayLike | None" = None,
        *,
        failures: "list | None" = None,
        now: "ArrayLike | None" = None,
    ) -> UnitStates:
        """
        Units' current states as a :class:`UnitStates`, for
        :meth:`next_failure_random`, :meth:`next_failure_qf`,
        :meth:`next_failure_sf` and :meth:`next_failure_hf` (their
        ``states=``) and for :meth:`advance`. :meth:`unit_states` is the
        same as a table.

        Parameters
        ----------
        age : array_like, optional
            Units that have run this long since new without a failure.
        failures : list of array_like, optional
            Units with these failure histories instead (keyword only):
            each unit's failure times so far, on the model's clock (from
            new), each failure repaired at once.
        now : array_like, optional
            With ``failures`` (keyword only), the time now for each unit,
            at or after its last failure; by default its last failure (0
            for a unit with none).

        Without any of them the units are the fitted data's items, each
        at the end of its history.

        Returns
        -------
        UnitStates
            One entry per unit; the restoration rule's state is in
            ``age`` (the virtual age now, for G1 on the base lifetime's
            axis with ``scale``) or ``reduction`` (ARI).

        Examples
        --------
        A Kijima-I unit with ``q = 0.5`` that failed at 4 and 10 and has
        run to 12 has the virtual age :math:`0.5 \\times 10 + 2 = 7`:

        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> state = model.states(failures=[[4.0, 10.0]], now=[12.0])
        >>> state.age, state.failure_count
        (array([7.]), array([2]))
        """
        if failures is None:
            if now is not None:
                raise ValueError(
                    "now goes with failures (the time now after each "
                    "unit's failures); for units with no failure give "
                    "age instead"
                )
            return self._states(age)
        if age is not None:
            raise ValueError(
                "give the units by age (no failures yet) or by their "
                "failures (and now), not both"
            )
        histories = []
        for k, f in enumerate(failures):
            times = np.atleast_1d(np.asarray(f, dtype=float)).reshape(-1)
            if not np.all(np.isfinite(times)) or np.any(times < 0):
                raise ValueError(
                    "failures must be finite times of at least 0; unit "
                    f"{k} has {times[~(np.isfinite(times) & (times >= 0))]}"
                )
            histories.append(np.sort(times))
        last = np.array([h[-1] if h.size else 0.0 for h in histories])
        if now is None:
            times_now = last
        else:
            times_now = np.broadcast_to(
                np.asarray(now, dtype=float), last.shape
            ).copy()
            bad = ~np.isfinite(times_now) | (times_now < last)
            if bad.any():
                raise ValueError(
                    "now must be finite and at or after each unit's last "
                    "failure; units {} are not".format(np.flatnonzero(bad))
                )
        return self._state_of(None, histories, times_now)

    def advance(
        self, states: UnitStates, x: ArrayLike, failed: "ArrayLike" = True
    ) -> UnitStates:
        """
        The units' states a further ``x`` from now: after a failure at
        ``now + x``, repaired at once by the fitted restoration rule
        (``failed=True``), or after running ``x`` without one
        (``failed=False``).

        The repair is the model's own: the Kijima virtual age (``q`` of
        the generalized renewal process), the arithmetic reduction of age
        or of intensity with memory ``m`` (ARA, ARI) or the next gap's
        scale :math:`(1 + q)^j` (G1). With :meth:`next_failure_random`
        it steps units forward event by event, so a simulation can put
        its own events between failures; time a unit spends out of
        service is simply not added to its clock.

        Parameters
        ----------
        states : UnitStates
            The units' states now, from :meth:`states` or an earlier
            call.
        x : array_like
            Each unit's time ahead of now (at least 0, finite): one value
            for every unit, or one per unit.
        failed : bool or array_like of bool, optional
            Whether each unit fails at ``now + x`` (the default) or has
            only run on: one value for every unit, or one per unit.

        Returns
        -------
        UnitStates
            The new states; ``states`` itself is not changed.

        Examples
        --------
        Kijima I, ``q = 0.5``: a new unit that fails at 4 is repaired to
        the virtual age 2, and after running 3 more it is at 5:

        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> state = model.advance(model.states(age=[0.0]), 4.0)
        >>> state.after_repair, state.age
        (array([2.]), array([2.]))
        >>> model.advance(state, 3.0, failed=False).age
        array([5.])
        """
        if not isinstance(states, UnitStates):
            raise TypeError(
                "states must be a UnitStates, as model.states() or "
                f"model.advance() gives; got {type(states).__name__}"
            )
        ahead = np.broadcast_to(
            np.asarray(self._ahead(x), dtype=float), states.now.shape
        )
        if not np.all(np.isfinite(ahead)):
            raise ValueError(
                "x is each unit's time ahead of now, so it must be "
                f"finite; got {ahead[~np.isfinite(ahead)][:5]}"
            )
        now = states.now + ahead
        fails = np.broadcast_to(np.asarray(failed, dtype=bool), now.shape)
        failures = [
            np.append(f, t) if d else f
            for f, t, d in zip(states.failures, now, fails)
        ]
        return self._state_of(states.units, failures, now)

    def _resolve_states(
        self, age: "ArrayLike | None", states: "UnitStates | None"
    ) -> UnitStates:
        """The units the next-failure methods are about: ``states`` as
        given, else those of ``age`` (or the fitted data's items)."""
        if states is None:
            return self._states(age)
        if age is not None:
            raise ValueError("give age or states, not both")
        if not isinstance(states, UnitStates):
            raise TypeError(
                "states must be a UnitStates, as model.states() or "
                f"model.advance() gives; got {type(states).__name__}"
            )
        return states

    def _next_failure_hazard(
        self, states: UnitStates, x: np.ndarray
    ) -> np.ndarray:
        """``(units, len(x))``: the cumulative intensity each unit
        accumulates over ``(now, now + x]`` if it does not fail."""
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self._family() == "ARI":
                now = states.now[:, None]
                cif = self.model.cif
                return (
                    np.asarray(cif(now + x), dtype=float)
                    - np.asarray(cif(now + 0.0 * x), dtype=float)
                    - states.reduction[:, None] * x
                )
            hazard, _, _ = _lifetime_functions(self.model)
            age = states.age[:, None]
            scaled = x / states.scale[:, None]
            return hazard(age + scaled) - hazard(age + 0.0 * scaled)

    @staticmethod
    def _ahead(x: ArrayLike) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if not np.all(values[~np.isnan(values)] >= 0):
            raise ValueError(
                "x is time ahead of now, so it cannot be negative; got "
                "{}".format(values[values < 0][:5])
            )
        return values

    def next_failure_sf(
        self,
        x: ArrayLike,
        age: "ArrayLike | None" = None,
        *,
        states: "UnitStates | None" = None,
    ) -> np.ndarray:
        """
        The chance that each unit runs a further ``x`` from now without a
        failure, given its history: the survival function of the time to
        its next failure.

        From each unit's current state (:meth:`unit_states`): for the
        virtual-age models (``GeneralizedRenewal``, ``ARA``) the lifetime
        distribution's conditional survival from the unit's virtual age
        now, :math:`S(v + x) / S(v)`; for G1, the same for the base
        lifetime from the time since the last failure, on the time scale
        :math:`(1 + q)^j` of the unit's next gap; for ARI,
        :math:`\\exp(-[\\Lambda_0(t + x) - \\Lambda_0(t) - R\\, x])` with
        the intensity reduction :math:`R` in force.

        Parameters
        ----------
        x : array_like
            Times ahead of now (at least 0).
        age : array_like, optional
            Units that have run this long since new without a failure, in
            place of the fitted data's items (see :meth:`unit_states`).
        states : UnitStates, optional
            The units' states (keyword only), as :meth:`states` or
            :meth:`advance` gives them, in place of ``age`` or the fitted
            data's items.

        Returns
        -------
        numpy.ndarray
            ``(units, len(x))``, one row per unit (in the order of
            :meth:`unit_states`); ``(units,)`` for a scalar ``x``.

        Examples
        --------
        Under a Weibull lifetime (``alpha = 10``, ``beta = 2``), the
        chance of 5 more hours without a failure for a new unit and for
        one that has run 6 hours since new, :math:`e^{-(1.1^2 - 0.6^2)}`:

        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> model.next_failure_sf(5.0, age=[0.0, 6.0]).round(4)
        array([0.7788, 0.4274])

        For a fitted model, ``age`` is left out and the units are the
        fitted items, each from the state its own failures left it in.
        """
        values = self._ahead(x)
        states = self._resolve_states(age, states)
        flat = np.atleast_1d(values).reshape(-1)
        out = np.exp(-self._next_failure_hazard(states, flat))
        return out[:, 0] if values.ndim == 0 else out

    def next_failure_hf(
        self,
        x: ArrayLike,
        age: "ArrayLike | None" = None,
        *,
        states: "UnitStates | None" = None,
    ) -> np.ndarray:
        """
        Each unit's intensity a further ``x`` from now, if it has not
        failed by then: the hazard function of the time to its next
        failure (its conditional intensity until that failure).

        For the virtual-age models the lifetime hazard at the virtual age
        :math:`h(v + x)`; for G1 the base hazard on the next gap's time
        scale, :math:`h_0(s / c + x / c) / c` with :math:`c = (1 + q)^j`;
        for ARI :math:`\\lambda_0(t + x) - R`. See :meth:`next_failure_sf`
        for the arguments and the shape of the result.

        Examples
        --------
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> model.next_failure_hf(0.0, age=[0.0, 5.0]).round(4)
        array([0. , 0.1])
        """
        values = self._ahead(x)
        states = self._resolve_states(age, states)
        flat = np.atleast_1d(values).reshape(-1)[None, :]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self._family() == "ARI":
                out = (
                    np.asarray(
                        self.model.iif(states.now[:, None] + flat), dtype=float
                    )
                    - states.reduction[:, None]
                )
            else:
                scale = states.scale[:, None]
                out = (
                    np.asarray(
                        self.model.hf(states.age[:, None] + flat / scale),
                        dtype=float,
                    )
                    / scale
                )
        return out[:, 0] if values.ndim == 0 else out

    def next_failure_qf(
        self,
        p: ArrayLike,
        age: "ArrayLike | None" = None,
        *,
        states: "UnitStates | None" = None,
    ) -> np.ndarray:
        """
        The time ahead of now by which each unit fails with probability
        ``p``, given its history: the quantile function of the time to
        its next failure, the inverse of :meth:`next_failure_sf`
        (``next_failure_sf(next_failure_qf(p)) == 1 - p``).

        From each unit's current state (:meth:`unit_states`): for the
        virtual-age models (``GeneralizedRenewal``, ``ARA``) the lifetime
        distribution's conditional quantile from the unit's virtual age
        :math:`v`, the :math:`x` with :math:`H(v + x) - H(v) =
        -\\log(1 - p)`; for G1 the same for the base lifetime on the time
        scale :math:`(1 + q)^j` of the unit's next gap; for ARI the
        :math:`x` with :math:`\\Lambda_0(t + x) - \\Lambda_0(t) - R\\, x =
        -\\log(1 - p)`. Working on the cumulative hazard keeps the upper
        tail exact (a ``p`` near 1, a unit far into its life); a time is
        found to about the double's rounding of the unit's age, so a tiny
        gap after a long life carries that absolute error.

        Parameters
        ----------
        p : array_like
            Probabilities in [0, 1]. ``p = 0`` gives 0 and ``p = 1``
            ``inf``; a value outside [0, 1] gives ``nan``, with a
            warning.
        age : array_like, optional
            Units that have run this long since new without a failure, in
            place of the fitted data's items (see :meth:`unit_states`).
        states : UnitStates, optional
            The units' states (keyword only), as :meth:`states` or
            :meth:`advance` gives them, in place of ``age`` or the fitted
            data's items.

        Returns
        -------
        numpy.ndarray
            ``(units, len(p))``, one row per unit (in the order of
            :meth:`unit_states`); ``(units,)`` for a scalar ``p``.

        Examples
        --------
        Under a Weibull lifetime (``alpha = 10``, ``beta = 2``), the
        median time to the next failure of a new unit,
        :math:`10\\sqrt{\\log 2}`, and of one that has run 6 hours since
        new, :math:`10\\sqrt{0.36 + \\log 2} - 6`:

        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> model.next_failure_qf(0.5, age=[0.0, 6.0]).round(4)
        array([8.3255, 4.2614])
        >>> model.next_failure_sf(model.next_failure_qf(0.5, age=[6.0]),
        ...                       age=[6.0]).round(12)
        array([[0.5]])
        """
        values = np.asarray(p, dtype=float)
        outside = warn_outside_unit_interval(values, "next_failure_qf")
        unit_states = self._resolve_states(age, states)
        flat = np.where(outside, np.nan, values).reshape(-1)
        out = self._next_gaps(unit_states, flat[None, :])
        return out[:, 0] if values.ndim == 0 else out

    def next_failure_random(
        self,
        size: int,
        age: "ArrayLike | None" = None,
        *,
        states: "UnitStates | None" = None,
        random_state: Any = None,
    ) -> np.ndarray:
        """
        Draw each unit's time to its next failure, ahead of now, given its
        history: ``size`` draws per unit.

        Each draw is :meth:`next_failure_qf` at one uniform ``u``, the
        uniforms filling the ``(units, size)`` result row by row, so a
        draw can be repeated (common random numbers) by passing the same
        ``random_state``. With :meth:`advance` this steps units forward
        failure by failure under the fitted model -- alongside other
        events of a caller's own simulation (repairs that take time,
        maintenance, other components) -- and reproduces
        :meth:`time_terminated_simulation` and
        :meth:`count_terminated_simulation` in distribution, as below.

        Parameters
        ----------
        size : int
            The number of draws for each unit.
        age : array_like, optional
            Units that have run this long since new without a failure, in
            place of the fitted data's items (see :meth:`unit_states`).
        states : UnitStates, optional
            The units' states (keyword only), as :meth:`states` or
            :meth:`advance` gives them, in place of ``age`` or the fitted
            data's items.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a reproducible draw (keyword only); by
            default one seeded from numpy's global stream, so
            ``np.random.seed`` makes the draw reproducible.

        Returns
        -------
        numpy.ndarray
            ``(units, size)``: one row per unit (in the order of
            :meth:`unit_states`).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
        >>> model.next_failure_random(3, age=[0.0, 6.0],
        ...                           random_state=1).round(3)
        array([[ 8.468, 17.335,  3.946],
               [12.246,  2.566,  3.542]])

        Three units stepped through their first three failures, each
        repaired at once:

        >>> states = model.states(age=[0.0, 0.0, 0.0])
        >>> rng = np.random.default_rng(2)
        >>> for _ in range(3):
        ...     gap = model.next_failure_random(1, states=states,
        ...                                     random_state=rng)[:, 0]
        ...     states = model.advance(states, gap)
        >>> states.failure_count
        array([3, 3, 3])
        >>> states.now.round(3)
        array([ 9.173, 13.425, 21.141])
        """
        try:
            count = operator.index(size)
        except TypeError:
            count = -1
        if count < 0:
            raise ValueError(
                "size is the number of draws for each unit, a whole number "
                f"of at least 0; got {size!r}"
            )
        unit_states = self._resolve_states(age, states)
        from surpyval.utils.rng import as_generator

        u = as_generator(random_state).random((unit_states.now.size, count))
        return self._next_gaps(unit_states, u)

    def _next_gaps(self, states: UnitStates, p: np.ndarray) -> np.ndarray:
        """``(units, m)``: each unit's quantile ``p`` (broadcast to that
        shape) of the time to its next failure; see
        :meth:`next_failure_qf`. The gap is found from the Exp(1) amount
        ``-log1p(-p)`` by the samplers' own inversions
        (:func:`conditional_gaps_from_energy`, :func:`ari_gaps`)."""
        k = states.now.size
        p = np.broadcast_to(np.asarray(p, dtype=float), (k, np.shape(p)[1]))
        out = np.full(p.shape, np.nan)
        out[p == 0] = 0.0
        out[p == 1] = np.inf
        inside = (p > 0) & (p < 1)
        if not inside.any():
            return out
        rows = np.nonzero(inside)[0]
        energy = -np.log1p(-p[inside])
        if self._family() == "ARI":
            from .ari import ari_gaps

            out[inside] = ari_gaps(
                self.model.dist,
                np.asarray(self.model.params, dtype=float),
                states.now[rows],
                states.reduction[rows],
                energy,
                polish=True,
            )
        else:
            scale = states.scale[rows]
            out[inside] = scale * conditional_gaps_from_energy(
                self.model, states.age[rows], energy
            )
        return out

    def _simulate_from_states(
        self,
        states: UnitStates,
        until: float,
        items: int,
        random_state: Any,
        tol: float = 1e-8,
        max_events: int = 10_000,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Simulate ``items`` futures of every unit from its state, over
        ``(now, now + until]``. Returns, for each simulated failure, its
        unit, its replicate and its time ahead of now."""
        from surpyval.recurrent.simulation import (
            MAX_EVENTS_WARNING,
            STALLED_WARNING,
            simulate_sequences,
        )
        from surpyval.utils.rng import as_generator

        k = len(states.now)
        rows = np.repeat(np.arange(k), items)
        run = simulate_sequences(
            self._sampler_factory(self, rows.size, states.take(rows)),
            rows.size,
            as_generator(random_state),
            close=np.full(rows.size, float(until)),
            tol=tol,
            max_events=max_events,
        )
        if run.stalled:
            warnings.warn(STALLED_WARNING, stacklevel=3)
        if run.hit_max_events:
            warnings.warn(MAX_EVENTS_WARNING.format(max_events), stacklevel=3)
        failed = run.c == 0
        sequence = run.i[failed]
        return rows[sequence], sequence % items, run.x[failed]

    def __repr__(self) -> str:
        title = f"{self.kind} SurPyval Model"
        lines = [
            title,
            "=" * len(title),
            f"{self._dist_label:<20}: {self.model.dist.name}",
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
        lines.append(f"{self._restoration_label:<20}: {self.restoration}")
        if not hasattr(self, "_neg_ll"):
            # Built from parameters (or restored): no standard errors.
            param_string = "\n".join(
                f"{name:>10}" + ": " + str(p)
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
        check_alpha_ci(alpha_ci)
        self._check_inference()
        with ignoring_warnings():
            # The boundary is reported in the table (nan), not warned.
            cov = self.covariance()
        level = f"{100 * (1 - alpha_ci):g}%"
        rows = []
        for k, (value, (lo, hi)) in enumerate(
            zip(self._mle, self._parameter_bounds())
        ):
            var = float(cov[k, k])
            if k == 0 and self._edge_value() is not None:
                with ignoring_warnings():
                    cb = self.param_cb(self._restoration_param_name, alpha_ci)
                rows.append([value, np.nan, cb[0], cb[1]])
                continue
            if not var > 0:
                rows.append([value, np.nan, np.nan, np.nan])
                continue
            with ignoring_warnings():
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

        A model restored with ``from_dict`` returns the covariance it was
        saved with.
        """
        if not hasattr(self, "_neg_ll") and self._saved_covariance is not None:
            # The saved one, edge row and all: as the fit gave it.
            return np.array(self._saved_covariance, dtype=float)
        self._check_fitted()
        edge = self._edge_value()
        if edge is None:
            return super().covariance()
        mle = self._mle_values()

        def neg_ll_rest(params: np.ndarray) -> float:
            return self._neg_ll(np.r_[mle[0], params])

        H = scale_free_hessian(neg_ll_rest, mle[1:])
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
        method: str = "wald",
    ) -> np.ndarray:
        """
        Confidence bound(s) on a fitted parameter.

        Wald bounds from the observed information, on a scale chosen from
        the parameter's range so they stay inside it: log for one bounded
        below (the Kijima ``q``, a positive scale), logit for one bounded
        on both sides (``rho`` of ARA and ARI), natural otherwise.

        The Wald interval on the restoration parameter can under-cover in
        samples of the usual size: for a Kijima-I ``q`` of 0.4 with six
        units and about 70 failures, a nominal 90% interval covered 83%
        of the time (#665). ``method="lr"`` gives its profile-likelihood
        interval instead, as below, from the estimate out on each side.

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
        method : {'wald', 'lr'}, optional
            ``"wald"`` (the default) or ``"lr"``, the profile-likelihood
            interval, which is available for the restoration parameter
            only (the others are refused). On the edge of its range the
            restoration parameter's interval is the profile-likelihood one
            either way.

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
        if method not in ("wald", "lr"):
            raise option_error("method", method, ["wald", "lr"])
        restoration = name == self._restoration_param_name
        if method == "lr" and not restoration and name in self.parameter_names:
            raise ValueError(
                "method='lr' is available for the restoration parameter "
                "{!r} only; use method='wald' for {!r}.".format(
                    self._restoration_param_name, name
                )
            )
        edge = self._edge_value()
        if not restoration or (edge is None and method == "wald"):
            return super().param_cb(name, alpha_ci, bound)
        self._check_fitted()
        check_alpha_ci(alpha_ci)
        alpha, signs = bound_signs(alpha_ci, bound)
        crit = float(chi2.ppf(1.0 - 2.0 * alpha, 1)) if alpha < 0.5 else 0.0
        lower, upper = self._restoration_bounds
        if edge is None:
            # Inside its range (#665): from the estimate out on each side.
            start = float(self._mle[0])
            return np.array(
                [self._profile_end(start, crit, float(s)) for s in signs]
            )
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
        """The end, away from ``edge`` (the edge of the range the
        estimate is on, or the estimate itself), of the profile-likelihood
        interval: where twice the drop of the profile log-likelihood from
        its maximum reaches ``crit``; the far end of the range if it never
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
            with ignoring_warnings(), np.errstate(all="ignore"):
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
        with ignoring_warnings(), np.errstate(all="ignore"):
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
        # full one's except by the optimisers' tolerance, and a difference
        # within that tolerance either way is 0 (the restricted fit is the
        # full one). It matters at a boundary, where the p-value jumps from
        # 1 at a statistic of 0 to 0.5 just above it: a perfect-repair fit
        # (rho = 1 estimated) gave 1.1e-13 on some CPUs, so p = 0.5.
        stat = 2.0 * (ll_full - ll_restricted)
        if stat <= 1e-9 * max(1.0, abs(ll_full)):
            stat = 0.0
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
