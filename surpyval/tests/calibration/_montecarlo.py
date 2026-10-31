"""Monte Carlo acceptance rules shared by the calibration studies.

Each study simulates ``reps`` data sets from a model whose truth is known,
applies a method to each, and compares a Monte Carlo summary (a coverage, a
rejection rate, a standardised bias) with the value the method promises.

The tolerance rule
------------------
A summary passes when it is within

    ``Z_TOL`` Monte Carlo standard errors + ``slack``

of its target. The Monte Carlo standard error is that of the summary under
the *target* value (for a rate, ``sqrt(p (1 - p) / reps)`` at the nominal
``p``), so the band does not widen because a method is off. ``Z_TOL = 3``:
a correct method lands outside it with probability about 0.3% per check.

``slack`` is the allowance for the method being *approximate* at a finite
sample size -- the asymptotic interval that is 94.3% at n = 100 is not
wrong. It is set per study, stated beside the study, and kept at 0.01 for
coverages and sizes unless the study says otherwise; the sample sizes are
chosen large enough that a correct method is well inside that. The past
bugs this suite is designed around are far larger than 3 SE + slack: a
nominal 95% band that was really a 90% band, a test of nominal size 5%
rejecting a true null up to 89% of the time.

Every study uses a fixed seed, so a run is deterministic: a check that
passes today passes tomorrow unless the code changes. The 0.3% figure above
is how likely a correct method would have been to fail for the seed chosen,
not a flake rate.

Each check prints a one-line summary (run pytest with ``-rP`` to see them
for passing tests), so a run's log records how close every study is.
"""

import math
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext

import numpy as np

Z_TOL = 3.0


def process_pool(max_workers: "int | None" = None) -> ProcessPoolExecutor:
    """A process pool for a study's replicates, whose workers start fresh
    (``forkserver``) where the platform's default would fork them.

    Forking copies a process that is multi-threaded by then (the BLAS
    threads, and pytest-xdist's own under ``-n``), which Python warns may
    deadlock the child. On GitHub's Linux runners it made
    ``test_coverage_distributional`` about twice as slow as with fresh
    workers (its bootstrap study 1467 s against 687 s on 30 October 2026),
    and the calibration "rest" group ran past its 60-minute limit. macOS
    and Windows start fresh workers already (``spawn``), and keep it.
    Each replicate is computed from its own job, so the results are the
    same either way.
    """
    context: BaseContext
    if multiprocessing.get_start_method() == "fork":
        context = multiprocessing.get_context("forkserver")
    else:
        context = multiprocessing.get_context()
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=context)


def rate_se(p: float, reps: int) -> float:
    """Monte Carlo standard error of a proportion estimated from ``reps``."""
    return math.sqrt(p * (1.0 - p) / reps)


def _report(label: str, value: float, target: float, tol: float) -> str:
    return "{}: {:.4f} (target {:.4f}, tolerance +/- {:.4f})".format(
        label, value, target, tol
    )


def check_rate(
    hits: "int | float",
    reps: int,
    target: float,
    label: str,
    slack: float = 0.01,
    side: str = "both",
) -> float:
    """Assert that ``hits / reps`` is within the tolerance of ``target``.

    ``side="both"`` fails a rate too high or too low (a coverage or a test
    size), ``"lower"`` only a rate too low (a power that should be at least
    ``target``), ``"upper"`` only one too high. Returns the rate.
    """
    rate = float(hits) / reps
    tol = Z_TOL * rate_se(target, reps) + slack
    line = _report(label, rate, target, tol)
    print(line)
    if side in ("both", "lower"):
        assert rate >= target - tol, line
    if side in ("both", "upper"):
        assert rate <= target + tol, line
    return rate


def check_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: "float | np.ndarray",
    nominal: float,
    label: str,
    slack: float = 0.01,
) -> np.ndarray:
    """Coverage of ``truth`` by ``[lower, upper]`` over the first axis.

    ``lower`` and ``upper`` have one row per replicate (and optionally one
    column per quantity, each checked separately against ``nominal``). A
    ``nan`` bound counts as a miss: an interval the method failed to produce
    is not an interval that covered. Returns the coverages.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    reps = lower.shape[0]
    hit = (lower <= truth) & (truth <= upper)
    cov = np.atleast_1d(hit.mean(axis=0))
    tol = Z_TOL * rate_se(nominal, reps) + slack
    lines = [
        _report(f"{label} [{k}]", c, nominal, tol) for k, c in enumerate(cov)
    ]
    print("\n".join(lines))
    bad = [line for line, c in zip(lines, cov) if abs(c - nominal) > tol]
    assert not bad, "\n".join(bad)
    return cov


def check_bias(
    estimates: np.ndarray,
    truth: "float | np.ndarray",
    label: str,
    slack: float = 0.2,
    standard_errors: "np.ndarray | None" = None,
    se_slack: float = 0.1,
) -> None:
    """Parameter recovery: bias relative to the estimator's spread.

    ``estimates`` has one row per replicate and one column per parameter.
    The standardised bias ``(mean - truth) / sd`` has Monte Carlo standard
    error ``1 / sqrt(reps)``; it must be within ``Z_TOL`` of those plus
    ``slack``. ``slack`` (default 0.2 of a standard deviation) allows for
    the O(1/n) small-sample bias every MLE has, which is O(1/sqrt(n)) of its
    spread (see ``test_recovery``).

    With ``standard_errors`` (the per-replicate standard errors the model
    reports, same shape) the reported error is checked against the actual
    spread as well: ``rms(se) / sd`` must be within ``Z_TOL / sqrt(2 reps)``
    (the Monte Carlo error of an estimated sd, to first order) plus
    ``se_slack`` of 1.
    """
    est = np.atleast_2d(np.asarray(estimates, dtype=float))
    if est.shape[0] == 1:
        est = est.T
    reps = est.shape[0]
    ok = np.isfinite(est).all(axis=1)
    assert ok.mean() > 0.99, "{}: {} of {} fits failed".format(
        label, int((~ok).sum()), reps
    )
    est = est[ok]
    reps = est.shape[0]
    truth = np.broadcast_to(np.asarray(truth, dtype=float), est.shape[1:])
    sd = est.std(axis=0, ddof=1)
    z_bias = (est.mean(axis=0) - truth) / sd
    tol = Z_TOL / math.sqrt(reps) + slack
    bad = []
    for k in range(est.shape[1]):
        line = _report(f"{label} standardised bias [{k}]", z_bias[k], 0.0, tol)
        print(line)
        if abs(z_bias[k]) > tol:
            bad.append(line)
    if standard_errors is not None:
        se = np.atleast_2d(np.asarray(standard_errors, dtype=float))
        if se.shape[0] == 1:
            se = se.T
        se = se[ok]
        ratio = np.sqrt(np.nanmean(se**2, axis=0)) / sd
        tol_se = Z_TOL / math.sqrt(2.0 * reps) + se_slack
        for k in range(est.shape[1]):
            line = _report(f"{label} se / sd [{k}]", ratio[k], 1.0, tol_se)
            print(line)
            if abs(ratio[k] - 1.0) > tol_se:
                bad.append(line)
    assert not bad, "\n".join(bad)


def simulate_nhpp(
    rng: np.random.Generator,
    cif: Callable,
    inv_cif: Callable,
    systems: int,
    t_end: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Time-terminated NHPP data in ``xicn`` form.

    Each system's event count over ``[0, t_end]`` is Poisson with mean
    ``cif(t_end)`` and, given the count, its times are ``inv_cif`` of sorted
    uniforms on ``[0, cif(t_end)]``; one censoring row closes each system at
    ``t_end``. Written from the definition, not with the package's
    simulators, so a fault in those cannot cancel one in a fit.
    """
    xs: list = []
    ids: list = []
    cs: list = []
    total = cif(t_end)
    for k in range(systems):
        m = rng.poisson(total)
        times = np.sort(inv_cif(total * rng.uniform(size=m)))
        xs.extend([*times, t_end])
        ids.extend([k] * (m + 1))
        cs.extend([0] * m + [1])
    return np.array(xs), np.array(ids), np.array(cs)


def simulate_intervention(
    rng: np.random.Generator,
    scale: "float | np.ndarray",
    shape: float,
    ratio: float,
    interventions: "list | np.ndarray",
    ends: "list | np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Failure times of power-law systems whose intensity is multiplied by
    ``ratio`` after an intervention, in ``(x, i)`` form.

    System ``q`` (``i = q``) is observed on ``(0, ends[q]]`` with the
    cumulative intensity :math:`(t / \\text{scale}_q)^\\text{shape}`
    before ``interventions[q]`` and ``ratio`` times its increments after
    it (``shape = 1``: an HPP of rate ``1 / scale``). Given each period's
    Poisson count, its times are the inverse cumulative intensity of
    uniforms. Written from the definition, not with the package's
    simulators.
    """
    xs: list = []
    ids: list = []
    scales = np.broadcast_to(np.asarray(scale, dtype=float), len(ends))
    for q, (tau, end) in enumerate(zip(interventions, ends)):
        cut = min(tau, end)
        lo, hi = (cut / scales[q]) ** shape, (end / scales[q]) ** shape
        before = lo * rng.uniform(size=rng.poisson(lo))
        after = lo + (hi - lo) * rng.uniform(
            size=rng.poisson(ratio * (hi - lo))
        )
        times = scales[q] * np.concatenate([before, after]) ** (1 / shape)
        xs.extend(np.sort(times))
        ids.extend([q] * times.size)
    return np.array(xs), np.array(ids)
