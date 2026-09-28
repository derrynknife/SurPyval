"""Simulate and refit every registered model (#397).

The conformance registry (``surpyval/tests/conformance/registry.py``)
lists every public model kind with a fixture and a fit. Here each case
that can simulate from itself is taken through one loop:

1. the case's model fitted to its fixture is the truth;
2. ``reps`` data sets of ``n`` units (a few hundred) are drawn from it,
   with a fixed seed per case;
3. each is refitted by the case's own ``fit``;
4. two checks, at modest precision:

   - the mean estimate of each parameter is within ``Z_TOL / sqrt(reps)
     + slack`` standard deviations of the truth, the standard deviation
     being the spread of the estimates across the replicates
     (``_montecarlo.check_bias``; the model's own standard errors are
     principle 17's business, not this module's);
   - the mean refitted curve (``sf``, each cause's ``cif``, or the
     ``mcf`` / ``cif`` of a counting process) is within ``Z_TOL`` Monte
     Carlo standard errors plus ``curve_slack`` of the true curve at
     every point of a grid inside the data (5% to 95% of it), so its
     largest deviation, the sup norm, is bounded (:func:`check_curve`).

A likelihood term that is wrong but still "fits", and a simulator that
disagrees with its own model, both show up as a bias of several standard
deviations; a correct model's small-sample bias is within the slack
(``test_recovery`` explains the 0.2 for a maximum likelihood estimate).
For scale: refitting the ``Weibull[xcnt]`` draws with their delayed entry
ignored gives a shape bias of 1.44 standard deviations, and a Weibull
simulator 3% off in scale a bias of 1.04, against a tolerance of 0.5.

How each case's draws become data to refit -- the censoring, truncation,
covariate design, items or units -- is its :class:`Plan` in
:data:`PLANS`, keyed by the case name. The package's own simulators are
used wherever the model has one (``random``, ``random_data``, the
recurrent ``time_terminated_simulation_data``, the copula ``random``),
since whether they agree with their models is part of the question; a
model without one is drawn from by inverting its own survival function,
cumulative incidences or mean cumulative function (degradation readings
are simulated from the model's definition, since its ``random`` draws
lifetimes, not readings). Right censoring is
independent throughout: a unit's censoring time is the larger of two
further draws from its own distribution, which censors a third of the
units of a continuous model whatever its shape or support.

Semi- and non-parametric models (and those whose parameters change
meaning between data sets, like a spline's) have their curve checked
only. A parameter whose true value is on the boundary of its space (a
frailty variance or a restoration factor of 0, a repair efficiency of
1), or is an endpoint of the support (estimated by an extreme order
statistic, with a bias as large as its spread), is not held to the
bias rule; each plan's ``note`` says which and why. Every registered case
is either in :data:`PLANS` or in :data:`EXCLUDED` with a reason, which
``test_every_case_is_planned`` enforces.

Run with ``--run-calibration`` (nightly). The whole module is about 20
CPU-minutes, the slowest cases (ARA, GeneralizedOneRenewal, MixtureModel,
RoystonParmar) one to two minutes each: five to six minutes with
``-n auto`` on four cores.
"""

import math
import warnings
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from surpyval.tests.calibration._montecarlo import Z_TOL, check_bias
from surpyval.tests.conformance import registry as reg

CASE_BY_NAME = reg.CASE_BY_NAME


# ---------------------------------------------------------------------------
# The curve check
# ---------------------------------------------------------------------------
def check_curve(
    curves: np.ndarray,
    truth: np.ndarray,
    label: str,
    slack: float = 0.02,
) -> float:
    """The mean refitted curve against the true one, point by point.

    ``curves`` has one row per replicate and one column per grid point.
    At each point the mean over the replicates must be within ``Z_TOL``
    Monte Carlo standard errors (``sd / sqrt(reps)``) plus ``slack`` of
    the truth; ``slack`` allows for the O(1/n) bias of a smooth
    functional and for a step estimate read beside its step. Returns the
    sup norm of the mean curve's deviation.
    """
    curves = np.asarray(curves, dtype=float)
    truth = np.asarray(truth, dtype=float)
    reps = curves.shape[0]
    mean = curves.mean(axis=0)
    se = curves.std(axis=0, ddof=1) / math.sqrt(reps)
    tol = Z_TOL * se + slack
    gap = np.abs(mean - truth)
    worst = int(np.argmax(gap - tol))
    line = (
        "{} curve: sup |mean - truth| = {:.4f}; worst point {} of {}: "
        "{:.4f} against {:.4f} (tolerance +/- {:.4f})".format(
            label,
            float(gap.max()),
            worst,
            gap.size,
            mean[worst],
            truth[worst],
            tol[worst],
        )
    )
    print(line)
    assert np.isfinite(mean).all(), label + " curve: a refit gave NaN"
    assert (gap <= tol).all(), line
    return float(gap.max())


def check_params(
    estimates: np.ndarray, truth: np.ndarray, label: str, slack: float
) -> None:
    """``check_bias`` for the parameters that vary between refits; a
    parameter held fixed by the model (the unit scale of an accelerated
    life model, the ``n_trials`` of a Binomial, a point mass's location)
    must come back exactly."""
    est = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    fixed = np.isfinite(est).all(axis=0) & (np.ptp(est, axis=0) == 0)
    for k in np.flatnonzero(fixed):
        print("{} [{}] fixed at {:.6g}".format(label, k, est[0, k]))
        assert np.isclose(
            est[0, k], truth[k], rtol=1e-9
        ), "{} [{}]: every refit gave {} against the true {}".format(
            label, k, est[0, k], truth[k]
        )
    if (~fixed).any():
        check_bias(est[:, ~fixed], truth[~fixed], label, slack=slack)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
Draw = Callable[[np.random.Generator], dict]


@dataclass(frozen=True)
class Plan:
    """How one case is simulated, refitted and checked.

    ``simulate(case, truth, n)`` returns a function of a generator giving
    one data set (a dict for the case's ``fit``). ``params(model)`` is the
    parameter vector checked for bias (``None``: the curve only).
    ``curve(case, model, grid)`` and ``grid(case, truth, data)`` default
    to the interface's curve on a grid from the first data set's times.
    ``note`` names anything not checked, and why.
    """

    simulate: Callable[[Any, Any, int], Draw]
    n: int
    reps: int
    params: Callable[[Any], np.ndarray] | None = None
    curve: Callable[[Any, Any, np.ndarray], np.ndarray] | None = None
    grid: Callable[[Any, Any, dict], np.ndarray] | None = None
    # The true curve, when it is not ``curve`` of the truth.
    truth_curve: Callable[[Any, Any, np.ndarray], np.ndarray] | None = None
    slack: float = 0.2
    curve_slack: float = 0.02
    note: str = ""
    xfail: str = ""


def _censor(t: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> dict:
    """Right censoring of ``t`` at ``max(t1, t2)``: independent of ``t``,
    and a third of the units of a continuous model."""
    cens = np.maximum(t1, t2)
    # A unit that never fails (a step estimate that stops short of 0) is
    # censored; one censored at infinity, at the last finite time.
    out = (t > cens) | ~np.isfinite(t)
    x = np.where(out, cens, t)
    finite = np.isfinite(x)
    return {"x": np.where(finite, x, x[finite].max()), "c": out.astype(int)}


# --- univariate --------------------------------------------------------------
def _uni_censored(case, truth, n):
    def draw(rng):
        t = np.asarray(truth.random(3 * n), dtype=float).reshape(3, n)
        return _censor(*t)

    return draw


def _uni_plain(case, truth, n):
    return lambda rng: {"x": np.asarray(truth.random(n))}


def _uni_random_data(case, truth, n):
    # The model's own simulate-for-refit draw (#403): a unit that never
    # fails is right-censored after the last failure.
    def draw(rng):
        x, c, counts, _ = truth.random_data(n)
        return {"x": x, "c": c, "n": counts}

    return draw


def _np_censored(case, truth, n):
    # A step estimate's own sampler takes a generator.
    def draw(rng):
        t = truth.random(3 * n, random_state=rng).astype(float)
        return _censor(*t.reshape(3, n))

    return draw


def _step(truth) -> float:
    """The inspection interval of :func:`_xcnt`: a third of the median."""
    return float(np.ravel(truth.qf(np.array([0.5])))[0]) / 3.0


def _xcnt(sample, truth, n, rng):
    """Every kind of observation, with delayed entry.

    Four in ten units enter late (uniform up to the 20th percentile) and
    are seen only if they survive to entry; half the units are observed
    exactly, the rest at inspections a third of the median apart
    (interval censored, or left censored before the first inspection
    after entry); all are right censored as in :func:`_censor`, from
    their entry. ``sample(size, rng)`` draws from the model.
    """
    q20 = float(np.ravel(truth.qf(np.array([0.2])))[0])
    step = _step(truth)
    entry = np.where(rng.uniform(size=n) < 0.4, rng.uniform(0, q20, n), 0.0)
    t = sample(n, rng)
    while (t <= entry).any():
        redo = t <= entry
        t[redo] = sample(int(redo.sum()), rng)
    cens = entry + np.maximum(sample(n, rng), sample(n, rng))
    exact = rng.uniform(size=n) < 0.5
    out = t > cens
    # (lo, hi] holds t even when t is on an inspection (a discrete model)
    lo = (np.ceil(t / step) - 1) * step
    hi = np.minimum(lo + step, cens)
    left = ~exact & ~out & (lo <= entry)
    inter = ~exact & ~out & ~left
    x = np.column_stack([t, t])
    x[out] = cens[out, None]
    x[left] = hi[left, None]
    x[inter] = np.column_stack([lo, hi])[inter]
    c = np.zeros(n, int)
    c[out], c[left], c[inter] = 1, -1, 2
    return {"x": x, "c": c, "tl": entry}


def _inspected(case, truth, n):
    """A point mass at T can only be fitted from censored data: each unit
    is inspected once, at a uniform time on (0, 2 T), and is found failed
    (left censored) or not (right censored)."""
    T = float(truth.params[0])

    def draw(rng):
        s = rng.uniform(0, 2 * T, n)
        t = np.asarray(truth.random(n), dtype=float)
        return {"x": s, "c": np.where(t <= s, -1, 1)}

    return draw


def _xcnt_parametric(case, truth, n):
    return lambda rng: _xcnt(
        lambda k, g: np.asarray(truth.random(k), dtype=float), truth, n, rng
    )


def _xcnt_turnbull(case, truth, n):
    return lambda rng: _xcnt(
        lambda k, g: truth.random(k, random_state=g).astype(float),
        truth,
        n,
        rng,
    )


def _np_law(truth):
    """The law a step estimate's ``random`` draws from: its jumps at the
    distinct observed values, renormalised to sum to one when the
    estimate stops short of 0 (documented in ``NonParametric.random``)."""
    with np.errstate(all="ignore"):
        p = -np.diff(np.r_[1.0, truth.R])
    p = np.where(np.isfinite(p), p, 0.0)
    x, where = np.unique(truth.x, return_inverse=True)
    mass = np.bincount(where.ravel(), weights=p / p.sum())
    return x, mass


def _np_truth(case, truth, grid):
    """What a step estimate refitted to draws of ``random`` converges to:
    the survival function of :func:`_np_law`, or, for Nelson-Aalen,
    ``exp`` of minus its cumulative hazard (the Fleming-Harrington
    correction for ties makes that estimate the product limit again)."""
    x, mass = _np_law(truth)
    after = np.clip(1.0 - np.cumsum(mass), 0.0, 1.0)
    if case.name == "NelsonAalen":
        with np.errstate(all="ignore"):
            after = np.exp(-np.cumsum(mass / (after + mass)))
    k = np.searchsorted(x, grid, side="right") - 1
    return np.where(k >= 0, after[np.maximum(k, 0)], 1.0)


def _inspection_grid(case, truth, data):
    """The inspection times inside the data: where interval-censored data
    identify a non-parametric estimate."""
    x = np.asarray(data["x"], dtype=float).ravel()
    lo, hi = np.quantile(x[np.isfinite(x)], [0.05, 0.95])
    step = _step(truth)
    return step * np.arange(np.ceil(lo / step), np.floor(hi / step) + 1)


def _parametric_params(model):
    out = list(np.atleast_1d(model.params))
    if model.offset:
        out = [model.gamma] + out
    if model.lfp:
        out.append(model.p)
    if model.zi:
        out.append(model.f0)
    return np.array(out, dtype=float)


def _mixture_params(model):
    # Components ordered by scale, so the labels cannot switch.
    order = np.argsort(model.params[:, 0])
    return np.r_[model.params[order].ravel(), model.w[order][0]]


# --- covariates: drawing at the fixture's rows ------------------------------
def _dense_grid(S: Callable[[np.ndarray], np.ndarray], scale: float, size):
    """A grid from where ``S`` is 1 to where it is 0 (to 1e-7), or to
    where it stops falling (a step estimate that stops short of 0)."""
    hi = scale
    for _ in range(80):
        s_hi, s_next = S(np.array([hi, 2.0 * hi]))
        if s_hi <= 1e-7 or s_next >= s_hi:
            break
        hi *= 2.0
    if S(np.array([0.0]))[0] >= 1.0 - 1e-9:
        return np.r_[0.0, np.geomspace(hi * 1e-7, hi, size)]
    lo = -scale
    for _ in range(80):
        if S(np.array([lo]))[0] >= 1.0 - 1e-7:
            break
        lo *= 2.0
    return np.linspace(lo, hi, size)


def _inverse(S, scale, step=False, size=6000):
    """The generalised inverse of a survival curve ``S``: ``u`` -> the
    first time ``S`` is at or below ``u`` (``inf`` if it never is).

    A continuous ``S`` is inverted by linear interpolation on a dense
    grid; a step function (``step``) by the first grid point at or below,
    so every draw of one step lands on the same time.
    """
    grid = _dense_grid(S, scale, size if not step else 40000)
    s = np.minimum.accumulate(np.asarray(S(grid), dtype=float))

    def inv(u):
        if step:
            k = np.searchsorted(-s, -u, side="left")
            return np.where(
                k < s.size, grid[np.minimum(k, s.size - 1)], np.inf
            )
        return np.interp(u, s[::-1], grid[::-1])

    return inv


def _design(case, n):
    """The fixture's covariate rows (and group or stratum labels),
    repeated to ``n`` rows; groups are renumbered per copy so each copy's
    groups are new ones."""
    d = case.data()
    copies = max(1, n // d["Z"].shape[0])
    out = {"Z": np.tile(d["Z"], (copies, 1))}
    for key in ("groups", "strata"):
        if key in d:
            labels = np.asarray(d[key])
            width = labels.max() + 1
            out[key] = np.concatenate(
                [labels + width * j for j in range(copies)]
                if key == "groups"
                else [labels] * copies
            )
    return out


def _regression(own_random, step=False, keep=None, positive=False):
    """Draws at the fixture's rows, by the model's ``random`` where it
    has one, else by inverting its ``sf`` at each row. ``keep(truth, Z)``
    selects the rows to draw at. ``positive``: the draws are conditioned
    on a positive time and fitted as left truncated at 0 (see
    ``test_additive_hazards_random_below_zero``)."""

    def simulate(case, truth, n):
        design = _design(case, n)
        if keep is not None:
            rows = keep(truth, design["Z"])
            design = {k: v[rows] for k, v in design.items()}
        Z = design["Z"]
        strata = design.get("strata")
        keys = np.column_stack([Z, strata]) if strata is not None else Z
        unique, where = np.unique(keys, axis=0, return_inverse=True)
        where = where.ravel()
        scale = float(np.max(case.data()["x"]))
        samplers = []
        for key in unique:
            row = key[: Z.shape[1]]
            if own_random:
                samplers.append(
                    lambda size, g, row=row: _own_draws(
                        truth, size, row, positive
                    )
                )
                continue
            kw = {} if strata is None else {"stratum": key[-1]}

            def S(t, row=row, kw=kw):
                return _sf_at(case, truth, t, row, kw)

            inv = _inverse(S, scale, step=step)
            samplers.append(lambda size, g, inv=inv: inv(g.uniform(size=size)))

        def draw(rng):
            t = np.empty((3, Z.shape[0]))
            for k, sample in enumerate(samplers):
                rows = where == k
                t[:, rows] = sample(3 * rows.sum(), rng).reshape(3, -1)
            out = {**design, **_censor(*t)}
            if positive:
                out["t"] = np.tile([0.0, np.inf], (Z.shape[0], 1))
            return out

        return draw

    return simulate


def _own_draws(truth, size, row, positive):
    """``size`` draws of the model's ``random`` at one covariate row,
    redrawing any at or below ``TINY`` when ``positive``."""
    t = np.asarray(truth.random(size, row[None, :])[0], dtype=float)
    while positive and (t <= TINY).any():
        low = t <= TINY
        t[low] = np.asarray(truth.random(int(low.sum()), row[None, :])[0])
    return t


# Below this a draw of an additive-hazards model is the bisection's floor.
TINY = 1e-9


def _sf_at(case, model, t, row, kw=None):
    """``sf`` of a covariate model at one row, over the times ``t``."""
    if kw:
        return np.asarray(model.sf(t, row, **kw), dtype=float)
    return np.asarray(reg.call(case, model, "sf", t, Z=row), dtype=float)


def _positive_additive_rows(truth, Z):
    # An additive-hazards model is a distribution only where its hazard
    # h0(t) + beta'Z stays positive (documented; the registry excludes
    # "bounds" for it): the rows with beta'Z >= 0.
    beta = np.asarray(truth.params)[-Z.shape[1] :]
    return Z @ beta >= 0


def _frailty_params(model):
    # theta (the frailty variance) is last; see the plan's note.
    return np.asarray(model._param_vector(), dtype=float)[:-1]


# --- competing risks ---------------------------------------------------------
def _cif_rows(case, truth, rows, support):
    """Per cause, the cumulative incidence at ``support`` for each row."""
    out = []
    for row in rows:
        if case.interface == reg.CAUSES:
            row = None
        # FineGray has one cause of interest ("a"); see _cr below.
        events = case.events or (None,)
        cifs = [
            reg.call(case, truth, "cif", support, Z=row, event=e)
            for e in events
        ]
        out.append(np.maximum.accumulate(np.asarray(cifs, float), axis=1))
    return out


def _cr(case, truth, n):
    """Causes and times drawn from the model's cumulative incidences.

    For a step estimate the incidences jump at the fixture's times: a
    unit's time is the first at which the incidences' sum reaches a
    uniform, its cause chosen in proportion to the causes' jumps there;
    beyond their sum it has no event. A one-cause (Fine-Gray) model
    leaves the rest free: those units fail from the other cause at an
    exponential time with the median fixture time as mean.
    """
    d = case.data()
    support = np.unique(d["x"])
    with_Z = case.interface == reg.CAUSES_REGRESSION
    if with_Z:
        design = _design(case, n)
        rows, where = np.unique(design["Z"], axis=0, return_inverse=True)
        where = where.ravel()
    else:
        design, rows, where = {}, np.zeros((1, 1)), np.zeros(n, int)
    cifs = _cif_rows(case, truth, rows, support)
    causes = list(case.events) if case.events else ["a"]
    size = where.size

    def one(rng, count, cif):
        total = cif.sum(axis=0)
        assert total[-1] <= 1 + 1e-9, "cumulative incidences exceed one"
        u = rng.uniform(size=count)
        k = np.searchsorted(total, u, side="left")
        event = k < support.size
        k = np.minimum(k, support.size - 1)
        jumps = np.diff(np.column_stack([np.zeros(len(cif)), cif]), axis=1)
        p = jumps[:, k] / np.maximum(jumps[:, k].sum(axis=0), 1e-300)
        pick = (rng.uniform(size=count) > np.cumsum(p, axis=0)).sum(axis=0)
        t = np.where(event, support[k], np.inf)
        e = np.array([causes[min(j, len(causes) - 1)] for j in pick], object)
        if not case.events:
            # the other cause, free under a one-cause model
            other = rng.exponential(np.median(support), count)
            t = np.where(event, t, other)
            e = np.where(event, e, "b")
        return t, e

    def draw(rng):
        t = np.empty((3, size))
        e = np.empty(size, object)
        for j, cif in enumerate(cifs):
            rows_j = where == j
            m = int(rows_j.sum())
            draws = [one(rng, m, cif) for _ in range(3)]
            t[:, rows_j] = np.array([dr[0] for dr in draws])
            e[rows_j] = draws[0][1]
        cens = np.maximum(t[1], t[2])
        cens = np.where(np.isfinite(cens), cens, support[-1])
        out = t[0] > cens
        e[out] = None
        x = np.where(out, cens, t[0])
        return {"x": x, "e": e, **design}

    return draw


def _pcr(case, truth, n):
    # The model's own sampler: (time, cause) records.
    def draw(rng):
        rec = truth.random(3 * n, random_state=rng)
        t = np.asarray(rec["x"], float).reshape(3, n)
        e = np.asarray(rec["e"], object)[:n].copy()
        cens = np.maximum(t[1], t[2])
        out = t[0] > cens
        e[out] = None
        return {"x": np.where(out, cens, t[0]), "e": e}

    return draw


def _pcr_params(model):
    return np.concatenate([model.models[k].params for k in model.causes])


# --- recurrent events --------------------------------------------------------
T_REC = 60.0  # the fixture's observation window


def _xicn(data) -> dict:
    return {"x": data.x, "i": data.i, "c": data.c, "n": data.n}


def _recurrent_own(case, truth, n):
    # The model's own time-terminated simulation; ``n`` is the items.
    def draw(rng):
        return _xicn(
            truth.time_terminated_simulation_data(T_REC, items=n, seed=rng)
        )

    return draw


def _pi_own(case, truth, n):
    # The fixture's items have covariates 0, 1 and 0.5: n / 3 items each.
    levels = np.unique(case.data()["Z"])

    def draw(rng):
        parts = []
        for k, z in enumerate(levels):
            d = truth.time_terminated_simulation_data(
                T_REC, Z=[z], items=n // len(levels), seed=rng
            )
            parts.append((d.x, d.i + 1000 * k, d.c, d.n, np.full(d.x.size, z)))
        x, i, c, counts, Z = (np.concatenate(p) for p in zip(*parts))
        return {"x": x, "i": i, "c": c, "n": counts, "Z": Z[:, None]}

    return draw


def _nhpp_events(rng, cif, t_end, items):
    """Event times of ``items`` NHPPs of cumulative intensity ``cif`` on
    [0, t_end], one list per item (the definition, as
    ``_montecarlo.simulate_nhpp``)."""
    grid = np.linspace(0.0, t_end, 20001)
    lam = np.maximum.accumulate(np.asarray(cif(grid), dtype=float))
    out = []
    for _ in range(items):
        m = rng.poisson(lam[-1])
        u = np.sort(rng.uniform(0, lam[-1], m))
        out.append(np.interp(u, lam, grid))
    return out


def _cause_events(case, truth):
    """Each cause's cumulative intensity: the model's own (parametric),
    or its MCF interpolated linearly through (0, 0) (a step cumulative
    intensity would put an item's events at one instant, which the xicn
    format refuses)."""
    if hasattr(truth, "mcf_hat") and not case.events:
        x, m = np.r_[0.0, truth.x], np.r_[0.0, truth.mcf_hat]
        return {None: lambda t: np.interp(t, x, m)}
    if case.name == "CauseSpecificMCF":
        out = {}
        for e in case.events:
            sub = truth.models[e]
            x, m = np.r_[0.0, sub.x], np.r_[0.0, sub.mcf_hat]
            out[e] = lambda t, x=x, m=m: np.interp(t, x, m)
        return out
    return {e: (lambda t, e=e: truth.cif(t, e)) for e in case.events}


def _interpolated_truth(case, truth, grid):
    """The linearly interpolated MCF the data are drawn from: the true
    curve of an MCF case, in :func:`default_curve`'s order."""
    return np.concatenate(
        [f(grid) for f in _cause_events(case, truth).values()]
    )


def _superposed(case, truth, n):
    """Items with one NHPP per cause, superposed, each closed by a
    censoring row at the window's end."""
    intensities = _cause_events(case, truth)

    def draw(rng):
        xs, ids, cs, es = [], [], [], []
        per_cause = {
            e: _nhpp_events(rng, cif, T_REC, n)
            for e, cif in intensities.items()
        }
        for k in range(n):
            times = [
                (t, e) for e, lists in per_cause.items() for t in lists[k]
            ]
            times.sort(key=lambda te: te[0])
            xs.extend([t for t, _ in times] + [T_REC])
            es.extend([e for _, e in times] + [None])
            ids.extend([k] * (len(times) + 1))
            cs.extend([0] * len(times) + [1])
        out = {
            "x": np.array(xs),
            "i": np.array(ids),
            "c": np.array(cs),
            "n": np.ones(len(xs), int),
        }
        if case.events:
            out["e"] = np.array(es, dtype=object)
        return out

    return draw


def _renewal_params(boundary: bool):
    # _mle is (restoration, alpha, beta); a restoration at its bound is
    # left out (see the plan's note).
    def params(model):
        mle = np.asarray(model._mle, dtype=float)
        return mle[1:] if boundary else mle

    return params


def _mcf_curve(items):
    # A renewal model's MCF is simulated: many items, a fixed seed.
    def curve(case, model, grid):
        return np.asarray(model.mcf(grid, items=items, seed=1), dtype=float)

    return curve


def _cause_nhpp_params(model):
    return np.concatenate([model.models[e].params for e in model.event_types])


# --- degradation -------------------------------------------------------------
def _paths(case, truth, n):
    """``n`` units read at the fixture's times, their path parameters
    from the fitted population ``N(path_param_mean, path_param_cov)`` and
    the readings with the fitted measurement error."""
    if case.name == "InducedFailureDistribution":
        truth = reg.fitted(CASE_BY_NAME["DegradationAnalysis[linear]"])
    times = np.unique(case.data()["x"])
    mean = np.asarray(truth.path_param_mean, dtype=float)
    cov = np.asarray(truth.path_param_cov, dtype=float)
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    root = v * np.sqrt(np.maximum(w, 0.0))
    sd = math.sqrt(truth.measurement_var)

    def draw(rng):
        theta = mean + rng.standard_normal((n, mean.size)) @ root.T
        x = np.tile(times, n)
        y = np.concatenate(
            [truth.path_model.path(times, *th) for th in theta]
        ) + rng.normal(0.0, sd, x.size)
        return {"x": x, "y": y, "i": np.repeat(np.arange(n), times.size)}

    return draw


def _path_params(model):
    return np.r_[
        model.path_param_mean,
        model.measurement_var,
        np.diag(model.path_param_cov),
    ]


N_INDUCED = 20000


def _induced_curve(case, model, grid):
    # The population life the path model induces (the pseudo-failure-time
    # Weibull is a summary of it, not the law the paths are drawn from).
    life = model.induced_life(n_samples=N_INDUCED, random_state=0)
    return np.asarray(life.sf(grid), dtype=float)


def _process(kind):
    """Units read at the fixture's times, with independent increments:
    Normal(mu dt, sigma^2 dt) for a Wiener process, Gamma(alpha dt, rate
    beta) for a gamma process."""

    def simulate(case, truth, n):
        times = np.unique(case.data()["x"])
        dt = np.diff(times)

        def draw(rng):
            if kind == "wiener":
                steps = rng.normal(
                    truth.mu * dt, truth.sigma * np.sqrt(dt), (n, dt.size)
                )
            else:
                steps = rng.gamma(
                    truth.alpha * dt, 1.0 / truth.beta, (n, dt.size)
                )
            y = np.column_stack([np.zeros(n), np.cumsum(steps, axis=1)])
            return {
                "x": np.tile(times, n),
                "y": y.ravel(),
                "i": np.repeat(np.arange(n), times.size),
            }

        return draw

    return simulate


def _destructive(case, truth, n):
    times = np.unique(case.data()["x"])
    x = np.repeat(times, n // times.size)

    def draw(rng):
        y = truth.degradation_quantile(rng.uniform(size=x.size), x)
        return {"x": x, "y": np.asarray(y, dtype=float)}

    return draw


def _destructive_params(model):
    return np.r_[model.beta, model.sigma]


def _quantile_grid(case, truth, data):
    # The data are readings, not lifetimes: the grid is the true life's
    # 5% to 95% quantiles.
    S = lambda t: np.asarray(truth.sf(t), dtype=float)  # noqa: E731
    if case.name == "InducedFailureDistribution":
        return truth.qf(np.linspace(0.05, 0.95, 10))
    inv = _inverse(S, float(np.max(case.data()["x"])))
    return inv(np.linspace(0.95, 0.05, 10))


# --- copulas -----------------------------------------------------------------
def _copula(case, truth, n):
    return lambda rng: {"x": truth.random(n, random_state=rng)}


def _copula_params(model):
    return np.r_[
        model.params, np.concatenate([m.params for m in model.margins])
    ]


def _copula_grid(case, truth, data):
    # Pairs of the margins' quartiles and median.
    q = np.quantile(data["x"], [0.25, 0.5, 0.75], axis=0)
    return np.array([[a, b] for a in q[:, 0] for b in q[:, 1]])


# ---------------------------------------------------------------------------
# Curves and grids by interface
# ---------------------------------------------------------------------------
def _z_rows(case):
    return np.unique(case.Z, axis=0)[:3]


def default_curve(case, model, grid):
    """The case's survival-type curve on ``grid``: ``sf``; each cause's
    ``cif``; a counting process's ``mcf`` (or ``cif``); at up to three of
    the case's covariate rows for a covariate model."""
    face = case.interface
    if face == reg.BIVARIATE:
        return np.asarray(model.sf(grid), dtype=float)
    if face in (reg.COUNTING, reg.COUNTING_REGRESSION):
        fname = "mcf" if "mcf" in case.functions else "cif"
    elif face in (reg.CAUSES, reg.COUNTING_CAUSES):
        fname = case.event_functions[0]
    elif face == reg.CAUSES_REGRESSION:
        fname = "cif"
    else:
        fname = "sf"
    rows = _z_rows(case) if face in reg.WITH_COVARIATES else [None]
    events = case.events if fname in case.event_functions else (None,)
    out = [
        np.asarray(reg.call(case, model, fname, grid, Z=row, event=e), float)
        for row in rows
        for e in events
    ]
    return np.concatenate(out)


def default_grid(case, truth, data):
    """Ten points from the 5th to the 95th percentile of the first data
    set's observed times (its window, for a counting process)."""
    if case.interface in (
        reg.COUNTING,
        reg.COUNTING_REGRESSION,
        reg.COUNTING_CAUSES,
    ):
        return np.linspace(0.05 * T_REC, 0.95 * T_REC, 10)
    x = np.asarray(data["x"], dtype=float).ravel()
    x = x[np.isfinite(x)]
    grid = np.quantile(x, np.linspace(0.05, 0.95, 10))
    if not case.continuous:
        grid = np.unique(np.round(grid))
    return grid


# ---------------------------------------------------------------------------
# The plan of every case
# ---------------------------------------------------------------------------
_CENSORED = Plan(_uni_censored, 300, 100, params=_parametric_params)
_ENDPOINTS = (
    "the support's endpoints are estimated by extreme order statistics, "
    "whose bias is as large as their spread (O(1/n) both), so they are "
    "not held to the bias rule"
)

PLANS: dict[str, Plan] = {}
for _name in (
    "Weibull",
    "Exponential",
    "Gamma",
    "LogNormal",
    "LogLogistic",
    "ExpoWeibull",
    "Rayleigh",
    "Normal",
    "Gumbel",
    "GumbelLEV",
    "Logistic",
    "Beta",
    "ConformanceGompertz",
    "Poisson",
    "Geometric",
    "NegativeBinomial",
    "DiscreteWeibull",
    "Discretize(Weibull)",
):
    PLANS[_name] = _CENSORED
PLANS["ConformanceGompertz"] = Plan(
    _uni_censored, 300, 40, params=_parametric_params
)
PLANS["Uniform"] = Plan(_uni_censored, 300, 100, note=_ENDPOINTS)
PLANS["Beta4"] = Plan(
    _uni_censored,
    300,
    40,
    note="the curve only: " + _ENDPOINTS + "; and the fixture's alpha is "
    "1 (a density positive at a), so alpha and beta are estimated jointly "
    "with a and inherit their bias (1.5 and 1.2 sd at n = 300)",
)
PLANS["BetaGeometric"] = Plan(
    _uni_censored,
    300,
    100,
    note="the fixture's fit is at the geometric limit (alpha, beta ~ 1e5 "
    "with a fixed ratio), where only the ratio is identified: the curve "
    "only",
)
for _name in ("Binomial", "Bernoulli", "FixedEventProbability"):
    PLANS[_name] = Plan(_uni_plain, 300, 100, params=_parametric_params)
PLANS["ExactEventTime"] = Plan(_inspected, 300, 100, params=_parametric_params)
for _base in ("Weibull", "Exponential", "Gamma", "LogNormal"):
    for _variant in ("offset", "lfp", "zi"):
        PLANS[f"{_base}[{_variant}]"] = Plan(
            _uni_random_data, 300, 60, params=_parametric_params
        )
# A threshold (offset) where the density is positive or has a shape
# below 2 is an endpoint too (Smith 1985): Exponential and Weibull with
# the fixture's shape of 1.79. Gamma (shape 4) and LogNormal are regular.
for _base in ("Weibull", "Exponential"):
    PLANS[f"{_base}[offset]"] = Plan(
        _uni_random_data,
        300,
        60,
        params=lambda m: np.asarray(m.params, dtype=float),
        note="gamma: " + _ENDPOINTS + " (a threshold whose density is "
        "positive, or of shape < 2, is non-regular)",
    )
PLANS["Weibull[xcnt]"] = Plan(
    _xcnt_parametric, 300, 100, params=_parametric_params
)
_RENORMALISED = (
    "random() draws from the estimate renormalised over the observed "
    "values when it stops short of 0 (documented), so the true curve is "
    "that law's"
)
for _name in ("KaplanMeier", "NelsonAalen", "FlemingHarrington"):
    PLANS[_name] = Plan(
        _np_censored, 300, 100, truth_curve=_np_truth, note=_RENORMALISED
    )
PLANS["Turnbull"] = Plan(
    _xcnt_turnbull,
    300,
    60,
    grid=_inspection_grid,
    truth_curve=_np_truth,
    note=_RENORMALISED + "; read at the inspection times, where interval "
    "censored data identify it",
)
PLANS["MixtureModel"] = Plan(_uni_censored, 300, 60, params=_mixture_params)
PLANS["RoystonParmar"] = Plan(
    _uni_censored,
    300,
    60,
    note="the spline's knots are placed at the data's quantiles, so its "
    "coefficients mean something else in every refit: the curve only",
)

# Regression: n = 300 is the fixture's 30 covariate rows ten times.
_REG_N, _REG_REPS = 300, 40
_REAL_LINE = ("Normal", "Gumbel", "Logistic")
for _kind in ("PH", "AFT", "PO", "AH"):
    for _base in reg.BASELINES:
        PLANS[_base + _kind] = Plan(
            _regression(
                own_random=_kind in ("PH", "AH"),
                keep=_positive_additive_rows if _kind == "AH" else None,
                positive=_kind == "AH" and _base in _REAL_LINE,
            ),
            _REG_N,
            _REG_REPS,
            params=lambda m: np.asarray(m.params, dtype=float),
            note=(
                "drawn at the fixture's rows where beta'Z >= 0 only: "
                "elsewhere the fitted hazard h0(t) + beta'Z is negative "
                "early on (documented), so the model is not a distribution"
                + (
                    "; conditioned on t > 0 and fitted as left truncated "
                    "there (see test_additive_hazards_random_below_zero)"
                    if _base in _REAL_LINE
                    else ""
                )
                if _kind == "AH"
                else ""
            ),
        )
for _lm in reg.LIFE_MODELS + reg.DUAL_LIFE_MODELS:
    PLANS[f"WeibullAL[{_lm}]"] = Plan(
        _regression(own_random=True),
        _REG_N,
        _REG_REPS,
        params=lambda m: np.asarray(m.params, dtype=float),
    )
for _base in ("Weibull", "Exponential", "Gamma", "LogNormal"):
    PLANS[_base + "Frailty"] = Plan(
        _regression(own_random=False),
        _REG_N,
        _REG_REPS,
        params=_frailty_params,
        note="the fixture's frailty variance is 0, on its boundary: the "
        "data are drawn from the (then frailty-free) sf, and theta, whose "
        "estimate has a mass at 0 and is biased upwards by design, is not "
        "held to the bias rule",
    )
_SEMI = "semi-parametric: the curve only"
for _name in ("CoxPH", "CoxPH[strata]", "BuckleyJames"):
    PLANS[_name] = Plan(
        _regression(own_random=False, step=True),
        _REG_N,
        _REG_REPS,
        note=_SEMI,
    )

for _name in ("CompetingRisks[Nelson-Aalen]", "CompetingRisks[Kaplan-Meier]"):
    PLANS[_name] = Plan(_cr, 300, 60)
PLANS["ParametricCompetingRisks"] = Plan(_pcr, 300, 60, params=_pcr_params)
for _name in (
    "CompetingRisksProportionalHazards[Cox]",
    "CompetingRisksProportionalHazards[Fine-Gray]",
    "FineGray",
):
    PLANS[_name] = Plan(_cr, 300, 40, note=_SEMI)

# Recurrent events: n is the number of items, observed over [0, 60].
for _name in ("HPP", "CrowAMSAA", "Duane", "CoxLewis"):
    PLANS[_name] = Plan(
        _recurrent_own, 40, 100, params=lambda m: np.asarray(m.params)
    )
for _name in ("ProportionalIntensityHPP", "ProportionalIntensityNHPP"):
    PLANS[_name] = Plan(_pi_own, 42, 60, params=lambda m: np.asarray(m._mle))
# Items simulated for a renewal model's MCF (the true one included).
N_MCF = 5000
_AT_BOUND = (
    "the fixture's restoration is on its boundary ({}), where the "
    "estimate has a mass and is biased by design, so only alpha and beta "
    "are held to the bias rule"
)
for _name, _bound in (
    ("GeneralizedRenewal", "q = 0"),
    ("GeneralizedRenewal[kijima ii]", "q = 0"),
    ("ARA", "rho = 1"),
    ("ARI", "rho = 1"),
):
    PLANS[_name] = Plan(
        _recurrent_own,
        40,
        30,
        params=_renewal_params(boundary=True),
        curve=_mcf_curve(N_MCF),
        note=_AT_BOUND.format(_bound),
    )
PLANS["GeneralizedOneRenewal"] = Plan(
    _recurrent_own,
    40,
    20,
    params=_renewal_params(boundary=False),
    curve=_mcf_curve(N_MCF),
)
PLANS["NonParametricCounting"] = Plan(
    _superposed,
    40,
    60,
    truth_curve=_interpolated_truth,
    note="drawn from the fixture's MCF interpolated linearly (see "
    "_cause_events); the curve only",
)
PLANS["CauseSpecificMCF"] = Plan(
    _superposed,
    40,
    60,
    truth_curve=_interpolated_truth,
    note=PLANS["NonParametricCounting"].note,
)
PLANS["CauseSpecificNHPP"] = Plan(
    _superposed, 40, 60, params=_cause_nhpp_params
)

# Degradation: n is the number of units.
for _name in ("DegradationAnalysis[linear]", "DegradationAnalysis[power]"):
    PLANS[_name] = Plan(
        _paths,
        40,
        60,
        params=_path_params,
        curve=_induced_curve,
        grid=_quantile_grid,
        note="the curve is the induced life (the population's), not the "
        "pseudo-failure-time Weibull, which only summarises it",
    )
PLANS["InducedFailureDistribution"] = Plan(
    _paths,
    40,
    60,
    grid=_quantile_grid,
    note="its paths are drawn from the DegradationAnalysis[linear] fit it "
    "is induced from; a Monte Carlo distribution: the curve only",
)
PLANS["WienerProcess"] = Plan(
    _process("wiener"),
    30,
    100,
    params=lambda m: np.asarray(m.params),
    grid=_quantile_grid,
)
PLANS["GammaProcess"] = Plan(
    _process("gamma"),
    30,
    100,
    params=lambda m: np.asarray(m.params),
    grid=_quantile_grid,
)
PLANS["DestructiveDegradation"] = Plan(
    _destructive,
    200,
    100,
    params=_destructive_params,
    grid=_quantile_grid,
)

for _name in ("Independence", "Clayton", "Gumbel", "Frank", "Gaussian"):
    PLANS[_name + "Copula"] = Plan(
        _copula, 300, 60, params=_copula_params, grid=_copula_grid
    )


_NO_DATA = "built from its parameters (or a constant), not fitted to data"
_ML = (
    "beta ML: a tree refitted to data drawn from a tree grows its own "
    "partition, so there is neither a parameter nor a fixed curve to "
    "recover"
)
EXCLUDED: dict[str, str] = {
    "Hypoexponential": _NO_DATA,
    "NeverOccurs": _NO_DATA,
    "InstantlyOccurs": _NO_DATA,
    "SurvivalTree[weibull]": _ML,
    "SurvivalTree[exponential]": _ML,
    "SurvivalTree[non-parametric]": _ML,
    "RandomSurvivalForest": _ML,
    "AdditiveHazards": "documented: the Lin-Ying cumulative hazard need "
    "not be monotone, and the fixture's sf exceeds 1 (1.054 at t = 1 at "
    "the row (0, -0.8)) even where beta'Z >= 0: not a distribution to "
    "draw from",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_every_case_is_planned():
    names = set(CASE_BY_NAME)
    assert not set(PLANS) & set(EXCLUDED), set(PLANS) & set(EXCLUDED)
    assert set(PLANS) | set(EXCLUDED) == names, {
        "neither planned nor excluded": sorted(
            names - set(PLANS) - set(EXCLUDED)
        ),
        "not registered": sorted((set(PLANS) | set(EXCLUDED)) - names),
    }


def _cases():
    out = []
    for key in sorted(PLANS):
        plan = PLANS[key]
        marks = []
        if plan.xfail:
            marks.append(pytest.mark.xfail(strict=True, reason=plan.xfail))
        out.append(pytest.param(key, id=key, marks=marks))
    return out


@pytest.mark.parametrize("name", _cases())
def test_refit(name):
    case, plan = CASE_BY_NAME[name], PLANS[name]
    truth = reg.fitted(case)
    seed = zlib.crc32(name.encode())
    state = np.random.get_state()
    # The package's samplers draw from the global stream; seed it too.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    curve = plan.curve or default_curve
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            draw = plan.simulate(case, truth, plan.n)
            data = draw(rng)
            grid = (plan.grid or default_grid)(case, truth, data)
            true_curve = (plan.truth_curve or curve)(case, truth, grid)
            estimates, curves, failures = [], [], []
            for r in range(plan.reps):
                if r:
                    data = draw(rng)
                try:
                    model = case.fit(data)
                except Exception as error:  # a refit that fails is a finding
                    failures.append(repr(error))
                    continue
                if plan.params is not None:
                    estimates.append(plan.params(model))
                curves.append(curve(case, model, grid))
    finally:
        np.random.set_state(state)
    if plan.note:
        print(name + ": " + plan.note)
    print(
        "{}: {} refits of n = {}; {} failed".format(
            name, plan.reps, plan.n, len(failures)
        )
    )
    assert len(failures) <= plan.reps // 100, failures[:3]
    if plan.params is not None:
        check_params(np.array(estimates), plan.params(truth), name, plan.slack)
    check_curve(np.array(curves), true_curve, name, plan.curve_slack)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            f"{base}AH",
            marks=pytest.mark.xfail(
                strict=True,
                reason="#441: an additive-hazards model on a baseline over "
                "the whole real line puts ff(0) of its mass below t = 0 "
                "(GumbelAH fixture: 0.036 at Z = (0, -0.8)), but random() "
                "returns 2.7e-20, the floor of its bisection, for those "
                "draws; refitted as failures at ~0 they bias GumbelAH's "
                "sigma by -4.5 sd (n = 300, 40 refits)",
            ),
        )
        for base in _REAL_LINE
    ],
)
def test_additive_hazards_random_below_zero(name):
    """``random`` of an additive-hazards model agrees with its own ``ff``
    at 0: the proportional-hazards sampler does (NormalPH: 0.0109 of the
    draws below 0 against ff(0) = 0.0116); the additive one has none."""
    case = CASE_BY_NAME[name]
    model = reg.fitted(case)
    row = np.array([0.0, -0.8])
    state = np.random.get_state()
    np.random.seed(397)
    try:
        t = np.asarray(model.random(20000, row[None, :])[0])
    finally:
        np.random.set_state(state)
    below = float(np.mean(t < 0))
    ff0 = float(np.ravel(model.ff(np.array([0.0]), row))[0])
    tol = Z_TOL * math.sqrt(ff0 * (1 - ff0) / t.size) + 0.002
    print(
        "{}: {:.4f} of the draws below 0; ff(0) = {:.4f}".format(
            name, below, ff0
        )
    )
    assert abs(below - ff0) <= tol
