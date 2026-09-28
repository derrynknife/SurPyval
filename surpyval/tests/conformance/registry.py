"""The model registry behind the conformance suite (#379).

Every public model *kind* is registered here once: how to fit it on a
small deterministic fixture (and by its alternate fit paths), how its
functions are called, and which of the generic properties apply to it.
The property modules beside this file run every registered case through
every property that applies, so a new model gets the whole battery by
being registered, and ``test_completeness.py`` fails when something
public is neither registered nor listed in :data:`OUT_OF_SCOPE` with a
reason.

Registering a model
-------------------
Add a :class:`Case` to :data:`CASES` (the family helpers below do most
of the work). A case names

- ``fitters``: the public names it covers, as dotted paths
  (``"surpyval.Weibull"``); and ``model_class``, the class of the
  fitted model, which the completeness test also counts as covered;
- ``data`` and ``fit``: a function returning the fixture, a dict of
  keyword arguments, and one turning such a dict into a fitted model
  (usually ``fitter.fit(**data)``). The metamorphic properties rewrite
  the dict -- permute the per-row arrays named in ``rows``, rescale the
  times named in ``times``, replace counts ``n`` by repeated rows -- and
  refit;
- ``interface`` and ``functions``: how predictions are called (see the
  interface constants) and which functions the model has;
- ``exclude``: property -> the reason it does not hold for this model.

A property that *should* hold but fails is not excluded: it goes in
:data:`KNOWN_FAILURES` (at the end of this file) with a one-line
description of the failure, which makes it a strict xfail -- the suite
stays green, and fails the day the bug is fixed, as the reminder to
delete the entry. A known failure of one fit path or one missing input is
keyed ``"fit_paths[<path>]"`` or ``"missing_fit[<key>]"``.

Fixtures are tiny and built without a random number generator (quantiles
of a known distribution, in a fixed order), so a failure reproduces
from the case alone.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

import surpyval as sp
from surpyval import degradation as dg
from surpyval import multivariate as mv
from surpyval import recurrent as rc
from surpyval.beta import ml
from surpyval.tests.conformance.leaks import quiet
from surpyval.univariate import competing_risks as cr

# ---------------------------------------------------------------------------
# Interfaces: how a model's functions are called
# ---------------------------------------------------------------------------
UNIVARIATE = "univariate"  # f(x)
REGRESSION = "regression"  # f(x, Z), one covariate row per time
CAUSES = "causes"  # all-cause f(x); per-cause cif(x, event)
CAUSES_REGRESSION = "causes-regression"  # f(x, Z); cif(x, Z, event)
COUNTING = "counting"  # recurrent events: cif / iif / mcf (x)
COUNTING_REGRESSION = "counting-regression"  # cif(x, Z)
COUNTING_CAUSES = "counting-causes"  # mcf / cif (x, cause)
BIVARIATE = "bivariate"  # copulas: f(X), X of shape (m, 2)

WITH_COVARIATES = (REGRESSION, CAUSES_REGRESSION, COUNTING_REGRESSION)

# ---------------------------------------------------------------------------
# Properties. The name is what ``exclude`` / ``KNOWN_FAILURES`` refer to.
# ---------------------------------------------------------------------------
PROPERTIES: dict[str, str] = {
    "sf_ff": "sf + ff == 1",
    "Hf_sf": "Hf == -log(sf)",
    "df_hf_sf": (
        "df == hf * sf (continuous); df(k) == hf(k) * sf(k - 1) (discrete)"
    ),
    "qf_ff": "qf(ff(x)) == x inside the support",
    "cif_sum": "the causes' cumulative incidences sum to 1 - sf",
    "scalar": "a scalar query agrees with the same query as a 1-D array",
    "array2d": "a 2-D query keeps its shape and agrees element-wise",
    "empty": "an empty query returns an empty result",
    "query_order": "permuting the query permutes the result",
    "row_independence": (
        "covariate rows evaluated together == one at a time; one row "
        "broadcasts over the times"
    ),
    "units": "rescaling the time unit rescales the model, nothing else",
    "row_order": "permuting the data rows does not change the fit",
    "counts": "count weights n == the same rows repeated",
    "bounds": "probabilities in [0, 1] and monotone in time",
    "missing_query": "a NaN time or probability gives NaN there only (#375)",
    "missing_covariate": "a NaN covariate gives NaN for its row only",
    "missing_fit": "a missing time raises; a missing covariate is dropped",
    "seed_global": "np.random.seed reproduces a draw",
    "seed_explicit": (
        "an explicit seed reproduces a draw, is default_rng(seed), and "
        "leaves the global stream alone"
    ),
    "serialise": "strict-JSON to_dict / from_dict keeps every prediction",
    "fit_paths": "the alternate fit paths give the same model",
    "warn_once": (
        "a fit or a prediction gives each of its deliberate warnings at "
        "most once"
    ),
    "outside_data": (
        "a data-bounded estimate starts at its initial value before the "
        "first time and holds (or is NaN) after the last, for every "
        "function alike"
    ),
    # The option sweeps of test_options.py, over each case's ``bounds``
    # (see :class:`Bound`), ``interp`` and ``estimators``.
    "cb_declared": (
        "every uncertainty method of the model (one taking alpha_ci, "
        "confidence or bound) is swept, or excluded with a reason"
    ),
    "cb_contains": "the bounds contain the estimate of the same function",
    "cb_range": (
        "lower <= upper, inside the function's range or the parameter's "
        "support"
    ),
    "cb_sides": (
        "a one-sided bound at alpha is that end of the two-sided bound at "
        "2 alpha"
    ),
    "cb_nested": "a smaller alpha_ci gives a wider interval around the other",
    "cb_centre": "as alpha_ci -> 1 the interval closes onto the estimate",
    "cb_transform": (
        "bounds on ff are 1 - those on sf and bounds on Hf -log of them, "
        "the ends swapped"
    ),
    "cb_shape": (
        "(n, 2) [lower, upper] two-sided, (n,) one-sided; a scalar query "
        "is a one-element query"
    ),
    "cb_api": (
        "an unknown bound= raises ValueError; on='R' and on='F' are "
        "on='sf' and on='ff'"
    ),
    "interp": (
        "each interp value gives a valid curve that agrees with the "
        "step estimate at its step times"
    ),
    "interp_refused": "an unknown interp value raises ValueError",
    "estimators": "each estimation option gives a valid model",
    "estimators_agree": (
        "the estimation options agree on a large sample from the model"
    ),
    # test_convergence.py, over each case's ``starve``.
    "convergence": (
        "a fit that cannot converge warns or raises ValueError, never "
        "returns silently; a fit does not return its initial guess"
    ),
}

# Properties that refit the model (the slow ones).
REFIT_PROPERTIES = frozenset(
    {
        "units",
        "row_order",
        "counts",
        "missing_fit",
        "fit_paths",
        "warn_once",
        "convergence",
    }
)

# Which interfaces each property applies to. A property also needs the
# functions it uses (see ``cases_for``), and a case can narrow this
# further, with a reason, through ``exclude``.
_EVERY = frozenset(
    {
        UNIVARIATE,
        REGRESSION,
        CAUSES,
        CAUSES_REGRESSION,
        COUNTING,
        COUNTING_REGRESSION,
        COUNTING_CAUSES,
        BIVARIATE,
    }
)
_SURVIVAL = frozenset({UNIVARIATE, REGRESSION, CAUSES, CAUSES_REGRESSION})
_APPLICABLE: dict[str, frozenset[str]] = {
    "sf_ff": _SURVIVAL,
    "Hf_sf": _SURVIVAL,
    "df_hf_sf": frozenset({UNIVARIATE, REGRESSION}),
    "qf_ff": frozenset({UNIVARIATE}),
    "cif_sum": frozenset({CAUSES, CAUSES_REGRESSION}),
    "scalar": _EVERY - {BIVARIATE},
    "array2d": frozenset({UNIVARIATE, CAUSES, COUNTING, COUNTING_CAUSES}),
    "row_independence": frozenset(WITH_COVARIATES),
    "missing_covariate": frozenset(WITH_COVARIATES),
    "outside_data": _EVERY - {BIVARIATE},
}
for _prop in PROPERTIES:
    _APPLICABLE.setdefault(_prop, _EVERY)


@dataclass(frozen=True)
class Case:
    """One registered model kind; see the module docstring."""

    name: str
    fitters: tuple[str, ...]
    model_class: str
    interface: str
    data: Callable[[], dict]
    fit: Callable[[dict], Any]
    functions: tuple[str, ...]
    x: np.ndarray
    # Covariate rows for the query, one per time in ``x``.
    Z: np.ndarray | None = None
    # Causes, and the functions called once per cause.
    events: tuple = ()
    event_functions: tuple[str, ...] = ()
    # Keyword arguments every prediction call gets (e.g. ``stratum``).
    call_kwargs: dict = field(default_factory=dict)
    # Alternate fit paths: name -> function of the fixture dict.
    paths: dict[str, Callable[[dict], Any]] = field(default_factory=dict)
    # Per-row arrays of the fixture (permuted / repeated together) and the
    # entries that are times (rescaled by the units property).
    rows: tuple[str, ...] = ("x", "c", "n")
    times: tuple[str, ...] = ("x",)
    # Row-level covariate matrix key and whether a missing covariate at fit
    # time drops the row (independent rows) or raises (part of a unit).
    covariates: str | None = None
    drops_missing_covariate: bool = True
    # Grouping labels (frailty group, stratum): a missing one drops its row.
    labels: tuple[str, ...] = ()
    # Functions of a step estimate that are the jumps between the query
    # points (Conventions, "Function Conventions"), so a value depends on
    # its neighbours in the query and not only on its own time.
    jump_functions: tuple[str, ...] = ()
    # How a covariate matrix is read: one row per time ("paired", the
    # regression convention), every row at every time ("grid", the trees),
    # or one vector per call ("single"); see :func:`call`.
    z_style: str = "paired"
    continuous: bool = True
    # draw(model, seed) -> numbers; ``explicit_seed`` when it takes a seed.
    draw: Callable[[Any, Any], Any] | None = None
    explicit_seed: bool = False
    # A refit of the fixture that cannot converge (test_convergence.py):
    # an iteration limit too small, a start far from the maximum, or data
    # whose likelihood has no maximum. ``None`` for a fit with nothing to
    # starve, which then excludes "convergence" with the reason.
    starve: Callable[[dict], Any] | None = None
    # Relative tolerance of the refit comparisons (an optimiser's answer
    # moves with its starting point; exact estimators get 1e-9).
    rtol: float = 1e-4
    exclude: dict[str, str] = field(default_factory=dict)
    # Filled from KNOWN_FAILURES.
    xfail: dict[str, str] = field(default_factory=dict)
    # Properties marked ``slow`` for this case (``"*"`` for all of them),
    # which the pull-request conformance job skips.
    slow: frozenset[str] = frozenset()
    # The option sweeps (test_options.py), filled in by ``_OPTIONS`` below:
    # the uncertainty methods, the values ``interp=`` takes, and the
    # estimation options of the fit (keyword -> values), with a function
    # of the fitted model giving a large sample for them to agree on.
    bounds: tuple["Bound", ...] = ()
    interp: tuple[str, ...] = ()
    estimators: dict[str, tuple] = field(default_factory=dict)
    estimators_large: dict[str, tuple] = field(default_factory=dict)
    large: Callable[[Any], dict] | None = None
    # How far (absolute) the estimators' predictions may differ on it.
    estimators_atol: float = 0.02

    def applies(self, prop: str) -> bool:
        if prop in ("seed_global", "seed_explicit") and self.draw is None:
            return False
        if prop == "seed_explicit" and not self.explicit_seed:
            return False
        if prop == "fit_paths" and not self.paths:
            return False
        if prop.startswith("cb_") and prop != "cb_declared":
            if not self.bounds:
                return False
        if prop.startswith("interp") and not self.interp:
            return False
        if prop.startswith("estimators") and not self.estimators:
            return False
        if prop == "estimators_agree" and self.large is None:
            return False
        if prop == "convergence" and self.starve is None:
            return False
        return self.interface in _APPLICABLE[prop] and prop not in (
            self.exclude
        )

    def is_slow(self, prop: str) -> bool:
        return "*" in self.slow or prop in self.slow

    def has(self, *functions: str) -> bool:
        """Whether the model has all of ``functions`` (plain or per cause)."""
        own = set(self.functions) | set(self.event_functions)
        return set(functions) <= own


def cases_for(prop, needs=(), where=None):
    """The cases ``prop`` applies to, as ``pytest.param`` objects.

    ``needs`` names functions the property uses (cases without them are
    left out); ``where`` is an optional further filter. A case's
    ``xfail`` entry for ``prop`` becomes a strict xfail mark, and a slow
    property of a case gets the ``slow`` mark.
    """
    import pytest

    params = []
    for case in CASES:
        if not case.applies(prop) or not case.has(*needs):
            continue
        if where is not None and not where(case):
            continue
        marks = []
        if prop in case.xfail:
            marks.append(
                pytest.mark.xfail(strict=True, reason=case.xfail[prop])
            )
        if case.is_slow(prop):
            marks.append(pytest.mark.slow)
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


# ---------------------------------------------------------------------------
# Calling a model
# ---------------------------------------------------------------------------
# Probabilities at which a quantile function is queried.
Q_PROBS = np.array([0.05, 0.2, 0.5, 0.8, 0.95])


def query(case, fname):
    """The default query of ``fname``: probabilities for ``qf``, else the
    case's times."""
    return Q_PROBS if fname == "qf" else case.x


def call_native(case, model, fname, x, Z=None, event=None):
    """``model.<fname>`` called exactly as its interface documents."""
    f = getattr(model, fname)
    kw = dict(case.call_kwargs)
    if case.interface in WITH_COVARIATES:
        Z = case.Z if Z is None else Z
        if event is None:
            return f(x, Z, **kw)
        return f(x, Z, event, **kw)
    if event is None:
        return f(x, **kw)
    return f(x, event, **kw)


def call(case, model, fname, x, Z=None, event=None):
    """``model.<fname>`` at ``x``, one value per time.

    For a model with covariates, ``Z`` (default: the case's query rows)
    holds one row per time, or is one vector for every time. Models whose
    documented call differs are adapted: a tree or forest evaluates every
    row at every time (``z_style="grid"``, the diagonal is taken), and a
    Buckley-James model takes one vector per call (``z_style="single"``).
    """
    if case.interface not in WITH_COVARIATES or case.z_style == "paired":
        return call_native(case, model, fname, x, Z, event)
    Z = np.asarray(case.Z if Z is None else Z, dtype=float)
    x = np.asarray(x, dtype=float)
    if Z.ndim == 1:
        return call_native(case, model, fname, x, Z, event)
    if case.z_style == "grid":
        grid = np.asarray(call_native(case, model, fname, x, Z, event))
        return np.diagonal(grid.reshape(Z.shape[0], x.size)).copy()
    return np.array(
        [
            np.asarray(
                call_native(case, model, fname, x[k : k + 1], Z[k], event)
            ).item()
            for k in range(x.size)
        ]
    )


def calls(case):
    """(function, cause) pairs: each plain function with ``None``, and
    each per-cause function with every cause."""
    out = [(f, None) for f in case.functions]
    out += [(f, e) for f in case.event_functions for e in case.events]
    return out


def predictions(case, model, x=None, Z=None):
    """Every function of the model at the query, keyed by name.

    ``x`` replaces the case's times (``qf`` is always queried at
    :data:`Q_PROBS`).
    """
    out = {}
    for fname in case.functions:
        xq = query(case, fname) if x is None or fname == "qf" else x
        out[fname] = np.asarray(call(case, model, fname, xq, Z), float)
    for fname in case.event_functions:
        for e in case.events:
            xq = case.x if x is None else x
            out[f"{fname}[{e}]"] = np.asarray(
                call(case, model, fname, xq, Z, event=e), float
            )
    return out


@functools.lru_cache(maxsize=None)
def _fitted(name):
    case = CASE_BY_NAME[name]
    with quiet():
        return case.fit(case.data())


def fitted(case):
    """The case's model fitted to its fixture (cached per session).

    The same object is shared by every test, so tests must not change it.
    """
    return _fitted(case.name)


def refit(case, data):
    """Fit the case to ``data``, silencing the optimisers' warnings.

    A raw numerical warning leaking from the package is not silenced
    (see ``leaks.py``).
    """
    with quiet():
        return case.fit(data)


# ---------------------------------------------------------------------------
# Fixtures (deterministic: quantiles of a known model, in a fixed order)
# ---------------------------------------------------------------------------
def quantiles(size, scale=10.0, shape=2.0):
    """``size`` Weibull(scale, shape) plotting-position quantiles."""
    u = (np.arange(1, size + 1) - 0.3) / (size + 0.4)
    return scale * (-np.log1p(-u)) ** (1.0 / shape)


def scramble(size):
    """A fixed, thoroughly unsorted order of ``range(size)``."""
    stride = next(s for s in (5, 7, 11, 13) if np.gcd(s, size) == 1)
    return (np.arange(size) * stride + 3) % size


def uni_data():
    """12 rows: exact and right-censored values, some with counts > 1."""
    x = np.round(quantiles(12), 3)
    c = np.zeros(12, int)
    c[[4, 9]] = 1
    n = np.ones(12, int)
    n[[2, 7]] = [2, 3]
    return {"x": x, "c": c, "n": n}


def xcnt_data():
    """Every censoring type, with left truncation: the full data model."""
    x = np.array(
        [
            [1.0, 1.0],
            [2.0, 3.5],
            [3.0, 3.0],
            [4.0, 7.0],
            [5.5, 5.5],
            [6.0, 6.0],
            [7.5, 12.0],
            [9.0, 9.0],
            [11.0, 11.0],
            [14.0, 14.0],
        ]
    )
    c = np.array([0, 2, 0, 2, 1, -1, 2, 0, 0, 1])
    n = np.array([1, 1, 2, 1, 1, 1, 1, 2, 1, 1])
    tl = np.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    return {"x": x, "c": c, "n": n, "tl": tl}


def offset_data():
    d = uni_data()
    d["x"] = d["x"] + 5.0
    return d


def lfp_data():
    """A population a third of which is still running long after the rest
    have failed: a limited failure population."""
    d = uni_data()
    return {
        "x": np.r_[d["x"], 40.0, 45.0],
        "c": np.r_[d["c"], 1, 1],
        "n": np.r_[d["n"], 4, 4],
    }


def zi_data():
    """Two dead-on-arrival zeros in front of the usual sample."""
    d = uni_data()
    return {
        "x": np.r_[0.0, d["x"]],
        "c": np.r_[0, d["c"]],
        "n": np.r_[2, d["n"]],
    }


def mixture_data():
    x = np.r_[np.linspace(2.0, 6.0, 10), np.linspace(20.0, 40.0, 10)]
    n = np.ones(20, int)
    n[[2, 15]] = 2
    return {"x": x, "c": np.zeros(20, int), "n": n}


def discrete_data(start=0):
    x = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8]) + start
    c = np.zeros(12, int)
    c[[5, 10]] = 1
    n = np.ones(12, int)
    n[[2, 7]] = [2, 3]
    return {"x": x, "c": c, "n": n}


def binary_data():
    return {
        "x": np.array([0, 1, 1, 0, 1, 1, 0]),
        "n": np.array([1, 2, 1, 1, 1, 3, 1]),
    }


def exact_event_data():
    return {
        "x": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        "c": np.array([1, 1, -1, -1, -1]),
        "n": np.array([1, 2, 1, 1, 1]),
    }


def unit_interval_data():
    x = np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8])
    n = np.ones(10, int)
    n[[2, 6]] = 2
    return {"x": x, "c": np.zeros(10, int), "n": n}


N_REG = 30


def reg_data():
    """30 rows, a binary and a continuous covariate, 5 right censored."""
    z0 = np.tile([0.0, 1.0], N_REG // 2)
    z1 = np.round(np.linspace(-1.0, 1.0, N_REG), 3)
    life = quantiles(N_REG, 1.0, 2.0)[scramble(N_REG)]
    x = np.round(10.0 * np.exp(-0.5 * z0 + 0.3 * z1) * life, 3)
    c = np.zeros(N_REG, int)
    c[::7] = 1
    n = np.ones(N_REG, int)
    n[[3, 11]] = 2
    return {"x": x, "Z": np.column_stack([z0, z1]), "c": c, "n": n}


def stress_data(columns=1):
    """Accelerated life data: life falls with stress (1, 2 or 3)."""
    d = reg_data()
    s = np.repeat([1.0, 2.0, 3.0], N_REG // 3)
    life = quantiles(N_REG, 1.0, 2.0)[scramble(N_REG)]
    d["x"] = np.round(30.0 * s**-1.2 * life, 3)
    stress = s[:, None]
    if columns == 2:
        stress = np.column_stack([s, np.tile([1.0, 2.0], N_REG // 2)])
    d["Z"] = stress
    return d


def grouped_reg_data():
    d = reg_data()
    d["groups"] = np.repeat(np.arange(6), N_REG // 6)
    return d


def stratified_reg_data():
    d = reg_data()
    d["strata"] = np.repeat([0, 1], N_REG // 2)
    return d


X_UNI = np.array([0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 25.0])
X_DISC = np.arange(0.0, 11.0)
X_REG = np.array([0.5, 2.0, 4.0, 6.0, 8.0, 11.0, 15.0, 22.0])
Z_REG = np.array(
    [
        [0.0, -0.8],
        [1.0, -0.5],
        [0.0, 0.0],
        [1.0, 0.3],
        [0.0, 0.9],
        [1.0, 1.0],
        [0.0, 0.4],
        [1.0, -0.2],
    ]
)
X_STRESS = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 22.0])
Z_STRESS = np.array([[1.0], [1.5], [2.0], [2.5], [3.0], [1.2]])
Z_STRESS2 = np.array(
    [[1.0, 1.0], [1.5, 2.0], [2.0, 1.0], [2.5, 2.0], [3.0, 1.0], [1.2, 2.0]]
)


def cr_data(with_Z=False):
    """24 rows, causes "a" and "b" and censored (``None``) rows."""
    size = 24
    x = np.round(quantiles(size, 10.0, 1.5), 3)
    e = np.array(["a", "b", "a", None] * (size // 4), dtype=object)
    e = e[scramble(size)]
    n = np.ones(size, int)
    n[[2, 5]] = 2
    d = {"x": x, "e": e, "n": n}
    if with_Z:
        d["Z"] = np.tile([0.0, 1.0, 1.0], size // 3)[scramble(size)][:, None]
    return d


X_CR = np.array([0.5, 2.0, 4.0, 7.0, 10.0, 14.0, 20.0, 30.0])
Z_CR = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.5], [1.0]])


def recurrent_data(with_Z=False, with_e=False):
    """Three repairable items, each ending with a censoring row."""
    x = np.array(
        [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0]
    )
    i = np.array([1] * 6 + [2] * 5 + [3] * 5)
    c = np.zeros(16, int)
    c[[5, 10, 15]] = 1
    d = {"x": x, "i": i, "c": c, "n": np.ones(16, int)}
    if with_Z:
        d["Z"] = np.array([0.0] * 6 + [1.0] * 5 + [0.5] * 5)[:, None]
    if with_e:
        d["e"] = np.array(
            ["a", "b", "a", "b", "a", None, "a", "b", "a", "b", None]
            + ["a", "b", "a", "a", None],
            dtype=object,
        )
    return d


X_REC = np.array([1.0, 5.0, 12.0, 25.0, 40.0, 55.0])
Z_REC = np.array([[0.0], [1.0], [0.5], [0.0], [1.0], [0.2]])


def path_data():
    """6 units, 10 readings each, degrading linearly towards 150."""
    x = np.tile(np.arange(100.0, 1100.0, 100.0), 6)
    i = np.repeat(np.arange(6), 10)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37, 0.33, 0.40], 10)
    start = np.repeat([8.0, 12.0, 10.0, 9.0, 13.0, 11.0], 10)
    y = start + slopes * x + 1.5 * np.sin(np.arange(60) * 1.7)
    return {"x": x, "y": y, "i": i}


def process_data():
    """5 units, 11 readings each, of a process with positive increments."""
    t = np.tile(np.arange(0.0, 110.0, 10.0), 5)
    i = np.repeat(np.arange(5), 11)
    steps = 5.0 + 3.0 * np.sin(np.arange(50.0)).reshape(5, 10)
    y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
    return {"x": t, "y": y, "i": i}


def destructive_data():
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 5)
    y = np.exp(4.0 - 0.02 * x + 0.1 * np.sin(np.arange(20.0)))
    return {"x": x, "y": y}


X_PATH = np.array([150.0, 250.0, 320.0, 400.0, 480.0, 600.0])
X_PROC = np.array([50.0, 120.0, 160.0, 200.0, 260.0, 350.0])
X_DESTR = np.array([5.0, 20.0, 40.0, 60.0, 80.0, 120.0])


def copula_data():
    u = (np.arange(1, 21) - 0.3) / 20.4
    swap = np.arange(20) ^ 1  # neighbours swapped: positively dependent
    x1 = 10.0 * (-np.log1p(-u)) ** 0.5
    x2 = 5.0 * (-np.log1p(-u[swap])) ** (1 / 1.4)
    n = np.ones(20, int)
    n[[3, 8]] = 2
    return {"x": np.round(np.column_stack([x1, x2]), 3), "n": n}


X_COP = np.array(
    [[2.0, 1.0], [5.0, 2.0], [8.0, 3.0], [10.0, 5.0], [14.0, 7.0], [3.0, 6.0]]
)


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------
def flat(result):
    """The numbers in a draw's result, as one float array."""
    if isinstance(result, tuple):
        return np.concatenate([flat(r) for r in result])
    if hasattr(result, "mcf_hat"):
        return np.r_[np.asarray(result.x, float), result.mcf_hat]
    arr = np.asarray(result)
    if arr.dtype.names:
        # (time, cause) records: the times, then each cause's position in
        # the sorted causes
        causes = sorted({str(e) for e in arr["e"]})
        codes = [causes.index(str(e)) for e in arr["e"]]
        return np.r_[np.asarray(arr["x"], float), codes]
    return arr.astype(float).ravel()


def _seeded(fit, seed=0):
    """``fit`` run under a fixed global seed, leaving the global state as
    it was (for models that draw from the global stream while fitting)."""

    def run(data):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return fit(data)
        finally:
            np.random.set_state(state)

    return run


# ---------------------------------------------------------------------------
# Family helpers
# ---------------------------------------------------------------------------
UNI_FUNCTIONS = ("sf", "ff", "df", "hf", "Hf")


def _fit(fitter, **fixed):
    # An entry of the data dict overrides a fixed option, so the option
    # sweeps can refit with another value (``{**data, "how": "MPS"}``).
    return lambda d: fitter.fit(**{**fixed, **d})


def _parametric_paths(fitter, **fixed):
    def from_df(d):
        df = pd.DataFrame({"x": d["x"], "c": d["c"], "n": d["n"]})
        return fitter.fit_from_df(df, x="x", c="c", n="n", **fixed)

    def from_params(d):
        # The structure is passed only where the model has it, as the
        # Conventions page does (from_params rejects an explicit f0=0 for
        # a distribution that does not start at 0, and gamma=0 for one on
        # the whole real line).
        m = fitter.fit(**d, **fixed)
        structure = {"gamma": m.offset, "p": m.lfp, "f0": m.zi}
        kw = {k: getattr(m, k) for k, on in structure.items() if on}
        return fitter.from_params(m.params, **kw)

    def from_surpyval_data(d):
        data = sp.SurpyvalData(d["x"], d["c"], d["n"])
        return fitter.fit_from_surpyval_data(data, **fixed)

    return {
        "fit_from_df": from_df,
        "from_params": from_params,
        "fit_from_surpyval_data": from_surpyval_data,
    }


def continuous(name, fitter=None, data=uni_data, x=X_UNI, **kw):
    fitter = getattr(sp, name) if fitter is None else fitter
    fixed = kw.pop("fixed", {})
    return Case(
        name=kw.pop("case_name", name),
        fitters=kw.pop("fitters", (f"surpyval.{name}",)),
        model_class="surpyval.Parametric",
        interface=UNIVARIATE,
        data=data,
        fit=_fit(fitter, **fixed),
        functions=UNI_FUNCTIONS + ("qf",),
        x=x,
        paths=kw.pop("paths", _parametric_paths(fitter, **fixed)),
        draw=kw.pop("draw", lambda m, s: m.random(15)),
        **kw,
    )


def discrete(name, fitter=None, start=0, **kw):
    fitter = getattr(sp, name) if fitter is None else fitter
    return Case(
        name=kw.pop("case_name", name),
        fitters=kw.pop("fitters", (f"surpyval.{name}",)),
        model_class="surpyval.Parametric",
        interface=UNIVARIATE,
        data=functools.partial(discrete_data, start),
        fit=_fit(fitter),
        functions=UNI_FUNCTIONS + ("qf",),
        x=X_DISC,
        continuous=False,
        paths=kw.pop("paths", _parametric_paths(fitter)),
        draw=lambda m, s: m.random(15),
        exclude={
            "units": "support is the integers; rescaling leaves the lattice",
            **kw.pop("exclude", {}),
        },
        **kw,
    )


def regression(name, fitter, data=reg_data, x=X_REG, Z=Z_REG, **kw):
    def from_df(d, formula=False):
        cols = [f"z{k}" for k in range(d["Z"].shape[1])]
        df = pd.DataFrame(d["Z"], columns=cols)
        df["x"], df["c"], df["n"] = d["x"], d["c"], d["n"]
        if formula:
            return fitter.fit_from_df(
                df, x_col="x", c_col="c", n_col="n", formula=" + ".join(cols)
            )
        return fitter.fit_from_df(
            df, x_col="x", Z_cols=cols, c_col="c", n_col="n"
        )

    paths = {
        "fit_from_df": from_df,
        "formula": functools.partial(from_df, formula=True),
    }
    return Case(
        name=name,
        fitters=kw.pop("fitters", (f"surpyval.{name}",)),
        model_class=kw.pop(
            "model_class", "surpyval.ParametricRegressionModel"
        ),
        interface=REGRESSION,
        data=data,
        fit=kw.pop("fit", _fit(fitter)),
        functions=kw.pop("functions", UNI_FUNCTIONS),
        x=x,
        Z=Z,
        paths=kw.pop("paths", paths),
        rows=kw.pop("rows", ("x", "Z", "c", "n")),
        covariates="Z",
        **kw,
    )


# Parametric regression: every kind with every baseline is registered, and
# every case gets the properties that need no refit. The refitting ones run
# on a pull request for the Weibull baseline of each kind (a scale family)
# and LogNormal AFT (a log-location-scale one); the other baselines are
# marked slow, which keeps the pull-request job fast and still runs them
# in the full suite.
BASELINES: tuple[str, ...] = ("Weibull", "LogNormal", "Exponential")
BASELINES += ("Gamma", "Normal", "Gumbel", "Logistic")
FAST_REGRESSIONS: tuple[str, ...] = ("WeibullPH", "WeibullAFT", "WeibullPO")
FAST_REGRESSIONS += ("WeibullAH", "LogNormalAFT")


def _regression_family():
    out = []
    for kind in ("PH", "AFT", "PO", "AH"):
        for base in BASELINES:
            name = base + kind
            slow = (
                frozenset() if name in FAST_REGRESSIONS else REFIT_PROPERTIES
            )
            exclude = {}
            if kind == "AH":
                exclude["bounds"] = (
                    "documented: nothing keeps h0(x) + beta'Z positive "
                    "between the observed times, so sf can exceed 1"
                )
            out.append(
                regression(name, getattr(sp, name), slow=slow, exclude=exclude)
            )
    return out


LIFE_MODELS = (
    "Power",
    "InversePower",
    "Eyring",
    "InverseEyring",
    "Linear",
    "ExponentialLifeModel",
    "InverseExponential",
)
DUAL_LIFE_MODELS = ("DualExponential", "DualPower", "PowerExponential")


def _accelerated_life_family():
    out = []
    for lm in LIFE_MODELS + DUAL_LIFE_MODELS:
        dual = lm in DUAL_LIFE_MODELS
        fitter = sp.AcceleratedLife(sp.Weibull, getattr(sp, lm))
        out.append(
            regression(
                f"WeibullAL[{lm}]",
                fitter,
                data=functools.partial(stress_data, 2 if dual else 1),
                x=X_STRESS,
                Z=Z_STRESS2 if dual else Z_STRESS,
                fitters=(f"surpyval.{lm}",)
                + (("surpyval.AcceleratedLife",) if lm == "Power" else ()),
                slow=frozenset() if lm == "Power" else REFIT_PROPERTIES,
            )
        )
    return out


def _frailty_family():
    out = []
    for base in ("Weibull", "Exponential", "Gamma", "LogNormal"):
        fitter = getattr(sp, base + "Frailty")

        def from_df(d, fitter=fitter):
            df = pd.DataFrame(d["Z"], columns=["z0", "z1"])
            df["x"], df["c"], df["n"] = d["x"], d["c"], d["n"]
            df["g"] = d["groups"]
            return fitter.fit_from_df(
                df,
                x_col="x",
                group_col="g",
                Z_cols=["z0", "z1"],
                c_col="c",
                n_col="n",
            )

        out.append(
            regression(
                base + "Frailty",
                fitter,
                data=grouped_reg_data,
                fitters=(f"surpyval.{base}Frailty",)
                + (("surpyval.Frailty",) if base == "Weibull" else ()),
                model_class="surpyval.FrailtyModel",
                rows=("x", "Z", "c", "n", "groups"),
                labels=("groups",),
                paths={"fit_from_df": from_df},
                slow=frozenset() if base == "Weibull" else REFIT_PROPERTIES,
            )
        )
    return out


def _cox_paths():
    def from_df(d):
        df = pd.DataFrame(d["Z"], columns=["z0", "z1"])
        df["x"], df["c"], df["n"] = d["x"], d["c"], d["n"]
        return sp.CoxPH.fit_from_df(
            df, x_col="x", Z_cols=["z0", "z1"], c_col="c", n_col="n"
        )

    def tvc(d):
        # One (0, x] interval per subject is the time-fixed model.
        return sp.CoxPH.fit_tvc(
            np.arange(d["n"].sum()),
            np.zeros(d["n"].sum()),
            np.repeat(d["x"], d["n"]),
            np.repeat(d["c"], d["n"]),
            np.repeat(d["Z"], d["n"], axis=0),
        )

    return {"fit_from_df": from_df, "fit_tvc": tvc}


def _semi_parametric():
    step = (
        "the baseline is a step function: hf and df are its jumps, not "
        "a hazard rate and a density"
    )
    cox = regression(
        "CoxPH",
        sp.CoxPH,
        model_class="surpyval.SemiParametricRegressionModel",
        paths=_cox_paths(),
        jump_functions=("hf", "df"),
        exclude={"df_hf_sf": step},
        rtol=1e-6,
    )
    strat = regression(
        "CoxPH[strata]",
        sp.CoxPH,
        data=stratified_reg_data,
        fitters=(),
        model_class="surpyval.SemiParametricRegressionModel",
        call_kwargs={"stratum": 1},
        jump_functions=("hf", "df"),
        rows=("x", "Z", "c", "n", "strata"),
        labels=("strata",),
        paths={},
        exclude={
            "df_hf_sf": step,
            "serialise": "a stratified Cox model cannot be saved (documented"
            " in Conventions, 'Saving and Loading Models')",
        },
        rtol=1e-6,
    )
    ah = regression(
        "AdditiveHazards",
        sp.AdditiveHazards,
        model_class="surpyval.AdditiveHazardsModel",
        paths={
            "fit_from_df": lambda d: sp.AdditiveHazards.fit_from_df(
                pd.DataFrame(
                    {"x": d["x"], "c": d["c"], "n": d["n"]}
                    | {"z0": d["Z"][:, 0], "z1": d["Z"][:, 1]}
                ),
                x_col="x",
                Z_cols=["z0", "z1"],
                c_col="c",
                n_col="n",
            )
        },
        exclude={
            "df_hf_sf": step,
            "bounds": "documented: the Lin-Ying cumulative hazard need "
            "not be monotone and the implied survival can exceed 1",
        },
        rtol=1e-6,
    )
    bj = regression(
        "BuckleyJames",
        sp.BuckleyJames,
        model_class="surpyval.BuckleyJamesModel",
        functions=("sf", "ff", "Hf"),
        # Documented to take one covariate vector per call.
        z_style="single",
        paths={
            "fit_from_df": lambda d: sp.BuckleyJames.fit_from_df(
                pd.DataFrame(
                    {"x": d["x"], "c": d["c"], "n": d["n"]}
                    | {"z0": d["Z"][:, 0], "z1": d["Z"][:, 1]}
                ),
                x_col="x",
                Z_cols=["z0", "z1"],
                c_col="c",
                n_col="n",
            )
        },
        exclude={
            "df_hf_sf": "BuckleyJamesModel has no hf or df",
            "row_independence": "documented to take one covariate vector "
            "per call, so there are no rows to mix up",
        },
    )
    return [cox, strat, ah, bj]


def _trees():
    def tree(kind):
        return Case(
            name=f"SurvivalTree[{kind}]",
            fitters=(
                ("surpyval.beta.ml.SurvivalTree",) if kind == "weibull" else ()
            ),
            model_class="surpyval.beta.ml.SurvivalTree",
            interface=REGRESSION,
            data=reg_data,
            # A tree draws the features it considers at each split from the
            # global stream (n_features_split="sqrt"), so it is fitted
            # under a fixed global seed, as the forest is.
            fit=_seeded(_fit(ml.SurvivalTree, kind=kind)),
            functions=UNI_FUNCTIONS,
            x=X_REG,
            Z=Z_REG,
            rows=("x", "Z", "c", "n"),
            covariates="Z",
            z_style="grid",
            jump_functions=("hf", "df") if kind == "non-parametric" else (),
            # The draw is the fit itself: two fits under one seed agree.
            draw=lambda m, s: ml.SurvivalTree.fit(**reg_data(), kind=kind).sf(
                X_REG, Z_REG
            ),
            exclude=(
                {"df_hf_sf": "non-parametric leaves: hf and df are jumps"}
                if kind == "non-parametric"
                else {}
            ),
        )

    forest = Case(
        name="RandomSurvivalForest",
        fitters=("surpyval.beta.ml.RandomSurvivalForest",),
        model_class="surpyval.beta.ml.RandomSurvivalForest",
        interface=REGRESSION,
        data=reg_data,
        # The forest bootstraps from the global stream (it has no seed
        # argument), so the registry fits it under a fixed global seed.
        fit=_seeded(_fit(ml.RandomSurvivalForest, n_trees=3)),
        functions=UNI_FUNCTIONS,
        x=X_REG,
        Z=Z_REG,
        rows=("x", "Z", "c", "n"),
        covariates="Z",
        z_style="grid",
        # Each refit grows every tree again: left to the full suite.
        slow=REFIT_PROPERTIES,
        # The draw is the fit itself: two seeded fits must agree.
        draw=lambda m, s: m.__class__.fit(**reg_data(), n_trees=2).sf(
            X_REG, Z_REG
        ),
        exclude={
            "row_order": "the bootstrap draws rows by position, so a "
            "permutation changes which rows each tree sees",
            "counts": "the bootstrap draws rows, so a count of 2 and two "
            "rows are resampled differently",
            "df_hf_sf": "the forest averages the trees' sf, hf and df "
            "separately (documented), and the average of hf is not the "
            "hazard of the average sf",
            "Hf_sf": "Hf is documented as the trees' average Hf, not -log "
            "of the average sf (sf(ensemble_method='Hf') is exp(-Hf))",
        },
    )
    return [
        tree("weibull"),
        tree("exponential"),
        tree("non-parametric"),
        forest,
    ]


# ---------------------------------------------------------------------------
# Univariate cases
# ---------------------------------------------------------------------------
def _gompertz_fun(x, *params):
    from autograd import numpy as anp

    return params[0] * (anp.exp(params[1] * x) - 1)


GOMPERTZ = sp.CustomDistribution(
    "ConformanceGompertz",
    _gompertz_fun,
    ["nu", "b"],
    ((0, None), (0, None)),
    (0, np.inf),
)


def _nonparametric(name, data=uni_data, **kw):
    fitter = getattr(sp, name)

    def from_surpyval_data(d):
        data = sp.SurpyvalData(d["x"], d["c"], d["n"], tl=d.get("tl"))
        return fitter.fit(x=data.x, c=data.c, n=data.n, t=data.t)

    return Case(
        name=kw.pop("case_name", name),
        fitters=kw.pop("fitters", (f"surpyval.{name}",)),
        model_class="surpyval.NonParametric",
        interface=UNIVARIATE,
        data=data,
        fit=_fit(fitter),
        functions=UNI_FUNCTIONS + ("qf",),
        x=X_UNI,
        rows=kw.pop("rows", ("x", "c", "n")),
        times=kw.pop("times", ("x",)),
        paths={"surpyval_data": from_surpyval_data},
        draw=lambda m, s: m.random(15, random_state=s),
        explicit_seed=True,
        jump_functions=("hf", "df"),
        rtol=1e-9,
        exclude={
            "df_hf_sf": "a step function: hf and df are the jumps between "
            "the query points (Conventions, 'Function Conventions')",
            "qf_ff": "a step function: qf is a generalised inverse, so "
            "qf(ff(x)) is the step at or below x, not x",
            **kw.pop("exclude", {}),
        },
        **kw,
    )


def _univariate():
    # Gauss and Galton are other names for Normal and LogNormal.
    alias = {"Normal": "Gauss", "LogNormal": "Galton"}
    out = []
    for name in (
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
    ):
        fitters = (f"surpyval.{name}",)
        if name in alias:
            fitters += (f"surpyval.{alias[name]}",)
        out.append(continuous(name, fitters=fitters))
    out.append(continuous("Uniform"))
    out.append(
        continuous(
            "Beta",
            data=unit_interval_data,
            x=np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]),
            exclude={"units": "supported on [0, 1], not a scale family"},
        )
    )
    out.append(
        continuous(
            "Beta4",
            data=unit_interval_data,
            x=np.array([0.15, 0.2, 0.35, 0.5, 0.65, 0.75]),
            slow=frozenset({"*"}),
        )
    )
    # offset / limited failure population / zero-inflation, for the
    # half-line families where each is supported
    for name in ("Weibull", "Exponential", "Gamma", "LogNormal"):
        for variant, data in (
            ("offset", offset_data),
            ("lfp", lfp_data),
            ("zi", zi_data),
        ):
            out.append(
                continuous(
                    name,
                    case_name=f"{name}[{variant}]",
                    fitters=(),
                    data=data,
                    fixed={variant: True},
                    x=X_UNI + (5.0 if variant == "offset" else 0.0),
                    slow=(
                        frozenset() if name == "Weibull" else REFIT_PROPERTIES
                    ),
                    # the lifetimes (inf for a unit that never fails)
                    # and the survival data to refit (#403)
                    draw=lambda m, s: (m.random(15), m.random_data(15)),
                )
            )
    out.append(
        continuous(
            "Weibull",
            case_name="Weibull[xcnt]",
            fitters=(),
            data=xcnt_data,
            rows=("x", "c", "n", "tl"),
            times=("x", "tl"),
            paths={},
        )
    )
    out.append(
        continuous(
            "ConformanceGompertz",
            fitter=GOMPERTZ,
            fitters=("surpyval.CustomDistribution",),
            # autograd through a user function: each refit takes ~0.4 s
            slow=REFIT_PROPERTIES,
        )
    )
    # discrete
    out.append(discrete("Poisson"))
    for name in ("Geometric", "NegativeBinomial", "DiscreteWeibull"):
        out.append(discrete(name, start=1))
    out.append(
        discrete(
            "BetaGeometric",
            start=1,
        )
    )
    discretized = sp.Discretize(sp.Weibull)
    out.append(
        discrete(
            "Discretize(Weibull)",
            fitter=discretized,
            start=1,
            fitters=(
                "surpyval.Discretize",
                "surpyval.DiscretizedFitter",
            ),
        )
    )
    out.append(
        Case(
            name="Binomial",
            fitters=("surpyval.Binomial",),
            model_class="surpyval.Parametric",
            interface=UNIVARIATE,
            data=lambda: {"x": np.array([2, 3, 1, 4, 3, 2])},
            fit=lambda d: sp.Binomial.fit(**d, n_trials=5),
            functions=UNI_FUNCTIONS + ("qf",),
            x=np.arange(0.0, 6.0),
            continuous=False,
            rows=("x",),
            paths={
                "from_params": lambda d: sp.Binomial.from_params(
                    sp.Binomial.fit(**d, n_trials=5).params
                )
            },
            draw=lambda m, s: m.random(15),
            exclude={
                "units": "counts of successes out of n_trials",
                "counts": "Binomial.fit takes no counts",
            },
        )
    )
    for name in ("Bernoulli", "FixedEventProbability"):
        fitter = getattr(sp, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.{name}",),
                model_class="surpyval.Parametric",
                interface=UNIVARIATE,
                data=binary_data,
                fit=_fit(fitter),
                functions=("sf", "ff", "Hf"),
                # Bernoulli is defined at the outcomes 0 and 1 only.
                x=(
                    np.array([0.0, 1.0])
                    if name == "Bernoulli"
                    else np.array([0.0, 1.0, 2.0, 3.0, 5.0])
                ),
                continuous=False,
                rows=("x", "n"),
                paths={
                    "from_params": lambda d, f=fitter: f.from_params(
                        f.fit(**d).params
                    )
                },
                draw=lambda m, s: m.random(15),
                exclude={
                    "units": "the outcomes are 0 and 1, not times",
                    "df_hf_sf": "only sf, ff and Hf are part of its model",
                },
            )
        )
    out.append(
        Case(
            name="ExactEventTime",
            fitters=("surpyval.ExactEventTime",),
            model_class="surpyval.Parametric",
            interface=UNIVARIATE,
            data=exact_event_data,
            fit=_fit(sp.ExactEventTime),
            functions=("sf", "ff", "Hf", "qf"),
            x=np.array([1.0, 3.0, 3.4, 3.6, 4.5, 7.0]),
            paths={
                "from_params": lambda d: sp.ExactEventTime.from_params(
                    sp.ExactEventTime.fit(**d).params
                )
            },
            draw=lambda m, s: m.random(15),
            exclude={
                "df_hf_sf": "a point mass: no density or hazard rate",
                "qf_ff": "a point mass: ff takes only the values 0 and 1",
                "seed_global": "a point mass: every draw is T, whatever "
                "the seed",
            },
        )
    )
    no_data = "built from its parameters, not fitted to data"
    out.append(
        Case(
            name="Hypoexponential",
            fitters=("surpyval.Hypoexponential",),
            model_class="surpyval.Parametric",
            interface=UNIVARIATE,
            data=dict,
            fit=lambda d: sp.Hypoexponential.from_params([0.5, 1.5, 3.0]),
            functions=UNI_FUNCTIONS + ("qf",),
            x=np.array([0.1, 0.5, 1.0, 2.0, 4.0, 8.0]),
            rows=(),
            draw=lambda m, s: m.random(15),
            exclude={p: no_data for p in REFIT_PROPERTIES},
        )
    )
    for name in ("NeverOccurs", "InstantlyOccurs"):
        cls = getattr(sp, name)
        out.append(
            Case(
                name=name,
                fitters=(),
                model_class=f"surpyval.{name}",
                interface=UNIVARIATE,
                data=dict,
                fit=lambda d, cls=cls: cls,
                functions=UNI_FUNCTIONS + ("qf",),
                x=np.array([0.0, 1.0, 5.0, 100.0]),
                rows=(),
                draw=lambda m, s: m.random(5),
                exclude={
                    **{p: no_data for p in REFIT_PROPERTIES},
                    "df_hf_sf": "a point mass at 0 or infinity",
                    "qf_ff": "a point mass at 0 or infinity",
                    "seed_global": "a point mass: every draw is the same",
                },
            )
        )
    # non-parametric
    out.append(_nonparametric("KaplanMeier"))
    out.append(_nonparametric("NelsonAalen"))
    out.append(_nonparametric("FlemingHarrington"))
    out.append(
        _nonparametric(
            "Turnbull",
            data=xcnt_data,
            rows=("x", "c", "n", "tl"),
            times=("x", "tl"),
        )
    )
    # mixtures and splines
    out.append(
        Case(
            name="MixtureModel",
            fitters=("surpyval.MixtureModel",),
            model_class="surpyval.MixtureModel",
            interface=UNIVARIATE,
            data=mixture_data,
            fit=lambda d: _fit_mixture(d),
            functions=("sf", "ff", "df", "Hf"),
            x=np.array([1.0, 3.0, 5.0, 10.0, 22.0, 30.0, 45.0]),
            draw=lambda m, s: m.random(15),
            exclude={
                "df_hf_sf": "MixtureModel has no hf (Conventions)",
                "qf_ff": "MixtureModel has no qf (Conventions)",
            },
        )
    )
    out.append(
        Case(
            name="RoystonParmar",
            fitters=("surpyval.RoystonParmar",),
            model_class="surpyval.RoystonParmarModel",
            interface=UNIVARIATE,
            data=uni_data,
            fit=_fit(sp.RoystonParmar),
            functions=UNI_FUNCTIONS + ("qf",),
            x=X_UNI,
            draw=lambda m, s: m.random(15),
        )
    )
    return out


def _fit_mixture(d):
    model = sp.MixtureModel(dist=sp.Weibull, m=2)
    model.fit(**d)
    return model


# ---------------------------------------------------------------------------
# Competing risks
# ---------------------------------------------------------------------------
def _competing_risks():
    def cr_from_df(d):
        df = pd.DataFrame({"x": d["x"], "e": d["e"], "n": d["n"]})
        return cr.CompetingRisks.fit_from_df(
            df, x_col="x", e_col="e", n_col="n"
        )

    def pcr_from_df(d):
        df = pd.DataFrame({"x": d["x"], "e": d["e"], "n": d["n"]})
        return cr.ParametricCompetingRisks.fit_from_df(
            df, x_col="x", e_col="e", n_col="n"
        )

    def crph_from_df(d, how="Cox"):
        df = pd.DataFrame(
            {"x": d["x"], "e": d["e"], "n": d["n"], "z0": d["Z"][:, 0]}
        )
        return cr.CompetingRisksProportionalHazards.fit_from_df(
            df, x_col="x", e_col="e", Z_cols=["z0"], n_col="n", how=how
        )

    out = [
        Case(
            name=f"CompetingRisks[{method}]",
            fitters=(
                ("surpyval.univariate.competing_risks.CompetingRisks",)
                if method == "Nelson-Aalen"
                else ()
            ),
            model_class="surpyval.univariate.competing_risks.CompetingRisks",
            interface=CAUSES,
            data=cr_data,
            fit=_fit(cr.CompetingRisks, method=method),
            functions=("sf", "ff", "Hf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "e", "n"),
            paths=(
                {"fit_from_df": cr_from_df} if method == "Nelson-Aalen" else {}
            ),
            exclude=(
                {
                    "cif_sum": "documented: sf and ff report exp(-H) under "
                    "the Nelson-Aalen method, while cif is always "
                    "Aalen-Johansen, which sums to 1 - Kaplan-Meier (the "
                    "Kaplan-Meier case checks that)"
                }
                if method == "Nelson-Aalen"
                else {}
            ),
            rtol=1e-9,
        )
        for method in ("Nelson-Aalen", "Kaplan-Meier")
    ]
    out.append(
        Case(
            name="ParametricCompetingRisks",
            fitters=(
                "surpyval.univariate.competing_risks.ParametricCompetingRisks",
            ),
            model_class=(
                "surpyval.univariate.competing_risks.ParametricCompetingRisks"
            ),
            interface=CAUSES,
            data=cr_data,
            fit=_fit(cr.ParametricCompetingRisks),
            functions=("sf", "ff", "Hf"),
            event_functions=("cif",),
            events=("a", "b"),
            x=X_CR,
            rows=("x", "e", "n"),
            paths={"fit_from_df": pcr_from_df},
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
        )
    )
    for how in ("Cox", "Fine-Gray"):
        out.append(
            Case(
                name=f"CompetingRisksProportionalHazards[{how}]",
                fitters=(
                    (
                        "surpyval.univariate.competing_risks"
                        ".CompetingRisksProportionalHazards",
                    )
                    if how == "Cox"
                    else ()
                ),
                model_class=(
                    "surpyval.univariate.competing_risks"
                    ".CompetingRisksProportionalHazards"
                ),
                interface=CAUSES_REGRESSION,
                data=functools.partial(cr_data, True),
                fit=_fit(cr.CompetingRisksProportionalHazards, how=how),
                # The Fine-Gray form has no all-cause survival; its sf is a
                # cause's 1 - cif, reached through cif below.
                functions=("sf", "ff") if how == "Cox" else (),
                event_functions=("cif",),
                events=("a", "b"),
                x=X_CR,
                Z=Z_CR,
                rows=("x", "Z", "e", "n"),
                covariates="Z",
                paths={
                    "fit_from_df": functools.partial(crph_from_df, how=how)
                },
                rtol=1e-6,
                exclude=(
                    {
                        "cif_sum": "Fine-Gray models each cause's "
                        "subdistribution separately; their CIFs need not "
                        "sum to one minus a survival"
                    }
                    if how == "Fine-Gray"
                    else {}
                ),
            )
        )
    out.append(
        Case(
            name="FineGray",
            fitters=("surpyval.univariate.competing_risks.FineGray",),
            model_class=(
                "surpyval.univariate.competing_risks.regression.fine_gray"
                ".FineGrayModel"
            ),
            interface=CAUSES_REGRESSION,
            data=functools.partial(cr_data, True),
            fit=_fit(cr.FineGray, cause="a"),
            functions=("sf", "cif"),
            x=X_CR,
            Z=Z_CR,
            rows=("x", "Z", "e", "n"),
            covariates="Z",
            rtol=1e-6,
            exclude={
                "cif_sum": "one cause of interest: sf is 1 - cif by "
                "definition, checked by sf_ff instead",
            },
        )
    )
    return out


# ---------------------------------------------------------------------------
# Recurrent events
# ---------------------------------------------------------------------------
def _recurrent_data_path(fitter, **fixed):
    def run(d):
        data = sp.handle_xicn(d["x"], d["i"], d["c"], d["n"])
        return fitter.fit_from_recurrent_data(data, **fixed)

    return run


def _counting_draw(m, s):
    return m.count_terminated_simulation(3, items=2, seed=s)


def _recurrent():
    out = []
    for name in ("HPP", "CrowAMSAA", "Duane", "CoxLewis"):
        fitter = getattr(rc, name)
        paths = {"fit_from_recurrent_data": _recurrent_data_path(fitter)}
        paths["from_params"] = lambda d, f=fitter: f.from_params(
            f.fit(**d).params
        )
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.recurrent.{name}",),
                model_class=(
                    "surpyval.recurrent.parametric.parametric_recurrence"
                    ".ParametricRecurrenceModel"
                ),
                interface=COUNTING,
                data=recurrent_data,
                fit=_fit(fitter),
                functions=("cif", "iif"),
                x=X_REC,
                rows=("x", "i", "c", "n"),
                paths=paths,
                draw=_counting_draw,
                explicit_seed=True,
            )
        )
    out.append(
        Case(
            name="NonParametricCounting",
            fitters=("surpyval.recurrent.NonParametricCounting",),
            model_class="surpyval.recurrent.NonParametricCounting",
            interface=COUNTING,
            data=recurrent_data,
            fit=_fit(rc.NonParametricCounting),
            functions=("mcf",),
            x=X_REC,
            rows=("x", "i", "c", "n"),
            paths={
                "fit_from_recurrent_data": _recurrent_data_path(
                    rc.NonParametricCounting
                )
            },
            rtol=1e-9,
        )
    )
    for name in ("ProportionalIntensityHPP", "ProportionalIntensityNHPP"):
        fitter = getattr(rc, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.recurrent.{name}",),
                model_class=(
                    "surpyval.recurrent.regression.proportional_intensity"
                    ".ProportionalIntensityModel"
                ),
                interface=COUNTING_REGRESSION,
                data=functools.partial(recurrent_data, True),
                fit=_fit(fitter),
                functions=("cif", "iif"),
                x=X_REC,
                Z=Z_REC,
                rows=("x", "Z", "i", "c", "n"),
                covariates="Z",
                drops_missing_covariate=False,
                draw=lambda m, s: m.count_terminated_simulation(
                    3, items=2, seed=s, Z=[0.5]
                ),
                explicit_seed=True,
            )
        )
    renewals = (
        ("GeneralizedRenewal", {}),
        ("GeneralizedRenewal", {"kijima": "ii"}),
        ("GeneralizedOneRenewal", {}),
        ("ARA", {"m": 1}),
        ("ARI", {"m": 1}),
    )
    for name, kw in renewals:
        fitter = getattr(rc, name)
        label = name + ("[kijima ii]" if kw.get("kijima") == "ii" else "")
        out.append(
            Case(
                name=label,
                fitters=(
                    (f"surpyval.recurrent.{name}",) if "[" not in label else ()
                ),
                model_class="surpyval.recurrent.RenewalModel",
                interface=COUNTING,
                data=recurrent_data,
                fit=_fit(fitter, **kw),
                # The MCF is simulated; a fixed seed makes it a function.
                functions=("mcf",),
                call_kwargs={"items": 30, "seed": 1},
                x=X_REC,
                rows=("x", "i", "c", "n"),
                paths={
                    "fit_from_recurrent_data": _recurrent_data_path(
                        fitter, **kw
                    )
                },
                draw=lambda m, s: m.mcf(X_REC, items=5, seed=s),
                explicit_seed=True,
                slow=REFIT_PROPERTIES,
            )
        )
    out.append(
        Case(
            name="CauseSpecificMCF",
            fitters=("surpyval.recurrent.CauseSpecificMCF",),
            model_class="surpyval.recurrent.CauseSpecificMCF",
            interface=COUNTING_CAUSES,
            data=functools.partial(recurrent_data, False, True),
            fit=_fit(rc.CauseSpecificMCF),
            functions=(),
            event_functions=("mcf",),
            events=("a", "b"),
            x=X_REC,
            rows=("x", "i", "c", "n", "e"),
            rtol=1e-9,
        )
    )
    out.append(
        Case(
            name="CauseSpecificNHPP",
            fitters=("surpyval.recurrent.CauseSpecificNHPP",),
            model_class="surpyval.recurrent.CauseSpecificNHPP",
            interface=COUNTING_CAUSES,
            data=functools.partial(recurrent_data, False, True),
            fit=_fit(rc.CauseSpecificNHPP),
            functions=(),
            event_functions=("cif", "iif"),
            events=("a", "b"),
            x=X_REC,
            rows=("x", "i", "c", "n", "e"),
        )
    )
    # A count above one is refused on an observed recurrent event (the
    # xicn convention: several events at one instant are not a count).
    no_counts = {
        "counts": "the xicn format refuses a count above 1 on an event row"
    }
    return [replace(c, exclude={**no_counts, **c.exclude}) for c in out]


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def _degradation():
    def da(path):
        return Case(
            name=f"DegradationAnalysis[{path}]",
            fitters=(
                ("surpyval.degradation.DegradationAnalysis",)
                if path == "linear"
                else ()
            ),
            model_class="surpyval.degradation.DegradationModel",
            interface=UNIVARIATE,
            data=path_data,
            fit=_fit(dg.DegradationAnalysis, threshold=150.0, path=path),
            functions=UNI_FUNCTIONS + ("qf",),
            x=X_PATH,
            rows=("x", "y", "i"),
            paths={
                "fit_from_df": lambda d: dg.DegradationAnalysis.fit_from_df(
                    pd.DataFrame(d), threshold=150.0, path=path
                )
            },
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={"counts": "degradation readings carry no counts"},
        )

    out = [da("linear"), da("power")]
    for name in ("WienerProcess", "GammaProcess"):
        fitter = getattr(dg, name)
        out.append(
            Case(
                name=name,
                fitters=(f"surpyval.degradation.{name}",),
                model_class=f"surpyval.degradation.{name}Model",
                interface=UNIVARIATE,
                data=process_data,
                fit=_fit(fitter, threshold=100.0),
                functions=UNI_FUNCTIONS + ("qf",),
                x=X_PROC,
                rows=("x", "y", "i"),
                paths={
                    "fit_from_df": lambda d, f=fitter: f.fit_from_df(
                        pd.DataFrame(d), threshold=100.0
                    )
                },
                draw=lambda m, s: m.random(15, random_state=s),
                explicit_seed=True,
                exclude={"counts": "degradation readings carry no counts"},
            )
        )
    out.append(
        Case(
            name="DestructiveDegradation",
            fitters=("surpyval.degradation.DestructiveDegradation",),
            model_class="surpyval.degradation.DestructiveDegradationModel",
            interface=UNIVARIATE,
            data=destructive_data,
            fit=_fit(dg.DestructiveDegradation, threshold=20.0),
            functions=("sf", "ff", "df", "Hf"),
            x=X_DESTR,
            rows=("x", "y"),
            exclude={
                "counts": "destructive readings carry no counts",
                "df_hf_sf": "DestructiveDegradationModel has no hf",
                "qf_ff": "DestructiveDegradationModel has no qf",
            },
        )
    )
    out.append(
        Case(
            name="InducedFailureDistribution",
            fitters=(),
            model_class="surpyval.degradation.InducedFailureDistribution",
            interface=UNIVARIATE,
            data=path_data,
            fit=lambda d: dg.DegradationAnalysis.fit(
                **d, threshold=150.0
            ).induced_life(n_samples=400, random_state=0),
            functions=("sf", "ff"),
            x=X_PATH,
            rows=("x", "y", "i"),
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
            exclude={
                "counts": "degradation readings carry no counts",
                "df_hf_sf": "a Monte Carlo distribution: no hf or df",
                "qf_ff": "an empirical distribution of simulated lives",
            },
        )
    )
    return out


# ---------------------------------------------------------------------------
# Copulas
# ---------------------------------------------------------------------------
def _copulas():
    out = []
    for name in ("Independence", "Clayton", "Gumbel", "Frank", "Gaussian"):
        fitter = getattr(mv, name)

        def from_params(d, f=fitter):
            m = f.fit(**d, margins=[sp.Weibull, sp.Weibull])
            return f.from_params(m.params, margins=m.margins)

        out.append(
            Case(
                name=f"{name}Copula",
                fitters=(f"surpyval.multivariate.{name}",),
                model_class="surpyval.multivariate.CopulaModel",
                interface=BIVARIATE,
                data=copula_data,
                fit=_fit(fitter, margins=[sp.Weibull, sp.Weibull]),
                functions=("sf", "cdf", "pdf"),
                x=X_COP,
                rows=("x", "n"),
                paths={"from_params": from_params},
                draw=lambda m, s: m.random(15, random_state=s),
                explicit_seed=True,
            )
        )
    return out


CASES: list[Case] = (
    _univariate()
    + _regression_family()
    + _accelerated_life_family()
    + _frailty_family()
    + _semi_parametric()
    + _trees()
    + _competing_risks()
    + _recurrent()
    + _degradation()
    + _copulas()
)

# Fits whose optimum moves by more than 1e-4 (relative) when the data are
# rescaled or reordered: the optimiser stops at its own tolerance, and a
# flat likelihood turns that into a larger change in the predictions.
_LOOSE: tuple[str, ...] = ("LogNormalAH", "WeibullAL[InverseExponential]")
_LOOSE += ("CauseSpecificNHPP", "GammaProcess")
# LogNormalAFT agrees to 1e-4 with the numpy/scipy of the development
# environment but moved by 2.3e-4 in sf under the newer ones CI installs
# (numpy 2.5, scipy 1.18): the same optimiser-tolerance effect.
_LOOSE += ("LogNormalAFT",)
CASES = [replace(c, rtol=1e-3) if c.name in _LOOSE else c for c in CASES]


# ---------------------------------------------------------------------------
# Options (test_options.py): the uncertainty methods of each case, the
# values its interp= takes and the estimation options of its fit
# ---------------------------------------------------------------------------
# The significance levels every bound is swept over.
ALPHAS = (0.01, 0.05, 0.2)


@dataclass(frozen=True)
class Bound:
    """One uncertainty method of a case, swept by ``test_options.py``.

    ``kind`` is what it bounds:

    - ``"function"``: a function of time at the case's query, named by
      ``on=`` when the method takes it (every value in ``on`` is swept)
      and by ``point`` otherwise;
    - ``"param"``: ``param_cb``, each fitted parameter in turn;
    - ``"coef"``: every coefficient at once, with no ``bound=``;
    - ``"rul"``: the interval of ``predict_rul`` on the remaining life,
      called with each argument tuple in ``query``;
    - ``"summary"``: one number's interval (``mean_cb``, ``rmst``).

    ``bound=`` (when ``sides``) and the levels in :data:`ALPHAS` are
    swept; ``kwargs`` fixes the rest -- the variant (``method=``,
    ``bound_type=``, ``interp=``) or a bootstrap's size and seed.
    """

    method: str
    kind: str = "function"
    on: tuple[str, ...] = ()
    point: str = "sf"
    kwargs: dict = field(default_factory=dict)
    # Takes bound= ("two-sided", "lower", "upper").
    sides: bool = True
    # The level argument; "confidence" takes 1 - alpha.
    level: str = "alpha_ci"
    # A transformed Wald (delta-method) bound, which closes onto the
    # estimate as alpha_ci -> 1; a percentile bootstrap or a
    # likelihood-ratio search is not one.
    wald: bool = True
    # Documented to stay inside the function's range (a plain normal
    # interval is not).
    in_range: bool = True
    # Documented to give NaN in places: outside the range of the data, or
    # where a likelihood-ratio search fails (with a warning).
    nan_ok: bool = False
    # Called once per cause of the case.
    per_cause: bool = False
    # "function": times replacing the case's query (for a slow search);
    # "rul" / "summary": the argument tuples of the calls.
    query: tuple = ()
    # Relative tolerance of the equalities (a search's own tolerance).
    rtol: float = 1e-8
    slow: bool = False
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or self.method


_ON_ALL = ("sf", "ff", "Hf", "hf", "df")
_ON_SURVIVAL = ("sf", "ff", "Hf")

# Parametric fits with no covariance to bound with: cb and param_cb
# raise ValueError, as documented ("Only MLE has confidence bounds"; a
# closed-form estimate or a model built from its parameters carries none).
_NO_COVARIANCE = (
    "Binomial",
    "Bernoulli",
    "FixedEventProbability",
    "ExactEventTime",
    "Hypoexponential",
)
# The likelihood-ratio search runs pointwise, so it is swept at three
# times, and only in the full suite.
_LR_X = {"Weibull": np.array([4.0, 8.0, 13.0])}


def _parametric_bounds(case):
    on = tuple(f for f in _ON_ALL if f in case.functions)
    out = [
        Bound("cb", on=on, kwargs={"method": "wald"}, label="cb[wald]"),
        Bound(
            "param_cb",
            kind="param",
            kwargs={"method": "wald"},
            label="param_cb[wald]",
        ),
    ]
    # The likelihood-ratio search is swept on Weibull only: it takes
    # minutes a distribution (ExpoWeibull's param_cb sweep took 420 s).
    # (Documented: it is not available for offset, limited-failure or
    # zero-inflated models.)
    if case.name not in _LR_X:
        return tuple(out)
    x = _LR_X[case.name]
    lr = dict(wald=False, nan_ok=True, rtol=1e-3, slow=True)
    out.append(
        Bound(
            "cb",
            on=on,
            kwargs={"method": "lr"},
            query=tuple(x),
            label="cb[lr]",
            **lr,
        )
    )
    out.append(
        Bound(
            "param_cb",
            kind="param",
            kwargs={"method": "lr"},
            label="param_cb[lr]",
            **lr,
        )
    )
    return tuple(out)


def _nonparametric_bounds(case):
    out = []
    for bound_type in ("exp", "normal"):
        for interp in ("step", "linear", "cubic"):
            out.append(
                Bound(
                    "cb",
                    on=_ON_SURVIVAL,
                    kwargs={"bound_type": bound_type, "interp": interp},
                    in_range=bound_type == "exp",
                    nan_ok=True,
                    label=f"cb[{bound_type},{interp}]",
                )
            )
        for method in ("hall-wellner", "nair"):
            out.append(
                Bound(
                    "band",
                    kwargs={"method": method, "bound_type": bound_type},
                    sides=False,
                    # Not swept to alpha_ci -> 1: the critical-value search
                    # grows without bound there (13 s at alpha_ci = 0.9; at
                    # 1 - 1e-6 it asks for a 158 TiB grid): #420.
                    wald=False,
                    in_range=bound_type == "exp",
                    nan_ok=True,
                    label=f"band[{method},{bound_type}]",
                )
            )
        out.append(
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"bound_type": bound_type},
                sides=False,
                # Documented: the upper end is NaN where the interval is
                # open to the right.
                nan_ok=True,
                label=f"quantile_cb[{bound_type}]",
            )
        )
    out.append(
        Bound(
            "bootstrap_cb",
            kwargs={"B": 40, "random_state": 1},
            wald=False,
            nan_ok=True,
            # Each resample reruns the Turnbull EM.
            slow=case.name == "Turnbull",
        )
    )
    out.append(Bound("mean_cb", kind="summary", sides=False, query=((),)))
    out.append(
        Bound("rmst", kind="summary", sides=False, query=((6.0,), (12.0,)))
    )
    return tuple(out)


def _mcf_bounds(per_cause=False):
    return tuple(
        Bound(
            "mcf_cb",
            point="mcf",
            kwargs={"bound_type": bound_type, "interp": interp},
            level="confidence",
            in_range=bound_type == "exp",
            nan_ok=True,
            per_cause=per_cause,
            label=f"mcf_cb[{bound_type},{interp}]",
        )
        for bound_type in ("exp", "normal")
        for interp in ("step", "linear")
    )


_PARAM_CB = Bound("param_cb", kind="param")
_BOOT = {"n_boot": 20, "seed": 1}


def _bounds(case):
    """The uncertainty methods of ``case``'s model (see :class:`Bound`)."""
    cls = case.model_class.rsplit(".", 1)[-1]
    if cls == "Parametric":
        if case.name in _NO_COVARIANCE:
            return ()
        return _parametric_bounds(case)
    if cls == "NonParametric":
        return _nonparametric_bounds(case)
    if cls == "RoystonParmarModel":
        return (Bound("cb", on=_ON_SURVIVAL),)
    if cls == "ParametricRegressionModel":
        return (Bound("cb", on=_ON_ALL), _PARAM_CB)
    if cls == "FrailtyModel":
        return (_PARAM_CB,)
    if cls == "BuckleyJamesModel":
        return (
            Bound(
                "bootstrap_ci",
                kind="coef",
                kwargs=_BOOT,
                sides=False,
                wald=False,
            ),
        )
    if cls in ("ParametricRecurrenceModel", "ProportionalIntensityModel"):
        return (Bound("cif_cb", point="cif"), _PARAM_CB)
    if cls == "RenewalModel":
        return (_PARAM_CB,)
    if cls == "NonParametricCounting":
        return _mcf_bounds()
    if cls == "CauseSpecificMCF":
        return _mcf_bounds(per_cause=True)
    if cls == "DegradationModel":
        # A new unit's first three readings, still below the threshold.
        d = path_data()
        unit = (d["x"][:3], d["y"][:3])
        return (
            Bound(
                "cb",
                on=_ON_SURVIVAL,
                kwargs={"method": "analytic"},
                label="cb[analytic]",
            ),
            Bound(
                "cb",
                on=_ON_SURVIVAL,
                kwargs={"method": "bootstrap", "n_boot": 10, "seed": 1},
                wald=False,
                slow=True,
                label="cb[bootstrap]",
            ),
            Bound(
                "predict_rul",
                kind="rul",
                kwargs={"n_samples": 4000, "random_state": 1},
                sides=False,
                # Documented: a remaining life is negative once the unit
                # has most likely crossed the threshold.
                in_range=False,
                query=(unit,),
                rtol=1e-2,
            ),
        )
    if cls == "DestructiveDegradationModel":
        # Every call refits the model n_boot times.
        return (
            Bound(
                "cb",
                on=("sf", "ff"),
                kwargs={"n_boot": 10, "seed": 1},
                wald=False,
                slow=True,
            ),
        )
    if cls in ("WienerProcessModel", "GammaProcessModel"):
        return (
            Bound(
                "predict_rul",
                kind="rul",
                sides=False,
                query=((0.0,), (40.0,), (80.0,)),
                rtol=1e-6,
            ),
        )
    return ()


# interp= values: the documented ones, and the other scipy interp1d
# kinds the non-parametric functions are documented to accept.
_NP_INTERP: tuple[str, ...] = ("step", "linear", "cubic")
_NP_INTERP += ("nearest", "zero", "slinear", "quadratic", "previous", "next")
_INTERP = {
    "KaplanMeier": _NP_INTERP,
    "NelsonAalen": _NP_INTERP,
    "FlemingHarrington": _NP_INTERP,
    "Turnbull": _NP_INTERP,
    "NonParametricCounting": ("step", "linear"),
    "CauseSpecificMCF": ("step", "linear"),
    # Its functions take interp= (undocumented; "step" is the default).
    "CompetingRisksProportionalHazards[Cox]": ("step", "linear"),
}

# Estimation options. ``estimators`` are swept on the case's fixture;
# the agreement sweep adds the values in ``_LARGE_ONLY`` (the method of
# moments needs uncensored data, so it is refused on the fixtures) and
# refits the large sample of ``large`` (a function of the fitted model).
N_LARGE = 1000
_U_LARGE = (np.arange(1, N_LARGE + 1) - 0.5) / N_LARGE


def _quantile_sample(model):
    """A deterministic 'sample' of the model: its quantiles."""
    return {"x": np.asarray(model.qf(_U_LARGE), float)}


def _parametric_estimators(case):
    fitter = getattr(sp, case.name)
    how = ["MLE", "MPS", "MSE", "MPP"]
    if fitter.discrete:
        how.remove("MPS")  # documented: MPS needs a continuous CDF
    if not fitter.supports_mpp:
        how.remove("MPP")  # documented: not fitted by probability plotting
    return {"how": tuple(how)}


def _censored_sample(model):
    # Weibull(10, 2) quantiles, every fifth one right censored.
    x = quantiles(N_LARGE)
    return {"x": x, "c": (np.arange(N_LARGE) % 5 == 4).astype(int)}


def _cox_sample(model):
    size = 400
    z0 = np.tile([0.0, 1.0], size // 2)
    z1 = np.round(np.linspace(-1.0, 1.0, size), 3)
    life = quantiles(size, 1.0, 2.0)[scramble(size)]
    # Rounded to one decimal, so there are ties for the methods to handle.
    x = np.round(10.0 * np.exp(-0.5 * z0 + 0.3 * z1) * life, 1)
    c = (np.arange(size) % 7 == 0).astype(int)
    return {"x": x, "Z": np.column_stack([z0, z1]), "c": c}


def _cr_sample(model, with_Z=True):
    size = 480
    x = np.round(quantiles(size, 10.0, 1.5), 1)
    e = np.array(["a", "b", "a", None] * (size // 4), dtype=object)
    d = {"x": x, "e": e[scramble(size)]}
    if with_Z:
        Z = np.tile([0.0, 1.0, 1.0], size // 3)[:, None]
        d["Z"] = Z[scramble(size)]
    return d


def _recurrent_sample(model):
    data = model.time_terminated_simulation_data(60.0, items=40, seed=1)
    return {"x": data.x, "i": data.i, "c": data.c, "n": data.n}


def _copula_sample(model):
    return {"x": model.random(800, random_state=1)}


_PLAIN_CONTINUOUS: tuple[str, ...] = (
    "Weibull",
    "Exponential",
    "Gamma",
    "LogNormal",
)
_PLAIN_CONTINUOUS += ("LogLogistic", "ExpoWeibull", "Rayleigh", "Normal")
_PLAIN_CONTINUOUS += ("Gumbel", "GumbelLEV", "Logistic", "Uniform")
_PLAIN_DISCRETE: tuple[str, ...] = ("Poisson", "Geometric", "NegativeBinomial")
_PLAIN_DISCRETE += ("DiscreteWeibull",)
# The agreement sweeps run on a pull request for these; the rest are slow.
_FAST_ESTIMATORS: tuple[str, ...] = (
    "Weibull",
    "LogNormal",
    "Poisson",
    "Turnbull",
    "CoxPH",
)
_FAST_ESTIMATORS += ("GumbelCopula", "CrowAMSAA")


def _estimators(case):
    """(estimators, large-only values, large sample) of ``case``."""
    name = case.name
    if name in _PLAIN_CONTINUOUS + _PLAIN_DISCRETE:
        return (
            _parametric_estimators(case),
            {"how": ("MOM",)},
            _quantile_sample,
        )
    if name == "Turnbull":
        est = ("Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington")
        return {"turnbull_estimator": est}, {}, _censored_sample
    if name == "CoxPH":
        methods = ("breslow", "efron", "exact", "kalbfleisch-prentice")
        return {"method": methods}, {}, _cox_sample
    if name == "CompetingRisksProportionalHazards[Cox]":
        return {"tie_method": ("efron", "breslow")}, {}, _cr_sample
    if name == "CompetingRisks[Nelson-Aalen]":
        methods = ("Nelson-Aalen", "Kaplan-Meier")
        return (
            {"method": methods},
            {},
            functools.partial(_cr_sample, with_Z=False),
        )
    if name == "ParametricCompetingRisks":
        return {"how": ("MLE", "MPS", "MSE", "MPP")}, {}, None
    if name in ("CrowAMSAA", "Duane", "CoxLewis"):
        return {"how": ("MLE", "MSE")}, {}, _recurrent_sample
    if name == "CauseSpecificNHPP":
        return {"how": ("MLE", "MSE")}, {}, None
    if case.model_class == "surpyval.multivariate.CopulaModel":
        return {"how": ("IFM", "MLE")}, {}, _copula_sample
    if name == "DegradationAnalysis[linear]":
        est = {
            "how": ("MLE", "MPS", "MSE", "MPP"),
            "population_method": ("moments", "reml"),
        }
        return est, {}, None
    return {}, {}, None


# The bound sweeps run on a pull request for one or two cases of each
# family (each regression bound recomputes a numerical Hessian); the rest
# are slow.
_FAST_BOUNDS: tuple[str, ...] = (
    "Weibull",
    "Poisson",
    "Weibull[lfp]",
    "KaplanMeier",
)
_FAST_BOUNDS += ("Turnbull", "RoystonParmar", "WeibullPH", "WeibullFrailty")
_FAST_BOUNDS += ("HPP", "CrowAMSAA", "ProportionalIntensityHPP")
_FAST_BOUNDS += ("GeneralizedRenewal", "NonParametricCounting")
_FAST_BOUNDS += ("CauseSpecificMCF", "DegradationAnalysis[linear]")
_FAST_BOUNDS += ("WienerProcess",)


def _with_options(case):
    estimators, large_only, large = _estimators(case)
    bounds = _bounds(case)
    if case.name not in _FAST_BOUNDS:
        bounds = tuple(replace(b, slow=True) for b in bounds)
    slow = case.slow
    if large is not None and case.name not in _FAST_ESTIMATORS:
        slow = slow | {"estimators_agree"}
    exclude = case.exclude
    if case.name in _NO_COVARIANCE:
        exclude = {
            **exclude,
            "cb_declared": "no covariance: cb and param_cb raise "
            "ValueError, as documented",
        }
    return replace(
        case,
        exclude=exclude,
        bounds=bounds,
        interp=_INTERP.get(case.name, ()),
        estimators=estimators,
        estimators_large=large_only,
        large=large,
        slow=frozenset(slow),
    )


CASES = [_with_options(c) for c in CASES]


# ---------------------------------------------------------------------------
# Convergence (test_convergence.py): how each case's fit is starved, or why
# it cannot be
# ---------------------------------------------------------------------------
# A starved start is the fitted value times FAR (a millionth of the way
# into a bounded range): far enough that a search stopping where the
# gradient first looks flat stops short of the maximum.
FAR = 1e6


def _far(value, bound):
    lo, hi = bound
    return value * FAR if hi is None else lo + (hi - lo) / FAR


def _parametric_start(model):
    """The fitted parameters with the first one unbounded above (else the
    first) moved :data:`FAR` away, in ``init``'s order."""
    bounds = model.dist.bounds
    k = next((i for i, b in enumerate(bounds) if b[1] is None), 0)
    params = np.array(model.params, dtype=float)
    params[k] = _far(params[k], bounds[k])
    start = ([model.gamma] if model.offset else []) + list(params)
    start += [model.p] if model.lfp else []
    return start + ([model.f0] if model.zi else [])


def _scaled_start(params, k=0):
    start = np.array(params, dtype=float)
    start[k] *= FAR
    return start


def _far_start(case, start):
    """Refit ``case`` from ``init=start(model)``, ``model`` its fit to the
    fixture."""
    return lambda d: case.fit({**d, "init": start(_fitted(case.name))})


def _no_event_level(d):
    """The first covariate is 1 on exactly the censored rows: a group with
    no events, whose coefficient the likelihood drives to infinity."""
    out = dict(d)
    Z = np.array(d["Z"], dtype=float)
    if "e" in d:
        Z[:, 0] = [e is None for e in d["e"]]
    else:
        Z[:, 0] = np.asarray(d["c"]) == 1
    out["Z"] = Z
    return out


def _comonotone(d):
    """The second coordinate half the first: dependence at its limit."""
    x = np.array(d["x"], dtype=float)
    x[:, 1] = x[:, 0] / 2
    return {**d, "x": x}


def _noise_free(y):
    """Degradation readings exactly on a path: no noise to estimate."""
    return lambda d: {**d, "y": y(np.asarray(d["x"], dtype=float))}


def _starve(case):
    """How ``case``'s fit is starved (see ``Case.starve``), or ``None``."""
    name, fit = case.name, case.fit
    cls = case.model_class.rsplit(".", 1)[-1]
    if name in ("Turnbull", "BuckleyJames"):
        return lambda d: fit({**d, "max_iter": 1})
    if name.startswith("WeibullAL"):
        # The life model's first parameter (the first is a fixed
        # placeholder, the second the Weibull shape).
        return _far_start(case, lambda m: _scaled_start(m.params, 2))
    if cls in ("ParametricRegressionModel", "FrailtyModel"):
        return lambda d: fit(_no_event_level(d))
    if cls in ("SemiParametricRegressionModel", "FineGrayModel"):
        return lambda d: fit(_no_event_level(d))
    if cls == "CompetingRisksProportionalHazards":
        return lambda d: fit(_no_event_level(d))
    if cls == "Parametric":
        return _far_start(case, _parametric_start)
    if cls == "MixtureModel":
        # One component's data a point mass: its shape runs to infinity.
        return lambda d: fit({**d, "x": np.r_[np.full(10, 3.0), d["x"][10:]]})
    if cls == "ParametricCompetingRisks":
        # Every cause-b failure at one time: no maximum for its Weibull.
        return lambda d: fit({**d, "x": np.where(d["e"] == "b", 5.0, d["x"])})
    if cls == "ParametricRecurrenceModel":
        return _far_start(case, lambda m: _scaled_start(m.params))
    if cls == "ProportionalIntensityModel":
        return _far_start(
            case, lambda m: _scaled_start(np.r_[m.params, m.coeffs])
        )
    if cls == "CauseSpecificNHPP":
        return _far_start(case, lambda m: _scaled_start(m.models["a"].params))
    if cls == "RenewalModel":
        # [restoration, *distribution parameters]: the scale moved.
        return _far_start(
            case,
            lambda m: _scaled_start(np.r_[m.restoration, m.model.params], 1),
        )
    if cls == "CopulaModel" and name != "IndependenceCopula":
        return lambda d: fit(_comonotone(d))
    if cls == "DegradationModel":
        return lambda d: fit(_noise_free(lambda x: 10.0 + 0.35 * x)(d))
    if cls in ("WienerProcessModel", "GammaProcessModel"):
        return lambda d: fit(_noise_free(lambda x: 0.5 * x)(d))
    if cls == "DestructiveDegradationModel":
        return lambda d: fit(_noise_free(lambda x: np.exp(4.0 - 0.02 * x))(d))
    return None


# The fits with nothing to starve.
_CLOSED_FORM = "a closed-form estimate: no iteration to fail"
_EXACT = "an exact (product-limit or Nelson-Aalen type) estimator"
_NO_STARVE: dict[str, str] = {
    "Exponential": _CLOSED_FORM + " (failures / total time; init is unused)",
    "Uniform": _CLOSED_FORM + " (the sample extremes; init is unused)",
    "Binomial": _CLOSED_FORM,
    "Bernoulli": _CLOSED_FORM,
    "FixedEventProbability": _CLOSED_FORM,
    "ExactEventTime": _CLOSED_FORM,
    "AdditiveHazards": _CLOSED_FORM + " (Lin-Ying: a linear system)",
    "KaplanMeier": _EXACT,
    "NelsonAalen": _EXACT,
    "FlemingHarrington": _EXACT,
    "CompetingRisks[Nelson-Aalen]": _EXACT,
    "CompetingRisks[Kaplan-Meier]": _EXACT,
    "NonParametricCounting": _EXACT,
    "CauseSpecificMCF": _EXACT,
    "IndependenceCopula": "no dependence parameter: the margins are "
    "univariate fits, starved in their own cases",
    "InducedFailureDistribution": "a Monte Carlo of the DegradationAnalysis "
    "fit, which is starved in its own case",
    "RoystonParmar": "no public iteration limit or starting point, and no "
    "data found whose fit fails (its Nelder-Mead result is not checked "
    "for convergence, only for a finite likelihood)",
}
for _kind in ("weibull", "exponential", "non-parametric"):
    _NO_STARVE[f"SurvivalTree[{_kind}]"] = (
        "no public iteration limit or starting point: the splits are "
        "bounded searches, and a leaf is fitted when first used"
    )
_NO_STARVE["RandomSurvivalForest"] = _NO_STARVE["SurvivalTree[weibull]"]


# Fits whose initial guess is already the maximum, so returning it is right.
_START_IS_MAXIMUM = (
    "the initial guess is the maximum: the Normal MLE of log x (and the "
    "share of zeros for f0)"
)


def _with_convergence(case):
    if "convergence" in case.exclude:  # not fitted to data
        return case
    if case.name in _NO_STARVE:
        reason = _NO_STARVE[case.name]
        return replace(case, exclude={**case.exclude, "convergence": reason})
    exclude = case.exclude
    if case.name in ("LogNormal", "LogNormal[zi]"):
        exclude = {**exclude, "convergence[initial guess]": _START_IS_MAXIMUM}
    return replace(case, starve=_starve(case), exclude=exclude)


CASES = [_with_convergence(c) for c in CASES]

# ---------------------------------------------------------------------------
# Known failures: case -> property -> what goes wrong. Each becomes a
# strict xfail, so the suite stays green and fails (XPASS) the day the
# failure is fixed -- then delete the entry. Numbers are on the case's
# fixture; see the report for #379 for minimal reproductions.
# ---------------------------------------------------------------------------
_NP_SHAPES = {
    "array2d": "a (2, 2) query gives sf/ff/Hf of shape (2, 2, 2, 2) and "
    "hf/df raise ValueError (np.concatenate dimensions)",
    "empty": "hf and df of an empty query raise IndexError "
    "(nonparametric.py indexes element 0)",
}
_CONSTANT_HAZARD = (
    "hf ignores x, so hf(nan) is the constant rate instead of NaN"
)
KNOWN_FAILURES: dict[str, dict[str, str]] = {
    # -- shapes ---------------------------------------------------------
    "KaplanMeier": _NP_SHAPES,
    "NelsonAalen": _NP_SHAPES,
    "FlemingHarrington": _NP_SHAPES,
    "Turnbull": _NP_SHAPES,
    "BetaGeometric": {
        "array2d": "qf of a (2, 2) query raises ValueError (truth value "
        "of an array, beta_geometric.py)",
        "empty": "qf of an empty query raises IndexError",
        "missing_query": "qf(nan) is 1, not NaN",
    },
    "RoystonParmar": {
        "array2d": "every function of a (2, 2) query raises ValueError "
        "(matmul with the spline basis, royston_parmar.py)",
        "missing_query": "qf(nan) raises ValueError from the root finder "
        "('function value ... is NaN')",
    },
    "WienerProcess": {
        "array2d": "qf of a (2, 2) query raises ValueError (truth value "
        "of an array, process_models.py)",
    },
    "GammaProcess": {
        "array2d": "qf of a (2, 2) query raises ValueError (truth value "
        "of an array, process_models.py)",
    },
    "InducedFailureDistribution": {
        "array2d": "sf/ff of a (2, 2) query raise ValueError (broadcast "
        "against the samples, degradation_analysis.py)",
    },
    "GaussianCopula": {
        "empty": "sf/cdf of an empty (0, 2) query raise ValueError "
        "(scipy's multivariate normal cdf is handed zero rows)",
        "missing_query": "cdf and sf of a point with a NaN coordinate are "
        "numbers (cdf 0), not NaN",
    },
    # -- identities -----------------------------------------------------
    "Discretize(Weibull)": {
        "qf_ff": "qf(ff(k)) is k + 1 at some atoms (k = 5: 6): qf is "
        "ceil(continuous qf), which rounds k + 1e-15 up",
    },
    "CompetingRisksProportionalHazards[Cox]": {
        "cif_sum": "sf is exp(-H) but the CIFs are weighted by the "
        "product-limit survival: at t = 30 the CIFs sum to 1.0 while "
        "1 - sf is 0.975",
    },
    # -- missing values -------------------------------------------------
    "Exponential": {"missing_query": _CONSTANT_HAZARD},
    "Exponential[offset]": {"missing_query": _CONSTANT_HAZARD},
    "ExponentialPH": {"missing_query": _CONSTANT_HAZARD},
    "ExponentialAFT": {"missing_query": _CONSTANT_HAZARD},
    "ExponentialAH": {"missing_query": _CONSTANT_HAZARD},
    "SurvivalTree[exponential]": {"missing_query": _CONSTANT_HAZARD},
    "Geometric": {"missing_query": _CONSTANT_HAZARD},
    "HPP": {"missing_query": "iif(nan) is the constant rate, not NaN"},
    "ProportionalIntensityHPP": {
        "missing_query": "iif(nan) is the constant rate, not NaN"
    },
    "Uniform": {
        "missing_query": "sf, ff, df, hf and Hf of nan are 1, 0, 0, 0 and "
        "0, not NaN",
    },
    # -- units (and missing values) --------------------------------------
    "Beta4": {
        "units": "the fit depends on the unit: alpha, beta = 1.00, 1.19 on "
        "the data but 0.18, 0.18 on the data x 7.3 (the end points sit on "
        "the sample extremes)",
        "missing_query": "df(nan) is 0, not NaN",
    },
    "Binomial": {"missing_query": "hf(nan) is 0, not NaN"},
    "Bernoulli": {
        "missing_query": "sf, ff and Hf of nan raise ValueError ('defined "
        "at x = 0 and x = 1 only') instead of giving NaN",
    },
    "FixedEventProbability": {
        "missing_query": "sf, ff and Hf of nan are 0.3, 0.7 and 1.20, not "
        "NaN",
    },
    "ExactEventTime": {
        "missing_query": "sf, ff and Hf of nan are 0, 0 and 0 (so sf + ff "
        "is 0), and qf(nan) is T, not NaN",
    },
    "NeverOccurs": {
        "missing_query": "every function of nan is a number (sf 1, qf "
        "inf), not NaN",
    },
    "InstantlyOccurs": {
        "missing_query": "every function of nan is a number (sf 0, qf 0), "
        "not NaN",
    },
    "NonParametricCounting": {
        "missing_query": "mcf(nan) is the last value (4.33), not NaN",
    },
    "CoxLewis": {
        "seed_explicit": "count_terminated_simulation(3, items=2, seed=7) "
        "raises ValueError ('Event times x must be finite'): the fitted "
        "intensity falls (b = -0.021, cif(inf) = 6.04), so a sequence can "
        "stop short of its 4th event, and seed 7 draws one",
    },
    **{
        f"{base}Frailty": {
            "missing_fit[groups]": "a NaN group label is kept, silently, as "
            "a group of its own (n_groups 7, not 6; sf(5, [0, -0.8]) "
            "0.7825 where dropping the row gives 0.7418), and a None label "
            "raises TypeError",
        }
        for base in ("Weibull", "Exponential", "Gamma", "LogNormal")
    },
    "CauseSpecificMCF": {
        "missing_query": "mcf(nan, cause) is the last value (2.67), not "
        "NaN",
    },
}
for _name in (
    "GeneralizedRenewal",
    "GeneralizedRenewal[kijima ii]",
    "GeneralizedOneRenewal",
    "ARA",
    "ARI",
):
    KNOWN_FAILURES[_name] = {
        "empty": "mcf([]) raises ValueError (max of an empty array, "
        "simulation.py)",
        "missing_query": "mcf with a NaN time raises ValueError ('x' "
        "cannot be empty) instead of giving NaN there",
    }


# -- option sweeps (test_options.py) ------------------------------------
# Keyed "<property>[<bound name>]" (or "interp[<value>]",
# "estimators_agree[<option>]"); grouped as in the report for #379.
def _each(props, name, reason):
    return {f"{p}[{name}]": reason for p in props}


_RATE_AT_ZERO = (
    "hf/df bounds are NaN where the rate is 0 (the log-scale bound of "
    "0): Uniform hf(1) = 0 gives cb [nan, nan]"
)
_DISCRETE_HF = (
    "cb(on='hf') is centred on df(k)/sf(k), not the model's hf(k) = "
    "df(k)/sf(k-1)"
)
_LOGIT_CLIP = (
    "sf is clipped to 1e-15 on the logit scale, so the Hf bounds stop at "
    "-log(1e-15) = 34.54"
)
_NEGATIVE_VARIANCE = (
    "a parameter on the edge of its support has a non-positive variance "
    "(covariance diagonal"
)
_RP_SIDES = (
    "one-sided cb on ff and Hf returns the other side: ff(10) = 0.436, "
    "two-sided (alpha 0.1) [0.291, 0.615], bound='lower' (0.05) 0.615"
)
_MCF_BOTH = (
    "mcf_cb(bound='both') raises UnboundLocalError ('stat'), not ValueError"
)
_SCIPY_INTERP = (
    "interp='bogus' raises scipy's NotImplementedError, not ValueError"
)
_KM_CUBIC = (
    "with interp='cubic' the bound at x = 13 does not close onto sf as "
    "alpha_ci -> 1 (0.1873 vs 0.1948): the fill of the undefined last "
    "variance (lower 0, last finite upper) enters the PCHIP between the "
    "last knots"
)
_BOUND_PROPS: tuple[str, ...] = ("cb_contains", "cb_range", "cb_sides")
_BOUND_PROPS += ("cb_nested", "cb_centre")
_OPTION_FAILURES: dict[str, dict[str, str]] = {
    # A. rate bounds at a zero rate
    "Uniform": _each(
        ("cb_contains",),
        "cb[wald]",
        _RATE_AT_ZERO + "; and df bounds are NaN everywhere (df(5) = "
        "0.0688, cb [nan, nan])",
    ),
    "Weibull[offset]": _each(
        ("cb_contains",),
        "cb[wald]",
        _RATE_AT_ZERO + "; here hf(5.5) = 0 below gamma = 6.56",
    ),
    "Exponential[offset]": _each(
        ("cb_contains",), "cb[wald]", _RATE_AT_ZERO + "; here below gamma"
    ),
    # B. discrete hazard bounds (and A at k = 0)
    "Poisson": _each(
        ("cb_contains", "cb_centre"),
        "cb[wald]",
        _DISCRETE_HF + ": hf(10) = 0.728, cb [1.744, 4.096], df/sf(10) "
        "= 2.673",
    ),
    "Geometric": _each(
        ("cb_contains", "cb_centre"),
        "cb[wald]",
        _DISCRETE_HF + ": hf = p = 0.220, the bounds centre on p/(1-p) = "
        "0.283; and hf(0) = 0 has bounds [nan, nan]",
    ),
    "NegativeBinomial": _each(
        ("cb_contains", "cb_centre"),
        "cb[wald]",
        _DISCRETE_HF + ": hf(10) = 0.432, df/sf(10) = 0.760; and hf(0) "
        "= 0 has bounds [nan, nan]",
    ),
    "DiscreteWeibull": _each(
        ("cb_contains", "cb_centre"),
        "cb[wald]",
        _DISCRETE_HF + ": hf(10) = 0.474, cb [0.211, 3.842], df/sf(10) "
        "= 0.900; and hf(0) = 0 has bounds [nan, nan]",
    ),
    "Discretize(Weibull)": _each(
        ("cb_contains", "cb_centre"),
        "cb[wald]",
        _DISCRETE_HF + " (as DiscreteWeibull: hf(10) = 0.474, df/sf(10) "
        "= 0.900); and hf(0) = 0 has bounds [nan, nan]",
    ),
    # C. Royston-Parmar
    "RoystonParmar": {
        **_each(("cb_contains", "cb_sides", "cb_transform"), "cb", _RP_SIDES),
        "cb_api[cb]": "an unknown bound is not refused: cb(10, "
        "bound='both') returns 0.709, as 'upper'",
    },
    # D. recurrent MCF bounds
    **{
        name: {
            f"cb_api[mcf_cb[{t},{i}]]": _MCF_BOTH
            for t in ("exp", "normal")
            for i in ("step", "linear")
        }
        for name in ("NonParametricCounting", "CauseSpecificMCF")
    },
    # E. unknown interp
    **{
        name: {"interp_refused": _SCIPY_INTERP}
        for name in ("NelsonAalen", "FlemingHarrington", "Turnbull")
    },
    "CompetingRisksProportionalHazards[Cox]": {
        "interp_refused": "sf, ff, Hf, hf and df accept any interp "
        "(even 'bogus') and ignore it: interp='linear' is the step curve",
    },
    # F. Kaplan-Meier cubic interpolation
    "KaplanMeier": {
        "interp_refused": _SCIPY_INTERP,
        "interp[cubic]": "at the last time (sf = 0) the PCHIP sf is "
        "-2.3e-17, so Hf(16.954, interp='cubic') is NaN, not inf",
        "cb_centre[cb[exp,cubic]]": _KM_CUBIC,
        "cb_centre[cb[normal,cubic]]": _KM_CUBIC,
    },
    # G. logit clip of the regression survival bound
    "GumbelPH": _each(
        ("cb_contains", "cb_centre"),
        "cb",
        _LOGIT_CLIP + ": Hf(22, [1, -0.2]) = 110.6, bounds [34.54, 34.54]",
    ),
    "GumbelAFT": _each(
        ("cb_contains", "cb_centre"),
        "cb",
        _LOGIT_CLIP + ": Hf(22, [1, -0.2]) = 1376.8, bounds [34.54, " "34.54]",
    ),
    "NormalPH": _each(
        ("cb_centre",),
        "cb",
        _LOGIT_CLIP + ": Hf(22, [1, -0.2]) = 36.18, and the interval at "
        "alpha_ci -> 1 is 34.54",
    ),
    # H. boundary estimates with a non-positive variance
    "GeneralizedRenewal": _each(
        ("cb_contains",),
        "param_cb",
        _NEGATIVE_VARIANCE + " -0.031 for q = 2.7e-16 and -10.6 for "
        "alpha), and param_cb gives [nan, nan] for both, silently",
    ),
    "ARA": _each(
        ("cb_contains",),
        "param_cb",
        _NEGATIVE_VARIANCE + " -0.026 for rho = 1 - 3e-16 and -8.09 for "
        "alpha), and param_cb gives [nan, nan] for both, silently",
    ),
    "ARI": _each(
        _BOUND_PROPS,
        "param_cb",
        "rho = 1.0 exactly, the upper end of its (0, 1) support: "
        "param_cb('rho') raises ZeroDivisionError (the logit of 1); its "
        "variance is -0.0021",
    ),
    "Beta4": _each(
        ("cb_contains",),
        "param_cb[wald]",
        _NEGATIVE_VARIANCE + " -8.4e-5 for alpha = 1.00008, whose fit "
        "runs to the edge): param_cb('alpha') is [nan, nan], silently",
    ),
    # I. BetaGeometric's degenerate fit
    "BetaGeometric": {
        f"cb_contains[{name}]": "the fit runs to alpha, beta = 1.0e5, "
        "3.3e5 (the geometric limit) and every bound is NaN, silently"
        for name in ("cb[wald]", "param_cb[wald]")
    },
    # K. estimation options
    "CoxLewis": {
        "estimators_agree[how]": "how='MSE' misses on a simulated sample "
        "of the fitted model (40 items to t = 60): cif(55) is 113.4 by "
        "MSE, 4.40 by MLE, 4.14 true (params 0.98, -0.0099 vs -2.08, "
        "-0.0175)",
    },
    # L. on= aliases
    "DestructiveDegradation": {
        "cb_api[cb]": "cb(on='R') raises ValueError; the other cb "
        "methods accept 'R' and 'F' for 'sf' and 'ff'",
    },
}
# The issue tracking each case's option failures (by key where a case
# has failures of more than one kind); it leads each reason.
_OPTION_ISSUES: dict[str, str | dict[str, str]] = {
    "Uniform": "#413",
    "Weibull[offset]": "#413",
    "Exponential[offset]": "#413",
    "Poisson": "#414",
    "Geometric": "#414",
    "NegativeBinomial": "#414",
    "DiscreteWeibull": "#414",
    "Discretize(Weibull)": "#414",
    "RoystonParmar": "#415",
    "NonParametricCounting": "#416",
    "CauseSpecificMCF": "#416",
    "NelsonAalen": "#416",
    "FlemingHarrington": "#416",
    "Turnbull": "#416",
    "CompetingRisksProportionalHazards[Cox]": "#416",
    "DestructiveDegradation": "#416",
    "KaplanMeier": {
        "interp_refused": "#416",
        "interp[cubic]": "#417",
        "cb_centre[cb[exp,cubic]]": "#417",
        "cb_centre[cb[normal,cubic]]": "#417",
    },
    "GumbelPH": "#418",
    "GumbelAFT": "#418",
    "NormalPH": "#418",
    "GeneralizedRenewal": "#411",
    "ARA": "#411",
    "ARI": "#411",
    "Beta4": "#411",
    "BetaGeometric": "#392",
    "CoxLewis": "#419",
}
for _name, _failures in _OPTION_FAILURES.items():
    _issue = _OPTION_ISSUES[_name]
    _failures = {
        key: "{}: {}".format(
            _issue if isinstance(_issue, str) else _issue[key], reason
        )
        for key, reason in _failures.items()
    }
    KNOWN_FAILURES[_name] = {**KNOWN_FAILURES.get(_name, {}), **_failures}

# -- convergence (test_convergence.py) --------------------------------------
# Each starved fit returns silently -- no warning, no error -- a model that
# is not the maximum (a lower log-likelihood, "ll", than the fixture's
# fit) or, where the data have no maximum, a finite answer as if there
# were one. Grouped by the fault; the group's issue leads each reason.
_CONVERGENCE_ISSUES = {
    # univariate MLE: the ladder takes the first optimiser that reports
    # success, which BFGS does where the gradient first looks flat
    "start": "#427",
    # regression, Fine-Gray, copula, mixture and degradation fits of data
    # whose likelihood has no maximum (the #392 class, outside univariate)
    "no maximum": "#392",
    # accelerated life (parameter substitution): stops short, silently
    "al": "#428",
    # recurrent NHPP and renewal fits: the optimiser's result is unchecked
    "recurrent": "#429",
}
_FAR = "from init with its first parameter x1e6: "
_FAR_AL = "from init with the life model's first parameter x1e6: "
_FAR_SCALE = "from init with the scale x1e6: "
_NO_EVENTS = (
    "a covariate that is 1 on exactly the censored rows (a group with no "
    "events, so no finite coefficient; CoxPH warns 'Monotone partial "
    "likelihood' on such data) gets a coefficient of "
)
_CONVERGENCE_FAILURES: dict[str, tuple[str, str]] = {
    "Weibull": (
        "start",
        _FAR + "alpha, beta 1.03e7, 0.099 (ll -78.2), not 10.31, 2.32 "
        "(ll -37.9); sf(25) 0.757, not 0.0004",
    ),
    "Weibull[xcnt]": (
        "start",
        _FAR + "alpha, beta 0.034, 0.072 (ll -41.1), not 7.51, 1.35 "
        "(ll -24.3)",
    ),
    "Weibull[offset]": (
        "start",
        "from init with alpha x1e6: gamma, alpha, beta -5.37e6, 5.37e6, "
        "1.32e6 (ll -39.8), not 6.56, 8.53, 1.79 (ll -37.7)",
    ),
    "Weibull[lfp]": (
        "start",
        _FAR + "alpha, beta, p 9.82e6, 0.113, 1.0 (ll -80.3), not 9.82, "
        "2.31, 0.596 (ll -50.7)",
    ),
    "Weibull[zi]": (
        "start",
        _FAR + "alpha, beta 1.03e7, 0.099 (ll -84.3), not 10.31, 2.32 "
        "(ll -44.0)",
    ),
    "Gamma[lfp]": (
        "start",
        _FAR + "alpha, beta, p 3.66, 0.411, 0.644 (ll -50.97), not 4.16, "
        "0.478, 0.595 (ll -50.80); sf(25) 0.360, not 0.407",
    ),
    "LogNormal[offset]": (
        "start",
        "from init with mu x1e6: mu, sigma 2.77, 0.636 (ll -43.7), not "
        "2.73, 0.274 (ll -38.1)",
    ),
    "LogNormal[lfp]": (
        "start",
        _FAR + "mu, sigma 2.73, 2.66 (ll -62.8), not 2.04, 0.533 (ll -51.2)",
    ),
    "LogNormal[zi]": (
        "start",
        _FAR + "mu, sigma 2.33, 2.66 (ll -58.7), not 2.10, 0.550 (ll -44.6)",
    ),
    "LogLogistic": (
        "start",
        _FAR + "alpha, beta 8.42e6, 0.0 (ll -569.2), not 8.42, 3.15 "
        "(ll -38.6)",
    ),
    "ExpoWeibull": (
        "start",
        _FAR + "alpha, beta, mu 1.03e7, 1.16, 0.066 (ll -74.5), not 10.27, "
        "2.30, 1.01 (ll -37.9)",
    ),
    "Normal": (
        "start",
        _FAR + "mu, sigma 9.68, 9.62 (ll -44.0), not 9.08, 4.15 (ll -38.4)",
    ),
    "Gumbel": (
        "start",
        _FAR + "mu, sigma 1.12e7, 5.11e5 (ll -454.9), not 11.17, 4.06 "
        "(ll -39.8)",
    ),
    "Beta4": (
        "start",
        _FAR + "alpha stays at its start, 1.0e6 (ll -2.2e7), not 1.00 "
        "(ll 3.98); sf is 1 everywhere",
    ),
    "ConformanceGompertz": (
        "start",
        _FAR + "nu, b 1.39e5, 0.0 (ll -42.9), not 0.139, 0.194 (ll -38.5)",
    ),
    "NegativeBinomial": (
        "start",
        _FAR + "r, p 4.13e6, 1.0 (ll -30.4), not 4.13, 0.557 (ll -29.3)",
    ),
    "BetaGeometric": (
        "start",
        _FAR + "a stays at its start, 1.04e11 (ll -582.9), not 1.04e5 "
        "(ll -31.2)",
    ),
    "WeibullAL[InversePower]": (
        "al",
        _FAR_AL + "beta, a, n 1.22, 0.0115, 2.56 (ll -104.5), not 2.43, "
        "0.0298, 1.21 (ll -89.8)",
    ),
    "WeibullAL[Linear]": (
        "al",
        _FAR_AL + "beta, a, b 2.27, 40.19, -10.53 (ll -93.04), not 2.13, "
        "38.47, -9.94 (ll -92.91)",
    ),
    "WeibullAL[DualPower]": (
        "al",
        _FAR_AL + "n -0.138, not -0.130 (ll -89.6856, "
        "not -89.6850): sf off by up to 2.6% (relative) in the tail",
    ),
    "WeibullAL[PowerExponential]": (
        "al",
        _FAR_AL + "beta, c, a, n 2.46, 6.65, 1.98, -0.717 (ll -92.1), not "
        "2.50, 5.32, 1.93, -0.163 (ll -89.3)",
    ),
    "Duane": (
        "recurrent",
        _FAR + "alpha stays at its start, 7.76e5 (cif(5) inf, ll nan), not "
        "0.776 (ll -46.7): nhpp_fitter.py never checks res.success",
    ),
    "ProportionalIntensityNHPP": (
        "recurrent",
        _FAR + "the Duane alpha stays at its start, 7.76e5 (cif(5) inf, ll "
        "nan), not 0.776 (ll -46.7)",
    ),
    "CoxLewis": (
        "recurrent",
        _FAR + "alpha, beta -2.011, -0.0232 (ll -46.341), not -2.062, "
        "-0.0211 (ll -46.333); cif(25) 2.54, not 2.47",
    ),
    "GeneralizedOneRenewal": (
        "recurrent",
        _FAR_SCALE + "q, alpha, beta 1.08, 3.40, 2.13 (ll -37.9), not "
        "0.485, 5.72, 3.84 (ll -31.8); mcf(12) 1.93, not 1.23",
    ),
    "ARI": (
        "recurrent",
        _FAR_SCALE + "alpha stays at its start, 4.19e6 (ll -264.1), not "
        "4.19 (ll -37.4); mcf(55) 0, not 4.8",
    ),
    "FineGray": (
        "no maximum",
        _NO_EVENTS + "-12.87 (BFGS reports success); sf(30) 1.0",
    ),
    "CompetingRisksProportionalHazards[Fine-Gray]": (
        "no maximum",
        _NO_EVENTS + "-12.87 and -12.12 (causes a and b)",
    ),
    "LogNormalAH": (
        "no maximum",
        _NO_EVENTS + "-6.2e8, with mu, sigma 4.9e8, 3.0e8, and sf(2) is "
        "inf (the other AH baselines warn that the fit ended on the "
        "positivity boundary)",
    ),
    "MixtureModel": (
        "no maximum",
        "with 10 of the 20 rows at 3.0 (a point mass, so no maximum) the "
        "first component comes back as alpha, beta 3.0, 8955",
    ),
    "GammaProcess": (
        "no maximum",
        "noise-free readings (y = t / 2: every increment 5) have no "
        "maximum; the fit returns alpha, beta 1.0e6, 2.0e6 where "
        "WienerProcess raises ValueError ('the fitted diffusion sigma is 0')",
    ),
    "DestructiveDegradation": (
        "no maximum",
        "noise-free readings (y = exp(4 - 0.02 x) exactly) have no maximum; "
        "the fit returns sigma = 9.9e-16",
    ),
    "ClaytonCopula": (
        "no maximum",
        "comonotone data (x2 = x1 / 2: no finite theta) give theta 3.16e6",
    ),
    "GumbelCopula": (
        "no maximum",
        "comonotone data (x2 = x1 / 2: no finite theta) give theta 105.5, "
        "with a log-likelihood of inf",
    ),
    "FrankCopula": (
        "no maximum",
        "comonotone data (x2 = x1 / 2: no finite theta) give theta 1.24e7",
    ),
    "GaussianCopula": (
        "no maximum",
        "comonotone data (x2 = x1 / 2: rho -> 1) give rho 0.9999",
    ),
}
_REGRESSION_NO_EVENTS = {
    "WeibullPH": -16.31,
    "LogNormalPH": -19.38,
    "ExponentialPH": -18.55,
    "GammaPH": -18.47,
    "NormalPH": -20.41,
    "GumbelPH": -19.96,
    "LogisticPH": -18.28,
    "WeibullAFT": -16.77,
    "LogNormalAFT": -4.75,
    "ExponentialAFT": -31.58,
    "GammaAFT": -10.61,
    "NormalAFT": -31.27,
    "GumbelAFT": -31.57,
    "LogisticAFT": -29.60,
    "WeibullPO": 33.63,
    "LogNormalPO": 34.71,
    "ExponentialPO": 33.60,
    "GammaPO": 33.33,
    "NormalPO": 33.77,
    "GumbelPO": 32.01,
    "LogisticPO": 33.48,
    "WeibullFrailty": -9.22,
    "ExponentialFrailty": -33.62,
    "GammaFrailty": -32.60,
    "LogNormalFrailty": -31.46,
}
for _name, _coef in _REGRESSION_NO_EVENTS.items():
    _CONVERGENCE_FAILURES[_name] = ("no maximum", _NO_EVENTS + f"{_coef}")
for _name, (_group, _reason) in _CONVERGENCE_FAILURES.items():
    KNOWN_FAILURES[_name] = {
        **KNOWN_FAILURES.get(_name, {}),
        "convergence": f"{_CONVERGENCE_ISSUES[_group]}: {_reason}",
    }


# The issue that tracks each kind of known failure; its number leads the
# xfail reason, so a test report says where the fix is being worked on.
KNOWN_FAILURE_ISSUES: dict[str, str] = {
    "array2d": "#381",
    "empty": "#381",
    "missing_query": "#382",
    "qf_ff": "#383",
    "cif_sum": "#384",
    "units": "#385",
    "seed_explicit": "#386",
    "missing_fit[groups]": "#388",
}
# The option-sweep and outside-data failures name their issue in the
# reason itself (see _OPTION_ISSUES), as one key can fail for different
# reasons in different cases.


# Option conventions (test_options.py, CONVENTIONS) the package breaks:
# convention -> the inconsistency. Each is a strict xfail; the test's
# message lists every method concerned.
KNOWN_INCONSISTENCIES: dict[str, str] = {
    "alpha_ci": "NonParametricCounting.mcf_cb and the recurrent-event "
    "plots (NonParametricCounting, CauseSpecificMCF, "
    "ParametricRecurrenceModel, ProportionalIntensityModel) take "
    "confidence=0.95 where every other interval takes alpha_ci=0.05",
    "seed": "the random-number argument is random_state in random, "
    "band, bootstrap_cb, induced_life and DegradationModel.predict_rul, "
    "but seed in the recurrent simulations, BuckleyJames.bootstrap_ci, "
    "cramer_von_mises and the degradation cb",
    "resamples": "the bootstrap size is B in NonParametric.bootstrap_cb "
    "and n_boot everywhere else",
    "ties": "Cox's tie handling is method= in CoxPH.fit / fit_from_df / "
    "fit_tvc* but tie_method= in CoxPH.baseline and "
    "CompetingRisksProportionalHazards.fit / fit_from_df",
    "time": "the times are t, not x, in Parametric.cb, RoystonParmarModel "
    "(sf, ff, df, hf, Hf, cb), DestructiveDegradationModel (sf, ff, df, "
    "Hf, cb) and the Wiener / Gamma process models (sf, ff, df, hf, Hf)",
    "quantile": "qf takes u (NeverOccurs, InstantlyOccurs) or q "
    "(RoystonParmarModel), not p",
    "cause": "the per-cause argument is cause= in CauseSpecificMCF and "
    "CauseSpecificNHPP, event= in the competing-risks models",
    "Z": "DegradationModel.cb takes Z as its last keyword (x, on, "
    "alpha_ci, ..., Z), and WienerProcessModel / GammaProcessModel.random "
    "take (size, random_state, Z) where DegradationModel.random takes "
    "(size, Z, random_state)",
    "how": "CompetingRisksProportionalHazards.fit(how='Cox') chooses the "
    "model (Cox or Fine-Gray), where how= is the estimation method "
    "everywhere else",
    "id column": "the item-id column is i_col in CauseSpecificMCF / "
    "CauseSpecificNHPP.fit_from_df but id_col in the fit_tvc*_from_df "
    "methods",
}
# All tracked by one issue (principle 21).
KNOWN_INCONSISTENCIES = {
    key: "#422: " + reason for key, reason in KNOWN_INCONSISTENCIES.items()
}


def _with_issue(prop: str, reason: str) -> str:
    issue = KNOWN_FAILURE_ISSUES.get(prop)
    return "{}: {}".format(issue, reason) if issue else reason


CASES = [
    replace(
        c,
        xfail={
            **c.xfail,
            **{
                prop: _with_issue(prop, reason)
                for prop, reason in KNOWN_FAILURES.get(c.name, {}).items()
            },
        },
    )
    for c in CASES
]

CASE_BY_NAME: dict[str, Case] = {case.name: case for case in CASES}
assert len(CASE_BY_NAME) == len(CASES), "case names must be unique"


# ---------------------------------------------------------------------------
# Public names that are not registered, and why
# ---------------------------------------------------------------------------
_BASE = (
    "an abstract base or mixin; its concrete fitters are registered one by "
    "one"
)
_FACTORY = "a factory returning a registered fitter"
_RESULT = "a result record (a test statistic or prediction), not a model"
_DATA = "a data container or input schedule, not a model"
_FUNCTION = (
    "a function (data handling, statistical test, metric or reader), not "
    "a model; the metrics and tests have their own reference suites"
)
_COMPONENT = (
    "a building block of a registered model rather than a model itself"
)

OUT_OF_SCOPE: dict[str, str] = {
    # abstract bases and mixins
    "surpyval.Distribution": _BASE,
    "surpyval.ParametricDistribution": _BASE,
    "surpyval.NonParametricDistribution": _BASE,
    "surpyval.MultivariateDistribution": _BASE,
    "surpyval.univariate.parametric.ParametricFitter": _BASE,
    "surpyval.univariate.parametric.OptimisedFitMixin": _BASE,
    "surpyval.univariate.parametric.DiscreteParametricFitter": _BASE,
    "surpyval.AFTFitter": _BASE,
    "surpyval.ProportionalHazardsFitter": _BASE,
    "surpyval.ProportionalOddsFitter": _BASE,
    "surpyval.AdditiveHazardsFitter": _BASE,
    "surpyval.FrailtyFitter": _BASE,
    "surpyval.ParameterSubstitutionFitter": _BASE,
    "surpyval.LifeModel": _BASE,
    "surpyval.recurrent.CountingProcess": _BASE,
    "surpyval.multivariate.Copula": _BASE,
    "surpyval.degradation.PathModel": _BASE,
    # factories of registered fitters
    "surpyval.AFT": _FACTORY,
    "surpyval.PH": _FACTORY,
    "surpyval.PO": _FACTORY,
    "surpyval.AH": _FACTORY,
    # data, schedules and result records
    "surpyval.SurpyvalData": _DATA,
    "surpyval.RecurrentEventData": _DATA,
    "surpyval.multivariate.MultivariateSurpyvalData": _DATA,
    "surpyval.StepSchedule": _DATA,
    "surpyval.StepValuedError": "an exception type",
    "surpyval.LogRankResult": _RESULT,
    "surpyval.recurrent.TrendTestResult": _RESULT,
    "surpyval.recurrent.GoodnessOfFitResult": _RESULT,
    "surpyval.degradation.RULPrediction": _RESULT,
    "surpyval.degradation.ProcessRUL": _RESULT,
    # degradation building blocks
    "surpyval.degradation.LinkedPathModel": _COMPONENT
    + " (the stress link of an accelerated degradation fit, covered by "
    "tests/degradation/test_process_stress.py)",
    "surpyval.degradation.get_path_model": _FUNCTION,
}
# The path shapes of DegradationAnalysis. The linear and power paths are
# registered; one fixture cannot suit every shape (a straight-line
# fixture gives the curved paths a degenerate between-unit covariance, with
# a warning), and each path has its own tests in tests/degradation.
for _path in (
    "ExponentialPath",
    "GompertzPath",
    "LinearPath",
    "LloydLipowPath",
    "LogarithmicPath",
    "MichaelisMentenPath",
    "OffsetExponentialPath",
    "PowerPath",
    "QuadraticPath",
):
    OUT_OF_SCOPE[f"surpyval.degradation.{_path}"] = (
        _COMPONENT + ": a DegradationAnalysis path shape"
    )
for _fn in (
    "fs_to_xcnt",
    "fs_to_xrd",
    "fsl_to_xcnt",
    "fsli_handler",
    "fsli_to_xcnt",
    "round_sig",
    "xcn_to_fs",
    "xcnt_handler",
    "xcnt_to_xrd",
    "xrd_handler",
    "xrd_to_xcnt",
    "handle_xicn",
    "fit_best",
    "from_dict",
    "from_json",
    "logrank",
    "rmst_diff",
    "success_run",
    "gray_test",
    "auc_td",
    "brier_score",
    "integrated_brier_score",
    "survival_probability",
):
    OUT_OF_SCOPE[f"surpyval.{_fn}"] = _FUNCTION
for _fn in ("laplace", "mil_hdbk_189c"):
    OUT_OF_SCOPE[f"surpyval.recurrent.{_fn}"] = _FUNCTION
for _fn in (
    "filliben",
    "fleming_harrington",
    "fleming_harrington_variance",
    "greenwood_variance",
    "kaplan_meier",
    "nelson_aalen",
    "nelson_aalen_variance",
    "plotting_positions",
    "rank_adjust",
    "turnbull",
):
    OUT_OF_SCOPE[f"surpyval.univariate.nonparametric.{_fn}"] = (
        _FUNCTION + " (the estimator behind a registered fitter)"
    )
