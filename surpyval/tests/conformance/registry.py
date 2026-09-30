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
from surpyval.univariate.regression._fit_skeleton import ORIGIN_MAPS

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
    "scalar": (
        "a scalar query gives a scalar that agrees with the same query as "
        "a 1-D array"
    ),
    "array2d": "a 2-D query keeps its shape and agrees element-wise",
    "empty": "an empty query returns an empty result of its shape",
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
        "gives (2,) two-sided and a scalar one-sided"
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
    # test_aliasing.py, over each case's ``coefficients``.
    "aliasing": (
        "a covariate column repeating another is aliased: its coefficient "
        "nan and listed in ``aliased``, one warning naming it, and the "
        "other coefficients and every prediction those of the fit "
        "without it"
    ),
    "aliasing_constant": (
        "a constant covariate column is aliased, the same way, where the "
        "model has an intercept"
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
        "aliasing",
        "aliasing_constant",
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
    "row_independence": frozenset(WITH_COVARIATES),
    "missing_covariate": frozenset(WITH_COVARIATES),
    "outside_data": _EVERY - {BIVARIATE},
    "aliasing": frozenset(WITH_COVARIATES),
    "aliasing_constant": frozenset(WITH_COVARIATES),
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
    # The fitted model's covariate coefficients, one per column of Z (a
    # row per cause for a model fitted cause by cause), for
    # test_aliasing.py. A model with covariates must give them, or
    # exclude "aliasing" with the reason it has none.
    coefficients: Callable[[Any], Any] | None = None
    # Whether a constant covariate column is aliased: the model has an
    # intercept that absorbs it (a Cox-type baseline hazard, a scale
    # that a constant in the linear predictor moves).
    intercept: bool = False
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
        if prop == "aliasing_constant" and not self.intercept:
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
    ``xfail`` entry for ``prop`` becomes a strict xfail mark (non-strict
    where :data:`NON_STRICT` says the outcome depends on the build), and
    a slow property of a case gets the ``slow`` mark.
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
            strict = prop not in NON_STRICT.get(case.name, ())
            marks.append(
                pytest.mark.xfail(strict=strict, reason=case.xfail[prop])
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


def uni_exact_data():
    """``uni_data`` with every value exactly observed: the Uniform's MLE
    refuses censored values (#460)."""
    data = uni_data()
    data["c"] = np.zeros_like(data["c"])
    return data


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


def beta_geometric_data():
    """40 draws from BetaGeometric(3, 5), right censored above 8: more
    dispersed than a Geometric, so the fit has an interior maximum (a, b =
    1.72, 3.01). ``discrete_data`` is under-dispersed for it: its fit runs
    to the geometric limit, where the likelihood has no finite maximum
    (#392), and no bound can be tested there."""
    return {
        "x": np.array([1, 2, 3, 4, 6, 7, 8]),
        "c": np.array([0, 0, 0, 0, 0, 0, 1]),
        "n": np.array([14, 9, 4, 3, 4, 1, 5]),
    }


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
        draw=kw.pop("draw", lambda m, s: m.random(15, random_state=s)),
        explicit_seed=kw.pop("explicit_seed", True),
        **kw,
    )


def discrete(name, fitter=None, start=0, **kw):
    fitter = getattr(sp, name) if fitter is None else fitter
    return Case(
        name=kw.pop("case_name", name),
        fitters=kw.pop("fitters", (f"surpyval.{name}",)),
        model_class="surpyval.Parametric",
        interface=UNIVARIATE,
        data=kw.pop("data", functools.partial(discrete_data, start)),
        fit=_fit(fitter),
        functions=UNI_FUNCTIONS + ("qf",),
        x=X_DISC,
        continuous=False,
        paths=kw.pop("paths", _parametric_paths(fitter)),
        draw=lambda m, s: m.random(15, random_state=s),
        explicit_seed=True,
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
        coefficients=kw.pop("coefficients", _phi),
        **kw,
    )


def _phi(model):
    """A parametric regression's covariate coefficients."""
    return np.asarray(model.params, dtype=float)[model.k_dist :]


def _beta(model):
    return model.beta


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


# A constant column is aliased where the family has an intercept: where a
# constant in the linear predictor moves the baseline's parameters and
# nothing else (ORIGIN_MAPS, as the fit decides it).
_KINDS = {
    "PH": "Proportional Hazard",
    "AFT": "Accelerated Failure Time",
    "PO": "Proportional Odds",
    "AH": "Additive Hazards",
}


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
                regression(
                    name,
                    getattr(sp, name),
                    slow=slow,
                    exclude=exclude,
                    intercept=(_KINDS[kind], base) in ORIGIN_MAPS,
                )
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
                coefficients=None,
                exclude={
                    "aliasing": "a life model takes a fixed number of "
                    "stress columns (another raises a ValueError naming "
                    "it), and its parameters are not one per column; the "
                    "dual-stress models' equal columns are "
                    "tests/univariate/regression/test_aliasing.py's"
                },
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
                coefficients=_beta,
                intercept=("Proportional Hazard", base) in ORIGIN_MAPS,
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
        coefficients=_beta,
        intercept=True,
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
        coefficients=_beta,
        intercept=True,
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
        },
        rtol=1e-6,
        coefficients=_beta,
        intercept=True,
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
        coefficients=_beta,
        intercept=True,
    )
    return [cox, strat, ah, bj]


_NO_COEFFICIENTS = (
    "no coefficients: a tree splits on the columns, and a repeated column "
    "only offers the same splits again"
)


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
            # A tree draws the features it considers at each split
            # (n_features_split="sqrt"); with random_state=None from the
            # global stream, so it is fitted under a fixed global seed, as
            # the forest is.
            fit=_seeded(_fit(ml.SurvivalTree, kind=kind)),
            functions=UNI_FUNCTIONS,
            x=X_REG,
            Z=Z_REG,
            rows=("x", "Z", "c", "n"),
            covariates="Z",
            z_style="grid",
            jump_functions=("hf", "df") if kind == "non-parametric" else (),
            # The draw is the fit itself: two fits under one seed agree.
            draw=lambda m, s: ml.SurvivalTree.fit(
                **reg_data(), kind=kind, random_state=s
            ).sf(X_REG, Z_REG),
            explicit_seed=True,
            exclude={
                "aliasing": _NO_COEFFICIENTS,
                **(
                    {"df_hf_sf": "non-parametric leaves: hf and df are jumps"}
                    if kind == "non-parametric"
                    else {}
                ),
            },
        )

    forest = Case(
        name="RandomSurvivalForest",
        fitters=("surpyval.beta.ml.RandomSurvivalForest",),
        model_class="surpyval.beta.ml.RandomSurvivalForest",
        interface=REGRESSION,
        data=reg_data,
        # With random_state=None the forest bootstraps from the global
        # stream, so the registry fits it under a fixed global seed.
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
        draw=lambda m, s: m.__class__.fit(
            **reg_data(), n_trees=2, random_state=s
        ).sf(X_REG, Z_REG),
        explicit_seed=True,
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
            "aliasing": _NO_COEFFICIENTS,
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
    out.append(continuous("Uniform", data=uni_exact_data))
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
            exclude={
                "units": "its likelihood is unbounded (a shape below 1 "
                "puts infinite density at a support end), so the MLE has "
                "no maximum and stops where its search gave up, which "
                "depends on the units: shapes 1.00, 1.19 on the fixture, "
                "0.18, 0.18 on it times 7.3. The fit warns (No finite "
                "maximum, or not a verified maximum) and recommends "
                "how='MPS', which is unit-free: test_beta4_no_maximum.py "
                "(#385)"
            },
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
                    # the survival data to refit (#403), drawn from the
                    # lifetimes random draws (inf for a unit that never
                    # fails); one call, so a Generator seed is used once
                    draw=lambda m, s: m.random_data(15, random_state=s),
                    explicit_seed=True,
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
            data=beta_geometric_data,
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
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
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
                # Bernoulli's qf inverts its ff since #344; the flat
                # FixedEventProbability has no quantile to test.
                functions=(
                    ("sf", "ff", "Hf", "qf")
                    if name == "Bernoulli"
                    else ("sf", "ff", "Hf")
                ),
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
                draw=lambda m, s: m.random(15, random_state=s),
                explicit_seed=True,
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
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
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
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
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
                draw=lambda m, s: m.random(5, random_state=s),
                explicit_seed=True,
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
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
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
            draw=lambda m, s: m.random(15, random_state=s),
            explicit_seed=True,
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
            df, x_col="x", e_col="e", Z_cols=["z0"], n_col="n", model=how
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
            fit=_fit(cr.CompetingRisks, how=method),
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
                fit=_fit(cr.CompetingRisksProportionalHazards, model=how),
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
                coefficients=lambda m: m.betas,
                intercept=True,
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
            fit=_fit(cr.FineGray, event="a"),
            functions=("sf", "cif"),
            x=X_CR,
            Z=Z_CR,
            rows=("x", "Z", "e", "n"),
            covariates="Z",
            rtol=1e-6,
            coefficients=_beta,
            intercept=True,
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
    return m.count_terminated_simulation(3, items=2, random_state=s)


def _timed_draw(m, s):
    # For a falling intensity (the CoxLewis fixture: beta = -0.021, so
    # cif(inf) = 6.04), which count termination refuses (#386).
    return m.time_terminated_simulation(60.0, items=2, random_state=s)


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
                draw=_timed_draw if name == "CoxLewis" else _counting_draw,
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
                coefficients=lambda m: m.coeffs,
                # The baseline rate (HPP) or the Duane scale b absorbs a
                # constant.
                intercept=True,
                draw=lambda m, s: m.count_terminated_simulation(
                    3, items=2, random_state=s, Z=[0.5]
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
                call_kwargs={"items": 30, "random_state": 1},
                x=X_REC,
                rows=("x", "i", "c", "n"),
                paths={
                    "fit_from_recurrent_data": _recurrent_data_path(
                        fitter, **kw
                    )
                },
                draw=lambda m, s: m.mcf(X_REC, items=5, random_state=s),
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
# LogisticAH: counts against repeated rows agree to 1e-6 here, but CI's
# Python 3.11 runner (the same numpy 2.4 / scipy 1.17) stops 1.7e-4 apart
# in ff at a value near 0 (-0.0194), on every run: the same effect.
_LOOSE += ("LogisticAH",)
CASES = [replace(c, rtol=1e-3) if c.name in _LOOSE else c for c in CASES]


# ---------------------------------------------------------------------------
# fit_from_df paths (#511): every fitter reads a DataFrame, and gives the
# model its fit gives on the same arrays. Each family names the columns as
# its fit_from_df does (utils/dataframe.py).
# ---------------------------------------------------------------------------
def _frame(d, keys):
    """The per-row entries ``keys`` of a fixture as DataFrame columns (a
    2-D entry as columns ``<key>0``, ``<key>1``, ...), and the rest."""
    cols, rest = {}, {}
    for key, value in d.items():
        if key not in keys:
            rest[key] = value
            continue
        value = np.asarray(value)
        if value.ndim == 2:
            for k in range(value.shape[1]):
                cols[f"{key}{k}"] = value[:, k]
        else:
            cols[key] = value
    return pd.DataFrame(cols), rest


def _column_names(df, key):
    return [k for k in df.columns if k[: len(key)] == key and k != key]


def _uni_df(fitter, **fixed):
    """``x``, ``c``, ``n``, ``tl`` (and ``xl`` / ``xr`` for interval
    rows), named as the univariate fit_from_df names them."""

    def run(d):
        df, rest = _frame(d, ("x", "c", "n", "tl", "tr"))
        if "x" in df:
            names = {"x": "x"}
        else:
            df = df.rename(columns={"x0": "xl", "x1": "xr"})
            names = {"xl": "xl", "xr": "xr"}
        names |= {k: k for k in ("c", "n", "tl", "tr") if k in df}
        return fitter.fit_from_df(df, **names, **fixed, **rest)

    return run


def _col_df(fitter, keys=("x", "i", "c", "n", "e"), **fixed):
    """The ``<key>_col`` names (and ``Z_cols``) of the recurrent,
    regression and competing-risks fit_from_df."""

    def run(d):
        df, rest = _frame(d, keys + ("Z",))
        names = {f"{k}_col": k for k in keys if k in df}
        if "Z" in d:
            names["Z_cols"] = _column_names(df, "Z")
        return fitter.fit_from_df(df, **names, **fixed, **rest)

    return run


def _copula_df(fitter, **fixed):
    def run(d):
        df, rest = _frame(d, ("x", "n"))
        return fitter.fit_from_df(
            df, x=_column_names(df, "x"), n="n", **fixed, **rest
        )

    return run


def _df_paths():
    """Case name -> its fit_from_df path, for the cases that had none."""
    paths = {
        name: _uni_df(getattr(sp, name))
        for name in ("KaplanMeier", "NelsonAalen", "FlemingHarrington")
    }
    paths["Turnbull"] = _uni_df(sp.Turnbull)
    paths["Weibull[xcnt]"] = _uni_df(sp.Weibull)
    paths["RoystonParmar"] = _uni_df(sp.RoystonParmar)
    paths["MixtureModel"] = _uni_df(sp.MixtureModel, dist=sp.Weibull, m=2)
    paths["Binomial"] = _uni_df(sp.Binomial, n_trials=5)
    for name in ("Bernoulli", "FixedEventProbability", "ExactEventTime"):
        paths[name] = _uni_df(getattr(sp, name))
    for kind in ("weibull", "exponential", "non-parametric"):
        paths[f"SurvivalTree[{kind}]"] = _seeded(
            _col_df(ml.SurvivalTree, ("x", "c", "n"), kind=kind)
        )
    paths["RandomSurvivalForest"] = _seeded(
        _col_df(ml.RandomSurvivalForest, ("x", "c", "n"), n_trees=3)
    )
    paths["CoxPH[strata]"] = _col_df(
        sp.CoxPH, ("x", "c", "n", "strata"), tie_method="efron"
    )
    paths["CompetingRisks[Kaplan-Meier]"] = _col_df(
        cr.CompetingRisks, ("x", "e", "n"), how="Kaplan-Meier"
    )
    paths["FineGray"] = _col_df(cr.FineGray, ("x", "e", "n"), event="a")
    for name in ("HPP", "CrowAMSAA", "Duane", "CoxLewis"):
        paths[name] = _col_df(getattr(rc, name))
    paths["NonParametricCounting"] = _col_df(rc.NonParametricCounting)
    for name in ("ProportionalIntensityHPP", "ProportionalIntensityNHPP"):
        paths[name] = _col_df(getattr(rc, name))
    paths["GeneralizedRenewal"] = _col_df(rc.GeneralizedRenewal)
    paths["GeneralizedRenewal[kijima ii]"] = _col_df(
        rc.GeneralizedRenewal, kijima="ii"
    )
    paths["GeneralizedOneRenewal"] = _col_df(rc.GeneralizedOneRenewal)
    paths["ARA"] = _col_df(rc.ARA, m=1)
    paths["ARI"] = _col_df(rc.ARI, m=1)
    paths["CauseSpecificMCF"] = _col_df(rc.CauseSpecificMCF)
    paths["CauseSpecificNHPP"] = _col_df(rc.CauseSpecificNHPP)
    paths["DestructiveDegradation"] = lambda d: (
        dg.DestructiveDegradation.fit_from_df(
            pd.DataFrame(d), x="x", y="y", threshold=20.0
        )
    )
    for name in ("Independence", "Clayton", "Gumbel", "Frank", "Gaussian"):
        paths[f"{name}Copula"] = _copula_df(
            getattr(mv, name), margins=[sp.Weibull, sp.Weibull]
        )
    return paths


def _add_df_path(case, paths=_df_paths()):
    if case.name not in paths:
        return case
    assert "fit_from_df" not in case.paths, case.name
    return replace(
        case, paths={**case.paths, "fit_from_df": paths[case.name]}
    )


CASES = [_add_df_path(c) for c in CASES]

# Cases with no fit_from_df path, and why (checked by test_fit_paths.py).
NO_DF_PATH: dict[str, str] = {
    "Hypoexponential": "built from its parameters: its fit refuses data",
    "NeverOccurs": "a fixed model, not fitted",
    "InstantlyOccurs": "a fixed model, not fitted",
    "InducedFailureDistribution": "derived from a fitted degradation model",
}


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
    # The level argument, which takes alpha (the tail probability).
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
    # Run only in the nightly calibration job (``--run-calibration``):
    # a sweep that takes minutes, where a faster family already sweeps
    # the same method in every run.
    nightly: bool = False
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
# times, and only in the full suite. Rayleigh, Geometric and Uniform
# joined in #421 (a df bound stalled on the far side of the estimate;
# the Uniform's search stalled at the support's edge), then
# NegativeBinomial and ExpoWeibull (profiles that did not follow their
# valleys, bounds that were rounding noise where they are infinite, and
# bands that were not nested). Beta4's MLE has no maximum (#385).
_LR_X = {
    "Weibull": np.array([4.0, 8.0, 13.0]),
    "Rayleigh": np.array([3.2, 8.0, 14.6]),
    "Geometric": np.array([2.0, 5.0, 8.0]),
    "Uniform": np.array([3.2, 8.0, 14.6]),
    "NegativeBinomial": np.array([2.0, 5.0, 8.0]),
    # One time for ExpoWeibull: its searches in a three-parameter valley
    # take seconds each, and three times tripled a 17-minute sweep. The
    # tail (13) is where its bands were hardest (the hf nesting in #421).
    "ExpoWeibull": np.array([13.0]),
}
_LR_NIGHTLY = {"NegativeBinomial", "ExpoWeibull"}


# Fits with no parameter covariance by design, so no Wald bounds: the
# Uniform's MLE sits on the sample extremes, where the likelihood's
# curvature says nothing about its uncertainty (#460).
_NO_WALD = {"Uniform"}


def _parametric_bounds(case):
    on = tuple(f for f in _ON_ALL if f in case.functions)
    out = []
    if case.name not in _NO_WALD:
        out += [
            Bound("cb", on=on, kwargs={"method": "wald"}, label="cb[wald]"),
            Bound(
                "param_cb",
                kind="param",
                kwargs={"method": "wald"},
                label="param_cb[wald]",
            ),
        ]
        # Bounds on the B-lives and the mean (#494)
        out += [
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"method": "wald"},
                label="quantile_cb[wald]",
            ),
            Bound(
                "mean_cb",
                kind="summary",
                kwargs={"method": "wald"},
                query=((),),
                # Documented: with support ends among the parameters the
                # Wald bound is on the mean's own scale
                in_range=case.name != "Beta4",
                label="mean_cb[wald]",
            ),
        ]
    # The likelihood-ratio search is swept on the cases in _LR_X only.
    # The ExpoWeibull's and NegativeBinomial's sweeps (searches in
    # multi-parameter valleys, seconds each) took about ten minutes on
    # four cores, so they run nightly, with the calibration studies; the
    # other four sweep in every run, and test_likelihood_ratio_edges.py
    # checks those two families' edges and valleys directly (#421).
    # (Documented: it is not available for offset, limited-failure or
    # zero-inflated models.)
    if case.name not in _LR_X:
        return tuple(out)
    x = _LR_X[case.name]
    lr = dict(
        wald=False,
        nan_ok=True,
        rtol=1e-3,
        slow=True,
        nightly=case.name in _LR_NIGHTLY,
    )
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
    if case.continuous:
        # (a discrete quantile's bound inverts the band on ff at every
        # count up to it: a likelihood-ratio search at each)
        out.append(
            Bound(
                "quantile_cb",
                point="qf",
                kwargs={"method": "lr"},
                query=(0.1, 0.5, 0.9),
                label="quantile_cb[lr]",
                **lr,
            )
        )
        out.append(
            Bound(
                "mean_cb",
                kind="summary",
                kwargs={"method": "lr"},
                query=((),),
                label="mean_cb[lr]",
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
                    # Not a Wald interval: a whole path rarely stays inside
                    # a narrow strip, so the critical value does not go to
                    # 0 as alpha_ci -> 1 (0.27 at 1 - 1e-6, Hall-Wellner
                    # over the whole range), and the band does not close
                    # onto the estimate. (It used to be left out because
                    # the search at alpha_ci -> 1 did not end, #420.)
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
            kwargs={"n_boot": 40, "random_state": 1},
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
            in_range=bound_type == "exp",
            nan_ok=True,
            per_cause=per_cause,
            label=f"mcf_cb[{bound_type},{interp}]",
        )
        for bound_type in ("exp", "normal")
        for interp in ("step", "linear")
    )


_PARAM_CB = Bound("param_cb", kind="param")
_BOOT = {"n_boot": 20, "random_state": 1}


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
                kwargs={
                    "method": "bootstrap",
                    "n_boot": 10,
                    "random_state": 1,
                },
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
                kwargs={"n_boot": 10, "random_state": 1},
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
    # Its baselines are steps: interp= takes "step" only (#416).
    "CompetingRisksProportionalHazards[Cox]": ("step",),
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
    data = model.time_terminated_simulation_data(
        60.0, items=40, random_state=1
    )
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
        return {"tie_method": methods}, {}, _cox_sample
    if name == "CompetingRisksProportionalHazards[Cox]":
        return {"tie_method": ("efron", "breslow")}, {}, _cr_sample
    if name == "CompetingRisks[Nelson-Aalen]":
        methods = ("Nelson-Aalen", "Kaplan-Meier")
        return (
            {"how": methods},
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
    if name == "Logistic":
        # The scale, not the location: from a far location the fit
        # recovers with scipy 1.17 but stops short with 1.18, so only a
        # far scale fails the same way everywhere.
        return _far_start(case, lambda m: _scaled_start(m.params, 1))
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
KNOWN_FAILURES: dict[str, dict[str, str]] = {}


# -- option sweeps (test_options.py) ------------------------------------
# Keyed "<property>[<bound name>]" (or "interp[<value>]",
# "estimators_agree[<option>]"); grouped as in the report for #379.
def _each(props, name, reason):
    return {f"{p}[{name}]": reason for p in props}


_NEGATIVE_VARIANCE = (
    "no Wald interval exists, so param_cb is nan (with a warning saying "
    "why, #411): a parameter at or near the edge of its support has a "
    "negative variance (covariance diagonal"
)
_OPTION_FAILURES: dict[str, dict[str, str]] = {
    # H. boundary estimates with a non-positive variance
    "GeneralizedRenewal": _each(
        ("cb_contains",),
        "param_cb",
        _NEGATIVE_VARIANCE + " -0.031 for q = 2.7e-16 and -10.6 for " "alpha)",
    ),
    "ARA": _each(
        ("cb_contains",),
        "param_cb",
        _NEGATIVE_VARIANCE + " -0.026 for rho = 1 - 3e-16 and -8.09 for "
        "alpha)",
    ),
    "ARI": _each(
        ("cb_contains",),
        "param_cb",
        "no Wald interval exists, so param_cb is nan (with a warning saying "
        "why, #411): rho = 1.0 exactly, the upper end of its (0, 1) "
        "support, with a variance of -0.0021 (it raised "
        "ZeroDivisionError, the logit of 1)",
    ),
    "Beta4": _each(
        ("cb_contains",),
        "param_cb[wald]",
        _NEGATIVE_VARIANCE + " -8.4e-5 for alpha = 1.00008, whose fit "
        "runs to the edge)",
    ),
}
# The issue tracking each case's option failures (by key where a case
# has failures of more than one kind); it leads each reason.
_OPTION_ISSUES: dict[str, str | dict[str, str]] = {
    "GeneralizedRenewal": "#461",
    "ARA": "#461",
    "ARI": "#461",
    "Beta4": "#461",
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
    # regression, Fine-Gray, copula, mixture and degradation fits of data
    # whose likelihood has no maximum (the #392 class, outside univariate)
    "no maximum": "#392",
}
# (The regression, frailty and Fine-Gray fits of a covariate level with no
# events now warn that the likelihood has no finite maximum, #392.)
_CONVERGENCE_FAILURES: dict[str, tuple[str, str]] = {}
for _name, (_group, _reason) in _CONVERGENCE_FAILURES.items():
    KNOWN_FAILURES[_name] = {
        **KNOWN_FAILURES.get(_name, {}),
        "convergence": f"{_CONVERGENCE_ISSUES[_group]}: {_reason}",
    }


# -- aliasing (test_aliasing.py) --------------------------------------------
# The proportional-intensity fits split a repeated column's effect between
# the two columns wherever BFGS stopped (HPP: -0.231 as -0.116 and -0.116;
# NHPP: -0.1155 and -0.1158), and a constant column took 0.056 from the
# baseline (HPP rate 0.0807 -> 0.0764), silently.
for _name in ("ProportionalIntensityHPP", "ProportionalIntensityNHPP"):
    KNOWN_FAILURES[_name] = {
        **KNOWN_FAILURES.get(_name, {}),
        "aliasing": "#502: a repeated covariate column is not aliased; "
        "the fit splits its coefficient between the two columns, silently",
        "aliasing_constant": "#502: a constant covariate column is not "
        "aliased; it takes part of the baseline rate, silently",
    }


# Known failures whose outcome depends on the numpy / scipy / BLAS build,
# so they are non-strict xfails: case name -> properties. The fits started
# far from the maximum were (#427, #428, #429); they now reach it, or say
# they did not, on every build.
NON_STRICT: dict[str, frozenset[str]] = {}


# The issue that tracks each kind of known failure; its number leads the
# xfail reason, so a test report says where the fix is being worked on.
KNOWN_FAILURE_ISSUES: dict[str, str] = {
    "missing_query": "#382",
    "qf_ff": "#383",
}
# The option-sweep and outside-data failures name their issue in the
# reason itself (see _OPTION_ISSUES), as one key can fail for different
# reasons in different cases.


# Option conventions (test_options.py, CONVENTIONS) the package breaks:
# convention -> the inconsistency. Each is a strict xfail; the test's
# message lists every method concerned.
# Options named differently somewhere, keyed by the convention they break,
# each reason starting "#NNN: " (principle 21). None since #422.
KNOWN_INCONSISTENCIES: dict[str, str] = {}


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
    "surpyval.CovariatePath": _DATA,
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
    "weibayes",
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
