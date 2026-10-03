"""The conformance registry's case record and family helpers (see
``registry.py``).

The interface constants and the properties, :class:`Case`, how a
case's functions are called and its fixture fitted, and the helpers
that build the cases of a whole family (``continuous``,
``discrete``, ``regression`` and the regression families).
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.beta import ml
from surpyval.tests.conformance.leaks import quiet
from surpyval.tests.conformance.registry_fixtures import (
    X_DISC,
    X_REG,
    X_STRESS,
    X_UNI,
    Z_REG,
    Z_STRESS,
    Z_STRESS2,
    discrete_data,
    grouped_reg_data,
    reg_data,
    stratified_reg_data,
    stress_data,
    uni_data,
)
from surpyval.univariate.regression import _kinds
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
    "qf_outside": (
        "a probability outside [0, 1] gives NaN there only, with one "
        "warning, from every qf: the model's and its distribution's (#611)"
    ),
    "missing_covariate": "a NaN covariate gives NaN for its row only",
    "missing_fit": "a missing time raises; a missing covariate is dropped",
    "seed_global": "np.random.seed reproduces a draw",
    "seed_explicit": (
        "an explicit seed reproduces a draw, is default_rng(seed), and "
        "leaves the global stream alone"
    ),
    "serialise": "strict-JSON to_dict / from_dict keeps every prediction",
    "pickle": (
        "a fitted model pickles, and the unpickled model predicts, bounds "
        "and saves (to_dict) exactly as the original (#573)"
    ),
    "pickle_paths": (
        "the model of every alternate fit path (fit_from_df, a formula, "
        "fit_tvc, ...) pickles and predicts exactly as before (#573)"
    ),
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
    # test_derivatives.py, for the cases of DIFFERENTIATED.
    "derivatives": (
        "the derivatives a fit or its inference takes (autograd's, or the "
        "model's own) agree with finite differences at the fit"
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
    # test_maximum.py, for the cases whose fit maximises a likelihood.
    "maximum": (
        "a likelihood fit's ``maximum`` says what it reached, it warns "
        "exactly when that is not a verified maximum, and a verified "
        "maximum has a zero gradient and a positive-definite Hessian"
    ),
    # test_comparison.py, for every case.
    "comparison": (
        "neg_ll(), aic(), aic_c() and bic() are methods and log_likelihood "
        "a number, -neg_ll(), wherever a model has them; aic() is "
        "2 k + 2 neg_ll() for a whole k (#572)"
    ),
    # test_attributes.py, for the model classes in DECLARED_ATTRIBUTES.
    "attributes": (
        "every way of building the model (fit, each alternate fit path, "
        "from_dict) gives it the same attributes, each declared on its "
        "class"
    ),
}

# The model classes that declare every attribute their builders set, which
# the "attributes" property checks.
DECLARED_ATTRIBUTES = frozenset(
    {
        "surpyval.ParametricRegressionModel",
        "surpyval.SemiParametricRegressionModel",
        "surpyval.ProportionalOddsModel",
        "surpyval.AdditiveHazardsModel",
        "surpyval.BuckleyJamesModel",
        "surpyval.univariate.competing_risks.regression.fine_gray"
        ".FineGrayModel",
    }
)

# The model classes whose fit or inference differentiates its likelihood
# (or, for a copula, its cdf; for a degradation path, the path), which the
# "derivatives" property checks against finite differences; and the cases
# of those classes that take no derivatives. The other models take none:
# their estimators are closed-form or nonparametric (Kaplan-Meier, Lin-Ying
# additive hazards, Buckley-James, the trees), or their fits search
# without a gradient and take a numerical Hessian (Royston-Parmar, Cox
# frailty, the NHPP and renewal fits, the Wiener, gamma-process and
# destructive degradation fits); the semi-parametric proportional odds
# fit's profile derivatives are internal to it; and the competing-risks
# Cox and Fine-Gray models fit each cause by CoxPH and FineGray, which
# are checked.
DIFFERENTIATED = frozenset(
    {
        "surpyval.Parametric",
        "surpyval.MixtureModel",
        "surpyval.ParametricRegressionModel",
        "surpyval.FrailtyModel",
        "surpyval.SemiParametricRegressionModel",
        "surpyval.univariate.competing_risks.regression.fine_gray"
        ".FineGrayModel",
        "surpyval.univariate.competing_risks.ParametricCompetingRisks",
        "surpyval.recurrent.parametric.parametric_recurrence"
        ".ParametricRecurrenceModel",
        "surpyval.recurrent.regression.proportional_intensity"
        ".ProportionalIntensityModel",
        "surpyval.multivariate.CopulaModel",
        "surpyval.degradation.DegradationModel",
    }
)
_NO_DERIVATIVES = "the fit takes no derivatives: "
NOT_DIFFERENTIATED = {
    "Uniform": _NO_DERIVATIVES + "the MLE is the sample extremes",
    "Binomial": _NO_DERIVATIVES + "a closed-form estimate",
    "Bernoulli": _NO_DERIVATIVES + "a closed-form estimate",
    "FixedEventProbability": _NO_DERIVATIVES + "a closed-form estimate",
    "ExactEventTime": _NO_DERIVATIVES + "a closed-form estimate",
    "Hypoexponential": _NO_DERIVATIVES + "the model is given its rates",
    **dict.fromkeys(
        ("CrowAMSAA", "Duane", "CoxLewis", "ProportionalIntensityNHPP"),
        _NO_DERIVATIVES + "a search on finite-difference gradients, and "
        "a numerical Hessian",
    ),
}

# The model classes whose estimate does not maximise a likelihood, and why;
# the "maximum" property does not apply to them (a case of another class
# whose fit is not a likelihood maximisation excludes it with the reason).
NOT_A_LIKELIHOOD_FIT: dict[str, str] = {
    "surpyval.NonParametric": "a product-limit or self-consistency estimate",
    "surpyval.NeverOccurs": "a fixed model, not fitted",
    "surpyval.InstantlyOccurs": "a fixed model, not fitted",
    "surpyval.AdditiveHazardsModel": (
        "Lin and Ying's estimating equations, a linear system"
    ),
    "surpyval.BuckleyJamesModel": (
        "the Buckley-James estimating equations, by iterated least squares"
    ),
    "surpyval.beta.ml.SurvivalTree": "a tree of greedy splits",
    "surpyval.beta.ml.RandomSurvivalForest": "an ensemble of trees",
    "surpyval.univariate.competing_risks.CompetingRisks": (
        "a non-parametric (Aalen-Johansen) estimate"
    ),
    "surpyval.recurrent.NonParametricCounting": (
        "the non-parametric mean cumulative function"
    ),
    "surpyval.recurrent.CauseSpecificMCF": (
        "the non-parametric mean cumulative function of each cause"
    ),
    "surpyval.degradation.DegradationModel": (
        "two stages: each unit's path by least squares (or a mixed model), "
        "then a fit of the pseudo-failure times, whose own maximum is a "
        "univariate one"
    ),
    "surpyval.degradation.InducedFailureDistribution": (
        "a Monte Carlo of a fitted degradation model"
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
        "attributes",
        "maximum",
        "pickle_paths",
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
        if prop == "derivatives" and (
            self.model_class not in DIFFERENTIATED
            or self.name in NOT_DIFFERENTIATED
        ):
            return False
        if prop == "aliasing_constant" and not self.intercept:
            return False
        if prop == "attributes" and (
            self.model_class not in DECLARED_ATTRIBUTES
        ):
            return False
        if prop == "maximum" and self.model_class in NOT_A_LIKELIHOOD_FIT:
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
    # The finished registry (known failures applied), which imports
    # this module: looked up when called, not at import.
    from surpyval.tests.conformance.registry import CASE_BY_NAME

    case = CASE_BY_NAME[name]
    with quiet():
        return case.fit(case.data())


def tvc_path(case):
    """The case's fit by ``fit_tvc`` where its fitter has one: one
    ``(0, x]`` interval per subject, the time-fixed data; else ``None``."""
    fitter = getattr(sp, case.name, None)
    if not hasattr(fitter, "fit_tvc"):
        return None

    def fit_tvc(d):
        n = np.asarray(d["n"])
        return fitter.fit_tvc(
            np.arange(n.sum()),
            np.zeros(n.sum()),
            np.repeat(d["x"], n),
            np.repeat(d["c"], n),
            np.repeat(d["Z"], n, axis=0),
        )

    return fit_tvc


def fitted(case):
    """The case's model fitted to its fixture (cached per session).

    The same object is shared by every test, so tests must not change it.
    """
    return _fitted(case.name)


def skip_without_finite_maximum(case):
    """Skip a check of a fit's inference where the fit has no finite
    maximum (its ``maximum`` says so, and it has warned that its standard
    errors and bounds are meaningless): a Beta4 whose end has reached its
    extreme observation, where the likelihood is unbounded. Its derivatives
    and Wald bounds there describe a point that is not an estimate."""
    model = fitted(case)
    if getattr(model, "maximum", None) == "no finite maximum":
        pytest.skip(
            f"{case.name}: the fit has no finite maximum, so its "
            "derivatives and bounds are not those of an estimate"
        )


def refit(case, data):
    """Fit the case to ``data``, silencing the optimisers' warnings.

    A raw numerical warning leaking from the package is not silenced
    (see ``leaks.py``).
    """
    with quiet():
        return case.fit(data)


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
        return fitter.fit_from_df(df, x_col="x", c_col="c", n_col="n", **fixed)

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
    "PH": _kinds.PROPORTIONAL_HAZARD,
    "AFT": _kinds.ACCELERATED_FAILURE_TIME,
    "PO": _kinds.PROPORTIONAL_ODDS,
    "AH": _kinds.ADDITIVE_HAZARD,
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
    "Exponential",
    "InverseExponential",
)
DUAL_LIFE_MODELS = ("DualExponential", "DualPower", "PowerExponential")


def _accelerated_life_family():
    out = []
    for lm in LIFE_MODELS + DUAL_LIFE_MODELS:
        dual = lm in DUAL_LIFE_MODELS
        fitter = sp.AcceleratedLife(sp.Weibull, getattr(sp.life_models, lm))
        out.append(
            regression(
                f"WeibullAL[{lm}]",
                fitter,
                data=functools.partial(stress_data, 2 if dual else 1),
                x=X_STRESS,
                Z=Z_STRESS2 if dual else Z_STRESS,
                fitters=(f"surpyval.life_models.{lm}",)
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
    # One coefficient per stress column (#530): any number of columns,
    # of any sign, so the ordinary regression data serve.
    out.append(
        regression(
            "WeibullAL[GeneralLogLinear]",
            sp.AcceleratedLife(sp.Weibull, sp.life_models.GeneralLogLinear),
            fitters=("surpyval.life_models.GeneralLogLinear",),
            slow=REFIT_PROPERTIES,
            coefficients=None,
            exclude={
                "aliasing": "model.aliased are positions in phi_params, "
                "after the constant c, not columns of Z; the aliasing is "
                "tests/univariate/regression/test_general_log_linear.py's"
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
    # The log-normal frailty (#343), whose group integral is by quadrature.
    lognormal = sp.Frailty(sp.Weibull, family="lognormal")
    out.append(
        replace(
            out[0],
            name="WeibullFrailty[lognormal]",
            fitters=(),
            fit=_fit(lognormal),
            paths={
                "fit_from_df": lambda d: lognormal.fit_from_df(
                    pd.DataFrame(d["Z"], columns=["z0", "z1"]).assign(
                        x=d["x"], c=d["c"], n=d["n"], g=d["groups"]
                    ),
                    x_col="x",
                    group_col="g",
                    Z_cols=["z0", "z1"],
                    c_col="c",
                    n_col="n",
                )
            },
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
    cox_frailty = regression(
        "CoxFrailty",
        sp.CoxFrailty,
        data=grouped_reg_data,
        model_class="surpyval.CoxFrailtyModel",
        rows=("x", "Z", "c", "n", "groups"),
        labels=("groups",),
        paths={
            "fit_from_df": lambda d: sp.CoxFrailty.fit_from_df(
                pd.DataFrame(d["Z"], columns=["z0", "z1"]).assign(
                    x=d["x"], c=d["c"], n=d["n"], g=d["groups"]
                ),
                x_col="x",
                group_col="g",
                Z_cols=["z0", "z1"],
                c_col="c",
                n_col="n",
            )
        },
        jump_functions=("hf", "df"),
        exclude={"df_hf_sf": step},
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
    po = regression(
        "ProportionalOdds",
        sp.ProportionalOdds,
        model_class="surpyval.ProportionalOddsModel",
        jump_functions=("hf", "df"),
        exclude={"df_hf_sf": step},
        rtol=1e-6,
        coefficients=_beta,
        intercept=True,
    )
    return [cox, strat, cox_frailty, ah, bj, po]


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
# The uncertainty methods swept by test_options.py
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
