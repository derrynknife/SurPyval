"""The model registry behind the conformance suite (#379).

Every public model *kind* is registered here once: how to fit it on a
small deterministic fixture (and by its alternate fit paths), how its
functions are called, and which of the generic properties apply to it.
The property modules beside this file run every registered case through
every property that applies, so a new model gets the whole battery by
being registered, and ``test_completeness.py`` fails when something
public is neither registered nor listed in :data:`OUT_OF_SCOPE` with a
reason.

The registry is four modules beside this one, gathered here; import its
names from this module:

- ``registry_fixtures.py``: the deterministic fixtures;
- ``registry_families.py``: the interface constants, the properties,
  :class:`Case` and :class:`Bound`, how a case is called and fitted, and
  the family helpers (``continuous``, ``discrete``, ``regression`` and
  the regression families);
- ``registry_cases.py``: every case, with its fit paths, bound and option
  sweeps, and the starved starts of the convergence test;
- ``registry_known_failures.py``: :data:`KNOWN_FAILURES` and the other
  known failures and inconsistencies.

This module applies the known failures to the cases, and holds
:func:`cases_for` and :data:`OUT_OF_SCOPE`.

Registering a model
-------------------
Add a :class:`Case` to :data:`CASES` in ``registry_cases.py`` (the family
helpers in ``registry_families.py`` do most of the work). A case names

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
:data:`KNOWN_FAILURES` (``registry_known_failures.py``) with a one-line
description of the failure, which makes it a strict xfail -- the suite
stays green, and fails the day the bug is fixed, as the reminder to
delete the entry. A known failure of one fit path or one missing input is
keyed ``"fit_paths[<path>]"`` or ``"missing_fit[<key>]"``.

Fixtures are tiny and built without a random number generator (quantiles
of a known distribution, in a fixed order), so a failure reproduces
from the case alone.
"""

from dataclasses import replace

from surpyval.tests.conformance.registry_cases import (  # noqa: F401
    _BOOT,
    _CLOSED_FORM,
    _COPULA_FAMILIES,
    _EXACT,
    _FAST_BOUNDS,
    _FAST_ESTIMATORS,
    _INTERP,
    _LOOSE,
    _LR_NIGHTLY,
    _LR_X,
    _NO_COVARIANCE,
    _NO_STARVE,
    _NO_WALD,
    _NP_INTERP,
    _ON_ALL,
    _ON_SURVIVAL,
    _PARAM_CB,
    _PLAIN_CONTINUOUS,
    _PLAIN_DISCRETE,
    _ROTATED_COPULAS,
    _START_IS_MAXIMUM,
    _U_LARGE,
    CASES,
    FAR,
    GOMPERTZ,
    N_LARGE,
    NO_DF_PATH,
    _add_df_path,
    _bounds,
    _censored_sample,
    _col_df,
    _column_names,
    _comonotone,
    _competing_risks,
    _copula_df,
    _copula_sample,
    _copulas,
    _counting_draw,
    _cox_sample,
    _cr_sample,
    _degradation,
    _df_paths,
    _estimators,
    _far,
    _far_start,
    _fit_mixture,
    _frame,
    _gompertz_fun,
    _mcf_bounds,
    _no_event_level,
    _noise_free,
    _nonparametric,
    _nonparametric_bounds,
    _parametric_bounds,
    _parametric_estimators,
    _parametric_start,
    _quantile_sample,
    _recurrent,
    _recurrent_data_path,
    _recurrent_sample,
    _scaled_start,
    _starve,
    _timed_draw,
    _uni_df,
    _univariate,
    _with_convergence,
    _with_options,
)
from surpyval.tests.conformance.registry_families import (  # noqa: F401
    _APPLICABLE,
    _EVERY,
    _KINDS,
    _NO_COEFFICIENTS,
    _SURVIVAL,
    ALPHAS,
    BASELINES,
    BIVARIATE,
    CAUSES,
    CAUSES_REGRESSION,
    COUNTING,
    COUNTING_CAUSES,
    COUNTING_REGRESSION,
    DECLARED_ATTRIBUTES,
    DIFFERENTIATED,
    DUAL_LIFE_MODELS,
    FAST_REGRESSIONS,
    LIFE_MODELS,
    NOT_A_LIKELIHOOD_FIT,
    NOT_DIFFERENTIATED,
    PROPERTIES,
    Q_PROBS,
    REFIT_PROPERTIES,
    REGRESSION,
    UNI_FUNCTIONS,
    UNIVARIATE,
    WITH_COVARIATES,
    Bound,
    Case,
    _accelerated_life_family,
    _beta,
    _cox_paths,
    _fit,
    _fitted,
    _frailty_family,
    _parametric_paths,
    _phi,
    _regression_family,
    _seeded,
    _semi_parametric,
    _trees,
    call,
    call_native,
    calls,
    continuous,
    discrete,
    fitted,
    flat,
    predictions,
    query,
    refit,
    regression,
    skip_without_finite_maximum,
    tvc_path,
)
from surpyval.tests.conformance.registry_fixtures import (  # noqa: F401
    N_REG,
    X_COP,
    X_CR,
    X_DESTR,
    X_DISC,
    X_PATH,
    X_PROC,
    X_REC,
    X_REG,
    X_STRESS,
    X_UNI,
    Z_AH,
    Z_CR,
    Z_REC,
    Z_REG,
    Z_STRESS,
    Z_STRESS2,
    ah_data,
    beta_geometric_data,
    binary_data,
    copula_data,
    cr_data,
    destructive_data,
    discrete_data,
    exact_event_data,
    grouped_reg_data,
    lfp_count_data,
    lfp_data,
    mixture_data,
    offset_data,
    path_data,
    process_data,
    quantiles,
    recurrent_data,
    reg_data,
    scramble,
    stratified_reg_data,
    stress_data,
    uni_data,
    uni_exact_data,
    unit_interval_data,
    xcnt_data,
    zi_data,
)
from surpyval.tests.conformance.registry_known_failures import (  # noqa: F401
    _CONVERGENCE_FAILURES,
    _CONVERGENCE_ISSUES,
    _NEGATIVE_VARIANCE,
    _OPTION_FAILURES,
    _OPTION_ISSUES,
    KNOWN_FAILURE_ISSUES,
    KNOWN_FAILURES,
    KNOWN_INCONSISTENCIES,
    NON_STRICT,
    _each,
    _with_issue,
)


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
    "surpyval.CoxFrailtyFitter": _BASE,
    "surpyval.ParameterSubstitutionFitter": _BASE,
    "surpyval.life_models.LifeModel": _BASE,
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
    "forecast",
    "from_dict",
    "from_json",
    "logrank",
    "rmst_diff",
    "success_run",
    "weibayes",
    "gray_test",
    "auc_td",
    "brier_score",
    "concordance_index",
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
OUT_OF_SCOPE["surpyval.univariate.nonparametric.canonical_heuristic"] = (
    _FUNCTION + " (reads a plotting-position heuristic's name)"
)
