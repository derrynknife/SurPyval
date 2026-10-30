"""The container and dtype of the inputs do not change a fit (principle 4's
neighbour: what matters is the numbers, not how they arrive).

Each registered case is refitted to its own fixture passed another way,
and every prediction compared with the fit of the same numbers as float64
NumPy arrays:

- **float32**: the float-valued arrays (times, covariates, readings) as
  ``np.float32``, against the fit of ``v.astype(np.float32).astype(float)``
  -- the same numbers in float64, since float32 rounding legitimately
  changes the data;
- **integer**: the times rounded to integers (``round(10 x)``) as
  ``int64``, against the fit of the same integers as float64 (a case whose
  rounded data the fitter refuses is skipped, with the reason);
- **pandas**: each 1-D per-row array as a ``pd.Series`` and the
  covariate matrix as a ``pd.DataFrame``, each with a *different*
  shuffled index, against the array fit: the library reads them by
  position. A 2-D array of times (intervals, a copula's pairs) stays an
  array: as an unnamed DataFrame it is refused on purpose, with a message
  pointing to ``fit_from_df``;
- **lists**: each per-row array as a Python list (a 2-D one as a list of
  lists; a copula's pairs as one list per series, the form its data
  documents).

And for the queries: a float32 query gives what the same numbers in
float64 give, and a list query what the array gives, for every function
of every case.

The comparisons are to ``rtol=1e-12``: the library casts its inputs to
float64 on entry, so the two fits see the same numbers, and the seeded
fits (the trees, the forest) draw the same stream. Which cases each
variant applies to:

- every case with a per-row fixture (``Case.rows``) gets the pandas and
  list variants: the univariate parametric, non-parametric and Bayesian
  cases, the regressions (parametric PH/AFT/PO/AH and accelerated life,
  frailty, Cox, additive hazards, Aalen, Buckley-James, proportional odds,
  distributional, Royston-Parmar, the trees and forest), competing risks,
  recurrent events, degradation and the copulas;
- float32 applies where the fixture has float-valued arrays (not the
  discrete distributions, whose times are integers);
- the integer variant applies to the cases with float times in the
  univariate, regression and competing-risks interfaces; not to recurrent
  events (rounding puts several events of a unit at one time), degradation
  (the readings, not the times, carry the model) nor the copulas (the
  margins are fitted on the unit square's scale);
- the cases with no fixture (``Hypoexponential``, ``NeverOccurs``,
  ``InstantlyOccurs``: models given their parameters) get only the query
  variants.

The joint model's ``markers`` (readings by age, a list of dicts per row)
stay as they are: they are not an array.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval.tests.conformance.leaks import quiet
from surpyval.tests.conformance.registry import (
    BIVARIATE,
    CASES,
    COUNTING,
    COUNTING_CAUSES,
    COUNTING_REGRESSION,
    call,
    calls,
    cases_for,
    fitted,
    predictions,
    query,
    refit,
)

RTOL = 1e-12

# Interfaces the integer variant does not apply to (see the docstring).
_NO_INTEGER_INTERFACES = (
    COUNTING,
    COUNTING_REGRESSION,
    COUNTING_CAUSES,
    BIVARIATE,
)
_DEGRADATION = "surpyval.degradation."


def _params(variant, where=lambda case: True):
    return [
        pytest.param(
            case,
            id=case.name,
            marks=[pytest.mark.slow] if case.is_slow("row_order") else [],
        )
        for case in CASES
        if case.rows and where(case)
    ]


def _arrays(case, data):
    """The per-row entries of ``data`` that are arrays."""
    return [
        key
        for key in case.rows
        if key in data and isinstance(data[key], np.ndarray)
    ]


def _float_keys(case, data):
    return [
        key
        for key in _arrays(case, data)
        if np.issubdtype(data[key].dtype, np.floating)
    ]


def _time_keys(case, data):
    return [key for key in case.times if key in _float_keys(case, data)]


def _fit_or_skip(case, data, what):
    try:
        return refit(case, data)
    except Exception as error:  # noqa: BLE001 - the reason is reported
        pytest.skip(
            f"{case.name}: the fixture {what} is refused "
            f"({type(error).__name__}: {error})"
        )


def _compare(case, got, ref, variant):
    assert got.keys() == ref.keys()
    for key in ref:
        np.testing.assert_allclose(
            got[key],
            ref[key],
            rtol=RTOL,
            atol=0,
            err_msg=f"{case.name}: {key}, {variant}",
        )


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def _has_floats(case):
    return bool(_float_keys(case, case.data()))


def _takes_integers(case):
    return (
        case.interface not in _NO_INTEGER_INTERFACES
        and not case.model_class.startswith(_DEGRADATION)
        and bool(_time_keys(case, case.data()))
    )


@pytest.mark.parametrize("case", _params("float32", _has_floats))
def test_float32_inputs(case):
    data = case.data()
    keys = _float_keys(case, data)
    single = {**data, **{k: data[k].astype(np.float32) for k in keys}}
    same = {k: single[k].astype(np.float64) for k in keys}
    ref = _fit_or_skip(case, {**data, **same}, "in float32 precision")
    got = refit(case, single)
    _compare(case, predictions(case, got), predictions(case, ref), "float32")


@pytest.mark.parametrize("case", _params("integer", _takes_integers))
def test_integer_times(case):
    data = case.data()
    keys = _time_keys(case, data)
    ints = {k: np.round(data[k] * 10).astype(np.int64) for k in keys}
    floats = {k: v.astype(np.float64) for k, v in ints.items()}
    ref = _fit_or_skip(case, {**data, **floats}, "rounded to integers")
    got = refit(case, {**data, **ints})
    _compare(case, predictions(case, got), predictions(case, ref), "int64")


def _pandas(case, key, value, index):
    if value.ndim == 1:
        return pd.Series(value, index=index)
    if key == case.covariates:
        return pd.DataFrame(value, index=index)
    # A 2-D time array (intervals, a copula's pairs) as a DataFrame is
    # refused on purpose: its columns must be named (fit_from_df).
    return value


@pytest.mark.parametrize("case", _params("pandas"))
def test_pandas_inputs_with_unrelated_indexes(case):
    data = case.data()
    rng = np.random.default_rng(20261030)
    frames = {}
    for key in _arrays(case, data):
        size = len(data[key])
        # A different shuffled index for each argument, offset so that no
        # two share labels in the same places.
        index = rng.permutation(size) + 1000 * (len(frames) + 1)
        frames[key] = _pandas(case, key, data[key], index)
    got = refit(case, {**data, **frames})
    _compare(
        case, predictions(case, got), predictions(case, fitted(case)), "pd"
    )


@pytest.mark.parametrize("case", _params("lists"))
def test_list_inputs(case):
    data = case.data()
    lists = {key: data[key].tolist() for key in _arrays(case, data)}
    if case.interface == BIVARIATE:
        # A copula reads a list as one sequence per series (documented in
        # MultivariateSurpyvalData), so the (m, 2) pairs go as two lists.
        lists["x"] = data["x"].T.tolist()
    got = refit(case, {**data, **lists})
    _compare(
        case,
        predictions(case, got),
        predictions(case, fitted(case)),
        "list",
    )


# ---------------------------------------------------------------------------
# The query
# ---------------------------------------------------------------------------
# A float32 query was computed in single precision by the parametric
# distributions on the plain path (no offset): ``_query_array`` in
# univariate/parametric/parametric.py kept the query's dtype, and under
# NumPy 2's promotion rules the arithmetic with Python floats stayed in
# float32 (Gamma, LogNormal, LogLogistic, ExpoWeibull, Poisson,
# NegativeBinomial and DiscreteWeibull, with their lfp and zi forms,
# differed from the float64 query by about 1e-8 relative). It now
# converts a float32 query to float64 first.


def _query_params():
    return [
        pytest.param(*ps.values, id=ps.id, marks=ps.marks)
        for ps in cases_for("query_order")
    ]


@pytest.mark.parametrize("case", _query_params())
def test_query_containers(case):
    """A float32 query gives exactly what the same numbers in float64
    give, and a list query what the array gives."""
    model = fitted(case)
    for fname, event in calls(case):
        x = np.asarray(query(case, fname))
        Z = None if case.Z is None or fname == "qf" else case.Z
        single = x.astype(np.float32)
        with quiet():
            got = call(case, model, fname, single, Z, event)
            ref = call(case, model, fname, single.astype(np.float64), Z, event)
        np.testing.assert_array_equal(
            np.asarray(got, float),
            np.asarray(ref, float),
            err_msg=f"{case.name}.{fname}[{event}], float32 query",
        )
        with quiet():
            got = call(case, model, fname, x.tolist(), Z, event)
            ref = call(case, model, fname, x, Z, event)
        np.testing.assert_array_equal(
            np.asarray(got, float),
            np.asarray(ref, float),
            err_msg=f"{case.name}.{fname}[{event}], list query",
        )


def test_the_families_are_covered():
    names = {p.id for p in _params("lists")}
    for name in (
        "Weibull",
        "KaplanMeier",
        "WeibullPH",
        "CoxPH",
        "AalenAdditive",
        "WeibullDistributional",
        "RoystonParmarPH",
        "CompetingRisks[Kaplan-Meier]",
        "FineGray",
        "Weibull[bayes]",
    ):
        assert name in names, name
    ints = {p.id for p in _params("integer", _takes_integers)}
    assert {"Weibull", "CoxPH", "FineGray"} <= ints
