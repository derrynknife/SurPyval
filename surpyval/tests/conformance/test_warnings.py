"""Warnings: deliberate, and once per problem (#379, principle 22).

``conftest.py`` fails any conformance test during which a raw numerical
warning leaks out of the package (``leaks.py`` says what counts). This
module

- checks that the leak detector tells a leak from a deliberate warning
  and from the tests' own arithmetic;
- pins the values of the expressions whose harmless warning was
  silenced in place (the inf or NaN they give is the right answer, and
  they give it without a warning);
- reproduces each known leak, and each wrong result found behind one,
  as a strict xfail;
- checks, for every registered model, that a fit and each prediction
  give each deliberate warning at most once (``warn_once``).
"""

import os
import warnings
from collections import Counter

import numpy as np
import pytest
from scipy.stats import lognorm

import surpyval as sp
from surpyval.tests.conformance import leaks
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    call,
    calls,
    cases_for,
    query,
)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------
def _run_as_package_code(source, **names):
    """Run ``source`` as if it were a line of a module in the package."""
    path = os.path.join(leaks.PACKAGE, "_leak_probe.py")
    exec(
        compile(source, path, "exec"),
        {"np": np, "warnings": warnings, **names},
    )


def _leaks_of(source, **names):
    # record=True: what the detector passes on stays here, rather than
    # reaching conftest.py's check of this very test.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with leaks.watch() as found:
            _run_as_package_code(source, **names)
    return found


def test_numpy_warning_in_package_code_is_a_leak():
    (leak,) = _leaks_of("np.log(np.zeros(1))")
    assert leak.message == "divide by zero encountered in log"
    assert leak.package_frame == "_leak_probe.py:<module>"


def test_warning_from_third_party_code_called_by_the_package_is_a_leak():
    # numpy's Python-level nanmean warns from inside numpy.
    (leak,) = _leaks_of("np.nanmean(np.array([np.nan]))")
    assert leak.message == "Mean of empty slice"
    assert "numpy" in leak.raised_at


def test_deliberate_warning_is_not_a_leak():
    source = "warnings.warn('the fit did not converge', RuntimeWarning)"
    assert _leaks_of(source) == []
    with warnings.catch_warnings(record=True):
        with leaks.deliberate() as found:
            _run_as_package_code(source + "\n" + source)
    assert found == [(RuntimeWarning, "the fit did not converge")] * 2


def test_the_tests_own_warnings_are_not_leaks():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with leaks.watch() as found:
            np.log(np.zeros(1))
    assert found == []


def test_quiet_drops_deliberate_warnings_and_passes_leaks_on():
    source = "warnings.warn('deliberate')\nnp.log(np.zeros(1))"
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with leaks.watch() as found:
            with leaks.quiet():
                _run_as_package_code(source)
    assert [leak.message for leak in found] == [
        "divide by zero encountered in log"
    ]
    assert [str(w.message) for w in shown] == [
        "divide by zero encountered in log"
    ]


# ---------------------------------------------------------------------------
# Silenced in place: the values are right and unchanged
# ---------------------------------------------------------------------------
def _no_leak(f, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with leaks.watch() as found:
            out = f(*args)
    assert found == [], leaks.report(found)
    return np.asarray(out, dtype=float)


def test_kaplan_meier_Hf_and_hf_past_zero_survival():
    # The estimate reaches zero at 3: H is inf from there, and the hazard
    # of the step containing x (hf, forward filled) is the infinite jump.
    # The values are those of the code before the silencing.
    model = sp.KaplanMeier.fit([1.0, 2.0, 3.0])
    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    Hf = _no_leak(model.Hf, x)
    hf = _no_leak(model.hf, x)
    log = np.log
    np.testing.assert_allclose(
        Hf[:3], [0.0, log(3 / 2), log(3)], rtol=1e-12, atol=0
    )
    assert np.all(np.isposinf(Hf[3:]))
    np.testing.assert_allclose(
        hf[:3], [log(3 / 2), log(3 / 2), log(2)], rtol=1e-12
    )
    assert np.all(np.isposinf(hf[3:]))


def test_binomial_Hf_is_inf_from_n_on():
    model = sp.Binomial.from_params([5, 0.5])
    Hf = _no_leak(model.Hf, np.arange(7.0))
    sf = 1 - np.cumsum([1, 5, 10, 10, 5, 1]) / 32
    np.testing.assert_allclose(Hf[:5], -np.log(sf[:5]), rtol=1e-12)
    assert np.all(np.isposinf(Hf[5:]))


def test_weibull_df_and_hf_at_zero_below_shape_one():
    # beta < 1: density and hazard are unbounded at the origin.
    x = np.array([0.0, 1.0])
    df = _no_leak(sp.Weibull.df, x, 2.0, 0.5)
    hf = _no_leak(sp.Weibull.hf, x, 2.0, 0.5)
    assert np.isposinf(df[0]) and np.isposinf(hf[0])
    h1 = 0.25 * 0.5**-0.5
    np.testing.assert_allclose(hf[1], h1, rtol=1e-12)
    np.testing.assert_allclose(df[1], h1 * np.exp(-(0.5**0.5)), rtol=1e-12)
    # beta = 1 and beta > 1 are finite at 0, as before.
    np.testing.assert_allclose(
        _no_leak(sp.Weibull.df, np.zeros(2), 2.0, np.array([1.0, 3.0])),
        [0.5, 0.0],
    )


def test_lognormal_sf_at_zero():
    sf = _no_leak(sp.LogNormal.sf, np.array([0.0, 1.0]), 0.0, 1.0)
    np.testing.assert_array_equal(sf, [1.0, 0.5])


# ---------------------------------------------------------------------------
# Known leaks and the results behind them (strict xfails)
# ---------------------------------------------------------------------------
def test_every_known_leak_has_a_reason():
    for key, reason in leaks.KNOWN_LEAKS.items():
        assert reason.strip(), key


def test_logistic_sf_far_below_the_location():
    # #410: sf was e / (1 + e) with e = exp(-(x - mu) / sigma), inf / inf
    # = NaN with a raw warning once (mu - x) / sigma > 709. It is now the
    # logistic function of -(x - mu) / sigma.
    sf = _no_leak(sp.Logistic.sf, np.array([0.0, 1000.0]), 1000.0, 1.0)
    np.testing.assert_allclose(sf, [1.0, 0.5], rtol=1e-12)


def test_wald_bound_at_a_boundary_estimate():
    # #411: q = 2.7e-16 on its bound, with variances -0.031 and -10.6 on
    # the inverse Hessian's diagonal, gave [nan, nan] with numpy's raw
    # sqrt warning. It is now nan with a warning saying why (a deliberate
    # warning, not a leak).
    # A fresh fit, not the shared cached one: another test's param_cb on
    # that object can leave its covariance computed, and the leak would
    # then not recur here.
    case = CASE_BY_NAME["GeneralizedRenewal"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = case.fit(case.data())
    for name in ("alpha", "q"):
        _no_leak(model.param_cb, name)


def test_kaplan_meier_df_where_survival_reaches_zero():
    # #408: df was hf * exp(-Hf), inf * 0 = NaN with a raw warning, at and
    # after the time the estimate reaches zero. It is now the drop in sf
    # over the step hf takes: 1/3 for the step to zero at 3, carried past
    # it as hf carries the infinite jump.
    model = sp.KaplanMeier.fit([1.0, 2.0, 3.0])
    x = np.array([2.5, 3.5, 4.5])
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with leaks.watch() as found:
            df = model.df(x)
    assert found == [], leaks.report(found)
    assert np.all(np.isfinite(df)) and np.all((0 <= df) & (df <= 1)), df
    np.testing.assert_allclose(df, [1 / 3, 1 / 3, 1 / 3], rtol=1e-12)


# Separated data (one event, at the largest covariate values) with an
# intercept column, which the partial likelihood cannot identify; found
# by properties/test_regression.py::test_rows_are_independent (#409).
_COX_CONSTANT = dict(
    x=np.array([6.5, 12.0, 2.0, 13.0]),
    Z=np.array(
        [[-2.0, 0.5, 1.0], [1.0, 1.5, 1.0], [2.0, 2.0, 1.0], [2.0, 1.0, 1.0]]
    ),
    c=np.array([0, 0, 0, 1]),
    n=np.array([3, 2, 2, 3]),
)


def test_cox_constant_column_on_separated_data():
    # #409: the constant column's coefficient ran off to 3.1e14, so
    # exp(beta'Z) overflowed (about ten raw numpy warnings) and every
    # prediction was NaN. The column has no coefficient in a Cox model:
    # it is aliased (#476), with one warning naming it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with leaks.watch() as found:
            model = sp.CoxPH.fit(**_COX_CONSTANT, center=True)
    assert found == [], leaks.report(found)
    # The other two columns separate the data: that is the second problem,
    # and the second warning.
    assert [str(w.message)[:28] for w in caught] == [
        "Covariate column(s) 2 of Z c",
        "No finite maximum: the parti",
    ]
    assert np.isnan(model.beta[2])
    # Without it the data are still separated: the one warning is the
    # monotone-likelihood one, and the predictions are finite. (The
    # coefficients run off far enough that the baseline at Z = 0
    # underflows, which the default refuses, #463; at the covariate means
    # it is representable.)
    data = dict(_COX_CONSTANT, Z=_COX_CONSTANT["Z"][:, :2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with leaks.watch() as found:
            model = sp.CoxPH.fit(**data, center=True)
            sf = model.sf(np.array([5.0]), np.array([[0.5, 1.5]]))
    assert found == [], leaks.report(found)
    assert [str(w.message)[:28] for w in caught] == [
        "No finite maximum: the parti"
    ]
    assert np.all(np.isfinite(sf)), sf


def test_truncated_log_likelihood_where_the_truncation_point_underflows():
    # #412: the log-likelihood of left-truncated data was log f(x) -
    # log(1 - F(tl)), so once F(tl) rounded to 1 it was +inf (or NaN,
    # inf - inf) with a raw divide warning: LogNormal, x = [2, 3], tl = 1,
    # mu = -5, sigma = 0.6 gave +inf, where it is exactly -23.73.
    x, tl = np.array([2.0, 3.0]), np.ones(2)
    data = sp.SurpyvalData(
        x=x,
        c=np.zeros(2, dtype=int),
        n=np.ones(2, dtype=int),
        tl=tl,
        tr=np.full(2, np.inf),
    )
    exact = lognorm(0.6, scale=np.exp(-5.0))
    expected = np.sum(exact.logpdf(x) - exact.logsf(tl))  # -23.73
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with leaks.watch() as found:
            ll = sp.LogNormal._log_likelihood(data, -5.0, 0.6, 0.0, 0.0, 1.0)
    assert found == [], leaks.report(found)
    np.testing.assert_allclose(float(ll), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# One warning per problem
# ---------------------------------------------------------------------------
def _assert_once(found, what):
    repeated = {
        f"{cat.__name__}: {msg}": k
        for (cat, msg), k in Counter(found).items()
        if k > 1
    }
    assert not repeated, f"{what} repeats a warning: {repeated}"


def _missing_covariate(case, data):
    out = dict(data)
    Z = np.array(data[case.covariates], dtype=float)
    Z[len(Z) // 2, 0] = np.nan
    out[case.covariates] = Z
    return out


@pytest.mark.parametrize("case", cases_for("warn_once"))
def test_each_deliberate_warning_once_per_call(case):
    data = case.data()
    with leaks.deliberate() as found:
        model = case.fit(data)
    _assert_once(found, "the fit")
    if case.covariates and case.drops_missing_covariate:
        # A dropped row warns ("Dropped 1 of n rows"): once, not once per
        # layer the data pass through.
        with leaks.deliberate() as found:
            case.fit(_missing_covariate(case, data))
        _assert_once(found, "the fit with a missing covariate")
    for fname, event in calls(case):
        with leaks.deliberate() as found:
            call(case, model, fname, query(case, fname), None, event)
        _assert_once(found, fname if event is None else f"{fname}[{event}]")


def test_the_once_check_sees_the_dropped_rows_warning():
    # The recorder behind warn_once does see the package's warnings.
    (case,) = [p.values[0] for p in cases_for("warn_once") if p.id == "CoxPH"]
    with leaks.deliberate() as found:
        case.fit(_missing_covariate(case, case.data()))
    assert [m for _, m in found if m.startswith("Dropped")], found
