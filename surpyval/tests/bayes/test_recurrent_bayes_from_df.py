"""``fit_bayes_from_df`` of the recurrent-event fitters: the columns of an
event log, read as ``fit_from_df`` reads them, give the posterior
``fit_bayes`` gives on the same data as arrays (no number changes)."""

import inspect

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
import surpyval.recurrent as rc

PROCESS_FITTERS = [rc.CrowAMSAA, rc.HPP, rc.Duane, rc.CoxLewis]
REGRESSION_FITTERS = [
    rc.ProportionalIntensityHPP,
    rc.ProportionalIntensityNHPP,
]
ALL_FITTERS = [*PROCESS_FITTERS, *REGRESSION_FITTERS, rc.CauseSpecificNHPP]


def _name(fitter):
    return getattr(fitter, "name", None) or getattr(
        fitter, "__name__", type(fitter).__name__
    )


def log(systems=10, seed=7, end=300.0, covariates=1, causes=False):
    """Power-law histories: x, i, c, a covariate matrix and, for the
    cause-specific fit, a cause per event."""
    rng = np.random.default_rng(seed)
    x, i, c, Z, e = [], [], [], [], []
    for k in range(systems):
        z = list(rng.normal(size=covariates))
        m = rng.poisson(6)
        t = np.sort(end * rng.uniform(size=m) ** 1.2)
        x += [*t, end]
        i += [k + 1] * (m + 1)
        c += [0] * m + [1]
        Z += [z] * (m + 1)
        e += [str(rng.choice(["seal", "pump"])) for _ in range(m)] + [None]
    return (
        np.array(x),
        np.array(i),
        np.array(c),
        np.array(Z),
        np.array(e, dtype=object) if causes else None,
    )


def assert_same(a, b, times=(50.0, 150.0, 250.0)):
    """Two Bayesian fits agree to the last digit: the same engine, names,
    credible intervals, and the cumulative intensity at a few ages."""
    assert type(a) is type(b)
    if hasattr(a, "models"):
        assert list(a.models) == list(b.models)
        for cause in a.models:
            assert_same(a.models[cause], b.models[cause], times)
        return
    assert a.method == b.method
    assert a.parameter_names == b.parameter_names
    for name in a.parameter_names:
        np.testing.assert_array_equal(a.param_cri(name), b.param_cri(name))


@pytest.mark.parametrize("fitter", PROCESS_FITTERS, ids=_name)
def test_process_fitters_match_the_arrays(fitter):
    x, i, c, _, _ = log()
    df = pd.DataFrame({"hours": x, "unit": i, "gone": c})
    by_df = fitter.fit_bayes_from_df(
        df, x_col="hours", i_col="unit", c_col="gone"
    )
    by_arrays = fitter.fit_bayes(x, i, c)
    assert_same(by_df, by_arrays)
    np.testing.assert_array_equal(
        by_df.cif([50.0, 150.0]), by_arrays.cif([50.0, 150.0])
    )


def test_priors_options_and_extra_columns_pass_through():
    x, i, c, _, _ = log()
    # a non-default mapping: other names, other column order, columns the
    # fit does not use, and a count column of ones
    df = pd.DataFrame(
        {
            "noise": np.arange(len(x)),
            "n_events": np.ones(len(x), dtype=int),
            "gone": c,
            "unit": i,
            "hours": x,
        }
    )
    priors = {"beta": sp.prior.around(1.2, between=(0.6, 2.0))}
    by_df = rc.CrowAMSAA.fit_bayes_from_df(
        df,
        x_col="hours",
        i_col="unit",
        c_col="gone",
        n_col="n_events",
        priors=priors,
        method="laplace",
        importance=True,
    )
    by_arrays = rc.CrowAMSAA.fit_bayes(
        x,
        i,
        c,
        np.ones(len(x), dtype=int),
        priors=priors,
        method="laplace",
        importance=True,
    )
    assert by_df.method == "laplace"
    assert_same(by_df, by_arrays)


def test_hpp_closed_form_with_a_gamma_prior():
    x, i, c, _, _ = log()
    df = pd.DataFrame({"x": x, "i": i, "c": c})
    prior = {"lambda": sp.Gamma.from_params([2.0, 100.0])}
    by_df = rc.HPP.fit_bayes_from_df(
        df, x_col="x", i_col="i", c_col="c", priors=prior
    )
    assert by_df.method == "conjugate"
    assert_same(by_df, rc.HPP.fit_bayes(x, i, c, priors=prior))


def test_truncation_columns_are_read_like_fit_from_df():
    x, i, c, _, _ = log()
    start = np.where(i % 2 == 0, 10.0, 0.0)
    df = pd.DataFrame({"x": x, "i": i, "c": c, "entry": start})
    x2 = x + start
    df["x"] = x2
    by_df = rc.CrowAMSAA.fit_bayes_from_df(
        df, x_col="x", i_col="i", c_col="c", tl_col="entry"
    )
    assert_same(by_df, rc.CrowAMSAA.fit_bayes(x2, i, c, tl=start))


@pytest.mark.parametrize("fitter", REGRESSION_FITTERS, ids=_name)
def test_regression_fitters_match_the_arrays(fitter):
    x, i, c, Z, _ = log()
    df = pd.DataFrame({"hours": x, "truck": i, "gone": c, "load": Z[:, 0]})
    by_df = fitter.fit_bayes_from_df(
        df, x_col="hours", Z_cols="load", i_col="truck", c_col="gone"
    )
    by_arrays = fitter.fit_bayes(x, Z, i, c)
    # the coefficient is named by its column
    assert by_df.parameter_names[-1] == "load"
    assert by_arrays.parameter_names[-1] == "coef_0"
    for k in range(len(by_df.parameter_names)):
        np.testing.assert_array_equal(
            by_df.param_cri(by_df.parameter_names[k]),
            by_arrays.param_cri(by_arrays.parameter_names[k]),
        )


def test_regression_other_baseline_and_laplace_beyond_three_parameters():
    x, i, c, Z, _ = log(systems=30, covariates=3, seed=3)
    names = ["a", "b", "c"]
    df = pd.DataFrame(Z, columns=names)
    df["hours"], df["truck"], df["gone"] = x, i, c
    by_df = rc.ProportionalIntensityNHPP.fit_bayes_from_df(
        df,
        x_col="hours",
        Z_cols=names,
        i_col="truck",
        c_col="gone",
        baseline=rc.CrowAMSAA,
    )
    by_arrays = rc.ProportionalIntensityNHPP.fit_bayes(
        x, Z, i, c, baseline=rc.CrowAMSAA
    )
    assert by_df.method == by_arrays.method == "laplace"
    np.testing.assert_array_equal(
        by_df.draw_models(3, random_state=1)[0].params,
        by_arrays.draw_models(3, random_state=1)[0].params,
    )


@pytest.mark.parametrize("dist", [rc.CrowAMSAA, rc.HPP])
def test_cause_specific_matches_the_arrays(dist):
    x, i, c, _, e = log(causes=True)
    df = pd.DataFrame({"hours": x, "truck": i, "gone": c, "cause": e})
    priors = (
        {"beta": sp.prior.between(0.4, 2.0)} if dist is rc.CrowAMSAA else {}
    )
    by_df = rc.CauseSpecificNHPP.fit_bayes_from_df(
        df,
        x_col="hours",
        e_col="cause",
        i_col="truck",
        c_col="gone",
        dist=dist,
        priors=priors or None,
    )
    by_arrays = rc.CauseSpecificNHPP.fit_bayes(
        x, i, c, e=e, dist=dist, priors=priors or None
    )
    assert_same(by_df, by_arrays)
    for cause in by_df.models:
        np.testing.assert_array_equal(
            by_df.cif([100.0, 200.0], cause),
            by_arrays.cif([100.0, 200.0], cause),
        )


def test_the_unit_and_terminal_column_are_taken_as_fit_from_df_takes_them():
    x, i, c, _, _ = log()
    df = pd.DataFrame({"x": x, "i": i, "c": c})
    model = rc.CrowAMSAA.fit_bayes_from_df(
        df, x_col="x", i_col="i", c_col="c", unit="hours"
    )
    assert model.unit == "hours"
    flags = np.zeros(len(x))
    model = rc.CrowAMSAA.fit_bayes_from_df(
        df.assign(a=flags), x_col="x", i_col="i", c_col="c", a_col="a"
    )
    assert_same(model, rc.CrowAMSAA.fit_bayes(x, i, c))


@pytest.mark.parametrize("fitter", ALL_FITTERS, ids=_name)
def test_column_arguments_mirror_fit_from_df(fitter):
    """The same column names, in the same order, with the same defaults."""
    plain = inspect.signature(fitter.fit_from_df).parameters
    bayes = inspect.signature(fitter.fit_bayes_from_df).parameters

    def columns(params):
        return [
            (name, p.default)
            for name, p in params.items()
            if name.endswith(("_col", "_cols")) or name == "df"
        ]

    assert columns(bayes) == columns(plain)


def test_refusals_are_those_of_fit_from_df():
    x, i, c, _, _ = log()
    df = pd.DataFrame({"x": x, "i": i, "c": c})
    with pytest.raises(TypeError, match="name the column with 'c_col'"):
        rc.CrowAMSAA.fit_bayes_from_df(df, x_col="x", c="c")
    with pytest.raises(TypeError, match="name the column with 'c_col'"):
        rc.CrowAMSAA.fit_from_df(df, x_col="x", c="c")
    with pytest.raises(ValueError, match="not a column"):
        rc.CrowAMSAA.fit_bayes_from_df(df, x_col="hours")
    with pytest.raises(ValueError, match="DataFrame"):
        rc.CrowAMSAA.fit_bayes_from_df({"x": x}, x_col="x")
    # the cause-specific fit reads its columns itself: a missing cause
    # column is refused as fit_from_df refuses it
    for method in ("fit_from_df", "fit_bayes_from_df"):
        with pytest.raises(KeyError, match="'e'"):
            getattr(rc.CauseSpecificNHPP, method)(df, x_col="x", e_col="e")
    with pytest.raises(TypeError, match="Z_cols"):
        rc.ProportionalIntensityHPP.fit_bayes_from_df(
            df, x_col="x", Z_cols="i", Z=[1]
        )
