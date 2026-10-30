"""The Cox family of ``diagnose()`` (#868): its checks on data built to
pass or to fail each of them, the advice each failure calls for, its
refusals, and the options of the Cox fit it supports. The statistics
themselves are checked against R in
``tests/reference/test_cox_diagnostics.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.diagnostics.cox import INFLUENCE_LIMIT


def _crossing(rng, n):
    """A binary covariate whose hazard ratio reverses at t = 0.5: three
    times the baseline before it, a third after."""
    z = rng.binomial(1, 0.5, n).astype(float)
    e = rng.exponential(size=n)
    exposed = np.where(e < 1.5, e / 3, 0.5 + 3 * (e - 1.5))
    return z, np.where(z == 1, exposed, e)


def _fit(df, Z_cols=None, formula=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.CoxPH.fit_from_df(
            df, x_col="x", c_col="c", Z_cols=Z_cols, formula=formula, **kwargs
        )


def _frame(t, cens, **covariates):
    return pd.DataFrame(
        dict(x=np.minimum(t, cens), c=(t > cens).astype(int), **covariates)
    )


@pytest.fixture(scope="module")
def binary_violation():
    rng = np.random.default_rng(11)
    z, t = _crossing(rng, 400)
    w = rng.normal(size=400)
    t = t * np.exp(-0.3 * w)
    return _fit(_frame(t, np.full(400, 3.0), z=z, w=w), ["z", "w"])


@pytest.fixture(scope="module")
def meets_assumptions():
    rng = np.random.default_rng(5)
    n = 300
    z, w = rng.binomial(1, 0.5, n), rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.7 * z + 0.5 * w))
    return _fit(_frame(t, rng.uniform(0, 3, n), z=z, w=w), ["z", "w"])


def _rows(d, check, verdict=None):
    table = d.table[d.table["check"] == check]
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


# -- the checks and their advice ----------------------------------------------
def test_data_that_meet_the_assumptions_pass(meets_assumptions):
    d = meets_assumptions.diagnose()
    assert d.ok, d
    assert not d.advice
    assert set(d.table["check"]) == {
        "proportional_hazards",
        "functional_form",
        "influence",
        "outliers",
        "calibration",
    }
    # z is binary: no functional-form row, and a note says why
    assert set(_rows(d, "functional_form")["covariates"]) == {("w",)}
    assert any("two-valued" in note for note in d.notes)


def test_a_binary_violation_is_advised_to_stratify(binary_violation):
    d = binary_violation.diagnose()
    failed = _rows(d, "proportional_hazards", "fail")
    assert ("z",) in set(failed["covariates"])
    assert ("w",) not in set(failed["covariates"])
    # A covariate fails if either transform fails; the advice is given
    # once.
    stratify = [a for a in d.advice if "Stratify on z" in a]
    assert len(stratify) == 1
    assert "CoxPH.fit(..., strata=" in stratify[0]
    assert 'strata_col="z"' in stratify[0]
    assert stratify[0].features == (
        "CoxPH.fit(strata=)",
        "CoxPH.fit_from_df(strata_col=)",
    )
    assert stratify[0].covariates == ("z",)
    assert stratify[0].check == "proportional_hazards"
    assert not any("fit_tvc" in a for a in d.advice if "of z" in a)
    detail = failed.loc[failed["covariates"] == ("z",), "detail"].iloc[0]
    assert detail == "its coefficient falls with time"


def test_a_continuous_violation_is_advised_a_time_varying_effect():
    rng = np.random.default_rng(3)
    n = 500
    w = rng.normal(size=n)
    e = rng.exponential(size=n)
    # beta(t) = +1 before t = 0.5, -1 after
    rate_early, rate_late = np.exp(w), np.exp(-w)
    t = np.where(
        e < 0.5 * rate_early,
        e / rate_early,
        0.5 + (e - 0.5 * rate_early) / rate_late,
    )
    d = _fit(_frame(t, np.full(n, 4.0), w=w), ["w"]).diagnose()
    assert ("w",) in set(
        _rows(d, "proportional_hazards", "fail")["covariates"]
    )
    (advice,) = (a for a in d.advice if a.check == "proportional_hazards")
    assert "CoxPH.fit_tvc" in advice and "pandas.cut" in advice
    assert "WeibullAFT.fit" in advice and "LogNormalAFT.fit" in advice
    assert "Stratify on" not in advice
    assert advice.features[0] == "CoxPH.fit_tvc"


def test_a_few_levelled_covariate_is_stratified():
    rng = np.random.default_rng(11)
    z, t = _crossing(rng, 600)
    level = rng.integers(0, 3, 600)  # three levels: categorical
    stage = np.where(z == 1, level + 1, 0)  # 0..3, the violation in 1..3
    d = _fit(_frame(t, np.full(600, 3.0), stage=stage), ["stage"]).diagnose()
    assert ("stage",) in set(
        _rows(d, "proportional_hazards", "fail")["covariates"]
    )
    assert any("Stratify on stage" in a for a in d.advice)


def test_a_formula_dummy_is_stratified_on_its_variable():
    rng = np.random.default_rng(11)
    z, t = _crossing(rng, 400)
    df = _frame(t, np.full(400, 3.0), group=np.where(z == 1, "b", "a"))
    d = _fit(df, formula="C(group)").diagnose()
    assert list(d.columns) == ["C(group)[T.b]"]
    assert any("Stratify on group" in a for a in d.advice)
    assert any('strata_col="group"' in a for a in d.advice)


def test_a_non_linear_effect_fails_functional_form():
    rng = np.random.default_rng(2)
    n = 400
    u = rng.uniform(-2, 2, n)
    t = rng.exponential(1 / np.exp(0.8 * u**2))
    d = _fit(_frame(t, rng.uniform(0, 2, n), u=u), ["u"]).diagnose()
    form = _rows(d, "functional_form")
    assert form["verdict"].tolist() == ["fail"]
    assert (
        form["detail"].iloc[0] == "its effect on the log hazard is not linear"
    )
    (advice,) = (a for a in d.advice if a.check == "functional_form")
    assert 'formula="np.log(u) + ..."' in advice and "pandas.cut" in advice
    assert advice.features == ("CoxPH.fit_from_df(formula=)",)
    # ... and the fix the advice names is a formula the fit takes
    df = _frame(t, rng.uniform(0, 2, n), u=u)
    _fit(df, formula="np.square(u)")


def test_an_influential_row_warns():
    rng = np.random.default_rng(4)
    n = 30
    z = rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.8 * z))
    z[0], t[0] = 4.0, 50.0  # far out, and lives far longer than it should
    d = _fit(_frame(t, np.full(n, 60.0), z=z), ["z"]).diagnose()
    row = _rows(d, "influence").iloc[0]
    assert row["verdict"] == "warn" and row["statistic"] > INFLUENCE_LIMIT
    assert pd.isna(row["p"])
    assert row["detail"].startswith("row 0 moves the coefficient")
    (advice,) = (a for a in d.advice if a.check == "influence")
    assert advice.startswith("Row 0 has a large influence")
    # A screen warns, never fails (the row also breaks the tests' rows)
    assert "fail" not in set(_rows(d, "influence")["verdict"])


def test_an_outlier_warns_with_the_bonferroni_cutoff():
    from scipy.stats import norm

    rng = np.random.default_rng(6)
    n = 200
    z = rng.normal(size=n)
    t = rng.exponential(1 / np.exp(z))
    z[0], t[0] = 3.0, 30.0  # high risk, censored very late
    df = _frame(t, np.full(n, 25.0), z=z)
    d = _fit(df, ["z"]).diagnose()
    row = _rows(d, "outliers").iloc[0]
    cutoff = norm.isf(0.05 / (2 * n))
    assert row["verdict"] == "warn" and row["statistic"] > cutoff
    assert row["detail"].startswith("row 0")
    assert f"{cutoff:.3g}" in row["detail"]
    assert any(a.check == "outliers" and "row 0" in a for a in d.advice)


def test_clustered_rows_are_offered_robust_standard_errors():
    i = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9]
    xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 0, 0, 4]
    xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8, 2, 5, 4, 4, 9]
    c = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    Z = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxPH.fit_tvc(i, xl, xr, c, Z)
    d = model.diagnose()
    (advice,) = (a for a in d.advice if a.check is None)
    assert "5 subjects have more than one row" in advice
    assert "summary(robust=True)" in advice
    assert advice.features == ("summary(robust=)", "robust_summary")
    assert not any("no clusters" in note for note in d.notes)


def test_no_clusters_is_noted(meets_assumptions):
    d = meets_assumptions.diagnose()
    assert any("no clusters" in note for note in d.notes)
    assert not any(a.check is None for a in d.advice)


def test_a_joint_failure_alone_has_its_own_advice(
    monkeypatch, meets_assumptions
):
    import surpyval.diagnostics.cox as cox

    def fake(model, transform):
        entry = {"statistic": 1.0, "df": 1, "p_value": 0.3}
        return {
            "per_covariate": [dict(entry), dict(entry)],
            "global": {"statistic": 9.0, "df": 2, "p_value": 0.01},
            "transform": transform,
        }

    monkeypatch.setattr(cox, "zph_score_test", fake)
    d = meets_assumptions.diagnose()
    ph = [a for a in d.advice if a.check == "proportional_hazards"]
    assert len(ph) == 2  # one per transform
    assert all("GLOBAL" in a and "no single covariate" in a for a in ph)
    assert ph[0].covariates == ("z", "w")


def test_a_test_that_cannot_be_computed_warns():
    # Every event at one time: no spread in time to test a trend against
    rng = np.random.default_rng(1)
    n = 30
    z = rng.normal(size=n)
    x = np.where(np.arange(n) < 12, 1.0, 2.0)
    c = (np.arange(n) >= 12).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = sp.CoxPH.fit(x, z[:, None], c=c).diagnose()
    ph = _rows(d, "proportional_hazards")
    assert ph["p"].isna().all() and (ph["verdict"] == "warn").all()
    assert set(ph["detail"]) <= {
        "the test could not be computed on these data",
        "the joint test of every covariate",
    }
    assert d.ok


# -- what the Cox fit supports ------------------------------------------------
@pytest.mark.parametrize(
    "ties", ["efron", "breslow", "exact", "kalbfleisch-prentice"]
)
def test_every_tie_method_with_delayed_entry(ties):
    rng = np.random.default_rng(8)
    n = 150
    z = rng.normal(size=n)
    tl = np.where(rng.uniform(size=n) < 0.3, rng.uniform(0, 0.5, n), 0.0)
    t = tl + np.ceil(rng.exponential(1 / np.exp(0.5 * z)) * 4) / 4
    df = _frame(t, np.full(n, 3.0), z=z)
    df["tl"] = tl
    d = _fit(df, ["z"], tie_method=ties, tl_col="tl").diagnose()
    assert np.isfinite(d.table["statistic"]).all()
    noted = any("Breslow's risk sets" in note for note in d.notes)
    assert noted == (ties not in ("efron", "breslow"))


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_counts_equal_repetition(ties):
    # Principle 5: a row with count n is n identical rows, for every
    # test and screen (the screens' row numbers aside).
    rng = np.random.default_rng(3)
    m = 40
    z, w = rng.normal(size=m), rng.integers(0, 2, m)
    x = np.ceil(rng.exponential(1 / np.exp(0.5 * z + 0.3 * w)) * 5) / 5
    c = (rng.uniform(size=m) < 0.2).astype(int)
    n = rng.integers(1, 4, m)
    Z = np.column_stack([z, w])
    counted = sp.CoxPH.fit(x, Z, c=c, n=n, tie_method=ties).diagnose()
    written = sp.CoxPH.fit(
        np.repeat(x, n),
        np.repeat(Z, n, axis=0),
        c=np.repeat(c, n),
        tie_method=ties,
    ).diagnose()
    columns = ["check", "transform", "covariates", "df", "verdict"]
    pd.testing.assert_frame_equal(
        counted.table[columns], written.table[columns]
    )
    np.testing.assert_allclose(
        counted.table["statistic"], written.table["statistic"], rtol=1e-9
    )


def test_covariate_names_come_from_a_dataframe_Z(meets_assumptions):
    rng = np.random.default_rng(1)
    Z = pd.DataFrame(
        {"dose": rng.normal(size=60), "arm": rng.integers(0, 2, 60)}
    )
    t = rng.exponential(size=60)
    d = sp.CoxPH.fit(t, Z).diagnose()
    assert d.columns == ["dose", "arm"]
    assert ("dose",) in set(d.table["covariates"])
    d = sp.CoxPH.fit(t, Z.values).diagnose()
    assert d.columns == ["coef_0", "coef_1"]


def test_a_stratified_fit_refuses_naming_the_fix():
    rng = np.random.default_rng(1)
    t, z, s = (
        rng.exponential(size=50),
        rng.normal(size=50),
        rng.integers(0, 2, 50),
    )
    model = sp.CoxPH.fit(t, z[:, None], strata=s)
    with pytest.raises(
        NotImplementedError, match="stratified.*without strata="
    ):
        model.diagnose()


def test_a_restored_model_refuses(meets_assumptions):
    restored = sp.from_dict(meets_assumptions.to_dict())
    with pytest.raises(ValueError, match="from_dict"):
        restored.diagnose()


def test_an_aliased_fit_refuses():
    rng = np.random.default_rng(1)
    z = rng.normal(size=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxPH.fit(rng.exponential(size=40), np.column_stack([z, z]))
    with pytest.raises(ValueError, match="aliased"):
        model.diagnose()


def test_a_fit_with_no_finite_maximum_refuses():
    x = np.arange(1.0, 11.0)
    z = (x <= 5).astype(float)  # every early failure has z = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxPH.fit(x, z[:, None])
    assert model.maximum == "no finite maximum"
    with pytest.raises(ValueError, match="finite maximum"):
        model.diagnose()


def test_plot_draws_every_panel(binary_violation):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = binary_violation.diagnose().plot(n_boot=10, random_state=0)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[0].startswith("z: p = ")
    assert "w: functional form" in titles
    assert "overall fit (Cox-Snell)" in titles
    # 2 PH, 1 functional form, 2 influence, outliers, overall fit and
    # calibration at three times
    assert len(titles) == 2 + 1 + 2 + 1 + 1 + 3
    plt.close(fig)
