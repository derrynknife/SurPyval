"""The competing-risks families of ``diagnose()`` (#874): the
cause-specific Cox checks and the Fine-Gray checks, against R, on data
built to pass or to fail each of them, the advice each failure calls for,
the refusals, the plots and the report's round trip.

The R values are in ``tests/reference/data/r_competing_diagnostics.json``
(``scripts/reference/reference_r_competing_diagnostics.R``): ``cox.zph``
and the score test of an added square on each cause's own
``coxph(Surv(time, cause == k) ~ ...)``, and the same on the weighted
counting-process data whose weighted ``coxph`` fit is the Fine-Gray fit,
with the cif and rank scales given to ``cox.zph`` as functions. The data
are the PBC trial's first rows (transplant and death, few ties) and a
seeded synthetic set with integer times (many ties, censorings tied with
events). The statistics are smooth functions of the coefficients, which
agree with R's to about 1e-9, so they agree to ``rtol=1e-6``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis
from surpyval.diagnostics.competing_risks import (
    fine_gray_added_covariate_test,
    fine_gray_zph_test,
)
from surpyval.tests.reference._data import fixture, values

CR = sp.CompetingRisksProportionalHazards
SOURCE = "r_competing_diagnostics"
TOL = dict(rtol=1e-6, atol=1e-10)


# -- data ---------------------------------------------------------------------
def _reference_frame(name):
    d = fixture(name)
    x = d["years"] if name == "pbc" else d["x"]
    cause = d["cause"].astype(int)
    e = np.where(cause == 0, None, cause).astype(object)
    covs = (
        ["drug", "age", "log_bili", "female"]
        if name == "pbc"
        else ["group", "z"]
    )
    df = pd.DataFrame({c: d[c] for c in covs})
    df["x"], df["e"] = x, e
    return df, covs


def _fit(df, cols, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CR.fit_from_df(df, x_col="x", e_col="e", Z_cols=cols, **kwargs)


def _crossing(rng, z):
    """A latent time whose hazard ratio for ``z = 1`` is 3 before t = 0.5
    and a third after."""
    e = rng.exponential(size=z.size)
    exposed = np.where(e < 1.5, e / 3, 0.5 + 3 * (e - 1.5))
    return np.where(z == 1, exposed, e)


def _mild(rng, z1, z2, k=0.3):
    """A latent time whose hazard is ``exp(k (z1 + z2))`` times the
    baseline before t = 0.5 and ``exp(-k (z1 + z2))`` after."""
    e = rng.exponential(size=z1.size)
    early, late = np.exp(k * (z1 + z2)), np.exp(-k * (z1 + z2))
    return np.where(e < 0.5 * early, e / early, 0.5 + (e - 0.5 * early) / late)


def _frame(t_a, t_b, cens, **covariates):
    first = np.minimum(t_a, t_b)
    e = np.where(cens < first, None, np.where(t_a < t_b, "a", "b"))
    return pd.DataFrame(
        dict(x=np.minimum(first, cens), e=e.astype(object), **covariates)
    )


def _fine_gray_frame(rng, lp, cens, p=0.6, **covariates):
    """Data from a Fine-Gray model (Fine and Gray 1999's simulation):
    cause a's cumulative incidence is ``1 - (1 - p (1 - exp(-t)))^exp(lp)``,
    and a unit that does not fail from a fails from b, at a unit
    exponential time."""
    r = np.exp(lp)
    P = 1 - (1 - p) ** r
    is_a = rng.uniform(size=lp.size) < P
    u = rng.uniform(size=lp.size)
    t_a = -np.log(1 - (1 - (1 - u * P) ** (1 / r)) / p)
    t = np.where(is_a, t_a, rng.exponential(size=lp.size))
    e = np.where(cens < t, None, np.where(is_a, "a", "b"))
    return pd.DataFrame(
        dict(x=np.minimum(t, cens), e=e.astype(object), **covariates)
    )


def _rows(d, check, cause=None, verdict=None):
    t = d.table[d.table["check"] == check]
    if cause is not None:
        t = t[t["cause"] == cause]
    if verdict is not None:
        t = t[t["verdict"] == verdict]
    return t


@pytest.fixture(scope="module")
def crossing():
    """Cause a's hazard ratio for z reverses; cause b is proportional."""
    rng = np.random.default_rng(11)
    n = 500
    z = rng.binomial(1, 0.5, n).astype(float)
    w = rng.normal(size=n)
    t_b = rng.exponential(2 / np.exp(0.5 * w))
    df = _frame(2 * _crossing(rng, z), t_b, np.full(n, 4.0), z=z, w=w)
    return _fit(df, ["z", "w"])


@pytest.fixture(scope="module")
def proportional():
    rng = np.random.default_rng(5)
    n = 500
    z, w = rng.binomial(1, 0.5, n).astype(float), rng.normal(size=n)
    t_a = rng.exponential(1 / np.exp(0.7 * z + 0.5 * w))
    t_b = rng.exponential(2 / np.exp(-0.4 * z))
    return _frame(t_a, t_b, rng.uniform(0, 3, n), z=z, w=w)


@pytest.fixture(scope="module")
def fine_gray_proportional():
    rng = np.random.default_rng(1)
    n = 500
    z = rng.binomial(1, 0.5, n).astype(float)
    w = rng.normal(size=n)
    return _fine_gray_frame(rng, 0.7 * z, rng.uniform(0, 4, n), z=z, w=w)


# -- against R ----------------------------------------------------------------
def _ph_rows(d, cause, transform):
    t = _rows(d, "proportional_hazards", str(cause))
    return t[t["transform"] == transform]


@pytest.mark.parametrize("name", ["pbc", "competing"])
@pytest.mark.parametrize("cause", [1, 2])
def test_cause_specific_matches_cox_zph(name, cause):
    df, covs = _reference_frame(name)
    model = _fit(df, covs)
    ref = values(SOURCE, "cs_{}_cause{}".format(name, cause))
    assert_allclose(model.models[cause].beta, ref["coef"], **TOL)
    d = model.diagnose()
    for transform in ("km", "rank"):
        rows = _ph_rows(d, cause, transform)
        assert [
            r[0] if len(r) == 1 else "GLOBAL" for r in rows.covariates
        ] == (covs + ["GLOBAL"])
        assert_allclose(rows["statistic"], ref[transform + "_chisq"], **TOL)
        assert_allclose(rows["p"], ref[transform + "_p"], **TOL)
        assert list(rows["df"]) == list(ref[transform + "_df"])
    form = _rows(d, "functional_form", str(cause))
    for _, row in form.iterrows():
        (v,) = row["covariates"]
        assert_allclose(row["statistic"], ref["square_" + v], **TOL)
    assert {r[0] for r in form["covariates"]} == {
        k[len("square_") :] for k in ref if k.startswith("square_")
    }


@pytest.mark.parametrize("name", ["pbc", "competing"])
@pytest.mark.parametrize("cause", [1, 2])
def test_fine_gray_matches_cox_zph_on_the_weighted_data(name, cause):
    df, covs = _reference_frame(name)
    model = _fit(df, covs, model="Fine-Gray")
    ref = values(SOURCE, "fg_{}_cause{}".format(name, cause))
    assert_allclose(model.models[cause].beta, ref["coef"], rtol=1e-8)
    d = model.diagnose()
    for transform in ("cif", "rank"):
        rows = _ph_rows(d, cause, transform)
        assert_allclose(rows["statistic"], ref[transform + "_chisq"], **TOL)
        assert_allclose(rows["p"], ref[transform + "_p"], **TOL)
    form = _rows(d, "functional_form", str(cause))
    assert len(form) == sum(k.startswith("square_") for k in ref)
    for _, row in form.iterrows():
        (v,) = row["covariates"]
        assert_allclose(row["statistic"], ref["square_" + v], **TOL)


def test_fine_gray_residuals_match_r():
    df, covs = _reference_frame("pbc")
    model = _fit(df, covs, model="Fine-Gray").models[2]
    ref = values(SOURCE, "fg_pbc_cause2")
    res = fine_gray_zph_test(model, "cif")
    # Tied event times come in either order: compare sorted.
    ours = np.column_stack([res["time"], res["g"], res["scaled_schoenfeld"]])
    theirs = np.column_stack([ref["cif_time"], ref["cif_x"], ref["cif_y"]])
    ours = ours[np.lexsort(ours.T[::-1])]
    theirs = theirs[np.lexsort(theirs.T[::-1])]
    assert_allclose(ours, theirs, **TOL)


def test_fine_gray_functions_on_a_fine_gray_fit():
    df, covs = _reference_frame("pbc")
    e = df["e"].where(df["e"].notna(), None)
    model = sp.FineGray.fit(df["x"], df[covs], e, event=2)
    ref = values(SOURCE, "fg_pbc_cause2")
    for transform in ("cif", "rank"):
        res = fine_gray_zph_test(model, transform)
        stats = [r["statistic"] for r in res["per_covariate"]]
        stats.append(res["global"]["statistic"])
        assert_allclose(stats, ref[transform + "_chisq"], **TOL)
    # The information is the inverse of the model's covariance.
    assert_allclose(
        np.linalg.inv(res["information"]), model.covariance(), rtol=1e-8
    )
    res = fine_gray_added_covariate_test(model, df["age"] ** 2)
    assert_allclose(res["statistic"], ref["square_age"], **TOL)
    with pytest.raises(ValueError, match="transform"):
        fine_gray_zph_test(model, "km")
    with pytest.raises(ValueError, match="one row per row"):
        fine_gray_added_covariate_test(model, np.ones(3))


def test_the_rows_of_a_cause_are_its_own_cox_models(crossing):
    # The cause-specific rows equal CoxPH.fit(x, Z, c_k).diagnose()'s,
    # less its calibration rows (no calibration per cause, #881)
    d = crossing.diagnose()
    for cause in ("a", "b"):
        own = crossing.models[cause].diagnose()
        assert own.family == "Cox proportional hazards"
        own.table = own.table[own.table["check"] != "calibration"]
        own.table = own.table.reset_index(drop=True)
        ours = d.table[d.table["cause"] == cause].reset_index(drop=True)
        assert len(ours) == len(own.table)
        for col in ("check", "test", "statistic", "p", "verdict"):
            assert list(ours[col].fillna(-1)) == list(
                own.table[col].fillna(-1)
            ), col


def test_the_rows_of_a_cause_are_a_cox_fit_by_hand(proportional):
    # CoxPH.fit(x, Z, c_k), c_k censoring every other cause, run directly
    d = _fit(proportional, ["z", "w"]).diagnose()
    x, Z = proportional["x"].values, proportional[["z", "w"]].values
    for cause in ("a", "b"):
        c_k = np.where(proportional["e"] == cause, 0, 1)
        own = sp.CoxPH.fit(x, Z, c=c_k).diagnose()
        ours = d.table[d.table["cause"] == cause]
        for check in ("proportional_hazards", "functional_form"):
            assert_allclose(
                ours.loc[ours["check"] == check, "statistic"],
                own.table.loc[own.table["check"] == check, "statistic"],
                rtol=1e-9,
            )


# -- the cause-specific family ------------------------------------------------
def test_a_crossing_cause_alone_is_flagged_and_stratified(crossing):
    d = crossing.diagnose()
    assert d.family == "Competing risks (cause-specific Cox)"
    assert d.model == "CompetingRisksProportionalHazards"
    assert list(d.table.columns[:3]) == ["check", "cause", "transform"]
    failed = _rows(d, "proportional_hazards", verdict="fail")
    assert set(failed["cause"]) == {"a"}
    assert ("z",) in set(failed["covariates"])
    assert ("w",) not in set(failed["covariates"])
    stratify = [a for a in d.advice if "Stratify" in a]
    assert len(stratify) == 1
    assert "cause 'a'" in stratify[0]
    assert (
        "CoxPH.fit(x, Z, c=np.where(e == 'a', 0, 1), strata=<the z "
        "column>)" in stratify[0]
    )
    assert stratify[0].features == ("CoxPH.fit(strata=)",)
    assert stratify[0].covariates == ("z",)
    # The model-level advice: Fine-Gray where incidence is the question
    (fg,) = [a for a in d.advice if "Fine-Gray" in a]
    assert fg.check is None
    assert fg.features == (
        "CompetingRisksProportionalHazards.fit(model=)",
        "FineGray.fit",
    )
    detail = failed.loc[failed["covariates"] == ("z",), "detail"].iloc[0]
    assert detail == "its coefficient falls with time"


def test_data_that_meet_the_assumptions_pass(proportional):
    d = _fit(proportional, ["z", "w"]).diagnose()
    assert d.ok, d
    assert not d.advice
    assert set(d.table["cause"]) == {"a", "b"}
    assert set(d.table["check"]) == {
        "proportional_hazards",
        "functional_form",
        "influence",
        "outliers",
    }
    # The standing notes
    assert any("Tsiatis 1975" in n for n in d.notes)
    assert any("aetiology" in n for n in d.notes)
    assert any("two-valued" in n for n in d.notes)


def test_a_continuous_violation_is_advised_a_time_varying_effect():
    rng = np.random.default_rng(3)
    n = 500
    w = rng.normal(size=n)
    e = rng.exponential(size=n)
    early, late = np.exp(w), np.exp(-w)
    t_a = np.where(e < 0.5 * early, e / early, 0.5 + (e - 0.5 * early) / late)
    df = _frame(t_a, rng.exponential(3, n), np.full(n, 4.0), w=w)
    d = _fit(df, ["w"]).diagnose()
    assert set(_rows(d, "proportional_hazards", verdict="fail")["cause"]) == {
        "a"
    }
    (advice,) = [a for a in d.advice if a.check == "proportional_hazards"]
    assert "CoxPH.fit_tvc" in advice and "pandas.cut" in advice
    assert "cause 'a'" in advice and "Stratify" not in advice
    assert advice.features == ("CoxPH.fit_tvc", "CoxPH.fit(strata=)")


def test_a_global_only_failure_is_advised():
    rng = np.random.default_rng(3)
    n = 300
    z1 = rng.binomial(1, 0.5, n).astype(float)
    z2 = rng.binomial(1, 0.5, n).astype(float)
    df = _frame(
        _mild(rng, z1, z2),
        rng.exponential(2, n),
        np.full(n, 3.0),
        z1=z1,
        z2=z2,
    )
    # GLOBAL p is 1.5e-4 on both scales, every single covariate's above
    # 5e-4.
    d = _fit(df, ["z1", "z2"]).diagnose(alpha=3e-4)
    failed = _rows(d, "proportional_hazards", verdict="fail")
    assert set(failed["covariates"].map(len)) == {2}
    advice = [a for a in d.advice if "GLOBAL" in a]
    assert len(advice) == 2  # one per scale
    assert all("cause 'a'" in a for a in advice)
    assert advice[0].features == ("CoxPH.fit(strata=)", "CoxPH.fit_tvc")


def test_a_non_linear_effect_fails_functional_form():
    rng = np.random.default_rng(2)
    n = 500
    w = rng.normal(size=n)
    t_a = rng.exponential(1 / np.exp(0.8 * w**2))
    df = _frame(t_a, rng.exponential(2, n), rng.uniform(0, 3, n), w=w)
    d = _fit(df, ["w"]).diagnose()
    failed = _rows(d, "functional_form", verdict="fail")
    assert list(failed["cause"]) == ["a"]
    (advice,) = [a for a in d.advice if a.check == "functional_form"]
    assert "cause 'a'" in advice and 'formula="np.log(w)' in advice
    assert advice.features == (
        "CompetingRisksProportionalHazards.fit_from_df(formula=)",
    )


def test_an_influential_outlying_row_warns():
    rng = np.random.default_rng(3)
    n = 60
    w = rng.normal(size=n)
    t_a = rng.exponential(1 / np.exp(1.0 * w))
    t_b = rng.exponential(2, n)
    # A unit far below the others' risk that fails from a at once
    w[0], t_a[0], t_b[0] = -4.0, 1e-3, 10.0
    d = _fit(_frame(t_a, t_b, np.full(n, 5.0), w=w), ["w"]).diagnose()
    assert list(_rows(d, "influence", "b")["verdict"]) == ["ok"]
    influence = _rows(d, "influence", "a")
    assert list(influence["verdict"]) == ["warn"]
    assert list(_rows(d, "outliers", "a")["verdict"]) == ["warn"]
    (infl,) = [a for a in d.advice if a.check == "influence"]
    assert "Row 0" in infl and "models['a'].compute_residuals" in infl
    (out,) = [a for a in d.advice if a.check == "outliers"]
    assert "cause 'a'" in out and "row 0" in out
    assert out.features == ("compute_residuals(kind=)",)


def test_columns_and_alpha(crossing):
    d = crossing.diagnose(columns=["w"], alpha=0.01)
    assert d.columns == ["w"]
    single = d.table[d.table["covariates"].map(len) == 1]
    assert set(single["covariates"]) == {("w",)}
    assert d.alpha == 0.01
    with pytest.raises(ValueError, match="'columns'"):
        crossing.diagnose(columns=["a: z"])


# -- the Fine-Gray family -----------------------------------------------------
def test_fine_gray_data_pass(fine_gray_proportional):
    for model in (
        _fit(fine_gray_proportional, ["z", "w"], model="Fine-Gray"),
        sp.FineGray.fit_from_df(
            fine_gray_proportional,
            x_col="x",
            e_col="e",
            Z_cols=["z", "w"],
            event="a",
        ),
    ):
        d = model.diagnose()
        assert d.family == "Competing risks (Fine-Gray)"
        assert d.ok, d
        assert not d.advice
        assert set(d.table["check"]) == {
            "proportional_hazards",
            "functional_form",
        }
        assert set(d.table["transform"].dropna()) == {"cif", "rank"}
        assert any("Tsiatis 1975" in n for n in d.notes)
        assert any("robust variance" in n for n in d.notes)
    assert set(d.table["cause"]) == {"a"}
    assert d.model == "FineGrayModel"


def test_a_fine_gray_violation_is_flagged_and_advised():
    rng = np.random.default_rng(11)
    n = 500
    z = rng.binomial(1, 0.5, n).astype(float)
    df = _frame(_crossing(rng, z), rng.exponential(2, n), np.full(n, 4.0), z=z)
    d = _fit(df, ["z"], model="Fine-Gray").diagnose()
    failed = _rows(d, "proportional_hazards", verdict="fail")
    assert set(failed["cause"]) == {"a"}
    (advice,) = [a for a in d.advice if a.check == "proportional_hazards"]
    assert "cause 'a'" in advice
    assert 'model="Cox"' in advice and "gray_test" in advice
    assert advice.features == (
        "CompetingRisksProportionalHazards.fit(model=)",
        "CompetingRisks.fit",
        "gray_test",
    )
    assert "subdistribution coefficient falls" in failed["detail"].iloc[0]


def test_a_fine_gray_global_only_failure_is_advised():
    rng = np.random.default_rng(5)
    n = 300
    z1 = rng.binomial(1, 0.5, n).astype(float)
    z2 = rng.binomial(1, 0.5, n).astype(float)
    df = _frame(
        _mild(rng, z1, z2),
        rng.exponential(2, n),
        np.full(n, 3.0),
        z1=z1,
        z2=z2,
    )
    # GLOBAL p about 1.2e-4 on both scales, each covariate's above 1.3e-3
    d = _fit(df, ["z1", "z2"], model="Fine-Gray").diagnose(alpha=5e-4)
    failed = _rows(d, "proportional_hazards", verdict="fail")
    assert set(failed["covariates"].map(len)) == {2}
    advice = [a for a in d.advice if "GLOBAL" in a]
    assert len(advice) == 2  # one per scale
    assert all("cause 'a'" in a for a in advice)
    assert advice[0].features == (
        "CompetingRisksProportionalHazards.fit(model=)",
    )


def test_a_fine_gray_non_linear_effect_fails_functional_form():
    rng = np.random.default_rng(3)
    n = 500
    w = rng.normal(size=n)
    df = _fine_gray_frame(rng, 0.8 * w**2, rng.uniform(0, 4, n), w=w)
    d = _fit(df, ["w"], model="Fine-Gray").diagnose()
    assert "a" in set(_rows(d, "functional_form", verdict="fail")["cause"])
    advice = [a for a in d.advice if a.check == "functional_form"]
    assert any("cause 'a'" in a for a in advice)
    assert 'model="Fine-Gray", formula="np.log(w)' in advice[0]


# -- refusals -----------------------------------------------------------------
def test_restored_models_refuse(crossing, fine_gray_proportional):
    restored = sp.from_dict(crossing.to_dict())
    assert restored.models is None
    with pytest.raises(ValueError, match="from_dict"):
        restored.diagnose()
    fg = _fit(fine_gray_proportional, ["z"], model="Fine-Gray")
    with pytest.raises(ValueError, match="from_dict"):
        sp.from_dict(fg.to_dict()).diagnose()
    one = fg.models["a"]
    assert one._fit_data is not None
    with pytest.raises(ValueError, match="Fine-Gray model restored"):
        type(one).from_dict(one.to_dict()).diagnose()


def test_parametric_competing_risks_refuses_naming_the_issue():
    rng = np.random.default_rng(0)
    x = rng.weibull(1.5, 100)
    e = rng.choice(["a", "b"], 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.ParametricCompetingRisks.fit(x, e)
    with pytest.raises(NotImplementedError, match="#874") as info:
        model.diagnose()
    assert "#875" in str(info.value)


def test_an_aliased_coefficient_refuses():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    df = _frame(
        rng.exponential(size=n), rng.exponential(size=n), np.full(n, 3.0)
    )
    df["z"], df["z2"] = z, 2 * z
    with pytest.raises(ValueError, match="Cause 'a': .*aliased"):
        _fit(df, ["z", "z2"]).diagnose()
    with pytest.raises(ValueError, match="Cause 'a': .*aliased"):
        _fit(df, ["z", "z2"], model="Fine-Gray").diagnose()


def test_no_finite_maximum_refuses():
    rng = np.random.default_rng(0)
    n = 100
    t_a, t_b = rng.exponential(size=n), rng.exponential(size=n)
    df = _frame(t_a, t_b, np.full(n, 5.0))
    df["z"] = (df["e"] == "a").astype(float)  # separates cause a's events
    df["w"] = rng.normal(size=n)
    for model in ("Cox", "Fine-Gray"):
        fitted = _fit(df, ["z", "w"], model=model)
        assert fitted.maximum == "no finite maximum"
        with pytest.raises(ValueError, match="Cause 'a': .*finite maximum"):
            fitted.diagnose()


# -- the report ---------------------------------------------------------------
def test_plot_draws_by_cause(crossing, fine_gray_proportional):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = crossing.diagnose()
    fig = d.plot("proportional_hazards", n_boot=10, random_state=1)
    titles = [ax.get_title() for ax in fig.axes]
    assert len(titles) == 4  # two causes, two covariates
    assert titles[0].startswith("cause a: z: p = ")
    assert titles[2].startswith("cause b: z")
    plt.close(fig)
    fig = d.plot(n_boot=0)
    assert any("overall fit" in ax.get_title() for ax in fig.axes)
    plt.close(fig)
    d = _fit(fine_gray_proportional, ["z", "w"], model="Fine-Gray").diagnose()
    fig = d.plot(n_boot=10, random_state=2)
    titles = [ax.get_title() for ax in fig.axes]
    assert "cause a: z: p = " in titles[0] and titles[0].endswith("(cif)")
    assert "cause a: w: functional form" in titles
    assert titles[-1] == "cause b: w: functional form"
    plt.close(fig)
    with pytest.raises(ValueError, match="'which'"):
        d.plot("influence")


def test_to_dict_round_trips(crossing, fine_gray_proportional):
    for d in (
        crossing.diagnose(),
        _fit(fine_gray_proportional, ["z", "w"], model="Fine-Gray").diagnose(),
    ):
        text = json.dumps(d.to_dict(), allow_nan=False)
        back = Diagnosis.from_dict(json.loads(text))
        assert back.to_dict() == d.to_dict()
        assert list(back.table.columns) == list(d.table.columns)
        assert all(isinstance(r["cause"], str) for r in d.to_dict()["table"])


def test_raise_if_failed_names_the_cause(crossing):
    with pytest.raises(sp.diagnostics.DiagnosticsFailed) as info:
        crossing.diagnose().raise_if_failed()
    assert "cause a" in str(info.value)


def test_the_cause_models_are_kept(crossing):
    for cause, row in crossing.event_idx_map.items():
        assert_allclose(crossing.models[cause].beta, crossing.betas[row])
    import pickle

    again = pickle.loads(pickle.dumps(crossing))
    assert set(again.models) == {"a", "b"}
