"""The accelerated life family of ``diagnose()`` (#871): its two
likelihood-ratio tests against R, its checks on data built to pass or to
fail each of them, the advice each failure calls for, its refusals and
its plots.

The reference values are R's ``survival::survreg`` on Nelson's (1990)
insulating-fluid data, from ``scripts/reference/reference_r_accelerated_
life.R`` (R 4.3.3, survival 3.5.8), pasted below: the log-likelihoods of
each level fitted alone, of the common-shape model with a life at each
level (``~ factor(V)``) and of the life models survreg writes as
log-linear models (Power ``~ log(V)``, Exponential ``~ I(1/V)``, Eyring
and InverseEyring with an offset of ``-log(V)`` and ``log(V)``,
GeneralLogLinear ``~ V``), and the two tests built from them. The
log-likelihoods agree to the precision of the fits, about 1e-6.
"""

import copy
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import life_models as lm
from surpyval.diagnostics import Diagnosis
from surpyval.diagnostics.accelerated_life import DELTA_AIC

#: Time to breakdown (minutes) of an insulating fluid at seven voltages
#: (kV): Nelson (1990), Accelerated Testing, chapter 3. The Weibull and
#: power-law fit gives Nelson's shape 0.777 and exponent -17.7.
FLUID = {
    26: [5.79, 1579.52, 2323.70],
    28: [68.85, 108.29, 110.29, 426.07, 1067.60],
    30: [7.74, 17.05, 20.46, 21.02, 22.66, 43.40, 47.30, 139.07, 144.12]
    + [175.88, 194.90],
    32: [0.27, 0.40, 0.69, 0.79, 2.75, 3.91, 9.88, 13.95, 15.93, 27.80]
    + [53.24, 82.85, 89.29, 100.58, 215.10],
    34: [0.19, 0.78, 0.96, 1.31, 2.78, 3.16, 4.15, 4.67, 4.85, 6.50, 7.35]
    + [8.01, 8.27, 12.06, 31.75, 32.52, 33.91, 36.71, 72.89],
    36: [0.35, 0.59, 0.96, 0.99, 1.69, 1.97, 2.07, 2.58, 2.71, 2.90, 3.67]
    + [3.99, 5.35, 13.77, 25.50],
    38: [0.09, 0.39, 0.47, 0.73, 0.74, 1.13, 1.40, 2.38],
}
X = np.concatenate([np.array(v) for v in FLUID.values()])
V = np.concatenate([[float(k)] * len(v) for k, v in FLUID.items()])

#: R's values (scripts/reference/reference_r_accelerated_life.R)
R: dict[str, dict] = {
    "weibull_complete": {
        "loglik": {
            "Power": -300.817420918309,
            "Exponential": -301.441035333287,
            "Eyring": -301.396316646099,
            "InverseEyring": -301.486816635028,
            "GeneralLogLinear": -300.535924813629,
        },
        "shape": (8.79399689064144, 6, 0.185499296570397),
        "lof": (2.33905340891386, 5, 0.800514313287846),
    },
    "weibull_censored": {
        "shape": (8.41073756261733, 5, 0.135004835452272),
        "lof": (2.4450568743066, 4, 0.654500484279483),
    },
    "lognormal_complete": {
        "loglik": {
            "Power": -303.601886152819,
            "Exponential": -303.78289705497,
            "Eyring": -303.764069472245,
            "InverseEyring": -303.802729039181,
            "GeneralLogLinear": -303.689362804071,
        },
        "shape": (16.791938003206, 6, 0.0100791003969729),
        "lof": (2.54295011327497, 5, 0.770013262379025),
    },
    "exponential_complete": {
        "loglik": {
            "Power": -305.537270054426,
            "Exponential": -306.726422905734,
            "Eyring": -306.643450735608,
            "InverseEyring": -306.811181414002,
            "GeneralLogLinear": -304.891220716566,
        },
        "lof": (4.87862091229931, 5, 0.430872492031304),
    },
}
CASES = {
    "weibull_complete": (sp.Weibull, None),
    "weibull_censored": (sp.Weibull, 500.0),
    "lognormal_complete": (sp.LogNormal, None),
    "exponential_complete": (sp.Exponential, None),
}


def _fit(dist, x, Z, life_model=lm.Power, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.AcceleratedLife(dist, life_model).fit(x, Z, **kwargs)


def _fluid(case):
    dist, censor = CASES[case]
    if censor is None:
        return _fit(dist, X, V)
    return _fit(dist, np.minimum(X, censor), V, c=(X > censor).astype(int))


def _row(d, check, life_model=None):
    table = d.table[d.table["check"] == check]
    if life_model is not None:
        table = table[table["life_model"] == life_model]
    assert len(table) == 1, table
    return table.iloc[0]


def _test_values(row):
    return row["statistic"], int(row["df"]), row["p"]


# -- against R ----------------------------------------------------------------
@pytest.fixture(scope="module")
def reports():
    return {case: _fluid(case).diagnose() for case in CASES}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_lack_of_fit_test_is_r_s(reports, case):
    stat, df, p = _test_values(_row(reports[case], "life_stress"))
    r_stat, r_df, r_p = R[case]["lof"]
    assert df == r_df
    np.testing.assert_allclose(stat, r_stat, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p, r_p, rtol=1e-5)


@pytest.mark.parametrize("case", sorted(set(CASES) - {"exponential_complete"}))
def test_the_common_shape_test_is_r_s(reports, case):
    stat, df, p = _test_values(_row(reports[case], "common_shape"))
    r_stat, r_df, r_p = R[case]["shape"]
    assert df == r_df
    np.testing.assert_allclose(stat, r_stat, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p, r_p, rtol=1e-5)


@pytest.mark.parametrize("case", sorted(c for c in R if "loglik" in R[c]))
def test_the_compared_life_models_are_r_s(reports, case):
    d = reports[case]
    k = 2 + (CASES[case][0] is not sp.Exponential)
    for name, loglik in R[case]["loglik"].items():
        aic = _row(d, "life_model_comparison", name)["statistic"]
        np.testing.assert_allclose(aic, 2 * k - 2 * loglik, rtol=1e-8)


def test_the_insulating_fluid_meets_the_assumptions(reports):
    d = reports["weibull_complete"]
    assert d.ok and not d.advice, d
    assert d.family == "Accelerated life"
    assert d.columns == ["stress"]
    # Nelson's estimates: the transcription is his data
    model = _fluid("weibull_complete")
    np.testing.assert_allclose(model.params[1], 0.777, atol=5e-4)
    np.testing.assert_allclose(model.params[3], -17.73, atol=5e-3)


def test_the_lognormal_fails_common_shape_and_is_advised(reports):
    d = reports["lognormal_complete"]
    assert _row(d, "common_shape")["verdict"] == "fail"
    advice = [a for a in d.advice if a.check == "common_shape"]
    assert len(advice) == 1
    assert advice[0].features == ("AcceleratedLife", "LogNormal.fit")
    assert "sigma from" in advice[0]


def test_a_level_with_one_failure_is_left_out_and_noted(reports):
    d = reports["weibull_censored"]
    assert any("leaves out 1 stress level (26)" in n for n in d.notes)
    assert "6 levels" in _row(d, "common_shape")["detail"]


def test_the_exponential_has_no_shape_to_test(reports):
    d = reports["exponential_complete"]
    assert "common_shape" not in set(d.table["check"])
    assert any("no free parameter besides its life" in n for n in d.notes)


# -- the tests by hand ------------------------------------------------------
def test_the_tests_are_likelihood_ratios_of_separate_fits():
    rng = np.random.default_rng(4)
    S = np.repeat([1.0, 2.0, 3.0], 25)
    t = sp.Weibull.random(len(S), 1, 2, random_state=rng) * 50 * S**-1.2
    c = (t > 30).astype(int)
    t = np.minimum(t, 30)
    model = _fit(sp.Weibull, t, S, c=c)
    d = model.diagnose()
    separate = sum(
        -sp.Weibull.fit(t[S == s], c=c[S == s]).neg_ll() for s in (1, 2, 3)
    )
    # The common-shape, separate-life model, fitted here by maximising
    # its own likelihood
    from scipy.optimize import minimize

    def neg_ll(p):
        beta, alphas = np.exp(p[0]), np.exp(p[1:])
        out = 0.0
        for s, a in zip((1, 2, 3), alphas):
            m = sp.Weibull.from_params([a, beta])
            k = S == s
            out -= np.sum(np.log(m.df(t[k][c[k] == 0])))
            out -= np.sum(np.log(m.sf(t[k][c[k] == 1])))
        return out

    common = -minimize(neg_ll, np.log([2.0, 40, 25, 20]), method="BFGS").fun
    stat, df, _ = _test_values(_row(d, "common_shape"))
    assert df == 2
    np.testing.assert_allclose(stat, 2 * (separate - common), rtol=1e-4)
    stat, df, _ = _test_values(_row(d, "life_stress"))
    assert df == 1
    np.testing.assert_allclose(
        stat, 2 * (common + model.neg_ll()), rtol=1e-4, atol=1e-6
    )


def test_counts_are_repeated_rows():
    rng = np.random.default_rng(2)
    S = np.repeat([1.0, 2.0, 4.0], 12)
    t = np.round(sp.Weibull.random(36, 1, 2, random_state=rng) * 40 / S, 1)
    frame = pd.DataFrame({"t": t, "S": S}).value_counts().reset_index()
    grouped = _fit(
        sp.Weibull, frame["t"].values, frame["S"].values, n=frame["count"]
    ).diagnose()
    rows = _fit(sp.Weibull, t, S).diagnose()
    numeric = ["statistic", "p"]
    pd.testing.assert_frame_equal(
        grouped.table[numeric], rows.table[numeric], rtol=1e-5
    )


# -- the checks and their advice ----------------------------------------------
def _weibull_power(seed, levels=(1.0, 2.0, 3.0, 4.0), n=40, beta=2.0):
    rng = np.random.default_rng(seed)
    S = np.repeat(levels, n)
    beta = np.broadcast_to(beta, S.shape)
    t = np.array(
        [sp.Weibull.random(1, 1, b, random_state=rng)[0] for b in beta]
    )
    return t * 100 * S**-1.5, S


def test_data_that_meet_the_assumptions_pass():
    d = _fit(sp.Weibull, *_weibull_power(0)).diagnose()
    assert d.ok and not d.advice, d
    assert set(d.table["verdict"]) == {"ok"}
    assert list(dict.fromkeys(d.table["check"])) == [
        "common_shape",
        "life_stress",
        "life_model_comparison",
        "extrapolation",
    ]
    own = _row(d, "life_model_comparison", "Power")
    assert own["detail"] == "the fitted life model"
    # InversePower is Power: it is not compared
    assert "InversePower" not in set(d.table["life_model"])


def test_a_shape_that_changes_with_stress_fails():
    S = np.repeat([1.0, 2.0, 3.0, 4.0], 40)
    t, S = _weibull_power(1, beta=np.where(S >= 4, 1.0, 3.0))
    d = _fit(sp.Weibull, t, S).diagnose()
    row = _row(d, "common_shape")
    assert row["verdict"] == "fail"
    assert row["detail"].startswith("the shape differs")
    [advice] = [a for a in d.advice if a.check == "common_shape"]
    assert advice.features == ("AcceleratedLife", "Weibull.fit")


def test_a_wrong_life_model_fails_and_the_best_is_advised():
    rng = np.random.default_rng(3)
    S = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 40)
    t = sp.Weibull.random(len(S), 1, 4, random_state=rng) * (2000 - 350 * S)
    d = _fit(sp.Weibull, t, S).diagnose()
    assert _row(d, "life_stress")["verdict"] == "fail"
    assert _row(d, "life_model_comparison", "Linear")["verdict"] == "warn"
    [advice] = [a for a in d.advice if a.check == "life_stress"]
    assert advice.features == ("AcceleratedLife", "life_models.Linear")
    assert "life_models.Linear" in advice
    # The failed test names the best model; the comparison does not
    # repeat it
    assert not [a for a in d.advice if a.check == "life_model_comparison"]


def test_no_better_life_model_is_advised_a_life_model_of_ones_own():
    rng = np.random.default_rng(0)
    S = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 30)
    t = sp.Weibull.random(len(S), 1, 3, random_state=rng)
    t = t * 100 * (1 + (S - 3) ** 2)  # no monotone life fits
    # The candidate with the lowest AIC: none is better than it
    d = _fit(sp.Weibull, t, S, life_model=lm.InverseEyring).diagnose()
    assert _row(d, "life_stress")["verdict"] == "fail"
    assert (
        d.table.loc[d.table["check"] == "life_model_comparison", "verdict"]
        == "ok"
    ).all()
    [advice] = [a for a in d.advice if a.check == "life_stress"]
    assert advice.features == (
        "life_models.LifeModel",
        "life_models.DualPower",
    )


def test_a_better_life_model_warns_and_is_advised():
    rng = np.random.default_rng(0)
    S = np.repeat([1.0, 2.0, 4.0], 30)
    t = (
        sp.Weibull.random(len(S), 1, 3, random_state=rng)
        * 100
        * np.exp(-0.6 * S)
    )
    # lack of fit p = 0.012: a warning at alpha = 0.01, not a failure
    d = _fit(sp.Weibull, t, S).diagnose(alpha=0.01)
    assert _row(d, "life_stress")["verdict"] == "warn"
    rows = d.table[d.table["verdict"] == "warn"]
    better = rows[rows["check"] == "life_model_comparison"]
    assert len(better)
    best = better.sort_values("statistic").iloc[0]
    own = _row(d, "life_model_comparison", "Power")["statistic"]
    assert own - best["statistic"] > DELTA_AIC
    [advice] = [a for a in d.advice if a.check == "life_model_comparison"]
    assert advice.features == (
        "AcceleratedLife",
        "life_models." + best["life_model"],
    )


def test_two_stress_levels_warn_and_are_advised_more():
    t, S = _weibull_power(5, levels=(1.0, 2.0), n=30)
    d = _fit(sp.Weibull, t, S).diagnose()
    assert d.ok  # warnings only
    lof = _row(d, "life_stress")
    assert lof["verdict"] == "warn" and np.isnan(lof["p"])
    assert "passes through every level's life" in lof["detail"]
    row = _row(d, "extrapolation")
    assert row["verdict"] == "warn" and row["statistic"] == 2
    [advice] = [a for a in d.advice if a.check == "extrapolation"]
    assert advice.features == (
        "test_planning.accelerated_test_plan",
        "test_planning.accelerated_test_precision",
    )
    assert "three or more levels" in advice


def test_a_continuous_stress_says_the_tests_need_levels():
    rng = np.random.default_rng(1)
    S = rng.uniform(1, 5, 80)
    t = sp.Weibull.random(80, 1, 2, random_state=rng) * 100 * S**-1.5
    d = _fit(sp.Weibull, t, S).diagnose()
    for check in ("common_shape", "life_stress"):
        row = _row(d, check)
        assert row["verdict"] == "warn" and np.isnan(row["p"])
        assert "the stress is continuous" in row["detail"]
    # The comparison needs no levels
    comparison = d.table[d.table["check"] == "life_model_comparison"]
    assert comparison["statistic"].notna().all() and len(comparison) > 1


def test_two_stresses_and_columns():
    rng = np.random.default_rng(3)
    T = np.tile(np.repeat([320.0, 350.0, 380.0], 20), 2)
    U = np.repeat([5.0, 10.0], 60)
    t = sp.Weibull.random(120, 1, 2, random_state=rng)
    t = t * np.exp(-5 + 4000 / T) * U**-1.2
    frame = pd.DataFrame({"t": t, "T": T, "U": U})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.AcceleratedLife(
            sp.Weibull, lm.PowerExponential
        ).fit_from_df(frame, x_col="t", Z_cols=["T", "U"])
    d = model.diagnose()
    assert d.columns == ["T", "U"]
    compared = set(
        d.table.loc[d.table["check"] == "life_model_comparison", "life_model"]
    )
    assert compared == {
        "PowerExponential",
        "DualPower",
        "DualExponential",
        "GeneralLogLinear",
    }
    levels = d.table[d.table["check"] == "extrapolation"]
    assert list(levels["covariates"]) == [("T",), ("U",)]
    assert list(levels["verdict"]) == ["ok", "warn"]
    only = model.diagnose(columns=["T"])
    levels = only.table[only.table["check"] == "extrapolation"]
    assert list(levels["covariates"]) == [("T",)]
    with pytest.raises(ValueError, match="'columns'"):
        model.diagnose(columns=["stress"])


def test_fixed_parameters_are_held_in_the_tests():
    t, S = _weibull_power(2)
    shape_fixed = _fit(sp.Weibull, t, S, fixed={"beta": 2.0}).diagnose()
    assert "common_shape" not in set(shape_fixed.table["check"])
    assert any("held fixed" in n for n in shape_fixed.notes)
    exponent_fixed = _fit(sp.Weibull, t, S, fixed={"n": -1.5}).diagnose()
    # 4 levels, one free life-model parameter
    assert _row(exponent_fixed, "life_stress")["df"] == 3


def test_interval_censored_and_truncated_data():
    t, S = _weibull_power(3)
    xl = np.floor(t)
    model = _fit(
        sp.Weibull,
        np.column_stack([xl, xl + 1]),
        S,
        c=np.full(len(S), 2),
        tl=0.0,
    )
    d = model.diagnose()
    tests = d.table[d.table["check"].isin(["common_shape", "life_stress"])]
    assert tests["p"].notna().all()


# -- refusals -----------------------------------------------------------------
def test_a_restored_model_is_refused():
    model = _fit(sp.Weibull, X, V)
    with pytest.raises(ValueError, match="from_dict"):
        sp.from_dict(model.to_dict()).diagnose()


def test_a_fit_with_no_finite_maximum_is_refused():
    model = copy.copy(_fit(sp.Weibull, X, V))
    model.maximum = "no finite maximum"
    with pytest.raises(ValueError, match="finite maximum"):
        model.diagnose()


def test_other_regressions_are_not_this_family():
    t, S = _weibull_power(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft = sp.WeibullAFT.fit(t, S)
    assert aft.diagnose().family != "Accelerated life"


# -- the report -------------------------------------------------------------
def test_to_dict_round_trips(reports):
    d = reports["lognormal_complete"]
    out = d.to_dict()
    back = Diagnosis.from_dict(json.loads(json.dumps(out, allow_nan=False)))
    assert back.to_dict() == out
    pd.testing.assert_frame_equal(back.table, d.table)
    assert [a.features for a in back.advice] == [a.features for a in d.advice]


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _fluid("weibull_complete").diagnose()
    fig = d.plot()
    assert len(fig.axes) == 2
    # One probability plot line per level and one dashed model line each
    lines = fig.axes[0].get_lines()
    assert sum(line.get_linestyle() == "--" for line in lines) == len(FLUID)
    plt.close(fig)
    fig = d.plot("life_stress")
    assert len(fig.axes) == 1
    plt.close(fig)
    with pytest.raises(ValueError, match="which"):
        d.plot("extrapolation")


# -- the use stress (the family's option) ---------------------------------
def _use_row(d, name="stress"):
    table = d.table[
        (d.table["check"] == "extrapolation")
        & (d.table["test"] == "use stress vs tested range")
        & (d.table["covariates"] == (name,))
    ]
    assert len(table) == 1, d.table
    return table.iloc[0]


def test_the_use_stress_is_measured_on_the_life_model_s_scale():
    model = _fit(sp.Weibull, X, V)
    near = model.diagnose(use_stress=20.0)
    row = _use_row(near)
    # The power law is linear in log V: 20 kV is 0.69 of the tested
    # range's width (26 to 38 kV) below it
    width = np.log(38) - np.log(26)
    expected = (np.log(26) - np.log(20)) / width
    np.testing.assert_allclose(row["statistic"], expected, rtol=1e-12)
    assert row["verdict"] == "ok" and near.ok and not near.advice
    assert "log scale of the Power model" in row["detail"]
    assert not any("diagnose(use_stress=...)" in n for n in near.notes)
    inside = _use_row(model.diagnose(use_stress=30.0))
    assert inside["statistic"] == 0 and inside["verdict"] == "ok"
    # The comparison gives each life model's life at the use stress
    own = _row(near, "life_model_comparison", "Power")
    life = float(np.ravel(model.phi(np.array([[20.0]])))[0])
    assert own["detail"].endswith("life at the use stress {:.3g}".format(life))


def test_a_far_use_stress_warns_and_is_advised_caution():
    model = _fit(sp.Weibull, X, V)
    d = model.diagnose(use_stress=10.0)
    row = _use_row(d)
    width = np.log(38) - np.log(26)
    np.testing.assert_allclose(
        row["statistic"], (np.log(26) - np.log(10)) / width
    )
    assert row["verdict"] == "warn" and d.ok
    [advice] = [a for a in d.advice if a.check == "extrapolation"]
    assert advice.features == ("cb(Z=)", "test_planning.accelerated_test_plan")
    assert "far outside the tested range" in advice


def test_an_arrhenius_model_measures_the_use_stress_in_one_over_kelvin():
    rng = np.random.default_rng(7)
    T = np.repeat([350.0, 375.0, 400.0], 30)
    t = sp.Weibull.random(90, 1, 2, random_state=rng) * np.exp(-10 + 6000 / T)
    model = _fit(sp.Weibull, t, T, life_model=lm.Arrhenius)
    row = _use_row(model.diagnose(use_stress=300.0))
    expected = (1 / 300 - 1 / 350) / (1 / 350 - 1 / 400)
    np.testing.assert_allclose(row["statistic"], expected, rtol=1e-12)
    assert row["verdict"] == "warn"
    assert "1 / stress scale" in row["detail"]


def test_two_stresses_take_a_use_stress_each():
    rng = np.random.default_rng(3)
    T = np.tile(np.repeat([320.0, 350.0, 380.0], 20), 2)
    U = np.repeat([5.0, 10.0], 60)
    t = sp.Weibull.random(120, 1, 2, random_state=rng)
    t = t * np.exp(-5 + 4000 / T) * U**-1.2
    model = _fit(sp.Weibull, t, np.column_stack([T, U]), lm.PowerExponential)
    d = model.diagnose(use_stress=[300.0, 2.0])
    temperature, voltage = _use_row(d, "stress_0"), _use_row(d, "stress_1")
    np.testing.assert_allclose(
        temperature["statistic"],
        (1 / 300 - 1 / 320) / (1 / 320 - 1 / 380),
    )
    np.testing.assert_allclose(
        voltage["statistic"], (np.log(5) - np.log(2)) / np.log(2)
    )
    only = model.diagnose(use_stress=[300.0, 2.0], columns=["stress_1"])
    use_rows = only.table[only.table["test"] == "use stress vs tested range"]
    assert list(use_rows["covariates"]) == [("stress_1",)]


@pytest.mark.parametrize("bad", [[20.0, 30.0], -5.0, 0.0, "x", np.nan, []])
def test_a_use_stress_must_be_one_stress_the_model_takes(bad):
    model = _fit(sp.Weibull, X, V)
    with pytest.raises(ValueError, match="'use_stress'"):
        model.diagnose(use_stress=bad)


def test_an_option_the_family_does_not_take_is_refused():
    model = _fit(sp.Weibull, X, V)
    with pytest.raises(ValueError, match="no option 'stress'.*'use_stress'"):
        model.diagnose(stress=20.0)
    from surpyval.datasets import load_rossi_static

    df = load_rossi_static()
    cox = sp.CoxPH.fit(
        df["week"].values, df[["fin"]].values, c=1 - df["arrest"].values
    )
    with pytest.raises(ValueError, match="no option 'use_stress'.*'t0'"):
        cox.diagnose(use_stress=20.0)


def test_without_a_use_stress_the_report_says_how_to_give_one():
    d = _fit(sp.Weibull, X, V).diagnose()
    assert any("diagnose(use_stress=...)" in n for n in d.notes)
    assert "use stress vs tested range" not in set(d.table["test"])


def test_the_plot_marks_the_use_stress():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _fit(sp.Weibull, X, V).diagnose(use_stress=20.0)
    fig = d.plot("life_stress")
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "use stress" in labels
    plt.close(fig)
