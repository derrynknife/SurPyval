"""Penalised (ridge, lasso, elastic-net) fits beyond Cox and the
parametric PH, AFT and PO families: Royston-Parmar regression, the Cox
model with time-varying covariates (``fit_tvc_penalised`` and
``fit_tvc_timeline_penalised``), the cause-specific competing-risks Cox
model, the proportional intensity HPP and the distributional
regressions.

The references are R's glmnet 5.2 (R 4.6.2, survival 3.8.12; threshold
1e-30), printed by ``scripts/reference/reference_r_penalised_more.R``:
the cause-specific Cox model is glmnet's ``family = "cox"`` fitted
cause by cause (the other causes censored) on the PBC data at
enrolment, and the HPP is glmnet's
``family = "poisson"`` of each Rossi subject's arrests with
``offset = log(week)``. The start-stop Cox references are those of
``test_penalised.py`` (glmnet on the Rossi person-week rows). The
Royston-Parmar fit is checked as the identity it is: the PH or PO family
on the spline baseline, fitted by its own ``fit_penalised``.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval import CoxPH, RoystonParmarPH, RoystonParmarPO
from surpyval.datasets import (
    load_pbc2,
    load_rossi_static,
    load_rossi_time_varying,
)
from surpyval.recurrent import ProportionalIntensityHPP
from surpyval.tests.univariate.regression.test_penalised import (
    GLMNET_START_STOP,
)
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression.royston_parmar_regression import (
    regression_knots,
)

ROSSI = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]

# glmnet 5.2: (ties, lambda, alpha, cause) -> the coefficients of
# age, female, drug, log(bilirubin), albumin
GLMNET_CAUSES = {
    ("breslow", 0.05, 0.0, "dead"): [
        0.03593411707,
        -0.196127087,
        -0.1295466734,
        0.8600482356,
        -0.9179839007,
    ],
    ("breslow", 0.05, 0.0, "transplanted"): [
        -0.05101252184,
        -0.1714245156,
        -0.1744820108,
        0.4953815918,
        -0.4526800693,
    ],
    ("breslow", 0.05, 0.5, "dead"): [
        0.03358555895,
        -0.08945704621,
        -0.009408961298,
        0.8687833467,
        -0.8232257614,
    ],
    ("breslow", 0.05, 0.5, "transplanted"): [
        -0.03964717103,
        0.0,
        0.0,
        0.402394533,
        0.0,
    ],
    ("breslow", 0.02, 1.0, "dead"): [
        0.03673691794,
        -0.07024604467,
        -0.04428616315,
        0.9408327528,
        -0.8511476695,
    ],
    ("breslow", 0.02, 1.0, "transplanted"): [
        -0.05679189987,
        0.0,
        0.0,
        0.5916812592,
        -0.02412807257,
    ],
    ("efron", 0.05, 0.0, "dead"): [
        0.03590525405,
        -0.1957737226,
        -0.1288075617,
        0.8604360313,
        -0.9191226761,
    ],
    ("efron", 0.05, 0.0, "transplanted"): [
        -0.05101252184,
        -0.1714245156,
        -0.1744820108,
        0.4953815918,
        -0.4526800693,
    ],
    ("efron", 0.05, 0.5, "dead"): [
        0.03355511773,
        -0.08909013517,
        -0.008628287644,
        0.8692023327,
        -0.8245008304,
    ],
    ("efron", 0.05, 0.5, "transplanted"): [
        -0.03964717103,
        0.0,
        0.0,
        0.402394533,
        0.0,
    ],
    ("efron", 0.02, 1.0, "dead"): [
        0.03670059052,
        -0.0698523811,
        -0.04337258801,
        0.9412971509,
        -0.8525980565,
    ],
    ("efron", 0.02, 1.0, "transplanted"): [
        -0.05679189987,
        0.0,
        0.0,
        0.5916812592,
        -0.02412807257,
    ],
}

# glmnet 5.2 Poisson with offset log(week): (lambda, alpha) ->
# (intercept, the coefficients of ROSSI)
GLMNET_HPP = {
    (0.05, 0.0): (
        -4.279551552,
        [
            -0.308184877,
            -0.04233692599,
            0.239687392,
            -0.1857827944,
            -0.3558258429,
            -0.06565342614,
            0.07514199864,
        ],
    ),
    (0.02, 0.5): (
        -4.250997877,
        [
            -0.270194229,
            -0.04416189955,
            0.1589976352,
            -0.1444054382,
            -0.2940733338,
            -0.001024740201,
            0.07641417948,
        ],
    ),
    (0.01, 1.0): (
        -4.203645482,
        [
            -0.2798402928,
            -0.04689396241,
            0.1683719738,
            -0.1348930712,
            -0.3038968569,
            -0.00191642615,
            0.07871151207,
        ],
    ),
}


def _pbc():
    df = load_pbc2().drop_duplicates("id")
    Z = np.column_stack(
        [
            df["age"],
            df["sex"] == "female",
            df["drug"] == "D-penicil",
            np.log(df["serBilir"]),
            df["albumin"],
        ]
    ).astype(float)
    e = np.where(df["status"] == "alive", None, df["status"]).astype(object)
    return df["years"].to_numpy(float), Z, e


def _rossi():
    df = load_rossi_static()
    return (
        df["week"].to_numpy(float),
        df[ROSSI].to_numpy(float),
        1 - df["arrest"].to_numpy(),
    )


def _rp_data():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(200, 4))
    x = 10 * rng.weibull(1.5, 200) * np.exp(-0.5 * Z[:, 0])
    c = (rng.uniform(size=200) < 0.2).astype(int)
    return x, Z, c


# -- Royston-Parmar ---------------------------------------------------------


@pytest.mark.parametrize("fitter", [RoystonParmarPH, RoystonParmarPO])
@pytest.mark.parametrize("l1_ratio", [0.0, 1.0])
def test_royston_parmar_is_the_family_on_the_spline(fitter, l1_ratio):
    x, Z, c = _rp_data()
    model = fitter.fit_penalised(x, Z, c=c, penalizer=0.05, l1_ratio=l1_ratio)
    knots = regression_knots(x, Z, c, None, None, fitter.df)
    direct = fitter.family(knots).fit_penalised(
        x, Z, c=c, penalizer=0.05, l1_ratio=l1_ratio
    )
    np.testing.assert_array_equal(model.params, direct.params)
    assert model.maximum == "verified"
    assert model.penalty.effective_df == direct.penalty.effective_df
    if l1_ratio == 1.0:
        assert np.any(model.params[-4:] == 0)


def test_royston_parmar_tiny_penalty_is_the_fit():
    x, Z, c = _rp_data()
    plain = RoystonParmarPH.fit(x, Z, c=c)
    tiny = RoystonParmarPH.fit_penalised(x, Z, c=c, penalizer=1e-9)
    np.testing.assert_allclose(tiny.params, plain.params, atol=1e-5)


# -- Cox with time-varying covariates ----------------------------------------


def _rossi_tv():
    tv = load_rossi_time_varying()
    Z = np.column_stack(
        [tv["fin"] == "yes", tv["age"], tv["prio"], tv["employed"] == "yes"]
    ).astype(float)
    return (
        tv["id"].to_numpy(),
        tv["start"].to_numpy(float),
        tv["stop"].to_numpy(float),
        1 - tv["event"].to_numpy(),
        Z,
    )


@pytest.mark.parametrize("setting", sorted(GLMNET_START_STOP))
def test_fit_tvc_penalised_matches_glmnet(setting):
    ties, lam, alpha = setting
    i, start, stop, c, Z = _rossi_tv()
    model = CoxPH.fit_tvc_penalised(
        i, start, stop, c, Z, penalizer=lam, l1_ratio=alpha, tie_method=ties
    )
    assert model.is_tvc and model.maximum == "verified"
    ref = np.asarray(GLMNET_START_STOP[setting])
    np.testing.assert_array_equal(model.params == 0, ref == 0)
    np.testing.assert_allclose(model.params, ref, rtol=1e-6, atol=1e-10)


def test_fit_tvc_timeline_penalised_is_the_start_stop_fit():
    from surpyval.univariate.regression.proportional_hazards.tvc import (
        handle_tvc_timeline,
    )

    i = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    x = [0, 2, 0, 3, 0, 1, 6, 0, 4, 0, 5]
    Z = [
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [0.0],
        [0.0],
    ]
    c = [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    timeline = CoxPH.fit_tvc_timeline_penalised(i, x, Z, c, penalizer=0.2)
    i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_tvc_timeline(i, x, Z, c, None)
    start_stop = CoxPH.fit_tvc_penalised(
        i_ss, xl, xr, c_ss, Z_ss, n=n_ss, penalizer=0.2
    )
    assert timeline.is_tvc and timeline.penalty is not None
    np.testing.assert_array_equal(timeline.params, start_stop.params)


def test_fit_tvc_penalised_refuses_inference_and_weights():
    i, start, stop, c, Z = _rossi_tv()
    model = CoxPH.fit_tvc_penalised(i, start, stop, c, Z, penalizer=0.01)
    with pytest.raises(ValueError, match="fitted with a penalty"):
        model.standard_errors()
    with pytest.raises(ValueError, match="does not take weights"):
        CoxPH.fit_tvc_penalised(i, start, stop, c, Z, w=np.ones(len(i)))


# -- cause-specific competing risks Cox --------------------------------------


@pytest.mark.parametrize("setting", sorted({k[:3] for k in GLMNET_CAUSES}))
def test_cause_specific_cox_matches_glmnet_per_cause(setting):
    ties, lam, alpha = setting
    x, Z, e = _pbc()
    model = CompetingRisksProportionalHazards.fit_penalised(
        x, Z, e, penalizer=lam, l1_ratio=alpha, tie_method=ties
    )
    assert model.maximum == "verified"
    for cause, row in model.event_idx_map.items():
        ref = np.asarray(GLMNET_CAUSES[(ties, lam, alpha, cause)])
        got = model.betas[row]
        np.testing.assert_array_equal(got == 0, ref == 0)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-10)


def test_cause_specific_cox_penalty_and_refusals():
    x, Z, e = _pbc()
    model = CompetingRisksProportionalHazards.fit_penalised(
        x, Z, e, penalizer=0.05, l1_ratio=0.5
    )
    dfs = [m.penalty.effective_df for m in model.models.values()]
    assert model.penalty.effective_df == pytest.approx(sum(dfs))
    assert model.aic() == pytest.approx(
        2 * sum(dfs) + 2 * sum(m.neg_ll() for m in model.models.values())
    )
    for call in (model.covariance, model.standard_errors):
        with pytest.raises(ValueError, match="fitted with a penalty"):
            call()
    with pytest.raises(ValueError, match="fitted with a penalty"):
        model.p_values
    table = model.summary()
    assert np.all(np.isnan(table["se(coef)"]))
    assert "Penalty" in repr(model)
    restored = sp.from_dict(model.to_dict())
    assert restored.penalty.effective_df == model.penalty.effective_df
    np.testing.assert_array_equal(restored.betas, model.betas)
    with pytest.raises(ValueError, match="fitted with a penalty"):
        restored.covariance()
    np.testing.assert_allclose(
        restored.cif([2, 5], Z[:1], "dead"), model.cif([2, 5], Z[:1], "dead")
    )
    with pytest.raises(ValueError, match="does not take weights"):
        CompetingRisksProportionalHazards.fit_penalised(
            x, Z, e, w=np.ones(len(x))
        )


# -- proportional intensity HPP ----------------------------------------------


@pytest.mark.parametrize("setting", sorted(GLMNET_HPP))
def test_hpp_matches_glmnet_poisson(setting):
    lam, alpha = setting
    x, Z, c = _rossi()
    model = ProportionalIntensityHPP.fit_penalised(
        x, Z, i=np.arange(x.size), c=c, penalizer=lam, l1_ratio=alpha
    )
    assert model.maximum == "verified"
    intercept, ref = GLMNET_HPP[setting]
    ref = np.asarray(ref)
    np.testing.assert_allclose(np.log(model.params[0]), intercept, rtol=1e-6)
    np.testing.assert_array_equal(model.coeffs == 0, ref == 0)
    np.testing.assert_allclose(model.coeffs, ref, rtol=1e-6, atol=1e-10)


def test_hpp_penalty_and_refusals():
    x, Z, c = _rossi()
    i = np.arange(x.size)
    model = ProportionalIntensityHPP.fit_penalised(
        x, Z, i=i, c=c, penalizer=0.02
    )
    k = model.penalty.effective_df
    assert 1 < k < 1 + Z.shape[1]
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    for call in (
        model.covariance,
        model.standard_errors,
        lambda: model.cif_cb([5.0], Z[0]),
    ):
        with pytest.raises(ValueError, match="fitted with a penalty"):
            call()
    with pytest.raises(ValueError, match="penalised fit"):
        model.cramer_von_mises(n_boot=2, random_state=1)
    assert np.all(np.isnan(model.summary()["se"]))
    assert "Penalty" in repr(model)
    restored = sp.from_dict(model.to_dict())
    assert restored.penalty.effective_df == model.penalty.effective_df
    with pytest.raises(ValueError, match="fitted with a penalty"):
        restored.covariance()
    np.testing.assert_allclose(
        restored.cif([5.0], Z[0]), model.cif([5.0], Z[0])
    )
    with pytest.raises(ValueError, match="does not take weights"):
        ProportionalIntensityHPP.fit_penalised(x, Z, i=i, c=c, w=1.0)


def test_hpp_tiny_penalty_is_the_fit():
    x, Z, c = _rossi()
    i = np.arange(x.size)
    plain = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    tiny = ProportionalIntensityHPP.fit_penalised(
        x, Z, i=i, c=c, penalizer=1e-10
    )
    np.testing.assert_allclose(tiny.coeffs, plain.coeffs, atol=1e-5)
    np.testing.assert_allclose(tiny.params, plain.params, rtol=1e-5)


# -- distributional regression -----------------------------------------------


def _distributional_data():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(300, 3))
    x = 10 * np.exp(-0.5 * Z[:, 0]) * rng.weibull(1.5 * np.exp(0.2 * Z[:, 1]))
    c = (rng.uniform(size=300) < 0.2).astype(int)
    return x, Z, c


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_distributional_on_the_scale_is_weibull_aft(l1_ratio):
    # Covariates on log(alpha) alone: WeibullAFT, the signs changed
    # (alpha * exp(b'Z) against S0(t exp(b'Z))). Both minimise the same
    # penalised objective, so they agree to the solvers' tolerance.
    from surpyval import WeibullAFT, WeibullDistributional

    x, Z, c = _distributional_data()
    model = WeibullDistributional.fit_penalised(
        x,
        Z,
        c=c,
        penalizer=0.05,
        l1_ratio=l1_ratio,
        covariates={"alpha": [0, 1, 2]},
    )
    aft = WeibullAFT.fit_penalised(
        x, Z, c=c, penalizer=0.05, l1_ratio=l1_ratio
    )
    assert model.maximum == aft.maximum == "verified"
    np.testing.assert_allclose(model.params[:2], aft.params[:2], rtol=1e-7)
    np.testing.assert_array_equal(model.params[2:] == 0, aft.params[2:] == 0)
    np.testing.assert_allclose(
        model.params[2:], -aft.params[2:], rtol=1e-6, atol=1e-9
    )
    assert model.penalty.effective_df == pytest.approx(
        aft.penalty.effective_df, rel=1e-6
    )


def test_distributional_optimality_conditions_per_parameter_penalty():
    # A penalty per parameter: the optimality (KKT) conditions of the
    # objective on the covariates' own scale, the gradient of the
    # likelihood over N against lambda_j * s_j (the L1 part) at each
    # coefficient, with s_j its column's standard deviation.
    from autograd import grad

    from surpyval import WeibullDistributional

    x, Z, c = _distributional_data()
    lam = {"alpha": 0.02, "beta": 0.1}
    model = WeibullDistributional.fit_penalised(
        x, Z, c=c, penalizer=lam, l1_ratio=1.0
    )
    assert model.maximum == "verified"
    np.testing.assert_array_equal(
        model.penalty.penalizer, [0.02, 0.02, 0.02, 0.1, 0.1, 0.1]
    )
    params = np.asarray(model.params, dtype=float)
    g = np.asarray(
        grad(lambda v: model.model.neg_ll(model.data, *v) / x.size)(params)
    )
    # The intercepts are free: the gradient is 0 there.
    np.testing.assert_allclose(g[:2], 0, atol=1e-9)
    weight = model.penalty.penalizer * model.penalty.scale
    b, gb = params[2:], g[2:]
    nonzero = b != 0
    assert nonzero.any() and (~nonzero).any()
    np.testing.assert_allclose(
        gb[nonzero] + weight[nonzero] * np.sign(b[nonzero]), 0, atol=1e-9
    )
    assert np.all(np.abs(gb[~nonzero]) <= weight[~nonzero] + 1e-9)


def test_distributional_penalizer_forms_and_refusals():
    from surpyval import WeibullDistributional

    x, Z, c = _distributional_data()
    per_column = WeibullDistributional.fit_penalised(
        x, Z, c=c, penalizer=[0.1, 0.0, 0.2]
    )
    np.testing.assert_array_equal(
        per_column.penalty.penalizer, [0.1, 0.0, 0.2, 0.1, 0.0, 0.2]
    )
    with pytest.raises(ValueError, match="among the distribution"):
        WeibullDistributional.fit_penalised(x, Z, penalizer={"shape": 1})
    with pytest.raises(ValueError, match=r"penalizer\['beta'\]"):
        WeibullDistributional.fit_penalised(
            x, Z, penalizer={"beta": [1.0, 2.0]}
        )
    with pytest.raises(ValueError, match="does not take weights"):
        WeibullDistributional.fit_penalised(x, Z, w=np.ones(x.size))
    model = WeibullDistributional.fit_penalised(x, Z, c=c, penalizer=0.05)
    for call in (
        model.covariance,
        model.standard_errors,
        lambda: model.parameter_cb("beta", Z[:1]),
        lambda: model.parameter_cb("beta", Z[:1], method="lr"),
        lambda: model.cb(5.0, Z[:1]),
    ):
        with pytest.raises(ValueError, match="fitted with a penalty"):
            call()
    with pytest.raises(ValueError, match="penalised fit"):
        model.likelihood_ratio_test()
    k = model.penalty.effective_df
    assert 2 < k < 8
    assert model.aic() == pytest.approx(2 * k + 2 * model.neg_ll())
    restored = sp.from_dict(model.to_dict())
    assert "covariance" not in model.to_dict()
    assert restored.penalty.effective_df == k
    np.testing.assert_allclose(
        restored.sf([2.0, 5.0], Z[:2]), model.sf([2.0, 5.0], Z[:2])
    )
    with pytest.raises(ValueError, match="fitted with a penalty"):
        restored.covariance()
