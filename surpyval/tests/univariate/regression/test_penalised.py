"""Penalised (ridge, lasso, elastic-net) regression: ``fit_penalised`` on
``CoxPH`` and the parametric families.

The Cox references are R's glmnet 5.0 (R 4.4.3) on the Rossi data, whose
objective ``fit_penalised`` minimises: the negative partial
log-likelihood over the number of units plus the penalty on the
coefficients of the covariates divided by their standard deviations
(divisor ``n``), with ``alpha`` the ``l1_ratio``. They were computed with

.. code-block:: r

    library(glmnet); library(survival)
    cols <- c("fin", "age", "race", "wexp", "mar", "paro", "prio")
    X <- as.matrix(rossi[, cols]); y <- Surv(rossi$week, rossi$arrest)
    path <- exp(seq(log(1), log(lambda), length.out = 60))
    f <- glmnet(X, y, family = "cox", alpha = alpha, lambda = path,
                cox.ties = ties, control = list(thresh = 1e-20, maxit = 1e7))
    f$beta[, ncol(f$beta)]

and the ridge fit is checked against lifelines 0.30.3's
``CoxPHFitter(penalizer=0.05)`` too, which standardises with divisor
``n - 1`` (so the comparison fits ``Z / sd`` with ``standardise=False``).
lifelines' lasso smooths the L1 term, so it is not used for ``l1_ratio >
0``. The parametric fits are checked against the optimality conditions of
a likelihood written out here with numpy.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval import CoxPH, WeibullAFT, WeibullPH
from surpyval.tests._helpers import rossi_with_censoring

COLS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
# glmnet 5.0: (ties, lambda, alpha) -> the coefficients of COLS
GLMNET = {
    ("breslow", 0.05, 0.0): [
        -0.3177817267,
        -0.04341975571,
        0.2465026511,
        -0.1925263943,
        -0.3606632164,
        -0.06747926915,
        0.07942497933,
    ],
    ("breslow", 0.05, 0.5): [
        -0.15438861,
        -0.03335587921,
        0,
        -0.1304277354,
        -0.136758636,
        0,
        0.06378222305,
    ],
    ("breslow", 0.02, 1.0): [
        -0.208400425,
        -0.04120304247,
        0.05431403433,
        -0.1247599978,
        -0.1958697797,
        0,
        0.07393582437,
    ],
    ("breslow", 0.1, 1.0): [0, 0, 0, 0, 0, 0, 0],
    ("efron", 0.05, 0.0): [
        -0.3181463442,
        -0.04354533615,
        0.2463550295,
        -0.1918379718,
        -0.3613748669,
        -0.0673805425,
        0.07972828235,
    ],
    ("efron", 0.05, 0.5): [
        -0.154811372,
        -0.03349818176,
        0,
        -0.129776426,
        -0.137713197,
        0,
        0.0640866214,
    ],
    ("efron", 0.02, 1.0): [
        -0.2088478799,
        -0.04137844013,
        0.0543776343,
        -0.1237592983,
        -0.196906578,
        0,
        0.07428783074,
    ],
    ("efron", 0.1, 1.0): [0, 0, 0, 0, 0, 0, 0],
}
# lifelines 0.30.3, CoxPHFitter(penalizer=0.05).params_[COLS]
LIFELINES_RIDGE = [
    -0.3180237527,
    -0.0435201137,
    0.2462333138,
    -0.1918874465,
    -0.3612223154,
    -0.0673515145,
    0.0797095081,
]


def _rossi():
    df = rossi_with_censoring()
    Z = df[COLS].to_numpy(dtype=float)
    return df["week"].to_numpy(float), df["censored"].to_numpy(), Z


@pytest.mark.parametrize("setting", sorted(GLMNET))
def test_cox_matches_glmnet(setting):
    ties, lam, alpha = setting
    x, c, Z = _rossi()
    model = CoxPH.fit_penalised(
        x, Z, c=c, penalizer=lam, l1_ratio=alpha, tie_method=ties
    )
    assert model.maximum == "verified"
    ref = np.asarray(GLMNET[setting])
    # The same coefficients are exactly 0.
    np.testing.assert_array_equal(model.params == 0, ref == 0)
    np.testing.assert_allclose(model.params, ref, rtol=1e-6, atol=1e-10)


def test_cox_ridge_matches_lifelines():
    x, c, Z = _rossi()
    sd = Z.std(0, ddof=1)
    model = CoxPH.fit_penalised(
        x, Z / sd, c=c, penalizer=0.05, standardise=False
    )
    # lifelines stops its Newton steps at its default precision: it is
    # within 7e-5 (relative) of this fit, which glmnet run to a 1e-20
    # threshold matches to 1e-6 (test_cox_matches_glmnet).
    np.testing.assert_allclose(model.params / sd, LIFELINES_RIDGE, rtol=1e-4)


def test_standardise_is_the_same_as_scaling_by_hand():
    # standardise=True divides by the count-weighted sd with divisor N.
    x, c, Z = _rossi()
    sd = Z.std(0)
    auto = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.05, l1_ratio=0.5)
    hand = CoxPH.fit_penalised(
        x, Z / sd, c=c, penalizer=0.05, l1_ratio=0.5, standardise=False
    )
    np.testing.assert_allclose(auto.params, hand.params / sd, atol=1e-9)
    np.testing.assert_allclose(auto.penalty.scale, sd)


def test_zero_penalty_is_the_unpenalised_fit():
    x, c, Z = _rossi()
    np.testing.assert_allclose(
        CoxPH.fit_penalised(x, Z, c=c, penalizer=0.0).params,
        CoxPH.fit(x, Z, c=c).params,
        rtol=1e-7,
    )
    y = x[:200]
    np.testing.assert_allclose(
        WeibullPH.fit_penalised(y, Z[:200], c=c[:200], penalizer=0.0).params,
        WeibullPH.fit(y, Z[:200], c=c[:200]).params,
        rtol=1e-5,
    )


def test_per_coefficient_penalty_and_units():
    # A coefficient with penalizer 0 is not shrunk; with standardise the
    # fit does not depend on a covariate's units (principle 6).
    x, c, Z = _rossi()
    pen = np.full(7, 0.1)
    pen[1] = 0.0  # age
    model = CoxPH.fit_penalised(x, Z, c=c, penalizer=pen, l1_ratio=1.0)
    assert model.params[1] != 0
    Z2 = Z.copy()
    Z2[:, 1] *= 52.0  # age in weeks
    model2 = CoxPH.fit_penalised(x, Z2, c=c, penalizer=pen, l1_ratio=1.0)
    np.testing.assert_allclose(model2.params[1] * 52.0, model.params[1])
    np.testing.assert_allclose(
        np.delete(model2.params, 1), np.delete(model.params, 1), atol=1e-9
    )


def test_counts_equal_repetition():
    x, c, Z = _rossi()
    n = 1 + np.arange(x.size) % 3
    rep = np.repeat(np.arange(x.size), n)
    for fitter in (CoxPH, WeibullAFT):
        counted = fitter.fit_penalised(
            x, Z, c=c, n=n, penalizer=0.03, l1_ratio=0.7
        )
        expanded = fitter.fit_penalised(
            x[rep], Z[rep], c=c[rep], penalizer=0.03, l1_ratio=0.7
        )
        np.testing.assert_allclose(
            counted.params, expanded.params, rtol=1e-6, atol=1e-9
        )


def _weibull_aft_neg_ll(theta, x, c, Z):
    alpha, beta, b = theta[0], theta[1], theta[2:]
    phi = np.exp(Z @ b)
    u = phi * x / alpha
    log_df = np.log(beta / alpha * phi) + (beta - 1) * np.log(u) - u**beta
    return -np.sum(np.where(c == 0, log_df, -(u**beta)))


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_parametric_optimality_conditions(l1_ratio):
    # At the penalised optimum the gradient of (negative log-likelihood /
    # N + ridge) is 0 for the baseline, -l1 * sign(b) for a coefficient not
    # at 0, and at most l1 in size for one at 0 (the lasso's conditions),
    # on a likelihood written out here.
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(150, 6))
    t = 10 * rng.weibull(1.5, 150) * np.exp(-0.6 * Z[:, 0] + 0.3 * Z[:, 1])
    c = (t > 15).astype(int)
    x = np.minimum(t, 15)
    lam = 0.04
    model = WeibullAFT.fit_penalised(
        x, Z, c=c, penalizer=lam, l1_ratio=l1_ratio, standardise=False
    )
    assert model.maximum == "verified"
    theta = np.asarray(model.params, dtype=float)
    N = x.size
    h = 1e-6 * np.maximum(np.abs(theta), 1.0)
    g = np.array(
        [
            (
                _weibull_aft_neg_ll(theta + h[j] * e, x, c, Z)
                - _weibull_aft_neg_ll(theta - h[j] * e, x, c, Z)
            )
            / (2 * h[j] * N)
            for j, e in enumerate(np.eye(theta.size))
        ]
    )
    b = theta[2:]
    g[2:] += lam * (1 - l1_ratio) * b
    np.testing.assert_allclose(g[:2], 0, atol=1e-6)
    l1 = lam * l1_ratio
    nonzero = b != 0
    np.testing.assert_allclose(
        g[2:][nonzero], -l1 * np.sign(b[nonzero]), atol=1e-6
    )
    assert np.all(np.abs(g[2:][~nonzero]) <= l1 + 1e-6)
    # The two real effects are kept, with their signs.
    assert b[0] > 0 and b[1] < 0


def test_effective_df_and_information_criteria():
    x, c, Z = _rossi()
    lasso = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.05, l1_ratio=1.0)
    assert lasso.penalty.effective_df == pytest.approx(
        np.sum(lasso.params != 0)
    )
    ridge = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.05)
    assert 0 < ridge.penalty.effective_df < 7
    assert ridge.aic() == pytest.approx(
        2 * ridge.penalty.effective_df + 2 * ridge.neg_ll()
    )
    weibull = WeibullAFT.fit_penalised(x, Z, c=c, penalizer=0.05)
    assert 2 < weibull.penalty.effective_df < 9


def test_no_inference_and_saving():
    x, c, Z = _rossi()
    for fitter in (CoxPH, WeibullAFT):
        model = fitter.fit_penalised(x, Z, c=c, penalizer=0.05, l1_ratio=1)
        with pytest.raises(ValueError, match="fitted with a penalty"):
            model.covariance()
        with pytest.raises(ValueError, match="fitted with a penalty"):
            model.standard_errors()
        assert np.all(np.isnan(model.summary()["se(coef)"]))
        restored = sp.from_dict(model.to_dict())
        assert restored.penalty.l1_ratio == 1.0
        np.testing.assert_allclose(
            restored.sf([10.0, 30.0], Z[:2]), model.sf([10.0, 30.0], Z[:2])
        )
        with pytest.raises(ValueError, match="fitted with a penalty"):
            restored.covariance()
    with pytest.raises(ValueError, match="fitted with a penalty"):
        model.cb([10.0], Z[:1])
    with pytest.raises(ValueError, match="fitted with a penalty"):
        model.param_cb("alpha", method="lr")
    cox = CoxPH.fit_penalised(x, Z, c=c)
    with pytest.raises(ValueError, match="fitted with a penalty"):
        cox.robust_summary()


def test_refusals():
    x, c, Z = _rossi()
    with pytest.raises(ValueError, match="penalizer"):
        CoxPH.fit_penalised(x, Z, c=c, penalizer=-1)
    with pytest.raises(ValueError, match="one per covariate"):
        CoxPH.fit_penalised(x, Z, c=c, penalizer=[0.1, 0.2])
    with pytest.raises(ValueError, match="l1_ratio"):
        WeibullAFT.fit_penalised(x, Z, c=c, l1_ratio=2)
    with pytest.raises(ValueError, match="standardise"):
        WeibullAFT.fit_penalised(x, Z, c=c, standardise="yes")
    import autograd.numpy as anp

    def linear(Z, *params):
        return 1.0 + anp.dot(Z, anp.array(params))

    custom = sp.ProportionalHazardsFitter(
        "LinearRR",
        sp.Weibull,
        linear,
        "Linear [1 + beta'Z]",
        phi_bounds=lambda Z: ((0, None),) * Z.shape[1],
        phi_param_map=lambda Z: {f"beta_{i}": i for i in range(Z.shape[1])},
    )
    with pytest.raises(ValueError, match="custom phi"):
        custom.fit_penalised(x, Z[:, :1], c=c)


def test_from_a_dataframe():
    df = rossi_with_censoring()
    model = CoxPH.fit_penalised(
        x=df["week"], Z=df[COLS], c=df["censored"], penalizer=0.05
    )
    assert model.feature_names == COLS or model.feature_names is None
    np.testing.assert_allclose(
        model.params,
        CoxPH.fit_penalised(
            df["week"].to_numpy(),
            df[COLS].to_numpy(float),
            c=df["censored"].to_numpy(),
            penalizer=0.05,
        ).params,
    )
