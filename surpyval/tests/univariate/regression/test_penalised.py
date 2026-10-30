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

The stratified and start-stop (delayed entry) Cox references are glmnet
5.0 (R 4.5.3, survival 3.8.12) on the same path and threshold: the
stratified fit on the Rossi data stratified by ``wexp``,

.. code-block:: r

    cols <- c("fin", "age", "race", "mar", "paro", "prio")
    y <- stratifySurv(Surv(rossi$week, rossi$arrest), rossi$wexp)

and the start-stop fit on the Rossi person-week rows
(``load_rossi_time_varying``), whose ``N`` is the number of rows,

.. code-block:: r

    X <- cbind(fin = tv$fin == "yes", age = tv$age, prio = tv$prio,
               employed = tv$employed == "yes")
    y <- Surv(tv$start, tv$stop, tv$event)
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval import CoxPH, WeibullAFT, WeibullPH, WeibullPO
from surpyval.datasets import load_rossi_time_varying
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


STRATA_COLS = ["fin", "age", "race", "mar", "paro", "prio"]
# glmnet 5.0 with stratifySurv(..., wexp): (ties, lambda, alpha) -> the
# coefficients of STRATA_COLS
GLMNET_STRATA = {
    ("breslow", 0.05, 0.0): [
        -0.3185775381,
        -0.04285477551,
        0.23866585,
        -0.3625120598,
        -0.06467838826,
        0.0771738552,
    ],
    ("efron", 0.05, 0.5): [
        -0.1583552863,
        -0.02745214871,
        0,
        -0.07356925996,
        0,
        0.05434588815,
    ],
    ("efron", 0.02, 1.0): [
        -0.2114012855,
        -0.03622957166,
        0.04036774389,
        -0.1504041,
        0,
        0.06599474893,
    ],
    ("breslow", 0.02, 1.0): [
        -0.2109381591,
        -0.03605078685,
        0.03849294725,
        -0.1483179771,
        0,
        0.06566015655,
    ],
}
# glmnet 5.0 with Surv(start, stop, event) on the person-week rows:
# (ties, lambda, alpha) -> the coefficients of fin, age, prio, employed
GLMNET_START_STOP = {
    ("breslow", 0.001, 0.0): [
        -0.2856937991,
        -0.0408473614,
        0.07698696698,
        -1.069066555,
    ],
    ("efron", 0.001, 0.5): [
        -0.1521832308,
        -0.02851530175,
        0.06146276745,
        -0.9995899881,
    ],
    ("efron", 0.0005, 1.0): [
        -0.1629145004,
        -0.03062204964,
        0.06393156014,
        -1.117395676,
    ],
    ("breslow", 0.0005, 1.0): [
        -0.1622162333,
        -0.03046667939,
        0.06359260361,
        -1.114059698,
    ],
}


def _rossi_strata():
    df = rossi_with_censoring()
    Z = df[STRATA_COLS].to_numpy(dtype=float)
    return df["week"].to_numpy(float), df["censored"].to_numpy(), Z, df["wexp"]


def _rossi_start_stop():
    tv = load_rossi_time_varying()
    Z = np.column_stack(
        [tv["fin"] == "yes", tv["age"], tv["prio"], tv["employed"] == "yes"]
    ).astype(float)
    return (
        tv["start"].to_numpy(float),
        tv["stop"].to_numpy(float),
        1 - tv["event"].to_numpy(),
        Z,
    )


@pytest.mark.parametrize("setting", sorted(GLMNET_STRATA))
def test_stratified_cox_matches_glmnet(setting):
    ties, lam, alpha = setting
    x, c, Z, wexp = _rossi_strata()
    model = CoxPH.fit_penalised(
        x,
        Z,
        c=c,
        strata=wexp,
        penalizer=lam,
        l1_ratio=alpha,
        tie_method=ties,
    )
    assert model.maximum == "verified"
    assert model.is_stratified and model.strata_labels == [0, 1]
    ref = np.asarray(GLMNET_STRATA[setting])
    np.testing.assert_array_equal(model.params == 0, ref == 0)
    np.testing.assert_allclose(model.params, ref, rtol=1e-6, atol=1e-10)


def test_stratified_zero_penalty_is_the_stratified_fit():
    x, c, Z, wexp = _rossi_strata()
    pen = CoxPH.fit_penalised(x, Z, c=c, strata=wexp, penalizer=0.0)
    fit = CoxPH.fit(x, Z, c=c, strata=wexp)
    np.testing.assert_allclose(pen.params, fit.params, rtol=1e-7)
    # The same baseline in each stratum, and the same predictions.
    for s in (0, 1):
        np.testing.assert_allclose(
            pen.sf([10.0, 30.0, 50.0], Z[:3], stratum=s),
            fit.sf([10.0, 30.0, 50.0], Z[:3], stratum=s),
            rtol=1e-7,
        )
    assert pen.neg_ll() == pytest.approx(fit.neg_ll(), rel=1e-12)


def test_stratified_penalty_scale_is_over_every_stratum():
    # N is the units over all the strata, and the covariates are
    # standardised over all the rows, not within each stratum.
    x, c, Z, wexp = _rossi_strata()
    model = CoxPH.fit_penalised(x, Z, c=c, strata=wexp, penalizer=0.05)
    np.testing.assert_allclose(model.penalty.scale, Z.std(0))
    hand = CoxPH.fit_penalised(
        x, Z / Z.std(0), c=c, strata=wexp, penalizer=0.05, standardise=False
    )
    np.testing.assert_allclose(model.params, hand.params / Z.std(0))


def test_penalised_cox_polish_reaches_a_stationary_point():
    # The last Newton step changes the objective by less than its
    # rounding, so the polish must judge it by the gradient: the answer
    # is stationary to rounding, not to where L-BFGS-B stopped.
    x, c, Z, wexp = _rossi_strata()
    model = CoxPH.fit_penalised(
        x, Z / Z.std(0), c=c, strata=wexp, penalizer=0.05, standardise=False
    )
    N = float(x.size)
    beta = model.params
    grad = np.atleast_1d(model.jac(beta)[0]) / N + 0.05 * beta
    assert np.max(np.abs(grad)) < 1e-13


@pytest.mark.parametrize("setting", sorted(GLMNET_START_STOP))
def test_start_stop_cox_matches_glmnet(setting):
    # Delayed entry: each row enters the risk sets at its start.
    ties, lam, alpha = setting
    start, stop, c, Z = _rossi_start_stop()
    model = CoxPH.fit_penalised(
        stop,
        Z,
        c=c,
        tl=start,
        penalizer=lam,
        l1_ratio=alpha,
        tie_method=ties,
    )
    assert model.maximum == "verified"
    ref = np.asarray(GLMNET_START_STOP[setting])
    np.testing.assert_array_equal(model.params == 0, ref == 0)
    np.testing.assert_allclose(model.params, ref, rtol=1e-6, atol=1e-10)


def test_cox_truncation_as_t_or_data():
    # t (and data holding the truncation) with no upper bound is tl.
    x, c, Z = _rossi()
    tl = np.where(np.arange(x.size) % 3 == 0, 0.5 * x, -np.inf)
    by_tl = CoxPH.fit_penalised(x, Z, c=c, tl=tl, penalizer=0.05)
    t = np.column_stack([tl, np.full(x.size, np.inf)])
    by_t = CoxPH.fit_penalised(x, Z, c=c, t=t, penalizer=0.05)
    data = sp.SurpyvalData(x=x, c=c, tl=tl, Z=Z)
    by_data = CoxPH.fit_penalised(data, penalizer=0.05)
    np.testing.assert_array_equal(by_t.params, by_tl.params)
    np.testing.assert_array_equal(by_data.params, by_tl.params)
    assert not np.allclose(
        by_tl.params, CoxPH.fit_penalised(x, Z, c=c, penalizer=0.05).params
    )


def test_cox_refuses_right_truncation():
    x, c, Z = _rossi()
    tr = x + 10.0
    match = "parametric penalised fit"
    with pytest.raises(ValueError, match=match):
        CoxPH.fit_penalised(x, Z, c=c, tr=tr)
    with pytest.raises(ValueError, match=match):
        CoxPH.fit_penalised(x, Z, c=c, t=np.column_stack([0 * x, tr]))
    with pytest.raises(ValueError, match=match):
        CoxPH.fit_penalised(sp.SurpyvalData(x=x, c=c, tr=tr, Z=Z))
    with pytest.raises(ValueError, match="once"):
        CoxPH.fit_penalised(
            x, Z, c=c, tl=0 * x, t=np.column_stack([0 * x, tr])
        )


def _weibull_aft_neg_ll_truncated(theta, x, c, Z, tl, tr):
    alpha, beta, b = theta[0], theta[1], theta[2:]
    phi = np.exp(Z @ b)

    def sf(t):
        with np.errstate(invalid="ignore"):
            u = np.where(np.isfinite(t), phi * np.maximum(t, 0) / alpha, 0)
        out = np.exp(-(u**beta))
        return np.where(np.isposinf(t), 0.0, out)

    u = phi * x / alpha
    log_df = np.log(beta / alpha * phi) + (beta - 1) * np.log(u) - u**beta
    log_obs = np.where(c == 0, log_df, np.log(sf(x) - sf(tr)))
    return -np.sum(log_obs - np.log(sf(tl) - sf(tr)))


def _truncated_data():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(300, 4))
    t = 10 * rng.weibull(1.5, 300) * np.exp(-0.5 * Z[:, 0] + 0.3 * Z[:, 1])
    tl = rng.uniform(0, 3, 300)
    tr = tl + rng.uniform(8, 30, 300)
    keep = (t > tl) & (t < tr)
    x, Z, tl, tr = t[keep], Z[keep], tl[keep], tr[keep]
    c = (x > 12).astype(int)
    return np.minimum(x, 12.0), Z, c, tl, tr


TRUNCATIONS = ["left", "right", "both"]


def _truncation_kwargs(which, tl, tr):
    return {
        "left": {"tl": tl},
        "right": {"tr": tr},
        "both": {"tl": tl, "tr": tr},
    }[which]


@pytest.mark.parametrize("which", TRUNCATIONS)
@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_parametric_truncated_optimality_conditions(which, l1_ratio):
    # The optimality conditions, as in test_parametric_optimality_conditions,
    # of a truncated likelihood written out here.
    x, Z, c, tl, tr = _truncated_data()
    kwargs = _truncation_kwargs(which, tl, tr)
    lo = kwargs.get("tl", np.full(x.size, -np.inf))
    hi = kwargs.get("tr", np.full(x.size, np.inf))
    lam = 0.04
    model = WeibullAFT.fit_penalised(
        x,
        Z,
        c=c,
        penalizer=lam,
        l1_ratio=l1_ratio,
        standardise=False,
        **kwargs,
    )
    assert model.maximum == "verified"
    theta = np.asarray(model.params, dtype=float)
    N = x.size

    def f(th):
        return _weibull_aft_neg_ll_truncated(th, x, c, Z, lo, hi)

    assert f(theta) == pytest.approx(model.neg_ll(), rel=1e-10)
    h = 1e-6 * np.maximum(np.abs(theta), 1.0)
    g = np.array(
        [
            (f(theta + h[j] * e) - f(theta - h[j] * e)) / (2 * h[j] * N)
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


@pytest.mark.parametrize("which", TRUNCATIONS)
@pytest.mark.parametrize("fitter", [WeibullPH, WeibullAFT, WeibullPO])
def test_tiny_penalty_is_the_truncated_fit(fitter, which):
    # A negligible penalty reproduces the ordinary truncated maximum
    # likelihood fit: the same likelihood, and coefficients the same to a
    # small fraction of their standard errors (the ordinary fit stops at
    # its own tolerance, a little short of where the penalised solve does).
    x, Z, c, tl, tr = _truncated_data()
    kwargs = _truncation_kwargs(which, tl, tr)
    fit = fitter.fit(x, Z, c=c, **kwargs)
    pen = fitter.fit_penalised(x, Z, c=c, penalizer=1e-8, **kwargs)
    assert pen.maximum == "verified"
    assert pen.neg_ll() == pytest.approx(fit.neg_ll(), rel=1e-10)
    assert pen.neg_ll() <= fit.neg_ll() + 1e-9
    se = fit.standard_errors()
    assert np.all(np.abs(pen.params - fit.params) <= 1e-2 * se)


def test_refuses_weights():
    x, c, Z = _rossi()
    w = np.linspace(0.5, 2.0, x.size)
    data = sp.SurpyvalData(x=x, c=c, Z=Z, w=w)
    for fitter in (CoxPH, WeibullPH):
        with pytest.raises(ValueError, match="does not take weights"):
            fitter.fit_penalised(x, Z, c=c, w=w)
        with pytest.raises(ValueError, match="does not take weights"):
            fitter.fit_penalised(data)


def test_refuses_robust_errors_and_tests():
    x, c, Z = _rossi()
    cox = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.05)
    stratified = CoxPH.fit_penalised(
        x, Z[:, 1:], c=c, strata=Z[:, 0], penalizer=0.05
    )
    weibull = WeibullPH.fit_penalised(x, Z, c=c, penalizer=0.05)
    for model in (cox, stratified):
        with pytest.raises(ValueError, match="fitted with a penalty"):
            model.robust_covariance()
        with pytest.raises(ValueError, match="fitted with a penalty"):
            model.robust_covariance(cluster=np.arange(x.size) // 2)
        with pytest.raises(ValueError, match="fitted with a penalty"):
            model.summary(robust=True)
    with pytest.raises(ValueError, match="penalised fit"):
        weibull.robust_covariance()
    with pytest.raises(ValueError, match="penalised fit"):
        weibull.robust_standard_errors(cluster=np.arange(x.size) // 2)
    with pytest.raises(ValueError, match="penalised fit"):
        weibull.summary(robust=True)
    for model in (cox, stratified, weibull):
        with pytest.raises(ValueError, match="not available for a penalised"):
            model.likelihood_ratio_test()


@pytest.mark.parametrize("kind", ["cox", "stratified", "weibull"])
def test_information_criteria_count_the_effective_df(kind):
    x, c, Z = _rossi()
    if kind == "cox":
        model = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.05)
    elif kind == "stratified":
        model = CoxPH.fit_penalised(
            x, Z[:, 1:], c=c, strata=Z[:, 0], penalizer=0.05
        )
    else:
        model = WeibullPH.fit_penalised(x, Z, c=c, penalizer=0.05)
    k = model.penalty.effective_df
    n_events = float(np.sum(c == 0))
    nll = model.neg_ll()
    assert 0 < k < len(model.params)
    assert model.aic() == pytest.approx(2 * k + 2 * nll)
    assert model.bic() == pytest.approx(k * np.log(n_events) + 2 * nll)
    assert model.aic_c() == pytest.approx(
        2 * k + 2 * nll + 2 * k * (k + 1) / (n_events - k - 1)
    )


def test_time_unit_does_not_change_the_coefficients():
    # Principle 6: the coefficients are log ratios, so a penalty on them
    # does not depend on the unit of time (weeks or days here).
    x, c, Z = _rossi()
    for fitter in (CoxPH, WeibullPH, WeibullAFT):
        weeks = fitter.fit_penalised(x, Z, c=c, penalizer=0.03, l1_ratio=0.5)
        days = fitter.fit_penalised(
            7 * x, Z, c=c, penalizer=0.03, l1_ratio=0.5
        )
        np.testing.assert_array_equal(
            weeks.params[-7:] == 0, days.params[-7:] == 0
        )
        np.testing.assert_allclose(
            weeks.params[-7:], days.params[-7:], rtol=1e-6, atol=1e-9
        )


def _small():
    """120 Weibull lifetimes on two normal covariates, the first with a
    real effect, about a fifth right-censored (fixed seed)."""
    np.random.seed(7)
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(120, 2))
    x = sp.Weibull.random(120, 10, 2) * np.exp(-0.5 * Z[:, 0])
    c = (rng.uniform(size=120) < 0.2).astype(int)
    return x, Z, c


@pytest.mark.parametrize(
    "tie_method", ["efron", "breslow", "exact", "kalbfleisch-prentice"]
)
def test_zero_penalty_keeps_the_tie_method(tie_method):
    """audit R-C: with tied event times, a zero penalty is the unpenalised
    fit under the same tie method (Efron the default of both)."""
    x, Z, c = _small()
    x = np.round(x, 1)
    assert np.unique(x[c == 0]).size < np.sum(c == 0)  # there are ties
    pen = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.0, tie_method=tie_method)
    fit = CoxPH.fit(x, Z, c=c, tie_method=tie_method)
    np.testing.assert_allclose(pen.params, fit.params, rtol=1e-5)
    if tie_method == "efron":
        default = CoxPH.fit_penalised(x, Z, c=c, penalizer=0.0)
        np.testing.assert_array_equal(default.params, pen.params)
    # The tie methods give different answers on these data.
    if tie_method == "breslow":
        efron = CoxPH.fit(x, Z, c=c)
        assert not np.allclose(pen.params, efron.params, rtol=1e-5)


def test_unknown_tie_method_is_refused_by_name():
    """audit R-C: an unknown ``tie_method`` raises, naming it."""
    x, Z, c = _small()
    with pytest.raises(ValueError, match="'tie_method'.*'nope'"):
        CoxPH.fit_penalised(x, Z, c=c, tie_method="nope")


def _mixed_censoring_data():
    """Exact, right-, left- and interval-censored rows (no interval starts
    at 0), with counts 1, 2, 3 in turn."""
    rng = np.random.default_rng(5)
    N = 200
    Z = rng.normal(size=(N, 6))
    t = 10 * rng.weibull(1.5, N) * np.exp(-0.6 * Z[:, 0] + 0.3 * Z[:, 1])
    k = np.arange(N) % 5
    c = np.where(t > 15, 1, np.where(k == 1, -1, np.where(k == 2, 2, 0)))
    xl = np.where(
        c == 1,
        15.0,
        np.where(c == -1, 1.3 * t, np.where(c == 2, 0.7 * t, t)),
    )
    xr = np.where(c == 2, 1.4 * t, xl)
    n = 1 + np.arange(N) % 3
    return xl, xr, Z, c, n


def _weibull_aft_neg_ll_censored(theta, xl, xr, c, n, Z):
    """The Weibull AFT negative log-likelihood with every censoring kind
    and counts, written out with numpy."""
    alpha, beta, b = theta[0], theta[1], theta[2:]
    phi = np.exp(Z @ b)

    def sf(t):
        return np.exp(-((phi * t / alpha) ** beta))

    u = phi * xl / alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        log_df = np.log(beta / alpha * phi) + (beta - 1) * np.log(u) - u**beta
        ll = np.select(
            [c == 0, c == 1, c == -1, c == 2],
            [log_df, -(u**beta), np.log1p(-sf(xl)), np.log(sf(xl) - sf(xr))],
        )
    return -np.sum(n * ll)


@pytest.mark.parametrize("l1_ratio", [0.5, 1.0])
def test_parametric_optimality_conditions_every_censoring_and_counts(
    l1_ratio,
):
    """audit R-D: the optimality conditions of
    test_parametric_optimality_conditions hold for a fit with exact,
    right-, left- and interval-censored rows and counts ``n``, on a
    likelihood written out here (``N`` is the sum of the counts); and the
    counts give the fit on the rows repeated."""
    xl, xr, Z, c, n = _mixed_censoring_data()
    X = np.column_stack([xl, xr])
    lam = 0.04
    model = WeibullAFT.fit_penalised(
        X, Z, c=c, n=n, penalizer=lam, l1_ratio=l1_ratio, standardise=False
    )
    assert model.maximum == "verified"
    theta = np.asarray(model.params, dtype=float)

    def f(th):
        return _weibull_aft_neg_ll_censored(th, xl, xr, c, n, Z)

    assert f(theta) == pytest.approx(model.neg_ll(), rel=1e-10)
    N = n.sum()
    h = 1e-6 * np.maximum(np.abs(theta), 1.0)
    g = np.array(
        [
            (f(theta + h[j] * e) - f(theta - h[j] * e)) / (2 * h[j] * N)
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
    assert b[0] > 0 and b[1] < 0
    rep = np.repeat(np.arange(n.size), n)
    expanded = WeibullAFT.fit_penalised(
        X[rep],
        Z[rep],
        c=c[rep],
        penalizer=lam,
        l1_ratio=l1_ratio,
        standardise=False,
    )
    np.testing.assert_allclose(
        model.params, expanded.params, rtol=1e-6, atol=1e-9
    )


def test_interval_from_zero_is_taken_by_the_penalised_fit():
    """audit R-D: an interval that starts at 0 (a row known only to have
    failed by ``r``) is a valid row; the penalised fit takes it as the
    unpenalised fit does."""
    xl, xr, Z, c, n = _mixed_censoring_data()
    first = np.flatnonzero(c == 2)[0]
    xl, xr = xl.copy(), xr.copy()
    xl[first], xr[first] = 0.0, 1.0
    X = np.column_stack([xl, xr])
    WeibullAFT.fit(X, Z, c=c)  # the unpenalised fit takes it
    # The same row written as left-censored at r: F(r) - F(0) is F(r)
    left = X.copy()
    left[first, 0] = 1.0
    c_left = c.copy()
    c_left[first] = -1
    for l1_ratio in (0.0, 0.5, 1.0):
        model = WeibullAFT.fit_penalised(
            X, Z, c=c, penalizer=0.04, l1_ratio=l1_ratio, standardise=False
        )
        assert model.maximum == "verified"
        same = WeibullAFT.fit_penalised(
            left,
            Z,
            c=c_left,
            penalizer=0.04,
            l1_ratio=l1_ratio,
            standardise=False,
        )
        # Two runs of the solver on the rows in different orders: the
        # same maximum, each to the solver's tolerance
        np.testing.assert_allclose(model.neg_ll(), same.neg_ll(), rtol=1e-9)
        np.testing.assert_allclose(model.params, same.params, atol=1e-6)


def test_summary_of_a_penalised_fit_has_estimates_without_inference():
    """audit R-E: ``summary()`` of a penalised Cox and parametric fit
    shows the estimates, with nan standard errors, bounds and p-values."""
    x, Z, c = _small()
    for fitter in (CoxPH, WeibullAFT):
        model = fitter.fit_penalised(x, Z, c=c, penalizer=0.1)
        table = model.summary()
        if table.index.nlevels > 1:
            table = table.loc["coefficients"]
        np.testing.assert_array_equal(table["coef"], model.params[-2:])
        np.testing.assert_allclose(
            table["exp(coef)"], np.exp(model.params[-2:])
        )
        assert np.all(np.isfinite(table["coef"]))
        for column in table.columns:
            if column.startswith(("se(", "z", "p")) or " lower " in column:
                assert table[column].isna().all(), column
            if " upper " in column:
                assert table[column].isna().all(), column


def _refusal(call):
    with pytest.raises(ValueError) as info:
        call()
    return str(info.value)


@pytest.mark.parametrize("fitter", [CoxPH, WeibullAFT])
def test_penalised_round_trip_keeps_the_penalty_and_refusals(fitter):
    """audit R-U: strict JSON ``to_dict``/``from_dict`` and pickle keep the
    penalty, the information criteria and the predictions exactly, and the
    restored model refuses inference with the same messages."""
    import json
    import pickle

    x, Z, c = _small()
    model = fitter.fit_penalised(x, Z, c=c, penalizer=0.1, l1_ratio=0.5)
    text = json.dumps(model.to_dict(), allow_nan=False)
    restored = [
        sp.from_dict(json.loads(text)),
        pickle.loads(pickle.dumps(model)),
    ]
    refusals = {
        "covariance": lambda m: m.covariance(),
        "robust_covariance": lambda m: m.robust_covariance(),
        "likelihood_ratio_test": lambda m: m.likelihood_ratio_test(),
    }
    if fitter is not CoxPH:  # a Cox model has no cb at all
        refusals["cb"] = lambda m: m.cb([3.0], Z[:1])
    messages = {k: _refusal(lambda: f(model)) for k, f in refusals.items()}
    assert "fitted with a penalty" in messages["covariance"]
    assert "penalised fit" in messages["likelihood_ratio_test"]
    for back in restored:
        np.testing.assert_array_equal(
            back.penalty.penalizer, model.penalty.penalizer
        )
        assert back.penalty.l1_ratio == model.penalty.l1_ratio == 0.5
        assert back.penalty.effective_df == model.penalty.effective_df
        assert back.aic() == model.aic()
        np.testing.assert_array_equal(
            back.sf([3.0, 6.0], Z[:2]), model.sf([3.0, 6.0], Z[:2])
        )
        for key, call in refusals.items():
            assert _refusal(lambda: call(back)) == messages[key], key


def test_zero_and_huge_per_coefficient_penalties():
    """audit R-V: a coefficient with penalty 0 is not shrunk and one with a
    huge lasso penalty is exactly 0, so the other is the unpenalised fit
    on its own column (the objective in it is the unpenalised
    likelihood)."""
    x, Z, c = _small()
    cox = CoxPH.fit_penalised(x, Z, c=c, penalizer=[0.0, 50.0], l1_ratio=1.0)
    assert cox.params[1] == 0
    np.testing.assert_allclose(
        cox.params[0], CoxPH.fit(x, Z[:, :1], c=c).params[0], rtol=1e-5
    )
    weibull = WeibullPH.fit_penalised(
        x, Z, c=c, penalizer=[0.0, 50.0], l1_ratio=1.0
    )
    assert weibull.params[3] == 0
    np.testing.assert_allclose(
        weibull.params[:3],
        WeibullPH.fit(x, Z[:, :1], c=c).params,
        rtol=1e-4,
    )


def test_stratified_counts_equal_repetition_and_with_entry():
    """audit R-P: a stratified penalised Cox fit with counts ``n`` is the
    fit on the rows repeated; with strata and delayed entry together it
    runs to a verified answer with a finite effective df (no reference
    computes both together)."""
    x, Z, c = _small()
    strata = np.arange(x.size) % 2
    n = 1 + np.arange(x.size) % 3
    rep = np.repeat(np.arange(x.size), n)
    counted = CoxPH.fit_penalised(
        x, Z, c=c, n=n, strata=strata, penalizer=0.05, l1_ratio=0.5
    )
    expanded = CoxPH.fit_penalised(
        x[rep],
        Z[rep],
        c=c[rep],
        strata=strata[rep],
        penalizer=0.05,
        l1_ratio=0.5,
    )
    np.testing.assert_allclose(
        counted.params, expanded.params, rtol=1e-6, atol=1e-9
    )
    tl = np.where(np.arange(x.size) % 4 == 0, 0.3 * x, 0.0)
    model = CoxPH.fit_penalised(
        x, Z, c=c, strata=strata, tl=tl, penalizer=0.05
    )
    assert model.maximum == "verified"
    assert np.isfinite(model.penalty.effective_df)
    assert 0 < model.penalty.effective_df < 2
