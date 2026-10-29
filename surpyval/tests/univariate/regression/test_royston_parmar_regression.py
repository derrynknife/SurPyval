"""Royston-Parmar (spline baseline) regression: ``RoystonParmarPH``,
``RoystonParmarPO`` and ``RoystonParmarRegression``.

The references are R's flexsurv 2.3.2 (R 4.4.3), whose
``flexsurvspline`` fits the same model on the same knots (quantiles of the
log event times) and the same restricted cubic spline basis, on the Rossi
data:

.. code-block:: r

    library(flexsurv); library(survival)
    f <- flexsurvspline(Surv(week, arrest) ~ fin + age + prio, data = rossi,
                        k = 2, scale = "hazard",
                        control = list(reltol = 1e-14, maxit = 10000))
    f$loglik; f$knots; f$coefficients; sqrt(diag(f$cov))
    summary(f, newdata = data.frame(fin = c(0, 1), age = c(25, 40),
                                    prio = c(3, 0)),
            t = c(10, 52), type = "survival", ci = FALSE)

``k`` internal knots is ``df = k + 1``. On the odds scale flexsurv's
covariates add to the log odds of failure, SurPyval's multiply the
survival odds (as every ``PO`` model), so the coefficients have the
opposite sign.
"""

import pickle

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    RoystonParmarPH,
    RoystonParmarPO,
    RoystonParmarRegression,
    WeibullPH,
)
from surpyval.tests._helpers import rossi_with_censoring

COLS = ["fin", "age", "prio"]
# flexsurv 2.3.2: (scale, k) -> log-likelihood, knots, coefficients
# (gamma0, ..., fin, age, prio), standard errors, and the survival at 10
# and 52 weeks for (fin, age, prio) = (0, 25, 3) and (1, 40, 0).
FLEXSURV = {
    ("hazard", 0): dict(
        ll=-682.0412779767,
        knots=[0, 3.95124371858],
        coef=[
            -5.284672096,
            1.40036983,
            -0.3493973567,
            -0.06689164724,
            0.09774171662,
        ],
        se=[
            0.6922486593,
            0.1249811247,
            0.190255076,
            0.02083637789,
            0.02731716316,
        ],
        sf=[0.9684211795, 0.7240750181, 0.9938318522, 0.9396442078],
    ),
    ("hazard", 2): dict(
        ll=-681.7225895507,
        knots=[0, 2.99573227355, 3.61091791264, 3.95124371858],
        coef=[
            -5.918051001,
            1.789159383,
            0.2579708973,
            -0.5164031954,
            -0.3488359566,
            -0.06697865951,
            0.09735827116,
        ],
        se=[
            1.246703839,
            0.5836815636,
            0.3461626596,
            0.7545243816,
            0.1902533314,
            0.02084633261,
            0.02729851733,
        ],
        sf=[0.9671562286, 0.7241594649, 0.9935788104, 0.9396415592],
    ),
    ("odds", 0): dict(
        ll=-682.9318000381,
        knots=[0, 3.95124371858],
        coef=[
            -5.599427264,
            1.53106397,
            -0.3953017802,
            -0.0696824593,
            0.1137198573,
        ],
        se=[
            0.7435026355,
            0.1328910767,
            0.220863855,
            0.02268927401,
            0.03326664811,
        ],
        sf=[0.9699658006, 0.7212604116, 0.9948141509, 0.9389128152],
    ),
    ("odds", 2): dict(
        ll=-682.5411458792,
        knots=[0, 2.99573227355, 3.61091791264, 3.95124371858],
        coef=[
            -5.893008041,
            1.771359743,
            0.3159486125,
            -0.7206607974,
            -0.3957416423,
            -0.07019715088,
            0.1132281811,
        ],
        se=[
            1.278017939,
            0.5921487877,
            0.3710178496,
            0.8206603392,
            0.2208522884,
            0.02272541504,
            0.03323243764,
        ],
        sf=[0.9678456111, 0.7203361699, 0.9944748646, 0.9390327304],
    ),
}
NEW = np.array([[0.0, 25.0, 3.0], [1.0, 40.0, 0.0]])


def _fit(scale, k):
    df = rossi_with_censoring()
    fitter = RoystonParmarRegression(df=k + 1, scale=scale)
    return fitter.fit_from_df(df, x_col="week", c_col="censored", Z_cols=COLS)


@pytest.mark.parametrize("setting", sorted(FLEXSURV))
def test_matches_flexsurvspline(setting):
    scale, k = setting
    ref = FLEXSURV[setting]
    model = _fit(scale, k)
    assert model.maximum == "verified"
    np.testing.assert_allclose(
        model.model.dist.knots, ref["knots"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(-model.neg_ll(), ref["ll"], atol=1e-6)
    coef = np.asarray(model.params, dtype=float)
    sign = np.ones(coef.size)
    if scale == "odds":
        sign[k + 2 :] = -1.0  # PO multiplies the survival odds
    np.testing.assert_allclose(coef * sign, ref["coef"], rtol=1e-4)
    np.testing.assert_allclose(model.standard_errors(), ref["se"], rtol=1e-3)
    x = np.array([10.0, 52.0])
    sf = np.concatenate([model.sf(x, NEW[[i, i]]) for i in range(2)])
    np.testing.assert_allclose(sf, ref["sf"], rtol=1e-6)


def test_one_degree_of_freedom_is_the_weibull():
    # df = 1 on the hazard scale: log H = gamma_0 + gamma_1 log t + b'Z,
    # the Weibull proportional hazards model.
    df = rossi_with_censoring()
    spline = RoystonParmarRegression(df=1).fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=COLS
    )
    weibull = WeibullPH.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=COLS
    )
    np.testing.assert_allclose(spline.neg_ll(), weibull.neg_ll(), rtol=1e-9)
    np.testing.assert_allclose(
        spline.params[2:], weibull.params[2:], rtol=1e-4
    )
    x = np.array([3.0, 20.0, 52.0])
    np.testing.assert_allclose(
        spline.sf(x, NEW[[0, 1, 1]]), weibull.sf(x, NEW[[0, 1, 1]]), rtol=1e-5
    )


def test_every_censoring_kind_and_truncation():
    # Interval, left and right censoring with left truncation: the fit
    # reaches a verified maximum and its likelihood is the univariate
    # spline's when the coefficients are 0.
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(200, 1))
    t = 8 * rng.weibull(1.7, 200) * np.exp(-0.3 * Z[:, 0])
    x = np.column_stack([t, t])
    c = np.zeros(200, int)
    c[::5] = 1
    c[1::5] = 2
    # Inspections around each time (intervals that only meet at one time
    # would have no finite maximum, #392).
    x[1::5, 0] = 0.8 * t[1::5]
    x[1::5, 1] = 1.25 * t[1::5]
    left = (np.arange(200) % 9 == 2) & (c == 0)
    c[left] = -1
    keep = t > 0.5
    model = RoystonParmarPH.fit(
        x[keep], Z[keep], c=c[keep], tl=np.full(keep.sum(), 0.5)
    )
    assert model.maximum == "verified"
    assert model.params[-1] > 0
    uni = sp.RoystonParmar.fit(
        x=x[keep],
        c=c[keep],
        tl=0.5,
        knots=model.model.dist.knots,
    )
    zeroed = np.append(np.asarray(uni.params), 0.0)
    np.testing.assert_allclose(
        model.model.neg_ll(model.data, *zeroed), uni.neg_ll(), rtol=1e-9
    )


def test_bounds_saving_and_pickling():
    model = _fit("hazard", 2)
    lr = model.param_cb("prio", method="lr")
    wald = model.param_cb("prio")
    assert lr[0] < model.params[-1] < lr[1]
    assert np.all(np.isfinite(lr)) and np.all(np.isfinite(wald))
    band = model.cb([10.0, 52.0], NEW, on="sf")
    sf = model.sf([10.0, 52.0], NEW)
    assert np.all(band[:, 0] < sf) and np.all(sf < band[:, 1])
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.sf([10.0, 52.0], NEW), sf, rtol=1e-12)
    np.testing.assert_allclose(restored.cb([10.0, 52.0], NEW), band, rtol=1e-9)
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(again.sf([10.0, 52.0], NEW), sf)
    q = model.qf([0.1, 0.2], NEW)
    np.testing.assert_allclose(model.ff(q, NEW), [0.1, 0.2], rtol=1e-9)


def test_knots_given_and_refusals():
    df = rossi_with_censoring()
    knots = np.log([5.0, 20.0, 52.0])
    model = RoystonParmarRegression(knots=knots).fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=COLS
    )
    np.testing.assert_allclose(model.model.dist.knots, knots)
    with pytest.raises(ValueError, match="neither proportional hazards"):
        RoystonParmarRegression(scale="normal")
    with pytest.raises(ValueError, match="scale"):
        RoystonParmarRegression(scale="probit")
    with pytest.raises(ValueError, match="`df`"):
        RoystonParmarRegression(df=0)
    with pytest.raises(ValueError, match="distinct event times"):
        RoystonParmarRegression(df=6).fit([1.0, 2.0, 3.0], [[0], [1], [0]])
    assert RoystonParmarPO.scale == "odds"
