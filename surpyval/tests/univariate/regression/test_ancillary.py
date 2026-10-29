"""Covariates on a baseline's other parameters: ``Ancillary``.

The references are R's flexsurv 2.3.2 (R 4.4.3) on the Rossi data,
``flexsurvreg(..., anc = list(...))``, which fits the same models:

.. code-block:: r

    library(flexsurv); library(survival)
    f <- flexsurvreg(Surv(week, arrest) ~ fin + age + prio,
                     anc = list(shape = ~ fin + age + prio), data = rossi,
                     dist = "weibull",
                     control = list(reltol = 1e-14, maxit = 10000))
    f$loglik; f$res.t[, "est"]; sqrt(diag(f$cov))

flexsurv's Weibull is ``S = exp(-(t / scale)^shape)`` with ``log(scale)``
linear in the covariates and ``log(shape)`` too, which is
``Ancillary(WeibullAFT, "beta")``: ``scale = alpha exp(-b'Z)`` (so
flexsurv's scale coefficients are SurPyval's with the sign changed) and
``shape = beta exp(d'Z)``. Its ``res.t`` are on the log scale for the two
positive parameters. Its log-normal is ``meanlog = mu - b'Z`` and
``log(sdlog)`` linear, ``Ancillary(LogNormalAFT, "sigma")``.
"""

import pickle

import numpy as np
import pytest

import surpyval as sp
from surpyval import Ancillary, LogNormalAFT, WeibullAFT
from surpyval.tests._helpers import rossi_with_censoring

COLS = ["fin", "age", "prio"]

# flexsurv 2.3.2, dist = "weibull", anc = list(shape = ~ fin + age + prio):
# res.t est: shape, scale (log), fin, age, prio, shape(fin, age, prio).
FLEXSURV_WEIBULL = dict(
    ll=-677.8448400538,
    est=[
        1.186267038362,
        2.481876880588,
        0.103467817295,
        0.110503872114,
        -0.072553460457,
        0.120655978393,
        -0.040214369712,
        0.007794810289,
    ],
    se=[
        0.3640077778,
        0.6250423654,
        0.1778268403,
        0.03029464833,
        0.02424394122,
        0.15860594,
        0.01433815331,
        0.02463362191,
    ],
)
# flexsurv 2.3.2, dist = "lnorm", anc = list(sdlog = ~ fin + age + prio):
# meanlog, sdlog (log), fin, age, prio, sdlog(fin, age, prio).
FLEXSURV_LNORM = dict(
    ll=-678.2025002535,
    est=[
        2.18872046712,
        -0.52794684656,
        0.16965599794,
        0.12726347588,
        -0.10746785592,
        -0.09208623026,
        0.04219701523,
        -0.04520569365,
    ],
)
# dist = "weibull", anc = list(shape = ~ age): the log-likelihood.
FLEXSURV_WEIBULL_AGE_LL = -678.2186895557


def _fit(fitter, parameter, **kw):
    df = rossi_with_censoring()
    return Ancillary(fitter, parameter).fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=COLS, **kw
    )


def test_weibull_shape_matches_flexsurv():
    model = _fit(WeibullAFT, "beta")
    assert model.maximum == "verified"
    assert model.parameter_names == [
        "alpha",
        "beta",
        "fin",
        "age",
        "prio",
        "beta_fin",
        "beta_age",
        "beta_prio",
    ]
    np.testing.assert_allclose(
        -model.neg_ll(), FLEXSURV_WEIBULL["ll"], atol=1e-6
    )
    p = np.asarray(model.params, dtype=float)
    ours = np.r_[np.log(p[1]), np.log(p[0]), -p[2:5], p[5:]]
    np.testing.assert_allclose(ours, FLEXSURV_WEIBULL["est"], rtol=1e-4)
    se = model.standard_errors()
    ours_se = np.r_[se[1] / p[1], se[0] / p[0], se[2:]]
    np.testing.assert_allclose(ours_se, FLEXSURV_WEIBULL["se"], rtol=1e-3)


def test_lognormal_sigma_is_flexsurvs_model():
    # The same model: at flexsurv's estimates the likelihood is
    # flexsurv's. flexsurv's optimiser stopped short of the maximum here,
    # which this fit reaches (6e-4 higher, verified).
    model = _fit(LogNormalAFT, "sigma")
    assert model.maximum == "verified"
    e = FLEXSURV_LNORM["est"]
    at_flexsurv = [e[0], np.exp(e[1]), *(-np.asarray(e[2:5])), *e[5:]]
    np.testing.assert_allclose(
        -model.model.neg_ll(model.data, *at_flexsurv),
        FLEXSURV_LNORM["ll"],
        atol=1e-8,
    )
    assert -model.neg_ll() >= FLEXSURV_LNORM["ll"]


def test_fixed_leaves_a_column_out():
    model = _fit(WeibullAFT, "beta", fixed={"beta_fin": 0.0, "beta_prio": 0.0})
    np.testing.assert_allclose(
        -model.neg_ll(), FLEXSURV_WEIBULL_AGE_LL, atol=1e-6
    )
    assert model.k == 6


def test_zero_ancillary_coefficients_are_the_plain_model():
    df = rossi_with_censoring()
    plain = WeibullAFT.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=COLS
    )
    fixed = {f"beta_{c}": 0.0 for c in COLS}
    model = _fit(WeibullAFT, "beta", fixed=fixed)
    np.testing.assert_allclose(model.neg_ll(), plain.neg_ll(), rtol=1e-9)
    np.testing.assert_allclose(model.params[:5], plain.params, rtol=1e-4)
    Z = df[COLS].to_numpy(float)[:3]
    np.testing.assert_allclose(
        model.sf([10.0, 30.0, 52.0], Z),
        plain.sf([10.0, 30.0, 52.0], Z),
        rtol=1e-5,
    )


def test_predictions_saving_and_bounds():
    model = _fit(WeibullAFT, "beta")
    Z = np.array([[0.0, 20.0, 1.0], [1.0, 35.0, 4.0]])
    x = np.array([10.0, 52.0])
    # The shape is row by row: the survival is the Weibull's with each
    # row's scale and shape.
    p = model.params
    scale = p[0] * np.exp(-Z @ p[2:5])
    shape = p[1] * np.exp(Z @ p[5:])
    np.testing.assert_allclose(
        model.sf(x, Z), np.exp(-((x / scale) ** shape)), rtol=1e-12
    )
    q = model.qf([0.25, 0.5], Z)
    np.testing.assert_allclose(model.ff(q, Z), [0.25, 0.5], rtol=1e-9)
    band = model.cb(x, Z)
    sf = model.sf(x, Z)
    assert np.all(band[:, 0] < sf) and np.all(sf < band[:, 1])
    lr = model.param_cb("beta_age", method="lr")
    assert lr[0] < p[6] < lr[1]
    table = model.summary()
    assert list(table.loc["coefficients"].index) == model.parameter_names[2:]
    assert np.all(np.isfinite(table.loc["coefficients"]["p"]))
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.sf(x, Z), sf, rtol=1e-12)
    np.testing.assert_allclose(restored.cb(x, Z), band, rtol=1e-9)
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(again.sf(x, Z), sf)


def test_refusals():
    with pytest.raises(ValueError, match="already move"):
        Ancillary(WeibullAFT, "alpha")
    with pytest.raises(ValueError, match="Unknown parameter"):
        Ancillary(WeibullAFT, "shape")
    with pytest.raises(ValueError, match="at least one parameter"):
        Ancillary(WeibullAFT, [])
    with pytest.raises(ValueError, match="AFT fitter"):
        Ancillary(sp.WeibullPH, "beta")
    with pytest.raises(ValueError, match="AFT fitter"):
        Ancillary(sp.Weibull, "beta")
    with pytest.raises(ValueError, match="already move"):
        Ancillary(sp.NormalAFT, "sigma")
