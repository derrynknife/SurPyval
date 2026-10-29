"""Distributional regression: covariates on every parameter.

The flexsurv references are R's flexsurv 2.3.2 (R 4.4.3) on the Rossi
data, ``flexsurvreg(..., anc = list(...))``, which fits the same model:

.. code-block:: r

    library(flexsurv); library(survival)
    f <- flexsurvreg(Surv(week, arrest) ~ fin + age + prio,
                     anc = list(shape = ~ fin + age + prio), data = rossi,
                     dist = "weibull",
                     control = list(reltol = 1e-14, maxit = 10000))
    f$loglik; f$res.t[, "est"]; sqrt(diag(f$cov))

flexsurv's Weibull is ``S = exp(-(t / scale)^shape)`` with ``log(scale)``
and ``log(shape)`` linear in the covariates: SurPyval's ``alpha`` and
``beta`` with their default log links and the same signs. Its ``res.t``
holds ``log(shape)`` and ``log(scale)`` for the intercepts. Its log-normal
is ``meanlog`` linear and ``log(sdlog)`` linear: ``mu`` and ``sigma``.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import Distributional, WeibullDistributional
from surpyval.tests._helpers import rossi_with_censoring
from surpyval.univariate.regression.tvc_schedule import StepSchedule

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


def _rossi(fitter=WeibullDistributional, **kw):
    df = rossi_with_censoring()
    return fitter.fit_from_df(
        df, x_col="week", c_col="censored", Z_cols=COLS, **kw
    )


def _load_data(n=300, seed=3):
    """Times whose Weibull scale and shape both depend on a load."""
    rng = np.random.default_rng(seed)
    load = rng.uniform(0, 1, n)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
    return x, load


# -- references ------------------------------------------------------------


def test_weibull_matches_flexsurv():
    model = _rossi()
    assert model.maximum == "verified"
    assert model.parameter_names == [
        "alpha",
        "beta",
        "alpha_fin",
        "alpha_age",
        "alpha_prio",
        "beta_fin",
        "beta_age",
        "beta_prio",
    ]
    np.testing.assert_allclose(
        -model.neg_ll(), FLEXSURV_WEIBULL["ll"], atol=1e-6
    )
    p = np.asarray(model.params, dtype=float)
    ours = np.r_[np.log(p[1]), np.log(p[0]), p[2:]]
    np.testing.assert_allclose(ours, FLEXSURV_WEIBULL["est"], rtol=1e-4)
    se = model.standard_errors()
    ours_se = np.r_[se[1] / p[1], se[0] / p[0], se[2:]]
    np.testing.assert_allclose(ours_se, FLEXSURV_WEIBULL["se"], rtol=1e-3)


def test_lognormal_is_flexsurvs_model():
    # At flexsurv's estimates the likelihood is flexsurv's; flexsurv's
    # optimiser stopped short of the maximum, which this fit reaches.
    model = _rossi(sp.LogNormalDistributional)
    assert model.maximum == "verified"
    e = FLEXSURV_LNORM["est"]
    at_flexsurv = [e[0], np.exp(e[1]), *e[2:]]
    np.testing.assert_allclose(
        -model.model.neg_ll(model.data, *at_flexsurv),
        FLEXSURV_LNORM["ll"],
        atol=1e-8,
    )
    assert -model.neg_ll() >= FLEXSURV_LNORM["ll"]


def test_formulas_choose_each_parameters_terms():
    df = rossi_with_censoring()
    model = WeibullDistributional.fit_from_df(
        df,
        x_col="week",
        c_col="censored",
        formulas={"alpha": "fin + age + prio", "beta": "age"},
    )
    assert model.parameter_names == [
        "alpha",
        "beta",
        "alpha_fin",
        "alpha_age",
        "alpha_prio",
        "beta_age",
    ]
    np.testing.assert_allclose(
        -model.neg_ll(), FLEXSURV_WEIBULL_AGE_LL, atol=1e-6
    )
    # The same by column names, and by a coefficient held at 0
    by_name = _rossi(covariates={"alpha": COLS, "beta": ["age"]})
    np.testing.assert_allclose(by_name.params, model.params, rtol=1e-6)
    held = _rossi(fixed={"beta_fin": 0.0, "beta_prio": 0.0})
    np.testing.assert_allclose(
        -held.neg_ll(), FLEXSURV_WEIBULL_AGE_LL, atol=1e-6
    )
    assert held.k == 6
    # Predictions from a DataFrame go through the formula
    rows = df.head(3)
    np.testing.assert_allclose(
        model.sf([10.0, 20.0, 30.0], rows),
        model.sf([10.0, 20.0, 30.0], rows[COLS].to_numpy(float)),
    )


# -- the special cases ------------------------------------------------------


@pytest.mark.parametrize(
    "dist, aft, parameter, sign",
    [
        ("Weibull", "WeibullAFT", "alpha", -1.0),
        ("Exponential", "ExponentialAFT", "failure_rate", 1.0),
        ("LogNormal", "LogNormalAFT", "mu", -1.0),
        ("LogLogistic", None, "alpha", -1.0),
        ("Gamma", "GammaAFT", "beta", 1.0),
    ],
)
def test_covariates_on_the_scale_alone_are_the_aft_model(
    dist, aft, parameter, sign
):
    df = rossi_with_censoring()
    Z = df[["fin", "age"]].to_numpy(float)
    x, c = df["week"].to_numpy(float), df["censored"].to_numpy()
    distribution = getattr(sp, dist)
    aft_fitter = getattr(sp, aft) if aft else sp.AFT(distribution)
    plain = aft_fitter.fit(x, Z, c=c)
    model = Distributional(distribution).fit(
        x, Z, c=c, covariates={parameter: [0, 1]}
    )
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.neg_ll(), plain.neg_ll(), rtol=1e-9)
    k = model.k_dist
    np.testing.assert_allclose(model.params[:k], plain.params[:k], rtol=1e-5)
    np.testing.assert_allclose(
        model.params[k:], sign * plain.params[k:], rtol=1e-5
    )
    np.testing.assert_allclose(
        model.standard_errors(), plain.standard_errors(), rtol=1e-4
    )
    t = np.array([10.0, 30.0, 52.0])
    np.testing.assert_allclose(model.sf(t, Z[:3]), plain.sf(t, Z[:3]), 1e-6)


def test_weibull_ph_is_covariates_on_the_scale():
    # H = (x / alpha)^beta e^(b z) = (x / (alpha e^(-b z / beta)))^beta
    df = rossi_with_censoring()
    x, c = df["week"].to_numpy(float), df["censored"].to_numpy()
    z = df["fin"].to_numpy(float)
    ph = sp.WeibullPH.fit(x, z, c=c)
    model = WeibullDistributional.fit(x, z, c=c, covariates={"alpha": [0]})
    np.testing.assert_allclose(model.neg_ll(), ph.neg_ll(), rtol=1e-9)
    alpha, beta, alpha_z = model.params
    np.testing.assert_allclose(-beta * alpha_z, ph.params[2], rtol=1e-5)


def test_power_life_model_is_covariates_on_log_stress():
    rng = np.random.default_rng(11)
    stress = np.repeat([20.0, 30.0, 40.0], 40)
    x = sp.Weibull.random(120, 10, 3, random_state=rng) * (100.0 / stress)
    al = sp.AcceleratedLife(sp.Weibull, sp.life_models.Power).fit(x, Z=stress)
    model = WeibullDistributional.fit(
        x, np.log(stress), covariates={"alpha": [0]}
    )
    np.testing.assert_allclose(model.neg_ll(), al.neg_ll(), rtol=1e-8)
    np.testing.assert_allclose(
        model.sf([5.0, 9.0], np.log([[25.0], [35.0]])),
        al.sf([5.0, 9.0], [[25.0], [35.0]]),
        rtol=1e-5,
    )


def test_exponential_is_the_weibull_with_its_shape_fixed_at_1():
    x, load = _load_data()
    weibull = WeibullDistributional.fit(
        x, load, covariates={"alpha": [0]}, fixed={"beta": 1.0}
    )
    expo = sp.ExponentialDistributional.fit(x, load)
    np.testing.assert_allclose(weibull.neg_ll(), expo.neg_ll(), rtol=1e-9)
    # Two searches (the Weibull's at Z = 0, its shape held there), each to
    # its own tolerance
    np.testing.assert_allclose(1 / weibull.params[0], expo.params[0], 1e-5)
    np.testing.assert_allclose(weibull.params[2], -expo.params[1], 1e-5)


# -- the likelihood on every kind of data ------------------------------------


def _by_hand(model, x, c, Z, tl=None):
    """The log-likelihood written out from each row's distribution."""
    dist = model.distribution
    ll = 0.0
    params = model.parameters_at(Z)
    for j in range(len(c)):
        p = params[j]
        if c[j] == 0:
            ll += np.log(dist.df(x[j][0], *p))
        elif c[j] == 1:
            ll += np.log(dist.sf(x[j][0], *p))
        elif c[j] == -1:
            # Below x and above the truncation: (tl, x]
            low = 0.0 if tl is None else dist.ff(tl[j], *p)
            ll += np.log(dist.ff(x[j][0], *p) - low)
        else:
            ll += np.log(dist.ff(x[j][1], *p) - dist.ff(x[j][0], *p))
        if tl is not None:
            ll -= np.log(dist.sf(tl[j], *p))
    return ll


def test_every_censoring_and_truncation():
    x, load = _load_data(120, seed=5)
    rng = np.random.default_rng(6)
    c = rng.choice([0, 1, -1, 2], size=x.size, p=[0.5, 0.2, 0.1, 0.2])
    xx = np.c_[x, x]
    xx[c == 2, 1] = x[c == 2] * 1.5
    tl = np.where(rng.uniform(size=x.size) < 0.3, 0.2 * x, 0.0)
    for dist in ("Weibull", "LogNormal", "Gamma"):
        model = Distributional(getattr(sp, dist)).fit(xx, load, c=c, tl=tl)
        assert model.maximum == "verified"
        ll = _by_hand(model, xx, c, load.reshape(-1, 1), tl)
        np.testing.assert_allclose(-model.neg_ll(), ll, rtol=1e-10)


def test_counts_are_repetition_and_order_does_not_matter():
    x, load = _load_data(60)
    n = np.tile([1, 2, 3], 20)
    model = WeibullDistributional.fit(x, load, n=n)
    rep = WeibullDistributional.fit(np.repeat(x, n), np.repeat(load, n))
    np.testing.assert_allclose(model.params, rep.params, rtol=1e-6)
    order = np.random.default_rng(1).permutation(x.size)
    perm = WeibullDistributional.fit(x[order], load[order], n=n[order])
    np.testing.assert_allclose(model.params, perm.params, rtol=1e-12)


def test_covariate_origin_and_scale():
    x, load = _load_data()
    model = WeibullDistributional.fit(x, load)
    # A year-like origin: the same model at the means
    far = WeibullDistributional.fit(x, load + 2000.0, center=True)
    near = WeibullDistributional.fit(x, load, center=True)
    np.testing.assert_allclose(far.params, near.params, rtol=1e-6)
    np.testing.assert_allclose(far.center, near.center + 2000.0)
    np.testing.assert_allclose(
        near.sf([5.0], [[0.3]]), model.sf([5.0], [[0.3]]), rtol=1e-7
    )
    # Scale: the coefficients divide, nothing else moves
    for factor in (1e-6, 731.0):
        scaled = WeibullDistributional.fit(x, load * factor)
        assert scaled.maximum == "verified"
        np.testing.assert_allclose(
            scaled.params[2:] * factor, model.params[2:], rtol=1e-5
        )
        np.testing.assert_allclose(scaled.neg_ll(), model.neg_ll(), 1e-9)


def test_a_year_at_z_0_is_refused_with_advice():
    x, load = _load_data()
    with pytest.raises(ValueError, match="center=True"):
        WeibullDistributional.fit(x, load + 1e4)


# -- aliasing and no finite maximum -----------------------------------------


def test_a_repeated_column_is_aliased_on_its_parameter_only():
    x, load = _load_data()
    other = np.random.default_rng(2).uniform(size=x.size)
    Z = np.c_[load, other, load]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = WeibullDistributional.fit(
            x, Z, covariates={"alpha": [0, 1], "beta": [0, 2]}
        )
    found = [w for w in caught if "cannot be estimated" in str(w.message)]
    assert len(found) == 1
    assert str(found[0].message).startswith("Covariate column(s) 2 of Z")
    names = model.parameter_names
    assert np.isnan(model.params[names.index("beta_coef_2")])
    assert np.isfinite(model.params[names.index("alpha_coef_1")])
    np.testing.assert_array_equal(model.aliased, [2])
    assert np.isnan(model.standard_errors()[names.index("beta_coef_2")])
    plain = WeibullDistributional.fit(
        x, Z[:, :2], covariates={"alpha": [0, 1], "beta": [0]}
    )
    np.testing.assert_allclose(model.neg_ll(), plain.neg_ll(), rtol=1e-10)


def test_a_level_with_failures_at_one_time_has_no_finite_shape():
    rng = np.random.default_rng(4)
    a = sp.Weibull.random(40, 10, 2, random_state=rng)
    b = np.full(20, 7.0)
    x = np.r_[a, b]
    z = np.r_[np.zeros(40), np.ones(20)]
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = WeibullDistributional.fit(x, z)
    assert model.maximum == "no finite maximum"


def test_a_starved_search_is_not_silent():
    x, load = _load_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = WeibullDistributional.fit(
            x, load, init=[1e6, 1e-3, 30.0, -30.0]
        )
    if model.maximum != "verified":
        assert caught
    ref = WeibullDistributional.fit(x, load)
    if model.maximum == "verified":
        np.testing.assert_allclose(model.params, ref.params, rtol=1e-5)


# -- refusals ---------------------------------------------------------------


def test_refusals_say_what_to_do():
    with pytest.raises(ValueError, match="not available for the Uniform"):
        Distributional(sp.Uniform)
    with pytest.raises(ValueError, match="parametric distribution"):
        Distributional(sp.WeibullAFT)
    with pytest.raises(ValueError, match="Use one of"):
        Distributional(sp.Weibull, links={"beta": "identity"})
    with pytest.raises(ValueError, match="Unknown parameter"):
        Distributional(sp.Weibull, links={"shape": "log"})
    x, load = _load_data(50)
    with pytest.raises(ValueError, match="Unknown parameter"):
        WeibullDistributional.fit(x, load, covariates={"shape": [0]})
    with pytest.raises(ValueError, match="column numbers"):
        WeibullDistributional.fit(x, load, covariates={"beta": ["load"]})
    with pytest.raises(ValueError, match="has column 3"):
        WeibullDistributional.fit(x, load, covariates={"beta": [3]})
    with pytest.raises(ValueError, match="more than once"):
        WeibullDistributional.fit(x, load, covariates={"beta": [0, 0]})
    df = pd.DataFrame({"x": x, "load": load})
    with pytest.raises(ValueError, match="alone"):
        WeibullDistributional.fit_from_df(
            df, x_col="x", Z_cols=["load"], formulas={"beta": "load"}
        )
    with pytest.raises(ValueError, match="intercept-only"):
        WeibullDistributional.fit_from_df(
            df, x_col="x", formulas={"beta": "1"}
        )
    model = WeibullDistributional.fit(x, load)
    with pytest.raises(NotImplementedError, match="parameters_at"):
        model.phi([[0.5]])
    with pytest.raises(NotImplementedError):
        model.sf_tvc([1.0], StepSchedule.constant(np.array([0.5])))


def test_a_log_link_on_a_location():
    rng = np.random.default_rng(8)
    z = rng.uniform(0, 1, 200)
    x = rng.normal(20 * np.exp(0.3 * z), 2.0)
    model = sp.NormalDistributional.fit(
        x, z, covariates={"mu": [0]}, links={"mu": "log"}
    )
    assert model.links == ("log", "log")
    np.testing.assert_allclose(model.params[2], 0.3, atol=0.05)


# -- predictions, bounds and saving -----------------------------------------


def test_predictions():
    x, load = _load_data()
    model = WeibullDistributional.fit(x, load)
    Z = np.array([[0.1], [0.9]])
    alpha, beta = model.parameters_at(Z).T
    a0, b0, a1, b1 = model.params
    np.testing.assert_allclose(alpha, a0 * np.exp(a1 * Z[:, 0]))
    np.testing.assert_allclose(beta, b0 * np.exp(b1 * Z[:, 0]))
    t = np.array([3.0, 8.0])
    np.testing.assert_allclose(
        model.sf(t, Z), np.exp(-((t / alpha) ** beta)), rtol=1e-12
    )
    q = model.qf([0.1, 0.5], Z)
    np.testing.assert_allclose(model.ff(q, Z), [0.1, 0.5], rtol=1e-10)
    np.testing.assert_allclose(
        model.mean(Z), alpha * sp.Weibull.mean(1, beta) / 1, rtol=1e-10
    )
    one = model.distribution_at([0.9])
    np.testing.assert_allclose(one.sf(8.0), model.sf(8.0, [0.9]), 1e-12)
    grid = model.qf([0.1, 0.5, 0.9], Z, grid=True)
    assert grid.shape == (2, 3)
    draws, rows = model.random(4, Z, random_state=1)
    assert draws.shape == (8,) and rows.shape == (8, 1)
    # Given an age, the survival of what remains
    np.testing.assert_allclose(
        model.cs(2.0, 3.0, [0.9]),
        model.sf(5.0, [0.9]) / model.sf(3.0, [0.9]),
        rtol=1e-10,
    )


def test_bounds_saving_and_pickle():
    x, load = _load_data()
    model = WeibullDistributional.fit(x, load)
    Z = np.array([[0.2], [0.8]])
    t = np.array([4.0, 6.0])
    sf = model.sf(t, Z)
    band = model.cb(t, Z)
    assert np.all(band[:, 0] < sf) and np.all(sf < band[:, 1])
    lr = model.param_cb("beta_coef_0", method="lr")
    assert lr[0] < model.params[3] < lr[1]
    boot = model.param_cb(
        "beta_coef_0", method="bootstrap", n_boot=30, random_state=1
    )
    assert boot[0] < model.params[3] < boot[1]
    table = model.summary()
    assert list(table.loc["coefficients"].index) == [
        "alpha_coef_0",
        "beta_coef_0",
    ]
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.sf(t, Z), sf, rtol=1e-14)
    np.testing.assert_allclose(restored.cb(t, Z), band, rtol=1e-9)
    np.testing.assert_allclose(restored.qf(0.3, Z), model.qf(0.3, Z))
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(again.cb(t, Z), band, rtol=1e-12)


def test_spline_terms_save_and_predict():
    x, load = _load_data()
    df = pd.DataFrame({"x": x, "load": load})
    model = WeibullDistributional.fit_from_df(
        df,
        x_col="x",
        formulas={"alpha": "load", "beta": "bs(load, df=3)"},
    )
    assert model.maximum == "verified"
    assert len(model.covariate_positions[1]) == 3
    query = pd.DataFrame({"load": [0.2, 0.5, 0.8]})
    restored = sp.from_dict(model.to_dict())
    np.testing.assert_allclose(
        restored.sf([5.0, 5.0, 5.0], query),
        model.sf([5.0, 5.0, 5.0], query),
        rtol=1e-14,
    )
    # More flexible than the straight line it contains
    line = WeibullDistributional.fit_from_df(df, x_col="x", Z_cols=["load"])
    assert model.neg_ll() <= line.neg_ll() + 1e-8


# -- each row's parameter: bounds and plot ----------------------------------


def test_parameter_bounds_at_rows():
    x, load = _load_data()
    model = WeibullDistributional.fit(x, load)
    Z = np.array([[0.0], [0.5], [1.0]])
    est = model.parameters_at(Z)[:, 1]
    for method in ("wald", "lr"):
        band = model.parameter_cb("beta", Z, method=method)
        assert band.shape == (3, 2)
        assert np.all((band[:, 0] < est) & (est < band[:, 1]))
        # At Z = 0 it is the bound on the parameter itself
        np.testing.assert_allclose(
            band[0], model.param_cb("beta", method=method), rtol=1e-6
        )
    lower = model.parameter_cb("beta", Z, bound="lower")
    assert lower.shape == (3,) and np.all(lower < est)
    boot = model.parameter_cb(
        "beta", Z, method="bootstrap", n_boot=40, random_state=2
    )
    assert np.all((boot[:, 0] < est) & (est < boot[:, 1]))
    # A higher confidence gives a wider band
    wide = model.parameter_cb("beta", Z, alpha_ci=0.01)
    assert np.all(wide[:, 0] < model.parameter_cb("beta", Z)[:, 0])
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.parameter_cb("shape", Z)


def test_plot_parameter():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, load = _load_data()
    df = pd.DataFrame({"x": x, "load": load})
    model = WeibullDistributional.fit_from_df(df, x_col="x", Z_cols=["load"])
    fig, ax = plt.subplots()
    model.plot_parameter("beta", "load", ax=ax)
    line = ax.get_lines()[-1]
    np.testing.assert_allclose(
        line.get_ydata(),
        model.parameters_at(line.get_xdata().reshape(-1, 1))[:, 1],
    )
    plt.close(fig)
    with pytest.raises(ValueError, match="No covariate"):
        model.plot_parameter("beta", "temp")
    restored = sp.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="values="):
        restored.plot_parameter("beta", 0)


# -- the other distributions --------------------------------------------------


def test_three_parameters_and_proportions():
    rng = np.random.default_rng(12)
    z = rng.binomial(1, 0.5, 400).astype(float)
    x = sp.ExpoWeibull.random(400, 10, 1.5, 2.0, random_state=rng)
    x = x * np.exp(-0.4 * z)
    model = Distributional(sp.ExpoWeibull).fit(x, z, covariates={"alpha": [0]})
    assert model.maximum == "verified"
    assert model.parameter_names == ["alpha", "beta", "mu", "alpha_coef_0"]
    np.testing.assert_allclose(model.params[3], -0.4, atol=0.15)
    # A Beta on (0, 1), its first shape moved by z
    p = rng.beta(2.0 * np.exp(0.5 * z), 3.0)
    beta = Distributional(sp.Beta).fit(p, z, covariates={"alpha": [0]})
    assert beta.maximum == "verified"
    np.testing.assert_allclose(beta.params[2], 0.5, atol=0.25)


def test_a_centred_spline_basis_needs_no_aliasing():
    x, load = _load_data()
    df = pd.DataFrame({"x": x, "load": load})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = WeibullDistributional.fit_from_df(
            df,
            x_col="x",
            formulas={"beta": "cr(load, df=3, constraints='center')"},
        )
    assert not model.aliased.size
    with pytest.warns(UserWarning, match="cannot be estimated"):
        WeibullDistributional.fit_from_df(
            df, x_col="x", formulas={"beta": "cr(load, df=3)"}
        )
