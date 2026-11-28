"""Reliability growth of one-shot items: Crow-AMSAA discrete, Lloyd-Lipow,
Gompertz and modified Gompertz, fitted to successes and trials by stage.

References: MIL-HDBK-189C (2011) section 6.2.4 and its table IX example
for the discrete AMSAA model; the closed-form least squares of
Lloyd-Lipow; exact recovery from noise-free reliabilities for the
Gompertz curves; and simulation recovery for every maximum likelihood
fit. (The fits were also checked, outside the suite, against the worked
examples of a commercial package's documentation: Lloyd-Lipow least
squares and maximum likelihood, Gompertz and modified Gompertz least
squares, all to the four figures printed.)
"""

import numpy as np
import pandas as pd
import pytest
from autograd import grad

from surpyval.recurrent import (
    CrowAMSAADiscrete,
    DiscreteGrowthModel,
    GompertzGrowth,
    LloydLipow,
    ModifiedGompertzGrowth,
)

# -- Crow-AMSAA discrete: MIL-HDBK-189C table IX --------------------------

HANDBOOK_FAILURES = [5, 3, 4, 4]
HANDBOOK_TRIALS = [14, 19, 15, 20]


@pytest.fixture(scope="module")
def amsaa():
    return CrowAMSAADiscrete.fit(HANDBOOK_FAILURES, HANDBOOK_TRIALS)


def test_amsaa_handbook_example(amsaa):
    lam, beta = amsaa.params
    assert lam == pytest.approx(0.595, abs=5e-4)
    assert beta == pytest.approx(0.780, abs=5e-4)
    np.testing.assert_allclose(
        1 - amsaa.reliability([1, 2, 3, 4]),
        [0.333, 0.234, 0.206, 0.190],
        atol=5e-4,
    )


def test_amsaa_is_the_maximum(amsaa):
    # The handbook's likelihood equations: the score is zero.
    fitter, data = amsaa._fitter, amsaa._d
    score = grad(lambda p: fitter._nll_params(p, data))(amsaa.params)
    np.testing.assert_allclose(score, 0.0, atol=1e-5)


def test_amsaa_configuration_reliability_formula(amsaa):
    lam, beta = amsaa.params
    T = np.cumsum(HANDBOOK_TRIALS, dtype=float)
    prev = np.concatenate([[0.0], T[:-1]])
    f = lam * (T**beta - prev**beta) / np.array(HANDBOOK_TRIALS)
    np.testing.assert_allclose(amsaa.reliability([1, 2, 3, 4]), 1 - f)
    # The next configuration of 20 trials, and of one trial.
    nxt = 1 - lam * ((68 + 20) ** beta - 68**beta) / 20
    assert amsaa.next_reliability() == pytest.approx(nxt)
    one = 1 - lam * (69**beta - 68**beta)
    assert amsaa.next_reliability(1) == pytest.approx(one)
    assert amsaa.limit == 1.0


def test_amsaa_handbook_lower_bound(amsaa):
    from scipy import stats

    r4 = amsaa.reliability(4)
    expected = 1 - (1 - r4) * stats.chi2.ppf(0.8, 34) / 32
    assert amsaa.handbook_lower_bound(0.2) == pytest.approx(expected)
    assert amsaa.handbook_lower_bound(0.2) < r4


def test_amsaa_refusals(amsaa):
    with pytest.raises(ValueError, match="takes no stage"):
        CrowAMSAADiscrete.fit([1, 2], [5, 5], stage=[1, 2])
    with pytest.raises(ValueError, match="not from reliabilities"):
        CrowAMSAADiscrete.fit(reliability=[0.5, 0.6])
    with pytest.raises(ValueError, match="maximum likelihood only"):
        CrowAMSAADiscrete.fit([1, 2], [5, 5], method="least_squares")
    with pytest.raises(ValueError, match="fitted configuration"):
        amsaa.reliability(5)


# -- Lloyd-Lipow --------------------------------------------------------------

LL_FAILURES = [6, 5, 4, 5, 3, 3, 2, 3, 2, 1]
LL_TRIALS = [10, 10, 12, 12, 12, 12, 14, 14, 14, 14]


def test_lloyd_lipow_least_squares_closed_form():
    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS, method="least_squares")
    y = 1 - np.array(LL_FAILURES) / np.array(LL_TRIALS)
    X = np.column_stack([np.ones(10), -1.0 / np.arange(1, 11)])
    expected = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(model.params, expected, rtol=1e-12)
    # The same from the reliabilities alone.
    alone = LloydLipow.fit(reliability=y)
    np.testing.assert_allclose(alone.params, expected, rtol=1e-12)
    assert alone.method == "least_squares"
    assert np.isnan(alone.log_likelihood)
    # Ordinary least squares covariance.
    resid = y - X @ expected
    sigma2 = resid @ resid / 8
    np.testing.assert_allclose(
        model.cov, sigma2 * np.linalg.inv(X.T @ X), rtol=1e-8
    )


def test_lloyd_lipow_mle_is_the_maximum():
    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS)
    score = grad(lambda p: model._fitter._nll_params(p, model._d))(
        model.params
    )
    np.testing.assert_allclose(score, 0.0, atol=1e-5)
    k = np.arange(1, 11)
    np.testing.assert_allclose(
        model.reliability(k), model.params[0] - model.params[1] / k
    )
    assert model.next_reliability() == pytest.approx(
        model.params[0] - model.params[1] / 11
    )
    assert model.limit == model.params[0]


# -- Gompertz -----------------------------------------------------------------


def test_gompertz_recovers_a_noise_free_curve():
    t = np.arange(0, 9, dtype=float)
    truth = np.array([0.95, 0.4, 0.65])
    r = truth[0] * truth[1] ** (truth[2] ** t)
    model = GompertzGrowth.fit(reliability=r, stage=t)
    np.testing.assert_allclose(model.params, truth, rtol=1e-7)
    assert model.reliability(20.0) == pytest.approx(
        truth[0] * truth[1] ** (truth[2] ** 20.0), rel=1e-7
    )


def test_modified_gompertz_recovers_a_noise_free_curve():
    t = np.arange(0, 10, dtype=float)
    truth = np.array([0.65, 0.01, 0.5, 0.3])
    r = truth[3] + truth[0] * truth[1] ** (truth[2] ** t)
    model = ModifiedGompertzGrowth.fit(reliability=r, stage=t)
    np.testing.assert_allclose(model.params, truth, rtol=1e-5)
    assert model.limit == pytest.approx(0.95, rel=1e-6)


# -- simulation recovery ------------------------------------------------------


def _amsaa_r(params, n):
    T = np.cumsum(n, dtype=float)
    prev = np.concatenate([[0.0], T[:-1]])
    return 1 - params[0] * (T ** params[1] - prev ** params[1]) / n


CASES = {
    "lloyd-lipow": (
        LloydLipow,
        np.array([0.92, 0.35]),
        lambda p, k, n: p[0] - p[1] / k,
    ),
    "gompertz": (
        GompertzGrowth,
        np.array([0.95, 0.45, 0.6]),
        lambda p, k, n: p[0] * p[1] ** (p[2] ** k),
    ),
    "modified-gompertz": (
        ModifiedGompertzGrowth,
        np.array([0.6, 0.02, 0.55, 0.35]),
        lambda p, k, n: p[3] + p[0] * p[1] ** (p[2] ** k),
    ),
    "crow-amsaa-discrete": (
        CrowAMSAADiscrete,
        np.array([0.5, 0.75]),
        lambda p, k, n: _amsaa_r(p, n),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_simulation_recovery(name):
    """Large stages (2000 trials each, 10 stages): the fit lands within
    four standard errors of the truth, and the standard errors are
    small."""
    fitter, truth, curve = CASES[name]
    rng = np.random.default_rng(20261125)
    k = np.arange(1, 11, dtype=float)
    n = np.full(10, 2000)
    failures = rng.binomial(n, 1 - curve(truth, k, n))
    model = fitter.fit(failures, n)
    se = np.sqrt(np.diag(model.cov))
    assert np.all(np.abs(model.params - truth) < 4 * se), (
        model.params,
        truth,
        se,
    )
    # The maximum is at least as high as the truth's likelihood, and the
    # truth is inside the fitted reliabilities' 99.99% Wald bounds (a
    # first check of 0.01 everywhere failed for the modified Gompertz at
    # stage 1, 0.0116 off, where b = 0.02 is weakly identified; that fit's
    # likelihood is 0.67 above the truth's).
    assert -model._fitter._nll_params(truth, model._d) <= (
        model.log_likelihood + 1e-9
    )
    stages = k if name != "crow-amsaa-discrete" else np.arange(1, 11)
    cb = model.reliability_cb(stages, alpha_ci=1e-4)
    r = curve(truth, k, n)
    assert np.all((cb[:, 0] <= r) & (r <= cb[:, 1]))


# -- bounds -------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_bounds_bracket_and_agree_for_large_samples(name):
    fitter, truth, curve = CASES[name]
    rng = np.random.default_rng(7)
    k = np.arange(1, 11, dtype=float)
    n = np.full(10, 2000)
    model = fitter.fit(rng.binomial(n, 1 - curve(truth, k, n)), n)
    r = model.reliability(10)
    wald = model.reliability_cb(10, alpha_ci=0.1)
    lr = model.reliability_cb(10, alpha_ci=0.1, method="lr")
    assert wald[0] < r < wald[1]
    assert lr[0] < r < lr[1]
    # With so much data the two agree closely.
    np.testing.assert_allclose(lr, wald, atol=0.15 * (wald[1] - wald[0]))
    # A one-sided bound at alpha is the two-sided bound at 2 alpha.
    assert model.reliability_cb(
        10, alpha_ci=0.05, bound="lower"
    ) == pytest.approx(wald[0])
    assert model.reliability_cb(
        10, alpha_ci=0.05, bound="upper", method="lr"
    ) == pytest.approx(lr[1], abs=1e-6)


def test_wald_bound_formula():
    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS)
    from scipy import stats

    r = model.reliability(4)
    g = np.array([1.0, -1.0 / 4])
    se = np.sqrt(g @ model.cov @ g)
    w = stats.norm.ppf(0.975) * se / (r * (1 - r))
    expected = [r / (r + (1 - r) * np.exp(w)), r / (r + (1 - r) * np.exp(-w))]
    np.testing.assert_allclose(model.reliability_cb(4), expected)
    cb = model.reliability_cb([2, 4, 6])
    assert cb.shape == (3, 2)
    nxt = model.next_reliability_cb()
    assert nxt[0] < model.next_reliability() < nxt[1]


def test_lr_bound_is_where_the_profile_drops():
    from scipy import stats

    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS)
    lo, hi = model.reliability_cb(5, alpha_ci=0.05, method="lr")
    drop = stats.chi2.ppf(0.95, 1) / 2
    for r in (lo, hi):
        profile = model._fitter._profile_nll(r, 5.0, model._d, model.params)
        assert profile + model.log_likelihood == pytest.approx(drop, abs=1e-5)


def test_param_cb():
    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS)
    table = model.param_cb(0.1)
    assert list(table.index) == ["R_inf", "alpha"]
    assert np.all(table["lower"] < table["value"])
    assert np.all(table["value"] < table["upper"])
    assert table.loc["R_inf", "upper"] < 1


# -- refusals and display -----------------------------------------------------


def test_refusals():
    with pytest.raises(ValueError, match="Every trial succeeded"):
        LloydLipow.fit([0, 0, 0], [5, 5, 5])
    with pytest.raises(ValueError, match="needs the failures and trials"):
        LloydLipow.fit(reliability=[0.5, 0.6, 0.7], method="mle")
    with pytest.raises(ValueError, match="between 0 and trials"):
        LloydLipow.fit([6, 1], [5, 5])
    with pytest.raises(ValueError, match="not a\\s+percentage"):
        GompertzGrowth.fit(reliability=[58, 66, 72, 78], stage=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="more than 3 stages"):
        GompertzGrowth.fit(reliability=[0.5, 0.6, 0.7])
    with pytest.raises(ValueError, match="stage must increase"):
        LloydLipow.fit([1, 1, 1], [5, 5, 5], stage=[1, 3, 2])
    with pytest.raises(ValueError, match="positive"):
        LloydLipow.fit([1, 1, 1], [5, 5, 5], stage=[0, 1, 2])
    ls = LloydLipow.fit(LL_FAILURES, LL_TRIALS, method="least_squares")
    with pytest.raises(ValueError, match="needs a likelihood"):
        ls.reliability_cb(3, method="lr")
    with pytest.raises(ValueError, match="CrowAMSAADiscrete only"):
        ls.next_reliability(10)
    with pytest.raises(ValueError, match="reliability_cb"):
        ls.handbook_lower_bound()


def test_repr_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS)
    assert isinstance(model, DiscreteGrowthModel)
    assert "Lloyd-Lipow" in repr(model)
    ax = model.plot()
    assert len(ax.lines) == 1
    CrowAMSAADiscrete.fit(HANDBOOK_FAILURES, HANDBOOK_TRIALS).plot()
    plt.close("all")


# -- bootstrap bounds ---------------------------------------------------------


def _ll_least_squares(failures, trials):
    """Lloyd-Lipow's closed-form least squares fit, written out again."""
    y = 1 - np.asarray(failures, float) / trials
    x = 1 / np.arange(1, len(y) + 1, dtype=float)
    K = len(y)
    alpha = (x.sum() * y.sum() - K * (x * y).sum()) / (
        K * (x**2).sum() - x.sum() ** 2
    )
    return (y.sum() + alpha * x.sum()) / K, alpha


def test_bootstrap_is_the_percentiles_of_refits():
    # Least squares Lloyd-Lipow refits in closed form, so the bootstrap
    # can be repeated by hand, draw for draw.
    trials = np.array(LL_TRIALS)
    model = LloydLipow.fit(LL_FAILURES, LL_TRIALS, method="least_squares")
    cb = model.reliability_cb(
        [3, 6], alpha_ci=0.1, method="bootstrap", n_boot=400, random_state=5
    )
    rng = np.random.default_rng(5)
    chance = 1 - model.reliability(np.arange(1, len(trials) + 1))
    draws = []
    for _ in range(400):
        r_inf, alpha = _ll_least_squares(rng.binomial(trials, chance), trials)
        draws.append([r_inf - alpha / 3, r_inf - alpha / 6])
    draws = np.array(draws)
    expected = np.column_stack(
        [np.quantile(draws, 0.05, axis=0), np.quantile(draws, 0.95, axis=0)]
    )
    np.testing.assert_allclose(cb, expected, rtol=1e-12)
    lower = model.reliability_cb(
        6,
        alpha_ci=0.05,
        bound="lower",
        method="bootstrap",
        n_boot=400,
        random_state=5,
    )
    assert lower == pytest.approx(expected[1, 0], rel=1e-12)


def test_bootstrap_is_seeded_and_agrees_with_wald_for_large_samples():
    fitter, truth, curve = CASES["crow-amsaa-discrete"]
    rng = np.random.default_rng(11)
    n = np.full(10, 2000)
    model = fitter.fit(
        rng.binomial(n, 1 - curve(truth, np.arange(1, 11.0), n)), n
    )
    kw = dict(alpha_ci=0.1, method="bootstrap", n_boot=300, random_state=3)
    boot = model.reliability_cb(10, **kw)
    np.testing.assert_array_equal(boot, model.reliability_cb(10, **kw))
    wald = model.reliability_cb(10, alpha_ci=0.1)
    assert boot[0] < model.reliability(10) < boot[1]
    np.testing.assert_allclose(boot, wald, atol=0.15 * (wald[1] - wald[0]))
    nxt = model.next_reliability_cb(**kw)
    assert nxt[0] < model.next_reliability() < nxt[1]


@pytest.mark.parametrize("name", list(CASES))
def test_bootstrap_starts_from_the_fit(name):
    # Each replicate's search starts at the fit's parameters, so the map
    # back to the search's coordinates must invert the forward map.
    from surpyval.recurrent.discrete_growth import _Data

    fitter, truth, _ = CASES[name]
    stage = np.arange(1, 11.0)
    frame = pd.DataFrame(
        {
            "stage": stage,
            "trials": np.full(10, 50.0),
            "failures": np.full(10, 5.0),
            "reliability": np.full(10, 0.9),
        }
    )
    data = _Data(frame)
    u = fitter._from_params(truth, data)
    np.testing.assert_allclose(fitter._to_params(u, data), truth, rtol=1e-12)


def test_bootstrap_refusals():
    model = GompertzGrowth.fit(
        reliability=[0.58, 0.66, 0.725, 0.78, 0.82, 0.85],
        stage=[0, 1, 2, 3, 4, 5],
    )
    with pytest.raises(ValueError, match="reliabilities alone"):
        model.reliability_cb(3, method="bootstrap")
    amsaa = CrowAMSAADiscrete.fit(HANDBOOK_FAILURES, HANDBOOK_TRIALS)
    with pytest.raises(ValueError, match="n_boot"):
        amsaa.reliability_cb(4, method="bootstrap", n_boot=1)
    with pytest.raises(ValueError, match="fitted configuration"):
        amsaa.reliability_cb(9, method="bootstrap", n_boot=10)
    with pytest.raises(ValueError, match="method must be"):
        amsaa.reliability_cb(4, method="profile")
    with pytest.raises(ValueError, match="'wald' or 'bootstrap'"):
        amsaa.next_reliability_cb(method="lr")


# -- saving -------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: CrowAMSAADiscrete.fit(HANDBOOK_FAILURES, HANDBOOK_TRIALS),
        lambda: LloydLipow.fit(LL_FAILURES, LL_TRIALS),
        lambda: LloydLipow.fit(LL_FAILURES, LL_TRIALS, method="least_squares"),
        lambda: GompertzGrowth.fit(
            reliability=[0.58, 0.66, 0.725, 0.78, 0.82, 0.85],
            stage=[0, 1, 2, 3, 4, 5],
        ),
        lambda: ModifiedGompertzGrowth.fit(
            reliability=[0.42, 0.5683, 0.7012, 0.7907, 0.8426, 0.8706, 0.8851],
            stage=[0, 1, 2, 3, 4, 5, 6],
        ),
    ],
)
def test_round_trip(make):
    import surpyval

    model = make()
    back = surpyval.from_json(model.to_json())
    assert isinstance(back, DiscreteGrowthModel)
    assert (back.name, back.method, back.param_names) == (
        model.name,
        model.method,
        model.param_names,
    )
    np.testing.assert_array_equal(back.params, model.params)
    np.testing.assert_array_equal(back.cov, model.cov)
    np.testing.assert_array_equal(back.data.to_numpy(), model.data.to_numpy())
    stage = back.data["stage"].iloc[-1]
    assert back.reliability(stage) == model.reliability(stage)
    np.testing.assert_array_equal(
        back.reliability_cb(stage), model.reliability_cb(stage)
    )
    assert back.next_reliability() == model.next_reliability()
    ll = model.log_likelihood
    assert back.log_likelihood == ll or (
        np.isnan(ll) and np.isnan(back.log_likelihood)
    )
    assert DiscreteGrowthModel.from_dict(model.to_dict()).name == model.name


def test_from_dict_refusals():
    d = LloydLipow.fit(LL_FAILURES, LL_TRIALS).to_dict()
    with pytest.raises(ValueError, match="Unknown discrete growth model"):
        DiscreteGrowthModel.from_dict({**d, "growth_model": "Duane"})
    with pytest.raises(ValueError, match="2 parameters"):
        DiscreteGrowthModel.from_dict({**d, "params": [0.5]})
    with pytest.raises(ValueError, match="'method'|method"):
        DiscreteGrowthModel.from_dict({**d, "method": "bayes"})
    bad = dict(d)
    del bad["cov"]
    with pytest.raises(ValueError, match="'cov'"):
        DiscreteGrowthModel.from_dict(bad)
