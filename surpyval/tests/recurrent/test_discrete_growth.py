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
