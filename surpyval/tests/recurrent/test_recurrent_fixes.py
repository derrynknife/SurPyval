"""Recurrent-event fixes: default starts, MCF bounds and variance."""

import warnings

import numpy as np
import pytest

from surpyval.recurrent import (
    CrowAMSAA,
    Duane,
    NonParametricCounting,
    ProportionalIntensityNHPP,
)


def _power_law_fleet(lam, beta, T, n_items=20, coef=0.7, seed=3):
    rng = np.random.default_rng(seed)
    x, i, c, z = [], [], [], []
    for k in range(n_items):
        zk = k % 2
        cumulative = 0.0
        while True:
            cumulative += rng.exponential()
            t = (cumulative / (lam * np.exp(coef * zk))) ** (1 / beta)
            if t > T:
                break
            x.append(t), i.append(k), c.append(0), z.append(zk)
        x.append(T), i.append(k), c.append(1), z.append(zk)
    return np.array(x), np.array(i), np.array(c), np.array(z).reshape(-1, 1)


@pytest.mark.parametrize(
    "lam, beta, T", [(1e-4, 2.0, 300.0), (1e-5, 2.5, 200.0)]
)
def test_duane_proportional_intensity_reaches_the_crow_amsaa_optimum(
    lam, beta, T
):
    # From a unit start the Duane fit stopped 15-30 AIC short
    x, i, c, Z = _power_law_fleet(lam, beta, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        duane = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=Duane)
        crow = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    assert duane.aic == pytest.approx(crow.aic, abs=0.01)
    assert duane.coeffs == pytest.approx(crow.coeffs, abs=1e-3)


def _heterogeneous_fleet(n_items=30, T=10.0, seed=0):
    rng = np.random.default_rng(seed)
    x, i, c = [], [], []
    for k in range(n_items):
        rate = rng.gamma(0.5, 2.0) * 0.5
        t = 0.0
        while True:
            t += rng.exponential(1 / rate)
            if t > T:
                break
            x.append(t), i.append(k), c.append(0)
        x.append(T), i.append(k), c.append(1)
    return np.array(x), np.array(i), np.array(c)


def test_mcf_normal_bounds_are_estimate_plus_minus_z_se():
    x, i, c = _heterogeneous_fleet()
    model = NonParametricCounting.fit(x, i, c)
    t = np.array([5.0])
    k = np.searchsorted(model.x, 5.0, side="right") - 1
    se = np.sqrt(model.var[k])
    lower, upper = model.mcf_cb(t, bound_type="normal")[0]
    assert lower == pytest.approx(model.mcf_hat[k] - 1.959964 * se)
    assert upper == pytest.approx(model.mcf_hat[k] + 1.959964 * se)


def test_mcf_variance_is_robust_to_heterogeneous_items():
    # Lawless-Nadeau: close to the sampling variance even when items have
    # very different rates; the per-step variance was ~8 times too small
    estimates, variances = [], []
    for seed in range(150):
        x, i, c = _heterogeneous_fleet(seed=seed)
        model = NonParametricCounting.fit(x, i, c)
        k = np.searchsorted(model.x, 7.0, side="right") - 1
        estimates.append(model.mcf_hat[k])
        variances.append(model.var[k])
    ratio = np.mean(variances) / np.var(estimates)
    assert 0.75 < ratio < 1.3


def test_mcf_variance_matches_the_per_step_form_for_single_events():
    # one event per item, no ties: no within-item covariance to add, so
    # the robust variance reduces to the per-step one
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    i = np.array([1, 2, 3, 4, 5])
    c = np.zeros(5, dtype=int)
    model = NonParametricCounting.fit(x, i, c)
    naive = NonParametricCounting.from_xrd(model.x, model.r, model.d)
    np.testing.assert_allclose(model.var, naive.var)


def _small_covariate_fleet():
    x = [2.0, 5.0, 3.0, 7.0, 1.0, 4.0, 2.0, 6.0]
    i = [1, 1, 2, 2, 3, 3, 4, 4]
    c = [0, 1, 0, 1, 0, 1, 0, 1]
    z = {1: 0.2, 2: 0.5, 3: 0.8, 4: 0.3}
    return x, i, c, z


def test_covariates_accept_a_dict_of_scalars_and_a_1d_array():
    from surpyval.recurrent import ProportionalIntensityHPP

    x, i, c, z = _small_covariate_fleet()
    Z_rows = np.array([[z[k]] for k in i])
    as_rows = ProportionalIntensityHPP.fit(x, Z_rows, i=i, c=c)
    as_dict = ProportionalIntensityHPP.fit(x, z, i=i, c=c)
    as_1d = ProportionalIntensityHPP.fit(x, Z_rows.ravel(), i=i, c=c)
    np.testing.assert_allclose(as_dict.params, as_rows.params)
    np.testing.assert_allclose(as_1d.params, as_rows.params)
    with pytest.raises(ValueError, match="no covariates for item"):
        ProportionalIntensityHPP.fit(x, {1: 0.2, 2: 0.5}, i=i, c=c)


def test_hpp_works_as_a_cause_specific_baseline_and_from_params():
    from surpyval.recurrent import HPP, CauseSpecificNHPP

    x = [1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 8.0, 9.0, 10.0, 10.0]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2]
    c = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    e = ["a", "b", "a", "a", "b", "a", "b", "b", None, None]
    model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e, dist=HPP)
    # each cause's rate is its events over the total exposure (20)
    assert model.models["a"].params[0] == pytest.approx(4 / 20)
    assert model.models["b"].params[0] == pytest.approx(4 / 20)

    given = HPP.from_params([0.5])
    assert given.cif(np.array([2.0]))[0] == pytest.approx(1.0)
    assert "given parameters" in repr(given)
    fitted = HPP.fit([1.0, 2.0, 3.0, 4.0], c=[0, 0, 0, 1])
    assert "Fitted by           : MLE" in repr(fitted)


def test_mse_fit_says_why_it_has_no_likelihood():
    from surpyval.recurrent import CrowAMSAA

    x = [1.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    model = CrowAMSAA.fit(x, c=[0] * 7 + [1], how="MSE")
    assert "MSE" in repr(model)
    with pytest.raises(ValueError, match="how='MSE'"):
        model.aic


def test_cause_specific_mcf_plot_draws_bounds_on_request():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from surpyval.recurrent import CauseSpecificMCF

    x = [1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 8.0, 9.0, 10.0, 10.0]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2]
    c = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    e = ["a", "b", "a", "a", "b", "a", "b", "b", None, None]
    model = CauseSpecificMCF.fit(x, i=i, c=c, e=e)
    _, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False)
    assert len(ax.lines) == 2
    _, ax = plt.subplots()
    model.plot(ax=ax, confidence=0.8)
    # an MCF plus a [lower, upper] pair per cause
    assert len(ax.lines) == 6
    upper = model.models["a"].mcf_cb(model.models["a"].x, confidence=0.8)
    np.testing.assert_allclose(ax.lines[2].get_ydata(), upper[:, 1])
    plt.close("all")
