"""The restricted mean survival time of the parametric models:
``Parametric.rmst``, ``rmst_cb`` and ``mean(tau=)``, the parametric
regression models' ``rmst``, ``rmst_cb`` and ``rmst_diff`` at ``Z``, and
``surpyval.rmst_diff`` between parametric fits. The values against R are
in ``reference/test_rmst.py``; these are the properties."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

import surpyval as sp


def _area(model, tau, points=None):
    """The area under ``model.sf`` from 0 to ``tau`` by quadrature."""
    return quad(
        lambda t: float(model.sf(t)),
        0,
        tau,
        epsabs=0,
        epsrel=1e-12,
        limit=200,
        points=points,
    )[0]


@pytest.mark.parametrize(
    "dist, params",
    [
        (sp.Exponential, [0.3]),
        (sp.Weibull, [10.0, 3.0]),
        (sp.Weibull, [10.0, 0.4]),
        (sp.Gamma, [2.5, 0.4]),
        (sp.Gamma, [0.5, 0.1]),
        (sp.LogNormal, [2.0, 0.8]),
        (sp.LogLogistic, [10.0, 2.5]),
        (sp.Normal, [8.0, 3.0]),
        (sp.Gumbel, [8.0, 2.0]),
    ],
)
@pytest.mark.parametrize("tau", [0.5, 5.0, 12.0, 60.0])
def test_rmst_is_the_area_under_sf(dist, params, tau):
    # Closed forms (Exponential, Weibull, Gamma, LogNormal) and the
    # quadrature of the others, against an independent quadrature.
    model = dist.from_params(params)
    assert model.mean(tau=tau) == pytest.approx(
        _area(model, tau), rel=1e-9, abs=1e-12
    )


def test_rmst_limits():
    model = sp.Weibull.from_params([10.0, 2.0])
    assert model.mean(tau=0) == 0.0
    taus = np.linspace(0.1, 50, 30)
    values = np.array([model.mean(tau=t) for t in taus])
    # At most tau, increasing, and reaching the mean
    assert np.all(values <= taus) and np.all(np.diff(values) > 0)
    assert model.mean(tau=1e3) == pytest.approx(model.mean(), rel=1e-12)
    # A tiny horizon: nothing has failed yet
    assert model.mean(tau=1e-6) == pytest.approx(1e-6, rel=1e-9)


def test_rmst_offset_lfp_zi_are_the_area_under_their_sf():
    offset = sp.Weibull.from_params([10.0, 2.0], gamma=3.0)
    assert offset.mean(tau=2.0) == 2.0  # before the offset
    assert offset.mean(tau=9.0) == pytest.approx(
        _area(offset, 9.0, points=[3.0]), rel=1e-10
    )
    lfp = sp.Weibull.from_params([10.0, 2.0], lfp_p=0.7)
    # (1 - p) tau for the units that never fail: finite, while the mean
    # is not
    base = sp.Weibull.from_params([10.0, 2.0]).mean(tau=20.0)
    assert lfp.mean(tau=20.0) == pytest.approx(0.3 * 20 + 0.7 * base)
    assert np.isinf(lfp.mean())
    zi = sp.Weibull.from_params([10.0, 2.0], f0=0.2)
    assert zi.mean(tau=20.0) == pytest.approx(0.8 * base)


@pytest.fixture(scope="module")
def weibull_fit():
    rng = np.random.default_rng(3)
    x = sp.Weibull.qf(rng.uniform(size=40), 10.0, 2.5)
    c = (x > 12).astype(int)
    return sp.Weibull.fit(np.minimum(x, 12), c=c)


def test_rmst_dict_matches_rmst_cb_and_mean(weibull_fit):
    model = weibull_fit
    res = model.rmst(8.0)
    assert set(res) == {"rmst", "se", "lower", "upper", "tau"}
    assert res["rmst"] == model.mean(tau=8.0)
    assert_allclose(model.rmst_cb(8.0), [res["lower"], res["upper"]])
    assert res["lower"] < res["rmst"] < res["upper"] < 8.0
    # The default horizon is the largest observed time
    assert model.rmst()["tau"] == 12.0
    # One-sided bounds at half the level are the two-sided ends
    lower = model.rmst_cb(8.0, alpha_ci=0.025, bound="lower")
    upper = model.rmst_cb(8.0, alpha_ci=0.025, bound="upper")
    assert_allclose([lower, upper], [res["lower"], res["upper"]])


def test_rmst_cb_shapes_and_methods(weibull_fit):
    model = weibull_fit
    assert model.rmst_cb(5.0).shape == (2,)
    assert model.rmst_cb([5.0, 8.0]).shape == (2, 2)
    assert model.rmst_cb([5.0, 8.0], bound="upper").shape == (2,)
    assert_allclose(model.rmst_cb(0.0), [0.0, 0.0])
    point = model.mean(tau=8.0)
    lr = model.rmst_cb(8.0, method="lr")
    assert lr[0] < point < lr[1]
    # The likelihood-ratio and Wald bounds agree to first order
    assert_allclose(lr, model.rmst_cb(8.0), rtol=0.03)
    boot = model.rmst_cb(8.0, method="bootstrap", n_boot=100, random_state=1)
    assert boot[0] < point < boot[1]
    assert_allclose(boot, model.rmst_cb(8.0), rtol=0.05)


def test_rmst_offset_warns_and_lfp_bounds():
    rng = np.random.default_rng(4)
    x = 2.0 + sp.Weibull.qf(rng.uniform(size=40), 10.0, 2.0)
    model = sp.Weibull.fit(x, offset=True)
    with pytest.warns(UserWarning, match="offset"):
        model.rmst(8.0)
    y = sp.Weibull.qf(rng.uniform(size=80), 10.0, 2.0)
    y = np.where(rng.uniform(size=80) < 0.3, 40.0, y)
    lfp = sp.Weibull.fit(np.minimum(y, 30), c=(y >= 30).astype(int), lfp=True)
    res = lfp.rmst(20.0)
    assert res["lower"] < res["rmst"] < res["upper"] < 20.0
    lr = lfp.rmst_cb(20.0, method="lr")
    assert lr[0] < res["rmst"] < lr[1]


def test_rmst_refusals():
    with pytest.raises(ValueError, match="mean\\(tau=8\\)"):
        sp.Weibull.from_params([10, 2]).rmst(8.0)
    with pytest.raises(ValueError, match="pass tau|data"):
        sp.Weibull.from_params([10, 2]).rmst()
    model = sp.Weibull.fit([1.0, 2.0, 3.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="tau"):
        model.rmst(-1.0)
    with pytest.raises(ValueError, match="mean\\(\\)"):
        model.rmst_cb(np.inf)
    with pytest.raises(ValueError, match="rmst_cb"):
        model.rmst([1.0, 2.0])
    geometric = sp.Geometric.fit([1, 2, 2, 3, 5, 8])
    with pytest.raises(ValueError, match="discrete"):
        geometric.rmst(4)


def test_wald_coverage_exponential():
    # The 95% Wald interval on the restricted mean to tau = 1 (about the
    # median) of an Exponential with rate 1, 40 units censored at 2:
    # 400 repetitions with a fixed seed. Result as found: see the assert.
    rng = np.random.default_rng(20261105)
    truth = sp.Exponential.from_params([1.0]).mean(tau=1.0)
    hits = 0
    reps = 400
    for _ in range(reps):
        x = rng.exponential(1.0, 40)
        model = sp.Exponential.fit(np.minimum(x, 2.0), c=(x > 2.0) * 1)
        lo, hi = model.rmst_cb(1.0)
        hits += lo <= truth <= hi
    # Binomial standard error at 95% and 400 reps is 1.1%
    assert 0.92 <= hits / reps <= 0.98


def test_rmst_diff_with_parametric_fits():
    a = sp.Weibull.fit([2, 3, 4, 5, 6, 7])
    b = sp.Weibull.fit([1, 2, 2, 3, 4, 5])
    res = sp.rmst_diff(a, b)
    ra, rb = a.rmst(5.0), b.rmst(5.0)
    assert res["tau"] == 5.0
    assert res["difference"] == pytest.approx(ra["rmst"] - rb["rmst"])
    assert res["se"] == pytest.approx(np.hypot(ra["se"], rb["se"]))
    # Swapping the groups changes the sign only
    swapped = sp.rmst_diff(b, a)
    assert swapped["difference"] == pytest.approx(-res["difference"])
    assert swapped["p_value"] == pytest.approx(res["p_value"])
    # A Kaplan-Meier and a parametric fit can be compared
    km = sp.KaplanMeier.fit([1, 2, 2, 3, 4, 5])
    mixed = sp.rmst_diff(a, km)
    assert mixed["rmst_b"] == pytest.approx(km.rmst(5.0)["rmst"])


@pytest.fixture(scope="module")
def ph_fit():
    np.random.seed(1)
    Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    x = sp.Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    return sp.WeibullPH.fit(x, Z)


def test_regression_rmst_is_the_area_under_sf(ph_fit):
    model = ph_fit
    for row in ([0], [1]):
        area = quad(
            lambda t: float(model.sf(t, row)), 0, 8, epsabs=0, epsrel=1e-12
        )[0]
        assert model.rmst(8.0, row)["rmst"] == pytest.approx(area, rel=1e-9)
    # A PH model at Z = 0 is its baseline Weibull
    alpha, beta = model.params[:2]
    base = sp.Weibull.from_params([alpha, beta]).mean(tau=8.0)
    assert model.rmst(8.0, [0])["rmst"] == pytest.approx(base, rel=1e-9)


def test_regression_rmst_cb_and_diff(ph_fit):
    model = ph_fit
    res = model.rmst(8.0, [[0], [1]])
    assert res["rmst"].shape == (2,)
    cb = model.rmst_cb(8.0, [[0], [1]])
    assert_allclose(cb, np.column_stack([res["lower"], res["upper"]]))
    assert model.rmst_cb(8.0, [1]).shape == (2,)
    assert model.rmst_cb([4.0, 8.0], [1], bound="lower").shape == (2,)
    diff = model.rmst_diff(8.0, [0], [1])
    assert diff["difference"] == pytest.approx(res["rmst"][0] - res["rmst"][1])
    # The two RMSTs share the parameters: the difference's standard error
    # is not that of independent groups
    assert diff["se"] != pytest.approx(np.hypot(*res["se"]), rel=1e-3)
    same = model.rmst_diff(8.0, [1], [1])
    assert same["difference"] == 0.0 and same["se"] == 0.0
    boot = model.rmst_cb(
        8.0, [1], method="bootstrap", n_boot=60, random_state=2
    )
    assert boot[0] < res["rmst"][1] < boot[1]
    with pytest.raises(ValueError, match="likelihood-ratio"):
        model.rmst_cb(8.0, [1], method="lr")
    with pytest.raises(ValueError, match="one row"):
        model.rmst_diff(8.0, [[0], [1]], [1])


def test_regression_rmst_missing_covariate_is_nan(ph_fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ph_fit.rmst(8.0, [[np.nan], [1]])
    assert np.isnan(res["rmst"][0]) and np.isfinite(res["rmst"][1])
