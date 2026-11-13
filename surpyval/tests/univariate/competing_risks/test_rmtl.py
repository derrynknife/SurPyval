"""The restricted mean time lost to a cause: ``CompetingRisks.rmtl``,
``ParametricCompetingRisks.rmtl`` and ``surpyval.rmtl_diff``. The checks
against R are in ``tests/reference/test_rmtl.py``."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

import surpyval as sp

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
E = np.array(["a", "b", "a", None, "a", "b", "a", None, "b", "a"], object)


def test_causes_add_up_to_tau_less_the_all_cause_rmst():
    model = sp.CompetingRisks.fit(X, E)
    km = sp.KaplanMeier.fit(X, c=(E == None).astype(int))  # noqa: E711
    for tau in (0.5, 3.0, 7.5, 10.0):
        total = model.rmtl("a", tau=tau)["rmtl"]
        total += model.rmtl("b", tau=tau)["rmtl"]
        assert_allclose(total, tau - km.rmst(tau=tau)["rmst"], rtol=1e-12)


def test_rmtl_is_the_area_under_the_cif():
    model = sp.CompetingRisks.fit(X, E)
    grid = np.linspace(0, 8.5, 850_001)
    area = np.sum(model.cif(grid[:-1], "a") * np.diff(grid))
    assert_allclose(model.rmtl("a", tau=8.5)["rmtl"], area, atol=2e-5)


def test_before_the_first_time_nothing_is_lost():
    res = sp.CompetingRisks.fit(X, E).rmtl("a", tau=0.5)
    assert res["rmtl"] == 0.0 and res["se"] == 0.0


def test_default_tau_and_interval():
    model = sp.CompetingRisks.fit(X, E)
    res = model.rmtl("b")
    assert res["tau"] == 10.0
    assert_allclose(
        [res["lower"], res["upper"]],
        res["rmtl"] + np.array([-1, 1]) * 1.959963984540054 * res["se"],
    )


def test_tau_past_the_data_is_refused_while_units_are_event_free():
    running = E.copy()
    running[-1] = None  # the last unit is still event-free at 10
    model = sp.CompetingRisks.fit(X, running)
    with pytest.raises(ValueError, match="tau = 12 is past"):
        model.rmtl("a", tau=12)
    # Once every unit has failed the incidences are flat: any tau is fine.
    done = sp.CompetingRisks.fit(X, E)
    res = done.rmtl("a", tau=20)
    cif = float(done.cif(10, "a"))
    assert_allclose(res["rmtl"], done.rmtl("a", tau=10)["rmtl"] + 10 * cif)


def test_refusals_name_the_argument():
    model = sp.CompetingRisks.fit(X, E)
    with pytest.raises(ValueError):
        model.rmtl(None)
    with pytest.raises(ValueError):
        model.rmtl("c")
    with pytest.raises(ValueError, match="tau must be"):
        model.rmtl("a", tau=-1)
    with pytest.raises(ValueError, match="alpha_ci"):
        model.rmtl("a", alpha_ci=1.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = sp.CompetingRisks.fit(X, E, w=np.linspace(0.5, 2, 10))
    with pytest.raises(ValueError, match="weights"):
        weighted.rmtl("a")


def test_rmtl_survives_a_round_trip():
    model = sp.CompetingRisks.fit(X, E)
    back = sp.CompetingRisks.from_dict(model.to_dict())
    assert back.rmtl("a", tau=7) == model.rmtl("a", tau=7)


def test_parametric_exponential_closed_form():
    model = sp.ParametricCompetingRisks.fit(X, E, dist=sp.Exponential)
    la = float(model.models["a"].params[0])
    lb = float(model.models["b"].params[0])
    tau = 8.0

    def closed(a, b):
        lam = a + b
        return a / lam * (tau - (1 - np.exp(-lam * tau)) / lam)

    res = model.rmtl("a", tau=tau)
    assert_allclose(res["rmtl"], closed(la, lb), rtol=1e-9)
    h = 1e-6
    grad = np.array(
        [
            (closed(la + h, lb) - closed(la - h, lb)) / (2 * h),
            (closed(la, lb + h) - closed(la, lb - h)) / (2 * h),
        ]
    )
    se = np.sqrt(grad @ model.covariance() @ grad)
    assert_allclose(res["se"], se, rtol=1e-5)
    assert 0 < res["lower"] < res["rmtl"] < res["upper"] < tau


def test_parametric_weibull_matches_the_integral_of_its_cif():
    model = sp.ParametricCompetingRisks.fit(X, E, dist=sp.Weibull)
    want = quad(lambda t: model.cif(t, "a"), 0, 9, epsabs=0, epsrel=1e-12)
    assert_allclose(model.rmtl("a", tau=9)["rmtl"], want[0], rtol=1e-8)
    # Past the data the fitted distributions extrapolate.
    assert model.rmtl("a", tau=30)["rmtl"] > model.rmtl("a", tau=9)["rmtl"]


def test_rmtl_diff_mixes_models_and_takes_the_common_horizon():
    a = sp.CompetingRisks.fit(X, E)
    b = sp.ParametricCompetingRisks.fit(X[:8] + 1, E[:8], dist=sp.Weibull)
    res = sp.rmtl_diff(a, b, "a")
    assert res["tau"] == 9.0
    ra, rb = a.rmtl("a", tau=9), b.rmtl("a", tau=9)
    assert_allclose(res["difference"], ra["rmtl"] - rb["rmtl"])
    assert_allclose(res["se"], np.hypot(ra["se"], rb["se"]))
    assert res["ratio_lower"] < res["ratio"] < res["ratio_upper"]
    with pytest.raises(ValueError, match="model_b"):
        sp.rmtl_diff(b, sp.CompetingRisks.fit(X[:8], E[:8]), "a", tau=9.5)


def test_rmtl_diff_of_identical_groups_is_zero():
    a = sp.CompetingRisks.fit(X, E)
    res = sp.rmtl_diff(a, a, "b", tau=8)
    assert res["difference"] == 0.0 and res["p_value"] == 1.0
    assert res["ratio"] == 1.0
