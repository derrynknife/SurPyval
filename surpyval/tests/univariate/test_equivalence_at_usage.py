"""The life at one usage rate (``EquivalenceModel.at_usage``): an
equivalent-scale model with the rate bound, as a plain life model on the
first clock.

Every function is checked against the model's own ``*_at_usage`` methods,
the density and cumulative hazard against their definitions, and the
draws by Monte Carlo against the survival function. The seeds and sample
sizes were fixed before the results were seen.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import kstest

import surpyval as sp

K_TRUE = 10.0
RATES = (0.0, 0.02, 0.1, 0.5)
DISTS = (
    (sp.Weibull, [2000.0, 2.0]),
    (sp.LogNormal, [7.0, 0.6]),
    (sp.LogLogistic, [1500.0, 3.0]),
    (sp.Gamma, [2.5, 0.002]),
    (sp.Exponential, [1e-3]),
)


def _fitted(seed=31, n=500):
    rng = np.random.default_rng(seed)
    rate = rng.choice(RATES, n)
    life = 2000.0 * rng.weibull(2.0, n) / (1.0 + K_TRUE * rate)
    end = rng.uniform(500.0, 3000.0, n)
    h = np.minimum(life, end)
    c = (life > end).astype(int)
    return sp.WeibullEquivalence.fit(
        h, rate * h, c=c, clocks=("hours", "starts"), unit="hrs"
    )


@pytest.mark.parametrize("dist, params", DISTS, ids=lambda d: str(d)[:12])
@pytest.mark.parametrize("rate", [0.0, 0.1, 0.7])
def test_functions_are_the_models_at_usage(dist, params, rate):
    model = sp.Equivalence(dist).from_params([*params, K_TRUE])
    life = model.at_usage(rate)
    x = np.array([0.0, 50.0, 400.0, 1500.0, 6000.0])
    a = 1.0 + K_TRUE * rate
    assert life.acceleration == a
    np.testing.assert_array_equal(life.sf(x), model.sf_at_usage(x, rate))
    np.testing.assert_array_equal(life.ff(x), model.ff_at_usage(x, rate))
    np.testing.assert_array_equal(life.hf(x), model.hf_at_usage(x, rate))
    np.testing.assert_allclose(life.df(x), model.df(a * x) * a, rtol=1e-14)
    np.testing.assert_allclose(life.Hf(x), model.Hf(a * x), rtol=1e-14)
    p = np.array([0.01, 0.1, 0.5, 0.9])
    np.testing.assert_array_equal(life.qf(p), model.qf_at_usage(p, rate))
    assert life.mean() == model.mean_at_usage(rate)
    # the density is the derivative of ff, and hf = df / sf
    h = 1e-3
    xs = np.array([200.0, 900.0])
    slope = (life.ff(xs + h) - life.ff(xs - h)) / (2 * h)
    np.testing.assert_allclose(life.df(xs), slope, rtol=1e-6)
    np.testing.assert_allclose(
        life.hf(xs), life.df(xs) / life.sf(xs), rtol=1e-10
    )
    # the restricted mean is the area under sf
    area = quad(lambda t: float(life.sf(t)), 0.0, 800.0)[0]
    assert life.mean(tau=800.0) == pytest.approx(area, rel=1e-7)


def test_shapes_follow_the_query():
    life = sp.WeibullEquivalence.from_params([2000, 2, 10]).at_usage(0.1)
    assert np.ndim(life.sf(100.0)) == 0
    assert life.sf([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
    assert life.df([1.0, 2.0, 3.0]).shape == (3,)
    assert np.ndim(life.Hf(5.0)) == 0
    assert life.qf([0.1, 0.5]).shape == (2,)


def test_random_is_qf_of_uniforms_and_the_models_first_clock():
    model = sp.WeibullEquivalence.from_params([2000, 2, 10])
    life = model.at_usage(0.3)
    hours, starts = model.random(50, rate=0.3, random_state=11)
    np.testing.assert_allclose(
        life.random(50, random_state=11), hours, rtol=1e-15
    )
    u = np.random.default_rng(5).uniform(size=20)
    np.testing.assert_allclose(
        life.random(20, random_state=5), life.qf(u), rtol=1e-12
    )
    # the global stream, as Parametric.random
    np.random.seed(2)
    first = life.random(4)
    np.random.seed(2)
    np.testing.assert_array_equal(life.random(4), first)


@pytest.mark.parametrize("dist, params", DISTS, ids=lambda d: str(d)[:12])
def test_monte_carlo_draws_follow_the_life(dist, params):
    life = sp.Equivalence(dist).from_params([*params, K_TRUE]).at_usage(0.2)
    draws = life.random(20000, random_state=20261114)
    assert kstest(draws, lambda t: life.ff(t)).pvalue > 0.01
    assert draws.mean() == pytest.approx(
        life.mean(), rel=4 * draws.std() / np.sqrt(draws.size) / life.mean()
    )


def test_bounds_are_the_models_at_usage():
    model = _fitted()
    life = model.at_usage(0.5)
    x = np.array([200.0, 400.0])
    np.testing.assert_array_equal(life.cb(x), model.cb_at_usage(x, 0.5))
    np.testing.assert_array_equal(
        life.cb(x, on="ff", alpha_ci=0.1, bound="upper"),
        model.cb_at_usage(x, 0.5, on="ff", alpha_ci=0.1, bound="upper"),
    )
    np.testing.assert_array_equal(
        life.quantile_cb(0.1), model.quantile_cb_at_usage(0.1, 0.5)
    )


def test_rmst_interval_includes_k_and_contains_the_point():
    model = _fitted()
    life = model.at_usage(0.5)
    res = life.rmst(300.0)
    assert res["rmst"] == pytest.approx(life.mean(tau=300.0), rel=1e-12)
    assert 0 < res["lower"] < res["rmst"] < res["upper"] < 300.0
    assert res["tau"] == 300.0 and res["se"] > 0
    # with k's variance and covariances set to 0 the interval narrows
    narrow = model.at_usage(0.5)
    cov = model.covariance().copy()
    cov[-1, :] = cov[:, -1] = 0.0
    narrow.model = sp.from_dict(
        {**model.to_dict(), "covariance": cov.tolist()}
    )
    held = narrow.rmst(300.0)
    assert held["rmst"] == pytest.approx(res["rmst"], rel=1e-12)
    assert held["se"] < res["se"]
    # a wider interval at a lower level
    wide = life.rmst(300.0, alpha_ci=0.01)
    assert wide["lower"] < res["lower"] and wide["upper"] > res["upper"]


def test_a_model_with_no_covariance_refuses_bounds_saying_what_to_use():
    life = sp.WeibullEquivalence.from_params([2000, 2, 10]).at_usage(0.1)
    with pytest.raises(ValueError, match=r"life.mean\(tau=300\)"):
        life.rmst(300)
    with pytest.raises(ValueError, match="no covariance"):
        life.cb(100.0)


def test_rate_is_one_finite_number_at_least_zero():
    model = sp.WeibullEquivalence.from_params([2000, 2, 10])
    assert model.at_usage(np.array([0.2])).rate == 0.2
    assert model.at_usage(0).acceleration == 1.0
    with pytest.raises(ValueError, match="once per rate"):
        model.at_usage([0.1, 0.2])
    for bad in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="0 or more"):
            model.at_usage(bad)
    life = model.at_usage(0.1)
    for bad in (-1.0, np.inf, [1.0, 2.0]):
        with pytest.raises(ValueError, match="tau"):
            life.mean(tau=bad)


def test_labels_are_kept_as_typed_and_shown():
    model = sp.WeibullEquivalence.from_params(
        [2000, 2, 10], clocks=("run hrs", "starts"), unit="hrs"
    )
    life = model.at_usage(0.25)
    assert life.unit == "hrs"
    assert life.clocks == ("run hrs", "starts")
    assert repr(life) == (
        "LifeAtUsage(WeibullEquivalence, clocks=('run hrs', 'starts'), "
        "rate=0.25)"
    )
    assert (
        sp.WeibullEquivalence.from_params([2000, 2, 10]).at_usage(0.1).unit
        is None
    )


def test_restored_model_gives_the_same_life():
    model = _fitted()
    restored = sp.from_dict(model.to_dict())
    a, b = model.at_usage(0.1), restored.at_usage(0.1)
    x = np.array([100.0, 700.0])
    np.testing.assert_allclose(a.sf(x), b.sf(x), rtol=1e-14)
    np.testing.assert_allclose(a.cb(x), b.cb(x), rtol=1e-12)
    assert b.unit == "hrs"
