"""The mean residual life, ``mrl`` (#825): E[T - x | T > x], on the
distributions, the fitted parametric models (with ``mrl_cb``) and the
non-parametric estimators (restricted to ``tau``)."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import poisson

import surpyval as sp

CONTINUOUS = [
    ("Weibull", (10.0, 2.0)),
    ("Weibull", (10.0, 0.5)),
    ("Exponential", (0.2,)),
    ("Normal", (3.0, 2.0)),
    ("Gumbel", (3.0, 2.0)),
    ("Logistic", (3.0, 2.0)),
    ("LogNormal", (2.0, 0.5)),
    ("LogLogistic", (10.0, 3.0)),
    ("Gamma", (2.0, 0.5)),
    ("ExpoWeibull", (10.0, 2.0, 1.5)),
    ("Rayleigh", (3.0,)),
    ("Beta", (2.0, 3.0)),
    ("Uniform", (1.0, 5.0)),
]

DISCRETE = [
    ("Poisson", (3.0,)),
    ("Geometric", (0.3,)),
    ("NegativeBinomial", (3.0, 0.4)),
    ("Binomial", (10, 0.3)),
]


def _integrated(dist, params, x):
    """The defining integral, by quadrature of ``sf`` in pieces."""
    lo, hi = dist._support_edges(*params)
    end = hi if np.isfinite(hi) else np.inf
    sf = lambda u: float(dist.sf(u, *params))  # noqa: E731
    mid = float(dist.qf(0.999, *params))
    head = quad(sf, x, max(x, mid), epsabs=0, epsrel=1e-12, limit=200)[0]
    tail = quad(sf, max(x, mid), end, epsabs=0, epsrel=1e-12, limit=200)[0]
    return (head + tail) / sf(x)


@pytest.mark.parametrize("name, params", CONTINUOUS + DISCRETE)
def test_mrl_at_the_support_start_is_the_mean(name, params):
    dist = getattr(sp, name)
    lo = dist._support_edges(*params)[0]
    start = 0.0 if not np.isfinite(lo) else lo
    if not np.isfinite(lo):
        start = -1e3 * float(dist.mean(*params) or 1) - 1e3
    assert dist.mrl(start, *params) == pytest.approx(
        float(dist.mean(*params)) - start, rel=1e-8
    )


@pytest.mark.parametrize("name, params", CONTINUOUS)
def test_mrl_is_the_integral_of_the_survival(name, params):
    dist = getattr(sp, name)
    x = np.asarray(dist.qf(np.array([0.05, 0.3, 0.7, 0.95]), *params))
    got = dist.mrl(x, *params)
    want = [_integrated(dist, params, float(t)) for t in x]
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_weibull_closed_form_far_in_the_tail():
    # sf(300) underflows; the MRL tends to alpha**beta / (beta x**(beta -
    # 1)), here 100 / 600, and stays finite.
    assert sp.Weibull.sf(300.0, 10.0, 2.0) == 0.0
    got = sp.Weibull.mrl(300.0, 10.0, 2.0)
    assert got == pytest.approx(100 / 600 * (1 - 1 / 1800), rel=1e-6)


def test_exponential_is_memoryless():
    np.testing.assert_allclose(
        sp.Exponential.mrl([0.0, 1.0, 100.0, 1e6], 0.2), 5.0, rtol=1e-15
    )
    assert sp.Exponential.mrl(-2.0, 0.2) == pytest.approx(7.0)


def test_geometric_is_memoryless_on_the_integers():
    # Trials to the first success: E[T - m | T > m] = 1 / p at every
    # integer m, and a part of a trial already survived is taken off.
    np.testing.assert_allclose(
        sp.Geometric.mrl([0.0, 1.0, 5.0], 0.3), 1 / 0.3, rtol=1e-12
    )
    assert sp.Geometric.mrl(1.5, 0.3) == pytest.approx(1 / 0.3 - 0.5)


def test_poisson_matches_the_sum_of_the_mass():
    k = np.arange(0, 200)
    for m in [0, 2, 5, 10]:
        mass = poisson.pmf(k, 3.0)
        want = np.sum((k - m) * mass * (k > m)) / poisson.sf(m, 3.0)
        assert sp.Poisson.mrl(float(m), 3.0) == pytest.approx(want, rel=1e-12)


def test_ends_of_the_support_and_missing_values():
    # Uniform(1, 5): (5 - x) / 2 inside, the mean less x before it, and
    # nan from its end on, where no unit survives.
    got = sp.Uniform.mrl([0.0, 1.0, 3.0, 5.0, 6.0, np.nan], 1.0, 5.0)
    np.testing.assert_allclose(got, [3.0, 2.0, 1.0, np.nan, np.nan, np.nan])


def test_infinite_mean_gives_infinite_mrl():
    # A LogLogistic with beta <= 1 has an infinite mean (it was nan).
    assert sp.LogLogistic.mean(10.0, 0.9) == np.inf
    np.testing.assert_array_equal(
        sp.LogLogistic.mrl([1.0, 5.0], 10.0, 0.9), np.inf
    )


def test_model_mrl_shape_and_conventions():
    model = sp.Weibull.from_params([10.0, 2.0])
    assert np.ndim(model.mrl(5.0)) == 0
    assert model.mrl([[0.0, 5.0], [10.0, 20.0]]).shape == (2, 2)
    assert model.mrl(0.0) == pytest.approx(model.mean())
    # An offset shifts it
    shifted = sp.Weibull.from_params([10.0, 2.0], gamma=3.0)
    assert shifted.mrl(8.0) == pytest.approx(model.mrl(5.0))
    # A zero-inflated model: the mass at 0 is ahead only before 0
    zi = sp.Weibull.from_params([10.0, 2.0], f0=0.2)
    assert zi.mrl(5.0) == pytest.approx(model.mrl(5.0))
    assert zi.mrl(-1.0) == pytest.approx(zi.mean() + 1.0)
    # A limited failure population: some survivors never fail
    lfp = sp.Weibull.from_params([10.0, 2.0], lfp_p=0.8)
    np.testing.assert_array_equal(lfp.mrl([1.0, np.nan]), [np.inf, np.nan])


def test_mrl_cb():
    rng = np.random.default_rng(1)
    x = sp.Weibull.random(60, 10, 3, random_state=rng)
    model = sp.Weibull.fit(x)
    ages = np.array([0.0, 5.0, 9.0])
    value = model.mrl(ages)
    wald = model.mrl_cb(ages)
    lr = model.mrl_cb(ages, method="lr")
    boot = model.mrl_cb(ages, method="bootstrap", n_boot=100, random_state=1)
    for cb in (wald, lr, boot):
        assert cb.shape == (3, 2)
        assert np.all(cb[:, 0] < value) and np.all(value < cb[:, 1])
    # At 0 the bound is the mean's
    np.testing.assert_allclose(wald[0], model.mean_cb(), rtol=1e-5)
    np.testing.assert_allclose(lr[0], model.mean_cb(method="lr"), rtol=1e-6)
    # One-sided, scalar in scalar out, missing in missing out
    assert np.ndim(model.mrl_cb(5.0, bound="lower")) == 0
    assert model.mrl_cb(5.0, bound="lower") == pytest.approx(
        model.mrl_cb(5.0, alpha_ci=0.1)[0]
    )
    assert np.all(np.isnan(model.mrl_cb(np.nan)))


def test_mrl_cb_limited_failure_population_is_infinite():
    rng = np.random.default_rng(2)
    x = sp.Weibull.random(60, 10, 3, random_state=rng)
    model = sp.Weibull.fit(x, c=(x > 9).astype(int), lfp=True)
    assert model.lfp_p < 1
    np.testing.assert_array_equal(model.mrl_cb(5.0), [np.inf, np.inf])


def test_nonparametric_restricted_mrl():
    # Without censoring the KM MRL is the mean remaining time of the
    # units still running.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0])
    model = sp.KaplanMeier.fit(x)
    for t in [0.0, 2.5, 4.0, 7.0]:
        assert model.mrl(t) == pytest.approx(np.mean(x[x > t] - t))
    assert model.mrl(-1.0) == pytest.approx(model.mean() + 1.0)
    assert np.isnan(model.mrl(11.0))
    assert model.mrl([[0.0], [2.5]]).shape == (2, 1)
    # Restricted to tau: E[min(T, tau) - x | T > x]
    assert model.mrl(2.5, tau=6.0) == pytest.approx(
        np.mean(np.minimum(x[x > 2.5], 6.0) - 2.5)
    )
    assert model.mrl(6.5, tau=6.0) == 0.0


def test_nonparametric_mrl_refuses_tau_past_the_data():
    model = sp.KaplanMeier.fit([1, 2, 3, 4, 5, 6], c=[0, 1, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="not estimable"):
        model.mrl(1.0, tau=20.0)
