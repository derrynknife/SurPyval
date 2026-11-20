"""The level of a process model's unit known not to have failed.

``level_at(t, y0=, t0=, given_survival=True)`` is the distribution of the
level at ``t`` of a unit that has not reached the threshold between ``t0``
and ``t``: for a gamma process the increase truncated at the distance to
the threshold, for a Wiener process the normal less its reflection in the
threshold (the method of images). The checks: the density integrates to 1
and to ``ff``; ``qf`` inverts ``ff``; all of the mass is below the
threshold; the moments against quadrature; the limit as the threshold
moves far away (the unconditioned level); the stress clock; and draws of
whole paths with the failed ones discarded (fixed seeds, the
Kolmogorov-Smirnov 0.1% band). The Wiener simulation discards a path that
crosses between grid points by the Brownian-bridge crossing probability,
so the reference is exact and not biased by the step.
"""

import json

import numpy as np
import pytest
from scipy import integrate, stats

from surpyval.degradation import GammaProcessModel, WienerProcessModel
from surpyval.tests._helpers import KS_BAND_20000, ks_distance

THRESHOLD = 100.0

# (model, t, y0, t0): near the threshold, so the conditioning matters.
SETTINGS = {
    "wiener": (WienerProcessModel(0.5, 2.0, THRESHOLD), 180.0, 10.0, 20.0),
    "wiener-negative-drift": (
        WienerProcessModel(-0.1, 3.0, THRESHOLD),
        150.0,
        85.0,
        50.0,
    ),
    "gamma": (GammaProcessModel(2.0, 4.0, THRESHOLD), 200.0, None, 0.0),
    "gamma-small-shape": (
        GammaProcessModel(0.05, 0.1, THRESHOLD),
        300.0,
        60.0,
        100.0,
    ),
}


def _level(key, **kw):
    model, t, y0, t0 = SETTINGS[key]
    return model.level_at(t, y0=y0, t0=t0, given_survival=True, **kw)


def _support(level):
    lo = level.y0 if level.family == "gamma" else -np.inf
    return lo, THRESHOLD


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_density_integrates_to_one_and_to_ff(key):
    level = _level(key)
    lo, hi = _support(level)
    total, _ = integrate.quad(
        lambda y: float(level.df(y)), lo, hi, limit=400, epsabs=1e-12
    )
    assert total == pytest.approx(1.0, abs=1e-8)
    for p in (0.1, 0.5, 0.9):
        y = float(level.qf(p))
        part, _ = integrate.quad(
            lambda v: float(level.df(v)), lo, y, limit=400, epsabs=1e-12
        )
        assert part == pytest.approx(float(level.ff(y)), abs=1e-8)


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_mass_is_below_the_threshold(key):
    level = _level(key)
    assert level.given_survival and level.threshold == THRESHOLD
    above = np.array([THRESHOLD, THRESHOLD + 1.0, np.inf])
    np.testing.assert_array_equal(level.ff(above), 1.0)
    np.testing.assert_array_equal(level.sf(above), 0.0)
    np.testing.assert_array_equal(level.df(above), 0.0)
    assert float(level.qf(1.0)) == THRESHOLD
    y = np.linspace(_support(level)[0] if level.family == "gamma" else 0, 99)
    np.testing.assert_allclose(level.sf(y) + level.ff(y), 1.0, atol=1e-14)
    assert np.all(np.diff(level.ff(y)) >= 0.0)


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_qf_inverts_ff(key):
    level = _level(key)
    p = np.array([1e-6, 0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999])
    np.testing.assert_allclose(level.ff(level.qf(p)), p, rtol=1e-9)
    assert np.isnan(level.qf(np.nan))
    lower = float(level.qf(0.0))
    assert lower == (level.y0 if level.family == "gamma" else -np.inf)


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_moments_against_quadrature(key):
    level = _level(key)
    lo, hi = _support(level)

    def moment(k):
        val, _ = integrate.quad(
            lambda y: y**k * float(level.df(y)), lo, hi, limit=400
        )
        return val

    mean = moment(1)
    sd = np.sqrt(moment(2) - mean**2)
    assert level.mean() == pytest.approx(mean, rel=1e-8)
    assert level.to_dict()["sd"] == pytest.approx(sd, rel=1e-6)


def test_wiener_against_the_image_formula():
    # The density written out from the method of images, independently.
    mu, sigma, L, y0, d = 0.5, 2.0, THRESHOLD, 10.0, 160.0
    level = _level("wiener")
    weight = np.exp(2.0 * mu * (L - y0) / sigma**2)
    spread = sigma * np.sqrt(d)
    y = np.linspace(40.0, 99.5, 25)
    raw = stats.norm.pdf(y, y0 + mu * d, spread) - weight * stats.norm.pdf(
        y, 2.0 * L - y0 + mu * d, spread
    )
    surv = WienerProcessModel(mu, sigma, L).sf(d, y0=y0)
    np.testing.assert_allclose(level.df(y), raw / surv, rtol=1e-10)


def test_gamma_is_the_truncated_increase():
    level = _level("gamma")
    inc = stats.gamma(2.0 * 200.0, scale=1.0 / 4.0)
    y = np.linspace(85.0, 99.9, 20)
    expected = inc.cdf(y) / inc.cdf(THRESHOLD)
    np.testing.assert_allclose(level.ff(y), expected, rtol=1e-12)
    np.testing.assert_allclose(
        level.df(y), inc.pdf(y) / inc.cdf(THRESHOLD), rtol=1e-12
    )


@pytest.mark.parametrize(
    "model",
    [WienerProcessModel(0.5, 2.0, 1e4), GammaProcessModel(2.0, 4.0, 1e4)],
    ids=["wiener", "gamma"],
)
def test_far_threshold_is_the_unconditioned_level(model):
    free = model.level_at(150.0, y0=5.0, t0=30.0)
    given = model.level_at(150.0, y0=5.0, t0=30.0, given_survival=True)
    y = np.linspace(free.qf(0.001), free.qf(0.999), 30)
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(given.ff(y), free.ff(y), rtol=1e-12)
    np.testing.assert_allclose(given.df(y), free.df(y), rtol=1e-12)
    np.testing.assert_allclose(given.qf(p), free.qf(p), rtol=1e-10)
    assert given.mean() == pytest.approx(free.mean(), rel=1e-12)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_stress_runs_the_clock(kind):
    # With stress, the level given survival is the unstressed one over the
    # process-clock time.
    g, z_ref, z = -5000.0, 1.0 / 323.0, 1.0 / 348.0
    af = float(np.exp(g * (z - z_ref)))
    if kind == "wiener":
        stressed = WienerProcessModel(
            0.5, 2.0, THRESHOLD, gamma=[g], stress_ref=[z_ref]
        )
        plain = WienerProcessModel(0.5, 2.0, THRESHOLD)
    else:
        stressed = GammaProcessModel(
            2.0, 4.0, THRESHOLD, gamma=[g], stress_ref=[z_ref]
        )
        plain = GammaProcessModel(2.0, 4.0, THRESHOLD)
    t0, t = 10.0, 10.0 + 60.0 / af
    a = stressed.level_at(t, [z], y0=20.0, t0=t0, given_survival=True)
    b = plain.level_at(70.0, y0=20.0, t0=10.0, given_survival=True)
    y = np.linspace(40.0, 99.0, 12)
    np.testing.assert_allclose(a.ff(y), b.ff(y), rtol=1e-9)


def _wiener_survivors(mu, sigma, L, y0, d, n_paths, steps, rng):
    """Final levels of the paths that never reach L, on a grid of
    ``steps`` with the Brownian-bridge crossing between grid points."""
    dt = d / steps
    level = np.full(n_paths, y0)
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(steps):
        nxt = level + rng.normal(mu * dt, sigma * np.sqrt(dt), n_paths)
        u = rng.uniform(size=n_paths)
        gap0, gap1 = L - level, L - nxt
        bridge = np.exp(
            -2.0 * np.maximum(gap0, 0) * np.maximum(gap1, 0) / (sigma**2 * dt)
        )
        alive &= (nxt < L) & (u > bridge)
        level = nxt
    return level[alive]


def test_wiener_against_simulated_paths():
    mu, sigma, y0, t0, t = 0.5, 2.0, 10.0, 20.0, 180.0
    rng = np.random.default_rng(20261120)
    survivors = _wiener_survivors(
        mu, sigma, THRESHOLD, y0, t - t0, 60_000, 40, rng
    )
    survivors = survivors[:20_000]
    assert survivors.size == 20_000
    level = _level("wiener")
    d = ks_distance(survivors, level.ff)
    assert d <= KS_BAND_20000, d


def test_gamma_against_simulated_paths():
    model, t, _, _ = SETTINGS["gamma"]
    rng = np.random.default_rng(20261121)
    steps = rng.gamma(2.0 * 10.0, 1.0 / 4.0, size=(60_000, 20))
    paths = np.cumsum(steps, axis=1)
    alive = paths[:, -1] < THRESHOLD  # monotone: the end is the maximum
    survivors = paths[alive, -1][:20_000]
    assert survivors.size == 20_000
    level = model.level_at(t, given_survival=True)
    d = ks_distance(survivors, level.ff)
    assert d <= KS_BAND_20000, d


@pytest.mark.parametrize("key", ["wiener", "gamma"])
def test_draws_are_reproducible_and_follow_ff(key):
    level = _level(key)
    a = level.random(20_000, random_state=7)
    b = level.random(20_000, random_state=7)
    np.testing.assert_array_equal(a, b)
    assert a.max() < THRESHOLD
    assert ks_distance(a, level.ff) <= KS_BAND_20000


def test_exposure_zero_is_the_reading():
    model = WienerProcessModel(0.5, 2.0, THRESHOLD)
    level = model.level_at(30.0, y0=40.0, t0=30.0, given_survival=True)
    assert float(level.qf(0.5)) == 40.0
    assert level.mean() == 40.0
    assert float(level.ff(39.0)) == 0.0 and float(level.ff(40.0)) == 1.0


def test_refusals():
    model = WienerProcessModel(0.5, 2.0, THRESHOLD)
    with pytest.raises(ValueError, match="given_survival=False"):
        model.level_at(50.0, y0=100.0, t0=10.0, given_survival=True)
    with pytest.raises(ValueError, match="given_survival must be True"):
        model.level_at(50.0, given_survival="yes")
    gamma = GammaProcessModel(2.0, 4.0, THRESHOLD)
    with pytest.raises(ValueError, match="earlier t"):
        gamma.level_at(2000.0, given_survival=True)
    # Without conditioning, a reading past the threshold is still taken.
    assert model.level_at(50.0, y0=120.0, t0=10.0).mean() == 140.0


def test_to_dict_and_repr():
    level = _level("wiener")
    d = json.loads(json.dumps(level.to_dict()))
    assert d["given_survival"] is True and d["threshold"] == THRESHOLD
    assert "given_survival=True" in repr(level)
    free = SETTINGS["wiener"][0].level_at(50.0).to_dict()
    assert free["given_survival"] is False and free["threshold"] is None
