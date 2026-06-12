import numpy as np
import pytest

import surpyval
from surpyval.univariate.nonparametric.nonparametric import NonParametric


def _censored_model():
    x = np.array([4.0, 7, 9, 13, 16, 21, 28, 33, 41, 50])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    return surpyval.KaplanMeier.fit(x, c=c)


def test_hall_wellner_critical_value_approaches_kolmogorov():
    # Over the whole [0, 1] interval the supremum of |Brownian bridge|
    # has the Kolmogorov distribution; its 0.95 quantile is ~1.358.
    cv = NonParametric._band_critical_value(
        1e-4,
        1 - 1e-4,
        0.05,
        standardized=False,
        n_sims=20000,
        random_state=1,
    )
    assert np.isclose(cv, 1.358, atol=0.05)


def test_band_contains_pointwise_bounds():
    model = _censored_model()
    pw = model.cb(model.x)
    for method in ["hall-wellner", "nair"]:
        band = model.band(method=method)
        finite = np.isfinite(band[:, 0]) & np.isfinite(pw[:, 0])
        assert np.all(band[finite, 0] <= pw[finite, 0] + 1e-9)
        assert np.all(band[finite, 1] >= pw[finite, 1] - 1e-9)


def test_band_reproducible_with_fixed_seed():
    model = _censored_model()
    assert np.allclose(model.band(), model.band(), equal_nan=True)


def test_band_normal_bound_type_shape():
    model = _censored_model()
    band = model.band(bound_type="normal")
    assert band.shape == (model.x.size, 2)


def test_band_invalid_args():
    model = _censored_model()
    with pytest.raises(ValueError):
        model.band(method="nope")
    with pytest.raises(ValueError):
        model.band(bound_type="regular")


def test_band_fit_from_ecdf_raises():
    model = NonParametric.fit_from_ecdf(
        np.array([1.0, 2, 3]), np.array([0.9, 0.5, 0.1])
    )
    with pytest.raises(ValueError, match="variance"):
        model.band()


def test_band_simultaneous_coverage():
    # The whole-curve coverage of the band should be close to nominal,
    # and markedly better than the pointwise bounds which are not
    # designed for simultaneous coverage.
    rng = np.random.default_rng(3)
    n, reps = 50, 150
    pw_cover = 0
    band_cover = 0
    for _ in range(reps):
        x = rng.exponential(1.0, n)
        cens = rng.exponential(2.0, n)
        obs = np.minimum(x, cens)
        c = (x > cens).astype(int)
        model = surpyval.KaplanMeier.fit(obs, c=c)
        grid = model.x
        S_true = np.exp(-grid)
        pw = model.cb(grid)
        band = model.band(n_sims=2000, random_state=1)
        f = np.isfinite(band[:, 0])
        pw_cover += np.all((pw[f, 0] <= S_true[f]) & (S_true[f] <= pw[f, 1]))
        band_cover += np.all(
            (band[f, 0] <= S_true[f]) & (S_true[f] <= band[f, 1])
        )
    assert band_cover / reps > 0.88
    assert band_cover / reps > pw_cover / reps


def test_smoothed_hazard_recovers_constant():
    rng = np.random.default_rng(4)
    x = rng.exponential(1.0, 2000)
    model = surpyval.NelsonAalen.fit(x)
    h = model.smoothed_hf([0.3, 0.6, 1.0, 1.5], bandwidth=0.5)
    assert np.allclose(h, 1.0, atol=0.15)


def test_smoothed_hazard_recovers_linear_weibull():
    # Weibull with shape 2 has hazard h(t) = 2t.
    x = surpyval.Weibull.random(4000, 1.0, 2.0)
    model = surpyval.NelsonAalen.fit(x)
    t = np.array([0.5, 1.0, 1.5])
    h = model.smoothed_hf(t, bandwidth=0.3)
    assert np.allclose(h, 2 * t, rtol=0.15)


def test_smoothed_hazard_nan_outside_range():
    model = surpyval.NelsonAalen.fit(np.array([1.0, 2, 3, 4, 5]))
    h = model.smoothed_hf([-1.0, 1e6])
    assert np.isnan(h).all()


def test_smoothed_hazard_bad_bandwidth():
    model = surpyval.NelsonAalen.fit(np.array([1.0, 2, 3]))
    with pytest.raises(ValueError):
        model.smoothed_hf([2.0], bandwidth=-1)
